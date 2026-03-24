// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AIMD-controlled token bucket rate limiter for ObjectStore operations.
//!
//! Wraps any [`object_store::ObjectStore`] with per-category token buckets
//! whose fill rates are dynamically adjusted by AIMD controllers. When cloud
//! stores return HTTP 429/503, the fill rate decreases multiplicatively. During
//! sustained success windows, it increases additively.
//!
//! Operations are split into four independent categories — **read**, **write**,
//! **delete**, **list** — each with its own AIMD controller and token bucket.
//! This prevents a burst of reads from starving writes, and vice versa.
//!
//! # Example
//!
//! ```ignore
//! use lance_io::object_store::throttle::{AimdThrottleConfig, AimdThrottledStore};
//!
//! let throttled = AimdThrottledStore::new(target, AimdThrottleConfig::default()).unwrap();
//! ```

use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use lance_core::utils::aimd::{AimdConfig, AimdController, RequestOutcome};
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OSResult, UploadPart,
};
use tokio::sync::Mutex;
use tracing::debug;

/// Check whether an `object_store::Error` represents a throttle response
/// (HTTP 429 / 503) from a cloud object store.
///
/// Regrettably, this information is not fully exposed by the `object_store` crate.
/// There is no generic mechanism for a custom object store to return a throttle error.
///
/// However, the builtin object stores all use RetryError when retries are configured and
/// throttle errors are returned.  Sadly, RetryError is not a public type, so we have to
/// infer it from the error message.  This is potentially dangerous because these errors
/// often include the URI itself and that URI could have any characters in it (e.g. if we
/// look for 429 then we might match a 429 in a UUID).These error messages currently look like:
///
/// ", after ... retries, max_retries: ..., retry_timeout: ..."
///
/// So, as a crude heuristic, which should work for the builtin object stores, but won't
/// work for custom object stores, we simply look for the string "retries, max_retries"
/// in the error message.
pub fn is_throttle_error(err: &object_store::Error) -> bool {
    // Only Generic errors can carry throttle responses
    if let object_store::Error::Generic { source, .. } = err {
        source.to_string().contains("retries, max_retries")
    } else {
        false
    }
}

/// Configuration for the AIMD-throttled ObjectStore wrapper.
///
/// Each operation category (read, write, delete, list) has its own AIMD config.
/// Use [`with_aimd`](AimdThrottleConfig::with_aimd) to set all categories at
/// once, or per-category methods like [`with_read_aimd`](AimdThrottleConfig::with_read_aimd)
/// for fine-grained control.
#[derive(Debug, Clone)]
pub struct AimdThrottleConfig {
    /// AIMD configuration for read operations (get, get_opts, get_range, get_ranges, head).
    pub read: AimdConfig,
    /// AIMD configuration for write operations (put, put_opts, put_multipart, copy, rename, etc.).
    pub write: AimdConfig,
    /// AIMD configuration for delete operations.
    pub delete: AimdConfig,
    /// AIMD configuration for list operations (list_with_delimiter).
    pub list: AimdConfig,
    /// Maximum tokens that can accumulate for bursts (shared across all categories).
    pub burst_capacity: u32,
}

impl Default for AimdThrottleConfig {
    fn default() -> Self {
        let aimd = AimdConfig::default();
        Self {
            read: aimd.clone(),
            write: aimd.clone(),
            delete: aimd.clone(),
            list: aimd,
            burst_capacity: 100,
        }
    }
}

impl AimdThrottleConfig {
    /// Set the AIMD configuration for all four operation categories at once.
    pub fn with_aimd(self, aimd: AimdConfig) -> Self {
        Self {
            read: aimd.clone(),
            write: aimd.clone(),
            delete: aimd.clone(),
            list: aimd,
            ..self
        }
    }

    /// Set the AIMD configuration for read operations.
    pub fn with_read_aimd(self, aimd: AimdConfig) -> Self {
        Self { read: aimd, ..self }
    }

    /// Set the AIMD configuration for write operations.
    pub fn with_write_aimd(self, aimd: AimdConfig) -> Self {
        Self {
            write: aimd,
            ..self
        }
    }

    /// Set the AIMD configuration for delete operations.
    pub fn with_delete_aimd(self, aimd: AimdConfig) -> Self {
        Self {
            delete: aimd,
            ..self
        }
    }

    /// Set the AIMD configuration for list operations.
    pub fn with_list_aimd(self, aimd: AimdConfig) -> Self {
        Self { list: aimd, ..self }
    }

    pub fn with_burst_capacity(self, burst_capacity: u32) -> Self {
        Self {
            burst_capacity,
            ..self
        }
    }
}

struct TokenBucketState {
    tokens: f64,
    last_refill: std::time::Instant,
    rate: f64,
}

/// Per-category throttle state: an AIMD controller paired with a token bucket.
struct OperationThrottle {
    controller: AimdController,
    bucket: Mutex<TokenBucketState>,
    burst_capacity: f64,
}

impl OperationThrottle {
    fn new(aimd_config: AimdConfig, burst_capacity: f64) -> lance_core::Result<Self> {
        let initial_rate = aimd_config.initial_rate;
        let controller = AimdController::new(aimd_config)?;
        Ok(Self {
            controller,
            bucket: Mutex::new(TokenBucketState {
                tokens: burst_capacity,
                last_refill: std::time::Instant::now(),
                rate: initial_rate,
            }),
            burst_capacity,
        })
    }

    /// Acquire a token from the bucket, sleeping if none are available.
    ///
    /// Each caller reserves a token immediately (allowing `tokens` to go
    /// negative) so that concurrent waiters queue behind each other instead
    /// of all waking at the same instant (thundering herd).
    async fn acquire_token(&self) {
        let sleep_duration = {
            let mut bucket = self.bucket.lock().await;
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
            bucket.tokens = (bucket.tokens + elapsed * bucket.rate).min(self.burst_capacity);
            bucket.last_refill = now;

            // Reserve a token (may go negative to queue behind other waiters)
            bucket.tokens -= 1.0;

            if bucket.tokens >= 0.0 {
                // Had a token available, no need to sleep
                return;
            }

            // Sleep proportional to our position in the queue
            std::time::Duration::from_secs_f64(-bucket.tokens / bucket.rate)
        };

        tokio::time::sleep(sleep_duration).await;
    }

    /// Update the bucket's fill rate from the controller.
    async fn update_bucket_rate(&self, new_rate: f64) {
        let mut bucket = self.bucket.lock().await;
        bucket.rate = new_rate;
    }

    /// Classify a result and feed it back to the AIMD controller without
    /// acquiring a token. Uses `try_lock` for the bucket update so that if the
    /// bucket lock is contended the rate update is deferred to the next
    /// `throttled()` call.
    fn observe_outcome<T>(&self, result: &OSResult<T>) {
        let outcome = match result {
            Ok(_) => RequestOutcome::Success,
            Err(err) if is_throttle_error(err) => {
                debug!("Throttle error detected in stream, decreasing rate");
                RequestOutcome::Throttled
            }
            Err(_) => RequestOutcome::Success,
        };
        let new_rate = self.controller.record_outcome(outcome);
        if let Ok(mut bucket) = self.bucket.try_lock() {
            bucket.rate = new_rate;
        }
    }

    /// Execute an operation with throttling: acquire token, run, classify result.
    async fn throttled<T, F, Fut>(&self, f: F) -> OSResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = OSResult<T>>,
    {
        self.acquire_token().await;
        let result = f().await;
        let outcome = match &result {
            Ok(_) => RequestOutcome::Success,
            Err(err) if is_throttle_error(err) => {
                debug!("Throttle error detected, decreasing rate");
                RequestOutcome::Throttled
            }
            Err(_) => RequestOutcome::Success, // Non-throttle errors don't indicate capacity problems
        };
        let new_rate = self.controller.record_outcome(outcome);
        self.update_bucket_rate(new_rate).await;
        result
    }
}

impl Debug for OperationThrottle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperationThrottle")
            .field("controller", &self.controller)
            .field("burst_capacity", &self.burst_capacity)
            .finish()
    }
}

/// A [`MultipartUpload`] wrapper that throttles `put_part` and observes
/// outcomes from `put_part` and `complete`, feeding them back to the write
/// AIMD controller.
struct ThrottledMultipartUpload {
    target: Box<dyn MultipartUpload>,
    write: Arc<OperationThrottle>,
}

impl Debug for ThrottledMultipartUpload {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThrottledMultipartUpload").finish()
    }
}

#[async_trait]
impl MultipartUpload for ThrottledMultipartUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let write = Arc::clone(&self.write);
        let fut = self.target.put_part(data);
        Box::pin(async move {
            write.acquire_token().await;
            let result = fut.await;
            write.observe_outcome(&result);
            result
        })
    }

    async fn complete(&mut self) -> OSResult<PutResult> {
        let result = self.target.complete().await;
        self.write.observe_outcome(&result);
        result
    }

    async fn abort(&mut self) -> OSResult<()> {
        self.target.abort().await
    }
}

/// An ObjectStore wrapper that rate-limits operations using per-category token
/// buckets whose fill rates are controlled by AIMD algorithms.
///
/// Operations are split into four independent categories:
/// - **read**: `get`, `get_opts`, `get_range`, `get_ranges`, `head`
/// - **write**: `put`, `put_opts`, `put_multipart`, `put_multipart_opts`, `copy`, `copy_if_not_exists`, `rename`, `rename_if_not_exists`
/// - **delete**: `delete`
/// - **list**: `list_with_delimiter`
///
/// Streaming operations (`list`, `list_with_offset`, `delete_stream`) do not acquire tokens,
/// but observe each yielded item and feed the result back to the AIMD controller so it can
/// adjust the rate for other operations in the same category.
///
/// This is not perfect but probably as close as we can get without moving the throttle into
/// the object_store crate itself.
pub struct AimdThrottledStore {
    target: Arc<dyn ObjectStore>,
    read: Arc<OperationThrottle>,
    write: Arc<OperationThrottle>,
    delete: Arc<OperationThrottle>,
    list: Arc<OperationThrottle>,
}

impl Debug for AimdThrottledStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AimdThrottledStore")
            .field("target", &self.target)
            .field("read", &self.read)
            .field("write", &self.write)
            .field("delete", &self.delete)
            .field("list", &self.list)
            .finish()
    }
}

impl Display for AimdThrottledStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AimdThrottledStore({})", self.target)
    }
}

impl AimdThrottledStore {
    pub fn new(
        target: Arc<dyn ObjectStore>,
        config: AimdThrottleConfig,
    ) -> lance_core::Result<Self> {
        let burst = config.burst_capacity as f64;
        Ok(Self {
            target,
            read: Arc::new(OperationThrottle::new(config.read, burst)?),
            write: Arc::new(OperationThrottle::new(config.write, burst)?),
            delete: Arc::new(OperationThrottle::new(config.delete, burst)?),
            list: Arc::new(OperationThrottle::new(config.list, burst)?),
        })
    }
}

#[async_trait]
#[deny(clippy::missing_trait_methods)]
impl ObjectStore for AimdThrottledStore {
    async fn put(&self, location: &Path, bytes: PutPayload) -> OSResult<PutResult> {
        self.write
            .throttled(|| self.target.put(location, bytes))
            .await
    }

    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.write
            .throttled(|| self.target.put_opts(location, bytes, opts))
            .await
    }

    async fn put_multipart(&self, location: &Path) -> OSResult<Box<dyn MultipartUpload>> {
        let target = self
            .write
            .throttled(|| self.target.put_multipart(location))
            .await?;
        Ok(Box::new(ThrottledMultipartUpload {
            target,
            write: Arc::clone(&self.write),
        }))
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        let target = self
            .write
            .throttled(|| self.target.put_multipart_opts(location, opts))
            .await?;
        Ok(Box::new(ThrottledMultipartUpload {
            target,
            write: Arc::clone(&self.write),
        }))
    }

    async fn get(&self, location: &Path) -> OSResult<GetResult> {
        self.read.throttled(|| self.target.get(location)).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.read
            .throttled(|| self.target.get_opts(location, options))
            .await
    }

    async fn get_range(&self, location: &Path, range: Range<u64>) -> OSResult<Bytes> {
        self.read
            .throttled(|| self.target.get_range(location, range.clone()))
            .await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.read
            .throttled(|| self.target.get_ranges(location, ranges))
            .await
    }

    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.read.throttled(|| self.target.head(location)).await
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.delete.throttled(|| self.target.delete(location)).await
    }

    fn delete_stream<'a>(
        &'a self,
        locations: BoxStream<'a, OSResult<Path>>,
    ) -> BoxStream<'a, OSResult<Path>> {
        self.target
            .delete_stream(locations)
            .map(|item| {
                self.delete.observe_outcome(&item);
                item
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let throttle = Arc::clone(&self.list);
        self.target
            .list(prefix)
            .map(move |item| {
                throttle.observe_outcome(&item);
                item
            })
            .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let throttle = Arc::clone(&self.list);
        self.target
            .list_with_offset(prefix, offset)
            .map(move |item| {
                throttle.observe_outcome(&item);
                item
            })
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.list
            .throttled(|| self.target.list_with_delimiter(prefix))
            .await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.write.throttled(|| self.target.copy(from, to)).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.write.throttled(|| self.target.rename(from, to)).await
    }

    async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.write
            .throttled(|| self.target.rename_if_not_exists(from, to))
            .await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.write
            .throttled(|| self.target.copy_if_not_exists(from, to))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn make_generic_error(msg: &str) -> object_store::Error {
        object_store::Error::Generic {
            store: "test",
            source: msg.into(),
        }
    }

    #[rstest]
    #[case::retry_error("Error after 10 retries, max_retries: 10, retry_timeout: 180s", true)]
    #[case::retries_in_message(
        "request failed, after 3 retries, max_retries: 5, retry_timeout: 60s",
        true
    )]
    #[case::not_found("Object not found", false)]
    #[case::permission_denied("Access denied", false)]
    #[case::timeout("Connection timed out", false)]
    #[case::http_429_without_retries("HTTP 429 Too Many Requests", false)]
    #[case::slowdown_without_retries("SlowDown: Please reduce your request rate", false)]
    fn test_is_throttle_error(#[case] msg: &str, #[case] expected: bool) {
        let err = make_generic_error(msg);
        assert_eq!(
            is_throttle_error(&err),
            expected,
            "is_throttle_error for '{}' should be {}",
            msg,
            expected
        );
    }

    #[test]
    fn test_non_generic_errors_are_not_throttle() {
        let err = object_store::Error::NotFound {
            path: "test".to_string(),
            source: "not found".into(),
        };
        assert!(!is_throttle_error(&err));
    }

    #[tokio::test]
    async fn test_basic_put_get_through_wrapper() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        let path = Path::from("test/file.txt");
        let data = PutPayload::from_static(b"hello world");
        throttled.put(&path, data).await.unwrap();

        let result = throttled.get(&path).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"hello world");
    }

    #[tokio::test]
    async fn test_rate_decreases_on_throttle() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default().with_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        let initial_rate = throttled.read.controller.current_rate();
        assert_eq!(initial_rate, 100.0);

        // Simulate a throttle outcome directly
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Throttled);

        // Wait for window to expire and trigger evaluation
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);

        let new_rate = throttled.read.controller.current_rate();
        assert!(
            new_rate < initial_rate,
            "Rate should decrease after throttle: {} < {}",
            new_rate,
            initial_rate
        );
    }

    #[tokio::test]
    async fn test_rate_recovers_on_success() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default().with_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_additive_increment(10.0)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        // First decrease via throttle
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Throttled);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);
        let decreased_rate = throttled.read.controller.current_rate();
        assert_eq!(decreased_rate, 50.0);

        // Now recover via success
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);
        let recovered_rate = throttled.read.controller.current_rate();
        assert_eq!(recovered_rate, 60.0);
    }

    #[tokio::test]
    async fn test_as_dyn_object_store() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let throttled: Arc<dyn ObjectStore> =
            Arc::new(AimdThrottledStore::new(store, AimdThrottleConfig::default()).unwrap());

        let path = Path::from("test/data.bin");
        let data = PutPayload::from_static(b"test data");
        throttled.put(&path, data).await.unwrap();

        let result = throttled.get(&path).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"test data");
    }

    #[tokio::test]
    async fn test_token_bucket_delays_when_exhausted() {
        let store = Arc::new(InMemory::new());
        // Very low rate and burst capacity to force waiting
        let config = AimdThrottleConfig::default()
            .with_burst_capacity(1)
            .with_aimd(AimdConfig::default().with_initial_rate(10.0));
        let throttled = Arc::new(AimdThrottledStore::new(store, config).unwrap());

        let path = Path::from("test/file.txt");
        let data = PutPayload::from_static(b"data");
        throttled.put(&path, data).await.unwrap();

        // After consuming the burst token, the next request should take ~100ms
        // (1 token / 10 tokens-per-sec). We verify it takes at least 50ms.
        let start = std::time::Instant::now();
        let data2 = PutPayload::from_static(b"data2");
        throttled.put(&path, data2).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed >= std::time::Duration::from_millis(50),
            "Expected delay for token refill, but elapsed was {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_list_observes_outcomes() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store.clone(), config).unwrap();

        let path = Path::from("prefix/file.txt");
        let data = PutPayload::from_static(b"data");
        store.put(&path, data).await.unwrap();

        let items: Vec<_> = throttled.list(Some(&Path::from("prefix"))).collect().await;
        assert_eq!(items.len(), 1);
        assert!(items[0].is_ok());
    }

    /// A mock store whose `list` stream yields a configurable sequence of
    /// Ok / throttle-error items. Used to verify that the AIMD wrapper
    /// observes errors surfaced inside list streams.
    struct ThrottlingListMockStore {
        inner: InMemory,
        /// Number of throttle errors to inject at the start of each list call.
        throttle_count: usize,
    }

    impl Display for ThrottlingListMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "ThrottlingListMockStore")
        }
    }

    impl Debug for ThrottlingListMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ThrottlingListMockStore").finish()
        }
    }

    #[async_trait]
    impl ObjectStore for ThrottlingListMockStore {
        async fn put(&self, location: &Path, bytes: PutPayload) -> OSResult<PutResult> {
            self.inner.put(location, bytes).await
        }
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }
        async fn put_multipart(&self, location: &Path) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart(location).await
        }
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }
        async fn get(&self, location: &Path) -> OSResult<GetResult> {
            self.inner.get(location).await
        }
        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            self.inner.get_opts(location, options).await
        }
        async fn get_range(&self, location: &Path, range: Range<u64>) -> OSResult<Bytes> {
            self.inner.get_range(location, range).await
        }
        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }
        async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
            self.inner.head(location).await
        }
        async fn delete(&self, location: &Path) -> OSResult<()> {
            self.inner.delete(location).await
        }
        fn delete_stream<'a>(
            &'a self,
            locations: BoxStream<'a, OSResult<Path>>,
        ) -> BoxStream<'a, OSResult<Path>> {
            self.inner.delete_stream(locations)
        }
        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            let n = self.throttle_count;
            let inner_stream = self.inner.list(prefix);
            let errors = futures::stream::iter((0..n).map(|_| {
                Err(object_store::Error::Generic {
                    store: "ThrottlingListMock",
                    source: "request failed, after 3 retries, max_retries: 5, retry_timeout: 60s"
                        .into(),
                })
            }));
            errors.chain(inner_stream).boxed()
        }
        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }
        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.copy(from, to).await
        }
        async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.rename(from, to).await
        }
        async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.rename_if_not_exists(from, to).await
        }
        async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.copy_if_not_exists(from, to).await
        }
    }

    #[tokio::test]
    async fn test_list_stream_throttle_errors_decrease_rate() {
        let mock = Arc::new(ThrottlingListMockStore {
            inner: InMemory::new(),
            throttle_count: 5,
        });

        // Seed a file so the real items come through after the errors.
        mock.put(
            &Path::from("prefix/file.txt"),
            PutPayload::from_static(b"data"),
        )
        .await
        .unwrap();

        let config = AimdThrottleConfig::default().with_list_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(mock as Arc<dyn ObjectStore>, config).unwrap();

        let initial_rate = throttled.list.controller.current_rate();
        assert_eq!(initial_rate, 100.0);

        let items: Vec<_> = throttled.list(Some(&Path::from("prefix"))).collect().await;

        // 5 errors + 1 real item
        assert_eq!(items.len(), 6);
        assert!(items[0].is_err());
        assert!(items[5].is_ok());

        // Wait for the AIMD window to expire and trigger evaluation.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .list
            .controller
            .record_outcome(RequestOutcome::Success);

        let new_rate = throttled.list.controller.current_rate();
        assert!(
            new_rate < initial_rate,
            "List rate should decrease after stream throttle errors: {} < {}",
            new_rate,
            initial_rate
        );
    }

    #[tokio::test]
    async fn test_per_category_independence() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default().with_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        // Push the read controller into a throttled state
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Throttled);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);

        let read_rate = throttled.read.controller.current_rate();
        let write_rate = throttled.write.controller.current_rate();
        let delete_rate = throttled.delete.controller.current_rate();
        let list_rate = throttled.list.controller.current_rate();

        assert_eq!(read_rate, 50.0, "Read rate should have decreased");
        assert_eq!(write_rate, 100.0, "Write rate should be unaffected");
        assert_eq!(delete_rate, 100.0, "Delete rate should be unaffected");
        assert_eq!(list_rate, 100.0, "List rate should be unaffected");
    }

    #[tokio::test]
    async fn test_per_category_config() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default()
            .with_read_aimd(AimdConfig::default().with_initial_rate(200.0))
            .with_write_aimd(AimdConfig::default().with_initial_rate(100.0))
            .with_delete_aimd(AimdConfig::default().with_initial_rate(50.0))
            .with_list_aimd(AimdConfig::default().with_initial_rate(25.0));
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        assert_eq!(throttled.read.controller.current_rate(), 200.0);
        assert_eq!(throttled.write.controller.current_rate(), 100.0);
        assert_eq!(throttled.delete.controller.current_rate(), 50.0);
        assert_eq!(throttled.list.controller.current_rate(), 25.0);
    }

    /// A mock [`ObjectStore`] that measures request rate over a sliding window
    /// and returns 503 errors when the rate exceeds a configurable threshold.
    /// Write and metadata-only operations are not rate-limited.
    struct RateLimitingMockStore {
        inner: InMemory,
        /// Timestamps of recent successful (admitted) requests.
        timestamps: std::sync::Mutex<VecDeque<std::time::Instant>>,
        /// Maximum requests allowed within `window`.
        max_per_window: usize,
        /// Sliding window duration.
        window: std::time::Duration,
        success_count: AtomicU64,
        throttle_count: AtomicU64,
    }

    impl RateLimitingMockStore {
        fn new(max_per_window: usize, window: std::time::Duration) -> Self {
            Self {
                inner: InMemory::new(),
                timestamps: std::sync::Mutex::new(VecDeque::new()),
                max_per_window,
                window,
                success_count: AtomicU64::new(0),
                throttle_count: AtomicU64::new(0),
            }
        }

        /// Returns `true` if the request is admitted, `false` if throttled.
        fn check_rate(&self) -> bool {
            let mut ts = self.timestamps.lock().unwrap();
            let now = std::time::Instant::now();
            while let Some(&front) = ts.front() {
                if now.duration_since(front) > self.window {
                    ts.pop_front();
                } else {
                    break;
                }
            }
            if ts.len() >= self.max_per_window {
                self.throttle_count.fetch_add(1, Ordering::Relaxed);
                false
            } else {
                ts.push_back(now);
                self.success_count.fetch_add(1, Ordering::Relaxed);
                true
            }
        }

        fn throttle_error() -> object_store::Error {
            object_store::Error::Generic {
                store: "RateLimitingMock",
                source: "request failed, after 10 retries, max_retries: 10, retry_timeout: 180s"
                    .into(),
            }
        }
    }

    impl Display for RateLimitingMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "RateLimitingMockStore")
        }
    }

    impl Debug for RateLimitingMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RateLimitingMockStore").finish()
        }
    }

    #[async_trait]
    impl ObjectStore for RateLimitingMockStore {
        async fn put(&self, location: &Path, bytes: PutPayload) -> OSResult<PutResult> {
            self.inner.put(location, bytes).await
        }

        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }

        async fn put_multipart(&self, location: &Path) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart(location).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get(&self, location: &Path) -> OSResult<GetResult> {
            if self.check_rate() {
                self.inner.get(location).await
            } else {
                Err(Self::throttle_error())
            }
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            if self.check_rate() {
                self.inner.get_opts(location, options).await
            } else {
                Err(Self::throttle_error())
            }
        }

        async fn get_range(&self, location: &Path, range: Range<u64>) -> OSResult<Bytes> {
            if self.check_rate() {
                self.inner.get_range(location, range).await
            } else {
                Err(Self::throttle_error())
            }
        }

        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            if self.check_rate() {
                self.inner.get_ranges(location, ranges).await
            } else {
                Err(Self::throttle_error())
            }
        }

        async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
            if self.check_rate() {
                self.inner.head(location).await
            } else {
                Err(Self::throttle_error())
            }
        }

        async fn delete(&self, location: &Path) -> OSResult<()> {
            self.inner.delete(location).await
        }

        fn delete_stream<'a>(
            &'a self,
            locations: BoxStream<'a, OSResult<Path>>,
        ) -> BoxStream<'a, OSResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.copy(from, to).await
        }

        async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.rename(from, to).await
        }

        async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.rename_if_not_exists(from, to).await
        }

        async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
            self.inner.copy_if_not_exists(from, to).await
        }
    }

    /// Verify that multiple concurrent readers sharing an AIMD-throttled store
    /// converge to the backend's actual capacity.
    ///
    /// Setup:
    /// - Mock backend allows 30 requests per 100ms (= 300 req/s).
    /// - 5 reader tasks, each with their own [`AimdThrottledStore`] wrapping
    ///   the shared mock.
    /// - AIMD: 100ms window, initial rate 100 req/s, decrease 0.5, increase 2.
    /// - Readers issue `head()` requests as fast as the throttle allows for 2s.
    ///
    /// Expected behaviour:
    /// - Initial burst (100 burst tokens × 5 readers) overshoots the mock
    ///   capacity, causing many 503s. Each reader's AIMD halves its rate.
    /// - After the transient, each reader converges to ~60 req/s (300/5).
    /// - Over 2 seconds, total successful requests should be in the range
    ///   [300, 900] (theoretical max ≈ 600).
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_aimd_throttle_under_concurrent_load() {
        let mock = Arc::new(RateLimitingMockStore::new(
            30,
            std::time::Duration::from_millis(100),
        ));

        // Seed a test file so head() succeeds when admitted.
        let path = Path::from("test/data.bin");
        mock.put(&path, PutPayload::from_static(b"test data"))
            .await
            .unwrap();

        let aimd = AimdConfig::default()
            .with_initial_rate(100.0)
            .with_decrease_factor(0.5)
            .with_additive_increment(2.0)
            .with_window_duration(std::time::Duration::from_millis(100));
        let throttle_config = AimdThrottleConfig::default()
            .with_aimd(aimd)
            .with_burst_capacity(100);

        let num_readers = 5;
        let test_duration = std::time::Duration::from_secs(2);
        let mut handles = Vec::new();

        for _ in 0..num_readers {
            let store = Arc::new(
                AimdThrottledStore::new(
                    mock.clone() as Arc<dyn ObjectStore>,
                    throttle_config.clone(),
                )
                .unwrap(),
            );
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let deadline = std::time::Instant::now() + test_duration;
                let mut count = 0u64;
                while std::time::Instant::now() < deadline {
                    let _ = store.head(&p).await;
                    count += 1;
                }
                count
            }));
        }

        let mut total_reader_requests = 0u64;
        for handle in handles {
            total_reader_requests += handle.await.unwrap();
        }

        let successes = mock.success_count.load(Ordering::Relaxed);
        let throttled = mock.throttle_count.load(Ordering::Relaxed);
        let total_mock = successes + throttled;

        // Reader-side count must match mock-side count.
        assert_eq!(
            total_reader_requests, total_mock,
            "Reader-side count ({total_reader_requests}) != mock-side count ({total_mock})"
        );

        // Mock capacity is 30/100ms = 300 req/s. Over 2s the theoretical max is
        // ~600 successful requests. With AIMD ramp-up, expect somewhat fewer.
        assert!(
            successes >= 300,
            "Expected >= 300 successful requests over 2s, got {successes}"
        );
        assert!(
            successes <= 900,
            "Expected <= 900 successful requests, got {successes}"
        );

        // The initial burst exceeds mock capacity, so throttling must occur.
        assert!(throttled > 0, "Expected some throttled requests but got 0");

        // Without AIMD, raw tokio tasks against InMemory would fire 100k+ req/s.
        // AIMD should keep the total well under 5000 over 2s.
        assert!(
            total_mock <= 5000,
            "AIMD should limit total requests, got {total_mock}"
        );
    }
}
