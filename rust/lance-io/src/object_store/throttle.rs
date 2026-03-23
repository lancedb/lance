// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AIMD-controlled token bucket rate limiter for ObjectStore operations.
//!
//! Wraps any [`object_store::ObjectStore`] with a token bucket whose fill rate
//! is dynamically adjusted by an AIMD controller. When cloud stores return
//! HTTP 429/503, the fill rate decreases multiplicatively. During sustained
//! success windows, it increases additively.
//!
//! # Example
//!
//! ```ignore
//! use lance_io::object_store::throttle::{AimdThrottleConfig, AimdThrottleWrapper};
//!
//! let wrapper = AimdThrottleWrapper::new(AimdThrottleConfig::default());
//! // Use as ObjectStoreParams::object_store_wrapper
//! ```

use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use lance_core::utils::aimd::{AimdConfig, AimdController, RequestOutcome};
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OSResult,
};
use tokio::sync::Mutex;
use tracing::debug;

use super::WrappingObjectStore;

/// Check whether an `object_store::Error` represents a throttle response
/// (HTTP 429 / 503) from a cloud object store.
///
/// The `object_store` crate surfaces these as `Error::Generic` with the HTTP
/// status or cloud-specific message embedded in the source chain. We match
/// against known patterns from S3, GCS, and Azure.
pub fn is_throttle_error(err: &object_store::Error) -> bool {
    // Only Generic errors can carry throttle responses
    if let object_store::Error::Generic { source, .. } = err {
        let msg = source.to_string();
        // Check for common throttle patterns from cloud stores
        msg.contains("429")
            || msg.contains("Too Many Requests")
            || msg.contains("503")
            || msg.contains("Service Unavailable")
            || msg.contains("SlowDown")
            || msg.contains("Throttling")
            || msg.contains("RequestLimitExceeded")
    } else {
        false
    }
}

/// Configuration for the AIMD-throttled ObjectStore wrapper.
#[derive(Debug, Clone)]
pub struct AimdThrottleConfig {
    /// AIMD algorithm configuration.
    pub aimd: AimdConfig,
    /// Maximum tokens that can accumulate for bursts.
    pub burst_capacity: u32,
}

impl Default for AimdThrottleConfig {
    fn default() -> Self {
        Self {
            aimd: AimdConfig::default(),
            burst_capacity: 100,
        }
    }
}

impl AimdThrottleConfig {
    pub fn with_aimd(self, aimd: AimdConfig) -> Self {
        Self { aimd, ..self }
    }

    pub fn with_burst_capacity(self, burst_capacity: u32) -> Self {
        Self {
            burst_capacity,
            ..self
        }
    }
}

/// Factory that creates [`AimdThrottledStore`] wrappers.
///
/// Implements [`WrappingObjectStore`] so it can be passed to
/// `ObjectStoreParams::object_store_wrapper`.
#[derive(Debug, Clone)]
pub struct AimdThrottleWrapper {
    config: AimdThrottleConfig,
}

impl AimdThrottleWrapper {
    pub fn new(config: AimdThrottleConfig) -> Self {
        Self { config }
    }
}

impl WrappingObjectStore for AimdThrottleWrapper {
    fn wrap(&self, _store_prefix: &str, target: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        // unwrap is safe: config validation would have already been done, and
        // if not, default config always validates.
        Arc::new(AimdThrottledStore::new(target, self.config.clone()).expect("invalid AIMD config"))
    }
}

struct TokenBucketState {
    tokens: f64,
    last_refill: std::time::Instant,
    rate: f64,
}

/// An ObjectStore wrapper that rate-limits operations using a token bucket
/// whose fill rate is controlled by an AIMD algorithm.
pub struct AimdThrottledStore {
    target: Arc<dyn ObjectStore>,
    controller: AimdController,
    bucket: Mutex<TokenBucketState>,
    burst_capacity: f64,
}

impl Debug for AimdThrottledStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AimdThrottledStore")
            .field("target", &self.target)
            .field("controller", &self.controller)
            .field("burst_capacity", &self.burst_capacity)
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
        let initial_rate = config.aimd.initial_rate;
        let burst_capacity = config.burst_capacity as f64;
        let controller = AimdController::new(config.aimd)?;
        Ok(Self {
            target,
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
    async fn acquire_token(&self) {
        loop {
            let sleep_duration = {
                let mut bucket = self.bucket.lock().await;
                let now = std::time::Instant::now();
                let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
                bucket.tokens = (bucket.tokens + elapsed * bucket.rate).min(self.burst_capacity);
                bucket.last_refill = now;

                if bucket.tokens >= 1.0 {
                    bucket.tokens -= 1.0;
                    return;
                }

                // Calculate how long to wait for one token
                let deficit = 1.0 - bucket.tokens;
                std::time::Duration::from_secs_f64(deficit / bucket.rate)
            };

            tokio::time::sleep(sleep_duration).await;
        }
    }

    /// Update the bucket's fill rate from the controller.
    async fn update_bucket_rate(&self, new_rate: f64) {
        let mut bucket = self.bucket.lock().await;
        bucket.rate = new_rate;
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

#[async_trait]
#[deny(clippy::missing_trait_methods)]
impl ObjectStore for AimdThrottledStore {
    async fn put(&self, location: &Path, bytes: PutPayload) -> OSResult<PutResult> {
        self.throttled(|| self.target.put(location, bytes)).await
    }

    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.throttled(|| self.target.put_opts(location, bytes, opts))
            .await
    }

    async fn put_multipart(&self, location: &Path) -> OSResult<Box<dyn MultipartUpload>> {
        self.throttled(|| self.target.put_multipart(location)).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.throttled(|| self.target.put_multipart_opts(location, opts))
            .await
    }

    async fn get(&self, location: &Path) -> OSResult<GetResult> {
        self.throttled(|| self.target.get(location)).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.throttled(|| self.target.get_opts(location, options))
            .await
    }

    async fn get_range(&self, location: &Path, range: Range<u64>) -> OSResult<Bytes> {
        self.throttled(|| self.target.get_range(location, range.clone()))
            .await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.throttled(|| self.target.get_ranges(location, ranges))
            .await
    }

    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.throttled(|| self.target.head(location)).await
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.throttled(|| self.target.delete(location)).await
    }

    fn delete_stream<'a>(
        &'a self,
        locations: BoxStream<'a, OSResult<Path>>,
    ) -> BoxStream<'a, OSResult<Path>> {
        self.target.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.target.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.target.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.throttled(|| self.target.list_with_delimiter(prefix))
            .await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.throttled(|| self.target.copy(from, to)).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.throttled(|| self.target.rename(from, to)).await
    }

    async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.throttled(|| self.target.rename_if_not_exists(from, to))
            .await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.throttled(|| self.target.copy_if_not_exists(from, to))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use rstest::rstest;

    fn make_generic_error(msg: &str) -> object_store::Error {
        object_store::Error::Generic {
            store: "test",
            source: msg.into(),
        }
    }

    #[rstest]
    #[case::http_429("HTTP 429 Too Many Requests", true)]
    #[case::too_many_requests("Too Many Requests", true)]
    #[case::http_503("HTTP 503 Service Unavailable", true)]
    #[case::service_unavailable("Service Unavailable", true)]
    #[case::s3_slowdown("SlowDown: Please reduce your request rate", true)]
    #[case::throttling("Throttling: Rate exceeded", true)]
    #[case::request_limit("RequestLimitExceeded", true)]
    #[case::not_found("Object not found", false)]
    #[case::permission_denied("Access denied", false)]
    #[case::timeout("Connection timed out", false)]
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

        let initial_rate = throttled.controller.current_rate();
        assert_eq!(initial_rate, 100.0);

        // Simulate a throttle outcome directly
        throttled
            .controller
            .record_outcome(RequestOutcome::Throttled);

        // Wait for window to expire and trigger evaluation
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled.controller.record_outcome(RequestOutcome::Success);

        let new_rate = throttled.controller.current_rate();
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
            .controller
            .record_outcome(RequestOutcome::Throttled);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled.controller.record_outcome(RequestOutcome::Success);
        let decreased_rate = throttled.controller.current_rate();
        assert_eq!(decreased_rate, 50.0);

        // Now recover via success
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled.controller.record_outcome(RequestOutcome::Success);
        let recovered_rate = throttled.controller.current_rate();
        assert_eq!(recovered_rate, 60.0);
    }

    #[tokio::test]
    async fn test_wrapping_object_store_trait() {
        let wrapper = AimdThrottleWrapper::new(AimdThrottleConfig::default());
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let wrapped = wrapper.wrap("test://", store);

        let path = Path::from("test/data.bin");
        let data = PutPayload::from_static(b"test data");
        wrapped.put(&path, data).await.unwrap();

        let result = wrapped.get(&path).await.unwrap();
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
    async fn test_list_passthrough() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store.clone(), config).unwrap();

        let path = Path::from("prefix/file.txt");
        let data = PutPayload::from_static(b"data");
        store.put(&path, data).await.unwrap();

        use futures::StreamExt;
        let items: Vec<_> = throttled.list(Some(&Path::from("prefix"))).collect().await;
        assert_eq!(items.len(), 1);
        assert!(items[0].is_ok());
    }
}
