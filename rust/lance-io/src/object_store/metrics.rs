// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Publishes object store metrics via the [`metrics`] crate.
//!
//! The [`metrics`] facade lets downstream applications install any recorder
//! (Prometheus, OpenTelemetry, etc.) without Lance depending on a specific
//! backend. When no recorder is installed the calls are cheap no-ops.
//!
//! Two layers cooperate:
//!
//! * [`MeteredObjectStore`] wraps any [`object_store::ObjectStore`] and records
//!   per-operation request counts, transferred bytes, latency, errors, and the
//!   number of requests currently in flight. It works for every store
//!   regardless of backend.
//! * [`MeteringHttpConnector`] wraps the HTTP client used by the native cloud
//!   stores (S3 / GCS / Azure) and records throttle responses (HTTP 429 / 503)
//!   per attempt. Because `object_store`'s retry loop re-issues each request
//!   through the [`HttpService`](object_store::client::HttpService), this sees
//!   every retried throttle, which a store-level wrapper cannot observe.
//!
//! The two layers have different coverage: every store gets the request-level
//! metrics from [`MeteredObjectStore`], but only the native cloud stores get
//! the HTTP-level throttle metrics. Opendal-backed stores (tos, oss, etc.)
//! bypass `object_store`'s HTTP client, so there is no place to install the
//! connector for them.
//!
//! All metrics carry a `scheme` label (e.g. `s3`, `gs`, `azure`).

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as OSResult,
    UploadPart,
};

/// Total number of object store requests, labelled by `operation` and `scheme`.
const METRIC_REQUESTS: &str = "lance_object_store_requests_total";
/// Total bytes transferred by object store requests, labelled by `operation` and `scheme`.
const METRIC_BYTES: &str = "lance_object_store_request_bytes_total";
/// Object store request latency in seconds, labelled by `operation` and `scheme`.
const METRIC_DURATION: &str = "lance_object_store_request_duration_seconds";
/// Total number of failed object store requests, labelled by `operation` and `scheme`.
const METRIC_ERRORS: &str = "lance_object_store_errors_total";
/// Total number of throttle responses (HTTP 429 / 503) seen at the HTTP layer,
/// labelled by `status` and `scheme`. Counts every attempt, including retries.
const METRIC_THROTTLE: &str = "lance_object_store_throttle_total";
/// Number of object store requests currently in flight, labelled by `operation`
/// and `scheme`.
const METRIC_IN_FLIGHT: &str = "lance_object_store_in_flight_requests";

/// Record the outcome of a unary request: count, latency, bytes (on success), and errors.
fn record_request<T>(
    scheme: &str,
    operation: &'static str,
    start: Instant,
    bytes: u64,
    result: &OSResult<T>,
) {
    let elapsed = start.elapsed().as_secs_f64();
    metrics::counter!(METRIC_REQUESTS, "operation" => operation, "scheme" => scheme.to_owned())
        .increment(1);
    metrics::histogram!(METRIC_DURATION, "operation" => operation, "scheme" => scheme.to_owned())
        .record(elapsed);
    match result {
        Ok(_) => {
            if bytes > 0 {
                metrics::counter!(METRIC_BYTES, "operation" => operation, "scheme" => scheme.to_owned())
                    .increment(bytes);
            }
        }
        Err(_) => {
            metrics::counter!(METRIC_ERRORS, "operation" => operation, "scheme" => scheme.to_owned())
                .increment(1);
        }
    }
}

/// Record a single request count without latency, used for streaming operations
/// (list / delete) whose work happens lazily as the stream is polled.
fn record_count(scheme: &str, operation: &'static str) {
    metrics::counter!(METRIC_REQUESTS, "operation" => operation, "scheme" => scheme.to_owned())
        .increment(1);
}

fn record_error(scheme: &str, operation: &'static str) {
    metrics::counter!(METRIC_ERRORS, "operation" => operation, "scheme" => scheme.to_owned())
        .increment(1);
}

/// Raises the in-flight gauge for an operation on creation and lowers it on
/// drop, so the count stays balanced even if the request future or stream is
/// cancelled or dropped before completing.
struct InFlightGuard {
    scheme: String,
    operation: &'static str,
}

impl InFlightGuard {
    fn new(scheme: &str, operation: &'static str) -> Self {
        metrics::gauge!(METRIC_IN_FLIGHT, "operation" => operation, "scheme" => scheme.to_owned())
            .increment(1.0);
        Self {
            scheme: scheme.to_owned(),
            operation,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        metrics::gauge!(METRIC_IN_FLIGHT, "operation" => self.operation, "scheme" => self.scheme.clone())
            .decrement(1.0);
    }
}

#[derive(Debug)]
pub struct MeteredObjectStore {
    target: Arc<dyn object_store::ObjectStore>,
    scheme: String,
}

impl std::fmt::Display for MeteredObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MeteredObjectStore({})", self.target)
    }
}

#[async_trait::async_trait]
#[deny(clippy::missing_trait_methods)]
impl object_store::ObjectStore for MeteredObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        let size = bytes.content_length() as u64;
        let _in_flight = InFlightGuard::new(&self.scheme, "put");
        let start = Instant::now();
        let result = self.target.put_opts(location, bytes, opts).await;
        record_request(&self.scheme, "put", start, size, &result);
        result
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        let upload = self.target.put_multipart_opts(location, opts).await?;
        Ok(Box::new(MeteredMultipartUpload {
            target: upload,
            scheme: self.scheme.clone(),
        }))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        // `head()` is implemented as a `get_opts` call with `head = true`, so we
        // distinguish it here to keep HEAD and GET as separate operations.
        let is_head = options.head;
        let operation = if is_head { "head" } else { "get" };
        let _in_flight = InFlightGuard::new(&self.scheme, operation);
        let start = Instant::now();
        let result = self.target.get_opts(location, options).await;
        // A HEAD transfers only metadata, so it has no payload bytes.
        let bytes = match &result {
            Ok(res) if !is_head => res.range.end - res.range.start,
            _ => 0,
        };
        record_request(&self.scheme, operation, start, bytes, &result);
        result
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        let _in_flight = InFlightGuard::new(&self.scheme, "get");
        let start = Instant::now();
        let result = self.target.get_ranges(location, ranges).await;
        let bytes = match &result {
            Ok(parts) => parts.iter().map(|b| b.len() as u64).sum(),
            Err(_) => 0,
        };
        record_request(&self.scheme, "get", start, bytes, &result);
        result
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        let scheme = self.scheme.clone();
        let in_flight = InFlightGuard::new(&self.scheme, "delete");
        self.target
            .delete_stream(locations)
            .map(move |result| {
                // Reference `in_flight` so this `move` closure captures (owns)
                // the guard, keeping the gauge raised until the stream is
                // dropped (a move closure only captures the variables it uses).
                let _in_flight = &in_flight;
                // Each yielded path is one delete; failures additionally bump the error counter.
                record_count(&scheme, "delete");
                if result.is_err() {
                    record_error(&scheme, "delete");
                }
                result
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        record_count(&self.scheme, "list");
        meter_list_stream(
            self.target.list(prefix),
            self.scheme.clone(),
            InFlightGuard::new(&self.scheme, "list"),
        )
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        record_count(&self.scheme, "list");
        meter_list_stream(
            self.target.list_with_offset(prefix, offset),
            self.scheme.clone(),
            InFlightGuard::new(&self.scheme, "list"),
        )
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        let _in_flight = InFlightGuard::new(&self.scheme, "list");
        let start = Instant::now();
        let result = self.target.list_with_delimiter(prefix).await;
        record_request(&self.scheme, "list", start, 0, &result);
        result
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        let _in_flight = InFlightGuard::new(&self.scheme, "copy");
        let start = Instant::now();
        let result = self.target.copy_opts(from, to, opts).await;
        record_request(&self.scheme, "copy", start, 0, &result);
        result
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        let _in_flight = InFlightGuard::new(&self.scheme, "rename");
        let start = Instant::now();
        let result = self.target.rename_opts(from, to, opts).await;
        record_request(&self.scheme, "rename", start, 0, &result);
        result
    }
}

/// Count errors yielded while draining a list stream. The request itself is
/// counted once when the stream is created (a single LIST may return many items).
fn meter_list_stream(
    stream: BoxStream<'static, OSResult<ObjectMeta>>,
    scheme: String,
    in_flight: InFlightGuard,
) -> BoxStream<'static, OSResult<ObjectMeta>> {
    stream
        .map(move |result| {
            // Reference `in_flight` so this `move` closure captures (owns) the
            // guard: a move closure only captures the variables it uses, and
            // holding it here keeps the gauge raised until the stream is dropped.
            let _in_flight = &in_flight;
            if result.is_err() {
                record_error(&scheme, "list");
            }
            result
        })
        .boxed()
}

#[derive(Debug)]
struct MeteredMultipartUpload {
    target: Box<dyn MultipartUpload>,
    scheme: String,
}

#[async_trait::async_trait]
impl MultipartUpload for MeteredMultipartUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        // Each part upload is a distinct `put` request, so it records the same
        // count / bytes / latency / error set as a unary put.
        let scheme = self.scheme.clone();
        let size = data.content_length() as u64;
        let inner = self.target.put_part(data);
        async move {
            let _in_flight = InFlightGuard::new(&scheme, "put");
            let start = Instant::now();
            let result = inner.await;
            record_request(&scheme, "put", start, size, &result);
            result
        }
        .boxed()
    }

    async fn complete(&mut self) -> OSResult<PutResult> {
        self.target.complete().await
    }

    async fn abort(&mut self) -> OSResult<()> {
        self.target.abort().await
    }
}

pub trait ObjectStoreMetricsExt {
    /// Wrap this store so its operations publish metrics under the given `scheme` label.
    fn metered(self, scheme: String) -> Arc<dyn object_store::ObjectStore>;
}

impl ObjectStoreMetricsExt for Arc<dyn object_store::ObjectStore> {
    fn metered(self, scheme: String) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(MeteredObjectStore {
            target: self,
            scheme,
        })
    }
}

// --- Layer 2: HTTP-level throttle metrics for native cloud stores ---

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
mod http {
    use super::*;
    use object_store::client::{
        ClientOptions, HttpClient, HttpConnector, HttpError, HttpRequest, HttpResponse,
        HttpService, ReqwestConnector,
    };

    /// An [`HttpConnector`] that records throttle responses observed by the
    /// underlying HTTP client. Install it on the S3 / GCS / Azure builders via
    /// `with_http_connector`.
    #[derive(Debug)]
    pub struct MeteringHttpConnector {
        scheme: String,
        inner: ReqwestConnector,
    }

    impl MeteringHttpConnector {
        pub fn new(scheme: String) -> Self {
            Self {
                scheme,
                inner: ReqwestConnector::default(),
            }
        }
    }

    impl HttpConnector for MeteringHttpConnector {
        fn connect(&self, options: &ClientOptions) -> object_store::Result<HttpClient> {
            let client = self.inner.connect(options)?;
            Ok(HttpClient::new(MeteringHttpService {
                scheme: self.scheme.clone(),
                inner: client,
            }))
        }
    }

    #[derive(Debug)]
    struct MeteringHttpService {
        scheme: String,
        inner: HttpClient,
    }

    #[async_trait::async_trait]
    impl HttpService for MeteringHttpService {
        async fn call(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
            let response = self.inner.execute(req).await?;
            let status = response.status();
            // 429 (throttling) and 5xx (e.g. 503 service unavailable) are the
            // statuses that drive object_store's retry loop. Each attempt that
            // hits one is recorded with its numeric status, so 429 and 503 can
            // be told apart.
            if status.as_u16() == 429 || status.is_server_error() {
                metrics::counter!(
                    METRIC_THROTTLE,
                    "status" => status.as_u16().to_string(),
                    "scheme" => self.scheme.clone(),
                )
                .increment(1);
            }
            Ok(response)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        use object_store::client::{HttpRequestBody, HttpResponseBody};

        /// A mock [`HttpService`] that always responds with a fixed status code.
        #[derive(Debug)]
        struct StaticStatusService {
            status: u16,
        }

        #[async_trait::async_trait]
        impl HttpService for StaticStatusService {
            async fn call(&self, _req: HttpRequest) -> Result<HttpResponse, HttpError> {
                Ok(::http::Response::builder()
                    .status(self.status)
                    .body(HttpResponseBody::from(Bytes::new()))
                    .unwrap())
            }
        }

        fn request() -> HttpRequest {
            ::http::Request::builder()
                .method("GET")
                .uri("http://example.com/obj")
                .body(HttpRequestBody::empty())
                .unwrap()
        }

        fn throttle_count(
            metrics: &[(metrics::Key, DebugValue)],
            scheme: &str,
            status: &str,
        ) -> u64 {
            for (key, value) in metrics {
                if key.name() != METRIC_THROTTLE {
                    continue;
                }
                let labels: std::collections::HashMap<&str, &str> =
                    key.labels().map(|l| (l.key(), l.value())).collect();
                if labels.get("scheme") == Some(&scheme)
                    && labels.get("status") == Some(&status)
                    && let DebugValue::Counter(v) = value
                {
                    return *v;
                }
            }
            0
        }

        #[test]
        fn test_throttle_responses_counted_by_status() {
            let recorder = DebuggingRecorder::new();
            let snapshotter = recorder.snapshotter();
            metrics::with_local_recorder(&recorder, || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    // Each attempt that object_store retries flows through `call`
                    // again; here we simulate that by issuing several responses.
                    // The scheme is baked into the connector, so it labels the metric.
                    for (scheme, status) in [
                        ("s3", 429u16),
                        ("s3", 503),
                        ("s3", 503),
                        ("s3", 500),
                        ("s3", 200),
                        ("s3", 404),
                        ("gs", 429),
                    ] {
                        let service = MeteringHttpService {
                            scheme: scheme.into(),
                            inner: HttpClient::new(StaticStatusService { status }),
                        };
                        service.call(request()).await.unwrap();
                    }
                });
            });

            let recorded: Vec<_> = snapshotter
                .snapshot()
                .into_vec()
                .into_iter()
                .map(|(ck, _unit, _desc, value)| (ck.key().clone(), value))
                .collect();

            assert_eq!(throttle_count(&recorded, "s3", "429"), 1);
            assert_eq!(throttle_count(&recorded, "s3", "503"), 2);
            // Other server errors (5xx) are retryable and counted by their status.
            assert_eq!(throttle_count(&recorded, "s3", "500"), 1);
            // Success and non-retryable client errors are not throttles.
            assert_eq!(throttle_count(&recorded, "s3", "200"), 0);
            assert_eq!(throttle_count(&recorded, "s3", "404"), 0);
            // The scheme label is taken from the connector, not shared across schemes.
            assert_eq!(throttle_count(&recorded, "gs", "429"), 1);
            assert_eq!(throttle_count(&recorded, "gs", "503"), 0);
        }
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
pub use http::MeteringHttpConnector;

#[cfg(test)]
mod tests {
    use super::*;

    use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
    use object_store::memory::InMemory;
    use object_store::{ObjectStoreExt, PutPayload};

    fn payload(data: &[u8]) -> PutPayload {
        PutPayload::from_bytes(Bytes::copy_from_slice(data))
    }

    fn metered_store() -> Arc<dyn object_store::ObjectStore> {
        (Arc::new(InMemory::new()) as Arc<dyn object_store::ObjectStore>).metered("memory".into())
    }

    /// A single materialized snapshot of recorded metrics. It must be taken
    /// only once: the snapshotter *drains* histogram samples on every
    /// `snapshot()` call, so a second snapshot would see empty histograms.
    type Metrics = Vec<(metrics::Key, DebugValue)>;

    /// Materialize the current recorder state. Histogram samples are *drained*
    /// on each call, so a metric must be read from a single snapshot.
    fn snapshot(snapshotter: &Snapshotter) -> Metrics {
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(ck, _unit, _desc, value)| (ck.key().clone(), value))
            .collect()
    }

    /// Run an async closure with a thread-local metrics recorder installed and
    /// return the resulting metrics. Uses a current-thread runtime so all polls
    /// happen on the thread that holds the recorder guard.
    fn capture_metrics<F, Fut>(f: F) -> Metrics
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(f());
        });
        snapshot(&snapshotter)
    }

    fn key_matches(key: &metrics::Key, name: &str, labels: &[(&str, &str)]) -> bool {
        if key.name() != name {
            return false;
        }
        let actual: std::collections::HashSet<(&str, &str)> =
            key.labels().map(|l| (l.key(), l.value())).collect();
        labels.len() == actual.len() && labels.iter().all(|l| actual.contains(l))
    }

    fn counter_value(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> u64 {
        for (key, value) in metrics {
            if key_matches(key, name, labels)
                && let DebugValue::Counter(v) = value
            {
                return *v;
            }
        }
        0
    }

    fn histogram_count(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> usize {
        for (key, value) in metrics {
            if key_matches(key, name, labels)
                && let DebugValue::Histogram(samples) = value
            {
                return samples.len();
            }
        }
        0
    }

    fn gauge_value(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> f64 {
        for (key, value) in metrics {
            if key_matches(key, name, labels)
                && let DebugValue::Gauge(v) = value
            {
                return v.0;
            }
        }
        0.0
    }

    fn has_metric(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> bool {
        metrics
            .iter()
            .any(|(key, _)| key_matches(key, name, labels))
    }

    #[test]
    fn test_put_records_count_bytes_and_latency() {
        let data = b"hello world";
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            store
                .put(&Path::from("a/b.bin"), payload(data))
                .await
                .unwrap();
        });

        let labels = [("operation", "put"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 1);
        assert_eq!(
            counter_value(&recorded, METRIC_BYTES, &labels),
            data.len() as u64
        );
        assert_eq!(histogram_count(&recorded, METRIC_DURATION, &labels), 1);
    }

    #[test]
    fn test_get_records_count_and_bytes() {
        let data = b"hello world";
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            let path = Path::from("a/b.bin");
            store.put(&path, payload(data)).await.unwrap();
            store.get(&path).await.unwrap();
        });

        let labels = [("operation", "get"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 1);
        assert_eq!(
            counter_value(&recorded, METRIC_BYTES, &labels),
            data.len() as u64
        );
    }

    #[test]
    fn test_head_is_a_separate_operation() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            let path = Path::from("a/b.bin");
            store.put(&path, payload(b"hello world")).await.unwrap();
            store.head(&path).await.unwrap();
        });

        assert_eq!(
            counter_value(
                &recorded,
                METRIC_REQUESTS,
                &[("operation", "head"), ("scheme", "memory")]
            ),
            1
        );
        // The head call must not be counted as a get.
        assert_eq!(
            counter_value(
                &recorded,
                METRIC_REQUESTS,
                &[("operation", "get"), ("scheme", "memory")]
            ),
            0
        );
        // A HEAD transfers only metadata, so it records no payload bytes.
        assert_eq!(
            counter_value(
                &recorded,
                METRIC_BYTES,
                &[("operation", "head"), ("scheme", "memory")]
            ),
            0
        );
    }

    #[test]
    fn test_delete_records_count_per_path() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            let path = Path::from("a/b.bin");
            store.put(&path, payload(b"x")).await.unwrap();
            store.delete(&path).await.unwrap();
        });

        assert_eq!(
            counter_value(
                &recorded,
                METRIC_REQUESTS,
                &[("operation", "delete"), ("scheme", "memory")]
            ),
            1
        );
    }

    #[test]
    fn test_list_counts_one_request_not_per_item() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            for i in 0..3 {
                store
                    .put(&Path::from(format!("a/{i}.bin")), payload(b"x"))
                    .await
                    .unwrap();
            }
            let _: Vec<_> = store.list(Some(&Path::from("a"))).collect().await;
        });

        assert_eq!(
            counter_value(
                &recorded,
                METRIC_REQUESTS,
                &[("operation", "list"), ("scheme", "memory")]
            ),
            1
        );
    }

    #[test]
    fn test_error_is_counted() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            // Getting a missing object errors.
            let _ = store.get(&Path::from("does/not/exist")).await;
        });

        let labels = [("operation", "get"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_ERRORS, &labels), 1);
        // A failed request is still counted as a request, with latency recorded.
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 1);
        assert_eq!(histogram_count(&recorded, METRIC_DURATION, &labels), 1);
        // No bytes are transferred on a failed get.
        assert_eq!(counter_value(&recorded, METRIC_BYTES, &labels), 0);
    }

    #[test]
    fn test_get_ranges_sums_part_bytes_and_labels_get() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            let path = Path::from("a/b.bin");
            store.put(&path, payload(b"hello world")).await.unwrap();
            // Two disjoint ranges of 3 bytes each.
            store.get_ranges(&path, &[2..5, 6..9]).await.unwrap();
        });

        let labels = [("operation", "get"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 1);
        assert_eq!(counter_value(&recorded, METRIC_BYTES, &labels), 6);
        assert_eq!(histogram_count(&recorded, METRIC_DURATION, &labels), 1);
    }

    #[test]
    fn test_copy_and_rename_record_zero_bytes() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            store
                .put(&Path::from("a/src"), payload(b"x"))
                .await
                .unwrap();
            store
                .copy(&Path::from("a/src"), &Path::from("a/copy"))
                .await
                .unwrap();
            store
                .rename(&Path::from("a/copy"), &Path::from("a/moved"))
                .await
                .unwrap();
        });

        for operation in ["copy", "rename"] {
            let labels = [("operation", operation), ("scheme", "memory")];
            assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 1);
            assert_eq!(counter_value(&recorded, METRIC_BYTES, &labels), 0);
            assert_eq!(histogram_count(&recorded, METRIC_DURATION, &labels), 1);
        }
    }

    #[test]
    fn test_list_with_delimiter_records_latency() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            store.put(&Path::from("a/b"), payload(b"x")).await.unwrap();
            store
                .list_with_delimiter(Some(&Path::from("a")))
                .await
                .unwrap();
        });

        let labels = [("operation", "list"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 1);
        assert_eq!(histogram_count(&recorded, METRIC_DURATION, &labels), 1);
        assert_eq!(counter_value(&recorded, METRIC_BYTES, &labels), 0);
    }

    #[test]
    fn test_list_with_offset_counts_one_request() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            for i in 0..3 {
                store
                    .put(&Path::from(format!("a/{i}")), payload(b"x"))
                    .await
                    .unwrap();
            }
            let _: Vec<_> = store
                .list_with_offset(Some(&Path::from("a")), &Path::from("a/0"))
                .collect()
                .await;
        });

        assert_eq!(
            counter_value(
                &recorded,
                METRIC_REQUESTS,
                &[("operation", "list"), ("scheme", "memory")]
            ),
            1
        );
    }

    #[test]
    fn test_multipart_records_each_part_as_put() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            let mut upload = store.put_multipart(&Path::from("a/big")).await.unwrap();
            upload.put_part(payload(b"hello")).await.unwrap(); // 5 bytes
            upload.put_part(payload(b"world!!")).await.unwrap(); // 7 bytes
            upload.complete().await.unwrap();
        });

        let labels = [("operation", "put"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 2);
        assert_eq!(counter_value(&recorded, METRIC_BYTES, &labels), 12);
        // Each part records its own latency sample, like a unary put.
        assert_eq!(histogram_count(&recorded, METRIC_DURATION, &labels), 2);
        // A successful part upload records no error.
        assert_eq!(counter_value(&recorded, METRIC_ERRORS, &labels), 0);
    }

    #[test]
    fn test_multipart_part_error_is_counted() {
        let recorded = capture_metrics(|| async {
            let store = (Arc::new(FailingStreamStore) as Arc<dyn object_store::ObjectStore>)
                .metered("memory".into());
            let mut upload = store.put_multipart(&Path::from("a/big")).await.unwrap();
            let _ = upload.put_part(payload(b"data")).await;
        });

        let labels = [("operation", "put"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &labels), 1);
        assert_eq!(counter_value(&recorded, METRIC_ERRORS, &labels), 1);
        assert_eq!(histogram_count(&recorded, METRIC_DURATION, &labels), 1);
        // A failed part transfers no counted bytes.
        assert_eq!(counter_value(&recorded, METRIC_BYTES, &labels), 0);
    }

    #[test]
    fn test_in_flight_guard_tracks_and_releases() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let labels = [("operation", "get"), ("scheme", "memory")];
        metrics::with_local_recorder(&recorder, || {
            let g1 = InFlightGuard::new("memory", "get");
            let g2 = InFlightGuard::new("memory", "get");
            assert_eq!(
                gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                2.0
            );
            drop(g1);
            assert_eq!(
                gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                1.0
            );
            drop(g2);
            assert_eq!(
                gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                0.0
            );
        });
    }

    #[test]
    fn test_in_flight_gauge_is_wired_and_balances() {
        let recorded = capture_metrics(|| async {
            let store = metered_store();
            let path = Path::from("a/b.bin");
            store.put(&path, payload(b"hello")).await.unwrap();
            store.get(&path).await.unwrap();
        });

        // The gauge is emitted for each operation (guard is wired in) and, once
        // the operation completes, balances back to zero.
        for operation in ["put", "get"] {
            let labels = [("operation", operation), ("scheme", "memory")];
            assert!(has_metric(&recorded, METRIC_IN_FLIGHT, &labels));
            assert_eq!(gauge_value(&recorded, METRIC_IN_FLIGHT, &labels), 0.0);
        }
    }

    #[test]
    fn test_list_stream_holds_in_flight_until_dropped() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let labels = [("operation", "list"), ("scheme", "memory")];
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(async {
                let store = metered_store();
                store.put(&Path::from("a/x"), payload(b"x")).await.unwrap();

                // Creating the stream raises the gauge; it stays raised until the
                // stream is dropped, even before any items are drained.
                let stream = store.list(Some(&Path::from("a")));
                assert_eq!(
                    gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                    1.0
                );
                drop(stream);
                assert_eq!(
                    gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                    0.0
                );
            });
        });
    }

    #[test]
    fn test_delete_stream_holds_in_flight_until_dropped() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let labels = [("operation", "delete"), ("scheme", "memory")];
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(async {
                let store = metered_store();
                let locations = futures::stream::iter(vec![Ok(Path::from("a/b"))]).boxed();

                // Like list, creating the delete stream raises the gauge and holds
                // it until the stream is dropped, before any items are drained.
                let stream = store.delete_stream(locations);
                assert_eq!(
                    gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                    1.0
                );
                drop(stream);
                assert_eq!(
                    gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                    0.0
                );
            });
        });
    }

    #[test]
    fn test_in_flight_released_when_operation_future_dropped() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let labels = [("operation", "get"), ("scheme", "memory")];
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(async {
                let started = Arc::new(tokio::sync::Notify::new());
                // Never signalled: the request stays blocked mid-flight.
                let release = Arc::new(tokio::sync::Notify::new());
                let store = (Arc::new(BlockingStore {
                    started: started.clone(),
                    release,
                }) as Arc<dyn object_store::ObjectStore>)
                    .metered("memory".into());

                let path = Path::from("a/b");
                let mut fut = Box::pin(store.get(&path));
                // Drive the request until it is blocked inside the inner store.
                tokio::select! {
                    _ = &mut fut => unreachable!("the blocking store never returns"),
                    _ = started.notified() => {}
                }

                // The gauge is raised while the request is outstanding, and
                // dropping the future before it completes releases it.
                assert_eq!(
                    gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                    1.0
                );
                drop(fut);
                assert_eq!(
                    gauge_value(&snapshot(&snapshotter), METRIC_IN_FLIGHT, &labels),
                    0.0
                );
            });
        });
    }

    #[test]
    fn test_streaming_errors_are_counted() {
        let recorded = capture_metrics(|| async {
            let delete_store = (Arc::new(FailingStreamStore) as Arc<dyn object_store::ObjectStore>)
                .metered("memory".into());
            let _ = delete_store.delete(&Path::from("a/b")).await;

            let list_store = (Arc::new(FailingStreamStore) as Arc<dyn object_store::ObjectStore>)
                .metered("memory".into());
            let _: Vec<_> = list_store.list(None).collect().await;
        });

        // delete_stream counts the item and records an error when it fails.
        let delete_labels = [("operation", "delete"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &delete_labels), 1);
        assert_eq!(counter_value(&recorded, METRIC_ERRORS, &delete_labels), 1);

        // A list request is counted once; a failure while draining records an error.
        let list_labels = [("operation", "list"), ("scheme", "memory")];
        assert_eq!(counter_value(&recorded, METRIC_REQUESTS, &list_labels), 1);
        assert_eq!(counter_value(&recorded, METRIC_ERRORS, &list_labels), 1);
    }

    /// A store whose stream-producing operations always yield an error, used to
    /// exercise the error branches of the streaming wrappers.
    #[derive(Debug)]
    struct FailingStreamStore;

    impl std::fmt::Display for FailingStreamStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FailingStreamStore")
        }
    }

    fn test_error() -> object_store::Error {
        object_store::Error::Generic {
            store: "FailingStreamStore",
            source: "injected failure".into(),
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for FailingStreamStore {
        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> OSResult<PutResult> {
            unimplemented!()
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            Ok(Box::new(FailingUpload))
        }

        async fn get_opts(&self, _location: &Path, _options: GetOptions) -> OSResult<GetResult> {
            unimplemented!()
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            futures::stream::once(async { Err(test_error()) }).boxed()
        }

        fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            futures::stream::once(async { Err(test_error()) }).boxed()
        }

        fn list_with_offset(
            &self,
            _prefix: Option<&Path>,
            _offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            unimplemented!()
        }

        async fn list_with_delimiter(&self, _prefix: Option<&Path>) -> OSResult<ListResult> {
            unimplemented!()
        }

        async fn copy_opts(&self, _from: &Path, _to: &Path, _opts: CopyOptions) -> OSResult<()> {
            unimplemented!()
        }

        async fn rename_opts(
            &self,
            _from: &Path,
            _to: &Path,
            _opts: RenameOptions,
        ) -> OSResult<()> {
            unimplemented!()
        }
    }

    /// A [`MultipartUpload`] whose part uploads always fail, used to exercise the
    /// error branch of the metered `put_part`.
    #[derive(Debug)]
    struct FailingUpload;

    #[async_trait::async_trait]
    impl MultipartUpload for FailingUpload {
        fn put_part(&mut self, _data: PutPayload) -> UploadPart {
            async { Err(test_error()) }.boxed()
        }

        async fn complete(&mut self) -> OSResult<PutResult> {
            unimplemented!()
        }

        async fn abort(&mut self) -> OSResult<()> {
            unimplemented!()
        }
    }

    /// A store whose `get_opts` blocks after signalling `started`, so a request
    /// can be observed mid-flight and then dropped before it completes.
    #[derive(Debug)]
    struct BlockingStore {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    impl std::fmt::Display for BlockingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BlockingStore")
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for BlockingStore {
        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> OSResult<PutResult> {
            unimplemented!()
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            unimplemented!()
        }

        async fn get_opts(&self, _location: &Path, _options: GetOptions) -> OSResult<GetResult> {
            self.started.notify_one();
            self.release.notified().await;
            unreachable!("release is never signalled in the test")
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            unimplemented!()
        }

        fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            unimplemented!()
        }

        fn list_with_offset(
            &self,
            _prefix: Option<&Path>,
            _offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            unimplemented!()
        }

        async fn list_with_delimiter(&self, _prefix: Option<&Path>) -> OSResult<ListResult> {
            unimplemented!()
        }

        async fn copy_opts(&self, _from: &Path, _to: &Path, _opts: CopyOptions) -> OSResult<()> {
            unimplemented!()
        }

        async fn rename_opts(
            &self,
            _from: &Path,
            _to: &Path,
            _opts: RenameOptions,
        ) -> OSResult<()> {
            unimplemented!()
        }
    }
}
