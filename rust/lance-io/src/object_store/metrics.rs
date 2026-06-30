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
//!   per-operation request counts, transferred bytes, latency, and errors. It
//!   works for every store regardless of backend.
//! * [`MeteringHttpConnector`] wraps the HTTP client used by the native cloud
//!   stores (S3 / GCS / Azure) and records throttle responses (HTTP 429 / 503)
//!   per attempt. Because `object_store`'s retry loop re-issues each request
//!   through the [`HttpService`](object_store::client::HttpService), this sees
//!   every retried throttle, which a store-level wrapper cannot observe.
//!
//! All metrics carry a `scheme` label (e.g. `s3`, `gs`, `azure`).

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
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
        self.target
            .delete_stream(locations)
            .map(move |result| {
                match &result {
                    Ok(_) => record_count(&scheme, "delete"),
                    Err(_) => {
                        record_count(&scheme, "delete");
                        record_error(&scheme, "delete");
                    }
                }
                result
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        record_count(&self.scheme, "list");
        meter_list_stream(self.target.list(prefix), self.scheme.clone())
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
        )
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        let start = Instant::now();
        let result = self.target.list_with_delimiter(prefix).await;
        record_request(&self.scheme, "list", start, 0, &result);
        result
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        let start = Instant::now();
        let result = self.target.copy_opts(from, to, opts).await;
        record_request(&self.scheme, "copy", start, 0, &result);
        result
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
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
) -> BoxStream<'static, OSResult<ObjectMeta>> {
    stream
        .map(move |result| {
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
        record_count(&self.scheme, "put");
        metrics::counter!(METRIC_BYTES, "operation" => "put", "scheme" => self.scheme.clone())
            .increment(data.content_length() as u64);
        self.target.put_part(data)
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
                    for status in [429u16, 503, 503, 200, 404] {
                        let service = MeteringHttpService {
                            scheme: "s3".into(),
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
            // Success and non-retryable client errors are not throttles.
            assert_eq!(throttle_count(&recorded, "s3", "200"), 0);
            assert_eq!(throttle_count(&recorded, "s3", "404"), 0);
        }
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
pub use http::MeteringHttpConnector;

#[cfg(test)]
mod tests {
    use super::*;

    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
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
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(ck, _unit, _desc, value)| (ck.key().clone(), value))
            .collect()
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

        assert_eq!(
            counter_value(
                &recorded,
                METRIC_ERRORS,
                &[("operation", "get"), ("scheme", "memory")]
            ),
            1
        );
    }
}
