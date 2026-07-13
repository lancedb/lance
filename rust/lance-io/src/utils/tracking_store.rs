// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Make assertions about IO operations to an [ObjectStore].
//!
//! When testing code that performs IO, you will often want to make assertions
//! about the number of reads and writes performed, the amount of data read or
//! written, and the number of disjoint periods where at least one IO is in-flight.
//!
//! This modules provides [`IOTracker`] which can be used to wrap any object store.
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::pin::Pin;
#[cfg(feature = "test-util")]
use std::sync::atomic::AtomicU16;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
    Result as OSResult, UploadPart,
};

use crate::object_store::WrappingObjectStore;

/// Stable logical operation categories for I/O accounting.
///
/// Request records preserve the concrete API method for existing diagnostics,
/// while this category lets callers aggregate equivalent methods such as
/// `list`, `list_with_offset`, and `list_with_delimiter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoOperation {
    /// Object payload read, including ranged and multi-range reads.
    Get,
    /// Metadata-only object read.
    Head,
    /// Object enumeration with any supported pagination/delimiter API.
    List,
    /// Object data write, including multipart parts.
    Put,
    /// Object deletion.
    Delete,
    /// Server-side object copy.
    Copy,
    /// Server-side object rename.
    Rename,
    /// Operation without a more specific stable category.
    Other,
}

impl IoOperation {
    fn from_method(method: &str) -> Self {
        match method {
            "head" => Self::Head,
            "list" | "list_with_offset" | "list_with_delimiter" => Self::List,
            "put" | "put_opts" | "put_part" => Self::Put,
            "delete" => Self::Delete,
            "copy" => Self::Copy,
            "rename" => Self::Rename,
            method if method.starts_with("get") => Self::Get,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct IOTracker(Arc<Mutex<IoStats>>);

impl IOTracker {
    /// Wrap an object store and record its logical I/O in this tracker.
    pub fn wrap_store(&self, target: Arc<dyn ObjectStore>) -> Arc<IoTrackingStore> {
        Arc::new(IoTrackingStore::new(target, self.0.clone()))
    }

    /// Get IO statistics and reset the counters (incremental pattern).
    ///
    /// This returns the accumulated statistics since the last call and resets
    /// the internal counters to zero.
    pub fn incremental_stats(&self) -> IoStats {
        std::mem::take(&mut *self.0.lock().unwrap())
    }

    /// Get a snapshot of current IO statistics without resetting counters.
    ///
    /// This returns a clone of the current statistics without modifying the
    /// internal state. Use this when you need to check stats without resetting.
    pub fn stats(&self) -> IoStats {
        self.0.lock().unwrap().clone()
    }

    /// Record a read operation for tracking.
    ///
    /// This is used by readers that bypass the ObjectStore layer (like LocalObjectReader)
    /// to ensure their IO operations are still tracked.
    pub fn record_read(
        &self,
        #[allow(unused_variables)] method: &'static str,
        #[allow(unused_variables)] path: Path,
        num_bytes: u64,
        #[allow(unused_variables)] range: Option<Range<u64>>,
    ) {
        let mut stats = self.0.lock().unwrap();
        stats.read_iops += 1;
        stats.read_bytes += num_bytes;
        #[cfg(feature = "test-util")]
        stats.requests.push(IoRequestRecord {
            method,
            operation: IoOperation::from_method(method),
            path,
            range,
            num_bytes,
        });
    }

    /// Record a write operation for tracking.
    ///
    /// This is used by writers that bypass the ObjectStore layer (like LocalWriter)
    /// to ensure their IO operations are still tracked.
    pub fn record_write(
        &self,
        #[allow(unused_variables)] method: &'static str,
        #[allow(unused_variables)] path: Path,
        num_bytes: u64,
    ) {
        let mut stats = self.0.lock().unwrap();
        stats.write_iops += 1;
        stats.written_bytes += num_bytes;
        #[cfg(feature = "test-util")]
        stats.requests.push(IoRequestRecord {
            method,
            operation: IoOperation::from_method(method),
            path,
            range: None,
            num_bytes,
        });
    }
}

impl WrappingObjectStore for IOTracker {
    fn wrap(&self, _store_prefix: &str, target: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(IoTrackingStore::new(target, self.0.clone()))
    }
}

#[derive(Debug, Default, Clone)]
pub struct IoStats {
    pub read_iops: u64,
    pub read_bytes: u64,
    pub write_iops: u64,
    pub written_bytes: u64,
    // This is only really meaningful in tests where there isn't any concurrent IO.
    #[cfg(feature = "test-util")]
    /// Number of disjoint periods where at least one IO is in-flight.
    pub num_stages: u64,
    #[cfg(feature = "test-util")]
    pub requests: Vec<IoRequestRecord>,
}

/// Assertions on IO statistics.
/// assert_io_eq!(io_stats, read_iops, 1);
/// assert_io_eq!(io_stats, write_iops, 0, "should be no writes");
/// assert_io_eq!(io_stats, num_hops, 1, "should be just {}", "one hop");
#[cfg(feature = "test-util")]
#[macro_export]
macro_rules! assert_io_eq {
    ($io_stats:expr, $field:ident, $expected:expr) => {
        assert_eq!(
            $io_stats.$field, $expected,
            "Expected {} to be {}, got {}. Requests: {:#?}",
            stringify!($field),
            $expected,
            $io_stats.$field,
            $io_stats.requests
        );
    };
    ($io_stats:expr, $field:ident, $expected:expr, $($arg:tt)+) => {
        assert_eq!(
            $io_stats.$field, $expected,
            "Expected {} to be {}, got {}. Requests: {:#?} {}",
            stringify!($field),
            $expected,
            $io_stats.$field,
            $io_stats.requests,
            format_args!($($arg)+)
        );
    };
}

#[cfg(feature = "test-util")]
#[macro_export]
macro_rules! assert_io_gt {
    ($io_stats:expr, $field:ident, $expected:expr) => {
        assert!(
            $io_stats.$field > $expected,
            "Expected {} to be > {}, got {}. Requests: {:#?}",
            stringify!($field),
            $expected,
            $io_stats.$field,
            $io_stats.requests
        );
    };
    ($io_stats:expr, $field:ident, $expected:expr, $($arg:tt)+) => {
        assert!(
            $io_stats.$field > $expected,
            "Expected {} to be > {}, got {}. Requests: {:#?} {}",
            stringify!($field),
            $expected,
            $io_stats.$field,
            $io_stats.requests,
            format_args!($($arg)+)
        );
    };
}

#[cfg(feature = "test-util")]
#[macro_export]
macro_rules! assert_io_lt {
    ($io_stats:expr, $field:ident, $expected:expr) => {
        assert!(
            $io_stats.$field < $expected,
            "Expected {} to be < {}, got {}. Requests: {:#?}",
            stringify!($field),
            $expected,
            $io_stats.$field,
            $io_stats.requests
        );
    };
    ($io_stats:expr, $field:ident, $expected:expr, $($arg:tt)+) => {
        assert!(
            $io_stats.$field < $expected,
            "Expected {} to be < {}, got {}. Requests: {:#?} {}",
            stringify!($field),
            $expected,
            $io_stats.$field,
            $io_stats.requests,
            format_args!($($arg)+)
        );
    };
}

// These request records only exist for test-only diagnostics.
#[cfg(feature = "test-util")]
#[derive(Clone)]
pub struct IoRequestRecord {
    /// Concrete object-store API method that Lance issued.
    pub method: &'static str,
    /// Logical operation category, independent of the concrete API method.
    pub operation: IoOperation,
    pub path: Path,
    /// Requested bounded byte range, when one was supplied.
    pub range: Option<Range<u64>>,
    /// Bytes returned or written by this logical operation.
    ///
    /// Streamed GET requests are recorded when the response is accepted, and
    /// their byte count is updated as chunks are consumed. File payloads report
    /// the exact returned range because consumption happens outside the
    /// object-store boundary.
    pub num_bytes: u64,
}

#[cfg(feature = "test-util")]
impl std::fmt::Debug for IoRequestRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // For example: "put /path/to/file range: 0-100"
        write!(
            f,
            "IORequest(method={}, operation={:?}, path=\"{}\"",
            self.method, self.operation, self.path
        )?;
        if let Some(range) = &self.range {
            write!(f, ", range={:?}", range)?;
        }
        write!(f, ", num_bytes={}", self.num_bytes)?;
        write!(f, ")")?;
        Ok(())
    }
}

impl Display for IoStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

#[derive(Debug)]
pub struct IoTrackingStore {
    target: Arc<dyn ObjectStore>,
    stats: Arc<Mutex<IoStats>>,
    #[cfg(feature = "test-util")]
    active_requests: Arc<AtomicU16>,
}

impl Display for IoTrackingStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl IoTrackingStore {
    pub fn new(target: Arc<dyn ObjectStore>, stats: Arc<Mutex<IoStats>>) -> Self {
        Self {
            target,
            stats,
            #[cfg(feature = "test-util")]
            active_requests: Arc::new(AtomicU16::new(0)),
        }
    }

    fn record_read(
        &self,
        method: &'static str,
        path: Path,
        num_bytes: u64,
        range: Option<Range<u64>>,
    ) {
        self.record_read_as(
            method,
            IoOperation::from_method(method),
            path,
            num_bytes,
            range,
        );
    }

    fn record_read_as(
        &self,
        method: &'static str,
        #[allow(unused_variables)] operation: IoOperation,
        path: Path,
        num_bytes: u64,
        range: Option<Range<u64>>,
    ) {
        let mut stats = self.stats.lock().unwrap();
        stats.read_iops += 1;
        stats.read_bytes += num_bytes;
        #[cfg(feature = "test-util")]
        stats.requests.push(IoRequestRecord {
            method,
            operation,
            path,
            range,
            num_bytes,
        });
        #[cfg(not(feature = "test-util"))]
        let _ = (method, path, range); // Suppress unused variable warnings
    }

    fn record_write(&self, method: &'static str, path: Path, num_bytes: u64) {
        let mut stats = self.stats.lock().unwrap();
        stats.write_iops += 1;
        stats.written_bytes += num_bytes;
        #[cfg(feature = "test-util")]
        stats.requests.push(IoRequestRecord {
            method,
            operation: IoOperation::from_method(method),
            path,
            range: None,
            num_bytes,
        });
        #[cfg(not(feature = "test-util"))]
        let _ = (method, path); // Suppress unused variable warnings
    }

    #[cfg(feature = "test-util")]
    fn stage_guard(&self) -> StageGuard {
        StageGuard::new(self.active_requests.clone(), self.stats.clone())
    }

    #[cfg(not(feature = "test-util"))]
    fn stage_guard(&self) -> StageGuard {
        StageGuard
    }
}

#[async_trait::async_trait]
#[deny(clippy::missing_trait_methods)]
impl ObjectStore for IoTrackingStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        let _guard = self.stage_guard();
        self.record_write(
            "put_opts",
            location.to_owned(),
            bytes.content_length() as u64,
        );
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        let _guard = self.stage_guard();
        let target = self.target.put_multipart_opts(location, opts).await?;
        Ok(Box::new(IoTrackingMultipartUpload {
            target,
            stats: self.stats.clone(),
            #[cfg(feature = "test-util")]
            path: location.to_owned(),
            #[cfg(feature = "test-util")]
            _guard,
        }))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        let guard = self.stage_guard();
        let is_head = options.head;
        let range = match &options.range {
            Some(GetRange::Bounded(range)) => Some(range.clone()),
            _ => None, // TODO: fill in other options.
        };
        let result = match self.target.get_opts(location, options).await {
            Ok(result) => result,
            Err(error) => return Err(error),
        };

        if is_head {
            self.record_read_as("get_opts", IoOperation::Head, location.to_owned(), 0, range);
            return Ok(result);
        }

        Ok(track_get_result(
            result,
            self.stats.clone(),
            location.to_owned(),
            range,
            guard,
        ))
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        let _guard = self.stage_guard();
        let result = self.target.get_ranges(location, ranges).await;
        if let Ok(result) = &result {
            self.record_read(
                "get_ranges",
                location.to_owned(),
                result.iter().map(|b| b.len() as u64).sum(),
                None,
            );
        }
        result
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        // A delete stream is one logical request. Native stores may split or
        // batch it into multiple HTTP attempts, which is measured separately
        // below object_store's retry boundary.
        self.record_write("delete", Path::default(), 0);
        self.target.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let _guard = self.stage_guard();
        self.record_read("list", prefix.cloned().unwrap_or_default(), 0, None);
        self.target.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.record_read(
            "list_with_offset",
            prefix.cloned().unwrap_or_default(),
            0,
            None,
        );
        self.target.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        let _guard = self.stage_guard();
        self.record_read(
            "list_with_delimiter",
            prefix.cloned().unwrap_or_default(),
            0,
            None,
        );
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        let _guard = self.stage_guard();
        self.record_write("copy", from.to_owned(), 0);
        self.target.copy_opts(from, to, opts).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        let _guard = self.stage_guard();
        self.record_write("rename", from.to_owned(), 0);
        self.target.rename_opts(from, to, opts).await
    }
}

/// Wrap a successful GET result so streamed payloads report bytes as they are
/// consumed. A file payload cannot be observed after it leaves the object-store
/// boundary, so its exact returned range is recorded immediately.
fn track_get_result(
    mut result: GetResult,
    stats: Arc<Mutex<IoStats>>,
    path: Path,
    requested_range: Option<Range<u64>>,
    guard: StageGuard,
) -> GetResult {
    match result.payload {
        GetResultPayload::Stream(stream) => {
            let request_index =
                begin_stream_read(&stats, "get_opts", IoOperation::Get, path, requested_range);
            result.payload = GetResultPayload::Stream(
                IoTrackingGetStream {
                    inner: stream,
                    stats,
                    request_index,
                    _guard: guard,
                }
                .boxed(),
            );
        }
        other => {
            let num_bytes = result.range.end - result.range.start;
            record_read_stats(
                &stats,
                "get_opts",
                IoOperation::Get,
                path,
                num_bytes,
                requested_range,
            );
            result.payload = other;
        }
    }
    result
}

fn begin_stream_read(
    stats: &Mutex<IoStats>,
    #[allow(unused_variables)] method: &'static str,
    #[allow(unused_variables)] operation: IoOperation,
    #[allow(unused_variables)] path: Path,
    #[allow(unused_variables)] range: Option<Range<u64>>,
) -> Option<usize> {
    let mut stats = stats.lock().unwrap();
    stats.read_iops += 1;
    #[cfg(feature = "test-util")]
    {
        let request_index = stats.requests.len();
        stats.requests.push(IoRequestRecord {
            method,
            operation,
            path,
            range,
            num_bytes: 0,
        });
        Some(request_index)
    }
    #[cfg(not(feature = "test-util"))]
    {
        None
    }
}

fn record_stream_bytes(
    stats: &Mutex<IoStats>,
    #[allow(unused_variables)] request_index: Option<usize>,
    num_bytes: u64,
) {
    let mut stats = stats.lock().unwrap();
    stats.read_bytes += num_bytes;
    #[cfg(feature = "test-util")]
    if let Some(request) = request_index.and_then(|index| stats.requests.get_mut(index)) {
        request.num_bytes += num_bytes;
    }
}

fn record_read_stats(
    stats: &Mutex<IoStats>,
    #[allow(unused_variables)] method: &'static str,
    #[allow(unused_variables)] operation: IoOperation,
    #[allow(unused_variables)] path: Path,
    num_bytes: u64,
    #[allow(unused_variables)] range: Option<Range<u64>>,
) {
    let mut stats = stats.lock().unwrap();
    stats.read_iops += 1;
    stats.read_bytes += num_bytes;
    #[cfg(feature = "test-util")]
    stats.requests.push(IoRequestRecord {
        method,
        operation,
        path,
        range,
        num_bytes,
    });
}

struct IoTrackingGetStream {
    inner: BoxStream<'static, OSResult<Bytes>>,
    stats: Arc<Mutex<IoStats>>,
    request_index: Option<usize>,
    _guard: StageGuard,
}

impl Stream for IoTrackingGetStream {
    type Item = OSResult<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                record_stream_bytes(&self.stats, self.request_index, chunk.len() as u64);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[derive(Debug)]
struct IoTrackingMultipartUpload {
    target: Box<dyn MultipartUpload>,
    #[cfg(feature = "test-util")]
    path: Path,
    stats: Arc<Mutex<IoStats>>,
    #[cfg(feature = "test-util")]
    _guard: StageGuard,
}

#[async_trait::async_trait]
impl MultipartUpload for IoTrackingMultipartUpload {
    async fn abort(&mut self) -> OSResult<()> {
        self.target.abort().await
    }

    async fn complete(&mut self) -> OSResult<PutResult> {
        self.target.complete().await
    }

    fn put_part(&mut self, payload: PutPayload) -> UploadPart {
        {
            let mut stats = self.stats.lock().unwrap();
            stats.write_iops += 1;
            stats.written_bytes += payload.content_length() as u64;
            #[cfg(feature = "test-util")]
            stats.requests.push(IoRequestRecord {
                method: "put_part",
                operation: IoOperation::Put,
                path: self.path.to_owned(),
                range: None,
                num_bytes: payload.content_length() as u64,
            });
        }
        self.target.put_part(payload)
    }
}

#[cfg(feature = "test-util")]
#[derive(Debug)]
struct StageGuard {
    active_requests: Arc<AtomicU16>,
    stats: Arc<Mutex<IoStats>>,
}

#[cfg(not(feature = "test-util"))]
struct StageGuard;

#[cfg(feature = "test-util")]
impl StageGuard {
    fn new(active_requests: Arc<AtomicU16>, stats: Arc<Mutex<IoStats>>) -> Self {
        active_requests.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self {
            active_requests,
            stats,
        }
    }
}

#[cfg(feature = "test-util")]
impl Drop for StageGuard {
    fn drop(&mut self) {
        if self
            .active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
            == 1
        {
            let mut stats = self.stats.lock().unwrap();
            stats.num_stages += 1;
        }
    }
}

#[cfg(all(test, feature = "test-util"))]
mod tests {
    use chrono::Utc;
    use futures::{TryStreamExt, stream};
    use object_store::memory::InMemory;
    use object_store::{Attributes, ObjectStoreExt};

    use super::*;

    #[tokio::test]
    async fn records_actual_streamed_get_bytes() {
        let tracker = IOTracker::default();
        let store = tracker.wrap_store(Arc::new(InMemory::new()));
        let location = Path::from("object");
        let result = GetResult {
            payload: GetResultPayload::Stream(
                stream::iter([
                    Ok(Bytes::from_static(b"abc")),
                    Ok(Bytes::from_static(b"de")),
                ])
                .boxed(),
            ),
            meta: ObjectMeta {
                location: location.clone(),
                last_modified: Utc::now(),
                size: 1_000,
                e_tag: None,
                version: None,
            },
            // Deliberately differ from the payload size. The request record
            // must reflect observed payload bytes, not declared range length.
            range: 0..1_000,
            attributes: Attributes::default(),
        };

        let result = track_get_result(
            result,
            tracker.0.clone(),
            location,
            Some(0..1_000),
            store.stage_guard(),
        );
        let stats = tracker.stats();
        assert_eq!(stats.read_iops, 1);
        assert_eq!(stats.read_bytes, 0);
        assert_eq!(stats.requests[0].num_bytes, 0);

        assert_eq!(result.bytes().await.unwrap(), Bytes::from_static(b"abcde"));

        let stats = tracker.stats();
        assert_eq!(stats.read_iops, 1);
        assert_eq!(stats.read_bytes, 5);
        assert_eq!(stats.requests.len(), 1);
        assert_eq!(stats.requests[0].method, "get_opts");
        assert_eq!(stats.requests[0].operation, IoOperation::Get);
        assert_eq!(stats.requests[0].range, Some(0..1_000));
        assert_eq!(stats.requests[0].num_bytes, 5);
    }

    #[tokio::test]
    async fn distinguishes_head_list_and_list_with_delimiter() {
        let tracker = IOTracker::default();
        let store = IoTrackingStore::new(Arc::new(InMemory::new()), tracker.0.clone());
        let location = Path::from("prefix/object");
        store
            .put(&location, PutPayload::from_static(b"payload"))
            .await
            .unwrap();
        tracker.incremental_stats();

        store.head(&location).await.unwrap();
        store
            .list(Some(&Path::from("prefix")))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        store
            .list_with_delimiter(Some(&Path::from("prefix")))
            .await
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.read_iops, 3);
        assert_eq!(stats.read_bytes, 0);
        let methods = stats
            .requests
            .iter()
            .map(|request| request.method)
            .collect::<Vec<_>>();
        assert_eq!(methods, ["get_opts", "list", "list_with_delimiter"]);
        let operations = stats
            .requests
            .iter()
            .map(|request| request.operation)
            .collect::<Vec<_>>();
        assert_eq!(
            operations,
            [IoOperation::Head, IoOperation::List, IoOperation::List]
        );
        assert!(stats.requests.iter().all(|request| request.num_bytes == 0));
    }

    #[tokio::test]
    async fn records_write_bytes_on_each_request() {
        let tracker = IOTracker::default();
        let store = IoTrackingStore::new(Arc::new(InMemory::new()), tracker.0.clone());
        store
            .put(&Path::from("object"), PutPayload::from_static(b"payload"))
            .await
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.write_iops, 1);
        assert_eq!(stats.written_bytes, 7);
        assert_eq!(stats.requests.len(), 1);
        assert_eq!(stats.requests[0].method, "put_opts");
        assert_eq!(stats.requests[0].operation, IoOperation::Put);
        assert_eq!(stats.requests[0].num_bytes, 7);
    }

    #[tokio::test]
    async fn records_bulk_delete_as_one_logical_request() {
        let tracker = IOTracker::default();
        let store = IoTrackingStore::new(Arc::new(InMemory::new()), tracker.0.clone());
        let first = Path::from("first");
        let second = Path::from("second");
        for path in [&first, &second] {
            store
                .put(path, PutPayload::from_static(b"payload"))
                .await
                .unwrap();
        }
        tracker.incremental_stats();

        store
            .delete_stream(stream::iter([Ok(first.clone()), Ok(second.clone())]).boxed())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.write_iops, 1);
        assert_eq!(stats.requests.len(), 1);
        assert_eq!(stats.requests[0].method, "delete");
        assert_eq!(stats.requests[0].operation, IoOperation::Delete);
        assert_eq!(stats.requests[0].num_bytes, 0);
    }
}
