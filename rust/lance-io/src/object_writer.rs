// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::Poll;
use std::time::Instant;

use crate::object_store::ObjectStore as LanceObjectStore;
use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::{MultipartUpload, ObjectStoreExt};
use object_store::{ObjectStore, path::Path};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::task::JoinSet;

use lance_core::{Error, Result};
use tracing::Instrument;

use crate::traits::Writer;
use crate::utils::tracking_store::{IOTracker, IoMetricsGuard};
use tokio::runtime::Handle;

/// Start at 5MB.
const INITIAL_UPLOAD_STEP: usize = 1024 * 1024 * 5;

fn max_upload_parallelism() -> usize {
    static MAX_UPLOAD_PARALLELISM: OnceLock<usize> = OnceLock::new();
    *MAX_UPLOAD_PARALLELISM.get_or_init(|| {
        std::env::var("LANCE_UPLOAD_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10)
    })
}

/// Maximum body size for a single S3 PUT: strictly less than 5 GiB.
/// AWS rejects single-PUT bodies of exactly 5 GiB (= 5 * 1024^3) with
/// `EntityTooLarge`, so we clamp `LANCE_INITIAL_UPLOAD_SIZE` one byte
/// below that threshold to keep the buffer-fills-to-clamp single-PUT
/// path safe. See lance#6750 for the related txn-file write fix.
const MAX_UPLOAD_PART_SIZE: usize = 1024 * 1024 * 1024 * 5 - 1;

/// Clamps a requested upload part size to the valid [5MB, 5GB] range.
/// Returns the clamped value and whether clamping was necessary.
fn clamp_initial_upload_size(raw: usize) -> (usize, bool) {
    let clamped = raw.clamp(INITIAL_UPLOAD_STEP, MAX_UPLOAD_PART_SIZE);
    (clamped, clamped != raw)
}

fn initial_upload_size() -> usize {
    static LANCE_INITIAL_UPLOAD_SIZE: OnceLock<usize> = OnceLock::new();
    *LANCE_INITIAL_UPLOAD_SIZE.get_or_init(|| {
        let Some(raw) = std::env::var("LANCE_INITIAL_UPLOAD_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        else {
            return INITIAL_UPLOAD_STEP;
        };
        let (clamped, was_clamped) = clamp_initial_upload_size(raw);
        if was_clamped {
            // OnceLock caches the result, so this warning fires at most once per process.
            tracing::warn!(
                requested = raw,
                clamped,
                "LANCE_INITIAL_UPLOAD_SIZE must be between 5MB and 5GB; clamping to valid range"
            );
        }
        clamped
    })
}

/// Writer to an object in an object store.
///
/// If the object is small enough, the writer will upload the object in a single
/// PUT request. If the object is larger, the writer will create a multipart
/// upload and upload parts in parallel.
///
/// This implements the `AsyncWrite` trait.
pub struct ObjectWriter {
    state: UploadState,
    path: Arc<Path>,
    cursor: usize,
    buffer: Vec<u8>,
    // TODO: use constant size to support R2
    use_constant_size_upload_parts: bool,
}

#[derive(Debug, Clone, Default)]
pub struct WriteResult {
    pub size: usize,
    pub e_tag: Option<String>,
}

/// An object-store upload failure, annotated with what Lance was uploading.
///
/// `object_store` reports its own elapsed time, but its clock starts inside
/// `RetryContext::new`, which runs on the *first poll* of the request future.
/// The `elapsed` reported here is measured from the moment Lance handed the
/// request to the uploader, so the two together tell a slow request (both
/// durations agree) apart from one whose task sat unpolled before it ever
/// issued (this duration is much larger). That distinction is what identifies
/// runtime starvation as the cause of a whole-request timeout, and it is not
/// recoverable from the object-store error alone.
#[derive(Debug)]
struct UploadFailure {
    context: String,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl UploadFailure {
    fn new(context: String, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self {
            context,
            source: Box::new(source),
        }
    }

    fn into_io_error(self) -> io::Error {
        io::Error::other(self)
    }
}

impl std::fmt::Display for UploadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.context, self.source)
    }
}

impl std::error::Error for UploadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

type UploadResult<T> = std::result::Result<T, UploadFailure>;

/// Describes the upload knobs in effect, for inclusion in failure messages.
///
/// Both are process-global and read from the environment, so a failure that is
/// sensitive to either is impossible to interpret without them.
fn upload_settings() -> String {
    format!(
        "LANCE_INITIAL_UPLOAD_SIZE={} bytes, LANCE_UPLOAD_CONCURRENCY={}",
        initial_upload_size(),
        max_upload_parallelism()
    )
}

enum UploadState {
    /// The writer has been opened but no data has been written yet. Will be in
    /// this state until the buffer is full or the writer is shut down.
    Started(Arc<dyn ObjectStore>),
    /// The writer is in the process of creating a multipart upload.
    CreatingUpload(BoxFuture<'static, UploadResult<Box<dyn MultipartUpload>>>),
    /// The writer is in the process of uploading parts.
    InProgress {
        part_idx: u16,
        upload: Box<dyn MultipartUpload>,
        futures: JoinSet<UploadResult<()>>,
    },
    /// The writer is in the process of uploading data in a single PUT request.
    /// This happens when shutdown is called before the buffer is full.
    PuttingSingle(BoxFuture<'static, UploadResult<WriteResult>>),
    /// The writer is in the process of completing the multipart upload.
    Completing(BoxFuture<'static, UploadResult<WriteResult>>),
    /// The writer has been shut down and all data has been written.
    Done(WriteResult),
}

/// Methods for state transitions.
impl UploadState {
    fn started_to_putting_single(&mut self, path: Arc<Path>, buffer: Vec<u8>) {
        // To get owned self, we temporarily swap with Done.
        let this = std::mem::replace(self, Self::Done(WriteResult::default()));
        *self = match this {
            Self::Started(store) => {
                let started_at = Instant::now();
                let fut = async move {
                    let size = buffer.len();
                    let res = store.put(&path, buffer.into()).await.map_err(|source| {
                        UploadFailure::new(
                            format!(
                                "single PUT of {path} failed after {:?} ({size} bytes, {})",
                                started_at.elapsed(),
                                upload_settings()
                            ),
                            source,
                        )
                    })?;
                    Ok(WriteResult {
                        size,
                        e_tag: res.e_tag,
                    })
                };
                Self::PuttingSingle(Box::pin(fut))
            }
            _ => unreachable!(),
        }
    }

    fn in_progress_to_completing(&mut self, path: Arc<Path>, bytes_written: usize) {
        // To get owned self, we temporarily swap with Done.
        let this = std::mem::replace(self, Self::Done(WriteResult::default()));
        *self = match this {
            Self::InProgress {
                mut upload,
                futures,
                part_idx,
            } => {
                debug_assert!(futures.is_empty());
                let started_at = Instant::now();
                let fut = async move {
                    let res = upload.complete().await.map_err(|source| {
                        UploadFailure::new(
                            format!(
                                "completing multipart upload of {path} failed after {:?} \
                                 ({part_idx} parts, {bytes_written} bytes, {})",
                                started_at.elapsed(),
                                upload_settings()
                            ),
                            source,
                        )
                    })?;
                    Ok(WriteResult {
                        size: 0, // This will be set properly later.
                        e_tag: res.e_tag,
                    })
                };
                Self::Completing(Box::pin(fut))
            }
            _ => unreachable!(),
        };
    }
}

impl ObjectWriter {
    pub async fn new(object_store: &LanceObjectStore, path: &Path) -> Result<Self> {
        Ok(Self {
            state: UploadState::Started(object_store.inner.clone()),
            cursor: 0,
            path: Arc::new(path.clone()),
            buffer: Vec::with_capacity(initial_upload_size()),
            use_constant_size_upload_parts: object_store.use_constant_size_upload_parts,
        })
    }

    /// Returns the contents of `buffer` as a `Bytes` object and resets `buffer`.
    /// The new capacity of `buffer` is determined by the current part index.
    fn next_part_buffer(buffer: &mut Vec<u8>, part_idx: u16, constant_upload_size: bool) -> Bytes {
        let new_capacity = if constant_upload_size {
            // The store does not support variable part sizes, so use the initial size.
            initial_upload_size()
        } else {
            // Increase the upload size every 100 parts. This gives maximum part size of 2.5TB.
            initial_upload_size().max(((part_idx / 100) as usize + 1) * INITIAL_UPLOAD_STEP)
        };
        let new_buffer = Vec::with_capacity(new_capacity);
        let part = std::mem::replace(buffer, new_buffer);
        Bytes::from(part)
    }

    fn put_part(
        upload: &mut dyn MultipartUpload,
        buffer: Bytes,
        path: Arc<Path>,
        part_idx: u16,
        parts_in_flight: usize,
    ) -> BoxFuture<'static, UploadResult<()>> {
        let part_size = buffer.len();
        log::debug!("MultipartUpload submitting part with {} bytes", part_size);
        // Stamped before the future is spawned so the reported duration covers
        // any time the task spent waiting to be polled, not just the request.
        let queued_at = Instant::now();
        let fut = upload.put_part(buffer.into());
        Box::pin(async move {
            fut.await.map_err(|source| {
                UploadFailure::new(
                    format!(
                        "multipart upload of part {part_idx} of {path} failed after {:?} \
                         ({part_size} bytes, {parts_in_flight} parts in flight, {})",
                        queued_at.elapsed(),
                        upload_settings()
                    ),
                    source,
                )
            })
        })
    }

    fn poll_tasks(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::result::Result<(), io::Error> {
        let mut_self = &mut *self;
        loop {
            match &mut mut_self.state {
                UploadState::Started(_) | UploadState::Done(_) => break,
                UploadState::CreatingUpload(fut) => match fut.poll_unpin(cx) {
                    Poll::Ready(Ok(mut upload)) => {
                        let mut futures = JoinSet::new();

                        let data = Self::next_part_buffer(
                            &mut mut_self.buffer,
                            0,
                            mut_self.use_constant_size_upload_parts,
                        );
                        futures.spawn(Self::put_part(
                            upload.as_mut(),
                            data,
                            mut_self.path.clone(),
                            0,
                            0,
                        ));

                        mut_self.state = UploadState::InProgress {
                            part_idx: 1, // We just used 0
                            futures,
                            upload,
                        };
                    }
                    Poll::Ready(Err(err)) => return Err(err.into_io_error()),
                    Poll::Pending => break,
                },
                UploadState::InProgress { futures, .. } => {
                    while let Poll::Ready(Some(res)) = futures.poll_join_next(cx) {
                        match res {
                            Ok(Ok(())) => {}
                            Err(err) => {
                                return Err(UploadFailure::new(
                                    format!(
                                        "multipart upload task for {} did not complete",
                                        mut_self.path
                                    ),
                                    err,
                                )
                                .into_io_error());
                            }
                            Ok(Err(err)) => return Err(err.into_io_error()),
                        }
                    }
                    break;
                }
                UploadState::PuttingSingle(fut) | UploadState::Completing(fut) => {
                    match fut.poll_unpin(cx) {
                        Poll::Ready(Ok(mut res)) => {
                            res.size = mut_self.cursor;
                            mut_self.state = UploadState::Done(res)
                        }
                        Poll::Ready(Err(err)) => return Err(err.into_io_error()),
                        Poll::Pending => break,
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn abort(&mut self) {
        let state = std::mem::replace(&mut self.state, UploadState::Done(WriteResult::default()));
        if let UploadState::InProgress { mut upload, .. } = state {
            let _ = upload.abort().await;
        }
    }
}

impl Drop for ObjectWriter {
    fn drop(&mut self) {
        // If there is a multipart upload started but not finished, we should abort it.
        if matches!(self.state, UploadState::InProgress { .. }) {
            // Take ownership of the state.
            let state =
                std::mem::replace(&mut self.state, UploadState::Done(WriteResult::default()));
            if let UploadState::InProgress { mut upload, .. } = state
                && let Ok(handle) = Handle::try_current()
            {
                handle.spawn(async move {
                    let _ = upload.abort().await;
                });
            }
        }
    }
}

impl AsyncWrite for ObjectWriter {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::result::Result<usize, std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        // Fill buffer up to remaining capacity.
        let remaining_capacity = self.buffer.capacity() - self.buffer.len();
        let bytes_to_write = std::cmp::min(remaining_capacity, buf.len());
        self.buffer.extend_from_slice(&buf[..bytes_to_write]);
        self.cursor += bytes_to_write;

        // Rust needs a little help to borrow self mutably and immutably at the same time
        // through a Pin.
        let mut_self = &mut *self;

        // Instantiate next request, if available.
        if mut_self.buffer.capacity() == mut_self.buffer.len() {
            match &mut mut_self.state {
                UploadState::Started(store) => {
                    let path = mut_self.path.clone();
                    let store = store.clone();
                    let started_at = Instant::now();
                    let fut = Box::pin(async move {
                        store.put_multipart(path.as_ref()).await.map_err(|source| {
                            UploadFailure::new(
                                format!(
                                    "failed to create multipart upload for {path} after {:?} ({})",
                                    started_at.elapsed(),
                                    upload_settings()
                                ),
                                source,
                            )
                        })
                    });
                    self.state = UploadState::CreatingUpload(fut);
                }
                // TODO: Make max concurrency configurable from storage options.
                UploadState::InProgress {
                    upload,
                    part_idx,
                    futures,
                    ..
                } if futures.len() < max_upload_parallelism() => {
                    let data = Self::next_part_buffer(
                        &mut mut_self.buffer,
                        *part_idx,
                        mut_self.use_constant_size_upload_parts,
                    );
                    let parts_in_flight = futures.len();
                    futures.spawn(
                        Self::put_part(
                            upload.as_mut(),
                            data,
                            mut_self.path.clone(),
                            *part_idx,
                            parts_in_flight,
                        )
                        .instrument(tracing::Span::current()),
                    );
                    *part_idx += 1;
                }
                _ => {}
            }
        }

        self.poll_tasks(cx)?;

        match bytes_to_write {
            0 => Poll::Pending,
            _ => Poll::Ready(Ok(bytes_to_write)),
        }
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        match &self.state {
            UploadState::Started(_) | UploadState::Done(_) => Poll::Ready(Ok(())),
            UploadState::CreatingUpload(_)
            | UploadState::Completing(_)
            | UploadState::PuttingSingle(_) => Poll::Pending,
            UploadState::InProgress { futures, .. } => {
                if futures.is_empty() {
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Pending
                }
            }
        }
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        loop {
            self.as_mut().poll_tasks(cx)?;

            // Rust needs a little help to borrow self mutably and immutably at the same time
            // through a Pin.
            let mut_self = &mut *self;
            match &mut mut_self.state {
                UploadState::Done(_) => return Poll::Ready(Ok(())),
                UploadState::CreatingUpload(_)
                | UploadState::PuttingSingle(_)
                | UploadState::Completing(_) => return Poll::Pending,
                UploadState::Started(_) => {
                    // If we didn't start a multipart upload, we can just do a single put.
                    let part = std::mem::take(&mut mut_self.buffer);
                    let path = mut_self.path.clone();
                    self.state.started_to_putting_single(path, part);
                }
                UploadState::InProgress {
                    upload,
                    futures,
                    part_idx,
                } => {
                    // Flush final batch
                    if !mut_self.buffer.is_empty() && futures.len() < max_upload_parallelism() {
                        // We can just use `take` since we don't need the buffer anymore.
                        let data = Bytes::from(std::mem::take(&mut mut_self.buffer));
                        let parts_in_flight = futures.len();
                        let final_part_idx = *part_idx;
                        futures.spawn(
                            Self::put_part(
                                upload.as_mut(),
                                data,
                                mut_self.path.clone(),
                                final_part_idx,
                                parts_in_flight,
                            )
                            .instrument(tracing::Span::current()),
                        );
                        // We need to go back to beginning of loop to poll the
                        // new feature and get the waker registered on the ctx.
                        continue;
                    }

                    // We handle the transition from in progress to completing here.
                    if futures.is_empty() {
                        let path = mut_self.path.clone();
                        let bytes_written = mut_self.cursor;
                        self.state.in_progress_to_completing(path, bytes_written);
                    } else {
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Writer for ObjectWriter {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.cursor)
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        AsyncWriteExt::shutdown(self).await.map_err(|e| {
            Error::io(format!(
                "failed to shutdown object writer for {}: {}",
                self.path, e
            ))
        })?;
        if let UploadState::Done(result) = &self.state {
            Ok(result.clone())
        } else {
            unreachable!()
        }
    }
}

pub struct LocalWriter {
    path: Path,
    state: LocalWriteState,
}

#[derive(Default)]
enum LocalWriteState {
    Writing(Box<WritingState>),
    Finishing {
        size: usize,
        future: BoxFuture<'static, Result<WriteResult>>,
    },
    Done(WriteResult),
    #[default]
    Poisoned,
}

struct WritingState {
    writer: tokio::io::BufWriter<tokio::fs::File>,
    cursor: usize,
    /// Temp path that auto-deletes on drop. Set to `None` after `persist()`.
    temp_path: tempfile::TempPath,
    io_tracker: Arc<IOTracker>,
    /// The whole file is reported as a single `put`, so this covers everything
    /// from opening the file to it being durable under its final path. A writer
    /// dropped before `persist()` records nothing, like an aborted upload.
    metrics: IoMetricsGuard,
}

impl LocalWriter {
    pub fn new(
        file: tokio::fs::File,
        path: Path,
        temp_path: tempfile::TempPath,
        io_tracker: Arc<IOTracker>,
    ) -> Self {
        Self {
            path,
            state: LocalWriteState::Writing(Box::new(WritingState {
                writer: tokio::io::BufWriter::new(file),
                cursor: 0,
                temp_path,
                metrics: io_tracker.begin_io("put"),
                io_tracker,
            })),
        }
    }

    fn already_closed_err(path: &Path) -> io::Error {
        io::Error::other(format!(
            "cannot write to LocalWriter for {} after shutdown",
            path
        ))
    }

    fn poisoned_err(path: &Path) -> io::Error {
        io::Error::other(format!("LocalWriter for {} is in poisoned state", path))
    }

    async fn persist(
        temp_path: tempfile::TempPath,
        final_path: Path,
        size: usize,
        io_tracker: Arc<IOTracker>,
        metrics: IoMetricsGuard,
    ) -> Result<WriteResult> {
        let local_path = crate::local::to_local_path(&final_path);
        let persisted = tokio::task::spawn_blocking(move || -> Result<String> {
            temp_path.persist(&local_path).map_err(|e| {
                Error::io(format!(
                    "failed to persist temp file to {}: {}",
                    local_path, e.error
                ))
            })?;

            let metadata = std::fs::metadata(&local_path).map_err(|e| {
                Error::io(format!("failed to read metadata for {}: {}", local_path, e))
            })?;
            Ok(get_etag(&metadata))
        })
        .await
        .map_err(|e| Error::io(format!("spawn_blocking failed: {}", e)))
        .and_then(|e_tag| e_tag);

        metrics.record(&persisted, size as u64);
        let e_tag = persisted?;

        io_tracker.record_write("put", final_path, size as u64);

        Ok(WriteResult {
            size,
            e_tag: Some(e_tag),
        })
    }
}

impl AsyncWrite for LocalWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> Poll<std::result::Result<usize, std::io::Error>> {
        if let LocalWriteState::Writing(state) = &mut self.state {
            let poll = Pin::new(&mut state.writer).poll_write(cx, buf);
            if let Poll::Ready(Ok(n)) = &poll {
                state.cursor += *n;
            }
            poll
        } else {
            Poll::Ready(Err(Self::already_closed_err(&self.path)))
        }
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        if let LocalWriteState::Writing(state) = &mut self.state {
            Pin::new(&mut state.writer).poll_flush(cx)
        } else {
            Poll::Ready(Err(Self::already_closed_err(&self.path)))
        }
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        let mut_self = &mut *self;
        loop {
            match &mut mut_self.state {
                LocalWriteState::Writing(state) => {
                    if Pin::new(&mut state.writer).poll_shutdown(cx).is_pending() {
                        return Poll::Pending;
                    }

                    // Write is complete, we can transition to persisting.
                    let LocalWriteState::Writing(state) =
                        std::mem::replace(&mut mut_self.state, LocalWriteState::Poisoned)
                    else {
                        unreachable!()
                    };
                    let size = state.cursor;
                    mut_self.state = LocalWriteState::Finishing {
                        size,
                        future: Box::pin(Self::persist(
                            state.temp_path,
                            mut_self.path.clone(),
                            size,
                            state.io_tracker,
                            state.metrics,
                        )),
                    };
                }
                LocalWriteState::Finishing { future, .. } => match future.poll_unpin(cx) {
                    Poll::Ready(Ok(result)) => mut_self.state = LocalWriteState::Done(result),
                    Poll::Ready(Err(e)) => {
                        return Poll::Ready(Err(io::Error::other(e)));
                    }
                    Poll::Pending => return Poll::Pending,
                },
                LocalWriteState::Done(_) => return Poll::Ready(Ok(())),
                LocalWriteState::Poisoned => {
                    return Poll::Ready(Err(Self::poisoned_err(&self.path)));
                }
            }
        }
    }
}

#[async_trait]
impl Writer for LocalWriter {
    async fn tell(&mut self) -> Result<usize> {
        match &mut self.state {
            LocalWriteState::Writing(state) => Ok(state.cursor),
            LocalWriteState::Finishing { size, .. } => Ok(*size),
            LocalWriteState::Done(result) => Ok(result.size),
            LocalWriteState::Poisoned => Err(Self::poisoned_err(&self.path).into()),
        }
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        AsyncWriteExt::shutdown(self).await.map_err(|e| {
            Error::io(format!(
                "failed to shutdown local writer for {}: {}",
                self.path, e
            ))
        })?;

        match &self.state {
            LocalWriteState::Done(result) => Ok(result.clone()),
            _ => unreachable!(),
        }
    }
}

// Based on object store's implementation.
pub fn get_etag(metadata: &std::fs::Metadata) -> String {
    let inode = get_inode(metadata);
    let size = metadata.len();
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
        .unwrap_or_default()
        .as_micros();

    // Use an ETag scheme based on that used by many popular HTTP servers
    // <https://httpd.apache.org/docs/2.2/mod/core.html#fileetag>
    format!("{inode:x}-{mtime:x}-{size:x}")
}

#[cfg(unix)]
fn get_inode(metadata: &std::fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(metadata)
}

#[cfg(not(unix))]
fn get_inode(_metadata: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use futures::stream::BoxStream;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions,
        PutOptions, PutPayload, PutResult, RenameOptions, Result as OSResult, UploadPart,
    };
    use tokio::io::AsyncWriteExt;

    use super::*;

    /// Which stage of an upload the mock store rejects.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FailAt {
        CreateMultipart,
        PutPart,
        Complete,
        SinglePut,
    }

    fn rejected(stage: &'static str) -> object_store::Error {
        object_store::Error::Generic {
            store: "FailingUploadStore",
            source: format!("{stage} rejected by test").into(),
        }
    }

    #[derive(Debug)]
    struct FailingUpload {
        fail_at: FailAt,
    }

    #[async_trait]
    impl MultipartUpload for FailingUpload {
        fn put_part(&mut self, _data: PutPayload) -> UploadPart {
            let fails = self.fail_at == FailAt::PutPart;
            Box::pin(async move { if fails { Err(rejected("part")) } else { Ok(()) } })
        }

        async fn complete(&mut self) -> OSResult<PutResult> {
            if self.fail_at == FailAt::Complete {
                Err(rejected("complete"))
            } else {
                Ok(PutResult {
                    e_tag: None,
                    version: None,
                })
            }
        }

        async fn abort(&mut self) -> OSResult<()> {
            Ok(())
        }
    }

    /// Rejects exactly one stage of an upload so each failure site can be
    /// exercised on its own.
    #[derive(Debug)]
    struct FailingUploadStore {
        fail_at: FailAt,
    }

    impl std::fmt::Display for FailingUploadStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FailingUploadStore")
        }
    }

    #[async_trait]
    impl ObjectStore for FailingUploadStore {
        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> OSResult<PutResult> {
            if self.fail_at == FailAt::SinglePut {
                Err(rejected("single put"))
            } else {
                Ok(PutResult {
                    e_tag: None,
                    version: None,
                })
            }
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            if self.fail_at == FailAt::CreateMultipart {
                Err(rejected("create multipart"))
            } else {
                Ok(Box::new(FailingUpload {
                    fail_at: self.fail_at,
                }))
            }
        }

        async fn get_opts(&self, _location: &Path, _options: GetOptions) -> OSResult<GetResult> {
            unimplemented!()
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

    const FAILING_UPLOAD_PATH: &str = "part_7_invert.lance";

    /// Drives a write against a store that rejects `fail_at`, returning the
    /// error. The failure can surface either from a write or from shutdown
    /// depending on when the rejected request is reaped, so both are checked.
    async fn failing_upload(fail_at: FailAt, num_bytes: usize) -> io::Error {
        let mut store = LanceObjectStore::memory();
        store.inner = Arc::new(FailingUploadStore { fail_at });

        let mut writer = ObjectWriter::new(&store, &Path::from(FAILING_UPLOAD_PATH))
            .await
            .unwrap();
        let buf = vec![0u8; num_bytes];
        match writer.write_all(buf.as_slice()).await {
            Err(err) => err,
            Ok(()) => AsyncWriteExt::shutdown(&mut writer)
                .await
                .expect_err("upload should have failed"),
        }
    }

    #[tokio::test]
    async fn test_part_upload_failure_reports_upload_context() {
        // Two parts' worth, so the failing part has a sibling in flight.
        let err = failing_upload(FailAt::PutPart, INITIAL_UPLOAD_STEP * 2).await;
        let message = err.to_string();

        assert!(
            message.contains("multipart upload of part"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(FAILING_UPLOAD_PATH),
            "should name the object: {message}"
        );
        assert!(
            message.contains(&format!("{INITIAL_UPLOAD_STEP} bytes")),
            "should report the part size: {message}"
        );
        assert!(
            message.contains("parts in flight"),
            "should report upload concurrency in use: {message}"
        );
        assert!(
            message.contains("LANCE_INITIAL_UPLOAD_SIZE")
                && message.contains("LANCE_UPLOAD_CONCURRENCY"),
            "should report the knobs governing the request: {message}"
        );
        assert!(
            message.contains("part rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    // The elapsed time is the whole point of the added context: it is what tells
    // a slow request apart from one whose task was never polled.
    #[tokio::test]
    async fn test_part_upload_failure_reports_elapsed_time() {
        let err = failing_upload(FailAt::PutPart, INITIAL_UPLOAD_STEP * 2).await;
        let message = err.to_string();
        assert!(
            message.contains("failed after"),
            "should report how long the request took: {message}"
        );
    }

    #[tokio::test]
    async fn test_part_upload_failure_preserves_source_chain() {
        let err = failing_upload(FailAt::PutPart, INITIAL_UPLOAD_STEP * 2).await;

        let failure = err
            .get_ref()
            .expect("io error should carry the upload failure");
        let source = std::error::Error::source(failure)
            .expect("upload failure should expose the object store error");
        assert!(
            source.downcast_ref::<object_store::Error>().is_some(),
            "source should still be the object store error, got: {source}"
        );
    }

    #[tokio::test]
    async fn test_create_multipart_failure_reports_upload_context() {
        let err = failing_upload(FailAt::CreateMultipart, INITIAL_UPLOAD_STEP * 2).await;
        let message = err.to_string();

        assert!(
            message.contains("failed to create multipart upload for"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(FAILING_UPLOAD_PATH),
            "should name the object: {message}"
        );
        assert!(
            message.contains("create multipart rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    #[tokio::test]
    async fn test_complete_multipart_failure_reports_upload_context() {
        let num_bytes = INITIAL_UPLOAD_STEP * 2;
        let err = failing_upload(FailAt::Complete, num_bytes).await;
        let message = err.to_string();

        assert!(
            message.contains("completing multipart upload of"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(&format!("{num_bytes} bytes")),
            "should report how much had been written: {message}"
        );
        assert!(
            message.contains("complete rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    #[tokio::test]
    async fn test_single_put_failure_reports_upload_context() {
        // Below the multipart threshold, so shutdown takes the single-PUT path.
        let err = failing_upload(FailAt::SinglePut, 256).await;
        let message = err.to_string();

        assert!(
            message.contains("single PUT of"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(FAILING_UPLOAD_PATH),
            "should name the object: {message}"
        );
        assert!(
            message.contains("256 bytes"),
            "should report the body size: {message}"
        );
        assert!(
            message.contains("single put rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    #[tokio::test]
    async fn test_write() {
        let store = LanceObjectStore::memory();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        assert_eq!(object_writer.tell().await.unwrap(), 0);

        let buf = vec![0; 256];
        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 512);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256 * 3);

        let res = Writer::shutdown(&mut object_writer).await.unwrap();
        assert_eq!(res.size, 256 * 3);

        // Trigger multi part upload
        let mut object_writer = ObjectWriter::new(&store, &Path::from("/bar"))
            .await
            .unwrap();
        let buf = vec![0; INITIAL_UPLOAD_STEP / 3 * 2];
        for i in 0..5 {
            // Write more data to trigger the multipart upload
            // This should be enough to trigger a multipart upload
            object_writer.write_all(buf.as_slice()).await.unwrap();
            // Check the cursor
            assert_eq!(object_writer.tell().await.unwrap(), (i + 1) * buf.len());
        }
        let res = Writer::shutdown(&mut object_writer).await.unwrap();
        assert_eq!(res.size, buf.len() * 5);
    }

    #[tokio::test]
    async fn test_abort_write() {
        let store = LanceObjectStore::memory();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        object_writer.abort().await;
    }

    #[tokio::test]
    async fn test_local_writer_shutdown() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("test_local_writer.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let io_tracker = Arc::new(IOTracker::default());

        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, io_tracker.clone());

        let data = b"hello local writer";
        writer.write_all(data).await.unwrap();

        // Before shutdown, the final path should not exist
        assert!(!file_path.exists());
        // But the temp file should exist
        assert!(temp_file_path.exists());

        let result = Writer::shutdown(&mut writer).await.unwrap();
        assert_eq!(result.size, data.len());
        assert!(result.e_tag.is_some());
        assert!(!result.e_tag.as_ref().unwrap().is_empty());

        // After shutdown, the final path should exist and temp should be gone
        assert!(file_path.exists());
        assert!(!temp_file_path.exists());

        let stats = io_tracker.stats();
        assert_eq!(stats.write_iops, 1);
        assert_eq!(stats.written_bytes, data.len() as u64);
    }

    #[tokio::test]
    async fn test_local_writer_drop_cleans_up() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("test_drop.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let io_tracker = Arc::new(IOTracker::default());

        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, io_tracker);

        writer.write_all(b"some data").await.unwrap();
        assert!(temp_file_path.exists());

        // Drop without shutdown should clean up the temp file
        drop(writer);
        assert!(!temp_file_path.exists());
        assert!(!file_path.exists());
    }

    #[test]
    fn clamp_initial_upload_size_below_min_is_clamped_up() {
        assert_eq!(clamp_initial_upload_size(0), (INITIAL_UPLOAD_STEP, true));
        assert_eq!(
            clamp_initial_upload_size(INITIAL_UPLOAD_STEP - 1),
            (INITIAL_UPLOAD_STEP, true)
        );
    }

    #[test]
    fn clamp_initial_upload_size_within_range_is_unchanged() {
        assert_eq!(
            clamp_initial_upload_size(INITIAL_UPLOAD_STEP),
            (INITIAL_UPLOAD_STEP, false)
        );
        assert_eq!(
            clamp_initial_upload_size(MAX_UPLOAD_PART_SIZE),
            (MAX_UPLOAD_PART_SIZE, false)
        );
        let mid = INITIAL_UPLOAD_STEP * 8; // 40MB, in range
        assert_eq!(clamp_initial_upload_size(mid), (mid, false));
    }

    #[test]
    fn clamp_initial_upload_size_above_max_is_clamped_down() {
        assert_eq!(
            clamp_initial_upload_size(MAX_UPLOAD_PART_SIZE + 1),
            (MAX_UPLOAD_PART_SIZE, true)
        );
        assert_eq!(
            clamp_initial_upload_size(usize::MAX),
            (MAX_UPLOAD_PART_SIZE, true)
        );
    }

    /// Regression for the foot-gun where `LANCE_INITIAL_UPLOAD_SIZE=5368709120`
    /// (exactly 5 GiB, Pucheng's setting) caused a single-PUT of 5 GiB on
    /// shutdown — which S3 rejects with `EntityTooLarge`. After tightening
    /// `MAX_UPLOAD_PART_SIZE` to 5 GiB - 1, raw 5 GiB must clamp DOWN.
    #[test]
    fn clamp_initial_upload_size_at_5gib_clamps_down() {
        let exactly_5_gib: usize = 5 * 1024 * 1024 * 1024;
        assert_eq!(
            clamp_initial_upload_size(exactly_5_gib),
            (MAX_UPLOAD_PART_SIZE, true)
        );
    }
}
