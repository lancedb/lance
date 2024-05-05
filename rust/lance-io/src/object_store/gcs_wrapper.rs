// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Wrappers around object_store that apply tracing

use std::io;
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::Poll;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesUnordered};
use futures::{FutureExt, StreamExt};
use object_store::gcp::GoogleCloudStorage;
use object_store::multipart::{MultiPartStore, PartId};
use object_store::path::Path;
use object_store::{
    Error as OSError, GetOptions, GetResult, ListResult, MultipartId, ObjectMeta, ObjectStore,
    PutOptions, PutResult, Result as OSResult,
};
use rand::Rng;
use tokio::io::AsyncWrite;

fn max_upload_parallelism() -> usize {
    static MAX_UPLOAD_PARALLELISM: OnceLock<usize> = OnceLock::new();
    *MAX_UPLOAD_PARALLELISM.get_or_init(|| {
        std::env::var("LANCE_UPLOAD_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10)
    })
}

fn max_conn_reset_retries() -> u16 {
    static MAX_CONN_RESET_RETRIES: OnceLock<u16> = OnceLock::new();
    *MAX_CONN_RESET_RETRIES.get_or_init(|| {
        std::env::var("LANCE_CONN_RESET_RETRIES")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(20)
    })
}

/// Wrapper around GoogleCloudStorage with a larger maximum upload size.
///
/// This will be obsolete once object_store 0.10.0 is released.
#[derive(Debug)]
pub struct PatchedGoogleCloudStorage(pub Arc<GoogleCloudStorage>);

impl std::fmt::Display for PatchedGoogleCloudStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PatchedGoogleCloudStorage({})", self.0)
    }
}

#[async_trait]
impl ObjectStore for PatchedGoogleCloudStorage {
    async fn put(&self, location: &Path, bytes: Bytes) -> OSResult<PutResult> {
        self.0.put(location, bytes).await
    }

    async fn put_opts(
        &self,
        location: &Path,
        bytes: Bytes,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.0.put_opts(location, bytes, opts).await
    }

    async fn put_multipart(
        &self,
        location: &Path,
    ) -> OSResult<(MultipartId, Box<dyn AsyncWrite + Unpin + Send>)> {
        // We don't return a real multipart id here. This will be addressed
        // in object_store 0.10.0.
        Upload::new(self.0.clone(), location.clone())
            .map(|upload| (MultipartId::default(), Box::new(upload) as _))
    }

    async fn abort_multipart(&self, _location: &Path, _multipart_id: &MultipartId) -> OSResult<()> {
        // TODO: Once we fix the API above, we can support this.
        return Err(OSError::NotSupported {
            source: "abort_multipart is not supported for Google Cloud Storage".into(),
        });
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.0.get_opts(location, options).await
    }

    async fn get_range(&self, location: &Path, range: Range<usize>) -> OSResult<Bytes> {
        self.0.get_range(location, range).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<usize>]) -> OSResult<Vec<Bytes>> {
        self.0.get_ranges(location, ranges).await
    }

    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.0.head(location).await
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.0.delete(location).await
    }

    fn delete_stream<'a>(
        &'a self,
        locations: BoxStream<'a, OSResult<Path>>,
    ) -> BoxStream<'a, OSResult<Path>> {
        self.0.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'_, OSResult<ObjectMeta>> {
        self.0.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.0.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.0.copy(from, to).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.0.rename(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.0.copy_if_not_exists(from, to).await
    }
}

enum UploadState {
    /// The writer has been opened but no data has been written yet. Will be in
    /// this state until the buffer is full or the writer is shut down.
    Started,
    /// The writer is in the process of creating a multipart upload.
    CreatingUpload(BoxFuture<'static, OSResult<MultipartId>>),
    /// The writer is in the process of uploading parts.
    InProgress {
        multipart_id: Arc<MultipartId>,
        part_idx: u16,
        futures: FuturesUnordered<
            BoxFuture<'static, std::result::Result<(u16, PartId), UploadPutError>>,
        >,
        part_ids: Vec<Option<PartId>>,
    },
    /// The writer is in the process of uploading data in a single PUT request.
    /// This happens when shutdown is called before the buffer is full.
    PuttingSingle(BoxFuture<'static, OSResult<()>>),
    /// The writer is in the process of completing the multipart upload.
    Completing(BoxFuture<'static, OSResult<()>>),
    /// The writer has been shut down and all data has been written.
    Done,
}

/// Start at 5MB.
const INITIAL_UPLOAD_SIZE: usize = 1024 * 1024 * 5;

struct Upload {
    store: Arc<GoogleCloudStorage>,
    path: Arc<Path>,
    buffer: Vec<u8>,
    state: UploadState,
    connection_resets: u16,
}

impl Upload {
    fn new(store: Arc<GoogleCloudStorage>, path: Path) -> OSResult<Self> {
        Ok(Self {
            store,
            path: Arc::new(path),
            buffer: Vec::with_capacity(INITIAL_UPLOAD_SIZE),
            state: UploadState::Started,
            connection_resets: 0,
        })
    }

    /// Returns the contents of `buffer` as a `Bytes` object and resets `buffer`.
    /// The new capacity of `buffer` is determined by the current part index.
    fn next_part_buffer(buffer: &mut Vec<u8>, part_idx: u16) -> Bytes {
        // Increase the upload size every 100 parts. This gives maximum part size of 2.5TB.
        let new_capacity = ((part_idx / 100) as usize + 1) * INITIAL_UPLOAD_SIZE;
        let new_buffer = Vec::with_capacity(new_capacity);
        let part = std::mem::replace(buffer, new_buffer);
        Bytes::from(part)
    }

    fn put_part(
        path: Arc<Path>,
        store: Arc<GoogleCloudStorage>,
        buffer: Bytes,
        part_idx: u16,
        multipart_id: Arc<MultipartId>,
        sleep: Option<std::time::Duration>,
    ) -> BoxFuture<'static, std::result::Result<(u16, PartId), UploadPutError>> {
        Box::pin(async move {
            if let Some(sleep) = sleep {
                tokio::time::sleep(sleep).await;
            }
            let part_id = store
                .put_part(
                    path.as_ref(),
                    multipart_id.as_ref(),
                    part_idx as usize,
                    buffer.clone(),
                )
                .await
                .map_err(|source| UploadPutError {
                    part_idx,
                    buffer,
                    source,
                })?;
            Ok((part_idx, part_id))
        })
    }

    fn poll_tasks(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Result<(), io::Error> {
        let mut_self = &mut *self;
        loop {
            match &mut mut_self.state {
                UploadState::Started | UploadState::Done => break,
                UploadState::CreatingUpload(ref mut fut) => match fut.poll_unpin(cx) {
                    Poll::Ready(Ok(multipart_id)) => {
                        let futures = FuturesUnordered::new();
                        let multipart_id = Arc::new(multipart_id);

                        let data = Self::next_part_buffer(&mut mut_self.buffer, 0);
                        futures.push(Self::put_part(
                            mut_self.path.clone(),
                            mut_self.store.clone(),
                            data,
                            0,
                            multipart_id.clone(),
                            None,
                        ));

                        mut_self.state = UploadState::InProgress {
                            multipart_id,
                            part_idx: 1, // We just used 0
                            futures,
                            part_ids: Vec::new(),
                        };
                    }
                    Poll::Ready(Err(e)) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::Other, e))
                    }
                    Poll::Pending => break,
                },
                UploadState::InProgress {
                    futures,
                    part_ids,
                    multipart_id,
                    ..
                } => {
                    while let Poll::Ready(Some(res)) = futures.poll_next_unpin(cx) {
                        match res {
                            Ok((part_idx, part_id)) => {
                                let total_parts = part_ids.len();
                                part_ids.resize(total_parts.max(part_idx as usize + 1), None);
                                part_ids[part_idx as usize] = Some(part_id);
                            }
                            Err(UploadPutError {
                                source: OSError::Generic { source, .. },
                                part_idx,
                                buffer,
                            }) if source
                                .to_string()
                                .to_lowercase()
                                .contains("connection reset by peer") =>
                            {
                                if mut_self.connection_resets < max_conn_reset_retries() {
                                    // Retry, but only up to max_conn_reset_retries of them.
                                    mut_self.connection_resets += 1;

                                    // Resubmit with random jitter
                                    let sleep_time_ms = rand::thread_rng().gen_range(2_000..8_000);
                                    let sleep_time =
                                        std::time::Duration::from_millis(sleep_time_ms);

                                    futures.push(Self::put_part(
                                        mut_self.path.clone(),
                                        mut_self.store.clone(),
                                        buffer,
                                        part_idx,
                                        multipart_id.clone(),
                                        Some(sleep_time),
                                    ));
                                } else {
                                    return Err(io::Error::new(
                                        io::ErrorKind::ConnectionReset,
                                        Box::new(ConnectionResetError {
                                            message: format!(
                                                "Hit max retries ({}) for connection reset",
                                                max_conn_reset_retries()
                                            ),
                                            source,
                                        }),
                                    ));
                                }
                            }
                            Err(err) => return Err(err.source.into()),
                        }
                    }
                    break;
                }
                UploadState::PuttingSingle(ref mut fut) | UploadState::Completing(ref mut fut) => {
                    match fut.poll_unpin(cx) {
                        Poll::Ready(Ok(())) => mut_self.state = UploadState::Done,
                        Poll::Ready(Err(e)) => {
                            return Err(std::io::Error::new(std::io::ErrorKind::Other, e))
                        }
                        Poll::Pending => break,
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ConnectionResetError {
    message: String,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl std::error::Error for ConnectionResetError {}

impl std::fmt::Display for ConnectionResetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.message, self.source)
    }
}

impl AsyncWrite for Upload {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        // Fill buffer up to remaining capacity.
        let remaining_capacity = self.buffer.capacity() - self.buffer.len();
        let bytes_to_write = std::cmp::min(remaining_capacity, buf.len());
        self.buffer.extend_from_slice(&buf[..bytes_to_write]);

        // Rust needs a little help to borrow self mutably and immutably at the same time
        // through a Pin.
        let mut_self = &mut *self;

        // Instantiate next request, if available.
        if mut_self.buffer.capacity() == mut_self.buffer.len() {
            match &mut mut_self.state {
                UploadState::Started => {
                    let store = self.store.clone();
                    let path = self.path.clone();
                    let fut = Box::pin(async move { store.create_multipart(path.as_ref()).await });
                    self.state = UploadState::CreatingUpload(fut);
                }
                UploadState::InProgress {
                    multipart_id,
                    part_idx,
                    futures,
                    ..
                } => {
                    // TODO: Make max concurrency configurable.
                    if futures.len() < max_upload_parallelism() {
                        let data = Self::next_part_buffer(&mut mut_self.buffer, *part_idx);
                        futures.push(Self::put_part(
                            mut_self.path.clone(),
                            mut_self.store.clone(),
                            data,
                            *part_idx,
                            multipart_id.clone(),
                            None,
                        ));
                        *part_idx += 1;
                    }
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
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        match &self.state {
            UploadState::Started | UploadState::Done => Poll::Ready(Ok(())),
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
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        loop {
            self.as_mut().poll_tasks(cx)?;

            // Rust needs a little help to borrow self mutably and immutably at the same time
            // through a Pin.
            let mut_self = &mut *self;
            match &mut mut_self.state {
                UploadState::Done => return Poll::Ready(Ok(())),
                UploadState::CreatingUpload(_)
                | UploadState::PuttingSingle(_)
                | UploadState::Completing(_) => return Poll::Pending,
                UploadState::Started => {
                    // If we didn't start a multipart upload, we can just do a single put.
                    let part = Bytes::from(std::mem::take(&mut mut_self.buffer));
                    let path = mut_self.path.clone();
                    let store = mut_self.store.clone();
                    let fut = Box::pin(async move {
                        store.put(&path, part).await?;
                        Ok(())
                    });
                    self.state = UploadState::PuttingSingle(fut);
                }
                UploadState::InProgress {
                    futures,
                    part_ids,
                    multipart_id,
                    part_idx,
                } => {
                    // Flush final batch
                    if !mut_self.buffer.is_empty() && futures.len() < max_upload_parallelism() {
                        // We can just use `take` since we don't need the buffer anymore.
                        let data = Bytes::from(std::mem::take(&mut mut_self.buffer));
                        futures.push(Self::put_part(
                            mut_self.path.clone(),
                            mut_self.store.clone(),
                            data,
                            *part_idx,
                            multipart_id.clone(),
                            None,
                        ));
                        // We need to go back to beginning of loop to poll the
                        // new feature and get the waker registered on the ctx.
                        continue;
                    }

                    // We handle the transition from in progress to completing here.
                    if futures.is_empty() {
                        let part_ids = std::mem::take(part_ids)
                            .into_iter()
                            .map(|maybe_id| {
                                maybe_id.ok_or_else(|| {
                                    io::Error::new(io::ErrorKind::Other, "missing part id")
                                })
                            })
                            .collect::<io::Result<Vec<_>>>()?;
                        let path = mut_self.path.clone();
                        let store = mut_self.store.clone();
                        let multipart_id = multipart_id.clone();
                        let fut = Box::pin(async move {
                            store
                                .complete_multipart(&path, &multipart_id, part_ids)
                                .await?;
                            Ok(())
                        });
                        self.state = UploadState::Completing(fut);
                    } else {
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

/// Returned error from trying to upload a part.
/// Has the part_idx and buffer so we can pass
/// them to the retry logic.
struct UploadPutError {
    part_idx: u16,
    buffer: Bytes,
    source: OSError,
}
