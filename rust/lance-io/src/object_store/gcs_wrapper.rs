// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Wrappers around object_store that apply tracing

use std::io;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
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

    async fn abort_multipart(&self, location: &Path, multipart_id: &MultipartId) -> OSResult<()> {
        MultiPartStore::abort_multipart(self.0.as_ref(), location, multipart_id).await
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
    Pending,
    CreatingUpload(BoxFuture<'static, OSResult<MultipartId>>),
    InProgress {
        multipart_id: Arc<MultipartId>,
        part_idx: u16,
        futures: FuturesUnordered<
            BoxFuture<'static, std::result::Result<(u16, PartId), UploadPutError>>,
        >,
        part_ids: Vec<Option<PartId>>,
    },
    PuttingSingle(BoxFuture<'static, OSResult<()>>),
    Completing(BoxFuture<'static, OSResult<()>>),
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
            state: UploadState::Pending,
            connection_resets: 0,
        })
    }

    fn put_next_part(
        path: &Arc<Path>,
        store: &Arc<GoogleCloudStorage>,
        buffer: &mut Vec<u8>,
        part_idx: &mut u16,
        multipart_id: &Arc<MultipartId>,
    ) -> BoxFuture<'static, std::result::Result<(u16, PartId), UploadPutError>> {
        // Increase the upload size every 100 parts. This gives maximum part size of 2.5TB.
        let new_capacity = ((*part_idx / 100) as usize + 1) * INITIAL_UPLOAD_SIZE;
        let new_buffer = Vec::with_capacity(new_capacity);
        let part = std::mem::replace(buffer, new_buffer);
        let part = Bytes::from(part);

        let part_idx_clone = *part_idx;
        *part_idx += 1;
        let store = store.clone();
        let multipart_id = multipart_id.clone();
        let path = path.clone();
        Self::put_part(
            path.clone(),
            store.clone(),
            part,
            part_idx_clone,
            multipart_id.clone(),
            None,
        )
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
                UploadState::Pending | UploadState::Done => break,
                UploadState::CreatingUpload(ref mut fut) => match fut.poll_unpin(cx) {
                    Poll::Ready(Ok(multipart_id)) => {
                        let futures = FuturesUnordered::new();
                        let multipart_id = Arc::new(multipart_id);

                        futures.push(Self::put_next_part(
                            &mut_self.path,
                            &mut_self.store,
                            &mut mut_self.buffer,
                            &mut 0,
                            &multipart_id,
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
                            }) if source.to_string().contains("Connection reset by peer")
                                && mut_self.connection_resets < 20 =>
                            {
                                // Retry, but only up to 20 of them.
                                mut_self.connection_resets += 1;

                                // Resubmit with random jitter
                                let sleep_time_ms = rand::thread_rng().gen_range(2_000..8_000);
                                let sleep_time = std::time::Duration::from_millis(sleep_time_ms);

                                futures.push(Self::put_part(
                                    mut_self.path.clone(),
                                    mut_self.store.clone(),
                                    buffer,
                                    part_idx,
                                    multipart_id.clone(),
                                    Some(sleep_time),
                                ));
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
                UploadState::Pending => {
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
                    if futures.len() < 10 {
                        futures.push(Self::put_next_part(
                            &mut_self.path,
                            &mut_self.store,
                            &mut mut_self.buffer,
                            part_idx,
                            multipart_id,
                        ));
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
            UploadState::Pending | UploadState::Done => Poll::Ready(Ok(())),
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
        // Rust needs a little help to borrow self mutably and immutably at the same time
        // through a Pin.

        loop {
            self.as_mut().poll_tasks(cx)?;

            let mut_self = &mut *self;
            match &mut mut_self.state {
                UploadState::Done => return Poll::Ready(Ok(())),
                UploadState::CreatingUpload(_)
                | UploadState::PuttingSingle(_)
                | UploadState::Completing(_) => return Poll::Pending,
                UploadState::Pending => {
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
                    if !mut_self.buffer.is_empty() && futures.len() < 10 {
                        futures.push(Self::put_next_part(
                            &mut_self.path,
                            &mut_self.store,
                            &mut mut_self.buffer,
                            part_idx,
                            multipart_id,
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
/// Has the part_idx and buffer so we can
struct UploadPutError {
    part_idx: u16,
    buffer: Bytes,
    source: OSError,
}
