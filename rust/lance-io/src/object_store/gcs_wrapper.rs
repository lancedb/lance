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
    GetOptions, GetResult, ListResult, MultipartId, ObjectMeta, ObjectStore, PutOptions, PutResult,
    Result as OSResult,
};
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
        futures: FuturesUnordered<BoxFuture<'static, OSResult<(u16, PartId)>>>,
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
}

impl Upload {
    fn new(store: Arc<GoogleCloudStorage>, path: Path) -> OSResult<Self> {
        Ok(Self {
            store,
            path: Arc::new(path),
            buffer: Vec::with_capacity(INITIAL_UPLOAD_SIZE),
            state: UploadState::Pending,
        })
    }

    fn poll_tasks(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Result<(), io::Error> {
        loop {
            match &mut self.state {
                UploadState::Pending | UploadState::Done => break,
                UploadState::CreatingUpload(ref mut fut) => match fut.poll_unpin(cx) {
                    Poll::Ready(Ok(multipart_id)) => {
                        self.state = UploadState::InProgress {
                            multipart_id: Arc::new(multipart_id),
                            part_idx: 0,
                            futures: FuturesUnordered::new(),
                            part_ids: Vec::new(),
                        };
                    }
                    Poll::Ready(Err(e)) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::Other, e))
                    }
                    Poll::Pending => break,
                },
                UploadState::InProgress {
                    futures, part_ids, ..
                } => {
                    while let Poll::Ready(Some(res)) = futures.poll_next_unpin(cx) {
                        let (part_idx, part_id) = res?;
                        let total_parts = part_ids.len();
                        part_ids.resize(total_parts.max(part_idx as usize + 1), None);
                        part_ids[part_idx as usize] = Some(part_id);
                    }
                }
                UploadState::PuttingSingle(ref mut fut) | UploadState::Completing(ref mut fut) => {
                    match fut.poll_unpin(cx) {
                        Poll::Ready(Ok(())) => self.state = UploadState::Done,
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
        let state_ref = &mut mut_self.state;
        let buffer_ref = &mut mut_self.buffer;
        let store_ref = &mut_self.store;
        let path_ref = &mut_self.path;

        // Instantiate next request, if available.
        if buffer_ref.capacity() == buffer_ref.len() {
            match state_ref {
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
                    if buffer_ref.len() >= buffer_ref.capacity() && futures.len() < 10 {
                        // Increase the upload size every 100 parts. This gives maximum part size of 2.5TB.
                        let new_capacity = ((*part_idx / 100) as usize + 1) * INITIAL_UPLOAD_SIZE;
                        let new_buffer = Vec::with_capacity(new_capacity);
                        let part = std::mem::replace(buffer_ref, new_buffer);
                        let part = Bytes::from(part);

                        let part_idx_clone = *part_idx;
                        let store = store_ref.clone();
                        let multipart_id = multipart_id.clone();
                        let path = path_ref.clone();
                        let fut = Box::pin(async move {
                            let part_id = store
                                .put_part(
                                    path.as_ref(),
                                    multipart_id.as_ref(),
                                    part_idx_clone as usize,
                                    part,
                                )
                                .await?;
                            Ok((part_idx_clone, part_id))
                        });
                        futures.push(fut);
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
        self.as_mut().poll_tasks(cx)?;

        // Rust needs a little help to borrow self mutably and immutably at the same time
        // through a Pin.
        let mut_self = &mut *self;
        let state_ref = &mut mut_self.state;
        let buffer_ref = &mut mut_self.buffer;
        let store_ref = &mut_self.store;
        let path_ref = &mut_self.path;

        match state_ref {
            UploadState::Done => Poll::Ready(Ok(())),
            UploadState::CreatingUpload(_)
            | UploadState::PuttingSingle(_)
            | UploadState::Completing(_) => Poll::Pending,
            UploadState::Pending => {
                // If we didn't start a multipart upload, we can just do a single put.
                let part = Bytes::from(std::mem::take(buffer_ref));
                let path = path_ref.clone();
                let store = store_ref.clone();
                let fut = Box::pin(async move {
                    store.put(&path, part).await?;
                    Ok(())
                });
                self.state = UploadState::PuttingSingle(fut);
                self.as_mut().poll_tasks(cx)?;
                // Just in case the put immediately finishes, we recurse here.
                self.poll_shutdown(cx)
            }
            UploadState::InProgress {
                futures,
                part_ids,
                multipart_id,
                ..
            } => {
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
                    let path = path_ref.clone();
                    let store = store_ref.clone();
                    let multipart_id = multipart_id.clone();
                    let fut = Box::pin(async move {
                        store
                            .complete_multipart(&path, &multipart_id, part_ids)
                            .await?;
                        Ok(())
                    });
                    self.state = UploadState::Completing(fut);
                    self.as_mut().poll_tasks(cx)?;
                    // Just in case the completion immediately finishes, we recurse here.
                    self.poll_shutdown(cx)
                } else {
                    Poll::Pending
                }
            }
        }
    }
}
