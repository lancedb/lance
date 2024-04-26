// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Wrappers around object_store that apply tracing

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartId, ObjectMeta, PutOptions, PutResult,
    Result as OSResult,
};
use pin_project::pin_project;
use tokio::io::AsyncWrite;
use tracing::{debug_span, instrument, Span};

#[pin_project]
pub struct TracedAsyncWrite {
    write_span: Span,
    finish_span: Option<Span>,
    #[pin]
    target: Box<dyn AsyncWrite + Unpin + Send>,
}

impl AsyncWrite for TracedAsyncWrite {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        let this = self.project();
        let _guard = this.write_span.enter();
        this.target.poll_write(cx, buf)
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        let this = self.project();
        let _guard = this.write_span.enter();
        this.target.poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        let this = self.project();
        // TODO: Replace with get_or_insert_with when
        let _guard = this
            .finish_span
            .get_or_insert_with(|| debug_span!("put_multipart_finish"))
            .enter();
        this.target.poll_shutdown(cx)
    }
}

#[derive(Debug)]
pub struct TracedObjectStore {
    target: Arc<dyn object_store::ObjectStore>,
}

impl std::fmt::Display for TracedObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("TracedObjectStore({})", self.target))
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for TracedObjectStore {
    #[instrument(level = "debug", skip(self, bytes))]
    async fn put(&self, location: &Path, bytes: Bytes) -> OSResult<PutResult> {
        self.target.put(location, bytes).await
    }

    #[instrument(level = "debug", skip(self, bytes))]
    async fn put_opts(
        &self,
        location: &Path,
        bytes: Bytes,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart(
        &self,
        location: &Path,
    ) -> OSResult<(MultipartId, Box<dyn AsyncWrite + Unpin + Send>)> {
        let (multipart_id, async_write) = self.target.put_multipart(location).await?;
        Ok((
            multipart_id,
            Box::new(TracedAsyncWrite {
                write_span: debug_span!("put_multipart"),
                finish_span: None,
                target: async_write,
            }) as Box<dyn AsyncWrite + Unpin + Send>,
        ))
    }

    #[instrument(level = "debug", skip(self))]
    async fn abort_multipart(&self, location: &Path, multipart_id: &MultipartId) -> OSResult<()> {
        self.target.abort_multipart(location, multipart_id).await
    }

    #[instrument(level = "debug", skip(self, options))]
    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.target.get_opts(location, options).await
    }

    #[instrument(level = "debug", skip(self))]
    async fn get_range(&self, location: &Path, range: Range<usize>) -> OSResult<Bytes> {
        self.target.get_range(location, range).await
    }

    #[instrument(level = "debug", skip(self, ranges))]
    async fn get_ranges(&self, location: &Path, ranges: &[Range<usize>]) -> OSResult<Vec<Bytes>> {
        self.target.get_ranges(location, ranges).await
    }

    #[instrument(level = "debug", skip(self))]
    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.target.head(location).await
    }

    #[instrument(level = "debug", skip(self))]
    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.target.delete(location).await
    }

    #[instrument(level = "debug", skip_all)]
    fn delete_stream<'a>(
        &'a self,
        locations: BoxStream<'a, OSResult<Path>>,
    ) -> BoxStream<'a, OSResult<Path>> {
        self.target.delete_stream(locations)
    }

    #[instrument(level = "debug", skip(self))]
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'_, OSResult<ObjectMeta>> {
        self.target.list(prefix)
    }

    #[instrument(level = "debug", skip(self))]
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    #[instrument(level = "debug", skip(self))]
    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.target.copy(from, to).await
    }

    #[instrument(level = "debug", skip(self))]
    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.target.rename(from, to).await
    }

    #[instrument(level = "debug", skip(self))]
    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.target.copy_if_not_exists(from, to).await
    }
}

pub trait ObjectStoreTracingExt {
    fn traced(self) -> Arc<dyn object_store::ObjectStore>;
}

impl ObjectStoreTracingExt for Arc<dyn object_store::ObjectStore> {
    fn traced(self) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(TracedObjectStore { target: self })
    }
}

impl<T: object_store::ObjectStore> ObjectStoreTracingExt for Arc<T> {
    fn traced(self) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(TracedObjectStore { target: self })
    }
}
