// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use deepsize::DeepSizeOf;
use futures::future::BoxFuture;
use lance_core::Result;
use object_store::{path::Path, ObjectStore};
use tokio::sync::OnceCell;
use tracing::instrument;

use crate::{object_store::DEFAULT_CLOUD_IO_PARALLELISM, traits::Reader};

/// Object Reader
///
/// Object Store + Base Path
#[derive(Debug)]
pub struct CloudObjectReader {
    // Object Store.
    pub object_store: Arc<dyn ObjectStore>,
    // File path
    pub path: Path,
    // File size, if known.
    size: OnceCell<usize>,

    block_size: usize,
}

impl DeepSizeOf for CloudObjectReader {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // Skipping object_store because there is no easy way to do that and it shouldn't be too big
        self.path.as_ref().deep_size_of_children(context)
    }
}

impl CloudObjectReader {
    /// Create an ObjectReader from URI
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        path: Path,
        block_size: usize,
        known_size: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            object_store,
            path,
            size: OnceCell::new_with(known_size),
            block_size,
        })
    }

    // Retries for the initial request are handled by object store, but
    // there are no retries for failures that occur during the streaming
    // of the response body. Thus we add an outer retry loop here.
    async fn do_with_retry<'a, O>(
        &self,
        f: impl Fn() -> BoxFuture<'a, std::result::Result<O, object_store::Error>>,
    ) -> object_store::Result<O> {
        let mut retries = 3;
        loop {
            match f().await {
                Ok(val) => return Ok(val),
                Err(err) => {
                    if retries == 0 {
                        return Err(err);
                    }
                    retries -= 1;
                }
            }
        }
    }
}

#[async_trait]
impl Reader for CloudObjectReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn io_parallelism(&self) -> usize {
        DEFAULT_CLOUD_IO_PARALLELISM
    }

    /// Object/File Size.
    async fn size(&self) -> object_store::Result<usize> {
        self.size
            .get_or_try_init(|| async move {
                let meta = self
                    .do_with_retry(|| self.object_store.head(&self.path))
                    .await?;
                Ok(meta.size)
            })
            .await
            .cloned()
    }

    #[instrument(level = "debug", skip(self))]
    async fn get_range(&self, range: Range<usize>) -> object_store::Result<Bytes> {
        self.do_with_retry(|| self.object_store.get_range(&self.path, range.clone()))
            .await
    }
}
