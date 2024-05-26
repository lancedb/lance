// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::BoxFuture;
use lance_core::Result;
use object_store::{path::Path, ObjectStore};
use tracing::instrument;

use crate::traits::Reader;

/// Object Reader
///
/// Object Store + Base Path
#[derive(Debug)]
pub struct CloudObjectReader {
    // Object Store.
    pub object_store: Arc<dyn ObjectStore>,
    // File path
    pub path: Path,

    block_size: usize,
}

impl CloudObjectReader {
    /// Create an ObjectReader from URI
    pub fn new(object_store: Arc<dyn ObjectStore>, path: Path, block_size: usize) -> Result<Self> {
        Ok(Self {
            object_store,
            path,
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

    /// Object/File Size.
    async fn size(&self) -> object_store::Result<usize> {
        let meta = self
            .do_with_retry(|| self.object_store.head(&self.path))
            .await?;
        Ok(meta.size)
    }

    #[instrument(level = "debug", skip(self))]
    async fn get_range(&self, range: Range<usize>) -> object_store::Result<Bytes> {
        self.do_with_retry(|| self.object_store.get_range(&self.path, range.clone()))
            .await
    }
}
