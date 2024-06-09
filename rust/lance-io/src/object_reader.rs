// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use deepsize::DeepSizeOf;
use lance_core::Result;
use object_store::{path::Path, ObjectStore};
use tokio::sync::OnceCell;
use tracing::instrument;

use crate::{traits::Reader, utils::do_with_retry};

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
        self.size
            .get_or_try_init(|| async move {
                let meta = do_with_retry(|| self.object_store.head(&self.path)).await?;
                Ok(meta.size)
            })
            .await
            .cloned()
    }

    #[instrument(level = "debug", skip(self))]
    async fn get_range(&self, range: Range<usize>) -> object_store::Result<Bytes> {
        do_with_retry(|| self.object_store.get_range(&self.path, range.clone())).await
    }
}
