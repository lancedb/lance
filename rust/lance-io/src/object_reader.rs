// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::BoxFuture;
use lance_core::Result;
use object_store::{path::Path, ObjectStore};

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
    ) -> Result<O> {
        let mut retries = 3;
        loop {
            match f().await {
                Ok(val) => return Ok(val),
                Err(err) => {
                    if retries == 0 {
                        return Err(err.into());
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
    async fn size(&self) -> Result<usize> {
        let meta = self
            .do_with_retry(|| self.object_store.head(&self.path))
            .await?;
        Ok(meta.size)
    }

    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        self.do_with_retry(|| self.object_store.get_range(&self.path, range.clone()))
            .await
    }
}
