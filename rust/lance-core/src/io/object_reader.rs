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
use object_store::{path::Path, ObjectStore};
use snafu::{location, Location};

use super::Reader;
use crate::error::{Result, Error};

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
        Ok(self.object_store.head(&self.path).await?.size)
    }

    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        // Retries for the initial request are handled by object store, but
        // there are no retries for failures that occur during the streaming
        // of the response body. Thus we add an outer retry loop here.
        let mut retries = 3;
        loop {
            let task = self.object_store.get_range(&self.path, range.clone());
            let res = tokio::time::timeout(std::time::Duration::from_secs(10), task).await;
            match res {
                Ok(Ok(bytes)) => return Ok(bytes),
                Err(_) => {
                    if retries == 0 {
                        return Err(Error::IO { message: "timeout".to_string(), location: location!() });
                    }
                    retries -= 1;
                }
                Ok(Err(err)) => {
                    if retries == 0 {
                        return Err(err.into());
                    }
                    retries -= 1;
                }
            }
        }
    }
}
