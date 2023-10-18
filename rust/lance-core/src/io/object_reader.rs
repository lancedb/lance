// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::{path::Path, ObjectStore};

use super::Reader;
use crate::error::Result;

/// Object Reader
///
/// Object Store + Base Path
#[derive(Debug)]
pub struct CloudObjectReader {
    // Object Store.
    // TODO: can we use reference instead?
    pub object_store: Arc<dyn ObjectStore>,
    // File path
    pub path: Path,

    block_size: usize,
}

impl<'a> CloudObjectReader {
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
        Ok(self.object_store.get_range(&self.path, range).await?)
    }
}
