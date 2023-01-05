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

use std::cmp::min;
use std::sync::Arc;

use std::io::{Error, ErrorKind, Result};

use byteorder::{ByteOrder, LittleEndian};
use object_store::{path::Path, ObjectMeta, ObjectStore};
use prost::Message;

/// Object Reader
///
/// Object Store + Base Path
pub struct ObjectReader {
    // Index Path
    path: Path,
    // Object Store.
    // TODO: can we use reference instead?
    object_store: Arc<dyn ObjectStore>,
    cached_metadata: Option<ObjectMeta>,
    prefetch_size: usize,
}

impl ObjectReader {
    /// Create an ObjectReader from URI
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        path: Path,
        prefetch_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            object_store,
            path,
            cached_metadata: None,
            prefetch_size,
        })
    }

    /// Read a protobuf message at position `pos`.
    pub async fn read_message<M: Message + Default>(&mut self, pos: usize) -> Result<M> {
        if self.cached_metadata.is_none() {
            self.cached_metadata = Some(self.object_store.head(&self.path).await?);
        };
        let file_size: usize;
        if let Some(metadata) = self.cached_metadata.clone() {
            if pos > metadata.size {
                return Err(Error::new(ErrorKind::InvalidData, "file size is too small"));
            }
            file_size = metadata.size;
        } else {
            panic!("Should not get here");
        }

        let range = pos..min(pos + self.prefetch_size, file_size);
        let buf = self.object_store.get_range(&self.path, range).await?;

        let msg_len = LittleEndian::read_u32(&buf) as usize;
        Ok(M::decode(&buf[4..4 + msg_len])?)
    }
}

#[cfg(test)]
mod tests {}
