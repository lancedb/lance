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

use std::io::{Error, ErrorKind, Result};
use std::ops::Range;

use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use object_store::{path::Path, ObjectMeta};
use prost::Message;

use crate::io::ObjectStore;

/// Object Reader
///
/// Object Store + Base Path
pub struct ObjectReader<'a> {
    // Object Store.
    // TODO: can we use reference instead?
    pub object_store: &'a ObjectStore,
    // File path
    pub path: Path,
    cached_metadata: Option<ObjectMeta>,
    prefetch_size: usize,
}

impl<'a> ObjectReader<'a> {
    /// Create an ObjectReader from URI
    pub fn new(object_store: &'a ObjectStore, path: Path, prefetch_size: usize) -> Result<Self> {
        Ok(Self {
            object_store,
            path,
            cached_metadata: None,
            prefetch_size,
        })
    }

    /// Object/File Size.
    pub async fn size(&mut self) -> Result<usize> {
        if self.cached_metadata.is_none() {
            self.cached_metadata = Some(self.object_store.inner.head(&self.path).await?);
        };
        Ok(self.cached_metadata.as_ref().map_or(0, |m| m.size))
    }

    /// Read a protobuf message at position `pos`.
    pub async fn read_message<M: Message + Default>(&mut self, pos: usize) -> Result<M> {
        let file_size = self.size().await?;
        if pos > file_size {
            return Err(Error::new(ErrorKind::InvalidData, "file size is too small"));
        }

        let range = pos..min(pos + self.prefetch_size, file_size);
        let buf = self.object_store.inner.get_range(&self.path, range).await?;

        let msg_len = LittleEndian::read_u32(&buf) as usize;
        Ok(M::decode(&buf[4..4 + msg_len])?)
    }

    pub async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        let bytes = self.object_store.inner.get_range(&self.path, range).await?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {}
