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

use std::io::{Error, ErrorKind, Result};

use byteorder::{ByteOrder, LittleEndian};
use object_store::{aws::AmazonS3, path::Path, ObjectMeta, ObjectStore};
use prost::Message;

const PREFETCH_SIZE_CLOUD_STORAGE: usize = 64 * 1024;
const PREFETCH_SIZE_SSD: usize = 4 * 1024;

/// Object Reader
///
/// Object Store + Base Path
pub struct ObjectReader<'a, S: ObjectStore> {
    // Index Path
    path: Path,
    // Object Store
    object_store: &'a S,
    cached_metadata: Option<ObjectMeta>,
    prefetch_size: usize,
}

impl<'a, S: ObjectStore> ObjectReader<'a, S> {
    /// Create an ObjectReader from URI
    pub fn new(object_store: &'a S, path: Path) -> Result<Self> {
        Ok(Self {
            object_store,
            path,
            cached_metadata: None,
            prefetch_size: match object_store {
                AmazonS3 => PREFETCH_SIZE_CLOUD_STORAGE,
                _ => PREFETCH_SIZE_SSD,
            }
        })
    }

    /// Read a protobuf message at position.
    pub async fn read_message<M: Message + Default>(&mut self, pos: u64) -> Result<M> {
        if self.cached_metadata.is_none() {
            self.cached_metadata = Some(self.object_store.head(&self.path).await?);
        };
        if let Some(metadata) = self.cached_metadata.clone() {
            if pos as usize > metadata.size {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "access position beyond file size",
                ));
            }
        } else {
            panic!("Should not get here");
        }

        let range = pos as usize..(pos as usize + self.prefetch_size);
        let buf = self.object_store.get_range(&self.path, range).await?;

        let msg_len = LittleEndian::read_u64(&buf) as usize;
        Ok(M::decode(&buf[8..8 + msg_len])?)
    }
}
