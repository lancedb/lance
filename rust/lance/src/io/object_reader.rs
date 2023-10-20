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
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use lance_core::io::Reader;
use object_store::{path::Path, ObjectStore};
use prost::Message;
use snafu::{location, Location};

use crate::error::{Error, Result};
use crate::format::ProtoStruct;

pub use lance_core::io::Reader as ObjectReader;

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

/// Read a protobuf message at file position 'pos'.
pub(crate) async fn read_message<M: Message + Default>(
    reader: &dyn Reader,
    pos: usize,
) -> Result<M> {
    let file_size = reader.size().await?;
    if pos > file_size {
        return Err(Error::IO {
            message: "file size is too small".to_string(),
            location: location!(),
        });
    }

    let range = pos..min(pos + 4096, file_size);
    let buf = reader.get_range(range.clone()).await?;
    let msg_len = LittleEndian::read_u32(&buf) as usize;

    if msg_len + 4 > buf.len() {
        let remaining_range = range.end..min(4 + pos + msg_len, file_size);
        let remaining_bytes = reader.get_range(remaining_range).await?;
        let buf = [buf, remaining_bytes].concat();
        assert!(buf.len() >= msg_len + 4);
        Ok(M::decode(&buf[4..4 + msg_len])?)
    } else {
        Ok(M::decode(&buf[4..4 + msg_len])?)
    }
}

/// Read a Protobuf-backed struct at file position: `pos`.
pub(crate) async fn read_struct<
    'm,
    M: Message + Default + 'static,
    T: ProtoStruct<Proto = M> + From<M>,
>(
    reader: &dyn Reader,
    pos: usize,
) -> Result<T> {
    let msg = read_message::<M>(reader, pos).await?;
    let obj = T::from(msg);
    Ok(obj)
}
