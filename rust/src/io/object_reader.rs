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

use arrow_array::{
    types::{BinaryType, LargeBinaryType, LargeUtf8Type, Utf8Type},
    ArrayRef,
};
use arrow_schema::DataType;
use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use object_store::path::Path;
use prost::Message;

use super::ReadBatchParams;
use crate::arrow::*;
use crate::encodings::{binary::BinaryDecoder, plain::PlainDecoder, AsyncIndex, Decoder};
use crate::error::{Error, Result};
use crate::format::ProtoStruct;
use crate::io::ObjectStore;

#[async_trait]
pub trait ObjectReader: Send + Sync {
    /// The file path of the object to read.
    fn path(&self) -> &Path;

    /// Suggest optimal I/O size per storage device.
    fn block_size(&self) -> usize;

    /// Object/File Size.
    async fn size(&self) -> Result<usize>;

    async fn get_range(&self, range: Range<usize>) -> Result<Bytes>;
}

/// Object Reader
///
/// Object Store + Base Path
#[derive(Debug)]
pub struct CloudObjectReader {
    // Object Store.
    // TODO: can we use reference instead?
    pub object_store: ObjectStore,
    // File path
    pub path: Path,

    block_size: usize,
}

impl<'a> CloudObjectReader {
    /// Create an ObjectReader from URI
    pub fn new(object_store: &'a ObjectStore, path: Path, block_size: usize) -> Result<Self> {
        Ok(Self {
            object_store: object_store.clone(),
            path,
            block_size,
        })
    }
}

#[async_trait]
impl ObjectReader for CloudObjectReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    /// Object/File Size.
    async fn size(&self) -> Result<usize> {
        Ok(self.object_store.inner.head(&self.path).await?.size)
    }

    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        Ok(self.object_store.inner.get_range(&self.path, range).await?)
    }
}

/// Read a protobuf message at file position 'pos'.
pub(crate) async fn read_message<M: Message + Default>(
    reader: &dyn ObjectReader,
    pos: usize,
) -> Result<M> {
    let file_size = reader.size().await?;
    if pos > file_size {
        return Err(Error::IO {
            message: "file size is too small".to_string(),
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
    reader: &dyn ObjectReader,
    pos: usize,
) -> Result<T> {
    let msg = read_message::<M>(reader, pos).await?;
    let obj = T::from(msg);
    Ok(obj)
}

/// Read a fixed stride array from disk.
///
pub(crate) async fn read_fixed_stride_array(
    reader: &dyn ObjectReader,
    data_type: &DataType,
    position: usize,
    length: usize,
    params: impl Into<ReadBatchParams>,
) -> Result<ArrayRef> {
    if !data_type.is_fixed_stride() {
        return Err(Error::Schema {
            message: format!("{data_type} is not a fixed stride type"),
        });
    }
    // TODO: support more than plain encoding here.
    let decoder = PlainDecoder::new(reader, data_type, position, length)?;
    decoder.get(params.into()).await
}

pub(crate) async fn read_binary_array(
    reader: &dyn ObjectReader,
    data_type: &DataType,
    nullable: bool,
    position: usize,
    length: usize,
    params: impl Into<ReadBatchParams>,
) -> Result<ArrayRef> {
    use arrow_schema::DataType::*;
    let decoder: Box<dyn Decoder<Output = Result<ArrayRef>> + Send> = match data_type {
        Utf8 => Box::new(BinaryDecoder::<Utf8Type>::new(
            reader, position, length, nullable,
        )),
        Binary => Box::new(BinaryDecoder::<BinaryType>::new(
            reader, position, length, nullable,
        )),
        LargeUtf8 => Box::new(BinaryDecoder::<LargeUtf8Type>::new(
            reader, position, length, nullable,
        )),
        LargeBinary => Box::new(BinaryDecoder::<LargeBinaryType>::new(
            reader, position, length, nullable,
        )),
        _ => {
            return Err(Error::IO {
                message: format!("Unsupported binary type: {data_type}",),
            })
        }
    };
    let fut = decoder.as_ref().get(params.into());
    fut.await
}

#[cfg(test)]
mod tests {}
