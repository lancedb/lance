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
    types::{
        BinaryType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, LargeBinaryType, LargeUtf8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
        Utf8Type,
    },
    ArrayRef,
};
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use object_store::{path::Path, ObjectMeta};
use prost::Message;

use crate::encodings::{binary::BinaryDecoder, plain::PlainDecoder, Decoder};
use crate::error::{Error, Result};
use crate::io::ObjectStore;

/// Object Reader
///
/// Object Store + Base Path
#[derive(Debug)]
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
            return Err(Error::IO("file size is too small".to_string()));
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

    /// Read a primitive array from disk.
    ///
    pub async fn read_primitive_array(
        &self,
        data_type: &DataType,
        position: usize,
        length: usize,
    ) -> Result<ArrayRef> {
        assert!(data_type.is_primitive());

        // TODO: support more than plain encoding here.
        use arrow_schema::DataType::*;

        macro_rules! create_plain_decoder {
            ($a:ty) => {
                Box::new(PlainDecoder::<$a>::new(self, position, length)?)
            };
        }

        let decoder: Box<dyn Decoder + Send> = match data_type {
            Int8 => create_plain_decoder!(Int8Type),
            Int16 => create_plain_decoder!(Int16Type),
            Int32 => create_plain_decoder!(Int32Type),
            Int64 => create_plain_decoder!(Int64Type),
            UInt8 => create_plain_decoder!(UInt8Type),
            UInt16 => create_plain_decoder!(UInt16Type),
            UInt32 => create_plain_decoder!(UInt32Type),
            UInt64 => create_plain_decoder!(UInt64Type),
            Float16 => create_plain_decoder!(Float16Type),
            Float32 => create_plain_decoder!(Float32Type),
            Float64 => create_plain_decoder!(Float64Type),
            _ => {
                return Err(Error::Schema(format!(
                    "Unsupport primitive type: {}",
                    data_type
                )))
            }
        };
        let fut = decoder.decode();
        fut.await
    }

    pub async fn read_binary_array(
        &self,
        data_type: &DataType,
        position: usize,
        length: usize,
    ) -> Result<ArrayRef> {
        use arrow_schema::DataType::*;
        let decoder: Box<dyn Decoder + Send> = match data_type {
            Utf8 => Box::new(BinaryDecoder::<Utf8Type>::new(&self, position, length)),
            Binary => Box::new(BinaryDecoder::<BinaryType>::new(&self, position, length)),
            _ => {
                return Err(Error::IO(format!(
                    "Unsupported binary type: {}",
                    data_type,
                )))
            }
        };
        let fut = decoder.decode();
        fut.await
    }

}

#[cfg(test)]
mod tests {}
