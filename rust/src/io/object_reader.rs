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
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use object_store::{path::Path, ObjectMeta};
use prost::Message;

use crate::arrow::*;
use crate::encodings::{binary::BinaryDecoder, plain::PlainDecoder, Decoder};
use crate::error::{Error, Result};
use crate::format::ProtoStruct;
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

    /// Read a Protobuf-backed struct at file position: `pos`.
    pub async fn read_struct<
        'm,
        M: Message + Default + 'static,
        T: ProtoStruct<Proto = M> + From<M>,
    >(
        &mut self,
        pos: usize,
    ) -> Result<T> {
        let msg = self.read_message::<M>(pos).await?;
        let obj = T::from(msg);
        Ok(obj)
    }

    /// Read a protobuf message at position `pos`.
    pub async fn read_message<M: Message + Default>(&mut self, pos: usize) -> Result<M> {
        let file_size = self.size().await?;
        if pos > file_size {
            return Err(Error::IO("file size is too small".to_string()));
        }

        let range = pos..min(pos + self.prefetch_size, file_size);
        let buf = self
            .object_store
            .inner
            .get_range(&self.path, range.clone())
            .await?;
        let msg_len = LittleEndian::read_u32(&buf) as usize;

        if msg_len + 4 > buf.len() {
            let remaining_range = range.end..min(4 + pos + msg_len, file_size);
            let remaining_bytes = self
                .object_store
                .inner
                .get_range(&self.path, remaining_range)
                .await?;
            let buf = [buf, remaining_bytes].concat();
            assert!(buf.len() >= msg_len + 4);
            Ok(M::decode(&buf[4..4 + msg_len])?)
        } else {
            Ok(M::decode(&buf[4..4 + msg_len])?)
        }
    }

    pub async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        let bytes = self.object_store.inner.get_range(&self.path, range).await?;
        Ok(bytes)
    }

    /// Read a fixed stride array from disk.
    ///
    pub async fn read_fixed_stride_array(
        &self,
        data_type: &DataType,
        position: usize,
        length: usize,
    ) -> Result<ArrayRef> {
        if !data_type.is_fixed_stride() {
            return Err(Error::Schema(format!(
                "{} is not a fixed stride type",
                data_type
            )));
        }
        // TODO: support more than plain encoding here.
        let decoder = PlainDecoder::new(self, data_type, position, length)?;
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
            Utf8 => Box::new(BinaryDecoder::<Utf8Type>::new(self, position, length)),
            Binary => Box::new(BinaryDecoder::<BinaryType>::new(self, position, length)),
            LargeUtf8 => Box::new(BinaryDecoder::<LargeUtf8Type>::new(self, position, length)),
            LargeBinary => Box::new(BinaryDecoder::<LargeBinaryType>::new(
                self, position, length,
            )),
            _ => {
                return Err(Error::IO(
                    format!("Unsupported binary type: {}", data_type,),
                ))
            }
        };
        let fut = decoder.decode();
        fut.await
    }
}

#[cfg(test)]
mod tests {}
