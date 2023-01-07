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

//! Plain encoding
//!
//! Plain encoding works with primitive types, i.e., `boolean`, `i8...i64`,
//! it stores the array directly in the file. It offers O(1) read access.

use std::any::Any;
use std::ops::Range;

use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, make_array, PrimitiveArray};
use arrow_array::types::*;
use arrow_buffer::{bit_util, Buffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use crate::Error;

use super::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;

/// Encoder for plain encoding.
///
pub struct PlainEncoder<'a> {
    writer: &'a mut ObjectWriter<'a>,
    data_type: &'a DataType
}

impl<'a> PlainEncoder<'a> {
    pub fn new(writer: &'a mut ObjectWriter<'a>, data_type: &'a DataType) -> PlainEncoder<'a> {
        PlainEncoder {
            writer,
            data_type
        }
    }

    /// Encode an array of a batch.
    /// Returns the offset of the metadata
    pub async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        let offset = self.writer.tell() as usize;

        let data = array.data().buffers()[0].as_slice();
        self.writer.write_all(data).await?;

        Ok(offset)
    }
}

/// Decoder for plain encoding.
pub struct PlainDecoder<'a> {
    reader: &'a ObjectReader<'a>,
    data_type: &'a DataType,
    /// The start position of the batch in the file.
    position: usize,
    /// Number of the rows in this batch.
    length: usize
}


impl<'a> PlainDecoder<'a> {
    pub fn new(
        reader: &'a ObjectReader,
        data_type: &'a DataType,
        position: usize,
        length: usize
    ) -> Result<PlainDecoder<'a>> {
        Ok(PlainDecoder {
            reader,
            data_type,
            position,
            length
        })
    }

    pub async fn at(&self, _idx: usize) -> Result<Option<Box<dyn Any>>> {
        todo!()
    }

    fn get_byte_width(&self) -> Result<usize> {
        match self.data_type {
            DataType::Int8 => Ok(Int8Type::get_byte_width()),
            DataType::Int16 => Ok(Int16Type::get_byte_width()),
            DataType::Int32 => Ok(Int32Type::get_byte_width()),
            DataType::Int64 => Ok(Int64Type::get_byte_width()),
            DataType::UInt8 => Ok(UInt8Type::get_byte_width()),
            DataType::UInt16 => Ok(UInt16Type::get_byte_width()),
            DataType::UInt32 => Ok(UInt32Type::get_byte_width()),
            DataType::UInt64 => Ok(UInt64Type::get_byte_width()),
            DataType::Float16 => Ok(Float16Type::get_byte_width()),
            DataType::Float32 => Ok(Float32Type::get_byte_width()),
            DataType::Float64 => Ok(Float64Type::get_byte_width()),
            _ => Err(Error::Schema(format!("Unsupport primitive type: {}", self.data_type)))
        }
    }
}

#[async_trait]
impl<'a> Decoder for PlainDecoder<'a> {
    async fn decode(&self) -> Result<ArrayRef> {
        let array_bytes = match self.data_type {
            DataType::Boolean => bit_util::ceil(self.length, 8),
            _ => self.get_byte_width()? * self.length
        };
        let range = Range {
            start: self.position,
            end: self.position + array_bytes,
        };

        let data = self.reader.get_range(range).await?;
        let buf: Buffer = data.into();
        let array_data = ArrayDataBuilder::new(self.data_type.clone())
            .len(self.length)
            .null_count(0)
            .add_buffer(buf)
            .build()?;
        Ok(make_array(array_data))
    }
}


#[cfg(test)]
mod tests {
    use crate::io::ObjectStore;
    use arrow_array::cast::{as_boolean_array, as_primitive_array};
    use arrow_array::types::Int32Type;
    use arrow_array::{BooleanArray, Int32Array};
    use object_store::path::Path;
    use tokio::io::AsyncWriteExt;

    use crate::io::object_writer::ObjectWriter;

    use super::*;

    #[tokio::test]
    async fn test_encode_decode_int_array() {
        let store = ObjectStore::new(":memory:").unwrap();
        let path = Path::from("/foo");
        let (_, mut writer) = store.inner.put_multipart(&path).await.unwrap();

        let arr = Int32Array::from(Vec::from_iter(1..4096));
        {
            let mut object_writer = ObjectWriter::new(writer.as_mut());
            let mut encoder = PlainEncoder::new(&mut object_writer, &DataType::Int32);

            assert_eq!(encoder.encode(&arr).await.unwrap(), 0);
        }
        writer.shutdown().await.unwrap();

        let mut reader = store.open(&path).await.unwrap();
        assert!(reader.size().await.unwrap() > 0);
        let decoder = PlainDecoder::new(&reader, &DataType::Int32, 0, arr.len()).unwrap();
        let read_arr = decoder.decode().await.unwrap();
        let expect_arr = as_primitive_array::<Int32Type>(read_arr.as_ref());
        assert_eq!(expect_arr, &arr);
    }

    #[tokio::test]
    async fn test_encode_decode_bool_array() {
        let store = ObjectStore::new(":memory:").unwrap();
        let path = Path::from("/foo");
        let (_, mut writer) = store.inner.put_multipart(&path).await.unwrap();

        let arr = BooleanArray::from(vec![true, false].repeat(100));
        {
            let mut object_writer = ObjectWriter::new(writer.as_mut());
            let mut encoder = PlainEncoder::new(&mut object_writer, &DataType::Boolean);

            assert_eq!(encoder.encode(&arr).await.unwrap(), 0);
        }
        writer.shutdown().await.unwrap();

        let mut reader = store.open(&path).await.unwrap();
        assert!(reader.size().await.unwrap() > 0);
        let decoder = PlainDecoder::new(&reader, &DataType::Boolean, 0, arr.len()).unwrap();
        let read_arr = decoder.decode().await.unwrap();
        let expect_arr = as_boolean_array(read_arr.as_ref());
        assert_eq!(expect_arr, &arr);
    }
}
