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
use std::sync::Arc;

use arrow_array::types::*;
use arrow_array::{make_array, Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray};
use arrow_buffer::{bit_util, Buffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;

use crate::Error;
use async_trait::async_trait;
use tokio::io::AsyncWriteExt;

use super::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;

/// Encoder for plain encoding.
///
pub struct PlainEncoder<'a> {
    writer: &'a mut ObjectWriter<'a>,
    data_type: &'a DataType,
}

impl<'a> PlainEncoder<'a> {
    pub fn new(writer: &'a mut ObjectWriter<'a>, data_type: &'a DataType) -> PlainEncoder<'a> {
        PlainEncoder { writer, data_type }
    }

    /// Encode an array of a batch.
    /// Returns the offset of the metadata
    pub async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        let offset = self.writer.tell() as usize;

        let data = match array.data_type() {
            DataType::FixedSizeList(_, _) => array.data().child_data()[0].buffers()[0].as_ref(),
            _ => array.data().buffers()[0].as_ref(),
        };
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
    length: usize,
}

impl<'a> PlainDecoder<'a> {
    pub fn new(
        reader: &'a ObjectReader,
        data_type: &'a DataType,
        position: usize,
        length: usize,
    ) -> Result<PlainDecoder<'a>> {
        Ok(PlainDecoder {
            reader,
            data_type,
            position,
            length,
        })
    }

    pub async fn at(&self, _idx: usize) -> Result<Option<Box<dyn Any>>> {
        todo!()
    }
}

fn get_byte_width(data_type: &DataType) -> Result<usize> {
    match data_type {
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
        _ => Err(Error::Schema(format!(
            "Unsupport primitive type: {}",
            data_type
        ))),
    }
}

#[async_trait]
impl<'a> Decoder for PlainDecoder<'a> {
    async fn decode(&self) -> Result<ArrayRef> {
        if let DataType::FixedSizeList(items, list_size) = self.data_type {
            let data_type = items.data_type().clone();
            if !data_type.is_primitive() && data_type != DataType::Boolean {
                return Err(Error::Schema(
                    "Items for fixed size list should be primitives".to_string(),
                ));
            };
            let item_decoder = PlainDecoder::new(
                self.reader,
                items.data_type(),
                self.position,
                self.length * (*list_size) as usize,
            )?;
            let item_array = item_decoder.decode().await?;
            let array_data = ArrayDataBuilder::new(self.data_type.clone())
                .len(self.length)
                .null_count(0)
                .add_child_data(item_array.data().clone())
                .build()?;
            Ok(Arc::new(FixedSizeListArray::from(array_data)) as ArrayRef)
        } else {
            let array_bytes = match self.data_type {
                DataType::Boolean => bit_util::ceil(self.length, 8),
                _ => get_byte_width(self.data_type)? * self.length,
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
}

#[cfg(test)]
mod tests {
    use crate::io::ObjectStore;
    use arrow_array::cast::as_boolean_array;
    use arrow_array::*;
    use arrow_schema::Field;
    use object_store::path::Path;
    use rand::prelude::*;
    use std::sync::Arc;
    use tokio::io::AsyncWriteExt;

    use super::*;
    use crate::arrow::*;
    use crate::io::object_writer::ObjectWriter;

    #[tokio::test]
    async fn test_encode_decode_primitive_array() {
        let int_types = vec![
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ];
        let input: Vec<i64> = Vec::from_iter(1..127 as i64);
        for t in int_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let arr = make_array_(&t, &buffer).await;
            test_round_trip(Arc::new(arr) as ArrayRef, t).await;
        }

        let float_types = vec![DataType::Float16, DataType::Float32, DataType::Float64];
        let mut rng = rand::thread_rng();
        let input: Vec<f64> = (1..127).map(|_| rng.gen()).collect();
        for t in float_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let arr = make_array_(&t, &buffer).await;
            test_round_trip(Arc::new(arr) as ArrayRef, t).await;
        }
    }

    async fn test_round_trip(expected: ArrayRef, data_type: DataType) {
        let store = ObjectStore::new(":memory:").unwrap();
        let path = Path::from("/foo");
        let (_, mut writer) = store.inner.put_multipart(&path).await.unwrap();

        {
            let mut object_writer = ObjectWriter::new(writer.as_mut());
            let mut encoder = PlainEncoder::new(&mut object_writer, &data_type);

            assert_eq!(encoder.encode(expected.as_ref()).await.unwrap(), 0);
        }
        writer.shutdown().await.unwrap();

        let mut reader = store.open(&path).await.unwrap();
        assert!(reader.size().await.unwrap() > 0);
        let decoder = PlainDecoder::new(&reader, &data_type, 0, expected.len()).unwrap();
        let arr = decoder.decode().await.unwrap();
        let actual = arr.as_ref();
        assert_eq!(expected.as_ref(), actual);
    }

    #[tokio::test]
    async fn test_encode_decode_bool_array() {
        let arr = BooleanArray::from(vec![true, false].repeat(100));
        test_round_trip(Arc::new(arr) as ArrayRef, DataType::Boolean).await;
    }

    #[tokio::test]
    async fn test_encode_decode_fixed_size_list_array() {
        let int_types = vec![
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ];
        let input = Vec::from_iter(1..127 as i64);
        for t in int_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let items = make_array_(&t, &buffer).await;
            let arr = FixedSizeListArray::new(items, 3).unwrap();
            let list_type = DataType::FixedSizeList(Box::new(Field::new("item", t, true)), 3);
            test_round_trip(Arc::new(arr) as ArrayRef, list_type).await;
        }

        let float_types = vec![DataType::Float16, DataType::Float32, DataType::Float64];
        let mut rng = rand::thread_rng();
        let input: Vec<f64> = (1..127).map(|_| rng.gen()).collect();
        for t in float_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let items = make_array_(&t, &buffer).await;
            let arr = FixedSizeListArray::new(items, 3).unwrap();
            let list_type = DataType::FixedSizeList(Box::new(Field::new("item", t, true)), 3);
            test_round_trip(Arc::new(arr) as ArrayRef, list_type).await;
        }

        let items = BooleanArray::from(vec![true, false, true].repeat(42));
        let arr = FixedSizeListArray::new(items, 3).unwrap();
        let list_type =
            DataType::FixedSizeList(Box::new(Field::new("item", DataType::Boolean, true)), 3);
        test_round_trip(Arc::new(arr) as ArrayRef, list_type).await;
    }

    async fn make_array_(data_type: &DataType, buffer: &Buffer) -> ArrayRef {
        make_array(
            ArrayDataBuilder::new(data_type.clone())
                .len(126)
                .add_buffer(buffer.clone())
                .build()
                .unwrap(),
        )
    }
}
