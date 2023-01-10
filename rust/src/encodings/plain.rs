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
//! Plain encoding works with fixed stride types, i.e., `boolean`, `i8...i64`, `f16...f64`,
//! it stores the array directly in the file. It offers O(1) read access.

use std::any::Any;
use std::cmp::min;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::types::*;
use arrow_array::{
    make_array, Array, ArrayRef, ArrowPrimitiveType, FixedSizeBinaryArray, FixedSizeListArray,
    UInt8Array,
};
use arrow_buffer::{bit_util, Buffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};
use async_recursion::async_recursion;
use async_trait::async_trait;
use tokio::io::AsyncWriteExt;

use crate::arrow::FixedSizeBinaryArrayExt;
use crate::arrow::FixedSizeListArrayExt;
use crate::arrow::*;
use crate::Error;

use super::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;

/// Encoder for plain encoding.
///
pub struct PlainEncoder<'a> {
    writer: &'a mut ObjectWriter,
    data_type: &'a DataType,
}

impl<'a> PlainEncoder<'a> {
    pub fn new(writer: &'a mut ObjectWriter, data_type: &'a DataType) -> PlainEncoder<'a> {
        PlainEncoder { writer, data_type }
    }

    /// Encode an array of a batch.
    /// Returns the offset of the metadata
    pub async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        self.encode_internal(array, self.data_type).await
    }

    #[async_recursion]
    async fn encode_internal(&mut self, array: &dyn Array, data_type: &DataType) -> Result<usize> {
        if let DataType::FixedSizeList(items, _) = data_type {
            self.encode_fixed_size_list(array, items).await
        } else {
            self.encode_primitive(array).await
        }
    }

    async fn encode_primitive(&mut self, array: &dyn Array) -> Result<usize> {
        let offset = self.writer.tell();
        let data = array.data().buffers()[0].as_ref();
        self.writer.write_all(data).await?;
        Ok(offset)
    }

    async fn encode_fixed_size_list(&mut self, array: &dyn Array, items: &Field) -> Result<usize> {
        let list_array = array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                Error::Schema(format!(
                    "Needed a FixedSizeListArray but got {}",
                    array.data_type()
                ))
            })?;
        self.encode_internal(list_array.values().as_ref(), items.data_type())
            .await
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

    async fn decode_primitive(&self) -> Result<ArrayRef> {
        let array_bytes = match self.data_type {
            DataType::Boolean => bit_util::ceil(self.length, 8),
            _ => get_primitive_byte_width(self.data_type)? * self.length,
        };
        let mut start = self.position;
        let end = self.position + array_bytes;
        let mut ranges: Vec<Range<usize>> = vec![];
        while start < end {
            ranges.push(start..min(start + 1 * 1024 * 1024, end));
            start += 1 * 1024 * 1024;
        }

        let data = self.reader.get_ranges(&ranges).await?;
        let buf: Buffer = data.concat().into();
        let array_data = ArrayDataBuilder::new(self.data_type.clone())
            .len(self.length)
            .null_count(0)
            .add_buffer(buf)
            .build()?;
        Ok(make_array(array_data))
    }

    async fn decode_fixed_size_list(
        &self,
        items: &Box<Field>,
        list_size: &i32,
    ) -> Result<ArrayRef> {
        if !items.data_type().is_fixed_stride() {
            return Err(Error::Schema(format!(
                "Items for fixed size list should be primitives but found {}",
                items.data_type()
            )));
        };
        let item_decoder = PlainDecoder::new(
            self.reader,
            items.data_type(),
            self.position,
            self.length * (*list_size) as usize,
        )?;
        let item_array = item_decoder.decode().await?;
        Ok(Arc::new(FixedSizeListArray::try_new(item_array, *list_size)?) as ArrayRef)
    }

    async fn decode_fixed_size_binary(&self, stride: &i32) -> Result<ArrayRef> {
        let bytes_decoder = PlainDecoder::new(
            self.reader,
            &DataType::UInt8,
            self.position,
            self.length * (*stride) as usize,
        )?;
        let bytes_array = bytes_decoder.decode().await?;
        let values = bytes_array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| {
                Error::Schema("Could not cast to UInt8Array for FixedSizeBinary".to_string())
            })?;
        Ok(Arc::new(FixedSizeBinaryArray::try_new(values, *stride)?) as ArrayRef)
    }
}

fn get_primitive_byte_width(data_type: &DataType) -> Result<usize> {
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
        match self.data_type {
            DataType::FixedSizeList(items, list_size) => {
                self.decode_fixed_size_list(items, list_size).await
            }
            DataType::FixedSizeBinary(stride) => self.decode_fixed_size_binary(stride).await,
            _ => self.decode_primitive().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::io::ObjectStore;
    use arrow_array::*;
    use arrow_schema::Field;
    use object_store::path::Path;
    use rand::prelude::*;
    use std::borrow::Borrow;
    use std::sync::Arc;

    use super::*;
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
        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        let mut encoder = PlainEncoder::new(&mut object_writer, &data_type);

        assert_eq!(encoder.encode(expected.as_ref()).await.unwrap(), 0);
        object_writer.shutdown().await.unwrap();

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
            let arr = FixedSizeListArray::try_new(items, 3).unwrap();
            let list_type = DataType::FixedSizeList(Box::new(Field::new("item", t, true)), 3);
            test_round_trip(Arc::new(arr) as ArrayRef, list_type).await;
        }

        let float_types = vec![DataType::Float16, DataType::Float32, DataType::Float64];
        let mut rng = rand::thread_rng();
        let input: Vec<f64> = (1..127).map(|_| rng.gen()).collect();
        for t in float_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let items = make_array_(&t, &buffer).await;
            let arr = FixedSizeListArray::try_new(items, 3).unwrap();
            let list_type = DataType::FixedSizeList(Box::new(Field::new("item", t, true)), 3);
            test_round_trip(Arc::new(arr) as ArrayRef, list_type).await;
        }

        let items = BooleanArray::from(vec![true, false, true].repeat(42));
        let arr = FixedSizeListArray::try_new(items, 3).unwrap();
        let list_type =
            DataType::FixedSizeList(Box::new(Field::new("item", DataType::Boolean, true)), 3);
        test_round_trip(Arc::new(arr) as ArrayRef, list_type).await;
    }

    #[tokio::test]
    async fn test_encode_decode_fixed_size_binary_array() {
        let t = DataType::FixedSizeBinary(3);
        let values = UInt8Array::from(Vec::from_iter(1..127 as u8));
        let arr = FixedSizeBinaryArray::try_new(&values, 3).unwrap();
        test_round_trip(Arc::new(arr) as ArrayRef, t).await;
    }

    #[tokio::test]
    async fn test_encode_decode_nested_fixed_size_list() {
        // FixedSizeList of FixedSizeList
        let inner = DataType::FixedSizeList(Box::new(Field::new("item", DataType::Int64, true)), 2);
        let t = DataType::FixedSizeList(Box::new(Field::new("item", inner, true)), 2);
        let values = Int64Array::from_iter_values(1..=120 as i64);
        let arr = FixedSizeListArray::try_new(FixedSizeListArray::try_new(values, 2).unwrap(), 2)
            .unwrap();
        test_round_trip(Arc::new(arr) as ArrayRef, t).await;

        // FixedSizeList of FixedSizeBinary
        let inner = DataType::FixedSizeBinary(2);
        let t = DataType::FixedSizeList(Box::new(Field::new("item", inner, true)), 2);
        let values = UInt8Array::from_iter_values(1..=120 as u8);
        let arr = FixedSizeListArray::try_new(
            FixedSizeBinaryArray::try_new(&values, 2).unwrap().borrow(),
            2,
        )
        .unwrap();
        test_round_trip(Arc::new(arr) as ArrayRef, t).await;
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
