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

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use std::sync::Arc;

use arrow_arith::arithmetic::subtract_scalar;
use arrow_array::cast::as_primitive_array;
use arrow_array::UInt32Array;
use arrow_array::{
    make_array, new_empty_array, Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray,
    UInt8Array,
};
use arrow_buffer::{bit_util, Buffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};
use arrow_select::concat::concat;
use arrow_select::take::take;
use async_recursion::async_recursion;
use async_trait::async_trait;
use futures::stream::{self, StreamExt, TryStreamExt};
use tokio::io::AsyncWriteExt;

use super::Decoder;
use crate::arrow::FixedSizeBinaryArrayExt;
use crate::arrow::FixedSizeListArrayExt;
use crate::arrow::*;
use crate::encodings::AsyncIndex;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;
use crate::io::ReadBatchParams;
use crate::Error;

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

    /// Encode primitive values.
    async fn encode_primitive(&mut self, array: &dyn Array) -> Result<usize> {
        let offset = self.writer.tell();
        let data = array.data().buffers()[0].as_slice();
        self.writer.write_all(data).await?;
        Ok(offset)
    }

    /// Encode fixed size list.
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
    reader: &'a dyn ObjectReader,
    data_type: &'a DataType,
    /// The start position of the batch in the file.
    position: usize,
    /// Number of the rows in this batch.
    length: usize,
}

/// Calculate offset in bytes from the row offset.
#[inline]
fn make_byte_offset(data_type: &DataType, row_offset: usize) -> Result<usize> {
    Ok(match data_type {
        DataType::Boolean => bit_util::ceil(row_offset, 8),
        _ => data_type.byte_width() * row_offset,
    })
}

impl<'a> PlainDecoder<'a> {
    pub fn new(
        reader: &'a dyn ObjectReader,
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

    /// Decode primitive values, from "offset" to "offset + length".
    ///
    async fn decode_primitive(&self, start: usize, end: usize) -> Result<ArrayRef> {
        if end > self.length {
            return Err(Error::IO(format!(
                "PlainDecoder: request([{}..{}]) out of range: [0..{}]",
                start, end, self.length
            )));
        }
        let start_offset = make_byte_offset(self.data_type, start)?;
        let end_offset = make_byte_offset(self.data_type, end)?;
        let range = Range {
            start: self.position + start_offset,
            end: self.position + end_offset,
        };

        let data = self.reader.get_range(range).await?;
        let buf: Buffer = data.into();
        let array_data = ArrayDataBuilder::new(self.data_type.clone())
            .len(end - start)
            .null_count(0)
            .add_buffer(buf)
            .build()?;
        Ok(make_array(array_data))
    }

    async fn decode_fixed_size_list(
        &self,
        items: &Field,
        list_size: i32,
        start: usize,
        end: usize,
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
            self.length * list_size as usize,
        )?;
        let item_array = item_decoder
            .get(start * list_size as usize..end * list_size as usize)
            .await?;
        Ok(Arc::new(FixedSizeListArray::try_new(item_array, list_size)?) as ArrayRef)
    }

    async fn decode_fixed_size_binary(
        &self,
        stride: i32,
        start: usize,
        end: usize,
    ) -> Result<ArrayRef> {
        let bytes_decoder = PlainDecoder::new(
            self.reader,
            &DataType::UInt8,
            self.position,
            self.length * stride as usize,
        )?;
        let bytes_array = bytes_decoder
            .get(start * stride as usize..end * stride as usize)
            .await?;
        let values = bytes_array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| {
                Error::Schema("Could not cast to UInt8Array for FixedSizeBinary".to_string())
            })?;
        Ok(Arc::new(FixedSizeBinaryArray::try_new(values, stride)?) as ArrayRef)
    }
}

#[async_trait]
impl<'a> Decoder for PlainDecoder<'a> {
    async fn decode(&self) -> Result<ArrayRef> {
        self.get(0..self.length).await
    }

    async fn take(&self, indices: &UInt32Array) -> Result<ArrayRef> {
        if indices.is_empty() {
            return Ok(new_empty_array(self.data_type));
        }
        let block_size = self.reader.prefetch_size() as u32;
        let byte_width = self.data_type.byte_width() as u32;

        let mut chunk_ranges = vec![];
        let mut start: u32 = 0;
        for j in 0..(indices.len() - 1) as u32 {
            if indices.value(j as usize + 1) * byte_width
                > indices.value(start as usize) * byte_width + block_size
            {
                chunk_ranges.push(start..j + 1);
                start = j + 1;
            }
        }
        // Remaining
        chunk_ranges.push(start..indices.len() as u32);

        let arrays = stream::iter(chunk_ranges)
            .map(|cr| async move {
                let index_chunk = indices.slice(cr.start as usize, cr.len());
                let request: &UInt32Array = as_primitive_array(&index_chunk);

                let start = request.value(0);
                let end = request.value(request.len() - 1);
                let array = self.get(start as usize..end as usize + 1).await?;
                let adjusted_offsets = subtract_scalar(request, start)?;
                Ok::<ArrayRef, Error>(take(&array, &adjusted_offsets, None)?)
            })
            .buffered(8)
            .try_collect::<Vec<_>>()
            .await?;
        let references = arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        Ok(concat(&references)?)
    }
}

#[async_trait]
impl AsyncIndex<usize> for PlainDecoder<'_> {
    // TODO: should this return a Scalar value?
    type Output = Result<ArrayRef>;

    async fn get(&self, index: usize) -> Self::Output {
        self.get(index..index + 1).await
    }
}

#[async_trait]
impl AsyncIndex<Range<usize>> for PlainDecoder<'_> {
    type Output = Result<ArrayRef>;

    async fn get(&self, index: Range<usize>) -> Self::Output {
        if index.is_empty() {
            return Ok(new_empty_array(self.data_type));
        }
        match self.data_type {
            DataType::FixedSizeList(items, list_size) => {
                self.decode_fixed_size_list(items, *list_size, index.start, index.end)
                    .await
            }
            DataType::FixedSizeBinary(stride) => {
                self.decode_fixed_size_binary(*stride, index.start, index.end)
                    .await
            }
            _ => self.decode_primitive(index.start, index.end).await,
        }
    }
}

#[async_trait]
impl AsyncIndex<RangeFrom<usize>> for PlainDecoder<'_> {
    type Output = Result<ArrayRef>;

    async fn get(&self, index: RangeFrom<usize>) -> Self::Output {
        self.get(index.start..self.length).await
    }
}

#[async_trait]
impl AsyncIndex<RangeTo<usize>> for PlainDecoder<'_> {
    type Output = Result<ArrayRef>;

    async fn get(&self, index: RangeTo<usize>) -> Self::Output {
        self.get(0..index.end).await
    }
}

#[async_trait]
impl AsyncIndex<RangeFull> for PlainDecoder<'_> {
    type Output = Result<ArrayRef>;

    async fn get(&self, _: RangeFull) -> Self::Output {
        self.get(0..self.length).await
    }
}

#[async_trait]
impl AsyncIndex<ReadBatchParams> for PlainDecoder<'_> {
    type Output = Result<ArrayRef>;

    async fn get(&self, params: ReadBatchParams) -> Self::Output {
        match params {
            ReadBatchParams::Range(r) => self.get(r).await,
            ReadBatchParams::RangeFull => self.get(..).await,
            ReadBatchParams::RangeTo(r) => self.get(r).await,
            ReadBatchParams::RangeFrom(r) => self.get(r).await,
            ReadBatchParams::Indices(indices) => self.take(&indices).await,
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
        let store = ObjectStore::new(":memory:").await.unwrap();
        let path = Path::from("/foo");
        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        let mut encoder = PlainEncoder::new(&mut object_writer, &data_type);

        assert_eq!(encoder.encode(expected.as_ref()).await.unwrap(), 0);
        object_writer.shutdown().await.unwrap();

        let reader = store.open(&path).await.unwrap();
        assert!(reader.size().await.unwrap() > 0);
        let decoder = PlainDecoder::new(reader.as_ref(), &data_type, 0, expected.len()).unwrap();
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

    #[tokio::test]
    async fn test_decode_by_range() {
        let store = ObjectStore::memory();
        let path = Path::from("/scalar");
        let array = Int32Array::from_iter_values([0, 1, 2, 3, 4, 5]);
        let mut writer = store.create(&path).await.unwrap();
        let mut encoder = PlainEncoder::new(&mut writer, array.data_type());
        assert_eq!(encoder.encode(&array).await.unwrap(), 0);
        writer.shutdown().await.unwrap();

        let reader = store.open(&path).await.unwrap();
        assert!(reader.size().await.unwrap() > 0);
        let decoder =
            PlainDecoder::new(reader.as_ref(), array.data_type(), 0, array.len()).unwrap();
        assert_eq!(
            decoder.get(2..4).await.unwrap().as_ref(),
            &Int32Array::from_iter_values([2, 3])
        );

        assert_eq!(
            decoder.get(..4).await.unwrap().as_ref(),
            &Int32Array::from_iter_values([0, 1, 2, 3])
        );

        assert_eq!(
            decoder.get(2..).await.unwrap().as_ref(),
            &Int32Array::from_iter_values([2, 3, 4, 5])
        );

        assert_eq!(
            &decoder.get(2..2).await.unwrap(),
            &new_empty_array(&DataType::Int32)
        );

        assert_eq!(
            &decoder.get(5..2).await.unwrap(),
            &new_empty_array(&DataType::Int32)
        );

        assert!(decoder.get(3..1000).await.is_err());
    }

    #[tokio::test]
    async fn test_take() {
        let store = ObjectStore::memory();
        let path = Path::from("/takes");
        let array = Int32Array::from_iter_values(0..100);

        let mut writer = store.create(&path).await.unwrap();
        let mut encoder = PlainEncoder::new(&mut writer, array.data_type());
        assert_eq!(encoder.encode(&array).await.unwrap(), 0);
        writer.shutdown().await.unwrap();

        let reader = store.open(&path).await.unwrap();
        assert!(reader.size().await.unwrap() > 0);
        let decoder =
            PlainDecoder::new(reader.as_ref(), array.data_type(), 0, array.len()).unwrap();

        let results = decoder
            .take(&UInt32Array::from_iter(
                [2, 4, 5, 20, 30, 55, 60].iter().map(|i| *i as u32),
            ))
            .await
            .unwrap();
        assert_eq!(
            results.as_ref(),
            &Int32Array::from_iter_values([2, 4, 5, 20, 30, 55, 60])
        );
    }
}
