// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Plain encoding
//!
//! Plain encoding works with fixed stride types, i.e., `boolean`, `i8...i64`, `f16...f64`,
//! it stores the array directly in the file. It offers O(1) read access.

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use std::ptr::NonNull;
use std::slice::from_raw_parts;
use std::sync::Arc;

use arrow::array::{as_boolean_array, BooleanBuilder};
use arrow_arith::numeric::sub;
use arrow_array::cast::as_primitive_array;
use arrow_array::{
    make_array, new_empty_array, Array, ArrayRef, BooleanArray, FixedSizeBinaryArray,
    FixedSizeListArray, UInt32Array, UInt8Array,
};
use arrow_buffer::{bit_util, Buffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};
use arrow_select::concat::concat;
use arrow_select::take::take;
use async_recursion::async_recursion;
use async_trait::async_trait;
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::io::Writer;
use snafu::{location, Location};
use tokio::io::AsyncWriteExt;

use crate::arrow::*;
use crate::encodings::AsyncIndex;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::ReadBatchParams;
use crate::Error;

use super::Decoder;

/// Parallelism factor decides how many run parallel I/O issued per CPU core.
/// This is a heuristic value, with the assumption NVME and S3/GCS can
/// handles large mount of parallel I/O & large disk-queue.
const PARALLELISM_FACTOR: usize = 4;

/// Encoder for plain encoding.
///
pub struct PlainEncoder<'a> {
    writer: &'a mut dyn Writer,
    data_type: &'a DataType,
}

impl<'a> PlainEncoder<'a> {
    pub fn new(writer: &'a mut dyn Writer, data_type: &'a DataType) -> PlainEncoder<'a> {
        PlainEncoder { writer, data_type }
    }

    /// Write an continuous plain-encoded array to the writer.
    pub async fn write(writer: &'a mut dyn Writer, arrays: &[&'a dyn Array]) -> Result<usize> {
        let pos = writer.tell();
        if !arrays.is_empty() {
            let mut encoder = Self::new(writer, arrays[0].data_type());
            encoder.encode(arrays).await?;
        }
        Ok(pos)
    }

    /// Encode an slice of an Array of a batch.
    /// Returns the offset of the metadata
    pub async fn encode(&mut self, arrays: &[&dyn Array]) -> Result<usize> {
        self.encode_internal(arrays, self.data_type).await
    }

    #[async_recursion]
    async fn encode_internal(
        &mut self,
        array: &[&dyn Array],
        data_type: &DataType,
    ) -> Result<usize> {
        if let DataType::FixedSizeList(items, _) = data_type {
            self.encode_fixed_size_list(array, items).await
        } else {
            self.encode_primitive(array).await
        }
    }

    async fn encode_boolean(&mut self, arrays: &[&BooleanArray]) -> Result<()> {
        let capacity: usize = arrays.iter().map(|a| a.len()).sum();
        let mut builder = BooleanBuilder::with_capacity(capacity);

        for array in arrays {
            for val in array.iter() {
                builder.append_value(val.unwrap_or_default());
            }
        }

        let boolean_array = builder.finish();
        self.writer
            .write_all(boolean_array.into_data().buffers()[0].as_slice())
            .await?;
        Ok(())
    }

    /// Encode array of primitive values.
    async fn encode_primitive(&mut self, arrays: &[&dyn Array]) -> Result<usize> {
        assert!(!arrays.is_empty());
        let data_type = arrays[0].data_type();
        let offset = self.writer.tell();

        if matches!(data_type, DataType::Boolean) {
            let boolean_arr = arrays
                .iter()
                .map(|a| as_boolean_array(*a))
                .collect::<Vec<&BooleanArray>>();
            self.encode_boolean(boolean_arr.as_slice()).await?;
        } else {
            let byte_width = data_type.byte_width();
            for a in arrays.iter() {
                let data = a.to_data();
                let slice = unsafe {
                    from_raw_parts(
                        data.buffers()[0].as_ptr().add(a.offset() * byte_width),
                        a.len() * byte_width,
                    )
                };
                self.writer.write_all(slice).await?;
            }
        }
        Ok(offset)
    }

    /// Encode fixed size list.
    async fn encode_fixed_size_list(
        &mut self,
        arrays: &[&dyn Array],
        items: &Field,
    ) -> Result<usize> {
        let mut value_arrs: Vec<ArrayRef> = Vec::new();

        for array in arrays {
            let list_array = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| Error::Schema {
                    message: format!("Needed a FixedSizeListArray but got {}", array.data_type()),
                    location: location!(),
                })?;
            let offset = list_array.value_offset(0) as usize;
            let length = list_array.len();
            let value_length = list_array.value_length() as usize;
            let value_array = list_array.values().slice(offset, length * value_length);
            value_arrs.push(value_array);
        }

        self.encode_internal(
            value_arrs
                .iter()
                .map(|a| a.as_ref())
                .collect::<Vec<_>>()
                .as_slice(),
            items.data_type(),
        )
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

/// Get byte range from the row offset range.
#[inline]
fn get_byte_range(data_type: &DataType, row_range: Range<usize>) -> Range<usize> {
    match data_type {
        DataType::Boolean => row_range.start / 8..bit_util::ceil(row_range.end, 8),
        _ => row_range.start * data_type.byte_width()..row_range.end * data_type.byte_width(),
    }
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
            return Err(Error::IO {
                message: format!(
                    "PlainDecoder: request([{}..{}]) out of range: [0..{}]",
                    start, end, self.length
                ),
                location: location!(),
            });
        }
        let byte_range = get_byte_range(self.data_type, start..end);
        let range = Range {
            start: self.position + byte_range.start,
            end: self.position + byte_range.end,
        };

        let data = self.reader.get_range(range).await?;
        let is_aligned = matches!(self.data_type, DataType::Boolean)
            || data.as_ptr().align_offset(self.data_type.byte_width()) == 0;
        // TODO: replace this with safe method once arrow-rs 47.0.0 comes out.
        // https://github.com/lancedb/lance/issues/1237
        // Zero-copy conversion from bytes
        // Safety: the bytes are owned by the `data` value, so the pointer
        // will be valid for the lifetime of the Arc we are passing in.
        let buf = if is_aligned {
            unsafe {
                Buffer::from_custom_allocation(
                    NonNull::new(data.as_ptr() as _).unwrap(),
                    data.len(),
                    Arc::new(data),
                )
            }
        } else {
            // If data isn't aligned appropriately for the data type, we need to copy
            // the buffer.
            Buffer::from(data)
        };

        // booleans are bitpacked, so we need an offset to provide the exact
        // requested range.
        let offset = if self.data_type == &DataType::Boolean {
            start % 8
        } else {
            0
        };

        let array_data = ArrayDataBuilder::new(self.data_type.clone())
            .len(end - start)
            .offset(offset)
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
            return Err(Error::Schema {
                message: format!(
                    "Items for fixed size list should be primitives but found {}",
                    items.data_type()
                ),
                location: location!(),
            });
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
        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            item_array, list_size,
        )?) as ArrayRef)
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
            .ok_or_else(|| Error::Schema {
                message: "Could not cast to UInt8Array for FixedSizeBinary".to_string(),
                location: location!(),
            })?;
        Ok(Arc::new(FixedSizeBinaryArray::try_new_from_values(values, stride)?) as ArrayRef)
    }

    async fn take_boolean(&self, indices: &UInt32Array) -> Result<ArrayRef> {
        let block_size = self.reader.block_size() as u32;
        let boolean_block_size = block_size * 8;

        let mut chunk_ranges = vec![];
        let mut start: u32 = 0;
        for j in 0..(indices.len() - 1) as u32 {
            if (indices.value(j as usize + 1) / boolean_block_size)
                > (indices.value(start as usize) / boolean_block_size)
            {
                let next_start = j + 1;
                chunk_ranges.push(start..next_start);
                start = next_start;
            }
        }
        // Remaining
        chunk_ranges.push(start..indices.len() as u32);

        let arrays = stream::iter(chunk_ranges)
            .map(|cr| async move {
                let index_chunk = indices.slice(cr.start as usize, cr.len());
                // request contains the array indices we are retrieving in this chunk.
                let request: &UInt32Array = as_primitive_array(&index_chunk);

                // Get the starting index
                let start = request.value(0);
                // Final index is the last value
                let end = request.value(request.len() - 1);
                let array = self.get(start as usize..end as usize + 1).await?;

                let shifted_indices = sub(request, &UInt32Array::new_scalar(start))?;
                Ok::<ArrayRef, Error>(take(&array, &shifted_indices, None)?)
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        let references = arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        Ok(concat(&references)?)
    }
}

fn make_chunked_requests(
    indices: &[u32],
    byte_width: usize,
    block_size: usize,
) -> Vec<Range<usize>> {
    let mut chunked_ranges = vec![];
    let mut start: usize = 0;
    // Note: limit the I/O size to the block size.
    //
    // Another option could be checking whether `indices[i]` and `indices[i+1]` are not
    // farther way than the block size:
    //    indices[i] * byte_width + block_size < indices[i+1] * byte_width
    // It might allow slightly larger sequential reads.
    for i in 0..indices.len() - 1 {
        if indices[i + 1] as usize * byte_width > indices[start] as usize * byte_width + block_size
        {
            chunked_ranges.push(start..i + 1);
            start = i + 1;
        }
    }
    chunked_ranges.push(start..indices.len());
    chunked_ranges
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

        if matches!(self.data_type, DataType::Boolean) {
            return self.take_boolean(indices).await;
        }

        let block_size = self.reader.block_size();
        let byte_width = self.data_type.byte_width();

        let chunked_ranges = make_chunked_requests(indices.values(), byte_width, block_size);

        let arrays = stream::iter(chunked_ranges)
            .map(|cr| async move {
                let index_chunk = indices.slice(cr.start, cr.len());
                let request: &UInt32Array = as_primitive_array(&index_chunk);

                let start = request.value(0);
                let end = request.value(request.len() - 1);
                let array = self.get(start as usize..end as usize + 1).await?;
                let adjusted_offsets = sub(request, &UInt32Array::new_scalar(start))?;
                Ok::<ArrayRef, Error>(take(&array, &adjusted_offsets, None)?)
            })
            .buffered(num_cpus::get() * PARALLELISM_FACTOR)
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
    use std::ops::Deref;
    use std::sync::Arc;

    use arrow::compute::concat_batches;
    use arrow_array::*;
    use arrow_schema::{Field, Schema as ArrowSchema};
    use object_store::path::Path;
    use rand::prelude::*;

    use crate::datatypes::Schema;
    use crate::io::object_writer::ObjectWriter;
    use crate::io::{FileReader, FileWriter, ObjectStore};

    use super::*;

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
        let input: Vec<i64> = Vec::from_iter(1..127_i64);
        for t in int_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let mut arrs: Vec<ArrayRef> = Vec::new();
            for _ in 0..10 {
                arrs.push(Arc::new(make_array_(&t, &buffer).await));
            }
            test_round_trip(arrs.as_slice(), t).await;
        }

        let float_types = vec![DataType::Float16, DataType::Float32, DataType::Float64];
        let mut rng = rand::thread_rng();
        let input: Vec<f64> = (1..127).map(|_| rng.gen()).collect();
        for t in float_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let mut arrs: Vec<ArrayRef> = Vec::new();

            for _ in 0..10 {
                arrs.push(Arc::new(make_array_(&t, &buffer).await));
            }
            test_round_trip(arrs.as_slice(), t).await;
        }
    }

    async fn test_round_trip(expected: &[ArrayRef], data_type: DataType) {
        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        let mut encoder = PlainEncoder::new(&mut object_writer, &data_type);

        let expected_as_array = expected
            .iter()
            .map(|e| e.as_ref())
            .collect::<Vec<&dyn Array>>();
        assert_eq!(
            encoder.encode(expected_as_array.as_slice()).await.unwrap(),
            0
        );
        object_writer.shutdown().await.unwrap();

        let reader = store.open(&path).await.unwrap();
        assert!(reader.size().await.unwrap() > 0);
        // Expected size is the total of all arrays
        let expected_size = expected.iter().map(|e| e.len()).sum();
        let decoder = PlainDecoder::new(reader.as_ref(), &data_type, 0, expected_size).unwrap();
        let arr = decoder.decode().await.unwrap();
        let actual = arr.as_ref();
        let expected_merged = concat(expected_as_array.as_slice()).unwrap();
        assert_eq!(expected_merged.deref(), actual);
        assert_eq!(expected_size, actual.len());
    }

    #[tokio::test]
    async fn test_encode_decode_bool_array() {
        let mut arrs: Vec<ArrayRef> = Vec::new();

        for _ in 0..10 {
            // It is important that the boolean array length is < 8 so we can test if the Arrays are merged correctly
            arrs.push(Arc::new(BooleanArray::from(vec![true, true, true])) as ArrayRef);
        }
        test_round_trip(arrs.as_slice(), DataType::Boolean).await;
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
        let input = Vec::from_iter(1..127_i64);
        for t in int_types {
            let buffer = Buffer::from_slice_ref(input.as_slice());
            let list_type =
                DataType::FixedSizeList(Arc::new(Field::new("item", t.clone(), true)), 3);
            let mut arrs: Vec<ArrayRef> = Vec::new();

            for _ in 0..10 {
                let items = make_array_(&t.clone(), &buffer).await;
                let arr = FixedSizeListArray::try_new_from_values(items, 3).unwrap();
                arrs.push(Arc::new(arr) as ArrayRef);
            }
            test_round_trip(arrs.as_slice(), list_type).await;
        }
    }

    #[tokio::test]
    async fn test_encode_decode_fixed_size_binary_array() {
        let t = DataType::FixedSizeBinary(3);
        let mut arrs: Vec<ArrayRef> = Vec::new();

        for _ in 0..10 {
            let values = UInt8Array::from(Vec::from_iter(1..127_u8));
            let arr = FixedSizeBinaryArray::try_new_from_values(&values, 3).unwrap();
            arrs.push(Arc::new(arr) as ArrayRef);
        }
        test_round_trip(arrs.as_slice(), t).await;
    }

    #[tokio::test]
    async fn test_encode_decode_nested_fixed_size_list() {
        // FixedSizeList of FixedSizeList
        let inner = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int64, true)), 2);
        let t = DataType::FixedSizeList(Arc::new(Field::new("item", inner, true)), 2);
        let mut arrs: Vec<ArrayRef> = Vec::new();

        for _ in 0..10 {
            let values = Int64Array::from_iter_values(1..=120_i64);
            let arr = FixedSizeListArray::try_new_from_values(
                FixedSizeListArray::try_new_from_values(values, 2).unwrap(),
                2,
            )
            .unwrap();
            arrs.push(Arc::new(arr) as ArrayRef);
        }
        test_round_trip(arrs.as_slice(), t).await;

        // FixedSizeList of FixedSizeBinary
        let inner = DataType::FixedSizeBinary(2);
        let t = DataType::FixedSizeList(Arc::new(Field::new("item", inner, true)), 2);
        let mut arrs: Vec<ArrayRef> = Vec::new();

        for _ in 0..10 {
            let values = UInt8Array::from_iter_values(1..=120_u8);
            let arr = FixedSizeListArray::try_new_from_values(
                FixedSizeBinaryArray::try_new_from_values(&values, 2).unwrap(),
                2,
            )
            .unwrap();
            arrs.push(Arc::new(arr) as ArrayRef);
        }
        test_round_trip(arrs.as_slice(), t).await;
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
        assert_eq!(encoder.encode(&[&array]).await.unwrap(), 0);
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
            &decoder.get(5..5).await.unwrap(),
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
        assert_eq!(encoder.encode(&[&array]).await.unwrap(), 0);
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

    async fn read_file_as_one_batch(object_store: &ObjectStore, path: &Path) -> RecordBatch {
        let reader = FileReader::try_new(object_store, path).await.unwrap();
        let mut batches = vec![];
        for i in 0..reader.num_batches() {
            batches.push(
                reader
                    .read_batch(i as i32, .., reader.schema())
                    .await
                    .unwrap(),
            );
        }
        let arrow_schema = Arc::new(reader.schema().into());
        concat_batches(&arrow_schema, &batches).unwrap()
    }

    /// Test encoding arrays that share the same underneath buffer.
    #[tokio::test]
    async fn test_encode_slice() {
        let store = ObjectStore::memory();
        let path = Path::from("/shared_slice");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer = FileWriter::try_new(&store, &path, schema, &Default::default())
            .await
            .unwrap();

        let array = Int32Array::from_iter_values(0..1000);

        for i in (0..1000).step_by(4) {
            let data = array.slice(i, 4);
            file_writer
                .write(&[RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(data)]).unwrap()])
                .await
                .unwrap();
        }
        file_writer.finish().await.unwrap();
        assert!(store.size(&path).await.unwrap() < 2 * 8 * 1000);

        let batch = read_file_as_one_batch(&store, &path).await;
        assert_eq!(batch.column_by_name("i").unwrap().as_ref(), &array);
    }

    #[tokio::test]
    async fn test_boolean_slice() {
        let store = ObjectStore::memory();
        let path = Path::from("/bool_slice");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "b",
            DataType::Boolean,
            true,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer =
            FileWriter::try_new(&store, &path, schema.clone(), &Default::default())
                .await
                .unwrap();

        let array = BooleanArray::from((0..120).map(|v| v % 5 == 0).collect::<Vec<_>>());
        for i in 0..10 {
            let data = array.slice(i * 12, 12); // one and half byte
            file_writer
                .write(&[RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(data)]).unwrap()])
                .await
                .unwrap();
        }
        file_writer.finish().await.unwrap();

        let batch = read_file_as_one_batch(&store, &path).await;
        assert_eq!(batch.column_by_name("b").unwrap().as_ref(), &array);

        let array = BooleanArray::from(vec![Some(true), Some(false), None]);
        let mut file_writer = FileWriter::try_new(&store, &path, schema, &Default::default())
            .await
            .unwrap();
        file_writer
            .write(&[RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(array)]).unwrap()])
            .await
            .unwrap();
        file_writer.finish().await.unwrap();
        let batch = read_file_as_one_batch(&store, &path).await;

        // None default to Some(false), since we don't support null values yet.
        let expected = BooleanArray::from(vec![Some(true), Some(false), Some(false)]);
        assert_eq!(batch.column_by_name("b").unwrap().as_ref(), &expected);
    }

    #[tokio::test]
    async fn test_encode_fixed_size_list_slice() {
        let store = ObjectStore::memory();
        let path = Path::from("/shared_slice");

        let array = Int32Array::from_iter_values(0..1600);
        let fixed_size_list = FixedSizeListArray::try_new_from_values(array, 16).unwrap();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "fl",
            fixed_size_list.data_type().clone(),
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer = FileWriter::try_new(&store, &path, schema, &Default::default())
            .await
            .unwrap();

        for i in (0..100).step_by(4) {
            let slice: FixedSizeListArray = fixed_size_list.slice(i, 4);
            file_writer
                .write(&[
                    RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(slice)]).unwrap(),
                ])
                .await
                .unwrap();
        }
        file_writer.finish().await.unwrap();

        let batch = read_file_as_one_batch(&store, &path).await;
        assert_eq!(
            batch.column_by_name("fl").unwrap().as_ref(),
            &fixed_size_list
        );
    }

    #[tokio::test]
    async fn test_take_boolean() {
        let store = ObjectStore::memory();
        let path = Path::from("/take_bools");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "b",
            DataType::Boolean,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer =
            FileWriter::try_new(&store, &path, schema.clone(), &Default::default())
                .await
                .unwrap();

        let array = BooleanArray::from((0..120).map(|v| v % 5 == 0).collect::<Vec<_>>());
        let batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(array.clone())]).unwrap();
        file_writer.write(&[batch]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual = reader
            .take(&[2, 4, 5, 8, 63, 64, 65], &schema)
            .await
            .unwrap();

        assert_eq!(
            actual.column_by_name("b").unwrap().as_ref(),
            &BooleanArray::from(vec![false, false, true, false, false, false, true])
        );
    }

    #[tokio::test]
    async fn test_take_boolean_beyond_chunk() {
        let mut store = ObjectStore::memory();
        store.set_block_size(256);
        let path = Path::from("/take_bools");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "b",
            DataType::Boolean,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer =
            FileWriter::try_new(&store, &path, schema.clone(), &Default::default())
                .await
                .unwrap();

        let array = BooleanArray::from((0..5000).map(|v| v % 5 == 0).collect::<Vec<_>>());
        let batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(array.clone())]).unwrap();
        file_writer.write(&[batch]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual = reader.take(&[2, 4, 5, 8, 4555], &schema).await.unwrap();

        assert_eq!(
            actual.column_by_name("b").unwrap().as_ref(),
            &BooleanArray::from(vec![false, false, true, false, true])
        );
    }

    #[test]
    fn test_make_chunked_request() {
        let byte_width: usize = 4096; // 4K
        let prefetch_size: usize = 64 * 1024; // 64KB.
        let u32_overflow: usize = u32::MAX as usize + 10;

        let indices: Vec<u32> = vec![
            1,
            10,
            20,
            100,
            120,
            (u32_overflow / byte_width) as u32, // Two overflow offsets
            (u32_overflow / byte_width) as u32 + 100,
        ];
        let chunks = make_chunked_requests(&indices, byte_width, prefetch_size);
        assert_eq!(chunks.len(), 6, "got chunks: {:?}", chunks);
        assert_eq!(chunks, vec![(0..2), (2..3), (3..4), (4..5), (5..6), (6..7)])
    }
}
