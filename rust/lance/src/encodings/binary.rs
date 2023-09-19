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

//! Var-length binary encoding.
//!

use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_arith::arithmetic::subtract_scalar;
use arrow_array::builder::{ArrayBuilder, PrimitiveBuilder};
use arrow_array::{
    cast::as_primitive_array,
    new_empty_array,
    types::{BinaryType, ByteArrayType, Int64Type, LargeBinaryType, LargeUtf8Type, Utf8Type},
    Array, ArrayRef, GenericByteArray, Int64Array, OffsetSizeTrait, UInt32Array,
};
use arrow_buffer::{bit_util, ArrowNativeType, Buffer, MutableBuffer};
use arrow_cast::cast::cast;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use arrow_select::{concat::concat, take::take};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, repeat_with, StreamExt, TryStreamExt};
use tokio::io::AsyncWriteExt;

use super::Encoder;
use super::{plain::PlainDecoder, AsyncIndex};
use crate::encodings::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;
use crate::io::ReadBatchParams;
use snafu::{location, Location};

/// Encoder for Var-binary encoding.
pub struct BinaryEncoder<'a> {
    writer: &'a mut ObjectWriter,
}

impl<'a> BinaryEncoder<'a> {
    pub fn new(writer: &'a mut ObjectWriter) -> Self {
        Self { writer }
    }

    async fn encode_typed_arr<T: ByteArrayType>(&mut self, arrs: &[&dyn Array]) -> Result<usize> {
        let capacity: usize = arrs.iter().map(|a| a.len()).sum();
        let mut pos_builder: PrimitiveBuilder<Int64Type> =
            PrimitiveBuilder::with_capacity(capacity + 1);

        let mut last_offset: usize = self.writer.tell();
        pos_builder.append_value(last_offset as i64);
        for array in arrs.iter() {
            let arr = array
                .as_any()
                .downcast_ref::<GenericByteArray<T>>()
                .unwrap();

            let offsets = arr.value_offsets();

            let start = offsets[0].as_usize();
            let end = offsets[offsets.len() - 1].as_usize();
            let b = unsafe {
                std::slice::from_raw_parts(
                    arr.to_data().buffers()[1].as_ptr().add(start),
                    end - start,
                )
            };
            self.writer.write_all(b).await?;

            let start_offset = offsets[0].as_usize();
            offsets
                .iter()
                .skip(1)
                .map(|b| b.as_usize() - start_offset + last_offset)
                .for_each(|o| pos_builder.append_value(o as i64));
            last_offset = pos_builder.values_slice()[pos_builder.len() - 1] as usize;
        }

        let positions_offset = self.writer.tell();
        let pos_array = pos_builder.finish();
        self.writer
            .write_all(pos_array.to_data().buffers()[0].as_slice())
            .await?;
        Ok(positions_offset)
    }
}

#[async_trait]
impl<'a> Encoder for BinaryEncoder<'a> {
    async fn encode(&mut self, arrs: &[&dyn Array]) -> Result<usize> {
        assert!(!arrs.is_empty());
        let data_type = arrs[0].data_type();
        match data_type {
            DataType::Utf8 => self.encode_typed_arr::<Utf8Type>(arrs).await,
            DataType::Binary => self.encode_typed_arr::<BinaryType>(arrs).await,
            DataType::LargeUtf8 => self.encode_typed_arr::<LargeUtf8Type>(arrs).await,
            DataType::LargeBinary => self.encode_typed_arr::<LargeBinaryType>(arrs).await,
            _ => {
                return Err(crate::Error::IO {
                    message: format!("Binary encoder does not support {}", data_type),
                    location: location!(),
                })
            }
        }
    }
}

/// Var-binary encoding decoder.
pub struct BinaryDecoder<'a, T: ByteArrayType> {
    reader: &'a dyn ObjectReader,

    position: usize,

    length: usize,

    nullable: bool,

    phantom: PhantomData<T>,
}

/// Var-length Binary Decoder
///
impl<'a, T: ByteArrayType> BinaryDecoder<'a, T> {
    /// Create a [BinaryEncoder] to decode one batch.
    ///
    ///  - `position`, file position where this batch starts.
    ///  - `length`, the number of records in this batch.
    ///  - `nullable`, whether this batch contains nullable value.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use arrow_array::types::Utf8Type;
    /// use object_store::path::Path;
    /// use lance::io::ObjectStore;
    /// use lance::encodings::binary::BinaryDecoder;
    ///
    /// async {
    ///     let object_store = ObjectStore::memory();
    ///     let path = Path::from("/data.lance");
    ///     let reader = object_store.open(&path).await.unwrap();
    ///     let string_decoder = BinaryDecoder::<Utf8Type>::new(reader.as_ref(), 100, 1024, true);
    /// };
    /// ```
    pub fn new(
        reader: &'a dyn ObjectReader,
        position: usize,
        length: usize,
        nullable: bool,
    ) -> Self {
        Self {
            reader,
            position,
            length,
            nullable,
            phantom: PhantomData,
        }
    }

    /// Get the position array for the batch.
    async fn get_positions(&self, index: Range<usize>) -> Result<Arc<Int64Array>> {
        let position_decoder = PlainDecoder::new(
            self.reader,
            &DataType::Int64,
            self.position,
            self.length + 1,
        )?;
        let values = position_decoder.get(index.start..index.end + 1).await?;
        Ok(Arc::new(as_primitive_array(&values).clone()))
    }

    /// Read the array with batch positions and range.
    ///
    /// Parameters
    ///
    ///  - *positions*: position array for the batch.
    ///  - *range*: range of rows to read.
    async fn get_range(&self, positions: &Int64Array, range: Range<usize>) -> Result<ArrayRef> {
        assert!(positions.len() >= range.end);
        let start = positions.value(range.start);
        let end = positions.value(range.end);

        let slice = positions.slice(range.start, range.len() + 1);
        let offset_data = if T::Offset::IS_LARGE {
            subtract_scalar(&slice, start)?.into_data()
        } else {
            cast(
                &(Arc::new(subtract_scalar(&slice, start)?) as ArrayRef),
                &DataType::Int32,
            )?
            .into_data()
        };

        let bytes: Bytes = if start >= end {
            Bytes::new()
        } else {
            self.reader.get_range(start as usize..end as usize).await?
        };

        let mut data_builder = ArrayDataBuilder::new(T::DATA_TYPE)
            .len(range.len())
            .null_count(0);

        // Count nulls
        if self.nullable {
            let mut null_count = 0;
            let mut null_buf = MutableBuffer::new_null(self.length);
            slice.values().windows(2).enumerate().for_each(|(idx, w)| {
                if w[0] == w[1] {
                    bit_util::unset_bit(null_buf.as_mut(), idx);
                    null_count += 1;
                } else {
                    bit_util::set_bit(null_buf.as_mut(), idx);
                }
            });
            data_builder = data_builder
                .null_count(null_count)
                .null_bit_buffer(Some(null_buf.into()));
        }

        // TODO: replace this with safe method once arrow-rs 47.0.0 comes out.
        // https://github.com/lancedb/lance/issues/1237
        // Zero-copy conversion from bytes
        // Safety: the bytes are owned by the `data` value, so the pointer
        // will be valid for the lifetime of the Arc we are passing in.
        let buf = unsafe {
            Buffer::from_custom_allocation(
                NonNull::new(bytes.as_ptr() as _).unwrap(),
                bytes.len(),
                Arc::new(bytes),
            )
        };

        let array_data = data_builder
            .add_buffer(offset_data.buffers()[0].clone())
            .add_buffer(buf)
            .build()?;

        Ok(Arc::new(GenericByteArray::<T>::from(array_data)))
    }

    async fn take_internal(
        &self,
        positions: &Int64Array,
        indices: &UInt32Array,
    ) -> Result<ArrayRef> {
        let start = indices.value(0);
        let end = indices.value(indices.len() - 1);
        let array = self
            .get_range(positions, start as usize..end as usize + 1)
            .await?;
        let adjusted_offsets = subtract_scalar(indices, start)?;
        Ok(take(&array, &adjusted_offsets, None)?)
    }
}

fn plan_take_chunks(
    positions: &Int64Array,
    indices: &UInt32Array,
    min_io_size: i64,
) -> Result<Vec<UInt32Array>> {
    let start = indices.value(0);
    let indices = subtract_scalar(indices, start)?;

    let mut chunks: Vec<UInt32Array> = vec![];
    let mut start_idx = 0;
    for i in 0..indices.len() {
        let current = indices.value(i) as usize;
        if positions.value(current) - positions.value(indices.value(start_idx) as usize)
            > min_io_size
        {
            chunks.push(as_primitive_array(&indices.slice(start_idx, i - start_idx)).clone());
            start_idx = i;
        }
    }
    chunks.push(as_primitive_array(&indices.slice(start_idx, indices.len() - start_idx)).clone());

    Ok(chunks)
}

#[async_trait]
impl<'a, T: ByteArrayType> Decoder for BinaryDecoder<'a, T> {
    async fn decode(&self) -> Result<ArrayRef> {
        self.get(..).await
    }

    async fn take(&self, indices: &UInt32Array) -> Result<ArrayRef> {
        if indices.is_empty() {
            return Ok(new_empty_array(&T::DATA_TYPE));
        }

        let start = indices.value(0);
        let end = indices.value(indices.len() - 1);

        // TODO: make min batch size configurable.
        // TODO: make reading positions in chunks too.
        const MIN_IO_SIZE: i64 = 64 * 1024; // 64KB
        let positions = self
            .get_positions(start as usize..(end + 1) as usize)
            .await?;
        let chunks = plan_take_chunks(&positions, indices, MIN_IO_SIZE)?;

        let arrays = stream::iter(chunks)
            .zip(repeat_with(|| positions.clone()))
            .map(|(indices, positions)| async move {
                self.take_internal(positions.as_ref(), &indices).await
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        Ok(concat(
            arrays
                .iter()
                .map(|a| a.as_ref())
                .collect::<Vec<_>>()
                .as_slice(),
        )?)
    }
}

#[async_trait]
impl<'a, T: ByteArrayType> AsyncIndex<usize> for BinaryDecoder<'a, T> {
    type Output = Result<ArrayRef>;

    async fn get(&self, index: usize) -> Self::Output {
        self.get(index..index + 1).await
    }
}

#[async_trait]
impl<'a, T: ByteArrayType> AsyncIndex<RangeFrom<usize>> for BinaryDecoder<'a, T> {
    type Output = Result<ArrayRef>;

    async fn get(&self, index: RangeFrom<usize>) -> Self::Output {
        self.get(index.start..self.length).await
    }
}

#[async_trait]
impl<'a, T: ByteArrayType> AsyncIndex<RangeTo<usize>> for BinaryDecoder<'a, T> {
    type Output = Result<ArrayRef>;

    async fn get(&self, index: RangeTo<usize>) -> Self::Output {
        self.get(0..index.end).await
    }
}

#[async_trait]
impl<'a, T: ByteArrayType> AsyncIndex<RangeFull> for BinaryDecoder<'a, T> {
    type Output = Result<ArrayRef>;

    async fn get(&self, _: RangeFull) -> Self::Output {
        self.get(0..self.length).await
    }
}

#[async_trait]
impl<'a, T: ByteArrayType> AsyncIndex<ReadBatchParams> for BinaryDecoder<'a, T> {
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

#[async_trait]
impl<'a, T: ByteArrayType> AsyncIndex<Range<usize>> for BinaryDecoder<'a, T> {
    type Output = Result<ArrayRef>;

    async fn get(&self, index: Range<usize>) -> Self::Output {
        let position_decoder = PlainDecoder::new(
            self.reader,
            &DataType::Int64,
            self.position,
            self.length + 1,
        )?;
        let positions = position_decoder.get(index.start..index.end + 1).await?;
        let int64_positions: &Int64Array = as_primitive_array(&positions);

        self.get_range(int64_positions, 0..index.len()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_select::concat::concat;

    use arrow_array::{
        cast::AsArray, new_empty_array, types::GenericStringType, BinaryArray, GenericStringArray,
        LargeStringArray, OffsetSizeTrait, StringArray,
    };
    use object_store::path::Path;

    use crate::io::ObjectStore;

    async fn write_test_data<O: OffsetSizeTrait>(
        store: &ObjectStore,
        path: &Path,
        arr: &[&GenericStringArray<O>],
    ) -> Result<usize> {
        let mut object_writer = ObjectWriter::new(store, path).await.unwrap();
        // Write some garbage to reset "tell()".
        object_writer.write_all(b"1234").await.unwrap();
        let mut encoder = BinaryEncoder::new(&mut object_writer);

        let arrs = arr.iter().map(|a| a as &dyn Array).collect::<Vec<_>>();
        let pos = encoder.encode(arrs.as_slice()).await.unwrap();
        object_writer.shutdown().await.unwrap();
        Ok(pos)
    }

    async fn test_round_trips<O: OffsetSizeTrait>(arrs: &[&GenericStringArray<O>]) {
        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let pos = write_test_data(&store, &path, arrs).await.unwrap();
        let reader = store.open(&path).await.unwrap();
        let read_len = arrs.iter().map(|a| a.len()).sum();
        let decoder =
            BinaryDecoder::<GenericStringType<O>>::new(reader.as_ref(), pos, read_len, true);
        let actual_arr = decoder.decode().await.unwrap();

        let arrs_ref = arrs.iter().map(|a| a as &dyn Array).collect::<Vec<_>>();
        let expected = concat(arrs_ref.as_slice()).unwrap();
        assert_eq!(
            actual_arr
                .as_any()
                .downcast_ref::<GenericStringArray<O>>()
                .unwrap(),
            expected
                .as_any()
                .downcast_ref::<GenericStringArray<O>>()
                .unwrap(),
        );
    }

    #[tokio::test]
    async fn test_write_binary_data() {
        test_round_trips(&[&StringArray::from(vec!["a", "b", "cd", "efg"])]).await;
        test_round_trips(&[&StringArray::from(vec![Some("a"), None, Some("cd"), None])]).await;
        test_round_trips(&[
            &StringArray::from(vec![Some("a"), None, Some("cd"), None]),
            &StringArray::from(vec![Some("f"), None, Some("gh"), None]),
            &StringArray::from(vec![Some("t"), None, Some("uv"), None]),
        ])
        .await;
        test_round_trips(&[&LargeStringArray::from(vec!["a", "b", "cd", "efg"])]).await;
        test_round_trips(&[&LargeStringArray::from(vec![
            Some("a"),
            None,
            Some("cd"),
            None,
        ])])
        .await;
        test_round_trips(&[
            &LargeStringArray::from(vec![Some("a"), Some("b")]),
            &LargeStringArray::from(vec![Some("c")]),
            &LargeStringArray::from(vec![Some("d"), Some("e")]),
        ])
        .await;
    }

    #[tokio::test]
    async fn test_write_binary_data_with_offset() {
        let array: StringArray = StringArray::from(vec![Some("d"), Some("e")]).slice(1, 1);
        test_round_trips(&[&array]).await;
    }

    #[tokio::test]
    async fn test_range_query() {
        let data = StringArray::from_iter_values(["a", "b", "c", "d", "e", "f", "g"]);

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        // Write some gabage to reset "tell()".
        object_writer.write_all(b"1234").await.unwrap();
        let mut encoder = BinaryEncoder::new(&mut object_writer);
        let pos = encoder.encode(&[&data]).await.unwrap();
        object_writer.shutdown().await.unwrap();

        let reader = store.open(&path).await.unwrap();
        let decoder = BinaryDecoder::<Utf8Type>::new(reader.as_ref(), pos, data.len(), false);
        assert_eq!(
            decoder.decode().await.unwrap().as_ref(),
            &StringArray::from_iter_values(["a", "b", "c", "d", "e", "f", "g"])
        );

        assert_eq!(
            decoder.get(..).await.unwrap().as_ref(),
            &StringArray::from_iter_values(["a", "b", "c", "d", "e", "f", "g"])
        );

        assert_eq!(
            decoder.get(2..5).await.unwrap().as_ref(),
            &StringArray::from_iter_values(["c", "d", "e"])
        );

        assert_eq!(
            decoder.get(..5).await.unwrap().as_ref(),
            &StringArray::from_iter_values(["a", "b", "c", "d", "e"])
        );

        assert_eq!(
            decoder.get(4..).await.unwrap().as_ref(),
            &StringArray::from_iter_values(["e", "f", "g"])
        );
        assert_eq!(
            decoder.get(2..2).await.unwrap().as_ref(),
            &new_empty_array(&DataType::Utf8)
        );
        assert!(decoder.get(100..100).await.is_err());
    }

    #[tokio::test]
    async fn test_take() {
        let data = StringArray::from_iter_values(["a", "b", "c", "d", "e", "f", "g"]);

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let pos = write_test_data(&store, &path, &[&data]).await.unwrap();

        let reader = store.open(&path).await.unwrap();
        let decoder = BinaryDecoder::<Utf8Type>::new(reader.as_ref(), pos, data.len(), false);

        let actual = decoder
            .take(&UInt32Array::from_iter_values([1, 2, 5]))
            .await
            .unwrap();
        assert_eq!(
            actual.as_ref(),
            &StringArray::from_iter_values(["b", "c", "f"])
        );
    }

    #[tokio::test]
    async fn test_take_sparse_indices() {
        let data = StringArray::from_iter_values((0..1000000).map(|v| format!("string-{v}")));

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let pos = write_test_data(&store, &path, &[&data]).await.unwrap();

        let reader = store.open(&path).await.unwrap();
        let decoder = BinaryDecoder::<Utf8Type>::new(reader.as_ref(), pos, data.len(), false);

        let positions = decoder.get_positions(1..999998).await.unwrap();
        let indices = UInt32Array::from_iter_values([1, 999998]);
        let chunks = plan_take_chunks(positions.as_ref(), &indices, 64 * 1024).unwrap();
        // Relative offset within the positions.
        assert_eq!(
            chunks,
            vec![
                UInt32Array::from_iter_values([0]),
                UInt32Array::from_iter_values([999997])
            ]
        );

        let actual = decoder
            .take(&UInt32Array::from_iter_values([1, 999998]))
            .await
            .unwrap();
        assert_eq!(
            actual.as_ref(),
            &StringArray::from_iter_values(["string-1", "string-999998"])
        );
    }

    #[tokio::test]
    async fn test_write_slice() {
        let data = StringArray::from_iter_values((0..100).map(|v| format!("abcdef-{v:#03}")));
        let store = ObjectStore::memory();
        let path = Path::from("/slices");

        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        let mut encoder = BinaryEncoder::new(&mut object_writer);
        for i in 0..10 {
            let pos = encoder.encode(&[&data.slice(i * 10, 10)]).await.unwrap();
            assert_eq!(pos, (i * (8 * 11) /* offset array */ + (i + 1) * (10 * 10)));
        }
    }

    #[tokio::test]
    async fn test_write_binary_with_nulls() {
        let data = BinaryArray::from_iter((0..60000).map(|v| {
            if v % 4 != 0 {
                Some::<&[u8]>(b"abcdefgh")
            } else {
                None
            }
        }));
        let store = ObjectStore::memory();
        let path = Path::from("/slices");

        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        // Write some garbage to reset "tell()".
        object_writer.write_all(b"1234").await.unwrap();
        let mut encoder = BinaryEncoder::new(&mut object_writer);

        // let arrs = arr.iter().map(|a| a as &dyn Array).collect::<Vec<_>>();
        let pos = encoder.encode(&[&data]).await.unwrap();
        object_writer.shutdown().await.unwrap();

        let reader = store.open(&path).await.unwrap();
        let decoder = BinaryDecoder::<BinaryType>::new(reader.as_ref(), pos, data.len(), true);
        let idx = UInt32Array::from(vec![0_u32, 5_u32, 59996_u32]);
        let actual = decoder.take(&idx).await.unwrap();
        let values: Vec<Option<&[u8]>> = vec![None, Some(b"abcdefgh"), None];
        assert_eq!(actual.as_binary::<i32>(), &BinaryArray::from(values));
    }
}
