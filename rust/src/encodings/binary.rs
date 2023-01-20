//! Var-length Binary Encoding
//!

use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use std::sync::Arc;

use arrow_arith::arithmetic::{subtract_scalar, subtract_scalar_dyn};
use arrow_array::cast::as_primitive_array;
use arrow_array::{
    types::{BinaryType, ByteArrayType, Int64Type, LargeBinaryType, LargeUtf8Type, Utf8Type},
    Array, ArrayRef, GenericByteArray, Int64Array, OffsetSizeTrait, PrimitiveArray,
};
use arrow_buffer::{bit_util, ArrowNativeType, MutableBuffer};
use arrow_cast::cast::cast;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use async_trait::async_trait;
use tokio::io::AsyncWriteExt;

use super::Encoder;
use super::{plain::PlainDecoder, AsyncIndex};
use crate::encodings::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;

/// Encoder for Var-binary encoding.
pub struct BinaryEncoder<'a> {
    writer: &'a mut ObjectWriter,
}

impl<'a> BinaryEncoder<'a> {
    pub fn new(writer: &'a mut ObjectWriter) -> Self {
        Self { writer }
    }

    async fn encode_typed_arr<T: ByteArrayType>(&mut self, array: &dyn Array) -> Result<usize> {
        let arr = array
            .as_any()
            .downcast_ref::<GenericByteArray<T>>()
            .unwrap();

        let value_offset = self.writer.tell();
        self.writer.write_all(arr.value_data()).await?;
        let offset = self.writer.tell();

        let offsets = arr.value_offsets();
        let start_offset = offsets[0];
        // Did not use `add_scalar(positions, value_offset)`, so we can save a memory copy.
        let positions = PrimitiveArray::<Int64Type>::from_iter(
            offsets
                .iter()
                .map(|o| (((*o - start_offset).as_usize() + value_offset) as i64)),
        );
        self.writer
            .write_all(positions.data().buffers()[0].as_slice())
            .await?;

        Ok(offset)
    }
}

#[async_trait]
impl<'a> Encoder for BinaryEncoder<'a> {
    async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        match array.data_type() {
            DataType::Utf8 => self.encode_typed_arr::<Utf8Type>(array).await,
            DataType::Binary => self.encode_typed_arr::<BinaryType>(array).await,
            DataType::LargeUtf8 => self.encode_typed_arr::<LargeUtf8Type>(array).await,
            DataType::LargeBinary => self.encode_typed_arr::<LargeBinaryType>(array).await,
            _ => {
                return Err(crate::Error::IO(format!(
                    "Binary encoder does not support {}",
                    array.data_type()
                )))
            }
        }
    }
}

/// Var-binary encoding decoder.
pub struct BinaryDecoder<'a, T: ByteArrayType> {
    reader: &'a ObjectReader<'a>,

    position: usize,

    length: usize,

    phantom: PhantomData<T>,
}

/// Var-length Binary Decoder
///
impl<'a, T: ByteArrayType> BinaryDecoder<'a, T> {
    /// Create a [BinaryEncoder] to decode one batch.
    ///
    ///  - `position`, file position where this batch starts.
    ///  - `length`, the number of records in this batch.
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
    ///     let object_store = ObjectStore::new(":memory:").unwrap();
    ///     let path = Path::from("/data.lance");
    ///     let reader = object_store.open(&path).await.unwrap();
    ///     let string_decoder = BinaryDecoder::<Utf8Type>::new(&reader, 100, 1024);
    /// };
    /// ```
    pub fn new(reader: &'a ObjectReader, position: usize, length: usize) -> Self {
        Self {
            reader,
            position,
            length,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<'a, T: ByteArrayType> Decoder for BinaryDecoder<'a, T> {
    async fn decode(&self) -> Result<ArrayRef> {
        self.get(..).await
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

        let start_position = int64_positions.value(0);

        let offset_data = if T::Offset::IS_LARGE {
            subtract_scalar(int64_positions, start_position)?.into_data()
        } else {
            cast(
                &subtract_scalar_dyn::<Int64Type>(&positions, start_position)?,
                &DataType::Int32,
            )?
            .into_data()
        };

        // Count nulls
        let mut null_buf = MutableBuffer::new_null(self.length);
        let mut null_count = 0;
        int64_positions
            .values()
            .windows(2)
            .enumerate()
            .for_each(|(idx, w)| {
                if w[0] == w[1] {
                    bit_util::unset_bit(null_buf.as_mut(), idx);
                    null_count += 1;
                } else {
                    bit_util::set_bit(null_buf.as_mut(), idx);
                }
            });

        let read_len = int64_positions.value(int64_positions.len() - 1) - start_position;
        let bytes = self
            .reader
            .get_range(start_position as usize..(start_position + read_len) as usize)
            .await?;

        let array_data = ArrayDataBuilder::new(T::DATA_TYPE)
            .len(index.len())
            .null_count(null_count)
            .null_bit_buffer(Some(null_buf.into()))
            .add_buffer(offset_data.buffers()[0].clone())
            .add_buffer(bytes.into())
            .build()?;

        Ok(Arc::new(GenericByteArray::<T>::from(array_data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{
        new_empty_array, types::GenericStringType, GenericStringArray, LargeStringArray,
        OffsetSizeTrait, StringArray,
    };
    use object_store::path::Path;

    use crate::io::ObjectStore;

    async fn test_round_trips<O: OffsetSizeTrait>(arr: &GenericStringArray<O>) {
        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        // Write some gabage to reset "tell()".
        object_writer.write_all(b"1234").await.unwrap();
        let mut encoder = BinaryEncoder::new(&mut object_writer);
        let pos = encoder.encode(&arr).await.unwrap();
        object_writer.shutdown().await.unwrap();

        let mut reader = store.open(&path).await.unwrap();
        let decoder = BinaryDecoder::<GenericStringType<O>>::new(&mut reader, pos, arr.len());
        let actual_arr = decoder.decode().await.unwrap();
        assert_eq!(
            actual_arr
                .as_any()
                .downcast_ref::<GenericStringArray<O>>()
                .unwrap(),
            arr
        );
    }

    #[tokio::test]
    async fn test_write_binary_data() {
        test_round_trips(&StringArray::from(vec!["a", "b", "cd", "efg"])).await;
        test_round_trips(&StringArray::from(vec![Some("a"), None, Some("cd"), None])).await;

        test_round_trips(&LargeStringArray::from(vec!["a", "b", "cd", "efg"])).await;
        test_round_trips(&LargeStringArray::from(vec![
            Some("a"),
            None,
            Some("cd"),
            None,
        ]))
        .await;
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
        let pos = encoder.encode(&data).await.unwrap();
        object_writer.shutdown().await.unwrap();

        let mut reader = store.open(&path).await.unwrap();
        let decoder = BinaryDecoder::<Utf8Type>::new(&mut reader, pos, data.len());
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
}
