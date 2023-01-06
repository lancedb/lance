//! Var-length Binary Encoding
//!

use std::io::Result;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow_array::types::{ByteArrayType, Int64Type};
use arrow_array::{Array, ArrayRef, GenericByteArray, Int32Array, Int64Array};
use arrow_data::ArrayDataBuilder;
use async_trait::async_trait;

use crate::encodings::Decoder;
use crate::io::object_reader::ObjectReader;

use super::plain::PlainDecoder;

pub struct BinaryEncoder {}

pub struct BinaryDecoder<'a, T: ByteArrayType> {
    reader: &'a ObjectReader<'a>,

    position: usize,

    length: usize,

    phantom: PhantomData<T>,
}

/// Var-length Binary Decoder
///
impl<'a, T: ByteArrayType> BinaryDecoder<'a, T> {
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
        let position_decoder =
            PlainDecoder::<Int64Type>::new(self.reader, self.position, self.length + 1)?;
        let positions = position_decoder.decode().await?;
        let int64_positions = positions.as_any().downcast_ref::<Int64Array>().unwrap();

        let start_position = int64_positions.value(0);
        let offset_arr = Int32Array::from(
            int64_positions
                .iter()
                .map(|v| v.map(|o| (o - start_position) as i32))
                .collect::<Vec<_>>(),
        );
        let offset_data = offset_arr.into_data();

        let read_len = int64_positions.value(int64_positions.len() - 1) - start_position;
        let bytes = self
            .reader
            .get_range(start_position as usize..(start_position + read_len) as usize)
            .await?;
        let array_data = ArrayDataBuilder::new(T::DATA_TYPE)
            .len(self.length)
            .null_count(0)
            .add_buffer(offset_data.buffers()[0].clone())
            .add_buffer(bytes.into())
            .build()
            .unwrap();

        Ok(Arc::new(GenericByteArray::<T>::from(array_data)))
    }
}
