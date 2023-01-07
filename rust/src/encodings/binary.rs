//! Var-length Binary Encoding
//!

use std::marker::PhantomData;
use std::sync::Arc;

use arrow_array::types::{ByteArrayType, Int64Type};
use arrow_array::{Array, ArrayRef, GenericByteArray, Int32Array, Int64Array};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use async_trait::async_trait;

use super::plain::PlainDecoder;
use crate::encodings::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;

pub struct BinaryEncoder {}

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
        let position_decoder =
            PlainDecoder::new(self.reader, &DataType::Int64, self.position, self.length + 1)?;
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
            .build()?;

        Ok(Arc::new(GenericByteArray::<T>::from(array_data)))
    }
}
