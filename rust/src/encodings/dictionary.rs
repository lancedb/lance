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

//! Dictinary encoding.
//!

use std::sync::Arc;

use arrow_array::cast::{as_dictionary_array, as_primitive_array};
use arrow_array::types::{
    ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow_array::{Array, ArrayRef, DictionaryArray, PrimitiveArray, UInt32Array};
use arrow_schema::DataType;
use async_trait::async_trait;

use super::plain::PlainEncoder;
use super::AsyncIndex;
use crate::encodings::plain::PlainDecoder;
use crate::encodings::{Decoder, Encoder};
use crate::error::Result;
use crate::io::{object_reader::ObjectReader, object_writer::ObjectWriter, ReadBatchParams};
use crate::Error;

/// Encoder for Dictionary encoding.
pub struct DictionaryEncoder<'a> {
    writer: &'a mut ObjectWriter,
    key_type: &'a DataType,
}

impl<'a> DictionaryEncoder<'a> {
    pub fn new(writer: &'a mut ObjectWriter, key_type: &'a DataType) -> Self {
        Self { writer, key_type }
    }

    async fn write_typed_array<T: ArrowDictionaryKeyType>(
        &mut self,
        arr: &dyn Array,
    ) -> Result<usize> {
        let pos = self.writer.tell();
        let dict_arr = as_dictionary_array::<T>(arr);

        let mut plain_encoder = PlainEncoder::new(self.writer, dict_arr.data_type());
        plain_encoder.encode(dict_arr.keys()).await?;
        Ok(pos)
    }
}

#[async_trait]
impl<'a> Encoder for DictionaryEncoder<'a> {
    async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        use DataType::*;

        match self.key_type {
            UInt8 => self.write_typed_array::<UInt8Type>(array).await,
            UInt16 => self.write_typed_array::<UInt16Type>(array).await,
            UInt32 => self.write_typed_array::<UInt32Type>(array).await,
            UInt64 => self.write_typed_array::<UInt64Type>(array).await,
            Int8 => self.write_typed_array::<Int8Type>(array).await,
            Int16 => self.write_typed_array::<Int16Type>(array).await,
            Int32 => self.write_typed_array::<Int32Type>(array).await,
            Int64 => self.write_typed_array::<Int64Type>(array).await,
            _ => Err(Error::Schema(format!(
                "DictionaryEncoder: unsurpported key type: {:?}",
                self.key_type
            ))),
        }
    }
}

/// Decoder for Dictionary encoding.
#[derive(Debug)]
pub struct DictionaryDecoder<'a> {
    reader: &'a ObjectReader<'a>,
    /// The start position of the batch in the file.
    position: usize,
    /// Number of the rows in this batch.
    length: usize,
    /// The dictionary data type
    data_type: &'a DataType,
    /// Value array,
    value_arr: ArrayRef,
}

impl<'a> DictionaryDecoder<'a> {
    pub fn new(
        reader: &'a ObjectReader<'a>,
        position: usize,
        length: usize,
        data_type: &'a DataType,
        value_arr: ArrayRef,
    ) -> Self {
        assert!(matches!(data_type, DataType::Dictionary(_, _)));
        Self {
            reader,
            position,
            length,
            data_type,
            value_arr,
        }
    }

    async fn decode_index(&self, index_type: &DataType) -> Result<ArrayRef> {
        let index_decoder = PlainDecoder::new(self.reader, index_type, self.position, self.length)?;
        let index_array = index_decoder.decode().await?;

        match index_type {
            DataType::Int8 => self.make_dict_array::<Int8Type>(index_array).await,
            DataType::Int16 => self.make_dict_array::<Int16Type>(index_array).await,
            DataType::Int32 => self.make_dict_array::<Int32Type>(index_array).await,
            DataType::Int64 => self.make_dict_array::<Int64Type>(index_array).await,
            DataType::UInt8 => self.make_dict_array::<UInt8Type>(index_array).await,
            DataType::UInt16 => self.make_dict_array::<UInt16Type>(index_array).await,
            DataType::UInt32 => self.make_dict_array::<UInt32Type>(index_array).await,
            DataType::UInt64 => self.make_dict_array::<UInt64Type>(index_array).await,
            _ => {
                return Err(Error::Arrow(format!(
                    "Dictionary encoding does not support index type: {}",
                    index_type
                )))
            }
        }
    }

    async fn make_dict_array<T: ArrowDictionaryKeyType + Sync + Send>(
        &self,
        index_array: ArrayRef,
    ) -> Result<ArrayRef> {
        let keys: &PrimitiveArray<T> = as_primitive_array(index_array.as_ref());
        Ok(Arc::new(DictionaryArray::try_new(keys, &self.value_arr)?))
    }
}

#[async_trait]
impl<'a> Decoder for DictionaryDecoder<'a> {
    async fn decode(&self) -> Result<ArrayRef> {
        use DataType::*;
        if let Dictionary(index_type, _) = &self.data_type {
            assert!(index_type.as_ref().is_dictionary_key_type());
            self.decode_index(index_type).await
        } else {
            Err(Error::Arrow(format!(
                "Not a dictionary type: {}",
                self.data_type
            )))
        }
    }

    async fn take(&self, indices: &UInt32Array) -> Result<ArrayRef> {
        todo!()
    }
}

#[async_trait]
impl<'a> AsyncIndex<usize> for DictionaryDecoder<'a> {
    type Output = Result<ArrayRef>;

    async fn get(&self, _index: usize) -> Self::Output {
        todo!()
    }
}

#[async_trait]
impl<'a> AsyncIndex<ReadBatchParams> for DictionaryDecoder<'a> {
    type Output = Result<ArrayRef>;

    async fn get(&self, params: ReadBatchParams) -> Self::Output {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        encodings::plain::PlainEncoder,
        io::{object_writer::ObjectWriter, ObjectStore},
    };
    use arrow_array::Array;
    use object_store::path::Path;

    async fn test_dict_decoder_for_type<T: ArrowDictionaryKeyType>() {
        let values = vec!["a", "b", "b", "a", "c"];
        let arr: DictionaryArray<T> = values.into_iter().collect();

        let store = ObjectStore::new(":memory:").unwrap();
        let path = Path::from("/foo");

        let pos;
        {
            let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
            let mut encoder = PlainEncoder::new(&mut object_writer, arr.keys().data_type());
            pos = encoder.encode(arr.keys()).await.unwrap();
            object_writer.shutdown().await.unwrap();
        }

        let reader = store.open(&path).await.unwrap();
        let decoder = DictionaryDecoder::new(
            &reader,
            pos,
            arr.len(),
            arr.data_type(),
            arr.values().clone(),
        );

        let expected_data = decoder.decode().await.unwrap();
        assert_eq!(
            &arr,
            expected_data
                .as_any()
                .downcast_ref::<DictionaryArray<T>>()
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_dict_decoder() {
        test_dict_decoder_for_type::<Int8Type>().await;
        test_dict_decoder_for_type::<Int16Type>().await;
        test_dict_decoder_for_type::<Int32Type>().await;
        test_dict_decoder_for_type::<Int64Type>().await;

        test_dict_decoder_for_type::<UInt8Type>().await;
        test_dict_decoder_for_type::<UInt16Type>().await;
        test_dict_decoder_for_type::<UInt32Type>().await;
        test_dict_decoder_for_type::<UInt64Type>().await;
    }
}
