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

use arrow_array::types::{
    ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow_array::{ArrayRef, DictionaryArray, PrimitiveArray};
use arrow_schema::DataType;
use async_trait::async_trait;

use crate::encodings::plain::PlainDecoder;
use crate::encodings::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::Error;

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

    async fn decode_index<T: ArrowDictionaryKeyType + Sync + Send>(&self) -> Result<ArrayRef> {
        let index_decoder = PlainDecoder::<T>::new(self.reader, self.position, self.length)?;
        let arr = index_decoder.decode().await?;
        let keys = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        Ok(Arc::new(DictionaryArray::try_new(keys, &self.value_arr)?))
    }
}

#[async_trait]
impl<'a> Decoder for DictionaryDecoder<'a> {
    async fn decode(&self) -> Result<ArrayRef> {
        use DataType::*;
        if let Dictionary(index_type, _) = &self.data_type {
            assert!(index_type.as_ref().is_dictionary_key_type());

            match index_type.as_ref() {
                Int8 => self.decode_index::<Int8Type>().await,
                Int16 => self.decode_index::<Int16Type>().await,
                Int32 => self.decode_index::<Int32Type>().await,
                Int64 => self.decode_index::<Int64Type>().await,
                UInt8 => self.decode_index::<UInt8Type>().await,
                UInt16 => self.decode_index::<UInt16Type>().await,
                UInt32 => self.decode_index::<UInt32Type>().await,
                UInt64 => self.decode_index::<UInt64Type>().await,
                _ => {
                    return Err(Error::Arrow(format!(
                        "Dictionary encoding does not support index type: {}",
                        index_type
                    )))
                }
            }
        } else {
            Err(Error::Arrow(format!(
                "Not a dictionary type: {}",
                self.data_type
            )))
        }
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
    use tokio::io::AsyncWriteExt;

    async fn test_dict_decoder_for_type<T: ArrowDictionaryKeyType>() {
        let values = vec!["a", "b", "b", "a", "c"];
        let arr: DictionaryArray<T> = values.into_iter().collect();

        let store = ObjectStore::new(":memory:").unwrap();
        let path = Path::from("/foo");

        let pos;
        {
            let (_, mut writer) = store.inner.put_multipart(&path).await.unwrap();
            let mut object_writer = ObjectWriter::new(writer.as_mut());
            let mut encoder = PlainEncoder::<T>::new(&mut object_writer);
            pos = encoder.encode(arr.keys()).await.unwrap();
            writer.shutdown().await.unwrap();
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
