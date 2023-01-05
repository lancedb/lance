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
//! it stores the array directly in the file. Therefore, it offers O(1) read
//! access.

use std::io::{ErrorKind, Result};
use std::marker::PhantomData;
use std::ops::Range;

use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use object_store::path::Path;
use object_store::ObjectStore;
use tokio::io::AsyncWriteExt;

use crate::io::object_writer::ObjectWriter;

/// Encoder for plain encoding.
///
pub struct PlainEncoder<'a, T: ArrowPrimitiveType> {
    writer: &'a mut ObjectWriter<'a>,
    phantom: PhantomData<T>,
}

impl<'a, T: ArrowPrimitiveType> PlainEncoder<'a, T> {
    pub fn new(writer: &'a mut ObjectWriter<'a>) -> PlainEncoder<'a, T> {
        PlainEncoder {
            writer,
            phantom: PhantomData,
        }
    }

    /// Encode an array of a batch.
    /// Returns the offset of the metadata
    pub async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        let offset = self.writer.tell() as usize;

        let data = array.data().buffers()[0].as_slice();
        self.writer.write_all(data).await?;

        Ok(offset)
    }
}

/// Decoder for plain encoding.
pub struct PlainDecoder<'a, T: ArrowPrimitiveType> {
    object_store: &'a dyn ObjectStore,
    /// File path.
    path: &'a Path,
    /// The start position of the batch in the file.
    position: usize,
    /// Number of the rows in this batch.
    length: usize,

    phantom: PhantomData<T>,
}

impl<'a, T: ArrowPrimitiveType> PlainDecoder<'a, T> {
    pub fn new(
        object_store: &'a dyn ObjectStore,
        path: &'a Path,
        position: usize,
        length: usize,
    ) -> Result<PlainDecoder<'a, T>> {
        Ok(PlainDecoder {
            object_store,
            path,
            position,
            length,
            phantom: PhantomData,
        })
    }

    pub async fn at(&self, _idx: usize) -> Result<Option<T>> {
        todo!()
    }

    pub async fn decode(&self) -> Result<Box<dyn Array>> {
        let array_bytes = T::get_byte_width() * self.length;
        let range = Range {
            start: self.position,
            end: self.position + array_bytes,
        };

        // if self.nullable {
        //     Err("b");
        // } else {
        let data = self.object_store.get_range(self.path, range).await?;
        // A memory copy occurs here.
        // TODO: zero-copy
        // https://docs.rs/arrow-buffer/29.0.0/arrow_buffer/struct.Buffer.html#method.from_custom_allocation
        let buf: Buffer = data.into();
        let array_data = match ArrayDataBuilder::new(T::DATA_TYPE)
            .len(self.length)
            .null_count(0)
            .add_buffer(buf)
            .build()
        {
            Ok(d) => d,
            Err(e) => return Err(std::io::Error::new(ErrorKind::InvalidData, e.to_string())),
        };
        Ok(Box::new(PrimitiveArray::<T>::from(array_data)))
    }
    // }
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::as_primitive_array;
    use arrow_array::types::Int32Type;
    use arrow_array::Int32Array;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::ObjectStore;
    use tokio::io::AsyncWriteExt;

    use crate::io::object_writer::ObjectWriter;

    use super::*;

    #[tokio::test]
    async fn test_encode_decode_int_array() {
        let store = InMemory::new();
        let path = Path::from("/foo");
        let (_, mut writer) = store.put_multipart(&path).await.unwrap();

        let arr = Int32Array::from(Vec::from_iter(1..4096));
        {
            let mut object_writer = ObjectWriter::new(writer.as_mut());
            let mut encoder = PlainEncoder::<Int32Type>::new(&mut object_writer);

            assert_eq!(encoder.encode(&arr).await.unwrap(), 0);
        }
        writer.shutdown().await.unwrap();

        assert!(store.head(&Path::from("/foo")).await.unwrap().size > 0);
        let decoder = PlainDecoder::<Int32Type>::new(&store, &path, 0, arr.len()).unwrap();
        let read_arr = decoder.decode().await.unwrap();
        let expect_arr = as_primitive_array::<Int32Type>(read_arr.as_ref());
        assert_eq!(expect_arr, &arr);
    }
}
