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
use std::mem::size_of;
use std::ops::Range;

use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::Buffer;
use arrow_data::{ArrayDataBuilder, Bitmap};
use object_store::path::Path;
use object_store::ObjectStore;
use tokio::io::AsyncWriteExt;

use crate::io::object_writer::ObjectWriter;

const MAX_BLOCK_SIZE: usize = 1024 * 1024 * 8; // 8MB
const MIN_BLOCK_SIZE: usize = 512; // 512 bytes

/// Plain Encoder
pub struct PlainEncoder<'a, T: ArrowPrimitiveType> {
    writer: &'a mut ObjectWriter<'a>,
    nullable: bool,
    // Number of bytes.
    block_size: usize,

    phantom: PhantomData<T>,
}

fn get_rows_per_block<T>(block_size: usize) -> usize {
    assert!(
        block_size.is_power_of_two() && (MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&block_size),
    );

    // 1 bit for validity map, and 8 bits * size_of(T) for actual data.
    block_size * 8 / (1 + size_of::<T>() * 8)
}

impl<'a, T: ArrowPrimitiveType> PlainEncoder<'a, T> {
    pub fn new(
        writer: &'a mut ObjectWriter<'a>,
        nullable: bool,
        block_size: usize,
    ) -> PlainEncoder<'a, T> {
        assert!(block_size.is_power_of_two() && block_size >= 512);
        PlainEncoder {
            writer,
            nullable,
            block_size,
            phantom: PhantomData,
        }
    }

    /// Encode nullable data.
    async fn encode_nullables(&mut self, array: &dyn Array) -> Result<()> {
        // The data is nullable, we need to store its validity bitmap.
        let rows_per_page = get_rows_per_block::<T::Native>(self.block_size);
        let mut start = 0;
        while start < array.len() {
            let arr_block = array.slice(start, rows_per_page);
            let arr_data = arr_block.data();
            // Write null bitmap
            self.writer
                .write_all(
                    arr_data
                        .null_bitmap()
                        .unwrap_or(&Bitmap::new(rows_per_page))
                        .buffer()
                        .as_slice(),
                )
                .await?;
            // Write the array data
            self.writer
                .write_all(arr_data.buffers()[0].slice(arr_data.offset()).as_slice())
                .await?;
            start += rows_per_page;
        }
        Ok(())
    }

    /// Encode an array of a batch.
    /// Returns the offset of the metadata
    pub async fn encode(&mut self, array: &dyn Array) -> Result<usize> {
        let offset = self.writer.tell() as usize;
        if self.nullable {
            self.encode_nullables(array).await?;
        } else {
            let data = array.data().buffers()[0].as_slice();
            self.writer.write_all(data).await?;
        }

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
    /// If the code is nullable and have validity bitmap in it.
    nullable: bool,
    /// Number of records per block.
    block_size: usize,

    phantom: PhantomData<T>,
}

impl<'a, T: ArrowPrimitiveType> PlainDecoder<'a, T> {
    pub fn new(
        object_store: &'a dyn ObjectStore,
        path: &'a Path,
        position: usize,
        length: usize,
        nullable: bool,
        block_size: usize,
    ) -> Result<PlainDecoder<'a, T>> {
        assert!(block_size.is_power_of_two() && block_size >= 512);
        Ok(PlainDecoder {
            object_store,
            path,
            position,
            length,
            nullable,
            block_size,
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

    #[test]
    fn test_rows_per_block() {
        assert_eq!(504, get_rows_per_block::<i64>(4096));
        assert_eq!(504, get_rows_per_block::<u64>(4096));
        assert_eq!(504, get_rows_per_block::<f64>(4096));
    }

    #[tokio::test]
    async fn test_encode_decode_int_array() {
        let store = InMemory::new();
        let path = Path::from("/foo");
        let (_, mut writer) = store.put_multipart(&path).await.unwrap();

        let arr = Int32Array::from(Vec::from_iter(1..4096));
        {
            let mut object_writer = ObjectWriter::new(writer.as_mut());
            let mut encoder = PlainEncoder::<Int32Type>::new(&mut object_writer, false, 1024);

            assert_eq!(encoder.encode(&arr).await.unwrap(), 0);
        }
        writer.shutdown().await.unwrap();

        assert!(store.head(&Path::from("/foo")).await.unwrap().size > 0);
        let decoder =
            PlainDecoder::<Int32Type>::new(&store, &path, 0, arr.len(), false, 1024).unwrap();
        let read_arr = decoder.decode().await.unwrap();
        let expect_arr = as_primitive_array::<Int32Type>(read_arr.as_ref());
        assert_eq!(expect_arr, &arr);
    }

}
