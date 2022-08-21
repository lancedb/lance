//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//! Plain encoding

use arrow::array::{ArrayDataBuilder, PrimitiveArray};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::ArrowPrimitiveType;
use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom};

use crate::encodings::Decoder;

/// Plain Decoder
pub struct PlainDecoder<'a, R: Read + Seek> {
    file: &'a mut R,
    position: u64,
    page_length: i64,
}

impl<'a, R: Read + Seek> PlainDecoder<'a, R> {
    pub fn new(file: &'a mut R, position: u64, page_length: i64) -> Self {
        PlainDecoder {
            file,
            position,
            page_length,
        }
    }
}

impl<'a, R: Read + Seek, T: ArrowPrimitiveType> Decoder<T> for PlainDecoder<'a, R> {
    type ArrowType = T;

    fn decode(&mut self, offset: i32, length: Option<i32>) -> Result<PrimitiveArray<T>> {
        let read_len = length.unwrap_or((self.page_length - (offset as i64)) as i32) as usize;
        (*self.file).seek(SeekFrom::Start(self.position + offset as u64))?;
        let mut mutable_buf = MutableBuffer::new(read_len * T::get_byte_width());
        (*self.file).read_exact(mutable_buf.as_slice_mut())?;
        let builder = ArrayDataBuilder::new(T::DATA_TYPE).buffers(vec![mutable_buf.into()]);
        let data = match builder.build() {
            Ok(d) => d,
            Err(e) => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid builder: {}", e),
                ))
            }
        };
        Ok(PrimitiveArray::<T>::from(data))
    }

    fn take(&mut self, indices: &dyn arrow::array::Array) -> Result<PrimitiveArray<T>> {
        todo!()
    }

    fn value(&self, i: usize) -> Result<T::Native> {
        todo!()
    }
}
