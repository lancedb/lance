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

use arrow::array::{make_array, new_empty_array, ArrayDataBuilder, ArrayRef, Int32Array};
use arrow::buffer::MutableBuffer;
use arrow::compute::{subtract_scalar_dyn, take};
use arrow::datatypes::{ArrowPrimitiveType, Int32Type};
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
    fn decode(&mut self, offset: i32, length: &Option<i32>) -> Result<ArrayRef> {

        let read_len = length.unwrap_or((self.page_length - (offset as i64)) as i32) as usize;
        (*self.file).seek(SeekFrom::Start(self.position + offset as u64))?;
        // let mut mutable_buf = Buffer::new(read_len * T::get_byte_width());
        let mut buf = vec![0u8; read_len * T::get_byte_width()];
        (*self.file).read_exact(& mut buf)?;
        todo!()
    }
}

impl<'a, R: Read + Seek, T: ArrowPrimitiveType> Decoder<T> for PlainDecoder<'a, R> {
    type ArrowType = T;

    fn decode(&mut self, offset: i32, length: &Option<i32>) -> Result<ArrayRef> {
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
        Ok(make_array(data))
    }

    fn take(&mut self, indices: &Int32Array) -> Result<ArrayRef> {
        if indices.is_empty() {
            return Ok(new_empty_array(&T::DATA_TYPE));
        }
        let start = indices.value(0);
        let length = indices.values().last().map(|i| i - start);
        // Not sure why it needs cast.
        let values = <PlainDecoder<'a, R> as Decoder<T>>::decode(self, start, &length)?;
        let reset_indices = match subtract_scalar_dyn::<Int32Type>(&values, start) {
            Ok(arr) => arr,
            Err(e) => return Err(Error::new(ErrorKind::InvalidData, e.to_string())),
        };
        match take(
            &values,
            reset_indices.as_any().downcast_ref::<Int32Array>().unwrap(),
            None,
        ) {
            Ok(arr) => Ok(arr),
            Err(e) => Err(Error::new(ErrorKind::InvalidData, format!("Error take indices: {}", e)))
        }
    }

    fn value(&self, i: usize) -> Result<T::Native> {
        todo!()
    }
}
