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

use std::any::TypeId;
use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom};

use arrow2::array::{Array, MutableArray, MutablePrimitiveArray, PrimitiveArray};
use arrow2::array::new_empty_array;
use arrow2::array::Int32Array;
use arrow2::compute::arithmetics::basic::sub_scalar;
use arrow2::compute::take::take;
use arrow2::types::NativeType;

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

impl<'a, R: Read + Seek, T: NativeType> Decoder<T> for PlainDecoder<'a, R> {
    type ArrowType = T;

    fn decode(&mut self, offset: i32, length: &Option<i32>) -> Result<Box<dyn Array>> {
        let read_len = length.unwrap_or((self.page_length - (offset as i64)) as i32) as usize;
        (*self.file).seek(SeekFrom::Start(self.position + offset as u64))?;
        let mut buffer = vec![T::default(); read_len];

        if TypeId::of::<byteorder::NativeEndian>() == TypeId::of::<byteorder::LittleEndian>() {
            let slice = bytemuck::cast_slice_mut(&mut buffer);
            (*self.file).read_exact(slice)?;
            let arr = PrimitiveArray::from_vec(buffer);
            Ok(Box::new(arr))
        } else {
            let mut slice = vec![0u8; read_len * std::mem::size_of::<T>()];
            (*self.file).read_exact(&mut slice)?;
            let chunks = slice.chunks_exact(std::mem::size_of::<T>());
            buffer
                .as_mut_slice()
                .iter_mut()
                .zip(chunks)
                .try_for_each(|(slot, chunk)| {
                    let a: T::Bytes = match chunk.try_into() {
                        Ok(a) => a,
                        Err(_) => unreachable!(),
                    };
                    *slot = T::from_le_bytes(a);
                    Ok::<(), Error>(())
                })?;
            let arr = PrimitiveArray::from_vec(buffer);
            Ok(Box::new(arr))
        }
    }

    fn take(&mut self, indices: &Int32Array) -> Result<Box<dyn Array>> {
        if indices.len() == 0 {
            return Ok(new_empty_array(T::PRIMITIVE.into()));
        }

        let start = indices.value(0);
        let length = indices.values().last().map(|i| i - start);
        let values = <PlainDecoder<'a, R> as Decoder<T>>::decode(self, start, &length)?;
        let reset_indices = sub_scalar(&indices, &start);

        let res = take(values.as_ref(), &reset_indices);
        match res {
            Ok(arr) => Ok(arr),
            Err(e) => Err(Error::new(ErrorKind::InvalidData, format!("Error take indices: {}", e)))
        }
    }

    fn value(&self, i: usize) -> Result<T> {
        todo!()
    }
}
