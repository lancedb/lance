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

//! Lance Encodings

use std::io;

use arrow::array::Array;
use arrow::array::Int32Array;

/// Encoder.
pub trait Encoder {
    fn write(&mut self, array: &dyn Array) -> Result<i64, io::Error>;
}

/// Decoder.
pub trait Decoder {
    fn take(&mut self, indices: &dyn Int32Array) -> Result<Array, io::Error>;
}