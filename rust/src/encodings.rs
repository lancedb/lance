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

use std::fmt;
use std::io::Result;

use arrow2::array::{Array, Int32Array};
use arrow2::types::NativeType;

pub mod plain;

#[derive(Debug)]
pub enum Encoding {
    Plain,
    VarBinary,
    Dictionary,
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plain => write!(f, "plain"),
            Self::VarBinary => write!(f, "var_binary"),
            Self::Dictionary => write!(f, "dictionary"),
        }
    }
}

/// Encoder.
pub trait Encoder {
    /// Write an Arrow array to the file, returns the position in the file.
    fn write(&mut self, array: &dyn Array) -> Result<i64>;
}

/// Decoder.
pub trait Decoder<T: NativeType> {
    type ArrowType;

    fn decode(&mut self, offset: i32, length: &Option<i32>) -> Result<Box<dyn Array>> ;

    fn take(&mut self, indices: &Int32Array) -> Result<Box<dyn Array>>;

    fn value(&self, i: usize) -> Result<T>;
}
