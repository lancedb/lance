// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! IO utilities for Lance Columnar Format.
//!

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use arrow_array::UInt32Array;

pub mod local;
pub mod object_reader;
mod stream;
mod traits;
mod utils;

pub use object_reader::CloudObjectReader;
pub use stream::{RecordBatchStream, RecordBatchStreamAdapter};
pub use traits::*;
pub use utils::*;

/// Parameter to be used to read a batch.
#[derive(Debug, Clone)]
pub enum ReadBatchParams {
    Range(Range<usize>),

    RangeFull,

    RangeTo(RangeTo<usize>),

    RangeFrom(RangeFrom<usize>),

    Indices(UInt32Array),
}

/// Default of ReadBatchParams is reading the full batch.
impl Default for ReadBatchParams {
    fn default() -> Self {
        Self::RangeFull
    }
}

impl From<&[u32]> for ReadBatchParams {
    fn from(value: &[u32]) -> Self {
        Self::Indices(UInt32Array::from_iter_values(value.iter().copied()))
    }
}

impl From<UInt32Array> for ReadBatchParams {
    fn from(value: UInt32Array) -> Self {
        Self::Indices(value)
    }
}

impl From<RangeFull> for ReadBatchParams {
    fn from(_: RangeFull) -> Self {
        Self::RangeFull
    }
}

impl From<Range<usize>> for ReadBatchParams {
    fn from(r: Range<usize>) -> Self {
        Self::Range(r)
    }
}

impl From<RangeTo<usize>> for ReadBatchParams {
    fn from(r: RangeTo<usize>) -> Self {
        Self::RangeTo(r)
    }
}

impl From<RangeFrom<usize>> for ReadBatchParams {
    fn from(r: RangeFrom<usize>) -> Self {
        Self::RangeFrom(r)
    }
}

impl From<&Self> for ReadBatchParams {
    fn from(params: &Self) -> Self {
        params.clone()
    }
}
