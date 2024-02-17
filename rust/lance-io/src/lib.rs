// Copyright 2024 Lance Developers.
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
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use arrow_array::UInt32Array;

pub mod encodings;
pub mod local;
pub mod object_reader;
pub mod object_store;
pub mod object_writer;
pub mod scheduler;
pub mod stream;
pub mod traits;
pub mod utils;

/// Defines a selection of rows to read from a file/batch
#[derive(Debug, Clone)]
pub enum ReadBatchParams {
    /// Select a contiguous range of rows
    Range(Range<usize>),
    /// Select all rows (this is the default)
    RangeFull,
    /// Select all rows up to a given index
    RangeTo(RangeTo<usize>),
    /// Select all rows starting at a given index
    RangeFrom(RangeFrom<usize>),
    /// Select scattered non-contiguous rows
    Indices(UInt32Array),
}

impl Default for ReadBatchParams {
    fn default() -> Self {
        // Default of ReadBatchParams is reading the full batch.
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
