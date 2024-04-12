// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use arrow_array::UInt32Array;

pub mod encodings;
pub mod ffi;
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

impl ReadBatchParams {
    pub fn to_offsets(&self, base_offset: u32, length: u32, total_num_rows: u32) -> Vec<u32> {
        match self {
            Self::Indices(indices) => indices
                .slice(base_offset as usize, length as usize)
                .values()
                .iter()
                .copied()
                .collect(),
            Self::Range(r) => {
                (r.start as u32 + base_offset..r.start as u32 + base_offset + length).collect()
            }
            Self::RangeFull => (base_offset..base_offset + length).collect(),
            Self::RangeTo(r) => {
                let start = r.end as u32 - total_num_rows + base_offset;
                (start..start + length).collect()
            }
            Self::RangeFrom(r) => {
                (r.start as u32 + base_offset..r.start as u32 + base_offset + length).collect()
            }
        }
    }
}
