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
    /// Convert a read range into a vector of row offsets
    ///
    /// Can take in a `base_offset` and `length` to allow paging through the range
    ///
    /// For example, if the range is `Range(10..20)` and `base_offset` is 5 and `length` is 3,
    /// the output will be `15..18`
    pub fn to_offsets(&self, base_offset: u32, length: u32) -> Vec<u32> {
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
            Self::RangeTo(_) => (base_offset..base_offset + length).collect(),
            Self::RangeFrom(r) => {
                (r.start as u32 + base_offset..r.start as u32 + base_offset + length).collect()
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::ops::{RangeFrom, RangeTo};

    use arrow_array::UInt32Array;

    use crate::ReadBatchParams;

    #[test]
    fn test_params_to_offsets() {
        let check = |params: ReadBatchParams, base_offset, length, expected: Vec<u32>| {
            let offsets = params.to_offsets(base_offset, length);
            assert_eq!(offsets, expected);
        };

        check(ReadBatchParams::RangeFull, 0, 100, (0..100).collect());
        check(ReadBatchParams::RangeFull, 50, 100, (50..150).collect());
        check(
            ReadBatchParams::RangeFrom(RangeFrom { start: 500 }),
            0,
            100,
            (500..600).collect(),
        );
        check(
            ReadBatchParams::RangeFrom(RangeFrom { start: 500 }),
            100,
            100,
            (600..700).collect(),
        );
        check(
            ReadBatchParams::RangeTo(RangeTo { end: 800 }),
            0,
            100,
            (0..100).collect(),
        );
        check(
            ReadBatchParams::RangeTo(RangeTo { end: 800 }),
            200,
            100,
            (200..300).collect(),
        );
        check(
            ReadBatchParams::Indices(UInt32Array::from(vec![1, 3, 5, 7, 9])),
            0,
            2,
            vec![1, 3],
        );
        check(
            ReadBatchParams::Indices(UInt32Array::from(vec![1, 3, 5, 7, 9])),
            2,
            2,
            vec![5, 7],
        );
    }
}
