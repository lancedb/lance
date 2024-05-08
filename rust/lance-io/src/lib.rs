// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use arrow::datatypes::UInt32Type;
use arrow_array::{PrimitiveArray, UInt32Array};
use snafu::{location, Location};

use lance_core::{Error, Result};

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
    /// Validate that the selection is valid given the length of the batch
    pub fn valid_given_len(&self, len: usize) -> bool {
        match self {
            Self::Indices(indices) => indices.iter().all(|i| i.unwrap_or(0) < len as u32),
            Self::Range(r) => r.start < len && r.end <= len,
            Self::RangeFull => true,
            Self::RangeTo(r) => r.end <= len,
            Self::RangeFrom(r) => r.start < len,
        }
    }

    /// Slice the selection
    ///
    /// For example, given ReadBatchParams::RangeFull and slice(10, 20), the output will be
    /// ReadBatchParams::Range(10..20)
    ///
    /// Given ReadBatchParams::Range(10..20) and slice(5, 3), the output will be
    /// ReadBatchParams::Range(15..18)
    ///
    /// Given ReadBatchParams::RangeTo(20) and slice(10, 5), the output will be
    /// ReadBatchParams::Range(10..15)
    ///
    /// Given ReadBatchParams::RangeFrom(20) and slice(10, 5), the output will be
    /// ReadBatchParams::Range(30..35)
    ///
    /// Given ReadBatchParams::Indices([1, 3, 5, 7, 9]) and slice(1, 3), the output will be
    /// ReadBatchParams::Indices([3, 5, 7])
    ///
    /// You cannot slice beyond the bounds of the selection and an attempt to do so will
    /// return an error.
    pub fn slice(&self, start: usize, length: usize) -> Result<Self> {
        let out_of_bounds = |size: usize| {
            Err(Error::InvalidInput {
                source: format!(
                    "Cannot slice from {} with length {} given a selection of size {}",
                    start, length, size
                )
                .into(),
                location: location!(),
            })
        };

        match self {
            Self::Indices(indices) => {
                if start + length > indices.len() {
                    return out_of_bounds(indices.len());
                }
                Ok(Self::Indices(indices.slice(start, length)))
            }
            Self::Range(r) => {
                if (r.start + start + length) > r.end {
                    return out_of_bounds(r.end - r.start);
                }
                Ok(Self::Range((r.start + start)..(r.start + start + length)))
            }
            Self::RangeFull => Ok(Self::Range(start..(start + length))),
            Self::RangeTo(range) => {
                if start + length > range.end {
                    return out_of_bounds(range.end);
                }
                Ok(Self::Range(start..(start + length)))
            }
            Self::RangeFrom(r) => {
                // No way to validate out_of_bounds, assume caller will do so
                Ok(Self::Range((r.start + start)..(r.start + start + length)))
            }
        }
    }

    /// Convert a read range into a vector of row offsets
    ///
    /// RangeFull and RangeFrom are unbounded and cannot be converted into row offsets
    /// and any attempt to do so will return an error.  Call slice first
    pub fn to_offsets(&self) -> Result<PrimitiveArray<UInt32Type>> {
        match self {
            Self::Indices(indices) => Ok(indices.clone()),
            Self::Range(r) => Ok(UInt32Array::from(Vec::from_iter(
                r.start as u32..r.end as u32,
            ))),
            Self::RangeFull => Err(Error::invalid_input(
                "cannot materialize RangeFull",
                location!(),
            )),
            Self::RangeTo(r) => Ok(UInt32Array::from(Vec::from_iter(0..r.end as u32))),
            Self::RangeFrom(_) => Err(Error::invalid_input(
                "cannot materialize RangeFrom",
                location!(),
            )),
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
            let offsets = params
                .slice(base_offset, length)
                .unwrap()
                .to_offsets()
                .unwrap();
            let expected = UInt32Array::from(expected);
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

        let check_error = |params: ReadBatchParams, base_offset, length| {
            assert!(params.slice(base_offset, length).is_err());
        };

        check_error(ReadBatchParams::Indices(UInt32Array::from(vec![1])), 0, 2);
        check_error(ReadBatchParams::Indices(UInt32Array::from(vec![1])), 1, 1);
        check_error(ReadBatchParams::Range(0..10), 5, 6);
        check_error(ReadBatchParams::RangeTo(RangeTo { end: 10 }), 5, 6);

        assert!(ReadBatchParams::RangeFull.to_offsets().is_err());
        assert!(ReadBatchParams::RangeFrom(RangeFrom { start: 10 })
            .to_offsets()
            .is_err());
    }
}
