// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};

pub mod binary;
#[cfg(feature = "bitpacking")]
pub mod bitpacking;
pub mod block;
pub mod byte_stream_split;
pub mod constant;
pub mod delta;
pub mod dictionary;
pub mod fsst;
pub mod general;
pub mod packed;
pub mod range;
pub mod rle;
pub mod value;

pub(crate) fn checked_vec_capacity(
    num_values: u64,
    bytes_per_value: usize,
    label: &str,
) -> Result<usize> {
    let capacity = usize::try_from(num_values)
        .map_err(|_| Error::invalid_input(format!("{label} cardinality does not fit usize")))?;
    let output_bytes = capacity
        .checked_mul(bytes_per_value)
        .ok_or_else(|| Error::invalid_input(format!("{label} byte length overflows usize")))?;
    if output_bytes > isize::MAX as usize {
        return Err(Error::invalid_input(format!(
            "{label} byte length {output_bytes} exceeds isize::MAX"
        )));
    }
    Ok(capacity)
}

pub(crate) fn try_vec_with_capacity<T>(num_values: u64, label: &str) -> Result<Vec<T>> {
    let capacity = checked_vec_capacity(num_values, std::mem::size_of::<T>(), label)?;
    let output_bytes = capacity * std::mem::size_of::<T>();
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|error| {
        Error::invalid_input(format!(
            "{label} could not reserve {capacity} values ({output_bytes} bytes): {error}"
        ))
    })?;
    Ok(values)
}
