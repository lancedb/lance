// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Hamming distance.

use std::sync::Arc;

use arrow_array::{cast::AsArray, types::UInt8Type, Array, Float32Array};
use arrow_schema::DataType;

use crate::{Error, Result};

/// Hamming distance between two vectors.
#[inline]
pub fn hamming(x: &[u8], y: &[u8]) -> f32 {
    hamming_autovec::<64>(x, y)
}

#[inline]
fn hamming_autovec<const L: usize>(x: &[u8], y: &[u8]) -> f32 {
    let x_chunk = x.chunks_exact(L);
    let y_chunk = y.chunks_exact(L);
    let sum = x_chunk
        .remainder()
        .iter()
        .zip(y_chunk.remainder())
        .map(|(&a, &b)| (a ^ b).count_ones())
        .sum::<u32>();
    (sum + x_chunk
        .zip(y_chunk)
        .map(|(x, y)| {
            x.iter()
                .zip(y.iter())
                .map(|(&a, &b)| (a ^ b).count_ones())
                .sum::<u32>()
        })
        .sum::<u32>()) as f32
}

pub fn hamming_distance_batch<'a>(
    from: &'a [u8],
    to: &'a [u8],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a> {
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);
    Box::new(to.chunks_exact(dimension).map(|v| hamming(from, v)))
}

pub fn hamming_distance_arrow_batch(from: &dyn Array, to: &dyn Array) -> Result<Arc<Float32Array>> {
    let dists = match *from.data_type() {
        DataType::UInt8 => hamming_distance_batch(
            from.as_primitive::<UInt8Type>().values(),
            to.as_primitive::<UInt8Type>().values(),
            from.len(),
        ),
        _ => {
            return Err(Error::InvalidArgumentError(format!(
                "Unsupported data type: {:?}",
                from.data_type()
            )))
        }
    };

    Ok(Arc::new(Float32Array::new(
        dists.collect(),
        to.nulls().cloned(),
    )))
}

/// Scalar version of hamming distance. Used for benchmarks.
#[inline]
pub fn hamming_scalar(x: &[u8], y: &[u8]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi ^ yi).count_ones())
        .sum::<u32>() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming() {
        let x = vec![0b1101_1010, 0b1010_1010, 0b1010_1010];
        let y = vec![0b1101_1010, 0b1010_1010, 0b1010_1010];
        assert_eq!(hamming(&x, &y), 0.0);

        let y = vec![0b1101_1010, 0b1010_1010, 0b1010_1000];
        assert_eq!(hamming(&x, &y), 1.0);

        let y = vec![0b1101_1010, 0b1010_1010, 0b1010_1001];
        assert_eq!(hamming(&x, &y), 2.0);
    }
}
