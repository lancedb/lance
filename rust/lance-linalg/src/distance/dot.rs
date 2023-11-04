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

//! Dot product.

use std::iter::Sum;
use std::ops::Neg;
use std::sync::Arc;

use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use half::{bf16, f16};
use lance_arrow::bfloat16::BFloat16Type;
use lance_arrow::{ArrowFloatType, FloatToArrayType};
use num_traits::real::Real;

use crate::simd::{
    f32::{f32x16, f32x8},
    FloatSimd, SIMD,
};

/// Naive implementation of dot product.
#[inline]
pub fn dot<T: Real + Sum>(from: &[T], to: &[T]) -> T {
    from.iter().zip(to.iter()).map(|(x, y)| x.mul(*y)).sum()
}

/// Dot product
pub trait Dot: ArrowFloatType {
    type Output: Neg<Output = Self::Native>;

    /// Dot product.
    fn dot(x: &[Self::Native], y: &[Self::Native]) -> Self::Output;
}

impl Dot for BFloat16Type {
    type Output = bf16;

    fn dot(x: &[bf16], y: &[bf16]) -> bf16 {
        dot(x, y)
    }
}

impl Dot for Float16Type {
    type Output = f16;

    fn dot(x: &[f16], y: &[f16]) -> f16 {
        dot(x, y)
    }
}

impl Dot for Float32Type {
    type Output = f32;

    fn dot(x: &[f32], other: &[f32]) -> f32 {
        let dim = x.len();
        let unrolling_len = dim / 16 * 16;
        let mut sum16 = f32x16::zeros();
        for i in (0..unrolling_len).step_by(16) {
            unsafe {
                let x = f32x16::load_unaligned(x.as_ptr().add(i));
                let y = f32x16::load_unaligned(other.as_ptr().add(i));
                sum16.multiply_add(x, y);
            }
        }

        let aligned_len = dim / 8 * 8;
        let mut sum8 = f32x8::zeros();
        for i in (unrolling_len..aligned_len).step_by(8) {
            unsafe {
                let x = f32x8::load_unaligned(x.as_ptr().add(i));
                let y = f32x8::load_unaligned(other.as_ptr().add(i));
                sum8.multiply_add(x, y);
            }
        }

        let mut sum = sum16.reduce_sum() + sum8.reduce_sum();
        if aligned_len < dim {
            sum += dot(&x[aligned_len..], &other[aligned_len..]);
        }

        sum
    }
}

impl Dot for Float64Type {
    type Output = f64;

    fn dot(x: &[f64], y: &[f64]) -> f64 {
        dot(x, y)
    }
}

/// Negative dot product, to present the relative order of dot distance.
pub fn dot_distance_batch<'a>(
    from: &'a [f32],
    to: &'a [f32],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a> {
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);

    Box::new(to.chunks_exact(dimension).map(|v| dot_distance(from, v)))
}

/// Compute negative dot product distance between a vector and a batch of vectors.
///
/// Null buffer of `to` is propagated to the returned array.
///
/// Parameters
///
/// - `from`: the vector to compute distance from.
/// - `to`: a list of vectors to compute distance to.
///
/// # Panics
///
/// Panics if the length of `from` is not equal to the dimension (value length) of `to`.
pub fn dot_distance_arrow_batch(from: &[f32], to: &FixedSizeListArray) -> Arc<Float32Array> {
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values = to.values().as_primitive::<Float32Type>().values();
    let dists = to_values
        .chunks_exact(dimension)
        .map(|v| dot_distance(from, v));

    Arc::new(Float32Array::new(dists.collect(), to.nulls().cloned()))
}

/// Negative dot distance.
#[inline]
pub fn dot_distance<T: FloatToArrayType + Neg<Output = T>>(from: &[T], to: &[T]) -> T
where
    T::ArrowType: Dot,
{
    T::ArrowType::dot(from, to).neg()
}

#[cfg(test)]
mod tests {

    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn test_dot() {
        let x: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..120).map(|v| v as f32).collect();

        assert_eq!(Float32Type::dot(&x, &y), dot(&x, &y));

        let x: Vec<f16> = (0..20).map(|v| f16::from_i32(v).unwrap()).collect();
        let y: Vec<f16> = (100..120).map(|v| f16::from_i32(v).unwrap()).collect();
        assert_eq!(Float16Type::dot(&x, &y), dot(&x, &y));

        let x: Vec<f64> = (20..40).map(|v| f64::from_i32(v).unwrap()).collect();
        let y: Vec<f64> = (120..140).map(|v| f64::from_i32(v).unwrap()).collect();
        assert_eq!(Float64Type::dot(&x, &y), dot(&x, &y));
    }
}
