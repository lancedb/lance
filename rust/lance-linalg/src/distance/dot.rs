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
use std::ops::{AddAssign, Neg};
use std::sync::Arc;

use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use half::{bf16, f16};
use lance_arrow::bfloat16::BFloat16Type;
use lance_arrow::{ArrowFloatType, FloatToArrayType};
use num_traits::real::Real;

/// Default implementation of dot product.
#[inline]
pub fn dot<T: Real + Sum + AddAssign>(from: &[T], to: &[T]) -> T {
    const LANES: usize = 16;
    let x_chunks = to.chunks_exact(LANES);
    let y_chunks = from.chunks_exact(LANES);
    let mut sum = if x_chunks.remainder().is_empty() {
        T::zero()
    } else {
        x_chunks
            .remainder()
            .iter()
            .zip(y_chunks.remainder().iter())
            .map(|(&x, &y)| x * y)
            .sum::<T>()
    };
    // Use known size to allow LLVM to kick in auto-vectorization.
    let mut sums = [T::zero(); LANES];
    for (x, y) in x_chunks.zip(y_chunks) {
        for i in (0..16) {
            sums[i] += x[i] * y[i];
        }
    }
    for i in 0..4 {
        sum = sum + (sums[i] + sums[i + 4]) + (sums[i + 8] + sums[i + 12]);
    }
    sum
}

/// Dot product
pub trait Dot: ArrowFloatType {
    type Output: Neg<Output = Self::Native>;

    /// Dot product.
    fn dot(x: &[Self::Native], y: &[Self::Native]) -> Self::Output;
}

impl Dot for BFloat16Type {
    type Output = bf16;

    #[inline]
    fn dot(x: &[bf16], y: &[bf16]) -> bf16 {
        dot(x, y)
    }
}

impl Dot for Float16Type {
    type Output = f16;

    #[inline]
    fn dot(x: &[f16], y: &[f16]) -> f16 {
        dot(x, y)
    }
}

impl Dot for Float32Type {
    type Output = f32;

    // #[inline]
    fn dot(x: &[f32], other: &[f32]) -> f32 {
        let mut sum16 = f32x16::zeros();
        x.chunks_exact(128)
            .zip(other.chunks_exact(16))
            .for_each(|(x, y)| unsafe {
                let x1 = f32x16::load_unaligned(x.as_ptr());
                let x2 = f32x16::load_unaligned(x.as_ptr().add(16));
                let x3 = f32x16::load_unaligned(x.as_ptr().add(32));
                let x4 = f32x16::load_unaligned(x.as_ptr().add(48));
                let x5 = f32x16::load_unaligned(x.as_ptr().add(64));
                let x6 = f32x16::load_unaligned(x.as_ptr().add(80));
                let x7 = f32x16::load_unaligned(x.as_ptr().add(96));
                let x8 = f32x16::load_unaligned(x.as_ptr().add(112));

                let y1 = f32x16::load_unaligned(y.as_ptr());
                let y2 = f32x16::load_unaligned(y.as_ptr().add(16));
                let y3 = f32x16::load_unaligned(y.as_ptr().add(32));
                let y4 = f32x16::load_unaligned(y.as_ptr().add(48));
                let y5 = f32x16::load_unaligned(y.as_ptr().add(64));
                let y6 = f32x16::load_unaligned(y.as_ptr().add(80));
                let y7 = f32x16::load_unaligned(y.as_ptr().add(96));
                let y8 = f32x16::load_unaligned(y.as_ptr().add(112));

                sum16 += ((x1 * y1 + x2 * y2)
                    + (x3 * y3 + x4 * y4))
                    + ((x5 * y5 + x6 * y6)
                    + (x7 * y7 + x8 * y8));
            });
        sum16.reduce_sum()

        // let aligned_len = dim / 8 * 8;
        // let mut sum8 = f32x8::zeros();
        // for i in (unrolling_len..aligned_len).step_by(8) {
        //     unsafe {
        //         let x = f32x8::load_unaligned(x.as_ptr().add(i));
        //         let y = f32x8::load_unaligned(other.as_ptr().add(i));
        //         sum8.multiply_add(x, y);
        //     }
        // }

        // let mut sum = sum16.reduce_sum() + sum8.reduce_sum();
        // if aligned_len < dim {
        //     sum += dot(&x[aligned_len..], &other[aligned_len..]);
        // }

        // sum
    }
}

impl Dot for Float64Type {
    type Output = f64;

    #[inline]
    fn dot(x: &[f64], y: &[f64]) -> f64 {
        dot(x, y)
    }
}

/// Negative dot product, to present the relative order of dot distance.
pub fn dot_distance_batch<'a, T: FloatToArrayType>(
    from: &'a [T],
    to: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = T> + 'a>
where
    T::ArrowType: Dot,
{
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
