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
use std::sync::Arc;

use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use half::{bf16, f16};
use num_traits::real::Real;

/// Naive implementation of dot product.
#[inline]
pub fn dot<T: Real + Sum>(from: &[T], to: &[T]) -> T {
    from.iter().zip(to.iter()).map(|(x, y)| x.mul(*y)).sum()
}

/// Dot product
pub trait Dot {
    type Output;

    /// Dot product.
    fn dot(&self, other: &Self) -> Self::Output;
}

impl Dot for [bf16] {
    type Output = bf16;

    fn dot(&self, other: &[bf16]) -> bf16 {
        dot(self, other)
    }
}

impl Dot for [f16] {
    type Output = f16;

    fn dot(&self, other: &[f16]) -> f16 {
        dot(self, other)
    }
}

impl Dot for [f32] {
    type Output = f32;

    fn dot(&self, other: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") {
                return x86_64::avx::dot_f32(self, other);
            }
        }

        dot(self, other)
    }
}

impl Dot for [f64] {
    type Output = f64;

    fn dot(&self, other: &[f64]) -> f64 {
        dot(self, other)
    }
}

/// Negative dot product, to present the relative order of dot distance.
pub fn dot_distance_batch(from: &[f32], to: &[f32], dimension: usize) -> Arc<Float32Array> {
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);

    let dists = to.chunks_exact(dimension).map(|v| dot_distance(from, v));

    Arc::new(Float32Array::new(dists.collect(), None))
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
pub fn dot_distance(from: &[f32], to: &[f32]) -> f32 {
    -from.dot(to)
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {

    pub mod avx {
        use crate::distance::x86_64::avx::*;
        use std::arch::x86_64::*;

        #[inline]
        pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
            let len = x.len() / 8 * 8;
            let mut sum = unsafe {
                let mut sums = _mm256_setzero_ps();
                x.chunks_exact(8).zip(y.chunks_exact(8)).for_each(|(a, b)| {
                    let x = _mm256_loadu_ps(a.as_ptr());
                    let y = _mm256_loadu_ps(b.as_ptr());
                    sums = _mm256_fmadd_ps(x, y, sums);
                });
                add_f32_register(sums)
            };
            sum += x[len..]
                .iter()
                .zip(y[len..].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            sum
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use num_traits::FromPrimitive;

    #[test]
    fn test_dot() {
        let x: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..120).map(|v| v as f32).collect();

        assert_eq!(x.dot(&y), dot(&x, &y));

        let x: Vec<f16> = (0..20).map(|v| f16::from_i32(v).unwrap()).collect();
        let y: Vec<f16> = (100..120).map(|v| f16::from_i32(v).unwrap()).collect();
        assert_eq!(x.dot(&y), dot(&x, &y));

        let x: Vec<f64> = (20..40).map(|v| f64::from_i32(v).unwrap()).collect();
        let y: Vec<f64> = (120..140).map(|v| f64::from_i32(v).unwrap()).collect();
        assert_eq!(x.dot(&y), dot(&x, &y));
    }
}
