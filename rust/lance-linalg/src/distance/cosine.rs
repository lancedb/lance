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

//! Cosine distance
//!
//! <https://en.wikipedia.org/wiki/Cosine_similarity>
//!
//! `bf16, f16, f32, f64` types are supported.

use std::iter::Sum;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use half::{bf16, f16};
use num_traits::real::Real;
use num_traits::{AsPrimitive, FromPrimitive};

use super::dot::dot;
use super::norm_l2::{norm_l2, Normalize};
use crate::simd::{
    f32::{f32x16, f32x8},
    SIMD,
};

/// Cosine Distance
pub trait Cosine {
    type Output;

    /// Cosine distance between two vectors.
    fn cosine(&self, other: &Self) -> Self::Output;

    /// Fast cosine function, that assumes that the norm of the first vector is already known.
    fn cosine_fast(&self, x_norm: Self::Output, other: &Self) -> Self::Output;

    /// Cosine between two vectors, with the L2 norms of both vectors already known.
    fn cosine_with_norms(
        &self,
        x_norm: Self::Output,
        y_norm: Self::Output,
        y: &Self,
    ) -> Self::Output;
}

impl Cosine for [bf16] {
    type Output = bf16;

    #[inline]
    fn cosine(&self, other: &Self) -> Self::Output {
        let x_norm = self.norm_l2();
        self.cosine_fast(x_norm, other)
    }

    #[inline]
    fn cosine_fast(&self, x_norm: Self::Output, other: &Self) -> Self::Output {
        // TODO: Implement SIMD
        cosine_scalar(self, x_norm, other)
    }

    #[inline]
    fn cosine_with_norms(
        &self,
        x_norm: Self::Output,
        y_norm: Self::Output,
        y: &Self,
    ) -> Self::Output {
        cosine_scalar_fast(self, x_norm, y, y_norm)
    }
}

impl Cosine for [f16] {
    type Output = f16;

    #[inline]
    fn cosine(&self, other: &Self) -> Self::Output {
        let x_norm = self.norm_l2();
        self.cosine_fast(x_norm, other)
    }

    #[inline]
    fn cosine_fast(&self, x_norm: Self::Output, other: &Self) -> Self::Output {
        // TODO: Implement SIMD
        cosine_scalar(self, x_norm, other)
    }

    #[inline]
    fn cosine_with_norms(
        &self,
        x_norm: Self::Output,
        y_norm: Self::Output,
        y: &Self,
    ) -> Self::Output {
        cosine_scalar_fast(self, x_norm, y, y_norm)
    }
}

impl Cosine for [f32] {
    type Output = f32;

    #[inline]
    fn cosine(&self, other: &[f32]) -> f32 {
        let x_norm = norm_l2(self);
        self.cosine_fast(x_norm, other)
    }

    #[inline]
    fn cosine_fast(&self, x_norm: Self::Output, other: &Self) -> Self::Output {
        let dim = self.len();
        let unrolled_len = dim / 16 * 16;
        let mut y_norm16 = f32x16::zeros();
        let mut xy16 = f32x16::zeros();
        for i in (0..unrolled_len).step_by(16) {
            unsafe {
                let x = f32x16::load_unaligned(self.as_ptr().add(i));
                let y = f32x16::load_unaligned(other.as_ptr().add(i));
                xy16.multiply_add(x, y);
                y_norm16.multiply_add(y, y);
            }
        }
        let aligned_len = dim / 8 * 8;
        let mut y_norm8 = f32x8::zeros();
        let mut xy8 = f32x8::zeros();
        for i in (unrolled_len..aligned_len).step_by(8) {
            unsafe {
                let x = f32x8::load_unaligned(self.as_ptr().add(i));
                let y = f32x8::load_unaligned(other.as_ptr().add(i));
                xy8.multiply_add(x, y);
                y_norm8.multiply_add(y, y);
            }
        }
        let y_norm =
            y_norm16.reduce_sum() + y_norm8.reduce_sum() + norm_l2(&other[aligned_len..]).powi(2);
        let xy =
            xy16.reduce_sum() + xy8.reduce_sum() + dot(&self[aligned_len..], &other[aligned_len..]);
        1.0 - xy / x_norm / y_norm.sqrt()
    }

    #[inline]
    fn cosine_with_norms(
        &self,
        x_norm: Self::Output,
        y_norm: Self::Output,
        y: &Self,
    ) -> Self::Output {
        let dim = self.len();
        let unrolled_len = dim / 16 * 16;
        let mut xy16 = f32x16::zeros();
        for i in (0..unrolled_len).step_by(16) {
            unsafe {
                let x = f32x16::load_unaligned(self.as_ptr().add(i));
                let y = f32x16::load_unaligned(y.as_ptr().add(i));
                xy16.multiply_add(x, y);
            }
        }
        let aligned_len = dim / 8 * 8;
        let mut xy8 = f32x8::zeros();
        for i in (unrolled_len..aligned_len).step_by(8) {
            unsafe {
                let x = f32x8::load_unaligned(self.as_ptr().add(i));
                let y = f32x8::load_unaligned(y.as_ptr().add(i));
                xy8.multiply_add(x, y);
            }
        }
        let xy =
            xy16.reduce_sum() + xy8.reduce_sum() + dot(&self[aligned_len..], &y[aligned_len..]);
        1.0 - xy / x_norm / y_norm
    }
}

impl Cosine for [f64] {
    type Output = f64;

    #[inline]
    fn cosine(&self, other: &Self) -> Self::Output {
        let x_norm = self.norm_l2();
        self.cosine_fast(x_norm, other)
    }

    #[inline]
    fn cosine_fast(&self, x_norm: Self::Output, other: &Self) -> Self::Output {
        // TODO: Implement SIMD
        cosine_scalar(self, x_norm, other)
    }

    #[inline]
    fn cosine_with_norms(
        &self,
        x_norm: Self::Output,
        y_norm: Self::Output,
        y: &Self,
    ) -> Self::Output {
        cosine_scalar_fast(self, x_norm, y, y_norm)
    }
}

/// Fallback non-SIMD implementation
#[allow(dead_code)] // Does not fallback on aarch64.
#[inline]
fn cosine_scalar<T: Real + Sum + AsPrimitive<f64> + FromPrimitive>(
    x: &[T],
    x_norm: T,
    y: &[T],
) -> T {
    let y_sq = dot(y, y);
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    // use f64 for overflow protection.
    T::from_f64(1.0 - (xy.as_() / (x_norm.as_() * (y_sq.sqrt()).as_()))).unwrap()
}

#[inline]
fn cosine_scalar_fast<T: Real + Sum + AsPrimitive<f64> + FromPrimitive>(
    x: &[T],
    x_norm: T,
    y: &[T],
    y_norm: T,
) -> T {
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    // use f64 for overflow protection.
    T::from_f64(1.0 - (xy.as_() / (x_norm.as_() * y_norm.as_()))).unwrap()
}

/// Cosine distance function between two vectors.
pub fn cosine_distance<T: Cosine + ?Sized>(from: &T, to: &T) -> T::Output {
    from.cosine(to)
}

mod f32 {
    use super::*;

    // TODO: how can we explicity infer N?
    #[inline]
    pub(super) fn cosine_once<S: SIMD<f32, N>, const N: usize>(
        x: &[f32],
        x_norm: f32,
        y: &[f32],
    ) -> f32 {
        let x = unsafe { S::load_unaligned(x.as_ptr()) };
        let y = unsafe { S::load_unaligned(y.as_ptr()) };
        let y2 = y * y;
        let xy = x * y;
        1.0 - xy.reduce_sum() / x_norm / y2.reduce_sum().sqrt()
    }
}

/// Cosine Distance
///
/// <https://en.wikipedia.org/wiki/Cosine_similarity>
///
/// Parameters
/// -----------
///
/// - *from*: the vector to compute distance from.
/// - *to*: the batch of vectors to compute distance to.
/// - *dimension*: the dimension of the vector.
///
/// Returns
/// -------
/// An iterator of pair-wise cosine distance between from vector to each vector in the batch.
///
pub fn cosine_distance_batch<'a>(
    from: &'a [f32],
    batch: &'a [f32],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a> {
    let x_norm = norm_l2(from);

    match dimension {
        8 => Box::new(
            batch
                .chunks_exact(dimension)
                .map(move |y| f32::cosine_once::<f32x8, 8>(from, x_norm, y)),
        ),
        16 => Box::new(
            batch
                .chunks_exact(dimension)
                .map(move |y| f32::cosine_once::<f32x16, 16>(from, x_norm, y)),
        ),
        _ => Box::new(
            batch
                .chunks_exact(dimension)
                .map(move |y| from.cosine_fast(x_norm, y)),
        ),
    }
}

/// Compute Cosine distance between a vector and a batch of vectors.
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
pub fn cosine_distance_arrow_batch(from: &[f32], to: &FixedSizeListArray) -> Arc<Float32Array> {
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    let x_norm = norm_l2(from);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values = to.values().as_primitive::<Float32Type>().values();
    let dists = to_values
        .chunks_exact(dimension)
        .map(|v| from.cosine_fast(x_norm, v));

    Arc::new(Float32Array::new(dists.collect(), to.nulls().cloned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_cosine() {
        let x: Float32Array = (1..9).map(|v| v as f32).collect();
        let y: Float32Array = (100..108).map(|v| v as f32).collect();
        let d = cosine_distance_batch(x.values(), y.values(), 8).collect::<Vec<_>>();
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d[0], 1.0 - 0.900_957);

        let x = Float32Array::from_iter_values([3.0, 45.0, 7.0, 2.0, 5.0, 20.0, 13.0, 12.0]);
        let y = Float32Array::from_iter_values([2.0, 54.0, 13.0, 15.0, 22.0, 34.0, 50.0, 1.0]);
        let d = cosine_distance_batch(x.values(), y.values(), 8).collect::<Vec<_>>();
        // from sklearn.metrics.pairwise import cosine_similarity
        assert_relative_eq!(d[0], 1.0 - 0.873_580_63);
    }

    #[test]
    fn test_cosine_not_aligned() {
        let x: Float32Array = vec![16_f32, 32_f32].into();
        let y: Float32Array = vec![1_f32, 2_f32, 4_f32, 8_f32].into();
        let d = cosine_distance_batch(x.values(), y.values(), 2).collect::<Vec<_>>();
        assert_relative_eq!(d[0], 0.0);
        assert_relative_eq!(d[0], 0.0);
    }
}
