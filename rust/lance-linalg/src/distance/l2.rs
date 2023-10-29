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

//! L2 (Euclidean) distance.
//!

use std::iter::Sum;
use std::sync::Arc;

use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use half::{bf16, f16};
use num_traits::real::Real;

use crate::simd::{f32::f32x8, SIMD};

/// Calculate the L2 distance between two vectors.
///
pub trait L2 {
    type Output;

    /// Calculate the L2 distance between two vectors.
    fn l2(&self, other: &Self) -> Self::Output;
}

/// Calculate the L2 distance between two vectors, using scalar operations.
///
/// Rely on compiler auto-vectorization.
#[inline]
fn l2_scalar<T: Real + Sum>(from: &[T], to: &[T]) -> T {
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| (a.sub(*b).powi(2)))
        .sum::<T>()
}

impl L2 for [bf16] {
    type Output = bf16;

    #[inline]
    fn l2(&self, other: &[bf16]) -> bf16 {
        // TODO: add SIMD support
        l2_scalar(self, other)
    }
}

impl L2 for [f16] {
    type Output = f16;

    #[inline]
    fn l2(&self, other: &[f16]) -> f16 {
        // TODO: add SIMD support
        l2_scalar(self, other)
    }
}

impl L2 for [f32] {
    type Output = f32;

    #[inline]
    fn l2(&self, other: &[f32]) -> f32 {
        let len = self.len();
        if len % 16 == 0 {  // Likely
            let mut sum1 = f32x8::splat(0.0);
            let mut sum2 = f32x8::splat(0.0);

            for i in (0..len).step_by(16) {
                unsafe {
                    let mut x1 = f32x8::load_unaligned(self.as_ptr().add(i));
                    let mut x2 = f32x8::load_unaligned(self.as_ptr().add(i + 8));
                    let y1 = f32x8::load_unaligned(other.as_ptr().add(i));
                    let y2 = f32x8::load_unaligned(other.as_ptr().add(i + 8));
                    x1 -= y1;
                    x2 -= y2;
                    sum1.multiply_add(x1, x1);
                    sum2.multiply_add(x2, x2);
                }
            }
            sum1 += sum2;
            return sum1.reduce_sum();
        } else if len % 8 == 0 {
            let mut sum1 = f32x8::splat(0.0);
            for i in (0..len).step_by(8) {
                unsafe {
                    let mut x = f32x8::load_unaligned(self.as_ptr().add(i));
                    let y = f32x8::load_unaligned(other.as_ptr().add(i));
                    x -= y;
                    sum1.multiply_add(x, x);
                }
            }
            return sum1.reduce_sum();
        } else {
            // Fallback to scalar
            l2_scalar(self, other)
        }
    }
}

impl L2 for Float32Array {
    type Output = f32;

    #[inline]
    fn l2(&self, other: &Self) -> f32 {
        self.values().l2(other.values())
    }
}

impl L2 for [f64] {
    type Output = f64;

    #[inline]
    fn l2(&self, other: &[f64]) -> f64 {
        // TODO: add SIMD support
        l2_scalar(self, other)
    }
}

/// Compute L2 distance between two vectors.
#[inline]
pub fn l2_distance(from: &[f32], to: &[f32]) -> f32 {
    from.l2(to)
}

/// Compute L2 distance between a vector and a batch of vectors.
///
/// Parameters
///
/// - `from`: the vector to compute distance from.
/// - `to`: a list of vectors to compute distance to.
/// - `dimension`: the dimension of the vectors.
pub fn l2_distance_batch(from: &[f32], to: &[f32], dimension: usize) -> Arc<Float32Array> {
    assert_eq!(from.len(), dimension);
    assert_eq!(to.len() % dimension, 0);

    let dists = to.chunks_exact(dimension).map(|v| from.l2(v));
    Arc::new(Float32Array::new(dists.collect(), None))
}

/// Compute L2 distance between a vector and a batch of vectors.
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
pub fn l2_distance_arrow_batch(from: &[f32], to: &FixedSizeListArray) -> Arc<Float32Array> {
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values = to.values().as_primitive::<Float32Type>().values();
    let dists = to_values.chunks_exact(dimension).map(|v| from.l2(v));

    Arc::new(Float32Array::new(dists.collect(), to.nulls().cloned()))
}


#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow_array::{cast::AsArray, types::Float32Type, FixedSizeListArray};

    #[test]
    fn test_euclidean_distance() {
        let mat = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some((0..8).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((1..9).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((2..10).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((3..11).map(|v| Some(v as f32)).collect::<Vec<_>>()),
            ],
            8,
        );
        let point = Float32Array::from((2..10).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let distances = l2_distance_batch(
            point.values(),
            mat.values().as_primitive::<Float32Type>().values(),
            8,
        );

        assert_eq!(
            distances.as_ref(),
            &Float32Array::from(vec![32.0, 8.0, 0.0, 8.0])
        );
    }

    #[test]
    fn test_not_aligned() {
        let mat = (0..6)
            .chain(0..8)
            .chain(1..9)
            .chain(2..10)
            .chain(3..11)
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let point = Float32Array::from((0..10).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let distances = l2_distance_batch(&point.values()[2..], &mat[6..], 8);

        assert_eq!(
            distances.as_ref(),
            &Float32Array::from(vec![32.0, 8.0, 0.0, 8.0])
        );
    }
    #[test]
    fn test_odd_length_vector() {
        let mat = Float32Array::from_iter((0..5).map(|v| Some(v as f32)));
        let point = Float32Array::from((2..7).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let distances = l2_distance_batch(point.values(), mat.values(), 5);

        assert_eq!(distances.as_ref(), &Float32Array::from(vec![20.0]));
    }

    #[test]
    fn test_l2_distance_cases() {
        let values: Float32Array = vec![
            0.25335717, 0.24663818, 0.26330215, 0.14988247, 0.06042378, 0.21077952, 0.26687378,
            0.22145681, 0.18319066, 0.18688454, 0.05216244, 0.11470364, 0.10554603, 0.19964123,
            0.06387895, 0.18992095, 0.00123718, 0.13500804, 0.09516747, 0.19508345, 0.2582458,
            0.1211653, 0.21121833, 0.24809816, 0.04078768, 0.19586588, 0.16496408, 0.14766085,
            0.04898421, 0.14728612, 0.21263947, 0.16763233,
        ]
        .into();

        let q: Float32Array = vec![
            0.18549609,
            0.29954708,
            0.28318876,
            0.05424477,
            0.093134984,
            0.21580857,
            0.2951282,
            0.19866848,
            0.13868214,
            0.19819534,
            0.23271298,
            0.047727287,
            0.14394054,
            0.023316395,
            0.18589257,
            0.037315924,
            0.07037327,
            0.32609823,
            0.07344752,
            0.020155912,
            0.18485495,
            0.32763934,
            0.14296658,
            0.04498596,
            0.06254237,
            0.24348071,
            0.16009757,
            0.053892266,
            0.05918874,
            0.040363103,
            0.19913352,
            0.14545348,
        ]
        .into();

        let d = l2_distance_batch(q.values(), values.values(), 32);
        assert_relative_eq!(0.319_357_84, d.value(0));
    }
}
