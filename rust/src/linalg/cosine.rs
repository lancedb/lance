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

use arrow_array::Float32Array;
use num_traits::real::Real;

use super::dot::Dot;
use crate::linalg::normalize::Normalize;

pub trait Cosine<T: Real>: Normalize<T> {
    /// Cosine distance
    fn cosine(&self, y: &Self) -> T {
        let x_norm = self.norm();
        self.cosine_fast(x_norm, y)
    }

    /// Cosine distance, fast version.
    ///
    /// This version assumes that the input vectors are already normalized.
    fn cosine_fast(&self, x_norm: T, y: &Self) -> T;
}

#[inline]
fn cosine_scalar<T: Normalize<V> + Dot<V> + ?Sized, V: Real>(x: &T, y: &T, x_norm: V) -> V {
    let y_norm = y.norm();
    let xy = x.dot(y);
    V::one() - xy / (x_norm * y_norm)
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn cosine_dist_neon_f32(x: &[f32], y: &[f32], x_norm: f32) -> f32 {
    use std::arch::aarch64::*;
    let len = x.len() / 4 * 4;
    let buf = [0.0_f32; 4];
    let mut xy = vld1q_f32(buf.as_ptr());
    let mut y_sq = xy;
    for i in (0..len).step_by(4) {
        let left = vld1q_f32(x.as_ptr().add(i));
        let right = vld1q_f32(y.as_ptr().add(i));
        xy = vfmaq_f32(xy, left, right);
        y_sq = vfmaq_f32(y_sq, right, right);
    }
    let mut xy = vaddvq_f32(xy);
    let mut y_sq = vaddvq_f32(y_sq);
    // Remaining
    x[len..]
        .iter()
        .zip(y[len..].iter())
        .for_each(|(x, y)| {
            xy += x * y;
            y_sq += y.powi(2)
        });
    1.0 - xy / (x_norm * y_sq.sqrt())
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn cosine_dist_fma_f32(x_vector: &[f32], y_vector: &[f32], x_norm: f32) -> f32 {
    use crate::linalg::add::add_fma_f32;
    use std::arch::x86_64::*;

    let len = x_vector.len() / 8 * 8;
    let mut xy = _mm256_setzero_ps();
    let mut y_sq = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        let x = _mm256_loadu_ps(x_vector.as_ptr().add(i));
        let y = _mm256_loadu_ps(y_vector.as_ptr().add(i));
        xy = _mm256_fmadd_ps(x, y, xy);
        y_sq = _mm256_fmadd_ps(y, y, y_sq);
    }
    let mut xy = add_fma_f32(xy);
    let mut y_sq = add_fma_f32(y_sq);
    // Remaining
    x_vector[len..]
        .iter()
        .zip(y_vector[len..].iter())
        .for_each(|(x, y)| {
            xy += x * y;
            y_sq += y.powi(2)
        });

    1.0 - xy / (x_norm * y_sq.sqrt())
}

impl Cosine<f32> for [f32] {
    fn cosine_fast(&self, x_norm: f32, y: &Self) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::is_aarch64_feature_detected;
            if is_aarch64_feature_detected!("neon") {
                return unsafe { cosine_dist_neon_f32(self, y, x_norm) };
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") {
                return unsafe { cosine_dist_fma_f32(self, y, x_norm) };
            }
        }

        // Fallback
        cosine_scalar(self, y, x_norm)
    }
}

/// Cosine Distance
///
/// <https://en.wikipedia.org/wiki/Cosine_similarity>
pub fn cosine_distance(from: &[f32], to: &[f32], dimension: usize) -> Float32Array {
    assert_eq!(from.len(), dimension);

    let x_norm = from.norm();
    let distances: Float32Array = to
        .chunks_exact(dimension)
        .map(|y| from.cosine_fast(x_norm, y))
        .collect();
    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_cosine() {
        let x: Float32Array = (1..9).map(|v| v as f32).collect();
        let y: Float32Array = (100..108).map(|v| v as f32).collect();
        let d = cosine_distance(x.values(), y.values(), 8);
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d.value(0), 1.0 - 0.90095701);

        let x = Float32Array::from_iter_values([3.0, 45.0, 7.0, 2.0, 5.0, 20.0, 13.0, 12.0]);
        let y = Float32Array::from_iter_values([2.0, 54.0, 13.0, 15.0, 22.0, 34.0, 50.0, 1.0]);
        let d = cosine_distance(x.values(), y.values(), 8);
        // from sklearn.metrics.pairwise import cosine_similarity
        assert_relative_eq!(d.value(0), 1.0 - 0.8735806510613104);
    }

    #[test]
    fn test_not_aligned_cosine() {
        let x: Float32Array = (1..50).map(|v| v as f32).collect();
        let y: Float32Array = (100..150).map(|v| v as f32).collect();

        // 15 elements, that is not aligned with 8 byte boundary.
        let d = x.values()[2..17].cosine(&y.values()[3..18]);
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d, 0.067156250);
    }
}
