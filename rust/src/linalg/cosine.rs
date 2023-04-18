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

use std::iter::Sum;
use std::sync::Arc;

use arrow_array::Float32Array;
use num_traits::real::Real;

use super::dot::dot;
use super::normalize::normalize;

/// Cosine Distance
pub trait Cosine {
    type Output;

    /// Cosine distance between two vectors.
    fn cosine(&self, other: &Self) -> Self::Output;

    /// Fast cosine function, that assumes that the norm of the first vector is already known.
    fn cosine_fast(&self, x_norm: Self::Output, other: &Self) -> Self::Output;
}

impl Cosine for [f32] {
    type Output = f32;

    #[inline]
    fn cosine(&self, other: &[f32]) -> f32 {
        let x_norm = normalize(self);
        self.cosine_fast(x_norm, other)
    }

    #[inline]
    fn cosine_fast(&self, x_norm: Self::Output, other: &Self) -> Self::Output {
        #[cfg(target_arch = "aarch64")]
        {
            return aarch64::neon::cosine_f32(self, other, x_norm);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") {
                return x86_64::avx::cosine_f32(self, other, x_norm);
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        cosine_scalar(self, x_norm, other)
    }
}

/// Fallback non-SIMD implementation
#[allow(dead_code)] // Does not fallback on aarch64.
#[inline]
fn cosine_scalar<T: Real + Sum>(x: &[T], x_norm: T, y: &[T]) -> T {
    let y_sq = dot(y, y);
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    T::one().sub(xy.div(x_norm.mul(y_sq.sqrt())))
}

/// Cosine distance function between two vectors.
pub fn cosine_distance(from: &[f32], to: &[f32]) -> f32 {
    from.cosine(to)
}

/// Cosine Distance
///
/// <https://en.wikipedia.org/wiki/Cosine_similarity>
pub fn cosine_distance_batch(from: &[f32], to: &[f32], dimension: usize) -> Arc<Float32Array> {
    let x_norm = normalize(from);

    let dists = unsafe {
        Float32Array::from_trusted_len_iter(
            to.chunks_exact(dimension)
                .map(|y| Some(from.cosine_fast(x_norm, y))),
        )
    };
    Arc::new(dists)
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    mod avx {
        use super::*;

        #[inline]
        fn cosine_f32(x_vector: &[f32], y_vector: &[f32], x_norm: f32) -> f32 {
            unsafe {
                use super::compute::add_fma;

                let len = x_vector.len();
                let mut xy = _mm256_setzero_ps();
                let mut y_sq = _mm256_setzero_ps();
                for i in (0..len).step_by(8) {
                    let x = _mm256_load_ps(x_vector.as_ptr().add(i));
                    let y = _mm256_load_ps(y_vector.as_ptr().add(i));
                    xy = _mm256_fmadd_ps(x, y, xy);
                    y_sq = _mm256_fmadd_ps(y, y, y_sq);
                }
                1.0 - add_fma(xy) / (x_norm * add_fma(y_sq).sqrt())
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    pub(crate) mod neon {
        use super::*;

        #[inline]
        pub(crate) fn cosine_f32(x: &[f32], y: &[f32], x_norm: f32) -> f32 {
            unsafe {
                let len = x.len();
                let buf = [0.0_f32; 4];
                let mut xy = vld1q_f32(buf.as_ptr());
                let mut y_sq = xy;
                for i in (0..len).step_by(4) {
                    let left = vld1q_f32(x.as_ptr().add(i));
                    let right = vld1q_f32(y.as_ptr().add(i));
                    xy = vfmaq_f32(xy, left, right);
                    y_sq = vfmaq_f32(y_sq, right, right);
                }
                1.0 - vaddvq_f32(xy) / (x_norm * vaddvq_f32(y_sq).sqrt())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_cosine() {
        let x: Float32Array = (1..9).map(|v| v as f32).collect();
        let y: Float32Array = (100..108).map(|v| v as f32).collect();
        let d = cosine_distance_batch(x.values(), y.values(), 8);
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d.value(0), 1.0 - 0.90095701);

        let x = Float32Array::from_iter_values([3.0, 45.0, 7.0, 2.0, 5.0, 20.0, 13.0, 12.0]);
        let y = Float32Array::from_iter_values([2.0, 54.0, 13.0, 15.0, 22.0, 34.0, 50.0, 1.0]);
        let d = cosine_distance_batch(x.values(), y.values(), 8);
        // from sklearn.metrics.pairwise import cosine_similarity
        assert_relative_eq!(d.value(0), 1.0 - 0.8735806510613104);
    }
}
