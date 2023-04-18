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

use num_traits::real::Real;

use super::dot::dot;

/// Cosine Distance
pub trait Cosine {
    type Output;

    fn cosine(&self, other: &Self) -> Self::Output;
}

impl Cosine for [f32] {
    type Output = f32;

    fn cosine(&self, other: &[f32]) -> f32 {
        cosine_distance(self, other)
    }
}

/// Fallback
fn cosine_scalar<T: Real + Sum>(x: &[T], y: &[T]) -> T {
    let x_sq = dot(x, x);
    let y_sq = dot(y, y);
    let xy = dot(x, y);
    T::one().sub(xy.div(x_sq.sqrt().mul(y_sq.sqrt())))
}

pub fn cosine_distance(from: &[f32], to: &[f32]) -> f32 {
    cosine_scalar(from, to)
}

/// Cosine Distance
///
/// <https://en.wikipedia.org/wiki/Cosine_similarity>
pub fn cosine_distance_batch(from: &[f32], to: &[f32], dimension: usize) -> Arc<Float32Array> {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon")
            && is_simd_aligned(from.as_ptr(), 16)
            && is_simd_aligned(to.as_ptr(), 16)
            && from.len() % 4 == 0
        {
            return cosine_dist_simd(from, to, dimension);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma")
            && is_simd_aligned(from.as_ptr(), 32)
            && is_simd_aligned(to.as_ptr(), 32)
            && from.len() % 8 == 0
        {
            return cosine_dist_simd(from, to, dimension);
        }
    }

    // Fallback
    cosine_dist_slow(from, to, dimension)
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

    mod neon {
        use super::*;

        #[inline]
        fn cosine_f32(x: &[f32], y: &[f32], x_norm: f32) -> f32 {
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
