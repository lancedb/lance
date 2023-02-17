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

use std::sync::Arc;

use arrow_array::Float32Array;

use super::compute::normalize;
use super::Distance;
use crate::Result;

/// Cosine Distance
///
/// <https://en.wikipedia.org/wiki/Cosine_similarity>
#[derive(Debug, Clone, Default)]
pub struct CosineDistance {}

/// Fallback Cosine Distance function.
fn cosine_dist(from: &Float32Array, to: &Float32Array, dimension: usize) -> Arc<Float32Array> {
    assert_eq!(from.len(), dimension);
    let n = to.len() / dimension;

    let distances: Float32Array = (0..n)
        .map(|idx| {
            let vector = &to.values()[idx * dimension..(idx + 1) * dimension];
            let mut x_sq = 0_f32;
            let mut y_sq = 0_f32;
            let mut xy = 0_f32;
            from.values().iter().zip(vector.iter()).for_each(|(x, y)| {
                xy += x * y;
                x_sq += x.powi(2);
                y_sq += y.powi(2);
            });
            xy / (x_sq.sqrt() * y_sq.sqrt())
        })
        .collect();
    Arc::new(distances)
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn cosine_dist_neon(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::aarch64::*;
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
    vaddvq_f32(xy) / vaddvq_f32(y_sq)
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn cosine_dist_fma(x: &[f32], y: &[f32]) -> f32 {
    use super::compute::add_fma;
    use std::arch::x86_64::*;

    let len = x.len();
    let mut xy = _mm256_setzero_ps();
    let mut y_sq = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        // Cache line-aligned
        let left = _mm256_load_ps(x.as_ptr().add(i));
        let right = _mm256_load_ps(y.as_ptr().add(i));
        xy = _mm256_fmadd_ps(left, right, xy);
        y_sq = _mm256_fmadd_ps(right, right, y_sq);
    }
    add_fma(xy) / add_fma(y_sq)
}

#[inline]
fn cosine_dist_simd(from: &Float32Array, to: &Float32Array, dimension: usize) -> Arc<Float32Array> {
    assert!(to.len() % dimension == 0);
    use arrow::array::Float32Builder;

    let x = from.values();
    let to_values = to.values();
    let x_sq = normalize(x);
    let n = to.len() / dimension;
    let mut builder = Float32Builder::with_capacity(n);
    for y in to_values.chunks_exact(dimension) {
        #[cfg(any(target_arch = "aarch64"))]
        {
            builder.append_value(unsafe { cosine_dist_neon(x, y) } / x_sq);
        }
        #[cfg(any(target_arch = "x86_64"))]
        {
            builder.append_value(unsafe { cosine_dist_fma(x, y) } / x_sq);
        }
    }
    Arc::new(builder.finish())
}

impl Distance for CosineDistance {
    fn distance(
        &self,
        from: &Float32Array,
        to: &Float32Array,
        dimension: usize,
    ) -> Result<Arc<Float32Array>> {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::is_aarch64_feature_detected;
            if is_aarch64_feature_detected!("neon") && from.len() % 4 == 0 {
                return Ok(cosine_dist_simd(from, to, dimension));
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") && from.len() % 8 == 0 {
                return Ok(cosine_dist_simd(from, to, dimension));
            }
        }

        // Fallback
        Ok(cosine_dist(from, to, dimension))
    }
}
