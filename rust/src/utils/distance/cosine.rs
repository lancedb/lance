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
unsafe fn cosine_dist_neon(x: &[f32], y: &[f32], x_norm: f32) -> f32 {
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
    vaddvq_f32(xy) / (x_norm * vaddvq_f32(y_sq).sqrt())
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn cosine_dist_fma(x_vector: &[f32], y_vector: &[f32], x_norm: f32) -> f32 {
    use super::compute::add_fma;
    use std::arch::x86_64::*;

    let len = x_vector.len();
    let mut xy = _mm256_setzero_ps();
    let mut y_sq = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        let x = _mm256_load_ps(x_vector.as_ptr().add(i));
        let y = _mm256_load_ps(y_vector.as_ptr().add(i));
        xy = _mm256_fmadd_ps(x, y, xy);
        y_sq = _mm256_fmadd_ps(y, y, y_sq);
    }
    add_fma(xy) / (x_norm * add_fma(y_sq).sqrt())
}

#[inline]
fn cosine_dist_simd(from: &Float32Array, to: &Float32Array, dimension: usize) -> Arc<Float32Array> {
    assert!(to.len() % dimension == 0);
    use arrow::array::Float32Builder;

    let x = from.values();
    let to_values = to.values();
    let x_norm = normalize(x);
    let n = to.len() / dimension;
    let mut builder = Float32Builder::with_capacity(n);
    for y in to_values.chunks_exact(dimension) {
        #[cfg(any(target_arch = "aarch64"))]
        {
            builder.append_value(unsafe { cosine_dist_neon(x, y, x_norm) });
        }
        #[cfg(any(target_arch = "x86_64"))]
        {
            builder.append_value(unsafe { cosine_dist_fma(x, y, x_norm) });
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

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_cosine() {
        let dist = CosineDistance::default();
        let x: Float32Array = (1..9).map(|v| v as f32).collect();
        let y: Float32Array = (100..108).map(|v| v as f32).collect();
        let d = dist.distance(&x, &y, 8).unwrap();
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d.value(0), 0.90095701);

        let x = Float32Array::from_iter_values([3.0, 45.0, 7.0, 2.0, 5.0, 20.0, 13.0, 12.0]);
        let y = Float32Array::from_iter_values([2.0, 54.0, 13.0, 15.0, 22.0, 34.0, 50.0, 1.0]);
        let d = dist.distance(&x, &y, 8).unwrap();
        // from sklearn.metrics.pairwise import cosine_similarity
        assert_relative_eq!(d.value(0), 0.8735806510613104);
    }
}
