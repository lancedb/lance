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

use std::sync::Arc;

use arrow_array::Float32Array;

use super::is_simd_aligned;
use crate::Result;

// TODO: wait [std::simd] to be stable to replace manually written AVX/FMA code.
//
// `from` and `to` must have the same length.
#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn euclidean_distance_fma(from: &[f32], to: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(from.len(), to.len());

    let len = from.len();
    let mut sums = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        // Cache line-aligned
        let left = _mm256_load_ps(from.as_ptr().add(i));
        let right = _mm256_load_ps(to.as_ptr().add(i));
        let sub = _mm256_sub_ps(left, right);
        // sum = sub * sub + sum
        sums = _mm256_fmadd_ps(sub, sub, sums);
    }
    // Shift and add vector, until only 1 value left.
    // sums = [x0-x7], shift = [x4-x7]
    let mut shift = _mm256_permute2f128_ps(sums, sums, 1);
    // [x0+x4, x1+x5, ..]
    sums = _mm256_add_ps(sums, shift);
    shift = _mm256_permute_ps(sums, 14);
    sums = _mm256_add_ps(sums, shift);
    sums = _mm256_hadd_ps(sums, sums);
    let mut results: [f32; 8] = [0f32; 8];
    _mm256_storeu_ps(results.as_mut_ptr(), sums);
    results[0]
}

#[inline]
fn l2_distance_slow(from: &[f32], to: &[f32]) -> f32 {
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_distance_neon(from: &[f32], to: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = from.len();
    let buf = [0.0_f32; 4];
    let mut sum = vld1q_f32(buf.as_ptr());
    for i in (0..len).step_by(4) {
        let left = vld1q_f32(from.as_ptr().add(i));
        let right = vld1q_f32(to.as_ptr().add(i));
        let sub = vsubq_f32(left, right);
        sum = vfmaq_f32(sum, sub, sub);
    }
    vaddvq_f32(sum)
}

fn l2_distance_simd(from: &[f32], to: &[f32], dimension: usize) -> Result<Arc<Float32Array>> {
    let n = to.len() / dimension;

    let scores: Float32Array = unsafe {
        Float32Array::from_trusted_len_iter(
            (0..n)
                .map(|idx| {
                    #[cfg(any(target_arch = "x86_64"))]
                    {
                        euclidean_distance_fma(from, &to[idx * dimension..(idx + 1) * dimension])
                    }

                    #[cfg(any(target_arch = "aarch64"))]
                    {
                        l2_distance_neon(from, &to[idx * dimension..(idx + 1) * dimension])
                    }
                })
                .map(Some),
        )
    };
    Ok(Arc::new(scores))
}

/// Compute L2 distance
///
/// Parameters
///
/// - `from`: the vector to compute distance from.
/// - `to`: a list of vectors to compute distance to.
/// - `dimension`: the dimension of the vectors.
pub fn l2_distance(from: &[f32], to: &[f32], dimension: usize) -> Result<Arc<Float32Array>> {
    assert_eq!(from.len(), dimension);
    assert_eq!(to.len() % dimension, 0);

    #[cfg(any(target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma")
            && is_simd_aligned(from.as_ptr(), 32)
            && is_simd_aligned(to.as_ptr(), 32)
            && from.len() % 8 == 0
        {
            return l2_distance_simd(from, to, dimension);
        }
    }

    #[cfg(any(target_arch = "aarch64"))]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon")
            && is_simd_aligned(from.as_ptr(), 16)
            && is_simd_aligned(to.as_ptr(), 16)
            && from.len() % 4 == 0
        {
            return l2_distance_simd(from, to, dimension);
        }
    }

    // Fallback to slow version
    let scores: Float32Array = unsafe {
        Float32Array::from_trusted_len_iter(
            to.chunks_exact(dimension)
                .map(|v| l2_distance_slow(from, v))
                .map(Some),
        )
    };
    Ok(Arc::new(scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow::array::{as_primitive_array, FixedSizeListArray};
    use arrow_array::types::Float32Type;

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
        let scores = l2_distance(
            point.values(),
            as_primitive_array::<Float32Type>(mat.values().as_ref()).values(),
            8,
        )
        .unwrap();

        assert_eq!(
            scores.as_ref(),
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
        let scores = l2_distance(&point.values()[2..], &mat[6..], 8).unwrap();

        assert_eq!(
            scores.as_ref(),
            &Float32Array::from(vec![32.0, 8.0, 0.0, 8.0])
        );
    }
    #[test]
    fn test_odd_length_vector() {
        let mat = Float32Array::from_iter((0..5).map(|v| Some(v as f32)));
        let point = Float32Array::from((2..7).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let scores = l2_distance(point.values(), mat.values(), 5).unwrap();

        assert_eq!(scores.as_ref(), &Float32Array::from(vec![20.0]));
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

        let d = l2_distance(q.values(), values.values(), 32).unwrap();
        assert_relative_eq!(0.31935785197341404, d.value(0));
    }
}
