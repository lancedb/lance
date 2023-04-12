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

use arrow_array::Float32Array;
use num_traits::real::Real;

/// Trait for calculating L2 distance.
///
pub trait L2<T: Real> {
    fn l2(&self, other: &Self) -> T;
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn l2_fma_f32(from: &[f32], to: &[f32]) -> f32 {
    use super::add::add_fma_f32;
    use std::arch::x86_64::*;

    debug_assert_eq!(from.len(), to.len());

    let len = from.len() / 8 * 8;
    let mut sums = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        // We use `_mm256_loadu_ps` instead of `_mm256_load_ps` to make it work with unaligned data.
        // Benchmark on an AMD Ryzen 5900X shows that it adds less than 2% overhead with aligned vectors.
        let left = _mm256_loadu_ps(from.as_ptr().add(i));
        let right = _mm256_loadu_ps(to.as_ptr().add(i));
        let sub = _mm256_sub_ps(left, right);
        // sum = sub * sub + sum
        sums = _mm256_fmadd_ps(sub, sub, sums);
    }
    let sum = add_fma_f32(sums);

    from[len..]
        .iter()
        .zip(to[len..].iter())
        .fold(sum, |acc, (a, b)| acc + (a - b).powi(2))
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_neon_f32(from: &[f32], to: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = from.len() / 4 * 4;
    let buf = [0.0_f32; 4];
    let mut sum = vld1q_f32(buf.as_ptr());
    for i in (0..len).step_by(4) {
        let left = vld1q_f32(from.as_ptr().add(i));
        let right = vld1q_f32(to.as_ptr().add(i));
        let sub = vsubq_f32(left, right);
        sum = vfmaq_f32(sum, sub, sub);
    }
    let sum = vaddvq_f32(sum);
    from[len..]
        .iter()
        .zip(to[len..].iter())
        .fold(sum, |acc, (a, b)| acc + (a - b).powi(2))
}

/// Fall back to scalar implementation.
///
/// With `RUSTFLAGS="-C target-cpu=native"`, rustc will auto-vectorize to some extent.
///
/// `rustc 1.68` uses SSE and does loop-unfolding, inspect here
/// <https://rust.godbolt.org/>.
fn l2_scalar<T: Real + Sum>(from: &[T], to: &[T]) -> T {
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| (a.sub(*b).powi(2)))
        .sum::<T>()
}

impl L2<f32> for [f32] {
    #[inline]
    fn l2(&self, other: &Self) -> f32 {
        #[cfg(any(target_arch = "aarch64"))]
        {
            use std::arch::is_aarch64_feature_detected;
            if is_aarch64_feature_detected!("neon") {
                unsafe {
                    return l2_neon_f32(self, other);
                }
            }
        }

        #[cfg(any(target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("fma") {
                unsafe {
                    return l2_fma_f32(self, other);
                }
            }
        }

        l2_scalar(self, other)
    }
}

impl L2<f32> for Vec<f32> {
    fn l2(&self, other: &Self) -> f32 {
        self.as_slice().l2(other.as_slice())
    }
}

impl L2<f32> for Float32Array {
    fn l2(&self, other: &Self) -> f32 {
        self.values().l2(other.values())
    }
}

pub fn l2_distance(from: &[f32], to: &[f32], dimension: usize) -> Float32Array {
    assert_eq!(from.len(), dimension);
    assert_eq!(to.len() % dimension, 0);

    let scores: Float32Array = unsafe {
        Float32Array::from_trusted_len_iter(
            to.chunks_exact(dimension).map(|v| from.l2(v)).map(Some),
        )
    };
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_l2_distance_cases() {
        let values = vec![
            0.25335717, 0.24663818, 0.26330215, 0.14988247, 0.06042378, 0.21077952, 0.26687378,
            0.22145681, 0.18319066, 0.18688454, 0.05216244, 0.11470364, 0.10554603, 0.19964123,
            0.06387895, 0.18992095, 0.00123718, 0.13500804, 0.09516747, 0.19508345, 0.2582458,
            0.1211653, 0.21121833, 0.24809816, 0.04078768, 0.19586588, 0.16496408, 0.14766085,
            0.04898421, 0.14728612, 0.21263947, 0.16763233,
        ];

        let q = vec![
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
        ];

        let d = q.l2(&values);
        // From numpy.linage.norm(a-b)
        assert_relative_eq!(0.3193578, d);

        let fb_d = l2_scalar(q.as_slice(), values.as_slice());
        assert_relative_eq!(d, fb_d);

        let value_array: Float32Array = values.into();
        let q_array: Float32Array = q.into();

        assert_relative_eq!(d, q_array.l2(&value_array));
    }

    #[test]
    fn test_not_aligned() {
        let v1: Float32Array = (0..14).map(|v| v as f32).collect();
        let v2: Float32Array = (0..10).map(|v| v as f32).collect();
        let score = v1.values()[6..].l2(&v2.values()[2..]);

        assert_eq!(score, 128.0);
    }

    #[test]
    fn test_odd_size_arrays() {
        let v1 = (0..18).map(|v| v as f32).collect::<Vec<_>>();
        let v2 = (2..20).map(|v| v as f32).collect::<Vec<_>>();
        let score = v1[0..13].l2(&v2[0..13]);

        assert_relative_eq!(score, 4.0 * 13.0);

        let v1 = (0..3).map(|v| v as f32).collect::<Vec<_>>();
        let v2 = (2..5).map(|v| v as f32).collect::<Vec<_>>();
        let score = v1.l2(&v2);

        assert_relative_eq!(score, 4.0 * 3.0);
    }
}
