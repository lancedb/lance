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

/// Normalize a vector.
///
/// The parameters must be cache line aligned. For example, from
/// Arrow Arrays, i.e., Float32Array
#[inline]
pub fn norm_l2(vector: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::neon::norm_l2(vector)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return x86_64::avx::norm_l2_f32(vector);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {

    pub mod avx {
        use crate::linalg::x86_64::avx::*;
        use std::arch::x86_64::*;

        #[inline]
        pub fn norm_l2_f32(vector: &[f32]) -> f32 {
            let len = vector.len() / 8 * 8;
            let mut sum = unsafe {
                let mut sums = _mm256_setzero_ps();
                vector.chunks_exact(8).for_each(|chunk| {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    sums = _mm256_fmadd_ps(x, x, sums);
                });
                add_f32_register(sums)
            };
            sum += vector[len..].iter().map(|v| v * v).sum::<f32>();
            sum.sqrt()
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    pub mod neon {
        use std::arch::aarch64::*;

        #[inline]
        pub fn norm_l2(vector: &[f32]) -> f32 {
            let len = vector.len() / 4 * 4;
            let mut sum = unsafe {
                let buf = [0.0_f32; 4];
                let mut sum = vld1q_f32(buf.as_ptr());
                for i in (0..len).step_by(4) {
                    let x = vld1q_f32(vector.as_ptr().add(i));
                    sum = vfmaq_f32(sum, x, x);
                }
                vaddvq_f32(sum)
            };
            sum += vector[len..].iter().map(|v| v.powi(2)).sum::<f32>();
            sum.sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_l2() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = norm_l2(&data);
        assert_eq!(result, (1..=8).map(|v| (v * v) as f32).sum::<f32>().sqrt());

        let not_aligned = norm_l2(&data[2..]);
        assert_eq!(
            not_aligned,
            (3..=8).map(|v| (v * v) as f32).sum::<f32>().sqrt()
        );
    }
}
