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
pub fn normalize(vector: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::neon::normalize_f32(vector)
    }

    #[cfg(target_arch = "x86_64")]
    {
        x86_64::avx::normalize_f32(vector)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {

    #[target_feature(enable = "fma")]
    pub(crate) mod avx {
        use std::arch::x86_64::*;

        #[inline]
        pub(crate) fn normalize_f32(vector: &[f32]) -> f32 {
            unsafe {
                let mut sums = _mm256_setzero_ps();
                for i in (0..vector.len()).step_by(8) {
                    // Cache line-aligned
                    let x = _mm256_load_ps(vector.as_ptr().add(i));
                    sums = _mm256_fmadd_ps(x, x, sums);
                }
                add_fma(sums).sqrt()
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    pub(crate) mod neon {
        use std::arch::aarch64::*;

        #[inline]
        pub(crate) fn normalize_f32(vector: &[f32]) -> f32 {
            unsafe {
                let buf = [0.0_f32; 4];
                let mut sum = vld1q_f32(buf.as_ptr());
                let n = vector.len();
                for i in (0..n).step_by(4) {
                    let x = vld1q_f32(vector.as_ptr().add(i));
                    sum = vfmaq_f32(sum, x, x);
                }
                vaddvq_f32(sum).sqrt()
            }
        }
    }
}
