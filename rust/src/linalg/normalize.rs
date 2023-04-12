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

use arrow_array::Float32Array;

use super::add::add_fma_f32;

pub trait Normalize {
    type Output;
    /// Normalize a vector.
    fn normalize(&self) -> Self::Output;
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn normalize_neon(vector: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let buf = [0.0_f32; 4];
    let mut sum = vld1q_f32(buf.as_ptr());
    let n = vector.len();
    for i in (0..n).step_by(4) {
        let x = vld1q_f32(vector.as_ptr().add(i));
        sum = vfmaq_f32(sum, x, x);
    }
    vaddvq_f32(sum).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn normalize_fma(vector: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sums = _mm256_setzero_ps();
    for i in (0..vector.len()).step_by(8) {
        // Cache line-aligned
        let x = _mm256_load_ps(vector.as_ptr().add(i));
        sums = _mm256_fmadd_ps(x, x, sums);
    }
    add_fma_f32(sums).sqrt()
}

impl Normalize for [f32] {
    type Output = f32;

    #[inline]
    fn normalize(&self) -> f32 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            return normalize_neon(self);
        }

        #[cfg(target_arch = "x86_64")]
        {
            unsafe { return normalize_fma(self) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

impl Normalize for Float32Array {
    type Output = f32;

    #[inline]
    fn normalize(&self) -> f32 {
        self.values().normalize()
    }
}
