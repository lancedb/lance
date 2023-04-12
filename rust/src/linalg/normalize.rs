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
use num_traits::real::Real;

/// Normalize a vector.
///
/// ```text
/// || x || = sqrt(\sum_i x_i^2)
/// ```
pub trait Normalize<O: Real> {
    /// Normalize a vector.
    fn norm(&self) -> O;
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn normalize_neon_f32(vector: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let buf = [0.0_f32; 4];
    let mut sum = vld1q_f32(buf.as_ptr());
    let n = vector.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        let x = vld1q_f32(vector.as_ptr().add(i));
        sum = vfmaq_f32(sum, x, x);
    }
    let mut sum = vaddvq_f32(sum);
    sum += vector[n..].iter().map(|v| v * v).sum::<f32>();
    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn normalize_fma(vector: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    use super::add::add_fma_f32;

    let len = vector.len() / 8 * 8;

    let mut sums = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        // Cache line-aligned
        let x = _mm256_loadu_ps(vector.as_ptr().add(i));
        sums = _mm256_fmadd_ps(x, x, sums);
    }
    let mut sums = add_fma_f32(sums);
    sums += vector[len..].iter().map(|v| v * v).sum::<f32>();
    sums.sqrt()
}

impl Normalize<f32> for [f32] {
    #[inline]
    fn norm(&self) -> f32 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            normalize_neon_f32(self)
        }

        #[cfg(target_arch = "x86_64")]
        {
            unsafe { normalize_fma(self) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

impl Normalize<f32> for Float32Array {
    #[inline]
    fn norm(&self) -> f32 {
        self.values().norm()
    }
}
