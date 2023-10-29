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

//! `f32x8`, 8 of f32 values.s

use std::fmt::Formatter;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4x2_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32_x2, vmulq_f32,
    vst1q_f32_x2, vsubq_f32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use super::SIMD;

/// 8 of 32-bit `f32` values. Use 256-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct f32x8(std::arch::x86_64::__m256);

/// 8 of 32-bit `f32` values. Use 256-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct f32x8(float32x4x2_t);

impl std::fmt::Debug for f32x8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0.0_f32; 8];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        write!(f, "f32x8({:?})", arr)
    }
}

impl SIMD<f32> for f32x8 {
    fn splat(val: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_set1_ps(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(vdupq_n_f32(val), vdupq_n_f32(val)))
        }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_load_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        Self::loadu(ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_loadu_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        Self(vld1q_f32_x2(ptr))
    }

    unsafe fn store(&self, ptr: *mut f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_store_ps(ptr, self.0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x2(ptr, self.0);
        }
    }

    unsafe fn store_unaligned(&self, ptr: *mut f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_storeu_ps(ptr, self.0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x2(ptr, self.0);
        }
    }

    fn argmin(&self) -> u32 {
        todo!()
    }

    fn argmax(&self) -> u32 {
        todo!()
    }

    fn multiply_add(&mut self, a: Self, b: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm256_fmadd_ps(a.0, b.0, self.0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vfmaq_f32(self.0 .0, a.0 .0, b.0 .0);
            self.0 .1 = vfmaq_f32(self.0 .1, a.0 .1, b.0 .1);
        }
    }
    #[inline]
    fn reduce_sum(&self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut sum = self.0;
            // Shift and add vector, until only 1 value left.
            // sums = [x0-x7], shift = [x4-x7]
            let mut shift = _mm256_permute2f128_ps(sum, sum, 1);
            // [x0+x4, x1+x5, ..]
            sum = _mm256_add_ps(sum, shift);
            shift = _mm256_permute_ps(sum, 14);
            sum = _mm256_add_ps(sum, shift);
            sum = _mm256_hadd_ps(sum, sum);
            let mut results: [f32; 8] = [0f32; 8];
            _mm256_storeu_ps(results.as_mut_ptr(), sum);
            results[0]
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let sum = vaddq_f32(self.0 .0, self.0 .1);
            vaddvq_f32(sum)
        }
    }
}

impl Add for f32x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_add_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vaddq_f32(self.0 .0, rhs.0 .0),
                vaddq_f32(self.0 .1, rhs.0 .1),
            ))
        }
    }
}

impl AddAssign for f32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm256_add_ps(self.0, rhs.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vaddq_f32(self.0 .0, rhs.0 .0);
            self.0 .1 = vaddq_f32(self.0 .1, rhs.0 .1);
        }
    }
}

impl Sub for f32x8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_sub_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vsubq_f32(self.0 .0, rhs.0 .0),
                vsubq_f32(self.0 .1, rhs.0 .1),
            ))
        }
    }
}

impl SubAssign for f32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm256_sub_ps(self.0, rhs.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vsubq_f32(self.0 .0, rhs.0 .0);
            self.0 .1 = vsubq_f32(self.0 .1, rhs.0 .1);
        }
    }
}

impl Mul for f32x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_mul_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vmulq_f32(self.0 .0, rhs.0 .0),
                vmulq_f32(self.0 .1, rhs.0 .1),
            ))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = (0..8).map(|f| f as f32).collect::<Vec<_>>();
        let b = (10..18).map(|f| f as f32).collect::<Vec<_>>();

        let mut simd_a = f32x8::load_unaligned(a.as_ptr());
        let simd_b = f32x8::load_unaligned(b.as_ptr());
        simd_a -= simd_b;
        assert_eq!(simd_a.reduce_sum(), -80.0);

        let mut simd_power = f32x8::splat(0.0);
        simd_power.multiply_add(simd_a, simd_a);

        assert_eq!(
            "f32x8([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])",
            format!("{:?}", simd_power)
        );
    }
}
