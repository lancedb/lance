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

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4x2_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32_x2, vsubq_f32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__mm256;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use super::SIMD;

/// 8 of 32-bit `f32` values. Use 256-bit SIMD if possible.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__mm256;
/// 8 of 32-bit `f32` values. Use 256-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone, Copy)]
pub struct f32x8(float32x4x2_t);

impl SIMD<f32> for f32x8 {
    fn splat(val: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(__mm256::set1_ps(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(vdupq_n_f32(val), vdupq_n_f32(val)))
        }
    }

    #[inline]
    fn load(ptr: *const f32) -> Self {
        Self::loadu(ptr)
    }

    #[inline]
    fn loadu(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(__mm256_loadu_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vld1q_f32_x2(ptr))
        }
    }

    fn argmin(&self) -> u32 {
        todo!()
    }

    fn argmax(&self) -> u32 {
        todo!()
    }

    #[inline]
    fn reduce_sum(&self) -> f32 {
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
            Self(__mm256_add_ps(self.0, rhs.0))
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
            Self(__mm256_add_ps(self.0, rhs.0))
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
            Self(__mm256_sub_ps(self.0, rhs.0))
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
            Self(__mm256_sub_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vsubq_f32(self.0 .0, rhs.0 .0);
            self.0 .1 = vsubq_f32(self.0 .1, rhs.0 .1);
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

        let mut simd_a = f32x8::load(a.as_ptr());
        let simd_b = f32x8::load(b.as_ptr());
        simd_a -= simd_b;
        assert_eq!(simd_a.reduce_sum(), -80.0);
    }
}
