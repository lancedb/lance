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

use std::fmt::Formatter;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::SIMD;

#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct i32x8(int32x4x2_t);

impl std::fmt::Debug for i32x8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0; 8];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        write!(f, "i32x8({:?})", arr)
    }
}

impl From<&[i32]> for i32x8 {
    fn from(value: &[i32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl From<&[i32; 8]> for i32x8 {
    fn from(value: &[i32; 8]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<i32, 8> for i32x8 {
    #[inline]
    fn splat(val: i32) -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(int32x4x2_t(vdupq_n_s32(val), vdupq_n_s32(val)))
        }
    }

    #[inline]
    fn zeros() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_setzero_i32())
        }
        #[cfg(target_arch = "aarch64")]
        Self::splat(0)
    }

    #[inline]
    fn gather(slice: &[i32], indices: &Self) -> Self {
        todo!()
    }

    #[inline]
    unsafe fn load(ptr: *const i32) -> Self {
        #[cfg(target_arch = "aarch64")]
        Self(vld1q_s32_x2(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const i32) -> Self {
        #[cfg(target_arch = "aarch64")]
        Self(vld1q_s32_x2(ptr))
    }

    unsafe fn store(&self, ptr: *mut i32) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_s32_x2(ptr, self.0)
        }
    }

    unsafe fn store_unaligned(&self, ptr: *mut i32) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_s32_x2(ptr, self.0)
        }
    }

    fn reduce_sum(&self) -> i32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let sum = vaddq_s32(self.0 .0, self.0 .1);
            vaddvq_s32(sum)
        }
    }

    fn reduce_min(&self) -> i32 {
        todo!()
    }

    fn min(&self, rhs: &Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(int32x4x2_t(
                vminq_s32(self.0 .0, rhs.0 .0),
                vminq_s32(self.0 .1, rhs.0 .1),
            ))
        }
    }
}

impl Add for i32x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(int32x4x2_t(
                vaddq_s32(self.0 .0, rhs.0 .0),
                vaddq_s32(self.0 .1, rhs.0 .1),
            ))
        }
    }
}

impl AddAssign for i32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vaddq_s32(self.0 .0, rhs.0 .0);
            self.0 .1 = vaddq_s32(self.0 .1, rhs.0 .1);
        }
    }
}

impl Sub for i32x8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(int32x4x2_t(
                vsubq_s32(self.0 .0, rhs.0 .0),
                vsubq_s32(self.0 .1, rhs.0 .1),
            ))
        }
    }
}

impl SubAssign for i32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vsubq_s32(self.0 .0, rhs.0 .0);
            self.0 .1 = vsubq_s32(self.0 .1, rhs.0 .1);
        }
    }
}

impl Mul for i32x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(int32x4x2_t(
                vmulq_s32(self.0 .0, rhs.0 .0),
                vmulq_s32(self.0 .1, rhs.0 .1),
            ))
        }
    }
}

#[cfg(test)]
mod tests {}
