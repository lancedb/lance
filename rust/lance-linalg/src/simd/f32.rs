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

//! `f32x8`, 8 of `f32` values.s

use std::fmt::Formatter;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use super::{i32::i32x8, FloatSimd, SIMD};

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

impl f32x8 {
    pub fn gather(slice: &[f32], indices: &i32x8) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_i32gather_ps::<1>(slice.as_ptr(), indices.0))
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            // aarch64 does not have relevant SIMD instructions.
            let idx = indices.as_array();
            let ptr = slice.as_ptr();

            let values = [
                *ptr.add(idx[0] as usize),
                *ptr.add(idx[1] as usize),
                *ptr.add(idx[2] as usize),
                *ptr.add(idx[3] as usize),
                *ptr.add(idx[4] as usize),
                *ptr.add(idx[5] as usize),
                *ptr.add(idx[6] as usize),
                *ptr.add(idx[7] as usize),
            ];
            Self::load_unaligned(values.as_ptr())
        }
    }
}

impl From<&[f32]> for f32x8 {
    fn from(value: &[f32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f32; 8]> for f32x8 {
    fn from(value: &'a [f32; 8]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f32, 8> for f32x8 {
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

    fn zeros() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_setzero_ps())
        }
        #[cfg(target_arch = "aarch64")]
        Self::splat(0.0)
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_load_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        Self::load_unaligned(ptr)
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

    fn reduce_min(&self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                let mut min = self.0;
                // Shift and add vector, until only 1 value left.
                // sums = [x0-x7], shift = [x4-x7]
                let mut shift = _mm256_permute2f128_ps(min, min, 1);
                // [x0+x4, x1+x5, ..]
                min = _mm256_min_ps(min, shift);
                shift = _mm256_permute_ps(min, 14);
                min = _mm256_min_ps(min, shift);
                let mut results: [f32; 8] = [0f32; 8];
                _mm256_storeu_ps(results.as_mut_ptr(), min);
                results[0]
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let m = vminq_f32(self.0 .0, self.0 .1);
            vminvq_f32(m)
        }
    }

    fn min(&self, rhs: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_min_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vminq_f32(self.0 .0, rhs.0 .0),
                vminq_f32(self.0 .1, rhs.0 .1),
            ))
        }
    }

    fn find(&self, val: f32) -> Option<i32> {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            // let tgt = _mm256_set1_ps(val);
            // let mask = _mm256_cmpeq_ps_mask(self.0, tgt);
            // if mask != 0 {
            //     return Some(mask.trailing_zeros() as i32);
            // }
            todo!()
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            for i in 0..8 {
                if self.as_array().get_unchecked(i) == &val {
                    return Some(i as i32);
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let tgt = vdupq_n_f32(val);
            let mut arr = [0; 8];
            let mask1 = vceqq_f32(self.0 .0, tgt);
            let mask2 = vceqq_f32(self.0 .1, tgt);
            vst1q_u32(arr.as_mut_ptr(), mask1);
            vst1q_u32(arr.as_mut_ptr().add(4), mask2);
            for i in 0..8 {
                if arr.get_unchecked(i) != &0 {
                    return Some(i as i32);
                }
            }
        }
        None
    }
}

impl FloatSimd<f32, 8> for f32x8 {
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

/// 16 of 32-bit `f32` values. Use 512-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
#[derive(Clone, Copy)]
pub struct f32x16(__m256, __m256);
#[allow(non_camel_case_types)]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy)]
pub struct f32x16(__m512);

/// 16 of 32-bit `f32` values. Use 512-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct f32x16(float32x4x4_t);

impl std::fmt::Debug for f32x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0.0_f32; 16];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        write!(f, "f32x16({:?})", arr)
    }
}

impl From<&[f32]> for f32x16 {
    fn from(value: &[f32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f32; 16]> for f32x16 {
    fn from(value: &'a [f32; 16]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f32, 16> for f32x16 {
    #[inline]

    fn splat(val: f32) -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_set1_ps(val))
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_set1_ps(val), _mm256_set1_ps(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vdupq_n_f32(val),
                vdupq_n_f32(val),
                vdupq_n_f32(val),
                vdupq_n_f32(val),
            ))
        }
    }

    #[inline]
    fn zeros() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_setzero_ps())
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_setzero_ps(), _mm256_setzero_ps())
        }
        #[cfg(target_arch = "aarch64")]
        Self::splat(0.0)
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_load_ps(ptr), _mm256_load_ps(ptr.add(8)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_load_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        Self::load_unaligned(ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_loadu_ps(ptr), _mm256_loadu_ps(ptr.add(8)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_loadu_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        Self(vld1q_f32_x4(ptr))
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut f32) {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            _mm512_store_ps(ptr, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            _mm256_store_ps(ptr, self.0);
            _mm256_store_ps(ptr.add(8), self.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x4(ptr, self.0);
        }
    }

    #[inline]

    unsafe fn store_unaligned(&self, ptr: *mut f32) {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            _mm512_storeu_ps(ptr, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            _mm256_storeu_ps(ptr, self.0);
            _mm256_storeu_ps(ptr.add(8), self.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x4(ptr, self.0);
        }
    }

    fn reduce_sum(&self) -> f32 {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            _mm512_mask_reduce_add_ps(0xFFFF, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            let mut sum = _mm256_add_ps(self.0, self.1);
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
            let mut sum1 = vaddq_f32(self.0 .0, self.0 .1);
            let sum2 = vaddq_f32(self.0 .2, self.0 .3);
            sum1 = vaddq_f32(sum1, sum2);
            vaddvq_f32(sum1)
        }
    }

    #[inline]
    fn reduce_min(&self) -> f32 {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            _mm512_mask_reduce_min_ps(0xFFFF, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            let mut m1 = _mm256_min_ps(self.0, self.1);
            let mut m2 = _mm256_permute2f128_ps(m1, m1, 1);
            m1 = _mm256_min_ps(m1, m2);
            m2 = _mm256_permute_ps(m1, 14);
            m1 = _mm256_min_ps(m1, m2);
            let mut results: [f32; 8] = [0f32; 8];
            _mm256_storeu_ps(results.as_mut_ptr(), m1);
            results[0]
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let m1 = vminq_f32(self.0 .0, self.0 .1);
            let m2 = vminq_f32(self.0 .2, self.0 .3);
            let m = vminq_f32(m1, m2);
            vminvq_f32(m)
        }
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_min_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_min_ps(self.0, rhs.0), _mm256_min_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vminq_f32(self.0 .0, rhs.0 .0),
                vminq_f32(self.0 .1, rhs.0 .1),
                vminq_f32(self.0 .2, rhs.0 .2),
                vminq_f32(self.0 .3, rhs.0 .3),
            ))
        }
    }

    fn find(&self, val: f32) -> Option<i32> {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            // let tgt = _mm512_set1_ps(val);
            // let mask = _mm512_cmpeq_ps_mask(self.0, tgt);
            // if mask != 0 {
            //     return Some(mask.trailing_zeros() as i32);
            // }
            todo!()
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            // _mm256_cmpeq_ps_mask requires "avx512l".
            for i in 0..16 {
                if self.as_array().get_unchecked(i) == &val {
                    return Some(i as i32);
                }
            }
            None
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let tgt = vdupq_n_f32(val);
            let mut arr = [0; 16];
            let mask1 = vceqq_f32(self.0 .0, tgt);
            let mask2 = vceqq_f32(self.0 .1, tgt);
            let mask3 = vceqq_f32(self.0 .2, tgt);
            let mask4 = vceqq_f32(self.0 .3, tgt);

            vst1q_u32(arr.as_mut_ptr(), mask1);
            vst1q_u32(arr.as_mut_ptr().add(4), mask2);
            vst1q_u32(arr.as_mut_ptr().add(8), mask3);
            vst1q_u32(arr.as_mut_ptr().add(12), mask4);

            for i in 0..16 {
                if arr.get_unchecked(i) != &0 {
                    return Some(i as i32);
                }
            }
            None
        }
    }
}

impl FloatSimd<f32, 16> for f32x16 {
    #[inline]
    fn multiply_add(&mut self, a: Self, b: Self) {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            self.0 = _mm512_fmadd_ps(a.0, b.0, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            self.0 = _mm256_fmadd_ps(a.0, b.0, self.0);
            self.1 = _mm256_fmadd_ps(a.1, b.1, self.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vfmaq_f32(self.0 .0, a.0 .0, b.0 .0);
            self.0 .1 = vfmaq_f32(self.0 .1, a.0 .1, b.0 .1);
            self.0 .2 = vfmaq_f32(self.0 .2, a.0 .2, b.0 .2);
            self.0 .3 = vfmaq_f32(self.0 .3, a.0 .3, b.0 .3);
        }
    }
}

impl Add for f32x16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_add_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_add_ps(self.0, rhs.0), _mm256_add_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vaddq_f32(self.0 .0, rhs.0 .0),
                vaddq_f32(self.0 .1, rhs.0 .1),
                vaddq_f32(self.0 .2, rhs.0 .2),
                vaddq_f32(self.0 .3, rhs.0 .3),
            ))
        }
    }
}

impl AddAssign for f32x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            self.0 = _mm512_add_ps(self.0, rhs.0)
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            self.0 = _mm256_add_ps(self.0, rhs.0);
            self.1 = _mm256_add_ps(self.1, rhs.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vaddq_f32(self.0 .0, rhs.0 .0);
            self.0 .1 = vaddq_f32(self.0 .1, rhs.0 .1);
            self.0 .2 = vaddq_f32(self.0 .2, rhs.0 .2);
            self.0 .3 = vaddq_f32(self.0 .3, rhs.0 .3);
        }
    }
}

impl Mul for f32x16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_mul_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_mul_ps(self.0, rhs.0), _mm256_mul_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vmulq_f32(self.0 .0, rhs.0 .0),
                vmulq_f32(self.0 .1, rhs.0 .1),
                vmulq_f32(self.0 .2, rhs.0 .2),
                vmulq_f32(self.0 .3, rhs.0 .3),
            ))
        }
    }
}

impl Sub for f32x16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            Self(_mm512_sub_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            Self(_mm256_sub_ps(self.0, rhs.0), _mm256_sub_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vsubq_f32(self.0 .0, rhs.0 .0),
                vsubq_f32(self.0 .1, rhs.0 .1),
                vsubq_f32(self.0 .2, rhs.0 .2),
                vsubq_f32(self.0 .3, rhs.0 .3),
            ))
        }
    }
}

impl SubAssign for f32x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        unsafe {
            self.0 = _mm512_sub_ps(self.0, rhs.0)
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
        unsafe {
            self.0 = _mm256_sub_ps(self.0, rhs.0);
            self.1 = _mm256_sub_ps(self.1, rhs.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 .0 = vsubq_f32(self.0 .0, rhs.0 .0);
            self.0 .1 = vsubq_f32(self.0 .1, rhs.0 .1);
            self.0 .2 = vsubq_f32(self.0 .2, rhs.0 .2);
            self.0 .3 = vsubq_f32(self.0 .3, rhs.0 .3);
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

        let mut simd_a = unsafe { f32x8::load_unaligned(a.as_ptr()) };
        let simd_b = unsafe { f32x8::load_unaligned(b.as_ptr()) };

        let simd_add = simd_a + simd_b;
        assert!((0..8)
            .zip(simd_add.as_array().iter())
            .all(|(x, &y)| (x + x + 10) as f32 == y));

        let simd_mul = simd_a * simd_b;
        assert!((0..8)
            .zip(simd_mul.as_array().iter())
            .all(|(x, &y)| (x * (x + 10)) as f32 == y));

        let simd_sub = simd_b - simd_a;
        assert!(simd_sub.as_array().iter().all(|&v| v == 10.0));

        simd_a -= simd_b;
        assert_eq!(simd_a.reduce_sum(), -80.0);

        let mut simd_power = f32x8::splat(0.0);
        simd_power.multiply_add(simd_a, simd_a);

        assert_eq!(
            "f32x8([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])",
            format!("{:?}", simd_power)
        );
    }

    #[test]
    fn test_f32x8_cmp_ops() {
        let a = [1.0_f32, 2.0, 5.0, 6.0, 7.0, 3.0, 2.0, 1.0];
        let b = [2.0_f32, 1.0, 4.0, 5.0, 9.0, 5.0, 6.0, 2.0];
        let simd_a: f32x8 = (&a).into();
        let simd_b: f32x8 = (&b).into();

        let min_simd = simd_a.min(&simd_b);
        assert_eq!(
            min_simd.as_array(),
            [1.0, 1.0, 4.0, 5.0, 7.0, 3.0, 2.0, 1.0]
        );
        let min_val = min_simd.reduce_min();
        assert_eq!(min_val, 1.0);

        assert_eq!(Some(2), simd_a.find(5.0));
        assert_eq!(Some(1), simd_a.find(2.0));
        assert_eq!(None, simd_a.find(-200.0));
    }

    #[test]
    fn test_basic_f32x16_ops() {
        let a = (0..16).map(|f| f as f32).collect::<Vec<_>>();
        let b = (10..26).map(|f| f as f32).collect::<Vec<_>>();

        let mut simd_a = unsafe { f32x16::load_unaligned(a.as_ptr()) };
        let simd_b = unsafe { f32x16::load_unaligned(b.as_ptr()) };

        let simd_add = simd_a + simd_b;
        assert!((0..16)
            .zip(simd_add.as_array().iter())
            .all(|(x, &y)| (x + x + 10) as f32 == y));

        let simd_mul = simd_a * simd_b;
        assert!((0..16)
            .zip(simd_mul.as_array().iter())
            .all(|(x, &y)| (x * (x + 10)) as f32 == y));

        simd_a -= simd_b;
        assert_eq!(simd_a.reduce_sum(), -160.0);

        let mut simd_power = f32x16::zeros();
        simd_power.multiply_add(simd_a, simd_a);

        assert_eq!(
            format!("f32x16({:?})", [100.0; 16]),
            format!("{:?}", simd_power)
        );
    }

    #[test]
    fn test_f32x16_cmp_ops() {
        let a = [
            1.0_f32, 2.0, 5.0, 6.0, 7.0, 3.0, 2.0, 1.0, -0.5, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0,
        ];
        let b = [
            2.0_f32, 1.0, 4.0, 5.0, 9.0, 5.0, 6.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 1.0,
        ];
        let simd_a: f32x16 = (&a).into();
        let simd_b: f32x16 = (&b).into();

        let min_simd = simd_a.min(&simd_b);
        assert_eq!(
            min_simd.as_array(),
            [1.0, 1.0, 4.0, 5.0, 7.0, 3.0, 2.0, 1.0, -0.5, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 1.0]
        );
        let min_val = min_simd.reduce_min();
        assert_eq!(min_val, -0.5);

        assert_eq!(Some(2), simd_a.find(5.0));
        assert_eq!(Some(1), simd_a.find(2.0));
        assert_eq!(Some(13), simd_a.find(9.0));
        assert_eq!(None, simd_a.find(-200.0));
    }
}
