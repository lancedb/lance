// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `u8x8`, 8 of `u8` values

use std::fmt::Formatter;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "loongarch64")]
use std::arch::loongarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "loongarch64")]
use std::mem::transmute;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use super::{Shuffle, SIMD};

/// 16 of 8-bit `u8` values.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct u8x16(pub __m128i);

/// 16 of 32-bit `f32` values. Use 512-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct u8x16(pub uint8x16_t);

impl u8x16 {
    #[inline]
    pub fn bit_and(self, mask: u8) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_and_si128(self.0, _mm_set1_epi8(mask as i8)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vandq_u8(self.0, vdupq_n_u8(mask)))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfand_b(self.0, mask))
        }
    }

    #[inline]
    pub fn right_shift_4(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_srli_epi16(self.0, 4))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vshrq_n_u8::<4>(self.0))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfrsh_b(self.0, 4))
        }
    }
}

impl std::fmt::Debug for u8x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0u8; 16];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        write!(f, "u8x16({:?})", arr)
    }
}

impl From<&[u8]> for u8x16 {
    fn from(value: &[u8]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [u8; 16]> for u8x16 {
    fn from(value: &'a [u8; 16]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<u8, 16> for u8x16 {
    #[inline]

    fn splat(val: u8) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_set1_epi8(val as i8))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vdupq_n_u8(val))
        }
    }

    #[inline]
    fn zeros() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_setzero_si128())
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::splat(0)
        }
    }

    #[inline]
    unsafe fn load(ptr: *const u8) -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm_loadu_epi8(ptr as *const i8))
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::load_unaligned(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const u8) -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm_loadu_epi8(ptr as *const i8))
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self(vld1q_u8(ptr))
        }
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut u8) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            _mm_storeu_epi8(ptr as *mut i8, self.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_u8(ptr, self.0)
        }
    }

    #[inline]

    unsafe fn store_unaligned(&self, ptr: *mut u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_storeu_epi8(ptr as *mut i8, self.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_u8(ptr, self.0)
        }
    }

    fn reduce_sum(&self) -> u8 {
        todo!("the signature of reduce_sum is not correct");
        // #[cfg(target_arch = "x86_64")]
        // unsafe {
        //     let zeros = _mm_setzero_si128();
        //     let sum = _mm_sad_epu8(self.0, zeros);

        //     let lower = _mm_cvtsi128_si64(sum) as u32;
        //     let upper = _mm_extract_epi64(sum, 1) as u32;
        //     lower + upper
        // }
        // #[cfg(target_arch = "aarch64")]
        // unsafe {
        //     let low = vget_low_u8(self.0);
        //     let high = vget_high_u8(self.0);
        //     let sum = vaddl_u8(low, high);
        //     let sum16 = vaddw_u16(vdupq_n_u32(0), sum);
        //     let total = vpadd_u32(vget_low_u32(sum16), vget_high_u32(sum16));
        //     vget_lane_u32(total, 0) + vget_lane_u32(total, 1)
        // }
    }

    #[inline]
    fn reduce_min(&self) -> u8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let low = _mm_and_si128(self.0, _mm_set1_epi8(0xFF));
            let high = _mm_srli_si128(self.0, 8);
            let min_low = _mm_min_epu8(low, high);
            let min_low = _mm_min_epu8(min_low, _mm_srli_si128(min_low, 4));
            let min_low = _mm_min_epu8(min_low, _mm_srli_si128(min_low, 2));
            let min_low = _mm_min_epu8(min_low, _mm_srli_si128(min_low, 1));
            _mm_extract_epi8(min_low, 0) as u8
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            vminvq_u8(self.0)
        }
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_min_epu8(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vminq_u8(self.0, rhs.0))
        }
    }

    fn find(&self, _val: u8) -> Option<i32> {
        todo!()
    }
}

impl Shuffle for u8x16 {
    fn shuffle(&self, indices: u8x16) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_shuffle_epi8(self.0, indices.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vqtbl1q_u8(self.0, indices.0))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvqtbl_b(self.0, indices.0))
        }
    }
}

impl Add for u8x16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_add_epi8(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vqaddq_u8(self.0, rhs.0))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfadd_b(self.0, rhs.0))
        }
    }
}

impl AddAssign for u8x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm_add_epi8(self.0, rhs.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vaddq_u8(self.0, rhs.0)
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfadd_b(self.0, rhs.0)
        }
    }
}

impl Mul for u8x16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_mullo_epi16(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vmulq_u8(self.0, rhs.0))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfmul_b(self.0, rhs.0))
        }
    }
}

impl Sub for u8x16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm_sub_epi8(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(vsubq_u8(self.0, rhs.0))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfsub_b(self.0, rhs.0))
        }
    }
}

impl SubAssign for u8x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm_sub_epi8(self.0, rhs.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vsubq_u8(self.0, rhs.0)
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfsub_b(self.0, rhs.0)
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = (0..16).map(|f| f as u8).collect::<Vec<_>>();
        let b = (16..32).map(|f| f as u8).collect::<Vec<_>>();

        let simd_a = unsafe { u8x16::load_unaligned(a.as_ptr()) };
        let simd_b = unsafe { u8x16::load_unaligned(b.as_ptr()) };

        let simd_add = simd_a + simd_b;
        (0..16)
            .zip(simd_add.as_array().iter())
            .for_each(|(x, &y)| assert_eq!((x + x + 16) as u8, y));

        let simd_mul = simd_a * simd_b;
        (0..16)
            .zip(simd_mul.as_array().iter())
            .for_each(|(x, &y)| assert_eq!((x * (x + 16)) as u8, y));

        let simd_sub = simd_b - simd_a;
        simd_sub.as_array().iter().for_each(|&v| assert_eq!(v, 16));
    }

    #[test]
    fn test_basic_u8x16_ops() {
        let a = (0..16).map(|f| f as u8).collect::<Vec<_>>();
        let b = (16..32).map(|f| f as u8).collect::<Vec<_>>();

        let simd_a = unsafe { u8x16::load_unaligned(a.as_ptr()) };
        let simd_b = unsafe { u8x16::load_unaligned(b.as_ptr()) };

        let simd_add = simd_a + simd_b;
        assert!((0..16)
            .zip(simd_add.as_array().iter())
            .all(|(x, &y)| (x + x + 16) as u8 == y));

        let simd_mul = simd_a * simd_b;
        assert!((0..16)
            .zip(simd_mul.as_array().iter())
            .all(|(x, &y)| (x * (x + 16)) as u8 == y));
    }
}
