// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{iter::Sum, ops::AddAssign};

use half::{bf16, f16};
#[cfg(feature = "fp16kernels")]
use lance_core::utils::cpu::SimdSupport;
#[allow(unused_imports)]
use lance_core::utils::cpu::FP16_SIMD_SUPPORT;
use num_traits::{AsPrimitive, Num};

/// L2 normalization
pub trait Normalize: Num {
    /// L2 Normalization over a Vector.
    fn norm_l2(vector: &[Self]) -> f32;
}

#[cfg(feature = "fp16kernels")]
mod kernel {
    use super::*;

    // These are the `norm_l2_f16` function in f16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn norm_l2_f16_neon(ptr: *const f16, len: u32) -> f32;
        #[cfg(all(kernel_suppport = "avx512", target_arch = "x86_64"))]
        pub fn norm_l2_f16_avx512(ptr: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn norm_l2_f16_avx2(ptr: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn norm_l2_f16_lsx(ptr: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn norm_l2_f16_lasx(ptr: *const f16, len: u32) -> f32;
    }
}

impl Normalize for u8 {
    #[inline]
    fn norm_l2(vector: &[Self]) -> f32 {
        norm_l2_impl::<Self, f32, 16>(vector)
    }
}

impl Normalize for f16 {
    #[inline]
    fn norm_l2(vector: &[Self]) -> f32 {
        match *FP16_SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                kernel::norm_l2_f16_neon(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_suppport = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512 => unsafe {
                kernel::norm_l2_f16_avx512(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 => unsafe {
                kernel::norm_l2_f16_avx2(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lasx => unsafe {
                kernel::norm_l2_f16_lasx(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lsx => unsafe {
                kernel::norm_l2_f16_lsx(vector.as_ptr(), vector.len() as u32)
            },
            _ => norm_l2_impl::<Self, f32, 32>(vector),
        }
    }
}

impl Normalize for bf16 {
    #[inline]
    fn norm_l2(vector: &[Self]) -> f32 {
        norm_l2_impl::<Self, f32, 32>(vector)
    }
}

impl Normalize for f32 {
    #[inline]
    fn norm_l2(vector: &[Self]) -> f32 {
        norm_l2_impl::<Self, Self, 16>(vector)
    }
}

impl Normalize for f64 {
    #[inline]
    fn norm_l2(vector: &[Self]) -> f32 {
        norm_l2_impl::<Self, Self, 8>(vector)
    }
}

/// NOTE: this is only pub for benchmarking purposes
#[inline]
pub fn norm_l2_impl<
    T: AsPrimitive<Output>,
    Output: AsPrimitive<f32> + Num + Copy + Sum + 'static + AddAssign,
    const LANES: usize,
>(
    vector: &[T],
) -> f32 {
    let chunks = vector.chunks_exact(LANES);
    let sum = if chunks.remainder().is_empty() {
        Output::zero()
    } else {
        chunks
            .remainder()
            .iter()
            .map(|&v| v.as_() * v.as_())
            .sum::<Output>()
    };
    let mut sums = [Output::zero(); LANES];
    for chunk in chunks {
        for i in 0..LANES {
            sums[i] += chunk[i].as_() * chunk[i].as_();
        }
    }
    (sum + sums.iter().copied().sum::<Output>()).as_().sqrt()
}

/// Normalize a vector.
///
/// The parameters must be cache line aligned. For example, from
/// Arrow Arrays, i.e., Float32Array
#[inline]
pub fn norm_l2<T: Normalize>(vector: &[T]) -> f32 {
    T::norm_l2(vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{arbitrary_bf16, arbitrary_f16, arbitrary_f32, arbitrary_f64};
    use num_traits::ToPrimitive;
    use proptest::prelude::*;

    /// Reference implementation of L2 norm.
    fn norm_l2_reference(data: &[f64]) -> f32 {
        data.iter().map(|v| (*v * *v)).sum::<f64>().sqrt() as f32
    }

    fn do_norm_l2_test<T: Normalize + ToPrimitive>(
        data: &[T],
    ) -> std::result::Result<(), TestCaseError> {
        let f64_data = data
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect::<Vec<f64>>();

        let result = norm_l2(data);
        let reference = norm_l2_reference(&f64_data);

        prop_assert!(approx::relative_eq!(result, reference, max_relative = 1e-6));
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn test_l2_norm_f16(data in prop::collection::vec(arbitrary_f16(), 4..4048)) {
            do_norm_l2_test(&data)?;
        }

        #[test]
        fn test_l2_norm_bf16(data in prop::collection::vec(arbitrary_bf16(), 4..4048)){
            do_norm_l2_test(&data)?;
        }

        #[test]
        fn test_l2_norm_f32(data in prop::collection::vec(arbitrary_f32(), 4..4048)){
            do_norm_l2_test(&data)?;
        }

        #[test]
        fn test_l2_norm_f64(data in prop::collection::vec(arbitrary_f64(), 4..4048)){
            do_norm_l2_test(&data)?;
        }
    }
}
