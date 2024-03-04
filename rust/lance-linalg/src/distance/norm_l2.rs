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

use std::{iter::Sum, ops::AddAssign};

use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use half::{bf16, f16};
use lance_arrow::{bfloat16::BFloat16Type, ArrowFloatType, FloatToArrayType};
#[cfg(feature = "fp16kernels")]
use lance_core::utils::cpu::SimdSupport;
#[allow(unused_imports)]
use lance_core::utils::cpu::FP16_SIMD_SUPPORT;
use num_traits::{AsPrimitive, Float};

use crate::simd::{
    f32::{f32x16, f32x8},
    SIMD,
};

/// L2 normalization
pub trait Normalize: ArrowFloatType {
    /// L2 Normalization over a Vector.
    fn norm_l2(vector: &[Self::Native]) -> f32;
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
    }
}

impl Normalize for Float16Type {
    #[inline]
    fn norm_l2(vector: &[Self::Native]) -> f32 {
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
            _ => norm_l2_impl::<f16, f32, 32>(vector),
        }
    }
}

impl Normalize for BFloat16Type {
    #[inline]
    fn norm_l2(vector: &[Self::Native]) -> f32 {
        norm_l2_impl::<bf16, f32, 32>(vector)
    }
}

impl Normalize for Float32Type {
    #[inline]
    fn norm_l2(vector: &[Self::Native]) -> f32 {
        let dim = vector.len();
        if dim % 16 == 0 {
            let mut sum = f32x16::zeros();
            for i in (0..dim).step_by(16) {
                let x = unsafe { f32x16::load_unaligned(vector.as_ptr().add(i)) };
                sum += x * x;
            }
            sum.reduce_sum().sqrt()
        } else if dim % 8 == 0 {
            let mut sum = f32x8::zeros();
            for i in (0..dim).step_by(8) {
                let x = unsafe { f32x8::load_unaligned(vector.as_ptr().add(i)) };
                sum += x * x;
            }
            sum.reduce_sum().sqrt()
        } else {
            // Fallback to scalar
            norm_l2_impl::<f32, f32, 16>(vector)
        }
    }
}

impl Normalize for Float64Type {
    #[inline]
    fn norm_l2(vector: &[Self::Native]) -> f32 {
        norm_l2_impl::<f64, f64, 8>(vector) as f32
    }
}

/// NOTE: this is only pub for benchmarking purposes
#[inline]
pub fn norm_l2_impl<
    T: AsPrimitive<Output>,
    Output: Float + Sum + 'static + AddAssign,
    const LANES: usize,
>(
    vector: &[T],
) -> Output {
    let chunks = vector.chunks_exact(LANES);
    let sum = if chunks.remainder().is_empty() {
        Output::zero()
    } else {
        chunks
            .remainder()
            .iter()
            .map(|&v| v.as_().powi(2))
            .sum::<Output>()
    };
    let mut sums = [Output::zero(); LANES];
    for chunk in chunks {
        for i in 0..LANES {
            sums[i] += chunk[i].as_().powi(2);
        }
    }
    (sum + sums.iter().copied().sum::<Output>()).sqrt()
}

/// Normalize a vector.
///
/// The parameters must be cache line aligned. For example, from
/// Arrow Arrays, i.e., Float32Array
#[inline]
pub fn norm_l2<T: FloatToArrayType>(vector: &[T]) -> f32
where
    T::ArrowType: Normalize,
{
    T::ArrowType::norm_l2(vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{arbitrary_bf16, arbitrary_f16, arbitrary_f32, arbitrary_f64};
    use proptest::prelude::*;

    /// Reference implementation of L2 norm.
    fn norm_l2_reference(data: &[f64]) -> f32 {
        data.iter().map(|v| (*v * *v)).sum::<f64>().sqrt() as f32
    }

    fn do_norm_l2_test<T: FloatToArrayType>(data: &[T]) -> std::result::Result<(), TestCaseError>
    where
        T::ArrowType: Normalize,
    {
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
