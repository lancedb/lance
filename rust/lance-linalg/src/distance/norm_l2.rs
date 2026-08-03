// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{iter::Sum, ops::AddAssign};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, Float32Array};
use arrow_schema::{ArrowError, DataType};
use half::{bf16, f16};
#[allow(unused_imports)]
use lance_core::utils::cpu::SIMD_SUPPORT;
#[cfg(feature = "fp16kernels")]
use lance_core::utils::cpu::SimdSupport;
use num_traits::{AsPrimitive, Float, Num};

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
    unsafe extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn norm_l2_f16_neon(ptr: *const f16, len: u32) -> f32;
        #[cfg(all(kernel_support = "avx512_f16", target_arch = "x86_64"))]
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
        match *SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                kernel::norm_l2_f16_neon(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_support = "avx512_f16",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512FP16 => unsafe {
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

#[cfg(feature = "fp16kernels")]
mod bf16_kernel {
    use half::bf16;

    unsafe extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn norm_l2_bf16_neon(ptr: *const bf16, len: u32) -> f32;
        #[cfg(all(kernel_support = "avx512_bf16", target_arch = "x86_64"))]
        pub fn norm_l2_bf16_avx512(ptr: *const bf16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn norm_l2_bf16_avx2(ptr: *const bf16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn norm_l2_bf16_lsx(ptr: *const bf16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn norm_l2_bf16_lasx(ptr: *const bf16, len: u32) -> f32;
    }
}

impl Normalize for bf16 {
    #[inline]
    fn norm_l2(vector: &[Self]) -> f32 {
        match *SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                bf16_kernel::norm_l2_bf16_neon(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_support = "avx512_bf16",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512FP16 => unsafe {
                bf16_kernel::norm_l2_bf16_avx512(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 | SimdSupport::Avx512 => unsafe {
                bf16_kernel::norm_l2_bf16_avx2(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lasx => unsafe {
                bf16_kernel::norm_l2_bf16_lasx(vector.as_ptr(), vector.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lsx => unsafe {
                bf16_kernel::norm_l2_bf16_lsx(vector.as_ptr(), vector.len() as u32)
            },
            _ => norm_l2_impl::<Self, f32, 32>(vector),
        }
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
        norm_l2_f64_simd(vector)
    }
}

/// Explicit SIMD implementation of L2 norm for f64.
///
/// Two-level unrolling: f64x8 main loop, f64x4 remainder, scalar tail.
#[inline]
pub fn norm_l2_f64_simd(vector: &[f64]) -> f32 {
    use crate::simd::f64::{f64x4, f64x8};
    use crate::simd::{FloatSimd, SIMD};

    let dim = vector.len();
    let unrolled_len = dim / 8 * 8;

    let mut acc8 = f64x8::zeros();
    for i in (0..unrolled_len).step_by(8) {
        unsafe {
            let v = f64x8::load_unaligned(vector.as_ptr().add(i));
            acc8.multiply_add(v, v);
        }
    }

    let aligned_len = dim / 4 * 4;
    let mut acc4 = f64x4::zeros();
    for i in (unrolled_len..aligned_len).step_by(4) {
        unsafe {
            let v = f64x4::load_unaligned(vector.as_ptr().add(i));
            acc4.multiply_add(v, v);
        }
    }

    let tail: f64 = vector[aligned_len..].iter().map(|&v| v * v).sum();
    (acc8.reduce_sum() + acc4.reduce_sum() + tail).sqrt() as f32
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
pub fn norm_l2<T: Normalize>(vector: &[T]) -> f32 {
    T::norm_l2(vector)
}

/// L2 norm of every vector in a [FixedSizeListArray], Returns one norm per row.
pub fn norm_l2_fsl(fsl: &FixedSizeListArray) -> crate::Result<Float32Array> {
    let dim = fsl.value_length() as usize;
    if dim == 0 {
        return Err(ArrowError::InvalidArgumentError(
            "cannot compute L2 norms of a FixedSizeListArray with value_length 0".into(),
        ));
    }
    let values = fsl.values();
    Ok(match fsl.value_type() {
        DataType::Float16 => values
            .as_primitive::<Float16Type>()
            .values()
            .chunks_exact(dim)
            .map(<f16 as Normalize>::norm_l2)
            .collect(),
        DataType::Float32 => values
            .as_primitive::<Float32Type>()
            .values()
            .chunks_exact(dim)
            .map(<f32 as Normalize>::norm_l2)
            .collect(),
        DataType::Float64 => values
            .as_primitive::<Float64Type>()
            .values()
            .chunks_exact(dim)
            .map(<f64 as Normalize>::norm_l2)
            .collect(),
        value_type => {
            return Err(ArrowError::SchemaError(format!(
                "norm_l2_fsl only supports float16/float32/float64 vectors, got: {value_type}"
            )));
        }
    })
}

pub fn norm_squared_fsl(fsl: &FixedSizeListArray) -> Vec<f32> {
    let dim = fsl.value_length() as usize;
    match fsl.value_type() {
        DataType::Float16 => fsl
            .values()
            .as_primitive::<Float16Type>()
            .values()
            .chunks_exact(dim)
            .map(|v| v.iter().map(|v| v * v).sum::<f16>().to_f32())
            .collect::<Vec<_>>(),
        DataType::Float32 => fsl
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .chunks_exact(dim)
            .map(|v| v.iter().map(|v| v * v).sum::<f32>())
            .collect::<Vec<_>>(),
        DataType::Float64 => fsl
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .chunks_exact(dim)
            .map(|v| v.iter().map(|v| v * v).sum::<f64>() as f32)
            .collect::<Vec<_>>(),
        _ => {
            unimplemented!("Unsupported data type: {}", fsl.value_type())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{arbitrary_bf16, arbitrary_f16, arbitrary_f32, arbitrary_f64};
    use arrow_array::{Float16Array, Float64Array, UInt8Array};
    use lance_arrow::FixedSizeListArrayExt;
    use num_traits::ToPrimitive;
    use proptest::prelude::*;

    /// Reference implementation of L2 norm.
    fn norm_l2_reference(data: &[f64]) -> f32 {
        data.iter().map(|v| *v * *v).sum::<f64>().sqrt() as f32
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

    #[test]
    fn test_norm_l2_fsl_f32() {
        let values = Float32Array::from(vec![3.0, 4.0, 6.0, 8.0]);
        let fsl = FixedSizeListArray::try_new_from_values(values, 2).unwrap();

        let norms = norm_l2_fsl(&fsl).unwrap();
        assert_eq!(norms.len(), 2);
        assert!(approx::relative_eq!(
            norms.value(0),
            5.0,
            max_relative = 1e-6
        ));
        assert!(approx::relative_eq!(
            norms.value(1),
            10.0,
            max_relative = 1e-6
        ));
    }

    #[test]
    fn test_norm_l2_fsl_f16_and_f64_match_norm_l2() {
        // The FSL helper must agree with the per-vector `norm_l2` kernel for
        // every supported value type, since callers cache these norms and feed
        // them to `cosine_with_norms`.
        let f16_vals: Vec<f16> = [3.0f32, 4.0, 6.0, 8.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let f16_fsl =
            FixedSizeListArray::try_new_from_values(Float16Array::from(f16_vals), 2).unwrap();
        let f16_norms = norm_l2_fsl(&f16_fsl).unwrap();
        assert_eq!(
            f16_norms.value(0),
            norm_l2(&[f16::from_f32(3.0), f16::from_f32(4.0)])
        );
        assert_eq!(
            f16_norms.value(1),
            norm_l2(&[f16::from_f32(6.0), f16::from_f32(8.0)])
        );

        let f64_fsl = FixedSizeListArray::try_new_from_values(
            Float64Array::from(vec![3.0, 4.0, 6.0, 8.0]),
            2,
        )
        .unwrap();
        let f64_norms = norm_l2_fsl(&f64_fsl).unwrap();
        assert_eq!(f64_norms.value(0), norm_l2(&[3.0f64, 4.0]));
        assert_eq!(f64_norms.value(1), norm_l2(&[6.0f64, 8.0]));
    }

    #[test]
    fn test_norm_l2_fsl_rejects_unsupported_value_type() {
        let fsl = FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![1u8, 2, 3, 4]), 2)
            .unwrap();
        let err = norm_l2_fsl(&fsl).unwrap_err().to_string();
        assert!(err.contains("float16/float32/float64"), "got: {err}");
    }
}
