// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dot product.

use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::Arc;

use crate::Error;
use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use half::{bf16, f16};
use lance_arrow::{ArrowFloatType, FloatArray};
#[cfg(feature = "fp16kernels")]
use lance_core::utils::cpu::SimdSupport;
use lance_core::utils::cpu::FP16_SIMD_SUPPORT;
use num_traits::{real::Real, AsPrimitive, Num};

use crate::simd::{
    f32::{f32x16, f32x8},
    SIMD,
};
use crate::Result;

/// Default implementation of dot product.
///
// The following code has been tuned for auto-vectorization.
// Please make sure run `cargo bench --bench dot` with and without AVX-512 before any change.
// Tested `target-features`: avx512f,avx512vl,f16c
#[inline]
fn dot_scalar<
    T: AsPrimitive<Output>,
    Output: Real + Sum + AddAssign + 'static,
    const LANES: usize,
>(
    from: &[T],
    to: &[T],
) -> Output {
    let x_chunks = to.chunks_exact(LANES);
    let y_chunks = from.chunks_exact(LANES);
    let sum = if x_chunks.remainder().is_empty() {
        Output::zero()
    } else {
        x_chunks
            .remainder()
            .iter()
            .zip(y_chunks.remainder().iter())
            .map(|(&x, &y)| x.as_() * y.as_())
            .sum::<Output>()
    };
    // Use known size to allow LLVM to kick in auto-vectorization.
    let mut sums = [Output::zero(); LANES];
    for (x, y) in x_chunks.zip(y_chunks) {
        for i in 0..LANES {
            sums[i] += x[i].as_() * y[i].as_();
        }
    }
    sum + sums.iter().copied().sum::<Output>()
}

/// Dot product.
#[inline]
pub fn dot<T: Dot>(from: &[T], to: &[T]) -> f32 {
    T::dot(from, to)
}

/// Negative [Dot] distance.
#[inline]
pub fn dot_distance<T: Dot>(from: &[T], to: &[T]) -> f32 {
    1.0 - T::dot(from, to)
}

/// Dot product
pub trait Dot: Num {
    /// Dot product.
    fn dot(x: &[Self], y: &[Self]) -> f32;
}

impl Dot for bf16 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        dot_scalar::<Self, f32, 32>(x, y)
    }
}

#[cfg(feature = "fp16kernels")]
mod kernel {
    use super::*;

    // These are the `dot_f16` function in f16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn dot_f16_neon(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(all(kernel_support = "avx512", target_arch = "x86_64"))]
        pub fn dot_f16_avx512(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn dot_f16_avx2(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn dot_f16_lsx(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn dot_f16_lasx(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
    }
}

impl Dot for f16 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        match *FP16_SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                kernel::dot_f16_neon(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_support = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512 => unsafe {
                kernel::dot_f16_avx512(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 => unsafe {
                kernel::dot_f16_avx2(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lasx => unsafe {
                kernel::dot_f16_lasx(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lsx => unsafe {
                kernel::dot_f16_lsx(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            _ => dot_scalar::<Self, f32, 16>(x, y),
        }
    }
}

impl Dot for f32 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        // Manually unrolled 8 times to get enough registers.
        // TODO: avx512 can unroll more
        let x_unrolled_chunks = x.chunks_exact(64);
        let y_unrolled_chunks = y.chunks_exact(64);

        // 8 float32 SIMD
        let x_aligned_chunks = x_unrolled_chunks.remainder().chunks_exact(8);
        let y_aligned_chunks = y_unrolled_chunks.remainder().chunks_exact(8);

        let sum = if x_aligned_chunks.remainder().is_empty() {
            0.0
        } else {
            debug_assert_eq!(
                x_aligned_chunks.remainder().len(),
                y_aligned_chunks.remainder().len()
            );
            x_aligned_chunks
                .remainder()
                .iter()
                .zip(y_aligned_chunks.remainder().iter())
                .map(|(&x, &y)| x * y)
                .sum()
        };

        let mut sum8 = f32x8::zeros();
        x_aligned_chunks
            .zip(y_aligned_chunks)
            .for_each(|(x_chunk, y_chunk)| unsafe {
                let x1 = f32x8::load_unaligned(x_chunk.as_ptr());
                let y1 = f32x8::load_unaligned(y_chunk.as_ptr());
                sum8 += x1 * y1;
            });

        let mut sum16 = f32x16::zeros();
        x_unrolled_chunks
            .zip(y_unrolled_chunks)
            .for_each(|(x, y)| unsafe {
                let x1 = f32x16::load_unaligned(x.as_ptr());
                let x2 = f32x16::load_unaligned(x.as_ptr().add(16));
                let x3 = f32x16::load_unaligned(x.as_ptr().add(32));
                let x4 = f32x16::load_unaligned(x.as_ptr().add(48));

                let y1 = f32x16::load_unaligned(y.as_ptr());
                let y2 = f32x16::load_unaligned(y.as_ptr().add(16));
                let y3 = f32x16::load_unaligned(y.as_ptr().add(32));
                let y4 = f32x16::load_unaligned(y.as_ptr().add(48));

                sum16 += (x1 * y1 + x2 * y2) + (x3 * y3 + x4 * y4);
            });
        sum16.reduce_sum() + sum8.reduce_sum() + sum
    }
}

impl Dot for f64 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        dot_scalar::<Self, Self, 8>(x, y) as f32
    }
}

impl Dot for u8 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        // TODO: this is not optimized for auto vectorization yet.
        x.iter()
            .zip(y.iter())
            .map(|(&x_i, &y_i)| x_i as u32 * y_i as u32)
            .sum::<u32>() as f32
    }
}

/// Negative dot product, to present the relative order of dot distance.
pub fn dot_distance_batch<'a, T: Dot>(
    from: &'a [T],
    to: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a> {
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);
    Box::new(to.chunks_exact(dimension).map(|v| dot_distance(from, v)))
}

fn do_dot_distance_arrow_batch<T: ArrowFloatType>(
    from: &T::ArrayType,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>>
where
    T::Native: Dot,
{
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values =
        to.values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::InvalidArgumentError(format!(
                "Invalid type: expect {:?} got {:?}",
                from.data_type(),
                to.value_type()
            )))?;

    let dists = to_values
        .as_slice()
        .chunks_exact(dimension)
        .map(|v| dot_distance(from.as_slice(), v));

    Ok(Arc::new(Float32Array::new(
        dists.collect(),
        to.nulls().cloned(),
    )))
}

/// Compute negative dot product distance between a vector and a batch of vectors.
///
/// Null buffer of `to` is propagated to the returned array.
///
/// Parameters
///
/// - `from`: the vector to compute distance from.
/// - `to`: a list of vectors to compute distance to.
///
/// # Panics
///
/// Panics if the length of `from` is not equal to the dimension (value length) of `to`.
pub fn dot_distance_arrow_batch(
    from: &dyn Array,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    match *from.data_type() {
        DataType::Float16 => do_dot_distance_arrow_batch::<Float16Type>(from.as_primitive(), to),
        DataType::Float32 => do_dot_distance_arrow_batch::<Float32Type>(from.as_primitive(), to),
        DataType::Float64 => do_dot_distance_arrow_batch::<Float64Type>(from.as_primitive(), to),
        _ => Err(Error::InvalidArgumentError(format!(
            "Unsupported data type: {:?}",
            from.data_type()
        ))),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::test_utils::{
        arbitrary_bf16, arbitrary_f16, arbitrary_f32, arbitrary_f64, arbitrary_vector_pair,
    };
    use num_traits::{Float, FromPrimitive};
    use proptest::prelude::*;

    #[test]
    fn test_dot() {
        let x: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..120).map(|v| v as f32).collect();

        assert_eq!(f32::dot(&x, &y), dot(&x, &y));

        let x: Vec<f32> = (0..512).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..612).map(|v| v as f32).collect();

        assert_eq!(f32::dot(&x, &y), dot(&x, &y));

        let x: Vec<f16> = (0..20).map(|v| f16::from_i32(v).unwrap()).collect();
        let y: Vec<f16> = (100..120).map(|v| f16::from_i32(v).unwrap()).collect();
        assert_eq!(f16::dot(&x, &y), dot(&x, &y));

        let x: Vec<f64> = (20..40).map(|v| f64::from_i32(v).unwrap()).collect();
        let y: Vec<f64> = (120..140).map(|v| f64::from_i32(v).unwrap()).collect();
        assert_eq!(f64::dot(&x, &y), dot(&x, &y));
    }

    /// Reference implementation of dot product.
    fn dot_scalar_ref(x: &[f64], y: &[f64]) -> f32 {
        x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum::<f64>() as f32
    }

    // Accuracy of dot product depends on the size of the components
    // of the vector.
    // Imagine that each `x_i` can vary by `є * |x_i|`. Similarly for `y_i`.
    // (Basically, it's accurate to ±(1 + є) * |x_i|).
    // Error for `sum(x, y)` is `є_x + є_y`. Error for multiple is `є_x * x + є_y * y`.
    // See: https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
    // The multiplication of `x_i` and `y_i` can vary by `(є * |x_i|) * |y_i| + (є * |y_i|) * |x_i|`.
    // This simplifies to `2 * є * (|x_i| + |y_i|)`.
    // So the error for the sum of all the multiplications is `є * sum(|x_i| + |y_i|)`.
    fn max_error<T: Float + AsPrimitive<f64>>(x: &[f64], y: &[f64]) -> f32 {
        let dot = x
            .iter()
            .cloned()
            .zip(y.iter().cloned())
            .map(|(x, y)| x.abs() * y.abs())
            .sum::<f64>();
        (2.0 * T::epsilon().as_() * dot) as f32
    }

    fn do_dot_test<T: Dot + AsPrimitive<f64> + Float>(
        x: &[T],
        y: &[T],
    ) -> std::result::Result<(), TestCaseError> {
        let f64_x = x.iter().map(|&v| v.as_()).collect::<Vec<f64>>();
        let f64_y = y.iter().map(|&v| v.as_()).collect::<Vec<f64>>();

        let expected = dot_scalar_ref(&f64_x, &f64_y);
        let result = dot(x, y);

        let max_error = max_error::<T>(&f64_x, &f64_y);

        prop_assert!(approx::relative_eq!(expected, result, epsilon = max_error));
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn test_dot_f16((x, y) in arbitrary_vector_pair(arbitrary_f16, 4..4048)) {
            do_dot_test(&x, &y)?;
        }

        #[test]
        fn test_dot_bf16((x, y) in arbitrary_vector_pair(arbitrary_bf16, 4..4048)){
            do_dot_test(&x, &y)?;
        }

        #[test]
        fn test_dot_f32((x, y) in arbitrary_vector_pair(arbitrary_f32, 4..4048)){
            do_dot_test(&x, &y)?;
        }

        #[test]
        fn test_dot_f64((x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)){
            do_dot_test(&x, &y)?;
        }
    }
}
