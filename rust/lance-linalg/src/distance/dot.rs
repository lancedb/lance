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

//! Dot product.

use std::iter::Sum;
use std::ops::{AddAssign, Neg};
use std::sync::Arc;

use crate::Error;
use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use half::{bf16, f16};
use lance_arrow::bfloat16::BFloat16Type;
use lance_arrow::{ArrowFloatType, FloatArray, FloatToArrayType};
use lance_core::utils::cpu::{SimdSupport, FP16_SIMD_SUPPORT};
use num_traits::real::Real;
use num_traits::AsPrimitive;

#[cfg(all(target_os = "linux", feature = "avx512fp16", target_arch = "x86_64"))]
use lance_core::utils::cpu::x86::AVX512_F16_SUPPORTED;

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
fn dot_scalar<T: Real + Sum + AddAssign + AsPrimitive<f32>, const LANES: usize>(
    from: &[T],
    to: &[T],
) -> f32 {
    let x_chunks = to.chunks_exact(LANES);
    let y_chunks = from.chunks_exact(LANES);
    let sum = if x_chunks.remainder().is_empty() {
        T::zero()
    } else {
        x_chunks
            .remainder()
            .iter()
            .zip(y_chunks.remainder().iter())
            .map(|(&x, &y)| x * y)
            .sum::<T>()
    };
    // Use known size to allow LLVM to kick in auto-vectorization.
    let mut sums = [T::zero(); LANES];
    for (x, y) in x_chunks.zip(y_chunks) {
        for i in 0..LANES {
            sums[i] += x[i] * y[i];
        }
    }
    (sum + sums.iter().copied().sum::<T>()).as_()
}

/// Dot product.
#[inline]
pub fn dot<T: FloatToArrayType + Neg<Output = T>>(from: &[T], to: &[T]) -> f32
where
    T::ArrowType: Dot,
{
    T::ArrowType::dot(from, to)
}

/// Negative dot distance.
#[inline]
pub fn dot_distance<T: FloatToArrayType + Neg<Output = T>>(from: &[T], to: &[T]) -> f32
where
    T::ArrowType: Dot,
{
    T::ArrowType::dot(from, to).neg()
}

/// Dot product
pub trait Dot: ArrowFloatType {
    /// Dot product.
    fn dot(x: &[Self::Native], y: &[Self::Native]) -> f32;
}

impl Dot for BFloat16Type {
    #[inline]
    fn dot(x: &[bf16], y: &[bf16]) -> f32 {
        dot_scalar::<bf16, 32>(x, y)
    }
}

mod kernel {
    use super::*;

    // These are the `dot_f16` function in f16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    extern "C" {
        pub fn dot_f16_base(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "aarch64")]
        pub fn dot_f16_neon(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn dot_f16_avx512(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn dot_f16_avx2(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
    }
}

impl Dot for Float16Type {
    #[inline]
    fn dot(x: &[f16], y: &[f16]) -> f32 {
        match *FP16_SIMD_SUPPORT {
            #[cfg(target_arch = "aarch64")]
            SimdSupport::Neon => unsafe {
                kernel::dot_f16_neon(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(target_arch = "x86_64")]
            SimdSupport::Avx512 => unsafe {
                kernel::dot_f16_avx512(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(target_arch = "x86_64")]
            SimdSupport::Avx2 => unsafe {
                kernel::dot_f16_avx2(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            _ => unsafe { kernel::dot_f16_base(x.as_ptr(), y.as_ptr(), x.len() as u32) },
        }
    }
}

impl Dot for Float32Type {
    #[inline]
    fn dot(x: &[f32], y: &[f32]) -> f32 {
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

impl Dot for Float64Type {
    #[inline]
    fn dot(x: &[f64], y: &[f64]) -> f32 {
        dot_scalar::<f64, 8>(x, y)
    }
}

/// Negative dot product, to present the relative order of dot distance.
pub fn dot_distance_batch<'a, T: FloatToArrayType>(
    from: &'a [T],
    to: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a>
where
    T::ArrowType: Dot,
{
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);
    Box::new(to.chunks_exact(dimension).map(|v| dot_distance(from, v)))
}

fn do_dot_distance_arrow_batch<T: ArrowFloatType + Dot>(
    from: &T::ArrayType,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
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
    use num_traits::FromPrimitive;

    #[test]
    fn test_dot() {
        let x: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..120).map(|v| v as f32).collect();

        assert_eq!(Float32Type::dot(&x, &y), dot(&x, &y));

        let x: Vec<f32> = (0..512).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..612).map(|v| v as f32).collect();

        assert_eq!(Float32Type::dot(&x, &y), dot(&x, &y));

        let x: Vec<f16> = (0..20).map(|v| f16::from_i32(v).unwrap()).collect();
        let y: Vec<f16> = (100..120).map(|v| f16::from_i32(v).unwrap()).collect();
        assert_eq!(Float16Type::dot(&x, &y), dot(&x, &y));

        let x: Vec<f64> = (20..40).map(|v| f64::from_i32(v).unwrap()).collect();
        let y: Vec<f64> = (120..140).map(|v| f64::from_i32(v).unwrap()).collect();
        assert_eq!(Float64Type::dot(&x, &y), dot(&x, &y));
    }
}
