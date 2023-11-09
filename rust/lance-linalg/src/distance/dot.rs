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

use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use half::{bf16, f16};
use lance_arrow::bfloat16::BFloat16Type;
use lance_arrow::{ArrowFloatType, FloatToArrayType};
use num_traits::real::Real;

use crate::simd::{
    f32::{f32x16, f32x8},
    SIMD,
};

/// Default implementation of dot product.
///
// The following code has been tuned for auto-vectorization.
// Please make sure run `cargo bench --bench dot` with and without AVX-512 before any change.
// Tested `target-features`: avx512f,avx512vl,f16c
#[inline]
fn dot_scalar<T: Real + Sum + AddAssign, const LANES: usize>(from: &[T], to: &[T]) -> T {
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
    sum + sums.iter().copied().sum::<T>()
}

/// Dot product.
#[inline]
pub fn dot<T: FloatToArrayType + Neg<Output = T>>(from: &[T], to: &[T]) -> T
where
    T::ArrowType: Dot,
{
    T::ArrowType::dot(from, to)
}

/// Negative dot distance.
#[inline]
pub fn dot_distance<T: FloatToArrayType + Neg<Output = T>>(from: &[T], to: &[T]) -> T
where
    T::ArrowType: Dot,
{
    T::ArrowType::dot(from, to).neg()
}

/// Dot product
pub trait Dot: ArrowFloatType {
    /// Dot product.
    fn dot(x: &[Self::Native], y: &[Self::Native]) -> Self::Native;
}

impl Dot for BFloat16Type {
    #[inline]
    fn dot(x: &[bf16], y: &[bf16]) -> bf16 {
        dot_scalar::<bf16, 16>(x, y)
    }
}

#[cfg(any(
    all(target_os = "macos", target_feature = "neon"),
    all(target_os = "linux", feature = "avx512fp16")
))]
mod kernel {
    use super::*;

    extern "C" {
        pub fn dot_f16(ptr1: *const f16, ptr2: *const f16, len: u32) -> f16;
    }
}

impl Dot for Float16Type {
    #[inline]
    fn dot(x: &[f16], y: &[f16]) -> f16 {
        #[cfg(any(
            all(target_os = "macos", target_feature = "neon"),
            all(target_os = "linux", feature = "avx512fp16")
        ))]
        unsafe {
            self::kernel::dot_f16(x.as_ptr(), y.as_ptr(), x.len() as u32)
        }
        #[cfg(not(any(
            all(target_os = "macos", target_feature = "neon"),
            all(target_os = "linux", feature = "avx512fp16"))
        ))]
        {
            dot_scalar::<f16, 16>(x, y)
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
    fn dot(x: &[f64], y: &[f64]) -> f64 {
        dot_scalar::<f64, 8>(x, y)
    }
}

/// Negative dot product, to present the relative order of dot distance.
pub fn dot_distance_batch<'a, T: FloatToArrayType>(
    from: &'a [T],
    to: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = T> + 'a>
where
    T::ArrowType: Dot,
{
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);
    Box::new(to.chunks_exact(dimension).map(|v| dot_distance(from, v)))
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
pub fn dot_distance_arrow_batch(from: &[f32], to: &FixedSizeListArray) -> Arc<Float32Array> {
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values = to.values().as_primitive::<Float32Type>().values();
    let dists = to_values
        .chunks_exact(dimension)
        .map(|v| dot_distance(from, v));

    Arc::new(Float32Array::new(dists.collect(), to.nulls().cloned()))
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
