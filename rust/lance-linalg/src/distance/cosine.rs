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

//! Cosine distance
//!
//! <https://en.wikipedia.org/wiki/Cosine_similarity>
//!
//! `bf16, f16, f32, f64` types are supported.

use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
    Array, FixedSizeListArray, Float32Array,
};
use arrow_schema::DataType;
use half::f16;
use lance_arrow::bfloat16::BFloat16Type;
use lance_arrow::{ArrowFloatType, FloatArray, FloatToArrayType};
#[cfg(feature = "fp16kernels")]
use lance_core::utils::cpu::SimdSupport;
use lance_core::utils::cpu::FP16_SIMD_SUPPORT;
use num_traits::{AsPrimitive, FromPrimitive};

use super::norm_l2::norm_l2;
use super::{dot::dot, Normalize};
use crate::simd::{
    f32::{f32x16, f32x8},
    FloatSimd, SIMD,
};
use crate::{Error, Result};

/// Cosine Distance
pub trait Cosine: super::dot::Dot + Normalize
where
    Self::Native: AsPrimitive<f64> + AsPrimitive<f32>,
    <Self::Native as FloatToArrayType>::ArrowType: Cosine + super::dot::Dot,
{
    /// Cosine distance between two vectors.
    #[inline]
    fn cosine(x: &[Self::Native], other: &[Self::Native]) -> f32
    where
        <<Self as ArrowFloatType>::Native as FloatToArrayType>::ArrowType: super::dot::Dot,
    {
        let x_norm = norm_l2(x);
        Self::cosine_fast(x, x_norm, other)
    }

    /// Fast cosine function, that assumes that the norm of the first vector is already known.
    #[inline]
    fn cosine_fast(x: &[Self::Native], x_norm: f32, y: &[Self::Native]) -> f32
    where
        <<Self as ArrowFloatType>::Native as FloatToArrayType>::ArrowType: super::dot::Dot,
    {
        cosine_scalar(x, x_norm, y)
    }

    /// Cosine between two vectors, with the L2 norms of both vectors already known.
    #[inline]
    fn cosine_with_norms(x: &[Self::Native], x_norm: f32, y_norm: f32, y: &[Self::Native]) -> f32
    where
        <<Self as ArrowFloatType>::Native as FloatToArrayType>::ArrowType: Cosine,
    {
        cosine_scalar_fast(x, x_norm, y, y_norm)
    }

    fn cosine_batch<'a>(
        x: &'a [Self::Native],
        batch: &'a [Self::Native],
        dimension: usize,
    ) -> Box<dyn Iterator<Item = f32> + 'a> {
        let x_norm = norm_l2(x);

        Box::new(
            batch
                .chunks_exact(dimension)
                .map(move |y| Self::cosine_fast(x, x_norm, y)),
        )
    }
}

impl Cosine for BFloat16Type {}

#[cfg(feature = "fp16kernels")]
mod kernel {
    use super::*;

    // These are the `cosine_f16` function in f16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn cosine_f16_neon(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
        #[cfg(all(kernel_suppport = "avx512", target_arch = "x86_64"))]
        pub fn cosine_f16_avx512(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn cosine_f16_avx2(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
    }
}

impl Cosine for Float16Type {
    fn cosine_fast(x: &[f16], x_norm: f32, y: &[f16]) -> f32 {
        match *FP16_SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                kernel::cosine_f16_neon(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_suppport = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512 => unsafe {
                kernel::cosine_f16_avx512(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 => unsafe {
                kernel::cosine_f16_avx2(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            _ => cosine_scalar(x, x_norm, y),
        }
    }
}

/// f32 kernels for Cosine
mod f32 {
    use super::*;

    // TODO: how can we explicity infer N?
    #[inline]
    pub(super) fn cosine_once<S: SIMD<f32, N>, const N: usize>(
        x: &[f32],
        x_norm: f32,
        y: &[f32],
    ) -> f32 {
        let x = unsafe { S::load_unaligned(x.as_ptr()) };
        let y = unsafe { S::load_unaligned(y.as_ptr()) };
        let y2 = y * y;
        let xy = x * y;
        1.0 - xy.reduce_sum() / x_norm / y2.reduce_sum().sqrt()
    }
}

impl Cosine for Float32Type {
    #[inline]
    fn cosine_fast(x: &[f32], x_norm: f32, other: &[f32]) -> f32 {
        let dim = x.len();
        let unrolled_len = dim / 16 * 16;
        let mut y_norm16 = f32x16::zeros();
        let mut xy16 = f32x16::zeros();
        for i in (0..unrolled_len).step_by(16) {
            unsafe {
                let x = f32x16::load_unaligned(x.as_ptr().add(i));
                let y = f32x16::load_unaligned(other.as_ptr().add(i));
                xy16.multiply_add(x, y);
                y_norm16.multiply_add(y, y);
            }
        }
        let aligned_len = dim / 8 * 8;
        let mut y_norm8 = f32x8::zeros();
        let mut xy8 = f32x8::zeros();
        for i in (unrolled_len..aligned_len).step_by(8) {
            unsafe {
                let x = f32x8::load_unaligned(x.as_ptr().add(i));
                let y = f32x8::load_unaligned(other.as_ptr().add(i));
                xy8.multiply_add(x, y);
                y_norm8.multiply_add(y, y);
            }
        }
        let y_norm =
            y_norm16.reduce_sum() + y_norm8.reduce_sum() + norm_l2(&other[aligned_len..]).powi(2);
        let xy =
            xy16.reduce_sum() + xy8.reduce_sum() + dot(&x[aligned_len..], &other[aligned_len..]);
        1.0 - xy / x_norm / y_norm.sqrt()
    }

    #[inline]
    fn cosine_with_norms(x: &[f32], x_norm: f32, y_norm: f32, y: &[f32]) -> Self::Native {
        let dim = x.len();
        let unrolled_len = dim / 16 * 16;
        let mut xy16 = f32x16::zeros();
        for i in (0..unrolled_len).step_by(16) {
            unsafe {
                let x = f32x16::load_unaligned(x.as_ptr().add(i));
                let y = f32x16::load_unaligned(y.as_ptr().add(i));
                xy16.multiply_add(x, y);
            }
        }
        let aligned_len = dim / 8 * 8;
        let mut xy8 = f32x8::zeros();
        for i in (unrolled_len..aligned_len).step_by(8) {
            unsafe {
                let x = f32x8::load_unaligned(x.as_ptr().add(i));
                let y = f32x8::load_unaligned(y.as_ptr().add(i));
                xy8.multiply_add(x, y);
            }
        }
        let xy = xy16.reduce_sum() + xy8.reduce_sum() + dot(&x[aligned_len..], &y[aligned_len..]);
        1.0 - xy / x_norm / y_norm
    }

    fn cosine_batch<'a>(
        x: &'a [Self::Native],
        batch: &'a [Self::Native],
        dimension: usize,
    ) -> Box<dyn Iterator<Item = Self::Native> + 'a> {
        let x_norm = norm_l2(x);

        match dimension {
            8 => Box::new(
                batch
                    .chunks_exact(dimension)
                    .map(move |y| f32::cosine_once::<f32x8, 8>(x, x_norm, y)),
            ),
            16 => Box::new(
                batch
                    .chunks_exact(dimension)
                    .map(move |y| f32::cosine_once::<f32x16, 16>(x, x_norm, y)),
            ),
            _ => Box::new(
                batch
                    .chunks_exact(dimension)
                    .map(move |y| Self::cosine_fast(x, x_norm, y)),
            ),
        }
    }
}

impl Cosine for Float64Type {}

/// Fallback non-SIMD implementation
#[allow(dead_code)] // Does not fallback on aarch64.
#[inline]
fn cosine_scalar<T: AsPrimitive<f64> + FromPrimitive + FloatToArrayType>(
    x: &[T],
    x_norm: f32,
    y: &[T],
) -> f32
where
    <T as FloatToArrayType>::ArrowType: super::dot::Dot,
{
    let y_sq = dot(y, y);
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    1.0 - xy / (x_norm * y_sq.sqrt())
}

#[inline]
pub(crate) fn cosine_scalar_fast<T: FloatToArrayType>(
    x: &[T],
    x_norm: f32,
    y: &[T],
    y_norm: f32,
) -> f32
where
    T::ArrowType: Cosine,
{
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    // use f64 for overflow protection.
    1.0 - (xy / (x_norm * y_norm))
}

/// Cosine distance function between two vectors.
pub fn cosine_distance<T: FloatToArrayType>(from: &[T], to: &[T]) -> f32
where
    T::ArrowType: Cosine,
{
    T::ArrowType::cosine(from, to)
}

/// Cosine Distance
///
/// <https://en.wikipedia.org/wiki/Cosine_similarity>
///
/// Parameters
/// -----------
///
/// - *from*: the vector to compute distance from.
/// - *to*: the batch of vectors to compute distance to.
/// - *dimension*: the dimension of the vector.
///
/// Returns
/// -------
/// An iterator of pair-wise cosine distance between from vector to each vector in the batch.
///
pub fn cosine_distance_batch<'a, T: FloatToArrayType>(
    from: &'a [T],
    batch: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a>
where
    T::ArrowType: Cosine,
{
    T::ArrowType::cosine_batch(from, batch, dimension)
}

fn do_cosine_distance_arrow_batch<T: ArrowFloatType + Cosine>(
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
                "Unsupported data type {:?}",
                to.values().data_type()
            )))?;
    let dists = cosine_distance_batch(from.as_slice(), to_values.as_slice(), dimension);

    Ok(Arc::new(Float32Array::new(
        dists.collect(),
        to.nulls().cloned(),
    )))
}

/// Compute Cosine distance between a vector and a batch of vectors.
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
pub fn cosine_distance_arrow_batch(
    from: &dyn Array,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
    match *from.data_type() {
        DataType::Float16 => do_cosine_distance_arrow_batch::<Float16Type>(from.as_primitive(), to),
        DataType::Float32 => do_cosine_distance_arrow_batch::<Float32Type>(from.as_primitive(), to),
        DataType::Float64 => do_cosine_distance_arrow_batch::<Float64Type>(from.as_primitive(), to),
        _ => Err(Error::InvalidArgumentError(format!(
            "Unsupported data type {:?}",
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
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    fn cosine_dist_brute_force(x: &[f32], y: &[f32]) -> f32 {
        let xy = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f32>();
        let x_sq = x.iter().map(|&xi| xi * xi).sum::<f32>().sqrt();
        let y_sq = y.iter().map(|&yi| yi * yi).sum::<f32>().sqrt();
        1.0 - xy / x_sq / y_sq
    }

    #[test]
    fn test_cosine() {
        let x: Float32Array = (1..9).map(|v| v as f32).collect();
        let y: Float32Array = (100..108).map(|v| v as f32).collect();
        let d = cosine_distance_batch(x.values(), y.values(), 8).collect::<Vec<_>>();
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d[0], 1.0 - 0.900_957);

        let x = Float32Array::from_iter_values([3.0, 45.0, 7.0, 2.0, 5.0, 20.0, 13.0, 12.0]);
        let y = Float32Array::from_iter_values([2.0, 54.0, 13.0, 15.0, 22.0, 34.0, 50.0, 1.0]);
        let d = cosine_distance_batch(x.values(), y.values(), 8).collect::<Vec<_>>();
        // from sklearn.metrics.pairwise import cosine_similarity
        assert_relative_eq!(d[0], 1.0 - 0.873_580_63);
    }

    #[test]
    fn test_cosine_large() {
        let total = 1024;
        let x = (0..total).map(|v| v as f32).collect::<Vec<_>>();
        let y = (1024..1024 + total).map(|v| v as f32).collect::<Vec<_>>();
        let d = cosine_distance_batch(&x, &y, total).collect::<Vec<_>>();
        assert_relative_eq!(d[0], cosine_dist_brute_force(&x, &y));
    }

    #[test]
    fn test_cosine_not_aligned() {
        let x: Float32Array = vec![16_f32, 32_f32].into();
        let y: Float32Array = vec![1_f32, 2_f32, 4_f32, 8_f32].into();
        let d = cosine_distance_batch(x.values(), y.values(), 2).collect::<Vec<_>>();
        assert_relative_eq!(d[0], 0.0);
        assert_relative_eq!(d[0], 0.0);
    }

    /// Reference implementation of cosine distance, plus error propagation.
    ///
    /// Pass `rel_err` to provide the allowed relative error in the dot product
    /// results. This function will then compute the expected absolute error.
    fn cosine_ref(x: &[f64], y: &[f64], rel_err: f64) -> (f32, f32) {
        let xy = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f64>();
        let x_sq = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        let y_sq = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
        let expected = (1.0 - xy / x_sq / y_sq) as f32;

        let factor = 1.0 + rel_err;
        let low = (1.0 - (xy * factor) / (x_sq / factor) / (y_sq / factor)) as f32;
        let high = (1.0 - (xy / factor) / (x_sq * factor) / (y_sq * factor)) as f32;
        let low = (expected - low).abs();
        let high = (expected - high).abs();
        let error = low.max(high);

        (expected, error)
    }

    fn do_cosine_test<T: FloatToArrayType>(
        x: &[T],
        y: &[T],
    ) -> std::result::Result<(), TestCaseError>
    where
        T::ArrowType: Cosine,
    {
        let x_f64 = x.iter().map(|&v| v.as_()).collect::<Vec<_>>();
        let y_f64 = y.iter().map(|&v| v.as_()).collect::<Vec<_>>();

        let (expected, max_error) = cosine_ref(&x_f64, &y_f64, 1e-6);
        let result = T::ArrowType::cosine(x, y);

        prop_assert!(approx::relative_eq!(result, expected, epsilon = max_error));
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn test_cosine_f16((x, y) in arbitrary_vector_pair(arbitrary_f16, 4..4048)) {
            // Cosine requires non-zero vectors
            prop_assume!(norm_l2(&x) > 1e-6);
            prop_assume!(norm_l2(&y) > 1e-6);
            do_cosine_test(&x, &y)?;
        }

        #[test]
        fn test_cosine_bf16((x, y) in arbitrary_vector_pair(arbitrary_bf16, 4..4048)){
            prop_assume!(norm_l2(&x) > 1e-6);
            prop_assume!(norm_l2(&y) > 1e-6);
            do_cosine_test(&x, &y)?;
        }

        #[test]
        fn test_cosine_f32((x, y) in arbitrary_vector_pair(arbitrary_f32, 4..4048)){
            prop_assume!(norm_l2(&x) > 1e-10);
            prop_assume!(norm_l2(&y) > 1e-10);
            do_cosine_test(&x, &y)?;
        }

        #[test]
        fn test_cosine_f64((x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)){
            prop_assume!(norm_l2(&x) > 1e-20);
            prop_assume!(norm_l2(&y) > 1e-20);
            do_cosine_test(&x, &y)?;
        }
    }
}
