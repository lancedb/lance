// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! L2 (Euclidean) distance.
//!

use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
    Array, FixedSizeListArray, Float32Array,
};
use arrow_schema::DataType;
use half::{bf16, f16};
use lance_arrow::{ArrowFloatType, FloatArray};
#[cfg(feature = "fp16kernels")]
use lance_core::utils::cpu::SimdSupport;
use lance_core::utils::cpu::FP16_SIMD_SUPPORT;
use num_traits::{AsPrimitive, Float, Num};

use crate::simd::{
    f32::{f32x16, f32x8},
    SIMD,
};
use crate::{Error, Result};

/// Calculate the L2 distance between two vectors.
///
pub trait L2: Num {
    /// Calculate the L2 distance between two vectors.
    fn l2(x: &[Self], y: &[Self]) -> f32;

    fn l2_batch<'a>(
        x: &'a [Self],
        y: &'a [Self],
        dimension: usize,
    ) -> Box<dyn Iterator<Item = f32> + 'a> {
        Box::new(y.chunks_exact(dimension).map(|v| Self::l2(x, v)))
    }
}

#[inline]
pub fn l2<T: L2>(from: &[T], to: &[T]) -> f32 {
    T::l2(from, to)
}

/// Calculate L2 distance between two uint8 slices.
#[inline]
pub fn l2_distance_uint_scalar(key: &[u8], target: &[u8]) -> f32 {
    key.iter()
        .zip(target.iter())
        .map(|(&x, &y)| (x.abs_diff(y) as u32).pow(2))
        .sum::<u32>() as f32
}

/// Calculate the L2 distance between two vectors, using scalar operations.
///
/// It relies on LLVM for auto-vectorization and unrolling.
///
/// This is pub for test/benchmark only. use [l2] instead.
#[inline]
pub fn l2_scalar<
    T: AsPrimitive<Output>,
    Output: Float + Sum + AddAssign + 'static,
    const LANES: usize,
>(
    from: &[T],
    to: &[T],
) -> Output {
    let x_chunks = from.chunks_exact(LANES);
    let y_chunks = to.chunks_exact(LANES);

    let s = if !x_chunks.remainder().is_empty() {
        x_chunks
            .remainder()
            .iter()
            .zip(y_chunks.remainder())
            .map(|(&x, &y)| {
                let diff = x.as_() - y.as_();
                diff * diff
            })
            .sum::<Output>()
    } else {
        Output::zero()
    };

    let mut sums = [Output::zero(); LANES];
    for (x, y) in x_chunks.zip(y_chunks) {
        for i in 0..LANES {
            let diff = x[i].as_() - y[i].as_();
            sums[i] += diff * diff;
        }
    }

    s + sums.iter().copied().sum()
}

impl L2 for bf16 {
    #[inline]
    fn l2(x: &[Self], y: &[Self]) -> f32 {
        // TODO: add SIMD support
        l2_scalar::<Self, f32, 16>(x, y)
    }
}

#[cfg(feature = "fp16kernels")]
mod kernel {
    use super::*;

    // These are the `l2_f16` function in f16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn l2_f16_neon(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(all(kernel_suppport = "avx512", target_arch = "x86_64"))]
        pub fn l2_f16_avx512(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn l2_f16_avx2(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
    }
}

impl L2 for f16 {
    #[inline]
    fn l2(x: &[Self], y: &[Self]) -> f32 {
        match *FP16_SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                kernel::l2_f16_neon(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_suppport = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512 => unsafe {
                kernel::l2_f16_avx512(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 => unsafe {
                kernel::l2_f16_avx2(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            _ => l2_scalar::<Self, f32, 16>(x, y),
        }
    }
}

impl L2 for f32 {
    #[inline]
    fn l2(x: &[Self], y: &[Self]) -> f32 {
        l2_scalar::<Self, Self, 32>(x, y)
    }

    fn l2_batch<'a>(
        x: &'a [Self],
        y: &'a [Self],
        dimension: usize,
    ) -> Box<dyn Iterator<Item = f32> + 'a> {
        use self::f32::l2_once;
        // Dispatch based on the dimension.
        match dimension {
            8 => Box::new(
                y.chunks_exact(dimension)
                    .map(move |v| l2_once::<f32x8, 8>(x, v)),
            ),
            16 => Box::new(
                y.chunks_exact(dimension)
                    .map(move |v| l2_once::<f32x16, 16>(x, v)),
            ),
            _ => Box::new(y.chunks_exact(dimension).map(|v| Self::l2(x, v))),
        }
    }
}

impl L2 for f64 {
    #[inline]
    fn l2(x: &[Self], y: &[Self]) -> f32 {
        l2_scalar::<Self, Self, 8>(x, y) as f32
    }
}

/// Compute L2 distance between two vectors.
#[inline]
pub fn l2_distance(from: &[f32], to: &[f32]) -> f32 {
    l2(from, to)
}

// f32 kernels for L2
mod f32 {
    use super::*;

    #[inline]
    pub fn l2_once<S: SIMD<f32, N>, const N: usize>(x: &[f32], y: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), N);
        debug_assert_eq!(y.len(), N);
        let x = unsafe { S::load_unaligned(x.as_ptr()) };
        let y = unsafe { S::load_unaligned(y.as_ptr()) };
        let s = x - y;
        (s * s).reduce_sum()
    }
}

/// Compute L2 distance between a vector and a batch of vectors.
///
/// Parameters
///
/// - `from`: the vector to compute distance from.
/// - `to`: a list of vectors to compute distance to.
/// - `dimension`: the dimension of the vectors.
///
/// Returns
///
/// An iterator of pair-wise distance between `from` vector to each vector in the batch.
pub fn l2_distance_batch<'a, T: L2>(
    from: &'a [T],
    to: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a> {
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);

    Box::new(T::l2_batch(from, to, dimension))
}

fn do_l2_distance_arrow_batch<T: ArrowFloatType>(
    from: &T::ArrayType,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>>
where
    T::Native: L2,
{
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values =
        to.values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::ComputeError(format!(
                "Cannot downcast to the same type: {} != {}",
                T::FLOAT_TYPE,
                to.value_type()
            )))?;
    let dists = l2_distance_batch(from.as_slice(), to_values.as_slice(), dimension);

    Ok(Arc::new(Float32Array::new(
        dists.collect(),
        to.nulls().cloned(),
    )))
}

/// Compute L2 distance between a vector and a batch of vectors.
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
pub fn l2_distance_arrow_batch(
    from: &dyn Array,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
    match *from.data_type() {
        DataType::Float16 => do_l2_distance_arrow_batch::<Float16Type>(from.as_primitive(), to),
        DataType::Float32 => do_l2_distance_arrow_batch::<Float32Type>(from.as_primitive(), to),
        DataType::Float64 => do_l2_distance_arrow_batch::<Float64Type>(from.as_primitive(), to),
        _ => Err(Error::ComputeError(format!(
            "Unsupported data type: {}",
            from.data_type()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use num_traits::ToPrimitive;
    use proptest::prelude::*;

    use crate::test_utils::{
        arbitrary_bf16, arbitrary_f16, arbitrary_f32, arbitrary_f64, arbitrary_vector_pair,
    };

    #[test]
    fn test_euclidean_distance() {
        let mat = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some((0..8).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((1..9).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((2..10).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((3..11).map(|v| Some(v as f32)).collect::<Vec<_>>()),
            ],
            8,
        );
        let point = Float32Array::from((2..10).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let distances = l2_distance_batch(
            point.values(),
            mat.values().as_primitive::<Float32Type>().values(),
            8,
        )
        .collect::<Vec<_>>();

        assert_eq!(distances, vec![32.0, 8.0, 0.0, 8.0]);
    }

    #[test]
    fn test_not_aligned() {
        let mat = (0..6)
            .chain(0..8)
            .chain(1..9)
            .chain(2..10)
            .chain(3..11)
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let point = Float32Array::from((0..10).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let distances = l2_distance_batch(&point.values()[2..], &mat[6..], 8).collect::<Vec<_>>();

        assert_eq!(distances, vec![32.0, 8.0, 0.0, 8.0]);
    }

    #[test]
    fn test_odd_length_vector() {
        let mat = Float32Array::from_iter((0..5).map(|v| Some(v as f32)));
        let point = Float32Array::from((2..7).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let distances = l2_distance_batch(point.values(), mat.values(), 5).collect::<Vec<_>>();

        assert_eq!(distances, vec![20.0]);
    }

    #[test]
    fn test_l2_distance_cases() {
        let values: Float32Array = vec![
            0.25335717, 0.24663818, 0.26330215, 0.14988247, 0.06042378, 0.21077952, 0.26687378,
            0.22145681, 0.18319066, 0.18688454, 0.05216244, 0.11470364, 0.10554603, 0.19964123,
            0.06387895, 0.18992095, 0.00123718, 0.13500804, 0.09516747, 0.19508345, 0.2582458,
            0.1211653, 0.21121833, 0.24809816, 0.04078768, 0.19586588, 0.16496408, 0.14766085,
            0.04898421, 0.14728612, 0.21263947, 0.16763233,
        ]
        .into();

        let q: Float32Array = vec![
            0.18549609,
            0.29954708,
            0.28318876,
            0.05424477,
            0.093134984,
            0.21580857,
            0.2951282,
            0.19866848,
            0.13868214,
            0.19819534,
            0.23271298,
            0.047727287,
            0.14394054,
            0.023316395,
            0.18589257,
            0.037315924,
            0.07037327,
            0.32609823,
            0.07344752,
            0.020155912,
            0.18485495,
            0.32763934,
            0.14296658,
            0.04498596,
            0.06254237,
            0.24348071,
            0.16009757,
            0.053892266,
            0.05918874,
            0.040363103,
            0.19913352,
            0.14545348,
        ]
        .into();

        let d = l2_distance_batch(q.values(), values.values(), 32).collect::<Vec<_>>();
        assert_relative_eq!(0.319_357_84, d[0]);
    }

    /// Reference implementation of L2 distance.
    ///
    /// Note that we skip the final square root step for performance reasons.
    fn l2_distance_reference(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<f64>()
    }

    fn do_l2_test<T: L2 + ToPrimitive>(x: &[T], y: &[T]) -> std::result::Result<(), TestCaseError> {
        let x_f64 = x.iter().map(|v| v.to_f64().unwrap()).collect::<Vec<f64>>();
        let y_f64 = y.iter().map(|v| v.to_f64().unwrap()).collect::<Vec<f64>>();

        let result = l2(x, y);
        let reference = l2_distance_reference(&x_f64, &y_f64) as f32;

        prop_assert!(approx::relative_eq!(result, reference, max_relative = 1e-6));
        Ok(())
    }

    #[test]
    fn test_l2_distance_f16_max() {
        let x = vec![f16::MAX; 4048];
        let y = vec![-f16::MAX; 4048];
        do_l2_test(&x, &y).unwrap();
    }

    // Test L2 distance over different types.
    // * L2 is valid over the entire range of f16.
    // * L2 is valid over f32 and bf16 in the range of +-1e12.
    // * L2 for f64 should match the reference implementation.
    proptest::proptest! {
        #[test]
        fn test_l2_distance_f16((x, y) in arbitrary_vector_pair(arbitrary_f16, 4..4048)) {
            do_l2_test(&x, &y)?;
        }

        #[test]
        fn test_l2_distance_bf16((x, y) in arbitrary_vector_pair(arbitrary_bf16, 4..4048)){
            do_l2_test(&x, &y)?;
        }

        #[test]
        fn test_l2_distance_f32((x, y) in arbitrary_vector_pair(arbitrary_f32, 4..4048)){
            do_l2_test(&x, &y)?;
        }

        #[test]
        fn test_l2_distance_f64((x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)){
            do_l2_test(&x, &y)?;
        }
    }

    #[test]
    fn test_uint8_l2_edge_cases() {
        let q = vec![0_u8; 2048];
        let v = vec![0_u8; 2048];
        assert_eq!(l2_distance_uint_scalar(&q, &v), 0.0);

        let q = vec![0_u8; 2048];
        let v = vec![255_u8; 2048];
        assert_eq!(
            l2_distance_uint_scalar(&q, &v),
            (255_u32.pow(2) * 2048) as f32
        );
        assert_eq!(
            l2_distance_uint_scalar(&v, &q),
            (255_u32.pow(2) * 2048) as f32
        );
    }
}
