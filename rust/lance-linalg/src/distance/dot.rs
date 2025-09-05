// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dot product.

use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::Arc;

use crate::Error;
use arrow_array::types::{Float16Type, Float64Type, Int8Type};
use arrow_array::{cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use half::{bf16, f16};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use lance_core::assume_eq;
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
            _ => dot_scalar::<Self, f32, 32>(x, y),
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
    assume_eq!(from.len(), dimension);
    assume_eq!(to.len() % dimension, 0);
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
        DataType::Int8 => do_dot_distance_arrow_batch::<Float32Type>(
            &from
                .as_primitive::<Int8Type>()
                .into_iter()
                .map(|x| x.unwrap() as f32)
                .collect(),
            &to.convert_to_floating_point()?,
        ),
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

    #[test]
    fn test_dot_extreme_values() {
        // Regression test for extreme values that caused flaky behavior
        // These are the exact values from the CI failure  
        let x: Vec<f32> = vec![
            -4.504179e-35,
            -6.940286e-35,
            -22993777000.0,
            -0.0,
            0.0,
            8.411721e-37,
            -1.8470535e-34,
            1.08e-43,
            1.0656711e-19,
            0.0,
            1.967528e-24,
            1.21e-42,
            -1.8179308e-31,
            -4.989e-42,
            9.4e-44,
            -1.685e-41,
            1.078483e-24,
            -7.8667613e-22,
            7.324632e-39,
            -2.5249052e-5,
            -1847662200.0,
            0.0,
            -0.0,
            -2.9142303e-27,
            1.0101552e-10,
            733.6715,
            -2934.6672,
            0.0,
            0.0,
            0.0003117052,
            0.0,
            1.4814563e-23,
            0.0,
            9.991778e-21,
            4.053663e-5,
            -0.0,
            0.0,
            -1.3988695e-25,
            1.1127306e-38,
            0.0,
            -187175780.0,
            0.0,
            -8.473848e-15,
            -0.0,
            0.0,
            1.911e-42,
            9.5937565e-23,
            1.0065404e-37,
            -1.0102623e-29,
            -4.322055e-33,
            7.083123e-36,
            -0.006094696,
            -0.0,
            1.9478673e-30,
            8e-45,
            -0.0,
            6.0685276e-38,
            -0.00047238052,
            -3.7667966e-26,
            -1.0561133e-37,
            1.8194475e-30,
            0.0,
            4.236324e-22,
            -1.5928516,
            -4.671088e-7,
            -861781200.0,
            2.0688347e-10,
            -0.0,
            -4.4131188e-20,
            0.0,
            0.0,
            -1.7960927e-33,
            2.5471578e-36,
            3.1625095e-19,
            0.0,
            -6.275448e-5,
            0.0,
            3.2553245e-38,
            -1.155e-42,
            -3.827356e-12,
            2.6315785e-38,
            -0.0,
            -0.0,
            0.0,
            2.5075549e-25,
            2.7777784e-28,
            0.0,
            -0.0,
            -611251400000.0,
            4.12209e-40,
            0.0,
            3.8432404e-6,
            0.0,
            21.64729,
            1.8713403e-36,
            -4.4796344e-14,
            1.25869e-15,
            -6.577759,
            -0.0,
            0.0,
            0.0,
            0.046763014,
            5.981667e-19,
            -0.0,
            202893570.0,
            0.0,
            -410665700000.0,
            -8.882696e-39,
            -3.438232e-7,
            0.0,
            0.0,
            0.0,
            -6.11353e-28,
            0.08997257,
            -6.6968943e-31,
            4715518.5,
            6.2406404e-28,
            5.5734883e-9,
            1.1846173e-35,
            -0.0,
            4955.6343,
            0.0,
            -0.0,
            -9.548364e-13,
            -205.2491,
            0.0,
            1.4660074e-23,
            0.0,
            1.4591507e-13,
            -3.038155e-39,
            -0.0,
            -8e-45,
            0.0,
            0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.00015257283,
            2.7170168e-36,
            0.0,
            -0.0,
            0.0,
            -29715.53,
            6.519663e-10,
            -3433551.3,
            0.0,
            -176900.69,
            0.0,
            -3.038815e-21,
            -6.08e-43,
            -1.9185694e-19,
            -3.5712726e-28,
            -2.8923204e-18,
            -2.2801848e-20,
            -4.146301e-27,
            3.3953828e-15,
            -2.8869811e-18,
            -0.0,
            -26.57341,
            -19.40107,
            -0.0,
            -5.2780595e-17,
            -0.0,
            -6.776829e-19,
            0.0,
            0.0057117464,
            -2.916685e-11,
            -0.0,
            -5.2588316e-12,
            0.0,
            -6.892276e-6,
            -3.675926e-8,
            -0.0,
            -6036890600.0,
            -3.4806274e-19,
            41700812.0,
            -5.267e-42,
            1.218953e-37,
            -8470197.0,
            -1.505619e-29,
            -6.727934e-16,
            -2.8231465e-19,
            -3.2705324e-7,
            1.5141415e-13,
            2.9312415e-7,
            0.0,
            -0.0,
            -0.0,
            -584.3237,
            -0.0,
            -0.0,
            0.0,
            7.185695e-11,
            3.6452552e-13,
            859.5377,
            0.0,
            -2.234823e-35,
            0.0,
            9727661000.0,
            131.5387,
            5.9675294e-26,
            -13513000.0,
            5.584677e-32,
            448277250.0,
            -2.1185427e-31,
            0.24143945,
            -4.7436727e-24,
            0.0,
            1.6426033e-30,
            -0.0,
            1.12799e-40,
            -0.0,
            0.0,
            0.0,
            -0.0,
            -43811.63,
            0.0,
            1.7378182e-22,
            -0.0,
            0.0,
            0.0009182903,
            4.5697652e-7,
            0.0,
            0.0,
            5.340737e-39,
            -3.656389e-22,
            0.0,
            0.0,
            -4.272013e-12,
            6.772428e-33,
            1.1e-44,
            3.8817e-41,
            -0.0,
            -0.0,
            311267600.0,
            6.772635e-34,
            -3.535845e-8,
            -1.289e-42,
            -0.0,
            -0.0,
            -1.1016808e-35,
            -9.7121e-41,
            -0.0,
            -0.0,
            0.0,
            -0.0,
            -1.0439663e-30,
            5.5664275e-26,
            1383.5166,
            -0.0,
            2.770896e-25,
            9.334817e-35,
            -8.5262784e-19,
            -0.0,
            5.772252e-5,
            1.1945679e-17,
            -9903.426,
            -780475.06,
            -0.0,
            4.630104e-27,
            -213.1363,
            -3.232695e-25,
            0.0,
            -3.2e-44,
            1.5508155e-14,
            -5.241775e-6,
            -5.404091e-31,
            -0.0,
            -1.842159e-12,
            1.0556299e-26,
            400709500.0,
            2966947.0,
            0.0004178385,
            0.0,
            1.4201803e-35,
            -9.35e-43,
            -2.121297e-12,
            1.1952359e-17,
            -0.0,
            1.637634e-34,
            9.625398e-11,
            -5.357233e-16,
            220303.17,
            -1872.4443,
            -1.3988779e-28,
            -2.9826145,
            -0.00017423676,
            0.0,
            -1.01342e-40,
            -6.7617817e-37,
            -0.0,
            -10957667.0,
            -9.24076e-33,
            -23846.906,
            -0.0,
            -0.0,
            -0.0,
            1e-45,
            0.0,
            1.0878464e-22,
        ];
        
        let y: Vec<f32> = vec![
            2.9350092e-12,
            3.9506538e-29,
            -4.707454,
            -1.276565e-30,
            1.1977486e-15,
            -4579.363,
            -0.0,
            -1718552.8,
            -0.0,
            -2.933985e-23,
            -2.8652335e-16,
            1.6358324e-33,
            1.1687766e-21,
            0.0,
            -0.0,
            1.5336201,
            0.0,
            1.6220985e-26,
            0.0,
            -0.0,
            0.0,
            1.1010311e-31,
            0.0,
            3.3194612e-32,
            -1.4863635e-28,
            -0.0,
            -1.6104909e-32,
            -0.0,
            5.77522e-17,
            0.0,
            8.625872e-36,
            -9.473266e-6,
            -199669270000.0,
            -258714090000.0,
            -0.0,
            -1.8695279e-29,
            -4.3005834e-8,
            1.1575e-32,
            5.1090964e-32,
            0.0,
            2.840808e-39,
            0.0,
            -7.887321e-30,
            1.6968768e-22,
            0.0,
            -0.39720598,
            1.2650749e-15,
            1e-45,
            45050536000.0,
            0.0,
            -9.569336,
            -1.5390805e-21,
            5.8923085e-28,
            0.0,
            0.0,
            0.0,
            -0.0,
            3.413561e-22,
            -2.2717533e-19,
            -0.0,
            -0.0,
            -1.8273192e-35,
            -4e-45,
            -170245.14,
            0.0,
            -1.9635254e-7,
            0.0,
            -0.0,
            -3.536558e-31,
            -0.0,
            6.400437e-10,
            -1.9923125e-17,
            -48526180.0,
            -0.0,
            -0.0,
            -136352.94,
            -0.0,
            -3.7115024e-5,
            0.041506715,
            -0.0,
            -3e-45,
            -0.0,
            2.4249575e-36,
            -0.0,
            -115056.22,
            0.0,
            7.2116916e-13,
            -0.0,
            1178234200.0,
            0.0,
            0.0,
            3.400528e-7,
            -5.3513065e-9,
            202302850000.0,
            -2.7526997e-28,
            -5.150438e-33,
            -4.4402783e-26,
            -1.1711071e-33,
            0.0,
            346274530.0,
            -7815157.5,
            160.03455,
            -1544.6039,
            -36.794537,
            18288096.0,
            0.0,
            -1.707e-42,
            13.415292,
            0.0,
            -0.0,
            9.096172e-18,
            -0.0,
            -2.899874e-21,
            -0.0,
            -6.1670073e-19,
            4.930063e-27,
            -0.0,
            2.0284858e-6,
            -0.0,
            -14596139000.0,
            9.19235e-40,
            0.0,
            0.0,
            7.896789e-16,
            -0.054308772,
            2.0664963e-6,
            0.0,
            -0.0027661538,
            -1.5077192e-10,
            -0.0,
            4000395300.0,
            -2.606e-42,
            0.0,
            -0.0,
            -0.0,
            3.522782e-38,
            -4.75212e-27,
            200195490000.0,
            4.43462e-39,
            -1.816453e-36,
            1.8523193e-14,
            1.6962617e-33,
            -0.0,
            4.61736e-40,
            1.4779517e-23,
            5.968935e-12,
            3.38e-42,
            -4.474905e-12,
            -2.3748173e-8,
            0.0,
            -0.0,
            -8.122328e-39,
            1.1990494e-16,
            -0.0,
            -0.0,
            -9.217337e-7,
            -0.0,
            0.0,
            -1.0822406e-27,
            0.0,
            2.8072703e-38,
            -87329800.0,
            0.0,
            -0.0,
            -0.0,
            -1.0388364e-10,
            -7.2084496e-14,
            -0.018565925,
            0.0,
            0.0,
            -0.0,
            -0.0,
            2.4731876e-35,
            -1.6793824e-22,
            -0.0,
            -0.000684925,
            -0.002876458,
            -0.0,
            -4.7488643e-20,
            -0.0,
            -1e-45,
            2.8e-44,
            -0.0,
            -198.22353,
            -3.3081708e-19,
            1.4871667e-12,
            2.3208244e-27,
            8.4226604e-29,
            183528510.0,
            -327323350000.0,
            -5.820993e-15,
            -4.524537e-19,
            6e-45,
            -0.0,
            -8.1302716e-13,
            10807.731,
            -0.0,
            -6.3556503e-16,
            -1.9247638e-13,
            2.1922937e-14,
            881.73175,
            -1.4855824e-28,
            -3.1189404e-7,
            0.0,
            7.0598963e-7,
            -5.95398e-40,
            -1.3576041e-25,
            -0.0,
            4.503671e-35,
            -0.0042624963,
            0.0,
            -0.0,
            -2.4573963e-38,
            -0.0,
            0.0,
            -1.5972273e-24,
            9.4e-43,
            -258.44247,
            -0.0,
            6.090188e-28,
            -1.8696995e-27,
            -5.0122348e-26,
            0.0050251097,
            -178.4213,
            1.671188e-34,
            0.75764704,
            -0.00890574,
            -0.9703398,
            -7.4286095e-7,
            -7.1398596e-37,
            2.3778351e-14,
            -8.955944e-11,
            -0.0,
            9.884807e-18,
            0.0,
            -1.5859358e-24,
            -0.0,
            0.0,
            1864578400.0,
            -0.0,
            -1.4110525e-16,
            0.0,
            -18.92523,
            -0.0,
            -9.679867e-11,
            1.13e-42,
            -0.0,
            5.76504e-40,
            0.0,
            7e-44,
            2.7617115e-20,
            -220194.55,
            2.2883553e-6,
            -1.8578e-41,
            517788800000.0,
            -0.0,
            0.09157088,
            -0.00011154764,
            -8.5617344e-33,
            0.0024144235,
            0.0,
            -9.887074e-36,
            -7.5216764e-23,
            -4.018076e-34,
            -3.5520386e-15,
            3.248783e-22,
            26830287000.0,
            -7.2747e-41,
            -0.00045551115,
            -4.9092097e-33,
            37253.445,
            0.0,
            -0.0,
            -0.0,
            2.9717355,
            -3.4161188e-15,
            1.0750436e-8,
            -0.0,
            3.3506562e-29,
            -1835837.3,
            -0.0,
            311427780.0,
            -5.040339e-33,
            -0.0,
            0.0,
            5.215277e-30,
            -132484415000.0,
            0.0,
            -5.2653254e-32,
            8.432514e-35,
            -1e-45,
            71402770000.0,
            0.0,
            0.0,
            5.2698903e-18,
            3.561896e-22,
            0.0,
            -2.6859079e-24,
            -1.5242048e-5,
            7.319399e-17,
        ];

        // Use the same test function that the proptest uses to ensure consistency
        // This test verifies our improved error tolerance handles extreme values  
        do_dot_test(&x, &y).unwrap();
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

        // Calculate relative error based on epsilon and dot product magnitude
        let base_error = 2.0 * T::epsilon().as_() * dot;

        // Add extra tolerance for accumulated rounding errors
        // The factor scales with sqrt of vector length for error accumulation
        // This is especially important for SIMD implementations which may have
        // different ordering of operations compared to the scalar reference
        let length_factor = (x.len() as f64).sqrt();
        let accumulated_error = T::epsilon().as_() * length_factor;

        // Use a minimum absolute error threshold for extreme small values
        // This handles cases where the dot product is near zero
        let min_absolute_error = 1e-6_f64;

        // Combine all error sources
        let total_error = base_error + accumulated_error;
        total_error.max(min_absolute_error) as f32
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
