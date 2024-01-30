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

use std::iter::Sum;

use half::{bf16, f16};
use num_traits::{AsPrimitive, Float};

#[cfg(all(target_os = "linux", feature = "avx512fp16", target_arch = "x86_64"))]
use lance_core::utils::cpu::x86::is_avx512_fp16_supported;

use crate::simd::{
    f32::{f32x16, f32x8},
    SIMD,
};

/// L2 normalization
pub trait Normalize<T: Float> {
    /// L2 Normalization over a Vector.
    fn norm_l2(&self) -> f32;
}

// `avx512fp16` is not supported in rustc yet. Once it is supported, we can
// move it to target_feture.
#[cfg(any(
    all(target_os = "macos", target_feature = "neon"),
    feature = "avx512fp16"
))]
mod kernel {
    use super::*;

    extern "C" {
        pub fn norm_l2_f16(ptr: *const f16, len: u32) -> f32;
    }
}

impl Normalize<f16> for &[f16] {
    // #[inline]
    fn norm_l2(&self) -> f32 {
        #[cfg(all(target_os = "macos", target_feature = "neon"))]
        unsafe {
            kernel::norm_l2_f16(self.as_ptr(), self.len() as u32)
        }

        #[cfg(all(target_os = "linux", feature = "avx512fp16", target_arch = "x86_64"))]
        if is_avx512_fp16_supported() {
            unsafe { kernel::norm_l2_f16(self.as_ptr(), self.len() as u32) }
        } else {
            norm_l2_f16_impl(self)
        }

        #[cfg(not(any(
            all(target_os = "macos", target_feature = "neon"),
            feature = "avx512fp16"
        )))]
        norm_l2_f16_impl(self)
    }
}

#[inline]
#[cfg(not(any(
    all(target_os = "macos", target_feature = "neon"),
    feature = "avx512fp16"
)))]
fn norm_l2_f16_impl(arr: &[f16]) -> f32 {
    // Please run `cargo bench --bench norm_l2" on Apple Silicon when
    // change the following code.
    const LANES: usize = 16;
    let chunks = arr.chunks_exact(LANES);
    let sum = if chunks.remainder().is_empty() {
        0.0
    } else {
        chunks
            .remainder()
            .iter()
            .map(|v| v.to_f32().powi(2))
            .sum::<f32>()
    };

    let mut sums: [f32; LANES] = [0_f32; LANES];
    for chk in chunks {
        // Convert to f32
        let mut f32_vals: [f32; LANES] = [0_f32; LANES];
        for i in 0..LANES {
            f32_vals[i] = chk[i].to_f32();
        }
        // Vectorized multiply
        for i in 0..LANES {
            sums[i] += f32_vals[i].powi(2);
        }
    }
    (sums.iter().copied().sum::<f32>() + sum).sqrt()
}

impl Normalize<bf16> for &[bf16] {
    #[inline]
    fn norm_l2(&self) -> f32 {
        norm_l2_impl::<bf16, 32>(self)
    }
}

impl Normalize<f32> for &[f32] {
    #[inline]
    fn norm_l2(&self) -> f32 {
        let dim = self.len();
        if dim % 16 == 0 {
            let mut sum = f32x16::zeros();
            for i in (0..dim).step_by(16) {
                let x = unsafe { f32x16::load_unaligned(self.as_ptr().add(i)) };
                sum += x * x;
            }
            sum.reduce_sum().sqrt()
        } else if dim % 8 == 0 {
            let mut sum = f32x8::zeros();
            for i in (0..dim).step_by(8) {
                let x = unsafe { f32x8::load_unaligned(self.as_ptr().add(i)) };
                sum += x * x;
            }
            sum.reduce_sum().sqrt()
        } else {
            // Fallback to scalar
            norm_l2(self)
        }
    }
}

impl Normalize<f64> for &[f64] {
    #[inline]
    fn norm_l2(&self) -> f32 {
        norm_l2(self)
    }
}

#[inline]
fn norm_l2_impl<T: Float + Sum + AsPrimitive<f32>, const LANES: usize>(vector: &[T]) -> f32 {
    let chunks = vector.chunks_exact(LANES);
    let sum = if chunks.remainder().is_empty() {
        T::zero()
    } else {
        chunks.remainder().iter().map(|&v| v.powi(2)).sum::<T>()
    };
    let mut sums = [T::zero(); LANES];
    for chunk in chunks {
        for i in 0..LANES {
            sums[i] = sums[i].add(chunk[i].powi(2));
        }
    }
    (sum + sums.iter().copied().sum::<T>()).sqrt().as_()
}

/// Normalize a vector.
///
/// The parameters must be cache line aligned. For example, from
/// Arrow Arrays, i.e., Float32Array
#[inline]
pub fn norm_l2<T: Float + Sum + AsPrimitive<f32>>(vector: &[T]) -> f32 {
    const LANES: usize = 16;
    norm_l2_impl::<T, LANES>(vector)
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use approx::assert_relative_eq;
    use num_traits::FromPrimitive;

    use super::*;

    fn do_norm_l2_test<T: Float + FromPrimitive + Sum + Debug>()
    where
        for<'a> &'a [T]: Normalize<T>,
    {
        let data = (1..=37)
            .map(|v| T::from_i32(v).unwrap())
            .collect::<Vec<T>>();

        let result = data.as_slice().norm_l2();
        assert_relative_eq!(
            result as f64,
            (1..=37)
                .map(|v| f64::from_i32(v * v).unwrap())
                .sum::<f64>()
                .sqrt(),
            max_relative = 1.0,
        );

        let not_aligned = (&data[2..]).norm_l2();
        assert_relative_eq!(
            not_aligned as f64,
            (3..=37)
                .map(|v| f64::from_i32(v * v).unwrap())
                .sum::<f64>()
                .sqrt(),
            max_relative = 1.0,
        );
    }

    #[test]
    fn test_norm_l2() {
        do_norm_l2_test::<bf16>();
        do_norm_l2_test::<f16>();
        do_norm_l2_test::<f32>();
        do_norm_l2_test::<f64>();
    }
}
