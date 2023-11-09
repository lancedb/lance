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
use num_traits::Float;

use crate::simd::{
    f32::{f32x16, f32x8},
    SIMD,
};

/// L2 normalization
pub trait Normalize<T: Float> {
    type Output;

    /// L2 Normalization over a Vector.
    fn norm_l2(&self) -> Self::Output;
}

#[cfg(any(target_feature = "neon", target_feature = "avx512fp16"))]
mod kernel {
    use super::*;

    extern "C" {
        pub fn norm_l2_f16(ptr: *const f16, len: u64) -> f16;
    }
}

impl Normalize<f16> for &[f16] {
    type Output = f16;

    // #[inline]
    fn norm_l2(&self) -> Self::Output {
        #[cfg(any(target_feature = "neon", target_feature = "avx512fp16"))]
        unsafe {
            kernel::norm_l2_f16(self.as_ptr(), self.len() as u64)
        }
        #[cfg(not(any(target_feature = "neon", target_feature = "avx512fp16")))]
        {
            // Please run `cargo bench --bench norm_l2" on Apple Silicon when
            // change the following code.
            const LANES: usize = 4;
            let chunks = self.chunks_exact(LANES);
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
            f16::from_f32((sums.iter().copied().sum::<f32>() + sum).sqrt())
        }
    }
}

impl Normalize<bf16> for &[bf16] {
    type Output = bf16;

    #[inline]
    fn norm_l2(&self) -> Self::Output {
        norm_l2(self)
    }
}

impl Normalize<f32> for &[f32] {
    type Output = f32;

    #[inline]
    fn norm_l2(&self) -> Self::Output {
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
    type Output = f64;

    #[inline]
    fn norm_l2(&self) -> Self::Output {
        norm_l2(self)
    }
}

/// Normalize a vector.
///
/// The parameters must be cache line aligned. For example, from
/// Arrow Arrays, i.e., Float32Array
#[inline]
pub fn norm_l2<T: Float + Sum>(vector: &[T]) -> T {
    const LANES: usize = 16;
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
    (sum + sums.iter().copied().sum::<T>()).sqrt()
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use approx::assert_relative_eq;
    use num_traits::{AsPrimitive, FromPrimitive};

    use super::*;

    fn do_norm_l2_test<T: Float + FromPrimitive + Sum + Debug>()
    where
        for<'a> &'a [T]: Normalize<T>,
        for<'a> <&'a [T] as Normalize<T>>::Output: PartialEq<T> + Debug + AsPrimitive<f64>,
    {
        let data = (1..=37)
            .map(|v| T::from_i32(v).unwrap())
            .collect::<Vec<T>>();

        let result = data.as_slice().norm_l2();
        assert_relative_eq!(
            result.as_(),
            (1..=37)
                .map(|v| f64::from_i32(v * v).unwrap())
                .sum::<f64>()
                .sqrt(),
            max_relative = 1.0,
        );

        let not_aligned = (&data[2..]).norm_l2();
        assert_relative_eq!(
            not_aligned.as_(),
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
