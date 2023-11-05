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
use num_traits::{real::Real, Float};

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

impl Normalize<f16> for &[f16] {
    type Output = f16;

    #[inline]
    fn norm_l2(&self) -> Self::Output {
        norm_l2(self)
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
pub fn norm_l2<T: Real + Sum>(vector: &[T]) -> T {
    const LANES: usize = 16;
    let chunks = vector.chunks_exact(LANES);
    let sum = if chunks.remainder().is_empty() {
        T::zero()
    } else {
        chunks.remainder().iter().map(|&v| v * v).sum::<T>()
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
    use num_traits::FromPrimitive;

    use super::*;

    macro_rules! do_norm_l2_test {
        ($t: ty) => {
            let data = (1..=8)
                .map(|v| <$t>::from_i32(v).unwrap())
                .collect::<Vec<$t>>();

            let result = data.as_slice().norm_l2();
            assert_eq!(
                result,
                Real::sqrt((1..=8).map(|v| <$t>::from_i32(v * v).unwrap()).sum::<$t>())
            );

            let not_aligned = (&data[2..]).norm_l2();
            assert_eq!(
                not_aligned,
                Real::sqrt((3..=8).map(|v| <$t>::from_i32(v * v).unwrap()).sum::<$t>())
            );
        };
    }

    #[test]
    fn test_norm_l2() {
        do_norm_l2_test!(bf16);
        do_norm_l2_test!(f16);
        do_norm_l2_test!(f32);
        do_norm_l2_test!(f64);
    }
}
