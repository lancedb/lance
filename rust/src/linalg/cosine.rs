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

use num_traits::real::Real;

use super::dot::Dot;

/// Fallback Cosine Distance function.
fn cosine_scalar<T: Real>(from: &[T], to: &[T]) -> T {
    let x_sq = T::zero();
    let mut y_sq = T::zero();
    let mut xy = T::zero();
    from.iter().zip(to.iter()).for_each(|(x, y)| {
        xy = xy.add(x.mul(*y));
        x_sq = x.powi(2);
        y_sq += y.powi(2);
    });
    1.0 - xy / (x_sq.sqrt() * y_sq.sqrt())
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    mod avx {
        use super::*;

        #[inline]
        unsafe fn cosine_f32(x_vector: &[f32], y_vector: &[f32], x_norm: f32) -> f32 {
            use super::compute::add_fma;

            let len = x_vector.len();
            let mut xy = _mm256_setzero_ps();
            let mut y_sq = _mm256_setzero_ps();
            for i in (0..len).step_by(8) {
                let x = _mm256_load_ps(x_vector.as_ptr().add(i));
                let y = _mm256_load_ps(y_vector.as_ptr().add(i));
                xy = _mm256_fmadd_ps(x, y, xy);
                y_sq = _mm256_fmadd_ps(y, y, y_sq);
            }
            1.0 - add_fma(xy) / (x_norm * add_fma(y_sq).sqrt())
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    mod neon {
        use super::*;

        #[inline]
        unsafe fn cosine_f32(x: &[f32], y: &[f32], x_norm: f32) -> f32 {
            let len = x.len();
            let buf = [0.0_f32; 4];
            let mut xy = vld1q_f32(buf.as_ptr());
            let mut y_sq = xy;
            for i in (0..len).step_by(4) {
                let left = vld1q_f32(x.as_ptr().add(i));
                let right = vld1q_f32(y.as_ptr().add(i));
                xy = vfmaq_f32(xy, left, right);
                y_sq = vfmaq_f32(y_sq, right, right);
            }
            1.0 - vaddvq_f32(xy) / (x_norm * vaddvq_f32(y_sq).sqrt())
        }
    }
}
