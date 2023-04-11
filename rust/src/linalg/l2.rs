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

use arrow_array::Float32Array;

pub trait L2 {
    fn l2(&self, other: &Self) -> f32;
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_neon(from: &[f32], to: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = from.len();
    let buf = [0.0_f32; 4];
    let mut sum = vld1q_f32(buf.as_ptr());
    for i in (0..len).step_by(4) {
        let left = vld1q_f32(from.as_ptr().add(i));
        let right = vld1q_f32(to.as_ptr().add(i));
        let sub = vsubq_f32(left, right);
        sum = vfmaq_f32(sum, sub, sub);
    }
    vaddvq_f32(sum)
}

/// Fall back to scalar implementation.
fn l2_scalar(from: &[f32], to: &[f32]) -> f32 {
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

impl L2 for [f32] {
    fn l2(&self, other: &Self) -> f32 {
        #[cfg(any(target_arch = "aarch64"))]
        {
            use std::arch::is_aarch64_feature_detected;
            if is_aarch64_feature_detected!("neon") {
                unsafe {
                    return l2_neon(self, other);
                }
            }
        }

        l2_scalar(self, other)
    }
}

impl L2 for Float32Array {
    fn l2(&self, other: &Self) -> f32 {
        self.values().l2(other.values())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow::array::{as_primitive_array, FixedSizeListArray};
    use arrow_array::types::Float32Type;
}
