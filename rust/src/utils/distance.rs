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

//! Compute distance
//!

pub mod compute;

#[inline]
pub(crate) fn simd_alignment() -> i32 {
    #[cfg(any(target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma") {
            return 8;
        }
    }

    #[cfg(any(target_arch = "aarch64"))]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") {
            return 4;
        }
    }

    1
}

/// Is the pointer aligned to the given alignment?
fn is_simd_aligned(ptr: *const f32, alignment: usize) -> bool {
    ptr as usize % alignment == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Float32Array;

    #[test]
    fn test_is_simd_aligned() {
        let arr = Float32Array::from_iter([1.9, 2.3, 3.4, 4.5]);
        assert!(is_simd_aligned(arr.values().as_ptr(), 32));

        unsafe {
            assert!(!is_simd_aligned(arr.values().as_ptr().add(3), 32));
        }
    }
}
