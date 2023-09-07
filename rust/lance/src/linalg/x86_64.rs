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

//! Common X86_64 SIMD routines

pub(super) mod avx {

    #[inline]
    pub unsafe fn add_f32_register(x: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;

        let mut sums = x;
        let mut shift = _mm256_permute2f128_ps(sums, sums, 1);
        // [x0+x4, x1+x5, ..]
        sums = _mm256_add_ps(sums, shift);
        shift = _mm256_permute_ps(sums, 14);
        sums = _mm256_add_ps(sums, shift);
        sums = _mm256_hadd_ps(sums, sums);
        let mut results: [f32; 8] = [0f32; 8];
        _mm256_storeu_ps(results.as_mut_ptr(), sums);
        results[0]
    }
}
