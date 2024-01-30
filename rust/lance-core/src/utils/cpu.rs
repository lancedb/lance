// Copyright 2024 Lance Developers.
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

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use core::arch::x86_64::__cpuid;

    use lazy_static::lazy_static;

    #[inline]
    fn check_flag(x: usize, position: u32) -> bool {
        x & (1 << position) != 0
    }

    lazy_static! {
        pub static ref AVX512_F16_SUPPORTED: bool = {
                    // this macro does many OS checks/etc. to determine if allowed to use AVX512
            if !is_x86_feature_detected!("avx512f") {
                return false;
            }

            // EAX=7, ECX=0: Extended Features (includes AVX512)
            // More info on calling CPUID can be found here (section 1.4)
            // https://www.intel.com/content/dam/develop/external/us/en/documents/architecture-instruction-set-extensions-programming-reference.pdf
            let ext_cpuid_result = unsafe { __cpuid(7) };
            let avx512_fp16 = check_flag(ext_cpuid_result.edx as usize, 23);

            avx512_fp16
        };
    }
}
