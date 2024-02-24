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

use lazy_static::lazy_static;

/// A level of SIMD support for some feature
pub enum SimdSupport {
    None,
    Neon,
    Sse,
    Avx2,
    Avx512,
}

lazy_static! {
    /// Support for FP16 SIMD operations
    pub static ref FP16_SIMD_SUPPORT: SimdSupport = {
        #[cfg(target_arch = "aarch64")]
        {
            if aarch64::has_neon_f16_support() {
                SimdSupport::Neon
            } else {
                SimdSupport::None
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if x86::has_avx512_f16_support() {
                SimdSupport::Avx512
            } else if is_x86_feature_detected!("avx2") {
                SimdSupport::Avx2
            } else {
                SimdSupport::None
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::__cpuid;

    use lazy_static::lazy_static;

    #[inline]
    fn check_flag(x: usize, position: u32) -> bool {
        x & (1 << position) != 0
    }

    pub fn has_avx512_f16_support() -> bool {
        // this macro does many OS checks/etc. to determine if allowed to use AVX512
        if !is_x86_feature_detected!("avx512f") {
            return false;
        }

        // EAX=7, ECX=0: Extended Features (includes AVX512)
        // More info on calling CPUID can be found here (section 1.4)
        // https://www.intel.com/content/dam/develop/external/us/en/documents/architecture-instruction-set-extensions-programming-reference.pdf
        let ext_cpuid_result = unsafe { __cpuid(7) };
        check_flag(ext_cpuid_result.edx as usize, 23)
    }
}

// Inspired by https://github.com/RustCrypto/utils/blob/master/cpufeatures/src/aarch64.rs
// aarch64 doesn't have userspace feature detection built in, so we have to call
// into OS-specific functions to check for features.

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        // Maybe we can assume it's there?
        true
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        // See: https://github.com/rust-lang/libc/blob/7ce81ca7aeb56aae7ca0237ef9353d58f3d7d2f1/src/unix/linux_like/linux/gnu/b64/aarch64/mod.rs#L533
        libc::__getauxval(libc::AT_HWCAP) & hwcaps::HWCAP_FPHP != 0
    }
}