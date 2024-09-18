// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lazy_static::lazy_static;

/// A level of SIMD support for some feature
pub enum SimdSupport {
    None,
    Neon,
    Sse,
    Avx2,
    Avx512,
    Lsx,
    Lasx,
}

lazy_static! {
    /// Support for FP16 SIMD operations
    pub static ref FP16_SIMD_SUPPORT: SimdSupport = {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("fp16") {
                SimdSupport::Neon
            } else {
                SimdSupport::None
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512fp16") {
                SimdSupport::Avx512
            } else if is_x86_feature_detected!("avx2") {
                SimdSupport::Avx2
            } else {
                SimdSupport::None
            }
        }
        #[cfg(target_arch = "loongarch64")]
        {
            if loongarch64::has_lasx_support() {
                SimdSupport::Lasx
            } else if loongarch64::has_lsx_support() {
                SimdSupport::Lsx
            } else {
                SimdSupport::None
            }
        }
    };
}


#[cfg(target_arch = "loongarch64")]
mod loongarch64 {
    pub fn has_lsx_support() -> bool {
        // See: https://github.com/rust-lang/libc/blob/7ce81ca7aeb56aae7ca0237ef9353d58f3d7d2f1/src/unix/linux_like/linux/gnu/b64/loongarch64/mod.rs#L263
        let flags = unsafe { libc::getauxval(libc::AT_HWCAP) };
        flags & libc::HWCAP_LOONGARCH_LSX != 0
    }
    pub fn has_lasx_support() -> bool {
        // See: https://github.com/rust-lang/libc/blob/7ce81ca7aeb56aae7ca0237ef9353d58f3d7d2f1/src/unix/linux_like/linux/gnu/b64/loongarch64/mod.rs#L264
        let flags = unsafe { libc::getauxval(libc::AT_HWCAP) };
        flags & libc::HWCAP_LOONGARCH_LASX != 0
    }
}
