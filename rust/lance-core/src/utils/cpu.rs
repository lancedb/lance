// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use std::sync::LazyLock;

/// A level of SIMD support for some feature.
///
/// `#[non_exhaustive]` so future tiers (e.g. AVX-512 BF16, AMX) can be added
/// without breaking external `match` consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimdSupport {
    None,
    Neon,
    Sse,
    /// AVX (256-bit float ops) but no FMA and no AVX2.
    /// Intel Sandy Bridge / Ivy Bridge.
    Avx,
    /// AVX + FMA but no AVX2.
    /// AMD Piledriver / Steamroller / FX-7500.
    AvxFma,
    Avx2,
    Avx512,
    Avx512FP16,
    Lsx,
    Lasx,
}

impl fmt::Display for SimdSupport {
    /// Formats the tier name in lowercase, matching pyarrow's
    /// `runtime_info().simd_level` convention.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::None => "none",
            Self::Neon => "neon",
            Self::Sse => "sse",
            Self::Avx => "avx",
            Self::AvxFma => "avx_fma",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx512FP16 => "avx512_fp16",
            Self::Lsx => "lsx",
            Self::Lasx => "lasx",
        };
        f.write_str(name)
    }
}

/// Snapshot of the SIMD tier lance dispatches to on the current host, plus the
/// raw CPU features detected for diagnostic purposes.
///
/// Mirrors the role of `pyarrow.runtime_info()`: a single, cheap call users can
/// make to verify which SIMD tier the runtime selected and what underlying
/// features the host advertises.
#[derive(Debug, Clone)]
pub struct SimdInfo {
    /// The SIMD tier lance dispatches to at runtime on this host.
    pub tier: SimdSupport,
    /// The architecture name (e.g. "x86_64", "aarch64", "loongarch64").
    pub target_arch: &'static str,
    /// Raw CPU feature flags detected on this host (x86_64 only; empty on
    /// other architectures). Each entry is a feature name like "avx2",
    /// "fma", "avx512f", "popcnt", etc.
    pub host_features: Vec<&'static str>,
}

/// Returns a snapshot of the SIMD tier lance is using on this host along with
/// the raw CPU feature flags that drove the decision.
///
/// Useful for performance debugging and giving users a way to verify which
/// dispatch tier they are hitting without rebuilding lance.
pub fn simd_info() -> SimdInfo {
    SimdInfo {
        tier: *SIMD_SUPPORT,
        target_arch: std::env::consts::ARCH,
        host_features: detect_host_features(),
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_host_features() -> Vec<&'static str> {
    // Each call must be inline: `is_x86_feature_detected!` does its own custom
    // input parsing and rejects feature names received via a `macro_rules!`
    // `:literal` metavariable on some toolchains.
    let mut features = Vec::new();
    if is_x86_feature_detected!("sse2") {
        features.push("sse2");
    }
    if is_x86_feature_detected!("sse3") {
        features.push("sse3");
    }
    if is_x86_feature_detected!("ssse3") {
        features.push("ssse3");
    }
    if is_x86_feature_detected!("sse4.1") {
        features.push("sse4.1");
    }
    if is_x86_feature_detected!("sse4.2") {
        features.push("sse4.2");
    }
    if is_x86_feature_detected!("popcnt") {
        features.push("popcnt");
    }
    if is_x86_feature_detected!("avx") {
        features.push("avx");
    }
    if is_x86_feature_detected!("avx2") {
        features.push("avx2");
    }
    if is_x86_feature_detected!("fma") {
        features.push("fma");
    }
    if is_x86_feature_detected!("f16c") {
        features.push("f16c");
    }
    if is_x86_feature_detected!("bmi1") {
        features.push("bmi1");
    }
    if is_x86_feature_detected!("bmi2") {
        features.push("bmi2");
    }
    if is_x86_feature_detected!("avx512f") {
        features.push("avx512f");
    }
    if is_x86_feature_detected!("avx512bw") {
        features.push("avx512bw");
    }
    if is_x86_feature_detected!("avx512cd") {
        features.push("avx512cd");
    }
    if is_x86_feature_detected!("avx512dq") {
        features.push("avx512dq");
    }
    if is_x86_feature_detected!("avx512vl") {
        features.push("avx512vl");
    }
    features
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_host_features() -> Vec<&'static str> {
    Vec::new()
}

/// Support for SIMD operations
pub static SIMD_SUPPORT: LazyLock<SimdSupport> = LazyLock::new(|| {
    #[cfg(all(target_arch = "aarch64", any(target_os = "ios", target_os = "tvos")))]
    {
        // AArch64 iOS/tvOS has NEON; fp16 arithmetic is available on modern targets.
        SimdSupport::Neon
    }
    #[cfg(all(
        target_arch = "aarch64",
        not(any(target_os = "ios", target_os = "tvos"))
    ))]
    {
        if aarch64::has_neon_f16_support() {
            SimdSupport::Neon
        } else {
            SimdSupport::None
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx512() {
            if x86::has_avx512_f16_support() {
                SimdSupport::Avx512FP16
            } else {
                SimdSupport::Avx512
            }
        } else if is_x86_feature_detected!("avx2") {
            SimdSupport::Avx2
        } else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            // AMD Piledriver / Steamroller / FX-7500: 256-bit float ops + FMA but no AVX2.
            SimdSupport::AvxFma
        } else if is_x86_feature_detected!("avx") {
            // Intel Sandy Bridge / Ivy Bridge: 256-bit float ops without FMA.
            SimdSupport::Avx
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
});

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::__cpuid;

    #[inline]
    fn check_flag(x: usize, position: u32) -> bool {
        x & (1 << position) != 0
    }

    pub fn has_avx512_f16_support() -> bool {
        // this macro does many OS checks/etc. to determine if allowed to use AVX512
        if !has_avx512() {
            return false;
        }

        // EAX=7, ECX=0: Extended Features (includes AVX512)
        // More info on calling CPUID can be found here (section 1.4)
        // https://www.intel.com/content/dam/develop/external/us/en/documents/architecture-instruction-set-extensions-programming-reference.pdf
        // __cpuid is safe in nightly but unsafe in stable, allow both
        #[allow(unused_unsafe)]
        let ext_cpuid_result = unsafe { __cpuid(7) };
        check_flag(ext_cpuid_result.edx as usize, 23)
    }

    pub fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
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
        let flags = unsafe { libc::getauxval(libc::AT_HWCAP) };
        flags & libc::HWCAP_FPHP != 0
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "windows"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        // https://github.com/lance-format/lance/issues/2411
        false
    }
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

#[cfg(all(target_arch = "aarch64", target_os = "android"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_info_exposes_tier() {
        let info = simd_info();
        assert_eq!(info.target_arch, std::env::consts::ARCH);
        // Tier should match the detected SIMD support.
        assert_eq!(info.tier, *SIMD_SUPPORT);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_info_features_include_baseline() {
        let info = simd_info();
        // The x86_64 ABI mandates SSE2, so it must always be present on this
        // architecture.
        assert!(info.host_features.contains(&"sse2"));
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn simd_info_features_empty_off_x86_64() {
        let info = simd_info();
        assert!(info.host_features.is_empty());
    }

    #[test]
    fn simd_support_display_matches_lowercase_convention() {
        assert_eq!(SimdSupport::Avx2.to_string(), "avx2");
        assert_eq!(SimdSupport::AvxFma.to_string(), "avx_fma");
        assert_eq!(SimdSupport::Avx.to_string(), "avx");
        assert_eq!(SimdSupport::Avx512.to_string(), "avx512");
        assert_eq!(SimdSupport::Avx512FP16.to_string(), "avx512_fp16");
        assert_eq!(SimdSupport::None.to_string(), "none");
        assert_eq!(SimdSupport::Neon.to_string(), "neon");
        assert_eq!(SimdSupport::Sse.to_string(), "sse");
        assert_eq!(SimdSupport::Lsx.to_string(), "lsx");
        assert_eq!(SimdSupport::Lasx.to_string(), "lasx");
    }
}
