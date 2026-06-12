// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SIMD kernels for quantizing the RaBitQ FastScan distance table.
//!
//! Once per (query, probed partition) the `dim * 4`-entry `f32` distance
//! table is quantized into `u8` (fast/normal approx modes) or `u16`
//! (accurate mode) FastScan LUT entries: a min/max pass over the table
//! followed by an affine quantize-and-narrow pass. Both passes are branchy
//! in scalar form, so they get the same runtime-dispatch treatment as
//! [`super::ex_dot`]: explicit AVX-512/AVX2 kernels on x86_64 and a portable
//! fold elsewhere that LLVM auto-vectorizes (NEON is part of the aarch64
//! baseline).
//!
//! Table values are sums of rotated-query components: always finite, never
//! NaN, so lanewise IEEE `min`/`max` matches `total_cmp` ordering. The only
//! divergence is the sign of zero, which callers cannot observe: `d - qmin`
//! and the `qmin == qmax` early-out are arithmetically identical either way.
//!
//! Quantization rounds half-to-even (the SSE/AVX default rounding mode)
//! rather than `f32::round`'s half-away-from-zero so that the scalar
//! fallback and the SIMD kernels agree bit-exactly; relative to the pre-SIMD
//! implementation this can move a LUT entry by 1 on exact .5 ties, which is
//! within the table's inherent quantization error.

use std::mem::MaybeUninit;
use std::sync::LazyLock;

type MinMaxFn = fn(&[f32]) -> (f32, f32);
type QuantizeU8Fn = fn(&[f32], f32, f32, &mut [MaybeUninit<u8>]);
type QuantizeU16Fn = fn(&[f32], f32, f32, &mut [MaybeUninit<u16>]);

/// Quantize `dist_table` into `u8` FastScan LUT entries in the caller-owned
/// scratch buffer, returning the table's `(min, max)`.
///
/// An all-equal table (e.g. from an all-zero query) produces a zeroed LUT;
/// callers reconstruct distances from `min` alone. `dist_table` must be
/// non-empty and all values finite.
pub fn quantize_dist_table_into(
    dist_table: &[f32],
    quantized_dist_table: &mut Vec<u8>,
) -> (f32, f32) {
    debug_assert!(!dist_table.is_empty(), "dist table must be non-empty");
    let (qmin, qmax) = min_max(dist_table);
    let range = qmax - qmin;
    let factor = u8::MAX as f32 / range;
    // A zero range happens if the query is all zeros. A non-finite `range`
    // or `factor` (table spread overflowing f32 or below ~255/f32::MAX)
    // would produce NaN/inf products on which the kernels' saturating
    // narrows disagree. Either way the table carries no information at u8
    // resolution: emit a zeroed LUT that callers reconstruct from `min`.
    if !range.is_finite() || !factor.is_finite() {
        quantized_dist_table.clear();
        quantized_dist_table.resize(dist_table.len(), 0);
        return (qmin, qmax);
    }
    quantized_dist_table.clear();
    quantized_dist_table.reserve(dist_table.len());
    quantize_u8(
        dist_table,
        qmin,
        factor,
        &mut quantized_dist_table.spare_capacity_mut()[..dist_table.len()],
    );
    // SAFETY: the kernel initialized every element in the reserved range.
    unsafe {
        quantized_dist_table.set_len(dist_table.len());
    }
    (qmin, qmax)
}

/// `u16` variant of [`quantize_dist_table_into`] for the accurate approx
/// mode.
pub fn quantize_dist_table_u16_into(
    dist_table: &[f32],
    quantized_dist_table: &mut Vec<u16>,
) -> (f32, f32) {
    debug_assert!(!dist_table.is_empty(), "dist table must be non-empty");
    let (qmin, qmax) = min_max(dist_table);
    let range = qmax - qmin;
    let factor = u16::MAX as f32 / range;
    // See quantize_dist_table_into: degenerate ranges collapse to a zeroed
    // LUT to keep all kernels bit-exact.
    if !range.is_finite() || !factor.is_finite() {
        quantized_dist_table.clear();
        quantized_dist_table.resize(dist_table.len(), 0);
        return (qmin, qmax);
    }
    quantized_dist_table.clear();
    quantized_dist_table.reserve(dist_table.len());
    quantize_u16(
        dist_table,
        qmin,
        factor,
        &mut quantized_dist_table.spare_capacity_mut()[..dist_table.len()],
    );
    // SAFETY: the kernel initialized every element in the reserved range.
    unsafe {
        quantized_dist_table.set_len(dist_table.len());
    }
    (qmin, qmax)
}

fn min_max(values: &[f32]) -> (f32, f32) {
    static KERNEL: LazyLock<MinMaxFn> = LazyLock::new(select_min_max);
    KERNEL(values)
}

fn quantize_u8(values: &[f32], qmin: f32, factor: f32, out: &mut [MaybeUninit<u8>]) {
    static KERNEL: LazyLock<QuantizeU8Fn> = LazyLock::new(select_quantize_u8);
    KERNEL(values, qmin, factor, out)
}

fn quantize_u16(values: &[f32], qmin: f32, factor: f32, out: &mut [MaybeUninit<u16>]) {
    static KERNEL: LazyLock<QuantizeU16Fn> = LazyLock::new(select_quantize_u16);
    KERNEL(values, qmin, factor, out)
}

fn select_min_max() -> MinMaxFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return x86::min_max_avx512_dispatch;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86::min_max_avx2_dispatch;
        }
    }
    min_max_fold
}

fn select_quantize_u8() -> QuantizeU8Fn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return x86::quantize_u8_avx512_dispatch;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86::quantize_u8_avx2_dispatch;
        }
    }
    quantize_u8_scalar
}

fn select_quantize_u16() -> QuantizeU16Fn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return x86::quantize_u16_avx512_dispatch;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86::quantize_u16_avx2_dispatch;
        }
    }
    quantize_u16_scalar
}

const FOLD_LANES: usize = 16;

/// Portable 16-lane min/max fold; the scalar fallback and the aarch64 path.
/// The `if` comparisons (rather than `f32::min`/`max`, which carry NaN
/// bookkeeping) lower to lanewise min/max instructions on targets with
/// baseline SIMD.
fn min_max_fold(values: &[f32]) -> (f32, f32) {
    let mut mins = [f32::INFINITY; FOLD_LANES];
    let mut maxs = [f32::NEG_INFINITY; FOLD_LANES];
    let mut chunks = values.chunks_exact(FOLD_LANES);
    for chunk in &mut chunks {
        let chunk: &[f32; FOLD_LANES] = chunk.try_into().expect("chunks_exact length");
        for (i, &v) in chunk.iter().enumerate() {
            mins[i] = if v < mins[i] { v } else { mins[i] };
            maxs[i] = if v > maxs[i] { v } else { maxs[i] };
        }
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in mins {
        min = if v < min { v } else { min };
    }
    for v in maxs {
        max = if v > max { v } else { max };
    }
    for &v in chunks.remainder() {
        min = if v < min { v } else { min };
        max = if v > max { v } else { max };
    }
    (min, max)
}

fn quantize_u8_scalar(values: &[f32], qmin: f32, factor: f32, out: &mut [MaybeUninit<u8>]) {
    debug_assert_eq!(values.len(), out.len());
    for (quantized, &d) in out.iter_mut().zip(values) {
        quantized.write(((d - qmin) * factor).round_ties_even() as u8);
    }
}

fn quantize_u16_scalar(values: &[f32], qmin: f32, factor: f32, out: &mut [MaybeUninit<u16>]) {
    debug_assert_eq!(values.len(), out.len());
    for (quantized, &d) in out.iter_mut().zip(values) {
        quantized.write(((d - qmin) * factor).round_ties_even() as u16);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;

    use super::{quantize_u8_scalar, quantize_u16_scalar};

    pub(super) fn min_max_avx512_dispatch(values: &[f32]) -> (f32, f32) {
        // SAFETY: only selected when AVX-512F was detected.
        unsafe { min_max_avx512(values) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn min_max_avx512(values: &[f32]) -> (f32, f32) {
        // Two accumulators per direction break the lanewise min/max latency
        // chain; they are reduced once at the end.
        let mut min0 = _mm512_set1_ps(f32::INFINITY);
        let mut min1 = min0;
        let mut max0 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut max1 = max0;
        let mut chunks = values.chunks_exact(32);
        for chunk in &mut chunks {
            // SAFETY: the chunk holds 32 consecutive floats.
            let (v0, v1) = unsafe {
                (
                    _mm512_loadu_ps(chunk.as_ptr()),
                    _mm512_loadu_ps(chunk.as_ptr().add(16)),
                )
            };
            min0 = _mm512_min_ps(min0, v0);
            max0 = _mm512_max_ps(max0, v0);
            min1 = _mm512_min_ps(min1, v1);
            max1 = _mm512_max_ps(max1, v1);
        }
        let mut min = _mm512_reduce_min_ps(_mm512_min_ps(min0, min1));
        let mut max = _mm512_reduce_max_ps(_mm512_max_ps(max0, max1));
        for &v in chunks.remainder() {
            min = if v < min { v } else { min };
            max = if v > max { v } else { max };
        }
        (min, max)
    }

    pub(super) fn min_max_avx2_dispatch(values: &[f32]) -> (f32, f32) {
        // SAFETY: only selected when AVX2 was detected.
        unsafe { min_max_avx2(values) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn min_max_avx2(values: &[f32]) -> (f32, f32) {
        let mut min0 = _mm256_set1_ps(f32::INFINITY);
        let mut min1 = min0;
        let mut max0 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max1 = max0;
        let mut chunks = values.chunks_exact(16);
        for chunk in &mut chunks {
            // SAFETY: the chunk holds 16 consecutive floats.
            let (v0, v1) = unsafe {
                (
                    _mm256_loadu_ps(chunk.as_ptr()),
                    _mm256_loadu_ps(chunk.as_ptr().add(8)),
                )
            };
            min0 = _mm256_min_ps(min0, v0);
            max0 = _mm256_max_ps(max0, v0);
            min1 = _mm256_min_ps(min1, v1);
            max1 = _mm256_max_ps(max1, v1);
        }
        let mut min = reduce_min_avx2(_mm256_min_ps(min0, min1));
        let mut max = reduce_max_avx2(_mm256_max_ps(max0, max1));
        for &v in chunks.remainder() {
            min = if v < min { v } else { min };
            max = if v > max { v } else { max };
        }
        (min, max)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn reduce_min_avx2(v: __m256) -> f32 {
        let halves = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let pairs = _mm_min_ps(halves, _mm_movehl_ps(halves, halves));
        let single = _mm_min_ss(pairs, _mm_shuffle_ps::<0b01>(pairs, pairs));
        _mm_cvtss_f32(single)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn reduce_max_avx2(v: __m256) -> f32 {
        let halves = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let pairs = _mm_max_ps(halves, _mm_movehl_ps(halves, halves));
        let single = _mm_max_ss(pairs, _mm_shuffle_ps::<0b01>(pairs, pairs));
        _mm_cvtss_f32(single)
    }

    /// Load 16 floats and affine-quantize them into `i32` lanes; the convert
    /// rounds to nearest-even (the MXCSR default).
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn quantize16_epi32(src: *const f32, min: __m512, factor: __m512) -> __m512i {
        // SAFETY: the caller guarantees 16 floats are readable at `src`.
        let v = unsafe { _mm512_loadu_ps(src) };
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(v, min), factor))
    }

    pub(super) fn quantize_u8_avx512_dispatch(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u8>],
    ) {
        // SAFETY: only selected when AVX-512F was detected.
        unsafe { quantize_u8_avx512(values, qmin, factor, out) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn quantize_u8_avx512(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u8>],
    ) {
        debug_assert_eq!(values.len(), out.len());
        let min = _mm512_set1_ps(qmin);
        let factor_v = _mm512_set1_ps(factor);
        let full = values.len() - values.len() % 16;
        let src = values.as_ptr();
        let dst = out.as_mut_ptr().cast::<u8>();
        for i in (0..full).step_by(16) {
            // SAFETY: `i + 16 <= values.len() == out.len()`.
            unsafe {
                let q = quantize16_epi32(src.add(i), min, factor_v);
                // Unsigned-saturating i32 -> u8 narrow: lanes are in
                // [0, 255] plus float epsilon, which saturation clips.
                _mm_storeu_si128(dst.add(i).cast(), _mm512_cvtusepi32_epi8(q));
            }
        }
        quantize_u8_scalar(&values[full..], qmin, factor, &mut out[full..]);
    }

    pub(super) fn quantize_u16_avx512_dispatch(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u16>],
    ) {
        // SAFETY: only selected when AVX-512F was detected.
        unsafe { quantize_u16_avx512(values, qmin, factor, out) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn quantize_u16_avx512(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u16>],
    ) {
        debug_assert_eq!(values.len(), out.len());
        let min = _mm512_set1_ps(qmin);
        let factor_v = _mm512_set1_ps(factor);
        let full = values.len() - values.len() % 16;
        let src = values.as_ptr();
        let dst = out.as_mut_ptr().cast::<u16>();
        for i in (0..full).step_by(16) {
            // SAFETY: `i + 16 <= values.len() == out.len()`.
            unsafe {
                let q = quantize16_epi32(src.add(i), min, factor_v);
                _mm256_storeu_si256(dst.add(i).cast(), _mm512_cvtusepi32_epi16(q));
            }
        }
        quantize_u16_scalar(&values[full..], qmin, factor, &mut out[full..]);
    }

    /// Load 8 floats and affine-quantize them into `i32` lanes; the convert
    /// rounds to nearest-even (the MXCSR default).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize8_epi32(src: *const f32, min: __m256, factor: __m256) -> __m256i {
        // SAFETY: the caller guarantees 8 floats are readable at `src`.
        let v = unsafe { _mm256_loadu_ps(src) };
        _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(v, min), factor))
    }

    pub(super) fn quantize_u8_avx2_dispatch(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u8>],
    ) {
        // SAFETY: only selected when AVX2 was detected.
        unsafe { quantize_u8_avx2(values, qmin, factor, out) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quantize_u8_avx2(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u8>],
    ) {
        debug_assert_eq!(values.len(), out.len());
        let min = _mm256_set1_ps(qmin);
        let factor_v = _mm256_set1_ps(factor);
        // The 32->16 and 16->8 packs interleave the two 128-bit lanes; this
        // permutation of 32-bit groups restores natural order.
        let restore = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let full = values.len() - values.len() % 32;
        let src = values.as_ptr();
        let dst = out.as_mut_ptr().cast::<u8>();
        for i in (0..full).step_by(32) {
            // SAFETY: `i + 32 <= values.len() == out.len()`.
            unsafe {
                let q0 = quantize8_epi32(src.add(i), min, factor_v);
                let q1 = quantize8_epi32(src.add(i + 8), min, factor_v);
                let q2 = quantize8_epi32(src.add(i + 16), min, factor_v);
                let q3 = quantize8_epi32(src.add(i + 24), min, factor_v);
                // Unsigned-saturating i32 -> u16 -> u8 narrows: lanes are in
                // [0, 255] plus float epsilon, which saturation clips.
                let lo = _mm256_packus_epi32(q0, q1);
                let hi = _mm256_packus_epi32(q2, q3);
                let bytes = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(lo, hi), restore);
                _mm256_storeu_si256(dst.add(i).cast(), bytes);
            }
        }
        quantize_u8_scalar(&values[full..], qmin, factor, &mut out[full..]);
    }

    pub(super) fn quantize_u16_avx2_dispatch(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u16>],
    ) {
        // SAFETY: only selected when AVX2 was detected.
        unsafe { quantize_u16_avx2(values, qmin, factor, out) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quantize_u16_avx2(
        values: &[f32],
        qmin: f32,
        factor: f32,
        out: &mut [MaybeUninit<u16>],
    ) {
        debug_assert_eq!(values.len(), out.len());
        let min = _mm256_set1_ps(qmin);
        let factor_v = _mm256_set1_ps(factor);
        let full = values.len() - values.len() % 16;
        let src = values.as_ptr();
        let dst = out.as_mut_ptr().cast::<u16>();
        for i in (0..full).step_by(16) {
            // SAFETY: `i + 16 <= values.len() == out.len()`.
            unsafe {
                let q0 = quantize8_epi32(src.add(i), min, factor_v);
                let q1 = quantize8_epi32(src.add(i + 8), min, factor_v);
                // The pack interleaves the 128-bit lanes as
                // [q0_lo, q1_lo, q0_hi, q1_hi]; the 64-bit-lane permute
                // restores [q0_lo, q0_hi, q1_lo, q1_hi].
                let packed = _mm256_packus_epi32(q0, q1);
                let words = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
                _mm256_storeu_si256(dst.add(i).cast(), words);
            }
        }
        quantize_u16_scalar(&values[full..], qmin, factor, &mut out[full..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    /// Straightforward scalar reference implementing the documented
    /// semantics: `total_cmp` min/max plus nearest-even rounding.
    fn reference_min_max(values: &[f32]) -> (f32, f32) {
        let min = values
            .iter()
            .cloned()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap();
        let max = values
            .iter()
            .cloned()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        (min, max)
    }

    fn reference_u8(values: &[f32]) -> (Vec<u8>, f32, f32) {
        let (qmin, qmax) = reference_min_max(values);
        let range = qmax - qmin;
        let factor = u8::MAX as f32 / range;
        if !range.is_finite() || !factor.is_finite() {
            return (vec![0; values.len()], qmin, qmax);
        }
        let quantized = values
            .iter()
            .map(|&d| ((d - qmin) * factor).round_ties_even() as u8)
            .collect();
        (quantized, qmin, qmax)
    }

    fn reference_u16(values: &[f32]) -> (Vec<u16>, f32, f32) {
        let (qmin, qmax) = reference_min_max(values);
        let range = qmax - qmin;
        let factor = u16::MAX as f32 / range;
        if !range.is_finite() || !factor.is_finite() {
            return (vec![0; values.len()], qmin, qmax);
        }
        let quantized = values
            .iter()
            .map(|&d| ((d - qmin) * factor).round_ties_even() as u16)
            .collect();
        (quantized, qmin, qmax)
    }

    fn available_kernels() -> Vec<(&'static str, MinMaxFn, QuantizeU8Fn, QuantizeU16Fn)> {
        // `mut` is only exercised on x86_64 where extra kernels may be pushed.
        #[allow(unused_mut)]
        let mut kernels = vec![(
            "scalar",
            min_max_fold as MinMaxFn,
            quantize_u8_scalar as QuantizeU8Fn,
            quantize_u16_scalar as QuantizeU16Fn,
        )];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                kernels.push((
                    "avx2",
                    x86::min_max_avx2_dispatch,
                    x86::quantize_u8_avx2_dispatch,
                    x86::quantize_u16_avx2_dispatch,
                ));
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                kernels.push((
                    "avx512",
                    x86::min_max_avx512_dispatch,
                    x86::quantize_u8_avx512_dispatch,
                    x86::quantize_u16_avx512_dispatch,
                ));
            }
        }
        kernels
    }

    /// Every available kernel must agree bit-exactly with the reference on
    /// the given input.
    fn check_against_reference(values: &[f32]) {
        let (expected_u8, expected_min, expected_max) = reference_u8(values);
        let (expected_u16, ..) = reference_u16(values);

        for (name, min_max_fn, quantize_u8_fn, quantize_u16_fn) in available_kernels() {
            let (qmin, qmax) = min_max_fn(values);
            assert_eq!(
                (qmin, qmax),
                (expected_min, expected_max),
                "kernel={name} len={}",
                values.len()
            );

            // The public entry points never hand a degenerate factor to the
            // quantize kernels; mirror that here.
            let factor_u8 = u8::MAX as f32 / (qmax - qmin);
            if (qmax - qmin).is_finite() && factor_u8.is_finite() {
                let mut out_u8 = Vec::with_capacity(values.len());
                quantize_u8_fn(
                    values,
                    qmin,
                    factor_u8,
                    &mut out_u8.spare_capacity_mut()[..values.len()],
                );
                // SAFETY: the kernel initialized every element.
                unsafe { out_u8.set_len(values.len()) };
                assert_eq!(out_u8, expected_u8, "kernel={name} len={}", values.len());
            }

            let factor_u16 = u16::MAX as f32 / (qmax - qmin);
            if (qmax - qmin).is_finite() && factor_u16.is_finite() {
                let mut out_u16 = Vec::with_capacity(values.len());
                quantize_u16_fn(
                    values,
                    qmin,
                    factor_u16,
                    &mut out_u16.spare_capacity_mut()[..values.len()],
                );
                // SAFETY: the kernel initialized every element.
                unsafe { out_u16.set_len(values.len()) };
                assert_eq!(out_u16, expected_u16, "kernel={name} len={}", values.len());
            }
        }

        // The public entry points exercise the dispatched kernels plus the
        // scratch-buffer handling.
        let mut out_u8 = Vec::new();
        assert_eq!(
            quantize_dist_table_into(values, &mut out_u8),
            (expected_min, expected_max)
        );
        assert_eq!(out_u8, expected_u8);
        let mut out_u16 = Vec::new();
        assert_eq!(
            quantize_dist_table_u16_into(values, &mut out_u16),
            (expected_min, expected_max)
        );
        assert_eq!(out_u16, expected_u16);
    }

    #[rstest]
    fn test_quantize_matches_reference(
        #[values(1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 100, 6144, 6160)] len: usize,
        #[values(1.0, 1e-3, 1e4)] scale: f32,
    ) {
        let mut rng = SmallRng::seed_from_u64(42 + len as u64);
        let values = (0..len)
            .map(|_| rng.random_range(-scale..scale))
            .collect::<Vec<_>>();
        check_against_reference(&values);
    }

    /// Integer tables with range 510 (resp. 131070) make `factor` exactly
    /// 0.5, so odd values land on exact .5 ties; all kernels must round them
    /// to even and agree with each other.
    #[test]
    fn test_exact_half_ties_round_to_even() {
        let values = (0..=510).map(|v| v as f32).collect::<Vec<_>>();
        check_against_reference(&values);
        let mut quantized = Vec::new();
        let (qmin, qmax) = quantize_dist_table_into(&values, &mut quantized);
        assert_eq!((qmin, qmax), (0.0, 510.0));
        // Spot-check nearest-even: 0.5 -> 0, 1.5 -> 2, 127.5 -> 128,
        // 254.5 -> 254.
        assert_eq!(&quantized[..6], &[0, 0, 1, 2, 2, 2]);
        assert_eq!(quantized[255], 128);
        assert_eq!(quantized[509], 254);
        assert_eq!(quantized[510], 255);

        // Integers up to 131070 are exactly representable in f32.
        let values = (0..=510).map(|v| (v * 257) as f32).collect::<Vec<_>>();
        check_against_reference(&values);
        let mut quantized = Vec::new();
        let (qmin, qmax) = quantize_dist_table_u16_into(&values, &mut quantized);
        assert_eq!((qmin, qmax), (0.0, 131070.0));
        // value * 0.5 = 128.5 -> 128, 385.5 -> 386 under nearest-even.
        assert_eq!(&quantized[..4], &[0, 128, 257, 386]);
        assert_eq!(quantized[510], u16::MAX);
    }

    #[test]
    fn test_negative_and_mixed_sign_values() {
        let mut rng = SmallRng::seed_from_u64(7);
        let values = (0..1000)
            .map(|_| rng.random_range(-100.0f32..-1.0))
            .collect::<Vec<_>>();
        check_against_reference(&values);
        let values = (0..999)
            .map(|i| (i as f32 - 499.5) * 0.75)
            .collect::<Vec<_>>();
        check_against_reference(&values);
    }

    #[rstest]
    fn test_all_equal_input_zeroes_table(#[values(0.0, -7.25, 3.5)] value: f32) {
        let values = vec![value; 100];
        check_against_reference(&values);
        let mut quantized = vec![1u8; 5];
        let (qmin, qmax) = quantize_dist_table_into(&values, &mut quantized);
        assert_eq!((qmin, qmax), (value, value));
        assert_eq!(quantized, vec![0; 100]);
        let mut quantized = vec![1u16; 5];
        let (qmin, qmax) = quantize_dist_table_u16_into(&values, &mut quantized);
        assert_eq!((qmin, qmax), (value, value));
        assert_eq!(quantized, vec![0; 100]);
    }

    /// Degenerate finite ranges make `factor` (or `range` itself) non-finite
    /// and must collapse to a zeroed LUT on every kernel: NaN/inf products
    /// would otherwise hit the saturating narrows, which disagree across
    /// scalar, AVX2, and AVX-512.
    #[test]
    fn test_degenerate_range_zeroes_table() {
        // factor = 255 / 1e-38 overflows to +inf.
        let mut tiny_range = vec![0.0f32; 32];
        tiny_range[1] = 1e-38;
        // qmax - qmin overflows to +inf, so factor is 0 and products are NaN.
        let mut huge_range = vec![0.0f32; 32];
        huge_range[0] = -2e38;
        huge_range[1] = 2e38;
        // factor = 65535 / 1e-35 overflows only in the u16 variant; the u8
        // variant still quantizes normally.
        let mut u16_only = vec![0.0f32; 32];
        u16_only[1] = 1e-35;

        for values in [&tiny_range, &huge_range, &u16_only] {
            check_against_reference(values);
        }
        let mut quantized_u8 = Vec::new();
        let (qmin, qmax) = quantize_dist_table_into(&tiny_range, &mut quantized_u8);
        assert_eq!((qmin, qmax), (0.0, 1e-38));
        assert_eq!(quantized_u8, vec![0; 32]);
        quantize_dist_table_into(&huge_range, &mut quantized_u8);
        assert_eq!(quantized_u8, vec![0; 32]);
        let mut quantized_u16 = Vec::new();
        quantize_dist_table_u16_into(&u16_only, &mut quantized_u16);
        assert_eq!(quantized_u16, vec![0; 32]);
        quantize_dist_table_into(&u16_only, &mut quantized_u8);
        assert_eq!(quantized_u8[1], u8::MAX);
    }

    /// `-0.0 == 0.0` must keep taking the all-equal early-out even though
    /// SIMD min/max may pick either sign for the extremes.
    #[test]
    fn test_signed_zero_mix_zeroes_table() {
        let mut values = vec![0.0f32; 64];
        values.iter_mut().step_by(2).for_each(|v| *v = -0.0);
        let mut quantized = Vec::new();
        let (qmin, qmax) = quantize_dist_table_into(&values, &mut quantized);
        assert_eq!(qmin, qmax);
        assert_eq!(quantized, vec![0; 64]);
    }

    /// The scratch buffer must be fully overwritten across reuses with
    /// different lengths.
    #[test]
    fn test_scratch_buffer_reuse() {
        let mut rng = SmallRng::seed_from_u64(11);
        let mut scratch_u8 = vec![7u8; 500];
        let mut scratch_u16 = vec![7u16; 500];
        for len in [48, 512, 16] {
            let values = (0..len)
                .map(|_| rng.random_range(-1.0f32..1.0))
                .collect::<Vec<_>>();
            quantize_dist_table_into(&values, &mut scratch_u8);
            assert_eq!(scratch_u8, reference_u8(&values).0);
            quantize_dist_table_u16_into(&values, &mut scratch_u16);
            assert_eq!(scratch_u16, reference_u16(&values).0);
        }
    }
}
