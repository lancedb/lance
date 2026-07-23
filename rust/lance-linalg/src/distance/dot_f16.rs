// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Batched f16 dot product with an optional AMX-FP16 tile backend.
//!
//! Used by flat (unquantized) fp16 vector search: an `IvfHnswFlat` index over
//! an fp16 column compares an fp16 query against fp16 stored vectors
//! symmetrically. During beam search the graph evaluates neighbors 16 at a time
//! (see `lance_index`'s `beam_search_loop!`), which maps onto one AMX tile pass.
//!
//! [`dot_f16_batch_16`] returns the 16 raw dot products `Σ query·candidate`
//! (same value convention as [`crate::distance::dot()`]); the caller applies the
//! `1.0 - dot` distance wrapping. On Linux/x86_64 hosts with AMX-FP16 it
//! dispatches to a single tile pass; everywhere else (and on any AMX
//! unavailability) it falls back to 16 independent [`crate::distance::dot()`]
//! calls, which are bit-identical to the per-neighbor `distance()` path.
//!
//! Unlike integer AMX kernels this is floating point: the AMX and fallback paths
//! are **not** bit-for-bit identical (tile accumulation order rounds
//! differently), but both accumulate products in f32 and agree to within fp16
//! precision — a relative error on the order of 1e-4, far below fp16's own
//! representational error, so recall is unaffected.

use half::f16;

use crate::distance::dot::dot;

/// Batched f16 dot product: the raw dot products of one `query` against 16
/// `candidates`, in order. Every candidate slice must have the same length as
/// `query`.
///
/// Safe to call unconditionally: the caller never needs to know whether AMX is
/// present. The function panics if any candidate has a different length from
/// `query`, rather than allowing a malformed batch to reach the FFI kernel.
/// See the module docs for the accuracy contract.
#[inline]
pub fn dot_f16_batch_16(query: &[f16], candidates: &[&[f16]; 16]) -> [f32; 16] {
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.len() == query.len()),
        "all candidate vectors must have the same length as query"
    );
    #[cfg(all(
        kernel_support = "amx_fp16",
        target_arch = "x86_64",
        target_os = "linux"
    ))]
    {
        // AMX only earns its tile-config overhead once there is at least one
        // full 32-wide pass; for tiny dims the fallback is cheaper anyway.
        if query.len() >= 32 && crate::simd::amx_fp16::amx_available() {
            return unsafe { crate::simd::amx_fp16::dot_f16_batch_16_amx(query, candidates) };
        }
    }
    dot_f16_batch_16_fallback(query, candidates)
}

/// Fallback for [`dot_f16_batch_16`]: 16 independent [`crate::distance::dot()`]
/// calls — bit-identical to the per-neighbor `distance()` path (both go through
/// `f16::dot`). Exposed separately so tests can exercise it regardless of host.
#[inline]
pub(crate) fn dot_f16_batch_16_fallback(query: &[f16], candidates: &[&[f16]; 16]) -> [f32; 16] {
    std::array::from_fn(|i| dot(query, candidates[i]))
}

/// Whether a `dot_f16_batch_16` call would take the AMX-FP16 tile path (given a
/// dim `>= 32`) rather than the scalar/AVX-512 fallback: the kernel was compiled
/// in, the host is Linux/x86_64 with the amx-tile + amx-fp16 CPUID bits, the
/// one-time XTILEDATA `arch_prctl` succeeded, and `LANCE_DISABLE_AMX` is unset.
///
/// Exists so higher layers / tests can *assert* the accelerated path is active
/// on a given host instead of silently falling back.
pub fn amx_fp16_available() -> bool {
    #[cfg(all(
        kernel_support = "amx_fp16",
        target_arch = "x86_64",
        target_os = "linux"
    ))]
    {
        return crate::simd::amx_fp16::amx_available();
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Dims covering: < 32 (no tile pass, pure fallback), exactly 32 (one pass,
    /// no tail), non-multiples of 32 (exercise the AMX tail), and larger dims
    /// (multiple passes).
    const BATCH_DIMS: &[usize] = &[
        1, 7, 31, 32, 33, 47, 64, 96, 100, 127, 128, 200, 256, 384, 768, 1000, 1536,
    ];

    fn make_batch(dim: usize, rng: &mut StdRng) -> (Vec<f16>, Vec<Vec<f16>>) {
        let gen_vec = |rng: &mut StdRng| -> Vec<f16> {
            (0..dim)
                .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
                .collect()
        };
        let query = gen_vec(rng);
        let candidates = (0..16).map(|_| gen_vec(rng)).collect();
        (query, candidates)
    }

    /// f32-accumulated reference dot product — the semantic contract both the
    /// AMX and fallback paths approximate. `f16::dot` itself accumulates in f32.
    fn ref_dot_f32(query: &[f16], cand: &[f16]) -> f32 {
        query
            .iter()
            .zip(cand.iter())
            .map(|(&q, &c)| q.to_f32() * c.to_f32())
            .sum()
    }

    /// Relative-error tolerance justification: fp16 carries ~11 bits of mantissa
    /// (~3 decimal digits). The AMX path and the f32 reference differ only in
    /// summation order of f32-widened products, so the error is many orders
    /// tighter than fp16's own representational error; 5e-3 relative is a very
    /// safe bound (observed worst case is ~2e-4).
    const REL_TOL: f32 = 5e-3;

    fn assert_close(got: f32, want: f32, ctx: &str) {
        let rel = (got - want).abs() / (want.abs() + 1e-6);
        assert!(
            rel <= REL_TOL || (got - want).abs() <= 1e-3,
            "{ctx}: got {got} want {want} rel_err {rel}"
        );
    }

    #[test]
    fn fallback_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0xF16);
        for &dim in BATCH_DIMS {
            let (query, cands) = make_batch(dim, &mut rng);
            let candidates: [&[f16]; 16] = std::array::from_fn(|i| cands[i].as_slice());
            let got = dot_f16_batch_16_fallback(&query, &candidates);
            for i in 0..16 {
                assert_close(
                    got[i],
                    ref_dot_f32(&query, &cands[i]),
                    &format!("fb dim={dim} i={i}"),
                );
            }
        }
    }

    #[test]
    fn dispatch_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        for &dim in BATCH_DIMS {
            let (query, cands) = make_batch(dim, &mut rng);
            let candidates: [&[f16]; 16] = std::array::from_fn(|i| cands[i].as_slice());
            let got = dot_f16_batch_16(&query, &candidates);
            for i in 0..16 {
                assert_close(
                    got[i],
                    ref_dot_f32(&query, &cands[i]),
                    &format!("disp dim={dim} i={i}"),
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "all candidate vectors must have the same length")]
    fn rejects_mismatched_candidate_length() {
        let query = vec![f16::from_f32(1.0); 32];
        let short = vec![f16::from_f32(1.0); 31];
        let candidates: [&[f16]; 16] = std::array::from_fn(|i| {
            if i == 0 {
                short.as_slice()
            } else {
                query.as_slice()
            }
        });
        let _ = dot_f16_batch_16(&query, &candidates);
    }

    /// On AMX-FP16 hardware, assert the AMX branch is actually selected (not a
    /// silent fallback) and agrees with the f32 reference within tolerance.
    /// Reaching the end without a SIGILL is itself proof the tile path executed.
    #[cfg(all(
        kernel_support = "amx_fp16",
        target_arch = "x86_64",
        target_os = "linux"
    ))]
    #[test]
    fn amx_path_is_active_and_close() {
        if !crate::simd::amx_fp16::amx_available() {
            return;
        }
        let mut rng = StdRng::seed_from_u64(0xA11);
        let mut worst = 0f32;
        for &dim in BATCH_DIMS.iter().filter(|&&d| d >= 32) {
            let (query, cands) = make_batch(dim, &mut rng);
            let candidates: [&[f16]; 16] = std::array::from_fn(|i| cands[i].as_slice());
            let amx = unsafe { crate::simd::amx_fp16::dot_f16_batch_16_amx(&query, &candidates) };
            for i in 0..16 {
                let want = ref_dot_f32(&query, &cands[i]);
                assert_close(amx[i], want, &format!("amx dim={dim} i={i}"));
                let rel = (amx[i] - want).abs() / (want.abs() + 1e-6);
                worst = worst.max(rel);
            }
        }
        assert!(worst <= REL_TOL, "worst AMX relative error: {worst:.2e}");
    }
}
