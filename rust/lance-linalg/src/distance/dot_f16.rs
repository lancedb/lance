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

/// Batched f16 dot product: the raw dot products of one `query` against the
/// first `len` of 16 `candidates`, in order. Every candidate slice must have the
/// same length as `query`, and `len` must be in `1..=16`.
///
/// A batch shorter than 16 is the common case on the beam-search path, so `len`
/// is not a convenience: it is what stops a one-neighbor flush from doing 16
/// candidates' worth of work. Lanes `len..16` are returned as `0` rather than
/// left unspecified, so both the AMX and fallback paths agree exactly on what a
/// caller that reads past `len` would see.
///
/// Safe to call unconditionally: the caller never needs to know whether AMX is
/// present. The function panics if any candidate has a different length from
/// `query`, or if `len` is out of range, rather than allowing a malformed batch
/// to reach the FFI kernel. See the module docs for the accuracy contract.
#[inline]
pub fn dot_f16_batch_16(query: &[f16], candidates: &[&[f16]; 16], len: usize) -> [f32; 16] {
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.len() == query.len()),
        "all candidate vectors must have the same length as query"
    );
    assert!(
        (1..=16).contains(&len),
        "batch length must be in 1..=16, got {len}"
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
            return unsafe { crate::simd::amx_fp16::dot_f16_batch_16_amx(query, candidates, len) };
        }
    }
    dot_f16_batch_16_fallback(query, candidates, len)
}

/// Fallback for [`dot_f16_batch_16`]: `len` independent [`crate::distance::dot()`]
/// calls — bit-identical to the per-neighbor `distance()` path (both go through
/// `f16::dot`) — and `0` for the remaining lanes, matching what the kernel
/// leaves there. Exposed separately so tests can exercise it regardless of host.
#[inline]
pub(crate) fn dot_f16_batch_16_fallback(
    query: &[f16],
    candidates: &[&[f16]; 16],
    len: usize,
) -> [f32; 16] {
    std::array::from_fn(|i| {
        if i < len {
            dot(query, candidates[i])
        } else {
            0.0
        }
    })
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
            let got = dot_f16_batch_16_fallback(&query, &candidates, 16);
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
            let got = dot_f16_batch_16(&query, &candidates, 16);
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
        let _ = dot_f16_batch_16(&query, &candidates, 16);
    }

    /// A live lane must be bit-identical whatever `len` the batch was issued at.
    /// Shortening the batch changes only how many rows are gathered, never a
    /// row's contents nor the order the tile passes accumulate them, so anything
    /// less than bit-exact here would mean a staging offset moved with `len` —
    /// exactly the bug a tolerance would hide. Lanes past `len` must read 0 on
    /// both paths: dispatch is host-dependent, so a caller must not be able to
    /// tell which one ran.
    #[test]
    fn partial_len_matches_full_batch_and_zeroes_the_rest() {
        let mut rng = StdRng::seed_from_u64(0x1EE);
        for &dim in BATCH_DIMS {
            let (query, cands) = make_batch(dim, &mut rng);
            let candidates: [&[f16]; 16] = std::array::from_fn(|i| cands[i].as_slice());
            let full = dot_f16_batch_16(&query, &candidates, 16);
            let full_fb = dot_f16_batch_16_fallback(&query, &candidates, 16);
            for len in 1..=16 {
                let got = dot_f16_batch_16(&query, &candidates, len);
                let got_fb = dot_f16_batch_16_fallback(&query, &candidates, len);
                for i in 0..len {
                    assert_eq!(
                        got[i].to_bits(),
                        full[i].to_bits(),
                        "dim={dim} len={len} i={i}: {} vs {}",
                        got[i],
                        full[i]
                    );
                    assert_eq!(
                        got_fb[i].to_bits(),
                        full_fb[i].to_bits(),
                        "fb dim={dim} i={i}"
                    );
                }
                for i in len..16 {
                    assert_eq!(got[i], 0.0, "dim={dim} len={len}: lane {i} must be 0");
                    assert_eq!(got_fb[i], 0.0, "fb dim={dim} len={len}: lane {i} must be 0");
                }
            }
        }
    }

    /// `len` is rejected, never clamped: the kernel indexes its staging buffer
    /// by it, and a caller that meant 16 and passed 17 should hear about it.
    #[rstest::rstest]
    #[case::zero(0)]
    #[case::seventeen(17)]
    #[should_panic(expected = "batch length must be in 1..=16")]
    fn rejects_out_of_range_len(#[case] len: usize) {
        let query = vec![f16::from_f32(1.0); 32];
        let candidates: [&[f16]; 16] = std::array::from_fn(|_| query.as_slice());
        let _ = dot_f16_batch_16(&query, &candidates, len);
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
            let amx =
                unsafe { crate::simd::amx_fp16::dot_f16_batch_16_amx(&query, &candidates, 16) };
            for i in 0..16 {
                let want = ref_dot_f32(&query, &cands[i]);
                assert_close(amx[i], want, &format!("amx dim={dim} i={i}"));
                let rel = (amx[i] - want).abs() / (want.abs() + 1e-6);
                worst = worst.max(rel);
            }
        }
        assert!(worst <= REL_TOL, "worst AMX relative error: {worst:.2e}");
    }

    /// The batch-16 kernel's tile shape, pinned byte for byte.
    ///
    /// The shape is the one thing a refactor of `amx_fp16.c` can change with no
    /// visible symptom until it runs on AMX hardware, where a wrong shape is a
    /// #UD or wrong results rather than a clean failure. Asserting the
    /// configuration image directly keeps that a check every host can run,
    /// whether or not it has AMX.
    #[cfg(all(
        kernel_support = "amx_fp16",
        target_arch = "x86_64",
        target_os = "linux"
    ))]
    #[test]
    fn search_tile_config_image_is_pinned() {
        use crate::simd::amx_fp16::{AMX_CFG_SEARCH, tilecfg_image};

        // Image layout per Intel SDM: palette_id, start_row, 14 reserved bytes,
        // a u16 colsb[16] array, then a u8 rows[16] array.
        const COLSB: usize = 16;
        const ROWS: usize = 48;

        let mut want = [0u8; 64];
        want[0] = 1; // palette_id
        // C = tmm0: 16 x 1 fp32, fed by three independent (A, B) pairs so three
        // TDPFP16PS can be in flight at once: A = tmm1/3/5, each 16 x 32 fp16;
        // B = tmm2/4/6, each 16 x 2 fp16. tmm7 is left unconfigured.
        for (tmm, colsb) in [
            (0usize, 4u16),
            (1, 64),
            (2, 4),
            (3, 64),
            (4, 4),
            (5, 64),
            (6, 4),
        ] {
            want[COLSB + tmm * 2..COLSB + tmm * 2 + 2].copy_from_slice(&colsb.to_le_bytes());
            want[ROWS + tmm] = 16;
        }

        let got = tilecfg_image(AMX_CFG_SEARCH).expect("search config kind must be known");
        assert_eq!(got, want, "batch-16 tile configuration changed");
    }

    /// An unknown config kind must be rejected rather than answered with a
    /// zeroed image: an all-zero (unconfigured) shape #UDs on the first tile op.
    #[cfg(all(
        kernel_support = "amx_fp16",
        target_arch = "x86_64",
        target_os = "linux"
    ))]
    #[test]
    fn unknown_tile_config_kind_is_rejected() {
        assert!(crate::simd::amx_fp16::tilecfg_image(-1).is_none());
    }
}
