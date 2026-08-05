// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AMX-FP16 accelerated f16 x f16 dot products.
//!
//! One shape: `dot_f16_batch_16_amx`, one query against 16 candidates, for flat
//! fp16 search. It is named in a plain code span rather than an intra-doc link:
//! it is `kernel_support = "amx_fp16"`-gated, so a link from these unconditional
//! module docs is unresolved — and hence a rustdoc error under `-D warnings` —
//! on any build without the `amx` feature.
//!
//! The tile math lives in `amx_fp16.c` (compiled by `build.rs` with a compiler
//! new enough for `-mamx-fp16`, which sets `kernel_support = "amx_fp16"`). This
//! module holds the FFI declaration and the runtime safety gate. Everything
//! here is crate-internal; the safe public entry point is
//! [`crate::distance::dot_f16::dot_f16_batch_16`], and the availability gate is
//! [`crate::distance::dot_f16::amx_fp16_available`].
//!
//! ## Safety gate
//!
//! A set CPUID bit is not sufficient to run AMX tile instructions on Linux:
//! the OS must first grant the extended (XTILEDATA)
//! state via `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`; skipping
//! that SIGILLs the first tile instruction. `amx_available` requires, and
//! caches process-wide, all of:
//!   1. `target_arch = "x86_64"` and `target_os = "linux"` (compile-time cfg),
//!   2. the amx-tile CPUID bit (leaf 7, sub-leaf 0, EDX bit 24) **and** the
//!      amx-fp16 CPUID bit (leaf 7, sub-leaf 1, EAX bit 21),
//!   3. a successful one-time `arch_prctl` permission request.
//!
//! On any failure the caller falls back to the existing AVX-512-FP16 / scalar
//! `f16::dot` path, so results are unchanged (both accumulate in f32; only the
//! summation order — hence fp16-level rounding — differs).
//!
//! ## XTILEDATA is a shared AMX state permission
//!
//! XTILEDATA (component 18) is the single dynamically-enabled XSAVE state
//! backing the physical TMM tile registers; it is requested per-*state*, not
//! per-*instruction*, so one grant covers every AMX compute instruction — see
//! Linux `Documentation/arch/x86/xstate.rst` ("Dynamically Enabled XSAVE
//! Features", AMX example) and Intel SDM Vol.1 §13.3. The syscall is
//! idempotent, so requesting an already-granted permission again is harmless.

#[cfg(all(
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
use half::f16;

#[cfg(all(
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
unsafe extern "C" {
    /// arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA); 0 on success.
    fn lance_amx_fp16_request_perm() -> i32;

    /// out[i] = sum_d f32(query[d]) * f32(candidates[i][d]), i in 0..count;
    /// out[count..16] = 0. `query` is `dim` IEEE binary16 bit patterns;
    /// `candidates` is 16 pointers to `dim` of them each, of which only the
    /// first `count` (1..=16) are read; `out` holds 16 f32. The kernel gathers
    /// the rows itself, k-block by k-block. See `amx_fp16.c`.
    fn lance_amx_dot_f16_batch_16(
        query: *const u16,
        candidates: *const *const u16,
        count: usize,
        dim: usize,
        out: *mut f32,
    );

    /// Writes the 64-byte LDTILECFG image for `cfg_kind` into `out` without
    /// loading it; 0 on success, -1 for an unknown kind. See `amx_fp16.c`.
    #[cfg(test)]
    fn lance_amx_tilecfg_image(cfg_kind: i32, out: *mut u8) -> i32;
}

/// Config kind for [`tilecfg_image`]: the batch-16 search kernel's tile shape.
/// Must match `LANCE_AMX_CFG_SEARCH` in `amx_fp16.c`.
#[cfg(all(
    test,
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
pub(crate) const AMX_CFG_SEARCH: i32 = 0;

/// The 64-byte LDTILECFG image a kernel would configure, without loading it.
/// `None` if `cfg_kind` is not one of the `AMX_CFG_*` constants.
///
/// Exists for the tests: a wrong tile shape never surfaces as a clean error —
/// it is a #UD or silently wrong results — so the shape is pinned directly
/// rather than inferred from kernel output.
#[cfg(all(
    test,
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
pub(crate) fn tilecfg_image(cfg_kind: i32) -> Option<[u8; 64]> {
    let mut image = [0u8; 64];
    // SAFETY: the C side writes exactly `sizeof(lance_amx_tilecfg)` bytes, which
    // a `_Static_assert` there pins to 64 — the length of `image`.
    let rc = unsafe { lance_amx_tilecfg_image(cfg_kind, image.as_mut_ptr()) };
    (rc == 0).then_some(image)
}

/// True iff AMX-FP16 tile instructions can be executed safely in this process.
/// Evaluated once and cached; the `arch_prctl` permission request (a syscall)
/// happens at most once, process-wide.
///
/// Setting the `LANCE_DISABLE_AMX` environment variable forces this to `false`
/// so the scalar/AVX-512 fallback can be exercised on AMX hardware.
#[cfg(all(
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
pub(crate) fn amx_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        if std::env::var_os("LANCE_DISABLE_AMX").is_some() {
            return false;
        }
        if !detect_amx_fp16() {
            return false;
        }
        // Request XTILEDATA permission; without a 0 return, any tile instruction
        // would SIGILL, so treat anything else as unavailable.
        unsafe { lance_amx_fp16_request_perm() == 0 }
    })
}

/// AMX-TILE = CPUID leaf 7, sub-leaf 0, EDX bit 24. AMX-FP16 = CPUID leaf 7,
/// sub-leaf 1, EAX bit 21. Both required.
#[cfg(all(
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
fn detect_amx_fp16() -> bool {
    use std::arch::x86_64::__cpuid_count;
    // `__cpuid_count` is safe on nightly but `unsafe` on stable; allow both.
    #[allow(unused_unsafe)]
    let leaf7_0 = unsafe { __cpuid_count(7, 0) };
    let amx_tile = (leaf7_0.edx & (1 << 24)) != 0;
    #[allow(unused_unsafe)]
    let leaf7_1 = unsafe { __cpuid_count(7, 1) };
    let amx_fp16 = (leaf7_1.eax & (1 << 21)) != 0;
    amx_tile && amx_fp16
}

/// Batched AMX-FP16 dot product: one query against the first `len` of 16
/// candidates. Returns the 16 raw dot products (`Σ query·candidate`, no `1.0 -`
/// distance wrapping); lanes `len..16` are 0.
///
/// Hands the kernel 16 pointers rather than a packed `16 x dim` buffer. The
/// candidates still have to be brought together for a tile load, but the kernel
/// does it one k-block at a time into 3 KB of stack, which overlaps the copies
/// with the tile ops; packing all `16 * dim * 2` bytes here first could not
/// overlap with anything, and at dim 1024 that is 32 KB against a 48 KB L1D.
/// See the kernel comment in `amx_fp16.c` for the measurements behind this.
///
/// `len` is what keeps a partial batch cheap: the tile pass is a fixed cost for
/// 16 lanes either way, but only `len` rows are gathered.
///
/// # Safety
/// `amx_available` must have returned `true`. `len` must be in `1..=16` — the
/// kernel does not clamp it, and a larger value walks its staging buffer off the
/// end; [`crate::distance::dot_f16::dot_f16_batch_16`] is where that is
/// rejected. Every candidate slice must have length `query.len()`, and must
/// outlive the call.
#[cfg(all(
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
pub(crate) unsafe fn dot_f16_batch_16_amx(
    query: &[f16],
    candidates: &[&[f16]; 16],
    len: usize,
) -> [f32; 16] {
    debug_assert!((1..=16).contains(&len), "len ({len}) must be in 1..=16");
    let dim = query.len();
    // half::f16 is #[repr(transparent)] over u16, so the casts below reinterpret
    // the identical IEEE binary16 bit patterns the kernel expects. All 16 slots
    // are filled even though the kernel reads only `len` of them: the caller
    // already holds 16 valid slices, so there is nothing to gain from leaving
    // the tail of the array undefined.
    let mut rows = [std::ptr::null::<u16>(); 16];
    for (i, cand) in candidates.iter().enumerate() {
        debug_assert_eq!(
            cand.len(),
            dim,
            "candidate {i} length must equal query length"
        );
        rows[i] = cand.as_ptr() as *const u16;
    }
    let mut out = [0f32; 16];
    unsafe {
        lance_amx_dot_f16_batch_16(
            query.as_ptr() as *const u16,
            rows.as_ptr(),
            len,
            dim,
            out.as_mut_ptr(),
        );
    }
    out
}
