// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AMX-FP16 accelerated batched f16 x f16 dot product for flat fp16 search.
//!
//! The tile math lives in `amx_fp16.c` (compiled by `build.rs` with a compiler
//! new enough for `-mamx-fp16`, which sets `kernel_support = "amx_fp16"`). This
//! module holds the FFI declarations and the runtime safety gate. The safe
//! public entry point is [`crate::distance::dot_f16::dot_f16_batch_16`].
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

    /// out[i] = sum_d f32(query[d]) * f32(candidates[i*stride + d]), i in 0..16.
    /// `query` / `candidates` are IEEE binary16 bit patterns; `stride >= dim`
    /// (in halfwords); `out` holds 16 f32. See `amx_fp16.c`.
    fn lance_amx_dot_f16_batch_16(
        query: *const u16,
        candidates: *const u16,
        dim: usize,
        stride: usize,
        out: *mut f32,
    );
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

/// Batched AMX-FP16 dot product. Gathers the 16 (non-contiguous) candidate
/// slices into a contiguous `16 x dim` scratch buffer so the tile load can use
/// a single fixed row stride, then runs the tile kernel. Returns the 16 raw
/// dot products (`Σ query·candidate`, no `1.0 -` distance wrapping).
///
/// The gather scratch is a reused thread-local buffer, not a fresh allocation
/// per call: this runs on the hot search path (once per 16-neighbor beam-search
/// flush), where a per-batch heap allocation (48 KB at dim 1536) would churn the
/// allocator and erode the tile speedup, especially at high query concurrency.
///
/// # Safety
/// `amx_available` must have returned `true`. Every candidate slice must have
/// length `query.len()`.
#[cfg(all(
    kernel_support = "amx_fp16",
    target_arch = "x86_64",
    target_os = "linux"
))]
pub(crate) unsafe fn dot_f16_batch_16_amx(query: &[f16], candidates: &[&[f16]; 16]) -> [f32; 16] {
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
    }
    let dim = query.len();
    SCRATCH.with(|cell| {
        let mut gathered = cell.borrow_mut();
        gathered.clear();
        gathered.reserve(16 * dim);
        for (i, cand) in candidates.iter().enumerate() {
            debug_assert_eq!(
                cand.len(),
                dim,
                "candidate {i} length must equal query length"
            );
            gathered.extend_from_slice(cand);
        }
        let mut out = [0f32; 16];
        // half::f16 is #[repr(transparent)] over u16, so the pointer casts below
        // reinterpret the identical IEEE binary16 bit patterns the kernel expects.
        unsafe {
            lance_amx_dot_f16_batch_16(
                query.as_ptr() as *const u16,
                gathered.as_ptr() as *const u16,
                dim,
                dim, // stride: rows packed tightly at `dim` halfwords
                out.as_mut_ptr(),
            );
        }
        out
    })
}
