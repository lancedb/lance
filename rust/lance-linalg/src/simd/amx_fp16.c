// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// AMX-FP16 batched f16 x f16 dot-product kernel for flat (unquantized) fp16
// vector search.
//
// Computes 16 dot products of one IEEE-754 binary16 query vector against 16
// binary16 candidate vectors in a single sequence of AMX tile passes, using
// TDPFP16PS (fp16 x fp16 -> fp32 accumulate). It follows the same tile
// roles/configuration as other AMX kernels and is ported in spirit from
// FAISS PR facebookresearch/faiss#5235's AMX-BF16 kernel: fp16 and bf16 are
// both 2-byte tile elements consumed by the same `_tile_dp*ps` shape, so only
// the instruction (`_tile_dpbf16ps` -> `_tile_dpfp16ps`) and the element ->
// float conversion (bf16 bit-shift -> real IEEE fp16 via F16C `_cvtsh_ss`)
// differ.
//
// Tile layout (one TDPFP16PS pass covers K = 32 fp16 dims):
//   C (tmm0): 16 rows x 1  fp32  -- the 16 results          colsb=4,  rows=16
//   A (tmm1): 16 rows x 32 fp16  -- 16 candidate rows        colsb=64, rows=16
//   B (tmm2): 16 rows x 2  fp16  -- query, VNNI-packed N=1   colsb=4,  rows=16
//
// The C/A/B tile configuration uses colsb = 4/64/4. With N = 1 the query
// needs no VNNI repacking:
// B.row[k].fp16[i] = query[k*2 + i] is just the contiguous query halfwords,
// obtained by loading with a 4-byte row stride. The candidates form the A
// tile's rows, gathered contiguously (16 rows of `stride` halfwords) by the
// Rust caller so the tile load uses one fixed row stride.
//
// For dim > 32 we accumulate across floor(dim/32) tile passes into the same C
// tile. The tail (dim % 32 dims) is computed in scalar fp32 afterwards. The
// result is NOT bit-exact against a sequential scalar loop:
// floating-point tile-order accumulation rounds differently. It matches an
// f32-accumulated reference dot product to within fp16 precision, which is all
// the fp16 distance path requires (see amx_fp16.rs / the Rust-side tests).
//
// SAFETY: executing any AMX tile instruction without first (a) confirming the
// amx-tile + amx-fp16 CPUID bits and (b) obtaining XTILEDATA permission from
// the kernel raises SIGILL. Both are the Rust caller's responsibility (see
// `simd/amx_fp16.rs`); `lance_amx_fp16_request_perm` below performs (b).

// Must precede all includes: exposes the glibc `syscall()` prototype from
// <unistd.h>, which -std=c17 otherwise hides.
#define _GNU_SOURCE

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Ask the kernel to enable AMX tile data state for this process, via
// arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA). Returns 0 on success,
// non-zero otherwise. Requesting XTILEDATA (18) implicitly also grants
// XTILECFG (17). Constants per Linux Documentation/arch/x86/xstate.rst.
//
// XTILEDATA is the single dynamically-enabled XSAVE state component backing the
// physical TMM tile registers, shared by every AMX compute instruction
// (TDPBUUD / TDPBF16PS / TDPFP16PS ...). The syscall is idempotent, so
// requesting an already-granted permission is harmless.
int lance_amx_fp16_request_perm(void) {
#ifdef __linux__
  const unsigned long ARCH_REQ_XCOMP_PERM = 0x1023;
  const unsigned long XFEATURE_XTILEDATA = 18;
  return (int)syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
#else
  return -1;
#endif
}

// The 64-byte tile configuration image loaded by LDTILECFG (`_tile_loadconfig`).
// Layout per Intel SDM: palette_id, start_row, 14 reserved bytes, then a u16
// colsb[16] (bytes-per-row) array and a u8 rows[16] array. Only slots 0..2 are
// meaningful here; the rest must be zero.
typedef struct __attribute__((packed)) {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} lance_amx_fp16_tilecfg;

// out[i] = sum_{d in 0..dim} f32(query[d]) * f32(candidates[i*stride + d]).
//
// `query`, `candidates` -- IEEE-754 binary16 values as raw uint16_t bit
//                          patterns (half::f16 has identical layout).
// `dim`    -- vector dimension (query has `dim` valid halfwords).
// `stride` -- per-row stride, in halfwords, of the gathered candidate buffer;
//             must be >= dim so each of the 16 rows holds `dim` valid halfwords.
// `out`    -- destination for 16 fp32 dot products.
void lance_amx_dot_f16_batch_16(const uint16_t *query, const uint16_t *candidates,
                                size_t dim, size_t stride, float *out) {
  const size_t full = (dim / 32) * 32;  // dims covered by full 32-wide passes

  if (full > 0) {
    lance_amx_fp16_tilecfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    cfg.rows[0] = 16;  cfg.colsb[0] = 4;   // tmm0 = C: 16 x 1 fp32
    cfg.rows[1] = 16;  cfg.colsb[1] = 64;  // tmm1 = A: 16 x 32 fp16 (candidates)
    cfg.rows[2] = 16;  cfg.colsb[2] = 4;   // tmm2 = B: 16 x 2 fp16 (query, N=1)
    // Some GCC versions do not model _tile_loadconfig as reading the 64-byte
    // cfg image and dead-store-eliminate the rows/colsb writes above, loading an
    // all-zero (unconfigured) tile shape -> #UD on the first tile op (documented
    // in FAISS #5235). Force the stores to be observable. Harmless under clang.
    __asm__ volatile("" : : "m"(cfg) : "memory");
    _tile_loadconfig(&cfg);

    const size_t stride_bytes = stride * sizeof(uint16_t);
    _tile_zero(0);
    for (size_t k = 0; k < full; k += 32) {
      _tile_loadd(1, candidates + k, stride_bytes);  // A: 16 cand rows, cols [k,k+32)
      _tile_loadd(2, query + k, 4);                  // B: query[k..k+32], 16 rows x 2 fp16
      _tile_dpfp16ps(0, 1, 2);                       // C += A * B  (fp16 x fp16 -> fp32)
    }
    _tile_stored(0, out, 4);  // 16 fp32, row stride 4 bytes
    _tile_release();
  } else {
    for (int i = 0; i < 16; i++) out[i] = 0.0f;
  }

  // Tail: the dims not covered by a full 32-wide tile, in scalar fp32. F16C
  // `_cvtsh_ss` is an exact (lossless) binary16 -> binary32 widening.
  const size_t tail = dim - full;
  if (tail > 0) {
    for (int i = 0; i < 16; i++) {
      float acc = 0.0f;
      const uint16_t *row = candidates + (size_t)i * stride;
      for (size_t d = full; d < dim; d++) {
        acc += _cvtsh_ss(row[d]) * _cvtsh_ss(query[d]);
      }
      out[i] += acc;
    }
  }
}
