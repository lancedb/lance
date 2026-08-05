// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// AMX-FP16 tile kernels, and the tile-configuration plumbing they share.
//
// Every kernel here computes fp16 x fp16 dot products with TDPFP16PS
// (fp16 x fp16 -> fp32 accumulate). The tile *shapes* differ per kernel and are
// declared next to each one; the code that turns a shape into a loaded
// LDTILECFG image is shared, because the one subtle step in it (the
// dead-store barrier below) fails silently and only on some compilers, so it
// must exist exactly once.
//
// One kernel:
//   * `lance_amx_dot_f16_batch_16` -- one query against 16 candidates, for flat
//     fp16 HNSW beam search. Ported in spirit from FAISS PR
//     facebookresearch/faiss#5235's AMX-BF16 kernel: fp16 and bf16 are both
//     2-byte tile elements consumed by the same `_tile_dp*ps` shape, so only the
//     instruction (`_tile_dpbf16ps` -> `_tile_dpfp16ps`) and the element ->
//     float conversion (bf16 bit-shift -> real IEEE fp16 via F16C `_cvtsh_ss`)
//     differ. Its C tile is fed by three independent (A, B) tile pairs so three
//     TDPFP16PS can be in flight at once; see the tile roles below.
// ## Tile configuration is cached per thread, not reloaded per call
//
// The kernel's tile *shape* does not depend on its arguments: `dim` only
// changes the loop trip count, so it has exactly one compile-time
// constant shape (`SEARCH_TILES`). LDTILECFG, on the other hand,
// is a full reconfiguration of the tile unit and costs on the order of a few
// hundred cycles -- more than the ~64 cycles of useful tile work a dim-128
// search call performs. Paying it per call would leave the search kernel
// spending most of its time configuring.
//
// So the configuration is loaded lazily and remembered in `t_loaded_cfg`, a
// per-thread record of which shape is currently live on this logical processor.
// `lance_amx_tile_ensure` reloads only when the *other* kernel has since taken
// over the tile unit, which is why a kernel must pass its own
// `LANCE_AMX_CFG_*` kind: the kind is the cache key, and a kernel that shared
// a key with a different shape would run against the wrong one.
//
// The corollary is that the kernel does not execute TILERELEASE on the way out;
// doing so would throw away exactly the state being cached. Tile data therefore
// stays live (non-INIT) on threads that have run a kernel, so their context
// switches carry the 8 KB of XTILEDATA in XSAVE until something releases it.
// That is the intended trade: the search path issues these calls back to back,
// once per beam-search flush. `lance_amx_tile_release` is the explicit teardown
// for a caller that wants the state back -- shutdown, or a test that wants the
// next call to reconfigure.
//
// ## Thread safety
//
// LDTILECFG sets per-logical-processor state. The spec tables below are
// `static const` and therefore shareable, and the 64-byte config *image* is
// always built on the calling thread's stack -- never in shared storage -- so
// two threads running different kernels concurrently cannot corrupt each
// other's configuration. `t_loaded_cfg` is `__thread` for the same reason: it
// mirrors state the kernel owns per thread, and XTILECFG travels with the
// thread across context switches and migrations as part of its XSAVE area.
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

// ---------------------------------------------------------------------------
// XTILEDATA permission
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Shared tile configuration
// ---------------------------------------------------------------------------

// Tile configuration kinds. One per distinct tile shape, and simultaneously the
// cache key `lance_amx_tile_ensure` compares against `t_loaded_cfg` and the
// selector `lance_amx_tilecfg_image` exposes to the Rust tests. Kept in sync
// with the `AMX_CFG_*` constants in `simd/amx_fp16.rs`.
#define LANCE_AMX_CFG_SEARCH 0

// The 64-byte tile configuration image loaded by LDTILECFG
// (`_tile_loadconfig`). Layout per Intel SDM: palette_id, start_row, 14
// reserved bytes, then a u16 colsb[16] (bytes-per-row) array and a u8 rows[16]
// array. Slots not named by a kernel's spec table stay zero.
//
// 64-byte aligned so the configuration load does not straddle a cache line.
typedef struct __attribute__((packed, aligned(64))) {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} lance_amx_tilecfg;

// The image LDTILECFG reads is exactly 64 bytes; a layout change that altered
// the size would silently feed the instruction garbage.
_Static_assert(sizeof(lance_amx_tilecfg) == 64,
               "LDTILECFG image must be exactly 64 bytes");

// One tile register's shape. `tmm` is the register index (0..7), `colsb` its
// bytes per row, `rows` its row count.
typedef struct {
  uint8_t tmm;
  uint8_t rows;
  uint16_t colsb;
} lance_amx_tile_spec;

#define LANCE_AMX_TILE_COUNT(specs) (sizeof(specs) / sizeof((specs)[0]))

// Fill `cfg` with the LDTILECFG image described by `specs`, without loading it.
// Split from the load so `lance_amx_tilecfg_image` can hand the Rust tests the
// exact bytes a kernel would configure.
static void lance_amx_tilecfg_build(lance_amx_tilecfg *cfg,
                                    const lance_amx_tile_spec *specs,
                                    size_t n) {
  memset(cfg, 0, sizeof(*cfg));
  cfg->palette_id = 1;
  for (size_t i = 0; i < n; i++) {
    cfg->rows[specs[i].tmm] = specs[i].rows;
    cfg->colsb[specs[i].tmm] = specs[i].colsb;
  }
}

// Which `LANCE_AMX_CFG_*` shape is live on this thread's tile unit; -1 for
// "none loaded", the state every thread starts in and the one
// `lance_amx_tile_release` restores.
static __thread int t_loaded_cfg = -1;

// Make `cfg_kind` the loaded tile configuration, doing nothing if it already
// is. See the file header for why this is cached rather than done per call.
//
// The barrier is not optional: some GCC versions do not model
// _tile_loadconfig as reading the 64-byte cfg image, dead-store-eliminate the
// rows/colsb writes, and load an all-zero (unconfigured) tile shape -> #UD on
// the first tile op (documented in FAISS #5235). Keeping it here, in the one
// function that owns the image, is why kernels do not build configurations
// themselves. Harmless under clang.
static inline void lance_amx_tile_ensure(int cfg_kind,
                                         const lance_amx_tile_spec *specs,
                                         size_t n) {
  if (t_loaded_cfg == cfg_kind) return;
  lance_amx_tilecfg cfg;
  lance_amx_tilecfg_build(&cfg, specs, n);
  __asm__ volatile("" : : "m"(cfg) : "memory");
  _tile_loadconfig(&cfg);
  t_loaded_cfg = cfg_kind;
}

// Release this thread's tile state and forget the cached configuration, so the
// next kernel call reconfigures from scratch.
//
// The kernels deliberately never do this themselves. It exists for callers that
// need the tile unit back in its INIT state -- a thread parking for a long time
// (tile data is 8 KB of XSAVE state per context switch while it stays live), or
// a test asserting that reconfiguration happens.
void lance_amx_tile_release(void) {
  _tile_release();
  t_loaded_cfg = -1;
}

// ---------------------------------------------------------------------------
// Kernel: batch-16 search (one query x 16 candidates)
// ---------------------------------------------------------------------------

// Tile roles. These are immediate operands of the tile intrinsics, so they must
// be compile-time constants; `#define` rather than `enum` avoids relying on how
// strictly a compiler treats enum constants as immediates.
//
// One TDPFP16PS pass covers K = 32 fp16 dims. With N = 1 the query needs no
// VNNI repacking: B.row[k].fp16[i] = query[k*2 + i] is just the contiguous
// query halfwords, obtained by loading with a 4-byte row stride. The candidates
// form the A tile's rows; they live at unrelated addresses, so this kernel
// stages them a few k-blocks at a time into a fixed-stride scratch buffer that
// the tile load can read (see `lance_amx_stage_rows`).
//
// Three (A, B) pairs, not one. A single pair would make the loop strictly
// serial -- every TDPFP16PS waiting on the two loads that just overwrote its
// own operands -- so the tile unit would idle through each load's latency.
// Three independent pairs let three k-blocks' loads issue before the first
// TDPFP16PS needs its result, which is enough to keep the dp ops back to back.
// Seven tiles is what that costs; tmm7 is left unconfigured.
#define SEARCH_TMM_C 0   // 16 results x 1 fp32
#define SEARCH_TMM_A0 1  // 16 candidate rows x 32 fp16, k-block 3t
#define SEARCH_TMM_B0 2  // query, VNNI-packed at N = 1, k-block 3t
#define SEARCH_TMM_A1 3  // k-block 3t + 1
#define SEARCH_TMM_B1 4
#define SEARCH_TMM_A2 5  // k-block 3t + 2
#define SEARCH_TMM_B2 6

static const lance_amx_tile_spec SEARCH_TILES[] = {
    {SEARCH_TMM_C, 16, 4},   {SEARCH_TMM_A0, 16, 64}, {SEARCH_TMM_B0, 16, 4},
    {SEARCH_TMM_A1, 16, 64}, {SEARCH_TMM_B1, 16, 4},  {SEARCH_TMM_A2, 16, 64},
    {SEARCH_TMM_B2, 16, 4},
};

// Halfwords of one k-block: 32 fp16 dims, the K a single TDPFP16PS covers.
#define SEARCH_K_BLOCK 32

// Bytes one tile row spans in one k-block: 32 fp16, the A tile's full row width.
#define SEARCH_ROW_BYTES (SEARCH_K_BLOCK * (int)sizeof(uint16_t))

// K-blocks gathered per staging step. Three, so one staged buffer feeds exactly
// the three (A, B) pairs the main loop issues together.
#define SEARCH_STAGE_BLOCKS 3
#define SEARCH_STAGE_ROW_BYTES (SEARCH_STAGE_BLOCKS * SEARCH_ROW_BYTES)

// One staging buffer: the 16 rows a tile load always reads, whatever the batch
// actually holds.
#define SEARCH_STAGE_BYTES (16 * SEARCH_STAGE_ROW_BYTES)

// The furthest-reaching tile load is A2: 64 bytes at offset 128 of the last of
// 16 rows, so the last byte it touches is at 128 + 15*192 + 63 and the span it
// covers is exactly SEARCH_STAGE_BYTES. Asserted because an overrun here would
// be a stack smash with no other symptom.
//
// What this actually pins is that the three A tiles tile one staging row with
// no gap and no overlap, i.e. SEARCH_STAGE_BLOCKS == 3; it does not check
// SEARCH_K_BLOCK, nor the 64-byte A-tile width, which is hardcoded separately
// in SEARCH_TILES.
_Static_assert(2 * SEARCH_ROW_BYTES + 15 * SEARCH_STAGE_ROW_BYTES +
                       SEARCH_ROW_BYTES ==
                   SEARCH_STAGE_BYTES,
               "the three A tiles must tile a staging row exactly");

// Gather `row_bytes` starting at k-block `kb` out of each of the first `count`
// candidates into `dst`, one candidate per row.
//
// Rows are SEARCH_STAGE_ROW_BYTES apart even when `row_bytes` is smaller: an A
// tile only reads 64 bytes at its own offset within a row, so a wider stride
// simply leaves the trailing bytes unread. Keeping the stride fixed across
// every staging step is what lets rows [count, 16) be zeroed once per call --
// each step then rewrites the same rows at the same addresses.
//
// The k-blocks a row covers are adjacent inside the candidate vector, so each
// candidate costs exactly one straight-line copy no matter how many k-blocks the
// step covers. `row_bytes` should stay a compile-time constant at every call
// site: a constant size expands to inline wide moves, while a variable one
// becomes a libc `memcpy` call that `-funroll-loops` then multiplies (five PLT
// calls for the one remainder loop, measured under clang-16).
//
// Zeroing the padding rows is deliberately *not* routed through here: it writes
// rows [count, 16) rather than [0, count), and when the rows are full width it
// is one contiguous `memset` rather than a per-row loop.
static inline void lance_amx_stage_rows(uint8_t *dst,
                                        const uint16_t *const *candidates,
                                        size_t count, size_t kb,
                                        size_t row_bytes) {
  for (size_t n = 0; n < count; n++) {
    memcpy(dst + n * SEARCH_STAGE_ROW_BYTES,
           candidates[n] + kb * SEARCH_K_BLOCK, row_bytes);
  }
}

// out[i] = sum_{d in 0..dim} f32(query[d]) * f32(candidates[i][d]), i < count.
//
// `query`      -- IEEE-754 binary16 values as raw uint16_t bit patterns
//                 (half::f16 has identical layout); `dim` valid halfwords.
// `candidates` -- pointers to `dim` halfwords each; only the first `count` are
//                 read. The vectors live wherever the storage put them; this
//                 kernel owns the gather.
// `count`      -- candidates carrying a real neighbor. **Precondition:
//                 1 <= count <= 16**, rejected at the Rust boundary
//                 (`dot_f16_batch_16`) rather than clamped here; a larger value
//                 would run the gather off the end of a staging buffer.
// `dim`        -- vector dimension.
// `out`        -- destination for 16 fp32 dot products. Lanes [count, 16) are
//                 written as 0, not left untouched.
//
// ## Why `count` rather than always 16
//
// The caller pads a partial beam-search flush up to 16 by repeating candidate 0
// (see `flush_batch16!`), and partial flushes are the common case: in
// steady-state beam search most of a node's neighbors are already visited.
// Staging all 16 rows would copy that one vector 16 times -- 32 KB of stores at
// dim 1024 to produce one useful distance. Only rows [0, count) are gathered
// instead, so the copying scales with the neighbors actually found while the
// tile work stays one fixed-cost pass. The padded rows still have to exist,
// since a tile load reads 16 rows unconditionally; they are zeroed once per
// call, and an all-zero A row yields a zero dot product.
//
// ## Why the gather is here and not in the caller
//
// A tile load reads 16 rows at one fixed stride from one base pointer, and the
// 16 candidates of a beam-search flush are 16 unrelated addresses, so their
// bytes have to be brought together somewhere. Doing it in the caller means
// copying 16 * dim * 2 bytes -- 24 KB at dim 768, 32 KB at dim 1024 -- and every
// one of those bytes has to land before the first TDPFP16PS can issue. Against a
// 48 KB L1D that buffer alone is half the cache, and the copy is pure exposed
// latency: nothing overlaps it.
//
// Staging inside the k-block loop instead keeps the working buffer at 3 KB and
// lets the copies for one triple run underneath the tile ops of the previous
// one. The bytes moved are identical; what changes is that they move while the
// tile unit is busy rather than before it starts. (Measured on the caller-side
// version at dim 1024: __memmove 3.7M cycles/query against 0.5M for the scalar
// kernel, IPC 1.70 -> 1.17, and a critical path 18% longer even though total CPU
// work per query was 6.5% lower. The same structure is what
// epeshared/hnswlib-amx uses for its AMX-BF16 kernel.)
//
// For dim > 32 we accumulate across floor(dim/32) tile passes into the same C
// tile, three k-blocks at a time. The tail (dim % 32 dims) is computed in
// scalar fp32 afterwards. The result is NOT bit-exact against a sequential
// scalar loop: floating-point tile-order accumulation rounds differently. It
// matches an f32-accumulated reference dot product to within fp16 precision,
// which is all the fp16 distance path requires (see amx_fp16.rs / the Rust-side
// tests).
void lance_amx_dot_f16_batch_16(const uint16_t *query,
                                const uint16_t *const *candidates, size_t count,
                                size_t dim, float *out) {
  const size_t blocks = dim / SEARCH_K_BLOCK;   // whole 32-wide tile passes
  const size_t full = blocks * SEARCH_K_BLOCK;  // dims they cover

  if (blocks > 0) {
    lance_amx_tile_ensure(LANCE_AMX_CFG_SEARCH, SEARCH_TILES,
                          LANCE_AMX_TILE_COUNT(SEARCH_TILES));

    // Two staging buffers, 16 rows x 192 bytes each. Double buffered so a
    // triple's gather can be issued a whole triple before the tile loads that
    // read it: a tile load cannot take its data from the store buffer, so the
    // gather's stores have to reach L1 first, and with one buffer there is
    // nothing to overlap that drain with. 3 KB each, so both sit in L1D
    // alongside the candidate rows streaming through it.
    __attribute__((aligned(64))) uint8_t stage[2][SEARCH_STAGE_BYTES];

    const size_t triples = blocks / SEARCH_STAGE_BLOCKS;
    const size_t rem = blocks % SEARCH_STAGE_BLOCKS;

    // Rows [count, 16) are padding that a tile load reads but no candidate
    // fills, so they are zeroed here and an all-zero A row then contributes a
    // zero dot product. Once per call is enough: every staging step below
    // writes rows [0, count) only, always at the same row stride, so nothing
    // disturbs the padding again.
    //
    // Only bytes some tile load actually reads are zeroed. This cost is the one
    // part of the call that does not shrink with `dim`, so zeroing the full
    // 2 x 15 x 192 bytes unconditionally would dominate short vectors:
    //   * `stage[1]` is tile-loaded only if the main loop reaches a second
    //     iteration, or the remainder lands on it (an odd `triples`). A single
    //     triple with no remainder never reads it at all.
    //   * A padding row is read to its full width only by the main loop. With
    //     `triples == 0` the remainder is the only reader, and it loads A0 --
    //     plus A1 when `rem == 2` -- so only the first `rem` k-block slots of
    //     each row are ever seen.
    if (count < 16) {
      const size_t pad_off = count * SEARCH_STAGE_ROW_BYTES;
      if (triples > 0) {
        // Full-width padding rows are contiguous: one memset covers them all.
        memset(stage[0] + pad_off, 0, SEARCH_STAGE_BYTES - pad_off);
        if (triples > 1 || rem > 0) {
          memset(stage[1] + pad_off, 0, SEARCH_STAGE_BYTES - pad_off);
        }
      } else {
        for (size_t n = count; n < 16; n++) {
          memset(stage[0] + n * SEARCH_STAGE_ROW_BYTES, 0,
                 rem * (size_t)SEARCH_ROW_BYTES);
        }
      }
    }

    _tile_zero(SEARCH_TMM_C);

    // Take ownership of the destination line now (PREFETCHW), so the RFO
    // overlaps the tile work instead of stalling TILESTORED at the end. The
    // store is 64 bytes issued as one instruction, and waiting on the RFO there
    // backs up the store queue: measured on the FAISS #5235 BF16 kernel as
    // SQ_Full 28.9% of cycles, with XQ.FULL_CYCLES 56x an AVX-512 baseline.
    _mm_prefetch((const char *)out, _MM_HINT_ET0);

    // Prologue of the software pipeline below, and the one gather in the call
    // with no tile work ahead of it to hide behind.
    if (triples > 0) {
      lance_amx_stage_rows(stage[0], candidates, count, 0,
                           SEARCH_STAGE_ROW_BYTES);

      // The in-loop prefetch below runs two triples ahead, so k-blocks
      // [3, 6) -- read by the very first iteration's gather -- would otherwise
      // be touched by nothing. Guarded on the address staying inside the
      // vectors, which also covers a remainder that follows a single triple.
      if (SEARCH_STAGE_BLOCKS < blocks) {
        for (size_t n = 0; n < count; n++) {
          _mm_prefetch(
              (const char *)(candidates[n] +
                             SEARCH_STAGE_BLOCKS * (size_t)SEARCH_K_BLOCK),
              _MM_HINT_T0);
        }
      }
    }

    size_t kb = 0;
    for (size_t t = 0; t < triples; t++, kb += SEARCH_STAGE_BLOCKS) {
      const uint8_t *st = stage[t & 1];

      // Gather the *next* triple before issuing this one's tile ops, into the
      // buffer the previous iteration's tile loads have already consumed. Those
      // stores then have a full triple of tile work to commit to L1 under,
      // instead of the tile load immediately below them waiting on the drain.
      if (t + 1 < triples) {
        // Open the candidate streams the iteration after this one will copy
        // from, so that copy finds them resident. With the rows at 16 unrelated
        // addresses there is no single stride for a hardware prefetcher to
        // latch onto; what it can do is run each candidate forward once that
        // candidate has been touched, and these touches are what start it.
        // (Measured on the FAISS #5235 BF16 kernel without any prefetch: L1D MPI
        // 2.6x and DTLB load MPI 14.6x an AVX-512 baseline.) Guarded so the last
        // iterations stay inside the vectors.
        if (kb + 2 * SEARCH_STAGE_BLOCKS < blocks) {
          const size_t ahead = (kb + 2 * SEARCH_STAGE_BLOCKS) * SEARCH_K_BLOCK;
          for (size_t n = 0; n < count; n++) {
            _mm_prefetch((const char *)(candidates[n] + ahead), _MM_HINT_T0);
          }
        }
        lance_amx_stage_rows(stage[(t + 1) & 1], candidates, count,
                             kb + SEARCH_STAGE_BLOCKS, SEARCH_STAGE_ROW_BYTES);
      }

      // All six loads first: they are independent, so the three dp ops below
      // issue back to back rather than each waiting on its own operands.
      _tile_loadd(SEARCH_TMM_A0, st + 0 * SEARCH_ROW_BYTES,
                  SEARCH_STAGE_ROW_BYTES);
      _tile_loadd(SEARCH_TMM_B0, query + kb * SEARCH_K_BLOCK, 4);
      _tile_loadd(SEARCH_TMM_A1, st + 1 * SEARCH_ROW_BYTES,
                  SEARCH_STAGE_ROW_BYTES);
      _tile_loadd(SEARCH_TMM_B1, query + (kb + 1) * SEARCH_K_BLOCK, 4);
      _tile_loadd(SEARCH_TMM_A2, st + 2 * SEARCH_ROW_BYTES,
                  SEARCH_STAGE_ROW_BYTES);
      _tile_loadd(SEARCH_TMM_B2, query + (kb + 2) * SEARCH_K_BLOCK, 4);

      // C += A * B  (fp16 x fp16 -> fp32)
      _tile_dpfp16ps(SEARCH_TMM_C, SEARCH_TMM_A0, SEARCH_TMM_B0);
      _tile_dpfp16ps(SEARCH_TMM_C, SEARCH_TMM_A1, SEARCH_TMM_B1);
      _tile_dpfp16ps(SEARCH_TMM_C, SEARCH_TMM_A2, SEARCH_TMM_B2);
    }

    // The 1 or 2 k-blocks left over when `blocks` is not a multiple of three.
    // `stage[triples & 1]` is the buffer the loop above neither wrote nor read
    // last, so filling it now cannot collide with a tile load still in flight.
    if (rem > 0) {
      uint8_t *st = stage[triples & 1];
      // Two constant-size cases rather than one `rem * 64` copy, because a
      // variable-length memcpy here does not stay one instruction: clang-16
      // emits a libc call and `-funroll-loops` then multiplies it, measured as
      // five `memcpy@PLT` calls for this one loop (the pre-`count` kernel,
      // whose loop ran to 16, paid sixteen). The main loop's gather is
      // unaffected either way -- it keeps its inline wide moves -- so this is
      // about the remainder alone. The branch costs one predictable compare.
      if (rem == 2) {
        lance_amx_stage_rows(st, candidates, count, kb, 2 * SEARCH_ROW_BYTES);
      } else {
        lance_amx_stage_rows(st, candidates, count, kb, SEARCH_ROW_BYTES);
      }

      // Loaded at the main loop's row stride even though only `rem` k-blocks are
      // live: A0 and A1 read their own 64 bytes at offsets 0 and 64, which is
      // what the staging step just filled, and the padding rows are already zero
      // at exactly this stride.
      _tile_loadd(SEARCH_TMM_A0, st, SEARCH_STAGE_ROW_BYTES);
      _tile_loadd(SEARCH_TMM_B0, query + kb * SEARCH_K_BLOCK, 4);
      if (rem == 2) {
        _tile_loadd(SEARCH_TMM_A1, st + SEARCH_ROW_BYTES,
                    SEARCH_STAGE_ROW_BYTES);
        _tile_loadd(SEARCH_TMM_B1, query + (kb + 1) * SEARCH_K_BLOCK, 4);
      }
      _tile_dpfp16ps(SEARCH_TMM_C, SEARCH_TMM_A0, SEARCH_TMM_B0);
      if (rem == 2) {
        _tile_dpfp16ps(SEARCH_TMM_C, SEARCH_TMM_A1, SEARCH_TMM_B1);
      }
    }

    _tile_stored(SEARCH_TMM_C, out, 4);  // 16 fp32, row stride 4 bytes
  } else {
    for (int i = 0; i < 16; i++) out[i] = 0.0f;
  }

  // Tail: the dims not covered by a full 32-wide tile, in scalar fp32. F16C
  // `_cvtsh_ss` is an exact (lossless) binary16 -> binary32 widening. Lanes at
  // or past `count` are skipped rather than accumulated onto: they are already
  // 0 (zeroed staging rows, or the no-tile-pass branch above) and must stay so.
  const size_t tail = dim - full;
  if (tail > 0) {
    for (size_t i = 0; i < count; i++) {
      float acc = 0.0f;
      const uint16_t *row = candidates[i];
      for (size_t d = full; d < dim; d++) {
        acc += _cvtsh_ss(row[d]) * _cvtsh_ss(query[d]);
      }
      out[i] += acc;
    }
  }
}

// ---------------------------------------------------------------------------
// Configuration introspection
// ---------------------------------------------------------------------------

// Write the 64-byte LDTILECFG image for `cfg_kind` (a `LANCE_AMX_CFG_*`
// constant) into `out`, without loading it. Returns 0 on success, -1 for an
// unknown `cfg_kind`.
//
// Exposed so the Rust tests can pin each kernel's tile shape byte for byte. A
// wrong shape does not fail cleanly — it is a #UD or silently wrong results —
// so the shape is asserted directly rather than inferred from kernel output.
int lance_amx_tilecfg_image(int cfg_kind, uint8_t *out) {
  const lance_amx_tile_spec *specs;
  size_t n;

  switch (cfg_kind) {
    case LANCE_AMX_CFG_SEARCH:
      specs = SEARCH_TILES;
      n = LANCE_AMX_TILE_COUNT(SEARCH_TILES);
      break;
    default:
      return -1;
  }

  lance_amx_tilecfg cfg;
  lance_amx_tilecfg_build(&cfg, specs, n);
  memcpy(out, &cfg, sizeof(cfg));
  return 0;
}
