// Copyright 2023 Lance Developers.
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

#include <stddef.h>
#include <stdint.h>


#ifdef __ARM_NEON
#define LANES 4
#define UNROLL_COUNT 2
#endif  // __ARM_NEON

#ifdef __x86_64__
#define LANES 64
#define UNROLL_COUNT 4
#endif  // X86_64


/// Works on NEON + FP16 or AVX512FP16
//
// Please make sure run "cargo bench --bench norm_l2" on both Apple Silicon and
// X86_64, before you change this function.

#ifdef __ARM_NEON

_Float16 norm_l2_f16(const _Float16* data, uint32_t dimension) {
  _Float16 vsum[LANES] = {0};

#pragma clang loop unroll_count(UNROLL_COUNT) interleave(enable)
  for (uint32_t i = 0; i < dimension / LANES * LANES; i += LANES) {
#pragma clang loop vectorize(enable) interleave(enable)
    for (uint32_t j = 0; j < LANES; j++) {
      _Float16 v = data[i + j];
      vsum[j] += v * v;
    }
  }

  _Float16 sum = 0;
  #pragma clang loop vectorize(enable) interleave(enable)
  for (size_t i = 0; i < LANES; i++) sum += vsum[i];

  // Remaining
  #pragma clang loop unroll(enable) interleave(enable)
  for (uint32_t i = dimension / LANES * LANES; i < dimension; i++) {
    _Float16 v = data[i];
    sum += v * v;
  }
  return sum;
}

_Float16 dot_f16(const _Float16* x, const _Float16* y, uint32_t dimension) {
    _Float16 vsum[LANES] = {0};

  uint32_t remaining_start = dimension / LANES * LANES;
#pragma clang loop unroll_count(UNROLL_COUNT) interleave(enable)
  for (uint32_t i = 0; i < remaining_start; i += LANES) {
    #pragma clang loop vectorize(enable)
    for (uint32_t j = 0; j < LANES; j++) {
      vsum[j] += x[i + j] * y[i + j];
    }
  }

  _Float16 sum = 0;
  #pragma clang loop vectorize(enable) interleave(enable)
  for (size_t i = 0; i < LANES; i++) sum += vsum[i];

  #pragma clang loop unroll(enable) interleave(enable)
  for (uint32_t i = remaining_start; i < dimension; i++) {
    sum += x[i] * y[i];
  }


  return sum;
}
#endif

#ifdef __x86_64__

_Float16 norm_l2_f16(const _Float16* data, uint32_t dimension) {
  _Float16 sum = 0;

#pragma clang loop unroll_count(UNROLL_COUNT) vectorize_width(LANES) interleave(enable)
  for (uint32_t i = 0; i < dimension; i++) {
     _Float16 v = data[i];
    sum += v * v;
  }
  return sum;
}

#endif  // X86_64
