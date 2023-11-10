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

#include <immintrin.h>

/// Works on NEON + FP16 or AVX512FP16
_Float16 norm_l2_f16(const _Float16* data, uint32_t dimension) {
  _Float16 sum = 0;

#pragma clang loop unroll(enable) vectorize(enable) interleave(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    sum += data[i] * data[i];
  }
  return sum;
}

/// @brief Dot product of two f16 vectors.
/// @param x A f16 vector
/// @param y A f16 vector
/// @param dimension The dimension of the vectors
/// @return The dot product of the two vectors.
_Float16 dot_f16(const _Float16* x, const _Float16* y, uint32_t dimension) {
  _Float16 sum = 0;

#pragma clang loop unroll(enable) interleave(enable) vectorize(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    sum += x[i] * y[i];
  }
  return sum;
}


_Float16 l2_f16(const _Float16* x, const _Float16* y, uint32_t dimension) {
// #if defined(__AVX512F__) && defined(__AVX512BF16__)
// __m512h x2 = _mm512_set_ph(0.0);
// __m512h y2 = _mm512_set_ph(0.0);
__m512h xy = _mm512_set1_ph(0.0);

  for (uint32_t i = 0; i < dimension / 32 * 32; i += 32) {
    // _Float16 s = x[i] - y[i];
    // xy += s * s;
    __m512h xi = _mm512_loadu_ph(x + i);
    __m512h yi = _mm512_loadu_ph(y + i);
    __m512h s = _mm512_sub_ph(xi, yi);
    __m512h s2 = _mm512_mul_ph(s, s);
    xy = _mm512_add_ph(xy, s2);
    // __m512i xy_vec = _mm512_dpbf16_ps(x_vec, y_vec, 0xFF);

  }

  return _mm512_reduce_add_ph(xy);
}
