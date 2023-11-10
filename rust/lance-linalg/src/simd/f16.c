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

#ifdef __X86_64__
#include <immintrin.h>
#endif // __X86_64__

/// Works on NEON + FP16 or AVX512FP16
float norm_l2_f16(const _Float16 *data, uint32_t dimension) {
  _Float16 sum = 0;

#pragma clang loop unroll(enable) vectorize(enable) interleave(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    sum += data[i] * data[i];
  }
  return (float) sum;
}

/// @brief Dot product of two f16 vectors.
/// @param x A f16 vector
/// @param y A f16 vector
/// @param dimension The dimension of the vectors
/// @return The dot product of the two vectors.
float dot_f16(const _Float16 *x, const _Float16 *y, uint32_t dimension) {
  _Float16 sum = 0;

#pragma clang loop unroll(enable) interleave(enable) vectorize(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    sum += x[i] * y[i];
  }
  return (float) sum;
}

float l2_f16(const _Float16 *x, const _Float16 *y, uint32_t dimension) {
  _Float16 sum = 0.0;

#pragma clang loop unroll(enable) interleave(enable) vectorize_width(32)
  for (uint32_t i = 0; i < dimension; i++) {
    _Float16 s = x[i] - y[i];
    sum += s * s;
  }
  return (float) sum;
}
