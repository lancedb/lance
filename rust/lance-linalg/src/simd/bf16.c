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

/// Works on NEON + FP16 or AVX512_BF16
float norm_l2_bf16(const __bf16 *data, uint32_t dimension) {
  __bf16 sum = 0;

#pragma clang loop unroll(enable) vectorize(enable) interleave(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    sum += data[i] * data[i];
  }
  return (float) sum;
}

float l2_bf16(const __bf16 *x, const __bf16 *y, uint32_t dimension) {
  __bf16 sum = 0.0;

#pragma clang loop unroll(enable) interleave(enable) vectorize_width(32)
  for (uint32_t i = 0; i < dimension; i++) {
    __bf16 s = x[i] - y[i];
    sum += s * s;
  }
  return (float) sum;
}