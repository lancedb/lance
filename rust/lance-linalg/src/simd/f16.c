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

#ifndef LANES
#define LANES 8
#endif

/// Works on NEON + FP16 or AVX512FP16
_Float16 norm_l2_f16(_Float16* data, size_t dimension) {
  _Float16 sums[LANES];
#pragma clang loop unroll(enable) vectorize(enable)
  for (size_t i = 0; i < LANES; i++) {
    sums[i] = 0;
  }

#pragma clang loop unroll(enable) vectorize(enable) interleave(enable)
  for (size_t i = 0; i < dimension; i += LANES) {
    for (size_t j = 0; j < LANES; j++) {
      sums[j] += data[i + j] * data[i + j];
    }
  }

  size_t remaining_start = dimension / (4 * LANES) * LANES * 4;
  if (remaining_start < dimension) {
    // [[unlikey]]
    for (size_t i = dimension / (4 * LANES) * LANES * 4; i < dimension; i++) {
      sums[0] += data[i] * data[i];
    }
  }

  _Float16 sum = 0;
#pragma clang loop vectorize(enable)
  for (size_t i = 0; i < LANES; i++) {
    sum += sums[i];
  }
  return sum;
}
