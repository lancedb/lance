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
#define LANES 64
#endif

/// Works on NEON + FP16 or AVX512FP16
//
// Please make sure run "cargo bench --bench norm_l2" on both Apple Silicon and
// X86_64, before you change this function.
_Float16 norm_l2_f16(_Float16* data, size_t dimension) {
  _Float16 sum = 0;
#pragma clang loop unroll_count(4) vectorize_width(LANES)
  for (size_t i = 0; i < dimension; i++) {
    _Float16 v = data[i];
    sum += v * v;
  }

  return sum;
}
