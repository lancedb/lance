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

_Float16 neon_norm_l2_f16(_Float16* data, size_t dimension) {

    _Float16 sum0 = 0;
    _Float16 sum1 = 0;
    _Float16 sum2 = 0;
    _Float16 sum3 = 0;

    #pragma clang loop unroll(enable) vectorize(enable) interleave(enable)
    for (size_t i = 0; i < dimension; i +=4) {
            _Float16 v = data[i];
            sum0 += v * v;
            sum1 += data[i + 1] * data[i + 1];
            sum2 += data[i  + 2] * data[i + 2];
            sum3 += data[i + 3] * data[i + 3];
    }
    for (size_t i = dimension / 4 * 4; i < dimension; i++) {
        sum0 += data[i] * data[i];
    }
    return sum1 + sum2 + sum0 + sum3;
}
