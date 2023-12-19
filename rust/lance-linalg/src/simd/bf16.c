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
#include <math.h>

// Intel intrinsics docs:
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=bf16&expand=2557&ig_expand=3099,2671,2673

// Also see https://github.com/ashvardanian/SimSIMD/blob/main/include/simsimd/spatial.h
// https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html

/// Needs avx512bf16 and avx512bw

float norm_l2_bf16(__bf16 const* data, size_t n) {
    __m512 distance_vec = _mm512_set1_ps(0);
    __m512bh data_vec;
    __mmask32 mask;

    for (size_t i = 0; i < n; i += 32) {
        mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
        data_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, data + i));
        distance_vec = _mm512_dpbf16_ps(distance_vec, data_vec, data_vec);
    }
    return _mm512_reduce_add_ps(distance_vec);
}

float dot_bf16(__bf16 const* a, __bf16 const* b, size_t n) {
    __m512 ab_vec = _mm512_set1_ps(0);
    __m512bh a_vec, b_vec;
    __mmask32 mask;
    size_t i = 0;

    while (n > 0) {
        if (n < 32) {
            mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
            a_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
            b_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
            n = 0;
        } else {
            a_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
            b_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
            a += 32, b += 32, n -= 32, i += 32;
        }
        ab_vec = _mm512_dpbf16_ps(ab_vec, a_vec, b_vec);
    }

    return _mm512_reduce_add_ps(ab_vec);
}

float l2_bf16(__bf16 const* a, __bf16 const* b, size_t n) {
    __m512 d2_vec = _mm512_set1_ps(0);
    __m512 a_vec, b_vec, d_vec;
    __mmask32 mask;

    for (size_t i = 0; i < n; i += 32) {
        mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
        a_vec = _mm512_cvtpbh_ps(_mm256_castsi256_ph(_mm256_maskz_loadu_epi16(mask, a + i)));
        b_vec = _mm512_cvtpbh_ps(_mm256_castsi256_ph(_mm256_maskz_loadu_epi16(mask, b + i)));
        d_vec = _mm512_sub_ps(a_vec, b_vec);
        d2_vec = _mm512_fmadd_ps(d2_vec, d_vec, d_vec);
    }

    return _mm512_reduce_add_ps(d2_vec); // returns distance squared
}

float cosine_bf16(__bf16 const* a, __bf16 const* b, size_t n) {
    __m512 ab_vec = _mm512_set1_ph(0);
    __m512 a2_vec = _mm512_set1_ph(0);
    __m512 b2_vec = _mm512_set1_ph(0);
    __m512bh a_vec, b_vec;
    __mmask32 mask;
    size_t i = 0;

    while (n > 0) {
        if (n < 32) {
            mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
            a_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
            b_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
            n = 0;
        } else {
            a_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
            b_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
            a += 32, b += 32, n -= 32, i += 32;
        }
        ab_vec = _mm512_dpbf16_ps(ab_vec, a_vec, b_vec);
        a2_vec = _mm512_dpbf16_ps(a2_vec, a_vec, a_vec);
        b2_vec = _mm512_dpbf16_ps(b2_vec, b_vec, b_vec);
    }

    float ab = _mm512_reduce_add_ps(ab_vec);
    float a2 = _mm512_reduce_add_ps(a2_vec);
    float b2 = _mm512_reduce_add_ps(b2_vec);
    return 1 - ab / sqrt(a2 * b2);
}
