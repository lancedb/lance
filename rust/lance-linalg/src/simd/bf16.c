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

// See https://github.com/ashvardanian/SimSIMD/blob/main/include/simsimd/spatial.h
// https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html

/// Needs avx512bf16 and avx512bw

float norm_l2_bf16(__bf16 const* a, __bf16 const* b, size_t d) {
    __m512h d_vec = _mm512_set1_ph(0);
    for (size_t i = 0; i < d; i += 32) {
//        __mmask16 mask = d - i >= 32 ? 0xFFFFFFFF : ((1u << (d - i)) - 1u);
        __mmask16 mask = _bzhi_u32(0xFFFFFFFF, d);
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        d_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec));
    }
    return _mm512_reduce_add_ph(d_vec);
}

float dot_bf16(__bf16 const* a, __bf16 const* b, size_t n) {
    __m512h ab_vec = _mm512_set1_ph(0);
    __m512i a_vec, b_vec;

    while (n > 0) {
        if (n < 32) {
//            __mmask16 mask = (1u << n) - 1u;
            __mmask16 mask = _bzhi_u32(0xFFFFFFFF, n);
            a_vec = _mm512_maskz_loadu_epi16(mask, a);
            b_vec = _mm512_maskz_loadu_epi16(mask, b);
            n = 0;
        } else {
            a_vec = _mm512_loadu_epi16(a);
            b_vec = _mm512_loadu_epi16(b);
            a += 32, b += 32, n -= 32;
        }
        ab_vec = _mm512_fmadd_ph(a_vec, b_vec, ab_vec);
    }

    return _mm512_reduce_add_ph(ab_vec);
}

float l2_bf16(__bf16 const* a, __bf16 const* b, size_t d) {
    __m512h d2_vec = _mm512_set1_ph(0);
    for (size_t i = 0; i < d; i += 32) {
//        __mmask16 mask = d - i >= 32 ? 0xFFFFFFFF : ((1u << (d - i)) - 1u);
        __mmask16 mask = _bzhi_u32(0xFFFFFFFF, d);
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        __m512h d_vec = _mm512_sub_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec));
        d2_vec = _mm512_fmadd_ph(d_vec, d_vec, d2_vec);
    }
    return _mm512_reduce_add_ph(d2_vec);
}

float cosine_bf16(__bf16 const* a, __bf16 const* b, size_t n) {
    __m512h ab_vec = _mm512_set1_ph(0);
    __m512h a2_vec = _mm512_set1_ph(0);
    __m512h b2_vec = _mm512_set1_ph(0);
    __m512i a_vec, b_vec;

    while (n > 0) {
        if (n < 32) {
//            __mmask16 mask = (1u << n) - 1u;
            __mmask16 mask = _bzhi_u32(0xFFFFFFFF, n);
            a_vec = _mm512_maskz_loadu_epi16(mask, a);
            b_vec = _mm512_maskz_loadu_epi16(mask, b);
            n = 0;
        } else {
            a_vec = _mm512_loadu_epi16(a);
            b_vec = _mm512_loadu_epi16(b);
            a += 32, b += 32, n -= 32;
        }
        ab_vec = _mm512_fmadd_ph(a_vec, b_vec, ab_vec);
        a2_vec = _mm512_fmadd_ph(a_vec, a_vec, a2_vec);
        b2_vec = _mm512_fmadd_ph(b_vec, b_vec, b2_vec);
    }

    __bf16 ab = _mm512_reduce_add_ph(ab_vec);
    __bf16 a2 = _mm512_reduce_add_ph(a2_vec);
    __bf16 b2 = _mm512_reduce_add_ph(b2_vec);
    return 1 - ab / (a2 * b2);

    // Compute the reciprocal square roots of a2 and b2
    // __m128 rsqrts = __m128(_mm_set_ps(0.f, 0.f, a2 + 1.e-9f, b2 + 1.e-9f));
    // __m128 rsqrt_a2 = _mm_cvtss_f32(rsqrts);
    // f32_t rsqrt_b2 = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    // return 1 - ab * rsqrt_a2 * rsqrt_b2;
//    return 1;
}
