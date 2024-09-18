// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#include <stddef.h>
#include <stdint.h>
#include <math.h>

// Because we might be compiling this library multiple times, we need to
// add a suffix to each of the function names.
#define FUNC_CAT_INNER(A, B) A##B
#define FUNC_CAT(A, B) FUNC_CAT_INNER(A, B)
#define FUNC(N) FUNC_CAT(N, SUFFIX)

// TODO: I wonder if we could re-purpose this macro to compile bf16 kernels?
#if defined(__clang__)
#if __FLT16_MANT_DIG__
// Clang supports _Float16
#define FP16 _Float16
#else
#define FP16 __fp16
#endif
#elif defined(__GNUC__) || defined(__GNUG__)
#define FP16 _Float16
#endif
// Note: MSVC doesn't support _Float16 yet, so we can't use it here.

float FUNC(norm_l2_f16)(const FP16 *data, uint32_t dimension) {
  float sum = 0;

#pragma clang loop unroll(enable) vectorize(enable) interleave(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    sum += (float) data[i] * (float) data[i];
  }
  return sqrtf(sum);
}

/// @brief Dot product of two f16 vectors.
/// @param x A f16 vector
/// @param y A f16 vector
/// @param dimension The dimension of the vectors
/// @return The dot product of the two vectors.
float FUNC(dot_f16)(const FP16 *x, const FP16 *y, uint32_t dimension) {
  float sum = 0;

#pragma clang loop unroll(enable) interleave(enable) vectorize(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    // Use float32 as the accumulator to avoid overflow.
    sum += x[i] * y[i];
  }
  return sum;
}

float FUNC(l2_f16)(const FP16 *x, const FP16 *y, uint32_t dimension) {
  float sum = 0;

#pragma clang loop unroll(enable) interleave(enable) vectorize(enable)
  for (uint32_t i = 0; i < dimension; i++) {
#if defined(__aarch64__)
    // on aarch64 with fp16, this is 2x faster.
    FP16 sub = x[i] - y[i];
#else
    float sub = x[i] - y[i];
#endif
    // Use float32 as the accumulator to avoid overflow.
    sum += sub * sub;
  }
  return sum;
}

float FUNC(cosine_f16)(const FP16 *x, float x_norm, const FP16 *y, uint32_t dimension) {
  float dot = 0.0;
  float l2_y = 0.0;

  // Instead of using functions above, we combine the loop to reduce overhead
  // of the fp16 to fp32 conversion.
#pragma clang loop unroll(enable) interleave(enable) vectorize(enable)
  for (uint32_t i = 0; i < dimension; i++) {
    // FP16 y_i = y[i];
    dot += x[i] * y[i];
    l2_y += y[i] * y[i];
  }

  return 1.0 - dot / (x_norm * sqrtf(l2_y));
}
