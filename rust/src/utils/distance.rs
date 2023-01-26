// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Compute distance
//!

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array};

use arrow_schema::DataType;

use crate::Result;

// TODO: wait [std::simd] to be stable to replace manually written AVX/FMA code.
//
// `from` and `to` must have the same length.
#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn euclidean_distance_fma(from: &[f32], to: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(from.len(), to.len());

    let len = from.len();
    let mut sums = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        // Cache line-aligned
        let left = _mm256_load_ps(from.as_ptr().add(i));
        let right = _mm256_load_ps(to.as_ptr().add(i));
        let sub = _mm256_sub_ps(left, right);
        // sum = sub * sub + sum
        sums = _mm256_fmadd_ps(sub, sub, sums);
    }
    // Shift and add vector, until only 1 value left.
    for _ in 0..3 {
        let shifted = _mm256_permute2f128_ps(sums, sums, 81);
        sums = _mm256_add_ps(sums, shifted);
    }
    let mut results: [f32; 8] = [0f32; 8];
    _mm256_storeu_ps(results.as_mut_ptr(), sums);
    results[7]
}

/// Calculate L2 distance directly using Arrow compute kernels.
///
#[inline]
pub fn l2_distance_arrow(from: &Float32Array, to: &Float32Array) -> f32 {
    let a = from.values();
    let b = to.values();
    let mut d = 0.0;
    // Better chance to auto-vectorization.
    let l = a.len();
    for i in 0..l {
        let s = a[i] - b[i];
        d += s * s;
    }
    d
}

#[cfg(any(target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_distance_neon(from: &[f32], to: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = from.len();
    let buf = [0.0_f32; 4];
    let mut sum = vld1q_f32(buf.as_ptr());
    for i in (0..len).step_by(4) {
        let left = vld1q_f32(from.as_ptr().add(i));
        let right = vld1q_f32(to.as_ptr().add(i));
        let sub = vsubq_f32(left, right);
        sum = vfmaq_laneq_f32(sum, sub, sub, 1);
    }
    vaddvq_f32(sum)
}

#[cfg(feature = "blas")]
pub fn l2_distance_blas(from: &Float32Array, to: &FixedSizeListArray) -> Result<Arc<Float32Array>> {
    use arrow_array::{cast::as_primitive_array, types::Float32Type};
    use arrow_buffer::MutableBuffer;

    #[cfg(target_os = "macos")]
    use accelerate_src;
    use cblas::*;

    let inner_array = to.values();
    let buffer = as_primitive_array::<Float32Type>(&inner_array).values();
    let dimension = from.len() as i32;
    let from_vector = from.values();

    let scores =
        Float32Array::from_iter_values(buffer.chunks_exact(dimension as usize).map(|t| unsafe {
            // Allocating new buffer is FASTER (12%) than zero out existing buffer.
            let mut buf = MutableBuffer::from_len_zeroed(dimension as usize * 4);
            let result: &mut [f32] = buf.typed_data_mut();
            // Set result buffer to x
            saxpy(dimension, 1.0, from_vector, 1, result, 1);
            // x - y
            saxpy(dimension, -1.0, t, 1, result, 1);
            // (x-y)^2
            sdot(dimension, result, 1, result, 1)
        }));
    Ok(Arc::new(scores))
}

fn l2_distance_simd(from: &Float32Array, to: &FixedSizeListArray) -> Result<Arc<Float32Array>> {
    use arrow_array::{cast::as_primitive_array, types::Float32Type};

    let inner_array = to.values();
    let buffer = as_primitive_array::<Float32Type>(&inner_array).values();
    let dimension = from.len();
    let from_vector = from.values();

    let scores: Float32Array = unsafe {
        Float32Array::from_trusted_len_iter(
            (0..to.len())
                .map(|idx| {
                    #[cfg(any(target_arch = "x86_64"))]
                    {
                        return euclidean_distance_fma(
                            from_vector,
                            &buffer[idx * dimension..(idx + 1) * dimension],
                        );
                    }

                    #[cfg(any(target_arch = "aarch64"))]
                    {
                        return l2_distance_neon(
                            from_vector,
                            &buffer[idx * dimension..(idx + 1) * dimension],
                        );
                    }
                })
                .map(Some),
        )
    };
    Ok(Arc::new(scores))
}

/// Euclidean Distance (L2) from one vector to a list of vectors.
pub fn l2_distance(from: &Float32Array, to: &FixedSizeListArray) -> Result<Arc<Float32Array>> {
    assert_eq!(from.len(), to.value_length() as usize);
    assert_eq!(to.value_type(), DataType::Float32);

    #[cfg(any(target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma") && from.len() % 8 == 0 {
            return l2_distance_simd(from, to);
        }
    }

    // We've found that using Apple Accelerate (BLAS) is faster than Neon SIMD.
    // It might due to the case where MacOS can use AMX coprocessor to compute
    // the vectors, thus there are more CPU cycles for run the rest of the tasks.
    // So if the BLAS feature is enabled, we will prefer to use the BLAS routine.
    #[cfg(feature = "blas")]
    {
        return l2_distance_blas(from, to);
    }

    #[cfg(any(target_arch = "aarch64"))]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") && from.len() % 4 == 0 {
            return l2_distance_simd(from, to);
        }
    }

    // Fallback
    let scores: Float32Array = unsafe {
        Float32Array::from_trusted_len_iter(
            (0..to.len())
                .map(|idx| {
                    let left = to.value(idx);
                    let arr = left.as_any().downcast_ref::<Float32Array>().unwrap();
                    l2_distance_arrow(from, arr)
                })
                .map(Some),
        )
    };
    Ok(Arc::new(scores))
}

#[cfg(test)]
mod tests {

    use super::*;

    use arrow_array::types::Float32Type;

    #[test]
    fn test_euclidean_distance() {
        let mat = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some((0..8).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((1..9).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((2..10).map(|v| Some(v as f32)).collect::<Vec<_>>()),
                Some((3..11).map(|v| Some(v as f32)).collect::<Vec<_>>()),
            ],
            8,
        );
        let point = Float32Array::from((2..10).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let scores = l2_distance(&point, &mat).unwrap();

        assert_eq!(
            scores.as_ref(),
            &Float32Array::from(vec![32.0, 8.0, 0.0, 8.0])
        );
    }

    #[test]
    fn test_odd_length_vector() {
        let mat = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some((0..5).map(|v| Some(v as f32)).collect::<Vec<_>>())],
            5,
        );
        let point = Float32Array::from((2..7).map(|v| Some(v as f32)).collect::<Vec<_>>());
        let scores = l2_distance(&point, &mat).unwrap();

        assert_eq!(scores.as_ref(), &Float32Array::from(vec![20.0]));
    }
}
