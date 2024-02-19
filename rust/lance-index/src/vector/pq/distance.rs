// Copyright 2024 Lance Developers.
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

use std::cmp::min;

use lance_arrow::FloatToArrayType;
use lance_linalg::distance::{l2_distance_batch, L2};

use super::{num_centroids, utils::get_sub_vector_centroids};

/// Build a Distance Table from the query to each PQ centroid
/// using L2 distance.
pub(super) fn build_distance_table_l2<T: FloatToArrayType>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32>
where
    T::ArrowType: L2,
{
    let dimension = query.len();

    let sub_vector_length = dimension / num_sub_vectors;
    query
        .chunks_exact(sub_vector_length)
        .enumerate()
        .flat_map(|(i, sub_vec)| {
            let subvec_centroids =
                get_sub_vector_centroids(codebook, dimension, num_bits, num_sub_vectors, i);
            l2_distance_batch(sub_vec, subvec_centroids, sub_vector_length)
        })
        .collect()
}

/// Compute L2 distance from the query to all code.
///
/// Type parameters
/// ---------------
/// - C: the tile size of code-book to run at once.
/// - V: the tile size of PQ code to run at once.
///
/// Parameters
/// ----------
/// - distance_table: the pre-computed L2 distance table.
///   It is a flatten array of [num_sub_vectors, num_centroids] f32.
/// - num_bits: the number of bits used for PQ.
/// - num_sub_vectors: the number of sub-vectors.
/// - code: the PQ code to be used to compute the distances.
///
/// Returns
/// -------
///  The squared L2 distance.
///
#[inline]
pub(super) fn compute_l2_distance<const C: usize, const V: usize>(
    distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    let num_centroids = num_centroids(num_bits);

    let iter = code.chunks_exact(num_sub_vectors * V);
    let distances = iter.clone().flat_map(|c| {
        let mut sums = [0.0_f32; V];
        for i in (0..num_sub_vectors).step_by(C) {
            for (vec_idx, sum) in sums.iter_mut().enumerate() {
                let vec_start = vec_idx * num_sub_vectors;
                #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
                {
                    use std::arch::x86_64::*;
                    if i + C <= num_sub_vectors {
                        let mut offsets = [(i * num_centroids) as i32; C];
                        for k in 0..C {
                            offsets[k] += (k * num_centroids) as i32 + c[vec_start + k] as i32;
                        }
                        unsafe {
                            let simd_offsets = _mm512_loadu_epi32(offsets.as_ptr());
                            let v = _mm512_i32gather_ps(
                                simd_offsets,
                                distance_table.as_ptr() as *const u8,
                                4,
                            );
                            *sum += _mm512_reduce_add_ps(v);
                        }
                    } else {
                        let mut s = 0.0;
                        for k in 0..num_sub_vectors - i {
                            *sum +=
                                distance_table[(i + k) * num_centroids + c[vec_start + k] as usize];
                        }
                    }
                }
                #[cfg(not(all(feature = "nightly", target_feature = "avx512f")))]
                {
                    let s = c[vec_start + i..]
                        .iter()
                        .take(min(C, num_sub_vectors - i))
                        .enumerate()
                        .map(|(k, c)| distance_table[(i + k) * num_centroids + *c as usize])
                        .sum::<f32>();
                    *sum += s;
                }
            }
        }
        sums.into_iter()
    });
    // Remainder
    let remainder = iter.remainder().chunks(num_sub_vectors).map(|c| {
        c.iter()
            .enumerate()
            .map(|(sub_vec_idx, code)| distance_table[sub_vec_idx * num_centroids + *code as usize])
            .sum::<f32>()
    });
    distances.chain(remainder).collect()
}
