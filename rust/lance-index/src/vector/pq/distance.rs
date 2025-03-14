// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::cmp::min;
use std::collections::BinaryHeap;

use itertools::Itertools;
use lance_linalg::distance::{dot_distance_batch, l2_distance_batch, Dot, L2};
use lance_linalg::simd::u8::u8x16;
use lance_linalg::simd::{Shuffle, SIMD};
use lance_table::utils::LanceIteratorExtension;

use crate::vector::graph::OrderedNode;

use super::{num_centroids, utils::get_sub_vector_centroids};

/// Build a Distance Table from the query to each PQ centroid
/// using L2 distance.
pub fn build_distance_table_l2<T: L2>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    match num_bits {
        4 => build_distance_table_l2_impl::<4, T>(codebook, num_sub_vectors, query),
        8 => build_distance_table_l2_impl::<8, T>(codebook, num_sub_vectors, query),
        _ => panic!("Unsupported number of bits: {}", num_bits),
    }
}

#[inline]
pub fn build_distance_table_l2_impl<const NUM_BITS: u32, T: L2>(
    codebook: &[T],
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    let dimension = query.len();
    let sub_vector_length = dimension / num_sub_vectors;
    let num_centroids = 2_usize.pow(NUM_BITS);
    query
        .chunks_exact(sub_vector_length)
        .enumerate()
        .flat_map(|(i, sub_vec)| {
            let subvec_centroids =
                get_sub_vector_centroids::<NUM_BITS, _>(codebook, dimension, num_sub_vectors, i);
            l2_distance_batch(sub_vec, subvec_centroids, sub_vector_length)
        })
        .exact_size(num_sub_vectors * num_centroids)
        .collect()
}

/// Build a Distance Table from the query to each PQ centroid
/// using Dot distance.
pub fn build_distance_table_dot<T: Dot>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    match num_bits {
        4 => build_distance_table_dot_impl::<4, T>(codebook, num_sub_vectors, query),
        8 => build_distance_table_dot_impl::<8, T>(codebook, num_sub_vectors, query),
        _ => panic!("Unsupported number of bits: {}", num_bits),
    }
}

#[inline]
pub fn build_distance_table_dot_impl<const NUM_BITS: u32, T: Dot>(
    codebook: &[T],
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    let dimension = query.len();
    let sub_vector_length = dimension / num_sub_vectors;
    let num_centroids = 2_usize.pow(NUM_BITS);
    query
        .chunks_exact(sub_vector_length)
        .enumerate()
        .flat_map(|(i, sub_vec)| {
            let subvec_centroids =
                get_sub_vector_centroids::<NUM_BITS, _>(codebook, dimension, num_sub_vectors, i);
            dot_distance_batch(sub_vec, subvec_centroids, sub_vector_length)
        })
        .exact_size(num_sub_vectors * num_centroids)
        .collect()
}

/// Compute L2 distance from the query to all code.
///
/// Parameters
/// ----------
/// - distance_table: the pre-computed L2 distance table.
///   It is a flatten array of [num_sub_vectors, num_centroids] f32.
/// - num_bits: the number of bits used for PQ.
/// - num_sub_vectors: the number of sub-vectors.
/// - code: the transposed PQ code to be used to compute the distances.
///
/// Returns
/// -------
///  The squared L2 distance.
///
#[inline]
pub(super) fn compute_pq_distance(
    distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    if num_bits == 4 {
        return compute_pq_distance_4bit(distance_table, num_sub_vectors, code);
    }
    // here `code` has been transposed,
    // so code[i][j] is the code of i-th sub-vector of the j-th vector,
    // and `code` is a flatten array of [num_sub_vectors, num_vectors] u8,
    // so code[i * num_vectors + j] is the code of i-th sub-vector of the j-th vector.
    let num_vectors = code.len() / num_sub_vectors;
    let mut distances = vec![0.0_f32; num_vectors];
    // it must be 8
    const NUM_CENTROIDS: usize = 2_usize.pow(8);
    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        let dist_table =
            &distance_table[sub_vec_idx * NUM_CENTROIDS..(sub_vec_idx + 1) * NUM_CENTROIDS];
        debug_assert_eq!(vec_indices.len(), distances.len());
        vec_indices
            .iter()
            .zip(distances.iter_mut())
            .for_each(|(&centroid_idx, sum)| {
                *sum += dist_table[centroid_idx as usize];
            });
    }

    distances
}

#[inline]
pub(super) fn topk_pq_distance(
    distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    code: &[u8],
    k: usize,
) -> Vec<OrderedNode> {
    // here `code` has been transposed,
    // so code[i][j] is the code of i-th sub-vector of the j-th vector,
    // and `code` is a flatten array of [num_sub_vectors, num_vectors] u8,
    // so code[i * num_vectors + j] is the code of i-th sub-vector of the j-th vector.

    let num_vectors = code.len() / num_sub_vectors;

    if k >= num_vectors {
        return compute_pq_distance(distance_table, num_bits, num_sub_vectors, code)
            .into_iter()
            .enumerate()
            .map(|(idx, dist)| OrderedNode::new(idx as u32, dist.into()))
            .collect();
    }

    // it must be 8
    const NUM_CENTROIDS: usize = 2_usize.pow(8);

    // compute the first k distances then we can have the threshold
    let mut distances = vec![0.0_f32; num_vectors];
    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        let dist_table =
            &distance_table[sub_vec_idx * NUM_CENTROIDS..(sub_vec_idx + 1) * NUM_CENTROIDS];
        vec_indices
            .iter()
            .take(k)
            .zip(distances.iter_mut())
            .for_each(|(&centroid_idx, sum)| {
                *sum += dist_table[centroid_idx as usize];
            });
    }

    let mut results = BinaryHeap::from_iter(
        distances
            .iter()
            .take(k)
            .enumerate()
            .map(|(id, dist)| OrderedNode::new(id as u32, (*dist).into())),
    );

    // warmup phase
    let mut candidates = (k..num_vectors).collect::<Vec<_>>();
    let mut sub_vec_idx = 0;
    for vec_indices in code.chunks_exact(num_vectors) {
        let dist_table =
            &distance_table[sub_vec_idx * NUM_CENTROIDS..(sub_vec_idx + 1) * NUM_CENTROIDS];
        let left_distances = &mut distances[k..];
        vec_indices[k..]
            .iter()
            .zip(left_distances.iter_mut())
            .for_each(|(&centroid_idx, sum)| {
                *sum += dist_table[centroid_idx as usize];
            });
        sub_vec_idx += 1;

        // try pruning,
        // this is a little bit tricky, but I'd like to update the candidates in-place
        let threshold = results.peek().unwrap().dist.0;
        let mut idx = 0;
        for i in 0..candidates.len() {
            if distances[candidates[i]] < threshold {
                candidates[idx] = candidates[i];
                idx += 1;
            }
        }

        // go prune phase if only 25% of candidates are left
        if candidates.len() < num_vectors / 4 {
            break;
        }
    }

    // prune phase
    for (sub_vec_idx, vec_indices) in code[sub_vec_idx * num_vectors..]
        .chunks_exact(num_vectors)
        .enumerate()
    {
        let dist_table =
            &distance_table[sub_vec_idx * NUM_CENTROIDS..(sub_vec_idx + 1) * NUM_CENTROIDS];
        for i in candidates.iter() {
            let centroid_idx = vec_indices[*i] as usize;
            distances[*i] += dist_table[centroid_idx];
        }
    }

    for id in candidates.into_iter() {
        let threshold = results.peek().unwrap().dist.0;
        if distances[id] < threshold {
            results.pop();
            results.push(OrderedNode::new(id as u32, distances[id].into()));
        }
    }

    results.into_vec()
}

#[inline]
pub(super) fn compute_pq_distance_4bit(
    distance_table: &[f32],
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    let (qmin, qmax, distance_table) = quantize_distance_table(distance_table);
    let num_vectors = code.len() * 2 / num_sub_vectors;
    // store the distances in f32 to avoid overflow
    let mut distances = vec![0.0f32; num_vectors];
    const NUM_CENTROIDS: usize = 2_usize.pow(4);
    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        debug_assert_eq!(vec_indices.len(), distances.len());
        let origin_dist_table = unsafe {
            u8x16::load_unaligned(distance_table.as_ptr().add(sub_vec_idx * 2 * NUM_CENTROIDS))
        };
        let origin_next_dist_table = unsafe {
            u8x16::load_unaligned(
                distance_table
                    .as_ptr()
                    .add((sub_vec_idx * 2 + 1) * NUM_CENTROIDS),
            )
        };
        for i in (0..num_vectors - NUM_CENTROIDS + 1).step_by(NUM_CENTROIDS) {
            let vec_indices = unsafe { u8x16::load_unaligned(vec_indices.as_ptr().add(i)) };
            let distances = &mut distances[i..i + NUM_CENTROIDS];

            // compute current distances
            let current_indices = vec_indices.bit_and(0x0F);
            let dist_table = origin_dist_table;
            let results = dist_table.shuffle(current_indices);
            debug_assert_eq!(dist_table.as_array(), origin_dist_table.as_array());

            // compute next distances
            let next_indices = vec_indices.right_shift::<4>();
            let next_dist_table = origin_next_dist_table;
            let results = results + next_dist_table.shuffle(next_indices);

            results
                .as_array()
                .into_iter()
                .zip(distances.iter_mut())
                .for_each(|(d, sum)| {
                    *sum += d as f32;
                });
        }
        let remainder = num_vectors % NUM_CENTROIDS;
        if remainder > 0 {
            let vec_indices = &vec_indices[num_vectors - remainder..];
            let distances = &mut distances[num_vectors - remainder..];
            let dist_table = &distance_table[sub_vec_idx * 2 * NUM_CENTROIDS..];
            let next_dist_table = &distance_table[(sub_vec_idx * 2 + 1) * NUM_CENTROIDS..];
            for (i, &centroid_idx) in vec_indices.iter().enumerate() {
                let current_idx = centroid_idx & 0xF;
                let next_idx = centroid_idx >> 4;
                distances[i] += dist_table[current_idx as usize] as f32;
                distances[i] += next_dist_table[next_idx as usize] as f32;
            }
        }
    }

    // need to dequantize the distances
    // to make the distances comparable to the others from the other partitions
    distances.iter_mut().for_each(|d| {
        *d = *d * (qmax - qmin) / 255.0 + qmin;
    });
    distances
}

// Quantize the distance table to u8,
// map distance `d` to `(d-qmin) * 255 / (qmax-qmin)`m
// used for only 4bit PQ so num_centroids must be 16
// returns (qmin, qmax, quantized_distance_table)
#[inline]
fn quantize_distance_table(distance_table: &[f32]) -> (f32, f32, Vec<u8>) {
    const NUM_CENTROIDS: usize = 16;
    let qmin = distance_table.iter().cloned().fold(f32::INFINITY, f32::min);
    let qmax = distance_table
        .chunks(NUM_CENTROIDS)
        .tuple_windows()
        .map(|(a, b)| {
            let a_max = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let b_max = b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            a_max + b_max
        })
        .fold(f32::NEG_INFINITY, f32::max);
    let quantized_dist_table = distance_table
        .iter()
        .map(|&d| ((d - qmin) * 255.0 / (qmax - qmin)).ceil() as u8)
        .collect();

    (qmin, qmax, quantized_dist_table)
}

/// Compute L2 distance from the query to all code without transposing the code.
/// for testing only
///
/// Type parameters
/// ---------------
/// - C: the tile size of code-book to run at once.
/// - V: the tile size of PQ code to run at once.
///
#[allow(dead_code)]
fn compute_l2_distance_without_transposing<const C: usize, const V: usize>(
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
                let s = c[vec_start + i..]
                    .iter()
                    .take(min(C, num_sub_vectors - i))
                    .enumerate()
                    .map(|(k, c)| distance_table[(i + k) * num_centroids + *c as usize])
                    .sum::<f32>();
                *sum += s;
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

#[cfg(test)]
mod tests {
    use crate::vector::pq::storage::transpose;

    use super::*;
    use arrow_array::UInt8Array;

    #[test]
    fn test_compute_on_transposed_codes() {
        let num_vectors = 100;
        let num_sub_vectors = 4;
        let num_bits = 8;
        let dimension = 16;
        let codebook =
            Vec::from_iter((0..num_sub_vectors * num_vectors * dimension).map(|v| v as f32));
        let query = Vec::from_iter((0..dimension).map(|v| v as f32));
        let distance_table = build_distance_table_l2(&codebook, num_bits, num_sub_vectors, &query);

        let pq_codes = Vec::from_iter((0..num_vectors * num_sub_vectors).map(|v| v as u8));
        let pq_codes = UInt8Array::from_iter_values(pq_codes);
        let transposed_codes = transpose(&pq_codes, num_vectors, num_sub_vectors);
        let distances = compute_pq_distance(
            &distance_table,
            num_bits,
            num_sub_vectors,
            transposed_codes.values(),
        );
        let expected = compute_l2_distance_without_transposing::<4, 1>(
            &distance_table,
            num_bits,
            num_sub_vectors,
            pq_codes.values(),
        );
        assert_eq!(distances, expected);
    }
}
