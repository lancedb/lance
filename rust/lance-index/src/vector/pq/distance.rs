// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_linalg::distance::{dot_distance_batch, l2_distance_batch, Dot, L2};

use super::{num_centroids, utils::get_sub_vector_centroids};

/// Build a Distance Table from the query to each PQ centroid
/// using L2 distance.
pub(super) fn build_distance_table_l2<T: L2>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
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

/// Build a Distance Table from the query to each PQ centroid
/// using Dot distance.
pub(super) fn build_distance_table_dot<T: Dot>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    let dimension = query.len();
    let sub_vector_length = dimension / num_sub_vectors;
    query
        .chunks_exact(sub_vector_length)
        .enumerate()
        .flat_map(|(i, sub_vec)| {
            let subvec_centroids =
                get_sub_vector_centroids(codebook, dimension, num_bits, num_sub_vectors, i);
            dot_distance_batch(sub_vec, subvec_centroids, sub_vector_length)
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
    // here `code` has been transposed,
    // so code[i][j] is the code of i-th sub-vector of the j-th vector,
    // and `code` is a flatten array of [num_sub_vectors, num_vectors] u8,
    // so code[i * num_vectors + j] is the code of i-th sub-vector of the j-th vector.

    // `distance_table` is a flatten array of [num_sub_vectors, num_centroids] f32,
    let num_vectors = code.len() / num_sub_vectors;
    let mut distances = vec![0.0_f32; num_vectors];
    let num_centroids = num_centroids(num_bits);
    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        let dist_table = &distance_table[sub_vec_idx * num_centroids..];
        vec_indices
            .iter()
            .zip(distances.iter_mut())
            .for_each(|(&centroid_idx, sum)| {
                *sum += dist_table[centroid_idx as usize];
            });
    }

    distances
}
