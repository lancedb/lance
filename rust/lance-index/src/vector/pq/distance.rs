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

use lance_arrow::ArrowFloatType;
use lance_linalg::distance::{l2_distance_batch, L2};

use super::utils::get_sub_vector_centroids;

pub(super) fn build_distance_table_l2<T: ArrowFloatType + L2>(
    codebook: &[T::Native],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T::Native],
) -> Vec<f32> {
    let dimension = codebook.len();

    let sub_vector_length = query.len() / num_sub_vectors;
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
