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

use std::sync::Arc;

use arrow_array::{cast::as_primitive_array, Array, FixedSizeListArray, Float32Array};

use crate::{arrow::*, utils::distance::l2_distance};

/// Product Quantization, optimized for [Apache Arrow].
pub struct ProductQuantizer {
    /// Number of bits for the centroids.
    ///
    /// Only support 8, as one of `u8` byte, for now.
    pub nbits: u32,

    /// Number of sub-vectors.
    pub num_sub_vectors: usize,

    /// Vector dimension.
    dimension: u32,

    /// PQ codebook
    ///
    /// ```((2 ^ nbits) * num_subvectors)``` of `f32`
    ///
    /// Use a layout that is cache / SIMD friendly to compute centroid.
    /// But not sure how to make distance lookup via PQ code lookup
    /// be cache friendly tho.
    ///
    /// Layout:
    ///
    ///  - *row*: all centroids for the same sub-vector.
    ///  - *column*: the centroid value of the n-th sub-vector.
    ///
    /// ```text
    /// // Centroids for a sub-vector.
    /// Codebook[sub_vector_id][pq_code]
    /// ```
    pub codebook: Option<Arc<Float32Array>>,
}

impl ProductQuantizer {
    /// Build a Product quantizer with `m` sub-vectors, and `nbits` to present centroids.
    pub fn new(m: u32, nbits: u32, dimension: u32) -> Self {
        assert!(nbits == 8, "nbits can only be 8");
        Self {
            nbits,
            num_sub_vectors: m as usize,
            dimension,
            codebook: None,
        }
    }

    /// Re-construct ProductQuantizer with the centroids.
    pub fn new_with_centroids(
        num_bits: u32,
        num_sub_vectors: u32,
        centroids: Arc<Float32Array>,
    ) -> Self {
        Self {
            nbits: num_bits,
            num_sub_vectors: num_sub_vectors as usize,
            dimension: centroids.len() as u32 / ProductQuantizer::num_centorids(num_bits),
            codebook: Some(centroids),
        }
    }

    pub fn num_centorids(num_bits: u32) -> u32 {
        2_u32.pow(num_bits)
    }

    /// Calculate codebook length.
    pub fn codebook_length(num_bits: u32, num_sub_vectors: u32) -> u32 {
        ProductQuantizer::num_centorids(num_bits) * num_sub_vectors
    }

    /// Get the centroids for one sub-vector.
    pub fn centroids(&self, sub_vector_idx: usize) -> FixedSizeListArray {
        assert!(sub_vector_idx < self.num_sub_vectors as usize);
        assert!(self.codebook.is_some());

        let codebook = self.codebook.as_ref().unwrap();
        let arr = codebook.slice(
            sub_vector_idx * self.dimension as usize,
            self.dimension as usize,
        );
        let f32_arr: &Float32Array = as_primitive_array(&arr);
        FixedSizeListArray::try_new(f32_arr, self.dimension as i32 / self.num_sub_vectors as i32)
            .unwrap()
    }

    /// Search a Residual vector `key`.
    ///
    pub fn search(&self, code: &[u8], key: &Float32Array) -> Arc<Float32Array> {
        assert_eq!(code.len() % self.num_sub_vectors as usize, 0);
        // Build distance table for each sub-centroid to the query key.
        //
        // Distance table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let mut distance_table: Vec<f32> = vec![];

        let sub_vector_length = self.dimension as usize / self.num_sub_vectors;
        for i in 0..self.num_sub_vectors {
            let from = key.slice(i * sub_vector_length, sub_vector_length).clone();
            let subvec_centroids = self.centroids(i);
            let distances = l2_distance(as_primitive_array(&from), &subvec_centroids).unwrap();
            distance_table.extend(distances.iter().map(|d| d.unwrap_or(0.0)));
        }

        let distances_per_code =
            Float32Array::from_iter(code.chunks_exact(self.num_sub_vectors as usize).map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        distance_table[sub_vec_idx * self.num_sub_vectors + *centroid as usize]
                    })
                    .sum::<f32>()
            }));
        Arc::new(distances_per_code)
    }
}
