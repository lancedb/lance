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

use arrow_array::UInt8Array;
use arrow_array::{
    builder::Float32Builder, cast::as_primitive_array, Array, FixedSizeListArray, Float32Array,
};
use arrow_schema::DataType;

use super::kmeans::train_kmean_model_from_array;
use crate::arrow::*;
use crate::{index::ann::distance::euclidean_distance, Result};

/// Divide a 2D vector in [`FixedSizeListArray`]     to `n` sub-vectors.
///
/// For example, for a `[1024x1M]` matrix, when `n = 8`, this function divides
/// the matrix into  `[128x1M; 8]` vector of matrix.
fn divide_to_subvectors(array: &FixedSizeListArray, n: i32) -> Vec<FixedSizeListArray> {
    assert!(!array.is_empty());

    let sub_vector_length = (array.value_length() / n) as usize;
    let capacity = array.len() * sub_vector_length;
    let mut subarrays = vec![];

    // TODO: very intensive memory copy involved!!! But this is on the write path.
    // TODO: optimize for memory copy later.
    for i in 0..n as usize {
        let mut builder = Float32Builder::with_capacity(capacity);
        for j in 0..array.len() {
            let arr = array.value(j);
            let row: &Float32Array = as_primitive_array(&arr);
            let start = i * sub_vector_length;

            for k in start..start + sub_vector_length {
                builder.append_value(row.value(k));
            }
        }
        let values = builder.finish();
        let sub_array = FixedSizeListArray::try_new(values, sub_vector_length as i32).unwrap();
        subarrays.push(sub_array);
    }
    subarrays
}

/// Product Quantization, optimized for [Apache Arrow].
///
///
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
    pub codebook: Option<Float32Array>,
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

    /// Re-constructed
    pub fn new_with_centroids(
        num_bits: u32,
        num_sub_vectors: u32,
        centroids: &Float32Array,
    ) -> Self {
        Self {
            nbits: num_bits,
            num_sub_vectors: num_sub_vectors as usize,
            dimension: centroids.len() as u32 / ProductQuantizer::num_centorids(num_bits),
            codebook: Some(centroids.clone()),
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
            let distances =
                euclidean_distance(as_primitive_array(&from), &subvec_centroids).unwrap();
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

    /// Transform the vector array to PQ code array.
    fn transform(&self, sub_vectors: &[FixedSizeListArray]) -> FixedSizeListArray {
        assert_eq!(sub_vectors.len(), self.num_sub_vectors as usize);

        let now = std::time::Instant::now();
        let capacity = sub_vectors.len() * sub_vectors[0].len();
        let mut pg_codebook_builder: Vec<u8> = vec![0; capacity];
        for (idx, vec) in sub_vectors.iter().enumerate() {
            // Centroids for sub-vector.
            let centroids = self.centroids(idx);
            for i in 0..vec.len() {
                let value = vec.value(i);
                let vector: &Float32Array = as_primitive_array(value.as_ref());
                let id =
                    argmin(euclidean_distance(vector, &centroids).unwrap().as_ref()).unwrap() as u8;
                pg_codebook_builder[i * self.num_sub_vectors as usize + idx] = id;
            }
        }
        let values = UInt8Array::from_iter(pg_codebook_builder);
        println!("Time on transform PQ code: {}", now.elapsed().as_secs_f32());
        FixedSizeListArray::try_new(values, self.num_sub_vectors as i32).unwrap()
    }

    /// Train a [ProductQuantizer] using an array of vectors.
    pub fn fit_transform(&mut self, data: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        assert!(data.value_length() % self.num_sub_vectors as i32 == 0);
        assert_eq!(data.value_type(), DataType::Float32);
        assert_eq!(data.null_count(), 0);

        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors as i32);
        let num_centorids = 2_u32.pow(self.nbits);
        let dimension = data.value_length() as usize / self.num_sub_vectors;

        let mut codebook_builder =
            Float32Builder::with_capacity(num_centorids as usize * data.value_length() as usize);
        // TODO: parallel training.
        let now = std::time::Instant::now();
        for sub_vec in &sub_vectors {
            // Centroids for one sub vector.
            let values = sub_vec.values();
            let flatten_array: &Float32Array = as_primitive_array(&values);
            let centroids =
                train_kmean_model_from_array(flatten_array, dimension, num_centorids, 100)?;
            // TODO: COPIED COPIED COPIED
            unsafe {
                codebook_builder.append_trusted_len_iter(centroids);
            }
        }
        let pd_centroids = codebook_builder.finish();
        println!("Train PQ code: {}", now.elapsed().as_secs_f32());
        println!("Code: {:?}", pd_centroids);

        self.codebook = Some(pd_centroids);
        Ok(self.transform(&sub_vectors))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow_array::types::Float32Type;

    #[test]
    fn test_divide_to_subvectors() {
        let values = Float32Array::from_iter((0..320).map(|v| v as f32));
        // A [10, 32] array.
        let mat = FixedSizeListArray::try_new(values, 32).unwrap();
        let sub_vectors = divide_to_subvectors(&mat, 4);
        assert_eq!(sub_vectors.len(), 4);
        assert_eq!(sub_vectors[0].len(), 10);
        assert_eq!(sub_vectors[0].value_length(), 8);

        assert_eq!(
            sub_vectors[0],
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                (0..10)
                    .map(|i| {
                        Some(
                            (i * 32..i * 32 + 8)
                                .map(|v| Some(v as f32))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
                8
            )
        );
    }

    #[test]
    fn test_fit_and_transfor() {}
}
