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

use arrow_array::{
    builder::Float32Builder, cast::as_primitive_array, Array, FixedSizeListArray, Float32Array,
    RecordBatch,
};
use arrow_array::{ArrayRef, UInt64Array, UInt8Array};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::take::take;
use futures::stream::{self, StreamExt, TryStreamExt};
use rand::SeedableRng;

use crate::index::pb;
use crate::index::vector::kmeans::train_kmeans;
use crate::io::object_reader::{read_fixed_stride_array, ObjectReader};
use crate::Result;
use crate::{arrow::*, utils::distance::l2_distance};

/// Product Quantization Index.
///
pub struct PQIndex<'a> {
    /// Number of bits for the centroids.
    ///
    /// Only support 8, as one of `u8` byte now.
    pub nbits: u32,

    /// Number of sub-vectors.
    pub num_sub_vectors: usize,

    /// Vector dimension.
    pub dimension: usize,

    /// Product quantizer.
    pub pq: &'a ProductQuantizer,

    /// PQ code
    pub code: Arc<UInt8Array>,

    /// ROW Id used to refer to the actual row in dataset.
    pub row_ids: Arc<UInt64Array>,
}

impl<'a> PQIndex<'a> {
    /// Load a PQ index (page) from the disk.
    pub async fn load(
        reader: &dyn ObjectReader,
        pq: &'a ProductQuantizer,
        offset: usize,
        length: usize,
    ) -> Result<PQIndex<'a>> {
        let pq_code_length = pq.num_sub_vectors * length;
        let pq_code =
            read_fixed_stride_array(reader, &DataType::UInt8, offset, pq_code_length, ..).await?;

        let row_id_offset = offset + pq_code_length /* *1 */;
        let row_ids =
            read_fixed_stride_array(reader, &DataType::UInt64, row_id_offset, length, ..).await?;

        Ok(Self {
            nbits: pq.num_bits,
            num_sub_vectors: pq.num_sub_vectors,
            dimension: pq.dimension,
            code: Arc::new(as_primitive_array(&pq_code).clone()),
            row_ids: Arc::new(as_primitive_array(&row_ids).clone()),
            pq,
        })
    }

    /// Search top-k nearest neighbors for `key` within one PQ partition.
    ///
    pub fn search(&self, key: &Float32Array, k: usize) -> Result<RecordBatch> {
        assert_eq!(self.code.len() % self.num_sub_vectors, 0);

        // Build distance table for each sub-centroid to the query key.
        //
        // Distance table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let mut distance_table: Vec<f32> = vec![];

        let sub_vector_length = self.dimension / self.num_sub_vectors;
        for i in 0..self.num_sub_vectors {
            let from = key.slice(i * sub_vector_length, sub_vector_length);
            let subvec_centroids = self.pq.centroids(i);
            let distances = l2_distance(as_primitive_array(&from), &subvec_centroids)?;
            distance_table.extend(distances.values());
        }

        let scores = Arc::new(Float32Array::from_iter(
            self.code
                .values()
                .chunks_exact(self.num_sub_vectors as usize)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .map(|(sub_vec_idx, centroid)| {
                            distance_table[sub_vec_idx * 256 + *centroid as usize]
                        })
                        .sum::<f32>()
                }),
        )) as ArrayRef;
        let indices = sort_to_indices(&scores, None, Some(k))?;
        let scores = take(&scores, &indices, None)?;
        let row_ids = take(self.row_ids.as_ref(), &indices, None)?;

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("score", DataType::Float32, false),
            ArrowField::new("_rowid", DataType::UInt64, false),
        ]));
        Ok(RecordBatch::try_new(schema, vec![scores, row_ids])?)
    }
}

/// Product Quantization, optimized for [Apache Arrow] buffer memory layout.
///
#[derive(Debug)]
pub struct ProductQuantizer {
    /// Number of bits for the centroids.
    ///
    /// Only support 8, as one of `u8` byte now.
    pub num_bits: u32,

    /// Number of sub-vectors.
    pub num_sub_vectors: usize,

    /// Vector dimension.
    dimension: usize,

    /// PQ codebook
    ///
    /// ```((2 ^ nbits) * num_subvector * sub_vector_length)``` of `f32`
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
    pub fn new(m: usize, nbits: u32, dimension: usize) -> Self {
        assert!(nbits == 8, "nbits can only be 8");
        Self {
            num_bits: nbits,
            num_sub_vectors: m,
            dimension,
            codebook: None,
        }
    }

    pub fn num_centroids(num_bits: u32) -> usize {
        2_usize.pow(num_bits)
    }

    /// Calculate codebook length.
    pub fn codebook_length(num_bits: u32, num_sub_vectors: usize) -> usize {
        ProductQuantizer::num_centroids(num_bits) * num_sub_vectors
    }

    /// Get the centroids for one sub-vector.
    pub fn centroids(&self, sub_vector_idx: usize) -> Arc<FixedSizeListArray> {
        assert!(sub_vector_idx < self.num_sub_vectors as usize);
        assert!(self.codebook.is_some());

        let num_centroids = ProductQuantizer::num_centroids(self.num_bits);
        let sub_vector_width = self.dimension / self.num_sub_vectors;
        let codebook = self.codebook.as_ref().unwrap();
        let arr = codebook.slice(
            sub_vector_idx * num_centroids * sub_vector_width as usize,
            num_centroids * sub_vector_width as usize,
        );
        let f32_arr: &Float32Array = as_primitive_array(&arr);
        Arc::new(FixedSizeListArray::try_new(f32_arr, sub_vector_width as i32).unwrap())
    }

    /// Transform the vector array to PQ code array.
    async fn transform(
        &self,
        sub_vectors: &[Arc<FixedSizeListArray>],
    ) -> Result<FixedSizeListArray> {
        assert_eq!(sub_vectors.len(), self.num_sub_vectors as usize);

        let vectors = sub_vectors.iter().map(|v| v.clone()).collect::<Vec<_>>();
        let all_centroids = (0..sub_vectors.len())
            .map(|idx| self.centroids(idx))
            .collect::<Vec<_>>();
        let pq_code = stream::iter(vectors)
            .zip(stream::iter(all_centroids))
            .map(|(vec, centroid)| async move {
                tokio::task::spawn_blocking(move || {
                    // TODO Use tiling to improve cache efficiency.
                    (0..vec.len())
                        .map(|i| {
                            let value = vec.value(i);
                            let vector: &Float32Array = as_primitive_array(value.as_ref());
                            let id =
                                argmin(l2_distance(vector, centroid.as_ref()).unwrap().as_ref())
                                    .unwrap() as u8;
                            id
                        })
                        .collect::<Vec<_>>()
                })
                .await
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;

        // Need to transpose pq_code to column oriented.
        let capacity = sub_vectors.len() * sub_vectors[0].len();
        let mut pq_codebook_builder: Vec<u8> = vec![0; capacity];
        for i in 0..pq_code.len() {
            let vec = pq_code[i].as_slice();
            for j in 0..vec.len() {
                pq_codebook_builder[j * self.num_sub_vectors as usize + i] = vec[j];
            }
        }

        let values = UInt8Array::from_iter_values(pq_codebook_builder);
        FixedSizeListArray::try_new(values, self.num_sub_vectors as i32)
    }

    /// Train a [ProductQuantizer] using an array of vectors.
    pub async fn fit_transform(&mut self, data: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        assert!(data.value_length() % self.num_sub_vectors as i32 == 0);
        assert_eq!(data.value_type(), DataType::Float32);
        assert_eq!(data.null_count(), 0);

        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors as i32);
        let num_centorids = 2_u32.pow(self.num_bits);
        let dimension = data.value_length() as usize / self.num_sub_vectors;

        let mut codebook_builder =
            Float32Builder::with_capacity(num_centorids as usize * data.value_length() as usize);
        let rng = rand::rngs::SmallRng::from_entropy();

        // TODO: parallel training.
        for sub_vec in &sub_vectors {
            // Centroids for one sub vector.
            let values = sub_vec.values();
            let flatten_array: &Float32Array = as_primitive_array(&values);
            let centroids =
                train_kmeans(flatten_array, dimension, num_centorids, 25, rng.clone()).await?;
            // TODO: COPIED COPIED COPIED
            unsafe {
                codebook_builder.append_trusted_len_iter(centroids.values().iter().copied());
            }
        }
        let pd_centroids = codebook_builder.finish();
        self.codebook = Some(Arc::new(pd_centroids));
        self.transform(&sub_vectors).await
    }
}

impl From<&pb::Pq> for ProductQuantizer {
    fn from(proto: &pb::Pq) -> Self {
        Self {
            num_bits: proto.num_bits,
            num_sub_vectors: proto.num_sub_vectors as usize,
            dimension: proto.dimension as usize,
            codebook: Some(Arc::new(Float32Array::from_iter_values(
                proto.codebook.iter().copied(),
            ))),
        }
    }
}

impl From<&ProductQuantizer> for pb::Pq {
    fn from(pq: &ProductQuantizer) -> Self {
        Self {
            num_bits: pq.num_bits,
            num_sub_vectors: pq.num_sub_vectors as u32,
            dimension: pq.dimension as u32,
            codebook: pq.codebook.as_ref().unwrap().values().to_vec(),
        }
    }
}

/// Divide a 2D vector in [`FixedSizeListArray`] to `m` sub-vectors.
///
/// For example, for a `[1024x1M]` matrix, when `n = 8`, this function divides
/// the matrix into  `[128x1M; 8]` vector of matrix.
fn divide_to_subvectors(array: &FixedSizeListArray, m: i32) -> Vec<Arc<FixedSizeListArray>> {
    assert!(!array.is_empty());

    let sub_vector_length = (array.value_length() / m) as usize;
    let capacity = array.len() * sub_vector_length;
    let mut subarrays = vec![];

    // TODO: very intensive memory copy involved!!! But this is on the write path.
    // Optimize for memory copy later.
    for i in 0..m as usize {
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
        let sub_array =
            Arc::new(FixedSizeListArray::try_new(values, sub_vector_length as i32).unwrap());
        subarrays.push(sub_array);
    }
    subarrays
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
            sub_vectors[0].as_ref(),
            &FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
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
}
