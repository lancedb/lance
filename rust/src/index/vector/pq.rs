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

use std::sync::Arc;

use arrow_arith::aggregate::min;
use arrow_array::{
    builder::Float32Builder, cast::as_primitive_array, Array, ArrayRef, FixedSizeListArray,
    Float32Array, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::take::take;
use futures::{stream, StreamExt, TryStreamExt};
use rand::SeedableRng;

use super::MetricType;
use crate::arrow::linalg::MatrixView;
use crate::arrow::*;
use crate::index::{pb, vector::kmeans::train_kmeans};
use crate::io::object_reader::{read_fixed_stride_array, ObjectReader};
use crate::utils::distance::compute::normalize;
use crate::utils::distance::l2::l2_distance;
use crate::{Error, Result};

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

    metric_type: MetricType,
}

impl<'a> PQIndex<'a> {
    /// Load a PQ index (page) from the disk.
    pub async fn load(
        reader: &dyn ObjectReader,
        pq: &'a ProductQuantizer,
        metric_type: MetricType,
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
            metric_type,
        })
    }

    fn fast_l2_scores(&self, key: &Float32Array) -> Result<ArrayRef> {
        // Build distance table for each sub-centroid to the query key.
        //
        // Distance table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let mut distance_table: Vec<f32> = vec![];

        let sub_vector_length = self.dimension / self.num_sub_vectors;
        for i in 0..self.num_sub_vectors {
            let from = key.slice(i * sub_vector_length, sub_vector_length);
            let subvec_centroids = self.pq.centroids(i).unwrap();
            let distances = l2_distance(
                as_primitive_array(&from),
                &subvec_centroids,
                sub_vector_length,
            )?;
            distance_table.extend(distances.values());
        }

        Ok(Arc::new(Float32Array::from_iter(
            self.code
                .values()
                .chunks_exact(self.num_sub_vectors)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .map(|(sub_vec_idx, centroid)| {
                            distance_table[sub_vec_idx * 256 + *centroid as usize]
                        })
                        .sum::<f32>()
                }),
        )))
    }

    fn cosine_scores(&self, key: &Float32Array) -> Result<ArrayRef> {
        // Build two tables for cosine distance.
        //
        // xy table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        // y_norm table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let mut xy_table: Vec<f32> = vec![];
        let mut y_norm_table: Vec<f32> = vec![];

        let x_norm = normalize(key.values()).powi(2);

        let sub_vector_length = self.dimension / self.num_sub_vectors;
        for i in 0..self.num_sub_vectors {
            let slice = key.slice(i * sub_vector_length, sub_vector_length);
            let key_sub_vector: &Float32Array = as_primitive_array(slice.as_ref());
            let sub_vector_centroids = self.pq.centroids(i).unwrap();
            let xy = sub_vector_centroids
                .as_ref()
                .values()
                .chunks_exact(sub_vector_length)
                .map(|cent| {
                    // Accelerate this later.
                    cent.iter()
                        .zip(key_sub_vector.values().iter())
                        .map(|(y, x)| (x - y).powi(2))
                        .sum::<f32>()
                });
            xy_table.extend(xy);

            let y_norm = sub_vector_centroids
                .as_ref()
                .values()
                .chunks_exact(sub_vector_length)
                .map(|cent| {
                    // Accelerate this later.
                    cent.iter().map(|y| y.powi(2)).sum::<f32>()
                });
            y_norm_table.extend(y_norm);
        }

        Ok(Arc::new(Float32Array::from_iter(
            self.code
                .values()
                .chunks_exact(self.num_sub_vectors)
                .map(|c| {
                    let xy = c
                        .iter()
                        .enumerate()
                        .map(|(sub_vec_idx, centroid)| {
                            let idx = sub_vec_idx * 256 + *centroid as usize;
                            xy_table[idx]
                        })
                        .sum::<f32>();
                    let y_norm = c
                        .iter()
                        .enumerate()
                        .map(|(sub_vec_idx, centroid)| {
                            let idx = sub_vec_idx * 256 + *centroid as usize;
                            y_norm_table[idx]
                        })
                        .sum::<f32>();
                    xy / (x_norm.sqrt() * y_norm.sqrt())
                }),
        )))
    }

    /// Search top-k nearest neighbors for `key` within one PQ partition.
    ///
    pub fn search(&self, key: &Float32Array, k: usize) -> Result<RecordBatch> {
        assert_eq!(self.code.len() % self.num_sub_vectors, 0);

        let scores = if self.metric_type == MetricType::L2 {
            self.fast_l2_scores(key)?
        } else {
            self.cosine_scores(key)?
        };

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
    pub dimension: usize,

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
        Self::num_centroids(num_bits) * num_sub_vectors
    }

    /// Get the centroids for one sub-vector.
    ///
    /// Returns a flatten `num_centroids * sub_vector_width` f32 array.
    pub fn centroids(&self, sub_vector_idx: usize) -> Option<Arc<Float32Array>> {
        assert!(sub_vector_idx < self.num_sub_vectors);
        if self.codebook.is_none() {
            return None;
        }

        let num_centroids = Self::num_centroids(self.num_bits);
        let sub_vector_width = self.dimension / self.num_sub_vectors;
        let codebook = self.codebook.as_ref().unwrap();

        let length = num_centroids * sub_vector_width;
        let offset = sub_vector_idx * length;
        let arr = codebook.slice(offset, length);
        Some(Arc::new(as_primitive_array(&arr).clone()))
    }

    /// Reconstruct a vector from its PQ code.
    /// It only supports U8 code for now.
    pub fn reconstruct(&self, code: &[u8]) -> Arc<Float32Array> {
        assert_eq!(code.len(), self.num_sub_vectors);
        let mut builder = Float32Builder::with_capacity(self.dimension);
        let sub_vector_dim = self.dimension / self.num_sub_vectors;
        for i in 0..code.len() {
            let centroids = self.centroids(i).unwrap();
            let sub_code = code[i];
            builder.append_slice(
                &centroids.values()
                    [sub_code as usize * sub_vector_dim..(sub_code as usize + 1) * sub_vector_dim],
            );
        }
        Arc::new(builder.finish())
    }

    /// Compute the quantization distortion (E).
    ///
    /// Quantization distortion is the difference between the centroids
    /// from the PQ code to the actual vector.
    pub async fn distortion(&self, data: &MatrixView, metric_type: MetricType) -> Result<f64> {
        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors);
        debug_assert_eq!(sub_vectors.len(), self.num_sub_vectors);

        let vectors = sub_vectors.to_vec();
        let all_centroids = (0..sub_vectors.len())
            .map(|idx| self.centroids(idx).unwrap())
            .collect::<Vec<_>>();
        let distortion = stream::iter(vectors)
            .zip(stream::iter(all_centroids))
            .map(|(vec, centroid)| async move {
                tokio::task::spawn_blocking(move || {
                    let dist_func = metric_type.func();
                    (0..vec.len())
                        .map(|i| {
                            let value = vec.value(i);
                            let vector: &Float32Array = as_primitive_array(value.as_ref());
                            let distances =
                                dist_func(vector, centroid.as_ref(), vector.len()).unwrap();
                            min(distances.as_ref()).unwrap_or(0.0)
                        })
                        .sum::<f32>() as f64 // in case of overflow
                })
                .await
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?
            .iter()
            .sum::<f64>();
        Ok(distortion / data.num_rows() as f64)
    }

    /// Transform the vector array to PQ code array.
    pub async fn transform(
        &self,
        data: &MatrixView,
        metric_type: MetricType,
    ) -> Result<FixedSizeListArray> {
        let all_centroids = (0..self.num_sub_vectors)
            .map(|idx| self.centroids(idx))
            .collect::<Vec<_>>();
        let dist_func = metric_type.func();

        let flatten_data = data.data();
        let num_sub_vectors = self.num_sub_vectors;
        let dim = self.dimension;
        let num_rows = data.num_rows();
        let values = tokio::task::spawn_blocking(move || {
            let capacity = num_sub_vectors * num_rows;
            let mut builder: Vec<u8> = vec![0; capacity];
            // Dimension of each sub-vector.
            let sub_dim = dim / num_sub_vectors;
            for i in 0..num_rows {
                let row_offset = i * dim;
                for sub_idx in 0..num_sub_vectors {
                    let offset = row_offset + sub_idx * sub_dim;
                    let sub_vector = flatten_data.slice(offset, sub_dim);
                    let centroids = all_centroids[sub_idx].as_ref();
                    let code = argmin(
                        dist_func(as_primitive_array(sub_vector.as_ref()), centroids, sub_dim)?
                            .as_ref(),
                    )
                    .unwrap();
                    builder[i * num_sub_vectors + sub_idx] = code as u8;
                }
            }
            Ok::<UInt8Array, Error>(UInt8Array::from_iter_values(builder))
        })
        .await??;

        FixedSizeListArray::try_new(values, self.num_sub_vectors as i32)
    }

    /// Train [`ProductQuantizer`] using vectors.
    ///
    /// This method can be called repeatly, and it will start training from the previous codebook.
    pub async fn train(
        &mut self,
        data: &MatrixView,
        metric_type: MetricType,
        max_iters: usize,
    ) -> Result<()> {
        assert!(data.num_columns() % self.num_sub_vectors == 0);
        assert_eq!(data.data().null_count(), 0);

        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors);
        let num_centroids = 2_usize.pow(self.num_bits);
        let dimension = data.num_columns();
        let sub_vector_dimension = dimension / self.num_sub_vectors;

        let mut codebook_builder = Float32Builder::with_capacity(num_centroids * dimension);
        let rng = rand::rngs::SmallRng::from_entropy();

        // TODO: parallel training.
        for i in 0..sub_vectors.len() {
            // Centroids for one sub vector.
            let sub_vec = &sub_vectors[i];
            let values = sub_vec.values();
            let flatten_array: &Float32Array = as_primitive_array(&values);
            let previous_centroids = self.centroids(i);
            let centroids = train_kmeans(
                flatten_array,
                previous_centroids,
                sub_vector_dimension,
                num_centroids,
                max_iters as u32,
                rng.clone(),
                metric_type,
            )
            .await?;
            unsafe {
                codebook_builder.append_trusted_len_iter(centroids.values().iter().copied());
            }
        }
        let pd_centroids = codebook_builder.finish();
        self.codebook = Some(Arc::new(pd_centroids));

        Ok(())
    }

    /// Train a [ProductQuantizer] using an array of vectors.
    pub async fn fit_transform(
        &mut self,
        mat: &MatrixView,
        metric_type: MetricType,
    ) -> Result<FixedSizeListArray> {
        self.train(mat, metric_type, 50).await?;
        self.transform(mat, metric_type).await
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
fn divide_to_subvectors(data: &MatrixView, m: usize) -> Vec<Arc<FixedSizeListArray>> {
    assert!(!data.num_rows() > 0);

    let sub_vector_length = (data.num_columns() / m) as usize;
    let capacity = data.num_rows() * sub_vector_length;
    let mut subarrays = vec![];

    // TODO: very intensive memory copy involved!!! But this is on the write path.
    // Optimize for memory copy later.
    for i in 0..m as usize {
        let mut builder = Float32Builder::with_capacity(capacity);
        for j in 0..data.num_rows() {
            let arr = data.row(j).unwrap();
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
        let mat = MatrixView::new(values.into(), 32);
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
