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

use std::any::Any;
use std::sync::Arc;

use arrow_arith::aggregate::min;
use arrow_array::{
    builder::Float32Builder, cast::as_primitive_array, types::Float32Type, Array, ArrayRef,
    FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::take::take;
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_linalg::{
    distance::{l2::l2_distance_batch, norm_l2::norm_l2, Dot, Normalize},
    kernels::argmin_opt,
    matrix::MatrixView,
};
use rand::SeedableRng;
use serde::Serialize;

use super::{MetricType, Query, VectorIndex};
use crate::arrow::*;
use crate::dataset::ROW_ID;
use crate::index::{
    pb, prefilter::PreFilter, vector::kmeans::train_kmeans, vector::DIST_COL, Index,
};
use crate::io::object_reader::{read_fixed_stride_array, ObjectReader};
use crate::{Error, Result};

/// Product Quantization Index.
///
pub struct PQIndex {
    /// Number of bits for the centroids.
    ///
    /// Only support 8, as one of `u8` byte now.
    pub nbits: u32,

    /// Number of sub-vectors.
    pub num_sub_vectors: usize,

    /// Vector dimension.
    pub dimension: usize,

    /// Product quantizer.
    pub pq: Arc<ProductQuantizer>,

    /// PQ code
    pub code: Option<Arc<UInt8Array>>,

    /// ROW Id used to refer to the actual row in dataset.
    pub row_ids: Option<Arc<UInt64Array>>,

    /// Metric type.
    metric_type: MetricType,
}

impl std::fmt::Debug for PQIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PQ(m={}, nbits={}, {})",
            self.num_sub_vectors, self.nbits, self.metric_type
        )
    }
}

impl PQIndex {
    /// Load a PQ index (page) from the disk.
    pub(crate) fn new(pq: Arc<ProductQuantizer>, metric_type: MetricType) -> Self {
        Self {
            nbits: pq.num_bits,
            num_sub_vectors: pq.num_sub_vectors,
            dimension: pq.dimension,
            code: None,
            row_ids: None,
            pq,
            metric_type,
        }
    }

    fn fast_l2_distances(&self, key: &Float32Array, code: &UInt8Array) -> Result<ArrayRef> {
        // Build distance table for each sub-centroid to the query key.
        //
        // Distance table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let mut distance_table: Vec<f32> = vec![];

        let sub_vector_length = self.dimension / self.num_sub_vectors;
        for i in 0..self.num_sub_vectors {
            let from = key.slice(i * sub_vector_length, sub_vector_length);
            let subvec_centroids = self.pq.centroids(i).ok_or_else(|| Error::Index {
                message: "PQIndex::l2_distances: PQ is not initialized".to_string(),
            })?;
            let distances = l2_distance_batch(
                as_primitive_array::<Float32Type>(&from).values(),
                subvec_centroids.values(),
                sub_vector_length,
            );
            distance_table.extend(distances.values());
        }

        Ok(Arc::new(unsafe {
            Float32Array::from_trusted_len_iter(
                code.values().chunks_exact(self.num_sub_vectors).map(|c| {
                    Some(
                        c.iter()
                            .enumerate()
                            .map(|(sub_vec_idx, centroid)| {
                                distance_table[sub_vec_idx * 256 + *centroid as usize]
                            })
                            .sum::<f32>(),
                    )
                }),
            )
        }))
    }

    /// Pre-compute cosine distance to each sub-centroids.
    ///
    /// Parameters
    ///  - query: the query vector, with shape (dimension, )
    ///  - code: the PQ code in one partition.
    ///
    fn cosine_distances(&self, query: &Float32Array, code: &UInt8Array) -> Result<ArrayRef> {
        // Build two tables for cosine distance.
        //
        // xy table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        // y_norm table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let num_centroids = ProductQuantizer::num_centroids(self.nbits);
        let mut xy_table: Vec<f32> = Vec::with_capacity(self.num_sub_vectors * num_centroids);
        let mut y2_table: Vec<f32> = Vec::with_capacity(self.num_sub_vectors * num_centroids);

        let x_norm = norm_l2(query.values());
        let sub_vector_length = self.dimension / self.num_sub_vectors;
        for i in 0..self.num_sub_vectors {
            // The sub-vector section of the query vector.
            let key_sub_vector =
                &query.values()[i * sub_vector_length..(i + 1) * sub_vector_length];
            let sub_vector_centroids = self.pq.centroids(i).ok_or_else(|| Error::Index {
                message: "PQIndex::cosine_distances: PQ is not initialized".to_string(),
            })?;
            let xy = sub_vector_centroids
                .values()
                .chunks_exact(sub_vector_length)
                .map(|cent| cent.dot(key_sub_vector));
            xy_table.extend(xy);

            let y2 = sub_vector_centroids
                .values()
                .chunks_exact(sub_vector_length)
                .map(|cent| cent.norm_l2().powf(2.0));
            y2_table.extend(y2);
        }

        // Compute distance from the pre-compute table.
        Ok(Arc::new(Float32Array::from_iter(
            code.values().chunks_exact(self.num_sub_vectors).map(|c| {
                let xy = c
                    .iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        let idx = sub_vec_idx * num_centroids + *centroid as usize;
                        xy_table[idx]
                    })
                    .sum::<f32>();
                let y2 = c
                    .iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        let idx = sub_vec_idx * num_centroids + *centroid as usize;
                        y2_table[idx]
                    })
                    .sum::<f32>();
                1.0 - xy / (x_norm * y2.sqrt())
            }),
        )))
    }

    /// Filter the row id and PQ code arrays based on the pre-filter.
    async fn filter_arrays(
        &self,
        pre_filter: &PreFilter,
    ) -> Result<(Arc<UInt8Array>, Arc<UInt64Array>)> {
        if self.code.is_none() || self.row_ids.is_none() {
            return Err(Error::Index {
                message: "PQIndex::filter_arrays: PQ is not initialized".to_string(),
            });
        }
        let code = self.code.clone().unwrap();
        let row_ids = self.row_ids.clone().unwrap();
        let indices_to_keep = pre_filter.filter_row_ids(row_ids.values()).await?;
        let indices_to_keep = UInt64Array::from(indices_to_keep);

        let row_ids = take(row_ids.as_ref(), &indices_to_keep, None)?;
        let row_ids = Arc::new(as_primitive_array(&row_ids).clone());

        let code = FixedSizeListArray::try_new_from_values(
            code.as_ref().clone(),
            self.pq.num_sub_vectors as i32,
        )
        .unwrap();
        let code = take(&code, &indices_to_keep, None)?;
        let code = as_fixed_size_list_array(&code).values().clone();
        let code = Arc::new(as_primitive_array(&code).clone());

        Ok((code, row_ids))
    }
}

#[derive(Serialize)]
pub struct PQIndexStatistics {
    index_type: String,
    nbits: u32,
    num_sub_vectors: usize,
    dimension: usize,
    metric_type: String,
}

impl Index for PQIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(PQIndexStatistics {
            index_type: "PQ".to_string(),
            nbits: self.nbits,
            num_sub_vectors: self.num_sub_vectors,
            dimension: self.dimension,
            metric_type: self.metric_type.to_string(),
        })?)
    }
}

#[async_trait]
impl VectorIndex for PQIndex {
    /// Search top-k nearest neighbors for `key` within one PQ partition.
    ///
    async fn search(&self, query: &Query, pre_filter: &PreFilter) -> Result<RecordBatch> {
        if self.code.is_none() || self.row_ids.is_none() {
            return Err(Error::Index {
                message: "PQIndex::search: PQ is not initialized".to_string(),
            });
        }

        let (code, row_ids) = if pre_filter.is_empty() {
            (
                self.code.as_ref().unwrap().clone(),
                self.row_ids.as_ref().unwrap().clone(),
            )
        } else {
            self.filter_arrays(pre_filter).await?
        };

        // Pre-compute distance table for each sub-vector.
        let distances = if self.metric_type == MetricType::L2 {
            self.fast_l2_distances(&query.key, code.as_ref())?
        } else {
            self.cosine_distances(&query.key, code.as_ref())?
        };

        debug_assert_eq!(distances.len(), row_ids.len());

        let limit = query.k * query.refine_factor.unwrap_or(1) as usize;
        let indices = sort_to_indices(&distances, None, Some(limit))?;
        let distances = take(&distances, &indices, None)?;
        let row_ids = take(row_ids.as_ref(), &indices, None)?;

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(DIST_COL, DataType::Float32, false),
            ArrowField::new(ROW_ID, DataType::UInt64, false),
        ]));
        Ok(RecordBatch::try_new(schema, vec![distances, row_ids])?)
    }

    fn is_loadable(&self) -> bool {
        true
    }

    /// Load a PQ index (page) from the disk.
    async fn load(
        &self,
        reader: &dyn ObjectReader,
        offset: usize,
        length: usize,
    ) -> Result<Arc<dyn VectorIndex>> {
        let pq_code_length = self.pq.num_sub_vectors * length;
        let pq_code =
            read_fixed_stride_array(reader, &DataType::UInt8, offset, pq_code_length, ..).await?;

        let row_id_offset = offset + pq_code_length /* *1 */;
        let row_ids =
            read_fixed_stride_array(reader, &DataType::UInt64, row_id_offset, length, ..).await?;

        Ok(Arc::new(Self {
            nbits: self.pq.num_bits,
            num_sub_vectors: self.pq.num_sub_vectors,
            dimension: self.pq.dimension,
            code: Some(Arc::new(as_primitive_array(&pq_code).clone())),
            row_ids: Some(Arc::new(as_primitive_array(&row_ids).clone())),
            pq: self.pq.clone(),
            metric_type: self.metric_type,
        }))
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

#[allow(clippy::fallible_impl_from)]
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

/// Train product quantization over (OPQ-rotated) residual vectors.
pub(crate) async fn train_pq(
    data: &MatrixView<Float32Type>,
    params: &PQBuildParams,
) -> Result<ProductQuantizer> {
    let mut pq = ProductQuantizer::new(
        params.num_sub_vectors,
        params.num_bits as u32,
        data.num_columns(),
    );
    pq.train(data, params).await?;
    Ok(pq)
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::relative_eq;
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

    #[ignore]
    #[tokio::test]
    async fn test_train_pq_iteratively() {
        let mut params = PQBuildParams::new(2, 8);
        params.max_iters = 1;

        let values = Float32Array::from_iter((0..16000).map(|v| v as f32));
        // A 16-dim array.
        let dim = 16;
        let mat = MatrixView::new(values.into(), dim);
        let mut pq = ProductQuantizer::new(2, 8, dim);
        pq.train(&mat, &params).await.unwrap();

        // Init centroids
        let centroids = pq.codebook.as_ref().unwrap().clone();

        // Keep training 10 times
        pq.train(&mat, &params).await.unwrap();

        let mut actual_pq = ProductQuantizer {
            num_bits: 8,
            num_sub_vectors: 2,
            dimension: dim,
            codebook: Some(centroids),
        };
        // Iteratively train for 10 times.
        for _ in 0..10 {
            let code = actual_pq.transform(&mat, MetricType::L2).await.unwrap();
            actual_pq.reset_centroids(&mat, &code).unwrap();
            actual_pq.train(&mat, &params).await.unwrap();
        }

        let result = pq.codebook.unwrap();
        let expected = actual_pq.codebook.unwrap();
        result
            .values()
            .iter()
            .zip(expected.values())
            .for_each(|(&r, &e)| {
                assert!(relative_eq!(r, e));
            });
    }
}
