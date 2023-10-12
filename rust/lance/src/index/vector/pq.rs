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
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    cast::as_primitive_array, types::Float32Type, Array, ArrayRef, FixedSizeListArray,
    Float32Array, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::take::take;
use async_trait::async_trait;
// Re-export
pub use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_linalg::{
    distance::{l2::l2_distance_batch, norm_l2::norm_l2, Dot, Normalize},
    matrix::MatrixView,
};
use serde::Serialize;
use tracing::instrument;

use super::{MetricType, Query, VectorIndex};
use crate::dataset::ROW_ID;
use crate::index::{pb, prefilter::PreFilter, vector::DIST_COL, Index};
use crate::io::object_reader::{read_fixed_stride_array, ObjectReader};
use crate::{arrow::*, utils::tokio::spawn_cpu};
use crate::{Error, Result};

/// Product Quantization Index.
///
#[derive(Clone)]
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
            let subvec_centroids = self.pq.centroids(i);
            let distances = l2_distance_batch(
                as_primitive_array::<Float32Type>(&from).values(),
                subvec_centroids,
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
            let sub_vector_centroids = self.pq.centroids(i);
            let xy = sub_vector_centroids
                .chunks_exact(sub_vector_length)
                .map(|cent| cent.dot(key_sub_vector));
            xy_table.extend(xy);

            let y2 = sub_vector_centroids
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
    fn filter_arrays(
        pre_filter: &PreFilter,
        code: Arc<UInt8Array>,
        row_ids: Arc<UInt64Array>,
        num_sub_vectors: i32,
    ) -> Result<(Arc<UInt8Array>, Arc<UInt64Array>)> {
        let indices_to_keep = pre_filter.filter_row_ids(row_ids.values());
        let indices_to_keep = UInt64Array::from(indices_to_keep);

        let row_ids = take(row_ids.as_ref(), &indices_to_keep, None)?;
        let row_ids = Arc::new(as_primitive_array(&row_ids).clone());

        let code = FixedSizeListArray::try_new_from_values(code.as_ref().clone(), num_sub_vectors)
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
    #[instrument(level = "debug", skip_all, name = "PQIndex::search")]
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
        if self.code.is_none() || self.row_ids.is_none() {
            return Err(Error::Index {
                message: "PQIndex::search: PQ is not initialized".to_string(),
            });
        }
        pre_filter.wait_for_ready().await?;

        let code = self.code.as_ref().unwrap().clone();
        let row_ids = self.row_ids.as_ref().unwrap().clone();

        let this = self.clone();
        let query = query.clone();
        let num_sub_vectors = self.pq.num_sub_vectors as i32;
        spawn_cpu(move || {
            let (code, row_ids) = if pre_filter.is_empty() {
                Ok((code, row_ids))
            } else {
                Self::filter_arrays(pre_filter.as_ref(), code, row_ids, num_sub_vectors)
            }?;

            // Pre-compute distance table for each sub-vector.
            let distances = if this.metric_type == MetricType::L2 {
                this.fast_l2_distances(&query.key, code.as_ref())?
            } else {
                this.cosine_distances(&query.key, code.as_ref())?
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
        })
        .await
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
    ) -> Result<Box<dyn VectorIndex>> {
        let pq_code_length = self.pq.num_sub_vectors * length;
        let pq_code =
            read_fixed_stride_array(reader, &DataType::UInt8, offset, pq_code_length, ..).await?;

        let row_id_offset = offset + pq_code_length /* *1 */;
        let row_ids =
            read_fixed_stride_array(reader, &DataType::UInt64, row_id_offset, length, ..).await?;

        Ok(Box::new(Self {
            nbits: self.pq.num_bits,
            num_sub_vectors: self.pq.num_sub_vectors,
            dimension: self.pq.dimension,
            code: Some(Arc::new(as_primitive_array(&pq_code).clone())),
            row_ids: Some(Arc::new(as_primitive_array(&row_ids).clone())),
            pq: self.pq.clone(),
            metric_type: self.metric_type,
        }))
    }

    fn check_can_remap(&self) -> Result<()> {
        Ok(())
    }

    fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        let code = self
            .code
            .as_ref()
            .unwrap()
            .values()
            .chunks_exact(self.num_sub_vectors);
        let row_ids = self.row_ids.as_ref().unwrap().values().iter();
        let remapped = row_ids
            .zip(code)
            .filter_map(|(old_row_id, code)| {
                let new_row_id = mapping.get(old_row_id).cloned();
                // If the row id is not in the mapping then this row is not remapped and we keep as is
                let new_row_id = new_row_id.unwrap_or(Some(*old_row_id));
                new_row_id.map(|new_row_id| (new_row_id, code))
            })
            .collect::<Vec<_>>();

        self.row_ids = Some(Arc::new(UInt64Array::from_iter_values(
            remapped.iter().map(|(row_id, _)| *row_id),
        )));
        self.code = Some(Arc::new(UInt8Array::from_iter_values(
            remapped.into_iter().flat_map(|(_, code)| code).copied(),
        )));
        Ok(())
    }
}

#[allow(clippy::fallible_impl_from)]
impl From<&ProductQuantizer> for pb::Pq {
    fn from(pq: &ProductQuantizer) -> Self {
        Self {
            num_bits: pq.num_bits,
            num_sub_vectors: pq.num_sub_vectors as u32,
            dimension: pq.dimension as u32,
            codebook: pq.codebook.values().to_vec(),
        }
    }
}

/// Train product quantization over (OPQ-rotated) residual vectors.
pub(crate) async fn train_pq(
    data: &MatrixView<Float32Type>,
    params: &PQBuildParams,
) -> Result<ProductQuantizer> {
    ProductQuantizer::train(data, params).await
}

#[cfg(test)]
mod tests {}
