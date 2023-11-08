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
    cast::{as_primitive_array, AsArray},
    Array, FixedSizeListArray, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::take::take;
use async_trait::async_trait;
// Re-export
use lance_core::{
    io::{read_fixed_stride_array, Reader},
    ROW_ID_FIELD,
};
pub use lance_index::vector::pq::{PQBuildParams, ProductQuantizerImpl};
use lance_index::{
    vector::{pq::ProductQuantizer, Query, DIST_COL},
    Index, IndexType,
};
use lance_linalg::distance::MetricType;
use serde::Serialize;
use snafu::{location, Location};
use tracing::{instrument, Instrument};

use super::VectorIndex;
use crate::index::prefilter::PreFilter;
use crate::{arrow::*, utils::tokio::spawn_cpu};
use crate::{Error, Result};

/// Product Quantization Index.
///
#[derive(Clone)]
pub struct PQIndex {
    /// Product quantizer.
    pub pq: Arc<dyn ProductQuantizer>,

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
            self.pq.num_sub_vectors(),
            self.pq.num_bits(),
            self.metric_type
        )
    }
}

impl PQIndex {
    /// Load a PQ index (page) from the disk.
    pub(crate) fn new(pq: Arc<dyn ProductQuantizer>, metric_type: MetricType) -> Self {
        Self {
            code: None,
            row_ids: None,
            pq,
            metric_type,
        }
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

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn statistics(&self) -> Result<String> {
        Ok(serde_json::to_string(&PQIndexStatistics {
            index_type: "PQ".to_string(),
            nbits: self.pq.num_bits(),
            num_sub_vectors: self.pq.num_sub_vectors(),
            dimension: self.pq.dimension(),
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
                location: location!(),
            });
        }
        pre_filter.wait_for_ready().await?;

        let code = self.code.as_ref().unwrap().clone();
        let row_ids = self.row_ids.as_ref().unwrap().clone();

        let pq = self.pq.clone();
        let query = query.clone();
        let num_sub_vectors = self.pq.num_sub_vectors() as i32;
        spawn_cpu(move || {
            let (code, row_ids) = if pre_filter.is_empty() {
                Ok((code, row_ids))
            } else {
                Self::filter_arrays(pre_filter.as_ref(), code, row_ids, num_sub_vectors)
            }?;

            // Pre-compute distance table for each sub-vector.
            let distances = pq.build_distance_table(query.key.as_ref(), &code)?;

            debug_assert_eq!(distances.len(), row_ids.len());

            let limit = query.k * query.refine_factor.unwrap_or(1) as usize;
            let indices = sort_to_indices(&distances, None, Some(limit))?;
            let distances = take(&distances, &indices, None)?;
            let row_ids = take(row_ids.as_ref(), &indices, None)?;

            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new(DIST_COL, DataType::Float32, true),
                ROW_ID_FIELD.clone(),
            ]));
            Ok(RecordBatch::try_new(schema, vec![distances, row_ids])?)
        })
        .in_current_span()
        .await
    }

    fn is_loadable(&self) -> bool {
        true
    }

    /// Load a PQ index (page) from the disk.
    async fn load(
        &self,
        reader: &dyn Reader,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let pq_code_length = self.pq.num_sub_vectors() * length;
        let pq_code =
            read_fixed_stride_array(reader, &DataType::UInt8, offset, pq_code_length, ..).await?;

        let row_id_offset = offset + pq_code_length /* *1 */;
        let row_ids =
            read_fixed_stride_array(reader, &DataType::UInt64, row_id_offset, length, ..).await?;

        Ok(Box::new(Self {
            code: Some(Arc::new(pq_code.as_primitive().clone())),
            row_ids: Some(Arc::new(row_ids.as_primitive().clone())),
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
            .chunks_exact(self.pq.num_sub_vectors());
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

#[cfg(test)]
mod tests {}
