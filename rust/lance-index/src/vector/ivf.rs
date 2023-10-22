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

//! IVF - Inverted File Index

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt32Type},
    Array, ArrayRef, Float32Array, RecordBatch, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::Field;
use arrow_select::take::take;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_linalg::{distance::MetricType, MatrixView};
use snafu::{location, Location};
use tracing::instrument;

use super::{PART_ID_COLUMN, PQ_CODE_COLUMN, RESIDUAL_COLUMN};
use crate::vector::{
    pq::{transform::PQTransformer, ProductQuantizer},
    residual::ResidualTransform,
    transform::Transformer,
};

/// IVF - IVF file partition
///
#[derive(Debug, Clone)]
pub struct Ivf {
    /// KMean model of the IVF
    ///
    /// It is a 2-D `(num_partitions * dimension)` of float32 array, 64-bit aligned via Arrow
    /// memory allocator.
    centroids: MatrixView<Float32Type>,

    /// Transform applied to each partition.
    transforms: Vec<Arc<dyn Transformer>>,

    /// Metric type to compute pair-wise vector distance.
    metric_type: MetricType,

    /// Only covers a range of partitions.
    partition_range: Option<Range<u32>>,
}

impl Ivf {
    pub fn new(
        centroids: MatrixView<Float32Type>,
        metric_type: MetricType,
        transforms: Vec<Arc<dyn Transformer>>,
    ) -> Self {
        Self {
            centroids,
            metric_type,
            transforms,
            partition_range: None,
        }
    }

    pub fn new_with_pq(
        centroids: MatrixView<Float32Type>,
        metric_type: MetricType,
        vector_column: &str,
        pq: Arc<ProductQuantizer>,
    ) -> Self {
        Self {
            centroids: centroids.clone(),
            metric_type,
            transforms: vec![
                Arc::new(ResidualTransform::new(
                    centroids,
                    PART_ID_COLUMN,
                    vector_column,
                )),
                Arc::new(PQTransformer::new(
                    pq.clone(),
                    RESIDUAL_COLUMN,
                    PQ_CODE_COLUMN,
                )),
            ],
            partition_range: None,
        }
    }

    pub fn new_with_range(
        centroids: MatrixView<Float32Type>,
        metric_type: MetricType,
        transforms: Vec<Arc<dyn Transformer>>,
        range: Range<u32>,
    ) -> Self {
        Self {
            centroids,
            metric_type,
            transforms,
            partition_range: Some(range),
        }
    }

    fn dimension(&self) -> usize {
        self.centroids.ndim()
    }

    /// Use the query vector to find `nprobes` closest partitions.
    pub fn find_partitions(&self, query: &Float32Array, nprobes: usize) -> Result<UInt32Array> {
        if query.len() != self.dimension() {
            return Err(Error::IO {
                message: format!(
                    "Ivf::find_partition: dimension mismatch: {} != {}",
                    query.len(),
                    self.dimension()
                ),
                location: location!(),
            });
        }
        let dist_func = self.metric_type.batch_func();
        let centroid_values = self.centroids.data();
        let distances =
            dist_func(query.values(), centroid_values.values(), self.dimension()) as ArrayRef;
        let top_k_partitions = sort_to_indices(&distances, None, Some(nprobes))?;
        Ok(top_k_partitions)
    }

    /// Partition a batch of vectors into multiple batches, each batch contains vectors and other data.
    ///
    /// It transform a [RecordBatch] that contains one vector column into a record batch with
    /// schema `(PART_ID_COLUMN, ...)`, where [PART_ID_COLUMN] has the partition id for each vector.
    ///
    /// Note that the vector column might be transformed by the `transforms` in the IVF.
    ///
    /// **Warning**: unstable API.
    pub async fn partition_transform(
        &self,
        batch: &RecordBatch,
        column: &str,
    ) -> Result<RecordBatch> {
        let vector_arr = batch.column_by_name(column).ok_or(Error::Index {
            message: format!("Column {} does not exist.", column),
        })?;
        let data = vector_arr.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "Column {} is not a vector type: {}",
                column,
                vector_arr.data_type()
            ),
        })?;
        let matrix = Arc::new(MatrixView::<Float32Type>::try_from(data)?);
        let part_ids = self.compute_partitions(matrix).await;

        let (part_ids, batch) = if let Some(part_range) = self.partition_range.as_ref() {
            let idx_in_range: UInt32Array = part_ids
                .values()
                .iter()
                .enumerate()
                .filter(|(_, part_id)| part_range.contains(*part_id))
                .map(|(idx, _)| idx as u32)
                .collect();
            let part_ids = take(&part_ids, &idx_in_range, None)?
                .as_primitive::<UInt32Type>()
                .clone();
            let batch = batch.take(&idx_in_range)?;
            (part_ids, batch)
        } else {
            (part_ids, batch.clone())
        };

        let field = Field::new(PART_ID_COLUMN, part_ids.data_type().clone(), false);
        let mut batch = batch.try_with_column(field, Arc::new(part_ids))?;

        // Transform each batch
        for transform in self.transforms.as_slice() {
            batch = transform.transform(&batch).await?;
        }

        Ok(batch)
    }

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(level = "debug", skip(data))]
    async fn compute_partitions(&self, data: &MatrixView<Float32Type>) -> UInt32Array {
        use lance_linalg::kmeans::compute_partitions;

        let dimension = data.ndim();
        let centroids = self.centroids.data();
        let data = data.data();
        let metric_type = self.metric_type;

        tokio::task::spawn_blocking(move || {
            compute_partitions(centroids.values(), data.values(), dimension, metric_type).into()
        })
        .await
        .unwrap()
    }
}
