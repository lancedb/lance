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

use std::sync::Arc;

use arrow_array::{
    cast::AsArray, types::Float32Type, Array, ArrayRef, FixedSizeListArray, Float32Array,
    RecordBatch, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::Field;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_linalg::{
    distance::{Cosine, Dot, MetricType, Normalize, L2},
    kernels::argmin,
    MatrixView,
};
use snafu::{location, Location};
use tracing::instrument;

use super::PART_ID_COLUMN;
use crate::vector::transform::Transformer;

/// IVF - IVF file partition
///
#[derive(Debug, Clone)]
pub struct Ivf {
    /// KMean model of the IVF
    ///
    /// It is a 2-D `(num_partitions * dimension)` of float32 array, 64-bit aligned via Arrow
    /// memory allocator.
    centroids: Arc<FixedSizeListArray>,

    /// Transform applied to each partition.
    transforms: Vec<Arc<dyn Transformer>>,

    /// Metric type to compute pair-wise vector distance.
    metric_type: MetricType,
}

impl Ivf {
    pub fn new(
        centroids: Arc<FixedSizeListArray>,
        metric_type: MetricType,
        transforms: Vec<Arc<dyn Transformer>>,
    ) -> Self {
        Self {
            centroids,
            metric_type,
            transforms,
        }
    }

    fn dimension(&self) -> usize {
        self.centroids.value_length() as usize
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
        let centroid_values = self.centroids.values();
        let distances = dist_func(
            query.values(),
            centroid_values.as_primitive::<Float32Type>().values(),
            self.dimension(),
        ) as ArrayRef;
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
    pub fn partition_transform(&self, batch: &RecordBatch, column: &str) -> Result<RecordBatch> {
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
        let matrix = MatrixView::<Float32Type>::try_from(data)?;
        let part_ids = self.compute_partitions(&matrix);

        let field = Field::new(PART_ID_COLUMN, part_ids.data_type().clone(), false);
        let mut batch = batch.try_with_column(field, Arc::new(part_ids))?;

        // Transform each batch
        for transform in self.transforms.as_slice() {
            batch = transform.transform(&batch)?;
        }

        Ok(batch)
    }

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(skip(data))]
    fn compute_partitions(&self, data: &MatrixView<Float32Type>) -> UInt32Array {
        let ndim = data.ndim();
        let centroids_arr: &Float32Array = self.centroids.values().as_primitive();
        let centroid_norms = centroids_arr
            .values()
            .chunks(ndim)
            .map(|centroid| centroid.norm_l2())
            .collect::<Vec<_>>();
        UInt32Array::from_iter_values(data.iter().map(|row| {
            argmin(
                centroids_arr
                    .values()
                    .chunks(ndim)
                    .zip(centroid_norms.iter())
                    .map(|(centroid, &norm)| match self.metric_type {
                        MetricType::L2 => row.l2(centroid),
                        MetricType::Cosine => centroid.cosine_fast(norm, row),
                        MetricType::Dot => row.dot(centroid),
                    }),
            )
            .expect("argmin should always return a value")
        }))
    }
}
