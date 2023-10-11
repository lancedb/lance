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

use arrow_array::{
    cast::AsArray, types::Float32Type, Array, ArrayRef, FixedSizeListArray, Float32Array,
    RecordBatch, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use lance_core::{Error, Result};
use lance_linalg::{
    distance::{Cosine, Dot, MetricType, Normalize, L2},
    kernels::argmin,
    MatrixView,
};
use snafu::{location, Location};
use tracing::instrument;

use crate::vector::transform::Transformer;

pub const PQ_CODE_COLUMN: &str = "__pq_code";

/// IVF - IVF file partition
///
#[derive(Debug, Clone)]
pub(crate) struct Ivf {
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

    /// Number of IVF partitions.
    fn num_partitions(&self) -> usize {
        self.centroids.len()
    }

    /// Use the query vector to find `nprobes` closest partitions.
    fn find_partitions(
        &self,
        query: &Float32Array,
        nprobes: usize,
        metric_type: MetricType,
    ) -> Result<UInt32Array> {
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
        let dist_func = metric_type.batch_func();
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
    pub fn partition_transform(
        &self,
        batch: &RecordBatch,
        column: &str,
    ) -> Vec<(u32, RecordBatch)> {
        todo!()
    }

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(skip(data))]
    fn compute_partitions(
        &self,
        data: &MatrixView<Float32Type>,
        metric_type: MetricType,
    ) -> UInt32Array {
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
                    .map(|(centroid, &norm)| match metric_type {
                        MetricType::L2 => row.l2(centroid),
                        MetricType::Cosine => centroid.cosine_fast(norm, row),
                        MetricType::Dot => row.dot(centroid),
                    }),
            )
            .expect("argmin should always return a value")
        }))
    }
}
