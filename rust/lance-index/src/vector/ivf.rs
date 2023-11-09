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

use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{cast::AsArray, types::UInt32Type, Array, RecordBatch, UInt32Array};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field};
use arrow_select::take::take;
use lance_arrow::{ArrowFloatType, FloatArray, RecordBatchExt};
use lance_core::{Error, Result};
use lance_linalg::{
    distance::{
        cosine_distance_batch, dot_distance_batch, l2_distance_batch, Cosine, Dot, MetricType, L2,
    },
    MatrixView,
};
use snafu::{location, Location};
use tracing::instrument;

use super::{PART_ID_COLUMN, PQ_CODE_COLUMN, RESIDUAL_COLUMN};
use crate::vector::{
    pq::{transform::PQTransformer, ProductQuantizer},
    residual::ResidualTransform,
    transform::Transformer,
};

fn new_ivf_impl<T: ArrowFloatType + Dot + Cosine + L2 + 'static>(
    centroids: &T::ArrayType,
    dimension: usize,
    metric_type: MetricType,
    transforms: Vec<Arc<dyn Transformer>>,
    range: Option<Range<u32>>,
) -> Arc<dyn Ivf> {
    let mat = MatrixView::<T>::new(Arc::new(centroids.clone()), dimension);
    Arc::new(IvfImpl::<T>::new(mat, metric_type, transforms, range))
}

/// Create an IVF from the flatten centroids.
///
/// Parameters
/// ----------
/// - *centroids*: a flatten floating number array of centroids.
/// - *dimension*: dimension of the vector.
/// - *metric_type*: metric type to compute pair-wise vector distance.
/// - *transforms*: a list of transforms to apply to the vector column.
/// - *range*: only covers a range of partitions. Default is None
pub fn new_ivf(
    centroids: &dyn Array,
    dimension: usize,
    metric_type: MetricType,
    transforms: Vec<Arc<dyn Transformer>>,
    range: Option<Range<u32>>,
) -> Result<Arc<dyn Ivf>> {
    match centroids.data_type() {
        DataType::Float16 => Ok(new_ivf_impl::<Float16Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            transforms,
            range,
        )),
        DataType::Float32 => Ok(new_ivf_impl::<Float32Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            transforms,
            range,
        )),
        DataType::Float64 => Ok(new_ivf_impl::<Float64Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            transforms,
            range,
        )),
        _ => Err(Error::Index {
            message: format!(
                "new_ivf: centroids is not expected type: {}",
                centroids.data_type()
            ),
            location: location!(),
        }),
    }
}

fn new_ivf_with_pq_impl<T: ArrowFloatType + Dot + Cosine + L2 + 'static>(
    centroids: &T::ArrayType,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    pq: Arc<dyn ProductQuantizer>,
) -> Arc<dyn Ivf> {
    let mat = MatrixView::<T>::new(Arc::new(centroids.clone()), dimension);
    Arc::new(IvfImpl::<T>::new_with_pq(
        mat,
        metric_type,
        vector_column,
        pq,
    ))
}
pub fn new_ivf_with_pq(
    centroids: &dyn Array,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    pq: Arc<dyn ProductQuantizer>,
) -> Result<Arc<dyn Ivf>> {
    match centroids.data_type() {
        DataType::Float16 => Ok(new_ivf_with_pq_impl::<Float16Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            pq,
        )),
        DataType::Float32 => Ok(new_ivf_with_pq_impl::<Float32Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            pq,
        )),
        DataType::Float64 => Ok(new_ivf_with_pq_impl::<Float64Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            pq,
        )),
        _ => Err(Error::Index {
            message: format!(
                "new_ivf_with_pq: centroids is not expected type: {}",
                centroids.data_type()
            ),
            location: location!(),
        }),
    }
}

/// IVF - IVF file partition
///
pub trait Ivf {
    /// Find the closest partitions for the query vector.
    fn find_partitions(&self, query: &dyn Array, nprobes: usize) -> Result<UInt32Array>;
}

/// IVF - IVF file partition
///
#[derive(Debug, Clone)]
pub struct IvfImpl<T: ArrowFloatType + Dot + L2 + Cosine> {
    /// KMean model of the IVF
    ///
    /// It is a 2-D `(num_partitions * dimension)` of float32 array, 64-bit aligned via Arrow
    /// memory allocator.
    centroids: MatrixView<T>,

    /// Transform applied to each partition.
    transforms: Vec<Arc<dyn Transformer>>,

    /// Metric type to compute pair-wise vector distance.
    metric_type: MetricType,

    /// Only covers a range of partitions.
    partition_range: Option<Range<u32>>,
}

impl<T: ArrowFloatType + Dot + L2 + Cosine + 'static> IvfImpl<T> {
    pub fn new(
        centroids: MatrixView<T>,
        metric_type: MetricType,
        transforms: Vec<Arc<dyn Transformer>>,
        range: Option<Range<u32>>,
    ) -> Self {
        Self {
            centroids,
            metric_type,
            transforms,
            partition_range: range,
        }
    }

    fn new_with_pq(
        centroids: MatrixView<T>,
        metric_type: MetricType,
        vector_column: &str,
        pq: Arc<dyn ProductQuantizer>,
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

    fn dimension(&self) -> usize {
        self.centroids.ndim()
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
            location: location!(),
        })?;
        let data = vector_arr.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "Column {} is not a vector type: {}",
                column,
                vector_arr.data_type()
            ),
            location: location!(),
        })?;

        let flatten_data = data
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Column {} is not a vector type: expect: {} got {}",
                    column,
                    T::FLOAT_TYPE,
                    vector_arr.data_type()
                ),
                location: location!(),
            })?;
        let matrix = MatrixView::<T>::new(Arc::new(flatten_data.clone()), data.value_length());
        let part_ids = self.compute_partitions(&matrix).await;

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
    pub async fn compute_partitions(&self, data: &MatrixView<T>) -> UInt32Array {
        use lance_linalg::kmeans::compute_partitions;

        let dimension = data.ndim();
        let centroids = self.centroids.data();
        let data = data.data();
        let metric_type = self.metric_type;

        tokio::task::spawn_blocking(move || {
            compute_partitions::<T>(
                centroids.as_slice(),
                data.as_slice(),
                dimension,
                metric_type,
            )
            .into()
        })
        .await
        .expect("compute_partitions: schedule CPU task")
    }
}

impl<T: ArrowFloatType + Dot + L2 + Cosine + 'static> Ivf for IvfImpl<T> {
    fn find_partitions(&self, query: &dyn Array, nprobes: usize) -> Result<UInt32Array> {
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
        let query = query
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Ivf::find_partition: query is not expected type: {} got {}",
                    T::FLOAT_TYPE,
                    query.data_type()
                ),
                location: Default::default(),
            })?;
        let centroid_values = self.centroids.data();
        let centroids = centroid_values.as_slice();
        let dim = query.len();
        let distances: T::ArrayType = match self.metric_type {
            lance_linalg::distance::DistanceType::L2 => {
                l2_distance_batch(query.as_slice(), centroids, dim)
            }
            lance_linalg::distance::DistanceType::Cosine => {
                cosine_distance_batch(query.as_slice(), centroids, dim)
            }
            lance_linalg::distance::DistanceType::Dot => {
                dot_distance_batch(query.as_slice(), centroids, dim)
            }
        }
        .collect::<Vec<_>>()
        .into();
        let top_k_partitions = sort_to_indices(&distances, None, Some(nprobes))?;
        Ok(top_k_partitions)
    }
}
