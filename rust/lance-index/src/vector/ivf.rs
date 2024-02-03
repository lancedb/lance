// Copyright 2024 Lance Developers.
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
use arrow_array::{
    cast::AsArray, types::UInt32Type, Array, FixedSizeListArray, RecordBatch, UInt32Array,
};
use arrow_schema::{DataType, Field};
use arrow_select::take::take;
use async_trait::async_trait;
use futures::{stream, StreamExt};
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::{
    distance::{Cosine, Dot, MetricType, L2},
    MatrixView,
};
use log::info;
use snafu::{location, Location};
use tracing::{instrument, Instrument};

pub mod builder;
pub mod shuffler;

use super::{PART_ID_COLUMN, PQ_CODE_COLUMN, RESIDUAL_COLUMN};
use crate::vector::{
    pq::{transform::PQTransformer, ProductQuantizer},
    residual::ResidualTransform,
    transform::Transformer,
};
pub use builder::IvfBuildParams;
use lance_linalg::kmeans::KMeans;

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
    range: Option<Range<u32>>,
) -> Arc<dyn Ivf> {
    let mat = MatrixView::<T>::new(Arc::new(centroids.clone()), dimension);
    Arc::new(IvfImpl::<T>::new_with_pq(
        mat,
        metric_type,
        vector_column,
        pq,
        range,
    ))
}

pub fn new_ivf_with_pq(
    centroids: &dyn Array,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    pq: Arc<dyn ProductQuantizer>,
    range: Option<Range<u32>>,
) -> Result<Arc<dyn Ivf>> {
    match centroids.data_type() {
        DataType::Float16 => Ok(new_ivf_with_pq_impl::<Float16Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            pq,
            range,
        )),
        DataType::Float32 => Ok(new_ivf_with_pq_impl::<Float32Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            pq,
            range,
        )),
        DataType::Float64 => Ok(new_ivf_with_pq_impl::<Float64Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            pq,
            range,
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
#[async_trait]
pub trait Ivf: Send + Sync + std::fmt::Debug {
    /// Compute the partitions for each vector in the input data.
    ///
    /// Parameters
    /// ----------
    /// *data*: a matrix of vectors.
    ///
    /// Returns
    /// -------
    /// A 1-D array of partition id for each vector.
    ///
    /// Raises [Error] if the input data type does not match with the IVF model.
    ///
    async fn compute_partitions(&self, data: &FixedSizeListArray) -> Result<UInt32Array>;

    /// Compute residual vector.
    ///
    /// A residual vector is `original vector - centroids`.
    ///
    /// Parameters:
    ///  - *original*: original vector.
    ///  - *partitions*: partition ID of each original vector. If not provided, it will be computed
    ///   on the flight.
    ///
    async fn compute_residual(
        &self,
        original: &FixedSizeListArray,
        partitions: Option<&UInt32Array>,
    ) -> Result<FixedSizeListArray>;

    /// Find the closest partitions for the query vector.
    fn find_partitions(&self, query: &dyn Array, nprobes: usize) -> Result<UInt32Array>;

    /// Partition a batch of vectors into multiple batches, each batch contains vectors and other data.
    ///
    /// It transform a [RecordBatch] that contains one vector column into a record batch with
    /// schema `(PART_ID_COLUMN, ...)`, where [PART_ID_COLUMN] has the partition id for each vector.
    ///
    /// Parameters
    /// ----------
    /// - *batch*: input [RecordBatch]
    /// - *column: the name of the vector column to be partitioned and transformed.
    /// - *partion_ids*: optional precomputed partition IDs for each vector.
    /// Note that the vector column might be transformed by the `transforms` in the IVF.
    ///
    /// **Warning**: unstable API.
    async fn partition_transform(
        &self,
        batch: &RecordBatch,
        column: &str,
        partition_ids: Option<UInt32Array>,
    ) -> Result<RecordBatch>;
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
        range: Option<Range<u32>>,
    ) -> Self {
        let transforms: Vec<Arc<dyn Transformer>> = if pq.use_residual() {
            vec![
                Arc::new(ResidualTransform::new(
                    centroids.clone(),
                    PART_ID_COLUMN,
                    vector_column,
                )),
                Arc::new(PQTransformer::new(
                    pq.clone(),
                    RESIDUAL_COLUMN,
                    PQ_CODE_COLUMN,
                )),
            ]
        } else {
            vec![Arc::new(PQTransformer::new(
                pq.clone(),
                vector_column,
                PQ_CODE_COLUMN,
            ))]
        };
        Self {
            centroids: centroids.clone(),
            metric_type,
            transforms,
            partition_range: range,
        }
    }

    fn dimension(&self) -> usize {
        self.centroids.ndim()
    }

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(level = "debug", skip(data))]
    async fn do_compute_partitions(&self, data: &MatrixView<T>) -> UInt32Array {
        use lance_linalg::kmeans::compute_partitions;

        let dimension = data.ndim();
        let centroids = self.centroids.data();
        let data = data.data();
        let metric_type = self.metric_type;

        let num_centroids = centroids.len() / dimension;
        let num_rows = data.len() / dimension;

        let chunks = std::cmp::min(num_cpus::get(), num_rows);

        info!(
            "computing partition on {} chunks, out of {} centroids, and {} vectors",
            chunks, num_centroids, num_rows,
        );
        // TODO: when usize::div_ceil() comes to stable Rust, we can use it here.
        let chunk_size = num_rows / chunks + if num_rows % chunks > 0 { 1 } else { 0 };
        let stride = chunk_size * dimension;

        let result: Vec<Vec<Option<u32>>> = stream::iter(0..chunks)
            .map(|chunk_id| stride * chunk_id..std::cmp::min(stride * (chunk_id + 1), data.len()))
            // When there are a large number of CPUs and a small number of rows,
            // it's possible there isn't an split of rows that there isn't
            // an even split of rows that both covers all CPUs and all rows.
            // For example, for 400 rows and 32 CPUs, 12-element chunks (12 * 32 = 384)
            // wouldn't cover all rows but 13-element chunks (13 * 32 = 416) would
            // have one empty chunk at the end. This filter removes those empty chunks.
            .filter(|range| futures::future::ready(range.start < range.end))
            .map(|range| async {
                let range: Range<usize> = range;
                let centroids = centroids.clone();
                let data = Arc::new(
                    data.slice(range.start, range.end - range.start)
                        .as_any()
                        .downcast_ref::<T::ArrayType>()
                        .unwrap()
                        .clone(),
                );

                compute_partitions::<T>(centroids, data, dimension, metric_type)
                    .in_current_span()
                    .await
            })
            .buffered(chunks)
            .collect::<Vec<_>>()
            .await;

        UInt32Array::from_iter(result.iter().flatten().copied())
    }
}

#[async_trait]
impl<T: ArrowFloatType + Dot + L2 + Cosine + 'static> Ivf for IvfImpl<T> {
    async fn compute_partitions(&self, data: &FixedSizeListArray) -> Result<UInt32Array> {
        let array = data
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Ivf::compute_partitions: data is not expected type: {} got {}",
                    T::FLOAT_TYPE,
                    data.values().data_type()
                ),
                location: Default::default(),
            })?;
        let mat = MatrixView::<T>::new(Arc::new(array.clone()), data.value_length());
        Ok(self.do_compute_partitions(&mat).await)
    }

    async fn compute_residual(
        &self,
        original: &FixedSizeListArray,
        partitions: Option<&UInt32Array>,
    ) -> Result<FixedSizeListArray> {
        let flatten_arr = original
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Ivf::compute_residual: original is not expected type: {} got {}",
                    T::FLOAT_TYPE,
                    original.values().data_type()
                ),
                location: Default::default(),
            })?;

        let part_ids = if let Some(part_ids) = partitions {
            part_ids.clone()
        } else {
            self.compute_partitions(original).await?
        };
        let dim = original.value_length() as usize;
        let mut residual_arr: Vec<T::Native> = Vec::with_capacity(original.values().len());
        flatten_arr
            .as_slice()
            .chunks_exact(dim)
            .zip(part_ids.values())
            .for_each(|(vector, &part_id)| {
                let centroid = self.centroids.row(part_id as usize).unwrap();
                residual_arr.extend(vector.iter().zip(centroid.iter()).map(|(&v, &c)| v - c));
            });
        let arr = T::ArrayType::from(residual_arr);
        Ok(FixedSizeListArray::try_new_from_values(arr, dim as i32)?)
    }

    fn find_partitions(&self, query: &dyn Array, nprobes: usize) -> Result<UInt32Array> {
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
        // todo: hold kmeans in this struct.
        let kmeans = KMeans::<T>::with_centroids(
            self.centroids.data().clone(),
            self.dimension(),
            self.metric_type,
        );
        Ok(kmeans.find_partitions(query.as_slice(), nprobes)?)
    }

    async fn partition_transform(
        &self,
        batch: &RecordBatch,
        column: &str,
        partition_ids: Option<UInt32Array>,
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

        let part_ids = if let Some(part_ids) = partition_ids {
            part_ids
        } else {
            self.compute_partitions(data).await?
        };

        let (part_ids, batch) = if let Some(part_range) = self.partition_range.as_ref() {
            let idx_in_range: UInt32Array = part_ids
                .iter()
                .enumerate()
                .filter(|(_idx, part_id)| part_id.map(|r| part_range.contains(&r)).unwrap_or(false))
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
}
