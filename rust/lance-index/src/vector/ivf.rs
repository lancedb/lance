// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File Index

use std::ops::Range;
use std::sync::Arc;

use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{
    cast::AsArray, Array, ArrowPrimitiveType, FixedSizeListArray, RecordBatch, UInt32Array,
};
use arrow_schema::DataType;
use async_trait::async_trait;
use lance_linalg::distance::Normalize;
use snafu::{location, Location};

pub use builder::IvfBuildParams;
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::{
    distance::{Dot, MetricType, L2},
    kmeans::KMeans,
    Clustering, MatrixView,
};

use crate::vector::ivf::transform::IvfTransformer;
use crate::vector::{
    pq::{transform::PQTransformer, ProductQuantizer},
    residual::ResidualTransform,
    transform::Transformer,
};

use super::quantizer::Quantizer;
use super::transform::DropColumn;
use super::{PART_ID_COLUMN, PQ_CODE_COLUMN, RESIDUAL_COLUMN};

pub mod builder;
pub mod shuffler;
pub mod storage;
mod transform;

fn new_ivf_impl<T: ArrowFloatType + ArrowPrimitiveType>(
    centroids: &T::ArrayType,
    dimension: usize,
    metric_type: MetricType,
    transforms: Vec<Arc<dyn Transformer>>,
    range: Option<Range<u32>>,
) -> Arc<dyn Ivf>
where
    <T as ArrowFloatType>::Native: Dot + L2 + Normalize,
{
    let mat = MatrixView::<T>::new(Arc::new(centroids.clone()), dimension);
    Arc::new(IvfImpl::<T>::new(mat, metric_type, "", transforms, range))
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

fn new_ivf_with_pq_impl<T: ArrowFloatType + ArrowPrimitiveType>(
    centroids: &T::ArrayType,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    pq: Arc<dyn ProductQuantizer>,
    range: Option<Range<u32>>,
) -> Arc<dyn Ivf>
where
    <T as ArrowFloatType>::Native: Dot + L2 + Normalize,
{
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

pub fn new_ivf_with_sq(
    centroids: &dyn Array,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    range: Option<Range<u32>>,
) -> Result<Arc<dyn Ivf>> {
    let ivf = match centroids.data_type() {
        DataType::Float16 => new_ivf_with_sq_impl::<Float16Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            range,
        ),
        DataType::Float32 => new_ivf_with_sq_impl::<Float32Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            range,
        ),
        DataType::Float64 => new_ivf_with_sq_impl::<Float64Type>(
            centroids.as_primitive(),
            dimension,
            metric_type,
            vector_column,
            range,
        ),
        _ => {
            return Err(Error::Index {
                message: format!(
                    "new_ivf_with_sq: centroids is not expected type: {}",
                    centroids.data_type()
                ),
                location: location!(),
            })
        }
    };

    Ok(ivf)
}

fn new_ivf_with_sq_impl<T: ArrowFloatType + ArrowPrimitiveType>(
    centroids: &T::ArrayType,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    range: Option<Range<u32>>,
) -> Arc<dyn Ivf>
where
    <T as ArrowFloatType>::Native: Dot + L2 + Normalize,
{
    let mat = MatrixView::<T>::new(Arc::new(centroids.clone()), dimension);
    Arc::new(IvfImpl::<T>::new_with_sq(
        mat,
        metric_type,
        vector_column,
        range,
    ))
}

pub fn new_ivf_with_quantizer(
    centroids: &dyn Array,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    quantizer: Quantizer,
    range: Option<Range<u32>>,
) -> Result<Arc<dyn Ivf>> {
    match quantizer {
        Quantizer::Product(pq) => {
            new_ivf_with_pq(centroids, dimension, metric_type, vector_column, pq, range)
        }
        Quantizer::Scalar(_) => {
            new_ivf_with_sq(centroids, dimension, metric_type, vector_column, range)
        }
    }
}

/// IVF - IVF file partition
///
pub trait Ivf: Send + Sync + std::fmt::Debug + Transformer {
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
    fn compute_partitions(&self, data: &FixedSizeListArray) -> Result<UInt32Array>;

    /// Compute residual vector.
    ///
    /// A residual vector is `original vector - centroids`.
    ///
    /// Parameters:
    ///  - *original*: original vector.
    ///  - *partitions*: partition ID of each original vector. If not provided, it will be computed
    ///   on the flight.
    ///
    fn compute_residual(
        &self,
        original: &FixedSizeListArray,
        partitions: Option<&UInt32Array>,
    ) -> Result<FixedSizeListArray>;

    /// Find the closest partitions for the query vector.
    fn find_partitions(&self, query: &dyn Array, nprobes: usize) -> Result<UInt32Array>;
}

/// IVF - IVF file partition
///
#[derive(Debug, Clone)]
pub struct IvfImpl<T: ArrowFloatType>
where
    T::Native: Dot + L2,
{
    /// KMean model of the IVF
    ///
    /// It is a 2-D `(num_partitions * dimension)` of float32 array, 64-bit aligned via Arrow
    /// memory allocator.
    centroids: MatrixView<T>,

    /// Transform applied to each partition.
    transforms: Vec<Arc<dyn Transformer>>,

    ivf_transform: Arc<IvfTransformer<T>>,

    /// Metric type to compute pair-wise vector distance.
    metric_type: MetricType,
}

impl<T: ArrowFloatType + ArrowPrimitiveType> IvfImpl<T>
where
    <T as ArrowFloatType>::Native: Dot + L2 + Normalize,
{
    pub fn new(
        centroids: MatrixView<T>,
        metric_type: MetricType,
        vector_column: &str,
        transforms: Vec<Arc<dyn Transformer>>,
        _range: Option<Range<u32>>,
    ) -> Self {
        let ivf_transform = Arc::new(IvfTransformer::new(
            centroids.clone(),
            metric_type,
            vector_column,
        ));
        Self {
            centroids,
            metric_type,
            transforms,
            ivf_transform,
        }
    }

    fn new_with_pq(
        centroids: MatrixView<T>,
        metric_type: MetricType,
        vector_column: &str,
        pq: Arc<dyn ProductQuantizer>,
        range: Option<Range<u32>>,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> = vec![];

        let mt = if metric_type == MetricType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            metric_type
        };

        let ivf_transform = Arc::new(IvfTransformer::new(centroids.clone(), mt, vector_column));
        transforms.push(ivf_transform.clone());

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        if pq.use_residual() {
            transforms.push(Arc::new(ResidualTransform::new(
                centroids.clone(),
                PART_ID_COLUMN,
                vector_column,
            )));
            transforms.push(Arc::new(PQTransformer::new(
                pq.clone(),
                RESIDUAL_COLUMN,
                PQ_CODE_COLUMN,
            )));
        } else {
            transforms.push(Arc::new(PQTransformer::new(
                pq.clone(),
                vector_column,
                PQ_CODE_COLUMN,
            )));
        };
        Self {
            centroids: centroids.clone(),
            metric_type,
            transforms,
            ivf_transform,
        }
    }

    fn new_with_sq(
        centroids: MatrixView<T>,
        metric_type: MetricType,
        vector_column: &str,
        range: Option<Range<u32>>,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> = vec![];

        let mt = if metric_type == MetricType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            metric_type
        };

        let ivf_transform = Arc::new(IvfTransformer::new(centroids.clone(), mt, vector_column));
        transforms.push(ivf_transform.clone());

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        // For SQ we will transofrm the vector to SQ code while building the index,
        // so simply drop the vector column now.
        transforms.push(Arc::new(DropColumn::new(vector_column)));
        Self {
            centroids: centroids.clone(),
            metric_type,
            transforms,
            ivf_transform,
        }
    }

    fn dimension(&self) -> usize {
        self.centroids.ndim()
    }
}

#[async_trait]
impl<T: ArrowFloatType + ArrowPrimitiveType> Ivf for IvfImpl<T>
where
    <T as ArrowFloatType>::Native: Dot + L2 + Normalize,
{
    fn compute_partitions(&self, data: &FixedSizeListArray) -> Result<UInt32Array> {
        Ok(self.ivf_transform.compute_partitions(&data))
    }

    fn compute_residual(
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
            self.compute_partitions(original)?
        };
        let dim = original.value_length() as usize;
        let residual_arr = flatten_arr
            .as_slice()
            .chunks_exact(dim)
            .zip(part_ids.values())
            .flat_map(|(vector, &part_id)| {
                let centroid = self.centroids.row_ref(part_id as usize).unwrap();
                vector.iter().zip(centroid.iter()).map(|(&v, &c)| v - c)
            })
            .collect::<Vec<_>>();
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
        let mt = if self.metric_type == MetricType::Cosine {
            MetricType::L2
        } else {
            self.metric_type
        };
        let kmeans =
            KMeans::<T>::with_centroids(self.centroids.data().clone(), self.dimension(), mt);
        Ok(kmeans.find_partitions(query.as_slice(), nprobes)?)
    }
}

#[async_trait]
impl<T: ArrowFloatType> Transformer for IvfImpl<T>
where
    T::Native: Dot + L2,
{
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let mut batch = batch.clone();
        for transform in self.transforms.as_slice() {
            batch = transform.transform(&batch).await?;
        }
        Ok(batch)
    }
}
