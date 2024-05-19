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
use snafu::{location, Location};

pub use builder::IvfBuildParams;
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::{
    distance::{DistanceType, Dot, MetricType, Normalize, L2},
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
use super::residual::compute_residual;
use super::transform::DropColumn;
use super::{PART_ID_COLUMN, PQ_CODE_COLUMN, RESIDUAL_COLUMN};

pub mod builder;
pub mod shuffler;
pub mod storage;
mod transform;

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
    centroids: Arc<FixedSizeListArray>,
    dimension: usize,
    metric_type: DistanceType,
    transforms: Vec<Arc<dyn Transformer>>,
    range: Option<Range<u32>>,
) -> Ivf {
    Ivf::new(centroids, metric_type, "", transforms, range)
}

pub fn new_ivf_with_pq(
    centroids: Arc<FixedSizeListArray>,
    distance_type: MetricType,
    vector_column: &str,
    pq: Arc<dyn ProductQuantizer>,
    range: Option<Range<u32>>,
) -> Result<Ivf> {
    Ok(Ivf::new_with_pq(
        centroids,
        distance_type,
        vector_column,
        pq,
        range,
    ))
}

pub fn new_ivf_with_sq(
    centroids: Arc<FixedSizeListArray>,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    range: Option<Range<u32>>,
) -> Result<Arc<Ivf>> {
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
) -> Arc<Ivf>
where
    <T as ArrowFloatType>::Native: Dot + L2 + Normalize,
{
    let mat = MatrixView::<T>::new(Arc::new(centroids.clone()), dimension);
    Arc::new(Ivf::new_with_sq(mat, metric_type, vector_column, range))
}

pub fn new_ivf_with_quantizer(
    centroids: Arc<FixedSizeListArray>,
    dimension: usize,
    metric_type: MetricType,
    vector_column: &str,
    quantizer: Quantizer,
    range: Option<Range<u32>>,
) -> Result<Ivf> {
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
#[derive(Debug)]
pub struct Ivf {
    /// KMean model of the IVF
    ///
    /// It is a 2-D `(num_partitions * dimension)` of floating array.
    centroids: Arc<FixedSizeListArray>,

    /// Transform applied to each partition.
    transforms: Vec<Arc<dyn Transformer>>,

    ivf_transform: Arc<IvfTransformer>,

    /// Metric type to compute pair-wise vector distance.
    distance_type: DistanceType,
}

impl Ivf {
    pub fn new(
        centroids: Arc<FixedSizeListArray>,
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
            distance_type: metric_type,
            transforms,
            ivf_transform,
        }
    }

    /// Create a IVF_PQ struct.
    pub fn with_pq(
        centroids: Arc<FixedSizeListArray>,
        distance_type: DistanceType,
        vector_column: &str,
        pq: Arc<dyn ProductQuantizer>,
        range: Option<Range<u32>>,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> = vec![];

        let mt = if distance_type == MetricType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            distance_type
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
            distance_type,
            transforms,
            ivf_transform,
        }
    }

    fn new_with_sq(
        centroids: Arc<FixedSizeListArray>,
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
            centroids,
            distance_type: metric_type,
            transforms,
            ivf_transform,
        }
    }

    fn dimension(&self) -> usize {
        self.centroids.ndim()
    }

    fn compute_partitions(&self, data: &FixedSizeListArray) -> Result<UInt32Array> {
        Ok(self.ivf_transform.compute_partitions(data))
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
        let mt = if self.distance_type == MetricType::Cosine {
            MetricType::L2
        } else {
            self.distance_type
        };
        let kmeans =
            KMeans::<T>::with_centroids(self.centroids.data().clone(), self.dimension(), mt);
        Ok(kmeans.find_partitions(query.as_slice(), nprobes)?)
    }
}

impl Transformer for Ivf {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let mut batch = batch.clone();
        for transform in self.transforms.as_slice() {
            batch = transform.transform(&batch)?;
        }
        Ok(batch)
    }
}
