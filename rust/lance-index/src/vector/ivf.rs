// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File Index

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array};

pub use builder::IvfBuildParams;
use lance_core::Result;
use lance_linalg::{
    distance::{DistanceType, MetricType},
    kmeans::{compute_partitions_arrow_array, kmeans_find_partitions_arrow_array},
};

use crate::vector::ivf::transform::PartitionTransformer;
use crate::vector::{
    pq::{transform::PQTransformer, ProductQuantizer},
    residual::ResidualTransform,
    transform::Transformer,
};

use super::{quantizer::Quantizer, residual::compute_residual};
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
pub fn new_ivf_transformer(
    centroids: FixedSizeListArray,
    metric_type: DistanceType,
    transforms: Vec<Arc<dyn Transformer>>,
) -> IvfTransformer {
    IvfTransformer::new(centroids, metric_type, transforms)
}

pub fn new_ivf_transformer_with_quantizer(
    centroids: FixedSizeListArray,
    metric_type: MetricType,
    vector_column: &str,
    quantizer: Quantizer,
    range: Option<Range<u32>>,
) -> Result<IvfTransformer> {
    match quantizer {
        Quantizer::Flat(_) => Ok(IvfTransformer::new_flat(
            centroids,
            metric_type,
            vector_column,
            range,
        )),
        Quantizer::Product(pq) => Ok(IvfTransformer::with_pq(
            centroids,
            metric_type,
            vector_column,
            pq,
            range,
        )),
        Quantizer::Scalar(_) => Ok(IvfTransformer::with_sq(
            centroids,
            metric_type,
            vector_column,
            range,
        )),
    }
}

/// IVF - IVF file partition
///
#[derive(Debug)]
pub struct IvfTransformer {
    /// Centroids of a cluster algorithm, to run IVF.
    ///
    /// It is a 2-D `(num_partitions * dimension)` of floating array.
    centroids: FixedSizeListArray,

    /// Transform applied to each partition.
    transforms: Vec<Arc<dyn Transformer>>,

    /// Metric type to compute pair-wise vector distance.
    distance_type: DistanceType,
}

impl IvfTransformer {
    /// Create a new Ivf model.
    pub fn new(
        centroids: FixedSizeListArray,
        metric_type: MetricType,
        transforms: Vec<Arc<dyn Transformer>>,
    ) -> Self {
        Self {
            centroids,
            distance_type: metric_type,
            transforms,
        }
    }

    pub fn new_flat(
        centroids: FixedSizeListArray,
        distance_type: DistanceType,
        vector_column: &str,
        range: Option<Range<u32>>,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> = vec![];

        let dt = if distance_type == DistanceType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            distance_type
        };

        let ivf_transform = Arc::new(PartitionTransformer::new(
            centroids.clone(),
            dt,
            vector_column,
        ));
        transforms.push(ivf_transform.clone());

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        Self {
            centroids,
            distance_type,
            transforms,
        }
    }

    /// Create a IVF_PQ struct.
    pub fn with_pq(
        centroids: FixedSizeListArray,
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

        let partition_transform = Arc::new(PartitionTransformer::new(
            centroids.clone(),
            mt,
            vector_column,
        ));
        transforms.push(partition_transform.clone());

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
            centroids,
            distance_type,
            transforms,
        }
    }

    fn with_sq(
        centroids: FixedSizeListArray,
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

        let partition_transformer = Arc::new(PartitionTransformer::new(
            centroids.clone(),
            mt,
            vector_column,
        ));
        transforms.push(partition_transformer.clone());

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        Self {
            centroids,
            distance_type: metric_type,
            transforms,
        }
    }

    #[inline]
    pub fn compute_residual(&self, data: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        compute_residual(&self.centroids, data, Some(self.distance_type), None)
    }

    #[inline]
    pub fn compute_partitions(&self, data: &FixedSizeListArray) -> Result<UInt32Array> {
        Ok(compute_partitions_arrow_array(&self.centroids, data, self.distance_type)?.into())
    }

    pub fn find_partitions(&self, query: &dyn Array, nprobes: usize) -> Result<UInt32Array> {
        Ok(kmeans_find_partitions_arrow_array(
            &self.centroids,
            query,
            nprobes,
            self.distance_type,
        )?)
    }
}

impl Transformer for IvfTransformer {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let mut batch = batch.clone();
        for transform in self.transforms.as_slice() {
            batch = transform.transform(&batch)?;
        }
        Ok(batch)
    }
}
