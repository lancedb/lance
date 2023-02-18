// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Vector Index for Fast Approximate Nearest Neighbor (ANN) Search
//!

use std::any::Any;
use std::sync::Arc;

use crate::datatypes::Schema;
use arrow_array::{Float32Array, RecordBatch};
use async_trait::async_trait;

pub mod flat;
pub mod ivf;
mod kmeans;
mod pq;

use super::IndexParams;
use crate::{
    utils::distance::{cosine::cosine_distance, l2::l2_distance},
    Result, Error,
};

/// Query parameters for the vector indices
#[derive(Debug, Clone)]
pub struct Query {
    pub column: String,
    /// The vector to be searched.
    pub key: Arc<Float32Array>,
    /// Top k results to return.
    pub k: usize,
    /// The number of probs to load and search.
    pub nprobs: usize,

    /// If presented, apply a refine step.
    /// TODO: should we support fraction / float number here?
    pub refine_factor: Option<u32>,

    /// Distance metric type
    pub metric_type: MetricType,
}

/// Vector Index for (Approximate) Nearest Neighbor (ANN) Search.
#[async_trait]
pub trait VectorIndex {
    /// Search the vector for nearest neighbors.
    ///
    /// It returns a [RecordBatch] with Schema of:
    ///
    /// ```
    /// use arrow_schema::{Schema, Field, DataType};
    ///
    /// Schema::new(vec![
    ///   Field::new("_rowid", DataType::UInt64, false),
    ///   Field::new("score", DataType::Float32, false),
    /// ]);
    /// ```
    ///
    /// *WARNINGS*:
    ///  - Only supports `f32` now. Will add f64/f16 later.
    async fn search(&self, query: &Query) -> Result<RecordBatch>;
}

/// Distance metrics type.
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    L2,
    Cosine,
}

impl MetricType {
    pub fn func(
        &self,
    ) -> Arc<dyn Fn(&Float32Array, &Float32Array, usize) -> Result<Arc<Float32Array>> + Send + Sync>
    {
        match self {
            Self::L2 => Arc::new(l2_distance),
            Self::Cosine => Arc::new(cosine_distance),
        }
    }
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::L2 => "l2",
                Self::Cosine => "cosine",
            }
        )
    }
}

impl From<super::pb::VectorMetricType> for MetricType {
    fn from(proto: super::pb::VectorMetricType) -> Self {
        match proto {
            super::pb::VectorMetricType::L2 => Self::L2,
            super::pb::VectorMetricType::Cosine => Self::Cosine,
        }
    }
}

impl From<MetricType> for super::pb::VectorMetricType {
    fn from(mt: MetricType) -> Self {
        match mt {
            MetricType::L2 => Self::L2,
            MetricType::Cosine => Self::Cosine,
        }
    }
}

/// The parameters to build vector index.
pub struct VectorIndexParams {
    // This is hard coded for IVF_PQ for now. Can refactor later to support more.
    /// The number of IVF partitions
    pub num_partitions: u32,

    /// the number of bits to present the centroids used in PQ.
    pub nbits: u8,

    /// the number of sub vectors used in PQ.
    pub num_sub_vectors: u32,

    /// Vector distance metrics type.
    pub metrics_type: MetricType,
}

impl VectorIndexParams {
    /// Create index parameters for `IVF_PQ` index.
    ///
    /// Parameters
    ///
    ///  - `num_partitions`: the number of IVF partitions.
    ///  - `nbits`: the number of bits to present the centroids used in PQ. Can only be `8` for now.
    ///  - `num_sub_vectors`: the number of sub vectors used in PQ.
    pub fn ivf_pq(
        num_partitions: u32,
        nbits: u8,
        num_sub_vectors: u32,
        metrics_type: MetricType,
    ) -> Self {
        Self {
            num_partitions,
            nbits,
            num_sub_vectors,
            metrics_type,
        }
    }
}

impl Default for VectorIndexParams {
    fn default() -> Self {
        Self {
            num_partitions: 32,
            nbits: 8,
            num_sub_vectors: 16,
            metrics_type: MetricType::L2,
        }
    }
}

impl IndexParams for VectorIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
