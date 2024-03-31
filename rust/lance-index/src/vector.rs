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

//! Vector Index
//!

use arrow_array::ArrayRef;
use lance_linalg::distance::MetricType;

pub mod bq;
pub mod flat;
pub mod graph;
pub mod hnsw;
pub mod ivf;
pub mod kmeans;
pub mod pq;
pub mod residual;
pub mod sq;
pub mod transform;
pub mod utils;

// TODO: Make these crate private once the migration from lance to lance-index is done.
pub const PQ_CODE_COLUMN: &str = "__pq_code";
pub const PART_ID_COLUMN: &str = "__ivf_part_id";
pub const DIST_COL: &str = "_distance";

use super::pb;
pub use residual::RESIDUAL_COLUMN;

/// Query parameters for the vector indices
#[derive(Debug, Clone)]
pub struct Query {
    /// The column to be searched.
    pub column: String,

    /// The vector to be searched.
    pub key: ArrayRef,

    /// Top k results to return.
    pub k: usize,

    /// The number of probes to load and search.
    pub nprobes: usize,

    /// If presented, apply a refine step.
    /// TODO: should we support fraction / float number here?
    pub refine_factor: Option<u32>,

    /// Distance metric type
    pub metric_type: MetricType,

    /// Whether to use an ANN index if available
    pub use_index: bool,
}

impl From<pb::VectorMetricType> for MetricType {
    fn from(proto: pb::VectorMetricType) -> Self {
        match proto {
            pb::VectorMetricType::L2 => Self::L2,
            pb::VectorMetricType::Cosine => Self::Cosine,
            pb::VectorMetricType::Dot => Self::Dot,
        }
    }
}

impl From<MetricType> for pb::VectorMetricType {
    fn from(mt: MetricType) -> Self {
        match mt {
            MetricType::L2 => Self::L2,
            MetricType::Cosine => Self::Cosine,
            MetricType::Dot => Self::Dot,
        }
    }
}
