// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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
pub mod quantizer;
pub mod residual;
pub mod sq;
pub mod transform;
pub mod utils;

// TODO: Make these crate private once the migration from lance to lance-index is done.
pub const PQ_CODE_COLUMN: &str = "__pq_code";
pub const SQ_CODE_COLUMN: &str = "__sq_code";
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

    /// The number of candidates to reserve while searching.
    /// this is an optional parameter for HNSW related index types.
    pub ef: Option<usize>,

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
