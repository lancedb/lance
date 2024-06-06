// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Index
//!

use std::{collections::HashMap, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use async_trait::async_trait;
use lance_core::{Result, ROW_ID_FIELD};
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use lazy_static::lazy_static;

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
pub mod storage;
pub mod transform;
pub mod utils;
pub mod v3;

use super::pb;
use crate::{prefilter::PreFilter, Index};
pub use residual::RESIDUAL_COLUMN;

// TODO: Make these crate private once the migration from lance to lance-index is done.
pub const DIST_COL: &str = "_distance";
pub const DISTANCE_TYPE_KEY: &str = "distance_type";
pub const INDEX_UUID_COLUMN: &str = "__index_uuid";
pub const PART_ID_COLUMN: &str = "__ivf_part_id";
pub const PQ_CODE_COLUMN: &str = "__pq_code";
pub const SQ_CODE_COLUMN: &str = "__sq_code";

lazy_static! {
    pub static ref VECTOR_RESULT_SCHEMA: arrow_schema::SchemaRef =
        arrow_schema::SchemaRef::new(arrow_schema::Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new(DIST_COL, arrow_schema::DataType::Float32, false),
        ]));
}

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
    pub metric_type: DistanceType,

    /// Whether to use an ANN index if available
    pub use_index: bool,
}

impl From<pb::VectorMetricType> for DistanceType {
    fn from(proto: pb::VectorMetricType) -> Self {
        match proto {
            pb::VectorMetricType::L2 => Self::L2,
            pb::VectorMetricType::Cosine => Self::Cosine,
            pb::VectorMetricType::Dot => Self::Dot,
            pb::VectorMetricType::Hamming => Self::Hamming,
        }
    }
}

impl From<DistanceType> for pb::VectorMetricType {
    fn from(mt: DistanceType) -> Self {
        match mt {
            DistanceType::L2 => Self::L2,
            DistanceType::Cosine => Self::Cosine,
            DistanceType::Dot => Self::Dot,
            DistanceType::Hamming => Self::Hamming,
        }
    }
}

/// Vector Index for (Approximate) Nearest Neighbor (ANN) Search.
#[async_trait]
#[allow(clippy::redundant_pub_crate)]
pub trait VectorIndex: Send + Sync + std::fmt::Debug + Index {
    /// Search the vector for nearest neighbors.
    ///
    /// It returns a [RecordBatch] with Schema of:
    ///
    /// ```
    /// use arrow_schema::{Schema, Field, DataType};
    ///
    /// Schema::new(vec![
    ///   Field::new("_rowid", DataType::UInt64, true),
    ///   Field::new("_distance", DataType::Float32, false),
    /// ]);
    /// ```
    ///
    /// The `pre_filter` argument is used to filter out row ids that we know are
    /// not relevant to the query. For example, it removes deleted rows.
    ///
    /// *WARNINGS*:
    ///  - Only supports `f32` now. Will add f64/f16 later.
    async fn search(&self, query: &Query, pre_filter: Arc<dyn PreFilter>) -> Result<RecordBatch>;

    /// If the index is loadable by IVF, so it can be a sub-index that
    /// is loaded on demand by IVF.
    fn is_loadable(&self) -> bool;

    /// Use residual vector to search.
    fn use_residual(&self) -> bool;

    /// If the index can be remapped return Ok.  Else return an error
    /// explaining why not
    fn check_can_remap(&self) -> Result<()>;

    /// Load the index from the reader on-demand.
    async fn load(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>>;

    /// Load the partition from the reader on-demand.
    async fn load_partition(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
        _partition_id: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        self.load(reader, offset, length).await
    }

    /// Return the IDs of rows in the index.
    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_>;

    /// Remap the index according to mapping
    ///
    /// Each item in mapping describes an old row id -> new row id
    /// pair.  If old row id -> None then that row id has been
    /// deleted and can be removed from the index.
    ///
    /// If an old row id is not in the mapping then it should be
    /// left alone.
    fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()>;

    /// The metric type of this vector index.
    fn metric_type(&self) -> DistanceType;
}
