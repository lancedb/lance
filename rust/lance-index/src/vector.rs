// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Index
//!

use std::any::Any;
use std::fmt::Debug;
use std::{collections::HashMap, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use ivf::storage::IvfModel;
use lance_core::{Result, ROW_ID_FIELD};
use lance_io::object_store::ObjectStore;
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use object_store::path::Path;
use quantizer::{QuantizationType, Quantizer};
use std::sync::LazyLock;
use v3::subindex::SubIndexType;

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
use crate::metrics::MetricsCollector;
use crate::{prefilter::PreFilter, Index};
pub use residual::RESIDUAL_COLUMN;

// TODO: Make these crate private once the migration from lance to lance-index is done.
pub const DIST_COL: &str = "_distance";
pub const DISTANCE_TYPE_KEY: &str = "distance_type";
pub const INDEX_UUID_COLUMN: &str = "__index_uuid";
pub const PART_ID_COLUMN: &str = "__ivf_part_id";
pub const PQ_CODE_COLUMN: &str = "__pq_code";
pub const SQ_CODE_COLUMN: &str = "__sq_code";
pub const LOSS_METADATA_KEY: &str = "_loss";

pub static VECTOR_RESULT_SCHEMA: LazyLock<arrow_schema::SchemaRef> = LazyLock::new(|| {
    arrow_schema::SchemaRef::new(arrow_schema::Schema::new(vec![
        Field::new(DIST_COL, arrow_schema::DataType::Float32, false),
        ROW_ID_FIELD.clone(),
    ]))
});

pub static PART_ID_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(PART_ID_COLUMN, arrow_schema::DataType::UInt32, true)
});

/// Query parameters for the vector indices
#[derive(Debug, Clone)]
pub struct Query {
    /// The column to be searched.
    pub column: String,

    /// The vector to be searched.
    pub key: ArrayRef,

    /// Top k results to return.
    pub k: usize,

    /// The lower bound (inclusive) of the distance to be searched.
    pub lower_bound: Option<f32>,

    /// The upper bound (exclusive) of the distance to be searched.
    pub upper_bound: Option<f32>,

    /// The minimum number of probes to load and search.  More partitions
    /// will only be loaded if we have not found k results.
    pub minimum_nprobes: usize,

    /// The maximum number of probes to load and search.  If not set then
    /// ALL partitions will be searched, if needed, to satisfy k results.
    pub maximum_nprobes: Option<usize>,

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
///
/// Vector indices are often built as a chain of indices.  For example, IVF -> PQ
/// or IVF -> HNSW -> SQ.
///
/// We use one trait for both the top-level and the sub-indices.  Typically the top-level
/// search is a partition-aware search and all sub-indices are whole-index searches.
#[async_trait]
#[allow(clippy::redundant_pub_crate)]
pub trait VectorIndex: Send + Sync + std::fmt::Debug + Index {
    /// Search entire index for k nearest neighbors.
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
    /// not relevant to the query. For example, it removes deleted rows or rows that
    /// do not match a user-provided filter.
    async fn search(
        &self,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch>;

    /// Find partitions that may contain nearest neighbors.
    ///
    /// If maximum_nprobes is set then this method will return the partitions
    /// that are most likely to contain the nearest neighbors (e.g. the closest
    /// partitions to the query vector).
    ///
    /// The results should be in sorted order from closest to farthest.
    fn find_partitions(&self, query: &Query) -> Result<UInt32Array>;

    /// Get the total number of partitions in the index.
    fn total_partitions(&self) -> usize;

    /// Search a single partition for nearest neighbors.
    ///
    /// This method should return the same results as [`VectorIndex::search`] method except
    /// that it will only search a single partition.
    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch>;

    /// If the index is loadable by IVF, so it can be a sub-index that
    /// is loaded on demand by IVF.
    fn is_loadable(&self) -> bool;

    /// Use residual vector to search.
    fn use_residual(&self) -> bool;

    // async fn append(&self, batches: Vec<RecordBatch>) -> Result<()>;
    // async fn merge(&self, indices: Vec<Arc<dyn VectorIndex>>) -> Result<()>;

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

    // for IVF only
    async fn partition_reader(
        &self,
        _partition_id: usize,
        _with_vector: bool,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SendableRecordBatchStream> {
        unimplemented!("only for IVF")
    }

    // for SubIndex only
    async fn to_batch_stream(&self, with_vector: bool) -> Result<SendableRecordBatchStream>;

    fn num_rows(&self) -> u64;

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
    async fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()>;

    /// Remap the index according to mapping
    ///
    /// write the remapped index to the index_dir
    /// this is available for only v3 index
    async fn remap_to(
        self: Arc<Self>,
        _store: ObjectStore,
        _mapping: &HashMap<u64, Option<u64>>,
        _column: String,
        _index_dir: Path,
    ) -> Result<()> {
        unimplemented!("only for v3 index")
    }

    /// The metric type of this vector index.
    fn metric_type(&self) -> DistanceType;

    fn ivf_model(&self) -> &IvfModel;
    fn quantizer(&self) -> Quantizer;

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType);
}

// it can be an IVF index or a partition of IVF index
pub trait VectorIndexCacheEntry: Debug + Send + Sync + DeepSizeOf {
    fn as_any(&self) -> &dyn Any;
}
