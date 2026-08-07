// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use arrow_array::{ArrayRef, RecordBatch};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

use crate::metrics::MetricsCollector;
use crate::vector::graph::OrderedNode;
use crate::vector::storage::{QueryResidual, QueryScratch, VectorStore};
use crate::vector::{flat, hnsw};
use crate::{prefilter::PreFilter, vector::Query};

/// Optional external accelerator for building an IVF sub-index.
///
/// The accelerator only changes how the in-memory graph is constructed. Lance
/// still owns quantization, persistence, and query execution.
#[derive(Debug, Clone)]
pub enum SubIndexBuildAccelerator {
    /// Build a CAGRA graph through the cuVS C API and import it as HNSW.
    Cagra(CagraBuildAccelerator),
}

impl SubIndexBuildAccelerator {
    /// Create a cuVS CAGRA accelerator from an absolute `libcuvs_c` path.
    pub fn cagra(library_path: impl Into<String>) -> Self {
        Self::Cagra(CagraBuildAccelerator {
            library_path: library_path.into(),
            is_disabled: Arc::new(AtomicBool::new(false)),
            is_required: false,
        })
    }

    /// Create a cuVS CAGRA accelerator that does not fall back to CPU.
    pub fn cagra_required(library_path: impl Into<String>) -> Self {
        Self::Cagra(CagraBuildAccelerator {
            library_path: library_path.into(),
            is_disabled: Arc::new(AtomicBool::new(false)),
            is_required: true,
        })
    }
}

/// Shared state for one cuVS CAGRA index build.
///
/// A failing cuVS call disables acceleration for the remaining IVF partitions,
/// avoiding repeated loader or CUDA failures before their CPU fallback.
#[derive(Debug, Clone)]
pub struct CagraBuildAccelerator {
    library_path: String,
    is_disabled: Arc<AtomicBool>,
    is_required: bool,
}

impl CagraBuildAccelerator {
    pub(crate) fn library_path(&self) -> &str {
        &self.library_path
    }

    pub(crate) fn is_disabled(&self) -> bool {
        self.is_disabled.load(Ordering::Relaxed)
    }

    pub(crate) fn is_required(&self) -> bool {
        self.is_required
    }

    /// Disable the accelerator and return whether this call changed its state.
    pub(crate) fn disable(&self) -> bool {
        !self.is_disabled.swap(true, Ordering::Relaxed)
    }
}

/// A sub index for IVF index
pub trait IvfSubIndex: Send + Sync + Debug + DeepSizeOf {
    type QueryParams: Send + Sync + for<'a> From<&'a Query>;
    type BuildParams: Clone + Send + Sync;

    /// Load the sub index from a record batch with a single row
    fn load(data: RecordBatch) -> Result<Self>
    where
        Self: Sized;

    fn name() -> &'static str;

    fn metadata_key() -> &'static str;

    /// Return the schema of the sub index
    fn schema() -> arrow_schema::SchemaRef;

    /// The subset of [`Self::schema`] that [`Self::load`] actually reads.
    ///
    /// Index files always carry the full `schema()`, so narrowing the read is
    /// purely a storage optimization: it keeps write-only columns from being
    /// fetched, without changing what is written. `None` reads every column.
    fn read_columns() -> Option<&'static [&'static str]> {
        None
    }

    /// Search the sub index for nearest neighbors.
    /// # Arguments:
    /// * `query` - The query vector
    /// * `k` - The number of nearest neighbors to return
    /// * `params` - The query parameters
    /// * `prefilter` - The prefilter object indicating which vectors to skip
    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch>;

    /// Search the sub-index, reusing scratch buffers owned by the caller.
    #[allow(clippy::too_many_arguments)]
    fn search_with_scratch(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
        _residual: Option<QueryResidual<'_>>,
        _scratch: &mut QueryScratch,
    ) -> Result<RecordBatch> {
        self.search(query, k, params, storage, prefilter, metrics)
    }

    /// Return true if this sub-index can accumulate candidates into a caller-owned heap.
    fn supports_global_topk_heap() -> bool {
        false
    }

    /// Search this partition and accumulate candidates into a caller-owned top-k heap.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_topk(
        &self,
        _query: ArrayRef,
        _k: usize,
        _params: Self::QueryParams,
        _storage: &impl VectorStore,
        _prefilter: Arc<dyn PreFilter>,
        _heap: &mut BinaryHeap<OrderedNode<u64>>,
        _metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        unimplemented!("global top-k heap search is not supported for this sub-index")
    }

    /// Search this partition and accumulate candidates into a caller-owned top-k heap,
    /// reusing scratch buffers owned by the caller.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_topk_with_scratch(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        _residual: Option<QueryResidual<'_>>,
        _scratch: &mut QueryScratch,
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        self.accumulate_topk(query, k, params, storage, prefilter, heap, metrics)
    }

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index_vectors(storage: &impl VectorStore, params: Self::BuildParams) -> Result<Self>
    where
        Self: Sized;

    /// Build with an optional external accelerator.
    ///
    /// Sub-index implementations that do not support the requested accelerator
    /// retain their normal CPU build behavior.
    fn index_vectors_with_accelerator(
        storage: &impl VectorStore,
        params: Self::BuildParams,
        _accelerator: &SubIndexBuildAccelerator,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Self::index_vectors(storage, params)
    }

    fn remap(&self, mapping: &RowAddrRemap, store: &impl VectorStore) -> Result<Self>
    where
        Self: Sized;

    /// Encode the sub index into a record batch
    fn to_batch(&self) -> Result<RecordBatch>;
}

#[derive(Debug, Clone, Copy)]
pub enum SubIndexType {
    Flat,
    Hnsw,
}

impl std::fmt::Display for SubIndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flat => write!(f, "{}", flat::index::FlatIndex::name()),
            Self::Hnsw => write!(f, "{}", hnsw::builder::HNSW::name()),
        }
    }
}

impl TryFrom<&str> for SubIndexType {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "FLAT" => Ok(Self::Flat),
            "HNSW" => Ok(Self::Hnsw),
            _ => Err(Error::index(format!("unknown sub index type {}", value))),
        }
    }
}
