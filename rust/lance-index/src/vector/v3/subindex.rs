// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use deepsize::DeepSizeOf;
use lance_core::Result;

use crate::vector::storage::VectorStore;
use crate::{prefilter::PreFilter, vector::Query};

pub const SUB_INDEX_METADATA_KEY: &str = "sub_index_metadata";

/// A sub index for IVF index
pub trait IvfSubIndex: Send + Sync + DeepSizeOf {
    type QueryParams: Send + Sync + Default + for<'a> From<&'a Query>;
    type BuildParams: Clone;

    /// Load the sub index from a record batch with a single row
    fn load(data: RecordBatch) -> Result<Self>
    where
        Self: Sized;

    fn use_residual() -> bool;

    fn name(&self) -> &str;

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
    ) -> Result<RecordBatch>;

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index_vectors(&self, storage: &impl VectorStore, params: Self::BuildParams) -> Result<()>;

    /// Return the schema of the sub index
    fn schema(&self) -> arrow_schema::SchemaRef;

    /// Encode the sub index into a record batch
    fn to_batch(&self) -> Result<RecordBatch>;
}
