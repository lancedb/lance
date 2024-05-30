// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use lance_core::Result;
use num_traits::Num;

use super::storage::VectorStore;

/// A prefilter that can be used to skip vectors during search
///
/// Note: there is a `struct PreFilter` in `lance`. However we can't depend on `lance` in `lance-index`
/// because it would create a circular dependency.
///
/// By defining a trait here, we can implement the trait for `lance::PreFilter`
/// and not have the circular dependency
pub trait PreFilter {
    fn no_prefilter() -> Arc<NoPreFilter> {
        Arc::new(NoPreFilter {})
    }

    fn should_drop(&self, id: u64) -> bool;
}

/// A prefilter that does not skip any vectors
pub struct NoPreFilter {}

impl PreFilter for NoPreFilter {
    fn should_drop(&self, _id: u64) -> bool {
        false
    }
}

/// A sub index for IVF index
pub trait IvfSubIndex: Send + Sync + Sized {
    type QueryParams: Default;
    type BuildParams: Clone;

    fn name(&self) -> &str;

    /// Search the sub index for nearest neighbors.
    /// # Arguments:
    /// * `query` - The query vector
    /// * `k` - The number of nearest neighbors to return
    /// * `params` - The query parameters
    /// * `prefilter` - The prefilter object indicating which vectors to skip
    fn search<T: Num>(
        &self,
        query: &[T],
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<impl PreFilter>,
    ) -> Result<RecordBatch>;

    /// Load the sub index from a record batch with a single row
    fn load(data: RecordBatch) -> Result<Self>;

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index_vectors(&self, storage: &impl VectorStore, params: Self::BuildParams) -> Result<()>;

    /// Return the schema of the sub index
    fn schema(&self) -> arrow_schema::SchemaRef;

    /// Encode the sub index into a record batch
    fn to_batch(&self) -> Result<RecordBatch>;
}
