// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{RecordBatch, StructArray};
use lance_core::Result;
use num_traits::Num;
use roaring::RoaringBitmap;

use super::storage::VectorStore;

/// A sub index for IVF index
pub trait IvfSubIndex: Send + Sync + Sized {
    type QueryParams: Default;

    fn name(&self) -> &str;

    /// Search the sub index for nearest neighbors.
    /// # Arguments:
    /// * `query` - The query vector
    /// * `k` - The number of nearest neighbors to return
    /// * `params` - The query parameters
    /// * `pre_filter_bitmap` - The pre filter bitmap indicating which vectors to skip
    fn search<T: Num>(
        &self,
        query: &[T],
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        pre_filter_bitmap: Option<RoaringBitmap>,
    ) -> Result<RecordBatch>;

    /// Load the sub index from a struct array with a single row
    fn load(data: StructArray) -> Result<Self>;

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index_vectors(&self, storage: &impl VectorStore) -> Result<()>;

    /// Encode the sub index into a struct array
    fn to_array(&self) -> Result<StructArray>;
}
