// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{RecordBatch, StructArray};
use arrow_schema::SchemaRef;
use lance_core::Result;
use num_traits::Num;
use roaring::RoaringBitmap;

use super::storage::VectorStore;

/// A sub index for IVF index
pub trait IvfSubIndex<T: Num, Store: VectorStore>: Send + Sync + Sized {
    type QueryParams: Default;

    fn index_name(&self) -> &str;

    /// Search the sub index for nearest neighbors.
    /// # Arguments:
    /// * `query` - The query vector
    /// * `k` - The number of nearest neighbors to return
    /// * `params` - The query parameters
    /// * `pre_filter_bitmap` - The pre filter bitmap indicating which vectors to skip
    fn search(
        &self,
        query: &[T],
        k: usize,
        params: Self::QueryParams,
        storage: &Store,
        pre_filter_bitmap: Option<RoaringBitmap>,
    ) -> Result<RecordBatch>;

    // check if the builder supports the metadata schema requested
    fn supports_metadata(&self, _schema: SchemaRef) -> bool {
        false
    }

    /// Load the sub index from a struct array with a single row
    fn load(data: StructArray) -> Result<Self>;

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index(&self, storage: &Store) -> Result<()>;

    /// Turn the sub index into a struct array
    fn to_array(&self) -> Result<StructArray>;

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    /// The returned value is a struct array with a SINGLE ROW and always have the same schema
    ///
    /// It is recommended to not implement this method and use the default implementation
    /// The implementation should implement `index` and `to_array` method instead, as they make the index
    /// appendable
    ///
    /// NOTE: we use a single record batch to avoid the need of async operations to read the data
    ///
    /// The roundtrip looks like
    /// IvfSubIndexBuilder.index(data).to_array()
    fn build(&self, storage: &Store) -> Result<StructArray> {
        self.index(storage)?;
        self.to_array()
    }
}
