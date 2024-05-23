// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, StructArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use itertools::Itertools;
use lance_core::{Result, ROW_ID_FIELD};
use num_traits::Num;

use crate::vector::v3::storage::DistCalculator;
use crate::vector::{
    graph::{OrderedFloat, OrderedNode},
    DIST_COL,
};

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

    fn index_name(&self) -> &str;

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

    // check if the builder supports the metadata schema requested
    fn supports_metadata(&self, _schema: SchemaRef) -> bool {
        false
    }

    /// Load the sub index from a struct array with a single row
    fn load(data: StructArray) -> Result<Self>;

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index(&self, storage: &impl VectorStore) -> Result<()>;

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
    fn build(&self, storage: &impl VectorStore) -> Result<StructArray> {
        self.index(storage)?;
        self.to_array()
    }
}

/// A Flat index is any index that stores no metadata, and
/// during query, it simply scans over the storage and returns the top k results
pub struct FlatIndex {}

lazy_static::lazy_static! {
    static ref ANN_SEARCH_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ]));
}

impl IvfSubIndex for FlatIndex {
    type QueryParams = ();

    fn index_name(&self) -> &str {
        "FLAT"
    }

    fn search<T: Num>(
        &self,
        query: &[T],
        k: usize,
        _params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<impl PreFilter>,
    ) -> Result<RecordBatch> {
        let dist_calc = storage.dist_calculator_from_native(query);
        let (row_ids, dists): (Vec<u64>, Vec<f32>) = (0..storage.len())
            .filter(|&id| !prefilter.should_drop(storage.row_ids()[id]))
            .map(|id| OrderedNode {
                id: id as u32,
                dist: OrderedFloat(dist_calc.distance(id as u32)),
            })
            .sorted_unstable()
            .take(k)
            .map(
                |OrderedNode {
                     id,
                     dist: OrderedFloat(dist),
                 }| (storage.row_ids()[id as usize], dist),
            )
            .unzip();

        let (row_ids, dists) = (UInt64Array::from(row_ids), Float32Array::from(dists));

        Ok(RecordBatch::try_new(
            ANN_SEARCH_SCHEMA.clone(),
            vec![Arc::new(dists), Arc::new(row_ids)],
        )?)
    }

    fn load(_: StructArray) -> Result<Self> {
        Ok(Self {})
    }

    fn index(&self, _: &impl VectorStore) -> Result<()> {
        Ok(())
    }

    fn to_array(&self) -> Result<StructArray> {
        Ok(StructArray::from(vec![]))
    }
}
