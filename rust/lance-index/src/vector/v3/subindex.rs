// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, StructArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use itertools::Itertools;
use lance_core::{Result, ROW_ID_FIELD};
use num_traits::Num;
use roaring::RoaringBitmap;

use crate::vector::v3::storage::DistCalculator;
use crate::vector::{
    graph::{OrderedFloat, OrderedNode},
    DIST_COL,
};

use super::storage::VectorStore;

/// A sub index for IVF index
pub trait IvfSubIndex: Send + Sync + Sized {
    type QueryParams: Default;

    fn index_name(&self) -> &str;

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

struct FlatIndex {}

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
        pre_filter_bitmap: Option<RoaringBitmap>,
    ) -> Result<RecordBatch> {
        let dist_calc = storage.dist_calculator_from_native(query);
        let (row_ids, dists): (Vec<u64>, Vec<f32>) = (0..storage.len())
            .filter(|id| {
                let should_drop = pre_filter_bitmap
                    .as_ref()
                    .map(|bitmap| bitmap.contains(*id as u32));
                let should_drop = should_drop.unwrap_or(false);
                !should_drop
            })
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
