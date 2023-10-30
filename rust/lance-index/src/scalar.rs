// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Scalar indices for metadata search & filtering

use std::{ops::Bound, sync::Arc};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::Schema;
use async_trait::async_trait;
use datafusion_common::scalar::ScalarValue;

use lance_core::Result;

pub mod btree;
pub mod flat;
pub mod lance;

#[async_trait]
pub trait IndexWriter: Send {
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64>;
    async fn finish(&mut self) -> Result<()>;
}

#[async_trait]
pub trait IndexReader: Send + Sync {
    async fn read_record_batch(&self, offset: u64) -> Result<RecordBatch>;
}

#[async_trait]
pub trait IndexStore: std::fmt::Debug + Send + Sync {
    async fn new_index_file(&self, name: &str, schema: Arc<Schema>)
        -> Result<Box<dyn IndexWriter>>;

    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>>;
}

/// A query that a scalar index can satisfy
///
/// This is a subset of expression operators that is often referred to as the
/// "sargable" operators
///
/// Note that negation is not included.  Negation should be applied later.  For
/// example, to invert an equality query (e.g. all rows where the value is not 7)
/// you can grab all rows where the value = 7 and then do an inverted take (or use
/// a block list instead of an allow list for prefiltering)
#[derive(Debug)]
pub enum ScalarQuery {
    /// Retrieve all row ids where the value is in the given [min, max) range
    Range(Bound<ScalarValue>, Bound<ScalarValue>),
    /// Retrieve all row ids where the value is in the given set of values
    IsIn(Vec<ScalarValue>),
    /// Retrieve all row ids where the value is exactly the given value
    Equals(ScalarValue),
    /// Retrieve all row ids where the value is null
    IsNull(),
}

#[async_trait]
pub trait ScalarIndex: Send + Sync + std::fmt::Debug {
    /// Searches the scalar index
    ///
    /// Returns all row ids that satisfy the query, these row ids are not neccesarily ordered
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array>;

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized;
}
