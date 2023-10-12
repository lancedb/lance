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

//! Traits for vector index.
//!

use std::{any::Any, sync::Arc};

use arrow_array::types::Float32Type;
use arrow_array::RecordBatch;
use async_trait::async_trait;

use lance_linalg::MatrixView;

use super::Query;
use crate::{
    index::{pb::Transform, prefilter::PreFilter, Index},
    io::{object_reader::ObjectReader, object_writer::ObjectWriter},
    Result,
};

/// Some indices can be split into shards / partitions and embedded into an
/// index file.  When that happens we typically store some global metadata about
/// the index that contains the location and size of the shards.  That metadata
/// can be deserialized into a shard loader which later loads the actual shards
///
/// Currently this is only implemented for PQ which stores the offset & # of rows
/// for each shard.
#[async_trait]
pub(crate) trait IndexShardLoader: Index + Send + Sync + std::fmt::Debug {
    /// Load the shard at the given index
    async fn load(
        &self,
        reader: &dyn ObjectReader,
        shard_index: usize,
    ) -> Result<Box<dyn VectorIndex>>;

    /// Store the shard into the disk, at the current location, and update the shard
    /// loader metadata
    async fn store(
        &mut self,
        index: Arc<dyn VectorIndex>,
        writer: &mut ObjectWriter,
        shard_index: usize,
    ) -> Result<()>;

    fn any_mut(&mut self) -> &mut dyn Any;
}

/// Vector Index for (Approximate) Nearest Neighbor (ANN) Search.
#[async_trait]
#[allow(clippy::redundant_pub_crate)]
pub(crate) trait VectorIndex: Send + Sync + std::fmt::Debug + Index {
    /// Search the vector for nearest neighbors.
    ///
    /// It returns a [RecordBatch] with Schema of:
    ///
    /// ```
    /// use arrow_schema::{Schema, Field, DataType};
    ///
    /// Schema::new(vec![
    ///   Field::new("_rowid", DataType::UInt64, false),
    ///   Field::new("_distance", DataType::Float32, false),
    /// ]);
    /// ```
    ///
    /// The `pre_filter` argument is used to filter out row ids that we know are
    /// not relevant to the query. For example, it removes deleted rows.
    ///
    /// *WARNINGS*:
    ///  - Only supports `f32` now. Will add f64/f16 later.
    async fn search(&self, query: &Query, pre_filter: &PreFilter) -> Result<RecordBatch>;

    /// If the index is loadable by IVF, so it can be a sub-index that
    /// is loaded on demand by IVF.
    fn is_loadable(&self) -> bool;
}

/// Transformer on vectors.
#[async_trait]
pub trait Transformer: std::fmt::Debug + Sync + Send {
    /// Train the transformer.
    ///
    /// Parameters:
    /// - *data*: training vectors.
    async fn train(&mut self, data: &MatrixView<Float32Type>) -> Result<()>;

    /// Apply transform on the matrix `data`.
    ///
    /// Returns a new Matrix instead.
    async fn transform(&self, data: &MatrixView<Float32Type>) -> Result<MatrixView<Float32Type>>;

    async fn save(&self, writer: &mut ObjectWriter) -> Result<Transform>;
}
