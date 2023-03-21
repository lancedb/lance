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

use std::sync::Arc;

use arrow_array::RecordBatch;
use async_trait::async_trait;

use super::Query;
use crate::{arrow::linalg::MatrixView, index::pb, io::object_reader::ObjectReader, Result};

/// Vector Index for (Approximate) Nearest Neighbor (ANN) Search.
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Search the vector for nearest neighbors.
    ///
    /// It returns a [RecordBatch] with Schema of:
    ///
    /// ```
    /// use arrow_schema::{Schema, Field, DataType};
    ///
    /// Schema::new(vec![
    ///   Field::new("_rowid", DataType::UInt64, false),
    ///   Field::new("score", DataType::Float32, false),
    /// ]);
    /// ```
    ///
    /// *WARNINGS*:
    ///  - Only supports `f32` now. Will add f64/f16 later.
    async fn search(&self, query: &Query) -> Result<RecordBatch>;
}

/// A [`VectorIndex`] that can be loaded on-demand, usually as a IVF partition.
#[async_trait]
pub trait LoadableVectorIndex: VectorIndex {
    /// Load the index from the reader.
    ///
    /// Parameters:
    /// - *reader*: the object reader.
    /// - *offset*: the offset of the index in file.
    /// - *length*: the number of vectors in the partition.
    async fn load(
        &self,
        reader: &dyn ObjectReader,
        offset: usize,
        length: usize,
    ) -> Result<Arc<dyn LoadableVectorIndex>>;
}

/// Transformer on vectors.
#[async_trait]
pub trait Transformer: std::fmt::Debug + Sync + Send {
    /// Train the transformer.
    ///
    /// Parameters:
    /// - *data*: training vectors.
    async fn train(&mut self, data: &MatrixView) -> Result<()>;

    /// Apply transform on the matrix `data`.
    ///
    /// Returns a new Matrix instead.
    async fn transform(&self, data: &MatrixView) -> Result<MatrixView>;

    /// Try to convert into protobuf.
    ///
    /// TODO: can we use TryFrom/TryInto as trait constrats?
    fn try_into_pb(&self) -> Result<pb::Transform>;
}
