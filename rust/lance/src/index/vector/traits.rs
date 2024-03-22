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

use std::{collections::HashMap, sync::Arc};

use arrow_array::{types::Float32Type, FixedSizeListArray, RecordBatch};
use arrow_schema::Field as ArrowField;
use async_trait::async_trait;

use lance_core::Result;
use lance_index::{vector::Query, Index};
use lance_io::{object_writer::ObjectWriter, traits::Reader};
use lance_linalg::{distance::MetricType, MatrixView};

use crate::index::{pb::Transform, prefilter::PreFilter};

/// Vector Index for (Approximate) Nearest Neighbor (ANN) Search.
#[async_trait]
#[allow(clippy::redundant_pub_crate)]
pub trait VectorIndex: Send + Sync + std::fmt::Debug + Index {
    /// Search the vector for nearest neighbors.
    ///
    /// It returns a [RecordBatch] with Schema of:
    ///
    /// ```
    /// use arrow_schema::{Schema, Field, DataType};
    ///
    /// Schema::new(vec![
    ///   Field::new("_rowid", DataType::UInt64, true),
    ///   Field::new("_distance", DataType::Float32, false),
    /// ]);
    /// ```
    ///
    /// The `pre_filter` argument is used to filter out row ids that we know are
    /// not relevant to the query. For example, it removes deleted rows.
    ///
    /// *WARNINGS*:
    ///  - Only supports `f32` now. Will add f64/f16 later.
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch>;

    /// If the index is loadable by IVF, so it can be a sub-index that
    /// is loaded on demand by IVF.
    fn is_loadable(&self) -> bool;

    /// Use residual vector to search.
    fn use_residual(&self) -> bool;

    /// If the index can be remapped return Ok.  Else return an error
    /// explaining why not
    fn check_can_remap(&self) -> Result<()>;

    /// Load the index from the reader on-demand.
    async fn load(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>>;

    /// Load the partition from the reader on-demand.
    async fn load_partition(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
        _partition_id: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        self.load(reader, offset, length).await
    }

    /// Remap the index according to mapping
    ///
    /// Each item in mapping describes an old row id -> new row id
    /// pair.  If old row id -> None then that row id has been
    /// deleted and can be removed from the index.
    ///
    /// If an old row id is not in the mapping then it should be
    /// left alone.
    fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()>;

    /// Transform the index in place to be for a different data type.
    ///
    /// This can be used, for example, to transform a float32 index to a float16 index.
    fn cast(&mut self, to: &ArrowField) -> Result<()>;

    /// The metric type of this vector index.
    fn metric_type(&self) -> MetricType;
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
    async fn transform(&self, data: &FixedSizeListArray) -> Result<FixedSizeListArray>;

    async fn save(&self, writer: &mut ObjectWriter) -> Result<Transform>;
}
