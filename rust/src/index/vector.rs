// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Vector Index for Fast Approximate Nearest Neighbor (ANN) Search
//!

use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch};
use async_trait::async_trait;

pub mod flat;
pub mod ivf;
mod kmeans;
mod pq;

use crate::Result;

/// Query parameters for the vector indices
#[derive(Debug, Clone)]
pub struct Query {
    pub column: String,
    /// The vector to be searched.
    pub key: Arc<Float32Array>,
    /// Top k results to return.
    pub k: usize,
    /// The number of probs to load and search.
    pub nprobs: usize,
}

/// Vector Index for (Appoximate) Nearest Neighbor (ANN) Search.
#[async_trait]
pub trait VectorIndex {
    /// Search the vector for nearest neighbours.
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
