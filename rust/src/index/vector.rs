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

use std::any::Any;
use std::sync::Arc;

use crate::datatypes::Schema;
use arrow_array::{Float32Array, RecordBatch};
use async_trait::async_trait;

pub mod flat;
pub mod ivf;
mod kmeans;
mod pq;

use super::IndexParams;
use crate::{Error, Result};

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

    /// If presented, apply a refine step.
    /// TODO: should we support fraction / float number here?
    pub refine_factor: Option<u32>,
}

/// Vector Index for (Approximate) Nearest Neighbor (ANN) Search.
#[async_trait]
pub trait VectorIndex {
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

/// The parameters to build vector index.
pub struct VectorIndexParams {
    // This is hard coded for IVF_PQ for now. Can refactor later to support more.
    /// The number of IVF partitions
    pub num_partitions: u32,

    /// the number of bits to present the centroids used in PQ.
    pub nbits: u8,

    /// the number of sub vectors used in PQ.
    pub num_sub_vectors: u32,
}

impl VectorIndexParams {
    /// Create index parameters for `IVF_PQ` index.
    ///
    /// Parameters
    ///
    ///  - `num_partitions`: the number of IVF partitions.
    ///  - `nbits`: the number of bits to present the centroids used in PQ. Can only be `8` for now.
    ///  - `num_sub_vectors`: the number of sub vectors used in PQ.
    pub fn ivf_pq(num_partitions: u32, nbits: u8, num_sub_vectors: u32) -> Self {
        Self {
            num_partitions,
            nbits,
            num_sub_vectors,
        }
    }
}

impl Default for VectorIndexParams {
    fn default() -> Self {
        Self {
            num_partitions: 32,
            nbits: 8,
            num_sub_vectors: 16,
        }
    }
}

impl IndexParams for VectorIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
