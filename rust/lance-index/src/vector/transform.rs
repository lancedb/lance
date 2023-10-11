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

//! Vector Transforms
//!

use arrow_array::{types::Float32Type, RecordBatch};

use lance_core::Result;
use lance_linalg::MatrixView;

/// Transform of a Vector Matrix.
///
///
pub trait Transformer: std::fmt::Debug + Sync + Send {
    /// Transform a [`RecordBatch`] of vectors
    ///
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch>;
}

/// Compute the residual vector of a Vector Matrix to their centroids.
#[derive(Clone)]
pub struct ResidualTransform {
    centroids: MatrixView<Float32Type>,

    /// Partition Column
    part_col: String,

    /// Vector Column
    vec_col: String,
}

impl std::fmt::Debug for ResidualTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResidualTransform")
    }
}

impl ResidualTransform {
    pub fn new(centroids: MatrixView<Float32Type>, part_col: &str, column: &str) -> Self {
        Self {
            centroids,
            part_col: part_col.to_owned(),
            vec_col: column.to_owned(),
        }
    }
}

impl Transformer for ResidualTransform {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        todo!()
    }
}
