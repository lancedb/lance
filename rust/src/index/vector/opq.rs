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

//! Optimized Product Quantization
//!
//! [Optimized Product Quantization for Approximate Nearest Neighbor Search
//! (CVPR' 13)](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)

use arrow_array::Float32Array;

/// Rotation matrix `R` described in Optimized Product Quantization.
pub struct OPQ {
    rotation: Float32Array,
    dimension: usize,
}

impl OPQ {
    /// Train a Optimized Product Quantization.
    ///
    /// Parameters:
    ///
    /// - *data*: training dataset.
    /// - *dimension*: dimension of the training dataset.
    /// - *num_sub_vectors*: the number of sub vectors in the product quantization.
    pub async fn new(
        data: &Float32Array,
        dimension: usize,
        num_sub_vectors: usize,
        num_iterations: usize,
    ) -> Self {
        assert_eq!(data.len() % dimension, 0);

        for i in 0..num_iterations {

        }
        todo!()
    }
}

#[cfg(test)]
mod tests {}
