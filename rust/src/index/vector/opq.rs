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

use std::sync::Arc;

use arrow::array::{as_primitive_array, Float32Builder};
use arrow_array::{Array, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, UInt8Array};

use crate::arrow::linalg::*;
use crate::arrow::FixedSizeListArrayExt;
use crate::Result;

use super::pq::ProductQuantizer;
use super::MetricType;

/// Rotation matrix `R` described in Optimized Product Quantization.
pub struct OptimizedProductQuantizer {
    num_sub_vectors: usize,

    /// Number of bits to present centroids in each sub-vector.
    num_bits: u32,
}

impl OptimizedProductQuantizer {
    /// Train a Optimized Product Quantization.
    ///
    /// Parameters:
    ///
    /// - *data*: training dataset.
    /// - *dimension*: dimension of the training dataset.
    /// - *num_sub_vectors*: the number of sub vectors in the product quantization.
    /// - *num_iterations*: The number of iterations to train on OPQ rotation matrix.
    pub async fn new(
        data: &Float32Array,
        dimension: usize,
        num_sub_vectors: usize,
        num_bits: u32,
    ) -> Self {
        assert_eq!(data.len() % dimension, 0);
        Self {
            num_sub_vectors,
            num_bits,
        }
    }

    /// Train the opq
    pub async fn train(
        &mut self,
        data: &MatrixView,
        metric_type: MetricType,
        num_iters: usize,
    ) -> Result<FixedSizeBinaryArray> {
        let dim = data.num_columns();

        let num_centroids = ProductQuantizer::num_centroids(self.num_bits);
        // See in Faiss, it does not train more than `256*n_centroids` samples
        let train = if data.num_rows() > num_centroids * 256 {
            println!(
                "Sample {} out of {} to train kmeans of {} dim, {} clusters",
                256 * num_centroids,
                data.num_rows(),
                data.num_columns(),
                num_centroids,
            );
            data.sample(num_centroids * 256)
        } else {
            data.clone()
        };

        // Initialize R (rotation matrix)
        let mut rotation = MatrixView::random(dim, dim);
        for _ in 0..num_iters {
            // Training data, this is the `X`, described in CVPR' 13
            let train = train.dot(&rotation)?;
            rotation = self.train_once(&train, metric_type).await?;
        }
        todo!()
    }

    /// Train once and return the rotation matrix and PQ codebook.
    async fn train_once(&self, train: &MatrixView, metric_type: MetricType) -> Result<MatrixView> {
        let dim = train.num_columns();
        // TODO: make PQ::fit_transform work with MatrixView.
        let fixed_list = FixedSizeListArray::try_new(train.data().as_ref(), dim as i32)?;
        let mut pq = ProductQuantizer::new(self.num_sub_vectors, self.num_bits, dim);
        let pq_code = pq.fit_transform(&fixed_list, metric_type).await?;

        // Reconstruct Y
        let mut builder = Float32Builder::with_capacity(train.num_columns() * train.num_rows());
        for i in 0..pq_code.len() {
            let code_arr = pq_code.value(i);
            let code: &UInt8Array = as_primitive_array(code_arr.as_ref());
            let reconstructed_vector = pq.reconstruct(code.values());
            builder.append_slice(reconstructed_vector.values());
        }
        // Reconstructed vectors, `Y` in Section 3.1 in CVPR' 13.
        let y_data = Arc::new(builder.finish());
        let y = MatrixView::new(y_data, dim);
        // Solving `min||RX - Y||` in Section 3.1 in CVPR' 13.
        //
        //  X * T(Y) = U * S * T(V)
        //  R = V * T(U)
        let x_yt = train.dot(&y.transpose())?;
        let (u, _, vt) = x_yt.svd()?;
        let rotation = vt.transpose().dot(&u.transpose())?;

        Ok(rotation)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_train_opq() {

    }
}
