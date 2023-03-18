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

use std::sync::Arc;

use arrow::array::{as_primitive_array, Float32Builder};
use arrow_array::{Array, Float32Array, UInt8Array};
use arrow_schema::DataType;
use async_trait::async_trait;

use super::{pq::ProductQuantizer, MetricType, Transformer};
use crate::arrow::linalg::*;
use crate::index::pb::{Transform, TransformType};
use crate::io::object_reader::{read_fixed_stride_array, ObjectReader};
use crate::{Error, Result};

/// Rotation matrix `R` described in Optimized Product Quantization.
///
/// [Optimized Product Quantization for Approximate Nearest Neighbor Search
/// (CVPR' 13)](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)
#[derive(Debug)]
pub struct OptimizedProductQuantizer {
    num_sub_vectors: usize,

    /// Number of bits to present centroids in each sub-vector.
    num_bits: u32,

    /// OPQ rotation
    pub rotation: Option<MatrixView>,

    /// The offset where the matrix is stored in the index file.
    pub file_position: Option<usize>,

    /// The metric to compute the distance.
    metric_type: MetricType,

    /// Number of iterations to train OPQ.
    num_iters: usize,
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
    pub fn new(
        num_sub_vectors: usize,
        num_bits: u32,
        metric_type: MetricType,
        num_iters: usize,
    ) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            rotation: None,
            file_position: None,
            metric_type,
            num_iters,
        }
    }

    /// Load the optimized product quantizer.
    pub async fn load(reader: &dyn ObjectReader, position: usize, shape: &[usize]) -> Result<Self> {
        let dim = shape[0];
        let length = dim * dim;
        let data =
            read_fixed_stride_array(reader, &DataType::Float32, position, length, ..).await?;
        let f32_data: Float32Array = as_primitive_array(data.as_ref()).clone();
        let rotation = Some(MatrixView::new(Arc::new(f32_data), dim));
        Ok(Self {
            num_sub_vectors: 0,
            num_bits: 0,
            rotation,
            file_position: None,
            metric_type: MetricType::L2,
            num_iters: 0,
        })
    }

    /// Train once and return the rotation matrix and PQ codebook.
    async fn train_once(
        &self,
        train: &MatrixView,
        metric_type: MetricType,
    ) -> Result<MatrixView> {
        let dim = train.num_columns();

        // Iteratively train PQ.
        // train 4 times per iteration, see Faiss.
        let mut pq = ProductQuantizer::new(self.num_sub_vectors, self.num_bits, dim);
        pq.train(&train, metric_type, 10).await?;
        let pq_code = pq.transform(&train, metric_type).await?;

        println!("PQ distortion: {}", pq.distortion(&train, metric_type).await?);

        // Reconstruct Y
        let mut builder = Float32Builder::with_capacity(train.num_columns() * train.num_rows());
        for i in 0..pq_code.len() {
            let code_arr = pq_code.value(i);
            let code: &UInt8Array = as_primitive_array(code_arr.as_ref());
            let reconstructed_vector = pq.reconstruct(code.values());
            builder.append_slice(reconstructed_vector.values());
        }
        // Reconstructed vectors, `Y` in Section 3.1 in CVPR' 13.
        // X and Y are column major described in the paper.
        let y_data = Arc::new(builder.finish());
        let y = MatrixView::new(y_data, dim).transpose();
        // Solving `min||RX - Y||` in Section 3.1 in CVPR' 13.
        //
        //  X * T(Y) = U * S * T(V)
        //  R = V * T(U)
        let x_yt = train.transpose().dot(&y.transpose())?;
        let (u, _, vt) = x_yt.svd()?;
        let rotation = vt.transpose().dot(&u.transpose())?;

        Ok(rotation)
    }
}

#[async_trait]
impl Transformer for OptimizedProductQuantizer {
    async fn train(&mut self, data: &MatrixView) -> Result<()> {
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
        let mut rotation = MatrixView::identity(dim);
        for _ in 0..self.num_iters {
            // Training data, this is the `X`, described in CVPR' 13
            let rotated_data = train.dot(&rotation)?;
            let rot = self.train_once(&rotated_data, self.metric_type).await?;
            rotation = rot;
        }
        self.rotation = Some(rotation);
        Ok(())
    }

    /// Apply OPQ transform
    async fn transform(&self, data: &MatrixView) -> Result<MatrixView> {
        let rotation = self.rotation.as_ref().unwrap();
        let result = rotation.dot(&data.transpose())?.transpose();
        Ok(MatrixView::new(result.data(), result.num_columns()))
    }

    fn try_into_pb(&self) -> Result<Transform> {
        self.try_into()
    }
}

impl TryFrom<&OptimizedProductQuantizer> for Transform {
    type Error = Error;

    fn try_from(opq: &OptimizedProductQuantizer) -> Result<Self> {
        if opq.file_position.is_none() {
            return Err(Error::Index("OPQ has not been persisted yet".to_string()));
        }
        let rotation = opq
            .rotation
            .as_ref()
            .ok_or(Error::Index("OPQ is not trained".to_string()))?;
        Ok(Transform {
            position: opq.file_position.unwrap() as u64,
            shape: vec![rotation.num_rows() as u32, rotation.num_columns() as u32],
            r#type: TransformType::Opq.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Float32Array;

    use crate::arrow::linalg::MatrixView;

    #[tokio::test]
    async fn test_train_opq() {
        const DIM: usize = 32;
        let data = Arc::new(Float32Array::from_iter((0..12800).map(|v| v as f32)));
        let matrix = MatrixView::new(data, DIM);

        let mut opq = OptimizedProductQuantizer::new(4, 8, MetricType::L2, 10);
        opq.train(&matrix).await.unwrap();

        assert_eq!(opq.rotation.as_ref().unwrap().num_rows(), DIM);
        assert_eq!(opq.rotation.as_ref().unwrap().num_columns(), DIM);
    }
}
