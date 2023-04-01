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

use std::any::Any;
use std::sync::Arc;

use arrow::array::{as_primitive_array, Float32Builder};
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array};
use arrow_schema::DataType;
use async_trait::async_trait;

use super::Query;
use super::{pq::ProductQuantizer, MetricType, Transformer, VectorIndex};
use crate::arrow::linalg::*;
use crate::index::pb::{Transform, TransformType};
use crate::io::object_reader::{read_fixed_stride_array, ObjectReader};
use crate::io::object_writer::ObjectWriter;
use crate::{Error, Result};

const OPQ_PQ_INIT_ITERATIONS: usize = 10;

/// Rotation matrix `R` described in Optimized Product Quantization.
///
/// [Optimized Product Quantization for Approximate Nearest Neighbor Search
/// (CVPR' 13)](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)
#[derive(Debug, Clone)]
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
        pq: &mut ProductQuantizer,
        train: &MatrixView,
        metric_type: MetricType,
    ) -> Result<(MatrixView, FixedSizeListArray)> {
        let dim = train.num_columns();

        // Train few times to get a better rotation matrix. See Faiss.
        pq.train(&train, metric_type, 1).await?;
        let pq_code = pq.transform(&train, metric_type).await?;

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

        Ok((rotation, pq_code))
    }
}

/// Initialize rotation matrix.
fn init_rotation(dimension: usize) -> Result<MatrixView> {
    let mat = MatrixView::random(dimension, dimension);
    let (u, _, vt) = mat.svd()?;
    let r = vt.transpose().dot(&u.transpose())?;
    Ok(r)
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

        // Run a few iterations to get the initialized centroids.
        let mut pq = ProductQuantizer::new(self.num_sub_vectors, self.num_bits, dim);
        pq.train(&train, self.metric_type, OPQ_PQ_INIT_ITERATIONS)
            .await?;
        let mut pq_code = pq.transform(&train, self.metric_type).await?;

        // Initialize R (rotation matrix)
        let mut rotation = init_rotation(dim)?;
        for i in 0..self.num_iters {
            // Training data, this is the `X`, described in CVPR' 13
            let rotated_data = train.dot(&rotation)?;
            // Reset the pq centroids after rotation.
            pq.reset_centroids(&rotated_data, &pq_code)?;
            (rotation, pq_code) = self
                .train_once(&mut pq, &rotated_data, self.metric_type)
                .await?;
            if (i + 1) % 5 == 0 {
                println!(
                    "Training OPQ iteration {}/{}, PQ distortion={}",
                    i + 1,
                    self.num_iters,
                    pq.distortion(&rotated_data, self.metric_type).await?
                );
            }
        }
        self.rotation = Some(rotation);
        Ok(())
    }

    /// Apply OPQ transform
    async fn transform(&self, data: &MatrixView) -> Result<MatrixView> {
        let result = self
            .rotation
            .as_ref()
            .unwrap()
            .dot(&data.transpose())?
            .transpose();
        Ok(MatrixView::new(result.data(), result.num_columns()))
    }

    /// Write the OPQ rotation matrix to disk.
    async fn save(&self, writer: &mut ObjectWriter) -> Result<Transform> {
        let mut this = self.clone();
        if this.rotation.is_none() {
            return Err(Error::Index("OPQ is not trained".to_string()));
        };
        let rotation = this.rotation.as_ref().unwrap();
        let position = writer
            .write_plain_encoded_array(rotation.data().as_ref())
            .await?;
        this.file_position = Some(position);
        (&this).try_into()
    }
}

pub struct OPQIndex {
    sub_index: Arc<dyn VectorIndex>,

    opq: OptimizedProductQuantizer,
}

impl OPQIndex {
    pub fn new(sub_index: Arc<dyn VectorIndex>, opq: OptimizedProductQuantizer) -> Self {
        Self { sub_index, opq }
    }
}

impl std::fmt::Debug for OPQIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OPQIndex(dim={}) -> {:?}",
            self.opq
                .rotation
                .as_ref()
                .map(|m| m.num_columns())
                .unwrap_or(0),
            self.sub_index
        )
    }
}

#[async_trait]
impl VectorIndex for OPQIndex {
    async fn search(&self, query: &Query) -> Result<RecordBatch> {
        let mat = MatrixView::new(query.key.clone(), query.key.len());
        let transformed = self.opq.transform(&mat).await?;
        let mut transformed_query = query.clone();
        transformed_query.key = transformed.data();
        self.sub_index.search(&transformed_query).await
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_loadable(&self) -> bool {
        false
    }

    async fn load(
        &self,
        _reader: &dyn ObjectReader,
        _offset: usize,
        _length: usize,
    ) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::Index("OPQ does not support load".to_string()))
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

    use approx::assert_relative_eq;
    use arrow::compute::{max, min};
    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatchReader, UInt64Array};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

    use crate::arrow::{linalg::MatrixView, *};
    use crate::dataset::{Dataset, ROW_ID};
    use crate::index::{
        vector::{ivf::IVFIndex, open_index, opq::OPQIndex, VectorIndexParams},
        IndexType,
    };

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

    async fn test_build_index(with_opq: bool) {
        let tmp_dir = tempfile::tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "vector",
            DataType::FixedSizeList(
                Box::new(ArrowField::new("item", DataType::Float32, true)),
                64,
            ),
            true,
        )]));

        let vectors = Float32Array::from_iter_values((0..32000).map(|x| x as f32));
        let vectors = FixedSizeListArray::try_new(vectors, 64).unwrap();
        let batch = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(vectors)],
        )
        .unwrap()
        .into()]);
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batch);
        Dataset::write(&mut reader, tmp_dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

        let dataset = Dataset::open(tmp_dir.path().to_str().unwrap())
            .await
            .unwrap();
        let column = "vector";

        let params = VectorIndexParams::ivf_pq(4, 8, 4, with_opq, MetricType::L2, 3);
        let dataset = dataset
            .create_index(&[column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let index_file = std::fs::read_dir(tmp_dir.path().join("_indices"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let uuid = index_file.file_name().to_str().unwrap().to_string();

        let index = open_index(&dataset, &uuid).await.unwrap();

        if with_opq {
            let opq_idx = index.as_any().downcast_ref::<OPQIndex>().unwrap();
            assert!(opq_idx.opq.rotation.is_some());

            let rotation = opq_idx.opq.rotation.as_ref().unwrap();
            assert_eq!(rotation.num_rows(), 64);
            assert_eq!(rotation.num_columns(), 64);
        } else {
            assert!(index.as_any().is::<IVFIndex>());
        }

        let query = Query {
            column: "vector".to_string(),
            k: 4,
            nprobes: 10,
            refine_factor: None,
            metric_type: MetricType::L2,
            use_index: true,
            key: Float32Array::from_iter_values((0..64).map(|x| x as f32 + 640.0)).into(),
        };
        let results = index.search(&query).await.unwrap();
        let row_ids: &UInt64Array = as_primitive_array(&results[ROW_ID]);
        assert_eq!(row_ids.len(), 4);
        assert!(row_ids.values().contains(&10));
        assert_eq!(min(row_ids).unwrap() + 3, max(row_ids).unwrap());
    }

    #[tokio::test]
    async fn test_build_index_with_opq() {
        test_build_index(true).await;
    }

    #[tokio::test]
    async fn test_build_index_without_opq() {
        test_build_index(false).await;
    }

    #[test]
    fn test_init_rotation() {
        let dim: usize = 64;
        let r = init_rotation(dim).unwrap();
        // R^T * R = I
        let i = r.transpose().dot(&r).unwrap();

        assert_relative_eq!(
            i.data().values(),
            MatrixView::identity(dim).data().values(),
            epsilon = 0.001
        );
    }
}
