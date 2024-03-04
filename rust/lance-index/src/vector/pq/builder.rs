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

//! Product Quantizer Builder
//!

use std::sync::Arc;

use crate::pb;
use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{
    cast::AsArray, types::Float32Type, Array, ArrayRef, Float32Array, PrimitiveArray,
};
use arrow_array::{ArrowNumericType, FixedSizeListArray};
use arrow_schema::DataType;
use futures::{stream, StreamExt, TryStreamExt};
use lance_arrow::{ArrowFloatType, FloatArray};
use lance_core::{Error, Result};
use lance_linalg::distance::{Dot, L2};
use lance_linalg::{distance::MetricType, MatrixView};
use rand::SeedableRng;
use snafu::{location, Location};

use super::utils::divide_to_subvectors;
use super::ProductQuantizer;
use crate::pb::Pq;
use crate::vector::{kmeans::train_kmeans, pq::ProductQuantizerImpl};

/// Parameters for building product quantizer.
#[derive(Debug, Clone)]
pub struct PQBuildParams {
    /// Number of sub-vectors to build PQ code
    pub num_sub_vectors: usize,

    /// The number of bits to present one PQ centroid.
    pub num_bits: usize,

    /// Train as optimized product quantization.
    pub use_opq: bool,

    /// The max number of iterations for kmeans training.
    pub max_iters: usize,

    /// Max number of iterations to train Optimized Product Quantization,
    /// if `use_opq` is true.
    pub max_opq_iters: usize,

    /// User provided codebook.
    pub codebook: Option<ArrayRef>,

    /// Sample rate to train PQ codebook.
    pub sample_rate: usize,
}

impl Default for PQBuildParams {
    fn default() -> Self {
        Self {
            num_sub_vectors: 16,
            num_bits: 8,
            use_opq: false,
            max_iters: 50,
            max_opq_iters: 50,
            codebook: None,
            sample_rate: 256,
        }
    }
}

impl PQBuildParams {
    pub fn new(num_sub_vectors: usize, num_bits: usize) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            ..Default::default()
        }
    }

    pub fn with_codebook(num_sub_vectors: usize, num_bits: usize, codebook: ArrayRef) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            codebook: Some(codebook),
            ..Default::default()
        }
    }

    pub async fn build_from_matrix<T: ArrowFloatType + Dot + L2 + 'static>(
        &self,
        data: &MatrixView<T>,
        metric_type: MetricType,
    ) -> Result<Arc<dyn ProductQuantizer + 'static>> {
        assert_ne!(
            metric_type,
            MetricType::Cosine,
            "PQ code does not support cosine"
        );

        const REDOS: usize = 1;

        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors);
        let num_centroids = 2_usize.pow(self.num_bits as u32);
        let dimension = data.num_columns();
        let sub_vector_dimension = dimension / self.num_sub_vectors;

        let d = stream::iter(sub_vectors.into_iter())
            .map(|sub_vec| async move {
                let rng = rand::rngs::SmallRng::from_entropy();
                train_kmeans::<T>(
                    sub_vec.as_ref(),
                    None,
                    sub_vector_dimension,
                    num_centroids,
                    self.max_iters as u32,
                    REDOS,
                    rng.clone(),
                    metric_type,
                    self.sample_rate,
                )
                .await
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        let mut codebook_builder = Vec::with_capacity(num_centroids * dimension);
        for centroid in d.iter() {
            codebook_builder.extend_from_slice(centroid.as_slice());
        }

        let pd_centroids = T::ArrayType::from(codebook_builder);

        Ok(Arc::new(ProductQuantizerImpl::<T>::new(
            self.num_sub_vectors,
            self.num_bits as u32,
            dimension,
            Arc::new(pd_centroids),
            metric_type,
        )))
    }

    /// Build a [ProductQuantizer] from the given data.
    ///
    /// If the [MetricType] is [MetricType::Cosine], the input data will be normalized.
    pub async fn build(
        &self,
        data: &dyn Array,
        metric_type: MetricType,
    ) -> Result<Arc<dyn ProductQuantizer>> {
        assert_eq!(data.null_count(), 0);
        let fsl = data.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "PQ builder: input is not a FixedSizeList: {}",
                data.data_type()
            ),
            location: location!(),
        })?;
        // TODO: support bf16 later.
        match fsl.value_type() {
            DataType::Float16 => {
                let data = MatrixView::<Float16Type>::try_from(fsl)?;
                self.build_from_matrix(&data, metric_type).await
            }
            DataType::Float32 => {
                let data = MatrixView::<Float32Type>::try_from(fsl)?;
                self.build_from_matrix(&data, metric_type).await
            }
            DataType::Float64 => {
                let data = MatrixView::<Float64Type>::try_from(fsl)?;
                self.build_from_matrix(&data, metric_type).await
            }
            _ => Err(Error::Index {
                message: format!("PQ builder: unsupported data type: {}", fsl.value_type()),
                location: location!(),
            }),
        }
    }
}

fn create_typed_pq<
    T: ArrowFloatType<ArrayType = PrimitiveArray<T>> + ArrowNumericType + L2 + Dot,
>(
    proto: &Pq,
    metric_type: MetricType,
    array: &dyn Array,
) -> Arc<dyn ProductQuantizer> {
    Arc::new(ProductQuantizerImpl::<T>::new(
        proto.num_sub_vectors as usize,
        proto.num_bits,
        proto.dimension as usize,
        Arc::new(array.as_primitive::<T>().clone()),
        metric_type,
    ))
}

/// Load ProductQuantizer from Protobuf
pub fn from_proto(proto: &Pq, metric_type: MetricType) -> Result<Arc<dyn ProductQuantizer>> {
    let mt = if metric_type == MetricType::Cosine {
        MetricType::L2
    } else {
        metric_type
    };

    if let Some(tensor) = &proto.codebook_tensor {
        let fsl = FixedSizeListArray::try_from(tensor)?;

        match pb::tensor::DataType::try_from(tensor.data_type)? {
            pb::tensor::DataType::Bfloat16 => {
                unimplemented!()
            }
            pb::tensor::DataType::Float16 => {
                Ok(create_typed_pq::<Float16Type>(proto, mt, fsl.values()))
            }
            pb::tensor::DataType::Float32 => {
                Ok(create_typed_pq::<Float32Type>(proto, mt, fsl.values()))
            }
            pb::tensor::DataType::Float64 => {
                Ok(create_typed_pq::<Float64Type>(proto, mt, fsl.values()))
            }
            _ => Err(Error::Index {
                message: format!("PQ builder: unsupported data type: {:?}", tensor.data_type),
                location: location!(),
            }),
        }
    } else {
        Ok(Arc::new(ProductQuantizerImpl::<Float32Type>::new(
            proto.num_sub_vectors as usize,
            proto.num_bits,
            proto.dimension as usize,
            Arc::new(Float32Array::from_iter_values(
                proto.codebook.iter().copied(),
            )),
            metric_type,
        )))
    }
}
