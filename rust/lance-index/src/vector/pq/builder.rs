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

use arrow_array::FixedSizeListArray;
use arrow_array::{cast::AsArray, types::Float32Type, Array, ArrayRef, Float32Array};
use futures::{stream, StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_linalg::{distance::MetricType, MatrixView};
use rand::{self, SeedableRng};
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

    /// Build a [ProductQuantizer] from the given data.
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
        let data = MatrixView::<Float32Type>::try_from(fsl)?;

        let sub_vectors = divide_to_subvectors(&data, self.num_sub_vectors);
        let num_centroids = 2_usize.pow(self.num_bits as u32);
        let dimension = data.num_columns();
        let sub_vector_dimension = dimension / self.num_sub_vectors;

        const REDOS: usize = 1;
        // TODO: parallel training.
        let d = stream::iter(sub_vectors.into_iter())
            .map(|sub_vec| async move {
                let rng = rand::rngs::SmallRng::from_entropy();

                // Centroids for one sub vector.
                let sub_vec = sub_vec.clone();
                let flatten_array: &Float32Array = sub_vec.values().as_primitive();
                train_kmeans(
                    flatten_array,
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
            codebook_builder.extend_from_slice(centroid.values());
        }

        let pd_centroids = Float32Array::from(codebook_builder);

        Ok(Arc::new(ProductQuantizerImpl::new(
            self.num_sub_vectors,
            self.num_bits as u32,
            dimension,
            Arc::new(pd_centroids),
            metric_type,
        )))
    }
}

/// Load ProductQuantizer from Protobuf
pub fn from_proto(proto: &Pq, metric_type: MetricType) -> Result<Arc<dyn ProductQuantizer>> {
    if let Some(tensor) = &proto.codebook_tensor {
        let fsl = FixedSizeListArray::try_from(tensor)?;

        Ok(Arc::new(ProductQuantizerImpl::new(
            proto.num_sub_vectors as usize,
            proto.num_bits,
            proto.dimension as usize,
            Arc::new(fsl.values().as_primitive().clone()), // Support multi-data type later.
            metric_type,
        )))
    } else {
        Ok(Arc::new(ProductQuantizerImpl::new(
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
