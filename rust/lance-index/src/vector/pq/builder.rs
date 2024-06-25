// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantizer Builder
//!


use crate::vector::quantizer::QuantizerBuildParams;
use arrow::datatypes::ArrowPrimitiveType;
use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{
    cast::AsArray, types::Float32Type, Array, ArrayRef,
};
use arrow_array::{FixedSizeListArray};
use arrow_schema::DataType;
use futures::{stream, StreamExt, TryStreamExt};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use lance_core::{Error, Result};
use lance_linalg::distance::{Dot, Normalize, L2};
use lance_linalg::{distance::MetricType, MatrixView};
use rand::SeedableRng;
use snafu::{location, Location};

use super::utils::divide_to_subvectors;
use super::ProductQuantizer;
use crate::vector::{kmeans::train_kmeans};

/// Parameters for building product quantizer.
#[derive(Debug, Clone)]
pub struct PQBuildParams {
    /// Number of sub-vectors to build PQ code
    pub num_sub_vectors: usize,

    /// The number of bits to present one PQ centroid.
    pub num_bits: usize,

    /// The max number of iterations for kmeans training.
    pub max_iters: usize,

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
            max_iters: 50,
            codebook: None,
            sample_rate: 256,
        }
    }
}

impl QuantizerBuildParams for PQBuildParams {
    fn sample_size(&self) -> usize {
        self.sample_rate * 2_usize.pow(self.num_bits as u32)
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

    pub async fn build_from_matrix<T: ArrowFloatType + 'static + ArrowPrimitiveType>(
        &self,
        data: &MatrixView<T>,
        metric_type: MetricType,
    ) -> Result<ProductQuantizer>
    where
        <T as ArrowFloatType>::Native: Dot + L2 + Normalize,
    {
        assert_ne!(
            metric_type,
            MetricType::Cosine,
            "PQ code does not support cosine"
        );

        const REDOS: usize = 1;

        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors)?;
        let num_centroids = 2_usize.pow(self.num_bits as u32);
        let dimension = data.num_columns();
        let sub_vector_dimension = dimension / self.num_sub_vectors;

        let d = stream::iter(sub_vectors.into_iter())
            .map(|sub_vec| async move {
                let rng = rand::rngs::SmallRng::from_entropy();
                train_kmeans::<T>(
                    sub_vec.as_ref(),
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
            let c = centroid.as_any().downcast_ref::<T::ArrayType>().unwrap();
            codebook_builder.extend_from_slice(c.as_slice());
        }

        let pd_centroids = T::ArrayType::from(codebook_builder);

        Ok(ProductQuantizer::new(
            self.num_sub_vectors,
            self.num_bits as u32,
            dimension,
            FixedSizeListArray::try_new_from_values(pd_centroids, dimension as i32)?,
            metric_type,
        ))
    }

    /// Build a [ProductQuantizer] from the given data.
    ///
    /// If the [MetricType] is [MetricType::Cosine], the input data will be normalized.
    pub async fn build(
        &self,
        data: &dyn Array,
        metric_type: MetricType,
    ) -> Result<ProductQuantizer> {
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
