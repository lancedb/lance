// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantizer Builder
//!

use crate::vector::quantizer::QuantizerBuildParams;
use arrow::datatypes::ArrowPrimitiveType;
use arrow_array::FixedSizeListArray;
use arrow_array::{Array, ArrayRef};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use lance_core::Result;
use lance_linalg::distance::{DistanceType, Dot, Normalize, L2};
use lance_linalg::{distance::MetricType, MatrixView};
use rand::SeedableRng;
use rayon::prelude::*;

use super::utils::divide_to_subvectors;
use super::ProductQuantizer;
use crate::vector::kmeans::train_kmeans;

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

    fn use_residual(distance_type: DistanceType) -> bool {
        matches!(distance_type, DistanceType::L2 | DistanceType::Cosine)
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

    pub fn build_from_matrix<T: ArrowFloatType + 'static + ArrowPrimitiveType>(
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

        let d = sub_vectors
            .into_par_iter()
            .map(|sub_vec| {
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
            })
            .collect::<Result<Vec<_>>>()?;
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
}
