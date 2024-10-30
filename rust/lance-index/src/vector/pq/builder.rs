// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantizer Builder
//!

use crate::vector::quantizer::QuantizerBuildParams;
use arrow::array::PrimitiveBuilder;
use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{cast::AsArray, types::Float32Type, Array, ArrayRef};
use arrow_array::{ArrowNumericType, FixedSizeListArray, PrimitiveArray};
use arrow_schema::DataType;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::{Dot, Normalize, L2};
use rand::SeedableRng;
use rayon::prelude::*;
use snafu::{location, Location};

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

    /// Run kmeans `REDOS` times and take the best result.
    /// Default to 1.
    pub kmeans_redos: usize,

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
            kmeans_redos: 1,
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

    fn build_from_fsl<T: ArrowNumericType>(
        &self,
        data: &FixedSizeListArray,
        distance_type: DistanceType,
    ) -> Result<ProductQuantizer>
    where
        T::Native: Dot + L2 + Normalize,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        assert_ne!(
            distance_type,
            DistanceType::Cosine,
            "PQ code does not support cosine"
        );

        let sub_vectors = divide_to_subvectors::<T>(data, self.num_sub_vectors)?;
        let num_centroids = 2_usize.pow(self.num_bits as u32);
        let dimension = data.value_length() as usize;
        let sub_vector_dimension = dimension / self.num_sub_vectors;

        let d = sub_vectors
            .into_par_iter()
            .map(|sub_vec| {
                let rng = rand::rngs::SmallRng::from_entropy();
                train_kmeans::<T>(
                    &sub_vec,
                    sub_vector_dimension,
                    num_centroids,
                    self.max_iters as u32,
                    self.kmeans_redos,
                    rng,
                    distance_type,
                    self.sample_rate,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut codebook_builder = PrimitiveBuilder::<T>::with_capacity(num_centroids * dimension);
        for centroid in d.iter() {
            let c = centroid
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .expect("failed to downcast to PrimitiveArray");
            codebook_builder.append_slice(c.values());
        }

        let pd_centroids = codebook_builder.finish();

        Ok(ProductQuantizer::new(
            self.num_sub_vectors,
            self.num_bits as u32,
            dimension,
            FixedSizeListArray::try_new_from_values(pd_centroids, dimension as i32)?,
            distance_type,
        ))
    }

    /// Build a [ProductQuantizer] from the given data.
    ///
    /// If the [MetricType] is [MetricType::Cosine], the input data will be normalized.
    pub fn build(&self, data: &dyn Array, distance_type: DistanceType) -> Result<ProductQuantizer> {
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
            DataType::Float16 => self.build_from_fsl::<Float16Type>(fsl, distance_type),
            DataType::Float32 => self.build_from_fsl::<Float32Type>(fsl, distance_type),
            DataType::Float64 => self.build_from_fsl::<Float64Type>(fsl, distance_type),
            _ => Err(Error::Index {
                message: format!("PQ builder: unsupported data type: {}", fsl.value_type()),
                location: location!(),
            }),
        }
    }
}
