// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Global PQ (Product Quantization) codebook trainer for distributed vector indexing

use arrow_array::{cast::AsArray, Array, FixedSizeListArray};
use arrow_schema::FieldRef;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;

use crate::vector::kmeans::{train_kmeans, KMeansParams};
use crate::vector::pq::{PQBuildParams, ProductQuantizer};

/// Global PQ codebook trainer that samples from entire dataset
pub struct GlobalPqTrainer {
    num_sub_vectors: usize,
    num_bits: usize,
    sample_rate: usize,
    max_iters: usize,
    distance_type: DistanceType,
}

impl GlobalPqTrainer {
    /// Create a new global PQ trainer
    pub fn new(
        num_sub_vectors: usize,
        num_bits: usize,
        sample_rate: usize,
        max_iters: usize,
        distance_type: DistanceType,
    ) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            sample_rate,
            max_iters,
            distance_type,
        }
    }

    /// Train a global PQ codebook from training data
    pub async fn train_global_pq_codebook(
        &self,
        training_vectors: &FixedSizeListArray,
    ) -> Result<ProductQuantizer> {
        log::info!(
            "Starting global PQ codebook training: num_sub_vectors={}, num_bits={}, sample_rate={}",
            self.num_sub_vectors,
            self.num_bits,
            self.sample_rate
        );

        let dimension = training_vectors.value_length() as usize;
        self.validate_dimension(dimension)?;

        log::debug!(
            "Training on {} vectors with dimension {}",
            training_vectors.len(),
            dimension
        );

        // Ensure enough samples for PQ training
        let num_centroids = 2_usize.pow(self.num_bits as u32);
        if training_vectors.len() < num_centroids {
            return Err(Error::Index {
                message: format!(
                    "Not enough rows to train global PQ codebook. Requires {:?} rows but only {:?} available",
                    num_centroids,
                    training_vectors.len()
                ),
                location: snafu::location!(),
            });
        }

        // Create PQ build parameters
        let _pq_params = PQBuildParams {
            num_sub_vectors: self.num_sub_vectors,
            num_bits: self.num_bits,
            max_iters: self.max_iters,
            sample_rate: self.sample_rate,
            ..Default::default()
        };

        // Build the global PQ codebook using the existing PQ training logic
        let sub_vector_dimension = dimension / self.num_sub_vectors;
        let num_centroids = 2_usize.pow(self.num_bits as u32);

        let mut codebook_values = Vec::new();

        // Train k-means for each sub-vector
        for sub_vec_idx in 0..self.num_sub_vectors {
            // Extract sub-vectors
            let mut sub_vector_data = Vec::new();
            for vec_idx in 0..training_vectors.len() {
                let vector = training_vectors.value(vec_idx);
                let values = vector
                    .as_primitive::<arrow_array::types::Float32Type>()
                    .values();
                let start_idx = sub_vec_idx * sub_vector_dimension;
                let end_idx = start_idx + sub_vector_dimension;
                sub_vector_data.extend_from_slice(&values[start_idx..end_idx]);
            }

            let sub_vector_array = arrow_array::Float32Array::from(sub_vector_data);

            // Create KMeans parameters
            let kmeans_params = KMeansParams {
                max_iters: self.max_iters as u32,
                ..Default::default()
            };

            // Train k-means
            let kmeans_result = train_kmeans(
                &sub_vector_array,
                kmeans_params,
                num_centroids,
                self.sample_rate,
                sub_vector_dimension,
            )?;

            // Extract centroids
            let centroids = kmeans_result
                .centroids
                .as_primitive::<arrow_array::types::Float32Type>()
                .values();
            codebook_values.extend_from_slice(centroids);
        }

        // Create the codebook array
        let codebook_array = arrow_array::Float32Array::from(codebook_values);
        let field = FieldRef::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::Float32,
            false,
        ));
        let codebook = FixedSizeListArray::new(
            field,
            dimension as i32,
            std::sync::Arc::new(codebook_array) as arrow_array::ArrayRef,
            None,
        );

        let pq_quantizer = ProductQuantizer::new(
            self.num_sub_vectors,
            self.num_bits as u32,
            dimension,
            codebook,
            self.distance_type,
        );

        log::info!(
            "Successfully trained global PQ codebook with dimension {} and {} sub-vectors",
            dimension,
            self.num_sub_vectors
        );

        Ok(pq_quantizer)
    }

    /// Sample training data globally from the entire dataset (placeholder)
    async fn _sample_training_data_global(
        &self,
        _dataset_path: &str,
        _column: &str,
        _expected_sample_size: usize,
        _fragments: Option<&[lance_table::format::Fragment]>,
    ) -> Result<FixedSizeListArray> {
        // This is a placeholder implementation
        // In a real implementation, this would:
        // 1. Connect to the dataset
        // 2. Sample vectors globally
        // 3. Return the sampled data as FixedSizeListArray

        Err(Error::invalid_input(
            "Global sampling not implemented in this example. Use pre-sampled data.".to_string(),
            snafu::location!(),
        ))
    }

    /// Validate PQ parameters against vector dimension
    pub fn validate_dimension(&self, dimension: usize) -> Result<()> {
        if dimension % self.num_sub_vectors != 0 {
            return Err(Error::invalid_input(
                format!(
                    "Vector dimension {} must be divisible by num_sub_vectors {}",
                    dimension, self.num_sub_vectors
                ),
                snafu::location!(),
            ));
        }
        Ok(())
    }

    /// Get the number of sub-vectors
    pub fn num_sub_vectors(&self) -> usize {
        self.num_sub_vectors
    }

    /// Get the number of bits per sub-vector
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }
}

/// Convenience function to train global PQ codebook from pre-sampled data
pub async fn train_global_pq_codebook(
    training_vectors: &FixedSizeListArray,
    num_sub_vectors: usize,
    num_bits: usize,
    sample_rate: usize,
    max_iters: usize,
    distance_type: DistanceType,
) -> Result<ProductQuantizer> {
    let trainer = GlobalPqTrainer::new(
        num_sub_vectors,
        num_bits,
        sample_rate,
        max_iters,
        distance_type,
    );

    trainer.train_global_pq_codebook(training_vectors).await
}

/// Sample training data globally from the entire dataset (placeholder)
pub async fn sample_training_data_global(
    _dataset_path: &str,
    _column: &str,
    _expected_sample_size: usize,
    _distance_type: DistanceType,
) -> Result<FixedSizeListArray> {
    // This is a placeholder implementation
    // In a real implementation, this would:
    // 1. Connect to the dataset
    // 2. Sample vectors globally
    // 3. Return the sampled data as FixedSizeListArray

    Err(Error::invalid_input(
        "Global sampling not implemented in this example. Use pre-sampled data.".to_string(),
        snafu::location!(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_global_pq_trainer_creation() {
        let trainer = GlobalPqTrainer::new(8, 8, 256, 50, DistanceType::L2);
        assert_eq!(trainer.num_sub_vectors, 8);
        assert_eq!(trainer.num_bits, 8);
        assert_eq!(trainer.sample_rate, 256);
        assert_eq!(trainer.max_iters, 50);
    }

    #[tokio::test]
    async fn test_dimension_validation() {
        let trainer = GlobalPqTrainer::new(8, 8, 256, 50, DistanceType::L2);

        // Valid dimension
        assert!(trainer.validate_dimension(128).is_ok());

        // Invalid dimension
        assert!(trainer.validate_dimension(100).is_err());
    }
}
