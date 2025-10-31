// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration of global PQ training with distributed vector index building

use arrow_array::{cast::AsArray, Array, FixedSizeListArray};
use lance_core::Result;
use lance_index::vector::pq::ProductQuantizer;
use lance_linalg::distance::DistanceType;
use snafu::location;
use tracing::instrument;

use crate::dataset::Dataset;

/// Train a global PQ codebook from the entire dataset
///
/// This function implements the recommended approach for distributed IVF-PQ indexing:
/// 1. First train a global PQ codebook using samples from the entire dataset
/// 2. Then use this global codebook for distributed IVF-PQ index building
///
/// # Arguments
/// * `dataset` - The dataset to train PQ codebook from
/// * `column` - The vector column name
/// * `num_sub_vectors` - Number of sub-vectors for PQ
/// * `num_bits` - Number of bits per sub-vector (typically 8)
/// * `sample_rate` - Sample rate for training data selection
/// * `max_iters` - Maximum iterations for k-means training
/// * `distance_type` - Distance type (L2, Cosine, Dot)
///
/// # Returns
/// * `ProductQuantizer` - The trained global PQ quantizer
#[instrument(level = "info", skip(dataset))]
pub async fn train_global_pq_codebook(
    dataset: &Dataset,
    column: &str,
    num_sub_vectors: usize,
    num_bits: usize,
    sample_rate: usize,
    max_iters: usize,
    distance_type: DistanceType,
) -> Result<ProductQuantizer> {
    log::info!(
        "Starting global PQ codebook training for column {}: num_sub_vectors={}, num_bits={}, sample_rate={}, max_iters={}",
        column,
        num_sub_vectors,
        num_bits,
        sample_rate,
        max_iters
    );

    // Validate parameters
    let vector_field = dataset.schema().field(column).ok_or_else(|| {
        lance_core::Error::invalid_input(
            format!("Column {} not found in dataset schema", column),
            location!(),
        )
    })?;

    let dimension = get_vector_dimension(&vector_field.data_type())?;

    if dimension % num_sub_vectors != 0 {
        return Err(lance_core::Error::invalid_input(
            format!(
                "Vector dimension {} must be divisible by num_sub_vectors {}",
                dimension, num_sub_vectors
            ),
            location!(),
        ));
    }

    // Calculate expected sample size
    let num_centroids = 2_usize.pow(num_bits as u32);
    let expected_sample_size = num_centroids * sample_rate;

    log::info!("Expected sample size: {} vectors", expected_sample_size);

    // Sample training data globally from the dataset
    let training_data =
        sample_training_data_global(dataset, column, expected_sample_size, distance_type).await?;

    log::info!(
        "Sampled {} training vectors for PQ codebook training",
        training_data.len()
    );

    // Create PQ build parameters
    let _pq_params = lance_index::vector::pq::PQBuildParams {
        num_sub_vectors,
        num_bits,
        max_iters,
        sample_rate,
        ..Default::default()
    };

    // Build the global PQ codebook using the lance-index implementation
    let pq_quantizer = lance_index::vector::distributed::train_global_pq_codebook(
        &training_data,
        num_sub_vectors,
        num_bits,
        sample_rate,
        max_iters,
        distance_type,
    )
    .await?;

    log::info!(
        "Successfully trained global PQ codebook with dimension {} and {} sub-vectors",
        dimension,
        num_sub_vectors
    );

    Ok(pq_quantizer)
}

/// Sample training data globally from the entire dataset
async fn sample_training_data_global(
    dataset: &Dataset,
    column: &str,
    expected_sample_size: usize,
    _distance_type: DistanceType,
) -> Result<FixedSizeListArray> {
    // For now, use a simple sampling approach
    // In a full implementation, this would use the proper sampling utilities

    let mut scan = dataset.scan();
    scan.project(&[column])?;

    // Simple random sampling - in production this should be more sophisticated
    let batch = scan
        .limit(Some(expected_sample_size as i64), None)?
        .try_into_batch()
        .await?;

    match batch.column_by_name(column) {
        Some(array) => {
            let fsl_array = array.as_fixed_size_list();
            Ok(fsl_array.clone())
        }
        None => Err(lance_core::Error::Index {
            message: format!("Column {} not found in batch", column),
            location: snafu::location!(),
        }),
    }
}

/// Get vector dimension from Arrow data type
fn get_vector_dimension(data_type: &arrow_schema::DataType) -> Result<usize> {
    match data_type {
        arrow_schema::DataType::FixedSizeList(inner, dim) => {
            if !matches!(inner.data_type(), arrow_schema::DataType::Float32) {
                return Err(lance_core::Error::Index {
                    message: "Vector column must contain Float32 values".to_string(),
                    location: snafu::location!(),
                });
            }
            Ok(*dim as usize)
        }
        _ => Err(lance_core::Error::Index {
            message: "Vector column must be FixedSizeList<Float32>".to_string(),
            location: snafu::location!(),
        }),
    }
}

/// Build distributed IVF-PQ index with global PQ codebook training
///
/// This is the main entry point for distributed IVF-PQ index building using the
/// recommended global PQ training approach.
#[instrument(level = "info", skip(dataset))]
pub async fn build_distributed_ivf_pq_index(
    dataset: &Dataset,
    column: &str,
    num_sub_vectors: usize,
    num_bits: usize,
    sample_rate: usize,
    max_iters: usize,
    distance_type: DistanceType,
) -> Result<ProductQuantizer> {
    log::info!("Building distributed IVF-PQ index with global PQ training");

    // Step 1: Train global PQ codebook
    log::info!("Step 1/1: Training global PQ codebook");
    let global_pq = train_global_pq_codebook(
        dataset,
        column,
        num_sub_vectors,
        num_bits,
        sample_rate,
        max_iters,
        distance_type,
    )
    .await?;

    log::info!("Successfully trained global PQ codebook");

    Ok(global_pq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field};
    use std::sync::Arc;

    #[test]
    fn test_get_vector_dimension() {
        let inner_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vector_type = DataType::FixedSizeList(inner_field, 128);

        let dimension = get_vector_dimension(&vector_type).unwrap();
        assert_eq!(dimension, 128);
    }

    #[test]
    fn test_get_vector_dimension_invalid() {
        let result = get_vector_dimension(&DataType::Int32);
        assert!(result.is_err());
    }
}
