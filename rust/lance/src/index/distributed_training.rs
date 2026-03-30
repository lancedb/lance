// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Distributed Vector Index Training
//!
//! Provides two training strategies for IVF-based vector indices:
//!
//! - **Strategy A (`train_local`)**: Single-node training using existing `build_ivf_model` /
//!   `build_pq_model` functions. Suitable for moderate data sizes.
//!
//! - **Strategy B**: Distributed sampling where workers sample data in parallel, then a master
//!   node aggregates and trains. The three-step flow is:
//!   1. `create_sample_tasks` — plan sampling across workers
//!   2. `execute_sample_task` — each worker samples its assigned fragments
//!   3. `train_from_samples` — master concatenates samples and trains models
//!
//! This module also computes global SQ bounds in the distributed path, ensuring consistent
//! scalar quantization across all workers.
//!
//! **Scope**: Rust layer only. Proto schema, Java JNI, and serialization are deferred.

use std::ops::Range;
use std::sync::Arc;

use arrow::compute::concat;
use arrow_array::{Array, FixedSizeListArray};
use lance_core::{Error, Result};
use lance_index::progress::IndexBuildProgress;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::builder::PQBuildParams;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::{Quantization, QuantizerBuildParams};
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::sq::ScalarQuantizer;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl_owned;
use tracing::warn;

use crate::dataset::Dataset;

use super::vector::ivf::{build_ivf_model, train_ivf_model};
use super::vector::pq::build_pq_model;
use super::vector::utils::{
    filter_finite_training_data, get_vector_dim, maybe_sample_training_data,
};

/// Configuration for distributed training.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// Vector column name.
    pub column: String,
    /// IVF parameters (must have `num_partitions` set).
    pub ivf_params: IvfBuildParams,
    /// PQ parameters. Mutually exclusive with `sq_params`.
    pub pq_params: Option<PQBuildParams>,
    /// SQ parameters. Mutually exclusive with `pq_params`.
    pub sq_params: Option<SQBuildParams>,
    /// Distance metric type.
    pub distance_type: DistanceType,
    /// Progress callback for training stages.
    pub progress: Arc<dyn IndexBuildProgress>,
}

/// Trained model outputs from either strategy.
#[derive(Clone, Debug)]
pub struct TrainedModels {
    /// IVF model (centroids + KMeans loss).
    pub ivf_model: IvfModel,
    /// PQ model if PQ training was requested.
    pub pq: Option<ProductQuantizer>,
    /// Global SQ bounds (min..max across all dimensions) if SQ training was requested.
    pub sq_bounds: Option<Range<f64>>,
    /// The distance type used for training (may differ from the original if Cosine was remapped
    /// to L2 after normalization — but this field stores the *original* user-specified metric).
    pub distance_type: DistanceType,
}

impl TrainedModels {
    /// Number of IVF partitions, derived from the centroids.
    pub fn num_partitions(&self) -> usize {
        self.ivf_model.num_partitions()
    }

    /// Vector dimension, derived from the centroids.
    pub fn dimension(&self) -> usize {
        self.ivf_model.dimension()
    }
}

/// A sampling task for a single worker (Strategy B).
pub struct SampleTask {
    /// Vector column name (self-contained for future serialization).
    pub column: String,
    /// Fragment IDs assigned to this worker.
    pub fragment_ids: Vec<u32>,
    /// Number of vectors to sample.
    pub sample_size: usize,
}

/// Sampling result from a single worker (Strategy B).
pub struct SampleResult {
    /// Sampled vectors (already filtered for NaN/Inf).
    pub sample_data: FixedSizeListArray,
    /// Local SQ bounds if SQ was requested. `None` if SQ is not needed or data was empty.
    pub sq_bounds: Option<Range<f64>>,
    /// Total row count of the fragments assigned to this task.
    pub source_row_count: u64,
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_config_common(config: &TrainingConfig) -> Result<()> {
    // Note: this validation requires explicit num_partitions; `target_partition_size` mode
    // (where num_partitions is derived at runtime) is not supported in the distributed path.
    if config.ivf_params.num_partitions.is_none()
        || config.ivf_params.num_partitions == Some(0)
    {
        return Err(Error::invalid_input(
            "ivf_params.num_partitions must be set and > 0 for distributed training \
             (target_partition_size mode is not supported)",
        ));
    }
    if config.pq_params.is_some() && config.sq_params.is_some() {
        return Err(Error::invalid_input(
            "pq_params and sq_params are mutually exclusive; set at most one",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy A — single-node training
// ---------------------------------------------------------------------------

/// Train IVF (+ optional PQ or SQ) models on a single node.
///
/// This reuses the existing `build_ivf_model` / `build_pq_model` functions and is suitable
/// when the dataset fits comfortably in the master's memory.
pub async fn train_local(
    dataset: &Dataset,
    config: &TrainingConfig,
) -> Result<TrainedModels> {
    validate_config_common(config)?;

    let dim = get_vector_dim(dataset.schema(), &config.column)?;

    // Train IVF
    let ivf_model = build_ivf_model(
        dataset,
        &config.column,
        dim,
        config.distance_type,
        &config.ivf_params,
        None,
        config.progress.clone(),
    )
    .await?;

    // Train PQ if requested
    let pq = if let Some(pq_params) = &config.pq_params {
        Some(
            build_pq_model(
                dataset,
                &config.column,
                dim,
                config.distance_type,
                pq_params,
                Some(&ivf_model),
            )
            .await?,
        )
    } else {
        None
    };

    // Compute SQ bounds if requested
    let sq_bounds = if let Some(sq_params) = &config.sq_params {
        let sample_size = sq_params.sample_size();
        let sample = maybe_sample_training_data(dataset, &config.column, sample_size, None).await?;
        if sample.is_empty() {
            None
        } else {
            // For Cosine metric, normalize before computing bounds (SQ encoding operates on
            // normalized data).
            let sample = if config.distance_type == DistanceType::Cosine {
                normalize_fsl_owned(sample)?
            } else {
                sample
            };
            let sq = ScalarQuantizer::build(&sample, config.distance_type, sq_params)?;
            Some(sq.bounds())
        }
    } else {
        None
    };

    Ok(TrainedModels {
        ivf_model,
        pq,
        sq_bounds,
        distance_type: config.distance_type,
    })
}

// ---------------------------------------------------------------------------
// Strategy B — distributed sampling
// ---------------------------------------------------------------------------

/// Plan sampling tasks for distributed workers.
///
/// Distributes fragments across `num_workers` workers using round-robin assignment.
/// Returns at most `min(num_workers, num_fragments)` tasks (no empty tasks are produced).
pub fn create_sample_tasks(
    dataset: &Dataset,
    config: &TrainingConfig,
    num_workers: usize,
) -> Result<Vec<SampleTask>> {
    validate_config_common(config)?;

    if num_workers == 0 {
        return Err(Error::invalid_input("num_workers must be > 0"));
    }

    let fragments = dataset.get_fragments();
    if fragments.is_empty() {
        return Err(Error::invalid_input(
            "Dataset has no fragments; cannot create sample tasks",
        ));
    }

    let fragment_ids: Vec<u32> = fragments
        .iter()
        .map(|f| {
            u32::try_from(f.id()).map_err(|_| {
                Error::invalid_input(format!(
                    "Fragment ID {} exceeds u32::MAX; cannot use in sampling API",
                    f.id()
                ))
            })
        })
        .collect::<Result<_>>()?;
    let num_fragments = fragment_ids.len();
    let effective_workers = num_workers.min(num_fragments);

    // Compute required sample size as max across IVF, PQ, and SQ requirements.
    let num_partitions = config.ivf_params.num_partitions.unwrap();
    let ivf_sample = num_partitions * config.ivf_params.sample_rate;
    let pq_sample = config
        .pq_params
        .as_ref()
        .map(|p| p.sample_size())
        .unwrap_or(0);
    let sq_sample = config
        .sq_params
        .as_ref()
        .map(|p| p.sample_size())
        .unwrap_or(0);
    let total_sample_size = ivf_sample.max(pq_sample).max(sq_sample);

    // Round-robin assignment of fragments to workers.
    let mut worker_fragments: Vec<Vec<u32>> = (0..effective_workers)
        .map(|i| {
            // Pre-size: each worker gets roughly num_fragments / effective_workers fragments.
            let cap = num_fragments / effective_workers + if i < num_fragments % effective_workers { 1 } else { 0 };
            Vec::with_capacity(cap)
        })
        .collect();
    for (i, fid) in fragment_ids.iter().enumerate() {
        worker_fragments[i % effective_workers].push(*fid);
    }

    // Distribute samples evenly, spreading the remainder across the first workers.
    let base_sample = total_sample_size / effective_workers;
    let remainder = total_sample_size % effective_workers;

    let tasks = worker_fragments
        .into_iter()
        .enumerate()
        .map(|(i, fids)| SampleTask {
            column: config.column.clone(),
            fragment_ids: fids,
            sample_size: base_sample + if i < remainder { 1 } else { 0 },
        })
        .collect();

    Ok(tasks)
}

/// Execute a sampling task on a worker node.
///
/// Samples vectors from the assigned fragments, filters non-finite values, and optionally
/// computes local SQ bounds. When `metric_type` is Cosine, the sample is normalized before
/// computing SQ bounds (to match the representation used during SQ encoding).
pub async fn execute_sample_task(
    dataset: &Dataset,
    task: &SampleTask,
    sq_params: Option<&SQBuildParams>,
    metric_type: DistanceType,
) -> Result<SampleResult> {
    // Sample from the assigned fragments.
    let sample = maybe_sample_training_data(
        dataset,
        &task.column,
        task.sample_size,
        Some(&task.fragment_ids),
    )
    .await?;

    let sample = filter_finite_training_data(sample)?;

    // Count source rows from fragment metadata directly (avoids duplicate I/O).
    let source_row_count: u64 = task
        .fragment_ids
        .iter()
        .filter_map(|fid| dataset.get_fragment(*fid as usize))
        .map(|f| f.metadata().physical_rows.unwrap_or(0) as u64)
        .sum();

    if sample.is_empty() {
        return Ok(SampleResult {
            sample_data: sample,
            sq_bounds: None,
            source_row_count,
        });
    }

    // Compute local SQ bounds if requested.
    let sq_bounds = if let Some(sq_p) = sq_params {
        // Normalize for Cosine before computing bounds.
        if metric_type == DistanceType::Cosine {
            let normalized = normalize_fsl_owned(sample.clone())?;
            let sq = ScalarQuantizer::build(&normalized, metric_type, sq_p)?;
            Some(sq.bounds())
        } else {
            let sq = ScalarQuantizer::build(&sample, metric_type, sq_p)?;
            Some(sq.bounds())
        }
    } else {
        None
    };

    Ok(SampleResult {
        sample_data: sample,
        sq_bounds,
        source_row_count,
    })
}

/// Aggregate sample results and train IVF (+ optional PQ/SQ) models on the master node.
///
/// Concatenates sampled data from all workers, merges SQ bounds globally, normalizes for
/// Cosine if needed, and trains the IVF/PQ models.
pub async fn train_from_samples(
    sample_results: Vec<SampleResult>,
    config: &TrainingConfig,
) -> Result<TrainedModels> {
    validate_config_common(config)?;

    if sample_results.is_empty() {
        return Err(Error::invalid_input(
            "sample_results must not be empty",
        ));
    }

    // Merge SQ bounds from all workers (extract before consuming sample_results).
    let sq_bounds = {
        let sq_results: Vec<&Range<f64>> = sample_results
            .iter()
            .filter_map(|r| r.sq_bounds.as_ref())
            .collect();
        if sq_results.is_empty() {
            None
        } else {
            Some(
                sq_results
                    .iter()
                    .map(|b| b.start)
                    .fold(f64::MAX, f64::min)
                    ..sq_results
                        .iter()
                        .map(|b| b.end)
                        .fold(f64::MIN, f64::max),
            )
        }
    };

    // Concatenate all sample data, then drop the originals to free memory.
    let arrays: Vec<&dyn arrow_array::Array> = sample_results
        .iter()
        .filter(|r| !r.sample_data.is_empty())
        .map(|r| &r.sample_data as &dyn arrow_array::Array)
        .collect();

    if arrays.is_empty() {
        return Err(Error::invalid_input(
            "All sample results are empty; cannot train models",
        ));
    }

    let concatenated = concat(&arrays)?;
    drop(sample_results);
    let concatenated = concatenated.as_any().downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::invalid_input(
            "concat produced unexpected array type; expected FixedSizeListArray",
        ))?
        .clone();

    let num_partitions = config.ivf_params.num_partitions.unwrap();
    if concatenated.len() < num_partitions {
        return Err(Error::invalid_input(format!(
            "Concatenated sample has {} rows but num_partitions is {}; need at least as many rows as partitions",
            concatenated.len(),
            num_partitions,
        )));
    }

    let dim = concatenated.value_length() as usize;

    // Log memory warning for large concatenated samples.
    let elem_size = concatenated
        .value_type()
        .primitive_width()
        .unwrap_or(4) as u64;
    let estimated_bytes = (concatenated.len() as u64) * (dim as u64) * elem_size;
    if estimated_bytes > 4 * 1024 * 1024 * 1024 {
        warn!(
            "Concatenated training data is ~{:.1} GB; consider reducing sample_rate or num_workers",
            estimated_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Normalize for Cosine metric, then remap to L2.
    let (training_data, effective_metric) = if config.distance_type == DistanceType::Cosine {
        (normalize_fsl_owned(concatenated)?, DistanceType::L2)
    } else {
        (concatenated, config.distance_type)
    };

    // Train IVF model.
    let ivf_model = train_ivf_model(
        None,
        &training_data,
        effective_metric,
        &config.ivf_params,
        config.progress.clone(),
    )
    .await?;

    // Train PQ model if requested.
    let pq = if let Some(pq_params) = &config.pq_params {
        // Compute residuals for PQ training (Cosine has been remapped to L2 above).
        let pq_data = if effective_metric == DistanceType::L2 {
            let centroids = ivf_model.centroids.clone().ok_or_else(|| {
                Error::internal("IvfModel missing centroids after training")
            })?;
            let ivf_transformer = lance_index::vector::ivf::new_ivf_transformer(
                centroids,
                DistanceType::L2,
                vec![],
            );
            ivf_transformer.compute_residual(&training_data)?
        } else {
            training_data.clone()
        };
        // Drop training_data to reduce peak memory before PQ build.
        drop(training_data);

        Some(pq_params.build(&pq_data, effective_metric)?)
    } else {
        None
    };

    Ok(TrainedModels {
        ivf_model,
        pq,
        sq_bounds,
        distance_type: config.distance_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::Float32Type;
    use arrow_array::{Array, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_datagen::{Dimension, RowCount, array, gen_batch};
    use lance_index::progress::NoopIndexBuildProgress;
    use rstest::rstest;

    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};

    fn noop_progress() -> Arc<dyn IndexBuildProgress> {
        Arc::new(NoopIndexBuildProgress)
    }

    /// Create a single-fragment in-memory dataset with the given number of rows and dimension.
    async fn create_dataset(num_rows: usize, dim: usize) -> Dataset {
        let batch = gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(dim as u32)),
            )
            .into_batch_rows(RowCount::from(num_rows as u64))
            .unwrap();
        InsertBuilder::new("memory://")
            .execute(vec![batch])
            .await
            .unwrap()
    }

    /// Create a multi-fragment in-memory dataset by appending multiple batches.
    async fn create_multi_fragment_dataset(
        rows_per_fragment: usize,
        num_fragments: usize,
        dim: usize,
    ) -> Dataset {
        assert!(num_fragments > 0);
        let first_batch = gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(dim as u32)),
            )
            .into_batch_rows(RowCount::from(rows_per_fragment as u64))
            .unwrap();
        let mut dataset = InsertBuilder::new("memory://")
            .execute(vec![first_batch])
            .await
            .unwrap();

        for _ in 1..num_fragments {
            let batch = gen_batch()
                .col(
                    "vector",
                    array::rand_vec::<Float32Type>(Dimension::from(dim as u32)),
                )
                .into_batch_rows(RowCount::from(rows_per_fragment as u64))
                .unwrap();
            dataset = InsertBuilder::new(Arc::new(dataset))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                })
                .execute(vec![batch])
                .await
                .unwrap();
        }

        dataset
    }

    /// Create a dataset with extreme value ranges across fragments for SQ bounds testing.
    async fn create_extreme_range_dataset(dim: usize) -> Dataset {
        // Fragment 0: vectors in [0, 1]
        let values_low: Vec<f32> = (0..250 * dim).map(|i| (i % dim) as f32 / dim as f32).collect();
        let fsl_low = FixedSizeListArray::try_new_from_values(
            Float32Array::from(values_low),
            dim as i32,
        )
        .unwrap();
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "vector",
            fsl_low.data_type().clone(),
            false,
        )]));
        let batch_low =
            arrow_array::RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl_low)]).unwrap();

        let dataset = InsertBuilder::new("memory://")
            .execute(vec![batch_low])
            .await
            .unwrap();

        // Fragment 1: vectors in [0, 100]
        let values_high: Vec<f32> =
            (0..250 * dim).map(|i| (i % dim) as f32 * 100.0 / dim as f32).collect();
        let fsl_high = FixedSizeListArray::try_new_from_values(
            Float32Array::from(values_high),
            dim as i32,
        )
        .unwrap();
        let batch_high =
            arrow_array::RecordBatch::try_new(schema, vec![Arc::new(fsl_high)]).unwrap();

        InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![batch_high])
            .await
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // 7.1 Strategy A tests (parameterized: IVF_PQ / IVF_SQ × L2 / Cosine)
    // -----------------------------------------------------------------------

    #[rstest]
    #[case::ivf_pq_l2(Some(PQBuildParams::new(2, 8)), None, DistanceType::L2)]
    #[case::ivf_sq_l2(None, Some(SQBuildParams::default()), DistanceType::L2)]
    #[case::ivf_sq_cosine(None, Some(SQBuildParams::default()), DistanceType::Cosine)]
    #[tokio::test]
    async fn test_train_local(
        #[case] pq_params: Option<PQBuildParams>,
        #[case] sq_params: Option<SQBuildParams>,
        #[case] metric: DistanceType,
    ) {
        let dim = 32;
        let dataset = create_dataset(1000, dim).await;

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(4),
            pq_params: pq_params.clone(),
            sq_params: sq_params.clone(),
            distance_type: metric,
            progress: noop_progress(),
        };

        let models = train_local(&dataset, &config).await.unwrap();

        assert_eq!(models.num_partitions(), 4);
        assert_eq!(models.dimension(), dim);

        if pq_params.is_some() {
            let pq = models.pq.as_ref().unwrap();
            assert_eq!(pq.num_sub_vectors, 2);
        } else {
            assert!(models.pq.is_none());
        }

        if sq_params.is_some() {
            let bounds = models.sq_bounds.as_ref().unwrap();
            assert!(bounds.start < bounds.end, "SQ bounds should be valid: {bounds:?}");
        } else {
            assert!(models.sq_bounds.is_none());
        }
    }

    // -----------------------------------------------------------------------
    // 7.2 Strategy B end-to-end test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_strategy_b_end_to_end() {
        let dim = 32;
        let dataset = create_multi_fragment_dataset(250, 4, dim).await;
        assert_eq!(dataset.get_fragments().len(), 4);

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(4),
            pq_params: Some(PQBuildParams::new(2, 8)),
            sq_params: None,
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        let tasks = create_sample_tasks(&dataset, &config, 2).unwrap();
        assert_eq!(tasks.len(), 2);

        let mut results = Vec::new();
        for task in &tasks {
            let result = execute_sample_task(&dataset, task, None, config.distance_type)
                .await
                .unwrap();
            assert!(!result.sample_data.is_empty());
            results.push(result);
        }

        let models = train_from_samples(results, &config).await.unwrap();

        assert_eq!(models.num_partitions(), 4);
        assert_eq!(models.dimension(), dim);
        assert!(models.pq.is_some());
    }

    // -----------------------------------------------------------------------
    // 7.3 Strategy A vs B structural consistency
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_strategy_a_vs_b_consistency() {
        let dim = 32;
        let dataset = create_multi_fragment_dataset(250, 4, dim).await;

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(4),
            pq_params: None,
            sq_params: Some(SQBuildParams::default()),
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        // Strategy A
        let models_a = train_local(&dataset, &config).await.unwrap();

        // Strategy B
        let tasks = create_sample_tasks(&dataset, &config, 2).unwrap();
        let mut results = Vec::new();
        for task in &tasks {
            let result = execute_sample_task(
                &dataset,
                task,
                config.sq_params.as_ref(),
                config.distance_type,
            )
            .await
            .unwrap();
            results.push(result);
        }
        let models_b = train_from_samples(results, &config).await.unwrap();

        // Structural consistency (not numerical equality due to sampling randomness).
        assert_eq!(models_a.num_partitions(), models_b.num_partitions());
        assert_eq!(models_a.dimension(), models_b.dimension());

        // Both should produce SQ bounds that are valid ranges.
        let bounds_a = models_a.sq_bounds.as_ref().unwrap();
        let bounds_b = models_b.sq_bounds.as_ref().unwrap();
        assert!(bounds_a.start < bounds_a.end);
        assert!(bounds_b.start < bounds_b.end);
    }

    // -----------------------------------------------------------------------
    // 7.4 SQ bounds distributed consistency (extreme values)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_sq_bounds_distributed_consistency() {
        let dim = 16;
        let dataset = create_extreme_range_dataset(dim).await;
        assert_eq!(dataset.get_fragments().len(), 2);

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(2),
            pq_params: None,
            sq_params: Some(SQBuildParams::default()),
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        let tasks = create_sample_tasks(&dataset, &config, 2).unwrap();
        let mut results = Vec::new();
        for task in &tasks {
            let result = execute_sample_task(
                &dataset,
                task,
                config.sq_params.as_ref(),
                config.distance_type,
            )
            .await
            .unwrap();
            results.push(result);
        }
        let models = train_from_samples(results, &config).await.unwrap();

        let bounds = models.sq_bounds.as_ref().unwrap();
        // Global bounds should cover the full range [~0, ~100].
        assert!(bounds.start <= 0.1, "global min should be near 0, got {}", bounds.start);
        assert!(bounds.end >= 90.0, "global max should be near 100, got {}", bounds.end);
    }

    // -----------------------------------------------------------------------
    // 7.5 Edge cases
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_workers_exceed_fragments() {
        let dim = 16;
        let dataset = create_multi_fragment_dataset(100, 2, dim).await;

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(2),
            pq_params: None,
            sq_params: None,
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        // 10 workers but only 2 fragments → should produce 2 tasks
        let tasks = create_sample_tasks(&dataset, &config, 10).unwrap();
        assert_eq!(tasks.len(), 2);
    }

    #[tokio::test]
    async fn test_zero_partitions_error() {
        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams {
                num_partitions: Some(0),
                ..Default::default()
            },
            pq_params: None,
            sq_params: None,
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        let dataset = create_dataset(100, 16).await;
        let err = train_local(&dataset, &config).await.unwrap_err();
        assert!(
            err.to_string().contains("num_partitions"),
            "Expected error about num_partitions, got: {err}",
        );
    }

    #[tokio::test]
    async fn test_pq_and_sq_mutually_exclusive() {
        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(2),
            pq_params: Some(PQBuildParams::new(2, 8)),
            sq_params: Some(SQBuildParams::default()),
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        let dataset = create_dataset(100, 16).await;
        let err = train_local(&dataset, &config).await.unwrap_err();
        assert!(
            err.to_string().contains("mutually exclusive"),
            "Expected mutual exclusivity error, got: {err}",
        );
    }

    #[tokio::test]
    async fn test_empty_sample_results_error() {
        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(2),
            pq_params: None,
            sq_params: None,
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        let err = train_from_samples(vec![], &config).await.unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "Expected empty error, got: {err}",
        );
    }

    // -----------------------------------------------------------------------
    // 7.6 Downstream integration test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_trained_models_integration_with_index_builder() {
        let dim = 32;
        let dataset = create_dataset(1000, dim).await;

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(4),
            pq_params: Some(PQBuildParams::new(2, 8)),
            sq_params: None,
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        let models = train_local(&dataset, &config).await.unwrap();

        // Verify the models can be used with downstream types.
        // IvfModel can be passed to IvfIndexBuilder.with_ivf().
        assert!(models.ivf_model.centroids.is_some());
        assert_eq!(models.ivf_model.num_partitions(), 4);

        // ProductQuantizer is ready to use.
        let pq = models.pq.unwrap();
        assert_eq!(pq.num_sub_vectors, 2);
    }

    #[tokio::test]
    async fn test_sq_bounds_can_construct_quantizer() {
        let dim = 16;
        let dataset = create_dataset(1000, dim).await;

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(2),
            pq_params: None,
            sq_params: Some(SQBuildParams::default()),
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        let models = train_local(&dataset, &config).await.unwrap();

        let bounds = models.sq_bounds.unwrap();
        // Construct a ScalarQuantizer from the trained bounds.
        let sq = ScalarQuantizer::with_bounds(8, dim, bounds.clone());
        assert_eq!(sq.bounds(), bounds);
    }

    // -----------------------------------------------------------------------
    // 7.7 Single-worker distributed path
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_single_worker_distributed() {
        let dim = 32;
        let dataset = create_multi_fragment_dataset(250, 3, dim).await;

        let config = TrainingConfig {
            column: "vector".to_string(),
            ivf_params: IvfBuildParams::new(4),
            pq_params: Some(PQBuildParams::new(2, 8)),
            sq_params: None,
            distance_type: DistanceType::L2,
            progress: noop_progress(),
        };

        // Single worker gets all fragments.
        let tasks = create_sample_tasks(&dataset, &config, 1).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].fragment_ids.len(), 3);

        let result = execute_sample_task(&dataset, &tasks[0], None, config.distance_type)
            .await
            .unwrap();
        assert!(!result.sample_data.is_empty());

        let models = train_from_samples(vec![result], &config).await.unwrap();
        assert_eq!(models.num_partitions(), 4);
        assert_eq!(models.dimension(), dim);
        assert!(models.pq.is_some());
    }

    // -----------------------------------------------------------------------
    // 7.8 Sample remainder distribution
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_remainder_distribution() {
        // Verify that remainder samples are distributed across workers.
        // With 7 total and 4 workers: 2,2,2,1 (not 1,1,1,1 losing 3).
        let total = 7usize;
        let workers = 4usize;
        let base = total / workers;
        let remainder = total % workers;
        let sizes: Vec<usize> = (0..workers)
            .map(|i| base + if i < remainder { 1 } else { 0 })
            .collect();
        assert_eq!(sizes, vec![2, 2, 2, 1]);
        assert_eq!(sizes.iter().sum::<usize>(), total);
    }
}
