// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::vector::ivf::builder::IvfBuildParams;
use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use snafu::location;
use std::collections::HashMap;
use std::sync::Arc;

/// Fragment data for distributed processing
#[derive(Debug, Clone)]
pub struct Fragment {
    pub id: usize,
    pub data_path: String,
    pub row_count: usize,
    pub sample_override: Option<FixedSizeListArray>,
}

impl Fragment {
    pub async fn count_rows(&self) -> Result<usize> {
        Ok(self.row_count)
    }

    pub async fn sample_vectors(
        &self,
        _column: &str,
        sample_size: usize,
    ) -> Result<FixedSizeListArray> {
        if let Some(fsl) = self.sample_override.as_ref() {
            if fsl.len() <= sample_size {
                return Ok(fsl.clone());
            }
            let dim = fsl.value_length();
            let values = fsl
                .values()
                .as_primitive::<arrow::datatypes::Float32Type>()
                .values();
            let slice = Float32Array::from(values[..(sample_size * dim as usize)].to_vec());
            let field = Arc::new(Field::new("item", DataType::Float32, false));
            return Ok(FixedSizeListArray::new(field, dim, Arc::new(slice), None));
        }

        let dim = 128;
        let actual_sample_size = sample_size.min(self.row_count);
        let values = Float32Array::from(vec![0.0; dim * actual_sample_size]);
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        Ok(FixedSizeListArray::new(
            field,
            dim as i32,
            Arc::new(values),
            None,
        ))
    }
}

/// Distributed IVF training coordinator
pub struct DistributedIvfCoordinator {
    config: IvfBuildParams,
    num_fragments: usize,
    quality_threshold: f32,
    total_dataset_size: usize,
}

impl DistributedIvfCoordinator {
    pub fn new(config: IvfBuildParams, num_fragments: usize) -> Self {
        Self {
            config,
            num_fragments,
            quality_threshold: 0.85,
            total_dataset_size: 1000000,
        }
    }

    pub fn set_quality_threshold(&mut self, threshold: f32) {
        self.quality_threshold = threshold;
    }

    pub fn set_total_dataset_size(&mut self, size: usize) {
        self.total_dataset_size = size;
    }

    pub async fn train_distributed_ivf(
        &self,
        fragments: &[Fragment],
        column: &str,
        num_partitions: usize,
        sample_rate: usize,
    ) -> Result<IvfModel> {
        self.train_distributed_ivf_with_retry(fragments, column, num_partitions, sample_rate, 0)
            .await
    }

    async fn train_distributed_ivf_with_retry(
        &self,
        fragments: &[Fragment],
        column: &str,
        num_partitions: usize,
        sample_rate: usize,
        retry_count: usize,
    ) -> Result<IvfModel> {
        log::info!(
            "Starting distributed IVF training with {} fragments",
            fragments.len()
        );

        let fragment_samples = stream::iter(fragments)
            .map(|fragment| async move {
                let fragment_size = fragment.count_rows().await?;
                let local_sample_size =
                    ((sample_rate as f64) * (num_partitions as f64) * (fragment_size as f64))
                        / (self.total_dataset_size as f64);
                let local_sample_size = local_sample_size.max(100.0) as usize;

                log::debug!(
                    "Fragment {}: {} vectors from {} total",
                    fragment.id,
                    local_sample_size,
                    fragment_size
                );

                fragment.sample_vectors(column, local_sample_size).await
            })
            .buffered(self.num_fragments.min(8))
            .try_collect::<Vec<_>>()
            .await?;

        let merged_samples = self.merge_fragment_samples(fragment_samples)?;

        let centroids = self
            .train_kmeans_distributed(merged_samples, num_partitions)
            .await?;

        let quality_score = self.validate_centroids_quality(&centroids, fragments)?;

        if quality_score < self.quality_threshold as f64 && retry_count < 3 {
            log::warn!(
                "Centroids quality {} below threshold {}, triggering retraining (attempt {})",
                quality_score,
                self.quality_threshold,
                retry_count + 1
            );

            return Box::pin(self.train_distributed_ivf_with_retry(
                fragments,
                column,
                num_partitions,
                sample_rate.saturating_mul(2),
                retry_count + 1,
            ))
            .await;
        } else if retry_count >= 3 {
            log::warn!(
                "Maximum retry attempts reached, returning current model with quality {:.3}",
                quality_score
            );
        }

        log::info!(
            "Distributed IVF training completed with quality score: {:.3}",
            quality_score
        );

        Ok(IvfModel {
            centroids: Some(Arc::new(centroids)),
            num_partitions,
        })
    }

    /// Merge fragment samples
    fn merge_fragment_samples(
        &self,
        fragment_samples: Vec<FixedSizeListArray>,
    ) -> Result<FixedSizeListArray> {
        if fragment_samples.is_empty() {
            return Err(Error::Index {
                message: "No fragment samples to merge".to_string(),
                location: location!(),
            });
        }

        let first_sample = &fragment_samples[0];
        let dim = first_sample.value_length() as usize;
        let total_samples: usize = fragment_samples.iter().map(|s| s.len()).sum();

        log::info!(
            "Merging {} samples, {} vectors, dim {}",
            fragment_samples.len(),
            total_samples,
            dim
        );

        let mut merged_values = Vec::with_capacity(total_samples * dim);

        for sample in fragment_samples {
            let values = sample
                .values()
                .as_primitive::<arrow::datatypes::Float32Type>();
            merged_values.extend_from_slice(values.values());
        }

        let merged_array = Float32Array::from(merged_values);
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        Ok(FixedSizeListArray::new(
            field,
            dim as i32,
            Arc::new(merged_array),
            None,
        ))
    }

    /// Distributed K-means (local merged samples for quick validation)
    #[allow(clippy::needless_range_loop)]
    pub(crate) async fn train_kmeans_distributed(
        &self,
        training_data: FixedSizeListArray,
        num_partitions: usize,
    ) -> Result<FixedSizeListArray> {
        log::info!(
            "Training K-means with {} samples, {} partitions",
            training_data.len(),
            num_partitions
        );

        let dim = training_data.value_length() as usize;
        let values = training_data
            .values()
            .as_primitive::<arrow::datatypes::Float32Type>();

        // Initialization: deterministic farthest-point (KMeans++-like) to improve robustness
        if training_data.len() < num_partitions {
            return Err(Error::Index {
                message: format!(
                    "Not enough samples for kmeans: k={} > n={}",
                    num_partitions,
                    training_data.len()
                ),
                location: location!(),
            });
        }
        let mut centroids = vec![0f32; num_partitions * dim];
        // First centroid: mean of all samples (stable and reproducible)
        {
            let inv = 1f32 / training_data.len() as f32;
            for i in 0..training_data.len() {
                let row = &values.values()[i * dim..(i + 1) * dim];
                for d in 0..dim {
                    centroids[d] += row[d] * inv;
                }
            }
        }
        // Subsequent centroids: farthest-point from the nearest existing centroid
        let mut chosen = vec![0usize; num_partitions];
        chosen[0] = usize::MAX; // marker for mean, not an actual row id
        for k in 1..num_partitions {
            let mut best_idx = 0usize;
            let mut best_dist = f32::MIN;
            for i in 0..training_data.len() {
                let row = &values.values()[i * dim..(i + 1) * dim];
                // distance to nearest existing centroid
                let mut min_dist = f32::MAX;
                for c in 0..k {
                    let c_off = c * dim;
                    let mut dist = 0f32;
                    for d in 0..dim {
                        let diff = row[d] - centroids[c_off + d];
                        dist += diff * diff;
                    }
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = i;
                }
            }
            chosen[k] = best_idx;
            let c_off = k * dim;
            let row = &values.values()[best_idx * dim..(best_idx + 1) * dim];
            centroids[c_off..c_off + dim].copy_from_slice(row);
        }

        // EM iterations
        let max_iters = self.config.max_iters.max(10);
        let tol = 1e-4f32;
        let mut last_loss = f64::MAX;
        for _ in 0..max_iters {
            // E-step: assignment and local sums
            let mut sums = vec![vec![0f32; dim]; num_partitions];
            let mut counts = vec![0usize; num_partitions];
            let mut loss_acc = 0f64;
            for i in 0..training_data.len() {
                let row = &values.values()[i * dim..(i + 1) * dim];
                // Nearest centroid (L2)
                let mut best = 0usize;
                let mut best_dist = f32::MAX;
                for k in 0..num_partitions {
                    let mut dist = 0f32;
                    let c = &centroids[k * dim..(k + 1) * dim];
                    for d in 0..dim {
                        let diff = row[d] - c[d];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best = k;
                    }
                }
                counts[best] += 1;
                for d in 0..dim {
                    sums[best][d] += row[d];
                }
                loss_acc += best_dist as f64;
            }
            // M-step: update centroids; keep centroid if cluster is empty
            for k in 0..num_partitions {
                if counts[k] > 0 {
                    let inv = 1f32 / counts[k] as f32;
                    for d in 0..dim {
                        centroids[k * dim + d] = sums[k][d] * inv;
                    }
                }
            }
            if (last_loss - loss_acc).abs() <= tol as f64 * (loss_acc.max(1.0)) {
                break;
            }
            last_loss = loss_acc;
        }

        let centroids_array = Float32Array::from(centroids);
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        Ok(FixedSizeListArray::new(
            field,
            dim as i32,
            Arc::new(centroids_array),
            None,
        ))
    }

    /// AllReduce-based K-means for multi-worker verification
    #[allow(clippy::needless_range_loop)]
    pub async fn train_kmeans_allreduce(
        &self,
        local_data: &FixedSizeListArray,
        num_partitions: usize,
        comm: &dyn super::communicator::Communicator,
    ) -> Result<FixedSizeListArray> {
        let dim = local_data.value_length() as usize;
        let values = local_data
            .values()
            .as_primitive::<arrow::datatypes::Float32Type>();
        if num_partitions == 0 {
            return Err(Error::Index {
                message: "k must be > 0".to_string(),
                location: location!(),
            });
        }
        if local_data.len() < num_partitions && comm.rank() == 0 {
            return Err(Error::invalid_input(
                format!(
                    "k {} greater than local sample {}; training may be unstable",
                    num_partitions,
                    local_data.len()
                ),
                location!(),
            ));
        }
        // Deterministic init: root uses first k vectors (wrap around) then broadcasts
        let mut centroids = vec![vec![0f32; dim]; num_partitions];
        if comm.rank() == 0 {
            for k in 0..num_partitions {
                let src = k % local_data.len();
                let row = &values.values()[src * dim..(src + 1) * dim];
                centroids[k].copy_from_slice(row);
            }
        }
        comm.bcast_centroids(&mut centroids, 0);

        let max_iters = self.config.max_iters.max(10);
        let _tol = 1e-4f64;
        let mut _last_loss_glb = f64::MAX;
        for _ in 0..max_iters {
            // Local E-step
            let mut local_sums = vec![vec![0f32; dim]; num_partitions];
            let mut local_cnt = vec![0usize; num_partitions];
            let mut local_loss = 0f64;
            for i in 0..local_data.len() {
                let row = &values.values()[i * dim..(i + 1) * dim];
                let mut best = 0usize;
                let mut best_dist = f32::MAX;
                for k in 0..num_partitions {
                    let mut dist = 0f32;
                    for d in 0..dim {
                        let diff = row[d] - centroids[k][d];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best = k;
                    }
                }
                local_cnt[best] += 1;
                for d in 0..dim {
                    local_sums[best][d] += row[d];
                }
                local_loss += best_dist as f64;
            }
            // AllReduce
            let (glob_sums, glob_cnt) = comm.allreduce_sums_counts(&local_sums, &local_cnt);
            // Update centroids using global sums
            for k in 0..num_partitions {
                if glob_cnt[k] > 0 {
                    let inv = 1f32 / glob_cnt[k] as f32;
                    for d in 0..dim {
                        centroids[k][d] = glob_sums[k][d] * inv;
                    }
                }
            }
            // Convergence check: fixed iterations; no early exit before barrier
            _last_loss_glb = local_loss;
            comm.barrier();
        }
        // Pack result
        let flat: Vec<f32> = centroids.into_iter().flatten().collect();
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        Ok(FixedSizeListArray::new(
            field,
            dim as i32,
            Arc::new(Float32Array::from(flat)),
            None,
        ))
    }

    /// Validate centroid quality
    pub fn validate_centroids_quality(
        &self,
        centroids: &FixedSizeListArray,
        fragments: &[Fragment],
    ) -> Result<f64> {
        if centroids.is_empty() {
            return Ok(0.0);
        }

        // Compute intra/inter cluster metrics
        let intra_cluster_variance = self.compute_intra_cluster_variance(centroids, fragments)?;
        let inter_cluster_distance = self.compute_inter_cluster_distance(centroids)?;

        // Return quality score = inter / intra variance
        let quality_score = if intra_cluster_variance > 0.0 {
            inter_cluster_distance / intra_cluster_variance
        } else {
            0.0
        };

        log::debug!(
            "Quality metrics - Inter-cluster: {:.3}, Intra-cluster: {:.3}, Score: {:.3}",
            inter_cluster_distance,
            intra_cluster_variance,
            quality_score
        );

        Ok(quality_score)
    }

    /// Compute intra-cluster variance
    #[allow(clippy::needless_range_loop)]
    fn compute_intra_cluster_variance(
        &self,
        centroids: &FixedSizeListArray,
        _fragments: &[Fragment],
    ) -> Result<f64> {
        // Approximate via average distance between centroids
        let num_centroids = centroids.len();
        if num_centroids < 2 {
            return Ok(1.0);
        }

        let dim = centroids.value_length() as usize;
        let values = centroids
            .values()
            .as_primitive::<arrow::datatypes::Float32Type>();

        let mut total_variance = 0.0f64;
        let mut count = 0;

        for i in 0..num_centroids {
            for j in (i + 1)..num_centroids {
                let mut distance_sq: f32 = 0.0;
                for d in 0..dim {
                    let diff = values.value(i * dim + d) - values.value(j * dim + d);
                    distance_sq += diff * diff;
                }
                total_variance += distance_sq as f64;
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_variance / count as f64
        } else {
            1.0
        })
    }

    /// Compute inter-cluster distance
    #[allow(clippy::needless_range_loop)]
    fn compute_inter_cluster_distance(&self, centroids: &FixedSizeListArray) -> Result<f64> {
        let num_centroids = centroids.len();
        if num_centroids < 2 {
            return Ok(0.0);
        }

        let dim = centroids.value_length() as usize;
        let values = centroids
            .values()
            .as_primitive::<arrow::datatypes::Float32Type>();

        let mut min_distance = f64::MAX;

        for i in 0..num_centroids {
            for j in (i + 1)..num_centroids {
                let mut distance_sq = 0.0;
                for d in 0..dim {
                    let diff = values.value(i * dim + d) - values.value(j * dim + d);
                    distance_sq += (diff * diff) as f64;
                }
                let distance = distance_sq.sqrt();
                if distance < min_distance {
                    min_distance = distance;
                }
            }
        }

        Ok(if min_distance == f64::MAX {
            0.0
        } else {
            min_distance
        })
    }

    pub async fn train_ivf_centroids(&self, _fragments: Vec<String>) -> Result<FixedSizeListArray> {
        // Simplified implementation for compilation
        // In real implementation, this would coordinate distributed training
        let values = Float32Array::from(vec![0.0; 128 * 100]); // 100 centroids, 128D
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        Ok(FixedSizeListArray::new(field, 128, Arc::new(values), None))
    }

    pub fn validate_ivf_quality(&self, centroids: &FixedSizeListArray) -> bool {
        // Simplified quality validation
        centroids.len() > 0
    }

    pub fn adjust_parameters_for_distribution(&mut self) {
        // Adjust k-means parameters for distributed execution
        if self.num_fragments > 1 {
            // Increase sample rate for better coverage
            self.config.sample_rate = (self.config.sample_rate * 2).min(100);
        }
    }
}

/// IVF model
#[derive(Debug, Clone)]
pub struct IvfModel {
    pub centroids: Option<Arc<FixedSizeListArray>>,
    pub num_partitions: usize,
}

impl IvfModel {
    pub fn new(centroids: FixedSizeListArray, _metadata: Option<()>) -> Self {
        let num_partitions = centroids.len();
        Self {
            centroids: Some(Arc::new(centroids)),
            num_partitions,
        }
    }

    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Assign vectors to IVF partitions
    pub fn assign_partitions(
        &self,
        _vectors: &FixedSizeListArray,
    ) -> Result<Vec<PartitionAssignment>> {
        // Simplified: random assignment
        let assignments = vec![PartitionAssignment {
            fragment_id: 0,
            partition_assignments: HashMap::new(),
        }];
        Ok(assignments)
    }
}

/// Partition assignment
#[derive(Debug, Clone)]
pub struct PartitionAssignment {
    pub fragment_id: usize,
    pub partition_assignments: HashMap<usize, Vec<usize>>, // partition_id -> vector_indices
}
