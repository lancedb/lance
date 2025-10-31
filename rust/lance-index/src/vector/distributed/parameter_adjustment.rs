// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Parameter adjustment strategies for distributed vector index building

use super::parameter_optimizer::{AdaptiveParameterOptimizer, DataCharacteristics};
use crate::vector::hnsw::builder::HnswBuildParams;
use crate::vector::ivf::builder::IvfBuildParams;
use lance_core::Result;

/// IVF parameter adjustments for distributed execution
pub fn adjust_ivf_params_for_distributed(
    base_params: &IvfBuildParams,
    num_fragments: usize,
    total_vectors: usize,
) -> IvfBuildParams {
    log::info!(
        "Adjusting IVF parameters for distributed execution: {} fragments, {} vectors",
        num_fragments,
        total_vectors
    );

    let mut adjusted_params = base_params.clone();

    // Increase sample rate to compensate distributed sampling bias
    adjusted_params.sample_rate = base_params.sample_rate * 2;

    // Increase max iterations to ensure convergence
    adjusted_params.max_iters = base_params.max_iters + 20;

    // Adjust partition count based on number of fragments
    if let Some(base_partitions) = base_params.num_partitions {
        adjusted_params.num_partitions = Some(optimize_partition_count(
            base_partitions,
            num_fragments,
            total_vectors,
        ));
    }

    // Adjust shuffle parameters for distributed environment
    adjusted_params.shuffle_partition_batches =
        (base_params.shuffle_partition_batches * num_fragments).min(10240);
    adjusted_params.shuffle_partition_concurrency =
        (base_params.shuffle_partition_concurrency + num_fragments / 2).min(8);

    log::debug!("IVF parameter adjustments: sample_rate: {} -> {}, max_iters: {} -> {}, partitions: {:?} -> {:?}",
        base_params.sample_rate, adjusted_params.sample_rate,
        base_params.max_iters, adjusted_params.max_iters,
        base_params.num_partitions, adjusted_params.num_partitions);

    adjusted_params
}

/// HNSW parameter adjustments for distributed execution
pub fn adjust_hnsw_params_for_distributed(
    base_params: &HnswBuildParams,
    fragment_size_ratio: f64,
) -> HnswBuildParams {
    log::info!(
        "Adjusting HNSW parameters for distributed execution: fragment_size_ratio: {:.3}",
        fragment_size_ratio
    );

    let mut adjusted_params = base_params.clone();

    // Increase connectivity to compensate graph partitioning
    adjusted_params.m = ((base_params.m as f64) * 1.5) as usize;

    // Increase search depth during construction
    adjusted_params.ef_construction = ((base_params.ef_construction as f64) * 1.2) as usize;

    // Adjust max level based on fragment size ratio
    adjusted_params.max_level = adjust_max_level(base_params.max_level, fragment_size_ratio);

    log::debug!(
        "HNSW parameter adjustments: m: {} -> {}, ef_construction: {} -> {}, max_level: {} -> {}",
        base_params.m,
        adjusted_params.m,
        base_params.ef_construction,
        adjusted_params.ef_construction,
        base_params.max_level,
        adjusted_params.max_level
    );

    adjusted_params
}

/// Optimize partition count
fn optimize_partition_count(
    base_partitions: usize,
    num_fragments: usize,
    total_vectors: usize,
) -> usize {
    // Ensure each fragment has at least one partition
    let min_partitions = num_fragments;

    // Adjust partitions based on data volume and fragment count
    let fragment_adjustment = (num_fragments as f64).sqrt();
    let size_adjustment = if total_vectors > 1_000_000 {
        1.2
    } else if total_vectors > 100_000 {
        1.1
    } else {
        1.0
    };

    let adjusted_partitions =
        ((base_partitions as f64) * fragment_adjustment * size_adjustment) as usize;

    // Limit partition count range
    let max_partitions = (total_vectors / 1000).min(4096); // at least 1000 vectors per partition, max 4096 partitions

    adjusted_partitions.clamp(min_partitions, max_partitions)
}

/// Adjust HNSW max level
fn adjust_max_level(base_max_level: u16, fragment_size_ratio: f64) -> u16 {
    if fragment_size_ratio < 0.1 {
        // Small fragment: reduce level
        (base_max_level as f64 * 0.8) as u16
    } else if fragment_size_ratio > 0.5 {
        // Large fragment: increase level
        (base_max_level as f64 * 1.2) as u16
    } else {
        base_max_level
    }
}

/// Dynamic parameter optimizer
pub struct DistributedParameterOptimizer {
    optimizer: AdaptiveParameterOptimizer,
    data_characteristics: DataCharacteristics,
}

impl DistributedParameterOptimizer {
    pub fn new(total_vectors: usize, dimension: usize, num_fragments: usize) -> Self {
        Self {
            optimizer: AdaptiveParameterOptimizer::new(),
            data_characteristics: DataCharacteristics::new(total_vectors, dimension, num_fragments),
        }
    }

    /// Get optimized IVF parameters
    pub fn get_optimized_ivf_params(&self, base_params: &IvfBuildParams) -> Result<IvfBuildParams> {
        // First apply distributed adjustments
        let distributed_params = adjust_ivf_params_for_distributed(
            base_params,
            self.data_characteristics.num_fragments,
            self.data_characteristics.total_vectors,
        );

        // Then apply adaptive optimization
        let adjustments = self.optimizer.suggest_parameter_adjustments(
            &distributed_params,
            &HnswBuildParams::default(), // Temporarily use default values
            &self.data_characteristics,
        );

        let mut optimized_params = distributed_params;
        adjustments.apply_to_ivf(&mut optimized_params);

        Ok(optimized_params)
    }

    /// Get optimized HNSW parameters
    pub fn get_optimized_hnsw_params(
        &self,
        base_params: &HnswBuildParams,
    ) -> Result<HnswBuildParams> {
        // Compute fragment size ratio
        let fragment_size_ratio = 1.0 / self.data_characteristics.num_fragments as f64;

        // First apply distributed adjustments
        let distributed_params =
            adjust_hnsw_params_for_distributed(base_params, fragment_size_ratio);

        // Then apply adaptive optimization
        let adjustments = self.optimizer.suggest_parameter_adjustments(
            &IvfBuildParams::default(), // Temporarily use default values
            &distributed_params,
            &self.data_characteristics,
        );

        let mut optimized_params = distributed_params;
        adjustments.apply_to_hnsw(&mut optimized_params);

        Ok(optimized_params)
    }

    /// Update data characteristics
    pub fn update_data_characteristics(&mut self, characteristics: DataCharacteristics) {
        self.data_characteristics = characteristics;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_ivf_params_for_distributed() {
        let base_params = IvfBuildParams {
            num_partitions: Some(256),
            sample_rate: 256,
            max_iters: 50,
            shuffle_partition_batches: 1024,
            shuffle_partition_concurrency: 2,
            ..Default::default()
        };

        let adjusted = adjust_ivf_params_for_distributed(&base_params, 4, 1_000_000);

        assert_eq!(adjusted.sample_rate, 512); // doubled
        assert_eq!(adjusted.max_iters, 70); // +20
        assert!(adjusted.num_partitions.unwrap() >= 4); // at least num_fragments
    }

    #[test]
    fn test_adjust_hnsw_params_for_distributed() {
        let base_params = HnswBuildParams {
            m: 20,
            ef_construction: 150,
            max_level: 7,
            ..Default::default()
        };

        let adjusted = adjust_hnsw_params_for_distributed(&base_params, 0.25);

        assert_eq!(adjusted.m, 30); // 1.5x
        assert_eq!(adjusted.ef_construction, 180); // 1.2x
        assert_eq!(adjusted.max_level, 7); // unchanged for normal ratio
    }

    #[test]
    fn test_optimize_partition_count() {
        // Test with small dataset
        let result = optimize_partition_count(256, 4, 100_000);
        assert!(result >= 4); // at least num_fragments

        // Test with large dataset
        let result = optimize_partition_count(256, 8, 10_000_000);
        assert!(result >= 8); // at least num_fragments
        assert!(result <= 4096); // max limit
    }

    #[test]
    fn test_adjust_max_level() {
        // Small fragment
        assert_eq!(adjust_max_level(10, 0.05), 8);

        // Large fragment
        assert_eq!(adjust_max_level(10, 0.8), 12);

        // Normal fragment
        assert_eq!(adjust_max_level(10, 0.3), 10);
    }
}
