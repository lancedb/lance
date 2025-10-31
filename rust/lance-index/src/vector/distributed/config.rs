// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Configuration for distributed vector index building

use crate::vector::hnsw::builder::HnswBuildParams;
use crate::vector::ivf::builder::IvfBuildParams;

/// Configuration for distributed IVF training
#[derive(Debug, Clone)]
pub struct DistributedIvfConfig {
    /// Base IVF parameters
    pub base_params: IvfBuildParams,

    /// Multiplier for sample rate in distributed training
    pub sample_rate_multiplier: f64,

    /// Additional iterations for distributed K-means
    pub max_iters_bonus: usize,

    /// Quality threshold for centroids validation
    pub centroids_quality_threshold: f64,

    /// Enable adaptive retraining if quality is low
    pub enable_adaptive_retraining: bool,
}

impl Default for DistributedIvfConfig {
    fn default() -> Self {
        Self {
            base_params: IvfBuildParams::default(),
            sample_rate_multiplier: 2.0,
            max_iters_bonus: 20,
            centroids_quality_threshold: 0.8,
            enable_adaptive_retraining: true,
        }
    }
}

/// Configuration for distributed HNSW building
#[derive(Debug, Clone)]
pub struct DistributedHnswConfig {
    /// Base HNSW parameters
    pub base_params: HnswBuildParams,

    /// Multiplier for M (number of connections) to compensate for graph partitioning
    pub m_multiplier: f64,

    /// Multiplier for ef_construction to improve quality
    pub ef_construction_multiplier: f64,

    /// Enable connectivity optimization after merging
    pub enable_connectivity_optimization: bool,

    /// Search radius for weak node optimization
    pub optimization_search_radius: usize,
}

impl Default for DistributedHnswConfig {
    fn default() -> Self {
        Self {
            base_params: HnswBuildParams::default(),
            m_multiplier: 1.5,
            ef_construction_multiplier: 1.2,
            enable_connectivity_optimization: true,
            optimization_search_radius: 50,
        }
    }
}

/// Configuration for distributed vector index building
#[derive(Debug, Clone)]
pub struct DistributedVectorIndexConfig {
    /// IVF configuration
    pub ivf_config: DistributedIvfConfig,

    /// HNSW configuration
    pub hnsw_config: DistributedHnswConfig,

    /// Number of fragments to process in parallel
    pub max_parallelism: usize,

    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for DistributedVectorIndexConfig {
    fn default() -> Self {
        Self {
            ivf_config: DistributedIvfConfig::default(),
            hnsw_config: DistributedHnswConfig::default(),
            max_parallelism: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            batch_size: 10000,
        }
    }
}
