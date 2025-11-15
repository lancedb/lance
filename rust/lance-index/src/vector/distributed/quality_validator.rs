// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::FixedSizeListArray;
use std::collections::HashMap;

/// Quality metrics for distributed IVF-HNSW
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    pub ivf_balance_score: f32,
    pub hnsw_connectivity_score: f32,
    pub recall_at_k: HashMap<usize, f32>,
    pub latency_ms: f32,
    pub memory_usage_mb: f64,
}

/// Validator for distributed index quality
pub struct QualityValidator {
    min_ivf_balance_score: f32,
    min_hnsw_connectivity_score: f32,
    min_recall_threshold: f32,
    max_latency_ms: f32,
}

impl QualityValidator {
    pub fn new() -> Self {
        Self {
            min_ivf_balance_score: 0.7,
            min_hnsw_connectivity_score: 0.8,
            min_recall_threshold: 0.9,
            max_latency_ms: 100.0,
        }
    }

    pub fn set_thresholds(
        &mut self,
        ivf_balance: f32,
        hnsw_connectivity: f32,
        recall: f32,
        latency: f32,
    ) {
        self.min_ivf_balance_score = ivf_balance;
        self.min_hnsw_connectivity_score = hnsw_connectivity;
        self.min_recall_threshold = recall;
        self.max_latency_ms = latency;
    }

    pub fn validate_ivf_quality(
        &self,
        _centroids: &FixedSizeListArray,
        partition_sizes: &[usize],
    ) -> bool {
        if partition_sizes.is_empty() {
            return false;
        }

        let total_vectors: usize = partition_sizes.iter().sum();
        let avg_size = total_vectors as f32 / partition_sizes.len() as f32;

        // Calculate balance score (lower variance = better)
        let variance: f32 = partition_sizes
            .iter()
            .map(|&size| {
                let diff = size as f32 - avg_size;
                diff * diff
            })
            .sum::<f32>()
            / partition_sizes.len() as f32;

        let balance_score = 1.0 - (variance.sqrt() / avg_size).min(1.0);

        balance_score >= self.min_ivf_balance_score
    }

    pub fn validate_hnsw_quality(&self, connectivity_stats: &HashMap<usize, usize>) -> bool {
        if connectivity_stats.is_empty() {
            return false;
        }

        let total_nodes: usize = connectivity_stats.keys().len();
        let connected_nodes: usize = connectivity_stats
            .values()
            .filter(|&&degree| degree > 0)
            .count();

        let connectivity_score = connected_nodes as f32 / total_nodes as f32;

        connectivity_score >= self.min_hnsw_connectivity_score
    }

    pub fn validate_overall_quality(&self, metrics: &QualityMetrics) -> bool {
        metrics.ivf_balance_score >= self.min_ivf_balance_score
            && metrics.hnsw_connectivity_score >= self.min_hnsw_connectivity_score
            && metrics.recall_at_k.get(&10).unwrap_or(&0.0) >= &self.min_recall_threshold
            && metrics.latency_ms <= self.max_latency_ms
    }

    pub fn generate_quality_report(&self, metrics: &QualityMetrics) -> String {
        format!(
            "Quality Report:
            IVF Balance Score: {:.2} (threshold: {:.2})
            HNSW Connectivity Score: {:.2} (threshold: {:.2})
            Recall@10: {:.2} (threshold: {:.2})
            Latency: {:.2}ms (max: {:.2}ms)
            Memory Usage: {:.2}MB",
            metrics.ivf_balance_score,
            self.min_ivf_balance_score,
            metrics.hnsw_connectivity_score,
            self.min_hnsw_connectivity_score,
            metrics.recall_at_k.get(&10).unwrap_or(&0.0),
            self.min_recall_threshold,
            metrics.latency_ms,
            self.max_latency_ms,
            metrics.memory_usage_mb
        )
    }
}

impl Default for QualityValidator {
    fn default() -> Self {
        Self::new()
    }
}
