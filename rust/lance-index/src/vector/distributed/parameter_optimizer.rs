// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Adaptive parameter optimization for distributed vector index building

use std::collections::VecDeque;
use std::time::Duration;

// use lance_core::Result;

use crate::vector::hnsw::builder::HnswBuildParams;
use crate::vector::ivf::builder::IvfBuildParams;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BuildMetrics {
    pub ivf_quality_score: f64,
    pub hnsw_connectivity_score: f64,
    pub ivf_training_time: Duration,
    pub hnsw_building_time: Duration,
    pub merge_time: Duration,
    pub peak_memory_usage: usize,
    pub memory_efficiency: f64,
    pub partition_balance_score: f64,
    pub fragment_imbalance: f64,
    pub recall_at_k: Vec<f64>,
    pub precision_at_k: Vec<f64>,
}

impl BuildMetrics {
    pub fn new() -> Self {
        Self {
            ivf_quality_score: 0.0,
            hnsw_connectivity_score: 0.0,
            ivf_training_time: Duration::default(),
            hnsw_building_time: Duration::default(),
            merge_time: Duration::default(),
            peak_memory_usage: 0,
            memory_efficiency: 0.0,
            partition_balance_score: 0.0,
            fragment_imbalance: 0.0,
            recall_at_k: Vec::new(),
            precision_at_k: Vec::new(),
        }
    }
}

#[allow(dead_code)]
pub struct AdaptiveParameterOptimizer {
    metrics_history: VecDeque<BuildMetrics>,
    max_history_size: usize,
    thresholds: OptimizationThresholds,
    learning_rate: f64,
}

impl AdaptiveParameterOptimizer {
    pub fn new() -> Self {
        Self {
            metrics_history: VecDeque::with_capacity(100),
            max_history_size: 100,
            thresholds: OptimizationThresholds::default(),
            learning_rate: 0.1,
        }
    }

    pub fn add_metrics(&mut self, metrics: BuildMetrics) {
        self.metrics_history.push_back(metrics);
        if self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }
    }

    pub fn suggest_parameter_adjustments(
        &self,
        current_ivf_params: &IvfBuildParams,
        current_hnsw_params: &HnswBuildParams,
        data_characteristics: &DataCharacteristics,
    ) -> ParameterAdjustments {
        if self.metrics_history.is_empty() {
            return ParameterAdjustments::default();
        }

        let recent_metrics = self.get_recent_metrics();
        let trends = self.calculate_trends();

        ParameterAdjustments {
            ivf_adjustments: self.suggest_ivf_adjustments(
                current_ivf_params,
                &recent_metrics,
                &trends,
                data_characteristics,
            ),
            hnsw_adjustments: self.suggest_hnsw_adjustments(
                current_hnsw_params,
                &recent_metrics,
                &trends,
                data_characteristics,
            ),
            confidence_score: self.calculate_confidence_score(&recent_metrics),
        }
    }

    fn get_recent_metrics(&self) -> Vec<BuildMetrics> {
        let recent_count = (self.metrics_history.len() / 2).max(5);
        self.metrics_history
            .iter()
            .rev()
            .take(recent_count)
            .cloned()
            .collect()
    }

    fn calculate_trends(&self) -> MetricTrends {
        if self.metrics_history.len() < 2 {
            return MetricTrends::default();
        }

        let recent = self.get_recent_metrics();
        let older = self
            .metrics_history
            .iter()
            .take(self.metrics_history.len() / 2)
            .cloned()
            .collect::<Vec<_>>();

        MetricTrends {
            ivf_quality_trend: self.calculate_trend(&recent, &older, |m| m.ivf_quality_score),
            hnsw_connectivity_trend: self
                .calculate_trend(&recent, &older, |m| m.hnsw_connectivity_score),
            build_time_trend: self.calculate_trend(&recent, &older, |m| {
                m.ivf_training_time.as_secs_f64() + m.hnsw_building_time.as_secs_f64()
            }),
            memory_usage_trend: self
                .calculate_trend(&recent, &older, |m| m.peak_memory_usage as f64),
        }
    }

    fn calculate_trend<F>(
        &self,
        recent: &[BuildMetrics],
        older: &[BuildMetrics],
        extractor: F,
    ) -> f64
    where
        F: Fn(&BuildMetrics) -> f64,
    {
        let recent_avg = recent.iter().map(&extractor).sum::<f64>() / recent.len() as f64;
        let older_avg = older.iter().map(&extractor).sum::<f64>() / older.len() as f64;

        if older_avg == 0.0 {
            0.0
        } else {
            (recent_avg - older_avg) / older_avg
        }
    }

    fn suggest_ivf_adjustments(
        &self,
        current_params: &IvfBuildParams,
        recent_metrics: &[BuildMetrics],
        trends: &MetricTrends,
        data_characteristics: &DataCharacteristics,
    ) -> IvfAdjustments {
        let avg_quality = recent_metrics
            .iter()
            .map(|m| m.ivf_quality_score)
            .sum::<f64>()
            / recent_metrics.len() as f64;
        let _avg_balance = recent_metrics
            .iter()
            .map(|m| m.partition_balance_score)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        let mut adjustments = IvfAdjustments::default();

        // Adjust sample rate based on quality
        if avg_quality < self.thresholds.ivf_quality_threshold {
            let quality_factor = (self.thresholds.ivf_quality_threshold - avg_quality)
                / self.thresholds.ivf_quality_threshold;
            adjustments.sample_rate_multiplier = 1.0 + (quality_factor * self.learning_rate);
        }

        // Adjust max iterations based on quality trends
        if trends.ivf_quality_trend < 0.0 {
            adjustments.max_iters_bonus = (trends.ivf_quality_trend.abs() * 10.0) as usize;
        }

        // Adjust partition count based on data characteristics
        let optimal_partitions = self.calculate_optimal_partitions(
            data_characteristics.total_vectors,
            data_characteristics.target_partition_size,
            data_characteristics.num_fragments,
        );

        if let Some(current_partitions) = current_params.num_partitions {
            let partition_adjustment =
                (optimal_partitions as f64 - current_partitions as f64) / current_partitions as f64;
            adjustments.partition_count_adjustment = partition_adjustment.clamp(-0.5, 0.5);
        }

        adjustments
    }

    /// Suggest HNSW parameter adjustments
    fn suggest_hnsw_adjustments(
        &self,
        current_params: &HnswBuildParams,
        recent_metrics: &[BuildMetrics],
        trends: &MetricTrends,
        data_characteristics: &DataCharacteristics,
    ) -> HnswAdjustments {
        let avg_connectivity = recent_metrics
            .iter()
            .map(|m| m.hnsw_connectivity_score)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        let mut adjustments = HnswAdjustments::default();

        // Adjust M based on connectivity
        if avg_connectivity < self.thresholds.hnsw_connectivity_threshold {
            let connectivity_factor = (self.thresholds.hnsw_connectivity_threshold
                - avg_connectivity)
                / self.thresholds.hnsw_connectivity_threshold;
            adjustments.m_multiplier = 1.0 + (connectivity_factor * self.learning_rate);
        }

        // Adjust ef_construction based on build time trends
        if trends.build_time_trend > 0.1 {
            // Build time is increasing, reduce ef_construction
            adjustments.ef_construction_multiplier = 0.9;
        } else if trends.build_time_trend < -0.1 {
            // Build time is decreasing, can increase ef_construction
            adjustments.ef_construction_multiplier = 1.1;
        }

        // Adjust max_level based on data size
        let optimal_max_level = self.calculate_optimal_max_level(
            data_characteristics.total_vectors,
            data_characteristics.dimension,
        );

        adjustments.max_level_adjustment = (optimal_max_level as f64
            - current_params.max_level as f64)
            / current_params.max_level as f64;

        adjustments
    }

    /// Calculate optimal partition count
    fn calculate_optimal_partitions(
        &self,
        total_vectors: usize,
        target_partition_size: usize,
        num_fragments: usize,
    ) -> usize {
        let base_partitions = total_vectors / target_partition_size;

        // Ensure at least one partition per fragment
        let min_partitions = num_fragments;

        // Cap maximum partitions
        let max_partitions = (base_partitions * 2).min(4096);

        base_partitions.clamp(min_partitions, max_partitions)
    }

    /// Calculate optimal max_level for HNSW
    fn calculate_optimal_max_level(&self, total_vectors: usize, dimension: usize) -> u16 {
        // Based on HNSW paper recommendations
        let log_vectors = (total_vectors as f64).log2().max(1.0);
        let log_dim = (dimension as f64).log2().max(1.0);

        (log_vectors * 0.5 + log_dim * 0.3).round() as u16
    }

    /// Calculate confidence score for suggestions
    fn calculate_confidence_score(&self, recent_metrics: &[BuildMetrics]) -> f64 {
        if recent_metrics.len() < 3 {
            return 0.5;
        }

        let quality_variance = self.calculate_variance(recent_metrics, |m| m.ivf_quality_score);
        let time_variance = self.calculate_variance(recent_metrics, |m| {
            m.ivf_training_time.as_secs_f64() + m.hnsw_building_time.as_secs_f64()
        });

        // Higher variance means lower confidence
        let quality_confidence = 1.0 - (quality_variance / 0.1).min(1.0);
        let time_confidence = 1.0 - (time_variance / 100.0).min(1.0);

        (quality_confidence + time_confidence) / 2.0
    }

    /// Calculate variance of a metric
    fn calculate_variance<F>(&self, metrics: &[BuildMetrics], extractor: F) -> f64
    where
        F: Fn(&BuildMetrics) -> f64,
    {
        let mean = metrics.iter().map(&extractor).sum::<f64>() / metrics.len() as f64;
        let variance = metrics
            .iter()
            .map(|m| {
                let val = extractor(m);
                (val - mean).powi(2)
            })
            .sum::<f64>()
            / metrics.len() as f64;

        variance.sqrt()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OptimizationThresholds {
    pub ivf_quality_threshold: f64,
    pub hnsw_connectivity_threshold: f64,
    pub build_time_threshold: Duration,
    pub memory_usage_threshold: f64,
    pub partition_balance_threshold: f64,
}

impl Default for OptimizationThresholds {
    fn default() -> Self {
        Self {
            ivf_quality_threshold: 0.8,
            hnsw_connectivity_threshold: 0.85,
            build_time_threshold: Duration::from_secs(3600),
            memory_usage_threshold: 0.8,
            partition_balance_threshold: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DataCharacteristics {
    pub total_vectors: usize,
    pub dimension: usize,
    pub num_fragments: usize,
    pub target_partition_size: usize,
    pub data_distribution: DataDistribution,
    pub sparsity: f64,
}

impl DataCharacteristics {
    pub fn new(total_vectors: usize, dimension: usize, num_fragments: usize) -> Self {
        Self {
            total_vectors,
            dimension,
            num_fragments,
            target_partition_size: 8192, // Default
            data_distribution: DataDistribution::Uniform,
            sparsity: 0.0,
        }
    }
}

/// Data distribution characteristics
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum DataDistribution {
    Uniform,
    Skewed,
    Clustered,
    Sparse,
}

/// Parameter adjustments suggested by the optimizer
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ParameterAdjustments {
    pub ivf_adjustments: IvfAdjustments,
    pub hnsw_adjustments: HnswAdjustments,
    pub confidence_score: f64,
}

impl ParameterAdjustments {
    pub fn apply_to_ivf(&self, params: &mut IvfBuildParams) {
        params.sample_rate =
            (params.sample_rate as f64 * self.ivf_adjustments.sample_rate_multiplier) as usize;
        params.max_iters += self.ivf_adjustments.max_iters_bonus;

        if let Some(current_partitions) = params.num_partitions {
            let new_partitions = (current_partitions as f64
                * (1.0 + self.ivf_adjustments.partition_count_adjustment))
                as usize;
            params.num_partitions = Some(new_partitions.max(1));
        }
    }

    pub fn apply_to_hnsw(&self, params: &mut HnswBuildParams) {
        params.m = (params.m as f64 * self.hnsw_adjustments.m_multiplier) as usize;
        params.ef_construction = (params.ef_construction as f64
            * self.hnsw_adjustments.ef_construction_multiplier)
            as usize;
        params.max_level =
            (params.max_level as f64 * (1.0 + self.hnsw_adjustments.max_level_adjustment)) as u16;
    }
}

/// IVF parameter adjustments
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct IvfAdjustments {
    pub sample_rate_multiplier: f64,
    pub max_iters_bonus: usize,
    pub partition_count_adjustment: f64,
}

/// HNSW parameter adjustments
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct HnswAdjustments {
    pub m_multiplier: f64,
    pub ef_construction_multiplier: f64,
    pub max_level_adjustment: f64,
}

/// Metric trends for optimization
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct MetricTrends {
    pub ivf_quality_trend: f64,
    pub hnsw_connectivity_trend: f64,
    pub build_time_trend: f64,
    pub memory_usage_trend: f64,
}

impl Default for BuildMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AdaptiveParameterOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
