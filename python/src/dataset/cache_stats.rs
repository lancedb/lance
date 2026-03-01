// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache statistics exposed to Python

use pyo3::{pyclass, pymethods};

/// Cache statistics for a Lance cache (index or metadata).
///
/// Provides hit/miss counters, eviction tracking, capacity utilization,
/// and derived ratios. Returned by ``Dataset.index_cache_stats()`` and
/// ``Dataset.metadata_cache_stats()``.
#[pyclass(name = "CacheStats", module = "_lib", get_all)]
#[derive(Clone, Debug)]
pub struct PyCacheStats {
    /// Cumulative cache hits.
    pub hits: u64,
    /// Cumulative cache misses.
    pub misses: u64,
    /// Cumulative evictions due to capacity pressure.
    pub evictions: u64,
    /// Number of entries currently in the cache.
    pub num_entries: usize,
    /// Total size in bytes of all entries currently in the cache.
    pub size_bytes: usize,
    /// Maximum capacity in bytes configured for this cache.
    pub max_capacity_bytes: u64,
    /// Hit ratio ∈ [0.0, 1.0]. 0.0 when no lookups have occurred.
    pub hit_ratio: f32,
    /// Cache utilization ∈ [0.0, 1.0] — size_bytes / max_capacity_bytes.
    pub utilization: f32,
}

#[pymethods]
impl PyCacheStats {
    fn __repr__(&self) -> String {
        format!(
            "CacheStats(hits={}, misses={}, evictions={}, num_entries={}, \
             size_bytes={}, max_capacity_bytes={}, hit_ratio={:.4}, utilization={:.4})",
            self.hits,
            self.misses,
            self.evictions,
            self.num_entries,
            self.size_bytes,
            self.max_capacity_bytes,
            self.hit_ratio,
            self.utilization,
        )
    }
}

impl PyCacheStats {
    /// Convert from Lance's internal CacheStats type
    pub fn from_lance(stats: lance_core::cache::CacheStats) -> Self {
        Self {
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            num_entries: stats.num_entries,
            size_bytes: stats.size_bytes,
            max_capacity_bytes: stats.max_capacity_bytes,
            hit_ratio: stats.hit_ratio(),
            utilization: stats.utilization(),
        }
    }
}
