// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A trait used by the index to report metrics
///
/// Callers can implement this trait to collect metrics
pub trait MetricsCollector: Send + Sync {
    /// Record partition loads
    ///
    /// Many indices consist of partitions that may need to be loaded
    /// into cache.  For example, an inverted index or ngram index has a
    /// posting list for each token.
    ///
    /// In the ideal case, these shards are in the cache and will not need
    /// to be loaded from disk.  This method should not be called if the
    /// shard is in the cache.
    fn record_parts_loaded(&self, num_parts: usize);

    /// Record a shard load
    fn record_part_load(&self) {
        self.record_parts_loaded(1);
    }

    /// Record an index load
    ///
    /// This should be called when a scalar index is loaded from storage.
    /// It should not be called if the index is already in memory.
    fn record_index_loads(&self, num_indexes: usize);

    /// Record an index load
    fn record_index_load(&self) {
        self.record_index_loads(1);
    }

    /// Record the number of "comparisons" made by the index
    ///
    /// What exactly constitutes a comparison depends on the index type.
    /// For example, a B-tree index may make comparisons while searching for a value.
    /// On the other hand, a bitmap index makes comparisons when computing the intersection
    /// of two bitmaps.
    ///
    /// The goal is to provide some visibility into the compute cost of the search
    fn record_comparisons(&self, num_comparisons: usize);

    /// Record bytes loaded from storage during index operations
    ///
    /// This tracks the total number of bytes read from disk/object store
    /// for index-related data (partition data, posting lists, graph nodes, etc.).
    fn record_bytes_loaded(&self, _num_bytes: u64) {}

    /// Record the number of partitions probed during a search
    fn record_partitions_probed(&self, _count: usize) {}

    /// Record the number of candidate vectors evaluated during a search
    fn record_candidates_evaluated(&self, _count: usize) {}

    /// Record the wall-clock duration of a search operation in microseconds
    fn record_search_duration_us(&self, _duration: u64) {}
}

/// A no-op metrics collector that does nothing
pub struct NoOpMetricsCollector;

impl MetricsCollector for NoOpMetricsCollector {
    fn record_parts_loaded(&self, _num_parts: usize) {}
    fn record_index_loads(&self, _num_indexes: usize) {}
    fn record_comparisons(&self, _num_comparisons: usize) {}
}

#[derive(Default)]
pub struct LocalMetricsCollector {
    pub parts_loaded: AtomicUsize,
    pub index_loads: AtomicUsize,
    pub comparisons: AtomicUsize,
    pub bytes_loaded: AtomicU64,
    pub partitions_probed: AtomicUsize,
    pub candidates_evaluated: AtomicUsize,
    pub search_duration_us: AtomicU64,
}

impl LocalMetricsCollector {
    pub fn dump_into(self, other: &dyn MetricsCollector) {
        other.record_parts_loaded(self.parts_loaded.load(Ordering::Relaxed));
        other.record_index_loads(self.index_loads.load(Ordering::Relaxed));
        other.record_comparisons(self.comparisons.load(Ordering::Relaxed));
        other.record_bytes_loaded(self.bytes_loaded.load(Ordering::Relaxed));
        other.record_partitions_probed(self.partitions_probed.load(Ordering::Relaxed));
        other.record_candidates_evaluated(self.candidates_evaluated.load(Ordering::Relaxed));
        other.record_search_duration_us(self.search_duration_us.load(Ordering::Relaxed));
    }
}

impl MetricsCollector for LocalMetricsCollector {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.parts_loaded.fetch_add(num_parts, Ordering::Relaxed);
    }

    fn record_index_loads(&self, num_indexes: usize) {
        self.index_loads.fetch_add(num_indexes, Ordering::Relaxed);
    }

    fn record_comparisons(&self, num_comparisons: usize) {
        self.comparisons
            .fetch_add(num_comparisons, Ordering::Relaxed);
    }

    fn record_bytes_loaded(&self, num_bytes: u64) {
        self.bytes_loaded.fetch_add(num_bytes, Ordering::Relaxed);
    }

    fn record_partitions_probed(&self, count: usize) {
        self.partitions_probed.fetch_add(count, Ordering::Relaxed);
    }

    fn record_candidates_evaluated(&self, count: usize) {
        self.candidates_evaluated
            .fetch_add(count, Ordering::Relaxed);
    }

    fn record_search_duration_us(&self, duration: u64) {
        self.search_duration_us
            .fetch_add(duration, Ordering::Relaxed);
    }
}

/// A point-in-time snapshot of index metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MetricsSnapshot {
    pub parts_loaded: usize,
    pub index_loads: usize,
    pub comparisons: usize,
    pub bytes_loaded: u64,
    pub partitions_probed: usize,
    pub candidates_evaluated: usize,
    pub search_duration_us: u64,
}

impl LocalMetricsCollector {
    /// Take a consistent snapshot of the current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            parts_loaded: self.parts_loaded.load(Ordering::Relaxed),
            index_loads: self.index_loads.load(Ordering::Relaxed),
            comparisons: self.comparisons.load(Ordering::Relaxed),
            bytes_loaded: self.bytes_loaded.load(Ordering::Relaxed),
            partitions_probed: self.partitions_probed.load(Ordering::Relaxed),
            candidates_evaluated: self.candidates_evaluated.load(Ordering::Relaxed),
            search_duration_us: self.search_duration_us.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_metrics_new_fields() {
        let collector = LocalMetricsCollector::default();
        collector.record_partitions_probed(5);
        collector.record_candidates_evaluated(100);
        collector.record_search_duration_us(42);
        collector.record_bytes_loaded(8192);

        let snap = collector.snapshot();
        assert_eq!(snap.partitions_probed, 5);
        assert_eq!(snap.candidates_evaluated, 100);
        assert_eq!(snap.search_duration_us, 42);
        assert_eq!(snap.bytes_loaded, 8192);
    }

    #[test]
    fn test_dump_into_new_fields() {
        let local = LocalMetricsCollector::default();
        local.record_partitions_probed(3);
        local.record_candidates_evaluated(50);
        local.record_search_duration_us(99);
        local.record_parts_loaded(2);

        let target = LocalMetricsCollector::default();
        local.dump_into(&target);

        let snap = target.snapshot();
        assert_eq!(snap.partitions_probed, 3);
        assert_eq!(snap.candidates_evaluated, 50);
        assert_eq!(snap.search_duration_us, 99);
        assert_eq!(snap.parts_loaded, 2);
    }

    #[test]
    fn test_default_impls_noop() {
        let noop = NoOpMetricsCollector;
        // These should not panic — default no-op impls
        noop.record_bytes_loaded(4096);
        noop.record_partitions_probed(10);
        noop.record_candidates_evaluated(200);
        noop.record_search_duration_us(500);
    }
}
