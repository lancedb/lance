// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicUsize, Ordering};

pub const AND_CANDIDATES_SEEN_METRIC: &str = "and_candidates_seen";
pub const AND_CANDIDATES_PRUNED_BEFORE_RETURN_METRIC: &str = "and_candidates_pruned_before_return";
pub const AND_FULL_SCORES_METRIC: &str = "and_full_scores";
pub const FREQS_COLLECTED_METRIC: &str = "freqs_collected";
/// Metric name for documents admitted by the FTS candidate generator.
pub const FTS_CANDIDATES_VISITED_METRIC: &str = "fts_candidates_visited";
/// Metric name for FTS candidates whose complete score was computed.
pub const FTS_CANDIDATES_SCORED_METRIC: &str = "fts_candidates_scored";
/// Metric name for compressed FTS posting blocks decoded by a query.
pub const FTS_POSTING_BLOCKS_DECODED_METRIC: &str = "fts_posting_blocks_decoded";
/// Metric name for candidate documents checked against phrase positions.
pub const FTS_PHRASE_POSITION_CHECKS_METRIC: &str = "fts_phrase_position_checks";

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

    /// Record index cache hits observed while serving this query.
    ///
    /// A "hit" is one page-level lookup (partition, posting list, BTree page, etc.)
    /// that was served from the in-memory index cache without touching storage.
    fn record_index_cache_hits(&self, _num_hits: usize) {}

    /// Convenience for a single cache hit.
    fn record_index_cache_hit(&self) {
        self.record_index_cache_hits(1);
    }

    /// Record index cache misses observed while serving this query.
    ///
    /// A "miss" is one page-level lookup that had to be loaded from storage
    /// because it was not present in the cache.
    fn record_index_cache_misses(&self, _num_misses: usize) {}

    /// Convenience for a single cache miss.
    fn record_index_cache_miss(&self) {
        self.record_index_cache_misses(1);
    }

    /// Record AND candidates returned from WAND alignment to the scoring loop.
    ///
    /// This excludes candidates pruned before `next()` returns. Use this with
    /// `record_and_candidates_pruned_before_return` to recover total aligned
    /// AND candidates.
    fn record_and_candidates_seen(&self, _num_candidates: usize) {}

    /// Record AND candidates pruned during WAND alignment before `next()` returns.
    fn record_and_candidates_pruned_before_return(&self, _num_candidates: usize) {}

    fn record_and_full_scores(&self, _num_scores: usize) {}

    fn record_freqs_collected(&self, _num_collections: usize) {}

    /// Record documents admitted by the FTS candidate generator.
    fn record_fts_candidates_visited(&self, _num_candidates: usize) {}

    /// Record FTS candidates whose complete score was computed.
    fn record_fts_candidates_scored(&self, _num_candidates: usize) {}

    /// Record compressed FTS posting blocks decoded while evaluating a query.
    fn record_fts_posting_blocks_decoded(&self, _num_blocks: usize) {}

    /// Record candidate documents checked against phrase positions.
    fn record_fts_phrase_position_checks(&self, _num_checks: usize) {}

    /// Returns an optional sink for recording exact I/O statistics (bytes read,
    /// IOPS, and requests) performed on behalf of this collector.
    ///
    /// Index implementations that read from a
    /// [`lance_io::scheduler::ScanScheduler`] can attach the returned handle to
    /// their file readers so the I/O performed for a single query is measured
    /// and attributed here.  The default returns `None`, meaning the caller does
    /// not want I/O measured (and index implementations should then take their
    /// normal, uninstrumented read path).
    fn io_stats(&self) -> Option<lance_io::scheduler::IoStats> {
        None
    }
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
    // Kept `pub(crate)` so that adding new metric fields to this public struct
    // does not break downstream callers that construct or destructure the
    // existing three fields. Callers can still read cumulative values via
    // [`Self::index_cache_hits`] / [`Self::index_cache_misses`].
    pub(crate) index_cache_hits: AtomicUsize,
    pub(crate) index_cache_misses: AtomicUsize,
    pub(crate) fts_candidates_visited: AtomicUsize,
    pub(crate) fts_candidates_scored: AtomicUsize,
    pub(crate) fts_posting_blocks_decoded: AtomicUsize,
    pub(crate) fts_phrase_position_checks: AtomicUsize,
}

impl LocalMetricsCollector {
    pub fn dump_into(self, other: &dyn MetricsCollector) {
        other.record_parts_loaded(self.parts_loaded.load(Ordering::Relaxed));
        other.record_index_loads(self.index_loads.load(Ordering::Relaxed));
        other.record_comparisons(self.comparisons.load(Ordering::Relaxed));
        other.record_index_cache_hits(self.index_cache_hits.load(Ordering::Relaxed));
        other.record_index_cache_misses(self.index_cache_misses.load(Ordering::Relaxed));
        other.record_fts_candidates_visited(self.fts_candidates_visited.load(Ordering::Relaxed));
        other.record_fts_candidates_scored(self.fts_candidates_scored.load(Ordering::Relaxed));
        other.record_fts_posting_blocks_decoded(
            self.fts_posting_blocks_decoded.load(Ordering::Relaxed),
        );
        other.record_fts_phrase_position_checks(
            self.fts_phrase_position_checks.load(Ordering::Relaxed),
        );
    }

    /// Cumulative index cache hits recorded so far.
    pub fn index_cache_hits(&self) -> usize {
        self.index_cache_hits.load(Ordering::Relaxed)
    }

    /// Cumulative index cache misses recorded so far.
    pub fn index_cache_misses(&self) -> usize {
        self.index_cache_misses.load(Ordering::Relaxed)
    }

    /// Cumulative FTS candidates admitted by candidate generation.
    pub fn fts_candidates_visited(&self) -> usize {
        self.fts_candidates_visited.load(Ordering::Relaxed)
    }

    /// Cumulative FTS candidates whose complete score was computed.
    pub fn fts_candidates_scored(&self) -> usize {
        self.fts_candidates_scored.load(Ordering::Relaxed)
    }

    /// Cumulative compressed FTS posting blocks decoded.
    pub fn fts_posting_blocks_decoded(&self) -> usize {
        self.fts_posting_blocks_decoded.load(Ordering::Relaxed)
    }

    /// Cumulative candidate documents checked against phrase positions.
    pub fn fts_phrase_position_checks(&self) -> usize {
        self.fts_phrase_position_checks.load(Ordering::Relaxed)
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

    fn record_index_cache_hits(&self, num_hits: usize) {
        self.index_cache_hits.fetch_add(num_hits, Ordering::Relaxed);
    }

    fn record_index_cache_misses(&self, num_misses: usize) {
        self.index_cache_misses
            .fetch_add(num_misses, Ordering::Relaxed);
    }

    fn record_fts_candidates_visited(&self, num_candidates: usize) {
        self.fts_candidates_visited
            .fetch_add(num_candidates, Ordering::Relaxed);
    }

    fn record_fts_candidates_scored(&self, num_candidates: usize) {
        self.fts_candidates_scored
            .fetch_add(num_candidates, Ordering::Relaxed);
    }

    fn record_fts_posting_blocks_decoded(&self, num_blocks: usize) {
        self.fts_posting_blocks_decoded
            .fetch_add(num_blocks, Ordering::Relaxed);
    }

    fn record_fts_phrase_position_checks(&self, num_checks: usize) {
        self.fts_phrase_position_checks
            .fetch_add(num_checks, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumSink {
        parts: AtomicUsize,
        loads: AtomicUsize,
        comparisons: AtomicUsize,
        hits: AtomicUsize,
        misses: AtomicUsize,
        candidates_visited: AtomicUsize,
        candidates_scored: AtomicUsize,
        posting_blocks_decoded: AtomicUsize,
        phrase_position_checks: AtomicUsize,
    }

    impl MetricsCollector for SumSink {
        fn record_parts_loaded(&self, n: usize) {
            self.parts.fetch_add(n, Ordering::Relaxed);
        }
        fn record_index_loads(&self, n: usize) {
            self.loads.fetch_add(n, Ordering::Relaxed);
        }
        fn record_comparisons(&self, n: usize) {
            self.comparisons.fetch_add(n, Ordering::Relaxed);
        }
        fn record_index_cache_hits(&self, n: usize) {
            self.hits.fetch_add(n, Ordering::Relaxed);
        }
        fn record_index_cache_misses(&self, n: usize) {
            self.misses.fetch_add(n, Ordering::Relaxed);
        }
        fn record_fts_candidates_visited(&self, n: usize) {
            self.candidates_visited.fetch_add(n, Ordering::Relaxed);
        }
        fn record_fts_candidates_scored(&self, n: usize) {
            self.candidates_scored.fetch_add(n, Ordering::Relaxed);
        }
        fn record_fts_posting_blocks_decoded(&self, n: usize) {
            self.posting_blocks_decoded.fetch_add(n, Ordering::Relaxed);
        }
        fn record_fts_phrase_position_checks(&self, n: usize) {
            self.phrase_position_checks.fetch_add(n, Ordering::Relaxed);
        }
    }

    #[test]
    fn local_metrics_collector_forwards_counts() {
        let local = LocalMetricsCollector::default();
        local.record_index_cache_hit();
        local.record_index_cache_hit();
        local.record_index_cache_misses(3);
        local.record_part_load();
        local.record_index_load();
        local.record_comparisons(5);
        local.record_fts_candidates_visited(7);
        local.record_fts_candidates_scored(6);
        local.record_fts_posting_blocks_decoded(4);
        local.record_fts_phrase_position_checks(3);

        let sink = SumSink {
            parts: AtomicUsize::new(0),
            loads: AtomicUsize::new(0),
            comparisons: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            candidates_visited: AtomicUsize::new(0),
            candidates_scored: AtomicUsize::new(0),
            posting_blocks_decoded: AtomicUsize::new(0),
            phrase_position_checks: AtomicUsize::new(0),
        };
        local.dump_into(&sink);

        assert_eq!(sink.parts.load(Ordering::Relaxed), 1);
        assert_eq!(sink.loads.load(Ordering::Relaxed), 1);
        assert_eq!(sink.comparisons.load(Ordering::Relaxed), 5);
        assert_eq!(sink.hits.load(Ordering::Relaxed), 2);
        assert_eq!(sink.misses.load(Ordering::Relaxed), 3);
        assert_eq!(sink.candidates_visited.load(Ordering::Relaxed), 7);
        assert_eq!(sink.candidates_scored.load(Ordering::Relaxed), 6);
        assert_eq!(sink.posting_blocks_decoded.load(Ordering::Relaxed), 4);
        assert_eq!(sink.phrase_position_checks.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn no_op_metrics_collector_ignores_counts() {
        // Ensures existing implementors that do not override count methods
        // remain sound (default impl is a no-op).
        let collector = NoOpMetricsCollector;
        collector.record_index_cache_hit();
        collector.record_index_cache_miss();
        collector.record_index_cache_hits(10);
        collector.record_index_cache_misses(20);
        collector.record_fts_candidates_visited(10);
        collector.record_fts_candidates_scored(10);
        collector.record_fts_posting_blocks_decoded(10);
        collector.record_fts_phrase_position_checks(10);
    }
}
