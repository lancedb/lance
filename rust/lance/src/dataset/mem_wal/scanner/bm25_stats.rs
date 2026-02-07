// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Global BM25 statistics for cross-generation FTS scoring.
//!
//! When searching across multiple LSM generations, each has its own corpus
//! statistics. This module provides aggregated statistics so BM25 scores
//! are comparable across generations.

use std::collections::HashMap;

use futures::future::BoxFuture;
use lance_index::scalar::inverted::scorer::{idf, B, K1};

/// A deferred handle to global BM25 stats that are being computed asynchronously.
///
/// Stats collection is spawned as a background task at plan time. Execution
/// nodes `.await` the shared future when they need the stats (which may already
/// be resolved by then). All consumers share the same computation — the
/// underlying future runs at most once.
pub type DeferredBM25Stats = futures::future::Shared<BoxFuture<'static, GlobalBM25Stats>>;

/// Per-generation BM25 statistics, collected before aggregation.
#[derive(Debug, Clone)]
pub struct GenerationBM25Stats {
    pub num_docs: usize,
    pub total_tokens: u64,
    /// term -> number of documents containing the term in this generation.
    pub term_doc_freqs: HashMap<String, usize>,
}

impl GenerationBM25Stats {
    pub fn new(num_docs: usize, total_tokens: u64, term_doc_freqs: HashMap<String, usize>) -> Self {
        Self {
            num_docs,
            total_tokens,
            term_doc_freqs,
        }
    }
}

/// Aggregated BM25 statistics across all LSM generations.
///
/// Used to produce comparable BM25 scores when searching across
/// memtables and base table simultaneously.
#[derive(Debug, Clone)]
pub struct GlobalBM25Stats {
    pub num_docs: usize,
    pub total_tokens: u64,
    pub avg_doc_length: f32,
    /// term -> total number of documents containing the term across all generations.
    pub term_doc_freqs: HashMap<String, usize>,
}

impl GlobalBM25Stats {
    pub fn new(num_docs: usize, total_tokens: u64, term_doc_freqs: HashMap<String, usize>) -> Self {
        let avg_doc_length = if num_docs > 0 {
            total_tokens as f32 / num_docs as f32
        } else {
            1.0
        };
        Self {
            num_docs,
            total_tokens,
            avg_doc_length,
            term_doc_freqs,
        }
    }

    /// Aggregate multiple per-generation stats into global stats.
    pub fn aggregate(stats: &[GenerationBM25Stats]) -> Self {
        let num_docs: usize = stats.iter().map(|s| s.num_docs).sum();
        let total_tokens: u64 = stats.iter().map(|s| s.total_tokens).sum();
        let mut term_doc_freqs: HashMap<String, usize> = HashMap::new();
        for s in stats {
            for (term, freq) in &s.term_doc_freqs {
                *term_doc_freqs.entry(term.clone()).or_default() += freq;
            }
        }
        Self::new(num_docs, total_tokens, term_doc_freqs)
    }

    /// Compute IDF for a token using global stats.
    pub fn idf(&self, token: &str) -> f32 {
        let token_docs = self.term_doc_freqs.get(token).copied().unwrap_or(0);
        if token_docs == 0 {
            return 0.0;
        }
        idf(token_docs, self.num_docs)
    }

    /// Compute BM25 score for a single term occurrence.
    pub fn score(&self, token: &str, freq: u32, doc_length: u32) -> f32 {
        let idf_val = self.idf(token);
        if idf_val == 0.0 {
            return 0.0;
        }
        let freq = freq as f32;
        let doc_length = doc_length as f32;
        let doc_norm = K1 * (1.0 - B + B * doc_length / self.avg_doc_length);
        idf_val * (K1 + 1.0) * freq / (freq + doc_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_bm25_stats_new() {
        let mut term_freqs = HashMap::new();
        term_freqs.insert("hello".to_string(), 5);
        term_freqs.insert("world".to_string(), 3);

        let stats = GlobalBM25Stats::new(10, 100, term_freqs);
        assert_eq!(stats.num_docs, 10);
        assert_eq!(stats.total_tokens, 100);
        assert!((stats.avg_doc_length - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_global_bm25_stats_empty() {
        let stats = GlobalBM25Stats::new(0, 0, HashMap::new());
        assert_eq!(stats.num_docs, 0);
        assert!((stats.avg_doc_length - 1.0).abs() < f32::EPSILON);
        assert_eq!(stats.idf("unknown"), 0.0);
        assert_eq!(stats.score("unknown", 1, 10), 0.0);
    }

    #[test]
    fn test_aggregate_stats() {
        let gen1 = GenerationBM25Stats::new(
            5,
            50,
            HashMap::from([("hello".to_string(), 3), ("world".to_string(), 2)]),
        );
        let gen2 = GenerationBM25Stats::new(
            10,
            120,
            HashMap::from([("hello".to_string(), 4), ("rust".to_string(), 6)]),
        );

        let global = GlobalBM25Stats::aggregate(&[gen1, gen2]);
        assert_eq!(global.num_docs, 15);
        assert_eq!(global.total_tokens, 170);
        assert_eq!(global.term_doc_freqs["hello"], 7);
        assert_eq!(global.term_doc_freqs["world"], 2);
        assert_eq!(global.term_doc_freqs["rust"], 6);
    }

    #[test]
    fn test_idf_known_token() {
        let stats = GlobalBM25Stats::new(10, 100, HashMap::from([("hello".to_string(), 3)]));
        let idf_val = stats.idf("hello");
        assert!(idf_val > 0.0);
        // IDF = ln((10 - 3 + 0.5) / (3 + 0.5) + 1) = ln(7.5/3.5 + 1)
        let expected = ((10.0 - 3.0 + 0.5) / (3.0 + 0.5) + 1.0_f32).ln();
        assert!((idf_val - expected).abs() < 1e-6);
    }

    #[test]
    fn test_score_calculation() {
        let stats = GlobalBM25Stats::new(10, 100, HashMap::from([("hello".to_string(), 3)]));
        let score = stats.score("hello", 2, 10);
        assert!(score > 0.0);

        // Verify same result as manual BM25 calculation
        let idf_val = stats.idf("hello");
        let freq = 2.0_f32;
        let doc_norm = K1 * (1.0 - B + B * 10.0 / 10.0);
        let expected = idf_val * (K1 + 1.0) * freq / (freq + doc_norm);
        assert!((score - expected).abs() < 1e-6);
    }
}
