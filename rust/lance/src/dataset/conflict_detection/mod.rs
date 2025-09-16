// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conflict detection mechanisms for concurrent operations
//!
//! This module provides Bloom Filter-based conflict detection for merge insert operations.
//! It implements a two-tier storage strategy:
//! - For small datasets (<200KB): exact primary key mapping
//! - For large datasets (>=200KB): probabilistic Bloom Filter
//!
//! The conflict detection works by:
//! 1. Collecting primary keys during merge insert operations
//! 2. Building either exact mappings or Bloom Filters based on data size
//! 3. Storing the conflict detection data in transaction files
//! 4. Performing intersection-based conflict detection during commit

pub mod conflict_detector;
pub mod primary_key_filter;

pub use conflict_detector::{ConflictDetectionResult, ConflictDetector};
pub use primary_key_filter::{
    FilterType, PrimaryKeyBloomFilter, PrimaryKeyFilterModel, PrimaryKeyValue,
};

/// Threshold for switching between exact mapping and Bloom Filter (200KB)
pub const BLOOM_FILTER_THRESHOLD: usize = 200 * 1024;

/// Default false positive probability for Bloom Filters
pub const DEFAULT_FALSE_POSITIVE_PROBABILITY: f64 = 0.001; // 0.1%

/// Default expected number of items for Bloom Filter sizing
pub const DEFAULT_EXPECTED_ITEMS: u64 = 10000;
