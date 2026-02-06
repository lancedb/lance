// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution plan nodes for LSM scanner.
//!
//! This module contains custom DataFusion execution plan implementations
//! for LSM tree query execution:
//!
//! - [`MemtableGenTagExec`]: Wraps a scan to add `_memtable_gen` column
//! - [`DeduplicateExec`]: Deduplicates by primary key, keeping newest version
//! - [`BloomFilterGuardExec`]: Guards child execution with bloom filter check
//! - [`CoalesceFirstExec`]: Returns first non-empty result with short-circuit
//! - [`FilterStaleExec`]: Filters out rows with newer versions in higher generations

mod bloom_guard;
mod coalesce_first;
mod deduplicate;
mod filter_stale;
mod generation_tag;

pub use bloom_guard::{compute_pk_hash_from_scalars, BloomFilterGuardExec};
pub use coalesce_first::CoalesceFirstExec;
pub use deduplicate::{DeduplicateExec, ROW_ADDRESS_COLUMN};
pub use filter_stale::{FilterStaleExec, GenerationBloomFilter};
pub use generation_tag::{MemtableGenTagExec, MEMTABLE_GEN_COLUMN};
