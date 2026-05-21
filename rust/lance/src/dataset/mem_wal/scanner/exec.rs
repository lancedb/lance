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
//! - [`LsmSourceTagExec`]: Tags rows with `_memtable_gen` + `_freshness` for the vector-search global dedup
//! - [`LsmGlobalPkDedupExec`]: Single-pass cross-source PK dedup over the merged vector-search stream

mod bloom_guard;
mod coalesce_first;
mod deduplicate;
mod generation_tag;
mod global_pk_dedup;
mod source_tag;

pub use bloom_guard::{BloomFilterGuardExec, compute_pk_hash_from_scalars};
pub use coalesce_first::CoalesceFirstExec;
pub use deduplicate::{DeduplicateExec, ROW_ADDRESS_COLUMN};
pub use generation_tag::{MEMTABLE_GEN_COLUMN, MemtableGenTagExec};
pub use global_pk_dedup::LsmGlobalPkDedupExec;
pub use source_tag::{FRESHNESS_COLUMN, FreshnessPolarity, LsmSourceTagExec};
