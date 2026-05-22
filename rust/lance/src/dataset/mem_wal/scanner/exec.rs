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
//! - [`WithinSourceDedupExec`]: Deduplicates rows with the same PK from a single source (used by point lookup)
//! - [`PkHashFilterExec`]: Drops rows whose PK hash was superseded by a newer generation (vector-search block-list)

mod bloom_guard;
mod coalesce_first;
mod deduplicate;
mod generation_tag;
mod global_pk_dedup;
mod pk;
mod pk_hash_filter;
mod source_tag;
mod within_source_dedup;

pub use bloom_guard::{BloomFilterGuardExec, compute_pk_hash_from_scalars};
pub use coalesce_first::CoalesceFirstExec;
pub use deduplicate::{DeduplicateExec, ROW_ADDRESS_COLUMN};
pub use generation_tag::{MEMTABLE_GEN_COLUMN, MemtableGenTagExec};
pub use global_pk_dedup::LsmGlobalPkDedupExec;
pub use pk::{compute_pk_hash, resolve_pk_indices};
pub use pk_hash_filter::PkHashFilterExec;
pub use source_tag::{FRESHNESS_COLUMN, FreshnessPolarity, LsmSourceTagExec};
pub use within_source_dedup::{DedupDirection, WithinSourceDedupExec};
