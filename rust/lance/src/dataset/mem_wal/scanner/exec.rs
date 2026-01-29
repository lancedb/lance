// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution plan nodes for LSM scanner.
//!
//! This module contains custom DataFusion execution plan implementations
//! for LSM tree query execution:
//!
//! - [`MemtableGenTagExec`]: Wraps a scan to add `_memtable_gen` column
//! - [`DeduplicateExec`]: Deduplicates by primary key, keeping newest version

mod deduplicate;
mod generation_tag;

pub use deduplicate::{DeduplicateExec, ROW_ADDRESS_COLUMN};
pub use generation_tag::{MemtableGenTagExec, MEMTABLE_GEN_COLUMN};
