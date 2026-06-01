// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion ExecutionPlan implementations for MemWAL read path.
//!
//! This module contains execution nodes for:
//! - `MemTableScanExec` - Full table scan with MVCC visibility
//! - `BTreeIndexExec` - BTree index queries
//! - `VectorIndexExec` - HNSW vector search
//! - `MemTableBruteForceVectorExec` - KNN over the active memtable without an HNSW
//! - `FtsIndexExec` - Full-text search

mod brute_force_vector;
mod btree;
mod dedup_scan;
mod fts;
mod scan;
mod vector;

/// Internal active-memtable row position column.
///
/// This is an implementation detail used for MemWAL ordering / dedup. It is
/// not a Lance system column and must not be exposed in canonical LSM output.
pub const MEMWAL_ROW_POSITION_COLUMN: &str = "__memwal_row_position";

pub use brute_force_vector::MemTableBruteForceVectorExec;
pub use btree::BTreeIndexExec;
pub use dedup_scan::MemTableDedupScanExec;
pub use fts::FtsIndexExec;
pub use scan::{MemTableScanExec, ROW_ADDRESS_COLUMN};
pub use vector::VectorIndexExec;
