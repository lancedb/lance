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

pub use brute_force_vector::MemTableBruteForceVectorExec;
pub use btree::BTreeIndexExec;
pub use dedup_scan::MemTableDedupScanExec;
pub use fts::FtsIndexExec;
pub use scan::{MemTableScanExec, ROW_ADDRESS_COLUMN};
pub use vector::VectorIndexExec;
