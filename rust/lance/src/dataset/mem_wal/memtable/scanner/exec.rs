// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion ExecutionPlan implementations for MemWAL read path.
//!
//! This module contains execution nodes for:
//! - `MemTableScanExec` - Full table scan with MVCC visibility
//! - `BTreeIndexExec` - BTree index queries
//! - `VectorIndexExec` - IVF-PQ vector search
//! - `FtsIndexExec` - Full-text search

mod btree;
mod fts;
mod scan;
mod vector;

pub use btree::BTreeIndexExec;
pub use fts::FtsIndexExec;
pub use scan::MemTableScanExec;
pub use vector::VectorIndexExec;
