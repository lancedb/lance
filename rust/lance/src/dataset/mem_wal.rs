// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL - Log-Structured Merge (LSM) tree for Lance tables
//!
//! This module implements an LSM tree architecture for high-performance
//! streaming writes with durability guarantees via Write-Ahead Log (WAL).
//!
//! ## Architecture
//!
//! Each region has:
//! - A **MemTable** for in-memory data (immediately queryable)
//! - A **WAL Buffer** for durability (persisted to object storage)
//! - **In-memory indexes** (BTree, IVF-PQ, FTS) for indexed queries
//!
//! ## Write Path
//!
//! ```text
//! put(batch) → MemTable.insert() → WalBuffer.append() → [async flush to storage]
//!                   ↓
//!           IndexRegistry.update()
//! ```
//!
//! ## Durability
//!
//! Writers can be configured for:
//! - **Durable writes**: Wait for WAL flush before returning
//! - **Non-durable writes**: Buffer in memory, accept potential loss on crash
//!
//! ## Epoch-Based Fencing
//!
//! Each region has exactly one active writer at any time, enforced via
//! monotonically increasing writer epochs in the region manifest.

mod api;
mod index;
mod manifest;
pub mod memtable;
mod util;
mod wal;
pub mod write;

pub use api::{DatasetMemWalExt, MemWalConfig};
pub use manifest::RegionManifestStore;
pub use memtable::scanner::MemTableScanner;
pub use write::RegionWriter;
pub use write::RegionWriterConfig;
