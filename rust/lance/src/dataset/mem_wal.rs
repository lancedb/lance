// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL - Log-Structured Merge (LSM) tree for Lance tables
//!
//! This module implements an LSM tree architecture for high-performance
//! streaming writes with durability guarantees via Write-Ahead Log (WAL).
//!
//! ## Architecture
//!
//! Each shard has:
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
//! Each shard has exactly one active writer at any time, enforced via
//! monotonically increasing writer epochs in the shard manifest.

mod api;
mod hnsw;
mod index;
mod manifest;
pub mod memtable;
pub mod scanner;
pub mod util;
mod wal;
pub mod write;

pub use api::{DatasetMemWalExt, InitializeMemWalBuilder};

/// Column name for the WAL entry position stamped per row into flushed L0
/// generation files.
///
/// Together with [`WAL_POS_COLUMN`] this forms the composite recency key
/// `(_wal_seq, _wal_pos)` that L0→base compaction uses to select the per-PK
/// survivor by the authoritative WAL write order (the WAL appender packs
/// multiple batches into one entry, so the entry position alone is monotonic
/// but *not* per-write unique — [`WAL_POS_COLUMN`] discriminates same-entry
/// duplicates).
///
/// This is internal to the L0 file layout: it is never part of the user/base
/// schema, never reported by `is_system_column`, and never projected by the
/// read path, so it is invisible to readers.
pub const WAL_SEQ_COLUMN: &str = "_wal_seq";

/// Column name for the within-WAL-entry position. See [`WAL_SEQ_COLUMN`].
pub const WAL_POS_COLUMN: &str = "_wal_pos";
pub use manifest::ShardManifestStore;
pub use memtable::scanner::MemTableScanner;
pub use scanner::{LsmDataSource, LsmGeneration, LsmScanner, ShardSnapshot};
pub use wal::{WalAppendResult, WalAppender, WalReadEntry, WalTailer};
pub use write::ShardWriter;
pub use write::ShardWriterConfig;
