// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! LSM Scanner - Unified scanner for LSM tree data
//!
//! This module provides scanners that read from multiple data sources
//! in an LSM tree architecture:
//! - Base table (merged data)
//! - Flushed MemTables (persisted but not yet merged)
//! - Active MemTable (in-memory buffer)
//!
//! The scanner handles deduplication by primary key, keeping the newest
//! version based on generation number and row address.
//!
//! ## Supported Query Types
//!
//! - **Scan**: Full table scan with deduplication
//! - **Point Lookup**: Primary key-based lookup with bloom filter optimization
//! - **Vector Search**: KNN search with staleness detection
//!
//! ## Example
//!
//! ```ignore
//! use lance::dataset::mem_wal::scanner::LsmScanner;
//!
//! let scanner = LsmScanner::new(base_table, shard_snapshots, vec!["pk".to_string()])
//!     .project(&["id", "name"])
//!     .filter("id > 10")?
//!     .limit(100, None);
//!
//! let stream = scanner.try_into_stream().await?;
//! ```

mod block_list;
mod builder;
mod collector;
mod data_source;
pub mod exec;
pub(crate) mod flushed_cache;
mod fts_search;
mod planner;
mod point_lookup;
mod projection;
mod vector_search;

pub use block_list::write_pk_sidecar;
pub use builder::LsmScanner;
pub use collector::{
    ActiveMemTableRef, InMemoryMemTableRef, InMemoryMemTables, LsmDataSourceCollector,
};
pub use data_source::{
    FlushedGeneration, FreshTierWatermark, LsmDataSource, LsmGeneration, ShardSnapshot,
};
pub use flushed_cache::FlushedMemTableCache;
pub use fts_search::{LsmFtsSearchPlanner, SCORE_COLUMN};
pub use point_lookup::LsmPointLookupPlanner;
pub use projection::DISTANCE_COLUMN;
pub use vector_search::LsmVectorSearchPlanner;
