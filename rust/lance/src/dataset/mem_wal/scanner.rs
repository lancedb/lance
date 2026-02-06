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
//! let scanner = LsmScanner::new(base_table, region_snapshots, vec!["pk".to_string()])
//!     .project(&["id", "name"])
//!     .filter("id > 10")?
//!     .limit(100, None);
//!
//! let stream = scanner.try_into_stream().await?;
//! ```

mod builder;
mod collector;
mod data_source;
pub mod exec;
mod planner;
mod point_lookup;
mod vector_search;

pub use builder::LsmScanner;
pub use collector::{ActiveMemTableRef, LsmDataSourceCollector};
pub use data_source::{FlushedGeneration, LsmDataSource, LsmGeneration, RegionSnapshot};
pub use point_lookup::LsmPointLookupPlanner;
pub use vector_search::{LsmVectorSearchPlanner, DISTANCE_COLUMN};
