// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index store for MemTable write path.
//!
//! Maintains in-memory indexes that are updated synchronously with writes:
//! - BTree: Primary key and scalar field lookups
//! - HNSW: Vector similarity search (built incrementally, queryable while
//!   building, flushed as Lance HNSW + FLAT)
//! - FTS: Full-text search
//!
//! Other index types log a warning and are skipped.

#![allow(clippy::print_stderr)]
#![allow(clippy::type_complexity)]

mod btree;
mod fts;
mod hnsw;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::memtable::batch_store::StoredBatch;
use arrow_array::RecordBatch;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_index::pbold;
use lance_index::scalar::InvertedIndexParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use prost::Message as _;
use tracing::instrument;

/// Row position in MemTable.
///
/// This is the absolute row position across all batches in the MemTable.
/// When flushed to a single Lance file, this becomes the row ID directly.
pub type RowPosition = u64;

// Re-export public types used externally
pub use btree::{BTreeIndexConfig, BTreeMemIndex};
pub use fts::{FtsIndexConfig, FtsMemIndex, FtsQueryExpr, SearchOptions};
pub use hnsw::{HnswIndexConfig, HnswMemIndex};

// ============================================================================
// Index Store
// ============================================================================

/// Configuration for an index in MemWAL.
///
/// Each variant contains all the configuration needed for that index type.
/// `Hnsw` is boxed because `HnswBuildParams` is small but the variant may
/// grow with future config (e.g. shard-specific tuning).
#[derive(Debug, Clone)]
pub enum MemIndexConfig {
    /// BTree index for scalar fields (point lookups, range queries).
    BTree(BTreeIndexConfig),
    /// HNSW vector index built incrementally, queryable while building.
    Hnsw(Box<HnswIndexConfig>),
    /// Full-text search index.
    Fts(FtsIndexConfig),
}

impl MemIndexConfig {
    /// Get the index name.
    pub fn name(&self) -> &str {
        match self {
            Self::BTree(c) => &c.name,
            Self::Hnsw(c) => &c.name,
            Self::Fts(c) => &c.name,
        }
    }

    /// Get the field ID.
    pub fn field_id(&self) -> i32 {
        match self {
            Self::BTree(c) => c.field_id,
            Self::Hnsw(c) => c.field_id,
            Self::Fts(c) => c.field_id,
        }
    }

    /// Get the column name.
    pub fn column(&self) -> &str {
        match self {
            Self::BTree(c) => &c.column,
            Self::Hnsw(c) => &c.column,
            Self::Fts(c) => &c.column,
        }
    }

    /// Create a BTree index config from base table IndexMetadata.
    pub fn btree_from_metadata(index_meta: &IndexMetadata, schema: &LanceSchema) -> Result<Self> {
        let (field_id, column) = Self::extract_field_info(index_meta, schema)?;
        Ok(Self::BTree(BTreeIndexConfig {
            name: index_meta.name.clone(),
            field_id,
            column,
        }))
    }

    /// Create an FTS index config from base table IndexMetadata.
    pub fn fts_from_metadata(index_meta: &IndexMetadata, schema: &LanceSchema) -> Result<Self> {
        let (field_id, column) = Self::extract_field_info(index_meta, schema)?;

        // Extract InvertedIndexParams from index_details if available
        let params = if let Some(details_any) = &index_meta.index_details {
            if let Ok(details) = pbold::InvertedIndexDetails::decode(details_any.value.as_slice()) {
                InvertedIndexParams::try_from(&details)?
            } else {
                InvertedIndexParams::default()
            }
        } else {
            InvertedIndexParams::default()
        };

        Ok(Self::Fts(FtsIndexConfig::with_params(
            index_meta.name.clone(),
            field_id,
            column,
            params,
        )))
    }

    /// Create an HNSW vector index config.
    pub fn hnsw(name: String, field_id: i32, column: String, distance_type: DistanceType) -> Self {
        Self::Hnsw(Box::new(HnswIndexConfig::new(
            name,
            field_id,
            column,
            distance_type,
        )))
    }

    /// Create an HNSW vector index config with explicit build parameters.
    pub fn hnsw_with_params(
        name: String,
        field_id: i32,
        column: String,
        distance_type: DistanceType,
        build_params: HnswBuildParams,
    ) -> Self {
        Self::Hnsw(Box::new(
            HnswIndexConfig::new(name, field_id, column, distance_type)
                .with_build_params(build_params),
        ))
    }

    /// Detect index type from protobuf type_url.
    pub fn detect_index_type(type_url: &str) -> Result<&'static str> {
        if type_url.ends_with("BTreeIndexDetails") {
            Ok("btree")
        } else if type_url.ends_with("InvertedIndexDetails") {
            Ok("fts")
        } else if type_url.ends_with("VectorIndexDetails") {
            Ok("vector")
        } else {
            Err(Error::invalid_input(format!(
                "Unsupported index type for MemWAL: {}. Supported: BTree, Inverted, Vector",
                type_url
            )))
        }
    }

    /// Extract field ID and column name from index metadata.
    fn extract_field_info(
        index_meta: &IndexMetadata,
        schema: &LanceSchema,
    ) -> Result<(i32, String)> {
        let field_id = index_meta.fields.first().ok_or_else(|| {
            Error::invalid_input(format!("Index '{}' has no fields", index_meta.name))
        })?;

        let column = schema
            .field_by_id(*field_id)
            .map(|f| f.name.clone())
            .ok_or_else(|| {
                Error::invalid_input(format!("Field with id {} not found in schema", field_id))
            })?;

        Ok((*field_id, column))
    }
}

/// Registry managing all in-memory indexes for a MemTable.
///
/// Indexes are keyed by index name. Each index stores its field_id for
/// stable column-to-index resolution (column name → field_id → index).
///
/// The store maintains a global `max_indexed_batch_position` watermark that
/// tracks which batches have been indexed. All indexes are updated atomically,
/// so queries should only see data up to this watermark for consistent results.
pub struct IndexStore {
    /// BTree indexes keyed by index name.
    btree_indexes: HashMap<String, BTreeMemIndex>,
    /// HNSW vector indexes keyed by index name.
    hnsw_indexes: HashMap<String, HnswMemIndex>,
    /// FTS indexes keyed by index name.
    fts_indexes: HashMap<String, FtsMemIndex>,
    /// Maximum batch position that has been indexed across all indexes.
    /// Updated atomically after all indexes have processed a batch.
    max_indexed_batch_position: AtomicUsize,
}

impl Default for IndexStore {
    fn default() -> Self {
        Self {
            btree_indexes: HashMap::new(),
            hnsw_indexes: HashMap::new(),
            fts_indexes: HashMap::new(),
            max_indexed_batch_position: AtomicUsize::new(0),
        }
    }
}

impl std::fmt::Debug for IndexStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexStore")
            .field(
                "btree_indexes",
                &self.btree_indexes.keys().collect::<Vec<_>>(),
            )
            .field(
                "hnsw_indexes",
                &self.hnsw_indexes.keys().collect::<Vec<_>>(),
            )
            .field("fts_indexes", &self.fts_indexes.keys().collect::<Vec<_>>())
            .field(
                "max_indexed_batch_position",
                &self.max_indexed_batch_position.load(Ordering::Acquire),
            )
            .finish()
    }
}

impl IndexStore {
    /// Create a new empty index registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an index registry from index configurations.
    ///
    /// # Arguments
    ///
    /// * `configs` - Index configurations
    /// * `max_rows` - Maximum vectors / rows in memtable. Used to size the
    ///   pre-allocated HNSW graph and storage capacity.
    /// * `max_batches` - Maximum number of write batches the HNSW storage
    ///   can hold by reference (matches the writer's
    ///   `ShardWriterConfig::max_memtable_batches`).
    pub fn from_configs(
        configs: &[MemIndexConfig],
        max_rows: usize,
        max_batches: usize,
    ) -> Result<Self> {
        let mut registry = Self::new();

        for config in configs {
            match config {
                MemIndexConfig::BTree(c) => {
                    let index = BTreeMemIndex::new(c.field_id, c.column.clone());
                    registry.btree_indexes.insert(c.name.clone(), index);
                }
                MemIndexConfig::Hnsw(c) => {
                    let index = HnswMemIndex::with_capacity(
                        c.field_id,
                        c.column.clone(),
                        c.distance_type,
                        c.build_params.clone(),
                        max_rows,
                        max_batches,
                    );
                    registry.hnsw_indexes.insert(c.name.clone(), index);
                }
                MemIndexConfig::Fts(c) => {
                    let index =
                        FtsMemIndex::with_params(c.field_id, c.column.clone(), c.params.clone());
                    registry.fts_indexes.insert(c.name.clone(), index);
                }
            }
        }

        Ok(registry)
    }

    /// Add a BTree/scalar index (implemented using skip-list for better concurrency).
    pub fn add_btree(&mut self, name: String, field_id: i32, column: String) {
        self.btree_indexes
            .insert(name, BTreeMemIndex::new(field_id, column));
    }

    /// Add an HNSW vector index with default build parameters.
    pub fn add_hnsw(
        &mut self,
        name: String,
        field_id: i32,
        column: String,
        distance_type: DistanceType,
        capacity: usize,
        max_batches: usize,
    ) {
        self.hnsw_indexes.insert(
            name,
            HnswMemIndex::with_capacity(
                field_id,
                column,
                distance_type,
                HnswBuildParams::default(),
                capacity,
                max_batches,
            ),
        );
    }

    /// Add an HNSW vector index with explicit build parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn add_hnsw_with_params(
        &mut self,
        name: String,
        field_id: i32,
        column: String,
        distance_type: DistanceType,
        build_params: HnswBuildParams,
        capacity: usize,
        max_batches: usize,
    ) {
        self.hnsw_indexes.insert(
            name,
            HnswMemIndex::with_capacity(
                field_id,
                column,
                distance_type,
                build_params,
                capacity,
                max_batches,
            ),
        );
    }

    /// Add an FTS index with default tokenizer parameters.
    pub fn add_fts(&mut self, name: String, field_id: i32, column: String) {
        self.fts_indexes
            .insert(name, FtsMemIndex::new(field_id, column));
    }

    /// Add an FTS index with custom tokenizer parameters.
    pub fn add_fts_with_params(
        &mut self,
        name: String,
        field_id: i32,
        column: String,
        params: InvertedIndexParams,
    ) {
        self.fts_indexes
            .insert(name, FtsMemIndex::with_params(field_id, column, params));
    }

    /// Insert a batch into all indexes.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        self.insert_with_batch_position(batch, row_offset, None)
    }

    /// Insert a batch into all indexes with batch position tracking.
    #[instrument(name = "idx_insert_batch", level = "debug", skip_all, fields(num_rows = batch.num_rows(), row_offset, batch_position))]
    pub fn insert_with_batch_position(
        &self,
        batch: &RecordBatch,
        row_offset: u64,
        batch_position: Option<usize>,
    ) -> Result<()> {
        for index in self.btree_indexes.values() {
            index.insert(batch, row_offset)?;
        }
        for index in self.hnsw_indexes.values() {
            index.insert(batch, row_offset)?;
        }
        for index in self.fts_indexes.values() {
            index.insert(batch, row_offset)?;
        }

        // Update global watermark after all indexes have been updated
        if let Some(bp) = batch_position {
            self.update_max_indexed_batch_position(bp);
        }

        Ok(())
    }

    /// Update the maximum indexed batch position.
    ///
    /// Only updates if the new value is greater than the current value.
    fn update_max_indexed_batch_position(&self, batch_pos: usize) {
        let mut current = self.max_indexed_batch_position.load(Ordering::Acquire);
        while batch_pos > current {
            match self.max_indexed_batch_position.compare_exchange_weak(
                current,
                batch_pos,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    /// Insert multiple batches into all indexes with cross-batch optimization.
    #[instrument(name = "idx_insert_batches", level = "debug", skip_all, fields(batch_count = batches.len()))]
    pub fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }

        // BTree indexes: iterate batches (no cross-batch optimization benefit)
        for index in self.btree_indexes.values() {
            for stored in batches {
                index.insert(&stored.data, stored.row_offset)?;
            }
        }

        // HNSW indexes: use batched insert
        for index in self.hnsw_indexes.values() {
            index.insert_batches(batches)?;
        }

        // FTS indexes: iterate batches (potential future optimization)
        for index in self.fts_indexes.values() {
            for stored in batches {
                index.insert(&stored.data, stored.row_offset)?;
            }
        }

        // Update global watermark to the max batch position
        let max_bp = batches.iter().map(|b| b.batch_position).max().unwrap();
        self.update_max_indexed_batch_position(max_bp);

        Ok(())
    }

    /// Insert multiple batches into all indexes in parallel.
    ///
    /// Each individual index runs in its own thread, regardless of type.
    /// This maximizes parallelism when multiple indexes are maintained.
    ///
    /// This is used during WAL flush to parallelize index updates with WAL I/O.
    /// Insert batches into all indexes in parallel.
    ///
    /// Returns a map of index names to their update durations for performance tracking.
    #[allow(clippy::print_stderr)]
    #[instrument(name = "idx_insert_batches_parallel", level = "debug", skip_all, fields(batch_count = batches.len()))]
    pub fn insert_batches_parallel(
        &self,
        batches: &[StoredBatch],
    ) -> Result<std::collections::HashMap<String, std::time::Duration>> {
        use std::time::Instant;

        if batches.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        // Use std::thread::scope for parallel CPU-bound work
        std::thread::scope(|scope| {
            // Each handle returns (index_name, index_type, duration, Result)
            let mut handles: Vec<(
                &str,
                &str,
                std::thread::ScopedJoinHandle<'_, (std::time::Duration, Result<()>)>,
            )> = Vec::new();

            // Spawn a thread for each BTree index
            for (name, index) in &self.btree_indexes {
                let handle = scope.spawn(move || -> (std::time::Duration, Result<()>) {
                    let start = Instant::now();
                    let result = (|| {
                        for stored in batches {
                            index.insert(&stored.data, stored.row_offset)?;
                        }
                        Ok(())
                    })();
                    (start.elapsed(), result)
                });
                handles.push((name.as_str(), "btree", handle));
            }

            // Spawn a thread for each HNSW index
            for (name, index) in &self.hnsw_indexes {
                let handle = scope.spawn(move || -> (std::time::Duration, Result<()>) {
                    let start = Instant::now();
                    let result = index.insert_batches(batches);
                    (start.elapsed(), result)
                });
                handles.push((name.as_str(), "hnsw", handle));
            }

            // Spawn a thread for each FTS index
            for (name, index) in &self.fts_indexes {
                let handle = scope.spawn(move || -> (std::time::Duration, Result<()>) {
                    let start = Instant::now();
                    let result = (|| {
                        for stored in batches {
                            index.insert(&stored.data, stored.row_offset)?;
                        }
                        Ok(())
                    })();
                    (start.elapsed(), result)
                });
                handles.push((name.as_str(), "fts", handle));
            }

            // Collect results, log timing, and check for errors. Keep the raw
            // `Duration` so sub-millisecond timings (the steady-state case for
            // BTree updates) are preserved instead of getting truncated to 0.
            let mut first_error: Option<Error> = None;
            let mut timings: Vec<(&str, &str, std::time::Duration)> = Vec::new();

            for (name, idx_type, handle) in handles {
                match handle.join() {
                    Ok((duration, Ok(()))) => {
                        timings.push((name, idx_type, duration));
                    }
                    Ok((duration, Err(e))) => {
                        timings.push((name, idx_type, duration));
                        if first_error.is_none() {
                            first_error = Some(e);
                        }
                    }
                    Err(_) => {
                        if first_error.is_none() {
                            first_error =
                                Some(Error::internal(format!("Index '{}' thread panicked", name)));
                        }
                    }
                }
            }

            if let Some(e) = first_error {
                return Err(e);
            }

            let duration_map: std::collections::HashMap<String, std::time::Duration> = timings
                .into_iter()
                .map(|(name, _idx_type, duration)| (name.to_string(), duration))
                .collect();

            // Update global watermark to the max batch position
            let max_bp = batches.iter().map(|b| b.batch_position).max().unwrap();
            self.update_max_indexed_batch_position(max_bp);

            Ok(duration_map)
        })
    }

    /// Get a BTree index by name.
    pub fn get_btree(&self, name: &str) -> Option<&BTreeMemIndex> {
        self.btree_indexes.get(name)
    }

    /// Get an HNSW vector index by name.
    pub fn get_hnsw(&self, name: &str) -> Option<&HnswMemIndex> {
        self.hnsw_indexes.get(name)
    }

    /// Get an FTS index by name.
    pub fn get_fts(&self, name: &str) -> Option<&FtsMemIndex> {
        self.fts_indexes.get(name)
    }

    /// Get a BTree index by field ID.
    ///
    /// Searches through all BTree indexes to find one matching the field_id.
    /// Use this for column-to-index resolution (column → field_id → index).
    pub fn get_btree_by_field_id(&self, field_id: i32) -> Option<&BTreeMemIndex> {
        self.btree_indexes
            .values()
            .find(|idx| idx.field_id() == field_id)
    }

    /// Get an HNSW vector index by field ID.
    pub fn get_hnsw_by_field_id(&self, field_id: i32) -> Option<&HnswMemIndex> {
        self.hnsw_indexes
            .values()
            .find(|idx| idx.field_id() == field_id)
    }

    /// Get an FTS index by field ID.
    ///
    /// Searches through all FTS indexes to find one matching the field_id.
    /// Use this for column-to-index resolution (column → field_id → index).
    pub fn get_fts_by_field_id(&self, field_id: i32) -> Option<&FtsMemIndex> {
        self.fts_indexes
            .values()
            .find(|idx| idx.field_id() == field_id)
    }

    /// Get a BTree index by column name.
    pub fn get_btree_by_column(&self, column: &str) -> Option<&BTreeMemIndex> {
        self.btree_indexes
            .values()
            .find(|idx| idx.column_name() == column)
    }

    /// Get an HNSW vector index by column name.
    pub fn get_hnsw_by_column(&self, column: &str) -> Option<&HnswMemIndex> {
        self.hnsw_indexes
            .values()
            .find(|idx| idx.column_name() == column)
    }

    /// Get an FTS index by column name.
    pub fn get_fts_by_column(&self, column: &str) -> Option<&FtsMemIndex> {
        self.fts_indexes
            .values()
            .find(|idx| idx.column_name() == column)
    }

    /// Check if the registry has any indexes.
    pub fn is_empty(&self) -> bool {
        self.btree_indexes.is_empty() && self.hnsw_indexes.is_empty() && self.fts_indexes.is_empty()
    }

    /// Get the total number of indexes.
    pub fn len(&self) -> usize {
        self.btree_indexes.len() + self.hnsw_indexes.len() + self.fts_indexes.len()
    }

    /// Get the global maximum indexed batch position.
    ///
    /// Returns the batch position up to which all data has been indexed.
    /// Queries should use `min(max_visible_batch_position, max_indexed_batch_position)`
    /// as their effective visibility to ensure consistent results.
    ///
    /// Returns 0 if no data has been indexed yet.
    pub fn max_indexed_batch_position(&self) -> usize {
        self.max_indexed_batch_position.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use log::warn;
    use std::sync::Arc;

    /// Check if an index type is supported and log warning if not.
    fn check_index_type_supported(index_type: &str) -> bool {
        match index_type.to_lowercase().as_str() {
            "btree" | "scalar" => true,
            "hnsw" | "vector" => true,
            "fts" | "inverted" | "fulltext" => true,
            _ => {
                warn!(
                    "Index type '{}' is not supported for MemWAL. \
                     Supported types: btree, hnsw, fts. Skipping.",
                    index_type
                );
                false
            }
        }
    }

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("description", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, start_id: i32) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![start_id, start_id + 1, start_id + 2])),
                Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
                Arc::new(StringArray::from(vec![
                    "hello world",
                    "goodbye world",
                    "hello again",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_index_registry() {
        let schema = create_test_schema();
        let mut registry = IndexStore::new();

        // field_id 0 for "id" column, field_id 2 for "description" column
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());
        registry.add_fts("desc_idx".to_string(), 2, "description".to_string());

        assert_eq!(registry.len(), 2);

        let batch = create_test_batch(&schema, 0);
        registry.insert(&batch, 0).unwrap();

        let btree = registry.get_btree("id_idx").unwrap();
        assert_eq!(btree.len(), 3);

        let fts = registry.get_fts("desc_idx").unwrap();
        assert_eq!(fts.doc_count(), 3);
    }

    #[test]
    fn test_check_index_type_supported() {
        assert!(check_index_type_supported("btree"));
        assert!(check_index_type_supported("BTree"));
        assert!(check_index_type_supported("hnsw"));
        assert!(check_index_type_supported("vector"));
        assert!(check_index_type_supported("fts"));
        assert!(check_index_type_supported("inverted"));

        assert!(!check_index_type_supported("unknown"));
    }

    #[test]
    fn test_from_configs() {
        let configs = vec![
            MemIndexConfig::BTree(BTreeIndexConfig {
                name: "pk_idx".to_string(),
                field_id: 0,
                column: "id".to_string(),
            }),
            MemIndexConfig::Fts(FtsIndexConfig::new(
                "search_idx".to_string(),
                2,
                "description".to_string(),
            )),
        ];

        let registry = IndexStore::from_configs(&configs, 100_000, 1_000).unwrap();
        assert_eq!(registry.len(), 2);
        assert!(registry.get_btree("pk_idx").is_some());
        assert!(registry.get_fts("search_idx").is_some());
        // Also test field_id lookup
        assert!(registry.get_btree_by_field_id(0).is_some());
        assert!(registry.get_fts_by_field_id(2).is_some());
    }

    #[test]
    fn test_index_store_max_indexed_batch_position() {
        let schema = create_test_schema();
        let mut registry = IndexStore::new();

        // field_id 0 for "id" column, field_id 2 for "description" column
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());
        registry.add_fts("desc_idx".to_string(), 2, "description".to_string());

        // Initial watermark should be 0 (no data indexed yet)
        assert_eq!(registry.max_indexed_batch_position(), 0);

        // Insert with batch position tracking
        let batch = create_test_batch(&schema, 0);
        registry
            .insert_with_batch_position(&batch, 0, Some(5))
            .unwrap();

        // Now watermark should be 5
        assert_eq!(registry.max_indexed_batch_position(), 5);

        // Insert with higher batch position
        registry
            .insert_with_batch_position(&batch, 3, Some(10))
            .unwrap();

        // Watermark should advance to 10
        assert_eq!(registry.max_indexed_batch_position(), 10);

        // Insert without batch position shouldn't change watermark
        registry.insert(&batch, 6).unwrap();
        assert_eq!(registry.max_indexed_batch_position(), 10);
    }

    #[test]
    fn test_get_index_by_name_and_field_id() {
        let mut registry = IndexStore::new();
        // field_id 0 for "id" column, field_id 2 for "description" column
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());
        registry.add_fts("desc_idx".to_string(), 2, "description".to_string());

        // Lookup by name
        assert!(registry.get_btree("id_idx").is_some());
        assert!(registry.get_btree("nonexistent").is_none());
        assert!(registry.get_fts("desc_idx").is_some());
        assert!(registry.get_fts("id_idx").is_none());

        // Lookup by field ID
        assert!(registry.get_btree_by_field_id(0).is_some());
        assert!(registry.get_btree_by_field_id(999).is_none());
        assert!(registry.get_fts_by_field_id(2).is_some());
        assert!(registry.get_fts_by_field_id(0).is_none());

        // Lookup by column name
        assert!(registry.get_btree_by_column("id").is_some());
        assert!(registry.get_btree_by_column("nonexistent").is_none());
        assert!(registry.get_fts_by_column("description").is_some());
    }
}
