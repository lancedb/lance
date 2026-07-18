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

mod arena_skiplist;
mod btree;
mod fts;
mod hnsw;
mod pk_key;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use datafusion::common::ScalarValue;

use super::memtable::batch_store::StoredBatch;
use arrow_array::RecordBatch;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_index::pbold;
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::inverted::InvertedListFormatVersion;
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
pub use pk_key::encode_pk_tuple;

use pk_key::encode_pk_batch;

/// Synthetic column the composite PK index is keyed on: the order-preserving
/// encoded tuple (see [`encode_pk_tuple`]), stored as `Binary` so a
/// [`BTreeMemIndex`]'s byte backend indexes it directly.
const PK_KEY_COLUMN: &str = "__pk_key__";

/// The memtable's primary-key index, used to answer "newest visible version of
/// this key" for dedup. Single-column PKs reuse the column's compact typed
/// [`BTreeMemIndex`] (no second copy); composite PKs key a `BTreeMemIndex` on
/// the order-preserving encoded tuple ([`encode_pk_tuple`]) instead. Either way
/// the lookup is a single seek on one `BTreeMemIndex`.
enum PkIndex {
    /// Arity 1: aliases a `btree_indexes` entry, so the insert loop maintains it.
    Single(Arc<BTreeMemIndex>),
    /// Arity >= 2: a `BTreeMemIndex` over the encoded-tuple `Binary` key,
    /// maintained explicitly in the insert paths (the original batch lacks the
    /// synthetic key column). `columns` are the PK columns in order, resolved
    /// against each batch's schema at insert time.
    Composite {
        index: Arc<BTreeMemIndex>,
        columns: Vec<String>,
    },
}

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
        let params = params.format_version(Self::fts_format_version_from_metadata(index_meta)?);

        Ok(Self::Fts(FtsIndexConfig::try_with_params(
            index_meta.name.clone(),
            field_id,
            column,
            params,
        )?))
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

    fn fts_format_version_from_metadata(
        index_meta: &IndexMetadata,
    ) -> Result<InvertedListFormatVersion> {
        match index_meta.index_version {
            // Legacy Arrow FTS indexes did not use the v1/v2 metadata values, but
            // the maintained-index path can only write the modern format.
            0 | 1 => Ok(InvertedListFormatVersion::V1),
            2 => Ok(InvertedListFormatVersion::V2),
            3 => Ok(InvertedListFormatVersion::V3),
            version => Err(Error::invalid_input(format!(
                "FTS index '{}' has unsupported index_version {}; expected 0, 1, 2, or 3",
                index_meta.name, version
            ))),
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
/// The store also carries the MemTable's `max_visible_batch_position`
/// watermark — the highest batch position that is durable in the WAL and
/// therefore safe for scanners to read. Scanners snapshot this at plan
/// construction time so every plan keys on a stable MVCC cursor.
pub struct IndexStore {
    /// BTree indexes keyed by index name. `Arc` so the primary-key BTrees can be
    /// shared into [`Self::pk_btrees`] without a second copy or a second insert.
    btree_indexes: HashMap<String, Arc<BTreeMemIndex>>,
    /// HNSW vector indexes keyed by index name.
    hnsw_indexes: HashMap<String, HnswMemIndex>,
    /// FTS indexes keyed by index name.
    fts_indexes: HashMap<String, FtsMemIndex>,
    /// The primary-key index (single-column or composite), or `None` without a
    /// primary key. Queried via [`Self::pk_newest_visible`] (see
    /// [`Self::enable_pk_index`]).
    pk_index: Option<PkIndex>,
    /// Maximum batch position that is durable in the WAL and therefore
    /// visible to scanners. Advanced unconditionally after a WAL append
    /// succeeds; not gated on whether any indexes are configured.
    max_visible_batch_position: AtomicUsize,
    /// Conservative flag set once this memtable has observed any primary-key
    /// rewrite while maintaining a search index. Search planners can push top-k
    /// into HNSW/FTS for append-only PK data, but must switch to
    /// newest-before-top-k search after an overwrite.
    pk_has_overrides: AtomicBool,
}

impl Default for IndexStore {
    fn default() -> Self {
        Self {
            btree_indexes: HashMap::new(),
            hnsw_indexes: HashMap::new(),
            fts_indexes: HashMap::new(),
            pk_index: None,
            max_visible_batch_position: AtomicUsize::new(0),
            pk_has_overrides: AtomicBool::new(false),
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
                "pk_index",
                &match &self.pk_index {
                    None => "none".to_string(),
                    Some(PkIndex::Single(b)) => format!("single({})", b.column_name()),
                    Some(PkIndex::Composite { columns, .. }) => {
                        format!("composite({})", columns.join(", "))
                    }
                },
            )
            .field(
                "max_visible_batch_position",
                &self.max_visible_batch_position.load(Ordering::Acquire),
            )
            .field(
                "pk_has_overrides",
                &self.pk_has_overrides.load(Ordering::Acquire),
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
                    let index = Arc::new(BTreeMemIndex::new(c.field_id, c.column.clone()));
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
                    let index = FtsMemIndex::try_with_params(
                        c.field_id,
                        c.column.clone(),
                        c.params.clone(),
                    )?;
                    registry.fts_indexes.insert(c.name.clone(), index);
                }
            }
        }

        Ok(registry)
    }

    /// Add a BTree/scalar index (skip-list backed). Low-level / test helper;
    /// the production memtable path goes through [`Self::from_configs`].
    pub fn add_btree(&mut self, name: String, field_id: i32, column: String) {
        self.btree_indexes
            .insert(name, Arc::new(BTreeMemIndex::new(field_id, column)));
    }

    /// Add an HNSW vector index with default build parameters.
    ///
    /// HNSW indexes must be configured before rows are inserted into a
    /// PK-indexed memtable. The vector planner's append-only fast path relies
    /// on `pk_has_overrides` being maintained for every row visible to HNSW.
    pub fn add_hnsw(
        &mut self,
        name: String,
        field_id: i32,
        column: String,
        distance_type: DistanceType,
        capacity: usize,
        max_batches: usize,
    ) {
        assert!(
            self.pk_index.is_none() || self.pk_is_empty(),
            "HNSW indexes must be configured before inserting rows into a PK memtable"
        );
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
    ///
    /// See [`Self::add_hnsw`] for the PK-indexed memtable lifecycle invariant.
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
        assert!(
            self.pk_index.is_none() || self.pk_is_empty(),
            "HNSW indexes must be configured before inserting rows into a PK memtable"
        );
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
    ///
    /// FTS indexes must be configured before rows are inserted into a
    /// PK-indexed memtable. FTS top-k pushdown relies on `pk_has_overrides`
    /// being maintained for every row visible to the index.
    pub fn add_fts(&mut self, name: String, field_id: i32, column: String) {
        assert!(
            self.pk_index.is_none() || self.pk_is_empty(),
            "FTS indexes must be configured before inserting rows into a PK memtable"
        );
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
    ) -> Result<()> {
        assert!(
            self.pk_index.is_none() || self.pk_is_empty(),
            "FTS indexes must be configured before inserting rows into a PK memtable"
        );
        self.fts_indexes.insert(
            name,
            FtsMemIndex::try_with_params(field_id, column, params)?,
        );
        Ok(())
    }

    /// Maintain a primary-key index so the memtable can answer "newest visible
    /// version of this key" (see [`Self::pk_newest_visible`]).
    ///
    /// Single-column PKs reuse an existing BTree on the field, else auto-create
    /// one under a `__pk__*` name so the normal insert loop maintains it (no
    /// second copy). Composite (arity >= 2) PKs key a `BTreeMemIndex` on the
    /// order-preserving encoded tuple (synthetic `PK_KEY_COLUMN`), maintained
    /// explicitly in the insert paths. Call once at construction, after
    /// [`Self::from_configs`] and before any inserts; a no-op when `pk_columns`
    /// is empty. Search indexes (HNSW/FTS) must also still be empty so every
    /// search-visible row participates in PK override tracking.
    pub fn enable_pk_index(&mut self, pk_columns: &[(String, i32)]) {
        if !pk_columns.is_empty() {
            assert!(
                self.hnsw_indexes.values().all(|idx| idx.is_empty())
                    && self.fts_indexes.values().all(|idx| idx.is_empty()),
                "Primary-key indexes must be configured before inserting rows into a search-indexed memtable"
            );
        }
        self.pk_index = match pk_columns {
            [] => None,
            [(column, field_id)] => {
                let btree = match self
                    .btree_indexes
                    .values()
                    .find(|b| b.field_id() == *field_id)
                {
                    Some(existing) => existing.clone(),
                    None => {
                        let btree = Arc::new(BTreeMemIndex::new(*field_id, column.clone()));
                        self.btree_indexes
                            .insert(format!("__pk__{column}"), btree.clone());
                        btree
                    }
                };
                Some(PkIndex::Single(btree))
            }
            multi => Some(PkIndex::Composite {
                // Synthetic field id (-1): the composite index is held directly,
                // never resolved by field id.
                index: Arc::new(BTreeMemIndex::new(-1, PK_KEY_COLUMN.to_string())),
                columns: multi.iter().map(|(c, _)| c.clone()).collect(),
            }),
        };
    }

    /// Whether the memtable has a primary-key index.
    pub fn has_pk_index(&self) -> bool {
        self.pk_index.is_some()
    }

    /// Sorted `(value, row_id)` training batches for the flushed on-disk PK
    /// BTree (the sidecar dedup index). Single-column emits the typed PK value;
    /// composite emits the order-preserving `Binary` encoded tuple. Empty when
    /// there is no primary key. Row positions line up 1:1 with the forward-
    /// written data file, so they are the flushed row ids directly.
    pub fn pk_training_batches(&self, batch_size: usize) -> Result<Vec<RecordBatch>> {
        match &self.pk_index {
            None => Ok(Vec::new()),
            Some(PkIndex::Single(btree)) => btree.to_training_batches(batch_size),
            Some(PkIndex::Composite { index, .. }) => index.to_training_batches(batch_size),
        }
    }

    /// Resolve the PK columns' positions in `batch` (composite insert helper).
    fn pk_batch_indices(batch: &RecordBatch, columns: &[String]) -> Result<Vec<usize>> {
        columns
            .iter()
            .map(|c| {
                batch
                    .schema()
                    .column_with_name(c)
                    .map(|(i, _)| i)
                    .ok_or_else(|| {
                        Error::invalid_input(format!("PK column '{c}' not found in batch"))
                    })
            })
            .collect()
    }

    /// Maintain the composite PK index for `batch` (no-op for single/no PK):
    /// encode the PK columns into the synthetic `PK_KEY_COLUMN` `Binary` column
    /// and feed that to the keyed `BTreeMemIndex`.
    fn insert_composite_pk(
        &self,
        batch: &RecordBatch,
        row_offset: u64,
        report_existing: bool,
    ) -> Result<bool> {
        if let Some(PkIndex::Composite { index, columns }) = &self.pk_index {
            let pk_indices = Self::pk_batch_indices(batch, columns)?;
            let encoded = encode_pk_batch(batch, &pk_indices)?;
            let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                PK_KEY_COLUMN,
                arrow_schema::DataType::Binary,
                false,
            )]));
            let key_batch = RecordBatch::try_new(schema, vec![Arc::new(encoded)])
                .map_err(|e| Error::invalid_input(e.to_string()))?;
            if report_existing {
                return index.insert_and_report_existing(&key_batch, row_offset);
            }
            index.insert(&key_batch, row_offset)?;
        }
        Ok(false)
    }

    /// The newest row position of the primary-key tuple `values` (in PK order)
    /// visible at `max_visible_row`, or `None`. A single seek either way:
    /// single-column probes the typed BTree; composite probes the encoded-tuple
    /// index. Collision-free, since `position` is the row identity.
    pub fn pk_newest_visible(
        &self,
        values: &[ScalarValue],
        max_visible_row: RowPosition,
    ) -> Option<RowPosition> {
        match &self.pk_index {
            None => None,
            Some(PkIndex::Single(btree)) => btree.get_newest_visible(&values[0], max_visible_row),
            Some(PkIndex::Composite { index, .. }) => {
                // An unsupported PK type would have failed at insert, so the
                // index can't hold a tuple this fails to encode. The probe key is
                // the same `Binary`-encoded tuple the insert path indexed.
                let key = encode_pk_tuple(values).ok()?;
                index.get_newest_visible(&ScalarValue::Binary(Some(key)), max_visible_row)
            }
        }
    }

    /// Whether `position` is the newest visible row of `values` — the recency
    /// check the active index-search arms apply to drop predicate-crossing
    /// stale hits. Callers gate on [`Self::has_pk_index`] first, since this is
    /// `false` (drop) when the memtable has no primary-key index.
    pub fn pk_is_newest(
        &self,
        values: &[ScalarValue],
        position: RowPosition,
        max_visible_row: RowPosition,
    ) -> bool {
        self.pk_newest_visible(values, max_visible_row) == Some(position)
    }

    /// Whether `key` has any version visible at `max_visible_row` — the
    /// cross-source block-list's existence query, snapshot-bounded so a
    /// not-yet-visible write can't shadow an older visible copy.
    ///
    /// `key` is already in the index's key space: the typed PK value for a
    /// single-column key, the `Binary`-encoded tuple for a composite one (built
    /// by `block_list::on_disk_pk_key`, the same key the flushed on-disk index is
    /// probed with). Both arities forward it straight to the keyed BTree.
    pub fn pk_contains_key(&self, key: &ScalarValue, max_visible_row: RowPosition) -> bool {
        match &self.pk_index {
            None => false,
            Some(PkIndex::Single(btree)) | Some(PkIndex::Composite { index: btree, .. }) => {
                btree.get_newest_visible(key, max_visible_row).is_some()
            }
        }
    }

    /// Whether the primary-key index holds no rows (or doesn't exist).
    pub fn pk_is_empty(&self) -> bool {
        match &self.pk_index {
            None => true,
            Some(PkIndex::Single(btree)) => btree.is_empty(),
            Some(PkIndex::Composite { index, .. }) => index.is_empty(),
        }
    }

    /// Whether this memtable has observed at least one PK rewrite.
    ///
    /// This is intentionally conservative: once true, it never resets for the
    /// lifetime of the memtable. That is enough for query planning because a
    /// memtable is flushed as a unit, and any rewrite means search-index top-k
    /// pushdown can be polluted by stale entries that must be removed before
    /// top-k. Scalar-only PK tables skip tracking because no search index uses
    /// the flag.
    pub fn pk_has_overrides(&self) -> bool {
        self.pk_has_overrides.load(Ordering::Acquire)
    }

    fn should_track_pk_overrides(&self) -> bool {
        (!self.hnsw_indexes.is_empty() || !self.fts_indexes.is_empty()) && !self.pk_has_overrides()
    }

    fn is_single_pk_btree(&self, index: &Arc<BTreeMemIndex>) -> bool {
        matches!(&self.pk_index, Some(PkIndex::Single(pk)) if Arc::ptr_eq(pk, index))
    }

    fn mark_pk_overrides_if_needed(&self, had_existing_pk: bool) {
        if had_existing_pk {
            self.pk_has_overrides.store(true, Ordering::Release);
        }
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
        let track_pk_overrides = self.should_track_pk_overrides();
        for index in self.btree_indexes.values() {
            if track_pk_overrides && self.is_single_pk_btree(index) {
                let had_existing = index.insert_and_report_existing(batch, row_offset)?;
                self.mark_pk_overrides_if_needed(had_existing);
            } else {
                index.insert(batch, row_offset)?;
            }
        }
        for index in self.hnsw_indexes.values() {
            index.insert(batch, row_offset)?;
        }
        for index in self.fts_indexes.values() {
            index.insert(batch, row_offset)?;
        }
        // Single-column PK aliases a `btree_indexes` entry (maintained above);
        // a composite PK has its own index, maintained here.
        let had_existing = self.insert_composite_pk(batch, row_offset, track_pk_overrides)?;
        self.mark_pk_overrides_if_needed(had_existing);

        // Update global watermark after all indexes have been updated
        if let Some(bp) = batch_position {
            self.advance_max_visible_batch_position(bp);
        }

        Ok(())
    }

    /// Advance the visibility watermark to at least `batch_pos`.
    ///
    /// The watermark only ever moves forward (idempotent max). The vector
    /// planner relies on the insert paths setting `pk_has_overrides` before
    /// calling this method, so any snapshot that can see a PK rewrite also
    /// observes `pk_has_overrides == true`.
    pub(crate) fn advance_max_visible_batch_position(&self, batch_pos: usize) {
        let mut current = self.max_visible_batch_position.load(Ordering::Acquire);
        while batch_pos > current {
            match self.max_visible_batch_position.compare_exchange_weak(
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

        let track_pk_overrides = self.should_track_pk_overrides();
        // BTree indexes: iterate batches (no cross-batch optimization benefit)
        for index in self.btree_indexes.values() {
            let track_this_index = track_pk_overrides && self.is_single_pk_btree(index);
            let mut had_existing = false;
            for stored in batches {
                if track_this_index {
                    had_existing |=
                        index.insert_and_report_existing(&stored.data, stored.row_offset)?;
                } else {
                    index.insert(&stored.data, stored.row_offset)?;
                }
            }
            self.mark_pk_overrides_if_needed(had_existing);
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

        // Single-column PK aliases a `btree_indexes` entry (maintained above);
        // a composite PK has its own index, maintained here.
        let mut had_existing = false;
        for stored in batches {
            had_existing |=
                self.insert_composite_pk(&stored.data, stored.row_offset, track_pk_overrides)?;
        }
        self.mark_pk_overrides_if_needed(had_existing);

        // Update global watermark to the max batch position
        let max_bp = batches.iter().map(|b| b.batch_position).max().unwrap();
        self.advance_max_visible_batch_position(max_bp);

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

        let track_pk_overrides = self.should_track_pk_overrides();
        // Use std::thread::scope for parallel CPU-bound work
        std::thread::scope(|scope| {
            // Each handle returns (index_name, index_type, duration, Result)
            let mut handles: Vec<(
                &str,
                &str,
                std::thread::ScopedJoinHandle<'_, (std::time::Duration, Result<bool>)>,
            )> = Vec::new();

            // Spawn a thread for each BTree index
            for (name, index) in &self.btree_indexes {
                let track_this_index = track_pk_overrides && self.is_single_pk_btree(index);
                let handle = scope.spawn(move || -> (std::time::Duration, Result<bool>) {
                    let start = Instant::now();
                    let result = (|| {
                        let mut had_existing = false;
                        for stored in batches {
                            if track_this_index {
                                had_existing |= index
                                    .insert_and_report_existing(&stored.data, stored.row_offset)?;
                            } else {
                                index.insert(&stored.data, stored.row_offset)?;
                            }
                        }
                        Ok(had_existing)
                    })();
                    (start.elapsed(), result)
                });
                handles.push((name.as_str(), "btree", handle));
            }

            // Spawn a thread for each HNSW index
            for (name, index) in &self.hnsw_indexes {
                let handle = scope.spawn(move || -> (std::time::Duration, Result<bool>) {
                    let start = Instant::now();
                    let result = index.insert_batches(batches).map(|_| false);
                    (start.elapsed(), result)
                });
                handles.push((name.as_str(), "hnsw", handle));
            }

            // Spawn a thread for each FTS index
            for (name, index) in &self.fts_indexes {
                let handle = scope.spawn(move || -> (std::time::Duration, Result<bool>) {
                    let start = Instant::now();
                    let result = (|| {
                        for stored in batches {
                            index.insert(&stored.data, stored.row_offset)?;
                        }
                        Ok(false)
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
            let mut had_existing_pk = false;

            for (name, idx_type, handle) in handles {
                match handle.join() {
                    Ok((duration, Ok(had_existing))) => {
                        timings.push((name, idx_type, duration));
                        had_existing_pk |= had_existing;
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
            self.mark_pk_overrides_if_needed(had_existing_pk);

            let duration_map: std::collections::HashMap<String, std::time::Duration> = timings
                .into_iter()
                .map(|(name, _idx_type, duration)| (name.to_string(), duration))
                .collect();

            // Single-column PK aliases a `btree_indexes` entry — its thread above
            // already maintained it (and joined). A composite PK has its own
            // index; maintain it here before the watermark advances so the
            // visible prefix is fully indexed.
            let mut had_existing = false;
            for stored in batches {
                had_existing |=
                    self.insert_composite_pk(&stored.data, stored.row_offset, track_pk_overrides)?;
            }
            self.mark_pk_overrides_if_needed(had_existing);

            // Update global watermark to the max batch position
            let max_bp = batches.iter().map(|b| b.batch_position).max().unwrap();
            self.advance_max_visible_batch_position(max_bp);

            Ok(duration_map)
        })
    }

    /// Get a BTree index by name.
    pub fn get_btree(&self, name: &str) -> Option<&BTreeMemIndex> {
        self.btree_indexes.get(name).map(Arc::as_ref)
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
            .map(Arc::as_ref)
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
            .map(Arc::as_ref)
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

    /// Get the visibility watermark (max batch position safe to read).
    ///
    /// Returns the highest batch position whose data is durable in the WAL
    /// and therefore visible to scanners. Scanners snapshot this at plan
    /// construction time so every plan runs against a stable cursor.
    ///
    /// Returns 0 before any WAL flush has advanced the watermark.
    pub fn max_visible_batch_position(&self) -> usize {
        self.max_visible_batch_position.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use log::warn;
    use std::sync::Arc;
    use uuid::Uuid;

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

    fn fts_index_metadata(index_version: i32) -> IndexMetadata {
        let details =
            pbold::InvertedIndexDetails::try_from(&InvertedIndexParams::default()).unwrap();
        fts_index_metadata_with_details(index_version, Some(details))
    }

    fn fts_index_metadata_with_details(
        index_version: i32,
        details: Option<pbold::InvertedIndexDetails>,
    ) -> IndexMetadata {
        let index_details = details.map(|details| {
            let mut value = Vec::new();
            details.encode(&mut value).unwrap();
            Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/lance.index.InvertedIndexDetails".to_string(),
                value,
            })
        });

        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![2],
            name: "desc_idx".to_string(),
            dataset_version: 1,
            fragment_bitmap: None,
            index_details,
            index_version,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    /// Single-column `id` batch for primary-key lookup tests.
    fn id_batch(ids: &[i32]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(ids.to_vec()))],
        )
        .unwrap()
    }

    fn id_vector_batch(ids: &[i32]) -> RecordBatch {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                false,
            ),
        ]));
        let mut vectors = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        for id in ids {
            vectors.values().append_value(*id as f32);
            vectors.values().append_value(*id as f32 + 0.5);
            vectors.append(true);
        }
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(vectors.finish()),
            ],
        )
        .unwrap()
    }

    fn id_name_vector_batch(rows: &[(i32, &str)]) -> RecordBatch {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                false,
            ),
        ]));
        let mut ids = Vec::with_capacity(rows.len());
        let mut names = Vec::with_capacity(rows.len());
        let mut vectors = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        for (id, name) in rows {
            ids.push(*id);
            names.push(*name);
            vectors.values().append_value(*id as f32);
            vectors.values().append_value(name.len() as f32);
            vectors.append(true);
        }
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(vectors.finish()),
            ],
        )
        .unwrap()
    }

    #[test]
    fn pk_newest_visible_single_column() {
        let mut store = IndexStore::new();
        store.enable_pk_index(&[("id".to_string(), 0)]);
        // id=1 at positions 0 and 2 (an update), id=2 at position 1.
        store.insert(&id_batch(&[1, 2]), 0).unwrap();
        store.insert(&id_batch(&[1]), 2).unwrap();

        let one = [ScalarValue::Int32(Some(1))];
        // Watermark above the update sees the newest position; below it, the older.
        assert_eq!(store.pk_newest_visible(&one, 5), Some(2));
        assert_eq!(store.pk_newest_visible(&one, 1), Some(0));
        assert!(store.pk_is_newest(&one, 2, 5));
        assert!(!store.pk_is_newest(&one, 0, 5));
        // Absent key (probed by the typed value, as the block-list does).
        assert!(!store.pk_contains_key(&ScalarValue::Int32(Some(9)), 5));
    }

    #[test]
    fn pk_has_overrides_tracks_single_column_rewrites() {
        let mut store = IndexStore::new();
        store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        store.enable_pk_index(&[("id".to_string(), 0)]);

        store.insert(&id_vector_batch(&[1, 2]), 0).unwrap();
        assert!(
            !store.pk_has_overrides(),
            "append-only PK inserts should keep HNSW eligible"
        );

        store.insert(&id_vector_batch(&[3, 3]), 2).unwrap();
        assert!(
            store.pk_has_overrides(),
            "duplicate PKs within one insert must disable HNSW"
        );
    }

    #[test]
    #[should_panic(
        expected = "Primary-key indexes must be configured before inserting rows into a search-indexed memtable"
    )]
    fn enable_pk_index_after_search_rows_panics() {
        let mut store = IndexStore::new();
        store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        store.insert(&id_vector_batch(&[1, 2]), 0).unwrap();

        store.enable_pk_index(&[("id".to_string(), 0)]);
    }

    #[test]
    fn pk_has_overrides_tracks_single_column_rewrites_across_inserts() {
        let mut store = IndexStore::new();
        store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        store.enable_pk_index(&[("id".to_string(), 0)]);

        store.insert(&id_vector_batch(&[1, 2]), 0).unwrap();
        assert!(
            !store.pk_has_overrides(),
            "append-only PK inserts should keep HNSW eligible"
        );

        store.insert(&id_vector_batch(&[1]), 2).unwrap();
        assert!(
            store.pk_has_overrides(),
            "single-column PK rewrites across inserts must disable HNSW"
        );
    }

    #[test]
    fn pk_has_overrides_skips_scalar_only_tables() {
        let mut store = IndexStore::new();
        store.enable_pk_index(&[("id".to_string(), 0)]);

        store.insert(&id_batch(&[1, 1]), 0).unwrap();
        assert!(
            !store.pk_has_overrides(),
            "scalar-only PK tables should not pay override tracking cost"
        );
    }

    #[test]
    fn pk_has_overrides_tracks_fts_rewrites() {
        let mut store = IndexStore::new();
        store.enable_pk_index(&[("id".to_string(), 0)]);
        store.add_fts("text_fts".to_string(), 1, "text".to_string());

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(StringArray::from(vec!["alpha", "beta"])),
            ],
        )
        .unwrap();
        store.insert(&batch, 0).unwrap();
        assert!(
            store.pk_has_overrides(),
            "FTS PK rewrites must disable index-level FTS limit/WAND pushdown"
        );
    }

    #[test]
    fn pk_newest_visible_composite_seeks_encoded_tuple() {
        let mut store = IndexStore::new();
        store.enable_pk_index(&[("id".to_string(), 0), ("name".to_string(), 1)]);
        // Rows: (1,"a")@0, (1,"b")@1, (1,"a")@2 — an update of (1,"a").
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1])),
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
            ],
        )
        .unwrap();
        store.insert(&batch, 0).unwrap();

        let tuple_1a = [ScalarValue::Int32(Some(1)), ScalarValue::from("a")];
        let tuple_1b = [ScalarValue::Int32(Some(1)), ScalarValue::from("b")];
        // (1,"a")'s newest visible row is its re-write at position 2.
        assert_eq!(store.pk_newest_visible(&tuple_1a, 5), Some(2));
        assert!(store.pk_is_newest(&tuple_1a, 2, 5));
        assert!(!store.pk_is_newest(&tuple_1a, 0, 5));
        // (1,"b") only exists at position 1.
        assert_eq!(store.pk_newest_visible(&tuple_1b, 5), Some(1));
        // Watermark below the re-write: the older (1,"a")@0 is the newest visible.
        assert_eq!(store.pk_newest_visible(&tuple_1a, 1), Some(0));
        // An absent tuple (probed by its Binary-encoded key, as the block-list
        // does).
        let tuple_2a = [ScalarValue::Int32(Some(2)), ScalarValue::from("a")];
        let key_2a = ScalarValue::Binary(Some(encode_pk_tuple(&tuple_2a).unwrap()));
        assert!(!store.pk_contains_key(&key_2a, 5));
    }

    #[test]
    fn pk_has_overrides_tracks_composite_rewrites() {
        let mut store = IndexStore::new();
        store.add_hnsw(
            "vector_hnsw".to_string(),
            2,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        store.enable_pk_index(&[("id".to_string(), 0), ("name".to_string(), 1)]);
        let first = id_name_vector_batch(&[(1, "a"), (1, "b")]);
        store.insert(&first, 0).unwrap();
        assert!(!store.pk_has_overrides());

        let rewrite = id_name_vector_batch(&[(1, "a")]);
        store.insert(&rewrite, 2).unwrap();
        assert!(
            store.pk_has_overrides(),
            "repeated composite PK must disable HNSW"
        );
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
    fn fts_from_metadata_preserves_format_version() {
        let arrow_schema = create_test_schema();
        let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        for (index_version, expected_format_version) in [
            (0, InvertedListFormatVersion::V1),
            (1, InvertedListFormatVersion::V1),
            (2, InvertedListFormatVersion::V2),
        ] {
            let config =
                MemIndexConfig::fts_from_metadata(&fts_index_metadata(index_version), &schema)
                    .unwrap();

            match config {
                MemIndexConfig::Fts(config) => {
                    assert_eq!(
                        config.params.resolved_format_version(),
                        expected_format_version
                    );
                }
                _ => unreachable!("fts metadata should create an FTS config"),
            }
        }
    }

    #[test]
    fn fts_from_metadata_rejects_unsupported_format_version() {
        let arrow_schema = create_test_schema();
        let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        let err = MemIndexConfig::fts_from_metadata(&fts_index_metadata(4), &schema).unwrap_err();
        assert!(
            err.to_string().contains("unsupported index_version 4"),
            "{err}"
        );
    }

    #[test]
    fn fts_from_metadata_rejects_v3_without_256_block_size() {
        let arrow_schema = create_test_schema();
        let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

        let missing_details =
            MemIndexConfig::fts_from_metadata(&fts_index_metadata_with_details(3, None), &schema)
                .unwrap_err();
        assert!(
            missing_details
                .to_string()
                .contains("requires block_size=256"),
            "{missing_details}"
        );
        assert!(
            missing_details.to_string().contains("got 128"),
            "{missing_details}"
        );

        let default_details =
            MemIndexConfig::fts_from_metadata(&fts_index_metadata(3), &schema).unwrap_err();
        assert!(
            default_details
                .to_string()
                .contains("requires block_size=256"),
            "{default_details}"
        );
        assert!(
            default_details.to_string().contains("got 128"),
            "{default_details}"
        );
    }

    #[test]
    fn fts_from_metadata_accepts_v3_with_256_block_size() {
        let arrow_schema = create_test_schema();
        let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();
        let params = InvertedIndexParams::default().block_size(256).unwrap();
        let details = pbold::InvertedIndexDetails::try_from(&params).unwrap();

        let config = MemIndexConfig::fts_from_metadata(
            &fts_index_metadata_with_details(3, Some(details)),
            &schema,
        )
        .unwrap();

        match config {
            MemIndexConfig::Fts(config) => {
                assert_eq!(
                    config.params.resolved_format_version(),
                    InvertedListFormatVersion::V3
                );
                assert_eq!(config.params.posting_block_size(), 256);
            }
            _ => unreachable!("fts metadata should create an FTS config"),
        }
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
    fn test_index_store_max_visible_batch_position() {
        let schema = create_test_schema();
        let mut registry = IndexStore::new();

        // field_id 0 for "id" column, field_id 2 for "description" column
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());
        registry.add_fts("desc_idx".to_string(), 2, "description".to_string());

        // Initial watermark should be 0 (no data indexed yet)
        assert_eq!(registry.max_visible_batch_position(), 0);

        // Insert with batch position tracking
        let batch = create_test_batch(&schema, 0);
        registry
            .insert_with_batch_position(&batch, 0, Some(5))
            .unwrap();

        // Now watermark should be 5
        assert_eq!(registry.max_visible_batch_position(), 5);

        // Insert with higher batch position
        registry
            .insert_with_batch_position(&batch, 3, Some(10))
            .unwrap();

        // Watermark should advance to 10
        assert_eq!(registry.max_visible_batch_position(), 10);

        // Insert without batch position shouldn't change watermark
        registry.insert(&batch, 6).unwrap();
        assert_eq!(registry.max_visible_batch_position(), 10);
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
