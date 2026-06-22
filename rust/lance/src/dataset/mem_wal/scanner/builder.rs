// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! LSM Scanner builder.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::common::ScalarValue;
use datafusion::common::ToDFSchema;
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion::prelude::{Expr, SessionContext};
use futures::TryStreamExt;
use lance_core::{Error, Result, is_system_column};
use uuid::Uuid;

use super::collector::{InMemoryMemTableRef, InMemoryMemTables, LsmDataSourceCollector};
use super::data_source::{FreshTierWatermark, ShardSnapshot};
use super::flushed_cache::{DatasetCache, GenerationWarmer};
use super::planner::LsmScanPlanner;
use super::point_lookup::LsmPointLookupPlanner;
use crate::dataset::Dataset;
use crate::session::Session;

/// If `filter` is a point-lookup shape on `pk_col` — `pk = lit` (either
/// operand order) or `pk IN (lit, …)` — return the literal key values. Any
/// other shape returns `None`, so the scanner falls through to the general
/// scan plan. Type coercion is left to the lookup path (an exact-type literal
/// takes the fast BTree path; a coercible one falls back internally).
fn extract_pk_point_keys(filter: &Expr, pk_col: &str) -> Option<Vec<ScalarValue>> {
    match filter {
        Expr::BinaryExpr(b) if matches!(b.op, Operator::Eq) => {
            match (b.left.as_ref(), b.right.as_ref()) {
                (Expr::Column(c), Expr::Literal(lit, _))
                | (Expr::Literal(lit, _), Expr::Column(c))
                    if c.name == pk_col =>
                {
                    Some(vec![lit.clone()])
                }
                _ => None,
            }
        }
        Expr::InList(in_list) if !in_list.negated => {
            let Expr::Column(c) = in_list.expr.as_ref() else {
                return None;
            };
            if c.name != pk_col {
                return None;
            }
            let mut vals = Vec::with_capacity(in_list.list.len());
            for e in &in_list.list {
                let Expr::Literal(lit, _) = e else {
                    return None; // a non-literal IN element → not a point lookup
                };
                vals.push(lit.clone());
            }
            (!vals.is_empty()).then_some(vals)
        }
        _ => None,
    }
}

/// Either a base Lance table, or an explicit base path used to resolve
/// flushed-generation directories when no base dataset is configured.
enum BaseSource {
    Table(Arc<Dataset>),
    PathOnly(String),
}

/// Scanner for LSM tree data spanning base table, flushed MemTables, and active MemTable.
///
/// This scanner provides a unified interface for querying data across multiple
/// LSM tree levels:
/// - Base table (merged data, generation = 0)
/// - Flushed MemTables (persisted but not yet merged, generation = 1, 2, ...)
/// - Active MemTable (in-memory buffer, highest generation)
///
/// The scanner automatically handles deduplication by primary key, keeping
/// the newest version based on generation number and row address.
///
/// # Example
///
/// ```ignore
/// let scanner = LsmScanner::new(base_table, shard_snapshots, vec!["pk".to_string()])
///     .project(&["id", "name"])
///     .filter("id > 10")?
///     .limit(100, None);
///
/// let results = scanner.try_into_batch().await?;
/// ```
pub struct LsmScanner {
    // Data sources
    base: BaseSource,
    /// Schema used for projection, empty plans, and filter parsing.
    /// Derived from the base dataset when one is present, otherwise supplied
    /// explicitly by [`Self::without_base_table`].
    schema: SchemaRef,
    shard_snapshots: Vec<ShardSnapshot>,
    /// In-memory memtables by shard (active + frozen-awaiting-flush), so
    /// the scanner path carries frozen-undrained generations too.
    in_memory_memtables: HashMap<Uuid, InMemoryMemTables>,

    // Query configuration
    projection: Option<Vec<String>>,
    filter: Option<Expr>,
    limit: Option<usize>,
    offset: Option<usize>,

    // Internal columns
    with_row_address: bool,
    with_memtable_gen: bool,

    // Primary key columns (required for deduplication)
    pk_columns: Vec<String>,

    /// Session threaded into flushed-generation opens so the first open of
    /// each generation populates the shared index / file-metadata caches.
    /// Defaults to the base table's session when one is present.
    session: Option<Arc<Session>>,
    /// Cache of opened flushed-generation datasets. When set, repeated
    /// queries against the same generation skip the manifest read entirely.
    flushed_cache: Option<Arc<dyn DatasetCache>>,
    /// Optional warmer fired on first open of a flushed generation.
    warmer: Option<Arc<dyn GenerationWarmer>>,
    /// Over-fetch multiple for block-listed sources in search plans
    /// (see [`super::LsmFtsSearchPlanner::with_overfetch_factor`]).
    overfetch_factor: Option<f64>,
}

impl LsmScanner {
    /// Create a new LSM scanner.
    ///
    /// # Arguments
    ///
    /// * `base_table` - The base Lance table (merged data)
    /// * `shard_snapshots` - Snapshots of shard states from MemWAL index
    /// * `pk_columns` - Primary key column names for deduplication
    pub fn new(
        base_table: Arc<Dataset>,
        shard_snapshots: Vec<ShardSnapshot>,
        pk_columns: Vec<String>,
    ) -> Self {
        let lance_schema = base_table.schema();
        let arrow_schema: arrow_schema::Schema = lance_schema.into();
        // Default the session to the base table's so the common path reuses
        // the shared index / metadata caches without extra wiring. An
        // explicit `with_session` still overrides this.
        let session = Some(base_table.session());
        Self {
            base: BaseSource::Table(base_table),
            schema: Arc::new(arrow_schema),
            shard_snapshots,
            in_memory_memtables: HashMap::new(),
            projection: None,
            filter: None,
            limit: None,
            offset: None,
            with_row_address: false,
            with_memtable_gen: false,
            pk_columns,
            session,
            flushed_cache: None,
            warmer: None,
            overfetch_factor: None,
        }
    }

    /// Create a scanner that reads only the fresh tier (active memtable and
    /// flushed generations) without including a base Lance table.
    ///
    /// This is useful when the caller owns the base read path separately and
    /// only needs the WAL's contribution: active memtable ∪ L0 flushed
    /// generations. Deduplication semantics are unchanged — newer generations
    /// still win on PK conflicts.
    ///
    /// # Arguments
    ///
    /// * `schema` - Schema used for projection, filter parsing, and empty plans.
    ///   Should match the schema flushed generations were written with.
    /// * `base_path` - Table-root URI used to resolve relative flushed paths.
    /// * `shard_snapshots` - Snapshots of shard states from MemWAL index.
    /// * `pk_columns` - Primary key column names for deduplication.
    pub fn without_base_table(
        schema: SchemaRef,
        base_path: impl Into<String>,
        shard_snapshots: Vec<ShardSnapshot>,
        pk_columns: Vec<String>,
    ) -> Self {
        Self {
            base: BaseSource::PathOnly(base_path.into()),
            schema,
            shard_snapshots,
            in_memory_memtables: HashMap::new(),
            projection: None,
            filter: None,
            limit: None,
            offset: None,
            with_row_address: false,
            with_memtable_gen: false,
            pk_columns,
            session: None,
            flushed_cache: None,
            warmer: None,
            overfetch_factor: None,
        }
    }

    /// Set a shard's active memtable. Back-compat / test entry point; the
    /// read path uses [`Self::with_in_memory_memtables`]. Replaces the
    /// active memtable, preserving any frozen memtables already registered.
    pub fn with_active_memtable(mut self, shard_id: Uuid, memtable: InMemoryMemTableRef) -> Self {
        match self.in_memory_memtables.entry(shard_id) {
            Entry::Occupied(mut e) => e.get_mut().active = memtable,
            Entry::Vacant(e) => {
                e.insert(InMemoryMemTables {
                    active: memtable,
                    frozen: Vec::new(),
                });
            }
        }
        self
    }

    /// Register a shard's in-memory memtables (active + frozen-awaiting-
    /// flush) captured atomically by `ShardWriter::in_memory_memtable_refs`.
    /// The read path's entry point — closes the concurrent-read-vs-flush
    /// hole by carrying frozen-undrained generations into the scan.
    pub fn with_in_memory_memtables(
        mut self,
        shard_id: Uuid,
        memtables: InMemoryMemTables,
    ) -> Self {
        self.in_memory_memtables.insert(shard_id, memtables);
        self
    }

    /// Thread an existing session into flushed-generation opens.
    ///
    /// The first open of each flushed generation then populates the shared
    /// index / file-metadata caches, so later queries skip re-decoding them.
    /// When a base table is configured this defaults to its session; call
    /// this to override (e.g. on a fresh-tier-only scanner that owns its own
    /// long-lived session).
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Inject a cache of opened flushed-generation datasets.
    ///
    /// With a cache, repeated queries against the same generation become a
    /// pure `Arc::clone` with no manifest read or object-store I/O. The cache
    /// is owned and sized by the caller (any [`DatasetCache`] impl, e.g.
    /// [`FlushedMemTableCache`](super::FlushedMemTableCache)); not set by
    /// default, so behavior is unchanged unless opted in.
    pub fn with_flushed_cache(mut self, cache: Arc<dyn DatasetCache>) -> Self {
        self.flushed_cache = Some(cache);
        self
    }

    /// Inject the warmer fired on first open of a flushed generation. Not set by
    /// default, so behavior is unchanged unless opted in.
    pub fn with_warmer(mut self, warmer: Arc<dyn GenerationWarmer>) -> Self {
        self.warmer = Some(warmer);
        self
    }

    /// Set the over-fetch multiple block-listed sources use in search plans
    /// so they still yield `k` live rows after cross-generation dedup.
    /// Threaded into [`super::LsmFtsSearchPlanner`]; clamped to `>= 1.0`.
    pub fn with_overfetch_factor(mut self, factor: f64) -> Self {
        self.overfetch_factor = Some(factor);
        self
    }

    /// Project specific columns.
    ///
    /// If not called, all columns from the base schema are included.
    /// Primary key columns are always included for deduplication.
    pub fn project(mut self, columns: &[&str]) -> Self {
        self.projection = Some(columns.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Set filter expression using SQL-like syntax.
    ///
    /// The filter is pushed down to each data source when possible.
    pub fn filter(mut self, filter_expr: &str) -> Result<Self> {
        let ctx = SessionContext::new();
        let df_schema = self
            .schema
            .as_ref()
            .clone()
            .to_dfschema()
            .map_err(|e| Error::invalid_input(format!("Failed to create DFSchema: {}", e)))?;
        let expr = ctx.parse_sql_expr(filter_expr, &df_schema).map_err(|e| {
            Error::invalid_input(format!("Failed to parse filter expression: {}", e))
        })?;
        self.filter = Some(expr);
        Ok(self)
    }

    /// Set filter expression directly.
    pub fn filter_expr(mut self, expr: Expr) -> Self {
        self.filter = Some(expr);
        self
    }

    /// Limit the number of results.
    pub fn limit(mut self, limit: usize, offset: Option<usize>) -> Self {
        self.limit = Some(limit);
        self.offset = offset;
        self
    }

    /// Include `_rowaddr` column in output.
    ///
    /// The row address is used for ordering within a generation.
    pub fn with_row_address(mut self) -> Self {
        self.with_row_address = true;
        self
    }

    /// Include `_memtable_gen` column in output.
    ///
    /// The generation column shows which data source each row came from:
    /// - 0: Base table
    /// - 1, 2, ...: MemTable generations (higher = newer)
    pub fn with_memtable_gen(mut self) -> Self {
        self.with_memtable_gen = true;
        self
    }

    /// Get the output schema.
    pub fn schema(&self) -> SchemaRef {
        // For now, return the configured schema. Full implementation would
        // compute the projected schema with optional _gen/_rowaddr columns.
        self.schema.clone()
    }

    /// Whether the projection requests any system column (e.g. `_rowaddr`,
    /// `_memtable_gen`), which only the union scan path can produce.
    fn projection_has_system_columns(&self) -> bool {
        self.projection
            .as_ref()
            .map(|p| p.iter().any(|c| is_system_column(c)))
            .unwrap_or(false)
    }

    /// Create the execution plan.
    pub async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let collector = self.build_collector();
        let base_schema = self.schema();

        // Fast point-lookup routing: a `pk = lit` / `pk IN (..)` filter on the
        // single pk column — with no offset and no scan-only system columns —
        // bypasses the union/dedup scan for the direct BTree point-lookup path
        // (`LsmPointLookupPlanner`), composed as a normal `ExecutionPlan` so a
        // `limit` still applies on top. Any other shape falls through to the
        // general scan, so this never changes results for unmatched queries.
        if self.pk_columns.len() == 1
            && self.offset.is_none()
            && !self.with_memtable_gen
            && !self.with_row_address
            && !self.projection_has_system_columns()
            && let Some(filter) = &self.filter
            && let Some(keys) = extract_pk_point_keys(filter, &self.pk_columns[0])
        {
            let mut planner =
                LsmPointLookupPlanner::new(collector, self.pk_columns.clone(), base_schema);
            if let Some(session) = &self.session {
                planner = planner.with_session(session.clone());
            }
            if let Some(cache) = &self.flushed_cache {
                planner = planner.with_flushed_cache(cache.clone());
            }
            if let Some(warmer) = &self.warmer {
                planner = planner.with_warmer(warmer.clone());
            }
            let plan = planner
                .plan_point_lookup(&keys, self.projection.as_deref())
                .await?;
            return Ok(match self.limit {
                Some(n) => Arc::new(GlobalLimitExec::new(plan, 0, Some(n))),
                None => plan,
            });
        }

        let mut planner = LsmScanPlanner::new(collector, self.pk_columns.clone(), base_schema);
        if let Some(session) = &self.session {
            planner = planner.with_session(session.clone());
        }
        if let Some(cache) = &self.flushed_cache {
            planner = planner.with_flushed_cache(cache.clone());
        }
        if let Some(warmer) = &self.warmer {
            planner = planner.with_warmer(warmer.clone());
        }
        if let Some(factor) = self.overfetch_factor {
            planner = planner.with_overfetch_factor(factor);
        }

        planner
            .plan_scan(
                self.projection.as_deref(),
                self.filter.as_ref(),
                self.limit,
                self.offset,
                self.with_memtable_gen,
                self.with_row_address,
            )
            .await
    }

    /// Build a local-scoring FTS plan spanning base + flushed + active sources.
    ///
    /// Routes through [`super::LsmFtsSearchPlanner`]. Output schema is
    /// `projection ∪ pk_columns + _score`; per-source local BM25 `_score`
    /// is merged DESC and capped at `k`. `column` must be FTS-indexed on
    /// the queried sources.
    pub async fn full_text_search(
        &self,
        column: &str,
        query: lance_index::scalar::FullTextSearchQuery,
        k: usize,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let collector = self.build_collector();
        let base_schema = self.schema();
        let mut planner =
            super::LsmFtsSearchPlanner::new(collector, self.pk_columns.clone(), base_schema);
        if let Some(session) = &self.session {
            planner = planner.with_session(session.clone());
        }
        if let Some(cache) = &self.flushed_cache {
            planner = planner.with_flushed_cache(cache.clone());
        }
        if let Some(warmer) = &self.warmer {
            planner = planner.with_warmer(warmer.clone());
        }
        if let Some(factor) = self.overfetch_factor {
            planner = planner.with_overfetch_factor(factor);
        }
        planner
            .plan_search(column, query, k, self.projection.as_deref())
            .await
    }

    /// Execute the scan and return a stream of record batches.
    pub async fn try_into_stream(&self) -> Result<SendableRecordBatchStream> {
        let plan = self.create_plan().await?;
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        plan.execute(0, task_ctx)
            .map_err(|e| Error::io(format!("Failed to execute plan: {}", e)))
    }

    /// Execute the scan and collect all results into a single RecordBatch.
    pub async fn try_into_batch(&self) -> Result<RecordBatch> {
        let stream = self.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| Error::io(format!("Failed to collect batches: {}", e)))?;

        if batches.is_empty() {
            let schema = self.schema();
            return Ok(RecordBatch::new_empty(schema));
        }

        let schema = batches[0].schema();
        arrow_select::concat::concat_batches(&schema, &batches)
            .map_err(|e| Error::io(format!("Failed to concatenate batches: {}", e)))
    }

    /// Count the number of rows that match the query.
    pub async fn count_rows(&self) -> Result<u64> {
        let stream = self.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| Error::io(format!("Failed to count rows: {}", e)))?;

        Ok(batches.iter().map(|b| b.num_rows() as u64).sum())
    }

    /// Test which `pks` have been (re)written in the WAL fresh tier — the active
    /// and frozen memtables and flushed generations this scanner spans — i.e.
    /// are shadowed above the base table. `pks` is a batch whose columns include
    /// the primary-key columns; the returned `Vec<bool>` is aligned with its
    /// rows. Hashing matches the scanner's internal dedup, so the caller never
    /// hashes PKs itself. Flushed membership comes from the injected
    /// [`DatasetCache`] when one is set.
    pub async fn contains_pks(&self, pks: &RecordBatch) -> Result<Vec<bool>> {
        self.contains_pks_at(pks, None).await
    }

    /// As-of variant of [`Self::contains_pks`]. Membership is evaluated against
    /// a per-shard watermark on the fresh tier, supplied via `watermarks` (see
    /// [`FreshTierWatermark`]), matching the tier a prior scan observed and
    /// avoiding the two-snapshot skew that would drop a base row with no
    /// delivered replacement. `None` evaluates against the live tier.
    pub async fn contains_pks_at(
        &self,
        pks: &RecordBatch,
        watermarks: Option<&HashMap<Uuid, FreshTierWatermark>>,
    ) -> Result<Vec<bool>> {
        let sources = self.build_collector().collect()?;
        let memberships = super::block_list::fresh_tier_block_list(
            &sources,
            self.session.as_ref(),
            self.flushed_cache.as_ref(),
            watermarks,
        )
        .await?;
        let pk_indices = super::exec::resolve_pk_indices(pks, &self.pk_columns)
            .map_err(|e| Error::invalid_input(e.to_string()))?;
        // One key per row, in the index key space (typed value, or encoded
        // `Binary` tuple for a composite PK).
        let keys: Vec<ScalarValue> = (0..pks.num_rows())
            .map(|row| {
                let values: Vec<ScalarValue> = pk_indices
                    .iter()
                    .map(|&col| ScalarValue::try_from_array(pks.column(col), row))
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| Error::invalid_input(e.to_string()))?;
                super::block_list::on_disk_pk_key(&values)
            })
            .collect::<Result<_>>()?;

        // A row is contained if any generation contains its key. Probe each
        // generation once (batched), narrowing to still-unfound rows.
        let mut contained = vec![false; keys.len()];
        let mut live: Vec<usize> = (0..keys.len()).collect();
        for membership in &memberships {
            if live.is_empty() {
                break;
            }
            let live_keys: Vec<ScalarValue> = live.iter().map(|&i| keys[i].clone()).collect();
            let mask = membership.contains_keys(&live_keys).await?;
            let mut next_live = Vec::with_capacity(live.len());
            for (pos, &row) in live.iter().enumerate() {
                if mask[pos] {
                    contained[row] = true;
                } else {
                    next_live.push(row);
                }
            }
            live = next_live;
        }
        Ok(contained)
    }

    /// Build the data source collector.
    fn build_collector(&self) -> LsmDataSourceCollector {
        let mut collector = match &self.base {
            BaseSource::Table(dataset) => {
                LsmDataSourceCollector::new(dataset.clone(), self.shard_snapshots.clone())
            }
            BaseSource::PathOnly(path) => LsmDataSourceCollector::without_base_table(
                path.clone(),
                self.shard_snapshots.clone(),
            ),
        };

        for (shard_id, mems) in &self.in_memory_memtables {
            collector = collector.with_in_memory_memtables(*shard_id, mems.clone());
        }

        collector
    }
}

impl std::fmt::Debug for LsmScanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (label, value) = match &self.base {
            BaseSource::Table(dataset) => ("base_table", dataset.uri().to_string()),
            BaseSource::PathOnly(path) => ("base_path", path.clone()),
        };
        f.debug_struct("LsmScanner")
            .field(label, &value)
            .field("num_shards", &self.shard_snapshots.len())
            .field(
                "num_in_memory_memtables",
                &self
                    .in_memory_memtables
                    .values()
                    .map(|m| 1 + m.frozen.len())
                    .sum::<usize>(),
            )
            .field("projection", &self.projection)
            .field("limit", &self.limit)
            .field("offset", &self.offset)
            .field("pk_columns", &self.pk_columns)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsm_scanner_builder() {
        // Test that the builder pattern compiles and works
        // Full integration tests would require a real dataset

        let pk_columns = ["id".to_string()];
        let shard_snapshots: Vec<ShardSnapshot> = vec![];

        // We can't easily create an Arc<Dataset> without I/O,
        // so just test the type construction
        assert_eq!(pk_columns.len(), 1);
        assert!(shard_snapshots.is_empty());
    }

    #[test]
    fn test_shard_snapshot_construction() {
        use super::super::data_source::ShardSnapshot;

        let shard_id = Uuid::new_v4();
        let snapshot = ShardSnapshot::new(shard_id)
            .with_spec_id(1)
            .with_current_generation(5)
            .with_flushed_generation(1, "path/gen_1".to_string())
            .with_flushed_generation(2, "path/gen_2".to_string());

        assert_eq!(snapshot.shard_id, shard_id);
        assert_eq!(snapshot.spec_id, 1);
        assert_eq!(snapshot.current_generation, 5);
        assert_eq!(snapshot.flushed_generations.len(), 2);
    }

    #[test]
    fn test_in_memory_memtable_ref() {
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let index_store = Arc::new(IndexStore::new());
        let schema = Arc::new(arrow_schema::Schema::empty());

        let memtable_ref = InMemoryMemTableRef {
            batch_store,
            index_store,
            schema,
            generation: 10,
        };

        assert_eq!(memtable_ref.generation, 10);
    }

    /// Single-column `id: Int32` schema used by the PK-membership tests.
    fn pk_schema() -> SchemaRef {
        use arrow_schema::{DataType, Field, Schema};
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
    }

    /// A `RecordBatch` of `id` values against [`pk_schema`].
    fn id_pk_batch(ids: &[i32]) -> RecordBatch {
        use arrow_array::Int32Array;
        RecordBatch::try_new(pk_schema(), vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
    }

    /// An active/frozen memtable holding `ids` at `generation`, with a single
    /// batch and a maintained primary-key index on `id`.
    fn mk_pk_memtable(ids: &[i32], generation: u64) -> InMemoryMemTableRef {
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        let store = BatchStore::with_capacity(8);
        let mut index = IndexStore::new();
        index.enable_pk_index(&[("id".to_string(), 0)]);
        let b = id_pk_batch(ids);
        let (bp, off, _) = store.append(b.clone()).unwrap();
        index.insert_with_batch_position(&b, off, Some(bp)).unwrap();
        InMemoryMemTableRef {
            batch_store: Arc::new(store),
            index_store: Arc::new(index),
            schema: pk_schema(),
            generation,
        }
    }

    #[tokio::test]
    async fn contains_pks_reports_fresh_tier_membership() {
        // Fresh-tier only: active gen 2 (pk=1,2) + frozen gen 1 (pk=3).
        let shard = Uuid::new_v4();
        let scanner = LsmScanner::without_base_table(
            pk_schema(),
            "memory://t",
            vec![],
            vec!["id".to_string()],
        )
        .with_in_memory_memtables(
            shard,
            InMemoryMemTables {
                active: mk_pk_memtable(&[1, 2], 2),
                frozen: vec![mk_pk_memtable(&[3], 1)],
            },
        );

        // pk=1 (active), pk=4 (absent), pk=3 (frozen).
        let result = scanner
            .contains_pks(&id_pk_batch(&[1, 4, 3]))
            .await
            .unwrap();
        assert_eq!(result, vec![true, false, true]);
    }

    /// `contains_pks_at` probes each generation once over the still-unfound
    /// rows, so a multi-PK batch spanning several generations resolves to the
    /// right per-row mask — and a watermark bounds which generations count.
    #[tokio::test]
    async fn contains_pks_at_batched_probe_respects_watermark() {
        use crate::dataset::mem_wal::scanner::data_source::FreshTierWatermark;

        // active gen 2 (pk=1,2) + frozen gen 1 (pk=3,4).
        let shard = Uuid::new_v4();
        let scanner = LsmScanner::without_base_table(
            pk_schema(),
            "memory://t",
            vec![],
            vec!["id".to_string()],
        )
        .with_in_memory_memtables(
            shard,
            InMemoryMemTables {
                active: mk_pk_memtable(&[1, 2], 2),
                frozen: vec![mk_pk_memtable(&[3, 4], 1)],
            },
        );

        // Duplicate and out-of-order keys exercise the live-row narrowing: each
        // generation only re-probes the rows earlier generations didn't claim.
        let probe = id_pk_batch(&[4, 1, 9, 3, 2, 1]);

        // watermark=None → live tier: every PK present in either generation.
        let live = scanner.contains_pks_at(&probe, None).await.unwrap();
        assert_eq!(live, vec![true, true, false, true, true, true]);

        // watermark at gen 1 → active gen 2 rolled in after the snapshot and is
        // excluded; only the frozen gen 1 keys (3,4) remain members.
        let watermarks: HashMap<Uuid, FreshTierWatermark> = [(
            shard,
            FreshTierWatermark {
                active_generation: 1,
                active_batch_count: u64::MAX,
            },
        )]
        .into_iter()
        .collect();
        let bounded = scanner
            .contains_pks_at(&probe, Some(&watermarks))
            .await
            .unwrap();
        assert_eq!(bounded, vec![true, false, false, true, false, false]);
    }

    /// One active memtable with a maintained BTree on `id`, all rows visible.
    fn mk_indexed_memtable(schema: &SchemaRef, ids: &[i32], names: &[&str]) -> InMemoryMemTableRef {
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use arrow_array::{Int32Array, StringArray};

        let store = BatchStore::with_capacity(8);
        let mut index = IndexStore::new();
        index.add_btree("id_idx".to_string(), 0, "id".to_string());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
            ],
        )
        .unwrap();
        let (idx, row_offset, _) = store.append(batch.clone()).unwrap();
        index
            .insert_with_batch_position(&batch, row_offset, Some(idx))
            .unwrap();
        InMemoryMemTableRef {
            batch_store: Arc::new(store),
            index_store: Arc::new(index),
            schema: schema.clone(),
            generation: 1,
        }
    }

    #[tokio::test]
    async fn point_lookup_filter_routes_to_fast_path() {
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::physical_plan::displayable;
        use datafusion::prelude::{SessionContext, col, lit};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let memtable = mk_indexed_memtable(&schema, &[1, 2, 3, 4, 5], &["a", "b", "c", "d", "e"]);
        let shard = Uuid::new_v4();
        let scanner = || {
            LsmScanner::without_base_table(
                schema.clone(),
                "memory://t",
                vec![],
                vec!["id".to_string()],
            )
            .with_in_memory_memtables(
                shard,
                InMemoryMemTables {
                    active: memtable.clone(),
                    frozen: vec![],
                },
            )
        };
        let ctx = SessionContext::new();
        let count = |plan: Arc<dyn ExecutionPlan>| {
            let ctx = ctx.clone();
            async move {
                let rows: Vec<RecordBatch> = plan
                    .execute(0, ctx.task_ctx())
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();
                rows.iter().map(|b| b.num_rows()).sum::<usize>()
            }
        };

        // `id = 2` routes to the direct point-lookup node (OneShotStream), not the
        // union/dedup scan, and returns the one matching row.
        let plan = scanner()
            .filter_expr(col("id").eq(lit(2i32)))
            .create_plan()
            .await
            .unwrap();
        let disp = format!("{}", displayable(plan.as_ref()).indent(true));
        assert!(disp.contains("OneShotStream"), "pk=lit must route: {disp}");
        assert!(
            !disp.contains("Union"),
            "must not use the union path: {disp}"
        );
        assert_eq!(count(plan).await, 1);

        // `id IN (1, 3)` routes and returns both rows.
        let plan = scanner()
            .filter_expr(col("id").in_list(vec![lit(1i32), lit(3i32)], false))
            .create_plan()
            .await
            .unwrap();
        assert!(
            format!("{}", displayable(plan.as_ref()).indent(true)).contains("OneShotStream"),
            "pk IN (..) must route"
        );
        assert_eq!(count(plan).await, 2);

        // A range filter is NOT a point lookup → falls through to the scan path.
        let plan = scanner()
            .filter_expr(col("id").gt(lit(2i32)))
            .create_plan()
            .await
            .unwrap();
        assert!(
            !format!("{}", displayable(plan.as_ref()).indent(true)).contains("OneShotStream"),
            "range filter must not route to the point-lookup node"
        );
        assert_eq!(count(plan).await, 3); // 3,4,5
    }
}
