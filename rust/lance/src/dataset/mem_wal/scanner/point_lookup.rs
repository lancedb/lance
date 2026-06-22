// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Point lookup planner for LSM scanner.
//!
//! Provides efficient primary key-based point lookups across LSM levels.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use arrow_schema::{SchemaRef, SortOptions};
use datafusion::common::ScalarValue;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::prelude::{Expr, SessionContext};
use futures::TryStreamExt;
use lance_core::utils::bloomfilter::sbbf::Sbbf;
use lance_core::{Result, is_system_column};
use lance_datafusion::exec::OneShotExec;
use tracing::instrument;

use crate::dataset::mem_wal::index::IndexStore;
use crate::dataset::mem_wal::memtable::batch_store::BatchStore;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{BloomFilterGuardExec, CoalesceFirstExec, compute_pk_hash_from_scalars};
use super::flushed_cache::{DatasetCache, GenerationWarmer, open_flushed_dataset};
use super::projection::{
    build_scanner_projection, canonical_output_schema, null_columns, project_to_canonical,
    wants_row_address, wants_row_id,
};
use crate::session::Session;

/// Plans point lookup queries over LSM data.
///
/// Point lookups are optimized for primary key-based queries where we expect
/// to find at most one row. The query plan uses:
///
/// 1. **Bloom filter guards**: Skip generations that definitely don't contain the key
/// 2. **Short-circuit evaluation**: Stop after finding the first match
/// 3. **Newest-first ordering**: Check newer generations before older ones
///
/// # Query Plan Structure
///
/// Since data is stored in reverse order (newest first), we use `GlobalLimitExec`
/// with limit=1 to take the first (most recent) matching row.
///
/// ```text
/// CoalesceFirstExec: return_first_non_null
///   BloomFilterGuardExec: gen=3
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: memtable_gen_3
///   BloomFilterGuardExec: gen=2
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: flushed_gen_2
///   BloomFilterGuardExec: gen=1
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: flushed_gen_1
///   GlobalLimitExec: limit=1
///     FilterExec: pk = target
///       ScanExec: base_table
/// ```
///
/// The base table doesn't use a bloom filter guard because:
/// - It's the fallback when no memtable has the key
/// - Bloom filters for the base table would be too large
pub struct LsmPointLookupPlanner {
    /// Data source collector.
    collector: LsmDataSourceCollector,
    /// Primary key column names.
    pk_columns: Vec<String>,
    /// Schema of the base table.
    base_schema: SchemaRef,
    /// Bloom filters for each memtable generation.
    /// Map: generation -> bloom filter
    bloom_filters: std::collections::HashMap<u64, Arc<Sbbf>>,
    /// Session threaded into flushed-generation opens (shared caches).
    session: Option<Arc<Session>>,
    /// Cache of opened flushed-generation datasets.
    flushed_cache: Option<Arc<dyn DatasetCache>>,
    /// Optional warmer fired on first open of a flushed generation.
    warmer: Option<Arc<dyn GenerationWarmer>>,
    /// Precomputed canonical output schema for the no-projection case, so the
    /// hot `lookup(.., None)` path clones an `Arc` instead of rebuilding the
    /// schema on every call.
    none_target: SchemaRef,
    /// Shared DataFusion task context for plan execution. Built once and reused
    /// across lookups: `SessionContext::new()` per lookup is a real fixed cost
    /// on the plan fallback path (the part of point-lookup latency that doesn't
    /// scale with generation count).
    task_ctx: Arc<TaskContext>,
}

impl LsmPointLookupPlanner {
    /// Create a new planner.
    ///
    /// # Arguments
    ///
    /// * `collector` - Data source collector
    /// * `pk_columns` - Primary key column names
    /// * `base_schema` - Schema of the base table
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
    ) -> Self {
        let none_target = canonical_output_schema(None, &base_schema, &pk_columns, false);
        Self {
            collector,
            pk_columns,
            base_schema,
            bloom_filters: std::collections::HashMap::new(),
            session: None,
            flushed_cache: None,
            warmer: None,
            none_target,
            task_ctx: SessionContext::new().task_ctx(),
        }
    }

    /// Thread a session into flushed-generation opens so the first open
    /// populates the shared index / file-metadata caches.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Inject a cache of opened flushed-generation datasets, making repeated
    /// lookups against the same generation a pure `Arc::clone`. Populate it up
    /// front during scan setup via
    /// [`DatasetMemWalExt::prewarm_mem_wal`](crate::dataset::mem_wal::DatasetMemWalExt::prewarm_mem_wal)
    /// so the first gen-key lookup does not pay the dataset open.
    pub fn with_flushed_cache(mut self, cache: Arc<dyn DatasetCache>) -> Self {
        self.flushed_cache = Some(cache);
        self
    }

    /// Inject the warmer fired on first open of a flushed generation.
    pub fn with_warmer(mut self, warmer: Arc<dyn GenerationWarmer>) -> Self {
        self.warmer = Some(warmer);
        self
    }

    /// Add a bloom filter for a generation.
    ///
    /// Bloom filters are optional but improve performance by skipping
    /// generations that definitely don't contain the target key.
    pub fn with_bloom_filter(mut self, generation: u64, bloom_filter: Arc<Sbbf>) -> Self {
        self.bloom_filters.insert(generation, bloom_filter);
        self
    }

    /// Add multiple bloom filters.
    pub fn with_bloom_filters(
        mut self,
        bloom_filters: impl IntoIterator<Item = (u64, Arc<Sbbf>)>,
    ) -> Self {
        self.bloom_filters.extend(bloom_filters);
        self
    }

    /// Create a point lookup plan for the given primary key values.
    ///
    /// # Arguments
    ///
    /// * `pk_values` - Primary key values to look up (one value per pk column)
    /// * `projection` - Columns to include in output (None = all columns)
    ///
    /// # Returns
    ///
    /// An execution plan that returns at most one row - the newest version
    /// of the row with the given primary key.
    #[instrument(name = "lsm_point_lookup", level = "debug", skip_all, fields(pk_column_count = self.pk_columns.len()))]
    pub async fn plan_lookup(
        &self,
        pk_values: &[ScalarValue],
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if pk_values.len() != self.pk_columns.len() {
            return Err(lance_core::Error::invalid_input(format!(
                "Expected {} primary key values, got {}",
                self.pk_columns.len(),
                pk_values.len()
            )));
        }

        let pk_hash = compute_pk_hash_from_scalars(pk_values);
        let filter_expr = self.build_pk_filter_expr(pk_values)?;
        let sources = self.collector.collect()?;

        if sources.is_empty() {
            return self.empty_plan(projection);
        }

        // Sort by generation DESC (newest first)
        let mut sources: Vec<_> = sources.into_iter().collect();
        sources.sort_by_key(|b| std::cmp::Reverse(b.generation()));

        let mut source_plans = Vec::new();

        for source in sources {
            let generation = source.generation().as_u64();

            let scan = self
                .build_source_scan(&source, projection, &filter_expr)
                .await?;

            // Data is stored in reverse order, so first match is newest
            let limited: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(scan, 0, Some(1)));

            let guarded_plan: Arc<dyn ExecutionPlan> =
                if let Some(bf) = self.bloom_filters.get(&generation) {
                    Arc::new(BloomFilterGuardExec::new(
                        limited,
                        bf.clone(),
                        pk_hash,
                        generation,
                    ))
                } else {
                    limited
                };

            source_plans.push(guarded_plan);
        }

        let plan: Arc<dyn ExecutionPlan> = if source_plans.len() == 1 {
            source_plans.remove(0)
        } else {
            Arc::new(CoalesceFirstExec::new(source_plans))
        };

        Ok(plan)
    }

    /// Resolve a single-row point lookup, returning the newest matching row (a
    /// 1-row batch with the canonical output schema) or `None`.
    ///
    /// For a single-column primary key this probes the in-memory memtables'
    /// BTree index directly — no DataFusion plan — newest generation first, and
    /// returns on the first hit. Only when the lookup must consult an on-disk
    /// source (a flushed generation or the base table), a memtable lacks a
    /// BTree on the key, the key is multi-column, or the projection requests
    /// system columns does it fall back to [`Self::plan_lookup`]. The result is
    /// identical to executing `plan_lookup` and taking the first row; the fast
    /// path just skips the per-lookup plan/stream construction that dominates
    /// point-lookup latency.
    #[instrument(name = "lsm_lookup", level = "debug", skip_all)]
    pub async fn lookup(
        &self,
        pk_values: &[ScalarValue],
        projection: Option<&[String]>,
    ) -> Result<Option<RecordBatch>> {
        // Fast path: exactly one key value (which must match the single PK
        // column), the key's scalar type exactly matches the PK column's Arrow
        // type, and no system columns in the output. The length check is first
        // so `pk_values[0]` is only indexed once it is known to exist (an empty
        // slice falls through to the plan path, which returns a clean
        // `invalid_input` error rather than panicking). The exact-type
        // requirement avoids the `OrderableScalarValue` panic on comparing
        // mismatched variants — the plan path coerces, so a coercible-but-
        // different literal (e.g. `Int64` for an `Int32` PK) falls back.
        let fast_eligible = pk_values.len() == 1
            && self.pk_columns.len() == 1
            && self
                .base_schema
                .field_with_name(&self.pk_columns[0])
                .ok()
                .map(|f| f.data_type() == &pk_values[0].data_type())
                .unwrap_or(false);
        if fast_eligible {
            // Borrow the cached schema for the common `None` case (no `Arc`
            // clone — the clone would contend on a shared refcount under
            // concurrency); only an explicit projection builds a fresh schema.
            let projected;
            let target: &SchemaRef = match projection {
                None => &self.none_target,
                Some(_) => {
                    projected = canonical_output_schema(
                        projection,
                        &self.base_schema,
                        &self.pk_columns,
                        false,
                    );
                    &projected
                }
            };
            if !target.fields().iter().any(|f| is_system_column(f.name())) {
                // Probe in-memory memtables newest-first *by reference* (no
                // source `Arc` clones / allocation in the single-memtable case),
                // so concurrent readers don't contend on source refcounts.
                let outcome = self.collector.find_in_memory_newest_first(
                    |m| -> Result<Option<FastOutcome>> {
                        match probe_memtable(
                            &m.batch_store,
                            &m.index_store,
                            &self.pk_columns[0],
                            &pk_values[0],
                            target,
                        )? {
                            Probe::Hit(batch) => Ok(Some(FastOutcome::Hit(batch))),
                            Probe::Miss => Ok(None),
                            Probe::NoIndex => Ok(Some(FastOutcome::NeedsFallback)),
                        }
                    },
                )?;
                match outcome {
                    Some(FastOutcome::Hit(batch)) => return Ok(Some(batch)),
                    Some(FastOutcome::NeedsFallback) => { /* fall through to plan */ }
                    None => {
                        // Every in-memory memtable missed. If there is no
                        // on-disk source, the key does not exist; otherwise the
                        // plan path consults the base table / flushed gens.
                        if !self.collector.has_on_disk_sources() {
                            return Ok(None);
                        }
                    }
                }
            }
        }
        self.lookup_via_plan(pk_values, projection).await
    }

    /// Fallback: build and execute the DataFusion plan, returning its first row.
    async fn lookup_via_plan(
        &self,
        pk_values: &[ScalarValue],
        projection: Option<&[String]>,
    ) -> Result<Option<RecordBatch>> {
        let plan = self.plan_lookup(pk_values, projection).await?;
        let batches: Vec<RecordBatch> = plan
            .execute(0, self.task_ctx.clone())?
            .try_collect()
            .await?;
        for batch in batches {
            if batch.num_rows() > 0 {
                return Ok(Some(batch.slice(0, 1)));
            }
        }
        Ok(None)
    }

    /// Resolve many single-column keys in one pass, returning the found rows
    /// (newest visible per key) as a single `RecordBatch` in the canonical
    /// output schema. Missing keys are omitted; row order is not guaranteed to
    /// match the input (a set result, like the scan path). Amortizes per-call
    /// overhead and gathers rows columnar (one vectorized `take` per source
    /// batch). Equivalent to N× [`Self::lookup`], minus the per-key plan/stream.
    #[instrument(name = "lsm_lookup_many", level = "debug", skip_all, fields(n = keys.len()))]
    pub async fn lookup_many(
        &self,
        keys: &[ScalarValue],
        projection: Option<&[String]>,
    ) -> Result<RecordBatch> {
        let target = match projection {
            None => self.none_target.clone(),
            Some(_) => {
                canonical_output_schema(projection, &self.base_schema, &self.pk_columns, false)
            }
        };
        if keys.is_empty() {
            return Ok(RecordBatch::new_empty(target));
        }
        // One key: the batch grouping (refs vec + hash map + pending) has
        // nothing to amortize, so it's pure overhead — delegate to the cheaper
        // single-lookup path. Keeps a one-element `lookup_many` (e.g. a routed
        // `pk IN (x)`) as fast as `lookup`.
        if keys.len() == 1 {
            return Ok(self
                .lookup(keys, projection)
                .await?
                .unwrap_or_else(|| RecordBatch::new_empty(target)));
        }

        // Fast path: single pk column, every key matches the pk Arrow type, no
        // system columns in the output. Otherwise the per-key path (correct for
        // multi-column keys, coercible types, system-column projections).
        let pk_type = self
            .pk_columns
            .first()
            .and_then(|c| self.base_schema.field_with_name(c).ok())
            .map(|f| f.data_type().clone());
        let fast_eligible = self.pk_columns.len() == 1
            && !target.fields().iter().any(|f| is_system_column(f.name()))
            && pk_type
                .as_ref()
                .map(|t| keys.iter().all(|k| &k.data_type() == t))
                .unwrap_or(false);
        if !fast_eligible {
            return self
                .lookup_many_via_per_key(keys, projection, &target)
                .await;
        }

        let pk_col = &self.pk_columns[0];
        let refs = self.collector.in_memory_refs_newest_first();
        // Hits grouped by (memtable index, batch index) so each source batch is
        // gathered with a single `take`.
        let mut hits: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        let mut pending: Vec<ScalarValue> = Vec::new();
        for key in keys {
            let mut resolved = false;
            for (ri, m) in refs.iter().enumerate() {
                match probe_position(&m.batch_store, &m.index_store, pk_col, key)? {
                    ProbePos::Found { batch_idx, row } => {
                        hits.entry((ri, batch_idx)).or_default().push(row as u32);
                        resolved = true;
                        break;
                    }
                    ProbePos::Miss => continue,
                    ProbePos::NoIndex => {
                        // A memtable without the pk BTree can't be batch-probed;
                        // fall back to the fully-correct per-key path.
                        return self
                            .lookup_many_via_per_key(keys, projection, &target)
                            .await;
                    }
                }
            }
            if !resolved {
                pending.push(key.clone());
            }
        }

        let mut out: Vec<RecordBatch> = Vec::with_capacity(hits.len() + 1);
        for ((ri, batch_idx), rows) in hits {
            out.push(gather_rows(
                &refs[ri].batch_store,
                batch_idx,
                &rows,
                &target,
            )?);
        }
        // Keys absent from every in-memory memtable may live on disk; resolve
        // those via the plan path. (All-in-memory hit case: `pending` is empty.)
        if !pending.is_empty() && self.collector.has_on_disk_sources() {
            out.push(
                self.lookup_many_via_per_key(&pending, projection, &target)
                    .await?,
            );
        }

        match out.len() {
            0 => Ok(RecordBatch::new_empty(target)),
            1 => Ok(out.pop().unwrap()),
            _ => Ok(arrow_select::concat::concat_batches(&target, &out)?),
        }
    }

    /// Correctness fallback for [`Self::lookup_many`]: resolve each key with
    /// [`Self::lookup`] and concatenate.
    async fn lookup_many_via_per_key(
        &self,
        keys: &[ScalarValue],
        projection: Option<&[String]>,
        target: &SchemaRef,
    ) -> Result<RecordBatch> {
        let mut out: Vec<RecordBatch> = Vec::new();
        for key in keys {
            if let Some(b) = self.lookup(std::slice::from_ref(key), projection).await? {
                out.push(b);
            }
        }
        match out.len() {
            0 => Ok(RecordBatch::new_empty(target.clone())),
            1 => Ok(out.pop().unwrap()),
            _ => Ok(arrow_select::concat::concat_batches(target, &out)?),
        }
    }

    /// Build a composable one-shot `ExecutionPlan` that yields the point-lookup
    /// result for `keys`, so the LSM scanner can place limit / projection / etc.
    /// on top and use the fast path inside general query execution. A single
    /// key uses [`Self::lookup`]; multiple keys use [`Self::lookup_many`].
    pub async fn plan_point_lookup(
        &self,
        keys: &[ScalarValue],
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let batch = if keys.len() == 1 {
            match self.lookup(keys, projection).await? {
                Some(b) => b,
                None => RecordBatch::new_empty(canonical_output_schema(
                    projection,
                    &self.base_schema,
                    &self.pk_columns,
                    false,
                )),
            }
        } else {
            self.lookup_many(keys, projection).await?
        };
        let schema = batch.schema();
        let stream = futures::stream::once(async move { Ok(batch) });
        let adapter = RecordBatchStreamAdapter::new(schema, stream);
        Ok(Arc::new(OneShotExec::new(Box::pin(adapter))))
    }

    /// Build the filter expression for primary key equality.
    fn build_pk_filter_expr(&self, pk_values: &[ScalarValue]) -> Result<Expr> {
        use datafusion::prelude::{col, lit};

        let mut expr: Option<Expr> = None;

        for (col_name, value) in self.pk_columns.iter().zip(pk_values.iter()) {
            let eq_expr = col(col_name.as_str()).eq(lit(value.clone()));

            expr = Some(match expr {
                Some(e) => e.and(eq_expr),
                None => eq_expr,
            });
        }

        expr.ok_or_else(|| lance_core::Error::invalid_input("No primary key columns specified"))
    }

    /// Build scan plan for a single data source.
    ///
    /// Output is projected to the canonical schema so user-requested system
    /// columns appear at the requested position — NULL where the source
    /// doesn't produce them or where per-source values aren't meaningful.
    async fn build_source_scan(
        &self,
        source: &LsmDataSource,
        projection: Option<&[String]>,
        filter: &Expr,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let cols = build_scanner_projection(projection, &self.base_schema, &self.pk_columns);
        let target =
            canonical_output_schema(projection, &self.base_schema, &self.pk_columns, false);
        let want_row_id = wants_row_id(projection);
        let want_row_addr = wants_row_address(projection);
        let scan: Arc<dyn ExecutionPlan> = match source {
            LsmDataSource::BaseTable { dataset } => {
                let mut scanner = dataset.scan();
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                // Only the base produces row IDs callers can use against the
                // dataset (e.g. `take_rows`); non-base arms NULL via canonical.
                if want_row_id {
                    scanner.with_row_id();
                }
                if want_row_addr {
                    scanner.with_row_address();
                }
                scanner.filter_expr(filter.clone());
                scanner.create_plan().await?
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let dataset = open_flushed_dataset(
                    path,
                    self.session.as_ref(),
                    self.flushed_cache.as_ref(),
                    self.warmer.as_ref(),
                )
                .await?;
                let mut scanner = dataset.scan();
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                scanner.filter_expr(filter.clone());
                scanner.create_plan().await?
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;

                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                scanner.filter_expr(filter.clone());
                // Expose `_rowid` (the BatchStore row offset, monotonic with
                // insert order) so we can pick the most recently inserted
                // duplicate below. Without this, a `FilterExec → LIMIT 1`
                // over insert-ordered scan would return the *oldest* of
                // multiple rows sharing the target primary key.
                scanner.with_row_id();
                let raw = scanner.create_plan().await?;
                // The filter already restricts to the exact PK value, so the
                // scan yields that key's insert history. Within the active
                // memtable larger `_rowid` = newer insert, so sorting `_rowid`
                // DESC and keeping the first row picks the newest version — one
                // row per (value-exact) PK.
                let rowid_idx = raw.schema().index_of(lance_core::ROW_ID)?;
                let ordering = LexOrdering::new(vec![PhysicalSortExpr {
                    expr: Arc::new(Column::new(lance_core::ROW_ID, rowid_idx)),
                    options: SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                }])
                .ok_or_else(|| {
                    lance_core::Error::internal("point-lookup: failed to build _rowid ordering")
                })?;
                let newest: Arc<dyn ExecutionPlan> =
                    Arc::new(SortExec::new(ordering, raw).with_fetch(Some(1)));
                // Per-source `_rowid` would collide with the base table's;
                // NULL it before canonicalization (the value is internal to
                // this arm). project_to_canonical drops it entirely when
                // the user didn't request `_rowid` in the projection.
                null_columns(newest, &[lance_core::ROW_ID])?
            }
        };
        project_to_canonical(scan, &target)
    }

    /// Create an empty execution plan with the canonical output schema.
    fn empty_plan(&self, projection: Option<&[String]>) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::empty::EmptyExec;

        let schema =
            canonical_output_schema(projection, &self.base_schema, &self.pk_columns, false);
        Ok(Arc::new(EmptyExec::new(schema)))
    }
}

/// Result of probing the in-memory memtables newest-first in `lookup()`.
enum FastOutcome {
    /// A visible row was found; here it is, projected.
    Hit(RecordBatch),
    /// A memtable could not be probed directly (no BTree on the key) — the
    /// caller must fall back to the plan path.
    NeedsFallback,
}

/// Outcome of a direct BTree probe against one in-memory memtable.
enum Probe {
    /// The key was found; here is the newest visible row, projected.
    Hit(RecordBatch),
    /// The key is not present in this memtable (but may be in an older source).
    Miss,
    /// This memtable has no BTree on the key column, so it cannot be probed
    /// directly — the caller must fall back to the plan path.
    NoIndex,
}

/// Where a key's newest visible row lives within one in-memory memtable.
enum ProbePos {
    /// Found at `(batch_idx, row_in_batch)` in the memtable's `BatchStore`.
    Found {
        batch_idx: usize,
        row: usize,
    },
    Miss,
    NoIndex,
}

/// Resolve the `(batch_idx, row)` of a key's newest *visible* row in one
/// in-memory memtable via a seek-and-stop on the ordered skiplist
/// (`BTreeMemIndex::get_newest_visible`), honoring the MVCC watermark. No
/// materialization.
fn probe_position(
    batch_store: &BatchStore,
    index_store: &IndexStore,
    pk_column: &str,
    pk_value: &ScalarValue,
) -> Result<ProbePos> {
    // Visible batches are the committed prefix [0, last_visible_idx]; each
    // `StoredBatch` carries its cumulative `row_offset`, so visibility and the
    // position→batch mapping are O(1)/O(log) with no per-probe allocation.
    let len = batch_store.len();
    if len == 0 {
        return Ok(ProbePos::Miss);
    }
    let last_visible_idx = index_store.max_visible_batch_position().min(len - 1);
    let last = batch_store.get(last_visible_idx).ok_or_else(|| {
        lance_core::Error::internal("point-lookup: visible batch index out of range")
    })?;
    let visible_end = last.row_offset + last.num_rows as u64; // exclusive
    if visible_end == 0 {
        return Ok(ProbePos::Miss);
    }
    let max_visible_row = visible_end - 1;

    // A single-column primary key always has a value-keyed BTree (reused or
    // auto-created — see `IndexStore::enable_pk_index`): collision-free, so one
    // seek yields the answer with no re-check. Absent only when the table has no
    // PK index, where the caller falls back to the plan path.
    let Some(btree) = index_store.get_btree_by_column(pk_column) else {
        return Ok(ProbePos::NoIndex);
    };
    let Some(pos) = btree.get_newest_visible(pk_value, max_visible_row) else {
        return Ok(ProbePos::Miss);
    };
    let (batch_idx, row) = resolve_position(batch_store, last_visible_idx, pos)?;
    Ok(ProbePos::Found { batch_idx, row })
}

/// Map a global row `position` to its `(batch_idx, row_in_batch)` by binary
/// searching the visible batch prefix on cumulative `row_offset` (batches are
/// appended in order).
fn resolve_position(
    batch_store: &BatchStore,
    last_visible_idx: usize,
    position: u64,
) -> Result<(usize, usize)> {
    let (mut lo, mut hi) = (0usize, last_visible_idx);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let off = batch_store.get(mid).map(|b| b.row_offset).ok_or_else(|| {
            lance_core::Error::internal("point-lookup: batch index out of range during search")
        })?;
        if off <= position {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let stored = batch_store
        .get(lo)
        .ok_or_else(|| lance_core::Error::internal("point-lookup: resolved batch missing"))?;
    Ok((lo, (position - stored.row_offset) as usize))
}

/// Gather `rows` from `batch_store`'s batch `batch_idx` into the `target`
/// schema. A single row is a zero-copy `slice` (the common point-lookup case);
/// multiple rows use one vectorized `take` per column.
fn gather_rows(
    batch_store: &BatchStore,
    batch_idx: usize,
    rows: &[u32],
    target: &SchemaRef,
) -> Result<RecordBatch> {
    let stored = batch_store
        .get(batch_idx)
        .ok_or_else(|| lance_core::Error::internal("point-lookup: gather batch missing"))?;
    let indices = (rows.len() > 1).then(|| arrow_array::UInt32Array::from(rows.to_vec()));
    // Borrow the stored schema once (no `Arc` clone): `schema()` clones the
    // shared schema `Arc`, and under concurrency that refcount cache line
    // ping-pongs across cores. `schema_ref()` borrows it.
    let stored_schema = stored.data.schema_ref();
    let cols: Vec<Arc<dyn Array>> = target
        .fields()
        .iter()
        .map(|f| {
            let idx = stored_schema.index_of(f.name()).map_err(|_| {
                lance_core::Error::invalid_input(format!(
                    "point-lookup projection column '{}' not found in memtable batch",
                    f.name()
                ))
            })?;
            let col = stored.data.column(idx);
            // Single row: zero-copy `slice` (the common point-lookup case, and
            // measurably faster than `take` — copying regressed single-thread
            // ~30% with no N-thread gain). Multiple rows: one vectorized `take`.
            match &indices {
                None => Ok(col.slice(rows[0] as usize, 1)),
                Some(idxs) => arrow_select::take::take(col.as_ref(), idxs, None)
                    .map_err(lance_core::Error::from),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(target.clone(), cols)?)
}

/// Probe one in-memory memtable for a single key and materialize the newest
/// visible row into `target`. Thin wrapper over [`probe_position`] +
/// [`gather_rows`] used by [`LsmPointLookupPlanner::lookup`].
fn probe_memtable(
    batch_store: &BatchStore,
    index_store: &IndexStore,
    pk_column: &str,
    pk_value: &ScalarValue,
    target: &SchemaRef,
) -> Result<Probe> {
    match probe_position(batch_store, index_store, pk_column, pk_value)? {
        ProbePos::NoIndex => Ok(Probe::NoIndex),
        ProbePos::Miss => Ok(Probe::Miss),
        ProbePos::Found { batch_idx, row } => Ok(Probe::Hit(gather_rows(
            batch_store,
            batch_idx,
            &[row as u32],
            target,
        )?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use datafusion::physical_plan::displayable;
    use std::collections::HashMap;
    use uuid::Uuid;

    use crate::dataset::mem_wal::scanner::data_source::ShardSnapshot;
    use crate::dataset::{Dataset, WriteParams};

    fn create_pk_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);

        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, ids: &[i32], name_prefix: &str) -> RecordBatch {
        let names: Vec<String> = ids
            .iter()
            .map(|id| format!("{}_{}", name_prefix, id))
            .collect();
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    async fn create_dataset(uri: &str, batches: Vec<RecordBatch>) -> Dataset {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_point_lookup_plan_structure() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create collector without memtables
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Verify plan structure
        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));

        // Should have GlobalLimitExec with limit=1 (data is stored in reverse order)
        assert!(
            plan_str.contains("GlobalLimitExec"),
            "Should have GlobalLimitExec in plan: {}",
            plan_str
        );
    }

    #[tokio::test]
    async fn test_point_lookup_with_memtables() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create shard snapshot
        let shard_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let gen1_batch = create_test_batch(&schema, &[2], "gen1"); // Update id=2
        create_dataset(&gen1_uri, vec![gen1_batch]).await;

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        // Create collector
        let collector = LsmDataSourceCollector::new(base_dataset, vec![shard_snapshot]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Verify plan structure - should have CoalesceFirstExec with multiple children
        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));

        assert!(
            plan_str.contains("CoalesceFirstExec") || plan_str.contains("GlobalLimitExec"),
            "Should have CoalesceFirstExec or GlobalLimitExec in plan: {}",
            plan_str
        );
    }

    #[tokio::test]
    async fn test_point_lookup_with_bloom_filter() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create collector
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        // Create a bloom filter for generation 1 (simulating a memtable)
        let mut bf = Sbbf::with_ndv_fpp(100, 0.01).unwrap();
        let pk_hash = compute_pk_hash_from_scalars(&[ScalarValue::Int32(Some(2))]);
        bf.insert_hash(pk_hash);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone())
            .with_bloom_filter(1, Arc::new(bf));

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Plan should be valid
        assert!(plan.schema().field_with_name("id").is_ok());
    }

    #[tokio::test]
    async fn test_pk_filter_expr() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[1], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let pk_values = vec![ScalarValue::Int32(Some(42))];
        let expr = planner.build_pk_filter_expr(&pk_values).unwrap();

        // Verify expression is an equality
        let expr_str = format!("{}", expr);
        assert!(
            expr_str.contains("id"),
            "Expression should contain column name"
        );
    }

    #[tokio::test]
    async fn test_point_lookup_without_base_table() {
        use futures::TryStreamExt;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // No base dataset is created. We still need a base URI so the collector
        // can resolve flushed-generation paths.
        let base_uri = format!("{}/base", base_path);

        // Create a flushed generation under {base_uri}/_mem_wal/{shard}/gen_1
        let shard_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let gen1_batch = create_test_batch(&schema, &[2, 3], "gen1");
        create_dataset(&gen1_uri, vec![gen1_batch]).await;

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![shard_snapshot]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        // id=3 lives in the flushed generation
        let pk_values = vec![ScalarValue::Int32(Some(3))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));
        assert!(
            !plan_str.contains("base/data"),
            "Plan must not scan base table, got: {}",
            plan_str
        );
        assert!(plan_str.contains("gen_1"));

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);

        // id=99 doesn't exist anywhere → empty
        let plan = planner
            .plan_lookup(&[ScalarValue::Int32(Some(99))], None)
            .await
            .unwrap();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_point_lookup_projection_with_system_columns() {
        // Regression: system columns in projection used to error in the
        // active-arm MemTableScanner or get silently dropped. Verify they're
        // surfaced at the requested position with the correct NULL/real mix.
        use futures::TryStreamExt;
        use lance_core::is_system_column;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        // User requests `_rowaddr` between `id` and `name`, plus `_rowoffset` at end.
        let projection = vec![
            "id".to_string(),
            "_rowaddr".to_string(),
            "name".to_string(),
            "_rowoffset".to_string(),
        ];
        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner
            .plan_lookup(&pk_values, Some(&projection))
            .await
            .expect("planner must accept system columns in projection");

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "expected exactly one matching row");

        let out_schema = batches[0].schema();
        let out_cols: Vec<String> = out_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            out_cols,
            vec![
                "id".to_string(),
                "_rowaddr".to_string(),
                "name".to_string(),
                "_rowoffset".to_string(),
            ],
            "system columns must appear at the user's requested position"
        );

        // Hit row is from base → `_rowaddr` is real. `_rowoffset` stays
        // NULL (no scanner produces it).
        // (Test 5 — empty-plan with system columns — lives in the next
        // test below.)
        let rowaddr = batches[0].column_by_name("_rowaddr").unwrap();
        assert!(
            !rowaddr.is_null(0),
            "_rowaddr from base should be populated, got: {:?}",
            rowaddr
        );
        let rowoffset = batches[0].column_by_name("_rowoffset").unwrap();
        assert!(is_system_column("_rowoffset"));
        assert!(
            rowoffset.is_null(0),
            "_rowoffset has no per-source flag, must be NULL across LSM, got: {:?}",
            rowoffset
        );
    }

    #[tokio::test]
    async fn test_point_lookup_empty_plan_with_system_columns() {
        // Test 5 (point_lookup slice): with no sources, the empty plan
        // must still expose user-requested system columns at the
        // requested position.
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let projection = vec![
            "id".to_string(),
            "_rowaddr".to_string(),
            "name".to_string(),
            "_rowid".to_string(),
        ];
        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner
            .plan_lookup(&pk_values, Some(&projection))
            .await
            .expect("empty plan must accept system columns in projection");

        let names: Vec<String> = plan
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            names,
            vec![
                "id".to_string(),
                "_rowaddr".to_string(),
                "name".to_string(),
                "_rowid".to_string(),
            ],
            "empty point-lookup plan must honor user column order including system columns"
        );
    }

    #[tokio::test]
    async fn test_point_lookup_active_memtable_returns_newest_duplicate() {
        // Regression: same primary key inserted twice into one active
        // memtable must return the *newest* row. The bug was that
        // `FilterExec → LIMIT 1` over an insert-ordered scan returned the
        // first (oldest) match. The plan-path active arm now sorts `_rowid`
        // DESC and keeps the first row (largest `_rowid` = newest insert).
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use futures::TryStreamExt;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        // BTree on the PK so that `max_visible_batch_position` advances as
        // we insert, otherwise the scanner sees no batches at all.
        index_store.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Two writes to pk=1, then an unrelated pk=2. The "new" row goes
        // *second* so its `_rowid` is larger.
        let b_old = create_test_batch(&schema, &[1], "old");
        let b_new = create_test_batch(&schema, &[1], "new");
        let b_other = create_test_batch(&schema, &[2], "two");
        let (bp_old, off_old, _) = batch_store.append(b_old.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_old, off_old, Some(bp_old))
            .unwrap();
        let (bp_new, off_new, _) = batch_store.append(b_new.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_new, off_new, Some(bp_new))
            .unwrap();
        let (bp_other, off_other, _) = batch_store.append(b_other.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_other, off_other, Some(bp_other))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = Uuid::new_v4();
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                shard_id,
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store,
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let plan = planner
            .plan_lookup(&[ScalarValue::Int32(Some(1))], None)
            .await
            .unwrap();
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "expected exactly one row for pk=1");
        let name_col = batches[0].column_by_name("name").unwrap();
        let name_arr = name_col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            name_arr.value(0),
            "new_1",
            "active-arm lookup must return the newer insert, not the oldest"
        );
    }

    #[tokio::test]
    async fn test_point_lookup_probes_auto_created_pk_btree() {
        // No user `add_btree` on the PK column — only `enable_pk_index`, which
        // auto-creates a BTree on the primary key (the production default). The
        // fast probe must resolve the newest visible version through that
        // collision-free BTree rather than falling back to the plan path.
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        // No `add_btree` — `enable_pk_index` auto-creates the PK BTree.
        index_store.enable_pk_index(&[("id".to_string(), 0)]);

        // pk=1 written twice (the newer second), plus an unrelated pk=2.
        let b_old = create_test_batch(&schema, &[1], "old");
        let b_new = create_test_batch(&schema, &[1], "new");
        let b_other = create_test_batch(&schema, &[2], "two");
        let (bp_old, off_old, _) = batch_store.append(b_old.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_old, off_old, Some(bp_old))
            .unwrap();
        let (bp_new, off_new, _) = batch_store.append(b_new.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_new, off_new, Some(bp_new))
            .unwrap();
        let (bp_other, off_other, _) = batch_store.append(b_other.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_other, off_other, Some(bp_other))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = Uuid::new_v4();
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                shard_id,
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store,
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        // `lookup` takes the fast probe path (single-column PK, no system cols).
        let hit = planner
            .lookup(&[ScalarValue::Int32(Some(1))], None)
            .await
            .unwrap()
            .expect("pk=1 must be found via the PK-position index probe");
        assert_eq!(hit.num_rows(), 1);
        let name = hit
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            name.value(0),
            "new_1",
            "probe must return the newest version"
        );

        // An absent key resolves to None (no on-disk sources to consult).
        assert!(
            planner
                .lookup(&[ScalarValue::Int32(Some(999))], None)
                .await
                .unwrap()
                .is_none(),
            "absent key must miss"
        );
    }

    #[tokio::test]
    async fn test_point_lookup_flushed_memtable_returns_newest_duplicate() {
        // Regression / invariant pin: when a flushed memtable contains two
        // rows for the same PK, the lookup must return the newer one. The
        // flushed dataset is reverse-written (newest at the smallest
        // physical position), so we simulate that here by writing the
        // dataset with the new row first. The point-lookup plan today
        // returns the first match (smallest `_rowid`) under reverse-write,
        // and remains so after this change.
        use futures::TryStreamExt;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();
        let base_uri = format!("{}/base", base_path);

        // Simulated reverse-write: newest insert lives at row 0.
        let shard_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let row_new = create_test_batch(&schema, &[1], "new");
        let row_old = create_test_batch(&schema, &[1], "old");
        create_dataset(&gen1_uri, vec![row_new, row_old]).await;

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![shard_snapshot]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let plan = planner
            .plan_lookup(&[ScalarValue::Int32(Some(1))], None)
            .await
            .unwrap();
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "expected exactly one row for pk=1");
        let name_col = batches[0].column_by_name("name").unwrap();
        let name_arr = name_col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            name_arr.value(0),
            "new_1",
            "flushed-arm lookup must return the row at the smallest _rowid (newest under reverse-write)"
        );
    }

    /// Build an in-memory active memtable ref from batches, with a BTree on
    /// `id` and the visibility watermark advanced so every row is visible.
    fn active_memtable_ref(
        schema: &Arc<ArrowSchema>,
        batches: &[RecordBatch],
        generation: u64,
    ) -> crate::dataset::mem_wal::scanner::collector::InMemoryMemTableRef {
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTableRef;
        let batch_store = Arc::new(BatchStore::with_capacity(64));
        let mut index_store = IndexStore::new();
        index_store.add_btree("id_idx".to_string(), 0, "id".to_string());
        for b in batches {
            let (idx, row_offset, _) = batch_store.append(b.clone()).unwrap();
            index_store
                .insert_with_batch_position(b, row_offset, Some(idx))
                .unwrap();
        }
        InMemoryMemTableRef {
            batch_store,
            index_store: Arc::new(index_store),
            schema: schema.clone(),
            generation,
        }
    }

    fn id_at(batch: &RecordBatch) -> i32 {
        batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(0)
    }

    fn name_at(batch: &RecordBatch) -> String {
        batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0)
            .to_string()
    }

    #[tokio::test]
    async fn test_lookup_fast_path_active_hit_and_absent() {
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTables;
        let schema = create_pk_schema();
        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        let active = active_memtable_ref(
            &schema,
            &[create_test_batch(&schema, &[10, 20, 30], "v")],
            1,
        );
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                Uuid::new_v4(),
                InMemoryMemTables {
                    active,
                    frozen: vec![],
                },
            );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        let row = planner
            .lookup(&[ScalarValue::Int32(Some(20))], None)
            .await
            .unwrap()
            .expect("hit");
        assert_eq!(row.num_rows(), 1);
        assert_eq!(id_at(&row), 20);

        // Absent key, no on-disk source → fast path proves non-existence.
        assert!(
            planner
                .lookup(&[ScalarValue::Int32(Some(99))], None)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_lookup_fast_path_newest_duplicate() {
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTables;
        let schema = create_pk_schema();
        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        // Same pk inserted twice; the second (larger position) is newest.
        let active = active_memtable_ref(
            &schema,
            &[
                create_test_batch(&schema, &[5], "old"),
                create_test_batch(&schema, &[5], "new"),
            ],
            1,
        );
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                Uuid::new_v4(),
                InMemoryMemTables {
                    active,
                    frozen: vec![],
                },
            );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let row = planner
            .lookup(&[ScalarValue::Int32(Some(5))], None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(name_at(&row), "new_5", "must return the newest insert");
    }

    #[tokio::test]
    async fn test_lookup_miss_falls_back_to_base() {
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTables;
        let schema = create_pk_schema();
        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        let base = Arc::new(
            create_dataset(
                &base_uri,
                vec![create_test_batch(&schema, &[1, 2, 3], "base")],
            )
            .await,
        );
        let active = active_memtable_ref(&schema, &[create_test_batch(&schema, &[99], "act")], 1);
        let collector = LsmDataSourceCollector::new(base, vec![]).with_in_memory_memtables(
            Uuid::new_v4(),
            InMemoryMemTables {
                active,
                frozen: vec![],
            },
        );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        // In active only → fast-path hit.
        let row = planner
            .lookup(&[ScalarValue::Int32(Some(99))], None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(id_at(&row), 99);

        // Only in base → active misses, falls back to the plan path.
        let row = planner
            .lookup(&[ScalarValue::Int32(Some(2))], None)
            .await
            .unwrap()
            .expect("base hit via fallback");
        assert_eq!(id_at(&row), 2);
        assert_eq!(name_at(&row), "base_2");

        // Nowhere → None (fallback plan over base finds nothing).
        assert!(
            planner
                .lookup(&[ScalarValue::Int32(Some(1000))], None)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_lookup_projection_regular_columns() {
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTables;
        let schema = create_pk_schema();
        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        let active = active_memtable_ref(
            &schema,
            &[create_test_batch(&schema, &[10, 20, 30], "v")],
            1,
        );
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                Uuid::new_v4(),
                InMemoryMemTables {
                    active,
                    frozen: vec![],
                },
            );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let row = planner
            .lookup(&[ScalarValue::Int32(Some(20))], Some(&["name".to_string()]))
            .await
            .unwrap()
            .unwrap();
        // The canonical point-lookup schema always includes the pk column, so
        // a `name` projection yields `[name, id]` — matching the plan path.
        let row_schema = row.schema();
        let names: Vec<&str> = row_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["name", "id"]);
        assert_eq!(name_at(&row), "v_20");
        assert_eq!(id_at(&row), 20);
    }

    #[tokio::test]
    async fn test_lookup_type_mismatch_falls_back_no_panic() {
        // PK is Int32; an Int64 literal must NOT take the direct BTree probe
        // (which could panic comparing mismatched OrderableScalarValue
        // variants) — it falls back to the coercing plan path.
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTables;
        let schema = create_pk_schema();
        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        let active = active_memtable_ref(
            &schema,
            &[create_test_batch(&schema, &[10, 20, 30], "v")],
            1,
        );
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                Uuid::new_v4(),
                InMemoryMemTables {
                    active,
                    frozen: vec![],
                },
            );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let row = planner
            .lookup(&[ScalarValue::Int64(Some(20))], None)
            .await
            .expect("must not panic on a coercible-but-different key type")
            .expect("plan path coerces Int64 → Int32 and finds id=20");
        assert_eq!(id_at(&row), 20);
    }

    #[tokio::test]
    async fn test_lookup_empty_pk_values_errors_not_panics() {
        // Regression: the fast-path eligibility check must not index
        // `pk_values[0]` before verifying the slice is non-empty. An empty
        // slice falls through to the plan path's length validation.
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTables;
        let schema = create_pk_schema();
        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        let active = active_memtable_ref(&schema, &[create_test_batch(&schema, &[1], "v")], 1);
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                Uuid::new_v4(),
                InMemoryMemTables {
                    active,
                    frozen: vec![],
                },
            );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let err = planner.lookup(&[], None).await;
        assert!(err.is_err(), "empty pk_values must error, not panic");
    }

    fn sorted_ids(batch: &RecordBatch) -> Vec<i32> {
        let arr = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let mut v: Vec<i32> = (0..arr.len()).map(|i| arr.value(i)).collect();
        v.sort_unstable();
        v
    }

    fn active_planner(batches: &[RecordBatch]) -> LsmPointLookupPlanner {
        use crate::dataset::mem_wal::scanner::collector::InMemoryMemTables;
        let schema = create_pk_schema();
        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        let active = active_memtable_ref(&schema, batches, 1);
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                Uuid::new_v4(),
                InMemoryMemTables {
                    active,
                    frozen: vec![],
                },
            );
        LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema)
    }

    #[tokio::test]
    async fn test_lookup_many_hits_and_misses() {
        let schema = create_pk_schema();
        let planner = active_planner(&[create_test_batch(&schema, &[10, 20, 30], "v")]);
        // Mix present + absent keys; absent omitted, order not guaranteed.
        let keys = [
            ScalarValue::Int32(Some(30)),
            ScalarValue::Int32(Some(10)),
            ScalarValue::Int32(Some(999)),
            ScalarValue::Int32(Some(20)),
        ];
        let batch = planner.lookup_many(&keys, None).await.unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(sorted_ids(&batch), vec![10, 20, 30]);

        // Empty input → empty batch with the canonical schema.
        let empty = planner.lookup_many(&[], None).await.unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert!(empty.schema().field_with_name("id").is_ok());
    }

    #[tokio::test]
    async fn test_lookup_many_newest_duplicate() {
        let schema = create_pk_schema();
        // id=5 written twice; the batch get must return the newest ("new_5").
        let planner = active_planner(&[
            create_test_batch(&schema, &[5], "old"),
            create_test_batch(&schema, &[5, 7], "new"),
        ]);
        let batch = planner
            .lookup_many(
                &[ScalarValue::Int32(Some(5)), ScalarValue::Int32(Some(7))],
                None,
            )
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 2);
        let names = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let mut got: Vec<&str> = (0..names.len()).map(|i| names.value(i)).collect();
        got.sort_unstable();
        assert_eq!(got, vec!["new_5", "new_7"]);
    }

    #[tokio::test]
    async fn test_lookup_many_projection_and_equivalence_to_lookup() {
        let schema = create_pk_schema();
        let planner = active_planner(&[create_test_batch(&schema, &[1, 2, 3, 4], "v")]);
        let keys = [
            ScalarValue::Int32(Some(2)),
            ScalarValue::Int32(Some(4)),
            ScalarValue::Int32(Some(1)),
        ];
        // Projected batch get == set of single lookups, same schema.
        let proj = vec!["name".to_string()];
        let batch = planner.lookup_many(&keys, Some(&proj)).await.unwrap();
        let batch_schema = batch.schema();
        let names: Vec<&str> = batch_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["name", "id"]); // pk always appended
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(sorted_ids(&batch), vec![1, 2, 4]);
    }

    #[tokio::test]
    async fn test_plan_point_lookup_executes() {
        use futures::TryStreamExt;
        let schema = create_pk_schema();
        let planner = active_planner(&[create_test_batch(&schema, &[10, 20, 30], "v")]);
        let plan = planner
            .plan_point_lookup(
                &[ScalarValue::Int32(Some(10)), ScalarValue::Int32(Some(30))],
                None,
            )
            .await
            .unwrap();
        let ctx = datafusion::prelude::SessionContext::new();
        let batches: Vec<RecordBatch> = plan
            .execute(0, ctx.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }

    #[tokio::test]
    async fn test_lookup_against_from_configs_built_index() {
        // A point lookup against an index built the production way
        // (`IndexStore::from_configs`) resolves correctly via the seek-and-stop
        // skiplist probe.
        use crate::dataset::mem_wal::index::{BTreeIndexConfig, IndexStore, MemIndexConfig};
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};

        let schema = create_pk_schema();
        let batch = create_test_batch(&schema, &[10, 20, 30], "v");
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let index_store = IndexStore::from_configs(
            &[MemIndexConfig::BTree(BTreeIndexConfig {
                name: "id_idx".to_string(),
                field_id: 0,
                column: "id".to_string(),
            })],
            1000,
            100,
        )
        .unwrap();
        let (idx, row_offset, _) = batch_store.append(batch.clone()).unwrap();
        index_store
            .insert_with_batch_position(&batch, row_offset, Some(idx))
            .unwrap();

        let temp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp.path().to_str().unwrap());
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                Uuid::new_v4(),
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store: Arc::new(index_store),
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let row = planner
            .lookup(&[ScalarValue::Int32(Some(20))], None)
            .await
            .unwrap()
            .expect("range fallback must find the row");
        assert_eq!(id_at(&row), 20);
        assert!(
            planner
                .lookup(&[ScalarValue::Int32(Some(99))], None)
                .await
                .unwrap()
                .is_none(),
            "absent key must miss"
        );
    }
}
