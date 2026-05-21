// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Full-text search planner for LSM scanner.
//!
//! Builds an execution plan that scores an FTS query across the base
//! table, flushed memtable generations, and active/frozen-undrained
//! in-memory memtables, returning rows ordered by BM25 `_score` DESC.
//!
//! # Scoring modes
//!
//! - [`FtsScoringMode::Local`] — each source uses its own corpus
//!   statistics to score. Cross-source `_score` values are only
//!   approximately comparable, but the plan is single-pass and never
//!   coordinates stats across sources.
//! - [`FtsScoringMode::LocalWithGlobalRescore`] — each source returns
//!   top-K' candidates with the raw BM25 sufficient statistics
//!   (`doc_len`, per-term frequencies); the planner aggregates
//!   per-source `(N, sumdl, df_t)` into one global `MemBM25Scorer`,
//!   rescores every candidate with the global stats, and returns the
//!   pre-materialized top-k as a [`MemorySourceConfig`] exec.
//!
//! Staleness: per-source results are returned as-is. The same primary
//! key may appear from multiple sources if it was updated across
//! generations; the caller is responsible for dedup if they need it.
//! This is the user-chosen behavior captured in `DESIGN.md §3`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::UnionExec;
use lance_core::{Error, Result, is_system_column};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::document_tokenizer::DocType;
use lance_index::scalar::inverted::query::{
    FtsQuery as IndexFtsQuery, FtsSearchParams, Operator, Tokens, collect_query_tokens,
};
use lance_index::scalar::inverted::{InvertedIndex, InvertedIndexCandidate, MemBM25Scorer, Scorer};
use tracing::instrument;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::projection::project_to_canonical;
use crate::Dataset;
use crate::dataset::mem_wal::index::FtsCandidate;
use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// `_score` column name in FTS results — kept aligned with
/// `lance_index::scalar::inverted::SCORE_COL` so this module doesn't
/// require an import for one string constant.
pub const SCORE_COLUMN: &str = "_score";

/// Default candidate multiplier for `LocalWithGlobalRescore`.
///
/// Picked to match wjones127's draft on [discussion
/// #6789](https://github.com/lance-format/lance/discussions/6789): K' =
/// `rescore_factor * k`, floored at `max(k, 100)`. Subject to the
/// benchmark in `BENCH.md` — if the recall@k curve flattens earlier
/// we'll lower this.
pub const DEFAULT_RESCORE_FACTOR: u32 = 10;

/// Floor for K' to keep rescore reasonable when `k` is tiny
/// (e.g., `k = 1` shouldn't collapse to one candidate per source).
pub const MIN_RESCORE_CANDIDATES: usize = 100;

/// How per-source BM25 contributes to the final `_score`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtsScoringMode {
    /// Each source scores with its own corpus stats.
    ///
    /// Cheapest mode: a single round-trip and no coordinator state.
    /// `_score` values are NOT strictly comparable across sources,
    /// but ranking within each source is correct and the union is
    /// merged by `_score` DESC.
    Local,
    /// Each source returns top-K' candidates with raw
    /// `(doc_len, term_freqs)`; a coordinator rescores them with
    /// globally-aggregated BM25 statistics. K' = `rescore_factor * k`
    /// floored at [`MIN_RESCORE_CANDIDATES`].
    LocalWithGlobalRescore { rescore_factor: u32 },
}

impl FtsScoringMode {
    /// Convenience constructor for `LocalWithGlobalRescore` with the
    /// project default rescore factor.
    pub fn local_with_global_rescore_default() -> Self {
        Self::LocalWithGlobalRescore {
            rescore_factor: DEFAULT_RESCORE_FACTOR,
        }
    }

    /// Effective K' for a user-supplied `k` (floored at the minimum).
    pub fn rescore_k_prime(&self, k: usize) -> usize {
        match self {
            Self::Local => k,
            Self::LocalWithGlobalRescore { rescore_factor } => (*rescore_factor as usize)
                .saturating_mul(k.max(1))
                .max(MIN_RESCORE_CANDIDATES)
                .max(k),
        }
    }
}

/// Plans FTS queries over LSM data.
pub struct LsmFtsSearchPlanner {
    collector: LsmDataSourceCollector,
    pk_columns: Vec<String>,
    base_schema: SchemaRef,
}

impl LsmFtsSearchPlanner {
    /// Create a new planner.
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
    ) -> Self {
        Self {
            collector,
            pk_columns,
            base_schema,
        }
    }

    /// Build the FTS execution plan.
    ///
    /// # Arguments
    ///
    /// * `column` — text column to search; must have an FTS index on
    ///   the base dataset, every flushed memtable dataset, and every
    ///   active/frozen `IndexStore`.
    /// * `query` — the FTS query (match / phrase / boolean / fuzzy).
    /// * `k` — global top-k to return.
    /// * `projection` — user columns to project. PK columns are
    ///   auto-included. `_score` is always appended.
    /// * `mode` — see [`FtsScoringMode`].
    #[instrument(
        name = "lsm_fts_search",
        level = "info",
        skip_all,
        fields(column = %column, k, mode = ?mode)
    )]
    pub async fn plan_search(
        &self,
        column: &str,
        query: FullTextSearchQuery,
        k: usize,
        projection: Option<&[String]>,
        mode: FtsScoringMode,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match mode {
            FtsScoringMode::Local => self.plan_local(column, query, k, projection).await,
            FtsScoringMode::LocalWithGlobalRescore { rescore_factor } => {
                self.plan_rescore(column, query, k, projection, rescore_factor)
                    .await
            }
        }
    }

    /// Single-node implementation of wjones127's `LocalWithGlobalRescore`
    /// mode (discussion #6789). Orchestrates synchronously:
    ///
    /// 1. Tokenize the query against the first available source's
    ///    tokenizer (we assume all sources share the same FTS params).
    /// 2. Open the InvertedIndex for each Lance source and gather
    ///    `(N_i, sumdl_i, df_t_i)` from every source.
    /// 3. Aggregate into a single global `MemBM25Scorer`.
    /// 4. Run each source's candidate search with LOCAL stats (so each
    ///    segment uses its own WAND pruning thresholds).
    /// 5. Rescore the union of candidates with the global scorer.
    /// 6. Take the top-k by rescored `_score`.
    /// 7. Materialize user columns (active arm reads BatchStore; Lance
    ///    arms `take_rows`), assemble the output RecordBatch, and
    ///    return it as a `MemorySourceConfig` exec.
    ///
    /// The output is pre-materialized rather than streaming because
    /// rescore needs every candidate from every source in scope before
    /// it can pick the global top-k — a buffered exec would be the same
    /// shape under the hood. For the bench-relevant single-node case
    /// this is a clear win on simplicity at no correctness cost.
    async fn plan_rescore(
        &self,
        column: &str,
        query: FullTextSearchQuery,
        k: usize,
        projection: Option<&[String]>,
        rescore_factor: u32,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let sources = self.collector.collect()?;
        let target_schema = self.canonical_fts_schema(projection);
        if sources.is_empty() || k == 0 {
            return self.empty_plan(&target_schema);
        }

        // Step 1: pull a tokenizer + tokenize the query text.
        let match_text = extract_match_text(&query)?;
        let mut tokenizer = self.resolve_tokenizer(&sources, column).await?;
        let tokens_obj = collect_query_tokens(&match_text, &mut tokenizer);
        let token_strs: Vec<String> = (0..tokens_obj.len())
            .map(|i| tokens_obj.get_token(i).to_owned())
            .collect();
        if token_strs.is_empty() {
            return self.empty_plan(&target_schema);
        }

        let k_prime = FtsScoringMode::LocalWithGlobalRescore { rescore_factor }.rescore_k_prime(k);

        // Step 2: resolve each source to a `SourceHandle` and gather its stats.
        let mut handles: Vec<SourceHandle> = Vec::with_capacity(sources.len());
        let mut total_tokens: u64 = 0;
        let mut num_docs: usize = 0;
        let mut df_map: HashMap<String, usize> =
            token_strs.iter().map(|t| (t.clone(), 0usize)).collect();
        for source in &sources {
            let handle = self.resolve_handle(source, column).await?;
            let (tt, nd, df_vec) = handle.stats_for_terms(&token_strs)?;
            total_tokens += tt;
            num_docs += nd;
            for (t, c) in token_strs.iter().zip(df_vec.into_iter()) {
                *df_map.get_mut(t).expect("df entry seeded above") += c;
            }
            handles.push(handle);
        }
        if num_docs == 0 {
            return self.empty_plan(&target_schema);
        }
        let global_scorer = MemBM25Scorer::new(total_tokens, num_docs, df_map);

        // Step 3: per-source candidate search with LOCAL pruning (no base_scorer).
        let tokens_arc = Arc::new(Tokens::new(token_strs.clone(), DocType::Text));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(k_prime)));
        let mut rescored: Vec<RescoredCandidate> = Vec::new();
        for (source_idx, handle) in handles.iter().enumerate() {
            let candidates = handle
                .candidate_search(&tokens_arc, &params, &token_strs)
                .await?;
            for c in candidates {
                // Step 4: rescore with global scorer in-place.
                let score = bm25_score(&global_scorer, &token_strs, &c.term_freqs, c.doc_len);
                rescored.push(RescoredCandidate {
                    source_idx,
                    row_id: c.row_id,
                    score,
                });
            }
        }

        // Step 5: pick global top-k.
        rescored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rescored.truncate(k);
        if rescored.is_empty() {
            return self.empty_plan(&target_schema);
        }

        // Step 6: materialize user columns per source, in the order
        // determined by `rescored`. Each candidate carries the source it
        // came from so we know which BatchStore / Dataset to take from.
        let final_batch = self
            .materialize_rescored(&handles, &rescored, projection, &target_schema)
            .await?;

        // Step 7: wrap pre-computed batch in MemorySourceConfig.
        let exec =
            MemorySourceConfig::try_new_exec(&[vec![final_batch]], target_schema.clone(), None)
                .map_err(|e| Error::internal(format!("MemorySourceConfig failed: {e}")))?;
        Ok(exec)
    }

    /// Acquire a tokenizer compatible with every source's FTS index.
    ///
    /// We assume FTS-indexed sources in an LSM hierarchy share their
    /// `InvertedIndexParams` (otherwise their indexes wouldn't be
    /// merge-compatible). Pulls the tokenizer from the first source
    /// that has one; any later mismatch is the caller's bug.
    async fn resolve_tokenizer(
        &self,
        sources: &[LsmDataSource],
        column: &str,
    ) -> Result<Box<dyn lance_index::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer>>
    {
        for source in sources {
            match source {
                LsmDataSource::ActiveMemTable { index_store, .. } => {
                    if let Some(idx) = index_store.get_fts_by_column(column) {
                        return idx.params().build();
                    }
                }
                LsmDataSource::BaseTable { dataset } => {
                    if let Some(idx) = open_inverted_index(dataset, column).await? {
                        return Ok(idx.tokenizer());
                    }
                }
                LsmDataSource::FlushedMemTable { path, .. } => {
                    let dataset = crate::dataset::DatasetBuilder::from_uri(path)
                        .load()
                        .await?;
                    if let Some(idx) = open_inverted_index(&dataset, column).await? {
                        return Ok(idx.tokenizer());
                    }
                }
            }
        }
        Err(Error::invalid_input(format!(
            "No source carries an FTS index on column '{column}'; \
             cannot tokenize the query for LocalWithGlobalRescore mode."
        )))
    }

    async fn resolve_handle(&self, source: &LsmDataSource, column: &str) -> Result<SourceHandle> {
        match source {
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                let _ = index_store.get_fts_by_column(column).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Active memtable is missing an FTS index on column '{column}'"
                    ))
                })?;
                Ok(SourceHandle::Active {
                    batch_store: batch_store.clone(),
                    index_store: index_store.clone(),
                    schema: schema.clone(),
                    column: column.to_string(),
                })
            }
            LsmDataSource::BaseTable { dataset } => {
                let index = open_inverted_index(dataset, column).await?.ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Base table is missing an FTS index on column '{column}'"
                    ))
                })?;
                Ok(SourceHandle::Lance {
                    dataset: dataset.clone(),
                    index,
                })
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let dataset = crate::dataset::DatasetBuilder::from_uri(path)
                    .load()
                    .await?;
                let index = open_inverted_index(&dataset, column).await?.ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Flushed memtable at {path} is missing an FTS index on column '{column}'"
                    ))
                })?;
                Ok(SourceHandle::Lance {
                    dataset: Arc::new(dataset),
                    index,
                })
            }
        }
    }

    /// Materialize the rescored top-k into a single RecordBatch with
    /// the canonical FTS schema. Groups by source so we can issue one
    /// take per Lance source, then rebuilds the original row order.
    async fn materialize_rescored(
        &self,
        handles: &[SourceHandle],
        rescored: &[RescoredCandidate],
        projection: Option<&[String]>,
        target_schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let cols = self.fts_scanner_projection(projection);
        // Group candidate positions by source.
        let mut by_source: HashMap<usize, Vec<(usize, u64, f32)>> = HashMap::new();
        for (i, r) in rescored.iter().enumerate() {
            by_source
                .entry(r.source_idx)
                .or_default()
                .push((i, r.row_id, r.score));
        }

        // For each source, materialize its rows into a partial batch
        // that includes a synthetic `_order` column (the index in
        // `rescored`) so we can re-sort at the end.
        let mut partials: Vec<RecordBatch> = Vec::new();
        for (source_idx, mut entries) in by_source.into_iter() {
            // Stable to preserve relative order; not strictly needed
            // because we re-sort by `_order` below, but cheaper to keep
            // related row ids adjacent for take.
            entries.sort_by_key(|(_, rid, _)| *rid);
            let row_ids: Vec<u64> = entries.iter().map(|(_, rid, _)| *rid).collect();
            let scores: Vec<f32> = entries.iter().map(|(_, _, s)| *s).collect();
            let orders: Vec<u32> = entries.iter().map(|(i, _, _)| *i as u32).collect();

            let materialized = handles[source_idx]
                .materialize_rows(&row_ids, &cols)
                .await?;
            let mut columns: Vec<Arc<dyn Array>> = materialized.columns().to_vec();
            let mut fields: Vec<Arc<Field>> =
                materialized.schema().fields().iter().cloned().collect();
            columns.push(Arc::new(Float32Array::from(scores)));
            fields.push(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, true)));
            columns.push(Arc::new(UInt32Array::from(orders)));
            fields.push(Arc::new(Field::new(
                "__lsm_fts_order",
                DataType::UInt32,
                false,
            )));

            let schema = Arc::new(Schema::new(
                fields.iter().map(|f| (**f).clone()).collect::<Vec<_>>(),
            ));
            partials.push(RecordBatch::try_new(schema, columns)?);
        }

        if partials.is_empty() {
            // All sources returned 0 candidates after rescore.
            return Ok(RecordBatch::new_empty(target_schema.clone()));
        }

        // Concat across sources.
        let stitch_schema = partials[0].schema();
        let stitched = concat_batches(&stitch_schema, &partials)?;

        // Sort by `__lsm_fts_order` ASC so the output reflects the
        // top-k order from `rescored`.
        let order_col = stitched
            .column_by_name("__lsm_fts_order")
            .expect("__lsm_fts_order present after materialize")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("__lsm_fts_order is UInt32");
        let mut indices_with_order: Vec<(usize, u32)> = (0..order_col.len())
            .map(|i| (i, order_col.value(i)))
            .collect();
        indices_with_order.sort_by_key(|(_, o)| *o);
        let take_idx: UInt32Array = indices_with_order.iter().map(|(i, _)| *i as u32).collect();

        // Drop `__lsm_fts_order` from the final output by projecting on
        // the canonical schema's column names.
        let final_cols: Vec<Arc<dyn Array>> = target_schema
            .fields()
            .iter()
            .map(|f| {
                let src = stitched.column_by_name(f.name()).ok_or_else(|| {
                    Error::internal(format!(
                        "rescore materialization missing column '{}'",
                        f.name()
                    ))
                })?;
                let taken = take(src.as_ref(), &take_idx, None).map_err(|e| {
                    Error::internal(format!("take failed on column '{}': {e}", f.name()))
                })?;
                Ok::<_, Error>(taken)
            })
            .collect::<Result<_>>()?;
        Ok(RecordBatch::try_new(target_schema.clone(), final_cols)?)
    }

    async fn plan_local(
        &self,
        column: &str,
        query: FullTextSearchQuery,
        k: usize,
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let sources = self.collector.collect()?;
        let target_schema = self.canonical_fts_schema(projection);

        if sources.is_empty() {
            return self.empty_plan(&target_schema);
        }

        let mut per_source_plans: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(sources.len());
        for source in &sources {
            let plan = self
                .build_source_local(source, column, &query, k, projection)
                .await?;
            let normalized = project_to_canonical(plan, &target_schema)?;
            per_source_plans.push(normalized);
        }

        // Single source: skip Union and the merge.
        let merged: Arc<dyn ExecutionPlan> = if per_source_plans.len() == 1 {
            per_source_plans.into_iter().next().unwrap()
        } else {
            #[allow(deprecated)]
            let union: Arc<dyn ExecutionPlan> = Arc::new(UnionExec::new(per_source_plans));
            union
        };

        let score_idx = merged.schema().index_of(SCORE_COLUMN).map_err(|_| {
            Error::internal(format!(
                "{SCORE_COLUMN} missing from canonical FTS schema after merge"
            ))
        })?;

        let sort_expr = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new(SCORE_COLUMN, score_idx)),
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        }];
        let lex_ordering = LexOrdering::new(sort_expr).ok_or_else(|| {
            Error::internal("Failed to build LexOrdering for FTS _score sort".to_string())
        })?;

        // Per-partition sort with `fetch=k` so each upstream partition
        // can early-terminate at k; the preserving merge then does a
        // K-way heap merge also capped at k. Same pattern as
        // LsmVectorSearchPlanner.
        let per_partition_sorted: Arc<dyn ExecutionPlan> = Arc::new(
            SortExec::new(lex_ordering.clone(), merged)
                .with_preserve_partitioning(true)
                .with_fetch(Some(k)),
        );
        let merged_sorted: Arc<dyn ExecutionPlan> = Arc::new(
            SortPreservingMergeExec::new(lex_ordering, per_partition_sorted).with_fetch(Some(k)),
        );

        Ok(merged_sorted)
    }

    async fn build_source_local(
        &self,
        source: &LsmDataSource,
        column: &str,
        query: &FullTextSearchQuery,
        k: usize,
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match source {
            LsmDataSource::BaseTable { dataset } => {
                let mut scanner = dataset.scan();
                let cols = self.fts_scanner_projection(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                let bound_query = query
                    .clone()
                    .with_column(column.to_string())?
                    .limit(Some(k as i64));
                scanner.full_text_search(bound_query)?;
                scanner.create_plan().await
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let dataset = crate::dataset::DatasetBuilder::from_uri(path)
                    .load()
                    .await?;
                let mut scanner = dataset.scan();
                let cols = self.fts_scanner_projection(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                let bound_query = query
                    .clone()
                    .with_column(column.to_string())?
                    .limit(Some(k as i64));
                scanner.full_text_search(bound_query)?;
                scanner.create_plan().await
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());
                let cols = self.fts_scanner_projection(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                // `MemTableScanner::full_text_search` takes a raw match
                // string; richer query shapes (phrase/boolean/fuzzy)
                // can be plumbed via FtsQuery directly using the
                // private setter once we need them.
                let match_str = match &query.query {
                    lance_index::scalar::inverted::query::FtsQuery::Match(m) => m.terms.clone(),
                    other => {
                        return Err(Error::not_supported(format!(
                            "Active memtable FTS via LsmFtsSearchPlanner currently only \
                             supports MatchQuery, got: {other:?}"
                        )));
                    }
                };
                let _ = scanner.full_text_search(column, &match_str);
                // Active arm doesn't take a top-K hint via the builder
                // today; per-partition Sort+fetch above bounds the
                // emitted rows.
                let _ = k;
                scanner.create_plan().await
            }
        }
    }

    /// Columns to pass to the underlying scanner: user projection
    /// minus system / `_score`, with PK columns appended.
    fn fts_scanner_projection(&self, user_projection: Option<&[String]>) -> Vec<String> {
        let mut cols: Vec<String> = if let Some(p) = user_projection {
            p.iter()
                .filter(|c| !is_system_column(c) && c.as_str() != SCORE_COLUMN)
                .cloned()
                .collect()
        } else {
            self.base_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };
        for pk in &self.pk_columns {
            if !cols.contains(pk) {
                cols.push(pk.clone());
            }
        }
        cols
    }

    /// Canonical FTS output: user-projected cols + PK + `_score`.
    fn canonical_fts_schema(&self, user_projection: Option<&[String]>) -> SchemaRef {
        let mut ordered: Vec<String> = if let Some(p) = user_projection {
            p.to_vec()
        } else {
            self.base_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };
        for pk in &self.pk_columns {
            if !ordered.contains(pk) {
                ordered.push(pk.clone());
            }
        }
        if !ordered.iter().any(|c| c == SCORE_COLUMN) {
            ordered.push(SCORE_COLUMN.to_string());
        }
        let fields: Vec<Arc<Field>> = ordered
            .iter()
            .filter_map(|name| {
                if name == SCORE_COLUMN {
                    Some(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, true)))
                } else if is_system_column(name) {
                    Some(Arc::new(Field::new(name.clone(), DataType::UInt64, true)))
                } else {
                    self.base_schema
                        .field_with_name(name)
                        .ok()
                        .map(|f| Arc::new(f.clone()))
                }
            })
            .collect();
        Arc::new(Schema::new(fields))
    }

    fn empty_plan(&self, schema: &SchemaRef) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::empty::EmptyExec;
        Ok(Arc::new(EmptyExec::new(schema.clone())))
    }
}

/// One rescored hit threaded through the rescore orchestrator.
#[derive(Debug)]
struct RescoredCandidate {
    /// Index into the `handles` slice — tells materialization which
    /// source the `row_id` is relative to.
    source_idx: usize,
    /// Source-local row identifier.
    ///
    /// * Active arm: BatchStore row position.
    /// * Lance arm: Lance row id.
    row_id: u64,
    /// Score under the globally-aggregated BM25 statistics.
    score: f32,
}

/// Pre-resolved handle for a single LSM source. Created once per
/// rescore plan so we don't reopen `Dataset` / `InvertedIndex` twice
/// (once for stats, once for candidates).
enum SourceHandle {
    Active {
        batch_store: Arc<BatchStore>,
        index_store: Arc<IndexStore>,
        schema: SchemaRef,
        column: String,
    },
    Lance {
        dataset: Arc<Dataset>,
        index: Arc<InvertedIndex>,
    },
}

impl SourceHandle {
    fn stats_for_terms(&self, terms: &[String]) -> Result<(u64, usize, Vec<usize>)> {
        match self {
            Self::Active {
                index_store,
                column,
                ..
            } => {
                let idx = index_store
                    .get_fts_by_column(column)
                    .expect("active handle invariant: FTS index present");
                Ok(idx.bm25_stats_for_terms(terms))
            }
            Self::Lance { index, .. } => Ok(index.bm25_stats_for_terms(terms)),
        }
    }

    async fn candidate_search(
        &self,
        tokens: &Arc<Tokens>,
        params: &Arc<FtsSearchParams>,
        token_strs: &[String],
    ) -> Result<Vec<UnifiedCandidate>> {
        match self {
            Self::Active {
                index_store,
                column,
                ..
            } => {
                let idx = index_store
                    .get_fts_by_column(column)
                    .expect("active handle invariant: FTS index present");
                let k_prime = params.limit.unwrap_or(usize::MAX);
                let candidates = idx.search_candidates(token_strs, k_prime);
                Ok(candidates
                    .into_iter()
                    .map(UnifiedCandidate::from_fts_candidate)
                    .collect())
            }
            Self::Lance { index, .. } => {
                let prefilter = Arc::new(NoFilter);
                let metrics = Arc::new(NoOpMetricsCollector);
                let raw = index
                    .bm25_candidate_search(
                        tokens.clone(),
                        params.clone(),
                        Operator::Or,
                        prefilter,
                        metrics,
                        None,
                    )
                    .await?;
                Ok(raw
                    .into_iter()
                    .map(UnifiedCandidate::from_inverted_candidate)
                    .collect())
            }
        }
    }

    async fn materialize_rows(&self, row_ids: &[u64], cols: &[String]) -> Result<RecordBatch> {
        match self {
            Self::Active {
                batch_store,
                schema,
                ..
            } => active_materialize(batch_store, schema, row_ids, cols),
            Self::Lance { dataset, .. } => {
                // Project the dataset's Lance schema down to the requested
                // columns by name. Unknown names are dropped (`take_rows`
                // would otherwise error on schema construction).
                let names: Vec<&str> = cols
                    .iter()
                    .filter(|n| dataset.schema().field(n).is_some())
                    .map(|n| n.as_str())
                    .collect();
                let projection = dataset.schema().project(&names)?;
                Ok(dataset.take_rows(row_ids, Arc::new(projection)).await?)
            }
        }
    }
}

/// Common shape for one candidate, regardless of where it came from.
struct UnifiedCandidate {
    row_id: u64,
    doc_len: u32,
    term_freqs: Vec<u32>,
}

impl UnifiedCandidate {
    fn from_fts_candidate(c: FtsCandidate) -> Self {
        Self {
            row_id: c.row_position,
            doc_len: c.doc_len,
            term_freqs: c.term_freqs,
        }
    }

    fn from_inverted_candidate(c: InvertedIndexCandidate) -> Self {
        Self {
            row_id: c.row_id,
            doc_len: c.doc_length,
            term_freqs: c.term_freqs,
        }
    }
}

/// BM25 score from a scorer + raw per-doc stats.
fn bm25_score(scorer: &MemBM25Scorer, tokens: &[String], freqs: &[u32], doc_len: u32) -> f32 {
    let mut score = 0f32;
    for (ti, tok) in tokens.iter().enumerate() {
        let f = freqs[ti];
        if f > 0 {
            score += scorer.query_weight(tok) * scorer.doc_weight(f, doc_len);
        }
    }
    score
}

/// Pull the raw text out of a `FullTextSearchQuery` for tokenization.
///
/// Today we only handle `MatchQuery`; other shapes return a clear
/// `not_supported` error mirroring the Local-mode active-arm
/// restriction. Lifting this requires plumbing structured query shapes
/// through the rescore path, tracked in `PLAN.md`.
fn extract_match_text(query: &FullTextSearchQuery) -> Result<String> {
    match &query.query {
        IndexFtsQuery::Match(m) => Ok(m.terms.clone()),
        other => Err(Error::not_supported(format!(
            "LocalWithGlobalRescore currently supports only MatchQuery; got: {other:?}"
        ))),
    }
}

/// Open the column's inverted index from a Lance dataset, or `None`
/// if no FTS index exists for the column.
///
/// Uses the same criteria-based lookup as the base-table FTS exec path
/// (`load_scalar_index(... .for_column().supports_fts())`) rather than a
/// manual field-id scan, so flushed-generation datasets resolve their
/// maintained FTS index identically to how `scanner.full_text_search`
/// resolves it.
async fn open_inverted_index(
    dataset: &Dataset,
    column: &str,
) -> Result<Option<Arc<InvertedIndex>>> {
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
    use lance_index::IndexCriteria;

    let Some(meta) = dataset
        .load_scalar_index(IndexCriteria::default().for_column(column).supports_fts())
        .await?
    else {
        return Ok(None);
    };
    let uuid = meta.uuid.to_string();
    let opened = dataset
        .open_generic_index(column, &uuid, &lance_index::metrics::NoOpMetricsCollector)
        .await?;
    Ok(opened
        .as_any()
        .downcast_ref::<InvertedIndex>()
        .map(|inv| Arc::new(inv.clone())))
}

/// Materialize user-projected columns from the active memtable's
/// BatchStore for a sequence of BatchStore-row-position row ids.
fn active_materialize(
    batch_store: &Arc<BatchStore>,
    schema: &SchemaRef,
    row_ids: &[u64],
    cols: &[String],
) -> Result<RecordBatch> {
    // Pre-compute (start, end] ranges per batch so we can binary-search
    // a row position to its batch.
    struct BatchRange {
        start: u64,
        end: u64,
        batch_id: usize,
    }
    let mut ranges: Vec<BatchRange> = Vec::new();
    let mut cur: u64 = 0;
    for (batch_id, stored) in batch_store.iter().enumerate() {
        ranges.push(BatchRange {
            start: cur,
            end: cur + stored.num_rows as u64,
            batch_id,
        });
        cur += stored.num_rows as u64;
    }
    let find = |row_pos: u64| -> Option<&BatchRange> {
        let idx = ranges.partition_point(|r| r.end <= row_pos);
        ranges
            .get(idx)
            .filter(|r| row_pos >= r.start && row_pos < r.end)
    };

    // For each row id, append the relevant column slice to a vector.
    let col_indices: Vec<usize> = cols
        .iter()
        .map(|name| {
            schema.index_of(name).map_err(|_| {
                Error::internal(format!(
                    "active materialize: column '{name}' missing from BatchStore schema"
                ))
            })
        })
        .collect::<Result<_>>()?;
    let mut per_col: Vec<Vec<Arc<dyn Array>>> = vec![Vec::new(); col_indices.len()];
    for &row_pos in row_ids {
        let br = find(row_pos).ok_or_else(|| {
            Error::internal(format!(
                "active materialize: row position {row_pos} out of range"
            ))
        })?;
        let stored = batch_store.get(br.batch_id).ok_or_else(|| {
            Error::internal(format!(
                "active materialize: batch {} missing from store",
                br.batch_id
            ))
        })?;
        let local = (row_pos - br.start) as u32;
        let take_idx = UInt64Array::from(vec![local as u64]);
        for (slot, &src_idx) in per_col.iter_mut().zip(col_indices.iter()) {
            let col = stored.data.column(src_idx);
            let taken = take(col.as_ref(), &take_idx, None)?;
            slot.push(taken);
        }
    }
    let mut fields: Vec<Field> = Vec::with_capacity(col_indices.len());
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(col_indices.len());
    for (name, slot) in cols.iter().zip(per_col.into_iter()) {
        let src_field = schema.field_with_name(name).map_err(|e| {
            Error::internal(format!(
                "active materialize: field '{name}' lookup failed: {e}"
            ))
        })?;
        fields.push(src_field.clone());
        if slot.is_empty() {
            // No rows to take — build an empty array of the right type.
            let empty = arrow_array::new_empty_array(src_field.data_type());
            columns.push(empty);
        } else {
            let refs: Vec<&dyn Array> = slot.iter().map(|a| a.as_ref()).collect();
            let concatenated = arrow_select::concat::concat(&refs)?;
            columns.push(concatenated);
        }
    }
    let out_schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(out_schema, columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
    use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use std::collections::HashMap;

    fn fts_schema() -> Arc<ArrowSchema> {
        let mut id_meta = HashMap::new();
        id_meta.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_meta);
        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn make_batch(schema: &ArrowSchema, ids: &[i32], texts: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(texts.to_vec())),
            ],
        )
        .unwrap()
    }

    async fn write_dataset(uri: &str, batches: Vec<RecordBatch>) -> Dataset {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    #[test]
    fn rescore_k_prime_respects_floor_and_factor() {
        let mode = FtsScoringMode::LocalWithGlobalRescore { rescore_factor: 10 };
        // factor * k, floored at MIN_RESCORE_CANDIDATES
        assert_eq!(mode.rescore_k_prime(10), 100);
        assert_eq!(mode.rescore_k_prime(20), 200);
        // tiny k → floor kicks in
        assert_eq!(mode.rescore_k_prime(1), MIN_RESCORE_CANDIDATES);
        // Local mode passes k through
        assert_eq!(FtsScoringMode::Local.rescore_k_prime(50), 50);
    }

    #[tokio::test]
    async fn rescore_mode_unions_base_and_active_with_global_scores() {
        // End-to-end smoke for LocalWithGlobalRescore: a base + active
        // shape where the "lance" term appears in both. Score
        // recomputation under the global scorer must yield identical
        // scores for the two hits because both have freq=1 and dl=2 —
        // the global stats are corpus-wide so they see the same
        // (idf, avgdl) for both rows.
        use crate::index::DatasetIndexExt;
        use lance_index::IndexType;
        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

        let schema = fts_schema();
        let tmp = tempfile::tempdir().unwrap();

        // Base Lance dataset with FTS index.
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut base_ds = write_dataset(
            &base_uri,
            vec![make_batch(
                &schema,
                &[1, 2],
                &["lance fast", "unrelated text"],
            )],
        )
        .await;
        base_ds
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let base_ds = Arc::new(Dataset::open(&base_uri).await.unwrap());

        // Active memtable with FTS index over a different row.
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut indexes = IndexStore::new();
        indexes.add_fts("text_fts".to_string(), 1, "text".to_string());
        let active_batch = make_batch(&schema, &[3, 4], &["lance quick", "completely unrelated"]);
        batch_store.append(active_batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&active_batch, 0, Some(0))
            .unwrap();
        let indexes = Arc::new(indexes);

        let collector = LsmDataSourceCollector::new(base_ds, vec![]).with_in_memory_memtables(
            uuid::Uuid::new_v4(),
            InMemoryMemTables {
                active: InMemoryMemTableRef {
                    batch_store,
                    index_store: indexes,
                    schema: schema.clone(),
                    generation: 1,
                },
                frozen: vec![],
            },
        );

        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);
        let plan = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("lance".to_string()),
                10,
                None,
                FtsScoringMode::local_with_global_rescore_default(),
            )
            .await
            .expect("rescore planner should produce a base+active plan");

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // Both base id=1 and active id=3 contain "lance" → 2 hits.
        assert_eq!(total, 2, "expected exactly the 2 'lance' hits");

        let out = batches[0].schema();
        assert!(out.field_with_name(SCORE_COLUMN).is_ok());
        assert!(out.field_with_name("id").is_ok());

        // Collect (id, score) pairs.
        let mut hits: Vec<(i32, f32)> = Vec::new();
        for b in &batches {
            let ids = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let scores = b
                .column_by_name(SCORE_COLUMN)
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .unwrap();
            for i in 0..b.num_rows() {
                hits.push((ids.value(i), scores.value(i)));
            }
        }
        // Both hits must be present.
        let by_id: std::collections::HashMap<i32, f32> = hits.iter().copied().collect();
        let s1 = *by_id.get(&1).expect("base hit id=1 missing");
        let s3 = *by_id.get(&3).expect("active hit id=3 missing");
        // Global stats see N=4 docs, df("lance")=2. Both id=1 and id=3
        // have freq=1 and doc_len=2 → identical BM25 under global stats.
        assert!(
            (s1 - s3).abs() < 1e-5,
            "global rescore should give identical scores for symmetric hits; got s1={s1}, s3={s3}"
        );
        // Sort: scores descending.
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[tokio::test]
    async fn rescore_mode_active_only_runs_end_to_end() {
        // Cheaper regression that doesn't need a base Lance dataset:
        // just an active memtable. Validates the active-only candidate
        // path + rescore math.
        let schema = fts_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut indexes = IndexStore::new();
        indexes.add_fts("text_fts".to_string(), 1, "text".to_string());
        let batch = make_batch(
            &schema,
            &[1, 2, 3],
            &["lance lance lance", "lance once", "no match here"],
        );
        batch_store.append(batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();
        let indexes = Arc::new(indexes);

        let tmp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                uuid::Uuid::new_v4(),
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store: indexes,
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);
        let plan = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("lance".to_string()),
                10,
                None,
                FtsScoringMode::local_with_global_rescore_default(),
            )
            .await
            .expect("rescore planner should produce a plan");
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // id=1 (3 occurrences) and id=2 (1 occurrence) match; id=3 doesn't.
        assert_eq!(total, 2);
        // id=1 should outrank id=2 because tf is higher and doc length is similar.
        let first_id = batches[0]
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(0);
        assert_eq!(first_id, 1, "highest-tf doc should rank first");
    }

    #[tokio::test]
    async fn local_mode_unions_base_and_active_with_consistent_score_schema() {
        // Regression for the `_score` nullability mismatch between
        // FtsIndexExec (active arm) and FTS_SCHEMA (base/flushed). The
        // active-only test below would not catch this — UnionExec rejects
        // schema-inequality, so we need at least one base + one active
        // source to exercise that code path.
        use crate::index::DatasetIndexExt;
        use lance_index::IndexType;
        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

        let schema = fts_schema();
        let tmp = tempfile::tempdir().unwrap();

        // Base Lance dataset with FTS index on the `text` column.
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut base_ds = write_dataset(
            &base_uri,
            vec![make_batch(
                &schema,
                &[1, 2],
                &["lance rocks", "unrelated text"],
            )],
        )
        .await;
        base_ds
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let base_ds = Arc::new(Dataset::open(&base_uri).await.unwrap());

        // Active memtable with its own FTS index, containing a matching row.
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut indexes = IndexStore::new();
        indexes.add_fts("text_fts".to_string(), 1, "text".to_string());
        let active_batch = make_batch(
            &schema,
            &[3, 4],
            &["lance memwal goes fast", "completely unrelated"],
        );
        batch_store.append(active_batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&active_batch, 0, Some(0))
            .unwrap();
        let indexes = Arc::new(indexes);

        let collector = LsmDataSourceCollector::new(base_ds, vec![]).with_in_memory_memtables(
            uuid::Uuid::new_v4(),
            InMemoryMemTables {
                active: InMemoryMemTableRef {
                    batch_store,
                    index_store: indexes,
                    schema: schema.clone(),
                    generation: 1,
                },
                frozen: vec![],
            },
        );

        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);
        let plan = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("lance".to_string()),
                10,
                None,
                FtsScoringMode::Local,
            )
            .await
            .expect("planner should produce a base+active union plan");

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // Both base id=1 ("lance rocks") and active id=3 ("lance memwal ...")
        // should match. id=2 / id=4 do not contain "lance".
        assert!(
            total >= 2,
            "expected at least the 2 'lance' rows from base+active, got {total}"
        );

        // Both sources must agree on _score nullability — verifies the fix.
        let out = batches[0].schema();
        let score_field = out
            .field_with_name(SCORE_COLUMN)
            .expect("_score column missing from output");
        assert!(
            score_field.is_nullable(),
            "_score must be nullable to stay union-compatible across base+active"
        );

        // Sanity: ids contain at least one base hit (id=1) and one active hit (id=3).
        let mut ids: Vec<i32> = Vec::new();
        for b in &batches {
            let col = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..b.num_rows() {
                ids.push(col.value(i));
            }
        }
        assert!(ids.contains(&1), "missing base hit id=1; got ids={ids:?}");
        assert!(ids.contains(&3), "missing active hit id=3; got ids={ids:?}");
    }

    #[tokio::test]
    async fn local_mode_active_memtable_only_returns_score_sorted_hits() {
        let schema = fts_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut indexes = IndexStore::new();
        // text column has field_id 1 in fts_schema()
        indexes.add_fts("text_fts".to_string(), 1, "text".to_string());
        let batch = make_batch(
            &schema,
            &[1, 2, 3, 4],
            &[
                "lance is a columnar data format",
                "memwal handles streaming writes",
                "lance memwal lance lance",
                "completely unrelated",
            ],
        );
        batch_store.append(batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();
        let indexes = Arc::new(indexes);

        let tmp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                uuid::Uuid::new_v4(),
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store: indexes,
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);
        let plan = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("lance".to_string()),
                10,
                None,
                FtsScoringMode::Local,
            )
            .await
            .expect("local mode planner should produce a plan");

        // Plan executes and emits _score-sorted rows.
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(
            total >= 2,
            "expected at least the 2 'lance' rows, got {total}"
        );

        // Schema must include _score and the PK id.
        let out = batches[0].schema();
        assert!(out.field_with_name(SCORE_COLUMN).is_ok());
        assert!(out.field_with_name("id").is_ok());

        // _score must be non-ascending across the result.
        let mut prev_score: Option<f32> = None;
        for batch in &batches {
            let score = batch
                .column_by_name(SCORE_COLUMN)
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                let s = score.value(i);
                if let Some(p) = prev_score {
                    assert!(p >= s, "scores not sorted DESC: {p} then {s}");
                }
                prev_score = Some(s);
            }
        }
    }
}
