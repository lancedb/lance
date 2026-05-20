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
//!   (`doc_len`, per-term frequencies), and a coordinator rescores
//!   them with globally-aggregated stats. NOT YET IMPLEMENTED at the
//!   planner level — returns a descriptive error today; will land
//!   alongside the rescore-aware per-source exec nodes.
//!
//! Staleness: per-source results are returned as-is. The same primary
//! key may appear from multiple sources if it was updated across
//! generations; the caller is responsible for dedup if they need it.
//! This is the user-chosen behavior captured in `DESIGN.md §3`.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::UnionExec;
use lance_core::{Error, Result, is_system_column};
use lance_index::scalar::FullTextSearchQuery;
use tracing::instrument;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::projection::project_to_canonical;
use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;

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
            FtsScoringMode::LocalWithGlobalRescore { .. } => Err(Error::not_supported(format!(
                "LocalWithGlobalRescore FTS planner not yet implemented; tracked under \
                 ~/ai/analysis/lance/FTSRead/lsm-fts-search-with-global-rescore/PLAN.md \
                 (T3 phase 2). Use FtsScoringMode::Local for now (k={k}, column={column})."
            ))),
        }
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
    async fn rescore_mode_returns_clear_not_implemented_error() {
        let schema = fts_schema();
        let tmp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        write_dataset(&base_uri, vec![make_batch(&schema, &[1], &["hello"])]).await;
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![]);
        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);

        let err = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("hello".to_string()),
                10,
                None,
                FtsScoringMode::local_with_global_rescore_default(),
            )
            .await
            .expect_err("rescore mode must error until phase 2 lands");
        let msg = format!("{err}");
        assert!(
            msg.contains("LocalWithGlobalRescore"),
            "error must name the mode the user asked for: {msg}"
        );
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
