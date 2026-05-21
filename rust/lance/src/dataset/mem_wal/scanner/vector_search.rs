// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector search planner for LSM scanner.
//!
//! Provides KNN (K-Nearest Neighbors) search across LSM levels with staleness detection.

use std::sync::Arc;

use arrow_array::FixedSizeListArray;
use arrow_schema::SchemaRef;
use arrow_schema::SortOptions;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
#[allow(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::UnionExec;
use lance_core::Result;
use lance_core::datatypes::OnMissing;
use tracing::instrument;

use crate::dataset::Dataset;
use crate::io::exec::TakeExec;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{FreshnessPolarity, LsmGlobalPkDedupExec, LsmSourceTagExec};
use super::flushed_cache::{FlushedMemTableCache, open_flushed_dataset};
use super::projection::{
    DISTANCE_COLUMN, build_scanner_projection, canonical_internal_schema, canonical_output_schema,
    null_columns, project_to_canonical, wants_row_id,
};
use crate::session::Session;

/// Plans vector search queries over LSM data.
///
/// Each source independently runs KNN, then results are unioned and run
/// through a single global PK dedup that picks the row with the largest
/// `(generation, freshness)` tuple per primary key. Generation is the
/// source identity (base = 0, memtable gens 1..N, active = N+1) and
/// freshness is the per-source row order normalized so larger = newer
/// (see [`LsmSourceTagExec`]).
///
/// # Query Plan Structure
///
/// ```text
/// TakeExec (optional: fetch user-projected cols from base dataset)
///   SortPreservingMergeExec: order_by=[_distance ASC], fetch=k
///     SortExec: order_by=[_distance ASC], fetch=k          (per partition, parallel)
///       ProjectionExec (drops _memtable_gen, _freshness)
///         LsmGlobalPkDedupExec: pk=[…], gen=_memtable_gen, freshness=_freshness
///           CoalescePartitionsExec
///             UnionExec
///               ProjectionExec (canonical internal schema)
///                 ProjectionExec (null_columns _rowid)        (non-base only)
///                   LsmSourceTagExec: gen=N+1, polarity=InsertOrder        (active)
///                     KNNExec: active memtable, k=k
///               ProjectionExec (canonical internal schema)
///                 ProjectionExec (null_columns _rowid)
///                   LsmSourceTagExec: gen=N, polarity=ReverseWrite        (flushed)
///                     KNNExec: flushed gen N, k=k (fast_search)
///               … one per flushed gen …
///               ProjectionExec (canonical internal schema)
///                 LsmSourceTagExec: gen=0, polarity=InsertOrder            (base)
///                   KNNExec: base table, k=k (fast_search)[.refine()?]
/// ```
///
/// # Index-Only Search (fast_search)
///
/// For base table and flushed memtables we use `fast_search()` to only
/// search indexed data. This is correct because:
/// - Each flushed memtable has its own vector index built during flush.
/// - The active memtable covers any unindexed data.
/// - Searching unindexed data in base/flushed would be redundant.
///
/// # Dedup semantics
///
/// `LsmGlobalPkDedupExec` keeps the row whose `(generation, freshness)`
/// tuple is largest, so newer generations always win and ties within a
/// generation fall to the source-local freshness (larger row offset for
/// active memtables; smaller `_rowid` for flushed memtables, flipped by
/// `LsmSourceTagExec` so the comparison stays uniform).
pub struct LsmVectorSearchPlanner {
    /// Data source collector.
    collector: LsmDataSourceCollector,
    /// Primary key column names (used by the global dedup).
    pk_columns: Vec<String>,
    /// Schema of the base table.
    base_schema: SchemaRef,
    /// Vector column name.
    vector_column: String,
    /// Distance metric type (L2, Cosine, Dot, etc.).
    distance_type: lance_linalg::distance::DistanceType,
    /// Base dataset reference for post-rerank take.
    ///
    /// After the global PK dedup and sort, a `TakeExec` against this
    /// dataset materializes any user-projected columns that were not
    /// part of the per-source KNN output. Rows from memtables already
    /// carry all columns; the take only fetches additional data for
    /// base-table rows (which have a real `_rowid`).
    dataset: Option<Arc<Dataset>>,
    /// Session threaded into flushed-generation opens (shared caches).
    session: Option<Arc<Session>>,
    /// Cache of opened flushed-generation datasets.
    flushed_cache: Option<Arc<FlushedMemTableCache>>,
}

impl LsmVectorSearchPlanner {
    /// Create a new planner.
    ///
    /// # Arguments
    ///
    /// * `collector` - Data source collector
    /// * `pk_columns` - Primary key column names
    /// * `base_schema` - Schema of the base table
    /// * `vector_column` - Name of the vector column to search
    /// * `distance_type` - Distance metric (L2, Cosine, etc.)
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
        vector_column: String,
        distance_type: lance_linalg::distance::DistanceType,
    ) -> Self {
        Self {
            collector,
            pk_columns,
            base_schema,
            vector_column,
            distance_type,
            dataset: None,
            session: None,
            flushed_cache: None,
        }
    }

    /// Thread a session into flushed-generation opens so the first open
    /// populates the shared index / file-metadata caches.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Inject a cache of opened flushed-generation datasets, making repeated
    /// searches against the same generation a pure `Arc::clone`.
    pub fn with_flushed_cache(mut self, cache: Arc<FlushedMemTableCache>) -> Self {
        self.flushed_cache = Some(cache);
        self
    }

    /// Set the base dataset for post-rerank take.
    ///
    /// After global PK dedup and sort, a `TakeExec` against this dataset
    /// materializes any user-projected columns that were not part of the
    /// per-source KNN output. This is necessary because per-source KNN
    /// only returns the columns needed for dedup and ranking; the take
    /// step fetches the full user projection for the final top-k rows.
    pub fn with_dataset(mut self, dataset: Arc<Dataset>) -> Self {
        self.dataset = Some(dataset);
        self
    }

    /// Create a vector search plan.
    ///
    /// # Arguments
    ///
    /// * `query_vector` - Query vector for KNN search
    /// * `k` - Number of nearest neighbors to return
    /// * `nprobes` - Number of IVF partitions to search (for IVF-based indexes)
    /// * `projection` - Columns to include in output (None = all columns)
    /// * `refine_factor` - When set, the base-table arm of the KNN plan fetches
    ///   `k * refine_factor` candidates and re-ranks them with exact distances.
    ///   Useful when the base table uses an approximate index (IVF-PQ) so that
    ///   cross-source distance comparison is exact. Memtable arms use exact
    ///   HNSW search and do not need refine.
    ///
    /// # Returns
    ///
    /// An execution plan that returns the top-K nearest neighbors across all
    /// LSM levels, with stale results filtered out.
    #[instrument(name = "lsm_vector_search", level = "info", skip_all, fields(k, nprobes, vector_column = %self.vector_column, distance_type = ?self.distance_type))]
    pub async fn plan_search(
        &self,
        query_vector: &FixedSizeListArray,
        k: usize,
        nprobes: usize,
        projection: Option<&[String]>,
        refine_factor: Option<u32>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let sources = self.collector.collect()?;

        if sources.is_empty() {
            return self.empty_plan(projection);
        }

        let canonical_schema = canonical_output_schema(
            projection,
            &self.base_schema,
            &self.pk_columns,
            true, // include _distance — KNN always produces it
        );
        // The internal schema carries `_memtable_gen` + `_freshness`
        // through the union and the global dedup; both are dropped
        // afterwards by a project back to the canonical output schema.
        let internal_schema =
            canonical_internal_schema(projection, &self.base_schema, &self.pk_columns, true);

        let mut knn_plans = Vec::new();
        for source in &sources {
            let generation = source.generation();
            let is_base = matches!(source, LsmDataSource::BaseTable { .. });
            let knn = self
                .build_knn_plan(source, query_vector, k, nprobes, projection, refine_factor)
                .await?;
            // Tag rows with `(_memtable_gen, _freshness)`. Polarity differs
            // per source — see [`LsmSourceTagExec`] / [`FreshnessPolarity`]:
            //   * active memtable:  insert order, larger `_rowid` = newer
            //   * flushed memtable: reverse-written, smaller `_rowid` = newer
            //   * base table:       no duplicates expected; polarity moot
            let polarity = match source {
                LsmDataSource::FlushedMemTable { .. } => FreshnessPolarity::ReverseWrite,
                LsmDataSource::ActiveMemTable { .. } | LsmDataSource::BaseTable { .. } => {
                    FreshnessPolarity::InsertOrder
                }
            };
            let tagged: Arc<dyn ExecutionPlan> = Arc::new(LsmSourceTagExec::new(
                knn,
                generation,
                polarity,
                lance_core::ROW_ID,
            ));
            // Lance's `fast_search()` always produces `_rowid` whether or
            // not we asked for it; the active arm also produces `_rowid`
            // when we ask for it (to drive freshness). For non-base arms
            // the per-source value would collide with base row ids in the
            // canonical output, so NULL it before stitching into the
            // internal schema. The dedup has already consumed it via
            // `_freshness`.
            let after_null = if is_base {
                tagged
            } else {
                null_columns(tagged, &[lance_core::ROW_ID])?
            };
            // Normalize each source to the internal canonical schema
            // (canonical user cols + `_memtable_gen` + `_freshness`).
            let normalized = project_to_canonical(after_null, &internal_schema)?;
            knn_plans.push(normalized);
        }

        #[allow(deprecated)]
        let union: Arc<dyn ExecutionPlan> = Arc::new(UnionExec::new(knn_plans));

        // LsmGlobalPkDedupExec declares one output partition but only
        // reads partition 0 of its input — coalesce first or partitions
        // past the base table get silently dropped.
        let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(union));
        let deduped: Arc<dyn ExecutionPlan> = Arc::new(LsmGlobalPkDedupExec::new(
            coalesced,
            self.pk_columns.clone(),
            super::exec::MEMTABLE_GEN_COLUMN,
            super::exec::FRESHNESS_COLUMN,
        ));
        // Drop `_memtable_gen` and `_freshness` — they're internal-only.
        let merged: Arc<dyn ExecutionPlan> = project_to_canonical(deduped, &canonical_schema)?;

        let distance_idx = merged.schema().index_of(DISTANCE_COLUMN).map_err(|_| {
            lance_core::Error::invalid_input(format!(
                "Column '{}' not found in schema",
                DISTANCE_COLUMN
            ))
        })?;

        let sort_expr = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new(DISTANCE_COLUMN, distance_idx)),
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        }];

        let lex_ordering = LexOrdering::new(sort_expr).ok_or_else(|| {
            lance_core::Error::internal("Failed to create LexOrdering".to_string())
        })?;

        // Sort each partition's candidates in parallel, capped at k via `with_fetch`
        // so each partition early-terminates instead of materializing every candidate
        // into a global sort. SortPreservingMergeExec then does a p-way heap merge of
        // the pre-sorted streams (also capped at k), producing the final top-k.
        // This avoids serializing the merge on a single thread and pushes the k-limit
        // into per-partition work, where it actually helps.
        //
        // `with_preserve_partitioning(true)` is load-bearing: without it, SortExec
        // declares it needs a SinglePartition input but doesn't coalesce on its own —
        // execute(0) silently reads only partition 0 of the union and drops the rest.
        let per_partition_sorted: Arc<dyn ExecutionPlan> = Arc::new(
            SortExec::new(lex_ordering.clone(), merged)
                .with_preserve_partitioning(true)
                .with_fetch(Some(k)),
        );
        let merged_sorted: Arc<dyn ExecutionPlan> = Arc::new(
            SortPreservingMergeExec::new(lex_ordering, per_partition_sorted).with_fetch(Some(k)),
        );

        // After global rerank, take any user-projected columns that the
        // per-source KNN didn't return. This fetches from the base dataset
        // using `_rowid`; memtable rows (NULL `_rowid`) already carry all
        // their data so the take is a no-op for them.
        #[allow(deprecated)]
        let result = if let Some(dataset) = &self.dataset {
            let cols = build_scanner_projection(projection, &self.base_schema, &self.pk_columns);
            let output_projection = dataset
                .empty_projection()
                .union_columns(cols, OnMissing::Ignore)?;
            let coalesced: Arc<dyn ExecutionPlan> =
                Arc::new(CoalesceBatchesExec::new(merged_sorted.clone(), 8192));
            if let Some(take_plan) =
                TakeExec::try_new(dataset.clone(), coalesced, output_projection)?
            {
                Arc::new(take_plan) as Arc<dyn ExecutionPlan>
            } else {
                merged_sorted
            }
        } else {
            merged_sorted
        };

        Ok(result)
    }

    /// Build KNN plan for a single data source.
    async fn build_knn_plan(
        &self,
        source: &LsmDataSource,
        query_vector: &FixedSizeListArray,
        k: usize,
        nprobes: usize,
        projection: Option<&[String]>,
        refine_factor: Option<u32>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match source {
            LsmDataSource::BaseTable { dataset } => {
                let mut scanner = dataset.scan();
                let cols =
                    build_scanner_projection(projection, &self.base_schema, &self.pk_columns);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                // Only the base produces a meaningful `_rowid`. `_rowaddr`
                // can't be combined with `fast_search()` — the IVF index
                // doesn't preserve it and `TakeExec` refuses to insert it
                // post-search — so it stays NULL across all arms.
                if wants_row_id(projection) {
                    scanner.with_row_id();
                }
                let query_arr = single_query_array(query_vector);
                scanner.nearest(&self.vector_column, query_arr.as_ref(), k)?;
                scanner.nprobes(nprobes);
                scanner.distance_metric(self.distance_type);
                // Memtables cover unindexed rows; only search indexed data here.
                scanner.fast_search();
                // Re-rank base candidates with exact distances when set, so
                // they're directly comparable to MemTable distances in the merge.
                if let Some(factor) = refine_factor {
                    scanner.refine(factor);
                }
                scanner.create_plan().await
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let dataset =
                    open_flushed_dataset(path, self.session.as_ref(), self.flushed_cache.as_ref())
                        .await?;
                let mut scanner = dataset.scan();
                let cols =
                    build_scanner_projection(projection, &self.base_schema, &self.pk_columns);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                // No `with_row_id/address`: per-source IDs would collide with base.
                let query_arr = single_query_array(query_vector);
                scanner.nearest(&self.vector_column, query_arr.as_ref(), k)?;
                scanner.nprobes(nprobes);
                scanner.distance_metric(self.distance_type);
                scanner.fast_search();
                scanner.create_plan().await
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;
                use arrow_array::Array;

                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());
                // PK auto-included so the staleness filter retains its bloom hash key.
                let cols =
                    build_scanner_projection(projection, &self.base_schema, &self.pk_columns);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                // Expose `_rowid` (BatchStore row offset, monotonic with
                // insert order) so [`WithinSourceDedupExec`] can collapse
                // duplicate-PK rows to the newest insert. The value is
                // per-source and NULL'd before reaching the canonical merge.
                // (VectorIndexExec only plumbs `with_row_id`, not
                // `with_row_address`, but the two yield identical values
                // for an active memtable so either would work.)
                scanner.with_row_id();
                let query_arr: Arc<dyn Array> = Arc::new(query_vector.clone());
                scanner.nearest(&self.vector_column, query_arr, k);
                scanner.nprobes(nprobes);
                scanner.distance_metric(self.distance_type);
                scanner.create_plan().await
            }
        }
    }

    /// Create an empty execution plan with the canonical KNN output schema.
    fn empty_plan(&self, projection: Option<&[String]>) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::empty::EmptyExec;

        let schema = canonical_output_schema(projection, &self.base_schema, &self.pk_columns, true);
        Ok(Arc::new(EmptyExec::new(schema)))
    }
}

/// Convert a (typically single-row) FixedSizeList query into the array shape
/// `Scanner::nearest` expects:
///
/// - 1 row → the inner Float32Array (single-vector query). Passing the FSL
///   directly would make the scanner treat it as a multivector query and
///   reject it on a non-multivector column.
/// - >1 row → the FSL itself (multivector query path).
fn single_query_array(query_vector: &FixedSizeListArray) -> arrow_array::ArrayRef {
    use arrow_array::Array;
    if query_vector.len() == 1 {
        query_vector.value(0)
    } else {
        std::sync::Arc::new(query_vector.clone()) as arrow_array::ArrayRef
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::{
        Int32Array, RecordBatch, RecordBatchIterator, builder::FixedSizeListBuilder,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::collections::HashMap;

    fn create_vector_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);

        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new(
                "vector",
                // Use nullable=true to match what FixedSizeListBuilder produces
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
        ]))
    }

    fn create_query_vector() -> FixedSizeListArray {
        use arrow_array::builder::Float32Builder;

        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 4);
        builder.values().append_value(0.1);
        builder.values().append_value(0.2);
        builder.values().append_value(0.3);
        builder.values().append_value(0.4);
        builder.append(true);

        builder.finish()
    }

    fn create_test_batch(schema: &ArrowSchema, ids: &[i32]) -> RecordBatch {
        use arrow_array::builder::Float32Builder;

        let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), 4);
        for id in ids {
            let base = *id as f32 * 0.1;
            vector_builder.values().append_value(base);
            vector_builder.values().append_value(base + 0.1);
            vector_builder.values().append_value(base + 0.2);
            vector_builder.values().append_value(base + 0.3);
            vector_builder.append(true);
        }

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(vector_builder.finish()),
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
    async fn test_vector_search_plan_structure() {
        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[1, 2, 3]);
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema.clone(),
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        let query = create_query_vector();
        let plan = planner.plan_search(&query, 10, 8, None, None).await;

        // Plan construction must succeed. Execution against empty data is a
        // separate concern handled by integration tests.
        plan.expect("planner should produce a plan even when memtables are empty");
    }

    #[tokio::test]
    async fn test_projection_includes_pk() {
        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[1]);
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let _planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema.clone(),
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        // Project only "vector" - should also include "id" for staleness detection
        let cols =
            build_scanner_projection(Some(&["vector".to_string()]), &schema, &["id".to_string()]);

        assert!(cols.contains(&"vector".to_string()));
        assert!(cols.contains(&"id".to_string()));
    }

    #[tokio::test]
    async fn test_vector_search_base_plus_active_returns_distance() {
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use datafusion::prelude::SessionContext;
        use futures::TryStreamExt;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[10, 20, 30]);
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Active memtable with HNSW index over the "vector" column.
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        index_store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        let batch = create_test_batch(&schema, &[1, 2, 3, 4]);
        batch_store.append(batch.clone()).unwrap();
        index_store
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = uuid::Uuid::new_v4();
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]).with_in_memory_memtables(
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

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        let query = create_query_vector();
        let plan = planner
            .plan_search(&query, 3, 1, None, None)
            .await
            .expect("planner should produce a plan");

        let ctx = SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(total > 0, "expected at least one result row");

        let out_schema = batches[0].schema();
        let out_cols: Vec<String> = out_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert!(
            out_schema.field_with_name(DISTANCE_COLUMN).is_ok(),
            "output schema is missing `_distance` column. Got: {:?}",
            out_cols
        );
        // Internal columns must not leak: `_rowid` (added by Lance's fast_search
        // in the base/flushed arms) and `_memtable_gen` (added by the LSM merge
        // when bloom filters are present) are bookkeeping, not API.
        assert!(
            out_schema.field_with_name("_rowid").is_err(),
            "`_rowid` leaked into output: {:?}",
            out_cols
        );
        assert!(
            out_schema
                .field_with_name(super::super::exec::MEMTABLE_GEN_COLUMN)
                .is_err(),
            "`_memtable_gen` leaked into output: {:?}",
            out_cols
        );

        // The nearest neighbor to the query vector should be id=1 (exact match),
        // and its `_distance` must be ~0 — verifying both the column is present
        // and the values are correct.
        let id_col = batches[0]
            .column_by_name("id")
            .expect("id column missing")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id column should be Int32");
        let dist_col = batches[0]
            .column_by_name(DISTANCE_COLUMN)
            .expect("_distance column missing")
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .expect("_distance column should be Float32");
        assert_eq!(id_col.value(0), 1, "expected id=1 as nearest neighbor");
        assert!(
            dist_col.value(0).abs() < 1e-3,
            "expected near-zero distance for self-match, got {}",
            dist_col.value(0)
        );
    }

    #[tokio::test]
    async fn test_vector_search_with_projection_returns_distance_and_pk() {
        // Regression for: active arm previously did NOT call `build_projection_for_knn`,
        // so when the caller passed `projection=Some([...])`, the active arm's output
        // dropped the PK column (and the staleness/dedup pipeline lost its hash input).
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use datafusion::prelude::SessionContext;
        use futures::TryStreamExt;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[10]);
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        index_store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        let batch = create_test_batch(&schema, &[1, 2, 3, 4]);
        batch_store.append(batch.clone()).unwrap();
        index_store
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = uuid::Uuid::new_v4();
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]).with_in_memory_memtables(
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

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        // Caller asks for ONLY "vector"; PK "id" must still be in output because
        // canonical_output_schema auto-includes it.
        let query = create_query_vector();
        let projection = vec!["vector".to_string()];
        let plan = planner
            .plan_search(&query, 3, 1, Some(&projection), None)
            .await
            .expect("planner should produce a plan");

        let ctx = SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(total > 0, "expected at least one result row");

        let out_schema = batches[0].schema();
        assert!(
            out_schema.field_with_name("id").is_ok(),
            "PK column `id` should be auto-included even when user projects only `vector`"
        );
        assert!(out_schema.field_with_name("vector").is_ok());
        assert!(out_schema.field_with_name(DISTANCE_COLUMN).is_ok());
    }

    #[tokio::test]
    async fn test_vector_search_projection_with_explicit_distance_and_rowid() {
        // Regression: `_distance` / `_rowid` in projection must not break
        // the plan. `_distance` honored at requested position (no
        // duplication); `_rowid` honored as nullable.
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use datafusion::prelude::SessionContext;
        use futures::TryStreamExt;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[10]);
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        index_store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        let batch = create_test_batch(&schema, &[1, 2, 3, 4]);
        batch_store.append(batch.clone()).unwrap();
        index_store
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = uuid::Uuid::new_v4();
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]).with_in_memory_memtables(
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

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        let query = create_query_vector();
        // Caller explicitly puts `_distance` and `_rowid` in projection.
        let projection = vec![
            "_distance".to_string(),
            "vector".to_string(),
            "_rowid".to_string(),
        ];
        let plan = planner
            .plan_search(&query, 3, 1, Some(&projection), None)
            .await
            .expect(
                "planner must accept `_distance`/`_rowid` in projection without breaking the plan",
            );

        let ctx = SessionContext::new();
        let stream = plan
            .execute(0, ctx.task_ctx())
            .expect("plan must execute when `_distance`/`_rowid` are in projection");
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(total > 0, "expected at least one result row");

        let out_schema = batches[0].schema();
        // `_distance` must appear exactly once — auto-managed, never duplicated.
        let distance_count = out_schema
            .fields()
            .iter()
            .filter(|f| f.name() == DISTANCE_COLUMN)
            .count();
        assert_eq!(
            distance_count,
            1,
            "`_distance` must appear exactly once in output, got schema: {:?}",
            out_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>()
        );
        // User columns are still present.
        assert!(out_schema.field_with_name("vector").is_ok());
        // PK auto-included for staleness detection.
        assert!(out_schema.field_with_name("id").is_ok());
        // Top hit comes from the active memtable. Active-arm `_rowid` is
        // NULL by design (BatchStore position, not a Lance row id). Real
        // `_rowid` is only produced for base-table rows.
        assert!(out_schema.field_with_name("_rowid").is_ok());
        let rowid = batches[0].column_by_name("_rowid").unwrap();
        assert!(
            rowid.is_null(0),
            "active-memtable `_rowid` must be NULL (not a real Lance row id), got: {:?}",
            rowid
        );
    }

    #[tokio::test]
    async fn test_vector_search_strips_internal_columns_and_preserves_active_rows() {
        // Two regressions in one test:
        // (1) `LsmGlobalPkDedupExec` consumes `_memtable_gen` and `_freshness`
        //     but the user-visible output must NOT contain them — the
        //     post-dedup `project_to_canonical` is what strips them, so a
        //     refactor that drops that projection would leak these columns.
        // (2) `LsmGlobalPkDedupExec` declares one output partition but only
        //     reads partition 0 of its input. Without a `CoalescePartitionsExec`
        //     ahead of it, every union partition past partition 0 is silently
        //     dropped — i.e. active-memtable rows disappear when the union
        //     puts them in a non-zero partition.
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use datafusion::prelude::SessionContext;
        use futures::TryStreamExt;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[10]);
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        index_store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        let batch = create_test_batch(&schema, &[1, 2, 3, 4]);
        batch_store.append(batch.clone()).unwrap();
        index_store
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = uuid::Uuid::new_v4();
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]).with_in_memory_memtables(
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

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        let query = create_query_vector();
        let plan = planner
            .plan_search(&query, 3, 1, None, None)
            .await
            .expect("planner should produce a plan");

        // Plan must include the new global dedup (proves the pipeline is wired).
        let plan_str = format!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
        assert!(
            plan_str.contains("LsmGlobalPkDedupExec"),
            "expected new global-dedup pipeline, got:\n{}",
            plan_str
        );

        let ctx = SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(total > 0, "expected at least one result row");

        let out_schema = batches[0].schema();
        assert!(out_schema.field_with_name(DISTANCE_COLUMN).is_ok());
        for internal in [
            super::super::exec::MEMTABLE_GEN_COLUMN,
            super::super::exec::FRESHNESS_COLUMN,
        ] {
            assert!(
                out_schema.field_with_name(internal).is_err(),
                "`{}` leaked into output: {:?}",
                internal,
                out_schema
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect::<Vec<_>>(),
            );
        }

        // (2) Active-memtable rows must survive: collector emits base as
        // partition 0 of the union and the active memtable as partition 1+.
        // The active memtable holds ids 1..=4; the base holds id 10. At
        // least one id in 1..=4 must appear in the output, otherwise the
        // CoalescePartitionsExec was skipped and partitions 1+ were dropped.
        let mut all_ids: Vec<i32> = Vec::new();
        for batch in &batches {
            let id_col = batch
                .column_by_name("id")
                .expect("id column missing")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column should be Int32");
            for i in 0..batch.num_rows() {
                all_ids.push(id_col.value(i));
            }
        }
        assert!(
            all_ids.iter().any(|&id| (1..=4).contains(&id)),
            "expected at least one active-memtable row (id in 1..=4) — none found, so \
             active partitions were silently dropped. Got ids: {:?}",
            all_ids
        );
    }

    #[tokio::test]
    async fn test_vector_search_dedup_across_generations() {
        // Regression: same primary key inserted into two sources (older
        // flushed gen and newer active memtable) with different vectors.
        // Without the cross-source PK dedup the older flushed row would
        // still appear in top-k. The newer-generation row must win.
        //
        // We simulate a "flushed gen 1" by writing a tiny Lance dataset
        // under {base_uri}/_mem_wal/{shard}/gen_1 and pointing the
        // collector at it. Real flush would reverse-write, but for this
        // test we only have one row in the flushed gen so order is moot.
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::scanner::data_source::ShardSnapshot;
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use datafusion::prelude::SessionContext;
        use futures::TryStreamExt;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();
        let base_uri = format!("{}/base", base_path);

        // Flushed gen 1 holds an older version of pk=1 with a "wrong" vector.
        let shard_id = uuid::Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let old_pk1 = create_test_batch_with_vector(&schema, 1, [9.0, 9.0, 9.0, 9.0]);
        create_dataset(&gen1_uri, vec![old_pk1]).await;

        // Active memtable holds the newer version of pk=1 with the
        // "right" vector close to the query, plus an unrelated pk=2.
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        index_store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        let new_pk1 = create_test_batch_with_vector(&schema, 1, [0.1, 0.2, 0.3, 0.4]);
        let other = create_test_batch_with_vector(&schema, 2, [5.0, 5.0, 5.0, 5.0]);
        let (_, _, bp1) = batch_store.append(new_pk1.clone()).unwrap();
        index_store
            .insert_with_batch_position(&new_pk1, 0, Some(bp1))
            .unwrap();
        let (_, _, bp2) = batch_store.append(other.clone()).unwrap();
        index_store
            .insert_with_batch_position(&other, 1, Some(bp2))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![shard_snapshot])
            .with_in_memory_memtables(
                shard_id,
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store,
                        schema: schema.clone(),
                        generation: 2,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        let query = create_query_vector();
        let plan = planner.plan_search(&query, 5, 1, None, None).await.unwrap();
        let ctx = SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        let pk1_count = ids.iter().filter(|i| **i == 1).count();
        assert_eq!(
            pk1_count, 1,
            "pk=1 must appear exactly once after cross-source dedup; got ids={:?}",
            ids,
        );
    }

    #[tokio::test]
    async fn test_vector_search_system_columns_real_only_for_base() {
        // Covers tests 1+2+3 from the PR review:
        //   1. base-hit `_rowid`/`_rowaddr` carry real values
        //   2. flushed-memtable arm runs without erroring
        //   3. `_rowaddr` symmetry with `_rowid` (same code path, both are
        //      surfaced when requested and NULL'd outside the base arm)
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::scanner::data_source::ShardSnapshot;
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use crate::index::DatasetIndexExt;
        use crate::index::vector::VectorIndexParams;
        use datafusion::prelude::SessionContext;
        use futures::TryStreamExt;
        use lance_index::IndexType;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        // Base: id=1 (vector closest to query). Index it so fast_search
        // returns it.
        let base_batch = create_test_batch(&schema, &[1]);
        let mut base_dataset = create_dataset(&base_uri, vec![base_batch]).await;
        let ivf_flat = VectorIndexParams::ivf_flat(1, lance_linalg::distance::DistanceType::L2);
        base_dataset
            .create_index(&["vector"], IndexType::Vector, None, &ivf_flat, true)
            .await
            .unwrap();
        let base_dataset = Arc::new(base_dataset);

        // Flushed memtable: id=2 (a separate Lance dataset under
        // {base_uri}/_mem_wal/{shard}/gen_1) with its own vector index.
        let shard_id = uuid::Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let gen1_batch = create_test_batch(&schema, &[2]);
        let mut gen1_dataset = create_dataset(&gen1_uri, vec![gen1_batch]).await;
        gen1_dataset
            .create_index(&["vector"], IndexType::Vector, None, &ivf_flat, true)
            .await
            .unwrap();

        // Active memtable: id=3 with HNSW index.
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        index_store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );
        let active_batch = create_test_batch(&schema, &[3]);
        batch_store.append(active_batch.clone()).unwrap();
        index_store
            .insert_with_batch_position(&active_batch, 0, Some(0))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        let collector = LsmDataSourceCollector::new(base_dataset, vec![shard_snapshot])
            .with_in_memory_memtables(
                shard_id,
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store,
                        schema: schema.clone(),
                        generation: 2,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        // Project both system columns alongside data columns.
        let query = create_query_vector();
        let projection = vec![
            "id".to_string(),
            "_rowid".to_string(),
            "_rowaddr".to_string(),
            "vector".to_string(),
        ];
        let plan = planner
            .plan_search(&query, 3, 1, Some(&projection), None)
            .await
            .expect("planner should produce a plan");

        let ctx = SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // Top-3 over 3 ids, one per source. All should appear.
        assert_eq!(total, 3, "expected one row per source");

        // Group by id → (rowid_null, rowaddr_null).
        let mut seen: std::collections::HashMap<i32, (bool, bool)> =
            std::collections::HashMap::new();
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let rowid = batch.column_by_name("_rowid").unwrap();
            let rowaddr = batch.column_by_name("_rowaddr").unwrap();
            for i in 0..batch.num_rows() {
                seen.insert(ids.value(i), (rowid.is_null(i), rowaddr.is_null(i)));
            }
        }

        // id=1 (base): `_rowid` real (the index produces it). `_rowaddr` is
        // always NULL across vector_search — Lance's `fast_search()` can't
        // be combined with `with_row_address()`.
        let (rid_null, raddr_null) = seen.get(&1).expect("base row id=1 missing");
        assert!(
            !rid_null,
            "base row `_rowid` must be real (Lance row id), got NULL"
        );
        assert!(
            raddr_null,
            "`_rowaddr` is incompatible with vector_search's fast_search; must be NULL"
        );

        // id=2 (flushed): both NULL — per-source values would collide with base.
        let (rid_null, raddr_null) = seen.get(&2).expect("flushed row id=2 missing");
        assert!(rid_null, "flushed row `_rowid` must be NULL");
        assert!(raddr_null, "flushed row `_rowaddr` must be NULL");

        // id=3 (active): both NULL — BatchStore position is not a Lance row id.
        let (rid_null, raddr_null) = seen.get(&3).expect("active row id=3 missing");
        assert!(rid_null, "active row `_rowid` must be NULL");
        assert!(raddr_null, "active row `_rowaddr` must be NULL");
    }

    #[tokio::test]
    async fn test_vector_search_empty_plan_with_system_columns() {
        // Test 5 (vector_search slice): with no sources, the empty plan
        // must still expose user-requested system columns at the requested
        // position, plus `_distance` (always-on for KNN).
        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![]);
        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        let projection = vec![
            "_rowid".to_string(),
            "vector".to_string(),
            "_rowaddr".to_string(),
        ];
        let query = create_query_vector();
        let plan = planner
            .plan_search(&query, 5, 1, Some(&projection), None)
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
                "_rowid".to_string(),
                "vector".to_string(),
                "_rowaddr".to_string(),
                "id".to_string(),        // PK auto-appended
                "_distance".to_string(), // always-on for KNN
            ],
            "empty KNN plan must honor user position for system cols and append PK + _distance"
        );
    }

    #[tokio::test]
    async fn test_vector_search_without_base_table() {
        use futures::TryStreamExt;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        // No base dataset written. Plan construction must still succeed and
        // exclude any base-table scan node.
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![]);

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        let query = create_query_vector();
        let plan = planner
            .plan_search(&query, 10, 8, None, None)
            .await
            .expect("planner should produce a plan without a base table");

        let plan_str = format!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
        assert!(
            !plan_str.contains("base/data"),
            "Plan must not scan base table, got: {}",
            plan_str
        );

        // Execute the plan so runtime issues (schema mismatches, missing
        // sources, etc.) surface here rather than at the call site.
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan
            .execute(0, ctx.task_ctx())
            .expect("plan should execute without a base table");
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .expect("collecting batches should succeed");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0, "fresh tier with no sources should yield no rows");
    }

    /// Build a single-row batch with an explicit (id, vector) so tests can
    /// pin the within-source dedup against same-PK / different-vector
    /// inputs.
    fn create_test_batch_with_vector(
        schema: &ArrowSchema,
        id: i32,
        vector: [f32; 4],
    ) -> RecordBatch {
        use arrow_array::builder::Float32Builder;

        let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), 4);
        for v in &vector {
            vector_builder.values().append_value(*v);
        }
        vector_builder.append(true);
        let vector_array = vector_builder.finish();

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![id])), Arc::new(vector_array)],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_vector_search_dedup_within_active_memtable() {
        // Regression: same PK inserted twice into one active memtable with
        // *different* vectors. HNSW indexes each as a distinct node, so
        // without WithinSourceDedupExec a KNN can return both candidates
        // for the same PK and pollute top-k. The newer insert must win.
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use datafusion::prelude::SessionContext;
        use futures::TryStreamExt;

        let schema = create_vector_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        index_store.add_hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
            64,
            8,
        );

        // Two rows with pk=1, different vectors. b_new (last insert) is the
        // newer version that the LSM contract says should win.
        let b_old = create_test_batch_with_vector(&schema, 1, [9.0, 9.0, 9.0, 9.0]);
        let b_new = create_test_batch_with_vector(&schema, 1, [0.1, 0.2, 0.3, 0.4]);
        // An unrelated row so top-k has more than one PK to choose from.
        let b_other = create_test_batch_with_vector(&schema, 2, [5.0, 5.0, 5.0, 5.0]);

        let (_, _, bp_old) = batch_store.append(b_old.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_old, 0, Some(bp_old))
            .unwrap();
        let (_, _, bp_new) = batch_store.append(b_new.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_new, 1, Some(bp_new))
            .unwrap();
        let (_, _, bp_other) = batch_store.append(b_other.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_other, 2, Some(bp_other))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = uuid::Uuid::new_v4();
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

        let planner = LsmVectorSearchPlanner::new(
            collector,
            vec!["id".to_string()],
            schema,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        );

        // Query is exactly the *newer* vector for pk=1. If the older
        // vector for pk=1 leaks through, it'd appear in top-k too because
        // the older row's vector is far from the query but still a graph
        // node. After dedup we should see pk=1 exactly once.
        let query = create_query_vector();
        let plan = planner.plan_search(&query, 5, 1, None, None).await.unwrap();

        let ctx = SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        let pk1_count = ids.iter().filter(|i| **i == 1).count();
        assert_eq!(
            pk1_count, 1,
            "pk=1 must appear exactly once after within-source dedup; got ids={:?}",
            ids,
        );
    }
}
