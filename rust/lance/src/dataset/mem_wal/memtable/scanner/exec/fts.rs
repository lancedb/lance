// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FtsIndexExec - Full-text search with MVCC visibility.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::error::Result as DataFusionResult;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::stream::{self, StreamExt};
use lance_core::{Error, Result};
use snafu::location;

use super::super::builder::{FtsQuery, FtsQueryType};
use crate::dataset::mem_wal::index::{FtsQueryExpr, SearchOptions};
use crate::dataset::mem_wal::scanner::bm25_stats::{DeferredBM25Stats, GlobalBM25Stats};
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// Score column name in output.
pub const SCORE_COLUMN: &str = "_score";

/// Batch range info for efficient row position lookup.
#[derive(Debug, Clone)]
struct BatchRange {
    start: usize,
    end: usize,
    batch_id: usize,
}

/// ExecutionPlan node that queries FTS index with MVCC visibility.
pub struct FtsIndexExec {
    batch_store: Arc<BatchStore>,
    indexes: Arc<IndexStore>,
    query: FtsQuery,
    max_visible_batch_position: usize,
    projection: Option<Vec<usize>>,
    output_schema: SchemaRef,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
    /// Pre-computed batch ranges for O(log n) lookup.
    batch_ranges: Vec<BatchRange>,
    /// Maximum visible row position based on max_visible_batch_position (None if nothing visible).
    max_visible_row: Option<u64>,
    /// Whether to include _rowid column (row position) in output.
    with_row_id: bool,
    /// Deferred global BM25 stats for cross-generation scoring.
    /// Resolved lazily at execution time to avoid blocking plan construction.
    deferred_bm25_stats: Option<DeferredBM25Stats>,
}

impl Debug for FtsIndexExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FtsIndexExec")
            .field("column", &self.query.column)
            .field("query_type", &self.query.query_type)
            .field(
                "max_visible_batch_position",
                &self.max_visible_batch_position,
            )
            .field("with_row_id", &self.with_row_id)
            .finish()
    }
}

impl FtsIndexExec {
    /// Create a new FtsIndexExec.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - Lock-free batch store containing data
    /// * `indexes` - Index registry with FTS indexes
    /// * `query` - FTS query parameters
    /// * `max_visible_batch_position` - MVCC visibility sequence number
    /// * `projection` - Optional column indices to project
    /// * `base_schema` - Schema before adding score column (and _rowid if with_row_id)
    /// * `with_row_id` - Whether to include _rowid column (row position)
    pub fn new(
        batch_store: Arc<BatchStore>,
        indexes: Arc<IndexStore>,
        query: FtsQuery,
        max_visible_batch_position: usize,
        projection: Option<Vec<usize>>,
        base_schema: SchemaRef,
        with_row_id: bool,
    ) -> Result<Self> {
        // Verify the index exists for this column
        let column = &query.column;
        if indexes.get_fts_by_column(column).is_none() {
            return Err(Error::invalid_input(
                format!("No FTS index found for column '{}'", column),
                location!(),
            ));
        }

        // Build output schema: base fields + _score + optional _rowid
        let mut fields: Vec<Field> = base_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(SCORE_COLUMN, DataType::Float32, false));
        if with_row_id {
            fields.push(Field::new(lance_core::ROW_ID, DataType::UInt64, true));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        // Pre-compute batch ranges for O(log n) lookup and max visible row
        let mut batch_ranges = Vec::new();
        let mut current_row = 0usize;
        let mut max_visible_row_exclusive: u64 = 0;

        for (batch_id, stored_batch) in batch_store.iter().enumerate() {
            let batch_start = current_row;
            let batch_end = current_row + stored_batch.num_rows;
            batch_ranges.push(BatchRange {
                start: batch_start,
                end: batch_end,
                batch_id,
            });
            if batch_id <= max_visible_batch_position {
                max_visible_row_exclusive = batch_end as u64;
            }
            current_row = batch_end;
        }

        // Convert exclusive end to inclusive last position, or None if nothing visible
        let max_visible_row = if max_visible_row_exclusive > 0 {
            Some(max_visible_row_exclusive - 1)
        } else {
            None
        };

        Ok(Self {
            batch_store,
            indexes,
            query,
            max_visible_batch_position,
            projection,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            batch_ranges,
            max_visible_row,
            with_row_id,
            deferred_bm25_stats: None,
        })
    }

    /// Set deferred global BM25 stats for cross-generation scoring.
    ///
    /// The stats are resolved lazily at execution time. If the stats task
    /// hasn't completed yet, the execution stream will await its result.
    pub fn with_deferred_bm25_stats(mut self, stats: Option<DeferredBM25Stats>) -> Self {
        self.deferred_bm25_stats = stats;
        self
    }
}

/// Standalone query + materialization logic used by both sync and async execute paths.
///
/// Extracted from `FtsIndexExec` methods so it can be called from an async block
/// (which cannot borrow `&self` across an `.await`).
#[allow(clippy::too_many_arguments)]
fn execute_fts_query(
    batch_store: &Arc<BatchStore>,
    indexes: &Arc<IndexStore>,
    query: &FtsQuery,
    max_visible_row: Option<u64>,
    projection: Option<&[usize]>,
    output_schema: &SchemaRef,
    batch_ranges: &[BatchRange],
    with_row_id: bool,
    resolved_stats: Option<&GlobalBM25Stats>,
) -> DataFusionResult<Vec<RecordBatch>> {
    // Query the index
    let results = query_fts_index(indexes, query, resolved_stats);

    // Filter by visibility
    let mut visible_results = match max_visible_row {
        Some(max_visible) => results
            .into_iter()
            .filter(|&(pos, _)| pos <= max_visible)
            .collect(),
        None => vec![],
    };

    // Sort by score descending
    visible_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Materialize rows
    materialize_rows_sorted(
        &visible_results,
        batch_store,
        batch_ranges,
        projection,
        output_schema,
        with_row_id,
    )
}

/// Query the FTS index and return (row_position, score) pairs.
fn query_fts_index(
    indexes: &Arc<IndexStore>,
    query: &FtsQuery,
    resolved_stats: Option<&GlobalBM25Stats>,
) -> Vec<(u64, f32)> {
    let Some(index) = indexes.get_fts_by_column(&query.column) else {
        return vec![];
    };

    let query_expr = match &query.query_type {
        FtsQueryType::Match { query } => FtsQueryExpr::match_query(query),
        FtsQueryType::Phrase { query, slop } => FtsQueryExpr::phrase_with_slop(query, *slop),
        FtsQueryType::Boolean {
            must,
            should,
            must_not,
        } => {
            let mut builder = FtsQueryExpr::boolean();
            for term in must {
                builder = builder.must(FtsQueryExpr::match_query(term));
            }
            for term in should {
                builder = builder.should(FtsQueryExpr::match_query(term));
            }
            for term in must_not {
                builder = builder.must_not(FtsQueryExpr::match_query(term));
            }
            builder.build()
        }
        FtsQueryType::Fuzzy {
            query,
            fuzziness,
            max_expansions,
        } => FtsQueryExpr::fuzzy_with_options(query, *fuzziness, *max_expansions),
    };

    let mut options = SearchOptions::new().with_wand_factor(query.wand_factor);
    if let Some(limit) = query.limit {
        options = options.with_limit(limit);
    }
    if let Some(stats) = resolved_stats {
        options = options.with_bm25_override(Arc::new(stats.clone()));
    }
    let entries = index.search_with_options(&query_expr, options);

    entries
        .into_iter()
        .map(|entry| (entry.row_position, entry.score))
        .collect()
}

/// Find batch for a row position using binary search. O(log n).
#[inline]
fn find_batch(batch_ranges: &[BatchRange], row_pos: usize) -> Option<&BatchRange> {
    let idx = batch_ranges.partition_point(|b| b.end <= row_pos);
    batch_ranges
        .get(idx)
        .filter(|b| row_pos >= b.start && row_pos < b.end)
}

/// Materialize rows from batch store preserving input order (for sorted results).
fn materialize_rows_sorted(
    results: &[(u64, f32)],
    batch_store: &Arc<BatchStore>,
    batch_ranges: &[BatchRange],
    projection: Option<&[usize]>,
    output_schema: &SchemaRef,
    with_row_id: bool,
) -> DataFusionResult<Vec<RecordBatch>> {
    if results.is_empty() {
        return Ok(vec![]);
    }

    let mut all_scores: Vec<f32> = Vec::with_capacity(results.len());
    let mut all_row_positions: Vec<u64> = Vec::with_capacity(results.len());
    let mut all_columns: Vec<Vec<Arc<dyn arrow_array::Array>>> = Vec::new();

    if let Some(stored) = batch_store.get(0) {
        for _ in 0..stored.data.num_columns() {
            all_columns.push(Vec::with_capacity(results.len()));
        }
    }

    for &(pos, score) in results {
        if let Some(batch_range) = find_batch(batch_ranges, pos as usize) {
            if let Some(stored) = batch_store.get(batch_range.batch_id) {
                let row_in_batch = (pos as usize - batch_range.start) as u32;
                let indices = UInt32Array::from(vec![row_in_batch]);

                for (col_idx, col) in stored.data.columns().iter().enumerate() {
                    let taken = arrow_select::take::take(col.as_ref(), &indices, None).unwrap();
                    if all_columns.len() <= col_idx {
                        all_columns.push(Vec::new());
                    }
                    all_columns[col_idx].push(taken);
                }

                all_scores.push(score);
                all_row_positions.push(pos);
            }
        }
    }

    if all_scores.is_empty() {
        return Ok(vec![]);
    }

    let mut final_columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();

    for col_arrays in &all_columns {
        if !col_arrays.is_empty() {
            let refs: Vec<&dyn arrow_array::Array> =
                col_arrays.iter().map(|a| a.as_ref()).collect();
            let concatenated = arrow_select::concat::concat(&refs)?;
            final_columns.push(concatenated);
        }
    }

    final_columns.push(Arc::new(Float32Array::from(all_scores)));

    let mut projected_columns = if let Some(proj_indices) = projection {
        let mut projected: Vec<_> = proj_indices
            .iter()
            .map(|&i| final_columns[i].clone())
            .collect();
        projected.push(final_columns.last().unwrap().clone());
        projected
    } else {
        final_columns
    };

    if with_row_id {
        projected_columns.push(Arc::new(UInt64Array::from(all_row_positions)));
    }

    let batch = RecordBatch::try_new(output_schema.clone(), projected_columns)?;
    Ok(vec![batch])
}

impl DisplayAs for FtsIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "FtsIndexExec: column={}, query_type={:?}, with_row_id={}",
                    self.query.column, self.query.query_type, self.with_row_id
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FtsIndexExec\ncolumn={}\nquery_type={:?}\nwith_row_id={}",
                    self.query.column, self.query.query_type, self.with_row_id
                )
            }
        }
    }
}

impl ExecutionPlan for FtsIndexExec {
    fn name(&self) -> &str {
        "FtsIndexExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "FtsIndexExec does not have children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if self.deferred_bm25_stats.is_none() {
            // Fast path: no deferred stats, execute synchronously
            let batches = execute_fts_query(
                &self.batch_store,
                &self.indexes,
                &self.query,
                self.max_visible_row,
                self.projection.as_deref(),
                &self.output_schema,
                &self.batch_ranges,
                self.with_row_id,
                None,
            )?;
            let stream = stream::iter(batches.into_iter().map(Ok)).boxed();
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.output_schema.clone(),
                stream,
            )));
        }

        // Deferred path: clone fields needed by the async block
        let deferred_stats = self.deferred_bm25_stats.clone().unwrap();
        let batch_store = self.batch_store.clone();
        let indexes = self.indexes.clone();
        let query = self.query.clone();
        let max_visible_row = self.max_visible_row;
        let projection = self.projection.clone();
        let output_schema = self.output_schema.clone();
        let batch_ranges = self.batch_ranges.clone();
        let with_row_id = self.with_row_id;

        let output_schema_for_stream = output_schema.clone();
        let fut = async move {
            // Resolve deferred BM25 stats (may already be available)
            let resolved = deferred_stats.await;
            // Run the sync query/materialization logic
            execute_fts_query(
                &batch_store,
                &indexes,
                &query,
                max_visible_row,
                projection.as_deref(),
                &output_schema,
                &batch_ranges,
                with_row_id,
                Some(&resolved),
            )
        };

        let stream = futures::stream::once(fut)
            .flat_map(|result| match result {
                Ok(batches) => stream::iter(batches.into_iter().map(Ok)).boxed(),
                Err(e) => stream::once(async move { Err(e) }).boxed(),
            })
            .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema_for_stream,
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![],
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &Schema, start_id: i32) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![start_id, start_id + 1, start_id + 2])),
                Arc::new(StringArray::from(vec![
                    "hello world",
                    "goodbye world",
                    "hello again",
                ])),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_fts_index_search() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        // Create index registry with FTS index on "text" (field_id = 1)
        let mut registry = IndexStore::new();
        registry.add_fts("text_idx".to_string(), 1, "text".to_string());

        // Insert test data and update index
        let batch = create_test_batch(&schema, 0);
        registry.insert(&batch, 0).unwrap();
        batch_store.append(batch).unwrap();

        let indexes = Arc::new(registry);

        let query = FtsQuery::match_query("text", "hello");

        let exec = FtsIndexExec::new(batch_store, indexes, query, 0, None, schema, false).unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // "hello" appears in docs 0 and 2
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);

        // Check that _score column exists
        let result_schema = batches[0].schema();
        assert!(result_schema.field_with_name(SCORE_COLUMN).is_ok());
    }

    #[tokio::test]
    async fn test_fts_index_visibility() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut registry = IndexStore::new();
        registry.add_fts("text_idx".to_string(), 1, "text".to_string());

        // Insert two batches at positions 0 and 1
        // Each batch has 3 rows, so batch1 has rows 0-2, batch2 has rows 3-5
        let batch1 = create_test_batch(&schema, 0);
        let batch2 = create_test_batch(&schema, 5);
        registry.insert(&batch1, 0).unwrap();
        registry.insert(&batch2, 3).unwrap(); // start_row_id=3 since batch1 has 3 rows
        batch_store.append(batch1).unwrap();
        batch_store.append(batch2).unwrap();

        let indexes = Arc::new(registry);

        let query = FtsQuery::match_query("text", "hello");

        // Query with max_visible=0 should only see first batch
        let exec = FtsIndexExec::new(
            batch_store.clone(),
            indexes.clone(),
            query.clone(),
            0,
            None,
            schema.clone(),
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2); // "hello" in batch1 docs 0 and 2

        // Query with max_visible=1 should see both batches
        let exec = FtsIndexExec::new(batch_store, indexes, query, 1, None, schema, false).unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 4); // "hello" in both batches
    }

    #[test]
    fn test_score_column_name() {
        assert_eq!(SCORE_COLUMN, "_score");
    }
}
