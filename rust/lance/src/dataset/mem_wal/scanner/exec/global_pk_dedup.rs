// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Global, exact primary-key deduplication for the LSM vector-search
//! pipeline.
//!
//! Replaces the older two-step `WithinSourceDedupExec` + `FilterStaleExec`
//! design with a single streaming hash-by-PK pass over the merged stream.
//! For each PK the row with the largest `(generation, freshness)` tuple
//! wins — generation is the source identity (base = 0, memtable gens 1..N,
//! active = N+1) and freshness is the per-source row order normalized so
//! that "larger = newer" (see [`super::LsmSourceTagExec`]).
//!
//! Compared with the bloom-based staleness filter this is:
//!
//! - Exact (no false-positive recall loss, no top-k under-fill, no
//!   missing-bloom footgun).
//! - One node instead of two (no separate per-source dedup wrap).
//! - O(unique PKs in the merged stream) state — typically far smaller
//!   than the n_sources · k upper bound because most PKs collide across
//!   sources for typical LSM update workloads.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt, ready};

/// Cross-source PK dedup. Keeps one row per primary key — the one with
/// the largest `(generation, freshness)` tuple.
///
/// # Required input columns
///
/// - `pk_columns` — the primary key columns.
/// - `generation_column` (UInt64, NOT NULL) — typically
///   [`super::MEMTABLE_GEN_COLUMN`].
/// - `freshness_column` (UInt64, nullable) — typically
///   [`super::FRESHNESS_COLUMN`]. NULL-freshness rows are skipped (they
///   can't be ordered against real values).
///
/// The output schema is unchanged from the input. Callers that need to
/// drop the generation / freshness columns from the final output should
/// compose this node with a downstream `project_to_canonical`.
#[derive(Debug)]
pub struct LsmGlobalPkDedupExec {
    input: Arc<dyn ExecutionPlan>,
    pk_columns: Vec<String>,
    generation_column: String,
    freshness_column: String,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl LsmGlobalPkDedupExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        generation_column: impl Into<String>,
        freshness_column: impl Into<String>,
    ) -> Self {
        let schema = input.schema();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Self {
            input,
            pk_columns,
            generation_column: generation_column.into(),
            freshness_column: freshness_column.into(),
            schema,
            properties,
        }
    }

    pub fn pk_columns(&self) -> &[String] {
        &self.pk_columns
    }

    pub fn generation_column(&self) -> &str {
        &self.generation_column
    }

    pub fn freshness_column(&self) -> &str {
        &self.freshness_column
    }
}

impl DisplayAs for LsmGlobalPkDedupExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "LsmGlobalPkDedupExec: pk=[{}], gen={}, freshness={}",
                    self.pk_columns.join(", "),
                    self.generation_column,
                    self.freshness_column,
                )
            }
        }
    }
}

impl ExecutionPlan for LsmGlobalPkDedupExec {
    fn name(&self) -> &str {
        "LsmGlobalPkDedupExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "LsmGlobalPkDedupExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.pk_columns.clone(),
            self.generation_column.clone(),
            self.freshness_column.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(GlobalPkDedupStream {
            input: input_stream,
            pk_columns: self.pk_columns.clone(),
            generation_column: self.generation_column.clone(),
            freshness_column: self.freshness_column.clone(),
            schema: self.schema.clone(),
            winners: HashMap::new(),
            emitted: false,
        }))
    }
}

struct Winner {
    batch: RecordBatch,
    generation: u64,
    freshness: u64,
}

struct GlobalPkDedupStream {
    input: SendableRecordBatchStream,
    pk_columns: Vec<String>,
    generation_column: String,
    freshness_column: String,
    schema: SchemaRef,
    winners: HashMap<u64, Winner>,
    emitted: bool,
}

impl GlobalPkDedupStream {
    fn consume_batch(&mut self, batch: RecordBatch) -> DFResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let pk_indices = resolve_pk_indices(&batch, &self.pk_columns)?;
        let gen_arr = batch
            .column_by_name(&self.generation_column)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Generation column '{}' not found in batch",
                    self.generation_column
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Generation column '{}' is not UInt64",
                    self.generation_column
                ))
            })?;
        let fresh_arr = batch
            .column_by_name(&self.freshness_column)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Freshness column '{}' not found in batch",
                    self.freshness_column
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Freshness column '{}' is not UInt64",
                    self.freshness_column
                ))
            })?;

        for row_idx in 0..batch.num_rows() {
            if fresh_arr.is_null(row_idx) {
                // A NULL freshness can't be ordered against a real value;
                // skip rather than guess. Callers tag with a real value
                // for every row eligible to win.
                continue;
            }
            let generation = gen_arr.value(row_idx);
            let fresh = fresh_arr.value(row_idx);
            let pk_hash = compute_pk_hash(&batch, &pk_indices, row_idx);

            let take_row = match self.winners.get(&pk_hash) {
                None => true,
                Some(existing) => (generation, fresh) > (existing.generation, existing.freshness),
            };

            if take_row {
                let single = batch.slice(row_idx, 1);
                self.winners.insert(
                    pk_hash,
                    Winner {
                        batch: single,
                        generation,
                        freshness: fresh,
                    },
                );
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> DFResult<RecordBatch> {
        if self.winners.is_empty() {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }
        let batches: Vec<RecordBatch> = self.winners.drain().map(|(_, w)| w.batch).collect();
        let batch_refs: Vec<&RecordBatch> = batches.iter().collect();
        arrow_select::concat::concat_batches(&self.schema, batch_refs)
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

fn resolve_pk_indices(batch: &RecordBatch, pk_columns: &[String]) -> DFResult<Vec<usize>> {
    pk_columns
        .iter()
        .map(|col| {
            batch
                .schema()
                .column_with_name(col)
                .map(|(idx, _)| idx)
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "Primary key column '{}' not found",
                        col
                    ))
                })
        })
        .collect()
}

/// Hash a row's primary key. Mirrors the variants supported by
/// [`super::WithinSourceDedupExec`] / `BloomFilterGuardExec`, so a single
/// PK produces the same hash everywhere in the LSM scanner.
fn compute_pk_hash(batch: &RecordBatch, pk_indices: &[usize], row_idx: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for &col_idx in pk_indices {
        let col = batch.column(col_idx);
        let is_null = col.is_null(row_idx);
        is_null.hash(&mut hasher);

        if !is_null {
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

impl Stream for GlobalPkDedupStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.emitted {
                return Poll::Ready(None);
            }
            match ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    if let Err(e) = self.consume_batch(batch) {
                        self.emitted = true;
                        return Poll::Ready(Some(Err(e)));
                    }
                }
                Some(Err(e)) => {
                    self.emitted = true;
                    return Poll::Ready(Some(Err(e)));
                }
                None => {
                    self.emitted = true;
                    return Poll::Ready(Some(self.finalize()));
                }
            }
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for GlobalPkDedupStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::test::TestMemoryExec;
    use futures::TryStreamExt;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("_memtable_gen", DataType::UInt64, false),
            Field::new("_freshness", DataType::UInt64, true),
        ]))
    }

    fn batch(ids: &[i32], gens: &[u64], fresh: &[Option<u64>]) -> RecordBatch {
        RecordBatch::try_new(
            test_schema(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(UInt64Array::from(gens.to_vec())),
                Arc::new(UInt64Array::from(fresh.to_vec())),
            ],
        )
        .unwrap()
    }

    async fn run(batches: Vec<RecordBatch>) -> Vec<RecordBatch> {
        let schema = test_schema();
        let input = TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap();
        let exec =
            LsmGlobalPkDedupExec::new(input, vec!["id".to_string()], "_memtable_gen", "_freshness");
        let ctx = SessionContext::new();
        let stream = exec.execute(0, ctx.task_ctx()).unwrap();
        stream.try_collect().await.unwrap()
    }

    fn extract(batches: &[RecordBatch]) -> Vec<(i32, u64, Option<u64>)> {
        let mut rows = Vec::new();
        for b in batches {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let gens = b.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
            let fresh = b.column(2).as_any().downcast_ref::<UInt64Array>().unwrap();
            for i in 0..b.num_rows() {
                rows.push((
                    ids.value(i),
                    gens.value(i),
                    if fresh.is_null(i) {
                        None
                    } else {
                        Some(fresh.value(i))
                    },
                ));
            }
        }
        rows.sort_by_key(|r| r.0);
        rows
    }

    #[tokio::test]
    async fn keeps_higher_freshness_within_single_generation() {
        let b = batch(&[1, 1, 2], &[3, 3, 3], &[Some(10), Some(99), Some(5)]);
        let rows = extract(&run(vec![b]).await);
        assert_eq!(rows, vec![(1, 3, Some(99)), (2, 3, Some(5))]);
    }

    #[tokio::test]
    async fn higher_generation_beats_higher_freshness() {
        let b = batch(&[1, 1, 2], &[1, 2, 2], &[Some(u64::MAX), Some(0), Some(5)]);
        // id=1 in gen=2 with freshness 0 wins over gen=1 with freshness MAX.
        let rows = extract(&run(vec![b]).await);
        assert_eq!(rows, vec![(1, 2, Some(0)), (2, 2, Some(5))]);
    }

    #[tokio::test]
    async fn dedup_across_batches() {
        let b1 = batch(&[1, 2], &[1, 2], &[Some(5), Some(5)]);
        let b2 = batch(&[1, 3], &[3, 1], &[Some(0), Some(1)]);
        // id=1: gen=3 wins. id=2: only gen=2 row. id=3: only gen=1 row.
        let rows = extract(&run(vec![b1, b2]).await);
        assert_eq!(
            rows,
            vec![(1, 3, Some(0)), (2, 2, Some(5)), (3, 1, Some(1))],
        );
    }

    #[tokio::test]
    async fn null_freshness_skipped() {
        let b = batch(&[1, 1], &[5, 5], &[None, Some(0)]);
        // The null-freshness row is dropped; the real one wins by default.
        let rows = extract(&run(vec![b]).await);
        assert_eq!(rows, vec![(1, 5, Some(0))]);
    }

    #[tokio::test]
    async fn empty_input() {
        let total: usize = run(vec![]).await.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }
}
