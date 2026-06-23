// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Drop predicate-crossing stale rows from an active-memtable index search.
//!
//! The active memtable's HNSW / inverted index are append-only, so an updated
//! row's old entries stay live. When an update moves a row out of the query's
//! match set, the fresh version isn't in the index result, so a result-set
//! dedup (keep-newest among the returned rows) has nothing to suppress the
//! stale version against — and it leaks.
//!
//! This node closes that hole with a predicate-independent recency check: for
//! each hit it asks the memtable's maintained primary-key index
//! ([`IndexStore::pk_is_newest`]) whether the hit's own row position is the
//! newest version of its primary key visible at the query's `max_visible`
//! watermark, and keeps the hit **iff so**. A stale hit (some
//! newer version exists) is dropped even when that newer version never appears
//! in the result. This is exactly the seek point-lookup already does; the index
//! search arms simply didn't do it.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::compute::filter_record_batch;
use arrow_array::{Array, BooleanArray, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};

use super::pk::resolve_pk_indices;
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// Keeps only the index hits that are the newest visible version of their PK.
///
/// The input must expose all `pk_columns` and the `row_id_column` (`UInt64`,
/// the BatchStore row position). The output schema is unchanged.
pub struct NewestPkFilterExec {
    input: Arc<dyn ExecutionPlan>,
    pk_columns: Vec<String>,
    row_id_column: String,
    /// Holds the maintained primary-key index, queried per hit via
    /// [`IndexStore::pk_is_newest`].
    index_store: Arc<IndexStore>,
    /// Resolves the `max_visible` row watermark from the visible batch prefix.
    batch_store: Arc<BatchStore>,
    /// The MVCC batch-position snapshot the index search latched. Captured once
    /// at plan time and shared with the search so the recency check keys on the
    /// same snapshot the hits came from.
    max_visible_batch_position: usize,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for NewestPkFilterExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `BatchStore` / `IndexStore` aren't `Debug`; show only the knobs.
        f.debug_struct("NewestPkFilterExec")
            .field("pk_columns", &self.pk_columns)
            .field("row_id_column", &self.row_id_column)
            .field(
                "max_visible_batch_position",
                &self.max_visible_batch_position,
            )
            .finish()
    }
}

impl NewestPkFilterExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        row_id_column: impl Into<String>,
        index_store: Arc<IndexStore>,
        batch_store: Arc<BatchStore>,
        max_visible_batch_position: usize,
    ) -> Self {
        // A filter preserves the input schema and partitioning.
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Self {
            input,
            pk_columns,
            row_id_column: row_id_column.into(),
            index_store,
            batch_store,
            max_visible_batch_position,
            properties,
        }
    }

    /// The inclusive max visible row position for this snapshot, or `None` when
    /// no rows are visible.
    fn max_visible_row(&self) -> Option<u64> {
        self.batch_store
            .max_visible_row(self.max_visible_batch_position)
    }
}

impl DisplayAs for NewestPkFilterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "NewestPkFilterExec: pk=[{}], row_id={}, max_visible_batch={}",
                    self.pk_columns.join(", "),
                    self.row_id_column,
                    self.max_visible_batch_position,
                )
            }
        }
    }
}

impl ExecutionPlan for NewestPkFilterExec {
    fn name(&self) -> &str {
        "NewestPkFilterExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
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
            return Err(DataFusionError::Internal(
                "NewestPkFilterExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.pk_columns.clone(),
            self.row_id_column.clone(),
            self.index_store.clone(),
            self.batch_store.clone(),
            self.max_visible_batch_position,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(NewestPkFilterStream {
            input: input_stream,
            pk_columns: self.pk_columns.clone(),
            row_id_column: self.row_id_column.clone(),
            index_store: self.index_store.clone(),
            max_visible_row: self.max_visible_row(),
            schema: self.schema(),
        }))
    }
}

struct NewestPkFilterStream {
    input: SendableRecordBatchStream,
    pk_columns: Vec<String>,
    row_id_column: String,
    index_store: Arc<IndexStore>,
    /// Inclusive watermark snapshot; `None` when no rows are visible.
    max_visible_row: Option<u64>,
    schema: SchemaRef,
}

impl NewestPkFilterStream {
    fn filter_batch(&self, batch: RecordBatch) -> DFResult<RecordBatch> {
        // No primary-key index (memtable without a primary key), no visible
        // rows, or an empty batch: nothing to dedup against, so pass it through.
        if !self.index_store.has_pk_index() {
            return Ok(batch);
        }
        let Some(max_visible_row) = self.max_visible_row else {
            return Ok(batch);
        };
        if batch.num_rows() == 0 {
            return Ok(batch);
        }

        let pk_indices = resolve_pk_indices(&batch, &self.pk_columns)?;
        let row_ids = batch
            .column_by_name(&self.row_id_column)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "Row-id column '{}' not found in NewestPkFilterExec input",
                    self.row_id_column
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "Row-id column '{}' is not UInt64",
                    self.row_id_column
                ))
            })?;

        let mut keep = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            // A null row position can't be ordered; keep it rather than guess
            // (callers always project a real position here).
            if row_ids.is_null(row) {
                keep.push(true);
                continue;
            }
            let position = row_ids.value(row);
            let values: Vec<ScalarValue> = pk_indices
                .iter()
                .map(|&col| ScalarValue::try_from_array(batch.column(col), row))
                .collect::<DFResult<_>>()?;
            // Keep iff this hit is the newest visible version of its PK.
            keep.push(
                self.index_store
                    .pk_is_newest(&values, position, max_visible_row),
            );
        }
        filter_record_batch(&batch, &BooleanArray::from(keep))
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl Stream for NewestPkFilterStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Some(self.filter_batch(batch))),
            other => other,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for NewestPkFilterStream {
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

    /// Single-column `id` PK batch, one per append so a caller can control
    /// row-level visibility via `max_visible_batch_position`.
    fn id_batch(id: i32) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![id]))]).unwrap()
    }

    /// Index-search "hits": `(id, _rowid)` pairs the filter evaluates.
    fn hits(rows: &[(i32, u64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(lance_core::ROW_ID, DataType::UInt64, true),
        ]));
        let ids: Vec<i32> = rows.iter().map(|(id, _)| *id).collect();
        let rowids: Vec<u64> = rows.iter().map(|(_, p)| *p).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(UInt64Array::from(rowids)),
            ],
        )
        .unwrap()
    }

    /// Build an active memtable whose PK index + BatchStore hold one row per
    /// `id` in `appended` (positions 0..n), all committed.
    fn active(appended: &[i32]) -> (Arc<IndexStore>, Arc<BatchStore>) {
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index = IndexStore::new();
        index.enable_pk_index(&[("id".to_string(), 0)]);
        for &id in appended {
            let b = id_batch(id);
            let (bp, off, _) = batch_store.append(b.clone()).unwrap();
            index.insert_with_batch_position(&b, off, Some(bp)).unwrap();
        }
        (Arc::new(index), batch_store)
    }

    async fn run(
        index_store: Arc<IndexStore>,
        batch_store: Arc<BatchStore>,
        max_visible_batch_position: usize,
        hits_batch: RecordBatch,
    ) -> Vec<(i32, u64)> {
        let input =
            TestMemoryExec::try_new_exec(&[vec![hits_batch.clone()]], hits_batch.schema(), None)
                .unwrap();
        let exec = NewestPkFilterExec::new(
            input,
            vec!["id".to_string()],
            lance_core::ROW_ID,
            index_store,
            batch_store,
            max_visible_batch_position,
        );
        let ctx = SessionContext::new();
        let out: Vec<RecordBatch> = exec
            .execute(0, ctx.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for b in &out {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let pos = b.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
            for i in 0..b.num_rows() {
                rows.push((ids.value(i), pos.value(i)));
            }
        }
        rows
    }

    #[tokio::test]
    async fn keeps_only_the_newest_visible_position_per_pk() {
        // id=1 written at positions 0 and 2 (an update), id=2 at position 1; all
        // visible. A stale hit (id=1 @ 0) is dropped; the newest (id=1 @ 2) and
        // the unrelated id=2 survive — even though all three were "returned" by
        // the index search.
        let (index, store) = active(&[1, 2, 1]);
        let rows = run(index, store, 2, hits(&[(1, 0), (2, 1), (1, 2)])).await;
        assert_eq!(rows, vec![(2, 1), (1, 2)]);
    }

    #[tokio::test]
    async fn does_not_vanish_a_visible_row_under_a_newer_invisible_write() {
        // The store/index hold id=1 at positions 0 and 2, but the query latched
        // `max_visible_batch_position = 0` (only position 0 visible) — i.e. the
        // update at position 2 was committed *after* this query's snapshot. The
        // visible older row (id=1 @ 0) must be KEPT (its newest *visible* version
        // is itself), not dropped because of the not-yet-visible position 2.
        let (index, store) = active(&[1, 2, 1]);
        let kept = run(index.clone(), store.clone(), 0, hits(&[(1, 0)])).await;
        assert_eq!(kept, vec![(1, 0)], "visible row must not vanish");

        // And the not-yet-visible position is itself dropped (outside snapshot).
        let dropped = run(index, store, 0, hits(&[(1, 2)])).await;
        assert!(
            dropped.is_empty(),
            "row beyond the snapshot must be dropped"
        );
    }

    #[tokio::test]
    async fn passes_through_when_no_pk_index() {
        // A memtable without a primary-key index can't be deduped here, so the
        // filter is a pass-through rather than dropping everything.
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        batch_store.append(id_batch(1)).unwrap();
        let index = Arc::new(IndexStore::new()); // no enable_pk_index
        let rows = run(index, batch_store, 0, hits(&[(1, 0), (1, 9)])).await;
        assert_eq!(rows, vec![(1, 0), (1, 9)]);
    }
}
