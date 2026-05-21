// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Drop superseded rows from a per-source KNN result using a block-list bitmap.
//!
//! `BlockListFilterExec` removes every input row whose `_rowid` is blocked by a
//! [`RowAddrMask`] — i.e. a row superseded by a newer LSM generation (or within
//! its own generation). It is applied to each per-source KNN arm *before* the
//! cross-source union so that a stale row never reaches the merge, closing the
//! gap where a fresh row that fell out of its source's top-k could not suppress
//! the stale copy.
//!
//! The block-list is keyed by `_rowid` (the id each source's KNN emits); see
//! [`super::super::block_list`] for how the per-source masks are built.
//!
//! TODO(perf/correctness): this is a *post*-filter — it relies on the per-source
//! KNN over-fetching so that enough live rows survive the drop. That does not
//! guarantee K live results in the adversarial case (more blocked rows near the
//! query than the over-fetch covers). Push the same mask *into* the per-source
//! search as a true prefilter (the index keeps traversing until K rows pass), so
//! there are no holes to backfill.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::compute::filter_record_batch;
use arrow_array::{Array, BooleanArray, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};
use lance_core::utils::mask::{RowAddrMask, RowSetOps};

/// Filters out rows whose `row_id_column` value is blocked by `mask`.
#[derive(Debug)]
pub struct BlockListFilterExec {
    input: Arc<dyn ExecutionPlan>,
    mask: Arc<RowAddrMask>,
    row_id_column: String,
    properties: Arc<PlanProperties>,
}

impl BlockListFilterExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        mask: Arc<RowAddrMask>,
        row_id_column: impl Into<String>,
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
            mask,
            row_id_column: row_id_column.into(),
            properties,
        }
    }
}

impl DisplayAs for BlockListFilterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "BlockListFilterExec: row_id_col={}, blocked={}",
                    self.row_id_column,
                    self.mask
                        .block_list()
                        .and_then(|b| b.len())
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "?".to_string()),
                )
            }
        }
    }
}

impl ExecutionPlan for BlockListFilterExec {
    fn name(&self) -> &str {
        "BlockListFilterExec"
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
                "BlockListFilterExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.mask.clone(),
            self.row_id_column.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(BlockListFilterStream {
            input: input_stream,
            mask: self.mask.clone(),
            row_id_column: self.row_id_column.clone(),
            schema: self.schema(),
        }))
    }
}

struct BlockListFilterStream {
    input: SendableRecordBatchStream,
    mask: Arc<RowAddrMask>,
    row_id_column: String,
    schema: SchemaRef,
}

impl BlockListFilterStream {
    fn filter_batch(&self, batch: RecordBatch) -> DFResult<RecordBatch> {
        let row_ids = batch
            .column_by_name(&self.row_id_column)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "BlockListFilterExec: row id column '{}' not found",
                    self.row_id_column
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "BlockListFilterExec: row id column '{}' is not UInt64",
                    self.row_id_column
                ))
            })?;

        // Keep a row when its id passes the mask. A null id (should not occur for
        // these sources) is kept rather than silently dropped.
        let keep: BooleanArray = (0..row_ids.len())
            .map(|i| row_ids.is_null(i) || self.mask.selected(row_ids.value(i)))
            .collect();

        filter_record_batch(&batch, &keep)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl Stream for BlockListFilterStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Some(self.filter_batch(batch))),
            other => other,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for BlockListFilterStream {
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
    use lance_core::utils::mask::RowAddrTreeMap;

    fn batch(ids: &[i32], row_ids: &[u64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(UInt64Array::from(row_ids.to_vec())),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn drops_blocked_row_ids() {
        let b = batch(&[10, 20, 30], &[100, 200, 300]);
        // Block _rowid 200.
        let mut tree = RowAddrTreeMap::new();
        tree.insert(200);
        let mask = Arc::new(RowAddrMask::from_block(tree));

        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = BlockListFilterExec::new(input, mask, "_rowid");

        let ctx = SessionContext::new();
        let out: Vec<RecordBatch> = exec
            .execute(0, ctx.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let ids: Vec<i32> = out
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
        // Row with _rowid 200 (id=20) is dropped; the rest survive.
        assert_eq!(ids, vec![10, 30]);
    }

    #[tokio::test]
    async fn empty_mask_keeps_all_rows() {
        let b = batch(&[1, 2], &[0, 1]);
        let mask = Arc::new(RowAddrMask::from_block(RowAddrTreeMap::new()));
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = BlockListFilterExec::new(input, mask, "_rowid");
        let ctx = SessionContext::new();
        let out: Vec<RecordBatch> = exec
            .execute(0, ctx.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }
}
