// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Warn when the block-list post-filter leaves fewer than `k` rows.
//!
//! `UnderfillFilterWarnExec` is a pass-through node placed at the top of an LSM
//! vector-search plan *only when a per-source block-list was applied*. It counts
//! the rows that flow through and, at end of stream, logs a warning if the query
//! returned fewer than `k` — the signature of an under-fetch: the per-source
//! over-fetch (`STALE_OVERFETCH_FACTOR`) did not leave enough live candidates
//! after dropping superseded rows to fill the top-k.
//!
//! It is gated on filtering having happened, so a genuinely small result (a
//! dataset with fewer than `k` live rows) on an unfiltered query does not warn.
//! The fix for a real under-fetch is a true KNN prefilter (see
//! [`super::BlockListFilterExec`]).

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};
use tracing::warn;

/// Counts output rows and warns if fewer than `k` are produced.
#[derive(Debug)]
pub struct UnderfillFilterWarnExec {
    input: Arc<dyn ExecutionPlan>,
    k: usize,
    properties: Arc<PlanProperties>,
}

impl UnderfillFilterWarnExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, k: usize) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Self {
            input,
            k,
            properties,
        }
    }
}

impl DisplayAs for UnderfillFilterWarnExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "UnderfillFilterWarnExec: k={}", self.k)
            }
        }
    }
}

impl ExecutionPlan for UnderfillFilterWarnExec {
    fn name(&self) -> &str {
        "UnderfillFilterWarnExec"
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
                "UnderfillFilterWarnExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(children[0].clone(), self.k)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(UnderfillFilterWarnStream {
            input: input_stream,
            k: self.k,
            schema: self.schema(),
            seen: 0,
            warned: false,
        }))
    }
}

struct UnderfillFilterWarnStream {
    input: SendableRecordBatchStream,
    k: usize,
    schema: SchemaRef,
    seen: usize,
    warned: bool,
}

impl Stream for UnderfillFilterWarnStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                self.seen += batch.num_rows();
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(None) => {
                if !self.warned && self.seen < self.k {
                    warn!(
                        k = self.k,
                        returned = self.seen,
                        "LSM vector search returned fewer than k rows after the block-list \
                         post-filter; the per-source over-fetch (STALE_OVERFETCH_FACTOR) did not \
                         leave enough live candidates to fill the top-k. Raise the factor or move \
                         the block-list into the KNN as a true prefilter."
                    );
                    self.warned = true;
                }
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for UnderfillFilterWarnStream {
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

    fn batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
    }

    /// The node is a transparent pass-through regardless of the count (the
    /// warning is a side effect; rows are never altered or dropped).
    #[tokio::test]
    async fn passes_rows_through_unchanged() {
        let b = batch(&[1, 2]); // 2 rows, k=5 -> would warn, but rows must be intact
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = UnderfillFilterWarnExec::new(input, 5);
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
