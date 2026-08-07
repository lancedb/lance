// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Schema re-labeling execution node.

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

/// Re-labels every batch to an exact target schema, leaving the arrays
/// untouched.
///
/// A shard's storage schema widens non-PK columns to nullable (see
/// `relax_non_pk_nullability`), so plan arms disagree with the base-table arm on
/// nullability alone. `ProjectionExec` cannot pin this down: DataFusion derives
/// its output nullability from the expressions, not from the schema the planner
/// intended.
///
/// Used both ways: **widening**, so arms agree before `UnionExec` /
/// `CoalesceFirstExec`; and **narrowing** at the scan's output boundary, back to
/// the logical schema. Narrowing doubles as the tombstone-leak check —
/// `RecordBatch::try_new` rejects a null in a non-nullable column, so a leak
/// errors instead of reaching the caller.
#[derive(Debug)]
pub struct SchemaRelabelExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl SchemaRelabelExec {
    /// Wrap `input` so its batches are re-labeled to `schema`, which the caller
    /// must keep column-compatible (same count, order, and data types); only
    /// names, nullability, and metadata may differ. A mismatch surfaces per
    /// batch at execution time, not at plan time.
    pub fn new(input: Arc<dyn ExecutionPlan>, schema: SchemaRef) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Self {
            input,
            schema,
            properties,
        }
    }
}

impl DisplayAs for SchemaRelabelExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "SchemaRelabelExec")
            }
        }
    }
}

impl ExecutionPlan for SchemaRelabelExec {
    fn name(&self) -> &str {
        "SchemaRelabelExec"
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
            return Err(DataFusionError::Internal(
                "SchemaRelabelExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.schema.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        Ok(Box::pin(SchemaRelabelStream {
            input: self.input.execute(partition, context)?,
            schema: self.schema.clone(),
        }))
    }
}

struct SchemaRelabelStream {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
}

impl Stream for SchemaRelabelStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let schema = self.schema.clone();
                // Guards the column-less case: `try_new` infers the row count
                // from the first column and errors when there is none.
                let relabeled = if batch.num_rows() == 0 {
                    Ok(RecordBatch::new_empty(schema))
                } else {
                    RecordBatch::try_new(schema, batch.columns().to_vec())
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                };
                Poll::Ready(Some(relabeled))
            }
            other => other,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for SchemaRelabelStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::test::TestMemoryExec;
    use futures::TryStreamExt;

    fn schema_with(nullable: bool) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, nullable),
        ]))
    }

    fn source(batch: RecordBatch) -> Arc<dyn ExecutionPlan> {
        TestMemoryExec::try_new_exec(&[vec![batch.clone()]], batch.schema(), None).unwrap()
    }

    fn batch(schema: SchemaRef, names: Vec<Option<&str>>) -> RecordBatch {
        let ids: Vec<i32> = (0..names.len() as i32).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    async fn run(plan: Arc<dyn ExecutionPlan>) -> DFResult<Vec<RecordBatch>> {
        let ctx = SessionContext::new();
        plan.execute(0, ctx.task_ctx())?.try_collect().await
    }

    #[tokio::test]
    async fn widening_preserves_rows_and_reports_target_schema() {
        let input = source(batch(schema_with(false), vec![Some("a"), Some("b")]));
        let relabeled = Arc::new(SchemaRelabelExec::new(input, schema_with(true)));

        assert_eq!(relabeled.schema(), schema_with(true));
        let out = run(relabeled).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].schema(), schema_with(true));
        assert_eq!(out[0].num_rows(), 2);
    }

    #[tokio::test]
    async fn narrowing_succeeds_when_no_nulls_remain() {
        // Post-tombstone-filter: `name` is nullable in storage, but every
        // surviving row has a value.
        let input = source(batch(schema_with(true), vec![Some("a"), Some("b")]));
        let relabeled = Arc::new(SchemaRelabelExec::new(input, schema_with(false)));

        let out = run(relabeled).await.unwrap();
        assert_eq!(out[0].schema(), schema_with(false));
        assert_eq!(out[0].num_rows(), 2);
    }

    #[tokio::test]
    async fn narrowing_rejects_a_surviving_null() {
        // A tombstone that escaped its filter must error here, not reach the
        // caller as a row of nulls.
        let input = source(batch(schema_with(true), vec![Some("a"), None]));
        let relabeled = Arc::new(SchemaRelabelExec::new(input, schema_with(false)));

        let error = run(relabeled).await.unwrap_err().to_string();
        assert!(
            error.contains("non-nullable") && error.contains("name"),
            "expected a nullability error naming the column, got: {error}"
        );
    }

    #[tokio::test]
    async fn empty_batch_is_relabeled_without_row_count_inference() {
        let input = source(batch(schema_with(true), vec![]));
        let relabeled = Arc::new(SchemaRelabelExec::new(input, schema_with(false)));

        let out = run(relabeled).await.unwrap();
        assert!(out.iter().all(|b| b.num_rows() == 0));
        assert!(out.iter().all(|b| b.schema() == schema_with(false)));
    }
}
