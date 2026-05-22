// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Drop superseded rows from a per-source KNN result by primary-key hash.
//!
//! Drops a row when its PK hash ([`super::compute_pk_hash`]) is in any `blocked`
//! set — the newer generations' membership (`Arc<HashSet>`, shared, never merged;
//! base table: all generations). Only the KNN output is hashed.
//!
//! Cross-generation only: within-gen duplicates share a hash, so the global
//! dedup's `(generation, freshness)` tiebreaker collapses those instead.
//!
//! Post-filters an over-fetched KNN (the planner's `overfetch_factor`); warns
//! when a source had >= k candidates but < k survived (over-fetch too small).

use std::any::Any;
use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::compute::filter_record_batch;
use arrow_array::{BooleanArray, RecordBatch};
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

use super::pk::{compute_pk_hash, resolve_pk_indices};

/// Filters out rows whose PK hash is in any set of `blocked`.
#[derive(Debug)]
pub struct PkHashFilterExec {
    input: Arc<dyn ExecutionPlan>,
    pk_columns: Vec<String>,
    /// Newer generations' membership; a row is blocked if any set holds its hash.
    blocked: Vec<Arc<HashSet<u64>>>,
    /// Target neighbor count, used only to warn on a per-source under-fetch.
    k: usize,
    properties: Arc<PlanProperties>,
}

impl PkHashFilterExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        blocked: Vec<Arc<HashSet<u64>>>,
        k: usize,
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
            blocked,
            k,
            properties,
        }
    }
}

impl DisplayAs for PkHashFilterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                let total: usize = self.blocked.iter().map(|s| s.len()).sum();
                write!(
                    f,
                    "PkHashFilterExec: pk_cols=[{}], gens={}, blocked={}",
                    self.pk_columns.join(", "),
                    self.blocked.len(),
                    total,
                )
            }
        }
    }
}

impl ExecutionPlan for PkHashFilterExec {
    fn name(&self) -> &str {
        "PkHashFilterExec"
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
                "PkHashFilterExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.pk_columns.clone(),
            self.blocked.clone(),
            self.k,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(PkHashFilterStream {
            input: input_stream,
            pk_columns: self.pk_columns.clone(),
            blocked: self.blocked.clone(),
            k: self.k,
            schema: self.schema(),
            input_seen: 0,
            kept: 0,
            warned: false,
        }))
    }
}

struct PkHashFilterStream {
    input: SendableRecordBatchStream,
    pk_columns: Vec<String>,
    blocked: Vec<Arc<HashSet<u64>>>,
    k: usize,
    schema: SchemaRef,
    input_seen: usize,
    kept: usize,
    warned: bool,
}

impl PkHashFilterStream {
    fn filter_batch(&self, batch: RecordBatch) -> DFResult<RecordBatch> {
        if self.blocked.is_empty() || batch.num_rows() == 0 {
            return Ok(batch);
        }
        let pk_indices = resolve_pk_indices(&batch, &self.pk_columns)?;
        let keep: BooleanArray = (0..batch.num_rows())
            .map(|row| {
                let hash = compute_pk_hash(&batch, &pk_indices, row);
                !self.blocked.iter().any(|set| set.contains(&hash))
            })
            .collect();
        filter_record_batch(&batch, &keep)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl Stream for PkHashFilterStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                self.input_seen += batch.num_rows();
                match self.filter_batch(batch) {
                    Ok(out) => {
                        self.kept += out.num_rows();
                        Poll::Ready(Some(Ok(out)))
                    }
                    Err(e) => Poll::Ready(Some(Err(e))),
                }
            }
            Poll::Ready(None) => {
                // >= k candidates in, < k out: the over-fetch missed superseded rows.
                if !self.warned && self.input_seen >= self.k && self.kept < self.k {
                    warn!(
                        k = self.k,
                        fetched = self.input_seen,
                        kept = self.kept,
                        "LSM vector search: < k live rows survived the PK-hash post-filter; \
                         raise the over-fetch factor or use a true KNN prefilter."
                    );
                    self.warned = true;
                }
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for PkHashFilterStream {
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

    /// Hash a single-column Int32 PK value the way the exec does, so a test can
    /// build blocked sets from values rather than hand-computed hashes.
    fn hash_int_pk(id: i32) -> u64 {
        let batch = int_batch(&[id]);
        let pk_indices = resolve_pk_indices(&batch, &["id".to_string()]).unwrap();
        compute_pk_hash(&batch, &pk_indices, 0)
    }

    fn int_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
    }

    fn blocked(ids: &[i32]) -> Vec<Arc<HashSet<u64>>> {
        vec![Arc::new(ids.iter().map(|&id| hash_int_pk(id)).collect())]
    }

    async fn run(exec: PkHashFilterExec) -> Vec<i32> {
        let ctx = SessionContext::new();
        let out: Vec<RecordBatch> = exec
            .execute(0, ctx.task_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        out.iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[tokio::test]
    async fn drops_rows_with_blocked_pk_hash() {
        let b = int_batch(&[10, 20, 30]);
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, vec!["id".to_string()], blocked(&[20]), 1);
        assert_eq!(run(exec).await, vec![10, 30]);
    }

    #[tokio::test]
    async fn blocks_a_pk_present_in_any_generation_set() {
        // Two newer-gen sets: a row is dropped if either contains its PK.
        let b = int_batch(&[10, 20, 30]);
        let sets = vec![
            Arc::new(HashSet::from([hash_int_pk(10)])),
            Arc::new(HashSet::from([hash_int_pk(30)])),
        ];
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, vec!["id".to_string()], sets, 1);
        assert_eq!(run(exec).await, vec![20]);
    }

    #[tokio::test]
    async fn empty_blocked_keeps_all_rows() {
        let b = int_batch(&[1, 2, 3]);
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, vec!["id".to_string()], Vec::new(), 1);
        assert_eq!(run(exec).await, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn null_pk_is_hashed_consistently_and_blockable() {
        // A null PK hashes deterministically (compute_pk_hash hashes is_null),
        // so a superseded null-key row can be dropped like any other.
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let with_null = |ids: Vec<Option<i32>>| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(ids))]).unwrap()
        };
        let pk = vec!["id".to_string()];
        let null_row = with_null(vec![None]);
        let pk_indices = resolve_pk_indices(&null_row, &pk).unwrap();
        let sets = vec![Arc::new(HashSet::from([compute_pk_hash(
            &null_row,
            &pk_indices,
            0,
        )]))];

        // Rows: 10, NULL, 30 — only the NULL-key row is dropped.
        let b = with_null(vec![Some(10), None, Some(30)]);
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, pk, sets, 1);
        assert_eq!(run(exec).await, vec![10, 30]);
    }

    #[tokio::test]
    async fn composite_pk_hash_matches_block_set() {
        // Composite PK (id, name): block the (2, "b") tuple only.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let mk = |ids: &[i32], names: &[&str]| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids.to_vec())),
                    Arc::new(StringArray::from(names.to_vec())),
                ],
            )
            .unwrap()
        };
        let pk = vec!["id".to_string(), "name".to_string()];
        let one_row = mk(&[2], &["b"]);
        let pk_indices = resolve_pk_indices(&one_row, &pk).unwrap();
        let sets = vec![Arc::new(HashSet::from([compute_pk_hash(
            &one_row,
            &pk_indices,
            0,
        )]))];

        // (1,"a") and (2,"a") survive; only the exact (2,"b") tuple is dropped.
        let b = mk(&[1, 2, 2], &["a", "a", "b"]);
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, pk, sets, 1);
        assert_eq!(run(exec).await, vec![1, 2]);
    }
}
