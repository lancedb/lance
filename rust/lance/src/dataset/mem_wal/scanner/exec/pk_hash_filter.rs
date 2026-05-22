// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Drop superseded rows from a per-source KNN result by primary-key hash.
//!
//! Drops every row whose PK hash ([`super::compute_pk_hash`]) is in `blocked` —
//! `NEWER(G)` for a generation, the union of all generations for the base table
//! (see [`super::super::block_list`]). Only the KNN's output is hashed, so there
//! is no separate scan for superseded rows.
//!
//! Only *cross-generation* supersession is blocked: a PK in `NEWER(G)` makes
//! every copy stale, so dropping by hash needs no row address. *Within-gen*
//! duplicates share a hash and aren't in `blocked`; the global dedup's
//! `(generation, freshness)` tiebreaker collapses those.
//!
//! It post-filters an over-fetched KNN (`STALE_OVERFETCH_FACTOR`), so it warns
//! when a source produced >= k candidates but < k survived — the over-fetch was
//! too small. Per-source, so it can fire even when the merged top-k is full.

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

/// Filters out rows whose primary-key hash is in `blocked`.
#[derive(Debug)]
pub struct PkHashFilterExec {
    input: Arc<dyn ExecutionPlan>,
    pk_columns: Vec<String>,
    blocked: Arc<HashSet<u64>>,
    /// Target neighbor count, used only to warn on a per-source under-fetch.
    k: usize,
    properties: Arc<PlanProperties>,
}

impl PkHashFilterExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        blocked: Arc<HashSet<u64>>,
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
                write!(
                    f,
                    "PkHashFilterExec: pk_cols=[{}], blocked={}",
                    self.pk_columns.join(", "),
                    self.blocked.len(),
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
    blocked: Arc<HashSet<u64>>,
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
                !self
                    .blocked
                    .contains(&compute_pk_hash(&batch, &pk_indices, row))
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
                         raise STALE_OVERFETCH_FACTOR or use a true KNN prefilter."
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
    /// build the blocked set from values rather than hand-computed hashes.
    fn hash_int_pk(id: i32) -> u64 {
        let batch = int_batch(&[id]);
        let pk_indices = resolve_pk_indices(&batch, &["id".to_string()]).unwrap();
        compute_pk_hash(&batch, &pk_indices, 0)
    }

    fn int_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
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
        // Block pk=20.
        let blocked = Arc::new(HashSet::from([hash_int_pk(20)]));
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, vec!["id".to_string()], blocked, 1);
        assert_eq!(run(exec).await, vec![10, 30]);
    }

    #[tokio::test]
    async fn empty_blocked_set_keeps_all_rows() {
        let b = int_batch(&[1, 2, 3]);
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec =
            PkHashFilterExec::new(input, vec!["id".to_string()], Arc::new(HashSet::new()), 1);
        assert_eq!(run(exec).await, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn null_pk_is_hashed_consistently_and_blockable() {
        // A null PK hashes deterministically (compute_pk_hash hashes is_null),
        // so a superseded null-key base row can be dropped like any other.
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let with_null = |ids: Vec<Option<i32>>| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(ids))]).unwrap()
        };
        let pk = vec!["id".to_string()];
        // Hash of the null key (from a single null-row batch).
        let null_row = with_null(vec![None]);
        let pk_indices = resolve_pk_indices(&null_row, &pk).unwrap();
        let blocked = Arc::new(HashSet::from([compute_pk_hash(&null_row, &pk_indices, 0)]));

        // Rows: 10, NULL, 30 — only the NULL-key row is dropped.
        let b = with_null(vec![Some(10), None, Some(30)]);
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, pk, blocked, 1);
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
        let blocked = Arc::new(HashSet::from([compute_pk_hash(&one_row, &pk_indices, 0)]));

        // (1,"a") and (2,"a") survive; only the exact (2,"b") tuple is dropped.
        let b = mk(&[1, 2, 2], &["a", "a", "b"]);
        let input = TestMemoryExec::try_new_exec(&[vec![b.clone()]], b.schema(), None).unwrap();
        let exec = PkHashFilterExec::new(input, pk, blocked, 1);
        assert_eq!(run(exec).await, vec![1, 2]);
    }
}
