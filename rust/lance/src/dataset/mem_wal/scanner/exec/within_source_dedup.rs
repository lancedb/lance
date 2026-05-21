// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WithinSourceDedupExec - Deduplicates rows with the same primary key from a
//! single LSM source, keeping the newest insert.
//!
//! In MemWAL/LSM mode the same primary key can be written multiple times into
//! the same memtable. The active memtable stores rows in insert order (larger
//! `_rowaddr` = newer), while flushed memtables are reverse-written so that
//! within a flushed file the smallest `_rowid` is the newest insert (see
//! `memtable/flush.rs:152` and `hnsw/storage.rs:307`). Point lookup uses this
//! node to collapse such duplicates *within a single source* so that the
//! downstream `CoalesceFirstExec` / `LIMIT` sees at most one row per primary
//! key per source.

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

use super::pk::{compute_pk_hash, resolve_pk_indices};

/// Among rows that share a primary key, which row-address extreme identifies
/// the newest insert to keep. The kept row is always the freshest; only the
/// row address (`_rowaddr`/`_rowid`) used to find it differs by source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupDirection {
    /// Keep the row with the largest row-address value (active memtable: larger
    /// `_rowaddr` = inserted later).
    KeepMaxRowAddr,
    /// Keep the row with the smallest row-address value (flushed memtable under
    /// reverse-write: smaller `_rowid` = inserted later).
    KeepMinRowAddr,
}

/// Deduplicates rows from a single source by primary key, keeping the row
/// whose `row_addr_column` value wins per [`DedupDirection`].
///
/// # Required columns
///
/// The input must expose:
/// - All `pk_columns`
/// - `row_addr_column` of `UInt64` type
///
/// The output schema is unchanged from the input. Callers that need to hide
/// the row-address column from downstream consumers should compose this node
/// with `project_to_canonical` or `null_columns`.
///
/// # Performance
///
/// Memory: `O(unique primary keys in input)`. For point lookup the input is
/// already filtered to a single primary key so the map holds at most one
/// entry.
#[derive(Debug)]
pub struct WithinSourceDedupExec {
    input: Arc<dyn ExecutionPlan>,
    pk_columns: Vec<String>,
    row_addr_column: String,
    direction: DedupDirection,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl WithinSourceDedupExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        row_addr_column: impl Into<String>,
        direction: DedupDirection,
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
            row_addr_column: row_addr_column.into(),
            direction,
            schema,
            properties,
        }
    }

    pub fn pk_columns(&self) -> &[String] {
        &self.pk_columns
    }

    pub fn row_addr_column(&self) -> &str {
        &self.row_addr_column
    }

    pub fn direction(&self) -> DedupDirection {
        self.direction
    }
}

impl DisplayAs for WithinSourceDedupExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "WithinSourceDedupExec: pk=[{}], row_addr={}, direction={:?}",
                    self.pk_columns.join(", "),
                    self.row_addr_column,
                    self.direction,
                )
            }
        }
    }
}

impl ExecutionPlan for WithinSourceDedupExec {
    fn name(&self) -> &str {
        "WithinSourceDedupExec"
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
                "WithinSourceDedupExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.pk_columns.clone(),
            self.row_addr_column.clone(),
            self.direction,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(WithinSourceDedupStream {
            input: input_stream,
            pk_columns: self.pk_columns.clone(),
            row_addr_column: self.row_addr_column.clone(),
            direction: self.direction,
            schema: self.schema.clone(),
            winners: HashMap::new(),
            emitted: false,
        }))
    }
}

/// One winning row, materialized as a single-row `RecordBatch` so we don't
/// have to keep the source batch alive after we've picked the winner.
struct Winner {
    batch: RecordBatch,
    row_addr: u64,
}

struct WithinSourceDedupStream {
    input: SendableRecordBatchStream,
    pk_columns: Vec<String>,
    row_addr_column: String,
    direction: DedupDirection,
    schema: SchemaRef,
    winners: HashMap<u64, Winner>,
    emitted: bool,
}

impl WithinSourceDedupStream {
    fn consume_batch(&mut self, batch: RecordBatch) -> DFResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let pk_indices = resolve_pk_indices(&batch, &self.pk_columns)?;
        let row_addr_array = batch
            .column_by_name(&self.row_addr_column)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Row-address column '{}' not found in batch",
                    self.row_addr_column
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Row-address column '{}' is not UInt64",
                    self.row_addr_column
                ))
            })?;

        for row_idx in 0..batch.num_rows() {
            if row_addr_array.is_null(row_idx) {
                // A NULL row address can't be ordered against a real one. Skip
                // rather than guess — callers should always project a real
                // row-address column for dedup-eligible sources.
                continue;
            }
            let row_addr = row_addr_array.value(row_idx);
            let pk_hash = compute_pk_hash(&batch, &pk_indices, row_idx);

            let take_row = match self.winners.get(&pk_hash) {
                None => true,
                Some(existing) => match self.direction {
                    DedupDirection::KeepMaxRowAddr => row_addr > existing.row_addr,
                    DedupDirection::KeepMinRowAddr => row_addr < existing.row_addr,
                },
            };

            if take_row {
                let single = batch.slice(row_idx, 1);
                self.winners.insert(
                    pk_hash,
                    Winner {
                        batch: single,
                        row_addr,
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

impl Stream for WithinSourceDedupStream {
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

impl datafusion::physical_plan::RecordBatchStream for WithinSourceDedupStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::test::TestMemoryExec;
    use futures::TryStreamExt;

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_distance", DataType::Float32, true),
            Field::new("_row_addr", DataType::UInt64, true),
        ]))
    }

    fn batch(ids: &[i32], names: &[&str], distances: &[f32], row_addr: &[u64]) -> RecordBatch {
        let schema = create_test_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
                Arc::new(Float32Array::from(distances.to_vec())),
                Arc::new(UInt64Array::from(row_addr.to_vec())),
            ],
        )
        .unwrap()
    }

    async fn run(batches: Vec<RecordBatch>, direction: DedupDirection) -> Vec<RecordBatch> {
        let schema = create_test_schema();
        let input = TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap();
        let exec =
            WithinSourceDedupExec::new(input, vec!["id".to_string()], "_row_addr", direction);
        let ctx = SessionContext::new();
        let stream = exec.execute(0, ctx.task_ctx()).unwrap();
        stream.try_collect().await.unwrap()
    }

    fn extract(batches: &[RecordBatch]) -> Vec<(i32, String, u64)> {
        let mut out = Vec::new();
        for b in batches {
            let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let names = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let addr = b.column(3).as_any().downcast_ref::<UInt64Array>().unwrap();
            for i in 0..b.num_rows() {
                out.push((ids.value(i), names.value(i).to_string(), addr.value(i)));
            }
        }
        out.sort_by_key(|(id, _, _)| *id);
        out
    }

    #[tokio::test]
    async fn keep_max_picks_largest_row_addr() {
        // Active-memtable case: same pk inserted twice; newer = larger _rowaddr.
        let b1 = batch(
            &[1, 1, 2],
            &["old", "new", "two"],
            &[0.1, 0.2, 0.3],
            &[10, 99, 5],
        );
        let out = run(vec![b1], DedupDirection::KeepMaxRowAddr).await;
        let rows = extract(&out);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], (1, "new".to_string(), 99));
        assert_eq!(rows[1], (2, "two".to_string(), 5));
    }

    #[tokio::test]
    async fn keep_min_picks_smallest_row_addr() {
        // Flushed-memtable case under reverse-write: newer = smaller _rowid.
        let b1 = batch(
            &[1, 1, 2],
            &["old", "new", "two"],
            &[0.1, 0.2, 0.3],
            &[99, 10, 5],
        );
        let out = run(vec![b1], DedupDirection::KeepMinRowAddr).await;
        let rows = extract(&out);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], (1, "new".to_string(), 10));
        assert_eq!(rows[1], (2, "two".to_string(), 5));
    }

    #[tokio::test]
    async fn dedup_across_batches() {
        let b1 = batch(&[1, 2], &["a", "b"], &[0.1, 0.2], &[1, 1]);
        let b2 = batch(&[1, 3], &["a_new", "c"], &[0.5, 0.4], &[7, 1]);
        let out = run(vec![b1, b2], DedupDirection::KeepMaxRowAddr).await;
        let rows = extract(&out);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], (1, "a_new".to_string(), 7));
        assert_eq!(rows[1], (2, "b".to_string(), 1));
        assert_eq!(rows[2], (3, "c".to_string(), 1));
    }

    #[tokio::test]
    async fn empty_input() {
        let out = run(vec![], DedupDirection::KeepMaxRowAddr).await;
        let total: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn null_row_addr_skipped() {
        // Rows with NULL row address can't be ordered — they're dropped so they
        // don't accidentally become winners against real values.
        let schema = create_test_schema();
        let b = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(StringArray::from(vec!["nulladdr", "real"])),
                Arc::new(Float32Array::from(vec![0.1, 0.2])),
                Arc::new(UInt64Array::from(vec![None, Some(5)])),
            ],
        )
        .unwrap();
        let out = run(vec![b], DedupDirection::KeepMaxRowAddr).await;
        let rows = extract(&out);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], (1, "real".to_string(), 5));
    }
}
