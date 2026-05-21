// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-source tagging for the LSM vector-search dedup pipeline.
//!
//! `LsmSourceTagExec` appends two columns to each row of a per-source scan:
//! - `_memtable_gen` (UInt64): the source's generation number (base = 0,
//!   flushed gens 1..N, active memtable = N+1).
//! - `_freshness` (UInt64): a within-source "newness" indicator normalized
//!   so that *larger value = newer insert* regardless of which side
//!   produced it. The active memtable stores rows in insert order
//!   (`_freshness = _rowid`), while flushed memtables are reverse-written
//!   (`_freshness = u64::MAX - _rowid`).
//!
//! Together, the two columns let [`super::LsmGlobalPkDedupExec`] decide a
//! winner per primary key via a single lexicographic `(gen, freshness)`
//! comparison across the merged stream — no separate within-source dedup
//! and no bloom-based staleness filtering needed.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::error::Result as DFResult;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};

use crate::dataset::mem_wal::scanner::data_source::LsmGeneration;

use super::generation_tag::MEMTABLE_GEN_COLUMN;

/// Column name for the normalized within-source freshness. Higher = newer.
pub const FRESHNESS_COLUMN: &str = "_freshness";

/// Polarity for translating a source's row-id column into `_freshness`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessPolarity {
    /// `_freshness = row_id`. Used by sources that store rows in insert
    /// order (active memtable; also base table where duplicates aren't
    /// expected but the polarity must still be consistent).
    InsertOrder,
    /// `_freshness = u64::MAX - row_id`. Used by flushed memtables, which
    /// are reverse-written so a smaller `_rowid` is the newer insert.
    ReverseWrite,
}

/// Tag every row of a per-source scan with `_memtable_gen` + `_freshness`.
///
/// # Required input columns
///
/// - `row_id_column` (UInt64) — typically `_rowid`. Must be present on
///   every row; NULLs are propagated as NULL `_freshness` and will be
///   skipped by the downstream dedup.
///
/// # Output schema
///
/// Input schema + `_memtable_gen` (UInt64, NOT NULL) + `_freshness`
/// (UInt64, nullable to mirror the source's `_rowid` nullability).
#[derive(Debug)]
pub struct LsmSourceTagExec {
    input: Arc<dyn ExecutionPlan>,
    generation: LsmGeneration,
    polarity: FreshnessPolarity,
    row_id_column: String,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl LsmSourceTagExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        generation: LsmGeneration,
        polarity: FreshnessPolarity,
        row_id_column: impl Into<String>,
    ) -> Self {
        let input_schema = input.schema();
        let mut fields: Vec<Arc<Field>> = input_schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(
            MEMTABLE_GEN_COLUMN,
            DataType::UInt64,
            false,
        )));
        fields.push(Arc::new(Field::new(
            FRESHNESS_COLUMN,
            DataType::UInt64,
            true,
        )));
        let schema = Arc::new(Schema::new(fields));

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));

        Self {
            input,
            generation,
            polarity,
            row_id_column: row_id_column.into(),
            schema,
            properties,
        }
    }

    pub fn generation(&self) -> LsmGeneration {
        self.generation
    }

    pub fn polarity(&self) -> FreshnessPolarity {
        self.polarity
    }

    pub fn row_id_column(&self) -> &str {
        &self.row_id_column
    }
}

impl DisplayAs for LsmSourceTagExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "LsmSourceTagExec: gen={}, polarity={:?}, row_id_col={}",
                    self.generation, self.polarity, self.row_id_column,
                )
            }
        }
    }
}

impl ExecutionPlan for LsmSourceTagExec {
    fn name(&self) -> &str {
        "LsmSourceTagExec"
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
                "LsmSourceTagExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.generation,
            self.polarity,
            self.row_id_column.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(SourceTagStream {
            input: input_stream,
            generation: self.generation.as_u64(),
            polarity: self.polarity,
            row_id_column: self.row_id_column.clone(),
            schema: self.schema.clone(),
        }))
    }
}

struct SourceTagStream {
    input: SendableRecordBatchStream,
    generation: u64,
    polarity: FreshnessPolarity,
    row_id_column: String,
    schema: SchemaRef,
}

impl SourceTagStream {
    fn tag_batch(&self, batch: RecordBatch) -> DFResult<RecordBatch> {
        let num_rows = batch.num_rows();
        let gen_col: Arc<dyn Array> = Arc::new(UInt64Array::from(vec![self.generation; num_rows]));

        let row_id_arr = batch
            .column_by_name(&self.row_id_column)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Row id column '{}' not found in batch — LsmSourceTagExec needs the per-source row id to derive _freshness",
                    self.row_id_column
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Row id column '{}' is not UInt64",
                    self.row_id_column
                ))
            })?;

        let freshness: Arc<dyn Array> = match self.polarity {
            FreshnessPolarity::InsertOrder => Arc::new(row_id_arr.clone()),
            FreshnessPolarity::ReverseWrite => {
                let mut builder = arrow_array::builder::UInt64Builder::with_capacity(num_rows);
                for i in 0..num_rows {
                    if row_id_arr.is_null(i) {
                        builder.append_null();
                    } else {
                        builder.append_value(u64::MAX - row_id_arr.value(i));
                    }
                }
                Arc::new(builder.finish())
            }
        };

        let mut columns: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        columns.push(gen_col);
        columns.push(freshness);

        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl Stream for SourceTagStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let tagged = self.tag_batch(batch);
                Poll::Ready(Some(tagged))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for SourceTagStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::test::TestMemoryExec;
    use futures::TryStreamExt;

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("_rowid", DataType::UInt64, true),
        ]))
    }

    fn batch(ids: &[i32], row_ids: &[Option<u64>]) -> RecordBatch {
        RecordBatch::try_new(
            input_schema(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(UInt64Array::from(row_ids.to_vec())),
            ],
        )
        .unwrap()
    }

    async fn run(
        b: RecordBatch,
        generation: LsmGeneration,
        polarity: FreshnessPolarity,
    ) -> Vec<RecordBatch> {
        let schema = b.schema();
        let input = TestMemoryExec::try_new_exec(&[vec![b]], schema, None).unwrap();
        let exec = LsmSourceTagExec::new(input, generation, polarity, "_rowid");
        let ctx = SessionContext::new();
        let stream = exec.execute(0, ctx.task_ctx()).unwrap();
        stream.try_collect().await.unwrap()
    }

    fn columns(batches: &[RecordBatch]) -> (Vec<u64>, Vec<Option<u64>>) {
        let mut gens = Vec::new();
        let mut fresh = Vec::new();
        for b in batches {
            let g = b
                .column_by_name(MEMTABLE_GEN_COLUMN)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let f = b
                .column_by_name(FRESHNESS_COLUMN)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            for i in 0..b.num_rows() {
                gens.push(g.value(i));
                fresh.push(if f.is_null(i) { None } else { Some(f.value(i)) });
            }
        }
        (gens, fresh)
    }

    #[tokio::test]
    async fn insert_order_passes_row_id_through() {
        let b = batch(&[1, 2, 3], &[Some(0), Some(5), Some(99)]);
        let out = run(
            b,
            LsmGeneration::memtable(7),
            FreshnessPolarity::InsertOrder,
        )
        .await;
        let (gens, fresh) = columns(&out);
        assert_eq!(gens, vec![7, 7, 7]);
        assert_eq!(fresh, vec![Some(0), Some(5), Some(99)]);
    }

    #[tokio::test]
    async fn reverse_write_flips_row_id() {
        let b = batch(&[1, 2, 3], &[Some(0), Some(5), Some(99)]);
        let out = run(
            b,
            LsmGeneration::memtable(2),
            FreshnessPolarity::ReverseWrite,
        )
        .await;
        let (gens, fresh) = columns(&out);
        assert_eq!(gens, vec![2, 2, 2]);
        // Under reverse-write, smaller row_id = newer ⇒ larger _freshness.
        assert_eq!(
            fresh,
            vec![Some(u64::MAX), Some(u64::MAX - 5), Some(u64::MAX - 99)],
        );
    }

    #[tokio::test]
    async fn null_row_id_yields_null_freshness() {
        let b = batch(&[1, 2], &[None, Some(3)]);
        let out = run(
            b,
            LsmGeneration::memtable(1),
            FreshnessPolarity::ReverseWrite,
        )
        .await;
        let (_, fresh) = columns(&out);
        assert_eq!(fresh, vec![None, Some(u64::MAX - 3)]);
    }

    #[tokio::test]
    async fn base_table_generation_is_zero() {
        let b = batch(&[1], &[Some(0)]);
        let out = run(b, LsmGeneration::BASE_TABLE, FreshnessPolarity::InsertOrder).await;
        let (gens, _) = columns(&out);
        assert_eq!(gens, vec![0]);
    }

    #[tokio::test]
    async fn empty_batch_passthrough() {
        let schema = input_schema();
        let empty = RecordBatch::new_empty(schema);
        let out = run(
            empty,
            LsmGeneration::memtable(1),
            FreshnessPolarity::InsertOrder,
        )
        .await;
        let total: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }
}
