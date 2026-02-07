// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! TopKExec - Heap-based top-K selection by score.
//!
//! Replaces SortExec + GlobalLimitExec for FTS queries. Uses a min-heap
//! of size K so that only top-K rows are retained. Rows below the heap's
//! minimum score are discarded immediately (cross-source pruning).
//!
//! Complexity: O(N log K) vs O(N log N) for full sort.

use std::any::Any;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Float32Array, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};

/// Row reference for the min-heap: batch index + row index + score.
#[derive(Clone)]
struct ScoredRow {
    batch_idx: usize,
    row_idx: usize,
    score: f32,
}

impl PartialEq for ScoredRow {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredRow {}

impl PartialOrd for ScoredRow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredRow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Heap-based top-K selection by a score column.
///
/// Consumes all input rows, maintains a min-heap of size K, and outputs
/// a single RecordBatch with the K highest-scoring rows in descending order.
///
/// This provides cross-source pruning: as the heap fills with high-scoring
/// rows from one source, rows from other sources that score below the
/// heap minimum are immediately discarded without any heap operations.
#[derive(Debug)]
pub struct TopKExec {
    input: Arc<dyn ExecutionPlan>,
    /// Number of top results to keep.
    k: usize,
    /// Name of the score column.
    score_column: String,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl TopKExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, k: usize, score_column: String) -> Self {
        let schema = input.schema();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Self {
            input,
            k,
            score_column,
            schema,
            properties,
        }
    }
}

impl DisplayAs for TopKExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "TopKExec: k={}, score={}", self.k, self.score_column)
            }
        }
    }
}

impl ExecutionPlan for TopKExec {
    fn name(&self) -> &str {
        "TopKExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
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
                "TopKExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.k,
            self.score_column.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;

        let score_idx = self.schema.index_of(&self.score_column).map_err(|_| {
            datafusion::error::DataFusionError::Internal(format!(
                "Score column '{}' not found in schema",
                self.score_column
            ))
        })?;

        Ok(Box::pin(TopKStream::new(
            input_stream,
            self.k,
            score_idx,
            self.schema.clone(),
        )))
    }
}

/// Stream that collects all input, selects top-K by score, and emits them sorted DESC.
struct TopKStream {
    /// Input stream.
    input: SendableRecordBatchStream,
    /// Number of top results to keep.
    k: usize,
    /// Column index for the score.
    score_idx: usize,
    /// Output schema.
    schema: SchemaRef,
    /// Accumulated input batches (for row gathering after selection).
    batches: Vec<RecordBatch>,
    /// Min-heap of size K (Reverse to make BinaryHeap a min-heap).
    heap: BinaryHeap<Reverse<ScoredRow>>,
    /// Current minimum score in the heap (for fast pruning).
    min_score: f32,
    /// Whether we've finished consuming input and emitted the result.
    done: bool,
}

impl TopKStream {
    fn new(
        input: SendableRecordBatchStream,
        k: usize,
        score_idx: usize,
        schema: SchemaRef,
    ) -> Self {
        Self {
            input,
            k,
            score_idx,
            schema,
            batches: Vec::new(),
            heap: BinaryHeap::with_capacity(k + 1),
            min_score: f32::NEG_INFINITY,
            done: false,
        }
    }

    /// Process a batch: for each row, compete for a spot in the top-K heap.
    fn process_batch(&mut self, batch: &RecordBatch) {
        let batch_idx = self.batches.len() - 1;
        let scores = batch
            .column(self.score_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("Score column must be Float32");

        for row_idx in 0..batch.num_rows() {
            let score = scores.value(row_idx);

            // Cross-source pruning: skip rows that can't make top-K
            if self.heap.len() >= self.k && score <= self.min_score {
                continue;
            }

            let row = ScoredRow {
                batch_idx,
                row_idx,
                score,
            };

            if self.heap.len() < self.k {
                self.heap.push(Reverse(row));
                if self.heap.len() == self.k {
                    self.min_score = self.heap.peek().unwrap().0.score;
                }
            } else {
                // Replace the minimum element
                self.heap.pop();
                self.heap.push(Reverse(row));
                self.min_score = self.heap.peek().unwrap().0.score;
            }
        }
    }

    /// Drain the heap and gather rows from stored batches into a single output batch.
    fn build_output(&mut self) -> DFResult<RecordBatch> {
        if self.heap.is_empty() || self.batches.is_empty() {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }

        // Drain heap into sorted order (score DESC)
        let mut winners: Vec<ScoredRow> = self.heap.drain().map(|r| r.0).collect();
        winners.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Gather rows from stored batches
        let num_cols = self.schema.fields().len();
        let mut column_builders: Vec<Vec<Arc<dyn arrow_array::Array>>> = (0..num_cols)
            .map(|_| Vec::with_capacity(winners.len()))
            .collect();

        for winner in &winners {
            let batch = &self.batches[winner.batch_idx];
            for (col_idx, builder) in column_builders.iter_mut().enumerate() {
                builder.push(batch.column(col_idx).slice(winner.row_idx, 1));
            }
        }

        // Concatenate sliced arrays for each column
        let columns: Vec<Arc<dyn arrow_array::Array>> = column_builders
            .into_iter()
            .map(|slices| {
                let refs: Vec<&dyn arrow_array::Array> =
                    slices.iter().map(|a| a.as_ref()).collect();
                arrow::compute::concat(&refs)
                    .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
            })
            .collect::<DFResult<Vec<_>>>()?;

        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl Stream for TopKStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }

        // Consume all input batches
        loop {
            match self.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    if batch.num_rows() > 0 {
                        self.batches.push(batch);
                        // Process the batch we just pushed (it's the last one)
                        let batch_ref = self.batches.last().unwrap().clone();
                        self.process_batch(&batch_ref);
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    self.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // All input consumed, build and emit output
                    self.done = true;
                    let result = self.build_output();
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for TopKStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
