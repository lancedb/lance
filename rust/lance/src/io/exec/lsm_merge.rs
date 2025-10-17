// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution node for LSM-tree style merge of multiple datasets

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    RecordBatchStream as DFRecordBatchStream, SendableRecordBatchStream,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion_physical_expr::Partitioning;
use futures::stream::Stream;
use lance_core::ROW_ID;

use crate::Result;

/// Execution node that merges results from multiple datasets in LSM-tree style
///
/// This node takes multiple input streams (one per dataset) and merges them
/// based on the _rowid column. When duplicate _rowid values are encountered,
/// the value from the earlier stream (higher precedence) is kept.
///
/// The input streams must:
/// 1. Have the same schema
/// 2. Include the _rowid column
/// 3. Be provided in precedence order (highest to lowest)
#[derive(Debug)]
pub struct LsmMergeExec {
    /// Input execution plans, ordered by precedence (highest first)
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    /// Cached plan properties
    properties: PlanProperties,
}

impl LsmMergeExec {
    pub fn try_new(inputs: Vec<Arc<dyn ExecutionPlan>>) -> Result<Self> {
        if inputs.is_empty() {
            return Err(crate::Error::InvalidInput {
                source: "LsmMergeExec requires at least one input".into(),
                location: snafu::location!(),
            });
        }

        let schema = inputs[0].schema();

        // Validate all inputs have the same schema
        for (idx, input) in inputs.iter().enumerate().skip(1) {
            let input_schema = input.schema();
            if schema.fields() != input_schema.fields() {
                return Err(crate::Error::InvalidInput {
                    source: format!(
                        "Input {} has incompatible schema with input 0",
                        idx
                    )
                    .into(),
                    location: snafu::location!(),
                });
            }
        }

        // Validate schema contains _rowid column
        if schema.column_with_name(ROW_ID).is_none() {
            return Err(crate::Error::InvalidInput {
                source: "LsmMergeExec requires _rowid column in schema".into(),
                location: snafu::location!(),
            });
        }

        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self { inputs, properties })
    }
}

impl DisplayAs for LsmMergeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "LsmMergeExec: {} inputs", self.inputs.len())
    }
}

impl ExecutionPlan for LsmMergeExec {
    fn name(&self) -> &str {
        "LsmMergeExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.inputs[0].schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(children).map_err(|e| {
            DataFusionError::Plan(format!("Failed to create LsmMergeExec: {}", e))
        })?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution(
                "LsmMergeExec only supports single partition".to_string(),
            ));
        }

        let mut streams = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let stream = input.execute(0, context.clone())?;
            streams.push(stream);
        }

        let schema = self.schema();
        let row_id_index = schema
            .column_with_name(ROW_ID)
            .map(|(idx, _)| idx)
            .ok_or_else(|| {
                DataFusionError::Execution("_rowid column not found in schema".to_string())
            })?;

        Ok(Box::pin(LsmMergeStream::new(
            streams,
            schema,
            row_id_index,
        )))
    }
}

/// Stream that performs the LSM-tree merge
struct LsmMergeStream {
    streams: Vec<SendableRecordBatchStream>,
    schema: SchemaRef,
    row_id_index: usize,
    seen_row_ids: std::collections::HashSet<u64>,
    stream_idx: usize,
}

impl LsmMergeStream {
    fn new(
        streams: Vec<SendableRecordBatchStream>,
        schema: SchemaRef,
        row_id_index: usize,
    ) -> Self {
        Self {
            streams,
            schema,
            row_id_index,
            seen_row_ids: std::collections::HashSet::new(),
            stream_idx: 0,
        }
    }

    fn process_batch(&mut self, batch: RecordBatch) -> DFResult<Option<RecordBatch>> {
        use arrow_array::Array;
        use arrow_array::UInt64Array;

        let row_id_array = batch
            .column(self.row_id_index)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("_rowid column is not UInt64Array".to_string())
            })?;

        let mut keep_indices = Vec::new();
        for i in 0..batch.num_rows() {
            if row_id_array.is_null(i) {
                // Keep null row ids (deleted rows)
                keep_indices.push(i);
            } else {
                let row_id = row_id_array.value(i);
                if self.seen_row_ids.insert(row_id) {
                    // New row id, keep it
                    keep_indices.push(i);
                }
                // else: duplicate row_id, skip it
            }
        }

        if keep_indices.is_empty() {
            return Ok(None);
        }

        if keep_indices.len() == batch.num_rows() {
            // All rows are kept
            return Ok(Some(batch));
        }

        // Filter the batch to only include kept rows
        let indices = arrow_array::UInt32Array::from_iter_values(
            keep_indices.iter().map(|&i| i as u32),
        );
        let filtered = arrow_select::take::take_record_batch(&batch, &indices)?;
        Ok(Some(filtered))
    }
}

impl Stream for LsmMergeStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.stream_idx >= self.streams.len() {
                // All streams exhausted
                return Poll::Ready(None);
            }

            let stream_idx = self.stream_idx;
            match Pin::new(&mut self.streams[stream_idx]).poll_next(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    match self.process_batch(batch) {
                        Ok(Some(filtered_batch)) => {
                            return Poll::Ready(Some(Ok(filtered_batch)));
                        }
                        Ok(None) => {
                            // All rows filtered out, continue to next batch
                            continue;
                        }
                        Err(e) => {
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // Current stream exhausted, move to next
                    self.stream_idx += 1;
                    continue;
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

impl DFRecordBatchStream for LsmMergeStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
