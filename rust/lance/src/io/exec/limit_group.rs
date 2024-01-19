// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow_array::{builder::BooleanBuilder, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion::{
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, RecordBatchStream, SendableRecordBatchStream,
    },
    scalar::ScalarValue,
};
use futures::{Stream, StreamExt};

/// An execution node that partitions on given column(s) and limits the number of rows
/// per group
///
/// This node assumes that data has already been sorted by the partition columns.  It
/// will pick the first `group_limit` rows of each partition.
#[derive(Debug)]
pub struct LimitGroupExec {
    input: Arc<dyn ExecutionPlan>,
    group_columns: Vec<String>,
    group_limit: u32,
}

impl DisplayAs for LimitGroupExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "LimitGroup: limit={} group_columns=[{}]",
                    self.group_limit,
                    self.group_columns.join(", ")
                )
            }
        }
    }
}

impl LimitGroupExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        group_limit: u32,
        group_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            group_limit,
            group_columns,
        }
    }
}

struct LimitGroupStream {
    // The indices of the columns to partition on
    column_indices: Vec<usize>,
    // The input
    input: SendableRecordBatchStream,
    // How many values are allowed per row
    limit: u32,
    // How many rows have we seen for the current value
    prev_run: u32,
    // Values from the last row
    prev_row: Option<Vec<ScalarValue>>,
}

impl LimitGroupStream {
    fn limit_group(&mut self, batch: &RecordBatch) -> datafusion::error::Result<RecordBatch> {
        if batch.num_rows() == 0 {
            return Ok(batch.clone());
        }
        let partition_cols = self
            .column_indices
            .iter()
            .map(|idx| batch.column(*idx))
            .cloned()
            .collect::<Vec<_>>();
        let partitions = arrow::compute::partition(partition_cols.as_slice())?;
        let mut filter_builder = BooleanBuilder::with_capacity(batch.num_rows());

        // If there is a previous row
        let mut remaining = if let Some(prev_row) = &self.prev_row {
            // And it's the same as the first row
            if partition_cols
                .iter()
                .zip(prev_row.iter())
                .all(
                    |(col, prev_val)| match ScalarValue::try_from_array(&col, 0) {
                        Ok(val) => val == *prev_val,
                        Err(_) => false,
                    },
                )
            {
                // Then decrement the first run of the new batch by the prev run length
                if self.limit < self.prev_run {
                    0
                } else {
                    self.limit - self.prev_run
                }
            } else {
                self.limit
            }
        } else {
            self.limit
        };
        for range in partitions.ranges() {
            let run_length = (range.end - range.start) as u32;
            let to_keep = std::cmp::min(remaining, run_length);
            filter_builder.extend(std::iter::repeat(Some(true)).take(to_keep as usize));
            let to_skip = run_length - to_keep;
            filter_builder.extend(std::iter::repeat(Some(false)).take(to_skip as usize));
            remaining = self.limit;
            self.prev_run = run_length;
        }

        self.prev_row = Some(
            partition_cols
                .iter()
                .map(|arr| ScalarValue::try_from_array(&arr, arr.len() - 1))
                .collect::<datafusion::error::Result<Vec<_>>>()?,
        );

        let filter = filter_builder.finish();
        Ok(arrow::compute::filter_record_batch(batch, &filter)?)
    }
}

impl Stream for LimitGroupStream {
    type Item = datafusion::error::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => Some(self.limit_group(&batch)),
            other => other,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't know how many groups there are.  If there is one group per row we have the same
        // upper limit and lower limit.
        self.input.size_hint()
    }
}

impl RecordBatchStream for LimitGroupStream {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
}

impl ExecutionPlan for LimitGroupExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        self.input.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        // This node won't modify the order in any way
        self.input.output_ordering()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let schema = self.schema();
        let column_indices = self
            .group_columns
            .iter()
            .map(|group_column| schema.index_of(group_column))
            .collect::<arrow::error::Result<Vec<_>>>()?;
        Ok(Box::pin(LimitGroupStream {
            input,
            column_indices,
            limit: self.group_limit,
            prev_run: 0,
            prev_row: None,
        }))
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        todo!()
    }
}
