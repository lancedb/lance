// Copyright 2023 Lance Developers.
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

use std::any::Any;
use std::cmp::min;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema as ArrowSchema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};
use futures::stream::Stream;
use futures::{stream, Future};
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::tracing::StreamTracingExt;
use lance_core::ROW_ID_FIELD;
use tracing::Instrument;

use crate::dataset::fragment::{FileFragment, FragmentReader};
use crate::dataset::Dataset;
use crate::datatypes::Schema;
use crate::format::Fragment;

async fn open_file(
    file_fragment: FileFragment,
    projection: Arc<Schema>,
    with_row_id: bool,
    with_make_deletions_null: bool,
) -> Result<FragmentReader> {
    let mut reader = file_fragment.open(projection.as_ref()).await?;
    if with_row_id {
        reader.with_row_id();
    };
    if with_make_deletions_null {
        reader.with_make_deletions_null();
    };
    Ok(reader)
}

/// Convert a [`FragmentReader`] into a [`Stream`] of [`RecordBatch`].
fn scan_batches(
    reader: FragmentReader,
    read_size: usize,
) -> impl Stream<Item = Result<impl Future<Output = Result<RecordBatch>> + Send>> {
    // To make sure the reader lives long enough, we put it in an Arc.
    let reader = Arc::new(reader);
    let reader2 = reader.clone();

    let read_params_iter = (0..reader.num_batches()).flat_map(move |batch_id| {
        let rows_in_batch = reader.num_rows_in_batch(batch_id);
        (0..rows_in_batch)
            .step_by(read_size)
            .map(move |start| (batch_id, start..min(start + read_size, rows_in_batch)))
    });
    let batch_stream = stream::iter(read_params_iter).map(move |(batch_id, range)| {
        let reader = reader2.clone();
        // The Ok here is only here because try_flatten_unordered wants both the
        // outer *and* inner stream to be TryStream.
        let task = tokio::task::spawn(
            async move {
                reader
                    .read_batch(batch_id, range)
                    .await
                    .map_err(DataFusionError::from)
            }
            .in_current_span(),
        );

        Ok(async move { task.await.unwrap() })
    });

    Box::pin(batch_stream)
}

/// Dataset Scan Node.
pub struct LanceStream {
    inner_stream: stream::BoxStream<'static, Result<RecordBatch>>,

    /// Manifest of the dataset
    projection: Arc<Schema>,

    with_row_id: bool,
}

impl LanceStream {
    /// Create a new dataset scan node.
    ///
    /// Parameters
    ///
    ///  - ***dataset***: The source dataset.
    ///  - ***projection***: the projection [Schema].
    ///  - ***filter***: filter [`PhysicalExpr`], optional.
    ///  - ***read_size***: the number of rows to read for each request.
    ///  - ***batch_readahead***: the number of batches to read ahead.
    ///  - ***fragment_readahead***: the number of fragments to read ahead (only
    ///    if scan_in_order = false).
    ///  - ***with_row_id***: load row ID from the datasets.
    ///  - ***scan_in_order***: whether to scan the fragments in the provided order.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        read_size: usize,
        batch_readahead: usize,
        fragment_readahead: usize,
        with_row_id: bool,
        with_make_deletions_null: bool,
        scan_in_order: bool,
    ) -> Result<Self> {
        let project_schema = projection.clone();

        let file_fragments = fragments
            .iter()
            .map(|fragment| FileFragment::new(dataset.clone(), fragment.clone()))
            .collect::<Vec<_>>();

        let inner_stream = if scan_in_order {
            stream::iter(file_fragments)
                .map(move |file_fragment| {
                    Ok(open_file(
                        file_fragment,
                        project_schema.clone(),
                        with_row_id,
                        with_make_deletions_null,
                    ))
                })
                .try_buffered(fragment_readahead)
                .map_ok(move |reader| scan_batches(reader, read_size))
                // We must be waiting to finish a file before moving onto thenext. That's an issue.
                .try_flatten()
                // We buffer up to `batch_readahead` batches across all streams.
                .try_buffered(batch_readahead)
                .stream_in_current_span()
                .boxed()
        } else {
            stream::iter(file_fragments)
                .map(move |file_fragment| {
                    Ok(open_file(
                        file_fragment,
                        project_schema.clone(),
                        with_row_id,
                        with_make_deletions_null,
                    ))
                })
                .try_buffered(fragment_readahead)
                .map_ok(move |reader| scan_batches(reader, read_size))
                // When we flatten the streams (one stream per fragment), we allow
                // `fragment_readahead` stream to be read concurrently.
                .try_flatten_unordered(fragment_readahead)
                // We buffer up to `batch_readahead` batches across all streams.
                .try_buffer_unordered(batch_readahead)
                .stream_in_current_span()
                .boxed()
        };

        Ok(Self {
            inner_stream,
            projection,
            with_row_id,
        })
    }
}

impl core::fmt::Debug for LanceStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceStream")
            .field("projection", &self.projection)
            .field("with_row_id", &self.with_row_id)
            .finish()
    }
}

impl RecordBatchStream for LanceStream {
    fn schema(&self) -> SchemaRef {
        let schema: ArrowSchema = self.projection.as_ref().into();
        if self.with_row_id {
            let mut fields: Vec<Arc<Field>> = schema.fields.to_vec();
            fields.push(Arc::new(ROW_ID_FIELD.clone()));
            Arc::new(ArrowSchema::new(fields))
        } else {
            Arc::new(schema)
        }
    }
}

impl Stream for LanceStream {
    type Item = std::result::Result<RecordBatch, datafusion::error::DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).inner_stream.poll_next_unpin(cx)
    }
}

/// DataFusion [ExecutionPlan] for scanning one Lance dataset
#[derive(Debug)]
pub struct LanceScanExec {
    dataset: Arc<Dataset>,
    fragments: Arc<Vec<Fragment>>,
    projection: Arc<Schema>,
    read_size: usize,
    batch_readahead: usize,
    fragment_readahead: usize,
    with_row_id: bool,
    with_make_deletions_null: bool,
    ordered_output: bool,
}

impl DisplayAs for LanceScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let columns = self
                    .projection
                    .fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "LanceScan: uri={}, projection=[{}], row_id={}, ordered={}",
                    self.dataset.data_dir(),
                    columns,
                    self.with_row_id,
                    self.ordered_output
                )
            }
        }
    }
}

impl LanceScanExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        read_size: usize,
        batch_readahead: usize,
        fragment_readahead: usize,
        with_row_id: bool,
        with_make_deletions_null: bool,
        ordered_ouput: bool,
    ) -> Self {
        Self {
            dataset,
            fragments,
            projection,
            read_size,
            batch_readahead,
            fragment_readahead,
            with_row_id,
            with_make_deletions_null,
            ordered_output: ordered_ouput,
        }
    }
}

impl ExecutionPlan for LanceScanExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        let schema: ArrowSchema = self.projection.as_ref().into();
        if self.with_row_id {
            let mut fields: Vec<Arc<Field>> = schema.fields.to_vec();
            fields.push(Arc::new(ROW_ID_FIELD.clone()));
            Arc::new(ArrowSchema::new(fields))
        } else {
            Arc::new(schema)
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        None
    }

    /// Scan is the leaf node, so returns an empty vector.
    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(LanceStream::try_new(
            self.dataset.clone(),
            self.fragments.clone(),
            self.projection.clone(),
            self.read_size,
            self.batch_readahead,
            self.fragment_readahead,
            self.with_row_id,
            self.with_make_deletions_null,
            self.ordered_output,
        )?))
    }

    fn statistics(&self) -> datafusion::error::Result<Statistics> {
        // Some fragments from older datasets might have the row count stats missing.
        let (row_count, is_exact) =
            self.fragments
                .iter()
                .fold(
                    (0, true),
                    |(row_count, is_exact), fragment| match fragment.num_rows() {
                        Some(num_rows) => (row_count + num_rows, is_exact),
                        None => (row_count, false),
                    },
                );
        let num_rows = match is_exact {
            true => Precision::Exact(row_count),
            false => Precision::Absent,
        };

        Ok(Statistics {
            num_rows,
            ..datafusion::physical_plan::Statistics::new_unknown(self.schema().as_ref())
        })
    }
}
