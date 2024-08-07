// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_buffer::bit_util;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, Stream};
use futures::{stream, FutureExt, TryFutureExt};
use futures::{StreamExt, TryStreamExt};
use lance_arrow::SchemaExt;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::StreamTracingExt;
use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::Fragment;
use log::debug;

use crate::dataset::fragment::{FileFragment, FragmentReader};
use crate::dataset::scanner::DEFAULT_FRAGMENT_READAHEAD;
use crate::dataset::Dataset;
use crate::datatypes::Schema;
use crate::utils::default_deadlock_prevention_timeout;

async fn open_file(
    file_fragment: FileFragment,
    projection: Arc<Schema>,
    with_row_id: bool,
    with_row_address: bool,
    with_make_deletions_null: bool,
    scan_scheduler: Option<Arc<ScanScheduler>>,
) -> Result<FragmentReader> {
    let mut reader = file_fragment
        .open(
            projection.as_ref(),
            with_row_id,
            with_row_address,
            scan_scheduler,
        )
        .await?;

    if with_make_deletions_null {
        reader.with_make_deletions_null();
    };
    Ok(reader)
}

/// Dataset Scan Node.
pub struct LanceStream {
    inner_stream: stream::BoxStream<'static, Result<RecordBatch>>,

    /// Manifest of the dataset
    projection: Arc<Schema>,

    with_row_id: bool,

    with_row_address: bool,
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
    ///  - ***with_row_address***: load row address from the datasets.
    ///  - ***with_make_deletions_null***: make deletions null.
    ///  - ***scan_in_order***: whether to scan the fragments in the provided order.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        read_size: usize,
        batch_readahead: usize,
        fragment_readahead: Option<usize>,
        io_buffer_size: u64,
        with_row_id: bool,
        with_row_address: bool,
        with_make_deletions_null: bool,
        scan_in_order: bool,
    ) -> Result<Self> {
        let is_v2_scan = fragments
            .iter()
            .filter_map(|frag| frag.files.first().map(|f| !f.is_legacy_file()))
            .next()
            .unwrap_or(false);
        if is_v2_scan {
            Self::try_new_v2(
                dataset,
                fragments,
                projection,
                read_size,
                fragment_readahead,
                with_row_id,
                with_row_address,
                with_make_deletions_null,
                io_buffer_size,
            )
        } else {
            Self::try_new_v1(
                dataset,
                fragments,
                projection,
                read_size,
                batch_readahead,
                fragment_readahead.unwrap_or(DEFAULT_FRAGMENT_READAHEAD),
                with_row_id,
                with_row_address,
                with_make_deletions_null,
                scan_in_order,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_v2(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        batch_size: usize,
        fragment_parallelism: Option<usize>,
        with_row_id: bool,
        with_row_address: bool,
        with_make_deletions_null: bool,
        io_buffer_size: u64,
    ) -> Result<Self> {
        let project_schema = projection.clone();
        let io_parallelism = dataset.object_store.io_parallelism()?;
        let frag_parallelism = fragment_parallelism.unwrap_or_else(|| {
            // This is somewhat aggressive.  It assumes a single page per column.  If there are many pages per
            // column then we probably don't need to read that many files.  It's a little tricky to get the right
            // answer though and so we err on the side of speed over memory.  Users can tone down fragment_parallelism
            // by hand if needed.
            if projection.fields.is_empty() {
                io_parallelism as usize
            } else {
                bit_util::ceil(io_parallelism as usize, projection.fields.len())
            }
        });
        debug!(
            "Given io_parallelism={} and num_columns={} we will read {} fragments at once while scanning v2 dataset",
            io_parallelism,
            projection.fields.len(),
            frag_parallelism
        );

        let file_fragments = fragments
            .iter()
            .map(|fragment| FileFragment::new(dataset.clone(), fragment.clone()))
            .collect::<Vec<_>>();

        let scan_scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: io_buffer_size,
                deadlock_prevention_timeout: default_deadlock_prevention_timeout(),
            },
        );

        let batches = stream::iter(file_fragments)
            .map(move |file_fragment| {
                let project_schema = project_schema.clone();
                let scan_scheduler = scan_scheduler.clone();
                #[allow(clippy::type_complexity)]
                let frag_task: BoxFuture<
                    Result<BoxStream<Result<BoxFuture<Result<RecordBatch>>>>>,
                > = tokio::spawn(async move {
                    let reader = open_file(
                        file_fragment,
                        project_schema,
                        with_row_id,
                        with_row_address,
                        with_make_deletions_null,
                        Some(scan_scheduler),
                    )
                    .await?;
                    let batch_stream = reader.read_all(batch_size as u32)?.boxed();
                    let batch_stream: BoxStream<Result<BoxFuture<Result<RecordBatch>>>> =
                        batch_stream
                            .map(|fut| {
                                Result::Ok(
                                    fut.map_err(|e| DataFusionError::External(Box::new(e)))
                                        .boxed(),
                                )
                            })
                            .boxed();
                    Result::Ok(batch_stream)
                })
                .map(|res_res| res_res.unwrap())
                .boxed();
                Ok(frag_task)
            })
            // We need two levels of try_buffered here.  The first kicks off the tasks to read the fragments.
            // As soon as we open the fragment we will start scheduling and that will kick off many background
            // tasks (not tracked by this stream) to read I/O.  The limit here is really to limit how many open
            // files we have.  It's not going to have much affect on how much RAM we are using.
            .try_buffered(frag_parallelism)
            .boxed();
        let batches = batches
            .try_flatten()
            // The second try_buffered controls how many CPU decode tasks we kick off in parallel.
            //
            // TODO: Ideally this will eventually get tied into datafusion as a # of partitions.  This will let
            // us fully fuse decode into the first half of the plan.  Currently there is likely to be a thread
            // transfer between the two steps.
            .try_buffered(get_num_compute_intensive_cpus())
            .stream_in_current_span()
            .boxed();

        Ok(Self {
            inner_stream: batches,
            projection,
            with_row_id,
            with_row_address,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_v1(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        read_size: usize,
        batch_readahead: usize,
        fragment_readahead: usize,
        with_row_id: bool,
        with_row_address: bool,
        with_make_deletions_null: bool,
        scan_in_order: bool,
    ) -> Result<Self> {
        let project_schema = projection.clone();
        debug!(
            "Scanning v1 dataset with frag_readahead={} and batch_readahead={}",
            fragment_readahead, batch_readahead
        );

        let file_fragments = fragments
            .iter()
            .map(|fragment| FileFragment::new(dataset.clone(), fragment.clone()))
            .collect::<Vec<_>>();

        let inner_stream = if scan_in_order {
            let readers = stream::iter(file_fragments)
                .map(move |file_fragment| {
                    Ok(open_file(
                        file_fragment,
                        project_schema.clone(),
                        with_row_id,
                        with_row_address,
                        with_make_deletions_null,
                        None,
                    ))
                })
                .try_buffered(fragment_readahead);
            let tasks = readers.and_then(move |reader| {
                std::future::ready(
                    reader
                        .read_all(read_size as u32)
                        .map(|task_stream| task_stream.map(Ok))
                        .map_err(DataFusionError::from),
                )
            });
            tasks
                // We must be waiting to finish a file before moving onto thenext. That's an issue.
                .try_flatten()
                // We buffer up to `batch_readahead` batches across all streams.
                .try_buffered(batch_readahead)
                .stream_in_current_span()
                .boxed()
        } else {
            let readers = stream::iter(file_fragments)
                .map(move |file_fragment| {
                    Ok(open_file(
                        file_fragment,
                        project_schema.clone(),
                        with_row_id,
                        with_row_address,
                        with_make_deletions_null,
                        None,
                    ))
                })
                .try_buffered(fragment_readahead);
            let tasks = readers.and_then(move |reader| {
                std::future::ready(
                    reader
                        .read_all(read_size as u32)
                        .map(|task_stream| task_stream.map(Ok))
                        .map_err(DataFusionError::from),
                )
            });
            // When we flatten the streams (one stream per fragment), we allow
            // `fragment_readahead` stream to be read concurrently.
            tasks
                .try_flatten_unordered(fragment_readahead)
                // We buffer up to `batch_readahead` batches across all streams.
                .try_buffer_unordered(batch_readahead)
                .stream_in_current_span()
                .boxed()
        };

        let inner_stream = inner_stream
            .map(|batch| batch.map_err(DataFusionError::from))
            .boxed();

        Ok(Self {
            inner_stream,
            projection,
            with_row_id,
            with_row_address,
        })
    }
}

impl core::fmt::Debug for LanceStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceStream")
            .field("projection", &self.projection)
            .field("with_row_id", &self.with_row_id)
            .field("with_row_address", &self.with_row_address)
            .finish()
    }
}

impl RecordBatchStream for LanceStream {
    fn schema(&self) -> SchemaRef {
        let mut schema: ArrowSchema = self.projection.as_ref().into();
        if self.with_row_id {
            schema = schema.try_with_column(ROW_ID_FIELD.clone()).unwrap();
        }
        if self.with_row_address {
            schema = schema.try_with_column(ROW_ADDR_FIELD.clone()).unwrap();
        }
        Arc::new(schema)
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
    fragment_readahead: Option<usize>,
    io_buffer_size: u64,
    with_row_id: bool,
    with_row_address: bool,
    with_make_deletions_null: bool,
    ordered_output: bool,
    output_schema: Arc<ArrowSchema>,
    properties: PlanProperties,
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
                    "LanceScan: uri={}, projection=[{}], row_id={}, row_addr={}, ordered={}",
                    self.dataset.data_dir(),
                    columns,
                    self.with_row_id,
                    self.with_row_address,
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
        fragment_readahead: Option<usize>,
        io_buffer_size: u64,
        with_row_id: bool,
        with_row_address: bool,
        with_make_deletions_null: bool,
        ordered_ouput: bool,
    ) -> Self {
        let mut output_schema: ArrowSchema = projection.as_ref().into();

        if with_row_id {
            output_schema = output_schema.try_with_column(ROW_ID_FIELD.clone()).unwrap();
        }
        if with_row_address {
            output_schema = output_schema
                .try_with_column(ROW_ADDR_FIELD.clone())
                .unwrap();
        }
        let output_schema = Arc::new(output_schema);

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::RoundRobinBatch(1),
            datafusion::physical_plan::ExecutionMode::Bounded,
        );
        Self {
            dataset,
            fragments,
            projection,
            read_size,
            batch_readahead,
            fragment_readahead,
            io_buffer_size,
            with_row_id,
            with_row_address,
            with_make_deletions_null,
            ordered_output: ordered_ouput,
            output_schema,
            properties,
        }
    }
}

impl ExecutionPlan for LanceScanExec {
    fn name(&self) -> &str {
        "LanceScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    /// Scan is the leaf node, so returns an empty vector.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "LanceScanExec cannot be assigned children".to_string(),
            ))
        }
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
            self.io_buffer_size,
            self.with_row_id,
            self.with_row_address,
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

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}
