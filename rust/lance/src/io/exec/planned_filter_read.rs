// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::EmissionType, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion_expr::Expr;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::Statistics;
use futures::TryStreamExt;
use lance_core::datatypes::Projection;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use tokio::sync::Mutex as AsyncMutex;

use crate::Dataset;

use super::filtered_read::{
    FilteredReadGlobalMetrics, FilteredReadStream, FilteredReadThreadingMode, ScopedFragmentRead,
};
use super::planner::PlannedFragmentRead;

/// Core executor that takes pre-planned fragments and uses FilteredReadStream
pub struct PlannedFilterReadExec {
    dataset: Arc<Dataset>,
    planned_fragments: Vec<PlannedFragmentRead>,
    projection: Arc<Projection>,
    with_deleted_rows: bool,
    batch_size: u32,
    full_filter: Option<Expr>,
    refine_filter: Option<Expr>,
    output_schema: SchemaRef,
    limit: Option<usize>,
    fragment_readahead: Option<usize>,
    threading_mode: FilteredReadThreadingMode,
    metrics: ExecutionPlanMetricsSet,
    properties: PlanProperties,
    // When execute is first called we will initialize the FilteredReadStream.  In order to support
    // multiple partitions, each partition will share the stream.
    running_stream: Arc<AsyncMutex<Option<FilteredReadStream>>>,
}

impl PlannedFilterReadExec {
    pub fn new(
        dataset: Arc<Dataset>,
        planned_fragments: Vec<PlannedFragmentRead>,
        projection: Arc<Projection>,
        with_deleted_rows: bool,
        batch_size: u32,
        full_filter: Option<Expr>,
        refine_filter: Option<Expr>,
        output_schema: SchemaRef,
        limit: Option<usize>,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );

        Self {
            dataset,
            planned_fragments,
            projection,
            with_deleted_rows,
            batch_size,
            full_filter,
            refine_filter,
            output_schema,
            limit,
            fragment_readahead: None,
            threading_mode: FilteredReadThreadingMode::OnePartitionMultipleThreads(1),
            metrics: ExecutionPlanMetricsSet::new(),
            properties,
            running_stream: Arc::new(AsyncMutex::new(None)),
        }
    }

    pub fn with_fragment_readahead(mut self, readahead: usize) -> Self {
        self.fragment_readahead = Some(readahead);
        self
    }

    pub fn with_threading_mode(mut self, mode: FilteredReadThreadingMode) -> Self {
        self.threading_mode = mode;
        self
    }

    /// Convert PlannedFragmentRead to ScopedFragmentRead for execution
    #[allow(dead_code)]
    fn to_scoped_fragments(&self, scan_scheduler: Arc<ScanScheduler>) -> Vec<ScopedFragmentRead> {
        self.planned_fragments
            .iter()
            .map(|p| {
                // Determine which filter to use for this fragment
                let filter = if p.use_refine {
                    self.refine_filter.clone()
                } else {
                    self.full_filter.clone()
                };

                ScopedFragmentRead {
                    fragment: p.fragment.clone(),
                    ranges: p.ranges.clone(),
                    projection: self.projection.clone(),
                    with_deleted_rows: self.with_deleted_rows,
                    batch_size: self.batch_size,
                    filter,
                    priority: p.priority,
                    scan_scheduler: scan_scheduler.clone(),
                }
            })
            .collect()
    }
}

impl std::fmt::Debug for PlannedFilterReadExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlannedFilterReadExec")
            .field("num_fragments", &self.planned_fragments.len())
            .field("limit", &self.limit)
            .finish()
    }
}

impl DisplayAs for PlannedFilterReadExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "PlannedFilterReadExec: {} fragments",
                    self.planned_fragments.len()
                )?;
                if let Some(limit) = self.limit {
                    write!(f, ", limit={}", limit)?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PlannedFilterReadExec\nfragments={}\nlimit={:?}",
                    self.planned_fragments.len(),
                    self.limit
                )
            }
        }
    }
}

impl ExecutionPlan for PlannedFilterReadExec {
    fn name(&self) -> &str {
        "PlannedFilterReadExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution(format!(
                "PlannedFilterReadExec only supports a single partition, got {}",
                partition
            )));
        }

        let dataset = self.dataset.clone();
        let running_stream = self.running_stream.clone();
        let metrics = self.metrics.clone();
        let planned_fragments = self.planned_fragments.clone();
        let output_schema = self.output_schema.clone();
        let limit = self.limit;
        let fragment_readahead = self.fragment_readahead;
        let projection = self.projection.clone();
        let with_deleted_rows = self.with_deleted_rows;
        let batch_size = self.batch_size;
        let full_filter = self.full_filter.clone();
        let refine_filter = self.refine_filter.clone();

        let stream = futures::stream::once(async move {
            let mut running_stream_lock = running_stream.lock().await;
            if let Some(running_stream) = &*running_stream_lock {
                DataFusionResult::<SendableRecordBatchStream>::Ok(
                    running_stream.get_stream(&metrics, partition),
                )
            } else {
                // Create ScanScheduler only when actually executing (not during planning)
                let scan_scheduler = ScanScheduler::new(
                    Arc::new(dataset.object_store().clone()),
                    SchedulerConfig::max_bandwidth(dataset.object_store()),
                );

                let fragment_readahead = fragment_readahead.unwrap_or(4).max(1);

                // Convert PlannedFragmentRead to ScopedFragmentRead for execution
                let scoped_fragments: Vec<ScopedFragmentRead> = planned_fragments
                    .into_iter()
                    .map(|p| {
                        // Determine which filter to use for this fragment
                        let filter = if p.use_refine {
                            refine_filter.clone()
                        } else {
                            full_filter.clone()
                        };

                        ScopedFragmentRead {
                            fragment: p.fragment,
                            ranges: p.ranges,
                            projection: projection.clone(),
                            with_deleted_rows,
                            batch_size,
                            filter,
                            priority: p.priority,
                            scan_scheduler: scan_scheduler.clone(),
                        }
                    })
                    .collect();

                // Create global metrics
                let global_metrics = Arc::new(FilteredReadGlobalMetrics::new(&metrics));

                // Create FilteredReadStream from planned fragments
                let new_running_stream = FilteredReadStream::from_planned(
                    scoped_fragments,
                    output_schema,
                    scan_scheduler,
                    global_metrics,
                    fragment_readahead,
                    limit,
                );

                let first_stream = new_running_stream.get_stream(&metrics, partition);
                *running_stream_lock = Some(new_running_stream);
                DataFusionResult::Ok(first_stream)
            }
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![],
        })
    }
}
