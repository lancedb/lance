// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    fmt,
    sync::{Arc, Mutex},
    time::Instant,
};

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{streaming::StreamingTable, Session},
    dataframe::DataFrame,
    datasource::TableProvider,
    error::DataFusionError,
    execution::{context::SessionContext, TaskContext},
    logical_expr::{Expr, TableProviderFilterPushDown, TableType},
    physical_plan::{
        streaming::PartitionStream, 
        ExecutionPlan, 
        SendableRecordBatchStream,
        DisplayAs, 
        DisplayFormatType,
        PlanProperties,
        stream::RecordBatchStreamAdapter,
        metrics::MetricValue,
    },
};
use lance_arrow::SchemaExt;
use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD};
use lance_core::utils::futures::FinallyStreamExt;
use lance_core::utils::tracing::{TRACE_DATAFUSION, EXECUTION_PLAN_RUN};

use crate::Dataset;


#[derive(Debug)]
pub struct LanceTableProvider {
    dataset: Arc<Dataset>,
    full_schema: Arc<Schema>,
    row_id_idx: Option<usize>,
    row_addr_idx: Option<usize>,
    ordered: bool,
}

impl LanceTableProvider {
    pub fn new(dataset: Arc<Dataset>, with_row_id: bool, with_row_addr: bool) -> Self {
        Self::new_with_ordering(dataset, with_row_id, with_row_addr, true)
    }

    pub fn new_with_ordering(
        dataset: Arc<Dataset>,
        with_row_id: bool,
        with_row_addr: bool,
        ordered: bool,
    ) -> Self {
        let mut full_schema = Schema::from(dataset.schema());
        let mut row_id_idx = None;
        let mut row_addr_idx = None;
        if with_row_id {
            full_schema = full_schema.try_with_column(ROW_ID_FIELD.clone()).unwrap();
            row_id_idx = Some(full_schema.fields.len() - 1);
        }
        if with_row_addr {
            full_schema = full_schema.try_with_column(ROW_ADDR_FIELD.clone()).unwrap();
            row_addr_idx = Some(full_schema.fields.len() - 1);
        }
        Self {
            dataset,
            full_schema: Arc::new(full_schema),
            row_id_idx,
            row_addr_idx,
            ordered,
        }
    }

    pub fn dataset(&self) -> Arc<Dataset> {
        self.dataset.clone()
    }
}

#[async_trait]
impl TableProvider for LanceTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.full_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let mut scan = self.dataset.scan();
        match projection {
            Some(projection) if projection.is_empty() => {
                scan.empty_project()?;
            }
            Some(projection) => {
                let mut columns = Vec::with_capacity(projection.len());
                for field_idx in projection {
                    if Some(*field_idx) == self.row_id_idx {
                        scan.with_row_id();
                    } else if Some(*field_idx) == self.row_addr_idx {
                        scan.with_row_address();
                    } else {
                        columns.push(self.full_schema.field(*field_idx).name());
                    }
                }
                if !columns.is_empty() {
                    scan.project(&columns)?;
                }
            }
            _ => {}
        }

        let combined_filter = match filters.len() {
            0 => None,
            1 => Some(filters[0].clone()),
            _ => {
                let mut expr = filters[0].clone();
                for filter in &filters[1..] {
                    expr = Expr::and(expr, filter.clone());
                }
                Some(expr)
            }
        };
        if let Some(combined_filter) = combined_filter {
            scan.filter_expr(combined_filter);
        }
        scan.limit(limit.map(|l| l as i64), None)?;
        scan.scan_in_order(self.ordered);

        let plan = scan.create_plan().await.map_err(DataFusionError::from)?;
        
       
        // This will emit events to lance::datafusion target when enabled
        if std::env::var("LANCE_DATAFUSION_TRACING").is_ok() {
            Ok(Arc::new(TracedLanceExec::new(plan)))
        } else {
            Ok(plan)
        }
    }

    // Since we are using datafusion itself to apply the filters it should
    // be safe to assume that we can exactly apply any of the given pushdown
    // filters.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Exact)
            .collect())
    }
}

/// Statistics collected from DataFusion execution plans
#[derive(Debug, Clone, Default)]
pub struct DataFusionPlanStats {
    pub execution_time_ms: u64,
    pub io_stats: IoStats,
    pub scan_stats: ScanStats,
}

#[derive(Debug, Clone, Default)]
pub struct IoStats {
    pub iops: u64,
    pub requests: u64,
    pub bytes_read: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ScanStats {
    pub rows_scanned: u64,
    pub fragments_scanned: u64,
    pub ranges_scanned: u64,
}
/// A wrapper around ExecutionPlan that adds lance::datafusion tracing events
#[derive(Debug)]
struct TracedLanceExec {
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
    start_time: Arc<Mutex<Option<Instant>>>,
}

impl TracedLanceExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            properties: input.properties().clone(),
            input,
            start_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Collect comprehensive metrics from the execution plan
    fn collect_plan_stats(&self, start_time: Instant) -> DataFusionPlanStats {
        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        let mut stats = DataFusionPlanStats {
            execution_time_ms,
            ..Default::default()
        };
        
        // Recursively collect metrics from all plan nodes
        self.collect_metrics_recursive(self.input.as_ref(), &mut stats);
        
        
        stats
    }
    
    /// Recursively collect metrics from execution plan nodes
    fn collect_metrics_recursive(&self, plan: &dyn ExecutionPlan, stats: &mut DataFusionPlanStats) {
        if let Some(metrics) = plan.metrics() {
            for metric in metrics.iter() {
                match metric.value() {
                    MetricValue::Count { name, count } => {
                        match name.as_ref() {
                            "iops" => stats.io_stats.iops += count.value() as u64,
                            "requests" => stats.io_stats.requests += count.value() as u64,
                            "bytes_read" => stats.io_stats.bytes_read += count.value() as u64,
                            "rows_scanned" => stats.scan_stats.rows_scanned += count.value() as u64,
                            "fragments_scanned" => stats.scan_stats.fragments_scanned += count.value() as u64,
                            "ranges_scanned" => stats.scan_stats.ranges_scanned += count.value() as u64,
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        
        // Recursively process children
        for child in plan.children() {
            self.collect_metrics_recursive(child.as_ref(), stats);
        }
    }
}

impl DisplayAs for TracedLanceExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "TracedLanceExec")
            }
        }
    }
}

impl ExecutionPlan for TracedLanceExec {
    fn name(&self) -> &str {
        "TracedLanceExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
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
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(children[0].clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        // Record start time
        let start_time = Instant::now();
        *self.start_time.lock().unwrap() = Some(start_time);
        
        let stream = self.input.execute(partition, context)?;
        let schema = stream.schema();
        
        let traced_exec = Arc::new(TracedLanceExec::new(self.input.clone()));
        let traced_stream = stream.finally(move || {
           
            let stats = traced_exec.collect_plan_stats(start_time);
            
            // Emit tracing event following Lance conventions
            tracing::trace!(
                r#type = EXECUTION_PLAN_RUN,
                output_rows = stats.scan_stats.rows_scanned,
                iops = stats.io_stats.iops,
                requests = stats.io_stats.requests,
                bytes_read = stats.io_stats.bytes_read,
                rows_scanned = stats.scan_stats.rows_scanned,
                fragments_scanned = stats.scan_stats.fragments_scanned,
                ranges_scanned = stats.scan_stats.ranges_scanned,
                execution_time_ms = stats.execution_time_ms,
            )
        });
        
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, traced_stream)))
    }
}

pub trait SessionContextExt {
    /// Creates a DataFrame for reading a Lance dataset
    fn read_lance(
        &self,
        dataset: Arc<Dataset>,
        with_row_id: bool,
        with_row_addr: bool,
    ) -> datafusion::common::Result<DataFrame>;
    /// Creates a DataFrame for reading a Lance dataset without ordering
    fn read_lance_unordered(
        &self,
        dataset: Arc<Dataset>,
        with_row_id: bool,
        with_row_addr: bool,
    ) -> datafusion::common::Result<DataFrame>;
    /// Creates a DataFrame for reading a stream of data
    ///
    /// This dataframe may only be queried once, future queries will fail
    fn read_one_shot(
        &self,
        data: SendableRecordBatchStream,
    ) -> datafusion::common::Result<DataFrame>;
}

struct OneShotPartitionStream {
    data: Arc<Mutex<Option<SendableRecordBatchStream>>>,
    schema: Arc<Schema>,
}

impl OneShotPartitionStream {
    fn new(data: SendableRecordBatchStream) -> Self {
        let schema = data.schema();
        Self {
            data: Arc::new(Mutex::new(Some(data))),
            schema,
        }
    }
}

impl std::fmt::Debug for OneShotPartitionStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OneShotPartitionStream")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PartitionStream for OneShotPartitionStream {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let mut stream = self.data.lock().unwrap();
        stream
            .take()
            .expect("Attempt to consume a one shot dataframe multiple times")
    }
}

impl SessionContextExt for SessionContext {
    fn read_lance(
        &self,
        dataset: Arc<Dataset>,
        with_row_id: bool,
        with_row_addr: bool,
    ) -> datafusion::common::Result<DataFrame> {
        self.read_table(Arc::new(LanceTableProvider::new(
            dataset,
            with_row_id,
            with_row_addr,
        )))
    }

    fn read_lance_unordered(
        &self,
        dataset: Arc<Dataset>,
        with_row_id: bool,
        with_row_addr: bool,
    ) -> datafusion::common::Result<DataFrame> {
        self.read_table(Arc::new(LanceTableProvider::new_with_ordering(
            dataset,
            with_row_id,
            with_row_addr,
            false,
        )))
    }

    fn read_one_shot(
        &self,
        data: SendableRecordBatchStream,
    ) -> datafusion::common::Result<DataFrame> {
        let schema = data.schema();
        let part_stream = Arc::new(OneShotPartitionStream::new(data));
        let provider = StreamingTable::try_new(schema, vec![part_stream])?;
        self.read_table(Arc::new(provider))
    }
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use arrow::{
        array::AsArray,
        datatypes::{Int32Type, Int64Type},
    };
    use datafusion::prelude::SessionContext;
    use lance_datagen::array;
    use tempfile::tempdir;

    use crate::{
        datafusion::LanceTableProvider,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount},
    };

    #[tokio::test]
    pub async fn test_table_provider() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let data = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                test_uri,
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let ctx = SessionContext::new();

        ctx.register_table(
            "foo",
            Arc::new(LanceTableProvider::new(Arc::new(data), true, true)),
        )
        .unwrap();

        let df = ctx
            .sql("SELECT SUM(x) FROM foo WHERE y > 100")
            .await
            .unwrap();

        let results = df.collect().await.unwrap();
        assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        assert_eq!(results.num_columns(), 1);
        assert_eq!(results.num_rows(), 1);
        // SUM(0..100) - SUM(0..50) = 3675
        assert_eq!(results.column(0).as_primitive::<Int64Type>().value(0), 3675);
    }
}
