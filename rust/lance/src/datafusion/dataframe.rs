// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    fmt,
    sync::{Arc, Mutex},
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
    },
};
use lance_arrow::SchemaExt;
use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD};
use lance_core::utils::futures::FinallyStreamExt;
use lance_core::utils::tracing::{TRACE_EXECUTION, EXECUTION_PLAN_RUN};
use tracing;

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
        
        // Wrap the plan with tracing to emit lance::execution events
        Ok(Arc::new(TracedLanceExec::new(plan)))
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

/// A wrapper around ExecutionPlan that adds lance::execution tracing events
#[derive(Debug)]
struct TracedLanceExec {
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
}

impl TracedLanceExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            properties: input.properties().clone(),
            input,
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
        let stream = self.input.execute(partition, context)?;
        let schema = stream.schema();
        
        // Clone the input plan to access metrics after stream completion
        let plan_for_metrics = self.input.clone();
        
        let traced_stream = stream.finally(move || {
            // Just print the metrics object directly
            if let Some(metrics) = plan_for_metrics.metrics() {
                println!("Metrics: {:?}", metrics);
            } else {
                println!("No metrics available");
            }
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
