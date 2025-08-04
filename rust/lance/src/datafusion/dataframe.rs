// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
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
    physical_plan::{streaming::PartitionStream, ExecutionPlan, SendableRecordBatchStream},
};
use lance_arrow::SchemaExt;
use lance_core::{ROW_ADDR_FIELD, ROW_ID, ROW_ID_FIELD};

use crate::Dataset;

/// Check if an expression references the _rowid column
fn expr_references_rowid(expr: &Expr) -> bool {
    match expr {
        Expr::Column(col) => col.name == ROW_ID,
        Expr::BinaryExpr(binary) => {
            expr_references_rowid(&binary.left) || expr_references_rowid(&binary.right)
        }
        Expr::Not(inner) | Expr::IsNotNull(inner) | Expr::IsNull(inner) => {
            expr_references_rowid(inner)
        }
        Expr::InList(in_list) => {
            expr_references_rowid(&in_list.expr)
                || in_list.list.iter().any(|e| expr_references_rowid(e))
        }
        Expr::Case(case) => {
            case.expr
                .as_ref()
                .map_or(false, |e| expr_references_rowid(e))
                || case
                    .when_then_expr
                    .iter()
                    .any(|(when, then)| expr_references_rowid(when) || expr_references_rowid(then))
                || case
                    .else_expr
                    .as_ref()
                    .map_or(false, |e| expr_references_rowid(e))
        }
        _ => false,
    }
}

#[derive(Debug)]
pub struct LanceTableProvider {
    dataset: Arc<Dataset>,
    full_schema: Arc<Schema>,
    row_id_idx: Option<usize>,
    row_addr_idx: Option<usize>,
    ordered: bool,
    /// Track if row_id was explicitly requested vs auto-included for SQL compatibility
    explicit_row_id: bool,
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
        let row_id_idx;
        let mut row_addr_idx = None;

        // Always include _rowid in the schema to support SQL queries
        // This ensures SQL queries can reference _rowid without schema errors
        if full_schema.column_with_name(ROW_ID).is_none() {
            full_schema = full_schema.try_with_column(ROW_ID_FIELD.clone()).unwrap();
            row_id_idx = Some(full_schema.fields.len() - 1);
        } else {
            row_id_idx = full_schema.fields.iter().position(|f| f.name() == ROW_ID);
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
            explicit_row_id: with_row_id,
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

        // Check if _rowid is needed either in projection or filters
        let mut needs_row_id = self.explicit_row_id;

        // Check if _rowid is in the projection
        if let Some(projection) = projection {
            let mut columns = Vec::with_capacity(projection.len());
            for field_idx in projection {
                if Some(*field_idx) == self.row_id_idx {
                    needs_row_id = true;
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

        // Check if _rowid is referenced in filters
        for filter in filters {
            if expr_references_rowid(filter) {
                needs_row_id = true;
                break;
            }
        }

        // Enable row_id if needed
        if needs_row_id {
            scan.with_row_id();
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

        scan.create_plan().await.map_err(DataFusionError::from)
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
        let data = lance_datagen::gen()
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
