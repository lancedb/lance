use std::{any::Any, sync::Arc};

use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use datafusion::{
    datasource::TableProvider,
    error::Result as DatafusionResult,
    execution::context::SessionState,
    logical_expr::{LogicalPlan, TableProviderFilterPushDown, TableType},
    optimizer::utils::conjunction,
    physical_plan::{filter::FilterExec, limit::GlobalLimitExec, ExecutionPlan, Statistics},
    prelude::Expr,
};

use crate::{
    datafusion::physical_expr::column_names_in_expr,
    io::exec::{Planner, ProjectionExec},
    Dataset,
};

#[async_trait]
impl TableProvider for Dataset {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> Arc<ArrowSchema> {
        Arc::new(self.schema().into())
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn get_table_definition(&self) -> Option<&str> {
        None
    }

    fn get_logical_plan(&self) -> Option<&LogicalPlan> {
        None
    }

    async fn scan(
        &self,
        _: &SessionState,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DatafusionResult<Arc<dyn ExecutionPlan>> {
        let scanner = self.scan();
        let filter_expr = conjunction(filters.iter().cloned());
        // NOTE: we only support node that have one partition. So any nodes that
        // produce multiple need to be repartitioned to 1.
        let mut filter_expr = if let Some(filter) = filter_expr {
            let planner = Planner::new(Arc::new(scanner.dataset.schema().into()));
            Some(planner.create_physical_expr(&filter)?)
        } else {
            None
        };

        // Stage 1: source
        let mut plan: Arc<dyn ExecutionPlan> = if scanner.nearest.is_some() {
            if scanner.prefilter {
                let prefilter = filter_expr;
                filter_expr = None;
                scanner.knn(prefilter).await?
            } else {
                scanner.knn(None).await?
            }
        } else if let Some(expr) = filter_expr.as_ref() {
            let columns_in_filter = column_names_in_expr(expr.as_ref());
            let filter_schema = Arc::new(scanner.dataset.schema().project(&columns_in_filter)?);
            scanner.scan(true, false, filter_schema)
        } else {
            // Scan without filter or limits
            scanner.scan(
                scanner.with_row_id,
                false,
                scanner.projections.clone().into(),
            )
        };

        // Stage 2: filter
        if let Some(predicates) = filter_expr.as_ref() {
            let mut columns_in_filter = column_names_in_expr(predicates.as_ref());
            // If we are going to sort then grab the ordering column at the same time we grab
            // the columns we need for filtering
            if let Some(ordering) = &scanner.ordering {
                for column in ordering {
                    if !columns_in_filter.contains(&column.column_name) {
                        columns_in_filter.push(column.column_name.clone())
                    }
                }
            }
            let filter_schema = Arc::new(scanner.dataset.schema().project(&columns_in_filter)?);
            let remaining_schema = filter_schema.exclude(plan.schema().as_ref())?;
            if !remaining_schema.fields.is_empty() {
                // Not all columns for filter are ready, so we need to take them first
                plan = scanner.take(plan, &remaining_schema, scanner.batch_readahead)?;
            }
            plan = Arc::new(FilterExec::try_new(predicates.clone(), plan)?);
        }

        // Stage 3: limit / offset
        if let Some(limit) = limit {
            plan = Arc::new(GlobalLimitExec::new(plan, limit, None))
        }

        // Stage 4: take remaining columns / projection
        if let Some(projection) = projection {
            let schema = self.schema();
            let projection_columns = projection
                .into_iter()
                .map(|i| *i as i32)
                .collect::<Vec<_>>();

            let output_schema = schema.project_by_ids(&projection_columns);

            let remaining_schema = output_schema.exclude(plan.schema().as_ref())?;
            if !remaining_schema.fields.is_empty() {
                plan = scanner.take(plan, &remaining_schema, scanner.batch_readahead)?;
            }

            plan = Arc::new(ProjectionExec::try_new(plan, Arc::new(output_schema))?);
        }

        Ok(plan)
    }

    fn supports_filter_pushdown(
        &self,
        filter: &Expr,
    ) -> DatafusionResult<TableProviderFilterPushDown> {
        let schema = &self.manifest.schema;
        let planner = Planner::new(Arc::new(schema.into()));

        planner
            .create_physical_expr(filter)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
            .map(|_| TableProviderFilterPushDown::Exact)
    }

    fn statistics(&self) -> Option<Statistics> {
        // Statistics not yet implemented
        None
    }
}
