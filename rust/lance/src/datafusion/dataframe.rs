// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, streaming::StreamingTable},
    common::{
        Column,
        config::ConfigOptions,
        tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    },
    dataframe::DataFrame,
    datasource::{TableProvider, provider_as_source, source_as_provider},
    error::DataFusionError,
    execution::{TaskContext, context::SessionContext},
    logical_expr::{
        Expr, LogicalPlan, TableProviderFilterPushDown, TableType,
        expr::ScalarFunction,
        logical_plan::{Projection, TableScan},
    },
    optimizer::analyzer::AnalyzerRule,
    physical_plan::{ExecutionPlan, SendableRecordBatchStream, streaming::PartitionStream},
};
use futures::{StreamExt, TryStreamExt};
use lance_arrow::SchemaExt;
use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD};
use lance_datafusion::udf::{
    CELL_FLAG_ID_UDF, CELL_FLAG_UDF, DeferredCellFlagBinding, bound_cell_flag_flag_id,
    deferred_bound_cell_flag_udf, is_unbound_cell_flag_udf,
};
use lance_table::format::Fragment;

use crate::{
    Dataset,
    dataset::cell_flag::{cell_flag_call_flag_id, load_cell_flag_fragments},
};

/// A [TableProvider] for Lance datasets.
///
/// Note: Datafusion has no concept of "system columns".  As a result, you must specify
/// which schema columns should be included in the table's schema when you create the
/// provider.
///
/// This table provider should support:
///  - Filter pushdown
///  - Limit pushdown
///  - Projection pushdown
///
/// Note that LanceDB also has a TableProvider implementation that should be preferred
/// if you are working in LanceDB.
///
/// When registering this provider manually, call
/// [`register_cell_flag_analyzer`] on the same session context before
/// using `cell_flag(field, name)`. [`SessionContextExt::read_lance`] does this
/// automatically.
#[derive(Debug, Clone)]
pub struct LanceTableProvider {
    dataset: Arc<Dataset>,
    full_schema: Arc<Schema>,
    row_id_idx: Option<usize>,
    row_addr_idx: Option<usize>,
    ordered: bool,
    blob_handling: Option<lance_core::datatypes::BlobHandling>,
    cell_flag_bindings: HashMap<u32, DeferredCellFlagBinding>,
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
            blob_handling: None,
            cell_flag_bindings: HashMap::new(),
        }
    }

    /// Overrides how blob columns are read during [`TableProvider::scan`].
    /// When unset, the underlying dataset scan uses its default
    /// [`BlobHandling`](lance_core::datatypes::BlobHandling) policy.
    pub fn with_blob_handling(mut self, handling: lance_core::datatypes::BlobHandling) -> Self {
        let converted = self
            .dataset
            .full_projection()
            .with_blob_handling(handling.clone())
            .to_bare_schema();
        let mut full_schema = Schema::from(&converted);
        if self.row_id_idx.is_some() {
            full_schema = full_schema.try_with_column(ROW_ID_FIELD.clone()).unwrap();
        }
        if self.row_addr_idx.is_some() {
            full_schema = full_schema.try_with_column(ROW_ADDR_FIELD.clone()).unwrap();
        }
        self.full_schema = Arc::new(full_schema);
        self.blob_handling = Some(handling);
        self
    }

    pub fn dataset(&self) -> Arc<Dataset> {
        self.dataset.clone()
    }

    fn with_cell_flag_bindings(
        mut self,
        bindings: HashMap<u32, DeferredCellFlagBinding>,
    ) -> datafusion::common::Result<Self> {
        if self.row_addr_idx.is_none() {
            self.full_schema = Arc::new(
                self.full_schema
                    .as_ref()
                    .clone()
                    .try_with_column(ROW_ADDR_FIELD.clone())?,
            );
            self.row_addr_idx = Some(self.full_schema.fields.len() - 1);
        }
        self.cell_flag_bindings.extend(bindings);
        Ok(self)
    }

    async fn initialize_cell_flag_bindings(
        &self,
        selected_fragments: Option<&[Fragment]>,
    ) -> datafusion::common::Result<()> {
        if self.cell_flag_bindings.is_empty() {
            return Ok(());
        }
        let selected_fragment_ids = selected_fragments.map(|fragments| {
            Arc::new(
                fragments
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect::<HashSet<_>>(),
            )
        });
        let covered_fragments: roaring::RoaringBitmap = selected_fragments
            .unwrap_or(self.dataset.fragments())
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect();
        let io_parallelism = self.dataset.object_store.io_parallelism().max(1);
        let bindings = self
            .cell_flag_bindings
            .values()
            .cloned()
            .collect::<Vec<_>>();
        futures::stream::iter(bindings)
            .map(|binding| {
                let selected_fragment_ids = selected_fragment_ids.clone();
                let covered_fragments = covered_fragments.clone();
                async move {
                    let fragments = load_cell_flag_fragments(
                        &self.dataset,
                        binding.flag_id(),
                        selected_fragment_ids,
                    )
                    .await
                    .map_err(DataFusionError::from)?;
                    binding.initialize_with_coverage(fragments, covered_fragments)
                }
            })
            .buffer_unordered(io_parallelism)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }

    fn cell_flag_limit_fragments(
        &self,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Option<Vec<Fragment>> {
        let limit = limit?;
        if !filters.is_empty() || !self.ordered || self.dataset.manifest.uses_stable_row_ids() {
            return None;
        }
        let mut selected = Vec::new();
        let mut rows = 0usize;
        for fragment in self.dataset.fragments().iter() {
            if rows >= limit {
                break;
            }
            rows = rows.checked_add(fragment.num_rows()?)?;
            selected.push(fragment.clone());
        }
        Some(selected)
    }
}

#[async_trait]
impl TableProvider for LanceTableProvider {
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
        let limit_fragments = self.cell_flag_limit_fragments(filters, limit);
        self.initialize_cell_flag_bindings(limit_fragments.as_deref())
            .await?;
        let mut scan = self.dataset.scan();
        if let Some(handling) = self.blob_handling.clone() {
            scan.blob_handling(handling);
        }

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

const CELL_FLAG_ANALYZER_NAME: &str = "bind_lance_cell_flags";

#[derive(Debug)]
struct BindCellFlags;

fn contains_unbound_cell_flag(plan: &LogicalPlan) -> datafusion::common::Result<bool> {
    let mut found = false;
    plan.apply(|node| {
        for expression in node.expressions() {
            expression.apply(|node| {
                if let Expr::ScalarFunction(function) = node
                    && is_unbound_cell_flag_udf(function.func.as_ref())
                {
                    found = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
            if found {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

fn expression_contains_bound_cell_flag(expression: &Expr) -> datafusion::common::Result<bool> {
    let mut found = false;
    expression.apply(|node| {
        if let Expr::ScalarFunction(function) = node
            && bound_cell_flag_flag_id(function.func.as_ref()).is_some()
        {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

fn contains_provider(plan: &LogicalPlan, dataset: &Arc<Dataset>) -> bool {
    let mut found = false;
    let _ = plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && let Some(provider) = provider.downcast_ref::<LanceTableProvider>()
            && Arc::ptr_eq(&provider.dataset, dataset)
        {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn project_original_schema(
    plan: LogicalPlan,
    original_schema: &datafusion::common::DFSchema,
) -> datafusion::common::Result<LogicalPlan> {
    let expressions = (0..original_schema.fields().len())
        .map(|index| Expr::Column(Column::from(original_schema.qualified_field(index))))
        .collect();
    Projection::try_new(expressions, Arc::new(plan)).map(LogicalPlan::Projection)
}

impl BindCellFlags {
    fn find_provider(
        &self,
        plan: &LogicalPlan,
    ) -> datafusion::common::Result<Option<LanceTableProvider>> {
        let mut providers = Vec::new();
        plan.apply(|node| {
            if let LogicalPlan::TableScan(scan) = node
                && let Ok(provider) = source_as_provider(&scan.source)
                && let Some(provider) = provider.downcast_ref::<LanceTableProvider>()
                && !providers.iter().any(|existing: &LanceTableProvider| {
                    Arc::ptr_eq(&existing.dataset, &provider.dataset)
                })
            {
                providers.push(provider.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        match providers.len() {
            0 => Ok(None),
            1 => Ok(providers.pop()),
            _ => Err(DataFusionError::Plan(
                "cell_flag currently requires a logical plan with exactly one Lance dataset snapshot"
                    .to_string(),
            )),
        }
    }

    fn collect_flag_ids(
        &self,
        plan: &LogicalPlan,
        dataset: &Dataset,
    ) -> datafusion::common::Result<HashSet<u32>> {
        let mut flag_ids = HashSet::new();
        let mut captured_error = None;
        plan.apply(|node| {
            for expression in node.expressions() {
                expression.apply(|expr| {
                    match cell_flag_call_flag_id(dataset, expr) {
                        Ok(Some(flag_id)) => {
                            flag_ids.insert(flag_id);
                        }
                        Ok(None) => {}
                        Err(error) => {
                            captured_error = Some(DataFusionError::from(error));
                            return Ok(TreeNodeRecursion::Stop);
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                if captured_error.is_some() {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if let Some(error) = captured_error {
            Err(error)
        } else {
            Ok(flag_ids)
        }
    }

    fn rewrite_expression(
        &self,
        expression: Expr,
        dataset: &Dataset,
        functions: &HashMap<u32, Arc<datafusion::logical_expr::ScalarUDF>>,
    ) -> datafusion::common::Result<Expr> {
        let mut captured_error = None;
        let transformed = expression.transform(|node| {
            let flag_id = match cell_flag_call_flag_id(dataset, &node) {
                Ok(Some(flag_id)) => flag_id,
                Ok(None) => return Ok(Transformed::no(node)),
                Err(error) => {
                    captured_error = Some(DataFusionError::from(error));
                    return Ok(Transformed::no(node));
                }
            };
            let function = functions.get(&flag_id).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "missing deferred cell_flag binding for flag ID {}",
                    flag_id
                ))
            })?;
            Ok(Transformed::yes(Expr::ScalarFunction(
                ScalarFunction::new_udf(
                    function.clone(),
                    vec![Expr::Column(Column::new_unqualified(lance_core::ROW_ADDR))],
                ),
            )))
        })?;
        if let Some(error) = captured_error {
            Err(error)
        } else {
            Ok(transformed.data)
        }
    }

    fn rewrite_plan(
        &self,
        plan: LogicalPlan,
        required_by_parent: bool,
        provider: &LanceTableProvider,
        replacement: &LanceTableProvider,
        functions: &HashMap<u32, Arc<datafusion::logical_expr::ScalarUDF>>,
    ) -> datafusion::common::Result<LogicalPlan> {
        let original_schema = plan.schema().as_ref().clone();
        let mut expressions = plan
            .expressions()
            .into_iter()
            .map(|expression| self.rewrite_expression(expression, &provider.dataset, functions))
            .collect::<datafusion::common::Result<Vec<_>>>()?;
        let own_requires = expressions
            .iter()
            .map(expression_contains_bound_cell_flag)
            .collect::<datafusion::common::Result<Vec<_>>>()?
            .into_iter()
            .any(|value| value);
        let needs_row_address = required_by_parent || own_requires;

        if matches!(plan, LogicalPlan::Aggregate(_) | LogicalPlan::Distinct(_))
            && required_by_parent
        {
            return Err(DataFusionError::Plan(
                "cell_flag cannot be evaluated after an aggregate or distinct operation because physical row identity is no longer available"
                    .to_string(),
            ));
        }

        let rewritten = if let LogicalPlan::TableScan(scan) = &plan {
            let source = source_as_provider(&scan.source)?;
            if let Some(scan_provider) = source.downcast_ref::<LanceTableProvider>()
                && Arc::ptr_eq(&scan_provider.dataset, &provider.dataset)
            {
                let mut projection = scan.projection.clone();
                if needs_row_address {
                    let row_addr_idx = replacement.row_addr_idx.ok_or_else(|| {
                        DataFusionError::Internal(
                            "replacement Lance provider is missing _rowaddr".to_string(),
                        )
                    })?;
                    if let Some(indices) = &mut projection
                        && !indices.contains(&row_addr_idx)
                    {
                        indices.push(row_addr_idx);
                    }
                }
                LogicalPlan::TableScan(TableScan::try_new(
                    scan.table_name.clone(),
                    provider_as_source(Arc::new(replacement.clone())),
                    projection,
                    expressions,
                    scan.fetch,
                )?)
            } else {
                plan.clone()
            }
        } else {
            let child_required = needs_row_address;
            let inputs = plan
                .inputs()
                .into_iter()
                .map(|input| {
                    let pass_row_address =
                        child_required && contains_provider(input, &provider.dataset);
                    self.rewrite_plan(
                        input.clone(),
                        pass_row_address,
                        provider,
                        replacement,
                        functions,
                    )
                })
                .collect::<datafusion::common::Result<Vec<_>>>()?;

            if required_by_parent && matches!(plan, LogicalPlan::Projection(_)) {
                expressions.push(Expr::Column(Column::new_unqualified(lance_core::ROW_ADDR)));
            }
            plan.with_new_exprs(expressions, inputs)?
        };

        if !required_by_parent
            && rewritten.schema().fields().len() != original_schema.fields().len()
        {
            project_original_schema(rewritten, &original_schema)
        } else {
            Ok(rewritten)
        }
    }
}

impl AnalyzerRule for BindCellFlags {
    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<LogicalPlan> {
        if !contains_unbound_cell_flag(&plan)? {
            return Ok(plan);
        }
        let Some(provider) = self.find_provider(&plan)? else {
            return Ok(plan);
        };
        let flag_ids = self.collect_flag_ids(&plan, &provider.dataset)?;
        if flag_ids.is_empty() {
            return Ok(plan);
        }

        let mut functions = HashMap::new();
        let mut bindings = HashMap::new();
        for flag_id in flag_ids {
            let (function, binding) = deferred_bound_cell_flag_udf(flag_id);
            functions.insert(flag_id, Arc::new(function));
            bindings.insert(flag_id, binding);
        }
        let replacement = provider.clone().with_cell_flag_bindings(bindings)?;
        self.rewrite_plan(plan, false, &provider, &replacement, &functions)
    }

    fn name(&self) -> &str {
        CELL_FLAG_ANALYZER_NAME
    }
}

/// Register Lance's snapshot-binding analyzer for `cell_flag(field, name)`.
///
/// Call this once on any DataFusion context used with a manually registered
/// [`LanceTableProvider`]. [`SessionContextExt::read_lance`] registers it
/// automatically.
///
/// ```
/// use datafusion::prelude::SessionContext;
/// use lance::datafusion::register_cell_flag_analyzer;
///
/// let context = SessionContext::new();
/// register_cell_flag_analyzer(&context);
/// ```
pub fn register_cell_flag_analyzer(context: &SessionContext) {
    context.register_udf(CELL_FLAG_UDF.clone());
    context.register_udf(CELL_FLAG_ID_UDF.clone());
    if context
        .state()
        .analyzer()
        .rules
        .iter()
        .any(|rule| rule.name() == CELL_FLAG_ANALYZER_NAME)
    {
        return;
    }
    context.add_analyzer_rule(Arc::new(BindCellFlags));
}

pub trait SessionContextExt {
    /// Creates a DataFrame for reading a Lance dataset
    fn read_lance(
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

pub struct OneShotPartitionStream {
    data: Arc<Mutex<Option<SendableRecordBatchStream>>>,
    schema: Arc<Schema>,
}

impl OneShotPartitionStream {
    pub fn new(data: SendableRecordBatchStream) -> Self {
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
        register_cell_flag_analyzer(self);
        self.read_table(Arc::new(LanceTableProvider::new(
            dataset,
            with_row_id,
            with_row_addr,
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
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow::{
        array::{AsArray, BooleanArray, Int32Array, Int64Array, RecordBatch, RecordBatchIterator},
        datatypes::{DataType, Field, Schema},
        datatypes::{Int32Type, Int64Type},
    };
    use datafusion::{
        datasource::TableProvider,
        functions_aggregate::count::count,
        prelude::{SessionContext, col},
    };
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datafusion::udf::{
        bound_cell_flag_snapshot, cell_flag, deferred_bound_cell_flag_udf,
    };
    use lance_datagen::array;

    use crate::{
        Dataset,
        datafusion::{LanceTableProvider, SessionContextExt, register_cell_flag_analyzer},
        dataset::{UpdateBuilder, WriteParams},
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount},
    };

    #[tokio::test]
    pub async fn test_table_provider() {
        let test_uri = TempStrDir::default();
        let data = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                &test_uri,
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

    #[tokio::test]
    async fn test_table_provider_binds_cell_flags_in_projection_filter_and_aggregate() {
        let test_uri = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::new_null(8)),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_uri,
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let definition = dataset
            .register_cell_flag("embedding", "computed", false)
            .await
            .unwrap();
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 4, 7)")
            .unwrap()
            .set("embedding", "id * 10")
            .unwrap()
            .set_cell_flag("embedding", "computed", true)
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let dataset = Arc::new(result.new_dataset.as_ref().clone());

        let context = SessionContext::new();
        let (deferred_udf, deferred_binding) = deferred_bound_cell_flag_udf(definition.flag_id);
        let provider = LanceTableProvider::new(dataset.clone(), false, false)
            .with_cell_flag_bindings(HashMap::from([(definition.flag_id, deferred_binding)]))
            .unwrap();
        provider
            .scan(&context.state(), None, &[], Some(1))
            .await
            .unwrap();
        let (_, _, limited_coverage) = bound_cell_flag_snapshot(&deferred_udf).unwrap();
        assert_eq!(
            limited_coverage.iter().collect::<Vec<_>>(),
            vec![dataset.fragments().first().unwrap().id as u32]
        );
        provider
            .scan(&context.state(), None, &[], None)
            .await
            .unwrap();
        let (_, _, full_coverage) = bound_cell_flag_snapshot(&deferred_udf).unwrap();
        assert_eq!(
            full_coverage.iter().collect::<Vec<_>>(),
            dataset
                .fragments()
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect::<Vec<_>>()
        );

        let projected = context
            .read_lance(dataset.clone(), false, false)
            .unwrap()
            .select(vec![
                col("id"),
                cell_flag(col("embedding"), "computed").alias("flag_value"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut flag_values = projected
            .iter()
            .flat_map(|batch| {
                let ids = batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                let flag_value = batch
                    .column_by_name("flag_value")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap();
                (0..batch.num_rows()).map(|index| (ids.value(index), flag_value.value(index)))
            })
            .collect::<Vec<_>>();
        flag_values.sort_unstable_by_key(|(id, _)| *id);
        assert_eq!(
            flag_values,
            vec![
                (0, false),
                (1, true),
                (2, false),
                (3, false),
                (4, true),
                (5, false),
                (6, false),
                (7, true),
            ]
        );

        let filtered = context
            .read_lance(dataset.clone(), false, false)
            .unwrap()
            .filter(cell_flag(col("embedding"), "computed"))
            .unwrap()
            .select_columns(&["id"])
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut filtered_ids = filtered
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        filtered_ids.sort_unstable();
        assert_eq!(filtered_ids, vec![1, 4, 7]);

        let aggregated = context
            .read_lance(dataset.clone(), false, false)
            .unwrap()
            .aggregate(
                vec![cell_flag(col("embedding"), "computed").alias("flag_value")],
                vec![count(col("id")).alias("rows")],
            )
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut groups = aggregated
            .iter()
            .flat_map(|batch| {
                let flag_value = batch
                    .column_by_name("flag_value")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap();
                let rows = batch
                    .column_by_name("rows")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                (0..batch.num_rows()).map(|index| (flag_value.value(index), rows.value(index)))
            })
            .collect::<Vec<_>>();
        groups.sort_unstable();
        assert_eq!(groups, vec![(false, 5), (true, 3)]);

        let manual_context = SessionContext::new();
        register_cell_flag_analyzer(&manual_context);
        let ordered = manual_context
            .read_table(Arc::new(LanceTableProvider::new(
                dataset.clone(),
                false,
                false,
            )))
            .unwrap()
            .sort(vec![
                cell_flag(col("embedding"), "computed").sort(false, false),
                col("id").sort(true, false),
            ])
            .unwrap()
            .select_columns(&["id"])
            .unwrap()
            .collect()
            .await
            .unwrap();
        let ordered_ids = ordered
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(ordered_ids, vec![1, 4, 7, 0, 2, 3, 5, 6]);

        manual_context
            .register_table(
                "items",
                Arc::new(LanceTableProvider::new(dataset, false, false)),
            )
            .unwrap();
        let sql = manual_context
            .sql("SELECT id FROM items WHERE cell_flag(embedding, 'computed') ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let sql_ids = sql
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(sql_ids, vec![1, 4, 7]);
    }
}
