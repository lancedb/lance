// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, collections::HashMap, sync::Arc};

use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, SchemaRef};
use datafusion::{
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    error::DataFusionError,
    physical_optimizer::PhysicalOptimizerRule,
};
use datafusion_physical_expr::{
    expressions::Column, EquivalenceProperties, Partitioning, PhysicalExpr,
};
use datafusion_physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    metrics::ExecutionPlanMetricsSet,
    projection::ProjectionExec,
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics,
};
use futures::executor::block_on;
use itertools::Itertools;
use lance_core::{
    datatypes::format_field_path, Error as LanceError, Result as LanceResult, ROW_ADDR, ROW_ID,
};
use lance_index::{
    metrics::{MetricsCollector, NoOpMetricsCollector},
    scalar::{AnyQuery, ScalarIndex},
    DatasetIndexExt, ScalarIndexCriteria,
};
use roaring::RoaringBitmap;
use snafu::location;
use tokio::task::block_in_place;
use uuid::Uuid;

use crate::{
    index::prefilter::DatasetPreFilter,
    index::DatasetIndexInternalExt,
    io::exec::{
        filtered_read::FilteredReadExec,
        utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter},
        AddRowAddrExec, Planner,
    },
    Dataset,
};

#[derive(Debug, Clone)]
pub struct ColumnIndexContext {
    column_name: String,
    index_name: String,
    ids: Vec<Uuid>,
    index: Arc<dyn ScalarIndex>,
    fragment_bitmap: Option<RoaringBitmap>,
}

#[derive(Debug)]
pub struct DatasetIndexScanContext {
    columns: HashMap<i32, ColumnIndexContext>,
}

impl DatasetIndexScanContext {
    fn column_context(&self, field_id: i32) -> Option<&ColumnIndexContext> {
        self.columns.get(&field_id)
    }

    async fn prepare(
        dataset: Arc<Dataset>,
        columns: HashMap<i32, String>,
    ) -> LanceResult<Option<Self>> {
        let mut column_map = HashMap::new();
        for (field_id, column_name) in columns {
            let criteria = ScalarIndexCriteria::default().for_column(column_name.as_str());
            let Some(index_meta) = dataset.load_scalar_index(criteria).await? else {
                continue;
            };
            let index_uuid = index_meta.uuid.to_string();
            let index = dataset
                .open_scalar_index(column_name.as_str(), &index_uuid, &NoOpMetricsCollector)
                .await?;
            column_map.insert(
                field_id,
                ColumnIndexContext {
                    column_name: column_name.clone(),
                    index_name: index_meta.name.clone(),
                    ids: vec![index_meta.uuid],
                    index,
                    fragment_bitmap: index_meta.fragment_bitmap.clone(),
                },
            );
        }
        if column_map.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Self {
                columns: column_map,
            }))
        }
    }
}

struct Candidate {
    dataset: Arc<Dataset>,
    field_id: i32,
    column_name: String,
}

fn detect_candidate(read: &FilteredReadExec) -> Option<Candidate> {
    let options = read.options();
    let projection = &options.projection;
    if options.with_deleted_rows
        || options.refine_filter.is_some()
        || options.scan_range_before_filter.is_some()
        || options.scan_range_after_filter.is_some()
        || read.index_input().is_some()
        || projection.with_row_last_updated_at_version
        || projection.with_row_created_at_version
        || projection.field_ids.len() != 1
    {
        return None;
    }
    let dataset = Arc::clone(read.dataset());
    let schema = dataset.schema();
    let field_id = *projection.field_ids.iter().next()?;
    let field = schema.field_by_id(field_id)?;
    let column_name = if let Some(ancestors) = schema.field_ancestry_by_id(field.id) {
        let segments: Vec<&str> = ancestors.iter().map(|f| f.name.as_str()).collect();
        format_field_path(&segments)
    } else {
        field.name.clone()
    };
    if let Some(filter_expr) = options.full_filter.as_ref() {
        let filter_columns = Planner::column_names_in_expr(filter_expr);
        if filter_columns.iter().any(|col| col != &column_name) {
            return None;
        }
    }
    Some(Candidate {
        dataset,
        field_id,
        column_name,
    })
}

pub async fn prepare_index_scan_contexts(
    plan: &Arc<dyn ExecutionPlan>,
) -> LanceResult<HashMap<usize, Arc<DatasetIndexScanContext>>> {
    let mut requests: HashMap<usize, (Arc<Dataset>, HashMap<i32, String>)> = HashMap::new();

    plan.apply(|node| {
        if let Some(read) = node.as_any().downcast_ref::<FilteredReadExec>() {
            if let Some(candidate) = detect_candidate(read) {
                let key = Arc::as_ptr(&candidate.dataset) as usize;
                let Candidate {
                    dataset,
                    field_id,
                    column_name,
                } = candidate;
                let entry = requests
                    .entry(key)
                    .or_insert_with(|| (dataset.clone(), HashMap::new()));
                entry.1.entry(field_id).or_insert(column_name);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map_err(|err| LanceError::Execution {
        message: err.to_string(),
        location: location!(),
    })?;

    let mut contexts = HashMap::new();
    for (key, (dataset, columns)) in requests {
        if let Some(context) = DatasetIndexScanContext::prepare(dataset.clone(), columns).await? {
            contexts.insert(key, Arc::new(context));
        }
    }
    Ok(contexts)
}

#[derive(Clone, Debug)]
pub struct IndexScanExec {
    index_name: String,
    ids: Vec<Uuid>,
    indexes: Vec<Arc<dyn ScalarIndex>>,
    properties: PlanProperties,
    query: Option<Arc<dyn AnyQuery>>,
    dataset: Arc<Dataset>,
    fragment_bitmaps: Vec<Option<RoaringBitmap>>,
    metrics: ExecutionPlanMetricsSet,
}

impl IndexScanExec {
    fn new(
        index_name: String,
        ids: Vec<Uuid>,
        indexes: Vec<Arc<dyn ScalarIndex>>,
        output_schema: SchemaRef,
        query: Option<Arc<dyn AnyQuery>>,
        dataset: Arc<Dataset>,
        fragment_bitmaps: Vec<Option<RoaringBitmap>>,
    ) -> Self {
        let eq_properties = EquivalenceProperties::new(output_schema);
        let partitioning = Partitioning::UnknownPartitioning(indexes.len());
        let properties = PlanProperties::new(
            eq_properties,
            partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        assert_eq!(indexes.len(), fragment_bitmaps.len());

        Self {
            index_name,
            ids,
            indexes,
            properties,
            query,
            dataset,
            fragment_bitmaps,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for IndexScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let ids = self.ids.iter().map(|id| format!("\"{id}\"",)).join(", ");
                write!(
                    f,
                    "IndexScanExec: name=\"{}\", ids=[{}]",
                    self.index_name, ids
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "name=\"{}\"", self.index_name)?;
                for (i, id) in self.ids.iter().enumerate() {
                    write!(f, "ids[{}]=\"{}\"", i, id)?;
                }
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for IndexScanExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        Self::static_name()
    }

    fn schema(&self) -> SchemaRef {
        self.properties.equivalence_properties().schema().clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Plan(
                "IndexScanExec does not support children".into(),
            ))
        }
    }

    fn reset_state(self: Arc<Self>) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.as_ref().clone()
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion_physical_plan::SendableRecordBatchStream> {
        let index = self
            .indexes
            .get(partition)
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "Partition out of bounds {} versus len {}",
                    partition,
                    self.indexes.len()
                ))
            })?
            .clone();
        let batch_size = context.session_config().options().execution.batch_size;
        let query = self
            .query
            .as_ref()
            .map(|query| query.as_ref() as &dyn AnyQuery);
        let fragment_bitmap = self
            .fragment_bitmaps
            .get(partition)
            .cloned()
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "Partition metadata out of bounds {} versus len {}",
                    partition,
                    self.fragment_bitmaps.len()
                ))
            })?;
        let deletion_mask = match fragment_bitmap {
            Some(bitmap) => DatasetPreFilter::create_deletion_mask(self.dataset.clone(), bitmap)
                .map(|fut| block_in_place(|| block_on(fut)))
                .transpose()
                .map_err(|err| DataFusionError::External(Box::new(err)))?,
            None => None,
        };
        let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let collector: Arc<dyn MetricsCollector> = metrics.clone();
        let stream = index
            .scan(query, batch_size, deletion_mask, collector)?
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "Index {} does not support scanning.",
                    index.index_type()
                ))
            })?;
        let schema = stream.schema();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            schema,
            stream,
            partition,
            &self.metrics,
        )))
    }

    fn metrics(&self) -> Option<datafusion_physical_plan::metrics::MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(
        &self,
        _partition: Option<usize>,
    ) -> datafusion::error::Result<datafusion_physical_plan::Statistics> {
        let schema = self.schema();
        Ok(Statistics::new_unknown(schema.as_ref()))
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let _ = limit;
        None
    }
}

/// Physical optimizer rule to replace simple Lance reads with index scans.
///
/// If a `FilteredReadExec` only needs `_rowid` and a single indexed column,
/// it is replaced with `IndexScanExec`. When `_rowaddr` is also requested,
/// an `AddRowAddrExec` is appended after the scan so the public schema remains unchanged.
#[derive(Debug)]
pub struct ScanIndexRule {
    contexts: Arc<HashMap<usize, Arc<DatasetIndexScanContext>>>,
}

impl ScanIndexRule {
    pub fn new(contexts: HashMap<usize, Arc<DatasetIndexScanContext>>) -> Self {
        Self {
            contexts: Arc::new(contexts),
        }
    }

    fn rewrite_filtered_read(
        &self,
        read: &FilteredReadExec,
        context: &DatasetIndexScanContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>, DataFusionError> {
        // Only consider scans with no filters or index inputs.
        let options = read.options();
        if options.with_deleted_rows
            || options.refine_filter.is_some()
            || options.scan_range_before_filter.is_some()
            || options.scan_range_after_filter.is_some()
            || read.index_input().is_some()
        {
            return Ok(None);
        }

        let projection = &options.projection;

        if projection.with_row_last_updated_at_version
            || projection.with_row_created_at_version
            || projection.field_ids.len() != 1
        {
            return Ok(None);
        }

        // Determine the indexed column.
        let field_id = *projection.field_ids.iter().next().unwrap();
        let dataset = read.dataset().clone();
        let schema = dataset.schema();
        let field = match schema.field_by_id(field_id) {
            Some(field) => field,
            None => return Ok(None),
        };

        let Some(column_context) = context.column_context(field.id) else {
            return Ok(None);
        };

        if let Some(filter_expr) = options.full_filter.as_ref() {
            let filter_columns = Planner::column_names_in_expr(filter_expr);
            if filter_columns
                .iter()
                .any(|col| col != &column_context.column_name)
            {
                return Ok(None);
            }
        }

        // Determine the schema produced by the index scan.
        let read_schema = read.schema();
        let needs_row_addr = projection.with_row_addr;

        let arrow_field: ArrowField = field.into();
        let mut value_field_arrow = ArrowField::new(
            "values",
            arrow_field.data_type().clone(),
            arrow_field.is_nullable(),
        );
        if !arrow_field.metadata().is_empty() {
            value_field_arrow = value_field_arrow.with_metadata(arrow_field.metadata().clone());
        }
        let ids_field = ArrowField::new("ids", ArrowDataType::UInt64, true);
        let index_schema = Arc::new(arrow_schema::Schema::new(vec![
            value_field_arrow,
            ids_field,
        ]));
        let mut result_plan: Arc<dyn ExecutionPlan> = Arc::new(IndexScanExec::new(
            column_context.index_name.clone(),
            column_context.ids.clone(),
            vec![column_context.index.clone()],
            index_schema,
            None,
            dataset.clone(),
            vec![column_context.fragment_bitmap.clone()],
        ));

        let index_output_schema = result_plan.schema();
        let mut projection_expr = Vec::with_capacity(read_schema.fields().len());
        for field in read_schema.fields() {
            let expr = if field.name() == &column_context.column_name {
                Column::new_with_schema(
                    index_output_schema.field(0).name(),
                    index_output_schema.as_ref(),
                )
            } else if field.name() == ROW_ID {
                Column::new_with_schema(
                    index_output_schema.field(1).name(),
                    index_output_schema.as_ref(),
                )
            } else {
                Column::new_with_schema(field.name(), index_output_schema.as_ref())
            }
            .map_err(|err| DataFusionError::External(Box::new(err)))?;
            let expr_arc: Arc<dyn PhysicalExpr> = Arc::new(expr);
            projection_expr.push((expr_arc, field.name().clone()));
        }
        if !projection_expr.is_empty() {
            result_plan = Arc::new(
                ProjectionExec::try_new(projection_expr, result_plan.clone())
                    .map_err(|err| DataFusionError::External(Box::new(err)))?,
            );
        }

        if needs_row_addr {
            let Some(rowaddr_pos) = read_schema
                .fields()
                .iter()
                .position(|f| f.name() == ROW_ADDR)
            else {
                return Ok(None);
            };
            let add_row_addr = AddRowAddrExec::try_new(result_plan, dataset.clone(), rowaddr_pos)
                .map_err(|err| DataFusionError::External(Box::new(err)))?;
            Ok(Some(Arc::new(add_row_addr)))
        } else {
            Ok(Some(result_plan))
        }
    }
}

impl PhysicalOptimizerRule for ScanIndexRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        if self.contexts.is_empty() {
            return Ok(plan);
        }

        plan.transform_down(|plan| {
            if let Some(read) = plan.as_any().downcast_ref::<FilteredReadExec>() {
                let key = Arc::as_ptr(read.dataset()) as usize;
                if let Some(context) = self.contexts.get(&key) {
                    if let Some(new_plan) = self.rewrite_filtered_read(read, context)? {
                        return Ok(Transformed::yes(new_plan));
                    }
                }
            }
            Ok(Transformed::no(plan))
        })
        .map(|res| res.data)
    }

    fn name(&self) -> &str {
        "ScanIndex"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::record_batch::RecordBatch;
    use arrow::util::pretty::pretty_format_batches;
    use arrow_array::types::Int32Type;
    use datafusion::{
        config::ConfigOptions, execution::context::SessionContext,
        physical_optimizer::optimizer::PhysicalOptimizer,
    };
    use futures::StreamExt;
    use lance_datagen::array;
    use lance_index::{
        scalar::{BuiltinIndexType, ScalarIndexParams},
        IndexType,
    };

    use crate::{
        datafusion::LanceTableProvider,
        io::exec::get_physical_optimizer,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount},
    };

    fn run_optimizer(
        optimizer: &PhysicalOptimizer,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let config = ConfigOptions::new();
        let mut current = plan;
        for rule in optimizer.rules.iter() {
            current = rule.optimize(current, &config).unwrap();
        }
        current
    }

    async fn collect_plan(
        plan: Arc<dyn ExecutionPlan>,
    ) -> datafusion::error::Result<Vec<RecordBatch>> {
        let session = SessionContext::new();
        let task_ctx = session.task_ctx();
        let mut results = Vec::new();
        let partition_count = plan.properties().partitioning.partition_count();
        for partition in 0..partition_count {
            let mut stream = plan.execute(partition, task_ctx.clone())?;
            while let Some(batch) = stream.next().await {
                results.push(batch?);
            }
        }
        Ok(results)
    }

    fn batches_to_string(batches: &[RecordBatch]) -> String {
        pretty_format_batches(batches)
            .expect("format batches")
            .to_string()
    }

    #[tokio::test]
    async fn test_index_scan_optimizer_rewrite() {
        let mut dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(5))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["value"],
                IndexType::BTree,
                Some("value_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let dataset = Arc::new(dataset);
        let mut scanner = dataset.scan();
        scanner.project(&["value"]).unwrap();
        let plan = scanner.create_plan().await.unwrap();

        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();

        let contexts = prepare_index_scan_contexts(&plan).await.unwrap();
        let optimizer = get_physical_optimizer(contexts);
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexScanExec>().is_some() {
                    found_index_scan = true;
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        assert!(found_index_scan, "index scan not found in optimized plan");
        assert_eq!(
            batches_to_string(&expected),
            batches_to_string(&actual),
            "optimized results differed from original plan"
        );
    }

    #[tokio::test]
    async fn test_index_scan_optimizer_rewrite_index_column_only() {
        let mut dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(5))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["value"],
                IndexType::BTree,
                Some("value_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let dataset = Arc::new(dataset);
        let plan = dataset
            .scan()
            .project(&["value"])
            .unwrap()
            .create_plan()
            .await
            .unwrap();

        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();

        let contexts = prepare_index_scan_contexts(&plan).await.unwrap();
        let optimizer = get_physical_optimizer(contexts);
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexScanExec>().is_some() {
                    found_index_scan = true;
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        assert!(found_index_scan, "index scan not found in optimized plan");
        assert_eq!(optimized_plan.schema().fields().len(), 1);
        assert_eq!(optimized_plan.schema().field(0).name(), "value");
        assert_eq!(
            batches_to_string(&expected),
            batches_to_string(&actual),
            "optimized results differed from original plan"
        );
    }

    #[tokio::test]
    async fn test_index_scan_optimizer_rewrite_no_rowaddr() {
        let mut dataset = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(5))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["value"],
                IndexType::BTree,
                Some("value_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let dataset = Arc::new(dataset);
        let mut scanner = dataset.scan();
        scanner.project(&["value"]).unwrap();
        scanner.with_row_id();
        let plan = scanner.create_plan().await.unwrap();

        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();

        let contexts = prepare_index_scan_contexts(&plan).await.unwrap();
        let optimizer = get_physical_optimizer(contexts);
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        let mut found_add_row_addr = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexScanExec>().is_some() {
                    found_index_scan = true;
                }
                if node.as_any().downcast_ref::<AddRowAddrExec>().is_some() {
                    found_add_row_addr = true;
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        assert!(found_index_scan, "index scan not found in optimized plan");
        assert!(!found_add_row_addr, "AddRowAddrExec unexpectedly present");
        assert_eq!(
            batches_to_string(&expected),
            batches_to_string(&actual),
            "optimized results differed from original plan"
        );
    }

    #[tokio::test]
    async fn test_index_scan_optimizer_rewrite_join() {
        let mut dataset_a = lance_datagen::gen_batch()
            .col("value", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(5))
            .await
            .unwrap();
        let dataset_b = lance_datagen::gen_batch()
            .col("value", array::cycle::<Int32Type>(vec![0, 2, 4, 6, 8]))
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(5))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset_a
            .create_index(
                &["value"],
                IndexType::BTree,
                Some("value_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let dataset_a = Arc::new(dataset_a);
        let dataset_b = Arc::new(dataset_b);

        let ctx = SessionContext::new();
        ctx.register_table(
            "a",
            Arc::new(LanceTableProvider::new(dataset_a.clone(), true, false)),
        )
        .unwrap();
        ctx.register_table(
            "b",
            Arc::new(LanceTableProvider::new(dataset_b.clone(), false, false)),
        )
        .unwrap();

        let df = ctx
            .sql("SELECT a._rowid, a.value FROM a LEFT ANTI JOIN b ON a.value = b.value")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();

        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();

        let contexts = prepare_index_scan_contexts(&plan).await.unwrap();
        let optimizer = get_physical_optimizer(contexts);
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexScanExec>().is_some() {
                    found_index_scan = true;
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        assert!(
            found_index_scan,
            "index scan not found in optimized join plan"
        );
        assert_eq!(
            batches_to_string(&expected),
            batches_to_string(&actual),
            "optimized results differed from original plan"
        );
    }
}
