// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, sync::Arc};

use arrow_schema::SchemaRef;
use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    error::DataFusionError,
    physical_optimizer::PhysicalOptimizerRule,
};
use datafusion_expr::Expr;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    filter::FilterExec,
    metrics::ExecutionPlanMetricsSet,
    union::UnionExec,
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics,
};
use futures::{executor::block_on, FutureExt, TryStreamExt};
use itertools::Itertools;
use lance_core::ROW_ID;
use lance_index::{
    metrics::MetricsCollector,
    scalar::expression::{
        apply_scalar_indices, IndexedExpression, MultiQueryParser, ScalarIndexExpr,
    },
    DatasetIndexExt,
};
use lance_table::format::IndexMetadata;
use tokio::task::block_in_place;

use crate::{
    index::{
        prefilter::DatasetPreFilter, scalar::IndexDetails, DatasetIndexInternalExt, ScalarIndexInfo,
    },
    io::exec::{
        filtered_read::FilteredReadExec,
        utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter},
        AddRowAddrExec,
    },
    Dataset,
};

/// A physical node that satisfies a scan with just indexed data. This can
/// completely avoid IO when the index is in cache.
#[derive(Clone, Debug)]
pub struct IndexOnlyScanExec {
    index_name: String,
    with_row_id: bool,
    indexes: Vec<IndexMetadata>,
    index_expr: Option<ScalarIndexExpr>,
    dataset: Arc<Dataset>,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl IndexOnlyScanExec {
    fn new(
        index_name: String,
        indexes: Vec<IndexMetadata>,
        output_schema: SchemaRef,
        index_expr: Option<ScalarIndexExpr>,
        dataset: Arc<Dataset>,
    ) -> Self {
        let eq_properties = EquivalenceProperties::new(output_schema);
        let with_row_id = eq_properties.schema().fields().find(ROW_ID).is_some();
        let partitioning = Partitioning::UnknownPartitioning(indexes.len());
        let properties = PlanProperties::new(
            eq_properties,
            partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            index_name,
            with_row_id,
            indexes,
            index_expr,
            properties,
            dataset,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for IndexOnlyScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let ids = self
                    .indexes
                    .iter()
                    .map(|index| format!("\"{}\"", index.uuid))
                    .join(", ");
                write!(
                    f,
                    "IndexScanExec: name=\"{}\", ids=[{}]",
                    self.index_name, ids
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "name=\"{}\"", self.index_name)?;
                for (i, index) in self.indexes.iter().enumerate() {
                    write!(f, "ids[{}]=\"{}\"", i, index.uuid)?;
                }
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for IndexOnlyScanExec {
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

        // Extract the query from the index expression
        // For now, we only support simple Query variants, not complex And/Or/Not expressions
        let query_arc = match self.index_expr.as_ref() {
            Some(lance_index::scalar::expression::ScalarIndexExpr::Query(search)) => {
                Some(Arc::clone(&search.query))
            }
            Some(_) => {
                return Err(DataFusionError::Internal(
                    "Complex index expressions (And/Or/Not) are not yet supported for index-only scans".to_string()
                ));
            }
            None => None,
        };

        let deletion_mask = match index.fragment_bitmap {
            Some(bitmap) => DatasetPreFilter::create_deletion_mask(self.dataset.clone(), bitmap)
                .map(|fut| block_in_place(|| block_on(fut)))
                .transpose()
                .map_err(|err| DataFusionError::External(Box::new(err)))?,
            None => None,
        };
        let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let collector: Arc<dyn MetricsCollector> = metrics.clone();

        let dataset = self.dataset.clone();
        let stream = futures::stream::once(async move {
            let index = dataset
                .open_scalar_index(
                    "dummy", // TODO: can we avoid having to pass this.
                    &index.uuid.to_string(),
                    collector.as_ref(),
                )
                .await?;

            // TODO: push down projection
            let query = query_arc.as_ref().map(|q| q.as_ref());
            index
                .scan(query, batch_size, deletion_mask, collector)?
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "Index {} does not support scanning.",
                        index.index_type()
                    ))
                })
        })
        .try_flatten();

        let schema = self.schema();
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
pub struct ScanIndexRule;

impl ScanIndexRule {
    fn index_scan_supported(read: &FilteredReadExec) -> bool {
        let options = read.options();
        if options.fragments.is_some()
            || options.scan_range_after_filter.is_some()
            || options.scan_range_before_filter.is_some()
        {
            return false;
        }
        let projection = &options.projection;
        if projection.field_ids.len() > 1
            || projection.with_row_created_at_version
            || projection.with_row_last_updated_at_version
        {
            return false;
        }
        return true;
    }

    /// Look for an index on the given field_id that is scannable and can satisfy
    /// the filter the best.
    fn scannable_index(
        dataset: &Dataset,
        field_id: i32,
        predicate: Option<&Expr>,
    ) -> Option<(Vec<IndexMetadata>, IndexedExpression)> {
        let index_metas = dataset.load_indices().now_or_never()?.ok()?;

        let mut parts = Vec::new();
        let mut index_name = None;
        let mut indexed_expr = IndexedExpression::default();

        for index_metadata in index_metas.iter() {
            if index_name.as_ref() == Some(&index_metadata.name) {
                parts.push(index_metadata.clone());
                continue;
            } else if index_name.is_some() {
                // We already found an index.
                continue;
            };
            if !(&index_metadata.fields == &[field_id]) {
                continue;
            }
            let index_details = IndexDetails(index_metadata.index_details.clone()?);
            let plugin = index_details.get_plugin().ok()?;

            if !plugin.supports_scan() {
                continue;
            }

            if let Some(predicate) = predicate {
                let parser =
                    plugin.new_query_parser(index_metadata.name.clone(), &index_details.0)?;
                let mut index_info = ScalarIndexInfo::default();
                index_info.insert(
                    index_metadata.name.clone(),
                    dataset.schema().field_by_id(field_id)?.data_type().clone(),
                    Box::new(MultiQueryParser::single(parser)),
                );
                indexed_expr = apply_scalar_indices(predicate.clone(), &index_info).ok()?;
            }

            index_name = Some(index_metadata.name.clone());
            parts.push(index_metadata.clone());
        }

        index_name.map(|_| (parts, indexed_expr))
    }

    fn rewrite_filtered_read(
        read: &FilteredReadExec,
        indexes: Vec<IndexMetadata>,
        indexed_expr: IndexedExpression,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let mut unindexed_fragments = read.dataset().fragment_bitmap.as_ref().clone();
        for index in &indexes {
            let index_bitmap = index.fragment_bitmap.as_ref().ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "Index {} does not have fragment bitmap",
                    index.uuid
                ))
            })?;
            unindexed_fragments -= index_bitmap;
        }

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(IndexOnlyScanExec::new(
            indexes[0].name.clone(),
            indexes,
            read.schema(), // TODO: this doesn't work if row_addr was requested, since it needs to be added later.
            indexed_expr.scalar_query,
            read.dataset().clone(),
        ));

        if let Some(expr) = indexed_expr.refine_expr {
            // Create a planner to convert the logical expression to a physical expression
            use crate::io::exec::Planner;
            let planner = Planner::new(Arc::new(read.dataset().schema().into()));
            let physical_expr = planner.create_physical_expr(&expr).map_err(|e| {
                DataFusionError::Plan(format!("Failed to create physical expression: {}", e))
            })?;
            plan = Arc::new(FilterExec::try_new(physical_expr, plan)?)
        }

        if read.options().projection.with_row_addr {
            // Index only contains row ids, so we need to add row addr if requested
            plan = Arc::new(AddRowAddrExec::try_new(plan, read.dataset().clone(), 2)?)
        }

        if !unindexed_fragments.is_empty() {
            let mut options = read.options().clone();
            let unindexed_fragments = read
                .dataset()
                .fragments()
                .iter()
                .filter(|frag| unindexed_fragments.contains(frag.id as u32))
                .cloned()
                .collect::<Vec<_>>();
            options.fragments = Some(Arc::new(unindexed_fragments));
            let filtered_read = Arc::new(FilteredReadExec::try_new(
                read.dataset().clone(),
                options,
                read.index_input().cloned(),
            )?);
            plan = Arc::new(UnionExec::new(vec![filtered_read, plan]));
        }

        Ok(plan)
    }
}

impl PhysicalOptimizerRule for ScanIndexRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        plan.transform_down(|plan| {
            if let Some(read) = plan.as_any().downcast_ref::<FilteredReadExec>() {
                if Self::index_scan_supported(read) {
                    let projection = &read.options().projection;
                    let field_id = projection.field_ids.iter().next().cloned().unwrap();
                    let predicate = read.options().full_filter.as_ref();
                    if let Some((indexes, indexed_expr)) =
                        Self::scannable_index(&read.dataset(), field_id, predicate)
                    {
                        if let Ok(plan) = Self::rewrite_filtered_read(read, indexes, indexed_expr) {
                            return Ok(Transformed::yes(plan));
                        }
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
    use arrow_array::{ArrayRef, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
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
        dataset::scanner::Scanner,
        io::exec::get_physical_optimizer,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, TestDatasetGenerator},
    };
    use lance_core::utils::tempfile::TempDir;
    use lance_file::version::LanceFileVersion;

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
        let indices = dataset.load_indices().await.unwrap();
        let schema_field_info: Vec<_> = dataset
            .schema()
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.id))
            .collect();
        let loaded = dataset
            .load_scalar_index(ScalarIndexCriteria::default().for_column("value"))
            .await
            .unwrap()
            .is_some();
        eprintln!("load_scalar_index before scan: {}", loaded);
        let mut scanner = Scanner::new(dataset.clone());
        scanner.project(&["value"]).unwrap();
        scanner.filter("value > 10").unwrap();
        let plan = scanner.create_plan().await.unwrap();
        // eprintln!("plan:\n{}", plan.display_indent().unwrap());
        let debug_info = std::cell::RefCell::new(Vec::new());
        plan.apply(|node| {
            if let Some(read) = node.as_any().downcast_ref::<FilteredReadExec>() {
                let options = read.options();
                debug_info.borrow_mut().push((
                    options.with_deleted_rows,
                    options.projection.field_ids.clone(),
                    options.full_filter.clone(),
                    options.refine_filter.clone(),
                ));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        eprintln!("filtered_debug: {:?}", debug_info.borrow());
        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();
        let optimizer = get_physical_optimizer();
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexOnlyScanExec>().is_some() {
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
        let mut scanner = Scanner::new(dataset.clone());
        scanner.project(&["value"]).unwrap();
        let plan = scanner.create_plan().await.unwrap();

        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();

        let optimizer = get_physical_optimizer();
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexOnlyScanExec>().is_some() {
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
        let mut scanner = Scanner::new(dataset.clone());
        scanner.project(&["value"]).unwrap();
        scanner.with_row_id();
        let plan = scanner.create_plan().await.unwrap();

        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();

        let optimizer = get_physical_optimizer();
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        let mut found_add_row_addr = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexOnlyScanExec>().is_some() {
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
    async fn test_index_scan_optimizer_rewrite_refine_filter() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            ArrowDataType::Utf8,
            false,
        )]));
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["football", "apple", "soccer"])) as ArrayRef],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["basketball", "balloon", "car"])) as ArrayRef],
        )
        .unwrap();

        let temp_dir = TempDir::default();
        let dataset_uri = format!("{}/refine_filter", temp_dir.path_str());
        let mut dataset = TestDatasetGenerator::new(vec![batch1, batch2], LanceFileVersion::Stable)
            .make_hostile(&dataset_uri)
            .await;

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
        let mut scanner = Scanner::new(dataset.clone());
        scanner.project(&["value"]).unwrap();
        scanner.filter("value LIKE '%ball%'").unwrap();
        let plan = scanner.create_plan().await.unwrap();

        let expected = collect_plan(plan.clone()).await.unwrap();
        let plan = plan.reset_state().unwrap();

        let optimizer = get_physical_optimizer();
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        let mut filter_on_index = false;
        optimized_plan
            .apply(|node| {
                if let Some(filter) = node.as_any().downcast_ref::<FilterExec>() {
                    if let Some(child) = filter.children().first() {
                        if child.as_any().downcast_ref::<IndexOnlyScanExec>().is_some() {
                            filter_on_index = true;
                        }
                    }
                }
                if node.as_any().downcast_ref::<IndexOnlyScanExec>().is_some() {
                    found_index_scan = true;
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        assert!(found_index_scan, "index scan not found in optimized plan");
        assert!(
            filter_on_index,
            "refine filter was not applied after index scan"
        );
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

        let optimizer = get_physical_optimizer();
        let optimized_plan = run_optimizer(&optimizer, plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();

        let mut found_index_scan = false;
        optimized_plan
            .apply(|node| {
                if node.as_any().downcast_ref::<IndexOnlyScanExec>().is_some() {
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
