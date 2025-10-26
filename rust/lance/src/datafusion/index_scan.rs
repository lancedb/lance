// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, sync::Arc};

use arrow_schema::{Schema, SchemaRef};
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
use futures::{FutureExt, TryFutureExt, TryStreamExt};
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
                    "IndexOnlyScanExec: name=\"{}\", ids=[{}]",
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
                "IndexOnlyScanExec does not support children".into(),
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

        let collector: Arc<dyn MetricsCollector> =
            Arc::new(IndexMetrics::new(&self.metrics, partition));

        let dataset = self.dataset.clone();
        let with_row_id = self.with_row_id;
        let stream = futures::stream::once(async move {
            let load_deletion_mask = index
                .fragment_bitmap
                .and_then(|bitmap| DatasetPreFilter::create_deletion_mask(dataset.clone(), bitmap));
            let load_deletion_mask = if let Some(future) = load_deletion_mask {
                futures::future::Either::Left(future.map_ok(Some))
            } else {
                futures::future::Either::Right(futures::future::ok(None))
            };

            let uuid_str = index.uuid.to_string();
            let load_index = dataset.open_scalar_index(
                "dummy", // TODO: can we avoid having to pass this.
                &uuid_str,
                collector.as_ref(),
            );

            let (deletion_mask, index) = futures::try_join!(load_deletion_mask, load_index)?;

            // TODO: push down projection
            let query = query_arc.as_ref().map(|q| q.as_ref());
            index
                .scan(query, batch_size, with_row_id, deletion_mask, collector)?
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
/// it is replaced with `IndexOnlyScanExec`. When `_rowaddr` is also requested,
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
        true
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
            if index_metadata.fields != [field_id] {
                continue;
            }
            if let Some(bitmap) = &index_metadata.fragment_bitmap {
                // Don't want to use empty indexes.
                if bitmap.is_empty() {
                    continue;
                }
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

        // Build the schema that the index scan will return.
        // The index returns the indexed column + _rowid (if requested).
        // If _rowaddr is requested, we'll add it later with AddRowAddrExec.
        let projection = &read.options().projection;
        let field_id = projection
            .field_ids
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| DataFusionError::Plan("No fields in projection".to_string()))?;
        let dataset_schema = read.dataset().schema();
        let indexed_field = dataset_schema.field_by_id(field_id).ok_or_else(|| {
            DataFusionError::Plan(format!("Field {} not found in dataset schema", field_id))
        })?;

        let mut schema_fields = vec![Arc::new(indexed_field.into())];

        // Add _rowid if needed (needed for deletion mask or if _rowaddr will be added)
        if projection.with_row_id || projection.with_row_addr {
            use lance_core::ROW_ID_FIELD;
            schema_fields.push(Arc::new(ROW_ID_FIELD.clone()));
        }

        let index_scan_schema = Arc::new(Schema::new(schema_fields));

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(IndexOnlyScanExec::new(
            indexes[0].name.clone(),
            indexes,
            index_scan_schema,
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

        // TODO: handle case of with_row_addr but not with_row_id.

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
                        Self::scannable_index(read.dataset(), field_id, predicate)
                    {
                        match Self::rewrite_filtered_read(read, indexes, indexed_expr) {
                            Ok(plan) => return Ok(Transformed::yes(plan)),
                            Err(error) => {
                                dbg!(error);
                            }
                        }
                    }
                }
            }
            Ok(Transformed::no(plan))
        })
        .map(|res| res.data)
    }

    fn name(&self) -> &str {
        "ScanOnlyIndex"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::scanner::Scanner,
        io::exec::get_physical_optimizer,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount},
    };
    use arrow::record_batch::RecordBatch;
    use arrow::util::pretty::pretty_format_batches;
    use arrow_array::types::Int32Type;
    use datafusion::assert_batches_sorted_eq;
    use datafusion::{
        config::ConfigOptions, execution::context::SessionContext,
        physical_optimizer::optimizer::PhysicalOptimizer,
    };
    use datafusion_physical_plan::display::DisplayableExecutionPlan;
    use futures::StreamExt;
    use lance_core::{ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION, ROW_OFFSET};
    use lance_datagen::{array, ArrayGeneratorExt};
    use lance_index::{
        scalar::{BuiltinIndexType, ScalarIndexParams},
        IndexType,
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

    fn plan_to_string(plan: &dyn ExecutionPlan) -> String {
        format!("{}", DisplayableExecutionPlan::new(plan).indent(false))
    }

    async fn test_optimizer_rule(plan: Arc<dyn ExecutionPlan>, predicate: impl FnOnce(&str)) {
        let expected = collect_plan(plan.clone()).await.unwrap();
        let expected = format!("{}", pretty_format_batches(&expected).unwrap());
        dbg!(&expected);

        let plan = plan.reset_state().unwrap();
        let optimizer = get_physical_optimizer();
        let optimized_plan = run_optimizer(&optimizer, plan);

        let explained_plan = plan_to_string(optimized_plan.as_ref());
        predicate(&explained_plan);

        let actual = collect_plan(optimized_plan.clone()).await.unwrap();
        let expected_lines = expected.trim().lines().collect::<Vec<&str>>();
        dbg!(actual.as_slice());
        dbg!(expected_lines.as_slice());
        assert_batches_sorted_eq!(expected_lines, &actual);
    }

    async fn test_dataset() -> Arc<Dataset> {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "value",
                array::step::<Int32Type>().with_nulls(&[false, false, false, true]),
            )
            .col("other", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(20))
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
        dataset.delete("value % 7 = 0").await.unwrap();
        Arc::new(dataset)
    }

    #[tokio::test]
    async fn test_index_scan_project_filter() {
        let dataset = test_dataset().await;

        let queries = [
            Box::new(|scanner: &mut Scanner| {
                scanner.project(&["value"]).unwrap();
            }) as Box<dyn Fn(&mut Scanner)>,
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value"])
                    .unwrap()
                    .filter("value > 10")
                    .unwrap();
            }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value"])
                    .unwrap()
                    .filter("value % 2 = 1")
                    .unwrap();
            }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value"])
                    .unwrap()
                    .filter("value % 2 = 1 AND value > 10")
                    .unwrap();
            }),
            // TODO: figure out rowid filter
            // Box::new(|scanner: &mut Scanner| {
            //     scanner
            //         .project(&["value"])
            //         .unwrap()
            //         .filter("value < 100 AND _rowid > 10")
            //         .unwrap();
            // }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value"])
                    .unwrap()
                    .filter("value is null")
                    .unwrap();
            }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value"])
                    .unwrap()
                    .filter("value is not null")
                    .unwrap();
            }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value", ROW_ADDR])
                    .unwrap()
                    .filter("value > 10")
                    .unwrap();
            }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value", ROW_ID])
                    .unwrap()
                    .filter("value > 10")
                    .unwrap();
            }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value", ROW_ADDR, ROW_ID])
                    .unwrap()
                    .filter("value > 10")
                    .unwrap();
            }),
        ];

        for query in queries {
            let mut scanner = Scanner::new(dataset.clone());
            query(&mut scanner);
            let plan = scanner.create_unoptimized_plan().await.unwrap();

            let explained_plan = plan_to_string(plan.as_ref());
            assert!(!explained_plan.contains("IndexOnlyScanExec"));

            test_optimizer_rule(plan, |explained_plan| {
                assert!(
                    explained_plan.contains("IndexOnlyScanExec"),
                    "Expected IndexOnlyScanExec in plan, got:\n{}",
                    explained_plan
                );
            })
            .await;
        }
    }

    #[tokio::test]
    async fn test_not_optimized() {
        let dataset = test_dataset().await;

        let queries = [
            Box::new(|scanner: &mut Scanner| {
                scanner.project(&["value", "other"]).unwrap();
            }) as Box<dyn Fn(&mut Scanner)>,
            // TODO: why do these fail?
            // Box::new(|scanner: &mut Scanner| {
            //     scanner.project(&["value", ROW_OFFSET]).unwrap();
            // }),
            // Box::new(|scanner: &mut Scanner| {
            //     scanner
            //         .project(&["value", ROW_LAST_UPDATED_AT_VERSION])
            //         .unwrap();
            // }),
            // Box::new(|scanner: &mut Scanner| {
            //     scanner.project(&["value", ROW_CREATED_AT_VERSION]).unwrap();
            // }),
            Box::new(|scanner: &mut Scanner| {
                scanner
                    .project(&["value"])
                    .unwrap()
                    .filter("other < 100")
                    .unwrap();
            }),
        ];

        for query in queries {
            let mut scanner = Scanner::new(dataset.clone());
            query(&mut scanner);
            let plan = scanner.create_plan().await.unwrap();
            test_optimizer_rule(plan, |explained_plan| {
                assert!(
                    !explained_plan.contains("IndexOnlyScanExec"),
                    "Expected no IndexOnlyScanExec in plan, but found it:\n{}",
                    explained_plan
                );
            })
            .await;
        }
    }

    #[tokio::test]
    async fn test_empty_index() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "value",
                array::step::<Int32Type>().with_nulls(&[true, true, true, false]),
            )
            .col("other", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(20))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index_builder(&["value"], IndexType::BTree, &params)
            .train(false)
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());

        let plan = dataset
            .scan()
            .project(&["value"])
            .unwrap()
            .filter("value > 10")
            .unwrap()
            .create_plan()
            .await
            .unwrap();

        let plan_str = plan_to_string(plan.as_ref());
        assert!(!plan_str.contains("IndexOnlyScanExec"));

        let optimizer = get_physical_optimizer();
        let optimized_plan = run_optimizer(&optimizer, plan);

        let plan_str = plan_to_string(optimized_plan.as_ref());
        assert!(!plan_str.contains("IndexOnlyScanExec"));
    }
}
