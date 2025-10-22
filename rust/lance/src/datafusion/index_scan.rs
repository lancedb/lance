// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, sync::Arc};

use arrow_schema::SchemaRef;
use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    error::DataFusionError,
    physical_optimizer::PhysicalOptimizerRule,
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    metrics::ExecutionPlanMetricsSet,
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics,
};
use futures::executor::block_on;
use itertools::Itertools;
use lance_core::{datatypes::format_field_path, ROW_ADDR};
use lance_index::{
    metrics::{MetricsCollector, NoOpMetricsCollector},
    scalar::{AnyQuery, ScalarIndex},
    DatasetIndexExt, ScalarIndexCriteria,
};
use roaring::RoaringBitmap;
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
pub struct ScanIndexRule; // TODO: we might have to put index info in here.

impl ScanIndexRule {
    fn rewrite_filtered_read(
        read: &FilteredReadExec,
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

        if !projection.with_row_id
            || projection.with_row_last_updated_at_version
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

        let column_name = if let Some(ancestors) = schema.field_ancestry_by_id(field.id) {
            let segments: Vec<&str> = ancestors.iter().map(|f| f.name.as_str()).collect();
            format_field_path(&segments)
        } else {
            field.name.clone()
        };

        if let Some(filter_expr) = options.full_filter.as_ref() {
            let filter_columns = Planner::column_names_in_expr(filter_expr);
            if filter_columns.iter().any(|col| col != &column_name) {
                return Ok(None);
            }
        }

        // Find a matching scalar index for the column.
        let criteria = ScalarIndexCriteria::default().for_column(column_name.as_str());
        let index_meta = block_in_place(|| block_on(dataset.load_scalar_index(criteria)))
            .map_err(|err| DataFusionError::External(Box::new(err)))?;
        let Some(index_meta) = index_meta else {
            return Ok(None);
        };

        // Load the index implementation.
        let index_uuid = index_meta.uuid.to_string();
        let column_name_clone = column_name.clone();
        let index = block_in_place(|| {
            block_on(dataset.open_scalar_index(
                column_name_clone.as_str(),
                &index_uuid,
                &NoOpMetricsCollector,
            ))
        })
        .map_err(|err| DataFusionError::External(Box::new(err)))?;

        // Determine the schema produced by the index scan.
        let read_schema = read.schema();
        let needs_row_addr = projection.with_row_addr;

        let index_schema = if needs_row_addr {
            let fields: Vec<_> = read_schema
                .fields()
                .iter()
                .filter(|f| f.name() != ROW_ADDR)
                .cloned()
                .collect();
            Arc::new(arrow_schema::Schema::new(fields))
        } else {
            read_schema.clone()
        };

        let index_exec: Arc<dyn ExecutionPlan> = Arc::new(IndexScanExec::new(
            index_meta.name.clone(),
            vec![index_meta.uuid],
            vec![index],
            index_schema,
            None,
            dataset.clone(),
            vec![index_meta.fragment_bitmap.clone()],
        ));

        if needs_row_addr {
            let Some(rowaddr_pos) = read_schema
                .fields()
                .iter()
                .position(|f| f.name() == ROW_ADDR)
            else {
                return Ok(None);
            };
            let add_row_addr = AddRowAddrExec::try_new(index_exec, dataset.clone(), rowaddr_pos)
                .map_err(|err| DataFusionError::External(Box::new(err)))?;
            Ok(Some(Arc::new(add_row_addr)))
        } else {
            Ok(Some(index_exec))
        }
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
                if let Some(new_plan) = Self::rewrite_filtered_read(read)? {
                    return Ok(Transformed::yes(new_plan));
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

    #[tokio::test]
    async fn test_optimize_limit_scan() {
        todo!("validate operations get same result either way");
    }

    #[tokio::test]
    async fn test_optimize_join_index_cols() {
        todo!("validate operations get same result either way");
    }
}
