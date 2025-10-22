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
use datafusion_physical_expr::{EquivalenceProperties, LexOrdering, Partitioning};
use datafusion_physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    metrics::ExecutionPlanMetricsSet,
    sorts::sort::SortExec,
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use futures::executor::block_on;
use itertools::Itertools;
use lance_index::{
    metrics::MetricsCollector,
    scalar::{registry::ScalarIndexPluginRegistry, AnyQuery, ScalarIndex},
    DatasetIndexExt, IndexMetadata,
};
use roaring::RoaringBitmap;
use tokio::task::block_in_place;
use uuid::Uuid;

use crate::{
    index::prefilter::DatasetPreFilter,
    io::exec::{
        filtered_read::FilteredReadExec,
        utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter},
    },
    Dataset,
};

#[derive(Clone, Debug)]
pub struct IndexScanExec {
    index_name: String,
    ids: Vec<Uuid>,
    indexes: Vec<Arc<dyn ScalarIndex>>,
    properties: PlanProperties,
    limit: Option<usize>,
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
        limit: Option<usize>,
        query: Option<Arc<dyn AnyQuery>>,
        dataset: Arc<Dataset>,
        fragment_bitmaps: Vec<Option<RoaringBitmap>>,
    ) -> Self {
        let mut eq_properties = EquivalenceProperties::new(output_schema);
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
            limit,
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
            .scan(query, self.limit, batch_size, deletion_mask, collector)?
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
        todo!("Use same IO metrics from loading the indices")
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> datafusion::error::Result<datafusion_physical_plan::Statistics> {
        todo!("If index is loaded, we should be able to provide statistics eagerly")
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self {
            limit,
            ..self.clone()
        }))
    }
}

/// Physical optimizer rule to scan indexes when applicable.
///
/// Transforms `LanceRead -> SortExec -> GlobalLimitExec` into
/// `IndexScan -> SortPreservingMergeExec -> GlobalLimitExec -> Take` when the index sort order matches
/// the requested sort and when the filter in LanceRead (if any) is satisfiable
/// by the index.
///
/// Also transforms `LanceRead -> JoinExec` into
/// `IndexScan -> JoinExec -> Take` when joining on an indexed column.
#[derive(Debug)]
pub struct ScanIndexRule; // TODO: we might have to put index info in here.

impl ScanIndexRule {
    fn scannable_index(
        ds: &Dataset,
        fields: &[i32],
        ordering: Option<LexOrdering>,
        predicate: Option<Expr>,
    ) -> Vec<IndexMetadata> {
        todo!("Find a unique index that can provide the columns and is sorted on them.")
    }
}

impl PhysicalOptimizerRule for ScanIndexRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &datafusion::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        plan.transform_down(|plan| {
            // LanceRead -> SortExec
            if let Some(sort_exec) = plan.as_any().downcast_ref::<SortExec>() {
                // Check that we are limited in how much we are taking.
                // if sort_exec.fetch().is_some();
                if let Some(read) = plan.as_any().downcast_ref::<FilteredReadExec>() {
                    let ds = read.dataset();
                    if let Some(indices) = self.scannable_index(ds, todo!(), todo!(), todo!()) {
                        todo!("rewrite")
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

    #[tokio::test]
    async fn test_optimize_limit_scan() {
        todo!("validate operations get same result either way");
    }

    #[tokio::test]
    async fn test_optimize_join_index_cols() {
        todo!("validate operations get same result either way");
    }
}
