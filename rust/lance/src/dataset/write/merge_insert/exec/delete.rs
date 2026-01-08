// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex};

use arrow_array::{Array, RecordBatch, UInt64Array, UInt8Array};
use datafusion::common::Result as DFResult;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::{
    execution::{SendableRecordBatchStream, TaskContext},
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
        DisplayAs, ExecutionPlan, PlanProperties,
    },
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::StreamExt;
use lance_core::ROW_ADDR;
use roaring::RoaringTreemap;

use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::write::merge_insert::assign_action::Action;
use crate::dataset::write::merge_insert::{MergeInsertParams, MergeStats, MERGE_ACTION_COLUMN};
use crate::Dataset;

use super::{apply_deletions, MergeInsertMetrics};

/// Specialized physical execution node for delete-only merge insert operations.
///
/// This is an optimized path for when `WhenMatched::Delete` is used without inserts.
/// Unlike `FullSchemaMergeInsertExec`, this node:
/// - Only reads `_rowaddr` and `__action` columns (no data columns needed)
/// - Skips the write step entirely (no new fragments created)
/// - Only applies deletions to existing fragments
///
/// This is significantly more efficient for bulk delete operations where
/// we only need to identify matching rows and mark them as deleted.
#[derive(Debug)]
pub struct DeleteOnlyMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
    merge_stats: Arc<Mutex<Option<MergeStats>>>,
    transaction: Arc<Mutex<Option<Transaction>>>,
    affected_rows: Arc<Mutex<Option<RoaringTreemap>>>,
}

impl DeleteOnlyMergeInsertExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
    ) -> DFResult<Self> {
        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        let properties = PlanProperties::new(
            EquivalenceProperties::new(empty_schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self {
            input,
            dataset,
            params,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            merge_stats: Arc::new(Mutex::new(None)),
            transaction: Arc::new(Mutex::new(None)),
            affected_rows: Arc::new(Mutex::new(None)),
        })
    }

    /// Takes the merge statistics if the execution has completed.
    pub fn merge_stats(&self) -> Option<MergeStats> {
        self.merge_stats
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Takes the transaction if the execution has completed.
    pub fn transaction(&self) -> Option<Transaction> {
        self.transaction
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Takes the affected rows (deleted row addresses) if the execution has completed.
    pub fn affected_rows(&self) -> Option<RoaringTreemap> {
        self.affected_rows
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    async fn collect_deletions(
        mut input_stream: SendableRecordBatchStream,
        metrics: MergeInsertMetrics,
    ) -> DFResult<RoaringTreemap> {
        let schema = input_stream.schema();

        let (rowaddr_idx, _) = schema.column_with_name(ROW_ADDR).ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "Expected _rowaddr column in delete-only merge insert input".to_string(),
            )
        })?;

        let (action_idx, _) = schema
            .column_with_name(MERGE_ACTION_COLUMN)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Expected {} column in delete-only merge insert input",
                    MERGE_ACTION_COLUMN
                ))
            })?;

        let mut delete_row_addrs = RoaringTreemap::new();

        while let Some(batch_result) = input_stream.next().await {
            let batch = batch_result?;

            let row_addr_array = batch
                .column(rowaddr_idx)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(
                        "Expected UInt64Array for _rowaddr column".to_string(),
                    )
                })?;

            let action_array = batch
                .column(action_idx)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "Expected UInt8Array for {} column",
                        MERGE_ACTION_COLUMN
                    ))
                })?;

            for row_idx in 0..batch.num_rows() {
                let action_code = action_array.value(row_idx);
                let action = Action::try_from(action_code).map_err(|e| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "Invalid action code {}: {}",
                        action_code, e
                    ))
                })?;

                if action == Action::Delete && !row_addr_array.is_null(row_idx) {
                    let row_addr = row_addr_array.value(row_idx);
                    delete_row_addrs.insert(row_addr);
                    metrics.num_deleted_rows.add(1);
                }
            }
        }

        Ok(delete_row_addrs)
    }
}

impl DisplayAs for DeleteOnlyMergeInsertExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default
            | datafusion::physical_plan::DisplayFormatType::Verbose => {
                let on_keys = self.params.on.join(", ");
                write!(
                    f,
                    "DeleteOnlyMergeInsert: on=[{}], when_matched=Delete, when_not_matched=DoNothing",
                    on_keys
                )
            }
            datafusion::physical_plan::DisplayFormatType::TreeRender => {
                write!(f, "DeleteOnlyMergeInsert[{}]", self.dataset.uri())
            }
        }
    }
}

impl ExecutionPlan for DeleteOnlyMergeInsertExec {
    fn name(&self) -> &str {
        "DeleteOnlyMergeInsertExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(arrow_schema::Schema::empty())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "DeleteOnlyMergeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
            merge_stats: self.merge_stats.clone(),
            transaction: self.transaction.clone(),
            affected_rows: self.affected_rows.clone(),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }

    fn required_input_distribution(&self) -> Vec<datafusion_physical_expr::Distribution> {
        vec![datafusion_physical_expr::Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let _baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let metrics = MergeInsertMetrics::new(&self.metrics, partition);
        let input_stream = self.input.execute(partition, context)?;

        let dataset = self.dataset.clone();
        let merge_stats_holder = self.merge_stats.clone();
        let transaction_holder = self.transaction.clone();
        let affected_rows_holder = self.affected_rows.clone();
        let mem_wal_to_merge = self.params.mem_wal_to_merge.clone();

        let result_stream = futures::stream::once(async move {
            let delete_row_addrs = Self::collect_deletions(input_stream, metrics).await?;

            let (updated_fragments, removed_fragment_ids) =
                apply_deletions(&dataset, &delete_row_addrs)
                    .await
                    .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments: vec![],
                fields_modified: vec![],
                mem_wal_to_merge,
                fields_for_preserving_frag_bitmap: dataset
                    .schema()
                    .fields
                    .iter()
                    .map(|f| f.id as u32)
                    .collect(),
                update_mode: None,
                inserted_rows_filter: None, // Delete-only operations don't insert rows
            };

            let transaction = Transaction::new(dataset.manifest.version, operation, None);

            let num_deleted = delete_row_addrs.len();
            let stats = MergeStats {
                num_deleted_rows: num_deleted,
                num_inserted_rows: 0,
                num_updated_rows: 0,
                bytes_written: 0,
                num_files_written: 0,
                num_attempts: 1,
            };

            if let Ok(mut transaction_guard) = transaction_holder.lock() {
                transaction_guard.replace(transaction);
            }
            if let Ok(mut merge_stats_guard) = merge_stats_holder.lock() {
                merge_stats_guard.replace(stats);
            }
            if let Ok(mut affected_rows_guard) = affected_rows_holder.lock() {
                affected_rows_guard.replace(delete_row_addrs);
            }

            let empty_schema = Arc::new(arrow_schema::Schema::empty());
            let empty_batch = RecordBatch::new_empty(empty_schema);
            Ok(empty_batch)
        });

        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            empty_schema,
            result_stream,
        )))
    }
}
