// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex};

use arrow_array::{Array, RecordBatch, UInt64Array, UInt8Array};
use arrow_schema::Schema;
use arrow_select;
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
use futures::{stream, StreamExt};
use roaring::RoaringTreemap;

use crate::{
    dataset::{
        transaction::{Operation, Transaction},
        write::{
            merge_insert::{
                assign_action::Action, exec::MergeInsertMetrics, MergeInsertParams, MergeStats,
            },
            write_fragments_internal, WriteParams,
        },
    },
    Dataset, Result,
};
use lance_core::ROW_ADDR;
use lance_table::format::Fragment;
use std::collections::BTreeMap;

/// Shared state for merge insert operations to simplify lock management
struct MergeState {
    /// Row addresses that need to be deleted, due to a row update or delete action
    delete_row_addrs: RoaringTreemap,
    /// Merge operation metrics
    metrics: MergeInsertMetrics,
}

impl MergeState {
    fn new(metrics: MergeInsertMetrics) -> Self {
        Self {
            delete_row_addrs: RoaringTreemap::new(),
            metrics,
        }
    }

    /// Process a single row based on its action, updating internal state
    fn process_row_action(
        &mut self,
        action: Action,
        row_idx: usize,
        row_addr_array: &UInt64Array,
    ) -> DFResult<Option<usize>> {
        match action {
            Action::Delete => {
                // Delete action - only delete, don't write back
                if !row_addr_array.is_null(row_idx) {
                    let row_addr = row_addr_array.value(row_idx);
                    self.delete_row_addrs.insert(row_addr);
                    self.metrics.num_deleted_rows.add(1);
                }
                Ok(None) // Don't keep this row
            }
            Action::UpdateAll => {
                // Update action - delete old row AND insert new data
                if !row_addr_array.is_null(row_idx) {
                    let row_addr = row_addr_array.value(row_idx);
                    self.delete_row_addrs.insert(row_addr);
                    // Don't count as actual delete - this is an update
                }

                self.metrics.num_updated_rows.add(1);
                Ok(Some(row_idx)) // Keep this row for writing
            }
            Action::Insert => {
                // Insert action - just insert new data
                self.metrics.num_inserted_rows.add(1);
                Ok(Some(row_idx)) // Keep this row for writing
            }
            Action::Nothing => {
                // Do nothing action - keep the row but don't count it
                Ok(None)
            }
        }
    }
}

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by moving updated rows to new fragments. This mode
/// is most optimal when updating the full schema.
///
#[derive(Debug)]
pub struct FullSchemaMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
    merge_stats: Arc<Mutex<Option<MergeStats>>>,
    transaction: Arc<Mutex<Option<Transaction>>>,
    affected_rows: Arc<Mutex<Option<RoaringTreemap>>>,
}

impl FullSchemaMergeInsertExec {
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

    /// Returns the merge statistics if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn merge_stats(&self) -> Option<MergeStats> {
        self.merge_stats.lock().ok().and_then(|guard| guard.clone())
    }

    /// Returns the transaction if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn transaction(&self) -> Option<Transaction> {
        self.transaction.lock().ok().and_then(|guard| guard.clone())
    }

    /// Returns the affected rows (deleted/updated row addresses) if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn affected_rows(&self) -> Option<RoaringTreemap> {
        self.affected_rows
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    /// Creates a filtered stream that captures row addresses for deletion and returns
    /// a stream with only the source data columns (no _rowaddr or action columns)
    fn create_filtered_write_stream(
        &self,
        input_stream: SendableRecordBatchStream,
        merge_state: Arc<Mutex<MergeState>>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_schema = input_stream.schema();

        // Find column indices
        let (rowaddr_idx, _) = input_schema.column_with_name(ROW_ADDR).ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "Expected _rowaddr column in merge insert input".to_string(),
            )
        })?;

        let (action_idx, _) = input_schema.column_with_name("action").ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "Expected action column in merge insert input".to_string(),
            )
        })?;

        // Find all data columns to write (everything except special columns)
        // The schema from DataFusion optimization may have collapsed duplicate columns
        // from the logical join, leaving us with the merged data columns plus special columns
        let total_fields = input_schema.fields().len();

        // Select all columns that are data columns (not _rowaddr or action)
        // These represent the final merged data values to write
        let data_column_indices: Vec<usize> = (0..total_fields)
            .filter(|&idx| {
                let field = input_schema.field(idx);
                let name = field.name();
                // Skip special columns: _rowaddr and action
                idx != rowaddr_idx && idx != action_idx && name != ROW_ADDR && name != "action"
            })
            .collect();

        if data_column_indices.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "No data columns found in merge insert input".to_string(),
            ));
        }

        // Create output schema with only data columns
        let output_fields: Vec<_> = data_column_indices
            .iter()
            .map(|&idx| {
                let field = input_schema.field(idx);
                // Column names don't have prefixes, they have qualifiers in the DFSchema
                Arc::new(arrow_schema::Field::new(
                    field.name(),
                    field.data_type().clone(),
                    field.is_nullable(),
                ))
            })
            .collect();
        let output_schema = Arc::new(Schema::new(output_fields));

        // Create streaming transformation
        let output_schema_clone = output_schema.clone();
        let stream = input_stream.map(move |batch_result| -> DFResult<RecordBatch> {
            let batch = batch_result?;

            // Get row address and action arrays
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
                    datafusion::error::DataFusionError::Internal(
                        "Expected UInt8Array for action column".to_string(),
                    )
                })?;

            // Process each row using the shared state
            let mut keep_rows = Vec::new();

            let mut merge_state = merge_state.lock().map_err(|e| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Failed to lock merge state: {e}"
                ))
            })?;
            for row_idx in 0..batch.num_rows() {
                let action_code = action_array.value(row_idx);
                let action = Action::try_from(action_code).map_err(|e| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "Invalid action code {action_code}: {e}"
                    ))
                })?;

                // Use the shared state to process the row action
                if let Some(keep_row_idx) =
                    merge_state.process_row_action(action, row_idx, row_addr_array)?
                {
                    keep_rows.push(keep_row_idx);
                }
            }

            // If no rows to keep, return empty batch
            if keep_rows.is_empty() {
                let empty_columns: Vec<_> = output_schema_clone
                    .fields()
                    .iter()
                    .map(|field| arrow_array::new_empty_array(field.data_type()))
                    .collect();
                return RecordBatch::try_new(output_schema_clone.clone(), empty_columns)
                    .map_err(datafusion::error::DataFusionError::from);
            }

            // Create indices for rows to keep
            let indices = arrow_array::UInt32Array::from(
                keep_rows.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            );

            // Take only the rows we want to keep
            let filtered_batch = arrow_select::take::take_record_batch(&batch, &indices)?;

            // Project only the data columns
            let output_columns: Vec<_> = data_column_indices
                .iter()
                .map(|&idx| filtered_batch.column(idx).clone())
                .collect();

            RecordBatch::try_new(output_schema_clone.clone(), output_columns)
                .map_err(datafusion::error::DataFusionError::from)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }

    /// Delete a batch of rows by row address, returns the fragments modified and the fragments removed
    async fn apply_deletions(
        dataset: &Dataset,
        removed_row_addrs: &RoaringTreemap,
    ) -> Result<(Vec<Fragment>, Vec<u64>)> {
        let bitmaps = Arc::new(removed_row_addrs.bitmaps().collect::<BTreeMap<_, _>>());

        enum FragmentChange {
            Unchanged,
            Modified(Fragment),
            Removed(u64),
        }

        let mut updated_fragments = Vec::new();
        let mut removed_fragments = Vec::new();

        let mut stream = futures::stream::iter(dataset.get_fragments())
            .map(move |fragment| {
                let bitmaps_ref = bitmaps.clone();
                async move {
                    let fragment_id = fragment.id();
                    if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                        match fragment.extend_deletions(*bitmap).await {
                            Ok(Some(new_fragment)) => {
                                Ok(FragmentChange::Modified(new_fragment.metadata))
                            }
                            Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Ok(FragmentChange::Unchanged)
                    }
                }
            })
            .buffer_unordered(dataset.object_store.io_parallelism());

        while let Some(res) = stream.next().await.transpose()? {
            match res {
                FragmentChange::Unchanged => {}
                FragmentChange::Modified(fragment) => updated_fragments.push(fragment),
                FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
            }
        }

        Ok((updated_fragments, removed_fragments))
    }
}

impl DisplayAs for FullSchemaMergeInsertExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default
            | datafusion::physical_plan::DisplayFormatType::Verbose => {
                let on_keys = self.params.on.join(", ");
                let when_matched = match &self.params.when_matched {
                    crate::dataset::WhenMatched::DoNothing => "DoNothing",
                    crate::dataset::WhenMatched::UpdateAll => "UpdateAll",
                    crate::dataset::WhenMatched::UpdateIf(_) => "UpdateIf",
                };
                let when_not_matched = if self.params.insert_not_matched {
                    "InsertAll"
                } else {
                    "DoNothing"
                };
                let when_not_matched_by_source = match &self.params.delete_not_matched_by_source {
                    crate::dataset::WhenNotMatchedBySource::Keep => "Keep",
                    crate::dataset::WhenNotMatchedBySource::Delete => "Delete",
                    crate::dataset::WhenNotMatchedBySource::DeleteIf(_) => "DeleteIf",
                };

                write!(
                    f,
                    "MergeInsert: on=[{on_keys}], when_matched={when_matched}, when_not_matched={when_not_matched}, when_not_matched_by_source={when_not_matched_by_source}"
                )
            }
            datafusion::physical_plan::DisplayFormatType::TreeRender => {
                write!(f, "MergeInsert[{}]", self.dataset.uri())
            }
        }
    }
}

impl ExecutionPlan for FullSchemaMergeInsertExec {
    fn name(&self) -> &str {
        "FullSchemaMergeInsertExec"
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
                "FullSchemaMergeInsertExec requires exactly one child".to_string(),
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

    fn required_input_distribution(&self) -> Vec<datafusion_physical_expr::Distribution> {
        // We require a single partition for the merge operation to ensure all data is processed
        vec![datafusion_physical_expr::Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // We just want one stream.
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let _baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        // Input schema structure based on our logical plan:
        // - target._rowaddr: Address of existing rows to update/delete
        // - source.*: Source data columns (variable schema)
        // - action: Merge action (1=update, 2=insert, 0=delete, etc.)

        // Execute the input plan to get the merge data stream
        let input_stream = self.input.execute(partition, context)?;

        // Step 1: Create shared state and streaming processor for row addresses and write data
        let merge_state = Arc::new(Mutex::new(MergeState::new(MergeInsertMetrics::new(
            &self.metrics,
            partition,
        ))));
        let write_data_stream =
            self.create_filtered_write_stream(input_stream, merge_state.clone())?;

        // Use flat_map to handle the async write operation
        let dataset = self.dataset.clone();
        let merge_stats_holder = self.merge_stats.clone();
        let transaction_holder = self.transaction.clone();
        let affected_rows_holder = self.affected_rows.clone();

        let result_stream = stream::once(async move {
            // Step 2: Write new fragments using the filtered data (inserts + updates)
            let write_result = write_fragments_internal(
                Some(&dataset),
                dataset.object_store.clone(),
                &dataset.base,
                dataset.schema().clone(),
                write_data_stream,
                WriteParams::default(),
            )
            .await?;

            let new_fragments = write_result.default.0;

            // Step 3: Apply deletions to existing fragments
            let merge_state =
                Arc::into_inner(merge_state).expect("MergeState should only have 1 reference now");
            let merge_state =
                Mutex::into_inner(merge_state).expect("MergeState lock should be available");
            let delete_row_addrs_clone = merge_state.delete_row_addrs;

            let (updated_fragments, removed_fragment_ids) =
                Self::apply_deletions(&dataset, &delete_row_addrs_clone).await?;

            // Step 4: Create the transaction operation
            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified: vec![], // No fields are modified in schema for upsert
            };

            // Step 5: Create and store the transaction
            let transaction = Transaction::new(
                dataset.manifest.version,
                operation,
                /*blobs_op=*/ None,
                None,
            );

            // Step 6: Store transaction, merge stats, and affected rows for later retrieval
            {
                // Get the final stats from the shared state
                let stats = MergeStats::from(&merge_state.metrics);

                if let Ok(mut transaction_guard) = transaction_holder.lock() {
                    transaction_guard.replace(transaction);
                }
                if let Ok(mut merge_stats_guard) = merge_stats_holder.lock() {
                    merge_stats_guard.replace(stats);
                }
                if let Ok(mut affected_rows_guard) = affected_rows_holder.lock() {
                    affected_rows_guard.replace(delete_row_addrs_clone);
                }
            };

            // Step 7: Return empty result (write operations don't return data)
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
