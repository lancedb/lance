// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex, RwLock};

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
use roaring::RoaringTreemap;
use arrow_array::{Array, RecordBatch, UInt64Array, UInt8Array};
use arrow_schema::Schema;
use arrow_select;
use futures::{stream, StreamExt};

use crate::{
    dataset::{
        transaction::{Operation, Transaction},
        write::{merge_insert::{exec::MergeInsertMetrics, MergeInsertParams, MergeStats}, write_fragments_internal, WriteParams},
    },
    Dataset, Result,
};
use lance_core::ROW_ADDR;
use lance_table::format::Fragment;
use std::collections::BTreeMap;

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
}

impl FullSchemaMergeInsertExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
    ) -> DFResult<Self> {
        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        let properties = PlanProperties::new(
            EquivalenceProperties::new(empty_schema.clone()),
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
        })
    }

    /// Returns the merge statistics if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn merge_stats(&self) -> Option<MergeStats> {
        self.merge_stats.lock().unwrap().clone()
    }

    /// Returns the transaction if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn transaction(&self) -> Option<Transaction> {
        self.transaction.lock().unwrap().clone()
    }

    /// Creates a filtered stream that captures row addresses for deletion and returns
    /// a stream with only the source data columns (no _rowaddr or action columns)
    fn create_filtered_write_stream(
        &self,
        input_stream: SendableRecordBatchStream,
        delete_row_addrs: Arc<RwLock<RoaringTreemap>>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_schema = input_stream.schema();
        
        // Find column indices
        let (rowaddr_idx, _) = input_schema
            .column_with_name(ROW_ADDR)
            .ok_or_else(|| datafusion::error::DataFusionError::Internal(
                "Expected _rowaddr column in merge insert input".to_string()
            ))?;
        
        let (action_idx, _) = input_schema
            .column_with_name("action")
            .ok_or_else(|| datafusion::error::DataFusionError::Internal(
                "Expected action column in merge insert input".to_string()
            ))?;

        // Find all source columns by looking for columns that start with "source."
        let source_column_indices: Vec<usize> = input_schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(idx, field)| {
                if field.name().starts_with("source.") {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if source_column_indices.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "No source columns found in merge insert input".to_string()
            ));
        }

        // Create output schema with only source columns (removing "source." prefix)
        let output_fields: Vec<_> = source_column_indices
            .iter()
            .map(|&idx| {
                let field = input_schema.field(idx);
                let name = field.name().strip_prefix("source.").unwrap_or(field.name());
                Arc::new(arrow_schema::Field::new(name, field.data_type().clone(), field.is_nullable()))
            })
            .collect();
        let output_schema = Arc::new(Schema::new(output_fields));

        // Create streaming transformation
        let output_schema_clone = output_schema.clone();
        let stream = input_stream.map(move |batch_result| -> DFResult<RecordBatch> {
            let batch = batch_result?;
            
            // Get row address and action arrays
            let row_addr_array = batch.column(rowaddr_idx).as_any().downcast_ref::<UInt64Array>()
                .ok_or_else(|| datafusion::error::DataFusionError::Internal(
                    "Expected UInt64Array for _rowaddr column".to_string()
                ))?;
            
            let action_array = batch.column(action_idx).as_any().downcast_ref::<UInt8Array>()
                .ok_or_else(|| datafusion::error::DataFusionError::Internal(
                    "Expected UInt8Array for action column".to_string()
                ))?;

            // Collect row addresses for deletion and track non-delete rows
            let mut keep_rows = Vec::new();
            let mut delete_addrs_guard = delete_row_addrs.write().unwrap();

            for row_idx in 0..batch.num_rows() {
                let action = action_array.value(row_idx);
                
                if action == 0 { // Delete action
                    if !row_addr_array.is_null(row_idx) {
                        let row_addr = row_addr_array.value(row_idx);
                        delete_addrs_guard.insert(row_addr);
                    }
                } else {
                    // Keep this row for writing (insert or update)
                    keep_rows.push(row_idx);
                }
            }
            drop(delete_addrs_guard);

            // If no rows to keep, return empty batch
            if keep_rows.is_empty() {
                let empty_columns: Vec<_> = output_schema_clone
                    .fields()
                    .iter()
                    .map(|field| {
                        arrow_array::new_empty_array(field.data_type())
                    })
                    .collect();
                return RecordBatch::try_new(output_schema_clone.clone(), empty_columns).map_err(datafusion::error::DataFusionError::from);
            }

            // Create indices for rows to keep
            let indices = arrow_array::UInt32Array::from(
                keep_rows.iter().map(|&i| i as u32).collect::<Vec<_>>()
            );

            // Take only the rows we want to keep
            let filtered_batch = arrow_select::take::take_record_batch(&batch, &indices)?;

            // Project only the source columns
            let output_columns: Vec<_> = source_column_indices
                .iter()
                .map(|&idx| filtered_batch.column(idx).clone())
                .collect();

            RecordBatch::try_new(output_schema_clone.clone(), output_columns).map_err(datafusion::error::DataFusionError::from)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(output_schema, stream)))
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
            datafusion::physical_plan::DisplayFormatType::Default | datafusion::physical_plan::DisplayFormatType::Verbose => {
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
                    "MergeInsert: on=[{}], when_matched={}, when_not_matched={}, when_not_matched_by_source={}",
                    on_keys,
                    when_matched,
                    when_not_matched,
                    when_not_matched_by_source
                )
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
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        todo!("Also record the metrics here")
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let _baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let _merge_metrics = MergeInsertMetrics::new(&self.metrics, partition);

        // Input schema structure based on our logical plan:
        // - target._rowaddr: Address of existing rows to update/delete  
        // - source.*: Source data columns (variable schema)
        // - action: Merge action (1=update, 2=insert, 0=delete, etc.)
        
        // Execute the input plan to get the merge data stream
        let input_stream = self.input.execute(partition, context.clone())?;
        
        // Step 1: Create streaming processor for row addresses and write data
        let delete_row_addrs = Arc::new(RwLock::new(RoaringTreemap::new()));
        let write_data_stream = self.create_filtered_write_stream(input_stream, delete_row_addrs.clone())?;

        // Use flat_map to handle the async write operation
        let dataset = self.dataset.clone();
        let merge_stats = self.merge_stats.clone();
        let transaction_holder = self.transaction.clone();
        
        let result_stream = stream::once(async move {
            // Step 2: Write new fragments using the filtered data (inserts + updates)
            let write_result = write_fragments_internal(
                Some(&dataset),
                dataset.object_store.clone(),
                &dataset.base,
                dataset.schema().clone(),
                write_data_stream,
                WriteParams::default(),
            ).await?;

            let new_fragments = write_result.default.0;

            // Step 3: Apply deletions to existing fragments
            let delete_row_addrs_clone = {
                let delete_addrs_guard = delete_row_addrs.read().unwrap();
                delete_addrs_guard.clone()
            };

            let (updated_fragments, removed_fragment_ids) = 
                Self::apply_deletions(&dataset, &delete_row_addrs_clone).await?;

            // Step 4: Create the transaction operation
            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            };

            // Step 5: Create and store the transaction
            let transaction = Transaction::new(
                dataset.manifest.version,
                operation,
                /*blobs_op=*/ None,
                None,
            );

            // Step 6: Store transaction and merge stats for later retrieval
            {
                // For now, create basic merge stats (in a full implementation, these would be
                // collected during the stream processing)
                let stats = MergeStats {
                    num_inserted_rows: 0, // TODO: Track during stream processing
                    num_updated_rows: 0,  // TODO: Track during stream processing  
                    num_deleted_rows: delete_row_addrs_clone.len(),
                    num_attempts: 1,
                };

                transaction_holder.lock().unwrap().replace(transaction);
                merge_stats.lock().unwrap().replace(stats);
            };

            // Step 7: Return empty result (write operations don't return data)
            let empty_schema = Arc::new(arrow_schema::Schema::empty());
            let empty_batch = RecordBatch::new_empty(empty_schema);
            Ok(empty_batch)
        });

        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        Ok(Box::pin(RecordBatchStreamAdapter::new(empty_schema, result_stream)))
    }
}
