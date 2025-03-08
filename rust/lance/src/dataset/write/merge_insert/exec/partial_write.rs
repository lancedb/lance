// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex};

use arrow_array::{RecordBatch, RecordBatchIterator, UInt64Array};
use datafusion::common::Result as DFResult;
use datafusion::error::DataFusionError;
use datafusion::{
    execution::{memory_pool::MemoryConsumer, SendableRecordBatchStream, TaskContext},
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
        DisplayAs, ExecutionPlan, PlanProperties,
    },
    prelude::SessionContext,
};
use datafusion::{
    physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
    scalar::ScalarValue,
};
use datafusion_expr::{UserDefinedLogicalNode, UserDefinedLogicalNodeCore};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::{stream, StreamExt};
use lance_arrow::{interleave_batches, RecordBatchExt, SchemaExt};
use lance_core::{
    datatypes::{OnMissing, OnTypeMismatch},
    Error, ROW_ADDR,
};
use lance_datafusion::{
    chunker::chunk_stream, dataframe::DataFrameExt, exec::SessionContextExt,
    utils::reader_to_stream,
};
use lance_file::version::LanceFileVersion;
use lance_table::format::Fragment;
use snafu::location;
use tokio::task::JoinSet;

use crate::{
    dataset::{
        fragment::{FileFragment, FragReadConfig},
        write::{
            merge_insert::{exec::MergeInsertMetrics, MergeInsertParams},
            open_writer, write_fragments_internal,
        },
    },
    Dataset,
};

use super::MERGE_STATS_SCHEMA;

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by doing updates by writing new data files and apending
/// them to fragments in place. This mode is most optimal when updating a subset
/// of columns, particularly when the columns not being updated are expensive to
/// rewrite.
///
/// It returns a single batch, containing the statistics.
#[derive(Debug)]
pub struct PartialUpdateMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl PartialUpdateMergeInsertExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
    ) -> DFResult<Self> {
        let properties = PlanProperties::new(
            EquivalenceProperties::new((*MERGE_STATS_SCHEMA).clone()),
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
        })
    }

    async fn update_fragments(
        dataset: Arc<Dataset>,
        source: SendableRecordBatchStream,
        session_ctx: SessionContext,
    ) -> DFResult<(Vec<Fragment>, Vec<Fragment>)> {
        // Expected source schema: _rowaddr, updated_cols*
        use datafusion::logical_expr::{col, lit};
        let mut group_stream = session_ctx
            .read_one_shot(source)?
            .with_column("_fragment_id", col(ROW_ADDR) >> lit(32))?
            .sort(vec![col(ROW_ADDR).sort(true, true)])?
            .group_by_stream(&["_fragment_id"])
            .await?;

        // Can update the fragments in parallel.
        let updated_fragments = Arc::new(Mutex::new(Vec::new()));
        let new_fragments = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = JoinSet::new();
        let task_limit = dataset.object_store().io_parallelism();
        let mut reservation =
            MemoryConsumer::new("MergeInsert").register(session_ctx.task_ctx().memory_pool());

        while let Some((frag_id, batches)) = group_stream.next().await.transpose()? {
            // We shouldn't need much more memory beyond what is already in the batches.
            let mut memory_size = batches
                .iter()
                .map(|batch| batch.get_array_memory_size())
                .sum();

            loop {
                let have_additional_cpus = tasks.len() < task_limit;
                if have_additional_cpus {
                    if reservation.try_grow(memory_size).is_ok() {
                        break;
                    } else if tasks.is_empty() {
                        // If there are no tasks running, we can bypass the pool limits.
                        // This lets us handle the case where we have a single large batch.
                        memory_size = 0;
                        break;
                    }
                    // If we can't grow the reservation, we will wait for a task to finish
                }

                if let Some(res) = tasks.join_next().await {
                    let size = res.map_err(|err| {
                        DataFusionError::Internal(format!("Error in task: {}", err))
                    })??;
                    reservation.shrink(size);
                }
            }

            match frag_id.first() {
                Some(ScalarValue::UInt64(Some(frag_id))) => {
                    let frag_id = *frag_id;
                    let fragment =
                        dataset
                            .get_fragment(frag_id as usize)
                            .ok_or_else(|| Error::Internal {
                                message: format!(
                                    "Got non-existent fragment id from merge result: {}",
                                    frag_id
                                ),
                                location: location!(),
                            })?;
                    let metadata = fragment.metadata.clone();

                    let fut = Self::handle_fragment(
                        dataset.clone(),
                        fragment,
                        metadata,
                        batches,
                        updated_fragments.clone(),
                        memory_size,
                    );
                    tasks.spawn(fut);
                }
                Some(ScalarValue::Null | ScalarValue::UInt64(None)) => {
                    let fut = Self::handle_new_fragments(
                        dataset.clone(),
                        batches,
                        new_fragments.clone(),
                        memory_size,
                    );
                    tasks.spawn(fut);
                }
                _ => {
                    return Err(DataFusionError::Internal(format!(
                        "Got non-fragment id from merge result: {:?}",
                        frag_id
                    )));
                }
            };
        }

        while let Some(res) = tasks.join_next().await {
            let size =
                res.map_err(|err| DataFusionError::Internal(format!("Error in task: {}", err)))??;
            reservation.shrink(size);
        }
        let mut updated_fragments = Arc::try_unwrap(updated_fragments)
            .unwrap()
            .into_inner()
            .unwrap();

        // Collect the updated fragments, and map the field ids. Tombstone old ones
        // as needed.
        for fragment in &mut updated_fragments {
            let updated_fields = fragment.files.last().unwrap().fields.clone();
            for data_file in &mut fragment.files.iter_mut().rev().skip(1) {
                for field in &mut data_file.fields {
                    if updated_fields.contains(field) {
                        // Tombstone these fields
                        *field = -2;
                    }
                }
            }
        }

        let new_fragments = Arc::try_unwrap(new_fragments)
            .unwrap()
            .into_inner()
            .unwrap();

        Ok((updated_fragments, new_fragments))
    }

    async fn handle_fragment(
        dataset: Arc<Dataset>,
        fragment: FileFragment,
        mut metadata: Fragment,
        mut batches: Vec<RecordBatch>,
        updated_fragments: Arc<Mutex<Vec<Fragment>>>,
        reservation_size: usize,
    ) -> DFResult<usize> {
        // batches still have _rowaddr
        let write_schema = batches[0].schema().as_ref().without_column(ROW_ADDR);
        let write_schema = dataset.local_schema().project_by_schema(
            &write_schema,
            OnMissing::Error,
            OnTypeMismatch::Error,
        )?;

        let updated_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        if Some(updated_rows) == metadata.physical_rows {
            // All rows have been updated and there are no deletions. So we
            // don't need to merge in existing values.
            // Also, because we already sorted by row address, the rows
            // will be in the correct order.

            let data_storage_version = dataset
                .manifest()
                .data_storage_format
                .lance_file_version()?;
            let mut writer = open_writer(
                dataset.object_store(),
                &write_schema,
                &dataset.base,
                data_storage_version,
            )
            .await?;

            // We need to remove rowaddr before writing.
            batches
                .iter_mut()
                .try_for_each(|batch| match batch.drop_column(ROW_ADDR) {
                    Ok(b) => {
                        *batch = b;
                        Ok(())
                    }
                    Err(e) => Err(e),
                })?;

            if data_storage_version == LanceFileVersion::Legacy {
                // Need to match the existing batch size exactly, otherwise
                // we'll get errors.
                let reader = fragment
                    .open(
                        dataset.schema(),
                        FragReadConfig::default().with_row_address(true),
                        None,
                    )
                    .await?;
                let batch_size = reader.legacy_num_rows_in_batch(0).unwrap();
                let stream = stream::iter(batches.into_iter().map(Ok));
                let stream = Box::pin(RecordBatchStreamAdapter::new(
                    Arc::new((&write_schema).into()),
                    stream,
                ));
                let mut stream = chunk_stream(stream, batch_size as usize);
                while let Some(chunk) = stream.next().await {
                    writer.write(&chunk?).await?;
                }
            } else {
                writer.write(batches.as_slice()).await?;
            }

            let (_num_rows, data_file) = writer.finish().await?;

            metadata.files.push(data_file);
            updated_fragments.lock().unwrap().push(metadata);
        } else {
            // TODO: we could skip scanning row addresses we don't need.
            let update_schema = batches[0].schema();
            let read_columns = update_schema.field_names();
            let mut updater = fragment
                .updater(
                    Some(&read_columns),
                    Some((write_schema, dataset.schema().clone())),
                    None,
                )
                .await?;

            // We will use interleave to update the rows. The first batch
            // will be the original source data, and all subsequent batches
            // will be updates.
            let mut source_batches = Vec::with_capacity(batches.len() + 1);
            source_batches.push(batches[0].clone()); // placeholder for source data
            for batch in &batches {
                source_batches.push(batch.drop_column(ROW_ADDR)?);
            }

            // This function is here to help rustc with lifetimes.
            fn get_row_addr_iter(
                batches: &[RecordBatch],
            ) -> impl Iterator<Item = (u64, (usize, usize))> + '_ + Send {
                batches.iter().enumerate().flat_map(|(batch_idx, batch)| {
                    // The index in source batches will be one more.
                    let batch_idx = batch_idx + 1;
                    let row_addrs = batch
                        .column_by_name(ROW_ADDR)
                        .unwrap()
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    row_addrs
                        .values()
                        .iter()
                        .enumerate()
                        .map(move |(offset, row_addr)| (*row_addr, (batch_idx, offset)))
                })
            }
            let mut updated_row_addr_iter = get_row_addr_iter(&batches).peekable();

            while let Some(batch) = updater.next().await? {
                source_batches[0] = batch.project_by_schema(source_batches[1].schema().as_ref())?;

                let original_row_addrs = batch
                    .column_by_name(ROW_ADDR)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let indices = original_row_addrs
                    .values()
                    .into_iter()
                    .enumerate()
                    .map(|(original_offset, row_addr)| {
                        match updated_row_addr_iter.peek() {
                            Some((updated_row_addr, _)) if *updated_row_addr == *row_addr => {
                                updated_row_addr_iter.next().unwrap().1
                            }
                            // If we have passed the next updated row address, something went wrong.
                            Some((updated_row_addr, _)) => {
                                debug_assert!(
                                    *updated_row_addr > *row_addr,
                                    "Got updated row address that is not in the original batch"
                                );
                                (0, original_offset)
                            }
                            _ => (0, original_offset),
                        }
                    })
                    .collect::<Vec<_>>();

                let updated_batch = interleave_batches(&source_batches, &indices)?;

                updater.update(updated_batch).await?;
            }

            let updated_fragment = updater.finish().await?;
            updated_fragments.lock().unwrap().push(updated_fragment);
        }
        Ok(reservation_size)
    }

    async fn handle_new_fragments(
        dataset: Arc<Dataset>,
        batches: Vec<RecordBatch>,
        new_fragments: Arc<Mutex<Vec<Fragment>>>,
        reservation_size: usize,
    ) -> DFResult<usize> {
        // Batches still have _rowaddr (used elsewhere to merge with existing data)
        // We need to remove it before writing to Lance files.
        let num_fields = batches[0].schema().fields().len();
        let mut projection = Vec::with_capacity(num_fields - 1);
        for (i, field) in batches[0].schema().fields().iter().enumerate() {
            if field.name() != ROW_ADDR {
                projection.push(i);
            }
        }
        let write_schema = Arc::new(batches[0].schema().project(&projection).unwrap());

        let batches = batches
            .into_iter()
            .map(move |batch| batch.project(&projection));
        let reader = RecordBatchIterator::new(batches, write_schema.clone());
        let stream = reader_to_stream(Box::new(reader));

        let write_schema = dataset.schema().project_by_schema(
            write_schema.as_ref(),
            OnMissing::Error,
            OnTypeMismatch::Error,
        )?;

        let fragments = write_fragments_internal(
            Some(dataset.as_ref()),
            dataset.object_store.clone(),
            &dataset.base,
            write_schema,
            stream,
            Default::default(), // TODO: support write params.
        )
        .await?;

        new_fragments.lock().unwrap().extend(fragments.default.0);
        Ok(reservation_size)
    }
}

impl DisplayAs for PartialUpdateMergeInsertExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "PartialUpdateMergeInsertExec")
    }
}

impl ExecutionPlan for PartialUpdateMergeInsertExec {
    fn name(&self) -> &str {
        "PartialUpdateMergeInsertExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        (*MERGE_STATS_SCHEMA).clone()
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
                "PartialUpdateMergeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
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
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let merge_metrics = MergeInsertMetrics::new(&self.metrics, partition);

        todo!("Execute PartialUpdateMergeInsertExec")
    }
}
