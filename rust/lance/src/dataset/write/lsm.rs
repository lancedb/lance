// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Log-Structured Merge (LSM) job for continuous writing to a dataset with WAL support.
//!
//! This module provides a long-running job that accumulates record batches and periodically
//! flushes them to a Write-Ahead Log (WAL) with watermark tracking via MemWAL index.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_ipc::writer::FileWriter as IpcFileWriter;
use lance_core::{Error, Result};
use object_store::path::Path;
use snafu::location;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio::time::interval;

use crate::index::mem_wal::append_mem_wal_entry;
use crate::Dataset;

/// Parameters for configuring the Log-Structured Merge job
#[derive(Debug, Clone)]
pub struct LogStructuredMergeJobParams {
    /// The region name in the MemWAL index
    pub region: String,
    /// The MemWAL generation to write to
    pub generation: u64,
    /// The owner ID for MemWAL operations
    pub owner_id: String,
    /// The base path for WAL files
    pub wal_base_path: String,
    /// Interval between periodic flushes (default: 200ms)
    pub flush_interval: Duration,
}

impl Default for LogStructuredMergeJobParams {
    fn default() -> Self {
        Self {
            region: "GLOBAL".to_string(),
            generation: 0,
            owner_id: String::new(),
            wal_base_path: String::new(),
            flush_interval: Duration::from_millis(200),
        }
    }
}

/// Builder for creating a LogStructuredMergeJob
pub struct LogStructuredMergeJobBuilder {
    dataset: Arc<Dataset>,
    params: LogStructuredMergeJobParams,
}

impl LogStructuredMergeJobBuilder {
    /// Create a new builder for a LogStructuredMergeJob
    pub fn new(dataset: Arc<Dataset>, region: String) -> Self {
        Self {
            dataset,
            params: LogStructuredMergeJobParams {
                region,
                ..Default::default()
            },
        }
    }

    /// Set the MemWAL generation
    pub fn with_generation(mut self, generation: u64) -> Self {
        self.params.generation = generation;
        self
    }

    /// Set the owner ID
    pub fn with_owner_id(mut self, owner_id: String) -> Self {
        self.params.owner_id = owner_id;
        self
    }

    /// Set the WAL base path
    pub fn with_wal_base_path(mut self, path: String) -> Self {
        self.params.wal_base_path = path;
        self
    }

    /// Set the flush interval
    pub fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.params.flush_interval = interval;
        self
    }

    /// Build the LogStructuredMergeJob
    pub fn build(self) -> Result<LogStructuredMergeJob> {
        if self.params.region.is_empty() {
            return Err(Error::invalid_input(
                "Region name cannot be empty",
                location!(),
            ));
        }
        if self.params.owner_id.is_empty() {
            return Err(Error::invalid_input(
                "Owner ID cannot be empty",
                location!(),
            ));
        }
        if self.params.wal_base_path.is_empty() {
            return Err(Error::invalid_input(
                "WAL base path cannot be empty",
                location!(),
            ));
        }

        Ok(LogStructuredMergeJob {
            dataset: self.dataset,
            params: self.params,
            watermark: Arc::new(AtomicU64::new(0)),
            batch_sender: None,
            flush_task: None,
            is_running: Arc::new(AtomicBool::new(false)),
        })
    }
}

/// A long-running job for continuous writes with WAL support
pub struct LogStructuredMergeJob {
    dataset: Arc<Dataset>,
    params: LogStructuredMergeJobParams,
    watermark: Arc<AtomicU64>,
    batch_sender: Option<UnboundedSender<RecordBatch>>,
    flush_task: Option<JoinHandle<Result<()>>>,
    is_running: Arc<AtomicBool>,
}

impl LogStructuredMergeJob {
    /// Start the background flush task
    pub fn start(&mut self) -> Result<()> {
        if self.is_running.load(Ordering::SeqCst) {
            return Err(Error::invalid_input("Job is already running", location!()));
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.batch_sender = Some(tx);

        let dataset = self.dataset.clone();
        let params = self.params.clone();
        let watermark = self.watermark.clone();
        let is_running = self.is_running.clone();

        is_running.store(true, Ordering::SeqCst);

        let task =
            tokio::spawn(
                async move { flush_loop(dataset, params, watermark, rx, is_running).await },
            );

        self.flush_task = Some(task);

        Ok(())
    }

    /// Write a batch to the job (accumulates until next flush)
    pub fn write(&self, batch: RecordBatch) -> Result<()> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Err(Error::invalid_input(
                "Job is not running. Call start() first",
                location!(),
            ));
        }

        if let Some(sender) = &self.batch_sender {
            sender.send(batch).map_err(|_| {
                Error::invalid_input("Failed to send batch to flush task", location!())
            })?;
        }

        Ok(())
    }

    /// Stop the job and flush remaining data
    pub async fn stop(mut self) -> Result<()> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.is_running.store(false, Ordering::SeqCst);

        // Drop the sender to signal the flush task to exit
        drop(self.batch_sender.take());

        if let Some(task) = self.flush_task.take() {
            task.await.map_err(|e| {
                Error::invalid_input(format!("Flush task failed: {}", e), location!())
            })??;
        }

        Ok(())
    }

    /// Get the current watermark value
    pub fn watermark(&self) -> u64 {
        self.watermark.load(Ordering::SeqCst)
    }
}

/// Generate a reversed binary filename for the given watermark
fn watermark_to_filename(watermark: u64) -> String {
    format!("{:064b}.ipc", u64::MAX - watermark)
}

/// Main flush loop that runs in the background
async fn flush_loop(
    mut dataset: Arc<Dataset>,
    params: LogStructuredMergeJobParams,
    watermark: Arc<AtomicU64>,
    mut batch_receiver: UnboundedReceiver<RecordBatch>,
    is_running: Arc<AtomicBool>,
) -> Result<()> {
    let mut flush_interval = interval(params.flush_interval);
    let mut accumulated_batches = Vec::new();

    loop {
        tokio::select! {
            // Receive batches from the write() calls
            batch = batch_receiver.recv() => {
                match batch {
                    Some(batch) => {
                        accumulated_batches.push(batch);
                    }
                    None => {
                        // Channel closed, flush remaining and exit
                        if !accumulated_batches.is_empty() {
                            flush_batches(
                                Arc::get_mut(&mut dataset).unwrap(),
                                &params,
                                &watermark,
                                &mut accumulated_batches,
                            )
                            .await?;
                        }
                        break;
                    }
                }
            }
            // Periodic flush
            _ = flush_interval.tick() => {
                if !accumulated_batches.is_empty() {
                    flush_batches(
                        Arc::get_mut(&mut dataset).unwrap(),
                        &params,
                        &watermark,
                        &mut accumulated_batches,
                    )
                    .await?;
                }

                // Check if we should stop
                if !is_running.load(Ordering::SeqCst) {
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Flush accumulated batches to WAL and update MemWAL index
async fn flush_batches(
    dataset: &mut Dataset,
    params: &LogStructuredMergeJobParams,
    watermark: &Arc<AtomicU64>,
    batches: &mut Vec<RecordBatch>,
) -> Result<()> {
    if batches.is_empty() {
        return Ok(());
    }

    let current_watermark = watermark.fetch_add(1, Ordering::SeqCst);
    let filename = watermark_to_filename(current_watermark);
    let wal_path = Path::from(&params.wal_base_path).child(&filename);

    // Write batches to Arrow IPC file
    let object_store = dataset.object_store();
    let writer = object_store.create(&wal_path).await?;

    // We need to create the IPC writer
    let schema = batches[0].schema();
    let mut ipc_writer = IpcFileWriter::try_new(writer, schema.as_ref()).map_err(|e| {
        Error::invalid_input(format!("Failed to create IPC writer: {}", e), location!())
    })?;

    for batch in batches.drain(..) {
        ipc_writer.write(&batch).map_err(|e| {
            Error::invalid_input(format!("Failed to write batch: {}", e), location!())
        })?;
    }

    ipc_writer.finish().map_err(|e| {
        Error::invalid_input(format!("Failed to finish IPC writer: {}", e), location!())
    })?;

    // Update MemWAL index with the new watermark
    append_mem_wal_entry(
        dataset,
        &params.region,
        params.generation,
        current_watermark,
        &params.owner_id,
    )
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_datagen::{array, gen_batch, BatchCount, FragmentCount, FragmentRowCount, RowCount};
    use lance_index::mem_wal::MEM_WAL_INDEX_NAME;

    use crate::index::mem_wal::{advance_mem_wal_generation, load_mem_wal_index_details};
    use crate::utils::test::DatagenExt;

    #[test]
    fn test_watermark_filename() {
        assert_eq!(
            watermark_to_filename(0),
            "1111111111111111111111111111111111111111111111111111111111111111.ipc"
        );
        assert_eq!(
            watermark_to_filename(1),
            "1111111111111111111111111111111111111111111111111111111111111110.ipc"
        );
        assert_eq!(
            watermark_to_filename(5),
            "1111111111111111111111111111111111111111111111111111111111111010.ipc"
        );
        assert_eq!(
            watermark_to_filename(u64::MAX),
            "0000000000000000000000000000000000000000000000000000000000000000.ipc"
        );
    }

    #[tokio::test]
    async fn test_lsm_job_basic() {
        // Create a test dataset
        let mut dataset = gen_batch()
            .col("i", array::step::<arrow_array::types::Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        // Setup MemWAL
        let owner_id = "test_owner";
        advance_mem_wal_generation(
            &mut dataset,
            "GLOBAL",
            "mem_table_location_0",
            "wal_location_0",
            None,
            owner_id,
        )
        .await
        .unwrap();

        // Create LSM job
        let mut job =
            LogStructuredMergeJobBuilder::new(Arc::new(dataset.clone()), "GLOBAL".to_string())
                .with_generation(0)
                .with_owner_id(owner_id.to_string())
                .with_wal_base_path("memory://test_wal".to_string())
                .with_flush_interval(Duration::from_millis(100))
                .build()
                .unwrap();

        // Start the job
        job.start().unwrap();

        // Write some batches
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![4, 5, 6]))],
        )
        .unwrap();

        job.write(batch1).unwrap();
        job.write(batch2).unwrap();

        // Wait for flush
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Verify watermark increased
        assert!(job.watermark() > 0);

        // Stop the job
        job.stop().await.unwrap();

        // Verify MemWAL index was updated
        let indices = dataset.load_indices().await.unwrap();
        let mem_wal_index = indices
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .expect("MemWAL index should exist");

        let details = load_mem_wal_index_details(mem_wal_index.clone()).unwrap();
        let mem_wal = &details.mem_wal_list[0];
        let entries = mem_wal.wal_entries();

        // Should have at least one entry (the watermark)
        assert!(entries.len() > 0);
    }

    #[tokio::test]
    async fn test_builder_validation() {
        let dataset = gen_batch()
            .col("i", array::step::<arrow_array::types::Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        // Empty region should fail
        let result = LogStructuredMergeJobBuilder::new(Arc::new(dataset.clone()), "".to_string())
            .with_owner_id("owner".to_string())
            .with_wal_base_path("path".to_string())
            .build();
        assert!(result.is_err());

        // Empty owner_id should fail
        let result =
            LogStructuredMergeJobBuilder::new(Arc::new(dataset.clone()), "GLOBAL".to_string())
                .with_wal_base_path("path".to_string())
                .build();
        assert!(result.is_err());

        // Empty WAL path should fail
        let result =
            LogStructuredMergeJobBuilder::new(Arc::new(dataset.clone()), "GLOBAL".to_string())
                .with_owner_id("owner".to_string())
                .build();
        assert!(result.is_err());
    }
}
