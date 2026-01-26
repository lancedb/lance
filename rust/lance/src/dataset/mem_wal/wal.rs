// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write-Ahead Log (WAL) flusher for durability.
//!
//! Batches are written as Arrow IPC streams with writer epoch metadata for fencing.
//! WAL files use bit-reversed naming to distribute files evenly across S3 keyspace.

use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::Schema as ArrowSchema;
use bytes::Bytes;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use snafu::location;
use tokio::sync::{mpsc, watch};

use uuid::Uuid;

use super::util::{region_wal_path, wal_entry_filename, WatchableOnceCell};

use super::index::IndexStore;
use super::memtable::batch_store::{BatchStore, StoredBatch};

/// Key for storing writer epoch in Arrow IPC file schema metadata.
pub const WRITER_EPOCH_KEY: &str = "writer_epoch";

/// Watcher for batch durability using watermark-based tracking.
///
/// Instead of per-batch oneshot channels, this uses a shared watch channel
/// that broadcasts the durable watermark. The watcher waits until the
/// watermark reaches or exceeds its target batch ID.
#[derive(Clone)]
pub struct BatchDurableWatcher {
    /// Watch receiver for the durable watermark.
    rx: watch::Receiver<usize>,
    /// Target batch ID to wait for.
    target_batch_position: usize,
}

impl BatchDurableWatcher {
    /// Create a new watcher for a specific batch ID.
    pub fn new(rx: watch::Receiver<usize>, target_batch_position: usize) -> Self {
        Self {
            rx,
            target_batch_position,
        }
    }

    /// Wait until the batch is durable.
    ///
    /// Returns Ok(()) when `durable_watermark >= target_batch_position`.
    pub async fn wait(&mut self) -> Result<()> {
        loop {
            let current = *self.rx.borrow();
            if current >= self.target_batch_position {
                return Ok(());
            }
            self.rx
                .changed()
                .await
                .map_err(|_| Error::io("Durable watermark channel closed", location!()))?;
        }
    }

    /// Check if the batch is already durable (non-blocking).
    pub fn is_durable(&self) -> bool {
        *self.rx.borrow() >= self.target_batch_position
    }
}

impl std::fmt::Debug for BatchDurableWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchDurableWatcher")
            .field("target_batch_position", &self.target_batch_position)
            .field("current_watermark", &*self.rx.borrow())
            .finish()
    }
}

/// A single WAL entry representing a batch of batches.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// WAL entry position (0-based, sequential).
    pub position: u64,
    /// Writer epoch at the time of write.
    pub writer_epoch: u64,
    /// Number of batches in this WAL entry.
    pub num_batches: usize,
}

/// Result of a parallel WAL flush with index update.
#[derive(Debug, Clone)]
pub struct WalFlushResult {
    /// WAL entry that was written (if any).
    pub entry: Option<WalEntry>,
    /// Duration of WAL I/O operation.
    pub wal_io_duration: std::time::Duration,
    /// Overall wall-clock duration of the index update operation.
    /// This includes any overhead from thread scheduling and context switching.
    pub index_update_duration: std::time::Duration,
    /// Per-index update durations. Key is index name, value is duration.
    pub index_update_duration_breakdown: std::collections::HashMap<String, std::time::Duration>,
    /// Number of rows indexed.
    pub rows_indexed: usize,
    /// Size of WAL data written in bytes.
    pub wal_bytes: usize,
}

/// Message to trigger a WAL flush for a specific batch store.
///
/// This unified message handles both:
/// - Normal periodic flushes (specific end_batch_position)
/// - Freeze-time flushes (end_batch_position = usize::MAX to flush all)
pub struct TriggerWalFlush {
    /// The batch store to flush from.
    pub batch_store: Arc<BatchStore>,
    /// The indexes to update in parallel (for WAL-coupled index updates).
    pub indexes: Option<Arc<IndexStore>>,
    /// End batch position (exclusive) - flush batches after max_wal_flushed_batch_position up to this.
    /// Use usize::MAX to flush all pending batches.
    pub end_batch_position: usize,
    /// Optional cell to write completion result.
    /// Uses Result<WalFlushResult, String> since Error doesn't implement Clone.
    pub done: Option<WatchableOnceCell<std::result::Result<WalFlushResult, String>>>,
}

impl std::fmt::Debug for TriggerWalFlush {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriggerWalFlush")
            .field(
                "pending_batches",
                &self.batch_store.pending_wal_flush_count(),
            )
            .field("end_batch_position", &self.end_batch_position)
            .finish()
    }
}

/// Buffer for WAL operations.
///
/// Durability is tracked via a watch channel that broadcasts the durable watermark.
/// The actual flush watermark is stored in `BatchStore.max_flushed_batch_position`.
pub struct WalFlusher {
    /// Watch channel sender for durable watermark.
    /// Broadcasts the highest batch_position that is now durable.
    durable_watermark_tx: watch::Sender<usize>,
    /// Watch channel receiver for creating new watchers.
    durable_watermark_rx: watch::Receiver<usize>,
    /// Object store for writing WAL files.
    object_store: Option<Arc<ObjectStore>>,
    /// Region ID.
    region_id: Uuid,
    /// Writer epoch (stored in WAL entries for fencing).
    writer_epoch: u64,
    /// Next WAL entry ID to use.
    next_wal_entry_position: AtomicU64,
    /// Channel to send flush messages.
    flush_tx: Option<mpsc::UnboundedSender<TriggerWalFlush>>,
    /// WAL directory path.
    wal_dir: Path,
    /// Cell for WAL flush completion notification.
    /// Created at construction and recreated after each flush.
    /// Used by backpressure to wait for WAL flushes.
    wal_flush_cell: std::sync::Mutex<Option<WatchableOnceCell<super::write::DurabilityResult>>>,
}

impl WalFlusher {
    /// Create a new WAL flusher.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path within the object store (from ObjectStore::from_uri)
    /// * `region_id` - Region UUID
    /// * `writer_epoch` - Current writer epoch
    /// * `next_wal_entry_position` - Next WAL entry ID (from recovery or 1 for new region)
    pub fn new(
        base_path: &Path,
        region_id: Uuid,
        writer_epoch: u64,
        next_wal_entry_position: u64,
    ) -> Self {
        let wal_dir = region_wal_path(base_path, &region_id);
        // Initialize durable watermark at 0 (no batches durable yet)
        let (durable_watermark_tx, durable_watermark_rx) = watch::channel(0);
        // Create initial WAL flush cell for backpressure
        let wal_flush_cell = WatchableOnceCell::new();
        Self {
            durable_watermark_tx,
            durable_watermark_rx,
            object_store: None,
            region_id,
            writer_epoch,
            next_wal_entry_position: AtomicU64::new(next_wal_entry_position),
            flush_tx: None,
            wal_dir,
            wal_flush_cell: std::sync::Mutex::new(Some(wal_flush_cell)),
        }
    }

    /// Set the object store for WAL file operations.
    pub fn set_object_store(&mut self, object_store: Arc<ObjectStore>) {
        self.object_store = Some(object_store);
    }

    /// Set the flush channel for background flush handler.
    pub fn set_flush_channel(&mut self, tx: mpsc::UnboundedSender<TriggerWalFlush>) {
        self.flush_tx = Some(tx);
    }

    /// Track a batch for WAL durability.
    ///
    /// Returns a `BatchDurableWatcher` that can be awaited for durability.
    ///
    /// Note: The actual batch data is stored in the BatchStore.
    ///
    /// # Arguments
    ///
    /// * `batch_position` - Batch ID (index in the BatchStore)
    pub fn track_batch(&self, batch_position: usize) -> BatchDurableWatcher {
        // Return a watcher that waits for this batch to become durable
        // batch_position is 0-indexed, so we wait for watermark > batch_position (i.e., >= batch_position + 1)
        BatchDurableWatcher::new(self.durable_watermark_rx.clone(), batch_position + 1)
    }

    /// Get the current durable watermark.
    pub fn durable_watermark(&self) -> usize {
        *self.durable_watermark_rx.borrow()
    }

    /// Get a watcher for WAL flush completion.
    ///
    /// Returns a watcher that resolves when the next WAL flush completes.
    /// Used by backpressure to wait for WAL flushes when the buffer is full.
    pub fn wal_flush_watcher(
        &self,
    ) -> Option<super::util::WatchableOnceCellReader<super::write::DurabilityResult>> {
        self.wal_flush_cell
            .lock()
            .unwrap()
            .as_ref()
            .map(|cell| cell.reader())
    }

    /// Signal that a WAL flush has completed and create a new cell for the next flush.
    ///
    /// Called after each successful WAL flush to notify backpressure waiters.
    fn signal_wal_flush_complete(&self) {
        let mut guard = self.wal_flush_cell.lock().unwrap();
        // Signal the current cell
        if let Some(cell) = guard.take() {
            cell.write(super::write::DurabilityResult::ok());
        }
        // Create a new cell for the next flush
        *guard = Some(WatchableOnceCell::new());
    }

    /// Trigger an immediate flush for a specific batch store up to a specific batch ID.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - The batch store to flush from
    /// * `indexes` - Optional indexes to update in parallel with WAL I/O
    /// * `end_batch_position` - End batch ID (exclusive). Use usize::MAX to flush all pending.
    /// * `done` - Optional cell to write completion result
    pub fn trigger_flush(
        &self,
        batch_store: Arc<BatchStore>,
        indexes: Option<Arc<IndexStore>>,
        end_batch_position: usize,
        done: Option<WatchableOnceCell<std::result::Result<WalFlushResult, String>>>,
    ) -> Result<()> {
        if let Some(tx) = &self.flush_tx {
            tx.send(TriggerWalFlush {
                batch_store,
                indexes,
                end_batch_position,
                done,
            })
            .map_err(|_| Error::io("WAL flush channel closed", location!()))?;
        }
        Ok(())
    }

    /// Flush batches up to a specific end_batch_position with index updates.
    ///
    /// This method flushes batches from `(max_wal_flushed_batch_position + 1)` to `end_batch_position`,
    /// allowing each trigger to flush only the batches that existed at trigger time.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - The BatchStore to read batches from
    /// * `end_batch_position` - End batch ID (exclusive) - flush up to this batch
    /// * `indexes` - Optional IndexStore to update
    ///
    /// # Returns
    ///
    /// A `WalFlushResult` with timing metrics and the WAL entry.
    /// Returns empty result if nothing to flush (already flushed past end_batch_position).
    pub async fn flush_to_with_index_update(
        &self,
        batch_store: &BatchStore,
        end_batch_position: usize,
        indexes: Option<Arc<IndexStore>>,
    ) -> Result<WalFlushResult> {
        // Get current flush position from per-memtable watermark (inclusive)
        // start_batch_position is the first batch to flush
        let start_batch_position = batch_store
            .max_flushed_batch_position()
            .map(|w| w + 1)
            .unwrap_or(0);

        // If we've already flushed past this end, nothing to do
        if start_batch_position >= end_batch_position {
            return Ok(WalFlushResult {
                entry: None,
                wal_io_duration: std::time::Duration::ZERO,
                index_update_duration: std::time::Duration::ZERO,
                index_update_duration_breakdown: std::collections::HashMap::new(),
                rows_indexed: 0,
                wal_bytes: 0,
            });
        }

        let object_store = self
            .object_store
            .as_ref()
            .ok_or_else(|| Error::io("Object store not set on WAL flusher", location!()))?;

        let wal_entry_position = self.next_wal_entry_position.fetch_add(1, Ordering::SeqCst);
        let final_path = self.wal_entry_path(wal_entry_position);

        // Collect batches in range [start_batch_position, end_batch_position)
        let mut stored_batches: Vec<StoredBatch> =
            Vec::with_capacity(end_batch_position - start_batch_position);

        for batch_position in start_batch_position..end_batch_position {
            if let Some(stored) = batch_store.get(batch_position) {
                stored_batches.push(stored.clone());
            }
        }

        if stored_batches.is_empty() {
            return Ok(WalFlushResult {
                entry: None,
                wal_io_duration: std::time::Duration::ZERO,
                index_update_duration: std::time::Duration::ZERO,
                index_update_duration_breakdown: std::collections::HashMap::new(),
                rows_indexed: 0,
                wal_bytes: 0,
            });
        }

        let rows_to_index: usize = stored_batches.iter().map(|b| b.num_rows).sum();
        let num_batches = stored_batches.len();

        // Prepare WAL I/O data
        let schema = stored_batches[0].data.schema();
        let mut metadata = schema.metadata().clone();
        metadata.insert(WRITER_EPOCH_KEY.to_string(), self.writer_epoch.to_string());
        let schema_with_epoch = Arc::new(ArrowSchema::new_with_metadata(
            schema.fields().to_vec(),
            metadata,
        ));

        // Serialize WAL data as IPC stream (schema at start, no footer)
        let mut buffer = Vec::new();
        {
            let mut writer =
                StreamWriter::try_new(&mut buffer, &schema_with_epoch).map_err(|e| {
                    Error::io(
                        format!("Failed to create Arrow IPC stream writer: {}", e),
                        location!(),
                    )
                })?;

            for stored in &stored_batches {
                writer.write(&stored.data).map_err(|e| {
                    Error::io(
                        format!("Failed to write batch to Arrow IPC stream: {}", e),
                        location!(),
                    )
                })?;
            }

            writer.finish().map_err(|e| {
                Error::io(
                    format!("Failed to finish Arrow IPC stream: {}", e),
                    location!(),
                )
            })?;
        }

        let wal_bytes = buffer.len();

        // WAL I/O and index update in parallel
        let wal_path = final_path.clone();
        let wal_data = Bytes::from(buffer);
        let store = object_store.clone();

        // Returns (overall_duration, per_index_durations)
        let (wal_result, index_result) = if let Some(idx_registry) = indexes {
            let wal_future = async {
                let start = Instant::now();
                store
                    .inner
                    .put(&wal_path, wal_data.into())
                    .await
                    .map_err(|e| {
                        Error::io(format!("Failed to write WAL file: {}", e), location!())
                    })?;
                Ok::<_, Error>(start.elapsed())
            };

            let index_future = async {
                let start = Instant::now();
                let per_index = tokio::task::spawn_blocking(move || {
                    idx_registry.insert_batches_parallel(&stored_batches)
                })
                .await
                .map_err(|e| Error::Internal {
                    message: format!("Index update task panicked: {}", e),
                    location: location!(),
                })??;
                Ok::<_, Error>((start.elapsed(), per_index))
            };

            tokio::join!(wal_future, index_future)
        } else {
            let wal_future = async {
                let start = Instant::now();
                store
                    .inner
                    .put(&wal_path, wal_data.into())
                    .await
                    .map_err(|e| {
                        Error::io(format!("Failed to write WAL file: {}", e), location!())
                    })?;
                Ok::<_, Error>(start.elapsed())
            };

            (
                wal_future.await,
                Ok((std::time::Duration::ZERO, std::collections::HashMap::new())),
            )
        };

        let wal_io_duration = wal_result?;
        let (index_update_duration, index_update_duration_breakdown) = index_result?;

        // Update per-memtable watermark (inclusive: last batch ID that was flushed)
        batch_store.set_max_flushed_batch_position(end_batch_position - 1);

        // Notify durability waiters (global channel)
        let _ = self.durable_watermark_tx.send(end_batch_position);
        // Signal WAL flush completion for backpressure waiters
        self.signal_wal_flush_complete();

        let entry = WalEntry {
            position: wal_entry_position,
            writer_epoch: self.writer_epoch,
            num_batches,
        };

        Ok(WalFlushResult {
            entry: Some(entry),
            wal_io_duration,
            index_update_duration,
            index_update_duration_breakdown,
            rows_indexed: rows_to_index,
            wal_bytes,
        })
    }

    /// Get the current WAL ID (last written + 1).
    pub fn next_wal_entry_position(&self) -> u64 {
        self.next_wal_entry_position.load(Ordering::SeqCst)
    }

    /// Get the region ID.
    pub fn region_id(&self) -> Uuid {
        self.region_id
    }

    /// Get the writer epoch.
    pub fn writer_epoch(&self) -> u64 {
        self.writer_epoch
    }

    /// Get the path for a WAL entry.
    pub fn wal_entry_path(&self, wal_entry_position: u64) -> Path {
        let filename = wal_entry_filename(wal_entry_position);
        self.wal_dir.child(filename.as_str())
    }
}

/// A WAL entry read from storage for replay.
#[derive(Debug)]
pub struct WalEntryData {
    /// Writer epoch from the WAL entry.
    pub writer_epoch: u64,
    /// Record batches from the WAL entry.
    pub batches: Vec<RecordBatch>,
}

impl WalEntryData {
    /// Read a WAL entry from storage.
    ///
    /// # Arguments
    ///
    /// * `object_store` - Object store to read from
    /// * `path` - Path to the WAL entry (Arrow IPC file)
    ///
    /// # Returns
    ///
    /// The parsed WAL entry data, or an error if reading/parsing fails.
    pub async fn read(object_store: &ObjectStore, path: &Path) -> Result<Self> {
        // Read the file
        let data = object_store
            .inner
            .get(path)
            .await
            .map_err(|e| Error::io(format!("Failed to read WAL file: {}", e), location!()))?
            .bytes()
            .await
            .map_err(|e| Error::io(format!("Failed to get WAL file bytes: {}", e), location!()))?;

        // Parse as Arrow IPC stream
        let cursor = Cursor::new(data);
        let reader = StreamReader::try_new(cursor, None).map_err(|e| {
            Error::io(
                format!("Failed to open Arrow IPC stream reader: {}", e),
                location!(),
            )
        })?;

        // Extract writer epoch from schema metadata (at start of stream)
        let schema = reader.schema();
        let writer_epoch = schema
            .metadata()
            .get(WRITER_EPOCH_KEY)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        // Read all batches
        let mut batches = Vec::new();
        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                Error::io(
                    format!("Failed to read batch from Arrow IPC stream: {}", e),
                    location!(),
                )
            })?;
            batches.push(batch);
        }

        Ok(Self {
            writer_epoch,
            batches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, temp_dir)
    }

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &Schema, num_rows: usize) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
                Arc::new(StringArray::from_iter_values(
                    (0..num_rows).map(|i| format!("name_{}", i)),
                )),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_wal_flusher_track_batch() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let mut buffer = WalFlusher::new(&base_path, region_id, 1, 1);
        buffer.set_object_store(store);

        // Track a batch
        let watcher = buffer.track_batch(0);

        // Watcher should not be durable yet
        assert!(!watcher.is_durable());
    }

    #[tokio::test]
    async fn test_wal_flusher_flush_to_with_index_update() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let mut buffer = WalFlusher::new(&base_path, region_id, 1, 1);
        buffer.set_object_store(store);

        // Create a BatchStore with some data
        let schema = create_test_schema();
        let batch1 = create_test_batch(&schema, 10);
        let batch2 = create_test_batch(&schema, 5);

        let batch_store = BatchStore::with_capacity(10);
        batch_store.append(batch1).unwrap();
        batch_store.append(batch2).unwrap();

        // Track batch IDs in WAL flusher
        let mut watcher1 = buffer.track_batch(0);
        let mut watcher2 = buffer.track_batch(1);

        // Verify initial state
        assert!(!watcher1.is_durable());
        assert!(!watcher2.is_durable());
        assert!(batch_store.max_flushed_batch_position().is_none());

        // Flush all pending batches
        let result = buffer
            .flush_to_with_index_update(&batch_store, batch_store.len(), None)
            .await
            .unwrap();
        let entry = result.entry.unwrap();
        assert_eq!(entry.position, 1);
        assert_eq!(entry.writer_epoch, 1);
        assert_eq!(entry.num_batches, 2);
        // After flushing 2 batches (positions 0 and 1), max flushed position is 1 (inclusive)
        assert_eq!(batch_store.max_flushed_batch_position(), Some(1));

        // Watchers should be notified
        watcher1.wait().await.unwrap();
        watcher2.wait().await.unwrap();
        assert!(watcher1.is_durable());
        assert!(watcher2.is_durable());
    }

    #[tokio::test]
    async fn test_wal_entry_read() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let mut buffer = WalFlusher::new(&base_path, region_id, 42, 1);
        buffer.set_object_store(store.clone());

        // Create a BatchStore with some data
        let schema = create_test_schema();
        let batch_store = BatchStore::with_capacity(10);
        batch_store.append(create_test_batch(&schema, 10)).unwrap();
        batch_store.append(create_test_batch(&schema, 5)).unwrap();

        // Track batch IDs and flush all pending batches
        let _watcher1 = buffer.track_batch(0);
        let _watcher2 = buffer.track_batch(1);
        let result = buffer
            .flush_to_with_index_update(&batch_store, batch_store.len(), None)
            .await
            .unwrap();
        let entry = result.entry.unwrap();

        // Read back the WAL entry
        let wal_path = buffer.wal_entry_path(entry.position);
        let wal_data = WalEntryData::read(&store, &wal_path).await.unwrap();

        // Verify the read data
        assert_eq!(wal_data.writer_epoch, 42);
        assert_eq!(wal_data.batches.len(), 2);
        assert_eq!(wal_data.batches[0].num_rows(), 10);
        assert_eq!(wal_data.batches[1].num_rows(), 5);
    }
}
