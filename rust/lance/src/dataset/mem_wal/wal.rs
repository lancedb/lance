// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write-Ahead Log (WAL) flusher for durability.
//!
//! Batches are written as Arrow IPC streams with writer epoch metadata for fencing.
//! WAL files use bit-reversed naming to distribute files evenly across S3 keyspace.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::Schema as ArrowSchema;
use bytes::Bytes;
use futures::StreamExt;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use object_store::{PutMode, PutOptions};
use tokio::sync::{Mutex, mpsc, watch};

use tracing::instrument;
use uuid::Uuid;

use super::manifest::ShardManifestStore;
use super::util::{
    WatchableOnceCell, parse_bit_reversed_filename, shard_wal_path, wal_entry_filename,
};

use super::index::IndexStore;
use super::memtable::batch_store::{BatchStore, StoredBatch};

/// Key for storing writer epoch in Arrow IPC file schema metadata.
pub const WRITER_EPOCH_KEY: &str = "writer_epoch";

/// Watcher for batch durability using watermark-based tracking.
///
/// Uses a shared watch channel that broadcasts the durable watermark.
/// The watcher waits until the watermark reaches or exceeds its target batch ID.
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
                .map_err(|_| Error::io("Durable watermark channel closed"))?;
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
    /// Shard ID.
    shard_id: Uuid,
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
    /// * `shard_id` - Shard UUID
    /// * `writer_epoch` - Current writer epoch
    /// * `next_wal_entry_position` - Next WAL entry ID (from recovery or 1 for new shard)
    pub fn new(
        base_path: &Path,
        shard_id: Uuid,
        writer_epoch: u64,
        next_wal_entry_position: u64,
    ) -> Self {
        let wal_dir = shard_wal_path(base_path, &shard_id);
        // Initialize durable watermark at 0 (no batches durable yet)
        let (durable_watermark_tx, durable_watermark_rx) = watch::channel(0);
        // Create initial WAL flush cell for backpressure
        let wal_flush_cell = WatchableOnceCell::new();
        Self {
            durable_watermark_tx,
            durable_watermark_rx,
            object_store: None,
            shard_id,
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
    /// The actual batch data is stored in the BatchStore.
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
            .map_err(|_| Error::io("WAL flush channel closed"))?;
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
    #[instrument(name = "wal_flush", level = "info", skip_all, fields(shard_id = %self.shard_id, end_batch_position, has_indexes = indexes.is_some()))]
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
            .ok_or_else(|| Error::io("Object store not set on WAL flusher"))?;

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
                    Error::io(format!("Failed to create Arrow IPC stream writer: {}", e))
                })?;

            for stored in &stored_batches {
                writer.write(&stored.data).map_err(|e| {
                    Error::io(format!("Failed to write batch to Arrow IPC stream: {}", e))
                })?;
            }

            writer
                .finish()
                .map_err(|e| Error::io(format!("Failed to finish Arrow IPC stream: {}", e)))?;
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
                    .map_err(|e| Error::io(format!("Failed to write WAL file: {}", e)))?;
                Ok::<_, Error>(start.elapsed())
            };

            let index_future = async {
                let start = Instant::now();
                let per_index = tokio::task::spawn_blocking(move || {
                    idx_registry.insert_batches_parallel(&stored_batches)
                })
                .await
                .map_err(|e| Error::internal(format!("Index update task panicked: {}", e)))??;
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
                    .map_err(|e| Error::io(format!("Failed to write WAL file: {}", e)))?;
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

    /// Get the shard ID.
    pub fn shard_id(&self) -> Uuid {
        self.shard_id
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
    #[instrument(name = "wal_entry_read", level = "debug", skip_all, fields(path = %path))]
    pub async fn read(object_store: &ObjectStore, path: &Path) -> Result<Self> {
        // Read the file
        let data = object_store
            .inner
            .get(path)
            .await
            .map_err(|e| Error::io(format!("Failed to read WAL file: {}", e)))?
            .bytes()
            .await
            .map_err(|e| Error::io(format!("Failed to get WAL file bytes: {}", e)))?;

        // Parse as Arrow IPC stream
        let cursor = Cursor::new(data);
        let reader = StreamReader::try_new(cursor, None)
            .map_err(|e| Error::io(format!("Failed to open Arrow IPC stream reader: {}", e)))?;

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
                Error::io(format!("Failed to read batch from Arrow IPC stream: {}", e))
            })?;
            batches.push(batch);
        }

        Ok(Self {
            writer_epoch,
            batches,
        })
    }
}

// ============================================================================
// Generic WAL Appender and Tailer
// ============================================================================

const FIRST_WAL_ENTRY_POSITION: u64 = 1;
const MAX_APPEND_CREATE_CONFLICTS: usize = 1024;
const APPEND_CONFLICT_REFRESH_INTERVAL: usize = 16;
const MAX_CURSOR_PROBE: u64 = 4096;

/// Result of appending a WAL entry.
#[derive(Debug, Clone)]
pub struct WalAppendResult {
    pub shard_id: Uuid,
    pub entry_position: u64,
    pub num_batches: usize,
    pub num_rows: usize,
    pub wal_bytes: usize,
}

/// A WAL entry read from storage.
#[derive(Debug, Clone)]
pub struct WalReadEntry {
    pub shard_id: Uuid,
    pub entry_position: u64,
    /// Writer epoch recorded in the WAL entry's IPC schema metadata.
    /// Replay logic uses this to fence-check against the current epoch.
    pub writer_epoch: u64,
    pub batches: Vec<RecordBatch>,
}

/// WAL appender for a single MemWAL shard with epoch fencing.
///
/// Writes Arrow IPC entries atomically using put-if-not-exists. On conflict,
/// retries with the next position. Checks fencing on conflict or PUT failure.
#[derive(Debug)]
pub struct WalAppender {
    object_store: Arc<ObjectStore>,
    wal_dir: Path,
    manifest_store: Arc<ShardManifestStore>,
    shard_id: Uuid,
    writer_epoch: u64,
    next_entry_position: Mutex<Option<u64>>,
}

impl WalAppender {
    /// Open a WAL appender and claim a new writer epoch.
    pub async fn open(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        shard_id: Uuid,
        shard_spec_id: u32,
    ) -> Result<Self> {
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (writer_epoch, _) = manifest_store.claim_epoch(shard_spec_id).await?;
        Ok(Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            manifest_store,
            shard_id,
            writer_epoch,
            next_entry_position: Mutex::new(None),
        })
    }

    /// Shard id.
    pub fn shard_id(&self) -> Uuid {
        self.shard_id
    }

    /// Writer epoch recorded in the shard manifest.
    pub fn writer_epoch(&self) -> u64 {
        self.writer_epoch
    }

    /// Append batches as one durable WAL entry.
    pub async fn append(&self, batches: Vec<RecordBatch>) -> Result<WalAppendResult> {
        validate_appender_batches(&batches)?;
        let wal_data = Bytes::from(serialize_appender_batches(&batches, self.writer_epoch)?);
        let wal_bytes = wal_data.len();
        let num_batches = batches.len();
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum();

        let mut next_pos = self.next_entry_position.lock().await;
        if next_pos.is_none() {
            *next_pos = Some(self.scan_next_position().await?);
        }

        let mut conflicts = 0;
        loop {
            let pos = next_pos.ok_or_else(|| {
                Error::internal(format!(
                    "missing cached WAL position for shard {}",
                    self.shard_id
                ))
            })?;
            match atomic_put(
                self.object_store.as_ref(),
                &self.wal_dir,
                &wal_entry_filename(pos),
                wal_data.clone(),
            )
            .await
            {
                Ok(()) => {
                    *next_pos = Some(pos.checked_add(1).ok_or_else(|| {
                        Error::io(format!("WAL position overflow for shard {}", self.shard_id))
                    })?);
                    return Ok(WalAppendResult {
                        shard_id: self.shard_id,
                        entry_position: pos,
                        num_batches,
                        num_rows,
                        wal_bytes,
                    });
                }
                Err(AtomicPutError::AlreadyExists) => {
                    self.check_fenced().await?;
                    conflicts += 1;
                    if conflicts >= MAX_APPEND_CREATE_CONFLICTS {
                        return Err(Error::io(format!(
                            "WAL append for shard {} failed after {} conflicts",
                            self.shard_id, conflicts
                        )));
                    }
                    if conflicts % APPEND_CONFLICT_REFRESH_INTERVAL == 0 {
                        *next_pos = Some(self.scan_next_position().await?);
                    } else {
                        *next_pos = Some(pos.checked_add(1).ok_or_else(|| {
                            Error::io(format!("WAL position overflow for shard {}", self.shard_id))
                        })?);
                    }
                }
                Err(AtomicPutError::Other(error)) => {
                    self.check_fenced().await?;
                    return Err(error);
                }
            }
        }
    }

    /// Check that this writer's epoch has not been fenced.
    pub async fn check_fenced(&self) -> Result<()> {
        self.manifest_store.check_fenced(self.writer_epoch).await
    }

    async fn scan_next_position(&self) -> Result<u64> {
        scan_next_position(self.object_store.as_ref(), &self.wal_dir, self.shard_id).await
    }
}

/// Ordered reader for MemWAL entries from a single shard.
///
/// Uses `wal_entry_position_last_seen` from the shard manifest as a cursor
/// hint for `next_position()`, probing forward from the hint to find the true
/// tip before falling back to a full directory listing.
#[derive(Debug, Clone)]
pub struct WalTailer {
    object_store: Arc<ObjectStore>,
    wal_dir: Path,
    manifest_store: Arc<ShardManifestStore>,
    shard_id: Uuid,
    update_cursor: bool,
}

impl WalTailer {
    /// Create a WAL tailer for a shard.
    pub fn new(object_store: Arc<ObjectStore>, base_path: Path, shard_id: Uuid) -> Self {
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            manifest_store,
            shard_id,
            update_cursor: false,
        }
    }

    /// Enable async best-effort cursor updates on read.
    ///
    /// When enabled, successful `read_entry` calls asynchronously update
    /// `wal_entry_position_last_seen` in the shard manifest.
    pub fn with_cursor_updates(mut self, enabled: bool) -> Self {
        self.update_cursor = enabled;
        self
    }

    /// Read a WAL entry at the given position. Returns `None` if no entry exists.
    pub async fn read_entry(&self, entry_position: u64) -> Result<Option<WalReadEntry>> {
        let path = self.wal_dir.child(wal_entry_filename(entry_position));
        let data = match self.object_store.inner.get(&path).await {
            Ok(data) => data,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(e) => {
                return Err(Error::io(format!(
                    "failed to read WAL entry {} for shard {}: {}",
                    entry_position, self.shard_id, e
                )));
            }
        };
        let bytes = data.bytes().await.map_err(|e| {
            Error::io(format!(
                "failed to read WAL entry bytes at {} for shard {}: {}",
                path, self.shard_id, e
            ))
        })?;
        let (writer_epoch, batches) = deserialize_appender_batches(bytes)?;

        if self.update_cursor {
            let ms = self.manifest_store.clone();
            tokio::spawn(async move {
                let _ = best_effort_cursor_update(&ms, entry_position).await;
            });
        }

        Ok(Some(WalReadEntry {
            shard_id: self.shard_id,
            entry_position,
            writer_epoch,
            batches,
        }))
    }

    /// Find the next append position (one past the latest entry).
    pub async fn next_position(&self) -> Result<u64> {
        if let Some(hint) = self.manifest_cursor_hint().await
            && hint >= FIRST_WAL_ENTRY_POSITION
            && let Some(tip) = self.probe_forward(hint).await?
        {
            return Ok(tip);
        }
        scan_next_position(self.object_store.as_ref(), &self.wal_dir, self.shard_id).await
    }

    /// Find the earliest listed WAL position.
    pub async fn first_position(&self) -> Result<u64> {
        scan_first_position(self.object_store.as_ref(), &self.wal_dir, self.shard_id).await
    }

    async fn manifest_cursor_hint(&self) -> Option<u64> {
        let manifest = self.manifest_store.read_latest().await.ok()??;
        let hint = manifest.wal_entry_position_last_seen;
        if hint > 0 { Some(hint) } else { None }
    }

    async fn probe_forward(&self, hint: u64) -> Result<Option<u64>> {
        let path = self.wal_dir.child(wal_entry_filename(hint));
        match self.object_store.inner.head(&path).await {
            Ok(_) => {}
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(e) => {
                return Err(Error::io(format!(
                    "failed to check WAL entry {} for shard {}: {}",
                    hint, self.shard_id, e
                )));
            }
        }
        let mut pos = hint + 1;
        while pos - hint <= MAX_CURSOR_PROBE {
            let p = self.wal_dir.child(wal_entry_filename(pos));
            match self.object_store.inner.head(&p).await {
                Ok(_) => pos += 1,
                Err(object_store::Error::NotFound { .. }) => return Ok(Some(pos)),
                Err(e) => {
                    return Err(Error::io(format!(
                        "failed to check WAL entry {} for shard {}: {}",
                        pos, self.shard_id, e
                    )));
                }
            }
        }
        Ok(None)
    }
}

// --- helpers ---

fn validate_appender_batches(batches: &[RecordBatch]) -> Result<()> {
    if batches.is_empty() {
        return Err(Error::invalid_input(
            "cannot append an empty batch list to WAL",
        ));
    }
    let schema = batches[0].schema();
    for (idx, batch) in batches.iter().enumerate() {
        if batch.num_rows() == 0 {
            return Err(Error::invalid_input(format!(
                "cannot append empty batch at index {} to WAL",
                idx
            )));
        }
        if batch.schema_ref().fields() != schema.fields() {
            return Err(Error::invalid_input(format!(
                "batch at index {} has a different schema from the first batch",
                idx
            )));
        }
    }
    Ok(())
}

fn serialize_appender_batches(batches: &[RecordBatch], writer_epoch: u64) -> Result<Vec<u8>> {
    let schema = batches[0].schema();
    let mut metadata = schema.metadata().clone();
    metadata.insert(WRITER_EPOCH_KEY.to_string(), writer_epoch.to_string());
    let ipc_schema = Arc::new(ArrowSchema::new_with_metadata(
        schema.fields().to_vec(),
        metadata,
    ));
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &ipc_schema)
            .map_err(|e| Error::io(format!("failed to create Arrow IPC stream writer: {}", e)))?;
        for batch in batches {
            writer.write(batch).map_err(|e| {
                Error::io(format!("failed to write batch to WAL IPC stream: {}", e))
            })?;
        }
        writer
            .finish()
            .map_err(|e| Error::io(format!("failed to finish WAL IPC stream: {}", e)))?;
    }
    Ok(buffer)
}

fn deserialize_appender_batches(bytes: Bytes) -> Result<(u64, Vec<RecordBatch>)> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Error::io(format!("failed to open WAL IPC stream reader: {}", e)))?;
    let schema = reader.schema();
    let writer_epoch = schema
        .metadata()
        .get(WRITER_EPOCH_KEY)
        .ok_or_else(|| Error::io(format!("WAL entry missing {} metadata", WRITER_EPOCH_KEY)))?
        .parse::<u64>()
        .map_err(|e| {
            Error::io(format!(
                "WAL entry has malformed {} metadata: {}",
                WRITER_EPOCH_KEY, e
            ))
        })?;
    let mut clean_metadata = schema.metadata().clone();
    clean_metadata.remove(WRITER_EPOCH_KEY);
    let logical_schema = Arc::new(ArrowSchema::new_with_metadata(
        schema.fields().to_vec(),
        clean_metadata,
    ));
    let mut batches = Vec::new();
    for batch in reader {
        let batch =
            batch.map_err(|e| Error::io(format!("failed to read WAL IPC stream batch: {}", e)))?;
        let clean = RecordBatch::try_new(logical_schema.clone(), batch.columns().to_vec())
            .map_err(|e| Error::io(format!("failed to strip WAL metadata: {}", e)))?;
        batches.push(clean);
    }
    Ok((writer_epoch, batches))
}

enum AtomicPutError {
    AlreadyExists,
    Other(Error),
}

async fn atomic_put(
    object_store: &ObjectStore,
    dir: &Path,
    filename: &str,
    bytes: Bytes,
) -> std::result::Result<(), AtomicPutError> {
    let path = dir.child(filename);
    if object_store.is_local() {
        let temp = dir.child(format!("{}.tmp.{}", filename, Uuid::new_v4()));
        object_store
            .inner
            .put(&temp, bytes.into())
            .await
            .map_err(|e| {
                AtomicPutError::Other(Error::io(format!("failed to write temp file: {}", e)))
            })?;
        match object_store.inner.rename_if_not_exists(&temp, &path).await {
            Ok(()) => Ok(()),
            Err(object_store::Error::AlreadyExists { .. }) => {
                let _ = object_store.delete(&temp).await;
                Err(AtomicPutError::AlreadyExists)
            }
            Err(e) => {
                let _ = object_store.delete(&temp).await;
                Err(AtomicPutError::Other(Error::io(format!(
                    "failed to create {} atomically: {}",
                    path, e
                ))))
            }
        }
    } else {
        object_store
            .inner
            .put_opts(
                &path,
                bytes.into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| match e {
                object_store::Error::AlreadyExists { .. }
                | object_store::Error::Precondition { .. } => AtomicPutError::AlreadyExists,
                _ => AtomicPutError::Other(Error::io(format!(
                    "failed to create {} atomically: {}",
                    path, e
                ))),
            })?;
        Ok(())
    }
}

async fn scan_next_position(
    object_store: &ObjectStore,
    wal_dir: &Path,
    shard_id: Uuid,
) -> Result<u64> {
    let mut max_position = None::<u64>;
    let mut stream = object_store.inner.list(Some(wal_dir));
    while let Some(item) = stream.next().await {
        let meta = item.map_err(|e| {
            Error::io(format!(
                "failed to list WAL directory for shard {}: {}",
                shard_id, e
            ))
        })?;
        if let Some(filename) = meta.location.filename()
            && let Some(position) = parse_bit_reversed_filename(filename)
        {
            max_position = Some(max_position.map_or(position, |max| max.max(position)));
        }
    }
    match max_position {
        Some(pos) => pos
            .checked_add(1)
            .ok_or_else(|| Error::io(format!("WAL position overflow for shard {}", shard_id))),
        None => Ok(FIRST_WAL_ENTRY_POSITION),
    }
}

async fn scan_first_position(
    object_store: &ObjectStore,
    wal_dir: &Path,
    shard_id: Uuid,
) -> Result<u64> {
    let mut min_position = None::<u64>;
    let mut stream = object_store.inner.list(Some(wal_dir));
    while let Some(item) = stream.next().await {
        let meta = item.map_err(|e| {
            Error::io(format!(
                "failed to list WAL directory for shard {}: {}",
                shard_id, e
            ))
        })?;
        if let Some(filename) = meta.location.filename()
            && let Some(position) = parse_bit_reversed_filename(filename)
        {
            min_position = Some(min_position.map_or(position, |min| min.min(position)));
        }
    }
    Ok(min_position.unwrap_or(FIRST_WAL_ENTRY_POSITION))
}

async fn best_effort_cursor_update(manifest_store: &ShardManifestStore, entry_position: u64) {
    let Ok(Some(manifest)) = manifest_store.read_latest().await else {
        return;
    };
    if entry_position <= manifest.wal_entry_position_last_seen {
        return;
    }
    let mut updated = manifest;
    updated.version += 1;
    updated.wal_entry_position_last_seen = entry_position;
    let _ = manifest_store.write(&updated).await;
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
        let shard_id = Uuid::new_v4();
        let mut buffer = WalFlusher::new(&base_path, shard_id, 1, 1);
        buffer.set_object_store(store);

        // Track a batch
        let watcher = buffer.track_batch(0);

        // Watcher should not be durable yet
        assert!(!watcher.is_durable());
    }

    // Regression test: track_batch must return a watcher wired to the real
    // WAL watermark, NOT a pre-resolved watcher. A pre-resolved watcher would
    // cause durable writes to return before the WAL is actually flushed.
    #[tokio::test]
    async fn test_track_batch_watcher_blocks_until_flush() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let mut flusher = WalFlusher::new(&base_path, region_id, 1, 1);
        flusher.set_object_store(store);

        let schema = create_test_schema();
        let batch_store = BatchStore::with_capacity(10);
        batch_store.append(create_test_batch(&schema, 10)).unwrap();

        let mut watcher = flusher.track_batch(0);

        // wait() must NOT resolve before the flush happens
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(100), watcher.wait()).await;
        assert!(
            result.is_err(),
            "watcher resolved before WAL flush — durability guarantee broken"
        );

        // Flush, then the watcher should resolve
        flusher
            .flush_to_with_index_update(&batch_store, batch_store.len(), None)
            .await
            .unwrap();
        watcher.wait().await.unwrap();
        assert!(watcher.is_durable());
    }

    #[tokio::test]
    async fn test_wal_flusher_flush_to_with_index_update() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let mut buffer = WalFlusher::new(&base_path, shard_id, 1, 1);
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
        let shard_id = Uuid::new_v4();
        let mut buffer = WalFlusher::new(&base_path, shard_id, 42, 1);
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

    #[tokio::test]
    async fn test_wal_appender_and_tailer_round_trip() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();

        let appender = WalAppender::open(store.clone(), base_path.clone(), shard_id, 0)
            .await
            .unwrap();
        assert_eq!(appender.shard_id(), shard_id);
        assert_eq!(appender.writer_epoch(), 1);

        let schema = create_test_schema();
        let batch_a = create_test_batch(&schema, 4);
        let batch_b = create_test_batch(&schema, 2);

        let first = appender.append(vec![batch_a.clone()]).await.unwrap();
        assert_eq!(first.entry_position, FIRST_WAL_ENTRY_POSITION);
        assert_eq!(first.num_rows, 4);
        assert_eq!(first.num_batches, 1);
        assert!(first.wal_bytes > 0);

        let second = appender.append(vec![batch_b.clone()]).await.unwrap();
        assert_eq!(second.entry_position, FIRST_WAL_ENTRY_POSITION + 1);

        let tailer = WalTailer::new(store, base_path, shard_id);
        assert_eq!(tailer.first_position().await.unwrap(), first.entry_position);
        assert_eq!(
            tailer.next_position().await.unwrap(),
            second.entry_position + 1
        );

        let read = tailer
            .read_entry(first.entry_position)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(read.shard_id, shard_id);
        assert_eq!(read.writer_epoch, appender.writer_epoch());
        assert_eq!(read.batches.len(), 1);
        assert_eq!(read.batches[0].num_rows(), 4);
        // Writer-epoch IPC metadata must be stripped from logical batches.
        assert!(
            !read.batches[0]
                .schema()
                .metadata()
                .contains_key(WRITER_EPOCH_KEY)
        );

        assert!(tailer.read_entry(999).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_wal_appender_fenced_by_newer_writer() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();

        let first = WalAppender::open(store.clone(), base_path.clone(), shard_id, 0)
            .await
            .unwrap();
        assert_eq!(first.writer_epoch(), 1);

        let schema = create_test_schema();
        let batch = create_test_batch(&schema, 1);
        first.append(vec![batch.clone()]).await.unwrap();

        let second = WalAppender::open(store, base_path, shard_id, 0)
            .await
            .unwrap();
        assert_eq!(second.writer_epoch(), 2);
        // Newer writer races first to position 2.
        second.append(vec![batch.clone()]).await.unwrap();

        let err = first.check_fenced().await.unwrap_err();
        assert!(
            err.to_string().contains("Writer fenced"),
            "expected fence error, got: {err}"
        );

        // Fenced writer's cached next_pos still points at 2; the conflict path
        // must surface the fence error rather than silently advance.
        let err = first.append(vec![batch]).await.unwrap_err();
        assert!(
            err.to_string().contains("Writer fenced"),
            "expected fence error from append, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_wal_appender_rejects_invalid_input() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let appender = WalAppender::open(store, base_path, shard_id, 0)
            .await
            .unwrap();
        let err = appender.append(vec![]).await.unwrap_err();
        assert!(err.to_string().contains("empty batch list"));

        let schema = create_test_schema();
        let zero = RecordBatch::new_empty(schema);
        let err = appender.append(vec![zero]).await.unwrap_err();
        assert!(err.to_string().contains("empty batch"));
    }

    #[tokio::test]
    async fn test_wal_tailer_uses_manifest_cursor_hint() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let appender = WalAppender::open(store.clone(), base_path.clone(), shard_id, 0)
            .await
            .unwrap();

        let schema = create_test_schema();
        for _ in 0..3 {
            appender
                .append(vec![create_test_batch(&schema, 1)])
                .await
                .unwrap();
        }

        let tailer =
            WalTailer::new(store.clone(), base_path.clone(), shard_id).with_cursor_updates(true);
        let entry = tailer.read_entry(2).await.unwrap().unwrap();
        assert_eq!(entry.entry_position, 2);

        // Best-effort cursor update is async; poll briefly until it lands.
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);
        let mut hint = 0u64;
        for _ in 0..50 {
            if let Some(m) = manifest_store.read_latest().await.unwrap() {
                hint = m.wal_entry_position_last_seen;
                if hint >= 2 {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(hint >= 2, "cursor hint never updated, last={hint}");

        // next_position must still resolve to one past the last appended entry.
        assert_eq!(tailer.next_position().await.unwrap(), 4);
    }
}
