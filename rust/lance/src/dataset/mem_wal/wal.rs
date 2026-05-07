// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write-Ahead Log (WAL) flusher for durability.
//!
//! Batches are written as Arrow IPC streams with writer epoch metadata for fencing.
//! WAL files use bit-reversed naming to distribute files evenly across S3 keyspace.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
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
use object_store::ObjectStoreExt;
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

/// Source for a WAL flush — either a `BatchStore` range (MemTable mode) or
/// a drainable in-memory pending queue (WAL-only mode).
pub enum WalFlushSource {
    /// MemTable mode: read a `[max_flushed+1, end_batch_position)` range
    /// from a `BatchStore`. Indexes are updated in parallel with the WAL
    /// append.
    BatchStore {
        batch_store: Arc<BatchStore>,
        indexes: Option<Arc<IndexStore>>,
    },
    /// WAL-only mode: drain all pending batches from the shared
    /// `WalOnlyState`. There are no in-memory indexes to update.
    WalOnly { state: Arc<WalOnlyState> },
}

impl WalFlushSource {
    fn pending_count(&self) -> usize {
        match self {
            Self::BatchStore { batch_store, .. } => batch_store.pending_wal_flush_count(),
            Self::WalOnly { state } => state
                .pending
                .lock()
                .ok()
                .map(|p| p.batches.len())
                .unwrap_or(0),
        }
    }
}

/// Message to trigger a WAL flush.
///
/// Carries a `source` describing where to read batches from (BatchStore range
/// or pending queue) and an optional `done` cell for completion notification.
pub struct TriggerWalFlush {
    pub source: WalFlushSource,
    /// End batch position (exclusive). For `BatchStore`, flush batches after
    /// `max_flushed_batch_position` up to this. For `WalOnly`, indicates the
    /// position the durability watermark must reach for callers waiting on
    /// this flush. Use `usize::MAX` to flush all pending batches.
    pub end_batch_position: usize,
    /// Optional cell to write completion result.
    /// Uses Result<WalFlushResult, String> since Error doesn't implement Clone.
    pub done: Option<WatchableOnceCell<std::result::Result<WalFlushResult, String>>>,
}

impl std::fmt::Debug for TriggerWalFlush {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriggerWalFlush")
            .field("pending_batches", &self.source.pending_count())
            .field("end_batch_position", &self.end_batch_position)
            .finish()
    }
}

/// Drainable pending-batch queue used by `ShardWriter` in WAL-only mode.
///
/// Lives in `wal.rs` (not `write.rs`) so the WAL flush plumbing can name it
/// without a circular dependency.
#[derive(Debug, Default)]
pub struct WalOnlyState {
    pub pending: StdMutex<PendingWalBatches>,
}

#[derive(Debug, Default)]
pub struct PendingWalBatches {
    /// Batches accumulated since the last successful flush. Each entry
    /// pairs the `RecordBatch` with its memory footprint so we can keep
    /// `estimated_bytes` accurate when the front of the queue is committed
    /// after a successful append.
    batches: Vec<(RecordBatch, usize)>,
    /// Sum of memory footprints of batches currently in `batches`. Drives
    /// the size-based flush trigger.
    estimated_bytes: usize,
    /// Cumulative count of batches pushed since the writer was opened.
    /// Drives the monotonic batch positions returned in `WriteResult`.
    next_batch_position: usize,
}

/// Snapshot of the pending queue used by the flush handler to attempt a
/// WAL append without removing the batches first. On success the handler
/// calls `WalOnlyState::commit_flushed(snapshot.count)` to remove the front
/// of the queue; on failure the batches stay in the queue so a later flush
/// can retry them, instead of being silently lost.
#[derive(Debug)]
pub struct WalOnlySnapshot {
    pub batches: Vec<RecordBatch>,
    pub count: usize,
}

impl WalOnlyState {
    /// Push batches and return the assigned `[start, end)` position range.
    /// Holds the lock for the entire push so position assignment is atomic.
    pub fn push(&self, batches: Vec<RecordBatch>) -> std::ops::Range<usize> {
        let mut pending = self
            .pending
            .lock()
            .expect("WalOnlyState pending mutex poisoned");
        let start = pending.next_batch_position;
        let count = batches.len();
        for batch in batches.into_iter() {
            let bytes = batch.get_array_memory_size();
            pending.estimated_bytes += bytes;
            pending.batches.push((batch, bytes));
        }
        pending.next_batch_position += count;
        start..pending.next_batch_position
    }

    /// Snapshot the pending queue for an attempted WAL append. Clones the
    /// `RecordBatch` handles (cheap; `RecordBatch` is `Arc`-backed) without
    /// removing them. The caller must call `commit_flushed(snapshot.count)`
    /// on success; on failure, the batches remain in the queue for the next
    /// flush attempt.
    pub fn snapshot_pending(&self) -> WalOnlySnapshot {
        let pending = self
            .pending
            .lock()
            .expect("WalOnlyState pending mutex poisoned");
        let batches: Vec<RecordBatch> = pending.batches.iter().map(|(b, _)| b.clone()).collect();
        let count = batches.len();
        WalOnlySnapshot { batches, count }
    }

    /// Remove the first `count` batches from the pending queue and decrement
    /// `estimated_bytes` accordingly. Called by the flush handler after the
    /// WAL append for those batches succeeded.
    pub fn commit_flushed(&self, count: usize) {
        if count == 0 {
            return;
        }
        let mut pending = self
            .pending
            .lock()
            .expect("WalOnlyState pending mutex poisoned");
        let take = count.min(pending.batches.len());
        let bytes_removed: usize = pending.batches.drain(0..take).map(|(_, bytes)| bytes).sum();
        pending.estimated_bytes = pending.estimated_bytes.saturating_sub(bytes_removed);
    }

    /// Pending batch count (for stats / triggers).
    pub fn batch_count(&self) -> usize {
        self.pending
            .lock()
            .ok()
            .map(|p| p.batches.len())
            .unwrap_or(0)
    }

    /// Pending bytes (for size-based flush trigger).
    pub fn estimated_size(&self) -> usize {
        self.pending
            .lock()
            .ok()
            .map(|p| p.estimated_bytes)
            .unwrap_or(0)
    }

    /// Position the next pushed batch will get. Equivalent to "total pushed
    /// since open."
    pub fn next_batch_position(&self) -> usize {
        self.pending
            .lock()
            .ok()
            .map(|p| p.next_batch_position)
            .unwrap_or(0)
    }
}

/// Background WAL flush coordinator for `ShardWriter`.
///
/// `WalFlusher` owns the durability watermark watch channel, the trigger
/// channel for the background flush handler, and the wal-flush completion
/// cell used by backpressure waiters. It does **not** own the object store,
/// the writer epoch, or the next-position counter — those live on the
/// shared `WalAppender`. The flusher delegates the actual WAL write to the
/// appender, optionally running a parallel index update in MemTable mode.
pub struct WalFlusher {
    /// Watch channel sender for durable watermark.
    /// Broadcasts the highest batch_position that is now durable.
    durable_watermark_tx: watch::Sender<usize>,
    /// Watch channel receiver for creating new watchers.
    durable_watermark_rx: watch::Receiver<usize>,
    /// Underlying WAL append primitive — owns object store, epoch, and
    /// position discovery.
    wal_appender: Arc<WalAppender>,
    /// Shard ID (cached for tracing; same as `wal_appender.shard_id()`).
    shard_id: Uuid,
    /// Channel to send flush messages.
    flush_tx: Option<mpsc::UnboundedSender<TriggerWalFlush>>,
    /// Cell for WAL flush completion notification.
    /// Created at construction and recreated after each flush.
    /// Used by backpressure to wait for WAL flushes.
    wal_flush_cell: std::sync::Mutex<Option<WatchableOnceCell<super::write::DurabilityResult>>>,
}

impl WalFlusher {
    /// Create a new WAL flusher backed by an existing `WalAppender`.
    ///
    /// The appender owns object store, epoch, and position state. The
    /// flusher adds the durability watermark, trigger channel, and
    /// completion cell on top.
    pub fn new(wal_appender: Arc<WalAppender>) -> Self {
        let shard_id = wal_appender.shard_id();
        // Initialize durable watermark at 0 (no batches durable yet)
        let (durable_watermark_tx, durable_watermark_rx) = watch::channel(0);
        // Create initial WAL flush cell for backpressure
        let wal_flush_cell = WatchableOnceCell::new();
        Self {
            durable_watermark_tx,
            durable_watermark_rx,
            wal_appender,
            shard_id,
            flush_tx: None,
            wal_flush_cell: std::sync::Mutex::new(Some(wal_flush_cell)),
        }
    }

    /// Set the flush channel for background flush handler.
    pub fn set_flush_channel(&mut self, tx: mpsc::UnboundedSender<TriggerWalFlush>) {
        self.flush_tx = Some(tx);
    }

    /// Underlying WAL appender (for tests and debug accessors).
    pub fn wal_appender(&self) -> &Arc<WalAppender> {
        &self.wal_appender
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

    /// Trigger an immediate WAL flush.
    ///
    /// # Arguments
    ///
    /// * `source` - Where the flush should pull batches from (BatchStore range or pending queue).
    /// * `end_batch_position` - End batch position (exclusive); for `BatchStore`, the
    ///   range to flush; for `WalOnly`, the watermark callers are waiting for.
    /// * `done` - Optional cell to write completion result.
    pub fn trigger_flush(
        &self,
        source: WalFlushSource,
        end_batch_position: usize,
        done: Option<WatchableOnceCell<std::result::Result<WalFlushResult, String>>>,
    ) -> Result<()> {
        if let Some(tx) = &self.flush_tx {
            tx.send(TriggerWalFlush {
                source,
                end_batch_position,
                done,
            })
            .map_err(|_| Error::io("WAL flush channel closed"))?;
        }
        Ok(())
    }

    /// Flush from the given source up to `end_batch_position`, optionally
    /// updating in-memory indexes in parallel with the WAL append.
    ///
    /// Delegates the actual WAL write to the underlying `WalAppender`
    /// (atomic put-if-not-exists, conflict retry, fence-on-write).
    ///
    /// Returns an empty `WalFlushResult` if there is nothing to flush.
    #[instrument(name = "wal_flush", level = "info", skip_all, fields(shard_id = %self.shard_id, end_batch_position))]
    pub async fn flush(
        &self,
        source: &WalFlushSource,
        end_batch_position: usize,
    ) -> Result<WalFlushResult> {
        match source {
            WalFlushSource::BatchStore {
                batch_store,
                indexes,
            } => {
                self.flush_from_batch_store(batch_store, indexes.clone(), end_batch_position)
                    .await
            }
            WalFlushSource::WalOnly { state } => self.flush_from_wal_only(state).await,
        }
    }

    async fn flush_from_batch_store(
        &self,
        batch_store: &BatchStore,
        indexes: Option<Arc<IndexStore>>,
        end_batch_position: usize,
    ) -> Result<WalFlushResult> {
        // Get current flush position from per-memtable watermark (inclusive)
        // start_batch_position is the first batch to flush
        let start_batch_position = batch_store
            .max_flushed_batch_position()
            .map(|w| w + 1)
            .unwrap_or(0);

        // If we've already flushed past this end, nothing to do
        if start_batch_position >= end_batch_position {
            return Ok(empty_flush_result());
        }

        // Collect batches in range [start_batch_position, end_batch_position)
        let mut stored_batches: Vec<StoredBatch> =
            Vec::with_capacity(end_batch_position - start_batch_position);

        for batch_position in start_batch_position..end_batch_position {
            if let Some(stored) = batch_store.get(batch_position) {
                stored_batches.push(stored.clone());
            }
        }

        if stored_batches.is_empty() {
            return Ok(empty_flush_result());
        }

        let rows_to_index: usize = stored_batches.iter().map(|b| b.num_rows).sum();
        let record_batches: Vec<RecordBatch> =
            stored_batches.iter().map(|s| s.data.clone()).collect();

        let appender = self.wal_appender.clone();
        let (append_result, index_result) = if let Some(idx_registry) = indexes {
            let wal_future = async move {
                let start = Instant::now();
                let r = appender.append(record_batches).await?;
                Ok::<_, Error>((r, start.elapsed()))
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
            let wal_future = async move {
                let start = Instant::now();
                let r = appender.append(record_batches).await?;
                Ok::<_, Error>((r, start.elapsed()))
            };
            (
                wal_future.await,
                Ok((std::time::Duration::ZERO, std::collections::HashMap::new())),
            )
        };

        let (append_result, wal_io_duration) = append_result?;
        let (index_update_duration, index_update_duration_breakdown) = index_result?;

        // Update per-memtable watermark (inclusive: last batch ID that was flushed)
        batch_store.set_max_flushed_batch_position(end_batch_position - 1);

        // Notify durability waiters (global channel)
        let _ = self.durable_watermark_tx.send(end_batch_position);
        // Signal WAL flush completion for backpressure waiters
        self.signal_wal_flush_complete();

        Ok(WalFlushResult {
            entry: Some(WalEntry {
                position: append_result.entry_position,
                writer_epoch: self.wal_appender.writer_epoch(),
                num_batches: append_result.num_batches,
            }),
            wal_io_duration,
            index_update_duration,
            index_update_duration_breakdown,
            rows_indexed: rows_to_index,
            wal_bytes: append_result.wal_bytes,
        })
    }

    async fn flush_from_wal_only(&self, state: &Arc<WalOnlyState>) -> Result<WalFlushResult> {
        // Snapshot the pending queue (clone-only; cheap because RecordBatch
        // is Arc-backed). If the append fails the batches stay in the queue
        // for the next flush attempt. If it succeeds we truncate the front.
        let snapshot = state.snapshot_pending();
        if snapshot.count == 0 {
            return Ok(empty_flush_result());
        }

        let start = Instant::now();
        let append_result = self.wal_appender.append(snapshot.batches).await?;
        let wal_io_duration = start.elapsed();

        // Append succeeded — remove the flushed batches from the front of
        // the queue. Note: WAL-only mode does not use the global durability
        // watermark (`durable_watermark_tx`) — `put_wal_only` waits on the
        // per-trigger `done` cell instead — so we don't advance it here.
        // Same for the wal-flush-completion cell, which is only consulted
        // by MemTable-mode backpressure waiters.
        state.commit_flushed(snapshot.count);

        Ok(WalFlushResult {
            entry: Some(WalEntry {
                position: append_result.entry_position,
                writer_epoch: self.wal_appender.writer_epoch(),
                num_batches: append_result.num_batches,
            }),
            wal_io_duration,
            index_update_duration: std::time::Duration::ZERO,
            index_update_duration_breakdown: std::collections::HashMap::new(),
            rows_indexed: 0,
            wal_bytes: append_result.wal_bytes,
        })
    }

    /// Stats accessor: best-effort next-position hint, mirrored from the
    /// underlying appender.
    pub fn next_wal_entry_position(&self) -> u64 {
        self.wal_appender.next_entry_position_hint()
    }

    /// Get the shard ID.
    pub fn shard_id(&self) -> Uuid {
        self.shard_id
    }

    /// Get the writer epoch (delegated to the underlying appender).
    pub fn writer_epoch(&self) -> u64 {
        self.wal_appender.writer_epoch()
    }

    /// Get the path for a WAL entry.
    pub fn wal_entry_path(&self, wal_entry_position: u64) -> Path {
        self.wal_appender.wal_entry_path(wal_entry_position)
    }
}

/// Sentinel "nothing to flush" result, shared by `WalFlusher` and the
/// `WalFlushHandler` early-out path in `write.rs`.
pub fn empty_flush_result() -> WalFlushResult {
    WalFlushResult {
        entry: None,
        wal_io_duration: std::time::Duration::ZERO,
        index_update_duration: std::time::Duration::ZERO,
        index_update_duration_breakdown: std::collections::HashMap::new(),
        rows_indexed: 0,
        wal_bytes: 0,
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

const FIRST_WAL_ENTRY_POSITION: u64 = 0;
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
    /// Mirrors `next_entry_position` for cheap sync observability (stats).
    /// Seeded at construction from `manifest.wal_entry_position_last_seen + 1`
    /// so reopened shards report the post-recovery cursor immediately;
    /// updated after each successful append.
    next_entry_position_hint: AtomicU64,
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
        let (writer_epoch, manifest) = manifest_store.claim_epoch(shard_spec_id).await?;
        let position_hint = manifest
            .wal_entry_position_last_seen
            .max(manifest.replay_after_wal_entry_position)
            .saturating_add(1);
        Ok(Self::with_claimed_epoch(
            object_store,
            base_path,
            shard_id,
            manifest_store,
            writer_epoch,
            position_hint,
        ))
    }

    /// Create a WAL appender for a shard whose epoch was already claimed by
    /// the caller (e.g. `ShardWriter::open` claims once then injects the
    /// resulting epoch into both the shard writer and its appender).
    ///
    /// `next_entry_position_hint_seed` should be
    /// `manifest.wal_entry_position_last_seen + 1` so the sync observability
    /// accessor (`next_entry_position_hint` / `WalFlusher::next_wal_entry_position`)
    /// reflects the post-recovery cursor immediately after open instead of
    /// reading 0 until the first append has discovered the true tip. The
    /// authoritative position counter is still discovered lazily on the
    /// first append.
    pub(crate) fn with_claimed_epoch(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        shard_id: Uuid,
        manifest_store: Arc<ShardManifestStore>,
        writer_epoch: u64,
        next_entry_position_hint_seed: u64,
    ) -> Self {
        Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            manifest_store,
            shard_id,
            writer_epoch,
            next_entry_position: Mutex::new(None),
            next_entry_position_hint: AtomicU64::new(next_entry_position_hint_seed),
        }
    }

    /// Shard id.
    pub fn shard_id(&self) -> Uuid {
        self.shard_id
    }

    /// Writer epoch recorded in the shard manifest.
    pub fn writer_epoch(&self) -> u64 {
        self.writer_epoch
    }

    /// Resolved path for a WAL entry at `position`.
    pub(crate) fn wal_entry_path(&self, position: u64) -> Path {
        self.wal_dir.clone().join(wal_entry_filename(position))
    }

    /// Cheap sync accessor for the next entry position the appender will
    /// assign on its next `append`. Seeded from the shard manifest at
    /// construction (so reopened shards report the post-recovery cursor
    /// immediately) and updated after each successful append. Used for
    /// stats observability; not authoritative — the actual next position
    /// is discovered lazily by `append`.
    pub(crate) fn next_entry_position_hint(&self) -> u64 {
        self.next_entry_position_hint.load(Ordering::SeqCst)
    }

    /// Seed the appender's next-position counter from a known value
    /// (e.g. one past the highest WAL entry observed during MemTable
    /// replay on `ShardWriter::open`). Skips the first-append lazy
    /// discovery probe.
    pub(crate) async fn seed_next_position(&self, position: u64) {
        let mut guard = self.next_entry_position.lock().await;
        *guard = Some(position);
        self.next_entry_position_hint
            .store(position, Ordering::SeqCst);
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
            *next_pos = Some(self.discover_next_position().await?);
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
                    let next = pos.checked_add(1).ok_or_else(|| {
                        Error::io(format!("WAL position overflow for shard {}", self.shard_id))
                    })?;
                    *next_pos = Some(next);
                    self.next_entry_position_hint.store(next, Ordering::SeqCst);
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
                        *next_pos = Some(self.discover_next_position().await?);
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

    async fn discover_next_position(&self) -> Result<u64> {
        if let Ok(Some(manifest)) = self.manifest_store.read_latest().await {
            let hint = manifest.wal_entry_position_last_seen;
            if let Some(tip) = probe_forward_from(
                self.object_store.as_ref(),
                &self.wal_dir,
                self.shard_id,
                hint,
            )
            .await?
            {
                return Ok(tip);
            }
        }
        scan_next_position(self.object_store.as_ref(), &self.wal_dir, self.shard_id).await
    }
}

/// Ordered reader for MemWAL entries from a single shard.
///
/// Uses `wal_entry_position_last_seen` from the shard manifest as a cursor
/// hint for `next_position()`, probing forward from the hint to find the true
/// tip before falling back to a full directory listing.
///
/// Successful `read_entry` calls asynchronously update
/// `wal_entry_position_last_seen` in the shard manifest (fire-and-forget).
#[derive(Debug, Clone)]
pub struct WalTailer {
    object_store: Arc<ObjectStore>,
    wal_dir: Path,
    manifest_store: Arc<ShardManifestStore>,
    shard_id: Uuid,
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
        }
    }

    /// Read a WAL entry at the given position. Returns `None` if no entry exists.
    /// On success, asynchronously updates `wal_entry_position_last_seen` in the
    /// shard manifest as a best-effort cursor hint for future readers.
    pub async fn read_entry(&self, entry_position: u64) -> Result<Option<WalReadEntry>> {
        let path = self
            .wal_dir
            .clone()
            .join(wal_entry_filename(entry_position));
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

        let ms = self.manifest_store.clone();
        tokio::spawn(async move {
            let _ = best_effort_cursor_update(&ms, entry_position).await;
        });

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
        Some(manifest.wal_entry_position_last_seen)
    }

    async fn probe_forward(&self, hint: u64) -> Result<Option<u64>> {
        probe_forward_from(
            self.object_store.as_ref(),
            &self.wal_dir,
            self.shard_id,
            hint,
        )
        .await
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
    let path = dir.clone().join(filename);
    if object_store.is_local() {
        let temp = dir
            .clone()
            .join(format!("{}.tmp.{}", filename, Uuid::new_v4()));
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

/// Probe forward from a hint position to find the next unwritten position.
/// Returns `None` if the hint position itself doesn't exist (stale hint).
async fn probe_forward_from(
    object_store: &ObjectStore,
    wal_dir: &Path,
    shard_id: Uuid,
    hint: u64,
) -> Result<Option<u64>> {
    let path = wal_dir.clone().join(wal_entry_filename(hint));
    match object_store.inner.head(&path).await {
        Ok(_) => {}
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(e) => {
            return Err(Error::io(format!(
                "failed to check WAL entry {} for shard {}: {}",
                hint, shard_id, e
            )));
        }
    }
    let mut pos = hint + 1;
    while pos - hint <= MAX_CURSOR_PROBE {
        let p = wal_dir.clone().join(wal_entry_filename(pos));
        match object_store.inner.head(&p).await {
            Ok(_) => pos += 1,
            Err(object_store::Error::NotFound { .. }) => return Ok(Some(pos)),
            Err(e) => {
                return Err(Error::io(format!(
                    "failed to check WAL entry {} for shard {}: {}",
                    pos, shard_id, e
                )));
            }
        }
    }
    Ok(None)
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

    fn build_test_flusher(
        store: Arc<ObjectStore>,
        base_path: &Path,
        shard_id: Uuid,
        writer_epoch: u64,
    ) -> WalFlusher {
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            base_path,
            shard_id,
            2,
        ));
        let appender = Arc::new(WalAppender::with_claimed_epoch(
            store,
            base_path.clone(),
            shard_id,
            manifest_store,
            writer_epoch,
            // Tests start with no entries, so seed the hint at 0.
            0,
        ));
        WalFlusher::new(appender)
    }

    fn batch_store_source(batch_store: &Arc<BatchStore>) -> WalFlushSource {
        WalFlushSource::BatchStore {
            batch_store: batch_store.clone(),
            indexes: None,
        }
    }

    #[tokio::test]
    async fn test_wal_flusher_track_batch() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let buffer = build_test_flusher(store, &base_path, shard_id, 1);

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
        let flusher = build_test_flusher(store, &base_path, region_id, 1);

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(10));
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
        let source = batch_store_source(&batch_store);
        flusher.flush(&source, batch_store.len()).await.unwrap();
        watcher.wait().await.unwrap();
        assert!(watcher.is_durable());
    }

    #[tokio::test]
    async fn test_wal_flusher_flush_to_with_index_update() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let buffer = build_test_flusher(store, &base_path, shard_id, 1);

        // Create a BatchStore with some data
        let schema = create_test_schema();
        let batch1 = create_test_batch(&schema, 10);
        let batch2 = create_test_batch(&schema, 5);

        let batch_store = Arc::new(BatchStore::with_capacity(10));
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
        let source = batch_store_source(&batch_store);
        let result = buffer.flush(&source, batch_store.len()).await.unwrap();
        let entry = result.entry.unwrap();
        // First entry from a freshly-discovered position is 0 (atomic-create
        // path discovers the tip via list and starts at FIRST_WAL_ENTRY_POSITION).
        assert_eq!(entry.position, FIRST_WAL_ENTRY_POSITION);
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
        let buffer = build_test_flusher(store.clone(), &base_path, shard_id, 42);

        // Create a BatchStore with some data
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(10));
        batch_store.append(create_test_batch(&schema, 10)).unwrap();
        batch_store.append(create_test_batch(&schema, 5)).unwrap();

        // Track batch IDs and flush all pending batches
        let _watcher1 = buffer.track_batch(0);
        let _watcher2 = buffer.track_batch(1);
        let source = batch_store_source(&batch_store);
        let result = buffer.flush(&source, batch_store.len()).await.unwrap();
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

        let tailer = WalTailer::new(store.clone(), base_path.clone(), shard_id);
        let entry = tailer.read_entry(1).await.unwrap().unwrap();
        assert_eq!(entry.entry_position, 1);

        // Best-effort cursor update is async; poll briefly until it lands.
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);
        let mut hint = 0u64;
        for _ in 0..50 {
            if let Some(m) = manifest_store.read_latest().await.unwrap() {
                hint = m.wal_entry_position_last_seen;
                if hint >= 1 {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(hint >= 1, "cursor hint never updated, last={hint}");

        // next_position must still resolve to one past the last appended entry.
        assert_eq!(tailer.next_position().await.unwrap(), 3);
    }
}
