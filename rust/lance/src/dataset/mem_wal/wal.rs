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

use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::Schema as ArrowSchema;
use bytes::Bytes;
use futures::StreamExt;
use lance_core::{Error, FenceReason, Result};
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

/// Marks a WAL entry as a data-less fence sentinel (observability only;
/// replay skips sentinels via their empty batch list).
pub const FENCE_SENTINEL_KEY: &str = "fence_sentinel";

/// True if `error` is a peer fence (a successor claimed a higher epoch).
#[cfg(test)]
fn is_fence_error(error: &Error) -> bool {
    error.fence_reason() == Some(FenceReason::PeerClaimedEpoch)
}

/// True if `error` is terminal for the writer (either fence kind): the WAL will
/// never advance, so durability waiters must be woken and later writes rejected.
fn is_terminal_failure(error: &Error) -> bool {
    error.fence_reason().is_some()
}

/// Cloneable carrier that ferries a flush error across the async completion
/// channels (`WalFlusher::terminal_error` and the per-flush `done` cell), since
/// `lance_core::Error` is not `Clone`. Preserves the [`FenceReason`] so the typed
/// [`Error::Fenced`] can be rebuilt rather than flattened to a string.
#[derive(Clone, Debug)]
pub struct WalFlushFailure {
    /// The fence reason if terminal; `None` for an ordinary flush error.
    pub fence_reason: Option<FenceReason>,
    /// The error message carrying details about the flush failure.
    pub message: String,
}

impl WalFlushFailure {
    pub(crate) fn from_error(error: &Error) -> Self {
        Self {
            fence_reason: error.fence_reason(),
            message: error.to_string(),
        }
    }

    /// Rebuild a typed `Error`, restoring the fence reason when present.
    pub(crate) fn into_error(self) -> Error {
        match self.fence_reason {
            Some(FenceReason::PeerClaimedEpoch) => Error::fenced_by_peer(self.message),
            Some(FenceReason::PersistenceFailure) => Error::writer_poisoned(self.message),
            None => Error::io(self.message),
        }
    }
}

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
    /// Terminal flush failure shared with the flusher. When set, the watermark
    /// can never reach the target, so `wait` returns this typed error instead of
    /// blocking forever.
    terminal_error: Arc<StdMutex<Option<WalFlushFailure>>>,
}

impl BatchDurableWatcher {
    /// Create a new watcher for a specific batch ID.
    pub fn new(
        rx: watch::Receiver<usize>,
        target_batch_position: usize,
        terminal_error: Arc<StdMutex<Option<WalFlushFailure>>>,
    ) -> Self {
        Self {
            rx,
            target_batch_position,
            terminal_error,
        }
    }

    /// Wait until the batch is durable.
    ///
    /// Returns Ok(()) when `durable_watermark >= target_batch_position`, or
    /// Err if a terminal flush failure (e.g. a fence) means the watermark can
    /// never reach the target.
    pub async fn wait(&mut self) -> Result<()> {
        loop {
            if let Some(failure) = self.terminal_error.lock().unwrap().clone() {
                return Err(failure.into_error());
            }
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
    /// WAL entry position (1-based, sequential — see `FIRST_WAL_ENTRY_POSITION`).
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
    /// Uses `WalFlushFailure` (not `Error`) since `Error` doesn't implement
    /// `Clone`; the carrier preserves the fence reason so callers waiting on
    /// this cell still get a typed error.
    pub done: Option<WatchableOnceCell<std::result::Result<WalFlushResult, WalFlushFailure>>>,
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
    /// First terminal flush failure, shared with every `BatchDurableWatcher`. It
    /// wakes durability waiters (the watermark never advances) and is read by
    /// `check_poisoned` so the write path fails fast.
    terminal_error: Arc<StdMutex<Option<WalFlushFailure>>>,
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
            terminal_error: Arc::new(StdMutex::new(None)),
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
        BatchDurableWatcher::new(
            self.durable_watermark_rx.clone(),
            batch_position + 1,
            Arc::clone(&self.terminal_error),
        )
    }

    /// Latch a terminal flush failure and wake every durability waiter (the
    /// watermark never advances, so they must observe the error, not block).
    /// Idempotent: only the first failure is retained.
    fn mark_terminal_failure(&self, error: &Error) {
        {
            let mut slot = self.terminal_error.lock().unwrap();
            if slot.is_none() {
                *slot = Some(WalFlushFailure::from_error(error));
            }
        }
        // Wake `wait`ers without advancing the watermark; each re-checks
        // `terminal_error` and returns the error.
        self.durable_watermark_tx.send_modify(|_| {});
    }

    /// Fail fast with the typed error if this writer has been fenced (by a peer
    /// or its own persistence failure). The write path calls this before touching
    /// the memtable so a poisoned writer can't diverge further. Recovery is to
    /// reopen the shard (replay the WAL).
    pub fn check_poisoned(&self) -> Result<()> {
        if let Some(failure) = self.terminal_error.lock().unwrap().clone() {
            return Err(failure.into_error());
        }
        Ok(())
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
        done: Option<WatchableOnceCell<std::result::Result<WalFlushResult, WalFlushFailure>>>,
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
        let result = match source {
            WalFlushSource::BatchStore {
                batch_store,
                indexes,
            } => {
                self.flush_from_batch_store(batch_store, indexes.clone(), end_batch_position)
                    .await
            }
            WalFlushSource::WalOnly { state } => self.flush_from_wal_only(state).await,
        };
        // A terminal failure means the watermark can never advance; latch the
        // poison so waiters wake with the typed error and later writes fail fast.
        if let Err(e) = &result
            && is_terminal_failure(e)
        {
            self.mark_terminal_failure(e);
        }
        result
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

/// First valid WAL entry position. Positions are 1-based so that a
/// `ShardManifest::replay_after_wal_entry_position` of 0 unambiguously means
/// "no flush has ever stamped the cursor" — replay then starts at position 1
/// without needing to consult `flushed_generations`, which an external
/// compactor may legitimately drain back to empty.
const FIRST_WAL_ENTRY_POSITION: u64 = 1;
const MAX_APPEND_CREATE_CONFLICTS: usize = 1024;
const APPEND_CONFLICT_REFRESH_INTERVAL: usize = 16;
const MAX_CURSOR_PROBE: u64 = 4096;

/// Retry policy for transient WAL persistence failures before the writer
/// self-fences. On a non-conflict object-store error the appender retries the
/// *same* WAL position up to `max_retries` times with exponential backoff from
/// `base_delay`; exhausting the budget poisons the writer
/// ([`Error::writer_poisoned`]).
#[derive(Debug, Clone, Copy)]
pub struct WalRetryConfig {
    /// Maximum number of retries for a transient WAL write failure before the
    /// writer self-fences.
    pub max_retries: usize,
    /// Base duration for exponential backoff between retry attempts.
    pub base_delay: Duration,
}

impl Default for WalRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(50),
        }
    }
}

impl WalRetryConfig {
    /// Backoff before retry `attempt` (1-based), capped so a wedged store can't
    /// block the flush task indefinitely.
    fn backoff(&self, attempt: usize) -> Duration {
        const MAX_BACKOFF: Duration = Duration::from_secs(5);
        let shift = attempt.saturating_sub(1).min(16) as u32;
        self.base_delay
            .checked_mul(1u32 << shift)
            .unwrap_or(MAX_BACKOFF)
            .min(MAX_BACKOFF)
    }
}

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
    /// Retry budget for transient persistence failures before self-fencing.
    retry: WalRetryConfig,
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
            WalRetryConfig::default(),
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
        retry: WalRetryConfig,
    ) -> Self {
        Self {
            object_store,
            wal_dir: shard_wal_path(&base_path, &shard_id),
            manifest_store,
            shard_id,
            writer_epoch,
            next_entry_position: Mutex::new(None),
            next_entry_position_hint: AtomicU64::new(next_entry_position_hint_seed),
            retry,
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

        // `conflicts` counts position races across the append; `io_attempts` is
        // the per-position retry budget for transient PUT failures. The latter
        // only grows while we sit on one position — the sole advancing branch is
        // `io_attempts == 0`, so it never carries across positions.
        let mut conflicts = 0;
        let mut io_attempts = 0;
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
                    self.check_fenced().await?; // surfaces a peer takeover as a typed fence
                    // A slot we already failed to PUT is now occupied — ambiguous
                    // (our lost-ack, or a peer). We can't advance-and-rewrite (would
                    // duplicate) nor blindly accept it, so poison and let replay
                    // reconcile on reopen.
                    if io_attempts > 0 {
                        return Err(Error::writer_poisoned(format!(
                            "WAL position {} for shard {} was taken after a failed PUT; \
                             in-memory state may diverge from the durable WAL, reopen to replay",
                            pos, self.shard_id
                        )));
                    }
                    // First touch of this slot: an ordinary position conflict
                    // (stale cursor, our own earlier entries, or contention).
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
                    // A successor fence is terminal — don't waste the retry budget.
                    self.check_fenced().await?;
                    if io_attempts >= self.retry.max_retries {
                        return Err(Error::writer_poisoned(format!(
                            "WAL persistence failed for shard {} at position {} after {} retries; \
                             in-memory state may diverge from the durable WAL, reopen to replay: {}",
                            self.shard_id, pos, self.retry.max_retries, error
                        )));
                    }
                    io_attempts += 1;
                    tokio::time::sleep(self.retry.backoff(io_attempts)).await;
                    // Retry the same position (next_pos unchanged).
                }
            }
        }
    }

    /// Check that this writer's epoch has not been fenced.
    pub async fn check_fenced(&self) -> Result<()> {
        self.manifest_store.check_fenced(self.writer_epoch).await
    }

    /// Drop a data-less sentinel at the WAL tip so the predecessor's next
    /// `append` collides on PUT-IF-NOT-EXISTS and learns it is fenced, rather
    /// than succeeding into the empty next slot. Call *before* replay: any
    /// predecessor entry below the sentinel is then recovered, not orphaned.
    /// On a lost slot race, re-probes one past the winner. Seeds next position
    /// past the sentinel; returns the sentinel position.
    pub(crate) async fn write_fence_sentinel(&self) -> Result<u64> {
        let sentinel = Bytes::from(serialize_fence_sentinel(self.writer_epoch)?);
        let mut next_pos = self.next_entry_position.lock().await;
        let mut pos = match *next_pos {
            Some(p) => p,
            None => self.discover_next_position().await?,
        };
        let mut conflicts = 0;
        loop {
            match atomic_put(
                self.object_store.as_ref(),
                &self.wal_dir,
                &wal_entry_filename(pos),
                sentinel.clone(),
            )
            .await
            {
                Ok(()) => {
                    let next = pos.checked_add(1).ok_or_else(|| {
                        Error::io(format!("WAL position overflow for shard {}", self.shard_id))
                    })?;
                    *next_pos = Some(next);
                    self.next_entry_position_hint.store(next, Ordering::SeqCst);
                    return Ok(pos);
                }
                Err(AtomicPutError::AlreadyExists) => {
                    conflicts += 1;
                    if conflicts >= MAX_APPEND_CREATE_CONFLICTS {
                        return Err(Error::io(format!(
                            "fence sentinel write for shard {} failed after {} conflicts",
                            self.shard_id, conflicts
                        )));
                    }
                    pos = self.discover_next_position().await?;
                }
                Err(AtomicPutError::Other(error)) => return Err(error),
            }
        }
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

/// Data-less sentinel: an empty-schema Arrow IPC stream with the writer epoch
/// and a marker flag, no batches. Reads back as `(epoch, [])` so replay skips
/// it. See [`WalAppender::write_fence_sentinel`].
fn serialize_fence_sentinel(writer_epoch: u64) -> Result<Vec<u8>> {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(WRITER_EPOCH_KEY.to_string(), writer_epoch.to_string());
    metadata.insert(FENCE_SENTINEL_KEY.to_string(), "true".to_string());
    let ipc_schema = Arc::new(ArrowSchema::new_with_metadata(
        arrow_schema::Fields::empty(),
        metadata,
    ));
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &ipc_schema)
            .map_err(|e| Error::io(format!("failed to create fence sentinel IPC writer: {}", e)))?;
        writer
            .finish()
            .map_err(|e| Error::io(format!("failed to finish fence sentinel IPC stream: {}", e)))?;
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
    use crate::dataset::mem_wal::test_util::failing_memory_store;
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
            WalRetryConfig::default(),
        ));
        WalFlusher::new(appender)
    }

    fn batch_store_source(batch_store: &Arc<BatchStore>) -> WalFlushSource {
        WalFlushSource::BatchStore {
            batch_store: batch_store.clone(),
            indexes: None,
        }
    }

    fn batch_store_source_with_indexes(
        batch_store: &Arc<BatchStore>,
        indexes: &Arc<IndexStore>,
    ) -> WalFlushSource {
        WalFlushSource::BatchStore {
            batch_store: batch_store.clone(),
            indexes: Some(indexes.clone()),
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
        // First entry from a freshly-discovered position lands at
        // FIRST_WAL_ENTRY_POSITION (atomic-create path discovers the tip
        // via list).
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

    // Regression test for the visibility-cursor bug: with an empty IndexStore
    // (the common case for WAL-managed tables that mirror an index-less base
    // dataset), a WAL flush must still advance `max_visible_batch_position` so
    // scanners can see every batch up to the durable position — not just
    // batch 0. Before the fix, the cursor stayed at 0 for the lifetime of the
    // memtable and scanners returned only the first row.
    #[tokio::test]
    async fn test_wal_flush_advances_visibility_with_empty_indexes() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let flusher = build_test_flusher(store, &base_path, shard_id, 1);

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(10));
        for _ in 0..3 {
            batch_store.append(create_test_batch(&schema, 5)).unwrap();
        }

        // Empty registry, mimicking a memtable with `index_configs = []`.
        let indexes = Arc::new(IndexStore::new());
        assert_eq!(indexes.max_visible_batch_position(), 0);

        let source = batch_store_source_with_indexes(&batch_store, &indexes);
        flusher.flush(&source, batch_store.len()).await.unwrap();

        // Cursor must advance to the highest flushed batch position (2),
        // making all three batches visible to scanners.
        assert_eq!(indexes.max_visible_batch_position(), 2);
        assert_eq!(batch_store.max_flushed_batch_position(), Some(2));
    }

    // Regression guard for the indexed path: with at least one BTree index
    // configured, the cursor advance still fires (this was already working
    // before the fix — keeping the test to lock in the behavior).
    #[tokio::test]
    async fn test_wal_flush_advances_visibility_with_btree_index() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let flusher = build_test_flusher(store, &base_path, shard_id, 1);

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(10));
        for _ in 0..3 {
            batch_store.append(create_test_batch(&schema, 5)).unwrap();
        }

        let mut idx = IndexStore::new();
        idx.add_btree("id_idx".to_string(), 0, "id".to_string());
        let indexes = Arc::new(idx);

        let source = batch_store_source_with_indexes(&batch_store, &indexes);
        flusher.flush(&source, batch_store.len()).await.unwrap();

        assert_eq!(indexes.max_visible_batch_position(), 2);
        assert_eq!(batch_store.max_flushed_batch_position(), Some(2));
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
        // A peer takeover is a typed peer fence, distinct from a self-poison.
        assert_eq!(err.fence_reason(), Some(FenceReason::PeerClaimedEpoch));

        // Fenced writer's cached next_pos still points at 2; the conflict path
        // must surface the fence error rather than silently advance.
        let err = first.append(vec![batch]).await.unwrap_err();
        assert_eq!(err.fence_reason(), Some(FenceReason::PeerClaimedEpoch));
    }

    #[tokio::test]
    async fn test_fence_sentinel_fences_predecessor_without_successor_write() {
        // The race the sentinel closes: a successor claims a higher epoch but
        // has NOT yet written any data batch. Without the sentinel, the
        // predecessor's next append lands in the empty next slot, succeeds,
        // and false-acks. With the sentinel, the predecessor collides.
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();

        let first = WalAppender::open(store.clone(), base_path.clone(), shard_id, 0)
            .await
            .unwrap();
        let schema = create_test_schema();
        let batch = create_test_batch(&schema, 1);
        first.append(vec![batch.clone()]).await.unwrap(); // position 1

        // Successor claims epoch 2 and drops a sentinel at the tip (position 2)
        // — but writes no data of its own.
        let second = WalAppender::open(store.clone(), base_path.clone(), shard_id, 0)
            .await
            .unwrap();
        assert_eq!(second.writer_epoch(), 2);
        let sentinel_pos = second.write_fence_sentinel().await.unwrap();
        assert_eq!(sentinel_pos, 2, "sentinel should land at the tip");

        // Predecessor's next append collides with the sentinel and is fenced.
        let err = first.append(vec![batch.clone()]).await.unwrap_err();
        assert!(
            err.to_string().contains("Writer fenced"),
            "expected fence error from append, got: {err}"
        );

        // The sentinel is data-less: a tailer reads it back as zero batches so
        // replay skips it.
        let tailer = WalTailer::new(store.clone(), base_path.clone(), shard_id);
        let entry = tailer.read_entry(sentinel_pos).await.unwrap().unwrap();
        assert_eq!(entry.writer_epoch, 2);
        assert!(entry.batches.is_empty(), "sentinel must carry no batches");

        // Successor's own writes land after the sentinel (position 3).
        let res = second.append(vec![batch]).await.unwrap();
        assert_eq!(res.entry_position, 3);
    }

    // Regression: a fenced WAL flush never advances the durability watermark.
    // A `durable_write` put waits on a `BatchDurableWatcher`, so without
    // terminal-failure propagation the watcher blocks forever (the predecessor
    // pod's HTTP write hangs until the client times out). The flusher must
    // surface the fence through the watcher so the caller fails fast with 410.
    #[tokio::test]
    async fn test_durable_watcher_aborts_on_fence_instead_of_hanging() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let schema = create_test_schema();

        // Predecessor claims epoch 1 and writes one entry (position 1), seeding
        // its cached next position at 2. The flusher shares this appender.
        let first = Arc::new(
            WalAppender::open(store.clone(), base_path.clone(), shard_id, 0)
                .await
                .unwrap(),
        );
        assert_eq!(first.writer_epoch(), 1);
        first
            .append(vec![create_test_batch(&schema, 1)])
            .await
            .unwrap();
        let flusher = WalFlusher::new(Arc::clone(&first));

        // Successor claims epoch 2 and drops a sentinel at the predecessor's
        // next slot (position 2) — a rolling-restart pod replacement.
        let second = WalAppender::open(store.clone(), base_path.clone(), shard_id, 0)
            .await
            .unwrap();
        assert_eq!(second.writer_epoch(), 2);
        assert_eq!(second.write_fence_sentinel().await.unwrap(), 2);

        // A durable put on the predecessor: stage a batch and track it.
        let batch_store = Arc::new(BatchStore::with_capacity(10));
        batch_store.append(create_test_batch(&schema, 1)).unwrap();
        let mut watcher = flusher.track_batch(0);

        // Flushing collides with the sentinel and fences. Both the flush result
        // and the watcher must report the fence — and the watcher must resolve
        // promptly, not block on a watermark that can never advance.
        let source = batch_store_source(&batch_store);
        let flush_err = flusher.flush(&source, batch_store.len()).await.unwrap_err();
        assert!(
            is_fence_error(&flush_err),
            "expected fence error from flush, got: {flush_err}"
        );

        let waited = tokio::time::timeout(std::time::Duration::from_secs(5), watcher.wait()).await;
        let err = waited
            .expect("watcher.wait() hung after a fenced flush")
            .expect_err("watcher must surface the fence, not report success");
        assert!(
            is_fence_error(&err),
            "watcher must report the fence so the HTTP layer maps 410, got: {err}"
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
        // Three entries from a fresh shard land at 1, 2, 3, so next is 4.
        assert_eq!(tailer.next_position().await.unwrap(), 4);
    }

    // A transient PUT failure is retried at the same position and then succeeds.
    #[tokio::test]
    async fn test_append_retries_transient_failure_then_succeeds() {
        let (store, base, controls) = failing_memory_store().await;
        let shard_id = Uuid::new_v4();
        controls.fail_wal_puts(2); // 2 failures, under the default budget of 3
        let appender = WalAppender::open(store, base, shard_id, 0).await.unwrap();

        let schema = create_test_schema();
        let res = appender
            .append(vec![create_test_batch(&schema, 1)])
            .await
            .unwrap();
        assert_eq!(res.entry_position, FIRST_WAL_ENTRY_POSITION);
        assert_eq!(controls.attempts(), 3); // 2 failed + 1 succeeded
    }

    // Exhausting the retry budget poisons the writer with a typed persistence
    // failure (distinct from a peer fence).
    #[tokio::test]
    async fn test_append_poisons_after_exhausting_retries() {
        let (store, base, controls) = failing_memory_store().await;
        let shard_id = Uuid::new_v4();
        controls.fail_wal_puts(usize::MAX);
        let manifest_store = Arc::new(ShardManifestStore::new(store.clone(), &base, shard_id, 2));
        let (epoch, _) = manifest_store.claim_epoch(0).await.unwrap();
        let appender = WalAppender::with_claimed_epoch(
            store,
            base,
            shard_id,
            manifest_store,
            epoch,
            0,
            WalRetryConfig {
                max_retries: 2,
                base_delay: Duration::from_millis(1),
            },
        );

        let schema = create_test_schema();
        let err = appender
            .append(vec![create_test_batch(&schema, 1)])
            .await
            .unwrap_err();
        assert_eq!(err.fence_reason(), Some(FenceReason::PersistenceFailure));
        assert_eq!(controls.attempts(), 3); // io_attempts 0,1,2 then poison
    }

    // A lost acknowledgement (PUT errored but landed) poisons the writer without
    // ever writing a duplicate entry at the next position.
    #[tokio::test]
    async fn test_append_lost_ack_poisons_without_duplicate() {
        let (store, base, controls) = failing_memory_store().await;
        let shard_id = Uuid::new_v4();
        controls.set_lost_ack(true);
        controls.fail_wal_puts(1);
        let appender = WalAppender::open(store.clone(), base.clone(), shard_id, 0)
            .await
            .unwrap();

        let schema = create_test_schema();
        let err = appender
            .append(vec![create_test_batch(&schema, 1)])
            .await
            .unwrap_err();
        assert_eq!(err.fence_reason(), Some(FenceReason::PersistenceFailure));

        // The entry landed exactly once; the next slot stays empty.
        let tailer = WalTailer::new(store, base, shard_id);
        assert!(
            tailer
                .read_entry(FIRST_WAL_ENTRY_POSITION)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            tailer
                .read_entry(FIRST_WAL_ENTRY_POSITION + 1)
                .await
                .unwrap()
                .is_none()
        );
    }

    // A persistence failure during flush latches the poison: the flush result,
    // `check_poisoned`, and the durability watcher all report the typed error
    // (rather than the watcher hanging on a watermark that never advances).
    #[tokio::test]
    async fn test_flush_persistence_failure_poisons_and_wakes_waiter() {
        let (store, base, controls) = failing_memory_store().await;
        let shard_id = Uuid::new_v4();
        controls.fail_wal_puts(usize::MAX);
        let manifest_store = Arc::new(ShardManifestStore::new(store.clone(), &base, shard_id, 2));
        let (epoch, _) = manifest_store.claim_epoch(0).await.unwrap();
        let appender = Arc::new(WalAppender::with_claimed_epoch(
            store,
            base,
            shard_id,
            manifest_store,
            epoch,
            0,
            WalRetryConfig {
                max_retries: 1,
                base_delay: Duration::from_millis(1),
            },
        ));
        let flusher = WalFlusher::new(appender);

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(10));
        batch_store.append(create_test_batch(&schema, 1)).unwrap();
        let mut watcher = flusher.track_batch(0);

        let source = batch_store_source(&batch_store);
        let flush_err = flusher.flush(&source, batch_store.len()).await.unwrap_err();
        assert_eq!(
            flush_err.fence_reason(),
            Some(FenceReason::PersistenceFailure)
        );

        assert_eq!(
            flusher.check_poisoned().unwrap_err().fence_reason(),
            Some(FenceReason::PersistenceFailure)
        );

        let waited = tokio::time::timeout(Duration::from_secs(5), watcher.wait())
            .await
            .expect("watcher hung after a poisoning flush")
            .expect_err("watcher must surface the poison");
        assert_eq!(waited.fence_reason(), Some(FenceReason::PersistenceFailure));
    }
}
