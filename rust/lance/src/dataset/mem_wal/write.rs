// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stderr)]

//! Write path for MemWAL.
//!
//! This module contains all components for the write path:
//! - [`ShardWriter`] - Main writer interface for a single shard
//! - [`MemTable`] - In-memory table storing Arrow RecordBatches
//! - [`WalFlusher`] - Write-ahead log buffer for durability (Arrow IPC format)
//! - [`IndexStore`] - In-memory index management
//! - [`MemTableFlusher`] - Flush MemTable to storage as single Lance file

use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_index::mem_wal::ShardManifest;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_io::object_store::ObjectStore;
use log::{debug, error, info, warn};
use object_store::path::Path;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{Interval, interval_at};
use tokio_util::sync::CancellationToken;
use tracing::instrument;
use uuid::Uuid;

pub use super::index::{
    BTreeIndexConfig, BTreeMemIndex, FtsIndexConfig, HnswIndexConfig, IndexStore, MemIndexConfig,
};
pub use super::memtable::CacheConfig;
pub use super::memtable::MemTable;
pub use super::memtable::batch_store::{BatchStore, StoreFull, StoredBatch};
pub use super::memtable::flush::MemTableFlusher;
pub use super::memtable::scanner::MemTableScanner;
pub use super::util::{WatchableOnceCell, WatchableOnceCellReader};
pub use super::wal::{WalEntry, WalEntryData, WalFlushResult, WalFlusher};

use super::memtable::flush::TriggerMemTableFlush;
use super::scanner::GenerationWarmer;
use super::wal::{
    BatchDurableWatcher, TriggerWalFlush, WalAppender, WalFlushSource, WalOnlyState, WalTailer,
    empty_flush_result,
};
use super::{TOMBSTONE, schema_with_tombstone};

use super::manifest::ShardManifestStore;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a shard writer.
#[derive(Debug, Clone)]
pub struct ShardWriterConfig {
    /// Unique identifier for this shard (UUID v4).
    pub shard_id: Uuid,

    /// Shard spec ID this shard was created with.
    /// A value of 0 indicates a manually-created shard not governed by any spec.
    pub shard_spec_id: u32,

    /// Whether to wait for WAL flush before returning from writes.
    ///
    /// When true (durable writes):
    /// - Each write waits for WAL persistence before returning
    /// - Guarantees no data loss on crash
    /// - Higher latency due to object storage writes
    ///
    /// When false (non-durable writes):
    /// - Writes return immediately after buffering in memory
    /// - Potential data loss if process crashes before flush
    /// - Lower latency, batched S3 operations
    pub durable_write: bool,

    /// Whether to update indexes synchronously on each write.
    ///
    /// When true:
    /// - Newly written data is immediately searchable via indexes
    /// - Higher latency due to index update overhead
    ///
    /// When false:
    /// - Index updates are deferred
    /// - New data may not appear in index-accelerated queries immediately
    pub sync_indexed_write: bool,

    /// Maximum WAL buffer size in bytes before triggering a flush.
    ///
    /// This is a soft threshold - write batches are atomic and won't be split.
    /// WAL flushes when buffer exceeds this size OR when `max_wal_flush_interval` elapses.
    /// Default: 10MB
    pub max_wal_buffer_size: usize,

    /// Time-based WAL flush interval.
    ///
    /// WAL buffer will be flushed after this duration even if size threshold
    /// hasn't been reached. This ensures bounded data loss window in non-durable mode
    /// and prevents accumulating too much data before flushing to object storage.
    /// Default: 100ms
    pub max_wal_flush_interval: Option<Duration>,

    /// Maximum MemTable size in bytes before triggering a flush to storage.
    ///
    /// MemTable size is checked every `max_wal_flush_interval` (during WAL flush ticks).
    /// Default: 256MB
    pub max_memtable_size: usize,

    /// Maximum number of rows in a MemTable.
    ///
    /// Used to pre-allocate the in-memory HNSW graph and vector storage
    /// capacity. When the memtable reaches capacity, it will be flushed.
    /// Default: 100,000 rows
    pub max_memtable_rows: usize,

    /// Maximum number of batches in a MemTable.
    ///
    /// Used to pre-allocate batch storage. When this limit is reached,
    /// memtable will be flushed. Sized for typical ML workloads with
    /// 1024-dim vectors (~82KB per 20-row batch).
    /// Default: 8,000 batches
    pub max_memtable_batches: usize,

    /// Batch size for parallel HEAD requests when scanning for manifest versions.
    ///
    /// Higher values scan faster but use more parallel requests.
    /// Default: 2
    pub manifest_scan_batch_size: usize,

    /// Maximum unflushed bytes before applying backpressure.
    ///
    /// When total unflushed data (active memtable + frozen memtables) exceeds this,
    /// new writes will block until some data is flushed to storage.
    /// This prevents unbounded memory growth during write spikes.
    ///
    /// Default: 1GB
    pub max_unflushed_memtable_bytes: usize,

    /// Interval for logging warnings when writes are blocked by backpressure.
    ///
    /// When a write is blocked waiting for WAL flush, memtable flush, or index
    /// updates to complete, a warning is logged after this duration. The write
    /// will continue waiting indefinitely (it never fails due to backpressure),
    /// but warnings are logged at this interval to help diagnose slow flushes.
    ///
    /// Default: 30 seconds
    pub backpressure_log_interval: Duration,

    /// Maximum rows to buffer before flushing to async indexes.
    ///
    /// Only applies when `sync_indexed_write` is false. Larger values enable
    /// better vectorization but increase memory usage and latency before data
    /// becomes searchable.
    ///
    /// Default: 10,000 rows
    pub async_index_buffer_rows: usize,

    /// Maximum time to buffer before flushing to async indexes.
    ///
    /// Only applies when `sync_indexed_write` is false. Ensures bounded latency
    /// for data to become searchable even during low write throughput.
    ///
    /// Default: 1 second
    pub async_index_interval: Duration,

    /// Interval for periodic stats logging.
    ///
    /// Stats (write throughput, backpressure events, memtable size) are logged
    /// at this interval. Set to None to disable periodic stats logging.
    ///
    /// Default: 60 seconds
    pub stats_log_interval: Option<Duration>,

    /// How long a frozen memtable lingers in memory after its flush commits,
    /// before it is evicted and served only from the on-disk flushed dataset.
    ///
    /// `Duration::ZERO` (the default) disables retention: evict on commit, no
    /// sweep ticker. Correct for single-shot queries, which can't observe a
    /// generation evicted mid-read.
    ///
    /// A non-zero value is required only for queries split across reads (e.g.
    /// fresh tier and base table read separately, then deduped): the flushed
    /// dataset loses the per-batch boundaries that bound as-of membership
    /// (see [`crate::dataset::mem_wal::scanner::FreshTierWatermark`]), so a
    /// generation evicted between a query's reads can serve a stale row. Set it
    /// above the worst-case multi-part query latency, with margin.
    pub frozen_memtable_grace: Duration,

    /// Whether to maintain an in-memory MemTable on top of the WAL.
    ///
    /// When `true` (default), the writer maintains an in-memory `MemTable`,
    /// optionally with indexes, and asynchronously flushes frozen MemTables
    /// to Lance files alongside the WAL.
    ///
    /// When `false`, the writer skips the MemTable layer entirely:
    /// - No MemTable / BatchStore / IndexStore is allocated.
    /// - `index_configs` must be empty (validated at open time).
    /// - No MemTable freezing or Lance file flushing happens.
    /// - `max_unflushed_memtable_bytes` is reused as the backpressure
    ///   budget for the WAL-only pending-batch queue: `put` blocks while
    ///   the queue's estimated bytes meet or exceed this threshold.
    /// - The async batched WAL pipeline still runs, driven by the same
    ///   `max_wal_buffer_size`, `max_wal_flush_interval`, and
    ///   `durable_write` settings as MemTable mode.
    ///
    /// MemTable-tied tunables (`max_memtable_size`, `max_memtable_rows`,
    /// `max_memtable_batches`, `sync_indexed_write`, `async_index_buffer_rows`,
    /// `async_index_interval`) are ignored when `enable_memtable == false`.
    ///
    /// For raw single-entry synchronous atomic appends with no buffering and
    /// no background tasks, use `WalAppender` directly — it is a strictly
    /// lower-level primitive.
    pub enable_memtable: bool,

    /// Per-index HNSW build-parameter overrides for maintained vector indexes,
    /// keyed by index name.
    ///
    /// These control the in-memory HNSW graph this writer builds for its
    /// MemTable (and, on flush, the on-disk graph serialized from it). They are
    /// a property of the writer that builds the MemTable, not of the index
    /// definition: each flushed generation is independent, so different writers
    /// may use different parameters. An index without an entry uses the default
    /// build parameters. `num_edges` is the HNSW graph degree (level 0 retains
    /// `2 * num_edges`), equivalent to FAISS's `M`.
    ///
    /// Default: empty.
    pub hnsw_params: HashMap<String, HnswBuildParams>,

    /// Optional warmer fired pre-commit for each new generation (zero cold reads
    /// on first query). Wired to the flusher; supplied by the consumer (e.g. the
    /// WAL pod). Default: `None`.
    pub warmer: Option<Arc<dyn GenerationWarmer>>,
}

impl Default for ShardWriterConfig {
    fn default() -> Self {
        Self {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            sync_indexed_write: true,
            max_wal_buffer_size: 10 * 1024 * 1024, // 10MB
            max_wal_flush_interval: Some(Duration::from_millis(100)), // 100ms
            max_memtable_size: 256 * 1024 * 1024,  // 256MB
            max_memtable_rows: 100_000,            // 100k rows
            max_memtable_batches: 8_000,           // 8k batches
            manifest_scan_batch_size: 2,
            max_unflushed_memtable_bytes: 1024 * 1024 * 1024, // 1GB
            backpressure_log_interval: Duration::from_secs(30),
            async_index_buffer_rows: 10_000,
            async_index_interval: Duration::from_secs(1),
            stats_log_interval: Some(Duration::from_secs(60)), // 1 minute
            frozen_memtable_grace: Duration::ZERO,
            enable_memtable: true,
            hnsw_params: HashMap::new(),
            warmer: None,
        }
    }
}

impl ShardWriterConfig {
    /// Create a new configuration with the given shard ID.
    pub fn new(shard_id: Uuid) -> Self {
        Self {
            shard_id,
            ..Default::default()
        }
    }

    /// Set the shard spec ID.
    pub fn with_shard_spec_id(mut self, spec_id: u32) -> Self {
        self.shard_spec_id = spec_id;
        self
    }

    /// Set durable writes mode.
    pub fn with_durable_write(mut self, durable: bool) -> Self {
        self.durable_write = durable;
        self
    }

    /// Set indexed writes mode.
    pub fn with_sync_indexed_write(mut self, indexed: bool) -> Self {
        self.sync_indexed_write = indexed;
        self
    }

    /// Set maximum WAL buffer size.
    pub fn with_max_wal_buffer_size(mut self, size: usize) -> Self {
        self.max_wal_buffer_size = size;
        self
    }

    /// Set maximum flush interval.
    pub fn with_max_wal_flush_interval(mut self, interval: Duration) -> Self {
        self.max_wal_flush_interval = Some(interval);
        self
    }

    /// Set maximum MemTable size.
    pub fn with_max_memtable_size(mut self, size: usize) -> Self {
        self.max_memtable_size = size;
        self
    }

    /// Set maximum MemTable rows for index pre-allocation.
    pub fn with_max_memtable_rows(mut self, rows: usize) -> Self {
        self.max_memtable_rows = rows;
        self
    }

    /// Set maximum MemTable batches for batch store pre-allocation.
    pub fn with_max_memtable_batches(mut self, batches: usize) -> Self {
        self.max_memtable_batches = batches;
        self
    }

    /// Set manifest scan batch size.
    pub fn with_manifest_scan_batch_size(mut self, size: usize) -> Self {
        self.manifest_scan_batch_size = size;
        self
    }

    /// Set maximum unflushed bytes for backpressure.
    pub fn with_max_unflushed_memtable_bytes(mut self, size: usize) -> Self {
        self.max_unflushed_memtable_bytes = size;
        self
    }

    /// Set backpressure log interval.
    pub fn with_backpressure_log_interval(mut self, interval: Duration) -> Self {
        self.backpressure_log_interval = interval;
        self
    }

    /// Set async index buffer rows.
    pub fn with_async_index_buffer_rows(mut self, rows: usize) -> Self {
        self.async_index_buffer_rows = rows;
        self
    }

    /// Set async index interval.
    pub fn with_async_index_interval(mut self, interval: Duration) -> Self {
        self.async_index_interval = interval;
        self
    }

    /// Set stats logging interval. Use None to disable periodic stats logging.
    pub fn with_stats_log_interval(mut self, interval: Option<Duration>) -> Self {
        self.stats_log_interval = interval;
        self
    }

    /// Set how long a flushed memtable lingers in memory before eviction. MUST
    /// exceed the maximum query elapsed time — see `frozen_memtable_grace`.
    pub fn with_frozen_memtable_grace(mut self, grace: Duration) -> Self {
        self.frozen_memtable_grace = grace;
        self
    }

    /// Toggle the in-memory MemTable layer. See `enable_memtable` for the
    /// full WAL-only-mode contract. Defaults to `true`.
    pub fn with_enable_memtable(mut self, enable: bool) -> Self {
        self.enable_memtable = enable;
        self
    }

    /// Override the HNSW build parameters for a maintained vector index.
    ///
    /// Applies to the in-memory HNSW graph this writer builds for `index_name`
    /// (and the on-disk graph serialized from it on flush). Calling this
    /// repeatedly for the same index replaces the previous value.
    pub fn with_hnsw_params(
        mut self,
        index_name: impl Into<String>,
        params: HnswBuildParams,
    ) -> Self {
        self.hnsw_params.insert(index_name.into(), params);
        self
    }
}

// ============================================================================
// Background Task Infrastructure
// ============================================================================

/// Factory function for creating ticker messages.
type MessageFactory<T> = Box<dyn Fn() -> T + Send + Sync>;

/// Handler trait for processing messages in a background task.
#[async_trait]
pub trait MessageHandler<T: Send + Debug + 'static>: Send {
    /// Define periodic tickers that generate messages.
    fn tickers(&mut self) -> Vec<(Duration, MessageFactory<T>)> {
        vec![]
    }

    /// Handle a single message.
    async fn handle(&mut self, message: T) -> Result<()>;

    /// Cleanup on shutdown.
    async fn cleanup(&mut self, _shutdown_ok: bool) -> Result<()> {
        Ok(())
    }
}

/// Dispatcher that runs the event loop for a single message handler.
struct TaskDispatcher<T: Send + Debug> {
    handler: Box<dyn MessageHandler<T>>,
    rx: mpsc::UnboundedReceiver<T>,
    cancellation_token: CancellationToken,
    name: String,
}

impl<T: Send + Debug + 'static> TaskDispatcher<T> {
    async fn run(mut self) -> Result<()> {
        let tickers = self.handler.tickers();
        let mut ticker_intervals: Vec<(Interval, MessageFactory<T>)> = tickers
            .into_iter()
            .map(|(duration, factory)| {
                let interval = interval_at(tokio::time::Instant::now() + duration, duration);
                (interval, factory)
            })
            .collect();

        // A single failing message must not bring down the dispatcher:
        // the WAL flusher and MemTable flusher both run on this loop,
        // and dropping their channels deadlocks all subsequent puts
        // (and panics any task waiting on the corresponding watch).
        // Log and keep draining; the worst case for a transient flush
        // failure is replay from the WAL on next open.
        let result = loop {
            if ticker_intervals.is_empty() {
                tokio::select! {
                    biased;
                    _ = self.cancellation_token.cancelled() => {
                        debug!("Task '{}' received cancellation", self.name);
                        break Ok(());
                    }
                    msg = self.rx.recv() => {
                        match msg {
                            Some(message) => {
                                if let Err(e) = self.handler.handle(message).await {
                                    error!("Task '{}' error handling message: {}", self.name, e);
                                }
                            }
                            None => {
                                debug!("Task '{}' channel closed", self.name);
                                break Ok(());
                            }
                        }
                    }
                }
            } else {
                let first_ticker = ticker_intervals.first_mut().unwrap();
                let first_interval = &mut first_ticker.0;

                tokio::select! {
                    biased;
                    _ = self.cancellation_token.cancelled() => {
                        debug!("Task '{}' received cancellation", self.name);
                        break Ok(());
                    }
                    _ = first_interval.tick() => {
                        let message = (ticker_intervals[0].1)();
                        if let Err(e) = self.handler.handle(message).await {
                            error!("Task '{}' error handling ticker message: {}", self.name, e);
                        }
                    }
                    msg = self.rx.recv() => {
                        match msg {
                            Some(message) => {
                                if let Err(e) = self.handler.handle(message).await {
                                    error!("Task '{}' error handling message: {}", self.name, e);
                                }
                            }
                            None => {
                                debug!("Task '{}' channel closed", self.name);
                                break Ok(());
                            }
                        }
                    }
                }
            }
        };

        let cleanup_ok = result.is_ok();
        self.handler.cleanup(cleanup_ok).await?;

        info!("Task dispatcher '{}' stopped", self.name);
        result
    }
}

/// Executor that manages multiple background tasks.
pub struct TaskExecutor {
    tasks: StdRwLock<Vec<(String, JoinHandle<Result<()>>)>>,
    cancellation_token: CancellationToken,
}

impl TaskExecutor {
    pub fn new() -> Self {
        Self {
            tasks: StdRwLock::new(Vec::new()),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub fn add_handler<T: Send + Debug + 'static>(
        &self,
        name: String,
        handler: Box<dyn MessageHandler<T>>,
        rx: mpsc::UnboundedReceiver<T>,
    ) -> Result<()> {
        let dispatcher = TaskDispatcher {
            handler,
            rx,
            cancellation_token: self.cancellation_token.clone(),
            name: name.clone(),
        };

        let handle = tokio::spawn(async move { dispatcher.run().await });
        self.tasks.write().unwrap().push((name, handle));
        Ok(())
    }

    pub async fn shutdown_all(&self) -> Result<()> {
        info!("Shutting down all tasks");
        self.cancellation_token.cancel();

        let tasks = std::mem::take(&mut *self.tasks.write().unwrap());
        for (name, handle) in tasks {
            match handle.await {
                Ok(Ok(())) => debug!("Task '{}' completed successfully", name),
                Ok(Err(e)) => warn!("Task '{}' completed with error: {}", name, e),
                Err(e) => error!("Task '{}' panicked: {}", name, e),
            }
        }

        Ok(())
    }
}

impl Default for TaskExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Durability and Backpressure Types
// ============================================================================

/// Result of a durability notification.
///
/// This is a simple enum that can be cloned, unlike `Result<(), Error>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DurabilityResult {
    /// Write is now durable.
    Durable,
    /// Write failed with an error message.
    Failed(String),
}

impl DurabilityResult {
    /// Create a successful durability result.
    pub fn ok() -> Self {
        Self::Durable
    }

    /// Create a failed durability result.
    pub fn err(msg: impl Into<String>) -> Self {
        Self::Failed(msg.into())
    }

    /// Check if the result is durable.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Durable)
    }

    /// Convert to a Result.
    pub fn into_result(self) -> Result<()> {
        match self {
            Self::Durable => Ok(()),
            Self::Failed(msg) => Err(Error::io(msg)),
        }
    }
}

/// Type alias for durability watchers.
pub type DurabilityWatcher = WatchableOnceCellReader<DurabilityResult>;

/// Type alias for durability cells.
pub type DurabilityCell = WatchableOnceCell<DurabilityResult>;

/// Statistics for backpressure monitoring.
#[derive(Debug, Default)]
pub struct BackpressureStats {
    /// Total number of times backpressure was applied.
    total_count: AtomicU64,
    /// Total time spent waiting on backpressure (in milliseconds).
    total_wait_ms: AtomicU64,
}

impl BackpressureStats {
    /// Create new backpressure stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a backpressure event.
    pub fn record(&self, wait_ms: u64) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_wait_ms.fetch_add(wait_ms, Ordering::Relaxed);
    }

    /// Get the total backpressure count.
    pub fn count(&self) -> u64 {
        self.total_count.load(Ordering::Relaxed)
    }

    /// Get the total time spent waiting on backpressure.
    pub fn total_wait_ms(&self) -> u64 {
        self.total_wait_ms.load(Ordering::Relaxed)
    }

    /// Get a snapshot of all stats.
    pub fn snapshot(&self) -> BackpressureStatsSnapshot {
        BackpressureStatsSnapshot {
            total_count: self.total_count.load(Ordering::Relaxed),
            total_wait_ms: self.total_wait_ms.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of backpressure statistics.
#[derive(Debug, Clone, Default)]
pub struct BackpressureStatsSnapshot {
    /// Total number of times backpressure was applied.
    pub total_count: u64,
    /// Total time spent waiting on backpressure (in milliseconds).
    pub total_wait_ms: u64,
}

/// Backpressure controller for managing write flow.
pub struct BackpressureController {
    /// Configuration.
    config: ShardWriterConfig,
    /// Stats for monitoring.
    stats: Arc<BackpressureStats>,
}

impl BackpressureController {
    /// Create a new backpressure controller.
    pub fn new(config: ShardWriterConfig) -> Self {
        Self {
            config,
            stats: Arc::new(BackpressureStats::new()),
        }
    }

    /// Get backpressure stats.
    pub fn stats(&self) -> &Arc<BackpressureStats> {
        &self.stats
    }

    /// Check and apply backpressure if needed.
    ///
    /// This method blocks if the system is under memory pressure, waiting for
    /// frozen memtables to be flushed to storage until under threshold.
    ///
    /// Backpressure is applied when:
    /// - `unflushed_memtable_bytes` >= `max_unflushed_memtable_bytes`
    ///
    /// # Arguments
    /// - `get_state`: Closure that returns current (unflushed_memtable_bytes, oldest_memtable_watcher)
    ///
    /// The closure is called in a loop to get fresh state after each wait.
    pub async fn maybe_apply_backpressure<F>(&self, mut get_state: F) -> Result<()>
    where
        F: FnMut() -> (usize, Option<DurabilityWatcher>),
    {
        let start = std::time::Instant::now();
        let mut iteration = 0u32;

        loop {
            let (unflushed_memtable_bytes, oldest_watcher) = get_state();

            // Check if under threshold
            if unflushed_memtable_bytes < self.config.max_unflushed_memtable_bytes {
                if iteration > 0 {
                    let wait_ms = start.elapsed().as_millis() as u64;
                    self.stats.record(wait_ms);
                }
                return Ok(());
            }

            iteration += 1;

            debug!(
                "Backpressure triggered: unflushed_bytes={}, max={}, iteration={}",
                unflushed_memtable_bytes, self.config.max_unflushed_memtable_bytes, iteration
            );

            // Wait for oldest memtable to flush
            if let Some(mut mem_watcher) = oldest_watcher {
                tokio::select! {
                    _ = mem_watcher.await_value() => {}
                    _ = tokio::time::sleep(self.config.backpressure_log_interval) => {
                        warn!(
                            "Backpressure wait timeout, continuing to wait: unflushed_bytes={}, interval={}s, iteration={}",
                            unflushed_memtable_bytes,
                            self.config.backpressure_log_interval.as_secs(),
                            iteration
                        );
                    }
                }
            } else {
                // No watcher available - sleep briefly to avoid busy loop
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
    }
}

/// Result of a write operation.
#[derive(Debug)]
pub struct WriteResult {
    /// Range of batch positions [start, end) for inserted batches.
    /// For a single batch, this is [pos, pos+1).
    pub batch_positions: std::ops::Range<usize>,
}

/// A sealed memtable kept queryable in memory. `flushed_at_ms` is `None` while
/// the generation is still awaiting (or retrying) its flush, and `Some(t)` once
/// the flush commits — after which it lingers for `frozen_memtable_grace` so
/// in-flight as-of reads keep batch-resolved membership, then is swept.
struct FrozenMemTable {
    memtable: Arc<MemTable>,
    flushed_at_ms: Option<u64>,
}

/// ShardWriter state shared across tasks.
struct WriterState {
    memtable: MemTable,
    last_flushed_wal_entry_position: u64,
    /// Total size of frozen memtables (for backpressure).
    frozen_memtable_bytes: usize,
    /// Flush watchers for frozen memtables (for backpressure).
    frozen_flush_watchers: VecDeque<(usize, DurabilityWatcher)>,
    /// Sealed memtables, kept queryable so a concurrent reader sees no hole
    /// between `freeze_memtable` and the flush task's manifest commit, and for
    /// `frozen_memtable_grace` beyond it so as-of reads stay batch-resolved.
    /// Pushed in `freeze_memtable`; stamped `flushed_at_ms` by `flush_memtable`
    /// on commit success only (retained un-stamped on failure until a later
    /// flush or WAL replay on reopen); swept after the grace by `SweepExpired`.
    frozen_memtables: VecDeque<FrozenMemTable>,
    /// Flag to prevent duplicate memtable flush requests.
    flush_requested: bool,
    /// Counter for WAL flush threshold crossings.
    wal_flush_trigger_count: usize,
    /// Last time a WAL flush was triggered (for time-based flush).
    last_wal_flush_trigger_time: u64,
}

/// Capture a point-in-time scan handle to one in-memory memtable (active
/// or frozen — same shape). Shared by `active_memtable_ref` and
/// `in_memory_memtable_refs` so both stamp identical fields.
fn in_memory_ref(mt: &MemTable) -> crate::dataset::mem_wal::scanner::InMemoryMemTableRef {
    crate::dataset::mem_wal::scanner::InMemoryMemTableRef {
        batch_store: mt.batch_store(),
        index_store: mt
            .indexes_arc()
            .unwrap_or_else(|| Arc::new(IndexStore::new())),
        schema: mt.schema().clone(),
        generation: mt.generation(),
    }
}

fn start_time() -> std::time::Instant {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

fn now_millis() -> u64 {
    start_time().elapsed().as_millis() as u64
}

/// Replay WAL entries written after the last successfully-flushed generation
/// into the freshly-built MemTable. Updates any in-memory indexes attached to
/// the MemTable so replayed rows are immediately searchable.
///
/// Returns the WAL position the next `WalAppender::append` should use — i.e.
/// one past the highest replayed position, or the original start position if
/// the loop found nothing (we proved that position is empty by getting
/// `None` back from the tailer). The caller can pass this directly to
/// `WalAppender::seed_next_position` unconditionally so the first post-open
/// append skips its own discovery probe.
///
/// Aborts with an error if any replayed entry's `writer_epoch` is strictly
/// greater than `our_epoch` — that indicates a successor writer claimed the
/// shard between our `claim_epoch` and this replay, fencing us.
async fn replay_memtable_from_wal(
    object_store: Arc<ObjectStore>,
    base_path: Path,
    shard_id: Uuid,
    our_epoch: u64,
    manifest: &ShardManifest,
    memtable: &mut MemTable,
) -> Result<u64> {
    // WAL positions are 1-based (see `FIRST_WAL_ENTRY_POSITION`), so a
    // cursor of 0 means "no flush has ever stamped this shard" and replay
    // starts at position 1. After flushing position N the cursor holds N
    // and replay starts at N+1. The arithmetic collapses to a single
    // saturating_add(1) in both cases — we deliberately do not consult
    // `flushed_generations` here, since an external compactor may
    // legitimately drain that vector back to empty after merging its
    // contents into the base table.
    let start_position = manifest.replay_after_wal_entry_position.saturating_add(1);

    // The MemTable is always freshly built before this function runs, so
    // any existing BatchStore entries can only have come from this replay
    // pass. We index everything in `[0, batch_count)` at the end.
    debug_assert_eq!(memtable.batch_count(), 0);

    let tailer = WalTailer::new(object_store, base_path, shard_id);
    let mut position = start_position;

    loop {
        match tailer.read_entry(position).await? {
            // The first NotFound proves the WAL tip is at `position`, which
            // is the next write position to hand back.
            None => break,
            Some(entry) => {
                if entry.writer_epoch > our_epoch {
                    return Err(Error::io(format!(
                        "WAL replay aborted: entry at position {} has writer_epoch {} > our claimed epoch {} for shard {} (writer was fenced during open)",
                        position, entry.writer_epoch, our_epoch, shard_id
                    )));
                }
                // Fence sentinels deserialize to zero batches and are skipped
                // here — they carry only a position, no rows.
                if !entry.batches.is_empty() {
                    // Entries written before deletes existed lack `_tombstone`;
                    // inject `false` so they match the extended memtable schema.
                    // Normal entries already carry it and pass through unchanged.
                    let target_schema = memtable.schema().clone();
                    let batches = entry
                        .batches
                        .into_iter()
                        .map(|b| ensure_tombstone_column(b, &target_schema))
                        .collect::<Result<Vec<_>>>()?;
                    memtable.insert_batches_only(batches).await?;
                }
                position = position.checked_add(1).ok_or_else(|| {
                    Error::io(format!(
                        "WAL position overflow during replay for shard {}",
                        shard_id
                    ))
                })?;
            }
        }
    }

    // Update in-memory indexes with the replayed batches so readers see them
    // through the index path (matching what would have happened on the
    // pre-crash writer's WAL flush). Indexes from the previous writer don't
    // persist; this rebuilds them from the WAL.
    if let Some(indexes) = memtable.indexes_arc() {
        let batches_after = memtable.batch_count();
        if batches_after > 0 {
            let store = memtable.batch_store();
            let mut stored: Vec<StoredBatch> = Vec::with_capacity(batches_after);
            for pos in 0..batches_after {
                if let Some(s) = store.get(pos) {
                    stored.push(s.clone());
                }
            }
            tokio::task::spawn_blocking(move || indexes.insert_batches_parallel(&stored))
                .await
                .map_err(|e| {
                    Error::internal(format!("WAL replay index update task panicked: {}", e))
                })??;
        }
    }

    Ok(position)
}

/// Pair each primary-key column name with its field id (both derived from the
/// schema's primary key, in the same order) for [`IndexStore::enable_pk_index`].
fn pk_index_columns(pk_columns: &[String], pk_field_ids: &[i32]) -> Vec<(String, i32)> {
    pk_columns
        .iter()
        .cloned()
        .zip(pk_field_ids.iter().copied())
        .collect()
}

/// Ensure `batch` carries the `_tombstone` column required by the extended
/// memtable schema, injecting `false` for every row when it is absent.
///
/// Used on the normal write path ([`ShardWriter::put`]) where callers pass
/// base-shaped batches, and on WAL replay of entries written before deletes
/// existed (legacy entries lack the column). A batch that already carries
/// `_tombstone` (a normal replayed entry) is returned unchanged.
fn ensure_tombstone_column(
    batch: RecordBatch,
    target_schema: &Arc<ArrowSchema>,
) -> Result<RecordBatch> {
    if batch.schema().column_with_name(TOMBSTONE).is_some() {
        return Ok(batch);
    }
    let n = batch.num_rows();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(Arc::new(BooleanArray::from(vec![false; n])));
    RecordBatch::try_new(target_schema.clone(), columns).map_err(|e| {
        Error::invalid_input(format!(
            "failed to inject _tombstone column (does the batch match the base schema?): {}",
            e
        ))
    })
}

/// Build a tombstone batch from a key-only `keys` batch: the primary key
/// columns are carried through, `_tombstone` is set to `true`, and every other
/// column in the memtable schema is null.
///
/// Errors if `keys` is missing a primary key column, or if a non-PK column is
/// non-nullable (a tombstone must null it) — surfaced via the `RecordBatch`
/// validation.
fn build_tombstone_batch(
    keys: &RecordBatch,
    target_schema: &Arc<ArrowSchema>,
    pk_columns: &[String],
) -> Result<RecordBatch> {
    let n = keys.num_rows();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        let name = field.name();
        if name == TOMBSTONE {
            columns.push(Arc::new(BooleanArray::from(vec![true; n])));
        } else if pk_columns.iter().any(|c| c == name) {
            let col = keys.column_by_name(name).ok_or_else(|| {
                Error::invalid_input(format!(
                    "delete keys batch is missing primary key column '{}'",
                    name
                ))
            })?;
            columns.push(col.clone());
        } else {
            columns.push(new_null_array(field.data_type(), n));
        }
    }
    RecordBatch::try_new(target_schema.clone(), columns).map_err(|e| {
        Error::invalid_input(format!(
            "failed to build tombstone batch (is every non-primary-key column nullable?): {}",
            e
        ))
    })
}

/// Shared state for writer operations.
struct SharedWriterState {
    state: Arc<RwLock<WriterState>>,
    wal_flusher: Arc<WalFlusher>,
    wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
    memtable_flush_tx: mpsc::UnboundedSender<TriggerMemTableFlush>,
    config: ShardWriterConfig,
    schema: Arc<ArrowSchema>,
    pk_field_ids: Vec<i32>,
    /// Primary-key column names, used to (re)enable the PK-position index on
    /// each fresh active memtable created at freeze.
    pk_columns: Vec<String>,
    max_memtable_batches: usize,
    max_memtable_rows: usize,
    index_configs: Vec<MemIndexConfig>,
}

impl SharedWriterState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        state: Arc<RwLock<WriterState>>,
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        memtable_flush_tx: mpsc::UnboundedSender<TriggerMemTableFlush>,
        config: ShardWriterConfig,
        schema: Arc<ArrowSchema>,
        pk_field_ids: Vec<i32>,
        pk_columns: Vec<String>,
        max_memtable_batches: usize,
        max_memtable_rows: usize,
        index_configs: Vec<MemIndexConfig>,
    ) -> Self {
        Self {
            state,
            wal_flusher,
            wal_flush_tx,
            memtable_flush_tx,
            config,
            schema,
            pk_field_ids,
            pk_columns,
            max_memtable_batches,
            max_memtable_rows,
            index_configs,
        }
    }

    /// Freeze the current memtable and send it to the flush handler.
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn freeze_memtable(&self, state: &mut WriterState) -> Result<u64> {
        let pending_wal_range = state.memtable.batch_store().pending_wal_flush_range();
        let last_wal_entry_position = state.last_flushed_wal_entry_position;

        let old_batch_store = state.memtable.batch_store();
        let old_indexes = state.memtable.indexes_arc();

        let next_generation = state.memtable.generation() + 1;
        let mut new_memtable = MemTable::with_capacity(
            self.schema.clone(),
            next_generation,
            self.pk_field_ids.clone(),
            CacheConfig::default(),
            self.max_memtable_batches,
        )?;

        // Build an IndexStore when there are user indexes *or* a primary key:
        // the PK dedup index (and its flushed on-disk sidecar) is required for
        // cross-generation dedup even when no secondary index is configured.
        if !self.index_configs.is_empty() || !self.pk_columns.is_empty() {
            let mut indexes = IndexStore::from_configs(
                &self.index_configs,
                self.max_memtable_rows,
                self.max_memtable_batches,
            )?;
            indexes.enable_pk_index(&pk_index_columns(&self.pk_columns, &self.pk_field_ids));
            new_memtable.set_indexes_arc(Arc::new(indexes));
        }

        let mut old_memtable = std::mem::replace(&mut state.memtable, new_memtable);
        old_memtable.freeze(last_wal_entry_position);
        let _memtable_flush_watcher = old_memtable.create_memtable_flush_completion();

        if pending_wal_range.is_some() {
            let completion_cell: WatchableOnceCell<std::result::Result<WalFlushResult, String>> =
                WatchableOnceCell::new();
            let completion_reader = completion_cell.reader();
            old_memtable.set_wal_flush_completion(completion_reader);

            let end_batch_position = old_batch_store.len();
            self.wal_flusher.trigger_flush(
                WalFlushSource::BatchStore {
                    batch_store: old_batch_store,
                    indexes: old_indexes,
                },
                end_batch_position,
                Some(completion_cell),
            )?;
        }

        let frozen_size = old_memtable.estimated_size();
        state.frozen_memtable_bytes += frozen_size;

        let flush_watcher = old_memtable
            .get_memtable_flush_watcher()
            .expect("Flush watcher should exist after create_memtable_flush_completion");
        state
            .frozen_flush_watchers
            .push_back((frozen_size, flush_watcher));

        let frozen_memtable = Arc::new(old_memtable);

        // Keep this generation queryable past its manifest commit (swept after
        // the grace by `SweepExpired`). Arc refcount, not a copy — the flush
        // task holds it alive for the whole drain anyway.
        state.frozen_memtables.push_back(FrozenMemTable {
            memtable: frozen_memtable.clone(),
            flushed_at_ms: None,
        });

        debug!(
            "Frozen memtable generation {}, pending_count = {}",
            next_generation - 1,
            state.frozen_flush_watchers.len()
        );

        let _ = self.memtable_flush_tx.send(TriggerMemTableFlush::Flush {
            memtable: frozen_memtable,
            done: None,
        });

        Ok(next_generation)
    }

    /// Track batch for WAL durability.
    fn track_batch_for_wal(&self, batch_position: usize) -> super::wal::BatchDurableWatcher {
        self.wal_flusher.track_batch(batch_position)
    }

    /// Check if memtable flush is needed and trigger if so.
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn maybe_trigger_memtable_flush(&self, state: &mut WriterState) -> Result<()> {
        if state.flush_requested {
            return Ok(());
        }

        let should_flush = state.memtable.estimated_size() >= self.config.max_memtable_size
            || state.memtable.is_batch_store_full();

        if should_flush {
            state.flush_requested = true;
            self.freeze_memtable(state)?;
            state.flush_requested = false;
        }
        Ok(())
    }

    /// Check if WAL flush is needed and trigger if so.
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn maybe_trigger_wal_flush(&self, state: &mut WriterState) {
        let threshold = self.config.max_wal_buffer_size;

        let batch_count = state.memtable.batch_count();
        let total_bytes = state.memtable.estimated_size();
        let batch_store = state.memtable.batch_store();
        let indexes = state.memtable.indexes_arc();

        // Check if there are any unflushed batches
        let has_pending = batch_store.pending_wal_flush_count() > 0;

        // Check time-based trigger first
        let time_trigger = if let Some(interval) = self.config.max_wal_flush_interval {
            let interval_millis = interval.as_millis() as u64;
            let last_trigger = state.last_wal_flush_trigger_time;
            let now = now_millis();

            // If last_trigger is 0, this is the first write - start the timer but don't flush
            if last_trigger == 0 {
                state.last_wal_flush_trigger_time = now;
                None
            } else {
                let elapsed = now.saturating_sub(last_trigger);

                if elapsed >= interval_millis && has_pending {
                    state.last_wal_flush_trigger_time = now;
                    Some(now)
                } else {
                    None
                }
            }
        } else {
            None
        };

        // If time trigger fired, send a flush message
        if time_trigger.is_some() {
            let _ = self.wal_flush_tx.send(TriggerWalFlush {
                source: WalFlushSource::BatchStore {
                    batch_store,
                    indexes,
                },
                end_batch_position: batch_count,
                done: None,
            });
            return;
        }

        // Check size-based trigger
        if threshold == 0 {
            return;
        }

        // Calculate how many thresholds have been crossed (1 at 10MB, 2 at 20MB, etc.)
        let thresholds_crossed = total_bytes / threshold;

        // Trigger flush for each unclaimed threshold crossing
        while state.wal_flush_trigger_count < thresholds_crossed {
            state.wal_flush_trigger_count += 1;
            // Update last trigger time so time-based trigger doesn't fire immediately after
            state.last_wal_flush_trigger_time = now_millis();

            // Trigger WAL flush with captured batch range
            let _ = self.wal_flush_tx.send(TriggerWalFlush {
                source: WalFlushSource::BatchStore {
                    batch_store: batch_store.clone(),
                    indexes: indexes.clone(),
                },
                end_batch_position: batch_count,
                done: None,
            });
        }
    }
}

impl SharedWriterState {
    fn unflushed_memtable_bytes(&self) -> usize {
        // Total unflushed bytes = active memtable + all frozen memtables
        self.state
            .try_read()
            .ok()
            .map(|s| {
                let active = s.memtable.estimated_size();
                active + s.frozen_memtable_bytes
            })
            .unwrap_or(0)
    }

    fn oldest_memtable_watcher(&self) -> Option<DurabilityWatcher> {
        // Return a watcher for the oldest frozen memtable's flush completion.
        // If no frozen memtables, return the active memtable's watcher since it will
        // eventually be frozen and flushed.
        self.state.try_read().ok().and_then(|s| {
            // First try frozen memtable watchers
            s.frozen_flush_watchers
                .front()
                .map(|(_, watcher)| watcher.clone())
                // If no frozen memtables, use active memtable's watcher
                .or_else(|| s.memtable.get_memtable_flush_watcher())
        })
    }
}

/// Trigger-tracking state for WAL-only mode (no MemTable).
///
/// MemTable mode keeps these counters inside `WriterState`. WAL-only mode
/// has no `WriterState`, so the same counters live here, with one
/// adjustment: instead of a monotonic threshold-crossing count (which
/// can't decrement when the pending queue drains), we record the
/// `pending_bytes` value at the time of the last size-trigger. A trigger
/// fires whenever `pending_bytes` has grown by at least one
/// `max_wal_buffer_size` since `last_trigger_pending_bytes`. When the
/// pending queue is drained, `pending_bytes` drops below the recorded
/// baseline, and the next push detects this via `pending_bytes < baseline`
/// and resets the baseline.
#[derive(Debug, Default)]
struct WalOnlyTriggerState {
    /// `pending_bytes` value recorded the last time a size-based trigger
    /// fired. Resets to 0 when `pending_bytes` drops below it (drain
    /// happened since the last trigger).
    last_trigger_pending_bytes: usize,
    /// Last time a WAL flush was triggered (for time-based trigger).
    last_wal_flush_trigger_time: u64,
}

/// Per-mode state for `ShardWriter`.
enum WriterMode {
    /// Default mode: MemTable + indexes + Lance file flushing on top of
    /// the WAL.
    MemTable {
        state: Arc<RwLock<WriterState>>,
        writer_state: Arc<SharedWriterState>,
        backpressure: BackpressureController,
    },
    /// WAL-only mode: drainable pending-batch queue + WAL pipeline. No
    /// MemTable, no indexes, no Lance file flushing.
    WalOnly {
        state: Arc<WalOnlyState>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        trigger: StdRwLock<WalOnlyTriggerState>,
        backpressure: BackpressureController,
    },
}

/// Main writer for a MemWAL shard.
pub struct ShardWriter {
    config: ShardWriterConfig,
    epoch: u64,
    wal_flusher: Arc<WalFlusher>,
    task_executor: Arc<TaskExecutor>,
    manifest_store: Arc<ShardManifestStore>,
    stats: SharedWriteStats,
    mode: WriterMode,
}

impl ShardWriter {
    /// Open or create a ShardWriter.
    ///
    /// The `base_path` should come from `ObjectStore::from_uri()` to ensure
    /// WAL files are written inside the dataset directory.
    #[instrument(name = "sw_open", level = "info", skip_all, fields(shard_id = %config.shard_id, index_count = index_configs.len()))]
    pub async fn open(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: impl Into<String>,
        config: ShardWriterConfig,
        schema: Arc<ArrowSchema>,
        index_configs: Vec<MemIndexConfig>,
    ) -> Result<Self> {
        if !config.enable_memtable && !index_configs.is_empty() {
            return Err(Error::invalid_input(
                "indexes require enable_memtable = true; \
                 WAL-only mode does not maintain in-memory indexes",
            ));
        }

        // Callers pass the base schema; lance owns the `_tombstone` column and
        // appends it here so the memtable/generation schema = base + tombstone.
        // Idempotent, so a reopen that already extended the schema is a no-op.
        let schema = schema_with_tombstone(&schema);

        let base_uri = base_uri.into();
        let shard_id = config.shard_id;
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            shard_id,
            config.manifest_scan_batch_size,
        ));

        // Claim the shard (epoch-based fencing) — done once, then shared
        // with the WalAppender via `with_claimed_epoch`.
        let (epoch, manifest) = manifest_store.claim_epoch(config.shard_spec_id).await?;

        info!(
            "Opened ShardWriter for shard {} (epoch {}, generation {}, enable_memtable {})",
            shard_id, epoch, manifest.current_generation, config.enable_memtable
        );

        // Create WAL appender (owns object store, epoch, and position
        // state). Seed the appender's stats hint from the higher of the
        // manifest's two cursors: `replay_after_wal_entry_position` is
        // updated authoritatively at every MemTable flush, while
        // `wal_entry_position_last_seen` is a best-effort hint that may
        // lag behind. Either can lead the other depending on which was
        // updated last, so take the max (then +1) to get the most
        // accurate post-recovery cursor for `wal_stats()`.
        let position_hint_seed = manifest
            .wal_entry_position_last_seen
            .max(manifest.replay_after_wal_entry_position)
            .saturating_add(1);
        let wal_appender = Arc::new(WalAppender::with_claimed_epoch(
            object_store.clone(),
            base_path.clone(),
            shard_id,
            manifest_store.clone(),
            epoch,
            position_hint_seed,
        ));

        // Fence the predecessor before replay (see `write_fence_sentinel`).
        // Epoch 1 is a fresh shard with no predecessor to fence.
        if epoch >= 2 {
            wal_appender.write_fence_sentinel().await?;
        }

        // Create WAL flusher backed by the shared appender.
        let mut wal_flusher = WalFlusher::new(wal_appender);

        let (wal_flush_tx, wal_flush_rx) = mpsc::unbounded_channel();
        wal_flusher.set_flush_channel(wal_flush_tx.clone());
        let wal_flusher = Arc::new(wal_flusher);

        let stats = new_shared_stats();
        let task_executor = Arc::new(TaskExecutor::new());

        let mode = if config.enable_memtable {
            Self::open_memtable_mode(
                &config,
                &schema,
                &manifest,
                &index_configs,
                wal_flusher.clone(),
                wal_flush_tx,
                wal_flush_rx,
                object_store.clone(),
                base_path,
                base_uri,
                shard_id,
                epoch,
                manifest_store.clone(),
                stats.clone(),
                &task_executor,
            )
            .await?
        } else {
            Self::open_wal_only_mode(
                &config,
                wal_flusher.clone(),
                wal_flush_tx,
                wal_flush_rx,
                stats.clone(),
                &task_executor,
            )?
        };

        Ok(Self {
            config,
            epoch,
            wal_flusher,
            task_executor,
            manifest_store,
            stats,
            mode,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn open_memtable_mode(
        config: &ShardWriterConfig,
        schema: &Arc<ArrowSchema>,
        manifest: &ShardManifest,
        index_configs: &[MemIndexConfig],
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        wal_flush_rx: mpsc::UnboundedReceiver<TriggerWalFlush>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: String,
        shard_id: Uuid,
        epoch: u64,
        manifest_store: Arc<ShardManifestStore>,
        stats: SharedWriteStats,
        task_executor: &Arc<TaskExecutor>,
    ) -> Result<WriterMode> {
        // Create MemTable with primary key field IDs from schema
        let lance_schema = Schema::try_from(schema.as_ref())?;
        let pk_fields = lance_schema.unenforced_primary_key();
        let pk_field_ids: Vec<i32> = pk_fields.iter().map(|f| f.id).collect();
        let pk_columns: Vec<String> = pk_fields.iter().map(|f| f.name.clone()).collect();
        let mut memtable = MemTable::with_capacity(
            schema.clone(),
            manifest.current_generation,
            pk_field_ids.clone(),
            CacheConfig::default(),
            config.max_memtable_batches,
        )?;

        // Create indexes if configured and set them on the MemTable. The
        // PK-position index is enabled before any WAL replay below so replayed
        // rows are recorded in it. A primary key alone (no secondary index)
        // still needs the PK index so flush writes its on-disk dedup sidecar.
        if !index_configs.is_empty() || !pk_columns.is_empty() {
            let mut indexes = IndexStore::from_configs(
                index_configs,
                config.max_memtable_rows,
                config.max_memtable_batches,
            )?;
            indexes.enable_pk_index(&pk_index_columns(&pk_columns, &pk_field_ids));
            memtable.set_indexes_arc(Arc::new(indexes));
        }

        // Replay any WAL entries written after the last successfully-flushed
        // generation. Each entry's writer_epoch is checked against ours; an
        // entry with a strictly greater epoch indicates a successor writer
        // claimed the shard between our `claim_epoch` and replay, so we
        // abort the open with a fence error. The replay walked the tailer
        // up to the WAL tip, so we hand the discovered next-write position
        // straight to the appender — its first append skips the
        // discover_next_position probe entirely.
        let next_wal_position = replay_memtable_from_wal(
            object_store.clone(),
            base_path.clone(),
            shard_id,
            epoch,
            manifest,
            &mut memtable,
        )
        .await?;
        wal_flusher
            .wal_appender()
            .seed_next_position(next_wal_position)
            .await;

        // Seed the writer's covered-WAL cursor from the post-replay tip:
        // `next_wal_position` is one past the highest WAL entry we just
        // replayed into the active memtable, so everything strictly below
        // it is durably reflected in this writer's memtable. We can't
        // seed from `manifest.wal_entry_position_last_seen` — that field
        // is bumped on every successful tailer read by other readers, so
        // it may sit above what's actually covered by any flushed
        // generation. Subtracting 1 from a fresh shard's `next_wal_position`
        // of `FIRST_WAL_ENTRY_POSITION` (= 1) yields 0, which correctly
        // means "no entry covered yet."
        let initial_covered_wal_entry_position = next_wal_position.saturating_sub(1);

        let state = Arc::new(RwLock::new(WriterState {
            memtable,
            last_flushed_wal_entry_position: initial_covered_wal_entry_position,
            frozen_memtable_bytes: 0,
            frozen_flush_watchers: VecDeque::new(),
            frozen_memtables: VecDeque::new(),
            flush_requested: false,
            wal_flush_trigger_count: 0,
            last_wal_flush_trigger_time: 0,
        }));

        let (memtable_flush_tx, memtable_flush_rx) = mpsc::unbounded_channel();

        let flusher = Arc::new(
            MemTableFlusher::new(object_store, base_path, base_uri, shard_id, manifest_store)
                .with_warmer(config.warmer.clone()),
        );

        let backpressure = BackpressureController::new(config.clone());

        // Background WAL flush handler — parallel WAL I/O + index updates.
        let wal_handler =
            WalFlushHandler::new(wal_flusher.clone(), Some(state.clone()), stats.clone());
        task_executor.add_handler(
            "wal_flusher".to_string(),
            Box::new(wal_handler),
            wal_flush_rx,
        )?;

        // Background MemTable flush handler — frozen memtable to Lance file.
        // It rebuilds the same secondary indexes on each flushed generation.
        let memtable_handler = MemTableFlushHandler::new(
            state.clone(),
            flusher,
            epoch,
            index_configs.to_vec(),
            stats,
            config.frozen_memtable_grace,
        );
        task_executor.add_handler(
            "memtable_flusher".to_string(),
            Box::new(memtable_handler),
            memtable_flush_rx,
        )?;

        // Shared state used by `put()` to dispatch trigger checks.
        let writer_state = Arc::new(SharedWriterState::new(
            state.clone(),
            wal_flusher,
            wal_flush_tx,
            memtable_flush_tx,
            config.clone(),
            schema.clone(),
            pk_field_ids,
            pk_columns,
            config.max_memtable_batches,
            config.max_memtable_rows,
            index_configs.to_vec(),
        ));

        Ok(WriterMode::MemTable {
            state,
            writer_state,
            backpressure,
        })
    }

    fn open_wal_only_mode(
        config: &ShardWriterConfig,
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        wal_flush_rx: mpsc::UnboundedReceiver<TriggerWalFlush>,
        stats: SharedWriteStats,
        task_executor: &Arc<TaskExecutor>,
    ) -> Result<WriterMode> {
        // Background WAL flush handler — no MemTable state to consult, so
        // pass `None` for the frozen-vs-active detection.
        let wal_handler = WalFlushHandler::new(wal_flusher, None, stats);
        task_executor.add_handler(
            "wal_flusher".to_string(),
            Box::new(wal_handler),
            wal_flush_rx,
        )?;

        // Reuse `BackpressureController` (which is keyed off
        // `max_unflushed_memtable_bytes`) as the WAL-only backpressure
        // budget. WAL-only callers feed it `WalOnlyState::estimated_size()`.
        // Keeps the config knob meaningful in WAL-only mode and prevents
        // the pending queue from growing unbounded under non-durable writes.
        let backpressure = BackpressureController::new(config.clone());

        Ok(WriterMode::WalOnly {
            state: Arc::new(WalOnlyState::default()),
            wal_flush_tx,
            trigger: StdRwLock::new(WalOnlyTriggerState::default()),
            backpressure,
        })
    }

    /// Write record batches to the shard.
    ///
    /// All batches are inserted atomically with a single lock acquisition.
    /// This is more efficient than calling put() multiple times for Arrow IPC
    /// streams that contain multiple batches.
    ///
    /// # Arguments
    ///
    /// * `batches` - The record batches to write
    ///
    /// # Returns
    ///
    /// A WriteResult with batch position range and optional durability watcher.
    ///
    /// # Note
    ///
    /// Fencing is detected lazily during WAL flush via atomic writes.
    /// If another writer has taken over, the WAL flush will fail with
    /// `AlreadyExists`, indicating this writer has been fenced.
    #[instrument(name = "sw_put", level = "info", skip_all, fields(batch_count = batches.len(), shard_id = %self.config.shard_id))]
    pub async fn put(&self, batches: Vec<RecordBatch>) -> Result<WriteResult> {
        Self::validate_non_empty(&batches)?;

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                // Inject `_tombstone = false` so the batch matches the
                // extended memtable schema; callers only ever pass base-shaped
                // batches and never name the column.
                let batches = batches
                    .into_iter()
                    .map(|b| ensure_tombstone_column(b, &writer_state.schema))
                    .collect::<Result<Vec<_>>>()?;
                self.put_memtable(batches, state, writer_state, backpressure)
                    .await
            }
            WriterMode::WalOnly {
                state,
                wal_flush_tx,
                trigger,
                backpressure,
            } => {
                self.put_wal_only(batches, state, wal_flush_tx, trigger, backpressure)
                    .await
            }
        }
    }

    /// Delete rows by primary key.
    ///
    /// Each batch in `keys` must carry the shard's primary key column(s); other
    /// columns are ignored. lance builds a tombstone row per key — the primary
    /// key plus `_tombstone = true` and null in every other column — and
    /// appends it like an ordinary write. The tombstone is the newest value for
    /// its key: it wins newest-per-PK resolution (suppressing the older real
    /// row) and is then dropped from query results.
    ///
    /// Only supported in memtable mode. Because a tombstone nulls every non-PK
    /// column, those columns must be nullable in the base schema; a delete
    /// against a schema with a non-nullable non-PK column errors.
    ///
    /// ```
    /// # use lance::Result;
    /// # use lance::dataset::mem_wal::ShardWriter;
    /// # use arrow_array::RecordBatch;
    /// # async fn doc(writer: &ShardWriter, keys: Vec<RecordBatch>) -> Result<()> {
    /// writer.delete(keys).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "sw_delete", level = "info", skip_all, fields(batch_count = keys.len(), shard_id = %self.config.shard_id))]
    pub async fn delete(&self, keys: Vec<RecordBatch>) -> Result<WriteResult> {
        if keys.is_empty() {
            return Err(Error::invalid_input("Cannot delete with empty key list"));
        }
        for (i, batch) in keys.iter().enumerate() {
            if batch.num_rows() == 0 {
                return Err(Error::invalid_input(format!("Key batch {} is empty", i)));
            }
        }

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                if writer_state.pk_columns.is_empty() {
                    return Err(Error::invalid_input(
                        "delete requires a primary key, but this shard has no primary key columns",
                    ));
                }
                let tombstones = keys
                    .into_iter()
                    .map(|k| {
                        build_tombstone_batch(&k, &writer_state.schema, &writer_state.pk_columns)
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.put_memtable(tombstones, state, writer_state, backpressure)
                    .await
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "delete is only supported in memtable mode (enable_memtable = true)",
            )),
        }
    }

    /// Like [`Self::put`], but returns the durability watcher *without* awaiting
    /// it. The row is visible to reads on this writer the instant this returns;
    /// the caller awaits durability via the watcher (`None` when `durable_write`
    /// is off).
    ///
    /// This lets a caller hold an *external* lock across only the in-memory
    /// read-merge-insert and await durability after releasing it, so concurrent
    /// flushes still coalesce. The insert stays guarded by the internal
    /// `state_lock`, so `BatchStore`'s single-writer invariant holds regardless.
    ///
    /// MemTable mode only; errors in WAL-only mode (no in-memory tier).
    #[instrument(name = "sw_put_no_wait", level = "info", skip_all, fields(batch_count = batches.len(), shard_id = %self.config.shard_id))]
    pub async fn put_no_wait(
        &self,
        batches: Vec<RecordBatch>,
    ) -> Result<(WriteResult, Option<BatchDurableWatcher>)> {
        Self::validate_non_empty(&batches)?;

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                // Inject `_tombstone = false` to match the extended memtable
                // schema, mirroring `put`.
                let batches = batches
                    .into_iter()
                    .map(|b| ensure_tombstone_column(b, &writer_state.schema))
                    .collect::<Result<Vec<_>>>()?;
                self.put_memtable_no_wait(batches, state, writer_state, backpressure)
                    .await
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "put_no_wait is only supported in MemTable mode",
            )),
        }
    }

    fn validate_non_empty(batches: &[RecordBatch]) -> Result<()> {
        if batches.is_empty() {
            return Err(Error::invalid_input("Cannot write empty batch list"));
        }
        for (i, batch) in batches.iter().enumerate() {
            if batch.num_rows() == 0 {
                return Err(Error::invalid_input(format!("Batch {} is empty", i)));
            }
        }
        Ok(())
    }

    async fn put_memtable(
        &self,
        batches: Vec<RecordBatch>,
        state_lock: &Arc<RwLock<WriterState>>,
        writer_state: &Arc<SharedWriterState>,
        backpressure: &BackpressureController,
    ) -> Result<WriteResult> {
        let (result, watcher) = self
            .put_memtable_no_wait(batches, state_lock, writer_state, backpressure)
            .await?;
        // Wait for durability if configured (outside the lock).
        if let Some(mut watcher) = watcher {
            watcher.wait().await?;
        }
        Ok(result)
    }

    /// In-memory half of [`Self::put_memtable`]: insert under `state_lock`,
    /// trigger the WAL flush, and return the watcher un-awaited for the caller
    /// to wait on. `None` when `durable_write` is off. See [`Self::put_no_wait`].
    async fn put_memtable_no_wait(
        &self,
        batches: Vec<RecordBatch>,
        state_lock: &Arc<RwLock<WriterState>>,
        writer_state: &Arc<SharedWriterState>,
        backpressure: &BackpressureController,
    ) -> Result<(WriteResult, Option<BatchDurableWatcher>)> {
        // Apply backpressure if needed (before acquiring main lock)
        backpressure
            .maybe_apply_backpressure(|| {
                (
                    writer_state.unflushed_memtable_bytes(),
                    writer_state.oldest_memtable_watcher(),
                )
            })
            .await?;

        let start = std::time::Instant::now();

        // Acquire write lock for entire operation (atomic approach)
        let (batch_positions, durable_watcher, batch_store, indexes) = {
            let mut state = state_lock.write().await;

            // 1. Insert all batches into memtable atomically
            let results = state.memtable.insert_batches_only(batches).await?;

            // Get batch position range
            let start_pos = results.first().map(|(pos, _, _)| *pos).unwrap_or(0);
            let end_pos = results.last().map(|(pos, _, _)| pos + 1).unwrap_or(0);
            let batch_positions = start_pos..end_pos;

            // 2. Track last batch for WAL durability
            let durable_watcher = writer_state.track_batch_for_wal(end_pos.saturating_sub(1));

            // 3. Check if WAL flush should be triggered
            writer_state.maybe_trigger_wal_flush(&mut state);

            // 4. Check if memtable flush is needed
            if let Err(e) = writer_state.maybe_trigger_memtable_flush(&mut state) {
                warn!("Failed to trigger memtable flush: {}", e);
            }

            // Get batch_store and indexes while we have the lock (for durable_write case)
            let batch_store = state.memtable.batch_store();
            let indexes = state.memtable.indexes_arc();

            (batch_positions, durable_watcher, batch_store, indexes)
        }; // Lock released here

        self.stats.record_put(start.elapsed());

        // Trigger the flush here (outside the lock) so the watcher can resolve;
        // only the `wait()` is the caller's to schedule.
        let watcher = if self.config.durable_write {
            self.wal_flusher.trigger_flush(
                WalFlushSource::BatchStore {
                    batch_store,
                    indexes,
                },
                batch_positions.end,
                None,
            )?;
            Some(durable_watcher)
        } else {
            None
        };

        Ok((WriteResult { batch_positions }, watcher))
    }

    async fn put_wal_only(
        &self,
        batches: Vec<RecordBatch>,
        state: &Arc<WalOnlyState>,
        wal_flush_tx: &mpsc::UnboundedSender<TriggerWalFlush>,
        trigger: &StdRwLock<WalOnlyTriggerState>,
        backpressure: &BackpressureController,
    ) -> Result<WriteResult> {
        // Apply backpressure against the pending queue before pushing. The
        // budget reuses `max_unflushed_memtable_bytes` since WAL-only mode
        // shares the same "in-memory bytes waiting for durable storage"
        // shape as MemTable mode. WAL-only mode has no per-frozen-MemTable
        // watcher, so the backpressure loop falls back to its short sleep.
        backpressure
            .maybe_apply_backpressure(|| (state.estimated_size(), None))
            .await?;

        let start = std::time::Instant::now();

        // Push batches into the pending queue and capture the assigned
        // [start, end) range. `next_batch_position` is monotonic across the
        // writer's lifetime; positions are not BatchStore indices but they
        // are used the same way for durability tracking.
        let batch_positions = state.push(batches);

        // Time- and size-based triggers, mirroring MemTable mode but reading
        // pending bytes from `WalOnlyState` instead of an active MemTable.
        // Only fires for non-durable writes; durable writes go through the
        // explicit done-cell path below so flush errors (e.g., fence) reach
        // the caller.
        if !self.config.durable_write {
            let target_position = batch_positions.end;
            let pending_bytes = state.estimated_size();
            self.maybe_trigger_wal_flush_wal_only(
                state,
                wal_flush_tx,
                trigger,
                target_position,
                pending_bytes,
            );
        }

        self.stats.record_put(start.elapsed());

        // For durable writes, trigger an immediate flush and wait for the
        // done cell. Using the done cell instead of the durability watermark
        // watcher ensures flush errors (e.g., the WalAppender returning a
        // fence error) propagate back to `put` instead of hanging.
        if self.config.durable_write {
            let done = WatchableOnceCell::new();
            let reader = done.reader();
            self.wal_flusher.trigger_flush(
                WalFlushSource::WalOnly {
                    state: state.clone(),
                },
                batch_positions.end,
                Some(done),
            )?;
            let mut reader = reader;
            match reader.await_value().await {
                Some(Ok(_)) => {}
                Some(Err(msg)) => return Err(Error::io(msg)),
                None => {
                    return Err(Error::io(
                        "WAL flush handler exited before reporting durability",
                    ));
                }
            }
        }

        Ok(WriteResult { batch_positions })
    }

    /// WAL-only-mode size+time trigger. Mirrors `SharedWriterState::maybe_trigger_wal_flush`
    /// but reads its inputs from `WalOnlyState` (pending queue) instead of
    /// the active MemTable.
    fn maybe_trigger_wal_flush_wal_only(
        &self,
        state: &Arc<WalOnlyState>,
        wal_flush_tx: &mpsc::UnboundedSender<TriggerWalFlush>,
        trigger: &StdRwLock<WalOnlyTriggerState>,
        end_batch_position: usize,
        pending_bytes: usize,
    ) {
        let threshold = self.config.max_wal_buffer_size;
        let has_pending = state.batch_count() > 0;

        let mut t = trigger.write().unwrap();

        // Time-based trigger.
        if let Some(interval) = self.config.max_wal_flush_interval {
            let interval_millis = interval.as_millis() as u64;
            let now = now_millis();
            if t.last_wal_flush_trigger_time == 0 {
                t.last_wal_flush_trigger_time = now;
            } else {
                let elapsed = now.saturating_sub(t.last_wal_flush_trigger_time);
                if elapsed >= interval_millis && has_pending {
                    t.last_wal_flush_trigger_time = now;
                    let _ = wal_flush_tx.send(TriggerWalFlush {
                        source: WalFlushSource::WalOnly {
                            state: state.clone(),
                        },
                        end_batch_position,
                        done: None,
                    });
                    return;
                }
            }
        }

        if threshold == 0 {
            return;
        }

        // Size-based trigger: fire one trigger per `max_wal_buffer_size`
        // crossed since the last time we triggered. If the pending queue
        // shrank below the recorded baseline (a drain happened), reset the
        // baseline first so the next crossing fires correctly.
        if pending_bytes < t.last_trigger_pending_bytes {
            t.last_trigger_pending_bytes = 0;
        }
        while pending_bytes >= t.last_trigger_pending_bytes + threshold {
            t.last_trigger_pending_bytes += threshold;
            t.last_wal_flush_trigger_time = now_millis();
            let _ = wal_flush_tx.send(TriggerWalFlush {
                source: WalFlushSource::WalOnly {
                    state: state.clone(),
                },
                end_batch_position,
                done: None,
            });
        }
    }

    /// Get a snapshot of current write statistics.
    pub fn stats(&self) -> WriteStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the shared stats handle (for external monitoring).
    pub fn stats_handle(&self) -> SharedWriteStats {
        self.stats.clone()
    }

    /// Get the current shard manifest.
    pub async fn manifest(&self) -> Result<Option<ShardManifest>> {
        self.manifest_store.read_latest().await
    }

    /// Get the writer's epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Get the shard ID.
    pub fn shard_id(&self) -> Uuid {
        self.config.shard_id
    }

    /// Return `Err` if a successor writer has claimed a higher epoch.
    pub async fn check_fenced(&self) -> Result<()> {
        self.manifest_store.check_fenced(self.epoch).await
    }

    /// Get current MemTable statistics. Returns an error in WAL-only mode
    /// (no MemTable exists).
    pub async fn memtable_stats(&self) -> Result<MemTableStats> {
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        let batch_store = state.memtable.batch_store();
        let pending_wal = batch_store.pending_wal_flush_stats();
        Ok(MemTableStats {
            row_count: state.memtable.row_count(),
            batch_count: state.memtable.batch_count(),
            estimated_size: state.memtable.estimated_size(),
            generation: state.memtable.generation(),
            max_buffered_batch_position: batch_store.max_buffered_batch_position(),
            max_flushed_batch_position: batch_store.max_flushed_batch_position(),
            pending_wal_start_batch_position: pending_wal.start_batch_position,
            pending_wal_end_batch_position: pending_wal.end_batch_position,
            pending_wal_batch_count: pending_wal.batch_count,
            pending_wal_row_count: pending_wal.row_count,
            pending_wal_estimated_bytes: pending_wal.estimated_bytes,
        })
    }

    /// Create a scanner for querying the current MemTable data.
    ///
    /// The scanner provides read access to all data currently in the MemTable,
    /// with optional filtering, projection, and index support.
    ///
    /// The scanner captures the current `max_visible_batch_position` from the
    /// `IndexStore` at construction time to ensure consistent visibility.
    ///
    /// Returns an error in WAL-only mode.
    pub async fn scan(&self) -> Result<MemTableScanner> {
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        Ok(state.memtable.scan())
    }

    /// A handle to just the active memtable, for unified LSM scanning.
    /// Prefer [`Self::in_memory_memtable_refs`] on the read path — it also
    /// carries frozen-awaiting-flush generations.
    ///
    /// Returns an error in WAL-only mode.
    pub async fn active_memtable_ref(
        &self,
    ) -> Result<crate::dataset::mem_wal::scanner::InMemoryMemTableRef> {
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        Ok(in_memory_ref(&state.memtable))
    }

    /// The active memtable plus every frozen-awaiting-flush memtable,
    /// captured atomically under one `state.read()` (no torn freeze).
    /// Mirrors `WriterState { memtable, frozen_memtables }`. The WAL read
    /// path uses this instead of [`Self::active_memtable_ref`] so a
    /// concurrent reader sees no hole while a flush drains.
    ///
    /// Returns an error in WAL-only mode.
    pub async fn in_memory_memtable_refs(
        &self,
    ) -> Result<crate::dataset::mem_wal::scanner::InMemoryMemTables> {
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        Ok(crate::dataset::mem_wal::scanner::InMemoryMemTables {
            active: in_memory_ref(&state.memtable),
            frozen: state
                .frozen_memtables
                .iter()
                .map(|m| in_memory_ref(&m.memtable))
                .collect(),
        })
    }

    /// Returns the MemTable-mode state lock or a clear invalid_input error
    /// when running in WAL-only mode.
    fn memtable_state_lock(&self) -> Result<&Arc<RwLock<WriterState>>> {
        match &self.mode {
            WriterMode::MemTable { state, .. } => Ok(state),
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "MemTable accessor not available when enable_memtable = false (WAL-only mode)",
            )),
        }
    }

    /// Get WAL statistics.
    pub fn wal_stats(&self) -> WalStats {
        WalStats {
            next_wal_entry_position: self.wal_flusher.next_wal_entry_position(),
        }
    }

    /// Seal the active memtable so it's queued for L0 flush. No-op when
    /// the active memtable is empty. Errors in WAL-only mode or if this
    /// writer has been fenced by a successor. Pair with
    /// [`Self::wait_for_flush_drain`] to wait for the queued flush.
    ///
    /// Beyond test setup where deterministic flush points are required,
    /// this is the primary lever for callers that need to drive flushes
    /// out-of-band from the size/interval triggers — for example, to cap
    /// resident memtable bytes across many shards in a multi-table WAL
    /// writer process, or to drain the WAL ahead of a format change so
    /// the next epoch starts with no replayable entries from the old
    /// layout.
    #[instrument(name = "sw_force_seal_active", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn force_seal_active(&self) -> Result<()> {
        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                self.check_fenced().await?;
                let mut state = state.write().await;
                if state.memtable.batch_count() == 0 {
                    return Ok(());
                }
                writer_state.freeze_memtable(&mut state)?;
                Ok(())
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "force_seal_active not available in WAL-only mode (no MemTable)",
            )),
        }
    }

    /// Block until every frozen memtable in the L0 flush queue has
    /// landed and been recorded in the manifest. Does not wait on the
    /// active memtable — call [`Self::force_seal_active`] first if you
    /// want everything-on-disk. Errors in WAL-only mode, or if any
    /// awaited flush reports `DurabilityResult::Failed`.
    ///
    /// Useful in tests for deterministic post-flush assertions, and in
    /// production wherever a caller needs a synchronous fence after
    /// [`Self::force_seal_active`] — e.g. trimming memtable residency
    /// across shards in a multi-table WAL writer, or ensuring the WAL
    /// is fully drained to Lance storage before rolling to a new
    /// format/epoch.
    #[instrument(name = "sw_wait_for_flush_drain", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn wait_for_flush_drain(&self) -> Result<()> {
        let state_lock = match &self.mode {
            WriterMode::MemTable { state, .. } => state,
            WriterMode::WalOnly { .. } => {
                return Err(Error::invalid_input(
                    "wait_for_flush_drain not available in WAL-only mode (no MemTable)",
                ));
            }
        };

        loop {
            let watchers: Vec<DurabilityWatcher> = {
                let st = state_lock.read().await;
                st.frozen_flush_watchers
                    .iter()
                    .map(|(_, w)| w.clone())
                    .collect()
            };
            if watchers.is_empty() {
                return Ok(());
            }
            for mut w in watchers {
                match w.await_value().await {
                    Some(durability) => durability.into_result()?,
                    None => {
                        return Err(Error::io(
                            "MemTable flush handler exited before reporting completion",
                        ));
                    }
                }
            }
        }
    }

    /// Close the writer gracefully.
    ///
    /// Flushes pending data and shuts down background tasks.
    #[instrument(name = "sw_close", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn close(self) -> Result<()> {
        info!("Closing ShardWriter for shard {}", self.config.shard_id);

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                // Send final WAL flush message and wait for completion
                let st = state.read().await;
                let batch_store = st.memtable.batch_store();
                let indexes = st.memtable.indexes_arc();
                let batch_count = st.memtable.batch_count();
                drop(st);

                if batch_count > 0 {
                    let done = WatchableOnceCell::new();
                    let reader = done.reader();
                    if writer_state
                        .wal_flush_tx
                        .send(TriggerWalFlush {
                            source: WalFlushSource::BatchStore {
                                batch_store,
                                indexes,
                            },
                            end_batch_position: batch_count,
                            done: Some(done),
                        })
                        .is_ok()
                    {
                        let mut reader = reader;
                        let _ = reader.await_value().await;
                    }
                }

                // Freeze the active memtable (if any rows) so it joins the
                // pending-flush queue, then wait for every frozen
                // memtable's flush to complete. Without this, close() left
                // any rows that hadn't crossed the size/batch trigger
                // sitting in memory at shutdown — they were durable in the
                // WAL but never produced a Lance fragment, which is
                // surprising for callers who reasonably expect close() to
                // make all data fully durable in the LSM sense (and which
                // makes flush-cost benchmarks impossible to measure).
                //
                // Propagate any freeze error: at close time the caller
                // has explicitly asked for full durability, so silently
                // dropping a freeze failure would lose data without any
                // signal. If freeze fails, surface the error rather than
                // continuing on to drain only the pre-existing frozen
                // memtables (whose flushes can still be waited on, but
                // the caller now knows the close was incomplete).
                let watchers: Vec<_> = {
                    let mut st = state.write().await;
                    if st.memtable.row_count() > 0 {
                        writer_state.freeze_memtable(&mut st)?;
                    }
                    st.frozen_flush_watchers
                        .iter()
                        .map(|(_, w)| w.clone())
                        .collect()
                };
                for mut w in watchers {
                    let _ = w.await_value().await;
                }
            }
            WriterMode::WalOnly {
                state,
                wal_flush_tx,
                trigger: _,
                backpressure: _,
            } => {
                // Drain any pending batches via a final flush; wait for completion.
                let pending = state.batch_count();
                let end_position = state.next_batch_position();
                if pending > 0 {
                    let done = WatchableOnceCell::new();
                    let reader = done.reader();
                    if wal_flush_tx
                        .send(TriggerWalFlush {
                            source: WalFlushSource::WalOnly {
                                state: state.clone(),
                            },
                            end_batch_position: end_position,
                            done: Some(done),
                        })
                        .is_ok()
                    {
                        let mut reader = reader;
                        let _ = reader.await_value().await;
                    }
                }
            }
        }

        // Shutdown background tasks
        self.task_executor.shutdown_all().await?;

        info!("ShardWriter closed for shard {}", self.config.shard_id);
        Ok(())
    }
}

/// MemTable statistics.
#[derive(Debug, Clone)]
pub struct MemTableStats {
    pub row_count: usize,
    pub batch_count: usize,
    pub estimated_size: usize,
    pub generation: u64,
    pub max_buffered_batch_position: Option<usize>,
    pub max_flushed_batch_position: Option<usize>,
    pub pending_wal_start_batch_position: Option<usize>,
    pub pending_wal_end_batch_position: Option<usize>,
    pub pending_wal_batch_count: usize,
    pub pending_wal_row_count: usize,
    pub pending_wal_estimated_bytes: usize,
}

/// WAL statistics.
#[derive(Debug, Clone)]
pub struct WalStats {
    /// Next WAL entry position to be used.
    pub next_wal_entry_position: u64,
}

/// Background handler for WAL flush operations.
///
/// This handler does parallel WAL I/O + index updates during flush.
/// Indexes are passed through the TriggerWalFlush message.
struct WalFlushHandler {
    wal_flusher: Arc<WalFlusher>,
    /// MemTable-mode writer state, used to detect "frozen vs active" flushes
    /// via Arc::ptr_eq on the active batch_store. `None` when running in
    /// WAL-only mode (no MemTable, no frozen-vs-active distinction).
    memtable_state: Option<Arc<RwLock<WriterState>>>,
    stats: SharedWriteStats,
}

impl WalFlushHandler {
    fn new(
        wal_flusher: Arc<WalFlusher>,
        memtable_state: Option<Arc<RwLock<WriterState>>>,
        stats: SharedWriteStats,
    ) -> Self {
        Self {
            wal_flusher,
            memtable_state,
            stats,
        }
    }
}

#[async_trait]
impl MessageHandler<TriggerWalFlush> for WalFlushHandler {
    async fn handle(&mut self, message: TriggerWalFlush) -> Result<()> {
        let TriggerWalFlush {
            source,
            end_batch_position,
            done,
        } = message;

        let result = self.do_flush(source, end_batch_position).await;

        // Propagate the just-appended WAL entry position back into the
        // writer state so a subsequent MemTable freeze can stamp the
        // correct `covered_wal_entry_position` into the manifest. Without
        // this, `replay_after_wal_entry_position` stays at 0 and replay
        // re-reads already-flushed entries after restart.
        //
        // Always update state before signalling the completion cell so any
        // waiter that reads state immediately after the cell fires sees
        // the new position.
        if let (Ok(flush_result), Some(state_lock)) = (&result, &self.memtable_state)
            && let Some(entry) = &flush_result.entry
        {
            let mut state = state_lock.write().await;
            state.last_flushed_wal_entry_position =
                state.last_flushed_wal_entry_position.max(entry.position);
        }

        // Notify completion if requested
        if let Some(cell) = done {
            cell.write(result.map_err(|e| e.to_string()));
        }

        Ok(())
    }
}

impl WalFlushHandler {
    /// Unified flush method for both active and frozen memtables and for
    /// WAL-only mode.
    ///
    /// For BatchStore sources, detects frozen vs active flush by comparing
    /// the passed batch_store with the current active memtable's
    /// batch_store. If different, it's a frozen memtable flush.
    #[instrument(
        name = "wal_do_flush",
        level = "debug",
        skip_all,
        fields(end_batch_position)
    )]
    async fn do_flush(
        &self,
        source: WalFlushSource,
        end_batch_position: usize,
    ) -> Result<WalFlushResult> {
        let start = Instant::now();

        // Whether this flush actually updates any in-memory indexes — only
        // a BatchStore source carrying a non-empty `IndexStore` does. Used
        // to gate the `record_index_update` stat so WAL-only flushes don't
        // pollute the index-update counters.
        let has_indexes = matches!(
            &source,
            WalFlushSource::BatchStore {
                indexes: Some(_),
                ..
            }
        );

        // Early-out for BatchStore sources where the watermark already
        // covers the requested end position. Detection of "frozen flush"
        // requires the active memtable's batch_store; WAL-only handlers
        // don't have one (`memtable_state` is `None`) and never receive a
        // BatchStore source, so the early-out simplifies to the watermark
        // comparison.
        if let WalFlushSource::BatchStore { batch_store, .. } = &source {
            let max_flushed = batch_store.max_flushed_batch_position();
            let flushed_up_to = max_flushed.map(|p| p + 1).unwrap_or(0);
            let is_frozen_flush = if let Some(state_lock) = &self.memtable_state {
                let state = state_lock.read().await;
                !Arc::ptr_eq(batch_store, &state.memtable.batch_store())
            } else {
                false
            };
            if !is_frozen_flush && flushed_up_to >= end_batch_position {
                return Ok(empty_flush_result());
            }
        }

        // Delegate the actual flush to the WalFlusher.
        let flush_result = self.wal_flusher.flush(&source, end_batch_position).await?;

        let batches_flushed = flush_result
            .entry
            .as_ref()
            .map(|e| e.num_batches)
            .unwrap_or(0);

        if batches_flushed > 0 {
            self.stats
                .record_wal_flush(start.elapsed(), flush_result.wal_bytes);
            self.stats.record_wal_io(flush_result.wal_io_duration);
            if has_indexes {
                self.stats.record_index_update(
                    flush_result.index_update_duration,
                    flush_result.rows_indexed,
                );
            }
        }

        Ok(flush_result)
    }
}

/// Background handler for MemTable flush operations.
///
/// This handler receives frozen memtables directly via messages and flushes them to Lance storage.
/// Freezing is done by the writer (via SharedWriterState::freeze_memtable) to ensure
/// immediate memtable switching, so writes can continue on the new memtable while this
/// handler flushes in the background.
struct MemTableFlushHandler {
    state: Arc<RwLock<WriterState>>,
    flusher: Arc<MemTableFlusher>,
    epoch: u64,
    /// Secondary index configs to rebuild on each flushed generation. When
    /// non-empty the handler flushes via [`MemTableFlusher::flush_with_indexes`]
    /// so queries over flushed generations use index lookups instead of full
    /// scans — and so vector search's index-only `fast_search` can see the data
    /// at all.
    index_configs: Vec<MemIndexConfig>,
    stats: SharedWriteStats,
    /// How long a frozen memtable lingers in memory after its flush commits
    /// before `SweepExpired` evicts it. See `ShardWriterConfig::frozen_memtable_grace`.
    grace: Duration,
}

impl MemTableFlushHandler {
    fn new(
        state: Arc<RwLock<WriterState>>,
        flusher: Arc<MemTableFlusher>,
        epoch: u64,
        index_configs: Vec<MemIndexConfig>,
        stats: SharedWriteStats,
        grace: Duration,
    ) -> Self {
        Self {
            state,
            flusher,
            epoch,
            index_configs,
            stats,
            grace,
        }
    }

    /// Evict frozen memtables whose post-flush grace has elapsed. Un-stamped
    /// (not-yet-flushed) entries are always kept.
    async fn sweep_expired_frozen(&self) {
        let now = now_millis();
        let grace_ms = self.grace.as_millis() as u64;
        let mut state = self.state.write().await;
        state
            .frozen_memtables
            .retain(|frozen| match frozen.flushed_at_ms {
                Some(flushed_at) => now.saturating_sub(flushed_at) < grace_ms,
                None => true,
            });
    }
}

#[async_trait]
impl MessageHandler<TriggerMemTableFlush> for MemTableFlushHandler {
    fn tickers(&mut self) -> Vec<(Duration, MessageFactory<TriggerMemTableFlush>)> {
        // Zero grace evicts on commit, so no sweeper is needed.
        if self.grace.is_zero() {
            return vec![];
        }
        // Sweep often enough that eviction lags the grace by at most ~1/3, so a
        // generation lives no more than ~grace * 4/3 past its flush commit.
        let tick = (self.grace / 3).max(Duration::from_millis(100));
        vec![(tick, Box::new(|| TriggerMemTableFlush::SweepExpired))]
    }

    async fn handle(&mut self, message: TriggerMemTableFlush) -> Result<()> {
        match message {
            TriggerMemTableFlush::Flush { memtable, done } => {
                let result = self.flush_memtable(memtable).await;
                if let Some(tx) = done {
                    // Send result through the channel - caller is waiting for it
                    let _ = tx.send(result);
                } else {
                    // No done channel, propagate errors
                    result?;
                }
            }
            TriggerMemTableFlush::SweepExpired => self.sweep_expired_frozen().await,
        }
        Ok(())
    }
}

impl MemTableFlushHandler {
    /// Flush the given frozen memtable to Lance storage.
    ///
    /// This method waits for the WAL flush to complete (sent at freeze time),
    /// then flushes to Lance storage. The WAL flush is already queued by
    /// freeze_memtable to ensure strict ordering of WAL entries.
    ///
    /// Whether the flush succeeds or fails, the memtable's flush-completion
    /// watcher is always signaled and the backpressure queue is always drained
    /// for this memtable. Otherwise `wait_for_flush_drain` would observe a
    /// dropped watch channel and return `Err` instead of the actual outcome.
    #[instrument(name = "mt_flush", level = "info", skip_all, fields(generation = memtable.generation(), row_count = memtable.row_count()))]
    async fn flush_memtable(
        &mut self,
        memtable: Arc<MemTable>,
    ) -> Result<super::memtable::flush::FlushResult> {
        let start = Instant::now();
        let memtable_size = memtable.estimated_size();

        let flush_result = async {
            // Step 1: Wait for WAL flush completion (already queued at freeze time).
            // The TriggerWalFlush message was sent by freeze_memtable to ensure
            // strict ordering of WAL entries. If the freeze didn't trigger a
            // flush (no pending WAL range), there's no completion cell and the
            // memtable was already WAL-flushed by an earlier put.
            let wal_flushed_position =
                if let Some(mut completion_reader) = memtable.take_wal_flush_completion() {
                    match completion_reader.await_value().await {
                        Some(Ok(flush_result)) => flush_result.entry.map(|e| e.position),
                        Some(Err(e)) => return Err(Error::io(format!("WAL flush failed: {}", e))),
                        None => {
                            return Err(Error::io(
                                "WAL flush handler exited before reporting completion",
                            ));
                        }
                    }
                } else {
                    None
                };

            // Step 2: Flush the memtable to Lance storage. The covered WAL
            // entry position is either the one we just appended (per-memtable,
            // from the completion cell — authoritative even when concurrent
            // flushes have raced ahead in `state.last_flushed_wal_entry_position`)
            // or, when no flush was triggered at freeze time, the memtable's
            // frozen-at marker captured at freeze. Stamping this into the
            // manifest is what lets replay-on-reopen skip entries this
            // generation covers.
            let covered_wal_entry_position = wal_flushed_position
                .or_else(|| memtable.frozen_at_wal_entry_position())
                .unwrap_or(0);
            // Rebuild secondary indexes on the flushed generation so later
            // queries hit an index instead of scanning. Skip the extra
            // dataset open when there are no indexes to build. The indexed
            // path's future is boxed to keep this async block's nesting
            // under the type-layout recursion limit.
            if self.index_configs.is_empty() {
                self.flusher
                    .flush(&memtable, self.epoch, covered_wal_entry_position)
                    .await
            } else {
                Box::pin(self.flusher.flush_with_indexes(
                    &memtable,
                    self.epoch,
                    &self.index_configs,
                    covered_wal_entry_position,
                ))
                .await
            }
        }
        .await;

        // Step 3: Always signal completion (with the outcome) and drain
        // backpressure state for this memtable, even on failure.
        let durability = match &flush_result {
            Ok(_) => DurabilityResult::ok(),
            Err(e) => DurabilityResult::err(e.to_string()),
        };
        memtable.signal_memtable_flush_complete(durability);

        {
            let mut state = self.state.write().await;
            // Backpressure drain: unconditional so `wait_for_flush_drain`
            // sees the watcher's error signal, not a dropped channel.
            if let Some((_size, _watcher)) = state.frozen_flush_watchers.pop_front() {
                state.frozen_memtable_bytes =
                    state.frozen_memtable_bytes.saturating_sub(memtable_size);
            }
            // Retire the frozen handle on commit success, keyed by generation
            // (non-FIFO completion is fine). Zero grace evicts here; otherwise
            // stamp the grace clock so it lingers for multi-part as-of reads
            // until `SweepExpired`. On failure leave it un-stamped: rows stay in
            // the read union until a later flush or WAL replay, else a transient
            // error reopens the hole.
            if flush_result.is_ok() {
                let flushed_generation = memtable.generation();
                if self.grace.is_zero() {
                    state
                        .frozen_memtables
                        .retain(|frozen| frozen.memtable.generation() != flushed_generation);
                } else {
                    let now = now_millis();
                    for frozen in state.frozen_memtables.iter_mut() {
                        if frozen.memtable.generation() == flushed_generation {
                            frozen.flushed_at_ms = Some(now);
                        }
                    }
                }
            }
        }

        let result = flush_result?;

        self.stats
            .record_memtable_flush(start.elapsed(), result.rows_flushed);

        info!(
            "Flushed frozen memtable generation {} ({} rows in {:?})",
            result.generation.generation,
            result.rows_flushed,
            start.elapsed()
        );

        Ok(result)
    }
}

// ============================================================================
// Write Statistics
// ============================================================================

/// Write performance statistics.
///
/// All fields use atomic operations for thread-safe updates.
/// Use `snapshot()` to get a consistent view of all stats.
#[derive(Debug, Default)]
pub struct WriteStats {
    // Put operation stats
    put_count: AtomicU64,
    put_time_nanos: AtomicU64,

    // WAL flush stats (total time = max(wal_io, index_update) due to parallel execution)
    wal_flush_count: AtomicU64,
    wal_flush_time_nanos: AtomicU64,
    wal_flush_bytes: AtomicU64,

    // WAL flush sub-component stats (for diagnosing bottlenecks)
    wal_io_time_nanos: AtomicU64,
    wal_io_count: AtomicU64,
    index_update_time_nanos: AtomicU64,
    index_update_count: AtomicU64,
    index_update_rows: AtomicU64,

    // MemTable flush stats
    memtable_flush_count: AtomicU64,
    memtable_flush_time_nanos: AtomicU64,
    memtable_flush_rows: AtomicU64,
}

/// Snapshot of write statistics at a point in time.
#[derive(Debug, Clone)]
pub struct WriteStatsSnapshot {
    pub put_count: u64,
    pub put_time: Duration,

    pub wal_flush_count: u64,
    pub wal_flush_time: Duration,
    pub wal_flush_bytes: u64,

    // WAL flush sub-component stats
    pub wal_io_time: Duration,
    pub wal_io_count: u64,
    pub index_update_time: Duration,
    pub index_update_count: u64,
    pub index_update_rows: u64,

    pub memtable_flush_count: u64,
    pub memtable_flush_time: Duration,
    pub memtable_flush_rows: u64,
}

impl WriteStats {
    /// Create a new stats collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a put operation.
    pub fn record_put(&self, duration: Duration) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
        self.put_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record a WAL flush operation (total time including parallel I/O and index).
    pub fn record_wal_flush(&self, duration: Duration, bytes: usize) {
        self.wal_flush_count.fetch_add(1, Ordering::Relaxed);
        self.wal_flush_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.wal_flush_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Record WAL I/O duration (sub-component of WAL flush).
    pub fn record_wal_io(&self, duration: Duration) {
        self.wal_io_count.fetch_add(1, Ordering::Relaxed);
        self.wal_io_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record index update duration (sub-component of WAL flush).
    pub fn record_index_update(&self, duration: Duration, rows: usize) {
        self.index_update_count.fetch_add(1, Ordering::Relaxed);
        self.index_update_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.index_update_rows
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Record a MemTable flush operation.
    pub fn record_memtable_flush(&self, duration: Duration, rows: usize) {
        self.memtable_flush_count.fetch_add(1, Ordering::Relaxed);
        self.memtable_flush_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.memtable_flush_rows
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Get a snapshot of current statistics.
    pub fn snapshot(&self) -> WriteStatsSnapshot {
        WriteStatsSnapshot {
            put_count: self.put_count.load(Ordering::Relaxed),
            put_time: Duration::from_nanos(self.put_time_nanos.load(Ordering::Relaxed)),

            wal_flush_count: self.wal_flush_count.load(Ordering::Relaxed),
            wal_flush_time: Duration::from_nanos(self.wal_flush_time_nanos.load(Ordering::Relaxed)),
            wal_flush_bytes: self.wal_flush_bytes.load(Ordering::Relaxed),

            wal_io_time: Duration::from_nanos(self.wal_io_time_nanos.load(Ordering::Relaxed)),
            wal_io_count: self.wal_io_count.load(Ordering::Relaxed),
            index_update_time: Duration::from_nanos(
                self.index_update_time_nanos.load(Ordering::Relaxed),
            ),
            index_update_count: self.index_update_count.load(Ordering::Relaxed),
            index_update_rows: self.index_update_rows.load(Ordering::Relaxed),

            memtable_flush_count: self.memtable_flush_count.load(Ordering::Relaxed),
            memtable_flush_time: Duration::from_nanos(
                self.memtable_flush_time_nanos.load(Ordering::Relaxed),
            ),
            memtable_flush_rows: self.memtable_flush_rows.load(Ordering::Relaxed),
        }
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.put_count.store(0, Ordering::Relaxed);
        self.put_time_nanos.store(0, Ordering::Relaxed);

        self.wal_flush_count.store(0, Ordering::Relaxed);
        self.wal_flush_time_nanos.store(0, Ordering::Relaxed);
        self.wal_flush_bytes.store(0, Ordering::Relaxed);

        self.wal_io_time_nanos.store(0, Ordering::Relaxed);
        self.wal_io_count.store(0, Ordering::Relaxed);
        self.index_update_time_nanos.store(0, Ordering::Relaxed);
        self.index_update_count.store(0, Ordering::Relaxed);
        self.index_update_rows.store(0, Ordering::Relaxed);

        self.memtable_flush_count.store(0, Ordering::Relaxed);
        self.memtable_flush_time_nanos.store(0, Ordering::Relaxed);
        self.memtable_flush_rows.store(0, Ordering::Relaxed);
    }
}

impl WriteStatsSnapshot {
    /// Get average put latency.
    pub fn avg_put_latency(&self) -> Option<Duration> {
        if self.put_count > 0 {
            Some(self.put_time / self.put_count as u32)
        } else {
            None
        }
    }

    /// Get put throughput (puts per second based on time spent in puts).
    pub fn put_throughput(&self) -> f64 {
        if self.put_time.as_secs_f64() > 0.0 {
            self.put_count as f64 / self.put_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get average WAL flush latency.
    pub fn avg_wal_flush_latency(&self) -> Option<Duration> {
        if self.wal_flush_count > 0 {
            Some(self.wal_flush_time / self.wal_flush_count as u32)
        } else {
            None
        }
    }

    /// Get average WAL flush size in bytes.
    pub fn avg_wal_flush_bytes(&self) -> Option<u64> {
        if self.wal_flush_count > 0 {
            Some(self.wal_flush_bytes / self.wal_flush_count)
        } else {
            None
        }
    }

    /// Get WAL write throughput (bytes per second based on WAL flush time).
    pub fn wal_throughput_bytes(&self) -> f64 {
        if self.wal_flush_time.as_secs_f64() > 0.0 {
            self.wal_flush_bytes as f64 / self.wal_flush_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get average WAL I/O latency.
    pub fn avg_wal_io_latency(&self) -> Option<Duration> {
        if self.wal_io_count > 0 {
            Some(self.wal_io_time / self.wal_io_count as u32)
        } else {
            None
        }
    }

    /// Get average index update latency.
    pub fn avg_index_update_latency(&self) -> Option<Duration> {
        if self.index_update_count > 0 {
            Some(self.index_update_time / self.index_update_count as u32)
        } else {
            None
        }
    }

    /// Get average rows per index update.
    pub fn avg_index_update_rows(&self) -> Option<u64> {
        if self.index_update_count > 0 {
            Some(self.index_update_rows / self.index_update_count)
        } else {
            None
        }
    }

    /// Get average MemTable flush latency.
    pub fn avg_memtable_flush_latency(&self) -> Option<Duration> {
        if self.memtable_flush_count > 0 {
            Some(self.memtable_flush_time / self.memtable_flush_count as u32)
        } else {
            None
        }
    }

    /// Get average MemTable flush size in rows.
    pub fn avg_memtable_flush_rows(&self) -> Option<u64> {
        if self.memtable_flush_count > 0 {
            Some(self.memtable_flush_rows / self.memtable_flush_count)
        } else {
            None
        }
    }

    /// Log stats summary using tracing (for structured telemetry).
    pub fn log_summary(&self, prefix: &str) {
        tracing::info!(
            prefix = prefix,
            put_count = self.put_count,
            put_throughput = self.put_throughput(),
            put_avg_latency_us = self.avg_put_latency().unwrap_or_default().as_micros() as u64,
            wal_flush_count = self.wal_flush_count,
            wal_flush_bytes = self.wal_flush_bytes,
            wal_avg_latency_us =
                self.avg_wal_flush_latency().unwrap_or_default().as_micros() as u64,
            memtable_flush_count = self.memtable_flush_count,
            memtable_flush_rows = self.memtable_flush_rows,
            memtable_avg_latency_us = self
                .avg_memtable_flush_latency()
                .unwrap_or_default()
                .as_micros() as u64,
            "MemWAL stats summary"
        );
    }

    /// Log detailed WAL flush breakdown (WAL I/O vs index update) using tracing.
    pub fn log_wal_breakdown(&self, prefix: &str) {
        if self.wal_flush_count > 0 {
            tracing::info!(
                prefix = prefix,
                wal_total_latency_us =
                    self.avg_wal_flush_latency().unwrap_or_default().as_micros() as u64,
                wal_io_latency_us =
                    self.avg_wal_io_latency().unwrap_or_default().as_micros() as u64,
                index_update_latency_us = self
                    .avg_index_update_latency()
                    .unwrap_or_default()
                    .as_micros() as u64,
                index_update_rows = self.index_update_rows,
                "MemWAL WAL flush breakdown"
            );
        }
    }
}

/// Shared stats handle for use across components.
pub type SharedWriteStats = Arc<WriteStats>;

/// Create a new shared stats collector.
pub fn new_shared_stats() -> SharedWriteStats {
    Arc::new(WriteStats::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field};
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, String, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, uri, temp_dir)
    }

    /// Base schema with `id` marked as the unenforced primary key (delete needs
    /// a PK). `name` is nullable so a tombstone can null it.
    fn create_pk_test_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = std::collections::HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);
        Arc::new(ArrowSchema::new(vec![
            id,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn id_only_keys(ids: &[i32]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(ids.to_vec()))],
        )
        .unwrap()
    }

    #[test]
    fn test_ensure_tombstone_column_injects_false() {
        let base = create_test_schema();
        let target = schema_with_tombstone(&base);
        let out = ensure_tombstone_column(create_test_batch(&base, 0, 3), &target).unwrap();
        assert_eq!(out.schema(), target);
        let ts = out
            .column_by_name(TOMBSTONE)
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(
            (0..3).all(|i| !ts.value(i)),
            "put injects _tombstone = false"
        );
        // Idempotent: a batch already carrying the column passes through.
        let again = ensure_tombstone_column(out.clone(), &target).unwrap();
        assert_eq!(again.schema(), out.schema());
    }

    #[test]
    fn test_build_tombstone_batch_shape() {
        let target = schema_with_tombstone(&create_test_schema());
        let tomb =
            build_tombstone_batch(&id_only_keys(&[5, 7]), &target, &["id".to_string()]).unwrap();
        assert_eq!(tomb.schema(), target);
        assert_eq!(tomb.num_rows(), 2);
        let ids = tomb
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[5, 7]);
        let name = tomb.column_by_name("name").unwrap();
        assert!(
            name.is_null(0) && name.is_null(1),
            "non-PK columns are null in a tombstone"
        );
        let ts = tomb
            .column_by_name(TOMBSTONE)
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(ts.value(0) && ts.value(1), "_tombstone = true");
    }

    #[test]
    fn test_build_tombstone_batch_missing_pk_errors() {
        let target = schema_with_tombstone(&create_test_schema());
        let keys = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "other",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        assert!(build_tombstone_batch(&keys, &target, &["id".to_string()]).is_err());
    }

    #[test]
    fn test_build_tombstone_batch_non_nullable_nonpk_errors() {
        // A tombstone must null every non-PK column; a non-nullable one fails.
        let base = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
        ]));
        let target = schema_with_tombstone(&base);
        assert!(build_tombstone_batch(&id_only_keys(&[1]), &target, &["id".to_string()]).is_err());
    }

    #[tokio::test]
    async fn test_shard_writer_delete_round_trip() {
        use crate::dataset::mem_wal::scanner::LsmScanner;
        use futures::TryStreamExt;

        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: true,
            ..Default::default()
        };
        let shard_id = config.shard_id;
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // ids 0..5, then delete id=2.
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.delete(vec![id_only_keys(&[2])]).await.unwrap();

        let refs = writer.in_memory_memtable_refs().await.unwrap();
        let scanner = LsmScanner::without_base_table(
            schema.clone(),
            base_uri,
            vec![],
            vec!["id".to_string()],
        )
        .with_in_memory_memtables(shard_id, refs);
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut ids: Vec<i32> = Vec::new();
        for b in &batches {
            let arr = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            ids.extend((0..arr.len()).map(|i| arr.value(i)));
        }
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![0, 1, 3, 4],
            "id=2 deleted; tombstone not surfaced"
        );

        writer.close().await.unwrap();
    }

    /// Read every surviving `id` through the full LSM read path (which folds
    /// `NOT _tombstone` per source) over the writer's *flushed* generations,
    /// with an optional filter. Mirrors how a query reads a WAL table after a
    /// flush — the path the wallop fuzz exercised when it caught a deleted row
    /// resurfacing.
    async fn read_flushed_ids_via_lsm(
        writer: &ShardWriter,
        schema: Arc<ArrowSchema>,
        base_uri: &str,
        shard_id: Uuid,
        filter: Option<&str>,
    ) -> Vec<i32> {
        use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
        use futures::TryStreamExt;

        let manifest = writer.manifest().await.unwrap().unwrap();
        let mut snapshot =
            ShardSnapshot::new(shard_id).with_current_generation(manifest.current_generation);
        for fg in &manifest.flushed_generations {
            snapshot = snapshot.with_flushed_generation(fg.generation, fg.path.clone());
        }
        let mut scanner = LsmScanner::without_base_table(
            schema,
            base_uri.to_string(),
            vec![snapshot],
            vec!["id".to_string()],
        );
        if let Some(predicate) = filter {
            scanner = scanner.filter(predicate).unwrap();
        }
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut ids: Vec<i32> = Vec::new();
        for b in &batches {
            let arr = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            ids.extend((0..arr.len()).map(|i| arr.value(i)));
        }
        ids.sort_unstable();
        ids
    }

    fn flush_test_config(shard_id: Uuid) -> ShardWriterConfig {
        ShardWriterConfig {
            shard_id,
            durable_write: false,
            sync_indexed_write: true,
            manifest_scan_batch_size: 2,
            ..Default::default()
        }
    }

    /// Delete a key, then flush: the tombstone and the live row land in the
    /// *same* flushed generation, so flush-time dedup must keep the tombstone
    /// (newest) and the read must fold it away. Regression for the wallop
    /// phantom (deleted row resurfacing in a filtered read after flush).
    #[tokio::test]
    async fn test_shard_writer_delete_then_flush_round_trip() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let shard_id = Uuid::new_v4();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            flush_test_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // ids 0..5, delete id=2, THEN flush (insert + tombstone in one gen).
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.delete(vec![id_only_keys(&[2])]).await.unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        assert_eq!(
            read_flushed_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
            vec![0, 1, 3, 4],
            "id=2 deleted before flush; tombstone must not surface in a flushed-gen scan"
        );
        // The filtered read path (folds NOT _tombstone into the predicate) must
        // also drop it — this is the exact wallop failure shape (`id < 3`).
        assert_eq!(
            read_flushed_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 3"))
                .await,
            vec![0, 1],
            "filtered read after flush must not resurface deleted id=2"
        );

        writer.close().await.unwrap();
    }

    /// Insert+flush, then delete+flush: the tombstone lands in a *newer*
    /// generation than the live row, so the cross-generation block-list must
    /// mask the older row by PK. This is the wallop scenario (seed flushed,
    /// then delete, then flush).
    #[tokio::test]
    async fn test_shard_writer_delete_across_flushed_generations() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let shard_id = Uuid::new_v4();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            flush_test_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // gen 1: ids 0..5.
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // gen 2: tombstone for id=0 (a later generation than the live row).
        writer.delete(vec![id_only_keys(&[0])]).await.unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        assert_eq!(
            read_flushed_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
            vec![1, 2, 3, 4],
            "id=0 tombstoned in a newer gen must mask the older gen's live row"
        );
        assert_eq!(
            read_flushed_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 1"))
                .await,
            Vec::<i32>::new(),
            "filtered read 'id < 1' must not resurface cross-gen deleted id=0"
        );

        writer.close().await.unwrap();
    }

    /// Same as the cross-generation case, but the flushed generations carry a
    /// BTree index on `id` (as every wallop table does). A filtered read
    /// `id < 1` resolves through the scalar index; the `NOT _tombstone` residual
    /// must still be applied or the deleted row leaks. This is the exact wallop
    /// failure (BTree id + `FilteredRead 'id < 1'` resurfacing deleted id=0).
    #[tokio::test]
    async fn test_shard_writer_delete_across_flushed_generations_indexed() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let shard_id = Uuid::new_v4();
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            flush_test_config(shard_id),
            schema.clone(),
            index_configs,
        )
        .await
        .unwrap();

        // gen 1: ids 0..5 (indexed). gen 2: tombstone for id=0.
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();
        writer.delete(vec![id_only_keys(&[0])]).await.unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        assert_eq!(
            read_flushed_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
            vec![1, 2, 3, 4],
            "indexed cross-gen: full scan must mask deleted id=0"
        );
        assert_eq!(
            read_flushed_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 1"))
                .await,
            Vec::<i32>::new(),
            "indexed filtered read 'id < 1' must not resurface deleted id=0 (wallop repro)"
        );

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_delete_validation_errors() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;

        // No primary key on the schema → delete is rejected.
        let no_pk = create_test_schema();
        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            ShardWriterConfig {
                shard_id: Uuid::new_v4(),
                durable_write: false,
                ..Default::default()
            },
            no_pk,
            vec![],
        )
        .await
        .unwrap();
        assert!(
            writer.delete(vec![id_only_keys(&[1])]).await.is_err(),
            "delete without a primary key must error"
        );
        // Empty key list is rejected.
        assert!(writer.delete(vec![]).await.is_err());
        writer.close().await.unwrap();
    }

    /// The compaction Phase-1 hard-delete primitive: a key-only `merge_insert`
    /// with `WhenMatched::Delete` + `use_index` over a COMPOSITE join key
    /// `(id, shard_key)` where only `id` carries a scalar index. The partial
    /// index makes `indexed_join_keys` non-empty, so the scalar-index merge path
    /// engages — and it must still delete exactly the `(0, 0)` row. If it
    /// no-ops (or mismatches on the partial key), the compactor trims the WAL
    /// tombstone while the base row survives → the wallop resurface phantom.
    #[tokio::test]
    async fn test_indexed_composite_key_delete_removes_row() {
        use crate::Dataset;
        use crate::dataset::{WhenMatched, WhenNotMatched, WhenNotMatchedBySource};
        use crate::index::DatasetIndexExt;
        use arrow_array::{Int64Array, RecordBatchIterator};
        use lance_index::IndexType;
        use lance_index::scalar::ScalarIndexParams;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("shard_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ]));
        // ids 0..5, shard_key = id % 8 (the wallop composite PK), value = id*100.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..5)),
                Arc::new(Int64Array::from_iter_values((0..5).map(|i| i % 8))),
                Arc::new(Int64Array::from_iter_values((0..5).map(|i| i * 100))),
            ],
        )
        .unwrap();

        let uri = format!("shared-memory://phase1-delete-{}/", Uuid::new_v4().simple());
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            &uri,
            None,
        )
        .await
        .unwrap();
        // Scalar index on `id` only — `shard_key` is unindexed, mirroring the
        // WAL base table (BTree on `id`, composite PK `(id, shard_key)`).
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Phase-1 call: key-only source `(id=0, shard_key=0)`, delete-when-matched.
        let keys = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("shard_key", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![0_i64])),
                Arc::new(Int64Array::from(vec![0_i64])),
            ],
        )
        .unwrap();
        let mut builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset),
            vec!["id".to_string(), "shard_key".to_string()],
        )
        .unwrap();
        builder
            .when_matched(WhenMatched::Delete)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Keep)
            .use_index(true);
        let job = builder.try_build().unwrap();
        let keys_schema = keys.schema();
        let (dataset, _stats) = job
            .execute_reader(RecordBatchIterator::new([Ok(keys)], keys_schema))
            .await
            .unwrap();

        // id=0 must be gone — both from a full scan and from an indexed filter.
        let all = dataset.scan().try_into_batch().await.unwrap();
        let mut ids: Vec<i64> = all
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4],
            "Phase-1 must hard-delete (0, 0) from base"
        );

        let filtered = dataset
            .scan()
            .filter("id < 1")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            filtered.num_rows(),
            0,
            "indexed filter must not resurface the deleted composite key"
        );
    }

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, start_id: i32, num_rows: usize) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    start_id..start_id + num_rows as i32,
                )),
                Arc::new(StringArray::from_iter_values(
                    (0..num_rows).map(|i| format!("name_{}", start_id as usize + i)),
                )),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_shard_writer_basic_write() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            config.clone(),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Write a batch
        let batch = create_test_batch(&schema, 0, 10);
        let result = writer.put(vec![batch]).await.unwrap();

        assert_eq!(result.batch_positions, 0..1);

        // Check stats
        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);
        assert_eq!(stats.batch_count, 1);

        // Close writer
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_put_no_wait_durable_visible_then_durable() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let batch = create_test_batch(&schema, 0, 10);
        let (result, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
        assert_eq!(result.batch_positions, 0..1);

        // Row is visible in memory before durability is awaited.
        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);

        // durable_write is on, so a watcher is returned and resolves once the
        // triggered flush lands.
        let mut watcher = watcher.expect("durable_write returns a watcher");
        watcher.wait().await.unwrap();

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_put_no_wait_non_durable_returns_no_watcher() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let batch = create_test_batch(&schema, 0, 10);
        let (result, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
        assert_eq!(result.batch_positions, 0..1);
        assert!(watcher.is_none(), "non-durable put has nothing to await");

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_shard_writer_multiple_writes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        // Write multiple batches in a single put call
        let batches: Vec<_> = (0..5)
            .map(|i| create_test_batch(&schema, i * 10, 10))
            .collect();
        let result = writer.put(batches).await.unwrap();
        assert_eq!(result.batch_positions, 0..5);

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 50);
        assert_eq!(stats.batch_count, 5);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_shard_writer_with_indexes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: true,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            schema.clone(),
            index_configs,
        )
        .await
        .unwrap();

        // Write a batch
        let batch = create_test_batch(&schema, 0, 10);
        writer.put(vec![batch]).await.unwrap();

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);

        writer.close().await.unwrap();
    }

    /// End-to-end check that the background flush handler rebuilds secondary
    /// indexes on every flushed generation. Before this, the handler flushed
    /// via plain `flush`, leaving flushed generations unindexed — point
    /// lookups had to full-scan and vector search's index-only `fast_search`
    /// couldn't see the data at all.
    #[tokio::test]
    async fn test_flushed_generation_is_indexed() {
        use crate::index::DatasetIndexExt;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let config = ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: true,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            config,
            schema.clone(),
            index_configs,
        )
        .await
        .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();

        // Freeze the active memtable and wait until it lands on disk.
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // Resolve the flushed generation recorded in the manifest.
        let manifest = writer.manifest().await.unwrap().unwrap();
        assert_eq!(
            manifest.flushed_generations.len(),
            1,
            "expected exactly one flushed generation"
        );
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, shard_id, manifest.flushed_generations[0].path
        );

        // The flushed generation must carry the BTree index built during flush.
        let dataset = crate::Dataset::open(&gen_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "flushed generation should have one index");
        assert_eq!(indices[0].name, "id_idx");

        // A PK filter over it must resolve through the index, not a full scan.
        let mut scan = dataset.scan();
        scan.filter("id = 5").unwrap();
        scan.prefilter(true);
        let plan = scan.create_plan().await.unwrap();
        crate::utils::test::assert_plan_node_equals(
            plan,
            "LanceRead: ...full_filter=id = Int32(5)...
  ScalarIndexQuery: query=[id = 5]@id_idx(BTree)",
        )
        .await
        .unwrap();

        // And the index returns the correct row.
        let batch = dataset
            .scan()
            .filter("id = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);

        writer.close().await.unwrap();
    }

    /// Test memtable auto-flush triggered by size threshold.
    #[tokio::test]
    async fn test_shard_writer_auto_flush_by_size() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Use a small memtable size to trigger auto-flush
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 1024, // Very small - will trigger flush quickly
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        // Write batches until auto-flush triggers
        for i in 0..20 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        // Give time for background flush to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that generation increased (indicating flush happened)
        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation > initial_gen,
            "Generation should increment after auto-flush"
        );

        writer.close().await.unwrap();
    }

    /// Regression for #6713: a single failing `handle()` must not kill
    /// the dispatcher. Earlier the loop would `break Err(e)` on the
    /// first message error, dropping the rx side and stranding
    /// subsequent senders. The flusher tasks need to survive transient
    /// errors so the writer keeps making forward progress.
    #[tokio::test]
    async fn test_task_dispatcher_survives_handle_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FlakyHandler {
            call_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl MessageHandler<u32> for FlakyHandler {
            async fn handle(&mut self, message: u32) -> Result<()> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(Error::io("first message intentionally fails"))
                } else {
                    let _ = message;
                    Ok(())
                }
            }
        }

        let executor = TaskExecutor::new();
        let call_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::unbounded_channel::<u32>();
        executor
            .add_handler(
                "flaky".to_string(),
                Box::new(FlakyHandler {
                    call_count: call_count.clone(),
                }),
                rx,
            )
            .unwrap();

        // Send three messages: the first errors, the next two should
        // still be delivered to the (still-alive) handler.
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();

        for _ in 0..50 {
            if call_count.load(Ordering::SeqCst) >= 3 {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        }

        assert!(
            call_count.load(Ordering::SeqCst) >= 3,
            "dispatcher exited after the first error; only {} message(s) were handled",
            call_count.load(Ordering::SeqCst)
        );

        executor.shutdown_all().await.ok();
    }

    /// Same as the local-fs test but against memory:// — closer to S3
    /// semantics (conditional PUT, list-prefix consistency).
    #[tokio::test]
    async fn test_shard_writer_auto_flush_repeatedly_memory_store() {
        let base_uri = "memory:///bench_test_flush";
        let (store, base_path) = ObjectStore::from_uri(base_uri).await.unwrap();
        let base_uri = base_uri.to_string();
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        for i in 0..1000 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation >= initial_gen + 50,
            "expected many flushes; generation went {} → {}",
            initial_gen,
            stats.generation
        );

        writer.close().await.unwrap();
    }

    /// Regression for #6713: with durable_write=true and a memtable
    /// size threshold that fires frequently, the memtable flush task
    /// hit "Dataset already exists: …_gen_1" once the second flush
    /// started.
    #[tokio::test]
    async fn test_shard_writer_auto_flush_repeatedly_stress() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            // Tiny size threshold — every batch crosses it.
            max_memtable_size: 64,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        // Every put crosses the size threshold, so each one queues a
        // freeze. We want to catch any bug where two flushes collide on
        // path/generation. Drive 1000 puts so we get ≥ 100 flushes —
        // enough rope for the bug to show up.
        for i in 0..1000 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation >= initial_gen + 50,
            "expected many successful auto-flushes; generation went {} → {}",
            initial_gen,
            stats.generation
        );

        writer.close().await.unwrap();
    }

    /// Regression: `close()` must flush the active memtable (not just
    /// drain the WAL). Earlier, with a `max_memtable_size` set well
    /// above the workload, no auto-flush would fire and `close()`
    /// would return without producing a Lance fragment — data was
    /// durable in the WAL but no LSM-level generation existed,
    /// surprising callers and making flush-cost benchmarks impossible.
    ///
    /// The test verifies, end-to-end:
    /// 1. close() returns Ok (a freeze/flush error must propagate, not
    ///    be silently dropped).
    /// 2. The persisted shard manifest's `current_generation` has
    ///    advanced past the initial generation — direct evidence that
    ///    a MemTable flush + manifest commit happened during close()
    ///    rather than the active memtable being dropped on the floor.
    /// (Verifying replay's post-flush behavior is tangled with
    /// independent replay logic and is exercised by the dedicated
    /// `test_memtable_replay_*` tests.)
    #[tokio::test]
    async fn test_close_flushes_active_memtable() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Reuse the same shard_id when reopening so we observe state
        // produced by the first writer's `close()`.
        let shard_id = Uuid::new_v4();
        // Huge size threshold so puts never trigger an auto-flush.
        let config = ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: true,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: usize::MAX,
            max_unflushed_memtable_bytes: usize::MAX,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            config.clone(),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        for i in 0..50 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        // Pre-close sanity: no auto-flush should have fired.
        let stats_before = writer.memtable_stats().await.unwrap();
        assert_eq!(
            stats_before.generation, initial_gen,
            "no flush should have fired during puts (size threshold is usize::MAX)"
        );
        assert!(
            stats_before.row_count > 0,
            "memtable should hold the rows we just inserted"
        );

        // close() must succeed; any freeze/flush error must propagate.
        writer
            .close()
            .await
            .expect("close() must succeed and propagate any freeze/flush error");

        // Reopen the same shard and read the persisted manifest. The
        // active memtable from the closed writer was frozen + flushed
        // inside close(), which must have committed a new manifest
        // recording the advanced generation.
        let reopened =
            ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
                .await
                .unwrap();
        let manifest = reopened
            .manifest()
            .await
            .unwrap()
            .expect("reopened shard must have a persisted manifest");
        assert!(
            manifest.current_generation > initial_gen,
            "expected manifest current_generation to advance past {} after close() flushed the active memtable; got {}",
            initial_gen,
            manifest.current_generation,
        );

        reopened.close().await.unwrap();
    }

    /// Regression: the memtable flush should successfully fire many
    /// times in a row. A bug where every flush wrote the same path was
    /// caught by lance-format/lance#6713.
    #[tokio::test]
    async fn test_shard_writer_auto_flush_repeatedly() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // durable_write=true matches the LSM `merge_insert` defaults and
        // is the configuration that surfaced #6713 in the wild.
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            // Tiny size threshold so a few batches cross it.
            max_memtable_size: 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        // Drive enough write traffic to trigger several auto-flushes.
        // durable_write=true means each put waits for the WAL flush, so
        // we don't need explicit yields between puts.
        for i in 0..200 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        // Wait for the background memtable flushes to drain.
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Generation should have advanced by at least 3 — i.e. we want to
        // confirm multiple flushes succeeded back to back, not just one.
        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation >= initial_gen + 3,
            "expected ≥ 3 successful auto-flushes; generation went {} → {}",
            initial_gen,
            stats.generation
        );

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_no_backpressure_when_under_threshold() {
        let config = ShardWriterConfig::default().with_max_unflushed_memtable_bytes(1024 * 1024); // 1MB

        let controller = BackpressureController::new(config);

        // Should return immediately - well under threshold (100 bytes < 1MB)
        controller
            .maybe_apply_backpressure(|| (100, None))
            .await
            .unwrap();

        assert_eq!(controller.stats().count(), 0);
    }

    #[tokio::test]
    async fn test_backpressure_loops_until_under_threshold() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;

        let config = ShardWriterConfig::default()
            .with_max_unflushed_memtable_bytes(100) // Very low threshold
            .with_backpressure_log_interval(Duration::from_millis(50));

        let controller = BackpressureController::new(config);

        // Simulate: starts at 1000 bytes, drops by 400 each call (simulating memtable flushes)
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        controller
            .maybe_apply_backpressure(move || {
                let count = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // 1000 -> 600 -> 200 -> under threshold (need 3 iterations)
                let unflushed = 1000usize.saturating_sub(count * 400);
                (unflushed, None)
            })
            .await
            .unwrap();

        // Should have called get_state 4 times (initial + 3 waits until under 100)
        assert_eq!(call_count.load(std::sync::atomic::Ordering::Relaxed), 4);
        // Should have recorded backpressure wait time (waited 3 times)
        assert_eq!(controller.stats().count(), 1);
    }

    #[test]
    fn test_record_put() {
        let stats = WriteStats::new();
        stats.record_put(Duration::from_millis(10));
        stats.record_put(Duration::from_millis(20));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.put_count, 2);
        assert_eq!(snapshot.put_time, Duration::from_millis(30));
        assert_eq!(snapshot.avg_put_latency(), Some(Duration::from_millis(15)));
    }

    #[test]
    fn test_record_wal_flush() {
        let stats = WriteStats::new();
        stats.record_wal_flush(Duration::from_millis(100), 1024);
        stats.record_wal_flush(Duration::from_millis(200), 2048);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.wal_flush_count, 2);
        assert_eq!(snapshot.wal_flush_time, Duration::from_millis(300));
        assert_eq!(snapshot.wal_flush_bytes, 3072);
        assert_eq!(snapshot.avg_wal_flush_bytes(), Some(1536));
    }

    #[test]
    fn test_record_memtable_flush() {
        let stats = WriteStats::new();
        stats.record_memtable_flush(Duration::from_secs(1), 10000);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.memtable_flush_count, 1);
        assert_eq!(snapshot.memtable_flush_time, Duration::from_secs(1));
        assert_eq!(snapshot.memtable_flush_rows, 10000);
    }

    #[test]
    fn test_stats_reset() {
        let stats = WriteStats::new();
        stats.record_put(Duration::from_millis(10));
        stats.record_wal_flush(Duration::from_millis(100), 1024);

        stats.reset();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.put_count, 0);
        assert_eq!(snapshot.wal_flush_count, 0);
    }

    // ----- WAL-only mode (`enable_memtable = false`) -----

    fn wal_only_config(shard_id: Uuid) -> ShardWriterConfig {
        ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: true,
            enable_memtable: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            manifest_scan_batch_size: 2,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_wal_only_durable_round_trip() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let shard_id = Uuid::new_v4();
        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Two durable puts → two WAL entries (durable_write triggers an
        // explicit flush per put).
        let r1 = writer
            .put(vec![create_test_batch(&schema, 0, 4)])
            .await
            .unwrap();
        let r2 = writer
            .put(vec![create_test_batch(&schema, 100, 2)])
            .await
            .unwrap();
        assert_eq!(r1.batch_positions, 0..1);
        assert_eq!(r2.batch_positions, 1..2);

        writer.close().await.unwrap();

        // Read back via WalTailer. WAL positions are 1-based, so two
        // entries from a fresh shard land at 1 and 2.
        let tailer = WalTailer::new(store, base_path, shard_id);
        assert_eq!(tailer.first_position().await.unwrap(), 1);
        assert_eq!(tailer.next_position().await.unwrap(), 3);

        let e0 = tailer.read_entry(1).await.unwrap().unwrap();
        let e1 = tailer.read_entry(2).await.unwrap().unwrap();
        assert_eq!(e0.batches.len(), 1);
        assert_eq!(e0.batches[0].num_rows(), 4);
        assert_eq!(e1.batches.len(), 1);
        assert_eq!(e1.batches[0].num_rows(), 2);
        // Both entries from the same writer should have the same epoch.
        assert_eq!(e0.writer_epoch, e1.writer_epoch);
        assert!(e0.writer_epoch >= 1);
    }

    #[tokio::test]
    async fn test_wal_only_rejects_index_configs() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let err = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema,
            index_configs,
        )
        .await
        .err()
        .expect("expected invalid_input");
        assert!(
            err.to_string().contains("indexes require enable_memtable"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_wal_only_rejects_empty_batches() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Empty list.
        let err = writer.put(vec![]).await.err().unwrap();
        assert!(err.to_string().contains("empty batch list"));

        // Single empty batch.
        let zero = arrow_array::RecordBatch::new_empty(schema);
        let err = writer.put(vec![zero]).await.err().unwrap();
        assert!(err.to_string().contains("Batch 0 is empty"));

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_only_memtable_accessors_error() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema,
            vec![],
        )
        .await
        .unwrap();

        let err = writer.memtable_stats().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));
        let err = writer.scan().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));
        let err = writer.active_memtable_ref().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));
        let err = writer.in_memory_memtable_refs().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_only_async_batches_multiple_puts() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let mut config = wal_only_config(Uuid::new_v4());
        let shard_id = config.shard_id;
        // Non-durable: puts should accumulate in the pending queue until
        // close() drains them with a single WAL entry.
        config.durable_write = false;
        config.max_wal_flush_interval = None;
        config.max_wal_buffer_size = 100 * 1024 * 1024; // never crossed

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        for i in 0..3 {
            writer
                .put(vec![create_test_batch(&schema, i * 10, 10)])
                .await
                .unwrap();
        }

        writer.close().await.unwrap();

        // All three puts should be in a single WAL entry at position 1
        // (WAL positions are 1-based).
        let tailer = WalTailer::new(store, base_path, shard_id);
        assert_eq!(tailer.next_position().await.unwrap(), 2);
        let entry = tailer.read_entry(1).await.unwrap().unwrap();
        assert_eq!(entry.batches.len(), 3);
        for (i, batch) in entry.batches.iter().enumerate() {
            assert_eq!(batch.num_rows(), 10, "batch {i}");
        }
    }

    #[tokio::test]
    async fn test_wal_only_fencing() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer_a
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();

        // Writer B claims a higher epoch and writes — fences A.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        assert!(writer_b.epoch() > writer_a.epoch());
        writer_b
            .put(vec![create_test_batch(&schema, 1, 1)])
            .await
            .unwrap();

        // A's next durable put must fail with a fence error (the underlying
        // WalAppender::append surfaces it via atomic_put).
        let err = writer_a
            .put(vec![create_test_batch(&schema, 2, 1)])
            .await
            .expect_err("expected fence error");
        assert!(
            err.to_string().contains("Writer fenced"),
            "unexpected error: {err}"
        );

        writer_b.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_check_fenced_detects_successor_claim() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Not yet fenced.
        writer_a.check_fenced().await.unwrap();

        // Successor claims a higher epoch.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        assert!(writer_b.epoch() > writer_a.epoch());

        // A's check_fenced surfaces the fence without needing a put round-trip.
        let err = writer_a
            .check_fenced()
            .await
            .expect_err("expected fence error");
        assert!(
            err.to_string().contains("Writer fenced"),
            "unexpected error: {err}"
        );

        // B is the current writer and is not fenced.
        writer_b.check_fenced().await.unwrap();
        writer_b.close().await.unwrap();
    }

    // ----- MemTable replay on open -----

    fn memtable_config_with_pk(shard_id: Uuid) -> ShardWriterConfig {
        ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: true,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        }
    }

    fn schema_with_pk() -> Arc<ArrowSchema> {
        use arrow_schema::Field;
        // Mark `id` as the unenforced primary key.
        let pk_meta: std::collections::HashMap<String, String> = [(
            "lance-schema:unenforced-primary-key".to_string(),
            "1".to_string(),
        )]
        .into_iter()
        .collect();
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(pk_meta);
        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    /// Replay-on-open recovers durable WAL entries that were never flushed
    /// to a Lance generation. Setup: writer A durably writes batches, drops
    /// without close (so MemTable freeze never runs); writer B reopens and
    /// must see A's rows in its MemTable scan.
    #[tokio::test]
    async fn test_memtable_replay_recovers_unflushed_writes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Writer A: write two durable batches, then drop without close.
        // The WAL files persist; the in-memory MemTable does not.
        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                memtable_config_with_pk(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 0, 5)])
                .await
                .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 100, 3)])
                .await
                .unwrap();
            // intentionally drop without close()
        }

        // Writer B reopens. Replay must rehydrate A's two batches into the
        // active MemTable.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();

        let stats = writer_b.memtable_stats().await.unwrap();
        assert_eq!(
            stats.row_count, 8,
            "expected replay to insert 5 + 3 = 8 rows, got {}",
            stats.row_count
        );
        assert_eq!(
            stats.batch_count, 2,
            "expected replay to insert 2 batches, got {}",
            stats.batch_count
        );

        writer_b.close().await.unwrap();
    }

    /// Replay is a no-op on a fresh shard: the MemTable starts empty.
    #[tokio::test]
    async fn test_memtable_replay_no_op_on_fresh_shard() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();
        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 0);
        assert_eq!(stats.batch_count, 0);
        writer.close().await.unwrap();
    }

    /// Regression for the OSS-WAL compactor-drain bug: after a flush
    /// records its generation in the manifest and an external compactor
    /// later drains `flushed_generations` back to empty (the legitimate
    /// outcome of merging the generation into the base table), reopening
    /// the writer must not re-replay the already-flushed WAL entry into
    /// the active memtable.
    ///
    /// Under the pre-fix logic, replay disambiguated "fresh shard" from
    /// "flushed-then-compacted" with `flushed_generations.is_empty()`,
    /// which collapsed both cases into start-at-0. With 1-based WAL
    /// positions and a default cursor of 0 meaning "no flush stamped",
    /// the flush-then-drain sequence leaves `replay_after_wal_entry_position`
    /// pinned at the flushed position, so replay correctly starts past it.
    #[tokio::test]
    async fn test_memtable_replay_skips_entries_after_external_compaction() {
        use crate::dataset::mem_wal::ShardManifestStore;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Writer A: write 5 rows, close (forces a flush of the active
        // memtable). The manifest now records a flushed generation and
        // pins `replay_after_wal_entry_position` to the covered WAL entry.
        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                memtable_config_with_pk(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 0, 5)])
                .await
                .unwrap();
            writer_a.close().await.unwrap();
        }

        // Simulate an external compactor merging the flushed generation
        // into the base table: drain `flushed_generations` to empty via a
        // direct manifest commit. The cursor stays where the flush put it.
        let manifest_store = ShardManifestStore::new(store.clone(), &base_path, shard_id, 2);
        let pre = manifest_store.read_latest().await.unwrap().unwrap();
        assert!(
            !pre.flushed_generations.is_empty(),
            "writer A's close() should have stamped a flushed generation"
        );
        let cursor_at_flush = pre.replay_after_wal_entry_position;
        assert!(
            cursor_at_flush >= 1,
            "expected cursor to land on a 1-based WAL position after flush, got {cursor_at_flush}"
        );
        // Bump the epoch (claim_epoch) so we can commit_update without
        // being fenced; this also mirrors how a compactor process would
        // hold its own writer claim.
        let (compactor_epoch, _) = manifest_store.claim_epoch(pre.shard_spec_id).await.unwrap();
        manifest_store
            .commit_update(compactor_epoch, |current| ShardManifest {
                version: current.version + 1,
                flushed_generations: vec![],
                ..current.clone()
            })
            .await
            .unwrap();
        let post = manifest_store.read_latest().await.unwrap().unwrap();
        assert!(
            post.flushed_generations.is_empty(),
            "compactor drain should have left flushed_generations empty"
        );
        assert_eq!(
            post.replay_after_wal_entry_position, cursor_at_flush,
            "compactor must not touch the replay cursor"
        );

        // Writer B reopens. Pre-fix: replay saw flushed_generations empty,
        // restarted at WAL position 0, and re-inserted writer A's rows.
        // Post-fix: replay starts at cursor + 1, finds no entry, and the
        // memtable stays empty.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();
        let stats = writer_b.memtable_stats().await.unwrap();
        assert_eq!(
            stats.row_count, 0,
            "memtable must not re-replay compacted WAL entries; got {} rows",
            stats.row_count
        );
        assert_eq!(stats.batch_count, 0);
        writer_b.close().await.unwrap();
    }

    /// Replay aborts the open with a clear fence error if it encounters a
    /// WAL entry written with an epoch strictly greater than ours. Simulate
    /// the race where another writer wrote an entry with a higher epoch
    /// between our `claim_epoch` and our replay by injecting a high-epoch
    /// entry directly via `WalAppender::with_claimed_epoch` (which
    /// bypasses `claim_epoch` and so does not bump the manifest).
    #[tokio::test]
    async fn test_memtable_replay_fenced_aborts_open() {
        use crate::dataset::mem_wal::ShardManifestStore;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Writer A: write one durable batch (claims epoch 1, writes entry at position 1).
        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                memtable_config_with_pk(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 0, 1)])
                .await
                .unwrap();
            // drop without close
        }

        // Inject a WAL entry written with epoch 100 — far above whatever
        // claim_epoch will hand the next opener. The manifest is not
        // updated since we use `with_claimed_epoch` directly.
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let high_epoch_appender = WalAppender::with_claimed_epoch(
            store.clone(),
            base_path.clone(),
            shard_id,
            manifest_store,
            100,
            // hint seed irrelevant; the real position counter is discovered
            // lazily on the first append.
            0,
        );
        high_epoch_appender
            .append(vec![create_test_batch(&schema, 999, 1)])
            .await
            .unwrap();

        // Writer B opens. claim_epoch returns 2 (manifest's writer_epoch
        // was 1 before this open). Replay reads the injected entry, sees
        // epoch 100 > 2, and aborts with a fence error.
        let result = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await;
        let Err(err) = result else {
            panic!("expected open to fail with fence error during replay");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("WAL replay aborted") && msg.contains("fenced"),
            "unexpected error: {msg}"
        );
    }

    /// Regression: `wal_stats().next_wal_entry_position` must reflect the
    /// post-recovery cursor immediately on reopen, not 0 until the first
    /// append discovers the tip. Pre-fix the appender's hint was seeded at
    /// 0 and only updated after the first successful append, so external
    /// monitors saw 0 between open and first put on a shard with prior
    /// entries.
    #[tokio::test]
    async fn test_wal_stats_seeded_from_manifest_on_reopen() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        // First writer creates a shard, writes one entry, closes.
        let writer1 = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer1
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();
        writer1.close().await.unwrap();

        // Reopen: stats must reflect the post-recovery cursor immediately,
        // before any put has happened on this writer.
        let writer2 = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();
        let next = writer2.wal_stats().next_wal_entry_position;
        assert!(
            next >= 1,
            "expected wal_stats to reflect post-recovery cursor (>= 1) on reopen, got {next}"
        );

        writer2.close().await.unwrap();
    }

    /// Regression test for the size-based trigger after a drain.
    ///
    /// Earlier the WAL-only size trigger used a monotonic counter
    /// (`wal_flush_trigger_count`) which never reset across drains. After
    /// the first crossing the counter was >= 1 and `pending_bytes / threshold`
    /// could never grow past 1 (because pending_bytes resets on drain), so
    /// the size trigger silently stopped firing. This test pushes batches
    /// to cross the size threshold multiple times across drains and asserts
    /// every crossing produces a WAL entry.
    #[tokio::test]
    async fn test_wal_only_size_trigger_fires_repeatedly() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let mut config = wal_only_config(Uuid::new_v4());
        let shard_id = config.shard_id;
        // Non-durable so puts don't auto-flush. Time trigger off so only
        // the size trigger drives flushes.
        config.durable_write = false;
        config.max_wal_flush_interval = None;
        // Pick a tiny threshold so a single batch crosses it.
        config.max_wal_buffer_size = 1;

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Three puts, each large enough to cross the (1-byte) threshold.
        // Without the fix, only the first would trigger; the rest would
        // sit in the queue until close().
        for i in 0..3 {
            writer
                .put(vec![create_test_batch(&schema, i * 10, 10)])
                .await
                .unwrap();
            // Yield, then sleep, so the background flush handler can
            // drain the trigger queue before the next push — otherwise
            // multiple pending triggers can coalesce into a single drain.
            // 50ms historically failed on slow Windows CI runners; 250ms
            // gives a comfortable margin without making the suite slow.
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }

        writer.close().await.unwrap();

        // Each put should have produced its own WAL entry — three crossings,
        // three entries. Without the regression fix, all three batches end
        // up in a single entry written by `close()`.
        let tailer = WalTailer::new(store, base_path, shard_id);
        let next = tailer.next_position().await.unwrap();
        assert!(
            next >= 3,
            "expected at least 3 WAL entries (one per crossing), got next_position = {next}"
        );
    }

    /// Regression test for concurrent durable WAL-only puts on a fenced
    /// writer. Earlier `flush_from_wal_only` did a destructive `drain()`
    /// before calling `wal_appender.append`. If the append failed (e.g.
    /// fence), the drained batches were dropped — the next concurrent put
    /// would then see an empty pending queue and spuriously return Ok,
    /// hiding the data loss. With the snapshot/commit fix, the failed flush
    /// leaves the batches in the queue, and the concurrent put gets a clean
    /// fence error too (when its own flush attempts the same WAL position).
    #[tokio::test]
    async fn test_wal_only_fenced_concurrent_puts_do_not_silently_succeed() {
        use std::sync::Arc;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        // Writer A claims epoch 1, writes one entry (takes WAL position 1,
        // caches its next-position as 2 internally).
        let writer_a = Arc::new(
            ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                wal_only_config(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap(),
        );
        writer_a
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();

        // Writer B claims epoch 2 and writes its own entry (takes WAL
        // position 1). A is now fenced: A's next put will attempt WAL
        // position 1 (its cached next), collide with B's entry, and
        // surface a "Writer fenced" error from `check_fenced`.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer_b
            .put(vec![create_test_batch(&schema, 1, 1)])
            .await
            .unwrap();

        // Two concurrent durable puts on the (now-fenced) writer A. With
        // the destructive-drain bug, the first flush would consume both
        // pending batches into a failing append; the second flush would
        // see an empty queue and return spurious success, silently losing
        // the second put's data. With the snapshot/commit fix, the failed
        // append leaves both batches in the queue and the second flush
        // also fails with the fence error.
        let a1 = writer_a.clone();
        let a2 = writer_a.clone();
        let schema1 = schema.clone();
        let schema2 = schema.clone();
        let h1 = tokio::spawn(async move { a1.put(vec![create_test_batch(&schema1, 2, 1)]).await });
        let h2 = tokio::spawn(async move { a2.put(vec![create_test_batch(&schema2, 3, 1)]).await });

        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();

        assert!(
            r1.is_err() && r2.is_err(),
            "expected both concurrent puts on a fenced writer to fail, got r1={r1:?} r2={r2:?}",
        );

        writer_b.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_only_stats_no_memtable_flush() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();

        let stats_handle = writer.stats_handle();
        writer.close().await.unwrap();

        let snapshot = stats_handle.snapshot();
        assert!(snapshot.put_count >= 1, "expected at least one put");
        assert!(
            snapshot.wal_flush_count >= 1,
            "expected at least one WAL flush"
        );
        assert_eq!(
            snapshot.memtable_flush_count, 0,
            "WAL-only mode must never trigger a memtable flush"
        );
        assert_eq!(
            snapshot.index_update_count, 0,
            "WAL-only mode must never trigger an index update"
        );
    }

    #[tokio::test]
    async fn test_force_seal_active_and_wait_for_flush_drain() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Thresholds high enough that auto-flush won't fire; the seal is
        // the only thing that should rotate the memtable.
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 64 * 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;
        let flushed_before = writer
            .manifest()
            .await
            .unwrap()
            .map(|m| m.flushed_generations.len())
            .unwrap_or(0);

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.generation, initial_gen + 1);
        assert_eq!(stats.batch_count, 0);

        let manifest = writer
            .manifest()
            .await
            .unwrap()
            .expect("manifest should exist after flush");
        assert_eq!(manifest.flushed_generations.len(), flushed_before + 1);

        writer.close().await.unwrap();
    }

    /// On a successful flush commit the sealed generation's rows land in the
    /// manifest immediately, but the in-memory handle is NOT dropped — it
    /// lingers for `frozen_memtable_grace` (so in-flight as-of reads keep
    /// batch-resolved membership), then is swept by the `SweepExpired` ticker.
    #[tokio::test]
    async fn test_frozen_retained_during_grace_then_swept() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 64 * 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            // Short grace so the sweep is observable without a slow test.
            frozen_memtable_grace: Duration::from_secs(1),
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;
        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // Recorded in the manifest at commit time.
        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert!(
            manifest
                .flushed_generations
                .iter()
                .any(|g| g.generation == initial_gen),
            "flushed generation must be recorded in the manifest"
        );

        // Still queryable in memory immediately after commit (within grace).
        let refs = writer.in_memory_memtable_refs().await.unwrap();
        assert_eq!(refs.active.generation, initial_gen + 1);
        assert!(
            refs.frozen.iter().any(|f| f.generation == initial_gen),
            "flushed generation must stay queryable during the grace window"
        );

        // After the grace elapses (plus a sweep tick) the handle is evicted.
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        let refs = writer.in_memory_memtable_refs().await.unwrap();
        assert!(
            refs.frozen.is_empty(),
            "frozen handle must be swept once the grace elapses"
        );

        writer.close().await.unwrap();
    }

    /// With zero grace (the default) a frozen handle is evicted synchronously on
    /// flush commit — no sweep tick, no lingering window.
    #[tokio::test]
    async fn test_frozen_evicted_immediately_with_zero_grace() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 64 * 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            frozen_memtable_grace: Duration::ZERO,
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;
        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // Rows are durably in the manifest...
        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert!(
            manifest
                .flushed_generations
                .iter()
                .any(|g| g.generation == initial_gen),
            "flushed generation must be recorded in the manifest"
        );

        // ...and the in-memory handle is already gone, no sweep tick needed.
        let refs = writer.in_memory_memtable_refs().await.unwrap();
        assert!(
            refs.frozen.is_empty(),
            "frozen handle must be evicted on commit when grace is zero"
        );

        writer.close().await.unwrap();
    }

    /// Regression: a transient flush failure must NOT reopen the
    /// concurrent-read-vs-flush hole. The sealed generation stays in the
    /// queryable set (rows intact) until a later flush or WAL replay.
    /// Failure is induced deterministically by fencing the writer with a
    /// successor before the seal, so the flush's `check_fenced` rejects it.
    #[tokio::test]
    async fn test_frozen_retained_after_failed_flush() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        let initial_gen = writer_a.memtable_stats().await.unwrap().generation;
        writer_a
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();

        // Successor claims a higher epoch, fencing A.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        assert!(writer_b.epoch() > writer_a.epoch());

        // `force_seal_active` would reject up-front on a fenced writer;
        // freeze directly so the failure surfaces at flush-commit time —
        // exactly the freeze/flush race the fix guards.
        match &writer_a.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                let mut st = state.write().await;
                writer_state.freeze_memtable(&mut st).unwrap();
            }
            WriterMode::WalOnly { .. } => unreachable!("opened in memtable mode"),
        }

        // The fenced flush fails; the drain surfaces that error.
        assert!(
            writer_a.wait_for_flush_drain().await.is_err(),
            "fenced flush should fail the drain"
        );

        // The hole did not reopen: the sealed generation is still queryable
        // with its rows, alongside the new (empty) active generation.
        let refs = writer_a.in_memory_memtable_refs().await.unwrap();
        assert_eq!(refs.frozen.len(), 1, "sealed generation must be retained");
        assert_eq!(refs.frozen[0].generation, initial_gen);
        assert!(
            !refs.frozen[0].batch_store.is_empty(),
            "retained sealed memtable must still hold its rows"
        );
        assert_eq!(refs.active.generation, initial_gen + 1);

        writer_b.close().await.unwrap();
    }
}

#[cfg(test)]
mod shard_writer_tests {
    use std::sync::Arc;

    use crate::index::DatasetIndexExt;
    use arrow_array::{
        FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;
    use lance_index::scalar::inverted::InvertedIndexParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::builder::PQBuildParams;
    use lance_linalg::distance::MetricType;
    use uuid::Uuid;

    use crate::dataset::mem_wal::DatasetMemWalExt;
    use crate::dataset::{Dataset, WriteParams};
    use crate::index::vector::VectorIndexParams;

    use super::super::ShardWriterConfig;

    fn create_test_schema(vector_dim: i32) -> Arc<ArrowSchema> {
        use std::collections::HashMap;

        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int64, false).with_metadata(id_metadata);

        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vector_dim,
                ),
                true,
            ),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn create_append_only_schema(vector_dim: i32) -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vector_dim,
                ),
                true,
            ),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(
        schema: &ArrowSchema,
        start_id: i64,
        num_rows: usize,
        vector_dim: i32,
    ) -> RecordBatch {
        let vectors: Vec<f32> = (0..num_rows)
            .flat_map(|i| {
                let seed = (start_id as usize + i) as f32;
                (0..vector_dim as usize).map(move |d| (seed * 0.1 + d as f32 * 0.01).sin())
            })
            .collect();

        let vector_array =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vectors), vector_dim)
                .unwrap();

        let texts: Vec<String> = (0..num_rows)
            .map(|i| format!("Sample text for row {}", start_id as usize + i))
            .collect();

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    start_id..start_id + num_rows as i64,
                )),
                Arc::new(vector_array),
                Arc::new(StringArray::from_iter_values(texts)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_records_writer_config_defaults() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_writer_config_defaults_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        let writer_config = ShardWriterConfig::default()
            .with_durable_write(false)
            .with_max_memtable_size(8 * 1024 * 1024);

        dataset
            .initialize_mem_wal()
            .writer_config_defaults(writer_config)
            .add_writer_config_default("custom_knob", "custom_value")
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Defaults must survive the manifest round-trip so all writers share them.
        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        let defaults = &details.writer_config_defaults;
        // ShardWriterConfig tunables are recorded under their field names.
        assert_eq!(
            defaults.get("durable_write").map(String::as_str),
            Some("false")
        );
        assert_eq!(
            defaults.get("max_memtable_size").map(String::as_str),
            Some("8388608")
        );
        // Duration knobs are recorded in milliseconds with a `_ms` suffix.
        assert_eq!(
            defaults
                .get("max_wal_flush_interval_ms")
                .map(String::as_str),
            Some("100")
        );
        // Every tunable field is present.
        assert!(defaults.contains_key("sync_indexed_write"));
        assert!(defaults.contains_key("enable_memtable"));
        assert!(defaults.contains_key("async_index_interval_ms"));
        // add_writer_config_default records arbitrary keys.
        assert_eq!(
            defaults.get("custom_knob").map(String::as_str),
            Some("custom_value")
        );
        // Shard identity is not a configuration default.
        assert!(!defaults.contains_key("shard_id"));
        assert!(!defaults.contains_key("shard_spec_id"));
    }

    /// A maintained index can be split across multiple physical segments once a
    /// delta is appended over previously uncovered fragments (the distributed
    /// indexer / `optimize_indices(append)` flow). `mem_wal_writer` must resolve
    /// such an index by name without tripping the singular loader's "multiple
    /// indices of the same name" error — it only reads the shared type/params,
    /// which every segment carries identically.
    #[tokio::test]
    async fn test_mem_wal_writer_with_multi_segment_index() {
        use lance_index::optimize::OptimizeOptions;

        let vector_dim = 32;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_multi_segment_index_{}", Uuid::new_v4());

        // Initial fragment + an IVF vector index covering it.
        let initial = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");
        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create vector index");

        // Append a second fragment and index it as a *delta* (no merge), so the
        // index ends up with two physical segments sharing the name "vector_idx".
        let appended = create_test_batch(&schema, 256, 256, vector_dim);
        let append_batches = RecordBatchIterator::new([Ok(appended)], schema.clone());
        dataset
            .append(append_batches, None)
            .await
            .expect("Failed to append fragment");
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .expect("Failed to append index delta");

        // Precondition: the index is genuinely multi-segment, so the singular
        // `load_index_by_name` would error here.
        assert_eq!(
            dataset
                .load_indices_by_name("vector_idx")
                .await
                .unwrap()
                .len(),
            2,
            "expected two physical segments for the maintained index"
        );

        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // The regression: loading the multi-segment maintained index must succeed.
        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(
                shard_id,
                ShardWriterConfig::new(shard_id).with_durable_write(false),
            )
            .await
            .expect("mem_wal_writer must accept a multi-segment maintained index");

        // And the resulting writer is functional.
        writer
            .put(vec![create_test_batch(&schema, 200, 10, vector_dim)])
            .await
            .unwrap();
        assert_eq!(writer.memtable_stats().await.unwrap().row_count, 10);
        writer.close().await.unwrap();
    }

    /// A generation of only tombstones (delete nulls the vector, and the HNSW
    /// skips nulls) has an empty vector index. Flushing it must succeed —
    /// writing the data with no HNSW index for that generation — not fail the
    /// drain with "HnswMemIndex is empty". Regression for the wallop fuzz
    /// `flush drain failed: HnswMemIndex is empty` on a delete-only memtable.
    #[tokio::test]
    async fn test_flush_all_tombstone_generation_skips_empty_hnsw() {
        let vector_dim = 32;
        let schema = create_test_schema(vector_dim);
        // `shared-memory` (not `memory`): the flush writes the generation dataset
        // and the drain reopens it by URI, which a plain in-memory store wouldn't
        // share across opens. A unique authority isolates this test.
        let uri = format!("shared-memory://tombstone-{}/", Uuid::new_v4().simple());

        // Base dataset + maintained vector index, so the writer carries an HNSW
        // mem index.
        let initial = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &VectorIndexParams::ivf_flat(1, MetricType::L2),
                true,
            )
            .await
            .unwrap();
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .execute()
            .await
            .unwrap();

        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(
                shard_id,
                ShardWriterConfig::new(shard_id).with_durable_write(false),
            )
            .await
            .unwrap();

        // Delete only — the memtable holds a single tombstone with a null vector,
        // so its HNSW index is empty.
        let keys = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![0_i64]))],
        )
        .unwrap();
        writer.delete(vec![keys]).await.unwrap();

        // The drain must not error on the empty HNSW.
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // The tombstone-only generation still flushed (data without an HNSW index).
        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert_eq!(
            manifest.flushed_generations.len(),
            1,
            "the all-tombstone generation must still flush"
        );
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_writer_hnsw_params_override() {
        use lance_index::vector::hnsw::builder::HnswBuildParams;

        let vector_dim = 32;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_writer_hnsw_params_{}", Uuid::new_v4());

        let initial = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");
        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create vector index");

        // Persisting a ShardWriterConfig that carries HNSW params records them
        // in the writer-config defaults map under `hnsw.<index>.<field>` keys.
        let configured = ShardWriterConfig::new(Uuid::new_v4()).with_hnsw_params(
            "vector_idx",
            HnswBuildParams::default().num_edges(7).ef_construction(48),
        );
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .writer_config_defaults(configured)
            .execute()
            .await
            .expect("Failed to initialize MemWAL");
        let defaults = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .expect("MemWAL details should exist")
            .writer_config_defaults;
        assert_eq!(
            defaults
                .get("hnsw.vector_idx.num_edges")
                .map(String::as_str),
            Some("7")
        );
        assert_eq!(
            defaults
                .get("hnsw.vector_idx.ef_construction")
                .map(String::as_str),
            Some("48")
        );

        // The writer's per-index HNSW override flows into the in-memory index.
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_durable_write(false)
            .with_hnsw_params(
                "vector_idx",
                HnswBuildParams::default().num_edges(7).ef_construction(48),
            );
        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("mem_wal_writer must accept the HNSW-param override");
        writer
            .put(vec![create_test_batch(&schema, 300, 10, vector_dim)])
            .await
            .unwrap();

        let active = writer.active_memtable_ref().await.unwrap();
        let hnsw = active
            .index_store
            .get_hnsw("vector_idx")
            .expect("maintained HNSW index should exist in the memtable");
        assert_eq!(hnsw.build_params().m, 7);
        assert_eq!(hnsw.build_params().ef_construction, 48);
        drop(active);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_bucket_sharding() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_bucket_sharding_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // num_buckets out of range is rejected.
        let result = dataset
            .initialize_mem_wal()
            .bucket_sharding("id", 0)
            .execute()
            .await;
        assert!(result.is_err(), "num_buckets = 0 should be rejected");

        dataset
            .initialize_mem_wal()
            .bucket_sharding("text", 8)
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        assert_eq!(details.num_shards, 8);
        assert_eq!(details.sharding_specs.len(), 1);
        let field = &details.sharding_specs[0].fields[0];
        assert_eq!(field.transform.as_deref(), Some("bucket"));
        assert_eq!(
            field.parameters.get("num_buckets").map(String::as_str),
            Some("8")
        );
        assert_eq!(field.source_ids.len(), 1);
        let source_id = field.source_ids[0];
        let source_field = dataset.schema().field("text").expect("text field exists");
        assert_eq!(source_id, source_field.id);
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_bucket_sharding_without_primary_key() {
        let vector_dim = 128;
        let schema = create_append_only_schema(vector_dim);
        let uri = format!(
            "memory://test_bucket_sharding_no_primary_key_{}",
            Uuid::new_v4()
        );

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        dataset
            .initialize_mem_wal()
            .bucket_sharding("id", 8)
            .execute()
            .await
            .expect("Failed to initialize append-only MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        assert_eq!(details.num_shards, 8);
        assert_eq!(details.sharding_specs.len(), 1);
        let field = &details.sharding_specs[0].fields[0];
        assert_eq!(field.transform.as_deref(), Some("bucket"));
        assert_eq!(
            field.parameters.get("num_buckets").map(String::as_str),
            Some("8")
        );
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_unsharded() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_unsharded_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        assert_eq!(details.num_shards, 1);
        assert_eq!(details.sharding_specs.len(), 1);
        assert_eq!(
            details.sharding_specs[0].fields[0].transform.as_deref(),
            Some("unsharded")
        );
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_identity_sharding() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_identity_sharding_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // A column that does not exist is rejected.
        let result = dataset
            .initialize_mem_wal()
            .identity_sharding("nonexistent")
            .execute()
            .await;
        assert!(
            result.is_err(),
            "an unknown identity column should be rejected"
        );

        // A non-scalar column cannot be a shard key.
        let result = dataset
            .initialize_mem_wal()
            .identity_sharding("vector")
            .execute()
            .await;
        assert!(
            result.is_err(),
            "a non-scalar identity column should be rejected"
        );

        dataset
            .initialize_mem_wal()
            .identity_sharding("text")
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        // Identity sharding has an open-ended shard count.
        assert_eq!(details.num_shards, 0);
        assert_eq!(details.sharding_specs.len(), 1);
        let field = &details.sharding_specs[0].fields[0];
        assert_eq!(field.transform.as_deref(), Some("identity"));
        assert_eq!(field.result_type.as_str(), "utf8");
        assert_eq!(field.source_ids.len(), 1);
    }

    /// Quick smoke test for shard writer - runs against memory://
    /// Run with: cargo test -p lance shard_writer_tests::test_shard_writer_smoke -- --nocapture
    #[tokio::test]
    async fn test_shard_writer_smoke() {
        let vector_dim = 128;
        let batch_size = 20;
        let num_batches = 100;

        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_shard_writer_{}", Uuid::new_v4());

        // Create initial dataset
        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Initialize MemWAL (no indexes for smoke test)
        dataset
            .initialize_mem_wal()
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Create shard writer
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_durable_write(false)
            .with_sync_indexed_write(false);

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        // Pre-generate batches
        let batches: Vec<RecordBatch> = (0..num_batches)
            .map(|i| create_test_batch(&schema, (i * batch_size) as i64, batch_size, vector_dim))
            .collect();

        // Write all batches in a single put call for efficiency
        writer.put(batches).await.expect("Failed to write");

        writer.close().await.expect("Failed to close");
    }

    #[tokio::test]
    async fn test_shard_writer_with_vector_index_searches_active_memtable() {
        let vector_dim = 32;
        let batch_size = 20;
        let target_id = 1_000i64 + 37;

        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_shard_writer_hnsw_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create base vector index");

        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_durable_write(true)
            .with_sync_indexed_write(true);

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        let batches: Vec<RecordBatch> = (0..4)
            .map(|i| {
                create_test_batch(
                    &schema,
                    1_000 + (i * batch_size) as i64,
                    batch_size,
                    vector_dim,
                )
            })
            .collect();
        writer.put(batches).await.expect("Failed to write");

        let query = Float32Array::from_iter_values(
            (0..vector_dim as usize).map(|d| (target_id as f32 * 0.1 + d as f32 * 0.01).sin()),
        );
        let mut scanner = writer.scan().await.unwrap();
        scanner.nearest("vector", Arc::new(query), 80);
        let result = scanner.try_into_batch().await.expect("Failed to scan");

        assert!(result.num_rows() > 0, "vector query returned no rows");
        let id_col = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let dist_col = result
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let target_idx = (0..result.num_rows())
            .find(|&idx| id_col.value(idx) == target_id)
            .expect("target vector was not returned by active MemTable HNSW search");
        assert!(
            dist_col.value(target_idx) < 1e-6,
            "expected self-match distance near zero, got {}",
            dist_col.value(target_idx)
        );

        writer.close().await.expect("Failed to close");
    }

    /// Test shard writer against S3 with IVF-PQ, BTree, and FTS indexes (requires DATASET_PREFIX env var)
    /// Run with: DATASET_PREFIX=s3://bucket/path cargo test -p lance --release shard_writer_tests::test_shard_writer_s3_ivfpq -- --nocapture --ignored
    #[tokio::test]
    #[ignore]
    async fn test_shard_writer_s3_ivfpq() {
        let prefix = std::env::var("DATASET_PREFIX").expect("DATASET_PREFIX not set");

        let vector_dim = 512;
        let batch_size = 20;
        let num_batches = 10000;
        let num_partitions = 16;
        let num_sub_vectors = 64; // 512 / 8 = 64 subvectors

        let schema = create_test_schema(vector_dim);
        let uri = format!(
            "{}/test_s3_{}",
            prefix.trim_end_matches('/'),
            Uuid::new_v4()
        );

        // Create initial dataset with enough data for IVF-PQ training
        let initial_batch = create_test_batch(&schema, 0, 1000, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Create BTree index on id column
        let scalar_params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &scalar_params,
                false,
            )
            .await
            .expect("Failed to create BTree index");

        // Create FTS index on text column
        let fts_params = InvertedIndexParams::default();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &fts_params,
                false,
            )
            .await
            .expect("Failed to create FTS index");

        // Create IVF-PQ index on dataset

        let ivf_params = IvfBuildParams {
            num_partitions: Some(num_partitions),
            ..Default::default()
        };
        let pq_params = PQBuildParams {
            num_sub_vectors,
            num_bits: 8,
            ..Default::default()
        };
        let vector_params =
            VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create IVF-PQ index");

        // Initialize MemWAL with all three indexes
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["id_btree", "text_fts", "vector_idx"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Create shard writer with default config
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_durable_write(false)
            .with_sync_indexed_write(false);

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        // Pre-generate batches
        let batches: Vec<RecordBatch> = (0..num_batches)
            .map(|i| create_test_batch(&schema, (i * batch_size) as i64, batch_size, vector_dim))
            .collect();

        // Write all batches in a single put call for efficiency
        writer.put(batches).await.expect("Failed to write");

        writer.close().await.expect("Failed to close");
    }

    /// End-to-end correctness test for ShardWriter with multiple memtable flushes.
    ///
    /// This test verifies:
    /// 1. Multiple memtable flushes are triggered via small memtable size
    /// 2. File system layout is correct (WAL files, manifest, generation directories)
    /// 3. WAL entries contain expected data
    /// 4. Data can be read after each flush cycle
    /// 5. Manifest tracks flushed generations correctly
    ///
    /// Run with: cargo test -p lance shard_writer_tests::test_shard_writer_e2e_correctness -- --nocapture
    #[tokio::test]
    async fn test_shard_writer_e2e_correctness() {
        use std::time::Duration;
        use tempfile::TempDir;

        let vector_dim = 32;
        let rows_per_batch = 50;
        // Write enough to trigger ~3 memtable flushes with 50KB memtable size
        // Each batch is ~6KB (50 rows * 32 dims * 4 bytes/float + overhead)
        let num_write_rounds = 3;
        let batches_per_round = 3;

        // Create temp directory for the test
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let uri = format!("file://{}", temp_dir.path().display());

        let schema = create_test_schema(vector_dim);

        // Create initial dataset with enough rows for IVF-PQ training
        let initial_batch = create_test_batch(&schema, 0, 500, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Create BTree index
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .expect("Failed to create BTree index");

        // Initialize MemWAL with BTree index only (simpler for this test)
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["id_btree"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Create shard writer with small memtable size to trigger flushes
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_durable_write(true) // Ensure WAL files are written
            .with_sync_indexed_write(true)
            .with_max_memtable_size(50 * 1024) // 50KB - triggers flush after ~8 batches
            .with_max_wal_buffer_size(10 * 1024) // 10KB WAL buffer
            .with_max_wal_flush_interval(Duration::from_millis(50)); // Fast flush

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        let mut total_rows_written = 0i64;

        // Write data in rounds
        for _round in 0..num_write_rounds {
            let start_id = 500 + total_rows_written;
            let batches_to_write: Vec<RecordBatch> = (0..batches_per_round)
                .map(|i| {
                    create_test_batch(
                        &schema,
                        start_id + (i * rows_per_batch) as i64,
                        rows_per_batch,
                        vector_dim,
                    )
                })
                .collect();

            writer.put(batches_to_write).await.expect("Failed to write");

            total_rows_written += (batches_per_round * rows_per_batch) as i64;

            // Give time for WAL flush and potential memtable flush
            tokio::time::sleep(Duration::from_millis(150)).await;
        }

        // Close writer to ensure final flush
        writer.close().await.expect("Failed to close");

        // === VERIFY FILE SYSTEM LAYOUT ===
        let mem_wal_dir = temp_dir.path().join("_mem_wal").join(shard_id.to_string());
        assert!(mem_wal_dir.exists(), "MemWAL directory should exist");

        // Check WAL directory
        let wal_dir = mem_wal_dir.join("wal");
        assert!(wal_dir.exists(), "WAL directory should exist");
        let wal_files: Vec<_> = std::fs::read_dir(&wal_dir)
            .expect("Failed to read WAL dir")
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            !wal_files.is_empty(),
            "WAL directory should contain at least one file"
        );

        // Check manifest directory
        let manifest_dir = mem_wal_dir.join("manifest");
        assert!(manifest_dir.exists(), "Manifest directory should exist");
        let manifest_files: Vec<_> = std::fs::read_dir(&manifest_dir)
            .expect("Failed to read manifest dir")
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            !manifest_files.is_empty(),
            "Manifest directory should contain at least one file"
        );

        // Read and verify manifest
        let (store, base_path) = lance_io::object_store::ObjectStore::from_uri(&uri)
            .await
            .expect("Failed to open store");
        let manifest_store =
            super::super::manifest::ShardManifestStore::new(store, &base_path, shard_id, 2);
        let manifest = manifest_store
            .read_latest()
            .await
            .expect("Failed to read manifest")
            .expect("Manifest should exist");

        // Verify flushed generations exist on disk
        assert!(
            !manifest.flushed_generations.is_empty(),
            "Should have at least one flushed generation"
        );
        for flushed_gen in &manifest.flushed_generations {
            // The path stored in manifest is relative to the shard directory
            // Construct full path: temp_dir/_mem_wal/shard_id/generation_folder
            let gen_path = temp_dir
                .path()
                .join("_mem_wal")
                .join(shard_id.to_string())
                .join(&flushed_gen.path);

            // The generation directory should exist
            assert!(
                gen_path.exists(),
                "Flushed generation directory should exist at {:?}",
                gen_path
            );

            // Verify generation directory has files
            let gen_contents_count = std::fs::read_dir(&gen_path)
                .expect("Failed to read gen dir")
                .filter_map(|e| e.ok())
                .count();
            assert!(
                gen_contents_count > 0,
                "Generation directory should have files"
            );
        }

        // === VERIFY WAL ENTRIES ===
        // Verify WAL files have correct extension
        for wal_file in wal_files.iter().take(1) {
            let wal_path = wal_file.path();
            let file_name = wal_path.file_name().unwrap().to_string_lossy();
            assert!(
                file_name.ends_with(".arrow"),
                "WAL file should have .arrow extension"
            );
        }

        // === VERIFY DATA CAN BE READ FROM NEW WRITER ===
        // Re-open dataset and create new writer to verify recovery
        let dataset = Dataset::open(&uri).await.expect("Failed to reopen dataset");
        let new_shard_id = Uuid::new_v4();
        let new_config = ShardWriterConfig::new(new_shard_id)
            .with_durable_write(false)
            .with_sync_indexed_write(true);

        let new_writer = dataset
            .mem_wal_writer(new_shard_id, new_config)
            .await
            .expect("Failed to create new writer");

        // Write a test batch to verify the new shard works
        let verify_batch = create_test_batch(&schema, 10000, 10, vector_dim);
        new_writer
            .put(vec![verify_batch])
            .await
            .expect("Failed to write to new shard");

        let scanner = new_writer.scan().await.unwrap();
        let result = scanner.try_into_batch().await.expect("Failed to scan");
        assert_eq!(result.num_rows(), 10, "New shard should have 10 rows");

        new_writer
            .close()
            .await
            .expect("Failed to close new writer");
    }
}
