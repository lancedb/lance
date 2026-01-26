// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stderr)]

//! Write path for MemWAL.
//!
//! This module contains all components for the write path:
//! - [`RegionWriter`] - Main writer interface for a single region
//! - [`MemTable`] - In-memory table storing Arrow RecordBatches
//! - [`WalFlusher`] - Write-ahead log buffer for durability (Arrow IPC format)
//! - [`IndexStore`] - In-memory index management
//! - [`MemTableFlusher`] - Flush MemTable to storage as single Lance file

use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, Instant};

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_index::mem_wal::RegionManifest;
use lance_io::object_store::ObjectStore;
use log::{debug, error, info, warn};
use object_store::path::Path;
use snafu::location;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{interval_at, Interval};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub use super::index::{
    BTreeIndexConfig, BTreeMemIndex, FtsIndexConfig, IndexStore, IvfPqIndexConfig, MemIndexConfig,
};
pub use super::memtable::batch_store::{BatchStore, StoreFull, StoredBatch};
pub use super::memtable::flush::MemTableFlusher;
pub use super::memtable::scanner::MemTableScanner;
pub use super::memtable::CacheConfig;
pub use super::memtable::MemTable;
pub use super::util::{WatchableOnceCell, WatchableOnceCellReader};
pub use super::wal::{WalEntry, WalEntryData, WalFlushResult, WalFlusher};

use super::memtable::flush::TriggerMemTableFlush;
use super::wal::TriggerWalFlush;

use super::manifest::RegionManifestStore;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a region writer.
#[derive(Debug, Clone)]
pub struct RegionWriterConfig {
    /// Unique identifier for this region (UUID v4).
    pub region_id: Uuid,

    /// Region spec ID this region was created with.
    /// A value of 0 indicates a manually-created region not governed by any spec.
    pub region_spec_id: u32,

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
    /// Used to pre-allocate index storage (e.g., IVF-PQ partition capacity).
    /// When a partition reaches capacity, memtable will be flushed.
    /// Default: 100,000 rows
    pub max_memtable_rows: usize,

    /// Maximum number of batches in a MemTable.
    ///
    /// Used to pre-allocate batch storage. When this limit is reached,
    /// memtable will be flushed. Sized for typical ML workloads with
    /// 1024-dim vectors (~82KB per 20-row batch).
    /// Default: 8,000 batches
    pub max_memtable_batches: usize,

    /// Safety factor for IVF-PQ index partition capacity calculation.
    ///
    /// Accounts for non-uniform distribution of vectors across partitions.
    /// Higher values use more memory but reduce overflow risk.
    /// Partition capacity = min((max_rows / num_partitions) * safety_factor, max_rows)
    /// Default: 8
    pub ivf_index_partition_capacity_safety_factor: usize,

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
    /// better vectorization (especially for IVF-PQ) but increase memory usage
    /// and latency before data becomes searchable.
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
}

impl Default for RegionWriterConfig {
    fn default() -> Self {
        Self {
            region_id: Uuid::new_v4(),
            region_spec_id: 0,
            durable_write: true,
            sync_indexed_write: true,
            max_wal_buffer_size: 10 * 1024 * 1024, // 10MB
            max_wal_flush_interval: Some(Duration::from_millis(100)), // 100ms
            max_memtable_size: 256 * 1024 * 1024,  // 256MB
            max_memtable_rows: 100_000,            // 100k rows
            max_memtable_batches: 8_000,           // 8k batches
            ivf_index_partition_capacity_safety_factor: 8,
            manifest_scan_batch_size: 2,
            max_unflushed_memtable_bytes: 1024 * 1024 * 1024, // 1GB
            backpressure_log_interval: Duration::from_secs(30),
            async_index_buffer_rows: 10_000,
            async_index_interval: Duration::from_secs(1),
            stats_log_interval: Some(Duration::from_secs(60)), // 1 minute
        }
    }
}

impl RegionWriterConfig {
    /// Create a new configuration with the given region ID.
    pub fn new(region_id: Uuid) -> Self {
        Self {
            region_id,
            ..Default::default()
        }
    }

    /// Set the region spec ID.
    pub fn with_region_spec_id(mut self, spec_id: u32) -> Self {
        self.region_spec_id = spec_id;
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

    /// Set partition capacity safety factor for IVF-PQ indexes.
    pub fn with_ivf_index_partition_capacity_safety_factor(mut self, factor: usize) -> Self {
        self.ivf_index_partition_capacity_safety_factor = factor;
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
                                    break Err(e);
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
                            break Err(e);
                        }
                    }
                    msg = self.rx.recv() => {
                        match msg {
                            Some(message) => {
                                if let Err(e) = self.handler.handle(message).await {
                                    error!("Task '{}' error handling message: {}", self.name, e);
                                    break Err(e);
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
            Self::Failed(msg) => Err(Error::io(msg, location!())),
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
    config: RegionWriterConfig,
    /// Stats for monitoring.
    stats: Arc<BackpressureStats>,
}

impl BackpressureController {
    /// Create a new backpressure controller.
    pub fn new(config: RegionWriterConfig) -> Self {
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

/// RegionWriter state shared across tasks.
struct WriterState {
    memtable: MemTable,
    last_flushed_wal_entry_position: u64,
    /// Total size of frozen memtables (for backpressure).
    frozen_memtable_bytes: usize,
    /// Flush watchers for frozen memtables (for backpressure).
    frozen_flush_watchers: VecDeque<(usize, DurabilityWatcher)>,
    /// Flag to prevent duplicate memtable flush requests.
    flush_requested: bool,
    /// Counter for WAL flush threshold crossings.
    wal_flush_trigger_count: usize,
    /// Last time a WAL flush was triggered (for time-based flush).
    last_wal_flush_trigger_time: u64,
}

fn start_time() -> std::time::Instant {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

fn now_millis() -> u64 {
    start_time().elapsed().as_millis() as u64
}

/// Shared state for writer operations.
struct SharedWriterState {
    state: Arc<RwLock<WriterState>>,
    wal_flusher: Arc<WalFlusher>,
    wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
    memtable_flush_tx: mpsc::UnboundedSender<TriggerMemTableFlush>,
    config: RegionWriterConfig,
    schema: Arc<ArrowSchema>,
    pk_field_ids: Vec<i32>,
    max_memtable_batches: usize,
    max_memtable_rows: usize,
    ivf_index_partition_capacity_safety_factor: usize,
    index_configs: Vec<MemIndexConfig>,
}

impl SharedWriterState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        state: Arc<RwLock<WriterState>>,
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        memtable_flush_tx: mpsc::UnboundedSender<TriggerMemTableFlush>,
        config: RegionWriterConfig,
        schema: Arc<ArrowSchema>,
        pk_field_ids: Vec<i32>,
        max_memtable_batches: usize,
        max_memtable_rows: usize,
        ivf_index_partition_capacity_safety_factor: usize,
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
            max_memtable_batches,
            max_memtable_rows,
            ivf_index_partition_capacity_safety_factor,
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

        if !self.index_configs.is_empty() {
            let indexes = Arc::new(IndexStore::from_configs(
                &self.index_configs,
                self.max_memtable_rows,
                self.ivf_index_partition_capacity_safety_factor,
            )?);
            new_memtable.set_indexes_arc(indexes);
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
                old_batch_store,
                old_indexes,
                end_batch_position,
                Some(completion_cell),
            )?;
        }

        let frozen_size = old_memtable.estimated_size();
        state.frozen_memtable_bytes += frozen_size;
        state.last_flushed_wal_entry_position = last_wal_entry_position;

        let flush_watcher = old_memtable
            .get_memtable_flush_watcher()
            .expect("Flush watcher should exist after create_memtable_flush_completion");
        state
            .frozen_flush_watchers
            .push_back((frozen_size, flush_watcher));

        let frozen_memtable = Arc::new(old_memtable);

        debug!(
            "Frozen memtable generation {}, pending_count = {}",
            next_generation - 1,
            state.frozen_flush_watchers.len()
        );

        let _ = self.memtable_flush_tx.send(TriggerMemTableFlush {
            memtable: frozen_memtable,
            done: None,
        });

        Ok(next_generation)
    }

    /// Track batch for WAL durability.
    fn track_batch_for_wal(&self, batch_position: usize) -> DurabilityWatcher {
        let _wal_watcher = self.wal_flusher.track_batch(batch_position);
        // Return pre-resolved watcher for non-durable case
        let cell: WatchableOnceCell<DurabilityResult> = WatchableOnceCell::new();
        cell.write(DurabilityResult::ok());
        cell.reader()
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
                batch_store,
                indexes,
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
                batch_store: batch_store.clone(),
                indexes: indexes.clone(),
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

/// Main writer for a MemWAL region.
pub struct RegionWriter {
    config: RegionWriterConfig,
    epoch: u64,
    state: Arc<RwLock<WriterState>>,
    wal_flusher: Arc<WalFlusher>,
    task_executor: Arc<TaskExecutor>,
    manifest_store: Arc<RegionManifestStore>,
    stats: SharedWriteStats,
    writer_state: Arc<SharedWriterState>,
    backpressure: BackpressureController,
}

impl RegionWriter {
    /// Open or create a RegionWriter.
    ///
    /// The `base_path` should come from `ObjectStore::from_uri()` to ensure
    /// WAL files are written inside the dataset directory.
    pub async fn open(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: impl Into<String>,
        config: RegionWriterConfig,
        schema: Arc<ArrowSchema>,
        index_configs: Vec<MemIndexConfig>,
    ) -> Result<Self> {
        let base_uri = base_uri.into();
        let region_id = config.region_id;
        let manifest_store = Arc::new(RegionManifestStore::new(
            object_store.clone(),
            &base_path,
            region_id,
            config.manifest_scan_batch_size,
        ));

        // Claim the region (epoch-based fencing)
        let (epoch, manifest) = manifest_store.claim_epoch(config.region_spec_id).await?;

        info!(
            "Opened RegionWriter for region {} (epoch {}, generation {})",
            region_id, epoch, manifest.current_generation
        );

        // Create MemTable with primary key field IDs from schema
        let lance_schema = Schema::try_from(schema.as_ref())?;
        let pk_field_ids: Vec<i32> = lance_schema
            .unenforced_primary_key()
            .iter()
            .map(|f| f.id)
            .collect();
        let mut memtable = MemTable::with_capacity(
            schema.clone(),
            manifest.current_generation,
            pk_field_ids.clone(),
            CacheConfig::default(),
            config.max_memtable_batches,
        )?;

        // Create indexes if configured and set them on the MemTable
        // Indexes are always created when index_configs is non-empty
        // (they will be updated either sync or async based on config)
        if !index_configs.is_empty() {
            let indexes = Arc::new(IndexStore::from_configs(
                &index_configs,
                config.max_memtable_rows,
                config.ivf_index_partition_capacity_safety_factor,
            )?);
            memtable.set_indexes_arc(indexes);
        }

        let state = Arc::new(RwLock::new(WriterState {
            memtable,
            last_flushed_wal_entry_position: manifest.wal_entry_position_last_seen,
            frozen_memtable_bytes: 0,
            frozen_flush_watchers: VecDeque::new(),
            flush_requested: false,
            wal_flush_trigger_count: 0,
            last_wal_flush_trigger_time: 0,
        }));

        // Create WAL flusher
        let mut wal_flusher = WalFlusher::new(
            &base_path,
            region_id,
            epoch,
            manifest.wal_entry_position_last_seen + 1,
        );
        wal_flusher.set_object_store(object_store.clone());

        // Create channels for background tasks
        let (wal_flush_tx, wal_flush_rx) = mpsc::unbounded_channel();
        let (memtable_flush_tx, memtable_flush_rx) = mpsc::unbounded_channel();

        wal_flusher.set_flush_channel(wal_flush_tx.clone());
        let wal_flusher = Arc::new(wal_flusher);

        // Create flusher
        let flusher = Arc::new(MemTableFlusher::new(
            object_store.clone(),
            base_path,
            base_uri,
            region_id,
            manifest_store.clone(),
        ));

        // Create stats collector
        let stats = new_shared_stats();

        let backpressure = BackpressureController::new(config.clone());

        // Create task executor
        let task_executor = Arc::new(TaskExecutor::new());

        // Start background WAL flush handler
        // The WAL flush handler does parallel WAL I/O + index updates
        let wal_handler = WalFlushHandler::new(wal_flusher.clone(), state.clone(), stats.clone());
        task_executor.add_handler(
            "wal_flusher".to_string(),
            Box::new(wal_handler),
            wal_flush_rx,
        )?;

        // Start background MemTable flush handler
        let memtable_handler =
            MemTableFlushHandler::new(state.clone(), flusher, epoch, stats.clone());
        task_executor.add_handler(
            "memtable_flusher".to_string(),
            Box::new(memtable_handler),
            memtable_flush_rx,
        )?;

        // Create shared writer state for put() operations
        let writer_state = Arc::new(SharedWriterState::new(
            state.clone(),
            wal_flusher.clone(),
            wal_flush_tx,
            memtable_flush_tx,
            config.clone(),
            schema.clone(),
            pk_field_ids,
            config.max_memtable_batches,
            config.max_memtable_rows,
            config.ivf_index_partition_capacity_safety_factor,
            index_configs,
        ));

        Ok(Self {
            config,
            epoch,
            state,
            wal_flusher,
            task_executor,
            manifest_store,
            stats,
            writer_state,
            backpressure,
        })
    }

    /// Write record batches to the region.
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
    pub async fn put(&self, batches: Vec<RecordBatch>) -> Result<WriteResult> {
        if batches.is_empty() {
            return Err(Error::invalid_input(
                "Cannot write empty batch list",
                location!(),
            ));
        }

        // Validate no empty batches
        for (i, batch) in batches.iter().enumerate() {
            if batch.num_rows() == 0 {
                return Err(Error::invalid_input(
                    format!("Batch {} is empty", i),
                    location!(),
                ));
            }
        }

        // Apply backpressure if needed (before acquiring main lock)
        let writer_state = &self.writer_state;
        self.backpressure
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
            let mut state = self.state.write().await;

            // 1. Insert all batches into memtable atomically
            let results = state.memtable.insert_batches_only(batches).await?;

            // Get batch position range
            let start_pos = results.first().map(|(pos, _, _)| *pos).unwrap_or(0);
            let end_pos = results.last().map(|(pos, _, _)| pos + 1).unwrap_or(0);
            let batch_positions = start_pos..end_pos;

            // 2. Track last batch for WAL durability
            let durable_watcher = self
                .writer_state
                .track_batch_for_wal(end_pos.saturating_sub(1));

            // 3. Check if WAL flush should be triggered
            self.writer_state.maybe_trigger_wal_flush(&mut state);

            // 4. Check if memtable flush is needed
            if let Err(e) = self.writer_state.maybe_trigger_memtable_flush(&mut state) {
                warn!("Failed to trigger memtable flush: {}", e);
            }

            // Get batch_store and indexes while we have the lock (for durable_write case)
            let batch_store = state.memtable.batch_store();
            let indexes = state.memtable.indexes_arc();

            (batch_positions, durable_watcher, batch_store, indexes)
        }; // Lock released here

        self.stats.record_put(start.elapsed());

        // Wait for durability if configured (outside the lock)
        if self.config.durable_write {
            // Must trigger a flush to ensure durability (flush up to and including all batches)
            self.wal_flusher
                .trigger_flush(batch_store, indexes, batch_positions.end, None)?;
            durable_watcher.clone().await_value().await.into_result()?;
        }

        Ok(WriteResult { batch_positions })
    }

    /// Get a snapshot of current write statistics.
    pub fn stats(&self) -> WriteStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the shared stats handle (for external monitoring).
    pub fn stats_handle(&self) -> SharedWriteStats {
        self.stats.clone()
    }

    /// Get the current region manifest.
    pub async fn manifest(&self) -> Result<Option<RegionManifest>> {
        self.manifest_store.read_latest().await
    }

    /// Get the writer's epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Get the region ID.
    pub fn region_id(&self) -> Uuid {
        self.config.region_id
    }

    /// Get current MemTable statistics.
    pub async fn memtable_stats(&self) -> MemTableStats {
        let state = self.state.read().await;
        MemTableStats {
            row_count: state.memtable.row_count(),
            batch_count: state.memtable.batch_count(),
            estimated_size: state.memtable.estimated_size(),
            generation: state.memtable.generation(),
        }
    }

    /// Create a scanner for querying the current MemTable data.
    ///
    /// The scanner provides read access to all data currently in the MemTable,
    /// with optional filtering, projection, and index support.
    ///
    /// The scanner captures the current `max_indexed_batch_position` from the
    /// `IndexStore` at construction time to ensure consistent visibility.
    ///
    /// # Returns
    ///
    /// A `MemTableScanner` that can be used to execute queries.
    pub async fn scan(&self) -> MemTableScanner {
        let state = self.state.read().await;
        state.memtable.scan()
    }

    /// Get WAL statistics.
    pub fn wal_stats(&self) -> WalStats {
        WalStats {
            next_wal_entry_position: self.wal_flusher.next_wal_entry_position(),
        }
    }

    /// Close the writer gracefully.
    ///
    /// Flushes pending data and shuts down background tasks.
    pub async fn close(self) -> Result<()> {
        info!("Closing RegionWriter for region {}", self.config.region_id);

        // Send final WAL flush message and wait for completion
        let state = self.state.read().await;
        let batch_store = state.memtable.batch_store();
        let indexes = state.memtable.indexes_arc();
        let batch_count = state.memtable.batch_count();
        drop(state);

        // Only send flush if there are batches to flush
        if batch_count > 0 {
            // Create a completion cell to wait for flush
            let done = WatchableOnceCell::new();
            let reader = done.reader();

            // Send flush message with end_batch_position = batch_count to flush all pending
            if self
                .writer_state
                .wal_flush_tx
                .send(TriggerWalFlush {
                    batch_store,
                    indexes,
                    end_batch_position: batch_count,
                    done: Some(done),
                })
                .is_ok()
            {
                // Wait for flush to complete
                let mut reader = reader;
                let _ = reader.await_value().await;
            }
        }

        // Shutdown background tasks
        self.task_executor.shutdown_all().await?;

        info!("RegionWriter closed for region {}", self.config.region_id);
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
    state: Arc<RwLock<WriterState>>,
    stats: SharedWriteStats,
}

impl WalFlushHandler {
    fn new(
        wal_flusher: Arc<WalFlusher>,
        state: Arc<RwLock<WriterState>>,
        stats: SharedWriteStats,
    ) -> Self {
        Self {
            wal_flusher,
            state,
            stats,
        }
    }
}

#[async_trait]
impl MessageHandler<TriggerWalFlush> for WalFlushHandler {
    async fn handle(&mut self, message: TriggerWalFlush) -> Result<()> {
        let TriggerWalFlush {
            batch_store,
            indexes,
            end_batch_position,
            done,
        } = message;

        let result = self
            .do_flush(batch_store, indexes, end_batch_position)
            .await;

        // Notify completion if requested
        if let Some(cell) = done {
            cell.write(result.map_err(|e| e.to_string()));
        }

        Ok(())
    }
}

impl WalFlushHandler {
    /// Unified flush method for both active and frozen memtables.
    ///
    /// Detects frozen vs active flush by comparing the passed batch_store with the
    /// current active memtable's batch_store. If different, it's a frozen memtable flush.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - The batch store to flush from
    /// * `indexes` - Optional indexes to update in parallel with WAL I/O
    /// * `end_batch_position` - End batch ID (exclusive). Flush batches in (max_flushed, end_batch_position).
    async fn do_flush(
        &self,
        batch_store: Arc<BatchStore>,
        indexes: Option<Arc<IndexStore>>,
        end_batch_position: usize,
    ) -> Result<WalFlushResult> {
        let start = Instant::now();
        // Use batch_store's watermark - this is the authoritative source
        let max_flushed = batch_store.max_flushed_batch_position();
        // Convert to count-like value for comparison: number of batches already flushed
        let flushed_up_to = max_flushed.map(|p| p + 1).unwrap_or(0);

        // Detect if this is a frozen memtable flush by comparing batch_store pointers.
        // If the batch_store is different from the current active memtable's, it's frozen.
        let is_frozen_flush = {
            let state = self.state.read().await;
            !Arc::ptr_eq(&batch_store, &state.memtable.batch_store())
        };

        // Check if there's anything to flush (only skip for active memtable)
        if !is_frozen_flush && flushed_up_to >= end_batch_position {
            return Ok(WalFlushResult {
                entry: None,
                wal_io_duration: std::time::Duration::ZERO,
                index_update_duration: std::time::Duration::ZERO,
                index_update_duration_breakdown: std::collections::HashMap::new(),
                rows_indexed: 0,
                wal_bytes: 0,
            });
        }

        // Flush batches up to end_batch_position
        let flush_result = self
            .wal_flusher
            .flush_to_with_index_update(&batch_store, end_batch_position, indexes)
            .await?;

        let batches_flushed = flush_result
            .entry
            .as_ref()
            .map(|e| e.num_batches)
            .unwrap_or(0);

        // Note: WAL watermark is already updated by flush_to_with_index_update()
        // via batch_store.set_max_flushed_batch_position(). No need for separate mapping.

        // Record WAL flush stats
        if batches_flushed > 0 {
            self.stats
                .record_wal_flush(start.elapsed(), flush_result.wal_bytes);
            self.stats.record_wal_io(flush_result.wal_io_duration);
            self.stats.record_index_update(
                flush_result.index_update_duration,
                flush_result.rows_indexed,
            );
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
    stats: SharedWriteStats,
}

impl MemTableFlushHandler {
    fn new(
        state: Arc<RwLock<WriterState>>,
        flusher: Arc<MemTableFlusher>,
        epoch: u64,
        stats: SharedWriteStats,
    ) -> Self {
        Self {
            state,
            flusher,
            epoch,
            stats,
        }
    }
}

#[async_trait]
impl MessageHandler<TriggerMemTableFlush> for MemTableFlushHandler {
    async fn handle(&mut self, message: TriggerMemTableFlush) -> Result<()> {
        let TriggerMemTableFlush { memtable, done } = message;

        let result = self.flush_memtable(memtable).await;
        if let Some(tx) = done {
            // Send result through the channel - caller is waiting for it
            let _ = tx.send(result);
        } else {
            // No done channel, propagate errors
            result?;
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
    async fn flush_memtable(
        &mut self,
        memtable: Arc<MemTable>,
    ) -> Result<super::memtable::flush::FlushResult> {
        let start = Instant::now();
        let memtable_size = memtable.estimated_size();

        // Step 1: Wait for WAL flush completion (already queued at freeze time)
        // The TriggerWalFlush message was sent by freeze_memtable to ensure
        // strict ordering of WAL entries.
        if let Some(mut completion_reader) = memtable.take_wal_flush_completion() {
            completion_reader
                .await_value()
                .await
                .map_err(|e| Error::io(format!("WAL flush failed: {}", e), snafu::location!()))?;
        }

        // Step 2: Flush the memtable to Lance storage
        let result = self.flusher.flush(&memtable, self.epoch).await?;

        // Step 3: Signal completion and update backpressure tracking
        // Signal memtable flush completion for backpressure watchers
        memtable.signal_memtable_flush_complete();

        // Update backpressure tracking - remove the oldest watcher and decrement bytes
        {
            let mut state = self.state.write().await;
            if let Some((_size, _watcher)) = state.frozen_flush_watchers.pop_front() {
                state.frozen_memtable_bytes =
                    state.frozen_memtable_bytes.saturating_sub(memtable_size);
            }
        }

        // Record stats
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
    async fn test_region_writer_basic_write() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = RegionWriterConfig {
            region_id: Uuid::new_v4(),
            region_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = RegionWriter::open(
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
        let stats = writer.memtable_stats().await;
        assert_eq!(stats.row_count, 10);
        assert_eq!(stats.batch_count, 1);

        // Close writer
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_region_writer_multiple_writes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = RegionWriterConfig {
            region_id: Uuid::new_v4(),
            region_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = RegionWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        // Write multiple batches in a single put call
        let batches: Vec<_> = (0..5)
            .map(|i| create_test_batch(&schema, i * 10, 10))
            .collect();
        let result = writer.put(batches).await.unwrap();
        assert_eq!(result.batch_positions, 0..5);

        let stats = writer.memtable_stats().await;
        assert_eq!(stats.row_count, 50);
        assert_eq!(stats.batch_count, 5);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_region_writer_with_indexes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = RegionWriterConfig {
            region_id: Uuid::new_v4(),
            region_spec_id: 0,
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

        let writer = RegionWriter::open(
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

        let stats = writer.memtable_stats().await;
        assert_eq!(stats.row_count, 10);

        writer.close().await.unwrap();
    }

    /// Test memtable auto-flush triggered by size threshold.
    #[tokio::test]
    async fn test_region_writer_auto_flush_by_size() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Use a small memtable size to trigger auto-flush
        let config = RegionWriterConfig {
            region_id: Uuid::new_v4(),
            region_spec_id: 0,
            durable_write: false,
            sync_indexed_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: None,
            max_memtable_size: 1024, // Very small - will trigger flush quickly
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = RegionWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.generation;

        // Write batches until auto-flush triggers
        for i in 0..20 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        // Give time for background flush to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that generation increased (indicating flush happened)
        let stats = writer.memtable_stats().await;
        assert!(
            stats.generation > initial_gen,
            "Generation should increment after auto-flush"
        );

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_no_backpressure_when_under_threshold() {
        let config = RegionWriterConfig::default().with_max_unflushed_memtable_bytes(1024 * 1024); // 1MB

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

        let config = RegionWriterConfig::default()
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
}

#[cfg(test)]
mod region_writer_tests {
    use std::sync::Arc;

    use arrow_array::{
        FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::scalar::inverted::InvertedIndexParams;
    use lance_index::scalar::ScalarIndexParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::builder::PQBuildParams;
    use lance_index::{DatasetIndexExt, IndexType};
    use lance_linalg::distance::MetricType;
    use uuid::Uuid;

    use crate::dataset::mem_wal::{DatasetMemWalExt, MemWalConfig};
    use crate::dataset::{Dataset, WriteParams};
    use crate::index::vector::VectorIndexParams;

    use super::super::RegionWriterConfig;

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

    /// Quick smoke test for region writer - runs against memory://
    /// Run with: cargo test -p lance region_writer_tests::test_region_writer_smoke -- --nocapture
    #[tokio::test]
    async fn test_region_writer_smoke() {
        let vector_dim = 128;
        let batch_size = 20;
        let num_batches = 100;

        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_region_writer_{}", Uuid::new_v4());

        // Create initial dataset
        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Initialize MemWAL (no indexes for smoke test)
        dataset
            .initialize_mem_wal(MemWalConfig {
                region_spec: None,
                maintained_indexes: vec![],
            })
            .await
            .expect("Failed to initialize MemWAL");

        // Create region writer
        let region_id = Uuid::new_v4();
        let config = RegionWriterConfig::new(region_id)
            .with_durable_write(false)
            .with_sync_indexed_write(false);

        let writer = dataset
            .mem_wal_writer(region_id, config)
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

    /// Test region writer against S3 with IVF-PQ, BTree, and FTS indexes (requires DATASET_PREFIX env var)
    /// Run with: DATASET_PREFIX=s3://bucket/path cargo test -p lance --release region_writer_tests::test_region_writer_s3_ivfpq -- --nocapture --ignored
    #[tokio::test]
    #[ignore]
    async fn test_region_writer_s3_ivfpq() {
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
            .initialize_mem_wal(MemWalConfig {
                region_spec: None,
                maintained_indexes: vec![
                    "id_btree".to_string(),
                    "text_fts".to_string(),
                    "vector_idx".to_string(),
                ],
            })
            .await
            .expect("Failed to initialize MemWAL");

        // Create region writer with default config
        let region_id = Uuid::new_v4();
        let config = RegionWriterConfig::new(region_id)
            .with_durable_write(false)
            .with_sync_indexed_write(false);

        let writer = dataset
            .mem_wal_writer(region_id, config)
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

    /// End-to-end correctness test for RegionWriter with multiple memtable flushes.
    ///
    /// This test verifies:
    /// 1. Multiple memtable flushes are triggered via small memtable size
    /// 2. File system layout is correct (WAL files, manifest, generation directories)
    /// 3. WAL entries contain expected data
    /// 4. Data can be read after each flush cycle
    /// 5. Manifest tracks flushed generations correctly
    ///
    /// Run with: cargo test -p lance region_writer_tests::test_region_writer_e2e_correctness -- --nocapture
    #[tokio::test]
    async fn test_region_writer_e2e_correctness() {
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
            .initialize_mem_wal(MemWalConfig {
                region_spec: None,
                maintained_indexes: vec!["id_btree".to_string()],
            })
            .await
            .expect("Failed to initialize MemWAL");

        // Create region writer with small memtable size to trigger flushes
        let region_id = Uuid::new_v4();
        let config = RegionWriterConfig::new(region_id)
            .with_durable_write(true) // Ensure WAL files are written
            .with_sync_indexed_write(true)
            .with_max_memtable_size(50 * 1024) // 50KB - triggers flush after ~8 batches
            .with_max_wal_buffer_size(10 * 1024) // 10KB WAL buffer
            .with_max_wal_flush_interval(Duration::from_millis(50)); // Fast flush

        let writer = dataset
            .mem_wal_writer(region_id, config)
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
        let mem_wal_dir = temp_dir.path().join("_mem_wal").join(region_id.to_string());
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
            super::super::manifest::RegionManifestStore::new(store, &base_path, region_id, 2);
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
            // The path stored in manifest is relative to the region directory
            // Construct full path: temp_dir/_mem_wal/region_id/generation_folder
            let gen_path = temp_dir
                .path()
                .join("_mem_wal")
                .join(region_id.to_string())
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
        let new_region_id = Uuid::new_v4();
        let new_config = RegionWriterConfig::new(new_region_id)
            .with_durable_write(false)
            .with_sync_indexed_write(true);

        let new_writer = dataset
            .mem_wal_writer(new_region_id, new_config)
            .await
            .expect("Failed to create new writer");

        // Write a test batch to verify the new region works
        let verify_batch = create_test_batch(&schema, 10000, 10, vector_dim);
        new_writer
            .put(vec![verify_batch])
            .await
            .expect("Failed to write to new region");

        let scanner = new_writer.scan().await;
        let result = scanner.try_into_batch().await.expect("Failed to scan");
        assert_eq!(result.num_rows(), 10, "New region should have 10 rows");

        new_writer
            .close()
            .await
            .expect("Failed to close new writer");
    }
}
