// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lock-free append-only batch storage for MemTable.
//!
//! This module provides a high-performance, lock-free storage structure for
//! RecordBatches in the MemTable. It is designed for a single-writer,
//! multiple-reader scenario where:
//!
//! - A single writer task (WriteBatchHandler) appends batches
//! - Multiple reader tasks concurrently read batches
//! - No locks are needed for either reads or writes
//!
//! # Safety Model
//!
//! The lock-free design relies on these invariants:
//!
//! 1. **Single Writer**: Only one thread calls `append()` at a time.
//!    Enforced by the WriteBatchHandler architecture.
//!
//! 2. **Append-Only**: Once written, slots are never modified or removed
//!    until the entire store is dropped.
//!
//! 3. **Atomic Publishing**: Writer updates `committed_len` with Release
//!    ordering AFTER fully writing the slot. Readers load with Acquire
//!    ordering BEFORE reading slots.
//!
//! 4. **Fixed Capacity**: The store has a fixed capacity set at creation.
//!    When full, the MemTable should be flushed.
//!
//! # Memory Ordering
//!
//! ```text
//! Writer:                              Reader:
//! 1. Write data to slot[n]
//! 2. committed_len.store(n+1, Release)
//!    ─────────────────────────────────► synchronizes-with
//!                                      3. len = committed_len.load(Acquire)
//!                                      4. Read slot[i] where i < len
//! ```

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::RecordBatch;

/// A batch stored in the lock-free store.
#[derive(Clone)]
pub struct StoredBatch {
    /// The Arrow RecordBatch data.
    pub data: RecordBatch,
    /// Number of rows in this batch (cached for quick access).
    pub num_rows: usize,
    /// Row offset in the MemTable (cumulative rows before this batch).
    pub row_offset: u64,
    /// Position of this batch in the store (0-indexed).
    pub batch_position: usize,
}

impl StoredBatch {
    /// Create a new StoredBatch.
    pub fn new(data: RecordBatch, row_offset: u64, batch_position: usize) -> Self {
        let num_rows = data.num_rows();
        Self {
            data,
            num_rows,
            row_offset,
            batch_position,
        }
    }
}

/// Error returned when the store is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreFull;

impl std::fmt::Display for StoreFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BatchStore is full")
    }
}

impl std::error::Error for StoreFull {}

/// Lock-free append-only storage for memtable batches.
///
/// This structure provides O(1) lock-free appends and reads for a
/// single-writer, multiple-reader scenario.
///
/// # Example
///
/// ```ignore
/// let store = BatchStore::with_capacity(100);
///
/// // Writer (single thread)
/// store.append(batch1, 1)?;
/// store.append(batch2, 2)?;
///
/// // Readers (multiple threads, concurrent)
/// let len = store.len();
/// for i in 0..len {
///     let batch = store.get(i).unwrap();
///     // process batch...
/// }
/// ```
pub struct BatchStore {
    /// Pre-allocated storage slots.
    /// Each slot is either uninitialized or contains a valid StoredBatch.
    slots: Box<[UnsafeCell<MaybeUninit<StoredBatch>>]>,

    /// Number of committed (fully written) slots.
    /// Invariant: all slots [0, committed_len) contain valid data.
    committed_len: AtomicUsize,

    /// Total capacity (fixed at creation).
    capacity: usize,

    /// Total row count across all committed batches.
    total_rows: AtomicUsize,

    /// Estimated size in bytes (for flush threshold).
    estimated_bytes: AtomicUsize,

    /// WAL flush watermark: the last batch ID that has been flushed to WAL (inclusive).
    /// Uses usize::MAX as sentinel for "nothing flushed yet".
    /// This is per-memtable tracking, not global.
    max_flushed_batch_position: AtomicUsize,
}

// SAFETY: Safe to share across threads because:
// - Single writer guarantee (architectural invariant)
// - Readers only access committed slots (index < committed_len)
// - Atomic operations provide proper synchronization
// - Slots are never modified after being written
unsafe impl Sync for BatchStore {}
unsafe impl Send for BatchStore {}

impl BatchStore {
    /// Create a new store with the given capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of batches. Should be sized based on
    ///   `max_memtable_size / expected_avg_batch_size`.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");

        // Allocate uninitialized storage
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(UnsafeCell::new(MaybeUninit::uninit()));
        }

        Self {
            slots: slots.into_boxed_slice(),
            committed_len: AtomicUsize::new(0),
            capacity,
            total_rows: AtomicUsize::new(0),
            estimated_bytes: AtomicUsize::new(0),
            max_flushed_batch_position: AtomicUsize::new(usize::MAX), // Nothing flushed yet
        }
    }

    /// Calculate recommended capacity from memtable size configuration.
    ///
    /// Uses an assumed average batch size of 64KB with 20% buffer.
    pub fn recommended_capacity(max_memtable_bytes: usize) -> usize {
        const AVG_BATCH_SIZE: usize = 64 * 1024; // 64KB
        const BUFFER_FACTOR: f64 = 1.2;

        let estimated_batches = max_memtable_bytes / AVG_BATCH_SIZE;
        let capacity = ((estimated_batches as f64) * BUFFER_FACTOR) as usize;
        capacity.max(16) // Minimum 16 slots
    }

    /// Returns the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the store is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.committed_len.load(Ordering::Relaxed) >= self.capacity
    }

    /// Returns the number of remaining slots.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity
            .saturating_sub(self.committed_len.load(Ordering::Relaxed))
    }

    // =========================================================================
    // Writer API (Single Writer Only)
    // =========================================================================

    /// Append a batch to the store.
    ///
    /// # Safety Requirements
    ///
    /// This method MUST only be called from the single writer task.
    /// Concurrent calls from multiple threads cause undefined behavior.
    ///
    /// # Returns
    ///
    /// - `Ok((batch_position, row_offset, estimated_size))` - The index, row offset, and size of the appended batch
    /// - `Err(StoreFull)` - The store is at capacity, needs flush
    pub fn append(&self, batch: RecordBatch) -> Result<(usize, u64, usize), StoreFull> {
        // Load current length (Relaxed is fine - we're the only writer)
        let idx = self.committed_len.load(Ordering::Relaxed);

        if idx >= self.capacity {
            return Err(StoreFull);
        }

        let num_rows = batch.num_rows();
        let estimated_size = Self::estimate_batch_size(&batch);

        // Row offset is the total rows BEFORE this batch
        let row_offset = self.total_rows.load(Ordering::Relaxed) as u64;

        let stored = StoredBatch::new(batch, row_offset, idx);

        // SAFETY:
        // 1. idx < capacity, so slot exists
        // 2. Single writer guarantee - no concurrent writes to this slot
        // 3. Slot at idx is uninitialized (never written before, append-only)
        unsafe {
            let slot_ptr = self.slots[idx].get();
            std::ptr::write(slot_ptr, MaybeUninit::new(stored));
        }

        // Update counters (Relaxed - just tracking, not synchronization)
        self.total_rows.fetch_add(num_rows, Ordering::Relaxed);
        self.estimated_bytes
            .fetch_add(estimated_size, Ordering::Relaxed);

        // CRITICAL: Publish with Release ordering.
        // This ensures all writes above are visible to readers
        // who load committed_len with Acquire ordering.
        self.committed_len.store(idx + 1, Ordering::Release);

        Ok((idx, row_offset, estimated_size))
    }

    /// Append multiple batches to the store atomically.
    ///
    /// All batches are written before publishing, so readers see either
    /// none of the batches or all of them (atomic visibility).
    ///
    /// # Safety Requirements
    ///
    /// This method MUST only be called from the single writer task.
    /// Concurrent calls from multiple threads cause undefined behavior.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<(batch_position, row_offset, estimated_size)>)` - Info for each appended batch
    /// - `Err(StoreFull)` - Not enough capacity for all batches
    pub fn append_batches(
        &self,
        batches: Vec<RecordBatch>,
    ) -> Result<Vec<(usize, u64, usize)>, StoreFull> {
        if batches.is_empty() {
            return Ok(vec![]);
        }

        // Load current length (Relaxed is fine - we're the only writer)
        let start_idx = self.committed_len.load(Ordering::Relaxed);
        let count = batches.len();

        // Check capacity for ALL batches upfront
        if start_idx + count > self.capacity {
            return Err(StoreFull);
        }

        let mut results = Vec::with_capacity(count);
        let mut total_rows_added = 0usize;
        let mut total_bytes_added = 0usize;
        let mut row_offset = self.total_rows.load(Ordering::Relaxed) as u64;

        // Write all batches to slots (not yet visible to readers)
        for (i, batch) in batches.into_iter().enumerate() {
            let idx = start_idx + i;
            let num_rows = batch.num_rows();
            let estimated_size = Self::estimate_batch_size(&batch);

            let stored = StoredBatch::new(batch, row_offset, idx);

            // SAFETY:
            // 1. idx < capacity (checked above)
            // 2. Single writer guarantee - no concurrent writes to this slot
            // 3. Slot at idx is uninitialized (never written before, append-only)
            unsafe {
                let slot_ptr = self.slots[idx].get();
                std::ptr::write(slot_ptr, MaybeUninit::new(stored));
            }

            results.push((idx, row_offset, estimated_size));
            row_offset += num_rows as u64;
            total_rows_added += num_rows;
            total_bytes_added += estimated_size;
        }

        // Update counters (Relaxed - just tracking, not synchronization)
        self.total_rows
            .fetch_add(total_rows_added, Ordering::Relaxed);
        self.estimated_bytes
            .fetch_add(total_bytes_added, Ordering::Relaxed);

        // CRITICAL: Publish ALL batches at once with Release ordering.
        // This ensures all writes above are visible to readers
        // who load committed_len with Acquire ordering.
        self.committed_len
            .store(start_idx + count, Ordering::Release);

        Ok(results)
    }

    /// Estimate the memory size of a RecordBatch.
    fn estimate_batch_size(batch: &RecordBatch) -> usize {
        batch
            .columns()
            .iter()
            .map(|col| col.get_array_memory_size())
            .sum::<usize>()
            + std::mem::size_of::<RecordBatch>()
    }

    // =========================================================================
    // Reader API (Multiple Concurrent Readers)
    // =========================================================================

    /// Get the number of committed batches.
    #[inline]
    pub fn len(&self) -> usize {
        self.committed_len.load(Ordering::Acquire)
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the maximum buffered batch position (inclusive).
    ///
    /// Returns `None` if no batches have been buffered.
    /// Returns `Some(len - 1)` otherwise, which is the position of the last buffered batch.
    #[inline]
    pub fn max_buffered_batch_position(&self) -> Option<usize> {
        let len = self.len();
        if len == 0 {
            None
        } else {
            Some(len - 1)
        }
    }

    /// Get total row count.
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.total_rows.load(Ordering::Relaxed)
    }

    /// Get estimated size in bytes.
    #[inline]
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes.load(Ordering::Relaxed)
    }

    // =========================================================================
    // WAL Flush Tracking API
    // =========================================================================

    /// Get the WAL flush watermark (the last batch ID that was flushed, inclusive).
    /// Returns None if nothing has been flushed yet.
    #[inline]
    pub fn max_flushed_batch_position(&self) -> Option<usize> {
        let watermark = self.max_flushed_batch_position.load(Ordering::Acquire);
        if watermark == usize::MAX {
            None
        } else {
            Some(watermark)
        }
    }

    /// Update the WAL flush watermark after successful WAL flush.
    ///
    /// # Arguments
    ///
    /// * `batch_position` - The last batch ID that was flushed (inclusive)
    #[inline]
    pub fn set_max_flushed_batch_position(&self, batch_position: usize) {
        debug_assert!(
            batch_position != usize::MAX,
            "batch_position cannot be usize::MAX (reserved as sentinel)"
        );
        self.max_flushed_batch_position
            .store(batch_position, Ordering::Release);
    }

    /// Get the number of batches pending WAL flush.
    #[inline]
    pub fn pending_wal_flush_count(&self) -> usize {
        let committed = self.committed_len.load(Ordering::Acquire);
        let watermark = self.max_flushed_batch_position.load(Ordering::Acquire);
        if watermark == usize::MAX {
            // Nothing flushed yet, all committed batches are pending
            committed
        } else {
            // Batches [0, watermark] are flushed, so pending = committed - (watermark + 1)
            committed.saturating_sub(watermark + 1)
        }
    }

    /// Check if all committed batches have been WAL-flushed.
    #[inline]
    pub fn is_wal_flush_complete(&self) -> bool {
        self.pending_wal_flush_count() == 0
    }

    /// Get the range of batch IDs pending WAL flush: [start, end).
    /// Returns None if nothing pending.
    #[inline]
    pub fn pending_wal_flush_range(&self) -> Option<(usize, usize)> {
        let committed = self.committed_len.load(Ordering::Acquire);
        let watermark = self.max_flushed_batch_position.load(Ordering::Acquire);
        let start = if watermark == usize::MAX {
            0
        } else {
            watermark + 1
        };
        if committed > start {
            Some((start, committed))
        } else {
            None
        }
    }

    /// Get a reference to a batch by index.
    ///
    /// Returns `None` if index >= committed length.
    ///
    /// # Safety
    ///
    /// The returned reference is valid as long as `self` is not dropped.
    /// This is safe because:
    /// - We only access slots where index < committed_len (Acquire load)
    /// - Slots are never modified after being written
    /// - The store is append-only
    #[inline]
    pub fn get(&self, index: usize) -> Option<&StoredBatch> {
        // Acquire ordering synchronizes with Release in append()
        let len = self.committed_len.load(Ordering::Acquire);

        if index >= len {
            return None;
        }

        // SAFETY:
        // 1. index < len, and len was loaded with Acquire ordering
        // 2. The Release-Acquire pair ensures the write is visible
        // 3. Slots are never modified after writing (append-only)
        unsafe {
            let slot_ptr = self.slots[index].get();
            Some((*slot_ptr).assume_init_ref())
        }
    }

    /// Get the RecordBatch data at an index.
    #[inline]
    pub fn get_batch(&self, index: usize) -> Option<&RecordBatch> {
        self.get(index).map(|s| &s.data)
    }

    /// Iterate over all committed batches.
    ///
    /// The iterator captures a snapshot of the committed length at creation
    /// time, so it will not see batches appended during iteration.
    pub fn iter(&self) -> BatchStoreIter<'_> {
        let len = self.committed_len.load(Ordering::Acquire);
        BatchStoreIter {
            store: self,
            current: 0,
            len,
        }
    }

    /// Get all batches as a Vec (clones the RecordBatch data).
    pub fn to_vec(&self) -> Vec<RecordBatch> {
        self.iter().map(|b| b.data.clone()).collect()
    }

    /// Get all StoredBatches as a Vec (clones).
    pub fn to_stored_vec(&self) -> Vec<StoredBatch> {
        self.iter().cloned().collect()
    }

    // =========================================================================
    // Visibility API
    // =========================================================================

    /// Get batches visible up to a specific batch position (inclusive).
    ///
    /// A batch at position `i` is visible if `i <= max_visible_batch_position`.
    pub fn visible_batches(&self, max_visible_batch_position: usize) -> Vec<&StoredBatch> {
        let len = self.committed_len.load(Ordering::Acquire);
        let end = (max_visible_batch_position + 1).min(len);
        (0..end).filter_map(|i| self.get(i)).collect()
    }

    /// Get batch positions visible up to a specific batch position (inclusive).
    pub fn max_visible_batch_positions(&self, max_visible_batch_position: usize) -> Vec<usize> {
        let len = self.committed_len.load(Ordering::Acquire);
        let end = (max_visible_batch_position + 1).min(len);
        (0..end).collect()
    }

    /// Check if a specific batch is visible at a given visibility position.
    #[inline]
    pub fn is_batch_visible(
        &self,
        batch_position: usize,
        max_visible_batch_position: usize,
    ) -> bool {
        let len = self.committed_len.load(Ordering::Acquire);
        batch_position < len && batch_position <= max_visible_batch_position
    }

    /// Get visible RecordBatches (clones the data).
    pub fn visible_record_batches(&self, max_visible_batch_position: usize) -> Vec<RecordBatch> {
        self.visible_batches(max_visible_batch_position)
            .into_iter()
            .map(|b| b.data.clone())
            .collect()
    }

    /// Get visible RecordBatches with their row offsets.
    ///
    /// Returns tuples of (batch, row_offset) for each visible batch.
    /// The row_offset is the starting row position for that batch.
    pub fn visible_batches_with_offsets(
        &self,
        max_visible_batch_position: usize,
    ) -> Vec<(RecordBatch, u64)> {
        self.visible_batches(max_visible_batch_position)
            .into_iter()
            .map(|b| (b.data.clone(), b.row_offset))
            .collect()
    }
}

impl Drop for BatchStore {
    fn drop(&mut self) {
        // Get the committed length directly (no atomic needed, we have &mut self)
        let len = *self.committed_len.get_mut();

        // Drop all initialized slots
        for i in 0..len {
            // SAFETY: slots [0, len) are initialized and we have exclusive access
            unsafe {
                let slot_ptr = self.slots[i].get();
                std::ptr::drop_in_place((*slot_ptr).as_mut_ptr());
            }
        }
    }
}

/// Iterator over committed batches in a BatchStore.
///
/// This iterator captures a snapshot of the committed length at creation,
/// providing a consistent view even if new batches are appended during
/// iteration.
pub struct BatchStoreIter<'a> {
    store: &'a BatchStore,
    current: usize,
    len: usize,
}

impl<'a> Iterator for BatchStoreIter<'a> {
    type Item = &'a StoredBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.len {
            return None;
        }

        // SAFETY: current < len, which was captured with Acquire ordering
        let batch = unsafe {
            let slot_ptr = self.store.slots[self.current].get();
            (*slot_ptr).assume_init_ref()
        };

        self.current += 1;
        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BatchStoreIter<'_> {}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]))
    }

    fn create_test_batch(num_rows: usize) -> RecordBatch {
        let schema = create_test_schema();
        let ids: Vec<i32> = (0..num_rows as i32).collect();
        let values: Vec<i32> = ids.iter().map(|id| id * 10).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_create_store() {
        let store = BatchStore::with_capacity(10);
        assert_eq!(store.capacity(), 10);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert!(!store.is_full());
        assert_eq!(store.remaining_capacity(), 10);
    }

    #[test]
    fn test_append_single() {
        let store = BatchStore::with_capacity(10);
        let batch = create_test_batch(100);

        let (id, row_offset, _size) = store.append(batch).unwrap();
        assert_eq!(id, 0);
        assert_eq!(row_offset, 0); // First batch starts at row 0
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(store.total_rows(), 100);
    }

    #[test]
    fn test_append_multiple() {
        let store = BatchStore::with_capacity(10);

        let mut expected_row_offset = 0u64;
        for i in 0..5 {
            let num_rows = 10 * (i + 1);
            let batch = create_test_batch(num_rows);
            let (id, row_offset, _size) = store.append(batch).unwrap();
            assert_eq!(id, i);
            assert_eq!(row_offset, expected_row_offset);
            expected_row_offset += num_rows as u64;
        }

        assert_eq!(store.len(), 5);
        assert_eq!(store.total_rows(), 10 + 20 + 30 + 40 + 50);
    }

    #[test]
    fn test_capacity_limit() {
        let store = BatchStore::with_capacity(3);

        store.append(create_test_batch(10)).unwrap();
        store.append(create_test_batch(10)).unwrap();
        store.append(create_test_batch(10)).unwrap();

        assert!(store.is_full());
        assert_eq!(store.remaining_capacity(), 0);

        let result = store.append(create_test_batch(10));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StoreFull);
    }

    #[test]
    fn test_get_batch() {
        let store = BatchStore::with_capacity(10);

        let batch1 = create_test_batch(10);
        let batch2 = create_test_batch(20);

        store.append(batch1).unwrap();
        store.append(batch2).unwrap();

        let retrieved1 = store.get(0).unwrap();
        assert_eq!(retrieved1.num_rows, 10);
        assert_eq!(retrieved1.row_offset, 0);

        let retrieved2 = store.get(1).unwrap();
        assert_eq!(retrieved2.num_rows, 20);
        assert_eq!(retrieved2.row_offset, 10); // After first batch

        // Out of bounds
        assert!(store.get(2).is_none());
        assert!(store.get(100).is_none());
    }

    #[test]
    fn test_iter() {
        let store = BatchStore::with_capacity(10);

        for _ in 0..5 {
            store.append(create_test_batch(10)).unwrap();
        }

        let batches: Vec<_> = store.iter().collect();
        assert_eq!(batches.len(), 5);
    }

    #[test]
    fn test_visibility_filtering() {
        let store = BatchStore::with_capacity(10);

        store.append(create_test_batch(10)).unwrap(); // position 0
        store.append(create_test_batch(10)).unwrap(); // position 1
        store.append(create_test_batch(10)).unwrap(); // position 2
        store.append(create_test_batch(10)).unwrap(); // position 3
        store.append(create_test_batch(10)).unwrap(); // position 4

        // max_visible_batch_position=2 means positions 0, 1, 2 are visible
        let visible = store.max_visible_batch_positions(2);
        assert_eq!(visible, vec![0, 1, 2]);

        // max_visible_batch_position=4 means all visible
        let visible = store.max_visible_batch_positions(4);
        assert_eq!(visible, vec![0, 1, 2, 3, 4]);

        // max_visible_batch_position=0 means only position 0 visible
        let visible = store.max_visible_batch_positions(0);
        assert_eq!(visible, vec![0]);
    }

    #[test]
    fn test_is_batch_visible() {
        let store = BatchStore::with_capacity(10);

        store.append(create_test_batch(10)).unwrap(); // position 0
        store.append(create_test_batch(10)).unwrap(); // position 1
        store.append(create_test_batch(10)).unwrap(); // position 2

        // Batch at position 0 is visible when max_visible_batch_position >= 0
        assert!(store.is_batch_visible(0, 0));
        assert!(store.is_batch_visible(0, 1));
        assert!(store.is_batch_visible(0, 2));

        // Batch at position 2 is only visible when max_visible_batch_position >= 2
        assert!(!store.is_batch_visible(2, 1));
        assert!(store.is_batch_visible(2, 2));
        assert!(store.is_batch_visible(2, 3));

        // Batch 3 doesn't exist
        assert!(!store.is_batch_visible(3, 10));
    }

    #[test]
    fn test_recommended_capacity() {
        // 64MB memtable, 64KB avg batch = 1024 batches * 1.2 = ~1228
        let cap = BatchStore::recommended_capacity(64 * 1024 * 1024);
        assert!(
            (1200..=1300).contains(&cap),
            "capacity should be around 1200, got {}",
            cap
        );

        // Very small memtable should get minimum capacity
        let cap = BatchStore::recommended_capacity(1024);
        assert_eq!(cap, 16); // minimum
    }

    #[test]
    fn test_to_vec() {
        let store = BatchStore::with_capacity(10);

        let batch1 = create_test_batch(10);
        let batch2 = create_test_batch(20);

        store.append(batch1).unwrap();
        store.append(batch2).unwrap();

        let vec = store.to_vec();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0].num_rows(), 10);
        assert_eq!(vec[1].num_rows(), 20);
    }

    #[test]
    fn test_concurrent_readers() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(BatchStore::with_capacity(100));

        // Pre-populate with some batches
        for _ in 0..50 {
            store.append(create_test_batch(10)).unwrap();
        }

        // Spawn multiple reader threads
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let reader_store = store.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let len = reader_store.len();
                        assert_eq!(len, 50);

                        // Verify we can read all batches
                        for i in 0..len {
                            let batch = reader_store.get(i);
                            assert!(batch.is_some());
                            assert_eq!(batch.unwrap().num_rows, 10);
                        }

                        // Verify iterator
                        let count = reader_store.iter().count();
                        assert_eq!(count, 50);

                        thread::yield_now();
                    }
                })
            })
            .collect();

        for r in readers {
            r.join().unwrap();
        }
    }

    #[test]
    fn test_append_batches() {
        let store = BatchStore::with_capacity(10);

        let batches: Vec<_> = (0..5).map(|i| create_test_batch(10 * (i + 1))).collect();
        let results = store.append_batches(batches).unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(store.len(), 5);

        // Check batch positions are sequential
        for (i, (batch_pos, _, _)) in results.iter().enumerate() {
            assert_eq!(*batch_pos, i);
        }

        // Check row offsets are cumulative
        assert_eq!(results[0].1, 0); // First batch starts at 0
        assert_eq!(results[1].1, 10); // After 10 rows
        assert_eq!(results[2].1, 30); // After 10 + 20 rows
        assert_eq!(results[3].1, 60); // After 10 + 20 + 30 rows
        assert_eq!(results[4].1, 100); // After 10 + 20 + 30 + 40 rows

        // Check total rows
        assert_eq!(store.total_rows(), 10 + 20 + 30 + 40 + 50);
    }

    #[test]
    fn test_append_batches_capacity_check() {
        let store = BatchStore::with_capacity(3);

        // Append 2 batches, should succeed
        let batches: Vec<_> = (0..2).map(|_| create_test_batch(10)).collect();
        store.append_batches(batches).unwrap();
        assert_eq!(store.len(), 2);

        // Try to append 2 more, should fail (only 1 slot left)
        let batches: Vec<_> = (0..2).map(|_| create_test_batch(10)).collect();
        let result = store.append_batches(batches);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StoreFull);

        // Store should be unchanged
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_append_batches_empty() {
        let store = BatchStore::with_capacity(10);

        let results = store.append_batches(vec![]).unwrap();
        assert!(results.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_concurrent_read_write() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(BatchStore::with_capacity(200));
        let done = Arc::new(AtomicBool::new(false));

        // Writer thread (single writer)
        let writer_store = store.clone();
        let writer_done = done.clone();
        let writer = thread::spawn(move || {
            for _ in 0..100 {
                writer_store.append(create_test_batch(10)).unwrap();
                thread::yield_now();
            }
            writer_done.store(true, Ordering::Release);
        });

        // Reader threads (concurrent readers)
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let reader_store = store.clone();
                let reader_done = done.clone();
                thread::spawn(move || {
                    while !reader_done.load(Ordering::Acquire) {
                        let len = reader_store.len();

                        // Every batch we can see should be valid
                        for i in 0..len {
                            let batch = reader_store.get(i);
                            assert!(batch.is_some());
                        }

                        thread::yield_now();
                    }

                    // Final check - should see all 100 batches
                    assert_eq!(reader_store.len(), 100);
                })
            })
            .collect();

        writer.join().unwrap();
        for r in readers {
            r.join().unwrap();
        }
    }
}
