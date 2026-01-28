// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory IVF-PQ index for vector similarity search.
//!
//! Uses hybrid storage with pre-allocated primary buffers and SkipMap overflow.
//! Reuses IVF centroids and PQ codebook from the base table for consistent
//! distance computations.
//!
//! # Architecture
//!
//! Each partition uses hybrid storage:
//! - **Primary**: Pre-allocated `ColumnMajorIvfPqMemPartition` with transposed codes
//! - **Overflow**: `SkipMap` for when primary is full (row-major, transpose at search)
//!
//! This design ensures writes never block while optimizing the common case.
//!
//! # Safety Model
//!
//! Same as `BatchStore`:
//! - Single writer (WalFlushHandler during WAL flush)
//! - Multiple concurrent readers
//! - Append-only until memtable flush

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt8Array};
use crossbeam_skiplist::SkipMap;
use lance_core::{Error, Result};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::kmeans::compute_partitions_arrow_array;
use lance_index::vector::pq::storage::transpose;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::Quantization;
use lance_linalg::distance::DistanceType;
use snafu::location;

use crate::dataset::mem_wal::memtable::batch_store::StoredBatch;

pub use super::RowPosition;

// ============================================================================
// Lock-free IVF-PQ Partition Storage
// ============================================================================

/// Error when partition store is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionFull;

impl std::fmt::Display for PartitionFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IVF-PQ partition store is full")
    }
}

impl std::error::Error for PartitionFull {}

/// Lock-free storage for a single IVF partition with pre-transposed PQ codes.
///
/// Stores PQ codes in column-major (transposed) format for zero-cost
/// search-time access. Uses the same single-writer, multi-reader pattern
/// as `BatchStore`.
///
/// # Memory Layout
///
/// ```text
/// codes: [subvec_0_all_vectors | subvec_1_all_vectors | ... | subvec_n_all_vectors]
/// ```
///
/// Each subvector section has `capacity` bytes pre-allocated.
///
/// # Safety
///
/// - Single writer (WalFlushHandler during WAL flush)
/// - Multiple concurrent readers
/// - Append-only until memtable flush
#[derive(Debug)]
struct ColumnMajorIvfPqMemPartition {
    /// Pre-allocated column-major PQ codes.
    /// Layout: codes[subvec_idx * capacity + vector_idx] = code_byte
    codes: UnsafeCell<Box<[MaybeUninit<u8>]>>,

    /// Row positions for result mapping.
    row_positions: UnsafeCell<Box<[MaybeUninit<u64>]>>,

    /// Number of vectors committed (visible to readers).
    committed_len: AtomicUsize,

    /// Maximum vectors this partition can hold.
    capacity: usize,

    /// Number of sub-vectors (PQ code length).
    num_sub_vectors: usize,
}

// SAFETY: Single-writer pattern enforced by architecture.
// UnsafeCell contents are only mutated by single writer thread.
unsafe impl Sync for ColumnMajorIvfPqMemPartition {}
unsafe impl Send for ColumnMajorIvfPqMemPartition {}

impl ColumnMajorIvfPqMemPartition {
    /// Create a new partition store with given capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of vectors
    /// * `num_sub_vectors` - PQ code length (number of sub-vectors)
    ///
    /// # Panics
    ///
    /// Panics if capacity or num_sub_vectors is 0.
    fn new(capacity: usize, num_sub_vectors: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(num_sub_vectors > 0, "num_sub_vectors must be > 0");

        // Allocate codes: capacity * num_sub_vectors bytes
        let codes_size = capacity * num_sub_vectors;
        let mut codes = Vec::with_capacity(codes_size);
        for _ in 0..codes_size {
            codes.push(MaybeUninit::uninit());
        }

        // Allocate row positions: capacity u64s
        let mut row_positions = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            row_positions.push(MaybeUninit::uninit());
        }

        Self {
            codes: UnsafeCell::new(codes.into_boxed_slice()),
            row_positions: UnsafeCell::new(row_positions.into_boxed_slice()),
            committed_len: AtomicUsize::new(0),
            capacity,
            num_sub_vectors,
        }
    }

    /// Returns the number of committed vectors.
    #[inline]
    fn len(&self) -> usize {
        self.committed_len.load(Ordering::Acquire)
    }

    /// Returns remaining capacity.
    #[inline]
    fn remaining_capacity(&self) -> usize {
        self.capacity
            .saturating_sub(self.committed_len.load(Ordering::Relaxed))
    }

    /// Append a batch of already-transposed PQ codes.
    ///
    /// # Arguments
    ///
    /// * `transposed_codes` - Column-major codes from `transpose()`.
    ///   Layout: [subvec0_all, subvec1_all, ...] where each section
    ///   has `num_vectors` bytes.
    /// * `positions` - Row positions for each vector.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Successfully appended
    /// * `Err(PartitionFull)` - Not enough capacity
    ///
    /// # Safety
    ///
    /// Must be called from single writer thread only.
    fn append_transposed_batch(
        &self,
        transposed_codes: &[u8],
        positions: &[u64],
    ) -> std::result::Result<(), PartitionFull> {
        let num_vectors = positions.len();
        if num_vectors == 0 {
            return Ok(());
        }

        debug_assert_eq!(
            transposed_codes.len(),
            num_vectors * self.num_sub_vectors,
            "transposed_codes length mismatch: expected {}, got {}",
            num_vectors * self.num_sub_vectors,
            transposed_codes.len()
        );

        let committed = self.committed_len.load(Ordering::Relaxed);
        if committed + num_vectors > self.capacity {
            return Err(PartitionFull);
        }

        // SAFETY: Single writer, and we checked capacity.
        let codes = unsafe { &mut *self.codes.get() };
        let row_pos = unsafe { &mut *self.row_positions.get() };

        // Copy transposed codes column by column.
        // Source layout: [sv0_v0..sv0_vN, sv1_v0..sv1_vN, ...]
        // Dest layout:   [sv0_v0..sv0_vCAP, sv1_v0..sv1_vCAP, ...]
        for subvec_idx in 0..self.num_sub_vectors {
            let src_start = subvec_idx * num_vectors;
            let dst_start = subvec_idx * self.capacity + committed;

            for i in 0..num_vectors {
                codes[dst_start + i].write(transposed_codes[src_start + i]);
            }
        }

        // Copy row positions.
        for (i, &pos) in positions.iter().enumerate() {
            row_pos[committed + i].write(pos);
        }

        // Publish with release ordering.
        self.committed_len
            .store(committed + num_vectors, Ordering::Release);

        Ok(())
    }

    /// Get codes formatted for `ProductQuantizer::compute_distances()`.
    ///
    /// Copies committed codes to a contiguous buffer in column-major format.
    /// This is the format expected by `compute_distances()`.
    ///
    /// # Returns
    ///
    /// Tuple of (contiguous_codes, row_positions).
    fn get_codes_for_search(&self) -> (Vec<u8>, Vec<u64>) {
        let len = self.committed_len.load(Ordering::Acquire);
        if len == 0 {
            return (Vec::new(), Vec::new());
        }

        let codes = unsafe { &*self.codes.get() };
        let row_pos = unsafe { &*self.row_positions.get() };

        // Copy codes to contiguous buffer (remove capacity gaps).
        let mut result_codes = Vec::with_capacity(len * self.num_sub_vectors);
        for subvec_idx in 0..self.num_sub_vectors {
            let start = subvec_idx * self.capacity;
            for i in 0..len {
                // SAFETY: i < len <= committed_len, data was initialized.
                result_codes.push(unsafe { codes[start + i].assume_init() });
            }
        }

        // Copy row positions.
        let result_positions: Vec<u64> = (0..len)
            .map(|i| unsafe { row_pos[i].assume_init() })
            .collect();

        (result_codes, result_positions)
    }
}

/// A single IVF partition with primary (pre-transposed) and overflow (row-major) storage.
///
/// This is the main interface for partition storage, handling the split between
/// fast primary storage and overflow when primary is full.
#[derive(Debug)]
pub struct IvfPqMemPartition {
    /// Primary storage: pre-allocated, pre-transposed codes (fast search).
    primary: ColumnMajorIvfPqMemPartition,

    /// Overflow storage: SkipMap for when primary is full (slower search).
    /// Key: row_position, Value: row-major PQ code.
    overflow: SkipMap<u64, Vec<u8>>,

    /// Number of vectors in overflow (cached for fast access).
    overflow_count: AtomicUsize,

    /// Number of sub-vectors (code length).
    num_sub_vectors: usize,
}

impl IvfPqMemPartition {
    /// Create a new partition with given capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum vectors in primary storage
    /// * `num_sub_vectors` - PQ code length
    pub fn new(capacity: usize, num_sub_vectors: usize) -> Self {
        Self {
            primary: ColumnMajorIvfPqMemPartition::new(capacity, num_sub_vectors),
            overflow: SkipMap::new(),
            overflow_count: AtomicUsize::new(0),
            num_sub_vectors,
        }
    }

    /// Append a batch of vectors to this partition.
    ///
    /// Goes to primary if capacity available, otherwise overflow.
    /// Codes should be in row-major format; this method handles transpose.
    ///
    /// # Arguments
    ///
    /// * `row_major_codes` - Row-major PQ codes (as returned by `pq.quantize()`)
    /// * `positions` - Row positions for each vector
    pub fn append_batch(&self, row_major_codes: &[u8], positions: &[u64]) {
        let num_vectors = positions.len();
        if num_vectors == 0 {
            return;
        }

        debug_assert_eq!(
            row_major_codes.len(),
            num_vectors * self.num_sub_vectors,
            "row_major_codes length mismatch"
        );

        let primary_remaining = self.primary.remaining_capacity();

        if primary_remaining >= num_vectors {
            // All fit in primary - transpose and append.
            let codes_array = UInt8Array::from(row_major_codes.to_vec());
            let transposed =
                transpose::<UInt8Type>(&codes_array, num_vectors, self.num_sub_vectors);
            let _ = self
                .primary
                .append_transposed_batch(transposed.values(), positions);
        } else if primary_remaining > 0 {
            // Split: some go to primary, rest to overflow.
            let primary_count = primary_remaining;

            // Primary portion - transpose and append.
            let primary_codes = &row_major_codes[..primary_count * self.num_sub_vectors];
            let primary_positions = &positions[..primary_count];
            let codes_array = UInt8Array::from(primary_codes.to_vec());
            let transposed =
                transpose::<UInt8Type>(&codes_array, primary_count, self.num_sub_vectors);
            let _ = self
                .primary
                .append_transposed_batch(transposed.values(), primary_positions);

            // Overflow portion - store row-major.
            let overflow_count = num_vectors - primary_count;
            for i in 0..overflow_count {
                let idx = primary_count + i;
                let code_start = idx * self.num_sub_vectors;
                let code_end = code_start + self.num_sub_vectors;
                let code = row_major_codes[code_start..code_end].to_vec();
                self.overflow.insert(positions[idx], code);
            }
            self.overflow_count
                .fetch_add(overflow_count, Ordering::Relaxed);
        } else {
            // Primary full - all go to overflow.
            for (i, &pos) in positions.iter().enumerate() {
                let code_start = i * self.num_sub_vectors;
                let code_end = code_start + self.num_sub_vectors;
                let code = row_major_codes[code_start..code_end].to_vec();
                self.overflow.insert(pos, code);
            }
            self.overflow_count
                .fetch_add(num_vectors, Ordering::Relaxed);
        }
    }

    /// Check if this partition has overflow data.
    #[inline]
    pub fn has_overflow(&self) -> bool {
        self.overflow_count.load(Ordering::Relaxed) > 0
    }

    /// Total vectors in this partition.
    #[inline]
    pub fn len(&self) -> usize {
        self.primary.len() + self.overflow_count.load(Ordering::Relaxed)
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get primary codes for search (pre-transposed, fast).
    ///
    /// Returns (codes, positions) where codes are column-major.
    pub fn get_primary_codes_for_search(&self) -> (Vec<u8>, Vec<u64>) {
        self.primary.get_codes_for_search()
    }

    /// Get overflow codes for search.
    ///
    /// Returns (row_major_codes, positions). Caller must transpose before distance computation.
    pub fn get_overflow_codes_for_search(&self) -> (Vec<u8>, Vec<u64>) {
        let overflow_count = self.overflow_count.load(Ordering::Acquire);
        if overflow_count == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut codes = Vec::with_capacity(overflow_count * self.num_sub_vectors);
        let mut positions = Vec::with_capacity(overflow_count);

        for entry in self.overflow.iter() {
            positions.push(*entry.key());
            codes.extend_from_slice(entry.value());
        }

        (codes, positions)
    }
}

// ============================================================================
// IVF-PQ Memory Index
// ============================================================================

/// In-memory IVF-PQ index entry.
///
/// Stores partition assignment and PQ codes for each vector.
#[derive(Debug, Clone)]
pub struct IvfPqEntry {
    /// Row position in MemTable.
    pub row_position: RowPosition,
    /// PQ code for this vector (compressed representation).
    /// Length = num_sub_vectors (for 8-bit) or num_sub_vectors/2 (for 4-bit).
    pub pq_code: Vec<u8>,
}

/// In-memory IVF-PQ index for vector similarity search.
///
/// Reuses IVF centroids and PQ codebook from the base table to ensure
/// distance comparisons are consistent between the in-memory and base table indexes.
///
/// Uses hybrid storage for optimal performance:
/// - **Primary**: Pre-allocated `IvfPqMemPartition` stores with pre-transposed codes (fast search)
/// - **Overflow**: SkipMap fallback when primary is full (row-major, transpose at search)
///
/// This design ensures writes never block while optimizing the common case where
/// most data (typically 95%+) fits in the fast primary storage.
#[derive(Debug)]
pub struct IvfPqMemIndex {
    /// Field ID this index is built on.
    field_id: i32,
    /// Column name (for Arrow batch lookups).
    column_name: String,
    ivf_model: IvfModel,
    pq: ProductQuantizer,
    /// Per-partition stores with hybrid storage (primary + overflow).
    partitions: Vec<IvfPqMemPartition>,
    /// Total number of vectors indexed.
    vector_count: AtomicUsize,
    /// Distance type for partition assignment.
    distance_type: DistanceType,
    /// Number of partitions.
    num_partitions: usize,
    /// PQ code length per vector (num_sub_vectors for 8-bit, num_sub_vectors/2 for 4-bit).
    code_len: usize,
}

/// Default partition capacity when not specified.
/// This is a fallback - in practice, capacity should always be calculated
/// from memtable config using the safety factor.
const DEFAULT_PARTITION_CAPACITY: usize = 1024;

impl IvfPqMemIndex {
    /// Create a new IVF-PQ index with centroids and codebook from base table.
    ///
    /// Uses default partition capacity. For production use, prefer `with_capacity()`
    /// with capacity calculated from memtable config.
    ///
    /// # Arguments
    ///
    /// * `field_id` - Field ID the index is built on
    /// * `column_name` - Vector column name
    /// * `ivf_model` - IVF model with centroids from base table
    /// * `pq` - Product quantizer with codebook from base table
    /// * `distance_type` - Distance type for search
    pub fn new(
        field_id: i32,
        column_name: String,
        ivf_model: IvfModel,
        pq: ProductQuantizer,
        distance_type: DistanceType,
    ) -> Self {
        Self::with_capacity(
            field_id,
            column_name,
            ivf_model,
            pq,
            distance_type,
            DEFAULT_PARTITION_CAPACITY,
        )
    }

    /// Create a new IVF-PQ index with specified partition capacity.
    ///
    /// The partition capacity determines how many vectors each partition's
    /// primary storage can hold before overflowing to the slower SkipMap.
    ///
    /// # Arguments
    ///
    /// * `field_id` - Field ID the index is built on
    /// * `column_name` - Vector column name
    /// * `ivf_model` - IVF model with centroids from base table
    /// * `pq` - Product quantizer with codebook from base table
    /// * `distance_type` - Distance type for search
    /// * `partition_capacity` - Max vectors per partition in primary storage
    pub fn with_capacity(
        field_id: i32,
        column_name: String,
        ivf_model: IvfModel,
        pq: ProductQuantizer,
        distance_type: DistanceType,
        partition_capacity: usize,
    ) -> Self {
        let num_partitions = ivf_model.num_partitions();
        let code_len = pq.num_sub_vectors * pq.num_bits as usize / 8;

        // Pre-allocate all partition stores.
        let partitions: Vec<_> = (0..num_partitions)
            .map(|_| IvfPqMemPartition::new(partition_capacity, code_len))
            .collect();

        Self {
            field_id,
            column_name,
            ivf_model,
            pq,
            partitions,
            vector_count: AtomicUsize::new(0),
            distance_type,
            num_partitions,
            code_len,
        }
    }

    /// Get the field ID this index is built on.
    pub fn field_id(&self) -> i32 {
        self.field_id
    }

    /// Insert vectors from a batch into the index.
    ///
    /// For better performance with multiple batches, prefer `insert_batches()`
    /// which enables cross-batch vectorization.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        let col_idx = batch
            .schema()
            .column_with_name(&self.column_name)
            .map(|(idx, _)| idx);

        let Some(col_idx) = col_idx else {
            // Column not in this batch, skip
            return Ok(());
        };

        let column = batch.column(col_idx);
        let fsl = column.as_fixed_size_list_opt().ok_or_else(|| {
            Error::invalid_input(
                format!(
                    "Column '{}' is not a FixedSizeList, got {:?}",
                    self.column_name,
                    column.data_type()
                ),
                location!(),
            )
        })?;

        // Find partition assignments for all vectors using batch computation
        let centroids = self
            .ivf_model
            .centroids
            .as_ref()
            .ok_or_else(|| Error::invalid_input("IVF model has no centroids", location!()))?;
        let (partition_ids, _distances) =
            compute_partitions_arrow_array(centroids, fsl, self.distance_type)?;

        // Compute PQ codes for all vectors (row-major output)
        let pq_codes = self.pq.quantize(fsl)?;
        let pq_codes_fsl = pq_codes.as_fixed_size_list();
        let pq_codes_flat = pq_codes_fsl
            .values()
            .as_primitive::<arrow_array::types::UInt8Type>();

        // Group vectors by partition
        let mut partition_groups: Vec<Vec<usize>> = vec![Vec::new(); self.num_partitions];
        for (row_idx, partition_id) in partition_ids.iter().enumerate().take(batch.num_rows()) {
            if let Some(pid) = partition_id {
                if (*pid as usize) < self.num_partitions {
                    partition_groups[*pid as usize].push(row_idx);
                }
            }
        }

        // For each partition: gather codes and append
        let mut total_inserted = 0usize;

        for (partition_id, indices) in partition_groups.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }

            let num_vectors = indices.len();

            // Gather row-major codes for this partition
            let mut partition_codes: Vec<u8> = Vec::with_capacity(num_vectors * self.code_len);
            let mut partition_positions: Vec<u64> = Vec::with_capacity(num_vectors);

            for &row_idx in indices {
                let code_start = row_idx * self.code_len;
                let code_end = code_start + self.code_len;
                partition_codes.extend_from_slice(&pq_codes_flat.values()[code_start..code_end]);
                partition_positions.push(row_offset + row_idx as u64);
            }

            // Append to partition (handles primary vs overflow internally)
            self.partitions[partition_id].append_batch(&partition_codes, &partition_positions);

            total_inserted += num_vectors;
        }

        self.vector_count
            .fetch_add(total_inserted, Ordering::Relaxed);

        Ok(())
    }

    /// Insert vectors from multiple batches with cross-batch vectorization.
    ///
    /// This method concatenates vectors from all batches and processes them
    /// together for better SIMD utilization in partition assignment and PQ encoding.
    /// Vectors are stored in the partition's primary (pre-transposed) storage when
    /// capacity allows, otherwise in the overflow SkipMap.
    pub fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }

        // Collect vector arrays and track batch boundaries
        let mut vector_arrays: Vec<&FixedSizeListArray> = Vec::with_capacity(batches.len());
        let mut batch_infos: Vec<(u64, usize, usize)> = Vec::with_capacity(batches.len());

        for stored in batches {
            let col_idx = stored
                .data
                .schema()
                .column_with_name(&self.column_name)
                .map(|(idx, _)| idx);

            if let Some(col_idx) = col_idx {
                let column = stored.data.column(col_idx);
                if let Some(fsl) = column.as_fixed_size_list_opt() {
                    let num_vectors = fsl.len();
                    if num_vectors > 0 {
                        vector_arrays.push(fsl);
                        batch_infos.push((stored.row_offset, num_vectors, stored.batch_position));
                    }
                }
            }
        }

        if vector_arrays.is_empty() {
            return Ok(());
        }

        // Concatenate all vectors into a single array for vectorized processing
        let arrays_as_refs: Vec<&dyn Array> =
            vector_arrays.iter().map(|a| *a as &dyn Array).collect();
        let concatenated = arrow_select::concat::concat(&arrays_as_refs)?;
        let mega_fsl = concatenated.as_fixed_size_list();
        let total_vectors = mega_fsl.len();

        // Batch compute partition assignments (SIMD-optimized)
        let centroids = self
            .ivf_model
            .centroids
            .as_ref()
            .ok_or_else(|| Error::invalid_input("IVF model has no centroids", location!()))?;
        let (partition_ids, _distances) =
            compute_partitions_arrow_array(centroids, mega_fsl, self.distance_type)?;

        // Batch compute PQ codes (SIMD-optimized, row-major output)
        let pq_codes = self.pq.quantize(mega_fsl)?;
        let pq_codes_fsl = pq_codes.as_fixed_size_list();
        let pq_codes_flat = pq_codes_fsl
            .values()
            .as_primitive::<arrow_array::types::UInt8Type>();

        // Build row position mapping
        let mut row_positions: Vec<u64> = Vec::with_capacity(total_vectors);
        for (row_offset, num_vectors, _) in &batch_infos {
            for i in 0..*num_vectors {
                row_positions.push(row_offset + i as u64);
            }
        }

        // Group vectors by partition
        let mut partition_groups: Vec<Vec<usize>> = vec![Vec::new(); self.num_partitions];
        for (idx, pid) in partition_ids.iter().enumerate() {
            if let Some(pid) = pid {
                if (*pid as usize) < self.num_partitions {
                    partition_groups[*pid as usize].push(idx);
                }
            }
        }

        // For each partition: gather codes and append
        let mut total_inserted = 0usize;

        for (partition_id, indices) in partition_groups.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }

            let num_vectors = indices.len();

            // Gather row-major codes for this partition
            let mut partition_codes: Vec<u8> = Vec::with_capacity(num_vectors * self.code_len);
            let mut partition_positions: Vec<u64> = Vec::with_capacity(num_vectors);

            for &idx in indices {
                let code_start = idx * self.code_len;
                let code_end = code_start + self.code_len;
                partition_codes.extend_from_slice(&pq_codes_flat.values()[code_start..code_end]);
                partition_positions.push(row_positions[idx]);
            }

            // Append to partition (handles primary vs overflow internally)
            self.partitions[partition_id].append_batch(&partition_codes, &partition_positions);

            total_inserted += num_vectors;
        }

        self.vector_count
            .fetch_add(total_inserted, Ordering::Relaxed);

        Ok(())
    }

    /// Search for nearest neighbors with visibility filtering.
    ///
    /// Searches both primary (pre-transposed, fast) and overflow (needs transpose)
    /// storage and merges results. Only returns rows where `row_position <= max_row_position`.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector as FixedSizeListArray with single vector
    /// * `k` - Number of results to return
    /// * `nprobes` - Number of partitions to search
    /// * `max_row_position` - Maximum visible row position (for MVCC filtering)
    ///
    /// # Returns
    ///
    /// Vec of (distance, row_position) sorted by distance ascending.
    pub fn search(
        &self,
        query: &FixedSizeListArray,
        k: usize,
        nprobes: usize,
        max_row_position: RowPosition,
    ) -> Result<Vec<(f32, RowPosition)>> {
        if query.len() != 1 {
            return Err(Error::invalid_input(
                format!("Query must have exactly 1 vector, got {}", query.len()),
                location!(),
            ));
        }

        // Find nearest partitions to probe
        let query_values = query.value(0);
        let (partition_ids, _) =
            self.ivf_model
                .find_partitions(&query_values, nprobes, self.distance_type)?;

        let mut results: Vec<(f32, RowPosition)> = Vec::new();

        for i in 0..partition_ids.len() {
            let partition_id = partition_ids.value(i) as usize;
            if partition_id >= self.num_partitions {
                continue;
            }

            let partition = &self.partitions[partition_id];
            if partition.is_empty() {
                continue;
            }

            // Search primary storage (pre-transposed, fast path)
            let (primary_codes, primary_positions) = partition.get_primary_codes_for_search();
            if !primary_codes.is_empty() {
                let codes_array = UInt8Array::from(primary_codes);
                let distances = self.pq.compute_distances(&query_values, &codes_array)?;

                for (idx, &dist) in distances.values().iter().enumerate() {
                    let pos = primary_positions[idx];
                    if pos <= max_row_position {
                        results.push((dist, pos));
                    }
                }
            }

            // Search overflow storage (needs transpose)
            if partition.has_overflow() {
                let (overflow_codes_rowmajor, overflow_positions) =
                    partition.get_overflow_codes_for_search();

                if !overflow_codes_rowmajor.is_empty() {
                    let num_overflow = overflow_positions.len();

                    // Transpose to column-major for distance computation
                    let codes_array = UInt8Array::from(overflow_codes_rowmajor);
                    let transposed = transpose::<arrow_array::types::UInt8Type>(
                        &codes_array,
                        num_overflow,
                        self.code_len,
                    );
                    let distances = self.pq.compute_distances(&query_values, &transposed)?;

                    for (idx, &dist) in distances.values().iter().enumerate() {
                        let pos = overflow_positions[idx];
                        if pos <= max_row_position {
                            results.push((dist, pos));
                        }
                    }
                }
            }
        }

        // Sort by distance and take top-k
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Get total vector count.
    pub fn len(&self) -> usize {
        self.vector_count.load(Ordering::Relaxed)
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vector_count.load(Ordering::Relaxed) == 0
    }

    /// Get the column name.
    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    /// Get entries for a partition.
    /// Returns PQ codes in row-major format.
    pub fn get_partition(&self, partition_id: usize) -> Vec<IvfPqEntry> {
        if partition_id >= self.num_partitions {
            return Vec::new();
        }

        let partition = &self.partitions[partition_id];
        let mut entries = Vec::with_capacity(partition.len());

        // Get from primary storage (need to convert from column-major to row-major)
        let (primary_codes, primary_positions) = partition.get_primary_codes_for_search();
        if !primary_codes.is_empty() {
            let num_vectors = primary_positions.len();
            // primary_codes are column-major, need to transpose back to row-major
            for (i, &row_position) in primary_positions.iter().enumerate() {
                let mut pq_code = Vec::with_capacity(self.code_len);
                for sv in 0..self.code_len {
                    pq_code.push(primary_codes[sv * num_vectors + i]);
                }
                entries.push(IvfPqEntry {
                    row_position,
                    pq_code,
                });
            }
        }

        // Get from overflow storage (already row-major)
        let (overflow_codes, overflow_positions) = partition.get_overflow_codes_for_search();
        for (i, &row_position) in overflow_positions.iter().enumerate() {
            let code_start = i * self.code_len;
            let code_end = code_start + self.code_len;
            entries.push(IvfPqEntry {
                row_position,
                pq_code: overflow_codes[code_start..code_end].to_vec(),
            });
        }

        entries
    }

    /// Get the number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.ivf_model.num_partitions()
    }

    /// Get the IVF model (for advanced use).
    pub fn ivf_model(&self) -> &IvfModel {
        &self.ivf_model
    }

    /// Get the product quantizer (for advanced use).
    pub fn pq(&self) -> &ProductQuantizer {
        &self.pq
    }

    /// Get the distance type.
    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    /// Export partition data as RecordBatches for index creation.
    /// Each batch has schema: `_rowid` (UInt64), `__pq_code` (FixedSizeList<UInt8>).
    ///
    /// The PQ codes are stored row-major (not transposed), matching the format
    /// expected by the index builder's shuffle stage.
    pub fn to_partition_batches(&self) -> Result<Vec<(usize, RecordBatch)>> {
        use arrow_array::UInt64Array;
        use arrow_schema::{Field, Schema};
        use lance_core::ROW_ID;
        use lance_index::vector::PQ_CODE_COLUMN;
        use std::sync::Arc;

        let pq_code_len = self.pq.num_sub_vectors * self.pq.num_bits as usize / 8;

        // Schema for partition data: row_id and pq_code
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            Field::new(
                PQ_CODE_COLUMN,
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(Field::new("item", arrow_schema::DataType::UInt8, false)),
                    pq_code_len as i32,
                ),
                false,
            ),
        ]));

        let mut result = Vec::new();

        for part_id in 0..self.num_partitions {
            let entries = self.get_partition(part_id);
            if entries.is_empty() {
                continue;
            }

            // Collect row IDs
            let row_ids: Vec<u64> = entries.iter().map(|e| e.row_position).collect();
            let row_id_array = Arc::new(UInt64Array::from(row_ids));

            // Collect PQ codes into a flat array
            let mut pq_codes_flat: Vec<u8> = Vec::with_capacity(entries.len() * pq_code_len);
            for entry in &entries {
                pq_codes_flat.extend_from_slice(&entry.pq_code);
            }

            // Create FixedSizeList array for PQ codes with non-nullable inner field
            let pq_codes_array = UInt8Array::from(pq_codes_flat);
            let inner_field = Arc::new(Field::new("item", arrow_schema::DataType::UInt8, false));
            let pq_codes_fsl = Arc::new(
                FixedSizeListArray::try_new(
                    inner_field,
                    pq_code_len as i32,
                    Arc::new(pq_codes_array),
                    None,
                )
                .map_err(|e| {
                    Error::io(
                        format!("Failed to create PQ code array: {}", e),
                        location!(),
                    )
                })?,
            );

            let batch = RecordBatch::try_new(schema.clone(), vec![row_id_array, pq_codes_fsl])
                .map_err(|e| {
                    Error::io(
                        format!("Failed to create partition batch: {}", e),
                        location!(),
                    )
                })?;

            result.push((part_id, batch));
        }

        Ok(result)
    }
}

/// Configuration for an IVF-PQ vector index.
///
/// Contains the centroids and codebook from the base table
/// to ensure consistent distance computations.
#[derive(Debug, Clone)]
pub struct IvfPqIndexConfig {
    /// Index name.
    pub name: String,
    /// Field ID the index is built on.
    pub field_id: i32,
    /// Column name (for Arrow batch lookups).
    pub column: String,
    /// IVF model with centroids from base table.
    pub ivf_model: IvfModel,
    /// Product quantizer with codebook from base table.
    pub pq: ProductQuantizer,
    /// Distance type for search.
    pub distance_type: DistanceType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_store_append_transposed() {
        let store = ColumnMajorIvfPqMemPartition::new(100, 4);

        // Append 3 vectors with 4 sub-vectors each.
        // Transposed layout: [sv0_v0, sv0_v1, sv0_v2, sv1_v0, sv1_v1, sv1_v2, ...]
        let transposed_codes = vec![
            // SubVec 0
            10, 20, 30, // SubVec 1
            11, 21, 31, // SubVec 2
            12, 22, 32, // SubVec 3
            13, 23, 33,
        ];
        let positions = vec![100, 200, 300];

        store
            .append_transposed_batch(&transposed_codes, &positions)
            .unwrap();

        assert_eq!(store.len(), 3);
        assert_eq!(store.remaining_capacity(), 97);

        let (codes, pos) = store.get_codes_for_search();
        assert_eq!(pos, vec![100, 200, 300]);
        assert_eq!(codes, transposed_codes);
    }

    #[test]
    fn test_partition_store_full() {
        let store = ColumnMajorIvfPqMemPartition::new(2, 4);

        // First batch - fills capacity.
        let codes1 = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 vectors transposed
        let pos1 = vec![10, 20];
        store.append_transposed_batch(&codes1, &pos1).unwrap();

        assert_eq!(store.remaining_capacity(), 0);

        // Should fail - no capacity left.
        let codes2 = vec![9, 10, 11, 12];
        let pos2 = vec![30];
        assert!(store.append_transposed_batch(&codes2, &pos2).is_err());
    }

    #[test]
    fn test_ivfpq_partition_primary_only() {
        let partition = IvfPqMemPartition::new(100, 4);

        // Row-major codes for 3 vectors.
        let row_major = vec![
            10, 11, 12, 13, // vec 0
            20, 21, 22, 23, // vec 1
            30, 31, 32, 33, // vec 2
        ];
        let positions = vec![100, 200, 300];

        partition.append_batch(&row_major, &positions);

        assert_eq!(partition.len(), 3);
        assert!(!partition.has_overflow());

        let (codes, pos) = partition.get_primary_codes_for_search();
        assert_eq!(pos, vec![100, 200, 300]);
        // Codes should be transposed.
        assert_eq!(
            codes,
            vec![
                10, 20, 30, // sv0
                11, 21, 31, // sv1
                12, 22, 32, // sv2
                13, 23, 33, // sv3
            ]
        );
    }

    #[test]
    fn test_ivfpq_partition_overflow() {
        let partition = IvfPqMemPartition::new(2, 4); // Only 2 slots in primary.

        // Insert 4 vectors - 2 should go to primary, 2 to overflow.
        let row_major = vec![
            10, 11, 12, 13, // vec 0 -> primary
            20, 21, 22, 23, // vec 1 -> primary
            30, 31, 32, 33, // vec 2 -> overflow
            40, 41, 42, 43, // vec 3 -> overflow
        ];
        let positions = vec![100, 200, 300, 400];

        partition.append_batch(&row_major, &positions);

        assert_eq!(partition.len(), 4);
        assert!(partition.has_overflow());

        // Check primary (2 vectors, transposed).
        let (primary_codes, primary_pos) = partition.get_primary_codes_for_search();
        assert_eq!(primary_pos, vec![100, 200]);
        assert_eq!(
            primary_codes,
            vec![
                10, 20, // sv0
                11, 21, // sv1
                12, 22, // sv2
                13, 23, // sv3
            ]
        );

        // Check overflow (2 vectors, row-major).
        let (overflow_codes, overflow_pos) = partition.get_overflow_codes_for_search();
        assert_eq!(overflow_pos.len(), 2);
        assert!(overflow_pos.contains(&300));
        assert!(overflow_pos.contains(&400));
        assert_eq!(overflow_codes.len(), 8);
    }

    #[test]
    fn test_ivfpq_partition_all_overflow() {
        let partition = IvfPqMemPartition::new(2, 4);

        // Fill primary first.
        let batch1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        partition.append_batch(&batch1, &[10, 20]);
        assert!(!partition.has_overflow());

        // This batch should all go to overflow.
        let batch2 = vec![11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34];
        partition.append_batch(&batch2, &[30, 40, 50]);

        assert_eq!(partition.len(), 5);
        assert!(partition.has_overflow());
    }
}
