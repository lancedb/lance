// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory HNSW index for vector similarity search.
//!
//! Builds an HNSW graph on the fly while a MemTable is receiving writes.
//! Searches see all nodes whose insert has fully completed. At MemTable
//! flush time the graph is converted to the on-disk Lance HNSW format.
//!
//! # Architecture
//!
//! ```text
//! HnswMemIndex
//! ├── MemHnswStorage      lock-free flat-float vector store (single-writer)
//! └── OnlineHnswBuilder   incremental HNSW graph builder (lance-index)
//! ```
//!
//! Both structures are pre-allocated to `max_memtable_rows + slack` so writes
//! never need to grow shared structures concurrently.

#![allow(clippy::type_complexity)]

use std::any::Any;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::flat::storage::FLAT_COLUMN;
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::hnsw::{HNSW, OnlineHnswBuilder, builder::HnswBuildParams};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::{DistanceType, Dot, L2};

use super::super::memtable::batch_store::StoredBatch;

pub use super::RowPosition;

const MEM_HNSW_DIM_PLACEHOLDER: usize = 0;

/// Lock-free flat-float vector storage for in-memory HNSW.
///
/// # Concurrency
///
/// - Single writer (the MemTable's WAL flush handler thread).
/// - Multiple concurrent readers (queries and the HNSW search algorithms).
///
/// `vectors` and `row_positions` use `UnsafeCell` for interior mutation; the
/// writer publishes new committed entries by bumping `committed_len` with a
/// release store. Readers acquire-load `committed_len` and only read indices
/// `< committed_len`.
pub struct MemHnswStorage {
    /// Vector data, layout `[v0, v1, ..., v_{capacity - 1}]` with each
    /// `v_i` being `dim` consecutive `f32` slots. Total size = `capacity * dim`.
    vectors: UnsafeCell<Box<[MaybeUninit<f32>]>>,
    /// Row positions in the MemTable; one per slot, total size = `capacity`.
    row_positions: UnsafeCell<Box<[MaybeUninit<u64>]>>,
    capacity: usize,
    dim: usize,
    distance_type: DistanceType,
    /// Number of committed vectors. Reads must use Acquire; writes use Release.
    committed_len: Arc<AtomicUsize>,
    /// Schema cached for the `VectorStore` impl.
    schema: SchemaRef,
}

// SAFETY: `MemHnswStorage` follows a single-writer multi-reader model. The
// writer is the only mutator of the underlying `UnsafeCell` buffers; readers
// only access indices `< committed_len`, and `committed_len` is published
// with `Release` ordering so readers see initialized data.
unsafe impl Sync for MemHnswStorage {}
unsafe impl Send for MemHnswStorage {}

impl MemHnswStorage {
    /// Create a storage pre-allocated for `capacity` vectors of `dim` floats.
    pub fn with_capacity(capacity: usize, dim: usize, distance_type: DistanceType) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(dim > 0, "dim must be > 0");

        let mut vectors = Vec::with_capacity(capacity * dim);
        for _ in 0..capacity * dim {
            vectors.push(MaybeUninit::uninit());
        }
        let mut row_positions = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            row_positions.push(MaybeUninit::uninit());
        }

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                FLAT_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));

        Self {
            vectors: UnsafeCell::new(vectors.into_boxed_slice()),
            row_positions: UnsafeCell::new(row_positions.into_boxed_slice()),
            capacity,
            dim,
            distance_type,
            committed_len: Arc::new(AtomicUsize::new(0)),
            schema,
        }
    }

    /// Returns the committed vector count visible to readers.
    pub fn committed_len(&self) -> usize {
        self.committed_len.load(Ordering::Acquire)
    }

    /// Append a single vector + row position. Single-writer only.
    ///
    /// Returns the assigned id (=position in the storage). Returns `Err` if
    /// the storage is full.
    pub fn append(&self, vector: &[f32], row_position: u64) -> Result<u32> {
        if vector.len() != self.dim {
            return Err(Error::invalid_input(format!(
                "vector dim mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        let id = self.committed_len.load(Ordering::Relaxed);
        if id >= self.capacity {
            return Err(Error::invalid_input(format!(
                "MemHnswStorage capacity {} exhausted",
                self.capacity
            )));
        }

        // SAFETY: single writer, id < capacity.
        unsafe {
            let vectors = &mut *self.vectors.get();
            let base = id * self.dim;
            for (i, &v) in vector.iter().enumerate() {
                vectors[base + i].write(v);
            }
            let row_positions = &mut *self.row_positions.get();
            row_positions[id].write(row_position);
        }

        // Release publishes the writes above to readers.
        self.committed_len.store(id + 1, Ordering::Release);
        Ok(id as u32)
    }

    /// Get the row position for a committed id.
    pub fn row_position(&self, id: u32) -> u64 {
        debug_assert!((id as usize) < self.committed_len.load(Ordering::Acquire));
        // SAFETY: id < committed_len => initialized.
        unsafe { (*self.row_positions.get())[id as usize].assume_init() }
    }

    /// Get a slice view of the vector at id `id`. Lifetime is tied to the
    /// storage; the underlying memory is stable for the life of the storage.
    pub fn vector_slice(&self, id: u32) -> &[f32] {
        debug_assert!((id as usize) < self.committed_len.load(Ordering::Acquire));
        // SAFETY: id < committed_len => initialized; storage is single-writer,
        // and after Release on committed_len readers can safely read these
        // bytes; no further writer touches them (the writer only writes new
        // slots, never overwrites committed slots).
        unsafe {
            let vectors = &*self.vectors.get();
            let base = (id as usize) * self.dim;
            std::slice::from_raw_parts(vectors.as_ptr().add(base) as *const f32, self.dim)
        }
    }

    /// Materialize all committed vectors as a `RecordBatch` for serialization.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let len = self.committed_len();
        if len == 0 {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }
        let mut row_ids = Vec::with_capacity(len);
        let mut flat_values: Vec<f32> = Vec::with_capacity(len * self.dim);
        for id in 0..len {
            row_ids.push(self.row_position(id as u32));
            flat_values.extend_from_slice(self.vector_slice(id as u32));
        }
        let row_id_array = Arc::new(UInt64Array::from(row_ids));
        let flat_inner = Arc::new(Float32Array::from(flat_values));
        let flat_field = Arc::new(Field::new("item", DataType::Float32, true));
        let flat_array = Arc::new(FixedSizeListArray::try_new(
            flat_field,
            self.dim as i32,
            flat_inner,
            None,
        )?);
        Ok(RecordBatch::try_new(
            self.schema.clone(),
            vec![row_id_array, flat_array],
        )?)
    }

    /// Materialize as a `RecordBatch` with row positions reversed:
    /// `reversed_position = total_rows - original_position - 1`.
    /// Used at flush time when the data file is written in reverse order.
    pub fn to_record_batch_reversed(&self, total_rows: u64) -> Result<RecordBatch> {
        let len = self.committed_len();
        if len == 0 {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }
        let mut row_ids = Vec::with_capacity(len);
        let mut flat_values: Vec<f32> = Vec::with_capacity(len * self.dim);
        for id in 0..len {
            row_ids.push(total_rows - self.row_position(id as u32) - 1);
            flat_values.extend_from_slice(self.vector_slice(id as u32));
        }
        let row_id_array = Arc::new(UInt64Array::from(row_ids));
        let flat_inner = Arc::new(Float32Array::from(flat_values));
        let flat_field = Arc::new(Field::new("item", DataType::Float32, true));
        let flat_array = Arc::new(FixedSizeListArray::try_new(
            flat_field,
            self.dim as i32,
            flat_inner,
            None,
        )?);
        Ok(RecordBatch::try_new(
            self.schema.clone(),
            vec![row_id_array, flat_array],
        )?)
    }
}

/// A snapshot view of `MemHnswStorage` exposing `VectorStore` semantics.
///
/// Cloning the snapshot is cheap (`Arc::clone`). The snapshot's `len()` is
/// fixed at construction so distance calculators have a stable upper bound.
#[derive(Clone)]
pub struct MemHnswStorageView {
    storage: Arc<MemHnswStorage>,
    /// Snapshot of committed length at view construction time.
    visible_len: usize,
}

impl MemHnswStorage {
    /// Build a `VectorStore`-implementing snapshot view of the current
    /// committed contents.
    pub fn snapshot(self: &Arc<Self>) -> MemHnswStorageView {
        let visible_len = self.committed_len();
        MemHnswStorageView {
            storage: self.clone(),
            visible_len,
        }
    }
}

impl MemHnswStorageView {
    fn vector_slice(&self, id: u32) -> &[f32] {
        debug_assert!((id as usize) < self.visible_len);
        self.storage.vector_slice(id)
    }

    fn row_pos(&self, id: u32) -> u64 {
        debug_assert!((id as usize) < self.visible_len);
        self.storage.row_position(id)
    }
}

impl VectorStore for MemHnswStorageView {
    type DistanceCalculator<'a> = MemHnswDistCalc<'a>;

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.storage.schema
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send> {
        // Required by the trait but not on the hot path. Materialize from the
        // visible range.
        let mut row_ids = Vec::with_capacity(self.visible_len);
        let mut flat_values: Vec<f32> = Vec::with_capacity(self.visible_len * self.storage.dim);
        for id in 0..self.visible_len as u32 {
            row_ids.push(self.row_pos(id));
            flat_values.extend_from_slice(self.vector_slice(id));
        }
        let row_id_array = Arc::new(UInt64Array::from(row_ids)) as ArrayRef;
        let flat_inner = Arc::new(Float32Array::from(flat_values)) as ArrayRef;
        let flat_field = Arc::new(Field::new("item", DataType::Float32, true));
        let flat_array = Arc::new(FixedSizeListArray::try_new(
            flat_field,
            self.storage.dim as i32,
            flat_inner,
            None,
        )?) as ArrayRef;
        let batch =
            RecordBatch::try_new(self.storage.schema.clone(), vec![row_id_array, flat_array])?;
        Ok(std::iter::once(batch))
    }

    fn append_batch(&self, _batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        Err(Error::invalid_input(
            "MemHnswStorageView is read-only; append goes through MemHnswStorage::append",
        ))
    }

    fn len(&self) -> usize {
        self.visible_len
    }

    fn distance_type(&self) -> DistanceType {
        self.storage.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_pos(id)
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        // SAFETY: visible_len <= storage.committed_len at snapshot time; the
        // first `visible_len` slots are initialized and stable for the life of
        // the storage (single writer never overwrites committed entries).
        let storage = &self.storage;
        let slice: &[u64] = unsafe {
            std::slice::from_raw_parts(
                (*storage.row_positions.get()).as_ptr() as *const u64,
                self.visible_len,
            )
        };
        slice.iter()
    }

    fn dist_calculator(&self, query: ArrayRef, _dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        MemHnswDistCalc::new_for_query(self, query)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        MemHnswDistCalc::new_for_id(self, id)
    }
}

/// Distance calculator that operates over `MemHnswStorageView`'s f32 buffers.
pub struct MemHnswDistCalc<'a> {
    view: &'a MemHnswStorageView,
    query: Vec<f32>,
}

impl<'a> MemHnswDistCalc<'a> {
    fn new_for_query(view: &'a MemHnswStorageView, query: ArrayRef) -> Self {
        // The query may arrive as a single FixedSizeListArray (single vector)
        // or as a flat Float32Array. Accept either.
        let query_vec = if let Some(fsl) = query.as_fixed_size_list_opt() {
            fsl.values().as_primitive::<Float32Type>().values().to_vec()
        } else {
            query.as_primitive::<Float32Type>().values().to_vec()
        };
        Self {
            view,
            query: query_vec,
        }
    }

    fn new_for_id(view: &'a MemHnswStorageView, id: u32) -> Self {
        Self {
            view,
            query: view.vector_slice(id).to_vec(),
        }
    }
}

impl DistCalculator for MemHnswDistCalc<'_> {
    fn distance(&self, id: u32) -> f32 {
        let v = self.view.vector_slice(id);
        compute_distance(&self.query, v, self.view.storage.distance_type)
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.view.visible_len);
        for id in 0..self.view.visible_len as u32 {
            let v = self.view.vector_slice(id);
            out.push(compute_distance(
                &self.query,
                v,
                self.view.storage.distance_type,
            ));
        }
        out
    }

    fn prefetch(&self, _id: u32) {
        // Optional: could prefetch the f32 slice. Not implemented for now.
    }
}

fn compute_distance(query: &[f32], vector: &[f32], distance_type: DistanceType) -> f32 {
    match distance_type {
        DistanceType::L2 => f32::l2(query, vector),
        DistanceType::Cosine => {
            // Cosine on the primitive slice — match FlatFloatStorage's behavior.
            // FlatFloatStorage uses `distance_type.func()` which dispatches.
            let f = distance_type.func();
            f(query, vector)
        }
        DistanceType::Dot => f32::dot(query, vector),
        _ => {
            let f = distance_type.func();
            f(query, vector)
        }
    }
}

// ============================================================================
// HnswMemIndex
// ============================================================================

/// Configuration for an in-memory HNSW index.
#[derive(Debug, Clone)]
pub struct HnswIndexConfig {
    pub name: String,
    pub field_id: i32,
    /// Vector column name for batch lookups.
    pub column: String,
    pub distance_type: DistanceType,
    pub build_params: HnswBuildParams,
}

impl HnswIndexConfig {
    pub fn new(name: String, field_id: i32, column: String, distance_type: DistanceType) -> Self {
        Self {
            name,
            field_id,
            column,
            distance_type,
            build_params: HnswBuildParams::default(),
        }
    }

    pub fn with_build_params(mut self, params: HnswBuildParams) -> Self {
        self.build_params = params;
        self
    }
}

/// In-memory HNSW index queryable while building.
pub struct HnswMemIndex {
    field_id: i32,
    column: String,
    distance_type: DistanceType,
    /// Vector dimension (lazy-initialized on first insert).
    dim: AtomicUsize,
    /// Capacity (max vectors) — set at construction.
    capacity: usize,
    /// Build parameters (passed to the online builder once dim is known).
    build_params: HnswBuildParams,
    /// Lazily-initialized storage and builder. We initialize on first insert
    /// so we can derive `dim` from the data.
    state: std::sync::OnceLock<HnswState>,
}

struct HnswState {
    storage: Arc<MemHnswStorage>,
    builder: OnlineHnswBuilder,
}

impl std::fmt::Debug for HnswMemIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswMemIndex")
            .field("field_id", &self.field_id)
            .field("column", &self.column)
            .field("distance_type", &self.distance_type)
            .field("dim", &self.dim.load(Ordering::Acquire))
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish()
    }
}

impl HnswMemIndex {
    pub fn with_capacity(
        field_id: i32,
        column: String,
        distance_type: DistanceType,
        build_params: HnswBuildParams,
        capacity: usize,
    ) -> Self {
        Self {
            field_id,
            column,
            distance_type,
            dim: AtomicUsize::new(MEM_HNSW_DIM_PLACEHOLDER),
            capacity,
            build_params,
            state: std::sync::OnceLock::new(),
        }
    }

    pub fn field_id(&self) -> i32 {
        self.field_id
    }

    pub fn column_name(&self) -> &str {
        &self.column
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    pub fn build_params(&self) -> &HnswBuildParams {
        &self.build_params
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Vector dimension. Returns 0 before the first insert (dim is derived
    /// from the first incoming batch).
    pub fn dim(&self) -> usize {
        self.dim.load(Ordering::Acquire)
    }

    pub fn len(&self) -> usize {
        self.state.get().map(|s| s.builder.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ensure state is initialized once we know the vector dimension.
    fn ensure_state(&self, dim: usize) -> &HnswState {
        self.state.get_or_init(|| {
            self.dim.store(dim, Ordering::Release);
            let storage = Arc::new(MemHnswStorage::with_capacity(
                self.capacity,
                dim,
                self.distance_type,
            ));
            let builder =
                OnlineHnswBuilder::with_capacity(self.capacity, self.build_params.clone());
            HnswState { storage, builder }
        })
    }

    /// Insert vectors from a single batch.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        let Some((col_idx, _)) = batch.schema().column_with_name(&self.column) else {
            return Ok(());
        };
        let column = batch.column(col_idx);
        let fsl = column.as_fixed_size_list_opt().ok_or_else(|| {
            Error::invalid_input(format!(
                "Column '{}' is not a FixedSizeList, got {:?}",
                self.column,
                column.data_type()
            ))
        })?;
        if fsl.is_empty() {
            return Ok(());
        }

        let dim = fsl.value_length() as usize;
        let state = self.ensure_state(dim);
        // Snapshot of storage that the inserts will share. Storage's snapshot
        // sees the latest committed_len at the time of construction; we
        // rebuild snapshots after each append so the HNSW insert sees the
        // newly added vector.
        let values = fsl.values().as_primitive::<Float32Type>().values();
        for i in 0..fsl.len() {
            let row_position = row_offset + i as u64;
            let base = i * dim;
            let id = state
                .storage
                .append(&values[base..base + dim], row_position)?;
            // After append, snapshot includes id.
            let view = state.storage.snapshot();
            state.builder.insert(id, &view);
        }

        Ok(())
    }

    /// Insert vectors from multiple batches (cross-batch friendly).
    pub fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        for stored in batches {
            self.insert(&stored.data, stored.row_offset)?;
        }
        Ok(())
    }

    /// Search for the k nearest neighbors of `query` with MVCC visibility.
    ///
    /// Distances returned are exact (FLAT-backed). Only rows with
    /// `row_position <= max_row_position` are returned.
    pub fn search(
        &self,
        query: &FixedSizeListArray,
        k: usize,
        ef: Option<usize>,
        max_row_position: RowPosition,
    ) -> Result<Vec<(f32, RowPosition)>> {
        if query.len() != 1 {
            return Err(Error::invalid_input(format!(
                "Query must have exactly 1 vector, got {}",
                query.len()
            )));
        }
        let Some(state) = self.state.get() else {
            return Ok(Vec::new());
        };
        let view = state.storage.snapshot();
        let ef_actual = ef.unwrap_or(k.max(64));
        let query_arr: ArrayRef = query.value(0);
        let candidates: Vec<OrderedNode> = state.builder.search(query_arr, k, ef_actual, &view);

        let mut out: Vec<(f32, RowPosition)> = candidates
            .into_iter()
            .filter_map(|n| {
                let pos = view.row_pos(n.id);
                if pos <= max_row_position {
                    Some((n.dist.0, pos))
                } else {
                    None
                }
            })
            .collect();
        // search_inner may return at most `ef_actual` items; slice to k.
        if out.len() > k {
            out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(k);
        }
        Ok(out)
    }

    /// Snapshot the in-memory HNSW into the Lance on-disk representation:
    /// returns the graph + the FLAT vector storage record batch.
    ///
    /// Returns `Some((hnsw, storage_batch))` if there is at least one
    /// inserted vector; `None` otherwise. Caller must ensure no concurrent
    /// inserts while this runs.
    ///
    /// `total_rows`: when `Some(n)`, row positions in the storage batch are
    /// reversed (`n - pos - 1`); when `None`, they are written as-is.
    pub fn to_lance_hnsw(&self, total_rows: Option<u64>) -> Result<Option<(HNSW, RecordBatch)>> {
        let Some(state) = self.state.get() else {
            return Ok(None);
        };
        if state.builder.is_empty() {
            return Ok(None);
        }
        let storage_batch = match total_rows {
            Some(n) => state.storage.to_record_batch_reversed(n)?,
            None => state.storage.to_record_batch()?,
        };
        let hnsw = state.builder.to_hnsw();
        Ok(Some((hnsw, storage_batch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use lance_arrow::FixedSizeListArrayExt;

    fn make_batch(start_id: i32, n: usize, dim: usize) -> RecordBatch {
        let ids: Vec<i32> = (start_id..start_id + n as i32).collect();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for &id in &ids {
            for d in 0..dim {
                flat.push((id as f32 * 0.01) + (d as f32 * 0.001));
            }
        }
        let inner = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new_from_values(inner, dim as i32).unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids)), Arc::new(fsl)]).unwrap()
    }

    #[test]
    fn test_storage_append_and_read() {
        let storage = MemHnswStorage::with_capacity(8, 4, DistanceType::L2);
        let v0 = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![5.0, 6.0, 7.0, 8.0];
        let id0 = storage.append(&v0, 100).unwrap();
        let id1 = storage.append(&v1, 200).unwrap();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(storage.committed_len(), 2);
        assert_eq!(storage.row_position(0), 100);
        assert_eq!(storage.row_position(1), 200);
        assert_eq!(storage.vector_slice(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(storage.vector_slice(1), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_storage_capacity_exhausted() {
        let storage = MemHnswStorage::with_capacity(2, 2, DistanceType::L2);
        storage.append(&[1.0, 1.0], 0).unwrap();
        storage.append(&[2.0, 2.0], 1).unwrap();
        assert!(storage.append(&[3.0, 3.0], 2).is_err());
    }

    #[test]
    fn test_index_insert_and_search() {
        let dim = 8;
        let n = 200;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
            n,
        );

        let batch = make_batch(0, n, dim);
        index.insert(&batch, 0).unwrap();
        assert_eq!(index.len(), n);

        // Query with a vector identical to row 5: it should be the nearest result.
        let fsl = batch.column_by_name("vector").unwrap().as_fixed_size_list();
        let query_inner =
            Float32Array::from(fsl.value(5).as_primitive::<Float32Type>().values().to_vec());
        let query = FixedSizeListArray::try_new_from_values(query_inner, dim as i32).unwrap();

        let results = index.search(&query, 5, Some(32), u64::MAX).unwrap();
        assert!(!results.is_empty());
        // The closest result should have row position 5 and distance ~0.
        let (best_dist, best_pos) = results[0];
        assert!(
            best_dist < 1e-4,
            "expected near-zero distance, got {}",
            best_dist
        );
        assert_eq!(best_pos, 5);
    }

    #[test]
    fn test_index_visibility_filter() {
        let dim = 8;
        let n = 50;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
            n,
        );
        let batch = make_batch(0, n, dim);
        index.insert(&batch, 0).unwrap();

        let fsl = batch.column_by_name("vector").unwrap().as_fixed_size_list();
        let query_inner = Float32Array::from(
            fsl.value(40)
                .as_primitive::<Float32Type>()
                .values()
                .to_vec(),
        );
        let query = FixedSizeListArray::try_new_from_values(query_inner, dim as i32).unwrap();

        // Limit visibility to row 10.
        let results = index.search(&query, 5, Some(32), 10).unwrap();
        for (_, pos) in &results {
            assert!(*pos <= 10);
        }
    }

    #[test]
    fn test_index_empty_search() {
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default(),
            16,
        );
        // Build a query of dim 4 — but the index has no state yet. Should
        // return empty without panicking.
        let inner = Float32Array::from(vec![0.0; 4]);
        let query = FixedSizeListArray::try_new_from_values(inner, 4).unwrap();
        let results = index.search(&query, 5, None, u64::MAX).unwrap();
        assert!(results.is_empty());
    }
}
