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

/// Reference into the writer's BatchStore for a single appended batch.
///
/// `MemHnswStorage` keeps these instead of copying vectors. The Arc keeps
/// the underlying Arrow buffer alive for the storage's lifetime; `values_ptr`
/// is cached to avoid repeated downcast on the hot dist-calc path.
struct MemHnswBatch {
    /// Owned reference to the batch's vector column. Held to keep the
    /// underlying Arrow buffer alive for the lifetime of this storage entry;
    /// `values_ptr` borrows into this Arc.
    #[allow(dead_code)]
    arrow_array: Arc<FixedSizeListArray>,
    /// Cached pointer to the first f32 in the values buffer.
    values_ptr: *const f32,
}

/// Maps a vector id to its location in the per-batch storage. `id` is dense
/// `[0, committed_len)`; `(batch_idx, offset)` indexes into the `batches`
/// table.
#[derive(Copy, Clone)]
struct RowLookup {
    batch_idx: u32,
    offset: u32,
}

/// Lock-free flat-float vector storage for in-memory HNSW.
///
/// # Storage layout
///
/// Vectors are *not copied*. The storage holds `Arc<FixedSizeListArray>`
/// references into the writer's BatchStore — the same allocations the
/// writer's query path materializes from. A small `row_to_batch` table maps
/// each id to `(batch_idx, offset_within_batch)` so distance calc is O(1).
///
/// Memory cost relative to the previous owns-its-own-buffer design:
/// - Saved: `capacity * dim * 4` bytes (the duplicate Float32 buffer).
/// - Added: `capacity * 8` bytes (`row_to_batch`) + `max_batches *
///   sizeof(MemHnswBatch)` for the per-batch metadata.
///
/// At dim=1024 / 1M rows this is ~4 GB saved vs ~8 MB added.
///
/// # Concurrency
///
/// - Single writer appends batches via `append_batch`.
/// - Multiple concurrent readers call `vector_slice` / `row_position`.
///
/// Publication: writer initializes all metadata for a batch (slot in
/// `batches`, entries in `row_to_batch`, entries in `row_positions`)
/// *before* incrementing `committed_len` with `Release`. Readers acquire-load
/// `committed_len` and only access indices `< committed_len`, so all writes
/// above are visible. `committed_batches` follows the same release/acquire
/// pattern.
pub struct MemHnswStorage {
    /// Per-batch metadata. Slot `i` is initialized iff `i < committed_batches`.
    batches: *mut MaybeUninit<MemHnswBatch>,
    /// Number of committed batches. Reads use Acquire; writes use Release.
    committed_batches: AtomicUsize,
    /// id → (batch_idx, offset_in_batch). Slot at index `id` is initialized
    /// iff `id < committed_len`. 8 bytes per row.
    row_to_batch: *mut MaybeUninit<RowLookup>,
    /// Pointer to the start of the row-positions buffer. Slot at index `id`
    /// is initialized iff `id < committed_len`. 8 bytes per row.
    row_positions_ptr: *mut MaybeUninit<u64>,
    capacity: usize,
    max_batches: usize,
    dim: usize,
    distance_type: DistanceType,
    /// Number of committed vectors. Reads must use Acquire; writes use Release.
    committed_len: Arc<AtomicUsize>,
    /// Schema cached for the `VectorStore` impl.
    schema: SchemaRef,
}

// SAFETY: `MemHnswStorage` follows a single-writer multi-reader model. The
// writer is the only mutator of the underlying buffers; readers only access
// indices `< committed_len` (or `< committed_batches` for the batches table),
// and those counters are published with `Release` ordering so readers see
// initialized data. All buffer access uses raw pointers to avoid Rust-level
// aliasing of references.
unsafe impl Sync for MemHnswStorage {}
unsafe impl Send for MemHnswStorage {}

impl Drop for MemHnswStorage {
    fn drop(&mut self) {
        // Drop committed batch entries to release their Arcs, then free the
        // backing allocations. Only the writer-owned writes matter here:
        // we have `&mut self` so there are no concurrent readers.
        unsafe {
            let committed_batches = self.committed_batches.load(Ordering::Acquire);
            for i in 0..committed_batches {
                let slot = self.batches.add(i);
                std::ptr::drop_in_place(slot.cast::<MemHnswBatch>());
            }
            let _: Box<[MaybeUninit<MemHnswBatch>]> = Box::from_raw(
                std::ptr::slice_from_raw_parts_mut(self.batches, self.max_batches),
            );
            let _: Box<[MaybeUninit<RowLookup>]> = Box::from_raw(
                std::ptr::slice_from_raw_parts_mut(self.row_to_batch, self.capacity),
            );
            let _: Box<[MaybeUninit<u64>]> = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.row_positions_ptr,
                self.capacity,
            ));
        }
    }
}

impl MemHnswStorage {
    /// Create a storage pre-allocated for `capacity` vectors of `dim` floats
    /// across at most `max_batches` batches. Vectors are stored by reference
    /// to the appended Arrow `FixedSizeListArray`s, not copied.
    pub fn with_capacity(
        capacity: usize,
        max_batches: usize,
        dim: usize,
        distance_type: DistanceType,
    ) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(dim > 0, "dim must be > 0");
        assert!(max_batches > 0, "max_batches must be > 0");
        assert!(
            capacity <= u32::MAX as usize,
            "MemHnswStorage capacity must fit in u32"
        );
        assert!(
            max_batches <= u32::MAX as usize,
            "MemHnswStorage max_batches must fit in u32"
        );

        let batches: Box<[MaybeUninit<MemHnswBatch>]> = (0..max_batches)
            .map(|_| MaybeUninit::uninit())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let row_to_batch: Box<[MaybeUninit<RowLookup>]> = (0..capacity)
            .map(|_| MaybeUninit::uninit())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let row_positions: Box<[MaybeUninit<u64>]> = (0..capacity)
            .map(|_| MaybeUninit::uninit())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let batches_ptr = Box::into_raw(batches) as *mut MaybeUninit<MemHnswBatch>;
        let row_to_batch_ptr = Box::into_raw(row_to_batch) as *mut MaybeUninit<RowLookup>;
        let row_positions_ptr = Box::into_raw(row_positions) as *mut MaybeUninit<u64>;

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
            batches: batches_ptr,
            committed_batches: AtomicUsize::new(0),
            row_to_batch: row_to_batch_ptr,
            row_positions_ptr,
            capacity,
            max_batches,
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

    /// Append a whole batch by reference. Single-writer only.
    ///
    /// `vectors` is borrowed (Arc-cloned internally); the underlying Arrow
    /// buffer is kept alive via the stored Arc. `row_positions` are computed
    /// from `row_offset` (the row position assigned to row 0 of this batch).
    pub fn append_batch(
        &self,
        vectors: Arc<FixedSizeListArray>,
        row_offset: u64,
    ) -> Result<std::ops::Range<u32>> {
        let n = vectors.len();
        if n == 0 {
            let id = self.committed_len.load(Ordering::Relaxed) as u32;
            return Ok(id..id);
        }
        if vectors.value_length() as usize != self.dim {
            return Err(Error::invalid_input(format!(
                "batch vector dim mismatch: expected {}, got {}",
                self.dim,
                vectors.value_length()
            )));
        }

        let id_start = self.committed_len.load(Ordering::Relaxed);
        if id_start + n > self.capacity {
            return Err(Error::invalid_input(format!(
                "MemHnswStorage capacity {} exhausted (have {}, need {})",
                self.capacity,
                id_start,
                id_start + n
            )));
        }
        let batch_idx = self.committed_batches.load(Ordering::Relaxed);
        if batch_idx >= self.max_batches {
            return Err(Error::invalid_input(format!(
                "MemHnswStorage max_batches {} exhausted",
                self.max_batches
            )));
        }

        // Cache the values buffer pointer for fast access. The Arc keeps
        // this buffer alive for the life of the storage entry.
        let values_arr = vectors.values();
        let values_f32 = values_arr.as_primitive::<Float32Type>();
        let values_ptr = values_f32.values().as_ptr();

        // SAFETY: single writer, slots at id_start..id_start+n and
        // batch_idx are reserved and not yet visible to readers (committed_*
        // counters haven't been incremented).
        unsafe {
            let batch_slot = self.batches.add(batch_idx);
            batch_slot.write(MaybeUninit::new(MemHnswBatch {
                arrow_array: vectors,
                values_ptr,
            }));

            for i in 0..n {
                self.row_to_batch
                    .add(id_start + i)
                    .write(MaybeUninit::new(RowLookup {
                        batch_idx: batch_idx as u32,
                        offset: i as u32,
                    }));
                self.row_positions_ptr
                    .add(id_start + i)
                    .write(MaybeUninit::new(row_offset + i as u64));
            }
        }

        // Publish committed_batches first so readers that observe a future
        // committed_len's increased value will also see the new batch slot.
        self.committed_batches
            .store(batch_idx + 1, Ordering::Release);
        self.committed_len.store(id_start + n, Ordering::Release);

        Ok(id_start as u32..(id_start + n) as u32)
    }

    /// Convenience for tests: append a single Float32 vector by value as a
    /// freshly-allocated 1-row Arrow batch. Production code goes through
    /// `append_batch` which is zero-copy.
    #[cfg(test)]
    pub fn append(&self, vector: &[f32], row_position: u64) -> Result<u32> {
        let inner = Arc::new(Float32Array::from(vector.to_vec())) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let fsl = Arc::new(FixedSizeListArray::try_new(
            field,
            vector.len() as i32,
            inner,
            None,
        )?);
        let range = self.append_batch(fsl, row_position)?;
        Ok(range.start)
    }

    /// Get the row position for a committed id.
    pub fn row_position(&self, id: u32) -> u64 {
        debug_assert!((id as usize) < self.committed_len.load(Ordering::Acquire));
        // SAFETY: id < committed_len => initialized; the slot is never
        // overwritten after publication.
        unsafe { self.row_positions_ptr.add(id as usize).read().assume_init() }
    }

    /// Get a slice view of the vector at id `id`. The slice borrows the
    /// underlying Arrow buffer; lifetime is tied to the storage (which holds
    /// an Arc to the buffer).
    pub fn vector_slice(&self, id: u32) -> &[f32] {
        debug_assert!((id as usize) < self.committed_len.load(Ordering::Acquire));
        // SAFETY: id < committed_len => row_to_batch[id] initialized AND
        // the referenced batches[batch_idx] is initialized (writer publishes
        // committed_batches before committed_len). The Arrow buffer behind
        // values_ptr stays valid because storage holds an Arc to it.
        unsafe {
            let lookup = self.row_to_batch.add(id as usize).read().assume_init();
            let batch_slot = self.batches.add(lookup.batch_idx as usize);
            let batch = batch_slot.cast::<MemHnswBatch>().as_ref().unwrap();
            let base = batch.values_ptr.add((lookup.offset as usize) * self.dim);
            std::slice::from_raw_parts(base, self.dim)
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
        // the storage (single writer never overwrites committed entries). We
        // build the slice from the raw pointer without materializing any
        // reference to the surrounding buffer.
        let slice: &[u64] = unsafe {
            std::slice::from_raw_parts(
                self.storage.row_positions_ptr as *const u64,
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
///
/// Two construction modes:
/// - From an external query (`new_for_query`): the query is owned because the
///   caller's `ArrayRef` lifetime isn't tied to the view.
/// - From an existing id (`new_for_id`): the query borrows the view's own
///   buffer to avoid a `dim*4`-byte copy on the hot insert path. This matters
///   because `select_neighbors_heuristic` builds an `ef_construction`-sized
///   pool of these, so the per-call copy was order of `ef * dim * 4` bytes
///   per insert.
pub struct MemHnswDistCalc<'a> {
    view: &'a MemHnswStorageView,
    /// `None` is a tombstone used when `new_for_id` is called with an id past
    /// the view's snapshot — `distance()` returns +inf so the search path
    /// drops that candidate.
    query: Option<std::borrow::Cow<'a, [f32]>>,
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
            query: Some(std::borrow::Cow::Owned(query_vec)),
        }
    }

    fn new_for_id(view: &'a MemHnswStorageView, id: u32) -> Self {
        // The graph's `level_neighbors` lists are published via `ArcSwap`
        // and may include ids beyond this view's snapshot — the writer could
        // have appended a new vector and published reverse edges referencing
        // it after this view was constructed. If the seed candidate is out
        // of snapshot, we tombstone the calculator.
        let query = if (id as usize) < view.visible_len {
            // Borrow the view's own buffer instead of copying — the buffer
            // outlives `'a` because the view holds an Arc to the storage.
            Some(std::borrow::Cow::Borrowed(view.vector_slice(id)))
        } else {
            None
        };
        Self { view, query }
    }
}

impl DistCalculator for MemHnswDistCalc<'_> {
    fn distance(&self, id: u32) -> f32 {
        let Some(query) = self.query.as_deref() else {
            return f32::INFINITY;
        };
        if (id as usize) >= self.view.visible_len {
            return f32::INFINITY;
        }
        let v = self.view.vector_slice(id);
        compute_distance(query, v, self.view.storage.distance_type)
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        let Some(query) = self.query.as_deref() else {
            return vec![f32::INFINITY; self.view.visible_len];
        };
        let mut out = Vec::with_capacity(self.view.visible_len);
        for id in 0..self.view.visible_len as u32 {
            let v = self.view.vector_slice(id);
            out.push(compute_distance(query, v, self.view.storage.distance_type));
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
        DistanceType::Dot => f32::dot(query, vector),
        // Cosine and other variants go through the dispatched fn.
        _ => distance_type.func()(query, vector),
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
    /// Maximum number of batches the storage can hold by reference.
    max_batches: usize,
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
        max_batches: usize,
    ) -> Self {
        Self {
            field_id,
            column,
            distance_type,
            dim: AtomicUsize::new(MEM_HNSW_DIM_PLACEHOLDER),
            capacity,
            max_batches,
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
                self.max_batches,
                dim,
                self.distance_type,
            ));
            let builder =
                OnlineHnswBuilder::with_capacity(self.capacity, self.build_params.clone());
            HnswState { storage, builder }
        })
    }

    /// Insert vectors from a single batch.
    ///
    /// The batch's vector column is appended **by reference** — the Arrow
    /// buffer is not copied; the storage holds an Arc to the
    /// `FixedSizeListArray` for the lifetime of the MemTable.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        // Bail loudly if the column is missing rather than silently skipping —
        // a partial-schema batch reaching this code path means the index will
        // silently desync from the data.
        let (col_idx, _) = batch
            .schema()
            .column_with_name(&self.column)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "HNSW index column '{}' is not in the inserted batch schema",
                    self.column
                ))
            })?;
        let column = batch.column(col_idx);
        let fsl_ref = column.as_fixed_size_list_opt().ok_or_else(|| {
            Error::invalid_input(format!(
                "Column '{}' is not a FixedSizeList, got {:?}",
                self.column,
                column.data_type()
            ))
        })?;
        if fsl_ref.is_empty() {
            return Ok(());
        }
        // Validate Float32 element type — the fast path assumes raw f32.
        if fsl_ref.values().as_primitive_opt::<Float32Type>().is_none() {
            return Err(Error::invalid_input(format!(
                "Column '{}' must be FixedSizeList<Float32>, got values type {:?}",
                self.column,
                fsl_ref.values().data_type()
            )));
        }
        // Vector indexes don't have a sensible "null vector" semantics —
        // distances on uninitialized buffer bytes are garbage. Reject FSLs
        // that contain null rows up front.
        if fsl_ref.null_count() > 0 {
            return Err(Error::invalid_input(format!(
                "HNSW index column '{}' has {} null row(s); null vectors are not supported",
                self.column,
                fsl_ref.null_count()
            )));
        }
        let fsl: Arc<FixedSizeListArray> = Arc::new(fsl_ref.clone());
        let dim = fsl.value_length() as usize;
        let state = self.ensure_state(dim);

        // Append the batch by reference (zero-copy) and get the id range
        // assigned to its rows.
        let id_range = state.storage.append_batch(fsl, row_offset)?;
        // Per-row HNSW insert; each insert sees a fresh snapshot so it
        // observes its own predecessor in the same batch.
        for id in id_range {
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

        // Drop any candidate id past the snapshot — see the long comment on
        // `MemHnswDistCalc::new_for_id` for why ArcSwap-published neighbors
        // can reference ids the view doesn't yet cover. Also drop +inf
        // distances that the dist-calc tombstone produced for those ids.
        let mut out: Vec<(f32, RowPosition)> = candidates
            .into_iter()
            .filter_map(|n| {
                if (n.id as usize) >= view.visible_len || !n.dist.0.is_finite() {
                    return None;
                }
                let pos = view.row_pos(n.id);
                if pos <= max_row_position {
                    Some((n.dist.0, pos))
                } else {
                    None
                }
            })
            .collect();
        // `OnlineHnswBuilder::search` returns up to `ef` candidates ordered by
        // distance. After visibility filtering some may be dropped, so re-sort
        // and truncate to `k` here (the post-filter ordering may still hold but
        // the sort is cheap and keeps the contract robust).
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
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
        let storage = MemHnswStorage::with_capacity(8, 4, 4, DistanceType::L2);
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
        let storage = MemHnswStorage::with_capacity(2, 4, 2, DistanceType::L2);
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
            64,
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
            64,
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
            16,
        );
        // Build a query of dim 4 — but the index has no state yet. Should
        // return empty without panicking.
        let inner = Float32Array::from(vec![0.0; 4]);
        let query = FixedSizeListArray::try_new_from_values(inner, 4).unwrap();
        let results = index.search(&query, 5, None, u64::MAX).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_concurrent_insert_and_search() {
        // One writer thread inserts vectors, multiple reader threads
        // hammer `search` against the index. Verifies that a search
        // executing while inserts are in flight doesn't read uninitialized
        // memory or panic — the dist-calc bounds check should drop any
        // neighbor id past the snapshot's visible range.
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering as StdOrdering};
        use std::thread;

        let dim = 16;
        let n = 500;
        let index = StdArc::new(HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(8).ef_construction(32),
            n,
            256,
        ));

        // Pre-insert one vector so the index has dim and an entry point.
        let initial = make_batch(-1, 1, dim);
        index.insert(&initial, 0).unwrap();

        let stop = StdArc::new(AtomicBool::new(false));

        // Reader threads
        let mut reader_handles = Vec::new();
        for _ in 0..4 {
            let index = index.clone();
            let stop = stop.clone();
            reader_handles.push(thread::spawn(move || {
                let inner = Float32Array::from(vec![0.5_f32; dim]);
                let query = FixedSizeListArray::try_new_from_values(inner, dim as i32).unwrap();
                let mut iters = 0u64;
                while !stop.load(StdOrdering::Relaxed) {
                    let _ = index.search(&query, 5, Some(32), u64::MAX).unwrap();
                    iters += 1;
                }
                iters
            }));
        }

        // Writer thread inserts the rest of the vectors one batch at a time.
        let writer_index = index.clone();
        let writer_handle = thread::spawn(move || {
            for i in 1..(n / 5) {
                let batch = make_batch(i as i32 * 5, 5, dim);
                let row_offset = (i as u64) * 5 + 1; // offset by the initial 1-vector batch
                writer_index.insert(&batch, row_offset).unwrap();
            }
        });

        writer_handle.join().unwrap();
        stop.store(true, StdOrdering::Release);
        let mut total_reader_iters = 0u64;
        for h in reader_handles {
            total_reader_iters += h.join().unwrap();
        }

        // Sanity: at least the writer made progress and readers ran.
        assert!(index.len() > 1);
        assert!(total_reader_iters > 0);
    }
}
