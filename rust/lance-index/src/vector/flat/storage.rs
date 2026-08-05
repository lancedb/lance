// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{borrow::Cow, sync::Arc};

use super::index::FlatMetadata;
use crate::frag_reuse::FragReuseIndex;
use crate::vector::quantizer::QuantizerStorage;
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::utils::do_prefetch;
use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow::datatypes::{Float16Type, Float64Type, UInt8Type};
use arrow_array::ArrowPrimitiveType;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array,
    types::{Float32Type, UInt64Type},
};
use arrow_schema::{DataType, SchemaRef};
use half::f16;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ID, Result};
use lance_file::versions::v1::reader::FileReader as V1FileReader;
use lance_linalg::distance::dot_f16::{amx_fp16_available, dot_f16_batch_16};
use lance_linalg::distance::hamming::hamming;
use lance_linalg::distance::{Cosine, DistanceType, Dot, L2};

pub const FLAT_COLUMN: &str = "flat";

/// All data are stored in memory
#[derive(Debug, Clone)]
pub struct FlatFloatStorage {
    metadata: FlatMetadata,
    batch: RecordBatch,
    distance_type: DistanceType,

    // helper fields
    pub(super) row_ids: Arc<UInt64Array>,
    vectors: Arc<FixedSizeListArray>,
}

impl DeepSizeOf for FlatFloatStorage {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        self.batch.get_array_memory_size()
    }
}

#[async_trait::async_trait]
impl QuantizerStorage for FlatFloatStorage {
    type Metadata = FlatMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let batch = if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            frag_reuse_index_ref.remap_row_ids_record_batch(batch, 0)?
        } else {
            batch
        };

        let row_ids = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        let vectors = Arc::new(
            batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            vectors,
        })
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        _: &V1FileReader,
        _: std::ops::Range<usize>,
        _: DistanceType,
        _: &Self::Metadata,
        _: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        unimplemented!("Flat will be used in new index builder which doesn't require this")
    }
}

impl FlatFloatStorage {
    // used for only testing
    pub fn new(vectors: FixedSizeListArray, distance_type: DistanceType) -> Self {
        let row_ids = Arc::new(UInt64Array::from_iter_values(0..vectors.len() as u64));
        let vectors = Arc::new(vectors);

        let batch = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids.clone() as ArrayRef, true),
            (FLAT_COLUMN, vectors.clone() as ArrayRef, true),
        ])
        .unwrap();

        Self {
            metadata: FlatMetadata {
                dim: vectors.value_length() as usize,
            },
            batch,
            distance_type,
            row_ids,
            vectors,
        }
    }

    pub fn vector(&self, id: u32) -> ArrayRef {
        self.vectors.value(id as usize)
    }
}

impl VectorStore for FlatFloatStorage {
    type DistanceCalculator<'a> = FlatFloatDistanceCalc<'a>;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        Ok([self.batch.clone()].into_iter())
    }

    fn append_batch(&self, batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        // TODO: use chunked storage
        let new_batch = concat_batches(&batch.schema(), vec![&self.batch, &batch])?;
        let mut storage = self.clone();
        storage.row_ids = Arc::new(
            new_batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        storage.vectors = Arc::new(
            new_batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        storage.batch = new_batch;
        Ok(storage)
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.values()[id as usize]
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.values().iter()
    }

    fn dist_calculator(&self, query: ArrayRef, _dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new(self.vectors.as_ref(), query, self.distance_type)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new_from_id(self.vectors.as_ref(), id, self.distance_type)
    }
}

/// All data are stored in memory
#[derive(Debug, Clone)]
pub struct FlatBinStorage {
    metadata: FlatMetadata,
    batch: RecordBatch,
    distance_type: DistanceType,

    // helper fields
    pub(super) row_ids: Arc<UInt64Array>,
    vectors: Arc<FixedSizeListArray>,
}

impl DeepSizeOf for FlatBinStorage {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        self.batch.get_array_memory_size()
    }
}

#[async_trait::async_trait]
impl QuantizerStorage for FlatBinStorage {
    type Metadata = FlatMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let batch = if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            frag_reuse_index_ref.remap_row_ids_record_batch(batch, 0)?
        } else {
            batch
        };

        let row_ids = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        let vectors = Arc::new(
            batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            vectors,
        })
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        _: &V1FileReader,
        _: std::ops::Range<usize>,
        _: DistanceType,
        _: &Self::Metadata,
        _: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        unimplemented!("Flat will be used in new index builder which doesn't require this")
    }
}

impl FlatBinStorage {
    // used for only testing
    pub fn new(vectors: FixedSizeListArray, distance_type: DistanceType) -> Self {
        let row_ids = Arc::new(UInt64Array::from_iter_values(0..vectors.len() as u64));
        let vectors = Arc::new(vectors);

        let batch = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids.clone() as ArrayRef, true),
            (FLAT_COLUMN, vectors.clone() as ArrayRef, true),
        ])
        .unwrap();

        Self {
            metadata: FlatMetadata {
                dim: vectors.value_length() as usize,
            },
            batch,
            distance_type,
            row_ids,
            vectors,
        }
    }

    pub fn vector(&self, id: u32) -> ArrayRef {
        self.vectors.value(id as usize)
    }
}

impl VectorStore for FlatBinStorage {
    type DistanceCalculator<'a> = FlatDistanceCal<'a, UInt8Type>;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        Ok([self.batch.clone()].into_iter())
    }

    fn append_batch(&self, batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        // TODO: use chunked storage
        let new_batch = concat_batches(&batch.schema(), vec![&self.batch, &batch])?;
        let mut storage = self.clone();
        storage.row_ids = Arc::new(
            new_batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        storage.vectors = Arc::new(
            new_batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        storage.batch = new_batch;
        Ok(storage)
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.values()[id as usize]
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.values().iter()
    }

    fn dist_calculator(&self, query: ArrayRef, _dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new_binary(self.vectors.as_ref(), query, self.distance_type)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new_binary_from_id(self.vectors.as_ref(), id, self.distance_type)
    }
}

pub struct FlatDistanceCal<'a, T: ArrowPrimitiveType> {
    vectors: &'a [T::Native],
    query: Cow<'a, [T::Native]>,
    dimension: usize,
    /// Metric this calculator computes. Retained (beyond `distance_fn`) so the
    /// fp16 `distance_batch_16` override can recognize the `Dot` case that maps
    /// onto the batched AMX-FP16 tile kernel.
    distance_type: DistanceType,
    #[allow(clippy::type_complexity)]
    distance_fn: fn(&[T::Native], &[T::Native]) -> f32,
}

impl<'a, T> FlatDistanceCal<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: L2 + Cosine + Dot,
{
    fn new(vectors: &'a FixedSizeListArray, query: ArrayRef, distance_type: DistanceType) -> Self {
        // Gained significant performance improvement by using strong typed primitive slice.
        let flat_array = vectors.values().as_primitive::<T>();
        let dimension = vectors.value_length() as usize;
        Self {
            vectors: flat_array.values(),
            query: Cow::Owned(query.as_primitive::<T>().values().to_vec()),
            dimension,
            distance_type,
            distance_fn: distance_type.func(),
        }
    }

    fn new_from_id(vectors: &'a FixedSizeListArray, id: u32, distance_type: DistanceType) -> Self {
        let flat_array = vectors.values().as_primitive::<T>();
        let dimension = vectors.value_length() as usize;
        let vectors = flat_array.values();
        let id = id as usize;
        Self {
            vectors,
            query: Cow::Borrowed(&vectors[dimension * id..dimension * (id + 1)]),
            dimension,
            distance_type,
            distance_fn: distance_type.func(),
        }
    }
}

impl<'a> FlatDistanceCal<'a, UInt8Type> {
    fn new_binary(
        vectors: &'a FixedSizeListArray,
        query: ArrayRef,
        _distance_type: DistanceType,
    ) -> Self {
        // Gained significant performance improvement by using strong typed primitive slice.
        // TODO: to support other data types other than `f32`, make FlatDistanceCal a generic struct.
        let flat_array = vectors.values().as_primitive::<UInt8Type>();
        let dimension = vectors.value_length() as usize;
        Self {
            vectors: flat_array.values(),
            query: Cow::Owned(query.as_primitive::<UInt8Type>().values().to_vec()),
            dimension,
            distance_type: _distance_type,
            distance_fn: hamming,
        }
    }

    fn new_binary_from_id(
        vectors: &'a FixedSizeListArray,
        id: u32,
        _distance_type: DistanceType,
    ) -> Self {
        let flat_array = vectors.values().as_primitive::<UInt8Type>();
        let dimension = vectors.value_length() as usize;
        let vectors = flat_array.values();
        let id = id as usize;
        Self {
            vectors,
            query: Cow::Borrowed(&vectors[dimension * id..dimension * (id + 1)]),
            dimension,
            distance_type: _distance_type,
            distance_fn: hamming,
        }
    }
}

impl<T: ArrowPrimitiveType> FlatDistanceCal<'_, T> {
    #[inline]
    fn get_vector(&self, id: u32) -> &[T::Native] {
        &self.vectors[self.dimension * id as usize..self.dimension * (id + 1) as usize]
    }
}

impl FlatDistanceCal<'_, Float16Type> {
    /// Whether [`Self::dot_distance_batch_16_amx`] would actually reach the
    /// AMX-FP16 tile kernel: the metric must be `Dot`, the dimension must cover
    /// at least one full 32-wide tile pass (below that `dot_f16_batch_16` falls
    /// back regardless), and AMX-FP16 must be usable in this process.
    ///
    /// Single source of truth shared by the `distance_batch_16` override and
    /// `has_batch_16_kernel`, so the two cannot drift apart and leave HNSW
    /// buffering neighbors for a fused kernel it would never reach.
    #[inline]
    fn has_amx_dot_kernel(&self) -> bool {
        self.distance_type == DistanceType::Dot && self.dimension >= 32 && amx_fp16_available()
    }

    /// The `Dot` distances of the query against the first `len` of `ids`, fused
    /// into a single AMX-FP16 tile pass when the hardware/kernel is available
    /// (otherwise `len` scalar `f16::dot`s — bit-identical to the per-id
    /// `distance()` path).
    ///
    /// [`dot_f16_batch_16`] returns raw dot products `Σ q·c`; `Dot` distance is
    /// `1.0 - Σ q·c` (see `lance_linalg`'s `dot_distance`), applied here so the
    /// result matches [`DistCalculator::distance`] to within fp16 precision.
    /// Only valid for `DistanceType::Dot`; the enum caller guarantees that.
    ///
    /// All 16 candidate slices are still resolved, since `ids` past `len` are
    /// the caller's padding and therefore valid ids; the kernel reads only the
    /// first `len` of them.
    #[inline]
    fn dot_distance_batch_16_amx(&self, ids: &[u32; 16], dists: &mut [f32; 16], len: usize) {
        let query: &[f16] = self.query.as_ref();
        let candidates: [&[f16]; 16] = std::array::from_fn(|i| self.get_vector(ids[i]));
        let dots = dot_f16_batch_16(query, &candidates, len);
        for (dist, dot) in dists.iter_mut().zip(dots.iter()).take(len) {
            *dist = 1.0 - *dot;
        }
    }
}

impl<T: ArrowPrimitiveType> DistCalculator for FlatDistanceCal<'_, T> {
    #[inline]
    fn distance(&self, id: u32) -> f32 {
        let query = self.query.as_ref();
        let vector = self.get_vector(id);
        (self.distance_fn)(query, vector)
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        let query = self.query.as_ref();
        self.vectors
            .chunks_exact(self.dimension)
            .map(|vector| (self.distance_fn)(query, vector))
            .collect()
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        let vector = self.get_vector(id);
        do_prefetch(vector.as_ptr_range())
    }
}

pub enum FlatFloatDistanceCalc<'a> {
    Float16(FlatDistanceCal<'a, Float16Type>),
    Float32(FlatDistanceCal<'a, Float32Type>),
    Float64(FlatDistanceCal<'a, Float64Type>),
}

impl<'a> FlatFloatDistanceCalc<'a> {
    fn new(vectors: &'a FixedSizeListArray, query: ArrayRef, distance_type: DistanceType) -> Self {
        match vectors.value_type() {
            DataType::Float16 => Self::Float16(FlatDistanceCal::<Float16Type>::new(
                vectors,
                query,
                distance_type,
            )),
            DataType::Float32 => Self::Float32(FlatDistanceCal::<Float32Type>::new(
                vectors,
                query,
                distance_type,
            )),
            DataType::Float64 => Self::Float64(FlatDistanceCal::<Float64Type>::new(
                vectors,
                query,
                distance_type,
            )),
            dt => panic!("flat float storage does not support data type {dt}"),
        }
    }

    fn new_from_id(vectors: &'a FixedSizeListArray, id: u32, distance_type: DistanceType) -> Self {
        match vectors.value_type() {
            DataType::Float16 => Self::Float16(FlatDistanceCal::<Float16Type>::new_from_id(
                vectors,
                id,
                distance_type,
            )),
            DataType::Float32 => Self::Float32(FlatDistanceCal::<Float32Type>::new_from_id(
                vectors,
                id,
                distance_type,
            )),
            DataType::Float64 => Self::Float64(FlatDistanceCal::<Float64Type>::new_from_id(
                vectors,
                id,
                distance_type,
            )),
            dt => panic!("flat float storage does not support data type {dt}"),
        }
    }
}

impl DistCalculator for FlatFloatDistanceCalc<'_> {
    fn distance(&self, id: u32) -> f32 {
        match self {
            Self::Float16(calc) => calc.distance(id),
            Self::Float32(calc) => calc.distance(id),
            Self::Float64(calc) => calc.distance(id),
        }
    }

    fn distance_all(&self, k_hint: usize) -> Vec<f32> {
        match self {
            Self::Float16(calc) => calc.distance_all(k_hint),
            Self::Float32(calc) => calc.distance_all(k_hint),
            Self::Float64(calc) => calc.distance_all(k_hint),
        }
    }

    /// fp16 `Dot` with AMX-FP16 available is the only configuration here backed
    /// by a fused kernel, so it is the only one that reports `true` — and hence
    /// the only one for which HNSW buffers neighbors 16 at a time at all.
    fn has_batch_16_kernel(&self) -> bool {
        matches!(self, Self::Float16(calc) if calc.has_amx_dot_kernel())
    }

    /// Fuse the `len` neighbor distances (issued in batches of up to 16 by
    /// HNSW's `beam_search_loop!`) into one AMX-FP16 tile pass for the fp16
    /// `Dot` case — the only path with a batched kernel. Every other case (fp16
    /// L2/Cosine, f32, f64) keeps the trait default of `len` individual
    /// `distance()` calls, so results there are byte-identical to before. The
    /// guard matches `has_batch_16_kernel` exactly; it stays here because this
    /// is a public trait method that external callers may invoke without
    /// consulting it.
    fn distance_batch_16(&self, ids: &[u32; 16], dists: &mut [f32; 16], len: usize) {
        debug_assert!((1..=16).contains(&len), "len ({len}) must be in 1..=16");
        if let Self::Float16(calc) = self
            && calc.has_amx_dot_kernel()
        {
            calc.dot_distance_batch_16_amx(ids, dists, len);
            return;
        }
        for (dist, &id) in dists.iter_mut().zip(ids.iter()).take(len) {
            *dist = self.distance(id);
        }
    }

    fn prefetch(&self, id: u32) {
        match self {
            Self::Float16(calc) => calc.prefetch(id),
            Self::Float32(calc) => calc.prefetch(id),
            Self::Float64(calc) => calc.prefetch(id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float16Array, Float64Array};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;

    fn make_f16_storage() -> FlatFloatStorage {
        let values = Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
        ]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        FlatFloatStorage::new(vectors, DistanceType::L2)
    }

    fn make_f64_storage() -> FlatFloatStorage {
        let values = Float64Array::from(vec![1.0, 2.0, 4.0, 6.0]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        FlatFloatStorage::new(vectors, DistanceType::L2)
    }

    #[test]
    fn test_flat_float_storage_distance_f16() {
        let storage = make_f16_storage();
        let query: ArrayRef = Arc::new(Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
        ]));

        let calc = storage.dist_calculator(query, 0.0);
        let distances = calc.distance_all(2);

        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 25.0).abs() < 1e-4);
    }

    #[test]
    fn test_flat_float_storage_distance_f64() {
        let storage = make_f64_storage();
        let query: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));

        let calc = storage.dist_calculator(query, 0.0);
        let distances = calc.distance_all(2);

        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 25.0).abs() < 1e-6);
    }

    // --- fp16 Dot batched (AMX-FP16) distance_batch_16 override ---

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    /// Random `n x dim` fp16 vectors plus a random fp16 query. Values in [-1, 1).
    fn make_random_f16_vectors(n: usize, dim: usize, seed: u64) -> (FixedSizeListArray, ArrayRef) {
        let mut rng = StdRng::seed_from_u64(seed);
        let values = Float16Array::from_iter_values(
            (0..n * dim).map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0))),
        );
        let vectors = FixedSizeListArray::try_new_from_values(values, dim as i32).unwrap();
        let query: ArrayRef = Arc::new(Float16Array::from_iter_values(
            (0..dim).map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0))),
        ));
        (vectors, query)
    }

    /// Random `n x dim` fp16 storage with the given metric, plus a random fp16
    /// query array. Values in [-1, 1).
    fn make_random_f16(
        n: usize,
        dim: usize,
        dt: DistanceType,
        seed: u64,
    ) -> (FlatFloatStorage, ArrayRef) {
        let (vectors, query) = make_random_f16_vectors(n, dim, seed);
        (FlatFloatStorage::new(vectors, dt), query)
    }

    /// A value no metric here can produce, so a stray write into `dists[len..]`
    /// shows up as an obviously wrong number rather than a plausible distance.
    const BATCH_SENTINEL: f32 = -12345.0;

    /// Assert `distance_batch_16(ids, .., len)` computes exactly lanes
    /// `[0, len)` and leaves the rest as the caller left them.
    fn check_partial_batch_16<C: DistCalculator>(calc: &C, ids: &[u32; 16], len: usize, ctx: &str) {
        let mut dists = [BATCH_SENTINEL; 16];
        calc.distance_batch_16(ids, &mut dists, len);
        for (i, &id) in ids.iter().enumerate().take(len) {
            let single = calc.distance(id);
            let rel = (dists[i] - single).abs() / (single.abs() + 1e-6);
            assert!(
                rel <= 5e-3 || (dists[i] - single).abs() <= 1e-3,
                "{ctx} len={len} lane {i}: batch {} vs single {single}",
                dists[i]
            );
        }
        for (i, &d) in dists.iter().enumerate().skip(len) {
            assert_eq!(
                d.to_bits(),
                BATCH_SENTINEL.to_bits(),
                "{ctx} len={len}: lane {i} must be untouched, got {d}"
            );
        }
    }

    /// A partial batch must score exactly its live lanes and write nothing else.
    ///
    /// Every other `distance_batch_16` test here passes 16, which cannot tell
    /// `len` being honored from `len` being ignored. A regression that dropped
    /// the `.take(len)` and clobbered the caller's `dists[len..]` is what the
    /// sentinel catches; one where `len` was ignored outright is what comparing
    /// the live lanes against per-id `distance()` catches.
    ///
    /// Covers both arms of the fp16 override — `Dot` reaches the batched
    /// AMX-FP16 kernel on a host that has it, `L2` falls through to per-id calls
    /// — and `FlatDistanceCal`, which does not override the method at all and so
    /// exercises the `DistCalculator::distance_batch_16` default.
    #[rstest]
    #[case::dot(DistanceType::Dot)]
    #[case::l2(DistanceType::L2)]
    fn test_distance_batch_16_partial_len(#[case] dt: DistanceType) {
        let dim = 128;
        let (vectors, query) = make_random_f16_vectors(64, dim, 0xBA7C);
        let storage = FlatFloatStorage::new(vectors.clone(), dt);
        let enum_calc = storage.dist_calculator(query.clone(), 0.0);
        let default_calc = FlatDistanceCal::<Float16Type>::new(&vectors, query, dt);

        let ids: [u32; 16] = std::array::from_fn(|i| (i as u32 * 3 + 5) % 64);
        for len in 1..=16 {
            check_partial_batch_16(&enum_calc, &ids, len, "FlatFloatDistanceCalc");
            check_partial_batch_16(&default_calc, &ids, len, "trait default");
        }
    }

    /// The fp16 `Dot` `distance_batch_16` override (AMX-FP16 when active) must
    /// agree with 16 individual `distance()` calls. This is floating point, so
    /// the bar is a relative tolerance, not bit-exactness: the tile kernel sums
    /// f32-widened products in a different order than the per-id `f16::dot`.
    /// 5e-3 relative is ~an order below fp16's own representational error; the
    /// observed gap is ~1e-4. Dims include non-multiples of 32 (tail path).
    #[test]
    fn test_distance_batch_16_f16_dot_matches_single() {
        for dim in [32usize, 47, 64, 100, 128, 200, 768, 1536] {
            let (storage, query) = make_random_f16(64, dim, DistanceType::Dot, 0xF16 + dim as u64);
            let calc = storage.dist_calculator(query, 0.0);
            let ids: [u32; 16] = std::array::from_fn(|i| i as u32 + 7);

            let mut batch = [0f32; 16];
            calc.distance_batch_16(&ids, &mut batch, 16);

            for (i, &id) in ids.iter().enumerate() {
                let single = calc.distance(id);
                let rel = (batch[i] - single).abs() / (single.abs() + 1e-6);
                assert!(
                    rel <= 5e-3 || (batch[i] - single).abs() <= 1e-3,
                    "dim={dim} id={id}: batch {} vs single {} (rel {rel})",
                    batch[i],
                    single
                );
            }
        }
    }

    /// On an AMX-FP16 host (env not force-disabling), prove the accelerated tile
    /// path is actually the one the fp16 Dot override selected, not a silent
    /// fallback. `amx_fp16_available()` is a runtime probe (the compile-time
    /// `kernel_support` cfg is only set for the `lance-linalg` crate, so it can't
    /// be used to gate here). Reaching the end without SIGILL also proves the
    /// tile instructions executed safely.
    #[test]
    fn test_distance_batch_16_f16_amx_active() {
        if !lance_linalg::distance::dot_f16::amx_fp16_available() {
            return; // This CI host may not expose AMX-FP16.
        }
        let (storage, query) = make_random_f16(64, 128, DistanceType::Dot, 42);
        let calc = storage.dist_calculator(query, 0.0);
        let ids: [u32; 16] = std::array::from_fn(|i| i as u32);
        let mut batch = [0f32; 16];
        calc.distance_batch_16(&ids, &mut batch, 16);
        for (i, &id) in ids.iter().enumerate() {
            let single = calc.distance(id);
            assert!((batch[i] - single).abs() <= 1e-2, "id={id}");
        }
    }

    /// Non-Dot metrics keep the default per-id path, so the batch must be
    /// bit-for-bit identical to individual `distance()` calls (no AMX involved).
    #[test]
    fn test_distance_batch_16_f16_l2_falls_through() {
        let (storage, query) = make_random_f16(64, 100, DistanceType::L2, 5);
        let calc = storage.dist_calculator(query, 0.0);
        let ids: [u32; 16] = std::array::from_fn(|i| i as u32 + 3);
        let mut batch = [0f32; 16];
        calc.distance_batch_16(&ids, &mut batch, 16);
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(batch[i].to_bits(), calc.distance(id).to_bits(), "id={id}");
        }
    }
}
