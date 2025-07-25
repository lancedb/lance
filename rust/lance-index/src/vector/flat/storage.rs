// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use super::index::FlatMetadata;
use crate::frag_reuse::FragReuseIndex;
use crate::vector::quantizer::QuantizerStorage;
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::utils::do_prefetch;
use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow::datatypes::UInt8Type;
use arrow_array::ArrowPrimitiveType;
use arrow_array::{
    types::{Float32Type, UInt64Type},
    Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array,
};
use arrow_schema::SchemaRef;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_linalg::distance::hamming::hamming;
use lance_linalg::distance::DistanceType;
use snafu::location;

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
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
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
                .ok_or(Error::Schema {
                    message: format!("column {} not found", ROW_ID),
                    location: location!(),
                })?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        let vectors = Arc::new(
            batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::Schema {
                    message: "column flat not found".to_string(),
                    location: location!(),
                })?
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
        _: &FileReader,
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
    type DistanceCalculator<'a> = FlatDistanceCal<'a, Float32Type>;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        Ok([self.batch.clone()].into_iter())
    }

    fn append_batch(&self, batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        // TODO: use chunked storage
        let new_batch = concat_batches(&batch.schema(), vec![&self.batch, &batch].into_iter())?;
        let mut storage = self.clone();
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

    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new(self.vectors.as_ref(), query, self.distance_type)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new(
            self.vectors.as_ref(),
            self.vectors.value(id as usize),
            self.distance_type,
        )
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
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
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
                .ok_or(Error::Schema {
                    message: format!("column {} not found", ROW_ID),
                    location: location!(),
                })?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        let vectors = Arc::new(
            batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::Schema {
                    message: "column flat not found".to_string(),
                    location: location!(),
                })?
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
        _: &FileReader,
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
        let new_batch = concat_batches(&batch.schema(), vec![&self.batch, &batch].into_iter())?;
        let mut storage = self.clone();
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

    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new(self.vectors.as_ref(), query, self.distance_type)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new(
            self.vectors.as_ref(),
            self.vectors.value(id as usize),
            self.distance_type,
        )
    }
}

pub struct FlatDistanceCal<'a, T: ArrowPrimitiveType> {
    vectors: &'a [T::Native],
    query: Vec<T::Native>,
    dimension: usize,
    #[allow(clippy::type_complexity)]
    distance_fn: fn(&[T::Native], &[T::Native]) -> f32,
}

impl<'a> FlatDistanceCal<'a, Float32Type> {
    fn new(vectors: &'a FixedSizeListArray, query: ArrayRef, distance_type: DistanceType) -> Self {
        // Gained significant performance improvement by using strong typed primitive slice.
        // TODO: to support other data types other than `f32`, make FlatDistanceCal a generic struct.
        let flat_array = vectors.values().as_primitive::<Float32Type>();
        let dimension = vectors.value_length() as usize;
        Self {
            vectors: flat_array.values(),
            query: query.as_primitive::<Float32Type>().values().to_vec(),
            dimension,
            distance_fn: distance_type.func(),
        }
    }
}

impl<'a> FlatDistanceCal<'a, UInt8Type> {
    fn new(vectors: &'a FixedSizeListArray, query: ArrayRef, _distance_type: DistanceType) -> Self {
        // Gained significant performance improvement by using strong typed primitive slice.
        // TODO: to support other data types other than `f32`, make FlatDistanceCal a generic struct.
        let flat_array = vectors.values().as_primitive::<UInt8Type>();
        let dimension = vectors.value_length() as usize;
        Self {
            vectors: flat_array.values(),
            query: query.as_primitive::<UInt8Type>().values().to_vec(),
            dimension,
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

impl<T: ArrowPrimitiveType> DistCalculator for FlatDistanceCal<'_, T> {
    #[inline]
    fn distance(&self, id: u32) -> f32 {
        let vector = self.get_vector(id);
        (self.distance_fn)(&self.query, vector)
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        let query = &self.query;
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
