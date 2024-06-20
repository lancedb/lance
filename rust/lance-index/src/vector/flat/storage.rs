// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory graph representations.

use std::sync::Arc;

use crate::vector::quantizer::QuantizerStorage;
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::utils::do_prefetch;
use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow_array::{
    types::{Float32Type, UInt64Type},
    Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, SchemaRef};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_linalg::distance::DistanceType;
use snafu::{location, Location};

use super::index::FlatMetadata;

pub const FLAT_COLUMN: &str = "flat";

/// All data are stored in memory
#[derive(Clone)]
pub struct FlatStorage {
    batch: RecordBatch,
    distance_type: DistanceType,

    // helper fields
    pub(super) row_ids: Arc<UInt64Array>,
    vectors: Arc<FixedSizeListArray>,
}

impl DeepSizeOf for FlatStorage {
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
        self.batch.get_array_memory_size()
    }
}

#[async_trait::async_trait]
impl QuantizerStorage for FlatStorage {
    type Metadata = FlatMetadata;
    async fn load_partition(
        _: &FileReader,
        _: std::ops::Range<usize>,
        _: DistanceType,
        _: &Self::Metadata,
    ) -> Result<Self> {
        unimplemented!("Flat will be used in new index builder which doesn't require this")
    }
}

impl FlatStorage {
    // deprecated, use `try_from_batch` instead
    pub fn new(vectors: FixedSizeListArray, distance_type: DistanceType) -> Self {
        let row_ids = Arc::new(UInt64Array::from_iter_values(0..vectors.len() as u64));
        let vectors = Arc::new(vectors);

        let batch = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids.clone() as ArrayRef, true),
            (FLAT_COLUMN, vectors.clone() as ArrayRef, true),
        ])
        .unwrap();

        Self {
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

impl VectorStore for FlatStorage {
    type DistanceCalculator<'a> = FlatDistanceCal<'a>;

    fn try_from_batch(batch: RecordBatch, distance_type: DistanceType) -> Result<Self> {
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
            batch,
            distance_type,
            row_ids,
            vectors,
        })
    }

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
        FlatDistanceCal::new(self.vectors.as_ref(), query, self.distance_type)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        FlatDistanceCal::new(
            self.vectors.as_ref(),
            self.vectors.value(id as usize),
            self.distance_type,
        )
    }

    /// Distance between two vectors.
    fn distance_between(&self, a: u32, b: u32) -> f32 {
        match self.vectors.value_type() {
            DataType::Float32 => {
                let vector1 = self.vectors.value(a as usize);
                let vector2 = self.vectors.value(b as usize);
                self.distance_type.func()(
                    vector1.as_primitive::<Float32Type>().values(),
                    vector2.as_primitive::<Float32Type>().values(),
                )
            }
            _ => unimplemented!(),
        }
    }
}

pub struct FlatDistanceCal<'a> {
    vectors: &'a [f32],
    query: Vec<f32>,
    dimension: usize,
    distance_fn: fn(&[f32], &[f32]) -> f32,
}

impl<'a> FlatDistanceCal<'a> {
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

    #[inline]
    fn get_vector(&self, id: u32) -> &[f32] {
        &self.vectors[self.dimension * id as usize..self.dimension * (id + 1) as usize]
    }
}

impl DistCalculator for FlatDistanceCal<'_> {
    #[inline]
    fn distance(&self, id: u32) -> f32 {
        let vector = self.get_vector(id);
        (self.distance_fn)(&self.query, vector)
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        let vector = self.get_vector(id);
        do_prefetch(vector.as_ptr_range())
    }
}
