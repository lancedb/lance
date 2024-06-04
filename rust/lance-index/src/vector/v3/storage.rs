// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, SchemaRef};
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use num_traits::Num;
use snafu::{location, Location};

use crate::vector::quantizer::Quantization;

/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;
    fn prefetch(&self, _id: u32) {}
}

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory raw vectors or on disk PQ code.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
///
/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait VectorStore: Send + Sync {
    type DistanceCalculator<'a>: DistCalculator
    where
        Self: 'a;

    fn try_from_batch(batch: RecordBatch, distance_type: DistanceType) -> Result<Self>
    where
        Self: Sized;

    fn as_any(&self) -> &dyn Any;

    fn schema(&self) -> &SchemaRef;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>>;

    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return [DistanceType].
    fn distance_type(&self) -> DistanceType;

    /// Get the lance ROW ID from one vector.
    fn row_id(&self, id: u32) -> u64;

    fn row_ids(&self) -> impl Iterator<Item = &u64>;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_>;

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_>;

    fn distance_between(&self, a: u32, b: u32) -> f32;

    fn dist_calculator_from_native<T: Num>(&self, _query: &[T]) -> Self::DistanceCalculator<'_> {
        todo!("Implement this")
    }
}

pub struct StorageBuilder<Q: Quantization> {
    column: String,
    distance_type: DistanceType,
    quantizer: Q,
}

impl<Q: Quantization> StorageBuilder<Q> {
    pub fn new(column: String, distance_type: DistanceType, quantizer: Q) -> Self {
        Self {
            column,
            distance_type,
            quantizer,
        }
    }

    pub fn build(&self, batch: &RecordBatch) -> Result<Q::Storage> {
        let vectors = batch.column_by_name(&self.column).ok_or(Error::Schema {
            message: format!("column {} not found", self.column),
            location: location!(),
        })?;
        let code_array = self.quantizer.quantize(vectors.as_ref())?;
        let batch = batch.drop_column(&self.column)?.try_with_column(
            Field::new(
                self.quantizer.column(),
                code_array.data_type().clone(),
                true,
            ),
            code_array,
        )?;
        Q::Storage::try_from_batch(batch, self.distance_type)
    }
}
