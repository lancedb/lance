// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::types::UInt32Type;
use arrow_array::UInt32Array;
use arrow_array::{cast::AsArray, Array, FixedSizeListArray, RecordBatch};
use arrow_schema::Field;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, RecordBatchExt};
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use lance_linalg::kmeans::compute_partitions;
use lance_linalg::MatrixView;
use num_traits::Float;
use snafu::{location, Location};
use std::sync::Arc;

use super::transform::Transformer;

pub const RESIDUAL_COLUMN: &str = "__residual_vector";

/// Compute the residual vector of a Vector Matrix to their centroids.
///
/// The residual vector is the difference between the original vector and the centroid.
///
#[derive(Clone)]
pub struct ResidualTransform<T: ArrowFloatType> {
    centroids: MatrixView<T>,

    /// Partition Column
    part_col: String,

    /// Vector Column
    vec_col: String,
}

impl<T: ArrowFloatType> std::fmt::Debug for ResidualTransform<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResidualTransform")
    }
}

impl<T: ArrowFloatType> ResidualTransform<T> {
    pub fn new(centroids: MatrixView<T>, part_col: &str, column: &str) -> Self {
        Self {
            centroids,
            part_col: part_col.to_owned(),
            vec_col: column.to_owned(),
        }
    }
}

/// Compute residual vectors from the original vectors and centroids.
///
/// ## Parameter
/// - `centroids`: The KMeans centroids.
/// - `vectors`: The original vectors to compute residual vectors.
/// - `distance_type`: The distance type to compute the residual vector.
/// - `partitions`: The partition ID for each vector, if present.
pub(crate) fn compute_residual(
    centroids: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
    distance_type: DistanceType,
    partitions: Option<&UInt32Array>,
) -> Result<FixedSizeListArray> {
    if centroids.value_length() != vectors.value_length() {
        return Err(Error::Index {
            message: format!(
                "Compute residual vector: centroid and vector length mismatch: centroid: {}, vector: {}",
                centroids.value_length(),
                vectors.value_length(),
            ),
            location: location!(),
        });
    }
    if centroids.value_type() != vectors.value_type() {
        return Err(Error::Index {
            message: format!(
                "Compute residual vector: centroids and vector type mismatch: centroid: {}, vector: {}",
                centroids.value_type(),
                vectors.value_type(),
            ),
            location: location!(),
        });
    }
    let part_id = partitions
        .map(|p| p.clone())
        .unwrap_or_else(|| UInt32Array::from(vec![0; vectors.len() as usize]));

    todo!()
}

impl<T: ArrowFloatType> Transformer for ResidualTransform<T> {
    /// Replace the original vector in the [`RecordBatch`] to residual vectors.
    ///
    /// The new [`RecordBatch`] will have a new column named [`RESIDUAL_COLUMN`].
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let part_ids = batch.column_by_name(&self.part_col).ok_or(Error::Index {
            message: format!(
                "Compute residual vector: partition id column not found: {}",
                self.part_col
            ),
            location: location!(),
        })?;
        let original = batch.column_by_name(&self.vec_col).ok_or(Error::Index {
            message: format!(
                "Compute residual vector: original vector column not found: {}",
                self.vec_col
            ),
            location: location!(),
        })?;
        let original_vectors = original.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "Compute residual vector: original vector column {} is not fixed size list: {}",
                self.vec_col,
                original.data_type(),
            ),
            location: location!(),
        })?;

        // BFloat16Array is not supported via `as_primitive()` cast yet, so we have to do
        // `downcast_ref()` for now.
        let flatten_data = original_vectors
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Compute residual vector: original vector column {} is not expected type: expect: {}, got {}",
                    self.vec_col,
                    T::FLOAT_TYPE,
                    original_vectors.value_type(),
                ),
                location: location!(),
            })?;
        let dim = original_vectors.value_length();
        let mut residual_arr: Vec<T::Native> = Vec::with_capacity(flatten_data.len());
        flatten_data
            .as_slice()
            .chunks_exact(dim as usize)
            .zip(part_ids.as_primitive::<UInt32Type>().values().iter())
            .for_each(|(vector, &part_id)| {
                let centroid = self.centroids.row_ref(part_id as usize).unwrap();
                // TODO: SIMD
                residual_arr.extend(
                    vector
                        .iter()
                        .zip(centroid.iter())
                        .map(|(v, cent)| *v - *cent),
                );
            });
        let residual_arr =
            FixedSizeListArray::try_new_from_values(T::ArrayType::from(residual_arr), dim)?;

        // Replace original column with residual column.
        let batch = batch.drop_column(&self.vec_col)?;

        let residual_field = Field::new(RESIDUAL_COLUMN, residual_arr.data_type().clone(), false);

        let batch = batch.try_with_column(residual_field, Arc::new(residual_arr))?;
        Ok(batch)
    }
}
