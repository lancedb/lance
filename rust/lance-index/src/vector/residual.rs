// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{ArrowPrimitiveType, Float16Type, Float32Type, Float64Type, UInt32Type},
    Array, FixedSizeListArray, PrimitiveArray, RecordBatch, UInt32Array,
};
use arrow_schema::{DataType, Field};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, Dot, L2};
use lance_linalg::kmeans::compute_partitions;
use num_traits::Float;
use rayon::prelude::*;
use snafu::{location, Location};

use super::transform::Transformer;

pub const RESIDUAL_COLUMN: &str = "__residual_vector";

/// Compute the residual vector of a Vector Matrix to their centroids.
///
/// The residual vector is the difference between the original vector and the centroid.
///
#[derive(Clone)]
pub struct ResidualTransform {
    /// Flattend centroids.
    centroids: FixedSizeListArray,

    /// Partition Column
    part_col: String,

    /// Vector Column
    vec_col: String,
}

impl std::fmt::Debug for ResidualTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResidualTransform")
    }
}

impl ResidualTransform {
    pub fn new(centroids: FixedSizeListArray, part_col: &str, column: &str) -> Self {
        Self {
            centroids,
            part_col: part_col.to_owned(),
            vec_col: column.to_owned(),
        }
    }
}

fn do_compute_residual<T: ArrowPrimitiveType>(
    centroids: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
    distance_type: Option<DistanceType>,
    partitions: Option<&UInt32Array>,
) -> Result<FixedSizeListArray>
where
    T::Native: Float + L2 + Dot,
{
    let dimension = centroids.value_length() as usize;
    let centroids_slice = centroids.values().as_primitive::<T>().values();
    let vectors_slice = vectors.values().as_primitive::<T>().values();

    let part_ids = partitions.cloned().unwrap_or_else(|| {
        compute_partitions(
            centroids_slice,
            vectors_slice,
            dimension,
            distance_type.expect("provide either partitions or distance type"),
        )
        .into()
    });

    let residuals = vectors_slice
        .par_chunks(dimension)
        .enumerate()
        .flat_map(|(idx, vector)| {
            let part_id = part_ids.value(idx) as usize;
            let c = &centroids_slice[part_id * dimension..(part_id + 1) * dimension];
            vector
                .par_iter()
                .zip(c.par_iter())
                .map(|(v, cent)| *v - *cent)
        })
        .collect::<Vec<_>>();
    let residual_arr = PrimitiveArray::<T>::from_iter_values(residuals);
    Ok(FixedSizeListArray::try_new_from_values(
        residual_arr,
        dimension as i32,
    )?)
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
    distance_type: Option<DistanceType>,
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
    // TODO: Bf16 is not supported yet.
    match (centroids.value_type(), vectors.value_type()) {
        (DataType::Float16, DataType::Float16) => {
            do_compute_residual::<Float16Type>(centroids, vectors, distance_type, partitions)
        }
        (DataType::Float32, DataType::Float32) => {
            do_compute_residual::<Float32Type>(centroids, vectors, distance_type, partitions)
        }
        (DataType::Float64, DataType::Float64) => {
            do_compute_residual::<Float64Type>(centroids, vectors, distance_type, partitions)
        }
        _ => Err(Error::Index {
            message: format!(
                "Compute residual vector: centroids and vector type mismatch: centroid: {}, vector: {}",
                centroids.value_type(),
                vectors.value_type(),
            ),
            location: location!(),
        })
    }
}

impl Transformer for ResidualTransform {
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

        let part_ids_ref = part_ids.as_primitive::<UInt32Type>();
        let residual_arr =
            compute_residual(&self.centroids, original_vectors, None, Some(part_ids_ref))?;

        // Replace original column with residual column.
        let batch = batch.drop_column(&self.vec_col)?;

        let residual_field = Field::new(RESIDUAL_COLUMN, residual_arr.data_type().clone(), false);
        let batch = batch.try_with_column(residual_field, Arc::new(residual_arr))?;
        Ok(batch)
    }
}
