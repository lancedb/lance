// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use arrow_array::types::UInt32Type;
use arrow_array::{
    cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float32Array, RecordBatch,
};
use arrow_schema::Field;
use async_trait::async_trait;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result};
use lance_linalg::MatrixView;
use snafu::{location, Location};
use std::sync::Arc;

use super::transform::Transformer;

pub const RESIDUAL_COLUMN: &str = "__residual_vector";

/// Compute the residual vector of a Vector Matrix to their centroids.
///
/// The residual vector is the difference between the original vector and the centroid.
///
#[derive(Clone)]
pub struct ResidualTransform {
    centroids: MatrixView<Float32Type>,

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
    pub fn new(centroids: MatrixView<Float32Type>, part_col: &str, column: &str) -> Self {
        Self {
            centroids,
            part_col: part_col.to_owned(),
            vec_col: column.to_owned(),
        }
    }
}

#[async_trait]
impl Transformer for ResidualTransform {
    /// Replace the original vector in the [`RecordBatch`] to residual vectors.
    ///
    /// The new [`RecordBatch`] will have a new column named [`RESIDUAL_COLUMN`].
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
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
        let original_matrix = MatrixView::<Float32Type>::try_from(original_vectors)?;
        let mut residual_arr: Vec<f32> =
            Vec::with_capacity(original_matrix.num_rows() * original_matrix.ndim());
        original_matrix
            .iter()
            .zip(part_ids.as_primitive::<UInt32Type>().values().iter())
            .for_each(|(vector, &part_id)| {
                let centroid = self.centroids.row(part_id as usize).unwrap();
                // TODO: SIMD
                residual_arr.extend(
                    vector
                        .iter()
                        .zip(centroid.iter())
                        .map(|(v, cent)| *v - *cent),
                );
            });
        let residual_arr = FixedSizeListArray::try_new_from_values(
            Float32Array::from(residual_arr),
            original_matrix.ndim() as i32,
        )?;

        // Replace original column with residual column.
        let batch = batch.drop_column(&self.vec_col)?;

        let residual_field = Field::new(RESIDUAL_COLUMN, residual_arr.data_type().clone(), false);

        let batch = batch.try_with_column(residual_field, Arc::new(residual_arr))?;
        Ok(batch)
    }
}
