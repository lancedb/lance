// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Vector Index.
//!

use arrow_array::{make_array, Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField};
use lance_arrow::*;
use lance_core::{Error, Result, ROW_ID};
use lance_linalg::distance::DistanceType;
use snafu::{location, Location};
use tracing::instrument;

use super::DIST_COL;

pub mod index;
pub mod storage;

fn distance_field() -> ArrowField {
    ArrowField::new(DIST_COL, DataType::Float32, true)
}

#[instrument(level = "debug", skip_all)]
pub async fn compute_distance(
    key: ArrayRef,
    dt: DistanceType,
    column: &str,
    mut batch: RecordBatch,
) -> Result<RecordBatch> {
    if batch.column_by_name(DIST_COL).is_some() {
        // Ignore the distance calculated from inner vector index.
        batch = batch.drop_column(DIST_COL)?;
    }
    let vectors = batch.column_by_name(column).ok_or_else(|| Error::Schema {
        message: format!("column {} does not exist in dataset", column),
        location: location!(),
    })?;

    // A selection vector may have been applied to _rowid column, so we need to
    // push that onto vectors if possible.
    let vectors = as_fixed_size_list_array(vectors.as_ref()).clone();
    let validity_buffer = if let Some(rowids) = batch.column_by_name(ROW_ID) {
        rowids.nulls().map(|nulls| nulls.buffer().clone())
    } else {
        None
    };

    let vectors = vectors
        .into_data()
        .into_builder()
        .null_bit_buffer(validity_buffer)
        .build()
        .map(make_array)?;
    let vectors = as_fixed_size_list_array(vectors.as_ref()).clone();

    tokio::task::spawn_blocking(move || {
        let distances = dt.arrow_batch_func()(key.as_ref(), &vectors)? as ArrayRef;

        batch
            .try_with_column(distance_field(), distances)
            .map_err(|e| Error::Execution {
                message: format!("Failed to adding distance column: {}", e),
                location: location!(),
            })
    })
    .await
    .unwrap()
}
