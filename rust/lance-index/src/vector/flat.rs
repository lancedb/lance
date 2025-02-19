// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Vector Index.
//!

use std::sync::Arc;

use arrow::{array::AsArray, buffer::NullBuffer};
use arrow_array::{make_array, Array, ArrayRef, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField};
use lance_arrow::*;
use lance_core::{Error, Result, ROW_ID};
use lance_linalg::distance::{multivec_distance, DistanceType};
use snafu::location;
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
    let vectors = batch
        .column_by_name(column)
        .ok_or_else(|| Error::Schema {
            message: format!("column {} does not exist in dataset", column),
            location: location!(),
        })?
        .clone();

    let validity_buffer = if let Some(rowids) = batch.column_by_name(ROW_ID) {
        NullBuffer::union(rowids.nulls(), vectors.nulls())
    } else {
        vectors.nulls().cloned()
    };

    tokio::task::spawn_blocking(move || {
        // A selection vector may have been applied to _rowid column, so we need to
        // push that onto vectors if possible.

        let vectors = vectors
            .into_data()
            .into_builder()
            .null_bit_buffer(validity_buffer.map(|b| b.buffer().clone()))
            .build()
            .map(make_array)?;
        let distances = match vectors.data_type() {
            DataType::FixedSizeList(_, _) => {
                let vectors = vectors.as_fixed_size_list();
                dt.arrow_batch_func()(key.as_ref(), vectors)? as ArrayRef
            }
            DataType::List(_) => {
                let vectors = vectors.as_list();
                let dists = multivec_distance(key.as_ref(), vectors, dt)?;
                Arc::new(Float32Array::from(dists))
            }
            _ => {
                unreachable!()
            }
        };

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
