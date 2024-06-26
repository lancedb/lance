// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Vector Index.
//!

use std::sync::Arc;

use arrow_array::{
    cast::AsArray, make_array, Array, ArrayRef, FixedSizeListArray, RecordBatch, StructArray,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, SchemaRef};
use arrow_select::{concat::concat, take::take};
use futures::{
    future,
    stream::{StreamExt, TryStreamExt},
};
use lance_arrow::*;
use lance_core::{Error, Result, ROW_ID};
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::DistanceType;
use snafu::{location, Location};
use tracing::instrument;

use super::{Query, DIST_COL};

pub mod index;
pub mod storage;

fn distance_field() -> ArrowField {
    ArrowField::new(DIST_COL, DataType::Float32, true)
}

#[instrument(level = "debug", skip_all)]
pub async fn flat_search(
    stream: impl RecordBatchStream + 'static,
    query: &Query,
) -> Result<RecordBatch> {
    let input_schema = stream.schema();
    let dt = query.metric_type;
    let key = query.key.clone();

    let batches = stream
        .try_filter(|batch| future::ready(batch.num_rows() > 0))
        .map(|batch| {
            let key = key.clone();
            let column = query.column.clone();
            async move { flat_search_batch(key, dt, &column, batch?).await }
        })
        .buffer_unordered(num_cpus::get())
        .try_collect::<Vec<_>>()
        .await?;

    if batches.is_empty() {
        if input_schema.column_with_name(DIST_COL).is_none() {
            let schema_with_distance = input_schema.try_with_column(distance_field())?;
            return Ok(RecordBatch::new_empty(schema_with_distance.into()));
        } else {
            return Ok(RecordBatch::new_empty(input_schema));
        }
    }

    // TODO: waiting to do this until the end adds quite a bit of latency. We should
    // do this in a streaming fashion. See also: https://github.com/lancedb/lance/issues/1324
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let distances = batch.column_by_name(DIST_COL).unwrap();
    let indices = sort_to_indices(distances, None, Some(query.k))?;

    let struct_arr = StructArray::from(batch);
    let selected_arr = take(&struct_arr, &indices, None)?;
    Ok(selected_arr.as_struct().into())
}

#[instrument(level = "debug", skip(key, batch))]
pub async fn flat_search_batch(
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

        let batch = batch
            .try_with_column(distance_field(), distances)
            .map_err(|e| Error::Execution {
                message: format!("Failed to adding distance column: {}", e),
                location: location!(),
            });
        println!("batch: {:?}", batch);
        batch
    })
    .await
    .unwrap()
}

fn concat_batches<'a>(
    schema: &SchemaRef,
    input_batches: impl IntoIterator<Item = &'a RecordBatch>,
) -> Result<RecordBatch> {
    let batches: Vec<&RecordBatch> = input_batches.into_iter().collect();
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }
    let field_num = schema.fields().len();
    let mut arrays = Vec::with_capacity(field_num);
    for i in 0..field_num {
        let in_arrays = &batches
            .iter()
            .map(|batch| batch.column(i).as_ref())
            .collect::<Vec<_>>();
        let data_type = in_arrays[0].data_type();
        let array = match data_type {
            DataType::FixedSizeList(f, size) if f.data_type().is_floating() => {
                concat_fsl(in_arrays, *size)?
            }
            _ => concat(in_arrays)?,
        };
        arrays.push(array);
    }
    Ok(RecordBatch::try_new(schema.clone(), arrays)?)
}

/// Optimized version of concatenating fixed-size list. Upstream does not yet
/// pre-size FSL arrays correctly. We can remove this once they do.
fn concat_fsl(arrays: &[&dyn Array], size: i32) -> Result<ArrayRef> {
    let values_arrays: Vec<_> = arrays
        .iter()
        .map(|&arr| as_fixed_size_list_array(arr).values().as_ref())
        .collect();
    let values = concat(&values_arrays)?;
    Ok(Arc::new(FixedSizeListArray::try_new_from_values(
        values, size,
    )?))
}
