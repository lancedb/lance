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

//! Flat Vector Index.
//!

use std::sync::Arc;

use arrow_array::{
    cast::AsArray, make_array, Array, ArrayRef, FixedSizeListArray, RecordBatch, StructArray,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, SchemaRef, SortOptions};
use arrow_select::{concat::concat, take::take};
use futures::{
    future,
    stream::{repeat_with, StreamExt, TryStreamExt},
};
use lance_arrow::*;
use lance_core::{Error, Result, ROW_ID};
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::DistanceType;
use snafu::{location, Location};
use tracing::instrument;

use super::{Query, DIST_COL};

fn distance_field() -> ArrowField {
    ArrowField::new(DIST_COL, DataType::Float32, true)
}

#[instrument(level = "debug", skip_all)]
pub async fn flat_search(
    stream: impl RecordBatchStream + 'static,
    query: &Query,
) -> Result<RecordBatch> {
    let input_schema = stream.schema();
    let batches = stream
        .try_filter(|batch| future::ready(batch.num_rows() > 0))
        .zip(repeat_with(|| query.metric_type))
        .map(|(batch, mt)| async move { flat_search_batch(query, mt, batch?).await })
        .buffer_unordered(16)
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

#[instrument(level = "debug", skip(query, batch))]
async fn flat_search_batch(
    query: &Query,
    mt: DistanceType,
    mut batch: RecordBatch,
) -> Result<RecordBatch> {
    let key = query.key.clone();
    let k = query.k;
    if batch.column_by_name(DIST_COL).is_some() {
        // Ignore the distance calculated from inner vector index.
        batch = batch.drop_column(DIST_COL)?;
    }
    let vectors = batch
        .column_by_name(&query.column)
        .ok_or_else(|| Error::Schema {
            message: format!("column {} does not exist in dataset", query.column),
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
        let distances = mt.arrow_batch_func()(key.as_ref(), &vectors)? as ArrayRef;

        // We don't want any nulls in result, so limit to k or the number of valid values.
        let k = std::cmp::min(k, distances.len() - distances.null_count());

        let sort_options = SortOptions {
            nulls_first: false,
            ..Default::default()
        };
        let indices = sort_to_indices(&distances, Some(sort_options), Some(k))?;

        let batch_with_distance = batch.try_with_column(distance_field(), distances)?;
        let struct_arr = StructArray::from(batch_with_distance);
        let selected_arr = take(&struct_arr, &indices, None)?;
        Ok::<RecordBatch, Error>(selected_arr.as_struct().into())
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
