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

use arrow::array::as_primitive_array;
use arrow::datatypes::Float32Type;
use arrow_array::{cast::as_struct_array, ArrayRef, RecordBatch, StructArray};
use arrow_array::{Array, FixedSizeListArray};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, SchemaRef};
use arrow_select::{concat::concat, take::take};
use futures::future;
use futures::stream::{repeat_with, StreamExt, TryStreamExt};
use lance_linalg::distance::DistanceType;
use snafu::{location, Location};
use tracing::instrument;

use super::{Query, DIST_COL};
use crate::arrow::*;
use crate::io::RecordBatchStream;
use crate::{Error, Result};

fn distance_field() -> ArrowField {
    ArrowField::new(DIST_COL, DataType::Float32, false)
}

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
        let schema_with_distance = input_schema.try_with_column(distance_field())?;
        return Ok(RecordBatch::new_empty(schema_with_distance.into()));
    }

    // TODO: waiting to do this until the end adds quite a bit of latency. We should
    // do this in a streaming fashion. See also: https://github.com/lancedb/lance/issues/1324
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let distances = batch.column_by_name(DIST_COL).unwrap();
    let indices = sort_to_indices(distances, None, Some(query.k))?;

    let struct_arr = StructArray::from(batch);
    let selected_arr = take(&struct_arr, &indices, None)?;
    Ok(as_struct_array(&selected_arr).into())
}

#[instrument(skip(query, batch))]
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
        })?
        .clone();
    let flatten_vectors = as_fixed_size_list_array(vectors.as_ref()).values().clone();
    tokio::task::spawn_blocking(move || {
        let distances = mt.batch_func()(
            key.values(),
            as_primitive_array::<Float32Type>(flatten_vectors.as_ref()).values(),
            key.len(),
        ) as ArrayRef;

        let indices = sort_to_indices(&distances, None, Some(k))?;
        let batch_with_distance = batch.try_with_column(distance_field(), distances)?;
        let struct_arr = StructArray::from(batch_with_distance);
        let selected_arr = take(&struct_arr, &indices, None)?;
        Ok::<RecordBatch, Error>(as_struct_array(&selected_arr).into())
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
                concat_fsl(&in_arrays, *size)?
            }
            _ => concat(&in_arrays)?,
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
