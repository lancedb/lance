// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Flat Vector Index.

use arrow::array::as_primitive_array;
use arrow_array::{cast::as_struct_array, ArrayRef, RecordBatch, StructArray};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField};
use arrow_select::{concat::concat_batches, take::take};
use futures::stream::{repeat_with, StreamExt, TryStreamExt};

use super::Query;
use crate::arrow::*;
use crate::dataset::scanner::RecordBatchStream;
use crate::{Error, Result};

/// Run flat search over a stream of RecordBatch.
pub async fn flat_search(stream: &RecordBatchStream, query: &Query) -> Result<RecordBatch> {
    const SCORE_COLUMN: &str = "score";

    let schema_with_score = stream.schema().
    let batches = stream
        .zip(repeat_with(|| query.metric_type))
        .map(|(batch, mt)| async move {
            let k = query.key.clone();
            let mut batch = batch?;
            if batch.column_by_name(SCORE_COLUMN).is_some() {
                // Ignore the score calculated from inner vector index.
                batch = batch.drop_column(SCORE_COLUMN)?;
            }
            let vectors = batch
                .column_by_name(&query.column)
                .ok_or_else(|| {
                    Error::Schema(format!("column {} does not exist in dataset", query.column))
                })?
                .clone();
            let flatten_vectors = as_fixed_size_list_array(vectors.as_ref()).values();
            let scores = tokio::task::spawn_blocking(move || {
                mt.func()(&k, as_primitive_array(flatten_vectors.as_ref()), k.len()).unwrap()
            })
            .await? as ArrayRef;

            // TODO: use heap
            let indices = sort_to_indices(&scores, None, Some(query.k))?;
            let batch_with_score = batch.try_with_column(
                ArrowField::new(SCORE_COLUMN, DataType::Float32, false),
                scores,
            )?;
            let struct_arr = StructArray::from(batch_with_score);
            let selected_arr = take(&struct_arr, &indices, None)?;
            Ok::<RecordBatch, Error>(as_struct_array(&selected_arr).into())
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(stream.schema().clone()));
    }
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let scores = batch.column_by_name(SCORE_COLUMN).unwrap();
    let indices = sort_to_indices(scores, None, Some(query.k))?;

    let struct_arr = StructArray::from(batch);
    let selected_arr = take(&struct_arr, &indices, None)?;
    Ok(as_struct_array(&selected_arr).into())
}
