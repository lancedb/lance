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

use arrow_array::{cast::as_struct_array, ArrayRef, RecordBatch, StructArray};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField};
use arrow_select::{concat::concat_batches, take::take};
use async_trait::async_trait;
use futures::stream::{Stream, StreamExt, TryStreamExt};

use super::{Query, VectorIndex};
use crate::arrow::*;
use crate::dataset::Dataset;
use crate::utils::distance::l2_distance;
use crate::{Error, Result};

/// Flat Vector Index.
///
/// Flat index is a meta index. It does not build extra index structure,
/// and does exhaustive search.
///
/// Flat index always provides 100% recall because exhaustive search over the
/// uncompressed original vectors.
///
/// Reference:
///   - <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
#[derive(Debug)]
pub struct FlatIndex<'a> {
    dataset: &'a Dataset,

    /// Vector column to search for.
    column: String,
}

impl<'a> FlatIndex<'a> {
    pub fn try_new(dataset: &'a Dataset, name: &str) -> Result<Self> {
        Ok(Self {
            dataset,
            column: name.to_string(),
        })
    }
}

pub async fn flat_search(
    stream: impl Stream<Item = Result<RecordBatch>>,
    query: &Query,
) -> Result<RecordBatch> {
    let score_column = "score";

    let batches = stream
        .map(|batch| async move {
            let k = query.key.clone();
            let mut batch = batch?;
            if batch.column_by_name(score_column).is_some() {
                batch = batch.drop_column(score_column)?;
            }
            let vectors = batch
                .column_by_name(&query.column)
                .ok_or_else(|| {
                    Error::Schema(format!("column {} does not exist in dataset", query.column,))
                })?
                .clone();
            let scores = tokio::task::spawn_blocking(move || {
                l2_distance(&k, as_fixed_size_list_array(&vectors)).unwrap()
            })
            .await? as ArrayRef;

            // TODO: use heap
            let indices = sort_to_indices(&scores, None, Some(query.k))?;
            let batch_with_score = batch.try_with_column(
                ArrowField::new(score_column, DataType::Float32, false),
                scores,
            )?;
            let struct_arr = StructArray::from(batch_with_score);
            let selected_arr = take(&struct_arr, &indices, None)?;
            Ok::<RecordBatch, Error>(as_struct_array(&selected_arr).into())
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let scores = batch.column_by_name(score_column).unwrap();
    let indices = sort_to_indices(scores, None, Some(query.k))?;

    let struct_arr = StructArray::from(batch);
    let selected_arr = take(&struct_arr, &indices, None)?;
    Ok(as_struct_array(&selected_arr).into())
}

#[async_trait]
impl VectorIndex for FlatIndex<'_> {
    /// Search the flat index.
    async fn search(&self, params: &Query) -> Result<RecordBatch> {
        let stream = self
            .dataset
            .scan()
            .project(&[&self.column])?
            .with_row_id()
            .try_into_stream()
            .await?;
        flat_search(stream, params).await
    }
}
