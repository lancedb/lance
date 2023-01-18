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

use std::sync::Arc;

use arrow_array::{
    cast::{as_primitive_array, as_struct_array},
    Float32Array, RecordBatch,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use futures::stream::StreamExt;

use super::{Query, VectorIndex};
use crate::arrow::*;
use crate::dataset::Dataset;
use crate::Result;

/// Flat Vector Index.
///
/// Flat index is a meta index. It does not build extra index structure,
/// and just uses the full scan to compute the distances.
///
/// Reference:
///   - <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
pub struct FlatIndex<'a> {
    dataset: &'a Dataset,

    /// Index name.
    name: String,

    /// Vector column to search for.
    column: String,
}

impl<'a> VectorIndex for FlatIndex<'_> {
    async fn search(&self, params: &Query) -> Result<RecordBatch> {
        let stream = self
            .dataset
            .scan()
            .project(&[&self.column])?
            .with_row_id()
            .into_stream();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("_rowid", DataType::UInt64, false),
            ArrowField::new("score", DataType::Float32, false),
        ]));
        let key_arr: &Float32Array = as_primitive_array(&params.key);
        let all_scores = stream
            .map(|b| async {
                if let Ok(batch) = b {
                    let key_arr = key_arr.clone();
                    let value_arr = batch.column_with_name(&self.column).unwrap().clone();
                    let scores = tokio::task::spawn_blocking(move || {
                        let targets = downcast_array::<FixedSizeListArray>(&value_arr);
                        euclidean_distance(&key_arr, &targets).unwrap()
                    })
                    .await
                    .unwrap();
                    Ok(RecordBatch::try_new(
                        schema.clone(),
                        vec![batch.column_with_name("_rowid").unwrap().clone(), scores],
                    )?)
                } else {
                    b
                }
            })
            .buffered(10)
            .collect::<Vec<_>>()
            .await;
        let scores = concat_batches(
            &schema,
            all_scores
                .iter()
                .map(|s| s.as_ref().unwrap().clone())
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        let scores_arr = scores.column_with_name("score").unwrap();
        let indices = sort_to_indices(scores_arr, None, Some(params.k))?;

        let struct_arr = StructArray::from(scores);
        let taken_scores = take(&struct_arr, &indices, None)?;
        Ok(as_struct_array(&taken_scores).into())
    }
}
