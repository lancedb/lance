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
use arrow_select::{concat::concat, take::take};
use async_trait::async_trait;
use futures::TryStreamExt;

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

    /// Index name.
    name: String,

    /// Vector column to search for.
    column: String,
}

impl<'a> FlatIndex<'a> {
    pub fn try_new(dataset: &'a Dataset, name: &str) -> Result<Self> {
        Ok(Self {
            dataset,
            name: name.to_string(),
            column: name.to_string(),
        })
    }
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
            .into_stream();

        let score_and_row_ids = stream
            .and_then(|batch| async move {
                let k = params.key.clone();
                let batch = batch.clone();
                let vectors = batch
                    .column_by_name(&params.column)
                    .ok_or_else(|| {
                        Error::Schema(format!("column {} does not exist in dataset", self.column,))
                    })?
                    .clone();
                let scores = tokio::task::spawn_blocking(move || {
                    l2_distance(&k, as_fixed_size_list_array(&vectors)).unwrap()
                })
                .await?;
                // TODO: only pick top-k in each batch first.
                let row_id_array = batch["_rowid"].clone();
                Ok((scores as ArrayRef, row_id_array))
            })
            .try_collect::<Vec<_>>()
            .await?;

        let scores_arrays = score_and_row_ids
            .iter()
            .map(|(score, _)| score.as_ref())
            .collect::<Vec<_>>();
        let row_ids_arrays = score_and_row_ids
            .iter()
            .map(|(_, row_id)| row_id.as_ref())
            .collect::<Vec<_>>();
        let scores = concat(&scores_arrays)?;
        let row_ids = concat(&row_ids_arrays)?;

        let indices = sort_to_indices(&scores, None, Some(params.k))?;

        let struct_arr = StructArray::try_from(vec![
            (ArrowField::new("_rowid", DataType::UInt64, false), row_ids),
            (ArrowField::new("score", DataType::Float32, false), scores),
        ])
        .map_err(|e| Error::IO(format!("Can not build struct array: {}", e.to_string())))?;
        let taken_scores = take(&struct_arr, &indices, None)?;
        Ok(as_struct_array(&taken_scores).into())
    }
}
