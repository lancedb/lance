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

use std::any::Any;
use std::sync::Arc;

use arrow::array::as_primitive_array;
use arrow_array::{cast::as_struct_array, ArrayRef, RecordBatch, StructArray};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField};
use arrow_select::{concat::concat_batches, take::take};
use async_trait::async_trait;
use futures::stream::{repeat_with, Stream, StreamExt, TryStreamExt};

use super::{Query, VectorIndex, SCORE_COL};
use crate::arrow::*;
use crate::dataset::Dataset;
use crate::io::object_reader::ObjectReader;
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
    let batches = stream
        .zip(repeat_with(|| query.metric_type))
        .map(|(batch, mt)| async move {
            let k = query.key.clone();
            let mut batch = batch?;
            if batch.column_by_name(SCORE_COL).is_some() {
                // Ignore the score calculated from inner vector index.
                batch = batch.drop_column(SCORE_COL)?;
            }
            let vectors = batch
                .column_by_name(&query.column)
                .ok_or_else(|| {
                    Error::Schema(format!("column {} does not exist in dataset", query.column))
                })?
                .clone();
            let flatten_vectors = as_fixed_size_list_array(vectors.as_ref()).values().clone();
            let scores = tokio::task::spawn_blocking(move || {
                mt.func()(&k, as_primitive_array(flatten_vectors.as_ref()), k.len()).unwrap()
            })
            .await? as ArrayRef;

            // TODO: use heap
            let indices = sort_to_indices(&scores, None, Some(query.k))?;
            let batch_with_score = batch
                .try_with_column(ArrowField::new(SCORE_COL, DataType::Float32, false), scores)?;
            let struct_arr = StructArray::from(batch_with_score);
            let selected_arr = take(&struct_arr, &indices, None)?;
            Ok::<RecordBatch, Error>(as_struct_array(&selected_arr).into())
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let scores = batch.column_by_name(SCORE_COL).unwrap();
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

    fn as_any(&self) -> &dyn Any {
        // Can not use `self` because of the lifetime.
        todo!()
    }

    fn is_loadable(&self) -> bool {
        false
    }

    async fn load(
        &self,
        _reader: &dyn ObjectReader,
        _offset: usize,
        _length: usize,
    ) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::Index("Flat index does not support load".to_string()))
    }
}
