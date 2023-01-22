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

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow_array::cast::as_struct_array;
use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField};
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;

use super::{ExecNode, NodeType};
use crate::arrow::*;
use crate::index::vector::Query;
use crate::utils::distance::l2_distance;
use crate::{Error, Result};

/// KNN node for post-filtering.
pub struct KNNFlat {
    rx: Receiver<Result<RecordBatch>>,

    _bg_thread: JoinHandle<()>,
}

impl KNNFlat {
    pub(crate) fn new(
        child: impl Stream<Item = Result<RecordBatch>> + Unpin + Send + 'static,
        query: &Query,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);

        let q = query.clone();
        let bg_thread = tokio::spawn(async move {
            let batches = child
                .zip(stream::repeat_with(|| q.clone()))
                .then(|(batch, q)| async move {
                    let batch = batch?;
                    let vectors = batch.column_with_name(&q.column).unwrap().clone();
                    let scores = tokio::task::spawn_blocking(move || {
                        l2_distance(&q.key, as_fixed_size_list_array(&vectors)).unwrap()
                    })
                    .await? as ArrayRef;

                    // TODO: use heap
                    let batch_with_score = batch.try_with_column(
                        ArrowField::new("score", DataType::Float32, false),
                        scores.clone(),
                    )?;
                    let indices = sort_to_indices(&scores, None, Some(q.k))?;
                    let struct_arr = StructArray::from(batch_with_score);
                    let selected_arr = take(&struct_arr, &indices, None)?;
                    Ok::<RecordBatch, Error>(as_struct_array(&selected_arr).into())
                })
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
            let scores = batch.column_by_name("score").unwrap();
            let indices = sort_to_indices(scores, None, Some(q.k)).unwrap();

            let struct_arr = StructArray::from(batch);
            let selected_arr = take(&struct_arr, &indices, None).unwrap();

            if !tx.is_closed() {
                if let Err(e) = tx.send(Ok(as_struct_array(&selected_arr).into())).await {
                    eprintln!("KNNFlat tx.send error: {}", e)
                };
            }
            drop(tx);
        });

        Self {
            rx,
            _bg_thread: bg_thread,
        }
    }
}

impl ExecNode for KNNFlat {
    fn node_type(&self) -> NodeType {
        NodeType::KnnFlat
    }
}

impl Stream for KNNFlat {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{cast::as_primitive_array, FixedSizeListArray, Int32Array, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use crate::arrow::RecordBatchBuffer;
    use crate::arrow::*;
    use crate::dataset::{Dataset, WriteParams};
    use crate::utils::testing::generate_random_array;

    #[tokio::test]
    async fn knn_flat_search() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Box::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));

        let mut batches = RecordBatchBuffer::new(
            (0..20)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                            Arc::new(
                                FixedSizeListArray::try_new(generate_random_array(128 * 20), 128)
                                    .unwrap(),
                            ),
                            Arc::new(StringArray::from_iter_values(
                                (i * 20..(i + 1) * 20).map(|i| format!("s3://bucket/file-{}", i)),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect(),
        );
        // println!("Batches: {:?}", batches);

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        Dataset::create(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let vector_arr = batches.batches[0].column_by_name("vector").unwrap();
        let q = as_fixed_size_list_array(&vector_arr).value(5);
        let stream = dataset
            .scan()
            .nearest("vector", as_primitive_array(&q), 10)
            .into_stream();
        let results = stream.try_collect::<Vec<_>>().await.unwrap();
        println!("Tell me results: {:?}", results);
    }
}
