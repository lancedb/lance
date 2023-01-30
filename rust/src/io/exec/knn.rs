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
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use futures::stream::Stream;
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;

use super::{ExecNode, NodeType};
use crate::dataset::Dataset;
use crate::index::vector::flat::flat_search;
use crate::index::vector::ivf::IvfPQIndex;
use crate::index::vector::{Query, VectorIndex};
use crate::io::exec::ExecNodeBox;
use crate::{Error, Result};

/// KNN node for post-filtering.
pub struct KNNFlat {
    rx: Receiver<Result<RecordBatch>>,

    _bg_thread: JoinHandle<()>,
}

impl KNNFlat {
    /// Construct a [KNNFlat] node.
    pub(crate) fn new(child: ExecNodeBox, query: &Query) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);

        let q = query.clone();
        let bg_thread = tokio::spawn(async move {
            let result = match flat_search(child, &q).await {
                Ok(b) => b,
                Err(e) => {
                    tx.send(Err(Error::IO(format!("Failed to compute scores: {e}"))))
                        .await
                        .expect("KNNFlat failed to send message");
                    return;
                }
            };

            if !tx.is_closed() {
                if let Err(e) = tx.send(Ok(result)).await {
                    eprintln!("KNNFlat tx.send error: {e}")
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

/// KNN Node from reading a vector index.
pub struct KNNIndex {
    rx: Receiver<Result<RecordBatch>>,

    _bg_thread: JoinHandle<()>,
}

impl KNNIndex {
    pub fn new(dataset: Arc<Dataset>, index_name: &str, query: &Query) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);

        let q = query.clone();
        let name = index_name.to_string();
        let bg_thread = tokio::spawn(async move {
            let index = match IvfPQIndex::new(&dataset, &name).await {
                Ok(idx) => idx,
                Err(e) => {
                    tx.send(Err(Error::IO(format!(
                        "Failed to open vector index: {name}: {e}"
                    ))))
                    .await
                    .expect("KNNFlat failed to send message");
                    return;
                }
            };
            let result = match index.search(&q).await {
                Ok(b) => b,
                Err(e) => {
                    tx.send(Err(Error::IO(format!("Failed to compute scores: {e}"))))
                        .await
                        .expect("KNNFlat failed to send message");
                    return;
                }
            };

            if !tx.is_closed() {
                if let Err(e) = tx.send(Ok(result)).await {
                    eprintln!("KNNFlat tx.send error: {e}")
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

impl ExecNode for KNNIndex {
    fn node_type(&self) -> NodeType {
        NodeType::Knn
    }
}

impl Stream for KNNIndex {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{
        cast::as_primitive_array, FixedSizeListArray, Int32Array, RecordBatchReader, StringArray,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use tempfile::tempdir;

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

        let batches = RecordBatchBuffer::new(
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        let vector_arr = batches.batches[0].column_by_name("vector").unwrap();
        let q = as_fixed_size_list_array(&vector_arr).value(5);

        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stream = dataset
            .scan()
            .nearest("vector", as_primitive_array(&q), 10)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();
        let results = stream.try_collect::<Vec<_>>().await.unwrap();

        assert!(results[0].schema().column_with_name("score").is_some());

        assert_eq!(results.len(), 1);

        let stream = dataset.scan().try_into_stream().await.unwrap();
        let expected = flat_search(
            stream,
            &Query {
                column: "vector".to_string(),
                key: Arc::new(as_primitive_array(&q).clone()),
                k: 10,
                nprobs: 0,
                refine_factor: None,
            },
        )
        .await
        .unwrap();

        assert_eq!(expected, results[0]);
    }
}
