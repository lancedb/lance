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

use arrow_array::RecordBatch;
use futures::stream::Stream;

use super::{ExecNode, NodeType};
use crate::index::vector::Query;
use crate::Result;

/// KNN node for post-filtering.
pub struct KNNFlat {
    child: Box<dyn ExecNode + Unpin + Send>,

    query: Query,
}

impl KNNFlat {
    pub(crate) fn new(child: Box<dyn ExecNode + Unpin + Send>, query: &Query) -> Self {
        // assert_eq!(child.node_type(), NodeType::Scan, "")
        Self {
            child,
            query: query.clone(),
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
        //Pin::into_inner(self).rx.poll_recv(cx)
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use object_store::path::Path;
    use tempfile::tempdir;

    use crate::arrow::*;
    use crate::dataset::{Dataset, WriteParams};
    use crate::utils::testing::generate_random_array;
    use crate::{arrow::RecordBatchBuffer, io::ObjectStore};

    #[tokio::test]
    async fn knn_flat_search() {
        let store = ObjectStore::memory();
        let path = Path::from("/flat");

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
        println!("Batches: {:?}", batches);

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        Dataset::create(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();
    }
}
