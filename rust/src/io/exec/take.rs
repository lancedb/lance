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

use arrow_array::cast::as_primitive_array;
use arrow_array::{RecordBatch, UInt64Array};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use super::{ExecNode, NodeType};
use crate::arrow::RecordBatchExt;
use crate::dataset::Dataset;
use crate::datatypes::Schema;
use crate::io::exec::ExecNodeBox;
use crate::{Error, Result};

/// Dataset Take Node.
///
/// [Take] node takes the filtered batch from the child node.
/// It uses the `_rowid` to random access on [Dataset] to gather the final results.
pub(crate) struct Take {
    rx: Receiver<Result<RecordBatch>>,
    _bg_thread: JoinHandle<()>,
}

impl Take {
    /// Create a Take node with
    ///
    ///  - Dataset: the dataset to read from
    ///  - schema: projection schema for take node.
    ///  - child: the upstream ExedNode to feed data in.
    pub fn new(dataset: Arc<Dataset>, schema: Arc<Schema>, child: ExecNodeBox) -> Self {
        let (tx, rx) = mpsc::channel(4);

        let bg_thread = tokio::spawn(async move {
            if let Err(e) = child
                .zip(stream::repeat_with(|| (dataset.clone(), schema.clone())))
                .then(|(batch, (dataset, schema))| async move {
                    let batch = batch?;
                    let row_id_arr = batch.column_by_name("_rowid").unwrap();
                    let row_ids: &UInt64Array = as_primitive_array(row_id_arr);
                    let rows = dataset.take_rows(row_ids.values(), &schema).await?;
                    rows.merge(&batch)?.drop_column("_rowid")
                })
                .try_for_each(|b| async {
                    if tx.is_closed() {
                        eprintln!("ExecNode(Take): channel closed");
                        return Err(Error::IO("ExecNode(Take): channel closed".to_string()));
                    }
                    if let Err(e) = tx.send(Ok(b)).await {
                        eprintln!("ExecNode(Take): {}", e);
                        return Err(Error::IO("ExecNode(Take): channel closed".to_string()));
                    }
                    Ok(())
                })
                .await
            {
                if let Err(e) = tx.send(Err(e)).await {
                    eprintln!("ExecNode(Take): {}", e);
                }
            }
            drop(tx)
        });

        Self {
            rx,
            _bg_thread: bg_thread,
        }
    }
}

impl ExecNode for Take {
    fn node_type(&self) -> NodeType {
        NodeType::Take
    }
}

impl Stream for Take {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}
