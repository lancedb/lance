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
use futures::stream::{Stream, TryStreamExt};
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use super::{ExecNode, NodeType};
use crate::{Error, Result};

/// Dataset Scan Node.
pub(crate) struct Limit {
    rx: Receiver<Result<RecordBatch>>,
    _io_thread: JoinHandle<()>,
}

impl Limit {
    /// Create a new execution node to handle limit offset.
    pub fn new(
        child: impl ExecNode + Unpin + Send + 'static,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(4);
        let limit = limit.unwrap_or(0).clone();
        let offset = offset.unwrap_or(0).clone();
        let io_thread = tokio::spawn(async move {
            child
                .try_fold(
                    (offset, limit, tx),
                    |(mut off, mut lim, tx), mut b: RecordBatch| async move {
                        let nrows = b.num_rows() as i64;
                        if off > 0 {
                            if off > nrows {
                                // skip this batch if offset is more than num rows
                                off -= nrows;
                                return Ok((off, lim, tx));
                            } else {
                                // otherwise slice the batch starting from the offset
                                b = b.slice(off as usize, (nrows - off) as usize);
                                off = 0;
                            }
                        }

                        if lim > 0 {
                            if lim > nrows {
                                lim -= nrows;
                            } else {
                                // if this batch is longer than remaining limit
                                // then slice up to the remaining limit
                                b = b.slice(0, lim as usize);
                                lim = 0;
                            }
                        }

                        if tx.is_closed() {
                            eprintln!("ExecNode(Take): channel closed");
                            return Err(Error::IO("ExecNode(Take): channel closed".to_string()));
                        }
                        if let Err(e) = tx.send(Ok(b)).await {
                            eprintln!("ExecNode(Take): {}", e);
                            return Err(Error::IO("ExecNode(Take): channel closed".to_string()));
                        }
                        Ok((off, lim, tx))
                    },
                )
                .await
                .and_then(|(_off, _lim, tx)| {
                    drop(tx);
                    Ok(())
                })
                .unwrap();
        });

        Self {
            rx,
            _io_thread: io_thread,
        }
    }
}

impl ExecNode for Limit {
    fn node_type(&self) -> NodeType {
        NodeType::Limit
    }
}

impl Stream for Limit {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}
