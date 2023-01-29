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
use crate::io::exec::ExecNodeBox;
use crate::{Error, Result};

/// Dataset Scan Node.
pub(crate) struct Limit {
    rx: Receiver<Result<RecordBatch>>,
    _io_thread: JoinHandle<()>,
}

impl Limit {
    /// Create a new execution node to handle limit offset.
    pub fn new(child: ExecNodeBox, limit: Option<i64>, offset: Option<i64>) -> Self {
        let (tx, rx) = mpsc::channel(4);
        let limit = limit.unwrap_or(0).clone();
        let offset = offset.unwrap_or(0).clone();
        let io_thread = tokio::spawn(async move {
            child
                .try_fold(
                    (offset, limit, tx),
                    |(mut off, mut lim, tx), mut b: RecordBatch| async move {
                        let mut nrows = b.num_rows() as i64;
                        if off > 0 {
                            if off > nrows {
                                // skip this batch if offset is more than num rows
                                off -= nrows;
                                return Ok((off, lim, tx));
                            } else {
                                // otherwise slice the batch starting from the offset
                                b = b.slice(off as usize, (nrows - off) as usize);
                                nrows = b.num_rows() as i64;
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
                            eprintln!("ExecNode(Limit): channel closed");
                            return Err(Error::IO("ExecNode(Limit): channel closed".to_string()));
                        }
                        if let Err(e) = tx.send(Ok(b)).await {
                            eprintln!("ExecNode(Limit): {}", e);
                            return Err(Error::IO("ExecNode(Limit): channel closed".to_string()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::arrow::RecordBatchBuffer;
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::{ArrayRef, Int64Array, RecordBatchReader};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
    use arrow_select::concat::concat_batches;
    use futures::StreamExt;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_limit() {
        // TODO setting up a unit test for an ExecNode isn't simple.
        //      consider changing the interface to a stream of RecordBatch
        //      so for unit testing it's easy to setup the child node
        let temp = TempDir::new().unwrap();
        let mut file_path = PathBuf::from(temp.as_ref());
        file_path.push("limit_test.lance");
        let path = file_path.to_str().unwrap();
        let expected_batches = write_data(path).await;
        let expected_combined =
            concat_batches(&expected_batches[0].schema(), &expected_batches).unwrap();

        let dataset = Dataset::open(path).await.unwrap();
        let mut scanner = dataset.scan();
        scanner.limit(2, Some(19)).unwrap();
        let actual_batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .map(|b| b.unwrap())
            .collect::<Vec<RecordBatch>>()
            .await;
        let actual_combined = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        assert_eq!(expected_combined.slice(19, 2), actual_combined);
        // skipped 1 batch
        assert_eq!(actual_batches.len(), 2);
    }

    async fn write_data(path: &str) -> Vec<RecordBatch> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int64,
            true,
        )])) as SchemaRef;

        // Write 3 batches.
        let expected_batches: Vec<RecordBatch> = (0..3)
            .map(|batch_id| {
                let value_range: Range<i64> = batch_id * 10..batch_id * 10 + 10;
                let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter(
                    value_range.clone().collect::<Vec<_>>(),
                ))];
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect();
        let batches = RecordBatchBuffer::new(expected_batches.clone());
        let mut params = WriteParams::default();
        params.max_rows_per_group = 10;
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut reader, path, Some(params))
            .await
            .unwrap();
        expected_batches
    }
}
