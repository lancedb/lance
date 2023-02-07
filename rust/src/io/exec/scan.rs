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

use std::cmp::min;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use futures::stream::Stream;
use object_store::path::Path;
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use super::{ExecNode, NodeType};
use crate::format::Manifest;
use crate::io::{FileReader, ObjectStore};
use crate::{datatypes::Schema, format::Fragment};
use crate::{Error, Result};

/// Dataset Scan Node.
pub struct Scan {
    rx: Receiver<Result<RecordBatch>>,

    _io_thread: JoinHandle<()>,
}

impl Scan {
    /// Create a new dataset scan node.
    ///
    /// Parameters
    ///
    ///  - ***object_store***: The Object Store to operate the data.
    ///  - ***data_dir***: The base directory of the dataset.
    ///  - ***fragments***: list of [Fragment]s to open.
    ///  - ***projection***: the projection [Schema].
    ///  - ***manifest***: the [Manifest] of the dataset.
    ///  - ***read_size***: the number of rows to read for each request.
    ///  - ***prefetch_size***: the number of batches to read ahead.
    ///  - ***with_row_id***: load row ID from the datasets.
    ///
    pub fn new(
        object_store: Arc<ObjectStore>,
        data_dir: Path,
        fragments: Arc<Vec<Fragment>>,
        projection: &Schema,
        manifest: Arc<Manifest>,
        read_size: usize,
        prefetch_size: usize,
        with_row_id: bool,
    ) -> Self {
        let (tx, rx) = mpsc::channel(prefetch_size);

        let projection = projection.clone();
        let io_thread = tokio::spawn(async move {
            'outer: for frag in fragments.as_ref() {
                if tx.is_closed() {
                    return;
                }
                let data_file = &frag.files[0];
                let path = data_dir.child(data_file.path.clone());
                let reader = match FileReader::try_new_with_fragment(
                    &object_store,
                    &path,
                    frag.id,
                    Some(manifest.as_ref()),
                )
                .await
                {
                    Ok(mut r) => {
                        r.set_projection(projection.clone());
                        r.with_row_id(with_row_id);
                        r
                    }
                    Err(e) => {
                        tx.send(Err(Error::IO(format!("Failed to open file: {path}: {e}"))))
                            .await
                            .expect("Scanner sending error message");
                        // Stop reading.
                        break;
                    }
                };

                let r = &reader;
                for batch_id in 0..reader.num_batches() as i32 {
                    let rows_in_batch = reader.num_rows_in_batch(batch_id);
                    for start in (0..rows_in_batch).step_by(read_size) {
                        let result = r
                            .read_batch(batch_id, start..min(start + read_size, rows_in_batch))
                            .await;
                        if let Err(err) = tx.send(result).await {
                            eprintln!("Failed to scan data: {err}");
                            break 'outer;
                        }
                    }
                }
            }

            drop(tx)
        });

        Self {
            rx,
            _io_thread: io_thread, // Drop the background I/O thread with the stream.
        }
    }
}

impl ExecNode for Scan {
    fn node_type(&self) -> NodeType {
        NodeType::Scan
    }
}

impl Stream for Scan {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}
