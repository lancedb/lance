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
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use object_store::path::Path;
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use super::{ExecNode, NodeType};
use crate::format::Manifest;
use crate::io::{FileReader, ObjectStore};
use crate::{datatypes::Schema, format::Fragment};
use crate::{Error, Result};

/// Dataset Scan Node.
pub(crate) struct Scan {
    rx: Receiver<Result<RecordBatch>>,

    _io_thread: JoinHandle<()>,
}

impl Scan {
    /// Create a new scan node.
    pub fn new(
        object_store: Arc<ObjectStore>,
        data_dir: Path,
        fragments: Arc<Vec<Fragment>>,
        projection: &Schema,
        manifest: Arc<Manifest>,
        prefetch_size: usize,
        with_row_id: bool,
    ) -> Self {
        let (tx, rx) = mpsc::channel(prefetch_size);

        let projection = projection.clone();
        let io_thread = tokio::spawn(async move {
            for frag in fragments.as_ref() {
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
                        tx.send(Err(Error::IO(format!(
                            "Failed to open file: {}: {}",
                            path, e
                        ))))
                        .await
                        .expect("Scanner sending error message");
                        // Stop reading.
                        break;
                    }
                };

                let r = &reader;
                match stream::iter(0..reader.num_batches())
                    .map(|batch_id| async move { r.read_batch(batch_id as i32, ..).await })
                    .buffer_unordered(prefetch_size)
                    .try_for_each(|b| async { tx.send(Ok(b)).await.map_err(|_| Error::Stop()) })
                    .await
                {
                    Ok(_) | Err(Error::Stop()) => {}
                    Err(e) => {
                        eprintln!("Failed to scan data: {e}");
                        break;
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
