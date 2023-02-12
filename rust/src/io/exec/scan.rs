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

use std::any::Any;
use std::cmp::min;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::error::DataFusionError;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::{
    ExecutionPlan, Partitioning, RecordBatchStream, SendableRecordBatchStream,
};
use futures::stream::Stream;
use object_store::path::Path;
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use crate::format::Manifest;
use crate::io::{FileReader, ObjectStore};
use crate::{datatypes::Schema, format::Fragment};

/// Dataset Scan Node.
#[derive(Debug)]
pub struct LanceStream {
    rx: Receiver<std::result::Result<RecordBatch, DataFusionError>>,

    _io_thread: JoinHandle<()>,

    /// Manifest of the dataset
    projection: Arc<Schema>,
}

impl LanceStream {
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
        projection: Arc<Schema>,
        manifest: Arc<Manifest>,
        read_size: usize,
        prefetch_size: usize,
        with_row_id: bool,
    ) -> Self {
        let (tx, rx) = mpsc::channel(prefetch_size);

        let project_schema = projection.clone();
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
                        r.set_projection(project_schema.as_ref().clone());
                        r.with_row_id(with_row_id);
                        r
                    }
                    Err(e) => {
                        tx.send(Err(DataFusionError::Execution(format!(
                            "Failed to open file: {path}: {e}"
                        ))))
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
                        if let Err(err) = tx.send(result.map_err(|e| e.into())).await {
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
            projection,
        }
    }
}

impl RecordBatchStream for LanceStream {
    fn schema(&self) -> SchemaRef {
        Arc::new(self.projection.as_ref().into())
    }
}

impl Stream for LanceStream {
    type Item = std::result::Result<RecordBatch, datafusion::error::DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

/// DataFusion [ExecutionPlan] for scanning one Lance dataset
pub struct LanceScanExec {
    object_store: Arc<ObjectStore>,
    data_dir: Path,
    fragments: Arc<Vec<Fragment>>,
    projection: Arc<Schema>,
    manifest: Arc<Manifest>,
    read_size: usize,
    prefetch_size: usize,
    with_row_id: bool,
}

impl std::fmt::Debug for LanceScanExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let columns = self
            .projection
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>();
        write!(
            f,
            "LanceScan(uri={}, projection={:#?}, row_id={})",
            self.data_dir, columns, self.with_row_id
        )
    }
}

impl LanceScanExec {
    pub fn new(
        object_store: Arc<ObjectStore>,
        data_dir: Path,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        manifest: Arc<Manifest>,
        read_size: usize,
        prefetch_size: usize,
        with_row_id: bool,
    ) -> Self {
        Self {
            object_store: object_store.clone(),
            data_dir,
            fragments: fragments.clone(),
            projection: projection.clone(),
            manifest: manifest.clone(),
            read_size,
            prefetch_size,
            with_row_id,
        }
    }
}

impl ExecutionPlan for LanceScanExec {
    fn as_any(&self) -> &dyn Any {
        let this = self;
        this
    }

    fn schema(&self) -> SchemaRef {
        Arc::new((&self.manifest.schema).into())
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    /// Scan is the leaf node, so returns an empty vector.
    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        Ok(Box::pin(LanceStream::new(
            self.object_store.clone(),
            self.data_dir.clone(),
            self.fragments.clone(),
            self.projection.clone(),
            self.manifest.clone(),
            self.read_size,
            self.prefetch_size,
            self.with_row_id,
        )))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        todo!()
    }
}
