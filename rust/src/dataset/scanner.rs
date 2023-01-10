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

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use futures::stream::Stream;
use object_store::path::Path;
use tokio::sync::mpsc::{self, Receiver};

use super::Dataset;
use crate::datatypes::Schema;
use crate::format::{Fragment, Manifest};
use crate::io::{FileReader, ObjectStore};
use crate::Result;

/// Dataset Scanner
///
/// ```rust,ignore
/// let dataset = Dataset::open(uri).await.unwrap();
/// let stream = dataset.scan()
///     .project(&["col", "col2.subfield"]).unwrap()
///     .limit(10)
///     .into_stream();
/// stream
///   .map(|batch| batch.num_rows())
///   .buffered(16)
///   .sum()
/// ```
pub struct Scanner<'a> {
    dataset: &'a Dataset,

    projections: Schema,

    // filter: how to present filter
    limit: Option<i64>,
    offset: Option<i64>,

    fragments: Vec<Fragment>,
}

impl<'a> Scanner<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            projections: dataset.schema().clone(),
            limit: None,
            offset: None,
            fragments: dataset.fragments().to_vec(),
        }
    }

    /// Projection.
    ///
    /// Only seelect the specific columns. If not specifid, all columns will be scanned.
    pub fn project(&mut self, columns: &[&str]) -> Result<&mut Self> {
        self.projections = self.dataset.schema().project(columns)?;
        Ok(self)
    }

    /// Set limit and offset.
    pub fn limit(&mut self, limit: i64, offset: Option<i64>) -> &mut Self {
        self.limit = Some(limit);
        self.offset = offset;
        self
    }

    /// The schema of the output, a.k.a, projection schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::new(ArrowSchema::from(&self.projections))
    }

    /// Create a stream of this Scanner.
    ///
    /// TODO: implement as IntoStream/IntoIterator.
    pub fn into_stream(&self) -> ScannerStream {
        const PREFECTH_SIZE: usize = 8;
        let object_store = self.dataset.object_store.clone();

        let data_dir = self.dataset.data_dir().clone();
        let fragments = self.fragments.clone();
        let manifest = self.dataset.manifest.clone();

        ScannerStream::new(
            object_store,
            data_dir,
            fragments,
            manifest,
            PREFECTH_SIZE,
            &self.projections,
        )
    }
}

pub struct ScannerStream {
    rx: Receiver<Result<RecordBatch>>,
}

impl ScannerStream {
    fn new(
        object_store: Arc<ObjectStore>,
        data_dir: Path,
        fragments: Vec<Fragment>,
        manifest: Arc<Manifest>,
        prefetch_size: usize,
        schema: &Schema,
    ) -> Self {
        let (tx, rx) = mpsc::channel(prefetch_size);

        let schema = schema.clone();
        tokio::spawn(async move {
            for frag in &fragments {
                let data_file = &frag.files[0];
                let path = data_dir.child(data_file.path.clone());
                let reader =
                    match FileReader::new(&object_store, &path, Some(manifest.as_ref())).await {
                        Ok(mut r) => {
                            r.set_projection(schema.clone());
                            r
                        }
                        Err(e) => {
                            tx.send(Err(e)).await.unwrap();
                            continue;
                        }
                    };
                for batch_id in 0..reader.num_batches() {
                    tx.send(reader.read_batch(batch_id as i32).await)
                        .await
                        .unwrap();
                }
            }
            drop(tx)
        });
        Self { rx }
    }
}

impl Stream for ScannerStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::into_inner(self).rx.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::stream::StreamExt;
}
