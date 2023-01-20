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

use arrow_array::{Float32Array, RecordBatch};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use futures::stream::Stream;
use futures::StreamExt;
use object_store::path::Path;

use super::Dataset;
use crate::datatypes::Schema;
use crate::format::{Fragment, Manifest};
use crate::index::vector::Query;
use crate::io::exec::{ExecNode, Scan};
use crate::io::ObjectStore;
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

    fragments: Arc<Vec<Fragment>>,

    nearest: Option<Query>,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,
}

impl<'a> Scanner<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            projections: dataset.schema().clone(),
            limit: None,
            offset: None,
            fragments: dataset.fragments().clone(),
            nearest: None,
            with_row_id: false,
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

    /// Find k-nearest neighbour within the vector column.
    pub fn nearest(&mut self, column: &str, q: &Float32Array, k: usize) -> &mut Self {
        self.nearest = Some(Query {
            column: column.to_string(),
            key: Arc::new(q.clone()),
            k,
            nprobs: 1,
        });
        self
    }

    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        self.nearest.as_mut().map(|q| q.nprobs = n);
        self
    }

    /// Instruct the scanner to return the `_rowid` meta column from the dataset.
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
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

        let data_dir = self.dataset.data_dir().clone();
        let manifest = self.dataset.manifest.clone();
        let with_row_id = self.with_row_id;
        let projection = &self.projections;

        ScannerStream::new(
            self.dataset.object_store.clone(),
            data_dir,
            self.fragments.clone(),
            manifest,
            PREFECTH_SIZE,
            projection,
            with_row_id,
        )
    }
}

#[pin_project::pin_project]
pub struct ScannerStream {
    #[pin]
    exec_node: Box<dyn ExecNode + Unpin + Send>,
}

impl ScannerStream {
    fn new<'a>(
        object_store: Arc<ObjectStore>,
        data_dir: Path,
        fragments: Arc<Vec<Fragment>>,
        manifest: Arc<Manifest>,
        prefetch_size: usize,
        schema: &Schema,
        with_row_id: bool,
    ) -> Self {
        let exec_node = Box::new(Scan::new(
            object_store,
            data_dir,
            fragments.clone(),
            schema,
            manifest.clone(),
            prefetch_size,
            with_row_id,
        ));

        Self { exec_node }
    }
}

impl Stream for ScannerStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        this.exec_node.poll_next_unpin(cx)
    }
}
