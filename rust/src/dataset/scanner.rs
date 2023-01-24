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
use arrow_schema::DataType::Float32;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use futures::stream::{Stream, StreamExt};

use super::Dataset;
use crate::datatypes::Schema;
use crate::format::Fragment;
use crate::index::vector::Query;
use crate::Error::IO;

use crate::io::exec::{ExecNodeBox, KNNFlat, Limit, Scan, Take};
use crate::Result;

/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";

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
pub struct Scanner {
    dataset: Arc<Dataset>,

    projections: Schema,

    // filter: how to present filter
    limit: Option<i64>,
    offset: Option<i64>,

    fragments: Arc<Vec<Fragment>>,

    nearest: Option<Query>,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,
}

impl<'a> Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection = dataset.schema().clone();
        let fragments = dataset.fragments().clone();
        Self {
            dataset,
            projections: projection,
            limit: None,
            offset: None,
            fragments,
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
    pub fn limit(&mut self, limit: i64, offset: Option<i64>) -> Result<&mut Self> {
        if limit < 0 {
            return Err(IO("Limit must be non-negative".to_string()));
        }
        if let Some(off) = offset {
            if off < 0 {
                return Err(IO("Offset must be non-negative".to_string()));
            }
        }
        self.limit = Some(limit);
        self.offset = offset;
        Ok(self)
    }

    /// Find k-nearest neighbour within the vector column.
    pub fn nearest(&mut self, column: &str, q: &Float32Array, k: usize) -> Result<&mut Self> {
        if k <= 0 {
            return Err(IO("k must be positive".to_string()));
        }
        if q.is_empty() {
            return Err(IO("Query vector must have non-zero length".to_string()));
        }
        // make sure the field exists
        self.dataset.schema().project(&[column])?;
        self.nearest = Some(Query {
            column: column.to_string(),
            key: Arc::new(q.clone()),
            k,
            nprobs: 1,
        });
        Ok(self)
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
        if self.nearest.as_ref().is_some() {
            let score = ArrowField::new("score", Float32, false);
            let score_schema = ArrowSchema::new(vec![score]);
            let to_merge = &Schema::try_from(&score_schema).unwrap();
            let merged = self.projections.merge(to_merge);
            SchemaRef::new(ArrowSchema::from(&merged))
        } else {
            Arc::new(ArrowSchema::from(&self.projections))
        }
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

        let mut exec_node: ExecNodeBox = if let Some(q) = self.nearest.as_ref() {
            let vector_scan_projection =
                Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
            let scan_node = Box::new(Scan::new(
                self.dataset.object_store.clone(),
                data_dir.clone(),
                self.fragments.clone(),
                &vector_scan_projection,
                manifest.clone(),
                PREFECTH_SIZE,
                true,
            ));
            let flat_knn_node = Box::new(KNNFlat::new(scan_node, q));
            Box::new(Take::new(
                self.dataset.clone(),
                Arc::new(projection.clone()),
                flat_knn_node,
            ))
        } else {
            Box::new(Scan::new(
                self.dataset.object_store.clone(),
                data_dir.clone(),
                self.fragments.clone(),
                projection,
                manifest.clone(),
                PREFECTH_SIZE,
                with_row_id,
            ))
        };

        if (self.limit.unwrap_or(0) > 0) || self.offset.is_some() {
            exec_node = Box::new(Limit::new(exec_node, self.limit, self.offset))
        }
        ScannerStream::new(exec_node)
    }
}

/// ScannerStream is a container to wrap different types of ExecNode.
#[pin_project::pin_project]
pub struct ScannerStream {
    #[pin]
    exec_node: ExecNodeBox,
}

impl ScannerStream {
    fn new<'a>(exec_node: ExecNodeBox) -> Self {
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
