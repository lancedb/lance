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
use crate::io::exec::{ExecNode, ExecNodeBox, KNNFlat, KNNIndex, Limit, Scan, Take};
use crate::{Error, Result};

/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";
pub const DEFAULT_BATCH_SIZE: usize = 8192;

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

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: usize,

    // filter: how to present filter
    limit: Option<i64>,
    offset: Option<i64>,

    fragments: Arc<Vec<Fragment>>,

    nearest: Option<Query>,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection = dataset.schema().clone();
        let fragments = dataset.fragments().clone();
        Self {
            dataset,
            projections: projection,
            batch_size: DEFAULT_BATCH_SIZE,
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

    /// Set the batch size.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = batch_size;
        self
    }

    /// Set limit and offset.
    pub fn limit(&mut self, limit: i64, offset: Option<i64>) -> Result<&mut Self> {
        if limit < 0 {
            return Err(Error::IO("Limit must be non-negative".to_string()));
        }
        if let Some(off) = offset {
            if off < 0 {
                return Err(Error::IO("Offset must be non-negative".to_string()));
            }
        }
        self.limit = Some(limit);
        self.offset = offset;
        Ok(self)
    }

    /// Find k-nearest neighbour within the vector column.
    pub fn nearest(&mut self, column: &str, q: &Float32Array, k: usize) -> Result<&mut Self> {
        if k == 0 {
            return Err(Error::IO("k must be positive".to_string()));
        }
        if q.is_empty() {
            return Err(Error::IO(
                "Query vector must have non-zero length".to_string(),
            ));
        }
        // make sure the field exists
        self.dataset.schema().project(&[column])?;
        self.nearest = Some(Query {
            column: column.to_string(),
            key: Arc::new(q.clone()),
            k,
            nprobs: 1,
            refine_factor: None,
        });
        Ok(self)
    }

    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.nprobs = n;
        }
        self
    }

    /// Apply a refine step to the vector search.
    ///
    /// A refine step uses the original vector values to re-rank the distances.
    pub fn refine(&mut self, factor: u32) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.refine_factor = Some(factor)
        };
        self
    }

    /// Instruct the scanner to return the `_rowid` meta column from the dataset.
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self
    }

    /// The schema of the output, a.k.a, projection schema.
    pub fn schema(&self) -> Result<SchemaRef> {
        if self.nearest.as_ref().is_some() {
            let q = self.nearest.as_ref().unwrap();
            let column: ArrowField = self
                .dataset
                .schema()
                .field(q.column.as_str())
                .ok_or_else(|| {
                    Error::Schema(format!("Vector column {} not found in schema", q.column))
                })?
                .into();
            let score = ArrowField::new("score", Float32, false);
            let score_schema = ArrowSchema::new(vec![column, score]);
            let to_merge = &Schema::try_from(&score_schema).unwrap();
            let merged = self.projections.merge(to_merge);
            Ok(SchemaRef::new(ArrowSchema::from(&merged)))
        } else {
            Ok(Arc::new(ArrowSchema::from(&self.projections)))
        }
    }

    fn should_use_index(&self) -> bool {
        self.nearest.is_some()
    }

    /// Create a stream of this Scanner.
    ///
    /// TODO: implement as IntoStream/IntoIterator.
    pub async fn try_into_stream(&self) -> Result<ScannerStream> {
        const PREFECTH_SIZE: usize = 8;

        let data_dir = self.dataset.data_dir();
        let manifest = self.dataset.manifest.clone();
        let with_row_id = self.with_row_id;
        let projection = &self.projections;

        let indices = if self.should_use_index() {
            self.dataset.load_indices().await?
        } else {
            vec![]
        };

        let mut exec_node: ExecNodeBox = if let Some(q) = self.nearest.as_ref() {
            let column_id = self
                .dataset
                .schema()
                .field(&q.column)
                .expect("vector column does not exist")
                .id;

            if let Some(rf) = q.refine_factor {
                if rf == 0 {
                    return Err(Error::IO("Refine factor can not be zero".to_string()));
                }
            }
            let knn_node: Box<dyn ExecNode + Send + Unpin> =
                if let Some(index) = indices.iter().find(|i| i.fields.contains(&column_id)) {
                    // There is an index built for the column.
                    // We will use the index.
                    let mut inner_query = q.clone();
                    inner_query.k = q.k * (q.refine_factor.unwrap_or(1) as usize);
                    Box::new(KNNIndex::new(
                        self.dataset.clone(),
                        &index.uuid.to_string(),
                        &inner_query,
                    ))
                } else {
                    let vector_scan_projection =
                        Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
                    let scan_node = Box::new(Scan::new(
                        self.dataset.object_store.clone(),
                        data_dir.clone(),
                        self.fragments.clone(),
                        &vector_scan_projection,
                        manifest.clone(),
                        self.batch_size,
                        PREFECTH_SIZE,
                        true,
                    ));
                    Box::new(KNNFlat::new(scan_node, &q.clone()))
                };

            let take_node = Box::new(Take::new(
                self.dataset.clone(),
                Arc::new(projection.clone()),
                knn_node,
            ));

            if q.refine_factor.is_some() {
                Box::new(KNNFlat::new(take_node, &q.clone()))
            } else {
                take_node
            }
        } else {
            Box::new(Scan::new(
                self.dataset.object_store.clone(),
                data_dir.clone(),
                self.fragments.clone(),
                projection,
                manifest.clone(),
                self.batch_size,
                PREFECTH_SIZE,
                with_row_id,
            ))
        };

        if (self.limit.unwrap_or(0) > 0) || self.offset.is_some() {
            exec_node = Box::new(Limit::new(exec_node, self.limit, self.offset))
        }
        Ok(ScannerStream::new(exec_node))
    }
}

/// ScannerStream is a container to wrap different types of ExecNode.
#[pin_project::pin_project]
pub struct ScannerStream {
    #[pin]
    exec_node: ExecNodeBox,
}

impl ScannerStream {
    fn new(exec_node: ExecNodeBox) -> Self {
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

#[cfg(test)]
mod test {

    use super::*;

    use crate::{arrow::RecordBatchBuffer, dataset::WriteParams};

    use arrow_array::{Int32Array, RecordBatchReader, StringArray};
    use arrow_schema::DataType;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_batch_size() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches = RecordBatchBuffer::new(
            (0..5)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                            Arc::new(StringArray::from_iter_values(
                                (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect(),
        );

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut stream = dataset
            .scan()
            .batch_size(8)
            .try_into_stream()
            .await
            .unwrap();
        for expected_len in [8, 8, 4, 8, 8, 4] {
            assert_eq!(
                stream.next().await.unwrap().unwrap().num_rows(),
                expected_len as usize
            );
        }
    }
}
