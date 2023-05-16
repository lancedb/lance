// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::error::Result;
use datafusion::physical_plan::{
    ExecutionPlan, Partitioning, RecordBatchStream, SendableRecordBatchStream,
};
use futures::stream;
use futures::stream::Stream;
use futures::{StreamExt, TryStreamExt};

use crate::dataset::fragment::{FileFragment, FragmentReader};
use crate::dataset::{Dataset, ROW_ID};
use crate::datatypes::Schema;
use crate::format::Fragment;

async fn open_file(
    file_fragment: FileFragment,
    projection: Arc<Schema>,
    with_row_id: bool,
) -> Result<FragmentReader> {
    let mut reader = file_fragment.open(projection.as_ref()).await?;
    if with_row_id {
        reader.with_row_id();
    };
    Ok(reader)
}

/// Dataset Scan Node.
// #[derive(Debug)]
pub struct LanceStream {
    inner_stream: stream::BoxStream<'static, Result<RecordBatch>>,

    /// Manifest of the dataset
    projection: Arc<Schema>,

    with_row_id: bool,
}

impl LanceStream {
    /// Create a new dataset scan node.
    ///
    /// Parameters
    ///
    ///  - ***dataset***: The source dataset.
    ///  - ***projection***: the projection [Schema].
    ///  - ***filter***: filter [`PhysicalExpr`], optional.
    ///  - ***read_size***: the number of rows to read for each request.
    ///  - ***batch_readahead***: the number of batches to read ahead.
    ///  - ***with_row_id***: load row ID from the datasets.
    ///  - ***ordered***: whether to scan the fragments in the provided order.
    ///read_size: usize,futures::iter::
    pub fn try_new(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        read_size: usize,
        batch_readahead: usize,
        with_row_id: bool,
        ordered: bool,
    ) -> Result<Self> {
        let project_schema = projection.clone();

        let file_fragments = fragments
            .iter()
            .map(|fragment| FileFragment::new(dataset.clone(), fragment.clone()))
            .collect::<Vec<_>>();

        let inner_stream = if ordered {
            stream::iter(file_fragments)
                .then(move |file_fragment| open_file(file_fragment, project_schema, with_row_id))
                .map_ok(|reader| reader.scan_batches(read_size))
                .try_flatten()
                .boxed()
        } else {
            stream::iter(file_fragments)
                .map(move |file_fragment| open_file(file_fragment, project_schema, with_row_id))
                .buffer_unordered(batch_readahead) // Multiple fragments concurrently
                .map_ok(|reader| reader.scan_batches(read_size))
                .try_flatten_unordered(batch_readahead)
                .boxed()
        };

        Ok(Self {
            inner_stream,
            projection,
            with_row_id,
        })
    }
}

impl RecordBatchStream for LanceStream {
    fn schema(&self) -> SchemaRef {
        let schema: ArrowSchema = self.projection.as_ref().into();
        if self.with_row_id {
            let mut fields: Vec<Arc<Field>> = schema.fields.to_vec();
            fields.push(Arc::new(Field::new(ROW_ID, DataType::UInt64, false)));
            Arc::new(ArrowSchema::new(fields))
        } else {
            Arc::new(schema)
        }
    }
}

impl Stream for LanceStream {
    type Item = std::result::Result<RecordBatch, datafusion::error::DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).inner_stream.poll_next_unpin(cx)
    }
}

/// DataFusion [ExecutionPlan] for scanning one Lance dataset
pub struct LanceScanExec {
    dataset: Arc<Dataset>,
    fragments: Arc<Vec<Fragment>>,
    projection: Arc<Schema>,
    read_size: usize,
    batch_readahead: usize,
    with_row_id: bool,
    ordered_output: bool,
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
            "LanceScan(uri={}, projection={:#?}, row_id={}, ordered={})",
            self.dataset.data_dir(),
            columns,
            self.with_row_id,
            self.ordered_output
        )
    }
}

impl LanceScanExec {
    pub fn new(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        read_size: usize,
        batch_readahead: usize,
        with_row_id: bool,
        ordered_ouput: bool,
    ) -> Self {
        Self {
            dataset,
            fragments,
            projection,
            read_size,
            batch_readahead,
            with_row_id,
            ordered_output: ordered_ouput,
        }
    }
}

impl ExecutionPlan for LanceScanExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        let schema: ArrowSchema = self.projection.as_ref().into();
        if self.with_row_id {
            let mut fields: Vec<Arc<Field>> = schema.fields.to_vec();
            fields.push(Arc::new(Field::new(ROW_ID, DataType::UInt64, false)));
            Arc::new(ArrowSchema::new(fields))
        } else {
            Arc::new(schema)
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
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
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(LanceStream::try_new(
            self.dataset.clone(),
            self.fragments.clone(),
            self.projection.clone(),
            self.read_size,
            self.batch_readahead,
            self.with_row_id,
            self.ordered_output,
        )?))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        todo!()
    }
}
