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

use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, RecordBatchStream,
};
use futures::{stream::BoxStream, Stream, StreamExt, TryFutureExt};
use lance_core::{
    format::{Fragment, RowAddress},
    Error, Result, ROW_ID_FIELD,
};
use lance_index::scalar::{
    expression::{ScalarIndexExpr, ScalarIndexLoader},
    ScalarIndex,
};
use pin_project::pin_project;
use snafu::{location, Location};
use tracing::instrument;

use crate::{index::DatasetIndexInternalExt, Dataset};

lazy_static::lazy_static! {
    pub static ref SCALAR_INDEX_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![Field::new("result".to_string(), DataType::Binary, true)]));
}

#[async_trait]
impl ScalarIndexLoader for Dataset {
    async fn load_index(&self, name: &str) -> Result<Arc<dyn ScalarIndex>> {
        let idx = self
            .load_scalar_index_for_column(name)
            .await?
            .ok_or_else(|| Error::Internal {
                message: format!("Scanner created plan for index query on {} but no index on dataset for that column", name),
                location: location!()
            })?;
        self.open_scalar_index(name, &idx.uuid.to_string()).await
    }
}

/// An execution node that performs a scalar index search
///
/// This does not actually scan any data.  We only look through the index to determine
/// the row ids that match the query.  The output of this node is a row id mask (serialized
/// into a record batch)
///
/// If the actual IDs are needed then use MaterializeIndexExec instead
#[derive(Debug)]
pub struct ScalarIndexExec {
    dataset: Arc<Dataset>,
    expr: ScalarIndexExpr,
}

impl DisplayAs for ScalarIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ScalarIndexQuery: query={}", self.expr)
            }
        }
    }
}

impl ScalarIndexExec {
    pub fn new(dataset: Arc<Dataset>, expr: ScalarIndexExpr) -> Self {
        Self { dataset, expr }
    }

    async fn do_execute(expr: ScalarIndexExpr, dataset: Arc<Dataset>) -> Result<RecordBatch> {
        let query_result = expr.evaluate(dataset.as_ref()).await?;
        let query_result_arr = query_result.into_arrow()?;
        Ok(RecordBatch::try_new(
            SCALAR_INDEX_SCHEMA.clone(),
            vec![Arc::new(query_result_arr)],
        )?)
    }
}

#[pin_project]
struct StreamWithSchema {
    #[pin]
    stream: BoxStream<'static, datafusion::common::Result<RecordBatch>>,
    schema: Arc<Schema>,
}

impl Stream for StreamWithSchema {
    type Item = datafusion::common::Result<RecordBatch>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.project();
        this.stream.poll_next(cx)
    }
}

impl RecordBatchStream for StreamWithSchema {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl ExecutionPlan for ScalarIndexExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        SCALAR_INDEX_SCHEMA.clone()
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        // No guarantee a scalar index scan will return row ids in any meaningful order
        None
    }

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
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let batch_fut = Self::do_execute(self.expr.clone(), self.dataset.clone());
        let stream = futures::stream::iter(vec![batch_fut])
            .then(|batch_fut| batch_fut.map_err(|err| err.into()))
            .boxed()
            as BoxStream<'static, datafusion::common::Result<RecordBatch>>;
        Ok(Box::pin(StreamWithSchema {
            schema: SCALAR_INDEX_SCHEMA.clone(),
            stream,
        }))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        todo!()
    }
}

lazy_static::lazy_static! {
    pub static ref MATERIALIZE_INDEX_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()]));
}

/// An execution node that performs a scalar index search and materializes the mask into row ids
///
/// First, the index is searched to determine the mask that should be applied.  Then, we take the
/// list of fragments, iterate through all possible row ids, and materialize the row ids that satisfy
/// the mask.  The output of this node is a list of row ids suitable for use in a take operation.
#[derive(Debug)]
pub struct MaterializeIndexExec {
    dataset: Arc<Dataset>,
    expr: ScalarIndexExpr,
    fragments: Arc<Vec<Fragment>>,
}

impl DisplayAs for MaterializeIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MaterializeIndex: query={}", self.expr)
            }
        }
    }
}

struct FragIdIter {
    src: Arc<Vec<Fragment>>,
    frag_idx: usize,
    idx_in_frag: usize,
}

impl FragIdIter {
    fn new(src: Arc<Vec<Fragment>>) -> Self {
        Self {
            src,
            frag_idx: 0,
            idx_in_frag: 0,
        }
    }
}

impl Iterator for FragIdIter {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while self.frag_idx < self.src.len() {
            let frag = &self.src[self.frag_idx];
            if self.idx_in_frag
                < frag
                    .physical_rows
                    .expect("Fragment doesn't have physical rows recorded")
            {
                let next_id =
                    RowAddress::new_from_parts(frag.id as u32, self.idx_in_frag as u32).into();
                self.idx_in_frag += 1;
                return Some(next_id);
            }
            self.frag_idx += 1;
            self.idx_in_frag = 0;
        }
        None
    }
}

impl MaterializeIndexExec {
    pub fn new(
        dataset: Arc<Dataset>,
        expr: ScalarIndexExpr,
        fragments: Arc<Vec<Fragment>>,
    ) -> Self {
        Self {
            dataset,
            expr,
            fragments,
        }
    }

    #[instrument(name = "materialize_scalar_index", skip_all, level = "debug")]
    async fn do_execute(
        expr: ScalarIndexExpr,
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
    ) -> Result<RecordBatch> {
        // TODO: multiple batches, stream without materializing all row ids in memory
        let mask = expr.evaluate(dataset.as_ref()).await?;
        let ids = match (mask.allow_list, mask.block_list) {
            (None, None) => FragIdIter::new(fragments).collect::<Vec<_>>(),
            (Some(allow_list), None) => FragIdIter::new(fragments)
                .filter(|row_id| allow_list.contains(*row_id))
                .collect(),
            (None, Some(block_list)) => FragIdIter::new(fragments)
                .filter(|row_id| !block_list.contains(*row_id))
                .collect(),
            (Some(allow_list), Some(block_list)) => FragIdIter::new(fragments)
                .filter(|row_id| !block_list.contains(*row_id) && allow_list.contains(*row_id))
                .collect(),
        };
        let ids = UInt64Array::from(ids);
        Ok(RecordBatch::try_new(
            MATERIALIZE_INDEX_SCHEMA.clone(),
            vec![Arc::new(ids)],
        )?)
    }
}

impl ExecutionPlan for MaterializeIndexExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        MATERIALIZE_INDEX_SCHEMA.clone()
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        // No guarantee a scalar index scan will return row ids in any meaningful order
        None
    }

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
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let batch_fut = Self::do_execute(
            self.expr.clone(),
            self.dataset.clone(),
            self.fragments.clone(),
        );
        let stream = futures::stream::iter(vec![batch_fut])
            .then(|batch_fut| batch_fut.map_err(|err| err.into()))
            .boxed()
            as BoxStream<'static, datafusion::common::Result<RecordBatch>>;
        Ok(Box::pin(StreamWithSchema {
            schema: MATERIALIZE_INDEX_SCHEMA.clone(),
            stream,
        }))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        todo!()
    }
}
