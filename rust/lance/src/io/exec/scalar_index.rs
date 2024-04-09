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
use datafusion::{
    physical_plan::{
        stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionMode,
        ExecutionPlan, Partitioning, PlanProperties,
    },
    scalar::ScalarValue,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{stream::BoxStream, Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::{
    utils::{address::RowAddress, mask::RowIdTreeMap},
    Error, Result, ROW_ID_FIELD,
};
use lance_index::{
    scalar::{
        expression::{ScalarIndexExpr, ScalarIndexLoader},
        ScalarIndex, ScalarQuery,
    },
    DatasetIndexExt,
};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use tracing::{debug_span, instrument};

use crate::{
    index::{prefilter::PreFilter, DatasetIndexInternalExt},
    Dataset,
};

lazy_static::lazy_static! {
    pub static ref SCALAR_INDEX_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![Field::new("result".to_string(), DataType::Binary, true)]));
    pub static ref SCALAR_INDEX_PROPERTIES: Arc<PlanProperties> = Arc::new(PlanProperties::new(
        EquivalenceProperties::new(SCALAR_INDEX_SCHEMA.clone()),
        Partitioning::RoundRobinBatch(1),
        ExecutionMode::Bounded,
    ));
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

impl ExecutionPlan for ScalarIndexExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        SCALAR_INDEX_SCHEMA.clone()
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
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            SCALAR_INDEX_SCHEMA.clone(),
            stream,
        )))
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        todo!()
    }

    fn properties(&self) -> &PlanProperties {
        &SCALAR_INDEX_PROPERTIES
    }
}

lazy_static::lazy_static! {
    pub static ref INDEX_LOOKUP_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()]));

}

/// An execution node that translates index values into row addresses
///
/// This can be combined with TakeExec to perform an "indexed take"
#[derive(Debug)]
pub struct MapIndexExec {
    dataset: Arc<Dataset>,
    column_name: String,
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
}

impl DisplayAs for MapIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "IndexedLookup")
            }
        }
    }
}

impl MapIndexExec {
    pub fn new(dataset: Arc<Dataset>, column_name: String, input: Arc<dyn ExecutionPlan>) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(INDEX_LOOKUP_SCHEMA.clone()),
            input.properties().output_partitioning().clone(),
            ExecutionMode::Bounded,
        );

        Self {
            dataset,
            column_name,
            input,
            properties,
        }
    }

    async fn map_batch(
        column_name: String,
        dataset: Arc<Dataset>,
        deletion_mask: Option<Arc<RowIdTreeMap>>,
        batch: RecordBatch,
    ) -> datafusion::error::Result<RecordBatch> {
        let index_vals = batch.column(0);
        let index_vals = (0..index_vals.len())
            .map(|idx| ScalarValue::try_from_array(index_vals, idx))
            .collect::<datafusion::error::Result<Vec<_>>>()?;
        let query = ScalarIndexExpr::Query(column_name.clone(), ScalarQuery::IsIn(index_vals));
        let row_addresses = query.evaluate(dataset.as_ref()).await?;
        debug_assert!(row_addresses.block_list.is_none());
        if let Some(allow_list) = row_addresses.allow_list {
            let allow_list =
                allow_list
                    .row_ids()
                    .ok_or(datafusion::error::DataFusionError::External(
                        "IndexedLookupExec: row addresses didn't have an iterable allow list"
                            .into(),
                    ))?;
            let mut allow_list = allow_list.map(u64::from).collect::<Vec<_>>();
            if let Some(deletion_mask) = deletion_mask {
                allow_list.retain(|row_id| !deletion_mask.contains(*row_id));
            }
            let allow_list = UInt64Array::from(allow_list);
            Ok(RecordBatch::try_new(
                INDEX_LOOKUP_SCHEMA.clone(),
                vec![Arc::new(allow_list)],
            )?)
        } else {
            Err(datafusion::error::DataFusionError::Internal(
                "IndexedLookupExec: row addresses didn't have an allow list".to_string(),
            ))
        }
    }

    async fn do_execute(
        input: datafusion::physical_plan::SendableRecordBatchStream,
        dataset: Arc<Dataset>,
        column_name: String,
    ) -> datafusion::error::Result<
        impl Stream<Item = datafusion::error::Result<RecordBatch>> + Send + 'static,
    > {
        let index = dataset
            .load_scalar_index_for_column(&column_name)
            .await?
            .unwrap();
        let deletion_mask_fut =
            PreFilter::create_deletion_mask(dataset.clone(), index.fragment_bitmap.unwrap());
        let deletion_mask = if let Some(deletion_mask_fut) = deletion_mask_fut {
            Some(deletion_mask_fut.await?)
        } else {
            None
        };
        Ok(input.and_then(move |res| {
            let column_name = column_name.clone();
            let dataset = dataset.clone();
            let deletion_mask = deletion_mask.clone();
            Self::map_batch(column_name, dataset, deletion_mask, res)
        }))
    }
}

impl ExecutionPlan for MapIndexExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        INDEX_LOOKUP_SCHEMA.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let index_vals = self.input.execute(partition, context)?;
        let stream_fut =
            Self::do_execute(index_vals, self.dataset.clone(), self.column_name.clone());
        let stream = futures::stream::iter(vec![stream_fut])
            .then(|stream_fut| stream_fut)
            .try_flatten()
            .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            INDEX_LOOKUP_SCHEMA.clone(),
            stream,
        )))
    }
}

lazy_static::lazy_static! {
    pub static ref MATERIALIZE_INDEX_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()]));
    pub static ref MATERIALIZE_INDEX_PROPERTIES: Arc<PlanProperties> = Arc::new(PlanProperties::new(
        EquivalenceProperties::new(MATERIALIZE_INDEX_SCHEMA.clone()),
        Partitioning::RoundRobinBatch(1),
        ExecutionMode::Bounded,
    ));
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
        let mask = expr.evaluate(dataset.as_ref());
        let span = debug_span!("create_prefilter");
        let prefilter = span.in_scope(|| {
            let fragment_bitmap =
                RoaringBitmap::from_iter(fragments.iter().map(|frag| frag.id as u32));
            // The user-requested `fragments` is guaranteed to be stricter than the index's fragment
            // bitmap.  This node only runs on indexed fragments and any fragments that were deleted
            // when the index was trained will still be deleted when the index is queried.
            PreFilter::create_deletion_mask(dataset.clone(), fragment_bitmap)
        });
        let mask = if let Some(prefilter) = prefilter {
            let (mask, prefilter) = futures::try_join!(mask, prefilter)?;
            mask.also_block((*prefilter).clone())
        } else {
            mask.await?
        };
        let span = debug_span!("make_ids");
        let ids = span.in_scope(|| match (mask.allow_list, mask.block_list) {
            (None, None) => FragIdIter::new(fragments).collect::<Vec<_>>(),
            (Some(mut allow_list), None) => {
                allow_list.remove_fragments(fragments.iter().map(|frag| frag.id as u32));
                if let Some(allow_list_iter) = allow_list.row_ids() {
                    allow_list_iter.map(u64::from).collect::<Vec<_>>()
                } else {
                    FragIdIter::new(fragments)
                        .filter(|row_id| allow_list.contains(*row_id))
                        .collect()
                }
            }
            (None, Some(block_list)) => FragIdIter::new(fragments)
                .filter(|row_id| !block_list.contains(*row_id))
                .collect(),
            (Some(mut allow_list), Some(block_list)) => {
                allow_list.remove_fragments(fragments.iter().map(|frag| frag.id as u32));
                if let Some(allow_list_iter) = allow_list.row_ids() {
                    allow_list_iter
                        .filter_map(|addr| {
                            let row_id = u64::from(addr);
                            if !block_list.contains(row_id) {
                                Some(row_id)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                } else {
                    FragIdIter::new(fragments)
                        .filter(|row_id| {
                            !block_list.contains(*row_id) && allow_list.contains(*row_id)
                        })
                        .collect()
                }
            }
        });
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

    fn properties(&self) -> &PlanProperties {
        MATERIALIZE_INDEX_PROPERTIES.as_ref()
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
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            MATERIALIZE_INDEX_SCHEMA.clone(),
            stream,
        )))
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        todo!()
    }
}
