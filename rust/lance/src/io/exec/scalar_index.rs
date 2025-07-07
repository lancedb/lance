// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, LazyLock};

use super::utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter};
use crate::{
    dataset::rowids::load_row_id_sequences,
    index::{prefilter::DatasetPreFilter, DatasetIndexInternalExt},
    Dataset,
};
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    common::{stats::Precision, Statistics},
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
        stream::RecordBatchStreamAdapter,
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    },
    scalar::ScalarValue,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{stream::BoxStream, Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::{
    utils::{
        address::RowAddress,
        mask::{RowIdMask, RowIdTreeMap},
    },
    Error, Result, ROW_ID_FIELD,
};
use lance_datafusion::chunker::break_stream;
use lance_index::{
    metrics::MetricsCollector,
    scalar::{
        expression::{IndexExprResult, ScalarIndexExpr, ScalarIndexLoader, ScalarIndexSearch},
        SargableQuery, ScalarIndex,
    },
    DatasetIndexExt, ScalarIndexCriteria,
};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use snafu::location;
use tracing::{debug_span, instrument};

pub static SCALAR_INDEX_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![Field::new(
        "result".to_string(),
        DataType::Binary,
        true,
    )]))
});

#[async_trait]
impl ScalarIndexLoader for Dataset {
    async fn load_index(
        &self,
        column: &str,
        index_name: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let idx = self
            .load_scalar_index(ScalarIndexCriteria::default().with_name(index_name))
            .await?
            .ok_or_else(|| Error::Internal {
                message: format!("Scanner created plan for index query on index {} for column {} but no usable index exists with that name", index_name, column),
                location: location!()
            })?;
        self.open_scalar_index(column, &idx.uuid.to_string(), metrics)
            .await
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
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for ScalarIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ScalarIndexQuery: query={}", self.expr)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "ScalarIndexQuery\nquery={}", self.expr)
            }
        }
    }
}

impl ScalarIndexExec {
    pub fn new(dataset: Arc<Dataset>, expr: ScalarIndexExpr) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(SCALAR_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            expr,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    async fn do_execute(
        expr: ScalarIndexExpr,
        dataset: Arc<Dataset>,
        metrics: IndexMetrics,
    ) -> Result<RecordBatch> {
        let query_result = expr.evaluate(dataset.as_ref(), &metrics).await?;
        let IndexExprResult::Exact(row_id_mask) = query_result else {
            todo!("Support for non-exact query results as pre-filter for vector search")
        };
        let row_id_mask_arr = row_id_mask.into_arrow()?;
        Ok(RecordBatch::try_new(
            SCALAR_INDEX_SCHEMA.clone(),
            vec![Arc::new(row_id_mask_arr)],
        )?)
    }
}

impl ExecutionPlan for ScalarIndexExec {
    fn name(&self) -> &str {
        "ScalarIndexExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        SCALAR_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            Err(datafusion::error::DataFusionError::Internal(
                "ScalarIndexExec does not have children".to_string(),
            ))
        } else {
            Ok(self)
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let metrics = IndexMetrics::new(&self.metrics, partition);
        let batch_fut = Self::do_execute(self.expr.clone(), self.dataset.clone(), metrics);
        let stream = futures::stream::iter(vec![batch_fut])
            .then(|batch_fut| batch_fut.map_err(|err| err.into()))
            .boxed()
            as BoxStream<'static, datafusion::common::Result<RecordBatch>>;
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            SCALAR_INDEX_SCHEMA.clone(),
            stream,
            partition,
            &self.metrics,
        )))
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(2),
            ..Statistics::new_unknown(&SCALAR_INDEX_SCHEMA)
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

pub static INDEX_LOOKUP_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()])));

/// An execution node that translates index values into row addresses
///
/// This can be combined with TakeExec to perform an "indexed take"
#[derive(Debug)]
pub struct MapIndexExec {
    dataset: Arc<Dataset>,
    column_name: String,
    index_name: String,
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for MapIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "IndexedLookup")
            }
        }
    }
}

impl MapIndexExec {
    pub fn new(
        dataset: Arc<Dataset>,
        column_name: String,
        index_name: String,
        input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(INDEX_LOOKUP_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            column_name,
            index_name,
            input,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    async fn map_batch(
        column_name: String,
        index_name: String,
        dataset: Arc<Dataset>,
        deletion_mask: Option<Arc<RowIdMask>>,
        batch: RecordBatch,
        metrics: Arc<IndexMetrics>,
    ) -> datafusion::error::Result<RecordBatch> {
        let index_vals = batch.column(0);
        let index_vals = (0..index_vals.len())
            .map(|idx| ScalarValue::try_from_array(index_vals, idx))
            .collect::<datafusion::error::Result<Vec<_>>>()?;
        let query = ScalarIndexExpr::Query(ScalarIndexSearch {
            column: column_name,
            index_name,
            query: Arc::new(SargableQuery::IsIn(index_vals)),
        });
        let query_result = query.evaluate(dataset.as_ref(), metrics.as_ref()).await?;
        let IndexExprResult::Exact(mut row_id_mask) = query_result else {
            todo!("Support for non-exact query results as input for merge_insert")
        };

        if let Some(deletion_mask) = deletion_mask.as_ref() {
            row_id_mask = row_id_mask & deletion_mask.as_ref().clone();
        }

        if let Some(mut allow_list) = row_id_mask.allow_list {
            // Flatten the allow list
            if let Some(block_list) = row_id_mask.block_list {
                allow_list -= &block_list;
            }

            let allow_list =
                allow_list
                    .row_ids()
                    .ok_or(datafusion::error::DataFusionError::External(
                        "IndexedLookupExec: row addresses didn't have an iterable allow list"
                            .into(),
                    ))?;
            let allow_list: UInt64Array = allow_list.map(u64::from).collect();
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
        index_name: String,
        metrics: Arc<IndexMetrics>,
    ) -> datafusion::error::Result<
        impl Stream<Item = datafusion::error::Result<RecordBatch>> + Send + 'static,
    > {
        let index = dataset
            .load_scalar_index(ScalarIndexCriteria::default().with_name(&index_name))
            .await?
            .unwrap();
        let deletion_mask_fut =
            DatasetPreFilter::create_deletion_mask(dataset.clone(), index.fragment_bitmap.unwrap());
        let deletion_mask = if let Some(deletion_mask_fut) = deletion_mask_fut {
            Some(deletion_mask_fut.await?)
        } else {
            None
        };
        Ok(input.and_then(move |res| {
            let column_name = column_name.clone();
            let index_name = index_name.clone();
            let dataset = dataset.clone();
            let deletion_mask = deletion_mask.clone();
            let metrics = metrics.clone();
            Self::map_batch(
                column_name,
                index_name,
                dataset,
                deletion_mask,
                res,
                metrics,
            )
        }))
    }
}

impl ExecutionPlan for MapIndexExec {
    fn name(&self) -> &str {
        "MapIndexExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        INDEX_LOOKUP_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            Err(datafusion::error::DataFusionError::Internal(
                "MapIndexExec requires exactly one child".to_string(),
            ))
        } else {
            Ok(Arc::new(Self::new(
                self.dataset.clone(),
                self.column_name.clone(),
                self.index_name.clone(),
                children.into_iter().next().unwrap(),
            )))
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let index_vals = self.input.execute(partition, context)?;
        let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let stream_fut = Self::do_execute(
            index_vals,
            self.dataset.clone(),
            self.column_name.clone(),
            self.index_name.clone(),
            metrics,
        );
        let stream = futures::stream::iter(vec![stream_fut])
            .then(|stream_fut| stream_fut)
            .try_flatten()
            .boxed();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            INDEX_LOOKUP_SCHEMA.clone(),
            stream,
            partition,
            &self.metrics,
        )))
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

pub static MATERIALIZE_INDEX_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()])));

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
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for MaterializeIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MaterializeIndex: query={}", self.expr)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "MaterializeIndex\nquery={}", self.expr)
            }
        }
    }
}

struct FragIdIter<'a> {
    src: &'a [Fragment],
    frag_idx: usize,
    idx_in_frag: usize,
}

impl<'a> FragIdIter<'a> {
    fn new(src: &'a [Fragment]) -> Self {
        Self {
            src,
            frag_idx: 0,
            idx_in_frag: 0,
        }
    }
}

impl Iterator for FragIdIter<'_> {
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
        let properties = PlanProperties::new(
            EquivalenceProperties::new(MATERIALIZE_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            expr,
            fragments,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    #[instrument(name = "materialize_scalar_index", skip_all, level = "debug")]
    async fn do_execute(
        expr: ScalarIndexExpr,
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        metrics: Arc<IndexMetrics>,
    ) -> Result<RecordBatch> {
        let expr_result = expr.evaluate(dataset.as_ref(), metrics.as_ref());
        let span = debug_span!("create_prefilter");
        let prefilter = span.in_scope(|| {
            let fragment_bitmap =
                RoaringBitmap::from_iter(fragments.iter().map(|frag| frag.id as u32));
            // The user-requested `fragments` is guaranteed to be stricter than the index's fragment
            // bitmap.  This node only runs on indexed fragments and any fragments that were deleted
            // when the index was trained will still be deleted when the index is queried.
            DatasetPreFilter::create_deletion_mask(dataset.clone(), fragment_bitmap)
        });
        let mask = if let Some(prefilter) = prefilter {
            let (expr_result, prefilter) = futures::try_join!(expr_result, prefilter)?;
            let mask = match expr_result {
                IndexExprResult::Exact(mask) => mask,
                IndexExprResult::AtMost(mask) => mask,
                IndexExprResult::AtLeast(_) => todo!("Support AtLeast in MaterializeIndexExec"),
            };
            mask & (*prefilter).clone()
        } else {
            let expr_result = expr_result.await?;
            match expr_result {
                IndexExprResult::Exact(mask) => mask,
                IndexExprResult::AtMost(mask) => mask,
                IndexExprResult::AtLeast(_) => todo!("Support AtLeast in MaterializeIndexExec"),
            }
        };
        let ids = row_ids_for_mask(mask, &dataset, &fragments).await?;
        let ids = UInt64Array::from(ids);
        Ok(RecordBatch::try_new(
            MATERIALIZE_INDEX_SCHEMA.clone(),
            vec![Arc::new(ids)],
        )?)
    }
}

#[instrument(name = "make_row_ids", skip(mask, dataset, fragments))]
async fn row_ids_for_mask(
    mask: RowIdMask,
    dataset: &Dataset,
    fragments: &[Fragment],
) -> Result<Vec<u64>> {
    match (mask.allow_list, mask.block_list) {
        (None, None) => {
            // Matches all row ids in the given fragments.
            if dataset.manifest.uses_move_stable_row_ids() {
                let sequences = load_row_id_sequences(dataset, fragments)
                    .map_ok(|(_frag_id, sequence)| sequence)
                    .try_collect::<Vec<_>>()
                    .await?;

                let capacity = sequences.iter().map(|seq| seq.len() as usize).sum();
                let mut row_ids = Vec::with_capacity(capacity);
                for sequence in sequences {
                    row_ids.extend(sequence.iter());
                }
                Ok(row_ids)
            } else {
                Ok(FragIdIter::new(fragments).collect::<Vec<_>>())
            }
        }
        (Some(mut allow_list), None) => {
            retain_fragments(&mut allow_list, fragments, dataset).await?;

            if let Some(allow_list_iter) = allow_list.row_ids() {
                Ok(allow_list_iter.map(u64::from).collect::<Vec<_>>())
            } else {
                // We shouldn't hit this branch if the row ids are stable.
                debug_assert!(!dataset.manifest.uses_move_stable_row_ids());
                Ok(FragIdIter::new(fragments)
                    .filter(|row_id| allow_list.contains(*row_id))
                    .collect())
            }
        }
        (None, Some(block_list)) => {
            if dataset.manifest.uses_move_stable_row_ids() {
                let sequences = load_row_id_sequences(dataset, fragments)
                    .map_ok(|(_frag_id, sequence)| sequence)
                    .try_collect::<Vec<_>>()
                    .await?;

                let mut capacity = sequences.iter().map(|seq| seq.len() as usize).sum();
                capacity -= block_list.len().expect("unknown block list len") as usize;
                let mut row_ids = Vec::with_capacity(capacity);
                for sequence in sequences {
                    row_ids.extend(
                        sequence
                            .iter()
                            .filter(|row_id| !block_list.contains(*row_id)),
                    );
                }
                Ok(row_ids)
            } else {
                Ok(FragIdIter::new(fragments)
                    .filter(|row_id| !block_list.contains(*row_id))
                    .collect())
            }
        }
        (Some(mut allow_list), Some(block_list)) => {
            // We need to filter out irrelevant fragments as well.
            retain_fragments(&mut allow_list, fragments, dataset).await?;

            if let Some(allow_list_iter) = allow_list.row_ids() {
                Ok(allow_list_iter
                    .filter_map(|addr| {
                        let row_id = u64::from(addr);
                        if !block_list.contains(row_id) {
                            Some(row_id)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>())
            } else {
                // We shouldn't hit this branch if the row ids are stable.
                debug_assert!(!dataset.manifest.uses_move_stable_row_ids());
                Ok(FragIdIter::new(fragments)
                    .filter(|row_id| !block_list.contains(*row_id) && allow_list.contains(*row_id))
                    .collect())
            }
        }
    }
}

async fn retain_fragments(
    allow_list: &mut RowIdTreeMap,
    fragments: &[Fragment],
    dataset: &Dataset,
) -> Result<()> {
    if dataset.manifest.uses_move_stable_row_ids() {
        let fragment_ids = load_row_id_sequences(dataset, fragments)
            .map_ok(|(_frag_id, sequence)| RowIdTreeMap::from(sequence.as_ref()))
            .try_fold(RowIdTreeMap::new(), |mut acc, tree| async {
                acc |= tree;
                Ok(acc)
            })
            .await?;
        *allow_list &= &fragment_ids;
    } else {
        // Assume row ids are addresses, so we can filter out fragments by their ids.
        allow_list.retain_fragments(fragments.iter().map(|frag| frag.id as u32));
    }
    Ok(())
}

impl ExecutionPlan for MaterializeIndexExec {
    fn name(&self) -> &str {
        "MaterializeIndexExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        MATERIALIZE_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            Err(datafusion::error::DataFusionError::Internal(
                "MaterializeIndexExec does not have children".to_string(),
            ))
        } else {
            Ok(self)
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let batch_fut = Self::do_execute(
            self.expr.clone(),
            self.dataset.clone(),
            self.fragments.clone(),
            metrics,
        );
        let stream = futures::stream::iter(vec![batch_fut])
            .then(|batch_fut| batch_fut.map_err(|err| err.into()))
            .boxed()
            as BoxStream<'static, datafusion::common::Result<RecordBatch>>;
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            MATERIALIZE_INDEX_SCHEMA.clone(),
            stream,
        ));
        let stream = break_stream(stream, context.session_config().batch_size());
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            MATERIALIZE_INDEX_SCHEMA.clone(),
            stream.map_err(|err| err.into()),
            partition,
            &self.metrics,
        )))
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        Ok(Statistics::new_unknown(&MATERIALIZE_INDEX_SCHEMA))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use std::{ops::Bound, sync::Arc};

    use arrow::datatypes::UInt64Type;
    use datafusion::{
        execution::TaskContext, physical_plan::ExecutionPlan, prelude::SessionConfig,
        scalar::ScalarValue,
    };
    use futures::TryStreamExt;
    use lance_datagen::gen;
    use lance_index::{
        scalar::{
            expression::{ScalarIndexExpr, ScalarIndexSearch},
            SargableQuery, ScalarIndexParams,
        },
        DatasetIndexExt, IndexType,
    };
    use tempfile::{tempdir, TempDir};

    use crate::{
        io::exec::scalar_index::MaterializeIndexExec,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, NoContextTestFixture},
        Dataset,
    };

    use super::{MapIndexExec, ScalarIndexExec};

    struct TestFixture {
        dataset: Arc<Dataset>,
        _tmp_dir_guard: TempDir,
    }

    async fn test_fixture() -> TestFixture {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut dataset = gen()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_dataset(
                test_uri,
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        dataset
            .create_index(
                &["ordered"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        TestFixture {
            dataset: Arc::new(dataset),
            _tmp_dir_guard: test_dir,
        }
    }

    #[tokio::test]
    async fn test_materialize_index_exec() {
        let TestFixture {
            dataset,
            _tmp_dir_guard,
        } = test_fixture().await;

        let query = ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "ordered".to_string(),
            index_name: "ordered_idx".to_string(),
            query: Arc::new(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt64(Some(47))),
            )),
        });

        let fragments = dataset.fragments().clone();

        let plan = MaterializeIndexExec::new(dataset, query, fragments);

        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 47);

        let context =
            TaskContext::default().with_session_config(SessionConfig::default().with_batch_size(5));
        let stream = plan.execute(0, Arc::new(context)).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 10);
        assert_eq!(batches[0].num_rows(), 5);
    }

    #[test]
    fn no_context_scalar_index() {
        // These tests ensure we can create nodes and call execute without a tokio Runtime
        // being active.  This is a requirement for proper implementation of a Datafusion foreign
        // table provider.
        let fixture = NoContextTestFixture::new();
        let arc_dasaset = Arc::new(fixture.dataset);

        let query = ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "ordered".to_string(),
            index_name: "ordered_idx".to_string(),
            query: Arc::new(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt64(Some(47))),
            )),
        });

        // These plans aren't even valid but it appears we defer all work (even validation) until
        // read time.
        let plan = ScalarIndexExec::new(arc_dasaset.clone(), query.clone());
        plan.execute(0, Arc::new(TaskContext::default())).unwrap();

        let plan = MapIndexExec::new(
            arc_dasaset.clone(),
            "ordered".to_string(),
            "ordered_idx".to_string(),
            Arc::new(plan),
        );
        plan.execute(0, Arc::new(TaskContext::default())).unwrap();

        let plan =
            MaterializeIndexExec::new(arc_dasaset.clone(), query, arc_dasaset.fragments().clone());
        plan.execute(0, Arc::new(TaskContext::default())).unwrap();
    }
}
