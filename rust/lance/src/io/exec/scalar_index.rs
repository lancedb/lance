// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, LazyLock};

use super::utils::{
    IndexMetrics, InstrumentedChildInputStream, InstrumentedRecordBatchStreamAdapter,
};
use crate::{
    Dataset,
    dataset::rowids::load_row_id_sequences,
    index::{
        prefilter::DatasetPreFilter,
        scalar_logical::{open_named_scalar_index, scalar_index_fragment_bitmap},
    },
};
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{Schema, SchemaRef};
use async_recursion::async_recursion;
use async_trait::async_trait;
use datafusion::{
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
    scalar::ScalarValue,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{StreamExt, TryFutureExt, TryStreamExt, stream::BoxStream};
use lance_core::{Error, ROW_ID_FIELD, Result, utils::address::RowAddress};
use lance_datafusion::{
    chunker::break_stream,
    utils::{
        ExecutionPlanMetricsSetExt, SCALAR_INDEX_SEARCH_TIME_METRIC, SCALAR_INDEX_SER_TIME_METRIC,
    },
};
use lance_index::{
    metrics::MetricsCollector,
    scalar::{
        SargableQuery, ScalarIndex,
        expression::{ScalarIndexExpr, ScalarIndexLoader, ScalarIndexSearch},
    },
};
use lance_select::{
    IndexExprResult, RowAddrMask, RowAddrTreeMap, RowSetOps, result::IndexExprResultWireFormat,
};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use tracing::{debug_span, instrument};

#[async_trait]
impl ScalarIndexLoader for Dataset {
    async fn load_index(
        &self,
        column: &str,
        index_name: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        open_named_scalar_index(self, column, index_name, metrics).await
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
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    result_format: IndexExprResultWireFormat,
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
    pub fn new(
        dataset: Arc<Dataset>,
        expr: ScalarIndexExpr,
        result_format: IndexExprResultWireFormat,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(result_format.schema().clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            expr,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            result_format,
        }
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn expr(&self) -> &ScalarIndexExpr {
        &self.expr
    }

    #[async_recursion]
    async fn fragments_covered_by_index_query(
        index_expr: &ScalarIndexExpr,
        dataset: &Dataset,
    ) -> Result<RoaringBitmap> {
        match index_expr {
            ScalarIndexExpr::And(lhs, rhs) => {
                Ok(Self::fragments_covered_by_index_query(lhs, dataset).await?
                    & Self::fragments_covered_by_index_query(rhs, dataset).await?)
            }
            ScalarIndexExpr::Or(lhs, rhs) => {
                Ok(Self::fragments_covered_by_index_query(lhs, dataset).await?
                    & Self::fragments_covered_by_index_query(rhs, dataset).await?)
            }
            ScalarIndexExpr::Not(expr) => {
                Self::fragments_covered_by_index_query(expr, dataset).await
            }
            ScalarIndexExpr::Query(search_key) => {
                scalar_index_fragment_bitmap(dataset, &search_key.column, &search_key.index_name)
                    .await?
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "Index not found even though it must have been found earlier: {}",
                            search_key.index_name
                        ))
                    })
            }
        }
    }

    async fn do_execute(
        expr: ScalarIndexExpr,
        dataset: Arc<Dataset>,
        plan_metrics: ExecutionPlanMetricsSet,
        result_format: IndexExprResultWireFormat,
    ) -> Result<RecordBatch> {
        let metrics = IndexMetrics::new(&plan_metrics, 0);
        let query_result = {
            let search_time = plan_metrics.new_time(SCALAR_INDEX_SEARCH_TIME_METRIC, 0);
            let _timer = search_time.timer();
            expr.evaluate(dataset.as_ref(), &metrics).await?
        };
        let fragments_covered_by_result =
            Self::fragments_covered_by_index_query(&expr, dataset.as_ref()).await?;
        {
            let ser_time = plan_metrics.new_time(SCALAR_INDEX_SER_TIME_METRIC, 0);
            let _timer = ser_time.timer();
            query_result.serialize(&fragments_covered_by_result, result_format)
        }
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
        self.result_format.schema().clone()
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
        let batch_fut = Self::do_execute(
            self.expr.clone(),
            self.dataset.clone(),
            self.metrics.clone(),
            self.result_format,
        );
        let stream = futures::stream::iter(vec![batch_fut])
            .then(|batch_fut| batch_fut.map_err(|err| err.into()))
            .boxed()
            as BoxStream<'static, datafusion::common::Result<RecordBatch>>;
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.result_format.schema().clone(),
            stream,
            partition,
            &self.metrics,
        )))
    }

    fn partition_statistics(
        &self,
        _partition: Option<usize>,
    ) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        Ok(datafusion::physical_plan::Statistics {
            num_rows: datafusion::common::stats::Precision::Exact(2),
            ..datafusion::physical_plan::Statistics::new_unknown(self.result_format.schema())
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
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
    properties: Arc<PlanProperties>,
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
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(INDEX_LOOKUP_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            column_name,
            index_name,
            input,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    async fn build_stream(
        input: datafusion::physical_plan::SendableRecordBatchStream,
        partition: usize,
        dataset: Arc<Dataset>,
        column_name: String,
        index_name: String,
        index_metrics: Arc<IndexMetrics>,
        metrics_set: ExecutionPlanMetricsSet,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        // Time the one-shot setup (fragment bitmap + deletion mask) so it's
        // attributed to this node's elapsed_compute. The helper itself only
        // times per-batch work.
        let elapsed_compute = datafusion::physical_plan::metrics::MetricBuilder::new(&metrics_set)
            .elapsed_compute(partition);
        let setup_start = std::time::Instant::now();
        let fragment_bitmap = scalar_index_fragment_bitmap(&dataset, &column_name, &index_name)
            .await?
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "IndexedLookupExec: index '{index_name}' on column '{column_name}' disappeared after planning"
                ))
            })?;
        let deletion_mask_fut =
            DatasetPreFilter::create_restricted_deletion_mask(dataset.clone(), fragment_bitmap);
        let deletion_mask = if let Some(fut) = deletion_mask_fut {
            Some(fut.await?)
        } else {
            None
        };
        elapsed_compute.add_duration(setup_start.elapsed());

        let helper = InstrumentedChildInputStream::new(
            input,
            INDEX_LOOKUP_SCHEMA.clone(),
            move |batch| {
                let column_name = column_name.clone();
                let index_name = index_name.clone();
                let dataset = dataset.clone();
                let deletion_mask = deletion_mask.clone();
                let metrics = index_metrics.clone();
                Self::map_batch(
                    column_name,
                    index_name,
                    dataset,
                    deletion_mask,
                    batch,
                    metrics,
                )
            },
            1,
            partition,
            &metrics_set,
        );
        Ok(Box::pin(helper))
    }

    async fn map_batch(
        column_name: String,
        index_name: String,
        dataset: Arc<Dataset>,
        deletion_mask: Option<Arc<RowAddrMask>>,
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
            // Internal IndexedLookup-style query — type is unknown at this layer
            index_type: String::new(),
            query: Arc::new(SargableQuery::IsIn(index_vals)),
            needs_recheck: false,
        });
        let query_result = query.evaluate(dataset.as_ref(), metrics.as_ref()).await?;
        if !query_result.is_exact() {
            todo!("Support for non-exact query results as input for merge_insert")
        }
        let mut row_addr_mask = query_result.upper;

        if let Some(deletion_mask) = deletion_mask.as_ref() {
            row_addr_mask = row_addr_mask & deletion_mask.as_ref().clone();
        }

        let row_id_iter = row_addr_mask
            .iter_addrs()
            .ok_or(datafusion::error::DataFusionError::Internal(
                "IndexedLookupExec: Cannot iterate over row addresses (BlockList or contains full fragments)".to_string(),
            ))?;
        let allow_list: UInt64Array = row_id_iter.map(u64::from).collect();
        Ok(RecordBatch::try_new(
            INDEX_LOOKUP_SCHEMA.clone(),
            vec![Arc::new(allow_list)],
        )?)
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
        let input = self.input.execute(partition, context)?;
        let stream_fut = Self::build_stream(
            input,
            partition,
            self.dataset.clone(),
            self.column_name.clone(),
            self.index_name.clone(),
            Arc::new(IndexMetrics::new(&self.metrics, partition)),
            self.metrics.clone(),
        );
        let stream = futures::stream::once(stream_fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            INDEX_LOOKUP_SCHEMA.clone(),
            stream,
        )))
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
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
    properties: Arc<PlanProperties>,
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
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(MATERIALIZE_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
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
        // MaterializeIndexExec emits a deterministic set of row ids. The
        // `upper` mask of the interval is the candidate set (the answer is
        // a subset of `upper`). For `Exact` results this is the exact
        // answer; for `AtMost` and Refined results it's a superset that
        // gets pruned downstream by `LanceFilterExec` (the full filter
        // runs on the materialized batches via the scan plan, so any
        // non-matching candidates in `upper` are dropped before they
        // reach the user). `AtLeast` carries an unbounded upper, so the
        // candidate set is the whole row space — not actionable here.
        let take_upper = |result: IndexExprResult| -> Result<RowAddrMask> {
            if result.is_at_least() && !result.is_exact() {
                todo!("Support AtLeast in MaterializeIndexExec")
            }
            Ok(result.upper)
        };
        let mask = if let Some(prefilter) = prefilter {
            let (expr_result, prefilter) = futures::try_join!(expr_result, prefilter)?;
            take_upper(expr_result)? & (*prefilter).clone()
        } else {
            take_upper(expr_result.await?)?
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
    mask: RowAddrMask,
    dataset: &Dataset,
    fragments: &[Fragment],
) -> Result<Vec<u64>> {
    match mask {
        RowAddrMask::BlockList(block_list) if block_list.is_empty() => {
            // Matches all row ids in the given fragments.
            if dataset.manifest.uses_stable_row_ids() {
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
        RowAddrMask::AllowList(mut allow_list) => {
            retain_fragments(&mut allow_list, fragments, dataset).await?;

            if let Some(allow_list_iter) = allow_list.row_addrs() {
                Ok(allow_list_iter.map(u64::from).collect::<Vec<_>>())
            } else {
                // We shouldn't hit this branch if the row ids are stable.
                debug_assert!(!dataset.manifest.uses_stable_row_ids());
                Ok(FragIdIter::new(fragments)
                    .filter(|row_id| allow_list.contains(*row_id))
                    .collect())
            }
        }
        RowAddrMask::BlockList(block_list) => {
            if dataset.manifest.uses_stable_row_ids() {
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
    }
}

async fn retain_fragments(
    allow_list: &mut RowAddrTreeMap,
    fragments: &[Fragment],
    dataset: &Dataset,
) -> Result<()> {
    if dataset.manifest.uses_stable_row_ids() {
        let fragment_ids = load_row_id_sequences(dataset, fragments)
            .map_ok(|(_frag_id, sequence)| RowAddrTreeMap::from(sequence.as_ref()))
            .try_fold(RowAddrTreeMap::new(), |mut acc, tree| async {
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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{ops::Bound, sync::Arc};

    use crate::index::DatasetIndexExt;
    use arrow::datatypes::UInt64Type;
    use arrow_schema::Schema;
    use datafusion::{
        execution::TaskContext, physical_plan::ExecutionPlan, prelude::SessionConfig,
        scalar::ScalarValue,
    };
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::gen_batch;
    use lance_index::{
        IndexType,
        scalar::{
            SargableQuery, ScalarIndexParams,
            expression::{ScalarIndexExpr, ScalarIndexSearch},
        },
    };
    use lance_select::result::IndexExprResultWireFormat;

    use crate::{
        Dataset,
        io::exec::scalar_index::MaterializeIndexExec,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, NoContextTestFixture},
    };

    use super::{MapIndexExec, ScalarIndexExec};

    struct TestFixture {
        dataset: Arc<Dataset>,
        _tmp_dir_guard: TempStrDir,
    }

    async fn test_fixture() -> TestFixture {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let mut dataset = gen_batch()
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
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt64(Some(47))),
            )),
            needs_recheck: false,
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

    /// `ScalarIndexExec::schema()` (and the stream it emits) must advertise
    /// the same schema the batch actually carries — otherwise downstream
    /// consumers that trust `ExecutionPlan::schema()` will see a different
    /// shape than they receive.
    ///
    /// The schema depends on the `IndexExprResultWireFormat` passed to `ScalarIndexExec::new`.
    #[tokio::test]
    async fn test_scalar_index_exec_advertises_correct_schema() {
        let TestFixture {
            dataset,
            _tmp_dir_guard,
        } = test_fixture().await;

        let query = ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "ordered".to_string(),
            index_name: "ordered_idx".to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt64(Some(47))),
            )),
            needs_recheck: false,
        });

        let verify = async |plan: ScalarIndexExec, schema: Arc<Schema>| {
            assert_eq!(plan.schema(), schema);
            assert_eq!(
                plan.partition_statistics(None)
                    .unwrap()
                    .column_statistics
                    .len(),
                schema.fields().len(),
            );

            let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
            assert_eq!(stream.schema(), schema);
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();
            assert_eq!(batches.len(), 1);
            assert_eq!(batches[0].schema(), schema);
        };

        let plan = ScalarIndexExec::new(
            dataset.clone(),
            query.clone(),
            IndexExprResultWireFormat::ThreeVariant,
        );
        let schema = IndexExprResultWireFormat::ThreeVariant.schema().clone();

        verify(plan, schema).await;

        let plan = ScalarIndexExec::new(dataset, query, IndexExprResultWireFormat::TwoMask);
        let schema = IndexExprResultWireFormat::TwoMask.schema().clone();

        verify(plan, schema).await;
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
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt64(Some(47))),
            )),
            needs_recheck: false,
        });

        // These plans aren't even valid but it appears we defer all work (even validation) until
        // read time.
        let plan = ScalarIndexExec::new(
            arc_dasaset.clone(),
            query.clone(),
            IndexExprResultWireFormat::default(),
        );
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
