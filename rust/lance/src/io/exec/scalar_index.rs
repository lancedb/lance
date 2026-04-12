// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, LazyLock};

use super::utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter};
use crate::{
    Dataset,
    dataset::rowids::load_row_id_sequences,
    index::{
        DatasetIndexExt, DatasetIndexInternalExt,
        coverage::fragments_covered_by_scalar_index_query, prefilter::DatasetPreFilter,
    },
};
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    common::{Statistics, stats::Precision},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
    scalar::ScalarValue,
};
use datafusion_physical_expr::EquivalenceProperties;
use deepsize::DeepSizeOf;
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt, stream::BoxStream};
use lance_core::utils::mask::RowSetOps;
use lance_core::{
    Error, ROW_ID_FIELD, Result,
    utils::{
        address::RowAddress,
        mask::{RowAddrMask, RowAddrTreeMap},
    },
};
use lance_datafusion::{
    chunker::break_stream,
    utils::{
        ExecutionPlanMetricsSetExt, SCALAR_INDEX_SEARCH_TIME_METRIC, SCALAR_INDEX_SER_TIME_METRIC,
    },
};
use lance_index::Index;
use lance_index::{
    IndexCriteria, IndexType,
    metrics::MetricsCollector,
    scalar::{
        SargableQuery, ScalarIndex, SearchResult,
        expression::{
            INDEX_EXPR_RESULT_SCHEMA, IndexExprResult, ScalarIndexExpr, ScalarIndexLoader,
            ScalarIndexSearch,
        },
    },
};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use serde_json::json;
use tracing::{debug_span, instrument};

#[derive(Debug, DeepSizeOf)]
struct LogicalScalarIndex {
    segments: Vec<Arc<dyn ScalarIndex>>,
    index_type: IndexType,
}

impl LogicalScalarIndex {
    fn try_new(segments: Vec<Arc<dyn ScalarIndex>>) -> Result<Self> {
        let Some(index_type) = segments.first().map(|segment| segment.index_type()) else {
            return Err(Error::internal(
                "Logical scalar index requires at least one segment".to_string(),
            ));
        };
        if !segments
            .iter()
            .all(|segment| segment.index_type() == index_type)
        {
            return Err(Error::internal(
                "Logical scalar index segments must all have the same index type".to_string(),
            ));
        }
        Ok(Self {
            segments,
            index_type,
        })
    }

    fn combine_search_results(results: &[SearchResult]) -> SearchResult {
        use lance_core::utils::mask::NullableRowAddrSet;

        let lower = NullableRowAddrSet::union_all(
            &results
                .iter()
                .filter_map(|result| match result {
                    SearchResult::Exact(rows) | SearchResult::AtLeast(rows) => Some(rows.clone()),
                    SearchResult::AtMost(_) => None,
                })
                .collect::<Vec<_>>(),
        );
        let upper = results
            .iter()
            .map(|result| match result {
                SearchResult::Exact(rows) | SearchResult::AtMost(rows) => Some(rows.clone()),
                SearchResult::AtLeast(_) => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(|rows| NullableRowAddrSet::union_all(&rows));

        match upper {
            Some(upper) if lower == upper => SearchResult::Exact(upper),
            Some(upper) => SearchResult::AtMost(upper),
            None => SearchResult::AtLeast(lower),
        }
    }
}

#[async_trait]
impl Index for LogicalScalarIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn lance_index::vector::VectorIndex>> {
        Err(Error::invalid_input(
            "Logical scalar index cannot be used as a vector index".to_string(),
        ))
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(json!({
            "num_segments": self.segments.len(),
            "segments": self.segments.iter().map(|segment| segment.statistics()).collect::<Result<Vec<_>>>()?,
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        for segment in &self.segments {
            segment.prewarm().await?;
        }
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        self.index_type
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut fragments = RoaringBitmap::new();
        for segment in &self.segments {
            fragments |= segment.calculate_included_frags().await?;
        }
        Ok(fragments)
    }
}

#[async_trait]
impl ScalarIndex for LogicalScalarIndex {
    async fn search(
        &self,
        query: &dyn lance_index::scalar::AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let mut results = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            results.push(segment.search(query, metrics).await?);
        }
        Ok(Self::combine_search_results(&results))
    }

    fn can_remap(&self) -> bool {
        false
    }

    async fn remap(
        &self,
        _mapping: &std::collections::HashMap<u64, Option<u64>>,
        _dest_store: &dyn lance_index::scalar::IndexStore,
    ) -> Result<lance_index::scalar::CreatedIndex> {
        Err(Error::not_supported(
            "Logical scalar index is query-only".to_string(),
        ))
    }

    async fn update(
        &self,
        _new_data: datafusion::execution::SendableRecordBatchStream,
        _dest_store: &dyn lance_index::scalar::IndexStore,
        _old_data_filter: Option<lance_index::scalar::OldIndexDataFilter>,
    ) -> Result<lance_index::scalar::CreatedIndex> {
        Err(Error::not_supported(
            "Logical scalar index is query-only".to_string(),
        ))
    }

    fn update_criteria(&self) -> lance_index::scalar::UpdateCriteria {
        lance_index::scalar::UpdateCriteria::requires_old_data(
            lance_index::scalar::registry::TrainingCriteria::new(
                lance_index::scalar::registry::TrainingOrdering::None,
            ),
        )
    }

    fn derive_index_params(&self) -> Result<lance_index::scalar::ScalarIndexParams> {
        self.segments[0].derive_index_params()
    }
}

#[async_trait]
impl ScalarIndexLoader for Dataset {
    async fn load_index(
        &self,
        column: &str,
        index_name: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let indices = self.load_indices_by_name(index_name).await?;
        match indices.len() {
            0 => Err(Error::internal(format!(
                "Scanner created plan for index query on index {} for column {} but no usable index exists with that name",
                index_name, column
            ))),
            1 => {
                self.open_scalar_index(column, &indices[0].uuid.to_string(), metrics)
                    .await
            }
            _ => {
                let mut segments = Vec::with_capacity(indices.len());
                for index in indices {
                    segments.push(
                        self.open_scalar_index(column, &index.uuid.to_string(), metrics)
                            .await?,
                    );
                }
                Ok(Arc::new(LogicalScalarIndex::try_new(segments)?) as Arc<dyn ScalarIndex>)
            }
        }
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
            EquivalenceProperties::new(INDEX_EXPR_RESULT_SCHEMA.clone()),
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
        plan_metrics: ExecutionPlanMetricsSet,
    ) -> Result<RecordBatch> {
        let metrics = IndexMetrics::new(&plan_metrics, 0);
        let query_result = {
            let search_time = plan_metrics.new_time(SCALAR_INDEX_SEARCH_TIME_METRIC, 0);
            let _timer = search_time.timer();
            expr.evaluate(dataset.as_ref(), &metrics).await?
        };
        let fragments_covered_by_result =
            fragments_covered_by_scalar_index_query(dataset.as_ref(), &expr).await?;
        {
            let ser_time = plan_metrics.new_time(SCALAR_INDEX_SER_TIME_METRIC, 0);
            let _timer = ser_time.timer();
            query_result.serialize_to_arrow(&fragments_covered_by_result)
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
        INDEX_EXPR_RESULT_SCHEMA.clone()
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
        );
        let stream = futures::stream::iter(vec![batch_fut])
            .then(|batch_fut| batch_fut.map_err(|err| err.into()))
            .boxed()
            as BoxStream<'static, datafusion::common::Result<RecordBatch>>;
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            INDEX_EXPR_RESULT_SCHEMA.clone(),
            stream,
            partition,
            &self.metrics,
        )))
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(2),
            ..Statistics::new_unknown(&INDEX_EXPR_RESULT_SCHEMA)
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
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
            query: Arc::new(SargableQuery::IsIn(index_vals)),
            needs_recheck: false,
        });
        let query_result = query.evaluate(dataset.as_ref(), metrics.as_ref()).await?;
        let IndexExprResult::Exact(mut row_addr_mask) = query_result else {
            todo!("Support for non-exact query results as input for merge_insert")
        };

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
            .load_scalar_index(IndexCriteria::default().with_name(&index_name))
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

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        Ok(Statistics::new_unknown(&MATERIALIZE_INDEX_SCHEMA))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{any::Any, collections::HashMap, ops::Bound, sync::Arc};

    use crate::index::DatasetIndexExt;
    use arrow::datatypes::UInt64Type;
    use async_trait::async_trait;
    use datafusion::{
        execution::TaskContext, physical_plan::ExecutionPlan, prelude::SessionConfig,
        scalar::ScalarValue,
    };
    use deepsize::DeepSizeOf;
    use futures::TryStreamExt;
    use lance_core::utils::mask::{RowAddrTreeMap, RowSetOps};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::gen_batch;
    use lance_index::{
        Index, IndexType,
        metrics::{MetricsCollector, NoOpMetricsCollector},
        scalar::{
            CreatedIndex, SargableQuery, ScalarIndex, ScalarIndexParams, SearchResult,
            UpdateCriteria,
            expression::{ScalarIndexExpr, ScalarIndexSearch},
            registry::{TrainingCriteria, TrainingOrdering},
        },
    };
    use roaring::RoaringBitmap;
    use serde_json::json;

    use crate::{
        Dataset, Error, Result,
        io::exec::scalar_index::MaterializeIndexExec,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, NoContextTestFixture},
    };

    use super::{LogicalScalarIndex, MapIndexExec, ScalarIndexExec};

    #[derive(Debug, DeepSizeOf)]
    struct StubScalarIndex {
        row_addrs: RowAddrTreeMap,
        index_type: IndexType,
    }

    #[async_trait]
    impl Index for StubScalarIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn lance_index::vector::VectorIndex>> {
            Err(Error::invalid_input(
                "Stub scalar index cannot be used as a vector index".to_string(),
            ))
        }

        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(json!({ "num_rows": self.row_addrs.len() }))
        }

        async fn prewarm(&self) -> Result<()> {
            Ok(())
        }

        fn index_type(&self) -> IndexType {
            self.index_type
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(self
                .row_addrs
                .iter()
                .map(|(fragment_id, _)| *fragment_id)
                .collect())
        }
    }

    #[async_trait]
    impl ScalarIndex for StubScalarIndex {
        async fn search(
            &self,
            _query: &dyn lance_index::scalar::AnyQuery,
            _metrics: &dyn MetricsCollector,
        ) -> Result<SearchResult> {
            Ok(SearchResult::exact(self.row_addrs.clone()))
        }

        fn can_remap(&self) -> bool {
            false
        }

        async fn remap(
            &self,
            _mapping: &HashMap<u64, Option<u64>>,
            _dest_store: &dyn lance_index::scalar::IndexStore,
        ) -> Result<CreatedIndex> {
            Err(Error::not_supported(
                "Stub scalar index cannot be remapped".to_string(),
            ))
        }

        async fn update(
            &self,
            _new_data: datafusion::execution::SendableRecordBatchStream,
            _dest_store: &dyn lance_index::scalar::IndexStore,
            _old_data_filter: Option<lance_index::scalar::OldIndexDataFilter>,
        ) -> Result<CreatedIndex> {
            Err(Error::not_supported(
                "Stub scalar index cannot be updated".to_string(),
            ))
        }

        fn update_criteria(&self) -> UpdateCriteria {
            UpdateCriteria::requires_old_data(TrainingCriteria::new(TrainingOrdering::None))
        }

        fn derive_index_params(&self) -> Result<ScalarIndexParams> {
            Ok(ScalarIndexParams::for_builtin(
                self.index_type.try_into().unwrap(),
            ))
        }
    }

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
            needs_recheck: false,
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

    #[tokio::test]
    async fn test_logical_scalar_index_search_unions_all_segments() {
        let logical_index = LogicalScalarIndex::try_new(vec![
            Arc::new(StubScalarIndex {
                row_addrs: [0_u64, 1_u64, (1_u64 << 32) | 3].into_iter().collect(),
                index_type: IndexType::BTree,
            }) as Arc<dyn ScalarIndex>,
            Arc::new(StubScalarIndex {
                row_addrs: [(2_u64 << 32) | 5, (3_u64 << 32) | 7].into_iter().collect(),
                index_type: IndexType::BTree,
            }) as Arc<dyn ScalarIndex>,
        ])
        .unwrap();

        let result = logical_index
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap();
        let SearchResult::Exact(row_addrs) = result else {
            panic!("logical scalar index should preserve exact segment results");
        };
        let row_addrs = row_addrs.true_rows();

        for row_addr in [
            0_u64,
            1_u64,
            (1_u64 << 32) | 3,
            (2_u64 << 32) | 5,
            (3_u64 << 32) | 7,
        ] {
            assert!(
                row_addrs.contains(row_addr),
                "missing row address {row_addr}"
            );
        }
        assert_eq!(row_addrs.len(), Some(5));
    }
}
