// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use super::{
    LogicalCoverageGroup,
    utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter},
};
use crate::{
    Dataset,
    dataset::rowids::{load_row_id_sequence, load_row_id_sequences},
    index::{
        DatasetIndexExt, PreFilter, logical_index_coverage_is_current,
        prefilter::DatasetPreFilter,
        scalar_logical::{
            open_named_scalar_index, open_named_scalar_index_with_readers,
            scalar_index_fragment_bitmap,
        },
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
        metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
    scalar::ScalarValue,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{FutureExt, StreamExt, TryFutureExt, TryStreamExt, stream::BoxStream};
use lance_core::{Error, ROW_ID_FIELD, Result, utils::address::RowAddress};
use lance_datafusion::{
    chunker::break_stream,
    utils::{
        ExecutionPlanMetricsSetExt, SCALAR_INDEX_SEARCH_TIME_METRIC, SCALAR_INDEX_SER_TIME_METRIC,
    },
};
use lance_index::scalar::lance_format::OpenedIndexFile;
use lance_index::{
    metrics::MetricsCollector,
    scalar::{
        SargableQuery, ScalarIndex,
        expression::{ScalarIndexExpr, ScalarIndexLoader, ScalarIndexSearch},
    },
};
use lance_select::{
    IndexExprResult, NullableIndexExprResult, NullableRowAddrMask, NullableRowAddrSet, RowAddrMask,
    RowAddrSelection, RowAddrTreeMap, RowSetOps, result::IndexExprResultWireFormat,
};
use lance_table::format::Fragment;
use roaring::{RoaringBitmap, RoaringTreemap};
use tracing::{debug_span, instrument};
use uuid::Uuid;

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

    async fn row_addr_result_to_row_ids(
        &self,
        result: NullableIndexExprResult,
    ) -> Result<NullableIndexExprResult> {
        // V2.3 address-oriented scalar plugins are trained with logical row
        // addresses, which are already the row-id domain. Legacy stable-row-id
        // indices still store physical addresses and need translation.
        if !self.manifest.uses_stable_row_ids() || self.manifest.uses_stable_logical_row_addresses()
        {
            return Ok(result);
        }

        let NullableIndexExprResult { lower, upper, .. } = result;
        let lower = translate_addr_mask_to_row_ids(self, lower).await?;
        let upper = translate_addr_mask_to_row_ids(self, upper).await?;
        Ok(NullableIndexExprResult::new(lower, upper))
    }
}

#[derive(Clone)]
struct PlannedScalarIndexLoader {
    dataset: Arc<Dataset>,
    preopened: Arc<HashMap<Uuid, OpenedIndexFile>>,
}

impl PlannedScalarIndexLoader {
    fn new(dataset: Arc<Dataset>, groups: &[LogicalCoverageGroup]) -> Self {
        let mut preopened = HashMap::new();
        for group in groups {
            for index_id in group.index_ids() {
                if let Some(opened) = group.preopened_index_file(index_id) {
                    preopened.insert(*index_id, opened);
                }
            }
        }
        Self {
            dataset,
            preopened: Arc::new(preopened),
        }
    }
}

#[async_trait]
impl ScalarIndexLoader for PlannedScalarIndexLoader {
    async fn load_index(
        &self,
        column: &str,
        index_name: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        open_named_scalar_index_with_readers(
            self.dataset.as_ref(),
            column,
            index_name,
            metrics,
            self.preopened.as_ref(),
        )
        .await
    }

    async fn row_addr_result_to_row_ids(
        &self,
        result: NullableIndexExprResult,
    ) -> Result<NullableIndexExprResult> {
        <Dataset as ScalarIndexLoader>::row_addr_result_to_row_ids(self.dataset.as_ref(), result)
            .await
    }
}

/// Translate an address-domain [`NullableRowAddrMask`] into the row-id domain
///
/// Address-domain index results are always positive allow-lists (`AtMost`), so
/// a block-list here would mean a boolean op was applied before translation,
/// which is unsupported.
async fn translate_addr_mask_to_row_ids(
    dataset: &Dataset,
    mask: NullableRowAddrMask,
) -> Result<NullableRowAddrMask> {
    match mask {
        NullableRowAddrMask::AllowList(set) => Ok(NullableRowAddrMask::AllowList(
            translate_addr_set_to_row_ids(dataset, set).await?,
        )),
        NullableRowAddrMask::BlockList(_) => Err(Error::internal(
            "cannot translate a block-list address mask to the row-id domain",
        )),
    }
}

async fn translate_addr_set_to_row_ids(
    dataset: &Dataset,
    set: NullableRowAddrSet,
) -> Result<NullableRowAddrSet> {
    let selected = translate_addr_treemap_to_row_ids(dataset, set.selected_rows()).await?;
    let nulls = translate_addr_treemap_to_row_ids(dataset, set.null_rows()).await?;
    Ok(NullableRowAddrSet::new(selected, nulls))
}

/// Map a set of physical row addresses to their stable row ids
///
/// For each fragment present in `addrs`, the live rows in physical order carry
/// the stable ids yielded by the fragment's [`RowIdSequence`] in the same
/// order. Zipping the two (skipping deleted physical offsets) gives the
/// `physical offset -> stable id` mapping. Addresses that point at deleted rows
/// have no live counterpart and are dropped, which is correct: those rows are
/// not part of the answer.
async fn translate_addr_treemap_to_row_ids(
    dataset: &Dataset,
    addrs: &RowAddrTreeMap,
) -> Result<RowAddrTreeMap> {
    let mut row_ids = RowAddrTreeMap::new();
    for (fragment_id, selection) in addrs.iter() {
        let file_fragment = dataset.get_fragment(*fragment_id as usize).ok_or_else(|| {
            Error::internal(format!(
                "fragment {fragment_id} referenced by an address-domain index result \
                 was not found in the dataset"
            ))
        })?;
        let sequence = load_row_id_sequence(dataset, file_fragment.metadata()).await?;

        match selection {
            RowAddrSelection::Full => {
                // The whole fragment is selected: every live row's id qualifies.
                row_ids |= RowAddrTreeMap::from(sequence.as_ref());
            }
            RowAddrSelection::Partial(offsets) => {
                let Some(max_offset) = offsets.max() else {
                    continue;
                };
                let (deletion_vector, num_physical_rows) = futures::try_join!(
                    file_fragment.get_deletion_vector(),
                    file_fragment.physical_rows()
                )?;
                let num_physical_rows = num_physical_rows as u32;
                let mut ids = sequence.iter();
                for physical_offset in 0..num_physical_rows {
                    if physical_offset > max_offset {
                        break;
                    }
                    let deleted = deletion_vector
                        .as_ref()
                        .is_some_and(|dv| dv.contains(physical_offset));
                    if deleted {
                        continue;
                    }
                    match ids.next() {
                        Some(id) => {
                            if offsets.contains(physical_offset) {
                                row_ids.insert(id);
                            }
                        }
                        None => break,
                    }
                }
            }
        }
    }
    Ok(row_ids)
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
    logical_coverage_groups: Option<Arc<Vec<LogicalCoverageGroup>>>,
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
            expr: expr.optimize(),
            logical_coverage_groups: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            result_format,
        }
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub(crate) fn new_with_logical_coverage(
        dataset: Arc<Dataset>,
        expr: ScalarIndexExpr,
        result_format: IndexExprResultWireFormat,
        logical_coverage_groups: Vec<LogicalCoverageGroup>,
    ) -> Self {
        let mut exec = Self::new(dataset, expr, result_format);
        exec.logical_coverage_groups = Some(Arc::new(logical_coverage_groups));
        exec
    }

    pub(crate) fn logical_coverage_groups(&self) -> Option<&Arc<Vec<LogicalCoverageGroup>>> {
        self.logical_coverage_groups.as_ref()
    }

    /// The parsed scalar-index expression this node will evaluate.
    pub fn expr(&self) -> &ScalarIndexExpr {
        &self.expr
    }

    /// Return the wire format used when serializing this exec's
    /// [`IndexExprResult`] output.
    pub fn result_format(&self) -> IndexExprResultWireFormat {
        self.result_format
    }

    #[async_recursion]
    pub async fn fragments_covered_by_index_query(
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

    fn logical_fragments_covered_by_index_query(
        index_expr: &ScalarIndexExpr,
        groups: &[LogicalCoverageGroup],
    ) -> Result<RoaringBitmap> {
        fn visit<'a>(
            index_expr: &ScalarIndexExpr,
            groups: &mut std::slice::Iter<'a, LogicalCoverageGroup>,
        ) -> Result<RoaringBitmap> {
            match index_expr {
                ScalarIndexExpr::And(left, right) | ScalarIndexExpr::Or(left, right) => {
                    Ok(visit(left, groups)? & visit(right, groups)?)
                }
                ScalarIndexExpr::Not(inner) => visit(inner, groups),
                ScalarIndexExpr::Query(_) => groups
                    .next()
                    .map(|group| group.fully_covered_logical_fragments().clone())
                    .ok_or_else(|| {
                        Error::internal(
                            "scalar index expression has more leaves than logical coverage groups",
                        )
                    }),
            }
        }

        let mut groups = groups.iter();
        let covered = visit(index_expr, &mut groups)?;
        if groups.next().is_some() {
            return Err(Error::internal(
                "scalar index expression has fewer leaves than logical coverage groups",
            ));
        }
        Ok(covered)
    }

    async fn do_execute(
        expr: ScalarIndexExpr,
        dataset: Arc<Dataset>,
        logical_coverage_groups: Option<Arc<Vec<LogicalCoverageGroup>>>,
        plan_metrics: ExecutionPlanMetricsSet,
        result_format: IndexExprResultWireFormat,
    ) -> Result<RecordBatch> {
        let metrics = IndexMetrics::new(&plan_metrics, 0);
        let query_result = {
            let search_time = plan_metrics.new_time(SCALAR_INDEX_SEARCH_TIME_METRIC, 0);
            let _timer = search_time.timer();
            if let Some(groups) = logical_coverage_groups.as_deref() {
                let loader = PlannedScalarIndexLoader::new(dataset.clone(), groups);
                expr.evaluate(&loader, &metrics).await?
            } else {
                expr.evaluate(dataset.as_ref(), &metrics).await?
            }
        };
        let fragments_covered_by_result = if let Some(groups) = logical_coverage_groups.as_deref() {
            Self::logical_fragments_covered_by_index_query(&expr, groups)?
        } else {
            Self::fragments_covered_by_index_query(&expr, dataset.as_ref()).await?
        };
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
            self.logical_coverage_groups.clone(),
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

/// A single scalar-index lookup used by [`MapIndexExec`].
///
/// `column` identifies a column whose values will be probed against the
/// index named `index_name`. Multiple lookups are intersected with logical
/// AND semantics inside `MapIndexExec`.
#[derive(Debug, Clone)]
pub struct IndexLookup {
    pub column: String,
    pub index_name: String,
}

impl IndexLookup {
    pub fn new(column: impl Into<String>, index_name: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            index_name: index_name.into(),
        }
    }
}

/// An execution node that translates index values into row addresses
///
/// This can be combined with TakeExec to perform an "indexed take".
///
/// Multiple `(column, index_name)` lookups can be supplied: the operator
/// expects one input column per lookup (in matching order) and emits the
/// row addresses where every column's value is present in its respective
/// index — that is, the AND of the per-column index probes. This lets a
/// composite-key join trim the candidate row set with every available
/// scalar index before the downstream take.
#[derive(Debug)]
pub struct MapIndexExec {
    dataset: Arc<Dataset>,
    lookups: Vec<IndexLookup>,
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
                write!(f, "IndexedLookup")?;
                if self.lookups.len() > 1 {
                    let cols = self
                        .lookups
                        .iter()
                        .map(|l| l.column.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(f, " [{cols}]")?;
                }
                Ok(())
            }
        }
    }
}

impl MapIndexExec {
    /// Convenience constructor for the common single-column case.
    pub fn new(
        dataset: Arc<Dataset>,
        column_name: String,
        index_name: String,
        input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        Self::new_multi(
            dataset,
            vec![IndexLookup::new(column_name, index_name)],
            input,
        )
    }

    /// Build a `MapIndexExec` that probes one or more scalar indices and
    /// emits the AND of their results. `lookups` must be non-empty and
    /// `input` must produce one column per lookup, in the same order.
    pub fn new_multi(
        dataset: Arc<Dataset>,
        lookups: Vec<IndexLookup>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        debug_assert!(
            !lookups.is_empty(),
            "MapIndexExec requires at least one index lookup"
        );
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(INDEX_LOOKUP_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            lookups,
            input,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    async fn build_stream(
        input: datafusion::physical_plan::SendableRecordBatchStream,
        partition: usize,
        dataset: Arc<Dataset>,
        lookups: Vec<IndexLookup>,
        index_metrics: Arc<IndexMetrics>,
        metrics_set: ExecutionPlanMetricsSet,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        // A row can be found by the composite probe only if it lives in a
        // fragment covered by *every* index in `lookups`; restrict the
        // deletion mask to that intersection so we only filter deletes we
        // could actually see.
        let mut fragment_bitmap: Option<RoaringBitmap> = None;
        for lookup in &lookups {
            let bm = scalar_index_fragment_bitmap(&dataset, &lookup.column, &lookup.index_name)
                .await?
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "IndexedLookupExec: index '{}' on column '{}' disappeared after planning",
                        lookup.index_name, lookup.column,
                    ))
                })?;
            fragment_bitmap = Some(match fragment_bitmap {
                None => bm,
                Some(acc) => acc & bm,
            });
        }
        let fragment_bitmap = fragment_bitmap.expect("MapIndexExec built with no lookups");
        let deletion_mask = if dataset.manifest.uses_stable_logical_row_addresses() {
            let mut index_metadata = Vec::new();
            for lookup in &lookups {
                for index in dataset.load_indices_by_name(&lookup.index_name).await? {
                    if logical_index_coverage_is_current(&dataset, &index)? {
                        index_metadata.push(index);
                    }
                }
            }
            let prefilter = DatasetPreFilter::new(dataset.clone(), &index_metadata, None);
            prefilter.wait_for_ready().await?;
            Some(prefilter.mask())
        } else if let Some(fut) =
            DatasetPreFilter::create_restricted_deletion_mask(dataset.clone(), fragment_bitmap)
        {
            Some(fut.await?)
        } else {
            None
        };

        let baseline = BaselineMetrics::new(&metrics_set, partition);
        let elapsed_compute = baseline.elapsed_compute().clone();
        let stream = input.then(move |batch_result| {
            let lookups = lookups.clone();
            let dataset = dataset.clone();
            let deletion_mask = deletion_mask.clone();
            let metrics = index_metrics.clone();
            let elapsed_compute = elapsed_compute.clone();
            async move {
                let batch = batch_result?;
                // Timer spans `map_batch`'s `.await` on purpose: that await is
                // the per-batch sargable index evaluation, which is the work
                // we want attributed here.
                let _t = elapsed_compute.timer();
                Self::map_batch(lookups, dataset, deletion_mask, batch, metrics).await
            }
        });
        let stream = stream.map(move |batch| {
            let poll = baseline.record_poll(std::task::Poll::Ready(Some(batch)));
            match poll {
                std::task::Poll::Ready(Some(b)) => b,
                _ => unreachable!("record_poll preserves Ready(Some) input"),
            }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            INDEX_LOOKUP_SCHEMA.clone(),
            stream,
        )))
    }

    /// Build the AND-of-IsIn `ScalarIndexExpr` describing this batch's
    /// composite lookup: each input column contributes one `IsIn` query
    /// against its matching index.
    fn build_query(
        lookups: &[IndexLookup],
        batch: &RecordBatch,
    ) -> datafusion::error::Result<ScalarIndexExpr> {
        let per_column = lookups.iter().enumerate().map(|(idx, lookup)| {
            let column = batch.column(idx);
            let values = (0..column.len())
                .map(|row| ScalarValue::try_from_array(column, row))
                .collect::<datafusion::error::Result<Vec<_>>>()?;
            Ok::<_, datafusion::error::DataFusionError>(ScalarIndexExpr::Query(ScalarIndexSearch {
                column: lookup.column.clone(),
                index_name: lookup.index_name.clone(),
                // Internal IndexedLookup-style query — type is unknown at this layer
                index_type: String::new(),
                query: Arc::new(SargableQuery::IsIn(values)),
                needs_recheck: false,
                fragment_bitmap: None,
            }))
        });

        per_column
            .reduce(|lhs, rhs| Ok(ScalarIndexExpr::And(Box::new(lhs?), Box::new(rhs?))))
            .expect("MapIndexExec built with no lookups")
    }

    async fn map_batch(
        lookups: Vec<IndexLookup>,
        dataset: Arc<Dataset>,
        deletion_mask: Option<Arc<RowAddrMask>>,
        batch: RecordBatch,
        metrics: Arc<IndexMetrics>,
    ) -> datafusion::error::Result<RecordBatch> {
        let query = Self::build_query(&lookups, &batch)?;
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
            Ok(Arc::new(Self::new_multi(
                self.dataset.clone(),
                self.lookups.clone(),
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
            self.lookups.clone(),
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
    logical_coverage_groups: Option<Arc<Vec<LogicalCoverageGroup>>>,
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

pub(crate) fn logical_coverage_effective_rows(
    coverage_groups: &[LogicalCoverageGroup],
) -> Result<Option<RoaringTreemap>> {
    let mut effective: Option<RoaringTreemap> = None;
    for group in coverage_groups {
        if let Some(group_rows) = group.effective_rows()? {
            effective = Some(match effective {
                Some(rows) => rows & group_rows,
                None => group_rows,
            });
        }
    }
    Ok(effective)
}

pub(crate) fn logical_liveness_scope(
    dataset: &Dataset,
    coverage_groups: &[LogicalCoverageGroup],
) -> Result<(RoaringBitmap, Vec<uuid::Uuid>)> {
    let index_ids = coverage_groups
        .iter()
        .flat_map(|group| group.index_ids().iter().copied())
        .collect::<Vec<_>>();
    let Some(effective) = logical_coverage_effective_rows(coverage_groups)? else {
        return Ok((dataset.fragment_bitmap.as_ref().clone(), index_ids));
    };
    let router = dataset.row_address_router()?;
    let mut physical_coverage = RoaringBitmap::new();
    for (logical_fragment_id, slots) in effective.bitmaps() {
        physical_coverage |=
            router.logical_selection_destination_fragments(logical_fragment_id, slots)?;
    }
    Ok((physical_coverage, index_ids))
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
            logical_coverage_groups: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub(crate) fn new_with_logical_coverage(
        dataset: Arc<Dataset>,
        expr: ScalarIndexExpr,
        fragments: Arc<Vec<Fragment>>,
        logical_coverage_groups: Option<Vec<LogicalCoverageGroup>>,
    ) -> Self {
        let mut exec = Self::new(dataset, expr, fragments);
        exec.logical_coverage_groups = logical_coverage_groups.map(Arc::new);
        exec
    }

    #[instrument(name = "materialize_scalar_index", skip_all, level = "debug")]
    async fn do_execute(
        expr: ScalarIndexExpr,
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        planned_logical_coverage_groups: Option<Arc<Vec<LogicalCoverageGroup>>>,
        metrics: Arc<IndexMetrics>,
    ) -> Result<RecordBatch> {
        let span = debug_span!("create_prefilter");
        let uses_logical_addresses = dataset.manifest.uses_stable_logical_row_addresses();
        let prefilter = if uses_logical_addresses {
            None
        } else {
            let fragment_bitmap =
                RoaringBitmap::from_iter(fragments.iter().map(|frag| frag.id as u32));
            // The user-requested `fragments` is guaranteed to be stricter than the index's fragment
            // bitmap.  This node only runs on indexed fragments and any fragments that were deleted
            // when the index was trained will still be deleted when the index is queried.
            span.in_scope(|| {
                DatasetPreFilter::create_deletion_mask(dataset.clone(), fragment_bitmap)
            })
        };
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
        let (mask, logical_coverage_groups) = if uses_logical_addresses {
            let coverage_groups = match planned_logical_coverage_groups {
                Some(groups) => groups.as_ref().clone(),
                None => dataset.scan().logical_coverage_groups(&expr).await?,
            };
            let loader = PlannedScalarIndexLoader::new(dataset.clone(), &coverage_groups);
            let expr_result = expr.evaluate(&loader, metrics.as_ref());
            let (physical_coverage, index_ids) =
                logical_liveness_scope(dataset.as_ref(), &coverage_groups)?;
            let logical_coverage = logical_coverage_effective_rows(&coverage_groups)?;
            let liveness = span.in_scope(|| {
                DatasetPreFilter::do_create_logical_liveness_mask_for_physical_coverage(
                    dataset.clone(),
                    physical_coverage,
                    logical_coverage,
                    index_ids,
                )
                .boxed()
            });
            let (expr_result, prefilter) = futures::try_join!(expr_result, liveness)?;
            (
                take_upper(expr_result)? & (*prefilter).clone(),
                Some(coverage_groups),
            )
        } else if let Some(prefilter) = prefilter {
            let expr_result = expr.evaluate(dataset.as_ref(), metrics.as_ref());
            let (expr_result, prefilter) = futures::try_join!(expr_result, prefilter)?;
            (take_upper(expr_result)? & (*prefilter).clone(), None)
        } else {
            (
                take_upper(expr.evaluate(dataset.as_ref(), metrics.as_ref()).await?)?,
                None,
            )
        };
        let ids = row_ids_for_mask(
            mask,
            &dataset,
            &fragments,
            logical_coverage_groups.as_deref(),
        )
        .await?;
        let ids = UInt64Array::from(ids);
        Ok(RecordBatch::try_new(
            MATERIALIZE_INDEX_SCHEMA.clone(),
            vec![Arc::new(ids)],
        )?)
    }
}

async fn logical_row_ids_for_mask(
    mask: &RowAddrMask,
    dataset: &Dataset,
    fragments: &[Fragment],
    coverage_groups: &[LogicalCoverageGroup],
) -> Result<Vec<u64>> {
    fn covered_by_all(groups: &[LogicalCoverageGroup], row_id: u64) -> Result<bool> {
        if groups.is_empty() {
            return Ok(false);
        }
        for group in groups {
            if !group.contains_effective(row_id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    const RESOLVE_BATCH_SIZE: usize = 64 * 1024;
    let target_fragments = fragments
        .iter()
        .map(|fragment| u32::try_from(fragment.id))
        .collect::<std::result::Result<RoaringBitmap, _>>()
        .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;

    // Positive index results are normally finite. Resolve only those logical
    // candidates forward and keep rows whose current physical owner is in the
    // scan's target fragments.
    if let RowAddrMask::AllowList(allow_list) = mask
        && let Some(row_addrs) = allow_list.row_addrs()
    {
        let mut candidates = Vec::new();
        for row_id in row_addrs.map(u64::from) {
            if covered_by_all(coverage_groups, row_id)? {
                candidates.push(row_id);
            }
        }
        let mut retained = Vec::with_capacity(candidates.len());
        for candidate_batch in candidates.chunks(RESOLVE_BATCH_SIZE) {
            let physical = dataset
                .resolve_logical_row_ids_async(candidate_batch)
                .await?;
            retained.extend(candidate_batch.iter().zip(physical).filter_map(
                |(logical, physical)| {
                    physical
                        .filter(|physical| target_fragments.contains(physical.fragment_id()))
                        .map(|_| *logical)
                },
            ));
        }
        return Ok(retained);
    }

    // NOT / match-all can be represented as a block list. Enumerate only the
    // requested physical fragments and verify each inverse candidate is the
    // logical identity's current owner. This is proportional to the target
    // scan, and never builds a table-wide reverse index.
    let mut row_ids = Vec::new();
    for fragment in fragments {
        let fragment_id = u32::try_from(fragment.id)
            .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;
        let physical_rows = u32::try_from(
            fragment
                .physical_rows
                .ok_or_else(|| Error::internal("fragment is missing physical row count"))?,
        )
        .map_err(|_| Error::invalid_input("physical fragment row count exceeds u32"))?;
        for start in (0..physical_rows).step_by(RESOLVE_BATCH_SIZE) {
            let end = physical_rows.min(start.saturating_add(RESOLVE_BATCH_SIZE as u32));
            let physical = (start..end)
                .map(|offset| RowAddress::new_from_parts(fragment_id, offset))
                .collect::<Vec<_>>();
            for logical in dataset
                .resolve_current_physical_row_ids_async(&physical)
                .await?
                .into_iter()
                .flatten()
                .map(|logical| logical.raw())
            {
                if mask.selected(logical) && covered_by_all(coverage_groups, logical)? {
                    row_ids.push(logical);
                }
            }
        }
    }
    Ok(row_ids)
}

#[instrument(name = "make_row_ids", skip(mask, dataset, fragments))]
async fn row_ids_for_mask(
    mask: RowAddrMask,
    dataset: &Dataset,
    fragments: &[Fragment],
    logical_coverage_groups: Option<&[LogicalCoverageGroup]>,
) -> Result<Vec<u64>> {
    if dataset.manifest.uses_stable_logical_row_addresses() {
        return logical_row_ids_for_mask(
            &mask,
            dataset,
            fragments,
            logical_coverage_groups.unwrap_or_default(),
        )
        .await;
    }
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
            self.logical_coverage_groups.clone(),
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
    use arrow::record_batch::RecordBatchIterator;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::Schema;
    use datafusion::{
        execution::TaskContext, physical_plan::ExecutionPlan, prelude::SessionConfig,
        scalar::ScalarValue,
    };
    use futures::TryStreamExt;
    use lance_core::utils::{address::RowAddress, tempfile::TempStrDir};
    use lance_datagen::gen_batch;
    use lance_index::{
        IndexType,
        scalar::{
            SargableQuery, ScalarIndexParams,
            expression::{ScalarIndexExpr, ScalarIndexSearch},
        },
    };
    use lance_select::{RowAddrTreeMap, result::IndexExprResultWireFormat};

    use crate::{
        Dataset,
        dataset::WriteParams,
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
            fragment_bitmap: None,
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

    #[tokio::test]
    async fn test_translate_addr_treemap_to_stable_row_ids() {
        let test_dir = TempStrDir::default();
        let batch = RecordBatch::try_from_iter(vec![(
            "id",
            Arc::new(Int32Array::from((0..10).collect::<Vec<_>>())) as ArrayRef,
        )])
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 5,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_dir.as_str(), Some(write_params))
            .await
            .unwrap();
        let fragment_id = dataset.get_fragments()[1].id() as u32;

        let mut full_fragment = RowAddrTreeMap::new();
        full_fragment.insert_fragment(fragment_id);
        let translated = super::translate_addr_treemap_to_row_ids(&dataset, &full_fragment)
            .await
            .unwrap();
        let row_ids = translated
            .get_fragment_bitmap(0)
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(row_ids, vec![5, 6, 7, 8, 9]);

        let mut partial_fragment = RowAddrTreeMap::new();
        partial_fragment.insert(RowAddress::new_from_parts(fragment_id, 1).into());
        partial_fragment.insert(RowAddress::new_from_parts(fragment_id, 3).into());
        let translated = super::translate_addr_treemap_to_row_ids(&dataset, &partial_fragment)
            .await
            .unwrap();
        let row_ids = translated
            .get_fragment_bitmap(0)
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(row_ids, vec![6, 8]);
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
            fragment_bitmap: None,
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
            fragment_bitmap: None,
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
