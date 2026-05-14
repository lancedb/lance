// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execute-time half of aggregate pushdown.
//!
//! [`AggregateIndexSearchExec`] computes partial aggregate state for one or
//! more aggregates by probing scalar indices, without scanning column data.
//! Its output schema matches what `AggregateExec(AggregateMode::Partial)`
//! would produce for the same aggregates, so a downstream `AggregateExec`
//! in `Final`/`FinalPartitioned` mode can combine us unchanged.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Int64Array, RecordBatch};
use arrow_schema::{Schema, SchemaRef};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    execution_plan::{Boundedness, EmissionType},
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::mask::{NullableRowAddrSet, RowAddrMask, RowAddrSelection, RowAddrTreeMap};
use lance_core::{Error, Result};
use lance_datafusion::utils::{ExecutionPlanMetricsSetExt, SCALAR_INDEX_SEARCH_TIME_METRIC};
use lance_index::expression::aggregate::{AggregateIndexSearch, CountQuery};
use lance_index::scalar::{ScalarIndex, SearchResult};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use tracing::instrument;

use super::utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter};
use crate::Dataset;
use crate::index::DatasetIndexExt;
use crate::index::prefilter::DatasetPreFilter;
use crate::index::scalar_logical::{open_named_scalar_index, scalar_index_fragment_bitmap};

/// An execution node that answers a set of aggregates from scalar indices.
///
/// The node returns a single record batch whose schema is the concatenation
/// of `state_fields()` for each aggregate in `aggregate_funcs`.
///
/// It optionally has a single child [`super::scalar_index::ScalarIndexExec`]
/// whose output is used as a prefilter for each aggregate.
#[derive(Debug)]
pub struct AggregateIndexSearchExec {
    dataset: Arc<Dataset>,
    aggregates: Vec<Arc<AggregateIndexSearch>>,
    aggregate_funcs: Vec<Arc<AggregateFunctionExpr>>,
    prefilter_input: Option<Arc<dyn ExecutionPlan>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for AggregateIndexSearchExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let names = self
            .aggregates
            .iter()
            .map(|agg| agg.to_string())
            .collect::<Vec<_>>()
            .join(",");
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "AggregateIndexSearch: aggs=[{}]", names)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "AggregateIndexSearch\naggs=[{}]", names)
            }
        }
    }
}

impl AggregateIndexSearchExec {
    /// Build a new node.
    ///
    /// `aggregates` and `aggregate_funcs` must have the same length — each
    /// aggregate query is paired with its DataFusion partial-state spec.
    /// `prefilter_input`, if present, must produce a single batch in the
    /// scalar-index result schema; that mask is intersected with the
    /// aggregate's natural fragment coverage and the active deletion mask.
    pub fn try_new(
        dataset: Arc<Dataset>,
        aggregates: Vec<Arc<AggregateIndexSearch>>,
        aggregate_funcs: Vec<Arc<AggregateFunctionExpr>>,
        prefilter_input: Option<Arc<dyn ExecutionPlan>>,
    ) -> Result<Self> {
        if aggregates.len() != aggregate_funcs.len() {
            return Err(Error::invalid_input(format!(
                "AggregateIndexSearchExec: aggregates ({}) and aggregate_funcs ({}) length mismatch",
                aggregates.len(),
                aggregate_funcs.len()
            )));
        }

        for agg in &aggregates {
            if agg.index_name.is_none() {
                // The only aggregate we can answer without an associated index
                // is a non-distinct COUNT.
                let count = agg
                    .query
                    .as_any()
                    .downcast_ref::<CountQuery>()
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "AggregateIndexSearchExec: aggregate {} has no associated index but is not a count",
                            agg
                        ))
                    })?;
                if count.is_distinct() {
                    return Err(Error::invalid_input(format!(
                        "AggregateIndexSearchExec: aggregate {} has no associated index but is a distinct count",
                        agg
                    )));
                }
            }
        }

        let state_fields = aggregate_funcs
            .iter()
            .map(|agg| agg.state_fields())
            .collect::<datafusion::error::Result<Vec<_>>>()
            .map_err(|e| Error::invalid_input(e.to_string()))?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let state_fields_owned: Vec<arrow_schema::Field> =
            state_fields.iter().map(|f| f.as_ref().clone()).collect();
        let schema: SchemaRef = Arc::new(Schema::new(state_fields_owned));

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Ok(Self {
            dataset,
            aggregates,
            aggregate_funcs,
            prefilter_input,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Drain `prefilter_input` (a [`super::scalar_index::ScalarIndexExec`]) to
    /// produce the row-address mask it serialized.
    async fn load_prefilter(
        prefilter_input: Arc<dyn ExecutionPlan>,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<RowAddrMask> {
        let mut stream = prefilter_input.execute(0, context).map_err(Error::from)?;
        let batch = stream
            .try_next()
            .await
            .map_err(Error::from)?
            .ok_or_else(|| {
                Error::internal(
                    "AggregateIndexSearchExec: prefilter input produced no batches".to_string(),
                )
            })?;
        // Drain any remaining batches so the upstream sees a clean shutdown.
        while stream.try_next().await.map_err(Error::from)?.is_some() {}

        let result_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "AggregateIndexSearchExec: prefilter result column has type {:?}, expected Binary",
                    batch.column(0).data_type()
                ))
            })?;
        RowAddrMask::from_arrow(result_col)
    }

    /// Look up the column name an index lives on by inspecting manifest metadata.
    async fn column_for_index(dataset: &Dataset, index_name: &str) -> Result<String> {
        let indices = dataset.load_indices_by_name(index_name).await?;
        let index = indices.into_iter().next().ok_or_else(|| {
            Error::internal(format!(
                "AggregateIndexSearchExec: no index named '{}' found",
                index_name
            ))
        })?;
        let field_id = *index.fields.first().ok_or_else(|| {
            Error::internal(format!(
                "AggregateIndexSearchExec: index '{}' has no field bindings",
                index_name
            ))
        })?;
        let field = dataset.schema().field_by_id(field_id).ok_or_else(|| {
            Error::internal(format!(
                "AggregateIndexSearchExec: index '{}' references unknown field id {}",
                index_name, field_id
            ))
        })?;
        Ok(field.name.clone())
    }

    /// Load every backing index referenced by the aggregates and the fragment
    /// bitmap each one covers.
    ///
    /// The returned vectors are aligned with `aggregates`: aggregates without
    /// an `index_name` produce `None` in `indices` and contribute no fragment
    /// bitmap to the intersection.
    async fn load_indices(
        dataset: Arc<Dataset>,
        aggregates: Vec<Arc<AggregateIndexSearch>>,
        index_metrics: IndexMetrics,
    ) -> Result<(Vec<Option<Arc<dyn ScalarIndex>>>, Option<RoaringBitmap>)> {
        let mut indices = Vec::with_capacity(aggregates.len());
        let mut fragments_intersection: Option<RoaringBitmap> = None;
        for agg in &aggregates {
            match &agg.index_name {
                None => indices.push(None),
                Some(index_name) => {
                    let column = Self::column_for_index(&dataset, index_name).await?;
                    let bitmap = scalar_index_fragment_bitmap(&dataset, &column, index_name)
                        .await?
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "AggregateIndexSearchExec: index '{}' has no fragment bitmap",
                                index_name
                            ))
                        })?;
                    fragments_intersection = Some(match fragments_intersection.take() {
                        None => bitmap,
                        Some(existing) => existing & bitmap,
                    });
                    let index =
                        open_named_scalar_index(&dataset, &column, index_name, &index_metrics)
                            .await?;
                    indices.push(Some(index));
                }
            }
        }
        Ok((indices, fragments_intersection))
    }

    /// Apply the user's algorithm to fold the prefilter, fragment allow list,
    /// and deletion mask into a single [`RowAddrMask`].
    ///
    /// The result is always an `AllowList` so it can be wrapped in a
    /// [`SearchResult::Exact`] for [`ScalarIndex::calculate_aggregate`].
    fn combine_masks(
        fragments_allow: RowAddrTreeMap,
        prefilter: Option<RowAddrMask>,
        deletion_mask: Option<Arc<RowAddrMask>>,
    ) -> RowAddrMask {
        let base = RowAddrMask::AllowList(fragments_allow);
        let after_prefilter = match prefilter {
            None => base,
            Some(prefilter) => base & prefilter,
        };
        match deletion_mask {
            None => after_prefilter,
            Some(deletion_mask) => after_prefilter & (*deletion_mask).clone(),
        }
    }

    /// Count the rows selected by `mask`, looking up `Full`-marker fragments
    /// in the manifest so we never need to materialize a `RoaringBitmap::full()`.
    fn count_from_mask(mask: &RowAddrMask, dataset: &Dataset) -> Result<i64> {
        let allow = mask.allow_list().ok_or_else(|| {
            Error::internal(
                "AggregateIndexSearchExec: combined mask is not an AllowList".to_string(),
            )
        })?;
        let frag_map: HashMap<u32, &Fragment> = dataset
            .fragments()
            .iter()
            .map(|f| (f.id as u32, f))
            .collect();
        let mut count = 0i64;
        for (frag_id, sel) in allow.iter() {
            match sel {
                RowAddrSelection::Full => {
                    // The fragment is in the allow list with no deletions
                    // touching it — its row count is the physical row count.
                    let frag = frag_map.get(frag_id).ok_or_else(|| {
                        Error::internal(format!(
                            "AggregateIndexSearchExec: fragment {} not found in manifest",
                            frag_id
                        ))
                    })?;
                    let n = frag.physical_rows.ok_or_else(|| {
                        Error::internal(format!(
                            "AggregateIndexSearchExec: physical_rows missing for fragment {}",
                            frag_id
                        ))
                    })?;
                    count += n as i64;
                }
                RowAddrSelection::Partial(bitmap) => {
                    count += bitmap.len() as i64;
                }
            }
        }
        Ok(count)
    }

    #[instrument(name = "aggregate_index_search", skip_all, level = "debug")]
    async fn do_execute(
        dataset: Arc<Dataset>,
        aggregates: Vec<Arc<AggregateIndexSearch>>,
        prefilter_input: Option<Arc<dyn ExecutionPlan>>,
        context: Arc<datafusion::execution::context::TaskContext>,
        plan_metrics: ExecutionPlanMetricsSet,
        schema: SchemaRef,
    ) -> Result<RecordBatch> {
        let index_metrics = IndexMetrics::new(&plan_metrics, 0);

        // Kick off the prefilter load and index loads in parallel.
        let prefilter_fut = async {
            match prefilter_input {
                None => Ok::<Option<RowAddrMask>, Error>(None),
                Some(input) => Self::load_prefilter(input, context.clone()).await.map(Some),
            }
        };
        let indices_fut = async {
            let timer = plan_metrics.new_time(SCALAR_INDEX_SEARCH_TIME_METRIC, 0);
            let _guard = timer.timer();
            Self::load_indices(dataset.clone(), aggregates.clone(), index_metrics.clone()).await
        };
        let (prefilter, (loaded_indices, fragments_intersection)) =
            futures::try_join!(prefilter_fut, indices_fut)?;

        // Fall back to all dataset fragments when no aggregate has an index —
        // we still need a set of fragments to anchor the deletion mask against.
        let fragments_covered = fragments_intersection.unwrap_or_else(|| {
            dataset
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .collect::<RoaringBitmap>()
        });

        // Build the fragments allow list as concrete `[0..physical_rows)`
        // ranges rather than `Full` markers. `Full` interacts poorly with
        // `BlockList` subtraction — `RowAddrTreeMap::Sub` materializes a
        // `RoaringBitmap::full()` (2^32 rows) per fragment when a `Full` entry
        // gets a partial block subtracted from it, which inflates counts and
        // is expensive. Concrete ranges avoid that path entirely and keep
        // `len()` exact at every combine step.
        let frag_map: HashMap<u32, &Fragment> = dataset
            .fragments()
            .iter()
            .map(|f| (f.id as u32, f))
            .collect();
        let mut fragments_allow = RowAddrTreeMap::new();
        for frag_id in fragments_covered.iter() {
            let frag = frag_map.get(&frag_id).ok_or_else(|| {
                Error::internal(format!(
                    "AggregateIndexSearchExec: fragment {} not in manifest",
                    frag_id
                ))
            })?;
            let physical = frag.physical_rows.ok_or_else(|| {
                Error::internal(format!(
                    "AggregateIndexSearchExec: physical_rows missing for fragment {}",
                    frag_id
                ))
            })?;
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert_range(0u32..(physical as u32));
            fragments_allow.insert_bitmap(frag_id, bitmap);
        }

        // Load the deletion mask for the covered fragments.
        let deletion_mask =
            match DatasetPreFilter::create_deletion_mask(dataset.clone(), fragments_covered) {
                Some(fut) => Some(fut.await?),
                None => None,
            };

        // Combine prefilter ∩ fragment-allow − deletion into a single AllowList.
        let combined = Self::combine_masks(fragments_allow, prefilter, deletion_mask);

        // Compute partial state, one aggregate at a time.
        let total_rows = dataset.count_all_rows().await? as u64;
        let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(aggregates.len());
        for (agg, index) in aggregates.iter().zip(loaded_indices.iter()) {
            match index {
                Some(index) => {
                    let allow_list = combined.allow_list().cloned().unwrap_or_default();
                    let search_result = SearchResult::Exact(NullableRowAddrSet::new(
                        allow_list,
                        RowAddrTreeMap::new(),
                    ));
                    let scalar = index
                        .calculate_aggregate(
                            agg.query.as_ref(),
                            Some(search_result),
                            total_rows,
                            &index_metrics,
                        )
                        .await?;
                    arrays.push(scalar.as_array().clone());
                }
                None => {
                    // Validated in `try_new`: this can only be non-distinct COUNT.
                    let count = Self::count_from_mask(&combined, dataset.as_ref())?;
                    let arr = Arc::new(Int64Array::from(vec![count])) as Arc<dyn Array>;
                    arrays.push(arr);
                }
            }
        }

        Ok(RecordBatch::try_new(schema, arrays)?)
    }
}

impl ExecutionPlan for AggregateIndexSearchExec {
    fn name(&self) -> &str {
        "AggregateIndexSearchExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_input {
            Some(input) => vec![input],
            None => vec![],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let prefilter_input = match children.len() {
            0 => None,
            1 => Some(children.into_iter().next().unwrap()),
            n => {
                return Err(datafusion::error::DataFusionError::Internal(format!(
                    "AggregateIndexSearchExec accepts 0 or 1 children, got {}",
                    n
                )));
            }
        };
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            aggregates: self.aggregates.clone(),
            aggregate_funcs: self.aggregate_funcs.clone(),
            prefilter_input,
            schema: self.schema.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let batch_fut = Self::do_execute(
            self.dataset.clone(),
            self.aggregates.clone(),
            self.prefilter_input.clone(),
            context,
            self.metrics.clone(),
            schema.clone(),
        );
        let stream = futures::stream::iter(vec![batch_fut])
            .then(|fut| async move { fut.await.map_err(|err| err.into()) })
            .boxed();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            schema,
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
            num_rows: datafusion::common::stats::Precision::Exact(1),
            ..datafusion::physical_plan::Statistics::new_unknown(&self.schema)
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

#[cfg(test)]
mod tests {
    use std::{ops::Bound, sync::Arc};

    use arrow::datatypes::{Int64Type, UInt64Type};
    use datafusion::common::DFSchema;
    use datafusion::execution::TaskContext;
    use datafusion::functions_aggregate;
    use datafusion::logical_expr::lit;
    use datafusion::physical_expr::execution_props::ExecutionProps;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_planner::create_aggregate_expr_and_maybe_filter;
    use datafusion::scalar::ScalarValue;
    use futures::TryStreamExt;
    use lance_core::utils::mask::{RowAddrMask, RowAddrTreeMap};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::gen_batch;
    use lance_index::IndexType;
    use lance_index::expression::aggregate::{AggregateIndexSearch, CountQuery};
    use lance_index::scalar::{
        SargableQuery, ScalarIndexParams,
        expression::{ScalarIndexExpr, ScalarIndexSearch},
    };

    use super::*;
    use crate::Dataset;
    use crate::index::DatasetIndexExt;
    use crate::io::exec::scalar_index::ScalarIndexExec;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    /// Build an `AggregateFunctionExpr` matching `COUNT(*)`.
    fn count_star_expr(input_schema: &SchemaRef) -> Arc<AggregateFunctionExpr> {
        let expr = functions_aggregate::count::count(lit(1));
        let df_schema = DFSchema::try_from(input_schema.as_ref().clone()).unwrap();
        let (agg_expr, _filter, _order_by) = create_aggregate_expr_and_maybe_filter(
            &expr,
            &df_schema,
            input_schema.as_ref(),
            &ExecutionProps::default(),
        )
        .unwrap();
        agg_expr
    }

    fn count_search(index_name: Option<&str>) -> Arc<AggregateIndexSearch> {
        Arc::new(AggregateIndexSearch {
            index_name: index_name.map(str::to_string),
            query: Arc::new(CountQuery::basic()),
            filter: None,
            original_expr: lit(0i64),
        })
    }

    struct Fixture {
        dataset: Arc<Dataset>,
        _tmp: TempStrDir,
    }

    /// 4 fragments × 10 rows, ascending `ordered` column with a BTree index.
    async fn make_fixture() -> Fixture {
        let tmp = TempStrDir::default();
        let mut dataset = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_dataset(
                tmp.as_str(),
                FragmentCount::from(4),
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

        Fixture {
            dataset: Arc::new(dataset),
            _tmp: tmp,
        }
    }

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![arrow_schema::Field::new(
            "ordered",
            arrow_schema::DataType::UInt64,
            false,
        )]))
    }

    async fn run(plan: AggregateIndexSearchExec) -> i64 {
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::PrimitiveArray<Int64Type>>()
            .expect("count partial state should be Int64")
            .value(0)
    }

    #[tokio::test]
    async fn try_new_rejects_length_mismatch() {
        let fixture = make_fixture().await;
        let schema = input_schema();
        let err = AggregateIndexSearchExec::try_new(
            fixture.dataset,
            vec![count_search(None)],
            vec![count_star_expr(&schema), count_star_expr(&schema)],
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "{err}");
    }

    #[tokio::test]
    async fn try_new_rejects_distinct_count_without_index() {
        let fixture = make_fixture().await;
        let schema = input_schema();
        let distinct = Arc::new(AggregateIndexSearch {
            index_name: None,
            query: Arc::new(CountQuery::distinct()),
            filter: None,
            original_expr: lit(0i64),
        });
        let err = AggregateIndexSearchExec::try_new(
            fixture.dataset,
            vec![distinct],
            vec![count_star_expr(&schema)],
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("distinct count"), "{err}");
    }

    #[tokio::test]
    async fn count_from_mask_mixes_full_and_partial() {
        // Synthesize an AllowList containing one Full-marker fragment and one
        // Partial bitmap; verify the Full fragment falls back to physical_rows
        // from the manifest and Partial falls back to bitmap.len().
        let fixture = make_fixture().await;
        let mut tm = RowAddrTreeMap::new();
        // Fragment 0: full (10 physical rows).
        tm.insert_fragment(0);
        // Fragment 1: partial with explicit row addrs.
        let row_addr_for = |frag_id: u32, offset: u32| ((frag_id as u64) << 32) | offset as u64;
        tm.insert(row_addr_for(1, 0));
        tm.insert(row_addr_for(1, 1));
        tm.insert(row_addr_for(1, 2));

        let mask = RowAddrMask::AllowList(tm);
        let count =
            AggregateIndexSearchExec::count_from_mask(&mask, fixture.dataset.as_ref()).unwrap();
        assert_eq!(count, 10 + 3);
    }

    #[tokio::test]
    async fn execute_count_no_prefilter() {
        let fixture = make_fixture().await;
        let dataset = fixture.dataset.clone();
        let schema = input_schema();
        let plan = AggregateIndexSearchExec::try_new(
            dataset.clone(),
            vec![count_search(None)],
            vec![count_star_expr(&schema)],
            None,
        )
        .unwrap();
        let count = run(plan).await;
        assert_eq!(count, 40); // 4 fragments × 10 rows
    }

    #[tokio::test]
    async fn execute_count_with_allow_list_prefilter() {
        let fixture = make_fixture().await;
        let dataset = fixture.dataset.clone();
        let schema = input_schema();

        // `ordered < 25` matches 25 rows across the four fragments.
        let prefilter_expr = ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "ordered".to_string(),
            index_name: "ordered_idx".to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt64(Some(25))),
            )),
            needs_recheck: false,
        });
        let prefilter: Arc<dyn ExecutionPlan> =
            Arc::new(ScalarIndexExec::new(dataset.clone(), prefilter_expr));

        let plan = AggregateIndexSearchExec::try_new(
            dataset.clone(),
            vec![count_search(None)],
            vec![count_star_expr(&schema)],
            Some(prefilter),
        )
        .unwrap();
        let count = run(plan).await;
        assert_eq!(count, 25);
    }

    #[tokio::test]
    async fn execute_count_with_block_list_prefilter() {
        let fixture = make_fixture().await;
        let dataset = fixture.dataset.clone();
        let schema = input_schema();

        // NOT(ordered < 25) is a block list of those 25 rows — 40 − 25 = 15.
        let prefilter_expr =
            ScalarIndexExpr::Not(Box::new(ScalarIndexExpr::Query(ScalarIndexSearch {
                column: "ordered".to_string(),
                index_name: "ordered_idx".to_string(),
                index_type: "BTree".to_string(),
                query: Arc::new(SargableQuery::Range(
                    Bound::Unbounded,
                    Bound::Excluded(ScalarValue::UInt64(Some(25))),
                )),
                needs_recheck: false,
            })));
        let prefilter: Arc<dyn ExecutionPlan> =
            Arc::new(ScalarIndexExec::new(dataset.clone(), prefilter_expr));

        let plan = AggregateIndexSearchExec::try_new(
            dataset.clone(),
            vec![count_search(None)],
            vec![count_star_expr(&schema)],
            Some(prefilter),
        )
        .unwrap();
        let count = run(plan).await;
        assert_eq!(count, 15);
    }

    #[tokio::test]
    async fn execute_count_respects_deletions() {
        let fixture = make_fixture().await;
        let mut dataset = (*fixture.dataset).clone();
        // Delete the first ten rows of the dataset (which live in fragment 0).
        dataset.delete("ordered < 10").await.unwrap();
        let dataset = Arc::new(dataset);

        let schema = input_schema();
        let plan = AggregateIndexSearchExec::try_new(
            dataset.clone(),
            vec![count_search(None)],
            vec![count_star_expr(&schema)],
            None,
        )
        .unwrap();
        let count = run(plan).await;
        assert_eq!(count, 30);
    }
}
