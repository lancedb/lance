// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Physical optimizer rule that rewrites index-answerable aggregates into
//! [`AggregateIndexSearchExec`].
//!
//! The v1 implementation only recognizes non-distinct `COUNT(<literal>)`
//! aggregates, but the surrounding plumbing — the exec, the trait, the rule
//! shape — is built to grow to other aggregates (`MIN`/`MAX` over a zone
//! map, `COUNT(DISTINCT)` over a bitmap dictionary, etc.) without changing
//! the plan layout below.
//!
//! Two rewritten shapes are emitted depending on whether the scalar index
//! backing the filter covers every dataset fragment.
//!
//! **Full coverage** (index ⊇ dataset, or no filter at all):
//!
//! ```text
//! AggregateExec(Final, aggs=[…], group_by=[])
//!   └── AggregateIndexSearchExec { prefilter_input = index_input }
//! ```
//!
//! **Partial coverage** (index ⊊ dataset — typically appended fragments):
//!
//! ```text
//! AggregateExec(Final, aggs=[…], group_by=[])
//!   └── UnionExec
//!         ├── AggregateIndexSearchExec(restrict_to_fragments = indexed)
//!         └── AggregateExec(Partial)
//!               └── FilteredReadExec(fragments = unindexed, full_filter = …)
//! ```
//!
//! [`AggregateIndexSearchExec`] emits partial-state, so the outer
//! `AggregateExec(Final)` performs the final combine in either shape.
//!
//! If the prefilter's index coverage is unknown (any leaf is missing
//! `fragment_bitmap`, e.g. constructed outside scanner planning), the rule
//! refuses to fire and leaves the existing scan path in place.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::lit;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::{
    ExecutionPlan,
    aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
    coalesce_partitions::CoalescePartitionsExec,
    union::UnionExec,
};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr::expressions::Literal;
use lance_index::expression::aggregate::{AggregateIndexSearch, CountQuery};
use lance_index::scalar::expression::ScalarIndexExpr;
use roaring::RoaringBitmap;

use super::aggregate_index::AggregateIndexSearchExec;
use super::filtered_read::{FilteredReadExec, FilteredReadOptions};
use super::scalar_index::ScalarIndexExec;

/// Physical optimizer rule that pushes index-answerable aggregates into
/// [`AggregateIndexSearchExec`], optionally splitting into a parallel scan
/// branch when the index has partial coverage of the dataset.
///
/// Only fires when the shape is verifiably safe; everything outside that
/// envelope (GROUP BY, residual filters, scan ranges, etc.) is left alone for
/// the normal scan path. v1 only recognizes non-distinct `COUNT(<literal>)`;
/// future aggregates plug in via the same rewrite, just with different
/// [`AggregateIndexSearch`] queries and `ScalarIndex::calculate_aggregate`
/// impls.
#[derive(Debug)]
pub struct AggregateIndexPushdown;

impl PhysicalOptimizerRule for AggregateIndexPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|plan| {
                let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() else {
                    return Ok(Transformed::no(plan));
                };
                if let Some(rewritten) = try_rewrite(agg)? {
                    return Ok(Transformed::yes(rewritten));
                }
                Ok(Transformed::no(plan))
            })?
            .data)
    }

    fn name(&self) -> &str {
        "aggregate_index_pushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn try_rewrite(agg: &AggregateExec) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
    // The Lance scanner emits AggregateMode::Single. Other modes mean
    // somebody else is already wrapping us in a partial/final pair; leave them
    // alone to avoid double-wrapping.
    if !matches!(agg.mode(), AggregateMode::Single) {
        return Ok(None);
    }
    if !agg.group_expr().is_empty() {
        return Ok(None);
    }
    if agg.aggr_expr().is_empty() {
        return Ok(None);
    }

    // v1: every aggregate must be a `COUNT(<literal>)` shape (i.e. COUNT(*) /
    // COUNT(1) / etc.) with no per-aggregate FILTER. As more aggregates grow
    // their own `ScalarIndex::calculate_aggregate` impls (MIN/MAX off a zone
    // map, exact `COUNT(DISTINCT)` off a bitmap, …) this gate should grow
    // accordingly — keep the per-aggregate FILTER rejection regardless,
    // since per-aggregate filters depend on column values we can't scan.
    for (af, filter) in agg.aggr_expr().iter().zip(agg.filter_expr().iter()) {
        if !is_count_star(af) {
            return Ok(None);
        }
        if filter.is_some() {
            return Ok(None);
        }
    }

    // The input must be a FilteredReadExec we can prove is safe to skip.
    let child = &agg.children()[0];
    let Some(filtered_read) = child.as_any().downcast_ref::<FilteredReadExec>() else {
        return Ok(None);
    };

    // Stable-row-id mode: `DatasetPreFilter::create_deletion_mask` produces an
    // AllowList in stable-id space, but `AggregateIndexSearchExec` builds its
    // fragments-allow list in row-address space. ANDing across the two yields
    // a silently wrong count (rows in fragments > 0 are dropped because their
    // stable ids and row addresses share a fragment-id bucket only by accident).
    // Until the exec can reconcile the two id spaces, refuse to fire.
    if filtered_read.dataset().manifest().uses_stable_row_ids() {
        return Ok(None);
    }

    let options = filtered_read.options();
    // A refine filter is a residual the index couldn't fully evaluate — it
    // needs column data to apply, which we can't.
    if options.refine_filter.is_some() {
        return Ok(None);
    }
    // A full_filter without an index_input means the filter is evaluated by
    // scanning every row; not pushdownable.
    if options.full_filter.is_some() && filtered_read.index_input().is_none() {
        return Ok(None);
    }
    // LIMIT/OFFSET would change the count.
    if options.scan_range_before_filter.is_some() || options.scan_range_after_filter.is_some() {
        return Ok(None);
    }
    // We rely on the deletion mask being applied; with_deleted_rows changes
    // that contract.
    if options.with_deleted_rows {
        return Ok(None);
    }
    // A pre-existing fragment subset would need to be intersected into the
    // coverage logic below. Punt for now.
    if options.fragments.is_some() {
        return Ok(None);
    }

    let dataset = filtered_read.dataset().clone();
    let dataset_fragments: RoaringBitmap =
        dataset.fragments().iter().map(|f| f.id as u32).collect();
    let prefilter_input = filtered_read.index_input().cloned();

    // If there is a prefilter, compute the index coverage from its
    // ScalarIndexExpr leaves. None means at least one leaf has no
    // fragment_bitmap and we can't reason about coverage — refuse to fire.
    let index_coverage = match &prefilter_input {
        None => None,
        Some(input) => {
            let scalar_exec = input
                .as_any()
                .downcast_ref::<ScalarIndexExec>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(
                    "AggregateIndexPushdown: FilteredReadExec.index_input is not a ScalarIndexExec"
                        .to_string(),
                )
                })?;
            let Some(coverage) = collect_coverage(scalar_exec.expr()) else {
                return Ok(None);
            };
            Some(coverage)
        }
    };

    let aggr_exprs: Vec<Arc<AggregateFunctionExpr>> = agg.aggr_expr().to_vec();
    let aggregates: Vec<Arc<AggregateIndexSearch>> = aggr_exprs
        .iter()
        .map(|_| {
            Arc::new(AggregateIndexSearch {
                index_name: None,
                query: Arc::new(CountQuery::basic()),
                filter: None,
                // `original_expr` is only used for `Display`; the physical
                // plan no longer carries the source `Expr`.
                original_expr: lit(0i64),
            })
        })
        .collect();

    // Decide on the plan shape. Three cases:
    //
    // 1. No prefilter (no filter at all): single pushdown branch over every
    //    dataset fragment. Always safe.
    // 2. Prefilter + index covers every dataset fragment: single pushdown
    //    branch, prefilter feeds in directly.
    // 3. Prefilter + index covers a strict subset: split into pushdown over
    //    indexed fragments + parallel scan over unindexed fragments.
    let (combined, partial_input_schema): (Arc<dyn ExecutionPlan>, _) = match index_coverage {
        None => {
            // No prefilter at all (verified above): nothing to restrict.
            let exec = AggregateIndexSearchExec::try_new_restricted(
                dataset,
                aggregates,
                aggr_exprs.clone(),
                prefilter_input,
                None,
            )?;
            let schema = exec.schema();
            (Arc::new(exec), schema)
        }
        Some(coverage) if (&dataset_fragments - &coverage).is_empty() => {
            // Prefilter exists and the index covers every dataset fragment —
            // safe to push the whole count down.
            let exec = AggregateIndexSearchExec::try_new_restricted(
                dataset,
                aggregates,
                aggr_exprs.clone(),
                prefilter_input,
                None,
            )?;
            let schema = exec.schema();
            (Arc::new(exec), schema)
        }
        Some(coverage) => {
            // Split plan: AggregateIndexSearchExec for the indexed fragments,
            // a normal scan + AggregateExec(Partial) for the rest.
            let uncovered = &dataset_fragments - &coverage;
            let pushdown_exec = AggregateIndexSearchExec::try_new_restricted(
                dataset,
                aggregates,
                aggr_exprs.clone(),
                prefilter_input,
                Some(&dataset_fragments & &coverage),
            )?;
            let partial_state_schema = pushdown_exec.schema();
            let pushdown_branch: Arc<dyn ExecutionPlan> = Arc::new(pushdown_exec);

            let scan_branch =
                build_scan_branch(filtered_read, options, &uncovered, aggr_exprs.clone())?;

            // Union exposes one partition per input; CoalescePartitionsExec
            // flattens them so the Final aggregate sees a single partition
            // with all the partial-state rows.
            let union = UnionExec::try_new(vec![pushdown_branch, scan_branch])?;
            let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(union));
            (coalesced, partial_state_schema)
        }
    };

    // Wrap with AggregateExec(Final) so a downstream consumer that expected
    // the original AggregateExec output schema continues to see it.
    let null_filters: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        (0..aggr_exprs.len()).map(|_| None).collect();
    let final_agg = AggregateExec::try_new(
        AggregateMode::Final,
        PhysicalGroupBy::default(),
        aggr_exprs,
        null_filters,
        combined,
        partial_input_schema,
    )?;
    Ok(Some(Arc::new(final_agg)))
}

/// Build the scan branch of a partial-coverage split: a `FilteredReadExec`
/// restricted to the uncovered fragments (no `index_input`, the original
/// `full_filter` applied per row) wrapped in `AggregateExec(Partial)` so its
/// partial state can be unioned with the pushdown branch.
fn build_scan_branch(
    filtered_read: &FilteredReadExec,
    options: &FilteredReadOptions,
    uncovered: &RoaringBitmap,
    aggr_exprs: Vec<Arc<AggregateFunctionExpr>>,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let dataset = filtered_read.dataset().clone();
    let uncovered_fragments: Vec<_> = dataset
        .manifest()
        .fragments
        .iter()
        .filter(|f| uncovered.contains(f.id as u32))
        .cloned()
        .collect();
    let mut scan_options = options.clone();
    scan_options.fragments = Some(Arc::new(uncovered_fragments));
    let scan = FilteredReadExec::try_new(dataset, scan_options, None)?;
    let scan: Arc<dyn ExecutionPlan> = Arc::new(scan);
    let scan_schema = scan.schema();
    let null_filters: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        (0..aggr_exprs.len()).map(|_| None).collect();
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        PhysicalGroupBy::default(),
        aggr_exprs,
        null_filters,
        scan,
        scan_schema,
    )?;
    Ok(Arc::new(partial))
}

/// Walk a `ScalarIndexExpr` and intersect the per-leaf `fragment_bitmap`.
///
/// Returns `None` if any leaf is missing a bitmap (coverage unknown). All
/// three combinators (`And`, `Or`, `Not`) reduce to "every leaf must cover the
/// fragment for us to give a definitive answer about it" — i.e. intersection.
fn collect_coverage(expr: &ScalarIndexExpr) -> Option<RoaringBitmap> {
    match expr {
        ScalarIndexExpr::Not(inner) => collect_coverage(inner),
        ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
            let l = collect_coverage(lhs)?;
            let r = collect_coverage(rhs)?;
            Some(l & r)
        }
        ScalarIndexExpr::Query(search) => search.fragment_bitmap.clone(),
    }
}

/// Returns `true` if `af` is `COUNT(<literal>)` with no DISTINCT.
fn is_count_star(af: &Arc<AggregateFunctionExpr>) -> bool {
    if af.fun().name() != "count" {
        return false;
    }
    if af.is_distinct() {
        return false;
    }
    let args = af.expressions();
    if args.len() != 1 {
        return false;
    }
    let Some(lit) = args[0].as_any().downcast_ref::<Literal>() else {
        return false;
    };
    // `COUNT(NULL)` would always return 0; rule it out so we don't accidentally
    // produce a wrong answer if the planner ever lets it through.
    !lit.value().is_null()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{Int64Type, UInt64Type};
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::physical_plan::{ExecutionPlan, displayable};
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::gen_batch;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;

    use super::*;
    use crate::Dataset;
    use crate::dataset::scanner::AggregateExpr;
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

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

    fn plan_contains_pushdown(plan: &Arc<dyn ExecutionPlan>) -> bool {
        let mut found = false;
        plan.apply(|node| {
            if node.as_any().is::<AggregateIndexSearchExec>() {
                found = true;
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
            }
        })
        .unwrap();
        found
    }

    fn plan_contains_union(plan: &Arc<dyn ExecutionPlan>) -> bool {
        let mut found = false;
        plan.apply(|node| {
            if node.as_any().is::<UnionExec>() {
                found = true;
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
            }
        })
        .unwrap();
        found
    }

    async fn run_count(
        scanner: &mut crate::dataset::scanner::Scanner,
    ) -> (Arc<dyn ExecutionPlan>, i64) {
        scanner
            .aggregate(AggregateExpr::builder().count_star().build())
            .unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let stream = datafusion::physical_plan::execute_stream(
            plan.clone(),
            Arc::new(datafusion::execution::TaskContext::default()),
        )
        .unwrap();
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(
            batches.len(),
            1,
            "count plan emitted {} batches",
            batches.len()
        );
        let count = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::PrimitiveArray<Int64Type>>()
            .expect("count column should be Int64")
            .value(0);
        (plan, count)
    }

    #[tokio::test]
    async fn rule_fires_on_unfiltered_count_star() {
        let fixture = make_fixture().await;
        let mut scanner = fixture.dataset.scan();
        let (plan, count) = run_count(&mut scanner).await;
        assert_eq!(count, 40);
        assert!(
            plan_contains_pushdown(&plan),
            "expected AggregateIndexSearchExec in plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
        assert!(
            !plan_contains_union(&plan),
            "no union expected for unfiltered count, got: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_fires_when_filter_fully_indexed() {
        let fixture = make_fixture().await;
        let mut scanner = fixture.dataset.scan();
        scanner.filter("ordered < 25").unwrap();
        let (plan, count) = run_count(&mut scanner).await;
        assert_eq!(count, 25);
        assert!(
            plan_contains_pushdown(&plan),
            "expected AggregateIndexSearchExec in plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
        assert!(
            !plan_contains_union(&plan),
            "no union expected when index covers every fragment, got: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_emits_split_plan_for_partial_index_coverage() {
        // Build index over 4 fragments, then append a 5th — the index now
        // covers a strict subset of the dataset. The rule must split into a
        // pushdown branch over the indexed fragments and a scan branch over
        // the rest, then sum the partials.
        use crate::dataset::WriteParams;
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
        let extra = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(10),
                lance_datagen::BatchCount::from(1),
            );
        let dataset = Dataset::write(
            extra,
            tmp.as_str(),
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Append,
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let dataset = Arc::new(dataset);

        let mut scanner = dataset.scan();
        scanner.filter("ordered < 100").unwrap();
        let (plan, count) = run_count(&mut scanner).await;
        // 5 fragments × 10 rows, all match `< 100`.
        assert_eq!(count, 50);
        assert!(
            plan_contains_pushdown(&plan),
            "expected pushdown branch in split plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
        assert!(
            plan_contains_union(&plan),
            "expected UnionExec for partial-coverage split, got: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_skips_with_stable_row_ids() {
        use crate::dataset::WriteParams;
        let tmp = TempStrDir::default();
        let mut dataset = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .into_dataset_with_params(
                tmp.as_str(),
                FragmentCount::from(2),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset.delete("ordered = 0").await.unwrap();
        let dataset = Arc::new(dataset);

        let mut scanner = dataset.scan();
        let (plan, count) = run_count(&mut scanner).await;
        assert_eq!(count, 19);
        assert!(
            !plan_contains_pushdown(&plan),
            "rule must not fire under stable row IDs, got plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_skips_when_filter_needs_refine() {
        let tmp = TempStrDir::default();
        let mut dataset = gen_batch()
            .col("ordered", lance_datagen::array::step::<UInt64Type>())
            .col("unindexed", lance_datagen::array::step::<UInt64Type>())
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
        let dataset = Arc::new(dataset);

        let mut scanner = dataset.scan();
        scanner.filter("unindexed > 5").unwrap();
        let (plan, count) = run_count(&mut scanner).await;
        assert_eq!(count, 34);
        assert!(
            !plan_contains_pushdown(&plan),
            "rule should not fire with non-indexed filter, got plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_skips_count_with_group_by() {
        let fixture = make_fixture().await;
        let mut scanner = fixture.dataset.scan();
        scanner
            .aggregate(
                AggregateExpr::builder()
                    .group_by("ordered")
                    .count_star()
                    .build(),
            )
            .unwrap();
        let plan = scanner.create_plan().await.unwrap();
        assert!(
            !plan_contains_pushdown(&plan),
            "rule should not fire for GROUP BY: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }
}
