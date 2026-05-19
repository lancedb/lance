// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Physical optimizer rule that rewrites `COUNT`-shaped aggregates into
//! [`AggregateIndexSearchExec`].
//!
//! v1 only fires for fully unfiltered counts — the simplest provably-safe
//! envelope. Filtered counts are deferred to a follow-up that can validate
//! the index covers every dataset fragment.
//!
//! Recognized shape:
//!
//! ```text
//! AggregateExec(Single, aggs=[COUNT(*)], group_by=[])
//!   └── FilteredReadExec { no full_filter, no refine_filter, no index_input,
//!                          no scan range, no with_deleted_rows, no fragment
//!                          subset, not stable-row-ids }
//! ```
//!
//! Rewritten to:
//!
//! ```text
//! AggregateExec(Final, aggs=[COUNT(*)], group_by=[])
//!   └── AggregateIndexSearchExec { prefilter_input = None }
//! ```
//!
//! [`AggregateIndexSearchExec`] emits partial-state, so the outer
//! `AggregateExec(Final)` performs the final combine.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::lit;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::{
    ExecutionPlan,
    aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr::expressions::Literal;
use lance_index::expression::aggregate::{AggregateIndexSearch, CountQuery};

use super::aggregate_index::AggregateIndexSearchExec;
use super::filtered_read::FilteredReadExec;

/// Physical optimizer rule that pushes `COUNT`-shaped aggregates into
/// [`AggregateIndexSearchExec`], answering them from index metadata + the
/// deletion mask + an optional scalar-index prefilter, without scanning column
/// data.
///
/// Only fires when the shape is verifiably safe; everything outside that
/// envelope (GROUP BY, residual filters, scan ranges, etc.) is left alone for
/// the normal scan path.
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

    // Every aggregate must be a `COUNT(<literal>)` shape (i.e. COUNT(*) /
    // COUNT(1) / etc.) with no per-aggregate FILTER. Anything that depends on
    // a column value can't be answered without scanning that column.
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
    // No filter at all is the only case v1 can prove correct. With a filter we
    // would also need to verify the scalar index covers every dataset fragment
    // (otherwise rows in unindexed fragments are silently dropped). That check
    // is async and not currently expressible in a sync PhysicalOptimizerRule;
    // until we plumb it through, leave the filtered case on the scan path.
    if options.full_filter.is_some()
        || options.refine_filter.is_some()
        || filtered_read.index_input().is_some()
    {
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
    // We assume the natural fragment coverage of the dataset; a fragment
    // subset would require routing it into the exec.
    if options.fragments.is_some() {
        return Ok(None);
    }

    let dataset = filtered_read.dataset().clone();
    let prefilter_input = filtered_read.index_input().cloned();
    let aggregates: Vec<Arc<AggregateIndexSearch>> = agg
        .aggr_expr()
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
    let aggregate_funcs: Vec<Arc<AggregateFunctionExpr>> = agg.aggr_expr().to_vec();

    let exec =
        AggregateIndexSearchExec::try_new(dataset, aggregates, aggregate_funcs, prefilter_input)?;
    let exec_schema = exec.schema();
    let exec: Arc<dyn ExecutionPlan> = Arc::new(exec);

    // Wrap with AggregateExec(Final) so a downstream consumer that expected
    // the original AggregateExec output schema continues to see it.
    let null_filters: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        (0..agg.aggr_expr().len()).map(|_| None).collect();
    let final_agg = AggregateExec::try_new(
        AggregateMode::Final,
        PhysicalGroupBy::default(),
        agg.aggr_expr().to_vec(),
        null_filters,
        exec,
        exec_schema,
    )?;
    Ok(Some(Arc::new(final_agg)))
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

    /// True if `plan` contains an `AggregateIndexSearchExec` anywhere in its tree.
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

    /// Drive the rule via `Scanner::create_plan` (which registers the rule
    /// through `get_physical_optimizer`) and return both the plan and the
    /// final count for inspection.
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
    }

    #[tokio::test]
    async fn rule_skips_when_filter_present_even_if_indexed() {
        // Deferred until the rule can verify the index covers every dataset
        // fragment — without that check, an index built before a fragment
        // append silently drops rows. See `rule_skips_partial_index_coverage`
        // below for the regression scenario this protects against.
        let fixture = make_fixture().await;
        let mut scanner = fixture.dataset.scan();
        scanner.filter("ordered < 25").unwrap();
        let (plan, count) = run_count(&mut scanner).await;
        assert_eq!(count, 25);
        assert!(
            !plan_contains_pushdown(&plan),
            "rule should not fire with any filter in v1, got plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_skips_partial_index_coverage() {
        // Regression: when an index doesn't cover every dataset fragment
        // (here, by appending a fragment after the index was built), the rule
        // must not fire — otherwise rows in unindexed fragments are silently
        // dropped. Today this is enforced by the blanket "no filter" gate.
        use crate::dataset::WriteParams;
        let tmp = TempStrDir::default();
        // Build a 4×10 dataset with a BTree index covering all 4 fragments.
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
        // Append a fragment after the index was built — it is unindexed.
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
            !plan_contains_pushdown(&plan),
            "rule must not fire when the index has partial coverage, got plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_skips_with_stable_row_ids() {
        // Regression: with stable row IDs the deletion mask is built in
        // stable-id space while fragments_allow is in row-address space.
        // ANDing across the two undercounts; refuse to fire.
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
        // Touch a deletion so we exercise the masks that would otherwise
        // collide across id spaces.
        dataset.delete("ordered = 0").await.unwrap();
        let dataset = Arc::new(dataset);

        let mut scanner = dataset.scan();
        let (plan, count) = run_count(&mut scanner).await;
        // 2 × 10 rows, minus the one deletion.
        assert_eq!(count, 19);
        assert!(
            !plan_contains_pushdown(&plan),
            "rule must not fire under stable row IDs, got plan: {}",
            displayable(plan.as_ref()).indent(true)
        );
    }

    #[tokio::test]
    async fn rule_skips_when_filter_needs_refine() {
        // No index on `unindexed`, so the filter must be applied during the
        // scan; the rule must not fire.
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
        // 40 rows total, values are 0..40 across fragments; `> 5` drops 0..6.
        // Right answer either way; the point is the rule didn't fire.
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
        // GROUP BY isn't supported by the rule yet — make sure we leave it alone.
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
