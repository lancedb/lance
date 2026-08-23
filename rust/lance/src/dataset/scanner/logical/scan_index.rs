// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Recording on a Lance scan's source how it will find its rows: a scalar index query, or a row
//! restriction that makes the read a take.

use std::any::Any;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::expr_rewriter::unnormalize_cols;
use datafusion::logical_expr::utils::conjunction;
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use lance_select::mask::RowAddrTreeMap;
use roaring::RoaringBitmap;

use super::PrefilterSourceKind;
use super::context::{OverlayStaleness, ScanPlanningContext};
use super::source::{LanceScanSource, ScanRestriction};
use crate::dataset::scanner::TakeOperation;

/// Derive each Lance scan's scalar index query and record it on the scan's source.
///
/// This is what gives the index decision a place in the plan. Until it runs, the decision exists
/// only inside `TableProvider::scan`, where no rule can see it — which is why the scan leaf was the
/// one coverage split that had to be written out by hand.
///
/// It runs after `PushDownFilter` for the same reason
/// [`ResolvePrefilterSource`] does: the predicate has to have reached its final position first.
#[derive(Debug)]
pub struct ResolveScalarIndexQuery {
    context: Arc<ScanPlanningContext>,
}

impl ResolveScalarIndexQuery {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl OptimizerRule for ResolveScalarIndexQuery {
    fn name(&self) -> &str {
        "resolve_scalar_index_query"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::TableScan(scan) = &plan else {
            return Ok(Transformed::no(plan));
        };
        // The physical planner unqualifies a `TableScan`'s filters before handing them to
        // `TableProvider::scan`, and the Lance expression planner expects the same bare columns.
        let filters = unnormalize_cols(scan.filters.iter().cloned());
        let resolved = with_lance_source(&plan, |source| {
            match source.filter_plan().is_some() || filters.is_empty() {
                true => None,
                false => Some(source.resolve_filter_plan(&filters, self.context.scalar_indices())),
            }
        });
        let Some(Some(filter_plan)) = resolved else {
            return Ok(Transformed::no(plan));
        };
        let filter_plan = filter_plan.map_err(datafusion::common::DataFusionError::from)?;
        Ok(Transformed::yes(map_lance_scan(&plan, |source| {
            source.with_filter_plan(filter_plan.clone())
        })?))
    }
}

/// Run `f` against the [`LanceScanSource`] behind a `TableScan`, if that is what this node is.
///
/// Closure-shaped rather than returning a reference because `source_as_provider` hands back an
/// owned `Arc`, so the borrow cannot outlive this call.
pub fn with_lance_source<R>(
    plan: &LogicalPlan,
    f: impl FnOnce(&LanceScanSource) -> R,
) -> Option<R> {
    let LogicalPlan::TableScan(scan) = plan else {
        return None;
    };
    let provider = source_as_provider(&scan.source).ok()?;
    let source = (provider.as_ref() as &dyn Any).downcast_ref::<LanceScanSource>()?;
    Some(f(source))
}

/// Rebuild `plan`, replacing every Lance scan leaf's source with `f` applied to it.
pub fn map_lance_scan(
    plan: &LogicalPlan,
    f: impl Fn(&LanceScanSource) -> LanceScanSource,
) -> datafusion::common::Result<LogicalPlan> {
    Ok(plan
        .clone()
        .transform_down(|node| {
            let Some(source) = with_lance_source(&node, &f) else {
                return Ok(Transformed::no(node));
            };
            let LogicalPlan::TableScan(scan) = &node else {
                return Ok(Transformed::no(node));
            };
            let mut scan = scan.clone();
            scan.source = provider_as_source(Arc::new(source));
            Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
        })?
        .data)
}

/// Rebuild `plan`, pointing every scan leaf at the same source narrowed by `restriction`.
///
/// The recursion exists because the predicate may not have reached the leaf: before
/// `PushDownFilter` runs there is a `Filter` in between, and that `Filter` has to be duplicated
/// onto every branch along with the scan.
pub fn restrict_scan(
    plan: &LogicalPlan,
    restriction: &ScanRestriction,
) -> datafusion::common::Result<LogicalPlan> {
    Ok(plan
        .clone()
        .transform_down(|node| {
            let LogicalPlan::TableScan(scan) = &node else {
                return Ok(Transformed::no(node));
            };
            let Some(provider) = source_as_provider(&scan.source).ok() else {
                return Ok(Transformed::no(node));
            };
            let Some(source) = (provider.as_ref() as &dyn Any).downcast_ref::<LanceScanSource>()
            else {
                return Ok(Transformed::no(node));
            };
            let mut scan = scan.clone();
            scan.source = provider_as_source(Arc::new(source.restricted_to(restriction)));
            Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
        })?
        .data)
}

/// The index lookup that can stand in for a search's whole prefilter subtree, if this one can.
///
/// A prefilter is normally a read that materializes the row ids the predicate selects. When the
/// predicate is answered exactly by a scalar index, the lookup already *is* that row set, so the
/// search can consume it directly and the read never happens. Mirrors the `ScalarIndexExec`
/// branch of `Scanner::prefilter_source`.
///
/// `required_fragments` are the fragments the search itself can return a row from: a candidate
/// set that cannot speak for one of them would silently drop rows the search would have found.
pub fn scalar_index_prefilter(
    input: &LogicalPlan,
    required_fragments: &RoaringBitmap,
    context: &ScanPlanningContext,
) -> Option<PrefilterSourceKind> {
    with_lance_source(input, |source| {
        let options = source.options();
        // The lookup reads the whole dataset, so a narrowing of the scan it replaces is lost with
        // the scan. Which narrowings matter differs:
        //
        // * A caller's `with_fragments` is enforced by this read and nothing else, so dropping it
        //   would let rows from other fragments through.
        // * A row restriction singles out rows whose index entries are not to be trusted, which is
        //   the opposite of what a lookup would answer.
        // * A restriction a coverage split added is neither: it narrows the read to what this
        //   branch is responsible for, and the branch's index can only emit rows from there
        //   anyway, so a wider allow list adds nothing.
        if context.take_settings().fragments.is_some()
            || options.rows.is_some()
            || options.overlay_block.is_some()
        {
            return None;
        }
        let filter_plan = source.filter_plan()?;
        if !filter_plan.is_exact_index_search() {
            return None;
        }
        let query = filter_plan.index_query.clone()?;
        // Stale entries would reach the search as candidates whose indexed values no longer hold.
        // The read this replaces is what masks them out, so keep it.
        if !matches!(
            context.index_query_staleness(&query),
            OverlayStaleness::None
        ) {
            return None;
        }
        let covered = context.index_query_coverage(&query)?;
        if !context.fast_search() && !required_fragments.is_subset(&covered) {
            return None;
        }
        Some(PrefilterSourceKind::ScalarIndexQuery {
            query: Arc::new(query),
            result_format: options.index_expr_result_format,
        })
    })?
}

pub fn restricts_candidates(plan: &LogicalPlan) -> bool {
    let mut restricts = false;
    let _ = plan.apply(|node| {
        match node {
            LogicalPlan::Filter(_) => restricts = true,
            LogicalPlan::TableScan(scan) if !scan.filters.is_empty() => restricts = true,
            _ => {}
        }
        if restricts {
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    });
    restricts
}

/// Turn a predicate that names its rows into a direct take.
///
/// `_rowid IN (10, 20)` leaves nothing to search for: the ids *are* the selection. Recording them
/// on the scan's source is the logical-layer version of `Scanner::take_source`, which assembles the
/// same read by hand.
///
/// Row ids are resolved here; `_rowaddr` and `_rowoffset` name physical positions, and turning
/// those into row ids reads row-id sequences and deletion vectors, so stage 2 does it and this
/// looks the answer up.
///
/// Runs after `PushDownFilter`, so the predicate has reached the scan, and before
/// [`ResolveScalarIndexQuery`] — a row restriction and a scalar index query compete for the same
/// slot on the read, and this one wins.
#[derive(Debug)]
pub struct ResolveTake {
    context: Arc<ScanPlanningContext>,
}

impl ResolveTake {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl OptimizerRule for ResolveTake {
    fn name(&self) -> &str {
        "resolve_take"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        match &plan {
            LogicalPlan::TableScan(_) => self.restrict_scan(plan),
            _ => Ok(Transformed::no(plan)),
        }
    }
}

impl ResolveTake {
    /// The row ids a take selects, or `None` when the plan cannot be rewritten.
    ///
    /// Only row ids are resolved here. `_rowaddr` and `_rowoffset` name physical positions, and
    /// translating those reads row-id sequences and deletion vectors, so stage 2 did it and this
    /// looks the answer up. A miss means stage 2 walked a different plan than this one, so the
    /// predicate is left alone rather than guessed at: reading every row and filtering is slower,
    /// not wrong.
    fn rows_for(&self, take: &TakeOperation) -> Option<Arc<RowAddrTreeMap>> {
        match take {
            TakeOperation::RowIds(ids) => {
                Some(Arc::new(RowAddrTreeMap::from_iter(ids.iter().copied())))
            }
            _ => self.context.take_rows(take).cloned(),
        }
    }

    fn restrict_scan(
        &self,
        plan: LogicalPlan,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::TableScan(scan) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let restrictable = with_lance_source(&plan, |source| source.options().rows.is_none());
        if restrictable != Some(true) || scan.filters.is_empty() {
            return Ok(Transformed::no(plan));
        }
        let Some(predicate) = conjunction(unnormalize_cols(scan.filters.iter().cloned())) else {
            return Ok(Transformed::no(plan));
        };
        let Some((take, remainder)) = TakeOperation::try_from_expr(&predicate) else {
            return Ok(Transformed::no(plan));
        };
        let Some(rows) = self.rows_for(&take) else {
            return Ok(Transformed::no(plan));
        };
        let mut scan = scan.clone();
        scan.filters = remainder.into_iter().collect();
        let restricted = LogicalPlan::TableScan(scan);
        Ok(Transformed::yes(map_lance_scan(&restricted, |source| {
            source.restricted_to(&ScanRestriction::Rows(rows.clone()))
        })?))
    }
}
