// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deriving each Lance scan's scalar index query and recording it on the scan's source.

use std::any::Any;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::expr_rewriter::unnormalize_cols;
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use super::context::ScanPlanningContext;
use super::source::{LanceScanSource, ScanRestriction};

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
