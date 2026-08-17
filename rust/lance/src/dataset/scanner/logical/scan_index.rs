// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Recording on a Lance scan's source how it will find its rows.

use std::any::Any;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::expr_rewriter::unnormalize_cols;
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use super::context::ScanPlanningContext;
use super::source::LanceScanSource;

/// Derive each Lance scan's scalar index query and record it on the scan's source.
///
/// This is what gives the index decision a place in the plan. Until it runs, the decision exists
/// only inside `TableProvider::scan`, where no rule can see it — which is why the scan leaf was the
/// one coverage split that had to be written out by hand.
///
/// It runs after `PushDownFilter`, so the predicate has reached its final position first.
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
