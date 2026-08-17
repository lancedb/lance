// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rule plumbing shared by every index type.

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::LogicalPlan;

/// Apply a node rewrite bottom-up over the whole plan.
///
/// An [`AnalyzerRule`] receives the entire plan, so the traversal that
/// [`OptimizerRule::apply_order`] used to supply is written out here instead. Bottom-up matters for
/// the same reason it did there: a rule that replaces a node with a subtree must not then descend
/// into the subtree it just built.
pub fn analyze_bottom_up(
    plan: LogicalPlan,
    rewrite: impl FnMut(LogicalPlan) -> datafusion::common::Result<Transformed<LogicalPlan>>,
) -> datafusion::common::Result<LogicalPlan> {
    Ok(plan.transform_up(rewrite)?.data)
}
