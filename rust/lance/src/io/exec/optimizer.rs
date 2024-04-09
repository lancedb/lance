// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Lance Physical Optimizer Rules

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    error::Result as DFResult,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{projection::ProjectionExec as DFProjectionExec, ExecutionPlan},
};
use datafusion_physical_expr::expressions::Column;

use super::TakeExec;

/// Rule that eliminates [TakeExec] nodes that are immediately followed by another [TakeExec].
pub struct CoalesceTake;

impl PhysicalOptimizerRule for CoalesceTake {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(&|plan| {
            if let Some(take) = plan.as_any().downcast_ref::<TakeExec>() {
                let child = &take.children()[0];
                if let Some(exec_child) = child.as_any().downcast_ref::<TakeExec>() {
                    let upstream_plan = exec_child.children();
                    return Ok(Transformed::yes(plan.with_new_children(upstream_plan)?));
                }
            }
            Ok(Transformed::no(plan))
        })
        .map(|plan| plan.data)
    }

    fn name(&self) -> &str {
        "coalesce_take"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Rule that eliminates [ProjectionExec] nodes that projects all columns
/// from its input with no additional expressions.
pub struct SimplifyProjection;

impl PhysicalOptimizerRule for SimplifyProjection {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(&|plan| {
            if let Some(proj) = plan.as_any().downcast_ref::<DFProjectionExec>() {
                let children = &proj.children();
                if children.len() != 1 {
                    return Ok(Transformed::no(plan));
                }

                let input = &children[0];

                // TODO: we could try to coalesce consecutive projections, something for later
                // For now, we just keep things simple and only remove NoOp projections

                // output has differnet schema, projection needed
                if input.schema() != proj.schema() {
                    return Ok(Transformed::no(plan));
                }

                if proj.expr().iter().enumerate().all(|(index, (expr, name))| {
                    if let Some(expr) = expr.as_any().downcast_ref::<Column>() {
                        // no renaming, no reordering
                        expr.index() == index && expr.name() == name
                    } else {
                        false
                    }
                }) {
                    return Ok(Transformed::yes(input.clone()));
                }
            }
            Ok(Transformed::no(plan))
        })
        .map(|plan| plan.data)
    }

    fn name(&self) -> &str {
        "simplify_projection"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
