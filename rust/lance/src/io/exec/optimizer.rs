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
    physical_plan::ExecutionPlan,
};

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
                    return Ok(Transformed::Yes(plan.with_new_children(upstream_plan)?));
                }
            }
            Ok(Transformed::No(plan))
        })
    }

    fn name(&self) -> &str {
        "coalesce_take"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
