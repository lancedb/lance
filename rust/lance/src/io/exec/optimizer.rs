// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Physical Optimizer Rules

use std::sync::Arc;

use super::TakeExec;
use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    error::Result as DFResult,
    physical_optimizer::{optimizer::PhysicalOptimizer, PhysicalOptimizerRule},
    physical_plan::{
        coalesce_batches::CoalesceBatchesExec, projection::ProjectionExec, ExecutionPlan,
    },
};
use datafusion_physical_expr::{expressions::Column, PhysicalExpr};

/// Rule that eliminates [TakeExec] nodes that are immediately followed by another [TakeExec].
#[derive(Debug)]
pub struct CoalesceTake;

impl CoalesceTake {
    fn field_order_differs(old_schema: &ArrowSchema, new_schema: &ArrowSchema) -> bool {
        old_schema
            .fields
            .iter()
            .zip(&new_schema.fields)
            .any(|(old, new)| old.name() != new.name())
    }

    fn remap_collapsed_output(
        old_schema: &ArrowSchema,
        new_schema: &ArrowSchema,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let mut project_exprs = Vec::with_capacity(old_schema.fields.len());
        for field in &old_schema.fields {
            project_exprs.push((
                Arc::new(Column::new_with_schema(field.name(), new_schema).unwrap())
                    as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            ));
        }
        Arc::new(ProjectionExec::try_new(project_exprs, plan).unwrap())
    }

    fn collapse_takes(
        inner_take: &TakeExec,
        outer_take: &TakeExec,
        outer_exec: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let inner_take_input = inner_take.children()[0].clone();
        let old_output_schema = outer_take.schema();
        let collapsed = outer_exec
            .with_new_children(vec![inner_take_input])
            .unwrap();
        let new_output_schema = collapsed.schema();

        // It's possible that collapsing the take can change the field order.  This disturbs DF's planner and
        // so we must restore it.
        if Self::field_order_differs(&old_output_schema, &new_output_schema) {
            Self::remap_collapsed_output(&old_output_schema, &new_output_schema, collapsed)
        } else {
            collapsed
        }
    }
}

impl PhysicalOptimizerRule for CoalesceTake {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|plan| {
                if let Some(outer_take) = plan.as_any().downcast_ref::<TakeExec>() {
                    let child = outer_take.children()[0];
                    // Case 1: TakeExec -> TakeExec
                    if let Some(inner_take) = child.as_any().downcast_ref::<TakeExec>() {
                        return Ok(Transformed::yes(Self::collapse_takes(
                            inner_take,
                            outer_take,
                            plan.clone(),
                        )));
                    // Case 2: TakeExec -> CoalesceBatchesExec -> TakeExec
                    } else if let Some(exec_child) =
                        child.as_any().downcast_ref::<CoalesceBatchesExec>()
                    {
                        let inner_child = exec_child.children()[0].clone();
                        if let Some(inner_take) = inner_child.as_any().downcast_ref::<TakeExec>() {
                            return Ok(Transformed::yes(Self::collapse_takes(
                                inner_take,
                                outer_take,
                                plan.clone(),
                            )));
                        }
                    }
                }
                Ok(Transformed::no(plan))
            })?
            .data)
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
#[derive(Debug)]
pub struct SimplifyProjection;

impl PhysicalOptimizerRule for SimplifyProjection {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(plan
            .transform_down(|plan| {
                if let Some(proj) = plan.as_any().downcast_ref::<ProjectionExec>() {
                    let children = proj.children();
                    if children.len() != 1 {
                        return Ok(Transformed::no(plan));
                    }

                    let input = children[0];

                    // TODO: we could try to coalesce consecutive projections, something for later
                    // For now, we just keep things simple and only remove NoOp projections

                    // output has different schema, projection needed
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
            })?
            .data)
    }

    fn name(&self) -> &str {
        "simplify_projection"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

pub fn get_physical_optimizer() -> PhysicalOptimizer {
    PhysicalOptimizer::with_rules(vec![
        Arc::new(crate::io::exec::optimizer::CoalesceTake),
        Arc::new(crate::io::exec::optimizer::SimplifyProjection),
    ])
}
