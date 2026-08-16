// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 4: dispatching each Lance logical node to its lowering.
//!
//! The dispatch itself is mechanical. Every decision was made by a rule already; by the time a
//! node reaches here it names exactly one execution strategy.

use std::sync::Arc;

use arrow_schema::SortOptions;
use async_trait::async_trait;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::expressions;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion_physical_expr::PhysicalSortExpr;

use super::row_offset::{RowOffsetNode, plan_row_offset};
use super::{LanceTakeNode, VectorRerankNode, VectorSearchNode};
use super::{plan_flat_knn, plan_take, plan_vector_search};
use crate::Result;

#[derive(Debug, Default)]
pub struct LanceExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for LanceExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> datafusion::common::Result<Option<Arc<dyn ExecutionPlan>>> {
        // FTS owns its own nodes, rules, and lowering; ask that module first.
        if let Some(plan) = super::fts::plan_extension(node, physical_inputs) {
            return Ok(Some(plan?));
        }

        let input = physical_inputs.first().cloned().ok_or_else(|| {
            datafusion::common::DataFusionError::Internal(
                "Lance logical nodes always have exactly one input".into(),
            )
        })?;

        if let Some(search) = node.as_any().downcast_ref::<VectorSearchNode>() {
            return Ok(Some(plan_vector_search(search, input)?));
        }
        if let Some(rerank) = node.as_any().downcast_ref::<VectorRerankNode>() {
            return Ok(Some(plan_flat_knn(
                rerank.query(),
                rerank.distance_type(),
                1,
                input,
            )?));
        }
        if let Some(take) = node.as_any().downcast_ref::<LanceTakeNode>() {
            return Ok(Some(plan_take(take, input)?));
        }
        if let Some(offsets) = node.as_any().downcast_ref::<RowOffsetNode>() {
            return Ok(Some(plan_row_offset(offsets, input)?));
        }
        Ok(None)
    }
}

pub fn sort_asc(column: &str, plan: &dyn ExecutionPlan) -> Result<PhysicalSortExpr> {
    Ok(PhysicalSortExpr {
        expr: expressions::col(column, plan.schema().as_ref())?,
        options: SortOptions {
            descending: false,
            nulls_first: false,
        },
    })
}
