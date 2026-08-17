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

use super::{LanceTakeNode, PrefilterSourceKind, VectorRerankNode, VectorSearchNode};
use super::{plan_flat_knn, plan_take, plan_vector_search};
use crate::Result;
use crate::dataset::Dataset;
use crate::io::exec::PreFilterSource;
use crate::io::exec::scalar_index::ScalarIndexExec;

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
                None,
                input,
            )?));
        }
        if let Some(take) = node.as_any().downcast_ref::<LanceTakeNode>() {
            return Ok(Some(plan_take(take, input)?));
        }
        Ok(None)
    }
}

/// Lower a search's candidate restriction.
///
/// `input` is the child plan. It is only read for [`PrefilterSourceKind::ChildRowIds`]; the other
/// two answer without it, and the child is planned but never executed.
pub fn plan_prefilter_source(
    kind: &PrefilterSourceKind,
    dataset: &Arc<Dataset>,
    input: Arc<dyn ExecutionPlan>,
) -> PreFilterSource {
    match kind {
        PrefilterSourceKind::None => PreFilterSource::None,
        PrefilterSourceKind::ChildRowIds => PreFilterSource::FilteredRowIds(input),
        PrefilterSourceKind::ScalarIndexQuery {
            query,
            result_format,
        } => PreFilterSource::ScalarIndexQuery(Arc::new(ScalarIndexExec::new(
            dataset.clone(),
            query.as_ref().clone(),
            *result_format,
        ))),
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
