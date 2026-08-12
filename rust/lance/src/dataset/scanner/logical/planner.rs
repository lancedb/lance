// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 4: lower Lance logical nodes to the exec nodes the read path already has.
//!
//! Everything that needs a judgment call is expected to have happened in the rules by the time
//! this runs, so the translation here is mechanical. It is still `async` — DataFusion's
//! `ExtensionPlanner` trait is — which means the split enforced by the staging is not a language
//! constraint but a deliberate one: keeping I/O out of here is what lets the same decisions be
//! made by synchronous optimizer rules.

use std::sync::Arc;

use arrow_schema::SortOptions;
use async_trait::async_trait;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::{col, lit};
use datafusion_physical_expr::PhysicalSortExpr;
use lance_core::{ROW_ID, datatypes::Projection};
use lance_index::vector::DIST_COL;

use lance_table::format::IndexMetadata;

use super::nodes::{LanceTakeNode, PrefilterSourceKind, VectorAccessPath, VectorSearchNode};
use crate::Result;
use crate::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};
use crate::io::exec::knn::{KnnBatchParams, new_knn_exec};
use crate::io::exec::{KNNVectorDistanceExec, LanceFilterExec, PreFilterSource};

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
        let input = physical_inputs.first().cloned().ok_or_else(|| {
            datafusion::common::DataFusionError::Internal(
                "Lance logical nodes always have exactly one input".into(),
            )
        })?;

        if let Some(search) = node.as_any().downcast_ref::<VectorSearchNode>() {
            return Ok(Some(plan_vector_search(search, input)?));
        }
        if let Some(take) = node.as_any().downcast_ref::<LanceTakeNode>() {
            return Ok(Some(plan_take(take, input)?));
        }
        Ok(None)
    }
}

/// Lower a vector search along whichever access path the rules chose.
///
/// Both branches end in the same normalizing projection. That is what makes the node's
/// `[_rowid, _distance]` output contract hold: the two exec trees carry different extra columns
/// (`KNNVectorDistanceExec` appends `_distance` to its input; `ANNSubIndex` emits
/// `[_distance, _rowid]`), and without the projection they would not be interchangeable.
fn plan_vector_search(
    node: &VectorSearchNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let searched = match node.access_path_resolution() {
        Some(VectorAccessPath::Index { segments }) => {
            let prefilter = match node.prefilter() {
                PrefilterSourceKind::None => PreFilterSource::None,
                // The child is already a `_rowid`-only read carrying the predicate — see
                // `VectorSearchNode::necessary_children_exprs`.
                PrefilterSourceKind::ChildRowIds => PreFilterSource::FilteredRowIds(input),
            };
            plan_indexed_search(node, segments, prefilter)?
        }
        // An unresolved node means the rule did not run; brute force is the answer that is
        // always correct, so it is the safe default rather than an error.
        Some(VectorAccessPath::Flat) | None => plan_flat_search(node, input)?,
    };
    normalize_search_output(searched)
}

fn plan_indexed_search(
    node: &VectorSearchNode,
    segments: &[IndexMetadata],
    prefilter: PreFilterSource,
) -> Result<Arc<dyn ExecutionPlan>> {
    let query = node.query();
    let fanout = new_knn_exec(node.dataset().clone(), segments, query, prefilter, None)?;

    // Over-fetch when refining: the extra candidates are what the exact re-rank chooses from.
    let fetch = query.k * query.refine_factor.unwrap_or(1) as usize;
    Ok(Arc::new(
        SortExec::new(
            [
                sort_asc(DIST_COL, fanout.as_ref())?,
                sort_asc(ROW_ID, fanout.as_ref())?,
            ]
            .into(),
            fanout,
        )
        .with_fetch(Some(fetch)),
    ))
}

fn plan_flat_search(
    node: &VectorSearchNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let query = node.query();
    let distance_type = node.distance_type();

    let distances = Arc::new(KNNVectorDistanceExec::try_new_batch(
        input,
        &query.column,
        query.key.clone(),
        KnnBatchParams {
            is_batch: false,
            query_count: 1,
            k: query.k,
            lower_bound: query.lower_bound,
            upper_bound: query.upper_bound,
            distance_type,
            retain_vector: false,
        },
    )?);

    let bounded = match (query.lower_bound, query.upper_bound) {
        (None, None) => distances as Arc<dyn ExecutionPlan>,
        (lower, upper) => {
            let mut predicate = None;
            if let Some(lower) = lower {
                predicate = Some(col(DIST_COL).gt_eq(lit(lower)));
            }
            if let Some(upper) = upper {
                let clause = col(DIST_COL).lt(lit(upper));
                predicate = Some(match predicate {
                    Some(existing) => existing.and(clause),
                    None => clause,
                });
            }
            Arc::new(LanceFilterExec::try_new(
                predicate.expect("at least one bound is set"),
                distances,
            )?)
        }
    };

    let sorted = Arc::new(
        SortExec::new(
            [
                sort_asc(DIST_COL, bounded.as_ref())?,
                sort_asc(ROW_ID, bounded.as_ref())?,
            ]
            .into(),
            bounded,
        )
        .with_fetch(Some(query.k)),
    );

    // A null distance means the row had no vector; it is not a neighbor at any rank.
    Ok(Arc::new(LanceFilterExec::try_new(
        col(DIST_COL).is_not_null(),
        sorted,
    )?))
}

/// Trim whatever the access path carried down to the node's declared `[_rowid, _distance]`.
fn normalize_search_output(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    Ok(Arc::new(ProjectionExec::try_new(
        vec![
            (
                expressions::col(ROW_ID, schema.as_ref())?,
                ROW_ID.to_string(),
            ),
            (
                expressions::col(DIST_COL, schema.as_ref())?,
                DIST_COL.to_string(),
            ),
        ],
        plan,
    )?))
}

fn plan_take(
    node: &LanceTakeNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(FilteredReadExec::try_new(
        node.dataset().clone(),
        take_options(node.projection().clone(), input.schema().as_ref()),
        Some(input),
    )?))
}

/// Carry through whichever identity columns the input already has, so the take does not drop
/// them. Mirrors `Scanner::take_current`.
fn take_options(
    mut projection: Projection,
    input_schema: &arrow_schema::Schema,
) -> FilteredReadOptions {
    projection.with_row_id |= input_schema.column_with_name(ROW_ID).is_some();
    projection.with_row_addr |= input_schema
        .column_with_name(lance_core::ROW_ADDR)
        .is_some();
    FilteredReadOptions::new(projection)
}

fn sort_asc(column: &str, plan: &dyn ExecutionPlan) -> Result<PhysicalSortExpr> {
    Ok(PhysicalSortExpr {
        expr: expressions::col(column, plan.schema().as_ref())?,
        options: SortOptions {
            descending: false,
            nulls_first: false,
        },
    })
}
