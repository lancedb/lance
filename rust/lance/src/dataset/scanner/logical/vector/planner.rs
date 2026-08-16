// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 4: lowering vector search nodes to execution plans.

use std::sync::Arc;

use datafusion::logical_expr::utils::conjunction;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::prelude::{col, lit};
use lance_core::ROW_ID;
use lance_index::vector::DIST_COL;
use lance_linalg::distance::DistanceType;
use lance_select::mask::RowAddrMask;

use lance_table::format::IndexMetadata;

use super::super::{PrefilterSourceKind, VectorAccessPath, VectorSearchNode};
use crate::Result;
use crate::io::exec::knn::{KnnBatchParams, new_knn_exec};
use crate::io::exec::{KNNVectorDistanceExec, LanceFilterExec, PreFilterSource};

use super::super::planner::sort_asc;

/// Lower a vector search along whichever access path the rules chose.
///
/// Both branches end in the same normalizing projection. That is what makes the node's
/// `[_rowid, _distance]` output contract hold: the two exec trees carry different extra columns
/// (`KNNVectorDistanceExec` appends `_distance` to its input; `ANNSubIndex` emits
/// `[_distance, _rowid]`), and without the projection they would not be interchangeable.
pub fn plan_vector_search(
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
        Some(VectorAccessPath::Flat) | None => {
            plan_flat_knn(node.query(), node.distance_type(), input)?
        }
    };
    normalize_search_output(searched)
}

pub fn plan_indexed_search(
    node: &VectorSearchNode,
    segments: &[IndexMetadata],
    prefilter: PreFilterSource,
) -> Result<Arc<dyn ExecutionPlan>> {
    let query = node.query();
    let block = node
        .overlay_block()
        .map(|rows| RowAddrMask::from_block(rows.as_ref().clone()));
    let fanout = new_knn_exec(node.dataset().clone(), segments, query, prefilter, block)?;

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

/// Brute-force top-`k` by distance over the input. Mirrors `Scanner::flat_knn`, and is shared by
/// the flat access path and by [`VectorRerankNode`] — they are the same computation, differing only
/// in what the caller does with the output schema.
pub fn plan_flat_knn(
    query: &lance_index::vector::Query,
    distance_type: DistanceType,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
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

    let lower = query
        .lower_bound
        .map(|bound| col(DIST_COL).gt_eq(lit(bound)));
    let upper = query.upper_bound.map(|bound| col(DIST_COL).lt(lit(bound)));
    let bounded = match conjunction([lower, upper].into_iter().flatten()) {
        None => distances as Arc<dyn ExecutionPlan>,
        Some(predicate) => Arc::new(LanceFilterExec::try_new(predicate, distances)?),
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
pub fn normalize_search_output(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
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
