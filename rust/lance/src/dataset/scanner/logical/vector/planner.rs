// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 4: lowering vector search nodes to execution plans.

use std::sync::Arc;

use arrow_schema::DataType;
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

use super::super::{VectorAccessPath, VectorSearchNode};
use crate::Result;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;

use crate::dataset::scanner::DEFAULT_XTR_OVERFETCH;
use crate::index::vector::utils::{get_vector_dim, get_vector_type};
use crate::io::exec::knn::{KnnBatchParams, MultivectorScoringExec, QUERY_INDEX_COL, new_knn_exec};
use crate::io::exec::{KNNVectorDistanceExec, LanceFilterExec, PreFilterSource};

use super::super::planner::{plan_prefilter_source, sort_asc};

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
            // The child is already a `_rowid`-only read carrying the predicate — see
            // `VectorSearchNode::necessary_children_exprs`.
            let prefilter = plan_prefilter_source(node.prefilter(), node.dataset(), input);
            plan_indexed_search(node, segments, prefilter)?
        }
        // An unresolved node means the rule did not run; brute force is the answer that is
        // always correct, so it is the safe default rather than an error.
        Some(VectorAccessPath::Flat) | None => plan_flat_knn(
            node.query(),
            node.distance_type(),
            node.batch_queries(),
            input,
        )?,
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
    let (vector_type, _) = get_vector_type(node.dataset().schema(), &query.column)?;
    let fanout = match vector_type {
        // A multivector row holds many vectors, so one row is scored from several fanouts at once
        // rather than one. Mirrors `Scanner::multivec_ann`.
        DataType::List(_) => plan_multivector_fanout(node, segments, prefilter, block)?,
        _ => new_knn_exec(node.dataset().clone(), segments, query, prefilter, block)?,
    };

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

/// Search each of a multivector query's vectors separately, then score rows across the results.
///
/// The per-vector searches deliberately over-fetch: XTR scores a row from whichever of its vectors
/// were retrieved, so recall depends on each search reaching deep enough to find them.
fn plan_multivector_fanout(
    node: &VectorSearchNode,
    segments: &[IndexMetadata],
    prefilter: PreFilterSource,
    block: Option<RowAddrMask>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let query = node.query();
    let dim = get_vector_dim(node.dataset().schema(), &query.column)?;
    let over_fetch = *DEFAULT_XTR_OVERFETCH;

    let mut searches = Vec::with_capacity(query.key.len() / dim);
    for offset in (0..query.key.len()).step_by(dim) {
        let mut single = query.clone();
        single.key = query.key.slice(offset, dim);
        // XTR scores from the retrieved vectors themselves, so there is nothing to refine against
        // — the factor is purely how deep each search goes.
        single.refine_factor = Some(over_fetch);
        let fanout = new_knn_exec(
            node.dataset().clone(),
            segments,
            &single,
            prefilter.clone(),
            block.clone(),
        )?;
        searches.push(Arc::new(
            SortExec::new(
                [
                    sort_asc(DIST_COL, fanout.as_ref())?,
                    sort_asc(ROW_ID, fanout.as_ref())?,
                ]
                .into(),
                fanout,
            )
            .with_fetch(Some(query.k * over_fetch as usize)),
        ) as Arc<dyn ExecutionPlan>);
    }

    Ok(Arc::new(MultivectorScoringExec::try_new(
        searches,
        query.clone(),
    )?))
}

/// Brute-force top-`k` by distance over the input. Mirrors `Scanner::flat_knn`, and is shared by
/// the flat access path and by [`VectorRerankNode`] — they are the same computation, differing only
/// in what the caller does with the output schema.
pub fn plan_flat_knn(
    query: &lance_index::vector::Query,
    distance_type: DistanceType,
    batch_queries: Option<usize>,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    // A batch scores every query vector in one pass over the rows, so it has to see all of them:
    // its top-`k` per query cannot be assembled from independent per-partition top-`k`s.
    let is_batch = batch_queries.is_some();
    let input = match is_batch {
        true => Arc::new(CoalescePartitionsExec::new(input)) as Arc<dyn ExecutionPlan>,
        false => input,
    };
    let distances = Arc::new(KNNVectorDistanceExec::try_new_batch(
        input,
        &query.column,
        query.key.clone(),
        KnnBatchParams {
            is_batch,
            query_count: batch_queries.unwrap_or(1),
            k: query.k,
            lower_bound: query.lower_bound,
            upper_bound: query.upper_bound,
            distance_type,
            // The take above the search refetches whatever the user projected, so keeping the
            // vector here would only widen the batches this node emits.
            retain_vector: false,
        },
    )?);

    // The batch node applies the distance bounds and the per-query top-`k` itself, and drops rows
    // with no vector on the way. There is nothing left for the tail below to do.
    if is_batch {
        return Ok(distances);
    }

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

/// Trim whatever the access path carried down to the node's declared output columns.
pub fn normalize_search_output(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let mut columns = Vec::with_capacity(3);
    // Only a batch search has this, and only the brute-force path produces it here — the indexed
    // path gets it from the projection `ExpandBatchSearch` puts above each expanded search.
    if schema.column_with_name(QUERY_INDEX_COL).is_some() {
        columns.push(QUERY_INDEX_COL);
    }
    columns.extend([ROW_ID, DIST_COL]);
    Ok(Arc::new(ProjectionExec::try_new(
        columns
            .into_iter()
            .map(|name| Ok((expressions::col(name, schema.as_ref())?, name.to_string())))
            .collect::<Result<Vec<_>>>()?,
        plan,
    )?))
}
