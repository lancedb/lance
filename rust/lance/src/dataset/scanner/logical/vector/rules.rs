// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Analyzer and optimizer rules owned by the vector index.

use std::sync::Arc;

use datafusion::common::DataFusionError;
use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{
    EmptyRelation, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{AnalyzerRule, ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::prelude::{col, lit};
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_index::vector::DIST_COL;
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;

use super::super::context::ScanPlanningContext;
use super::super::{LanceTakeNode, PrefilterSourceKind, VectorAccessPath, VectorSearchNode};
use super::super::{analyze_bottom_up, restricts_candidates, scalar_index_prefilter};
use crate::io::exec::knn::QUERY_INDEX_COL;

/// Decide whether each vector search uses its index or brute force.
///
/// Three things can rule the index out, and all of them are decisions rather than mechanics,
/// which is why they belong in a rule and not in the planner:
///
/// * the caller demanded exact results;
/// * there is no vector index on the column;
/// * the index was built with a different metric than the one being queried — using it would
///   answer a different question, so the imperative path falls back to brute force here too.
#[derive(Debug)]
pub struct ResolveVectorAccessPath {
    context: Arc<ScanPlanningContext>,
}

impl ResolveVectorAccessPath {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }

    /// Returns the access path, plus the metric to search with if the index's differs from the
    /// search's current one.
    fn resolve(&self, search: &VectorSearchNode) -> (VectorAccessPath, Option<DistanceType>) {
        if !search.accuracy().is_exact()
            && let Some(index) = self.context.vector_index(&search.query().column)
        {
            if !search.distance_type_requested() || index.metric == search.distance_type() {
                // A delta segment covering none of the fragments this scan will touch has nothing
                // to contribute, so it is dropped from the fan-out rather than searched and
                // discarded. If that leaves nothing, there is no index to search here at all.
                let segments = self.context.reachable_segments(&index.segments);
                if !segments.is_empty() {
                    let adopted = (index.metric != search.distance_type()).then_some(index.metric);
                    return (VectorAccessPath::Index { segments }, adopted);
                }
                return (VectorAccessPath::Flat, None);
            }
            log::warn!(
                "Requested metric {:?} is incompatible with index metric {:?}, falling back to brute-force search",
                search.distance_type(),
                index.metric,
            );
        }
        (VectorAccessPath::Flat, None)
    }
}

impl ResolveVectorAccessPath {
    fn rewrite_node(
        &self,
        plan: LogicalPlan,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(extension) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() else {
            return Ok(Transformed::no(plan));
        };
        // Not an idempotence guard: the builder resolves a search to `Flat` when its candidates
        // are the search space (a vector search over FTS results), and that is a decision, not a
        // default. See `builder::build`.
        if search.access_path_resolution().is_some() {
            return Ok(Transformed::no(plan));
        }

        // `fast_search` means "answer from indices only". With no usable index there is nothing to
        // answer from, so the result is empty rather than a brute-force scan. The partially-indexed
        // case is the same rule applied to one branch, and lives in `SplitPartiallyIndexedSearch`.
        let (resolved, adopted_metric) = self.resolve(search);
        if self.context.fast_search()
            && matches!(resolved, VectorAccessPath::Flat)
            && self.context.vector_index(&search.query().column).is_none()
        {
            return Ok(Transformed::yes(LogicalPlan::EmptyRelation(
                EmptyRelation {
                    produce_one_row: false,
                    schema: search.schema().clone(),
                },
            )));
        }

        let mut resolved = search.clone().with_resolution(resolved);
        if let Some(metric) = adopted_metric {
            resolved = resolved.with_distance_type(metric);
        }
        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(resolved),
        })))
    }
}

impl AnalyzerRule for ResolveVectorAccessPath {
    fn name(&self) -> &str {
        "resolve_vector_access_path"
    }

    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<LogicalPlan> {
        analyze_bottom_up(plan, |node| self.rewrite_node(node))
    }
}

/// Record whether an indexed search has a prefilter, and therefore whether its child plan is a
/// candidate restriction rather than dead weight.
///
/// This is the rule that reads the doc's structural claim off the plan: *a predicate below the
/// search is a prefilter*. Once DataFusion has pushed the predicate into the leaf, "below the
/// search" means "on the child `TableScan`", so both forms are checked.
///
/// It must run after `PushDownFilter`, so that the predicate has reached its final position, and
/// after `ResolveVectorAccessPath`, since only an indexed search has a prefilter source at all —
/// a brute-force search just scans the already-filtered child.
#[derive(Debug)]
pub struct ResolvePrefilterSource {
    context: Arc<ScanPlanningContext>,
}

impl ResolvePrefilterSource {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }

    /// The candidate source for a search whose index covers `segments`.
    fn kind_for(&self, input: &LogicalPlan, segments: &[IndexMetadata]) -> PrefilterSourceKind {
        let required = self.context.segment_coverage(segments);
        scalar_index_prefilter(input, &required, &self.context)
            .unwrap_or_else(|| prefilter_kind(input))
    }
}

impl OptimizerRule for ResolvePrefilterSource {
    fn name(&self) -> &str {
        "resolve_prefilter_source"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(extension) = &plan else {
            return Ok(Transformed::no(plan));
        };
        if let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() {
            let Some(VectorAccessPath::Index { segments }) = search.access_path_resolution() else {
                return Ok(Transformed::no(plan));
            };
            let kind = self.kind_for(search.input(), segments);
            if &kind == search.prefilter() {
                return Ok(Transformed::no(plan));
            }
            return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(search.clone().with_prefilter(kind)),
            })));
        }
        Ok(Transformed::no(plan))
    }
}

pub fn prefilter_kind(input: &LogicalPlan) -> PrefilterSourceKind {
    if restricts_candidates(input) {
        PrefilterSourceKind::ChildRowIds
    } else {
        PrefilterSourceKind::None
    }
}

/// Put the exact re-rank above an indexed search that asked to refine.
///
/// `refine_factor` means "over-fetch `k * factor` approximate candidates, then re-score them
/// exactly and keep the best `k`". The over-fetch is already a lowering detail of the indexed
/// search; the re-scoring is structure, so it belongs here:
///
/// ```text
/// VectorSearch{Flat, k}                       <- exact re-rank
///   Projection [_rowid, vec]
///     LanceTake [vec]
///       VectorSearch{Index, k, refine_factor} <- over-fetches k * factor
/// ```
///
/// That is the same subtree [`SplitPartiallyIndexedSearch`] builds, which is the point: "re-rank
/// approximate candidates exactly" is one operation in the logical layer, and both the refine knob
/// and partial index coverage are reasons to reach for it.
#[derive(Debug)]
pub struct ExpandVectorRefine {
    context: Arc<ScanPlanningContext>,
}

impl ExpandVectorRefine {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl AnalyzerRule for ExpandVectorRefine {
    fn name(&self) -> &str {
        "expand_vector_refine"
    }

    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<LogicalPlan> {
        analyze_bottom_up(plan, |node| self.rewrite_node(node))
    }
}

impl ExpandVectorRefine {
    fn rewrite_node(
        &self,
        plan: LogicalPlan,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(extension) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() else {
            return Ok(Transformed::no(plan));
        };
        // Only an indexed search produces approximate distances worth refining.
        if search.query().refine_factor.is_none()
            || !matches!(
                search.access_path_resolution(),
                Some(VectorAccessPath::Index { .. })
            )
        {
            return Ok(Transformed::no(plan));
        }

        let column = &search.query().column;
        let approximate = LogicalPlan::Extension(Extension {
            node: Arc::new(search.clone()),
        });
        let candidates = LogicalPlanBuilder::new(LogicalPlan::Extension(Extension {
            node: Arc::new(LanceTakeNode::try_new(
                approximate,
                search.dataset().clone(),
                search
                    .dataset()
                    .empty_projection()
                    .union_column(column, OnMissing::Error)?,
                self.context.take_settings().clone(),
            )?),
        }))
        // Drop the approximate `_distance` so the re-rank does not produce a second one.
        .project(vec![col(ROW_ID), col(column)])?
        .build()?;

        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(
                search
                    .clone()
                    .with_input(candidates)
                    .with_resolution(VectorAccessPath::Flat)
                    .with_prefilter(PrefilterSourceKind::None),
            ),
        })))
    }
}

/// Answer a batch of query vectors against an index by asking each one separately.
///
/// The index fanout searches one query vector at a time, so a batch of `n` becomes a union of `n`
/// searches over the same input, each tagged with the `query_index` that identifies which query it
/// answers:
///
/// ```text
/// Union
///   Projection [0 AS query_index, _rowid, _distance]
///     VectorSearch{Index, key=queries[0]}
///   Projection [1 AS query_index, _rowid, _distance]
///     VectorSearch{Index, key=queries[1]}
///   ...
/// ```
///
/// Expanding here rather than in the planner means every later rule — partial index coverage,
/// refine — sees ordinary single-query searches and needs to know nothing about batching. That is
/// also what the imperative path does, by re-entering `vector_search` once per query vector.
///
/// A brute-force batch search is not expanded: `KNNVectorDistanceExec` scores all `n` queries in
/// one pass over the rows, which is the whole reason to send them as a batch.
#[derive(Debug, Default)]
pub struct ExpandBatchSearch;

impl AnalyzerRule for ExpandBatchSearch {
    fn name(&self) -> &str {
        "expand_batch_search"
    }

    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<LogicalPlan> {
        analyze_bottom_up(plan, Self::rewrite_node)
    }
}

impl ExpandBatchSearch {
    fn rewrite_node(plan: LogicalPlan) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(extension) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() else {
            return Ok(Transformed::no(plan));
        };
        if search.batch_queries().is_none()
            || !matches!(
                search.access_path_resolution(),
                Some(VectorAccessPath::Index { .. })
            )
        {
            return Ok(Transformed::no(plan));
        }

        let mut branches = Vec::with_capacity(search.query_count());
        for query_index in 0..search.query_count() {
            let single = LogicalPlan::Extension(Extension {
                node: Arc::new(search.single_query(query_index)?),
            });
            branches.push(
                LogicalPlanBuilder::new(single)
                    .project(vec![
                        lit(query_index as i32).alias(QUERY_INDEX_COL),
                        col(ROW_ID),
                        col(DIST_COL),
                    ])?
                    .build()?,
            );
        }

        let mut branches = branches.into_iter();
        let mut unioned = LogicalPlanBuilder::new(branches.next().ok_or_else(|| {
            DataFusionError::Internal("a batch search has at least one query".into())
        })?);
        for branch in branches {
            unioned = unioned.union(branch)?;
        }
        Ok(Transformed::yes(unioned.build()?))
    }
}
