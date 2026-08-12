// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 3: Lance-owned optimizer rules.
//!
//! Each rule holds an [`Arc<ScanPlanningContext>`], which is the whole point of the staging: a
//! synchronous `OptimizerRule` gets to make decisions that needed I/O to inform.
//!
//! These rules are logical-to-logical. They resolve *which* access path a search will use rather
//! than building it, so that by the time the extension planner runs there is nothing left to
//! decide. That is the bet this spike is testing — that index reasoning fits in rules, and
//! lowering can stay mechanical.

use std::any::Any;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::logical_expr::{Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::prelude::col;
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_table::format::Fragment;

use super::context::ScanPlanningContext;
use super::nodes::{LanceTakeNode, PrefilterSourceKind, VectorAccessPath, VectorSearchNode};
use super::source::LanceScanSource;

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

    fn resolve(&self, search: &VectorSearchNode) -> VectorAccessPath {
        if !search.accuracy().is_exact()
            && let Some(index) = self.context.vector_index(&search.query().column)
        {
            if index.metric == search.distance_type() {
                return VectorAccessPath::Index {
                    segments: index.segments.clone(),
                };
            }
            log::warn!(
                "Requested metric {:?} is incompatible with index metric {:?}, falling back to brute-force search",
                search.distance_type(),
                index.metric,
            );
        }
        VectorAccessPath::Flat
    }
}

impl OptimizerRule for ResolveVectorAccessPath {
    fn name(&self) -> &str {
        "resolve_vector_access_path"
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
        let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() else {
            return Ok(Transformed::no(plan));
        };
        if search.access_path_resolution().is_some() {
            return Ok(Transformed::no(plan));
        }

        let resolved = search.clone().with_resolution(self.resolve(search));
        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(resolved),
        })))
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
#[derive(Debug, Default)]
pub struct ResolvePrefilterSource;

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
        let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() else {
            return Ok(Transformed::no(plan));
        };
        if !matches!(
            search.access_path_resolution(),
            Some(VectorAccessPath::Index { .. })
        ) {
            return Ok(Transformed::no(plan));
        }

        let kind = if restricts_candidates(search.input()) {
            PrefilterSourceKind::ChildRowIds
        } else {
            PrefilterSourceKind::None
        };
        if kind == search.prefilter() {
            return Ok(Transformed::no(plan));
        }

        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(search.clone().with_prefilter(kind)),
        })))
    }
}

/// Split a search over partially-indexed data into an indexed branch and a brute-force branch.
///
/// This is the design doc's Appendix A, expressed as a logical rewrite rather than as fan-out
/// inside the extension planner. A search whose index does not cover every fragment becomes:
///
/// ```text
/// VectorSearch{Flat, k}              <- exact re-rank over the merged candidates
///   Projection [_rowid, vec]         <- drop the branches' approximate _distance
///     LanceTake [vec]                <- fetch vectors for the candidates
///       Union
///         VectorSearch{Index}  over the indexed fragments
///         VectorSearch{Flat}   over the unindexed fragments
/// ```
///
/// Two things fall out of writing it this way. The outer exact search *is* the re-rank — the
/// same node type, no special-purpose operator — which is the structural version of the doc's
/// observation that the combined path reuses the refine mechanism. And the single `Filter` child
/// is duplicated onto both branches by ordinary plan surgery, so the "one predicate, two physical
/// placements" problem is solved once here instead of inside the planner.
#[derive(Debug)]
pub struct SplitPartiallyIndexedSearch {
    context: Arc<ScanPlanningContext>,
}

impl SplitPartiallyIndexedSearch {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl OptimizerRule for SplitPartiallyIndexedSearch {
    fn name(&self) -> &str {
        "split_partially_indexed_search"
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
        let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() else {
            return Ok(Transformed::no(plan));
        };
        // Only an indexed search can be partially covered; a flat search already reads everything.
        // This is also what stops the rewrite from recursing into the exact re-rank it produces.
        if !matches!(
            search.access_path_resolution(),
            Some(VectorAccessPath::Index { .. })
        ) || search.input_fully_indexed()
        {
            return Ok(Transformed::no(plan));
        }

        let column = &search.query().column;
        let Some(unindexed) = self.context.unindexed_fragments(column) else {
            return Ok(Transformed::no(plan));
        };
        if unindexed.is_empty() {
            return Ok(Transformed::no(plan));
        }
        let Some(indexed) = self.context.indexed_fragments(column) else {
            return Ok(Transformed::no(plan));
        };

        let indexed_branch = LogicalPlan::Extension(Extension {
            node: Arc::new(
                search
                    .clone()
                    .with_input(restrict_scan(search.input(), indexed)?)
                    .covering_only_indexed_input(),
            ),
        });
        let flat_branch = LogicalPlan::Extension(Extension {
            node: Arc::new(
                search
                    .clone()
                    .with_input(restrict_scan(search.input(), unindexed)?)
                    .with_resolution(VectorAccessPath::Flat),
            ),
        });

        let merged = LogicalPlanBuilder::new(indexed_branch)
            .union(flat_branch)?
            .build()?;
        let with_vectors = LogicalPlan::Extension(Extension {
            node: Arc::new(LanceTakeNode::try_new(
                merged,
                search.dataset().clone(),
                search
                    .dataset()
                    .empty_projection()
                    .union_column(column, OnMissing::Error)?,
            )?),
        });
        // The branches' `_distance` values are per-branch and, for the indexed one, approximate.
        // Dropping the column here both discards them and keeps the re-rank from producing a
        // duplicate `_distance`.
        let candidates = LogicalPlanBuilder::new(with_vectors)
            .project(vec![col(ROW_ID), col(column)])?
            .build()?;

        let rerank = search
            .clone()
            .with_input(candidates)
            .with_resolution(VectorAccessPath::Flat)
            .with_prefilter(PrefilterSourceKind::None);
        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(rerank),
        })))
    }
}

/// Rebuild `plan`, pointing every scan leaf at the same source restricted to `fragments`.
///
/// The recursion exists because the predicate may not have reached the leaf: before
/// `PushDownFilter` runs there is a `Filter` in between, and that `Filter` has to be duplicated
/// onto both branches along with the scan.
fn restrict_scan(
    plan: &LogicalPlan,
    fragments: Vec<Fragment>,
) -> datafusion::common::Result<LogicalPlan> {
    let fragments = Arc::new(fragments);
    Ok(plan
        .clone()
        .transform_down(|node| {
            let LogicalPlan::TableScan(scan) = &node else {
                return Ok(Transformed::no(node));
            };
            let Some(provider) = source_as_provider(&scan.source).ok() else {
                return Ok(Transformed::no(node));
            };
            let Some(source) = (provider.as_ref() as &dyn Any).downcast_ref::<LanceScanSource>()
            else {
                return Ok(Transformed::no(node));
            };
            let mut scan = scan.clone();
            scan.source =
                provider_as_source(Arc::new(source.restricted_to(fragments.as_ref().clone())));
            Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
        })?
        .data)
}

fn restricts_candidates(plan: &LogicalPlan) -> bool {
    let mut restricts = false;
    let _ = plan.apply(|node| {
        match node {
            LogicalPlan::Filter(_) => restricts = true,
            LogicalPlan::TableScan(scan) if !scan.filters.is_empty() => restricts = true,
            _ => {}
        }
        if restricts {
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    });
    restricts
}
