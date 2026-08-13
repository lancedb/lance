// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 3: Lance-owned logical rules.
//!
//! Each rule holds an [`Arc<ScanPlanningContext>`], which is the whole point of the staging: a
//! synchronous rule gets to make decisions that needed I/O to inform.
//!
//! These rules are logical-to-logical. They resolve *which* access path a search will use rather
//! than building it, so that by the time the extension planner runs there is nothing left to
//! decide. That is the bet this spike is testing — that index reasoning fits in rules, and
//! lowering can stay mechanical.
//!
//! Most of them are [`AnalyzerRule`]s, not `OptimizerRule`s. DataFusion documents an
//! `OptimizerRule` as computing "the same results, but in a potentially more efficient way", and
//! directs semantics-changing rewrites to the analyzer. These rules are not optional: skip access
//! path resolution or the coverage split and a search silently returns the wrong rows. The
//! analyzer also runs each rule exactly once, where the optimizer re-runs rules until the plan
//! stops changing — which is why none of these rules needs to recognize its own output.
//!
//! Three of them are exceptions, and instructively so: [`ResolvePrefilterSource`],
//! [`ResolveScalarIndexQuery`], and the scan-leaf half of [`SplitOnIndexCoverage`] are equally
//! mandatory but each has to observe the plan *after* `PushDownFilter` has moved the predicate —
//! the first asks "is there a predicate below the search?", the other two ask "which predicates
//! reached the leaf?". DataFusion's phase order is mandatory-then-optional; these are
//! optional-then-mandatory, so they run in the optimizer and carry their own idempotence guards.

use std::any::Any;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchemaRef, TableReference};
use datafusion::config::ConfigOptions;
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::logical_expr::expr_rewriter::unnormalize_cols;
use datafusion::logical_expr::{
    EmptyRelation, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{AnalyzerRule, ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::prelude::col;
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_select::mask::RowAddrTreeMap;
use lance_table::format::Fragment;

use super::context::{OverlayStaleness, ScanPlanningContext};
use super::fts::{FtsAccessPath, FtsLeafNode};
use super::nodes::{LanceTakeNode, PrefilterSourceKind, VectorAccessPath, VectorSearchNode};
use super::source::{LanceScanSource, ScanRestriction};

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
                    // A delta segment covering none of the fragments this scan will touch has
                    // nothing to contribute, so it is dropped from the fan-out rather than
                    // searched and discarded.
                    segments: self.context.reachable_segments(&index.segments),
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
        let resolved = self.resolve(search);
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

        let resolved = search.clone().with_resolution(resolved);
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

/// Apply a node rewrite bottom-up over the whole plan.
///
/// An [`AnalyzerRule`] receives the entire plan, so the traversal that
/// [`OptimizerRule::apply_order`] used to supply is written out here instead. Bottom-up matters for
/// the same reason it did there: a rule that replaces a node with a subtree must not then descend
/// into the subtree it just built.
pub(super) fn analyze_bottom_up(
    plan: LogicalPlan,
    rewrite: impl FnMut(LogicalPlan) -> datafusion::common::Result<Transformed<LogicalPlan>>,
) -> datafusion::common::Result<LogicalPlan> {
    Ok(plan.transform_up(rewrite)?.data)
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
        if let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>() {
            if !matches!(
                search.access_path_resolution(),
                Some(VectorAccessPath::Index { .. })
            ) {
                return Ok(Transformed::no(plan));
            }
            let kind = prefilter_kind(search.input());
            if kind == search.prefilter() {
                return Ok(Transformed::no(plan));
            }
            return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(search.clone().with_prefilter(kind)),
            })));
        }
        if let Some(leaf) = extension.node.as_any().downcast_ref::<FtsLeafNode>() {
            if !matches!(leaf.resolution(), Some(FtsAccessPath::Index { .. })) {
                return Ok(Transformed::no(plan));
            }
            let kind = prefilter_kind(leaf.input());
            if kind == leaf.prefilter() {
                return Ok(Transformed::no(plan));
            }
            return Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(leaf.clone().with_prefilter(kind)),
            })));
        }
        Ok(Transformed::no(plan))
    }
}

pub(super) fn prefilter_kind(input: &LogicalPlan) -> PrefilterSourceKind {
    if restricts_candidates(input) {
        PrefilterSourceKind::ChildRowIds
    } else {
        PrefilterSourceKind::None
    }
}

/// What an index does not answer for, among the rows a scan will touch.
///
/// The whole point of this type is that fragment-level and row-level coverage are the same
/// statement. `SplitPartiallyIndexedSearch` and `SplitPartiallyIndexedFts` in the imperative path
/// split on fragment coverage; the overlay stale-row handling threaded through
/// `Scanner::new_filtered_read`, `knn_combined`, and `plan_flat_match_query` splits on row
/// coverage. Both produce a brute-force branch reading exactly the rows in the hole, and the only
/// difference is how that branch's scan is narrowed — which is what [`ScanRestriction`] carries.
pub(super) enum IndexCoverage {
    /// The index answers for every row the scan will touch. Also the answer when coverage cannot
    /// be determined: an index that will not say what it covers is used as-is, which is what the
    /// imperative path does too.
    Complete,
    /// The index cannot be trusted for any row.
    Unusable,
    /// The indexed branch reads `indexed` and must not emit `block`; the brute-force branch reads
    /// the rows described by `gaps`.
    ///
    /// `indexed: None` means the indexed branch reads everything the node already reads — the
    /// scan-leaf case, where the hole is entirely at row level and there are no fragments to
    /// exclude.
    Partial {
        indexed: Option<Vec<Fragment>>,
        gaps: Vec<ScanRestriction>,
        block: Option<Arc<RowAddrTreeMap>>,
    },
}

impl IndexCoverage {
    /// Drop the brute-force branch, keeping the block mask.
    ///
    /// This is `fast_search`: "answer from indices only" means a coverage gap is not filled, it is
    /// simply not answered — and a stale entry is not an answer either, so the block stays.
    fn indexed_only(self) -> Self {
        match self {
            Self::Partial {
                indexed,
                gaps: _,
                block,
            } => Self::Partial {
                indexed,
                gaps: Vec::new(),
                block,
            },
            other => other,
        }
    }
}

/// A search node that answers from an index and can fall back to brute force for the rows its
/// index does not cover.
///
/// The trait is the seam that lets [`SplitOnIndexCoverage`] be one rule: what counts as coverage,
/// and how the branches merge back together, are per-index-kind decisions, while the split itself
/// is not.
pub(super) trait SplittableSearch {
    /// `false` when this node is already brute force, or is already the output of a split.
    fn is_splittable(&self) -> bool;
    fn coverage(&self, context: &ScanPlanningContext) -> IndexCoverage;
    fn input(&self) -> &LogicalPlan;
    /// Whether `fast_search` may drop this node's brute-force branch.
    ///
    /// True for a search, where `fast_search` is the user trading recall for latency. False for a
    /// scan's predicate: there the brute-force branch re-reads rows whose index entry an overlay
    /// invalidated, which repairs the index result rather than extending it past the index.
    fn honors_fast_search(&self) -> bool {
        true
    }
    /// This node reading only indexed rows, with `block` withheld from the index result.
    fn indexed_branch(&self, input: LogicalPlan, block: Option<Arc<RowAddrTreeMap>>)
    -> LogicalPlan;
    /// This node as a brute-force search over `input`.
    fn flat_branch(&self, input: LogicalPlan) -> LogicalPlan;
    /// Combine the branches into one plan producing this node's schema.
    fn merge(
        &self,
        indexed: LogicalPlan,
        flat: LogicalPlan,
        context: &ScanPlanningContext,
    ) -> datafusion::common::Result<LogicalPlan>;
}

/// Recognize the node kinds that can be split.
///
/// This downcast list is the one place the shared rule has to know which index kinds exist, and it
/// is exactly the registry a plugin system would own instead.
fn splittable(plan: &LogicalPlan) -> Option<&dyn SplittableSearch> {
    let LogicalPlan::Extension(extension) = plan else {
        return None;
    };
    let node = extension.node.as_any();
    if let Some(search) = node.downcast_ref::<VectorSearchNode>() {
        return Some(search);
    }
    if let Some(leaf) = node.downcast_ref::<FtsLeafNode>() {
        return Some(leaf);
    }
    None
}

/// Split a search into an indexed branch and a brute-force branch over the rows the index does not
/// answer for.
///
/// This is the design doc's Appendix A, generalized: a rewrite rather than fan-out inside the
/// extension planner, and driven by [`IndexCoverage`] rather than by fragment lists alone. For a
/// vector search it produces
///
/// ```text
/// VectorSearch{Flat, k}              <- exact re-rank over the merged candidates
///   Projection [_rowid, vec]         <- drop the branches' approximate _distance
///     LanceTake [vec]                <- fetch vectors for the candidates
///       Union
///         VectorSearch{Index}  over the indexed fragments, minus the stale rows
///         VectorSearch{Flat}   over the union of the coverage gaps
/// ```
///
/// Three things fall out of writing it this way. The outer exact search *is* the re-rank — the
/// same node type, no special-purpose operator. The single `Filter` child is duplicated onto both
/// branches by ordinary plan surgery, so the "one predicate, two physical placements" problem is
/// solved once here instead of inside the planner. And a data overlay's stale rows need no
/// machinery of their own: they are one more coverage gap, and the same union absorbs them.
#[derive(Debug)]
pub struct SplitOnIndexCoverage {
    context: Arc<ScanPlanningContext>,
    scope: SplitScope,
}

/// Which node kind a [`SplitOnIndexCoverage`] instance is responsible for.
///
/// The rule is registered twice, in two different stages, because the two kinds of coverage become
/// knowable at different times. A search node carries its index from the builder, so its coverage
/// is settled before any optimization. A scan's index query does not exist until `PushDownFilter`
/// has decided which predicates reach the leaf — so that half cannot run in the analyzer, however
/// mandatory it is. Same rewrite, same trait, different stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitScope {
    /// Vector and full-text search nodes. Runs in the analyzer.
    Searches,
    /// Scan leaves whose scalar index query has been resolved. Runs in the optimizer.
    Scans,
}

impl SplitOnIndexCoverage {
    pub fn searches(context: Arc<ScanPlanningContext>) -> Self {
        Self {
            context,
            scope: SplitScope::Searches,
        }
    }

    pub fn scans(context: Arc<ScanPlanningContext>) -> Self {
        Self {
            context,
            scope: SplitScope::Scans,
        }
    }

    fn rewrite_node(
        &self,
        plan: LogicalPlan,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let split = match self.scope {
            SplitScope::Searches => match splittable(&plan) {
                Some(node) => self.split(node, plan.schema())?,
                None => None,
            },
            SplitScope::Scans => match ScanCoverage::of(&plan, &self.context) {
                Some(node) => self.split(&node, plan.schema())?,
                None => None,
            },
        };
        Ok(match split {
            Some(split) => Transformed::yes(split),
            None => Transformed::no(plan),
        })
    }

    /// The split itself, shared by every index kind. `None` means the node was left alone.
    fn split(
        &self,
        node: &dyn SplittableSearch,
        schema: &DFSchemaRef,
    ) -> datafusion::common::Result<Option<LogicalPlan>> {
        if !node.is_splittable() {
            return Ok(None);
        }

        let fast_search = self.context.fast_search() && node.honors_fast_search();
        let coverage = match fast_search {
            true => node.coverage(&self.context).indexed_only(),
            false => node.coverage(&self.context),
        };
        match coverage {
            IndexCoverage::Complete => Ok(None),
            IndexCoverage::Unusable if fast_search => {
                Ok(Some(LogicalPlan::EmptyRelation(EmptyRelation {
                    produce_one_row: false,
                    schema: schema.clone(),
                })))
            }
            IndexCoverage::Unusable => Ok(Some(node.flat_branch(node.input().clone()))),
            IndexCoverage::Partial {
                indexed,
                gaps,
                block,
            } => {
                let indexed_input = match indexed {
                    Some(fragments) => restrict_scan(
                        node.input(),
                        &ScanRestriction::Fragments(Arc::new(fragments)),
                    )?,
                    None => node.input().clone(),
                };
                let indexed_branch = node.indexed_branch(indexed_input, block);
                if gaps.is_empty() {
                    return Ok(Some(indexed_branch));
                }
                let mut sources = Vec::with_capacity(gaps.len());
                for gap in &gaps {
                    sources.push(restrict_scan(node.input(), gap)?);
                }
                let mut builder = LogicalPlanBuilder::new(sources.remove(0));
                for source in sources {
                    builder = builder.union(source)?;
                }
                let flat_branch = node.flat_branch(builder.build()?);
                Ok(Some(node.merge(
                    indexed_branch,
                    flat_branch,
                    &self.context,
                )?))
            }
        }
    }
}

impl AnalyzerRule for SplitOnIndexCoverage {
    fn name(&self) -> &str {
        "split_on_index_coverage"
    }

    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<LogicalPlan> {
        analyze_bottom_up(plan, |node| self.rewrite_node(node))
    }
}

impl OptimizerRule for SplitOnIndexCoverage {
    fn name(&self) -> &str {
        "split_on_index_coverage"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        self.rewrite_node(plan)
    }
}

impl SplittableSearch for VectorSearchNode {
    fn is_splittable(&self) -> bool {
        // Only an indexed search can have a coverage gap; a flat search already reads everything.
        matches!(
            self.access_path_resolution(),
            Some(VectorAccessPath::Index { .. })
        )
    }

    fn coverage(&self, context: &ScanPlanningContext) -> IndexCoverage {
        let column = &self.query().column;
        let staleness = context
            .vector_index(column)
            .map(|index| &index.staleness)
            .unwrap_or(&OverlayStaleness::None);
        // An ANN segment that cannot name its fragments still answers by row address, so blocking
        // and re-scoring works regardless — the vector path never has to give up on the index.
        let stale = match staleness {
            OverlayStaleness::Rows(rows) => Some(rows.clone()),
            OverlayStaleness::None | OverlayStaleness::Unknown => None,
        };

        let (Some(unindexed), Some(indexed)) = (
            context.unindexed_fragments(column),
            context.indexed_fragments(column),
        ) else {
            return IndexCoverage::Complete;
        };
        if indexed.is_empty() {
            return IndexCoverage::Unusable;
        }
        if unindexed.is_empty() && stale.is_none() {
            return IndexCoverage::Complete;
        }

        let mut gaps = Vec::with_capacity(2);
        if !unindexed.is_empty() {
            gaps.push(ScanRestriction::Fragments(Arc::new(unindexed)));
        }
        if let Some(rows) = &stale {
            gaps.push(ScanRestriction::Rows(rows.clone()));
        }
        IndexCoverage::Partial {
            indexed: Some(indexed),
            gaps,
            block: stale,
        }
    }

    fn input(&self) -> &LogicalPlan {
        Self::input(self)
    }

    fn indexed_branch(
        &self,
        input: LogicalPlan,
        block: Option<Arc<RowAddrTreeMap>>,
    ) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(self.clone().with_input(input).with_overlay_block(block)),
        })
    }

    fn flat_branch(&self, input: LogicalPlan) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(
                self.clone()
                    .with_input(input)
                    .with_resolution(VectorAccessPath::Flat),
            ),
        })
    }

    fn merge(
        &self,
        indexed: LogicalPlan,
        flat: LogicalPlan,
        context: &ScanPlanningContext,
    ) -> datafusion::common::Result<LogicalPlan> {
        let column = &self.query().column;
        let merged = LogicalPlanBuilder::new(indexed).union(flat)?.build()?;
        let with_vectors = LogicalPlan::Extension(Extension {
            node: Arc::new(LanceTakeNode::try_new(
                merged,
                self.dataset().clone(),
                self.dataset()
                    .empty_projection()
                    .union_column(column, OnMissing::Error)?,
                context.take_settings().clone(),
            )?),
        });
        // The branches' `_distance` values are per-branch and, for the indexed one, approximate.
        // Dropping the column here both discards them and keeps the re-rank from producing a
        // duplicate `_distance`.
        let candidates = LogicalPlanBuilder::new(with_vectors)
            .project(vec![col(ROW_ID), col(column)])?
            .build()?;
        Ok(LogicalPlan::Extension(Extension {
            node: Arc::new(
                self.clone()
                    .with_input(candidates)
                    .with_resolution(VectorAccessPath::Flat)
                    .with_prefilter(PrefilterSourceKind::None)
                    .with_overlay_block(None),
            ),
        }))
    }
}

/// A Lance scan leaf whose scalar index query a data overlay has partly invalidated.
///
/// The third instance of the coverage split, and the one that needed a plan node to reach. Its
/// coverage hole is purely at row level — the index covers every fragment, it just describes some
/// rows as they were before an overlay changed them — which is why `indexed` is `None` and the
/// only gap is a row set.
struct ScanCoverage {
    scan: LogicalPlan,
    table_name: TableReference,
    stale: Arc<RowAddrTreeMap>,
}

impl ScanCoverage {
    /// Recognize a scan whose resolved index query has stale rows.
    ///
    /// Returns `None` for every scan on the common path: no index query, no overlays, or an index
    /// query no overlay touched.
    fn of(plan: &LogicalPlan, context: &ScanPlanningContext) -> Option<Self> {
        if !context.has_overlays() {
            return None;
        }
        let LogicalPlan::TableScan(scan) = plan else {
            return None;
        };
        // Idempotence: this rule runs in the optimizer, which loops to a fixed point, and the
        // indexed branch it produces is still a scan with an index query. The block it carries is
        // what says the split already happened here.
        let index_query = with_lance_source(plan, |source| match source.overlay_block() {
            Some(_) => None,
            None => source.filter_plan()?.index_query.clone(),
        })??;
        match context.index_query_staleness(&index_query) {
            OverlayStaleness::Rows(stale) => Some(Self {
                scan: plan.clone(),
                table_name: scan.table_name.clone(),
                stale,
            }),
            // A scalar index answers by row address whatever segment produced the entry, so there
            // is no case where blocking cannot express the hole.
            OverlayStaleness::None | OverlayStaleness::Unknown => None,
        }
    }
}

impl SplittableSearch for ScanCoverage {
    fn is_splittable(&self) -> bool {
        true
    }

    fn honors_fast_search(&self) -> bool {
        false
    }

    fn coverage(&self, _context: &ScanPlanningContext) -> IndexCoverage {
        IndexCoverage::Partial {
            indexed: None,
            gaps: vec![ScanRestriction::Rows(self.stale.clone())],
            block: Some(self.stale.clone()),
        }
    }

    fn input(&self) -> &LogicalPlan {
        &self.scan
    }

    fn indexed_branch(
        &self,
        input: LogicalPlan,
        block: Option<Arc<RowAddrTreeMap>>,
    ) -> LogicalPlan {
        let Some(block) = block else {
            return input;
        };
        map_lance_scan(&input, |source| source.blocking(block.clone())).unwrap_or(input)
    }

    /// The gap branch is already a scan of exactly the stale rows with its index turned off, so
    /// re-reading them from current values needs nothing further.
    fn flat_branch(&self, input: LogicalPlan) -> LogicalPlan {
        input
    }

    fn merge(
        &self,
        indexed: LogicalPlan,
        flat: LogicalPlan,
        _context: &ScanPlanningContext,
    ) -> datafusion::common::Result<LogicalPlan> {
        // Aliased back to the table's own name: a union derives its output schema from its inputs
        // and drops their relation qualifiers, which would strand every parent expression that
        // still names the table. Both branches read the same table, so the name is still true.
        LogicalPlanBuilder::new(indexed)
            .union(flat)?
            .alias(self.table_name.clone())?
            .build()
    }
}

/// Derive each Lance scan's scalar index query and record it on the scan's source.
///
/// This is what gives the index decision a place in the plan. Until it runs, the decision exists
/// only inside `TableProvider::scan`, where no rule can see it — which is why the scan leaf was the
/// one coverage split that had to be written out by hand.
///
/// It runs after `PushDownFilter` for the same reason
/// [`ResolvePrefilterSource`] does: the predicate has to have reached its final position first.
#[derive(Debug)]
pub struct ResolveScalarIndexQuery {
    context: Arc<ScanPlanningContext>,
}

impl ResolveScalarIndexQuery {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl OptimizerRule for ResolveScalarIndexQuery {
    fn name(&self) -> &str {
        "resolve_scalar_index_query"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::TableScan(scan) = &plan else {
            return Ok(Transformed::no(plan));
        };
        // The physical planner unqualifies a `TableScan`'s filters before handing them to
        // `TableProvider::scan`, and the Lance expression planner expects the same bare columns.
        let filters = unnormalize_cols(scan.filters.iter().cloned());
        let resolved = with_lance_source(&plan, |source| {
            match source.filter_plan().is_some() || filters.is_empty() {
                true => None,
                false => Some(source.resolve_filter_plan(&filters, self.context.scalar_indices())),
            }
        });
        let Some(Some(filter_plan)) = resolved else {
            return Ok(Transformed::no(plan));
        };
        let filter_plan = filter_plan.map_err(datafusion::common::DataFusionError::from)?;
        Ok(Transformed::yes(map_lance_scan(&plan, |source| {
            source.with_filter_plan(filter_plan.clone())
        })?))
    }
}

/// Run `f` against the [`LanceScanSource`] behind a `TableScan`, if that is what this node is.
///
/// Closure-shaped rather than returning a reference because `source_as_provider` hands back an
/// owned `Arc`, so the borrow cannot outlive this call.
fn with_lance_source<R>(plan: &LogicalPlan, f: impl FnOnce(&LanceScanSource) -> R) -> Option<R> {
    let LogicalPlan::TableScan(scan) = plan else {
        return None;
    };
    let provider = source_as_provider(&scan.source).ok()?;
    let source = (provider.as_ref() as &dyn Any).downcast_ref::<LanceScanSource>()?;
    Some(f(source))
}

/// Rebuild `plan`, replacing every Lance scan leaf's source with `f` applied to it.
fn map_lance_scan(
    plan: &LogicalPlan,
    f: impl Fn(&LanceScanSource) -> LanceScanSource,
) -> datafusion::common::Result<LogicalPlan> {
    Ok(plan
        .clone()
        .transform_down(|node| {
            let Some(source) = with_lance_source(&node, &f) else {
                return Ok(Transformed::no(node));
            };
            let LogicalPlan::TableScan(scan) = &node else {
                return Ok(Transformed::no(node));
            };
            let mut scan = scan.clone();
            scan.source = provider_as_source(Arc::new(source));
            Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
        })?
        .data)
}

/// Rebuild `plan`, pointing every scan leaf at the same source narrowed by `restriction`.
///
/// The recursion exists because the predicate may not have reached the leaf: before
/// `PushDownFilter` runs there is a `Filter` in between, and that `Filter` has to be duplicated
/// onto every branch along with the scan.
pub(super) fn restrict_scan(
    plan: &LogicalPlan,
    restriction: &ScanRestriction,
) -> datafusion::common::Result<LogicalPlan> {
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
            scan.source = provider_as_source(Arc::new(source.restricted_to(restriction)));
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
