// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Analyzer and optimizer rules that resolve FTS access paths and rewrite compound
//! subtrees.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::{AnalyzerRule, ApplyOrder, OptimizerConfig, OptimizerRule};
use lance_index::scalar::inverted::SCORE_COL;
use lance_index::scalar::inverted::query::FtsQuery;
use lance_select::mask::RowAddrTreeMap;

use super::super::context::{OverlayStaleness, ScanPlanningContext};
use super::super::nodes::PrefilterSourceKind;
use super::super::rules::{IndexCoverage, SplittableSearch};
use super::super::source::ScanRestriction;

use super::*;

// ---------------------------------------------------------------------------------------------
// Stage 3: rules
// ---------------------------------------------------------------------------------------------

/// The FTS-owned mandatory rewrites. Deciding whether a leaf can use its index is not an
/// optimization — an unresolved leaf cannot be lowered at all.
pub fn analyzer_rules(
    context: &Arc<ScanPlanningContext>,
) -> Vec<Arc<dyn AnalyzerRule + Send + Sync>> {
    vec![Arc::new(ResolveFtsAccessPath::new(context.clone()))]
}

/// The FTS-owned optimizations. `UseFtsCompoundScorer` is the only rule in the spike that fits
/// `OptimizerRule`'s documented contract: drop it and the plan still returns the same rows, just
/// by scoring each leaf separately and combining afterwards.
pub fn optimizer_rules(
    context: &Arc<ScanPlanningContext>,
) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    vec![Arc::new(UseFtsCompoundScorer::new(context.clone()))]
}

/// Decide whether each leaf uses its inverted index or scans text.
///
/// The index is ruled out when there is none, when it covers no fragment the scan will touch, or
/// — for a phrase query — when it was built without positions, which is the one case that is an
/// error rather than a fallback.
#[derive(Debug)]
pub struct ResolveFtsAccessPath {
    pub(super) context: Arc<ScanPlanningContext>,
}

impl ResolveFtsAccessPath {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl AnalyzerRule for ResolveFtsAccessPath {
    fn name(&self) -> &str {
        "resolve_fts_access_path"
    }

    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<LogicalPlan> {
        super::super::rules::analyze_bottom_up(plan, |node| self.rewrite_node(node))
    }
}

impl ResolveFtsAccessPath {
    fn rewrite_node(
        &self,
        plan: LogicalPlan,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(extension) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(leaf) = extension.node.as_any().downcast_ref::<FtsLeafNode>() else {
            return Ok(Transformed::no(plan));
        };
        // Not an idempotence guard: the builder resolves a leaf to `Flat` when its input already
        // *is* the candidate set — an FTS query scoring the output of a vector filter — and that
        // is a decision, not a default.
        if leaf.resolution.is_some() {
            return Ok(Transformed::no(plan));
        }

        // A phrase query over an index without positions was already rejected during prefetch, so
        // an index in the context is always usable by the time this runs.
        let resolved = match self.context.fts_index(leaf.column(), leaf.granularity()) {
            Some(index) if !(leaf.is_phrase() && !index.with_position) => FtsAccessPath::Index {
                segments: index.segments.clone(),
            },
            _ => FtsAccessPath::Flat,
        };

        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(leaf.clone().with_resolution(resolved)),
        })))
    }
}

/// Collapse a single-column compound query into one posting-list scorer.
///
/// The whole subtree — compound node, leaves, and each leaf's copy of the prefilter — becomes a
/// single [`FtsCompoundScorerNode`]. It is a nice demonstration of the shape the design doc is
/// after: a fast path that the imperative planner expresses as an early `return` from
/// `plan_fts` is, here, a rule that pattern-matches a subtree.
#[derive(Debug)]
pub struct UseFtsCompoundScorer {
    pub(super) context: Arc<ScanPlanningContext>,
}

impl UseFtsCompoundScorer {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }

    /// Every leaf under `node`, or `None` if the subtree holds anything else.
    fn leaves<'a>(&self, node: &'a FtsCompoundNode) -> Option<Vec<&'a FtsLeafNode>> {
        fn walk<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a FtsLeafNode>) -> bool {
            let LogicalPlan::Extension(extension) = plan else {
                return false;
            };
            if let Some(leaf) = extension.node.as_any().downcast_ref::<FtsLeafNode>() {
                out.push(leaf);
                return true;
            }
            if let Some(compound) = extension.node.as_any().downcast_ref::<FtsCompoundNode>() {
                return compound.inputs.iter().all(|input| walk(input, out));
            }
            false
        }
        let mut leaves = Vec::new();
        node.inputs
            .iter()
            .all(|input| walk(input, &mut leaves))
            .then_some(leaves)
    }
}

impl OptimizerRule for UseFtsCompoundScorer {
    fn name(&self) -> &str {
        "use_fts_compound_scorer"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // Top-down so the outermost compound node is collapsed whole, rather than an inner one
        // first leaving the parent unable to match.
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(extension) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(compound) = extension.node.as_any().downcast_ref::<FtsCompoundNode>() else {
            return Ok(Transformed::no(plan));
        };
        if compound.granularity.is_list_element() || !supports_compound_scorer(&compound.query) {
            return Ok(Transformed::no(plan));
        }
        let Some(leaves) = self.leaves(compound) else {
            return Ok(Transformed::no(plan));
        };
        let Some(first) = leaves.first() else {
            return Ok(Transformed::no(plan));
        };
        // Flat and posting-backed leaves do not share a document domain, so a mixed subtree has
        // to keep the general plan.
        if !leaves.iter().all(|leaf| {
            leaf.column() == first.column()
                && matches!(leaf.resolution, Some(FtsAccessPath::Index { .. }))
                && self
                    .context
                    .fts_unindexed_fragments(leaf.column(), leaf.granularity())
                    .is_none_or(|unindexed| unindexed.is_empty())
        }) {
            return Ok(Transformed::no(plan));
        }
        let Some(index) = self.context.fts_index(first.column(), first.granularity()) else {
            return Ok(Transformed::no(plan));
        };
        if contains_phrase_query(&compound.query) && !index.with_position {
            return Ok(Transformed::no(plan));
        }

        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(FtsCompoundScorerNode {
                input: first.input.clone(),
                dataset: first.dataset.clone(),
                query: compound.query.clone(),
                params: compound.params.clone(),
                segments: index.segments.clone(),
                // Computed here rather than read off the leaf: this rule may collapse the
                // subtree before `ResolvePrefilterSource` has visited it.
                prefilter: super::super::rules::prefilter_kind(&first.input),
                schema: compound.schema.clone(),
            }),
        })))
    }
}

/// The FTS half of [`SplitOnIndexCoverage`](super::rules::SplitOnIndexCoverage).
///
/// Structurally simpler than the vector half: there is no re-rank, because BM25 scores from the
/// two branches are directly comparable. The merge is a stock `Union` + `Sort` + `Limit`, which is
/// exactly what `Scanner::combine_fts_leaf_plans` builds by hand.
impl SplittableSearch for FtsLeafNode {
    fn is_splittable(&self) -> bool {
        matches!(self.resolution, Some(FtsAccessPath::Index { .. }))
    }

    fn coverage(&self, context: &ScanPlanningContext) -> IndexCoverage {
        let staleness = context
            .fts_index(self.column(), self.granularity())
            .map(|index| &index.staleness)
            .unwrap_or(&OverlayStaleness::None);
        let stale = match staleness {
            OverlayStaleness::Rows(rows) => Some(rows.clone()),
            OverlayStaleness::None => None,
            // A BM25 score depends on the whole indexed document set, so a segment that cannot say
            // which rows it indexed has nothing salvageable once an overlay touches it.
            OverlayStaleness::Unknown => return IndexCoverage::Unusable,
        };

        let (Some(unindexed), Some(indexed)) = (
            context.fts_unindexed_fragments(self.column(), self.granularity()),
            context.fts_indexed_fragments(self.column(), self.granularity()),
        ) else {
            return IndexCoverage::Complete;
        };
        // Nothing is indexed among the fragments we will touch: this is a flat search that happens
        // to have an index sitting next to it.
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
        &self.input
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
                    .with_resolution(FtsAccessPath::Flat)
                    .with_prefilter(PrefilterSourceKind::None)
                    .with_overlay_block(None),
            ),
        })
    }

    fn merge(
        &self,
        indexed: LogicalPlan,
        flat: LogicalPlan,
        _context: &ScanPlanningContext,
    ) -> datafusion::common::Result<LogicalPlan> {
        let mut merged = LogicalPlanBuilder::new(indexed)
            .union(flat)?
            .sort(vec![datafusion::prelude::col(SCORE_COL).sort(false, false)])?;
        if let Some(limit) = self.params.limit {
            merged = merged.limit(0, Some(limit))?;
        }
        merged.build()
    }
}

fn supports_compound_scorer(query: &FtsQuery) -> bool {
    fn supports_shape(query: &FtsQuery) -> bool {
        match query {
            FtsQuery::Match(_) | FtsQuery::Phrase(_) | FtsQuery::MultiMatch(_) => true,
            FtsQuery::Boolean(query) => {
                (!query.should.is_empty() || !query.must.is_empty())
                    && query
                        .should
                        .iter()
                        .chain(&query.must)
                        .chain(&query.must_not)
                        .all(supports_shape)
            }
            FtsQuery::Boost(query) => {
                supports_shape(&query.positive) && supports_shape(&query.negative)
            }
        }
    }
    !matches!(query, FtsQuery::Match(_) | FtsQuery::Phrase(_)) && supports_shape(query)
}

fn contains_phrase_query(query: &FtsQuery) -> bool {
    match query {
        FtsQuery::Phrase(_) => true,
        FtsQuery::Match(_) | FtsQuery::MultiMatch(_) => false,
        FtsQuery::Boost(query) => {
            contains_phrase_query(&query.positive) || contains_phrase_query(&query.negative)
        }
        FtsQuery::Boolean(query) => query
            .should
            .iter()
            .chain(&query.must)
            .chain(&query.must_not)
            .any(contains_phrase_query),
    }
}
