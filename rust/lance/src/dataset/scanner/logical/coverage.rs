// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Splitting a search across the fragments an index covers and those it does not.
//!
//! Cross-index by design: the [`SplittableSearch`] trait is what lets one rule serve every
//! search node type, and the scan leaf, without knowing what any of them are.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::{DFSchemaRef, TableReference};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{EmptyRelation, Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::{AnalyzerRule, ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::prelude::col;
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_select::mask::RowAddrTreeMap;
use lance_table::format::Fragment;

use super::context::{OverlayStaleness, ScanPlanningContext};
use super::fts::FtsLeafNode;
use super::source::ScanRestriction;
use super::{LanceTakeNode, PrefilterSourceKind, VectorAccessPath, VectorSearchNode};
use super::{analyze_bottom_up, map_lance_scan, restrict_scan, with_lance_source};

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
pub fn splittable(plan: &LogicalPlan) -> Option<&dyn SplittableSearch> {
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
