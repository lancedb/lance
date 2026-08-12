// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Full-text search, as logical nodes, rules, and lowering.
//!
//! Everything FTS-specific lives here rather than being spread across `nodes.rs` / `rules.rs` /
//! `planner.rs`. That is deliberate: the design doc asks whether an index type could one day ship
//! its own planning support, and keeping one index's nodes, rules, prefetch, and lowering in a
//! single file is the closest thing to an answer the spike can give. The rest of the module
//! reaches in through five entry points — [`build_source`], [`prefetch`], [`rules`],
//! [`plan_extension`], and [`collect_requests`].
//!
//! # Shape
//!
//! An [`FtsQuery`] is a recursive IR, so it maps onto a subtree rather than a single node:
//!
//! ```text
//! FtsCompound{Boolean}                 <- Boost / MultiMatch / Boolean
//!   FtsLeaf{Match, via=index}          <- one per Match / Phrase
//!     Filter / TableScan               <- prefilter source, or the text to scan
//!   FtsLeaf{Match, via=flat}
//!     Filter / TableScan
//! ```
//!
//! Each leaf resolves independently, which is what lets a partially-indexed column be handled by
//! the same split-into-two-branches rewrite the vector path uses.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::{Schema as ArrowSchema, SortOptions};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, DFSchemaRef, DataFusionError};
use datafusion::functions_aggregate;
use datafusion::logical_expr::{
    Expr, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, expressions};
use datafusion_physical_expr::{PhysicalExpr, PhysicalSortExpr};
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::query::{FtsQuery, FtsSearchParams, MatchQuery, Operator};
use lance_index::scalar::inverted::{DocumentGranularity, SCORE_COL, fts_schema};
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_select::mask::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::context::{OpaqueSegments, OverlayStaleness, ScanPlanningContext};
use super::nodes::{LanceTakeNode, PrefilterSourceKind, TakeSettings};
use super::rules::{IndexCoverage, SplittableSearch};
use super::source::ScanRestriction;
use crate::dataset::{Dataset, Scanner};
use crate::index::scalar::inverted::{ResolvedFtsField, load_segment_details, resolve_fts_field};
use crate::io::exec::PreFilterSource;
use crate::io::exec::fts::{
    BoolSlot, BooleanQueryExec, BoostQueryExec, CompoundQueryExec, FlatMatchFilterExec,
    FlatMatchQueryExec, FtsDocumentExec, MatchQueryExec, PhraseQueryExec,
    build_boolean_query_children_with_schema,
};
use crate::{Error, Result};

// ---------------------------------------------------------------------------------------------
// Stage 2: prefetch
// ---------------------------------------------------------------------------------------------

/// Everything known about one FTS index, captured once per planning invocation.
#[derive(Debug, Clone)]
pub struct FtsIndexInfo {
    /// Every committed segment of the index; a search fans out over all of them.
    pub segments: Vec<IndexMetadata>,
    /// Whether the index stores token positions. Only phrase queries need it, and finding out
    /// requires reading the index details, so it is resolved here rather than at lowering time.
    pub with_position: bool,
    /// Row-level coverage: which of this index's entries a data overlay has invalidated.
    pub staleness: OverlayStaleness,
}

impl FtsIndexInfo {
    fn covered_fragments(&self) -> Option<RoaringBitmap> {
        let mut covered = RoaringBitmap::new();
        for segment in &self.segments {
            covered |= segment.fragment_bitmap.as_ref()?;
        }
        Some(covered)
    }
}

/// Which FTS index a plan needs metadata for. One per (column, granularity) pair, since a column
/// may carry both a row-level and a list-element index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FtsIndexRequest {
    pub column: String,
    pub granularity: DocumentGranularity,
    /// Whether any leaf on this column is a phrase query, which is what makes `with_position`
    /// worth the extra read.
    pub needs_position: bool,
}

/// Walk the unoptimized plan for the FTS indices it will need.
pub fn collect_requests(plan: &LogicalPlan) -> Vec<FtsIndexRequest> {
    let mut requests: Vec<FtsIndexRequest> = Vec::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::Extension(extension) = node
            && let Some(leaf) = extension.node.as_any().downcast_ref::<FtsLeafNode>()
        {
            let needs_position = matches!(leaf.query, FtsQuery::Phrase(_));
            match requests.iter_mut().find(|r| {
                r.column == leaf.field.canonical_path && r.granularity == leaf.granularity
            }) {
                Some(existing) => existing.needs_position |= needs_position,
                None => requests.push(FtsIndexRequest {
                    column: leaf.field.canonical_path.clone(),
                    granularity: leaf.granularity,
                    needs_position,
                }),
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    requests
}

/// Load the index metadata for each request. The one stage that is allowed to do I/O.
///
/// Input validation that depends on this metadata happens here too, not in the rules: an error
/// raised inside an `OptimizerRule` comes back wrapped as a DataFusion `External` error, which
/// loses the [`Error`] variant callers match on.
pub async fn prefetch(
    dataset: &Arc<Dataset>,
    requests: &[FtsIndexRequest],
    target_fragments: &RoaringBitmap,
    fragments: &[Fragment],
) -> Result<HashMap<(String, DocumentGranularity), FtsIndexInfo>> {
    let mut loaded = HashMap::with_capacity(requests.len());
    for request in requests {
        let Some(segments) = crate::index::scalar::inverted::load_segments(
            dataset,
            &request.column,
            request.granularity,
        )
        .await?
        else {
            continue;
        };
        let with_position = if request.needs_position {
            load_segment_details(dataset, &request.column, &segments)
                .await?
                .with_position
        } else {
            false
        };
        // `Opaque`, not `Covering`: a legacy segment cannot say which rows it indexed, and a BM25
        // score depends on the whole indexed document set, so a relevant overlay invalidates it
        // wholesale rather than row by row.
        let staleness = super::context::overlay_staleness(
            dataset,
            &segments,
            fragments,
            OpaqueSegments::Opaque,
        )
        .await?;
        let info = FtsIndexInfo {
            segments,
            with_position,
            staleness,
        };
        // A phrase query only needs positions from an index it will actually read. When the index
        // covers none of the target fragments the query is answered by a flat scan, which computes
        // positions from the text itself.
        let index_is_reachable = info
            .covered_fragments()
            .is_none_or(|covered| !(&covered & target_fragments).is_empty());
        if request.needs_position && !with_position && index_is_reachable {
            return Err(Error::invalid_input(
                "position is not found but required for phrase queries, try recreating the index with position"
                    .to_string(),
            ));
        }
        loaded.insert((request.column.clone(), request.granularity), info);
    }
    Ok(loaded)
}

// ---------------------------------------------------------------------------------------------
// Logical nodes
// ---------------------------------------------------------------------------------------------

/// How an FTS leaf will actually be computed. Filled in by [`ResolveFtsAccessPath`].
#[derive(Debug, Clone)]
pub enum FtsAccessPath {
    /// Tokenize and score the input's text directly.
    Flat,
    /// Search the index's posting lists.
    Index { segments: Vec<IndexMetadata> },
}

impl FtsAccessPath {
    fn identity(&self) -> Option<Vec<Uuid>> {
        match self {
            Self::Flat => None,
            Self::Index { segments } => Some(segments.iter().map(|s| s.uuid).collect()),
        }
    }
}

impl PartialEq for FtsAccessPath {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for FtsAccessPath {}

impl Hash for FtsAccessPath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
    }
}

impl PartialOrd for FtsAccessPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.identity().partial_cmp(&other.identity())
    }
}

/// A single `Match` or `Phrase` query over the input's rows.
///
/// Output is `fts_schema(granularity)` — `[_rowid, _score]`, plus `_doc_index` for list-element
/// granularity — no matter which access path it lowers to, so the two are interchangeable in
/// exactly the way [`VectorSearchNode`](super::nodes::VectorSearchNode)'s two paths are.
#[derive(Debug, Clone)]
pub struct FtsLeafNode {
    input: LogicalPlan,
    dataset: Arc<Dataset>,
    /// Always `FtsQuery::Match` or `FtsQuery::Phrase`; the compound variants get their own node.
    query: FtsQuery,
    params: FtsSearchParams,
    granularity: DocumentGranularity,
    /// The schema-resolved field path. Computed once because both `necessary_children_exprs` and
    /// lowering need it, and `resolve_fts_field` is not free.
    field: ResolvedFtsField,
    resolution: Option<FtsAccessPath>,
    prefilter: PrefilterSourceKind,
    /// Set by [`SplitPartiallyIndexedFts`] on the branch it restricts to indexed fragments, for
    /// the same reason the vector path needs it: without it the rule re-splits its own output on
    /// DataFusion's next optimizer pass.
    input_fully_indexed: bool,
    /// Whether the input's row order is the answer's row order.
    ///
    /// A bounded flat leaf normally has to sort by score to produce the global top-k. When the
    /// leaf is re-ranking someone else's already-bounded result (`full_text_search` with a vector
    /// `query_filter`), the ordering contract belongs to that upstream search, and sorting here
    /// would reshuffle it.
    retains_input_order: bool,
    /// Rows the index must not emit, because a data overlay changed a value the index covers. Set
    /// by [`SplitOnIndexCoverage`](super::rules::SplitOnIndexCoverage), which puts the same rows on
    /// a flat branch so they are scored from their current text.
    overlay_block: Option<Arc<RowAddrTreeMap>>,
    schema: DFSchemaRef,
}

impl FtsLeafNode {
    pub fn try_new(
        input: LogicalPlan,
        dataset: Arc<Dataset>,
        query: FtsQuery,
        params: FtsSearchParams,
    ) -> Result<Self> {
        let (column, granularity) = match &query {
            FtsQuery::Match(q) => (q.column.clone(), q.document_granularity),
            FtsQuery::Phrase(q) => (q.column.clone(), q.document_granularity),
            other => {
                return Err(Error::internal(format!(
                    "FtsLeafNode only accepts Match and Phrase queries, got {other}"
                )));
            }
        };
        let column = column.ok_or_else(|| {
            Error::invalid_input("the column must be specified in the query".to_string())
        })?;
        let granularity = granularity.ok_or_else(|| {
            Error::internal("FTS query document granularity was not resolved".to_string())
        })?;
        let field = resolve_fts_field(dataset.schema(), &column, granularity)?;
        let schema = Arc::new(DFSchema::try_from(
            fts_schema(granularity).as_ref().clone(),
        )?);
        Ok(Self {
            input,
            dataset,
            query,
            params,
            granularity,
            field,
            resolution: None,
            prefilter: PrefilterSourceKind::default(),
            input_fully_indexed: false,
            retains_input_order: false,
            overlay_block: None,
            schema,
        })
    }

    pub fn column(&self) -> &str {
        &self.field.canonical_path
    }

    pub fn granularity(&self) -> DocumentGranularity {
        self.granularity
    }

    pub fn is_phrase(&self) -> bool {
        matches!(self.query, FtsQuery::Phrase(_))
    }

    pub fn input(&self) -> &LogicalPlan {
        &self.input
    }

    pub fn resolution(&self) -> Option<&FtsAccessPath> {
        self.resolution.as_ref()
    }

    pub fn prefilter(&self) -> PrefilterSourceKind {
        self.prefilter
    }

    pub fn input_fully_indexed(&self) -> bool {
        self.input_fully_indexed
    }

    pub fn with_input(mut self, input: LogicalPlan) -> Self {
        self.input = input;
        self
    }

    pub fn with_resolution(mut self, resolution: FtsAccessPath) -> Self {
        self.resolution = Some(resolution);
        self
    }

    pub fn with_prefilter(mut self, prefilter: PrefilterSourceKind) -> Self {
        self.prefilter = prefilter;
        self
    }

    pub fn covering_only_indexed_input(mut self) -> Self {
        self.input_fully_indexed = true;
        self
    }

    pub fn retaining_input_order(mut self) -> Self {
        self.retains_input_order = true;
        self
    }

    pub fn with_overlay_block(mut self, block: Option<Arc<RowAddrTreeMap>>) -> Self {
        self.overlay_block = block;
        self
    }

    fn reads_text(&self) -> bool {
        !matches!(self.resolution, Some(FtsAccessPath::Index { .. }))
    }
}

impl PartialEq for FtsLeafNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.query == other.query
            && self.params.limit == other.params.limit
            && self.granularity == other.granularity
            && self.resolution == other.resolution
            && self.prefilter == other.prefilter
            && self.input_fully_indexed == other.input_fully_indexed
            && self.overlay_block == other.overlay_block
    }
}

impl Eq for FtsLeafNode {}

impl Hash for FtsLeafNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.query.to_string().hash(state);
        self.granularity.hash(state);
        self.resolution.hash(state);
        self.prefilter.hash(state);
        self.input_fully_indexed.hash(state);
        // `RowAddrTreeMap` is not `Hash`; see the note on `VectorSearchNode`'s impl.
        self.overlay_block.is_some().hash(state);
    }
}

impl PartialOrd for FtsLeafNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self
            .field
            .canonical_path
            .partial_cmp(&other.field.canonical_path)
        {
            Some(Ordering::Equal) => self.input.partial_cmp(&other.input),
            other => other,
        }
    }
}

impl UserDefinedLogicalNodeCore for FtsLeafNode {
    fn name(&self) -> &str {
        "FtsLeaf"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FtsLeaf: column={}, query={}, limit={:?}",
            self.field.canonical_path, self.query, self.params.limit
        )?;
        match &self.resolution {
            Some(FtsAccessPath::Flat) => write!(f, ", via=flat")?,
            Some(FtsAccessPath::Index { segments }) => {
                write!(f, ", via=index(segments={})", segments.len())?
            }
            None => {}
        }
        if self.prefilter != PrefilterSourceKind::None {
            write!(f, ", prefilter={:?}", self.prefilter)?;
        }
        if self.overlay_block.is_some() {
            write!(f, ", overlay_block")?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() || inputs.len() != 1 {
            return Err(DataFusionError::Internal(
                "FtsLeaf takes exactly one input and no expressions".into(),
            ));
        }
        let mut node = self.clone();
        node.input = inputs.remove(0);
        Ok(node)
    }

    /// The same pivot the vector path turns on: an indexed leaf wants only row ids from its child
    /// — which is precisely a prefilter — while a flat leaf has to read the text itself.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let reads_text = self.reads_text();
        let root = &self.field.root_column;
        let needed = self
            .input
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() == ROW_ID || (reads_text && field.name() == root))
            .map(|(idx, _)| idx)
            .collect();
        Some(vec![needed])
    }
}

/// How a compound FTS node combines its children.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub enum FtsCompoundKind {
    /// Two children: positive, then negative.
    Boost,
    /// N children, one per sub-match.
    MultiMatch,
    /// Children laid out as `should ++ must ++ must_not`.
    Boolean {
        should: usize,
        must: usize,
        must_not: usize,
    },
}

/// `Boost`, `MultiMatch`, or `Boolean` over N already-scored children.
///
/// The children are ordinary logical inputs, which is what lets the leaf-level rules run
/// uniformly no matter how deeply a leaf is nested.
#[derive(Debug, Clone)]
pub struct FtsCompoundNode {
    inputs: Vec<LogicalPlan>,
    query: FtsQuery,
    params: FtsSearchParams,
    kind: FtsCompoundKind,
    granularity: DocumentGranularity,
    schema: DFSchemaRef,
}

impl FtsCompoundNode {
    pub fn try_new(
        inputs: Vec<LogicalPlan>,
        query: FtsQuery,
        params: FtsSearchParams,
        kind: FtsCompoundKind,
        granularity: DocumentGranularity,
    ) -> Result<Self> {
        let schema = Arc::new(DFSchema::try_from(
            fts_schema(granularity).as_ref().clone(),
        )?);
        Ok(Self {
            inputs,
            query,
            params,
            kind,
            granularity,
            schema,
        })
    }
}

impl PartialEq for FtsCompoundNode {
    fn eq(&self, other: &Self) -> bool {
        self.inputs == other.inputs && self.kind == other.kind && self.query == other.query
    }
}

impl Eq for FtsCompoundNode {}

impl Hash for FtsCompoundNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inputs.hash(state);
        self.kind.hash(state);
        self.query.to_string().hash(state);
    }
}

impl PartialOrd for FtsCompoundNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inputs.partial_cmp(&other.inputs)
    }
}

impl UserDefinedLogicalNodeCore for FtsCompoundNode {
    fn name(&self) -> &str {
        "FtsCompound"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inputs.iter().collect()
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FtsCompound: kind={:?}, limit={:?}",
            self.kind, self.params.limit
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() || inputs.len() != self.inputs.len() {
            return Err(DataFusionError::Internal(
                "FtsCompound input arity changed".into(),
            ));
        }
        let mut node = self.clone();
        node.inputs = inputs;
        Ok(node)
    }

    /// Children are scored FTS results; every column of each is consumed.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(
            self.inputs
                .iter()
                .map(|input| (0..input.schema().fields().len()).collect())
                .collect(),
        )
    }
}

/// A whole compound query answered by one posting-list scorer.
///
/// Produced by [`UseFtsCompoundScorer`], which collapses a qualifying `FtsCompound` subtree — and
/// with it every leaf and every leaf's copy of the prefilter — into this single node.
#[derive(Debug, Clone)]
pub struct FtsCompoundScorerNode {
    /// The prefilter subtree, kept even when unused so the node has an input to plan.
    input: LogicalPlan,
    dataset: Arc<Dataset>,
    query: FtsQuery,
    params: FtsSearchParams,
    segments: Vec<IndexMetadata>,
    prefilter: PrefilterSourceKind,
    schema: DFSchemaRef,
}

impl PartialEq for FtsCompoundScorerNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input && self.query == other.query && self.prefilter == other.prefilter
    }
}

impl Eq for FtsCompoundScorerNode {}

impl Hash for FtsCompoundScorerNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.query.to_string().hash(state);
        self.prefilter.hash(state);
    }
}

impl PartialOrd for FtsCompoundScorerNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.input.partial_cmp(&other.input)
    }
}

impl UserDefinedLogicalNodeCore for FtsCompoundScorerNode {
    fn name(&self) -> &str {
        "FtsCompoundScorer"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FtsCompoundScorer: query={}, segments={}, limit={:?}",
            self.query,
            self.segments.len(),
            self.params.limit
        )?;
        if self.prefilter != PrefilterSourceKind::None {
            write!(f, ", prefilter={:?}", self.prefilter)?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() || inputs.len() != 1 {
            return Err(DataFusionError::Internal(
                "FtsCompoundScorer takes exactly one input and no expressions".into(),
            ));
        }
        let mut node = self.clone();
        node.input = inputs.remove(0);
        Ok(node)
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let needed = self
            .input
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() == ROW_ID)
            .map(|(idx, _)| idx)
            .collect();
        Some(vec![needed])
    }
}

/// FTS used as a *filter* rather than as a source: keep the input's rows that match, and append
/// their `_score`.
///
/// Unlike [`FtsLeafNode`] this preserves its input's columns, so it cannot be the same node —
/// which is also true of the exec nodes (`FlatMatchFilterExec` vs `FlatMatchQueryExec`).
#[derive(Debug, Clone)]
pub struct FtsMatchFilterNode {
    input: LogicalPlan,
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    field: ResolvedFtsField,
    schema: DFSchemaRef,
}

impl FtsMatchFilterNode {
    pub fn try_new(
        input: LogicalPlan,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
    ) -> Result<Self> {
        let column = query.column.clone().ok_or_else(|| {
            Error::invalid_input("the column must be specified in the query".to_string())
        })?;
        let granularity = query.document_granularity.ok_or_else(|| {
            Error::internal("FTS Match query granularity was not resolved".to_string())
        })?;
        let field = resolve_fts_field(dataset.schema(), &column, granularity)?;
        let mut fields = input.schema().as_arrow().fields().to_vec();
        fields.push(Arc::new(lance_index::scalar::inverted::SCORE_FIELD.clone()));
        let schema = Arc::new(DFSchema::try_from(ArrowSchema::new(fields))?);
        Ok(Self {
            input,
            dataset,
            query,
            params,
            field,
            schema,
        })
    }
}

impl PartialEq for FtsMatchFilterNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input && self.query == other.query
    }
}

impl Eq for FtsMatchFilterNode {}

impl Hash for FtsMatchFilterNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.query.terms.hash(state);
    }
}

impl PartialOrd for FtsMatchFilterNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.input.partial_cmp(&other.input)
    }
}

impl UserDefinedLogicalNodeCore for FtsMatchFilterNode {
    fn name(&self) -> &str {
        "FtsMatchFilter"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FtsMatchFilter: column={}, query=[{}]",
            self.field.canonical_path, self.query.terms
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() || inputs.len() != 1 {
            return Err(DataFusionError::Internal(
                "FtsMatchFilter takes exactly one input and no expressions".into(),
            ));
        }
        Self::try_new(
            inputs.remove(0),
            self.dataset.clone(),
            self.query.clone(),
            self.params.clone(),
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// The filter reads the text and passes everything else through, so nothing can be dropped.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![(0..self.input.schema().fields().len()).collect()])
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 1: building
// ---------------------------------------------------------------------------------------------

fn extension(node: impl UserDefinedLogicalNodeCore) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}

/// Build the subtree for a full-text search source over `input`.
///
/// `limit` is the scanner's limit, which the imperative path folds into the root node's params;
/// compound children get `None` so intermediate results stay complete.
pub fn build_source(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FullTextSearchQuery,
    limit: Option<usize>,
) -> Result<LogicalPlan> {
    let mut params = query.params();
    if params.limit.is_none() {
        params = params.with_limit(limit);
    }
    build_query(input, dataset, &query.query, &params)
}

fn build_query(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FtsQuery,
    params: &FtsSearchParams,
) -> Result<LogicalPlan> {
    let granularity = granularity_of(query)?;
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) => Ok(extension(FtsLeafNode::try_new(
            input,
            dataset.clone(),
            query.clone(),
            params.clone(),
        )?)),
        // Compound parents require complete child results, so the limit stops here. This is the
        // "recursive planning contract" `FtsSearchParams::limit` documents.
        FtsQuery::Boost(boost) => {
            let unlimited = params.clone().with_limit(None);
            let children = vec![
                build_query(input.clone(), dataset, &boost.positive, &unlimited)?,
                build_query(input, dataset, &boost.negative, &unlimited)?,
            ];
            Ok(extension(FtsCompoundNode::try_new(
                children,
                query.clone(),
                params.clone(),
                FtsCompoundKind::Boost,
                granularity,
            )?))
        }
        FtsQuery::MultiMatch(multi) => {
            let children = multi
                .match_queries
                .iter()
                .map(|child| {
                    build_query(
                        input.clone(),
                        dataset,
                        &FtsQuery::Match(child.clone()),
                        params,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(extension(FtsCompoundNode::try_new(
                children,
                query.clone(),
                params.clone(),
                FtsCompoundKind::MultiMatch,
                granularity,
            )?))
        }
        FtsQuery::Boolean(boolean) => {
            let unlimited = params.clone().with_limit(None);
            let mut children = Vec::with_capacity(
                boolean.should.len() + boolean.must.len() + boolean.must_not.len(),
            );
            for child in boolean
                .should
                .iter()
                .chain(&boolean.must)
                .chain(&boolean.must_not)
            {
                children.push(build_query(input.clone(), dataset, child, &unlimited)?);
            }
            Ok(extension(FtsCompoundNode::try_new(
                children,
                query.clone(),
                params.clone(),
                FtsCompoundKind::Boolean {
                    should: boolean.should.len(),
                    must: boolean.must.len(),
                    must_not: boolean.must_not.len(),
                },
                granularity,
            )?))
        }
    }
}

fn granularity_of(query: &FtsQuery) -> Result<DocumentGranularity> {
    let missing = || Error::internal("FTS query document granularity was not resolved".to_string());
    match query {
        FtsQuery::Match(q) => q.document_granularity.ok_or_else(missing),
        FtsQuery::Phrase(q) => q.document_granularity.ok_or_else(missing),
        FtsQuery::Boost(q) => granularity_of(&q.positive),
        FtsQuery::MultiMatch(q) => q
            .match_queries
            .first()
            .and_then(|child| child.document_granularity)
            .ok_or_else(missing),
        FtsQuery::Boolean(q) => q
            .should
            .iter()
            .chain(&q.must)
            .chain(&q.must_not)
            .next()
            .ok_or_else(|| {
                Error::invalid_input(
                    "boolean query must have at least one should/must query".to_string(),
                )
            })
            .and_then(granularity_of),
    }
}

/// Re-rank an already-scored input by BM25, for `full_text_search` combined with a vector
/// `query_filter`.
///
/// A `Match` re-rank is just a flat leaf over the input, which is why it needs no node of its
/// own. Anything else is a join against an independently planned FTS tree — and that join is a
/// stock DataFusion node, not a Lance one.
pub fn build_rerank(
    input: LogicalPlan,
    scan: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FullTextSearchQuery,
    limit: Option<usize>,
    settings: &TakeSettings,
) -> Result<LogicalPlan> {
    match &query.query {
        // The imperative `fts_rerank` uses the query's own params here, without folding in the
        // scanner limit: the input is already bounded by the upstream search.
        FtsQuery::Match(match_query) => {
            let params = query.params();
            let column = match_query.column.clone().ok_or_else(|| {
                Error::invalid_input("the column must be specified in the query".to_string())
            })?;
            let granularity = match_query.document_granularity.ok_or_else(|| {
                Error::internal("FTS Match query granularity was not resolved".to_string())
            })?;
            let field = resolve_fts_field(dataset.schema(), &column, granularity)?;
            let input = take_column(input, dataset, &field.root_column, settings)?;
            Ok(extension(
                FtsLeafNode::try_new(
                    input,
                    dataset.clone(),
                    FtsQuery::Match(match_query.clone()),
                    params,
                )?
                .with_resolution(FtsAccessPath::Flat)
                .retaining_input_order(),
            ))
        }
        other => {
            let mut params = query.params();
            if params.limit.is_none() {
                params = params.with_limit(limit);
            }
            let fts = build_query(scan, dataset, other, &params)?;
            join_on_row_id(input, fts)
        }
    }
}

/// Insert a [`LanceTakeNode`] for `column` unless the input already carries it.
pub fn take_column(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    column: &str,
    settings: &TakeSettings,
) -> Result<LogicalPlan> {
    let projection = dataset
        .empty_projection()
        .union_column(column, OnMissing::Error)?;
    if LanceTakeNode::is_noop(&input, &projection)? {
        return Ok(input);
    }
    Ok(extension(LanceTakeNode::try_new(
        input,
        dataset.clone(),
        projection,
        settings.clone(),
    )?))
}

/// Inner-join two scored plans on `_rowid`, keeping one copy of each column.
fn join_on_row_id(left: LogicalPlan, right: LogicalPlan) -> Result<LogicalPlan> {
    // Both sides carry an unqualified `_rowid`, so the join key would be ambiguous without
    // relation aliases.
    let left = LogicalPlanBuilder::new(left).alias("search")?;
    let right = LogicalPlanBuilder::new(right).alias("fts")?.build()?;
    let key = |relation: &str| {
        datafusion::common::Column::new(
            Some(datafusion::common::TableReference::bare(relation)),
            ROW_ID,
        )
    };
    // `join_on` with an equality predicate lowers to a `NestedLoopJoinExec`; naming the keys is
    // what makes it a hash join, and a hash join also emits in probe-side (FTS score) order,
    // which is the ordering the imperative path's `HashJoinExec` produces.
    let joined = left.join(
        right,
        datafusion::logical_expr::JoinType::Inner,
        (vec![key("search")], vec![key("fts")]),
        None,
    )?;

    // Drop the right side's duplicate `_rowid`, matching the projection the imperative path
    // builds by hand over its `HashJoinExec`.
    let mut exprs = Vec::new();
    let mut seen_row_id = false;
    for (qualifier, field) in joined.schema().iter() {
        if field.name() == ROW_ID {
            if seen_row_id {
                continue;
            }
            seen_row_id = true;
        }
        exprs.push(Expr::Column(datafusion::common::Column::from((
            qualifier,
            field.as_ref(),
        ))));
    }
    Ok(joined.project(exprs)?.build()?)
}

/// Apply an FTS `query_filter` above `input`, as a postfilter.
pub fn build_match_filter(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FullTextSearchQuery,
    settings: &TakeSettings,
) -> Result<LogicalPlan> {
    let FtsQuery::Match(match_query) = &query.query else {
        return Err(Error::not_supported(
            "Only Match queries are supported currently when using FTS as a post-filter",
        ));
    };
    let granularity = match_query.document_granularity.ok_or_else(|| {
        Error::internal("FTS Match query granularity was not resolved".to_string())
    })?;
    let column = match_query.column.clone().ok_or_else(|| {
        Error::invalid_input("the column must be specified in the query".to_string())
    })?;
    let field = resolve_fts_field(dataset.schema(), &column, granularity)?;
    let input = take_column(input, dataset, &field.root_column, settings)?;
    Ok(extension(FtsMatchFilterNode::try_new(
        input,
        dataset.clone(),
        match_query.clone(),
        query.params(),
    )?))
}

/// Deduplicate an FTS filter's element-level hits down to one row per `_rowid`.
///
/// Stock `Aggregate` with no aggregate expressions — the imperative path builds the same thing
/// out of `RepartitionExec` + `AggregateExec` by hand.
pub fn dedupe_rows(input: LogicalPlan) -> Result<LogicalPlan> {
    Ok(LogicalPlanBuilder::new(input)
        .aggregate(vec![datafusion::prelude::col(ROW_ID)], Vec::<Expr>::new())?
        .build()?)
}

// ---------------------------------------------------------------------------------------------
// Stage 3: rules
// ---------------------------------------------------------------------------------------------

/// The FTS-owned optimizer rules, in the order they must run.
pub fn rules(context: &Arc<ScanPlanningContext>) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    vec![
        Arc::new(ResolveFtsAccessPath::new(context.clone())),
        Arc::new(UseFtsCompoundScorer::new(context.clone())),
    ]
}

/// Decide whether each leaf uses its inverted index or scans text.
///
/// The index is ruled out when there is none, when it covers no fragment the scan will touch, or
/// — for a phrase query — when it was built without positions, which is the one case that is an
/// error rather than a fallback.
#[derive(Debug)]
pub struct ResolveFtsAccessPath {
    context: Arc<ScanPlanningContext>,
}

impl ResolveFtsAccessPath {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl OptimizerRule for ResolveFtsAccessPath {
    fn name(&self) -> &str {
        "resolve_fts_access_path"
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
        let Some(leaf) = extension.node.as_any().downcast_ref::<FtsLeafNode>() else {
            return Ok(Transformed::no(plan));
        };
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
    context: Arc<ScanPlanningContext>,
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
                prefilter: super::rules::prefilter_kind(&first.input),
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
        matches!(self.resolution, Some(FtsAccessPath::Index { .. })) && !self.input_fully_indexed()
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
            indexed,
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
            node: Arc::new(
                self.clone()
                    .with_input(input)
                    .with_overlay_block(block)
                    .covering_only_indexed_input(),
            ),
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

// ---------------------------------------------------------------------------------------------
// Stage 4: lowering
// ---------------------------------------------------------------------------------------------

/// Lower any FTS node. Returns `None` for nodes this module does not own.
pub fn plan_extension(
    node: &dyn UserDefinedLogicalNode,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Option<Result<Arc<dyn ExecutionPlan>>> {
    if let Some(leaf) = node.as_any().downcast_ref::<FtsLeafNode>() {
        return Some(plan_leaf(leaf, inputs.first().cloned()?));
    }
    if let Some(compound) = node.as_any().downcast_ref::<FtsCompoundNode>() {
        return Some(plan_compound(compound, inputs));
    }
    if let Some(scorer) = node.as_any().downcast_ref::<FtsCompoundScorerNode>() {
        return Some(plan_compound_scorer(scorer, inputs.first().cloned()?));
    }
    if let Some(filter) = node.as_any().downcast_ref::<FtsMatchFilterNode>() {
        return Some(plan_match_filter(filter, inputs.first().cloned()?));
    }
    None
}

fn prefilter_source(kind: PrefilterSourceKind, input: Arc<dyn ExecutionPlan>) -> PreFilterSource {
    match kind {
        PrefilterSourceKind::None => PreFilterSource::None,
        PrefilterSourceKind::ChildRowIds => PreFilterSource::FilteredRowIds(input),
    }
}

fn plan_leaf(node: &FtsLeafNode, input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    match &node.resolution {
        Some(FtsAccessPath::Index { segments }) => {
            let prefilter = prefilter_source(node.prefilter, input);
            let block = node
                .overlay_block
                .as_ref()
                .map(|rows| RowAddrMask::from_block(rows.as_ref().clone()));
            let plan: Arc<dyn ExecutionPlan> = match &node.query {
                FtsQuery::Match(query) => {
                    let mut exec = MatchQueryExec::new_with_segments_and_document_granularity(
                        node.dataset.clone(),
                        query.clone(),
                        node.params.clone(),
                        prefilter,
                        segments.clone(),
                        node.granularity,
                    );
                    if let Some(block) = block {
                        exec = exec.with_overlay_block(block);
                    }
                    Arc::new(exec)
                }
                FtsQuery::Phrase(query) => {
                    let mut exec = PhraseQueryExec::new_with_segments_and_document_granularity(
                        node.dataset.clone(),
                        query.clone(),
                        node.params.clone(),
                        prefilter,
                        segments.clone(),
                        node.granularity,
                    );
                    if let Some(block) = block {
                        exec = exec.with_overlay_block(block);
                    }
                    Arc::new(exec)
                }
                other => {
                    return Err(Error::internal(format!(
                        "FtsLeaf holds a compound query: {other}"
                    )));
                }
            };
            Ok(plan)
        }
        // An unresolved leaf means the rule did not run; a flat scan is always correct.
        Some(FtsAccessPath::Flat) | None => plan_flat_leaf(node, input),
    }
}

/// The brute-force path: feed the input's text to `FlatMatchQueryExec`.
///
/// A phrase query becomes an `And` match with a slop parameter, which is how the imperative path
/// scores phrases over unindexed rows too.
fn plan_flat_leaf(
    node: &FtsLeafNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let (query, params) = match &node.query {
        FtsQuery::Match(query) => (query.clone(), node.params.clone()),
        FtsQuery::Phrase(phrase) => (
            MatchQuery::new(phrase.terms.clone())
                .with_column(phrase.column.clone())
                .with_operator(Operator::And)
                .with_document_granularity(node.granularity),
            node.params.clone().with_phrase_slop(Some(phrase.slop)),
        ),
        other => {
            return Err(Error::internal(format!(
                "FtsLeaf holds a compound query: {other}"
            )));
        }
    };

    let document_column = if node.field.has_lists() {
        VALUE_COLUMN_NAME.to_string()
    } else {
        node.field.canonical_path.clone()
    };
    let input = if node.field.has_lists() {
        Arc::new(FtsDocumentExec::new(input, node.field.clone())) as Arc<dyn ExecutionPlan>
    } else {
        ensure_column_alias(input, &node.dataset, &document_column)?
    };

    let scored = Arc::new(FlatMatchQueryExec::new_with_document_granularity(
        node.dataset.clone(),
        query,
        params,
        input,
        node.granularity,
        document_column,
    )) as Arc<dyn ExecutionPlan>;

    // `combine_fts_leaf_plans` sorts a flat-only leaf when the search is bounded; the index path
    // gets its top-k from the posting lists instead.
    match node.params.limit {
        Some(limit) if !node.retains_input_order => Ok(sort_by_score(scored, Some(limit))?),
        _ => Ok(scored),
    }
}

fn plan_compound(
    node: &FtsCompoundNode,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Result<Arc<dyn ExecutionPlan>> {
    match &node.kind {
        FtsCompoundKind::Boost => {
            let FtsQuery::Boost(query) = &node.query else {
                return Err(Error::internal(
                    "FtsCompound{Boost} holds a non-boost query",
                ));
            };
            let [positive, negative] = inputs else {
                return Err(Error::internal("boost query requires exactly two children"));
            };
            Ok(Arc::new(BoostQueryExec::new(
                query.clone(),
                node.params.clone(),
                positive.clone(),
                negative.clone(),
            )))
        }
        FtsCompoundKind::MultiMatch => plan_multi_match(node, inputs),
        FtsCompoundKind::Boolean {
            should,
            must,
            must_not,
        } => {
            let FtsQuery::Boolean(query) = &node.query else {
                return Err(Error::internal(
                    "FtsCompound{Boolean} holds a non-boolean query",
                ));
            };
            if inputs.len() != should + must + must_not {
                return Err(Error::internal("boolean query child arity changed"));
            }
            let schema = fts_schema(node.granularity);
            let (should_children, rest) = inputs.split_at(*should);
            let (must_children, must_not_children) = rest.split_at(*must);

            let should_plan = build_boolean_query_children_with_schema(
                BoolSlot::Should,
                should_children.to_vec(),
                schema.clone(),
            )?
            .ok_or_else(|| {
                Error::internal("boolean should planning returned no execution plan".to_string())
            })?;
            let must_plan = build_boolean_query_children_with_schema(
                BoolSlot::Must,
                must_children.to_vec(),
                schema.clone(),
            )?;
            let must_not_plan = build_boolean_query_children_with_schema(
                BoolSlot::MustNot,
                must_not_children.to_vec(),
                schema,
            )?
            .ok_or_else(|| {
                Error::internal("boolean must-not planning returned no execution plan".to_string())
            })?;

            if *should == 0 && must_plan.is_none() {
                return Err(Error::invalid_input(
                    "boolean query must have at least one should/must query".to_string(),
                ));
            }
            Ok(Arc::new(BooleanQueryExec::new(
                query.clone(),
                node.params.clone(),
                should_plan,
                must_plan,
                must_not_plan,
            )))
        }
    }
}

/// Union the sub-matches, keep the best score per row, and take the top k.
///
/// Everything here is stock relational algebra — union, group-by-max, sort-with-fetch — which is
/// a finding in itself: `MultiMatch` needs no Lance-specific execution at all.
fn plan_multi_match(
    node: &FtsCompoundNode,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Result<Arc<dyn ExecutionPlan>> {
    let unioned = UnionExec::try_new(inputs.to_vec())?;
    let schema = unioned.schema();
    let single = Arc::new(RepartitionExec::try_new(
        unioned,
        Partitioning::RoundRobinBatch(1),
    )?);
    let deduped = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new_single(vec![(
            expressions::col(ROW_ID, schema.as_ref())?,
            ROW_ID.to_string(),
        )]),
        vec![Arc::new(
            datafusion_physical_expr::aggregate::AggregateExprBuilder::new(
                functions_aggregate::min_max::max_udaf(),
                vec![expressions::col(SCORE_COL, &schema)?],
            )
            .schema(schema.clone())
            .alias(SCORE_COL)
            .build()?,
        )],
        vec![None],
        single,
        schema,
    )?);
    sort_by_score_and_row_id(deduped, node.params.limit)
}

fn plan_compound_scorer(
    node: &FtsCompoundScorerNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(CompoundQueryExec::new_with_segments(
        node.dataset.clone(),
        node.query.clone(),
        node.params.clone(),
        prefilter_source(node.prefilter, input),
        node.segments.clone(),
    )))
}

fn plan_match_filter(
    node: &FtsMatchFilterNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let input = if node.field.has_lists() {
        input
    } else {
        ensure_column_alias(input, &node.dataset, &node.field.canonical_path)?
    };
    Ok(Arc::new(FlatMatchFilterExec::new_with_resolved_field(
        input,
        node.dataset.clone(),
        node.query.clone(),
        node.params.clone(),
        node.field.clone(),
    )))
}

fn sort_by_score(
    plan: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let expr = PhysicalSortExpr {
        expr: expressions::col(SCORE_COL, plan.schema().as_ref())?,
        options: SortOptions {
            descending: true,
            nulls_first: false,
        },
    };
    Ok(Arc::new(
        SortExec::new([expr].into(), plan).with_fetch(fetch),
    ))
}

fn sort_by_score_and_row_id(
    plan: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let exprs = [
        PhysicalSortExpr {
            expr: expressions::col(SCORE_COL, schema.as_ref())?,
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        },
        PhysicalSortExpr {
            expr: expressions::col(ROW_ID, schema.as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        },
    ];
    Ok(Arc::new(
        SortExec::new(exprs.into(), plan).with_fetch(fetch),
    ))
}

/// Expose a (possibly nested) field path as a top-level column, as `Scanner::ensure_column_alias`
/// does: the reader produces the containing struct, but the FTS executor wants one named column.
fn ensure_column_alias(
    input: Arc<dyn ExecutionPlan>,
    dataset: &Arc<Dataset>,
    column: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = input.schema();
    if schema.column_with_name(column).is_some() {
        return Ok(input);
    }
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = schema
        .fields()
        .iter()
        .map(|field| {
            expressions::col(field.name(), schema.as_ref()).map(|expr| (expr, field.name().clone()))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    exprs.push((
        Scanner::create_column_expr(column, dataset.as_ref(), schema.as_ref())?,
        column.to_string(),
    ));
    Ok(Arc::new(ProjectionExec::try_new(exprs, input)?))
}

/// Fragments an FTS index does or does not cover, for the split rule's two branches.
pub fn partition_fragments(
    info: &FtsIndexInfo,
    fragments: &[Fragment],
    covered_side: bool,
) -> Option<Vec<Fragment>> {
    let covered = info.covered_fragments()?;
    Some(
        fragments
            .iter()
            .filter(|fragment| covered.contains(fragment.id as u32) == covered_side)
            .cloned()
            .collect(),
    )
}
