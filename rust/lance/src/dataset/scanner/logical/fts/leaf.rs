// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The FTS leaf node: one `Match` or `Phrase` against one column, and the access
//! path that resolves it to an index or a flat scan.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::common::plan_err;
use datafusion::common::{DFSchema, DFSchemaRef, DataFusionError};
use datafusion::logical_expr::{Expr, InvariantLevel, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_core::ROW_ID;
use lance_index::scalar::inverted::query::{FtsQuery, FtsSearchParams};
use lance_index::scalar::inverted::{DocumentGranularity, fts_schema};
use lance_select::mask::RowAddrTreeMap;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use super::super::PrefilterSourceKind;
use crate::dataset::Dataset;
use crate::index::scalar::inverted::{ResolvedFtsField, resolve_fts_field};
use crate::io::exec::fts::SharedFtsScorer;
use crate::{Error, Result};

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
/// exactly the way [`VectorSearchNode`](super::super::VectorSearchNode)'s two paths are.
#[derive(Debug, Clone)]
pub struct FtsLeafNode {
    pub(super) input: LogicalPlan,
    pub(super) dataset: Arc<Dataset>,
    /// Always `FtsQuery::Match` or `FtsQuery::Phrase`; the compound variants get their own node.
    pub(super) query: FtsQuery,
    pub(super) params: FtsSearchParams,
    pub(super) granularity: DocumentGranularity,
    /// The schema-resolved field path. Computed once because both `necessary_children_exprs` and
    /// lowering need it, and `resolve_fts_field` is not free.
    pub(super) field: ResolvedFtsField,
    pub(super) resolution: Option<FtsAccessPath>,
    pub(super) prefilter: PrefilterSourceKind,
    /// Whether the input's row order is the answer's row order.
    ///
    /// A bounded flat leaf normally has to sort by score to produce the global top-k. When the
    /// leaf is re-ranking someone else's already-bounded result (`full_text_search` with a vector
    /// `query_filter`), the ordering contract belongs to that upstream search, and sorting here
    /// would reshuffle it.
    pub(super) retains_input_order: bool,
    /// Rows the index must not emit, because a data overlay changed a value the index covers. Set
    /// by [`SplitOnIndexCoverage`](SplitOnIndexCoverage), which puts the same rows on
    /// a flat branch so they are scored from their current text.
    pub(super) overlay_block: Option<Arc<RowAddrTreeMap>>,
    /// The rendezvous through which the two branches of a split leaf agree on corpus statistics.
    ///
    /// A BM25 score is only meaningful relative to a corpus, so scores from an index that saw part
    /// of the data and a flat scan that saw the rest are not comparable — and the merge above them
    /// ranks the two together. The flat branch, which can see both halves, computes the combined
    /// statistics and publishes them here; the indexed branch waits for them instead of using its
    /// own. It is an execution-time object living in a logical node because only the rule that
    /// created both branches knows they are two halves of one search.
    ///
    /// Only list-element granularity needs it: whole-document scores already share the index's
    /// corpus-wide statistics.
    pub(super) shared_scorer: Option<Arc<SharedFtsScorer>>,
    pub(super) schema: DFSchemaRef,
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
            retains_input_order: false,
            overlay_block: None,
            shared_scorer: None,
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

    pub fn prefilter(&self) -> &PrefilterSourceKind {
        &self.prefilter
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

    pub fn retaining_input_order(mut self) -> Self {
        self.retains_input_order = true;
        self
    }

    pub fn with_overlay_block(mut self, block: Option<Arc<RowAddrTreeMap>>) -> Self {
        self.overlay_block = block;
        self
    }

    pub fn with_shared_scorer(mut self, scorer: Option<Arc<SharedFtsScorer>>) -> Self {
        self.shared_scorer = scorer;
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

    /// An unresolved leaf is not executable. See [`VectorSearchNode::check_invariants`].
    ///
    /// [`VectorSearchNode::check_invariants`]: super::super::VectorSearchNode
    fn check_invariants(&self, check: InvariantLevel) -> datafusion::common::Result<()> {
        if matches!(check, InvariantLevel::Executable) && self.resolution.is_none() {
            return plan_err!(
                "full-text search on column '{}' reached execution with no access path resolved",
                self.column()
            );
        }
        Ok(())
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
            write!(f, ", prefilter={}", self.prefilter)?;
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
