// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The vector search node: what the caller asked for, and how it will be answered.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::common::plan_err;
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{Expr, InvariantLevel, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index::vector::{ApproxMode, DIST_COL, Query};
use lance_linalg::distance::DistanceType;
use lance_select::mask::RowAddrTreeMap;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::Result;
use crate::dataset::Dataset;
use crate::index::vector::utils::{default_distance_type_for, get_vector_type};
use crate::io::exec::knn::query_index_field;

/// What answer the caller will accept.
///
/// This is the *semantic* half of a vector search and the only thing that authorizes an
/// approximate access path. No rule may strengthen [`Exact`](Self::Exact) into
/// [`Approximate`](Self::Approximate).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub enum SearchAccuracy {
    /// Only brute force, or an index that provably returns exact results.
    Exact,
    /// An ANN index may be used. The knobs that only mean something under approximation live
    /// here rather than beside the semantic contract.
    Approximate {
        minimum_nprobes: usize,
        maximum_nprobes: Option<usize>,
        ef: Option<usize>,
        /// Over-fetch `k * refine_factor` candidates, then re-rank them exactly.
        refine_factor: Option<u32>,
    },
}

impl SearchAccuracy {
    /// Derive the semantic contract from today's [`Query`].
    ///
    /// `Query` mixes the semantic contract (`use_index`, `approx_mode`) with the approximation
    /// knobs and with pure execution hints. Doing the split in exactly one place keeps the
    /// mapping auditable while both representations coexist.
    pub fn from_query(query: &Query) -> Self {
        if !query.use_index || query.approx_mode == ApproxMode::Accurate {
            return Self::Exact;
        }
        Self::Approximate {
            minimum_nprobes: query.minimum_nprobes,
            maximum_nprobes: query.maximum_nprobes,
            ef: query.ef,
            refine_factor: query.refine_factor,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// How a search will actually be computed. Filled in by
/// [`ResolveVectorAccessPath`](ResolveVectorAccessPath); `None` until then.
///
/// Keeping this on the node — rather than deciding it during lowering — is what lets the
/// combined indexed/unindexed case be expressed as a rewrite over two ordinary search nodes
/// instead of a special case inside the planner.
#[derive(Debug, Clone)]
pub enum VectorAccessPath {
    /// Brute force over the input.
    Flat,
    /// Fan out over the index's segments.
    Index { segments: Vec<IndexMetadata> },
}

impl VectorAccessPath {
    /// Segment uuids identify the access path; `IndexMetadata` is neither `Eq` nor `Hash`, and
    /// the rest of it is descriptive.
    fn identity(&self) -> Option<Vec<Uuid>> {
        match self {
            Self::Flat => None,
            Self::Index { segments } => Some(segments.iter().map(|s| s.uuid).collect()),
        }
    }
}

impl PartialEq for VectorAccessPath {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for VectorAccessPath {}

impl Hash for VectorAccessPath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
    }
}

impl PartialOrd for VectorAccessPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.identity().partial_cmp(&other.identity())
    }
}

/// Where the candidate restriction for an indexed search comes from. Decided by
/// [`ResolvePrefilterSource`](ResolvePrefilterSource).
///
/// This is the lowering artifact of a *prefilter*: a predicate positioned below the search.
/// Nothing about it is visible in the logical result — a postfilter puts the same predicate
/// above the search instead, and never produces one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Default)]
pub enum PrefilterSourceKind {
    /// Nothing below the search restricts the candidates.
    #[default]
    None,
    /// The child's row ids are the candidate set.
    ChildRowIds,
}

/// Nearest-neighbor search over the input's rows.
///
/// Output is `[_rowid, _distance]` regardless of how it is lowered: an index scan and a
/// brute-force scan are two implementations of the same logical operator, and downstream nodes
/// must not be able to tell them apart.
#[derive(Debug, Clone)]
pub struct VectorSearchNode {
    input: LogicalPlan,
    /// Held directly, as `MergeInsertWriteNode` does. The design doc proposed recovering it by
    /// downcasting the child `TableScan`'s `TableSource`; that turned out to buy nothing, since
    /// `LanceTakeNode` needs a handle regardless.
    dataset: Arc<Dataset>,
    /// Retained whole rather than decomposed: the physical nodes still take a `Query`, so
    /// rebuilding one during lowering would just be a lossy round trip.
    query: Query,
    accuracy: SearchAccuracy,
    /// Resolved up front from the column's element type when the caller did not name one.
    /// Leaving it unresolved would force lowering to reach for the dataset schema again.
    distance_type: DistanceType,
    /// How many query vectors `query.key` holds end to end.
    ///
    /// More than one is a batch search: every query is answered against the same rows, and the
    /// output gains a `_query_index` discriminator saying which query a row answers.
    query_count: usize,
    /// Whether `distance_type` is what the caller asked for, or a default stood in for them.
    ///
    /// The difference decides what a metric mismatch means. A caller who named a metric and got an
    /// index built with another one is asking a question that index cannot answer, so the search
    /// falls back to brute force. A caller who named none is not asking for any particular metric,
    /// so the index's own is adopted — which is what the imperative path does.
    distance_type_requested: bool,
    resolution: Option<VectorAccessPath>,
    prefilter: PrefilterSourceKind,
    /// Rows the index must not emit, because a data overlay committed after the index was built
    /// changed a value the index covers. Set by
    /// [`SplitOnIndexCoverage`](SplitOnIndexCoverage), which puts the same rows on a
    /// brute-force branch so they are answered from their current values.
    overlay_block: Option<Arc<RowAddrTreeMap>>,
    schema: DFSchemaRef,
}

impl VectorSearchNode {
    pub fn try_new(input: LogicalPlan, dataset: Arc<Dataset>, mut query: Query) -> Result<Self> {
        let accuracy = SearchAccuracy::from_query(&query);
        let distance_type_requested = query.metric_type.is_some();
        let distance_type = match query.metric_type {
            Some(metric) => metric,
            None => {
                let (_, element_type) = get_vector_type(dataset.schema(), &query.column)?;
                default_distance_type_for(&element_type)
            }
        };
        // Write the resolution back so the `Query` handed to the physical nodes is complete.
        // Leaving it `None` there makes them report `metric=default` in `EXPLAIN`.
        query.metric_type = Some(distance_type);
        let schema = Self::output_schema(1)?;
        Ok(Self {
            input,
            dataset,
            query,
            accuracy,
            distance_type,
            query_count: 1,
            distance_type_requested,
            resolution: None,
            prefilter: PrefilterSourceKind::default(),
            overlay_block: None,
            schema,
        })
    }

    /// The `[_rowid, _distance]` contract, plus the batch discriminator when there is one.
    fn output_schema(query_count: usize) -> Result<DFSchemaRef> {
        let mut fields = Vec::with_capacity(3);
        if query_count > 1 {
            fields.push(query_index_field());
        }
        fields.push(ROW_ID_FIELD.clone());
        fields.push(ArrowField::new(DIST_COL, DataType::Float32, true));
        Ok(Arc::new(DFSchema::try_from(ArrowSchema::new(fields))?))
    }

    /// Answer `count` query vectors at once, reading them end to end out of `query.key`.
    pub fn with_query_count(mut self, count: usize) -> Result<Self> {
        self.schema = Self::output_schema(count)?;
        self.query_count = count;
        Ok(self)
    }

    pub fn query_count(&self) -> usize {
        self.query_count
    }

    /// The `i`th query vector's own single-query search, for a rewrite that answers them one at a
    /// time. The input is shared: every query searches the same rows.
    pub fn single_query(&self, index: usize) -> Result<Self> {
        let dim = self.query.key.len() / self.query_count;
        let mut query = self.query.clone();
        query.key = self.query.key.slice(index * dim, dim);
        Ok(Self {
            query,
            query_count: 1,
            schema: Self::output_schema(1)?,
            ..self.clone()
        })
    }

    pub fn with_resolution(mut self, resolution: VectorAccessPath) -> Self {
        self.resolution = Some(resolution);
        self
    }

    /// Adopt an index's metric, for a search that did not name one.
    ///
    /// Written into the carried `Query` as well, because that is what the physical nodes read.
    pub fn with_distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = distance_type;
        self.query.metric_type = Some(distance_type);
        self
    }

    pub fn distance_type_requested(&self) -> bool {
        self.distance_type_requested
    }

    pub fn with_prefilter(mut self, prefilter: PrefilterSourceKind) -> Self {
        self.prefilter = prefilter;
        self
    }

    pub fn prefilter(&self) -> PrefilterSourceKind {
        self.prefilter
    }

    pub fn with_overlay_block(mut self, block: Option<Arc<RowAddrTreeMap>>) -> Self {
        self.overlay_block = block;
        self
    }

    pub fn overlay_block(&self) -> Option<&Arc<RowAddrTreeMap>> {
        self.overlay_block.as_ref()
    }

    pub fn input(&self) -> &LogicalPlan {
        &self.input
    }

    /// Re-parent the node. The output schema is fixed at `[_rowid, _distance]` and does not
    /// depend on the input, so no revalidation is needed.
    pub fn with_input(mut self, input: LogicalPlan) -> Self {
        self.input = input;
        self
    }

    pub fn access_path_resolution(&self) -> Option<&VectorAccessPath> {
        self.resolution.as_ref()
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn query(&self) -> &Query {
        &self.query
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    pub fn accuracy(&self) -> &SearchAccuracy {
        &self.accuracy
    }
}

impl PartialEq for VectorSearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.query.column == other.query.column
            && self.query.k == other.query.k
            && self.accuracy == other.accuracy
            && self.query_count == other.query_count
            && self.resolution == other.resolution
            && self.prefilter == other.prefilter
            && self.overlay_block == other.overlay_block
    }
}

impl Eq for VectorSearchNode {}

impl Hash for VectorSearchNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.query.column.hash(state);
        self.query.k.hash(state);
        self.accuracy.hash(state);
        self.query_count.hash(state);
        self.resolution.hash(state);
        self.prefilter.hash(state);
        // `RowAddrTreeMap` is not `Hash`, and the flag is the part that distinguishes plans: two
        // nodes with different block lists cannot share an input anyway.
        self.overlay_block.is_some().hash(state);
    }
}

impl PartialOrd for VectorSearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.query.column.partial_cmp(&other.query.column) {
            Some(Ordering::Equal) => self.input.partial_cmp(&other.input),
            other => other,
        }
    }
}

impl UserDefinedLogicalNodeCore for VectorSearchNode {
    fn name(&self) -> &str {
        "VectorSearch"
    }

    /// An unresolved search is not executable.
    ///
    /// The rules that resolve it are mandatory, and a missed one used to mean silently wrong rows
    /// — a brute-force fallback where an index was expected, or an index consulted for rows it no
    /// longer describes. DataFusion checks this at the end of the analyzer, which is the stage
    /// those rules run in, so a rule that fails to fire becomes a planning error instead.
    fn check_invariants(&self, check: InvariantLevel) -> datafusion::common::Result<()> {
        if matches!(check, InvariantLevel::Executable) && self.resolution.is_none() {
            return plan_err!(
                "vector search on column '{}' reached execution with no access path resolved",
                self.query.column
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

    /// The query vector is data, not an expression, so there is nothing here for DataFusion's
    /// expression rewriting to visit.
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VectorSearch: column={}, k={}, metric={}, accuracy={:?}",
            self.query.column, self.query.k, self.distance_type, self.accuracy
        )?;
        if self.query_count > 1 {
            write!(f, ", queries={}", self.query_count)?;
        }
        match &self.resolution {
            Some(VectorAccessPath::Flat) => write!(f, ", via=flat")?,
            Some(VectorAccessPath::Index { segments }) => {
                write!(f, ", via=index(deltas={})", segments.len())?
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
        if !exprs.is_empty() {
            return Err(datafusion::common::DataFusionError::Internal(
                "VectorSearch takes no expressions".into(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::common::DataFusionError::Internal(format!(
                "VectorSearch takes exactly one input, got {}",
                inputs.len()
            )));
        }
        Ok(Self {
            input: inputs.remove(0),
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            accuracy: self.accuracy.clone(),
            distance_type: self.distance_type,
            query_count: self.query_count,
            distance_type_requested: self.distance_type_requested,
            resolution: self.resolution.clone(),
            prefilter: self.prefilter,
            overlay_block: self.overlay_block.clone(),
            schema: self.schema.clone(),
        })
    }

    /// What the child has to produce depends on the access path, which is why this is only
    /// accurate after [`ResolveVectorAccessPath`](ResolveVectorAccessPath) has run:
    /// an indexed search reads vectors from the index and wants nothing but row ids from the
    /// child, while a brute-force search has to read the vectors itself.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let reads_vectors = !matches!(self.resolution, Some(VectorAccessPath::Index { .. }));
        let needed = self
            .input
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| {
                field.name() == ROW_ID || (reads_vectors && field.name() == &self.query.column)
            })
            .map(|(idx, _)| idx)
            .collect();
        Some(vec![needed])
    }
}
