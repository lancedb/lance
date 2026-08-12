// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance-specific logical plan nodes.
//!
//! Both nodes follow the shape of
//! [`MergeInsertWriteNode`](crate::dataset::write::merge_insert::logical_plan), the only other
//! `UserDefinedLogicalNodeCore` in the tree — including the hand-written `PartialEq`/`Hash`/
//! `PartialOrd`, which are needed because `Arc<Dataset>` implements none of them.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_core::datatypes::Projection;
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index::vector::{ApproxMode, DIST_COL, Query};
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::Result;
use crate::dataset::Dataset;
use crate::index::vector::utils::{default_distance_type_for, get_vector_type};
use crate::io::exec::TakeExec;

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
/// [`ResolveVectorAccessPath`](super::rules::ResolveVectorAccessPath); `None` until then.
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
/// [`ResolvePrefilterSource`](super::rules::ResolvePrefilterSource).
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
    resolution: Option<VectorAccessPath>,
    prefilter: PrefilterSourceKind,
    /// Whether the input is known to contain only fragments the index covers.
    ///
    /// False until [`SplitPartiallyIndexedSearch`](super::rules::SplitPartiallyIndexedSearch)
    /// establishes it. Without this the rule would re-split its own indexed branch on the
    /// optimizer's next pass, and the branch's scan — already narrowed to `[_rowid]` — cannot
    /// supply the vectors the nested brute-force branch would ask for.
    input_fully_indexed: bool,
    schema: DFSchemaRef,
}

impl VectorSearchNode {
    pub fn try_new(input: LogicalPlan, dataset: Arc<Dataset>, mut query: Query) -> Result<Self> {
        let accuracy = SearchAccuracy::from_query(&query);
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
        let schema = Arc::new(DFSchema::try_from(ArrowSchema::new(vec![
            ROW_ID_FIELD.clone(),
            ArrowField::new(DIST_COL, DataType::Float32, true),
        ]))?);
        Ok(Self {
            input,
            dataset,
            query,
            accuracy,
            distance_type,
            resolution: None,
            prefilter: PrefilterSourceKind::default(),
            input_fully_indexed: false,
            schema,
        })
    }

    pub fn with_resolution(mut self, resolution: VectorAccessPath) -> Self {
        self.resolution = Some(resolution);
        self
    }

    pub fn with_prefilter(mut self, prefilter: PrefilterSourceKind) -> Self {
        self.prefilter = prefilter;
        self
    }

    pub fn prefilter(&self) -> PrefilterSourceKind {
        self.prefilter
    }

    pub fn covering_only_indexed_input(mut self) -> Self {
        self.input_fully_indexed = true;
        self
    }

    pub fn input_fully_indexed(&self) -> bool {
        self.input_fully_indexed
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
            && self.resolution == other.resolution
            && self.prefilter == other.prefilter
            && self.input_fully_indexed == other.input_fully_indexed
    }
}

impl Eq for VectorSearchNode {}

impl Hash for VectorSearchNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.query.column.hash(state);
        self.query.k.hash(state);
        self.accuracy.hash(state);
        self.resolution.hash(state);
        self.prefilter.hash(state);
        self.input_fully_indexed.hash(state);
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
            resolution: self.resolution.clone(),
            prefilter: self.prefilter,
            input_fully_indexed: self.input_fully_indexed,
            schema: self.schema.clone(),
        })
    }

    /// What the child has to produce depends on the access path, which is why this is only
    /// accurate after [`ResolveVectorAccessPath`](super::rules::ResolveVectorAccessPath) has run:
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

/// Late materialization: fetch `projection`'s columns for rows the input has already identified,
/// keyed by `_rowid`.
///
/// The physical form is a row-stream `FilteredReadExec`, whose output is the input's columns
/// followed by the newly fetched ones. That ordering is reproduced here with the same helper the
/// physical node uses, so the logical and physical schemas cannot drift.
#[derive(Debug, Clone)]
pub struct LanceTakeNode {
    input: LogicalPlan,
    dataset: Arc<Dataset>,
    projection: Projection,
    schema: DFSchemaRef,
}

impl LanceTakeNode {
    pub fn try_new(
        input: LogicalPlan,
        dataset: Arc<Dataset>,
        projection: Projection,
    ) -> Result<Self> {
        let schema = Self::output_schema(&input, &dataset, &projection)?;
        Ok(Self {
            input,
            dataset,
            projection,
            schema,
        })
    }

    /// Whether a take is needed at all: if the input already carries every projected column,
    /// the node is a no-op and the builder should skip it.
    pub fn is_noop(input: &LogicalPlan, projection: &Projection) -> Result<bool> {
        let input_schema = input.schema().as_arrow().clone();
        let missing = projection
            .clone()
            .subtract_arrow_schema(&input_schema, lance_core::datatypes::OnMissing::Ignore)?;
        Ok(!missing.has_data_fields() && !missing.with_row_id && !missing.with_row_addr)
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    fn output_schema(
        input: &LogicalPlan,
        dataset: &Dataset,
        projection: &Projection,
    ) -> Result<DFSchemaRef> {
        let input_schema = input.schema().as_arrow().clone();
        let fields_to_read = projection
            .clone()
            .subtract_arrow_schema(&input_schema, lance_core::datatypes::OnMissing::Ignore)?;
        let output =
            TakeExec::calculate_output_schema(dataset.schema(), &input_schema, &fields_to_read);
        Ok(Arc::new(DFSchema::try_from(ArrowSchema::from(&output))?))
    }
}

impl PartialEq for LanceTakeNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.dataset.base == other.dataset.base
            && self.projection.field_ids == other.projection.field_ids
    }
}

impl Eq for LanceTakeNode {}

impl Hash for LanceTakeNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.dataset.base.hash(state);
        let mut ids = self
            .projection
            .field_ids
            .iter()
            .copied()
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.hash(state);
    }
}

impl PartialOrd for LanceTakeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.input.partial_cmp(&other.input)
    }
}

impl UserDefinedLogicalNodeCore for LanceTakeNode {
    fn name(&self) -> &str {
        "LanceTake"
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
        let mut columns = self
            .projection
            .to_bare_schema()
            .fields
            .iter()
            .map(|field| field.name.clone())
            .collect::<Vec<_>>();
        columns.sort();
        write!(
            f,
            "LanceTake: columns=[{}] by={}",
            columns.join(", "),
            ROW_ID
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() {
            return Err(datafusion::common::DataFusionError::Internal(
                "LanceTake takes no expressions".into(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::common::DataFusionError::Internal(format!(
                "LanceTake takes exactly one input, got {}",
                inputs.len()
            )));
        }
        Self::try_new(
            inputs.remove(0),
            self.dataset.clone(),
            self.projection.clone(),
        )
        .map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))
    }

    /// Every input column carries through to the output, so nothing below can be dropped.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![(0..self.input.schema().fields().len()).collect()])
    }
}
