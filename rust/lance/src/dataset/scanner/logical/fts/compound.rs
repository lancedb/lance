// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compound FTS nodes: `Boost`, `Boolean`, and `MultiMatch` over child leaves.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::common::{DFSchema, DFSchemaRef, DataFusionError};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_index::scalar::inverted::query::{FtsQuery, FtsSearchParams};
use lance_index::scalar::inverted::{DocumentGranularity, fts_schema};

use crate::Result;

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
    pub(super) inputs: Vec<LogicalPlan>,
    pub(super) query: FtsQuery,
    pub(super) params: FtsSearchParams,
    pub(super) kind: FtsCompoundKind,
    pub(super) granularity: DocumentGranularity,
    pub(super) schema: DFSchemaRef,
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
