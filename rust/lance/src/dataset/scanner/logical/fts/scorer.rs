// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The compound scorer node, which scores a whole compound subtree in one pass
//! instead of scoring each leaf and combining the results afterwards.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::common::{DFSchemaRef, DataFusionError};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_core::ROW_ID;
use lance_index::scalar::inverted::query::{FtsQuery, FtsSearchParams};
use lance_table::format::IndexMetadata;

use super::super::PrefilterSourceKind;
use crate::dataset::Dataset;

/// A whole compound query answered by one posting-list scorer.
///
/// Produced by [`UseFtsCompoundScorer`], which collapses a qualifying `FtsCompound` subtree — and
/// with it every leaf and every leaf's copy of the prefilter — into this single node.
#[derive(Debug, Clone)]
pub struct FtsCompoundScorerNode {
    /// The prefilter subtree, kept even when unused so the node has an input to plan.
    pub(super) input: LogicalPlan,
    pub(super) dataset: Arc<Dataset>,
    pub(super) query: FtsQuery,
    pub(super) params: FtsSearchParams,
    pub(super) segments: Vec<IndexMetadata>,
    pub(super) prefilter: PrefilterSourceKind,
    pub(super) schema: DFSchemaRef,
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
            write!(f, ", prefilter={}", self.prefilter)?;
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
