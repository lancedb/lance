// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The match-filter node: applies an FTS predicate to rows a search already produced.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use datafusion::common::{DFSchema, DFSchemaRef, DataFusionError};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_index::scalar::inverted::query::{FtsSearchParams, MatchQuery};

use crate::dataset::Dataset;
use crate::index::scalar::inverted::{ResolvedFtsField, resolve_fts_field};
use crate::{Error, Result};

/// FTS used as a *filter* rather than as a source: keep the input's rows that match, and append
/// their `_score`.
///
/// Unlike [`FtsLeafNode`] this preserves its input's columns, so it cannot be the same node —
/// which is also true of the exec nodes (`FlatMatchFilterExec` vs `FlatMatchQueryExec`).
#[derive(Debug, Clone)]
pub struct FtsMatchFilterNode {
    pub(super) input: LogicalPlan,
    pub(super) dataset: Arc<Dataset>,
    pub(super) query: MatchQuery,
    pub(super) params: FtsSearchParams,
    pub(super) field: ResolvedFtsField,
    pub(super) schema: DFSchemaRef,
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
