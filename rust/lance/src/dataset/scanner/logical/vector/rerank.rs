// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The rerank node, which merges the branches of a coverage split back into one
//! top-k answer.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_arrow::SchemaExt;
use lance_index::vector::{DIST_COL, Query};
use lance_linalg::distance::DistanceType;

use crate::Result;
use crate::dataset::Dataset;
use crate::index::vector::utils::{default_distance_type_for, get_vector_type};

/// Score the input's rows by distance to the query vector and keep the nearest `k`.
///
/// The difference from [`VectorSearchNode`] is what it does to the schema, and that follows from
/// where it sits: a search *is* the source, so it may narrow its output to `[_rowid, _distance]`,
/// while this sits above one and has to carry the columns already in flight. It also has no
/// access path — a filter over rows someone else chose is brute force by definition.
#[derive(Debug, Clone)]
pub struct VectorRerankNode {
    input: LogicalPlan,
    query: Query,
    distance_type: DistanceType,
    schema: DFSchemaRef,
}

impl VectorRerankNode {
    pub fn try_new(input: LogicalPlan, dataset: &Dataset, mut query: Query) -> Result<Self> {
        let distance_type = match query.metric_type {
            Some(metric) => metric,
            None => {
                let (_, element_type) = get_vector_type(dataset.schema(), &query.column)?;
                default_distance_type_for(&element_type)
            }
        };
        query.metric_type = Some(distance_type);

        // Mirrors `KNNVectorDistanceExec::try_new_batch`: an existing `_distance` is dropped so the
        // new one lands last.
        let mut arrow = input.schema().as_arrow().clone();
        if arrow.column_with_name(DIST_COL).is_some() {
            arrow = arrow.without_column(DIST_COL);
        }
        let arrow = arrow.try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))?;
        Ok(Self {
            input,
            query,
            distance_type,
            schema: Arc::new(DFSchema::try_from(arrow)?),
        })
    }

    pub fn query(&self) -> &Query {
        &self.query
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }
}

impl PartialEq for VectorRerankNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.query.column == other.query.column
            && self.query.k == other.query.k
    }
}

impl Eq for VectorRerankNode {}

impl Hash for VectorRerankNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.query.column.hash(state);
        self.query.k.hash(state);
    }
}

impl PartialOrd for VectorRerankNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.input.partial_cmp(&other.input)
    }
}

impl UserDefinedLogicalNodeCore for VectorRerankNode {
    fn name(&self) -> &str {
        "VectorRerank"
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
            "VectorRerank: column={}, k={}, metric={}",
            self.query.column, self.query.k, self.distance_type
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() {
            return Err(datafusion::common::DataFusionError::Internal(
                "VectorRerank takes no expressions".into(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::common::DataFusionError::Internal(format!(
                "VectorRerank takes exactly one input, got {}",
                inputs.len()
            )));
        }
        Ok(Self {
            input: inputs.remove(0),
            query: self.query.clone(),
            distance_type: self.distance_type,
            schema: self.schema.clone(),
        })
    }

    /// The vector column has to survive down to here, and every other input column is part of the
    /// output, so nothing below may be pruned.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![(0..self.input.schema().fields().len()).collect()])
    }
}
