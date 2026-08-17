// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The `_rowoffset` column: node, resolution rule, and lowering.
//!
//! A row's offset is its position in the dataset once deletions are accounted for, so computing one
//! needs every earlier fragment's row count and deletion vector. That is the only I/O any physical
//! node's constructor does, which is why the load happens in stage 2 and a rule hands the result to
//! the node here.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{DFSchema, DFSchemaRef, plan_err};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{
    Expr, Extension, InvariantLevel, LogicalPlan, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::projection::ProjectionExec;
use lance_core::{ROW_ADDR, ROW_OFFSET, ROW_OFFSET_FIELD};

use super::context::ScanPlanningContext;
use super::rules::analyze_bottom_up;
use crate::Result;
use crate::io::exec::{AddRowOffsetExec, RowOffsetMap};

/// Append `_rowoffset` to the input's columns.
#[derive(Clone)]
pub struct RowOffsetNode {
    input: LogicalPlan,
    /// The per-fragment state the computation needs. Filled in by [`ResolveRowOffsets`]; a node
    /// that still has `None` here cannot be lowered.
    offsets: Option<RowOffsetMap>,
    schema: DFSchemaRef,
}

impl RowOffsetNode {
    pub fn try_new(input: LogicalPlan) -> Result<Self> {
        if !input.schema().has_column_with_unqualified_name(ROW_ADDR) {
            return Err(crate::Error::internal(format!(
                "a {ROW_OFFSET} column is computed from {ROW_ADDR}, which its input does not have"
            )));
        }
        let mut fields = input.schema().as_arrow().fields().to_vec();
        fields.push(Arc::new(ROW_OFFSET_FIELD.clone()));
        let schema = Arc::new(DFSchema::try_from(ArrowSchema::new_with_metadata(
            fields,
            input.schema().as_arrow().metadata().clone(),
        ))?);
        Ok(Self {
            input,
            offsets: None,
            schema,
        })
    }

    pub fn offsets(&self) -> Option<&RowOffsetMap> {
        self.offsets.as_ref()
    }

    fn with_offsets(mut self, offsets: RowOffsetMap) -> Self {
        self.offsets = Some(offsets);
        self
    }
}

impl fmt::Debug for RowOffsetNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_for_explain(f)
    }
}

impl PartialEq for RowOffsetNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input && self.offsets.is_some() == other.offsets.is_some()
    }
}

impl Eq for RowOffsetNode {}

impl Hash for RowOffsetNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.offsets.is_some().hash(state);
    }
}

impl PartialOrd for RowOffsetNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.input.partial_cmp(&other.input)
    }
}

impl UserDefinedLogicalNodeCore for RowOffsetNode {
    fn name(&self) -> &str {
        "RowOffset"
    }

    fn check_invariants(&self, check: InvariantLevel) -> datafusion::common::Result<()> {
        if matches!(check, InvariantLevel::Executable) && self.offsets.is_none() {
            return plan_err!("{ROW_OFFSET} reached execution with no fragment offsets loaded");
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
        write!(f, "RowOffset")
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() {
            return Err(datafusion::common::DataFusionError::Internal(
                "RowOffset takes no expressions".into(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::common::DataFusionError::Internal(format!(
                "RowOffset takes exactly one input, got {}",
                inputs.len()
            )));
        }
        Ok(Self {
            input: inputs.remove(0),
            offsets: self.offsets.clone(),
            schema: self.schema.clone(),
        })
    }

    /// Every input column carries through, and `_rowaddr` is read even when nothing above wants it.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![(0..self.input.schema().fields().len()).collect()])
    }
}

/// Hand each `_rowoffset` node the fragment offsets stage 2 loaded for it.
#[derive(Debug)]
pub struct ResolveRowOffsets {
    context: Arc<ScanPlanningContext>,
}

impl ResolveRowOffsets {
    pub fn new(context: Arc<ScanPlanningContext>) -> Self {
        Self { context }
    }
}

impl AnalyzerRule for ResolveRowOffsets {
    fn name(&self) -> &str {
        "resolve_row_offsets"
    }

    fn analyze(
        &self,
        plan: LogicalPlan,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<LogicalPlan> {
        analyze_bottom_up(plan, |node| {
            let LogicalPlan::Extension(extension) = &node else {
                return Ok(Transformed::no(node));
            };
            let Some(offsets) = extension
                .node
                .as_any()
                .downcast_ref::<RowOffsetNode>()
                .filter(|row_offset| row_offset.offsets().is_none())
            else {
                return Ok(Transformed::no(node));
            };
            let Some(loaded) = self.context.row_offsets() else {
                return plan_err!(
                    "{ROW_OFFSET} was requested but stage 2 did not load its offsets"
                );
            };
            Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(offsets.clone().with_offsets(loaded.clone())),
            })))
        })
    }
}

pub fn plan_row_offset(
    node: &RowOffsetNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let offsets = node.offsets().ok_or_else(|| {
        crate::Error::internal(format!(
            "{ROW_OFFSET} lowering ran before its offsets loaded"
        ))
    })?;
    let with_offsets = Arc::new(AddRowOffsetExec::try_new_from_map(input, offsets)?);

    // The read emits the system columns in its own order, which need not be the order the node
    // declared. Restating the declared order here is what keeps the two schemas equal, which
    // DataFusion checks at every extension boundary.
    let schema = with_offsets.schema();
    let columns = node
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let name = field.name();
            Ok((expressions::col(name, schema.as_ref())?, name.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(ProjectionExec::try_new(columns, with_offsets)?))
}
