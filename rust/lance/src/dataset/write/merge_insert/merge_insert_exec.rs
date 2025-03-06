// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{cmp::Ordering, sync::Arc};

use arrow_schema::{DataType, Field, Schema};
use datafusion::{common::DFSchema, physical_plan::ExecutionPlan};
use datafusion_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use super::MergeInsertParams;

/// Logical plan node for merge insert write.
///
/// This takes a schema:
/// * `target`
/// * `source`
/// * `_rowid`
/// * `_rowaddr`
/// * `action`
///
/// And does the appropriate write operation.
///
/// The output of this node is a single row containing statistics about the operation.
#[derive(PartialEq, Eq, Hash, Debug)]
struct MergeInsertWriteNode {
    input: LogicalPlan,
    params: MergeInsertParams,
    schema: Arc<DFSchema>,
}

impl PartialOrd for MergeInsertWriteNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.params.partial_cmp(&other.params) {
            Some(Ordering::Equal) => self.input.partial_cmp(&other.input),
            cmp => cmp,
        }
    }
}

impl MergeInsertWriteNode {
    fn new(input: LogicalPlan, params: MergeInsertParams) -> Self {
        let schema = Schema::new(vec![
            Field::new("num_inserted_rows", DataType::UInt64, false),
            Field::new("num_updated_rows", DataType::UInt64, false),
            Field::new("num_deleted_rows", DataType::UInt64, false),
        ]);
        let schema = Arc::new(DFSchema::try_from(schema).unwrap());
        Self {
            input,
            params,
            schema,
        }
    }
}

impl UserDefinedLogicalNodeCore for MergeInsertWriteNode {
    fn name(&self) -> &str {
        "MergeInsertWrite"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &Arc<DFSchema> {
        &self.schema
    }

    fn expressions(&self) -> Vec<datafusion_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MergeInsertWrite")
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<datafusion_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::error::Result<Self> {
        if !exprs.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "MergeInsertWriteNode does not accept expressions".to_string(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "MergeInsertWriteNode requires exactly one input".to_string(),
            ));
        }
        Ok(Self::new(inputs[0].clone(), self.params.clone()))
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        todo!("Compute which columns are necessary for the write")
    }
}

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by moving updated rows to new fragments. This mode
/// is most optimal when updating the full schema.
///
/// It returns a single batch, containing the statistics.
struct FullSchemaMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    params: MergeInsertParams,
}

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by doing updates by writing new data files and apending
/// them to fragments in place. This mode is most optimal when updating a subset
/// of columns, particularly when the columns not being updated are expensive to
/// rewrite.
///
/// It returns a single batch, containing the statistics.
struct PartialUpdateMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    params: MergeInsertParams,
}
