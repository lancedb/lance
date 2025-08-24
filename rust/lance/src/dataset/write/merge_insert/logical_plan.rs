// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use datafusion::common::Result as DFResult;
use datafusion::{
    common::DFSchema,
    execution::SessionState,
    physical_plan::ExecutionPlan,
    physical_planner::{ExtensionPlanner, PhysicalPlanner},
};
use datafusion_expr::{LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore};
use lance_core::{ROW_ADDR, ROW_ID};
use std::{cmp::Ordering, sync::Arc};

use crate::{
    dataset::write::merge_insert::exec::FullSchemaMergeInsertExec, io::exec::ScalarIndexJoinExec,
    Dataset,
};

use super::{MergeInsertParams, MERGE_ACTION_COLUMN};

/// Logical plan node for merge insert write.
///
/// Expects input schema:
/// * `source.{col1, col2, ...}` - columns from the source relation
/// * `target.{col1, col2, ...}` - columns from the target relation
/// * `target._rowaddr` - special column to locate existing rows in the target
/// * `__action` - unqualified column that describes the action to perform.
///   See [`super::assign_action::merge_insert_action`]
///
/// Output is empty.
#[derive(Debug)]
pub struct MergeInsertWriteNode {
    input: LogicalPlan,
    pub(crate) dataset: Arc<Dataset>,
    pub(crate) params: MergeInsertParams,
    schema: Arc<DFSchema>,
}

impl PartialEq for MergeInsertWriteNode {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
            && self.input == other.input
            && self.dataset.base == other.dataset.base
    }
}

impl Eq for MergeInsertWriteNode {}

impl std::hash::Hash for MergeInsertWriteNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.params.hash(state);
        self.input.hash(state);
        self.dataset.base.hash(state);
    }
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
    pub fn new(input: LogicalPlan, dataset: Arc<Dataset>, params: MergeInsertParams) -> Self {
        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        let schema = Arc::new(DFSchema::try_from(empty_schema).unwrap());
        Self {
            input,
            dataset,
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
        let on_keys = self.params.on.join(", ");
        let when_matched = match &self.params.when_matched {
            crate::dataset::WhenMatched::DoNothing => "DoNothing",
            crate::dataset::WhenMatched::UpdateAll => "UpdateAll",
            crate::dataset::WhenMatched::UpdateIf(_) => "UpdateIf",
        };
        let when_not_matched = if self.params.insert_not_matched {
            "InsertAll"
        } else {
            "DoNothing"
        };
        let when_not_matched_by_source = match &self.params.delete_not_matched_by_source {
            crate::dataset::WhenNotMatchedBySource::Keep => "Keep",
            crate::dataset::WhenNotMatchedBySource::Delete => "Delete",
            crate::dataset::WhenNotMatchedBySource::DeleteIf(_) => "DeleteIf",
        };

        write!(
            f,
            "MergeInsertWrite: on=[{}], when_matched={}, when_not_matched={}, when_not_matched_by_source={}",
            on_keys,
            when_matched,
            when_not_matched,
            when_not_matched_by_source
        )
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
        Ok(Self::new(
            inputs[0].clone(),
            self.dataset.clone(),
            self.params.clone(),
        ))
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        // Going to need:
        // * all columns from the `source` relation
        // * `__action` column (unqualified)
        // * `target._rowaddr` column specifically

        let input_schema = self.input.schema();
        let mut necessary_columns = Vec::new();

        for (i, (qualifier, field)) in input_schema.iter().enumerate() {
            let should_include = match qualifier {
                // Include all source columns - they contain the new data to write
                Some(qualifier) if qualifier.table() == "source" => true,

                // Include target._rowaddr specifically - needed to locate existing rows for updates
                Some(qualifier) if qualifier.table() == "target" && field.name() == ROW_ADDR => {
                    true
                }

                // Include target._rowid specifically - needed to locate existing rows for updates
                Some(qualifier) if qualifier.table() == "target" && field.name() == ROW_ID => true,

                // Include unqualified columns like "__action" - tells us what operation to perform
                None if field.name() == MERGE_ACTION_COLUMN => true,

                // Skip other target columns (target.value, target.key, target._rowid) - not needed for write
                _ => false,
            };

            if should_include {
                necessary_columns.push(i);
            }
        }

        Some(vec![necessary_columns])
    }
}

/// Physical planner for MergeInsertWriteNode.
pub struct MergeInsertPlanner {}

#[async_trait]
impl ExtensionPlanner for MergeInsertPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        Ok(
            if let Some(write_node) = node.as_any().downcast_ref::<MergeInsertWriteNode>() {
                assert_eq!(logical_inputs.len(), 1, "Inconsistent number of inputs");
                assert_eq!(physical_inputs.len(), 1, "Inconsistent number of inputs");
                let exec = FullSchemaMergeInsertExec::try_new(
                    physical_inputs[0].clone(),
                    write_node.dataset.clone(),
                    write_node.params.clone(),
                )?;
                Some(Arc::new(exec))
            } else {
                None
            },
        )
    }
}

/// Logical plan node for scalar index join operation.
///
/// This node represents a join between source data and a target table using a scalar index
/// instead of a full table scan. It's used to optimize merge insert operations when a suitable
/// scalar index exists on the join column.
///
/// Expected input schema: source columns only
/// Expected output schema: source columns + _rowid column for matched target rows
#[derive(Debug, Clone)]
pub struct ScalarIndexJoinNode {
    /// Input logical plan (source data)
    input: LogicalPlan,

    /// Dataset reference (target table)
    dataset: Arc<Dataset>,

    /// Column to join on
    join_column: String,

    /// Name of scalar index to use
    index_name: String,

    /// Type of join (Inner, Left, Right)
    join_type: datafusion_expr::JoinType,

    /// Table reference for qualifying _rowid column
    table_reference: Option<datafusion::common::TableReference>,

    /// Output schema  
    schema: Arc<DFSchema>,
}

impl PartialEq for ScalarIndexJoinNode {
    fn eq(&self, other: &Self) -> bool {
        self.join_column == other.join_column
            && self.index_name == other.index_name
            && self.join_type == other.join_type
            && self.table_reference == other.table_reference
            && self.input == other.input
            && self.dataset.base == other.dataset.base
    }
}

impl Eq for ScalarIndexJoinNode {}

impl std::hash::Hash for ScalarIndexJoinNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.join_column.hash(state);
        self.index_name.hash(state);
        self.join_type.hash(state);
        self.table_reference.hash(state);
        self.input.hash(state);
        self.dataset.base.hash(state);
    }
}

impl PartialOrd for ScalarIndexJoinNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.join_column.partial_cmp(&other.join_column) {
            Some(Ordering::Equal) => self.index_name.partial_cmp(&other.index_name),
            cmp => cmp,
        }
    }
}

impl ScalarIndexJoinNode {
    pub fn try_new(
        input: LogicalPlan,
        dataset: Arc<Dataset>,
        join_column: String,
        index_name: String,
        join_type: datafusion_expr::JoinType,
        table_reference: Option<datafusion::common::TableReference>,
    ) -> datafusion::error::Result<Self> {
        // Build output schema: source columns + _rowid
        let input_schema = input.schema();

        // Extract fields and qualifiers from input schema
        let mut qualified_fields = Vec::new();
        for i in 0..input_schema.fields().len() {
            let (qualifier, field) = input_schema.qualified_field(i);
            qualified_fields.push((qualifier.cloned(), Arc::new(field.clone())));
        }

        // Add _rowid column (nullable for left and right joins)
        let rowid_nullable = matches!(
            join_type,
            datafusion_expr::JoinType::Left | datafusion_expr::JoinType::Right
        );
        let rowid_field = Arc::new(arrow_schema::Field::new(
            lance_core::ROW_ID,
            arrow_schema::DataType::UInt64,
            rowid_nullable,
        ));
        // _rowid column should use the provided table reference for qualification
        qualified_fields.push((table_reference.clone(), rowid_field));

        // Construct DFSchema with preserved qualifiers
        let schema = Arc::new(DFSchema::new_with_metadata(
            qualified_fields,
            input_schema.metadata().clone(),
        )?);

        Ok(Self {
            input,
            dataset,
            join_column,
            index_name,
            join_type,
            table_reference,
            schema,
        })
    }
}

impl UserDefinedLogicalNodeCore for ScalarIndexJoinNode {
    fn name(&self) -> &str {
        "ScalarIndexJoin"
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
        write!(
            f,
            "ScalarIndexJoin: on=[{}], index=[{}], type=[{:?}]",
            self.join_column, self.index_name, self.join_type
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<datafusion_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::error::Result<Self> {
        if !exprs.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "ScalarIndexJoinNode does not accept expressions".to_string(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "ScalarIndexJoinNode requires exactly one input".to_string(),
            ));
        }
        Self::try_new(
            inputs[0].clone(),
            self.dataset.clone(),
            self.join_column.clone(),
            self.index_name.clone(),
            self.join_type,
            self.table_reference.clone(),
        )
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        // We need all input columns for the join operation
        let input_schema = self.input.schema();
        let necessary_columns: Vec<usize> = (0..input_schema.fields().len()).collect();
        Some(vec![necessary_columns])
    }
}

/// Physical planner for ScalarIndexJoinNode.
pub struct ScalarIndexJoinPlanner {}

#[async_trait]
impl ExtensionPlanner for ScalarIndexJoinPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        Ok(
            if let Some(join_node) = node.as_any().downcast_ref::<ScalarIndexJoinNode>() {
                assert_eq!(logical_inputs.len(), 1, "Inconsistent number of inputs");
                assert_eq!(physical_inputs.len(), 1, "Inconsistent number of inputs");

                let exec = ScalarIndexJoinExec::try_new(
                    physical_inputs[0].clone(),
                    join_node.dataset.clone(),
                    join_node.join_column.clone(),
                    join_node.index_name.clone(),
                    join_node.join_type,
                )?;
                Some(Arc::new(exec))
            } else {
                None
            },
        )
    }
}

/// Logical node that adds a `_rowaddr` column based on an existing `_rowid` column.
///
/// This node converts `_rowid` to `_rowaddr` while preserving table qualifications.
/// For example, if input has `target._rowid`, output will have `target._rowaddr`.
#[derive(Debug)]
pub struct AddRowAddrNode {
    input: LogicalPlan,
    dataset: Arc<Dataset>,
    rowaddr_pos: usize,
    schema: Arc<DFSchema>,
}

impl PartialEq for AddRowAddrNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.dataset.base == other.dataset.base
            && self.rowaddr_pos == other.rowaddr_pos
    }
}

impl Eq for AddRowAddrNode {}

impl std::hash::Hash for AddRowAddrNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.dataset.base.hash(state);
        self.rowaddr_pos.hash(state);
    }
}

impl PartialOrd for AddRowAddrNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.input.partial_cmp(&other.input)
    }
}

impl AddRowAddrNode {
    pub fn try_new(
        input: LogicalPlan,
        dataset: Arc<Dataset>,
        rowaddr_pos: usize,
    ) -> datafusion::error::Result<Self> {
        let input_schema = input.schema();

        // Find the _rowid column and its qualifier
        let mut rowid_qualifier = None;
        let mut found_rowid = false;

        for i in 0..input_schema.fields().len() {
            let (qualifier, field) = input_schema.qualified_field(i);
            if field.name() == lance_core::ROW_ID {
                rowid_qualifier = qualifier.cloned();
                found_rowid = true;
                break;
            }
        }

        if !found_rowid {
            return Err(datafusion::error::DataFusionError::Internal(
                "AddRowAddrNode requires _rowid column in input".to_string(),
            ));
        }

        // Build output schema by adding _rowaddr column with same qualifier as _rowid
        let mut qualified_fields = Vec::new();
        for i in 0..input_schema.fields().len() {
            let (qualifier, field) = input_schema.qualified_field(i);
            if i == rowaddr_pos {
                // Insert _rowaddr column here with same qualifier as _rowid
                qualified_fields.push((
                    rowid_qualifier.clone(),
                    Arc::new(lance_core::ROW_ADDR_FIELD.clone()),
                ));
            }
            qualified_fields.push((qualifier.cloned(), Arc::new(field.clone())));
        }

        // If rowaddr_pos is at the end, add it after all existing fields
        if rowaddr_pos >= input_schema.fields().len() {
            qualified_fields.push((
                rowid_qualifier,
                Arc::new(lance_core::ROW_ADDR_FIELD.clone()),
            ));
        }

        let schema = Arc::new(DFSchema::new_with_metadata(
            qualified_fields,
            input_schema.metadata().clone(),
        )?);

        Ok(Self {
            input,
            dataset,
            rowaddr_pos,
            schema,
        })
    }
}

impl UserDefinedLogicalNodeCore for AddRowAddrNode {
    fn name(&self) -> &str {
        "AddRowAddr"
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
        write!(f, "AddRowAddr")
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<datafusion_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::error::Result<Self> {
        if !exprs.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "AddRowAddrNode does not accept expressions".to_string(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "AddRowAddrNode requires exactly one input".to_string(),
            ));
        }
        Self::try_new(inputs[0].clone(), self.dataset.clone(), self.rowaddr_pos)
    }
}

/// Physical planner for AddRowAddrNode.
pub struct AddRowAddrPlanner {}

#[async_trait]
impl ExtensionPlanner for AddRowAddrPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        Ok(
            if let Some(add_rowaddr_node) = node.as_any().downcast_ref::<AddRowAddrNode>() {
                assert_eq!(logical_inputs.len(), 1, "Inconsistent number of inputs");
                assert_eq!(physical_inputs.len(), 1, "Inconsistent number of inputs");

                let exec = crate::io::exec::AddRowAddrExec::try_new(
                    physical_inputs[0].clone(),
                    add_rowaddr_node.dataset.clone(),
                    add_rowaddr_node.rowaddr_pos,
                )?;
                Some(Arc::new(exec))
            } else {
                None
            },
        )
    }
}
