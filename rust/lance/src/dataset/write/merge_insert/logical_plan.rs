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
use lance_core::ROW_ADDR;
use std::{cmp::Ordering, sync::Arc};

use crate::{dataset::write::merge_insert::exec::FullSchemaMergeInsertExec, Dataset};

use super::MergeInsertParams;

/// Logical plan node for merge insert write.
///
/// Expects input schema:
/// * `source.{col1, col2, ...}` - columns from the source relation
/// * `target.{col1, col2, ...}` - columns from the target relation
/// * `target._rowaddr` - special column to locate existing rows in the target
/// * `action` - unqualified column that describes the action to perform.
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
        // * `action` column (unqualified)
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

                // Include unqualified columns like "action" - tells us what operation to perform
                None if field.name() == "action" => true,

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
