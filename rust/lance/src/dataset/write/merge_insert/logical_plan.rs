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
use std::{cmp::Ordering, sync::Arc};

use crate::{
    dataset::write::merge_insert::exec::{FullSchemaMergeInsertExec, PartialUpdateMergeInsertExec},
    Dataset,
};

use super::{exec::MERGE_STATS_SCHEMA, MergeInsertParams};

#[derive(Clone, PartialEq, Eq, Hash, Debug, Copy)]
enum MergeInsertWriteStyle {
    FullSchema,
    PartialUpdate,
}

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
#[derive(Debug)]
struct MergeInsertWriteNode {
    input: LogicalPlan,
    pub(crate) dataset: Arc<Dataset>,
    pub(crate) params: MergeInsertParams,
    pub(crate) write_style: MergeInsertWriteStyle,
    schema: Arc<DFSchema>,
}

impl PartialEq for MergeInsertWriteNode {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
            && self.input == other.input
            && self.write_style == other.write_style
            && self.dataset.base == other.dataset.base
    }
}

impl Eq for MergeInsertWriteNode {}

impl std::hash::Hash for MergeInsertWriteNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.params.hash(state);
        self.input.hash(state);
        self.write_style.hash(state);
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
    fn new(
        input: LogicalPlan,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
        write_style: MergeInsertWriteStyle,
    ) -> Self {
        let schema = Arc::new(DFSchema::try_from((*MERGE_STATS_SCHEMA).clone()).unwrap());
        Self {
            input,
            dataset,
            params,
            schema,
            write_style,
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
        Ok(Self::new(
            inputs[0].clone(),
            self.dataset.clone(),
            self.params.clone(),
            self.write_style,
        ))
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        todo!("Compute which columns are necessary for the write")
    }
}

/// Physical planner for MergeInsertWriteNode.
struct MergeInsertPlanner {}

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
                match write_node.write_style {
                    MergeInsertWriteStyle::FullSchema => {
                        let exec = FullSchemaMergeInsertExec::try_new(
                            physical_inputs[0].clone(),
                            write_node.dataset.clone(),
                            write_node.params.clone(),
                        )?;
                        Some(Arc::new(exec))
                    }
                    MergeInsertWriteStyle::PartialUpdate => {
                        let exec = PartialUpdateMergeInsertExec::try_new(
                            physical_inputs[0].clone(),
                            write_node.dataset.clone(),
                            write_node.params.clone(),
                        )?;
                        Some(Arc::new(exec))
                    }
                }
            } else {
                None
            },
        )
    }
}
