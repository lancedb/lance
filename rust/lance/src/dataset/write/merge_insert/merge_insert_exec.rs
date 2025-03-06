// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    cmp::Ordering,
    sync::{Arc, LazyLock},
};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::{
    common::DFSchema,
    execution::{SendableRecordBatchStream, SessionState, TaskContext},
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        DisplayAs, ExecutionPlan, PlanProperties,
    },
    physical_planner::{ExtensionPlanner, PhysicalPlanner},
};
use datafusion::{
    common::Result as DFResult,
    physical_plan::metrics::{Count, MetricBuilder},
};
use datafusion_expr::{LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};

use crate::Dataset;

use super::MergeInsertParams;

#[derive(Clone, PartialEq, Eq, Hash, Debug, Copy)]
enum MergeInsertWriteStyle {
    FullSchema,
    PartialUpdate,
}

static MERGE_STATS_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("num_inserted_rows", DataType::UInt64, false),
        Field::new("num_updated_rows", DataType::UInt64, false),
        Field::new("num_deleted_rows", DataType::UInt64, false),
    ]))
});

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

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by moving updated rows to new fragments. This mode
/// is most optimal when updating the full schema.
///
/// It returns a single batch, containing the statistics.
#[derive(Debug)]
struct FullSchemaMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl FullSchemaMergeInsertExec {
    fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
    ) -> DFResult<Self> {
        let properties = PlanProperties::new(
            EquivalenceProperties::new((*MERGE_STATS_SCHEMA).clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self {
            input,
            dataset,
            params,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for FullSchemaMergeInsertExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "FullSchemaMergeInsertExec")
    }
}

impl ExecutionPlan for FullSchemaMergeInsertExec {
    fn name(&self) -> &str {
        "FullSchemaMergeInsertExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        (*MERGE_STATS_SCHEMA).clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "FullSchemaMergeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        todo!("Also record the metrics here")
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let merge_metrics = MergeInsertMetrics::new(&self.metrics, partition);

        todo!("Execute FullSchemaMergeInsertExec")
    }
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
#[derive(Debug)]
struct PartialUpdateMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl PartialUpdateMergeInsertExec {
    fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
    ) -> DFResult<Self> {
        let properties = PlanProperties::new(
            EquivalenceProperties::new((*MERGE_STATS_SCHEMA).clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self {
            input,
            dataset,
            params,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for PartialUpdateMergeInsertExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "PartialUpdateMergeInsertExec")
    }
}

impl ExecutionPlan for PartialUpdateMergeInsertExec {
    fn name(&self) -> &str {
        "PartialUpdateMergeInsertExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        (*MERGE_STATS_SCHEMA).clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "PartialUpdateMergeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        todo!("Also record the metrics here")
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let merge_metrics = MergeInsertMetrics::new(&self.metrics, partition);

        todo!("Execute PartialUpdateMergeInsertExec")
    }
}

struct MergeInsertMetrics {
    pub num_inserted_rows: Count,
    pub num_updated_rows: Count,
    pub num_deleted_rows: Count,
}

impl MergeInsertMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let num_inserted_rows = MetricBuilder::new(metrics).counter("num_inserted_rows", partition);
        let num_updated_rows = MetricBuilder::new(metrics).counter("num_updated_rows", partition);
        let num_deleted_rows = MetricBuilder::new(metrics).counter("num_deleted_rows", partition);
        Self {
            num_inserted_rows,
            num_updated_rows,
            num_deleted_rows,
        }
    }

    pub fn as_batch(&self) -> RecordBatch {
        let num_inserted_rows = UInt64Array::from(vec![self.num_inserted_rows.value() as u64]);
        let num_updated_rows = UInt64Array::from(vec![self.num_updated_rows.value() as u64]);
        let num_deleted_rows = UInt64Array::from(vec![self.num_deleted_rows.value() as u64]);
        RecordBatch::try_new(
            (*MERGE_STATS_SCHEMA).clone(),
            vec![
                Arc::new(num_inserted_rows),
                Arc::new(num_updated_rows),
                Arc::new(num_deleted_rows),
            ],
        )
        .expect("Failed to create merge insert statistics batch")
    }
}
