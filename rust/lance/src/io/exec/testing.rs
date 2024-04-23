// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Testing Node
//!

use std::any::Any;
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::{
    common::Statistics,
    execution::context::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, PlanProperties,
        SendableRecordBatchStream,
    },
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};

#[derive(Debug)]
pub struct TestingExec {
    pub(crate) batches: Vec<RecordBatch>,
    properties: PlanProperties,
}

impl TestingExec {
    pub(crate) fn new(batches: Vec<RecordBatch>) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(batches[0].schema().clone()),
            Partitioning::RoundRobinBatch(1),
            ExecutionMode::Bounded,
        );
        Self {
            batches,
            properties,
        }
    }
}

impl DisplayAs for TestingExec {
    fn fmt_as(&self, t: DisplayFormatType, _f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => todo!(),
        }
    }
}

impl ExecutionPlan for TestingExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.batches[0].schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<std::sync::Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        todo!()
    }

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        Ok(Statistics::new_unknown(self.schema().as_ref()))
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        &self.properties
    }
}
