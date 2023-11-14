// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Utilities for working with datafusion execution plans

use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatchReader;
use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    execution::{
        context::{SessionConfig, SessionState},
        runtime_env::{RuntimeConfig, RuntimeEnv},
    },
    physical_plan::{
        stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionPlan,
        SendableRecordBatchStream,
    },
};
use datafusion_common::DataFusionError;
use datafusion_physical_expr::Partitioning;
use futures::TryStreamExt;

use lance_arrow::SchemaExt;
use lance_core::{datatypes::Schema, Error, Result};

pub struct OneShotExec {
    stream: Mutex<RefCell<Option<SendableRecordBatchStream>>>,
    // We save off a copy of the schema to speed up formatting and so ExecutionPlan::schema & display_as
    // can still function after exhuasted
    schema: Arc<ArrowSchema>,
}

impl OneShotExec {
    pub fn new(stream: SendableRecordBatchStream) -> Self {
        let schema = stream.schema().clone();
        Self {
            stream: Mutex::new(RefCell::new(Some(stream))),
            schema,
        }
    }
}

impl std::fmt::Debug for OneShotExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val_guard = self.stream.lock().unwrap();
        let stream = val_guard.borrow();
        f.debug_struct("OneShotExec")
            .field("exhausted", &stream.is_none())
            .field("schema", self.schema.as_ref())
            .finish()
    }
}

impl DisplayAs for OneShotExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let val_guard = self.stream.lock().unwrap();
        let stream = val_guard.borrow();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let exhausted = if stream.is_some() { "" } else { "EXHUASTED" };
                let columns = self
                    .schema
                    .field_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                write!(
                    f,
                    "OneShotStream: {} columns=[{}]",
                    exhausted,
                    columns.join(",")
                )
            }
        }
    }
}

/// Convert reader to a stream and a schema.
///
/// Will peek the first batch to get the dictionaries for dictionary columns.
///
/// NOTE: this does not validate the schema. For example, for appends the schema
/// should be checked to make sure it matches the existing dataset schema before
/// writing.
pub fn reader_to_stream(
    batches: Box<dyn RecordBatchReader + Send>,
) -> Result<(SendableRecordBatchStream, Schema)> {
    let arrow_schema = batches.schema();
    let mut schema: Schema = Schema::try_from(batches.schema().as_ref())?;
    let mut peekable = batches.peekable();
    if let Some(batch) = peekable.peek() {
        if let Ok(b) = batch {
            schema.set_dictionary(b)?;
        } else {
            return Err(Error::from(batch.as_ref().unwrap_err()));
        }
    }
    schema.validate()?;

    let stream = RecordBatchStreamAdapter::new(
        arrow_schema,
        futures::stream::iter(peekable).map_err(DataFusionError::from),
    );
    let stream = Box::pin(stream) as SendableRecordBatchStream;

    Ok((stream, schema))
}

impl ExecutionPlan for OneShotExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> datafusion_physical_expr::Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion_physical_expr::PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        let mut val_guard = self.stream.lock().unwrap();
        let stream = val_guard.get_mut();
        let stream = stream.take();
        if let Some(stream) = stream {
            Ok(stream)
        } else {
            panic!("Attempt to use OneShotExec more than once");
        }
    }

    fn statistics(&self) -> datafusion_common::Statistics {
        todo!()
    }
}

pub fn execute_plan(plan: Arc<dyn ExecutionPlan>) -> Result<SendableRecordBatchStream> {
    let session_config = SessionConfig::new();
    let runtime_config = RuntimeConfig::new();
    let runtime_env = Arc::new(RuntimeEnv::new(runtime_config)?);
    let session_state = SessionState::new_with_config_rt(session_config, runtime_env);
    // NOTE: we are only executing the first partition here. Therefore, if
    // the plan has more than one partition, we will be missing data.
    assert_eq!(plan.output_partitioning().partition_count(), 1);
    Ok(plan.execute(0, session_state.task_ctx())?)
}
