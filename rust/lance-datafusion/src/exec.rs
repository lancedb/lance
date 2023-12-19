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

use std::sync::{Arc, Mutex};

use arrow_array::RecordBatchReader;
use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    dataframe::DataFrame,
    datasource::streaming::StreamingTable,
    execution::{
        context::{SessionConfig, SessionContext, SessionState},
        runtime_env::{RuntimeConfig, RuntimeEnv},
        TaskContext,
    },
    physical_plan::{
        stream::RecordBatchStreamAdapter, streaming::PartitionStream, DisplayAs, DisplayFormatType,
        ExecutionPlan, SendableRecordBatchStream,
    },
};
use datafusion_common::DataFusionError;
use datafusion_physical_expr::Partitioning;
use futures::TryStreamExt;

use lance_arrow::SchemaExt;
use lance_core::{datatypes::Schema, Error, Result};

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

/// An source execution node created from an existing stream
///
/// It can only be used once, and will return the stream.  After that the node
/// is exhuasted.
pub struct OneShotExec {
    stream: Mutex<Option<SendableRecordBatchStream>>,
    // We save off a copy of the schema to speed up formatting and so ExecutionPlan::schema & display_as
    // can still function after exhuasted
    schema: Arc<ArrowSchema>,
}

impl OneShotExec {
    /// Create a new instance from a given stream
    pub fn new(stream: SendableRecordBatchStream) -> Self {
        let schema = stream.schema().clone();
        Self {
            stream: Mutex::new(Some(stream)),
            schema,
        }
    }
}

impl std::fmt::Debug for OneShotExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stream = self.stream.lock().unwrap();
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
        let stream = self.stream.lock().unwrap();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let exhausted = if stream.is_some() { "" } else { "EXHUASTED " };
                let columns = self
                    .schema
                    .field_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                write!(
                    f,
                    "OneShotStream: {}columns=[{}]",
                    exhausted,
                    columns.join(",")
                )
            }
        }
    }
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
        let stream = self
            .stream
            .lock()
            .map_err(|err| DataFusionError::Execution(err.to_string()))?
            .take();
        if let Some(stream) = stream {
            Ok(stream)
        } else {
            Err(DataFusionError::Execution(
                "OneShotExec has already been executed".to_string(),
            ))
        }
    }

    fn statistics(&self) -> datafusion_common::Result<datafusion_common::Statistics> {
        todo!()
    }
}

/// Executes a plan using default session & runtime configuration
///
/// Only executes a single partition.  Panics if the plan has more than one partition.
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

pub trait SessionContextExt {
    /// Creates a DataFrame for reading a stream of data
    ///
    /// This dataframe may only be queried once, future queries will fail
    fn read_one_shot(
        &self,
        data: SendableRecordBatchStream,
    ) -> datafusion::common::Result<DataFrame>;
}

struct OneShotPartitionStream {
    data: Arc<Mutex<Option<SendableRecordBatchStream>>>,
    schema: Arc<ArrowSchema>,
}

impl OneShotPartitionStream {
    fn new(data: SendableRecordBatchStream) -> Self {
        let schema = data.schema().clone();
        Self {
            data: Arc::new(Mutex::new(Some(data))),
            schema,
        }
    }
}

impl PartitionStream for OneShotPartitionStream {
    fn schema(&self) -> &arrow_schema::SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let mut stream = self.data.lock().unwrap();
        stream
            .take()
            .expect("Attempt to consume a one shot dataframe multiple times")
    }
}

impl SessionContextExt for SessionContext {
    fn read_one_shot(
        &self,
        data: SendableRecordBatchStream,
    ) -> datafusion::common::Result<DataFrame> {
        let schema = data.schema().clone();
        let part_stream = Arc::new(OneShotPartitionStream::new(data));
        let provider = StreamingTable::try_new(schema, vec![part_stream])?;
        self.read_table(Arc::new(provider))
    }
}
