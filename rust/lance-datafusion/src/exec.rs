// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with datafusion execution plans

use std::sync::{Arc, Mutex};

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    dataframe::DataFrame,
    datasource::streaming::StreamingTable,
    execution::{
        context::{SessionConfig, SessionContext},
        disk_manager::DiskManagerConfig,
        memory_pool::FairSpillPool,
        runtime_env::{RuntimeConfig, RuntimeEnv},
        TaskContext,
    },
    physical_plan::{
        display::DisplayableExecutionPlan, stream::RecordBatchStreamAdapter,
        streaming::PartitionStream, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
        SendableRecordBatchStream,
    },
};
use datafusion_common::{DataFusionError, Statistics};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use lazy_static::lazy_static;

use futures::stream;
use lance_arrow::SchemaExt;
use lance_core::Result;
use log::{debug, info, warn};

/// An source execution node created from an existing stream
///
/// It can only be used once, and will return the stream.  After that the node
/// is exhuasted.
///
/// Note: the stream should be finite, otherwise we will report datafusion properties
/// incorrectly.
pub struct OneShotExec {
    stream: Mutex<Option<SendableRecordBatchStream>>,
    // We save off a copy of the schema to speed up formatting and so ExecutionPlan::schema & display_as
    // can still function after exhuasted
    schema: Arc<ArrowSchema>,
    properties: PlanProperties,
}

impl OneShotExec {
    /// Create a new instance from a given stream
    pub fn new(stream: SendableRecordBatchStream) -> Self {
        let schema = stream.schema().clone();
        Self {
            stream: Mutex::new(Some(stream)),
            schema: schema.clone(),
            properties: PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::RoundRobinBatch(1),
                datafusion::physical_plan::ExecutionMode::Bounded,
            ),
        }
    }

    pub fn from_batch(batch: RecordBatch) -> Self {
        let schema = batch.schema();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch)]),
        ));
        Self::new(stream)
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
    fn name(&self) -> &str {
        "OneShotExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
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
        Ok(Statistics::new_unknown(&self.schema))
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        &self.properties
    }
}

#[derive(Debug, Default, Clone)]
pub struct LanceExecutionOptions {
    pub use_spilling: bool,
    pub mem_pool_size: Option<u64>,
}

const DEFAULT_LANCE_MEM_POOL_SIZE: u64 = 100 * 1024 * 1024;

impl LanceExecutionOptions {
    pub fn mem_pool_size(&self) -> u64 {
        self.mem_pool_size.unwrap_or_else(|| {
            std::env::var("LANCE_MEM_POOL_SIZE")
                .map(|s| match s.parse::<u64>() {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("Failed to parse LANCE_MEM_POOL_SIZE: {}, using default", e);
                        DEFAULT_LANCE_MEM_POOL_SIZE
                    }
                })
                .unwrap_or(DEFAULT_LANCE_MEM_POOL_SIZE)
        })
    }

    pub fn use_spilling(&self) -> bool {
        if !self.use_spilling {
            return false;
        }
        std::env::var("LANCE_BYPASS_SPILLING")
            .map(|_| {
                info!("Bypassing spilling because LANCE_BYPASS_SPILLING is set");
                false
            })
            .unwrap_or(true)
    }
}

pub fn new_session_context(options: LanceExecutionOptions) -> SessionContext {
    let session_config = SessionConfig::new();
    let mut runtime_config = RuntimeConfig::new();
    if options.use_spilling() {
        runtime_config.disk_manager = DiskManagerConfig::NewOs;
        runtime_config.memory_pool = Some(Arc::new(FairSpillPool::new(
            options.mem_pool_size() as usize
        )));
    }
    let runtime_env = Arc::new(RuntimeEnv::new(runtime_config).unwrap());
    SessionContext::new_with_config_rt(session_config, runtime_env)
}

lazy_static! {
    static ref DEFAULT_SESSION_CONTEXT: SessionContext =
        new_session_context(LanceExecutionOptions::default());
    static ref DEFAULT_SESSION_CONTEXT_WITH_SPILLING: SessionContext = {
        new_session_context(LanceExecutionOptions {
            use_spilling: true,
            ..Default::default()
        })
    };
}

pub fn get_session_context(options: LanceExecutionOptions) -> SessionContext {
    let session_ctx: SessionContext;
    if options.mem_pool_size() == DEFAULT_LANCE_MEM_POOL_SIZE {
        if options.use_spilling() {
            session_ctx = DEFAULT_SESSION_CONTEXT_WITH_SPILLING.clone();
        } else {
            session_ctx = DEFAULT_SESSION_CONTEXT.clone();
        }
    } else {
        session_ctx = new_session_context(options)
    }
    session_ctx
}

/// Executes a plan using default session & runtime configuration
///
/// Only executes a single partition.  Panics if the plan has more than one partition.
pub fn execute_plan(
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
) -> Result<SendableRecordBatchStream> {
    debug!(
        "Executing plan:\n{}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );

    let session_ctx = get_session_context(options);

    // NOTE: we are only executing the first partition here. Therefore, if
    // the plan has more than one partition, we will be missing data.
    assert_eq!(plan.properties().partitioning.partition_count(), 1);
    Ok(plan.execute(0, session_ctx.task_ctx())?)
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
