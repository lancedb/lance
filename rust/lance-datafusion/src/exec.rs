// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with datafusion execution plans

use std::sync::{Arc, Mutex};

use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    dataframe::DataFrame,
    datasource::streaming::StreamingTable,
    execution::{
        context::{SessionConfig, SessionContext, SessionState},
        disk_manager::DiskManagerConfig,
        memory_pool::FairSpillPool,
        runtime_env::{RuntimeConfig, RuntimeEnv},
        TaskContext,
    },
    physical_plan::{
        streaming::PartitionStream, DisplayAs, DisplayFormatType, ExecutionPlan,
        SendableRecordBatchStream,
    },
};
use datafusion_common::DataFusionError;
use datafusion_physical_expr::Partitioning;

use lance_arrow::SchemaExt;
use lance_core::Result;
use log::{info, warn};

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

/// Executes a plan using default session & runtime configuration
///
/// Only executes a single partition.  Panics if the plan has more than one partition.
pub fn execute_plan(
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
) -> Result<SendableRecordBatchStream> {
    let session_config = SessionConfig::new();
    let mut runtime_config = RuntimeConfig::new();
    if options.use_spilling() {
        runtime_config.disk_manager = DiskManagerConfig::NewOs;
        runtime_config.memory_pool = Some(Arc::new(FairSpillPool::new(
            options.mem_pool_size() as usize
        )));
    }
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
