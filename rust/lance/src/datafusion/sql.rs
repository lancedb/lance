use std::collections::HashMap;

use lance_datafusion::exec::{get_session_context, LanceExecutionOptions};

use crate::{dataset::scanner::DatasetRecordBatchStream, Dataset, Result};

pub struct SqlPlan {
    query: String,
    context: HashMap<String, Dataset>,
    execution_options: LanceExecutionOptions,
}

impl SqlPlan {
    pub fn new(query: String) -> Self {
        Self {
            query,
            context: HashMap::new(),
            execution_options: LanceExecutionOptions::default(),
        }
    }

    pub async fn execute(&self) -> Result<DatasetRecordBatchStream> {
        let session_context = get_session_context(&self.execution_options);

        let df = session_context.sql(&self.query).await?;
        let stream = df.execute_stream().await?;

        Ok(DatasetRecordBatchStream::new(stream))
    }
}
