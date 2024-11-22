use std::{collections::HashMap, sync::Arc};

use lance_datafusion::exec::{new_session_context, LanceExecutionOptions};

use crate::{dataset::scanner::DatasetRecordBatchStream, Dataset, Result};

use super::dataframe::LanceTableProvider;

/// An SQL query that can be executed
pub struct SqlPlan {
    query: String,
    context: HashMap<String, Arc<Dataset>>,
    execution_options: LanceExecutionOptions,
}

impl SqlPlan {
    /// Creates a new SQL with a given query string and context
    ///
    /// The context is a mapping of dataset aliases to datasets.
    /// This is how the SQL query can reference datasets.
    pub fn new(query: String, context: HashMap<String, Arc<Dataset>>) -> Self {
        Self {
            query,
            context,
            execution_options: LanceExecutionOptions::default(),
        }
    }

    /// Executes the SQL query and returns a stream of record batches
    pub async fn execute(&self) -> Result<DatasetRecordBatchStream> {
        let session_context = new_session_context(&self.execution_options);

        for (alias, dataset) in &self.context {
            let provider = Arc::new(LanceTableProvider::new(
                dataset.clone(),
                /*with_row_id= */ true,
                /*with_row_addr= */ true,
            ));
            session_context.register_table(alias, provider)?;
        }

        let df = session_context.sql(&self.query).await?;
        let stream = df.execute_stream().await?;

        Ok(DatasetRecordBatchStream::new(stream))
    }
}
