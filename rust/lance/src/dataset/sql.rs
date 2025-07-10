// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::datafusion::LanceTableProvider;
use crate::dataset::SqlOptions;
use crate::Dataset;
use arrow_array::RecordBatch;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::prelude::SessionContext;
use lance_core::Error;
use snafu::location;
use std::sync::Arc;

/// Customize the params of dataset's sql API.
#[derive(Clone, Debug)]
pub struct SqlOptions {
    /// the dataset to run the SQL query
    dataset: Option<Dataset>,

    /// the SQL query to run
    sql: String,

    /// the name of the table to register in the datafusion context
    table_name: String,

    /// if true, the query result will include the internal row id
    row_id: bool,

    /// if true, the query result will include the internal row address
    row_addr: bool,
}

impl SqlOptions {
    pub fn table_name(mut self, table_name: &str) -> Self {
        self.table_name = table_name.to_string();
        self
    }

    pub fn with_row_id(mut self, row_id: bool) -> Self {
        self.row_id = row_id;
        self
    }

    pub fn with_row_addr(mut self, row_addr: bool) -> Self {
        self.row_addr = row_addr;
        self
    }

    pub async fn execute(self) -> lance_core::Result<QueryResult> {
        let ctx = SessionContext::new();
        ctx.register_table(
            self.table_name,
            Arc::new(LanceTableProvider::new(
                Arc::new(self.dataset.unwrap()),
                self.row_id,
                self.row_addr,
            )),
        )?;
        let df = ctx.sql(&self.sql).await?;
        let result_stream = df.execute_stream().await.unwrap();
        Ok(QueryResult {
            stream: result_stream,
        })
    }
}

impl Default for SqlOptions {
    fn default() -> Self {
        Self {
            dataset: None,
            sql: "".to_string(),
            table_name: "".to_string(),
            row_id: false,
            row_addr: false,
        }
    }
}

pub struct QueryResult {
    stream: SendableRecordBatchStream,
}

impl QueryResult {
    pub fn into_stream(self) -> SendableRecordBatchStream {
        self.stream
    }

    pub async fn collect(self) -> lance_core::Result<Vec<RecordBatch>> {
        use futures::TryStreamExt;
        self.stream
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::DataFusionInnerError {
                source: e.into(),
                location: location!(),
            })
    }
}
