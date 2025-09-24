// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::datafusion::LanceTableProvider;
use crate::Dataset;
use arrow_array::RecordBatch;
use datafusion::dataframe::DataFrame;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use std::sync::Arc;

/// A SQL builder to prepare options for running SQL queries against a Lance dataset.
#[derive(Clone, Debug)]
pub struct SqlQueryBuilder {
    /// The dataset to run the SQL query
    pub(crate) dataset: Arc<Dataset>,

    /// The SQL query to run
    pub(crate) sql: String,

    /// the name of the table to register in the datafusion context
    pub(crate) table_name: String,

    /// If true, the query result will include the internal row id
    pub(crate) with_row_id: bool,

    /// If true, the query result will include the internal row address
    pub(crate) with_row_addr: bool,
}

impl SqlQueryBuilder {
    pub fn new(dataset: Dataset, sql: &str) -> Self {
        Self {
            dataset: Arc::new(dataset),
            sql: sql.to_string(),
            table_name: "dataset".to_string(),
            with_row_id: false,
            with_row_addr: false,
        }
    }

    /// The table name to register in the datafusion context.
    /// This is used to specify a "table name" for the dataset.
    /// So that you can run SQL queries against it.
    /// If not set, the default table name is "dataset".
    pub fn table_name(mut self, table_name: &str) -> Self {
        self.table_name = table_name.to_string();
        self
    }

    /// Specify if the query result should include the internal row id.
    /// If true, the query result will include an additional column named "_rowid".
    pub fn with_row_id(mut self, row_id: bool) -> Self {
        self.with_row_id = row_id;
        self
    }

    /// Specify if the query result should include the internal row address.
    /// If true, the query result will include an additional column named "_rowaddr".
    pub fn with_row_addr(mut self, row_addr: bool) -> Self {
        self.with_row_addr = row_addr;
        self
    }

    pub async fn build(self) -> lance_core::Result<SqlQuery> {
        let ctx = SessionContext::new();
        let row_id = self.with_row_id;
        let row_addr = self.with_row_addr;
        ctx.register_table(
            self.table_name,
            Arc::new(LanceTableProvider::new(
                self.dataset.clone(),
                row_id,
                row_addr,
            )),
        )?;
        let df = ctx.sql(&self.sql).await?;
        Ok(SqlQuery::new(df))
    }
}

pub struct SqlQuery {
    dataframe: DataFrame,
}

impl SqlQuery {
    pub fn new(dataframe: DataFrame) -> Self {
        Self { dataframe }
    }

    pub async fn into_stream(self) -> lance_core::Result<SendableRecordBatchStream> {
        self.dataframe.execute_stream().await.map_err(|e| e.into())
    }

    pub async fn into_batch_records(self) -> lance_core::Result<Vec<RecordBatch>> {
        self.into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| e.into())
    }

    pub fn into_dataframe(self) -> DataFrame {
        self.dataframe
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::test::{assert_string_matches, DatagenExt, FragmentCount, FragmentRowCount};

    use all_asserts::assert_true;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type, UInt64Type};

    use lance_datagen::{array, gen_batch};

    #[tokio::test]
    async fn test_sql_execute() {
        let mut ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("SELECT SUM(x) FROM foo WHERE y > 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        pretty_assertions::assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 1);
        pretty_assertions::assert_eq!(results.num_rows(), 1);
        // SUM(0..100) - SUM(0..50) = 3675
        pretty_assertions::assert_eq!(results.column(0).as_primitive::<Int64Type>().value(0), 3675);

        let results = ds
            .sql("SELECT x, y, _rowid, _rowaddr FROM foo where y > 100")
            .table_name("foo")
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let total_rows: usize = results.iter().map(|batch| batch.num_rows()).sum();
        let expect_rows = ds.count_rows(Some("y > 100".to_string())).await.unwrap();
        pretty_assertions::assert_eq!(total_rows, expect_rows);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 4);
        assert_true!(results.column(2).as_primitive::<UInt64Type>().value(0) > 100);
        assert_true!(results.column(3).as_primitive::<UInt64Type>().value(0) > 100);
    }

    #[tokio::test]
    async fn test_sql_count() {
        let mut ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("SELECT COUNT(*) FROM foo")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        pretty_assertions::assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 1);
        pretty_assertions::assert_eq!(results.num_rows(), 1);
        pretty_assertions::assert_eq!(results.column(0).as_primitive::<Int64Type>().value(0), 100);

        let results = ds
            .sql("SELECT COUNT(*) FROM foo where y >= 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        pretty_assertions::assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(results.num_columns(), 1);
        pretty_assertions::assert_eq!(results.num_rows(), 1);
        pretty_assertions::assert_eq!(results.column(0).as_primitive::<Int64Type>().value(0), 50);
    }

    #[tokio::test]
    async fn test_explain() {
        let mut ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("EXPLAIN SELECT * FROM foo where y >= 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let results = results.into_iter().next().unwrap();

        let plan = format!("{:?}", results);
        let expected_pattern = r#"...columns: [StringArray
[
  "logical_plan",
  "physical_plan",
], StringArray
[
  "TableScan: foo projection=[x, y], full_filters=[foo.y >= Int32(100)]",
  "ProjectionExec: expr=[x@0 as x, y@1 as y]\n  CooperativeExec\n    LanceRead: uri=test_sql_dataset/data, projection=[x, y], num_fragments=10, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=y >= Int32(100), refine_filter=y >= Int32(100)\n",
]], row_count: 2 }"#;
        assert_string_matches(&plan, expected_pattern).unwrap();
    }

    #[tokio::test]
    async fn test_analyze() {
        let mut ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .col("y", array::step_custom::<Int32Type>(0, 2))
            .into_dataset(
                "memory://test_sql_dataset",
                FragmentCount::from(10),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        let results = ds
            .sql("EXPLAIN ANALYZE SELECT * FROM foo where y >= 100")
            .table_name("foo")
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let results = results.into_iter().next().unwrap();

        let plan = format!("{:?}", results);
        let expected_pattern = r#"...columns: [StringArray
[
  "Plan with Metrics",
], StringArray
[
  "ProjectionExec: expr=[x@0 as x, y@1 as y], metrics=[output_rows=50, elapsed_compute=...]\n  CooperativeExec, metrics=[]\n    LanceRead: uri=test_sql_dataset/data, projection=[x, y], num_fragments=..., range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=y >= Int32(100), refine_filter=y >= Int32(100), metrics=[output_rows=..., elapsed_compute=..., fragments_scanned=..., ranges_scanned=..., rows_scanned=..., bytes_read=..., iops=..., requests=..., task_wait_time=...]\n",
]], row_count: 1 }"#;
        assert_string_matches(&plan, expected_pattern).unwrap();
    }
}
