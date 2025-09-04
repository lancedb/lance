// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::datafusion::LanceTableProvider;
use crate::Dataset;
use arrow_array::RecordBatch;
use datafusion::dataframe::DataFrame;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::prelude::SessionContext;
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
        // Create SessionContext with Lance physical optimizers
        use datafusion::execution::session_state::SessionStateBuilder;
        use std::sync::Arc;

        // Get the optimizer rules from the get_physical_optimizer function
        let optimizer = crate::io::exec::get_physical_optimizer();
        let mut builder = SessionStateBuilder::new().with_default_features();

        // Add each Lance physical optimizer rule to the session
        for rule in optimizer.rules {
            builder = builder.with_physical_optimizer_rule(rule);
        }

        let state = builder.build();

        let ctx = SessionContext::new_with_state(state);

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

    pub async fn into_stream(self) -> SendableRecordBatchStream {
        self.dataframe.execute_stream().await.unwrap()
    }

    pub async fn into_batch_records(self) -> lance_core::Result<Vec<RecordBatch>> {
        use futures::TryStreamExt;

        Ok(self
            .dataframe
            .execute_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?)
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
  "ProjectionExec: expr=[x@0 as x, y@1 as y]\n  LanceRead: uri=test_sql_dataset/data, projection=[x, y], num_fragments=10, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=y >= Int32(100), refine_filter=y >= Int32(100)\n",
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
  "ProjectionExec: expr=[x@0 as x, y@1 as y], metrics=[output_rows=50, elapsed_compute=...]\n  LanceRead: uri=test_sql_dataset/data, projection=[x, y], num_fragments=..., range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=y >= Int32(100), refine_filter=y >= Int32(100), metrics=[output_rows=..., elapsed_compute=..., bytes_read=..., fragments_scanned=..., iops=..., ranges_scanned=..., requests=..., rows_scanned=..., task_wait_time=...]\n",
]], row_count: 1 }"#;
        assert_string_matches(&plan, expected_pattern).unwrap();
    }

    #[tokio::test]
    async fn test_anti_join_not_exists_sql() {
        use crate::Dataset;
        use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use std::sync::Arc;
        use tempfile::tempdir;

        // Create test directory
        let test_dir = tempdir().unwrap();
        let large_table_uri = format!("{}/large_table", test_dir.path().to_str().unwrap());
        let small_table_uri = format!("{}/small_table", test_dir.path().to_str().unwrap());

        // Create schema for both tables
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("value", DataType::Utf8, false),
        ]));

        // Create large table (20 rows - simulating 600M)
        let large_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter(0..20)),
                Arc::new(StringArray::from(
                    (0..20).map(|i| format!("large_{}", i)).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(large_batch)], schema.clone());
        let large_dataset = Dataset::write(reader, &large_table_uri, None)
            .await
            .unwrap();

        // Create small exclusion table (5 rows - simulating 1M)
        // Exclude IDs: 2, 5, 8, 11, 14
        let small_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2, 5, 8, 11, 14])),
                Arc::new(StringArray::from(vec![
                    "excl_2", "excl_5", "excl_8", "excl_11", "excl_14",
                ])),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(small_batch)], schema);
        let small_dataset = Dataset::write(reader, &small_table_uri, None)
            .await
            .unwrap();

        // Register both tables in DataFusion context
        let ctx = datafusion::prelude::SessionContext::new();
        let large_provider = Arc::new(crate::datafusion::LanceTableProvider::new(
            Arc::new(large_dataset.clone()),
            false,
            false,
        ));
        let small_provider = Arc::new(crate::datafusion::LanceTableProvider::new(
            Arc::new(small_dataset.clone()),
            false,
            false,
        ));

        ctx.register_table("large_table", large_provider).unwrap();
        ctx.register_table("small_table", small_provider).unwrap();

        // SQL query with NOT EXISTS (anti-join pattern)
        let sql = r#"
            EXPLAIN ANALYZE SELECT *
            FROM large_table lt
            WHERE NOT EXISTS (
                SELECT 1 FROM small_table st WHERE st.id = lt.id
            )
            LIMIT 10
        "#;

        println!("\n=== SQL NOT EXISTS Test ===");
        println!("Query: {}", sql);
        println!("Large table: 20 rows, Small exclusion table: 5 rows");

        // Execute EXPLAIN ANALYZE to get the plan
        let df = ctx.sql(sql).await.unwrap();
        let results = df.collect().await.unwrap();

        // Print the EXPLAIN ANALYZE output
        println!("\nEXPLAIN ANALYZE Output:");
        for batch in &results {
            println!(
                "Batch: {} columns, {} rows",
                batch.num_columns(),
                batch.num_rows()
            );
            // EXPLAIN ANALYZE returns two columns: plan type and plan details
            if batch.num_columns() >= 2 && batch.num_rows() > 0 {
                let plan_details = batch.column(1); // Second column has the actual plan
                if let Some(string_array) = plan_details.as_any().downcast_ref::<StringArray>() {
                    for i in 0..batch.num_rows() {
                        let plan_text = string_array.value(i);
                        println!("{}", plan_text);
                    }
                }
            }
        }

        // Now run the actual query (without EXPLAIN ANALYZE) to get results
        let actual_sql = r#"
            SELECT *
            FROM large_table lt
            WHERE NOT EXISTS (
                SELECT 1 FROM small_table st WHERE st.id = lt.id
            )
            LIMIT 10
        "#;

        let df = ctx.sql(actual_sql).await.unwrap();
        let results = df.collect().await.unwrap();

        let mut result_ids = Vec::new();
        let excluded_ids = vec![2, 5, 8, 11, 14];

        for batch in &results {
            let id_array = batch.column(0).as_primitive::<Int32Type>();
            for i in 0..id_array.len() {
                let id = id_array.value(i);
                result_ids.push(id);

                // Verify excluded IDs are not in results
                assert!(
                    !excluded_ids.contains(&id),
                    "Found excluded ID {} in NOT EXISTS results",
                    id
                );
            }
        }

        // Should return exactly 10 rows due to LIMIT
        assert_eq!(
            result_ids.len(),
            10,
            "Should return exactly 10 rows due to LIMIT"
        );

        // Expected: first 10 non-excluded IDs
        let expected: Vec<i32> = (0..20)
            .filter(|i| !excluded_ids.contains(i))
            .take(10)
            .collect();
        assert_eq!(
            result_ids, expected,
            "NOT EXISTS results should match expected"
        );

        println!(
            "✅ SQL NOT EXISTS test passed: {} rows returned",
            result_ids.len()
        );
        println!("Returned IDs: {:?}", result_ids);
        println!("Successfully filtered out: {:?}", excluded_ids);
    }
}
