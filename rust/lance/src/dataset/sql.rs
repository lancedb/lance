// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::datafusion::LanceTableProvider;
use crate::dataset::utils::SchemaAdapter;
use arrow_array::RecordBatch;
use datafusion::dataframe::DataFrame;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{Expr as LogicalExpr, LogicalPlan};
use datafusion::prelude::SessionContext;
use datafusion::sql::{
    parser::Statement as DFStatement,
    sqlparser::ast::{Expr, Ident, SelectItem, SetExpr, Statement},
};
use futures::TryStreamExt;
use lance_core::{ROW_ADDR, ROW_ID, datatypes::BlobHandling};
use lance_datafusion::udf::register_functions;
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

    /// Override how blob columns are materialized for this query.
    pub(crate) blob_handling: Option<BlobHandling>,
}

impl SqlQueryBuilder {
    pub fn new(dataset: Dataset, sql: &str) -> Self {
        Self {
            dataset: Arc::new(dataset),
            sql: sql.to_string(),
            table_name: "dataset".to_string(),
            with_row_id: false,
            with_row_addr: false,
            blob_handling: None,
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
    ///
    /// The column is appended only when output rows map one-to-one to dataset
    /// rows. For other queries (DISTINCT, GROUP BY, aggregates, ...) it is not
    /// appended, but can still be referenced explicitly in the SQL text.
    pub fn with_row_id(mut self, row_id: bool) -> Self {
        self.with_row_id = row_id;
        self
    }

    /// Specify if the query result should include the internal row address.
    /// If true, the query result will include an additional column named "_rowaddr".
    ///
    /// The column is appended only when output rows map one-to-one to dataset
    /// rows. For other queries (DISTINCT, GROUP BY, aggregates, ...) it is not
    /// appended, but can still be referenced explicitly in the SQL text.
    pub fn with_row_addr(mut self, row_addr: bool) -> Self {
        self.with_row_addr = row_addr;
        self
    }

    /// Override how blob columns are materialized for this query.
    ///
    /// When unset, the underlying dataset scan uses its default
    /// [`BlobHandling::BlobsDescriptions`] policy.
    pub fn blob_handling(mut self, blob_handling: BlobHandling) -> Self {
        self.blob_handling = Some(blob_handling);
        self
    }

    pub async fn build(self) -> lance_core::Result<SqlQuery> {
        let ctx = SessionContext::new();
        let row_id = self.with_row_id;
        let row_addr = self.with_row_addr;
        let mut provider = LanceTableProvider::new(self.dataset.clone(), row_id, row_addr);
        if let Some(blob_handling) = self.blob_handling {
            provider = provider.with_blob_handling(blob_handling);
        }
        ctx.register_table(self.table_name, Arc::new(provider))?;
        register_functions(&ctx);
        let state = ctx.state();
        let dialect = state.config_options().sql_parser.dialect;
        let statement = state.sql_to_statement(&self.sql, &dialect)?;
        let mut projected = statement.clone();
        let columns = [(self.with_row_id, ROW_ID), (self.with_row_addr, ROW_ADDR)];
        let requested_names: Vec<&str> = columns
            .iter()
            .filter(|(enabled, _)| *enabled)
            .map(|(_, name)| *name)
            .collect();
        let plan = state.statement_to_plan(statement).await?;
        // Append the requested system columns only when the plan proves that
        // every output row maps one-to-one to a scanned source row (otherwise
        // aggregates, DISTINCT, joins, ... would be ambiguous or change the
        // query's relational results) and that no intermediate projection
        // shadows the system column names (otherwise the injected identifiers
        // would bind to derived expressions instead of the real metadata).
        let plan = if safe_to_inject_system_columns(&plan, &requested_names)
            && project_system_columns(&mut projected, &columns)
        {
            match state.statement_to_plan(projected).await {
                Ok(plan) => plan,
                // The rewritten statement can still fail to plan in corner
                // cases (e.g. the user aliased another expression to the same
                // system column name). Fall back to the original plan so the
                // query runs, just without the extra columns.
                Err(_) => plan,
            }
        } else {
            plan
        };
        let df = ctx.execute_logical_plan(plan).await?;
        Ok(SqlQuery::new(df))
    }
}

/// Returns true when injecting the system columns in `names` into the
/// query's top-level SELECT list is provably safe. Two properties must hold:
///
/// 1. Row identity: every output row corresponds to exactly one row scanned
///    from the source table, so the injection cannot change the values or
///    cardinality of the other columns. This is a conservative whitelist of
///    row-preserving operators; anything else (aggregates, DISTINCT, joins,
///    unions, windows, ...) collapses, duplicates, or synthesizes rows.
/// 2. Name lineage: no intermediate projection redefines a system column
///    name (e.g. `SELECT (_rowid + 1) AS _rowid` in a subquery). The
///    injected identifiers resolve by name in the outer scope, so a shadowed
///    name would bind them to a derived expression and return arbitrary
///    values as row metadata.
fn safe_to_inject_system_columns(plan: &LogicalPlan, names: &[&str]) -> bool {
    match plan {
        LogicalPlan::TableScan(_) => true,
        LogicalPlan::Projection(projection) => {
            let shadows_system_column = projection
                .schema
                .fields()
                .iter()
                .zip(&projection.expr)
                .filter(|(field, _)| names.contains(&field.name().as_str()))
                .any(|(field, expr)| {
                    let mut expr = expr;
                    while let LogicalExpr::Alias(alias) = expr {
                        expr = &alias.expr;
                    }
                    !matches!(expr, LogicalExpr::Column(column) if &column.name == field.name())
                });
            !shadows_system_column && safe_to_inject_system_columns(&projection.input, names)
        }
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::SubqueryAlias(_) => plan
            .inputs()
            .iter()
            .all(|input| safe_to_inject_system_columns(input, names)),
        _ => false,
    }
}

/// Appends each enabled system column in `columns` to the statement's SELECT
/// list unless the query already projects it (directly or via a wildcard).
/// Returns true if the statement was modified.
///
/// Only rewrites top-level `SELECT` statements; the caller must separately
/// verify that the injection is safe (see [`safe_to_inject_system_columns`])
/// before planning the rewritten statement.
fn project_system_columns(statement: &mut DFStatement, columns: &[(bool, &str)]) -> bool {
    let DFStatement::Statement(statement) = statement else {
        return false;
    };
    let Statement::Query(query) = statement.as_mut() else {
        return false;
    };
    let SetExpr::Select(select) = query.body.as_mut() else {
        return false;
    };

    let mut changed = false;
    for &(enabled, name) in columns {
        if !enabled {
            continue;
        }
        let already_projected = select
            .projection
            .iter()
            .any(|item| projects_column(item, name));
        if already_projected {
            continue;
        }
        select
            .projection
            .push(SelectItem::UnnamedExpr(Expr::Identifier(Ident::new(name))));
        changed = true;
    }
    changed
}

/// Returns true if the SELECT item already yields the column `name`, either
/// as a bare/qualified identifier (e.g. `_rowid`, `t._rowid`) or through a
/// wildcard (`*`, `t.*`), so injecting it again would duplicate the column.
///
/// Expressions that merely reference the column (e.g. `_rowid + 1`, aliases)
/// intentionally don't count: they produce a different output column.
fn projects_column(item: &SelectItem, name: &str) -> bool {
    match item {
        SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => true,
        SelectItem::UnnamedExpr(Expr::Identifier(ident)) => ident_matches(ident, name),
        SelectItem::UnnamedExpr(Expr::CompoundIdentifier(idents)) => idents
            .last()
            .is_some_and(|ident| ident_matches(ident, name)),
        _ => false,
    }
}

fn ident_matches(ident: &Ident, name: &str) -> bool {
    if ident.quote_style.is_some() {
        ident.value == name
    } else {
        ident.value.eq_ignore_ascii_case(name)
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
        let exec_node = self
            .dataframe
            .execute_stream()
            .await
            .map_err(lance_core::Error::from)?;
        let schema = exec_node.schema();
        if SchemaAdapter::requires_logical_conversion(&schema) {
            let adapter = SchemaAdapter::new(schema);
            Ok(adapter.to_logical_stream(exec_node))
        } else {
            Ok(exec_node)
        }
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
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount, assert_string_matches};
    use crate::{BlobArrayBuilder, blob_field};
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::Dataset;
    use crate::dataset::write::WriteParams;
    use all_asserts::assert_true;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type, UInt64Type};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::Schema as ArrowSchema;
    use arrow_schema::{DataType, Field};
    use lance_arrow::json::ARROW_JSON_EXT_NAME;
    use lance_arrow::{ARROW_EXT_NAME_KEY, SchemaExt};
    use lance_core::datatypes::BlobHandling;
    use lance_datagen::{array, gen_batch};
    use lance_file::version::LanceFileVersion;
    use rstest::rstest;

    #[tokio::test]
    async fn test_sql_execute() {
        let ds = gen_batch()
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
    async fn test_sql_includes_unprojected_system_columns() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_implicit_system_columns",
                FragmentCount::from(1),
                FragmentRowCount::from(2),
            )
            .await
            .unwrap();

        let batches = ds
            .sql("SELECT x FROM dataset")
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let batch = &batches[0];
        assert_eq!(batch.schema().fields().len(), 3);
        let row_id_index = batch.schema().index_of("_rowid").unwrap();
        let row_ids = batch.column(row_id_index).as_primitive::<UInt64Type>();
        assert_eq!(row_ids.values(), &[0, 1]);
        let row_addr_index = batch.schema().index_of("_rowaddr").unwrap();
        let row_addrs = batch.column(row_addr_index).as_primitive::<UInt64Type>();
        assert_eq!(row_addrs.values(), &[0, 1]);

        // Filter/sort/limit chains preserve row identity, so injection still
        // applies to them.
        let batches = ds
            .sql("SELECT x FROM dataset WHERE x >= 0 ORDER BY x DESC LIMIT 2")
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let batch = &batches[0];
        assert_eq!(batch.schema().fields().len(), 3);
        assert_eq!(
            batch["_rowid"].as_primitive::<UInt64Type>().values(),
            &[1, 0]
        );
        assert_eq!(
            batch["_rowaddr"].as_primitive::<UInt64Type>().values(),
            &[1, 0]
        );
    }

    #[tokio::test]
    async fn test_sql_wildcard_includes_system_columns_once() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_wildcard_system_columns",
                FragmentCount::from(1),
                FragmentRowCount::from(2),
            )
            .await
            .unwrap();

        let batches = ds
            .sql("SELECT * FROM dataset")
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let batch = &batches[0];
        assert_eq!(
            batch.schema().field_names(),
            vec!["x", "_rowid", "_rowaddr"]
        );
        assert_eq!(
            batch["_rowid"].as_primitive::<UInt64Type>().values(),
            &[0, 1]
        );
        assert_eq!(
            batch["_rowaddr"].as_primitive::<UInt64Type>().values(),
            &[0, 1]
        );
    }

    #[rstest]
    #[case::lowercase("SELECT x, _rowid FROM dataset", vec!["x", "_rowid"])]
    #[case::unquoted_uppercase("SELECT x, _ROWID FROM dataset", vec!["x", "_rowid"])]
    #[case::quoted(r#"SELECT x, "_rowid" FROM dataset"#, vec!["x", "_rowid"])]
    #[case::table_qualified("SELECT x, dataset._rowid FROM dataset", vec!["x", "_rowid"])]
    #[case::expression_reference("SELECT _rowid + 1 AS y FROM dataset", vec!["y", "_rowid"])]
    #[case::system_column_only("SELECT _rowid FROM dataset", vec!["_rowid"])]
    #[tokio::test]
    async fn test_sql_system_column_spellings(
        #[case] sql: &str,
        #[case] expected_columns: Vec<&str>,
    ) {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_system_column_spellings",
                FragmentCount::from(1),
                FragmentRowCount::from(2),
            )
            .await
            .unwrap();

        let batches = ds
            .sql(sql)
            .with_row_id(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let batch = &batches[0];
        assert_eq!(batch.schema().field_names(), expected_columns);
        assert_eq!(
            batch["_rowid"].as_primitive::<UInt64Type>().values(),
            &[0, 1]
        );
    }

    /// A derived subquery can alias arbitrary expressions to the reserved
    /// system column names. Injected identifiers resolve by name in the outer
    /// scope, so injection must be skipped there; otherwise the query would
    /// return derived values as row metadata.
    #[tokio::test]
    async fn test_sql_system_columns_not_injected_when_shadowed() {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_system_columns_shadowed",
                FragmentCount::from(1),
                FragmentRowCount::from(2),
            )
            .await
            .unwrap();

        let batches = ds
            .sql(
                "SELECT x FROM (SELECT x, (_rowid + 1) AS _rowid, (_rowaddr + 1) AS _rowaddr \
                 FROM dataset) s",
            )
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        assert_eq!(batches[0].schema().field_names(), vec!["x"]);

        // A subquery that passes the real scan columns through unchanged is
        // safe: injection still applies and returns the actual metadata.
        let batches = ds
            .sql("SELECT x FROM (SELECT x, _rowid, _rowaddr FROM dataset) s")
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let batch = &batches[0];
        assert_eq!(
            batch.schema().field_names(),
            vec!["x", "_rowid", "_rowaddr"]
        );
        assert_eq!(
            batch["_rowid"].as_primitive::<UInt64Type>().values(),
            &[0, 1]
        );
        assert_eq!(
            batch["_rowaddr"].as_primitive::<UInt64Type>().values(),
            &[0, 1]
        );
    }

    /// System columns must never be injected into queries whose output rows
    /// are not one-to-one with dataset rows: under GROUP BY ALL or DISTINCT
    /// the injected columns would become extra grouping/dedup keys and change
    /// the relational results.
    #[rstest]
    #[case::group_by_all("SELECT x % 1 AS k, COUNT(*) AS n FROM dataset GROUP BY ALL ORDER BY k")]
    #[case::group_by_expr("SELECT x % 1 AS k, COUNT(*) AS n FROM dataset GROUP BY k ORDER BY k")]
    #[case::distinct("SELECT DISTINCT x % 1 AS k FROM dataset ORDER BY k")]
    #[case::distinct_in_subquery(
        "SELECT k FROM (SELECT DISTINCT x % 1 AS k FROM dataset) ORDER BY k"
    )]
    #[case::bare_aggregate("SELECT COUNT(*) AS n FROM dataset")]
    #[tokio::test]
    async fn test_sql_system_columns_skip_cardinality_changing_queries(#[case] sql: &str) {
        let ds = gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_dataset(
                "memory://test_sql_system_columns_cardinality",
                FragmentCount::from(1),
                FragmentRowCount::from(2),
            )
            .await
            .unwrap();

        let baseline = ds
            .sql(sql)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let with_system_columns = ds
            .sql(sql)
            .with_row_id(true)
            .with_row_addr(true)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        pretty_assertions::assert_eq!(with_system_columns, baseline);
    }

    #[tokio::test]
    async fn test_sql_blob_all_binary() {
        let schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
        let mut blobs = BlobArrayBuilder::new(2);
        blobs.push_bytes(b"foo").unwrap();
        blobs.push_bytes(b"bar").unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://test_sql_blob_all_binary",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let batches = dataset
            .sql("SELECT blob FROM dataset")
            .blob_handling(BlobHandling::AllBinary)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let blobs = batches[0].column(0).as_binary::<i64>();
        assert_eq!(blobs.value(0), b"foo");
        assert_eq!(blobs.value(1), b"bar");

        // Expressions over the blob column require the planner to see the
        // materialized LargeBinary type instead of the blob descriptor struct.
        let batches = dataset
            .sql("SELECT blob = X'666f6f' FROM dataset")
            .blob_handling(BlobHandling::AllBinary)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let is_foo = batches[0].column(0).as_boolean();
        assert!(is_foo.value(0));
        assert!(!is_foo.value(1));

        let batches = dataset
            .sql("SELECT blob, _rowid, _rowaddr FROM dataset")
            .with_row_id(true)
            .with_row_addr(true)
            .blob_handling(BlobHandling::AllBinary)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let batch = &batches[0];
        let blobs = batch.column(0).as_binary::<i64>();
        assert_eq!(blobs.value(0), b"foo");
        assert_eq!(blobs.value(1), b"bar");
        let row_ids = batch.column(1).as_primitive::<UInt64Type>();
        assert_eq!(row_ids.value(0), 0);
        assert_eq!(row_ids.value(1), 1);
        let row_addrs = batch.column(2).as_primitive::<UInt64Type>();
        assert_eq!(row_addrs.value(0), 0);
        assert_eq!(row_addrs.value(1), 1);
    }

    #[tokio::test]
    async fn test_sql_count() {
        let ds = gen_batch()
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
        let ds = gen_batch()
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
        let ds = gen_batch()
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

    #[tokio::test]
    async fn test_nested_json_access() {
        let json_rows = vec![
            Some(r#"{"user": {"profile": {"name": "Alice", "settings": {"theme": "dark"}}}}"#),
            Some(r#"{"user": {"profile": {"name": "Bob", "settings": {"theme": "light"}}}}"#),
        ];
        let json_array = StringArray::from(json_rows);
        let id_array = Int32Array::from(vec![1, 2]);

        let mut metadata = HashMap::new();
        metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, true).with_metadata(metadata),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(json_array)],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        let ds = Dataset::write(reader, "memory://test_nested_json_access", None)
            .await
            .unwrap();

        let results = ds
            .sql(
                "SELECT id FROM dataset WHERE \
                 json_get_string(json_get(json_get(data, 'user'), 'profile'), 'name') = 'Alice'",
            )
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let batch = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(batch.num_rows(), 1);
        pretty_assertions::assert_eq!(batch.num_columns(), 1);
        pretty_assertions::assert_eq!(batch.column(0).as_primitive::<Int32Type>().value(0), 1);

        let results = ds
            .sql(
                "SELECT id FROM dataset WHERE \
                 json_extract(data, '$.user.profile.settings.theme') = '\"dark\"'",
            )
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let batch = results.into_iter().next().unwrap();
        pretty_assertions::assert_eq!(batch.num_rows(), 1);
        pretty_assertions::assert_eq!(batch.num_columns(), 1);
        pretty_assertions::assert_eq!(batch.column(0).as_primitive::<Int32Type>().value(0), 1);
    }
}
