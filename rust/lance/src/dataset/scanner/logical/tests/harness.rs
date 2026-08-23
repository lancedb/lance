// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared harness: the result oracles and the datasets the tests run against.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_datagen::{BatchCount, ByteCount, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;

use crate::Result;
use crate::dataset::{Dataset, Scanner};
use crate::utils::test::assert_plan_node_equals;

/// A scanner-configuring closure. Generic rather than a `fn` pointer so a case can close over its
/// query vector or filter string.
pub(super) trait ScanConfig: Fn(&mut Scanner) -> Result<&mut Scanner> {}
impl<F: Fn(&mut Scanner) -> Result<&mut Scanner>> ScanConfig for F {}

/// Pin a closure's lifetime to the `ScanConfig` bound.
///
/// Rust infers a single fixed lifetime for a closure argument unless something forces the
/// higher-ranked form, and `impl ScanConfig` is not enough on its own. Wrapping the closure here is
/// what makes it one; a plain `fn` item needs no wrapper.
pub(super) fn config<F>(f: F) -> F
where
    F: for<'a> Fn(&'a mut Scanner) -> Result<&'a mut Scanner>,
{
    f
}

/// Build a scanner, apply `config`, and plan it through the logical path.
///
/// `target_parallelism(1)` pins `EnforceDistribution`'s output so plan strings do not depend on
/// the machine's CPU count, matching the convention in the scanner's own plan tests.
pub(super) async fn logical_plan_for(
    dataset: &Dataset,
    config: impl ScanConfig,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    config(&mut scan)?;
    super::super::create_plan(&scan).await
}

/// As [`logical_plan_for`], but through the imperative path this module replaces.
///
/// Only the equivalence check below uses it. It goes away with the imperative path.
pub(super) async fn imperative_plan_for(
    dataset: &Dataset,
    config: impl ScanConfig,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    config(&mut scan)?;
    scan.create_plan().await
}

pub(super) async fn assert_logical_plan(
    dataset: &Dataset,
    config: impl ScanConfig,
    expected: &str,
) -> Result<()> {
    let plan = logical_plan_for(dataset, config).await?;
    assert_plan_node_equals(plan, expected).await
}

pub(super) async fn run(plan: Arc<dyn ExecutionPlan>) -> Result<RecordBatch> {
    let schema = plan.schema();
    let batches = execute_plan(plan, LanceExecutionOptions::default())?
        .try_collect::<Vec<_>>()
        .await?;
    Ok(concat_batches(&schema, &batches)?)
}

// ---------------------------------------------------------------------------------------------
// Result oracles
// ---------------------------------------------------------------------------------------------
//
// Every fixture in this module derives its data from the row's `i` value, so the answer to a query
// can be stated in terms of `i` and computed in Rust. That is what these helpers compare against:
// a claim about the data, not another planner's output. Results are identified by `_rowid`, which
// [`Fixture`] maps back to `i`, so a case can say which rows it expects without having to project
// the column it is filtering on.

/// The whole dataset, read once, as the reference the assertions are stated against.
pub(super) struct Fixture {
    rows: RecordBatch,
}

impl Fixture {
    /// A plain unfiltered scan, which is the one shape whose answer is not in question.
    pub(super) async fn read(dataset: &Dataset) -> Result<Self> {
        let mut scan = dataset.scan();
        scan.with_row_id();
        Ok(Self {
            rows: scan.try_into_batch().await?,
        })
    }

    pub(super) fn row_ids(&self) -> &[u64] {
        use arrow_array::cast::AsArray;
        use arrow_array::types::UInt64Type;
        self.rows[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
    }

    pub(super) fn ids(&self) -> &[i32] {
        use arrow::datatypes::Int32Type;
        use arrow_array::cast::AsArray;
        self.rows["i"].as_primitive::<Int32Type>().values()
    }

    /// The `i` values of `row_ids`, in the order given.
    pub(super) fn ids_of(&self, row_ids: &[u64]) -> Vec<i32> {
        let index: std::collections::HashMap<u64, i32> = self
            .row_ids()
            .iter()
            .copied()
            .zip(self.ids().iter().copied())
            .collect();
        row_ids
            .iter()
            .map(|row_id| index[row_id])
            .collect::<Vec<_>>()
    }

    /// A string column's values, in storage order.
    pub(super) fn strings(&self, column: &str) -> Vec<String> {
        use arrow_array::cast::AsArray;
        self.rows[column]
            .as_string::<i32>()
            .iter()
            .map(|value| value.unwrap_or_default().to_string())
            .collect()
    }

    /// The `i` values of every row satisfying `keep`, in storage order.
    pub(super) fn ids_where(&self, keep: impl Fn(i32) -> bool) -> Vec<i32> {
        self.ids().iter().copied().filter(|i| keep(*i)).collect()
    }

    /// The value of `column` for the row identified by `row_id`.
    fn cell(&self, column: &str, row_id: u64) -> arrow_array::ArrayRef {
        let position = self
            .row_ids()
            .iter()
            .position(|candidate| *candidate == row_id)
            .expect("result row is not in the dataset");
        self.rows[column].slice(position, 1)
    }
}

/// The `_rowid` column of a result, which every oracle here uses to identify rows.
pub(super) fn row_ids_of(batch: &RecordBatch) -> Vec<u64> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt64Type;
    batch[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
}

/// Run `config`'s scan with `_rowid` appended, so its rows can be named.
///
/// Every result oracle below funnels through here, so this is also where the path being replaced is
/// held to the same answer: the imperative path plans the same query, and the two must return the
/// same rows in the same order. The oracles state what the answer *is*; this states that nothing
/// changed on the way to it. Both halves go away together — the oracles stay, the comparison
/// leaves with the imperative path.
///
/// Row order is compared because it is observable: a path that returns the right rows in a
/// different order has still changed what a caller sees.
pub(super) async fn scan_rows(dataset: &Dataset, config: impl ScanConfig) -> Result<RecordBatch> {
    let config = config_with_row_id(config);
    let actual = run(logical_plan_for(dataset, &config).await?).await?;
    let expected = run(imperative_plan_for(dataset, &config).await?).await?;

    assert_eq!(
        expected.schema(),
        actual.schema(),
        "logical path produced a different output schema"
    );
    assert_eq!(expected, actual, "logical path produced different rows");
    Ok(actual)
}

fn config_with_row_id(config: impl ScanConfig) -> impl ScanConfig {
    move |scan: &mut Scanner| {
        config(scan)?;
        Ok(scan.with_row_id())
    }
}

/// Assert `config`'s scan returned exactly the rows whose `i` satisfies `keep`, in storage order,
/// and that every column it returned carries that row's value.
pub(super) async fn assert_scan_keeps(
    dataset: &Dataset,
    config: impl ScanConfig,
    keep: impl Fn(i32) -> bool,
) -> Result<()> {
    let fixture = Fixture::read(dataset).await?;
    assert_scan_returns(dataset, config, &fixture, fixture.ids_where(keep)).await
}

/// Assert `config`'s scan returned exactly the rows named by `expected`, in that order.
pub(super) async fn assert_scan_returns(
    dataset: &Dataset,
    config: impl ScanConfig,
    fixture: &Fixture,
    expected: Vec<i32>,
) -> Result<()> {
    let actual = scan_rows(dataset, config).await?;
    let row_ids = row_ids_of(&actual);
    assert_eq!(fixture.ids_of(&row_ids), expected, "wrong rows");
    assert_columns_match(&actual, fixture, &row_ids);
    Ok(())
}

/// Check every projected column against the dataset's own copy of that row.
///
/// Row identity alone would not catch a take that fetched the right rows and then misaligned their
/// values, which is the failure mode late materialization is most exposed to.
pub(super) fn assert_columns_match(actual: &RecordBatch, fixture: &Fixture, row_ids: &[u64]) {
    for field in actual.schema().fields() {
        // Only columns the dataset stores; scores and distances are the search's own output.
        if !fixture.rows.schema().fields().iter().any(|f| f == field) {
            continue;
        }
        for (position, row_id) in row_ids.iter().enumerate() {
            assert_eq!(
                &actual[field.name()].slice(position, 1),
                &fixture.cell(field.name(), *row_id),
                "column {} disagrees with the dataset at row {position}",
                field.name()
            );
        }
    }
}

/// Tag a reader's schema with dataset-level metadata.
///
/// The imperative suite's own fixtures do this deliberately — `scanner.rs:6489` says "so it tests
/// all paths that re-construct the schema along the way". The oracle needs it for the same reason:
/// a lowering step that rebuilds an output schema and drops its metadata is invisible against a
/// fixture that has none.
pub(super) fn tagged(
    reader: impl arrow_array::RecordBatchReader,
) -> impl arrow_array::RecordBatchReader {
    use arrow::datatypes::Schema as ArrowSchema;
    use arrow_array::{RecordBatch, RecordBatchIterator};

    let schema = Arc::new(ArrowSchema::new_with_metadata(
        reader.schema().fields().clone(),
        [("dataset".to_string(), "logical".to_string())].into(),
    ));
    let batches = reader
        .map(|batch| {
            let batch = batch?;
            RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    RecordBatchIterator::new(batches.into_iter().map(Ok), schema)
}

pub(super) async fn test_dataset() -> Dataset {
    test_dataset_versioned(LanceFileVersion::Stable).await
}

/// The `i`/`s` fixture, written at a chosen storage version.
///
/// Legacy (v1) storage is the one thing that changes the shape of the scan leaf, so the plain-scan
/// equivalence cases run against both versions rather than only the default.
pub(super) async fn test_dataset_versioned(version: LanceFileVersion) -> Dataset {
    use crate::dataset::WriteParams;
    use arrow::datatypes::Int32Type;

    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("s", array::rand_utf8(ByteCount::from(8), false))
        .into_reader_rows(RowCount::from(100), BatchCount::from(2));
    Dataset::write(
        tagged(data),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 100,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}
