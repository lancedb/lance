// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared harness: the equivalence oracle and the datasets the tests run against.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;

use crate::Result;
use crate::dataset::{Dataset, Scanner};
use crate::utils::test::assert_plan_node_equals;

/// A scanner-configuring closure. Generic rather than a `fn` pointer so a case can close over
/// its query vector or filter string, and taken by reference internally because every oracle
/// applies it twice — once per path.
pub(super) trait ScanConfig: Fn(&mut Scanner) -> Result<&mut Scanner> {}
impl<F: Fn(&mut Scanner) -> Result<&mut Scanner>> ScanConfig for F {}

/// Sort a result batch by `_rowid` so two plans can be compared as sets.
pub(super) fn sorted_by_row_id(batch: &RecordBatch) -> Result<RecordBatch> {
    use arrow::compute::{SortColumn, lexsort_to_indices, take};

    let row_id = batch
        .column_by_name(lance_core::ROW_ID)
        .expect("results must carry _rowid for set comparison");
    let indices = lexsort_to_indices(
        &[SortColumn {
            values: row_id.clone(),
            options: None,
        }],
        None,
    )?;
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
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

/// Execute both planning paths and assert they produce the same rows, in the same order.
///
/// This is the default, and deliberately so: row order is observable, so a path that returns the
/// right rows in a different order has still changed what a caller sees. Reach for
/// [`assert_paths_agree_unordered`] only where the order is genuinely not part of the contract,
/// and say why at the call site.
pub(super) async fn assert_paths_agree(dataset: &Dataset, config: impl ScanConfig) -> Result<()> {
    let expected = run(imperative_plan_for(dataset, &config).await?).await?;
    let actual = run(logical_plan_for(dataset, &config).await?).await?;

    assert_eq!(
        expected.schema(),
        actual.schema(),
        "logical path produced a different output schema"
    );
    assert_eq!(
        expected.num_rows(),
        actual.num_rows(),
        "logical path produced a different row count"
    );
    assert_eq!(expected, actual, "logical path produced different rows");
    Ok(())
}

/// As [`assert_paths_agree`], but comparing row *sets* rather than row sequences.
///
/// Only two shapes need this, and both are cases where nothing in either path establishes an
/// order: an all-ties FTS score sort, and a union of coverage branches. Each call site states
/// which. A new use is a claim that order does not matter — check that before making it.
pub(super) async fn assert_paths_agree_unordered(
    dataset: &Dataset,
    config: impl ScanConfig,
) -> Result<()> {
    let expected = sorted_by_row_id(&run(imperative_plan_for(dataset, &config).await?).await?)?;
    let actual = sorted_by_row_id(&run(logical_plan_for(dataset, &config).await?).await?)?;

    assert_eq!(
        expected.schema(),
        actual.schema(),
        "logical path produced a different output schema"
    );
    assert_eq!(
        expected.num_rows(),
        actual.num_rows(),
        "logical path produced a different row count"
    );
    assert_eq!(expected, actual, "logical path produced different rows");
    Ok(())
}

pub(super) const DIM: u32 = 32;

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

/// Written through a real (in-memory) object store rather than `into_ram_dataset`, so index
/// creation and `io_stats_incremental` both work.
pub(super) async fn vector_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use arrow::datatypes::{Float32Type, Int32Type};

    // 1024 rows so PQ has enough to train on, split into two fragments so plans exercise the
    // multi-fragment path.
    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("s", array::rand_utf8(ByteCount::from(8), false))
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(DIM)))
        .into_reader_rows(RowCount::from(256), BatchCount::from(4));
    Dataset::write(
        tagged(data),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 512,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

pub(super) fn query_vector() -> arrow_array::Float32Array {
    arrow_array::Float32Array::from((0..DIM).map(|v| v as f32).collect::<Vec<_>>())
}

/// A vector dataset with an IVF_PQ index covering every fragment.
pub(super) async fn indexed_vector_dataset() -> Dataset {
    indexed_vector_dataset_with_metric(lance_linalg::distance::DistanceType::L2).await
}

/// As [`indexed_vector_dataset`], but with the index built for a chosen metric.
pub(super) async fn indexed_vector_dataset_with_metric(
    metric: lance_linalg::distance::DistanceType,
) -> Dataset {
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use lance_index::IndexType;

    let mut dataset = vector_dataset().await;
    let params = VectorIndexParams::ivf_pq(2, 8, 2, metric, 2);
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset
}
