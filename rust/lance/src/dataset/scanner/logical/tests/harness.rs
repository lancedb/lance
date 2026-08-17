// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared harness: the result oracles and the datasets the tests run against.

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
pub(super) async fn scan_rows(dataset: &Dataset, config: impl ScanConfig) -> Result<RecordBatch> {
    run(logical_plan_for(dataset, config_with_row_id(config)).await?).await
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

/// A query inside the fixture's value range.
///
/// `rand_vec` draws each coordinate from [0, 1), so a query built from raw indices would sit far
/// outside the cloud, at nearly the same distance from every row. "The five nearest" is then a
/// coin flip no approximate index can be expected to win, and the recall assertions below would
/// measure the query rather than the plan.
pub(super) fn query_vector() -> arrow_array::Float32Array {
    arrow_array::Float32Array::from((0..DIM).map(|v| v as f32 / DIM as f32).collect::<Vec<_>>())
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
    // Two partitions, so the plan exercises the multi-partition path, and no quantizer. The
    // fixture's coordinates are uniform random, so in 32 dimensions its rows sit at nearly equal
    // distances from any query: a quantizer's error then exceeds the gaps it has to preserve, and
    // which rows come back varies with how k-means happened to land. That noise measures the
    // quantizer, not the plan, which is what these assertions are about.
    let params = VectorIndexParams::ivf_flat(2, metric);
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset
}

// ---------------------------------------------------------------------------------------------
// Search oracles
// ---------------------------------------------------------------------------------------------

/// The `i` values of the `k` vectors closest to `query`, computed by brute force over the whole
/// dataset. Ties broken by `i`, so the answer is a sequence rather than a set.
pub(super) async fn exact_neighbors(
    dataset: &Dataset,
    query: &arrow_array::Float32Array,
    metric: lance_linalg::distance::DistanceType,
    k: usize,
) -> Result<Vec<i32>> {
    use arrow_array::cast::AsArray;

    let fixture = Fixture::read(dataset).await?;
    let vectors = fixture.rows["vec"].as_fixed_size_list();
    let distances = metric.arrow_batch_func()(query, vectors)?;

    let mut ranked = fixture
        .ids()
        .iter()
        .copied()
        .zip(distances.values().iter().copied())
        .filter(|(_, distance)| !distance.is_nan())
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(k);
    Ok(ranked.into_iter().map(|(id, _)| id).collect())
}

/// Assert a vector search found at least `min_recall` of the true nearest neighbours, and returned
/// them in nondecreasing distance order.
///
/// Recall rather than equality because an IVF_PQ index is approximate by construction: it quantizes
/// the vectors and probes a subset of partitions, so an exact match would be a coincidence of the
/// fixture's size rather than a contract. Pass `1.0` where the search is brute force and the exact
/// answer *is* the contract.
pub(super) async fn assert_search_recall(
    dataset: &Dataset,
    config: impl ScanConfig,
    query: &arrow_array::Float32Array,
    metric: lance_linalg::distance::DistanceType,
    k: usize,
    min_recall: f64,
) -> Result<()> {
    let fixture = Fixture::read(dataset).await?;
    let expected = exact_neighbors(dataset, query, metric, k).await?;

    let actual = scan_rows(dataset, probe_every_partition(config)).await?;
    let row_ids = row_ids_of(&actual);
    assert_eq!(row_ids.len(), k, "search returned the wrong number of rows");
    assert_distances_ascending(&actual);
    assert_columns_match(&actual, &fixture, &row_ids);

    let found = fixture.ids_of(&row_ids);
    let hits = found.iter().filter(|id| expected.contains(id)).count();
    let recall = hits as f64 / expected.len() as f64;
    assert!(
        recall >= min_recall,
        "recall {recall} below {min_recall}: expected {expected:?}, got {found:?}"
    );
    Ok(())
}

/// Probe both of the fixture index's partitions.
///
/// The default probes one of the two, so recall then depends mostly on which partition k-means put
/// the neighbours in — it varies run to run and says nothing about the plan. Probing both leaves
/// quantization error, which is the approximation these assertions are about. Plan-shape tests do
/// not go through here, so they still pin the default `minimum_nprobes=1`.
pub(super) fn probe_every_partition(config: impl ScanConfig) -> impl ScanConfig {
    move |scan: &mut Scanner| {
        config(scan)?;
        Ok(scan.minimum_nprobes(2))
    }
}

/// Assert `_distance` never decreases down the result, which is the ordering a search promises.
pub(super) fn assert_distances_ascending(batch: &RecordBatch) {
    use arrow::datatypes::Float32Type;
    use arrow_array::cast::AsArray;

    let distances = batch[lance_index::vector::DIST_COL]
        .as_primitive::<Float32Type>()
        .values();
    assert!(
        distances.windows(2).all(|pair| pair[0] <= pair[1]),
        "distances are not ascending: {distances:?}"
    );
}

/// Assert `_score` never increases down the result: relevance order, most relevant first.
pub(super) fn assert_scores_descending(batch: &RecordBatch) {
    use arrow::datatypes::Float32Type;
    use arrow_array::cast::AsArray;

    let scores = batch[lance_index::scalar::inverted::SCORE_COL]
        .as_primitive::<Float32Type>()
        .values();
    assert!(
        scores.windows(2).all(|pair| pair[0] >= pair[1]),
        "scores are not descending: {scores:?}"
    );
}

/// Assert a full-text search returned exactly the documents `matches` names, scored in descending
/// order.
///
/// The row *set* is the part BM25 does not get a say in — a document either contains the terms or
/// it does not — so it can be stated from the fixture's text. Within that set the ranking is the
/// scorer's business, and only its direction is asserted here.
pub(super) async fn assert_fts_matches(
    dataset: &Dataset,
    config: impl ScanConfig,
    matches: impl Fn(i32) -> bool,
) -> Result<()> {
    let fixture = Fixture::read(dataset).await?;
    let actual = scan_rows(dataset, config).await?;
    let row_ids = row_ids_of(&actual);

    assert_scores_descending(&actual);
    assert_columns_match(&actual, &fixture, &row_ids);

    let mut found = fixture.ids_of(&row_ids);
    found.sort_unstable();
    found.dedup();
    assert_eq!(found, fixture.ids_where(matches), "wrong documents matched");
    Ok(())
}
