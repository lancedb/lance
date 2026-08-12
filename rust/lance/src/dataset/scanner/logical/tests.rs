// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The equivalence oracle for the logical-plan spike.
//!
//! Two kinds of check, deliberately:
//!
//! * [`assert_logical_plan`] pins the physical plan the logical path emits, so a change in
//!   DataFusion's rule behavior shows up as a test diff rather than a silent replan.
//! * [`assert_paths_agree`] executes *both* paths and compares the rows. This is the check that
//!   actually matters — the two paths are expected to emit different plan strings (the logical
//!   path gets DataFusion's projection pushdown for free, which the imperative builder does not
//!   do), so plan-string equality would be the wrong bar.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array, gen_batch};

use crate::Result;
use crate::dataset::{Dataset, Scanner};
use crate::utils::test::assert_plan_node_equals;

type ScanConfig = fn(&mut Scanner) -> Result<&mut Scanner>;

/// Build a scanner, apply `config`, and plan it through the logical path.
///
/// `target_parallelism(1)` pins `EnforceDistribution`'s output so plan strings do not depend on
/// the machine's CPU count, matching the convention in the scanner's own plan tests.
async fn logical_plan_for(dataset: &Dataset, config: ScanConfig) -> Result<Arc<dyn ExecutionPlan>> {
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    config(&mut scan)?;
    super::create_plan(&scan).await
}

async fn imperative_plan_for(
    dataset: &Dataset,
    config: ScanConfig,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    config(&mut scan)?;
    scan.create_plan().await
}

async fn assert_logical_plan(dataset: &Dataset, config: ScanConfig, expected: &str) -> Result<()> {
    let plan = logical_plan_for(dataset, config).await?;
    assert_plan_node_equals(plan, expected).await
}

async fn run(plan: Arc<dyn ExecutionPlan>) -> Result<RecordBatch> {
    let schema = plan.schema();
    let batches = execute_plan(plan, LanceExecutionOptions::default())?
        .try_collect::<Vec<_>>()
        .await?;
    Ok(concat_batches(&schema, &batches)?)
}

/// Execute both planning paths and assert they produce the same rows.
async fn assert_paths_agree(dataset: &Dataset, config: ScanConfig) -> Result<()> {
    let expected = run(imperative_plan_for(dataset, config).await?).await?;
    let actual = run(logical_plan_for(dataset, config).await?).await?;

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

const DIM: u32 = 32;

async fn test_dataset() -> Dataset {
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow::datatypes::Int32Type;

    gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("s", array::rand_utf8(ByteCount::from(8), false))
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(100))
        .await
        .unwrap()
}

/// Written through a real (in-memory) object store rather than `into_ram_dataset`, so index
/// creation and `io_stats_incremental` both work.
async fn vector_dataset() -> Dataset {
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
        data,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 512,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

fn query_vector() -> arrow_array::Float32Array {
    arrow_array::Float32Array::from((0..DIM).map(|v| v as f32).collect::<Vec<_>>())
}

/// A vector dataset with an IVF_PQ index covering every fragment.
async fn indexed_vector_dataset() -> Dataset {
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use lance_index::IndexType;
    use lance_linalg::distance::DistanceType;

    let mut dataset = vector_dataset().await;
    let params = VectorIndexParams::ivf_pq(2, 8, 2, DistanceType::L2, 2);
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset
}

#[tokio::test]
async fn test_filtered_scan_plan() {
    let dataset = test_dataset().await;

    // The whole plan is a single LanceRead: DataFusion pushed both the projection and the
    // predicate into the scan leaf, and the leaf claims exact filter pushdown, so no FilterExec
    // and no ProjectionExec survive.
    assert_logical_plan(
        &dataset,
        |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
        "LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, range_after=None, \
         row_id=false, row_addr=false, full_filter=i > Int32(10) AND i < Int32(20), \
         refine_filter=i > Int32(10) AND i < Int32(20)",
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_filtered_scan() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?.filter("i > 10 and i < 20")
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_full_scan() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| Ok(scan)).await.unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_limit() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.limit(Some(10), Some(5)))
        .await
        .unwrap();
}

#[tokio::test]
async fn test_flat_knn_plan() {
    let dataset = vector_dataset().await;

    // The projection below the search is narrowed to `[vec, _rowid]` by DataFusion's
    // `optimize_projections` reaching through the extension node via `necessary_children_exprs`;
    // `i` and `s` are fetched afterwards by the take.
    assert_logical_plan(
        &dataset,
        |scan| scan.nearest("vec", &query_vector(), 5),
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@1 as _distance]
  LanceRead: uri=..., projection=[i, s, vec], source=stream(_rowid)
    ProjectionExec: expr=[_rowid@1 as _rowid, _distance@2 as _distance]
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=[_distance@2 ASC NULLS LAST, _rowid@1 ASC NULLS LAST], preserve_partitioning=[false]
          KNNVectorDistance: metric=l2
            LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
            row_id=true, row_addr=false, full_filter=--, refine_filter=--",
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_flat_knn() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_flat_knn_with_projection() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["i"])?.nearest("vec", &query_vector(), 7)
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_ann_plan_uses_the_index() {
    let dataset = indexed_vector_dataset().await;

    // Same logical node as the flat case; the only difference is what `ResolveVectorAccessPath`
    // found in the planning context.
    assert_logical_plan(
        &dataset,
        |scan| scan.nearest("vec", &query_vector(), 5),
        "ProjectionExec: ...
  LanceRead: ...
    ProjectionExec: expr=[_rowid@1 as _rowid, _distance@0 as _distance]
      SortExec: TopK(fetch=5), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST], ...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_ann() {
    let dataset = indexed_vector_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

/// `use_index(false)` is a semantic downgrade to exact search, so the rule must not pick the
/// index even though one exists.
#[tokio::test]
async fn test_exact_accuracy_forces_flat() {
    let dataset = indexed_vector_dataset().await;
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    scan.nearest("vec", &query_vector(), 5).unwrap();
    scan.use_index(false);

    let plan = super::create_plan(&scan).await.unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(text.contains("KNNVectorDistance"), "{text}");
    assert!(!text.contains("ANNSubIndex"), "{text}");
}

/// Context collection is the only stage that reads storage, and what it reads is cached. So
/// planning the same query twice must issue no reads the second time — the same invariant
/// `test_scan_planning_io` pins for the imperative path.
#[tokio::test]
async fn test_planning_is_io_free_once_warm() {
    use lance_io::assert_io_eq;

    let dataset = indexed_vector_dataset().await;
    logical_plan_for(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
    dataset.object_store.as_ref().io_stats_incremental();

    logical_plan_for(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
}

/// Doc case (d): the predicate sits below the search, so it restricts the candidate set. It
/// reaches the index as a `PreFilterSource` — visible here as the second child of `ANNSubIndex`.
#[tokio::test]
async fn test_ann_prefilter_feeds_the_index() {
    let dataset = indexed_vector_dataset().await;

    assert_logical_plan(
        &dataset,
        |scan| {
            scan.prefilter(true)
                .filter("i > 10")?
                .nearest("vec", &query_vector(), 5)
        },
        "ProjectionExec: ...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
          row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)",
    )
    .await
    .unwrap();
}

/// Doc case (e): the same predicate above the search trims the results instead. The index sees
/// no prefilter, and a `FilterExec` survives above the take.
#[tokio::test]
async fn test_ann_postfilter_stays_above_the_search() {
    let dataset = indexed_vector_dataset().await;

    let plan = logical_plan_for(&dataset, |scan| {
        scan.prefilter(false)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );

    let filter_line = text
        .lines()
        .position(|line| line.contains("FilterExec: i@"))
        .unwrap_or_else(|| panic!("no postfilter in plan:\n{text}"));
    let search_line = text
        .lines()
        .position(|line| line.contains("ANNSubIndex"))
        .unwrap_or_else(|| panic!("no ANN search in plan:\n{text}"));

    // The load-bearing invariant: `PushDownFilter` must treat the search node as opaque. If it
    // ever learns to push through, a postfilter silently becomes a prefilter and the query
    // quietly changes meaning.
    assert!(
        filter_line < search_line,
        "postfilter sank below the search:\n{text}"
    );
}

#[tokio::test]
async fn test_paths_agree_on_ann_prefilter() {
    let dataset = indexed_vector_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.prefilter(true)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_flat_knn_prefilter() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.prefilter(true)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
}

/// Appendix A: an index that covers only some fragments.
async fn partially_indexed_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use arrow::datatypes::{Float32Type, Int32Type};

    let mut dataset = indexed_vector_dataset().await;
    let more = gen_batch()
        .col("i", array::step_custom::<Int32Type>(10_000, 1))
        .col("s", array::rand_utf8(ByteCount::from(8), false))
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(DIM)))
        .into_reader_rows(RowCount::from(64), BatchCount::from(1));
    dataset
        .append(
            more,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset
}

#[tokio::test]
async fn test_combined_knn_ann_splits_into_two_branches() {
    let dataset = partially_indexed_dataset().await;

    let plan = logical_plan_for(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );

    assert!(text.contains("UnionExec"), "no union in plan:\n{text}");
    assert!(text.contains("ANNSubIndex"), "no indexed branch:\n{text}");
    // Two brute-force nodes: the unindexed branch, and the exact re-rank above the union.
    assert_eq!(
        text.matches("KNNVectorDistance").count(),
        2,
        "expected a flat branch and a re-rank:\n{text}"
    );
}

#[tokio::test]
async fn test_paths_agree_on_combined_knn_ann() {
    let dataset = partially_indexed_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

/// The stress case from Appendix A: one logical `Filter` has to reach both branches — as a
/// prefilter source on the indexed side and as an ordinary predicate on the flat side.
#[tokio::test]
async fn test_paths_agree_on_combined_knn_ann_prefilter() {
    let dataset = partially_indexed_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.prefilter(true)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_unsupported_shape_is_rejected() {
    let dataset = test_dataset().await;
    let mut scan = dataset.scan();
    scan.order_by(Some(vec![
        crate::dataset::scanner::ColumnOrdering::asc_nulls_first("i".into()),
    ]))
    .unwrap();

    let err = super::create_plan(&scan).await.unwrap_err();
    assert!(
        matches!(err, crate::Error::NotSupported { .. }),
        "expected NotSupported, got {err:?}"
    );
    assert!(err.to_string().contains("ordering"), "{err}");
}
