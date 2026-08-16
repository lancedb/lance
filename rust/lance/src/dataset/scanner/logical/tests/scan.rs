// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plain scans: filter, projection, limit.
//!
//! These run against both storage versions. Legacy (v1) storage is the one case where the scan
//! leaf lowers to something else entirely — the frozen legacy builder rather than
//! `FilteredReadExec` — so it needs its own coverage of the same shapes, not just the default.

use lance_file::version::LanceFileVersion;

use crate::dataset::scanner::AggregateExpr;
use rstest::rstest;

use super::harness::*;

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

/// The v1 counterpart of [`test_filtered_scan_plan`].
///
/// Also a single node, but the legacy one: the leaf routed to the statistics-pushdown scan, which
/// applies the predicate itself and emits the projection, so nothing survives above it. That the
/// plan is not `LanceScan + FilterExec` is the point — the leaf reports exact pushdown on v1 and
/// keeps the only page pruning legacy storage has.
#[tokio::test]
async fn test_filtered_scan_plan_v1() {
    let dataset = test_dataset_versioned(LanceFileVersion::Legacy).await;

    assert_logical_plan(
        &dataset,
        |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
        "LancePushdownScan: uri=..., projection=[s], predicate=i > Int32(10) AND i < Int32(20), \
         row_id=false, row_addr=false, ordered=true",
    )
    .await
    .unwrap();
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_paths_agree_on_filtered_scan(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?.filter("i > 10 and i < 20")
    })
    .await
    .unwrap();
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_paths_agree_on_full_scan(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    assert_paths_agree(&dataset, |scan| Ok(scan)).await.unwrap();
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_paths_agree_on_limit(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    assert_paths_agree(&dataset, |scan| scan.limit(Some(10), Some(5)))
        .await
        .unwrap();
}

/// The legacy branch that does *not* apply the predicate.
///
/// Setting a batch size disqualifies the statistics-pushdown scan, so the leaf falls through to
/// the plain fragment scan and has to apply the predicate itself. Without that, this query would
/// return every row.
#[tokio::test]
async fn test_paths_agree_on_a_v1_scan_that_cannot_push_down() {
    let dataset = test_dataset_versioned(LanceFileVersion::Legacy).await;
    assert_paths_agree(&dataset, |scan| {
        scan.batch_size(16);
        scan.project(&["s"])?.filter("i > 10 and i < 20")
    })
    .await
    .unwrap();
}

/// The legacy branch that reaches the predicate through a scalar index.
#[tokio::test]
async fn test_paths_agree_on_a_v1_scalar_indexed_scan() {
    use crate::index::DatasetIndexExt;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;

    let mut dataset = test_dataset_versioned(LanceFileVersion::Legacy).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?.filter("i > 10 and i < 20")
    })
    .await
    .unwrap();
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_paths_agree_on_row_id_projection(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["i"])?.with_row_id().filter("i % 3 = 0")
    })
    .await
    .unwrap();
}

/// A dataset with deletions, so `include_deleted_rows` has something to surface.
async fn deleted_rows_dataset(version: LanceFileVersion) -> crate::dataset::Dataset {
    let mut dataset = test_dataset_versioned(version).await;
    dataset.delete("i % 7 = 0").await.unwrap();
    dataset
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_paths_agree_on_include_deleted_rows(#[case] version: LanceFileVersion) {
    let dataset = deleted_rows_dataset(version).await;
    assert_paths_agree(&dataset, |scan| {
        scan.include_deleted_rows();
        Ok(scan.project(&["i"])?.with_row_id())
    })
    .await
    .unwrap();
}

/// Deleted rows carry a null row id, so a search — which returns row ids — cannot include them.
/// Both paths reject it; this asserts the logical one does so for the same reason.
#[tokio::test]
async fn test_include_deleted_rows_is_rejected_for_a_search() {
    let dataset = vector_dataset().await;
    let mut scan = dataset.scan();
    scan.with_row_id().include_deleted_rows();
    scan.nearest("vec", &query_vector(), 5).unwrap();

    for plan in [
        super::super::create_plan(&scan).await,
        scan.create_plan().await,
    ] {
        let err = plan.expect_err("a search cannot include deleted rows");
        assert!(err.to_string().contains("deleted rows"), "{err}");
    }
}

/// Every batch but the last carries exactly the requested number of rows.
#[tokio::test]
async fn test_strict_batch_size() {
    use futures::TryStreamExt;
    use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};

    let dataset = test_dataset().await;
    let plan = logical_plan_for(&dataset, |scan| {
        scan.batch_size(32).strict_batch_size(true);
        scan.project(&["i"])
    })
    .await
    .unwrap();

    let batches = execute_plan(plan, LanceExecutionOptions::default())
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();
    assert_eq!(sizes.iter().sum::<usize>(), 200);
    let (last, rest) = sizes.split_last().expect("plan produced no batches");
    assert!(rest.iter().all(|size| *size == 32), "{sizes:?}");
    assert!(*last <= 32, "{sizes:?}");
}

/// Aggregates are stock DataFusion nodes, so the whole of `COUNT(*)` and `SUM` comes from the
/// builder emitting an `Aggregate` and letting projection pushdown find its columns.
#[rstest]
#[case::count_star(AggregateExpr::builder().count_star().build())]
#[case::sum(AggregateExpr::builder().sum("i").build())]
#[tokio::test]
async fn test_paths_agree_on_aggregates(#[case] aggregate: AggregateExpr) {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, move |scan| {
        scan.aggregate(aggregate.clone())?.filter("i > 10")
    })
    .await
    .unwrap();
}

/// A grouped aggregate, compared as a set: neither path promises an order for hash aggregation.
#[tokio::test]
async fn test_paths_agree_on_a_grouped_aggregate() {
    use arrow::compute::{SortColumn, lexsort_to_indices, take};

    let dataset = test_dataset().await;
    fn config(scan: &mut crate::dataset::Scanner) -> crate::Result<&mut crate::dataset::Scanner> {
        scan.aggregate(AggregateExpr::builder().group_by("s").count_star().build())?
            .filter("i > 10")
    }
    let by_group = |batch: arrow_array::RecordBatch| {
        let indices = lexsort_to_indices(
            &[SortColumn {
                values: batch.column(0).clone(),
                options: None,
            }],
            None,
        )
        .unwrap();
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(column.as_ref(), &indices, None).unwrap())
            .collect::<Vec<_>>();
        arrow_array::RecordBatch::try_new(batch.schema(), columns).unwrap()
    };

    let expected = by_group(
        run(imperative_plan_for(&dataset, config).await.unwrap())
            .await
            .unwrap(),
    );
    let actual = by_group(
        run(logical_plan_for(&dataset, config).await.unwrap())
            .await
            .unwrap(),
    );
    assert_eq!(expected, actual);
}
