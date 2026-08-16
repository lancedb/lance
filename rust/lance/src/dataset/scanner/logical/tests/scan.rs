// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plain scans: filter, projection, limit.
//!
//! These run against both storage versions. Legacy (v1) storage is the one case where the scan
//! leaf lowers to something else entirely — the frozen legacy builder rather than
//! `FilteredReadExec` — so it needs its own coverage of the same shapes, not just the default.

use lance_file::version::LanceFileVersion;
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
