// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plain scans: filter, projection, limit.

use lance_datafusion::exec::execute_plan;
use lance_datagen::BatchCount;

use crate::Result;
use crate::dataset::Scanner;

/// A scanner-configuring closure. Generic rather than a `fn` pointer so a case can close over
/// its query vector or filter string, and taken by reference internally because every oracle
/// applies it twice — once per path.
trait ScanConfig: Fn(&mut Scanner) -> Result<&mut Scanner> {}
impl<F: Fn(&mut Scanner) -> Result<&mut Scanner>> ScanConfig for F {}

/// Sort a result batch by `_rowid` so two plans can be compared as sets.
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
