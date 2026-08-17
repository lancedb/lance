// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plain scans: filter, projection, limit.
//!
//! These run against both storage versions. Legacy (v1) storage is the one case where the scan
//! leaf lowers to something else entirely — the frozen legacy builder rather than
//! `FilteredReadExec` — so it needs its own coverage of the same shapes, not just the default.

use lance_file::version::LanceFileVersion;

use crate::dataset::scanner::{AggregateExpr, MaterializationStyle};
use rstest::rstest;

use super::harness::*;

#[tokio::test]
async fn test_filtered_scan_plan() {
    let dataset = test_dataset().await;

    // DataFusion pushed both the projection and the predicate into the scan leaf, and the leaf
    // claims exact filter pushdown, so no FilterExec survives above it. What the leaf lowers to is
    // two reads rather than one: `s` is a string, so its width is unknown and the heuristic reads
    // it late — only for the rows the filter kept.
    assert_logical_plan(
        &dataset,
        |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
        "ProjectionExec: expr=[s@2 as s]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    LanceRead: uri=..., projection=[i], num_fragments=2, range_before=None, range_after=None, \
     row_id=true, row_addr=false, full_filter=i > Int32(10) AND i < Int32(20), \
     refine_filter=i > Int32(10) AND i < Int32(20)",
    )
    .await
    .unwrap();
}

/// Reading everything in one pass is what an explicitly early materialization asks for.
#[tokio::test]
async fn test_filtered_scan_plan_all_early() {
    let dataset = test_dataset().await;

    assert_logical_plan(
        &dataset,
        config(|scan| {
            scan.project(&["s"])?
                .filter("i > 10 and i < 20")?
                .materialization_style(MaterializationStyle::AllEarly);
            Ok(scan)
        }),
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
async fn test_filtered_scan_returns_the_matching_rows(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    assert_scan_keeps(
        &dataset,
        |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
        |i| i > 10 && i < 20,
    )
    .await
    .unwrap();
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_full_scan_returns_every_row(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    assert_scan_keeps(&dataset, |scan| Ok(scan), |_| true)
        .await
        .unwrap();
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_limit_skips_then_truncates(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    let fixture = Fixture::read(&dataset).await.unwrap();
    // Offset 5, limit 10: the rows are in storage order, so this is `i` 5 through 14.
    assert_scan_returns(
        &dataset,
        |scan| scan.limit(Some(10), Some(5)),
        &fixture,
        (5..15).collect(),
    )
    .await
    .unwrap();
}

/// The legacy branch that does *not* apply the predicate.
///
/// Setting a batch size disqualifies the statistics-pushdown scan, so the leaf falls through to
/// the plain fragment scan and has to apply the predicate itself. Without that, this query would
/// return every row.
#[tokio::test]
async fn test_a_v1_scan_that_cannot_push_down_still_filters() {
    let dataset = test_dataset_versioned(LanceFileVersion::Legacy).await;
    assert_scan_keeps(
        &dataset,
        |scan| {
            scan.batch_size(16);
            scan.project(&["s"])?.filter("i > 10 and i < 20")
        },
        |i| i > 10 && i < 20,
    )
    .await
    .unwrap();
}

/// The legacy branch that reaches the predicate through a scalar index.
#[tokio::test]
async fn test_a_v1_scalar_indexed_scan_returns_the_matching_rows() {
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

    assert_scan_keeps(
        &dataset,
        |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
        |i| i > 10 && i < 20,
    )
    .await
    .unwrap();
}

#[rstest]
#[case::v1(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_row_id_projection(#[case] version: LanceFileVersion) {
    let dataset = test_dataset_versioned(version).await;
    assert_scan_keeps(
        &dataset,
        |scan| scan.project(&["i"])?.with_row_id().filter("i % 3 = 0"),
        |i| i % 3 == 0,
    )
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
async fn test_include_deleted_rows_surfaces_them_with_a_null_row_id(
    #[case] version: LanceFileVersion,
) {
    use arrow::datatypes::Int32Type;
    use arrow_array::cast::AsArray;

    let dataset = deleted_rows_dataset(version).await;
    let batch = scan_rows(&dataset, |scan| {
        scan.include_deleted_rows();
        Ok(scan.project(&["i"])?.with_row_id())
    })
    .await
    .unwrap();

    // Every row is back, deleted ones included, and a null row id is what marks them: a deleted
    // row has no stable identity to hand out.
    let ids = batch["i"].as_primitive::<Int32Type>().values();
    assert_eq!(ids, &(0..200).collect::<Vec<i32>>());
    let deleted = batch[lance_core::ROW_ID]
        .nulls()
        .expect("deleted rows must have a null row id");
    for (id, live) in ids.iter().zip(deleted.iter()) {
        assert_eq!(live, id % 7 != 0, "row {id} has the wrong liveness");
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
// `i` is 0..199, so `i > 10` keeps 189 rows summing to 11 + 12 + ... + 199.
#[case::count_star(AggregateExpr::builder().count_star().build(), 189)]
#[case::sum(AggregateExpr::builder().sum("i").build(), (11..200i64).sum())]
#[tokio::test]
async fn test_aggregates(#[case] aggregate: AggregateExpr, #[case] expected: i64) {
    use arrow::datatypes::Int64Type;
    use arrow_array::cast::AsArray;

    let dataset = test_dataset().await;
    let plan = logical_plan_for(&dataset, move |scan| {
        scan.aggregate(aggregate.clone())?.filter("i > 10")
    })
    .await
    .unwrap();
    let batch = run(plan).await.unwrap();

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(
        batch.column(0).as_primitive::<Int64Type>().value(0),
        expected
    );
}

/// A grouped aggregate. Hash aggregation promises no order, so this asserts the groups as a set.
#[tokio::test]
async fn test_a_grouped_aggregate() {
    use arrow::datatypes::Int64Type;
    use arrow_array::cast::AsArray;
    use std::collections::HashMap;

    let dataset = test_dataset().await;
    let plan = logical_plan_for(&dataset, |scan| {
        scan.aggregate(AggregateExpr::builder().group_by("s").count_star().build())?
            .filter("i > 10")
    })
    .await
    .unwrap();
    let batch = run(plan).await.unwrap();

    let mut counted: HashMap<String, i64> = HashMap::new();
    let groups = batch.column(0).as_string::<i32>();
    let counts = batch.column(1).as_primitive::<Int64Type>();
    for row in 0..batch.num_rows() {
        counted.insert(groups.value(row).to_string(), counts.value(row));
    }

    // The fixture's `s` values are random, so how many groups there are is a property of the data
    // rather than of the query: count them the same way the aggregate should have.
    let fixture = Fixture::read(&dataset).await.unwrap();
    let mut expected: HashMap<String, i64> = HashMap::new();
    for (id, text) in fixture.ids().iter().zip(fixture.strings("s")) {
        if *id > 10 {
            *expected.entry(text).or_default() += 1;
        }
    }
    assert_eq!(counted, expected);
}

/// Which columns the scan reads alongside the filter and which it takes afterwards is a cost
/// decision: every style has to return the same rows.
#[rstest]
#[case::heuristic(MaterializationStyle::Heuristic)]
#[case::all_early(MaterializationStyle::AllEarly)]
#[case::all_late(MaterializationStyle::AllLate)]
#[tokio::test]
async fn test_materialization_style(
    #[case] style: MaterializationStyle,
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)] version: LanceFileVersion,
) {
    let dataset = test_dataset_versioned(version).await;
    let scan_config = config(move |scan: &mut crate::dataset::Scanner| {
        scan.project(&["i", "s"])?
            .filter("i > 10 and i < 20")?
            .materialization_style(style.clone());
        Ok(scan)
    });
    assert_scan_keeps(&dataset, scan_config, |i| i > 10 && i < 20)
        .await
        .unwrap();
}

/// A wide column the filter also reads is not deferrable: the filtered pass has to load it either
/// way, so the read stays a single pass and there is no take.
#[tokio::test]
async fn test_a_filtered_column_is_never_late() {
    let dataset = test_dataset().await;
    let scan_config = config(|scan: &mut crate::dataset::Scanner| {
        scan.project(&["s"])?
            .filter("s IS NOT NULL")?
            .materialization_style(MaterializationStyle::AllLate);
        Ok(scan)
    });
    assert_scan_keeps(&dataset, &scan_config, |_| true)
        .await
        .unwrap();

    let plan = logical_plan_for(&dataset, &scan_config).await.unwrap();
    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert_eq!(
        display.matches("LanceRead").count(),
        1,
        "expected a single read: {display}"
    );
}

/// `AllEarlyExcept` names the late columns by field id, so it also covers the case where only
/// part of the projection is deferred.
#[tokio::test]
async fn test_all_early_except() {
    let dataset = test_dataset().await;
    let style = MaterializationStyle::all_early_except(&["s"], dataset.schema()).unwrap();
    let scan_config = config(move |scan: &mut crate::dataset::Scanner| {
        scan.project(&["i", "s"])?
            .filter("i > 10 and i < 20")?
            .materialization_style(style.clone());
        Ok(scan)
    });
    assert_scan_keeps(&dataset, scan_config, |i| i > 10 && i < 20)
        .await
        .unwrap();
}

/// Late materialization is a cost decision, and equivalence cannot see cost. Assert the point of
/// it directly: a selective filter over a wide column reads less when the column is taken
/// afterwards than when it is read for every row.
#[tokio::test]
async fn test_late_materialization_reads_less() {
    use crate::dataset::{Dataset, WriteParams};
    use arrow::datatypes::Int32Type;
    use lance_datagen::{BatchCount, ByteCount, RowCount, array, gen_batch};

    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("wide", array::rand_fixedbin(ByteCount::from(4096), false))
        .into_reader_rows(RowCount::from(100), BatchCount::from(4));
    let dataset = Dataset::write(data, "memory://", Some(WriteParams::default()))
        .await
        .unwrap();

    async fn read_bytes(dataset: &Dataset, style: MaterializationStyle) -> u64 {
        let scan_config = config(move |scan: &mut crate::dataset::Scanner| {
            scan.project(&["wide"])?
                .filter("i = 7")?
                .materialization_style(style.clone());
            Ok(scan)
        });
        let plan = logical_plan_for(dataset, scan_config).await.unwrap();
        let _ = dataset.object_store.as_ref().io_stats_incremental();
        run(plan).await.unwrap();
        dataset
            .object_store
            .as_ref()
            .io_stats_incremental()
            .read_bytes
    }

    let early = read_bytes(&dataset, MaterializationStyle::AllEarly).await;
    let late = read_bytes(&dataset, MaterializationStyle::AllLate).await;
    assert!(late < early, "late read {late} bytes, early read {early}");
}

/// `SELECT 1 AS foo` reads nothing, and is refused rather than answered with an invented column.
#[tokio::test]
async fn test_a_projection_that_reads_nothing_is_rejected() {
    let dataset = test_dataset().await;
    let error = logical_plan_for(&dataset, |scan| {
        scan.project_with_transform(&[("foo", "1")])
    })
    .await
    .expect_err("nothing to read");

    assert!(
        matches!(error, crate::Error::NotSupported { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains("at least one column"), "{error}");
}
