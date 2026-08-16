// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whole-planner properties: guards, ordering, overlays, and output schema.

use futures::TryStreamExt;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_datagen::{Dimension, array, gen_batch};

use crate::Result;
use crate::dataset::Scanner;
use crate::dataset::scanner::ColumnOrdering;

/// A scanner-configuring closure. Generic rather than a `fn` pointer so a case can close over
/// its query vector or filter string, and taken by reference internally because every oracle
/// applies it twice — once per path.
trait ScanConfig: Fn(&mut Scanner) -> Result<&mut Scanner> {}
impl<F: Fn(&mut Scanner) -> Result<&mut Scanner>> ScanConfig for F {}

/// Sort a result batch by `_rowid` so two plans can be compared as sets.
use super::fts::*;
use super::harness::*;

#[tokio::test]
async fn test_unsupported_shape_is_rejected() {
    let dataset = test_dataset().await;
    let mut scan = dataset.scan();
    scan.with_row_id().include_deleted_rows();

    let err = super::super::create_plan(&scan).await.unwrap_err();
    assert!(
        matches!(err, crate::Error::NotSupported { .. }),
        "expected NotSupported, got {err:?}"
    );
    assert!(err.to_string().contains("include_deleted_rows"), "{err}");
}

#[tokio::test]
async fn test_paths_agree_on_ordering() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s", "i"])?
            .order_by(Some(vec![ColumnOrdering::desc_nulls_last("i".into())]))
    })
    .await
    .unwrap();
}

/// A limit must take the first rows *of the orderingwhich is only true if the sort is below the
/// limit. It also blocks the scan-range pushdown a bare limit would get.
#[tokio::test]
async fn test_paths_agree_on_ordering_with_limit() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s", "i"])?
            .limit(Some(7), Some(3))?
            .order_by(Some(vec![ColumnOrdering::desc_nulls_last("i".into())]))
    })
    .await
    .unwrap();
}

/// The ordering column is not projected, so it has to be read and then dropped again.
#[tokio::test]
async fn test_paths_agree_on_ordering_by_an_unprojected_column() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .filter("i > 20")?
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first("i".into())]))
    })
    .await
    .unwrap();
}

/// A vector `query_filter` under a *compound* FTS query takes the other rerank branch: the FTS
/// query is planned independently and joined to the vector results on `_rowid`. That join is the
/// only stock DataFusion join in the whole plan, and its physical form decides the output order —
/// so this asserts ordering, not just row membership.
#[tokio::test]
async fn test_paths_agree_on_vector_filter_prefiltering_a_compound_fts_search() {
    use crate::dataset::scanner::QueryFilter;
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::PhraseQuery;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(true)
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("hello world".to_owned())
                    .with_column(Some("s".to_owned()))
                    .into(),
            ))?
            .filter_query(QueryFilter::Vector(vector_filter_query()))
    })
    .await
    .unwrap();
}

/// An empty fragment selection: the imperative path recognizes it up front and emits an
/// `EmptyExec`, while the logical path plans the search normally and lets the take — restricted to
/// zero fragments — return nothing. The plan shapes differ; the answers must not.
#[tokio::test]
async fn test_paths_agree_on_an_empty_fragment_selection() {
    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.with_fragments(Vec::new());
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

/// A top-k whose result spans more than one output batch, at real parallelism. Regression test for
/// the take losing the search's distance ordering: `FilteredReadExec` advertises no output
/// ordering, so without an explicit sort above it `execute_plan` merges partitions with a plain
/// coalesce and the global order is scrambled.
#[tokio::test]
async fn test_flat_knn_large_limit_stays_globally_ordered() {
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow::datatypes::Float32Type;
    use arrow_array::cast::AsArray;
    use lance_index::vector::DIST_COL;

    let dataset = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
        .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(5_000))
        .await
        .unwrap();
    let query = arrow_array::Float32Array::from(vec![0.0_f32; 16]);

    // BATCH_SIZE_FALLBACK is 8192, so 12k results span batches. The scrambling is a scheduling
    // race, hence the repeats.
    for _ in 0..5 {
        let mut scan = dataset.scan();
        scan.nearest("vec", &query, 12_000).unwrap();
        scan.target_parallelism(8);
        let batch = run(super::super::create_plan(&scan).await.unwrap())
            .await
            .unwrap();

        assert_eq!(batch.num_rows(), 12_000);
        let distances = batch[DIST_COL].as_primitive::<Float32Type>();
        for pair in distances.values().windows(2) {
            assert!(
                pair[0] <= pair[1],
                "results must be globally sorted by distance, found {} before {}",
                pair[0],
                pair[1]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Data overlays: index coverage at row granularity
// ---------------------------------------------------------------------------------------------
//
// A data overlay committed after an index was built, touching a field that index covers, leaves
// the index describing values that no longer exist. Those rows are a coverage gap in exactly the
// sense a fragment the index never saw is one, and `SplitOnIndexCoverage` fills both the same way.
// The exhaustive behavioral coverage lives in `dataset::tests::dataset_overlay_index_masking`,
// which runs against whichever path the flag selects; these assert the two paths agree.

use crate::dataset::tests::dataset_overlay_index_masking as overlay_fixtures;

#[tokio::test]
async fn test_paths_agree_on_a_vector_search_over_an_overlay() {
    for stable_row_ids in [false, true] {
        let dataset = overlay_fixtures::create_vector_overlay_dataset(stable_row_ids).await;
        let query = arrow_array::Float32Array::from(overlay_fixtures::vec_query());
        assert_paths_agree(&dataset, |scan| {
            scan.nearest("vec", &query, 3)?
                .minimum_nprobes(1)
                .with_row_id()
                .project(&["id"])
        })
        .await
        .unwrap();
    }
}

/// `fast_search` blocks the stale rows but does not re-score them, so the two paths have to agree
/// on dropping the stale hit *and* on not surfacing the moved-on match.
#[tokio::test]
async fn test_paths_agree_on_a_fast_search_over_an_overlay() {
    let dataset = overlay_fixtures::create_vector_overlay_dataset(false).await;
    let query = arrow_array::Float32Array::from(overlay_fixtures::vec_query());
    assert_paths_agree(&dataset, |scan| {
        scan.nearest("vec", &query, 3)?
            .minimum_nprobes(1)
            .fast_search()
            .with_row_id()
            .project(&["id"])
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_an_fts_search_over_an_overlay() {
    let mut dataset = overlay_fixtures::create_text_dataset(false).await;
    overlay_fixtures::build_text_fts_index(&mut dataset).await;
    let dataset = overlay_fixtures::commit_overlay(
        dataset,
        "logical_fts_overlay",
        0,
        &[1],
        lance_table::format::overlay::OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([
            1,
        ])),
        vec![std::sync::Arc::new(arrow_array::StringArray::from(vec![
            "banana bread",
        ]))],
    )
    .await;

    assert_paths_agree(&dataset, |scan| {
        scan.project(&["id"])?
            .with_row_id()
            .full_text_search(text_match_query("apple"))
    })
    .await
    .unwrap();
}

/// The coverage split the shared rule cannot reach: a scalar index query is derived inside the
/// scan leaf, not carried by a logical node, so its stale rows are handled there.
#[tokio::test]
async fn test_paths_agree_on_a_scalar_index_over_an_overlay() {
    for stable_row_ids in [false, true] {
        let mut dataset = overlay_fixtures::create_base_dataset_with(stable_row_ids).await;
        overlay_fixtures::build_age_index(&mut dataset).await;
        let dataset = overlay_fixtures::commit_overlay(
            dataset,
            "logical_scalar_overlay",
            0,
            &[1],
            lance_table::format::overlay::OverlayCoverage::dense(
                roaring::RoaringBitmap::from_iter([1]),
            ),
            vec![overlay_fixtures::i32_array([Some(50)])],
        )
        .await;

        // `age = 50` now matches the overlaid row (whose index entry says 10) and the untouched
        // row that always said 50, so both the block and the re-read have to be right.
        for filter in ["age = 50", "age = 10", "age = 20"] {
            // Unordered: the coverage split makes this a union of an index lookup and a re-read of
            // the overlaid rows, and a union imposes no order between its branches. The two paths
            // build that union in different places — `scan_impl` versus `SplitOnIndexCoverage` —
            // so they interleave the branches differently while selecting the same rows.
            assert_paths_agree_unordered(&dataset, |scan| {
                scan.filter(filter)?.with_row_id().project(&["id"])
            })
            .await
            .unwrap();
        }
    }
}

/// The oracle compares the two paths to each other, so a schema fact both of them lose is
/// invisible to it. This asserts the fact against the dataset instead.
#[tokio::test]
async fn test_output_schema_keeps_dataset_metadata() -> Result<()> {
    let dataset = test_dataset().await;
    let expected = dataset.schema().metadata.clone();
    assert!(
        !expected.is_empty(),
        "fixture must carry schema metadata for this test to mean anything"
    );

    for (path, plan) in [
        (
            "imperative",
            imperative_plan_for(&dataset, |scan| scan.filter("i > 50")?.project(&["s"])).await?,
        ),
        (
            "logical",
            logical_plan_for(&dataset, |scan| scan.filter("i > 50")?.project(&["s"])).await?,
        ),
    ] {
        assert_eq!(
            plan.schema().metadata(),
            &expected,
            "{path} path's plan schema dropped the dataset's schema metadata"
        );
        // Checked separately from the plan schema because they can disagree, and it is the batch
        // the caller actually receives.
        let batches = execute_plan(plan, LanceExecutionOptions::default())?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(
            batches[0].schema().metadata(),
            &expected,
            "{path} path's output batches dropped the dataset's schema metadata"
        );
    }
    Ok(())
}

/// The check that replaced the deleted idempotence markers: a search whose access path was never
/// resolved is rejected as non-executable rather than lowered to a silent brute-force fallback.
///
/// DataFusion runs this check at the end of the analyzer, which is the stage the resolving rules
/// live in, so a rule that fails to fire surfaces as a planning error.
#[tokio::test]
async fn test_an_unresolved_search_is_not_executable() {
    use datafusion::logical_expr::InvariantLevel;

    let dataset = vector_dataset().await;
    let mut scan = dataset.scan();
    scan.nearest("vec", &query_vector(), 5).unwrap();

    let prepared = super::super::prepare::PreparedQueries::resolve(&scan)
        .await
        .unwrap();
    let plan = super::super::builder::build(&scan, &prepared).unwrap();

    let err = plan
        .check_invariants(InvariantLevel::Executable)
        .expect_err("an unresolved search must not pass the executable check");
    assert!(
        err.to_string().contains("no access path resolved"),
        "unexpected error: {err}"
    );
}
