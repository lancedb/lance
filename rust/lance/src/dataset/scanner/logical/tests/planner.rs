// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whole-planner properties: guards, ordering, overlays, and output schema.

use futures::TryStreamExt;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_datagen::{Dimension, array, gen_batch};

use crate::Result;
use crate::dataset::scanner::ColumnOrdering;

use super::fts::*;
use super::harness::*;

#[tokio::test]
async fn test_ordering() {
    let dataset = test_dataset().await;
    let fixture = Fixture::read(&dataset).await.unwrap();
    assert_scan_returns(
        &dataset,
        |scan| {
            scan.project(&["s", "i"])?
                .order_by(Some(vec![ColumnOrdering::desc_nulls_last("i".into())]))
        },
        &fixture,
        (0..200).rev().collect(),
    )
    .await
    .unwrap();
}

/// A limit must take the first rows *of the ordering*, which is only true if the sort is below
/// the limit. It also blocks the scan-range pushdown a bare limit would get.
#[tokio::test]
async fn test_ordering_with_limit() {
    let dataset = test_dataset().await;
    let fixture = Fixture::read(&dataset).await.unwrap();
    // Descending from 199, skip 3, keep 7: rows 196 down to 190. Taking the limit before the sort
    // would return the first rows of *storage* order instead.
    assert_scan_returns(
        &dataset,
        |scan| {
            scan.project(&["s", "i"])?
                .limit(Some(7), Some(3))?
                .order_by(Some(vec![ColumnOrdering::desc_nulls_last("i".into())]))
        },
        &fixture,
        (190..197).rev().collect(),
    )
    .await
    .unwrap();
}

/// The ordering column is not projected, so it has to be read and then dropped again.
#[tokio::test]
async fn test_ordering_by_an_unprojected_column() {
    let dataset = test_dataset().await;
    let fixture = Fixture::read(&dataset).await.unwrap();
    assert_scan_returns(
        &dataset,
        |scan| {
            scan.project(&["s"])?
                .filter("i > 20")?
                .order_by(Some(vec![ColumnOrdering::asc_nulls_first("i".into())]))
        },
        &fixture,
        (21..200).collect(),
    )
    .await
    .unwrap();
}

/// A vector `query_filter` under a *compound* FTS query takes the other rerank branch: the FTS
/// query is planned independently and joined to the vector results on `_rowid`. That join is the
/// only stock DataFusion join in the whole plan, and its physical form decides the output order —
/// so this asserts ordering, not just row membership.
#[tokio::test]
async fn test_vector_filter_prefiltering_a_compound_fts_search() {
    use crate::dataset::scanner::QueryFilter;
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::PhraseQuery;

    let dataset = fts_dataset().await;
    // Prefiltered, so the vector search spends its 20 first and the phrase trims what is left.
    let expected = hybrid_expectation(&dataset, contains_phrase("hello world"), 20, false).await;
    let fixture = Fixture::read(&dataset).await.unwrap();
    let batch = scan_rows(&dataset, |scan| {
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

    let mut found = fixture.ids_of(&row_ids_of(&batch));
    found.sort_unstable();
    assert_eq!(found, expected);
}

/// An empty fragment selection is planned like any other: the search runs and the take, restricted
/// to no fragments, returns nothing.
#[tokio::test]
async fn test_an_empty_fragment_selection_returns_nothing() {
    let dataset = fts_dataset().await;
    let batch = scan_rows(&dataset, |scan| {
        scan.with_fragments(Vec::new());
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
    assert_eq!(batch.num_rows(), 0);
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

/// A plan rebuilds its output schema several times on the way down, and the dataset's own schema
/// metadata has to survive every one of them.
#[tokio::test]
async fn test_output_schema_keeps_dataset_metadata() -> Result<()> {
    let dataset = test_dataset().await;
    let expected = dataset.schema().metadata.clone();
    assert!(
        !expected.is_empty(),
        "fixture must carry schema metadata for this test to mean anything"
    );

    let plan = logical_plan_for(&dataset, |scan| scan.filter("i > 50")?.project(&["s"])).await?;
    assert_eq!(
        plan.schema().metadata(),
        &expected,
        "the plan's schema dropped the dataset's schema metadata"
    );
    // Checked separately from the plan schema because they can disagree, and it is the batch the
    // caller actually receives.
    let batches = execute_plan(plan, LanceExecutionOptions::default())?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(
        batches[0].schema().metadata(),
        &expected,
        "the output batches dropped the dataset's schema metadata"
    );
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
