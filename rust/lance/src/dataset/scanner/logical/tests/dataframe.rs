// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance search as DataFrame operators.
//!
//! The scanner's fixed query shape is not the oracle here — the point of these is queries the
//! scanner cannot express — so they assert directly on the rows instead.

use std::sync::Arc;

use arrow::datatypes::{Float32Type, Int32Type};
use arrow_array::cast::AsArray;
use datafusion::prelude::{SessionContext, col, lit};
use lance_index::vector::Query;

use super::fts::{fts_dataset, match_query};
use super::harness::*;
use crate::datafusion::{LanceContextExt, LanceDataFrameExt};
use crate::dataset::Dataset;

fn vector_query(dataset: &Dataset, k: usize) -> Query {
    let mut scan = dataset.scan();
    scan.nearest("vec", &query_vector(), k).unwrap();
    scan.nearest.clone().expect("nearest sets the query")
}

/// A filter through the DataFrame API reaches the same rows as the scanner's own.
#[tokio::test]
async fn test_dataframe_filter_matches_the_scanner() {
    let dataset = Arc::new(test_dataset().await);

    let ctx = SessionContext::new();
    let plan = ctx
        .read_lance_dataset(dataset.clone())
        .unwrap()
        .select(vec![col("i")])
        .unwrap()
        .filter(col("i").gt(lit(10)).and(col("i").lt(lit(20))))
        .unwrap()
        .lance_plan()
        .await
        .unwrap();

    let batch = run(plan).await.unwrap();
    let values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
    assert_eq!(values, (11..20).collect::<Vec<_>>());
}

/// A vector search over a frame that is a plain scan is free to use the index.
#[tokio::test]
async fn test_dataframe_vector_search_uses_the_index() {
    let dataset = Arc::new(indexed_vector_dataset().await);

    let ctx = SessionContext::new();
    let plan = ctx
        .read_lance_dataset(dataset.clone())
        .unwrap()
        .select(vec![col("i")])
        .unwrap()
        .nearest(vector_query(&dataset, 10))
        .unwrap()
        .lance_plan()
        .await
        .unwrap();

    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(display.contains("ANNSubIndex"), "{display}");

    let batch = run(plan).await.unwrap();
    assert_eq!(batch.num_rows(), 10);
    let distances = batch[lance_index::vector::DIST_COL]
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(batch.column_by_name("i").is_some(), "take did not run");
}

/// A filter below the search is a prefilter — the scan leaf applies it, and the search only ever
/// sees the surviving rows.
#[tokio::test]
async fn test_dataframe_filter_below_a_search_is_a_prefilter() {
    let dataset = Arc::new(indexed_vector_dataset().await);

    let ctx = SessionContext::new();
    let plan = ctx
        .read_lance_dataset(dataset.clone())
        .unwrap()
        .select(vec![col("i")])
        .unwrap()
        .filter(col("i").lt(lit(100)))
        .unwrap()
        .nearest(vector_query(&dataset, 5))
        .unwrap()
        .lance_plan()
        .await
        .unwrap();

    let batch = run(plan).await.unwrap();
    assert_eq!(batch.num_rows(), 5);
    assert!(
        batch["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .all(|value| *value < 100),
        "{batch:?}"
    );
}

/// The headline: text search then vector search, in one plan.
///
/// The vector search scores the text matches exactly rather than consulting an index, because its
/// input is already a candidate set. The scanner expresses this only as a `query_filter`; here it
/// is just two operators stacked.
#[tokio::test]
async fn test_dataframe_hybrid_text_then_vector() {
    let dataset = Arc::new(fts_dataset().await);

    let ctx = SessionContext::new();
    let plan = ctx
        .read_lance_dataset(dataset.clone())
        .unwrap()
        .select(vec![col("i"), col("s")])
        .unwrap()
        .full_text_search(match_query("hello"))
        .await
        .unwrap()
        .nearest(vector_query(&dataset, 5))
        .unwrap()
        .lance_plan()
        .await
        .unwrap();

    let display = datafusion::physical_plan::displayable(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(display.contains("MatchQuery"), "{display}");
    assert!(
        !display.contains("ANNSubIndex"),
        "a search over candidates must be exact: {display}"
    );

    let batch = run(plan).await.unwrap();
    assert_eq!(batch.num_rows(), 5);
    for text in batch["s"].as_string::<i32>().iter() {
        assert!(
            text.expect("s is not nullable").contains("hello"),
            "{batch:?}"
        );
    }
}

/// A search needs a Lance dataset under it, and says so rather than panicking on the downcast.
#[tokio::test]
async fn test_search_needs_a_lance_frame() {
    let dataset = Arc::new(indexed_vector_dataset().await);
    let query = vector_query(&dataset, 5);

    let ctx = SessionContext::new();
    let frame = ctx.read_empty().unwrap();
    let err = frame.nearest(query).expect_err("not a Lance frame");
    assert!(err.to_string().contains("read_lance_dataset"), "{err}");
}

/// Aggregating a frame's search results.
///
/// A frame can put an aggregate straight above a search, where the scanner always has a take in
/// between, so this is the first place DataFusion compares the search node's declared schema against
/// the plan it lowered to.
#[tokio::test]
async fn test_dataframe_aggregate_over_a_search() {
    use datafusion::functions_aggregate::expr_fn::count;

    let dataset = Arc::new(vector_dataset().await);

    let ctx = SessionContext::new();
    let plan = ctx
        .read_lance_dataset(dataset.clone())
        .unwrap()
        .nearest(vector_query(&dataset, 10))
        .unwrap()
        .aggregate(vec![], vec![count(lit(1))])
        .unwrap()
        .lance_plan()
        .await
        .unwrap();

    let batch = run(plan).await.unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(
        batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>()
            .value(0),
        10
    );
}
