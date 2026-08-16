// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector search: flat, ANN, prefilter/postfilter, and partial index coverage.

use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array, gen_batch};

use crate::dataset::Dataset;

use super::harness::*;

#[tokio::test]
pub(super) async fn test_flat_knn_plan() {
    let dataset = vector_dataset().await;

    // The projection below the search is narrowed to `[vec, _rowid]` by DataFusion's
    // `optimize_projections` reaching through the extension node via `necessary_children_exprs`;
    // `i` and `s` are fetched afterwards by the take.
    //
    // There is exactly one sort, and it is the top-k. The builder also restates the distance
    // ordering above the take — a logical `Sort` is the only way to say "this is the result's
    // order" — and `EnforceSorting` drops it, because the take advertises that it preserved the
    // ordering it was given.
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
pub(super) async fn test_paths_agree_on_flat_knn() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_flat_knn_with_projection() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["i"])?.nearest("vec", &query_vector(), 7)
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_ann_plan_uses_the_index() {
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
pub(super) async fn test_paths_agree_on_ann() {
    let dataset = indexed_vector_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

/// `use_index(false)` is a semantic downgrade to exact search, so the rule must not pick the
/// index even though one exists.
#[tokio::test]
pub(super) async fn test_exact_accuracy_forces_flat() {
    let dataset = indexed_vector_dataset().await;
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    scan.nearest("vec", &query_vector(), 5).unwrap();
    scan.use_index(false);

    let plan = super::super::create_plan(&scan).await.unwrap();
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
pub(super) async fn test_planning_is_io_free_once_warm() {
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
pub(super) async fn test_ann_prefilter_feeds_the_index() {
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
pub(super) async fn test_ann_postfilter_stays_above_the_search() {
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
pub(super) async fn test_paths_agree_on_ann_prefilter() {
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
pub(super) async fn test_paths_agree_on_flat_knn_prefilter() {
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
pub(super) async fn partially_indexed_dataset() -> Dataset {
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
pub(super) async fn test_combined_knn_ann_splits_into_two_branches() {
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
pub(super) async fn test_paths_agree_on_combined_knn_ann() {
    let dataset = partially_indexed_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

/// The stress case from Appendix A: one logical `Filter` has to reach both branches — as a
/// prefilter source on the indexed side and as an ordinary predicate on the flat side.
#[tokio::test]
pub(super) async fn test_paths_agree_on_combined_knn_ann_prefilter() {
    let dataset = partially_indexed_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.prefilter(true)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
}

// ---------------------------------------------------------------------------------------------
// Index segment selection and metric resolution
// ---------------------------------------------------------------------------------------------

async fn index_segment_uuids(dataset: &Dataset) -> Vec<uuid::Uuid> {
    use crate::index::DatasetIndexExt;

    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .map(|index| index.uuid)
        .collect()
}

#[tokio::test]
async fn test_paths_agree_on_an_explicit_index_segment() {
    let dataset = indexed_vector_dataset().await;
    let segments = index_segment_uuids(&dataset).await;
    let query = query_vector();

    assert_paths_agree(&dataset, |scan| {
        scan.nearest("vec", &query, 10)?
            .minimum_nprobes(2)
            .with_index_segments(segments.clone())
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_unknown_index_segment_is_rejected() {
    let dataset = indexed_vector_dataset().await;
    let mut scan = dataset.scan();
    scan.nearest("vec", &query_vector(), 10)
        .unwrap()
        .with_index_segments(vec![uuid::Uuid::nil()])
        .unwrap();

    let err = super::super::create_plan(&scan)
        .await
        .expect_err("an unknown segment must be rejected");
    assert!(err.to_string().contains("unknown index segments"), "{err}");
}

/// Without an explicit segment list a metric mismatch falls back to brute force; with one it is an
/// error, because the caller named segments that cannot answer the question they asked.
#[tokio::test]
async fn test_index_segment_with_a_conflicting_metric_is_rejected() {
    use lance_linalg::distance::DistanceType;

    let dataset = indexed_vector_dataset_with_metric(DistanceType::Cosine).await;
    let segments = index_segment_uuids(&dataset).await;
    let mut scan = dataset.scan();
    scan.nearest("vec", &query_vector(), 10)
        .unwrap()
        .distance_metric(DistanceType::L2)
        .with_index_segments(segments)
        .unwrap();

    let err = super::super::create_plan(&scan)
        .await
        .expect_err("a conflicting metric must be rejected");
    assert!(err.to_string().contains("requested metric"), "{err}");
}

/// A search that names no metric adopts the index's rather than falling back to brute force on a
/// mismatch with the element type's default.
#[tokio::test]
async fn test_search_without_a_metric_adopts_the_index_metric() {
    use lance_linalg::distance::DistanceType;

    let dataset = indexed_vector_dataset_with_metric(DistanceType::Cosine).await;
    let query = query_vector();
    let plan = logical_plan_for(&dataset, |scan| {
        Ok(scan.nearest("vec", &query, 10)?.minimum_nprobes(2))
    })
    .await
    .unwrap();

    let display = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(
        display.contains("ANNSubIndex"),
        "search fell back to flat:\n{display}"
    );
    assert!(display.contains("metric=Cosine"), "{display}");

    assert_paths_agree(&dataset, |scan| {
        Ok(scan.nearest("vec", &query, 10)?.minimum_nprobes(2))
    })
    .await
    .unwrap();
}

/// A multivector column has its own physical path, which this one does not lower to.
///
/// Worth a test rather than a comment: nothing in the vector lowering inspects the column type, so
/// without the guard a multivector search would build a single-vector fanout and return wrong rows.
#[tokio::test]
async fn test_multivector_search_is_rejected() {
    let dataset = multivector_dataset().await;
    let mut scan = dataset.scan();
    scan.nearest("vec", &query_vector(), 5).unwrap();

    let err = super::super::create_plan(&scan)
        .await
        .expect_err("multivector search is not supported yet");
    assert!(err.to_string().contains("multivector"), "{err}");
}

/// `vec` is `List<FixedSizeList<Float32, DIM>>` — two vectors per row.
async fn multivector_dataset() -> crate::dataset::Dataset {
    use arrow::buffer::OffsetBuffer;
    use arrow_array::{
        FixedSizeListArray, Float32Array, Int32Array, ListArray, RecordBatch, RecordBatchIterator,
    };
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    const ROWS: usize = 64;
    const VECTORS_PER_ROW: usize = 2;

    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let vectors = FixedSizeListArray::try_new(
        item.clone(),
        DIM as i32,
        Arc::new(Float32Array::from(
            (0..ROWS * VECTORS_PER_ROW * DIM as usize)
                .map(|value| value as f32 % 13.0)
                .collect::<Vec<_>>(),
        )),
        None,
    )
    .unwrap();
    let element = Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(item, DIM as i32),
        true,
    ));
    let multivectors = ListArray::try_new(
        element.clone(),
        OffsetBuffer::from_lengths(std::iter::repeat_n(VECTORS_PER_ROW, ROWS)),
        Arc::new(vectors),
        None,
    )
    .unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("vec", DataType::List(element), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from((0..ROWS as i32).collect::<Vec<_>>())),
            Arc::new(multivectors),
        ],
    )
    .unwrap();
    crate::dataset::Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        "memory://",
        None,
    )
    .await
    .unwrap()
}

/// A batch of query vectors, built the way the scanner's own batch tests build one.
fn batch_query_vectors(count: usize) -> arrow_array::FixedSizeListArray {
    use arrow_array::FixedSizeListArray;
    use lance_arrow::FixedSizeListArrayExt;

    let values = arrow_array::Float32Array::from(
        (0..count)
            .flat_map(|query| (0..DIM).map(move |v| (v + query as u32) as f32))
            .collect::<Vec<_>>(),
    );
    FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap()
}

/// Brute force answers every query in one pass, so the batch stays a single node.
#[tokio::test]
async fn test_paths_agree_on_a_flat_batch_search() {
    let dataset = vector_dataset().await;
    let queries = batch_query_vectors(3);
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &queries, 4))
        .await
        .unwrap();
}

/// The index fanout is single-query, so a batch becomes a union of one search per query.
#[tokio::test]
async fn test_paths_agree_on_an_indexed_batch_search() {
    let dataset = indexed_vector_dataset().await;
    let queries = batch_query_vectors(3);
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &queries, 4))
        .await
        .unwrap();
}

/// A prefilter applies to every query in the batch.
#[tokio::test]
async fn test_paths_agree_on_a_prefiltered_batch_search() {
    let dataset = indexed_vector_dataset().await;
    let queries = batch_query_vectors(2);
    assert_paths_agree(&dataset, |scan| {
        scan.nearest("vec", &queries, 4)?
            .filter("i < 500")?
            .prefilter(true);
        Ok(scan)
    })
    .await
    .unwrap();
}

/// `query_index` leads the output and groups the results, whichever path answered them.
#[tokio::test]
async fn test_batch_search_groups_results_by_query() {
    use arrow::datatypes::Int32Type;
    use arrow_array::cast::AsArray;

    let dataset = indexed_vector_dataset().await;
    let queries = batch_query_vectors(3);
    let plan = logical_plan_for(&dataset, |scan| scan.nearest("vec", &queries, 4))
        .await
        .unwrap();
    let batch = run(plan).await.unwrap();

    assert_eq!(batch.schema().field(0).name(), "query_index");
    let indices = batch["query_index"]
        .as_primitive::<Int32Type>()
        .values()
        .to_vec();
    assert_eq!(indices, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
}
