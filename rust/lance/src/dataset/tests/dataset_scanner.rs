// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;
use std::vec;

use crate::index::vector::VectorIndexParams;
use lance_arrow::json::{is_arrow_json_field, json_field, JsonArray};
use lance_arrow::FixedSizeListArrayExt;

use arrow::compute::concat_batches;
use arrow_array::{Array, FixedSizeListArray};
use arrow_array::{Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use futures::TryStreamExt;
use lance_arrow::SchemaExt;
use lance_index::scalar::inverted::{
    query::PhraseQuery, tokenizer::InvertedIndexParams, SCORE_FIELD,
};
use lance_index::scalar::{BuiltinIndexType, FullTextSearchQuery, ScalarIndexParams};
use lance_index::{vector::DIST_COL, DatasetIndexExt, IndexType};
use lance_linalg::distance::MetricType;

use crate::dataset::scanner::{DatasetRecordBatchStream, QueryFilter};
use crate::Dataset;
use lance_index::scalar::inverted::query::FtsQuery;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::Query;
use pretty_assertions::assert_eq;
use lance_core::utils::tempfile::TempStrDir;
use lance_encoding::version::LanceFileVersion;
use crate::dataset::scanner::test_dataset::TestVectorDataset;
use crate::dataset::WriteParams;

#[tokio::test]
async fn test_vector_filter_fts_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    let query_vector = Arc::new(Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]));
    let vector_query = Query {
        column: "vector".to_string(),
        key: query_vector,
        k: 5,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 20,
        maximum_nprobes: None,
        ef: None,
        refine_factor: None,
        metric_type: Some(MetricType::L2),
        use_index: true,
        dist_q_c: 0.0,
    };

    // Case 1: search with prefilter=true, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 3: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[299, 300],
    )
    .await;

    // Case 4: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .filter("category='geography'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 6: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 255],
    )
    .await;
}

#[tokio::test]
async fn test_fts_filter_vector_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    // Case 1: search with prefilter=true, query_filter=match("text")
    let query_vector = Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]);
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 255, 252, 249, 246],
    )
    .await;

    // Case 3: search with prefilter=false, query_filter=match("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299],
    )
    .await;

    // Case 4: search with prefilter=false, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[299, 300],
    )
    .await;

    // Case 6: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300],
    )
    .await;
}

#[tokio::test]
async fn test_scan_limit_offset_preserves_json_extension_metadata() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        json_field("meta", true),
    ]));

    let json_array = JsonArray::try_from_iter((0..50).map(|i| Some(format!(r#"{{"i":{i}}}"#))))
        .unwrap()
        .into_inner();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..50)),
            Arc::new(json_array),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), None).unwrap();
    let batch_no_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_no_offset.schema().field_with_name("meta").unwrap()
    ));

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), Some(10)).unwrap();
    let batch_with_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_with_offset.schema().field_with_name("meta").unwrap()
    ));
    assert_eq!(batch_no_offset.schema(), batch_with_offset.schema());
}

async fn prepare_query_filter_dataset() -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        ),
        ArrowField::new("text", DataType::Utf8, false),
        ArrowField::new("category", DataType::Utf8, false),
    ]));

    // Prepare dataset
    let mut vectors = vec![];
    for i in 1..=300 {
        vectors.extend(vec![i as f32; 4]);
    }

    // id 256..298 has noop, others has text
    let mut text = vec![];
    for i in 1..=255 {
        text.push(format!("text {}", i));
    }
    for i in 256..=298 {
        text.push(format!("noop {}", i));
    }
    text.extend(vec!["text 299".to_string(), "text 300".to_string()]);

    let mut category = vec![];
    for i in 1..=300 {
        if i % 3 == 1 {
            category.push("literature".to_string());
        } else if i % 3 == 2 {
            category.push("science".to_string());
        } else {
            category.push("geography".to_string());
        }
    }

    let vectors = Float32Array::from(vectors);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(1..=300)),
            Arc::new(FixedSizeListArray::try_new_from_values(vectors, 4).unwrap()),
            Arc::new(StringArray::from(text)),
            Arc::new(StringArray::from(category)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create index
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        PQBuildParams::new(4, 8),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().with_position(true),
            true,
        )
        .await
        .unwrap();

    dataset
}

async fn check_results(
    stream: DatasetRecordBatchStream,
    expected_schema: SchemaRef,
    expected_ids: &[i32],
) {
    let results = stream.try_collect::<Vec<_>>().await.unwrap();
    let batch = concat_batches(&results[0].schema(), &results).unwrap();
    assert_eq!(batch.schema(), expected_schema);

    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), expected_ids);
}


#[tokio::test]
async fn test_prune_fragments_without_scalar_index_returns_all() {
    // Build a dataset with 5 fragments of 10 rows each: i = [0, 1, ..., 49].
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..50))],
    )
        .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10,
        max_rows_per_group: 10,
        ..Default::default()
    };
    let dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    let original_fragments = dataset.fragments().clone();

    // Without a scalar index, pruning should be a no-op and return all fragments.
    let pruned = dataset.prune_fragments("i >= 30").await.unwrap();

    std::assert_eq!(pruned.len(), original_fragments.len());
    let original_ids: Vec<u64> = original_fragments.iter().map(|f| f.id).collect();
    let pruned_ids: Vec<u64> = pruned.iter().map(|f| f.id).collect();
    std::assert_eq!(pruned_ids, original_ids);
}

#[tokio::test]
async fn test_prune_fragments_with_scalar_index_prunes_non_matching_fragments() {
    // Dataset with 5 fragments of 10 rows each: i = [0, 1, ..., 49].
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..50))],
    )
        .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10,
        max_rows_per_group: 10,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Create a scalar index on i so all current fragments are indexed.
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let fragments = dataset.fragments().clone();
    std::assert_eq!(fragments.len(), 5);

    // For filter i >= 30, all matching rows live in the last two fragments.
    let expected_tail_ids: Vec<u64> = fragments[fragments.len() - 2..]
        .iter()
        .map(|f| f.id)
        .collect();

    let pruned = dataset.prune_fragments("i >= 30").await.unwrap();
    let pruned_ids: Vec<u64> = pruned.iter().map(|f| f.id).collect();

    std::assert_eq!(pruned_ids, expected_tail_ids);
}

#[tokio::test]
async fn test_prune_fragments_with_scalar_index_and_mixed_or_filter_is_noop() {
    // Multi-column dataset with predictable fragment boundaries: 5 fragments
    // of 10 rows each, columns col_a, col_b, col_c.
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("col_a", DataType::Int32, false),
        ArrowField::new("col_b", DataType::Int32, false),
        ArrowField::new("col_c", DataType::Int32, false),
    ]));

    // col_a: 0..50 (monotonic sequence for range queries)
    let col_a = Int32Array::from_iter_values(0..50);
    // col_b: first fragment has small values (< 10) so rows there can only
    // match the filter via the non-indexed side `col_b < 10`; later fragments
    // have large values.
    let col_b = Int32Array::from_iter_values((0..50).map(|i| if i < 10 { i } else { 100 + i }));
    // col_c: arbitrary third column, no index.
    let col_c = Int32Array::from_iter_values((0..50).map(|i| i * 2));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(col_a), Arc::new(col_b), Arc::new(col_c)],
    )
        .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10,
        max_rows_per_group: 10,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Create a scalar index only on col_a.
    dataset
        .create_index(
            &["col_a"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let fragments = dataset.fragments().clone();
    std::assert_eq!(fragments.len(), 5);

    // For filter `col_a >= 10 OR col_b < 10`, only `col_a` is indexable. The
    // planner should treat this OR as mixed indexability and fall back to a
    // refine-only filter plan, so scalar-index-based pruning becomes a no-op
    // and all fragments are retained.
    let pruned = dataset
        .prune_fragments("col_a >= 10 OR col_b < 10")
        .await
        .unwrap();
    let original_ids: Vec<u64> = fragments.iter().map(|f| f.id).collect();
    let pruned_ids: Vec<u64> = pruned.iter().map(|f| f.id).collect();
    std::assert_eq!(pruned_ids, original_ids);
}

#[tokio::test]
async fn test_prune_fragments_keeps_fragments_outside_index_coverage() {
    // Use TestVectorDataset to construct a multi-fragment dataset with an Int32 column "i".
    // This matches the pattern used elsewhere in dataset_index.rs for multi-fragment tests.
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();

    // Build a scalar index on i covering the initial fragments.
    test_ds.make_scalar_index().await.unwrap();

    let before_fragments = test_ds.dataset.fragments().clone();
    let before_ids: HashSet<u64> = before_fragments.iter().map(|f| f.id).collect();

    // Append new data so the new fragment is not covered by the existing index.
    test_ds.append_new_data().await.unwrap();
    let all_fragments = test_ds.dataset.fragments().clone();
    let new_fragments: Vec<_> = all_fragments
        .iter()
        .filter(|f| !before_ids.contains(&f.id))
        .collect();

    // Sanity check: we expect exactly one newly appended fragment.
    std::assert_eq!(new_fragments.len(), 1);
    let new_fragment_id = new_fragments[0].id;

    // Use a filter that only matches early rows, which live in the original fragments.
    let pruned = test_ds.dataset.prune_fragments("i < 10").await.unwrap();
    let pruned_ids: HashSet<u64> = pruned.iter().map(|f| f.id).collect();

    // Fragments without index coverage must not be pruned.
    assert!(pruned_ids.contains(&new_fragment_id));
}

#[tokio::test]
async fn test_prune_fragments_with_zonemap_scalar_index_prunes_non_matching_fragments() {
    // Dataset with 5 fragments of 10 rows each: z = [0, 1, ..., 49].
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "z",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..50))],
    )
        .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10,
        max_rows_per_group: 10,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Create a ZoneMap scalar index on z so all current fragments are indexed.
    let zonemap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    dataset
        .create_index(&["z"], IndexType::Scalar, None, &zonemap_params, true)
        .await
        .unwrap();

    let fragments = dataset.fragments().clone();
    std::assert_eq!(fragments.len(), 5);

    // For filter z >= 30, all matching rows live in the last two fragments.
    // ZoneMap returns an AtMost allow-list mask, and scalar_indexed_prune_fragments
    // prunes covered fragments that have no allowed rows while keeping uncovered
    // fragments, so we expect only the tail fragments to remain.
    let expected_tail_ids: Vec<u64> = fragments[fragments.len() - 2..]
        .iter()
        .map(|f| f.id)
        .collect();

    let pruned = dataset.prune_fragments("z >= 30").await.unwrap();
    let pruned_ids: Vec<u64> = pruned.iter().map(|f| f.id).collect();

    std::assert_eq!(pruned_ids, expected_tail_ids);
}

#[tokio::test]
async fn test_prune_fragments_with_scalar_index_blocklist_partial_keeps_all_fragments() {
    // Dataset with 5 fragments of 10 rows each: i = [0, 1, ..., 49].
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..50))],
    )
        .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10,
        max_rows_per_group: 10,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Create a scalar BTree index on i so all current fragments are indexed.
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let original_fragments = dataset.fragments().clone();
    std::assert_eq!(original_fragments.len(), 5);

    // Filter i != 30 is implemented as NOT(i = 30). The scalar index evaluates the
    // equality as an exact allow-list and then negates it to an exact block-list
    // containing only the single row with i = 30. Since no fragment is fully
    // blocked in the resulting RowAddrMask::BlockList, scalar_indexed_prune_fragments
    // must keep all fragments and preserve their manifest order.
    let pruned = dataset.prune_fragments("i != 30").await.unwrap();

    std::assert_eq!(pruned.len(), original_fragments.len());
    let original_ids: Vec<u64> = original_fragments.iter().map(|f| f.id).collect();
    let pruned_ids: Vec<u64> = pruned.iter().map(|f| f.id).collect();
    std::assert_eq!(pruned_ids, original_ids);
}

#[tokio::test]
async fn test_prune_fragments_with_scalar_index_blocklist_empty_keeps_all_fragments() {
    // Dataset with 5 fragments of 10 rows each: i = [0, 1, ..., 49].
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..50))],
    )
        .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10,
        max_rows_per_group: 10,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Create a scalar BTree index on i so all current fragments are indexed.
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let original_fragments = dataset.fragments().clone();
    std::assert_eq!(original_fragments.len(), 5);

    // Filter i != 1000 is implemented as NOT(i = 1000). The equality matches no
    // rows, so the negated scalar index result is an exact block-list with an
    // empty RowAddrTreeMap. scalar_indexed_prune_fragments treats an empty
    // RowAddrMask::BlockList as "no fragment is blocked" and returns all
    // fragments unchanged.
    let pruned = dataset.prune_fragments("i != 1000").await.unwrap();

    std::assert_eq!(pruned.len(), original_fragments.len());
    let original_ids: Vec<u64> = original_fragments.iter().map(|f| f.id).collect();
    let pruned_ids: Vec<u64> = pruned.iter().map(|f| f.id).collect();
    std::assert_eq!(pruned_ids, original_ids);
}
