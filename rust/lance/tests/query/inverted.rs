// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use lance::Dataset;
use lance::dataset::scanner::ColumnOrdering;
use lance::dataset::{InsertBuilder, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::scalar::inverted::Language;
use lance_index::scalar::inverted::query::{FtsQuery, PhraseQuery};
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_table::format::IndexMetadata;

use super::{strip_score_column, test_fts, test_scan, test_take};
use crate::utils::DatasetTestCases;

// Build baseline inverted index parameters for tests, toggling token positions.
fn base_inverted_params(with_position: bool) -> InvertedIndexParams {
    InvertedIndexParams::new("simple".to_string(), Language::English)
        .with_position(with_position)
        .lower_case(true)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
        .max_token_length(None)
}

fn params_for(base_tokenizer: &str, lower_case: bool, with_position: bool) -> InvertedIndexParams {
    InvertedIndexParams::new(base_tokenizer.to_string(), Language::English)
        .with_position(with_position)
        .lower_case(lower_case)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
        .max_token_length(None)
}

// Execute a full-text search with optional filter and deterministic id ordering.
async fn run_fts(ds: &Dataset, query: FullTextSearchQuery, filter: Option<&str>) -> RecordBatch {
    let mut scanner = ds.scan();
    scanner.full_text_search(query).unwrap();
    if let Some(predicate) = filter {
        scanner.filter(predicate).unwrap();
    }
    scanner
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "id".to_string(),
        )]))
        .unwrap();
    scanner.try_into_batch().await.unwrap()
}

// Run an FTS query and assert results match a deterministic expected batch.
async fn assert_fts_expected(
    original: &RecordBatch,
    ds: &Dataset,
    query: FullTextSearchQuery,
    filter: Option<&str>,
    expected_ids: &[i32],
) {
    let scanned = run_fts(ds, query, filter).await;
    let scanned = strip_score_column(&scanned, original.schema().as_ref());

    let indices_u32: Vec<u32> = expected_ids.iter().map(|&i| i as u32).collect();
    let indices_array = UInt32Array::from(indices_u32);
    let expected = arrow::compute::take_record_batch(original, &indices_array).unwrap();

    // Ensure ordering is deterministic (id asc) and matches the expected rows.
    assert_eq!(&expected, &scanned);
}

#[tokio::test]
// Ensure indexed and non-indexed full-text search return the same ids.
async fn test_inverted_basic_equivalence() {
    let ids = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>()));
    let text_values = vec![
        Some("hello world"),
        Some("world hello"),
        Some("hello"),
        Some("lance database"),
        Some(""),
        None,
        Some("hello lance"),
        Some("lance"),
        Some("database"),
        Some("world"),
    ];
    let text = Arc::new(StringArray::from(text_values)) as ArrayRef;
    let batch = RecordBatch::try_from_iter(vec![("id", ids as ArrayRef), ("text", text)]).unwrap();

    DatasetTestCases::from_data(batch.clone())
        .run(|ds, original| async move {
            let mut ds = ds;
            let query = FullTextSearchQuery::new("hello".to_string())
                .with_column("text".to_string())
                .unwrap();

            let expected_ids = vec![0, 1, 2, 6];
            assert_fts_expected(&original, &ds, query.clone(), None, &expected_ids).await;

            let params = base_inverted_params(false);
            ds.create_index(&["text"], IndexType::Inverted, None, &params, true)
                .await
                .unwrap();
            assert_fts_expected(&original, &ds, query.clone(), None, &expected_ids).await;
            test_fts(&original, &ds, "text", "hello", None, true, false).await;

            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
        })
        .await;
}

#[tokio::test]
// Verify phrase queries require token positions and match contiguous terms.
async fn test_inverted_phrase_query_with_positions() {
    let ids = Arc::new(Int32Array::from((0..6).collect::<Vec<i32>>()));
    let text_values = vec![
        Some("lance database"),
        Some("lance and database"),
        Some("database lance"),
        Some("lance database test"),
        Some("lance database"),
        None,
    ];
    let text = Arc::new(StringArray::from(text_values)) as ArrayRef;
    let batch = RecordBatch::try_from_iter(vec![("id", ids as ArrayRef), ("text", text)]).unwrap();

    DatasetTestCases::from_data(batch.clone())
        .run(|ds, original| async move {
            let mut ds = ds;
            let params = base_inverted_params(true);
            ds.create_index(&["text"], IndexType::Inverted, None, &params, true)
                .await
                .unwrap();

            let phrase = PhraseQuery::new("lance database".to_string())
                .with_column(Some("text".to_string()));
            let query = FullTextSearchQuery::new_query(FtsQuery::Phrase(phrase));

            assert_fts_expected(&original, &ds, query, None, &[0, 3, 4]).await;
            test_fts(&original, &ds, "text", "lance database", None, true, true).await;
        })
        .await;
}

#[tokio::test]
async fn test_segmented_inverted_match_query() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let batches = vec![
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![Some("alpha lance"), Some("beta")])) as ArrayRef,
            ),
        ])
        .unwrap(),
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![Some("lance delta"), Some("gamma")])) as ArrayRef,
            ),
        ])
        .unwrap(),
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![Some("omega"), Some("lance omega")])) as ArrayRef,
            ),
        ])
        .unwrap(),
    ];
    let schema = batches[0].schema();
    let original = arrow_select::concat::concat_batches(&schema, &batches).unwrap();

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut ds = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = base_inverted_params(false);
    let fragment_ids = ds
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let mut metadatas = Vec::<IndexMetadata>::with_capacity(fragment_ids.len());
    for fragment_id in fragment_ids {
        let mut builder = ds
            .create_index_builder(&["text"], IndexType::Inverted, &params)
            .name("segmented_fts".to_string())
            .fragments(vec![fragment_id]);
        metadatas.push(builder.execute_uncommitted().await.unwrap());
    }
    let segments = ds
        .create_index_segment_builder()
        .with_index_type(IndexType::Inverted)
        .with_segments(metadatas.clone())
        .build_all()
        .await
        .unwrap();
    ds.commit_existing_index_segments("segmented_fts", "text", segments)
        .await
        .unwrap();
    assert!(metadatas.len() >= 2);
    assert_eq!(
        ds.load_indices_by_name("segmented_fts")
            .await
            .unwrap()
            .len(),
        metadatas.len()
    );

    let query = FullTextSearchQuery::new("lance".to_string())
        .with_column("text".to_string())
        .unwrap();
    assert_fts_expected(&original, &ds, query.clone(), None, &[0, 2, 5]).await;
    test_fts(&original, &ds, "text", "lance", None, true, false).await;
}

#[tokio::test]
async fn test_segmented_inverted_fuzzy_match_uses_global_idf() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let batches = vec![
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![0])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![Some("lance")])) as ArrayRef,
            ),
        ])
        .unwrap(),
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![1])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![Some("lance lance lance")])) as ArrayRef,
            ),
        ])
        .unwrap(),
    ];
    let schema = batches[0].schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let mut ds = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 1,
            max_rows_per_group: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = base_inverted_params(false);
    let fragment_ids = ds
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let mut metadatas = Vec::<IndexMetadata>::with_capacity(fragment_ids.len());
    for fragment_id in fragment_ids {
        let mut builder = ds
            .create_index_builder(&["text"], IndexType::Inverted, &params)
            .name("segmented_fuzzy".to_string())
            .fragments(vec![fragment_id]);
        metadatas.push(builder.execute_uncommitted().await.unwrap());
    }
    let segments = ds
        .create_index_segment_builder()
        .with_index_type(IndexType::Inverted)
        .with_segments(metadatas)
        .build_all()
        .await
        .unwrap();
    ds.commit_existing_index_segments("segmented_fuzzy", "text", segments)
        .await
        .unwrap();

    let batch = ds
        .scan()
        .full_text_search(
            FullTextSearchQuery::new_fuzzy("lnce".to_string(), Some(1))
                .with_column("text".to_string())
                .unwrap()
                .limit(Some(1)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch["id"].as_primitive::<arrow_array::types::Int32Type>();
    assert_eq!(ids.values(), &[1]);
}

#[tokio::test]
async fn test_segmented_inverted_phrase_query() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let batches = vec![
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![
                    Some("lance database"),
                    Some("database lance"),
                ])) as ArrayRef,
            ),
        ])
        .unwrap(),
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![
                    Some("lance database query"),
                    Some("lance and database"),
                ])) as ArrayRef,
            ),
        ])
        .unwrap(),
    ];
    let schema = batches[0].schema();
    let original = arrow_select::concat::concat_batches(&schema, &batches).unwrap();

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut ds = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = base_inverted_params(true);
    let fragment_ids = ds
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let mut metadatas = Vec::<IndexMetadata>::with_capacity(fragment_ids.len());
    for fragment_id in fragment_ids {
        let mut builder = ds
            .create_index_builder(&["text"], IndexType::Inverted, &params)
            .name("segmented_phrase_fts".to_string())
            .fragments(vec![fragment_id]);
        metadatas.push(builder.execute_uncommitted().await.unwrap());
    }
    let segments = ds
        .create_index_segment_builder()
        .with_index_type(IndexType::Inverted)
        .with_segments(metadatas)
        .build_all()
        .await
        .unwrap();
    ds.commit_existing_index_segments("segmented_phrase_fts", "text", segments)
        .await
        .unwrap();

    let phrase =
        PhraseQuery::new("lance database".to_string()).with_column(Some("text".to_string()));
    let query = FullTextSearchQuery::new_query(FtsQuery::Phrase(phrase));
    assert_fts_expected(&original, &ds, query, None, &[0, 2]).await;
    test_fts(&original, &ds, "text", "lance database", None, true, true).await;
}

#[tokio::test]
async fn test_segmented_inverted_match_query_with_unindexed_fragments() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let initial_batches = vec![
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![Some("lance zero"), Some("alpha")])) as ArrayRef,
            ),
        ])
        .unwrap(),
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef),
            (
                "text",
                Arc::new(StringArray::from(vec![Some("beta"), Some("lance three")])) as ArrayRef,
            ),
        ])
        .unwrap(),
    ];
    let schema = initial_batches[0].schema();
    let reader =
        RecordBatchIterator::new(initial_batches.clone().into_iter().map(Ok), schema.clone());
    let mut ds = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = base_inverted_params(false);
    let fragment_ids = ds
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let mut metadatas = Vec::<IndexMetadata>::with_capacity(fragment_ids.len());
    for fragment_id in fragment_ids {
        let mut builder = ds
            .create_index_builder(&["text"], IndexType::Inverted, &params)
            .name("segmented_mixed_fts".to_string())
            .fragments(vec![fragment_id]);
        metadatas.push(builder.execute_uncommitted().await.unwrap());
    }
    let segments = ds
        .create_index_segment_builder()
        .with_index_type(IndexType::Inverted)
        .with_segments(metadatas)
        .build_all()
        .await
        .unwrap();
    ds.commit_existing_index_segments("segmented_mixed_fts", "text", segments)
        .await
        .unwrap();

    let appended = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef),
        (
            "text",
            Arc::new(StringArray::from(vec![Some("lance four"), Some("omega")])) as ArrayRef,
        ),
    ])
    .unwrap();
    let appended_reader = RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema());
    ds.append(appended_reader, None).await.unwrap();

    let original = arrow_select::concat::concat_batches(
        &schema,
        &[
            initial_batches[0].clone(),
            initial_batches[1].clone(),
            appended,
        ],
    )
    .unwrap();
    let query = FullTextSearchQuery::new("lance".to_string())
        .with_column("text".to_string())
        .unwrap();
    assert_fts_expected(&original, &ds, query.clone(), None, &[0, 3, 4]).await;
    test_fts(&original, &ds, "text", "lance", None, true, false).await;
}

#[tokio::test]
// Validate filters are applied alongside inverted index search results.
async fn test_inverted_with_filter() {
    let ids = Arc::new(Int32Array::from((0..5).collect::<Vec<i32>>()));
    let text_values = vec![
        Some("lance database"),
        Some("lance vector"),
        Some("random text"),
        Some("lance"),
        None,
    ];
    let categories = vec![
        Some("keep"),
        Some("drop"),
        Some("keep"),
        Some("keep"),
        Some("keep"),
    ];
    let text = Arc::new(StringArray::from(text_values)) as ArrayRef;
    let category = Arc::new(StringArray::from(categories)) as ArrayRef;
    let batch = RecordBatch::try_from_iter(vec![
        ("id", ids as ArrayRef),
        ("text", text),
        ("category", category),
    ])
    .unwrap();

    DatasetTestCases::from_data(batch.clone())
        .with_index_types(
            "category",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds, original| async move {
            let mut ds = ds;
            let params = base_inverted_params(false);
            ds.create_index(&["text"], IndexType::Inverted, None, &params, true)
                .await
                .unwrap();

            let query = FullTextSearchQuery::new("lance".to_string())
                .with_column("text".to_string())
                .unwrap();
            assert_fts_expected(&original, &ds, query, Some("category = 'keep'"), &[0, 3]).await;
            test_fts(
                &original,
                &ds,
                "text",
                "lance",
                Some("category = 'keep'"),
                true,
                false,
            )
            .await;
        })
        .await;
}

#[tokio::test]
// Validate tokenizer/lowercase/position parameter combinations against expected matches.
async fn test_inverted_params_combinations() {
    let ids = Arc::new(Int32Array::from((0..5).collect::<Vec<i32>>()));
    let text_values = vec![
        Some("Hello there, this is a longer sentence about Lance."),
        Some("In this longer sentence we say hello to the database."),
        Some("Another line: hello world appears in a longer phrase."),
        Some("Saying HELLO loudly in a long sentence for testing."),
        None,
    ];
    let text = Arc::new(StringArray::from(text_values)) as ArrayRef;
    let batch = RecordBatch::try_from_iter(vec![("id", ids as ArrayRef), ("text", text)]).unwrap();

    let cases = vec![
        (
            "simple_lc_pos",
            params_for("simple", true, true),
            vec![0, 1, 2, 3],
            true,
        ),
        (
            "simple_no_lc",
            params_for("simple", false, false),
            vec![1, 2],
            false,
        ),
        (
            "whitespace_lc",
            params_for("whitespace", true, false),
            vec![0, 1, 2, 3],
            true,
        ),
        (
            "whitespace_no_lc_pos",
            params_for("whitespace", false, true),
            vec![1, 2],
            false,
        ),
    ];

    for (_name, params, expected, lower_case) in cases {
        let params = params.clone();
        let expected = expected.clone();
        DatasetTestCases::from_data(batch.clone())
            .with_index_types_and_inverted_index_params("text", [Some(IndexType::Inverted)], params)
            .run(|ds, original| {
                let expected = expected.clone();
                async move {
                    let query = FullTextSearchQuery::new("hello".to_string())
                        .with_column("text".to_string())
                        .unwrap();
                    assert_fts_expected(&original, &ds, query.clone(), None, &expected).await;
                    test_fts(&original, &ds, "text", "hello", None, lower_case, false).await;
                }
            })
            .await;
    }
}

/// Regression test: FTS query after deleting rows should not crash with
/// "Attempt to merge two RecordBatch with different sizes".
///
/// When stable row IDs are enabled, the FTS index may return row IDs for
/// deleted rows. The row ID index excludes deleted rows, so get_row_addrs()
/// must filter the input batch to match. Without this filtering, the
/// downstream merge in TakeExec fails with a size mismatch.
#[tokio::test]
async fn test_fts_after_delete_with_stable_row_ids() {
    let ids = Arc::new(Int32Array::from((0..20).collect::<Vec<i32>>()));
    // Give each row a unique word + a common word "shared"
    let texts: Vec<Option<&str>> = (0..20)
        .map(|i| match i % 4 {
            0 => Some("alpha shared"),
            1 => Some("beta shared"),
            2 => Some("gamma shared"),
            _ => Some("delta shared"),
        })
        .collect();
    let text_col = Arc::new(StringArray::from(texts));
    let batch = RecordBatch::try_from_iter(vec![
        ("id", ids as ArrayRef),
        ("text", text_col as ArrayRef),
    ])
    .unwrap();

    // Create dataset with stable row IDs
    let mut ds = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();

    // Create FTS index
    let params = InvertedIndexParams::default();
    ds.create_index_builder(&["text"], IndexType::Inverted, &params)
        .await
        .unwrap();

    // Delete some rows — these will still be referenced by the FTS index
    ds.delete("id IN (0, 1, 2, 3, 4)").await.unwrap();

    // FTS query for "shared" — matches ALL rows including deleted ones.
    // Before the fix, this would crash with a merge size mismatch.
    let query = FullTextSearchQuery::new("shared".to_string())
        .with_column("text".to_string())
        .unwrap();
    let mut scanner = ds.scan();
    scanner.full_text_search(query).unwrap();
    scanner
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "id".to_string(),
        )]))
        .unwrap();
    let result = scanner.try_into_batch().await.unwrap();

    // Should only have 15 rows (20 - 5 deleted)
    assert_eq!(result.num_rows(), 15);

    // Verify no deleted IDs are present
    let result_ids = result
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    for id in result_ids.values().iter() {
        assert!(*id >= 5, "Deleted row id {} should not appear", id);
    }
}
