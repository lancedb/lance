// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray, UInt32Array};
use lance::dataset::scanner::ColumnOrdering;
use lance::Dataset;
use lance_index::scalar::inverted::query::{FtsQuery, PhraseQuery};
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::{DatasetIndexExt, IndexType};
use tantivy::tokenizer::Language;

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
