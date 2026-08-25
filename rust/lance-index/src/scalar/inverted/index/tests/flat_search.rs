// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

fn text_stream(batches: Vec<Vec<String>>) -> SendableRecordBatchStream {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", DataType::Utf8, false),
    ]));
    let mut next_row_id = 0_u64;
    let batches = batches
        .into_iter()
        .map(|documents| {
            let row_ids = (next_row_id..next_row_id + documents.len() as u64).collect::<Vec<_>>();
            next_row_id += documents.len() as u64;
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
                    Arc::new(StringArray::from(documents)) as ArrayRef,
                ],
            )
            .map_err(datafusion_common::DataFusionError::from)
        })
        .collect::<Vec<_>>();
    Box::pin(RecordBatchStreamAdapter::new(schema, stream::iter(batches)))
}

fn simple_text_tokenizer() -> Box<dyn LanceTokenizer> {
    use crate::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
    use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};

    Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ))
}

#[tokio::test]
async fn flat_fuzzy_candidates_are_bounded_across_chunks_not_occurrences() {
    let query_tokens = Arc::new(Tokens::new(vec!["catx".to_string()], DocType::Text));
    let params = Arc::new(
        FtsSearchParams::new()
            .with_fuzziness(Some(1))
            .with_max_expansions(1),
    );
    let candidates = collect_flat_fuzzy_candidates(
        text_stream(vec![
            vec![format!("{} catb", "zzzz ".repeat(70_000))],
            vec![format!("{} cata", "yyyy ".repeat(70_000))],
        ]),
        "text",
        simple_text_tokenizer(),
        query_tokens,
        params,
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        candidates
            .get(&0)
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["cata".to_string()],
        "the global cap must retain the lexicographically smallest unique term"
    );
}

#[tokio::test]
async fn flat_fuzzy_stats_are_exact_for_the_final_vocabulary() {
    let final_tokens = Arc::new(Tokens::with_positions(
        vec!["cata".to_string(), "catb".to_string()],
        vec![0, 0],
        DocType::Text,
    ));
    let stats = collect_flat_bm25_stats(
        text_stream(vec![vec![
            "cata cata".to_string(),
            "catb cata".to_string(),
            String::new(),
        ]]),
        "text",
        simple_text_tokenizer(),
        final_tokens,
        None,
    )
    .await
    .unwrap();

    assert_eq!(stats.total_tokens, 4);
    assert_eq!(stats.num_docs(), 2);
    assert_eq!(stats.num_docs_containing_token("cata"), 2);
    assert_eq!(stats.num_docs_containing_token("catb"), 1);
}

#[tokio::test]
async fn flat_json_fuzzy_candidates_keep_path_exact_and_unicode_prefix() {
    let query_tokens = Arc::new(Tokens::new(
        vec!["payload,str,éclait".to_string()],
        DocType::Json,
    ));
    let params = Arc::new(
        FtsSearchParams::new()
            .with_fuzziness(Some(1))
            .with_prefix_length(1)
            .with_max_expansions(1),
    );
    let tokenizer = InvertedIndexParams::default()
        .lance_tokenizer("json".to_string())
        .ascii_folding(false)
        .build()
        .unwrap();
    let candidates = collect_flat_fuzzy_candidates(
        text_stream(vec![vec![
            r#"{"payload":"éclair","other":"éclair"}"#.to_string(),
        ]]),
        "text",
        tokenizer,
        query_tokens,
        params,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        candidates
            .get(&0)
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["payload,str,éclair".to_string()]
    );
}

#[tokio::test]
async fn row_list_raw_tokenization_matches_index_materialization() {
    let mut documents = GenericListBuilder::<i32, _>::new(GenericStringBuilder::<i32>::new());
    documents.values().append_value("alpha");
    documents.values().append_value("beta");
    documents.append(true);
    documents.append(true);
    documents.append(false);
    let documents = Arc::new(documents.finish()) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", documents.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0_u64, 1, 2])) as ArrayRef,
            documents,
        ],
    )
    .unwrap();
    let input = || -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch.clone())]),
        ))
    };
    let raw = || {
        InvertedIndexParams::new("raw".to_string(), Language::English)
            .lower_case(false)
            .stem(false)
            .remove_stop_words(false)
            .build()
            .unwrap()
    };
    let final_tokens = Arc::new(Tokens::new(vec!["alpha beta".to_string()], DocType::Text));

    let stats = collect_flat_bm25_stats(input(), "text", raw(), final_tokens.clone(), None)
        .await
        .unwrap();
    assert_eq!(
        stats.num_docs(),
        2,
        "raw empty lists emit the empty token; null lists remain absent"
    );
    assert_eq!(stats.total_tokens, 2);
    assert_eq!(stats.num_docs_containing_token("alpha beta"), 1);

    let counted = tokenize_and_count(input(), raw(), final_tokens, 1, None, 0, None)
        .await
        .unwrap();
    assert_eq!(counted.num_rows(), 2);
    assert_eq!(
        counted[FLAT_QUERY_TOKEN_COUNTS_COL]
            .as_fixed_size_list()
            .values()
            .as_primitive::<UInt64Type>()
            .values(),
        &[1, 0]
    );

    let candidates = collect_flat_fuzzy_candidates(
        input(),
        "text",
        raw(),
        Arc::new(Tokens::new(vec!["a".to_string()], DocType::Text)),
        Arc::new(
            FtsSearchParams::new()
                .with_fuzziness(Some(1))
                .with_max_expansions(1),
        ),
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        candidates
            .get(&0)
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec![String::new()],
        "raw empty-list vocabulary must participate in fuzzy rewrite"
    );
}

#[tokio::test]
async fn row_large_list_ngram_keeps_cross_element_tokens() {
    let mut documents = GenericListBuilder::<i64, _>::new(GenericStringBuilder::<i64>::new());
    documents.values().append_value("ab");
    documents.values().append_value("cd");
    documents.append(true);
    let documents = Arc::new(documents.finish()) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", documents.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0_u64])) as ArrayRef,
            documents,
        ],
    )
    .unwrap();
    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::iter(vec![Ok(batch)]),
    ));
    let tokenizer = InvertedIndexParams::new("ngram".to_string(), Language::English)
        .ngram_min_length(3)
        .ngram_max_length(3)
        .lower_case(false)
        .stem(false)
        .remove_stop_words(false)
        .build()
        .unwrap();
    let tokens = Arc::new(Tokens::new(vec!["b c".to_string()], DocType::Text));
    let counted = tokenize_and_count(input, tokenizer, tokens, 1, None, 0, None)
        .await
        .unwrap();
    assert_eq!(
        counted[FLAT_QUERY_TOKEN_COUNTS_COL]
            .as_fixed_size_list()
            .values()
            .as_primitive::<UInt64Type>()
            .value(0),
        1,
        "the ngram spanning the inserted space must match the index builder"
    );
}

#[tokio::test]
async fn list_element_raw_null_and_empty_match_builder_tokens() {
    let coordinate_column = doc_index_storage_column(0);
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new(&coordinate_column, DataType::UInt32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![7_u64, 7])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0, 1])) as ArrayRef,
            Arc::new(StringArray::from(vec![None, Some("")])) as ArrayRef,
        ],
    )
    .unwrap();
    let input = || -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch.clone())]),
        ))
    };
    let raw = || {
        InvertedIndexParams::new("raw".to_string(), Language::English)
            .lower_case(false)
            .stem(false)
            .remove_stop_words(false)
            .build()
            .unwrap()
    };
    let tokens = Arc::new(Tokens::new(vec![String::new()], DocType::Text));
    let stats = collect_flat_bm25_stats(input(), "text", raw(), tokens.clone(), None)
        .await
        .unwrap();
    assert_eq!(stats.num_docs(), 2);
    assert_eq!(stats.total_tokens, 2);
    assert_eq!(stats.num_docs_containing_token(""), 2);

    let counted = tokenize_and_count(input(), raw(), tokens, 2, None, 1, None)
        .await
        .unwrap();
    assert_eq!(counted.num_rows(), 2);
    assert_eq!(
        counted[FLAT_QUERY_TOKEN_COUNTS_COL]
            .as_fixed_size_list()
            .values()
            .as_primitive::<UInt64Type>()
            .values(),
        &[1, 1]
    );
}

#[tokio::test]
async fn fixed_flat_scorer_yields_before_input_eof() {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", DataType::Utf8, false),
    ]));
    let (sender, receiver) = futures::channel::mpsc::unbounded();
    let first = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0_u64])) as ArrayRef,
            Arc::new(StringArray::from(vec![format!(
                "needle {}",
                "filler ".repeat(90_000)
            )])) as ArrayRef,
        ],
    )
    .unwrap();
    sender.unbounded_send(Ok(first)).unwrap();
    let input: SendableRecordBatchStream =
        Box::pin(RecordBatchStreamAdapter::new(schema, receiver));
    let scorer = MemBM25Scorer::new(2, 2, HashMap::from([("needle".to_string(), 2)]));
    let stream = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        flat_bm25_search_stream_with_options_and_fixed_scorer(
            input,
            "text".to_string(),
            "needle".to_string(),
            simple_text_tokenizer(),
            scorer,
            FlatBm25SearchOptions {
                target_batch_size: 1,
                elapsed_compute: None,
                document_granularity: DocumentGranularity::Row,
                operator: Operator::Or,
                boost: 1.0,
                phrase_slop: None,
            },
        ),
    )
    .await
    .expect("fixed scorer construction must not wait for input EOF")
    .unwrap();
    futures::pin_mut!(stream);
    let first_output = tokio::time::timeout(std::time::Duration::from_secs(2), stream.try_next())
        .await
        .expect("the first oversized chunk must be scored before input EOF")
        .unwrap()
        .unwrap();
    assert_eq!(
        first_output[ROW_ID].as_primitive::<UInt64Type>().value(0),
        0
    );
    drop(sender);
}

#[tokio::test]
async fn fixed_flat_scorer_does_not_invent_weight_for_missing_term() {
    let scorer = MemBM25Scorer::new(14, 10, HashMap::new());
    let result = flat_bm25_search_stream_with_options_and_fixed_scorer(
        text_stream(vec![vec!["needle".to_string()]]),
        "text".to_string(),
        "needle".to_string(),
        simple_text_tokenizer(),
        scorer,
        FlatBm25SearchOptions {
            target_batch_size: 16,
            elapsed_compute: None,
            document_granularity: DocumentGranularity::Row,
            operator: Operator::Or,
            boost: 1.0,
            phrase_slop: None,
        },
    )
    .await
    .unwrap()
    .try_collect::<Vec<_>>()
    .await
    .unwrap();
    assert!(result.is_empty(), "missing scorer terms have zero weight");
}

#[tokio::test]
async fn flat_bm25_search_stream_with_metrics_records_elapsed_compute() {
    use crate::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
    use arrow_array::{StringArray, UInt64Array};
    use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};

    // Tiny stream of one batch containing the query term in two rows.
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3])),
            Arc::new(StringArray::from(vec![
                "the quick brown fox",
                "lazy dog sleeps",
                "the brown fox jumps over",
                "completely unrelated text",
            ])),
        ],
    )
    .unwrap();

    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(vec![Ok(batch)]),
    ));

    let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ));

    let elapsed_compute = Time::default();
    let result_stream = flat_bm25_search_stream_with_metrics(
        input,
        "text".to_string(),
        "fox".to_string(),
        tokenizer,
        None,
        100,
        Some(elapsed_compute.clone()),
    )
    .await
    .unwrap();

    let batches: Vec<_> = result_stream.try_collect().await.unwrap();
    assert!(!batches.is_empty(), "expected at least one scored batch");

    // Both phase 1 (tokenize_and_count's spawn_cpu) and phase 2 (sync
    // scoring) call `add_duration` on the metric; verify the handle
    // was actually populated.
    assert!(
        elapsed_compute.value() > 0,
        "elapsed_compute should have been populated; got 0"
    );
}

#[tokio::test]
async fn flat_bm25_phrase_honors_positions_slop_and_repeated_terms() {
    async fn search(query: &str, slop: u32) -> Vec<u64> {
        let schema = Arc::new(Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from_iter_values(0..5)),
                Arc::new(StringArray::from(vec![
                    "alpha beta",
                    "alpha gap beta",
                    "alpha gap gap beta",
                    "alpha alpha beta",
                    "beta alpha",
                ])),
            ],
        )
        .unwrap();
        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch)]),
        ));
        let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default()).build(),
        ));
        let (stream, _) = flat_bm25_search_stream_with_options_and_scorer(
            input,
            "text".to_string(),
            query.to_string(),
            tokenizer,
            None,
            FlatBm25SearchOptions {
                target_batch_size: 100,
                elapsed_compute: None,
                document_granularity: DocumentGranularity::Row,
                operator: Operator::And,
                boost: 1.0,
                phrase_slop: Some(slop),
            },
        )
        .await
        .unwrap();
        stream
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .iter()
            .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values())
            .copied()
            .collect()
    }

    assert_eq!(search("alpha beta", 0).await, vec![0, 3]);
    assert_eq!(search("alpha beta", 1).await, vec![0, 1, 3]);
    assert_eq!(search("alpha alpha", 0).await, vec![3]);
}

#[test]
fn flat_full_text_search_supports_phrase_queries() {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from_iter_values(0..4)),
            Arc::new(StringArray::from(vec![
                "alpha beta",
                "alpha gap beta",
                "alpha alpha beta",
                "beta alpha",
            ])),
        ],
    )
    .unwrap();

    assert_eq!(
        flat_full_text_search(&[&batch], "text", "\"alpha beta\"", None).unwrap(),
        vec![0, 2]
    );
    assert_eq!(
        flat_full_text_search(&[&batch], "text", "\"alpha alpha\"", None).unwrap(),
        vec![2]
    );
    assert!(!is_phrase_query("\""));
}

#[tokio::test]
async fn flat_bm25_skips_zero_token_documents_from_corpus_stats() {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![0_u64, 1, 2, 3, 4, 5])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some(""),
                Some("   "),
                Some("the"),
                Some("overlength"),
                None,
                Some("hello"),
            ])) as ArrayRef,
        ],
    )
    .unwrap();
    let params = InvertedIndexParams::new("whitespace".to_string(), Language::English)
        .remove_stop_words(true)
        .stem(false)
        .max_token_length(Some(6));
    let query_tokens = Arc::new(Tokens::new(vec!["hello".to_string()], DocType::Text));

    let counted_input = tokenize_and_count(
        stream::iter(vec![Ok(batch)]),
        params.build().unwrap(),
        query_tokens.clone(),
        1,
        None,
        0,
        None,
    )
    .await
    .unwrap();

    assert_eq!(counted_input.num_rows(), 1);
    assert_eq!(
        counted_input[ROW_ID].as_primitive::<UInt64Type>().values(),
        &[5]
    );
    let scorer = initialize_scorer(None, query_tokens.as_ref(), &counted_input);
    let expected_scorer = MemBM25Scorer::new(1, 1, HashMap::from([("hello".to_string(), 1)]));
    assert_eq!(scorer.total_tokens, 1);
    assert_eq!(scorer.num_docs(), 1);
    assert_eq!(scorer.num_docs_containing_token("hello"), 1);
    assert_eq!(scorer.avg_doc_length(), expected_scorer.avg_doc_length());
    assert_eq!(
        scorer.query_weight("hello"),
        expected_scorer.query_weight("hello")
    );
}

#[tokio::test]
async fn flat_bm25_preserves_zero_token_list_element_documents() {
    let coordinate_column = doc_index_storage_column(0);
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new(&coordinate_column, DataType::UInt32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![7_u64; 6])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3, 4, 5])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                None,
                Some(""),
                Some("   "),
                Some("the"),
                Some("overlength"),
                Some("hello"),
            ])) as ArrayRef,
        ],
    )
    .unwrap();
    let params = InvertedIndexParams::new("whitespace".to_string(), Language::English)
        .remove_stop_words(true)
        .stem(false)
        .max_token_length(Some(6));
    let query_tokens = Arc::new(Tokens::new(vec!["hello".to_string()], DocType::Text));

    let counted_input = tokenize_and_count(
        stream::iter(vec![Ok(batch)]),
        params.build().unwrap(),
        query_tokens.clone(),
        2,
        None,
        1,
        None,
    )
    .await
    .unwrap();

    assert_eq!(counted_input.num_rows(), 6);
    assert_eq!(
        counted_input[&coordinate_column]
            .as_primitive::<UInt32Type>()
            .values(),
        &[0, 1, 2, 3, 4, 5]
    );
    let scorer = initialize_scorer(None, query_tokens.as_ref(), &counted_input);
    assert_eq!(scorer.total_tokens, 1);
    assert_eq!(scorer.num_docs(), 6);
    assert_eq!(scorer.num_docs_containing_token("hello"), 1);
}

#[tokio::test]
async fn flat_bm25_search_uses_full_document_length_for_normalization() {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1])),
            Arc::new(StringArray::from(vec![
                "alpha",
                "alpha filler filler filler filler filler filler filler filler filler",
            ])),
        ],
    )
    .unwrap();

    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(vec![Ok(batch)]),
    ));
    let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ));

    let result_stream = flat_bm25_search_stream_with_metrics(
        input,
        "text".to_string(),
        "alpha".to_string(),
        tokenizer,
        None,
        100,
        None,
    )
    .await
    .unwrap();
    let batches: Vec<_> = result_stream.try_collect().await.unwrap();
    let scored = arrow::compute::concat_batches(&FTS_SCHEMA, &batches).unwrap();
    let row_ids = scored[ROW_ID].as_primitive::<UInt64Type>();
    let scores = scored[SCORE_COL].as_primitive::<Float32Type>();

    assert_eq!(row_ids.values(), &[0, 1]);
    assert!(
        scores.value(0) > scores.value(1),
        "same term frequency should score shorter document higher; short={}, long={}",
        scores.value(0),
        scores.value(1)
    );
}

#[tokio::test]
async fn flat_bm25_search_treats_string_lists_as_row_documents() {
    let mut docs_builder = GenericListBuilder::<i32, _>::new(GenericStringBuilder::<i32>::new());
    docs_builder.values().append_value("alpha");
    docs_builder.values().append_value("alpha beta");
    docs_builder.append(true);
    docs_builder.values().append_value("beta");
    docs_builder.append(true);
    docs_builder.append(true);
    docs_builder.values().append_null();
    docs_builder.append(true);
    docs_builder.append(false);

    let docs = Arc::new(docs_builder.finish()) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", docs.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3, 4])) as ArrayRef,
            docs,
        ],
    )
    .unwrap();

    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(vec![Ok(batch)]),
    ));
    let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ));

    let result_stream = flat_bm25_search_stream_with_metrics(
        input,
        "text".to_string(),
        "alpha".to_string(),
        tokenizer,
        None,
        100,
        None,
    )
    .await
    .unwrap();
    let batches: Vec<_> = result_stream.try_collect().await.unwrap();
    let scored = arrow::compute::concat_batches(&FTS_SCHEMA, &batches).unwrap();
    let row_ids = scored[ROW_ID].as_primitive::<UInt64Type>();

    assert_eq!(row_ids.values(), &[0]);
}

#[tokio::test]
async fn flat_bm25_search_code_and_uses_position_groups() {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("code", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3])),
            Arc::new(StringArray::from(vec![
                "get user name",
                "getUserName",
                "get user",
                "username",
            ])),
        ],
    )
    .unwrap();

    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(vec![Ok(batch)]),
    ));
    let tokenizer = InvertedIndexParams::code()
        .split_identifiers(true)
        .build()
        .unwrap();

    let result_stream = flat_bm25_search_stream_with_metrics_and_operator(
        input,
        "code".to_string(),
        "getUserName".to_string(),
        tokenizer,
        None,
        100,
        Operator::And,
        None,
    )
    .await
    .unwrap();

    let batches: Vec<_> = result_stream.try_collect().await.unwrap();
    let scored = arrow::compute::concat_batches(&FTS_SCHEMA, &batches).unwrap();
    let mut row_ids = scored[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    row_ids.sort_unstable();

    assert_eq!(row_ids, vec![0, 1]);
}

#[tokio::test]
async fn flat_bm25_search_code_and_counts_repeated_subwords() {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("code", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![0u64, 1])),
                Arc::new(StringArray::from(vec![
                    "pub fn edge_flat_generic_return<T>() -> Result<T, EdgeFlatError> where T: TryFrom<String> { todo!() }",
                    "pub fn edge_flat_generic_return<T>() -> Result<T> { todo!() }",
                ])),
            ],
        )
        .unwrap();

    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(vec![Ok(batch)]),
    ));
    let tokenizer = InvertedIndexParams::code().build().unwrap();

    let result_stream = flat_bm25_search_stream_with_metrics_and_operator(
        input,
        "code".to_string(),
        "edge_flat_generic_return TryFrom EdgeFlatError Result".to_string(),
        tokenizer,
        None,
        100,
        Operator::And,
        None,
    )
    .await
    .unwrap();

    let batches: Vec<_> = result_stream.try_collect().await.unwrap();
    let scored = arrow::compute::concat_batches(&FTS_SCHEMA, &batches).unwrap();
    let row_ids = scored[ROW_ID].as_primitive::<UInt64Type>().values();

    assert_eq!(row_ids, &[0]);
}
