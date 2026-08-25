// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

pub fn doc_index_storage_column(rank: usize) -> String {
    format!("{DOC_INDEX_STORAGE_PREFIX}{rank}")
}

pub fn document_coordinate_rank(schema: &arrow_schema::Schema) -> usize {
    (0..)
        .take_while(|rank| {
            schema
                .column_with_name(&doc_index_storage_column(*rank))
                .is_some()
        })
        .count()
}

pub fn flat_full_text_search(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<Box<dyn LanceTokenizer>>,
) -> Result<Vec<u64>> {
    if batches.is_empty() {
        return Ok(vec![]);
    }

    let (query, phrase_slop) = match phrase_query_text(query) {
        Some(query) => (query, Some(0)),
        None => (query, None),
    };

    match batches[0][doc_col].data_type() {
        DataType::Utf8 => {
            do_flat_full_text_search::<i32>(batches, doc_col, query, tokenizer, phrase_slop)
        }
        DataType::LargeUtf8 => {
            do_flat_full_text_search::<i64>(batches, doc_col, query, tokenizer, phrase_slop)
        }
        DataType::List(_) => {
            do_flat_full_text_search_list::<i32>(batches, doc_col, query, tokenizer, phrase_slop)
        }
        DataType::LargeList(_) => {
            do_flat_full_text_search_list::<i64>(batches, doc_col, query, tokenizer, phrase_slop)
        }
        data_type => Err(Error::invalid_input(format!(
            "unsupported data type {} for inverted index",
            data_type
        ))),
    }
}

pub(super) fn do_flat_full_text_search<Offset: OffsetSizeTrait>(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<Box<dyn LanceTokenizer>>,
    phrase_slop: Option<u32>,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let mut tokenizer =
        tokenizer.unwrap_or_else(|| InvertedIndexParams::default().build().unwrap());
    let query_tokens = collect_query_tokens(query, &mut tokenizer);

    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_string::<Offset>();
        for i in 0..row_id_array.len() {
            let doc = doc_array.value(i);
            if document_matches_flat_query(doc, &mut tokenizer, &query_tokens, phrase_slop)? {
                results.push(row_id_array.value(i));
            }
        }
    }

    Ok(results)
}

pub(super) fn do_flat_full_text_search_list<ListOffset: OffsetSizeTrait>(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<Box<dyn LanceTokenizer>>,
    phrase_slop: Option<u32>,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let mut tokenizer =
        tokenizer.unwrap_or_else(|| InvertedIndexParams::default().build().unwrap());
    let query_tokens = collect_query_tokens(query, &mut tokenizer);

    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_list::<ListOffset>();
        match doc_array.value_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {}
            data_type => {
                return Err(Error::invalid_input(format!(
                    "unsupported list item data type {} for inverted index",
                    data_type
                )));
            }
        }
        for i in 0..row_id_array.len() {
            if doc_array.is_null(i) {
                continue;
            }
            let elements = doc_array.value(i);
            let matches = if phrase_slop.is_some() {
                let document = iter_str_array(elements.as_ref())
                    .flatten()
                    .collect::<Vec<_>>()
                    .join(" ");
                document_matches_flat_query(&document, &mut tokenizer, &query_tokens, phrase_slop)?
            } else {
                iter_str_array(elements.as_ref())
                    .flatten()
                    .any(|element| has_query_token(element, &mut tokenizer, &query_tokens))
            };
            if matches {
                results.push(row_id_array.value(i));
            }
        }
    }

    Ok(results)
}

pub(super) fn document_matches_flat_query(
    document: &str,
    tokenizer: &mut Box<dyn LanceTokenizer>,
    query_tokens: &Tokens,
    phrase_slop: Option<u32>,
) -> Result<bool> {
    let Some(slop) = phrase_slop else {
        return Ok(has_query_token(document, tokenizer, query_tokens));
    };

    let mut document_positions = (0..query_tokens.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let mut stream = tokenizer.token_stream_for_doc(document);
    while let Some(token) = stream.next() {
        let position = u32::try_from(token.position).map_err(|_| {
            Error::invalid_input(format!(
                "flat FTS token position exceeds u32: {}",
                token.position
            ))
        })?;
        for (query_index, positions) in document_positions.iter_mut().enumerate() {
            if query_tokens.get_token(query_index) == token.text {
                positions.push(position);
            }
        }
    }
    Ok(phrase_matches_positions(
        query_tokens,
        &document_positions,
        slop,
    ))
}

pub(super) const FLAT_ALL_TOKENS_COL: &str = "all_tokens";
pub(super) const FLAT_QUERY_TOKEN_COUNTS_COL: &str = "query_token_counts";
pub(super) const FLAT_PHRASE_MATCH_COL: &str = "phrase_match";

pub(super) fn phrase_matches_positions(
    query_tokens: &Tokens,
    document_positions: &[Vec<u32>],
    slop: u32,
) -> bool {
    let Some(first_positions) = document_positions.first() else {
        return false;
    };
    if first_positions.is_empty() {
        return false;
    }

    let mut candidates = first_positions.clone();
    debug_assert_eq!(query_tokens.len(), document_positions.len());
    for (query_index, positions) in document_positions.iter().enumerate().skip(1) {
        let Some(query_delta) = query_tokens
            .position(query_index)
            .checked_sub(query_tokens.position(query_index - 1))
        else {
            return false;
        };
        let mut next_candidates = Vec::new();
        for &position in positions {
            let position = u64::from(position);
            if candidates.iter().any(|candidate| {
                let least = u64::from(*candidate) + u64::from(query_delta);
                least <= position && position <= least + u64::from(slop)
            }) {
                next_candidates.push(position as u32);
            }
        }
        if next_candidates.is_empty() {
            return false;
        }
        candidates = next_candidates;
    }
    true
}

/// If we accumulate this many bytes we warn the user they probably want to use an FTS index instead.
pub(super) const BYTES_ACCUMULATED_WARNING_THRESHOLD: u64 = 1024 * 1024 * 1024; // 1GB

/// Consumes a stream of record batches and produces token counts
///
/// The resulting batch will have three columns:
/// - row_id: the row id of the document
/// - all_tokens: the total number of tokens in the document
/// - query_token_counts: a fixed size list of the count of each query token in the document
///
/// This is an unbounded accumulation, however, for most queries, the per-row
/// growth will be fairly small.  As a result we can process millions of tokens
/// with fairly modest memory usage.
///
/// However, it is unwise to do a flat search across billions of rows.  An FTS
/// index should be created instead.
pub(super) async fn tokenize_and_count(
    input: impl Stream<Item = DataFusionResult<RecordBatch>> + Send,
    tokenizer: Box<dyn LanceTokenizer>,
    query_tokens: Arc<Tokens>,
    doc_col_idx: usize,
    elapsed_compute: Option<Time>,
    coordinate_rank: usize,
    phrase_slop: Option<u32>,
) -> DataFusionResult<RecordBatch> {
    let mut output_fields = vec![ROW_ID_FIELD.clone()];
    output_fields.extend(
        (0..coordinate_rank)
            .map(|rank| Field::new(doc_index_storage_column(rank), DataType::UInt32, false)),
    );
    output_fields.extend([
        Field::new(FLAT_ALL_TOKENS_COL, DataType::UInt64, false),
        Field::new(
            FLAT_QUERY_TOKEN_COUNTS_COL,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt64, true)),
                query_tokens.len() as i32,
            ),
            false,
        ),
    ]);
    if phrase_slop.is_some() {
        output_fields.push(Field::new(FLAT_PHRASE_MATCH_COL, DataType::Boolean, false));
    }
    let output_schema = Arc::new(Schema::new(output_fields));
    let output_schema_clone = output_schema.clone();
    let query_token_indices = Arc::new(query_token_indices(query_tokens.as_ref()));
    let bytes_accumulated = Arc::new(AtomicU64::new(0));
    let bytes_warning_emitted = Arc::new(AtomicBool::new(false));

    let batches = input
        .map(move |batch| {
            let mut tokenizer = tokenizer.box_clone();
            let output_schema = output_schema.clone();
            let query_tokens = query_tokens.clone();
            let query_token_indices = query_token_indices.clone();
            let bytes_accumulated = bytes_accumulated.clone();
            let bytes_warning_emitted = bytes_warning_emitted.clone();
            let elapsed_compute = elapsed_compute.clone();
            spawn_cpu(move || {
                // Time the per-batch CPU work so callers can attribute it to
                // `elapsed_compute` on a metric handle (the spawn_cpu worker
                // thread is invisible to the caller's poll timer otherwise).
                let start = std::time::Instant::now();
                let batch = batch?;
                let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
                let input_doc_indices = (0..coordinate_rank)
                    .map(|rank| {
                        let column_name = doc_index_storage_column(rank);
                        batch
                            .column_by_name(&column_name)
                            .ok_or_else(|| {
                                datafusion_common::DataFusionError::Internal(format!(
                                    "flat ListElement FTS is missing {column_name}"
                                ))
                            })
                            .map(|column| column.as_primitive::<UInt32Type>())
                    })
                    .collect::<DataFusionResult<Vec<_>>>()?;
                let mut row_ids = UInt64Builder::with_capacity(batch.num_rows());
                let mut doc_indices = (0..coordinate_rank)
                    .map(|_| UInt32Builder::with_capacity(batch.num_rows()))
                    .collect::<Vec<_>>();
                let mut all_token_counts = UInt64Builder::with_capacity(batch.num_rows());
                let mut query_token_counts = FixedSizeListBuilder::with_capacity(
                    UInt64Builder::with_capacity(batch.num_rows() * query_tokens.len()),
                    query_tokens.len() as i32,
                    batch.num_rows(),
                );
                let mut temp_query_token_counts = Vec::with_capacity(query_tokens.len());
                let mut temp_query_positions = (0..query_tokens.len())
                    .map(|_| Vec::new())
                    .collect::<Vec<_>>();
                let mut phrase_matches = phrase_slop
                    .map(|_| BooleanBuilder::with_capacity(batch.num_rows()));
                let mut count_text = |doc: &str,
                                      temp_query_token_counts: &mut Vec<u64>|
                 -> DataFusionResult<(u64, bool)> {
                    for positions in &mut temp_query_positions {
                        positions.clear();
                    }
                    let mut stream = tokenizer.token_stream_for_doc(doc);
                    let mut all_tokens = 0;
                    while let Some(token) = stream.next() {
                        all_tokens += 1;
                        if let Some(token_indices) = query_token_indices.get(&token.text) {
                            for token_index in token_indices {
                                temp_query_token_counts[*token_index] += 1;
                                if phrase_slop.is_some() {
                                    temp_query_positions[*token_index].push(
                                        u32::try_from(token.position).map_err(|_| {
                                            datafusion_common::DataFusionError::Execution(format!(
                                                "flat FTS token position exceeds u32: {}",
                                                token.position
                                            ))
                                        })?,
                                    );
                                }
                            }
                        }
                    }
                    let matches = phrase_slop.is_none_or(|slop| {
                        phrase_matches_positions(
                            query_tokens.as_ref(),
                            &temp_query_positions,
                            slop,
                        )
                    });
                    Ok((all_tokens, matches))
                };
                let mut append_counts = |row_index: usize,
                                         row_id: u64,
                                         all_tokens: u64,
                                         temp_query_token_counts: &[u64],
                                         phrase_match: bool|
                 -> DataFusionResult<()> {
                        row_ids.append_value(row_id);
                        for (builder, input) in
                            doc_indices.iter_mut().zip(input_doc_indices.iter())
                        {
                            builder.append_value(input.value(row_index));
                        }
                        all_token_counts.append_value(all_tokens);
                        for count in temp_query_token_counts.iter().copied() {
                            query_token_counts.values().append_value(count);
                        }
                        query_token_counts.append(true);
                        if let Some(builder) = phrase_matches.as_mut() {
                            builder.append_value(phrase_match);
                        }
                        Ok(())
                    };
                match batch.column(doc_col_idx).data_type() {
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                        let doc_iter = iter_str_array(batch.column(doc_col_idx));
                        for (row_index, (doc, row_id)) in
                            doc_iter.zip(row_id_array.values().iter()).enumerate()
                        {
                            temp_query_token_counts.clear();
                            temp_query_token_counts
                                .extend(std::iter::repeat_n(0, query_tokens.len()));

                            let (all_tokens, phrase_match) = match doc {
                                Some(doc) => count_text(doc, &mut temp_query_token_counts)?,
                                None if coordinate_rank > 0 => {
                                    count_text("", &mut temp_query_token_counts)?
                                }
                                None => (0, false),
                            };
                            if coordinate_rank > 0 || all_tokens > 0 {
                                append_counts(
                                    row_index,
                                    *row_id,
                                    all_tokens,
                                    &temp_query_token_counts,
                                    phrase_match,
                                )?;
                            }
                        }
                    }
                    DataType::List(_) => {
                        if coordinate_rank != 0 {
                            return DataFusionResult::Err(
                                datafusion_common::DataFusionError::Internal(
                                    "ListElement flat FTS input must be expanded to string documents"
                                        .to_string(),
                                ),
                            );
                        }
                        tokenize_and_count_list::<i32>(
                            batch.column(doc_col_idx),
                            row_id_array,
                            &mut count_text,
                            &mut append_counts,
                            &mut temp_query_token_counts,
                            query_tokens.len(),
                            phrase_slop.is_some(),
                        )?;
                    }
                    DataType::LargeList(_) => {
                        if coordinate_rank != 0 {
                            return DataFusionResult::Err(
                                datafusion_common::DataFusionError::Internal(
                                    "ListElement flat FTS input must be expanded to string documents"
                                        .to_string(),
                                ),
                            );
                        }
                        tokenize_and_count_list::<i64>(
                            batch.column(doc_col_idx),
                            row_id_array,
                            &mut count_text,
                            &mut append_counts,
                            &mut temp_query_token_counts,
                            query_tokens.len(),
                            phrase_slop.is_some(),
                        )?;
                    }
                    data_type => {
                        return DataFusionResult::Err(datafusion_common::DataFusionError::Execution(
                            format!("unsupported data type {} for flat full text search", data_type),
                        ));
                    }
                }
                let row_ids = row_ids.finish();
                let doc_indices = doc_indices
                    .into_iter()
                    .map(|mut builder| builder.finish())
                    .collect::<Vec<_>>();
                let all_token_counts = all_token_counts.finish();
                let query_token_counts = query_token_counts.finish();
                let mut columns = vec![Arc::new(row_ids) as ArrayRef];
                for doc_indices in doc_indices {
                    columns.push(Arc::new(doc_indices) as ArrayRef);
                }
                columns.extend([
                    Arc::new(all_token_counts) as ArrayRef,
                    Arc::new(query_token_counts) as ArrayRef,
                ]);
                if let Some(mut builder) = phrase_matches {
                    columns.push(Arc::new(builder.finish()) as ArrayRef);
                }
                let result_batch = RecordBatch::try_new(output_schema, columns)?;
                let bytes_accumulated = bytes_accumulated.fetch_add(result_batch.get_array_memory_size() as u64, Ordering::Relaxed);
                if bytes_accumulated > BYTES_ACCUMULATED_WARNING_THRESHOLD && !bytes_warning_emitted.swap(true, Ordering::Relaxed) {
                    tracing::warn!("Flat full text search is accumulating a large number of bytes.  Consider using an FTS index instead.");
                }

                if let Some(t) = &elapsed_compute {
                    t.add_duration(start.elapsed());
                }
                DataFusionResult::Ok(result_batch)
            })
        })
        .buffered(get_num_compute_intensive_cpus())
        .try_collect::<Vec<_>>()
        .await?;

    Ok(arrow::compute::concat_batches(
        &output_schema_clone,
        &batches,
    )?)
}

pub(super) fn tokenize_and_count_list<ListOffset: OffsetSizeTrait>(
    doc_col: &ArrayRef,
    row_id_array: &arrow_array::PrimitiveArray<UInt64Type>,
    count_text: &mut impl FnMut(&str, &mut Vec<u64>) -> DataFusionResult<(u64, bool)>,
    append_counts: &mut impl FnMut(usize, u64, u64, &[u64], bool) -> DataFusionResult<()>,
    temp_query_token_counts: &mut Vec<u64>,
    query_tokens_len: usize,
    _match_phrase: bool,
) -> DataFusionResult<()> {
    let doc_array = doc_col.as_list::<ListOffset>();
    match doc_array.value_type() {
        DataType::Utf8 | DataType::LargeUtf8 => {}
        data_type => {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "unsupported list item data type {} for flat full text search",
                data_type
            )));
        }
    }

    for i in 0..row_id_array.len() {
        temp_query_token_counts.clear();
        temp_query_token_counts.extend(std::iter::repeat_n(0, query_tokens_len));
        let mut all_tokens = 0;
        let mut phrase_match = false;
        if !doc_array.is_null(i) {
            let elements = doc_array.value(i);
            // Row-level list documents use the same materialization boundary
            // as the index builder: non-null elements are joined with one
            // ASCII space and tokenized once. This matters for raw/ngram
            // analyzers whose tokens can span an element boundary. ListElement
            // input is expanded upstream and never reaches this branch.
            let mut document = String::new();
            for element in iter_str_array(elements.as_ref()).flatten() {
                if !document.is_empty() {
                    document.push(' ');
                }
                document.push_str(element);
            }
            (all_tokens, phrase_match) = count_text(&document, temp_query_token_counts)?;
        }
        if all_tokens > 0 {
            append_counts(
                i,
                row_id_array.value(i),
                all_tokens,
                temp_query_token_counts,
                phrase_match,
            )?;
        }
    }

    Ok(())
}

pub(super) fn query_token_indices(query_tokens: &Tokens) -> HashMap<String, Vec<usize>> {
    let mut indices = HashMap::new();
    for idx in 0..query_tokens.len() {
        indices
            .entry(query_tokens.get_token(idx).to_string())
            .or_insert_with(Vec::new)
            .push(idx);
    }
    indices
}

/// Initialize the BM25 scorer
///
/// In order to calculate BM25 scores we need to know token counts for the entire corpus.  We extract these from the
/// counted input of the flat search combined with any counts recorded for the indexed portion.
pub(super) fn initialize_scorer(
    base_scorer: Option<&MemBM25Scorer>,
    query_tokens: &Tokens,
    counted_input: &RecordBatch,
) -> MemBM25Scorer {
    let mut total_tokens = 0;
    let mut num_docs = 0;
    let mut all_token_counts = vec![0; query_tokens.len()];

    if let Some(base_scorer) = base_scorer {
        total_tokens += base_scorer.total_tokens;
        num_docs += base_scorer.num_docs;
        for (token_index, token) in query_tokens.into_iter().enumerate() {
            all_token_counts[token_index] = base_scorer.num_docs_containing_token(token) as u64;
        }
    }

    num_docs += counted_input.num_rows();
    total_tokens +=
        arrow::compute::sum(counted_input[FLAT_ALL_TOKENS_COL].as_primitive::<UInt64Type>())
            .unwrap_or_default();

    let mut input_token_counters = counted_input[FLAT_QUERY_TOKEN_COUNTS_COL]
        .as_fixed_size_list()
        .values()
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();

    for _ in 0..counted_input.num_rows() {
        for token_count in all_token_counts.iter_mut() {
            if input_token_counters.next().unwrap_or_default() > 0 {
                *token_count += 1;
            }
        }
    }

    let token_counts_map = all_token_counts
        .into_iter()
        .enumerate()
        .map(|(token_index, count)| {
            (
                query_tokens.get_token(token_index).to_string(),
                count as usize,
            )
        })
        .collect::<HashMap<String, usize>>();
    MemBM25Scorer::new(total_tokens, num_docs, token_counts_map)
}

pub(super) fn flat_bm25_score(
    query_tokens: &Tokens,
    counted_input: &RecordBatch,
    scorer: &MemBM25Scorer,
    document_granularity: DocumentGranularity,
    operator: Operator,
    boost: f32,
    phrase_slop: Option<u32>,
) -> Result<RecordBatch> {
    let mut row_ids_builder = UInt64Builder::with_capacity(counted_input.num_rows());
    let mut scores_builder = Float32Builder::with_capacity(counted_input.num_rows());
    let coordinate_rank = document_coordinate_rank(counted_input.schema().as_ref());
    let input_doc_indices = (0..coordinate_rank)
        .map(|rank| {
            let coordinate_column = doc_index_storage_column(rank);
            counted_input
                .column_by_name(&coordinate_column)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "flat ListElement FTS is missing {coordinate_column}"
                    ))
                })
                .map(|column| column.as_primitive::<UInt32Type>())
        })
        .collect::<Result<Vec<_>>>()?;
    if document_granularity.is_list_element() != (coordinate_rank > 0) {
        return Err(Error::internal(
            "flat FTS document granularity does not match its coordinate columns".to_string(),
        ));
    }
    let mut doc_indices_builder = document_granularity.is_list_element().then(|| {
        ListBuilder::new(UInt32Builder::new()).with_field(Field::new(
            "item",
            DataType::UInt32,
            false,
        ))
    });
    let query_groups = query_position_groups(query_tokens);

    let mut row_ids_iter = counted_input[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    let mut all_token_counts_iter = counted_input[FLAT_ALL_TOKENS_COL]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    let mut query_token_counts_iter = counted_input[FLAT_QUERY_TOKEN_COUNTS_COL]
        .as_fixed_size_list()
        .values()
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    let phrase_matches = phrase_slop
        .map(|_| {
            counted_input
                .column_by_name(FLAT_PHRASE_MATCH_COL)
                .ok_or_else(|| Error::internal("flat phrase FTS is missing phrase matches"))
                .and_then(|column| {
                    column
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            Error::internal("flat phrase matches must be a Boolean column")
                        })
                })
        })
        .transpose()?;
    for input_index in 0..counted_input.num_rows() {
        let num_tokens_in_doc = all_token_counts_iter.next().expect_ok()?;
        let row_id = row_ids_iter.next().expect_ok()?;
        let mut query_token_counts = Vec::with_capacity(query_tokens.len());
        for _ in query_tokens {
            query_token_counts.push(query_token_counts_iter.next().expect_ok()?);
        }
        if num_tokens_in_doc == 0 {
            continue;
        }
        if operator == Operator::And
            && !query_groups
                .iter()
                .all(|group| group.iter().any(|idx| query_token_counts[*idx] > 0))
        {
            continue;
        }
        if phrase_matches.is_some_and(|matches| !matches.value(input_index)) {
            continue;
        }
        let doc_norm = K1 * (1.0 - B + B * num_tokens_in_doc as f32 / scorer.avg_doc_length());
        let mut score = 0.0;
        for (token, freq) in query_tokens.into_iter().zip(query_token_counts) {
            let freq = freq as f32;
            let idf = idf(scorer.num_docs_containing_token(token), scorer.num_docs());
            score += idf * (freq * (K1 + 1.0) / (freq + doc_norm));
        }
        if score > 0.0 {
            row_ids_builder.append_value(row_id);
            if let Some(builder) = doc_indices_builder.as_mut() {
                for input_doc_index in &input_doc_indices {
                    builder
                        .values()
                        .append_value(input_doc_index.value(input_index));
                }
                builder.append(true);
            }
            scores_builder.append_value(score * boost);
        }
    }

    let row_ids = row_ids_builder.finish();
    let scores = scores_builder.finish();
    let mut columns = vec![Arc::new(row_ids) as ArrayRef];
    if let Some(mut builder) = doc_indices_builder {
        columns.push(Arc::new(builder.finish()) as ArrayRef);
    }
    columns.push(Arc::new(scores) as ArrayRef);
    let batch = RecordBatch::try_new(fts_schema(document_granularity), columns)?;
    Ok(batch)
}

pub(super) fn query_position_groups(query_tokens: &Tokens) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut current_position = None;
    for idx in 0..query_tokens.len() {
        let position = query_tokens.position(idx);
        if current_position != Some(position) {
            current_position = Some(position);
            groups.push(Vec::new());
        }
        groups
            .last_mut()
            .expect("a group should exist after pushing for position")
            .push(idx);
    }
    groups
}

#[deprecated(
    note = "use `flat_bm25_search_stream_with_metrics` to record CPU compute \
            time on a metric handle; pass `None` for the old behavior"
)]
pub async fn flat_bm25_search_stream(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    base_scorer: Option<MemBM25Scorer>,
    target_batch_size: usize,
) -> DataFusionResult<SendableRecordBatchStream> {
    flat_bm25_search_stream_with_metrics(
        input,
        doc_col,
        query,
        tokenizer,
        base_scorer,
        target_batch_size,
        None,
    )
    .await
}

/// Same as [`flat_bm25_search_stream`] but accepts an optional `Time` handle
/// that, if provided, will receive the CPU time spent in (a) per-batch
/// tokenization on the `spawn_cpu` worker threads and (b) the synchronous
/// scoring phase. This lets a calling `ExecutionPlan` report accurate
/// `elapsed_compute` without double-counting upstream poll time.
pub async fn flat_bm25_search_stream_with_metrics(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    base_scorer: Option<MemBM25Scorer>,
    target_batch_size: usize,
    elapsed_compute: Option<Time>,
) -> DataFusionResult<SendableRecordBatchStream> {
    flat_bm25_search_stream_with_metrics_and_operator(
        input,
        doc_col,
        query,
        tokenizer,
        base_scorer,
        target_batch_size,
        Operator::Or,
        elapsed_compute,
    )
    .await
}

/// Same as [`flat_bm25_search_stream_with_metrics`] but applies the provided
/// match operator when deciding whether a flat-scanned row is a hit.
///
/// # Examples
///
/// ```no_run
/// # async fn example(
/// #     input: datafusion::execution::SendableRecordBatchStream,
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// use lance_index::scalar::inverted::{
///     flat_bm25_search_stream_with_metrics_and_operator, query::Operator, InvertedIndexParams,
/// };
///
/// let tokenizer = InvertedIndexParams::code().build()?;
/// let _stream = flat_bm25_search_stream_with_metrics_and_operator(
///     input,
///     "code".to_string(),
///     "Result".to_string(),
///     tokenizer,
///     None,
///     1024,
///     Operator::And,
///     None,
/// )
/// .await?;
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub async fn flat_bm25_search_stream_with_metrics_and_operator(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    base_scorer: Option<MemBM25Scorer>,
    target_batch_size: usize,
    operator: Operator,
    elapsed_compute: Option<Time>,
) -> DataFusionResult<SendableRecordBatchStream> {
    Ok(flat_bm25_search_stream_with_options_and_scorer(
        input,
        doc_col,
        query,
        tokenizer,
        base_scorer,
        FlatBm25SearchOptions {
            target_batch_size,
            elapsed_compute,
            document_granularity: DocumentGranularity::Row,
            operator,
            boost: 1.0,
            phrase_slop: None,
        },
    )
    .await?
    .0)
}

/// Options for flat BM25 search execution.
pub struct FlatBm25SearchOptions {
    /// Maximum output record batch size.
    pub target_batch_size: usize,
    /// Optional DataFusion metric tracking compute time.
    pub elapsed_compute: Option<Time>,
    /// Logical FTS document boundary.
    pub document_granularity: DocumentGranularity,
    /// Match operator applied within each logical document.
    pub operator: Operator,
    /// Score multiplier for the match query.
    pub boost: f32,
    /// Phrase slop. When set, documents must contain the query tokens in order.
    pub phrase_slop: Option<u32>,
}

fn collect_fuzzy_candidates_from_terms(
    terms: &BTreeSet<String>,
    query_tokens: &Tokens,
    params: &FtsSearchParams,
) -> Result<BTreeMap<u32, BTreeSet<String>>> {
    fn merge_stream<S>(
        stream: &mut S,
        candidates: &mut BTreeSet<String>,
        limit: usize,
    ) -> Result<()>
    where
        for<'a> S: fst::Streamer<'a, Item = &'a [u8]>,
    {
        while let Some(term) = stream.next() {
            let term = std::str::from_utf8(term).map_err(|error| {
                Error::index(format!(
                    "flat FTS vocabulary contains invalid UTF-8: {error}"
                ))
            })?;
            candidates.insert(term.to_string());
            while candidates.len() > limit {
                candidates.pop_last();
            }
            if candidates.len() == limit
                && candidates
                    .last()
                    .is_some_and(|largest| term >= largest.as_str())
            {
                break;
            }
        }
        Ok(())
    }

    if terms.is_empty() || params.max_expansions == 0 {
        return Ok(BTreeMap::new());
    }
    let mut builder = fst::SetBuilder::memory();
    for term in terms {
        builder.insert(term).map_err(|error| {
            Error::index(format!("failed to build flat FTS vocabulary: {error}"))
        })?;
    }
    let set =
        fst::Set::new(builder.into_inner().map_err(|error| {
            Error::index(format!("failed to finish flat FTS vocabulary: {error}"))
        })?)
        .map_err(|error| Error::index(format!("failed to open flat FTS vocabulary: {error}")))?;

    let mut source_terms_by_position = BTreeMap::<u32, Vec<&str>>::new();
    for token_index in 0..query_tokens.len() {
        source_terms_by_position
            .entry(query_tokens.position(token_index))
            .or_default()
            .push(query_tokens.get_token(token_index));
    }
    let mut result = BTreeMap::new();
    for (position, source_terms) in source_terms_by_position {
        let candidates = result.entry(position).or_insert_with(BTreeSet::new);
        for source_term in source_terms {
            let fuzzy = fuzzy_term_options(
                source_term,
                query_tokens.token_type(),
                params.fuzziness,
                params.prefix_length,
            );
            let levenshtein = fst::automaton::Levenshtein::new(source_term, fuzzy.edit_distance)
                .map_err(|error| {
                    Error::index(format!("failed to construct the fuzzy query: {error}"))
                })?;
            match fuzzy.exact_prefix {
                "" => {
                    let mut stream = set.search(levenshtein).into_stream();
                    merge_stream(&mut stream, candidates, params.max_expansions)?;
                }
                exact_prefix => {
                    let prefix = fst::automaton::Str::new(exact_prefix).starts_with();
                    let mut stream = set.search(levenshtein.intersection(prefix)).into_stream();
                    merge_stream(&mut stream, candidates, params.max_expansions)?;
                }
            }
        }
    }
    Ok(result)
}

fn collect_batch_terms(
    batch: &RecordBatch,
    doc_col_idx: usize,
    coordinate_rank: usize,
    mut tokenizer: Box<dyn LanceTokenizer>,
) -> DataFusionResult<BTreeSet<String>> {
    let mut terms = BTreeSet::new();
    let mut collect_text = |document: &str| {
        let mut stream = tokenizer.token_stream_for_doc(document);
        while let Some(token) = stream.next() {
            terms.insert(token.text.to_string());
        }
    };
    match batch.column(doc_col_idx).data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            for document in iter_str_array(batch.column(doc_col_idx)) {
                match document {
                    Some(document) => collect_text(document),
                    None if coordinate_rank > 0 => collect_text(""),
                    None => {}
                }
            }
        }
        DataType::List(_) => {
            let documents = batch.column(doc_col_idx).as_list::<i32>();
            for row in 0..documents.len() {
                if documents.is_null(row) {
                    continue;
                }
                let elements = documents.value(row);
                let document = materialize_flat_string_list(elements.as_ref());
                collect_text(&document);
            }
        }
        DataType::LargeList(_) => {
            let documents = batch.column(doc_col_idx).as_list::<i64>();
            for row in 0..documents.len() {
                if documents.is_null(row) {
                    continue;
                }
                let elements = documents.value(row);
                let document = materialize_flat_string_list(elements.as_ref());
                collect_text(&document);
            }
        }
        data_type => {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "unsupported data type {data_type} for flat full text search"
            )));
        }
    }
    Ok(terms)
}

fn materialize_flat_string_list(elements: &dyn Array) -> String {
    let mut document = String::new();
    for element in iter_str_array(elements).flatten() {
        if !document.is_empty() {
            document.push(' ');
        }
        document.push_str(element);
    }
    document
}

/// Collect a bounded fuzzy-vocabulary summary from current-value rows.
///
/// Each rechunked input block builds a transient ordered FST over its deduped
/// vocabulary. Each query-source automaton traverses that FST once, avoiding
/// per-occurrence edit-distance work, and only the lexicographically smallest
/// `max_expansions` candidates per query position survive the merge. Memory is
/// bounded by one input chunk plus `query_positions * max_expansions`.
#[doc(hidden)]
pub async fn collect_flat_fuzzy_candidates(
    input: SendableRecordBatchStream,
    doc_col: &str,
    tokenizer: Box<dyn LanceTokenizer>,
    query_tokens: Arc<Tokens>,
    params: Arc<FtsSearchParams>,
    elapsed_compute: Option<Time>,
) -> DataFusionResult<BTreeMap<u32, BTreeSet<String>>> {
    const ACCUMULATE_BYTES: usize = 256 * 1024;
    const SLICE_BYTES: usize = 512 * 1024;
    let input_schema = input.schema();
    let doc_col_idx = input_schema.index_of(doc_col)?;
    let coordinate_rank = document_coordinate_rank(input_schema.as_ref());
    let chunked = lance_arrow::stream::rechunk_stream_by_size(
        input,
        input_schema,
        ACCUMULATE_BYTES,
        SLICE_BYTES,
    );
    let max_expansions = params.max_expansions;
    let summaries = chunked
        .map(move |batch| {
            let tokenizer = tokenizer.box_clone();
            let query_tokens = query_tokens.clone();
            let params = params.clone();
            let elapsed_compute = elapsed_compute.clone();
            spawn_cpu(move || {
                let started = std::time::Instant::now();
                let terms = collect_batch_terms(&batch?, doc_col_idx, coordinate_rank, tokenizer)?;
                let candidates = collect_fuzzy_candidates_from_terms(
                    &terms,
                    query_tokens.as_ref(),
                    params.as_ref(),
                )?;
                if let Some(time) = elapsed_compute {
                    time.add_duration(started.elapsed());
                }
                DataFusionResult::Ok(candidates)
            })
        })
        .buffered(get_num_compute_intensive_cpus());

    let mut merged = BTreeMap::<u32, BTreeSet<String>>::new();
    futures::pin_mut!(summaries);
    while let Some(summary) = summaries.try_next().await? {
        for (position, candidates) in summary {
            let output = merged.entry(position).or_default();
            output.extend(candidates);
            while output.len() > max_expansions {
                output.pop_last();
            }
        }
    }
    Ok(merged)
}

#[derive(Default)]
struct FlatBm25Stats {
    total_tokens: u64,
    num_docs: usize,
    token_docs: Vec<usize>,
}

fn collect_batch_bm25_stats(
    batch: &RecordBatch,
    doc_col_idx: usize,
    coordinate_rank: usize,
    mut tokenizer: Box<dyn LanceTokenizer>,
    term_indices: &HashMap<String, usize>,
) -> DataFusionResult<FlatBm25Stats> {
    fn count_text(
        tokenizer: &mut Box<dyn LanceTokenizer>,
        document: &str,
        term_indices: &HashMap<String, usize>,
        seen: &mut [bool],
        touched: &mut Vec<usize>,
    ) -> u64 {
        let mut token_count = 0_u64;
        let mut stream = tokenizer.token_stream_for_doc(document);
        while let Some(token) = stream.next() {
            token_count += 1;
            if let Some(index) = term_indices.get(&token.text)
                && !seen[*index]
            {
                seen[*index] = true;
                touched.push(*index);
            }
        }
        token_count
    }

    fn finish_document(
        stats: &mut FlatBm25Stats,
        seen: &mut [bool],
        touched: &mut Vec<usize>,
        token_count: u64,
        retain_empty: bool,
    ) {
        if token_count == 0 && !retain_empty {
            for index in touched.drain(..) {
                seen[index] = false;
            }
            return;
        }
        stats.total_tokens += token_count;
        stats.num_docs += 1;
        for index in touched.drain(..) {
            stats.token_docs[index] += 1;
            seen[index] = false;
        }
    }

    let mut stats = FlatBm25Stats {
        token_docs: vec![0; term_indices.len()],
        ..Default::default()
    };
    let mut seen = vec![false; term_indices.len()];
    let mut touched = Vec::with_capacity(term_indices.len().min(16));
    match batch.column(doc_col_idx).data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            for document in iter_str_array(batch.column(doc_col_idx)) {
                let count = match document {
                    Some(document) => count_text(
                        &mut tokenizer,
                        document,
                        term_indices,
                        &mut seen,
                        &mut touched,
                    ),
                    None if coordinate_rank > 0 => {
                        count_text(&mut tokenizer, "", term_indices, &mut seen, &mut touched)
                    }
                    None => 0,
                };
                finish_document(
                    &mut stats,
                    &mut seen,
                    &mut touched,
                    count,
                    coordinate_rank > 0,
                );
            }
        }
        DataType::List(_) => {
            let documents = batch.column(doc_col_idx).as_list::<i32>();
            for row in 0..documents.len() {
                let mut count = 0;
                if !documents.is_null(row) {
                    let elements = documents.value(row);
                    let document = materialize_flat_string_list(elements.as_ref());
                    count = count_text(
                        &mut tokenizer,
                        &document,
                        term_indices,
                        &mut seen,
                        &mut touched,
                    );
                }
                finish_document(&mut stats, &mut seen, &mut touched, count, false);
            }
        }
        DataType::LargeList(_) => {
            let documents = batch.column(doc_col_idx).as_list::<i64>();
            for row in 0..documents.len() {
                let mut count = 0;
                if !documents.is_null(row) {
                    let elements = documents.value(row);
                    let document = materialize_flat_string_list(elements.as_ref());
                    count = count_text(
                        &mut tokenizer,
                        &document,
                        term_indices,
                        &mut seen,
                        &mut touched,
                    );
                }
                finish_document(&mut stats, &mut seen, &mut touched, count, false);
            }
        }
        data_type => {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "unsupported data type {data_type} for flat full text search"
            )));
        }
    }
    Ok(stats)
}

/// Collect exact BM25 corpus statistics for an already-resolved vocabulary.
/// Input is reduced chunk-by-chunk to counters; document text and per-row token
/// vectors are never retained across chunks.
#[doc(hidden)]
pub async fn collect_flat_bm25_stats(
    input: SendableRecordBatchStream,
    doc_col: &str,
    tokenizer: Box<dyn LanceTokenizer>,
    query_tokens: Arc<Tokens>,
    elapsed_compute: Option<Time>,
) -> DataFusionResult<MemBM25Scorer> {
    let input_schema = input.schema();
    let doc_col_idx = input_schema.index_of(doc_col)?;
    let coordinate_rank = document_coordinate_rank(input_schema.as_ref());
    let mut terms = Vec::new();
    let mut term_indices = HashMap::new();
    for token in query_tokens.as_ref() {
        if !term_indices.contains_key(token) {
            let index = terms.len();
            terms.push(token.clone());
            term_indices.insert(token.clone(), index);
        }
    }
    let term_indices = Arc::new(term_indices);
    let summaries = input
        .map(move |batch| {
            let tokenizer = tokenizer.box_clone();
            let term_indices = term_indices.clone();
            let elapsed_compute = elapsed_compute.clone();
            spawn_cpu(move || {
                let started = std::time::Instant::now();
                let stats = collect_batch_bm25_stats(
                    &batch?,
                    doc_col_idx,
                    coordinate_rank,
                    tokenizer,
                    term_indices.as_ref(),
                )?;
                if let Some(time) = elapsed_compute {
                    time.add_duration(started.elapsed());
                }
                DataFusionResult::Ok(stats)
            })
        })
        .buffered(get_num_compute_intensive_cpus());
    let mut merged = FlatBm25Stats {
        token_docs: vec![0; terms.len()],
        ..Default::default()
    };
    futures::pin_mut!(summaries);
    while let Some(stats) = summaries.try_next().await? {
        merged.total_tokens = merged
            .total_tokens
            .checked_add(stats.total_tokens)
            .ok_or_else(|| {
                datafusion_common::DataFusionError::Execution(
                    "flat FTS corpus token count overflows u64".to_string(),
                )
            })?;
        merged.num_docs = merged.num_docs.checked_add(stats.num_docs).ok_or_else(|| {
            datafusion_common::DataFusionError::Execution(
                "flat FTS corpus document count overflows usize".to_string(),
            )
        })?;
        for (total, count) in merged.token_docs.iter_mut().zip(stats.token_docs) {
            *total = total.checked_add(count).ok_or_else(|| {
                datafusion_common::DataFusionError::Execution(
                    "flat FTS document frequency overflows usize".to_string(),
                )
            })?;
        }
    }
    Ok(MemBM25Scorer::new(
        merged.total_tokens,
        merged.num_docs,
        terms.into_iter().zip(merged.token_docs).collect(),
    ))
}

/// Run a flat BM25 search and return the scorer initialized from both the
/// optional indexed corpus and the flat input corpus.
pub async fn flat_bm25_search_stream_with_options_and_scorer(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    base_scorer: Option<MemBM25Scorer>,
    options: FlatBm25SearchOptions,
) -> DataFusionResult<(SendableRecordBatchStream, MemBM25Scorer)> {
    flat_bm25_search_stream_with_scorer_mode(
        input,
        doc_col,
        query,
        tokenizer,
        FlatScorerMode::Extend(base_scorer),
        None,
        options,
    )
    .await
}

/// Run a flat BM25 search with fixed corpus-wide statistics.
///
/// Unlike [`flat_bm25_search_stream_with_options_and_scorer`], documents in
/// `input` are scored but are not added to the supplied scorer. This is used
/// when corpus statistics were collected from an unfiltered stream while the
/// candidate stream has already applied a query filter.
#[doc(hidden)]
pub async fn flat_bm25_search_stream_with_options_and_fixed_scorer(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    scorer: MemBM25Scorer,
    options: FlatBm25SearchOptions,
) -> DataFusionResult<SendableRecordBatchStream> {
    Ok(flat_bm25_search_stream_with_scorer_mode(
        input,
        doc_col,
        query,
        tokenizer,
        FlatScorerMode::Fixed(scorer),
        None,
        options,
    )
    .await?
    .0)
}

/// Score current-value rows with exactly the vocabulary and corpus statistics
/// selected by canonical mixed fuzzy preparation.
#[doc(hidden)]
pub async fn flat_bm25_search_stream_with_options_and_prepared(
    input: SendableRecordBatchStream,
    doc_col: String,
    tokenizer: Box<dyn LanceTokenizer>,
    prepared: Arc<crate::scalar::inverted::PreparedBm25Query>,
    options: FlatBm25SearchOptions,
) -> DataFusionResult<SendableRecordBatchStream> {
    if options.operator == Operator::And && !prepared.has_all_query_positions() {
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            fts_schema(options.document_granularity),
            stream::empty::<DataFusionResult<RecordBatch>>(),
        )));
    }
    Ok(flat_bm25_search_stream_with_scorer_mode(
        input,
        doc_col,
        String::new(),
        tokenizer,
        FlatScorerMode::Fixed(prepared.scorer().as_ref().clone()),
        Some(prepared.tokens().clone()),
        options,
    )
    .await?
    .0)
}

enum FlatScorerMode {
    Extend(Option<MemBM25Scorer>),
    Fixed(MemBM25Scorer),
}

async fn flat_bm25_search_stream_with_scorer_mode(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    scorer_mode: FlatScorerMode,
    prepared_tokens: Option<Arc<Tokens>>,
    options: FlatBm25SearchOptions,
) -> DataFusionResult<(SendableRecordBatchStream, MemBM25Scorer)> {
    let FlatBm25SearchOptions {
        target_batch_size,
        elapsed_compute,
        document_granularity,
        operator,
        boost,
        phrase_slop,
    } = options;
    let mut tokenizer = tokenizer;
    let output_schema = fts_schema(document_granularity);

    // Pre-await synchronous work: query tokenization + chunk-stream setup.
    let pre_await_start = std::time::Instant::now();
    let query_tokens =
        prepared_tokens.unwrap_or_else(|| Arc::new(collect_query_tokens(&query, &mut tokenizer)));

    // A query that tokenizes to no terms (e.g. only stop words) has no
    // searchable content and matches nothing. Return early rather than
    // proceeding. This mirrors the indexed search path, which already
    // short-circuits on empty query tokens.
    if query_tokens.is_empty() {
        let scorer = base_scorer.unwrap_or_else(|| MemBM25Scorer::new(0, 0, HashMap::new()));
        return Ok((
            Box::pin(RecordBatchStreamAdapter::new(
                output_schema,
                stream::empty::<DataFusionResult<RecordBatch>>(),
            )),
            scorer,
        ));
    }

    let input_schema = input.schema();
    let doc_col_idx = input_schema.index_of(&doc_col)?;
    let coordinate_rank = document_coordinate_rank(input_schema.as_ref());
    if document_granularity.is_list_element() != (coordinate_rank > 0) {
        return Err(datafusion_common::DataFusionError::Execution(
            "flat FTS document granularity does not match its input coordinate columns".to_string(),
        ));
    }

    // Accumulate small batches until this threshold before dispatching a task.
    const ACCUMULATE_BYTES: usize = 256 * 1024;
    // Slice oversized batches down to roughly this size.
    const SLICE_BYTES: usize = 512 * 1024;

    // Phase 1 - rechunk the input stream into appropriately sized chunks.  Tokenization is
    // fairly CPU-intensive, and we don't need too much data to justify a new thread task.
    let chunked = lance_arrow::stream::rechunk_stream_by_size(
        input,
        input_schema,
        ACCUMULATE_BYTES,
        SLICE_BYTES,
    );
    if let Some(t) = &elapsed_compute {
        t.add_duration(pre_await_start.elapsed());
    }

    // A fixed scorer already contains corpus-wide statistics, so every input
    // chunk can be tokenized and scored independently. Keep only the bounded
    // in-flight chunks owned by `buffered`; never concatenate O(rows * terms)
    // token-count state for the whole current-value corpus.
    let base_scorer = match scorer_mode {
        FlatScorerMode::Fixed(scorer) => {
            let returned_scorer = scorer.clone();
            let scorer = Arc::new(scorer);
            let scored_chunks = chunked
                .map(move |batch| {
                    let tokenizer = tokenizer.box_clone();
                    let query_tokens = query_tokens.clone();
                    let scorer = scorer.clone();
                    let elapsed_compute = elapsed_compute.clone();
                    async move {
                        let counted_input = tokenize_and_count(
                            stream::iter([batch]),
                            tokenizer,
                            query_tokens.clone(),
                            doc_col_idx,
                            elapsed_compute.clone(),
                            coordinate_rank,
                            phrase_slop,
                        )
                        .await?;
                        let scores = spawn_cpu(move || {
                            let started = std::time::Instant::now();
                            let scores = flat_bm25_score(
                                query_tokens.as_ref(),
                                &counted_input,
                                scorer.as_ref(),
                                document_granularity,
                                operator,
                                boost,
                                phrase_slop,
                            )?;
                            if let Some(time) = elapsed_compute {
                                time.add_duration(started.elapsed());
                            }
                            DataFusionResult::Ok(scores)
                        })
                        .await?;
                        let batches = (0..scores.num_rows().div_ceil(target_batch_size))
                            .map(|index| {
                                let start = index * target_batch_size;
                                let len = (scores.num_rows() - start).min(target_batch_size);
                                Ok(scores.slice(start, len))
                            })
                            .collect::<Vec<DataFusionResult<RecordBatch>>>();
                        DataFusionResult::Ok(stream::iter(batches))
                    }
                })
                .buffered(get_num_compute_intensive_cpus())
                .try_flatten()
                .boxed();
            return Ok((
                Box::pin(RecordBatchStreamAdapter::new(output_schema, scored_chunks)),
                returned_scorer,
            ));
        }
        FlatScorerMode::Extend(base_scorer) => base_scorer,
    };

    // Phase 2 - For each row we need to know the total number of tokens and the count of each
    // of the query tokens.  For example, if the query is "book" and the row is "the book shop"
    // and we are tokenizing with a whitespace tokenizer, we need to know that there are 3 tokens
    // and the token book appears once.
    let counted_input = tokenize_and_count(
        chunked,
        tokenizer,
        query_tokens.clone(),
        doc_col_idx,
        elapsed_compute.clone(),
        coordinate_rank,
        phrase_slop,
    )
    .await?;

    // Phase 3 - Calculate final scores (this is fairly cheap, probably don't need to parallelize).
    // All post-await work is synchronous; time the scorer + score + slicing loop together.
    let post_await_start = std::time::Instant::now();
    let scorer = initialize_scorer(base_scorer.as_ref(), query_tokens.as_ref(), &counted_input);
    let scores = flat_bm25_score(
        query_tokens.as_ref(),
        &counted_input,
        &scorer,
        document_granularity,
        operator,
        boost,
        phrase_slop,
    )?;

    // Finally we emit batches according to the target batch size
    let num_out_batches = scores.num_rows().div_ceil(target_batch_size);
    let mut batches = Vec::with_capacity(num_out_batches);
    for i in 0..num_out_batches {
        let start = i * target_batch_size;
        let len = (scores.num_rows() - start).min(target_batch_size);
        batches.push(Ok(scores.slice(start, len)));
    }
    if let Some(t) = &elapsed_compute {
        t.add_duration(post_await_start.elapsed());
    }
    Ok((
        Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream::iter(batches),
        )),
        scorer,
    ))
}

pub fn is_phrase_query(query: &str) -> bool {
    phrase_query_text(query).is_some()
}

pub(super) fn phrase_query_text(query: &str) -> Option<&str> {
    query.strip_prefix('\"')?.strip_suffix('\"')
}
