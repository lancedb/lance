// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Duration;
use std::{collections::HashMap, sync::Arc};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance_core::ROW_ID;
use lance_index::scalar::inverted::{
    flat_bm25_search,
    lance_tokenizer::{LanceTokenizer, TextTokenizer},
    query::collect_query_tokens,
    MemBM25Scorer, FTS_SCHEMA,
};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Zipf;

fn generate_batch(num_rows: usize) -> RecordBatch {
    const VOCAB_SIZE: usize = 50_000;
    const MIN_WORDS: usize = 5;
    const MAX_WORDS: usize = 80;
    const ZIPF_EXPONENT: f64 = 1.1;

    let vocab: Vec<String> = (0..VOCAB_SIZE).map(|i| format!("term{i:05}")).collect();
    let word_zipf = Zipf::new(VOCAB_SIZE as f64, ZIPF_EXPONENT).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    let schema = Arc::new(Schema::new(vec![
        Field::new("doc", DataType::LargeUtf8, false),
        Field::new(ROW_ID, DataType::UInt64, false),
    ]));

    let mut docs = Vec::with_capacity(num_rows);
    for _ in 0..num_rows {
        let num_words = rng.random_range(MIN_WORDS..=MAX_WORDS);
        let mut doc = String::with_capacity(num_words * 8);
        for j in 0..num_words {
            if j > 0 {
                doc.push(' ');
            }
            let idx = (rng.sample(word_zipf) as usize).clamp(1, VOCAB_SIZE) - 1;
            doc.push_str(&vocab[idx]);
        }
        docs.push(doc);
    }

    let doc_col = Arc::new(arrow_array::LargeStringArray::from(docs));
    let row_id_col = Arc::new(UInt64Array::from_iter_values(0..num_rows as u64));

    RecordBatch::try_new(schema.clone(), vec![doc_col, row_id_col]).unwrap()
}

fn bench_flat_bm25(c: &mut Criterion) {
    let num_rows = 100_000;

    let query = "term00001 term00010 term00100 term01000 term10000".to_string();
    let mut tokenizer = Box::new(TextTokenizer::new(
        tantivy::tokenizer::TextAnalyzer::builder(tantivy::tokenizer::SimpleTokenizer::default())
            .build(),
    )) as Box<dyn LanceTokenizer>;
    let tokens = collect_query_tokens(&query, &mut tokenizer, None);

    let mut bm25_scorer = MemBM25Scorer::new(0, 0, HashMap::new());

    let mut group = c.benchmark_group("flat_bm25_search_stream");

    let batch = generate_batch(num_rows);

    group.bench_function("search", |b| {
        b.iter(|| {
            let batch = batch.clone();
            let output_schema = FTS_SCHEMA.clone();
            let tokens = &tokens;
            let mut tokenizer = &mut tokenizer;
            let mut bm25_scorer = &mut bm25_scorer;
            let batch = flat_bm25_search(
                batch,
                "doc",
                &tokens,
                &mut tokenizer,
                &mut bm25_scorer,
                output_schema,
            )
            .unwrap();
            criterion::black_box(batch);
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_flat_bm25
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_flat_bm25
);

criterion_main!(benches);
