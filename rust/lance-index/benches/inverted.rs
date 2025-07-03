// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of HNSW graph.
//!
//!

use std::{sync::Arc, time::Duration};

use arrow_array::{LargeStringArray, RecordBatch, UInt64Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use itertools::Itertools;
use lance_core::cache::LanceCache;
use lance_core::ROW_ID;
use lance_datagen::{array, RowCount};
use lance_index::prefilter::NoFilter;
use lance_index::scalar::inverted::query::{FtsSearchParams, Operator};
use lance_index::scalar::inverted::{InvertedIndex, InvertedIndexBuilder};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::ScalarIndex;
use lance_index::{
    metrics::NoOpMetricsCollector, scalar::inverted::tokenizer::InvertedIndexParams,
};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn bench_inverted(c: &mut Criterion) {
    const TOTAL: usize = 1_000_000;

    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    let tempdir = tempfile::tempdir().unwrap();
    let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
    let store = rt.block_on(async {
        Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            index_dir,
            Arc::new(LanceCache::no_cache()),
        ))
    });

    // generate random words using lance-datagen
    let row_id_col = Arc::new(UInt64Array::from(
        (0..TOTAL).map(|i| i as u64).collect_vec(),
    ));

    // Generate random words with 1-100 words per document
    let mut words_gen = array::random_sentence(1, 100, true);
    let doc_col = words_gen
        .generate_default(RowCount::from(TOTAL as u64))
        .unwrap();
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("doc", arrow_schema::DataType::LargeUtf8, false),
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
        ])
        .into(),
        vec![doc_col.clone(), row_id_col],
    )
    .unwrap();

    c.bench_function(format!("invert_indexing({TOTAL})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let stream = RecordBatchStreamAdapter::new(
                batch.schema(),
                stream::iter(vec![Ok(batch.clone())]),
            );
            let stream = Box::pin(stream);
            let mut builder =
                InvertedIndexBuilder::new(InvertedIndexParams::default().with_position(false));
            black_box({
                builder.update(stream, store.as_ref()).await.unwrap();
                builder
            });
        })
    });
    let invert_index = rt.block_on(InvertedIndex::load(store, None)).unwrap();

    let params = FtsSearchParams::new().with_limit(Some(10));
    let no_filter = Arc::new(NoFilter);

    // Get some sample words from the generated documents for search
    let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
    let sample_doc = large_string_array.value(0);
    let sample_words: Vec<String> = sample_doc
        .split_whitespace()
        .map(|s| s.to_owned())
        .collect();

    c.bench_function(format!("invert_search({TOTAL})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            // Pick a random word from our sample
            let word_idx = rand::random::<usize>() % sample_words.len();
            black_box(
                invert_index
                    .bm25_search(
                        vec![sample_words[word_idx].clone()].into(),
                        params.clone().into(),
                        Operator::Or,
                        no_filter.clone(),
                        Arc::new(NoOpMetricsCollector),
                    )
                    .await
                    .unwrap(),
            );
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_inverted);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_inverted);

criterion_main!(benches);
