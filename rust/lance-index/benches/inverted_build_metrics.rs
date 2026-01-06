// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{sync::Arc, time::Duration};

use arrow_array::{RecordBatch, UInt64Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::cache::LanceCache;
use lance_core::ROW_ID;
use lance_datagen::{array, RowCount};
use lance_index::scalar::inverted::InvertedIndexBuilder;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn build_metrics_bench(c: &mut Criterion) {
    let total_docs: usize = std::env::var("LANCE_FTS_BENCH_DOCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);

    let cases = [
        ("short_no_pos", 3, 15, false),
        ("short_pos", 3, 15, true),
        ("long_no_pos", 50, 200, false),
    ];

    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    for (name, min_words, max_words, with_position) in cases {
        let mut words_gen = array::random_sentence(min_words, max_words, true);
        let doc_col = words_gen
            .generate_default(RowCount::from(total_docs as u64))
            .unwrap();
        let row_id_col = Arc::new(UInt64Array::from_iter_values(0..total_docs as u64));

        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", arrow_schema::DataType::LargeUtf8, false),
                arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            ])
            .into(),
            vec![doc_col.clone(), row_id_col],
        )
        .unwrap();

        c.bench_function(format!("invert_build_{name}_{total_docs}").as_str(), |b| {
            b.to_async(&rt).iter(|| async {
                let tempdir = tempfile::tempdir().unwrap();
                let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
                let store = Arc::new(LanceIndexStore::new(
                    Arc::new(ObjectStore::local()),
                    index_dir,
                    Arc::new(LanceCache::no_cache()),
                ));

                let stream = RecordBatchStreamAdapter::new(
                    batch.schema(),
                    stream::iter(vec![Ok(batch.clone())]),
                );
                let stream = Box::pin(stream);

                let params = InvertedIndexParams::new(
                    "whitespace".to_string(),
                    tantivy::tokenizer::Language::English,
                )
                .with_position(with_position)
                .remove_stop_words(false)
                .stem(false)
                .max_token_length(None);

                let mut builder = InvertedIndexBuilder::new(params);
                let metrics = builder.enable_metrics();
                builder.update(stream, store.as_ref()).await.unwrap();

                black_box(metrics.snapshot());
            })
        });
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = build_metrics_bench
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = build_metrics_bench
);

criterion_main!(benches);
