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
use lance_core::ROW_ID;
use lance_index::scalar::inverted::InvertedIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{ScalarIndex, ScalarQuery};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn bench_inverted(c: &mut Criterion) {
    const TOTAL: usize = 30_000_000;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let tempdir = tempfile::tempdir().unwrap();
    let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
    let store = Arc::new(LanceIndexStore::new(ObjectStore::local(), index_dir, None));

    let invert_index = InvertedIndex::default();
    // generate 2000 different tokens
    let tokens = random_word::all(random_word::Lang::En);
    let row_id_col = Arc::new(UInt64Array::from(
        (0..TOTAL).map(|i| i as u64).collect_vec(),
    ));
    let docs = (0..TOTAL)
        .map(|_| {
            let num_words = rand::random::<usize>() % 100 + 1;
            let doc = (0..num_words)
                .map(|_| tokens[rand::random::<usize>() % tokens.len()])
                .collect::<Vec<_>>();
            doc.join(" ")
        })
        .collect_vec();
    let doc_col = Arc::new(LargeStringArray::from(docs));
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("doc", arrow_schema::DataType::LargeUtf8, false),
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
        ])
        .into(),
        vec![doc_col.clone(), row_id_col.clone()],
    )
    .unwrap();
    let stream = RecordBatchStreamAdapter::new(batch.schema(), stream::iter(vec![Ok(batch)]));
    let stream = Box::pin(stream);

    rt.block_on(async { invert_index.update(stream, store.as_ref()).await.unwrap() });
    let invert_index = rt.block_on(InvertedIndex::load(store)).unwrap();

    c.bench_function(format!("invert({TOTAL})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            black_box(
                invert_index
                    .search(&ScalarQuery::FullTextSearch(vec![tokens
                        [rand::random::<usize>() % tokens.len()]
                    .to_owned()]))
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
