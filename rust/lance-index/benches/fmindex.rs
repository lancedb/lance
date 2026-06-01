// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{sync::Arc, time::Duration};

use arrow::array::AsArray;
use arrow_array::{LargeStringArray, RecordBatch, StringArray, UInt64Array};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use itertools::Itertools;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_datagen::{RowCount, array};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::fmindex::FMIndexPlugin;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{TextQuery, registry::ScalarIndexPlugin};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn load_data() -> (RecordBatch, Vec<RecordBatch>) {
    if let Ok(path) = std::env::var("FMINDEX_DATA_PATH") {
        let text = std::fs::read_to_string(&path).expect("Cannot read FMINDEX_DATA_PATH");
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        let total = lines.len();
        println!("Loaded {} lines from {}", total, path);

        let row_id_col = Arc::new(UInt64Array::from(
            (0..total).map(|i| i as u64).collect_vec(),
        ));
        let doc_col = Arc::new(LargeStringArray::from(
            lines.iter().map(|l| l.to_string()).collect_vec(),
        ));
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", arrow_schema::DataType::LargeUtf8, false),
                arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            ])
            .into(),
            vec![doc_col, row_id_col],
        )
        .unwrap();

        let batch_size = 1000.min(total);
        let num_batches = total / batch_size;
        let batches = (0..num_batches)
            .map(|i| batch.slice(i * batch_size, batch_size))
            .collect_vec();

        return (batch, batches);
    }

    let total = 10_000_000usize;
    let row_id_col = Arc::new(UInt64Array::from(
        (0..total).map(|i| i as u64).collect_vec(),
    ));
    let mut words_gen = array::random_sentence(1, 30, false);
    let doc_col = words_gen
        .generate_default(RowCount::from(total as u64))
        .unwrap();
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("doc", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
        ])
        .into(),
        vec![doc_col, row_id_col],
    )
    .unwrap();
    let batches = (0..1000).map(|i| batch.slice(i * 1000, 1000)).collect_vec();
    (batch, batches)
}

fn bench_fmindex(c: &mut Criterion) {
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

    let (batch, batches) = load_data();
    let total = batch.num_rows();

    let mut group = c.benchmark_group("fmindex_train");
    group.sample_size(10);
    group.bench_function(format!("fmindex_train({total})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let stream = RecordBatchStreamAdapter::new(
                batch.schema(),
                stream::iter(batches.clone().into_iter().map(Ok)),
            );
            let stream = Box::pin(stream);
            let req = FMIndexPlugin
                .new_training_request("", batch.schema().field(0))
                .unwrap();
            FMIndexPlugin
                .train_index(
                    stream,
                    store.as_ref(),
                    req,
                    None,
                    Arc::new(NoopIndexBuildProgress),
                )
                .await
                .unwrap();
        })
    });
    drop(group);

    let created = rt.block_on(async {
        let stream = RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::iter(batches.clone().into_iter().map(Ok)),
        );
        let stream = Box::pin(stream);
        let req = FMIndexPlugin
            .new_training_request("", batch.schema().field(0))
            .unwrap();
        let res = FMIndexPlugin
            .train_index(
                stream,
                store.as_ref(),
                req,
                None,
                Arc::new(NoopIndexBuildProgress),
            )
            .await
            .unwrap();

        if let Some(ref files) = res.files {
            for file in files {
                println!("FM-INDEX FILE: {} ({} bytes)", file.path, file.size_bytes);
            }
        }
        res
    });

    let mut group = c.benchmark_group("fmindex_search");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    let index = rt
        .block_on(FMIndexPlugin.load_index(
            store,
            &created.index_details,
            None,
            &LanceCache::no_cache(),
        ))
        .unwrap();
    group.bench_function(format!("fmindex_search_short({total})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let sample_idx = rand::random_range(0..batch.num_rows());
            let sample =
                if let Some(arr) = batch.column(0).as_any().downcast_ref::<LargeStringArray>() {
                    arr.value(sample_idx).to_string()
                } else {
                    batch
                        .column(0)
                        .as_string::<i32>()
                        .value(sample_idx)
                        .to_string()
                };
            let query_str = {
                let chars: Vec<char> = sample.chars().collect();
                if chars.len() > 10 {
                    let start = rand::random_range(0..chars.len() - 10);
                    let end = (start + rand::random_range(3..15)).min(chars.len());
                    chars[start..end].iter().collect::<String>()
                } else if !chars.is_empty() {
                    sample
                } else {
                    "the".to_string()
                }
            };
            black_box(
                index
                    .search(&TextQuery::StringContains(query_str), &NoOpMetricsCollector)
                    .await
                    .unwrap(),
            );
        })
    });

    group.bench_function(format!("fmindex_search_long({total})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let sample_idx = rand::random_range(0..batch.num_rows());
            let sample =
                if let Some(arr) = batch.column(0).as_any().downcast_ref::<LargeStringArray>() {
                    arr.value(sample_idx).to_string()
                } else {
                    batch
                        .column(0)
                        .as_string::<i32>()
                        .value(sample_idx)
                        .to_string()
                };
            let query_str = {
                let chars: Vec<char> = sample.chars().collect();
                if chars.len() > 50 {
                    let start = rand::random_range(0..chars.len() - 50);
                    let end = (start + rand::random_range(30..80)).min(chars.len());
                    chars[start..end].iter().collect::<String>()
                } else if chars.len() > 10 {
                    chars[..chars.len()].iter().collect::<String>()
                } else if !chars.is_empty() {
                    sample
                } else {
                    "function".to_string()
                }
            };
            black_box(
                index
                    .search(&TextQuery::StringContains(query_str), &NoOpMetricsCollector)
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
    targets = bench_fmindex);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_fmindex);

criterion_main!(benches);
