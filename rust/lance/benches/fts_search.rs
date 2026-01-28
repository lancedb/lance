// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// This is a rust end-to-end benchmark for full text search.  It is meant to be supplementary to the
/// python benchmark located at python/python/ci_benchmarks/benchmarks/test_fts_search.py.  You can use
/// the python/python/ci_benchmarks/datagen/wikipedia.py script to generate the dataset.  You will need
/// to set the LANCE_WIKIPEDIA_DATASET_PATH environment variable to the path of the dataset generated
/// by that script.
///
/// This benchmark is primarily intended for developers to use for profiling and debugging.  The python
/// benchmark is more comprehensive and will cover regression testing.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::TryStreamExt;
use lance::Dataset;
use lance_index::scalar::FullTextSearchQuery;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use std::env;

const WIKIPEDIA_DATASET_ENV_VAR: &str = "LANCE_WIKIPEDIA_DATASET_PATH";

/// Get the Wikipedia dataset path from environment variable.
/// Panics if the environment variable is not set.
fn get_wikipedia_dataset_path() -> String {
    env::var(WIKIPEDIA_DATASET_ENV_VAR).unwrap_or_else(|_| {
        panic!(
            "Environment variable {} must be set to the path of the indexed Wikipedia dataset",
            WIKIPEDIA_DATASET_ENV_VAR
        )
    })
}

/// Benchmark full text search on Wikipedia dataset with different K values
fn bench_fts_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dataset_path = get_wikipedia_dataset_path();

    // Open the dataset once
    let dataset = rt
        .block_on(Dataset::open(&dataset_path))
        .unwrap_or_else(|e| {
            panic!(
                "Failed to open Wikipedia dataset at '{}': {}",
                dataset_path, e
            )
        });

    // Test with different K values
    let k_values = [10, 100, 1000];

    let mut group = c.benchmark_group("fts_search_lost_episode");

    for k in k_values.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            b.iter(|| {
                rt.block_on(async {
                    let mut scanner = dataset.scan();
                    let mut stream = scanner
                        .full_text_search(FullTextSearchQuery::new("lost episode".to_string()))
                        .unwrap()
                        .limit(Some(k as i64), None)
                        .unwrap()
                        .project(&["_rowid"])
                        .unwrap()
                        .try_into_stream()
                        .await
                        .unwrap();

                    let mut num_rows = 0;
                    while let Some(batch) = stream.try_next().await.unwrap() {
                        num_rows += batch.num_rows();
                    }

                    // Verify we got results (should be at most k rows)
                    assert!(
                        num_rows <= k,
                        "Expected at most {} rows, got {}",
                        k,
                        num_rows
                    );
                })
            });
        });
    }

    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_fts_search
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_fts_search
);

criterion_main!(benches);
