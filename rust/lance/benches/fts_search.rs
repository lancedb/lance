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
use std::{env, sync::Arc};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use lance::{Dataset, dataset::WriteParams, index::DatasetIndexExt};
use lance_index::{
    IndexType,
    scalar::{FullTextSearchQuery, inverted::tokenizer::InvertedIndexParams},
};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use tempfile::TempDir;

const WIKIPEDIA_DATASET_ENV_VAR: &str = "LANCE_WIKIPEDIA_DATASET_PATH";
const INDEX_NAME: &str = "segmented_fts";
const INDEXED_FRAGMENT_COUNT: usize = 12;
const UNINDEXED_FRAGMENT_COUNT: usize = 1;
const ROWS_PER_FRAGMENT: usize = 64;

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

struct BenchDataset {
    _tmpdir: TempDir,
    dataset: Dataset,
}

fn create_fragment_batch(fragment_id: usize) -> RecordBatch {
    let start = (fragment_id * ROWS_PER_FRAGMENT) as i32;
    let ids = Arc::new(Int32Array::from_iter_values(
        start..start + ROWS_PER_FRAGMENT as i32,
    ));
    let texts = Arc::new(StringArray::from_iter_values((0..ROWS_PER_FRAGMENT).map(
        |row| {
            let term = match (fragment_id + row) % 4 {
                0 => "alpha",
                1 => "beta",
                2 => "gamma",
                _ => "delta",
            };
            format!("shared {term} fragment-{fragment_id} row-{row}")
        },
    )));
    RecordBatch::try_from_iter(vec![("id", ids as ArrayRef), ("text", texts as ArrayRef)]).unwrap()
}

fn grouped_fragment_ids(segment_count: usize) -> Vec<Vec<u32>> {
    let fragments_per_segment = INDEXED_FRAGMENT_COUNT / segment_count;
    (0..segment_count)
        .map(|segment_idx| {
            let start = segment_idx * fragments_per_segment;
            let end = start + fragments_per_segment;
            (start..end).map(|fragment_id| fragment_id as u32).collect()
        })
        .collect()
}

async fn build_segmented_fts_dataset(segment_count: usize) -> BenchDataset {
    let tmpdir = TempDir::new().unwrap();
    let uri = format!("file://{}", tmpdir.path().display());
    let batches = RecordBatchIterator::new(
        (0..(INDEXED_FRAGMENT_COUNT + UNINDEXED_FRAGMENT_COUNT))
            .map(|fragment_id| Ok(create_fragment_batch(fragment_id)))
            .collect::<Vec<_>>(),
        create_fragment_batch(0).schema(),
    );
    let mut dataset = Dataset::write(
        batches,
        &uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        dataset.get_fragments().len(),
        INDEXED_FRAGMENT_COUNT + UNINDEXED_FRAGMENT_COUNT
    );

    let params = InvertedIndexParams::default();
    let mut staged_segments = Vec::with_capacity(segment_count);
    for fragment_ids in grouped_fragment_ids(segment_count) {
        let segment = dataset
            .create_index_builder(&["text"], IndexType::Inverted, &params)
            .name(INDEX_NAME.to_string())
            .fragments(fragment_ids)
            .execute_uncommitted()
            .await
            .unwrap();
        staged_segments.push(segment);
    }
    let segments = dataset
        .create_index_segment_builder()
        .with_index_type(IndexType::Inverted)
        .with_segments(staged_segments)
        .build_all()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments(INDEX_NAME, "text", segments)
        .await
        .unwrap();

    BenchDataset {
        _tmpdir: tmpdir,
        dataset,
    }
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

fn bench_segmented_fts_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let bench_datasets = [1_usize, 2, 4, 6]
        .into_iter()
        .map(|segment_count| {
            (
                segment_count,
                rt.block_on(build_segmented_fts_dataset(segment_count)),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("fts_search_segment_count");
    for (segment_count, bench_dataset) in &bench_datasets {
        group.bench_with_input(
            BenchmarkId::from_parameter(segment_count),
            segment_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut scanner = bench_dataset.dataset.scan();
                        let query = FullTextSearchQuery::new("shared alpha".to_string())
                            .with_column("text".to_string())
                            .unwrap();
                        let mut stream = scanner
                            .full_text_search(query)
                            .unwrap()
                            .limit(Some(20), None)
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
                        assert!(num_rows <= 20);
                    })
                });
            },
        );
    }
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_fts_search, bench_segmented_fts_search
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_fts_search, bench_segmented_fts_search
);

criterion_main!(benches);
