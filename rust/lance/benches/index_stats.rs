// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for `Dataset::index_statistics`, comparing the current
//! implementation against the pre-P1/P2 baseline.
//!
//! Two optimizations are measured:
//!
//!  * **P1** — derive the total row count from the manifest instead of calling
//!    `Dataset::count_rows(None)`, which fans out per-fragment and can read
//!    fragment data files on datasets written by older Lance versions.
//!  * **P2** — cache `index_statistics` JSON keyed by manifest version in the
//!    dataset's `DSIndexCache`, so repeat calls at the same version are served
//!    from memory.
//!
//! The benchmark runs four functions on the same fixture in the same process:
//!
//!  * `index_stats/legacy_cold` — pre-P1/P2 behavior. Always calls
//!    `count_rows(None)`, never hits the cache. Exercised via the
//!    `#[doc(hidden)]` `bench_legacy_index_statistics` entry point.
//!  * `index_stats/cold` — current behavior with a cold dataset cache
//!    (session reopened every iteration). Isolates the P1 win against
//!    `legacy_cold`.
//!  * `index_stats/cached` — current behavior after the first call. Isolates
//!    the P2 win against `cold`.
//!  * `index_stats/count_rows_baseline` — wall time of the `count_rows(None)`
//!    fan-out alone, for context on what P1 removes from the hot path.

use std::sync::{Arc, OnceLock};

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::builder::DatasetBuilder;
use lance::index::DatasetIndexExt;
use lance::index::bench_legacy_index_statistics;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

/// Fixture shape. 10 M rows split into 10 000 fragments of 1 000 rows each —
/// large enough that `count_rows`'s per-fragment fan-out dominates, at a
/// fragment count closer to what's seen in production on tables that haven't
/// been compacted recently.
const NUM_FRAGMENTS: usize = 10_000;
const ROWS_PER_FRAGMENT: usize = 1_000;
const TOTAL_ROWS: usize = NUM_FRAGMENTS * ROWS_PER_FRAGMENT;
const INDEX_NAME: &str = "id_idx";

struct Fixture {
    // Kept alive for the lifetime of the benchmark so the on-disk data stays valid.
    _tempdir: TempStrDir,
    uri: String,
}

async fn build_fixture() -> Fixture {
    let tempdir = TempStrDir::default();
    let uri = tempdir.as_str().to_string();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    // One `Dataset::write` call produces multiple fragments when the input
    // exceeds `max_rows_per_file` — vastly cheaper than committing per fragment.
    // Feed the data in batches of `ROWS_PER_FRAGMENT` so memory stays bounded.
    let batches: Vec<RecordBatch> = (0..NUM_FRAGMENTS)
        .map(|f| {
            let start = (f * ROWS_PER_FRAGMENT) as i32;
            let values: Vec<i32> = (start..start + ROWS_PER_FRAGMENT as i32).collect();
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(values))]).unwrap()
        })
        .collect();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &uri, Some(params)).await.unwrap();
    assert_eq!(dataset.fragments().len(), NUM_FRAGMENTS);
    assert_eq!(dataset.count_rows(None).await.unwrap(), TOTAL_ROWS);

    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some(INDEX_NAME.into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    Fixture {
        _tempdir: tempdir,
        uri,
    }
}

/// Reopen the dataset so we get a fresh session (and therefore a fresh
/// `DSIndexCache`). This lets each iteration measure the cold path.
async fn open_cold(uri: &str) -> Dataset {
    DatasetBuilder::from_uri(uri).load().await.unwrap()
}

/// Process-wide runtime + fixture. Built lazily on first access so we pay the
/// 10 M-row dataset write once, not four times.
struct BenchEnv {
    rt: tokio::runtime::Runtime,
    fixture: Fixture,
}

fn env() -> &'static BenchEnv {
    static ENV: OnceLock<BenchEnv> = OnceLock::new();
    ENV.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let fixture = rt.block_on(build_fixture());
        // Parity check against the fresh fixture: legacy and current paths must
        // agree on every counter in the JSON payload before measurement begins.
        rt.block_on(async {
            let ds = open_cold(&fixture.uri).await;
            let legacy: serde_json::Value = serde_json::from_str(
                &bench_legacy_index_statistics(&ds, INDEX_NAME).await.unwrap(),
            )
            .unwrap();
            let current: serde_json::Value =
                serde_json::from_str(&ds.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
            for key in [
                "num_indexed_rows",
                "num_unindexed_rows",
                "num_indexed_fragments",
                "num_unindexed_fragments",
                "num_indices",
            ] {
                assert_eq!(
                    legacy[key], current[key],
                    "legacy and current paths disagree on {key}",
                );
            }
        });
        BenchEnv { rt, fixture }
    })
}

fn bench_count_rows(c: &mut Criterion) {
    let env = env();
    let dataset = env.rt.block_on(open_cold(&env.fixture.uri));

    c.bench_function("index_stats/count_rows_baseline", |b| {
        b.iter(|| {
            env.rt.block_on(async {
                let _ = dataset.count_rows(None).await.unwrap();
            })
        });
    });
}

fn bench_legacy_cold(c: &mut Criterion) {
    let env = env();

    // Pre-P1/P2 behavior: no cache, `count_rows(None)` every call.
    c.bench_function("index_stats/legacy_cold", |b| {
        b.iter(|| {
            env.rt.block_on(async {
                let ds = open_cold(&env.fixture.uri).await;
                let _ = bench_legacy_index_statistics(&ds, INDEX_NAME)
                    .await
                    .unwrap();
            })
        });
    });
}

fn bench_cold(c: &mut Criterion) {
    let env = env();

    // Current behavior, cold cache. Difference vs `legacy_cold` is the P1 win.
    c.bench_function("index_stats/cold", |b| {
        b.iter(|| {
            env.rt.block_on(async {
                let ds = open_cold(&env.fixture.uri).await;
                let _ = ds.index_statistics(INDEX_NAME).await.unwrap();
            })
        });
    });
}

fn bench_cached(c: &mut Criterion) {
    let env = env();
    let dataset = env.rt.block_on(open_cold(&env.fixture.uri));

    // Prime the cache.
    env.rt.block_on(async {
        let _ = dataset.index_statistics(INDEX_NAME).await.unwrap();
    });

    // Current behavior, warm cache. Difference vs `cold` is the P2 win.
    c.bench_function("index_stats/cached", |b| {
        b.iter(|| {
            env.rt.block_on(async {
                let _ = dataset.index_statistics(INDEX_NAME).await.unwrap();
            })
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(30)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_count_rows, bench_legacy_cold, bench_cold, bench_cached
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(30);
    targets = bench_count_rows, bench_legacy_cold, bench_cold, bench_cached
);

criterion_main!(benches);
