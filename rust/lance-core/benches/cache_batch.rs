// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Microbenchmarks for Lance cache get-or-insert paths.
//!
//! These benchmarks measure cache primitive overhead and duplicate-load
//! behavior. They intentionally use small `usize` values plus cheap/yielding
//! loaders so the cache control flow is visible in the measurements.
//!
//! They are not an end-to-end storage read benchmark. They do not model page
//! bytes, materialization cost, object-store range I/O, or real storage planning.
//! Use them as evidence for cache-level overhead and single-flight behavior, not
//! as proof that a higher-level storage read path is faster end to end.
//!
//! These benchmarks compare three ways to handle cold overlapping batch reads:
//! - per-key `get_or_insert_with_key`: keeps single-flight but loses coalesced loading.
//! - manual `get` + batch load + `insert`: keeps coalesced loading but duplicates
//!   cold overlapping work.
//! - `get_or_insert_with_key_batch`: keeps coalesced loading and backend single-flight.
//!
//! The no-overlap groups are intentionally included to show the control-flow
//! cost of the new primitive when there is no duplicate work to avoid. For the
//! eventual BTree integration, use an end-to-end benchmark with real page bytes,
//! materialization, and object-store range planning before making storage-path
//! performance claims.
//!
//! The assertions in this file are correctness guards for the measured scenario,
//! not replacement unit tests. They keep the benchmark from silently measuring a
//! broken setup after future cache changes.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_core::Result;
use lance_core::cache::{CacheBatchValue, CacheKey, LanceCache};
use tokio::sync::Barrier;
use tokio::task::JoinHandle;

const CACHE_CAPACITY_BYTES: usize = 64 * 1024 * 1024;
const BATCH_SIZES: [usize; 4] = [1, 8, 64, 512];
const CONCURRENCIES: [usize; 3] = [1, 8, 32];

// Bench-local typed key. The key string is cached so hot-path measurements do
// not include repeated integer formatting.
#[derive(Clone, Debug)]
struct BenchKey {
    id: usize,
    cache_key: Arc<str>,
}

impl BenchKey {
    fn new(id: usize) -> Self {
        Self {
            id,
            cache_key: Arc::from(id.to_string()),
        }
    }
}

impl CacheKey for BenchKey {
    type ValueType = usize;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.cache_key.as_ref())
    }

    fn type_name() -> &'static str {
        "cache_batch_bench::usize"
    }
}

#[derive(Clone, Copy, Debug)]
enum LoaderMode {
    Cheap,
    // A single scheduler yield is enough to exercise waiter paths without
    // making the benchmark mostly a sleep/timer benchmark.
    YieldOnce,
}

impl LoaderMode {
    fn name(self) -> &'static str {
        match self {
            Self::Cheap => "cheap",
            Self::YieldOnce => "yield_once",
        }
    }

    async fn wait(self) {
        match self {
            Self::Cheap => {}
            Self::YieldOnce => tokio::task::yield_now().await,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Overlap {
    Half,
    Full,
    None,
}

impl Overlap {
    fn name(self) -> &'static str {
        match self {
            Self::Half => "50pct",
            Self::Full => "100pct",
            Self::None => "0pct",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SingleParams {
    concurrency: usize,
    loader_mode: LoaderMode,
}

impl Display for SingleParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "concurrency={},loader={}",
            self.concurrency,
            self.loader_mode.name()
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct BatchParams {
    batch_size: usize,
    concurrency: usize,
    overlap: Overlap,
    loader_mode: LoaderMode,
}

impl Display for BatchParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "batch={},concurrency={},overlap={},loader={}",
            self.batch_size,
            self.concurrency,
            self.overlap.name(),
            self.loader_mode.name()
        )
    }
}

fn criterion_config() -> Criterion {
    // Keep the default command usable for PR review. For more stable numbers,
    // run Criterion with a larger sample size and measurement time locally.
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250))
}

fn make_cache() -> LanceCache {
    LanceCache::with_capacity(CACHE_CAPACITY_BYTES)
}

fn new_counts(count: usize) -> Arc<Vec<AtomicUsize>> {
    Arc::new((0..count).map(|_| AtomicUsize::new(0)).collect())
}

fn sum_counts(counts: &[AtomicUsize]) -> usize {
    counts
        .iter()
        .map(|count| count.load(Ordering::SeqCst))
        .sum()
}

fn assert_each_key_loaded_once(counts: &[AtomicUsize]) {
    for (id, count) in counts.iter().enumerate() {
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "key {id} should be loaded exactly once"
        );
    }
}

fn generate_batches(params: BatchParams) -> (Vec<Vec<BenchKey>>, usize) {
    match params.overlap {
        Overlap::Full => {
            // Every task asks for the same logical batch. This is the strongest
            // duplicate-load pressure on the flight registry.
            let keys = (0..params.batch_size)
                .map(BenchKey::new)
                .collect::<Vec<_>>();
            (vec![keys; params.concurrency], params.batch_size)
        }
        Overlap::Half => {
            // The first half is shared across all tasks and the second half is
            // task-local. This models broad range queries with partially
            // overlapping page windows.
            let shared_count = params.batch_size.div_ceil(2);
            let unique_per_task = params.batch_size - shared_count;
            let mut batches = Vec::with_capacity(params.concurrency);

            for task_idx in 0..params.concurrency {
                let mut keys = Vec::with_capacity(params.batch_size);
                keys.extend((0..shared_count).map(BenchKey::new));
                let unique_start = shared_count + task_idx * unique_per_task;
                keys.extend((unique_start..unique_start + unique_per_task).map(BenchKey::new));
                batches.push(keys);
            }

            (batches, shared_count + params.concurrency * unique_per_task)
        }
        Overlap::None => {
            // Disjoint batches should not benefit from single-flight. This case
            // isolates the extra registry work when there is no contention.
            let mut batches = Vec::with_capacity(params.concurrency);

            for task_idx in 0..params.concurrency {
                let start = task_idx * params.batch_size;
                batches.push(
                    (start..start + params.batch_size)
                        .map(BenchKey::new)
                        .collect(),
                );
            }

            (batches, params.concurrency * params.batch_size)
        }
    }
}

fn overlapping_batch_params() -> Vec<BatchParams> {
    // This is a representative matrix, not a full Cartesian product. It covers
    // the requested batch sizes, concurrency levels, overlap levels, and both
    // loader modes while keeping `cargo bench -- --sample-size 10` practical.
    vec![
        BatchParams {
            batch_size: 1,
            concurrency: 8,
            overlap: Overlap::Half,
            loader_mode: LoaderMode::Cheap,
        },
        BatchParams {
            batch_size: 8,
            concurrency: 8,
            overlap: Overlap::Half,
            loader_mode: LoaderMode::Cheap,
        },
        BatchParams {
            batch_size: 64,
            concurrency: 8,
            overlap: Overlap::Half,
            loader_mode: LoaderMode::Cheap,
        },
        BatchParams {
            batch_size: 512,
            concurrency: 8,
            overlap: Overlap::Half,
            loader_mode: LoaderMode::Cheap,
        },
        BatchParams {
            batch_size: 8,
            concurrency: 1,
            overlap: Overlap::Half,
            loader_mode: LoaderMode::Cheap,
        },
        BatchParams {
            batch_size: 8,
            concurrency: 32,
            overlap: Overlap::Half,
            loader_mode: LoaderMode::Cheap,
        },
        BatchParams {
            batch_size: 64,
            concurrency: 8,
            overlap: Overlap::Full,
            loader_mode: LoaderMode::Cheap,
        },
        BatchParams {
            batch_size: 64,
            concurrency: 8,
            overlap: Overlap::Half,
            loader_mode: LoaderMode::YieldOnce,
        },
        BatchParams {
            batch_size: 64,
            concurrency: 8,
            overlap: Overlap::Full,
            loader_mode: LoaderMode::YieldOnce,
        },
    ]
}

fn no_overlap_batch_params() -> Vec<BatchParams> {
    // Disjoint batches should not benefit from single-flight. This matrix keeps
    // the loader cheap so the measured cost is dominated by per-key cache get,
    // registry claim/complete, validation, and result assembly.
    let mut params = Vec::new();
    for batch_size in BATCH_SIZES {
        for concurrency in CONCURRENCIES {
            params.push(BatchParams {
                batch_size,
                concurrency,
                overlap: Overlap::None,
                loader_mode: LoaderMode::Cheap,
            });
        }
    }

    // Retain a yielding case to keep waiter/scheduler overhead visible without
    // making the no-overlap group mostly a timer benchmark.
    params.push(BatchParams {
        batch_size: 64,
        concurrency: 8,
        overlap: Overlap::None,
        loader_mode: LoaderMode::YieldOnce,
    });
    params.push(BatchParams {
        batch_size: 512,
        concurrency: 32,
        overlap: Overlap::None,
        loader_mode: LoaderMode::YieldOnce,
    });

    params
}

async fn join_tasks<T>(handles: Vec<JoinHandle<T>>) -> Vec<T> {
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(handle.await.expect("benchmark task should not panic"));
    }
    results
}

fn assert_arc_values(values: &[Arc<usize>], keys: &[BenchKey]) {
    assert_eq!(values.len(), keys.len());
    for (value, key) in values.iter().zip(keys) {
        assert_eq!(**value, key.id);
        black_box(**value);
    }
}

fn assert_batch_values(values: &[CacheBatchValue<usize>], keys: &[BenchKey]) {
    assert_eq!(values.len(), keys.len());
    for (value, key) in values.iter().zip(keys) {
        assert_eq!(*value.value, key.id);
        black_box((*value.value, value.was_cached));
    }
}

async fn load_keys(
    keys: Vec<BenchKey>,
    counts: Arc<Vec<AtomicUsize>>,
    loader_calls: Arc<AtomicUsize>,
    loader_mode: LoaderMode,
) -> Result<Vec<(BenchKey, usize)>> {
    loader_calls.fetch_add(1, Ordering::SeqCst);
    loader_mode.wait().await;
    Ok(keys
        .into_iter()
        .map(|key| {
            counts[key.id].fetch_add(1, Ordering::SeqCst);
            let value = key.id;
            (key, value)
        })
        .collect())
}

async fn run_single_cold_concurrent_same_key(params: SingleParams) {
    let cache = make_cache();
    let key = BenchKey::new(0);
    let loader_calls = Arc::new(AtomicUsize::new(0));
    // Start tasks together so a cold same-key miss actually exercises the
    // owner/waiter path instead of mostly measuring warm cache hits.
    let barrier = Arc::new(Barrier::new(params.concurrency));
    let mut handles = Vec::with_capacity(params.concurrency);

    for _ in 0..params.concurrency {
        let cache = cache.clone();
        let key = key.clone();
        let loader_calls = loader_calls.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            cache
                .get_or_insert_with_key(key, move || {
                    let loader_calls = loader_calls.clone();
                    async move {
                        loader_calls.fetch_add(1, Ordering::SeqCst);
                        params.loader_mode.wait().await;
                        Ok(7usize)
                    }
                })
                .await
        }));
    }

    for value in join_tasks(handles).await {
        assert_eq!(*value.expect("single get_or_insert should succeed"), 7);
    }
    assert_eq!(loader_calls.load(Ordering::SeqCst), 1);
    black_box(loader_calls.load(Ordering::SeqCst));
}

async fn run_batch_single_loop_overlap(params: BatchParams) {
    let cache = make_cache();
    let (batches, unique_key_count) = generate_batches(params);
    let counts = new_counts(unique_key_count);
    let loader_calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(params.concurrency));
    let mut handles = Vec::with_capacity(params.concurrency);

    for keys in batches {
        let cache = cache.clone();
        let counts = counts.clone();
        let loader_calls = loader_calls.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            let mut values = Vec::with_capacity(keys.len());
            for key in &keys {
                let value = cache
                    .get_or_insert_with_key(key.clone(), {
                        let key = key.clone();
                        let counts = counts.clone();
                        let loader_calls = loader_calls.clone();
                        move || {
                            let counts = counts.clone();
                            let loader_calls = loader_calls.clone();
                            let key = key.clone();
                            async move {
                                loader_calls.fetch_add(1, Ordering::SeqCst);
                                params.loader_mode.wait().await;
                                counts[key.id].fetch_add(1, Ordering::SeqCst);
                                Ok(key.id)
                            }
                        }
                    })
                    .await
                    .expect("single-loop get_or_insert should succeed");
                values.push(value);
            }
            assert_arc_values(&values, &keys);
        }));
    }

    join_tasks(handles).await;
    // The per-key workaround should still deduplicate each key, but it can only
    // invoke the loader one key at a time, so loader_calls == unique_key_count.
    assert_each_key_loaded_once(&counts);
    assert_eq!(loader_calls.load(Ordering::SeqCst), unique_key_count);
    black_box(loader_calls.load(Ordering::SeqCst));
}

async fn run_batch_manual_get_load_insert_overlap(params: BatchParams) {
    run_batch_manual_get_load_insert(params, true).await;
}

async fn run_batch_manual_get_load_insert_no_overlap(params: BatchParams) {
    run_batch_manual_get_load_insert(params, false).await;
}

async fn run_batch_manual_get_load_insert(params: BatchParams, force_all_gets_before_insert: bool) {
    let cache = make_cache();
    let (batches, unique_key_count) = generate_batches(params);
    let counts = new_counts(unique_key_count);
    let loader_calls = Arc::new(AtomicUsize::new(0));
    let start_barrier = Arc::new(Barrier::new(params.concurrency));
    // Overlap benchmarks force all tasks to finish the initial get phase before
    // any task inserts. This reliably exposes duplicate cold loads without
    // adding the same synchronization cost to the no-overlap baseline.
    let after_get_barrier =
        force_all_gets_before_insert.then(|| Arc::new(Barrier::new(params.concurrency)));
    let mut handles = Vec::with_capacity(params.concurrency);

    for keys in batches {
        let cache = cache.clone();
        let counts = counts.clone();
        let loader_calls = loader_calls.clone();
        let start_barrier = start_barrier.clone();
        let after_get_barrier = after_get_barrier.clone();
        handles.push(tokio::spawn(async move {
            start_barrier.wait().await;
            let mut values = Vec::with_capacity(keys.len());
            let mut missing = Vec::with_capacity(keys.len());

            for key in &keys {
                if let Some(value) = cache.get_with_key(key).await {
                    values.push(Some(value));
                } else {
                    values.push(None);
                    missing.push(key.clone());
                }
            }

            if let Some(after_get_barrier) = after_get_barrier {
                after_get_barrier.wait().await;
            }
            let loaded = load_keys(
                missing,
                counts.clone(),
                loader_calls.clone(),
                params.loader_mode,
            )
            .await
            .expect("manual batch loader should succeed");

            let mut loaded_by_id = HashMap::with_capacity(loaded.len());
            for (key, value) in loaded {
                let value = Arc::new(value);
                cache.insert_with_key(&key, value.clone()).await;
                loaded_by_id.insert(key.id, value);
            }

            let values = values
                .into_iter()
                .zip(&keys)
                .map(|(value, key)| {
                    value.unwrap_or_else(|| {
                        loaded_by_id
                            .remove(&key.id)
                            .expect("manual path should fill each value")
                    })
                })
                .collect::<Vec<_>>();
            assert_arc_values(&values, &keys);
        }));
    }

    join_tasks(handles).await;
    let loaded_entries = sum_counts(&counts);
    // In this workaround every task batch loads its own missing set. Overlap
    // intentionally duplicates shared keys; no-overlap is the coalesced-loader
    // baseline without backend flight-registry work.
    assert_eq!(loaded_entries, params.concurrency * params.batch_size);
    match params.overlap {
        Overlap::None => assert_eq!(loaded_entries, unique_key_count),
        Overlap::Half | Overlap::Full if params.concurrency > 1 => {
            assert!(
                loaded_entries > unique_key_count,
                "manual get/load/insert should duplicate overlapping cold loads"
            );
        }
        Overlap::Half | Overlap::Full => {}
    }
    black_box((loader_calls.load(Ordering::SeqCst), loaded_entries));
}

async fn run_batch_get_or_insert_many(params: BatchParams) {
    let cache = make_cache();
    let (batches, unique_key_count) = generate_batches(params);
    let counts = new_counts(unique_key_count);
    let loader_calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(params.concurrency));
    let mut handles = Vec::with_capacity(params.concurrency);

    for keys in batches {
        let cache = cache.clone();
        let counts = counts.clone();
        let loader_calls = loader_calls.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            let values =
                cache
                    .get_or_insert_with_key_batch(keys.clone(), move |owned_keys| {
                        let counts = counts.clone();
                        let loader_calls = loader_calls.clone();
                        async move {
                            load_keys(owned_keys, counts, loader_calls, params.loader_mode).await
                        }
                    })
                    .await
                    .expect("batch get_or_insert should succeed");
            assert_batch_values(&values, &keys);
        }));
    }

    join_tasks(handles).await;
    // New backend-level batch get-or-insert should keep coalesced loading while
    // still loading each logical key at most once across overlapping tasks.
    assert_each_key_loaded_once(&counts);
    assert!(loader_calls.load(Ordering::SeqCst) <= params.concurrency);
    black_box(loader_calls.load(Ordering::SeqCst));
}

fn bench_single_get_or_insert_hot(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/single_get_or_insert_hot");
    let cache = make_cache();
    let key = BenchKey::new(0);

    runtime.block_on(async {
        cache.insert_with_key(&key, Arc::new(7usize)).await;
    });

    group.bench_function("prefilled_key", |b| {
        b.to_async(&runtime).iter(|| {
            let cache = cache.clone();
            let key = key.clone();
            async move {
                let value = cache
                    .get_or_insert_with_key(key, || async {
                        unreachable!("hot cache benchmark loader should not run")
                    })
                    .await
                    .expect("hot get_or_insert should succeed");
                assert_eq!(*value, 7);
                black_box(*value);
            }
        });
    });
    group.finish();
}

fn bench_single_get_or_insert_cold_unique_key(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/single_get_or_insert_cold_unique_key");

    for loader_mode in [LoaderMode::Cheap, LoaderMode::YieldOnce] {
        group.bench_with_input(
            BenchmarkId::new("loader", loader_mode.name()),
            &loader_mode,
            |b, loader_mode| {
                b.to_async(&runtime).iter_custom(|iters| {
                    let loader_mode = *loader_mode;
                    async move {
                        let iter_count = usize::try_from(iters)
                            .expect("criterion iteration count should fit usize");
                        let cache = make_cache();
                        let keys = (0..iter_count).map(BenchKey::new).collect::<Vec<_>>();
                        let loader_calls = Arc::new(AtomicUsize::new(0));

                        // Setup above is intentionally outside the measured
                        // interval. This case isolates an uncontended cold miss:
                        // cache miss, flight claim/complete, loader execution,
                        // and cache insert for a key that no other task touches.
                        let start = Instant::now();
                        for key in keys {
                            let expected = key.id;
                            let value = cache
                                .get_or_insert_with_key(key, {
                                    let loader_calls = loader_calls.clone();
                                    move || {
                                        let loader_calls = loader_calls.clone();
                                        async move {
                                            loader_calls.fetch_add(1, Ordering::SeqCst);
                                            loader_mode.wait().await;
                                            Ok(expected)
                                        }
                                    }
                                })
                                .await
                                .expect("cold unique get_or_insert should succeed");
                            assert_eq!(*value, expected);
                            black_box(*value);
                        }
                        let elapsed = start.elapsed();

                        assert_eq!(loader_calls.load(Ordering::SeqCst), iter_count);
                        black_box(loader_calls.load(Ordering::SeqCst));
                        elapsed
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_single_get_or_insert_cold_concurrent_same_key(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/single_get_or_insert_cold_concurrent_same_key");

    for concurrency in CONCURRENCIES {
        for loader_mode in [LoaderMode::Cheap, LoaderMode::YieldOnce] {
            let params = SingleParams {
                concurrency,
                loader_mode,
            };
            group.bench_with_input(BenchmarkId::from_parameter(params), &params, |b, params| {
                b.to_async(&runtime)
                    .iter(|| run_single_cold_concurrent_same_key(*params));
            });
        }
    }
    group.finish();
}

fn bench_batch_single_loop_overlap(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/batch_single_loop_overlap");

    for params in overlapping_batch_params() {
        group.bench_with_input(BenchmarkId::from_parameter(params), &params, |b, params| {
            b.to_async(&runtime)
                .iter(|| run_batch_single_loop_overlap(*params));
        });
    }
    group.finish();
}

fn bench_batch_manual_get_load_insert_overlap(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/batch_manual_get_load_insert_overlap");

    for params in overlapping_batch_params() {
        group.bench_with_input(BenchmarkId::from_parameter(params), &params, |b, params| {
            b.to_async(&runtime)
                .iter(|| run_batch_manual_get_load_insert_overlap(*params));
        });
    }
    group.finish();
}

fn bench_batch_manual_get_load_insert_no_overlap(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/batch_manual_get_load_insert_no_overlap");

    for params in no_overlap_batch_params() {
        group.bench_with_input(BenchmarkId::from_parameter(params), &params, |b, params| {
            b.to_async(&runtime)
                .iter(|| run_batch_manual_get_load_insert_no_overlap(*params));
        });
    }
    group.finish();
}

fn bench_batch_get_or_insert_many_overlap(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/batch_get_or_insert_many_overlap");

    for params in overlapping_batch_params() {
        group.bench_with_input(BenchmarkId::from_parameter(params), &params, |b, params| {
            b.to_async(&runtime)
                .iter(|| run_batch_get_or_insert_many(*params));
        });
    }
    group.finish();
}

fn bench_batch_get_or_insert_many_no_overlap(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should start");
    let mut group = c.benchmark_group("cache_batch/batch_get_or_insert_many_no_overlap");

    for params in no_overlap_batch_params() {
        group.bench_with_input(BenchmarkId::from_parameter(params), &params, |b, params| {
            b.to_async(&runtime)
                .iter(|| run_batch_get_or_insert_many(*params));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        bench_single_get_or_insert_hot,
        bench_single_get_or_insert_cold_unique_key,
        bench_single_get_or_insert_cold_concurrent_same_key,
        bench_batch_single_loop_overlap,
        bench_batch_manual_get_load_insert_overlap,
        bench_batch_manual_get_load_insert_no_overlap,
        bench_batch_get_or_insert_many_overlap,
        bench_batch_get_or_insert_many_no_overlap
}
criterion_main!(benches);
