// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for manifest commit performance with many small fragments.
//!
//! This benchmark tests how performance degrades as the number of small fragments
//! grows. Each fragment contains only 10 rows, and we measure both:
//! - Commit time (manifest write only, excludes fragment data writing)
//! - Load time (manifest read from storage, no caching)
//!
//! Key optimizations:
//! - Uses shared session for commits to avoid re-reading old manifests
//! - Disables auto-cleanup to avoid background cleanup overhead
//! - Separates fragment writing from commit measurement
//! - Uses fresh session (no cache) for load measurement to force actual storage read
//!
//! ## Running against S3
//!
//! ```bash
//! export AWS_DEFAULT_REGION=us-east-1
//! export DATASET_PREFIX=s3://your-bucket/bench/manifest_commit
//! export NUM_ITERATIONS=100
//! cargo bench --bench manifest_commit
//! ```
//!
//! ## Running against local filesystem (with temp directory)
//!
//! ```bash
//! cargo bench --bench manifest_commit
//! ```
//!
//! ## Running against specific local directory
//!
//! ```bash
//! export DATASET_PREFIX=/tmp/bench/manifest_commit
//! export NUM_ITERATIONS=50
//! cargo bench --bench manifest_commit
//! ```
//!
//! ## Configuration
//!
//! - `DATASET_PREFIX`: Base URI for datasets (optional, e.g. s3://bucket/prefix or /tmp/bench). If not set, uses a temporary directory.
//! - `NUM_ITERATIONS`: Number of small fragment writes to perform (default: 100).
//! - `ROWS_PER_FRAGMENT`: Number of rows per fragment (default: 10).
//! - `DIRECT_CHECKOUT`: When "true", use checkout_version which bypasses listing. When "false" (default), use load() which includes listing.
//! - `DELETE_DATASET`: When "true", delete the dataset after benchmark completes. When "false" (default), keep the dataset for inspection.
//! - `WARM_SESSION`: When "true", use the same shared session for load() to test warm connection performance. When "false" (default), use fresh session for each load.

#![allow(clippy::print_stdout)]

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{CommitBuilder, Dataset, InsertBuilder, WriteMode, WriteParams};
use lance::session::Session;
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;
use uuid::Uuid;

const DEFAULT_ROWS_PER_FRAGMENT: usize = 10;
const DEFAULT_NUM_ITERATIONS: usize = 100;

fn get_rows_per_fragment() -> usize {
    std::env::var("ROWS_PER_FRAGMENT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ROWS_PER_FRAGMENT)
}

fn get_num_iterations() -> usize {
    std::env::var("NUM_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_ITERATIONS)
}

fn get_direct_checkout() -> bool {
    std::env::var("DIRECT_CHECKOUT")
        .map(|s| s.to_lowercase() == "true")
        .unwrap_or(false)
}

fn get_delete_dataset() -> bool {
    std::env::var("DELETE_DATASET")
        .map(|s| s.to_lowercase() == "true")
        .unwrap_or(false)
}

fn get_warm_session() -> bool {
    std::env::var("WARM_SESSION")
        .map(|s| s.to_lowercase() == "true")
        .unwrap_or(false)
}

fn get_dataset_prefix() -> String {
    std::env::var("DATASET_PREFIX").unwrap_or_else(|_| {
        let temp_dir = std::env::temp_dir().join(format!("lance_bench_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");
        temp_dir.to_string_lossy().to_string()
    })
}

fn get_storage_label(prefix: &str) -> &'static str {
    if prefix.starts_with("s3://") {
        "s3"
    } else if prefix.starts_with("gs://") {
        "gcs"
    } else if prefix.starts_with("az://") {
        "azure"
    } else if prefix.starts_with("memory://") {
        "memory"
    } else {
        "local"
    }
}

async fn create_initial_dataset(
    uri: &str,
    rows_per_fragment: usize,
    session: Arc<Session>,
) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = create_batch(schema.clone(), 0, rows_per_fragment);
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

    std::fs::remove_dir_all(uri).ok();

    let params = WriteParams {
        session: Some(session),
        skip_auto_cleanup: true,
        ..Default::default()
    };

    Dataset::write(reader, uri, Some(params))
        .await
        .expect("failed to create initial dataset")
}

fn create_batch(schema: Arc<ArrowSchema>, start_id: usize, num_rows: usize) -> RecordBatch {
    let ids = Int64Array::from_iter_values((start_id as i64)..((start_id + num_rows) as i64));
    let names = StringArray::from_iter_values(
        (start_id..(start_id + num_rows)).map(|i| format!("name_{}", i)),
    );

    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(names)])
        .expect("failed to create batch")
}

fn bench_manifest_commit(c: &mut Criterion) {
    let runtime = Runtime::new().expect("failed to build tokio runtime");

    let dataset_prefix = get_dataset_prefix();
    let num_iterations = get_num_iterations();
    let rows_per_fragment = get_rows_per_fragment();
    let direct_checkout = get_direct_checkout();
    let delete_dataset = get_delete_dataset();
    let warm_session = get_warm_session();
    let storage_label = get_storage_label(&dataset_prefix);

    let short_id = &Uuid::new_v4().to_string()[..8];
    let uri = format!(
        "{}/manifest_commit_{}",
        dataset_prefix.trim_end_matches('/'),
        short_id
    );

    println!("=== Manifest Commit Benchmark Setup ===");
    println!("Storage: {} ({})", uri, storage_label);
    println!("Rows per fragment: {}", rows_per_fragment);
    println!("Number of iterations: {}", num_iterations);
    println!(
        "Total fragments (including initial): {}",
        num_iterations + 1
    );
    println!(
        "Direct checkout: {} ({})",
        direct_checkout,
        if direct_checkout {
            "checkout_version - bypasses listing"
        } else {
            "load() - includes listing"
        }
    );
    println!("Delete dataset: {}", delete_dataset);
    println!(
        "Warm session: {} ({})",
        warm_session,
        if warm_session {
            "reuse session for load - tests warm connection"
        } else {
            "fresh session for load - tests cold start"
        }
    );
    println!();

    // Create a shared session to avoid re-opening old manifests
    let session = Arc::new(Session::default());

    let initial_dataset = runtime.block_on(create_initial_dataset(
        &uri,
        rows_per_fragment,
        session.clone(),
    ));

    // Keep a mutable dataset reference that we update after each commit
    let mut current_dataset = Arc::new(initial_dataset);

    let mut commit_latencies = Vec::with_capacity(num_iterations);
    let mut load_latencies = Vec::with_capacity(num_iterations);

    println!("Running commit and load benchmarks...");
    println!("fragments,commit_ms,load_ms");

    for i in 1..=num_iterations {
        let num_fragments = i + 1;

        let (commit_time, new_dataset) = {
            let dataset = current_dataset.clone();
            let session_clone = session.clone();
            runtime.block_on(async move {
                let schema: Arc<ArrowSchema> = Arc::new((&dataset.schema().clone()).into());
                let start_id = dataset.count_rows(None).await.unwrap() as usize;
                let batch = create_batch(schema.clone(), start_id, rows_per_fragment);

                let write_params = WriteParams {
                    mode: WriteMode::Append,
                    session: Some(session_clone.clone()),
                    skip_auto_cleanup: true,
                    ..Default::default()
                };

                // Write fragments without committing (not measured)
                let transaction = InsertBuilder::new(dataset.clone())
                    .with_params(&write_params)
                    .execute_uncommitted(vec![batch])
                    .await
                    .expect("failed to write fragment");

                // Measure only the commit time
                let start = Instant::now();
                let new_ds = CommitBuilder::new(dataset)
                    .with_session(session_clone)
                    .with_skip_auto_cleanup(true)
                    .execute(transaction)
                    .await
                    .expect("failed to commit");
                (start.elapsed(), Arc::new(new_ds))
            })
        };

        // Small delay to let fire-and-forget hint write complete
        // This avoids the hint write from previous commit interfering with load measurement
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Measure load time
        let load_time = if direct_checkout {
            // Direct checkout: use checkout_version which bypasses listing
            let dataset = current_dataset.clone();
            let new_version = (i + 1) as u64;
            runtime.block_on(async move {
                let start = Instant::now();
                let checked_out = dataset
                    .checkout_version(new_version)
                    .await
                    .expect("failed to checkout");
                let elapsed = start.elapsed();

                assert_eq!(
                    checked_out.manifest().fragments.len(),
                    num_fragments,
                    "Expected {} fragments",
                    num_fragments
                );
                elapsed
            })
        } else {
            // Load dataset
            let uri_ref = uri.as_str();
            let session_for_load = if warm_session {
                Some(session.clone())
            } else {
                None
            };
            runtime.block_on(async move {
                let start = Instant::now();
                let mut builder = DatasetBuilder::from_uri(uri_ref);
                if let Some(s) = session_for_load {
                    builder = builder.with_session(s);
                }
                let dataset = builder.load().await.expect("failed to load");
                let elapsed = start.elapsed();

                assert_eq!(
                    dataset.manifest().fragments.len(),
                    num_fragments,
                    "Expected {} fragments",
                    num_fragments
                );
                elapsed
            })
        };

        // Update current_dataset for next iteration
        current_dataset = new_dataset;

        commit_latencies.push(commit_time);
        load_latencies.push(load_time);

        println!(
            "{},{:.2},{:.2}",
            num_fragments,
            commit_time.as_secs_f64() * 1000.0,
            load_time.as_secs_f64() * 1000.0
        );
    }

    println!();
    println!("=== Summary Statistics ===");

    let avg_commit: f64 = commit_latencies
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / commit_latencies.len() as f64;
    let avg_load: f64 =
        load_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>() / load_latencies.len() as f64;

    let min_commit = commit_latencies.iter().min().unwrap();
    let max_commit = commit_latencies.iter().max().unwrap();
    let min_load = load_latencies.iter().min().unwrap();
    let max_load = load_latencies.iter().max().unwrap();

    println!(
        "Commit latency: avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_commit * 1000.0,
        min_commit.as_secs_f64() * 1000.0,
        max_commit.as_secs_f64() * 1000.0
    );
    println!(
        "Load latency:   avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_load * 1000.0,
        min_load.as_secs_f64() * 1000.0,
        max_load.as_secs_f64() * 1000.0
    );

    let first_10_avg_commit = commit_latencies
        .iter()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let last_10_avg_commit = commit_latencies
        .iter()
        .rev()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let first_10_avg_load = load_latencies
        .iter()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let last_10_avg_load = load_latencies
        .iter()
        .rev()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;

    println!();
    println!(
        "First 10 iterations avg: commit={:.2}ms, load={:.2}ms",
        first_10_avg_commit * 1000.0,
        first_10_avg_load * 1000.0
    );
    println!(
        "Last 10 iterations avg:  commit={:.2}ms, load={:.2}ms",
        last_10_avg_commit * 1000.0,
        last_10_avg_load * 1000.0
    );
    println!(
        "Degradation ratio: commit={:.2}x, load={:.2}x",
        last_10_avg_commit / first_10_avg_commit,
        last_10_avg_load / first_10_avg_load
    );

    let mut group = c.benchmark_group("manifest_commit");

    group.bench_function("avg_commit_latency", |b| {
        b.iter(|| std::time::Duration::from_secs_f64(avg_commit))
    });

    group.bench_function("avg_load_latency", |b| {
        b.iter(|| std::time::Duration::from_secs_f64(avg_load))
    });

    group.finish();

    if delete_dataset {
        std::fs::remove_dir_all(&uri).ok();
        println!("Dataset deleted: {}", uri);
    } else {
        println!("Dataset preserved: {}", uri);
    }
}

criterion_group!(benches, bench_manifest_commit);
criterion_main!(benches);
