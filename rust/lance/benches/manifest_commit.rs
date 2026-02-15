// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for manifest commit performance with many small fragments.
//!
//! This benchmark tests how performance degrades as the number of small fragments
//! grows. Each fragment contains only 10 rows, and we measure both:
//! - Commit time (write + manifest update)
//! - Dataset load time (opening the manifest)
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

#![allow(clippy::print_stdout)]

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::dataset::{Dataset, WriteMode, WriteParams};
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

async fn create_initial_dataset(uri: &str, rows_per_fragment: usize) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = create_batch(schema.clone(), 0, rows_per_fragment);
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

    std::fs::remove_dir_all(uri).ok();

    Dataset::write(reader, uri, None)
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

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|v| v * v).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (slope, intercept)
}

fn bench_manifest_commit(c: &mut Criterion) {
    let runtime = Runtime::new().expect("failed to build tokio runtime");

    let dataset_prefix = get_dataset_prefix();
    let num_iterations = get_num_iterations();
    let rows_per_fragment = get_rows_per_fragment();
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
    println!("Total fragments (including initial): {}", num_iterations + 1);
    println!();

    runtime.block_on(create_initial_dataset(&uri, rows_per_fragment));

    let mut write_latencies = Vec::with_capacity(num_iterations);
    let mut load_latencies = Vec::with_capacity(num_iterations);

    println!("Running write and load benchmarks...");
    println!("fragments,write_ms,load_ms");

    for i in 1..=num_iterations {
        let num_fragments = i + 1;

        let write_time = {
            let uri_ref = uri.as_str();
            runtime.block_on(async move {
                let dataset = Dataset::open(uri_ref).await.expect("failed to open dataset");
                let schema: Arc<ArrowSchema> = Arc::new((&dataset.schema().clone()).into());
                let start_id = dataset.count_rows(None).await.unwrap() as usize;
                let batch = create_batch(schema.clone(), start_id, rows_per_fragment);
                let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

                let write_params = WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                };

                let start = Instant::now();
                Dataset::write(reader, uri_ref, Some(write_params))
                    .await
                    .expect("failed to append");
                start.elapsed()
            })
        };

        let load_time = {
            let uri_ref = uri.as_str();
            runtime.block_on(async move {
                let start = Instant::now();
                let dataset = Dataset::open(uri_ref).await.expect("failed to open");
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

        write_latencies.push(write_time);
        load_latencies.push(load_time);

        println!(
            "{},{:.2},{:.2}",
            num_fragments,
            write_time.as_secs_f64() * 1000.0,
            load_time.as_secs_f64() * 1000.0
        );
    }

    println!();
    println!("=== Summary Statistics ===");

    let avg_write: f64 = write_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>()
        / write_latencies.len() as f64;
    let avg_load: f64 = load_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>()
        / load_latencies.len() as f64;

    let min_write = write_latencies.iter().min().unwrap();
    let max_write = write_latencies.iter().max().unwrap();
    let min_load = load_latencies.iter().min().unwrap();
    let max_load = load_latencies.iter().max().unwrap();

    println!("Write latency: avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_write * 1000.0, min_write.as_secs_f64() * 1000.0, max_write.as_secs_f64() * 1000.0);
    println!("Load latency:  avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_load * 1000.0, min_load.as_secs_f64() * 1000.0, max_load.as_secs_f64() * 1000.0);

    let fragment_counts: Vec<f64> = (2..=(num_iterations + 1)).map(|x| x as f64).collect();
    let write_ms: Vec<f64> = write_latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let load_ms: Vec<f64> = load_latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

    let (write_slope, write_intercept) = linear_regression(&fragment_counts, &write_ms);
    let (load_slope, load_intercept) = linear_regression(&fragment_counts, &load_ms);

    println!();
    println!("=== Linear Regression Analysis ===");
    println!("Write latency = {:.4}ms + {:.4}ms * fragments", write_intercept, write_slope);
    println!("Load latency  = {:.4}ms + {:.4}ms * fragments", load_intercept, load_slope);
    println!();
    println!("Per-fragment overhead:");
    println!("  Write: {:.4}ms per additional fragment", write_slope);
    println!("  Load:  {:.4}ms per additional fragment", load_slope);

    let first_10_avg_write = write_latencies.iter().take(10).map(|d| d.as_secs_f64()).sum::<f64>() / 10.0;
    let last_10_avg_write = write_latencies.iter().rev().take(10).map(|d| d.as_secs_f64()).sum::<f64>() / 10.0;
    let first_10_avg_load = load_latencies.iter().take(10).map(|d| d.as_secs_f64()).sum::<f64>() / 10.0;
    let last_10_avg_load = load_latencies.iter().rev().take(10).map(|d| d.as_secs_f64()).sum::<f64>() / 10.0;

    println!();
    println!("First 10 iterations avg: write={:.2}ms, load={:.2}ms",
        first_10_avg_write * 1000.0, first_10_avg_load * 1000.0);
    println!("Last 10 iterations avg:  write={:.2}ms, load={:.2}ms",
        last_10_avg_write * 1000.0, last_10_avg_load * 1000.0);
    println!("Degradation ratio: write={:.2}x, load={:.2}x",
        last_10_avg_write / first_10_avg_write,
        last_10_avg_load / first_10_avg_load);

    let mut group = c.benchmark_group("manifest_commit");

    group.bench_function("avg_write_latency", |b| {
        b.iter(|| std::time::Duration::from_secs_f64(avg_write))
    });

    group.bench_function("avg_load_latency", |b| {
        b.iter(|| std::time::Duration::from_secs_f64(avg_load))
    });

    group.bench_function("write_slope_per_fragment", |b| {
        b.iter(|| write_slope)
    });

    group.bench_function("load_slope_per_fragment", |b| {
        b.iter(|| load_slope)
    });

    group.finish();

    std::fs::remove_dir_all(&uri).ok();
}

criterion_group!(benches, bench_manifest_commit);
criterion_main!(benches);
