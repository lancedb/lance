// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for Lance queue producer write throughput.
//!
//! ## Running against S3
//!
//! ```bash
//! export AWS_DEFAULT_REGION=us-east-1
//! export DATASET_PREFIX=s3://your-bucket/bench/lance_queue
//! cargo bench -p lance-queue --bench queue_write
//! ```
//!
//! ## Running against local filesystem
//!
//! ```bash
//! cargo bench -p lance-queue --bench queue_write
//! ```
//!
//! ## Configuration
//!
//! - `DATASET_PREFIX`: Base URI for queue tables. If not set, uses a temporary local directory.
//! - `PARTITION_COUNTS`: Comma-separated partition counts to benchmark (default: `1,4,16`).
//! - `BATCH_SIZE`: Number of messages per producer send batch (default: `100`).
//! - `NUM_BATCHES`: Number of producer send batches per benchmark iteration (default: `100`).
//! - `PAYLOAD_BYTES`: Approximate bytes in each JSON payload body string (default: `256`).
//! - `SAMPLE_SIZE`: Number of Criterion samples (default: `10`, minimum: `10`).

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lance_queue::Queue;
use serde_json::{Value, json};
use uuid::Uuid;

const DEFAULT_PARTITION_COUNTS: &str = "1,4,16";
const DEFAULT_BATCH_SIZE: usize = 100;
const DEFAULT_NUM_BATCHES: usize = 100;
const DEFAULT_PAYLOAD_BYTES: usize = 256;

#[derive(Debug, Clone)]
struct InputBatch {
    ids: Vec<String>,
    payloads: Vec<Value>,
}

impl InputBatch {
    fn approximate_bytes(&self) -> u64 {
        self.ids.iter().map(|id| id.len() as u64).sum::<u64>()
            + self
                .payloads
                .iter()
                .map(|payload| payload.to_string().len() as u64)
                .sum::<u64>()
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn get_sample_size() -> usize {
    env_usize("SAMPLE_SIZE", 10).max(10)
}

fn get_partition_counts() -> Vec<u32> {
    let raw =
        std::env::var("PARTITION_COUNTS").unwrap_or_else(|_| DEFAULT_PARTITION_COUNTS.to_string());
    let partition_counts = raw
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return None;
            }
            match trimmed.parse::<u32>() {
                Ok(value) if value > 0 => Some(value),
                _ => {
                    eprintln!(
                        "Ignoring invalid PARTITION_COUNTS value '{}'; values must be positive integers",
                        trimmed
                    );
                    None
                }
            }
        })
        .collect::<Vec<_>>();
    if partition_counts.is_empty() {
        vec![1]
    } else {
        partition_counts
    }
}

fn get_dataset_prefix() -> String {
    std::env::var("DATASET_PREFIX").unwrap_or_else(|_| {
        let temp_dir = std::env::temp_dir().join(format!("lance_queue_bench_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).expect("failed to create benchmark temp directory");
        temp_dir.to_string_lossy().to_string()
    })
}

fn storage_label(prefix: &str) -> &'static str {
    if prefix.starts_with("s3://") {
        "s3"
    } else if prefix.starts_with("gs://") {
        "gcs"
    } else if prefix.starts_with("az://") || prefix.starts_with("abfss://") {
        "azure"
    } else {
        "local"
    }
}

fn queue_uri(prefix: &str, partition_count: u32) -> String {
    format!(
        "{}/queue_write_p{}_{}",
        prefix.trim_end_matches('/'),
        partition_count,
        Uuid::new_v4()
    )
}

fn payload_body(payload_bytes: usize) -> String {
    "x".repeat(payload_bytes)
}

fn make_input_batches(
    batch_size: usize,
    num_batches: usize,
    payload_bytes: usize,
) -> Vec<InputBatch> {
    let body = payload_body(payload_bytes);
    (0..num_batches)
        .map(|batch_idx| {
            let ids = (0..batch_size)
                .map(|row_idx| format!("message-{batch_idx}-{row_idx}"))
                .collect::<Vec<_>>();
            let payloads = (0..batch_size)
                .map(|row_idx| {
                    json!({
                        "batch": batch_idx,
                        "row": row_idx,
                        "body": body,
                    })
                })
                .collect::<Vec<_>>();
            InputBatch { ids, payloads }
        })
        .collect()
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn format_throughput(bytes_per_second: f64) -> String {
    if bytes_per_second >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.2} GB/s", bytes_per_second / (1024.0 * 1024.0 * 1024.0))
    } else if bytes_per_second >= 1024.0 * 1024.0 {
        format!("{:.2} MB/s", bytes_per_second / (1024.0 * 1024.0))
    } else if bytes_per_second >= 1024.0 {
        format!("{:.2} KB/s", bytes_per_second / 1024.0)
    } else {
        format!("{:.0} B/s", bytes_per_second)
    }
}

fn bench_queue_write(c: &mut Criterion) {
    let dataset_prefix = get_dataset_prefix();
    let partition_counts = get_partition_counts();
    let batch_size = env_usize("BATCH_SIZE", DEFAULT_BATCH_SIZE);
    let num_batches = env_usize("NUM_BATCHES", DEFAULT_NUM_BATCHES);
    let payload_bytes = env_usize("PAYLOAD_BYTES", DEFAULT_PAYLOAD_BYTES);
    let sample_size = get_sample_size();
    let total_rows = batch_size * num_batches;
    let input_batches = Arc::new(make_input_batches(batch_size, num_batches, payload_bytes));
    let total_bytes = input_batches
        .iter()
        .map(InputBatch::approximate_bytes)
        .sum::<u64>();

    println!("=== Lance Queue Write Benchmark Setup ===");
    println!("Storage: {}", dataset_prefix);
    println!("Partition counts: {:?}", partition_counts);
    println!("Batch size: {} messages", batch_size);
    println!("Num batches: {}", num_batches);
    println!("Total messages per iteration: {}", total_rows);
    println!(
        "Approx payload per iteration: {}",
        format_bytes(total_bytes)
    );
    println!("Benchmark samples: {}", sample_size);
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Lance Queue Write");
    group.throughput(Throughput::Bytes(total_bytes));
    group.sample_size(sample_size);
    group.warm_up_time(Duration::from_secs(1));

    for partition_count in partition_counts {
        let uri = queue_uri(&dataset_prefix, partition_count);
        let queue = rt
            .block_on(
                Queue::builder()
                    .uri(&uri)
                    .partition_count(partition_count)
                    .create(),
            )
            .unwrap_or_else(|error| panic!("failed to create queue at {uri}: {error}"));
        let producer = queue.producer();
        let input_batches = input_batches.clone();
        let stats_printed = Arc::new(AtomicBool::new(false));
        let storage = storage_label(&dataset_prefix);
        let label = format!(
            "p{} {}x{} payload={}B ({})",
            partition_count, num_batches, batch_size, payload_bytes, storage
        );

        println!("Running: {}", label);
        group.bench_with_input(
            BenchmarkId::new("producer_send_batch", &label),
            &partition_count,
            |b, &_partition_count| {
                let producer = producer.clone();
                let input_batches = input_batches.clone();
                let stats_printed = stats_printed.clone();
                b.to_async(&rt).iter_custom(|iters| {
                    let producer = producer.clone();
                    let input_batches = input_batches.clone();
                    let stats_printed = stats_printed.clone();
                    async move {
                        let mut total_duration = Duration::ZERO;
                        for iter in 0..iters {
                            let start = Instant::now();
                            let mut wal_bytes = 0u64;
                            let mut wal_entries = 0usize;
                            for batch in input_batches.iter() {
                                let result = producer
                                    .send_batch(batch.ids.clone(), batch.payloads.clone())
                                    .await
                                    .unwrap();
                                wal_bytes += result
                                    .entries
                                    .iter()
                                    .map(|entry| entry.wal_bytes as u64)
                                    .sum::<u64>();
                                wal_entries += result.entries.len();
                            }
                            let elapsed = start.elapsed();
                            total_duration += elapsed;

                            if iter == 0 && !stats_printed.swap(true, Ordering::SeqCst) {
                                println!(
                                    "  First sample: {} messages in {:?} ({:.0} msg/s, {})",
                                    total_rows,
                                    elapsed,
                                    total_rows as f64 / elapsed.as_secs_f64(),
                                    format_throughput(total_bytes as f64 / elapsed.as_secs_f64())
                                );
                                println!(
                                    "  WAL entries: {}, WAL bytes: {}",
                                    wal_entries,
                                    format_bytes(wal_bytes)
                                );
                            }
                        }
                        total_duration
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05);
    targets = bench_queue_write
);

criterion_main!(benches);
