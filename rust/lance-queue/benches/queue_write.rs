// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Matrix benchmark for Lance queue producer write throughput.
//!
//! The benchmark writes CSV rows to stdout and, when `RESULT_CSV` is set, to a
//! file. It is intended for longer S3 runs where we want trends across
//! partition counts, row counts, and producer batch sizes.
//!
//! ## Running against S3
//!
//! ```bash
//! export AWS_DEFAULT_REGION=us-east-1
//! export DATASET_PREFIX=s3://your-bucket/bench/lance_queue
//! export RESULT_CSV=/tmp/lance_queue_write.csv
//! cargo bench -p lance-queue --bench queue_write
//! ```
//!
//! ## Configuration
//!
//! - `DATASET_PREFIX`: Base URI for queue tables. If not set, uses a temporary local directory.
//! - `PAYLOAD_BYTES`: Approximate bytes in each JSON payload body string (default: `256`).
//! - `REPEATS`: Repeated measurements per scenario (default: `3`).
//! - `WRITE_CASES`: Optional explicit cases as `name:partitions:rows:batch_size` separated by `;`.
//! - `RESULT_CSV`: Optional output CSV file path.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Duration, Instant};

use lance_core::Result;
use lance_queue::Queue;
use serde_json::{Value, json};
use uuid::Uuid;

const DEFAULT_PAYLOAD_BYTES: usize = 256;
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_CASES: &[(&str, u32, usize, usize)] = &[
    ("horizontal_p1", 1, 500_000, 5_000),
    ("horizontal_p2", 2, 500_000, 10_000),
    ("horizontal_p4", 4, 500_000, 20_000),
    ("horizontal_p8", 8, 500_000, 40_000),
    ("horizontal_p16", 16, 500_000, 80_000),
    ("horizontal_p32", 32, 500_000, 160_000),
    ("horizontal_p64", 64, 500_000, 320_000),
    ("trend_50k_p16", 16, 50_000, 80_000),
    ("trend_200k_p16", 16, 200_000, 80_000),
    ("trend_500k_p16", 16, 500_000, 80_000),
    ("batch_1_p1", 1, 20_000, 1),
    ("batch_10_p1", 1, 20_000, 10),
    ("batch_100_p1", 1, 20_000, 100),
    ("batch_1000_p1", 1, 20_000, 1_000),
    ("batch_5000_p1", 1, 20_000, 5_000),
    ("batch_10000_p1", 1, 20_000, 10_000),
    ("max_500k_batch5000_p1", 1, 500_000, 5_000),
    ("max_500k_batch10000_p1", 1, 500_000, 10_000),
];

#[derive(Debug, Clone)]
struct WriteCase {
    name: String,
    partition_count: u32,
    rows: usize,
    batch_size: usize,
}

#[derive(Debug, Clone)]
struct InputBatch {
    ids: Vec<String>,
    payloads: Vec<Value>,
}

#[derive(Debug, Clone)]
struct WriteMeasurement {
    case_name: String,
    partition_count: u32,
    rows: usize,
    batch_size: usize,
    payload_bytes: usize,
    repeat: usize,
    elapsed: Duration,
    input_bytes: u64,
    wal_bytes: u64,
    wal_entries: usize,
}

impl WriteMeasurement {
    fn rows_per_second(&self) -> f64 {
        self.rows as f64 / self.elapsed.as_secs_f64()
    }

    fn input_mib_per_second(&self) -> f64 {
        self.input_bytes as f64 / self.elapsed.as_secs_f64() / (1024.0 * 1024.0)
    }

    fn wal_mib_per_second(&self) -> f64 {
        self.wal_bytes as f64 / self.elapsed.as_secs_f64() / (1024.0 * 1024.0)
    }

    fn csv_header() -> &'static str {
        "benchmark,case,partition_count,rows,batch_size,payload_bytes,repeat,elapsed_seconds,rows_per_second,input_mib_per_second,wal_mib_per_second,wal_bytes,wal_entries"
    }

    fn csv_row(&self) -> String {
        format!(
            "write,{},{},{},{},{},{},{:.6},{:.3},{:.3},{:.3},{},{}",
            self.case_name,
            self.partition_count,
            self.rows,
            self.batch_size,
            self.payload_bytes,
            self.repeat,
            self.elapsed.as_secs_f64(),
            self.rows_per_second(),
            self.input_mib_per_second(),
            self.wal_mib_per_second(),
            self.wal_bytes,
            self.wal_entries
        )
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn get_dataset_prefix() -> String {
    std::env::var("DATASET_PREFIX").unwrap_or_else(|_| {
        let temp_dir = std::env::temp_dir().join(format!("lance_queue_bench_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).expect("failed to create benchmark temp directory");
        temp_dir.to_string_lossy().to_string()
    })
}

fn parse_cases() -> Vec<WriteCase> {
    if let Ok(raw) = std::env::var("WRITE_CASES") {
        let parsed = raw
            .split(';')
            .filter_map(|case| {
                let parts = case.split(':').collect::<Vec<_>>();
                if parts.len() != 4 {
                    eprintln!("Ignoring invalid WRITE_CASES entry '{case}'");
                    return None;
                }
                let partition_count = parts[1].parse::<u32>().ok()?;
                let rows = parts[2].parse::<usize>().ok()?;
                let batch_size = parts[3].parse::<usize>().ok()?;
                if partition_count == 0 || rows == 0 || batch_size == 0 {
                    eprintln!("Ignoring non-positive WRITE_CASES entry '{case}'");
                    return None;
                }
                Some(WriteCase {
                    name: parts[0].to_string(),
                    partition_count,
                    rows,
                    batch_size,
                })
            })
            .collect::<Vec<_>>();
        if !parsed.is_empty() {
            return parsed;
        }
    }

    DEFAULT_CASES
        .iter()
        .map(|(name, partition_count, rows, batch_size)| WriteCase {
            name: (*name).to_string(),
            partition_count: *partition_count,
            rows: *rows,
            batch_size: *batch_size,
        })
        .collect()
}

fn queue_uri(prefix: &str, case_name: &str, repeat: usize) -> String {
    let safe_case_name = case_name.replace(|ch: char| !ch.is_ascii_alphanumeric(), "_");
    format!(
        "{}/queue_write_{}_r{}_{}",
        prefix.trim_end_matches('/'),
        safe_case_name,
        repeat,
        Uuid::new_v4()
    )
}

fn make_input_batches(rows: usize, batch_size: usize, payload_bytes: usize) -> Vec<InputBatch> {
    let body = "x".repeat(payload_bytes);
    let mut batches = Vec::with_capacity(rows.div_ceil(batch_size));
    let mut next_row = 0usize;
    while next_row < rows {
        let rows_in_batch = batch_size.min(rows - next_row);
        let ids = (0..rows_in_batch)
            .map(|idx| format!("message-{}", next_row + idx))
            .collect::<Vec<_>>();
        let payloads = (0..rows_in_batch)
            .map(|idx| {
                json!({
                    "row": next_row + idx,
                    "body": body,
                })
            })
            .collect::<Vec<_>>();
        batches.push(InputBatch { ids, payloads });
        next_row += rows_in_batch;
    }
    batches
}

fn input_bytes(batches: &[InputBatch]) -> u64 {
    batches
        .iter()
        .map(|batch| {
            batch.ids.iter().map(|id| id.len() as u64).sum::<u64>()
                + batch
                    .payloads
                    .iter()
                    .map(|payload| payload.to_string().len() as u64)
                    .sum::<u64>()
        })
        .sum()
}

fn result_writer() -> Result<Option<std::fs::File>> {
    let Some(path) = std::env::var("RESULT_CSV").ok() else {
        return Ok(None);
    };
    let exists = std::path::Path::new(&path).exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if !exists {
        writeln!(file, "{}", WriteMeasurement::csv_header())?;
    }
    Ok(Some(file))
}

fn write_measurement(
    writer: &mut Option<std::fs::File>,
    measurement: &WriteMeasurement,
) -> Result<()> {
    let row = measurement.csv_row();
    println!("{row}");
    if let Some(file) = writer {
        writeln!(file, "{row}")?;
        file.flush()?;
    }
    Ok(())
}

async fn run_case(
    dataset_prefix: &str,
    payload_bytes: usize,
    repeat: usize,
    case: &WriteCase,
) -> Result<WriteMeasurement> {
    let input_batches = make_input_batches(case.rows, case.batch_size, payload_bytes);
    let input_bytes = input_bytes(&input_batches);
    let queue = Queue::builder()
        .uri(queue_uri(dataset_prefix, &case.name, repeat))
        .partition_count(case.partition_count)
        .create()
        .await?;
    let producer = queue.producer();

    let warmup = make_input_batches(case.batch_size.min(1_000), case.batch_size, payload_bytes);
    for batch in warmup {
        producer.send_batch(batch.ids, batch.payloads).await?;
    }

    let start = Instant::now();
    let mut wal_bytes = 0u64;
    let mut wal_entries = 0usize;
    for batch in input_batches {
        let result = producer.send_batch(batch.ids, batch.payloads).await?;
        wal_bytes += result
            .entries
            .iter()
            .map(|entry| entry.wal_bytes as u64)
            .sum::<u64>();
        wal_entries += result.entries.len();
    }
    let elapsed = start.elapsed();

    Ok(WriteMeasurement {
        case_name: case.name.clone(),
        partition_count: case.partition_count,
        rows: case.rows,
        batch_size: case.batch_size,
        payload_bytes,
        repeat,
        elapsed,
        input_bytes,
        wal_bytes,
        wal_entries,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let dataset_prefix = get_dataset_prefix();
    let cases = parse_cases();
    let payload_bytes = env_usize("PAYLOAD_BYTES", DEFAULT_PAYLOAD_BYTES);
    let repeats = env_usize("REPEATS", DEFAULT_REPEATS).max(1);
    let mut writer = result_writer()?;

    println!("=== Lance Queue Write Benchmark ===");
    println!("dataset_prefix={dataset_prefix}");
    println!("payload_bytes={payload_bytes}");
    println!("repeats={repeats}");
    println!("cases={}", cases.len());
    println!("{}", WriteMeasurement::csv_header());

    for case in &cases {
        for repeat in 0..repeats {
            let measurement = run_case(&dataset_prefix, payload_bytes, repeat, case).await?;
            write_measurement(&mut writer, &measurement)?;
        }
    }

    Ok(())
}
