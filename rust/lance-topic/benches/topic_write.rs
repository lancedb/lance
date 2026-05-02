// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Matrix benchmark for Lance topic producer write throughput.
//!
//! The benchmark writes CSV rows to stdout and, when `RESULT_CSV` is set, to a
//! file. It is intended for longer S3 runs where we want trends across logical
//! partition counts, producer shard counts, row counts, and producer call batch
//! sizes.
//!
//! ## Running against S3
//!
//! ```bash
//! export AWS_DEFAULT_REGION=us-east-1
//! export DATASET_PREFIX=s3://your-bucket/bench/lance_topic
//! export RESULT_CSV=/tmp/lance_topic_write.csv
//! cargo bench -p lance-topic --bench topic_write
//! ```
//!
//! ## Configuration
//!
//! - `DATASET_PREFIX`: Directory namespace root URI for topic tables. If not set, uses a temporary local directory.
//! - `PAYLOAD_BYTES`: Approximate bytes in each JSON payload body string (default: `256`).
//! - `REPEATS`: Repeated measurements per scenario (default: `3`).
//! - `WRITE_CASES`: Optional explicit cases as `name:partitions:producers:rows:batch_size` separated by `;`.
//!   `batch_size` is the number of messages in each producer call; `1` is the
//!   unbatched `send` shape and commits one message at a time.
//! - `RESULT_CSV`: Optional output CSV file path.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Duration, Instant};

use futures::future::try_join_all;
use lance_core::Result;
use lance_topic::{Producer, Topic};
use serde_json::{Value, json};
use uuid::Uuid;

const DEFAULT_PAYLOAD_BYTES: usize = 256;
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_CASES: &[(&str, u32, u32, usize, usize)] = &[
    ("horizontal_p1_prod1", 1, 1, 500_000, 5_000),
    ("horizontal_p2_prod2", 2, 2, 500_000, 5_000),
    ("horizontal_p4_prod4", 4, 4, 500_000, 5_000),
    ("horizontal_p4_prod10", 4, 10, 500_000, 5_000),
    ("horizontal_p8_prod16", 8, 16, 500_000, 5_000),
    ("trend_50k_p4_prod10", 4, 10, 50_000, 5_000),
    ("trend_200k_p4_prod10", 4, 10, 200_000, 5_000),
    ("trend_500k_p4_prod10", 4, 10, 500_000, 5_000),
    ("batch_1_p1_prod1", 1, 1, 2_000, 1),
    ("batch_10_p1_prod1", 1, 1, 20_000, 10),
    ("batch_100_p1_prod1", 1, 1, 100_000, 100),
    ("batch_1000_p1_prod1", 1, 1, 200_000, 1_000),
    ("batch_5000_p1_prod1", 1, 1, 500_000, 5_000),
    ("batch_10000_p1_prod1", 1, 1, 500_000, 10_000),
];

#[derive(Debug, Clone)]
struct WriteCase {
    name: String,
    partition_count: u32,
    producer_count: u32,
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
    producer_count: u32,
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

    fn physical_shard_count(&self) -> u32 {
        self.partition_count * self.producer_count
    }

    fn csv_header() -> &'static str {
        "benchmark,case,partition_count,producer_count,physical_shard_count,rows,batch_size,payload_bytes,repeat,elapsed_seconds,rows_per_second,input_mib_per_second,wal_mib_per_second,wal_bytes,wal_entries"
    }

    fn csv_row(&self) -> String {
        format!(
            "write,{},{},{},{},{},{},{},{},{:.6},{:.3},{:.3},{:.3},{},{}",
            self.case_name,
            self.partition_count,
            self.producer_count,
            self.physical_shard_count(),
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
        let temp_dir = std::env::temp_dir().join(format!("lance_topic_bench_{}", Uuid::new_v4()));
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
                if parts.len() != 5 {
                    eprintln!("Ignoring invalid WRITE_CASES entry '{case}'");
                    return None;
                }
                let partition_count = parts[1].parse::<u32>().ok()?;
                let producer_count = parts[2].parse::<u32>().ok()?;
                let rows = parts[3].parse::<usize>().ok()?;
                let batch_size = parts[4].parse::<usize>().ok()?;
                if partition_count == 0 || producer_count == 0 || rows == 0 || batch_size == 0 {
                    eprintln!("Ignoring non-positive WRITE_CASES entry '{case}'");
                    return None;
                }
                Some(WriteCase {
                    name: parts[0].to_string(),
                    partition_count,
                    producer_count,
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
        .map(
            |(name, partition_count, producer_count, rows, batch_size)| WriteCase {
                name: (*name).to_string(),
                partition_count: *partition_count,
                producer_count: *producer_count,
                rows: *rows,
                batch_size: *batch_size,
            },
        )
        .collect()
}

fn topic_table_id(case_name: &str, repeat: usize) -> Vec<String> {
    let safe_case_name = case_name.replace(|ch: char| !ch.is_ascii_alphanumeric(), "_");
    vec![format!(
        "topic_write_{}_r{}_{}",
        safe_case_name,
        repeat,
        Uuid::new_v4()
    )]
}

fn rows_for_producer(total_rows: usize, producer_count: u32, producer_id: u32) -> usize {
    let producer_count = producer_count as usize;
    let producer_id = producer_id as usize;
    let base = total_rows / producer_count;
    let remainder = total_rows % producer_count;
    base + usize::from(producer_id < remainder)
}

fn make_input_batches(
    producer_id: u32,
    rows: usize,
    batch_size: usize,
    payload_bytes: usize,
    id_prefix: &str,
) -> Vec<InputBatch> {
    let body = "x".repeat(payload_bytes);
    let mut batches = Vec::with_capacity(rows.div_ceil(batch_size));
    let mut next_row = 0usize;
    while next_row < rows {
        let rows_in_batch = batch_size.min(rows - next_row);
        let ids = (0..rows_in_batch)
            .map(|idx| {
                format!(
                    "{id_prefix}-producer-{producer_id}-message-{}",
                    next_row + idx
                )
            })
            .collect::<Vec<_>>();
        let payloads = (0..rows_in_batch)
            .map(|idx| {
                json!({
                    "producer_id": producer_id,
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

fn make_batches_by_producer(
    producer_count: u32,
    rows: usize,
    batch_size: usize,
    payload_bytes: usize,
    id_prefix: &str,
) -> Vec<Vec<InputBatch>> {
    (0..producer_count)
        .map(|producer_id| {
            make_input_batches(
                producer_id,
                rows_for_producer(rows, producer_count, producer_id),
                batch_size,
                payload_bytes,
                id_prefix,
            )
        })
        .collect()
}

fn input_bytes(batches_by_producer: &[Vec<InputBatch>]) -> u64 {
    batches_by_producer
        .iter()
        .flatten()
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

async fn produce_batches(
    producers: &[Producer],
    batches_by_producer: Vec<Vec<InputBatch>>,
) -> Result<(u64, usize)> {
    let mut iterators = batches_by_producer
        .into_iter()
        .map(Vec::into_iter)
        .collect::<Vec<_>>();
    let mut wal_bytes = 0u64;
    let mut wal_entries = 0usize;

    loop {
        let mut produce_futures = Vec::with_capacity(producers.len());
        for (producer, batches) in producers.iter().zip(iterators.iter_mut()) {
            if let Some(batch) = batches.next() {
                produce_futures.push(producer.send_batch(batch.ids, batch.payloads));
            }
        }
        if produce_futures.is_empty() {
            break;
        }

        let results = try_join_all(produce_futures).await?;
        for result in results {
            wal_bytes += result
                .entries
                .iter()
                .map(|entry| entry.wal_bytes as u64)
                .sum::<u64>();
            wal_entries += result.entries.len();
        }
    }

    Ok((wal_bytes, wal_entries))
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
    let input_batches = make_batches_by_producer(
        case.producer_count,
        case.rows,
        case.batch_size,
        payload_bytes,
        "message",
    );
    let input_bytes = input_bytes(&input_batches);
    let topic = Topic::builder()
        .directory(dataset_prefix, topic_table_id(&case.name, repeat))
        .partition_count(case.partition_count)
        .create()
        .await?;
    let producers = try_join_all((0..case.producer_count).map(|producer_id| {
        let topic = topic.clone();
        async move { topic.producer(format!("producer-{}", producer_id)).await }
    }))
    .await?;

    let warmup = make_batches_by_producer(
        case.producer_count,
        (case.batch_size * case.producer_count as usize).min(1_000),
        case.batch_size,
        payload_bytes,
        "warmup",
    );
    produce_batches(&producers, warmup).await?;

    let start = Instant::now();
    let (wal_bytes, wal_entries) = produce_batches(&producers, input_batches).await?;
    let elapsed = start.elapsed();

    Ok(WriteMeasurement {
        case_name: case.name.clone(),
        partition_count: case.partition_count,
        producer_count: case.producer_count,
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

    println!("=== Lance Topic Write Benchmark ===");
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
