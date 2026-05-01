// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Matrix benchmark for Lance queue consumer read throughput.
//!
//! The benchmark writes CSV rows to stdout and, when `RESULT_CSV` is set, to a
//! file. Each case creates a single-logical-partition queue, seeds it before
//! timing, and then measures raw consumer poll throughput across all producer
//! shards for that logical partition.
//!
//! ## Running against S3
//!
//! ```bash
//! export AWS_DEFAULT_REGION=us-east-1
//! export DATASET_PREFIX=s3://your-bucket/bench/lance_queue
//! export RESULT_CSV=/tmp/lance_queue_read.csv
//! cargo bench -p lance-queue --bench queue_read
//! ```
//!
//! ## Configuration
//!
//! - `DATASET_PREFIX`: Base URI for queue tables. If not set, uses a temporary local directory.
//! - `PAYLOAD_BYTES`: Approximate bytes in each JSON payload body string (default: `256`).
//! - `REPEATS`: Repeated measurements per scenario (default: `3`).
//! - `READ_CASES`: Optional explicit cases as `name:producers:rows:write_batch_size:poll_entries:decode_messages` separated by `;`.
//! - `RESULT_CSV`: Optional output CSV file path.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Duration, Instant};

use futures::future::try_join_all;
use lance_core::{Error, Result};
use lance_queue::{ConsumerConfig, PollOptions, Producer, Queue, StartPosition};
use serde_json::{Value, json};
use uuid::Uuid;

const DEFAULT_PAYLOAD_BYTES: usize = 256;
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_CASES: &[(&str, u32, usize, usize, usize, bool)] = &[
    ("read_prod1_50k_poll32", 1, 50_000, 5_000, 32, false),
    ("read_prod1_200k_poll1", 1, 200_000, 5_000, 1, false),
    ("read_prod1_200k_poll8", 1, 200_000, 5_000, 8, false),
    ("read_prod1_200k_poll32", 1, 200_000, 5_000, 32, false),
    ("read_prod4_200k_poll32", 4, 200_000, 5_000, 32, false),
    ("read_prod8_500k_poll32", 8, 500_000, 5_000, 32, false),
    ("read_decode_prod1_200k_poll32", 1, 200_000, 5_000, 32, true),
];

#[derive(Debug, Clone)]
struct ReadCase {
    name: String,
    producer_count: u32,
    rows: usize,
    write_batch_size: usize,
    poll_entries: usize,
    decode_messages: bool,
}

#[derive(Debug, Clone)]
struct InputBatch {
    ids: Vec<String>,
    payloads: Vec<Value>,
}

#[derive(Debug, Clone)]
struct ReadMeasurement {
    case_name: String,
    producer_count: u32,
    rows: usize,
    write_batch_size: usize,
    poll_entries: usize,
    payload_bytes: usize,
    decode_messages: bool,
    repeat: usize,
    elapsed: Duration,
    input_bytes: u64,
    wal_entries_read: usize,
    arrow_batches_read: usize,
    polls: usize,
}

impl ReadMeasurement {
    fn rows_per_second(&self) -> f64 {
        self.rows as f64 / self.elapsed.as_secs_f64()
    }

    fn input_mib_per_second(&self) -> f64 {
        self.input_bytes as f64 / self.elapsed.as_secs_f64() / (1024.0 * 1024.0)
    }

    fn csv_header() -> &'static str {
        "benchmark,case,producer_count,rows,write_batch_size,poll_entries,payload_bytes,decode_messages,repeat,elapsed_seconds,rows_per_second,input_mib_per_second,wal_entries_read,arrow_batches_read,polls"
    }

    fn csv_row(&self) -> String {
        format!(
            "read,{},{},{},{},{},{},{},{},{:.6},{:.3},{:.3},{},{},{}",
            self.case_name,
            self.producer_count,
            self.rows,
            self.write_batch_size,
            self.poll_entries,
            self.payload_bytes,
            self.decode_messages,
            self.repeat,
            self.elapsed.as_secs_f64(),
            self.rows_per_second(),
            self.input_mib_per_second(),
            self.wal_entries_read,
            self.arrow_batches_read,
            self.polls
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
        let temp_dir =
            std::env::temp_dir().join(format!("lance_queue_read_bench_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).expect("failed to create benchmark temp directory");
        temp_dir.to_string_lossy().to_string()
    })
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => Some(true),
        "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

fn parse_cases() -> Vec<ReadCase> {
    if let Ok(raw) = std::env::var("READ_CASES") {
        let parsed = raw
            .split(';')
            .filter_map(|case| {
                let parts = case.split(':').collect::<Vec<_>>();
                if parts.len() != 6 {
                    eprintln!("Ignoring invalid READ_CASES entry '{case}'");
                    return None;
                }
                let producer_count = parts[1].parse::<u32>().ok()?;
                let rows = parts[2].parse::<usize>().ok()?;
                let write_batch_size = parts[3].parse::<usize>().ok()?;
                let poll_entries = parts[4].parse::<usize>().ok()?;
                let decode_messages = parse_bool(parts[5])?;
                if producer_count == 0 || rows == 0 || write_batch_size == 0 || poll_entries == 0 {
                    eprintln!("Ignoring non-positive READ_CASES entry '{case}'");
                    return None;
                }
                Some(ReadCase {
                    name: parts[0].to_string(),
                    producer_count,
                    rows,
                    write_batch_size,
                    poll_entries,
                    decode_messages,
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
            |(name, producer_count, rows, write_batch_size, poll_entries, decode_messages)| {
                ReadCase {
                    name: (*name).to_string(),
                    producer_count: *producer_count,
                    rows: *rows,
                    write_batch_size: *write_batch_size,
                    poll_entries: *poll_entries,
                    decode_messages: *decode_messages,
                }
            },
        )
        .collect()
}

fn queue_uri(prefix: &str, case_name: &str, repeat: usize) -> String {
    let safe_case_name = case_name.replace(|ch: char| !ch.is_ascii_alphanumeric(), "_");
    format!(
        "{}/queue_read_{}_r{}_{}",
        prefix.trim_end_matches('/'),
        safe_case_name,
        repeat,
        Uuid::new_v4()
    )
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
) -> Vec<InputBatch> {
    let body = "x".repeat(payload_bytes);
    let mut batches = Vec::with_capacity(rows.div_ceil(batch_size));
    let mut next_row = 0usize;
    while next_row < rows {
        let rows_in_batch = batch_size.min(rows - next_row);
        let ids = (0..rows_in_batch)
            .map(|idx| format!("producer-{producer_id}-message-{}", next_row + idx))
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
) -> Vec<Vec<InputBatch>> {
    (0..producer_count)
        .map(|producer_id| {
            make_input_batches(
                producer_id,
                rows_for_producer(rows, producer_count, producer_id),
                batch_size,
                payload_bytes,
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

async fn seed_queue(
    producers: &[Producer],
    batches_by_producer: Vec<Vec<InputBatch>>,
) -> Result<()> {
    let mut iterators = batches_by_producer
        .into_iter()
        .map(Vec::into_iter)
        .collect::<Vec<_>>();

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
        try_join_all(produce_futures).await?;
    }

    Ok(())
}

fn map_io_error(context: &str, error: std::io::Error) -> Error {
    Error::io(format!("{context}: {error}"))
}

fn result_writer() -> Result<Option<std::fs::File>> {
    let Some(path) = std::env::var("RESULT_CSV").ok() else {
        return Ok(None);
    };
    let exists = std::path::Path::new(&path).exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| map_io_error(&format!("failed to open RESULT_CSV {path}"), e))?;
    if !exists {
        writeln!(file, "{}", ReadMeasurement::csv_header())
            .map_err(|e| map_io_error("failed to write RESULT_CSV header", e))?;
    }
    Ok(Some(file))
}

fn write_measurement(
    writer: &mut Option<std::fs::File>,
    measurement: &ReadMeasurement,
) -> Result<()> {
    let row = measurement.csv_row();
    println!("{row}");
    if let Some(file) = writer {
        writeln!(file, "{row}").map_err(|e| map_io_error("failed to write RESULT_CSV row", e))?;
        file.flush()
            .map_err(|e| map_io_error("failed to flush RESULT_CSV", e))?;
    }
    Ok(())
}

async fn run_case(
    dataset_prefix: &str,
    payload_bytes: usize,
    repeat: usize,
    case: &ReadCase,
) -> Result<ReadMeasurement> {
    let input_batches = make_batches_by_producer(
        case.producer_count,
        case.rows,
        case.write_batch_size,
        payload_bytes,
    );
    let input_bytes = input_bytes(&input_batches);
    let queue = Queue::builder()
        .uri(queue_uri(dataset_prefix, &case.name, repeat))
        .partition_count(1)
        .producer_count(case.producer_count)
        .create()
        .await?;
    let producers = (0..case.producer_count)
        .map(|producer_id| queue.producer(producer_id))
        .collect::<Result<Vec<_>>>()?;
    seed_queue(&producers, input_batches).await?;

    let mut consumer = queue
        .consumer(
            ConsumerConfig::new(format!("read-bench-{repeat}"))
                .with_partitions([0])
                .with_start_position(StartPosition::Earliest),
        )
        .await?;
    let options = PollOptions {
        max_entries_per_partition: case.poll_entries,
    };

    let start = Instant::now();
    let mut rows_read = 0usize;
    let mut wal_entries_read = 0usize;
    let mut arrow_batches_read = 0usize;
    let mut polls = 0usize;
    while rows_read < case.rows {
        let batches = consumer.poll_with_options(options.clone()).await?;
        polls += 1;
        if batches.is_empty() {
            return Err(Error::io(format!(
                "read benchmark case '{}' reached end of WAL after {} rows, expected {}",
                case.name, rows_read, case.rows
            )));
        }

        for batch in batches {
            if case.decode_messages {
                let messages = batch.messages()?;
                rows_read += messages.len();
            } else {
                rows_read += batch.num_rows();
            }
            wal_entries_read += 1;
            arrow_batches_read += batch.batches.len();
        }
    }
    let elapsed = start.elapsed();

    if rows_read != case.rows {
        return Err(Error::io(format!(
            "read benchmark case '{}' read {} rows, expected {}",
            case.name, rows_read, case.rows
        )));
    }

    Ok(ReadMeasurement {
        case_name: case.name.clone(),
        producer_count: case.producer_count,
        rows: case.rows,
        write_batch_size: case.write_batch_size,
        poll_entries: case.poll_entries,
        payload_bytes,
        decode_messages: case.decode_messages,
        repeat,
        elapsed,
        input_bytes,
        wal_entries_read,
        arrow_batches_read,
        polls,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let dataset_prefix = get_dataset_prefix();
    let cases = parse_cases();
    let payload_bytes = env_usize("PAYLOAD_BYTES", DEFAULT_PAYLOAD_BYTES);
    let repeats = env_usize("REPEATS", DEFAULT_REPEATS).max(1);
    let mut writer = result_writer()?;

    println!("=== Lance Queue Read Benchmark ===");
    println!("dataset_prefix={dataset_prefix}");
    println!("payload_bytes={payload_bytes}");
    println!("repeats={repeats}");
    println!("cases={}", cases.len());
    println!("{}", ReadMeasurement::csv_header());

    for case in &cases {
        for repeat in 0..repeats {
            let measurement = run_case(&dataset_prefix, payload_bytes, repeat, case).await?;
            write_measurement(&mut writer, &measurement)?;
        }
    }

    Ok(())
}
