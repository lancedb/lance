// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reproducible end-to-end FTS benchmark for an existing Lance dataset.
//!
//! The executable emits JSON Lines so separate processes can be used for
//! first-touch measurements and the same binary can also run warm or
//! concurrent query panels.

#![allow(clippy::print_stdout)]

use std::error::Error;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{Float32Type, UInt64Type};
use clap::Parser;
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::index::DatasetIndexExt;
use lance_index::{FtsPrewarmOptions, PrewarmOptions, scalar::FullTextSearchQuery};
use serde_json::{Value, json};

type AnyError = Box<dyn Error + Send + Sync>;
type AnyResult<T> = Result<T, AnyError>;

const ROW_ID_COLUMN: &str = "_rowid";
const SCORE_COLUMN: &str = "_score";

#[derive(Debug, Parser)]
#[command(about = "Benchmark FTS queries against an existing Lance dataset")]
struct Args {
    /// Lance dataset URI, including s3:// URIs.
    #[arg(long)]
    uri: String,

    /// Indexed text column.
    #[arg(long)]
    column: String,

    /// Query to run. May be repeated and is appended after query-file entries.
    #[arg(long = "query")]
    queries: Vec<String>,

    /// UTF-8 query file, one query per non-empty, non-comment line.
    #[arg(long)]
    query_file: Option<PathBuf>,

    /// Number of results per query.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Sequential panel rounds discarded before measurement.
    #[arg(long, default_value_t = 0)]
    warmup_rounds: usize,

    /// Measured rounds over the complete query panel.
    #[arg(long, default_value_t = 1)]
    measured_rounds: usize,

    /// Maximum number of measured queries in flight.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Label copied to every output record (for example baseline or refactor).
    #[arg(long, default_value = "run")]
    label: String,

    /// Optional exact row-count assertion.
    #[arg(long)]
    expected_rows: Option<usize>,

    /// Optional index name assertion. At least one segment must have this name.
    #[arg(long)]
    expected_index: Option<String>,

    /// Index name passed to Dataset::prewarm_index before any measured query.
    #[arg(long)]
    prewarm_index: Option<String>,

    /// Also prewarm FTS positions for phrase queries.
    #[arg(long, default_value_t = false, requires = "prewarm_index")]
    prewarm_positions: bool,

    /// Dataset index-cache capacity in GiB.
    #[arg(long)]
    index_cache_size_gib: Option<usize>,

    /// Permit queries with no results. By default they fail the run so an
    /// accidentally irrelevant panel cannot produce plausible timings.
    #[arg(long, default_value_t = false)]
    allow_empty: bool,
}

#[derive(Debug)]
struct QueryMeasurement {
    round: usize,
    query_index: usize,
    query: String,
    latency_ms: f64,
    row_ids: Vec<u64>,
    score_bits: Vec<u32>,
    result_hash: String,
    peak_rss_kib: Option<u64>,
}

fn emit(value: Value) {
    println!("{value}");
}

fn load_queries(args: &Args) -> AnyResult<Vec<String>> {
    let mut queries = Vec::new();
    if let Some(path) = &args.query_file {
        let contents = fs::read_to_string(path)?;
        queries.extend(
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(str::to_owned),
        );
    }
    queries.extend(
        args.queries
            .iter()
            .map(|query| query.trim())
            .filter(|query| !query.is_empty())
            .map(str::to_owned),
    );
    if queries.is_empty() {
        return Err("provide at least one --query or --query-file entry".into());
    }
    Ok(queries)
}

fn result_hash(row_ids: &[u64], score_bits: &[u32]) -> String {
    // `DefaultHasher::new` has fixed keys. Hashing score bits, rather than
    // formatted floats, makes exact baseline/candidate parity machine-readable.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    row_ids.hash(&mut hasher);
    score_bits.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn linux_memory_kib(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix(field)?.trim();
        value.strip_suffix(" kB")?.trim().parse().ok()
    })
}

async fn run_query(
    dataset: &Dataset,
    column: &str,
    query: &str,
    k: usize,
    allow_empty: bool,
) -> AnyResult<(f64, Vec<u64>, Vec<u32>)> {
    let started = Instant::now();
    let mut scanner = dataset.scan();
    let query = FullTextSearchQuery::new(query.to_owned())
        .with_column(column.to_owned())?
        .limit(Some(i64::try_from(k)?));
    scanner.full_text_search(query)?.project(&[ROW_ID_COLUMN])?;
    let batches = scanner
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let result_count = batches.iter().map(|batch| batch.num_rows()).sum();
    let mut row_ids = Vec::with_capacity(result_count);
    let mut score_bits = Vec::with_capacity(result_count);
    for batch in batches {
        let rows = batch
            .column_by_name(ROW_ID_COLUMN)
            .ok_or_else(|| format!("FTS result is missing {ROW_ID_COLUMN}"))?
            .as_primitive::<UInt64Type>();
        let scores = batch
            .column_by_name(SCORE_COLUMN)
            .ok_or_else(|| format!("FTS result is missing {SCORE_COLUMN}"))?
            .as_primitive::<Float32Type>();
        if rows.len() != scores.len() {
            return Err(format!(
                "FTS result has {} row IDs but {} scores",
                rows.len(),
                scores.len()
            )
            .into());
        }
        for row in 0..rows.len() {
            if rows.is_null(row) || scores.is_null(row) {
                return Err(format!("FTS result row {row} contains a null row ID or score").into());
            }
            row_ids.push(rows.value(row));
            score_bits.push(scores.value(row).to_bits());
        }
    }
    if row_ids.len() > k {
        return Err(format!("FTS returned {} rows for k={k}", row_ids.len()).into());
    }
    if row_ids.is_empty() && !allow_empty {
        return Err("FTS query returned no rows; use --allow-empty only when intentional".into());
    }
    Ok((latency_ms, row_ids, score_bits))
}

fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * percentile / 100.0).round() as usize;
    sorted.get(index).copied()
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    let args = Args::parse();
    if args.k == 0 {
        return Err("--k must be greater than zero".into());
    }
    if args.measured_rounds == 0 {
        return Err("--measured-rounds must be greater than zero".into());
    }
    if args.concurrency == 0 {
        return Err("--concurrency must be greater than zero".into());
    }
    let queries = load_queries(&args)?;

    emit(json!({
        "event": "run_start",
        "label": args.label,
        "uri": args.uri,
        "column": args.column,
        "k": args.k,
        "query_count": queries.len(),
        "warmup_rounds": args.warmup_rounds,
        "measured_rounds": args.measured_rounds,
        "concurrency": args.concurrency,
        "prewarm_index": args.prewarm_index,
        "prewarm_positions": args.prewarm_positions,
        "index_cache_size_gib": args.index_cache_size_gib,
        "hostname": std::env::var("HOSTNAME").ok(),
        "available_parallelism": std::thread::available_parallelism().map(usize::from).ok(),
        "package_version": env!("CARGO_PKG_VERSION"),
    }));

    let open_started = Instant::now();
    let mut dataset_builder = DatasetBuilder::from_uri(&args.uri);
    if let Some(cache_size_gib) = args.index_cache_size_gib {
        let cache_size_bytes = cache_size_gib
            .checked_mul(1024 * 1024 * 1024)
            .ok_or("--index-cache-size-gib overflows usize")?;
        dataset_builder = dataset_builder.with_index_cache_size_bytes(cache_size_bytes);
    }
    let dataset = dataset_builder.load().await?;
    let open_ms = open_started.elapsed().as_secs_f64() * 1_000.0;
    let row_count = dataset.count_rows(None).await?;
    if let Some(expected) = args.expected_rows
        && row_count != expected
    {
        return Err(format!("dataset has {row_count} rows, expected {expected}").into());
    }
    let indices = dataset.load_indices().await?;
    if let Some(expected) = &args.expected_index
        && !indices.iter().any(|index| index.name == *expected)
    {
        return Err(format!("dataset does not contain index {expected:?}").into());
    }
    let index_summary = indices
        .iter()
        .map(|index| {
            json!({
                "name": index.name,
                "uuid": index.uuid.to_string(),
                "fields": index.fields,
                "dataset_version": index.dataset_version,
                "index_version": index.index_version,
                "fragment_count": index.fragment_bitmap.as_ref().map(|fragments| fragments.len()),
                "type_url": index.index_details.as_ref().map(|details| details.type_url.as_str()),
            })
        })
        .collect::<Vec<_>>();
    emit(json!({
        "event": "dataset_open",
        "label": args.label,
        "open_ms": open_ms,
        "dataset_version": dataset.version_id(),
        "row_count": row_count,
        "indices": index_summary,
        "rss_kib": linux_memory_kib("VmRSS:"),
        "peak_rss_kib": linux_memory_kib("VmHWM:"),
    }));

    if let Some(index_name) = &args.prewarm_index {
        let cache_before = dataset.session().index_cache_stats().await;
        let prewarm_started = Instant::now();
        if args.prewarm_positions {
            dataset
                .prewarm_index_with_options(
                    index_name,
                    &PrewarmOptions::Fts(FtsPrewarmOptions::new().with_position(true)),
                )
                .await?;
        } else {
            dataset.prewarm_index(index_name).await?;
        }
        let prewarm_ms = prewarm_started.elapsed().as_secs_f64() * 1_000.0;
        let cache_after = dataset.session().index_cache_stats().await;
        emit(json!({
            "event": "prewarm_complete",
            "label": args.label,
            "index": index_name,
            "with_positions": args.prewarm_positions,
            "prewarm_ms": prewarm_ms,
            "index_cache_entries_before": cache_before.num_entries,
            "index_cache_entries_after": cache_after.num_entries,
            "index_cache_size_bytes_before": cache_before.size_bytes,
            "index_cache_size_bytes_after": cache_after.size_bytes,
            "rss_kib": linux_memory_kib("VmRSS:"),
            "peak_rss_kib": linux_memory_kib("VmHWM:"),
        }));
    }

    for _ in 0..args.warmup_rounds {
        for query in &queries {
            let _ = run_query(&dataset, &args.column, query, args.k, args.allow_empty).await?;
        }
    }

    let dataset = Arc::new(dataset);
    let jobs = (0..args.measured_rounds)
        .flat_map(|round| {
            queries
                .iter()
                .cloned()
                .enumerate()
                .map(move |(query_index, query)| (round, query_index, query))
        })
        .collect::<Vec<_>>();
    let measured_started = Instant::now();
    let mut measurements = futures::stream::iter(jobs)
        .map(|(round, query_index, query)| {
            let dataset = dataset.clone();
            let column = args.column.clone();
            async move {
                let (latency_ms, row_ids, score_bits) =
                    run_query(&dataset, &column, &query, args.k, args.allow_empty).await?;
                Ok::<_, AnyError>(QueryMeasurement {
                    round,
                    query_index,
                    query,
                    latency_ms,
                    result_hash: result_hash(&row_ids, &score_bits),
                    row_ids,
                    score_bits,
                    peak_rss_kib: linux_memory_kib("VmHWM:"),
                })
            }
        })
        .buffer_unordered(args.concurrency)
        .try_collect::<Vec<_>>()
        .await?;
    let measured_seconds = measured_started.elapsed().as_secs_f64();
    measurements.sort_unstable_by_key(|measurement| (measurement.round, measurement.query_index));

    for measurement in &measurements {
        emit(json!({
            "event": "query",
            "label": args.label,
            "round": measurement.round,
            "query_index": measurement.query_index,
            "query": measurement.query,
            "latency_ms": measurement.latency_ms,
            "num_results": measurement.row_ids.len(),
            "result_hash": measurement.result_hash,
            "row_ids": measurement.row_ids,
            "score_bits": measurement.score_bits,
            "peak_rss_kib": measurement.peak_rss_kib,
        }));
    }

    let latencies = measurements
        .iter()
        .map(|measurement| measurement.latency_ms)
        .collect::<Vec<_>>();
    emit(json!({
        "event": "summary",
        "label": args.label,
        "query_executions": measurements.len(),
        "measured_seconds": measured_seconds,
        "throughput_qps": measurements.len() as f64 / measured_seconds,
        "latency_mean_ms": latencies.iter().sum::<f64>() / latencies.len() as f64,
        "latency_p50_ms": percentile(&latencies, 50.0),
        "latency_p95_ms": percentile(&latencies, 95.0),
        "latency_p99_ms": percentile(&latencies, 99.0),
        "rss_kib": linux_memory_kib("VmRSS:"),
        "peak_rss_kib": linux_memory_kib("VmHWM:"),
    }));
    Ok(())
}
