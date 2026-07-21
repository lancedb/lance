// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reproducible end-to-end FTS benchmark for an existing Lance dataset.
//!
//! The executable emits JSON Lines so separate processes can be used for
//! first-touch measurements and the same binary can also run warm or
//! concurrent query panels.

#![allow(clippy::print_stdout)]

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{Float32Type, UInt64Type};
use async_trait::async_trait;
use clap::Parser;
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_core::utils::address::RowAddress;
use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::inverted::query::{FtsSearchParams, Operator, collect_query_tokens};
use lance_index::scalar::inverted::{InvertedIndex, MemBM25Scorer};
use lance_index::scalar::{FullTextSearchQuery, ScalarIndex};
use lance_index::{FtsPrewarmOptions, PrewarmOptions};
use lance_select::{RowAddrMask, RowAddrTreeMap};
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

    /// Select every Nth physical row through a synthetic FTS prefilter. This
    /// bypasses scalar-filter construction so the benchmark isolates indexed
    /// FTS visibility and address projection.
    #[arg(long)]
    filter_stride: Option<u32>,

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

#[derive(Debug)]
struct StaticPreFilter {
    mask: Arc<RowAddrMask>,
}

#[async_trait]
impl PreFilter for StaticPreFilter {
    async fn wait_for_ready(&self) -> lance_core::Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.mask.clone()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        row_ids
            .enumerate()
            .filter_map(|(index, row_id)| self.mask.selected(*row_id).then_some(index as u64))
            .collect()
    }
}

#[derive(Debug)]
struct FilteredSearchContext {
    indices: Vec<Arc<dyn ScalarIndex>>,
    prefilter: Arc<StaticPreFilter>,
    selected_rows: u64,
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

async fn run_scanner_query(
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

async fn build_filter_mask(dataset: &Dataset, stride: u32) -> AnyResult<(Arc<RowAddrMask>, u64)> {
    if stride == 0 {
        return Err("--filter-stride must be greater than zero".into());
    }
    let fragments = futures::stream::iter(dataset.get_fragments())
        .map(|fragment| async move {
            let fragment_id = u32::try_from(fragment.id())?;
            let physical_rows = u32::try_from(fragment.physical_rows().await?)?;
            Ok::<_, AnyError>((fragment_id, physical_rows))
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;

    let mut addresses = RowAddrTreeMap::new();
    let mut selected_rows = 0_u64;
    for (fragment_id, physical_rows) in fragments {
        for row_offset in (0..physical_rows).step_by(stride as usize) {
            addresses.insert(u64::from(RowAddress::new_from_parts(
                fragment_id,
                row_offset,
            )));
            selected_rows += 1;
        }
    }
    Ok((
        Arc::new(RowAddrMask::from_allowed(addresses)),
        selected_rows,
    ))
}

async fn open_filtered_context(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    stride: u32,
) -> AnyResult<FilteredSearchContext> {
    let (mask, selected_rows) = build_filter_mask(dataset, stride).await?;
    let metadata = dataset.load_indices().await?;
    let matching = metadata
        .iter()
        .filter(|index| index.name == index_name)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return Err(format!("dataset does not contain index {index_name:?}").into());
    }

    let mut indices = Vec::with_capacity(matching.len());
    for index in matching {
        let opened = dataset
            .open_scalar_index(column, &index.uuid, &NoOpMetricsCollector)
            .await?;
        if opened.as_any().downcast_ref::<InvertedIndex>().is_none() {
            return Err(format!("index segment {} is not an inverted index", index.uuid).into());
        }
        indices.push(opened);
    }
    Ok(FilteredSearchContext {
        indices,
        prefilter: Arc::new(StaticPreFilter { mask }),
        selected_rows,
    })
}

async fn global_scorer(
    indices: &[Arc<dyn ScalarIndex>],
    terms: &[String],
) -> AnyResult<MemBM25Scorer> {
    let mut total_tokens = 0_u64;
    let mut num_docs = 0_usize;
    let mut token_docs =
        HashMap::<String, usize>::from_iter(terms.iter().cloned().map(|term| (term, 0)));
    for index in indices {
        let inverted = index
            .as_any()
            .downcast_ref::<InvertedIndex>()
            .expect("index type checked while opening filtered benchmark context");
        let (segment_tokens, segment_docs, segment_token_docs) =
            inverted.bm25_stats_for_terms(terms).await?;
        total_tokens = total_tokens
            .checked_add(segment_tokens)
            .ok_or("global token count overflow")?;
        num_docs = num_docs
            .checked_add(segment_docs)
            .ok_or("global document count overflow")?;
        for (term, count) in terms.iter().zip(segment_token_docs) {
            *token_docs
                .get_mut(term)
                .expect("scorer term initialized above") += count;
        }
    }
    Ok(MemBM25Scorer::new(total_tokens, num_docs, token_docs))
}

async fn run_filtered_query(
    context: &FilteredSearchContext,
    query: &str,
    k: usize,
    allow_empty: bool,
) -> AnyResult<(f64, Vec<u64>, Vec<u32>)> {
    let started = Instant::now();
    let first = context
        .indices
        .first()
        .ok_or("filtered benchmark has no index segments")?
        .as_any()
        .downcast_ref::<InvertedIndex>()
        .expect("index type checked while opening filtered benchmark context");
    let mut tokenizer = first.tokenizer();
    let tokens = Arc::new(collect_query_tokens(query, &mut tokenizer));
    let mut terms = Vec::new();
    for token in tokens.as_ref() {
        if !terms.contains(token) {
            terms.push(token.clone());
        }
    }
    let scorer = Arc::new(global_scorer(&context.indices, &terms).await?);
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(k)));
    let metrics: Arc<dyn MetricsCollector> = Arc::new(NoOpMetricsCollector);
    let searches = context.indices.iter().cloned().map(|index| {
        let tokens = tokens.clone();
        let params = params.clone();
        let prefilter = context.prefilter.clone();
        let metrics = metrics.clone();
        let scorer = scorer.clone();
        async move {
            let inverted = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .expect("index type checked while opening filtered benchmark context");
            inverted
                .bm25_search(
                    tokens,
                    params,
                    Operator::Or,
                    prefilter,
                    metrics,
                    Some(scorer.as_ref()),
                )
                .await
        }
    });
    let segment_results = futures::stream::iter(searches)
        .buffer_unordered(context.indices.len().max(1))
        .try_collect::<Vec<_>>()
        .await?;

    let mut results = segment_results
        .into_iter()
        .flat_map(|(row_ids, scores)| row_ids.into_iter().zip(scores))
        .collect::<Vec<_>>();
    results.sort_unstable_by(|(left_row, left_score), (right_row, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_row.cmp(right_row))
    });
    results.truncate(k);
    let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let (row_ids, scores): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    if row_ids.is_empty() && !allow_empty {
        return Err(
            "filtered FTS query returned no rows; use --allow-empty only when intentional".into(),
        );
    }
    if row_ids
        .iter()
        .any(|row_id| !context.prefilter.mask.selected(*row_id))
    {
        return Err("filtered FTS query returned a row outside the synthetic mask".into());
    }
    Ok((
        latency_ms,
        row_ids,
        scores.into_iter().map(f32::to_bits).collect(),
    ))
}

async fn run_query(
    dataset: &Dataset,
    filtered: Option<&FilteredSearchContext>,
    column: &str,
    query: &str,
    k: usize,
    allow_empty: bool,
) -> AnyResult<(f64, Vec<u64>, Vec<u32>)> {
    match filtered {
        Some(context) => run_filtered_query(context, query, k, allow_empty).await,
        None => run_scanner_query(dataset, column, query, k, allow_empty).await,
    }
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
        "filter_stride": args.filter_stride,
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
    let object_store = dataset.object_store(None).await?;
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
        let _ = object_store.io_stats_incremental();
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
        let prewarm_io = object_store.io_stats_incremental();
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
            "read_iops": prewarm_io.read_iops,
            "read_bytes": prewarm_io.read_bytes,
            "write_iops": prewarm_io.write_iops,
            "written_bytes": prewarm_io.written_bytes,
            "rss_kib": linux_memory_kib("VmRSS:"),
            "peak_rss_kib": linux_memory_kib("VmHWM:"),
        }));
    }

    let filtered = if let Some(stride) = args.filter_stride {
        let index_name = args
            .expected_index
            .as_deref()
            .ok_or("--filter-stride requires --expected-index")?;
        let setup_started = Instant::now();
        let context =
            Arc::new(open_filtered_context(&dataset, &args.column, index_name, stride).await?);
        emit(json!({
            "event": "filter_ready",
            "label": args.label,
            "stride": stride,
            "selected_rows": context.selected_rows,
            "segment_count": context.indices.len(),
            "setup_ms": setup_started.elapsed().as_secs_f64() * 1_000.0,
            "rss_kib": linux_memory_kib("VmRSS:"),
            "peak_rss_kib": linux_memory_kib("VmHWM:"),
        }));
        Some(context)
    } else {
        None
    };

    let _ = object_store.io_stats_incremental();
    for _ in 0..args.warmup_rounds {
        for query in &queries {
            let _ = run_query(
                &dataset,
                filtered.as_deref(),
                &args.column,
                query,
                args.k,
                args.allow_empty,
            )
            .await?;
        }
    }
    let warmup_io = object_store.io_stats_incremental();
    emit(json!({
        "event": "warmup_complete",
        "label": args.label,
        "rounds": args.warmup_rounds,
        "read_iops": warmup_io.read_iops,
        "read_bytes": warmup_io.read_bytes,
        "write_iops": warmup_io.write_iops,
        "written_bytes": warmup_io.written_bytes,
    }));

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
            let filtered = filtered.clone();
            let column = args.column.clone();
            async move {
                let (latency_ms, row_ids, score_bits) = run_query(
                    &dataset,
                    filtered.as_deref(),
                    &column,
                    &query,
                    args.k,
                    args.allow_empty,
                )
                .await?;
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
    let measured_io = object_store.io_stats_incremental();
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
        "read_iops": measured_io.read_iops,
        "read_bytes": measured_io.read_bytes,
        "write_iops": measured_io.write_iops,
        "written_bytes": measured_io.written_bytes,
        "rss_kib": linux_memory_kib("VmRSS:"),
        "peak_rss_kib": linux_memory_kib("VmHWM:"),
    }));
    Ok(())
}
