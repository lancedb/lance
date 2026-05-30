// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Head-to-head KV point-lookup benchmark: Lance MemTable vs RocksDB.
//!
//! One binary times **both** engines with identical key/value/query sets and
//! identical timing code, so the comparison is apples-to-apples. The RocksDB
//! arm is compiled only with `--features bench-rocksdb` (bundled librocksdb);
//! without it the bench runs the Lance arm alone.
//!
//! Both engines hold all `--rows` rows in a **single in-memory write buffer**
//! (Lance: one active MemTable, ShardWriter configured to never flush;
//! RocksDB: one skiplist memtable, `write_buffer_size` above the dataset so no
//! SST flush). The table has a **BTree index on the key column**; the Lance
//! MemTable maintains it. We measure:
//!
//!   - **write throughput** (rows/sec) for a fixed shuffled insert order
//!   - **read latency** (p50/p95/p99/mean, single-thread) and **QPS**
//!     (single- and N-thread) for a query set mixing hits and guaranteed
//!     misses (`--miss-ratio`)
//!   - **CPU** (getrusage user+sys per phase) and **peak RSS** (sampled from
//!     `/proc/self/statm` on Linux)
//!
//! Example:
//!
//! ```bash
//! cargo bench -p lance --bench mem_wal_kv_point_lookup --features bench-rocksdb -- \
//!   --rows 1000000 --value-size 100 --queries 5000 --miss-ratio 0.5 \
//!   --threads 8 --engine both --uri /tmp/kv_bench --output result.json
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance::dataset::mem_wal::scanner::{
    InMemoryMemTableRef, LsmDataSourceCollector, LsmPointLookupPlanner, ShardSnapshot,
};
use lance::dataset::mem_wal::{DatasetMemWalExt, ShardWriterConfig};
use lance::dataset::{Dataset, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::Result;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use serde_json::json;
use uuid::Uuid;

const KEY_COL: &str = "id";
const VALUE_COL: &str = "value";
const BTREE_INDEX_NAME: &str = "id_btree";

// ----------------------------------------------------------------------
// Deterministic PRNG (SplitMix64) — no external rand dependency, identical
// key/query streams across engines and across runs given the same seed.
// ----------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, n)`.
    fn next_below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next_u64() % n
    }
}

/// Fisher-Yates shuffle of `0..n` driven by the deterministic PRNG.
fn shuffled_keys(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = SplitMix64::new(seed);
    let mut v: Vec<i64> = (0..n as i64).collect();
    for i in (1..n).rev() {
        let j = rng.next_below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
    v
}

/// Fixed-size ASCII payload derived from the key (valid UTF-8 so it can be
/// stored in both a Lance `Utf8` column and a RocksDB byte value unchanged).
fn make_value(key: i64, value_size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; value_size];
    let base = key as u64;
    for (i, b) in buf.iter_mut().enumerate() {
        *b = b'a' + ((base.wrapping_add(i as u64)) % 26) as u8;
    }
    buf
}

// ----------------------------------------------------------------------
// Query set: a mix of guaranteed hits (existing keys) and guaranteed misses
// (keys in [rows, 2*rows)). Same set fed to both engines.
// ----------------------------------------------------------------------

fn build_queries(rows: usize, queries: usize, miss_ratio: f64, seed: u64) -> Vec<(i64, bool)> {
    let mut rng = SplitMix64::new(seed ^ 0xD1B54A32D192ED03);
    let misses = ((queries as f64) * miss_ratio).round() as usize;
    let mut out = Vec::with_capacity(queries);
    for i in 0..queries {
        if i < misses {
            // Guaranteed absent: [rows, 2*rows)
            let k = rows as i64 + rng.next_below(rows.max(1) as u64) as i64;
            out.push((k, false));
        } else {
            let k = rng.next_below(rows.max(1) as u64) as i64;
            out.push((k, true));
        }
    }
    // Interleave so hits and misses aren't phase-separated.
    for i in (1..out.len()).rev() {
        let j = rng.next_below(i as u64 + 1) as usize;
        out.swap(i, j);
    }
    out
}

// ----------------------------------------------------------------------
// Lance direct BTree fast-path (bypasses DataFusion)
// ----------------------------------------------------------------------

/// Resolve a single key against the active MemTable's BTree index without
/// building a DataFusion plan: probe the index, honor the MVCC visibility
/// watermark, pick the newest matching row position, and slice that one row
/// out of the BatchStore. Returns `None` if the key isn't present/visible.
///
/// This mirrors what `BTreeIndexExec` does internally, minus the plan/stream
/// machinery — it is the lower bound on how fast the current MemTable index
/// can answer a point lookup. Single-active-memtable only (the bench never
/// flushes), `KEY_COL` BTree assumed present.
fn fast_lookup(active: &InMemoryMemTableRef, key: i64) -> Option<RecordBatch> {
    use arrow_array::Array;

    let btree = active.index_store.get_btree_by_column(KEY_COL)?;
    let max_vbp = active.index_store.max_visible_batch_position();

    // Highest visible row (exclusive end) across batches whose position is
    // within the watermark. Batch position == iteration index for a
    // never-flushed store.
    let mut visible_end: u64 = 0;
    for (bp, sb) in active.batch_store.iter().enumerate() {
        if bp <= max_vbp {
            visible_end += sb.num_rows as u64;
        } else {
            break;
        }
    }
    if visible_end == 0 {
        return None;
    }
    let max_visible_row = visible_end - 1;

    // Newest visible row position carrying this key (largest position wins).
    let pos = btree
        .get(&ScalarValue::Int64(Some(key)))
        .into_iter()
        .filter(|&p| p <= max_visible_row)
        .max()?;

    // Map the global position to (batch, row) and slice one row.
    let mut start: u64 = 0;
    for sb in active.batch_store.iter() {
        let end = start + sb.num_rows as u64;
        if pos >= start && pos < end {
            let row = (pos - start) as usize;
            let cols: Vec<_> = sb.data.columns().iter().map(|c| c.slice(row, 1)).collect();
            return RecordBatch::try_new(sb.data.schema(), cols).ok();
        }
        start = end;
    }
    None
}

// ----------------------------------------------------------------------
// Latency stats
// ----------------------------------------------------------------------

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((pct / 100.0) * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct LatencyStats {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    mean_us: f64,
}

/// `latencies_us` carries sub-microsecond precision (nanoseconds / 1000), so
/// RocksDB's sub-µs point gets don't collapse to 0.
fn compute_stats(mut latencies_us: Vec<f64>) -> LatencyStats {
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = latencies_us.iter().sum::<f64>() / latencies_us.len().max(1) as f64;
    LatencyStats {
        p50_us: percentile(&latencies_us, 50.0),
        p95_us: percentile(&latencies_us, 95.0),
        p99_us: percentile(&latencies_us, 99.0),
        mean_us: mean,
    }
}

// ----------------------------------------------------------------------
// Profiling: CPU (getrusage) + peak RSS (/proc/self/statm sampler)
// ----------------------------------------------------------------------

/// User+sys CPU seconds consumed by the whole process so far.
fn process_cpu_secs() -> f64 {
    // SAFETY: getrusage with a zeroed rusage out-param is always sound.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0.0;
        }
        let u = ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 / 1e6;
        let s = ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 / 1e6;
        u + s
    }
}

/// Current resident set size in bytes (0 if unavailable, e.g. non-Linux).
fn current_rss_bytes() -> u64 {
    // /proc/self/statm: field 2 is resident pages.
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let mut it = statm.split_whitespace();
    let _total = it.next();
    let Some(resident) = it.next().and_then(|s| s.parse::<u64>().ok()) else {
        return 0;
    };
    let page = 4096u64; // Linux default page size
    resident * page
}

/// Background thread sampling peak RSS until stopped.
struct RssSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(current_rss_bytes()));
        let stop_c = stop.clone();
        let peak_c = peak.clone();
        let handle = std::thread::spawn(move || {
            while !stop_c.load(Ordering::Relaxed) {
                let rss = current_rss_bytes();
                peak_c.fetch_max(rss, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }
    fn peak_mb(&self) -> f64 {
        self.peak.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }
    fn stop(mut self) -> f64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.peak_mb()
    }
}

// ----------------------------------------------------------------------
// Engine result
// ----------------------------------------------------------------------

#[derive(Clone)]
struct EngineResult {
    engine: &'static str,
    write_rows_per_s: f64,
    write_cpu_s: f64,
    read_p50_us: f64,
    read_p95_us: f64,
    read_p99_us: f64,
    read_mean_us: f64,
    read_qps_1t: f64,
    read_qps_nt: f64,
    read_cpu_s: f64,
    hits: usize,
    misses_resolved: usize,
    peak_rss_mb: f64,
    rss_after_load_mb: f64,
}

impl EngineResult {
    fn to_json(&self, args: &Args) -> serde_json::Value {
        json!({
            "engine": self.engine,
            "rows": args.rows,
            "value_size": args.value_size,
            "queries": args.queries,
            "miss_ratio": args.miss_ratio,
            "threads": args.threads,
            "write_rows_per_s": self.write_rows_per_s as u64,
            "write_cpu_s": format!("{:.3}", self.write_cpu_s),
            "read_p50_us": (self.read_p50_us * 1000.0).round() / 1000.0,
            "read_p95_us": (self.read_p95_us * 1000.0).round() / 1000.0,
            "read_p99_us": (self.read_p99_us * 1000.0).round() / 1000.0,
            "read_mean_us": (self.read_mean_us * 1000.0).round() / 1000.0,
            "read_qps_1t": self.read_qps_1t as u64,
            "read_qps_nt": self.read_qps_nt as u64,
            "read_cpu_s": format!("{:.3}", self.read_cpu_s),
            "hits": self.hits,
            "misses_resolved": self.misses_resolved,
            "peak_rss_mb": self.peak_rss_mb as u64,
            "rss_after_load_mb": self.rss_after_load_mb as u64,
        })
    }
}

// ----------------------------------------------------------------------
// CLI args
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engine {
    Lance,
    Rocksdb,
    Both,
}

/// How the Lance arm resolves a point lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LanceReadMode {
    /// Build + execute a DataFusion `ExecutionPlan` per lookup (production
    /// `LsmPointLookupPlanner::plan_lookup` path).
    Plan,
    /// Probe the active MemTable's BTree index directly and materialize the
    /// row from the BatchStore, bypassing DataFusion. Single-active-memtable
    /// fast path (no flushed generations); misses fall through as "not found".
    Fast,
    /// Call the production `LsmPointLookupPlanner::lookup` API, which uses the
    /// direct BTree fast path internally and falls back to the plan path for
    /// on-disk sources. Measures the real shipped point-lookup latency.
    Api,
}

impl LanceReadMode {
    fn parse(v: &str) -> std::result::Result<Self, String> {
        match v {
            "plan" => Ok(Self::Plan),
            "fast" => Ok(Self::Fast),
            "api" => Ok(Self::Api),
            _ => Err(format!(
                "unknown lance-read-mode '{v}', expected plan|fast|api"
            )),
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Fast => "fast",
            Self::Api => "api",
        }
    }
}

impl Engine {
    fn parse(v: &str) -> std::result::Result<Self, String> {
        match v {
            "lance" => Ok(Self::Lance),
            "rocksdb" => Ok(Self::Rocksdb),
            "both" => Ok(Self::Both),
            _ => Err(format!("unknown engine '{v}', expected lance|rocksdb|both")),
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    rows: usize,
    value_size: usize,
    queries: usize,
    miss_ratio: f64,
    threads: usize,
    batch_rows: usize,
    engine: Engine,
    lance_read_mode: LanceReadMode,
    uri: String,
    seed: u64,
    /// Skip the RocksDB WAL on writes. Off by default so RocksDB writes a WAL
    /// like Lance's durable MemTable path, keeping the write comparison fair.
    rocksdb_disable_wal: bool,
    output: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            rows: 1_000_000,
            value_size: 100,
            queries: 5_000,
            miss_ratio: 0.5,
            threads: 8,
            batch_rows: 1_000,
            engine: Engine::Both,
            lance_read_mode: LanceReadMode::Plan,
            uri: String::new(),
            seed: 0x5EED,
            rocksdb_disable_wal: false,
            output: None,
        }
    }
}

fn parse_val<T>(flag: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|e| lance_core::Error::invalid_input(format!("invalid {flag}: {value} ({e})")))
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    let mut has_uri = false;
    while let Some(flag) = iter.next() {
        if flag == "--bench" {
            continue;
        }
        if flag == "--rocksdb-disable-wal" {
            args.rocksdb_disable_wal = true;
            continue;
        }
        let value = iter
            .next()
            .ok_or_else(|| lance_core::Error::invalid_input(format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--rows" => args.rows = parse_val(&flag, &value)?,
            "--value-size" => args.value_size = parse_val(&flag, &value)?,
            "--queries" => args.queries = parse_val(&flag, &value)?,
            "--miss-ratio" => args.miss_ratio = parse_val(&flag, &value)?,
            "--threads" => args.threads = parse_val(&flag, &value)?,
            "--batch-rows" => args.batch_rows = parse_val(&flag, &value)?,
            "--engine" => {
                args.engine = Engine::parse(&value).map_err(lance_core::Error::invalid_input)?
            }
            "--lance-read-mode" => {
                args.lance_read_mode =
                    LanceReadMode::parse(&value).map_err(lance_core::Error::invalid_input)?
            }
            "--uri" => {
                args.uri = value;
                has_uri = true;
            }
            "--seed" => args.seed = parse_val(&flag, &value)?,
            "--output" => args.output = Some(PathBuf::from(value)),
            _ => {
                return Err(lance_core::Error::invalid_input(format!(
                    "unknown argument: {flag}"
                )));
            }
        }
    }
    if !has_uri {
        return Err(lance_core::Error::invalid_input("--uri is required"));
    }
    if args.rows == 0 || args.batch_rows == 0 || args.value_size == 0 || args.queries == 0 {
        return Err(lance_core::Error::invalid_input(
            "rows, batch-rows, value-size, queries must be > 0",
        ));
    }
    if !(0.0..=1.0).contains(&args.miss_ratio) {
        return Err(lance_core::Error::invalid_input(
            "miss-ratio must be in [0, 1]",
        ));
    }
    Ok(args)
}

// ----------------------------------------------------------------------
// Schema / batch helpers (Lance)
// ----------------------------------------------------------------------

fn make_schema() -> Arc<ArrowSchema> {
    let mut id_meta = HashMap::new();
    id_meta.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    Arc::new(ArrowSchema::new(vec![
        Field::new(KEY_COL, DataType::Int64, false).with_metadata(id_meta),
        Field::new(VALUE_COL, DataType::Utf8, true),
    ]))
}

fn make_batch(schema: Arc<ArrowSchema>, keys: &[i64], value_size: usize) -> RecordBatch {
    let ids = Int64Array::from_iter_values(keys.iter().copied());
    let values: Vec<String> = keys
        .iter()
        .map(|k| {
            // make_value is valid ASCII, so from_utf8 never fails.
            String::from_utf8(make_value(*k, value_size)).unwrap()
        })
        .collect();
    let value_arr = StringArray::from_iter_values(values);
    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(value_arr)]).unwrap()
}

// ----------------------------------------------------------------------
// Lance engine
// ----------------------------------------------------------------------

async fn run_lance(
    args: &Args,
    insert_order: &[i64],
    queries: &[(i64, bool)],
) -> Result<EngineResult> {
    let sampler = RssSampler::start();
    let schema = make_schema();

    // 1-row sentinel base dataset (id = -1) so the lookup path is effectively
    // MemTable-only: query keys are 0..rows, never in the base table. The
    // base only ever contributes a 1-row scan on the miss path.
    let base_uri = format!("{}/base", args.uri.trim_end_matches('/'));
    let sentinel = make_batch(schema.clone(), &[-1], args.value_size);
    let reader = RecordBatchIterator::new([Ok(sentinel)], schema.clone());
    let mut dataset = Dataset::write(reader, &base_uri, Some(WriteParams::default())).await?;

    // BTree index on the key column, maintained by the MemTable.
    dataset
        .create_index(
            &[KEY_COL],
            IndexType::BTree,
            Some(BTREE_INDEX_NAME.to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await?;
    dataset
        .initialize_mem_wal()
        .maintained_indexes([BTREE_INDEX_NAME])
        .execute()
        .await?;

    let dataset = Arc::new(dataset);
    let arrow_schema: Arc<ArrowSchema> = Arc::new(ArrowSchema::from(dataset.schema()));

    // No-flush config: every *memtable*-flush threshold is set above the
    // dataset so the single active MemTable holds all rows (no generation is
    // sealed to disk). Read visibility is gated on the WAL durability
    // watermark (`max_visible_batch_position`), which only advances on a WAL
    // flush — so we use `durable_write=true`: each put flushes its batch to
    // the WAL and awaits, which both populates the maintained BTree and
    // advances the watermark, leaving every row visible the moment the write
    // loop ends (no background-drain race). This is the durable ingestion
    // path; per the goal it is acceptable for writes to be slower than
    // RocksDB. The RocksDB arm keeps its WAL on by default too (see
    // `--rocksdb-disable-wal`) so the write comparison is apples-to-apples.
    let shard_id = Uuid::new_v4();
    let big = args.rows.saturating_mul(args.value_size + 256).max(1 << 30);
    let config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: true,
        sync_indexed_write: true,
        max_memtable_size: big,
        max_memtable_rows: args.rows * 4 + 1_000_000,
        max_memtable_batches: args.rows / args.batch_rows + 1_000_000,
        max_unflushed_memtable_bytes: big,
        max_wal_flush_interval: Some(Duration::from_millis(100)),
        ..ShardWriterConfig::default()
    };
    let writer = dataset.mem_wal_writer(shard_id, config).await?;

    // --- write phase ---
    let cpu0 = process_cpu_secs();
    let t_write = Instant::now();
    let mut lo = 0usize;
    while lo < insert_order.len() {
        let hi = (lo + args.batch_rows).min(insert_order.len());
        let batch = make_batch(schema.clone(), &insert_order[lo..hi], args.value_size);
        writer.put(vec![batch]).await?;
        lo = hi;
    }
    let write_s = t_write.elapsed().as_secs_f64();
    let write_cpu_s = process_cpu_secs() - cpu0;
    let write_rows_per_s = args.rows as f64 / write_s.max(1e-9);
    let rss_after_load_mb = sampler.peak_mb();
    println!(
        "[lance] wrote {} rows in {:.2}s = {:.0} rows/s (cpu {:.2}s)",
        args.rows, write_s, write_rows_per_s, write_cpu_s
    );

    // Build the point-lookup planner over base + active MemTable.
    let manifest = writer.manifest().await?;
    let in_memory_refs = writer.in_memory_memtable_refs().await?;
    let mut shard_snapshot = ShardSnapshot::new(shard_id);
    if let Some(ref m) = manifest {
        shard_snapshot = shard_snapshot.with_current_generation(m.current_generation);
        for fg in &m.flushed_generations {
            shard_snapshot = shard_snapshot.with_flushed_generation(fg.generation, fg.path.clone());
        }
    }
    // Keep a handle to the active MemTable for the direct fast path before
    // the collector takes ownership of the refs.
    let active = Arc::new(in_memory_refs.active.clone());
    let collector = LsmDataSourceCollector::new(dataset.clone(), vec![shard_snapshot])
        .with_in_memory_memtables(shard_id, in_memory_refs);
    let planner = Arc::new(LsmPointLookupPlanner::new(
        collector,
        vec![KEY_COL.to_string()],
        arrow_schema,
    ));

    // Warmup + correctness: a hit key must resolve to exactly one row under
    // whichever read mode we're timing.
    {
        let probe = insert_order[insert_order.len() / 2];
        let n = match args.lance_read_mode {
            LanceReadMode::Plan => {
                let plan = planner
                    .plan_lookup(&[ScalarValue::Int64(Some(probe))], None)
                    .await?;
                let ctx = SessionContext::new();
                let batches: Vec<RecordBatch> =
                    plan.execute(0, ctx.task_ctx())?.try_collect().await?;
                batches.iter().map(|b| b.num_rows()).sum::<usize>()
            }
            LanceReadMode::Fast => fast_lookup(&active, probe)
                .map(|b| b.num_rows())
                .unwrap_or(0),
            LanceReadMode::Api => planner
                .lookup(&[ScalarValue::Int64(Some(probe))], None)
                .await?
                .map(|b| b.num_rows())
                .unwrap_or(0),
        };
        assert_eq!(n, 1, "warmup lookup for key {probe} returned {n} rows");
    }

    // --- read phase: single-thread latency + hit/miss accounting ---
    let cpu1 = process_cpu_secs();
    let ctx = SessionContext::new();
    let task_ctx = ctx.task_ctx();
    let mut latencies_us = Vec::with_capacity(queries.len());
    let mut hits = 0usize;
    let mut misses_resolved = 0usize;
    let t_read = Instant::now();
    for &(key, expect_hit) in queries {
        let t0 = Instant::now();
        let n = match args.lance_read_mode {
            LanceReadMode::Plan => {
                let plan = planner
                    .plan_lookup(&[ScalarValue::Int64(Some(key))], None)
                    .await?;
                let batches: Vec<RecordBatch> =
                    plan.execute(0, task_ctx.clone())?.try_collect().await?;
                batches.iter().map(|b| b.num_rows()).sum::<usize>()
            }
            LanceReadMode::Fast => fast_lookup(&active, key).map(|b| b.num_rows()).unwrap_or(0),
            LanceReadMode::Api => planner
                .lookup(&[ScalarValue::Int64(Some(key))], None)
                .await?
                .map(|b| b.num_rows())
                .unwrap_or(0),
        };
        latencies_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        if expect_hit {
            assert_eq!(n, 1, "expected hit for key {key}, got {n}");
            hits += 1;
        } else {
            assert_eq!(n, 0, "expected miss for key {key}, got {n}");
            misses_resolved += 1;
        }
    }
    let read_1t_s = t_read.elapsed().as_secs_f64();
    let read_qps_1t = queries.len() as f64 / read_1t_s.max(1e-9);
    let read_cpu_s = process_cpu_secs() - cpu1;
    let stats = compute_stats(latencies_us);

    // --- read phase: N-thread QPS ---
    let keys: Arc<Vec<i64>> = Arc::new(queries.iter().map(|(k, _)| *k).collect());
    let read_qps_nt = if args.threads <= 1 {
        read_qps_1t
    } else if args.lance_read_mode == LanceReadMode::Fast {
        // Direct path is synchronous; fan out over OS threads like RocksDB.
        let t = Instant::now();
        let mut handles = Vec::with_capacity(args.threads);
        for shard in 0..args.threads {
            let active = active.clone();
            let keys = keys.clone();
            let threads = args.threads;
            handles.push(std::thread::spawn(move || {
                let mut i = shard;
                while i < keys.len() {
                    std::hint::black_box(fast_lookup(&active, keys[i]));
                    i += threads;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        keys.len() as f64 / t.elapsed().as_secs_f64().max(1e-9)
    } else {
        // Plan and Api are async; fan out over tokio tasks.
        let mode = args.lance_read_mode;
        let t = Instant::now();
        let mut handles = Vec::with_capacity(args.threads);
        for shard in 0..args.threads {
            let planner = planner.clone();
            let keys = keys.clone();
            let threads = args.threads;
            handles.push(tokio::spawn(async move {
                let ctx = SessionContext::new();
                let task_ctx = ctx.task_ctx();
                let mut done = 0usize;
                let mut i = shard;
                while i < keys.len() {
                    match mode {
                        LanceReadMode::Api => {
                            std::hint::black_box(
                                planner
                                    .lookup(&[ScalarValue::Int64(Some(keys[i]))], None)
                                    .await
                                    .unwrap(),
                            );
                        }
                        _ => {
                            let plan = planner
                                .plan_lookup(&[ScalarValue::Int64(Some(keys[i]))], None)
                                .await
                                .unwrap();
                            let _b: Vec<RecordBatch> = plan
                                .execute(0, task_ctx.clone())
                                .unwrap()
                                .try_collect()
                                .await
                                .unwrap();
                        }
                    }
                    done += 1;
                    i += threads;
                }
                done
            }));
        }
        let mut total = 0usize;
        for h in handles {
            total += h.await.unwrap();
        }
        let s = t.elapsed().as_secs_f64();
        total as f64 / s.max(1e-9)
    };

    let peak_rss_mb = sampler.stop();
    println!(
        "[lance] read p50={:.2}us p95={:.2}us p99={:.2}us mean={:.2}us qps_1t={:.0} qps_{}t={:.0} (hits={} miss={}) peak_rss={:.0}MB",
        stats.p50_us,
        stats.p95_us,
        stats.p99_us,
        stats.mean_us,
        read_qps_1t,
        args.threads,
        read_qps_nt,
        hits,
        misses_resolved,
        peak_rss_mb
    );

    // Keep the active MemTable Arcs alive through the read phase; forget the
    // writer to skip its async-in-sync-drop path (mirrors sibling benches).
    std::mem::forget(writer);

    Ok(EngineResult {
        engine: match args.lance_read_mode {
            LanceReadMode::Plan => "lance",
            LanceReadMode::Fast => "lance-fast",
            LanceReadMode::Api => "lance-api",
        },
        write_rows_per_s,
        write_cpu_s,
        read_p50_us: stats.p50_us,
        read_p95_us: stats.p95_us,
        read_p99_us: stats.p99_us,
        read_mean_us: stats.mean_us,
        read_qps_1t,
        read_qps_nt,
        read_cpu_s,
        hits,
        misses_resolved,
        peak_rss_mb,
        rss_after_load_mb,
    })
}

// ----------------------------------------------------------------------
// RocksDB engine (only with --features bench-rocksdb)
// ----------------------------------------------------------------------

#[cfg(feature = "bench-rocksdb")]
fn run_rocksdb(args: &Args, insert_order: &[i64], queries: &[(i64, bool)]) -> Result<EngineResult> {
    use rocksdb::{DB, Options, WriteBatch, WriteOptions};

    let sampler = RssSampler::start();
    let db_path = format!("{}/rocksdb", args.uri.trim_end_matches('/'));
    let _ = std::fs::remove_dir_all(&db_path);

    // In-memory tuning: one skiplist memtable holds every row, no SST flush,
    // no auto compaction. write_buffer_size is set above the whole dataset.
    let write_buf = args.rows * (args.value_size + 200) + (64 << 20);
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(write_buf);
    opts.set_max_write_buffer_number(4);
    opts.set_min_write_buffer_number_to_merge(2);
    opts.set_disable_auto_compactions(true);
    // Never trigger a flush by buffer count either.
    opts.set_db_write_buffer_size(write_buf);

    let db = Arc::new(
        DB::open(&opts, &db_path)
            .map_err(|e| lance_core::Error::io(format!("rocksdb open: {e}")))?,
    );

    let mut wo = WriteOptions::default();
    // Default: WAL on (durable), matching Lance's durable_write=true path.
    // `--rocksdb-disable-wal` opts into RocksDB's faster no-WAL writes.
    wo.disable_wal(args.rocksdb_disable_wal);

    // --- write phase ---
    let cpu0 = process_cpu_secs();
    let t_write = Instant::now();
    let mut lo = 0usize;
    while lo < insert_order.len() {
        let hi = (lo + args.batch_rows).min(insert_order.len());
        let mut wb = WriteBatch::default();
        for &k in &insert_order[lo..hi] {
            wb.put(k.to_be_bytes(), make_value(k, args.value_size));
        }
        db.write_opt(wb, &wo)
            .map_err(|e| lance_core::Error::io(format!("rocksdb write: {e}")))?;
        lo = hi;
    }
    let write_s = t_write.elapsed().as_secs_f64();
    let write_cpu_s = process_cpu_secs() - cpu0;
    let write_rows_per_s = args.rows as f64 / write_s.max(1e-9);
    let rss_after_load_mb = sampler.peak_mb();
    println!(
        "[rocksdb] wrote {} rows in {:.2}s = {:.0} rows/s (cpu {:.2}s, write_buf {}MB)",
        args.rows,
        write_s,
        write_rows_per_s,
        write_cpu_s,
        write_buf >> 20
    );

    // --- read phase: single-thread latency ---
    let cpu1 = process_cpu_secs();
    let mut latencies_us = Vec::with_capacity(queries.len());
    let mut hits = 0usize;
    let mut misses_resolved = 0usize;
    let t_read = Instant::now();
    for &(key, expect_hit) in queries {
        let t0 = Instant::now();
        let got = db
            .get(key.to_be_bytes())
            .map_err(|e| lance_core::Error::io(format!("rocksdb get: {e}")))?;
        latencies_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        if expect_hit {
            assert!(got.is_some(), "expected hit for key {key}");
            hits += 1;
        } else {
            assert!(got.is_none(), "expected miss for key {key}");
            misses_resolved += 1;
        }
    }
    let read_1t_s = t_read.elapsed().as_secs_f64();
    let read_qps_1t = queries.len() as f64 / read_1t_s.max(1e-9);
    let read_cpu_s = process_cpu_secs() - cpu1;
    let stats = compute_stats(latencies_us);

    // --- read phase: N-thread QPS ---
    let read_qps_nt = if args.threads > 1 {
        let keys: Arc<Vec<i64>> = Arc::new(queries.iter().map(|(k, _)| *k).collect());
        let t = Instant::now();
        let mut handles = Vec::with_capacity(args.threads);
        for shard in 0..args.threads {
            let db = db.clone();
            let keys = keys.clone();
            let threads = args.threads;
            handles.push(std::thread::spawn(move || {
                let mut i = shard;
                while i < keys.len() {
                    let _ = db.get(keys[i].to_be_bytes()).unwrap();
                    i += threads;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let s = t.elapsed().as_secs_f64();
        keys.len() as f64 / s.max(1e-9)
    } else {
        read_qps_1t
    };

    let peak_rss_mb = sampler.stop();
    println!(
        "[rocksdb] read p50={:.2}us p95={:.2}us p99={:.2}us mean={:.2}us qps_1t={:.0} qps_{}t={:.0} (hits={} miss={}) peak_rss={:.0}MB",
        stats.p50_us,
        stats.p95_us,
        stats.p99_us,
        stats.mean_us,
        read_qps_1t,
        args.threads,
        read_qps_nt,
        hits,
        misses_resolved,
        peak_rss_mb
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&db_path);

    Ok(EngineResult {
        engine: "rocksdb",
        write_rows_per_s,
        write_cpu_s,
        read_p50_us: stats.p50_us,
        read_p95_us: stats.p95_us,
        read_p99_us: stats.p99_us,
        read_mean_us: stats.mean_us,
        read_qps_1t,
        read_qps_nt,
        read_cpu_s,
        hits,
        misses_resolved,
        peak_rss_mb,
        rss_after_load_mb,
    })
}

#[cfg(not(feature = "bench-rocksdb"))]
fn run_rocksdb(
    _args: &Args,
    _insert_order: &[i64],
    _queries: &[(i64, bool)],
) -> Result<EngineResult> {
    Err(lance_core::Error::invalid_input(
        "RocksDB arm not compiled; rebuild with --features bench-rocksdb",
    ))
}

// ----------------------------------------------------------------------
// Entrypoint
// ----------------------------------------------------------------------

fn print_comparison(results: &[EngineResult]) {
    println!("\n=== comparison ===");
    println!(
        "{:>9} {:>14} {:>12} {:>12} {:>12} {:>12} {:>12} {:>11} {:>11}",
        "engine",
        "write_rows/s",
        "rd_p50_us",
        "rd_p95_us",
        "rd_p99_us",
        "qps_1t",
        "qps_nt",
        "rss_mb",
        "rd_cpu_s"
    );
    for r in results {
        println!(
            "{:>9} {:>14.0} {:>12.2} {:>12.2} {:>12.2} {:>12.0} {:>12.0} {:>11.0} {:>11.3}",
            r.engine,
            r.write_rows_per_s,
            r.read_p50_us,
            r.read_p95_us,
            r.read_p99_us,
            r.read_qps_1t,
            r.read_qps_nt,
            r.peak_rss_mb,
            r.read_cpu_s,
        );
    }
    // Ratios when both ran.
    if let (Some(l), Some(rdb)) = (
        results.iter().find(|r| r.engine.starts_with("lance")),
        results.iter().find(|r| r.engine == "rocksdb"),
    ) {
        let safe = |a: f64, b: f64| if b > 0.0 { a / b } else { f64::NAN };
        println!(
            "\nlance/rocksdb ratios: write={:.2}x  read_p50={:.2}x  qps_1t={:.2}x  qps_nt={:.2}x  rss={:.2}x",
            safe(l.write_rows_per_s, rdb.write_rows_per_s),
            safe(l.read_p50_us, rdb.read_p50_us),
            safe(l.read_qps_1t, rdb.read_qps_1t),
            safe(l.read_qps_nt, rdb.read_qps_nt),
            safe(l.peak_rss_mb, rdb.rss_after_load_mb.max(rdb.peak_rss_mb)),
        );
        println!("(write/qps >1 = lance faster; read_p50/rss <1 = lance better)");
    }
}

async fn run(args: Args) -> Result<()> {
    println!(
        "bench=mem_wal_kv_point_lookup engine={:?} lance_read_mode={} rows={} value_size={} queries={} miss_ratio={} threads={} batch_rows={} uri={}",
        args.engine,
        args.lance_read_mode.as_str(),
        args.rows,
        args.value_size,
        args.queries,
        args.miss_ratio,
        args.threads,
        args.batch_rows,
        args.uri
    );

    let insert_order = shuffled_keys(args.rows, args.seed);
    let queries = build_queries(args.rows, args.queries, args.miss_ratio, args.seed);

    let mut results = Vec::new();
    if matches!(args.engine, Engine::Lance | Engine::Both) {
        results.push(run_lance(&args, &insert_order, &queries).await?);
    }
    if matches!(args.engine, Engine::Rocksdb | Engine::Both) {
        // RocksDB arm is synchronous; run it on a blocking thread so it does
        // not stall the tokio reactor.
        let a = args.clone();
        let io = insert_order.clone();
        let q = queries.clone();
        let res = tokio::task::spawn_blocking(move || run_rocksdb(&a, &io, &q))
            .await
            .map_err(|e| lance_core::Error::io(format!("rocksdb join: {e}")))??;
        results.push(res);
    }

    print_comparison(&results);

    let out = json!({
        "bench": "mem_wal_kv_point_lookup",
        "rows": args.rows,
        "value_size": args.value_size,
        "queries": args.queries,
        "miss_ratio": args.miss_ratio,
        "threads": args.threads,
        "results": results.iter().map(|r| r.to_json(&args)).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&out)
        .map_err(|e| lance_core::Error::io(format!("serialize: {e}")))?;
    if let Some(path) = &args.output {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, text.as_bytes())
            .map_err(|e| lance_core::Error::io(format!("write {}: {e}", path.display())))?;
    }
    println!("\n{text}");
    println!("=== DONE ===");
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| lance_core::Error::io(format!("build runtime: {e}")))?;
    runtime.block_on(run(args))
}
