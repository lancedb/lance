// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end benchmark for the SWMR FTS mem index using HuggingFace fineweb.
//!
//! For a single configuration, this binary:
//!  1. Downloads (and caches) one fineweb sample shard.
//!  2. Slices `BENCH_BASE_ROWS + BENCH_INGEST_ROWS` rows from it.
//!  3. (Once per `BENCH_RUN_ID`) writes the base 1M-row Lance dataset to
//!     `${DATASET_PREFIX}/${BENCH_RUN_ID}/base/`.
//!  4. Ingests `BENCH_INGEST_ROWS` rows through `ShardWriter` with the
//!     configured `max_memtable_rows`, `DURABLE_WRITE`, and `FTS_ENABLED`.
//!     Records write throughput.
//!  5. If FTS is enabled and `--with-read-test` is set: ingests
//!     `max_memtable_rows` rows into a fresh dataset with auto-flush
//!     disabled, queries the MemTable for 150 prebuilt queries (latency),
//!     forces a flush, queries the on-disk FTS for the same queries, and
//!     reports the top-10 set overlap (the user-approved "consistency"
//!     proxy for recall).
//!
//! Writes a structured `result.json` to `${RESULT_FILE}`. All paths,
//! credentials, and tunables are env-var driven so the same binary drives
//! all 12 configs from a shell loop.
//!
//! See `~/ai/analysis/lance/jack-MemTableFTSBetter/fineweb-fts-bench/DESIGN.md`.

#![recursion_limit = "256"]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance::dataset::mem_wal::index::{FtsQueryExpr, SearchOptions};
use lance::dataset::mem_wal::write::ShardWriterConfig;
use lance::dataset::mem_wal::{DatasetMemWalExt, MemWalConfig};
use lance::dataset::{Dataset, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::scalar::{FullTextSearchQuery, ScalarIndexParams};
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use serde::Serialize;
use uuid::Uuid;

const TEXT_COL: &str = "text";
const FTS_INDEX_NAME: &str = "text_fts";
const HF_API_LISTING: &str =
    "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT";
const HF_FILE_BASE: &str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/";

// ----------------------------------------------------------------------
// Configuration (env-driven)
// ----------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Config {
    dataset_prefix: String,
    run_id: String,
    config_name: String,
    max_memtable_rows: usize,
    durable_write: bool,
    fts_enabled: bool,
    base_rows: usize,
    ingest_rows: usize,
    batch_size: usize,
    cache_dir: std::path::PathBuf,
    result_file: std::path::PathBuf,
    /// When true, run the read-perf + consistency sub-test in addition to
    /// the throughput test. Auto-disabled when `fts_enabled = false`.
    with_read_test: bool,
    /// How many high-frequency single-token queries to include.
    num_token_queries: usize,
    /// How many random 2-token phrase queries to include.
    num_phrase_queries: usize,
    /// Top-K used for read latency and consistency.
    top_k: usize,
}

impl Config {
    fn from_env() -> Self {
        let dataset_prefix = std::env::var("DATASET_PREFIX")
            .unwrap_or_else(|_| "/tmp/bench/mem_fts_fineweb".to_string());
        let run_id = std::env::var("BENCH_RUN_ID").unwrap_or_else(|_| "dev".to_string());
        let max_memtable_rows = env_usize("BENCH_MAX_MEMTABLE_ROWS", 100_000);
        let durable_write = env_bool("DURABLE_WRITE", false);
        let fts_enabled = env_bool("FTS_ENABLED", false);
        let base_rows = env_usize("BENCH_BASE_ROWS", 1_000_000);
        let ingest_rows = env_usize("BENCH_INGEST_ROWS", 1_000_000);
        let batch_size = env_usize("BENCH_BATCH_SIZE", 1000);
        let cache_dir = std::env::var("BENCH_CACHE_DIR")
            .unwrap_or_else(|_| {
                std::env::temp_dir()
                    .join("mem_wal_fineweb_fts_cache")
                    .to_string_lossy()
                    .into_owned()
            })
            .into();
        let result_file = std::env::var("RESULT_FILE")
            .unwrap_or_else(|_| "result.json".to_string())
            .into();
        let with_read_test = env_bool("BENCH_WITH_READ_TEST", true) && fts_enabled;
        let num_token_queries = env_usize("BENCH_NUM_TOKEN_QUERIES", 100);
        let num_phrase_queries = env_usize("BENCH_NUM_PHRASE_QUERIES", 50);
        let top_k = env_usize("BENCH_TOP_K", 10);

        let config_name = format!(
            "mt{}_durable{}_fts{}",
            human_size(max_memtable_rows),
            if durable_write { "1" } else { "0" },
            if fts_enabled { "1" } else { "0" },
        );

        Self {
            dataset_prefix,
            run_id,
            config_name,
            max_memtable_rows,
            durable_write,
            fts_enabled,
            base_rows,
            ingest_rows,
            batch_size,
            cache_dir,
            result_file,
            with_read_test,
            num_token_queries,
            num_phrase_queries,
            top_k,
        }
    }

    #[allow(dead_code)]
    fn base_uri(&self) -> String {
        format!("{}/{}/base", self.dataset_prefix, self.run_id)
    }

    fn ingest_uri(&self) -> String {
        format!(
            "{}/{}/ingest_{}",
            self.dataset_prefix, self.run_id, self.config_name
        )
    }

    fn read_test_uri(&self) -> String {
        format!(
            "{}/{}/readtest_{}",
            self.dataset_prefix, self.run_id, self.config_name
        )
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key).ok().as_deref() {
        Some("1") | Some("yes") | Some("true") | Some("YES") | Some("TRUE") => true,
        Some("0") | Some("no") | Some("false") | Some("NO") | Some("FALSE") => false,
        _ => default,
    }
}

fn human_size(n: usize) -> String {
    if n % 1_000_000 == 0 {
        format!("{}M", n / 1_000_000)
    } else if n % 1_000 == 0 {
        format!("{}k", n / 1_000)
    } else {
        n.to_string()
    }
}

// ----------------------------------------------------------------------
// HF fineweb shard loading
// ----------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
}

async fn list_shard_paths() -> lance_core::Result<Vec<String>> {
    let entries: Vec<HfTreeEntry> = reqwest::get(HF_API_LISTING)
        .await
        .map_err(|e| lance_core::Error::io(format!("listing HTTP: {}", e)))?
        .json()
        .await
        .map_err(|e| lance_core::Error::io(format!("listing JSON: {}", e)))?;
    let mut shards: Vec<String> = entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.ends_with(".parquet"))
        .map(|e| e.path)
        .collect();
    shards.sort();
    Ok(shards)
}

async fn download_shard(rel_path: &str, dest: &std::path::Path) -> lance_core::Result<()> {
    if dest.exists() {
        return Ok(());
    }
    let url = format!("{}{}", HF_FILE_BASE, rel_path);
    let max_attempts = 5;
    for attempt in 1..=max_attempts {
        println!(
            "downloading {} (attempt {}/{}) ...",
            rel_path, attempt, max_attempts
        );
        let result: lance_core::Result<bytes::Bytes> = async {
            let resp = reqwest::get(&url)
                .await
                .map_err(|e| lance_core::Error::io(format!("download HTTP: {}", e)))?;
            if !resp.status().is_success() {
                return Err(lance_core::Error::io(format!(
                    "download {} → status {}",
                    url,
                    resp.status()
                )));
            }
            resp.bytes()
                .await
                .map_err(|e| lance_core::Error::io(format!("read body: {}", e)))
        }
        .await;
        match result {
            Ok(bytes) => {
                std::fs::write(dest, &bytes)
                    .map_err(|e| lance_core::Error::io(format!("write: {}", e)))?;
                println!(
                    "  wrote {:.1} MB to {}",
                    bytes.len() as f64 / 1024.0 / 1024.0,
                    dest.display()
                );
                return Ok(());
            }
            Err(e) if attempt < max_attempts => {
                let backoff = Duration::from_secs(2u64.pow(attempt as u32));
                eprintln!(
                    "  attempt {} failed: {}; retrying in {:?}",
                    attempt, e, backoff
                );
                tokio::time::sleep(backoff).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

async fn read_shard_text(
    path: &std::path::Path,
    out: &mut Vec<String>,
    max_rows: usize,
) -> lance_core::Result<usize> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| lance_core::Error::io(format!("open parquet: {}", e)))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file)
        .await
        .map_err(|e| lance_core::Error::io(format!("parquet builder: {}", e)))?;
    let mut stream = builder
        .build()
        .map_err(|e| lance_core::Error::io(format!("parquet stream: {}", e)))?;

    let mut taken = 0usize;
    while taken < max_rows {
        let Some(rb) = stream
            .try_next()
            .await
            .map_err(|e| lance_core::Error::io(format!("parquet read: {}", e)))?
        else {
            break;
        };
        let col = rb
            .column_by_name("text")
            .ok_or_else(|| lance_core::Error::io("text column missing".to_string()))?;
        let strs = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| lance_core::Error::io("text column not StringArray".to_string()))?;
        for i in 0..strs.len() {
            if taken >= max_rows {
                break;
            }
            if strs.is_null(i) {
                continue;
            }
            out.push(strs.value(i).to_string());
            taken += 1;
        }
    }
    Ok(taken)
}

async fn load_corpus(
    needed_rows: usize,
    cache_dir: &std::path::Path,
) -> lance_core::Result<Vec<String>> {
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| lance_core::Error::io(format!("mkdir cache: {}", e)))?;
    let shards = list_shard_paths().await?;
    println!("fineweb sample/10BT has {} parquet shards", shards.len());

    let mut buf: Vec<String> = Vec::with_capacity(needed_rows);
    for rel_path in &shards {
        if buf.len() >= needed_rows {
            break;
        }
        let local_name = rel_path.rsplit('/').next().unwrap_or(rel_path);
        let local = cache_dir.join(local_name);
        download_shard(rel_path, &local).await?;
        let want = needed_rows - buf.len();
        let got = read_shard_text(&local, &mut buf, want).await?;
        println!(
            "  shard {} → {} text rows (cumulative {})",
            local_name,
            got,
            buf.len()
        );
    }
    if buf.len() < needed_rows {
        eprintln!(
            "  warning: dataset exhausted at {} rows (asked {})",
            buf.len(),
            needed_rows
        );
    }
    Ok(buf)
}

// ----------------------------------------------------------------------
// Schema + batch helpers
// ----------------------------------------------------------------------

fn make_schema() -> Arc<ArrowSchema> {
    let mut id_meta = HashMap::new();
    id_meta.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    let id = Field::new("id", DataType::Int64, false).with_metadata(id_meta);
    Arc::new(ArrowSchema::new(vec![
        id,
        Field::new(TEXT_COL, DataType::Utf8, true),
    ]))
}

fn make_batch(start_id: i64, texts: &[String], schema: Arc<ArrowSchema>) -> RecordBatch {
    let n = texts.len();
    let ids: Vec<i64> = (start_id..start_id + n as i64).collect();
    let id_arr: ArrayRef = Arc::new(Int64Array::from(ids));
    let text_arr: ArrayRef = Arc::new(StringArray::from(texts.to_vec()));
    RecordBatch::try_new(schema, vec![id_arr, text_arr]).unwrap()
}

// ----------------------------------------------------------------------
// Base dataset
// ----------------------------------------------------------------------

async fn build_base_if_absent(
    base_uri: &str,
    schema: Arc<ArrowSchema>,
    base_texts: &[String],
    batch_size: usize,
    fts_enabled: bool,
) -> lance_core::Result<()> {
    if Dataset::open(base_uri).await.is_ok() {
        println!("base dataset already exists at {}, skipping build", base_uri);
        return Ok(());
    }
    println!(
        "building base dataset at {} ({} rows, batch_size {})",
        base_uri,
        base_texts.len(),
        batch_size
    );
    let total = base_texts.len();
    let mut batches = Vec::with_capacity(total.div_ceil(batch_size));
    let mut start = 0usize;
    while start < total {
        let end = (start + batch_size).min(total);
        batches.push(Ok(make_batch(
            start as i64,
            &base_texts[start..end],
            schema.clone(),
        )));
        start = end;
    }
    let reader = RecordBatchIterator::new(batches.into_iter(), schema.clone());
    let mut dataset = Dataset::write(reader, base_uri, Some(WriteParams::default())).await?;
    if fts_enabled {
        let fts_params = InvertedIndexParams::default();
        dataset
            .create_index(
                &[TEXT_COL],
                IndexType::Inverted,
                Some(FTS_INDEX_NAME.to_string()),
                &fts_params,
                true,
            )
            .await?;
    } else {
        // Even when MemWAL is configured without FTS, we still need a
        // BTree index on `id` so MemWAL has at least one maintained
        // index to reference.
        let pk_params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &pk_params,
                true,
            )
            .await?;
    }
    let maintained = if fts_enabled {
        vec![FTS_INDEX_NAME.to_string()]
    } else {
        vec!["id_btree".to_string()]
    };
    dataset
        .initialize_mem_wal(MemWalConfig {
            shard_spec: None,
            maintained_indexes: maintained,
        })
        .await?;
    Ok(())
}

// ----------------------------------------------------------------------
// Ingest
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct IngestStats {
    rows: usize,
    wall_seconds: f64,
    rows_per_sec: f64,
    /// p95 per-`put` latency in milliseconds.
    put_p95_ms: f64,
    put_p50_ms: f64,
    put_max_ms: f64,
    num_puts: usize,
}

async fn ingest_via_shard_writer(
    target_uri: &str,
    schema: Arc<ArrowSchema>,
    base_texts: &[String],
    ingest_texts: &[String],
    cfg: &Config,
    disable_auto_flush: bool,
) -> lance_core::Result<IngestStats> {
    // Build a fresh ingest dataset by cloning the base.
    println!("preparing ingest dataset at {}", target_uri);
    build_base_if_absent(
        target_uri,
        schema.clone(),
        base_texts,
        cfg.batch_size,
        cfg.fts_enabled,
    )
    .await?;
    let dataset = Arc::new(Dataset::open(target_uri).await?);

    let shard_id = Uuid::new_v4();
    let max_memtable_rows = if disable_auto_flush {
        cfg.ingest_rows.saturating_mul(2).max(2_000_000)
    } else {
        cfg.max_memtable_rows
    };
    let max_memtable_size = if disable_auto_flush {
        16 * 1024 * 1024 * 1024 // 16 GiB
    } else {
        16 * 1024 * 1024 * 1024
    };
    let writer_config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: cfg.durable_write,
        sync_indexed_write: true,
        max_memtable_size,
        max_memtable_rows,
        max_memtable_batches: 4_000_000,
        max_wal_flush_interval: Some(Duration::from_millis(200)),
        max_unflushed_memtable_bytes: usize::MAX / 2,
        ..ShardWriterConfig::default()
    };
    let writer = dataset
        .as_ref()
        .mem_wal_writer(shard_id, writer_config)
        .await?;

    // Ingest IDs start above the base table's last id to keep PK unique.
    let id_offset: i64 = cfg.base_rows as i64;
    let n = ingest_texts.len();
    let bs = cfg.batch_size;
    let total_batches = n.div_ceil(bs);

    let mut put_latencies: Vec<u128> = Vec::with_capacity(total_batches);
    let start = Instant::now();
    for i in 0..total_batches {
        let lo = i * bs;
        let hi = (lo + bs).min(n);
        let batch = make_batch(
            id_offset + lo as i64,
            &ingest_texts[lo..hi],
            schema.clone(),
        );
        let put_t = Instant::now();
        writer.put(vec![batch]).await?;
        put_latencies.push(put_t.elapsed().as_micros());
        if (i + 1) % 100 == 0 {
            let so_far = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 * bs as f64 / so_far.max(1e-9);
            println!(
                "  ingest progress: {}/{} batches ({:.0} rows/s)",
                i + 1,
                total_batches,
                rate
            );
        }
    }
    // Wait for index update to catch up if sync_indexed_write didn't fully drain.
    let target_batch_pos = total_batches.saturating_sub(1);
    loop {
        let active = writer.active_memtable_ref().await?;
        if active.index_store.max_indexed_batch_position() >= target_batch_pos {
            break;
        }
        drop(active);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let elapsed = start.elapsed();
    drop(writer);

    put_latencies.sort_unstable();
    let p50 = put_latencies[put_latencies.len() / 2] as f64 / 1000.0;
    let p95 = put_latencies[put_latencies.len() * 95 / 100] as f64 / 1000.0;
    let max = *put_latencies.iter().max().unwrap_or(&0) as f64 / 1000.0;
    Ok(IngestStats {
        rows: n,
        wall_seconds: elapsed.as_secs_f64(),
        rows_per_sec: n as f64 / elapsed.as_secs_f64().max(1e-9),
        put_p50_ms: p50,
        put_p95_ms: p95,
        put_max_ms: max,
        num_puts: total_batches,
    })
}

// ----------------------------------------------------------------------
// Query set
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct QuerySet {
    tokens: Vec<String>,
    phrases: Vec<String>,
}

fn build_query_set(sample_texts: &[&str], cfg: &Config) -> QuerySet {
    use lance_tokenizer::TokenStream;
    // ASCII stop-word-ish list used by the default English analyzer; we only
    // need a coarse filter for query selection here.
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by", "is",
        "was", "are", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its",
        "but", "not", "no", "if", "then", "than", "so", "do", "does", "did", "have", "has", "had",
        "will", "would", "should", "could", "can", "may", "might", "must", "i", "you", "he", "she",
        "we", "they", "them", "his", "her", "their", "our", "us", "me", "my", "your", "him",
    ];
    let mut tokenizer = InvertedIndexParams::default()
        .build()
        .expect("default tokenizer builds");
    let mut freq: HashMap<String, u64> = HashMap::new();
    for t in sample_texts.iter().take(50_000) {
        let mut stream = tokenizer.token_stream_for_doc(t);
        while let Some(tok) = stream.next() {
            if tok.text.len() < 3 || tok.text.len() > 24 {
                continue;
            }
            if STOPWORDS.contains(&tok.text.as_str()) {
                continue;
            }
            *freq.entry(tok.text.clone()).or_default() += 1;
        }
    }
    let mut by_freq: Vec<(String, u64)> = freq.into_iter().collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let tokens: Vec<String> = by_freq
        .into_iter()
        .map(|(t, _)| t)
        .take(cfg.num_token_queries)
        .collect();

    // Phrase queries: walk a deterministic stride of rows, take the first
    // two consecutive non-stopword non-short tokens.
    let mut phrases = Vec::with_capacity(cfg.num_phrase_queries);
    let stride = sample_texts.len().max(1) / cfg.num_phrase_queries.max(1);
    let mut idx = 0usize;
    while phrases.len() < cfg.num_phrase_queries && idx < sample_texts.len() {
        let t = sample_texts[idx];
        let mut stream = tokenizer.token_stream_for_doc(t);
        let mut acc: Vec<String> = Vec::new();
        while let Some(tok) = stream.next() {
            if tok.text.len() < 3 || tok.text.len() > 24 {
                continue;
            }
            if STOPWORDS.contains(&tok.text.as_str()) {
                continue;
            }
            acc.push(tok.text.clone());
            if acc.len() == 2 {
                phrases.push(format!("{} {}", acc[0], acc[1]));
                break;
            }
        }
        idx = idx.saturating_add(stride.max(1));
    }

    QuerySet { tokens, phrases }
}

// ----------------------------------------------------------------------
// Read test (FTS only): MemTable query latency + post-flush consistency
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct ReadStats {
    rows: usize,
    /// Average across all queries (token + phrase).
    mt_latency_avg_ms: f64,
    mt_latency_p50_ms: f64,
    mt_latency_p95_ms: f64,
    consistency_mean: f64,
    consistency_min: f64,
    num_queries: usize,
}

async fn run_read_test(
    target_uri: &str,
    schema: Arc<ArrowSchema>,
    base_texts: &[String],
    ingest_texts: &[String],
    queries: &QuerySet,
    cfg: &Config,
) -> lance_core::Result<ReadStats> {
    println!(
        "  read test: ingesting {} rows with auto-flush disabled",
        ingest_texts.len()
    );
    build_base_if_absent(
        target_uri,
        schema.clone(),
        base_texts,
        cfg.batch_size,
        true, // FTS index on the base, since this path is FTS-only.
    )
    .await?;
    let dataset = Arc::new(Dataset::open(target_uri).await?);
    let shard_id = Uuid::new_v4();
    let writer_config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: cfg.durable_write,
        sync_indexed_write: true,
        // Effectively disable auto-flush triggers so the MemTable holds
        // the full ingest_texts.len() rows for the query phase.
        max_memtable_size: 64 * 1024 * 1024 * 1024,
        max_memtable_rows: ingest_texts.len().saturating_mul(2),
        max_memtable_batches: 4_000_000,
        max_wal_flush_interval: Some(Duration::from_millis(200)),
        max_unflushed_memtable_bytes: usize::MAX / 2,
        ..ShardWriterConfig::default()
    };
    let writer = dataset
        .as_ref()
        .mem_wal_writer(shard_id, writer_config)
        .await?;

    let id_offset: i64 = cfg.base_rows as i64;
    let bs = cfg.batch_size;
    let n = ingest_texts.len();
    let total_batches = n.div_ceil(bs);
    for i in 0..total_batches {
        let lo = i * bs;
        let hi = (lo + bs).min(n);
        let batch = make_batch(
            id_offset + lo as i64,
            &ingest_texts[lo..hi],
            schema.clone(),
        );
        writer.put(vec![batch]).await?;
    }
    let target_batch_pos = total_batches.saturating_sub(1);
    loop {
        let active = writer.active_memtable_ref().await?;
        if active.index_store.max_indexed_batch_position() >= target_batch_pos {
            break;
        }
        drop(active);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // ----- MemTable phase -----
    let active = writer.active_memtable_ref().await?;
    let fts = active
        .index_store
        .get_fts(FTS_INDEX_NAME)
        .ok_or_else(|| lance_core::Error::invalid_input("FTS mem index not found"))?;

    let mut latencies_us: Vec<u128> = Vec::new();
    let mut mt_top10: Vec<HashSet<i64>> = Vec::new();

    let all_queries: Vec<(FtsQueryExpr, String)> = queries
        .tokens
        .iter()
        .map(|t| (FtsQueryExpr::match_query(t.clone()), t.clone()))
        .chain(
            queries
                .phrases
                .iter()
                .map(|p| (FtsQueryExpr::phrase(p.clone()), format!("\"{p}\""))),
        )
        .collect();

    // Build a row_position -> id map by scanning the active batches.
    // This is needed because the MemTable returns row_positions; the
    // post-flush on-disk FTS returns row_ids that match the `id` column.
    let mut row_to_id: HashMap<u64, i64> = HashMap::new();
    for stored in active.batch_store.iter() {
        let id_arr = stored
            .data
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| lance_core::Error::invalid_input("id col missing"))?;
        for r in 0..id_arr.len() {
            row_to_id.insert(stored.row_offset + r as u64, id_arr.value(r));
        }
    }

    for (q, label) in &all_queries {
        let opts = SearchOptions::new().with_limit(cfg.top_k);
        let t0 = Instant::now();
        let entries = fts.search_with_options(q, opts);
        latencies_us.push(t0.elapsed().as_micros());
        let mut ids = HashSet::with_capacity(cfg.top_k);
        for e in entries.iter().take(cfg.top_k) {
            if let Some(id) = row_to_id.get(&e.row_position) {
                ids.insert(*id);
            }
        }
        if mt_top10.len() < 3 {
            println!(
                "    [mt] {label}: {} hits, ids={:?}",
                entries.len(),
                ids.iter().take(3).collect::<Vec<_>>()
            );
        }
        mt_top10.push(ids);
    }
    drop(active);

    latencies_us.sort_unstable();
    let avg_us =
        latencies_us.iter().sum::<u128>() as f64 / latencies_us.len().max(1) as f64;
    let p50 = latencies_us[latencies_us.len() / 2] as f64 / 1000.0;
    let p95 = latencies_us[latencies_us.len() * 95 / 100] as f64 / 1000.0;

    // ----- Force flush, then on-disk phase -----
    println!("  read test: closing writer to force flush");
    writer.close().await?;
    let flushed_dataset = Dataset::open(target_uri).await?;

    let mut consistencies: Vec<f64> = Vec::with_capacity(all_queries.len());
    for ((q, label), mt_ids) in all_queries.iter().zip(mt_top10.iter()) {
        let fts_query = match q {
            FtsQueryExpr::Match { query, .. } => FullTextSearchQuery::new(query.clone()),
            FtsQueryExpr::Phrase { query, .. } => {
                FullTextSearchQuery::new(format!("\"{}\"", query))
            }
            _ => unreachable!("only match/phrase queries in this set"),
        };
        let mut scanner = flushed_dataset.scan();
        scanner.full_text_search(fts_query)?;
        scanner.limit(Some(cfg.top_k as i64), None)?;
        scanner.project(&["id"])?;
        let stream = scanner.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut disk_ids = HashSet::new();
        for b in &batches {
            let id_arr = b
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| lance_core::Error::invalid_input("disk id col missing"))?;
            for i in 0..id_arr.len() {
                disk_ids.insert(id_arr.value(i));
            }
        }
        let inter: usize = mt_ids.intersection(&disk_ids).count();
        let denom = mt_ids.len().max(disk_ids.len()).max(1);
        let cons = inter as f64 / denom as f64;
        if consistencies.len() < 3 {
            println!(
                "    [disk] {label}: {} hits; mt={} disk={} ∩={} cons={:.3}",
                disk_ids.len(),
                mt_ids.len(),
                disk_ids.len(),
                inter,
                cons
            );
        }
        consistencies.push(cons);
    }

    let cons_mean = consistencies.iter().sum::<f64>() / consistencies.len().max(1) as f64;
    let cons_min = consistencies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    Ok(ReadStats {
        rows: n,
        mt_latency_avg_ms: avg_us / 1000.0,
        mt_latency_p50_ms: p50,
        mt_latency_p95_ms: p95,
        consistency_mean: cons_mean,
        consistency_min: if cons_min.is_finite() { cons_min } else { 0.0 },
        num_queries: all_queries.len(),
    })
}

// ----------------------------------------------------------------------
// Top-level orchestration
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct RunResult {
    config_name: String,
    max_memtable_rows: usize,
    durable_write: bool,
    fts_enabled: bool,
    base_rows: usize,
    ingest_rows: usize,
    batch_size: usize,
    ingest: IngestStats,
    read: Option<ReadStats>,
    timestamp_utc: String,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> lance_core::Result<()> {
    let cfg = Config::from_env();
    println!("=== mem_wal_fineweb_fts === config = {:?}", cfg);

    let total_rows = cfg.base_rows + cfg.ingest_rows;
    let texts = load_corpus(total_rows, &cfg.cache_dir).await?;
    if texts.len() < total_rows {
        return Err(lance_core::Error::io(format!(
            "fineweb shards yielded only {} rows, need {}",
            texts.len(),
            total_rows
        )));
    }
    let base_texts = &texts[..cfg.base_rows];
    let ingest_texts = &texts[cfg.base_rows..cfg.base_rows + cfg.ingest_rows];

    let schema = make_schema();

    // Build query set once from the ingest slice (deterministic).
    let sample_refs: Vec<&str> = ingest_texts.iter().take(50_000).map(|s| s.as_str()).collect();
    let queries = build_query_set(&sample_refs, &cfg);
    println!(
        "query set: {} tokens + {} phrases",
        queries.tokens.len(),
        queries.phrases.len()
    );

    // Throughput sub-test: ingest 1M with the configured params.
    println!("\n--- throughput sub-test ---");
    let ingest_stats = ingest_via_shard_writer(
        &cfg.ingest_uri(),
        schema.clone(),
        base_texts,
        ingest_texts,
        &cfg,
        false, // auto-flush enabled (per max_memtable_rows)
    )
    .await?;
    println!("throughput: {:.1} rows/s", ingest_stats.rows_per_sec);

    // Read sub-test: only when FTS enabled and read test requested.
    let read_stats = if cfg.with_read_test {
        println!("\n--- read sub-test ---");
        let n_for_read = cfg.max_memtable_rows.min(ingest_texts.len());
        let read_ingest = &ingest_texts[..n_for_read];
        Some(
            run_read_test(
                &cfg.read_test_uri(),
                schema.clone(),
                base_texts,
                read_ingest,
                &queries,
                &cfg,
            )
            .await?,
        )
    } else {
        None
    };

    let timestamp_utc = chrono::Utc::now().to_rfc3339();
    let result = RunResult {
        config_name: cfg.config_name.clone(),
        max_memtable_rows: cfg.max_memtable_rows,
        durable_write: cfg.durable_write,
        fts_enabled: cfg.fts_enabled,
        base_rows: cfg.base_rows,
        ingest_rows: cfg.ingest_rows,
        batch_size: cfg.batch_size,
        ingest: ingest_stats,
        read: read_stats,
        timestamp_utc,
    };
    let json = serde_json::to_string_pretty(&result)
        .map_err(|e| lance_core::Error::io(format!("serialize result: {}", e)))?;
    if let Some(parent) = cfg.result_file.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(&cfg.result_file, json.as_bytes())
        .map_err(|e| lance_core::Error::io(format!("write result: {}", e)))?;
    println!("\nwrote result to {}", cfg.result_file.display());
    println!("=== DONE ===");
    let _ = result; // silence unused with no read test
    let _ = sample_refs;
    let _ = BTreeMap::<String, String>::new();
    Ok(())
}
