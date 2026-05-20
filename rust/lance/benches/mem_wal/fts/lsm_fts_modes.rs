// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark comparing `LsmScanner::full_text_search` scoring modes on
//! the LSM hierarchy with multiple flushed generations on a real
//! FineWeb corpus.
//!
//! Sibling of `mem_wal_fineweb_fts.rs`. Shares its FineWeb loader
//! shape (HF `sample/10BT` parquet shards, `--cache-dir` to amortize
//! downloads).
//!
//! Per shape × scoring mode the bench reports:
//!
//! * Wall-clock per query and aggregate latency percentiles.
//! * Top-K Jaccard vs a single-merged-index ground truth (the same
//!   FineWeb rows loaded into a single Lance dataset with one FTS
//!   index, queried via `scanner.full_text_search`).
//! * Pearson correlation of `_score` between LSM mode and ground
//!   truth on the intersection.
//!
//! Example:
//!
//! ```bash
//! cargo bench -p lance --bench lsm_fts_modes -- \
//!   --shape memwal_skewed --k 100 --num-queries 100 \
//!   --rescore-factor 10 \
//!   --cache-dir /tmp/fineweb-cache --output result.json
//! ```

#![recursion_limit = "256"]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance::dataset::mem_wal::scanner::{
    FtsScoringMode, InMemoryMemTableRef, InMemoryMemTables, LsmScanner,
};
use lance::dataset::mem_wal::write::{BatchStore, IndexStore};
use lance::dataset::{Dataset, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::Result;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_tokenizer::TokenStream;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use serde_json::json;
use uuid::Uuid;

const TEXT_COL: &str = "text";
const FTS_INDEX_NAME: &str = "text_fts";
const HF_API_LISTING: &str =
    "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT";
const HF_FILE_BASE: &str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/";

// ----------------------------------------------------------------------
// Shape
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// 4 equal-sized flushed gens + 1 equal-sized active. Cross-source
    /// stats are similar so Local should already be close to Rescore.
    Balanced,
    /// 1 huge base + 4 tiny flushed gens + 1 tiny active. The case
    /// where local-stats BM25 is most distorted vs a merged index.
    MemwalSkewed,
    /// Heterogeneous flushed sizes (1k+5k+25k+100k) + 25k active.
    GrowingLsm,
}

impl Shape {
    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "balanced" => Ok(Self::Balanced),
            "memwal_skewed" => Ok(Self::MemwalSkewed),
            "growing_lsm" => Ok(Self::GrowingLsm),
            other => Err(format!(
                "unknown shape '{other}', expected balanced|memwal_skewed|growing_lsm"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::MemwalSkewed => "memwal_skewed",
            Self::GrowingLsm => "growing_lsm",
        }
    }

    /// (base_rows, vec_of_flushed_gen_rows, active_rows). `base_rows = None`
    /// means no base table (fresh-tier-only). Designed so total
    /// `corpus_rows` is roughly comparable across shapes for fair Jaccard
    /// vs a single-merged baseline.
    fn slicing(self) -> (Option<usize>, Vec<usize>, usize) {
        match self {
            Self::Balanced => (Some(100_000), vec![25_000; 4], 25_000),
            Self::MemwalSkewed => (Some(1_000_000), vec![5_000; 4], 5_000),
            Self::GrowingLsm => (Some(100_000), vec![1_000, 5_000, 25_000, 100_000], 25_000),
        }
    }

    fn total_rows(self) -> usize {
        let (base, gens, active) = self.slicing();
        base.unwrap_or(0) + gens.iter().sum::<usize>() + active
    }
}

// ----------------------------------------------------------------------
// Args
// ----------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Args {
    shape: Shape,
    k: usize,
    num_queries: usize,
    rescore_factor: u32,
    cache_dir: PathBuf,
    work_dir: Option<PathBuf>,
    output: Option<PathBuf>,
    skip_baseline: bool,
    tokio_threads: usize,
    /// If `Some`, cap rows used per shape — useful for smoke testing
    /// without downloading hundreds of MB.
    max_corpus_rows: Option<usize>,
}

impl Default for Args {
    fn default() -> Self {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        Self {
            shape: Shape::Balanced,
            k: 100,
            num_queries: 100,
            rescore_factor: 10,
            cache_dir: std::env::temp_dir().join("mem_wal_fineweb_fts_cache"),
            work_dir: None,
            output: None,
            skip_baseline: false,
            tokio_threads: threads,
            max_corpus_rows: None,
        }
    }
}

fn parse<T>(flag: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|e| lance_core::Error::io(format!("flag {flag}: {e}")))
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut iter = raw.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--shape" => {
                args.shape = Shape::parse(
                    iter.next()
                        .ok_or_else(|| lance_core::Error::io("--shape needs value"))?,
                )
                .map_err(lance_core::Error::io)?
            }
            "--k" => {
                args.k = parse(
                    "--k",
                    iter.next()
                        .ok_or_else(|| lance_core::Error::io("--k needs value"))?,
                )?
            }
            "--num-queries" => {
                args.num_queries = parse(
                    "--num-queries",
                    iter.next()
                        .ok_or_else(|| lance_core::Error::io("--num-queries needs value"))?,
                )?
            }
            "--rescore-factor" => {
                args.rescore_factor = parse(
                    "--rescore-factor",
                    iter.next()
                        .ok_or_else(|| lance_core::Error::io("--rescore-factor needs value"))?,
                )?
            }
            "--cache-dir" => {
                args.cache_dir = PathBuf::from(
                    iter.next()
                        .ok_or_else(|| lance_core::Error::io("--cache-dir needs value"))?,
                )
            }
            "--work-dir" => {
                args.work_dir =
                    Some(PathBuf::from(iter.next().ok_or_else(|| {
                        lance_core::Error::io("--work-dir needs value")
                    })?))
            }
            "--output" => {
                args.output =
                    Some(PathBuf::from(iter.next().ok_or_else(|| {
                        lance_core::Error::io("--output needs value")
                    })?))
            }
            "--skip-baseline" => args.skip_baseline = true,
            "--tokio-threads" => {
                args.tokio_threads = parse(
                    "--tokio-threads",
                    iter.next()
                        .ok_or_else(|| lance_core::Error::io("--tokio-threads needs value"))?,
                )?
            }
            "--max-corpus-rows" => {
                args.max_corpus_rows = Some(parse(
                    "--max-corpus-rows",
                    iter.next()
                        .ok_or_else(|| lance_core::Error::io("--max-corpus-rows needs value"))?,
                )?)
            }
            // criterion-style noise we want to ignore so `cargo bench`
            // can hand us nothing extra without erroring.
            "--bench" | "--test" => {}
            other => {
                eprintln!("unknown flag: {other}");
                return Err(lance_core::Error::io(format!("unknown flag {other}")));
            }
        }
    }
    Ok(args)
}

// ----------------------------------------------------------------------
// FineWeb loading (mirrors mem_wal_fineweb_fts.rs)
// ----------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
}

async fn list_shard_paths() -> Result<Vec<String>> {
    let entries: Vec<HfTreeEntry> = reqwest::get(HF_API_LISTING)
        .await
        .map_err(|e| lance_core::Error::io(format!("listing HTTP: {e}")))?
        .json()
        .await
        .map_err(|e| lance_core::Error::io(format!("listing JSON: {e}")))?;
    let mut shards: Vec<String> = entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.ends_with(".parquet"))
        .map(|e| e.path)
        .collect();
    shards.sort();
    Ok(shards)
}

async fn download_shard(rel_path: &str, dest: &std::path::Path) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    let url = format!("{HF_FILE_BASE}{rel_path}");
    let tmp = dest.with_extension("part");
    for attempt in 1..=5u32 {
        println!("downloading {rel_path} (attempt {attempt}/5) ...");
        let result: Result<bytes::Bytes> = async {
            let resp = reqwest::get(&url)
                .await
                .map_err(|e| lance_core::Error::io(format!("download HTTP: {e}")))?;
            if !resp.status().is_success() {
                return Err(lance_core::Error::io(format!(
                    "download {url} -> status {}",
                    resp.status()
                )));
            }
            resp.bytes()
                .await
                .map_err(|e| lance_core::Error::io(format!("read body: {e}")))
        }
        .await;
        match result {
            Ok(bytes) => {
                std::fs::write(&tmp, &bytes)
                    .map_err(|e| lance_core::Error::io(format!("write: {e}")))?;
                std::fs::rename(&tmp, dest)
                    .map_err(|e| lance_core::Error::io(format!("rename: {e}")))?;
                println!(
                    "  wrote {:.1} MB to {}",
                    bytes.len() as f64 / 1024.0 / 1024.0,
                    dest.display()
                );
                return Ok(());
            }
            Err(e) if attempt < 5 => {
                eprintln!("  attempt {attempt} failed: {e}; retrying");
                tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
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
) -> Result<usize> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| lance_core::Error::io(format!("open parquet: {e}")))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file)
        .await
        .map_err(|e| lance_core::Error::io(format!("parquet builder: {e}")))?;
    let mut stream = builder
        .build()
        .map_err(|e| lance_core::Error::io(format!("parquet stream: {e}")))?;
    let mut taken = 0usize;
    while taken < max_rows {
        let Some(rb) = stream
            .try_next()
            .await
            .map_err(|e| lance_core::Error::io(format!("parquet read: {e}")))?
        else {
            break;
        };
        let col = rb
            .column_by_name(TEXT_COL)
            .ok_or_else(|| lance_core::Error::io("text column missing".to_string()))?;
        let strs = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| lance_core::Error::io("text not StringArray".to_string()))?;
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

async fn load_corpus(needed_rows: usize, cache_dir: &std::path::Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| lance_core::Error::io(format!("mkdir cache: {e}")))?;
    let shards = list_shard_paths().await?;
    println!("fineweb sample/10BT: {} shards", shards.len());
    let mut buf: Vec<String> = Vec::with_capacity(needed_rows);
    for rel in &shards {
        if buf.len() >= needed_rows {
            break;
        }
        let name = rel.rsplit('/').next().unwrap_or(rel);
        let local = cache_dir.join(name);
        download_shard(rel, &local).await?;
        let want = needed_rows - buf.len();
        let got = read_shard_text(&local, &mut buf, want).await?;
        println!("  shard {name} -> {got} rows (cumulative {})", buf.len());
    }
    if buf.len() < needed_rows {
        return Err(lance_core::Error::io(format!(
            "fineweb yielded only {} rows, need {needed_rows}",
            buf.len()
        )));
    }
    Ok(buf)
}

// ----------------------------------------------------------------------
// Dataset shaping
// ----------------------------------------------------------------------

fn make_schema() -> Arc<ArrowSchema> {
    let mut id_meta = HashMap::new();
    id_meta.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    let id_field = Field::new("id", DataType::Int64, false).with_metadata(id_meta);
    Arc::new(ArrowSchema::new(vec![
        id_field,
        Field::new(TEXT_COL, DataType::Utf8, true),
    ]))
}

fn slice_to_batch(schema: Arc<ArrowSchema>, start_id: i64, texts: &[String]) -> RecordBatch {
    let ids = Int64Array::from_iter_values(start_id..start_id + texts.len() as i64);
    let text = StringArray::from_iter_values(texts.iter().map(String::as_str));
    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(text)]).unwrap()
}

async fn write_lance(uri: &str, batches: Vec<RecordBatch>) -> Result<Dataset> {
    let schema = batches[0].schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    Dataset::write(reader, uri, Some(WriteParams::default())).await
}

async fn create_fts_index(ds: &mut Dataset) -> Result<()> {
    ds.create_index(
        &[TEXT_COL],
        IndexType::Inverted,
        Some(FTS_INDEX_NAME.to_string()),
        &InvertedIndexParams::default(),
        false,
    )
    .await?;
    Ok(())
}

/// Build the LSM shape: writes one Lance dataset per flushed gen (with
/// FTS index), and constructs an active in-memory memtable from the
/// active slice. Returns the collector inputs ready for `LsmScanner`.
async fn build_lsm_shape(
    shape: Shape,
    corpus: &[String],
    work_dir: &std::path::Path,
) -> Result<(
    Option<Arc<Dataset>>,
    Vec<lance::dataset::mem_wal::scanner::ShardSnapshot>,
    Uuid,
    InMemoryMemTables,
)> {
    let schema = make_schema();
    let (base_rows, gen_rows, active_rows) = shape.slicing();
    let total = base_rows.unwrap_or(0) + gen_rows.iter().sum::<usize>() + active_rows;
    assert!(
        corpus.len() >= total,
        "shape needs {total} rows, corpus has {}",
        corpus.len()
    );

    let mut cursor: usize = 0;
    let mut id_cursor: i64 = 0;
    let shard_id = Uuid::new_v4();

    // Base.
    let base = if let Some(n) = base_rows {
        let uri = format!("{}/base", work_dir.display());
        let mut ds = write_lance(
            &uri,
            vec![slice_to_batch(
                schema.clone(),
                id_cursor,
                &corpus[cursor..cursor + n],
            )],
        )
        .await?;
        create_fts_index(&mut ds).await?;
        let ds = Arc::new(Dataset::open(&uri).await?);
        cursor += n;
        id_cursor += n as i64;
        Some(ds)
    } else {
        None
    };

    // Flushed generations.
    let mut shard_snapshot = lance::dataset::mem_wal::scanner::ShardSnapshot::new(shard_id)
        .with_current_generation((gen_rows.len() as u64).max(1));
    let base_uri = base
        .as_ref()
        .map(|d| d.uri().to_string())
        .unwrap_or_else(|| format!("{}/base", work_dir.display()));
    for (i, &n) in gen_rows.iter().enumerate() {
        let gen_num = (i + 1) as u64;
        let rel = format!("gen_{gen_num}");
        let uri = format!("{base_uri}/_mem_wal/{shard_id}/{rel}");
        let mut ds = write_lance(
            &uri,
            vec![slice_to_batch(
                schema.clone(),
                id_cursor,
                &corpus[cursor..cursor + n],
            )],
        )
        .await?;
        create_fts_index(&mut ds).await?;
        cursor += n;
        id_cursor += n as i64;
        shard_snapshot = shard_snapshot.with_flushed_generation(gen_num, rel);
    }

    // Active memtable.
    let batch_store = Arc::new(BatchStore::with_capacity(active_rows.max(16)));
    let mut indexes = IndexStore::new();
    indexes.add_fts(FTS_INDEX_NAME.to_string(), 1, TEXT_COL.to_string());
    let active_batch = slice_to_batch(
        schema.clone(),
        id_cursor,
        &corpus[cursor..cursor + active_rows],
    );
    batch_store.append(active_batch.clone()).unwrap();
    indexes
        .insert_with_batch_position(&active_batch, 0, Some(0))
        .unwrap();
    let indexes = Arc::new(indexes);

    let in_memory = InMemoryMemTables {
        active: InMemoryMemTableRef {
            batch_store,
            index_store: indexes,
            schema,
            generation: (gen_rows.len() as u64) + 1,
        },
        frozen: vec![],
    };

    Ok((base, vec![shard_snapshot], shard_id, in_memory))
}

/// Build a single Lance dataset containing the full corpus + one FTS
/// index. Used as the ground-truth reference for Jaccard / score Pearson.
async fn build_baseline(corpus: &[String], work_dir: &std::path::Path) -> Result<Arc<Dataset>> {
    let schema = make_schema();
    let uri = format!("{}/baseline_merged", work_dir.display());
    let batch = slice_to_batch(schema, 0, corpus);
    let mut ds = write_lance(&uri, vec![batch]).await?;
    create_fts_index(&mut ds).await?;
    Ok(Arc::new(Dataset::open(&uri).await?))
}

// ----------------------------------------------------------------------
// Query selection
// ----------------------------------------------------------------------

/// Pick `n` representative single-term queries from the corpus.
///
/// Tokenizes a sample of the corpus with the default English tokenizer,
/// counts term frequencies, and returns terms in the "long tail" — not
/// the absolute most frequent (those match nearly every doc and don't
/// produce interesting BM25 rankings) and not the rarest (those match
/// nothing and produce empty top-K). Window roughly between the 80th
/// and 99th percentile of df.
fn pick_queries(corpus: &[String], n: usize) -> Vec<String> {
    let sample_n = corpus.len().min(2_000);
    let mut tokenizer = InvertedIndexParams::default().build().expect("tokenizer");
    let mut df: HashMap<String, usize> = HashMap::new();
    for text in corpus.iter().take(sample_n) {
        let mut stream = tokenizer.token_stream_for_doc(text);
        let mut seen: HashSet<String> = HashSet::new();
        while let Some(tok) = stream.next() {
            if seen.insert(tok.text.clone()) {
                *df.entry(tok.text.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut all: Vec<(String, usize)> = df.into_iter().collect();
    all.sort_by_key(|(_, c)| *c);
    // Pull from the 80th–99th percentile window.
    let lo = (all.len() as f64 * 0.80) as usize;
    let hi = (all.len() as f64 * 0.99) as usize;
    let window: &[(String, usize)] = &all[lo.min(all.len())..hi.min(all.len())];
    if window.is_empty() {
        return Vec::new();
    }
    let stride = (window.len() / n.max(1)).max(1);
    let mut out: Vec<String> = Vec::with_capacity(n);
    for (i, (term, _)) in window.iter().enumerate() {
        if out.len() >= n {
            break;
        }
        if i % stride == 0 {
            out.push(term.clone());
        }
    }
    out
}

// ----------------------------------------------------------------------
// Mode runner
// ----------------------------------------------------------------------

#[derive(Debug)]
struct ModeRun {
    /// Per-query top-k row id sets.
    top_ids: Vec<HashSet<i64>>,
    /// Per-query (id → score) maps, for Pearson on the intersection.
    scored: Vec<HashMap<i64, f32>>,
    latencies_us: Vec<u64>,
}

async fn run_mode(
    scanner: &LsmScanner,
    mode: FtsScoringMode,
    queries: &[String],
    k: usize,
) -> Result<ModeRun> {
    let mut top_ids = Vec::with_capacity(queries.len());
    let mut scored = Vec::with_capacity(queries.len());
    let mut latencies_us = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        let plan = scanner
            .full_text_search(TEXT_COL, FullTextSearchQuery::new(q.clone()), k, mode)
            .await?;
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan
            .execute(0, ctx.task_ctx())
            .map_err(|e| lance_core::Error::io(format!("plan execute for query '{q}': {e}")))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| lance_core::Error::io(format!("collect for query '{q}': {e}")))?;
        latencies_us.push(t.elapsed().as_micros() as u64);

        let mut ids: HashSet<i64> = HashSet::new();
        let mut score_map: HashMap<i64, f32> = HashMap::new();
        for b in &batches {
            let id_col = b
                .column_by_name("id")
                .expect("id col")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id Int64");
            let score_col = b
                .column_by_name("_score")
                .expect("_score col")
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .expect("_score Float32");
            for i in 0..b.num_rows() {
                let id = id_col.value(i);
                ids.insert(id);
                score_map.insert(id, score_col.value(i));
            }
        }
        top_ids.push(ids);
        scored.push(score_map);
    }
    Ok(ModeRun {
        top_ids,
        scored,
        latencies_us,
    })
}

/// Run the merged-index baseline (single Lance dataset). We reuse the
/// same `scanner.full_text_search` API on the merged dataset so the
/// score scale is comparable to the LSM result (`scanner` already uses
/// `build_global_bm25_scorer` for its internal multi-partition case).
async fn run_baseline(baseline: &Dataset, queries: &[String], k: usize) -> Result<ModeRun> {
    let mut top_ids = Vec::with_capacity(queries.len());
    let mut scored = Vec::with_capacity(queries.len());
    let mut latencies_us = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        let mut scanner = baseline.scan();
        scanner.project(&["id", TEXT_COL])?;
        scanner.full_text_search(
            FullTextSearchQuery::new(q.clone())
                .with_column(TEXT_COL.to_string())?
                .limit(Some(k as i64)),
        )?;
        let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
        latencies_us.push(t.elapsed().as_micros() as u64);

        let mut ids: HashSet<i64> = HashSet::new();
        let mut score_map: HashMap<i64, f32> = HashMap::new();
        for b in &batches {
            let id_col = b
                .column_by_name("id")
                .expect("id col")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id Int64");
            let score_col = b
                .column_by_name("_score")
                .expect("_score col")
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .expect("_score Float32");
            for i in 0..b.num_rows() {
                let id = id_col.value(i);
                ids.insert(id);
                score_map.insert(id, score_col.value(i));
            }
        }
        top_ids.push(ids);
        scored.push(score_map);
    }
    Ok(ModeRun {
        top_ids,
        scored,
        latencies_us,
    })
}

// ----------------------------------------------------------------------
// Metrics
// ----------------------------------------------------------------------

fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn mean_jaccard(a: &[HashSet<i64>], b: &[HashSet<i64>]) -> f64 {
    let pairs: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(x, y)| {
            if x.is_empty() && y.is_empty() {
                None
            } else {
                let inter = x.intersection(y).count() as f64;
                let union = x.union(y).count() as f64;
                Some(inter / union)
            }
        })
        .collect();
    if pairs.is_empty() {
        0.0
    } else {
        pairs.iter().sum::<f64>() / pairs.len() as f64
    }
}

/// Pearson correlation of scores on the intersection. Averaged over
/// queries that have at least 2 overlapping ids (Pearson is undefined
/// for fewer).
fn mean_pearson(a: &[HashMap<i64, f32>], b: &[HashMap<i64, f32>]) -> f64 {
    let pairs: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(x, y)| {
            let common: Vec<i64> = x.keys().filter(|k| y.contains_key(k)).copied().collect();
            if common.len() < 2 {
                return None;
            }
            let xs: Vec<f64> = common.iter().map(|i| x[i] as f64).collect();
            let ys: Vec<f64> = common.iter().map(|i| y[i] as f64).collect();
            let mx = xs.iter().sum::<f64>() / xs.len() as f64;
            let my = ys.iter().sum::<f64>() / ys.len() as f64;
            let num: f64 = xs
                .iter()
                .zip(ys.iter())
                .map(|(a, b)| (a - mx) * (b - my))
                .sum();
            let dx: f64 = xs.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
            let dy: f64 = ys.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
            if dx == 0.0 || dy == 0.0 {
                None
            } else {
                Some(num / (dx * dy))
            }
        })
        .collect();
    if pairs.is_empty() {
        0.0
    } else {
        pairs.iter().sum::<f64>() / pairs.len() as f64
    }
}

// ----------------------------------------------------------------------
// Run
// ----------------------------------------------------------------------

async fn run(args: Args) -> Result<()> {
    let needed = args
        .max_corpus_rows
        .unwrap_or_else(|| args.shape.total_rows());
    println!(
        "shape={}  needed_rows={}  k={}  num_queries={}  rescore_factor={}",
        args.shape.as_str(),
        needed,
        args.k,
        args.num_queries,
        args.rescore_factor
    );

    let corpus = load_corpus(needed, &args.cache_dir).await?;
    let queries = pick_queries(&corpus, args.num_queries);
    println!("picked {} query terms", queries.len());

    let work_dir = if let Some(d) = &args.work_dir {
        std::fs::create_dir_all(d).map_err(|e| lance_core::Error::io(format!("mkdir: {e}")))?;
        d.clone()
    } else {
        tempfile::tempdir()
            .map_err(|e| lance_core::Error::io(format!("tempdir: {e}")))?
            .keep()
    };

    let (base, shard_snapshots, shard_id, in_memory) =
        build_lsm_shape(args.shape, &corpus, &work_dir).await?;

    let pk_columns = vec!["id".to_string()];
    let scanner = if let Some(b) = base.clone() {
        LsmScanner::new(b, shard_snapshots.clone(), pk_columns.clone())
    } else {
        let schema = make_schema();
        LsmScanner::without_base_table(
            schema,
            format!("{}/base", work_dir.display()),
            shard_snapshots.clone(),
            pk_columns.clone(),
        )
    }
    .with_in_memory_memtables(shard_id, in_memory);

    println!("running Local mode ...");
    let local = run_mode(&scanner, FtsScoringMode::Local, &queries, args.k).await?;
    println!("running LocalWithGlobalRescore mode ...");
    let rescore = run_mode(
        &scanner,
        FtsScoringMode::LocalWithGlobalRescore {
            rescore_factor: args.rescore_factor,
        },
        &queries,
        args.k,
    )
    .await?;

    let baseline_run = if args.skip_baseline {
        None
    } else {
        println!("building merged-index baseline ...");
        let baseline = build_baseline(&corpus, &work_dir).await?;
        Some(run_baseline(&baseline, &queries, args.k).await?)
    };

    // Aggregate metrics.
    let lat = |run: &ModeRun| -> (f64, f64, f64, f64) {
        let mut v: Vec<f64> = run
            .latencies_us
            .iter()
            .map(|x| *x as f64 / 1000.0)
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (
            mean,
            percentile(&v, 50.0),
            percentile(&v, 95.0),
            percentile(&v, 99.0),
        )
    };
    let (local_mean, local_p50, local_p95, local_p99) = lat(&local);
    let (rescore_mean, rescore_p50, rescore_p95, rescore_p99) = lat(&rescore);

    let jaccard_local_rescore = mean_jaccard(&local.top_ids, &rescore.top_ids);
    let pearson_local_rescore = mean_pearson(&local.scored, &rescore.scored);
    let (jaccard_local_baseline, pearson_local_baseline) = if let Some(b) = &baseline_run {
        (
            mean_jaccard(&local.top_ids, &b.top_ids),
            mean_pearson(&local.scored, &b.scored),
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    let (jaccard_rescore_baseline, pearson_rescore_baseline) = if let Some(b) = &baseline_run {
        (
            mean_jaccard(&rescore.top_ids, &b.top_ids),
            mean_pearson(&rescore.scored, &b.scored),
        )
    } else {
        (f64::NAN, f64::NAN)
    };

    let summary = json!({
        "shape": args.shape.as_str(),
        "k": args.k,
        "num_queries": queries.len(),
        "rescore_factor": args.rescore_factor,
        "local": {
            "mean_ms": local_mean,
            "p50_ms": local_p50,
            "p95_ms": local_p95,
            "p99_ms": local_p99,
        },
        "rescore": {
            "mean_ms": rescore_mean,
            "p50_ms": rescore_p50,
            "p95_ms": rescore_p95,
            "p99_ms": rescore_p99,
        },
        "jaccard": {
            "local_vs_rescore": jaccard_local_rescore,
            "local_vs_baseline": jaccard_local_baseline,
            "rescore_vs_baseline": jaccard_rescore_baseline,
        },
        "pearson_score": {
            "local_vs_rescore": pearson_local_rescore,
            "local_vs_baseline": pearson_local_baseline,
            "rescore_vs_baseline": pearson_rescore_baseline,
        },
    });

    println!(
        "\n=== Result ===\n{}",
        serde_json::to_string_pretty(&summary).unwrap()
    );
    if let Some(path) = &args.output {
        std::fs::write(path, serde_json::to_string_pretty(&summary).unwrap())
            .map_err(|e| lance_core::Error::io(format!("write output: {e}")))?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(args.tokio_threads)
        .build()
        .map_err(|e| lance_core::Error::io(format!("tokio: {e}")))?;
    rt.block_on(run(args))
}
