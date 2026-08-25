// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE benchmark (discussion lance-format/lance#7499): add-column / backfill
//! write amplification, flat manifest vs the recursive self-balancing Bε-tree.
//!
//! Realistic embedding backfill commits touch only a handful of fragments (F) at
//! a time, so a full backfill over N fragments is N/F commits — up to hundreds of
//! thousands. Flat rewrites the whole (growing) manifest every commit regardless
//! of F, so its full-backfill write is (N/F) × manifest — petabytes at F=10,
//! N=1M. We measure per-commit cost and extrapolate:
//!   - flat: run a small uniform sample and extrapolate ×(N/F).
//!   - Bε-tree: run `BETREE_COMMITS` commits (steady state incl. splits), measure
//!     per-commit + cumulative, extrapolate ×(N/F).
//!
//! Data-file names use Lance's real 50-char format (see `support.rs`).
//!
//! ## Configuration (env)
//! - `BASE_URI`             s3://… / s3://…--x-s3 / local path. Default: temp dir.
//! - `NUM_FRAGMENTS`        bootstrap fragment count (N). Default 5000.
//! - `FRAGMENTS_PER_COMMIT` comma sweep of F. Default "10,100".
//! - `NODE_SIZE_MB`         comma sweep of the node-size limit (max_node_bytes) in MiB (fractional ok). Default "4,10".
//! - `FANOUT`               branching factor max_children_per_node (split above, merge below a quarter of it). Default 16.
//! - `BETREE_COMMITS`       Bε-tree commits to run per config (M). Default 3000.
//! - `FLAT_SAMPLE_COMMITS`  flat commits to sample. Default 20.
//! - `S3_EXPRESS`           "true" for S3 Express directory buckets.
//! - `AWS_REGION`           required for S3.

#![allow(clippy::print_stdout)]

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;
use uuid::Uuid;

use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema;
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::tracking_store::IOTracker;
use lance_table::betree::flat_baseline::FlatBaseline;
use lance_table::betree::node::fragment_logical_bytes;
use lance_table::betree::support::{
    make_backfill_data_file, make_fragment, make_fragment_with_files,
};
use lance_table::betree::{BeTree, BeTreeConfig, action};
use lance_table::format::Fragment;
use object_store::path::Path;

const MIB: u64 = 1024 * 1024;

fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_sweep_u64(key: &str, default: &str) -> Vec<u64> {
    env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn env_sweep_f64(key: &str, default: &str) -> Vec<f64> {
    env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn bench_schema() -> Schema {
    let arrow = ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int64, false),
        ArrowField::new("name", DataType::Utf8, false),
    ]);
    Schema::try_from(&arrow).unwrap()
}

fn storage_params(uri: &str, io: &IOTracker) -> ObjectStoreParams {
    let mut params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(io.clone())),
        ..Default::default()
    };
    if uri.starts_with("s3://") {
        let mut opts: HashMap<String, String> = HashMap::new();
        if let Ok(region) = env::var("AWS_REGION").or_else(|_| env::var("AWS_DEFAULT_REGION")) {
            opts.insert("region".to_string(), region);
        }
        for (env_key, opt_key) in [
            ("AWS_ACCESS_KEY_ID", "access_key_id"),
            ("AWS_SECRET_ACCESS_KEY", "secret_access_key"),
            ("AWS_SESSION_TOKEN", "session_token"),
        ] {
            if let Ok(v) = env::var(env_key) {
                opts.insert(opt_key.to_string(), v);
            }
        }
        let express =
            env::var("S3_EXPRESS").map(|v| v == "true").unwrap_or(false) || uri.contains("--x-s3");
        if express {
            opts.insert("s3_express".to_string(), "true".to_string());
        }
        params.storage_options_accessor =
            Some(Arc::new(StorageOptionsAccessor::with_static_options(opts)));
    }
    params
}

async fn build_store(
    uri: &str,
    io: &IOTracker,
) -> (Arc<ObjectStore>, Path, Arc<ScanScheduler>, Arc<LanceCache>) {
    let registry = Arc::new(ObjectStoreRegistry::default());
    let params = storage_params(uri, io);
    let (object_store, base) = ObjectStore::from_uri_and_params(registry, uri, &params)
        .await
        .expect("failed to build object store");
    let scheduler =
        ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
    let cache = Arc::new(LanceCache::with_capacity(0));
    (object_store, base, scheduler, cache)
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * (sorted_ms.len() as f64 - 1.0)).round() as usize;
    sorted_ms[rank.min(sorted_ms.len() - 1)]
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn commit_window(c: u64, f: u64, n: u64) -> std::ops::Range<u64> {
    let start = (c * f).min(n);
    let end = (start + f).min(n);
    start..end
}

#[derive(Default)]
struct RunResult {
    layout: &'static str,
    node_size_mib: f64,
    commits_run: u64,
    backfill_write_bytes: u64,
    bootstrap_write_bytes: u64,
    flushes: u64,
    splits: u64,
    merges: u64,
    height: u32,
    commit_mean_ms: f64,
    commit_p50_ms: f64,
    commit_p99_ms: f64,
    commit_max_ms: f64,
    materialized_fragments: usize,
    total_data_files: usize,
}

impl RunResult {
    fn per_commit_write_bytes(&self) -> f64 {
        self.backfill_write_bytes as f64 / self.commits_run.max(1) as f64
    }
    fn full_backfill_write_bytes(&self, n: u64, f: u64) -> f64 {
        let total_commits = n.div_ceil(f) as f64;
        self.bootstrap_write_bytes as f64 + self.per_commit_write_bytes() * total_commits
    }
}

async fn run_flat(uri: &str, n: u64, f: u64, sample_commits: u64) -> RunResult {
    let io = IOTracker::default();
    let (object_store, base, _sched, _cache) = build_store(uri, &io).await;
    let fragments: Vec<Fragment> = (0..n).map(make_fragment).collect();

    let mut flat = FlatBaseline::new(
        object_store.clone(),
        base.clone(),
        bench_schema(),
        fragments,
    );
    let bootstrap_write_bytes = flat.write().await.expect("flat bootstrap");

    let commits = sample_commits.min(n.div_ceil(f));
    let mut commit_ms: Vec<f64> = Vec::with_capacity(commits as usize);
    let mut backfill_write_bytes = 0u64;
    for c in 0..commits {
        let adds: Vec<_> = commit_window(c, f, n)
            .map(|id| (id, make_backfill_data_file(id, 0)))
            .collect();
        let start = Instant::now();
        backfill_write_bytes += flat
            .commit_add_data_files(&adds)
            .await
            .expect("flat commit");
        commit_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let final_version = flat.version();
    let manifest = FlatBaseline::cold_open(&object_store, &base, final_version)
        .await
        .expect("flat cold open");
    let total_data_files: usize = manifest.fragments.iter().map(|fr| fr.files.len()).sum();

    let commit_mean_ms = mean(&commit_ms);
    commit_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    RunResult {
        layout: "flat",
        commits_run: commits,
        backfill_write_bytes,
        bootstrap_write_bytes,
        commit_mean_ms,
        commit_p50_ms: percentile(&commit_ms, 50.0),
        commit_p99_ms: percentile(&commit_ms, 99.0),
        commit_max_ms: commit_ms.last().copied().unwrap_or(0.0),
        materialized_fragments: manifest.fragments.len(),
        total_data_files,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_betree(
    uri: &str,
    n: u64,
    f: u64,
    m_commits: u64,
    node_bytes: u64,
    max_children_per_node: u32,
) -> RunResult {
    let io = IOTracker::default();
    let (object_store, base, scheduler, cache) = build_store(uri, &io).await;
    let fragments: Vec<Fragment> = (0..n).map(make_fragment).collect();

    let config = BeTreeConfig::new(node_bytes, max_children_per_node);
    let (mut tree, boot) = BeTree::bootstrap(
        object_store.clone(),
        base.clone(),
        scheduler.clone(),
        cache.clone(),
        config,
        fragments,
        Vec::new(),
    )
    .await
    .expect("betree bootstrap");

    let commits = m_commits.min(n.div_ceil(f));
    let mut commit_ms: Vec<f64> = Vec::with_capacity(commits as usize);
    let mut backfill_write_bytes = 0u64;
    let (mut flushes, mut splits, mut merges, mut height) = (0u64, 0u64, 0u64, 0u32);
    for c in 0..commits {
        let actions: Vec<_> = commit_window(c, f, n)
            .map(|id| action::add_data_file(id, &make_backfill_data_file(id, 0)))
            .collect();
        let start = Instant::now();
        let stats = tree.commit(actions).await.expect("betree commit");
        commit_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        backfill_write_bytes += stats.io_write_bytes;
        flushes += stats.flushes;
        splits += stats.splits;
        merges += stats.merges;
        height = stats.height;
    }
    println!(
        "  [betree B={:.2} MiB, max_children_per_node {max_children_per_node}] {commits} commits: {flushes} flushes, {splits} splits, {merges} merges, height {height}",
        node_bytes as f64 / MIB as f64
    );

    let frags = BeTree::cold_open(object_store.clone(), base.clone(), scheduler, cache)
        .await
        .expect("betree cold open");
    let total_data_files: usize = frags.iter().map(|fr| fr.files.len()).sum();

    let commit_mean_ms = mean(&commit_ms);
    commit_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    RunResult {
        layout: "betree",
        node_size_mib: node_bytes as f64 / MIB as f64,
        commits_run: commits,
        backfill_write_bytes,
        bootstrap_write_bytes: boot.io_write_bytes,
        flushes,
        splits,
        merges,
        height,
        commit_mean_ms,
        commit_p50_ms: percentile(&commit_ms, 50.0),
        commit_p99_ms: percentile(&commit_ms, 99.0),
        commit_max_ms: commit_ms.last().copied().unwrap_or(0.0),
        materialized_fragments: frags.len(),
        total_data_files,
    }
}

fn print_result(r: &RunResult, n: u64, f: u64) {
    let touched = (r.commits_run * f).min(n) as usize;
    assert_eq!(
        r.materialized_fragments, n as usize,
        "{} lost fragments",
        r.layout
    );
    assert_eq!(
        r.total_data_files,
        n as usize + touched,
        "{} wrong data-file count: {} (expected {})",
        r.layout,
        r.total_data_files,
        n as usize + touched
    );

    let label = if r.layout == "flat" {
        "flat".to_string()
    } else {
        format!("betree/B={:.1}MiB", r.node_size_mib)
    };
    println!(
        "  {:<16} per-commit write={:>9.3} MiB | commit mean={:>8.2} p50={:>8.2} p99={:>8.2} max={:>8.2} ms \
         | full-backfill~={:>9.2} GiB | h={} flushes={} splits={} merges={}",
        label,
        r.per_commit_write_bytes() / MIB as f64,
        r.commit_mean_ms,
        r.commit_p50_ms,
        r.commit_p99_ms,
        r.commit_max_ms,
        r.full_backfill_write_bytes(n, f) / (1024.0 * MIB as f64),
        r.height,
        r.flushes,
        r.splits,
        r.merges,
    );
}

/// Billion-scale mode: bootstrap N fragments each already holding `base_files`
/// data files (streaming, so memory stays bounded), then measure the recursive
/// tree's structure + cost at that scale. Flat is reported analytically because
/// its single-object manifest (N × base_files entries) is infeasibly large.
#[allow(clippy::too_many_arguments)]
async fn run_scale(
    uri: &str,
    n: u64,
    base_files: u32,
    node_bytes: u64,
    max_children_per_node: u32,
    backfill_columns: u32,
    f: u64,
) {
    // Full materialize builds every fragment in RAM; skip past this many files.
    const MATERIALIZE_CAP: u64 = 150_000_000;
    let io = IOTracker::default();
    let (object_store, base, scheduler, cache) = build_store(uri, &io).await;
    let config = BeTreeConfig::new(node_bytes, max_children_per_node);
    let total_files = n * base_files as u64;

    let t0 = Instant::now();
    let (mut tree, boot) = BeTree::bootstrap_generate(
        object_store.clone(),
        base.clone(),
        scheduler.clone(),
        cache.clone(),
        config,
        n,
        |id| make_fragment_with_files(id, base_files),
        Vec::new(),
    )
    .await
    .expect("scale bootstrap");
    let bootstrap_secs = t0.elapsed().as_secs_f64();

    // Analytical flat manifest size = one DataFragment proto per fragment.
    let flat_manifest_bytes = fragment_logical_bytes(&make_fragment_with_files(0, base_files)) * n;

    println!(
        "  [B={:.1} MiB] bootstrap {total_files} data files ({n} frags x {base_files} files): \
         betree {:.2} GiB in {} leaves, height {}, {:.1}s",
        node_bytes as f64 / MIB as f64,
        boot.io_write_bytes as f64 / (1024.0 * MIB as f64),
        boot.num_leaves,
        boot.height,
        bootstrap_secs,
    );
    println!(
        "             flat manifest would be ~{:.2} GiB (single object) => betree columnar leaves are {:.0}x smaller",
        flat_manifest_bytes as f64 / (1024.0 * MIB as f64),
        flat_manifest_bytes as f64 / boot.io_write_bytes.max(1) as f64,
    );

    let mut final_added = 0u64;
    if backfill_columns > 0 {
        // A bounded per-commit sample (a full column would be N/F commits — up to
        // 100K at N=1M) is enough to characterize per-commit write at scale.
        const SAMPLE: u64 = 200;
        let sample = SAMPLE.min(n.div_ceil(f));
        let mut per_commit: Vec<f64> = Vec::new();
        let (mut bytes, mut flushes, mut splits, mut merges, mut added) = (0u64, 0, 0, 0, 0u64);
        for col in 0..backfill_columns {
            for c in 0..sample {
                let actions: Vec<_> = commit_window(c, f, n)
                    .map(|id| {
                        action::add_data_file(id, &make_backfill_data_file(id, base_files + col))
                    })
                    .collect();
                added += actions.len() as u64;
                let start = Instant::now();
                let s = tree.commit(actions).await.expect("scale commit");
                per_commit.push(start.elapsed().as_secs_f64() * 1000.0);
                bytes += s.io_write_bytes;
                flushes += s.flushes;
                splits += s.splits;
                merges += s.merges;
            }
        }
        per_commit.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "             backfill sample ({} commits) at scale: per-commit write={:.3} MiB \
             mean={:.2}ms p50={:.2} p99={:.2} | flushes={flushes} splits={splits} merges={merges} height={}",
            per_commit.len(),
            bytes as f64 / per_commit.len().max(1) as f64 / MIB as f64,
            mean(&per_commit),
            percentile(&per_commit, 50.0),
            percentile(&per_commit, 99.0),
            tree.height(),
        );
        final_added = added;
    }

    let final_files = total_files + final_added;
    if final_files <= MATERIALIZE_CAP {
        let t = Instant::now();
        let frags = BeTree::cold_open(object_store, base, scheduler, cache)
            .await
            .expect("scale cold open");
        let secs = t.elapsed().as_secs_f64();
        let got: u64 = frags.iter().map(|fr| fr.files.len() as u64).sum();
        assert_eq!(frags.len() as u64, n, "scale lost fragments");
        assert_eq!(got, final_files, "scale wrong data-file count");
        println!(
            "             cold-open (full materialize) {secs:.2}s: {} fragments / {got} data files verified",
            frags.len()
        );
    } else {
        println!(
            "             cold-open full materialize skipped (>{}M data files exceeds RAM); \
             bootstrap+backfill verified structurally",
            MATERIALIZE_CAP / 1_000_000
        );
    }
    println!();
}

/// Deep-flush mode: bootstrap a multi-level tree, then drive a sustained backfill
/// heavy enough that the root ε-buffer fills repeatedly and flushes **cascade
/// into internal nodes** (not just the root). Measures how deep flushes go
/// (`max_flush_depth`), the flush-depth histogram, per-commit write, and — after
/// the run — how full internal ε-buffers actually got (byte size per level).
#[allow(clippy::too_many_arguments)]
async fn run_deep_flush(
    uri: &str,
    n: u64,
    base_files: u32,
    node_bytes: u64,
    max_children_per_node: u32,
    f: u64,
    commits: u64,
) {
    let io = IOTracker::default();
    let (object_store, base, scheduler, cache) = build_store(uri, &io).await;
    let config = BeTreeConfig::new(node_bytes, max_children_per_node);
    let (mut tree, boot) = BeTree::bootstrap_generate(
        object_store.clone(),
        base.clone(),
        scheduler.clone(),
        cache.clone(),
        config,
        n,
        |id| make_fragment_with_files(id, base_files),
        Vec::new(),
    )
    .await
    .expect("deep bootstrap");
    println!(
        "  [B={:.0} MiB, max_children_per_node {max_children_per_node}] bootstrap {n} frags x {base_files} files: \
         {} leaves, height {}, {:.2} GiB",
        node_bytes as f64 / MIB as f64,
        boot.num_leaves,
        boot.height,
        boot.io_write_bytes as f64 / (1024.0 * MIB as f64),
    );

    let windows = n.div_ceil(f);
    let mut per_commit: Vec<f64> = Vec::new();
    let (mut bytes, mut flushes, mut splits, mut merges) = (0u64, 0u64, 0u64, 0u64);
    let mut depth_hist = [0u64; 16];
    let mut max_depth = 0u32;
    for c in 0..commits {
        let w = c % windows; // cycle across the whole key space
        let col = base_files + (c / windows) as u32; // a fresh column each full cycle
        let actions: Vec<_> = commit_window(w, f, n)
            .map(|id| action::add_data_file(id, &make_backfill_data_file(id, col)))
            .collect();
        let start = Instant::now();
        let s = tree.commit(actions).await.expect("deep commit");
        per_commit.push(start.elapsed().as_secs_f64() * 1000.0);
        bytes += s.io_write_bytes;
        flushes += s.flushes;
        splits += s.splits;
        merges += s.merges;
        let d = (s.max_flush_depth as usize).min(depth_hist.len() - 1);
        depth_hist[d] += 1;
        max_depth = max_depth.max(s.max_flush_depth);
    }
    per_commit.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "             {commits} commits (F={f}): {flushes} flushes, {splits} splits, {merges} merges, \
         **max_flush_depth={max_depth}**, height {}",
        tree.height()
    );
    println!(
        "             flush-depth histogram (commits by deepest flush level): {:?}",
        &depth_hist[..=(max_depth as usize).min(depth_hist.len() - 1)]
    );
    println!(
        "             per-commit write: mean={:.3} MiB p50={:.2}ms p99={:.2}ms max={:.2}ms",
        bytes as f64 / commits.max(1) as f64 / MIB as f64,
        percentile(&per_commit, 50.0),
        percentile(&per_commit, 99.0),
        per_commit.last().copied().unwrap_or(0.0),
    );

    // How full did internal ε-buffers get, per level?
    let sizes = tree.internal_node_sizes().await.expect("node sizes");
    let mut by_h: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
    for (h, b) in sizes {
        by_h.entry(h).or_default().push(b);
    }
    for (h, v) in by_h.iter().rev() {
        let min = *v.iter().min().unwrap();
        let max = *v.iter().max().unwrap();
        let mean = v.iter().sum::<u64>() / v.len() as u64;
        println!(
            "             internal h={h}: {} node(s), size min={:.3} mean={:.3} max={:.3} MiB (B={:.0} MiB)",
            v.len(),
            min as f64 / MIB as f64,
            mean as f64 / MIB as f64,
            max as f64 / MIB as f64,
            node_bytes as f64 / MIB as f64,
        );
    }
    println!();
}

fn bench_betree_backfill(c: &mut Criterion) {
    let runtime = Runtime::new().expect("tokio runtime");

    let base_uri = env::var("BASE_URI").unwrap_or_else(|_| {
        let dir = std::env::temp_dir().join(format!("betree_bench_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_string_lossy().to_string()
    });
    let n = env_u64("NUM_FRAGMENTS", 5000);
    let f_sweep = env_sweep_u64("FRAGMENTS_PER_COMMIT", "10,100");
    let node_sweep = env_sweep_f64("NODE_SIZE_MB", "4,10");
    let max_children_per_node = env_usize("FANOUT", 16) as u32;
    let betree_commits = env_u64("BETREE_COMMITS", 3000);
    let flat_sample = env_u64("FLAT_SAMPLE_COMMITS", 20);
    let run_tag = &Uuid::new_v4().to_string()[..8];

    let base_files = env_usize("BASE_FILES_PER_FRAGMENT", 1) as u32;
    let backfill_columns = env_usize("BACKFILL_COLUMNS", 1) as u32;

    // Deep-flush mode: sustained backfill that cascades flushes into internal nodes.
    let deep_commits = env_u64("DEEP_FLUSH_COMMITS", 0);
    if deep_commits > 0 {
        let f = *f_sweep.first().unwrap_or(&1000);
        println!("=== recursive Bε-tree DEEP-FLUSH benchmark ===");
        println!(
            "base_uri={base_uri}\nN={n} base_files_per_fragment={base_files} node_size_sweep={node_sweep:?} MiB \
             max_children_per_node={max_children_per_node} F={f} deep_flush_commits={deep_commits} run_tag={run_tag}\n"
        );
        for &node_mib in &node_sweep {
            let uri = format!(
                "{}/{run_tag}/deepB{node_mib}",
                base_uri.trim_end_matches('/')
            );
            runtime.block_on(run_deep_flush(
                &uri,
                n,
                base_files.max(2),
                (node_mib * MIB as f64) as u64,
                max_children_per_node,
                f,
                deep_commits,
            ));
        }
        let _ = c;
        return;
    }

    // Scale mode: bootstrap fat fragments (many data files) to reach billion-scale.
    if base_files > 1 {
        let f = *f_sweep.first().unwrap_or(&100);
        println!("=== recursive Bε-tree AT-SCALE benchmark ===");
        println!(
            "base_uri={base_uri}\nN={n} base_files_per_fragment={base_files} node_size_sweep={node_sweep:?} MiB \
             max_children_per_node={max_children_per_node} backfill_columns={backfill_columns} F={f} run_tag={run_tag}\n"
        );
        for &node_mib in &node_sweep {
            let uri = format!(
                "{}/{run_tag}/scaleB{node_mib}",
                base_uri.trim_end_matches('/')
            );
            runtime.block_on(run_scale(
                &uri,
                n,
                base_files,
                (node_mib * MIB as f64) as u64,
                max_children_per_node,
                backfill_columns,
                f,
            ));
        }
        let _ = c;
        return;
    }

    println!("=== recursive Bε-tree fine-grained backfill benchmark ===");
    println!("base_uri={base_uri}");
    println!(
        "N={n} fragments_per_commit={f_sweep:?} node_size_sweep={node_sweep:?} MiB max_children_per_node={max_children_per_node} betree_commits={betree_commits} flat_sample={flat_sample} run_tag={run_tag}"
    );
    println!();

    for &f in &f_sweep {
        let total_commits = n.div_ceil(f);
        println!(
            "--- F = {f} fragments/commit  (full backfill = {total_commits} commits at N={n}) ---"
        );
        let flat_uri = format!("{}/{run_tag}/f{f}/flat", base_uri.trim_end_matches('/'));
        let flat = runtime.block_on(run_flat(&flat_uri, n, f, flat_sample));
        print_result(&flat, n, f);

        for &node_mib in &node_sweep {
            let node_bytes = (node_mib * MIB as f64) as u64;
            let betree_uri = format!(
                "{}/{run_tag}/f{f}/betreeB{node_mib}",
                base_uri.trim_end_matches('/')
            );
            let betree = runtime.block_on(run_betree(
                &betree_uri,
                n,
                f,
                betree_commits,
                node_bytes,
                max_children_per_node,
            ));
            print_result(&betree, n, f);
            let write_ratio = flat.full_backfill_write_bytes(n, f)
                / betree.full_backfill_write_bytes(n, f).max(1.0);
            let per_commit_ratio =
                flat.per_commit_write_bytes() / betree.per_commit_write_bytes().max(1.0);
            let mean_speedup = flat.commit_mean_ms / betree.commit_mean_ms.max(0.001);
            println!(
                "  => B={node_mib} MiB: full-backfill {write_ratio:.0}x fewer write bytes \
                 ({per_commit_ratio:.0}x per commit), {mean_speedup:.1}x lower mean commit latency \
                 (flat {:.0}ms vs betree {:.0}ms)",
                flat.commit_mean_ms, betree.commit_mean_ms
            );
        }
        println!();
    }

    let _ = c;
}

criterion_group!(benches, bench_betree_backfill);
criterion_main!(benches);
