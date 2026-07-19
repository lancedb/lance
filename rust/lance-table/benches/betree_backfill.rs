// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE benchmark (discussion lance-format/lance#7499): add-column / backfill
//! write amplification, flat manifest vs Bε-tree — **fine-grained commit** regime.
//!
//! Realistic embedding backfill commits touch only a handful of fragments at a
//! time (GPU compute trickles in), so a full backfill over N fragments is N/F
//! commits — up to hundreds of thousands. Flat rewrites the whole (growing)
//! manifest on *every* commit regardless of F, so its full-backfill write is
//! (N/F) × manifest — petabytes at F=10, N=1M. We therefore measure **per-commit**
//! cost and **extrapolate** the full backfill:
//!   - flat: run a small uniform sample (`FLAT_SAMPLE_COMMITS`) — every commit is
//!     the same full-manifest rewrite — and extrapolate ×(N/F).
//!   - Bε-tree: run `BETREE_COMMITS` commits (enough to reach steady state with
//!     several flushes), measure per-commit + cumulative, extrapolate ×(N/F).
//!
//! Data-file names use Lance's real 50-char format (see `support.rs`).
//!
//! ## Configuration (env)
//! - `BASE_URI`             s3://… / s3://…--x-s3 / local path. Default: temp dir.
//! - `NUM_FRAGMENTS`        bootstrap fragment count (N). Default 5000.
//! - `FRAGMENTS_PER_COMMIT` comma sweep of F (fragments touched per commit). Default "10,100".
//! - `BETREE_COMMITS`       Bε-tree commits to run per config (M). Default 3000.
//! - `FLAT_SAMPLE_COMMITS`  flat commits to sample (uniform). Default 20.
//! - `BUFFER_CAP_MB`        comma sweep of ε (root buffer cap) in MiB, fractional ok. Default "1".
//! - `NUM_CHILDREN`         children the id space is partitioned into. Default 24 (~10 MiB nodes at N=1M).
//! - `S3_EXPRESS`           "true" for S3 Express directory buckets.
//! - `AWS_REGION`           required for S3.

#![allow(clippy::print_stdout)]

use std::collections::HashMap;
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
use lance_table::betree::support::{make_backfill_data_file, make_fragment};
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

/// Storage options + S3-Express flag derived from the URI / env.
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
    // Zero-capacity cache so cold-open reads hit storage (not a warm cache).
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

/// The fragment id window touched by commit `c` at `f` fragments/commit.
fn commit_window(c: u64, f: u64, n: u64) -> std::ops::Range<u64> {
    let start = (c * f).min(n);
    let end = (start + f).min(n);
    start..end
}

#[derive(Default)]
struct RunResult {
    layout: &'static str,
    buffer_cap_mib: f64,
    commits_run: u64,
    /// Bytes written across the measured commits (excludes bootstrap).
    backfill_write_bytes: u64,
    bootstrap_write_bytes: u64,
    flushes: u64,
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
    /// Extrapolated total write to backfill all N fragments (N/F commits + bootstrap).
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

    // Every flat commit is the same full-manifest rewrite, so a small sample
    // characterizes per-commit cost; we extrapolate the full backfill.
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

    // Cold-open only to verify correctness (the touched fragments gained a file).
    let final_version = flat.version();
    let manifest = FlatBaseline::cold_open(&object_store, &base, final_version)
        .await
        .expect("flat cold open");
    let total_data_files: usize = manifest.fragments.iter().map(|fr| fr.files.len()).sum();

    let commit_mean_ms = mean(&commit_ms);
    commit_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    RunResult {
        layout: "flat",
        buffer_cap_mib: 0.0,
        commits_run: commits,
        backfill_write_bytes,
        bootstrap_write_bytes,
        flushes: 0,
        commit_mean_ms,
        commit_p50_ms: percentile(&commit_ms, 50.0),
        commit_p99_ms: percentile(&commit_ms, 99.0),
        commit_max_ms: commit_ms.last().copied().unwrap_or(0.0),
        materialized_fragments: manifest.fragments.len(),
        total_data_files,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_betree(
    uri: &str,
    n: u64,
    f: u64,
    m_commits: u64,
    buffer_cap_bytes: u64,
    num_children: usize,
) -> RunResult {
    let io = IOTracker::default();
    let (object_store, base, scheduler, cache) = build_store(uri, &io).await;
    let fragments: Vec<Fragment> = (0..n).map(make_fragment).collect();

    let (mut tree, boot) = BeTree::bootstrap(
        object_store.clone(),
        base.clone(),
        scheduler.clone(),
        cache.clone(),
        BeTreeConfig {
            buffer_cap_bytes,
            num_children,
        },
        fragments,
        Vec::new(),
    )
    .await
    .expect("betree bootstrap");
    let bootstrap_write_bytes = boot.child_bytes + boot.root_bytes;

    let commits = m_commits.min(n.div_ceil(f));
    let mut commit_ms: Vec<f64> = Vec::with_capacity(commits as usize);
    let mut backfill_write_bytes = 0u64;
    let mut flushes = 0u64;
    for c in 0..commits {
        let actions: Vec<_> = commit_window(c, f, n)
            .map(|id| action::add_data_file(id, &make_backfill_data_file(id, 0)))
            .collect();
        let start = Instant::now();
        let stats = tree.commit(actions).await.expect("betree commit");
        commit_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        backfill_write_bytes += stats.write_bytes();
        if stats.flushed {
            flushes += 1;
        }
    }
    println!(
        "  [betree cap={:.2} MiB, {num_children} children] flushes in {commits} commits: {flushes}",
        buffer_cap_bytes as f64 / MIB as f64
    );

    // Cold-open only to verify correctness.
    let frags = BeTree::cold_open(object_store.clone(), base.clone(), scheduler, cache)
        .await
        .expect("betree cold open");
    let total_data_files: usize = frags.iter().map(|fr| fr.files.len()).sum();

    let commit_mean_ms = mean(&commit_ms);
    commit_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    RunResult {
        layout: "betree",
        buffer_cap_mib: buffer_cap_bytes as f64 / MIB as f64,
        commits_run: commits,
        backfill_write_bytes,
        bootstrap_write_bytes,
        flushes,
        commit_mean_ms,
        commit_p50_ms: percentile(&commit_ms, 50.0),
        commit_p99_ms: percentile(&commit_ms, 99.0),
        commit_max_ms: commit_ms.last().copied().unwrap_or(0.0),
        materialized_fragments: frags.len(),
        total_data_files,
    }
}

fn print_result(r: &RunResult, n: u64, f: u64) {
    // Correctness: the fragments touched by the measured commits gained exactly
    // one data file each; the rest keep their single base file.
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
        format!("betree/{:.2}MiB", r.buffer_cap_mib)
    };
    println!(
        "  {:<14} per-commit write={:>9.3} MiB | commit mean={:>8.2} p50={:>8.2} p99={:>8.2} max={:>8.2} ms \
         | full-backfill write~={:>9.2} GiB ({} flushes/{} commits)",
        label,
        r.per_commit_write_bytes() / MIB as f64,
        r.commit_mean_ms,
        r.commit_p50_ms,
        r.commit_p99_ms,
        r.commit_max_ms,
        r.full_backfill_write_bytes(n, f) / (1024.0 * MIB as f64),
        r.flushes,
        r.commits_run,
    );
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
    let cap_sweep = env_sweep_f64("BUFFER_CAP_MB", "1");
    let betree_commits = env_u64("BETREE_COMMITS", 3000);
    let flat_sample = env_u64("FLAT_SAMPLE_COMMITS", 20);
    let num_children = env_usize("NUM_CHILDREN", 24);
    let run_tag = &Uuid::new_v4().to_string()[..8];

    println!("=== Bε-tree fine-grained backfill benchmark ===");
    println!("base_uri={base_uri}");
    println!(
        "N={n} fragments_per_commit_sweep={f_sweep:?} cap_sweep={cap_sweep:?} MiB \
         num_children={num_children} betree_commits={betree_commits} flat_sample={flat_sample} run_tag={run_tag}"
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

        for &cap_mib in &cap_sweep {
            let cap_bytes = (cap_mib * MIB as f64) as u64;
            let betree_uri = format!(
                "{}/{run_tag}/f{f}/betree{cap_mib}",
                base_uri.trim_end_matches('/')
            );
            let betree = runtime.block_on(run_betree(
                &betree_uri,
                n,
                f,
                betree_commits,
                cap_bytes,
                num_children,
            ));
            print_result(&betree, n, f);
            let write_ratio = flat.full_backfill_write_bytes(n, f)
                / betree.full_backfill_write_bytes(n, f).max(1.0);
            let per_commit_ratio =
                flat.per_commit_write_bytes() / betree.per_commit_write_bytes().max(1.0);
            let mean_speedup = flat.commit_mean_ms / betree.commit_mean_ms.max(0.001);
            println!(
                "  => cap={cap_mib} MiB: full-backfill {write_ratio:.0}x fewer write bytes \
                 ({per_commit_ratio:.0}x per commit), {mean_speedup:.1}x lower mean commit latency \
                 (flat {:.0}ms vs betree {:.0}ms)",
                flat.commit_mean_ms, betree.commit_mean_ms
            );
        }
        println!();
    }

    // The printed table + logs are this bench's authoritative output; a single
    // S3 backfill sweep is far too heavy for Criterion's sampling harness, so we
    // register no (misleading) Criterion metric.
    let _ = c;
}

criterion_group!(benches, bench_betree_backfill);
criterion_main!(benches);
