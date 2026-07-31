// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stdout)]

//! Reproducible exact compound-FTS correctness and performance benchmark.
//!
//! Every timed optimized query is checked against an exhaustive result ordered
//! by `(score DESC, row_id ASC)`. Scores use the absolute and relative
//! tolerances declared in `workload.rs`. The benchmark emits raw JSONL records;
//! it deliberately does not calculate or claim a speedup. Compare two records
//! only when their dataset fingerprint, hardware, build profile, cache state,
//! iteration count, and configuration fields match.

mod workload;

use std::collections::{BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, StringArray, UInt64Array};
use chrono::Utc;
use clap::{Parser, ValueEnum};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance_datafusion::exec::{ExecutionSummaryCounts, OUTPUT_ROWS_METRIC};
use lance_index::IndexType;
use lance_index::metrics::{
    FTS_CANDIDATES_SCORED_METRIC, FTS_CANDIDATES_VISITED_METRIC, FTS_PHRASE_POSITION_CHECKS_METRIC,
    FTS_POSTING_BLOCKS_DECODED_METRIC,
};
use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
use serde_json::{Value, json};
use tempfile::TempDir;
use workload::{
    BenchResult, DatasetKind, FILTER_COLUMN, SCORE_ABS_TOLERANCE, SCORE_REL_TOLERANCE, TEXT_COLUMN,
    WorkloadSpec, assert_exact_top_k, execute_query, exhaustive_top_k, workload_specs,
};

const DATASET_SEED: u64 = 0x1597_C0DE;
const RICH_INDEX_NAME: &str = "compound_fts";
const WIDE_INDEX_PREFIX: &str = "compound_wide";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MatrixProfile {
    Smoke,
    Full,
}

impl MatrixProfile {
    fn rows_per_fragment(self) -> usize {
        match self {
            Self::Smoke => 256,
            Self::Full => 512,
        }
    }

    fn indexed_fragments(self) -> usize {
        match self {
            Self::Smoke => 4,
            Self::Full => 16,
        }
    }

    fn many_segments(self) -> usize {
        match self {
            Self::Smoke => 2,
            Self::Full => 8,
        }
    }

    fn should_clause_counts(self) -> &'static [usize] {
        match self {
            Self::Smoke => &[8],
            Self::Full => &[8, 32, 128],
        }
    }

    fn must_clause_counts(self) -> &'static [usize] {
        match self {
            Self::Smoke => &[2],
            Self::Full => &[2, 4, 8],
        }
    }

    fn nested_depths(self) -> &'static [usize] {
        match self {
            Self::Smoke => &[2],
            Self::Full => &[2, 4],
        }
    }

    fn multi_match_field_counts(self) -> &'static [usize] {
        match self {
            Self::Smoke => &[4],
            Self::Full => &[4, 32, 256, 500],
        }
    }

    fn max_fields(self) -> usize {
        *self
            .multi_match_field_counts()
            .iter()
            .max()
            .expect("profile has at least one field count")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DatasetShape {
    SingleSegment,
    ManySegments,
    ManyPartitions,
    WideFields,
}

impl DatasetShape {
    fn name(self) -> &'static str {
        match self {
            Self::SingleSegment => "single_segment",
            Self::ManySegments => "many_segments",
            Self::ManyPartitions => "many_partitions",
            Self::WideFields => "wide_fields",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CacheState {
    Cold,
    Warm,
}

impl CacheState {
    fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
        }
    }

    fn methodology(self) -> &'static str {
        match self {
            Self::Cold => {
                "fresh Dataset session per measured query; OS page cache is not forcibly dropped"
            }
            Self::Warm => {
                "one Dataset session, one untimed warm-up, then repeated measured queries"
            }
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Exact compound FTS correctness and performance benchmark")]
struct Args {
    /// Cargo appends this flag when launching a harness-free benchmark.
    #[arg(long = "bench", hide = true)]
    _bench: bool,

    /// Full acceptance matrix or a small local validation matrix.
    #[arg(long, value_enum, default_value = "full")]
    profile: MatrixProfile,

    /// Number of optimized executions per workload/cache-state point.
    #[arg(long, default_value_t = 20)]
    iterations: usize,

    /// Only verify oracle equality once per matrix point.
    #[arg(long)]
    verify_only: bool,

    /// Reusable dataset root. Omit to use a temporary directory.
    #[arg(long)]
    dataset_root: Option<PathBuf>,

    /// Remove and rebuild an explicitly supplied dataset root.
    #[arg(long, requires = "dataset_root")]
    rebuild: bool,

    /// Append JSONL output here in addition to stdout.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Label stored in every record, for example "baseline" or "current".
    #[arg(long, default_value = "current")]
    run_label: String,

    /// Stable identifier shared by comparable baseline/current runs.
    #[arg(long)]
    run_id: Option<String>,
}

#[derive(Clone, Debug)]
struct DatasetLocation {
    uri: String,
    shape: DatasetShape,
    segment_count: usize,
    partition_count: usize,
    indexed_rows: usize,
    overlay_rows: usize,
}

struct DatasetRegistry {
    rich: HashMap<&'static str, DatasetLocation>,
    wide: DatasetLocation,
}

struct DatasetRoot {
    _temporary: Option<TempDir>,
    path: PathBuf,
}

impl DatasetRoot {
    fn new(args: &Args) -> BenchResult<Self> {
        match &args.dataset_root {
            Some(path) => {
                if args.rebuild && path.exists() {
                    std::fs::remove_dir_all(path)?;
                }
                std::fs::create_dir_all(path)?;
                Ok(Self {
                    _temporary: None,
                    path: path.clone(),
                })
            }
            None => {
                let temporary = TempDir::new()?;
                let path = temporary.path().to_path_buf();
                Ok(Self {
                    _temporary: Some(temporary),
                    path,
                })
            }
        }
    }
}

fn deterministic_mix(row: usize, term: usize) -> u64 {
    let mut value = (row as u64)
        .wrapping_add(DATASET_SEED)
        .wrapping_add((term as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn rich_text(row: usize) -> String {
    if row % 256 < 4 {
        return ["corpus".to_string()]
            .into_iter()
            .chain((0..8).map(|term| format!("high{term:03}")))
            .collect::<Vec<_>>()
            .join(" ");
    }

    let mut tokens = Vec::with_capacity(160);
    tokens.push("corpus".to_string());
    for term in 0..128 {
        if !deterministic_mix(row, term).is_multiple_of(4) {
            tokens.push(format!("high{term:03}"));
        }
        if deterministic_mix(row, term + 128).is_multiple_of(64) {
            tokens.push(format!("low{term:03}"));
        }
    }
    if !row.is_multiple_of(3) {
        for term in 0..8 {
            tokens.push(format!("must{term:03}"));
        }
    }
    for term in 0..16 {
        if !deterministic_mix(row, term + 256).is_multiple_of(3) {
            tokens.push(format!("nested{term:03}"));
        }
    }
    if row.is_multiple_of(5) {
        tokens.extend([
            "phraserequired".to_string(),
            "quick".to_string(),
            "brown".to_string(),
        ]);
    }
    if row % 3 != 1 {
        tokens.push("boostpositive".to_string());
    }
    if row.is_multiple_of(2) {
        tokens.push("negativecommon".to_string());
    }
    if row.is_multiple_of(97) {
        tokens.push("negativerare".to_string());
    }
    tokens.push(format!("document{row:08}"));
    tokens.join(" ")
}

fn category(row: usize) -> &'static str {
    if row.is_multiple_of(16) {
        "keep"
    } else {
        "drop"
    }
}

fn rich_batch(start_row: usize, rows: usize) -> RecordBatch {
    let ids = Arc::new(UInt64Array::from_iter_values(
        start_row as u64..(start_row + rows) as u64,
    ));
    let texts = Arc::new(StringArray::from_iter_values(
        (start_row..start_row + rows).map(rich_text),
    ));
    let categories = Arc::new(StringArray::from_iter_values(
        (start_row..start_row + rows).map(category),
    ));
    RecordBatch::try_from_iter(vec![
        ("id", ids as ArrayRef),
        (TEXT_COLUMN, texts as ArrayRef),
        (FILTER_COLUMN, categories as ArrayRef),
    ])
    .expect("deterministic rich batch must be valid")
}

fn wide_batch(start_row: usize, rows: usize, field_count: usize) -> RecordBatch {
    let ids = Arc::new(UInt64Array::from_iter_values(
        start_row as u64..(start_row + rows) as u64,
    ));
    let categories = Arc::new(StringArray::from_iter_values(
        (start_row..start_row + rows).map(category),
    ));
    let patterns = (0..4)
        .map(|pattern| {
            Arc::new(StringArray::from_iter_values(
                (start_row..start_row + rows).map(move |row| {
                    if !(row + pattern).is_multiple_of(5) {
                        format!("widematch pattern{pattern}")
                    } else {
                        format!("other pattern{pattern}")
                    }
                }),
            )) as ArrayRef
        })
        .collect::<Vec<_>>();

    let mut columns = Vec::with_capacity(field_count + 2);
    columns.push(("id".to_string(), ids as ArrayRef));
    columns.push((FILTER_COLUMN.to_string(), categories as ArrayRef));
    for field in 0..field_count {
        columns.push((
            format!("field_{field:03}"),
            patterns[field % patterns.len()].clone(),
        ));
    }
    RecordBatch::try_from_iter(columns).expect("deterministic wide batch must be valid")
}

fn fragment_ids(dataset: &Dataset) -> Vec<u32> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect()
}

fn group_fragment_ids(fragment_ids: &[u32], group_count: usize) -> Vec<Vec<u32>> {
    let mut groups = vec![Vec::new(); group_count.min(fragment_ids.len()).max(1)];
    for (index, fragment_id) in fragment_ids.iter().copied().enumerate() {
        let group_index = index * groups.len() / fragment_ids.len();
        groups[group_index].push(fragment_id);
    }
    groups
}

fn count_partitions(segment: &lance_table::format::IndexMetadata) -> usize {
    segment
        .files
        .as_ref()
        .map(|files| {
            files
                .iter()
                .filter_map(|file| {
                    file.path
                        .strip_prefix("part_")
                        .and_then(|path| path.split_once('_'))
                        .map(|(partition_id, _)| partition_id.to_string())
                })
                .collect::<BTreeSet<_>>()
                .len()
        })
        .unwrap_or_default()
}

async fn append_batch(dataset: &mut Dataset, batch: RecordBatch) -> BenchResult<()> {
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    dataset.append(reader, None).await?;
    Ok(())
}

async fn add_filter_index(dataset: &mut Dataset) -> BenchResult<()> {
    dataset
        .create_index(
            &[FILTER_COLUMN],
            IndexType::BTree,
            Some("compound_category".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await?;
    Ok(())
}

async fn build_rich_dataset(
    root: &Path,
    profile: MatrixProfile,
    shape: DatasetShape,
) -> BenchResult<DatasetLocation> {
    let path = root.join(shape.name());
    let uri = path.to_string_lossy().into_owned();
    if path.join("_versions").exists() {
        let dataset = Dataset::open(&uri).await?;
        return describe_dataset(dataset, uri, shape, RICH_INDEX_NAME, profile).await;
    }

    let rows_per_fragment = profile.rows_per_fragment();
    let indexed_fragments = profile.indexed_fragments();
    let batches = (0..indexed_fragments)
        .map(|fragment| Ok(rich_batch(fragment * rows_per_fragment, rows_per_fragment)))
        .collect::<Vec<_>>();
    let schema = batches[0].as_ref().expect("first batch is valid").schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        &uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            ..Default::default()
        }),
    )
    .await?;
    let indexed_fragment_ids = fragment_ids(&dataset);
    let memory_limit_mb = if shape == DatasetShape::ManyPartitions {
        1
    } else {
        512
    };
    let num_workers = if shape == DatasetShape::ManyPartitions {
        2
    } else {
        1
    };
    let params = InvertedIndexParams::default()
        .with_position(true)
        .num_workers(num_workers)
        .memory_limit_mb(memory_limit_mb);
    let groups = match shape {
        DatasetShape::SingleSegment | DatasetShape::ManyPartitions => {
            vec![indexed_fragment_ids.clone()]
        }
        DatasetShape::ManySegments => {
            group_fragment_ids(&indexed_fragment_ids, profile.many_segments())
        }
        DatasetShape::WideFields => unreachable!("wide fields use a separate builder"),
    };
    let mut staged = Vec::with_capacity(groups.len());
    for group in groups {
        staged.push(
            dataset
                .create_index_builder(&[TEXT_COLUMN], IndexType::Inverted, &params)
                .name(RICH_INDEX_NAME.to_string())
                .fragments(group)
                .execute_uncommitted()
                .await?,
        );
    }
    dataset
        .commit_existing_index_segments(RICH_INDEX_NAME, TEXT_COLUMN, staged)
        .await?;

    let overlay_start = indexed_fragments * rows_per_fragment;
    append_batch(&mut dataset, rich_batch(overlay_start, rows_per_fragment)).await?;
    add_filter_index(&mut dataset).await?;
    describe_dataset(dataset, uri, shape, RICH_INDEX_NAME, profile).await
}

async fn build_wide_dataset(root: &Path, profile: MatrixProfile) -> BenchResult<DatasetLocation> {
    let shape = DatasetShape::WideFields;
    let path = root.join(shape.name());
    let uri = path.to_string_lossy().into_owned();
    if path.join("_versions").exists() {
        let dataset = Dataset::open(&uri).await?;
        return describe_dataset(
            dataset,
            uri,
            shape,
            &format!("{WIDE_INDEX_PREFIX}_000"),
            profile,
        )
        .await;
    }

    let rows_per_fragment = profile.rows_per_fragment();
    let indexed_fragments = profile.indexed_fragments();
    let field_count = profile.max_fields();
    let batches = (0..indexed_fragments)
        .map(|fragment| {
            Ok(wide_batch(
                fragment * rows_per_fragment,
                rows_per_fragment,
                field_count,
            ))
        })
        .collect::<Vec<_>>();
    let schema = batches[0].as_ref().expect("first batch is valid").schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        &uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            ..Default::default()
        }),
    )
    .await?;
    let indexed_fragment_ids = fragment_ids(&dataset);
    let params = InvertedIndexParams::default().num_workers(1);
    for field in 0..field_count {
        let column = format!("field_{field:03}");
        let index_name = format!("{WIDE_INDEX_PREFIX}_{field:03}");
        let segment = dataset
            .create_index_builder(&[column.as_str()], IndexType::Inverted, &params)
            .name(index_name.clone())
            .fragments(indexed_fragment_ids.clone())
            .execute_uncommitted()
            .await?;
        dataset
            .commit_existing_index_segments(&index_name, &column, vec![segment])
            .await?;
    }

    let overlay_start = indexed_fragments * rows_per_fragment;
    append_batch(
        &mut dataset,
        wide_batch(overlay_start, rows_per_fragment, field_count),
    )
    .await?;
    add_filter_index(&mut dataset).await?;
    describe_dataset(
        dataset,
        uri,
        shape,
        &format!("{WIDE_INDEX_PREFIX}_000"),
        profile,
    )
    .await
}

async fn describe_dataset(
    dataset: Dataset,
    uri: String,
    shape: DatasetShape,
    index_name: &str,
    profile: MatrixProfile,
) -> BenchResult<DatasetLocation> {
    let segments = dataset.load_indices_by_name(index_name).await?;
    let partition_count = segments.iter().map(count_partitions).sum();
    match shape {
        DatasetShape::SingleSegment if segments.len() != 1 => {
            return Err(format!(
                "single-segment dataset produced {} index segments",
                segments.len()
            )
            .into());
        }
        DatasetShape::ManySegments if segments.len() < 2 => {
            return Err("many-segment dataset produced fewer than two index segments".into());
        }
        DatasetShape::ManyPartitions if partition_count < 2 => {
            return Err(
                "many-partition dataset produced fewer than two index partitions; \
                 increase the profile size or lower its memory budget"
                    .into(),
            );
        }
        _ => {}
    }
    Ok(DatasetLocation {
        uri,
        shape,
        segment_count: segments.len(),
        partition_count,
        indexed_rows: profile.indexed_fragments() * profile.rows_per_fragment(),
        overlay_rows: profile.rows_per_fragment(),
    })
}

fn dataset_fingerprint(profile: MatrixProfile) -> Value {
    json!({
        "generator_version": 1,
        "seed": DATASET_SEED,
        "profile": format!("{profile:?}").to_lowercase(),
        "rows_per_fragment": profile.rows_per_fragment(),
        "indexed_fragments": profile.indexed_fragments(),
        "overlay_fragments": 1,
        "many_segments": profile.many_segments(),
        "max_fields": profile.max_fields(),
        "tokenizer": "default",
        "with_positions": true,
        "many_partition_memory_limit_mib": 1,
        "many_partition_workers": 2,
    })
}

async fn build_registry(root: &Path, profile: MatrixProfile) -> BenchResult<DatasetRegistry> {
    let expected_fingerprint = dataset_fingerprint(profile);
    let marker_path = root.join("compound_fts_config.json");
    if marker_path.exists() {
        let observed: Value = serde_json::from_reader(File::open(&marker_path)?)?;
        if observed != expected_fingerprint {
            return Err(format!(
                "dataset root {} was built with a different configuration; pass --rebuild",
                root.display()
            )
            .into());
        }
    }

    let mut rich = HashMap::new();
    for shape in [
        DatasetShape::SingleSegment,
        DatasetShape::ManySegments,
        DatasetShape::ManyPartitions,
    ] {
        rich.insert(
            shape.name(),
            build_rich_dataset(root, profile, shape).await?,
        );
    }
    let wide = build_wide_dataset(root, profile).await?;
    serde_json::to_writer_pretty(File::create(marker_path)?, &expected_fingerprint)?;
    Ok(DatasetRegistry { rich, wide })
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn machine_metadata() -> Value {
    let cpu = if cfg!(target_os = "linux") {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    line.strip_prefix("model name")
                        .and_then(|line| line.split_once(':'))
                        .map(|(_, value)| value.trim().to_string())
                })
            })
    } else if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        None
    };
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": cpu,
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "hostname": command_output("hostname", &[]),
    })
}

fn git_metadata() -> Value {
    json!({
        "commit": command_output("git", &["rev-parse", "HEAD"]),
        "dirty": command_output("git", &["status", "--porcelain"])
            .is_some_and(|status| !status.is_empty()),
        "rustc": command_output("rustc", &["--version"]),
        "debug_assertions": cfg!(debug_assertions),
    })
}

fn current_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
            return 0;
        };
        let resident_pages = statm
            .split_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or_default();
        // SAFETY: sysconf is a read-only libc query with no pointer arguments.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        resident_pages.saturating_mul(page_size.max(0) as u64)
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: proc_pidinfo initializes a fixed-size process-local out-param.
        unsafe {
            let mut task_info: libc::proc_taskinfo = std::mem::zeroed();
            let expected_size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
            let bytes_read = libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                &mut task_info as *mut libc::proc_taskinfo as *mut libc::c_void,
                expected_size,
            );
            if bytes_read == expected_size {
                task_info.pti_resident_size
            } else {
                0
            }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

struct RssSampler {
    baseline: u64,
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start() -> Self {
        let baseline = current_rss_bytes();
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(baseline));
        let thread_stop = stop.clone();
        let thread_peak = peak.clone();
        let handle = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                thread_peak.fetch_max(current_rss_bytes(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        Self {
            baseline,
            stop,
            peak,
            handle: Some(handle),
        }
    }

    fn finish(&mut self) -> (u64, u64) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.peak.fetch_max(current_rss_bytes(), Ordering::Relaxed);
        let peak = self.peak.load(Ordering::Relaxed);
        (peak, peak.saturating_sub(self.baseline))
    }

    fn stop(mut self) -> (u64, u64) {
        self.finish()
    }
}

impl Drop for RssSampler {
    fn drop(&mut self) {
        self.finish();
    }
}

fn percentile_duration(sorted: &[Duration], percentile: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((percentile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[rank]
}

fn median_count(values: &mut [usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn metric_count(stats: &ExecutionSummaryCounts, metric: &str) -> usize {
    stats.all_counts.get(metric).copied().unwrap_or_default()
}

fn summarize_metric(
    executions: &[workload::QueryExecution],
    metric: impl Fn(&ExecutionSummaryCounts) -> usize,
) -> usize {
    let mut values = executions
        .iter()
        .map(|execution| metric(&execution.stats))
        .collect::<Vec<_>>();
    median_count(&mut values)
}

fn locations_for<'a>(
    registry: &'a DatasetRegistry,
    workload: &WorkloadSpec,
) -> Vec<&'a DatasetLocation> {
    match workload.dataset_kind {
        DatasetKind::RichText => [
            DatasetShape::SingleSegment,
            DatasetShape::ManySegments,
            DatasetShape::ManyPartitions,
        ]
        .into_iter()
        .map(|shape| {
            registry
                .rich
                .get(shape.name())
                .expect("all rich shapes are registered")
        })
        .collect(),
        DatasetKind::WideFields => vec![&registry.wide],
    }
}

async fn run_point(
    workload: &WorkloadSpec,
    location: &DatasetLocation,
    k: usize,
    cache_state: CacheState,
    iterations: usize,
) -> BenchResult<Value> {
    let oracle_dataset = Dataset::open(&location.uri).await?;
    let exhaustive = execute_query(&oracle_dataset, workload, None).await?;
    let oracle = exhaustive_top_k(exhaustive.rows, k);

    let warm_dataset = if matches!(cache_state, CacheState::Warm) {
        let dataset = Dataset::open(&location.uri).await?;
        let warmup = execute_query(&dataset, workload, Some(k)).await?;
        assert_exact_top_k(&workload.name, &oracle, &warmup.rows)?;
        Some(dataset)
    } else {
        None
    };

    let sampler = RssSampler::start();
    let mut executions = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let execution = match &warm_dataset {
            Some(dataset) => execute_query(dataset, workload, Some(k)).await?,
            None => {
                let dataset = Dataset::open(&location.uri).await?;
                execute_query(&dataset, workload, Some(k)).await?
            }
        };
        assert_exact_top_k(&workload.name, &oracle, &execution.rows)?;
        executions.push(execution);
    }
    let (peak_rss_bytes, peak_rss_delta_bytes) = sampler.stop();
    let mut latencies = executions
        .iter()
        .map(|execution| execution.elapsed)
        .collect::<Vec<_>>();
    latencies.sort_unstable();

    Ok(json!({
        "record_type": "compound_fts_result",
        "workload": workload.name,
        "family": workload.family,
        "dataset_shape": location.shape.name(),
        "k": k,
        "cache_state": cache_state.name(),
        "cache_methodology": cache_state.methodology(),
        "iterations": iterations,
        "latency_p50_us": percentile_duration(&latencies, 0.50).as_micros(),
        "latency_p95_us": percentile_duration(&latencies, 0.95).as_micros(),
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_delta_bytes": peak_rss_delta_bytes,
        "candidates_visited": summarize_metric(&executions, |stats| {
            metric_count(stats, FTS_CANDIDATES_VISITED_METRIC)
        }),
        "candidates_scored": summarize_metric(&executions, |stats| {
            metric_count(stats, FTS_CANDIDATES_SCORED_METRIC)
        }),
        "posting_blocks_decoded": summarize_metric(&executions, |stats| {
            metric_count(stats, FTS_POSTING_BLOCKS_DECODED_METRIC)
        }),
        "phrase_position_checks": summarize_metric(&executions, |stats| {
            metric_count(stats, FTS_PHRASE_POSITION_CHECKS_METRIC)
        }),
        "rows_materialized": summarize_metric(&executions, |stats| {
            metric_count(stats, OUTPUT_ROWS_METRIC)
        }),
        "index_cache_hits": summarize_metric(&executions, |stats| stats.index_cache_hits()),
        "index_cache_misses": summarize_metric(&executions, |stats| stats.index_cache_misses()),
        "segments": location.segment_count,
        "partitions": location.partition_count,
        "indexed_rows": location.indexed_rows,
        "fresh_overlay_rows": location.overlay_rows,
        "fresh_overlay_supported": workload.exercises_fresh_overlay,
        "result_rows": oracle.len(),
        "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        "score_rel_tolerance": SCORE_REL_TOLERANCE,
        "oracle": "unlimited exhaustive execution, canonical score DESC / row_id ASC, truncate k",
    }))
}

fn write_record(record: &Value, output: &mut Option<BufWriter<File>>) -> BenchResult<()> {
    println!("{}", serde_json::to_string(record)?);
    if let Some(output) = output {
        serde_json::to_writer(&mut *output, record)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> BenchResult<()> {
    let args = Args::parse();
    if args.iterations == 0 {
        return Err("--iterations must be at least 1".into());
    }
    let iterations = if args.verify_only { 1 } else { args.iterations };
    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    let dataset_root = DatasetRoot::new(&args)?;
    let registry = build_registry(&dataset_root.path, args.profile).await?;
    let workloads = workload_specs(
        args.profile.should_clause_counts(),
        args.profile.must_clause_counts(),
        args.profile.nested_depths(),
        args.profile.multi_match_field_counts(),
    )?;
    let mut output = args
        .output
        .as_ref()
        .map(|path| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map(BufWriter::new)
        })
        .transpose()?;

    let fingerprint = dataset_fingerprint(args.profile);
    let machine = machine_metadata();
    let build = git_metadata();
    let common = json!({
        "run_id": run_id,
        "run_label": args.run_label,
        "dataset_fingerprint": fingerprint.clone(),
        "machine": machine.clone(),
        "build": build.clone(),
        "benchmark_profile": format!("{:?}", args.profile).to_lowercase(),
        "counter_aggregation": "median per measured iteration",
        "memory_scope": "maximum process RSS during all measured iterations for this point",
        "release_profile_required_for_comparison": true,
        "speedup_claimed": false,
    });
    write_record(
        &json!({
            "record_type": "compound_fts_run",
            "common": common,
            "workload_count": workloads.len(),
            "k_values": [10, 100],
            "cache_states": ["cold", "warm"],
            "dataset_root": dataset_root.path,
        }),
        &mut output,
    )?;

    for workload in &workloads {
        for location in locations_for(&registry, workload) {
            for k in [10, 100] {
                for cache_state in [CacheState::Cold, CacheState::Warm] {
                    let mut record =
                        run_point(workload, location, k, cache_state, iterations).await?;
                    let object = record
                        .as_object_mut()
                        .expect("benchmark result is a JSON object");
                    object.insert("run_id".to_string(), json!(run_id));
                    object.insert("run_label".to_string(), json!(args.run_label));
                    object.insert("dataset_fingerprint".to_string(), fingerprint.clone());
                    object.insert("machine".to_string(), machine.clone());
                    object.insert("build".to_string(), build.clone());
                    object.insert("speedup_claimed".to_string(), json!(false));
                    write_record(&record, &mut output)?;
                }
            }
        }
    }
    Ok(())
}
