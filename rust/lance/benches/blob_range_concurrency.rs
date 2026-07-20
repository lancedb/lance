// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stdout)]

use std::fmt::{Debug, Display, Formatter};
use std::fs;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow_array::{RecordBatch, RecordBatchIterator, UInt64Array};
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use bytes::Bytes;
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use futures::stream::BoxStream;
use lance::blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance::dataset::{BlobFile, Dataset, WriteParams};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::WrappingObjectStore;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
    Result as ObjectStoreResult,
};
use serde::Serialize;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const MIB: usize = 1024 * 1024;
const BLOB_COLUMN: &str = "blob";
const DEDICATED_DATASET: &str = "dedicated.lance";
const PACKED_DATASET: &str = "packed.lance";

type BenchError = Box<dyn std::error::Error + Send + Sync>;
type BenchResult<T> = Result<T, BenchError>;

#[derive(Debug, Parser)]
#[command(about = "Reproduce concurrent BlobFile range reads against object storage")]
struct Cli {
    /// Added automatically by `cargo bench` for custom harnesses.
    #[arg(long = "bench", global = true, hide = true)]
    _bench: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Write representative dedicated and packed blob datasets.
    Prepare(PrepareArgs),
    /// Run Lance and direct object-store range reads against prepared datasets.
    Run(RunArgs),
}

#[derive(Debug, Args)]
struct PrepareArgs {
    /// URI under which dedicated.lance and packed.lance are created.
    #[arg(long)]
    base_uri: String,

    /// Size of each logical blob value.
    #[arg(long, default_value_t = 500)]
    blob_size_mib: usize,

    /// Number of distinct dedicated blob objects.
    #[arg(long, default_value_t = 4)]
    dedicated_rows: usize,

    /// Number of values in the shared packed blob object.
    #[arg(long, default_value_t = 4)]
    packed_rows: usize,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// URI containing dedicated.lance and packed.lance.
    #[arg(long)]
    base_uri: String,

    /// Git revision of the Lance implementation under test.
    #[arg(long)]
    revision: String,

    /// Human-readable result label, such as before or after.
    #[arg(long)]
    label: String,

    /// EC2 instance ID used for the run.
    #[arg(long)]
    instance_id: String,

    /// EC2 instance type used for the run.
    #[arg(long)]
    instance_type: String,

    /// AWS region containing both the instance and S3 data.
    #[arg(long)]
    region: String,

    /// Comma-separated caller concurrency levels.
    #[arg(long, value_delimiter = ',', default_value = "1,4,16,32,64")]
    concurrencies: Vec<usize>,

    /// Number of samples launched per concurrency slot.
    #[arg(long, default_value_t = 16)]
    samples_per_worker: usize,

    /// Logical range size per sample.
    #[arg(long, default_value_t = 100 * 1024)]
    window_bytes: usize,

    /// Delay between launches, used to model naturally staggered callers.
    #[arg(long, default_value_t = 100)]
    stagger_micros: u64,

    /// Local JSON result path. The file is replaced atomically after validation.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug)]
enum Backend {
    Lance,
    DirectS3,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Lance => "lance",
            Self::DirectS3 => "direct_s3",
        }
    }
}

#[derive(Clone)]
struct Workload {
    name: &'static str,
    handles: Vec<Arc<BlobFile>>,
    object_store: Arc<dyn ObjectStore>,
}

#[derive(Debug, Default, Clone)]
struct RequestMetrics(Arc<RequestMetricsInner>);

#[derive(Debug, Default)]
struct RequestMetricsInner {
    gets: AtomicU64,
    heads: AtomicU64,
    physical_bytes: AtomicU64,
    active_gets: AtomicU64,
    peak_active_gets: AtomicU64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct RequestMetricsSnapshot {
    get_count: u64,
    head_count: u64,
    physical_bytes: u64,
    peak_in_flight_s3: u64,
}

impl RequestMetrics {
    fn reset(&self) {
        assert_eq!(
            self.0.active_gets.load(Ordering::Acquire),
            0,
            "cannot reset metrics while a GET is active"
        );
        self.0.gets.store(0, Ordering::Release);
        self.0.heads.store(0, Ordering::Release);
        self.0.physical_bytes.store(0, Ordering::Release);
        self.0.peak_active_gets.store(0, Ordering::Release);
    }

    fn snapshot(&self) -> RequestMetricsSnapshot {
        RequestMetricsSnapshot {
            get_count: self.0.gets.load(Ordering::Acquire),
            head_count: self.0.heads.load(Ordering::Acquire),
            physical_bytes: self.0.physical_bytes.load(Ordering::Acquire),
            peak_in_flight_s3: self.0.peak_active_gets.load(Ordering::Acquire),
        }
    }

    fn start_gets(&self, count: u64) -> ActiveGetGuard {
        self.0.gets.fetch_add(count, Ordering::AcqRel);
        let active = self.0.active_gets.fetch_add(count, Ordering::AcqRel) + count;
        self.0.peak_active_gets.fetch_max(active, Ordering::AcqRel);
        ActiveGetGuard {
            metrics: self.clone(),
            count,
        }
    }
}

impl WrappingObjectStore for RequestMetrics {
    fn wrap(&self, _store_prefix: &str, target: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(MeasuredObjectStore {
            target,
            metrics: self.clone(),
        })
    }
}

struct ActiveGetGuard {
    metrics: RequestMetrics,
    count: u64,
}

impl Drop for ActiveGetGuard {
    fn drop(&mut self) {
        self.metrics
            .0
            .active_gets
            .fetch_sub(self.count, Ordering::AcqRel);
    }
}

#[derive(Debug)]
struct MeasuredObjectStore {
    target: Arc<dyn ObjectStore>,
    metrics: RequestMetrics,
}

impl Display for MeasuredObjectStore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MeasuredObjectStore({})", self.target)
    }
}

#[async_trait]
impl ObjectStore for MeasuredObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> ObjectStoreResult<PutResult> {
        self.target.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.target.put_multipart_opts(location, options).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> ObjectStoreResult<GetResult> {
        if options.head {
            self.metrics.0.heads.fetch_add(1, Ordering::AcqRel);
            return self.target.get_opts(location, options).await;
        }

        let _guard = self.metrics.start_gets(1);
        let result = self.target.get_opts(location, options).await;
        if let Ok(result) = &result {
            self.metrics
                .0
                .physical_bytes
                .fetch_add(result.range.end - result.range.start, Ordering::AcqRel);
        }
        result
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> ObjectStoreResult<Vec<Bytes>> {
        let count = ranges.len() as u64;
        let _guard = self.metrics.start_gets(count);
        let result = self.target.get_ranges(location, ranges).await;
        if let Ok(bytes) = &result {
            let byte_count = bytes.iter().map(|bytes| bytes.len() as u64).sum::<u64>();
            self.metrics
                .0
                .physical_bytes
                .fetch_add(byte_count, Ordering::AcqRel);
        }
        result
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, ObjectStoreResult<Path>>,
    ) -> BoxStream<'static, ObjectStoreResult<Path>> {
        self.target.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
        self.target.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
        self.target.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> ObjectStoreResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> ObjectStoreResult<()> {
        self.target.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> ObjectStoreResult<()> {
        self.target.rename_opts(from, to, options).await
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    generated_at: String,
    label: String,
    revision: String,
    base_uri: String,
    region: String,
    instance_id: String,
    instance_type: String,
    window_bytes: usize,
    samples_per_worker: usize,
    stagger_micros: u64,
    concurrencies: Vec<usize>,
    results: Vec<BenchmarkResult>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    workload: &'static str,
    backend: &'static str,
    concurrency: usize,
    samples: usize,
    elapsed_seconds: f64,
    samples_per_second: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    logical_bytes: u64,
    #[serde(flatten)]
    requests: RequestMetricsSnapshot,
}

#[tokio::main]
async fn main() -> BenchResult<()> {
    match Cli::parse().command {
        Command::Prepare(args) => prepare(args).await,
        Command::Run(args) => run(args).await,
    }
}

async fn prepare(args: PrepareArgs) -> BenchResult<()> {
    if args.blob_size_mib == 0 || args.dedicated_rows < 2 || args.packed_rows < 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "blob_size_mib must be positive; dedicated_rows must be at least two and packed_rows at least four",
        )
        .into());
    }

    let blob_size = args.blob_size_mib.checked_mul(MIB).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "blob size overflowed usize",
        )
    })?;
    let mut payload = vec![0_u8; blob_size];
    for (index, byte) in payload.iter_mut().enumerate() {
        *byte = (index % 251) as u8;
    }

    let dedicated_threshold = NonZeroUsize::new(blob_size.div_ceil(2)).unwrap();
    let dedicated_options = BlobFieldOptions::default()
        .with_inline_size_threshold(0)
        .with_dedicated_size_threshold(dedicated_threshold);
    write_dataset(
        &dataset_uri(&args.base_uri, DEDICATED_DATASET),
        args.dedicated_rows,
        &payload,
        dedicated_options,
        None,
    )
    .await?;

    let packed_threshold = blob_size.checked_mul(2).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "packed threshold overflowed usize",
        )
    })?;
    let packed_options = BlobFieldOptions::default()
        .with_inline_size_threshold(0)
        .with_dedicated_size_threshold(NonZeroUsize::new(packed_threshold).unwrap());
    write_dataset(
        &dataset_uri(&args.base_uri, PACKED_DATASET),
        args.packed_rows,
        &payload,
        packed_options,
        Some(packed_threshold),
    )
    .await?;

    println!(
        "prepared dedicated_rows={} packed_rows={} blob_size_bytes={} under {}",
        args.dedicated_rows, args.packed_rows, blob_size, args.base_uri
    );
    Ok(())
}

async fn write_dataset(
    uri: &str,
    row_count: usize,
    payload: &[u8],
    options: BlobFieldOptions,
    blob_pack_file_size_threshold: Option<usize>,
) -> BenchResult<()> {
    let schema = Arc::new(ArrowSchema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::UInt64, false),
        blob_field_with_options(BLOB_COLUMN, false, options),
    ]));
    let ids = Arc::new(UInt64Array::from_iter_values(0..row_count as u64));
    let mut blobs = BlobArrayBuilder::new(row_count);
    for _ in 0..row_count {
        blobs.push_bytes(payload)?;
    }
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, blobs.finish()?])?;
    let reader = RecordBatchIterator::new([Ok(batch)], schema);
    let params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_3),
        max_rows_per_file: row_count,
        max_rows_per_group: row_count,
        blob_pack_file_size_threshold,
        ..Default::default()
    };
    Dataset::write(reader, uri, Some(params)).await?;
    Ok(())
}

async fn run(args: RunArgs) -> BenchResult<()> {
    validate_run_args(&args)?;

    let metrics = RequestMetrics::default();
    let workloads = load_workloads(&args.base_uri, metrics.clone()).await?;
    let mut results = Vec::new();

    for workload in &workloads {
        for backend in [Backend::DirectS3, Backend::Lance] {
            for &concurrency in &args.concurrencies {
                prewarm(workload).await?;
                metrics.reset();
                let result = run_one(
                    workload.clone(),
                    backend,
                    concurrency,
                    args.samples_per_worker,
                    args.window_bytes,
                    args.stagger_micros,
                    &metrics,
                )
                .await?;
                println!(
                    "workload={} backend={} concurrency={} samples_per_second={:.2} p50_ms={:.3} p95_ms={:.3} gets={} heads={} peak={}",
                    result.workload,
                    result.backend,
                    result.concurrency,
                    result.samples_per_second,
                    result.latency_p50_ms,
                    result.latency_p95_ms,
                    result.requests.get_count,
                    result.requests.head_count,
                    result.requests.peak_in_flight_s3,
                );
                results.push(result);
            }
        }
    }

    let report = BenchmarkReport {
        schema_version: 1,
        generated_at: Utc::now().to_rfc3339(),
        label: args.label,
        revision: args.revision,
        base_uri: args.base_uri,
        region: args.region,
        instance_id: args.instance_id,
        instance_type: args.instance_type,
        window_bytes: args.window_bytes,
        samples_per_worker: args.samples_per_worker,
        stagger_micros: args.stagger_micros,
        concurrencies: args.concurrencies,
        results,
    };
    write_json_atomically(&args.output, &report)?;
    Ok(())
}

fn validate_run_args(args: &RunArgs) -> BenchResult<()> {
    if args.concurrencies.is_empty()
        || args.concurrencies.contains(&0)
        || args.samples_per_worker == 0
        || args.window_bytes == 0
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "concurrencies, samples_per_worker, and window_bytes must be positive",
        )
        .into());
    }
    Ok(())
}

async fn load_workloads(base_uri: &str, metrics: RequestMetrics) -> BenchResult<Vec<Workload>> {
    let dedicated = Dataset::open(&dataset_uri(base_uri, DEDICATED_DATASET)).await?;
    let dedicated =
        Arc::new(dedicated.with_object_store_wrappers([
            Arc::new(metrics.clone()) as Arc<dyn WrappingObjectStore>
        ]));
    let dedicated_rows = dedicated.count_rows(None).await? as usize;
    let dedicated_indices = (0..dedicated_rows as u64).collect::<Vec<_>>();
    let dedicated_handles = dedicated
        .take_blobs_by_indices(&dedicated_indices, BLOB_COLUMN)
        .await?
        .into_iter()
        .map(Arc::new)
        .collect::<Vec<_>>();
    if dedicated_handles.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "dedicated dataset must contain at least two blob values",
        )
        .into());
    }
    if dedicated_handles[0].data_path() == dedicated_handles[1].data_path() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "dedicated benchmark values unexpectedly share one physical source",
        )
        .into());
    }
    let dedicated_store = dedicated.object_store(None).await?.inner.clone();

    let packed = Dataset::open(&dataset_uri(base_uri, PACKED_DATASET)).await?;
    let packed =
        Arc::new(packed.with_object_store_wrappers([
            Arc::new(metrics.clone()) as Arc<dyn WrappingObjectStore>
        ]));
    let packed_rows = packed.count_rows(None).await? as usize;
    let packed_indices = (0..packed_rows as u64).collect::<Vec<_>>();
    let packed_handles = packed
        .take_blobs_by_indices(&packed_indices, BLOB_COLUMN)
        .await?
        .into_iter()
        .map(Arc::new)
        .collect::<Vec<_>>();
    if packed_handles.len() < 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "packed dataset must contain at least four blob values",
        )
        .into());
    }
    let first_packed_path = packed_handles[0].data_path().clone();
    let packed_same_source = packed_handles
        .iter()
        .filter(|handle| handle.data_path() == &first_packed_path)
        .cloned()
        .collect::<Vec<_>>();
    if packed_same_source.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "packed benchmark did not place at least two values in one physical source",
        )
        .into());
    }
    let mut packed_source_paths = Vec::new();
    let mut packed_multiple_sources = Vec::new();
    for handle in &packed_handles {
        if !packed_source_paths
            .iter()
            .any(|path| path == handle.data_path())
        {
            packed_source_paths.push(handle.data_path().clone());
            packed_multiple_sources.push(handle.clone());
        }
    }
    if packed_multiple_sources.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "packed benchmark values did not resolve to multiple physical sources",
        )
        .into());
    }
    let packed_store = packed.object_store(None).await?.inner.clone();

    Ok(vec![
        Workload {
            name: "dedicated_same_source",
            handles: vec![dedicated_handles[0].clone()],
            object_store: dedicated_store.clone(),
        },
        Workload {
            name: "dedicated_multiple_sources",
            handles: dedicated_handles,
            object_store: dedicated_store,
        },
        Workload {
            name: "packed_same_source",
            handles: packed_same_source,
            object_store: packed_store.clone(),
        },
        Workload {
            name: "packed_multiple_sources",
            handles: packed_multiple_sources,
            object_store: packed_store,
        },
    ])
}

async fn prewarm(workload: &Workload) -> BenchResult<()> {
    for handle in &workload.handles {
        handle.read_range(0..1).await?;
        let physical = handle.position()..handle.position() + 1;
        let bytes = workload
            .object_store
            .get_ranges(handle.data_path(), &[physical])
            .await?;
        if bytes.first().is_none_or(Bytes::is_empty) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "prewarm range returned no bytes",
            )
            .into());
        }
    }
    Ok(())
}

async fn run_one(
    workload: Workload,
    backend: Backend,
    concurrency: usize,
    samples_per_worker: usize,
    window_bytes: usize,
    stagger_micros: u64,
    metrics: &RequestMetrics,
) -> BenchResult<BenchmarkResult> {
    for handle in &workload.handles {
        if window_bytes as u64 > handle.size() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "window size {} exceeds blob size {} for {}",
                    window_bytes,
                    handle.size(),
                    workload.name
                ),
            )
            .into());
        }
    }

    let sample_count = concurrency.checked_mul(samples_per_worker).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sample count overflowed usize",
        )
    })?;
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut tasks = JoinSet::<BenchResult<f64>>::new();
    let started = Instant::now();

    for sample_idx in 0..sample_count {
        let permit = semaphore.clone().acquire_owned().await?;
        if sample_idx != 0 && stagger_micros != 0 {
            tokio::time::sleep(Duration::from_micros(stagger_micros)).await;
        }
        let handle = workload.handles[sample_idx % workload.handles.len()].clone();
        let object_store = workload.object_store.clone();
        let logical_range = sample_range(sample_idx, handle.size(), window_bytes as u64);
        tasks.spawn(async move {
            let request_started = Instant::now();
            let bytes = match backend {
                Backend::Lance => handle.read_range(logical_range.clone()).await?,
                Backend::DirectS3 => {
                    let physical_range = (handle.position() + logical_range.start)
                        ..(handle.position() + logical_range.end);
                    object_store
                        .get_ranges(handle.data_path(), &[physical_range])
                        .await?
                        .into_iter()
                        .next()
                        .ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "direct S3 range returned no buffer",
                            )
                        })?
                }
            };
            let expected_len = (logical_range.end - logical_range.start) as usize;
            if bytes.len() != expected_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "range returned {} bytes, expected {}",
                        bytes.len(),
                        expected_len
                    ),
                )
                .into());
            }
            drop(permit);
            Ok(request_started.elapsed().as_secs_f64() * 1000.0)
        });
    }

    let mut latencies_ms = Vec::with_capacity(sample_count);
    while let Some(result) = tasks.join_next().await {
        latencies_ms.push(result??);
    }
    let elapsed = started.elapsed();
    latencies_ms.sort_by(f64::total_cmp);
    let request_metrics = metrics.snapshot();

    Ok(BenchmarkResult {
        workload: workload.name,
        backend: backend.as_str(),
        concurrency,
        samples: sample_count,
        elapsed_seconds: elapsed.as_secs_f64(),
        samples_per_second: sample_count as f64 / elapsed.as_secs_f64(),
        latency_p50_ms: percentile(&latencies_ms, 0.50),
        latency_p95_ms: percentile(&latencies_ms, 0.95),
        logical_bytes: (sample_count as u64) * (window_bytes as u64),
        requests: request_metrics,
    })
}

fn sample_range(sample_idx: usize, blob_size: u64, window_bytes: u64) -> Range<u64> {
    let max_start = blob_size - window_bytes;
    let mixed = (sample_idx as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xD1B5_4A32_D192_ED03);
    let start = if max_start == 0 {
        0
    } else {
        mixed % (max_start + 1)
    };
    start..start + window_bytes
}

fn percentile(sorted_values: &[f64], quantile: f64) -> f64 {
    let rank = (quantile * sorted_values.len() as f64).ceil() as usize;
    sorted_values[rank.saturating_sub(1)]
}

fn dataset_uri(base_uri: &str, dataset: &str) -> String {
    format!("{}/{}", base_uri.trim_end_matches('/'), dataset)
}

fn write_json_atomically(path: &FsPath, report: &BenchmarkReport) -> BenchResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    let _: serde_json::Value = serde_json::from_slice(&bytes)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, &bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}
