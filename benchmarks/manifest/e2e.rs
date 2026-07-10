// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Process-isolated end-to-end benchmark for protobuf and columnar manifests.

use std::fmt::{Debug, Formatter};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use clap::{Parser, ValueEnum};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, ReadParams};
use lance_core::datatypes::Schema;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::metrics::METRIC_REQUESTS;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_io::utils::tracking_store::{IOTracker, IoStats};
use lance_table::format::{
    DataFile, DeletionFile, DeletionFileType, Fragment, IndexMetadata, Manifest,
    Transaction as TableTransaction,
};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    commit_handler_from_url,
};
use lance_table::io::manifest::is_columnar_manifest_footer;
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use object_store::path::Path as ObjectPath;
use serde::{Deserialize, Serialize};
use tokio::sync::Barrier;

const SCHEMA_VERSION: u64 = 2;
const DEFAULT_SEED: u64 = 0x4c41_4e43_455f_4d46;
const ROWS_PER_FRAGMENT: usize = 1_024;
const WORKER_JOB_ENV: &str = "LANCE_MANIFEST_E2E_JOB";
const REQUIRED_FRAGMENT_SIZES: [usize; 4] = [1_000, 100_000, 1_000_000, 10_000_000];

type BenchError = Box<dyn std::error::Error + Send + Sync>;
type BenchResult<T> = Result<T, BenchError>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Scenario {
    S1,
    S2,
}

impl Scenario {
    fn as_str(self) -> &'static str {
        match self {
            Self::S1 => "S1",
            Self::S2 => "S2",
        }
    }

    fn path_component(self) -> &'static str {
        match self {
            Self::S1 => "s1",
            Self::S2 => "s2",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum ManifestFormat {
    Protobuf,
    Lance,
}

impl ManifestFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Protobuf => "protobuf",
            Self::Lance => "lance",
        }
    }

    fn storage_version(self) -> LanceFileVersion {
        match self {
            Self::Protobuf => LanceFileVersion::V2_2,
            Self::Lance => LanceFileVersion::V2_3,
        }
    }

    fn expects_columnar(self) -> bool {
        self == Self::Lance
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Storage {
    Ebs,
    S3,
}

impl Storage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ebs => "ebs",
            Self::S3 => "s3",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum BenchmarkOperation {
    Open,
    Commit,
    ConflictRetry,
    TimeTravel,
}

impl BenchmarkOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Commit => "commit",
            Self::ConflictRetry => "conflict_retry",
            Self::TimeTravel => "time_travel",
        }
    }

    fn supports_fragments(self, fragments: usize) -> bool {
        match self {
            Self::Commit | Self::ConflictRetry => {
                matches!(fragments, 1_000_000 | 10_000_000)
            }
            Self::Open | Self::TimeTravel => true,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run process-isolated Lance manifest end-to-end benchmarks")]
struct Args {
    /// Dataset root on the local EBS volume or S3.
    #[arg(long)]
    dataset_prefix: String,

    #[arg(long, value_enum)]
    storage: Storage,

    /// Create this JSONL file. Existing files are rejected.
    #[arg(long)]
    output: PathBuf,

    #[arg(long, value_enum, value_delimiter = ',', default_value = "s1,s2")]
    scenarios: Vec<Scenario>,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1000,100000,1000000,10000000"
    )]
    fragments: Vec<usize>,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "protobuf,lance"
    )]
    formats: Vec<ManifestFormat>,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "open,commit,conflict-retry,time-travel"
    )]
    operations: Vec<BenchmarkOperation>,

    #[arg(long, default_value_t = 5)]
    rounds: usize,

    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,

    /// Verified source revision supplied by run_e2e.py.
    #[arg(long)]
    commit: String,

    /// Benchmark host identity. Defaults to HOSTNAME or `hostname`.
    #[arg(long)]
    host: Option<String>,

    /// Preserve the unique run prefix after completion.
    #[arg(long, default_value_t = false)]
    keep_data: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum WorkerKind {
    SetupCreate,
    SetupAppend,
    Measure,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WorkerJob {
    kind: WorkerKind,
    uri: String,
    scenario: Scenario,
    fragments: usize,
    format: ManifestFormat,
    storage: Storage,
    operation: BenchmarkOperation,
    round: usize,
    seed: u64,
    commit: String,
    host: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct BenchmarkRecord {
    schema_version: u64,
    suite: String,
    scenario: String,
    fragments: usize,
    format: String,
    storage: String,
    operation: String,
    round: usize,
    wall_ns: u64,
    bytes: u64,
    peak_rss_bytes: u64,
    get_requests: u64,
    put_requests: u64,
    read_bytes: u64,
    write_bytes: u64,
    status: String,
    error: Option<String>,
    commit: String,
    seed: u64,
    host: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct ObjectMetrics {
    get_requests: u64,
    put_requests: u64,
}

impl ObjectMetrics {
    fn checked_delta(self, earlier: Self) -> BenchResult<Self> {
        Ok(Self {
            get_requests: self
                .get_requests
                .checked_sub(earlier.get_requests)
                .ok_or_else(|| bench_error("GET request counter moved backwards"))?,
            put_requests: self
                .put_requests
                .checked_sub(earlier.put_requests)
                .ok_or_else(|| bench_error("PUT request counter moved backwards"))?,
        })
    }
}

#[derive(Debug)]
struct Measurement {
    wall_ns: u64,
    peak_rss_bytes: u64,
    io: IoStats,
    metrics: ObjectMetrics,
}

struct BarrierCommitHandler {
    inner: Arc<dyn CommitHandler>,
    barrier: Barrier,
    initial_calls: AtomicUsize,
    conflicts: AtomicUsize,
}

impl BarrierCommitHandler {
    fn new(inner: Arc<dyn CommitHandler>) -> Self {
        Self {
            inner,
            barrier: Barrier::new(2),
            initial_calls: AtomicUsize::new(0),
            conflicts: AtomicUsize::new(0),
        }
    }

    fn conflicts(&self) -> usize {
        self.conflicts.load(Ordering::SeqCst)
    }
}

impl Debug for BarrierCommitHandler {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BarrierCommitHandler")
            .field("inner", &self.inner)
            .field("initial_calls", &self.initial_calls.load(Ordering::SeqCst))
            .field("conflicts", &self.conflicts())
            .finish()
    }
}

#[async_trait::async_trait]
impl CommitHandler for BarrierCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &ObjectPath,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<TableTransaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let call = self.initial_calls.fetch_add(1, Ordering::SeqCst);
        if call < 2
            && tokio::time::timeout(Duration::from_secs(300), self.barrier.wait())
                .await
                .is_err()
        {
            return Err(CommitError::OtherError(lance_core::Error::timeout(
                "timed out waiting for both conflict benchmark writers".to_string(),
            )));
        }
        let result = self
            .inner
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await;
        if matches!(result, Err(CommitError::CommitConflict)) {
            self.conflicts.fetch_add(1, Ordering::SeqCst);
        }
        result
    }
}

fn bench_error(message: impl Into<String>) -> BenchError {
    std::io::Error::other(message.into()).into()
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sample(seed: u64, fragment_id: u64, stream: u64) -> u64 {
    mix64(seed ^ fragment_id.wrapping_mul(0xd6e8_feb8_6659_fd93) ^ stream)
}

fn dataset_schema(num_fields: usize) -> BenchResult<Schema> {
    let fields = (0..num_fields)
        .map(|field_id| Field::new(format!("field_{field_id}"), DataType::Int64, true))
        .collect::<Vec<_>>();
    Ok(Schema::try_from(&ArrowSchema::new(fields))?)
}

fn data_file_template(num_fields: usize, format: ManifestFormat) -> DataFile {
    let fields = (0..num_fields as i32).collect::<Vec<_>>();
    let (major, minor) = format.storage_version().to_numbers();
    DataFile::new(
        String::new(),
        fields.clone(),
        fields,
        major,
        minor,
        NonZeroU64::new(1_048_576),
        None,
    )
}

fn short_path(fragment_id: u64, ordinal: usize) -> String {
    let value = fragment_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(ordinal as u64))
        .unwrap_or(fragment_id);
    format!("data/{value:032x}.lance")
}

fn long_path(fragment_id: u64, ordinal: usize, entropy: u64) -> String {
    format!("imports/customer-{fragment_id:016x}/partition-{entropy:016x}/part-{ordinal:02}.lance")
}

fn make_file(
    template: &DataFile,
    fragment_id: u64,
    ordinal: usize,
    is_long_path: bool,
    entropy: u64,
) -> DataFile {
    let mut file = template.clone();
    file.path = if is_long_path {
        long_path(fragment_id, ordinal, entropy)
    } else {
        short_path(fragment_id, ordinal)
    };
    file.file_size_bytes = NonZeroU64::new(1_048_576 + entropy % 65_536).into();
    file
}

fn make_fragment(
    scenario: Scenario,
    id: u64,
    seed: u64,
    template_8: &DataFile,
    template_32: &DataFile,
) -> Fragment {
    let layout = sample(seed, id, 0);
    let physical_rows = ROWS_PER_FRAGMENT + layout as usize % 17;
    let mut fragment = Fragment::new(id);
    match scenario {
        Scenario::S1 => {
            fragment
                .files
                .push(make_file(template_8, id, 0, false, layout));
            fragment.physical_rows = Some(ROWS_PER_FRAGMENT);
        }
        Scenario::S2 => {
            let template = if layout & 1 == 0 {
                template_8
            } else {
                template_32
            };
            fragment
                .files
                .push(make_file(template, id, 0, layout & 2 != 0, layout));
            if sample(seed, id, 1).is_multiple_of(20) {
                fragment.files.push(make_file(
                    template_32,
                    id,
                    1,
                    sample(seed, id, 2) & 1 != 0,
                    sample(seed, id, 3),
                ));
            }
            if sample(seed, id, 4).is_multiple_of(5) {
                fragment.deletion_file = Some(DeletionFile {
                    read_version: 1,
                    id,
                    file_type: if layout & 4 == 0 {
                        DeletionFileType::Array
                    } else {
                        DeletionFileType::Bitmap
                    },
                    num_deleted_rows: Some(1 + layout as usize % 31),
                    base_id: None,
                });
            }
            fragment.physical_rows = if sample(seed, id, 5).is_multiple_of(100) {
                None
            } else {
                Some(physical_rows)
            };
        }
    }
    fragment
}

fn make_fragments(
    scenario: Scenario,
    count: usize,
    seed: u64,
    format: ManifestFormat,
) -> Vec<Fragment> {
    let template_8 = data_file_template(8, format);
    let template_32 = data_file_template(32, format);
    let mut fragments = Vec::with_capacity(count);
    for index in 0..count {
        fragments.push(make_fragment(
            scenario,
            index as u64,
            seed,
            &template_8,
            &template_32,
        ));
    }
    // Real V2.3 datasets normalize legacy row statistics before committing. The
    // synthetic benchmark has no backing data files, so perform that deterministic
    // normalization outside every measured operation.
    for fragment in &mut fragments {
        if fragment.physical_rows.is_none() {
            let layout = sample(seed, fragment.id, 0);
            fragment.physical_rows = Some(ROWS_PER_FRAGMENT + layout as usize % 17);
        }
    }
    fragments
}

fn append_transaction(
    dataset: &Dataset,
    job: &WorkerJob,
    ordinal: u64,
) -> BenchResult<Transaction> {
    let fragment_count =
        u64::try_from(job.fragments).map_err(|_| bench_error("fragment count exceeds u64"))?;
    let round = u64::try_from(job.round).map_err(|_| bench_error("round exceeds u64"))?;
    let synthetic_id = round
        .checked_mul(2)
        .and_then(|value| fragment_count.checked_add(value))
        .and_then(|value| value.checked_add(ordinal))
        .ok_or_else(|| bench_error("synthetic fragment ID overflows u64"))?;
    let template_8 = data_file_template(8, job.format);
    let template_32 = data_file_template(32, job.format);
    let mut fragment = make_fragment(
        job.scenario,
        synthetic_id,
        job.seed,
        &template_8,
        &template_32,
    );
    if fragment.physical_rows.is_none() {
        let layout = sample(job.seed, synthetic_id, 0);
        fragment.physical_rows = Some(ROWS_PER_FRAGMENT + layout as usize % 17);
    }
    fragment.id = 0;
    Ok(Transaction::new(
        dataset.manifest().version,
        Operation::Append {
            fragments: vec![fragment],
        },
        None,
    ))
}

fn store_params(tracker: Arc<IOTracker>) -> ObjectStoreParams {
    ObjectStoreParams {
        object_store_wrapper: Some(tracker),
        ..Default::default()
    }
}

fn dataset_builder(job: &WorkerJob, tracker: Arc<IOTracker>) -> DatasetBuilder {
    DatasetBuilder::from_uri(&job.uri)
        .with_index_cache_size_bytes(0)
        .with_metadata_cache_size_bytes(0)
        .with_read_params(ReadParams {
            store_options: Some(store_params(tracker)),
            ..Default::default()
        })
}

async fn verify_dataset_format(dataset: &Dataset, format: ManifestFormat) -> BenchResult<()> {
    let storage_version = dataset
        .manifest()
        .data_storage_format
        .lance_file_version()?;
    if storage_version.resolve() != format.storage_version() {
        return Err(bench_error(format!(
            "dataset storage version is {}, expected {}",
            storage_version,
            format.storage_version()
        )));
    }

    let store = dataset.object_store(None).await?;
    let reader = store.open(&dataset.manifest_location().path).await?;
    let size = reader.size().await?;
    let tail_start = size.saturating_sub(64 * 1024);
    let tail = reader.get_range(tail_start..size).await?;
    let is_columnar = is_columnar_manifest_footer(&tail)?;
    if is_columnar != format.expects_columnar() {
        return Err(bench_error(format!(
            "{} storage produced a {} manifest footer",
            format.storage_version(),
            if is_columnar { "columnar" } else { "protobuf" }
        )));
    }
    Ok(())
}

async fn create_dataset_fixture(job: &WorkerJob) -> BenchResult<Dataset> {
    let num_fields = if job.scenario == Scenario::S1 { 8 } else { 32 };
    let operation = Operation::Overwrite {
        fragments: make_fragments(job.scenario, job.fragments, job.seed, job.format),
        schema: dataset_schema(num_fields)?,
        config_upsert_values: None,
        initial_bases: None,
    };
    let tracker = Arc::new(IOTracker::default());
    let dataset = CommitBuilder::new(job.uri.as_str())
        .with_store_params(store_params(tracker))
        .with_storage_format(job.format.storage_version())
        .with_skip_auto_cleanup(true)
        .execute(Transaction::new(0, operation, None))
        .await?;
    if dataset.count_fragments() != job.fragments {
        return Err(bench_error(format!(
            "setup created {} fragments, expected {}",
            dataset.count_fragments(),
            job.fragments
        )));
    }
    verify_dataset_format(&dataset, job.format).await?;
    Ok(dataset)
}

async fn setup_create(job: &WorkerJob) -> BenchResult<()> {
    create_dataset_fixture(job).await.map(|_| ())
}

async fn setup_append(job: &WorkerJob) -> BenchResult<()> {
    let tracker = Arc::new(IOTracker::default());
    let dataset = dataset_builder(job, tracker).load().await?;
    if dataset.count_fragments() + 1 != job.fragments {
        return Err(bench_error(format!(
            "time-travel base has {} fragments, expected {} before append",
            dataset.count_fragments(),
            job.fragments - 1
        )));
    }
    let transaction = append_transaction(&dataset, job, 0)?;
    let dataset = CommitBuilder::new(Arc::new(dataset))
        .with_storage_format(job.format.storage_version())
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    if dataset.count_fragments() != job.fragments {
        return Err(bench_error(format!(
            "time-travel setup has {} fragments, expected {}",
            dataset.count_fragments(),
            job.fragments
        )));
    }
    verify_dataset_format(&dataset, job.format).await
}

fn snapshot_object_metrics(snapshotter: &Snapshotter) -> ObjectMetrics {
    let mut totals = ObjectMetrics::default();
    for (composite_key, _unit, _description, value) in snapshotter.snapshot().into_vec() {
        let DebugValue::Counter(value) = value else {
            continue;
        };
        if composite_key.key().name() != METRIC_REQUESTS {
            continue;
        }
        let operation = composite_key
            .key()
            .labels()
            .find(|label| label.key() == "operation")
            .map(|label| label.value());
        match operation {
            Some("get") => totals.get_requests += value,
            Some("put" | "put_part") => totals.put_requests += value,
            _ => {}
        }
    }
    totals
}

fn duration_ns(started: Instant) -> BenchResult<u64> {
    u64::try_from(started.elapsed().as_nanos())
        .map_err(|_| bench_error("benchmark duration exceeds u64 nanoseconds"))
}

#[cfg(target_os = "macos")]
fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage initializes the provided rusage value for the current process.
    unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as u64
        } else {
            0
        }
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage initializes the provided rusage value for the current process.
    unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            (usage.ru_maxrss as u64).saturating_mul(1_024)
        } else {
            0
        }
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

fn finish_measurement(
    job: &WorkerJob,
    wall_ns: u64,
    io: IoStats,
    metrics_before: ObjectMetrics,
    snapshotter: &Snapshotter,
) -> BenchResult<Measurement> {
    let metric_delta = snapshot_object_metrics(snapshotter).checked_delta(metrics_before)?;
    if job.storage == Storage::S3 {
        let expects_get = matches!(
            job.operation,
            BenchmarkOperation::Open
                | BenchmarkOperation::ConflictRetry
                | BenchmarkOperation::TimeTravel
        );
        let expects_put = matches!(
            job.operation,
            BenchmarkOperation::Commit | BenchmarkOperation::ConflictRetry
        );
        if expects_get && metric_delta.get_requests == 0 {
            return Err(bench_error(
                "S3 operation recorded no GET metrics; build with the metrics feature",
            ));
        }
        if expects_put && metric_delta.put_requests == 0 {
            return Err(bench_error(
                "S3 operation recorded no PUT metrics; build with the metrics feature",
            ));
        }
    }
    Ok(Measurement {
        wall_ns,
        peak_rss_bytes: peak_rss_bytes(),
        io,
        metrics: metric_delta,
    })
}

async fn measure_open(
    job: &WorkerJob,
    tracker: Arc<IOTracker>,
    snapshotter: &Snapshotter,
) -> BenchResult<Measurement> {
    let metrics_before = snapshot_object_metrics(snapshotter);
    let started = Instant::now();
    let dataset = dataset_builder(job, tracker).load().await?;
    if dataset.count_fragments() != job.fragments {
        return Err(bench_error(format!(
            "cold open found {} fragments, expected {}",
            dataset.count_fragments(),
            job.fragments
        )));
    }
    let wall_ns = duration_ns(started)?;
    let store = dataset.object_store(None).await?;
    finish_measurement(
        job,
        wall_ns,
        store.io_stats_snapshot(),
        metrics_before,
        snapshotter,
    )
}

async fn measure_commit(
    job: &WorkerJob,
    tracker: Arc<IOTracker>,
    snapshotter: &Snapshotter,
) -> BenchResult<Measurement> {
    let dataset = Arc::new(dataset_builder(job, tracker).load().await?);
    if dataset.count_fragments() != job.fragments {
        return Err(bench_error(format!(
            "commit base has {} fragments, expected {}",
            dataset.count_fragments(),
            job.fragments
        )));
    }
    let store = dataset.object_store(None).await?;
    let _ = store.io_stats_incremental();
    let transaction = append_transaction(&dataset, job, 0)?;
    let metrics_before = snapshot_object_metrics(snapshotter);
    let started = Instant::now();
    let committed = CommitBuilder::new(dataset)
        .with_storage_format(job.format.storage_version())
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    let wall_ns = duration_ns(started)?;
    if committed.count_fragments() != job.fragments + 1 {
        return Err(bench_error(format!(
            "commit produced {} fragments, expected {}",
            committed.count_fragments(),
            job.fragments + 1
        )));
    }
    let measurement = finish_measurement(
        job,
        wall_ns,
        committed.object_store(None).await?.io_stats_incremental(),
        metrics_before,
        snapshotter,
    )?;
    verify_dataset_format(&committed, job.format).await?;
    Ok(measurement)
}

async fn measure_conflict_retry(
    job: &WorkerJob,
    tracker: Arc<IOTracker>,
    snapshotter: &Snapshotter,
) -> BenchResult<Measurement> {
    let dataset = Arc::new(dataset_builder(job, tracker).load().await?);
    if dataset.count_fragments() != job.fragments {
        return Err(bench_error(format!(
            "conflict base has {} fragments, expected {}",
            dataset.count_fragments(),
            job.fragments
        )));
    }
    let base_version = dataset.manifest().version;
    let first = append_transaction(&dataset, job, 0)?;
    let second = append_transaction(&dataset, job, 1)?;
    let handler = Arc::new(BarrierCommitHandler::new(
        commit_handler_from_url(&job.uri, &None).await?,
    ));
    let store = dataset.object_store(None).await?;
    let _ = store.io_stats_incremental();
    let metrics_before = snapshot_object_metrics(snapshotter);
    let started = Instant::now();
    let first_commit = CommitBuilder::new(dataset.clone())
        .with_commit_handler(handler.clone())
        .with_storage_format(job.format.storage_version())
        .with_max_retries(2)
        .with_skip_auto_cleanup(true)
        .execute(first);
    let second_commit = CommitBuilder::new(dataset.clone())
        .with_commit_handler(handler.clone())
        .with_storage_format(job.format.storage_version())
        .with_max_retries(2)
        .with_skip_auto_cleanup(true)
        .execute(second);
    let (first_result, second_result) = tokio::join!(first_commit, second_commit);
    let wall_ns = duration_ns(started)?;
    let first_dataset = first_result?;
    let second_dataset = second_result?;
    let mut versions = [
        first_dataset.manifest().version,
        second_dataset.manifest().version,
    ];
    versions.sort_unstable();
    let mut fragment_counts = [
        first_dataset.count_fragments(),
        second_dataset.count_fragments(),
    ];
    fragment_counts.sort_unstable();
    if versions != [base_version + 1, base_version + 2]
        || handler.conflicts() != 1
        || fragment_counts != [job.fragments + 1, job.fragments + 2]
    {
        return Err(bench_error(format!(
            "conflict retry produced versions {versions:?}, fragment counts {fragment_counts:?}, and {} conditional-put conflicts; expected consecutive versions, counts {} and {}, and exactly one conflict",
            handler.conflicts(),
            job.fragments + 1,
            job.fragments + 2,
        )));
    }
    let measurement = finish_measurement(
        job,
        wall_ns,
        dataset.object_store(None).await?.io_stats_incremental(),
        metrics_before,
        snapshotter,
    )?;
    verify_dataset_format(&first_dataset, job.format).await?;
    verify_dataset_format(&second_dataset, job.format).await?;
    Ok(measurement)
}

async fn measure_time_travel(
    job: &WorkerJob,
    tracker: Arc<IOTracker>,
    snapshotter: &Snapshotter,
) -> BenchResult<Measurement> {
    let metrics_before = snapshot_object_metrics(snapshotter);
    let started = Instant::now();
    let latest = dataset_builder(job, tracker).load().await?;
    if latest.count_fragments() != job.fragments || latest.manifest().version != 2 {
        return Err(bench_error(format!(
            "time-travel latest is version {} with {} fragments; expected version 2 with {} fragments",
            latest.manifest().version,
            latest.count_fragments(),
            job.fragments
        )));
    }
    let previous = latest.checkout_version(1).await?;
    if previous.count_fragments() + 1 != job.fragments {
        return Err(bench_error(format!(
            "time-travel version 1 has {} fragments, expected {}",
            previous.count_fragments(),
            job.fragments - 1
        )));
    }
    let returned = previous.checkout_version(2).await?;
    if returned.count_fragments() != job.fragments {
        return Err(bench_error(format!(
            "time-travel return has {} fragments, expected {}",
            returned.count_fragments(),
            job.fragments
        )));
    }
    let wall_ns = duration_ns(started)?;
    finish_measurement(
        job,
        wall_ns,
        returned.object_store(None).await?.io_stats_snapshot(),
        metrics_before,
        snapshotter,
    )
}

async fn measure(
    job: &WorkerJob,
    tracker: Arc<IOTracker>,
    snapshotter: &Snapshotter,
) -> BenchResult<Measurement> {
    match job.operation {
        BenchmarkOperation::Open => measure_open(job, tracker, snapshotter).await,
        BenchmarkOperation::Commit => measure_commit(job, tracker, snapshotter).await,
        BenchmarkOperation::ConflictRetry => {
            measure_conflict_retry(job, tracker, snapshotter).await
        }
        BenchmarkOperation::TimeTravel => measure_time_travel(job, tracker, snapshotter).await,
    }
}

fn successful_record(job: &WorkerJob, measurement: Measurement) -> BenchResult<BenchmarkRecord> {
    let (get_requests, put_requests) = if job.storage == Storage::S3 {
        (
            measurement.metrics.get_requests,
            measurement.metrics.put_requests,
        )
    } else {
        (measurement.io.read_iops, measurement.io.write_iops)
    };
    let bytes = measurement
        .io
        .read_bytes
        .checked_add(measurement.io.written_bytes)
        .ok_or_else(|| bench_error("total I/O bytes overflow u64"))?;
    Ok(BenchmarkRecord {
        schema_version: SCHEMA_VERSION,
        suite: "e2e".to_string(),
        scenario: job.scenario.as_str().to_string(),
        fragments: job.fragments,
        format: job.format.as_str().to_string(),
        storage: job.storage.as_str().to_string(),
        operation: job.operation.as_str().to_string(),
        round: job.round,
        wall_ns: measurement.wall_ns,
        bytes,
        peak_rss_bytes: measurement.peak_rss_bytes,
        get_requests,
        put_requests,
        read_bytes: measurement.io.read_bytes,
        // IOTracker records the payload before forwarding put_opts, so this
        // includes the losing conditional PUT in conflict_retry.
        write_bytes: measurement.io.written_bytes,
        status: "success".to_string(),
        error: None,
        commit: job.commit.clone(),
        seed: job.seed,
        host: job.host.clone(),
    })
}

fn failed_record(job: &WorkerJob, started: Instant, error: BenchError) -> BenchmarkRecord {
    BenchmarkRecord {
        schema_version: SCHEMA_VERSION,
        suite: "e2e".to_string(),
        scenario: job.scenario.as_str().to_string(),
        fragments: job.fragments,
        format: job.format.as_str().to_string(),
        storage: job.storage.as_str().to_string(),
        operation: job.operation.as_str().to_string(),
        round: job.round,
        wall_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        bytes: 0,
        peak_rss_bytes: peak_rss_bytes(),
        get_requests: 0,
        put_requests: 0,
        read_bytes: 0,
        write_bytes: 0,
        status: "error".to_string(),
        error: Some(error.to_string()),
        commit: job.commit.clone(),
        seed: job.seed,
        host: job.host.clone(),
    }
}

fn worker_entry(job: WorkerJob) -> BenchResult<()> {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder
        .install()
        .map_err(|error| bench_error(format!("failed to install metrics recorder: {error}")))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    match job.kind {
        WorkerKind::SetupCreate => runtime.block_on(setup_create(&job)),
        WorkerKind::SetupAppend => runtime.block_on(setup_append(&job)),
        WorkerKind::Measure => {
            let tracker = Arc::new(IOTracker::default());
            let started = Instant::now();
            let record = match runtime.block_on(measure(&job, tracker, &snapshotter)) {
                Ok(measurement) => successful_record(&job, measurement)?,
                Err(error) => failed_record(&job, started, error),
            };
            let mut stdout = std::io::stdout().lock();
            serde_json::to_writer(&mut stdout, &record)?;
            stdout.write_all(b"\n")?;
            Ok(())
        }
    }
}

fn resolve_host(explicit: Option<String>) -> BenchResult<String> {
    if let Some(host) = explicit {
        if host.trim().is_empty() {
            return Err(bench_error("--host must not be empty"));
        }
        return Ok(host);
    }
    if let Ok(host) = std::env::var("HOSTNAME")
        && !host.trim().is_empty()
    {
        return Ok(host);
    }
    let output = Command::new("hostname").stderr(Stdio::piped()).output()?;
    if !output.status.success() {
        return Err(bench_error(format!(
            "hostname failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let host = String::from_utf8(output.stdout)?.trim().to_string();
    if host.is_empty() {
        return Err(bench_error("hostname returned an empty value"));
    }
    Ok(host)
}

fn join_uri(prefix: &str, suffix: &str) -> String {
    format!(
        "{}/{}",
        prefix.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

fn dataset_uri(run_prefix: &str, job: &WorkerJob, include_round: bool) -> String {
    let suffix = format!(
        "{}/{}/{}/{}/{}",
        job.scenario.path_component(),
        job.fragments,
        job.format.as_str(),
        job.operation.as_str(),
        if include_round {
            format!("round-{}", job.round)
        } else {
            "shared".to_string()
        }
    );
    join_uri(run_prefix, &suffix)
}

fn spawn_worker(job: &WorkerJob, expects_record: bool) -> BenchResult<Option<BenchmarkRecord>> {
    let executable = std::env::current_exe()?;
    let output = Command::new(executable)
        .env(WORKER_JOB_ENV, serde_json::to_string(job)?)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        return Err(bench_error(format!(
            "{:?} worker failed for {}: {}",
            job.kind,
            job.uri,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    if !expects_record {
        if !String::from_utf8_lossy(&output.stdout).trim().is_empty() {
            return Err(bench_error(format!(
                "setup worker emitted unexpected stdout: {}",
                String::from_utf8_lossy(&output.stdout).trim()
            )));
        }
        return Ok(None);
    }
    let stdout = String::from_utf8(output.stdout)?;
    let mut lines = stdout.lines().filter(|line| !line.trim().is_empty());
    let line = lines
        .next()
        .ok_or_else(|| bench_error("measurement worker emitted no JSONL record"))?;
    if lines.next().is_some() {
        return Err(bench_error(
            "measurement worker emitted more than one JSONL record",
        ));
    }
    let record: BenchmarkRecord = serde_json::from_str(line)?;
    validate_worker_record(job, &record)?;
    Ok(Some(record))
}

fn validate_worker_record(job: &WorkerJob, record: &BenchmarkRecord) -> BenchResult<()> {
    let dimensions_match = record.schema_version == SCHEMA_VERSION
        && record.suite == "e2e"
        && record.scenario == job.scenario.as_str()
        && record.fragments == job.fragments
        && record.format == job.format.as_str()
        && record.storage == job.storage.as_str()
        && record.operation == job.operation.as_str()
        && record.round == job.round
        && record.commit == job.commit
        && record.seed == job.seed
        && record.host == job.host;
    if !dimensions_match {
        return Err(bench_error(format!(
            "worker JSONL dimensions do not match job: record={record:?}, job={job:?}"
        )));
    }
    if !matches!(record.status.as_str(), "success" | "error") {
        return Err(bench_error(format!(
            "worker emitted unsupported status '{}'",
            record.status
        )));
    }
    Ok(())
}

fn validate_args(args: &Args) -> BenchResult<()> {
    if args.commit.trim().is_empty() {
        return Err(bench_error("--commit must not be empty"));
    }
    if args.rounds < 5 {
        return Err(bench_error("--rounds must be at least 5"));
    }
    if args.scenarios.is_empty()
        || args.fragments.is_empty()
        || args.formats.is_empty()
        || args.operations.is_empty()
    {
        return Err(bench_error(
            "scenario, fragment, format, and operation filters must not be empty",
        ));
    }
    for fragments in &args.fragments {
        if !REQUIRED_FRAGMENT_SIZES.contains(fragments) {
            return Err(bench_error(format!(
                "unsupported fragment count {fragments}; expected one of {REQUIRED_FRAGMENT_SIZES:?}"
            )));
        }
    }
    match args.storage {
        Storage::Ebs if args.dataset_prefix.starts_with("s3") => {
            Err(bench_error("--storage ebs requires a local dataset prefix"))
        }
        Storage::S3 if !args.dataset_prefix.starts_with("s3://") => {
            Err(bench_error("--storage s3 requires an s3:// dataset prefix"))
        }
        _ => Ok(()),
    }
}

fn setup_case(job: &WorkerJob) -> BenchResult<()> {
    match job.operation {
        BenchmarkOperation::Open => {
            let mut setup = job.clone();
            setup.kind = WorkerKind::SetupCreate;
            spawn_worker(&setup, false)?;
        }
        BenchmarkOperation::TimeTravel => {
            let mut create = job.clone();
            create.kind = WorkerKind::SetupCreate;
            create.fragments = job
                .fragments
                .checked_sub(1)
                .ok_or_else(|| bench_error("time-travel fragment count underflow"))?;
            spawn_worker(&create, false)?;
            let mut append = job.clone();
            append.kind = WorkerKind::SetupAppend;
            spawn_worker(&append, false)?;
        }
        BenchmarkOperation::Commit | BenchmarkOperation::ConflictRetry => {
            let mut setup = job.clone();
            setup.kind = WorkerKind::SetupCreate;
            spawn_worker(&setup, false)?;
        }
    }
    Ok(())
}

fn run_matrix(args: &Args, run_prefix: &str, commit: &str, host: &str) -> BenchResult<()> {
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let mut output = BufWriter::new(output);
    let mut records = 0_usize;
    let mut failed_records = 0_usize;

    for scenario in &args.scenarios {
        for fragments in &args.fragments {
            for format in &args.formats {
                for operation in &args.operations {
                    if !operation.supports_fragments(*fragments) {
                        continue;
                    }
                    let include_round = matches!(
                        operation,
                        BenchmarkOperation::Commit | BenchmarkOperation::ConflictRetry
                    );
                    for round in 0..args.rounds {
                        let mut job = WorkerJob {
                            kind: WorkerKind::Measure,
                            uri: String::new(),
                            scenario: *scenario,
                            fragments: *fragments,
                            format: *format,
                            storage: args.storage,
                            operation: *operation,
                            round,
                            seed: args.seed,
                            commit: commit.to_string(),
                            host: host.to_string(),
                        };
                        job.uri = dataset_uri(run_prefix, &job, include_round);
                        if round == 0 || include_round {
                            setup_case(&job)?;
                        }
                        let record = spawn_worker(&job, true)?
                            .ok_or_else(|| bench_error("measurement worker returned no record"))?;
                        if record.status != "success" {
                            failed_records += 1;
                        }
                        serde_json::to_writer(&mut output, &record)?;
                        output.write_all(b"\n")?;
                        output.flush()?;
                        records += 1;
                    }
                }
            }
        }
    }
    if records == 0 {
        return Err(bench_error(
            "the selected filters produce no benchmark cases",
        ));
    }
    if failed_records > 0 {
        return Err(bench_error(format!(
            "{failed_records} of {records} benchmark records failed; inspect {}",
            args.output.display()
        )));
    }
    Ok(())
}

async fn cleanup_run_prefix(run_prefix: &str) -> BenchResult<()> {
    let (store, path) = ObjectStore::from_uri(run_prefix).await?;
    store.remove_dir_all(path).await?;
    Ok(())
}

fn controller_entry(args: Args) -> BenchResult<()> {
    validate_args(&args)?;
    let commit = args.commit.clone();
    let host = resolve_host(args.host.clone())?;
    let run_id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run_prefix = join_uri(
        &args.dataset_prefix,
        &format!("manifest-e2e-run-{}-{run_id}", std::process::id()),
    );
    let run_result = run_matrix(&args, &run_prefix, &commit, &host);
    let cleanup_result = if args.keep_data {
        Ok(())
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(cleanup_run_prefix(&run_prefix))
    };
    match (run_result, cleanup_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(run_error), Ok(())) => Err(run_error),
        (Ok(()), Err(cleanup_error)) => Err(cleanup_error),
        (Err(run_error), Err(cleanup_error)) => Err(bench_error(format!(
            "benchmark failed: {run_error}; cleanup also failed: {cleanup_error}"
        ))),
    }
}

fn main() -> BenchResult<()> {
    if let Ok(serialized_job) = std::env::var(WORKER_JOB_ENV) {
        let job: WorkerJob = serde_json::from_str(&serialized_job)?;
        worker_entry(job)
    } else {
        controller_entry(Args::parse())
    }
}

#[cfg(test)]
mod tests {
    // Cargo also checks this source as a harness-free benchmark, where test
    // imports are considered unused even though the test target uses them.
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn short_paths_are_43_bytes() {
        assert_eq!(short_path(42, 0).len(), 43);
    }

    #[test]
    fn storage_versions_select_manifest_formats() {
        assert_eq!(
            ManifestFormat::Protobuf.storage_version(),
            LanceFileVersion::V2_2
        );
        assert_eq!(
            ManifestFormat::Lance.storage_version(),
            LanceFileVersion::V2_3
        );
    }

    #[test]
    fn commit_cases_only_use_large_fragment_counts() {
        assert!(!BenchmarkOperation::Commit.supports_fragments(100_000));
        assert!(BenchmarkOperation::Commit.supports_fragments(1_000_000));
        assert!(BenchmarkOperation::ConflictRetry.supports_fragments(10_000_000));
    }

    async fn manifest_size_and_cold_open_io(
        scenario: Scenario,
        format: ManifestFormat,
    ) -> (u64, IoStats) {
        let temporary = tempfile::tempdir().unwrap();
        let job = WorkerJob {
            kind: WorkerKind::Measure,
            uri: temporary
                .path()
                .join(format!("dataset-{}", format.as_str()))
                .to_string_lossy()
                .into_owned(),
            scenario,
            fragments: 1_000,
            format,
            storage: Storage::Ebs,
            operation: BenchmarkOperation::Open,
            round: 0,
            seed: DEFAULT_SEED,
            commit: "test".to_string(),
            host: "test".to_string(),
        };

        let created = create_dataset_fixture(&job).await.unwrap();
        let manifest_reader = created
            .object_store(None)
            .await
            .unwrap()
            .open(&created.manifest_location().path)
            .await
            .unwrap();
        let manifest_size = u64::try_from(manifest_reader.size().await.unwrap()).unwrap();
        drop(created);

        let cold = dataset_builder(&job, Arc::new(IOTracker::default()))
            .load()
            .await
            .unwrap();
        assert_eq!(cold.count_fragments(), job.fragments);
        assert_eq!(
            cold.manifest()
                .data_storage_format
                .lance_file_version()
                .unwrap()
                .resolve(),
            format.storage_version()
        );
        let io = cold.object_store(None).await.unwrap().io_stats_snapshot();
        (manifest_size, io)
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn columnar_overwrite_manifest_is_smaller_and_no_more_expensive_to_open() {
        for scenario in [Scenario::S1, Scenario::S2] {
            let (protobuf_size, protobuf_io) =
                manifest_size_and_cold_open_io(scenario, ManifestFormat::Protobuf).await;
            let (lance_size, lance_io) =
                manifest_size_and_cold_open_io(scenario, ManifestFormat::Lance).await;

            assert!(
                lance_size < protobuf_size,
                "{scenario:?}: Lance manifest size {lance_size} must be smaller than protobuf {protobuf_size}"
            );
            assert_eq!(
                lance_io.read_bytes, lance_size,
                "{scenario:?}: a cold open should read the single Lance manifest exactly once"
            );
            assert!(
                lance_io.read_iops <= protobuf_io.read_iops,
                "{scenario:?}: Lance cold open used {} reads, protobuf used {}",
                lance_io.read_iops,
                protobuf_io.read_iops
            );
        }
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn conflict_retry_tracks_the_failed_conditional_put() {
        for format in [ManifestFormat::Protobuf, ManifestFormat::Lance] {
            let temporary = tempfile::tempdir().unwrap();
            let job = WorkerJob {
                kind: WorkerKind::Measure,
                uri: temporary
                    .path()
                    .join("dataset")
                    .to_string_lossy()
                    .into_owned(),
                scenario: Scenario::S1,
                fragments: 16,
                format,
                storage: Storage::Ebs,
                operation: BenchmarkOperation::ConflictRetry,
                round: 0,
                seed: DEFAULT_SEED,
                commit: "test".to_string(),
                host: "test".to_string(),
            };
            setup_create(&job).await.unwrap();

            let recorder = DebuggingRecorder::new();
            let measurement = measure_conflict_retry(
                &job,
                Arc::new(IOTracker::default()),
                &recorder.snapshotter(),
            )
            .await
            .unwrap();
            let manifest_puts = measurement
                .io
                .requests
                .iter()
                .filter(|request| {
                    request.method == "put_opts" && request.path.to_string().ends_with(".manifest")
                })
                .map(|request| request.path.to_string())
                .collect::<Vec<_>>();
            assert_eq!(manifest_puts.len(), 3, "{format:?}: {manifest_puts:?}");

            let mut unique_paths = manifest_puts.clone();
            unique_paths.sort();
            unique_paths.dedup();
            assert_eq!(unique_paths.len(), 2, "{format:?}: {manifest_puts:?}");
            assert!(
                manifest_puts.iter().any(|path| manifest_puts
                    .iter()
                    .filter(|other| *other == path)
                    .count()
                    == 2),
                "{format:?}: the failed conditional PUT must be recorded before retry"
            );
            assert!(measurement.io.written_bytes > 0);
        }
    }
}
