// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Process-isolated stable-row-address benchmark worker.
//!
//! The Python runner invokes this executable once per measured operation.  The
//! worker emits exactly one strict JSON record on stdout. Logical I/O comes from
//! the path-aware object-store tracker. Actual attempts come only from the native
//! cloud HTTP retry boundary and remain null for non-HTTP stores.

#![allow(clippy::print_stdout)]

mod evidence;
mod maintenance;
mod measurement;

use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchReader, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use clap::{Parser, ValueEnum};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::index::DatasetIndexRemapperOptions;
use lance::dataset::optimize::{
    CompactionMetrics, CompactionOptions, CompactionPlan, IndexRemapper, IndexRemapperOptions,
    RemappedIndex, RowAddressMaintenanceOptions, TaskData, commit_compaction,
    execute_compaction_plan_tasks, maintain_row_addresses, plan_compaction, rewrite_files_in_order,
};
use lance::dataset::scanner::ColumnOrdering;
use lance::dataset::{
    MergeInsertBuilder, NewColumnTransform, ReadParams, UpdateBuilder, WhenMatched, WhenNotMatched,
    WriteMode, WriteParams,
};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_core::ROW_ID;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::ScalarIndexParams;
use lance_io::object_store::ObjectStoreParams;
use lance_io::object_store::metrics::METRIC_HTTP_ATTEMPTS;
use lance_io::utils::tracking_store::{IOTracker, IoOperation, IoStats};
use lance_linalg::distance::MetricType;
use lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
use lance_table::format::{Fragment, PlacementMaintenanceRequired, pb};
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use prost::Message;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: u64 = 1;

type BenchError = Box<dyn std::error::Error + Send + Sync>;
type BenchResult<T> = Result<T, BenchError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchFormat {
    V22NoStable,
    V22Stable,
    V23Logical,
}

impl BenchFormat {
    fn name(self) -> &'static str {
        match self {
            Self::V22NoStable => "v22_no_stable",
            Self::V22Stable => "v22_stable",
            Self::V23Logical => "v23_logical",
        }
    }

    fn storage_version(self) -> LanceFileVersion {
        match self {
            Self::V22NoStable | Self::V22Stable => LanceFileVersion::V2_2,
            Self::V23Logical => LanceFileVersion::V2_3,
        }
    }

    fn enable_legacy_stable_row_ids(self) -> bool {
        self == Self::V22Stable
    }

    fn validate_dataset_header(self, dataset: &Dataset) -> BenchResult<()> {
        let manifest = dataset.manifest();
        let actual_version = manifest.data_storage_format.lance_file_version()?;
        let uses_legacy_stable_row_ids =
            (manifest.reader_feature_flags | manifest.writer_feature_flags) & FLAG_STABLE_ROW_IDS
                != 0;
        if actual_version != self.storage_version() {
            return Err(format!(
                "dataset format mismatch: requested={}, expected_version={}, actual_version={}",
                self.name(),
                self.storage_version(),
                actual_version
            )
            .into());
        }

        match self {
            Self::V22NoStable if uses_legacy_stable_row_ids => {
                Err("v22_no_stable unexpectedly set the legacy stable-row-id flag".into())
            }
            Self::V22Stable if !uses_legacy_stable_row_ids => {
                Err("v22_stable did not set the legacy stable-row-id flag".into())
            }
            Self::V23Logical => {
                if uses_legacy_stable_row_ids {
                    return Err("v23_logical unexpectedly set the legacy stable-row-id flag".into());
                }
                if !manifest.uses_stable_logical_row_addresses() {
                    return Err("v23_logical is missing the logical row-address contract".into());
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn validate_dataset(self, dataset: &Dataset) -> BenchResult<()> {
        self.validate_dataset_header(dataset)?;
        if self == Self::V23Logical {
            dataset.manifest().validate_row_address_contract()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Storage {
    Ebs,
    S3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum RunMode {
    Smoke,
    Release,
}

impl RunMode {
    fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Release => "release",
        }
    }
}

impl Storage {
    fn name(self) -> &'static str {
        match self {
            Self::Ebs => "ebs",
            Self::S3 => "s3",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Operation {
    Create,
    FixtureClone,
    Append,
    Delete,
    Update,
    MergeInsert,
    Backfill,
    DefaultCompactionPreflight,
    DefaultCompaction,
    RandomDeleteReclaim,
    BoundedRecluster,
    NormalizePlacement,
    Repack,
    Recluster,
    CheckpointGeneration,
    IndexBuild,
    IndexTake,
    IndexOptimize,
    Open,
    Scan,
    RowIdScan,
    Take,
}

impl Operation {
    fn is_read_path(self) -> bool {
        matches!(
            self,
            Self::Open | Self::Scan | Self::RowIdScan | Self::Take | Self::IndexTake
        )
    }

    fn name(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::FixtureClone => "fixture_clone",
            Self::Append => "append",
            Self::Delete => "delete",
            Self::Update => "update",
            Self::MergeInsert => "merge_insert",
            Self::Backfill => "backfill",
            Self::DefaultCompactionPreflight => "default_compaction_preflight",
            Self::DefaultCompaction => "default_compaction",
            Self::RandomDeleteReclaim => "random_delete_reclaim",
            Self::BoundedRecluster => "bounded_recluster",
            Self::NormalizePlacement => "normalize_placement",
            Self::Repack => "repack",
            Self::Recluster => "recluster",
            Self::CheckpointGeneration => "checkpoint_generation",
            Self::IndexBuild => "index_build",
            Self::IndexTake => "index_take",
            Self::IndexOptimize => "index_optimize",
            Self::Open => "open",
            Self::Scan => "scan",
            Self::RowIdScan => "row_id_scan",
            Self::Take => "take",
        }
    }

    fn timing_scope(self) -> &'static str {
        match self {
            Self::Create => "dataset_write_including_bounded_synthetic_stream_generation",
            Self::FixtureClone => "cold_session_shallow_clone_of_canonical_fixture",
            Self::Append => {
                "cold_session_open_and_append_including_bounded_synthetic_stream_generation"
            }
            Self::Delete => "cold_session_open_and_delete_commit",
            Self::Update => {
                "cold_session_open_and_update_commit_including_stream_generation_when_applicable"
            }
            Self::MergeInsert => {
                "cold_session_open_and_merge_insert_commit_including_bounded_synthetic_stream_generation"
            }
            Self::Backfill => "cold_session_open_and_row_aligned_backfill_commit",
            Self::DefaultCompactionPreflight => {
                "cold_session_open_and_default_compaction_plan_only"
            }
            Self::DefaultCompaction => "cold_session_open_and_default_compaction_commit",
            Self::RandomDeleteReclaim => {
                "cold_session_open_and_same_postcondition_random_delete_reclaim_commit"
            }
            Self::BoundedRecluster => "cold_session_open_and_bounded_recluster_commit",
            Self::NormalizePlacement => "cold_session_open_and_normalize_placement_commit",
            Self::Repack => "cold_session_open_and_repack_commit",
            Self::Recluster => "cold_session_open_and_recluster_commit",
            Self::CheckpointGeneration => "cold_session_open_and_generation_checkpoint_commit",
            Self::IndexBuild => "cold_session_open_and_index_build_commit",
            Self::IndexTake => "cold_session_open_and_index_lookup_and_take",
            Self::IndexOptimize => "cold_session_open_and_index_optimize_commit",
            Self::Open => "dataset_open_and_contract_validation",
            Self::Scan => "dataset_open_contract_validation_and_full_scan",
            Self::RowIdScan => {
                "cold_session_open_contract_validation_and_full_id_row_id_scan_selection_and_artifact_write"
            }
            Self::Take => "cold_session_open_and_take_rows_with_prepared_ids",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SchemaKind {
    Narrow16,
    Wide128,
    Vector,
}

impl SchemaKind {
    fn name(self) -> &'static str {
        match self {
            Self::Narrow16 => "narrow_16b",
            Self::Wide128 => "wide_128b",
            Self::Vector => "vector_f32_128",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum IndexKind {
    None,
    Scalar,
    Vector,
}

impl IndexKind {
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Scalar => "scalar_btree",
            Self::Vector => "vector_ivf_flat",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum UpdateDriver {
    Native,
    ExactMatchedMerge,
}

impl UpdateDriver {
    fn name(self) -> &'static str {
        match self {
            Self::Native => "native_update_builder",
            Self::ExactMatchedMerge => "exact_selection_matched_merge",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SelectionKind {
    Range,
    Random,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CompactionMode {
    Standard,
    FragmentReuse,
}

impl SelectionKind {
    fn name(self) -> &'static str {
        match self {
            Self::Range => "range",
            Self::Random => "uniform_without_replacement",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run one stable-row-address benchmark operation")]
struct Args {
    #[arg(long)]
    dataset_uri: String,

    /// Canonical immutable fixture used by `fixture-clone`.
    #[arg(long)]
    source_dataset_uri: Option<String>,

    #[arg(long, value_enum)]
    format: BenchFormat,

    #[arg(long, value_enum)]
    storage: Storage,

    #[arg(long, value_enum)]
    operation: Operation,

    #[arg(long, value_enum)]
    mode: RunMode,

    #[arg(long)]
    run_id: String,

    #[arg(long)]
    pair_id: String,

    #[arg(long)]
    round: usize,

    #[arg(long)]
    order_index: usize,

    #[arg(long)]
    rows: usize,

    #[arg(long)]
    rows_per_fragment: usize,

    #[arg(long)]
    take_count: usize,

    /// Expected live rows after the operation. The worker verifies this when it
    /// has a current Dataset handle.
    #[arg(long)]
    expected_rows: usize,

    /// Number of rows affected by a mutating operation.
    #[arg(long, default_value_t = 1)]
    mutation_count: usize,

    /// First ID used by range mutations and inserted rows.
    #[arg(long, default_value_t = 0)]
    id_start: u64,

    /// Workload-local step, independent of the paired-repeat round.
    #[arg(long, default_value_t = 0)]
    step: usize,

    /// Sampling nonce. Sustained runs freeze this at zero while changing
    /// `step`; adversarial runs advance both.
    #[arg(long, default_value_t = 0)]
    selection_step: usize,

    #[arg(long, default_value_t = 50)]
    match_percent: u8,

    #[arg(long, value_enum, default_value_t = SchemaKind::Narrow16)]
    schema_kind: SchemaKind,

    #[arg(long, value_enum, default_value_t = IndexKind::None)]
    index_kind: IndexKind,

    #[arg(long, value_enum, default_value_t = UpdateDriver::Native)]
    update_driver: UpdateDriver,

    #[arg(long, value_enum, default_value_t = SelectionKind::Range)]
    selection: SelectionKind,

    #[arg(long, value_enum, default_value_t = CompactionMode::Standard)]
    compaction_mode: CompactionMode,

    #[arg(long, default_value_t = 1_000_000)]
    target_rows_per_fragment: usize,

    #[arg(long)]
    seed: u64,

    #[arg(long)]
    commit: String,

    #[arg(long)]
    host: String,

    #[arg(long)]
    policy_sha256: String,

    #[arg(long)]
    policy_version: u64,

    /// Write a row-ID fixture. `row-id-scan` records the full scan while
    /// `take` retains the legacy untimed standalone-runner setup.
    #[arg(long, conflicts_with = "take_ids_input")]
    prepare_take_ids_output: Option<PathBuf>,

    /// Read the row-ID fixture before starting the timed take operation.
    #[arg(long, conflicts_with = "prepare_take_ids_output")]
    take_ids_input: Option<PathBuf>,

    /// Write an untimed deterministic physical-maintenance plan and exit.
    #[arg(long, conflicts_with = "maintenance_plan_input")]
    prepare_maintenance_plan_output: Option<PathBuf>,

    /// Apply the exact benchmark-owned physical-maintenance plan.
    #[arg(long, conflicts_with = "prepare_maintenance_plan_output")]
    maintenance_plan_input: Option<PathBuf>,

    /// SHA-256 of the canonical maintenance plan, computed by the runner.
    #[arg(long, requires = "maintenance_plan_input")]
    maintenance_plan_sha256: Option<String>,

    /// Validate plan compatibility against this format without writing data.
    #[arg(long, requires = "maintenance_plan_input")]
    validate_maintenance_plan_only: bool,

    #[arg(long, default_value_t = 134_217_728)]
    target_file_size_bytes: usize,

    #[arg(long, default_value_t = usize::MAX)]
    max_source_fragments_per_group: usize,
}

#[derive(Clone, Debug, Serialize)]
struct PmrIndexGenerationBlocker {
    index_id: String,
    index_name: String,
    field_ids: Vec<i32>,
    oldest_generation: u64,
    region_bytes: u64,
    blocked_transaction_start: u64,
    blocked_transaction_end: u64,
}

#[derive(Debug, Serialize)]
struct BenchmarkRecord {
    schema_version: u64,
    suite: &'static str,
    run_id: String,
    pair_id: String,
    commit: String,
    host: String,
    seed: u64,
    policy_sha256: String,
    policy_version: u64,
    mode: &'static str,
    format: &'static str,
    storage: &'static str,
    operation: &'static str,
    timing_scope: &'static str,
    round: usize,
    order_index: usize,
    dataset_uri: String,
    rows: usize,
    rows_per_fragment: usize,
    take_count: usize,
    expected_rows: usize,
    mutation_count: usize,
    id_start: u64,
    step: usize,
    selection_step: usize,
    match_percent: u8,
    schema_kind: &'static str,
    index_kind: &'static str,
    selection: &'static str,
    implementation_path: &'static str,
    maintenance_plan_path: Option<String>,
    maintenance_plan_sha256: Option<String>,
    started_at_unix_ns: u128,
    duration_ns: u64,
    result_rows: Option<u64>,
    dataset_version: Option<u64>,
    fragments: Option<u64>,
    physical_rows: Option<u64>,
    physical_data_bytes: Option<u64>,
    estimated_live_data_bytes: Option<u64>,
    scan_byte_amplification: Option<f64>,
    dataset_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    get_requests: Option<u64>,
    head_requests: Option<u64>,
    list_requests: Option<u64>,
    put_requests: Option<u64>,
    delete_requests: Option<u64>,
    actual_get_attempts: Option<u64>,
    actual_head_attempts: Option<u64>,
    actual_list_attempts: Option<u64>,
    actual_put_attempts: Option<u64>,
    actual_delete_attempts: Option<u64>,
    read_bytes: Option<u64>,
    write_bytes: Option<u64>,
    data_bytes: Option<u64>,
    index_bytes: Option<u64>,
    metadata_bytes: Option<u64>,
    manifest_bytes: Option<u64>,
    placement_root_bytes: Option<u64>,
    placement_delta_bytes: Option<u64>,
    placement_delta_claimed_bytes: Option<u64>,
    w_epoch_bytes: Option<u64>,
    coverage: Option<f64>,
    recall: Option<f64>,
    admission: Option<bool>,
    placement_maintenance_required: Option<bool>,
    pmr_reason: Option<&'static str>,
    pmr_projected_delta_bytes: Option<u64>,
    pmr_delta_limit_bytes: Option<u64>,
    pmr_projected_epoch_bytes: Option<u64>,
    pmr_epoch_limit_bytes: Option<u64>,
    pmr_generation_delta_bytes: Option<u64>,
    pmr_generation_epoch_bytes: Option<u64>,
    pmr_blocking_indices: Option<Vec<PmrIndexGenerationBlocker>>,
    rows_inserted: Option<u64>,
    rows_updated: Option<u64>,
    rows_deleted: Option<u64>,
    compacted_data_bytes: Option<u64>,
    index_storage_bytes_before: Option<u64>,
    row_addresses_remapped: Option<u64>,
    indices_remapped: Option<u64>,
    index_coverage_reuse: Option<f64>,
    layout_index_maintenance_ns: Option<u64>,
    fragment_reuse_index_present: Option<bool>,
    explicit_locator_objects_written: Option<u64>,
    explicit_locator_bytes_written: Option<u64>,
    compaction_groups_planned: Option<u64>,
    compaction_groups_admitted: Option<u64>,
    compaction_groups_not_admitted: Option<u64>,
    state_digest: Option<String>,
    physical_order_digest: Option<String>,
    io_by_path: Option<BTreeMap<String, PathIoMetrics>>,
    io_metrics_status: &'static str,
    status: &'static str,
    error: Option<String>,
}

#[derive(Debug, Default)]
struct OperationOutcome {
    result_rows: Option<u64>,
    dataset_version: Option<u64>,
    fragments: Option<u64>,
    physical_rows: Option<u64>,
    physical_data_bytes: Option<u64>,
    estimated_live_data_bytes: Option<u64>,
    scan_byte_amplification: Option<f64>,
    manifest_bytes: Option<u64>,
    placement_root_bytes: Option<u64>,
    placement_delta_bytes: Option<u64>,
    placement_delta_claimed_bytes: Option<u64>,
    w_epoch_bytes: Option<u64>,
    coverage: Option<f64>,
    recall: Option<f64>,
    admission: Option<bool>,
    placement_maintenance_required: Option<bool>,
    pmr_reason: Option<&'static str>,
    pmr_projected_delta_bytes: Option<u64>,
    pmr_delta_limit_bytes: Option<u64>,
    pmr_projected_epoch_bytes: Option<u64>,
    pmr_epoch_limit_bytes: Option<u64>,
    pmr_generation_delta_bytes: Option<u64>,
    pmr_generation_epoch_bytes: Option<u64>,
    pmr_blocking_indices: Option<Vec<PmrIndexGenerationBlocker>>,
    rows_inserted: Option<u64>,
    rows_updated: Option<u64>,
    rows_deleted: Option<u64>,
    compacted_data_bytes: Option<u64>,
    index_storage_bytes_before: Option<u64>,
    row_addresses_remapped: Option<u64>,
    indices_remapped: Option<u64>,
    index_coverage_reuse: Option<f64>,
    layout_index_maintenance_ns: Option<u64>,
    fragment_reuse_index_present: Option<bool>,
    explicit_locator_objects_written: Option<u64>,
    explicit_locator_bytes_written: Option<u64>,
    compaction_groups_planned: Option<u64>,
    compaction_groups_admitted: Option<u64>,
    compaction_groups_not_admitted: Option<u64>,
    state_digest: Option<String>,
    physical_order_digest: Option<String>,
}

struct ExecutedOperation {
    dataset: Arc<Dataset>,
    outcome: OperationOutcome,
}

impl ExecutedOperation {
    fn new(dataset: impl Into<Arc<Dataset>>, outcome: OperationOutcome) -> Self {
        Self {
            dataset: dataset.into(),
            outcome,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct PathIoMetrics {
    get_requests: u64,
    head_requests: u64,
    list_requests: u64,
    put_requests: u64,
    delete_requests: u64,
    read_bytes: u64,
    write_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ObjectMetrics {
    actual_get_attempts: u64,
    actual_head_attempts: u64,
    actual_list_attempts: u64,
    actual_put_attempts: u64,
    actual_delete_attempts: u64,
}

impl ObjectMetrics {
    fn checked_delta(self, earlier: Self) -> BenchResult<Self> {
        macro_rules! delta {
            ($field:ident) => {
                self.$field
                    .checked_sub(earlier.$field)
                    .ok_or_else(|| -> BenchError {
                        format!(
                            "object-store metric moved backwards: {}",
                            stringify!($field)
                        )
                        .into()
                    })?
            };
        }
        Ok(Self {
            actual_get_attempts: delta!(actual_get_attempts),
            actual_head_attempts: delta!(actual_head_attempts),
            actual_list_attempts: delta!(actual_list_attempts),
            actual_put_attempts: delta!(actual_put_attempts),
            actual_delete_attempts: delta!(actual_delete_attempts),
        })
    }
}

fn snapshot_object_metrics(snapshotter: &Snapshotter) -> ObjectMetrics {
    let mut metrics = ObjectMetrics::default();
    for (composite_key, _unit, _description, value) in snapshotter.snapshot().into_vec() {
        let DebugValue::Counter(value) = value else {
            continue;
        };
        let key = composite_key.key();
        if key.name() == METRIC_HTTP_ATTEMPTS {
            let method = key
                .labels()
                .find(|label| label.key() == "method")
                .map(|label| label.value());
            match method {
                Some("get") => metrics.actual_get_attempts += value,
                Some("head") => metrics.actual_head_attempts += value,
                Some("list") => metrics.actual_list_attempts += value,
                Some("put") => metrics.actual_put_attempts += value,
                Some("delete") => metrics.actual_delete_attempts += value,
                _ => {}
            }
            continue;
        }
    }
    metrics
}

fn path_io_metrics(stats: &IoStats) -> BTreeMap<String, PathIoMetrics> {
    let mut by_path = BTreeMap::from([
        ("data".to_string(), PathIoMetrics::default()),
        ("index".to_string(), PathIoMetrics::default()),
        ("metadata".to_string(), PathIoMetrics::default()),
        ("other".to_string(), PathIoMetrics::default()),
    ]);
    for request in &stats.requests {
        // `delete_stream` is one aggregate control request over a stream of
        // paths, so the tracker cannot attach one concrete object path. Keep
        // the request and classify its zero-byte sentinel as metadata/control.
        let category = maintenance::aggregate_control_path_category(
            request.operation,
            request.path.as_ref(),
            request.num_bytes,
        )
        .unwrap_or_else(|| path_category(request.path.as_ref()));
        let totals = by_path.get_mut(category).expect("fixed path category");
        match request.operation {
            IoOperation::Get => totals.get_requests += 1,
            IoOperation::Head => totals.head_requests += 1,
            IoOperation::List => totals.list_requests += 1,
            IoOperation::Put | IoOperation::Copy | IoOperation::Rename => {
                totals.put_requests += 1;
            }
            IoOperation::Delete => totals.delete_requests += 1,
            IoOperation::Other => {}
        }
        match request.operation {
            IoOperation::Get | IoOperation::Head | IoOperation::List => {
                totals.read_bytes += request.num_bytes;
            }
            IoOperation::Put | IoOperation::Delete | IoOperation::Copy | IoOperation::Rename => {
                totals.write_bytes += request.num_bytes
            }
            IoOperation::Other => {}
        }
    }
    by_path
}

fn path_category(path: &str) -> &'static str {
    let components = path.split('/');
    if components.clone().any(|component| component == "_indices") {
        "index"
    } else if components.clone().any(|component| {
        matches!(
            component,
            "_versions" | "_transactions" | "_deletions" | "_rowids" | "_refs" | "_tags"
        )
    }) {
        "metadata"
    } else if components.clone().any(|component| component == "data") {
        "data"
    } else {
        "other"
    }
}

fn path_metrics_are_empty(metrics: &PathIoMetrics) -> bool {
    metrics.get_requests == 0
        && metrics.head_requests == 0
        && metrics.list_requests == 0
        && metrics.put_requests == 0
        && metrics.delete_requests == 0
        && metrics.read_bytes == 0
        && metrics.write_bytes == 0
}

fn path_total_bytes(metrics: &BTreeMap<String, PathIoMetrics>, category: &str) -> u64 {
    metrics
        .get(category)
        .map(|metrics| metrics.read_bytes + metrics.write_bytes)
        .unwrap_or_default()
}

fn aggregate_path_metrics(metrics: &BTreeMap<String, PathIoMetrics>) -> PathIoMetrics {
    let mut aggregate = PathIoMetrics::default();
    for value in metrics.values() {
        aggregate.get_requests += value.get_requests;
        aggregate.head_requests += value.head_requests;
        aggregate.list_requests += value.list_requests;
        aggregate.put_requests += value.put_requests;
        aggregate.delete_requests += value.delete_requests;
        aggregate.read_bytes += value.read_bytes;
        aggregate.write_bytes += value.write_bytes;
    }
    aggregate
}

fn actual_attempts_cover_operation(operation: Operation, metrics: ObjectMetrics) -> bool {
    match operation {
        Operation::Create
        | Operation::FixtureClone
        | Operation::Append
        | Operation::Delete
        | Operation::Update
        | Operation::MergeInsert
        | Operation::Backfill
        | Operation::DefaultCompaction
        | Operation::RandomDeleteReclaim
        | Operation::BoundedRecluster
        | Operation::NormalizePlacement
        | Operation::Repack
        | Operation::Recluster
        | Operation::CheckpointGeneration
        | Operation::IndexBuild
        | Operation::IndexOptimize => {
            metrics.actual_get_attempts + metrics.actual_head_attempts + metrics.actual_put_attempts
                > 0
        }
        Operation::Open => metrics.actual_get_attempts + metrics.actual_head_attempts > 0,
        Operation::DefaultCompactionPreflight => {
            metrics.actual_get_attempts + metrics.actual_head_attempts > 0
        }
        Operation::Scan | Operation::RowIdScan | Operation::Take | Operation::IndexTake => {
            metrics.actual_get_attempts > 0
        }
    }
}

enum PreparedOperation {
    None,
    Batches(SyntheticBatchSource),
    Take {
        user_ids: Vec<u64>,
        row_ids: Vec<u64>,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TakeIdsArtifact {
    schema_version: u64,
    run_id: String,
    commit: String,
    policy_sha256: String,
    format: String,
    round: usize,
    rows: usize,
    take_count: usize,
    seed: u64,
    user_ids: Vec<u64>,
    row_ids: Vec<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MaintenancePlanArtifact {
    schema_version: u64,
    suite: String,
    run_id: String,
    commit: String,
    policy_sha256: String,
    source_format: String,
    source_dataset_uri: String,
    source_dataset_version: u64,
    schema_kind: String,
    expected_rows: usize,
    target_rows_per_fragment: usize,
    execution_target_rows_per_fragment: usize,
    target_file_size_bytes: usize,
    max_source_fragments_per_group: usize,
    fragment_count: usize,
    groups: Vec<MaintenancePlanGroup>,
    expected_output_live_rows: Vec<usize>,
    expected_output_fragment_count: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MaintenancePlanGroup {
    start_ordinal: usize,
    end_ordinal: usize,
    source_live_rows: usize,
    source_physical_rows: usize,
    source_physical_data_bytes: u64,
    source_live_data_bytes: u64,
    expected_output_fragments: usize,
}

fn main() -> BenchResult<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    if args.operation == Operation::Take
        && let Some(output) = &args.prepare_take_ids_output
    {
        runtime.block_on(write_take_ids_artifact(
            &args,
            output,
            Arc::new(IOTracker::default()),
        ))?;
        return Ok(());
    }
    if let Some(output) = &args.prepare_maintenance_plan_output {
        runtime.block_on(write_maintenance_plan_artifact(
            &args,
            output,
            Arc::new(IOTracker::default()),
        ))?;
        return Ok(());
    }
    if args.validate_maintenance_plan_only {
        runtime.block_on(async {
            let dataset = open_dataset(&args, Arc::new(IOTracker::default())).await?;
            args.format.validate_dataset(&dataset)?;
            read_maintenance_plan(&args, &dataset)?;
            Ok::<(), BenchError>(())
        })?;
        return Ok(());
    }
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install()?;
    let tracker = Arc::new(IOTracker::default());
    let prepared = match args.operation {
        Operation::Create => make_batches(&args, 0, args.rows, 0).map(PreparedOperation::Batches),
        Operation::Append => {
            make_batches(&args, args.id_start, args.mutation_count, args.step as u64)
                .map(PreparedOperation::Batches)
        }
        Operation::MergeInsert => make_merge_insert_batches(&args).map(PreparedOperation::Batches),
        Operation::Update if args.update_driver == UpdateDriver::ExactMatchedMerge => {
            make_exact_update_batches(&args).map(PreparedOperation::Batches)
        }
        Operation::Take | Operation::IndexTake => read_take_ids_artifact(&args),
        _ => Ok(PreparedOperation::None),
    };

    let metrics_before = snapshot_object_metrics(&snapshotter);
    let started_at_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let started = Instant::now();
    let execution_result = match prepared {
        Ok(prepared) => runtime.block_on(execute(&args, prepared, tracker.clone())),
        Err(error) => Err(error),
    };
    // Freeze latency, I/O, and RSS at the operation boundary. Manifest/root
    // evidence below is benchmark validation, not part of the operation cost.
    let measured = measurement::MeasuredOperation::freeze(execution_result, started.elapsed());
    let wrapper_io_stats = tracker.incremental_stats();
    let metric_result = snapshot_object_metrics(&snapshotter).checked_delta(metrics_before);
    let measured_peak_rss_bytes = peak_rss_bytes();
    let (duration_ns, result) = measured.finish_with(|result| {
        result
            .and_then(|executed| finish_outcome(&args, executed.dataset.as_ref(), executed.outcome))
    });

    let mut errors = Vec::new();
    let mut outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => {
            if let Some(diagnostic) = placement_maintenance_diagnostic(error.as_ref()) {
                diagnostic
            } else {
                errors.push(format!("{error:#}"));
                OperationOutcome::default()
            }
        }
    };
    let needs_delta_validation = outcome.placement_delta_claimed_bytes.is_some();
    let needs_coverage_validation = outcome.coverage.is_some();
    if errors.is_empty() && (needs_delta_validation || needs_coverage_validation) {
        let validation = runtime.block_on(async {
            let dataset = open_dataset(&args, Arc::new(IOTracker::default())).await?;
            if args.operation.is_read_path() {
                args.format.validate_dataset_header(&dataset)?;
            } else {
                args.format.validate_dataset(&dataset)?;
            }
            let delta = if needs_delta_validation {
                Some(exact_row_address_delta_bytes(&dataset).await?)
            } else {
                None
            };
            let coverage = if needs_coverage_validation {
                Some(effective_index_coverage(&dataset, args.index_kind).await?)
            } else {
                None
            };
            Ok::<_, BenchError>((delta, coverage))
        });
        match validation {
            Ok((delta, coverage)) => {
                outcome.placement_delta_bytes = delta;
                if let Some(coverage) = coverage {
                    outcome.coverage = Some(coverage);
                }
            }
            Err(error) => errors.push(format!("post-measurement evidence validation: {error:#}")),
        }
    }
    // The shared wrapper is propagated to primary and base-path stores. Dataset
    // internal stats are store-local and would omit shallow-clone base reads.
    let io_stats = wrapper_io_stats;
    let path_metrics = path_io_metrics(&io_stats);
    let metrics = match metric_result {
        Ok(metrics) => metrics,
        Err(error) => {
            errors.push(error.to_string());
            ObjectMetrics::default()
        }
    };
    if path_metrics
        .get("other")
        .is_some_and(|metrics| !path_metrics_are_empty(metrics))
    {
        errors.push(format!(
            "operation accessed unclassified object-store paths: {:?}",
            io_stats.requests
        ));
    }
    if io_stats
        .requests
        .iter()
        .any(|request| request.operation == IoOperation::Other)
    {
        errors.push(format!(
            "operation emitted unclassified logical I/O methods: {:?}",
            io_stats.requests
        ));
    }
    if args.storage == Storage::S3 && !actual_attempts_cover_operation(args.operation, metrics) {
        errors.push(format!(
            "native S3 operation recorded no required actual HTTP attempts: operation={}",
            args.operation.name()
        ));
    }
    let (status, error) = if errors.is_empty() {
        ("ok", None)
    } else {
        ("error", Some(errors.join("; ")))
    };

    let data_bytes = path_total_bytes(&path_metrics, "data");
    let index_bytes = path_total_bytes(&path_metrics, "index");
    let metadata_bytes = path_total_bytes(&path_metrics, "metadata");
    let logical_metrics = aggregate_path_metrics(&path_metrics);

    let dataset_bytes = if args.storage == Storage::Ebs {
        directory_size(Path::new(&args.dataset_uri)).ok()
    } else {
        None
    };

    let implementation_path = implementation_path(&args);
    let record = BenchmarkRecord {
        schema_version: SCHEMA_VERSION,
        suite: "stable_row_address_e2e",
        run_id: args.run_id,
        pair_id: args.pair_id,
        commit: args.commit,
        host: args.host,
        seed: args.seed,
        policy_sha256: args.policy_sha256,
        policy_version: args.policy_version,
        mode: args.mode.name(),
        format: args.format.name(),
        storage: args.storage.name(),
        operation: args.operation.name(),
        timing_scope: args.operation.timing_scope(),
        round: args.round,
        order_index: args.order_index,
        dataset_uri: args.dataset_uri,
        rows: args.rows,
        rows_per_fragment: args.rows_per_fragment,
        take_count: args.take_count,
        expected_rows: args.expected_rows,
        mutation_count: args.mutation_count,
        id_start: args.id_start,
        step: args.step,
        selection_step: args.selection_step,
        match_percent: args.match_percent,
        schema_kind: args.schema_kind.name(),
        index_kind: args.index_kind.name(),
        selection: args.selection.name(),
        implementation_path,
        maintenance_plan_path: args
            .maintenance_plan_input
            .as_ref()
            .map(|path| path.display().to_string()),
        maintenance_plan_sha256: args.maintenance_plan_sha256,
        started_at_unix_ns,
        duration_ns,
        result_rows: outcome.result_rows,
        dataset_version: outcome.dataset_version,
        fragments: outcome.fragments,
        physical_rows: outcome.physical_rows,
        physical_data_bytes: outcome.physical_data_bytes,
        estimated_live_data_bytes: outcome.estimated_live_data_bytes,
        scan_byte_amplification: outcome.scan_byte_amplification,
        dataset_bytes,
        peak_rss_bytes: measured_peak_rss_bytes,
        get_requests: Some(logical_metrics.get_requests),
        head_requests: Some(logical_metrics.head_requests),
        list_requests: Some(logical_metrics.list_requests),
        put_requests: Some(logical_metrics.put_requests),
        delete_requests: Some(logical_metrics.delete_requests),
        actual_get_attempts: (args.storage == Storage::S3).then_some(metrics.actual_get_attempts),
        actual_head_attempts: (args.storage == Storage::S3).then_some(metrics.actual_head_attempts),
        actual_list_attempts: (args.storage == Storage::S3).then_some(metrics.actual_list_attempts),
        actual_put_attempts: (args.storage == Storage::S3).then_some(metrics.actual_put_attempts),
        actual_delete_attempts: (args.storage == Storage::S3)
            .then_some(metrics.actual_delete_attempts),
        read_bytes: Some(logical_metrics.read_bytes),
        write_bytes: Some(logical_metrics.write_bytes),
        data_bytes: Some(data_bytes),
        index_bytes: Some(index_bytes),
        metadata_bytes: Some(metadata_bytes),
        manifest_bytes: outcome.manifest_bytes,
        placement_root_bytes: outcome.placement_root_bytes,
        placement_delta_bytes: outcome.placement_delta_bytes,
        placement_delta_claimed_bytes: outcome.placement_delta_claimed_bytes,
        w_epoch_bytes: outcome.w_epoch_bytes,
        coverage: outcome.coverage,
        recall: outcome.recall,
        admission: outcome.admission,
        placement_maintenance_required: outcome.placement_maintenance_required,
        pmr_reason: outcome.pmr_reason,
        pmr_projected_delta_bytes: outcome.pmr_projected_delta_bytes,
        pmr_delta_limit_bytes: outcome.pmr_delta_limit_bytes,
        pmr_projected_epoch_bytes: outcome.pmr_projected_epoch_bytes,
        pmr_epoch_limit_bytes: outcome.pmr_epoch_limit_bytes,
        pmr_generation_delta_bytes: outcome.pmr_generation_delta_bytes,
        pmr_generation_epoch_bytes: outcome.pmr_generation_epoch_bytes,
        pmr_blocking_indices: outcome.pmr_blocking_indices,
        rows_inserted: outcome.rows_inserted,
        rows_updated: outcome.rows_updated,
        rows_deleted: outcome.rows_deleted,
        compacted_data_bytes: outcome.compacted_data_bytes,
        index_storage_bytes_before: outcome.index_storage_bytes_before,
        row_addresses_remapped: outcome.row_addresses_remapped,
        indices_remapped: outcome.indices_remapped,
        index_coverage_reuse: outcome.index_coverage_reuse,
        layout_index_maintenance_ns: outcome.layout_index_maintenance_ns,
        fragment_reuse_index_present: outcome.fragment_reuse_index_present,
        explicit_locator_objects_written: outcome.explicit_locator_objects_written,
        explicit_locator_bytes_written: outcome.explicit_locator_bytes_written,
        compaction_groups_planned: outcome.compaction_groups_planned,
        compaction_groups_admitted: outcome.compaction_groups_admitted,
        compaction_groups_not_admitted: outcome.compaction_groups_not_admitted,
        state_digest: outcome.state_digest,
        physical_order_digest: outcome.physical_order_digest,
        io_by_path: Some(path_metrics),
        io_metrics_status: if args.storage == Storage::S3 {
            "measured"
        } else {
            "logical_only"
        },
        status,
        error,
    };
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

fn validate_args(args: &Args) -> BenchResult<()> {
    if args.rows == 0 {
        return Err("--rows must be greater than zero".into());
    }
    if args.rows_per_fragment == 0 {
        return Err("--rows-per-fragment must be greater than zero".into());
    }
    if args.take_count == 0 || args.take_count > args.expected_rows.max(args.rows) {
        return Err("--take-count must be positive and not exceed the addressable fixture".into());
    }
    if args.mutation_count == 0 {
        return Err("--mutation-count must be greater than zero".into());
    }
    if args.match_percent > 100 {
        return Err("--match-percent must be in 0..=100".into());
    }
    if args.target_rows_per_fragment == 0 {
        return Err("--target-rows-per-fragment must be greater than zero".into());
    }
    if args.compaction_mode != CompactionMode::Standard
        && args.operation != Operation::DefaultCompaction
    {
        return Err("--compaction-mode requires --operation=default-compaction".into());
    }
    if args.index_kind == IndexKind::Vector && args.schema_kind != SchemaKind::Vector {
        return Err("--index-kind=vector requires --schema-kind=vector".into());
    }
    if args.index_kind == IndexKind::None
        && matches!(
            args.operation,
            Operation::IndexBuild | Operation::IndexTake | Operation::IndexOptimize
        )
    {
        return Err("index operations require --index-kind=scalar or vector".into());
    }
    if matches!(args.operation, Operation::Delete | Operation::Update)
        && args.mutation_count > args.rows
    {
        return Err("delete/update mutation_count must not exceed base rows".into());
    }
    if args.operation == Operation::Update
        && args.selection == SelectionKind::Random
        && args.update_driver != UpdateDriver::ExactMatchedMerge
    {
        return Err("uniform random update requires --update-driver=exact-matched-merge".into());
    }
    if args.order_index >= 3 {
        return Err("--order-index must be in 0..3".into());
    }
    let has_prepare_take_ids = args.prepare_take_ids_output.is_some();
    let has_take_ids_input = args.take_ids_input.is_some();
    if args.operation == Operation::Take && !has_prepare_take_ids && !has_take_ids_input {
        return Err("take requires --take-ids-input or --prepare-take-ids-output".into());
    }
    if args.operation == Operation::IndexTake && !has_take_ids_input {
        return Err("index-take requires --take-ids-input".into());
    }
    if args.operation == Operation::RowIdScan && !has_prepare_take_ids {
        return Err("row-id-scan requires --prepare-take-ids-output".into());
    }
    if has_prepare_take_ids && !matches!(args.operation, Operation::Take | Operation::RowIdScan) {
        return Err("--prepare-take-ids-output requires --operation=take or row-id-scan".into());
    }
    if has_take_ids_input && !matches!(args.operation, Operation::Take | Operation::IndexTake) {
        return Err("--take-ids-input requires --operation=take or index-take".into());
    }
    let has_plan_output = args.prepare_maintenance_plan_output.is_some();
    let has_plan_input = args.maintenance_plan_input.is_some();
    if has_plan_output
        && !matches!(
            args.operation,
            Operation::DefaultCompaction
                | Operation::RandomDeleteReclaim
                | Operation::BoundedRecluster
                | Operation::NormalizePlacement
                | Operation::Repack
                | Operation::Recluster
        )
    {
        return Err("maintenance-plan preparation requires a relocation operation".into());
    }
    if has_plan_input
        && !matches!(
            args.operation,
            Operation::DefaultCompaction
                | Operation::RandomDeleteReclaim
                | Operation::BoundedRecluster
                | Operation::NormalizePlacement
                | Operation::Repack
                | Operation::Recluster
        )
    {
        return Err("maintenance-plan input requires a relocation operation".into());
    }
    if args.maintenance_plan_sha256.is_some() != has_plan_input {
        return Err(
            "--maintenance-plan-input and --maintenance-plan-sha256 must be provided together"
                .into(),
        );
    }
    if let Some(digest) = &args.maintenance_plan_sha256
        && !is_lower_hex(digest, 64)
    {
        return Err("--maintenance-plan-sha256 must be a lowercase SHA-256 digest".into());
    }
    if args.target_file_size_bytes == 0 || args.max_source_fragments_per_group == 0 {
        return Err(
            "maintenance target bytes and max source fragments must be greater than zero".into(),
        );
    }
    match (&args.operation, &args.source_dataset_uri) {
        (Operation::FixtureClone, Some(source)) if !source.trim().is_empty() => {}
        (Operation::FixtureClone, _) => {
            return Err("fixture-clone requires --source-dataset-uri".into());
        }
        (_, Some(_)) => {
            return Err("--source-dataset-uri requires --operation=fixture-clone".into());
        }
        (_, None) => {}
    }
    if !is_lower_hex(&args.commit, 40) {
        return Err("--commit must be a full lowercase 40-character Git SHA".into());
    }
    if !is_lower_hex(&args.policy_sha256, 64) {
        return Err("--policy-sha256 must be a lowercase SHA-256 digest".into());
    }
    for (name, value) in [
        ("--run-id", args.run_id.as_str()),
        ("--pair-id", args.pair_id.as_str()),
        ("--host", args.host.as_str()),
        ("--dataset-uri", args.dataset_uri.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("{name} must not be empty").into());
        }
    }
    Ok(())
}

fn implementation_path(args: &Args) -> &'static str {
    match args.operation {
        Operation::Update => args.update_driver.name(),
        Operation::NormalizePlacement
        | Operation::Repack
        | Operation::Recluster
        | Operation::CheckpointGeneration => "capability_gated_explicit_maintenance",
        Operation::DefaultCompactionPreflight => "default_compaction_plan_only",
        Operation::DefaultCompaction if args.compaction_mode == CompactionMode::FragmentReuse => {
            match args.format {
                BenchFormat::V22NoStable => "deferred_fragment_reuse_compaction",
                BenchFormat::V22Stable => "inline_index_remap_compaction",
                BenchFormat::V23Logical => "stable_logical_zero_remap_compaction",
            }
        }
        Operation::DefaultCompaction => "default_compaction",
        Operation::FixtureClone => "canonical_fixture_shallow_clone",
        Operation::RandomDeleteReclaim => match args.format {
            BenchFormat::V23Logical => "explicit_repack",
            BenchFormat::V22Stable if args.index_kind != IndexKind::None => {
                "same_postcondition_default_compaction_full_index_rebuild"
            }
            BenchFormat::V22NoStable | BenchFormat::V22Stable => {
                "same_postcondition_default_compaction"
            }
        },
        Operation::BoundedRecluster => match args.format {
            BenchFormat::V23Logical => "default_bounded_recluster_fast_path",
            BenchFormat::V22NoStable | BenchFormat::V22Stable => {
                "same_postcondition_bounded_recluster_rewrite"
            }
        },
        Operation::IndexBuild | Operation::IndexTake | Operation::IndexOptimize => {
            args.index_kind.name()
        }
        Operation::RowIdScan => "full_scan_business_id_and_row_id_selection",
        _ => "native_dataset_api",
    }
}

fn placement_maintenance_diagnostic(
    mut error: &(dyn std::error::Error + 'static),
) -> Option<OperationOutcome> {
    loop {
        if let Some(reason) = error.downcast_ref::<PlacementMaintenanceRequired>() {
            let mut outcome = OperationOutcome {
                admission: Some(false),
                placement_maintenance_required: Some(true),
                ..Default::default()
            };
            match reason {
                PlacementMaintenanceRequired::ProjectedDeltaBytes { projected, limit } => {
                    outcome.pmr_reason = Some("projected_delta_bytes");
                    outcome.pmr_projected_delta_bytes = Some(*projected);
                    outcome.pmr_delta_limit_bytes = Some(*limit);
                }
                PlacementMaintenanceRequired::ProjectedEpochBytes { projected, limit } => {
                    outcome.pmr_reason = Some("projected_epoch_bytes");
                    outcome.pmr_projected_epoch_bytes = Some(*projected);
                    outcome.pmr_epoch_limit_bytes = Some(*limit);
                }
                PlacementMaintenanceRequired::ExtentFanout { .. } => {
                    outcome.pmr_reason = Some("extent_fanout");
                }
                PlacementMaintenanceRequired::ExistingExplicitMapRequiresRewrite { .. } => {
                    outcome.pmr_reason = Some("existing_explicit_map_requires_rewrite");
                }
                PlacementMaintenanceRequired::ExplicitMapMetadataRequired { .. } => {
                    outcome.pmr_reason = Some("explicit_map_metadata_required");
                }
                PlacementMaintenanceRequired::SelectionSubtractionRequiresRewrite { .. } => {
                    outcome.pmr_reason = Some("selection_subtraction_requires_rewrite");
                }
                PlacementMaintenanceRequired::PackedRunSubtractionRequiresRewrite { .. } => {
                    outcome.pmr_reason = Some("packed_run_subtraction_requires_rewrite");
                }
                PlacementMaintenanceRequired::LogicalOrderRequiresRewrite { .. } => {
                    outcome.pmr_reason = Some("logical_order_requires_rewrite");
                }
                PlacementMaintenanceRequired::IndexGenerationBlocked {
                    projected_delta_bytes,
                    delta_limit,
                    projected_epoch_bytes,
                    epoch_limit,
                    generation_delta_bytes,
                    generation_epoch_bytes,
                    blocking_indices,
                } => {
                    outcome.pmr_reason = Some("index_generation_blocked");
                    outcome.pmr_projected_delta_bytes = Some(*projected_delta_bytes);
                    outcome.pmr_delta_limit_bytes = Some(*delta_limit);
                    outcome.pmr_projected_epoch_bytes = Some(*projected_epoch_bytes);
                    outcome.pmr_epoch_limit_bytes = Some(*epoch_limit);
                    outcome.pmr_generation_delta_bytes = Some(*generation_delta_bytes);
                    outcome.pmr_generation_epoch_bytes = Some(*generation_epoch_bytes);
                    outcome.pmr_blocking_indices = Some(
                        blocking_indices
                            .iter()
                            .map(|blocker| PmrIndexGenerationBlocker {
                                index_id: blocker.index_id.to_string(),
                                index_name: blocker.index_name.clone(),
                                field_ids: blocker.field_ids.clone(),
                                oldest_generation: blocker.oldest_generation,
                                region_bytes: blocker.region_bytes,
                                blocked_transaction_start: blocker.blocked_transaction_start,
                                blocked_transaction_end: blocker.blocked_transaction_end,
                            })
                            .collect(),
                    );
                }
            }
            return Some(outcome);
        }
        let source = error.source()?;
        error = source;
    }
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

const VECTOR_DIMENSION: usize = 128;
const WIDE_PAYLOAD_COLUMNS: usize = 14;

#[derive(Debug)]
enum SyntheticIds {
    Range { start: u64, len: usize },
    Explicit(Arc<Vec<u64>>),
}

impl SyntheticIds {
    fn len(&self) -> usize {
        match self {
            Self::Range { len, .. } => *len,
            Self::Explicit(ids) => ids.len(),
        }
    }

    fn slice(&self, start: usize, end: usize) -> BenchResult<Vec<u64>> {
        match self {
            Self::Range {
                start: first_id, ..
            } => {
                let start = first_id
                    .checked_add(start as u64)
                    .ok_or("synthetic ID range overflow")?;
                let end = first_id
                    .checked_add(end as u64)
                    .ok_or("synthetic ID range overflow")?;
                Ok((start..end).collect())
            }
            Self::Explicit(ids) => Ok(ids[start..end].to_vec()),
        }
    }
}

#[derive(Debug)]
struct SyntheticBatchSource {
    ids: SyntheticIds,
    schema_kind: SchemaKind,
    rows_per_batch: usize,
    base_rows: usize,
    seed: u64,
    value_epoch: u64,
}

impl SyntheticBatchSource {
    fn range(args: &Args, start: u64, len: usize, value_epoch: u64) -> BenchResult<Self> {
        start
            .checked_add(len as u64)
            .ok_or("synthetic ID range overflow")?;
        Ok(Self {
            ids: SyntheticIds::Range { start, len },
            schema_kind: args.schema_kind,
            rows_per_batch: args.rows_per_fragment.min(65_536),
            base_rows: args.rows,
            seed: args.seed,
            value_epoch,
        })
    }

    fn explicit(args: &Args, ids: Vec<u64>, value_epoch: u64) -> BenchResult<Self> {
        if ids.is_empty() {
            return Err("synthetic explicit ID set must not be empty".into());
        }
        if ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("synthetic explicit IDs must be strictly increasing".into());
        }
        Ok(Self {
            ids: SyntheticIds::Explicit(Arc::new(ids)),
            schema_kind: args.schema_kind,
            rows_per_batch: args.rows_per_fragment.min(65_536),
            base_rows: args.rows,
            seed: args.seed,
            value_epoch,
        })
    }

    fn into_reader(self) -> SyntheticBatchReader {
        SyntheticBatchReader {
            schema: synthetic_schema(self.schema_kind),
            source: self,
            offset: 0,
        }
    }
}

struct SyntheticBatchReader {
    source: SyntheticBatchSource,
    schema: Arc<Schema>,
    offset: usize,
}

impl Iterator for SyntheticBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.source.ids.len() {
            return None;
        }
        let end = (self.offset + self.source.rows_per_batch).min(self.source.ids.len());
        let ids = match self.source.ids.slice(self.offset, end) {
            Ok(ids) => ids,
            Err(error) => return Some(Err(ArrowError::ExternalError(error))),
        };
        self.offset = end;
        Some(make_synthetic_batch(
            self.schema.clone(),
            self.source.schema_kind,
            &ids,
            self.source.base_rows,
            self.source.seed,
            self.source.value_epoch,
        ))
    }
}

impl RecordBatchReader for SyntheticBatchReader {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

fn synthetic_schema(kind: SchemaKind) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("value", DataType::UInt64, false),
    ];
    match kind {
        SchemaKind::Narrow16 => {}
        SchemaKind::Wide128 => {
            fields.extend(
                (0..WIDE_PAYLOAD_COLUMNS).map(|index| {
                    Field::new(format!("payload_{index:02}"), DataType::UInt64, false)
                }),
            );
        }
        SchemaKind::Vector => fields.push(Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                VECTOR_DIMENSION as i32,
            ),
            false,
        )),
    }
    Arc::new(Schema::new(fields))
}

fn make_synthetic_batch(
    schema: Arc<Schema>,
    kind: SchemaKind,
    ids: &[u64],
    base_rows: usize,
    seed: u64,
    value_epoch: u64,
) -> Result<RecordBatch, ArrowError> {
    let values = ids
        .iter()
        .copied()
        .map(|id| synthetic_value(id, base_rows, seed, value_epoch));
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(ids.to_vec())),
        Arc::new(UInt64Array::from_iter_values(values)),
    ];
    match kind {
        SchemaKind::Narrow16 => {}
        SchemaKind::Wide128 => {
            for column in 0..WIDE_PAYLOAD_COLUMNS {
                columns.push(Arc::new(UInt64Array::from_iter_values(
                    ids.iter().copied().map(|id| {
                        mix64(
                            id ^ seed.rotate_left((column + 1) as u32)
                                ^ (column as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                        )
                    }),
                )));
            }
        }
        SchemaKind::Vector => {
            let vector_values = Float32Array::from_iter_values(ids.iter().flat_map(|id| {
                (0..VECTOR_DIMENSION).map(move |dimension| {
                    let bits =
                        mix64(*id ^ seed ^ (dimension as u64).wrapping_mul(0xd6e8_feb8_6659_fd93));
                    ((bits >> 40) as u32) as f32 / (u32::MAX >> 8) as f32
                })
            }));
            columns.push(Arc::new(FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                VECTOR_DIMENSION as i32,
                Arc::new(vector_values),
                None,
            )?));
        }
    }
    RecordBatch::try_new(schema, columns)
}

fn make_batches(
    args: &Args,
    start: u64,
    rows: usize,
    value_epoch: u64,
) -> BenchResult<SyntheticBatchSource> {
    SyntheticBatchSource::range(args, start, rows, value_epoch)
}

fn make_exact_update_batches(args: &Args) -> BenchResult<SyntheticBatchSource> {
    let mut ids = sample_positions(
        args.rows,
        args.mutation_count,
        args.seed,
        args.selection_step,
    )
    .into_iter()
    .map(|position| position as u64)
    .collect::<Vec<_>>();
    ids.sort_unstable();
    let mut source = SyntheticBatchSource::explicit(args, ids, args.step as u64 + 1)?;
    // Exact repeated updates carry only the join key and the value being changed.
    source.schema_kind = SchemaKind::Narrow16;
    Ok(source)
}

fn make_merge_insert_batches(args: &Args) -> BenchResult<SyntheticBatchSource> {
    let matched = args
        .mutation_count
        .checked_mul(args.match_percent as usize)
        .ok_or("merge match count overflow")?
        / 100;
    if matched > args.rows {
        return Err("merge matched rows exceed the base fixture".into());
    }
    let inserted = args.mutation_count - matched;
    let mut ids = (0..matched as u64).collect::<Vec<_>>();
    let inserted_end = args
        .id_start
        .checked_add(inserted as u64)
        .ok_or("merge inserted ID range overflow")?;
    ids.extend(args.id_start..inserted_end);
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("merge source matched and inserted ID ranges overlap".into());
    }
    SyntheticBatchSource::explicit(args, ids, args.step as u64 + 1)
}

fn synthetic_value(id: u64, base_rows: usize, seed: u64, value_epoch: u64) -> u64 {
    if value_epoch == 0 && id < base_rows as u64 {
        permute_range(id, base_rows as u64, seed)
    } else {
        mix64(id ^ seed ^ value_epoch.wrapping_mul(0xa076_1d64_78bd_642f))
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Cycle-walk an invertible keyed permutation over the next power-of-two
/// domain. Thresholding its output yields an exact-size, deterministic random
/// selection without storing an auxiliary selection column.
fn permute_range(value: u64, len: u64, seed: u64) -> u64 {
    debug_assert!(value < len && len > 0);
    let bits = (u64::BITS - len.saturating_sub(1).leading_zeros()).max(1);
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    };
    let mut candidate = value;
    loop {
        candidate = permute_power_of_two(candidate, mask, bits, seed);
        if candidate < len {
            return candidate;
        }
    }
}

fn permute_power_of_two(mut value: u64, mask: u64, bits: u32, seed: u64) -> u64 {
    for round in 0..4_u32 {
        let key = mix64(seed ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let right_shift = (bits / 2).max(1);
        value ^= value >> right_shift;
        value &= mask;
        value = value
            .wrapping_mul(key | 1)
            .wrapping_add(key.rotate_left(17))
            & mask;
        let left_shift = (bits / 3).max(1);
        value ^= value.wrapping_shl(left_shift) & mask;
        value &= mask;
    }
    value
}

fn store_params(tracker: Arc<IOTracker>) -> ObjectStoreParams {
    ObjectStoreParams {
        io_tracker: Some(tracker),
        ..Default::default()
    }
}

async fn open_dataset(args: &Args, tracker: Arc<IOTracker>) -> BenchResult<Dataset> {
    open_dataset_uri(&args.dataset_uri, tracker).await
}

async fn open_dataset_uri(uri: &str, tracker: Arc<IOTracker>) -> BenchResult<Dataset> {
    Ok(DatasetBuilder::from_uri(uri)
        .with_index_cache_size_bytes(0)
        .with_metadata_cache_size_bytes(0)
        .with_read_params(ReadParams {
            store_options: Some(store_params(tracker)),
            ..Default::default()
        })
        .load()
        .await?)
}

#[derive(Debug, Default)]
struct RemapMeasurement {
    row_addresses_remapped: u64,
    indices_remapped: u64,
    elapsed_ns: u64,
}

struct MeasuredIndexRemapperOptions {
    measurement: Arc<Mutex<RemapMeasurement>>,
}

impl IndexRemapperOptions for MeasuredIndexRemapperOptions {
    fn create_remapper(&self, dataset: &Dataset) -> lance::Result<Box<dyn IndexRemapper>> {
        let inner = DatasetIndexRemapperOptions::default().create_remapper(dataset)?;
        let rows_by_fragment = dataset
            .manifest()
            .fragments
            .iter()
            .map(|fragment| {
                (
                    fragment.id,
                    fragment.physical_rows.unwrap_or_default() as u64,
                )
            })
            .collect();
        Ok(Box::new(MeasuredIndexRemapper {
            inner,
            rows_by_fragment,
            measurement: self.measurement.clone(),
        }))
    }
}

struct MeasuredIndexRemapper {
    inner: Box<dyn IndexRemapper>,
    rows_by_fragment: BTreeMap<u64, u64>,
    measurement: Arc<Mutex<RemapMeasurement>>,
}

#[async_trait::async_trait]
impl IndexRemapper for MeasuredIndexRemapper {
    async fn remap_indices(
        &self,
        index_map: RowAddrRemap,
        affected_fragment_ids: &[u64],
    ) -> lance::Result<Vec<RemappedIndex>> {
        let remapped_rows = affected_fragment_ids
            .iter()
            .filter_map(|fragment_id| self.rows_by_fragment.get(fragment_id))
            .copied()
            .sum::<u64>();
        let started = Instant::now();
        let remapped = self
            .inner
            .remap_indices(index_map, affected_fragment_ids)
            .await?;
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let mut measurement = self.measurement.lock().map_err(|_| {
            lance_core::Error::internal("benchmark index-remap measurement mutex is poisoned")
        })?;
        measurement.row_addresses_remapped = measurement
            .row_addresses_remapped
            .saturating_add(remapped_rows);
        measurement.indices_remapped = measurement
            .indices_remapped
            .saturating_add(remapped.len() as u64);
        measurement.elapsed_ns = measurement.elapsed_ns.saturating_add(elapsed_ns);
        Ok(remapped)
    }
}

fn fragment_file_bytes<'a>(
    fragments: impl IntoIterator<Item = &'a lance_table::format::Fragment>,
) -> Option<u64> {
    fragments
        .into_iter()
        .flat_map(|fragment| &fragment.files)
        .try_fold(0_u64, |total, file| {
            total
                .checked_add(file.file_size_bytes.get()?.get())
                .or(None)
        })
}

fn ordered_physical_fragments(dataset: &Dataset) -> BenchResult<Vec<Fragment>> {
    let mut fragments = dataset.manifest().fragments.as_ref().clone();
    fragments.sort_by(maintenance::policy_fragment_order);
    if fragments.iter().any(|fragment| fragment.files.is_empty()) {
        return Err("physical maintenance cannot order a fragment without data files".into());
    }
    Ok(fragments)
}

fn fragment_live_rows(fragment: &Fragment) -> BenchResult<usize> {
    fragment
        .num_rows()
        .ok_or_else(|| "physical maintenance requires exact live row counts".into())
}

fn fragment_physical_rows(fragment: &Fragment) -> BenchResult<usize> {
    fragment
        .physical_rows
        .ok_or_else(|| "physical maintenance requires exact physical row counts".into())
}

fn fragment_data_bytes(fragment: &Fragment) -> BenchResult<u64> {
    fragment_file_bytes(std::iter::once(fragment))
        .ok_or_else(|| "physical maintenance requires cached data-file sizes".into())
}

fn fragment_live_data_bytes(fragment: &Fragment) -> BenchResult<u64> {
    let physical_rows = fragment_physical_rows(fragment)? as u128;
    if physical_rows == 0 {
        return Ok(0);
    }
    let live_bytes = (fragment_data_bytes(fragment)? as u128)
        .saturating_mul(fragment_live_rows(fragment)? as u128)
        .div_ceil(physical_rows);
    u64::try_from(live_bytes).map_err(|_| "maintenance live data byte estimate exceeds u64".into())
}

fn maintenance_execution_target_rows(
    target_file_size_bytes: usize,
    live_rows: usize,
    live_data_bytes: u64,
) -> BenchResult<usize> {
    if live_data_bytes == 0 {
        return Ok(live_rows.max(1));
    }
    let target = (target_file_size_bytes as u128).saturating_mul(live_rows as u128)
        / live_data_bytes as u128;
    usize::try_from(target.min(live_rows.max(1) as u128).max(1))
        .map_err(|_| "maintenance target row count exceeds usize".into())
}

fn maintenance_output_live_rows(live_rows: usize, target_rows: usize) -> Vec<usize> {
    let mut outputs = vec![target_rows; live_rows / target_rows];
    let remainder = live_rows % target_rows;
    if remainder != 0 {
        outputs.push(remainder);
    }
    if outputs.is_empty() {
        outputs.push(0);
    }
    outputs
}

fn build_maintenance_plan(args: &Args, dataset: &Dataset) -> BenchResult<MaintenancePlanArtifact> {
    let fragments = ordered_physical_fragments(dataset)?;
    if fragments.len() > args.max_source_fragments_per_group {
        return Err(format!(
            "maintenance source has {} fragments, exceeding the frozen group limit {}",
            fragments.len(),
            args.max_source_fragments_per_group
        )
        .into());
    }
    let live_rows = fragments.iter().try_fold(0_usize, |total, fragment| {
        total
            .checked_add(fragment_live_rows(fragment)?)
            .ok_or_else(|| -> BenchError { "maintenance plan live row count overflow".into() })
    })?;
    let physical_rows = fragments.iter().try_fold(0_usize, |total, fragment| {
        total
            .checked_add(fragment_physical_rows(fragment)?)
            .ok_or_else(|| -> BenchError { "maintenance plan physical row count overflow".into() })
    })?;
    let physical_data_bytes = fragments.iter().try_fold(0_u64, |total, fragment| {
        total
            .checked_add(fragment_data_bytes(fragment)?)
            .ok_or_else(|| -> BenchError { "maintenance plan data byte count overflow".into() })
    })?;
    let live_data_bytes = fragments.iter().try_fold(0_u64, |total, fragment| {
        total
            .checked_add(fragment_live_data_bytes(fragment)?)
            .ok_or_else(|| -> BenchError {
                "maintenance plan live data byte count overflow".into()
            })
    })?;
    let execution_target_rows_per_fragment =
        maintenance_execution_target_rows(args.target_file_size_bytes, live_rows, live_data_bytes)?;
    let expected_output_live_rows =
        maintenance_output_live_rows(live_rows, execution_target_rows_per_fragment);
    let expected_output_fragment_count = expected_output_live_rows.len();
    let groups = vec![MaintenancePlanGroup {
        start_ordinal: 0,
        end_ordinal: fragments.len(),
        source_live_rows: live_rows,
        source_physical_rows: physical_rows,
        source_physical_data_bytes: physical_data_bytes,
        source_live_data_bytes: live_data_bytes,
        expected_output_fragments: expected_output_fragment_count,
    }];
    Ok(MaintenancePlanArtifact {
        schema_version: 1,
        suite: "stable_row_address_physical_maintenance_plan".to_string(),
        run_id: args.run_id.clone(),
        commit: args.commit.clone(),
        policy_sha256: args.policy_sha256.clone(),
        source_format: args.format.name().to_string(),
        source_dataset_uri: args.dataset_uri.clone(),
        source_dataset_version: dataset.version().version,
        schema_kind: args.schema_kind.name().to_string(),
        expected_rows: args.expected_rows,
        target_rows_per_fragment: args.target_rows_per_fragment,
        execution_target_rows_per_fragment,
        target_file_size_bytes: args.target_file_size_bytes,
        max_source_fragments_per_group: args.max_source_fragments_per_group,
        fragment_count: fragments.len(),
        groups,
        expected_output_live_rows,
        expected_output_fragment_count,
    })
}

fn validate_maintenance_plan(
    args: &Args,
    plan: &MaintenancePlanArtifact,
    dataset: &Dataset,
) -> BenchResult<Vec<TaskData>> {
    if plan.schema_version != 1
        || plan.suite != "stable_row_address_physical_maintenance_plan"
        || plan.run_id != args.run_id
        || plan.commit != args.commit
        || plan.policy_sha256 != args.policy_sha256
        || plan.schema_kind != args.schema_kind.name()
        || plan.expected_rows != args.expected_rows
        || plan.target_rows_per_fragment != args.target_rows_per_fragment
        || plan.execution_target_rows_per_fragment == 0
        || plan.target_file_size_bytes != args.target_file_size_bytes
        || plan.max_source_fragments_per_group != args.max_source_fragments_per_group
        || plan.groups.len() != 1
        || plan.expected_output_live_rows
            != maintenance_output_live_rows(
                plan.expected_rows,
                plan.execution_target_rows_per_fragment,
            )
        || !matches!(
            plan.source_format.as_str(),
            "v22_no_stable" | "v22_stable" | "v23_logical"
        )
        || plan.source_dataset_uri.is_empty()
        || plan.source_dataset_version == 0
    {
        return Err(
            "maintenance plan provenance or frozen parameters do not match invocation".into(),
        );
    }
    let fragments = ordered_physical_fragments(dataset)?;
    if fragments.len() != plan.fragment_count {
        return Err(format!(
            "maintenance plan expected {} source fragments, found {}",
            plan.fragment_count,
            fragments.len()
        )
        .into());
    }
    let mut expected_start = 0_usize;
    let mut expected_outputs = 0_usize;
    let mut planned_live_rows = 0_usize;
    let mut tasks = Vec::with_capacity(plan.groups.len());
    for group in &plan.groups {
        if group.start_ordinal != expected_start
            || group.end_ordinal <= group.start_ordinal
            || group.end_ordinal > fragments.len()
            || group.end_ordinal - group.start_ordinal > plan.max_source_fragments_per_group
        {
            return Err("maintenance plan groups are not a contiguous bounded partition".into());
        }
        let sources = &fragments[group.start_ordinal..group.end_ordinal];
        let live_rows = sources.iter().try_fold(0_usize, |total, fragment| {
            total
                .checked_add(fragment_live_rows(fragment)?)
                .ok_or_else(|| -> BenchError {
                    "maintenance validation live row count overflow".into()
                })
        })?;
        let physical_rows = sources.iter().try_fold(0_usize, |total, fragment| {
            total
                .checked_add(fragment_physical_rows(fragment)?)
                .ok_or_else(|| -> BenchError {
                    "maintenance validation physical row count overflow".into()
                })
        })?;
        let physical_data_bytes = sources.iter().try_fold(0_u64, |total, fragment| {
            total
                .checked_add(fragment_data_bytes(fragment)?)
                .ok_or_else(|| -> BenchError {
                    "maintenance validation physical byte count overflow".into()
                })
        })?;
        let live_data_bytes = sources.iter().try_fold(0_u64, |total, fragment| {
            total
                .checked_add(fragment_live_data_bytes(fragment)?)
                .ok_or_else(|| -> BenchError {
                    "maintenance validation live byte count overflow".into()
                })
        })?;
        if live_rows != group.source_live_rows || physical_rows != group.source_physical_rows {
            return Err(format!(
                "maintenance plan source postcondition mismatch for ordinals {}..{}: \
                 live {live_rows}/{}, physical {physical_rows}/{}",
                group.start_ordinal,
                group.end_ordinal,
                group.source_live_rows,
                group.source_physical_rows
            )
            .into());
        }
        if args.format.name() == plan.source_format
            && args.dataset_uri == plan.source_dataset_uri
            && (dataset.version().version != plan.source_dataset_version
                || physical_data_bytes != group.source_physical_data_bytes
                || live_data_bytes != group.source_live_data_bytes)
        {
            return Err(format!(
                "maintenance source bytes or version changed after plan preparation: version {}/{}, physical bytes {physical_data_bytes}/{}, live bytes {live_data_bytes}/{}",
                dataset.version().version,
                plan.source_dataset_version,
                group.source_physical_data_bytes,
                group.source_live_data_bytes
            )
            .into());
        }
        let expected_group_outputs = live_rows
            .div_ceil(plan.execution_target_rows_per_fragment)
            .max(1);
        if group.expected_output_fragments != expected_group_outputs {
            return Err(format!(
                "maintenance plan group {}..{} expected {} outputs, byte target requires {expected_group_outputs}",
                group.start_ordinal,
                group.end_ordinal,
                group.expected_output_fragments
            )
            .into());
        }
        expected_outputs = expected_outputs
            .checked_add(expected_group_outputs)
            .ok_or("maintenance validation output fragment count overflow")?;
        planned_live_rows = planned_live_rows
            .checked_add(live_rows)
            .ok_or("maintenance validation live row count overflow")?;
        tasks.push(TaskData {
            fragments: maintenance::fragments_for_native_compaction(
                sources,
                args.format == BenchFormat::V22NoStable,
            ),
            v2_3_plan: None,
        });
        expected_start = group.end_ordinal;
    }
    if expected_start != fragments.len()
        || tasks.is_empty()
        || planned_live_rows != plan.expected_rows
        || expected_outputs != plan.expected_output_fragment_count
        || plan.execution_target_rows_per_fragment
            != maintenance_execution_target_rows(
                plan.target_file_size_bytes,
                plan.expected_rows,
                plan.groups[0].source_live_data_bytes,
            )?
    {
        return Err("maintenance plan does not cover every source fragment exactly once".into());
    }
    Ok(tasks)
}

fn read_maintenance_plan(args: &Args, dataset: &Dataset) -> BenchResult<MaintenancePlanArtifact> {
    let path = args
        .maintenance_plan_input
        .as_ref()
        .ok_or("relocation operation is missing its maintenance plan")?;
    let plan: MaintenancePlanArtifact = serde_json::from_slice(&fs::read(path)?)?;
    validate_maintenance_plan(args, &plan, dataset)?;
    Ok(plan)
}

fn verify_maintenance_output(
    dataset: &Dataset,
    plan: Option<&MaintenancePlanArtifact>,
) -> BenchResult<()> {
    let Some(plan) = plan else {
        return Ok(());
    };
    let fragments = ordered_physical_fragments(dataset)?;
    let actual_fragments = fragments.len();
    if actual_fragments != plan.expected_output_fragment_count {
        return Err(format!(
            "maintenance plan expected {} output fragments, operation produced {actual_fragments}",
            plan.expected_output_fragment_count
        )
        .into());
    }
    let live_rows = dataset
        .manifest()
        .fragments
        .iter()
        .try_fold(0_usize, |total, fragment| {
            total
                .checked_add(fragment_live_rows(fragment)?)
                .ok_or_else(|| -> BenchError {
                    "maintenance output live row count overflow".into()
                })
        })?;
    if live_rows != plan.expected_rows {
        return Err(format!(
            "maintenance plan expected {} live rows, operation produced {live_rows}",
            plan.expected_rows
        )
        .into());
    }
    let actual_rows = fragments
        .iter()
        .map(fragment_live_rows)
        .collect::<BenchResult<Vec<_>>>()?;
    if actual_rows != plan.expected_output_live_rows {
        return Err(format!(
            "maintenance output row boundaries differ from frozen plan: expected={:?}, actual={actual_rows:?}",
            plan.expected_output_live_rows
        )
        .into());
    }
    for fragment in &fragments {
        if fragment_physical_rows(fragment)? != fragment_live_rows(fragment)? {
            return Err("maintenance output retained deleted physical rows".into());
        }
    }
    Ok(())
}

async fn write_maintenance_plan_artifact(
    args: &Args,
    output: &Path,
    tracker: Arc<IOTracker>,
) -> BenchResult<()> {
    let dataset = open_dataset(args, tracker).await?;
    args.format.validate_dataset(&dataset)?;
    let plan = build_maintenance_plan(args, &dataset)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> BenchResult<()> {
        serde_json::to_writer(&mut file, &plan)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        if output.exists() {
            return Err(format!(
                "physical-maintenance plan already exists: {}",
                output.display()
            )
            .into());
        }
        fs::rename(&temporary, output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

async fn index_storage_bytes(dataset: &Dataset) -> BenchResult<Option<u64>> {
    let indices = dataset.load_indices().await?;
    let mut total = 0_u64;
    for index in indices.iter() {
        let Some(files) = &index.files else {
            return Ok(None);
        };
        for file in files {
            total = total
                .checked_add(file.size_bytes)
                .ok_or("index storage byte total overflow")?;
        }
    }
    Ok(Some(total))
}

fn covered_live_rows_for_fragments(
    dataset: &Dataset,
    covered_fragments: &roaring::RoaringBitmap,
) -> BenchResult<u64> {
    dataset
        .manifest()
        .fragments
        .iter()
        .filter(|fragment| {
            u32::try_from(fragment.id)
                .ok()
                .is_some_and(|id| covered_fragments.contains(id))
        })
        .try_fold(0_u64, |total, fragment| {
            total
                .checked_add(fragment_live_rows(fragment)? as u64)
                .ok_or_else(|| -> BenchError { "index coverage live row count overflow".into() })
        })
}

async fn effective_index_coverage(dataset: &Dataset, index_kind: IndexKind) -> BenchResult<f64> {
    let live_rows = dataset
        .manifest()
        .fragments
        .iter()
        .try_fold(0_u64, |total, fragment| {
            total
                .checked_add(fragment_live_rows(fragment)? as u64)
                .ok_or_else(|| -> BenchError { "dataset live row count overflow".into() })
        })?;
    if live_rows == 0 {
        return Ok(1.0);
    }
    let index_name = benchmark_index_name(index_kind);
    if index_kind == IndexKind::None {
        return Err("effective index coverage requires an index".into());
    }
    let metadata = dataset.load_indices_by_name(index_name).await?;
    let covered_rows = if dataset.manifest().uses_stable_logical_row_addresses() {
        evidence::effective_logical_index_covered_rows(dataset, &metadata, live_rows).await?
    } else {
        let mut covered_fragments = roaring::RoaringBitmap::new();
        for segment in &metadata {
            let bitmap = segment.fragment_bitmap.as_ref().ok_or_else(|| {
                "legacy index segment is missing physical fragment coverage".to_string()
            })?;
            covered_fragments |= bitmap;
        }
        covered_live_rows_for_fragments(dataset, &covered_fragments)?
    };
    if covered_rows > live_rows {
        return Err(format!(
            "effective index coverage contains {covered_rows} rows for a {live_rows}-row dataset"
        )
        .into());
    }
    Ok(covered_rows as f64 / live_rows as f64)
}

struct CompactionObservation {
    metrics: CompactionMetrics,
    compacted_data_bytes: Option<u64>,
    index_storage_bytes_before: Option<u64>,
    row_addresses_remapped: u64,
    indices_remapped: u64,
    index_coverage_reuse: Option<f64>,
    layout_index_maintenance_ns: u64,
    fragment_reuse_index_present: bool,
}

fn default_compaction_options(args: &Args) -> CompactionOptions {
    CompactionOptions {
        target_rows_per_fragment: args.target_rows_per_fragment,
        max_rows_per_group: args.rows_per_fragment,
        num_threads: Some(1),
        materialize_deletions: true,
        materialize_deletions_threshold: 0.0,
        defer_index_remap: args.compaction_mode == CompactionMode::FragmentReuse
            && args.format == BenchFormat::V22NoStable,
        ..Default::default()
    }
}

async fn compact_files_with_observation(
    args: &Args,
    dataset: &mut Dataset,
    mut options: CompactionOptions,
    maintenance_plan: Option<&MaintenancePlanArtifact>,
) -> BenchResult<CompactionObservation> {
    let indices_before = dataset.load_indices().await?;
    let index_storage_bytes_before = index_storage_bytes(dataset).await?;
    let mut layout_index_maintenance_ns = 0_u64;
    let plan = if let Some(maintenance_plan) = maintenance_plan {
        options.target_rows_per_fragment = maintenance_plan.execution_target_rows_per_fragment;
        // Validate and bind the benchmark-owned physical plan before any
        // maintenance write. A v2.2 index transaction requires a contiguous
        // rewrite group to be either fully indexed or fully unindexed. Repeated
        // catch-up creates multiple complete same-name segments, so consolidate
        // them inside this timed maintenance operation and then prove the
        // physical source is byte-for-byte unchanged.
        let tasks = validate_maintenance_plan(args, maintenance_plan, dataset)?;
        if matches!(
            args.format,
            BenchFormat::V22NoStable | BenchFormat::V22Stable
        ) && let Some(elapsed) =
            maintenance::consolidate_legacy_index_segments(dataset, indices_before.as_ref()).await?
        {
            layout_index_maintenance_ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        }
        CompactionPlan {
            tasks,
            read_version: dataset.version().version,
            options: options.clone(),
            planning_metrics: CompactionMetrics::default(),
        }
    } else {
        plan_compaction(dataset, &options).await?
    };
    let compacted_data_bytes =
        fragment_file_bytes(plan.tasks.iter().flat_map(|task| task.fragments.iter()));
    if plan.read_version() != dataset.version().version {
        return Err(format!(
            "compaction plan read version {} differs from dataset version {}",
            plan.read_version(),
            dataset.version().version
        )
        .into());
    }
    let (completed, execution_planning_metrics) = execute_compaction_plan_tasks(dataset, &plan)
        .await
        .map_err(|error| {
            format!(
                "compaction plan read version {}, dataset version {}: {error}",
                plan.read_version(),
                dataset.version().version
            )
        })?;
    let measurement = Arc::new(Mutex::new(RemapMeasurement::default()));
    let commit_started = Instant::now();
    let mut metrics = commit_compaction(
        dataset,
        completed,
        Arc::new(MeasuredIndexRemapperOptions {
            measurement: measurement.clone(),
        }),
        plan.options(),
    )
    .await?;
    layout_index_maintenance_ns = layout_index_maintenance_ns
        .saturating_add(u64::try_from(commit_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    metrics += plan.planning_metrics;
    metrics += execution_planning_metrics;
    let indices_after = dataset.load_indices().await?;
    let fragment_reuse_index_present = indices_after
        .iter()
        .any(|index| index.name == FRAG_REUSE_INDEX_NAME);
    let index_coverage_reuse = if indices_before.is_empty() {
        None
    } else {
        let reused = indices_before
            .iter()
            .filter(|before| indices_after.iter().any(|after| after.uuid == before.uuid))
            .count();
        Some(reused as f64 / indices_before.len() as f64)
    };
    let measurement = measurement.lock().map_err(|_| {
        lance_core::Error::internal("benchmark index-remap measurement mutex is poisoned")
    })?;
    if let Some(maintenance_plan) = maintenance_plan
        && dataset.get_fragments().len() != maintenance_plan.expected_output_fragment_count
    {
        return Err(format!(
            "maintenance plan expected {} output fragments, compaction produced {}",
            maintenance_plan.expected_output_fragment_count,
            dataset.get_fragments().len()
        )
        .into());
    }
    Ok(CompactionObservation {
        metrics,
        compacted_data_bytes,
        index_storage_bytes_before,
        row_addresses_remapped: measurement.row_addresses_remapped,
        indices_remapped: measurement.indices_remapped,
        index_coverage_reuse,
        layout_index_maintenance_ns,
        fragment_reuse_index_present,
    })
}

async fn rewrite_files_in_order_with_observation(
    args: &Args,
    dataset: &mut Dataset,
    mut options: CompactionOptions,
    maintenance_plan: &MaintenancePlanArtifact,
) -> BenchResult<CompactionObservation> {
    validate_maintenance_plan(args, maintenance_plan, dataset)?;
    options.target_rows_per_fragment = maintenance_plan.execution_target_rows_per_fragment;
    let indices_before = dataset.load_indices().await?;
    let index_storage_bytes_before = index_storage_bytes(dataset).await?;
    let compacted_data_bytes = fragment_file_bytes(dataset.manifest().fragments.iter());
    let measurement = Arc::new(Mutex::new(RemapMeasurement::default()));
    let metrics = rewrite_files_in_order(
        dataset,
        vec![ColumnOrdering::asc_nulls_first("id".to_string())],
        options,
        Some(Arc::new(MeasuredIndexRemapperOptions {
            measurement: measurement.clone(),
        })),
    )
    .await?;
    let indices_after = dataset.load_indices().await?;
    let fragment_reuse_index_present = indices_after
        .iter()
        .any(|index| index.name == FRAG_REUSE_INDEX_NAME);
    let index_coverage_reuse = if indices_before.is_empty() {
        None
    } else {
        let reused = indices_before
            .iter()
            .filter(|before| indices_after.iter().any(|after| after.uuid == before.uuid))
            .count();
        Some(reused as f64 / indices_before.len() as f64)
    };
    let measurement = measurement.lock().map_err(|_| {
        lance_core::Error::internal("benchmark index-remap measurement mutex is poisoned")
    })?;
    Ok(CompactionObservation {
        metrics,
        compacted_data_bytes,
        index_storage_bytes_before,
        row_addresses_remapped: measurement.row_addresses_remapped,
        indices_remapped: measurement.indices_remapped,
        index_coverage_reuse,
        layout_index_maintenance_ns: measurement.elapsed_ns,
        fragment_reuse_index_present,
    })
}

async fn execute(
    args: &Args,
    prepared: PreparedOperation,
    tracker: Arc<IOTracker>,
) -> BenchResult<ExecutedOperation> {
    match args.operation {
        Operation::Create => {
            let PreparedOperation::Batches(source) = prepared else {
                return Err("create operation is missing its synthetic source".into());
            };
            if args.storage == Storage::Ebs && Path::new(&args.dataset_uri).exists() {
                return Err(format!(
                    "create refuses to overwrite existing dataset: {}",
                    args.dataset_uri
                )
                .into());
            }
            let params = WriteParams {
                max_rows_per_file: args.rows_per_fragment,
                max_rows_per_group: args.rows_per_fragment.min(8_192),
                mode: WriteMode::Create,
                data_storage_version: Some(args.format.storage_version()),
                enable_stable_row_ids: args.format.enable_legacy_stable_row_ids(),
                store_params: Some(store_params(tracker)),
                ..Default::default()
            };
            let dataset =
                Dataset::write(source.into_reader(), &args.dataset_uri, Some(params)).await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.rows as u64),
                    rows_inserted: Some(args.rows as u64),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::FixtureClone => {
            if args.storage == Storage::Ebs && Path::new(&args.dataset_uri).exists() {
                return Err(format!(
                    "fixture-clone refuses to overwrite existing dataset: {}",
                    args.dataset_uri
                )
                .into());
            }
            let source_uri = args
                .source_dataset_uri
                .as_deref()
                .ok_or("validated fixture-clone source is missing")?;
            let mut source = open_dataset_uri(source_uri, tracker.clone()).await?;
            args.format.validate_dataset(&source)?;
            let source_version = source.version().version;
            let dataset = source
                .shallow_clone(
                    &args.dataset_uri,
                    source_version,
                    Some(store_params(tracker)),
                )
                .await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    coverage: (args.index_kind != IndexKind::None).then_some(1.0),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::Append => {
            let PreparedOperation::Batches(source) = prepared else {
                return Err("append operation is missing its synthetic source".into());
            };
            let params = WriteParams {
                max_rows_per_file: args.rows_per_fragment,
                max_rows_per_group: args.rows_per_fragment.min(8_192),
                mode: WriteMode::Append,
                data_storage_version: Some(args.format.storage_version()),
                enable_stable_row_ids: args.format.enable_legacy_stable_row_ids(),
                store_params: Some(store_params(tracker)),
                ..Default::default()
            };
            let dataset =
                Dataset::write(source.into_reader(), &args.dataset_uri, Some(params)).await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    rows_inserted: Some(args.mutation_count as u64),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::Delete => {
            let mut dataset = open_dataset(args, tracker).await?;
            args.format.validate_dataset(&dataset)?;
            let predicate = mutation_predicate(args)?;
            let deleted = dataset.delete(&predicate).await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    rows_deleted: Some(deleted.num_deleted_rows),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::Update => match args.update_driver {
            UpdateDriver::Native => {
                let dataset = open_dataset(args, tracker).await?;
                args.format.validate_dataset(&dataset)?;
                let predicate = mutation_predicate(args)?;
                let result = UpdateBuilder::new(Arc::new(dataset))
                    .update_where(&predicate)?
                    .set("value", &format!("value + {}", args.step + 1))?
                    .build()?
                    .execute()
                    .await?;
                Ok(ExecutedOperation::new(
                    result.new_dataset,
                    OperationOutcome {
                        result_rows: Some(args.expected_rows as u64),
                        rows_updated: Some(result.rows_updated),
                        admission: Some(true),
                        ..Default::default()
                    },
                ))
            }
            UpdateDriver::ExactMatchedMerge => {
                let PreparedOperation::Batches(source) = prepared else {
                    return Err("exact update operation is missing its selected source".into());
                };
                let dataset = Arc::new(open_dataset(args, tracker).await?);
                args.format.validate_dataset(dataset.as_ref())?;
                let mut builder = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?;
                builder
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::DoNothing);
                let (dataset, stats) = builder
                    .try_build()?
                    .execute_reader(
                        Box::new(source.into_reader()) as Box<dyn RecordBatchReader + Send>
                    )
                    .await?;
                Ok(ExecutedOperation::new(
                    dataset,
                    OperationOutcome {
                        result_rows: Some(args.expected_rows as u64),
                        rows_inserted: Some(stats.num_inserted_rows),
                        rows_updated: Some(stats.num_updated_rows),
                        rows_deleted: Some(stats.num_deleted_rows),
                        admission: Some(true),
                        ..Default::default()
                    },
                ))
            }
        },
        Operation::MergeInsert => {
            let PreparedOperation::Batches(source) = prepared else {
                return Err("merge_insert operation is missing its synthetic source".into());
            };
            let dataset = Arc::new(open_dataset(args, tracker).await?);
            args.format.validate_dataset(dataset.as_ref())?;
            let mut builder = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?;
            builder
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll);
            let (dataset, stats) = builder
                .try_build()?
                .execute_reader(Box::new(source.into_reader()) as Box<dyn RecordBatchReader + Send>)
                .await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    rows_inserted: Some(stats.num_inserted_rows),
                    rows_updated: Some(stats.num_updated_rows),
                    rows_deleted: Some(stats.num_deleted_rows),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::Backfill => {
            let mut dataset = open_dataset(args, tracker).await?;
            args.format.validate_dataset(&dataset)?;
            let expressions = (0..args.mutation_count)
                .map(|column| {
                    (
                        format!("backfill_{:03}_{column:03}", args.step),
                        format!("id + {}", args.step + column + 1),
                    )
                })
                .collect();
            dataset
                .add_columns(
                    NewColumnTransform::SqlExpressions(expressions),
                    None,
                    Some(args.rows_per_fragment.min(u32::MAX as usize) as u32),
                )
                .await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    rows_updated: Some(args.expected_rows as u64),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::DefaultCompactionPreflight => {
            let dataset = open_dataset(args, tracker).await?;
            // Cold open already validates the persisted v2.3 contract, and
            // finish_outcome performs full post-measurement validation.
            args.format.validate_dataset_header(&dataset)?;
            let source_version = dataset.version().version;
            let plan = plan_compaction(&dataset, &default_compaction_options(args)).await?;
            if plan.read_version() != source_version || dataset.version().version != source_version
            {
                return Err(format!(
                    "plan-only default compaction changed or mismatched source version: source={source_version}, plan={}, current={}",
                    plan.read_version(),
                    dataset.version().version
                )
                .into());
            }
            let admitted = plan.num_tasks();
            let rejected_planned = plan.planning_metrics.groups_planned;
            let not_admitted = plan.planning_metrics.groups_not_admitted;
            if rejected_planned != not_admitted {
                return Err(format!(
                    "default compaction preflight rejection counts disagree: planned={rejected_planned}, not_admitted={not_admitted}"
                )
                .into());
            }
            let planned = admitted
                .checked_add(rejected_planned)
                .ok_or("default compaction preflight group count overflow")?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    compaction_groups_planned: Some(planned as u64),
                    compaction_groups_admitted: Some(admitted as u64),
                    compaction_groups_not_admitted: Some(not_admitted as u64),
                    admission: Some(admitted == planned),
                    ..Default::default()
                },
            ))
        }
        Operation::DefaultCompaction => {
            let mut dataset = open_dataset(args, tracker).await?;
            // Cold open already validates the persisted v2.3 contract, and
            // finish_outcome performs full post-measurement validation. Keep
            // the timed path to the constant-size header check so v2.3 does
            // not pay for the same full manifest validation twice.
            args.format.validate_dataset_header(&dataset)?;
            let maintenance_plan = args
                .maintenance_plan_input
                .as_ref()
                .map(|_| read_maintenance_plan(args, &dataset))
                .transpose()?;
            let observation = compact_files_with_observation(
                args,
                &mut dataset,
                default_compaction_options(args),
                maintenance_plan.as_ref(),
            )
            .await?;
            verify_maintenance_output(&dataset, maintenance_plan.as_ref())?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    compacted_data_bytes: observation.compacted_data_bytes,
                    index_storage_bytes_before: observation.index_storage_bytes_before,
                    row_addresses_remapped: Some(observation.row_addresses_remapped),
                    indices_remapped: Some(observation.indices_remapped),
                    index_coverage_reuse: observation.index_coverage_reuse,
                    layout_index_maintenance_ns: Some(observation.layout_index_maintenance_ns),
                    fragment_reuse_index_present: Some(observation.fragment_reuse_index_present),
                    compaction_groups_planned: Some(observation.metrics.groups_planned as u64),
                    compaction_groups_admitted: Some(observation.metrics.groups_admitted as u64),
                    compaction_groups_not_admitted: Some(
                        observation.metrics.groups_not_admitted as u64,
                    ),
                    coverage: (args.index_kind != IndexKind::None).then_some(0.0),
                    admission: Some(
                        observation.metrics.groups_planned > 0
                            && observation.metrics.groups_admitted
                                == observation.metrics.groups_planned
                            && observation.metrics.groups_not_admitted == 0,
                    ),
                    ..Default::default()
                },
            ))
        }
        Operation::RandomDeleteReclaim => {
            let mut dataset = open_dataset(args, tracker).await?;
            args.format.validate_dataset(&dataset)?;
            let maintenance_plan = args
                .maintenance_plan_input
                .as_ref()
                .map(|_| read_maintenance_plan(args, &dataset))
                .transpose()?;
            let mut outcome = OperationOutcome {
                result_rows: Some(args.expected_rows as u64),
                coverage: (args.index_kind != IndexKind::None).then_some(0.0),
                admission: Some(true),
                ..Default::default()
            };
            if args.format == BenchFormat::V23Logical {
                let index_storage_bytes_before = index_storage_bytes(&dataset).await?;
                let compacted_data_bytes = fragment_file_bytes(dataset.manifest().fragments.iter());
                let indices_before = dataset.load_indices().await?;
                let commit_started = Instant::now();
                let mut options = RowAddressMaintenanceOptions::repack();
                options.target_rows_per_fragment = maintenance_plan
                    .as_ref()
                    .map(|plan| plan.execution_target_rows_per_fragment)
                    .unwrap_or(args.target_rows_per_fragment);
                options.max_rows_per_group = args.rows_per_fragment;
                let metrics = maintain_row_addresses(&mut dataset, options).await?;
                let indices_after = dataset.load_indices().await?;
                outcome.rows_updated = Some(metrics.rows_rewritten);
                outcome.explicit_locator_objects_written =
                    Some(metrics.locator_objects_written as u64);
                outcome.explicit_locator_bytes_written = Some(metrics.locator_bytes_written);
                outcome.compacted_data_bytes = compacted_data_bytes;
                outcome.index_storage_bytes_before = index_storage_bytes_before;
                outcome.row_addresses_remapped = Some(0);
                outcome.indices_remapped = Some(0);
                outcome.index_coverage_reuse = if indices_before.is_empty() {
                    None
                } else {
                    let reused = indices_before
                        .iter()
                        .filter(|before| {
                            indices_after.iter().any(|after| after.uuid == before.uuid)
                        })
                        .count();
                    Some(reused as f64 / indices_before.len() as f64)
                };
                outcome.layout_index_maintenance_ns =
                    Some(u64::try_from(commit_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
                outcome.compaction_groups_planned = Some(1);
                outcome.compaction_groups_admitted = Some(1);
                outcome.compaction_groups_not_admitted = Some(0);
                outcome.fragment_reuse_index_present = Some(false);
            } else {
                let observation = compact_files_with_observation(
                    args,
                    &mut dataset,
                    default_compaction_options(args),
                    maintenance_plan.as_ref(),
                )
                .await?;
                outcome.compacted_data_bytes = observation.compacted_data_bytes;
                outcome.index_storage_bytes_before = observation.index_storage_bytes_before;
                outcome.row_addresses_remapped = Some(observation.row_addresses_remapped);
                outcome.indices_remapped = Some(observation.indices_remapped);
                outcome.index_coverage_reuse = observation.index_coverage_reuse;
                outcome.layout_index_maintenance_ns = Some(observation.layout_index_maintenance_ns);
                outcome.fragment_reuse_index_present =
                    Some(observation.fragment_reuse_index_present);
                outcome.explicit_locator_objects_written = Some(0);
                outcome.explicit_locator_bytes_written = Some(0);
                outcome.compaction_groups_planned = Some(observation.metrics.groups_planned as u64);
                outcome.compaction_groups_admitted =
                    Some(observation.metrics.groups_admitted as u64);
                outcome.compaction_groups_not_admitted =
                    Some(observation.metrics.groups_not_admitted as u64);
                if args.format == BenchFormat::V22Stable && args.index_kind != IndexKind::None {
                    let rebuild_started = Instant::now();
                    rebuild_benchmark_indices(args, &mut dataset).await?;
                    let rebuild_ns =
                        u64::try_from(rebuild_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    outcome.layout_index_maintenance_ns = Some(
                        observation
                            .layout_index_maintenance_ns
                            .checked_add(rebuild_ns)
                            .ok_or("legacy stable index maintenance duration overflow")?,
                    );
                    outcome.index_coverage_reuse = Some(0.0);
                }
            }
            verify_maintenance_output(&dataset, maintenance_plan.as_ref())?;
            Ok(ExecutedOperation::new(dataset, outcome))
        }
        Operation::BoundedRecluster => {
            let mut dataset = open_dataset(args, tracker).await?;
            args.format.validate_dataset(&dataset)?;
            if args.maintenance_plan_input.is_none() {
                return Err("bounded recluster requires a paired maintenance plan".into());
            }
            let maintenance_plan = read_maintenance_plan(args, &dataset)?;
            let observation = rewrite_files_in_order_with_observation(
                args,
                &mut dataset,
                default_compaction_options(args),
                &maintenance_plan,
            )
            .await?;
            verify_maintenance_output(&dataset, Some(&maintenance_plan))?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    rows_updated: Some(args.expected_rows as u64),
                    compacted_data_bytes: observation.compacted_data_bytes,
                    index_storage_bytes_before: observation.index_storage_bytes_before,
                    row_addresses_remapped: Some(observation.row_addresses_remapped),
                    indices_remapped: Some(observation.indices_remapped),
                    index_coverage_reuse: observation.index_coverage_reuse,
                    layout_index_maintenance_ns: Some(observation.layout_index_maintenance_ns),
                    fragment_reuse_index_present: Some(observation.fragment_reuse_index_present),
                    explicit_locator_objects_written: Some(0),
                    explicit_locator_bytes_written: Some(0),
                    compaction_groups_planned: Some(observation.metrics.groups_planned as u64),
                    compaction_groups_admitted: Some(observation.metrics.groups_admitted as u64),
                    compaction_groups_not_admitted: Some(
                        observation.metrics.groups_not_admitted as u64,
                    ),
                    coverage: (args.index_kind != IndexKind::None).then_some(0.0),
                    admission: Some(
                        observation.metrics.groups_planned > 0
                            && observation.metrics.groups_admitted
                                == observation.metrics.groups_planned
                            && observation.metrics.groups_not_admitted == 0,
                    ),
                    ..Default::default()
                },
            ))
        }
        Operation::NormalizePlacement
        | Operation::Repack
        | Operation::Recluster
        | Operation::CheckpointGeneration => {
            let mut dataset = open_dataset(args, tracker).await?;
            args.format.validate_dataset(&dataset)?;
            if args.format != BenchFormat::V23Logical {
                return Err(format!(
                    "explicit maintenance capability '{}' requires v23_logical",
                    args.operation.name()
                )
                .into());
            }
            let mut options =
                match args.operation {
                    Operation::NormalizePlacement => {
                        RowAddressMaintenanceOptions::normalize_placement()
                    }
                    Operation::Repack => RowAddressMaintenanceOptions::repack(),
                    Operation::Recluster => RowAddressMaintenanceOptions::recluster(vec![
                        ColumnOrdering::asc_nulls_first("value".to_string()),
                    ]),
                    Operation::CheckpointGeneration => {
                        RowAddressMaintenanceOptions::checkpoint_generation()
                    }
                    _ => unreachable!(),
                };
            let maintenance_plan = args
                .maintenance_plan_input
                .as_ref()
                .map(|_| read_maintenance_plan(args, &dataset))
                .transpose()?;
            options.target_rows_per_fragment = maintenance_plan
                .as_ref()
                .map(|plan| plan.execution_target_rows_per_fragment)
                .unwrap_or(args.target_rows_per_fragment);
            options.max_rows_per_group = args.rows_per_fragment;
            let disclose_explicit_cost =
                matches!(args.operation, Operation::Repack | Operation::Recluster);
            let compacted_data_bytes = disclose_explicit_cost
                .then(|| fragment_file_bytes(dataset.manifest().fragments.iter()))
                .flatten();
            let index_storage_bytes_before = if disclose_explicit_cost {
                index_storage_bytes(&dataset).await?
            } else {
                None
            };
            let indices_before = if disclose_explicit_cost {
                Some(dataset.load_indices().await?)
            } else {
                None
            };
            let commit_started = Instant::now();
            let metrics = maintain_row_addresses(&mut dataset, options).await?;
            let layout_index_maintenance_ns =
                u64::try_from(commit_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            verify_maintenance_output(&dataset, maintenance_plan.as_ref())?;
            let indices_after = if disclose_explicit_cost {
                Some(dataset.load_indices().await?)
            } else {
                None
            };
            let index_coverage_reuse = indices_before.as_ref().and_then(|before| {
                if before.is_empty() {
                    return None;
                }
                let after = indices_after
                    .as_ref()
                    .expect("explicit cost loads both index sets");
                let reused = before
                    .iter()
                    .filter(|index| after.iter().any(|candidate| candidate.uuid == index.uuid))
                    .count();
                Some(reused as f64 / before.len() as f64)
            });
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    rows_updated: Some(metrics.rows_rewritten),
                    compacted_data_bytes,
                    index_storage_bytes_before,
                    row_addresses_remapped: disclose_explicit_cost.then_some(0),
                    indices_remapped: disclose_explicit_cost.then_some(0),
                    index_coverage_reuse,
                    layout_index_maintenance_ns: disclose_explicit_cost
                        .then_some(layout_index_maintenance_ns),
                    fragment_reuse_index_present: disclose_explicit_cost.then_some(false),
                    explicit_locator_objects_written: disclose_explicit_cost
                        .then_some(metrics.locator_objects_written as u64),
                    explicit_locator_bytes_written: disclose_explicit_cost
                        .then_some(metrics.locator_bytes_written),
                    coverage: (args.index_kind != IndexKind::None).then_some(0.0),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::IndexBuild => {
            let mut dataset = open_dataset(args, tracker).await?;
            args.format.validate_dataset(&dataset)?;
            rebuild_benchmark_indices(args, &mut dataset).await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    coverage: Some(1.0),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::IndexTake => {
            let PreparedOperation::Take { mut user_ids, .. } = prepared else {
                return Err("index take is missing prepared live user IDs".into());
            };
            let dataset = open_dataset(args, tracker).await?;
            let (result_rows, recall) = match args.index_kind {
                IndexKind::None => unreachable!("validated before execution"),
                IndexKind::Scalar => {
                    user_ids.sort_unstable();
                    let predicate = format!(
                        "id IN ({})",
                        user_ids
                            .iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    let result_rows = dataset
                        .scan()
                        .filter(&predicate)?
                        .project(&["id", "value"])?
                        .try_into_batch()
                        .await?;
                    let returned_ids = result_rows
                        .column_by_name("id")
                        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
                        .ok_or("scalar index take did not return the UInt64 id column")?;
                    let expected = user_ids.into_iter().collect::<HashSet<_>>();
                    let returned = returned_ids.iter().flatten().collect::<HashSet<_>>();
                    let matched = expected.intersection(&returned).count();
                    (
                        result_rows.num_rows(),
                        matched as f64 / args.take_count as f64,
                    )
                }
                IndexKind::Vector => {
                    // The untimed fixture scan selected a live business ID.
                    // Reconstruct its vector deterministically so the measured
                    // path contains only the indexed query.
                    let query_id = user_ids[0];
                    let query =
                        Float32Array::from_iter_values((0..VECTOR_DIMENSION).map(|dimension| {
                            let bits = mix64(
                                query_id
                                    ^ args.seed
                                    ^ (dimension as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
                            );
                            ((bits >> 40) as u32) as f32 / (u32::MAX >> 8) as f32
                        }));
                    let mut scanner = dataset.scan();
                    scanner.nearest("vector", &query, args.take_count)?;
                    let batch = scanner.try_into_batch().await?;
                    let ids = batch
                        .column_by_name("id")
                        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
                        .ok_or("vector index take did not return the UInt64 id column")?;
                    let exact_top_one_found = !ids.is_empty() && ids.value(0) == query_id;
                    (
                        batch.num_rows(),
                        if exact_top_one_found { 1.0 } else { 0.0 },
                    )
                }
            };
            if result_rows != args.take_count {
                return Err(format!(
                    "index take returned {result_rows} rows, expected {}",
                    args.take_count
                )
                .into());
            }
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(result_rows as u64),
                    coverage: Some(1.0),
                    recall: Some(recall),
                    ..Default::default()
                },
            ))
        }
        Operation::IndexOptimize => {
            let mut dataset = open_dataset(args, tracker).await?;
            args.format.validate_dataset(&dataset)?;
            dataset
                .optimize_indices(&OptimizeOptions::default())
                .await?;
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(args.expected_rows as u64),
                    coverage: Some(1.0),
                    admission: Some(true),
                    ..Default::default()
                },
            ))
        }
        Operation::Open => {
            let dataset = open_dataset(args, tracker).await?;
            Ok(ExecutedOperation::new(dataset, OperationOutcome::default()))
        }
        Operation::Scan => {
            let dataset = open_dataset(args, tracker).await?;
            let mut stream = dataset.scan().try_into_stream().await?;
            let mut result_rows = 0u64;
            let mut digest = StateDigest::default();
            while let Some(batch) = stream.try_next().await? {
                result_rows = result_rows
                    .checked_add(batch.num_rows() as u64)
                    .ok_or("scan row count overflow")?;
                digest.update(&batch)?;
            }
            if result_rows != args.expected_rows as u64 {
                return Err(format!(
                    "scan returned {result_rows} rows, expected {}",
                    args.expected_rows
                )
                .into());
            }
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(result_rows),
                    state_digest: Some(digest.finish()),
                    physical_order_digest: Some(digest.finish_ordered()),
                    ..Default::default()
                },
            ))
        }
        Operation::RowIdScan => {
            let output = args
                .prepare_take_ids_output
                .as_deref()
                .ok_or("validated row-id-scan output is missing")?;
            write_take_ids_artifact(args, output, tracker).await
        }
        Operation::Take => {
            let PreparedOperation::Take { row_ids, .. } = prepared else {
                return Err("take operation is missing prepared row IDs".into());
            };
            let dataset = open_dataset(args, tracker).await?;
            let batch = dataset
                .take_rows(&row_ids, dataset.schema().clone())
                .await?;
            if batch.num_rows() != args.take_count {
                return Err(format!(
                    "take returned {} rows, expected {}",
                    batch.num_rows(),
                    args.take_count
                )
                .into());
            }
            Ok(ExecutedOperation::new(
                dataset,
                OperationOutcome {
                    result_rows: Some(batch.num_rows() as u64),
                    ..Default::default()
                },
            ))
        }
    }
}

fn mutation_predicate(args: &Args) -> BenchResult<String> {
    match args.selection {
        SelectionKind::Range => {
            let end = args
                .id_start
                .checked_add(args.mutation_count as u64)
                .ok_or("mutation ID range overflow")?;
            Ok(format!("id >= {} AND id < {end}", args.id_start))
        }
        SelectionKind::Random if args.operation == Operation::Delete => {
            Ok(format!("value < {}", args.mutation_count))
        }
        SelectionKind::Random => {
            Err("random selection is materialized by the exact matched-merge update driver".into())
        }
    }
}

fn benchmark_index_name(kind: IndexKind) -> &'static str {
    match kind {
        IndexKind::None => "none",
        IndexKind::Scalar => "stable_row_address_scalar",
        IndexKind::Vector => "stable_row_address_vector",
    }
}

async fn rebuild_benchmark_indices(args: &Args, dataset: &mut Dataset) -> BenchResult<()> {
    let index_name = benchmark_index_name(args.index_kind);
    match args.index_kind {
        IndexKind::None => {
            return Err("benchmark index rebuild requires an index kind".into());
        }
        IndexKind::Scalar => {
            for (column, name) in [
                ("id", index_name),
                ("value", "stable_row_address_scalar_value"),
                ("id", "stable_row_address_scalar_id_secondary"),
            ] {
                dataset
                    .create_index(
                        &[column],
                        IndexType::BTree,
                        Some(name.to_string()),
                        &ScalarIndexParams::default(),
                        true,
                    )
                    .await?;
            }
        }
        IndexKind::Vector => {
            let partitions = (args.expected_rows / 4096).clamp(1, 256);
            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    Some(index_name.to_string()),
                    &VectorIndexParams::ivf_flat(partitions, MetricType::L2),
                    true,
                )
                .await?;
        }
    }
    Ok(())
}

async fn exact_row_address_delta_bytes(dataset: &Dataset) -> BenchResult<u64> {
    let indices = dataset.load_indices().await?;

    let mut fast_manifest = dataset.manifest().clone();
    fast_manifest.row_address_layout = fast_manifest
        .row_address_layout
        .as_ref()
        .map(|layout| layout.fast_admission_projection().map(Arc::new))
        .transpose()?;
    let fast_manifest_bytes = pb::Manifest::from(&fast_manifest).encoded_len() as u64;
    let fast_index_bytes = pb::IndexSection {
        indices: indices.iter().map(Into::into).collect(),
    }
    .encoded_len() as u64;

    let mut core_manifest = dataset.manifest().clone();
    core_manifest.row_address_layout = None;
    core_manifest.max_logical_fragment_id = None;
    let mut core_fragments = core_manifest.fragments.as_ref().clone();
    for fragment in &mut core_fragments {
        fragment.native_logical_domain = None;
    }
    core_manifest.fragments = Arc::new(core_fragments);
    let mut core_indices = indices.as_ref().clone();
    for index in &mut core_indices {
        index.row_reference_domain = None;
        index.logical_coverage = None;
    }
    let core_manifest_bytes = pb::Manifest::from(&core_manifest).encoded_len() as u64;
    let core_index_bytes = pb::IndexSection {
        indices: core_indices.iter().map(Into::into).collect(),
    }
    .encoded_len() as u64;

    fast_manifest_bytes
        .checked_add(fast_index_bytes)
        .and_then(|fast| {
            core_manifest_bytes
                .checked_add(core_index_bytes)
                .and_then(|core| fast.checked_sub(core))
        })
        .ok_or_else(|| "row-address manifest Delta overflow or underflow".into())
}

fn finish_outcome(
    args: &Args,
    dataset: &Dataset,
    mut outcome: OperationOutcome,
) -> BenchResult<OperationOutcome> {
    // This function runs entirely after the measurement boundary. Manifest
    // decoding already performed the full storage contract validation during a
    // cold open, so read evidence only needs the constant-size header contract.
    if args.operation.is_read_path() {
        args.format.validate_dataset_header(dataset)?;
    } else {
        args.format.validate_dataset(dataset)?;
    }
    outcome.dataset_version = Some(dataset.version().version);
    let live_rows = dataset
        .manifest()
        .fragments
        .iter()
        .try_fold(0_usize, |total, fragment| {
            total.checked_add(fragment.num_rows()?).or(None)
        })
        .ok_or("manifest does not contain an exact live row count")?;
    if live_rows != args.expected_rows {
        return Err(format!(
            "operation {} left {live_rows} live rows, expected {}",
            args.operation.name(),
            args.expected_rows
        )
        .into());
    }
    outcome.fragments = Some(dataset.get_fragments().len() as u64);
    let physical_rows = dataset
        .manifest()
        .fragments
        .iter()
        .try_fold(0_u64, |total, fragment| {
            total.checked_add(fragment.physical_rows? as u64).or(None)
        });
    outcome.physical_rows = physical_rows;
    let physical_data_bytes = dataset
        .manifest()
        .fragments
        .iter()
        .flat_map(|fragment| &fragment.files)
        .try_fold(0_u64, |total, file| {
            total
                .checked_add(file.file_size_bytes.get()?.get())
                .or(None)
        });
    outcome.physical_data_bytes = physical_data_bytes;
    let estimated_live_data_bytes = dataset
        .manifest()
        .fragments
        .iter()
        .try_fold(0_u128, |total, fragment| {
            let physical = fragment.physical_rows? as u128;
            if physical == 0 {
                return Some(total);
            }
            let live = fragment.num_rows()? as u128;
            let bytes = fragment.files.iter().try_fold(0_u128, |bytes, file| {
                bytes.checked_add(file.file_size_bytes.get()?.get() as u128)
            })?;
            total.checked_add(bytes.saturating_mul(live) / physical)
        })
        .and_then(|bytes| u64::try_from(bytes).ok());
    outcome.estimated_live_data_bytes = estimated_live_data_bytes;
    outcome.scan_byte_amplification = physical_data_bytes
        .zip(estimated_live_data_bytes)
        .and_then(|(physical, live)| (live > 0).then_some(physical as f64 / live as f64));
    let manifest_proto: pb::Manifest = dataset.manifest().into();
    outcome.manifest_bytes = Some(manifest_proto.encoded_len() as u64);
    if let Some(layout) = &dataset.manifest().row_address_layout {
        let layout_proto: pb::RowAddressLayout = layout.as_ref().into();
        outcome.placement_root_bytes = Some(layout_proto.encoded_len() as u64);
        outcome.placement_delta_claimed_bytes = Some(layout.debt_summary.fast_delta_bytes);
        outcome.w_epoch_bytes = Some(layout.debt_summary.metadata_bytes_written_since_maintenance);
    }
    Ok(outcome)
}

#[derive(Default)]
struct StateDigest {
    rows: u64,
    xor: u64,
    sum: u64,
    ordered_xor: u64,
    ordered_sum: u64,
}

impl StateDigest {
    fn update(&mut self, batch: &RecordBatch) -> BenchResult<()> {
        let schema = batch.schema();
        for row_index in 0..batch.num_rows() {
            let mut row_hash = 0xcbf2_9ce4_8422_2325_u64;
            for (field, column) in schema.fields().iter().zip(batch.columns()) {
                row_hash = mix64(row_hash ^ stable_name_hash(field.name()));
                if column.is_null(row_index) {
                    row_hash = mix64(row_hash ^ u64::MAX);
                    continue;
                }
                if let Some(values) = column.as_any().downcast_ref::<UInt64Array>() {
                    row_hash = mix64(row_hash ^ values.value(row_index));
                } else if let Some(vectors) = column.as_any().downcast_ref::<FixedSizeListArray>() {
                    let vector = vectors.value(row_index);
                    let values = vector
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or("state digest requires Float32 vector values")?;
                    for value in values.values() {
                        row_hash = mix64(row_hash ^ value.to_bits() as u64);
                    }
                } else {
                    return Err(format!(
                        "state digest does not support column {} with type {}",
                        field.name(),
                        field.data_type()
                    )
                    .into());
                }
            }
            self.xor ^= row_hash;
            self.sum = self.sum.wrapping_add(row_hash.rotate_left(17));
            let ordinal = self.rows + row_index as u64;
            self.ordered_xor =
                mix64(self.ordered_xor ^ row_hash ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            self.ordered_sum = self
                .ordered_sum
                .wrapping_mul(0xa076_1d64_78bd_642f)
                .wrapping_add(row_hash ^ ordinal.rotate_left(23));
        }
        self.rows = self
            .rows
            .checked_add(batch.num_rows() as u64)
            .ok_or("scan state digest row count overflow")?;
        Ok(())
    }

    fn finish(&self) -> String {
        format!("{:016x}{:016x}{:016x}", self.rows, self.xor, self.sum)
    }

    fn finish_ordered(&self) -> String {
        format!(
            "{:016x}{:016x}{:016x}",
            self.rows, self.ordered_xor, self.ordered_sum
        )
    }
}

fn stable_name_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3)
    })
}

async fn collect_take_ids(
    args: &Args,
    tracker: Arc<IOTracker>,
) -> BenchResult<(Dataset, Vec<u64>, Vec<u64>)> {
    let dataset = open_dataset(args, tracker).await?;
    args.format.validate_dataset_header(&dataset)?;
    let mut scanner = dataset.scan();
    scanner.project(&["id", ROW_ID])?;
    let mut stream = scanner.try_into_stream().await?;
    let mut selected = BinaryHeap::with_capacity(args.take_count);
    let selection_seed = args.seed
        ^ (args.step as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
        ^ (args.round as u64).rotate_left(23);
    let mut live_rows = 0_usize;
    while let Some(batch) = stream.try_next().await? {
        let user_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("take-ID setup did not return UInt64 id values")?;
        let row_ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("_rowid projection did not return UInt64")?;
        if user_ids.null_count() != 0 || row_ids.null_count() != 0 {
            return Err("take-ID setup returned null id or _rowid values".into());
        }
        live_rows = live_rows
            .checked_add(batch.num_rows())
            .ok_or("take-ID setup row count overflow")?;
        for (user_id, row_id) in user_ids.values().iter().zip(row_ids.values()) {
            let key = (mix64(*user_id ^ selection_seed), *user_id, *row_id);
            if selected.len() < args.take_count {
                selected.push(key);
            } else if selected
                .peek()
                .is_some_and(|largest| (key.0, key.1) < (largest.0, largest.1))
            {
                selected.pop();
                selected.push(key);
            }
        }
    }
    if live_rows != args.expected_rows {
        return Err(format!(
            "_rowid setup returned {} rows, expected {}",
            live_rows, args.expected_rows
        )
        .into());
    }
    let mut selected = selected.into_vec();
    selected.sort_unstable_by_key(|(hash, user_id, _)| (*hash, *user_id));
    let user_ids = selected.iter().map(|(_, user_id, _)| *user_id).collect();
    let row_ids = selected.into_iter().map(|(_, _, row_id)| row_id).collect();
    Ok((dataset, user_ids, row_ids))
}

async fn write_take_ids_artifact(
    args: &Args,
    output: &Path,
    tracker: Arc<IOTracker>,
) -> BenchResult<ExecutedOperation> {
    let (dataset, user_ids, row_ids) = collect_take_ids(args, tracker).await?;
    let artifact = TakeIdsArtifact {
        schema_version: 1,
        run_id: args.run_id.clone(),
        commit: args.commit.clone(),
        policy_sha256: args.policy_sha256.clone(),
        format: args.format.name().to_string(),
        round: args.round,
        rows: args.expected_rows,
        take_count: args.take_count,
        seed: args.seed,
        user_ids,
        row_ids,
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> BenchResult<()> {
        serde_json::to_writer(&mut file, &artifact)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        if output.exists() {
            return Err(format!("take-ID artifact already exists: {}", output.display()).into());
        }
        fs::rename(&temporary, output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(ExecutedOperation::new(
        dataset,
        OperationOutcome {
            result_rows: Some(args.expected_rows as u64),
            ..Default::default()
        },
    ))
}

fn read_take_ids_artifact(args: &Args) -> BenchResult<PreparedOperation> {
    let input = args
        .take_ids_input
        .as_ref()
        .ok_or("take or index-take operation requires --take-ids-input")?;
    let artifact: TakeIdsArtifact = serde_json::from_slice(&fs::read(input)?)?;
    let expected = (
        1,
        args.run_id.as_str(),
        args.commit.as_str(),
        args.policy_sha256.as_str(),
        args.format.name(),
        args.round,
        args.expected_rows,
        args.take_count,
        args.seed,
    );
    let actual = (
        artifact.schema_version,
        artifact.run_id.as_str(),
        artifact.commit.as_str(),
        artifact.policy_sha256.as_str(),
        artifact.format.as_str(),
        artifact.round,
        artifact.rows,
        artifact.take_count,
        artifact.seed,
    );
    if actual != expected {
        return Err(format!(
            "take-ID artifact provenance mismatch: expected={expected:?}, actual={actual:?}"
        )
        .into());
    }
    if artifact.user_ids.len() != args.take_count
        || artifact.row_ids.len() != args.take_count
        || artifact
            .user_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            != args.take_count
        || artifact
            .row_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            != args.take_count
    {
        return Err("take-ID artifact must contain take_count unique row IDs".into());
    }
    Ok(PreparedOperation::Take {
        user_ids: artifact.user_ids,
        row_ids: artifact.row_ids,
    })
}

fn sample_positions(rows: usize, take_count: usize, seed: u64, round: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed ^ (round as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
    let mut selected = HashSet::with_capacity(take_count);
    let mut positions = Vec::with_capacity(take_count);
    for upper in rows - take_count..rows {
        let candidate = rng.random_range(0..=upper);
        let position = if selected.insert(candidate) {
            candidate
        } else {
            let inserted = selected.insert(upper);
            debug_assert!(inserted);
            upper
        };
        positions.push(position);
    }
    positions
}

fn directory_size(path: &Path) -> std::io::Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    if path.is_file() {
        return Ok(fs::metadata(path)?.len());
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        total = total
            .checked_add(directory_size(&entry.path())?)
            .ok_or_else(|| std::io::Error::other("dataset size overflow"))?;
    }
    Ok(total)
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the provided rusage on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful getrusage call initialized usage.
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    {
        u64::try_from(usage.ru_maxrss).ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        u64::try_from(usage.ru_maxrss)
            .ok()
            .and_then(|value| value.checked_mul(1_024))
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}
