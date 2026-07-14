// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Table maintenance for optimizing table layout.
//!
//! As a table is updated, its layout can become suboptimal. For example, if
//! a series of small streaming appends are performed, eventually there will be
//! a large number of small files. This imposes an overhead to track the large
//! number of files and for very small files can make it harder to read data
//! efficiently. In this case, files can be compacted into fewer larger files.
//!
//! To compact files in a table, use the [compact_files] method. This currently
//! can compact in two cases:
//!
//! 1. If a fragment has fewer rows than the target number of rows per fragment.
//!    The fragment must also have neighbors that are also candidates for
//!    compaction.
//! 2. If a fragment has a higher percentage of deleted rows than the provided
//!    threshold.
//!
//! In addition to the rules above there may be restrictions due to indexes.
//! Before storage version 2.3, compaction changes physical row addresses and
//! indices that contain those addresses must be remapped. Storage version 2.3
//! indices store stable logical row addresses and are reused without rewriting
//! their postings. However, indexed and unindexed fragments still cannot be
//! combined for older index coverage contracts.
//!
//! ```rust
//! # use std::sync::Arc;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
//! # use arrow_schema::{Schema, Field, DataType};
//! use lance::{dataset::WriteParams, Dataset, dataset::optimize::compact_files};
//! // Remapping indices is ignored in this example.
//! use lance::dataset::optimize::IgnoreRemap;
//!
//! # let mut rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//! #
//! # let test_dir = lance_core::utils::tempfile::TempStrDir::default();
//! # let uri = test_dir.to_string();
//! let schema = Arc::new(Schema::new(vec![Field::new("test", DataType::Int64, false)]));
//! let data = RecordBatch::try_new(
//!     schema.clone(),
//!     vec![Arc::new(Int64Array::from_iter_values(0..10_000))]
//! ).unwrap();
//! let reader = RecordBatchIterator::new(vec![Ok(data)], schema);
//!
//! // Write 100 small files
//! let write_params = WriteParams { max_rows_per_file: 100, ..Default::default()};
//! let mut dataset = Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
//! assert_eq!(dataset.get_fragments().len(), 100);
//!
//! // Use compact_files() to consolidate the data to 1 fragment
//! let metrics = compact_files(&mut dataset, Default::default(), None).await.unwrap();
//! assert_eq!(metrics.fragments_removed, 100);
//! assert_eq!(metrics.fragments_added, 1);
//! assert_eq!(dataset.get_fragments().len(), 1);
//! # })
//! ```
//!
//! ## Distributed execution
//!
//! The [compact_files] method internally can use multiple threads, but
//! sometimes you might want to run it across multiple machines. To do this,
//! use the task API.
//!
//! ```text
//!                                      ┌──► CompactionTask.execute() ─► RewriteResult ─┐
//! plan_compaction() ─► CompactionPlan ─┼──► CompactionTask.execute() ─► RewriteResult ─┼─► commit_compaction()
//!                                      └──► CompactionTask.execute() ─► RewriteResult ─┘
//! ```
//!
//! [plan_compaction()] produces a [CompactionPlan]. This can be split into multiple
//! [CompactionTask], which can be serialized and sent to other machines. Calling
//! [CompactionTask::execute()] performs the compaction and returns a [RewriteResult].
//! The [RewriteResult] can be sent back to the coordinator, which can then call
//! [commit_compaction()] to commit the changes to the dataset.
//!
//! It's not required that all tasks are passed to [commit_compaction]. If some
//! didn't complete successfully or before a deadline, they can be omitted and
//! the successful tasks can be committed. You can also commit in batches if
//! you wish. As long as the tasks don't rewrite any of the same fragments,
//! they can be committed in any order.
use lance_core::utils::row_addr_remap::{GroupInput, RowAddrRemap};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::io::Cursor;
use std::ops::{AddAssign, Range};
use std::sync::Arc;

use super::fragment::FileFragment;
use super::index::DatasetIndexRemapperOptions;
use super::rowids::{
    get_row_id_index, load_row_id_sequences, resolve_logical_row_version_sequences,
};
use super::scanner::ColumnOrdering;
use super::transaction::{
    Operation, RewriteGroup, RewrittenIndex, RowAddressManifestApplyContext, Transaction,
    TransactionBuilder, with_strict_full_ordered_rewrite_property,
};
use super::utils::make_rowid_capture_stream;
use super::{WriteMode, WriteParams, cleanup_data_fragments, write_fragments_internal};
use crate::Dataset;
use crate::Result;
use crate::dataset::utils::CapturedRowIds;
use crate::index::DatasetIndexExt;
use crate::io::commit::{commit_transaction, migrate_fragments};
use arrow::array::AsArray;
use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::StructArray;
use arrow_array::builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder};
use arrow_buffer::NullBuffer;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr, expressions};
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use futures::{StreamExt, TryStreamExt};
use lance_core::Error;
use lance_core::datatypes::{BlobHandling, BlobKind};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{DATASET_COMPACTING_EVENT, TRACE_DATASET_EVENTS};
use lance_datafusion::exec::execute_plan;
use lance_index::frag_reuse::FragReuseGroup;
use lance_index::is_system_index;
use lance_table::format::{
    Fragment, IndexMetadata, PlacementMaintenanceRequired, RowAddressLayoutDelta,
    RowDatasetVersionMeta, RowDatasetVersionRun, RowDatasetVersionSequence, RowIdMeta,
};
use lance_table::rowids::segment::U64Segment;
use lance_table::rowids::{RowIdSequence, read_row_ids, write_row_ids};
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

mod binary_copy;
mod explicit_maintenance;
mod logical_row_addresses;
pub mod remapping;

use crate::index::frag_reuse::build_new_frag_reuse_index;
use crate::io::deletion::read_dataset_deletion_file;
use binary_copy::{BinaryCopyPlan, rewrite_files_binary_copy};
pub use explicit_maintenance::{
    RowAddressMaintenanceMetrics, RowAddressMaintenanceMode, RowAddressMaintenanceOptions,
    maintain_row_addresses,
};
#[cfg(test)]
use logical_row_addresses::validate_default_compaction_logical_order;
use logical_row_addresses::{add_rewrite_provenance, plan_default_compaction_rows};
pub use remapping::{IgnoreRemap, IndexRemapper, IndexRemapperOptions, RemappedIndex};

type RewriteRowProvenance = (Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>);
type RowVersionSequences = (
    Vec<RowDatasetVersionSequence>,
    Vec<RowDatasetVersionSequence>,
);

/// Controls how data is rewritten during compaction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompactionMode {
    /// Decode and re-encode data (default).
    Reencode,
    /// Try binary copy if fragments are compatible, fall back to [`Reencode`](CompactionMode::Reencode) otherwise.
    TryBinaryCopy,
    /// Use binary copy or fail if fragments are not compatible.
    ForceBinaryCopy,
}

impl TryFrom<&str> for CompactionMode {
    type Error = Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "reencode" => Ok(Self::Reencode),
            "try_binary_copy" => Ok(Self::TryBinaryCopy),
            "force_binary_copy" => Ok(Self::ForceBinaryCopy),
            _ => Err(Error::invalid_input(format!(
                "Invalid compaction mode \"{}\". Valid values: \"reencode\", \"try_binary_copy\", \"force_binary_copy\"",
                value
            ))),
        }
    }
}

/// Controls how the old-to-new row-address mapping is built when remapping
/// indices during compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IndexRemapMode {
    /// Store a compact remap and compute row-address mappings during lookup.
    ///
    /// Best for large compactions where peak memory is the constraint. Uses
    /// less memory, but each lookup does extra bitmap/range computation.
    Compact,
    /// Store the full row-address remap in memory for fast direct lookups.
    ///
    /// Best when the remap fits comfortably in memory and remap speed is the
    /// priority. Uses more peak memory because every rewritten/deleted row has
    /// a materialized mapping entry.
    #[default]
    Direct,
}

impl TryFrom<&str> for IndexRemapMode {
    type Error = Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "compact" => Ok(Self::Compact),
            "direct" => Ok(Self::Direct),
            _ => Err(Error::invalid_input(format!(
                "Invalid index remap mode \"{}\". Valid values: \"compact\", \"direct\"",
                value
            ))),
        }
    }
}

/// Options to be passed to [compact_files].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionOptions {
    /// Target number of rows per file. Defaults to 1 million.
    ///
    /// This is used to determine which fragments need compaction, as any
    /// fragments that have fewer rows than this value will be candidates for
    /// compaction.
    pub target_rows_per_fragment: usize,
    /// Max number of rows per group
    ///
    /// This does not affect which fragments need compaction, but does affect
    /// how they are re-written if selected.
    pub max_rows_per_group: usize,
    /// Max number of bytes per file
    ///
    /// This does not affect which frgamnets need compaction, but does affect
    /// how they are re-written if selected.
    ///
    /// If not specified then the default (see [`WriteParams`]) will be used.
    pub max_bytes_per_file: Option<usize>,
    /// Whether to compact fragments with deletions so there are no deletions.
    /// Defaults to true.
    pub materialize_deletions: bool,
    /// The fraction of rows that need to be deleted in a fragment before
    /// materializing the deletions. Defaults to 10% (0.1). Setting to zero (or
    /// lower) will materialize deletions for all fragments with deletions.
    /// Setting above 1.0 will never materialize deletions.
    pub materialize_deletions_threshold: f32,
    /// The number of threads to use (how many compaction tasks to run in parallel).
    /// Defaults to the number of compute-intensive CPUs.  Not used when running
    /// tasks manually using [`plan_compaction`]
    pub num_threads: Option<usize>,
    /// The batch size to use when scanning the input fragments.  If not
    /// specified then the default (see
    /// [`crate::dataset::Scanner::batch_size`]) will be used.
    pub batch_size: Option<usize>,
    /// The number of bytes to allow to queue up in the I/O buffer when scanning
    /// the input fragments.  If not specified then the default (see
    /// [`crate::dataset::Scanner::io_buffer_size`]) will be used.
    ///
    /// Increasing this can avoid a deadlock that occurs when a single batch of
    /// data is larger than the I/O buffer size.
    pub io_buffer_size: Option<u64>,
    /// Whether to defer remapping physical-address indices during compaction. If true,
    /// indices will not be remapped during this compaction operation. Instead, the fragment
    /// reuse index is updated and will be used to perform remapping later. This is a no-op for
    /// storage version 2.3, whose indices use stable logical row addresses, and is rejected for
    /// the legacy stable-row-id mode.
    pub defer_index_remap: bool,
    /// How the old-to-new row-address mapping used to remap indices is built.
    /// Defaults to [`IndexRemapMode::Direct`].
    #[serde(default)]
    pub index_remap_mode: IndexRemapMode,
    /// The compaction mode to use. When set, this takes priority over the
    /// deprecated `enable_binary_copy` and `enable_binary_copy_force` fields.
    ///
    /// Defaults to `None` (falls back to legacy boolean fields).
    pub compaction_mode: Option<CompactionMode>,
    /// Deprecated: use `compaction_mode` instead.
    #[deprecated(note = "Use `compaction_mode` instead")]
    pub enable_binary_copy: bool,
    /// Deprecated: use `compaction_mode` instead.
    #[deprecated(note = "Use `compaction_mode` instead")]
    pub enable_binary_copy_force: bool,
    /// The batch size in bytes for reading during binary copy operations.
    /// Controls how much data is read at once when performing binary copy.
    /// Defaults to 16MB (16 * 1024 * 1024).
    pub binary_copy_read_batch_bytes: Option<usize>,
    /// Maximum number of source fragments to compact in a single run. When set,
    /// tasks are included in the plan until adding the next task would exceed
    /// this limit. This allows for incremental compaction (e.g., compact 20
    /// fragments at a time).
    /// Defaults to `None` (no limit, all eligible fragments are compacted).
    pub max_source_fragments: Option<usize>,
    /// Transaction properties to store with this commit.
    ///
    /// These key-value pairs are stored in the transaction file
    /// and can be read later to identify the source of the commit
    /// (e.g., job_id for tracking completed compaction jobs).
    #[serde(skip)]
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
}

#[allow(deprecated)]
impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            // Matching defaults for WriteParams
            target_rows_per_fragment: 1024 * 1024,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.1,
            num_threads: None,
            max_bytes_per_file: None,
            batch_size: None,
            io_buffer_size: None,
            defer_index_remap: false,
            index_remap_mode: IndexRemapMode::Direct,
            compaction_mode: None,
            enable_binary_copy: false,
            enable_binary_copy_force: false,
            binary_copy_read_batch_bytes: Some(16 * 1024 * 1024),
            max_source_fragments: None,
            transaction_properties: None,
        }
    }
}

/// Config key prefix for compaction options stored in the dataset manifest.
pub const COMPACTION_CONFIG_PREFIX: &str = "lance.compaction.";

#[allow(deprecated)]
impl CompactionOptions {
    /// Create [`CompactionOptions`] by starting with defaults and applying any
    /// overrides found in the dataset manifest config.
    ///
    /// Config keys are prefixed with `lance.compaction.` and map to fields:
    /// - `lance.compaction.target_rows_per_fragment`
    /// - `lance.compaction.max_rows_per_group`
    /// - `lance.compaction.max_bytes_per_file`
    /// - `lance.compaction.materialize_deletions`
    /// - `lance.compaction.materialize_deletions_threshold`
    /// - `lance.compaction.defer_index_remap`
    /// - `lance.compaction.index_remap_mode`
    /// - `lance.compaction.batch_size`
    /// - `lance.compaction.io_buffer_size`
    /// - `lance.compaction.compaction_mode`
    /// - `lance.compaction.binary_copy_read_batch_bytes`
    /// - `lance.compaction.max_source_fragments`
    pub fn from_dataset_config(config: &HashMap<String, String>) -> Result<Self> {
        let mut opts = Self::default();
        opts.apply_dataset_config(config)?;
        Ok(opts)
    }

    /// Apply overrides from the dataset manifest config to this options struct.
    ///
    /// Only fields with corresponding config keys are modified; other fields
    /// retain their current values.
    pub fn apply_dataset_config(&mut self, config: &HashMap<String, String>) -> Result<()> {
        for (key, value) in config {
            let Some(field) = key.strip_prefix(COMPACTION_CONFIG_PREFIX) else {
                continue;
            };
            match field {
                "target_rows_per_fragment" => {
                    self.target_rows_per_fragment = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?;
                }
                "max_rows_per_group" => {
                    self.max_rows_per_group = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?;
                }
                "max_bytes_per_file" => {
                    self.max_bytes_per_file = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "materialize_deletions" => {
                    self.materialize_deletions = match value.to_lowercase().as_str() {
                        "true" => true,
                        "false" => false,
                        _ => {
                            return Err(Error::invalid_input(format!(
                                "Invalid value for {}: '{}' (expected 'true' or 'false')",
                                key, value
                            )));
                        }
                    };
                }
                "materialize_deletions_threshold" => {
                    self.materialize_deletions_threshold = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a float between 0.0 and 1.0)",
                            key, value
                        ))
                    })?;
                }
                "defer_index_remap" => {
                    self.defer_index_remap = match value.to_lowercase().as_str() {
                        "true" => true,
                        "false" => false,
                        _ => {
                            return Err(Error::invalid_input(format!(
                                "Invalid value for {}: '{}' (expected 'true' or 'false')",
                                key, value
                            )));
                        }
                    };
                }
                "index_remap_mode" => {
                    self.index_remap_mode = IndexRemapMode::try_from(value.as_str())?;
                }
                "batch_size" => {
                    self.batch_size = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "io_buffer_size" => {
                    self.io_buffer_size = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "compaction_mode" => {
                    self.compaction_mode = Some(CompactionMode::try_from(value.as_str())?);
                }
                "binary_copy_read_batch_bytes" => {
                    self.binary_copy_read_batch_bytes = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "max_source_fragments" => {
                    self.max_source_fragments = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                _ => {
                    warn!("Ignoring unknown compaction config key: {}", key);
                }
            }
        }
        Ok(())
    }

    pub fn validate(&mut self) {
        // If threshold is 100%, same as turning off deletion materialization.
        if self.materialize_deletions && self.materialize_deletions_threshold >= 1.0 {
            self.materialize_deletions = false;
        }
    }

    /// Returns the effective [`CompactionMode`], preferring the new
    /// `compaction_mode` field and falling back to the deprecated boolean
    /// fields for backwards compatibility.
    pub fn compaction_mode(&self) -> CompactionMode {
        if let Some(mode) = self.compaction_mode {
            return mode;
        }
        // Fall back to deprecated booleans
        match (self.enable_binary_copy, self.enable_binary_copy_force) {
            (true, true) => CompactionMode::ForceBinaryCopy,
            (true, false) => CompactionMode::TryBinaryCopy,
            _ => CompactionMode::Reencode,
        }
    }

    /// Set transaction properties to store in the commit manifest.
    pub fn transaction_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.transaction_properties = Some(Arc::new(properties));
        self
    }
}

/// Determine if page-level binary copy can safely merge the provided fragments.
///
/// Preconditions checked in order:
/// - Compaction mode is not `Reencode`
/// - Dataset storage format is non-legacy
/// - Fragment list is non-empty
/// - All data files share identical Lance file versions
/// - No fragment has a deletion file
///   TODO: Need to support schema evolution case like add column and drop column
/// - All data files share identical schema mappings (`fields`, `column_indices`)
/// - Input data files must not contain extra global buffers (beyond schema / file descriptor)
#[cfg(test)]
async fn can_use_binary_copy(
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> bool {
    plan_binary_copy_impl(dataset, options, fragments)
        .await
        .map(|plan| plan.is_some())
        .unwrap_or_else(|err| {
            log::warn!("Binary copy disabled due to error: {}", err);
            false
        })
}

async fn plan_binary_copy_impl(
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> Result<Option<BinaryCopyPlan>> {
    use lance_file::reader::FileReader as LFReader;
    use lance_file::version::LanceFileVersion;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use prost::Message;
    use prost_types::Any;

    if matches!(options.compaction_mode(), CompactionMode::Reencode) {
        log::debug!("Binary copy disabled: compaction mode is Reencode");
        return Ok(None);
    }

    let has_blob_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob());
    if has_blob_columns {
        log::debug!("Binary copy disabled: dataset contains blob columns");
        return Ok(None);
    }

    let storage_ok = dataset
        .manifest
        .data_storage_format
        .lance_file_version()
        .map(|v| !matches!(v.resolve(), LanceFileVersion::Legacy))
        .unwrap_or(false);
    if !storage_ok {
        log::debug!("Binary copy disabled: dataset uses legacy storage format");
        return Ok(None);
    }

    if fragments.is_empty() {
        log::debug!("Binary copy disabled: no fragments to compact");
        return Ok(None);
    }

    let storage_file_version = dataset
        .manifest
        .data_storage_format
        .lance_file_version()?
        .resolve();

    if fragments[0].files.len() != 1 {
        log::debug!(
            "Binary copy disabled: fragment {} does not have exactly one data file",
            fragments[0].id
        );
        return Ok(None);
    }
    let ref_fields = &fragments[0].files[0].fields;
    let ref_cols = &fragments[0].files[0].column_indices;
    let mut source_file_row_counts = Vec::with_capacity(fragments.len());
    let mut source_file_size_bytes = Vec::with_capacity(fragments.len());
    let mut baseline_column_encodings: Option<Vec<Vec<u8>>> = None;

    for fragment in fragments {
        if fragment.deletion_file.is_some() {
            log::debug!(
                "Binary copy disabled: fragment {} has a deletion file",
                fragment.id
            );
            return Ok(None);
        }
        if fragment.files.len() != 1 {
            log::debug!(
                "Binary copy disabled: fragment {} does not have exactly one data file",
                fragment.id
            );
            return Ok(None);
        }
        let data_file = &fragment.files[0];
        let version_ok = LanceFileVersion::try_from_major_minor(
            data_file.file_major_version,
            data_file.file_minor_version,
        )
        .map(|version| version.resolve())
        .is_ok_and(|version| version == storage_file_version);
        if !version_ok {
            log::debug!("Binary copy disabled: data files use different file versions");
            return Ok(None);
        }
        if data_file.fields != *ref_fields || data_file.column_indices != *ref_cols {
            log::debug!("Binary copy disabled: data files use different schema mappings");
            return Ok(None);
        }

        let object_store = match data_file.base_id {
            Some(base_id) => dataset.object_store(Some(base_id)).await?,
            None => dataset.object_store.clone(),
        };
        let full_path = dataset
            .data_file_dir(data_file)?
            .clone()
            .join(data_file.path.as_str());
        let scan_scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let file_scheduler = scan_scheduler
            .open_file_with_priority(&full_path, 0, &data_file.file_size_bytes)
            .await?;
        let file_meta = LFReader::read_all_metadata(&file_scheduler).await?;
        // Binary copy regenerates the footer, so user global buffers cannot be
        // carried into the output without a format-specific merge operation.
        if file_meta.file_buffers.len() > 1 {
            log::debug!(
                "Binary copy disabled: data file has extra global buffers (len={})",
                file_meta.file_buffers.len()
            );
            return Ok(None);
        }
        let metadata_version = LanceFileVersion::try_from_major_minor(
            file_meta.major_version as u32,
            file_meta.minor_version as u32,
        )?
        .resolve();
        if metadata_version != storage_file_version {
            log::debug!("Binary copy disabled: file footer version differs from the dataset");
            return Ok(None);
        }
        if file_meta.file_schema.as_ref() != dataset.schema() {
            log::debug!("Binary copy disabled: file footer schema differs from the dataset");
            return Ok(None);
        }
        let row_count = u32::try_from(file_meta.num_rows).map_err(|_| {
            Error::format_capacity_exceeded(format!(
                "binary-copy source file {} rows {} exceed u32 capacity",
                data_file.path, file_meta.num_rows
            ))
        })?;
        if row_count == 0 {
            log::debug!("Binary copy disabled: source data file contains no rows");
            return Ok(None);
        }
        if fragment
            .physical_rows
            .is_some_and(|rows| rows != row_count as usize)
        {
            log::debug!(
                "Binary copy disabled: fragment {} physical_rows disagrees with its data file",
                fragment.id
            );
            return Ok(None);
        }
        let column_encodings = file_meta
            .column_infos
            .iter()
            .map(|column| Ok(Any::from_msg(&column.encoding)?.encode_to_vec()))
            .collect::<Result<Vec<_>>>()?;
        if let Some(baseline) = &baseline_column_encodings {
            if baseline != &column_encodings {
                log::debug!("Binary copy disabled: column encodings differ between data files");
                return Ok(None);
            }
        } else {
            baseline_column_encodings = Some(column_encodings);
        }
        source_file_row_counts.push(row_count);
        source_file_size_bytes.push(file_meta.file_size_bytes);
    }

    Ok(Some(BinaryCopyPlan::try_new(
        source_file_row_counts,
        source_file_size_bytes,
        options.target_rows_per_fragment,
        options.max_bytes_per_file,
    )?))
}

/// Metrics returned by [compact_files].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionMetrics {
    /// The number of non-empty rewrite groups considered for execution.
    pub groups_planned: usize,
    /// The number of groups admitted before reading data pages or writing data files.
    pub groups_admitted: usize,
    /// The number of groups rejected before reading or writing data files.
    pub groups_not_admitted: usize,
    /// The number of fragments that have been overwritten.
    pub fragments_removed: usize,
    /// The number of new fragments that have been added.
    pub fragments_added: usize,
    /// The number of files that have been removed, including deletion files.
    pub files_removed: usize,
    /// The number of files that have been added, which is always equal to the
    /// number of fragments.
    pub files_added: usize,
}

impl AddAssign for CompactionMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.groups_planned += rhs.groups_planned;
        self.groups_admitted += rhs.groups_admitted;
        self.groups_not_admitted += rhs.groups_not_admitted;
        self.fragments_removed += rhs.fragments_removed;
        self.fragments_added += rhs.fragments_added;
        self.files_removed += rhs.files_removed;
        self.files_added += rhs.files_added;
    }
}

/// Trait for implementing custom compaction planning strategies.
///
/// This trait allows users to define their own compaction strategies by implementing
/// the `plan` method. The default implementation is provided by [`DefaultCompactionPlanner`].
#[async_trait::async_trait]
pub trait CompactionPlanner: Send + Sync {
    /// Build compaction plan.
    ///
    /// This method analyzes the dataset's fragments and generates a [`CompactionPlan`]
    /// containing a list of compaction tasks to execute.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Reference to the dataset to be compacted
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan>;
}

/// Formulate a plan to compact the files in a dataset
///
/// The compaction plan will contain a list of tasks to execute. Each task
/// will contain approximately `target_rows_per_fragment` rows and will be
/// rewriting fragments that are adjacent in the dataset's fragment list. Some
/// tasks may contain a single fragment when that fragment has deletions that
/// are being materialized and doesn't have any neighbors that need to be
/// compacted.
#[derive(Debug, Clone, Default)]
pub struct DefaultCompactionPlanner {
    options: CompactionOptions,
}

impl DefaultCompactionPlanner {
    pub fn new(mut options: CompactionOptions) -> Self {
        options.validate();
        Self { options }
    }
}

#[async_trait::async_trait]
impl CompactionPlanner for DefaultCompactionPlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan> {
        if self.options.defer_index_remap && dataset.manifest.uses_legacy_stable_row_ids() {
            return Err(Error::invalid_input(
                "defer_index_remap=true is not supported on datasets with legacy stable row IDs: \
                 legacy stable row IDs do not require index remapping during compaction, so \
                 there is nothing to defer."
                    .to_string(),
            ));
        }

        // get_fragments should be returning fragments in sorted order (by id)
        // and fragment ids should be unique
        let fragments = dataset.get_fragments();

        debug_assert!(
            fragments.windows(2).all(|w| w[0].id() < w[1].id()),
            "fragments in manifest are not sorted"
        );
        let mut fragment_metrics = futures::stream::iter(fragments)
            .map(|fragment| async move {
                match collect_metrics(&fragment).await {
                    Ok(metrics) => Ok((fragment.metadata, metrics)),
                    Err(e) => Err(e),
                }
            })
            .buffered(dataset.object_store.as_ref().io_parallelism());

        let index_fragmaps = load_index_fragmaps(dataset).await?;
        let indices_containing_frag = |frag_id: u32| {
            index_fragmaps
                .iter()
                .enumerate()
                .filter(|(_, bitmap)| bitmap.contains(frag_id))
                .map(|(pos, _)| pos)
                .collect::<Vec<_>>()
        };

        let mut candidate_bins: Vec<CandidateBin> = Vec::new();
        let mut current_bin: Option<CandidateBin> = None;
        let mut i = 0;

        while let Some(res) = fragment_metrics.next().await {
            let (fragment, metrics) = res?;

            let candidacy = if self.options.materialize_deletions
                && metrics.deletion_percentage() > self.options.materialize_deletions_threshold
            {
                Some(CompactionCandidacy::CompactItself)
            } else if metrics.physical_rows < self.options.target_rows_per_fragment {
                // Only want to compact if their are neighbors to compact such that
                // we can get a larger fragment.
                Some(CompactionCandidacy::CompactWithNeighbors)
            } else {
                // Not a candidate
                None
            };

            let indices = indices_containing_frag(fragment.id as u32);

            match (candidacy, &mut current_bin) {
                (None, None) => {} // keep searching
                (Some(candidacy), None) => {
                    // Start a new bin
                    current_bin = Some(CandidateBin {
                        fragments: vec![fragment],
                        pos_range: i..(i + 1),
                        candidacy: vec![candidacy],
                        row_counts: vec![metrics.num_rows()],
                        indices,
                    });
                }
                (Some(candidacy), Some(bin)) => {
                    // We cannot mix "indexed" and "non-indexed" fragments and so we only consider
                    // the existing bin if it contains the same indices
                    if bin.indices == indices {
                        // Add to current bin
                        bin.fragments.push(fragment);
                        bin.pos_range.end += 1;
                        bin.candidacy.push(candidacy);
                        bin.row_counts.push(metrics.num_rows());
                    } else {
                        // Index set is different.  Complete previous bin and start new one
                        candidate_bins.push(current_bin.take().unwrap());
                        current_bin = Some(CandidateBin {
                            fragments: vec![fragment],
                            pos_range: i..(i + 1),
                            candidacy: vec![candidacy],
                            row_counts: vec![metrics.num_rows()],
                            indices,
                        });
                    }
                }
                (None, Some(_)) => {
                    // Bin is complete
                    candidate_bins.push(current_bin.take().unwrap());
                }
            }

            i += 1;
        }

        // Flush the last bin
        if let Some(bin) = current_bin {
            candidate_bins.push(bin);
        }

        let all_tasks: Vec<TaskData> = candidate_bins
            .into_iter()
            .filter(|bin| !bin.is_noop())
            .flat_map(|bin| bin.split_for_size(self.options.target_rows_per_fragment))
            .map(|bin| TaskData {
                fragments: bin.fragments,
                v2_3_plan: None,
            })
            .collect();

        let tasks = if let Some(max_frags) = self.options.max_source_fragments {
            let mut total_frags = 0;
            all_tasks
                .into_iter()
                .take_while(|task| {
                    total_frags += task.fragments.len();
                    total_frags <= max_frags
                })
                .collect()
        } else {
            all_tasks
        };

        let mut compaction_plan =
            CompactionPlan::new(dataset.manifest.version, self.options.clone());
        if dataset.manifest.uses_stable_logical_row_addresses() {
            let mut prepared_tasks = Vec::with_capacity(tasks.len());
            for task in tasks {
                match preflight_v2_3_compaction_task(dataset, &task.fragments, &self.options)
                    .await?
                {
                    V2_3CompactionAdmission::Admitted(prepared) => {
                        prepared_tasks.push((task, prepared));
                    }
                    V2_3CompactionAdmission::NotAdmitted(reason) => {
                        compaction_plan.planning_metrics.groups_planned += 1;
                        compaction_plan.planning_metrics.groups_not_admitted += 1;
                        warn!(
                            "Skipping storage-version-2.3 compaction group before page reads or data writes: {reason}"
                        );
                    }
                }
            }

            if prepared_tasks.len() > 1 {
                let indices = dataset.load_indices().await?;
                while let Some(reason) = preflight_prepared_v2_3_compaction_tasks(
                    dataset,
                    &prepared_tasks,
                    indices.as_ref(),
                )? {
                    let (task, _) = prepared_tasks.pop().expect("prepared tasks is non-empty");
                    compaction_plan.planning_metrics.groups_planned += 1;
                    compaction_plan.planning_metrics.groups_not_admitted += 1;
                    warn!(
                        "Skipping storage-version-2.3 compaction group {:?} because the combined transaction was not admitted before page reads or data writes: {reason}",
                        task.fragments
                            .iter()
                            .map(|fragment| fragment.id)
                            .collect::<Vec<_>>()
                    );
                    if prepared_tasks.is_empty() {
                        break;
                    }
                }
            }
            compaction_plan
                .tasks
                .extend(prepared_tasks.into_iter().map(|(mut task, prepared)| {
                    task.v2_3_plan = Some(prepared.plan);
                    task
                }));
        } else {
            compaction_plan.extend_tasks(tasks);
        }

        Ok(compaction_plan)
    }
}

/// Compacts the files in the dataset without reordering them.
///
/// By default, this does a few things:
///  * Removes deleted rows from fragments.
///  * Removes dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// This method tries to preserve the insertion order of rows in the dataset.
///
/// If no compaction is needed, this method will not make a new version of the table.
pub async fn compact_files(
    dataset: &mut Dataset,
    options: CompactionOptions,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
) -> Result<CompactionMetrics> {
    info!(target: TRACE_DATASET_EVENTS, event=DATASET_COMPACTING_EVENT, uri = &dataset.uri);
    let planner = DefaultCompactionPlanner::new(options);
    compact_files_with_planner(dataset, remap_options, &planner).await
}

fn version_sequence_from_values(values: &[u64]) -> RowDatasetVersionSequence {
    let mut runs = Vec::new();
    let mut start = 0_u64;
    while start < values.len() as u64 {
        let version = values[start as usize];
        let mut end = start + 1;
        while end < values.len() as u64 && values[end as usize] == version {
            end += 1;
        }
        runs.push(RowDatasetVersionRun {
            span: U64Segment::Range(start..end),
            version,
        });
        start = end;
    }
    RowDatasetVersionSequence { runs }
}

async fn ordered_legacy_row_versions(
    dataset: &Dataset,
    row_ids: &RowIdSequence,
) -> Result<(RowDatasetVersionSequence, RowDatasetVersionSequence)> {
    let stable_index = get_row_id_index(dataset).await?;
    let fragments = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| (fragment.id as u32, fragment))
        .collect::<HashMap<_, _>>();
    let mut metadata = HashMap::<
        u32,
        (
            Option<RowDatasetVersionSequence>,
            Option<RowDatasetVersionSequence>,
        ),
    >::new();
    let mut created = Vec::with_capacity(row_ids.len() as usize);
    let mut updated = Vec::with_capacity(row_ids.len() as usize);

    for row_id in row_ids.iter() {
        let physical = if let Some(index) = &stable_index {
            index.get(row_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "ordered rewrite captured non-live stable row ID {row_id}"
                ))
            })?
        } else {
            RowAddress::from(row_id)
        };
        let fragment = fragments.get(&physical.fragment_id()).ok_or_else(|| {
            Error::invalid_input(format!(
                "ordered rewrite row {row_id} resolves to missing fragment {}",
                physical.fragment_id()
            ))
        })?;
        if let std::collections::hash_map::Entry::Vacant(entry) =
            metadata.entry(physical.fragment_id())
        {
            entry.insert((
                fragment
                    .created_at_version_meta
                    .as_ref()
                    .map(RowDatasetVersionMeta::load_sequence)
                    .transpose()?,
                fragment
                    .last_updated_at_version_meta
                    .as_ref()
                    .map(RowDatasetVersionMeta::load_sequence)
                    .transpose()?,
            ));
        }
        let (created_sequence, updated_sequence) = metadata
            .get(&physical.fragment_id())
            .expect("ordered rewrite version metadata was initialized");
        let offset = physical.row_offset() as usize;
        let created_version = match created_sequence {
            Some(sequence) => sequence.version_at(offset).ok_or_else(|| {
                Error::invalid_input(format!(
                    "created-at metadata for fragment {} does not cover row offset {offset}",
                    fragment.id
                ))
            })?,
            None => 1,
        };
        let updated_version = match updated_sequence {
            Some(sequence) => sequence.version_at(offset).ok_or_else(|| {
                Error::invalid_input(format!(
                    "last-updated metadata for fragment {} does not cover row offset {offset}",
                    fragment.id
                ))
            })?,
            None => created_version,
        };
        created.push(created_version);
        updated.push(updated_version);
    }
    Ok((
        version_sequence_from_values(&created),
        version_sequence_from_values(&updated),
    ))
}

async fn apply_ordered_legacy_provenance(
    dataset: &Dataset,
    row_ids: RowIdSequence,
    new_fragments: &mut [Fragment],
) -> Result<Option<Vec<u8>>> {
    let chunk_sizes = new_fragments
        .iter()
        .map(|fragment| fragment.physical_rows.unwrap_or_default() as u64)
        .collect::<Vec<_>>();
    if row_ids.len() != chunk_sizes.iter().sum::<u64>() {
        return Err(Error::invalid_input(
            "ordered rewrite row identity count disagrees with output fragments",
        ));
    }
    let (created, updated) = ordered_legacy_row_versions(dataset, &row_ids).await?;
    let created = lance_table::rowids::version::rechunk_version_sequences(
        [created],
        chunk_sizes.clone(),
        false,
    )?;
    let updated = lance_table::rowids::version::rechunk_version_sequences(
        [updated],
        chunk_sizes.clone(),
        false,
    )?;
    for ((fragment, created), updated) in new_fragments.iter_mut().zip(created).zip(updated) {
        fragment.created_at_version_meta = Some(RowDatasetVersionMeta::from_sequence(&created)?);
        fragment.last_updated_at_version_meta =
            Some(RowDatasetVersionMeta::from_sequence(&updated)?);
    }

    if dataset.manifest.uses_legacy_stable_row_ids() {
        let sequences = lance_table::rowids::rechunk_sequences([row_ids], chunk_sizes, false)?;
        for (fragment, sequence) in new_fragments.iter_mut().zip(sequences) {
            fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence)));
        }
        Ok(None)
    } else {
        Ok(Some(write_row_ids(&row_ids)))
    }
}

/// Rewrite all current fragments into a deterministic user-column order.
///
/// Storage version 2.3 uses bounded inline clustering and rejects the rewrite
/// before data-file writes if the source-contiguous extent or metadata budget
/// would be exceeded.  Older formats preserve their native identity contract
/// and synchronously remap physical-address indices.
pub async fn rewrite_files_in_order(
    dataset: &mut Dataset,
    ordering: Vec<ColumnOrdering>,
    mut options: CompactionOptions,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>,
) -> Result<CompactionMetrics> {
    if ordering.is_empty() {
        return Err(Error::invalid_input(
            "ordered rewrite requires at least one ordering column",
        ));
    }
    for column in &ordering {
        if column.column_name == lance_core::ROW_ID || column.column_name == lance_core::ROW_ADDR {
            return Err(Error::invalid_input(
                "ordered rewrite ordering must use user columns",
            ));
        }
        if dataset.schema().field(&column.column_name).is_none() {
            return Err(Error::invalid_input(format!(
                "ordered rewrite column {} does not exist",
                column.column_name
            )));
        }
    }
    if options.target_rows_per_fragment == 0 || options.max_rows_per_group == 0 {
        return Err(Error::invalid_input(
            "ordered rewrite row limits must be greater than zero",
        ));
    }
    if dataset.manifest.fragments.is_empty() {
        return Ok(CompactionMetrics::default());
    }
    options.transaction_properties = Some(with_strict_full_ordered_rewrite_property(
        options.transaction_properties.clone(),
    ));

    if dataset.manifest.uses_stable_logical_row_addresses() {
        let original_fragments = dataset.manifest.fragments.as_ref().clone();
        let files_removed = original_fragments
            .iter()
            .map(|fragment| fragment.files.len() + usize::from(fragment.deletion_file.is_some()))
            .sum();
        let metrics = maintain_row_addresses(
            dataset,
            RowAddressMaintenanceOptions {
                mode: RowAddressMaintenanceMode::BoundedRecluster { ordering },
                target_rows_per_fragment: options.target_rows_per_fragment,
                max_rows_per_group: options.max_rows_per_group,
                max_bytes_per_file: options.max_bytes_per_file,
                batch_size: options.batch_size,
                io_buffer_size: options.io_buffer_size,
                transaction_properties: options.transaction_properties,
            },
        )
        .await?;
        return Ok(CompactionMetrics {
            groups_planned: 1,
            groups_admitted: 1,
            fragments_removed: metrics.fragments_removed,
            fragments_added: metrics.fragments_added,
            files_removed,
            files_added: metrics.data_files_written,
            ..CompactionMetrics::default()
        });
    }
    if options.defer_index_remap {
        return Err(Error::invalid_input(
            "ordered rewrites cannot defer physical-address index remapping",
        ));
    }
    if !matches!(options.index_remap_mode, IndexRemapMode::Direct) {
        return Err(Error::invalid_input(
            "ordered rewrites require direct index remapping",
        ));
    }
    if !matches!(options.compaction_mode(), CompactionMode::Reencode) {
        return Err(Error::invalid_input(
            "ordered rewrites require re-encoding and cannot use binary copy",
        ));
    }
    let original_fragments = dataset.manifest.fragments.as_ref().clone();
    let index_fragmaps = load_index_fragmaps(dataset).await?;
    if index_fragmaps.iter().any(|fragment_bitmap| {
        original_fragments
            .iter()
            .any(|fragment| !fragment_bitmap.contains(fragment.id as u32))
    }) {
        return Err(Error::invalid_input(
            "ordered rewrite requires every non-system legacy index to cover the full current fragment set; consolidate, rebuild, or drop partial indices before rewriting",
        ));
    }
    let scan_fragments = migrate_fragments(dataset, &original_fragments, false).await?;
    let mut scanner = dataset.scan();
    if dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob() && !field.is_blob_v2())
    {
        scanner.blob_handling(BlobHandling::AllBinary);
    }
    let has_blob_v2_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob_v2());
    if has_blob_v2_columns {
        scanner.with_row_address();
    }
    if let Some(batch_size) = options.batch_size {
        scanner.batch_size(batch_size);
    }
    if let Some(io_buffer_size) = options.io_buffer_size {
        scanner.io_buffer_size(io_buffer_size);
    }
    scanner
        .with_fragments(scan_fragments)
        .with_row_id()
        .scan_in_order(false);
    let plan = scanner.create_plan().await?;
    let mut sort_exprs = ordering
        .iter()
        .map(|column| {
            Ok(PhysicalSortExpr {
                expr: expressions::col(&column.column_name, plan.schema().as_ref())?,
                options: arrow::compute::SortOptions {
                    descending: !column.ascending,
                    nulls_first: column.nulls_first,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    sort_exprs.push(PhysicalSortExpr {
        expr: expressions::col(lance_core::ROW_ID, plan.schema().as_ref())?,
        options: arrow::compute::SortOptions {
            descending: false,
            nulls_first: false,
        },
    });
    let ordering = LexOrdering::new(sort_exprs)
        .ok_or_else(|| Error::internal("ordered rewrite sort expression cannot be empty"))?;
    let sorted = execute_plan(
        Arc::new(SortExec::new(ordering, plan)),
        scanner.execution_options(),
    )?;
    // SequenceStyle preserves destination order for both legacy stable IDs and
    // physical row addresses. AddressStyle is a set and cannot represent a
    // permutation.
    let (stream, row_ids_rx) = make_rowid_capture_stream(sorted, true)?;
    let stream = if has_blob_v2_columns {
        transform_blob_v2_stream(dataset, stream)
    } else {
        stream
    };
    let mut write_params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        allow_external_blob_outside_bases: true,
        enable_stable_row_ids: dataset.manifest.uses_legacy_stable_row_ids(),
        ..Default::default()
    };
    if let Some(max_bytes_per_file) = options.max_bytes_per_file {
        write_params.max_bytes_per_file = max_bytes_per_file;
    }
    let (mut new_fragments, _) = write_fragments_internal(
        Some(dataset),
        dataset.object_store.clone(),
        &dataset.base,
        dataset.schema().clone(),
        stream,
        write_params,
        None,
    )
    .await?;
    let provenance = async {
        let captured = row_ids_rx.try_recv().map_err(|error| {
            Error::internal(format!(
                "ordered rewrite row identity capture did not complete: {error}"
            ))
        })?;
        let CapturedRowIds::SequenceStyle(row_ids) = captured else {
            return Err(Error::internal(
                "ordered rewrite captured identities without preserving order",
            ));
        };
        apply_ordered_legacy_provenance(dataset, row_ids, &mut new_fragments).await
    }
    .await;
    let ordered_row_addrs = match provenance {
        Ok(ordered_row_addrs) => ordered_row_addrs,
        Err(error) => {
            cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                .await;
            return Err(error);
        }
    };
    let metrics = CompactionMetrics {
        groups_planned: 1,
        groups_admitted: 1,
        fragments_removed: original_fragments.len(),
        fragments_added: new_fragments.len(),
        files_removed: original_fragments
            .iter()
            .map(|fragment| fragment.files.len() + usize::from(fragment.deletion_file.is_some()))
            .sum(),
        files_added: new_fragments
            .iter()
            .map(|fragment| fragment.files.len())
            .sum(),
        groups_not_admitted: 0,
    };
    let result = RewriteResult {
        metrics,
        new_fragments,
        read_version: dataset.manifest.version,
        original_fragments,
        row_addrs: None,
        ordered_row_addrs,
        logical_row_ids: None,
        retired_logical_row_ids: None,
    };
    commit_compaction(
        dataset,
        vec![result],
        remap_options.unwrap_or_else(|| Arc::new(DatasetIndexRemapperOptions::default())),
        &options,
    )
    .await
}

pub async fn compact_files_with_planner(
    dataset: &mut Dataset,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
    planner: &dyn CompactionPlanner,
) -> Result<CompactionMetrics> {
    let compaction_plan: CompactionPlan = planner.plan(dataset).await?;

    // If nothing to compact, don't make a commit.
    if compaction_plan.tasks().is_empty() {
        return Ok(compaction_plan.planning_metrics);
    }

    let dataset_ref = &dataset.clone();
    let planning_metrics = compaction_plan.planning_metrics.clone();

    let result_stream = futures::stream::iter(compaction_plan.tasks)
        .map(|task| rewrite_files(Cow::Borrowed(dataset_ref), task, &compaction_plan.options))
        .buffer_unordered(
            compaction_plan
                .options
                .num_threads
                .unwrap_or_else(get_num_compute_intensive_cpus),
        );

    let completed_tasks: Vec<RewriteResult> = result_stream.try_collect().await?;
    let remap_options = remap_options.unwrap_or(Arc::new(DatasetIndexRemapperOptions::default()));
    let mut metrics = commit_compaction(
        dataset,
        completed_tasks,
        remap_options,
        &compaction_plan.options,
    )
    .await?;
    metrics += planning_metrics;

    Ok(metrics)
}

/// Information about a fragment used to decide its fate in compaction
#[derive(Debug)]
struct FragmentMetrics {
    /// The number of original rows in the fragment
    pub physical_rows: usize,
    /// The number of rows that have been deleted
    pub num_deletions: usize,
}

impl FragmentMetrics {
    /// The fraction of rows that have been deleted
    fn deletion_percentage(&self) -> f32 {
        if self.physical_rows > 0 {
            self.num_deletions as f32 / self.physical_rows as f32
        } else {
            0.0
        }
    }

    /// The number of rows that are still in the fragment
    fn num_rows(&self) -> usize {
        self.physical_rows - self.num_deletions
    }
}

async fn collect_metrics(fragment: &FileFragment) -> Result<FragmentMetrics> {
    let physical_rows = fragment.physical_rows();
    let num_deletions = fragment.count_deletions();
    let (physical_rows, num_deletions) =
        futures::future::try_join(physical_rows, num_deletions).await?;
    Ok(FragmentMetrics {
        physical_rows,
        num_deletions,
    })
}

enum V2_3CompactionAdmission {
    Admitted(PreparedV2_3CompactionTask),
    NotAdmitted(String),
}

struct PreparedV2_3CompactionTask {
    plan: V2_3CompactionTaskPlan,
    current_deletion_vectors: BTreeMap<u32, RoaringBitmap>,
}

fn planned_output_row_counts(row_count: u64, rows_per_fragment: usize) -> Result<Vec<u32>> {
    let rows_per_fragment = u32::try_from(rows_per_fragment).map_err(|_| {
        Error::invalid_input("storage-version-2.3 target_rows_per_fragment must fit in u32")
    })?;
    if rows_per_fragment == 0 {
        return Err(Error::invalid_input(
            "storage-version-2.3 target_rows_per_fragment must be greater than zero",
        ));
    }
    let mut remaining = row_count;
    let fragment_count = usize::try_from(row_count.div_ceil(rows_per_fragment as u64))
        .map_err(|_| Error::invalid_input("planned output fragment count exceeds usize"))?;
    let mut counts = Vec::with_capacity(fragment_count);
    while remaining > 0 {
        let count = remaining.min(rows_per_fragment as u64) as u32;
        counts.push(count);
        remaining -= count as u64;
    }
    Ok(counts)
}

async fn planned_reencode_output_row_counts(
    dataset: &Dataset,
    fragments: &[Fragment],
    output_row_count: u64,
    options: &CompactionOptions,
) -> Result<Vec<u32>> {
    let Some(max_bytes_per_file) = options.max_bytes_per_file else {
        return planned_output_row_counts(output_row_count, options.target_rows_per_fragment);
    };
    if max_bytes_per_file == 0 {
        return Err(Error::invalid_input(
            "storage-version-2.3 max_bytes_per_file must be greater than zero",
        ));
    }

    let mut source_physical_rows = 0_u64;
    let mut source_size_bytes = 0_u64;
    for fragment in fragments {
        let physical_rows = u64::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input(format!(
                "storage-version-2.3 compaction source fragment {} is missing physical_rows",
                fragment.id
            ))
        })?)
        .map_err(|_| Error::format_capacity_exceeded("source physical_rows exceed u64"))?;
        source_physical_rows = source_physical_rows
            .checked_add(physical_rows)
            .ok_or_else(|| Error::format_capacity_exceeded("source row count exceeds u64"))?;
        for data_file in &fragment.files {
            let file_size = if let Some(file_size) = data_file.file_size_bytes.get() {
                u64::from(file_size)
            } else {
                let object_store = match data_file.base_id {
                    Some(base_id) => dataset.object_store(Some(base_id)).await?,
                    None => dataset.object_store.clone(),
                };
                let path = dataset
                    .data_file_dir(data_file)?
                    .clone()
                    .join(data_file.path.as_str());
                object_store.size(&path).await?
            };
            source_size_bytes = source_size_bytes.checked_add(file_size).ok_or_else(|| {
                Error::format_capacity_exceeded("source data-file bytes exceed u64")
            })?;
        }
    }

    let target_rows = u64::try_from(options.target_rows_per_fragment).map_err(|_| {
        Error::invalid_input("storage-version-2.3 target_rows_per_fragment exceeds u64")
    })?;
    if target_rows == 0 {
        return Err(Error::invalid_input(
            "storage-version-2.3 target_rows_per_fragment must be greater than zero",
        ));
    }
    let rows_by_bytes = if source_size_bytes == 0 || source_physical_rows == 0 {
        target_rows
    } else {
        let estimated = (max_bytes_per_file as u128)
            .checked_mul(u128::from(source_physical_rows))
            .ok_or_else(|| {
                Error::format_capacity_exceeded("estimated output byte boundary exceeds u128")
            })?
            / u128::from(source_size_bytes);
        u64::try_from(estimated.max(1)).map_err(|_| {
            Error::format_capacity_exceeded("estimated output row boundary exceeds u64")
        })?
    };
    let rows_per_fragment = usize::try_from(target_rows.min(rows_by_bytes)).map_err(|_| {
        Error::format_capacity_exceeded("estimated output row boundary exceeds usize")
    })?;
    planned_output_row_counts(output_row_count, rows_per_fragment)
}

async fn preflight_v2_3_compaction_task(
    dataset: &Dataset,
    fragments: &[Fragment],
    options: &CompactionOptions,
) -> Result<V2_3CompactionAdmission> {
    let rows = match plan_default_compaction_rows(dataset, fragments).await {
        Err(Error::NotSupported { source, .. })
            if source
                .downcast_ref::<PlacementMaintenanceRequired>()
                .is_some() =>
        {
            return Ok(V2_3CompactionAdmission::NotAdmitted(source.to_string()));
        }
        other => other?,
    };
    let sequence = read_row_ids(&rows.logical_row_ids)?;
    let physical_order_matches_logical = rows.current_deletion_vectors.is_empty()
        && rows.logical_run_ends.is_empty()
        && !rows.requires_full_logical_sort;
    let mut binary_copy_plan = if !matches!(options.compaction_mode(), CompactionMode::Reencode)
        && physical_order_matches_logical
    {
        plan_binary_copy_impl(dataset, options, fragments).await?
    } else {
        None
    };
    if binary_copy_plan.as_ref().is_some_and(|plan| {
        plan.source_file_row_counts
            .iter()
            .map(|rows| u64::from(*rows))
            .sum::<u64>()
            != sequence.len()
    }) {
        log::debug!(
            "Binary copy disabled: source file rows differ from the admitted logical sequence"
        );
        binary_copy_plan = None;
    }
    let rewrite_strategy = match (options.compaction_mode(), binary_copy_plan) {
        (_, Some(plan)) => V2_3CompactionRewriteStrategy::BinaryCopy(plan),
        (CompactionMode::ForceBinaryCopy, None) => {
            let reason = if physical_order_matches_logical {
                "source files are not binary-copy compatible"
            } else {
                "physical source order differs from the admitted logical row sequence"
            };
            return Err(Error::not_supported_source(
                format!("storage-version-2.3 ForceBinaryCopy is not supported: {reason}").into(),
            ));
        }
        (CompactionMode::TryBinaryCopy | CompactionMode::Reencode, None) => {
            V2_3CompactionRewriteStrategy::Reencode
        }
    };
    let output_row_counts = match &rewrite_strategy {
        V2_3CompactionRewriteStrategy::BinaryCopy(plan) => plan.output_row_counts.clone(),
        V2_3CompactionRewriteStrategy::Reencode => {
            planned_reencode_output_row_counts(dataset, fragments, sequence.len(), options).await?
        }
    };
    let planned_fragments = output_row_counts
        .iter()
        .map(|count| Fragment::new(0).with_physical_rows(*count as usize))
        .collect::<Vec<_>>();

    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| {
            Error::internal("storage-version-2.3 manifest is missing RowAddressLayout")
        })?;
    let mut delta = RowAddressLayoutDelta {
        expected_layout_fingerprint: layout.fingerprint.clone(),
        ..RowAddressLayoutDelta::default()
    };
    let mut next_ordinal = 0_u32;
    let mut source_domains = BTreeMap::new();
    add_rewrite_provenance(
        dataset,
        &planned_fragments,
        &rows.logical_row_ids,
        rows.retired_logical_row_ids.as_deref(),
        &mut next_ordinal,
        &mut source_domains,
        &mut delta,
    )?;
    delta.source_domains = source_domains.into_values().collect();

    let transaction = TransactionBuilder::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: fragments.to_vec(),
                new_fragments: planned_fragments,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    )
    .row_address_layout_delta(Some(delta))
    .build();
    let context = RowAddressManifestApplyContext {
        current_deletion_vectors: rows.current_deletion_vectors.clone(),
        ..RowAddressManifestApplyContext::default()
    };
    let indices = dataset.load_indices().await?;
    match transaction.build_manifest_with_row_address_context(
        Some(dataset.manifest.as_ref()),
        indices.as_ref().clone(),
        "default-compaction-preflight.txn",
        &Default::default(),
        Some(&context),
    ) {
        Ok(_) => Ok(V2_3CompactionAdmission::Admitted(
            PreparedV2_3CompactionTask {
                plan: V2_3CompactionTaskPlan {
                    logical_row_ids: rows.logical_row_ids,
                    retired_logical_row_ids: rows.retired_logical_row_ids,
                    output_row_counts,
                    logical_run_ends: rows.logical_run_ends,
                    requires_full_logical_sort: rows.requires_full_logical_sort,
                    rewrite_strategy,
                },
                current_deletion_vectors: rows.current_deletion_vectors,
            },
        )),
        Err(Error::NotSupported { source, .. })
            if source
                .downcast_ref::<PlacementMaintenanceRequired>()
                .is_some() =>
        {
            Ok(V2_3CompactionAdmission::NotAdmitted(source.to_string()))
        }
        Err(error) => Err(error),
    }
}

fn preflight_prepared_v2_3_compaction_tasks(
    dataset: &Dataset,
    tasks: &[(TaskData, PreparedV2_3CompactionTask)],
    indices: &[IndexMetadata],
) -> Result<Option<String>> {
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| {
            Error::internal("storage-version-2.3 manifest is missing RowAddressLayout")
        })?;
    let mut delta = RowAddressLayoutDelta {
        expected_layout_fingerprint: layout.fingerprint.clone(),
        ..RowAddressLayoutDelta::default()
    };
    let mut next_ordinal = 0_u32;
    let mut source_domains = BTreeMap::new();
    let mut groups = Vec::with_capacity(tasks.len());
    let mut current_deletion_vectors = BTreeMap::new();

    for (task, prepared) in tasks {
        let planned_fragments = prepared
            .plan
            .output_row_counts
            .iter()
            .map(|count| Fragment::new(0).with_physical_rows(*count as usize))
            .collect::<Vec<_>>();
        add_rewrite_provenance(
            dataset,
            &planned_fragments,
            &prepared.plan.logical_row_ids,
            prepared.plan.retired_logical_row_ids.as_deref(),
            &mut next_ordinal,
            &mut source_domains,
            &mut delta,
        )?;
        groups.push(RewriteGroup {
            old_fragments: task.fragments.clone(),
            new_fragments: planned_fragments,
        });
        for (fragment_id, deletions) in &prepared.current_deletion_vectors {
            if current_deletion_vectors
                .insert(*fragment_id, deletions.clone())
                .is_some()
            {
                return Err(Error::internal(format!(
                    "physical fragment {fragment_id} occurs in more than one compaction group"
                )));
            }
        }
    }
    delta.source_domains = source_domains.into_values().collect();

    let transaction = TransactionBuilder::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups,
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    )
    .row_address_layout_delta(Some(delta))
    .build();
    let context = RowAddressManifestApplyContext {
        current_deletion_vectors,
        ..RowAddressManifestApplyContext::default()
    };
    match transaction.build_manifest_with_row_address_context(
        Some(dataset.manifest.as_ref()),
        indices.to_vec(),
        "default-compaction-combined-preflight.txn",
        &Default::default(),
        Some(&context),
    ) {
        Ok(_) => Ok(None),
        Err(Error::NotSupported { source, .. })
            if source
                .downcast_ref::<PlacementMaintenanceRequired>()
                .is_some() =>
        {
            Ok(Some(source.to_string()))
        }
        Err(error) => Err(error),
    }
}

/// A plan for what groups of fragments to compact.
///
/// See [plan_compaction()] for more details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactionPlan {
    pub tasks: Vec<TaskData>,
    pub read_version: u64,
    pub options: CompactionOptions,
    #[serde(default)]
    pub planning_metrics: CompactionMetrics,
}

impl CompactionPlan {
    /// Retrieve standalone tasks that be be executed in a distributed fashion.
    pub fn compaction_tasks(&self) -> impl Iterator<Item = CompactionTask> + '_ {
        let read_version = self.read_version;
        let options = self.options.clone();
        self.tasks.iter().map(move |task| CompactionTask {
            task: task.clone(),
            read_version,
            options: options.clone(),
        })
    }

    /// The number of tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// The version of the dataset that was read to produce this plan.
    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    /// The options used to produce this plan.
    pub fn options(&self) -> &CompactionOptions {
        &self.options
    }
}

/// Classification for one blob v2 row during compaction.
///
/// - `Null`: NULL row or Inline blob with position=0 and size=0.
/// - `External`: External blob referenced by URI.
/// - `DataBlob`: Inline/Packed/Dedicated blob stored in Lance files.
enum RowClass {
    Null,
    External,
    DataBlob,
}

/// Check if a row is a null Inline blob.
///
/// This matches `BlobV2StructuralEncoder`'s behavior of encoding null rows as
/// Inline with position=0 and size=0, and `collect_blob_entries_v2`'s behavior
/// of skipping them.
fn is_inline_null_blob(
    kind: BlobKind,
    position_col: &arrow::array::UInt64Array,
    size_col: &arrow::array::UInt64Array,
    index: usize,
) -> bool {
    if kind != BlobKind::Inline {
        return false;
    }
    let position_is_empty = position_col.is_null(index) || position_col.value(index) == 0;
    let size_is_empty = size_col.is_null(index) || size_col.value(index) == 0;
    position_is_empty && size_is_empty
}

/// Column views for the 5 fields in a blob v2 descriptor struct.
struct BlobV2Descriptor<'a> {
    kind_col: &'a arrow::array::UInt8Array,
    position_col: &'a arrow::array::UInt64Array,
    size_col: &'a arrow::array::UInt64Array,
    blob_uri_col: &'a arrow::array::StringArray,
    blob_id_col: &'a arrow::array::UInt32Array,
}

impl<'a> BlobV2Descriptor<'a> {
    /// Extract the 5 descriptor arrays from a blob v2 descriptor struct array.
    fn try_from_struct(struct_arr: &'a StructArray, column_name: &str) -> Result<Self> {
        let kind_col = struct_arr
            .column_by_name("kind")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `kind` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt8Type>();
        let position_col = struct_arr
            .column_by_name("position")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `position` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt64Type>();
        let size_col = struct_arr
            .column_by_name("size")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `size` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt64Type>();
        let blob_uri_col = struct_arr
            .column_by_name("blob_uri")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `blob_uri` field",
                    column_name
                ))
            })?
            .as_string::<i32>();
        let blob_id_col = struct_arr
            .column_by_name("blob_id")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `blob_id` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt32Type>();
        Ok(Self {
            kind_col,
            position_col,
            size_col,
            blob_uri_col,
            blob_id_col,
        })
    }
}

/// Result of row classification for blob v2 compaction.
struct RowClassification {
    row_classes: Vec<RowClass>,
    blob_read_addrs: Vec<u64>,
}

/// Classify each row of a blob v2 column as Null, External, or DataBlob.
fn classify_rows(
    struct_arr: &StructArray,
    descriptor: &BlobV2Descriptor<'_>,
    row_addrs: &arrow::array::UInt64Array,
    column_name: &str,
) -> Result<RowClassification> {
    let num_rows = struct_arr.len();
    let mut row_classes = Vec::with_capacity(num_rows);
    let mut blob_read_addrs = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        if struct_arr.is_null(i) || descriptor.kind_col.is_null(i) {
            row_classes.push(RowClass::Null);
        } else {
            let kind = BlobKind::try_from(descriptor.kind_col.value(i)).map_err(|e| {
                Error::internal(format!(
                    "Blob v2 column '{}' has invalid kind at row {}: {e}",
                    column_name, i
                ))
            })?;
            if kind == BlobKind::External {
                row_classes.push(RowClass::External);
            } else if is_inline_null_blob(kind, descriptor.position_col, descriptor.size_col, i) {
                row_classes.push(RowClass::Null);
            } else {
                row_classes.push(RowClass::DataBlob);
                blob_read_addrs.push(row_addrs.value(i));
            }
        }
    }

    Ok(RowClassification {
        row_classes,
        blob_read_addrs,
    })
}

/// Build a blob v2 user-view struct array from classification and descriptor.
///
/// Reads blob data lazily using row addresses to avoid materializing all blob
/// payloads in memory at once.
async fn build_user_view_struct(
    dataset: &Arc<Dataset>,
    descriptor: &BlobV2Descriptor<'_>,
    classification: &RowClassification,
    column_name: &str,
    num_rows: usize,
    null_buffer: Option<NullBuffer>,
) -> Result<StructArray> {
    let blob_files = if classification.blob_read_addrs.is_empty() {
        Vec::new()
    } else {
        super::blob::take_blobs_by_addresses(dataset, &classification.blob_read_addrs, column_name)
            .await?
    };

    let mut data_builder = LargeBinaryBuilder::with_capacity(num_rows, 0);
    let mut uri_builder = StringBuilder::with_capacity(num_rows, 0);
    let mut out_position_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(num_rows);
    let mut out_size_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(num_rows);

    let mut blob_file_idx = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_rows {
        match classification.row_classes[i] {
            RowClass::Null => {
                data_builder.append_null();
                uri_builder.append_null();
                out_position_builder.append_null();
                out_size_builder.append_null();
            }
            RowClass::External => {
                data_builder.append_null();
                let base_id = descriptor.blob_id_col.value(i);
                let uri_val = descriptor.blob_uri_col.value(i);
                if base_id == 0 {
                    uri_builder.append_value(uri_val);
                } else {
                    let base = dataset.manifest().base_paths.get(&base_id).ok_or_else(|| {
                        Error::internal(format!(
                            "External blob in column '{}' references unknown base_id {}",
                            column_name, base_id
                        ))
                    })?;
                    let absolute_uri = format!("{}/{}", base.path.trim_end_matches('/'), uri_val);
                    uri_builder.append_value(&absolute_uri);
                }
                if descriptor.position_col.is_null(i) {
                    out_position_builder.append_null();
                } else {
                    out_position_builder.append_value(descriptor.position_col.value(i));
                }
                if descriptor.size_col.is_null(i) {
                    out_size_builder.append_null();
                } else {
                    out_size_builder.append_value(descriptor.size_col.value(i));
                }
            }
            RowClass::DataBlob => {
                let data = blob_files[blob_file_idx].read().await?;
                blob_file_idx += 1;
                data_builder.append_value(data.as_ref());
                uri_builder.append_null();
                out_position_builder.append_null();
                out_size_builder.append_null();
            }
        }
    }

    Ok(StructArray::try_new(
        lance_core::datatypes::BLOB_V2_USER_FIELDS.clone(),
        vec![
            Arc::new(data_builder.finish()),
            Arc::new(uri_builder.finish()),
            Arc::new(out_position_builder.finish()),
            Arc::new(out_size_builder.finish()),
        ],
        null_buffer,
    )?)
}

async fn transform_blob_v2_batch(
    dataset: &Arc<Dataset>,
    schema: &lance_core::datatypes::Schema,
    batch: RecordBatch,
) -> Result<RecordBatch> {
    let row_addr_idx = batch
        .schema()
        .column_with_name(lance_core::ROW_ADDR)
        .ok_or_else(|| {
            Error::internal(format!(
                "_rowaddr column missing from batch for blob v2 compaction, columns: {:?}",
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            ))
        })?
        .0;
    let row_addrs = batch.column(row_addr_idx).as_primitive::<UInt64Type>();

    let mut new_columns: Vec<Arc<dyn Array>> = Vec::new();
    let mut new_fields: Vec<Arc<arrow_schema::Field>> = Vec::new();

    let batch_schema = batch.schema();
    for (col_idx, field) in batch_schema.fields().iter().enumerate() {
        if field.name() == lance_core::ROW_ADDR {
            continue;
        }

        let lance_field = schema.field(field.name());
        let is_blob_v2 = lance_field.is_some_and(|f| f.is_blob_v2());

        if !is_blob_v2 {
            new_columns.push(batch.column(col_idx).clone());
            new_fields.push(field.clone());
            continue;
        }

        let struct_arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 column '{}' expected StructArray, got {:?}",
                    field.name(),
                    batch.column(col_idx).data_type()
                ))
            })?;

        let column_name = field.name();
        let descriptor = BlobV2Descriptor::try_from_struct(struct_arr, column_name)?;
        let classification = classify_rows(struct_arr, &descriptor, row_addrs, column_name)?;
        let num_rows = struct_arr.len();

        let new_struct = build_user_view_struct(
            dataset,
            &descriptor,
            &classification,
            column_name,
            num_rows,
            struct_arr.nulls().cloned(),
        )
        .await?;

        new_columns.push(Arc::new(new_struct));
        let logical_field = arrow_schema::Field::from(lance_field.ok_or_else(|| {
            Error::internal(format!(
                "Blob v2 column '{}' missing from dataset schema during compaction",
                field.name()
            ))
        })?);
        new_fields.push(Arc::new(
            arrow_schema::Field::new(
                field.name(),
                lance_core::datatypes::BLOB_V2_USER_TYPE.clone(),
                field.is_nullable(),
            )
            .with_metadata(logical_field.metadata().clone()),
        ));
    }

    let new_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
        new_fields
            .iter()
            .map(|f| f.as_ref().clone())
            .collect::<Vec<_>>(),
        batch_schema.metadata().clone(),
    ));

    Ok(RecordBatch::try_new(new_schema, new_columns)?)
}

fn blob_v2_user_view_schema(
    dataset: &Dataset,
    input_schema: &arrow_schema::Schema,
) -> Arc<arrow_schema::Schema> {
    let fields = input_schema
        .fields()
        .iter()
        .filter_map(|field| {
            if field.name() == lance_core::ROW_ADDR {
                return None;
            }
            let Some(lance_field) = dataset
                .schema()
                .field(field.name())
                .filter(|field| field.is_blob_v2())
            else {
                return Some(field.clone());
            };
            let logical_field = arrow_schema::Field::from(lance_field);
            Some(Arc::new(
                arrow_schema::Field::new(
                    field.name(),
                    lance_core::datatypes::BLOB_V2_USER_TYPE.clone(),
                    field.is_nullable(),
                )
                .with_metadata(logical_field.metadata().clone()),
            ))
        })
        .collect::<Vec<_>>();
    Arc::new(arrow_schema::Schema::new_with_metadata(
        fields,
        input_schema.metadata().clone(),
    ))
}

fn transform_blob_v2_stream(
    dataset: &Dataset,
    stream: SendableRecordBatchStream,
) -> SendableRecordBatchStream {
    let output_schema = blob_v2_user_view_schema(dataset, stream.schema().as_ref());
    let dataset = Arc::new(dataset.clone());
    let dataset_schema = dataset.schema().clone();
    let transformed = stream.then(move |batch_result| {
        let dataset = dataset.clone();
        let schema = dataset_schema.clone();
        async move {
            let batch = batch_result?;
            transform_blob_v2_batch(&dataset, &schema, batch)
                .await
                .map_err(|error| datafusion::error::DataFusionError::External(Box::new(error)))
        }
    });
    Box::pin(RecordBatchStreamAdapter::new(output_schema, transformed))
}

/// Build a scan reader for rewrite and optionally capture row IDs.
///
/// Parameters:
/// - `dataset`: Dataset handle used to create the scanner.
/// - `fragments`: When `with_frags` is true, restrict the scan to these old fragments
///   and preserve insertion order.
/// - `batch_size`: Optional batch size; if provided, set it on the scanner to control
///   read batching.
/// - `io_buffer_size`: Optional I/O buffer size in bytes; if provided, set it on the
///   scanner to control how much data is queued during reads.
/// - `with_frags`: Whether to scan only the specified old fragments and force
///   in-order reading.
/// - `capture_row_ids`: When index remapping is needed, include and capture the
///   `_rowid` column from the stream.
/// - `v2_3_plan`: When physical placement is not globally ordered, merge its
///   monotonic physical runs by `_rowid`, or use a full spillable sort only for
///   an internally non-monotonic source fragment.
///
/// Returns:
/// - `SendableRecordBatchStream`: The batch stream (with `_rowid` removed if captured)
///   to feed the rewrite path.
/// - `Option<Receiver<CapturedRowIds>>`: A receiver to obtain captured row IDs after the
///   stream completes; `None` if not capturing.
/// - `bool`: Whether the dataset has blob v2 columns and the stream includes `_rowaddr`.
async fn prepare_reader(
    dataset: &Dataset,
    fragments: &[Fragment],
    batch_size: Option<usize>,
    io_buffer_size: Option<u64>,
    with_frags: bool,
    capture_row_ids: bool,
    v2_3_plan: Option<&V2_3CompactionTaskPlan>,
) -> Result<(
    SendableRecordBatchStream,
    Option<std::sync::mpsc::Receiver<CapturedRowIds>>,
    bool,
)> {
    let mut scanner = dataset.scan();
    let has_legacy_blob_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob() && !field.is_blob_v2());
    if has_legacy_blob_columns {
        scanner.blob_handling(BlobHandling::AllBinary);
    }
    let has_blob_v2_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob_v2());
    if has_blob_v2_columns {
        scanner.with_row_address();
    }
    if let Some(bs) = batch_size {
        scanner.batch_size(bs);
    }
    if let Some(io_buffer_size) = io_buffer_size {
        scanner.io_buffer_size(io_buffer_size);
    }
    let needs_logical_reorder = v2_3_plan
        .is_some_and(|plan| plan.requires_full_logical_sort || !plan.logical_run_ends.is_empty());
    if needs_logical_reorder && !with_frags {
        return Err(Error::internal(
            "logical-order compaction requires an explicit source-fragment set",
        ));
    }
    if capture_row_ids || needs_logical_reorder {
        scanner.with_row_id();
    }

    let data = if let Some(plan) = v2_3_plan.filter(|_| needs_logical_reorder) {
        let ordering_for = |plan: &Arc<dyn ExecutionPlan>| -> Result<LexOrdering> {
            let row_id_sort = PhysicalSortExpr {
                expr: expressions::col(lance_core::ROW_ID, plan.schema().as_ref())?,
                options: arrow::compute::SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            };
            LexOrdering::new([row_id_sort])
                .ok_or_else(|| Error::internal("logical row ordering cannot be empty"))
        };

        let physical_plan: Arc<dyn ExecutionPlan> = if plan.requires_full_logical_sort {
            if !plan.logical_run_ends.is_empty() {
                return Err(Error::internal(
                    "full logical sort cannot also declare sort-preserving input runs",
                ));
            }
            scanner
                .with_fragments(fragments.to_vec())
                .scan_in_order(false);
            let input = scanner.create_plan().await?;
            let ordering = ordering_for(&input)?;
            Arc::new(SortExec::new(ordering, input))
        } else {
            let mut previous_end = 0_usize;
            let mut inputs =
                Vec::<Arc<dyn ExecutionPlan>>::with_capacity(plan.logical_run_ends.len());
            for end in &plan.logical_run_ends {
                let end = usize::try_from(*end)
                    .map_err(|_| Error::internal("logical run boundary does not fit in usize"))?;
                if end <= previous_end || end > fragments.len() {
                    return Err(Error::internal(format!(
                        "invalid logical run boundary {end} after {previous_end} for {} source fragments",
                        fragments.len()
                    )));
                }
                let mut run_scanner = scanner.clone();
                run_scanner
                    .with_fragments(fragments[previous_end..end].to_vec())
                    .scan_in_order(true);
                let input = run_scanner.create_plan().await?;
                if input.output_partitioning().partition_count() != 1 {
                    return Err(Error::internal(format!(
                        "ordered compaction run produced {} partitions instead of one",
                        input.output_partitioning().partition_count()
                    )));
                }
                inputs.push(input);
                previous_end = end;
            }
            if previous_end != fragments.len() {
                return Err(Error::internal(format!(
                    "logical run boundaries cover {previous_end} of {} source fragments",
                    fragments.len()
                )));
            }
            let union = UnionExec::try_new(inputs)?;
            let ordering = ordering_for(&union)?;
            Arc::new(SortPreservingMergeExec::new(ordering, union))
        };
        execute_plan(physical_plan, scanner.execution_options())?
    } else {
        if with_frags {
            scanner
                .with_fragments(fragments.to_vec())
                .scan_in_order(true);
        }
        SendableRecordBatchStream::from(scanner.try_into_stream().await?)
    };

    if capture_row_ids || needs_logical_reorder {
        let (data_no_row_ids, rx) =
            make_rowid_capture_stream(data, dataset.manifest.has_stable_row_identity())?;
        Ok((data_no_row_ids, Some(rx), has_blob_v2_columns))
    } else {
        Ok((data, None, has_blob_v2_columns))
    }
}

/// A single group of fragments to compact, which is a view into the compaction
/// plan. We keep the `replace_range` indices so we can map the result of the
/// compact back to the fragments it replaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskData {
    /// The fragments to compact.
    pub fragments: Vec<Fragment>,
    /// Exact storage-version-2.3 provenance and output row boundaries,
    /// admitted from manifest/footer metadata before any page read or data-file PUT.
    #[serde(default)]
    pub v2_3_plan: Option<V2_3CompactionTaskPlan>,
}

/// Metadata-only plan for one storage-version-2.3 compaction task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2_3CompactionTaskPlan {
    logical_row_ids: Vec<u8>,
    retired_logical_row_ids: Option<Vec<u8>>,
    output_row_counts: Vec<u32>,
    /// Exclusive source-fragment boundaries for already-sorted input runs.
    /// Multiple runs are merged without sorting user columns.
    #[serde(default)]
    logical_run_ends: Vec<u32>,
    /// A defensive fallback for a physical fragment whose own placement is
    /// internally non-monotonic.
    #[serde(default)]
    requires_full_logical_sort: bool,
    /// Data rewrite selected and admitted before any output object is created.
    #[serde(default)]
    rewrite_strategy: V2_3CompactionRewriteStrategy,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
enum V2_3CompactionRewriteStrategy {
    #[default]
    Reencode,
    BinaryCopy(BinaryCopyPlan),
}

/// A standalone task that can be serialized and sent to another machine for
/// execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionTask {
    pub task: TaskData,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl CompactionTask {
    /// Run the compaction task and return the result.
    ///
    /// This result should be later passed to [commit_compaction()] to commit
    /// the changes to the dataset.
    ///
    /// Note: you should pass the version of the dataset that is the same as
    /// the read version for this task (the same version from which the
    /// compaction was planned).
    pub async fn execute(&self, dataset: &Dataset) -> Result<RewriteResult> {
        let dataset = if dataset.manifest.version == self.read_version {
            Cow::Borrowed(dataset)
        } else {
            Cow::Owned(dataset.checkout_version(self.read_version).await?)
        };
        rewrite_files(dataset, self.task.clone(), &self.options).await
    }
}

impl CompactionPlan {
    fn new(read_version: u64, options: CompactionOptions) -> Self {
        Self {
            tasks: Vec::new(),
            read_version,
            options,
            planning_metrics: CompactionMetrics::default(),
        }
    }

    fn extend_tasks(&mut self, tasks: impl IntoIterator<Item = TaskData>) {
        self.tasks.extend(tasks);
    }

    fn tasks(&self) -> &[TaskData] {
        &self.tasks
    }
}

#[derive(Debug, Clone)]
enum CompactionCandidacy {
    /// Compact the fragment if it has neighbors that are also candidates
    CompactWithNeighbors,
    /// Compact the fragment regardless.
    CompactItself,
}

/// Internal struct used for planning compaction.
struct CandidateBin {
    pub fragments: Vec<Fragment>,
    pub pos_range: Range<usize>,
    pub candidacy: Vec<CompactionCandidacy>,
    pub row_counts: Vec<usize>,
    pub indices: Vec<usize>,
}

impl CandidateBin {
    /// Return true if compacting these fragments wouldn't do anything.
    fn is_noop(&self) -> bool {
        if self.fragments.is_empty() {
            return true;
        }
        // If there's only one fragment, it's a noop if it's not CompactItself
        if self.fragments.len() == 1 {
            matches!(self.candidacy[0], CompactionCandidacy::CompactWithNeighbors)
        } else {
            false
        }
    }

    /// Split into one or more bins with at least `min_num_rows` in them.
    fn split_for_size(mut self, min_num_rows: usize) -> Vec<Self> {
        let mut bins = Vec::new();

        loop {
            let mut bin_len = 0;
            let mut bin_row_count = 0;
            while bin_row_count < min_num_rows && bin_len < self.row_counts.len() {
                bin_row_count += self.row_counts[bin_len];
                bin_len += 1;
            }

            // If there's enough remaining to make another worthwhile bin, then
            // push what we have as a bin.
            if self.row_counts[bin_len..].iter().sum::<usize>() >= min_num_rows {
                bins.push(Self {
                    fragments: self.fragments.drain(0..bin_len).collect(),
                    pos_range: self.pos_range.start..(self.pos_range.start + bin_len),
                    candidacy: self.candidacy.drain(0..bin_len).collect(),
                    row_counts: self.row_counts.drain(0..bin_len).collect(),
                    // By the time we are splitting for size we are done considering indices
                    indices: Vec::new(),
                });
                self.pos_range.start += bin_len;
            } else {
                // Otherwise, just push the remaining fragments into the last bin
                bins.push(self);
                break;
            }
        }

        bins
    }
}

async fn load_index_fragmaps(dataset: &Dataset) -> Result<Vec<RoaringBitmap>> {
    // Storage version 2.3 index coverage is expressed in the stable logical
    // domain. Physical fragment movement neither changes that coverage nor
    // requires posting remap, so it must not split physical compaction bins.
    if dataset.manifest.uses_stable_logical_row_addresses() {
        return Ok(Vec::new());
    }
    let indices = dataset.load_indices().await?;
    let mut index_fragmaps = Vec::with_capacity(indices.len());
    // System indices (fragment-reuse, mem-wal) don't define data coverage and
    // aren't remapped per rewrite group, so they must not constrain compaction
    // bins -- otherwise deferred compaction's fragment-reuse index repeatedly
    // splits the small-fragment run and they never coalesce.
    for index in indices.iter().filter(|idx| !is_system_index(idx)) {
        if let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() {
            index_fragmaps.push(fragment_bitmap.clone());
        } else {
            let dataset_at_index = dataset.checkout_version(index.dataset_version).await?;
            let frags = dataset_at_index
                .manifest
                .fragments
                .iter()
                .map(|fragment| fragment.id as u32);
            index_fragmaps.push(RoaringBitmap::from_iter(frags));
        }
    }
    Ok(index_fragmaps)
}

pub async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<CompactionPlan> {
    let planner = DefaultCompactionPlanner::new(options.clone());
    planner.plan(dataset).await
}

/// The result of a single compaction task.
///
/// This should be passed to [commit_compaction()] to commit the operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewriteResult {
    pub metrics: CompactionMetrics,
    pub new_fragments: Vec<Fragment>,
    /// The version of the dataset that was read to perform this compaction.
    pub read_version: u64,
    /// The original fragments being replaced
    pub original_fragments: Vec<Fragment>,
    /// Serialized `RoaringTreemap` of the row addresses from the original
    /// fragments that were read during compaction.
    ///
    /// - `None` when configured with stable row IDs because the row ID
    ///   sequences are rechunked directly.
    /// - `Some` then these addresses are either (1) written to storage for
    ///   deferred index remap post-processing, or (2) used with reserved
    ///   fragment IDs to build old-to-new mappings.
    pub row_addrs: Option<Vec<u8>>,
    /// Physical row addresses in destination order for an ordered rewrite of
    /// a dataset without stable row identity.  Unlike `row_addrs`, this uses
    /// the RowIdSequence codec because the addresses are intentionally not
    /// sorted by their source physical location.
    #[serde(default)]
    pub ordered_row_addrs: Option<Vec<u8>>,
    /// Actual logical row-address sequence emitted by a storage-version-2.3
    /// rewrite, compressed with the stable RowIdSequence codec.
    #[serde(default)]
    pub logical_row_ids: Option<Vec<u8>>,
    /// Logical identities whose already-deleted physical rows were reclaimed
    /// by this rewrite.
    #[serde(default)]
    pub retired_logical_row_ids: Option<Vec<u8>>,
}

async fn reserve_fragment_ids(
    dataset: &Dataset,
    fragments: impl ExactSizeIterator<Item = &mut Fragment>,
) -> Result<()> {
    if fragments.len() == 0 {
        return Ok(());
    }
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments {
            num_fragments: fragments.len() as u32,
        },
        None,
    );

    let (manifest, _) = commit_transaction(
        dataset,
        dataset.object_store.as_ref(),
        dataset.commit_handler.as_ref(),
        &transaction,
        &Default::default(),
        &Default::default(),
        dataset.manifest_location.naming_scheme,
        None,
    )
    .await?;

    // Need +1 since max_fragment_id is inclusive in this case and ranges are exclusive
    let new_max_exclusive = manifest.max_fragment_id.unwrap_or(0) + 1;
    let reserved_ids = (new_max_exclusive - fragments.len() as u32)..(new_max_exclusive);

    for (fragment, new_id) in fragments.zip(reserved_ids) {
        fragment.id = new_id as u64;
    }

    Ok(())
}

/// Rewrite the files in a single task.
///
/// This assumes that the dataset is the correct read version to be compacted.
async fn rewrite_files(
    dataset: Cow<'_, Dataset>,
    task: TaskData,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let mut metrics = CompactionMetrics::default();

    if task.fragments.is_empty() {
        return Ok(RewriteResult {
            metrics,
            new_fragments: Vec::new(),
            read_version: dataset.manifest.version,
            original_fragments: task.fragments,
            row_addrs: None,
            ordered_row_addrs: None,
            logical_row_ids: None,
            retired_logical_row_ids: None,
        });
    }

    metrics.groups_planned = 1;
    let uses_v2_3_row_addresses = dataset.manifest.uses_stable_logical_row_addresses();
    let v2_3_plan = if uses_v2_3_row_addresses {
        if let Some(plan) = task.v2_3_plan.clone() {
            metrics.groups_admitted = 1;
            Some(plan)
        } else {
            match preflight_v2_3_compaction_task(dataset.as_ref(), &task.fragments, options).await?
            {
                V2_3CompactionAdmission::Admitted(prepared) => {
                    metrics.groups_admitted = 1;
                    Some(prepared.plan)
                }
                V2_3CompactionAdmission::NotAdmitted(reason) => {
                    metrics.groups_not_admitted = 1;
                    warn!(
                        "Skipping storage-version-2.3 compaction group before page reads or data writes: {reason}"
                    );
                    return Ok(RewriteResult {
                        metrics,
                        new_fragments: Vec::new(),
                        read_version: dataset.manifest.version,
                        original_fragments: Vec::new(),
                        row_addrs: None,
                        ordered_row_addrs: None,
                        logical_row_ids: None,
                        retired_logical_row_ids: None,
                    });
                }
            }
        }
    } else {
        metrics.groups_admitted = 1;
        None
    };

    let previous_writer_version = &dataset.manifest.writer_version;
    // The versions of Lance prior to when we started writing the writer version
    // sometimes wrote incorrect `Fragment.physical_rows` values, so we should
    // make sure to recompute them.
    // See: https://github.com/lance-format/lance/issues/1531
    let recompute_stats = previous_writer_version.is_none();

    // It's possible the fragments are old and don't have physical rows or
    // num deletions recorded. If that's the case, we need to grab and set that
    // information.
    let fragments = migrate_fragments(dataset.as_ref(), &task.fragments, recompute_stats).await?;
    let num_rows = fragments
        .iter()
        .map(|f| f.physical_rows.unwrap() as u64)
        .sum::<u64>();
    // Physical-address indices need remapping. Stable logical postings do not.
    let needs_remapping = !dataset.manifest.has_stable_row_identity();
    let mut new_fragments: Vec<Fragment>;
    let task_id = uuid::Uuid::new_v4();
    log::info!(
        "Compaction task {}: Begin compacting {} rows across {} fragments",
        task_id,
        num_rows,
        fragments.len()
    );
    let mode = options.compaction_mode();
    let binary_copy_plan = if let Some(plan) = v2_3_plan.as_ref() {
        match &plan.rewrite_strategy {
            V2_3CompactionRewriteStrategy::Reencode => None,
            V2_3CompactionRewriteStrategy::BinaryCopy(plan) => Some(plan.clone()),
        }
    } else {
        plan_binary_copy_impl(dataset.as_ref(), options, &fragments).await?
    };
    if binary_copy_plan.is_none() && matches!(mode, CompactionMode::ForceBinaryCopy) {
        return Err(Error::not_supported_source(
            format!("compaction task {}: binary copy is not supported", task_id).into(),
        ));
    }
    let mut row_ids_rx: Option<std::sync::mpsc::Receiver<CapturedRowIds>> = None;
    let mut reader: Option<SendableRecordBatchStream> = None;

    if binary_copy_plan.is_none() {
        let (prepared_reader, rx_initial, has_blob_v2_columns) = prepare_reader(
            dataset.as_ref(),
            &fragments,
            options.batch_size,
            options.io_buffer_size,
            true,
            needs_remapping,
            v2_3_plan.as_ref(),
        )
        .await?;
        row_ids_rx = rx_initial;

        let mut rows_read = 0;
        let schema = prepared_reader.schema();
        let reader_with_progress = prepared_reader.inspect_ok(move |batch| {
            rows_read += batch.num_rows();
            log::info!(
                "Compaction task {}: Read progress {}/{}",
                task_id,
                rows_read,
                num_rows,
            );
        });

        if has_blob_v2_columns {
            let progress_stream =
                Box::pin(RecordBatchStreamAdapter::new(schema, reader_with_progress));
            reader = Some(transform_blob_v2_stream(dataset.as_ref(), progress_stream));
        } else {
            reader = Some(Box::pin(RecordBatchStreamAdapter::new(
                schema,
                reader_with_progress,
            )));
        }
    }

    let mut params = WriteParams {
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        mode: WriteMode::Append,
        // External blobs may reference URIs outside the dataset's base_paths
        // (e.g. absolute file:// URIs with base_id == 0). Without this flag
        // the writer would reject such blobs.
        allow_external_blob_outside_bases: true,
        ..Default::default()
    };
    if let Some(max_bytes_per_file) = options.max_bytes_per_file {
        params.max_bytes_per_file = max_bytes_per_file;
    }
    if uses_v2_3_row_addresses {
        if let Some(plan) = v2_3_plan.as_ref()
            && matches!(
                &plan.rewrite_strategy,
                V2_3CompactionRewriteStrategy::Reencode
            )
            && let Some(rows_per_file) = plan.output_row_counts.first()
        {
            params.max_rows_per_file = *rows_per_file as usize;
        }
        // The byte target was converted into a deterministic row or source-file
        // boundary during preflight. Runtime bytes cannot change that boundary.
        params.max_bytes_per_file = usize::MAX;
    }

    if dataset.manifest.uses_stable_row_ids() {
        params.enable_stable_row_ids = true;
    }

    if let Some(binary_copy_plan) = binary_copy_plan.as_ref() {
        new_fragments = rewrite_files_binary_copy(
            dataset.as_ref(),
            &fragments,
            binary_copy_plan,
            options.binary_copy_read_batch_bytes,
        )
        .await?;

        if new_fragments.is_empty() && matches!(mode, CompactionMode::ForceBinaryCopy) {
            return Err(Error::not_supported_source(
                format!("compaction task {}: binary copy is not supported", task_id).into(),
            ));
        }

        if needs_remapping {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut addrs = RoaringTreemap::new();
            for frag in &fragments {
                let frag_id = frag.id as u32;
                let count = u64::try_from(frag.physical_rows.unwrap_or(0)).map_err(|_| {
                    Error::internal(format!(
                        "Fragment {} has too many physical rows to represent as row addresses",
                        frag.id
                    ))
                })?;
                let start = u64::from(lance_core::utils::address::RowAddress::first_row(frag_id));
                addrs.insert_range(start..start + count);
            }
            let captured = CapturedRowIds::AddressStyle(addrs);
            let _ = tx.send(captured);
            row_ids_rx = Some(rx);
        }
    } else {
        let (frags, _) = write_fragments_internal(
            Some(dataset.as_ref()),
            dataset.object_store.clone(),
            &dataset.base,
            dataset.schema().clone(),
            reader.expect("reader must be prepared for non-binary-copy path"),
            params,
            None,
        )
        .await?;
        new_fragments = frags;
    }

    log::info!("Compaction task {}: file written", task_id);

    // Wrap in an async block so `?` returns into `provenance_result` and we can
    // run cleanup before propagating the error.
    let provenance_result: Result<RewriteRowProvenance> = async {
        let mut row_addrs = None;
        let mut logical_row_ids = None;
        let retired = if let Some(plan) = v2_3_plan.as_ref() {
            if plan.requires_full_logical_sort || !plan.logical_run_ends.is_empty() {
                let captured = row_ids_rx
                    .take()
                    .ok_or_else(|| {
                        Error::internal(
                            "logical-order compaction did not capture its emitted row addresses",
                        )
                    })?
                    .try_recv()
                    .map_err(|error| {
                        Error::internal(format!(
                            "logical-order compaction row-address capture did not complete: {error}"
                        ))
                    })?;
                let CapturedRowIds::SequenceStyle(captured) = captured else {
                    return Err(Error::internal(
                        "logical-order compaction captured physical row addresses",
                    ));
                };
                let planned = read_row_ids(&plan.logical_row_ids)?;
                if captured.len() != planned.len() || !captured.iter().eq(planned.iter()) {
                    return Err(Error::internal(
                        "logical-order compaction output differs from its admitted row sequence",
                    ));
                }
            }
            let actual_row_counts = new_fragments
                .iter()
                .map(|fragment| {
                    u32::try_from(fragment.physical_rows.ok_or_else(|| {
                        Error::invalid_input(
                            "storage-version-2.3 compaction output is missing physical_rows",
                        )
                    })?)
                    .map_err(|_| {
                        Error::invalid_input(
                            "storage-version-2.3 compaction output rows exceed u32",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if actual_row_counts != plan.output_row_counts {
                return Err(Error::invalid_input(format!(
                    "storage-version-2.3 compaction output boundaries changed after admission: planned {:?}, wrote {:?}",
                    plan.output_row_counts, actual_row_counts
                )));
            }
            logical_row_ids = Some(plan.logical_row_ids.clone());
            plan.retired_logical_row_ids.clone()
        } else if let Some(row_ids_rx) = row_ids_rx {
            let captured_ids = row_ids_rx
                .try_recv()
                .map_err(|err| Error::internal(format!("Failed to receive row ids: {}", err)))?;
            match captured_ids {
                CapturedRowIds::AddressStyle(addresses) if needs_remapping => {
                    let mut serialized = Vec::with_capacity(addresses.serialized_size());
                    addresses.serialize_into(&mut serialized)?;
                    row_addrs = Some(serialized);
                }
                CapturedRowIds::SequenceStyle(_) | CapturedRowIds::AddressStyle(_) => {
                    return Err(Error::internal(
                        "compaction captured row identities using the wrong address domain",
                    ));
                }
            }
            None
        } else if dataset.manifest.uses_legacy_stable_row_ids() {
            log::info!("Compaction task {}: rechunking stable row ids", task_id);
            rechunk_stable_row_ids(dataset.as_ref(), &mut new_fragments, &fragments).await?;
            None
        } else {
            None
        };

        if dataset.manifest.has_stable_row_identity() {
            let source_order_matches_logical = v2_3_plan.as_ref().is_some_and(|plan| {
                !plan.requires_full_logical_sort && plan.logical_run_ends.is_empty()
            });
            recalc_versions_for_rewritten_fragments(
                dataset.as_ref(),
                &mut new_fragments,
                &fragments,
                logical_row_ids.as_deref(),
                source_order_matches_logical,
            )
            .await?;
        }
        Ok((row_addrs, logical_row_ids, retired))
    }
    .await;

    let (row_addrs, logical_row_ids, retired_logical_row_ids) = match provenance_result {
        Ok(v) => v,
        Err(e) => {
            cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                .await;
            return Err(e);
        }
    };

    metrics.files_removed = task
        .fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();
    metrics.fragments_removed = task.fragments.len();
    metrics.fragments_added = new_fragments.len();
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();

    log::info!("Compaction task {}: completed", task_id);

    Ok(RewriteResult {
        metrics,
        new_fragments,
        read_version: dataset.manifest.version,
        original_fragments: fragments,
        row_addrs,
        ordered_row_addrs: None,
        logical_row_ids,
        retired_logical_row_ids,
    })
}

async fn rechunk_stable_row_ids(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    old_fragments: &[Fragment],
) -> Result<()> {
    let mut old_sequences = load_row_id_sequences(dataset, old_fragments)
        .try_collect::<Vec<_>>()
        .await?;
    // Should sort them back into original order.
    old_sequences.sort_by_key(|(frag_id, _)| {
        old_fragments
            .iter()
            .position(|frag| frag.id as u32 == *frag_id)
            .expect("Fragment not found")
    });

    // Need to remove deleted rows
    futures::stream::iter(old_sequences.iter_mut().zip(old_fragments.iter()))
        .map(Ok)
        .try_for_each(|((_, seq), frag)| async move {
            if let Some(deletion_file) = &frag.deletion_file {
                let deletions = read_dataset_deletion_file(dataset, frag.id, deletion_file).await?;

                let mut new_seq = seq.as_ref().clone();
                new_seq.mask(deletions.to_sorted_iter())?;
                *seq = Arc::new(new_seq);
            }
            Ok::<(), crate::Error>(())
        })
        .await?;

    debug_assert_eq!(
        { old_sequences.iter().map(|(_, seq)| seq.len()).sum::<u64>() },
        {
            new_fragments
                .iter()
                .map(|frag| frag.physical_rows.unwrap() as u64)
                .sum::<u64>()
        },
        "{:?}",
        old_sequences
    );

    let new_sequences = lance_table::rowids::rechunk_sequences(
        old_sequences
            .into_iter()
            .map(|(_, seq)| seq.as_ref().clone()),
        new_fragments
            .iter()
            .map(|frag| frag.physical_rows.unwrap() as u64),
        false,
    )?;

    for (fragment, sequence) in new_fragments.iter_mut().zip(new_sequences) {
        // TODO: if large enough, serialize to separate file
        let serialized = lance_table::rowids::write_row_ids(&sequence);
        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
    }

    Ok(())
}

/// After row id rechunking, preserve per-row latest update versions by masking deletions and rechunking
async fn recalc_versions_for_rewritten_fragments(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    old_fragments: &[Fragment],
    logical_row_ids: Option<&[u8]>,
    source_order_matches_logical: bool,
) -> Result<()> {
    if dataset.manifest.uses_stable_logical_row_addresses() {
        let logical_row_ids = logical_row_ids.ok_or_else(|| {
            Error::internal("storage-version-2.3 compaction is missing its logical row sequence")
        })?;
        let logical_row_ids = read_row_ids(logical_row_ids)?;
        let chunk_sizes = new_fragments
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap_or_default() as u64)
            .collect::<Vec<_>>();
        let expected_rows = chunk_sizes.iter().sum::<u64>();
        if logical_row_ids.len() != expected_rows {
            return Err(Error::invalid_input(format!(
                "storage-version-2.3 compaction row-version provenance has {} logical rows for {expected_rows} output rows",
                logical_row_ids.len()
            )));
        }

        let source_versions = if source_order_matches_logical {
            source_ordered_v2_3_version_sequences(dataset, old_fragments).await?
        } else {
            None
        };
        let (created, updated) = match source_versions {
            Some((created, updated)) => (created, updated),
            None => {
                resolve_logical_row_version_sequences_bounded(dataset, &logical_row_ids).await?
            }
        };
        let created = lance_table::rowids::version::rechunk_version_sequences_by_run_lengths(
            created,
            chunk_sizes.clone(),
            false,
        )?;
        let updated = lance_table::rowids::version::rechunk_version_sequences_by_run_lengths(
            updated,
            chunk_sizes,
            false,
        )?;
        for ((fragment, created), updated) in new_fragments.iter_mut().zip(created).zip(updated) {
            fragment.created_at_version_meta = Some(
                lance_table::format::RowDatasetVersionMeta::from_sequence(&created)?,
            );
            fragment.last_updated_at_version_meta = Some(
                lance_table::format::RowDatasetVersionMeta::from_sequence(&updated)?,
            );
        }
        return Ok(());
    }

    // Load old per-row last_updated_at version sequences
    let mut old_last_updated_sequences: Vec<lance_table::format::RowDatasetVersionSequence> =
        Vec::with_capacity(old_fragments.len());
    // Load old per-row created_at version sequences
    let mut old_created_at_sequences: Vec<lance_table::format::RowDatasetVersionSequence> =
        Vec::with_capacity(old_fragments.len());

    for frag in old_fragments.iter() {
        let row_count = if let Some(row_id_meta) = &frag.row_id_meta {
            match row_id_meta {
                RowIdMeta::Inline(data) => lance_table::rowids::read_row_ids(data)?.len(),
                RowIdMeta::External(_file) => frag.physical_rows.unwrap_or(0) as u64,
            }
        } else {
            frag.physical_rows.unwrap_or(0) as u64
        };

        // Load created_at sequence (default to version 1 if missing)
        let mut created_at_seq = if let Some(version_meta) = &frag.created_at_version_meta {
            version_meta.load_sequence().map_err(|e| {
                Error::internal(format!("Failed to load created_at version sequence: {}", e))
            })?
        } else {
            // Default: treat all rows as created at version 1
            lance_table::format::RowDatasetVersionSequence::from_uniform_row_count(row_count, 1)
        };

        // Load last_updated_at sequence (default to same as created_at sequence)
        let mut last_updated_seq = if let Some(version_meta) = &frag.last_updated_at_version_meta {
            version_meta.load_sequence().map_err(|e| {
                Error::internal(format!(
                    "Failed to load last_updated_at version sequence: {}",
                    e
                ))
            })?
        } else {
            created_at_seq.clone()
        };

        // Apply deletion mask if present (positions are local offsets)
        if let Some(deletion_file) = &frag.deletion_file {
            let deletions = read_dataset_deletion_file(dataset, frag.id, deletion_file).await?;
            last_updated_seq.mask(deletions.to_sorted_iter())?;
            created_at_seq.mask(deletions.to_sorted_iter())?;
        }

        old_last_updated_sequences.push(last_updated_seq);
        old_created_at_sequences.push(created_at_seq);
    }

    // Ensure row counts match new fragments total
    let old_total: u64 = old_last_updated_sequences.iter().map(|s| s.len()).sum();
    let new_total: u64 = new_fragments
        .iter()
        .map(|f| f.physical_rows.unwrap_or(0) as u64)
        .sum();
    debug_assert_eq!(old_total, new_total);

    // Rechunk version runs aligned to new fragment sizes
    let chunk_sizes: Vec<u64> = new_fragments
        .iter()
        .map(|f| f.physical_rows.unwrap_or(0) as u64)
        .collect();

    let new_last_updated_sequences = lance_table::rowids::version::rechunk_version_sequences(
        old_last_updated_sequences,
        chunk_sizes.clone(),
        false,
    )?;

    let new_created_at_sequences = lance_table::rowids::version::rechunk_version_sequences(
        old_created_at_sequences,
        chunk_sizes,
        false,
    )?;

    // Set both version metadata on new fragments
    for ((fragment, last_updated_seq), created_at_seq) in new_fragments
        .iter_mut()
        .zip(new_last_updated_sequences)
        .zip(new_created_at_sequences)
    {
        fragment.last_updated_at_version_meta = Some(
            lance_table::format::RowDatasetVersionMeta::from_sequence(&last_updated_seq).unwrap(),
        );
        fragment.created_at_version_meta = Some(
            lance_table::format::RowDatasetVersionMeta::from_sequence(&created_at_seq).unwrap(),
        );
    }

    Ok(())
}

fn mask_source_ordered_version_sequence(
    sequence: &mut RowDatasetVersionSequence,
    deletions: &lance_core::utils::deletion::DeletionVector,
) -> Result<()> {
    let bitmap = match deletions {
        lance_core::utils::deletion::DeletionVector::NoDeletions => return Ok(()),
        lance_core::utils::deletion::DeletionVector::Set(offsets) => {
            Cow::Owned(offsets.iter().copied().collect::<RoaringBitmap>())
        }
        lance_core::utils::deletion::DeletionVector::Bitmap(bitmap) => Cow::Borrowed(bitmap),
    };
    let mut ranges = bitmap.iter();
    sequence.mask_ranges(std::iter::from_fn(move || {
        ranges
            .next_range()
            .map(|range| u64::from(*range.start())..u64::from(*range.end()) + 1)
    }))
}

/// Preserve version runs without resolving individual logical rows when the
/// admitted output order is exactly the live physical source order.
async fn source_ordered_v2_3_version_sequences(
    dataset: &Dataset,
    fragments: &[Fragment],
) -> Result<Option<RowVersionSequences>> {
    if fragments.iter().any(|fragment| {
        fragment.created_at_version_meta.is_none() && fragment.native_logical_domain.is_none()
    }) {
        // A relocated fragment should carry exact version metadata. Fall back
        // to bounded logical resolution for malformed or provisional metadata
        // instead of guessing a domain creation version.
        return Ok(None);
    }

    let sequences = futures::stream::iter(fragments)
        .map(|fragment| async move {
            let row_count = u64::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "storage-version-2.3 source fragment {} is missing physical_rows",
                    fragment.id
                ))
            })?)
            .map_err(|_| Error::format_capacity_exceeded("source physical_rows exceed u64"))?;
            let mut created = if let Some(metadata) = &fragment.created_at_version_meta {
                metadata.load_sequence()?
            } else {
                let native = fragment.native_logical_domain.ok_or_else(|| {
                    Error::internal("source-order row-version preflight changed during resolution")
                })?;
                RowDatasetVersionSequence::from_uniform_row_count(
                    row_count,
                    native.creation_version,
                )
            };
            let mut updated = if let Some(metadata) = &fragment.last_updated_at_version_meta {
                metadata.load_sequence()?
            } else {
                created.clone()
            };
            if created.len() != row_count || updated.len() != row_count {
                return Err(Error::invalid_input(format!(
                    "row-version metadata for source fragment {} covers created={} updated={} rows, expected {row_count}",
                    fragment.id,
                    created.len(),
                    updated.len()
                )));
            }

            if let Some(deletion_file) = &fragment.deletion_file {
                let deletions =
                    read_dataset_deletion_file(dataset, fragment.id, deletion_file).await?;
                mask_source_ordered_version_sequence(&mut created, deletions.as_ref())?;
                mask_source_ordered_version_sequence(&mut updated, deletions.as_ref())?;
            }
            Ok::<_, Error>((created, updated))
        })
        .buffered(dataset.object_store.io_parallelism())
        .try_collect::<Vec<_>>()
        .await?;
    let (created_sequences, updated_sequences) = sequences.into_iter().unzip();
    Ok(Some((created_sequences, updated_sequences)))
}

/// Resolve reordered provenance in bounded batches. The logical row sequence
/// remains compressed and the working row-ID allocation has a fixed ceiling.
async fn resolve_logical_row_version_sequences_bounded(
    dataset: &Dataset,
    logical_row_ids: &RowIdSequence,
) -> Result<RowVersionSequences> {
    const RESOLVE_BATCH_SIZE: usize = 65_536;

    let batch_count = usize::try_from(logical_row_ids.len().div_ceil(RESOLVE_BATCH_SIZE as u64))
        .map_err(|_| {
            Error::format_capacity_exceeded("logical row-version batch count exceeds usize")
        })?;
    let mut created_sequences = Vec::with_capacity(batch_count);
    let mut updated_sequences = Vec::with_capacity(batch_count);
    let mut batch = Vec::with_capacity(RESOLVE_BATCH_SIZE);

    for row_id in logical_row_ids.iter() {
        batch.push(row_id);
        if batch.len() == RESOLVE_BATCH_SIZE {
            let (created, updated) = resolve_logical_row_version_sequences(dataset, &batch).await?;
            created_sequences.push(created);
            updated_sequences.push(updated);
            batch.clear();
        }
    }
    if !batch.is_empty() {
        let (created, updated) = resolve_logical_row_version_sequences(dataset, &batch).await?;
        created_sequences.push(created);
        updated_sequences.push(updated);
    }

    Ok((created_sequences, updated_sequences))
}

/// Commit the results of file compaction.
///
/// It is not required that all tasks are passed to this method. If some failed,
/// they can be omitted and the successful tasks can be committed. However, once
/// some of the tasks have been committed, the remainder of the tasks will not
/// be able to be committed and should be considered cancelled.
pub async fn commit_compaction(
    dataset: &mut Dataset,
    completed_tasks: Vec<RewriteResult>,
    remap_options: Arc<dyn IndexRemapperOptions>,
    options: &CompactionOptions,
) -> Result<CompactionMetrics> {
    if completed_tasks.is_empty() {
        return Ok(CompactionMetrics::default());
    }

    let mut skipped_metrics = CompactionMetrics::default();
    let mut completed_tasks = completed_tasks
        .into_iter()
        .filter_map(|task| {
            if task.original_fragments.is_empty() && task.new_fragments.is_empty() {
                skipped_metrics += task.metrics;
                None
            } else {
                Some(task)
            }
        })
        .collect::<Vec<_>>();
    if completed_tasks.is_empty() {
        return Ok(skipped_metrics);
    }

    let uses_v2_3_row_addresses = dataset.manifest.uses_stable_logical_row_addresses();
    let has_stable_row_identity = dataset.manifest.has_stable_row_identity();
    let needs_remapping = !has_stable_row_identity && !options.defer_index_remap;
    let defers_remapping = !has_stable_row_identity && options.defer_index_remap;

    // Determine the earliest version at which compaction tasks were planned/executed.
    //
    // In distributed mode (e.g. Spark) the caller opens *two separate* Dataset
    // handles: one for `plan_compaction` (at version V) and a fresh one for
    // `commit_compaction` (at the latest version V+N).  Using `dataset.manifest.version`
    // (= V+N) as the transaction's `read_version` would cause the conflict checker to
    // scan only versions after V+N — finding nothing — and therefore silently skip any
    // concurrent DELETE/UPDATE that landed between V and V+N, resurrecting deleted rows.
    //
    // By anchoring `read_version` to the minimum version carried in the RewriteResults
    // we ensure the conflict checker covers the full range [V, V+N] and will reject the
    // commit with a retryable conflict error if a concurrent write touched the same
    // fragments.
    let tasks_read_version = completed_tasks
        .iter()
        .map(|t| t.read_version)
        .min()
        .unwrap_or(dataset.manifest.version);

    // Single reserve_fragment_ids for all address-style tasks
    let has_address_style = completed_tasks
        .iter()
        .any(|t| t.row_addrs.is_some() || t.ordered_row_addrs.is_some());
    if has_address_style {
        let frags: Vec<&mut Fragment> = completed_tasks
            .iter_mut()
            .filter(|t| t.row_addrs.is_some() || t.ordered_row_addrs.is_some())
            .flat_map(|t| t.new_fragments.iter_mut())
            .collect();
        reserve_fragment_ids(dataset, frags.into_iter()).await?;
    }

    let mut rewrite_groups = Vec::with_capacity(completed_tasks.len());
    let mut metrics = skipped_metrics;

    let mut remap_group_inputs: Vec<GroupInput> = Vec::new();
    let mut direct_row_id_map: HashMap<u64, Option<u64>> = HashMap::default();
    let mut frag_reuse_groups: Vec<FragReuseGroup> = Vec::new();
    let mut new_fragment_bitmap: RoaringBitmap = RoaringBitmap::new();
    let mut next_new_fragment_ordinal = 0_u32;
    let mut source_domains = BTreeMap::new();
    let mut row_address_layout_delta = if uses_v2_3_row_addresses {
        let layout = dataset
            .manifest
            .row_address_layout
            .as_ref()
            .ok_or_else(|| {
                Error::internal("storage-version-2.3 manifest is missing RowAddressLayout")
            })?;
        Some(RowAddressLayoutDelta {
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        })
    } else {
        None
    };

    for task in completed_tasks {
        metrics += task.metrics;
        if let Some(delta) = row_address_layout_delta.as_mut() {
            let logical_row_ids = task.logical_row_ids.as_deref().ok_or_else(|| {
                Error::invalid_input(
                    "storage-version-2.3 compaction result is missing logical output provenance",
                )
            })?;
            add_rewrite_provenance(
                dataset,
                &task.new_fragments,
                logical_row_ids,
                task.retired_logical_row_ids.as_deref(),
                &mut next_new_fragment_ordinal,
                &mut source_domains,
                delta,
            )?;
        }
        let rewrite_group = RewriteGroup {
            old_fragments: task.original_fragments.clone(),
            new_fragments: task.new_fragments.clone(),
        };

        if needs_remapping {
            if let Some(ordered_row_addrs) = task.ordered_row_addrs {
                if !matches!(options.index_remap_mode, IndexRemapMode::Direct) {
                    return Err(Error::invalid_input(
                        "ordered rewrites require direct index remapping",
                    ));
                }
                let ordered_row_addrs = read_row_ids(&ordered_row_addrs)?;
                direct_row_id_map.extend(remapping::transpose_ordered_row_addrs(
                    &ordered_row_addrs,
                    &task.original_fragments,
                    &task.new_fragments,
                )?);
            } else if let Some(row_addrs_bytes) = task.row_addrs {
                let row_addrs =
                    RoaringTreemap::deserialize_from(&mut Cursor::new(&row_addrs_bytes))?;
                match options.index_remap_mode {
                    IndexRemapMode::Direct => {
                        let transposed = remapping::transpose_row_addrs(
                            row_addrs,
                            &task.original_fragments,
                            &task.new_fragments,
                        );
                        direct_row_id_map.extend(transposed);
                    }
                    IndexRemapMode::Compact => {
                        let new_frags = task
                            .new_fragments
                            .iter()
                            .map(|f| {
                                let physical_rows = f.physical_rows.ok_or_else(|| {
                                    Error::invalid_input(format!(
                                        "compacted fragment {} is missing physical_rows",
                                        f.id
                                    ))
                                })?;
                                Ok((f.id as u32, physical_rows as u32))
                            })
                            .collect::<Result<Vec<_>>>()?;

                        remap_group_inputs.push(GroupInput {
                            rewritten_old_row_addrs: row_addrs,
                            old_frag_ids: task
                                .original_fragments
                                .iter()
                                .map(|f| f.id as u32)
                                .collect(),
                            new_frags,
                        });
                    }
                }
            }
        } else if defers_remapping {
            if task.ordered_row_addrs.is_some() {
                return Err(Error::invalid_input(
                    "ordered rewrites cannot defer index remapping",
                ));
            }
            let changed_row_addrs = task.row_addrs.ok_or_else(|| {
                Error::internal(
                    "defer_index_remap requires row_addrs but none were provided".to_string(),
                )
            })?;
            frag_reuse_groups.push(FragReuseGroup {
                changed_row_addrs,
                old_frags: task.original_fragments.iter().map(|f| f.into()).collect(),
                new_frags: task.new_fragments.iter().map(|f| f.into()).collect(),
            });

            task.new_fragments.iter().for_each(|frag| {
                new_fragment_bitmap.insert(frag.id as u32);
            });
        }
        rewrite_groups.push(rewrite_group);
    }
    if let Some(delta) = row_address_layout_delta.as_mut() {
        delta.source_domains = source_domains.into_values().collect();
    }

    let rewritten_indices = if needs_remapping {
        let index_remapper = remap_options.create_remapper(dataset)?;
        let affected_ids = rewrite_groups
            .iter()
            .flat_map(|group| group.old_fragments.iter().map(|frag| frag.id))
            .collect::<Vec<_>>();

        let remap = match options.index_remap_mode {
            IndexRemapMode::Direct => RowAddrRemap::direct(direct_row_id_map),
            IndexRemapMode::Compact => RowAddrRemap::compact(remap_group_inputs)?,
        };
        let remapped_indices = index_remapper.remap_indices(remap, &affected_ids).await?;
        remapped_indices
            .into_iter()
            .map(|rewritten| RewrittenIndex {
                old_id: rewritten.old_id,
                new_id: rewritten.new_id,
                new_index_details: rewritten.index_details,
                new_index_version: rewritten.index_version,
                new_index_files: rewritten.files,
            })
            .collect()
    } else if !options.defer_index_remap && !has_address_style && !uses_v2_3_row_addresses {
        // We need to reserve fragment ids here so that the fragment bitmap
        // can be updated for each index. Only needed for stable row IDs
        // since address-style IDs were already reserved above.
        let new_fragments = rewrite_groups
            .iter_mut()
            .flat_map(|group| group.new_fragments.iter_mut())
            .collect::<Vec<_>>();
        reserve_fragment_ids(dataset, new_fragments.into_iter()).await?;
        Vec::new()
    } else {
        Vec::new()
    };

    let frag_reuse_index = if defers_remapping {
        Some(build_new_frag_reuse_index(dataset, frag_reuse_groups, new_fragment_bitmap).await?)
    } else {
        None
    };

    // Collect new fragment paths before moving rewrite_groups into the transaction,
    // so we can clean them up if the commit fails.
    let all_new_fragments: Vec<Fragment> = rewrite_groups
        .iter()
        .flat_map(|g| g.new_fragments.iter().cloned())
        .collect();

    let transaction = TransactionBuilder::new(
        // Use the version at which the compaction tasks were *planned*, not the
        // version of the dataset handle passed to this function.  In distributed
        // mode the caller may open a fresh dataset at a later version (V+N), but
        // the tasks were executed against an older snapshot (V).  Anchoring the
        // transaction to V ensures the OCC conflict checker scans all writes that
        // landed between V and the commit point, detecting concurrent DELETE
        // transactions that would otherwise cause deleted rows to reappear.
        tasks_read_version,
        Operation::Rewrite {
            groups: rewrite_groups,
            rewritten_indices,
            frag_reuse_index,
        },
    )
    .row_address_layout_delta(row_address_layout_delta)
    .transaction_properties(options.transaction_properties.clone())
    .build();

    if let Err(e) = dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
    {
        // Retryable conflicts prove the files were not committed. Other
        // failures can be ambiguous, so leave immutable objects available to
        // version-aware GC in case the manifest PUT succeeded.
        if matches!(e, Error::RetryableCommitConflict { .. }) {
            cleanup_data_fragments(
                &dataset.object_store,
                &dataset.base,
                None,
                &all_new_fragments,
            )
            .await;
        }
        return Err(e);
    }

    Ok(metrics)
}

#[cfg(test)]
mod tests {

    mod binary_copy;
    use self::remapping::RemappedIndex;
    use super::*;
    use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
    use crate::dataset::optimize::remapping::{transpose_row_addrs, transpose_row_ids_from_digest};
    use crate::dataset::{UpdateBuilder, WriteDestination};
    use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
    use crate::index::vector::{StageParams, VectorIndexParams};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type, UInt64Type};
    use arrow_array::{
        ArrayRef, Float32Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
        PrimitiveArray, RecordBatch, RecordBatchIterator,
    };
    use arrow_schema::{DataType, Field, Schema};
    use arrow_select::concat::concat_batches;
    use async_trait::async_trait;
    use lance_arrow::BLOB_META_KEY;
    use lance_core::Error;
    use lance_core::ROW_ID;
    use lance_core::utils::address::{LogicalRowAddress, RowAddress};
    use lance_core::utils::deletion::DeletionVector;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::Dimension;
    use lance_file::version::LanceFileVersion;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::scalar::{
        BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
    };
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::{Index, IndexType};
    use lance_io::utils::tracking_store::{IOTracker, IoOperation};
    use lance_linalg::distance::{DistanceType, MetricType};
    use lance_table::io::deletion::write_deletion_file;
    use lance_table::io::manifest::read_manifest_indexes;
    use lance_table::rowids::RowIdSequence;
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
    use rstest::rstest;
    use std::collections::HashSet;
    use std::io::Cursor;
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn test_candidate_bin() {
        let empty_bin = CandidateBin {
            fragments: vec![],
            pos_range: 0..0,
            candidacy: vec![],
            row_counts: vec![],
            indices: vec![],
        };
        assert!(empty_bin.is_noop());

        let fragment = Fragment {
            id: 0,
            files: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(0),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            native_logical_domain: None,
        };
        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactWithNeighbors],
            row_counts: vec![100],
            indices: vec![],
        };
        assert!(single_bin.is_noop());

        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactItself],
            row_counts: vec![100],
            indices: vec![],
        };
        // Not a no-op because it's CompactItself
        assert!(!single_bin.is_noop());

        let big_bin = CandidateBin {
            fragments: std::iter::repeat_n(fragment, 8).collect(),
            pos_range: 0..8,
            candidacy: std::iter::repeat_n(CompactionCandidacy::CompactItself, 8).collect(),
            row_counts: vec![100, 400, 200, 200, 400, 300, 300, 100],
            indices: vec![],
            // Will group into: [[100, 400], [200, 200, 400], [300, 300, 100]]
            // with size = 500
        };
        assert!(!big_bin.is_noop());
        let split = big_bin.split_for_size(500);
        assert_eq!(split.len(), 3);
        assert_eq!(split[0].pos_range, 0..2);
        assert_eq!(split[1].pos_range, 2..5);
        assert_eq!(split[2].pos_range, 5..8);
    }

    #[test]
    fn test_v2_3_default_compaction_rejects_cross_domain_reorder() {
        let row_ids = [
            LogicalRowAddress::try_new_from_parts(1, 0).unwrap().raw(),
            LogicalRowAddress::try_new_from_parts(0, 0).unwrap().raw(),
        ];
        let error =
            validate_default_compaction_logical_order(&RowIdSequence::from(row_ids.as_slice()))
                .unwrap_err();
        assert!(matches!(&error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("explicit Recluster"));
    }

    #[test]
    fn test_v2_3_clustered_version_mask_is_range_bounded() {
        let mut deleted = RoaringBitmap::new();
        deleted.insert_range(25_000_000..75_000_000);
        let deletions = DeletionVector::Bitmap(deleted);
        let mut sequence = RowDatasetVersionSequence::from_uniform_row_count(100_000_000, 7);

        mask_source_ordered_version_sequence(&mut sequence, &deletions).unwrap();

        assert_eq!(sequence.len(), 50_000_000);
        assert_eq!(sequence.runs.len(), 1);
        assert_eq!(sequence.runs[0].span, U64Segment::Range(0..50_000_000));
        assert_eq!(sequence.runs[0].version, 7);
    }

    fn sample_data() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
        )
        .unwrap()
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct MockIndexRemapperExpectation {
        expected: HashMap<u64, Option<u64>>,
        answer: Vec<RemappedIndex>,
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct MockIndexRemapper {
        expectations: Vec<MockIndexRemapperExpectation>,
    }

    impl MockIndexRemapper {
        fn stringify_map(map: &HashMap<u64, Option<u64>>) -> String {
            let mut sorted_keys = map.keys().collect::<Vec<_>>();
            sorted_keys.sort();
            let mut first_keys = sorted_keys
                .into_iter()
                .take(10)
                .map(|key| {
                    format!(
                        "{}:{:?}",
                        RowAddress::from(*key),
                        map[key].map(RowAddress::from)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            if map.len() > 10 {
                first_keys.push_str(", ...");
            }
            let mut result_str = format!("(len={})", map.len());
            result_str.push_str(&first_keys);
            result_str
        }

        fn in_any_order(expectations: &[Self]) -> Self {
            let expectations = expectations
                .iter()
                .flat_map(|item| item.expectations.clone())
                .collect::<Vec<_>>();
            Self { expectations }
        }
    }

    #[async_trait]
    impl IndexRemapper for MockIndexRemapper {
        async fn remap_indices(
            &self,
            index_map: RowAddrRemap,
            _: &[u64],
        ) -> Result<Vec<RemappedIndex>> {
            for expectation in &self.expectations {
                let matches = match &index_map {
                    RowAddrRemap::Direct(map) => map == &expectation.expected,
                    RowAddrRemap::Compact(_) => {
                        let expected_frags: RoaringBitmap = expectation
                            .expected
                            .keys()
                            .map(|addr| (addr >> 32) as u32)
                            .collect();
                        index_map.affected_fragments() == expected_frags
                            && expectation
                                .expected
                                .iter()
                                .all(|(k, v)| index_map.get(*k) == Some(*v))
                    }
                };
                if matches {
                    return Ok(expectation.answer.clone());
                }
            }
            panic!(
                "Unexpected index map; expected one of:\n  {}",
                self.expectations
                    .iter()
                    .map(|expectation| Self::stringify_map(&expectation.expected))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            );
        }
    }

    impl IndexRemapperOptions for MockIndexRemapper {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(self.clone()))
        }
    }

    #[rstest]
    #[tokio::test]
    async fn ordered_rewrite_preserves_v2_2_identity_and_index_contract(
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let test_dir = TempStrDir::default();
        let values = Int32Array::from(vec![2, 3, 0, 1, 6, 7, 4, 5]);
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                enable_stable_row_ids: stable_row_ids,
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("i_idx".to_owned()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        let index_before = dataset.load_indices().await.unwrap()[0].uuid;

        async fn value_row_ids(dataset: &Dataset) -> HashMap<i32, u64> {
            let mut scanner = dataset.scan();
            scanner.project(&["i", ROW_ID]).unwrap();
            let batch = scanner.try_into_batch().await.unwrap();
            let values = batch["i"].as_primitive::<Int32Type>();
            let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            values
                .values()
                .iter()
                .copied()
                .zip(row_ids.values().iter().copied())
                .collect()
        }
        let row_ids_before = value_row_ids(&dataset).await;

        let metrics = rewrite_files_in_order(
            &mut dataset,
            vec![ColumnOrdering::asc_nulls_last("i".to_owned())],
            CompactionOptions {
                target_rows_per_fragment: 8,
                max_rows_per_group: 8,
                index_remap_mode: IndexRemapMode::Direct,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        let mut scanner = dataset.scan();
        scanner.scan_in_order(true).project(&["i"]).unwrap();
        let ordered = scanner.try_into_batch().await.unwrap();
        assert_eq!(
            ordered["i"].as_primitive::<Int32Type>().values(),
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
        let row_ids_after = value_row_ids(&dataset).await;
        if stable_row_ids {
            assert_eq!(row_ids_after, row_ids_before);
            assert_eq!(dataset.load_indices().await.unwrap()[0].uuid, index_before);
        } else {
            assert_ne!(row_ids_after, row_ids_before);
            assert_ne!(dataset.load_indices().await.unwrap()[0].uuid, index_before);
        }

        let mut indexed = dataset.scan();
        indexed.filter("i = 4").unwrap().project(&["i"]).unwrap();
        let indexed = indexed.try_into_batch().await.unwrap();
        assert_eq!(indexed.num_rows(), 1);
        assert_eq!(indexed["i"].as_primitive::<Int32Type>().value(0), 4);
    }

    #[tokio::test]
    async fn ordered_rewrite_rejects_partial_legacy_index_before_data_writes() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let initial = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![3, 2, 1, 0]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                enable_stable_row_ids: false,
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("i_idx".to_owned()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        let appended = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![7, 6, 5, 4]))],
        )
        .unwrap();
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(appended)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                enable_stable_row_ids: false,
                max_rows_per_file: 4,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 2);

        let tracker = Arc::new(IOTracker::default());
        dataset = dataset.with_object_store_wrappers([
            tracker.clone() as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        let version = dataset.version().version;
        let data_files = count_data_files_in(test_dir.as_str());
        tracker.incremental_stats();

        let error = rewrite_files_in_order(
            &mut dataset,
            vec![ColumnOrdering::asc_nulls_last("i".to_owned())],
            CompactionOptions {
                target_rows_per_fragment: 8,
                max_rows_per_group: 8,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap_err();

        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("full current fragment set"));
        assert_eq!(dataset.version().version, version);
        assert_eq!(count_data_files_in(test_dir.as_str()), data_files);
        let data_writes = tracker
            .incremental_stats()
            .requests
            .into_iter()
            .filter(|request| {
                request.operation == IoOperation::Put
                    && request.path.to_string().starts_with("data/")
            })
            .collect::<Vec<_>>();
        assert!(
            data_writes.is_empty(),
            "partial index coverage must fail before data PUTs: {data_writes:?}"
        );
    }

    #[tokio::test]
    async fn ordered_rewrite_removes_fully_deleted_legacy_fragment() {
        let test_dir = TempStrDir::default();
        let values = Int32Array::from(vec![7, 6, 5, 4, 3, 2, 1, 0]);
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                enable_stable_row_ids: false,
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 2);

        let deletion_vector = DeletionVector::Set(HashSet::from_iter(0..4));
        // Current delete writers remove fully deleted fragments immediately,
        // but older manifests can retain a full deletion vector. Recreate that
        // legacy snapshot state directly so migration filtering is exercised.
        let mut manifest = dataset.manifest.as_ref().clone();
        let mut fragments = manifest.fragments.as_ref().clone();
        for fragment in &mut fragments {
            fragment.deletion_file = write_deletion_file(
                &dataset.base,
                fragment.id,
                dataset.version().version,
                &deletion_vector,
                dataset.object_store.as_ref(),
            )
            .await
            .unwrap();
        }
        manifest.fragments = Arc::new(fragments);
        dataset.manifest = Arc::new(manifest);
        assert_eq!(dataset.manifest.fragments.len(), 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

        let metrics = rewrite_files_in_order(
            &mut dataset,
            vec![ColumnOrdering::asc_nulls_last("i".to_owned())],
            CompactionOptions {
                target_rows_per_fragment: 8,
                max_rows_per_group: 8,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(metrics.groups_planned, 1);
        assert_eq!(metrics.groups_admitted, 1);
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 0);
        assert_eq!(metrics.files_removed, 4);
        assert_eq!(metrics.files_added, 0);
        assert!(dataset.manifest.fragments.is_empty());
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_empty(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Compact an empty table
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), Arc::new(schema));
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        assert_eq!(metrics, CompactionMetrics::default());
        assert_eq!(dataset.manifest.version, 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_all_good(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Compact a table with nothing to do
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        // Just one file
        let write_params = WriteParams {
            max_rows_per_file: 10_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // There's only one file, so we can't compact any more if we wanted to.
        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Now split across multiple files
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 3_000,
            max_rows_per_group: 1_000,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);
    }

    async fn scan_v2_3_rows(dataset: &Dataset) -> Vec<(i64, u64)> {
        let mut scanner = dataset.scan();
        scanner.with_row_id();
        let batch = scanner.try_into_batch().await.unwrap();
        let values = batch
            .column_by_name("a")
            .unwrap()
            .as_primitive::<Int64Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>();
        (0..batch.num_rows())
            .map(|index| (values.value(index), row_ids.value(index)))
            .collect()
    }

    #[tokio::test]
    async fn test_v2_3_compaction_preserves_logical_ids_as_packed_run() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 5);
        let before = scan_v2_3_rows(&dataset).await;

        let options = CompactionOptions {
            target_rows_per_fragment: 20,
            max_rows_per_group: 20,
            compaction_mode: Some(CompactionMode::TryBinaryCopy),
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.num_tasks(), 1);
        assert_eq!(
            plan.tasks[0].v2_3_plan.as_ref().unwrap().output_row_counts,
            vec![20]
        );
        assert!(matches!(
            plan.tasks[0]
                .v2_3_plan
                .as_ref()
                .map(|plan| &plan.rewrite_strategy),
            Some(V2_3CompactionRewriteStrategy::BinaryCopy(_))
        ));

        let metrics = compact_files(
            &mut dataset,
            options,
            Some(Arc::new(MockIndexRemapper::default())),
        )
        .await
        .unwrap();

        assert_eq!(metrics.groups_planned, 1);
        assert_eq!(metrics.groups_admitted, 1);
        assert_eq!(metrics.groups_not_admitted, 0);
        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert_eq!(scan_v2_3_rows(&dataset).await, before);
        let fragment = &dataset.manifest.fragments[0];
        assert!(fragment.native_logical_domain.is_none());
        assert!(fragment.row_id_meta.is_none());
        let layout = dataset.manifest.row_address_layout.as_ref().unwrap();
        assert_eq!(layout.placements.len(), 1);
        assert!(matches!(
            layout.placements[0],
            lance_table::format::RowAddressPlacement::PackedRun(_)
        ));
    }

    #[tokio::test]
    async fn test_v2_3_source_order_compaction_preserves_version_runs() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let initial = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..4))],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 2,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let appended = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(4..8))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(appended)], schema),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 2,
                data_storage_version: Some(LanceFileVersion::V2_3),
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 4);
        assert_eq!(
            dataset
                .manifest
                .fragments
                .iter()
                .map(|fragment| fragment.native_logical_domain.unwrap().creation_version)
                .collect::<Vec<_>>(),
            vec![1, 1, 2, 2]
        );

        let options = CompactionOptions {
            target_rows_per_fragment: 8,
            max_rows_per_group: 8,
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let plan = plan.tasks[0].v2_3_plan.as_ref().unwrap();
        assert!(plan.logical_run_ends.is_empty());
        assert!(!plan.requires_full_logical_sort);

        compact_files(&mut dataset, options, None).await.unwrap();

        assert_eq!(dataset.manifest.fragments.len(), 1);
        let fragment = &dataset.manifest.fragments[0];
        let created = fragment
            .created_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        let updated = fragment
            .last_updated_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        assert_eq!(
            created.versions().collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 2, 2, 2, 2]
        );
        assert_eq!(updated, created);
        assert_eq!(created.runs.len(), 2);
    }

    #[tokio::test]
    async fn test_v2_3_force_binary_copy_rejects_incompatible_sources_before_writes() {
        use lance_io::utils::tracking_store::{IOTracker, IoOperation};

        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("a = 1").await.unwrap();
        let tracker = Arc::new(IOTracker::default());
        dataset = dataset.with_object_store_wrappers([
            tracker.clone() as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        let version = dataset.version().version;
        let data_files = count_data_files_in(test_dir.as_str());
        tracker.incremental_stats();

        let error = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                compaction_mode: Some(CompactionMode::ForceBinaryCopy),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap_err();

        assert!(matches!(&error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("ForceBinaryCopy"));
        assert_eq!(dataset.version().version, version);
        assert_eq!(count_data_files_in(test_dir.as_str()), data_files);
        let data_writes = tracker
            .incremental_stats()
            .requests
            .into_iter()
            .filter(|request| {
                request.operation == IoOperation::Put
                    && request.path.to_string().starts_with("data/")
            })
            .collect::<Vec<_>>();
        assert!(
            data_writes.is_empty(),
            "incompatible ForceBinaryCopy must fail before data PUTs: {data_writes:?}"
        );
    }

    #[tokio::test]
    async fn test_v2_3_reencode_freezes_max_bytes_as_row_boundaries() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let options = CompactionOptions {
            target_rows_per_fragment: 20,
            max_bytes_per_file: Some(256),
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let planned_counts = plan.tasks[0]
            .v2_3_plan
            .as_ref()
            .unwrap()
            .output_row_counts
            .clone();
        assert!(planned_counts.len() > 1);

        compact_files(&mut dataset, options, None).await.unwrap();
        let actual_counts = dataset
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(actual_counts, planned_counts);
    }

    #[tokio::test]
    async fn test_v2_3_compaction_combines_group_admission_and_metrics() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..40))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let options = CompactionOptions {
            target_rows_per_fragment: 8,
            max_rows_per_group: 8,
            num_threads: Some(2),
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.num_tasks(), 5);
        assert_eq!(plan.planning_metrics, CompactionMetrics::default());
        assert!(plan.tasks.iter().all(|task| {
            task.v2_3_plan
                .as_ref()
                .is_some_and(|plan| plan.output_row_counts == [8])
        }));

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(metrics.groups_planned, 5);
        assert_eq!(metrics.groups_admitted, 5);
        assert_eq!(metrics.groups_not_admitted, 0);
        assert_eq!(dataset.manifest.fragments.len(), 5);
    }

    #[tokio::test]
    async fn test_v2_3_not_admitted_result_does_not_commit_empty_rewrite() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..4))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let version = dataset.version().version;
        let data_files = count_data_files_in(test_dir.as_str());
        let metrics = CompactionMetrics {
            groups_planned: 1,
            groups_not_admitted: 1,
            ..CompactionMetrics::default()
        };
        let skipped = RewriteResult {
            metrics: metrics.clone(),
            new_fragments: Vec::new(),
            read_version: version,
            original_fragments: Vec::new(),
            row_addrs: None,
            ordered_row_addrs: None,
            logical_row_ids: None,
            retired_logical_row_ids: None,
        };

        let actual = commit_compaction(
            &mut dataset,
            vec![skipped],
            Arc::new(DatasetIndexRemapperOptions::default()),
            &CompactionOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(actual, metrics);
        assert_eq!(dataset.version().version, version);
        assert_eq!(count_data_files_in(test_dir.as_str()), data_files);
    }

    #[tokio::test]
    async fn test_v2_3_update_stripe_n_to_one_merges_logical_runs() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("a = 1")
            .unwrap()
            .set("a", "101")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut dataset = updated.new_dataset.as_ref().clone();
        let mut expected = scan_v2_3_rows(&dataset).await;
        expected.sort_by_key(|(_, row_id)| *row_id);
        let options = CompactionOptions {
            target_rows_per_fragment: 20,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks.len(), 1);
        let v2_3_plan = plan.tasks[0].v2_3_plan.as_ref().unwrap();
        assert_eq!(v2_3_plan.logical_run_ends, vec![5, 6]);
        assert!(!v2_3_plan.requires_full_logical_sort);

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(metrics.groups_planned, 1);
        assert_eq!(metrics.groups_admitted, 1);
        assert_eq!(metrics.groups_not_admitted, 0);
        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert_eq!(scan_v2_3_rows(&dataset).await, expected);

        let row_ids = expected
            .iter()
            .rev()
            .map(|(_, row_id)| *row_id)
            .collect::<Vec<_>>();
        let taken = dataset
            .take_rows(&row_ids, dataset.schema().clone())
            .await
            .unwrap();
        let taken_values = taken["a"].as_primitive::<Int64Type>().values().to_vec();
        assert_eq!(
            taken_values,
            expected
                .iter()
                .rev()
                .map(|(value, _)| *value)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_v2_3_clustered_delete_after_sparse_update_preserves_live_logical_id() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let original_row_id = scan_v2_3_rows(&dataset)
            .await
            .into_iter()
            .find_map(|(value, row_id)| (value == 1).then_some(row_id))
            .unwrap();

        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("a = 1")
            .unwrap()
            .set("a", "101")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut dataset = updated.new_dataset.as_ref().clone();
        dataset.delete("a = 2 OR a = 3").await.unwrap();
        let mut expected = scan_v2_3_rows(&dataset).await;
        expected.sort_by_key(|(_, row_id)| *row_id);
        assert!(expected.contains(&(101, original_row_id)));

        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(metrics.groups_admitted, 1);
        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert_eq!(scan_v2_3_rows(&dataset).await, expected);
        let taken = dataset
            .take_rows(&[original_row_id], dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(taken["a"].as_primitive::<Int64Type>().value(0), 101);
    }

    #[tokio::test]
    async fn test_v2_3_update_compaction_reuses_index_without_index_io() {
        use lance_io::utils::tracking_store::IOTracker;

        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
        ]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..20)),
                Arc::new(Int64Array::from_iter_values(100..120)),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data)], schema),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["key"],
                IndexType::Scalar,
                Some("key_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("a = 1")
            .unwrap()
            .set("a", "101")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let before_indices = updated.new_dataset.load_indices().await.unwrap();
        let tracker = Arc::new(IOTracker::default());
        let mut dataset = updated.new_dataset.with_object_store_wrappers([
            tracker.clone() as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        tracker.incremental_stats();

        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(metrics.groups_admitted, 1);
        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert_eq!(dataset.load_indices().await.unwrap(), before_indices);

        let stats = tracker.incremental_stats();
        let index_requests = stats
            .requests
            .iter()
            .filter(|request| request.path.to_string().contains("_indices"))
            .collect::<Vec<_>>();
        assert!(
            index_requests.is_empty(),
            "physical rewrite must not read or write logical index objects: {index_requests:?}"
        );

        let result = dataset
            .scan()
            .filter("key = 101")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(result["a"].as_primitive::<Int64Type>().value(0), 101);
    }

    #[tokio::test]
    async fn test_v2_3_compaction_not_admitted_writes_no_data_objects() {
        use lance_io::utils::tracking_store::{IOTracker, IoOperation};

        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let mut maintenance = RowAddressMaintenanceOptions::repack();
        maintenance.target_rows_per_fragment = 4;
        maintain_row_addresses(&mut dataset, maintenance)
            .await
            .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 5);

        let tracker = Arc::new(IOTracker::default());
        dataset = dataset.with_object_store_wrappers([
            tracker.clone() as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        let version = dataset.version().version;
        let data_files = count_data_files_in(test_dir.as_str());
        tracker.incremental_stats();

        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(metrics.groups_planned, 1);
        assert_eq!(metrics.groups_admitted, 0);
        assert_eq!(metrics.groups_not_admitted, 1);
        assert_eq!(dataset.version().version, version);
        assert_eq!(count_data_files_in(test_dir.as_str()), data_files);

        let data_writes = tracker
            .incremental_stats()
            .requests
            .into_iter()
            .filter(|request| {
                request.operation == IoOperation::Put
                    && request.path.to_string().starts_with("data/")
            })
            .collect::<Vec<_>>();
        assert!(
            data_writes.is_empty(),
            "not-admitted preflight must precede every data-object write: {data_writes:?}"
        );
    }

    #[tokio::test]
    async fn test_v2_3_group_high_entropy_deletes_are_not_admitted() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20_000))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 2_000,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 10);
        dataset.delete("a % 2 = 0").await.unwrap();

        let plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                target_rows_per_fragment: 20_000,
                max_rows_per_group: 20_000,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(plan.tasks.is_empty());
        assert_eq!(plan.planning_metrics.groups_planned, 1);
        assert_eq!(plan.planning_metrics.groups_admitted, 0);
        assert_eq!(plan.planning_metrics.groups_not_admitted, 1);
    }

    #[tokio::test]
    async fn test_v2_3_compaction_retires_deleted_logical_ids() {
        let test_dir = TempStrDir::default();
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let original = scan_v2_3_rows(&dataset).await;
        let retired = original
            .iter()
            .filter(|(value, _)| *value == 1 || *value == 3)
            .map(|(_, row_id)| *row_id)
            .collect::<Vec<_>>();
        dataset.delete("a = 1 OR a = 3").await.unwrap();
        let before = scan_v2_3_rows(&dataset).await;

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                max_rows_per_group: 20,
                materialize_deletions: true,
                materialize_deletions_threshold: 0.0,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert_eq!(scan_v2_3_rows(&dataset).await, before);
        assert!(dataset.manifest.fragments[0].deletion_file.is_none());
        assert_eq!(
            dataset.resolve_logical_row_ids(&retired).unwrap(),
            vec![None; retired.len()]
        );
        let layout = dataset.manifest.row_address_layout.as_ref().unwrap();
        let retired_addresses = retired
            .iter()
            .copied()
            .map(LogicalRowAddress::try_from)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            dataset
                .row_address_router()
                .unwrap()
                .resolve_many(&retired_addresses)
                .unwrap(),
            vec![lance_table::format::PlacementResolution::NotLive; retired.len()]
        );
        let mut sparse_block_ranges = Vec::new();
        layout
            .visit_retired_ranges(|range| sparse_block_ranges.push(range))
            .unwrap();
        assert!(!sparse_block_ranges.is_empty());
        for address in retired_addresses {
            assert!(sparse_block_ranges.iter().any(|range| {
                range.logical_fragment_id == address.logical_fragment_id()
                    && range.start_slot <= address.immutable_slot()
                    && address.immutable_slot() < range.end_slot
            }));
        }
        assert_eq!(layout.placements.len(), 1);
        assert!(matches!(
            layout.placements[0],
            lance_table::format::RowAddressPlacement::SparseSelection(_)
                | lance_table::format::RowAddressPlacement::Selected(_)
        ));
    }

    #[tokio::test]
    async fn test_compact_blob_columns() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("blob", DataType::LargeBinary, false)
                .with_metadata([(BLOB_META_KEY.to_string(), "true".to_string())].into()),
        ]));
        let expected_payload: Vec<Vec<u8>> =
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9, 10], vec![11]];
        let id_column: ArrayRef = Arc::new(Int32Array::from_iter_values(
            0..expected_payload.len() as i32,
        ));
        let blob_array: ArrayRef = Arc::new(LargeBinaryArray::from_iter(
            expected_payload.iter().map(|value| Some(value.as_slice())),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![id_column, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();
        assert!(dataset.get_fragments().len() > 1);

        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);

        let dataset = Arc::new(dataset);
        let row_indices: Vec<u64> = (0..expected_payload.len() as u64).collect();
        let blobs = dataset
            .take_blobs_by_indices(&row_indices, "blob")
            .await
            .unwrap();
        assert_eq!(blobs.len(), expected_payload.len());
        for (blob, expected) in blobs.iter().zip(expected_payload.iter()) {
            let bytes = blob.read().await.unwrap();
            assert_eq!(bytes.as_ref(), expected.as_slice());
        }
    }

    fn row_addrs(frag_idx: u32, offsets: Range<u32>) -> Range<u64> {
        let start = RowAddress::new_from_parts(frag_idx, offsets.start);
        let end = RowAddress::new_from_parts(frag_idx, offsets.end);
        start.into()..end.into()
    }

    // The outer list has one item per new fragment
    // The inner list has ranges of old row ids that map to the new fragment, in order
    fn expect_remap(
        ranges: &[Vec<(Range<u64>, bool)>],
        starting_new_frag_idx: u32,
    ) -> MockIndexRemapper {
        let mut expected_remap: HashMap<u64, Option<u64>> = HashMap::default();
        expected_remap.reserve(ranges.iter().map(|r| r.len()).sum());
        for (new_frag_offset, new_frag_ranges) in ranges.iter().enumerate() {
            let new_frag_idx = starting_new_frag_idx + new_frag_offset as u32;
            let mut row_offset = 0;
            for (old_id_range, is_found) in new_frag_ranges.iter() {
                for old_id in old_id_range.clone() {
                    if *is_found {
                        let new_id = RowAddress::new_from_parts(new_frag_idx, row_offset);
                        expected_remap.insert(old_id, Some(new_id.into()));
                        row_offset += 1;
                    } else {
                        expected_remap.insert(old_id, None);
                    }
                }
            }
        }
        MockIndexRemapper {
            expectations: vec![MockIndexRemapperExpectation {
                expected: expected_remap,
                answer: vec![],
            }],
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_many(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 3 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1200))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 400,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Append 2 large fragments (1k rows)
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(1200, 2000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete 1 row from first large fragment
        dataset.delete("a = 1300").await.unwrap();

        // Delete 20% of rows from second large fragment
        dataset.delete("a >= 2400 AND a < 2600").await.unwrap();

        // Append 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(3200, 600))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 300,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let first_new_frag_idx = 7;
        // Predicting the remap is difficult.  One task will remap to fragments 7/8 and the other
        // will remap to fragments 9/10 but we don't know which is which and so we just allow ourselves
        // to expect both possibilities.
        let remap_a = expect_remap(
            &[
                vec![
                    // 3 small fragments are rewritten to frags 7 & 8
                    (row_addrs(0, 0..400), true),
                    (row_addrs(1, 0..400), true),
                    (row_addrs(2, 0..200), true),
                ],
                vec![(row_addrs(2, 200..400), true)],
                // frag 3 is skipped since it does not have enough missing data
                // Frags 4, 5, and 6 are rewritten to frags 9 & 10
                vec![
                    // Only 800 of the 1000 rows taken from frag 4
                    (row_addrs(4, 0..200), true),
                    (row_addrs(4, 200..400), false),
                    (row_addrs(4, 400..1000), true),
                    // frags 5 compacted with frag 4
                    (row_addrs(5, 0..200), true),
                ],
                vec![(row_addrs(5, 200..300), true), (row_addrs(6, 0..300), true)],
            ],
            first_new_frag_idx,
        );
        let remap_b = expect_remap(
            &[
                // Frags 4, 5, and 6 are rewritten to frags 7 & 8
                vec![
                    (row_addrs(4, 0..200), true),
                    (row_addrs(4, 200..400), false),
                    (row_addrs(4, 400..1000), true),
                    (row_addrs(5, 0..200), true),
                ],
                vec![(row_addrs(5, 200..300), true), (row_addrs(6, 0..300), true)],
                // 3 small fragments rewritten to frags 9 & 10
                vec![
                    (row_addrs(0, 0..400), true),
                    (row_addrs(1, 0..400), true),
                    (row_addrs(2, 0..200), true),
                ],
                vec![(row_addrs(2, 200..400), true)],
            ],
            first_new_frag_idx,
        );

        // Create compaction plan
        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 2);
        assert_eq!(plan.tasks()[0].fragments.len(), 3);
        assert_eq!(plan.tasks()[1].fragments.len(), 3);

        assert_eq!(
            plan.tasks()[0]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            plan.tasks()[1]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![4, 5, 6]
        );

        let mock_remapper = MockIndexRemapper::in_any_order(&[remap_a, remap_b]);

        // Run compaction
        let metrics = compact_files(&mut dataset, options, Some(Arc::new(mock_remapper)))
            .await
            .unwrap();

        // Assert on metrics
        assert_eq!(metrics.fragments_removed, 6);
        assert_eq!(metrics.fragments_added, 4);
        assert_eq!(metrics.files_removed, 7); // 6 data files + 1 deletion file
        assert_eq!(metrics.files_added, 4);

        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id())
            .collect::<Vec<_>>();
        assert_eq!(fragment_ids, vec![3, 7, 8, 9, 10]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_data_files(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 5_000,
            max_rows_per_group: 1_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Add a column
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]);

        let data = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..10_000)),
                Arc::new(Float32Array::from_iter_values(
                    (0..10_000).map(|x| x as f32 * std::f32::consts::PI),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());

        dataset.merge(reader, "a", "a").await.unwrap();

        let expected_remap = expect_remap(
            &[vec![
                // 3 small fragments are rewritten entirely
                (row_addrs(0, 0..5000), true),
                (row_addrs(1, 0..5000), true),
            ]],
            2,
        );

        let plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(plan.tasks().len(), 1);
        assert_eq!(plan.tasks()[0].fragments.len(), 2);

        let metrics = compact_files(&mut dataset, plan.options, Some(Arc::new(expected_remap)))
            .await
            .unwrap();

        assert_eq!(metrics.files_removed, 4); // 2 fragments with 2 data files
        assert_eq!(metrics.files_added, 1); // 1 fragment with 1 data file
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        // Assert order unchanged and data is all there.
        let scanner = dataset.scan();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(scanned_data, data);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_with_io_buffer_size(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Compaction should succeed and produce correct results when an
        // explicit io_buffer_size is provided via CompactionOptions.
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 2 small fragments so there is something to compact.
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 5_000,
            max_rows_per_group: 1_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let options = CompactionOptions {
            // A generous buffer so the read does not deadlock on large batches.
            io_buffer_size: Some(256 * 1024 * 1024),
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 1);

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        // All rows are preserved after compaction.
        let scanner = dataset.scan();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();
        assert_eq!(scanned_data.num_rows(), data.num_rows());
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_deletions(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // For files that have few rows, we don't want to compact just 1 since
        // that won't do anything. But if there are deletions to materialize,
        // we want to do groups of 1. This test checks that.
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 1 fragment
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset.delete("a <= 500").await.unwrap();

        // Threshold must be satisfied
        let mut options = CompactionOptions {
            materialize_deletions_threshold: 0.8,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Ignore deletions if materialize_deletions is false
        options.materialize_deletions_threshold = 0.1;
        options.materialize_deletions = false;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Materialize deletions if threshold is met
        options.materialize_deletions = true;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 1);

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(metrics.fragments_removed, 1);
        assert_eq!(metrics.files_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_none());
    }

    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    struct IgnoreRemap {}

    #[async_trait]
    impl IndexRemapper for IgnoreRemap {
        async fn remap_indices(&self, _: RowAddrRemap, _: &[u64]) -> Result<Vec<RemappedIndex>> {
            Ok(Vec::new())
        }
    }

    impl IndexRemapperOptions for IgnoreRemap {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(Self {}))
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_compact_distributed(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        // Can run the tasks independently
        // Can provide subset of tasks to commit_compaction
        // Once committed, can't commit remaining tasks
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Write dataset as 9 1k row fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Plan compaction with 3 tasks
        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 3);

        let dataset_ref = &dataset;
        let mut results = futures::stream::iter(plan.compaction_tasks())
            .then(|task| async move { task.execute(dataset_ref).await.unwrap() })
            .collect::<Vec<_>>()
            .await;

        assert_eq!(results.len(), 3);

        assert_eq!(
            results[0]
                .original_fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(results[0].metrics.files_removed, 3);
        assert_eq!(results[0].metrics.files_added, 1);

        // Just commit the last task
        commit_compaction(
            &mut dataset,
            vec![results.pop().unwrap()],
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await
        .unwrap();

        // 1 commit for reserve fragments and 1 for final commit, both
        // from the call to commit_compaction
        assert_eq!(dataset.manifest.version, 3);

        // Can commit the remaining tasks
        commit_compaction(
            &mut dataset,
            results,
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await
        .unwrap();
        // 1 commit for reserve fragments and 1 for final commit, both
        // from the call to commit_compaction
        assert_eq!(dataset.manifest.version, 5);

        assert_eq!(dataset.manifest.uses_stable_row_ids(), use_stable_row_id,);
    }

    #[tokio::test]
    async fn test_stable_row_indices() {
        // Validate behavior of indices after compaction with stable row ids.
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(16).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));
        let mut dataset = Dataset::write(
            data_gen.batch(500),
            "memory://test/table",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 100, // 5 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Delete first 110 rows so rowids != final rowaddrs
        // First 100 rows deletes first file. Next 10 deletes part of second
        // file, so we will trigger the with deletions code path.
        dataset.delete("i < 110").await.unwrap();

        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("scalar".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vector".into()),
                &params,
                false,
            )
            .await
            .unwrap();

        async fn index_set(dataset: &Dataset) -> HashSet<Uuid> {
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .map(|index| index.uuid)
                .collect()
        }
        let indices = index_set(&dataset).await;

        async fn vector_query(dataset: &Dataset) -> RecordBatch {
            let mut scanner = dataset.scan();

            let query = Float32Array::from(vec![0.0f32; 16]);
            scanner
                .nearest("vec", &query, 10)
                .unwrap()
                .project(&["i"])
                .unwrap();

            scanner.try_into_batch().await.unwrap()
        }

        async fn scalar_query(dataset: &Dataset) -> RecordBatch {
            let mut scanner = dataset.scan();

            scanner.filter("i = 100").unwrap().project(&["i"]).unwrap();

            scanner.try_into_batch().await.unwrap()
        }

        let before_vec_result = vector_query(&dataset).await;
        let before_scalar_result = scalar_query(&dataset).await;

        let options = CompactionOptions {
            target_rows_per_fragment: 180,
            ..Default::default()
        };
        let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

        // The indices should be unchanged after compaction, since we are using
        // stable row ids.
        let current_indices = index_set(&dataset).await;
        assert_eq!(indices, current_indices);

        let after_vec_result = vector_query(&dataset).await;
        assert_eq!(before_vec_result, after_vec_result);

        let after_scalar_result = scalar_query(&dataset).await;
        assert_eq!(before_scalar_result, after_scalar_result);
    }

    // Regression test for https://github.com/lancedb/lance/issues/6161
    // When FragReuseIndexDetails exceeds 204800 bytes it is written to an external
    // file. Previously the file was silently dropped (temp file deleted) because
    // tokio::io::AsyncWriteExt::shutdown was called instead of
    // lance_io::traits::Writer::shutdown, which persists the temp file.
    #[tokio::test]
    async fn test_defer_index_remap_large_external_file() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create ~150 fragments × 1000 rows to produce a FragReuseIndexDetails
        // that exceeds the 204800-byte inline threshold (~302 KB serialized).
        let num_fragments = 150usize;
        let rows_per_fragment = 1000usize;
        let total_rows = num_fragments * rows_per_fragment;

        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));

        let mut dataset = Dataset::write(
            RecordBatchIterator::new(
                vec![Ok(RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(0..total_rows as i32)) as ArrayRef],
                )
                .unwrap())],
                schema.clone(),
            ),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: rows_per_fragment,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), num_fragments);

        // Delete a few rows from each fragment so compaction has something to do.
        dataset.delete("i % 1000 = 0").await.unwrap();

        compact_files(
            &mut dataset,
            CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Loading the FragReuseIndex details must succeed even when the details
        // were written to an external file.
        let frag_reuse_meta = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("fragment reuse index must exist after compaction");

        load_frag_reuse_index_details(&dataset, &frag_reuse_meta)
            .await
            .expect("loading large frag reuse index details must not fail");
    }

    #[tokio::test]
    async fn test_defer_index_remap_rejected_with_stable_row_ids() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9000))], data.schema());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 1000, // 9 fragments
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert!(dataset.manifest.uses_stable_row_ids());

        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            defer_index_remap: true,
            ..Default::default()
        };

        // Fails at planning time, before any fragment is rewritten.
        let plan_err = plan_compaction(&dataset, &options).await.unwrap_err();
        assert!(matches!(plan_err, Error::InvalidInput { .. }));
        let msg = plan_err.to_string();
        assert!(msg.contains("defer_index_remap"));
        assert!(msg.contains("stable row IDs"));

        // The full compact_files entry point fails the same way and leaves the
        // dataset untouched (no new manifest version, no orphaned data files).
        let version_before = dataset.manifest.version;
        let compact_err = compact_files(&mut dataset, options, None)
            .await
            .unwrap_err();
        assert!(matches!(compact_err, Error::InvalidInput { .. }));
        assert_eq!(dataset.manifest.version, version_before);
    }

    #[tokio::test]
    async fn test_defer_index_remap() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create another same dataset to mimic behavior without deferred index remap
        let mut data_gen2 = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset2 = Dataset::write(
            data_gen2.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Delete some rows to create deletions
        dataset.delete("i < 500").await.unwrap();
        dataset2.delete("i < 500").await.unwrap();

        // Create a scalar index to check this is not touched
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("scalar".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Verify the initial state - no fragment reuse index should exist
        let initial_indices = dataset.load_indices().await.unwrap();
        assert_eq!(initial_indices.len(), 1);
        assert_eq!(initial_indices[0].name, "scalar");

        // Store the original scalar index UUID for comparison
        let original_scalar_uuid = initial_indices[0].uuid;

        // Plan and execute compaction manually
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };
        let options2 = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: false,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let plan2 = plan_compaction(&dataset2, &options2).await.unwrap();

        let mut expected_all_old_frag_ids = Vec::new();
        let mut expected_all_new_frag_ids = Vec::new();
        let mut expected_all_new_frag_bitmap = RoaringBitmap::new();
        let mut expected_all_row_id_map = HashMap::new();
        let mut deferred_results = Vec::new();
        let mut immediate_results = Vec::new();

        for (task, task2) in plan.tasks().iter().zip(plan2.tasks()) {
            let deferred_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();
            let immediate_result =
                rewrite_files(Cow::Borrowed(&dataset2), task2.clone(), &options2)
                    .await
                    .unwrap();

            // Both should produce row_addrs (address-style row IDs)
            assert!(deferred_result.row_addrs.is_some());
            assert!(!deferred_result.row_addrs.as_ref().unwrap().is_empty());
            assert!(!deferred_result.row_addrs.as_ref().unwrap().is_empty());
            assert!(!deferred_result.original_fragments.is_empty());
            assert!(!deferred_result.new_fragments.is_empty());

            assert!(immediate_result.row_addrs.is_some());
            assert!(!immediate_result.original_fragments.is_empty());
            assert!(!immediate_result.new_fragments.is_empty());

            // Both should capture the same row addresses
            assert_eq!(deferred_result.row_addrs, immediate_result.row_addrs);

            deferred_results.push(deferred_result);
            immediate_results.push(immediate_result);
        }

        // Reserve fragment IDs for immediate results to build expected values
        {
            let frags: Vec<&mut Fragment> = immediate_results
                .iter_mut()
                .flat_map(|r| r.new_fragments.iter_mut())
                .collect();
            reserve_fragment_ids(&dataset2, frags.into_iter())
                .await
                .unwrap();
        }

        // Build expected values by transposing using the immediate results
        for immediate_result in &immediate_results {
            let row_addrs_bytes = immediate_result.row_addrs.as_ref().unwrap();
            let row_addrs =
                RoaringTreemap::deserialize_from(&mut Cursor::new(row_addrs_bytes)).unwrap();
            let transposed = transpose_row_addrs(
                row_addrs,
                &immediate_result.original_fragments,
                &immediate_result.new_fragments,
            );
            expected_all_row_id_map.extend(transposed);
            immediate_result.new_fragments.iter().for_each(|frag| {
                expected_all_new_frag_bitmap.insert(frag.id as u32);
            });
            expected_all_new_frag_ids.extend(
                immediate_result
                    .new_fragments
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>(),
            );
            expected_all_old_frag_ids.extend(
                immediate_result
                    .original_fragments
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>(),
            );
        }

        // Now commit the first compaction (using deferred results)
        let first_metrics = commit_compaction(
            &mut dataset,
            deferred_results.clone(),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Verify compaction happened
        assert!(first_metrics.fragments_removed > 0);
        assert!(first_metrics.fragments_added > 0);

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };

        assert_eq!(
            frag_reuse_index_meta.fragment_bitmap.clone().unwrap(),
            expected_all_new_frag_bitmap
        );
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        let frag_reuse_index =
            open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                .await
                .unwrap();
        let stats = frag_reuse_index.statistics().unwrap();
        assert_eq!(
            serde_json::to_string(&stats).unwrap(),
            dataset
                .index_statistics(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
        );

        // Verify the index has one version with the correct dataset version
        let compaction_version = &frag_reuse_index.details.versions[0];
        assert_eq!(frag_reuse_index.details.versions.len(), 1);
        assert_eq!(
            compaction_version.dataset_version,
            frag_reuse_index_meta.dataset_version
        );

        // Verify the index compaction version information matches the RewriteResults
        let mut compacted_all_old_frag_digests = Vec::new();
        let mut compacted_all_new_frag_digests = Vec::new();
        let mut transposed_map = HashMap::new();
        for group in compaction_version.groups.iter() {
            let changed_row_addr_bytes = &group.changed_row_addrs;
            let mut cursor = Cursor::new(&changed_row_addr_bytes);
            let changed_row_addrs = RoaringTreemap::deserialize_from(&mut cursor).unwrap();
            compacted_all_old_frag_digests.extend(group.old_frags.clone());
            compacted_all_new_frag_digests.extend(group.new_frags.clone());

            let group_transposed_map = transpose_row_ids_from_digest(
                changed_row_addrs,
                &group.old_frags,
                &group.new_frags,
            );
            transposed_map.extend(group_transposed_map);
        }
        assert_eq!(transposed_map, expected_all_row_id_map);
        assert_eq!(
            compacted_all_old_frag_digests
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            expected_all_old_frag_ids
        );
        assert_eq!(
            compacted_all_new_frag_digests
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            expected_all_new_frag_ids
        );

        // Verify the scalar index UUID is unchanged (it should not be remapped yet)
        let Some(current_scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        assert_eq!(current_scalar_index.uuid, original_scalar_uuid);
    }

    #[tokio::test]
    async fn test_defer_index_remap_multiple_compactions() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let mut compact_read_versions = Vec::new();
        for i in 0..10 {
            dataset
                .delete(&format!("i < {}", 500 * (i + 1)))
                .await
                .unwrap();
            let read_version = dataset.manifest.version;
            compact_files(&mut dataset, options.clone(), None)
                .await
                .unwrap();

            // Record the read version for verification if compaction has happened
            if dataset.manifest.version > read_version {
                compact_read_versions.push(read_version);
            }

            // Load and verify the fragment reuse index content
            let Some(frag_reuse_index_meta) = dataset
                .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
            else {
                panic!("Fragment reuse index must be available");
            };
            let frag_reuse_details =
                load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
                    .await
                    .unwrap();
            let frag_reuse_index =
                open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                    .await
                    .unwrap();

            // Verify the index has one version with the correct dataset version
            assert_eq!(
                frag_reuse_index
                    .details
                    .versions
                    .iter()
                    .map(|v| v.dataset_version)
                    .collect::<Vec<_>>(),
                compact_read_versions
            );
        }
    }

    #[tokio::test]
    async fn test_deferred_compaction_not_split_by_frag_reuse_index() {
        // A deferred compaction creates a fragment-reuse index covering its
        // output. Later small fragments must still compact together with that
        // (FRI-covered) output: the FRI is a system index and must not split the
        // compaction bin. Without the fix the FRI-covered fragment is isolated,
        // so only the new fragments merge and the count never returns to one.
        let data = sample_data();
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let options = CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        };

        // Two small fragments -> deferred compaction folds them into one,
        // creating the fragment-reuse index.
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 400))], data.schema());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 200,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        compact_files(&mut dataset, options.clone(), None)
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);
        assert!(
            dataset
                .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
                .is_some()
        );

        // Append two more small fragments, then compact again.
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(400, 400))], data.schema());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 200,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(
            dataset.get_fragments().len(),
            1,
            "FRI-covered fragment must compact together with the new fragments"
        );
    }

    #[tokio::test]
    async fn test_remap_index_after_compaction() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create a index to be remapped
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        // Remap without a frag reuse index should yield unsupported
        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };

        let result = remapping::remap_column_index(&mut dataset, &["i"], index_name.clone()).await;
        assert!(matches!(result, Err(Error::NotSupported { .. })));

        let plan = plan_compaction(&dataset, &options).await.unwrap();

        // Commit each rewrite task separately to simulate 3 compaction runs
        // being accumulated in the fragment reuse index
        for task in plan.tasks().iter() {
            let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();

            commit_compaction(
                &mut dataset,
                Vec::from([rewrite_result]),
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .unwrap();
        }

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        let frag_reuse_index =
            open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                .await
                .unwrap();

        assert_eq!(frag_reuse_index.details.versions.len(), plan.tasks().len());

        // Check auto-remap
        let mut all_fragment_bitmap = RoaringBitmap::new();
        dataset.fragments().iter().for_each(|f| {
            all_fragment_bitmap.insert(f.id as u32);
        });
        let Some(scalar_index_before_remap) = dataset.load_index_by_name("scalar").await.unwrap()
        else {
            panic!("scalar index must be available");
        };
        assert_eq!(
            scalar_index_before_remap.fragment_bitmap.unwrap(),
            all_fragment_bitmap
        );

        // Trigger index remap
        remapping::remap_column_index(&mut dataset, &["i"], index_name.clone())
            .await
            .unwrap();

        // Compare against original index
        let indices = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        let Some(remapped_scalar_index) = indices.into_iter().find(|idx| idx.name == "scalar")
        else {
            panic!("scalar index must be available");
        };
        assert_ne!(remapped_scalar_index.uuid, scalar_index.uuid);
        assert_eq!(
            remapped_scalar_index.fragment_bitmap.unwrap(),
            all_fragment_bitmap
        );
    }

    #[tokio::test]
    async fn test_concurrent_compaction_reindex_compaction_commit_first() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create an index
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Write some more data for reindexing
        Dataset::write(
            data_gen.batch(6_000),
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset.checkout_latest().await.unwrap();
        let mut dataset_clone = dataset.clone();

        // First commit a compaction with deferred remap
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Concurrent reindex should succeed
        dataset_clone
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Check new index does not cover the compacted files
        dataset.checkout_latest().await.unwrap();

        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        let index_frags = scalar_index
            .fragment_bitmap
            .unwrap()
            .iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            index_frags,
            dataset
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .collect::<HashSet<_>>()
        )
    }

    #[tokio::test]
    async fn test_concurrent_compaction_reindex_reindex_commit_first() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create an index
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Write some more data for reindexing
        Dataset::write(
            data_gen.batch(6_000),
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset.checkout_latest().await.unwrap();
        let mut dataset_clone = dataset.clone();

        // Concurrent reindex should succeed
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // First commit a compaction with deferred remap
        compact_files(
            &mut dataset_clone,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Check new index is auto-remapped
        dataset.checkout_latest().await.unwrap();
        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        let index_frags = scalar_index
            .fragment_bitmap
            .unwrap()
            .iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            index_frags,
            dataset
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .collect::<HashSet<_>>()
        )
    }

    #[tokio::test]
    async fn test_concurrent_cleanup_and_compaction_rebase_cleanup() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        let mut dataset_clone = dataset.clone();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };

        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // First commit the remaining 2 compaction tasks.
        let rewrite_result2 = rewrite_files(Cow::Borrowed(&dataset), tasks[1].clone(), &options)
            .await
            .unwrap();
        let rewritten_frags2 = rewrite_result2
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Get the new fragment IDs from the frag_reuse_index after commit
        let frag_reuse_index_meta2 = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .unwrap();
        let frag_reuse_details2 = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta2)
            .await
            .unwrap();
        let new_frags2 = frag_reuse_details2.versions.last().unwrap().new_frag_ids();

        let rewrite_result3 = rewrite_files(Cow::Borrowed(&dataset), tasks[2].clone(), &options)
            .await
            .unwrap();
        let rewritten_frags3 = rewrite_result3
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result3]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Get the new fragment IDs from the frag_reuse_index after commit
        let frag_reuse_index_meta3 = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .unwrap();
        let frag_reuse_details3 = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta3)
            .await
            .unwrap();
        let new_frags3 = frag_reuse_details3.versions.last().unwrap().new_frag_ids();

        // Concurrently commit a frag_reuse_index cleanup operation.
        // Because there is no index, it should remove the first version.
        // but after rebase it should contain the new compaction versions.
        cleanup_frag_reuse_index(&mut dataset_clone).await.unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 2);
        assert_eq!(
            frag_reuse_details.versions[0].old_frag_ids(),
            rewritten_frags2
        );
        assert_eq!(frag_reuse_details.versions[0].new_frag_ids(), new_frags2);
        assert_eq!(
            frag_reuse_details.versions[1].old_frag_ids(),
            rewritten_frags3
        );
        assert_eq!(frag_reuse_details.versions[1].new_frag_ids(), new_frags3);
    }

    #[tokio::test]
    async fn test_concurrent_cleanup_and_compaction_rebase_compaction() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        let mut dataset_clone = dataset.clone();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // First commit the frag_reuse_index cleanup
        // Because there is no index, it should remove the first version.
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 0);

        // Concurrently commit a rewrite
        // After rebase it should only contain the latest reuse version
        let rewrite_result2 =
            rewrite_files(Cow::Borrowed(&dataset_clone), tasks[1].clone(), &options)
                .await
                .unwrap();
        let rewritten_frags2 = rewrite_result2
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        commit_compaction(
            &mut dataset_clone,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);
        assert_eq!(
            frag_reuse_details.versions[0].old_frag_ids(),
            rewritten_frags2
        );
        // Verify new fragment IDs are non-zero (allocated by commit_compaction)
        let new_frags2 = frag_reuse_details.versions[0].new_frag_ids();
        assert!(new_frags2.iter().all(|id| *id != 0));
    }

    #[tokio::test]
    async fn test_concurrent_compactions_with_defer_index_remap() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        let mut dataset_clone = dataset.clone();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // Concurrently commit a rewrite should fail
        let rewrite_result2 =
            rewrite_files(Cow::Borrowed(&dataset_clone), tasks[1].clone(), &options)
                .await
                .unwrap();
        let result = commit_compaction(
            &mut dataset_clone,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await;
        assert!(matches!(result, Err(Error::RetryableCommitConflict { .. })));
    }

    #[tokio::test]
    async fn test_read_bitmap_index_with_defer_index_remap() {
        // Create a dataset with categorical values
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for each category
        let count1 = dataset
            .count_rows(Some("category = 1".to_owned()))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some("category = 2".to_owned()))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some("category = 3".to_owned()))
            .await
            .unwrap();

        // Create a bitmap index on the category column
        let index_name = Some("category_idx".into());
        dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices
            .iter()
            .find(|idx| idx.name == "category_idx")
            .unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("category_idx").await.unwrap() else {
            panic!("category index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("category = 1".to_owned()))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some("category = 2".to_owned()))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some("category = 3".to_owned()))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses bitmap index scan
        let mut scanner = dataset.scan();
        scanner.filter("category = 1").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[category = 1]@category_idx(Bitmap)"),
            "Expected index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_btree_index_with_defer_index_remap() {
        // Create a dataset with an incremental ID column
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(110), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for some ID ranges
        let count_low = dataset
            .count_rows(Some("id < 1000".to_owned()))
            .await
            .unwrap();
        let count_mid = dataset
            .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
            .await
            .unwrap();
        let count_high = dataset
            .count_rows(Some("id >= 5000".to_owned()))
            .await
            .unwrap();

        // Create a btree index on the id column
        let index_name = Some("id_idx".into());
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "id_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 50_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("id_idx").await.unwrap() else {
            panic!("id index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("id < 1000".to_owned()))
                .await
                .unwrap(),
            count_low
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
                .await
                .unwrap(),
            count_mid
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 5000".to_owned()))
                .await
                .unwrap(),
            count_high
        );

        // Verify that after index creation and compaction, scan uses btree index scan
        let mut scanner = dataset.scan();
        scanner.filter("id >= 2000 and id < 3000").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[id >= 2000 && id < 3000]@id_idx(BTree)"),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[rstest]
    #[case(IndexRemapMode::Compact)]
    #[case(IndexRemapMode::Direct)]
    #[tokio::test]
    async fn test_btree_index_remap_after_compaction(#[case] index_remap_mode: IndexRemapMode) {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Delete rows scattered across fragments so the remap must drop some old
        // addresses and shift the survivors.
        dataset.delete("id % 10 == 0").await.unwrap();

        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let count_low = dataset
            .count_rows(Some("id < 1000".to_owned()))
            .await
            .unwrap();
        let count_mid = dataset
            .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
            .await
            .unwrap();
        let count_high = dataset
            .count_rows(Some("id >= 5000".to_owned()))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 50_000,
            index_remap_mode,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // The index was remapped inline and must still drive scans.
        let mut scanner = dataset.scan();
        scanner.filter("id >= 2000 and id < 3000").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[id >= 2000 && id < 3000]@id_idx(BTree)"),
            "Expected scalar index query in plan: {}",
            plan
        );

        // Counts resolved through the remapped index match the pre-compaction
        // values in both remap modes.
        assert_eq!(
            dataset
                .count_rows(Some("id < 1000".to_owned()))
                .await
                .unwrap(),
            count_low
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
                .await
                .unwrap(),
            count_mid
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 5000".to_owned()))
                .await
                .unwrap(),
            count_high
        );
    }

    #[rstest]
    #[case(IndexRemapMode::Compact)]
    #[case(IndexRemapMode::Direct)]
    #[tokio::test]
    async fn test_ivf_pq_index_remap_after_compaction(#[case] index_remap_mode: IndexRemapMode) {
        use arrow_array::cast::AsArray;
        use lance_index::vector::pq::PQBuildParams;

        const DIM: u32 = 32;
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            small_ivf(),
            PQBuildParams {
                max_iters: 2,
                num_sub_vectors: 2,
                ..Default::default()
            },
        );
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();
        let original_uuid = dataset
            .load_index_by_name("vec_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid;

        // Delete rows scattered across fragments so the remap must drop some old
        // addresses and shift the survivors.
        dataset.delete("id % 10 == 0").await.unwrap();

        // Sample queries from surviving vectors and capture the pre-compaction
        // KNN answer and the surviving id set.
        let mut survivors: Vec<(i32, Vec<f32>)> = Vec::new();
        {
            let mut scanner = dataset.scan();
            scanner.project(&["id", "vec"]).unwrap();
            let batches = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            for batch in &batches {
                let ids = batch["id"].as_primitive::<Int32Type>();
                let vecs = batch["vec"].as_fixed_size_list();
                for i in 0..batch.num_rows() {
                    let v = vecs.value(i);
                    survivors.push((
                        ids.value(i),
                        v.as_primitive::<Float32Type>().values().to_vec(),
                    ));
                }
            }
        }
        let surviving_ids: std::collections::HashSet<i32> =
            survivors.iter().map(|(id, _)| *id).collect();
        let step = (survivors.len() / 16).max(1);
        let queries: Vec<Vec<f32>> = survivors
            .iter()
            .step_by(step)
            .map(|(_, v)| v.clone())
            .collect();
        let k = 10;
        let mut baseline: Vec<Vec<i32>> = Vec::new();
        for q in &queries {
            baseline.push(vector_knn_ids(&dataset, q, k).await);
        }

        // Inline remap (defer_index_remap = false): compaction physically
        // rebuilds the vector index through the configured remap mode.
        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 50_000,
                index_remap_mode,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // The index was physically remapped inline, so its uuid must change.
        assert_ne!(
            dataset
                .load_index_by_name("vec_idx")
                .await
                .unwrap()
                .unwrap()
                .uuid,
            original_uuid,
            "vector index must be physically remapped inline"
        );

        // The remap only relabels row addresses; it must not resurrect deleted
        // rows, and KNN must stay close to the pre-compaction answer in both
        // remap modes.
        for (i, q) in queries.iter().enumerate() {
            let after = vector_knn_ids(&dataset, q, k).await;
            for id in &after {
                assert!(
                    surviving_ids.contains(id),
                    "KNN returned id {id} that is not a surviving row (query #{i}, mode {index_remap_mode:?})"
                );
            }
            let overlap = after.iter().filter(|id| baseline[i].contains(id)).count();
            assert!(
                overlap >= 8,
                "KNN top-{k} diverged after compaction: overlap {overlap} < 8 (query #{i}, mode {index_remap_mode:?})"
            );
        }
    }

    #[rstest]
    #[case(IndexRemapMode::Compact)]
    #[case(IndexRemapMode::Direct)]
    #[tokio::test]
    async fn test_inverted_index_remap_after_compaction(#[case] index_remap_mode: IndexRemapMode) {
        use arrow_array::cast::AsArray;

        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col("doc", lance_datagen::array::random_sentence(1, 100, false))
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                Some("doc_idx".into()),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let original_uuid = dataset
            .load_index_by_name("doc_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid;

        // Sample a few words from a real document to drive full-text searches.
        let words: Vec<String> = {
            let mut scanner = dataset.scan();
            scanner
                .project(&["doc"])
                .unwrap()
                .limit(Some(1), None)
                .unwrap();
            let batches = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let mut words: Vec<String> = batches[0]["doc"]
                .as_string::<i32>()
                .value(0)
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            words.sort();
            words.dedup();
            words.truncate(3);
            words
        };
        assert!(!words.is_empty(), "sampled document must contain words");

        // Delete rows scattered across fragments so the remap must drop some old
        // addresses and shift the survivors.
        dataset.delete("id % 10 == 0").await.unwrap();

        // Capture the post-deletion full-text-search counts (resolved through the
        // index + deletion vectors) before compaction physically remaps.
        let mut before = Vec::new();
        for word in &words {
            let mut scanner = dataset.scan();
            scanner
                .full_text_search(FullTextSearchQuery::new(word.clone()))
                .unwrap();
            scanner.project::<String>(&[]).unwrap().with_row_id();
            before.push(scanner.count_rows().await.unwrap());
        }

        // Inline remap (defer_index_remap = false): compaction physically rebuilds
        // the inverted index through the configured remap mode.
        let options = CompactionOptions {
            target_rows_per_fragment: 50_000,
            index_remap_mode,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // The index was physically remapped inline, so its uuid must change.
        assert_ne!(
            dataset
                .load_index_by_name("doc_idx")
                .await
                .unwrap()
                .unwrap()
                .uuid,
            original_uuid,
            "inverted index must be physically remapped inline (mode {index_remap_mode:?})"
        );

        // The remapped index must still drive full-text search.
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(words[0].clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("MatchQuery"),
            "Expected inverted index scan in plan: {}",
            plan
        );

        // Counts resolved through the remapped index match the pre-compaction
        // values in both remap modes.
        for (word, expected) in words.iter().zip(before) {
            let mut scanner = dataset.scan();
            scanner
                .full_text_search(FullTextSearchQuery::new(word.clone()))
                .unwrap();
            scanner.project::<String>(&[]).unwrap().with_row_id();
            assert_eq!(
                scanner.count_rows().await.unwrap(),
                expected,
                "full-text count for {word:?} changed after compaction (mode {index_remap_mode:?})"
            );
        }
    }

    #[tokio::test]
    async fn test_read_inverted_index_with_defer_index_remap() {
        // Generate random words using lance-datagen
        let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
        let doc_col = words_gen
            .generate_default(lance_datagen::RowCount::from(6000))
            .unwrap();

        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
            vec![doc_col.clone()],
        )
        .unwrap();
        let schema_ref = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
        let mut dataset = Dataset::write(
            stream,
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Get initial counts for some word searches
        // Extract some test words from the generated documents
        let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let sample_words: Vec<String> = large_string_array
            .value(0)
            .split_whitespace()
            .take(10)
            .map(|s| s.to_string())
            .collect();
        let test_word1 = &sample_words[0];
        let test_word2 = &sample_words[1];
        let test_word3 = &sample_words[2];

        // Create an inverted index on the doc column
        let index_name = Some("doc_idx".into());
        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                index_name.clone(),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
            panic!("doc index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Initial scan
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count1 = scanner.count_rows().await.unwrap();
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count2 = scanner.count_rows().await.unwrap();
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count3 = scanner.count_rows().await.unwrap();

        // Verify that after index creation and compaction, scan uses inverted index scan
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("MatchQuery"),
            "Expected inverted index scan in plan: {}",
            plan
        );
        assert!(
            !plan.contains("LanceScan"),
            "Expected no fragment scan in plan: {}",
            plan
        );

        // Reindex to the latest
        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                index_name.clone(),
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Verify that scans still work correctly and return the same counts
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count1);
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count2);
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count3);
    }

    /// Deferred compaction that materializes deletions must not corrupt an
    /// inverted (FTS) index read through the fragment-reuse index. The index's
    /// posting lists reference doc_ids positionally; if the load-time remap
    /// dropped the deleted rows it would renumber the doc_ids and desync the
    /// posting lists (out-of-bounds `num_tokens`, wrong/stale row ids). The
    /// tombstone-preserve-positions load path must keep results correct in the
    /// FRI window and after the physical remap + trim.
    #[tokio::test]
    async fn test_read_inverted_index_with_defer_index_remap_and_deletions() {
        // Enough surviving docs for several compressed posting-list blocks
        // (BLOCK_SIZE = 128), split across several fragments so compaction has
        // real work — but no larger.
        const ROWS: i32 = 1200;
        const DELETED: i32 = 400;

        // Every row contains "lance", so the term matches all live rows; `id`
        // tells us exactly which rows survive.
        let ids = Int32Array::from_iter_values(0..ROWS);
        let docs = LargeStringArray::from_iter_values((0..ROWS).map(|_| "lance apple orange"));
        let batch = RecordBatch::try_new(
            Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("doc", DataType::LargeUtf8, false),
            ])
            .into(),
            vec![Arc::new(ids) as ArrayRef, Arc::new(docs) as ArrayRef],
        )
        .unwrap();
        let schema_ref = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
        let mut dataset = Dataset::write(
            stream,
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 200, // 6 fragments
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                Some("doc_idx".into()),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Delete a prefix, then deferred-compact so the deletions are
        // materialized into the fragment-reuse index the index is read through.
        dataset.delete(&format!("id < {DELETED}")).await.unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(
            dataset
                .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
                .is_some(),
            "deferred compaction must leave a fragment-reuse index"
        );

        // FTS "lance" → sorted surviving ids. Projecting `id` forces a take, so
        // a stale row address would error or return a wrong/dead row.
        async fn search_ids(dataset: &Dataset) -> Vec<i32> {
            let mut scanner = dataset.scan();
            scanner
                .full_text_search(FullTextSearchQuery::new("lance".to_owned()))
                .unwrap();
            scanner.project::<&str>(&["id"]).unwrap();
            let batches = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let mut ids: Vec<i32> = batches
                .iter()
                .flat_map(|b| {
                    b.column_by_name("id")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect();
            ids.sort_unstable();
            ids
        }

        let expected = (DELETED..ROWS).collect::<Vec<_>>();

        // FRI window: index read through the reuse index.
        let during = search_ids(&dataset).await;
        assert_eq!(
            during, expected,
            "FRI-window FTS must return exactly the surviving rows (no resurrection, no loss, no stale rows)"
        );

        // Physical remap + trim: must still be correct.
        remapping::remap_column_index(&mut dataset, &["doc"], Some("doc_idx".into()))
            .await
            .unwrap();
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        let after = search_ids(&dataset).await;
        assert_eq!(
            after, expected,
            "FTS must stay correct after physical remap + fragment-reuse trim"
        );
    }

    #[tokio::test]
    async fn test_read_ngram_index_with_defer_index_remap() {
        // Generate random words using lance-datagen
        let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
        let doc_col = words_gen
            .generate_default(lance_datagen::RowCount::from(6000))
            .unwrap();

        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
            vec![doc_col.clone()],
        )
        .unwrap();
        let schema_ref = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
        let mut dataset = Dataset::write(
            stream,
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Get initial counts for some word searches
        // Extract some test words from the generated documents
        let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let sample_words: Vec<String> = large_string_array
            .value(0)
            .split_whitespace()
            .take(10)
            .map(|s| s.to_string())
            .collect();
        let test_word1 = &sample_words[0];
        let test_word2 = &sample_words[1];
        let test_word3 = &sample_words[2];

        // Create an inverted index on the doc column
        let index_name = Some("doc_idx".into());
        dataset
            .create_index(
                &["doc"],
                IndexType::NGram,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

        // Initial scan
        let count1 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
            .await
            .unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
            panic!("doc index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses inverted index scan
        let mut scanner = dataset.scan();
        scanner
            .filter(&format!("contains(doc, '{}')", test_word1))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[contains(doc, Utf8"),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_label_list_index_with_defer_index_remap() {
        // Create a dataset with list data for labels
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "labels",
                lance_datagen::array::rand_list_any(
                    lance_datagen::array::cycle::<Int64Type>(vec![1, 2, 3, 4, 5]),
                    false,
                ),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for different label values
        let count1 = dataset
            .count_rows(Some("array_has_any(labels, [1])".to_owned()))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some("array_has_any(labels, [5])".to_owned()))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some("array_has_any(labels, [10])".to_owned()))
            .await
            .unwrap();

        // Create a label list index on the labels column
        let index_name = Some("labels_idx".into());
        dataset
            .create_index(
                &["labels"],
                IndexType::LabelList,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2000,
            defer_index_remap: true,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify that the index UUID remains unchanged
        let indices = dataset.load_indices().await.unwrap();
        let current_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [1])".to_owned()))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [5])".to_owned()))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [10])".to_owned()))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses label list index scan
        let mut scanner = dataset.scan();
        scanner.filter("array_has_any(labels, [1])").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains(
                "ScalarIndexQuery: query=[array_has_any(labels, List([1]))]@labels_idx(LabelList)",
            ),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_ivf_pq_index_v3_with_defer_index_remap() {
        // Create a dataset with vector data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get some query vectors for KNN search
        let query_vec1: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(0.0, 128));
        let query_vec2: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(1.1, 128));
        let query_vec3: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(2.2, 128));

        // Get initial KNN search results
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results1 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count1 = results1.len();

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec2, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results2 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count2 = results2.len();

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec3, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results3 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count3 = results3.len();

        // Create an IVF-PQ index on the vec column
        let index_name = Some("vec_idx".into());
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                index_name.clone(),
                &VectorIndexParams {
                    metric_type: DistanceType::L2,
                    stages: vec![
                        StageParams::Ivf(IvfBuildParams {
                            max_iters: 2,
                            num_partitions: Some(2),
                            sample_rate: 2,
                            ..Default::default()
                        }),
                        StageParams::PQ(PQBuildParams {
                            max_iters: 2,
                            num_sub_vectors: 2,
                            ..Default::default()
                        }),
                    ],
                    version: crate::index::vector::IndexFileVersion::V3,
                    skip_transpose: false,
                    runtime_hints: Default::default(),
                },
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("vec_idx").await.unwrap() else {
            panic!("vec index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that KNN searches still work correctly and return the same counts
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results1 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results1.len(), count1);

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec2, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results2 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results2.len(), count2);

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec3, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results3 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results3.len(), count3);

        // Verify that after index creation and compaction, scan uses vector index scan
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "Expected vector index scan in plan: {}",
            plan
        );
        assert!(
            !plan.contains("LanceScan"),
            "Expected no fragment scan in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_ivf_rq_index_v3_with_defer_index_remap() {
        use arrow_array::cast::AsArray;
        use lance_index::vector::bq::RQBuildParams;

        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let stored: Vec<Vec<f32>> = {
            let mut scanner = dataset.scan();
            scanner.project(&["vec"]).unwrap();
            let batches = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let mut out = Vec::new();
            for batch in &batches {
                let vecs = batch["vec"].as_fixed_size_list();
                for i in 0..batch.num_rows() {
                    let values = vecs.value(i);
                    let values = values.as_primitive::<Float32Type>();
                    out.push(values.values().to_vec());
                }
            }
            out
        };

        let index_name = Some("vec_idx".into());
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                index_name.clone(),
                &VectorIndexParams {
                    metric_type: DistanceType::L2,
                    stages: vec![
                        StageParams::Ivf(IvfBuildParams {
                            max_iters: 2,
                            num_partitions: Some(2),
                            sample_rate: 2,
                            ..Default::default()
                        }),
                        StageParams::RQ(RQBuildParams::new(1)),
                    ],
                    version: crate::index::vector::IndexFileVersion::V3,
                    skip_transpose: false,
                    runtime_hints: Default::default(),
                },
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        let Some(current_index) = dataset.load_index_by_name("vec_idx").await.unwrap() else {
            panic!("vec index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        let frag_reuse_present = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME);
        assert!(
            frag_reuse_present,
            "defer_index_remap must record a {} index",
            FRAG_REUSE_INDEX_NAME
        );

        let sample_step = (stored.len() / 8).max(1);
        let mut checked = 0;
        for query in stored.iter().step_by(sample_step) {
            let query_vec = PrimitiveArray::<Float32Type>::from_iter_values(query.iter().copied());
            let mut scanner = dataset.scan();
            scanner.nearest("vec", &query_vec, 5).unwrap();
            scanner.project(&["vec"]).unwrap().with_row_id();
            let batches = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert!(!batches.is_empty(), "query returned no batches");
            let top = &batches[0];
            assert!(top.num_rows() > 0, "query returned empty top batch");
            let top_vec = top["vec"].as_fixed_size_list().value(0);
            let top_vec = top_vec.as_primitive::<Float32Type>();
            assert_eq!(
                top_vec.values(),
                query.as_slice(),
                "top-1 self-recall returned a different vector than the query"
            );
            checked += 1;
        }
        assert!(checked > 0, "expected to check at least one stored vector");
    }

    /// Build an `id` + `vec` dataset, create the given IVF vector index,
    /// optionally delete rows, then run deferred compaction (which materializes
    /// the deletions into the fragment-reuse index) and assert that KNN over
    /// surviving vectors during the FRI window (a) never returns a deleted row
    /// and (b) stays consistent with the pre-compaction answer.
    ///
    /// The deletion path is the interesting one: materialized deletions drop
    /// rows from the quantization storage at load time, which shifts storage
    /// positions. Flat storage (FLAT/PQ/SQ/RQ) is scanned linearly so this is
    /// fine, but the HNSW graph addresses storage positionally and is not
    /// frag-reuse aware, so a desync would surface here as recall collapse or a
    /// resurrected/again-deleted row.
    /// Top-k `id`s for a KNN query against the `vec` column.
    async fn vector_knn_ids(dataset: &Dataset, query: &[f32], k: usize) -> Vec<i32> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Float32Type, Int32Type};
        let qa = PrimitiveArray::<Float32Type>::from_iter_values(query.iter().copied());
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &qa, k).unwrap();
        scanner.project(&["id"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut ids = Vec::new();
        for b in &batches {
            ids.extend(b["id"].as_primitive::<Int32Type>().values().iter().copied());
        }
        ids
    }

    async fn check_vector_defer_compaction(
        params: VectorIndexParams,
        delete_predicate: Option<&str>,
        k: usize,
        min_overlap: usize,
    ) {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Float32Type, Int32Type};
        use lance_datagen::Dimension;

        const DIM: u32 = 32;
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();
        let original_uuid = dataset
            .load_index_by_name("vec_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid;

        if let Some(pred) = delete_predicate {
            dataset.delete(pred).await.unwrap();
        }

        // Collect surviving (id, vec) pairs and the set of surviving ids.
        let mut survivors: Vec<(i32, Vec<f32>)> = Vec::new();
        {
            let mut scanner = dataset.scan();
            scanner.project(&["id", "vec"]).unwrap();
            let batches = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            for batch in &batches {
                let ids = batch["id"].as_primitive::<Int32Type>();
                let vecs = batch["vec"].as_fixed_size_list();
                for i in 0..batch.num_rows() {
                    let v = vecs.value(i);
                    let v = v.as_primitive::<Float32Type>().values().to_vec();
                    survivors.push((ids.value(i), v));
                }
            }
        }
        assert!(!survivors.is_empty());
        let surviving_ids: std::collections::HashSet<i32> =
            survivors.iter().map(|(id, _)| *id).collect();

        // Sample queries from survivors and capture the pre-compaction answer.
        let step = (survivors.len() / 16).max(1);
        let queries: Vec<(i32, Vec<f32>)> = survivors.iter().step_by(step).cloned().collect();
        let mut baseline: Vec<Vec<i32>> = Vec::new();
        for (_, q) in &queries {
            baseline.push(vector_knn_ids(&dataset, q, k).await);
        }

        // Deferred compaction materializes the deletions into the frag-reuse index.
        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
            "deferred compaction must record a frag-reuse index"
        );
        assert_eq!(
            dataset
                .load_index_by_name("vec_idx")
                .await
                .unwrap()
                .unwrap()
                .uuid,
            original_uuid,
            "index must not be physically remapped yet (FRI window)"
        );

        // During the FRI window: no deleted rows, and stable vs the baseline.
        for (i, (_, q)) in queries.iter().enumerate() {
            let after = vector_knn_ids(&dataset, q, k).await;
            for id in &after {
                assert!(
                    surviving_ids.contains(id),
                    "KNN returned id {id} that is not a surviving row (query #{i})"
                );
            }
            let overlap = after.iter().filter(|id| baseline[i].contains(id)).count();
            assert!(
                overlap >= min_overlap,
                "KNN top-{k} diverged after deferred compaction: overlap {overlap} < {min_overlap} (query #{i})"
            );
        }
    }

    fn small_ivf() -> lance_index::vector::ivf::IvfBuildParams {
        lance_index::vector::ivf::IvfBuildParams {
            max_iters: 2,
            num_partitions: Some(2),
            sample_rate: 2,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_ivf_flat_defer_compaction_with_deletions() {
        let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, small_ivf());
        // Flat storage is scanned linearly; dropping deleted rows is exact.
        check_vector_defer_compaction(params, Some("id < 1500"), 10, 10).await;
    }

    #[tokio::test]
    async fn test_ivf_hnsw_sq_defer_compaction_merge_only() {
        use lance_index::vector::{hnsw::builder::HnswBuildParams, sq::builder::SQBuildParams};
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            small_ivf(),
            HnswBuildParams::default(),
            SQBuildParams::default(),
        );
        // No deletions: storage positions are stable, so the graph stays aligned.
        check_vector_defer_compaction(params, None, 10, 9).await;
    }

    // NOTE: IVF_HNSW_* under materialized deletions is a known gap (lance#3993,
    // HNSW auto-remap not implemented) — the HNSW graph isn't realigned after the
    // frag-reuse drop. Deferred remap is gated off for HNSW tables, so there is
    // no lance-level reproducer here; the gate is tested in the data plane.
    // Merge-only HNSW is covered (see the *_remap_and_trim tests).

    #[tokio::test]
    async fn test_ivf_pq_defer_compaction_with_deletions() {
        use lance_index::vector::pq::PQBuildParams;
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            small_ivf(),
            PQBuildParams {
                max_iters: 2,
                num_sub_vectors: 2,
                ..Default::default()
            },
        );
        check_vector_defer_compaction(params, Some("id < 1500"), 10, 8).await;
    }

    #[tokio::test]
    async fn test_ivf_sq_defer_compaction_with_deletions() {
        use lance_index::vector::sq::builder::SQBuildParams;
        let params = VectorIndexParams::with_ivf_sq_params(
            DistanceType::L2,
            small_ivf(),
            SQBuildParams::default(),
        );
        check_vector_defer_compaction(params, Some("id < 1500"), 10, 8).await;
    }

    #[tokio::test]
    async fn test_ivf_rq_defer_compaction_with_deletions() {
        use lance_index::vector::bq::RQBuildParams;
        let params = VectorIndexParams::with_ivf_rq_params(
            DistanceType::L2,
            small_ivf(),
            RQBuildParams::new(1),
        );
        check_vector_defer_compaction(params, Some("id < 1500"), 10, 8).await;
    }

    /// Merge-only deferred compaction, then a PHYSICAL remap + FRI trim. Asserts
    /// the index is rebuilt, the fragment-reuse index trims to zero versions,
    /// and KNN stays consistent with the pre-compaction answer through both the
    /// FRI window and the physical remap. (HNSW rebuilds its graph on physical
    /// remap, so the overlap is recall-tolerant.)
    async fn check_vector_remap_and_trim(
        params: VectorIndexParams,
        k: usize,
        window_overlap: usize,
        post_remap_overlap: Option<usize>,
    ) {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Float32Type, Int32Type};
        use lance_datagen::Dimension;

        const DIM: u32 = 32;
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();
        let original_uuid = dataset
            .load_index_by_name("vec_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid;

        // Sample queries from stored vectors + capture the pre-compaction answer.
        let mut rows: Vec<Vec<f32>> = Vec::new();
        {
            let mut scanner = dataset.scan();
            scanner.project(&["vec"]).unwrap();
            let batches = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            for batch in &batches {
                let vecs = batch["vec"].as_fixed_size_list();
                for i in 0..batch.num_rows() {
                    let v = vecs.value(i);
                    rows.push(v.as_primitive::<Float32Type>().values().to_vec());
                }
            }
        }
        let step = (rows.len() / 16).max(1);
        let queries: Vec<Vec<f32>> = rows.iter().step_by(step).cloned().collect();
        let mut baseline: Vec<Vec<i32>> = Vec::new();
        for q in &queries {
            baseline.push(vector_knn_ids(&dataset, q, k).await);
        }

        // Merge-only deferred compaction.
        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(metrics.fragments_removed > 0);
        assert_eq!(
            dataset
                .load_index_by_name("vec_idx")
                .await
                .unwrap()
                .unwrap()
                .uuid,
            original_uuid,
            "index must not be physically remapped yet (FRI window)"
        );
        for (i, q) in queries.iter().enumerate() {
            let window = vector_knn_ids(&dataset, q, k).await;
            let overlap = window.iter().filter(|id| baseline[i].contains(id)).count();
            assert!(
                overlap >= window_overlap,
                "FRI-window KNN diverged: overlap {overlap} < {window_overlap} (query #{i})"
            );
        }

        // Physical remap + trim the fragment-reuse index.
        remapping::remap_column_index(&mut dataset, &["vec"], Some("vec_idx".into()))
            .await
            .unwrap();
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();

        let remapped_uuid = dataset
            .load_index_by_name("vec_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid;
        assert_ne!(
            remapped_uuid, original_uuid,
            "index should have been physically remapped"
        );
        if let Some(meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        {
            let versions = load_frag_reuse_index_details(&dataset, &meta)
                .await
                .unwrap()
                .versions
                .len();
            assert_eq!(versions, 0, "frag-reuse index must trim to zero versions");
        }

        for (i, q) in queries.iter().enumerate() {
            let after = vector_knn_ids(&dataset, q, k).await;
            // No stale/desynced addresses (a bad address fails the take above).
            assert!(
                !after.is_empty(),
                "post-remap KNN returned no rows (query #{i})"
            );
            // Physical remap rebuilds the HNSW graph, so recall is only compared
            // for the exact (non-HNSW) types.
            if let Some(min_overlap) = post_remap_overlap {
                let overlap = after.iter().filter(|id| baseline[i].contains(id)).count();
                assert!(
                    overlap >= min_overlap,
                    "post-remap KNN diverged: overlap {overlap} < {min_overlap} (query #{i})"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_ivf_flat_remap_and_trim() {
        let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, small_ivf());
        check_vector_remap_and_trim(params, 10, 8, Some(8)).await;
    }

    // Regression: PQ storage used to remap its codes through the frag-reuse
    // index but keep the pre-remap `row_ids` field, so search returned stale
    // (compacted-away) addresses and the take failed with "fragment ... does
    // not exist" — even merge-only, and only observable when the query fetches
    // row content (the existing `test_read_ivf_pq_index_v3_with_defer_index_remap`
    // projects no columns, so it never takes and missed this).
    #[tokio::test]
    async fn test_ivf_pq_remap_and_trim() {
        use lance_index::vector::pq::PQBuildParams;
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            small_ivf(),
            PQBuildParams {
                max_iters: 2,
                num_sub_vectors: 2,
                ..Default::default()
            },
        );
        check_vector_remap_and_trim(params, 10, 8, Some(8)).await;
    }

    #[tokio::test]
    async fn test_ivf_sq_remap_and_trim() {
        use lance_index::vector::sq::builder::SQBuildParams;
        let params = VectorIndexParams::with_ivf_sq_params(
            DistanceType::L2,
            small_ivf(),
            SQBuildParams::default(),
        );
        check_vector_remap_and_trim(params, 10, 8, Some(8)).await;
    }

    #[tokio::test]
    async fn test_ivf_rq_remap_and_trim() {
        use lance_index::vector::bq::RQBuildParams;
        let params = VectorIndexParams::with_ivf_rq_params(
            DistanceType::L2,
            small_ivf(),
            RQBuildParams::new(1),
        );
        check_vector_remap_and_trim(params, 10, 8, Some(8)).await;
    }

    #[tokio::test]
    async fn test_ivf_hnsw_sq_remap_and_trim() {
        use lance_index::vector::{hnsw::builder::HnswBuildParams, sq::builder::SQBuildParams};
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            small_ivf(),
            HnswBuildParams::default(),
            SQBuildParams::default(),
        );
        // Physical remap rebuilds the HNSW graph, so use a recall-tolerant overlap.
        check_vector_remap_and_trim(params, 10, 7, None).await;
    }

    #[tokio::test]
    async fn test_ivf_hnsw_pq_remap_and_trim() {
        use lance_index::vector::{hnsw::builder::HnswBuildParams, pq::PQBuildParams};
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            DistanceType::L2,
            small_ivf(),
            HnswBuildParams::default(),
            PQBuildParams {
                max_iters: 2,
                num_sub_vectors: 2,
                ..Default::default()
            },
        );
        check_vector_remap_and_trim(params, 10, 7, None).await;
    }

    // Scalar index correctness across deferred compaction WITH materialized
    // deletions. The existing test_read_*_index_with_defer_index_remap tests are
    // merge-only and project no columns (count-only), so they never take and
    // never exercise the deletion drop path. These add an `id` column, delete a
    // prefix, defer-compact, then run the indexed query *projecting id* (a take)
    // and assert no deleted row is returned. Bitmap/BTree have no positional
    // internal structure so the drop path is exact; the Inverted (FTS) index
    // does (see its test below), and currently desyncs under deletions.

    #[tokio::test]
    async fn test_bitmap_index_defer_compaction_with_deletions() {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int32Type;
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();
        dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                Some("category_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset.delete("id < 1500").await.unwrap();
        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
            "deferred compaction must record a frag-reuse index"
        );

        let mut scanner = dataset.scan();
        scanner.filter("category = 3").unwrap();
        scanner.project(&["id"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut returned = 0;
        for b in &batches {
            for id in b["id"].as_primitive::<Int32Type>().values() {
                assert!(
                    *id >= 1500,
                    "bitmap returned deleted id {id} in the FRI window"
                );
                returned += 1;
            }
        }
        assert!(returned > 0, "expected surviving category=3 rows");
    }

    // NOTE: Inverted/FTS under materialized deletions is broken (BM25 scores
    // via positional num_tokens[doc_id]; the frag-reuse drop shifts doc_id
    // positions -> out-of-bounds). It is gated off defer in the data plane
    // until fixed, so there is no lance-level reproducer here. Merge-only FTS
    // is covered by test_read_inverted_index_with_defer_index_remap.

    #[tokio::test]
    async fn test_default_compaction_planner() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let schema = data.schema();

        // Create dataset with multiple small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], schema.clone());
        let write_params = WriteParams {
            max_rows_per_file: 2000,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.get_fragments().len(), 5);

        // Test default planner
        let options = CompactionOptions {
            target_rows_per_fragment: 5000,
            materialize_deletions_threshold: 2.0,
            ..Default::default()
        };

        let planner = DefaultCompactionPlanner::new(options);
        let plan = planner.plan(&dataset).await.unwrap();

        // Should create tasks to compact small fragments
        assert!(!plan.tasks.is_empty());
        assert_eq!(plan.read_version, dataset.manifest.version);
        // make sure options.validate() worked
        assert!(!plan.options.materialize_deletions);
    }

    #[test]
    fn test_from_dataset_config() {
        let config = HashMap::from([
            (
                "lance.compaction.target_rows_per_fragment".to_string(),
                "500000".to_string(),
            ),
            (
                "lance.compaction.max_rows_per_group".to_string(),
                "2048".to_string(),
            ),
            (
                "lance.compaction.max_bytes_per_file".to_string(),
                "1000000".to_string(),
            ),
            (
                "lance.compaction.materialize_deletions".to_string(),
                "false".to_string(),
            ),
            (
                "lance.compaction.materialize_deletions_threshold".to_string(),
                "0.25".to_string(),
            ),
            (
                "lance.compaction.defer_index_remap".to_string(),
                "true".to_string(),
            ),
            (
                "lance.compaction.batch_size".to_string(),
                "4096".to_string(),
            ),
            (
                "lance.compaction.io_buffer_size".to_string(),
                "1073741824".to_string(),
            ),
            (
                "lance.compaction.compaction_mode".to_string(),
                "try_binary_copy".to_string(),
            ),
            (
                "lance.compaction.binary_copy_read_batch_bytes".to_string(),
                "8388608".to_string(),
            ),
            (
                "lance.compaction.index_remap_mode".to_string(),
                "compact".to_string(),
            ),
        ]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 500_000);
        assert_eq!(opts.max_rows_per_group, 2048);
        assert_eq!(opts.max_bytes_per_file, Some(1_000_000));
        assert!(!opts.materialize_deletions);
        assert!((opts.materialize_deletions_threshold - 0.25).abs() < f32::EPSILON);
        assert!(opts.defer_index_remap);
        assert_eq!(opts.batch_size, Some(4096));
        assert_eq!(opts.io_buffer_size, Some(1_073_741_824));
        assert_eq!(opts.compaction_mode, Some(CompactionMode::TryBinaryCopy));
        assert_eq!(opts.binary_copy_read_batch_bytes, Some(8_388_608));
        // A non-default value proves the config string was actually parsed.
        assert_eq!(opts.index_remap_mode, IndexRemapMode::Compact);
    }

    #[test]
    fn test_from_dataset_config_empty() {
        let config = HashMap::new();
        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        let defaults = CompactionOptions::default();
        assert_eq!(
            opts.target_rows_per_fragment,
            defaults.target_rows_per_fragment
        );
        assert_eq!(opts.max_rows_per_group, defaults.max_rows_per_group);
        assert_eq!(opts.max_bytes_per_file, defaults.max_bytes_per_file);
        assert_eq!(opts.materialize_deletions, defaults.materialize_deletions);
        assert_eq!(
            opts.materialize_deletions_threshold,
            defaults.materialize_deletions_threshold
        );
        assert_eq!(opts.defer_index_remap, defaults.defer_index_remap);
        assert_eq!(opts.index_remap_mode, defaults.index_remap_mode);
        assert_eq!(opts.index_remap_mode, IndexRemapMode::Direct);
        assert_eq!(opts.batch_size, defaults.batch_size);
        assert_eq!(opts.compaction_mode, defaults.compaction_mode);
        assert_eq!(
            opts.binary_copy_read_batch_bytes,
            defaults.binary_copy_read_batch_bytes
        );
    }

    #[test]
    fn test_from_dataset_config_partial() {
        let config = HashMap::from([(
            "lance.compaction.target_rows_per_fragment".to_string(),
            "500000".to_string(),
        )]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 500_000);
        // Other fields should remain at defaults
        let defaults = CompactionOptions::default();
        assert_eq!(opts.max_rows_per_group, defaults.max_rows_per_group);
        assert_eq!(opts.max_bytes_per_file, defaults.max_bytes_per_file);
        assert_eq!(opts.materialize_deletions, defaults.materialize_deletions);
        assert_eq!(opts.defer_index_remap, defaults.defer_index_remap);
        assert_eq!(opts.batch_size, defaults.batch_size);
        assert_eq!(opts.compaction_mode, defaults.compaction_mode);
        assert_eq!(
            opts.binary_copy_read_batch_bytes,
            defaults.binary_copy_read_batch_bytes
        );
    }

    #[test]
    fn test_from_dataset_config_ignores_other_keys() {
        let config = HashMap::from([
            (
                "lance.compaction.target_rows_per_fragment".to_string(),
                "500000".to_string(),
            ),
            (
                "lance.auto_cleanup.interval".to_string(),
                "3600".to_string(),
            ),
            ("some.other.key".to_string(), "value".to_string()),
        ]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 500_000);
    }

    #[test]
    fn test_from_dataset_config_invalid_value() {
        let config = HashMap::from([(
            "lance.compaction.target_rows_per_fragment".to_string(),
            "not_a_number".to_string(),
        )]);

        let result = CompactionOptions::from_dataset_config(&config);
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("target_rows_per_fragment"));
        assert!(err_msg.contains("not_a_number"));
    }

    #[test]
    fn test_from_dataset_config_invalid_bool() {
        let config = HashMap::from([(
            "lance.compaction.materialize_deletions".to_string(),
            "yes".to_string(),
        )]);

        let result = CompactionOptions::from_dataset_config(&config);
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("materialize_deletions"));
        assert!(err_msg.contains("yes"));
    }

    #[test]
    fn test_from_dataset_config_unknown_compaction_key() {
        // Unknown keys should be ignored (with a warning) for forwards compatibility
        let config = HashMap::from([(
            "lance.compaction.unknown_key".to_string(),
            "value".to_string(),
        )]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        // Should return defaults since the unknown key is skipped
        let defaults = CompactionOptions::default();
        assert_eq!(
            opts.target_rows_per_fragment,
            defaults.target_rows_per_fragment
        );
    }

    #[test]
    fn test_from_dataset_config_invalid_compaction_mode() {
        let config = HashMap::from([(
            "lance.compaction.compaction_mode".to_string(),
            "invalid_mode".to_string(),
        )]);

        let result = CompactionOptions::from_dataset_config(&config);
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("invalid_mode"));
    }

    #[test]
    fn test_apply_dataset_config_overrides() {
        let config = HashMap::from([(
            "lance.compaction.target_rows_per_fragment".to_string(),
            "500000".to_string(),
        )]);

        let mut opts = CompactionOptions {
            max_rows_per_group: 4096,
            ..Default::default()
        };
        opts.apply_dataset_config(&config).unwrap();

        // Config value should be applied
        assert_eq!(opts.target_rows_per_fragment, 500_000);
        // Explicitly set value should be preserved (config didn't have this key)
        assert_eq!(opts.max_rows_per_group, 4096);
    }

    #[test]
    fn test_apply_dataset_config_overwrites_matching_field() {
        let config = HashMap::from([(
            "lance.compaction.max_rows_per_group".to_string(),
            "2048".to_string(),
        )]);

        let mut opts = CompactionOptions {
            max_rows_per_group: 4096,
            ..Default::default()
        };
        opts.apply_dataset_config(&config).unwrap();

        // Config value should overwrite the pre-set value
        assert_eq!(opts.max_rows_per_group, 2048);
    }

    #[tokio::test]
    async fn test_max_source_fragments() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let schema = data.schema();

        // Create 10 small fragments (100 rows each) via 10 appends
        let write_params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.slice(0, 100))], schema.clone()),
            test_uri,
            Some(write_params.clone()),
        )
        .await
        .unwrap();
        for i in 1..10 {
            let mut append_params = write_params.clone();
            append_params.mode = WriteMode::Append;
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(data.slice(i * 100, 100))], schema.clone()),
                test_uri,
                Some(append_params),
            )
            .await
            .unwrap();
        }

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 10);

        // Plan without limit - all 10 fragments should be candidates.
        // Use a target that splits the 10 fragments into multiple tasks.
        let opts_no_limit = CompactionOptions {
            target_rows_per_fragment: 250,
            ..Default::default()
        };
        let plan_all = plan_compaction(&dataset, &opts_no_limit).await.unwrap();
        let total_source_frags: usize = plan_all.tasks().iter().map(|t| t.fragments.len()).sum();
        assert_eq!(total_source_frags, 10);
        assert!(
            plan_all.num_tasks() > 2,
            "need multiple tasks to test bounding, got {}",
            plan_all.num_tasks()
        );

        // Plan with max_source_fragments=4 should include tasks covering <= 4
        // source fragments
        let opts_bounded = CompactionOptions {
            target_rows_per_fragment: 250,
            max_source_fragments: Some(4),
            ..Default::default()
        };
        let plan_bounded = plan_compaction(&dataset, &opts_bounded).await.unwrap();
        let bounded_source_frags: usize =
            plan_bounded.tasks().iter().map(|t| t.fragments.len()).sum();
        assert!(
            bounded_source_frags <= 4,
            "expected at most 4 source fragments, got {bounded_source_frags}"
        );
        assert!(
            bounded_source_frags > 0,
            "expected at least 1 source fragment in bounded plan"
        );
        assert!(
            plan_bounded.num_tasks() < plan_all.num_tasks(),
            "bounded plan ({}) should have fewer tasks than unbounded ({})",
            plan_bounded.num_tasks(),
            plan_all.num_tasks()
        );

        // Execute bounded compaction incrementally
        let mut dataset = dataset;
        compact_files(&mut dataset, opts_bounded, None)
            .await
            .unwrap();
        let after_first = dataset.get_fragments().len();
        assert!(
            after_first < 10,
            "expected fewer than 10 fragments after first compaction, got {after_first}"
        );
        assert!(
            after_first > 1,
            "expected partial compaction (not fully compacted), got {after_first}"
        );

        // Run again to make more progress
        let opts_bounded = CompactionOptions {
            target_rows_per_fragment: 250,
            max_source_fragments: Some(4),
            ..Default::default()
        };
        compact_files(&mut dataset, opts_bounded, None)
            .await
            .unwrap();
        let after_second = dataset.get_fragments().len();
        assert!(
            after_second <= after_first,
            "expected progress: {after_second} should be <= {after_first}"
        );
    }

    #[tokio::test]
    async fn test_compaction_uses_manifest_config() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let schema = data.schema();

        // Create dataset with small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], schema.clone());
        let write_params = WriteParams {
            max_rows_per_file: 2000,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.get_fragments().len(), 5);

        // Set compaction config in manifest
        dataset
            .update_config([
                ("lance.compaction.target_rows_per_fragment", "5000"),
                ("lance.compaction.materialize_deletions_threshold", "2.0"),
            ])
            .await
            .unwrap();

        // Build options from the dataset config (as the bindings do)
        let opts = CompactionOptions::from_dataset_config(&dataset.manifest.config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 5000);
        assert!((opts.materialize_deletions_threshold - 2.0).abs() < f32::EPSILON);

        // Verify the config flows through plan_compaction
        let plan = plan_compaction(&dataset, &opts).await.unwrap();
        assert!(!plan.tasks.is_empty());
        assert_eq!(plan.options.target_rows_per_fragment, 5000);
        // validate() should have turned off materialize_deletions since threshold >= 1.0
        assert!(!plan.options.materialize_deletions);
    }

    // check_rewrite_txn takes the (None, Some(_)) branch when a Rewrite with
    // defer_index_remap=true is committed against a previously committed
    // CreateIndex, declaring COMPATIBLE without verifying that the Rewrite's
    // FRI groups don't straddle the CreateIndex's fragment bitmap. When a
    // group mixes indexed and unindexed fragments, commit succeeds and later
    // queries fail at load_indices with "split of indexed and non-indexed
    // data".
    #[tokio::test]
    async fn test_rewrite_fri_vs_create_index_conflict() {
        use crate::index::DatasetIndexExt;
        use crate::index::vector::VectorIndexParams;
        use futures::TryStreamExt;
        use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
        use lance_index::IndexType;
        use lance_linalg::distance::MetricType;

        async fn append_fragment(uri: &str, rows: u64) -> Dataset {
            let reader = gen_batch()
                .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
                .into_reader_rows(RowCount::from(rows), BatchCount::from(1));
            let params = WriteParams {
                max_rows_per_file: rows as usize,
                mode: WriteMode::Append,
                ..Default::default()
            };
            Dataset::write(reader, uri, Some(params)).await.unwrap()
        }

        let tmpdir = TempStrDir::default();
        let uri = format!("file://{}", tmpdir.as_str());

        // frag0 (256 rows) with a base IVF index.
        let reader = gen_batch()
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
            .into_reader_rows(RowCount::from(256), BatchCount::from(1));
        let mut dataset = Dataset::write(
            reader,
            &uri,
            Some(WriteParams {
                max_rows_per_file: 256,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let index_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 50);
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &index_params, true)
            .await
            .unwrap();

        // Append frag1 (unindexed), snapshot a stale handle pointing here,
        // then append frag2 (also unindexed).
        dataset = append_fragment(&uri, 64).await;
        let mut stale = dataset.clone();
        dataset = append_fragment(&uri, 64).await;

        // Plan + execute compaction of frag1+frag2 with deferred remap.
        let options = CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert!(!plan.tasks.is_empty());
        let snapshot = dataset.clone();
        let completed: Vec<RewriteResult> = futures::stream::iter(plan.tasks.into_iter())
            .map(|task| rewrite_files(Cow::Borrowed(&snapshot), task, &options))
            .buffer_unordered(1)
            .try_collect()
            .await
            .unwrap();

        // optimize_indices on the stale handle indexes frag1 only (frag2
        // didn't exist at that version), commits as CreateIndex. `dataset`
        // stays at its pre-optimize version so the Rewrite commit has to
        // conflict-check against this CreateIndex.
        stale
            .optimize_indices(&lance_index::optimize::OptimizeOptions::append())
            .await
            .unwrap();

        // Commit the pre-executed Rewrite. The FRI group [frag1, frag2]
        // straddles the new CreateIndex bitmap (frag1 indexed, frag2 not), so
        // check_rewrite_txn must reject this as a retryable conflict rather
        // than letting it commit into a broken state that fails queries.
        let err = commit_compaction(
            &mut dataset,
            completed,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .expect_err("commit should fail with retryable conflict");
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "unexpected error: {err}"
        );
    }

    /// Reproduce the distributed-compaction concurrent-delete data-resurrection bug.
    ///
    /// In the distributed (Spark) path the caller opens **two separate** `Dataset` handles:
    ///
    /// 1. dataset_plan  — used for `plan_compaction` (version = V)
    /// 2. dataset_commit — opened **fresh** for `commit_compaction` (version = V+N)
    ///
    /// Because `commit_compaction` builds the `Rewrite` transaction with
    /// `dataset.manifest.version` (= V+N), `load_and_sort_new_transactions` only
    /// scans versions after V+N and finds nothing.  Any concurrent DELETE that
    /// happened between V and V+N is silently ignored, causing the deleted rows to
    /// reappear in the compacted fragment.
    ///
    /// After the fix, `commit_compaction` uses `min(tasks.read_version)` (= V) as
    /// the transaction `read_version`, so the conflict checker correctly loads and
    /// rejects the DELETE, returning a retryable conflict error instead of silently
    /// resurrecting data.
    #[tokio::test]
    async fn test_distributed_compact_concurrent_delete_no_resurrection() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Write 4 fragments × 1 000 rows each (a=0..4000).
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..4_000))],
        )
        .unwrap();
        let mut dataset_plan = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data)], schema.clone()),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 1_000,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset_plan.manifest.version, 1);
        assert_eq!(dataset_plan.get_fragments().len(), 4);

        // ── Step 1: plan compaction at version V=1 ───────────────────────────────
        let options = CompactionOptions {
            target_rows_per_fragment: 10_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset_plan, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 1, "expected one compaction task");

        // ── Step 2: execute tasks (simulating distributed executors at V=1) ──────
        // Clone dataset_plan so the closure can own its copy while the original
        // remains available for the concurrent DELETE in Step 3.
        let dataset_for_tasks = dataset_plan.clone();
        let results: Vec<RewriteResult> = futures::stream::iter(plan.compaction_tasks())
            .then(|task| {
                let ds = dataset_for_tasks.clone();
                async move {
                    // Executors open the dataset at the planned read_version
                    task.execute(&ds).await.unwrap()
                }
            })
            .collect()
            .await;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].read_version, 1,
            "tasks must carry read_version=1"
        );

        // ── Step 3: concurrent DELETE commits at V=2 ─────────────────────────────
        // Delete rows where a < 1000 (the first 1 000 rows in fragment 0).
        dataset_plan.delete("a < 1000").await.unwrap();
        assert_eq!(dataset_plan.manifest.version, 2);

        // ── Step 4: the Spark driver opens a *fresh* dataset (latest = V=2) ──────
        // This is exactly what OptimizeExec.scala does for commitCompaction.
        let mut dataset_commit = Dataset::open(test_uri).await.unwrap();
        assert_eq!(
            dataset_commit.manifest.version, 2,
            "fresh dataset must be at the post-delete version"
        );

        // ── Step 5: commit_compaction with the stale results ─────────────────────
        let commit_result = commit_compaction(
            &mut dataset_commit,
            results,
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await;

        // ── Step 6: assert correct behaviour ─────────────────────────────────────
        // BEFORE fix: commit_result is Ok(…) and the deleted rows are resurrected.
        // AFTER  fix: commit_result is Err(retryable conflict), protecting data integrity.
        assert!(
            commit_result.is_err(),
            "commit_compaction must fail with a conflict error when a concurrent \
             DELETE touched the same fragments; got Ok instead — deleted rows were \
             silently resurrected"
        );
        let err_msg = commit_result.unwrap_err().to_string();
        assert!(
            err_msg.contains("retryable")
                || err_msg.contains("conflict")
                || err_msg.contains("preempted"),
            "expected a retryable conflict error, got: {err_msg}"
        );

        // The on-disk table must still reflect the DELETE (a < 1000 remains absent).
        let latest = Dataset::open(test_uri).await.unwrap();
        let row_count = latest
            .count_rows(Some("a < 1000".to_string()))
            .await
            .unwrap();
        assert_eq!(
            row_count, 0,
            "rows deleted before compaction must not be resurrected; found {row_count}"
        );
    }

    fn count_all_files_in(dir: &std::path::Path) -> std::io::Result<usize> {
        if !dir.exists() {
            return Ok(0);
        }
        let mut count = 0;
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                count += count_all_files_in(&path)?;
            } else if path.is_file() {
                // Ignore macOS system files if any
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|file_name| !file_name.starts_with('.'))
                {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    fn count_data_files_in(base_dir: &str) -> usize {
        let data_dir = std::path::Path::new(base_dir).join("data");
        count_all_files_in(&data_dir).unwrap_or(0)
    }

    /// A non-conflict commit error can be ambiguous, so rewritten files must
    /// remain available until version-aware GC proves they are unreferenced.
    #[tokio::test]
    async fn test_commit_compaction_retains_data_on_ambiguous_commit_failure() {
        use crate::dataset::builder::DatasetBuilder;
        use crate::utils::test::FailingProxyStore;
        use lance_io::object_store::ObjectStoreParams;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        // Prefix `/` so Windows drive letters (e.g. `C:`) don't get parsed as
        // the URL authority.
        let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
        let routed_uri = format!("file-object-store://{path_prefix}{test_uri}");

        let data = sample_data();
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 200))], data.schema());
        Dataset::write(
            reader,
            &routed_uri,
            Some(WriteParams {
                max_rows_per_file: 100,
                // Stable row IDs lets `commit_compaction` skip the
                // `reserve_fragment_ids` pre-commit (which would otherwise fail
                // *before* the new data files exist), isolating the failure to
                // the `apply_commit` call we want to test.
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let baseline_files = count_data_files_in(test_uri);

        let failing = Arc::new(FailingProxyStore::new());
        // The injected transaction-write error does not carry a proof that the
        // manifest was not committed, so immediate deletion is unsafe.
        failing.fail_after_n("put", "_transactions", 1, "injected commit failure");
        failing.fail_after_n(
            "put_multipart",
            "_transactions",
            1,
            "injected commit failure",
        );

        let mut dataset = DatasetBuilder::from_uri(&routed_uri)
            .with_read_params(crate::dataset::ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        };
        let result = compact_files(&mut dataset, options, None).await;
        assert!(
            result.is_err(),
            "Compaction should fail when transaction commit fails"
        );

        assert!(
            count_data_files_in(test_uri) > baseline_files,
            "ambiguous commit failures must retain rewritten data for version-aware GC"
        );
    }

    #[tokio::test]
    async fn test_commit_compaction_retains_blob_v2_sidecars_on_ambiguous_commit_failure() {
        use crate::BlobArrayBuilder;
        use crate::dataset::builder::DatasetBuilder;
        use crate::utils::test::FailingProxyStore;
        use lance_io::object_store::ObjectStoreParams;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
        let routed_uri = format!("file-object-store://{path_prefix}{test_uri}");

        let id_array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        // Use one packed blob and one dedicated blob to verify both sidecar layouts.
        let packed_data = vec![0u8; 100 * 1024];
        let dedicated_data = vec![1u8; 5 * 1024 * 1024];
        let mut blob_builder = BlobArrayBuilder::new(2);
        blob_builder.push_bytes(&packed_data).unwrap();
        blob_builder.push_bytes(&dedicated_data).unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        Dataset::write(
            reader,
            &routed_uri,
            Some(WriteParams {
                max_rows_per_file: 1, // Create 2 fragments
                enable_stable_row_ids: true,
                data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let baseline_files = count_data_files_in(test_uri);

        let failing = Arc::new(FailingProxyStore::new());
        failing.fail_after_n("put", "_transactions", 1, "injected commit failure");
        failing.fail_after_n(
            "put_multipart",
            "_transactions",
            1,
            "injected commit failure",
        );

        let mut dataset = DatasetBuilder::from_uri(&routed_uri)
            .with_read_params(crate::dataset::ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        };
        let result = compact_files(&mut dataset, options, None).await;
        assert!(
            result.is_err(),
            "Compaction should fail when transaction commit fails"
        );

        assert!(
            count_data_files_in(test_uri) > baseline_files,
            "ambiguous commit failures must retain blob v2 sidecars for version-aware GC"
        );
    }

    async fn read_blob_bytes_by_index(
        dataset: &Arc<Dataset>,
        column: &str,
    ) -> Vec<(i32, Option<Vec<u8>>)> {
        let mut scanner = dataset.scan();
        scanner.with_row_id();
        let batch = scanner
            .project(&["id", column])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>();

        let mut result = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let row_id = row_ids.value(i);
            let id = ids.value(i);
            let blobs = dataset.take_blobs(&[row_id], column).await.unwrap();
            if blobs.is_empty() {
                result.push((id, None));
            } else {
                let data = blobs[0].read().await.unwrap();
                result.push((id, Some(data.to_vec())));
            }
        }
        result
    }

    #[tokio::test]
    async fn test_compact_blob_v2_preserves_external_references() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;
        use lance_table::format::BasePath;

        let test_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        std::fs::write(&external_path, b"external-data").unwrap();
        let external_uri = format!("file://{}", external_path.display());
        let base_uri = format!("file://{}", external_dir.std_path().display());

        let mut blob_builder = BlobArrayBuilder::new(2);
        blob_builder.push_uri(external_uri.clone()).unwrap();
        blob_builder.push_bytes(b"inline-data").unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("external".to_string()),
                    path: base_uri,
                    is_dataset_root: false,
                }]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        for frag in dataset.get_fragments() {
            let rows = frag.physical_rows().await.unwrap();
            assert!(rows > 0, "fragment {} should have rows", frag.id());
        }

        let options = CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert!(
            !plan.tasks().is_empty(),
            "compaction plan should have tasks, got {} tasks",
            plan.tasks().len()
        );

        compact_files(&mut dataset, options, None).await.unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let scan_result = dataset
            .scan()
            .project(&["id", "blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(scan_result.num_rows(), 2);

        let ids = scan_result
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let mut id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
        id_values.sort();
        assert_eq!(id_values, vec![0, 1]);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(b"external-data".to_vec())),
                (1, Some(b"inline-data".to_vec()))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_packed_and_dedicated() {
        use crate::BlobArrayBuilder;
        use lance_arrow::BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY;
        use lance_core::utils::tempfile::TempDir;

        let test_dir = TempDir::default();

        let inline_data = b"small-inline-blob".as_slice();
        let packed_data: Vec<u8> = (0..64 * 1024 + 1024).map(|i| (i % 256) as u8).collect();
        let dedicated_data: Vec<u8> = (0..4 * 1024 * 1024 + 512)
            .map(|i| ((i + 97) % 256) as u8)
            .collect();

        let mut blob_builder = BlobArrayBuilder::new(3);
        blob_builder.push_bytes(inline_data).unwrap();
        blob_builder.push_bytes(&packed_data).unwrap();
        blob_builder.push_bytes(&dedicated_data).unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2]));
        let mut blob_field = crate::blob_field("blob", true);
        {
            let metadata = blob_field.metadata().clone();
            let mut new_metadata = metadata;
            new_metadata.insert(
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                (4 * 1024 * 1024).to_string(),
            );
            blob_field = blob_field.with_metadata(new_metadata);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            blob_field,
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 3);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let scan_result = dataset
            .scan()
            .project(&["id", "blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(scan_result.num_rows(), 3);

        let ids = scan_result
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
        assert_eq!(id_values, vec![0, 1, 2]);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(inline_data.to_vec())),
                (1, Some(packed_data)),
                (2, Some(dedicated_data))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_with_null_rows() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;

        let test_dir = TempDir::default();

        let mut blob_builder = BlobArrayBuilder::new(4);
        blob_builder.push_bytes(b"inline-0").unwrap();
        blob_builder.push_null().unwrap();
        blob_builder.push_bytes(b"inline-2").unwrap();
        blob_builder.push_null().unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef =
            Arc::new(Int32Array::from(vec![Some(0), Some(1), Some(2), Some(3)]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let scan_result = dataset
            .scan()
            .project(&["id", "blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(scan_result.num_rows(), 4);

        let ids = scan_result
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
        assert_eq!(id_values, vec![0, 1, 2, 3]);

        let blob_col = scan_result.column_by_name("blob").unwrap();
        assert!(
            matches!(blob_col.data_type(), DataType::Struct(_)),
            "blob column should be a struct after compaction"
        );

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(b"inline-0".to_vec())),
                (1, None),
                (2, Some(b"inline-2".to_vec())),
                (3, None)
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_deleted_rows_not_resurrected() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;

        let test_dir = TempDir::default();

        let mut blob_builder = BlobArrayBuilder::new(4);
        blob_builder.push_bytes(b"blob-0").unwrap();
        blob_builder.push_bytes(b"blob-1").unwrap();
        blob_builder.push_bytes(b"blob-2").unwrap();
        blob_builder.push_bytes(b"blob-3").unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        dataset.delete("id = 1").await.unwrap();
        dataset.delete("id = 2").await.unwrap();

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                materialize_deletions_threshold: 0.0,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        let scan_result = dataset
            .scan()
            .project(&["id", "blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(scan_result.num_rows(), 2);

        let ids = scan_result
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let mut id_values: Vec<i32> = ids.iter().map(|v| v.unwrap()).collect();
        id_values.sort();
        assert_eq!(id_values, vec![0, 3]);

        let blob_col = scan_result.column_by_name("blob").unwrap();
        let struct_arr = blob_col.as_any().downcast_ref::<StructArray>().unwrap();
        let kind_col = struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<UInt8Type>();

        for i in 0..kind_col.len() {
            assert!(
                !kind_col.is_null(i),
                "row {} should have a non-null kind after compaction of deleted rows",
                i
            );
        }

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![(0, Some(b"blob-0".to_vec())), (3, Some(b"blob-3".to_vec()))]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_external_and_data_blob_mixed() {
        use crate::BlobArrayBuilder;
        use lance_arrow::BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY;
        use lance_core::utils::tempfile::TempDir;
        use lance_table::format::BasePath;

        let test_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        std::fs::write(&external_path, b"external-payload").unwrap();
        let external_uri = format!("file://{}", external_path.display());
        let base_uri = format!("file://{}", external_dir.std_path().display());

        let packed_data: Vec<u8> = (0..64 * 1024 + 512).map(|i| (i % 256) as u8).collect();

        let mut blob_builder = BlobArrayBuilder::new(4);
        blob_builder.push_uri(external_uri.clone()).unwrap();
        blob_builder.push_bytes(&packed_data).unwrap();
        blob_builder.push_bytes(b"inline-small").unwrap();
        blob_builder.push_uri(external_uri.clone()).unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
        let mut blob_field = crate::blob_field("blob", true);
        {
            let mut new_metadata = blob_field.metadata().clone();
            new_metadata.insert(
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                (4 * 1024 * 1024).to_string(),
            );
            blob_field = blob_field.with_metadata(new_metadata);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            blob_field,
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("external".to_string()),
                    path: base_uri,
                    is_dataset_root: false,
                }]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(b"external-payload".to_vec())),
                (1, Some(packed_data)),
                (2, Some(b"inline-small".to_vec())),
                (3, Some(b"external-payload".to_vec()))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_multiple_blob_columns() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;

        let test_dir = TempDir::default();

        let mut image_builder = BlobArrayBuilder::new(3);
        image_builder.push_bytes(b"image-0").unwrap();
        image_builder.push_bytes(b"image-1").unwrap();
        image_builder.push_bytes(b"image-2").unwrap();
        let image_array: ArrayRef = image_builder.finish().unwrap();

        let mut thumb_builder = BlobArrayBuilder::new(3);
        thumb_builder.push_bytes(b"thumb-0").unwrap();
        thumb_builder.push_null().unwrap();
        thumb_builder.push_bytes(b"thumb-2").unwrap();
        let thumb_array: ArrayRef = thumb_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("image", true),
            crate::blob_field("thumbnail", true),
        ]));

        let batch =
            RecordBatch::try_new(schema.clone(), vec![id_array, image_array, thumb_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 3);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut image_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "image").await;
        image_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            image_values,
            vec![
                (0, Some(b"image-0".to_vec())),
                (1, Some(b"image-1".to_vec())),
                (2, Some(b"image-2".to_vec()))
            ]
        );

        let mut thumb_values =
            read_blob_bytes_by_index(&Arc::new(dataset.clone()), "thumbnail").await;
        thumb_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            thumb_values,
            vec![
                (0, Some(b"thumb-0".to_vec())),
                (1, None),
                (2, Some(b"thumb-2".to_vec()))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_external_and_null_mixed() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;
        use lance_table::format::BasePath;

        let test_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("mixed-external.bin");
        std::fs::write(&external_path, b"external-mixed-data").unwrap();
        let external_uri = format!("file://{}", external_path.display());
        let base_uri = format!("file://{}", external_dir.std_path().display());

        let mut blob_builder = BlobArrayBuilder::new(4);
        blob_builder.push_uri(external_uri.clone()).unwrap();
        blob_builder.push_null().unwrap();
        blob_builder.push_uri(external_uri.clone()).unwrap();
        blob_builder.push_null().unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("external".to_string()),
                    path: base_uri,
                    is_dataset_root: false,
                }]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(b"external-mixed-data".to_vec())),
                (1, None),
                (2, Some(b"external-mixed-data".to_vec())),
                (3, None)
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_all_null_and_all_external_fragments() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;
        use lance_table::format::BasePath;

        let test_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("all-ext.bin");
        std::fs::write(&external_path, b"all-external-data").unwrap();
        let external_uri = format!("file://{}", external_path.display());
        let base_uri = format!("file://{}", external_dir.std_path().display());

        let mut null_builder = BlobArrayBuilder::new(2);
        null_builder.push_null().unwrap();
        null_builder.push_null().unwrap();
        let null_array: ArrayRef = null_builder.finish().unwrap();

        let mut ext_builder = BlobArrayBuilder::new(2);
        ext_builder.push_uri(external_uri.clone()).unwrap();
        ext_builder.push_uri(external_uri.clone()).unwrap();
        let ext_array: ArrayRef = ext_builder.finish().unwrap();

        let id_null_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1]));
        let null_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));
        let null_batch =
            RecordBatch::try_new(null_schema.clone(), vec![id_null_array, null_array]).unwrap();

        let id_ext_array: ArrayRef = Arc::new(Int32Array::from(vec![2, 3]));
        let ext_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));
        let ext_batch =
            RecordBatch::try_new(ext_schema.clone(), vec![id_ext_array, ext_array]).unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new(
                vec![null_batch, ext_batch].into_iter().map(Ok),
                null_schema.clone(),
            ),
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("external".to_string()),
                    path: base_uri,
                    is_dataset_root: false,
                }]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, None),
                (1, None),
                (2, Some(b"all-external-data".to_vec())),
                (3, Some(b"all-external-data".to_vec()))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_external_with_multiple_base_ids() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;
        use lance_table::format::BasePath;

        let test_dir = TempDir::default();
        let base_a_dir = TempDir::default();
        let base_b_dir = TempDir::default();

        let path_a = base_a_dir.std_path().join("data-a.bin");
        std::fs::write(&path_a, b"from-base-a").unwrap();
        let uri_a = format!("file://{}", path_a.display());
        let base_uri_a = format!("file://{}", base_a_dir.std_path().display());

        let path_b = base_b_dir.std_path().join("data-b.bin");
        std::fs::write(&path_b, b"from-base-b").unwrap();
        let uri_b = format!("file://{}", path_b.display());
        let base_uri_b = format!("file://{}", base_b_dir.std_path().display());

        let mut blob_builder = BlobArrayBuilder::new(4);
        blob_builder.push_uri(uri_a.clone()).unwrap();
        blob_builder.push_uri(uri_b).unwrap();
        blob_builder.push_bytes(b"inline-data").unwrap();
        blob_builder.push_uri(uri_a).unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base_a".to_string()),
                        path: base_uri_a,
                        is_dataset_root: false,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base_b".to_string()),
                        path: base_uri_b,
                        is_dataset_root: false,
                    },
                ]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(b"from-base-a".to_vec())),
                (1, Some(b"from-base-b".to_vec())),
                (2, Some(b"inline-data".to_vec())),
                (3, Some(b"from-base-a".to_vec()))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_large_blobs() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;

        let test_dir = TempDir::default();

        let large_blob_a: Vec<u8> = (0..512 * 1024).map(|i| (i % 256) as u8).collect();
        let large_blob_b: Vec<u8> = (0..256 * 1024).map(|i| ((i + 42) % 256) as u8).collect();

        let mut blob_builder = BlobArrayBuilder::new(3);
        blob_builder.push_bytes(&large_blob_a).unwrap();
        blob_builder.push_bytes(&large_blob_b).unwrap();
        blob_builder.push_bytes(b"small-blob").unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 3);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(large_blob_a)),
                (1, Some(large_blob_b)),
                (2, Some(b"small-blob".to_vec()))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_blob_kind_reclassification() {
        use crate::BlobArrayBuilder;
        use lance_arrow::BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY;
        use lance_core::utils::tempfile::TempDir;

        let test_dir = TempDir::default();

        let medium_data: Vec<u8> = (0..32 * 1024).map(|i| (i % 256) as u8).collect();

        let mut blob_builder = BlobArrayBuilder::new(2);
        blob_builder.push_bytes(&medium_data).unwrap();
        blob_builder.push_bytes(&medium_data).unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1]));
        let mut blob_field = crate::blob_field("blob", true);
        {
            let mut new_metadata = blob_field.metadata().clone();
            new_metadata.insert(
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                (16 * 1024).to_string(),
            );
            blob_field = blob_field.with_metadata(new_metadata);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            blob_field,
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(medium_data.clone())),
                (1, Some(medium_data.clone()))
            ]
        );
    }

    #[tokio::test]
    async fn test_compact_blob_v2_multi_batch() {
        use crate::BlobArrayBuilder;
        use lance_core::utils::tempfile::TempDir;

        let test_dir = TempDir::default();

        let mut blob_builder = BlobArrayBuilder::new(6);
        blob_builder.push_bytes(b"batch-0-row-0").unwrap();
        blob_builder.push_bytes(b"batch-0-row-1").unwrap();
        blob_builder.push_bytes(b"batch-1-row-0").unwrap();
        blob_builder.push_null().unwrap();
        blob_builder.push_bytes(b"batch-1-row-2").unwrap();
        blob_builder.push_bytes(b"batch-1-row-3").unwrap();
        let blob_array: ArrayRef = blob_builder.finish().unwrap();

        let id_array: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            crate::blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 3);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 1024 * 1024,
                batch_size: Some(2),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), 1);

        let mut blob_values = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
        blob_values.sort_by_key(|(id, _)| *id);
        assert_eq!(
            blob_values,
            vec![
                (0, Some(b"batch-0-row-0".to_vec())),
                (1, Some(b"batch-0-row-1".to_vec())),
                (2, Some(b"batch-1-row-0".to_vec())),
                (3, None),
                (4, Some(b"batch-1-row-2".to_vec())),
                (5, Some(b"batch-1-row-3".to_vec()))
            ]
        );
    }
}
