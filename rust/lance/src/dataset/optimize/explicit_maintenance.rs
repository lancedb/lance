// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashMap};
use std::io::{Seek, SeekFrom};
use std::sync::Arc;

use arrow::compute::SortOptions;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_expr::LexOrdering;
use futures::TryStreamExt;
use lance_core::datatypes::BlobHandling;
use lance_core::utils::address::{LogicalRowAddress, RowAddress};
use lance_core::utils::tempfile::TempDir;
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_datafusion::exec::execute_plan;
use lance_file::version::LanceFileVersion;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_table::format::{
    ExplicitMapDestination, ExplicitMapPage, ExplicitMapRowAddressPlacement, ExplicitMapRowIdPage,
    Fragment, LogicalRowAddressRange, LogicalRowAddressSelection, ReplacedContentGeneration,
    RowAddressLayoutDelta, RowAddressLogicalDomain, RowAddressPlacementDelta,
    RowAddressPlacementKind, RowAddressSourceFloor, RowAddressTargetFragment,
    RowAddressTargetRange, RowDatasetVersionMeta, RowSequenceFingerprintBuilder,
    SparseSelectionSource, fingerprint_explicit_map_u64_page,
};
use object_store::path::Path;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::super::Dataset;
use super::super::explicit_row_addresses::EXPLICIT_ROW_ADDRESS_PAGE_ROWS;
use super::super::scanner::ColumnOrdering;
use super::super::transaction::RowAddressManifestApplyContext;
use super::super::transaction::{
    Operation, RewriteGroup, TransactionBuilder, generation_region_can_retire,
    with_strict_full_ordered_rewrite_property,
};
use super::super::utils::{CapturedRowIds, make_rowid_capture_stream};
use super::super::write::{
    WriteMode, WriteParams, cleanup_data_fragments, write_fragments_internal,
};
use super::logical_row_addresses::{add_rewrite_provenance, retired_logical_row_ids};
use crate::index::DatasetIndexExt;
use crate::io::deletion::read_dataset_deletion_file;

const EXPLICIT_ROW_ADDRESS_DIR: &str = "data/_row_addresses";

fn explicit_object_path(base: &Path, relative_path: &str) -> Path {
    relative_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .fold(base.clone(), |path, segment| path.join(segment))
}

/// An explicit storage-version-2.3 row-address maintenance operation.
#[derive(Debug, Clone)]
pub enum RowAddressMaintenanceMode {
    /// Rewrite in logical-address order and require an inline fast-path layout.
    NormalizePlacement,
    /// Cluster in the requested order when the resulting source-contiguous
    /// extents fit the inline fast-path layout budget.
    BoundedRecluster { ordering: Vec<ColumnOrdering> },
    /// Rewrite in logical-address order and externalize the permutation.
    Repack,
    /// Rewrite in the requested global order and externalize the permutation.
    Recluster { ordering: Vec<ColumnOrdering> },
    /// Retire generation regions already consumed by every current index.
    CheckpointGeneration,
}

/// Options for [`maintain_row_addresses`].
#[derive(Debug, Clone)]
pub struct RowAddressMaintenanceOptions {
    /// Maintenance behavior to execute.
    pub mode: RowAddressMaintenanceMode,
    /// Maximum rows written to one destination fragment.
    pub target_rows_per_fragment: usize,
    /// Maximum rows in one writer batch group.
    pub max_rows_per_group: usize,
    /// Optional physical file-size limit.  This is rejected by inline
    /// maintenance because exact preflight requires deterministic row-count
    /// boundaries.
    pub max_bytes_per_file: Option<usize>,
    /// Optional scan batch size.
    pub batch_size: Option<usize>,
    /// Optional input scan I/O buffer size.
    pub io_buffer_size: Option<u64>,
    /// Optional properties recorded on the maintenance transaction.
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
}

impl RowAddressMaintenanceOptions {
    /// Create options for inline placement normalization.
    pub fn normalize_placement() -> Self {
        Self {
            mode: RowAddressMaintenanceMode::NormalizePlacement,
            ..Self::default()
        }
    }

    /// Create options for admitted source-contiguous clustering.
    pub fn bounded_recluster(ordering: Vec<ColumnOrdering>) -> Self {
        Self {
            mode: RowAddressMaintenanceMode::BoundedRecluster { ordering },
            ..Self::default()
        }
    }

    /// Create options for logical-order packing with an external locator.
    pub fn repack() -> Self {
        Self {
            mode: RowAddressMaintenanceMode::Repack,
            ..Self::default()
        }
    }

    /// Create options for arbitrary physical reordering with an external locator.
    pub fn recluster(ordering: Vec<ColumnOrdering>) -> Self {
        Self {
            mode: RowAddressMaintenanceMode::Recluster { ordering },
            ..Self::default()
        }
    }

    /// Create options for retiring generation metadata consumed by current indices.
    pub fn checkpoint_generation() -> Self {
        Self {
            mode: RowAddressMaintenanceMode::CheckpointGeneration,
            ..Self::default()
        }
    }
}

impl Default for RowAddressMaintenanceOptions {
    fn default() -> Self {
        Self {
            mode: RowAddressMaintenanceMode::NormalizePlacement,
            target_rows_per_fragment: 1_000_000,
            max_rows_per_group: 1024,
            max_bytes_per_file: None,
            batch_size: None,
            io_buffer_size: None,
            transaction_properties: None,
        }
    }
}

/// Observable cost of explicit row-address maintenance.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowAddressMaintenanceMetrics {
    /// Physical fragments replaced by the maintenance commit.
    pub fragments_removed: usize,
    /// Physical fragments created by the maintenance commit.
    pub fragments_added: usize,
    /// User-data files written by the rewrite.
    pub data_files_written: usize,
    /// Locator plus hidden `_rowid` objects written by explicit mapping modes.
    pub locator_objects_written: usize,
    /// Total bytes in locator and hidden `_rowid` objects.
    pub locator_bytes_written: u64,
    /// Live rows physically rewritten.
    pub rows_rewritten: u64,
}

fn validate_options(dataset: &Dataset, options: &RowAddressMaintenanceOptions) -> Result<()> {
    if !dataset.manifest.uses_stable_logical_row_addresses() {
        return Err(Error::invalid_input(
            "row-address maintenance requires storage version 2.3",
        ));
    }
    if options.target_rows_per_fragment == 0 || options.max_rows_per_group == 0 {
        return Err(Error::invalid_input(
            "row-address maintenance row limits must be greater than zero",
        ));
    }
    if matches!(
        options.mode,
        RowAddressMaintenanceMode::NormalizePlacement
            | RowAddressMaintenanceMode::BoundedRecluster { .. }
    ) && options.max_bytes_per_file.is_some()
    {
        return Err(Error::invalid_input(
            "inline row-address maintenance requires row-count-only output boundaries so exact Delta/W preflight can finish before remote writes; max_bytes_per_file is not supported",
        ));
    }
    if let RowAddressMaintenanceMode::BoundedRecluster { ordering }
    | RowAddressMaintenanceMode::Recluster { ordering } = &options.mode
    {
        if ordering.is_empty() {
            return Err(Error::invalid_input(
                "clustering requires at least one ordering column",
            ));
        }
        for column in ordering {
            if column.column_name == ROW_ID || column.column_name == ROW_ADDR {
                return Err(Error::invalid_input(
                    "clustering ordering must use user columns; _rowid is appended as a deterministic tie-breaker",
                ));
            }
            if dataset.schema().field(&column.column_name).is_none() {
                return Err(Error::invalid_input(format!(
                    "Recluster ordering column {} does not exist",
                    column.column_name
                )));
            }
        }
    }
    Ok(())
}

async fn sorted_maintenance_stream(
    dataset: &Dataset,
    options: &RowAddressMaintenanceOptions,
) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
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
    scanner.with_row_id().scan_in_order(false);
    let mut plan = scanner.create_plan().await?;
    let mut ordering = match &options.mode {
        RowAddressMaintenanceMode::BoundedRecluster { ordering }
        | RowAddressMaintenanceMode::Recluster { ordering } => ordering
            .iter()
            .map(|column| {
                Ok(PhysicalSortExpr {
                    expr: expressions::col(&column.column_name, plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: !column.ascending,
                        nulls_first: column.nulls_first,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?,
        _ => Vec::new(),
    };
    ordering.push(PhysicalSortExpr {
        expr: expressions::col(ROW_ID, plan.schema().as_ref())?,
        options: SortOptions {
            descending: false,
            nulls_first: false,
        },
    });
    plan = Arc::new(SortExec::new(
        LexOrdering::new(ordering)
            .ok_or_else(|| Error::internal("maintenance ordering cannot be empty"))?,
        plan,
    ));
    let stream = execute_plan(plan, scanner.execution_options())?;
    if has_blob_v2_columns {
        Ok(super::transform_blob_v2_stream(dataset, stream))
    } else {
        Ok(stream)
    }
}

fn planned_fragments(row_count: usize, rows_per_fragment: usize) -> Vec<Fragment> {
    let mut remaining = row_count;
    let mut fragments = Vec::with_capacity(row_count.div_ceil(rows_per_fragment));
    while remaining > 0 {
        let rows = remaining.min(rows_per_fragment);
        fragments.push(Fragment::new(0).with_physical_rows(rows));
        remaining -= rows;
    }
    fragments
}

async fn spool_normalize_stream(
    mut stream: datafusion::physical_plan::SendableRecordBatchStream,
) -> Result<(
    datafusion::physical_plan::SendableRecordBatchStream,
    Vec<u64>,
)> {
    let input_schema = stream.schema();
    let (row_id_index, _) = input_schema
        .column_with_name(ROW_ID)
        .ok_or_else(|| Error::internal("NormalizePlacement sorted stream is missing _rowid"))?;
    let projection = (0..input_schema.fields().len())
        .filter(|index| *index != row_id_index)
        .collect::<Vec<_>>();
    let output_schema = Arc::new(input_schema.project(&projection)?);
    let temp_dir = TempDir::try_new()?;
    let spool_path = temp_dir.std_path().join("normalize-placement.arrow");
    let mut spool = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&spool_path)?;
    let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut spool, &input_schema)?;
    let mut row_ids = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        let ids = batch
            .column(row_id_index)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::internal("NormalizePlacement _rowid is not UInt64"))?;
        if ids.null_count() != 0 {
            return Err(Error::internal("NormalizePlacement _rowid contains nulls"));
        }
        row_ids.extend_from_slice(ids.values());
        writer.write(&batch)?;
    }
    writer.finish()?;
    drop(writer);
    spool.seek(SeekFrom::Start(0))?;
    let reader = arrow_ipc::reader::FileReader::try_new(spool, None)?;
    let replay = futures::stream::unfold(
        (reader, temp_dir, projection),
        |(mut reader, temp_dir, projection)| async move {
            let batch = reader.next()?;
            let batch = batch
                .map_err(datafusion::error::DataFusionError::from)
                .and_then(|batch| {
                    batch
                        .project(&projection)
                        .map_err(datafusion::error::DataFusionError::from)
                });
            Some((batch, (reader, temp_dir, projection)))
        },
    );
    Ok((
        Box::pin(RecordBatchStreamAdapter::new(output_schema, replay)),
        row_ids,
    ))
}

async fn normalize_delta(
    dataset: &Dataset,
    fragments: &[Fragment],
    row_ids: &[u64],
) -> Result<RowAddressLayoutDelta> {
    let mut delta = RowAddressLayoutDelta {
        expected_layout_fingerprint: dataset
            .manifest
            .row_address_layout
            .as_ref()
            .expect("validated storage-version-2.3 layout")
            .fingerprint
            .clone(),
        ..RowAddressLayoutDelta::default()
    };
    let mut sequence = lance_table::rowids::RowIdSequence::new();
    sequence.extend(lance_table::rowids::RowIdSequence::from(row_ids));
    let serialized = lance_table::rowids::write_row_ids(&sequence);
    let retired = retired_logical_row_ids(dataset, &dataset.manifest.fragments).await?;
    let mut next_ordinal = 0;
    let mut source_domains = BTreeMap::new();
    add_rewrite_provenance(
        dataset,
        fragments,
        &serialized,
        retired.as_deref(),
        &mut next_ordinal,
        &mut source_domains,
        &mut delta,
    )?;
    delta.source_domains = source_domains.into_values().collect();
    Ok(delta)
}

fn append_bounded_output(
    router: &lance_table::format::RowAddressRouter,
    rows: &[u64],
    target: RowAddressTargetRange,
    source_domains: &mut BTreeMap<u32, RowAddressLogicalDomain>,
    delta: &mut RowAddressLayoutDelta,
) -> Result<()> {
    if rows.is_empty() {
        return Err(Error::invalid_input(
            "bounded clustering output run must not be empty",
        ));
    }
    if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::invalid_input(
            "bounded clustering output run must be in strict logical source order",
        ));
    }

    let mut ranges = Vec::<LogicalRowAddressRange>::new();
    let mut stats = BTreeMap::<u32, (u64, u32, u32)>::new();
    for raw in rows {
        let address = LogicalRowAddress::try_from(*raw)?;
        let logical_fragment_id = address.logical_fragment_id();
        let slot = address.immutable_slot();
        if let Some(previous) = ranges.last_mut()
            && previous.logical_fragment_id == logical_fragment_id
            && previous.end_slot == slot
        {
            previous.end_slot = slot
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("logical row-address slot overflow"))?;
        } else {
            ranges.push(LogicalRowAddressRange::new(
                logical_fragment_id,
                slot,
                slot.checked_add(1)
                    .ok_or_else(|| Error::invalid_input("logical row-address slot overflow"))?,
            ));
        }
        stats
            .entry(logical_fragment_id)
            .and_modify(|(count, first, last)| {
                *count += 1;
                *first = (*first).min(slot);
                *last = (*last).max(slot);
            })
            .or_insert((1, slot, slot));
    }

    let mut all_domains_full = true;
    for (logical_fragment_id, (count, first, last)) in &stats {
        let domain = router
            .logical_domain(*logical_fragment_id)?
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "bounded clustering emitted unknown logical fragment {logical_fragment_id}"
                ))
            })?;
        if source_domains
            .insert(*logical_fragment_id, domain)
            .is_some_and(|previous| previous != domain)
        {
            return Err(Error::invalid_input(format!(
                "logical fragment {logical_fragment_id} has inconsistent source metadata"
            )));
        }
        all_domains_full &= *count == domain.slot_count as u64
            && *first == 0
            && last.checked_add(1) == Some(domain.slot_count);
    }

    let selection = LogicalRowAddressSelection::from_ranges(ranges.clone())?;
    let mut fingerprint = RowSequenceFingerprintBuilder::new(target);
    for range in ranges {
        fingerprint.update_range(range)?;
    }
    let output_cardinality = rows.len() as u64;
    let placement_kind = if stats.len() > 1 && all_domains_full {
        RowAddressPlacementKind::PackedRun
    } else if stats.len() == 1 {
        RowAddressPlacementKind::Selected
    } else {
        RowAddressPlacementKind::SparseSelection
    };
    delta.placements.push(RowAddressPlacementDelta {
        source_selections: vec![selection],
        target,
        placement_kind,
        output_cardinality,
        output_row_sequence_fingerprint: fingerprint.finish(output_cardinality)?,
    });
    Ok(())
}

async fn bounded_recluster_delta(
    dataset: &Dataset,
    fragments: &[Fragment],
    row_ids: &[u64],
) -> Result<RowAddressLayoutDelta> {
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .expect("validated storage-version-2.3 layout");
    let router = dataset.row_address_router()?;
    let mut delta = RowAddressLayoutDelta {
        expected_layout_fingerprint: layout.fingerprint.clone(),
        ..RowAddressLayoutDelta::default()
    };
    let mut source_domains = BTreeMap::new();
    let mut cursor = 0_usize;
    for (ordinal, fragment) in fragments.iter().enumerate() {
        let row_count = fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input("bounded clustering output fragment is missing physical_rows")
        })?;
        let end = cursor
            .checked_add(row_count)
            .ok_or_else(|| Error::invalid_input("bounded clustering row count overflow"))?;
        let fragment_rows = row_ids.get(cursor..end).ok_or_else(|| {
            Error::invalid_input(
                "bounded clustering output fragments exceed the captured logical row sequence",
            )
        })?;
        let fragment_ordinal = u32::try_from(ordinal)
            .map_err(|_| Error::invalid_input("bounded clustering fragment ordinal exceeds u32"))?;
        let mut run_start = 0_usize;
        for run_end in 1..=fragment_rows.len() {
            let ends_run = run_end == fragment_rows.len()
                || fragment_rows[run_end] <= fragment_rows[run_end - 1];
            if !ends_run {
                continue;
            }
            let target_start = u32::try_from(run_start).map_err(|_| {
                Error::invalid_input("bounded clustering target offset exceeds u32")
            })?;
            let target_end = u32::try_from(run_end).map_err(|_| {
                Error::invalid_input("bounded clustering target offset exceeds u32")
            })?;
            append_bounded_output(
                &router,
                &fragment_rows[run_start..run_end],
                RowAddressTargetRange {
                    fragment: RowAddressTargetFragment::NewFragmentOrdinal(fragment_ordinal),
                    start_offset: target_start,
                    end_offset: target_end,
                },
                &mut source_domains,
                &mut delta,
            )?;
            run_start = run_end;
        }
        cursor = end;
    }
    if cursor != row_ids.len() {
        return Err(Error::invalid_input(
            "captured logical row sequence exceeds bounded clustering output fragments",
        ));
    }

    if let Some(retired_row_ids) =
        retired_logical_row_ids(dataset, &dataset.manifest.fragments).await?
    {
        let retired = lance_table::rowids::read_row_ids(&retired_row_ids)?;
        let mut bitmap = RoaringTreemap::new();
        for raw in retired.iter() {
            let address = LogicalRowAddress::try_from(raw)?;
            let domain = router
                .logical_domain(address.logical_fragment_id())?
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "bounded clustering retired unknown logical fragment {}",
                        address.logical_fragment_id()
                    ))
                })?;
            if source_domains
                .insert(address.logical_fragment_id(), domain)
                .is_some_and(|previous| previous != domain)
            {
                return Err(Error::invalid_input(format!(
                    "logical fragment {} has inconsistent source metadata",
                    address.logical_fragment_id()
                )));
            }
            bitmap.insert(raw);
        }
        if !bitmap.is_empty() {
            delta
                .retired_selections
                .push(LogicalRowAddressSelection::from_bitmap(bitmap)?);
        }
    }
    delta.source_domains = source_domains.into_values().collect();
    Ok(delta)
}

async fn preflight_inline_rewrite(
    dataset: &Dataset,
    planned: Vec<Fragment>,
    delta: RowAddressLayoutDelta,
) -> Result<()> {
    let mut context = RowAddressManifestApplyContext::default();
    // Normalize replaces the complete physical snapshot.  ExplicitMap
    // retirements need this fact because the format-only validator cannot
    // read the external locator to recover the deleted destination.
    for fragment in dataset.manifest.fragments.iter() {
        context
            .newly_fully_deleted_source_fragments
            .insert(u32::try_from(fragment.id).map_err(|_| {
                Error::invalid_input(format!(
                    "storage-version-2.3 fragment id {} exceeds row-address capacity",
                    fragment.id
                ))
            })?);
    }
    for fragment in dataset
        .manifest
        .fragments
        .iter()
        .filter(|fragment| fragment.deletion_file.is_some())
    {
        let fragment_id = u32::try_from(fragment.id)
            .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;
        let deletions = read_dataset_deletion_file(
            dataset,
            fragment.id,
            fragment.deletion_file.as_ref().expect("filtered above"),
        )
        .await?;
        context
            .current_deletion_vectors
            .insert(fragment_id, deletions.to_sorted_iter().collect());
    }
    let transaction = TransactionBuilder::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: dataset.manifest.fragments.as_ref().clone(),
                new_fragments: planned,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    )
    .row_address_layout_delta(Some(delta))
    .build();
    let indices = dataset.load_indices().await?;
    transaction.build_manifest_with_row_address_context(
        Some(dataset.manifest.as_ref()),
        indices.as_ref().clone(),
        "inline-row-address-rewrite-preflight.txn",
        &Default::default(),
        Some(&context),
    )?;
    Ok(())
}

fn assign_physical_fragment_ids(dataset: &Dataset, fragments: &mut [Fragment]) -> Result<()> {
    let next = dataset
        .manifest
        .max_fragment_id
        .map(|maximum| maximum as u64 + 1)
        .unwrap_or(0);
    for (next, fragment) in (next..).zip(fragments) {
        if next >= u32::MAX as u64 {
            return Err(Error::invalid_input(
                "physical fragment id space is exhausted",
            ));
        }
        fragment.id = next;
        fragment.native_logical_domain = None;
        fragment.row_id_meta = None;
    }
    Ok(())
}

async fn preserve_row_version_metadata(
    dataset: &Dataset,
    row_ids: &[u64],
    fragments: &mut [Fragment],
) -> Result<()> {
    let (created, updated) =
        super::super::rowids::resolve_logical_row_version_sequences(dataset, row_ids).await?;
    let chunk_sizes = fragments
        .iter()
        .map(|fragment| fragment.physical_rows.unwrap_or_default() as u64)
        .collect::<Vec<_>>();
    let created = lance_table::rowids::version::rechunk_version_sequences(
        [created],
        chunk_sizes.clone(),
        false,
    )?;
    let updated =
        lance_table::rowids::version::rechunk_version_sequences([updated], chunk_sizes, false)?;
    for ((fragment, created), updated) in fragments.iter_mut().zip(created).zip(updated) {
        fragment.created_at_version_meta = Some(RowDatasetVersionMeta::from_sequence(&created)?);
        fragment.last_updated_at_version_meta =
            Some(RowDatasetVersionMeta::from_sequence(&updated)?);
    }
    Ok(())
}

async fn write_u64_file(
    dataset: &Dataset,
    relative_path: &str,
    names: &[&str],
    columns: &[&[u64]],
) -> Result<(u64, Vec<ExplicitMapRowIdPage>)> {
    let row_count = columns.first().map_or(0, |column| column.len());
    if names.is_empty()
        || names.len() != columns.len()
        || columns.iter().any(|column| column.len() != row_count)
    {
        return Err(Error::invalid_input(
            "ExplicitMap file columns must be non-empty and have equal lengths",
        ));
    }
    let fields = names
        .iter()
        .map(|name| Field::new(*name, DataType::UInt64, false))
        .collect::<Vec<_>>();
    let schema = Arc::new(ArrowSchema::new(fields));
    let path = explicit_object_path(&dataset.base, relative_path);
    let mut writer = FileWriter::try_new(
        dataset.object_store.create(&path).await?,
        schema.as_ref().try_into()?,
        FileWriterOptions {
            format_version: Some(LanceFileVersion::V2_3),
            data_cache_bytes: Some(
                (EXPLICIT_ROW_ADDRESS_PAGE_ROWS * std::mem::size_of::<u64>() * names.len()) as u64,
            ),
            max_page_bytes: Some(
                (EXPLICIT_ROW_ADDRESS_PAGE_ROWS * std::mem::size_of::<u64>()) as u64,
            ),
            ..Default::default()
        },
    )?;
    let mut pages = Vec::with_capacity(row_count.div_ceil(EXPLICIT_ROW_ADDRESS_PAGE_ROWS));
    for row_start in (0..row_count).step_by(EXPLICIT_ROW_ADDRESS_PAGE_ROWS) {
        let row_end = (row_start + EXPLICIT_ROW_ADDRESS_PAGE_ROWS).min(row_count);
        let page_columns = columns
            .iter()
            .map(|column| &column[row_start..row_end])
            .collect::<Vec<_>>();
        pages.push(ExplicitMapRowIdPage {
            row_start: row_start as u64,
            row_count: (row_end - row_start) as u64,
            content_fingerprint: fingerprint_explicit_map_u64_page(&page_columns)?,
        });
        let arrays = columns
            .iter()
            .map(|column| {
                Arc::new(UInt64Array::from(column[row_start..row_end].to_vec())) as ArrayRef
            })
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema.clone(), arrays)?;
        writer.write_batch(&batch).await?;
    }
    Ok((writer.finish().await?.size_bytes, pages))
}

async fn cleanup_explicit_paths(dataset: &Dataset, paths: &[Path]) {
    for path in paths {
        let _ = dataset.object_store.delete(path).await;
    }
}

async fn write_explicit_map(
    dataset: &Dataset,
    domains: &[RowAddressLogicalDomain],
    row_ids: &[u64],
    fragments: &[Fragment],
) -> Result<(ExplicitMapRowAddressPlacement, Vec<Path>, u64)> {
    let operation_id = Uuid::new_v4();
    let mut created_paths = Vec::with_capacity(fragments.len() + 1);
    let result = async {
        let mut destinations = Vec::with_capacity(fragments.len());
        let mut locator = Vec::with_capacity(row_ids.len());
        let mut cursor = 0_usize;
        let mut total_bytes = 0_u64;
        for (ordinal, fragment) in fragments.iter().enumerate() {
            let row_count = fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input("maintenance output fragment is missing physical_rows")
            })?;
            let end = cursor + row_count;
            let fragment_row_ids = row_ids.get(cursor..end).ok_or_else(|| {
                Error::internal("maintenance output fragment exceeds captured row-id sequence")
            })?;
            let relative_path =
                format!("{EXPLICIT_ROW_ADDRESS_DIR}/{operation_id}-destination-{ordinal}.lance");
            created_paths.push(explicit_object_path(&dataset.base, &relative_path));
            let (size, row_id_pages) =
                write_u64_file(dataset, &relative_path, &[ROW_ID], &[fragment_row_ids]).await?;
            total_bytes += size;
            let physical_fragment_id = u32::try_from(fragment.id)
                .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;
            for (offset, row_id) in fragment_row_ids.iter().copied().enumerate() {
                let offset = u32::try_from(offset).map_err(|_| {
                    Error::invalid_input("ExplicitMap destination row offset exceeds u32")
                })?;
                locator.push((
                    row_id,
                    u64::from(RowAddress::new_from_parts(physical_fragment_id, offset)),
                ));
            }
            let row_count = u32::try_from(row_count)
                .map_err(|_| Error::invalid_input("ExplicitMap destination exceeds u32 rows"))?;
            destinations.push(ExplicitMapDestination {
                physical_fragment_id,
                destination_start: 0,
                row_count,
                row_id_file_path: relative_path,
                row_id_file_size: size,
                row_id_pages,
            });
            cursor = end;
        }
        if cursor != row_ids.len() {
            return Err(Error::internal(
                "captured row-id sequence exceeds maintenance output fragments",
            ));
        }
        locator.sort_unstable_by_key(|(logical, _)| *logical);
        if locator.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(Error::invalid_input(
                "maintenance output contains duplicate logical row ids",
            ));
        }
        let (logical, physical): (Vec<_>, Vec<_>) = locator.into_iter().unzip();
        if logical.is_empty() {
            return Err(Error::invalid_input(
                "ExplicitMap requires at least one live row",
            ));
        }
        let object_path = format!("{EXPLICIT_ROW_ADDRESS_DIR}/{operation_id}-locator.lance");
        created_paths.push(explicit_object_path(&dataset.base, &object_path));
        let (object_size, locator_content_pages) = write_u64_file(
            dataset,
            &object_path,
            &[ROW_ID, ROW_ADDR],
            &[logical.as_slice(), physical.as_slice()],
        )
        .await?;
        let pages = logical
            .chunks(EXPLICIT_ROW_ADDRESS_PAGE_ROWS)
            .zip(locator_content_pages)
            .map(|(rows, content)| ExplicitMapPage {
                first_logical_address: rows[0],
                last_logical_address: rows[rows.len() - 1],
                row_start: content.row_start,
                row_count: content.row_count,
                content_fingerprint: content.content_fingerprint,
            })
            .collect::<Vec<_>>();
        total_bytes += object_size;
        if domains.is_empty() {
            return Err(Error::invalid_input("ExplicitMap requires source domains"));
        }
        let sources = domains
            .iter()
            .copied()
            .map(|source| {
                Ok(SparseSelectionSource {
                    source,
                    selection: Arc::new(LogicalRowAddressSelection::from_full_domains(&[source])?),
                    excluded: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((
            ExplicitMapRowAddressPlacement {
                sources,
                object_path,
                object_size,
                pages,
                destinations,
                base_id: None,
            },
            total_bytes,
        ))
    }
    .await;
    match result {
        Ok((placement, total_bytes)) => Ok((placement, created_paths, total_bytes)),
        Err(error) => {
            cleanup_explicit_paths(dataset, &created_paths).await;
            Err(error)
        }
    }
}

fn current_domains(dataset: &Dataset) -> Result<Vec<RowAddressLogicalDomain>> {
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| Error::internal("storage-version-2.3 manifest has no row-address layout"))?;
    let router = dataset.row_address_router()?;
    layout
        .current_logical_fragment_bitmap(&dataset.manifest.fragments)?
        .into_iter()
        .map(|logical_fragment_id| {
            router.logical_domain(logical_fragment_id)?.ok_or_else(|| {
                Error::internal(format!(
                    "logical fragment {logical_fragment_id} is missing domain metadata"
                ))
            })
        })
        .collect()
}

fn explicit_delta(
    dataset: &Dataset,
    domains: Vec<RowAddressLogicalDomain>,
    placement: ExplicitMapRowAddressPlacement,
    row_ids: &[u64],
) -> Result<RowAddressLayoutDelta> {
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| Error::internal("storage-version-2.3 manifest has no row-address layout"))?;
    let first_destination = placement
        .destinations
        .first()
        .ok_or_else(|| Error::invalid_input("ExplicitMap has no destinations"))?;
    let target = RowAddressTargetRange {
        fragment: RowAddressTargetFragment::ExistingFragmentId(
            first_destination.physical_fragment_id,
        ),
        start_offset: first_destination.destination_start,
        end_offset: first_destination.destination_start + first_destination.row_count,
    };
    let mut fingerprint = RowSequenceFingerprintBuilder::new(target);
    for row_id in row_ids {
        fingerprint.update(LogicalRowAddress::try_from(*row_id)?)?;
    }
    let mut domain_rows = RoaringTreemap::new();
    for domain in &domains {
        let start = u64::from(domain.logical_fragment_id) << 32;
        domain_rows.insert_range(start..start + u64::from(domain.slot_count));
    }
    let live_rows = row_ids.iter().copied().collect::<RoaringTreemap>();
    if !live_rows.is_subset(&domain_rows) {
        return Err(Error::invalid_input(
            "ExplicitMap output contains a row outside its source domains",
        ));
    }
    let mut replacement_retirements = domain_rows;
    replacement_retirements -= live_rows;
    let source_selection = LogicalRowAddressSelection::from_full_domains(&domains)?;
    let mut delta = RowAddressLayoutDelta {
        source_domains: domains,
        placements: vec![RowAddressPlacementDelta {
            source_selections: vec![source_selection],
            target,
            placement_kind: RowAddressPlacementKind::ExplicitMap,
            output_cardinality: row_ids.len() as u64,
            output_row_sequence_fingerprint: fingerprint.finish(row_ids.len() as u64)?,
        }],
        expected_layout_fingerprint: layout.fingerprint.clone(),
        explicit_map_placements: BTreeMap::from([(0, placement)]),
        ..RowAddressLayoutDelta::default()
    };
    if !replacement_retirements.is_empty() {
        delta
            .retired_selections
            .push(LogicalRowAddressSelection::from_bitmap(
                replacement_retirements,
            )?);
    }
    Ok(delta)
}

async fn generation_checkpoint_delta(dataset: &Dataset) -> Result<RowAddressLayoutDelta> {
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| Error::internal("storage-version-2.3 manifest has no row-address layout"))?;
    let indices = dataset.load_indices().await?;
    let mut replaced = Vec::new();
    for region in &layout.generation_regions {
        if generation_region_can_retire(region, indices.as_ref())? {
            replaced.push(ReplacedContentGeneration {
                selection: region.selection.as_ref().clone(),
                field_ids: region.field_ids.clone(),
                generation: region.generation,
            });
        }
    }
    Ok(RowAddressLayoutDelta {
        source_floors: layout
            .index_commit_floors
            .iter()
            .map(|floor| RowAddressSourceFloor {
                field_id: floor.field_id,
                generation: floor.generation,
            })
            .collect(),
        expected_layout_fingerprint: layout.fingerprint.clone(),
        replaced_generations: replaced,
        ..RowAddressLayoutDelta::default()
    })
}

async fn commit_checkpoint(
    dataset: &mut Dataset,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
) -> Result<RowAddressMaintenanceMetrics> {
    let delta = generation_checkpoint_delta(dataset).await?;
    let transaction = TransactionBuilder::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: Vec::new(),
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    )
    .row_address_layout_delta(Some(delta))
    .transaction_properties(transaction_properties)
    .build();
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;
    Ok(RowAddressMaintenanceMetrics::default())
}

/// Execute one explicit storage-version-2.3 row-address maintenance cycle.
///
/// Inline maintenance performs an exact metadata admission preflight before
/// opening remote data writers. `Repack` and `Recluster` intentionally publish
/// an external locator and expose its read and write cost.
///
/// ```no_run
/// # use lance::{Dataset, Result};
/// # use lance::dataset::optimize::{
/// #     maintain_row_addresses, RowAddressMaintenanceOptions,
/// # };
/// # async fn normalize(dataset: &mut Dataset) -> Result<()> {
/// let metrics = maintain_row_addresses(
///     dataset,
///     RowAddressMaintenanceOptions::normalize_placement(),
/// )
/// .await?;
/// println!("rewrote {} rows", metrics.rows_rewritten);
/// # Ok(())
/// # }
/// ```
pub async fn maintain_row_addresses(
    dataset: &mut Dataset,
    options: RowAddressMaintenanceOptions,
) -> Result<RowAddressMaintenanceMetrics> {
    validate_options(dataset, &options)?;
    if matches!(
        &options.mode,
        RowAddressMaintenanceMode::CheckpointGeneration
    ) {
        return commit_checkpoint(dataset, options.transaction_properties.clone()).await;
    }
    if dataset.manifest.fragments.is_empty() {
        return Ok(RowAddressMaintenanceMetrics::default());
    }
    let inline_preflight = matches!(
        &options.mode,
        RowAddressMaintenanceMode::NormalizePlacement
            | RowAddressMaintenanceMode::BoundedRecluster { .. }
    );
    let old_fragments = dataset.manifest.fragments.as_ref().clone();
    let sorted_stream = sorted_maintenance_stream(dataset, &options).await?;
    let mut row_ids_rx = None;
    let mut preflight_row_ids = None;
    let mut preflight_delta = None;
    let mut planned_counts = None;
    let stream = if inline_preflight {
        let (replay, row_ids) = spool_normalize_stream(sorted_stream).await?;
        let planned = planned_fragments(row_ids.len(), options.target_rows_per_fragment);
        let delta = match &options.mode {
            RowAddressMaintenanceMode::NormalizePlacement => {
                normalize_delta(dataset, &planned, &row_ids).await?
            }
            RowAddressMaintenanceMode::BoundedRecluster { .. } => {
                bounded_recluster_delta(dataset, &planned, &row_ids).await?
            }
            _ => unreachable!("inline preflight mode changed after validation"),
        };
        preflight_inline_rewrite(dataset, planned.clone(), delta.clone()).await?;
        planned_counts = Some(
            planned
                .iter()
                .map(|fragment| fragment.physical_rows.unwrap_or_default())
                .collect::<Vec<_>>(),
        );
        preflight_row_ids = Some(row_ids);
        preflight_delta = Some(delta);
        replay
    } else {
        let (stream, receiver) = make_rowid_capture_stream(sorted_stream, true)?;
        row_ids_rx = Some(receiver);
        stream
    };
    let mut write_params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        allow_external_blob_outside_bases: true,
        ..Default::default()
    };
    if let Some(max_bytes_per_file) = options.max_bytes_per_file {
        write_params.max_bytes_per_file = max_bytes_per_file;
    } else if inline_preflight {
        // Exact preflight plans row-count boundaries.  Byte-driven splitting
        // would make the target ranges unknowable before the remote write.
        write_params.max_bytes_per_file = usize::MAX;
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
    if matches!(
        &options.mode,
        RowAddressMaintenanceMode::Repack | RowAddressMaintenanceMode::Recluster { .. }
    ) {
        if let Err(error) = assign_physical_fragment_ids(dataset, &mut new_fragments) {
            cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                .await;
            return Err(error);
        }
    } else {
        for fragment in &mut new_fragments {
            fragment.id = 0;
            fragment.native_logical_domain = None;
            fragment.row_id_meta = None;
        }
    }
    if let Some(planned_counts) = planned_counts
        && new_fragments
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap_or_default())
            .collect::<Vec<_>>()
            != planned_counts
    {
        cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments).await;
        return Err(Error::internal(
            "inline row-address maintenance writer did not honor its preflight row boundaries",
        ));
    }
    let row_ids = if let Some(row_ids) = preflight_row_ids {
        row_ids
    } else {
        let captured = match row_ids_rx
            .expect("non-normalize maintenance has a capture receiver")
            .try_recv()
        {
            Ok(captured) => captured,
            Err(error) => {
                cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                    .await;
                return Err(Error::internal(format!(
                    "maintenance row-id capture did not complete after data write: {error}"
                )));
            }
        };
        let CapturedRowIds::SequenceStyle(sequence) = captured else {
            cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                .await;
            return Err(Error::internal(
                "storage-version-2.3 maintenance captured physical row addresses",
            ));
        };
        sequence.iter().collect::<Vec<_>>()
    };
    if let Err(error) = preserve_row_version_metadata(dataset, &row_ids, &mut new_fragments).await {
        cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments).await;
        return Err(error);
    }

    let mut extra_paths = Vec::new();
    let mut locator_bytes = 0_u64;
    let transaction_properties = if matches!(
        &options.mode,
        RowAddressMaintenanceMode::BoundedRecluster { .. }
            | RowAddressMaintenanceMode::Recluster { .. }
    ) {
        Some(with_strict_full_ordered_rewrite_property(
            options.transaction_properties.clone(),
        ))
    } else {
        options.transaction_properties.clone()
    };
    let delta_result = match options.mode {
        RowAddressMaintenanceMode::NormalizePlacement
        | RowAddressMaintenanceMode::BoundedRecluster { .. } => {
            Ok(preflight_delta.expect("inline maintenance delta was preflighted"))
        }
        RowAddressMaintenanceMode::Repack | RowAddressMaintenanceMode::Recluster { .. } => {
            match current_domains(dataset) {
                Ok(domains) => {
                    match write_explicit_map(dataset, &domains, &row_ids, &new_fragments).await {
                        Ok((placement, paths, bytes)) => {
                            extra_paths = paths;
                            locator_bytes = bytes;
                            explicit_delta(dataset, domains, placement, &row_ids)
                        }
                        Err(error) => Err(error),
                    }
                }
                Err(error) => Err(error),
            }
        }
        RowAddressMaintenanceMode::CheckpointGeneration => unreachable!(),
    };
    let delta = match delta_result {
        Ok(delta) => delta,
        Err(error) => {
            cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                .await;
            cleanup_explicit_paths(dataset, &extra_paths).await;
            return Err(error);
        }
    };

    let transaction = TransactionBuilder::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: old_fragments.clone(),
                new_fragments: new_fragments.clone(),
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    )
    .row_address_layout_delta(Some(delta))
    .transaction_properties(transaction_properties)
    .build();
    if let Err(error) = dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
    {
        // Retryable conflicts prove this attempt was not committed. Other
        // errors can be ambiguous (for example, the manifest PUT succeeded but
        // the response and reconciliation read both failed), so leave their
        // immutable objects for version-aware GC instead of risking deletion
        // of files referenced by a committed snapshot.
        if matches!(error, Error::RetryableCommitConflict { .. }) {
            cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &new_fragments)
                .await;
            cleanup_explicit_paths(dataset, &extra_paths).await;
        }
        return Err(error);
    }

    Ok(RowAddressMaintenanceMetrics {
        fragments_removed: old_fragments.len(),
        fragments_added: new_fragments.len(),
        data_files_written: new_fragments
            .iter()
            .map(|fragment| fragment.files.len())
            .sum(),
        locator_objects_written: if locator_bytes == 0 {
            0
        } else {
            new_fragments.len() + 1
        },
        locator_bytes_written: locator_bytes,
        rows_rewritten: row_ids.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
    use lance_index::IndexType;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_table::format::{
        ContentGenerationRegion, FieldGeneration, IndexMetadata, LogicalIndexCoverage,
        LogicalIndexCoverageShard, Manifest, RowAddressPlacement, Transaction,
    };
    use lance_table::io::commit::{
        CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    };
    use mock_instant::thread_local::MockClock;
    use object_store::path::Path;

    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{ProjectionRequest, UpdateBuilder, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::io::ObjectStore;
    use crate::{BlobArrayBuilder, blob_field};

    use super::*;

    async fn write_dataset(uri: &str, values: Vec<i32>) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(values))]).unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 64,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn scan_i_and_row_ids(dataset: &Dataset) -> Vec<(i32, u64)> {
        let mut scanner = dataset.scan();
        scanner.project(&["i", ROW_ID]).unwrap().scan_in_order(true);
        let batch = scanner.try_into_batch().await.unwrap();
        let values = batch["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let row_ids = batch[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        values
            .values()
            .iter()
            .copied()
            .zip(row_ids.values().iter().copied())
            .collect()
    }

    async fn scan_row_versions(dataset: &Dataset) -> HashMap<i32, (u64, u64, u64)> {
        let mut scanner = dataset.scan();
        scanner
            .project(&[
                "i",
                ROW_ID,
                ROW_CREATED_AT_VERSION,
                ROW_LAST_UPDATED_AT_VERSION,
            ])
            .unwrap()
            .scan_in_order(true);
        let batch = scanner.try_into_batch().await.unwrap();
        let values = batch["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let row_ids = batch[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let created = batch[ROW_CREATED_AT_VERSION]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let updated = batch[ROW_LAST_UPDATED_AT_VERSION]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        (0..batch.num_rows())
            .map(|row| {
                (
                    values.value(row),
                    (row_ids.value(row), created.value(row), updated.value(row)),
                )
            })
            .collect()
    }

    async fn write_versioned_v2_3_dataset(uri: &str) -> (Dataset, HashMap<i32, (u64, u64, u64)>) {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let initial = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..4))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let appended = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(4..8))],
        )
        .unwrap();
        dataset
            .append(
                RecordBatchIterator::new([Ok(appended)], schema),
                Some(WriteParams {
                    max_rows_per_file: 2,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert_eq!(dataset.version_id(), 2);
        let after_append = scan_row_versions(&dataset).await;
        for value in 0..4 {
            assert_eq!(after_append[&value].1, 1);
            assert_eq!(after_append[&value].2, 1);
        }
        for value in 4..8 {
            assert_eq!(after_append[&value].1, 2);
            assert_eq!(after_append[&value].2, 2);
        }
        let updated_row_id = after_append[&1].0;

        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i = 1")
            .unwrap()
            .set("i", "101")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let dataset = Arc::unwrap_or_clone(updated.new_dataset);
        assert_eq!(dataset.version_id(), 3);
        let expected = scan_row_versions(&dataset).await;
        assert_eq!(expected[&101], (updated_row_id, 1, 3));
        for value in [0, 2, 3] {
            assert_eq!(expected[&value].1, 1);
            assert_eq!(expected[&value].2, 1);
        }
        for value in 4..8 {
            assert_eq!(expected[&value].1, 2);
            assert_eq!(expected[&value].2, 2);
        }
        (dataset, expected)
    }

    #[tokio::test]
    async fn v2_3_row_versions_survive_compaction_repack_and_normalize() {
        let temp = TempStrDir::default();
        let (mut dataset, expected) = write_versioned_v2_3_dataset(temp.as_str()).await;

        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 16,
                max_rows_per_group: 16,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(metrics.fragments_removed > 0);
        assert_eq!(scan_row_versions(&dataset).await, expected);

        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 16,
                max_rows_per_group: 16,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap();
        assert!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .iter()
                .any(|placement| matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
        assert_eq!(scan_row_versions(&dataset).await, expected);

        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 16,
                max_rows_per_group: 16,
                ..RowAddressMaintenanceOptions::normalize_placement()
            },
        )
        .await
        .unwrap();
        assert_eq!(scan_row_versions(&dataset).await, expected);
        assert!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .iter()
                .all(|placement| !matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
    }

    #[tokio::test]
    async fn v2_3_row_versions_survive_recluster() {
        let temp = TempStrDir::default();
        let (mut dataset, expected) = write_versioned_v2_3_dataset(temp.as_str()).await;
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 3,
                max_rows_per_group: 3,
                ..RowAddressMaintenanceOptions::recluster(vec![ColumnOrdering::desc_nulls_last(
                    "i".to_owned(),
                )])
            },
        )
        .await
        .unwrap();
        assert!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .iter()
                .any(|placement| matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
        assert_eq!(scan_row_versions(&dataset).await, expected);
    }

    async fn write_blob_v2_dataset(uri: &str) -> Dataset {
        let inline = b"inline".to_vec();
        let packed = (0..70 * 1024)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let dedicated = (0..160 * 1024)
            .map(|index| ((index + 73) % 251) as u8)
            .collect::<Vec<_>>();
        let mut blobs = BlobArrayBuilder::new(4);
        blobs.push_bytes(&inline).unwrap();
        blobs.push_bytes(&packed).unwrap();
        blobs.push_bytes(&dedicated).unwrap();
        blobs.push_null().unwrap();
        let blob_array: ArrayRef = blobs.finish().unwrap();

        let mut blob = blob_field("blob", true);
        let mut metadata = blob.metadata().clone();
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_owned(),
            (128 * 1024).to_string(),
        );
        blob = blob.with_metadata(metadata);
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            blob,
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![2, 0, 3, 1])), blob_array],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn scan_blob_v2_rows(dataset: &Dataset) -> Vec<(i32, u64, Option<Vec<u8>>)> {
        let mut scanner = dataset.scan();
        scanner.project(&["i", ROW_ID]).unwrap().scan_in_order(true);
        let batch = scanner.try_into_batch().await.unwrap();
        let ids = batch["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let row_ids = batch[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let dataset = Arc::new(dataset.clone());
        let mut rows = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let row_id = row_ids.value(row);
            let blobs = dataset.take_blobs(&[row_id], "blob").await.unwrap();
            let value = if let Some(blob) = blobs.first() {
                Some(blob.read().await.unwrap().to_vec())
            } else {
                None
            };
            rows.push((ids.value(row), row_id, value));
        }
        rows
    }

    async fn assert_blob_v2_maintenance_roundtrip(
        uri: &str,
        options: RowAddressMaintenanceOptions,
        expected_order: &[i32],
    ) {
        let mut dataset = write_blob_v2_dataset(uri).await;
        let mut expected = scan_blob_v2_rows(&dataset).await;
        expected.sort_by_key(|(id, _, _)| *id);

        let metrics = maintain_row_addresses(&mut dataset, options).await.unwrap();
        assert_eq!(metrics.fragments_removed, 4);
        assert_eq!(metrics.fragments_added, 2);
        dataset.validate().await.unwrap();

        let rows = scan_blob_v2_rows(&dataset).await;
        assert_eq!(
            rows.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
            expected_order
        );
        let mut actual = rows;
        actual.sort_by_key(|(id, _, _)| *id);
        assert_eq!(actual, expected);

        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await
            .unwrap()
            .delete_unverified(true)
            .build();
        cleanup_old_versions(&dataset, policy).await.unwrap();
        drop(dataset);

        let reopened = Dataset::open(uri).await.unwrap();
        reopened.validate().await.unwrap();
        let mut reopened_rows = scan_blob_v2_rows(&reopened).await;
        reopened_rows.sort_by_key(|(id, _, _)| *id);
        assert_eq!(reopened_rows, expected);
    }

    #[tokio::test]
    async fn normalize_placement_roundtrips_blob_v2_storage_classes_and_null() {
        let temp = TempStrDir::default();
        assert_blob_v2_maintenance_roundtrip(
            &temp,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 2,
                max_rows_per_group: 2,
                ..RowAddressMaintenanceOptions::normalize_placement()
            },
            &[2, 0, 3, 1],
        )
        .await;
    }

    #[tokio::test]
    async fn repack_roundtrips_blob_v2_storage_classes_and_null() {
        let temp = TempStrDir::default();
        assert_blob_v2_maintenance_roundtrip(
            &temp,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 2,
                max_rows_per_group: 2,
                ..RowAddressMaintenanceOptions::repack()
            },
            &[2, 0, 3, 1],
        )
        .await;
    }

    #[tokio::test]
    async fn recluster_roundtrips_blob_v2_storage_classes_and_null() {
        let temp = TempStrDir::default();
        assert_blob_v2_maintenance_roundtrip(
            &temp,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 2,
                max_rows_per_group: 2,
                ..RowAddressMaintenanceOptions::recluster(vec![ColumnOrdering::desc_nulls_last(
                    "i".to_owned(),
                )])
            },
            &[3, 2, 1, 0],
        )
        .await;
    }

    async fn repacked_source(uri: &str) -> (Dataset, Vec<(i32, u64)>) {
        let mut dataset = write_dataset(uri, (0..20).collect()).await;
        dataset.delete("i % 2 = 0").await.unwrap();
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 3,
                max_rows_per_group: 3,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap();
        let expected = scan_i_and_row_ids(&dataset).await;
        (dataset, expected)
    }

    async fn assert_explicit_clone_roundtrip(dataset: &mut Dataset, expected: &[(i32, u64)]) {
        dataset.manifest.validate_row_address_contract().unwrap();
        assert_eq!(scan_i_and_row_ids(dataset).await, expected);
        let (expected_value, expected_row_id) = expected[3];
        let taken = dataset
            .take_rows(
                &[expected_row_id],
                ProjectionRequest::from_columns(["i", ROW_ID], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            taken["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            expected_value
        );
        assert_eq!(
            taken[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            expected_row_id
        );

        maintain_row_addresses(dataset, RowAddressMaintenanceOptions::normalize_placement())
            .await
            .unwrap();
        assert_eq!(scan_i_and_row_ids(dataset).await, expected);
        assert!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .iter()
                .all(|placement| !matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
    }

    fn coverage_index(
        name: &str,
        selection: LogicalRowAddressSelection,
        generation: u64,
    ) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![7],
            name: name.to_owned(),
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            row_reference_domain: None,
            logical_coverage: Some(
                LogicalIndexCoverage::new_exact(vec![
                    LogicalIndexCoverageShard::new_exact(
                        selection,
                        vec![7],
                        vec![FieldGeneration {
                            field_id: 7,
                            generation,
                        }],
                    )
                    .unwrap(),
                ])
                .unwrap(),
            ),
        }
    }

    #[test]
    fn checkpoint_requires_every_covering_index_to_be_fresh() {
        let domain = RowAddressLogicalDomain::new(0, 1, 1).unwrap();
        let selection = LogicalRowAddressSelection::from_full_domains(&[domain]).unwrap();
        let region = ContentGenerationRegion {
            selection: Arc::new(selection.clone()),
            field_ids: vec![7],
            generation: 2,
        };
        let fresh = coverage_index("fresh", selection.clone(), 2);
        let stale = coverage_index("stale", selection, 1);
        assert!(!generation_region_can_retire(&region, &[fresh.clone(), stale]).unwrap());

        let disjoint_domain = RowAddressLogicalDomain::new(1, 1, 1).unwrap();
        let disjoint = coverage_index(
            "disjoint-stale",
            LogicalRowAddressSelection::from_full_domains(&[disjoint_domain]).unwrap(),
            1,
        );
        assert!(generation_region_can_retire(&region, &[fresh, disjoint]).unwrap());
    }

    #[tokio::test]
    async fn repack_then_update_preserves_one_logical_owner() {
        let temp = TempStrDir::default();
        let (mut dataset, before) = repacked_source(&temp).await;
        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                Some("i_idx".to_owned()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        let row_id = before
            .iter()
            .find_map(|(value, row_id)| (*value == 3).then_some(*row_id))
            .unwrap();

        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i = 3")
            .unwrap()
            .set("i", "103")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut dataset = Arc::unwrap_or_clone(updated.new_dataset);
        dataset.manifest.validate_row_address_contract().unwrap();

        let rows = scan_i_and_row_ids(&dataset).await;
        assert_eq!(rows.len(), before.len());
        assert_eq!(
            rows.iter()
                .map(|(_, row_id)| *row_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            rows.len()
        );
        assert!(rows.contains(&(103, row_id)));
        assert!(!rows.iter().any(|(value, _)| *value == 3));

        let taken = dataset
            .take_rows(
                &[row_id],
                ProjectionRequest::from_columns(["i", ROW_ID], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            taken["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            103
        );
        assert_eq!(
            taken[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            row_id
        );

        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                Some("i_idx".to_owned()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                true,
            )
            .await
            .unwrap();
        let indexed = dataset
            .scan()
            .project(&["i", ROW_ID])
            .unwrap()
            .filter("i = 103")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut flat_scan = dataset.scan();
        flat_scan.use_scalar_index(false);
        flat_scan.project(&["i", ROW_ID]).unwrap();
        flat_scan.filter("i = 103").unwrap();
        let flat = flat_scan.try_into_batch().await.unwrap();
        assert_eq!(indexed.num_rows(), 1);
        assert_eq!(
            indexed["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            flat["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
        );
        assert_eq!(
            indexed[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values(),
            flat[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
        );

        let target_fragments = vec![dataset.fragments()[0].clone()];
        assert!(dataset.fragments().len() > 1);
        let mut indexed_not = dataset.scan();
        indexed_not
            .with_fragments(target_fragments.clone())
            .project(&[ROW_ID])
            .unwrap()
            .filter("NOT (i = 103)")
            .unwrap();
        let indexed_not = indexed_not.try_into_batch().await.unwrap();
        let mut flat_not = dataset.scan();
        flat_not
            .with_fragments(target_fragments)
            .use_scalar_index(false)
            .project(&[ROW_ID])
            .unwrap()
            .filter("NOT (i = 103)")
            .unwrap();
        let flat_not = flat_not.try_into_batch().await.unwrap();
        let mut indexed_row_ids = indexed_not[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        let mut flat_row_ids = flat_not[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        indexed_row_ids.sort_unstable();
        flat_row_ids.sort_unstable();
        assert!(!flat_row_ids.is_empty());
        assert_eq!(indexed_row_ids, flat_row_ids);

        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions::normalize_placement(),
        )
        .await
        .unwrap();
        dataset.manifest.validate_row_address_contract().unwrap();
        let mut normalized_rows = scan_i_and_row_ids(&dataset).await;
        normalized_rows.sort_unstable_by_key(|(_, row_id)| *row_id);
        let mut expected_rows = rows;
        expected_rows.sort_unstable_by_key(|(_, row_id)| *row_id);
        assert_eq!(normalized_rows, expected_rows);
        assert!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .iter()
                .all(|placement| !matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
    }

    #[tokio::test]
    async fn repack_random_delete_multi_fragment_roundtrip_and_index_fallback() {
        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, (0..20).collect()).await;
        let original = scan_i_and_row_ids(&dataset).await;
        let row_id_7 = original.iter().find(|(value, _)| *value == 7).unwrap().1;
        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                Some("i_idx".to_owned()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        dataset.delete("i % 2 = 0").await.unwrap();

        let metrics = maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 3,
                max_rows_per_group: 3,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap();
        assert!(metrics.fragments_added > 1);
        let layout = dataset.manifest.row_address_layout.as_ref().unwrap();
        let retired = layout.retired_logical_row_bitmap().unwrap();
        assert_eq!(
            retired.iter().collect::<Vec<_>>(),
            original
                .iter()
                .filter_map(|(value, row_id)| (value % 2 == 0).then_some(*row_id))
                .collect::<Vec<_>>()
        );
        let RowAddressPlacement::ExplicitMap(explicit) = &layout.placements[0] else {
            panic!("Repack must publish ExplicitMap")
        };
        assert_eq!(explicit.destinations.len(), metrics.fragments_added);
        assert_eq!(explicit.pages.len(), 3);
        assert_eq!(
            explicit
                .pages
                .iter()
                .map(|page| page.row_count)
                .sum::<u64>(),
            10
        );
        assert_eq!(explicit.pages[0].row_start, 0);
        assert_eq!(explicit.pages[1].row_start, explicit.pages[0].row_count);

        let scanned = scan_i_and_row_ids(&dataset).await;
        assert_eq!(
            scanned.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
            (0..20).filter(|value| value % 2 == 1).collect::<Vec<_>>()
        );
        assert_eq!(
            scanned.iter().find(|(value, _)| *value == 7).unwrap().1,
            row_id_7
        );
        let deleted_row_id = original.iter().find(|(value, _)| *value == 8).unwrap().1;
        assert_eq!(
            dataset
                .resolve_logical_row_ids_async(&[deleted_row_id])
                .await
                .unwrap(),
            vec![None]
        );
        let taken = dataset
            .take_rows(
                &[row_id_7],
                ProjectionRequest::from_columns(["i", ROW_ID], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            taken["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            7
        );
        assert_eq!(
            taken[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            row_id_7
        );

        let mut indexed = dataset.scan();
        indexed
            .project(&["i", ROW_ID])
            .unwrap()
            .filter("i = 7")
            .unwrap();
        let indexed = indexed.try_into_batch().await.unwrap();
        assert_eq!(indexed.num_rows(), 1);
        assert_eq!(
            indexed[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            row_id_7
        );

        // A partial deletion on an ExplicitMap destination must be translated
        // through the hidden destination row-id page when normalization retires
        // it.  This is the path that cannot use the inline inverse router.
        dataset.delete("i = 9").await.unwrap();
        let version_before_normalize = dataset.version_id();
        let normalize_metrics = maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions::normalize_placement(),
        )
        .await
        .unwrap();
        assert_eq!(normalize_metrics.rows_rewritten, 9);
        assert_eq!(dataset.version_id(), version_before_normalize + 1);
        assert!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .iter()
                .all(|placement| !matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
        let normalized = scan_i_and_row_ids(&dataset).await;
        assert_eq!(
            normalized
                .iter()
                .map(|(value, _)| *value)
                .collect::<Vec<_>>(),
            (0..20)
                .filter(|value| value % 2 == 1 && *value != 9)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            normalized.iter().find(|(value, _)| *value == 7).unwrap().1,
            row_id_7
        );
    }

    #[tokio::test]
    async fn repack_preserves_fully_retired_logical_domain_root() {
        let temp = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            temp.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 5);
        let original = scan_i_and_row_ids(&dataset).await;
        let original_ids = original.iter().copied().collect::<HashMap<_, _>>();

        let compaction = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                max_rows_per_group: 20,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(compaction.fragments_removed, 5);
        assert_eq!(compaction.fragments_added, 1);
        assert!(matches!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .as_slice(),
            [RowAddressPlacement::PackedRun(_)]
        ));

        let deleted = dataset.delete("i >= 4 AND i < 8").await.unwrap();
        assert_eq!(deleted.num_deleted_rows, 4);
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 16,
                max_rows_per_group: 16,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();
        dataset.manifest.validate_row_address_contract().unwrap();

        let layout = dataset.manifest.row_address_layout.as_ref().unwrap();
        let RowAddressPlacement::ExplicitMap(explicit) = &layout.placements[0] else {
            panic!("Repack must publish ExplicitMap")
        };
        assert_eq!(explicit.sources.len(), 5);
        assert_eq!(
            layout
                .retired_logical_row_bitmap()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            (4..8).map(|value| original_ids[&value]).collect::<Vec<_>>()
        );
        assert_eq!(
            scan_i_and_row_ids(&dataset).await,
            (0..4)
                .chain(8..20)
                .map(|value| (value, original_ids[&value]))
                .collect::<Vec<_>>()
        );
        assert!(
            dataset
                .resolve_logical_row_ids_async(
                    &(4..8).map(|value| original_ids[&value]).collect::<Vec<_>>()
                )
                .await
                .unwrap()
                .iter()
                .all(Option::is_none)
        );
        let taken = dataset
            .take_rows(
                &[original_ids[&10]],
                ProjectionRequest::from_columns(["i", ROW_ID], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            taken["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            10
        );
        assert_eq!(
            taken[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            original_ids[&10]
        );

        drop(dataset);
        let reopened = Dataset::open(temp.as_str()).await.unwrap();
        reopened.validate().await.unwrap();
        assert_eq!(
            scan_i_and_row_ids(&reopened).await,
            (0..4)
                .chain(8..20)
                .map(|value| (value, original_ids[&value]))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn shallow_clone_rebinds_explicit_map_objects_to_source_base() {
        let source_uri = TempStrDir::default();
        let clone_uri = TempStrDir::default();
        let (mut source, expected) = repacked_source(&source_uri).await;
        let cloned = source
            .shallow_clone(clone_uri.as_str(), source.version_id(), None)
            .await
            .unwrap();
        let explicit = cloned
            .manifest
            .row_address_layout
            .as_ref()
            .unwrap()
            .placements
            .iter()
            .find_map(|placement| match placement {
                RowAddressPlacement::ExplicitMap(explicit) => Some(explicit),
                _ => None,
            })
            .unwrap();
        let base_id = explicit
            .base_id
            .expect("shallow clone must externalize ownership");
        assert_eq!(cloned.manifest.base_paths[&base_id].path, source.uri());

        drop(cloned);
        let mut reopened = Dataset::open(clone_uri.as_str()).await.unwrap();
        assert_explicit_clone_roundtrip(&mut reopened, &expected).await;
    }

    #[tokio::test]
    async fn deep_clone_copies_and_localizes_explicit_map_objects() {
        let source_uri = TempStrDir::default();
        let clone_uri = TempStrDir::default();
        let (mut source, expected) = repacked_source(&source_uri).await;
        let cloned = source
            .deep_clone(clone_uri.as_str(), source.version_id(), None)
            .await
            .unwrap();
        let explicit = cloned
            .manifest
            .row_address_layout
            .as_ref()
            .unwrap()
            .placements
            .iter()
            .find_map(|placement| match placement {
                RowAddressPlacement::ExplicitMap(explicit) => Some(explicit.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(explicit.base_id, None);
        let paths = std::iter::once(explicit.object_path.as_str()).chain(
            explicit
                .destinations
                .iter()
                .map(|destination| destination.row_id_file_path.as_str()),
        );
        for relative_path in paths {
            cloned
                .object_store
                .open(&explicit_object_path(&cloned.base, relative_path))
                .await
                .unwrap();
        }

        drop(cloned);
        let mut reopened = Dataset::open(clone_uri.as_str()).await.unwrap();
        assert_explicit_clone_roundtrip(&mut reopened, &expected).await;
    }

    #[tokio::test]
    async fn bounded_recluster_preserves_identity_without_explicit_locator() {
        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, vec![2, 3, 0, 1, 6, 7, 4, 5]).await;
        let expected_ids = scan_i_and_row_ids(&dataset)
            .await
            .into_iter()
            .collect::<HashMap<_, _>>();

        let metrics = maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 8,
                max_rows_per_group: 8,
                ..RowAddressMaintenanceOptions::bounded_recluster(vec![
                    ColumnOrdering::asc_nulls_last("i".to_owned()),
                ])
            },
        )
        .await
        .unwrap();

        assert_eq!(metrics.locator_objects_written, 0);
        assert_eq!(metrics.locator_bytes_written, 0);
        let layout = dataset.manifest.row_address_layout.as_ref().unwrap();
        assert!(
            layout
                .placements
                .iter()
                .all(|placement| !matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
        let scanned = scan_i_and_row_ids(&dataset).await;
        assert_eq!(
            scanned.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        for (value, row_id) in scanned {
            assert_eq!(expected_ids[&value], row_id);
        }
    }

    #[tokio::test]
    async fn bounded_recluster_admits_monotonic_selection_with_many_deletion_holes() {
        let temp = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..80))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            temp.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                max_rows_per_file: 128,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("i % 2 = 0").await.unwrap();

        let metrics = maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 80,
                max_rows_per_group: 80,
                ..RowAddressMaintenanceOptions::bounded_recluster(vec![
                    ColumnOrdering::asc_nulls_last("i".to_owned()),
                ])
            },
        )
        .await
        .unwrap();

        assert_eq!(metrics.locator_objects_written, 0);
        assert_eq!(metrics.rows_rewritten, 40);
        assert!(
            dataset
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .placements
                .iter()
                .all(|placement| !matches!(placement, RowAddressPlacement::ExplicitMap(_)))
        );
    }

    #[tokio::test]
    async fn bounded_recluster_rejects_excessive_fanout_before_data_writes() {
        use lance_io::utils::tracking_store::{IOTracker, IoOperation};

        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, (0..40).collect()).await;
        let tracker = Arc::new(IOTracker::default());
        dataset = dataset.with_object_store_wrappers([
            tracker.clone() as Arc<dyn lance_io::object_store::WrappingObjectStore>
        ]);
        let version = dataset.version_id();
        tracker.incremental_stats();

        let error = maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 40,
                max_rows_per_group: 40,
                ..RowAddressMaintenanceOptions::bounded_recluster(vec![
                    ColumnOrdering::desc_nulls_last("i".to_owned()),
                ])
            },
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("ExtentFanout"), "{error}");
        assert_eq!(dataset.version_id(), version);
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
            "bounded clustering admission must precede data-object writes: {data_writes:?}"
        );
    }

    #[tokio::test]
    async fn recluster_arbitrary_order_preserves_identity_and_take() {
        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, vec![3, 1, 4, 2, 5, 0]).await;
        let original = scan_i_and_row_ids(&dataset).await;
        let expected_ids = original.into_iter().collect::<HashMap<_, _>>();
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 2,
                max_rows_per_group: 2,
                ..RowAddressMaintenanceOptions::recluster(vec![ColumnOrdering::desc_nulls_last(
                    "i".to_owned(),
                )])
            },
        )
        .await
        .unwrap();
        let scanned = scan_i_and_row_ids(&dataset).await;
        assert_eq!(
            scanned.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
            vec![5, 4, 3, 2, 1, 0]
        );
        for (value, row_id) in &scanned {
            assert_eq!(Some(row_id), expected_ids.get(value));
        }
        let row_id_1 = expected_ids[&1];
        let row_id_4 = expected_ids[&4];
        let taken = dataset
            .take_rows(&[row_id_1, row_id_4], dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(
            taken["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[1, 4]
        );
    }

    #[tokio::test]
    async fn explicit_take_reads_only_the_locator_page() {
        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, (0..12).collect()).await;
        let original = scan_i_and_row_ids(&dataset).await;
        let row_id_6 = original.iter().find(|(value, _)| *value == 6).unwrap().1;
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 12,
                max_rows_per_group: 4,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap();
        let explicit = dataset
            .manifest
            .row_address_layout
            .as_ref()
            .unwrap()
            .placements
            .iter()
            .find_map(|placement| match placement {
                RowAddressPlacement::ExplicitMap(explicit) => Some(explicit),
                _ => None,
            })
            .unwrap();
        assert_eq!(explicit.pages.len(), 3);

        let taken = dataset
            .take_rows(
                &[row_id_6],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            taken["i"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            6
        );
        assert_eq!(
            dataset
                .explicit_cached_ranges_for_path(&explicit.object_path)
                .await,
            vec![(4, 4)]
        );
    }

    #[tokio::test]
    async fn delete_full_explicit_destination_retires_exact_logical_rows() {
        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, (0..10).collect()).await;
        let original = scan_i_and_row_ids(&dataset).await;
        let original_ids = original.iter().copied().collect::<HashMap<_, _>>();
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 3,
                max_rows_per_group: 3,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap();

        let result = dataset.delete("i < 3").await.unwrap();
        assert_eq!(result.num_deleted_rows, 3);
        let scanned = scan_i_and_row_ids(&dataset).await;
        assert_eq!(
            scanned.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
            (3..10).collect::<Vec<_>>()
        );
        for (value, row_id) in scanned {
            assert_eq!(row_id, original_ids[&value]);
        }
        let deleted_ids = (0..3).map(|value| original_ids[&value]).collect::<Vec<_>>();
        assert!(
            dataset
                .resolve_logical_row_ids_async(&deleted_ids)
                .await
                .unwrap()
                .iter()
                .all(Option::is_none)
        );
        dataset.validate().await.unwrap();
    }

    #[derive(Debug)]
    struct FailingCommitHandler;

    #[async_trait::async_trait]
    impl CommitHandler for FailingCommitHandler {
        async fn commit(
            &self,
            _manifest: &mut Manifest,
            _indices: Option<Vec<IndexMetadata>>,
            _base_path: &Path,
            _object_store: &ObjectStore,
            _manifest_writer: ManifestWriter,
            _naming_scheme: ManifestNamingScheme,
            _transaction: Option<Transaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            Err(CommitError::OtherError(Error::io(
                "intentional maintenance commit failure",
            )))
        }
    }

    fn lance_files(path: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut result = Vec::new();
        let Ok(entries) = std::fs::read_dir(path) else {
            return result;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                result.extend(lance_files(&path));
            } else if path
                .extension()
                .is_some_and(|extension| extension == "lance")
            {
                result.push(path);
            }
        }
        result.sort();
        result
    }

    #[tokio::test]
    async fn ambiguous_explicit_commit_error_leaves_objects_for_gc() {
        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, (0..10).collect()).await;
        let before = lance_files(std::path::Path::new(temp.as_ref()).join("data").as_path());
        let original_commit_handler = dataset.commit_handler.clone();
        dataset.commit_handler = Arc::new(FailingCommitHandler);
        let error = maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 3,
                max_rows_per_group: 3,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("intentional maintenance commit failure")
        );
        let after = lance_files(std::path::Path::new(temp.as_ref()).join("data").as_path());
        assert!(after.len() > before.len());

        // Cleanup does not enumerate objects newer than the earliest retained
        // manifest. Publish a later metadata-only snapshot before asking GC to
        // classify the ambiguous attempt as abandoned.
        dataset.commit_handler = original_commit_handler;
        MockClock::set_system_time(
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                + std::time::Duration::from_secs(60),
        );
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions::checkpoint_generation(),
        )
        .await
        .unwrap();

        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await
            .unwrap()
            .delete_unverified(true)
            .build();
        cleanup_old_versions(&dataset, policy).await.unwrap();
        let collected = lance_files(std::path::Path::new(temp.as_ref()).join("data").as_path());
        assert_eq!(before, collected);
    }

    #[tokio::test]
    async fn cleanup_keeps_explicit_locator_snapshot_closure() {
        let temp = TempStrDir::default();
        let mut dataset = write_dataset(&temp, (0..10).collect()).await;
        maintain_row_addresses(
            &mut dataset,
            RowAddressMaintenanceOptions {
                target_rows_per_fragment: 3,
                max_rows_per_group: 3,
                ..RowAddressMaintenanceOptions::repack()
            },
        )
        .await
        .unwrap();
        let explicit = dataset
            .manifest
            .row_address_layout
            .as_ref()
            .unwrap()
            .placements
            .iter()
            .find_map(|placement| match placement {
                RowAddressPlacement::ExplicitMap(explicit) => Some(explicit.clone()),
                _ => None,
            })
            .unwrap();
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await
            .unwrap()
            .delete_unverified(true)
            .build();
        cleanup_old_versions(&dataset, policy).await.unwrap();
        assert!(
            dataset
                .object_store
                .exists(&explicit_object_path(&dataset.base, &explicit.object_path))
                .await
                .unwrap()
        );
        for destination in explicit.destinations {
            assert!(
                dataset
                    .object_store
                    .exists(&explicit_object_path(
                        &dataset.base,
                        &destination.row_id_file_path,
                    ))
                    .await
                    .unwrap()
            );
        }
    }
}
