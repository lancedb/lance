// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Overlay-aware compaction (the "Compaction" section of the Data Overlay Files
//! specification).
//!
//! Overlays accumulate read cost — every overlay is a bitmap to test and a file
//! to open — and an overlay newer than an index leaves that index serving stale
//! values until a query re-evaluates the covered rows on the flat path. This
//! module is aware of that per-fragment state and compacts it in one of two
//! modes:
//!
//! - **Overlay → overlay merge.** Collapse a fragment's overlays into a single,
//!   smaller overlay carrying the per-`(offset, field)` post-image, stamped with
//!   the **maximum** input `committed_version` so the exclusion semantics are
//!   preserved. The base and every index are untouched. Cheap; bounds read cost.
//! - **Overlay → base fold.** Materialize the post-image of every covered cell
//!   into a fresh base data file, tombstone the folded fields in the old files,
//!   and clear the fragment's `overlays`. Row addresses are preserved (a column
//!   rewrite, not a row rewrite). Because the fold removes the overlay that was
//!   excluding the covered rows from any index on a folded field, the same commit
//!   drops those fragments from the index's coverage so they fall to the flat
//!   path — the index can never be left serving stale values.
//!
//! The rewrite reads each required base/overlay cell at most once: overlay value
//! columns are read only at the **ranks that win** (a cell a newer overlay already
//! supplies is never fetched from an older overlay), via
//! [`FileFragment::read_overlay_field_winners`]. Base columns are read whole,
//! which is the cheaper coalesced read the specification permits.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::ArrayRef;
use object_store::path::Path;
use roaring::RoaringBitmap;
use uuid::Uuid;

use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::utils::CachedFileSize;
use lance_table::format::{DataFile, DataOverlayFile, Fragment, IndexMetadata, OverlayCoverage};

use crate::dataset::fragment::{FileFragment, FragReadConfig};
use crate::dataset::overlay::{assemble_overlay_column, route_overlays};
use crate::dataset::transaction::{Operation, UpdateMode};
use crate::dataset::{DATA_DIR, Dataset, WriteDestination};
use crate::index::DatasetIndexExt;

/// Thresholds the scheduler uses to pick a compaction mode per fragment.
#[derive(Debug, Clone)]
pub struct OverlayCompactionOptions {
    /// Fold a fragment to base when an index built on one of its overlaid fields
    /// is at least this many dataset versions behind the overlay that made it
    /// stale (the `committed_version - index.dataset_version` gap). A fold
    /// materializes post-images and reconciles the index, so `1` means "fold as
    /// soon as any index is stale with respect to an overlay". `0` disables
    /// fold-on-staleness.
    pub fold_index_version_gap: u64,
    /// Merge a fragment's overlays (overlay → overlay) once it accumulates at
    /// least this many, to bound per-read overlay cost. Only applies when the
    /// fragment is not already being folded.
    pub merge_overlay_count: usize,
}

impl Default for OverlayCompactionOptions {
    fn default() -> Self {
        Self {
            fold_index_version_gap: 1,
            merge_overlay_count: 4,
        }
    }
}

/// The action the scheduler selects for a fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayCompactionMode {
    /// Leave the fragment's overlays as they are.
    Skip,
    /// Collapse the overlays into a single, smaller overlay (base/indexes untouched).
    Merge,
    /// Materialize overlays into the base and reconcile affected indexes.
    Fold,
}

/// Per-fragment overlay state, the input to the scheduling decision.
#[derive(Debug, Clone)]
pub struct OverlayFragmentState {
    pub fragment_id: u64,
    /// Number of overlays attached to the fragment.
    pub overlay_count: usize,
    /// Distinct physical offsets covered by at least one overlay (union over fields).
    pub covered_offsets: u64,
    /// Sum over fields of each field's coverage popcount — the number of
    /// `(offset, field)` cells the overlays supply.
    pub covered_cells: u64,
    /// Dataset field ids touched by any overlay on the fragment, sorted ascending.
    pub covered_fields: Vec<i32>,
    /// Smallest `committed_version` among the fragment's overlays.
    pub min_committed_version: u64,
    /// Largest `committed_version` among the fragment's overlays.
    pub max_committed_version: u64,
    /// Versions between the current dataset version and the newest overlay — how
    /// long the newest overlay has gone un-compacted.
    pub base_version_gap: u64,
    /// Largest staleness gap to any index that covers this fragment and is built
    /// on a field the fragment overlays:
    /// `max(overlay.committed_version - index.dataset_version)` over such
    /// indexes, or `0` when no index is stale with respect to these overlays.
    /// This is the version-gap staleness signal that drives the fold decision.
    pub max_index_version_gap: u64,
}

impl OverlayFragmentState {
    /// Pick a mode from the version-gap staleness signal and overlay count.
    ///
    /// Fold takes precedence over merge: a merge keeps the overlays, so a stale
    /// index stays stale (queries keep paying the flat re-evaluation cost) —
    /// only a fold can reconcile it.
    pub fn choose_mode(&self, options: &OverlayCompactionOptions) -> OverlayCompactionMode {
        if self.overlay_count == 0 {
            return OverlayCompactionMode::Skip;
        }
        if options.fold_index_version_gap > 0
            && self.max_index_version_gap >= options.fold_index_version_gap
        {
            return OverlayCompactionMode::Fold;
        }
        if self.overlay_count >= options.merge_overlay_count {
            return OverlayCompactionMode::Merge;
        }
        OverlayCompactionMode::Skip
    }
}

/// One fragment's scheduled action.
#[derive(Debug, Clone)]
pub struct OverlayCompactionTask {
    pub state: OverlayFragmentState,
    pub mode: OverlayCompactionMode,
}

/// The scheduler's output: every overlaid fragment with the mode chosen for it.
#[derive(Debug, Clone)]
pub struct OverlayCompactionPlan {
    pub read_version: u64,
    pub tasks: Vec<OverlayCompactionTask>,
}

impl OverlayCompactionPlan {
    /// Tasks that actually do something (mode is not [`OverlayCompactionMode::Skip`]).
    pub fn actionable_tasks(&self) -> impl Iterator<Item = &OverlayCompactionTask> {
        self.tasks
            .iter()
            .filter(|t| t.mode != OverlayCompactionMode::Skip)
    }
}

/// Outcome of a [`compact_overlays`] run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OverlayCompactionMetrics {
    pub fragments_folded: usize,
    pub fragments_merged: usize,
    /// Total overlays removed across folded and merged fragments. A fold removes
    /// all of a fragment's overlays; a merge removes all but the single overlay
    /// it leaves behind.
    pub overlays_removed: usize,
}

/// Compute the overlay state of a single fragment for the scheduler.
fn overlay_fragment_state(
    fragment: &Fragment,
    indices: &[IndexMetadata],
    current_version: u64,
) -> Result<OverlayFragmentState> {
    let mut covered_union = RoaringBitmap::new();
    let mut covered_cells = 0u64;
    let mut covered_fields: Vec<i32> = Vec::new();
    // Newest overlay version per field, to measure index staleness per field.
    let mut field_max_version: HashMap<i32, u64> = HashMap::new();
    let mut min_v = u64::MAX;
    let mut max_v = 0u64;

    for overlay in &fragment.overlays {
        min_v = min_v.min(overlay.committed_version);
        max_v = max_v.max(overlay.committed_version);
        for (field_pos, &field_id) in overlay.data_file.fields.iter().enumerate() {
            if field_id < 0 {
                continue;
            }
            let coverage = overlay.coverage_for_field(field_pos)?;
            covered_cells += coverage.len();
            covered_union |= &*coverage;
            if !covered_fields.contains(&field_id) {
                covered_fields.push(field_id);
            }
            let entry = field_max_version.entry(field_id).or_insert(0);
            *entry = (*entry).max(overlay.committed_version);
        }
    }
    covered_fields.sort_unstable();

    let mut max_index_version_gap = 0u64;
    for index in indices {
        let covers_fragment = index
            .fragment_bitmap
            .as_ref()
            .is_some_and(|b| b.contains(fragment.id as u32));
        if !covers_fragment {
            continue;
        }
        for field_id in &index.fields {
            if let Some(&overlay_version) = field_max_version.get(field_id)
                && overlay_version > index.dataset_version
            {
                max_index_version_gap =
                    max_index_version_gap.max(overlay_version - index.dataset_version);
            }
        }
    }

    Ok(OverlayFragmentState {
        fragment_id: fragment.id,
        overlay_count: fragment.overlays.len(),
        covered_offsets: covered_union.len(),
        covered_cells,
        covered_fields,
        min_committed_version: if min_v == u64::MAX { 0 } else { min_v },
        max_committed_version: max_v,
        base_version_gap: current_version.saturating_sub(max_v),
        max_index_version_gap,
    })
}

/// Plan overlay compaction: gather each overlaid fragment's state and assign a
/// mode. Reads no data files — only manifest metadata and index descriptors.
pub async fn plan_overlay_compaction(
    dataset: &Dataset,
    options: &OverlayCompactionOptions,
) -> Result<OverlayCompactionPlan> {
    let indices = dataset.load_indices().await?;
    let current_version = dataset.manifest().version;

    let mut tasks = Vec::new();
    for fragment in dataset.get_fragments() {
        let metadata = fragment.metadata();
        if metadata.overlays.is_empty() {
            continue;
        }
        let state = overlay_fragment_state(metadata, &indices, current_version)?;
        let mode = state.choose_mode(options);
        tasks.push(OverlayCompactionTask { state, mode });
    }

    Ok(OverlayCompactionPlan {
        read_version: current_version,
        tasks,
    })
}

/// Plan and execute overlay compaction, committing the result.
///
/// Merges and folds are committed separately: a merge must leave indexes
/// untouched, so it is committed with no modified fields, while a fold lists its
/// folded fields so the affected fragments drop out of the covering indexes.
pub async fn compact_overlays(
    dataset: &mut Dataset,
    options: &OverlayCompactionOptions,
) -> Result<OverlayCompactionMetrics> {
    let plan = plan_overlay_compaction(dataset, options).await?;

    let mut merge_fragments: Vec<Fragment> = Vec::new();
    let mut fold_fragments: Vec<Fragment> = Vec::new();
    let mut folded_fields: HashSet<u32> = HashSet::new();
    let mut metrics = OverlayCompactionMetrics::default();

    for task in plan.actionable_tasks() {
        let fragment = dataset
            .get_fragment(task.state.fragment_id as usize)
            .ok_or_else(|| {
                Error::internal(format!(
                    "overlay compaction planned fragment {} which no longer exists",
                    task.state.fragment_id
                ))
            })?;
        match task.mode {
            OverlayCompactionMode::Skip => {}
            OverlayCompactionMode::Merge => {
                let (updated, removed) = merge_fragment_overlays(dataset, &fragment).await?;
                metrics.fragments_merged += 1;
                metrics.overlays_removed += removed;
                merge_fragments.push(updated);
            }
            OverlayCompactionMode::Fold => {
                let (updated, fields) = fold_fragment_overlays(dataset, &fragment).await?;
                metrics.fragments_folded += 1;
                metrics.overlays_removed += task.state.overlay_count;
                folded_fields.extend(fields);
                fold_fragments.push(updated);
            }
        }
    }

    if !merge_fragments.is_empty() {
        commit_update(dataset, merge_fragments, Vec::new()).await?;
    }
    if !fold_fragments.is_empty() {
        commit_update(dataset, fold_fragments, folded_fields.into_iter().collect()).await?;
    }

    Ok(metrics)
}

/// Commit an in-place column rewrite via [`Operation::Update`] in
/// [`UpdateMode::RewriteColumns`] mode, replacing the given fragments. The
/// fragments are taken as-is (so cleared/merged `overlays` and tombstoned fields
/// are preserved), and `fields_modified` drops the touched fragments from any
/// index covering those fields.
async fn commit_update(
    dataset: &mut Dataset,
    updated_fragments: Vec<Fragment>,
    fields_modified: Vec<u32>,
) -> Result<()> {
    let read_version = dataset.manifest().version;
    let operation = Operation::Update {
        removed_fragment_ids: Vec::new(),
        updated_fragments,
        new_fragments: Vec::new(),
        fields_modified,
        merged_generations: Vec::new(),
        fields_for_preserving_frag_bitmap: Vec::new(),
        update_mode: Some(UpdateMode::RewriteColumns),
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let committed = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        operation,
        Some(read_version),
        None,
        None,
        dataset.session.clone(),
        false,
    )
    .await?;
    *dataset = committed;
    Ok(())
}

/// Merge a fragment's overlays into a single overlay (overlay → overlay).
///
/// Returns the updated fragment (with `overlays` replaced by the single merged
/// overlay) and the number of overlays removed (`overlay_count - 1`).
async fn merge_fragment_overlays(
    dataset: &Dataset,
    fragment: &FileFragment,
) -> Result<(Fragment, usize)> {
    let metadata = fragment.metadata();
    let original_count = metadata.overlays.len();
    // Preserve the maximum input committed_version so exclusion semantics hold.
    let merged_version = metadata
        .overlays
        .iter()
        .map(|o| o.committed_version)
        .max()
        .ok_or_else(|| {
            Error::internal("merge_fragment_overlays called on a fragment with no overlays")
        })?;

    let schema = dataset.schema();
    let read_config = FragReadConfig::default();

    let field_ids = covered_field_ids(metadata)?;
    let mut fields = Vec::with_capacity(field_ids.len());
    let mut value_columns: Vec<ArrayRef> = Vec::with_capacity(field_ids.len());
    let mut per_field_coverage: Vec<RoaringBitmap> = Vec::with_capacity(field_ids.len());
    for field_id in &field_ids {
        let field = schema
            .field_by_id(*field_id)
            .ok_or_else(|| Error::internal(format!("overlay field {field_id} not in schema")))?;
        let winners = fragment
            .read_overlay_field_winners(field, &read_config)
            .await?;
        fields.push(field.clone());
        value_columns.push(winners.values);
        per_field_coverage.push(winners.coverage);
    }

    let overlay_schema = Schema {
        fields,
        metadata: Default::default(),
    };
    let data_file = write_data_file(dataset, &overlay_schema, value_columns).await?;

    // Use a single shared bitmap when every field covers the same offsets, else
    // store one bitmap per field (a sparse overlay).
    let coverage = if per_field_coverage.windows(2).all(|w| w[0] == w[1]) {
        OverlayCoverage::dense(per_field_coverage[0].clone())
    } else {
        OverlayCoverage::sparse(per_field_coverage)
    };

    let merged_overlay = DataOverlayFile {
        data_file,
        coverage,
        committed_version: merged_version,
    };

    let mut updated = metadata.clone();
    updated.overlays = vec![merged_overlay];
    Ok((updated, original_count - 1))
}

/// Fold a fragment's overlays into a fresh base data file (overlay → base).
///
/// Returns the updated fragment (new file added, folded fields tombstoned in the
/// old files, `overlays` cleared) and the folded field ids.
async fn fold_fragment_overlays(
    dataset: &Dataset,
    fragment: &FileFragment,
) -> Result<(Fragment, Vec<u32>)> {
    let metadata = fragment.metadata();
    let schema = dataset.schema();
    let read_config = FragReadConfig::default();

    let field_ids = covered_field_ids(metadata)?;
    let mut fields = Vec::with_capacity(field_ids.len());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(field_ids.len());
    for field_id in &field_ids {
        let field = schema
            .field_by_id(*field_id)
            .ok_or_else(|| Error::internal(format!("overlay field {field_id} not in schema")))?;
        // Read each winning overlay cell once; read the base column whole.
        let winners = fragment
            .read_overlay_field_winners(field, &read_config)
            .await?;
        let base = fragment.read_base_field_full(field, &read_config).await?;
        // The base column holds every physical row in order, so row `i` is
        // physical offset `i`. `winners.values` is already the field's post-image
        // indexed by rank in `winners.coverage`, so routing the full contiguous
        // range against that coverage needs exactly those values in order.
        let offsets: Vec<u32> = (0..base.len() as u32).collect();
        let routing = route_overlays(&offsets, &[&winners.coverage]);
        let folded = assemble_overlay_column(&base, &routing, &[winners.values])?;
        fields.push(field.clone());
        columns.push(folded);
    }

    let fold_schema = Schema {
        fields,
        metadata: Default::default(),
    };
    let new_file = write_data_file(dataset, &fold_schema, columns).await?;

    let folded_set: HashSet<i32> = field_ids.iter().copied().collect();
    let mut updated = metadata.clone();
    let new_file_idx = updated.files.len();
    updated.files.push(new_file);
    for (idx, file) in updated.files.iter_mut().enumerate() {
        if idx == new_file_idx {
            continue;
        }
        let tombstoned: Arc<[i32]> = file
            .fields
            .iter()
            .map(|&id| if folded_set.contains(&id) { -2 } else { id })
            .collect::<Vec<_>>()
            .into();
        file.fields = tombstoned;
    }
    updated.overlays.clear();

    let folded_fields = field_ids.iter().map(|&id| id as u32).collect();
    Ok((updated, folded_fields))
}

/// The non-tombstoned dataset field ids that any overlay on the fragment touches,
/// sorted ascending.
fn covered_field_ids(fragment: &Fragment) -> Result<Vec<i32>> {
    let mut ids: Vec<i32> = Vec::new();
    for overlay in &fragment.overlays {
        for &field_id in overlay.data_file.fields.iter() {
            if field_id >= 0 && !ids.contains(&field_id) {
                ids.push(field_id);
            }
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

/// Write `columns` (in `schema` field order, lengths may differ for a sparse
/// overlay) to a new data file under the dataset's `data/` directory and return
/// the resulting [`DataFile`] with its field/column index mapping populated.
async fn write_data_file(
    dataset: &Dataset,
    schema: &Schema,
    columns: Vec<ArrayRef>,
) -> Result<DataFile> {
    let version = dataset
        .manifest()
        .data_storage_format
        .lance_file_version()?;
    let filename = format!("{}.lance", Uuid::new_v4());
    let path: Path = dataset.base.clone().join(DATA_DIR).join(filename.as_str());
    let object_writer = dataset.object_store.create(&path).await?;
    let mut writer = FileWriter::try_new(
        object_writer,
        schema.clone(),
        FileWriterOptions {
            format_version: Some(version),
            ..Default::default()
        },
    )?;
    let (major, minor) = writer.version().to_numbers();
    for (column_index, array) in columns.into_iter().enumerate() {
        writer.write_column(column_index, array).await?;
    }
    let summary = writer.finish().await?;

    let mut data_file = DataFile::new_unstarted(filename, major, minor);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(field_id, _)| *field_id as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, column_index)| *column_index as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);
    Ok(data_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use arrow_array::{Array, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_io::{assert_io_lt, utils::tracking_store::IoStats};
    use uuid::Uuid;

    use crate::dataset::transaction::{DataOverlayGroup, Operation};
    use crate::dataset::{WriteDestination, WriteParams};
    use crate::index::DatasetIndexExt;

    fn bitmap(offsets: impl IntoIterator<Item = u32>) -> RoaringBitmap {
        RoaringBitmap::from_iter(offsets)
    }

    fn i32_array(values: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from_iter(values))
    }

    // ---------------------------------------------------------------------
    // Scheduler / awareness unit tests (synthetic metadata, no I/O)
    // ---------------------------------------------------------------------

    fn synth_overlay(fields: &[i32], offsets: &[u32], committed_version: u64) -> DataOverlayFile {
        DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", fields.to_vec(), None),
            coverage: OverlayCoverage::dense(bitmap(offsets.iter().copied())),
            committed_version,
        }
    }

    fn synth_index(
        name: &str,
        fields: &[i32],
        dataset_version: u64,
        frags: &[u32],
    ) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: fields.to_vec(),
            name: name.to_string(),
            dataset_version,
            fragment_bitmap: Some(bitmap(frags.iter().copied())),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    #[test]
    fn test_state_version_gaps_and_cell_counts() {
        let mut fragment = Fragment::new(0).with_physical_rows(6);
        // v3 overlays field 1 over {1,2}; v5 overlays fields {1,2} over {3}.
        fragment.overlays = vec![
            synth_overlay(&[1], &[1, 2], 3),
            synth_overlay(&[1, 2], &[3], 5),
        ];
        // Index on field 1 built at version 2, covering fragment 0.
        let indices = vec![synth_index("val_idx", &[1], 2, &[0])];

        let state = overlay_fragment_state(&fragment, &indices, 7).unwrap();
        assert_eq!(state.overlay_count, 2);
        assert_eq!(state.covered_fields, vec![1, 2]);
        // cells = v3{field1: 2} + v5{field1: 1, field2: 1} = 4.
        assert_eq!(state.covered_cells, 4);
        // distinct physical offsets = {1,2,3}.
        assert_eq!(state.covered_offsets, 3);
        assert_eq!(state.min_committed_version, 3);
        assert_eq!(state.max_committed_version, 5);
        assert_eq!(state.base_version_gap, 2); // current 7 - newest 5.
        // field 1's newest overlay (v5) vs index built at v2 -> gap 3.
        assert_eq!(state.max_index_version_gap, 3);
    }

    #[test]
    fn test_mode_fold_when_index_is_stale() {
        let mut fragment = Fragment::new(0).with_physical_rows(6);
        fragment.overlays = vec![synth_overlay(&[1], &[1], 5)];
        let indices = vec![synth_index("val_idx", &[1], 2, &[0])];
        let state = overlay_fragment_state(&fragment, &indices, 5).unwrap();
        assert_eq!(
            state.choose_mode(&OverlayCompactionOptions::default()),
            OverlayCompactionMode::Fold
        );
    }

    #[test]
    fn test_mode_field_aware_no_fold_for_unrelated_index() {
        let mut fragment = Fragment::new(0).with_physical_rows(6);
        // Overlay touches field 1 only.
        fragment.overlays = vec![synth_overlay(&[1], &[1], 5)];
        // Index is on field 2 (unrelated) -> not stale w.r.t. this overlay.
        let indices = vec![synth_index("other_idx", &[2], 2, &[0])];
        let state = overlay_fragment_state(&fragment, &indices, 5).unwrap();
        assert_eq!(state.max_index_version_gap, 0);
        // No stale index and only one overlay -> nothing to do.
        assert_eq!(
            state.choose_mode(&OverlayCompactionOptions::default()),
            OverlayCompactionMode::Skip
        );
    }

    #[test]
    fn test_mode_merge_on_overlay_count_without_stale_index() {
        let mut fragment = Fragment::new(0).with_physical_rows(6);
        fragment.overlays = (1..=4).map(|v| synth_overlay(&[1], &[1], v)).collect();
        // Index already current (built after every overlay) -> not stale.
        let indices = vec![synth_index("val_idx", &[1], 10, &[0])];
        let state = overlay_fragment_state(&fragment, &indices, 10).unwrap();
        assert_eq!(state.max_index_version_gap, 0);
        let options = OverlayCompactionOptions {
            fold_index_version_gap: 1,
            merge_overlay_count: 4,
        };
        assert_eq!(state.choose_mode(&options), OverlayCompactionMode::Merge);
    }

    #[test]
    fn test_mode_skip_below_thresholds() {
        let mut fragment = Fragment::new(0).with_physical_rows(6);
        fragment.overlays = vec![synth_overlay(&[1], &[1], 3), synth_overlay(&[1], &[2], 4)];
        let indices = vec![synth_index("val_idx", &[1], 10, &[0])];
        let state = overlay_fragment_state(&fragment, &indices, 10).unwrap();
        assert_eq!(
            state.choose_mode(&OverlayCompactionOptions::default()),
            OverlayCompactionMode::Skip
        );
    }

    // ---------------------------------------------------------------------
    // End-to-end execution tests
    // ---------------------------------------------------------------------

    /// Two-fragment Int32 dataset: `id` (field 0) = 0..12 and `val` (field 1) =
    /// id * 10, six rows per file (fragments 0 and 1).
    async fn create_base_dataset(uri: &str) -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new("val", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..12)),
                Arc::new(Int32Array::from_iter_values((0..12).map(|v| v * 10))),
            ],
        )
        .unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 6,
            max_rows_per_group: 6,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(reader, uri, Some(write_params))
            .await
            .unwrap()
    }

    /// Write an overlay file covering `fields` of `fragment_id` with the given
    /// coverage and per-field value columns, then commit it as a `DataOverlay`.
    async fn commit_overlay(
        dataset: Dataset,
        fragment_id: u64,
        fields: &[i32],
        coverage: OverlayCoverage,
        columns: Vec<ArrayRef>,
    ) -> Dataset {
        let read_version = dataset.version().version;
        let overlay_schema = dataset.schema().project_by_ids(fields, true);
        let filename = format!("{}.lance", Uuid::new_v4());
        let path = dataset.base.clone().join(DATA_DIR).join(filename.as_str());
        let obj_writer = dataset.object_store.create(&path).await.unwrap();
        let mut writer = FileWriter::try_new(
            obj_writer,
            overlay_schema,
            FileWriterOptions {
                format_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            },
        )
        .unwrap();
        let (major, minor) = writer.version().to_numbers();
        for (column_index, array) in columns.into_iter().enumerate() {
            writer.write_column(column_index, array).await.unwrap();
        }
        let summary = writer.finish().await.unwrap();

        let mut data_file = DataFile::new_unstarted(filename, major, minor);
        data_file.fields = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(f, _)| *f as i32)
            .collect::<Vec<_>>()
            .into();
        data_file.column_indices = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(_, c)| *c as i32)
            .collect::<Vec<_>>()
            .into();
        data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

        let overlay = DataOverlayFile {
            data_file,
            coverage,
            committed_version: 0,
        };
        Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataOverlay {
                groups: vec![DataOverlayGroup {
                    fragment_id,
                    overlays: vec![overlay],
                }],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap()
    }

    /// Scan `id` and `val` and return an `id -> val` map (order-independent).
    async fn id_val_map(dataset: &Dataset) -> BTreeMap<i32, Option<i32>> {
        let mut scanner = dataset.scan();
        scanner.project(&["id", "val"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut out = BTreeMap::new();
        for batch in batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let vals = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                let v = if vals.is_null(i) {
                    None
                } else {
                    Some(vals.value(i))
                };
                out.insert(ids.value(i), v);
            }
        }
        out
    }

    #[tokio::test]
    async fn test_fold_materializes_overlay_and_clears_overlays() {
        let dataset = create_base_dataset("memory://").await;
        // Overlay fragment 0: val[1] 10 -> 999, val[4] 40 -> 444.
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([1, 4])),
            vec![i32_array([Some(999), Some(444)])],
        )
        .await;

        let mut dataset = dataset;
        let fragment = dataset.get_fragment(0).unwrap();
        let (updated, fields) = fold_fragment_overlays(&dataset, &fragment).await.unwrap();
        assert_eq!(fields, vec![1]);
        assert!(updated.overlays.is_empty());
        commit_update(&mut dataset, vec![updated], fields)
            .await
            .unwrap();

        // Overlays gone; values materialized into the base.
        assert!(
            dataset
                .get_fragment(0)
                .unwrap()
                .metadata()
                .overlays
                .is_empty()
        );
        let map = id_val_map(&dataset).await;
        assert_eq!(map[&1], Some(999));
        assert_eq!(map[&4], Some(444));
        assert_eq!(map[&0], Some(0));
        assert_eq!(map[&2], Some(20));
        // Fragment 1 untouched.
        assert_eq!(map[&7], Some(70));
    }

    #[tokio::test]
    async fn test_fold_null_override() {
        let dataset = create_base_dataset("memory://").await;
        // Overlay sets val[2] to NULL (covered + null overrides to null).
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([2])),
            vec![i32_array([None])],
        )
        .await;

        let mut dataset = dataset;
        let fragment = dataset.get_fragment(0).unwrap();
        let (updated, fields) = fold_fragment_overlays(&dataset, &fragment).await.unwrap();
        commit_update(&mut dataset, vec![updated], fields)
            .await
            .unwrap();

        let map = id_val_map(&dataset).await;
        assert_eq!(map[&2], None); // overridden to null
        assert_eq!(map[&1], Some(10)); // untouched
    }

    #[tokio::test]
    async fn test_fold_multi_fragment_multi_overlay() {
        let dataset = create_base_dataset("memory://").await;
        // Two overlays on fragment 0 (newest wins on the shared offset 1).
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([1, 2])),
            vec![i32_array([Some(100), Some(200)])],
        )
        .await;
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([1])),
            vec![i32_array([Some(111)])],
        )
        .await;
        // One overlay on fragment 1: offset 0 is id 6 -> val 600.
        let dataset = commit_overlay(
            dataset,
            1,
            &[1],
            OverlayCoverage::dense(bitmap([0])),
            vec![i32_array([Some(600)])],
        )
        .await;

        let mut dataset = dataset;
        let frag0 = dataset.get_fragment(0).unwrap();
        let frag1 = dataset.get_fragment(1).unwrap();
        let (u0, f0) = fold_fragment_overlays(&dataset, &frag0).await.unwrap();
        let (u1, f1) = fold_fragment_overlays(&dataset, &frag1).await.unwrap();
        let mut fields = f0;
        fields.extend(f1);
        fields.sort_unstable();
        fields.dedup();
        commit_update(&mut dataset, vec![u0, u1], fields)
            .await
            .unwrap();

        let map = id_val_map(&dataset).await;
        assert_eq!(map[&1], Some(111)); // newest overlay won
        assert_eq!(map[&2], Some(200));
        assert_eq!(map[&6], Some(600));
        assert_eq!(map[&3], Some(30)); // untouched
        for frag in dataset.get_fragments() {
            assert!(frag.metadata().overlays.is_empty());
        }
    }

    #[tokio::test]
    async fn test_fold_reconciles_stale_index() {
        let test_dir = TempStrDir::default();
        let mut dataset = create_base_dataset(&test_dir).await;
        dataset
            .create_index(
                &["val"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Overlay val[1] 10 -> 999 (committed after the index -> index is stale).
        dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([1])),
            vec![i32_array([Some(999)])],
        )
        .await;

        // The scheduler should fold fragment 0 to reconcile the stale index.
        let plan = plan_overlay_compaction(&dataset, &OverlayCompactionOptions::default())
            .await
            .unwrap();
        let frag0_task = plan
            .tasks
            .iter()
            .find(|t| t.state.fragment_id == 0)
            .unwrap();
        assert_eq!(frag0_task.mode, OverlayCompactionMode::Fold);
        assert!(frag0_task.state.max_index_version_gap >= 1);

        let metrics = compact_overlays(&mut dataset, &OverlayCompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(metrics.fragments_folded, 1);

        // Index reconciled: fragment 0 dropped from the val index's coverage.
        let indices = dataset.load_indices().await.unwrap();
        let val_index = indices
            .iter()
            .find(|i| i.fields == vec![1])
            .expect("val index present");
        assert!(
            !val_index.fragment_bitmap.as_ref().unwrap().contains(0),
            "folded fragment must be removed from the stale index's coverage"
        );

        // And the query is correct: val = 999 finds id 1, the stale 10 is gone.
        let mut scanner = dataset.scan();
        scanner
            .filter("val = 999")
            .unwrap()
            .project(&["id"])
            .unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids.value(0), 1);
        assert!(
            dataset
                .get_fragment(0)
                .unwrap()
                .metadata()
                .overlays
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_merge_collapses_overlays_preserving_max_version() {
        let dataset = create_base_dataset("memory://").await;
        // Three overlays on fragment 0; newest (highest committed_version) wins.
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([1, 2])),
            vec![i32_array([Some(100), Some(200)])],
        )
        .await;
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([1, 3])),
            vec![i32_array([Some(111), Some(333)])],
        )
        .await;
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([4])),
            vec![i32_array([Some(444)])],
        )
        .await;
        let expected_version = dataset.version().version; // newest overlay's commit version

        let mut dataset = dataset;
        let fragment = dataset.get_fragment(0).unwrap();
        let (updated, removed) = merge_fragment_overlays(&dataset, &fragment).await.unwrap();
        assert_eq!(removed, 2); // 3 overlays -> 1
        assert_eq!(updated.overlays.len(), 1);
        let merged = &updated.overlays[0];
        assert_eq!(merged.committed_version, expected_version);
        // Union coverage over all three overlays for field 1.
        assert_eq!(*merged.coverage_for_field(0).unwrap(), bitmap([1, 2, 3, 4]));

        commit_update(&mut dataset, vec![updated], Vec::new())
            .await
            .unwrap();
        // The single merged overlay reproduces the newest-wins post-image on read.
        let map = id_val_map(&dataset).await;
        assert_eq!(map[&1], Some(111)); // second overlay newest for offset 1
        assert_eq!(map[&2], Some(200));
        assert_eq!(map[&3], Some(333));
        assert_eq!(map[&4], Some(444));
        assert_eq!(map[&0], Some(0)); // untouched
    }

    /// Build a single-fragment dataset of `n` rows (`val` = field 1) on a
    /// tracking local store, attach two overlays over `val` with the given
    /// coverage, merge, and return the overlay value bytes/iops the merge read.
    async fn merge_value_io(n: i32, cov_a: RoaringBitmap, cov_b: RoaringBitmap) -> IoStats {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new("val", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..n)),
                Arc::new(Int32Array::from_iter_values((0..n).map(|v| v * 10))),
            ],
        )
        .unwrap();
        let write_params = WriteParams {
            max_rows_per_file: n as usize,
            max_rows_per_group: n as usize,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(reader, &test_dir, Some(write_params))
            .await
            .unwrap();

        let vals_a: Vec<Option<i32>> = cov_a.iter().map(|o| Some(o as i32)).collect();
        let vals_b: Vec<Option<i32>> = cov_b.iter().map(|o| Some(-(o as i32) - 1)).collect();
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(cov_a),
            vec![i32_array(vals_a)],
        )
        .await;
        let dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(cov_b),
            vec![i32_array(vals_b)],
        )
        .await;

        let fragment = dataset.get_fragment(0).unwrap();
        // Reset counters, then measure only what the merge reads.
        let _ = dataset.object_store.io_stats_incremental();
        let _ = merge_fragment_overlays(&dataset, &fragment).await.unwrap();
        dataset.object_store.io_stats_incremental()
    }

    /// Read-each-cell-at-most-once: with equal total coverage, overlapping
    /// overlays read strictly fewer value bytes than disjoint ones, because a
    /// cell a newer overlay supplies is not also read from the older overlay.
    /// The disjoint case is the negative control that makes the bound meaningful.
    #[tokio::test]
    async fn test_merge_reads_each_winning_cell_at_most_once() {
        const N: i32 = 8000;
        // Both overlays cover 4000 offsets, so per-file fixed costs match.
        // Overlapping: cov_a = [0, 4000), cov_b = [2000, 6000). union = 6000;
        // offsets [2000, 4000) are read once, from the newer overlay b only.
        let overlap = merge_value_io(N, bitmap(0..4000), bitmap(2000..6000)).await;
        // Disjoint negative control: cov_a = [0, 4000), cov_b = [4000, 8000).
        // Same per-overlay sizes, but union = 8000 distinct cells.
        let disjoint = merge_value_io(N, bitmap(0..4000), bitmap(4000..8000)).await;

        // The only difference between the two is how many distinct cells are
        // fetched (6000 vs 8000). If overlapping cells were double-read, the
        // overlap case would not read strictly fewer value bytes.
        assert!(
            overlap.read_bytes > 0 && disjoint.read_bytes > 0,
            "both merges must read overlay values"
        );
        assert_io_lt!(
            overlap,
            read_bytes,
            disjoint.read_bytes,
            "overlapping coverage must not re-read cells a newer overlay already supplies \
             (overlap read {} bytes, disjoint {} bytes)",
            overlap.read_bytes,
            disjoint.read_bytes
        );
    }
}
