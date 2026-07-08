// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolution of data overlay files on read.
//!
//! An overlay supplies replacement values for some `(row, field)` cells without
//! rewriting the base data. Resolving a read means, for each row we return,
//! deciding whether its value comes from the base column or from an overlay.
//!
//! Three coordinate spaces show up throughout this module; keeping them straight
//! is most of the work:
//!
//! - `offset_in_frag`: a row's physical position in the fragment (0-based over all
//!   physical rows, ignoring deletions). This is how a cell is addressed on disk
//!   and in an overlay's coverage bitmap.
//! - `offset_in_batch`: a row's position within the batch we are currently
//!   assembling (0-based). The output column is indexed by this.
//! - `offset_in_overlay`: the position of a value in an overlay's value column.
//!   An overlay stores its values densely — one per covered cell, in ascending
//!   `offset_in_frag` order — so a covered cell's value is found by counting how
//!   many covered cells come before it. (That count is what a roaring bitmap calls
//!   the cell's "rank".)
//!
//! For a given field, the overlays covering it are consulted newest to oldest: the
//! first overlay that covers a row wins, and its value is read at that row's
//! `offset_in_overlay`. A row that no overlay covers keeps its base value.
//!
//! The rows to resolve are passed in as a list of `offset_in_frag` (one per output
//! row), so a single code path serves both scans (a contiguous range of offsets)
//! and `take` (arbitrary offsets).
//!
//! Deletions win over overlays, but nothing here handles that: the merge runs on
//! physical rows *before* deletions are applied, so an overlay value computed for a
//! deleted row is simply dropped along with the row. This matches the spec with no
//! special casing.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_select::interleave::interleave;
use futures::StreamExt;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use lance_table::format::DataFile;
use lance_table::format::overlay::DataOverlayFile;
use lance_table::utils::stream::ReadBatchFut;

use crate::dataset::fragment::{FileFragment, FragReadConfig, GenericFileReader};

/// Order a fragment's overlays from newest to oldest for read resolution.
///
/// Precedence is by `committed_version` (higher is newer); ties are broken by
/// position in the fragment's `overlays` list, where a later entry is newer.
/// Returns indices into `overlays`.
pub fn overlay_indices_newest_first(overlays: &[DataOverlayFile]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..overlays.len()).collect();
    indices.sort_by(|&a, &b| {
        overlays[b]
            .committed_version
            .cmp(&overlays[a].committed_version)
            .then(b.cmp(&a))
    });
    indices
}

/// The plan for merging one field's overlays into one batch: which source (base or
/// a particular overlay) supplies each output row, and which overlay values must be
/// fetched to do it.
///
/// Built by [`route_overlays`] from the coverage bitmaps alone — before any value
/// column is read — so the caller can fetch only the overlay values it will
/// actually use (see [`OverlayRouting::offsets_in_overlay`]) rather than whole
/// columns, then build the merged column with [`assemble_overlay_column`].
pub struct OverlayRouting {
    /// One `(source, position)` pair per output row, ready to hand to `interleave`.
    /// Source `0` is the base column, with `position` = the row's `offset_in_batch`;
    /// source `k + 1` is overlay `k`'s fetched values, with `position` = the row's
    /// index into those fetched values.
    indices: Vec<(usize, usize)>,
    /// Per overlay (newest-first): the sorted, deduplicated `offset_in_overlay`
    /// values this batch needs from that overlay — i.e. exactly which entries of its
    /// value column to fetch.
    offsets_in_overlay: Vec<Vec<u32>>,
    /// Whether any row is covered by an overlay at all (false ⇒ every row falls
    /// through to the base column).
    any_overlay: bool,
}

impl OverlayRouting {
    /// Per overlay (newest-first), the `offset_in_overlay` values to fetch from its
    /// value column.
    pub fn offsets_in_overlay(&self) -> &[Vec<u32>] {
        &self.offsets_in_overlay
    }

    /// True when no row is covered by any overlay, so the base column is already the
    /// answer and no overlay values need to be read.
    pub fn all_fall_through(&self) -> bool {
        !self.any_overlay
    }
}

/// For each row in `offsets_in_frag`, decide whether its value comes from the base
/// column or from an overlay — and if from an overlay, at which `offset_in_overlay`.
///
/// Only the coverage bitmaps are consulted (newest-first), so this runs before any
/// value column is read and reports exactly which overlay values the caller must
/// fetch.
///
/// A scan asks for a contiguous, ascending range of offsets, which enables a faster
/// bitmap-driven path ([`route_contiguous`]); `take` asks for arbitrary offsets and
/// uses the general path ([`route_arbitrary`]). Both produce identical routing.
pub fn route_overlays(
    offsets_in_frag: &[u32],
    coverages_newest_first: &[&RoaringBitmap],
) -> OverlayRouting {
    match contiguous_frag_start(offsets_in_frag) {
        Some(frag_start) => {
            route_contiguous(frag_start, offsets_in_frag.len(), coverages_newest_first)
        }
        None => route_arbitrary(offsets_in_frag, coverages_newest_first),
    }
}

/// If `offsets_in_frag` is a contiguous ascending run `[start, start + 1, ...]`,
/// return `start`; otherwise `None` (including when empty).
fn contiguous_frag_start(offsets_in_frag: &[u32]) -> Option<u32> {
    let start = *offsets_in_frag.first()?;
    offsets_in_frag
        .iter()
        .enumerate()
        .all(|(i, &offset)| offset as u64 == start as u64 + i as u64)
        .then_some(start)
}

/// Fast path for a scan, where the batch is a contiguous run of offsets starting at
/// `frag_start`. Because the offsets are contiguous, a row's `offset_in_batch` is
/// just `offset_in_frag - frag_start`, so a coverage's set bits map straight to
/// output rows — no need to test each row against each coverage.
///
/// For each coverage we intersect it with the batch's offset range. Roaring does
/// this a block at a time, so a coverage that does not overlap the batch (e.g. a
/// scan batch past the last cell this overlay touches) is skipped cheaply without
/// inspecting individual bits.
///
/// Within the batch a coverage's cells appear in ascending order, so their
/// `offset_in_overlay` values are consecutive: the first in-batch cell sits at
/// `offset_in_overlay = <cells this coverage has before the batch>` (a single
/// `rank` lookup), and each following cell is one more. Coverages are applied
/// newest-first, and the first overlay to claim a row wins.
fn route_contiguous(
    frag_start: u32,
    len: usize,
    coverages_newest_first: &[&RoaringBitmap],
) -> OverlayRouting {
    let mut offsets_in_overlay: Vec<Vec<u32>> = vec![Vec::new(); coverages_newest_first.len()];
    // Indexed by offset_in_batch: which (overlay, fetch position) supplies the row.
    let mut routed: Vec<Option<(usize, usize)>> = vec![None; len];
    let range_end = (frag_start as u64 + len as u64).min(u32::MAX as u64) as u32;
    let mut batch_range = RoaringBitmap::new();
    batch_range.insert_range(frag_start..range_end);

    for (k, coverage) in coverages_newest_first.iter().enumerate() {
        let covered_in_batch = *coverage & &batch_range;
        if covered_in_batch.is_empty() {
            continue;
        }
        // offset_in_overlay of this coverage's first in-batch cell = the number of
        // its cells that lie before the batch.
        let first_offset_in_overlay = if frag_start == 0 {
            0
        } else {
            coverage.rank(frag_start - 1) as u32
        };
        for (nth_in_batch, offset_in_frag) in covered_in_batch.iter().enumerate() {
            let offset_in_batch = (offset_in_frag - frag_start) as usize;
            if routed[offset_in_batch].is_none() {
                routed[offset_in_batch] = Some((k, offsets_in_overlay[k].len()));
                offsets_in_overlay[k].push(first_offset_in_overlay + nth_in_batch as u32);
            }
        }
    }

    let mut any_overlay = false;
    let indices = routed
        .into_iter()
        .enumerate()
        .map(|(offset_in_batch, routed)| match routed {
            None => (0, offset_in_batch),
            Some((k, fetch_pos)) => {
                any_overlay = true;
                (k + 1, fetch_pos)
            }
        })
        .collect();

    OverlayRouting {
        indices,
        offsets_in_overlay,
        any_overlay,
    }
}

/// General path for arbitrary offsets (e.g. `take`): test each row's
/// `offset_in_frag` against the coverages newest-first. `take` batches are small,
/// so this `O(rows * overlays)` probing is not a concern.
fn route_arbitrary(
    offsets_in_frag: &[u32],
    coverages_newest_first: &[&RoaringBitmap],
) -> OverlayRouting {
    // Per overlay: the distinct offset_in_overlay values this batch needs, sorted.
    let mut offset_sets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); coverages_newest_first.len()];
    // Per output row: the (overlay, offset_in_overlay) that supplies it, if any.
    let mut routed_per_row: Vec<Option<(usize, u32)>> = Vec::with_capacity(offsets_in_frag.len());
    for &offset_in_frag in offsets_in_frag {
        let mut routed = None;
        for (k, coverage) in coverages_newest_first.iter().enumerate() {
            if coverage.contains(offset_in_frag) {
                // offset_in_overlay = number of covered cells before this one.
                let offset_in_overlay = coverage.rank(offset_in_frag) as u32 - 1;
                offset_sets[k].insert(offset_in_overlay);
                routed = Some((k, offset_in_overlay));
                break;
            }
        }
        routed_per_row.push(routed);
    }

    let offsets_in_overlay: Vec<Vec<u32>> = offset_sets
        .iter()
        .map(|offsets| offsets.iter().copied().collect())
        .collect();
    // For each overlay, map an offset_in_overlay to its position in the fetched
    // (sorted, deduplicated) value list.
    let fetch_positions: Vec<HashMap<u32, usize>> = offsets_in_overlay
        .iter()
        .map(|offsets| {
            offsets
                .iter()
                .enumerate()
                .map(|(pos, &o)| (o, pos))
                .collect()
        })
        .collect();

    let mut any_overlay = false;
    let indices = routed_per_row
        .into_iter()
        .enumerate()
        .map(|(offset_in_batch, routed)| match routed {
            None => (0, offset_in_batch),
            Some((k, offset_in_overlay)) => {
                any_overlay = true;
                (k + 1, fetch_positions[k][&offset_in_overlay])
            }
        })
        .collect();

    OverlayRouting {
        indices,
        offsets_in_overlay,
        any_overlay,
    }
}

/// Build the merged column from `base` and the overlay values fetched for the
/// `offset_in_overlay` values [`route_overlays`] asked for.
///
/// `fetched_newest_first[k]` holds overlay `k`'s values for `routing`'s
/// `offsets_in_overlay()[k]`, in that order. The result has the same length and
/// type as `base`. A covered row whose overlay value is NULL resolves **to** NULL
/// (distinct from a fall-through, which keeps the base value).
pub fn assemble_overlay_column(
    base: &ArrayRef,
    routing: &OverlayRouting,
    fetched_newest_first: &[ArrayRef],
) -> Result<ArrayRef> {
    if routing.all_fall_through() {
        return Ok(base.clone());
    }
    if fetched_newest_first.len() != routing.offsets_in_overlay.len() {
        return Err(Error::invalid_input(format!(
            "overlay assembly got {} value columns but routing expects {}",
            fetched_newest_first.len(),
            routing.offsets_in_overlay.len()
        )));
    }
    for (k, values) in fetched_newest_first.iter().enumerate() {
        if values.len() != routing.offsets_in_overlay[k].len() {
            return Err(Error::invalid_input(format!(
                "overlay value column {} has {} values but {} were requested",
                k,
                values.len(),
                routing.offsets_in_overlay[k].len()
            )));
        }
    }

    let mut sources: Vec<&dyn Array> = Vec::with_capacity(fetched_newest_first.len() + 1);
    sources.push(base.as_ref());
    for values in fetched_newest_first {
        sources.push(values.as_ref());
    }
    interleave(&sources, &routing.indices).map_err(Error::from)
}

/// One overlay's contribution to one projected field, with its file reader opened:
/// the cells it covers, and the reader from which the field's values are fetched by
/// `offset_in_overlay` at merge time.
#[derive(Debug, Clone)]
struct LoadedFieldOverlay {
    /// The `offset_in_frag` cells this overlay covers for the field.
    coverage: Arc<RoaringBitmap>,
    /// Reader over the overlay data file, projected to the covered fields; shared
    /// across the fields that the same file covers.
    reader: Arc<dyn GenericFileReader>,
    /// Single-field projection used when fetching the value column.
    field_projection: Arc<Schema>,
}

/// The overlays that apply to a single projected field, ordered newest-first, with
/// readers opened and pruned to a specific read. `field_name` is the top-level
/// read-batch column name the plan applies to. Produced by [`resolve_overlays`] and
/// consumed by [`merge_overlay_batch`].
#[derive(Debug, Clone)]
pub struct FieldOverlayPlan {
    field_name: String,
    overlays_newest_first: Vec<LoadedFieldOverlay>,
}

/// One overlay file that may contribute to a read, before it is opened. Opened
/// lazily by [`resolve_overlays`], and only if the read actually touches it.
#[derive(Debug, Clone)]
struct PlannedOverlayFile {
    data_file: DataFile,
    /// The covered ∩ projected fields to project when the file is opened, so a
    /// single reader serves every field the file contributes to.
    open_projection: Arc<Schema>,
}

/// One overlay's contribution to one projected field, before the file is opened.
#[derive(Debug, Clone)]
struct PlannedFieldOverlay {
    /// Index into [`OverlayReadPlanner::files`] of the file that supplies the value.
    file: usize,
    coverage: Arc<RoaringBitmap>,
}

/// The overlays that apply to a single projected field, ordered newest-first,
/// before any file is opened.
#[derive(Debug, Clone)]
struct PlannedField {
    field_name: String,
    /// Single-field projection used when fetching this field's value column.
    field_projection: Arc<Schema>,
    overlays_newest_first: Vec<PlannedFieldOverlay>,
}

/// A fragment's overlay-resolution plan for a projection, derived from coverage
/// metadata alone — no file opened, no IO. [`resolve_overlays`] turns it into opened
/// [`FieldOverlayPlan`]s for one specific read, opening only the files whose cells
/// the read's rows actually touch.
#[derive(Debug, Clone)]
pub struct OverlayReadPlanner {
    files: Vec<PlannedOverlayFile>,
    fields: Vec<PlannedField>,
}

impl OverlayReadPlanner {
    /// True when no projected field has any overlay, so there is nothing to resolve.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Plan `fragment`'s overlay resolution for a projection from coverage metadata
/// alone. No files are opened here (see [`resolve_overlays`]) — this only reads the
/// already-parsed coverage bitmaps, so it is cheap enough to run on every open.
///
/// For each projected (top-level) field, the fragment's overlays are walked
/// newest-first; an overlay contributes if its `data_file.fields` includes the
/// field. Overlays on nested (non-top-level) fields are not yet supported and are
/// simply not matched here. Each contributing overlay *file* appears once in
/// `files`, shared by every field it covers.
pub fn plan_overlays(fragment: &FileFragment, projection: &Schema) -> Result<OverlayReadPlanner> {
    let order = overlay_indices_newest_first(&fragment.metadata.overlays);

    // One entry per contributing overlay file, newest-first. `pos_to_file[pos]` maps
    // a position in `order` to its index in `files` (None if it covers no projected
    // field, so it is never referenced and never opened).
    let mut files = Vec::new();
    let mut pos_to_file = vec![None; order.len()];
    for (pos, &overlay_idx) in order.iter().enumerate() {
        let overlay = &fragment.metadata.overlays[overlay_idx];
        let covered: Vec<lance_core::datatypes::Field> = projection
            .fields
            .iter()
            .filter(|f| overlay.data_file.fields.contains(&f.id))
            .cloned()
            .collect();
        if covered.is_empty() {
            continue;
        }
        pos_to_file[pos] = Some(files.len());
        files.push(PlannedOverlayFile {
            data_file: overlay.data_file.clone(),
            open_projection: Arc::new(Schema {
                fields: covered,
                metadata: Default::default(),
            }),
        });
    }

    let mut fields = Vec::new();
    for field in &projection.fields {
        let mut overlays_newest_first = Vec::new();
        for (pos, &overlay_idx) in order.iter().enumerate() {
            let overlay = &fragment.metadata.overlays[overlay_idx];
            let Some(field_pos) = overlay
                .data_file
                .fields
                .iter()
                .position(|&id| id == field.id)
            else {
                continue;
            };
            let Some(file) = pos_to_file[pos] else {
                continue;
            };
            overlays_newest_first.push(PlannedFieldOverlay {
                file,
                coverage: overlay.coverage_for_field(field_pos)?,
            });
        }
        if !overlays_newest_first.is_empty() {
            fields.push(PlannedField {
                field_name: field.name.clone(),
                field_projection: Arc::new(Schema {
                    fields: vec![field.clone()],
                    metadata: Default::default(),
                }),
                overlays_newest_first,
            });
        }
    }
    Ok(OverlayReadPlanner { files, fields })
}

/// Open the overlay readers a specific read needs and return the per-field plans to
/// merge, pruned to that read.
///
/// `offsets_in_frag` are the rows the read will return. An overlay whose coverage is
/// disjoint from those rows contributes nothing, so it is dropped and its file is
/// never opened — a `take` that misses an overlay's cells pays no IO for it. Each
/// surviving file is opened once, concurrently, projected to the covered fields; the
/// value bytes are still not read here (the per-batch [`merge_overlay_batch`] fetches
/// only the values it needs).
pub async fn resolve_overlays(
    planner: &OverlayReadPlanner,
    offsets_in_frag: &[u32],
    fragment: &FileFragment,
    read_config: &FragReadConfig,
) -> Result<Vec<FieldOverlayPlan>> {
    let read_offsets = read_offsets_bitmap(offsets_in_frag);

    // A file is opened only if some field it covers has cells among the requested
    // rows. This is the row-selection pruning: overlays outside the read are skipped.
    let mut file_needed = vec![false; planner.files.len()];
    for field in &planner.fields {
        for overlay in &field.overlays_newest_first {
            if !overlay.coverage.is_disjoint(&read_offsets) {
                file_needed[overlay.file] = true;
            }
        }
    }

    // Open each needed file once, concurrently. The reader is shared (via `Arc`) by
    // every field that file covers.
    //
    // TODO(overlay perf): these reads use the default reader priority. Once we
    // benchmark take/scan over overlays, decide whether overlay value reads should
    // inherit `read_config.reader_priority` (or get a dedicated priority) so they
    // schedule alongside the base reads.
    let opened: Vec<Option<Arc<dyn GenericFileReader>>> =
        futures::future::try_join_all(planner.files.iter().enumerate().map(|(i, file)| {
            let needed = file_needed[i];
            async move {
                if !needed {
                    return Ok::<_, Error>(None);
                }
                Ok(fragment
                    .open_reader(&file.data_file, Some(&file.open_projection), read_config)
                    .await?
                    .map(Arc::from))
            }
        }))
        .await?;

    let mut plans = Vec::new();
    for field in &planner.fields {
        let mut overlays_newest_first = Vec::new();
        for overlay in &field.overlays_newest_first {
            let Some(reader) = &opened[overlay.file] else {
                continue; // pruned: coverage disjoint from the read
            };
            overlays_newest_first.push(LoadedFieldOverlay {
                coverage: overlay.coverage.clone(),
                reader: reader.clone(),
                field_projection: field.field_projection.clone(),
            });
        }
        if !overlays_newest_first.is_empty() {
            plans.push(FieldOverlayPlan {
                field_name: field.field_name.clone(),
                overlays_newest_first,
            });
        }
    }
    Ok(plans)
}

/// The set of `offset_in_frag` a read will return, as a bitmap for cheap
/// intersection against overlay coverages. Contiguous scans build a single range;
/// arbitrary `take` offsets (small batches) are inserted individually.
fn read_offsets_bitmap(offsets_in_frag: &[u32]) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    match contiguous_frag_start(offsets_in_frag) {
        Some(start) => {
            let end = (start as u64 + offsets_in_frag.len() as u64).min(u32::MAX as u64) as u32;
            bitmap.insert_range(start..end);
        }
        None => bitmap.extend(offsets_in_frag.iter().copied()),
    }
    bitmap
}

/// Resolve overlays for one base batch: route each projected field against the
/// batch's `offsets_in_frag`, fetch only the overlay values the batch needs
/// (concurrently with the base read), and assemble the merged columns. Fields with
/// no plan, and the row-id/row-address system columns, pass through.
pub async fn merge_overlay_batch(
    base: ReadBatchFut,
    offsets_in_frag: &[u32],
    plans: &[FieldOverlayPlan],
) -> Result<RecordBatch> {
    let field_work = futures::future::try_join_all(plans.iter().map(|plan| async move {
        let coverages: Vec<&RoaringBitmap> = plan
            .overlays_newest_first
            .iter()
            .map(|overlay| overlay.coverage.as_ref())
            .collect();
        let routing = route_overlays(offsets_in_frag, &coverages);
        if routing.all_fall_through() {
            return Ok::<_, Error>((plan.field_name.as_str(), None));
        }
        let fetched = futures::future::try_join_all(
            plan.overlays_newest_first
                .iter()
                .zip(routing.offsets_in_overlay())
                .map(|(overlay, offsets_in_overlay)| {
                    fetch_overlay_values(
                        overlay.reader.as_ref(),
                        overlay.field_projection.clone(),
                        offsets_in_overlay,
                    )
                }),
        )
        .await?;
        Ok((plan.field_name.as_str(), Some((routing, fetched))))
    }));

    // The base read and every overlay value read proceed concurrently.
    let (batch, resolved) = futures::future::try_join(base, field_work).await?;

    let schema = batch.schema();
    let mut columns = batch.columns().to_vec();
    for (field_name, work) in resolved {
        let Some((routing, fetched)) = work else {
            continue;
        };
        let Some(idx) = schema.index_of(field_name).ok() else {
            // The plan's field is not in this batch's projection; skip it.
            continue;
        };
        columns[idx] = assemble_overlay_column(&columns[idx], &routing, &fetched)?;
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Fetch one overlay's values at the given `offsets_in_overlay` (sorted, unique):
/// the corresponding entries of its value column. Returns a column of
/// `offsets_in_overlay.len()` values in the same order; empty input reads nothing
/// and returns an empty column.
async fn fetch_overlay_values(
    reader: &dyn GenericFileReader,
    projection: Arc<Schema>,
    offsets_in_overlay: &[u32],
) -> Result<ArrayRef> {
    if offsets_in_overlay.is_empty() {
        return Ok(arrow_array::new_empty_array(
            &projection.fields[0].data_type(),
        ));
    }
    let mut tasks = reader
        .take_all_tasks(
            offsets_in_overlay,
            offsets_in_overlay.len() as u32,
            projection,
            None,
        )
        .await?;
    let mut chunks: Vec<ArrayRef> = Vec::new();
    while let Some(task) = tasks.next().await {
        let batch = task.task.await?;
        chunks.push(batch.column(0).clone());
    }
    let chunk_refs: Vec<&dyn arrow_array::Array> = chunks.iter().map(|a| a.as_ref()).collect();
    Ok(arrow_select::concat::concat(&chunk_refs)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray, UInt32Array};
    use std::sync::Arc;

    fn i32_array(values: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from_iter(values))
    }

    fn bitmap(offsets: impl IntoIterator<Item = u32>) -> RoaringBitmap {
        RoaringBitmap::from_iter(offsets)
    }

    /// Physical offsets for a contiguous range `[start, start + len)`.
    fn offsets(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    /// Drive the production flow purely in memory: route against the coverage
    /// bitmaps, then fetch just the requested `offset_in_overlay` entries from each
    /// overlay's *full* value column (exactly what the value-pushdown `take` does on
    /// disk), then assemble. `overlays_newest_first` holds each overlay's
    /// `(coverage, full value column indexed by offset_in_overlay)`.
    fn resolve(
        base: &ArrayRef,
        offsets: &[u32],
        overlays_newest_first: &[(RoaringBitmap, ArrayRef)],
    ) -> ArrayRef {
        let coverages: Vec<&RoaringBitmap> = overlays_newest_first.iter().map(|(c, _)| c).collect();
        let routing = route_overlays(offsets, &coverages);
        let fetched: Vec<ArrayRef> = overlays_newest_first
            .iter()
            .zip(routing.offsets_in_overlay())
            .map(|((_, full), offsets_in_overlay)| {
                let indices = UInt32Array::from(offsets_in_overlay.clone());
                arrow_select::take::take(full.as_ref(), &indices, None).unwrap()
            })
            .collect();
        assemble_overlay_column(base, &routing, &fetched).unwrap()
    }

    fn assert_i32_eq(actual: &ArrayRef, expected: impl IntoIterator<Item = Option<i32>>) {
        let actual = actual.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(actual, &Int32Array::from_iter(expected));
    }

    #[test]
    fn test_no_overlays_returns_base() {
        let base = i32_array([Some(1), Some(2), Some(3)]);
        let resolved = resolve(&base, &offsets(0, 3), &[]);
        assert_i32_eq(&resolved, [Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn test_single_overlay_value_offset() {
        // Base ages [30, 25, 40, 22]; overlay sets offset_in_frag 1 -> 26, whose
        // value sits at offset_in_overlay 0.
        let base = i32_array([Some(30), Some(25), Some(40), Some(22)]);
        let overlay = (bitmap([1]), i32_array([Some(26)]));
        let resolved = resolve(&base, &offsets(0, 4), &[overlay]);
        assert_i32_eq(&resolved, [Some(30), Some(26), Some(40), Some(22)]);
    }

    #[test]
    fn test_value_offsets_multiple_cells() {
        // Coverage {0, 2, 3} -> values at offset_in_overlay 0, 1, 2.
        let base = i32_array([Some(10), Some(11), Some(12), Some(13)]);
        let overlay = (
            bitmap([0, 2, 3]),
            i32_array([Some(100), Some(120), Some(130)]),
        );
        let resolved = resolve(&base, &offsets(0, 4), &[overlay]);
        assert_i32_eq(&resolved, [Some(100), Some(11), Some(120), Some(130)]);
    }

    #[test]
    fn test_newest_overlay_wins() {
        // Two overlays both cover offset_in_frag 1; the newest (first in the slice)
        // wins.
        let base = i32_array([Some(0), Some(1), Some(2)]);
        let newest = (bitmap([1]), i32_array([Some(999)]));
        let older = (bitmap([1, 2]), i32_array([Some(111), Some(222)]));
        let resolved = resolve(&base, &offsets(0, 3), &[newest, older]);
        // offset 1 -> newest (999); offset 2 -> only older covers it (222).
        assert_i32_eq(&resolved, [Some(0), Some(999), Some(222)]);
    }

    #[test]
    fn test_null_override_vs_fall_through() {
        // A covered offset with a NULL value overrides the cell to NULL; an
        // absent offset falls through to the base.
        let base = i32_array([Some(1), Some(2), Some(3)]);
        let overlay = (bitmap([0]), i32_array([None]));
        let resolved = resolve(&base, &offsets(0, 3), &[overlay]);
        assert_i32_eq(&resolved, [None, Some(2), Some(3)]);
    }

    #[test]
    fn test_physical_start_offset() {
        // The batch covers physical rows [10, 13); the overlay covers offset 11.
        let base = i32_array([Some(0), Some(0), Some(0)]);
        let overlay = (bitmap([11]), i32_array([Some(7)]));
        let resolved = resolve(&base, &offsets(10, 3), &[overlay]);
        assert_i32_eq(&resolved, [Some(0), Some(7), Some(0)]);
    }

    #[test]
    fn test_string_column_merge() {
        let base: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let overlay = (
            bitmap([0, 2]),
            Arc::new(StringArray::from(vec!["A", "C"])) as ArrayRef,
        );
        let resolved = resolve(&base, &offsets(0, 3), &[overlay]);
        let expected: ArrayRef = Arc::new(StringArray::from(vec!["A", "b", "C"]));
        assert_eq!(&resolved, &expected);
    }

    #[test]
    fn test_non_contiguous_offsets() {
        // `take` supplies arbitrary, non-contiguous offsets_in_frag. The base rows
        // correspond to offsets 5, 1, 8 (in that order); the overlay covers offsets
        // {1, 8}, whose values sit at offset_in_overlay 0, 1.
        let base = i32_array([Some(50), Some(10), Some(80)]);
        let overlay = (bitmap([1, 8]), i32_array([Some(11), Some(88)]));
        let resolved = resolve(&base, &[5, 1, 8], &[overlay]);
        // offset 5 uncovered -> base 50; offset 1 -> offset_in_overlay 0 (11);
        // offset 8 -> offset_in_overlay 1 (88).
        assert_i32_eq(&resolved, [Some(50), Some(11), Some(88)]);
    }

    #[test]
    fn test_routing_dedups_repeated_offsets() {
        // A `take` may request the same offset twice; both rows must route to the
        // same overlay value, and that value is fetched only once.
        let coverage = bitmap([2, 5]);
        let routing = route_overlays(&[5, 2, 5], &[&coverage]);
        // offset_in_frag 5 is offset_in_overlay 1, offset_in_frag 2 is
        // offset_in_overlay 0: distinct values {0, 1}, sorted.
        assert_eq!(routing.offsets_in_overlay(), &[vec![0, 1]]);
        let full = i32_array([Some(20), Some(50)]); // values at offset_in_overlay 0, 1
        let fetched = vec![
            arrow_select::take::take(
                full.as_ref(),
                &UInt32Array::from(routing.offsets_in_overlay()[0].clone()),
                None,
            )
            .unwrap(),
        ];
        let base = i32_array([Some(0), Some(0), Some(0)]);
        let resolved = assemble_overlay_column(&base, &routing, &fetched).unwrap();
        assert_i32_eq(&resolved, [Some(50), Some(20), Some(50)]);
    }

    #[test]
    fn test_assemble_value_count_mismatch_errors() {
        let coverage = bitmap([0, 1]);
        let routing = route_overlays(&[0, 1], &[&coverage]);
        let base = i32_array([Some(1), Some(2)]);
        // One value supplied for two requested offsets is a caller bug.
        let fetched = vec![i32_array([Some(9)])];
        assert!(assemble_overlay_column(&base, &routing, &fetched).is_err());
    }

    #[test]
    fn test_contiguous_fast_path_matches_general() {
        // The contiguous fast path must produce byte-for-byte identical routing to
        // the general offset-major path for any contiguous batch. Fuzz a range of
        // fragment starts, lengths, overlay counts, and coverage densities —
        // including bits outside the batch range — and compare both paths.
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..500 {
            let frag_start = next() % 64;
            let len = (next() % 48 + 1) as usize;
            let num_overlays = (next() % 5) as usize;
            let coverages: Vec<RoaringBitmap> = (0..num_overlays)
                .map(|_| {
                    let density = next() % 101;
                    let mut b = RoaringBitmap::new();
                    for off in frag_start.saturating_sub(3)..frag_start + len as u32 + 3 {
                        if next() % 100 < density {
                            b.insert(off);
                        }
                    }
                    b
                })
                .collect();
            let refs: Vec<&RoaringBitmap> = coverages.iter().collect();
            let contiguous_offsets: Vec<u32> = (frag_start..frag_start + len as u32).collect();

            let fast = route_contiguous(frag_start, len, &refs);
            let general = route_arbitrary(&contiguous_offsets, &refs);
            assert_eq!(fast.indices, general.indices, "indices differ");
            assert_eq!(
                fast.offsets_in_overlay, general.offsets_in_overlay,
                "offsets_in_overlay differ"
            );
            assert_eq!(fast.any_overlay, general.any_overlay, "any_overlay differs");
        }
    }

    #[test]
    fn test_overlay_ordering_newest_first() {
        use lance_table::format::DataFile;
        use lance_table::format::overlay::OverlayCoverage;
        let mk = |version: u64| DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", vec![1], None),
            coverage: OverlayCoverage::dense(RoaringBitmap::new()),
            committed_version: version,
        };
        // List order [v2, v5, v3]; newest-first should be v5(idx1), v3(idx2), v2(idx0).
        let overlays = vec![mk(2), mk(5), mk(3)];
        assert_eq!(overlay_indices_newest_first(&overlays), vec![1, 2, 0]);

        // Equal versions: later list position is newer.
        let overlays = vec![mk(4), mk(4)];
        assert_eq!(overlay_indices_newest_first(&overlays), vec![1, 0]);
    }
}
