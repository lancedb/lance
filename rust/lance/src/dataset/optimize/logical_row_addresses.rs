// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use futures::{StreamExt, TryStreamExt};
use lance_core::utils::address::{LogicalRowAddress, RowAddress};
use lance_core::{Error, Result};
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::{
    Fragment, LogicalRowAddressRange, LogicalRowAddressSelection, PlacementMaintenanceRequired,
    RowAddressLayout, RowAddressLayoutDelta, RowAddressLogicalDomain, RowAddressPlacement,
    RowAddressPlacementDelta, RowAddressPlacementKind, RowAddressRouter, RowAddressTargetFragment,
    RowAddressTargetRange, RowSequenceFingerprintBuilder,
};
use lance_table::rowids::{RowIdSequence, read_row_ids, rechunk_sequences, write_row_ids};
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::Dataset;
use crate::io::deletion::read_dataset_deletion_file;

pub(super) struct PlannedDefaultCompactionRows {
    pub logical_row_ids: Vec<u8>,
    pub retired_logical_row_ids: Option<Vec<u8>>,
    pub current_deletion_vectors: BTreeMap<u32, RoaringBitmap>,
    /// Exclusive source-fragment boundaries for monotonic input runs that
    /// must be merged by logical row address. Empty when the physical input is
    /// already globally ordered.
    pub logical_run_ends: Vec<u32>,
    /// True only when a source physical fragment is internally out of logical
    /// order and cannot participate in a sort-preserving run merge.
    pub requires_full_logical_sort: bool,
}

struct PhysicalLogicalSequence {
    destination_start: u32,
    physical_length: u32,
    logical_rows: RowIdSequence,
    excluded_physical_offsets: RoaringBitmap,
}

fn maintenance_required(reason: PlacementMaintenanceRequired) -> Error {
    Error::not_supported_source(Box::new(reason))
}

fn logical_range_sequence(
    logical_fragment_id: u32,
    start_slot: u32,
    end_slot: u32,
) -> Result<RowIdSequence> {
    if start_slot >= end_slot {
        return Ok(RowIdSequence::new());
    }
    let start = LogicalRowAddress::try_new_from_parts(logical_fragment_id, start_slot)?.raw();
    LogicalRowAddress::try_new_from_parts(logical_fragment_id, end_slot - 1)?;
    Ok(RowIdSequence::from(
        start..start + (end_slot - start_slot) as u64,
    ))
}

fn selection_sequence(selection: &LogicalRowAddressSelection) -> Result<RowIdSequence> {
    let mut sequence = RowIdSequence::new();
    if let Some(ranges) = selection.to_ranges_bounded(MAX_CLUSTERED_DELETION_RUNS)? {
        for range in ranges {
            sequence.extend(logical_range_sequence(
                range.logical_fragment_id,
                range.start_slot,
                range.end_slot,
            )?);
        }
    } else {
        let mut chunk = Vec::with_capacity(MAX_CLUSTERED_DELETION_RUNS);
        selection.try_for_each_address(|address| {
            chunk.push(address.raw());
            if chunk.len() == MAX_CLUSTERED_DELETION_RUNS {
                sequence.extend(RowIdSequence::from(chunk.as_slice()));
                chunk.clear();
            }
            Ok(())
        })?;
        if !chunk.is_empty() {
            sequence.extend(RowIdSequence::from(chunk.as_slice()));
        }
    }
    if sequence.len() != selection.cardinality() {
        return Err(Error::invalid_input(
            "logical selection ranges disagree with selection cardinality",
        ));
    }
    Ok(sequence)
}

fn shifted_physical_offsets(slots: &RoaringBitmap, offset: u32) -> Result<RoaringBitmap> {
    if offset == 0 {
        return Ok(slots.clone());
    }
    let mut shifted = RoaringBitmap::new();
    let mut runs = slots.iter();
    while let Some(run) = runs.next_range() {
        let start = run
            .start()
            .checked_add(offset)
            .ok_or_else(|| Error::format_capacity_exceeded("physical offset exceeds u32"))?;
        let end = run
            .end()
            .checked_add(offset)
            .ok_or_else(|| Error::format_capacity_exceeded("physical offset exceeds u32"))?;
        shifted.insert_range(start..=end);
    }
    Ok(shifted)
}

fn insert_physical_offset_range(offsets: &mut RoaringBitmap, start: u64, end: u64) -> Result<()> {
    let start = u32::try_from(start)
        .map_err(|_| Error::format_capacity_exceeded("excluded physical offset exceeds u32"))?;
    let end = u32::try_from(end)
        .map_err(|_| Error::format_capacity_exceeded("excluded physical end exceeds u32"))?;
    offsets.insert_range(start..end);
    Ok(())
}

fn direct_excluded_physical_offsets(
    source: &RowAddressLogicalDomain,
    destination_start: u32,
    excluded: Option<&LogicalRowAddressSelection>,
) -> Result<RoaringBitmap> {
    let mut offsets = RoaringBitmap::new();
    let Some(excluded) = excluded else {
        return Ok(offsets);
    };
    let excluded = excluded.to_roaring_treemap()?;
    let mut source_slots = None;
    for (logical_fragment_id, slots) in excluded.bitmaps() {
        if logical_fragment_id != source.logical_fragment_id
            || slots.max().is_some_and(|slot| slot >= source.slot_count)
        {
            return Err(Error::invalid_input(format!(
                "Direct exclusion domain {logical_fragment_id} is outside logical domain {}",
                source.logical_fragment_id
            )));
        }
        source_slots = Some(slots);
    }
    if let Some(source_slots) = source_slots {
        offsets = shifted_physical_offsets(source_slots, destination_start)?;
    }
    Ok(offsets)
}

fn append_selected_excluded_physical_offsets(
    selection: &LogicalRowAddressSelection,
    excluded: Option<&LogicalRowAddressSelection>,
    physical_start: u64,
    offsets: &mut RoaringBitmap,
) -> Result<()> {
    let Some(excluded) = excluded else {
        return Ok(());
    };
    let selected = selection.to_roaring_treemap()?;
    let excluded = excluded.to_roaring_treemap()?;
    let excluded_by_domain = excluded.bitmaps().collect::<BTreeMap<_, _>>();
    let mut observed_excluded_domains = BTreeSet::new();
    let mut selection_prefix = 0_u64;
    for (logical_fragment_id, selected_slots) in selected.bitmaps() {
        if let Some(excluded_slots) = excluded_by_domain.get(&logical_fragment_id) {
            let excluded_slots = *excluded_slots;
            observed_excluded_domains.insert(logical_fragment_id);
            let mut outside = excluded_slots.clone();
            outside -= selected_slots;
            if !outside.is_empty() {
                return Err(Error::invalid_input(format!(
                    "excluded logical domain {logical_fragment_id} is outside its placement selection"
                )));
            }
            if excluded_slots == selected_slots {
                let start = physical_start
                    .checked_add(selection_prefix)
                    .ok_or_else(|| {
                        Error::format_capacity_exceeded("excluded physical offset exceeds u64")
                    })?;
                let end = start.checked_add(selected_slots.len()).ok_or_else(|| {
                    Error::format_capacity_exceeded("excluded physical end exceeds u64")
                })?;
                insert_physical_offset_range(offsets, start, end)?;
            } else {
                let mut runs = excluded_slots.iter();
                while let Some(run) = runs.next_range() {
                    if !selected_slots.contains_range(run.clone()) {
                        return Err(Error::invalid_input(format!(
                            "excluded logical range {logical_fragment_id}:{}..={} is not contiguous in its placement selection",
                            run.start(),
                            run.end(),
                        )));
                    }
                    let first_ordinal = selected_slots
                        .rank(*run.start())
                        .checked_sub(1)
                        .ok_or_else(|| {
                            Error::internal("selected exclusion rank is unexpectedly zero")
                        })?;
                    let end_ordinal = selected_slots.rank(*run.end());
                    let start = physical_start
                        .checked_add(selection_prefix)
                        .and_then(|prefix| prefix.checked_add(first_ordinal))
                        .ok_or_else(|| {
                            Error::format_capacity_exceeded("excluded physical offset exceeds u64")
                        })?;
                    let end = physical_start
                        .checked_add(selection_prefix)
                        .and_then(|prefix| prefix.checked_add(end_ordinal))
                        .ok_or_else(|| {
                            Error::format_capacity_exceeded("excluded physical end exceeds u64")
                        })?;
                    insert_physical_offset_range(offsets, start, end)?;
                }
            }
        }
        selection_prefix = selection_prefix
            .checked_add(selected_slots.len())
            .ok_or_else(|| Error::format_capacity_exceeded("selection prefix exceeds u64"))?;
    }
    if observed_excluded_domains.len() != excluded_by_domain.len() {
        return Err(Error::invalid_input(
            "excluded logical selection references an absent placement domain",
        ));
    }
    Ok(())
}

fn physical_logical_sequences(
    layout: &RowAddressLayout,
    physical_fragment_ids: &BTreeSet<u32>,
) -> Result<BTreeMap<u32, Vec<PhysicalLogicalSequence>>> {
    let mut sequences = BTreeMap::<u32, Vec<PhysicalLogicalSequence>>::new();
    let mut push = |physical_fragment_id: u32, sequence: PhysicalLogicalSequence| {
        if physical_fragment_ids.contains(&physical_fragment_id) {
            sequences
                .entry(physical_fragment_id)
                .or_default()
                .push(sequence);
        }
    };

    for placement in &layout.placements {
        match placement {
            RowAddressPlacement::Direct(value) => push(
                value.destination_fragment_id,
                PhysicalLogicalSequence {
                    destination_start: value.destination_start,
                    physical_length: value.source.slot_count,
                    logical_rows: logical_range_sequence(
                        value.source.logical_fragment_id,
                        0,
                        value.source.slot_count,
                    )?,
                    excluded_physical_offsets: direct_excluded_physical_offsets(
                        &value.source,
                        value.destination_start,
                        value.excluded.as_deref(),
                    )?,
                },
            ),
            RowAddressPlacement::PackedRun(value) => {
                if !physical_fragment_ids.contains(&value.destination_fragment_id) {
                    continue;
                }
                let mut logical_rows = RowIdSequence::new();
                for ordinal in 0..value.domains.domain_count() {
                    let source = value.domains.domain_at(ordinal)?;
                    logical_rows.extend(logical_range_sequence(
                        source.logical_fragment_id,
                        0,
                        source.slot_count,
                    )?);
                }
                let physical_length = u32::try_from(value.domains.total_slot_count()?)
                    .map_err(|_| Error::invalid_input("PackedRun physical length exceeds u32"))?;
                push(
                    value.destination_fragment_id,
                    PhysicalLogicalSequence {
                        destination_start: value.destination_start,
                        physical_length,
                        logical_rows,
                        excluded_physical_offsets: RoaringBitmap::new(),
                    },
                );
            }
            RowAddressPlacement::Selected(value) => {
                if !physical_fragment_ids.contains(&value.destination_fragment_id) {
                    continue;
                }
                let physical_length = u32::try_from(value.selection.cardinality())
                    .map_err(|_| Error::invalid_input("Selected physical length exceeds u32"))?;
                let mut excluded_physical_offsets = RoaringBitmap::new();
                append_selected_excluded_physical_offsets(
                    &value.selection,
                    value.excluded.as_deref(),
                    u64::from(value.destination_start),
                    &mut excluded_physical_offsets,
                )?;
                push(
                    value.destination_fragment_id,
                    PhysicalLogicalSequence {
                        destination_start: value.destination_start,
                        physical_length,
                        logical_rows: selection_sequence(&value.selection)?,
                        excluded_physical_offsets,
                    },
                );
            }
            RowAddressPlacement::ExtentList(value) => {
                for extent in &value.extents {
                    push(
                        extent.destination_fragment_id,
                        PhysicalLogicalSequence {
                            destination_start: extent.destination_start,
                            physical_length: extent.length,
                            logical_rows: logical_range_sequence(
                                value.source.logical_fragment_id,
                                extent.source_start,
                                extent.source_start.checked_add(extent.length).ok_or_else(
                                    || Error::invalid_input("ExtentList source end exceeds u32"),
                                )?,
                            )?,
                            excluded_physical_offsets: RoaringBitmap::new(),
                        },
                    );
                }
            }
            RowAddressPlacement::SparseSelection(value) => {
                if !physical_fragment_ids.contains(&value.destination_fragment_id) {
                    continue;
                }
                let mut logical_rows = RowIdSequence::new();
                let mut excluded_physical_offsets = RoaringBitmap::new();
                let mut source_prefix = 0_u64;
                for source in &value.sources {
                    append_selected_excluded_physical_offsets(
                        &source.selection,
                        source.excluded.as_deref(),
                        u64::from(value.destination_start)
                            .checked_add(source_prefix)
                            .ok_or_else(|| {
                                Error::format_capacity_exceeded(
                                    "SparseSelection physical prefix exceeds u64",
                                )
                            })?,
                        &mut excluded_physical_offsets,
                    )?;
                    logical_rows.extend(selection_sequence(&source.selection)?);
                    source_prefix = source_prefix
                        .checked_add(source.selection.cardinality())
                        .ok_or_else(|| {
                            Error::format_capacity_exceeded(
                                "SparseSelection source cardinality exceeds u64",
                            )
                        })?;
                }
                let physical_length = u32::try_from(logical_rows.len()).map_err(|_| {
                    Error::invalid_input("SparseSelection physical length exceeds u32")
                })?;
                push(
                    value.destination_fragment_id,
                    PhysicalLogicalSequence {
                        destination_start: value.destination_start,
                        physical_length,
                        logical_rows,
                        excluded_physical_offsets,
                    },
                );
            }
            RowAddressPlacement::ExplicitMap(value) => {
                if value.destinations.iter().any(|destination| {
                    physical_fragment_ids.contains(&destination.physical_fragment_id)
                }) {
                    return Err(maintenance_required(
                        PlacementMaintenanceRequired::ExistingExplicitMapRequiresRewrite {
                            logical_fragment_id: value
                                .sources
                                .first()
                                .map(|source| source.source.logical_fragment_id)
                                .unwrap_or(u32::MAX),
                        },
                    ));
                }
            }
        }
    }
    for fragment_sequences in sequences.values_mut() {
        fragment_sequences.sort_by_key(|sequence| sequence.destination_start);
    }
    Ok(sequences)
}

fn validated_logical_ranges(sequence: &RowIdSequence) -> Result<Vec<std::ops::Range<u64>>> {
    let ranges = sequence.contiguous_ranges();
    for range in &ranges {
        if range.is_empty() {
            continue;
        }
        let first = LogicalRowAddress::try_from(range.start)?;
        let last = LogicalRowAddress::try_from(range.end - 1)?;
        if first.logical_fragment_id() != last.logical_fragment_id() {
            return Err(Error::invalid_input(
                "logical row-id sequence range crosses a logical fragment boundary",
            ));
        }
    }
    Ok(ranges)
}

fn canonical_logical_sequence(logical_rows: RowAddrTreeMap) -> Result<RowIdSequence> {
    let mut output = RowIdSequence::new();
    // SAFETY: every entry was constructed from a concrete RowIdSequence, so
    // the map cannot contain an unknown-size Full fragment marker.
    for (logical_fragment_id, slots) in unsafe { logical_rows.iter_runs() } {
        let end_slot = slots
            .end()
            .checked_add(1)
            .ok_or_else(|| Error::format_capacity_exceeded("logical row range end exceeds u32"))?;
        output.extend(logical_range_sequence(
            logical_fragment_id,
            *slots.start(),
            end_slot,
        )?);
    }
    Ok(output)
}

/// Maximum physical deletion runs admitted across one compaction group.
const MAX_CLUSTERED_DELETION_RUNS: usize = 4_096;
/// Serialized retirement payload ceiling for decoding directly to ranges.
/// Larger payloads stream into the final Roaring selection instead.
const MAX_RETIRED_RANGE_DECODE_BYTES: usize = 1 << 20;

/// Discover deletion runs by rank/select instead of visiting every deleted row.
/// `None` denotes high-entropy input, which default compaction must defer to
/// explicit placement maintenance before expanding row provenance.
fn clustered_deletion_ranges(deleted: &RoaringBitmap) -> Result<Option<Vec<Range<u32>>>> {
    let total = deleted.len();
    let mut ordinal = 0_u64;
    let mut ranges = Vec::new();
    while ordinal < total {
        if ranges.len() == MAX_CLUSTERED_DELETION_RUNS {
            return Ok(None);
        }
        let ordinal_u32 = u32::try_from(ordinal)
            .map_err(|_| Error::format_capacity_exceeded("deletion ordinal exceeds u32"))?;
        let start = deleted
            .select(ordinal_u32)
            .ok_or_else(|| Error::internal("Roaring deletion cardinality disagrees with select"))?;
        let remaining = total - ordinal;
        let address_space = u64::from(u32::MAX) - u64::from(start) + 1;
        let mut lower = 1_u64;
        let mut upper = remaining.min(address_space);
        while lower < upper {
            let candidate = lower + (upper - lower).div_ceil(2);
            let selected_ordinal = ordinal
                .checked_add(candidate - 1)
                .ok_or_else(|| Error::format_capacity_exceeded("deletion ordinal exceeds u64"))?;
            let selected_ordinal = u32::try_from(selected_ordinal)
                .map_err(|_| Error::format_capacity_exceeded("deletion ordinal exceeds u32"))?;
            let expected = u64::from(start) + candidate - 1;
            if deleted.select(selected_ordinal).map(u64::from) == Some(expected) {
                lower = candidate;
            } else {
                upper = candidate - 1;
            }
        }
        let end = u32::try_from(u64::from(start) + lower)
            .map_err(|_| Error::format_capacity_exceeded("deletion range end exceeds u32"))?;
        ranges.push(start..end);
        ordinal += lower;
    }
    Ok(Some(ranges))
}

fn admitted_deletion_ranges(
    deleted: &RoaringBitmap,
    logical_fragment_id: u32,
    group_deletion_entropy: &mut usize,
) -> Result<Vec<Range<u32>>> {
    let ranges = clustered_deletion_ranges(deleted)?.ok_or_else(|| {
        maintenance_required(
            PlacementMaintenanceRequired::SelectionSubtractionRequiresRewrite {
                logical_fragment_id,
            },
        )
    })?;
    charge_deletion_entropy(
        ranges.len().saturating_sub(1),
        logical_fragment_id,
        group_deletion_entropy,
    )?;
    Ok(ranges)
}

fn charge_deletion_entropy(
    additional_runs: usize,
    logical_fragment_id: u32,
    group_deletion_entropy: &mut usize,
) -> Result<()> {
    let projected_runs = group_deletion_entropy
        .checked_add(additional_runs)
        .ok_or_else(|| {
            Error::format_capacity_exceeded("compaction deletion run count exceeds usize")
        })?;
    if projected_runs > MAX_CLUSTERED_DELETION_RUNS {
        return Err(maintenance_required(
            PlacementMaintenanceRequired::SelectionSubtractionRequiresRewrite {
                logical_fragment_id,
            },
        ));
    }
    *group_deletion_entropy = projected_runs;
    Ok(())
}

fn slice_row_id_sequence(
    sequence: &RowIdSequence,
    offset: u32,
    length: u32,
) -> Result<RowIdSequence> {
    if length == 0 {
        return Ok(RowIdSequence::new());
    }
    let total = sequence.len();
    let offset = u64::from(offset);
    let length = u64::from(length);
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::format_capacity_exceeded("row-id slice end exceeds u64"))?;
    if end > total {
        return Err(Error::invalid_input(format!(
            "row-id slice {offset}..{end} exceeds sequence length {total}"
        )));
    }

    let mut chunk_sizes = Vec::with_capacity(3);
    let mut selected_chunk = 0;
    if offset != 0 {
        chunk_sizes.push(offset);
        selected_chunk = 1;
    }
    chunk_sizes.push(length);
    if end != total {
        chunk_sizes.push(total - end);
    }
    let mut chunks = rechunk_sequences([sequence.clone()], chunk_sizes, false)?;
    Ok(chunks.remove(selected_chunk))
}

fn append_sequence_range(
    sequence: &RowIdSequence,
    sequence_physical_start: u32,
    physical_range: Range<u32>,
    output: &mut RowIdSequence,
) -> Result<()> {
    let local_start = physical_range
        .start
        .checked_sub(sequence_physical_start)
        .ok_or_else(|| Error::internal("physical row range precedes its placement sequence"))?;
    let length = physical_range.end - physical_range.start;
    output.extend(slice_row_id_sequence(sequence, local_start, length)?);
    Ok(())
}

fn append_intersections(
    sequence: &PhysicalLogicalSequence,
    ranges: &[Range<u32>],
    output: &mut RowIdSequence,
) -> Result<()> {
    let sequence_end = sequence
        .destination_start
        .checked_add(sequence.physical_length)
        .ok_or_else(|| Error::invalid_input("placement physical range end exceeds u32"))?;
    for range in ranges {
        let start = range.start.max(sequence.destination_start);
        let end = range.end.min(sequence_end);
        if start < end {
            append_sequence_range(
                &sequence.logical_rows,
                sequence.destination_start,
                start..end,
                output,
            )?;
        }
    }
    Ok(())
}

fn append_complement(
    sequence: &PhysicalLogicalSequence,
    deleted: &[Range<u32>],
    output: &mut RowIdSequence,
) -> Result<()> {
    let sequence_end = sequence
        .destination_start
        .checked_add(sequence.physical_length)
        .ok_or_else(|| Error::invalid_input("placement physical range end exceeds u32"))?;
    let mut cursor = sequence.destination_start;
    for range in deleted {
        if range.end <= cursor {
            continue;
        }
        if range.start >= sequence_end {
            break;
        }
        let deleted_start = range.start.max(sequence.destination_start);
        let deleted_end = range.end.min(sequence_end);
        if cursor < deleted_start {
            append_sequence_range(
                &sequence.logical_rows,
                sequence.destination_start,
                cursor..deleted_start,
                output,
            )?;
        }
        cursor = cursor.max(deleted_end);
    }
    if cursor < sequence_end {
        append_sequence_range(
            &sequence.logical_rows,
            sequence.destination_start,
            cursor..sequence_end,
            output,
        )?;
    }
    Ok(())
}

fn append_sequence_provenance(
    sequence: &PhysicalLogicalSequence,
    deleted: &RoaringBitmap,
    group_deletion_entropy: &mut usize,
    retired: &mut RowIdSequence,
    live: &mut RowIdSequence,
) -> Result<()> {
    let logical_fragment_id = sequence
        .logical_rows
        .iter()
        .next()
        .map(LogicalRowAddress::try_from)
        .transpose()?
        .map(|address| address.logical_fragment_id())
        .unwrap_or(u32::MAX);
    let sequence_end = sequence
        .destination_start
        .checked_add(sequence.physical_length)
        .ok_or_else(|| Error::invalid_input("placement physical range end exceeds u32"))?;
    let mut sequence_offsets = RoaringBitmap::new();
    sequence_offsets.insert_range(sequence.destination_start..sequence_end);
    let mut owned_deleted = deleted & &sequence_offsets;
    owned_deleted -= &sequence.excluded_physical_offsets;
    let owned_deletion_ranges =
        admitted_deletion_ranges(&owned_deleted, logical_fragment_id, group_deletion_entropy)?;

    let mut retired_part = RowIdSequence::new();
    append_intersections(sequence, &owned_deletion_ranges, &mut retired_part)?;
    let mut live_part = RowIdSequence::new();
    append_complement(sequence, &owned_deletion_ranges, &mut live_part)?;

    if !sequence.excluded_physical_offsets.is_empty() {
        let mut live_mask = Vec::new();
        let mut deletion_range_index = 0_usize;
        let mut deleted_before = 0_u32;
        for excluded_offset in sequence
            .excluded_physical_offsets
            .range(sequence.destination_start..sequence_end)
        {
            while deletion_range_index < owned_deletion_ranges.len()
                && owned_deletion_ranges[deletion_range_index].end <= excluded_offset
            {
                deleted_before = deleted_before
                    .checked_add(
                        owned_deletion_ranges[deletion_range_index].end
                            - owned_deletion_ranges[deletion_range_index].start,
                    )
                    .ok_or_else(|| Error::format_capacity_exceeded("deleted prefix exceeds u32"))?;
                deletion_range_index += 1;
            }
            if owned_deletion_ranges
                .get(deletion_range_index)
                .is_some_and(|range| range.contains(&excluded_offset))
            {
                return Err(Error::internal(
                    "owned deletion still contains an excluded physical offset",
                ));
            }
            let local_offset = excluded_offset - sequence.destination_start;
            live_mask.push(local_offset.checked_sub(deleted_before).ok_or_else(|| {
                Error::internal("excluded physical offset precedes its deleted prefix")
            })?);
        }
        live_part.mask(live_mask)?;
    }

    retired.extend(retired_part);
    live.extend(live_part);
    Ok(())
}

fn sequences_have_overlapping_physical_ranges(
    sequences: &[PhysicalLogicalSequence],
) -> Result<bool> {
    let mut previous_end = None;
    for sequence in sequences {
        let sequence_end = sequence
            .destination_start
            .checked_add(sequence.physical_length)
            .ok_or_else(|| Error::invalid_input("placement physical range end exceeds u32"))?;
        if previous_end.is_some_and(|end| sequence.destination_start < end) {
            return Ok(true);
        }
        previous_end = Some(sequence_end);
    }
    Ok(false)
}

fn append_fragment_provenance(
    sequences: &[PhysicalLogicalSequence],
    deleted: &RoaringBitmap,
    group_deletion_entropy: &mut usize,
    retired: &mut RowIdSequence,
    live: &mut RowIdSequence,
) -> Result<()> {
    if !sequences_have_overlapping_physical_ranges(sequences)? {
        for sequence in sequences {
            append_sequence_provenance(sequence, deleted, group_deletion_entropy, retired, live)?;
        }
        return Ok(());
    }

    // Fragment reuse can place a replacement owner into a hole excluded from
    // an older placement.  Placement order is not physical row order, so merge
    // the active owner ranges before deciding whether compaction must sort.
    let mut claimed_offsets = RoaringBitmap::new();
    let mut live_ranges = Vec::<(Range<u32>, usize)>::new();
    for (sequence_index, sequence) in sequences.iter().enumerate() {
        let logical_fragment_id = sequence
            .logical_rows
            .iter()
            .next()
            .map(LogicalRowAddress::try_from)
            .transpose()?
            .map(|address| address.logical_fragment_id())
            .unwrap_or(u32::MAX);
        let sequence_end = sequence
            .destination_start
            .checked_add(sequence.physical_length)
            .ok_or_else(|| Error::invalid_input("placement physical range end exceeds u32"))?;
        let mut active_offsets = RoaringBitmap::new();
        active_offsets.insert_range(sequence.destination_start..sequence_end);
        active_offsets -= &sequence.excluded_physical_offsets;

        if !(&claimed_offsets & &active_offsets).is_empty() {
            return Err(Error::internal(
                "storage-version-2.3 compaction found overlapping physical row ownership",
            ));
        }
        claimed_offsets |= &active_offsets;

        let owned_deleted = deleted & &active_offsets;
        let owned_deletion_ranges =
            admitted_deletion_ranges(&owned_deleted, logical_fragment_id, group_deletion_entropy)?;
        append_intersections(sequence, &owned_deletion_ranges, retired)?;

        active_offsets -= deleted;
        let mut ranges = active_offsets.iter();
        while let Some(range) = ranges.next_range() {
            let end = range.end().checked_add(1).ok_or_else(|| {
                Error::format_capacity_exceeded("active physical range end exceeds u32")
            })?;
            live_ranges.push((*range.start()..end, sequence_index));
        }
    }

    live_ranges.sort_unstable_by_key(|(range, _)| range.start);
    for (range, sequence_index) in live_ranges {
        let sequence = &sequences[sequence_index];
        append_sequence_range(
            &sequence.logical_rows,
            sequence.destination_start,
            range,
            live,
        )?;
    }
    Ok(())
}

/// Build default-compaction identity provenance using only manifest placement
/// metadata and deletion vectors. No data file, footer, or user column is read.
pub(super) async fn plan_default_compaction_rows(
    dataset: &Dataset,
    fragments: &[Fragment],
) -> Result<PlannedDefaultCompactionRows> {
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| Error::internal("storage-version-2.3 manifest has no row-address layout"))?;
    let physical_fragment_ids = fragments
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id).map_err(|_| {
                Error::invalid_input(format!(
                    "physical fragment id {} exceeds row-address capacity",
                    fragment.id
                ))
            })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let fragment_states = futures::stream::iter(fragments.iter().cloned())
        .map(|fragment| async move {
            let deleted = if let Some(deletion_file) = fragment.deletion_file.as_ref() {
                let deletion_vector =
                    read_dataset_deletion_file(dataset, fragment.id, deletion_file).await?;
                RoaringBitmap::from(deletion_vector.as_ref())
            } else {
                RoaringBitmap::new()
            };
            Ok::<_, Error>((fragment, deleted))
        })
        .buffered(dataset.object_store.as_ref().io_parallelism())
        .try_collect::<Vec<_>>()
        .await?;
    let fragment_states = fragment_states
        .into_iter()
        .map(|(fragment, deleted)| {
            let physical_rows = u32::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "storage-version-2.3 compaction source fragment {} is missing physical_rows",
                    fragment.id
                ))
            })?)
            .map_err(|_| Error::invalid_input("fragment physical_rows exceeds u32"))?;
            if deleted.max().is_some_and(|offset| offset >= physical_rows) {
                return Err(Error::invalid_input(format!(
                    "deletion vector for fragment {} exceeds physical_rows {}",
                    fragment.id, physical_rows
                )));
            }
            Ok((fragment, physical_rows, deleted))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut placement_sequences = physical_logical_sequences(layout, &physical_fragment_ids)?;

    let mut canonical_live_rows = RowAddrTreeMap::new();
    let mut current_run_end = None;
    let mut logical_run_ends = Vec::new();
    let mut requires_full_logical_sort = false;
    let mut retired = RowIdSequence::new();
    let mut current_deletion_vectors = BTreeMap::new();
    let mut group_deletion_entropy = 0_usize;

    for (fragment_index, (fragment, physical_rows, deleted)) in
        fragment_states.into_iter().enumerate()
    {
        let fragment_id = u32::try_from(fragment.id).map_err(|_| {
            Error::invalid_input(format!(
                "physical fragment id {} exceeds row-address capacity",
                fragment.id
            ))
        })?;
        let mut fragment_sequence = RowIdSequence::new();
        if let Some(native) = fragment.native_logical_domain {
            let deleted_ranges = admitted_deletion_ranges(
                &deleted,
                native.logical_fragment_id,
                &mut group_deletion_entropy,
            )?;
            let mut cursor = 0_u32;
            for range in &deleted_ranges {
                if cursor < range.start {
                    fragment_sequence.extend(logical_range_sequence(
                        native.logical_fragment_id,
                        cursor,
                        range.start,
                    )?);
                }
                retired.extend(logical_range_sequence(
                    native.logical_fragment_id,
                    range.start,
                    range.end,
                )?);
                cursor = range.end;
            }
            if cursor < physical_rows {
                fragment_sequence.extend(logical_range_sequence(
                    native.logical_fragment_id,
                    cursor,
                    physical_rows,
                )?);
            }
        } else {
            layout.verify_visibility(&fragment, &deleted)?;
            let sequences = placement_sequences.remove(&fragment_id).unwrap_or_default();
            append_fragment_provenance(
                &sequences,
                &deleted,
                &mut group_deletion_entropy,
                &mut retired,
                &mut fragment_sequence,
            )?;
        }

        let fragment_len = fragment_sequence.len();
        let is_fragment_monotonic = fragment_sequence.is_strictly_sorted();
        let first = fragment_sequence.get(0);
        let last = usize::try_from(fragment_len)
            .ok()
            .and_then(|len| len.checked_sub(1))
            .and_then(|last| fragment_sequence.get(last));
        if !is_fragment_monotonic {
            requires_full_logical_sort = true;
        } else if let (Some(run_end), Some(first)) = (current_run_end, first)
            && run_end >= first
        {
            logical_run_ends.push(u32::try_from(fragment_index).map_err(|_| {
                Error::invalid_input("compaction source fragment count exceeds u32")
            })?);
            current_run_end = last;
        } else if let Some(last) = last {
            current_run_end = Some(last);
        }

        let fragment_rows = RowAddrTreeMap::from(&fragment_sequence);
        let canonical_rows_before = canonical_live_rows
            .len()
            .ok_or_else(|| Error::internal("logical live-row cardinality is unknown"))?;
        canonical_live_rows |= fragment_rows;
        let canonical_rows_after = canonical_live_rows
            .len()
            .ok_or_else(|| Error::internal("logical live-row cardinality is unknown"))?;
        if canonical_rows_after.checked_sub(canonical_rows_before) != Some(fragment_len) {
            return Err(Error::internal(format!(
                "storage-version-2.3 compaction found duplicate logical ownership in physical fragment {fragment_id}"
            )));
        }

        if !deleted.is_empty() {
            current_deletion_vectors.insert(fragment_id, deleted);
        }
    }

    if requires_full_logical_sort {
        logical_run_ends.clear();
    } else if !logical_run_ends.is_empty() {
        logical_run_ends.push(
            u32::try_from(fragments.len()).map_err(|_| {
                Error::invalid_input("compaction source fragment count exceeds u32")
            })?,
        );
    }
    let logical_sequence = canonical_logical_sequence(canonical_live_rows)?;

    let retired_logical_row_ids = (!retired.is_empty()).then(|| write_row_ids(&retired));

    Ok(PlannedDefaultCompactionRows {
        logical_row_ids: write_row_ids(&logical_sequence),
        retired_logical_row_ids,
        current_deletion_vectors,
        logical_run_ends,
        requires_full_logical_sort,
    })
}

pub(super) fn validate_default_compaction_logical_order(sequence: &RowIdSequence) -> Result<()> {
    let mut previous = None;
    for range in sequence.contiguous_ranges() {
        let first = LogicalRowAddress::try_from(range.start)?;
        let last = LogicalRowAddress::try_from(range.end - 1)?;
        if first.logical_fragment_id() != last.logical_fragment_id() {
            return Err(Error::invalid_input(
                "logical row-id sequence range crosses a logical fragment boundary",
            ));
        }
        if let Some(previous_address) = previous
            && previous_address >= range.start
        {
            return Err(maintenance_required(
                PlacementMaintenanceRequired::LogicalOrderRequiresRewrite {
                    previous_address,
                    next_address: range.start,
                },
            ));
        }
        previous = Some(range.end - 1);
    }
    Ok(())
}

struct LogicalRangeCursor {
    ranges: Vec<std::ops::Range<u64>>,
    index: usize,
}

impl LogicalRangeCursor {
    fn new(sequence: &RowIdSequence) -> Self {
        Self {
            ranges: sequence.contiguous_ranges(),
            index: 0,
        }
    }

    fn take(&mut self, mut row_count: u32) -> Result<Vec<LogicalRowAddressRange>> {
        let mut output = Vec::new();
        while row_count > 0 {
            let range = self.ranges.get_mut(self.index).ok_or_else(|| {
                Error::invalid_input(
                    "captured logical output sequence is shorter than compaction output rows",
                )
            })?;
            let available = range.end - range.start;
            let take = available.min(row_count as u64);
            let start = LogicalRowAddress::try_from(range.start)?;
            let last = LogicalRowAddress::try_from(range.start + take - 1)?;
            if start.logical_fragment_id() != last.logical_fragment_id() {
                return Err(Error::invalid_input(
                    "logical row-id sequence range crosses a logical fragment boundary",
                ));
            }
            output.push(LogicalRowAddressRange::new(
                start.logical_fragment_id(),
                start.immutable_slot(),
                last.immutable_slot().checked_add(1).ok_or_else(|| {
                    Error::invalid_input("logical row-address range end exceeds u32")
                })?,
            ));
            range.start += take;
            row_count -= take as u32;
            if range.start == range.end {
                self.index += 1;
            }
        }
        Ok(output)
    }

    fn is_empty(&self) -> bool {
        self.ranges[self.index..]
            .iter()
            .all(|range| range.is_empty())
    }
}

fn register_source_domain(
    router: &RowAddressRouter,
    source_domains: &mut BTreeMap<u32, RowAddressLogicalDomain>,
    logical_fragment_id: u32,
) -> Result<()> {
    let domain = router.logical_domain(logical_fragment_id)?.ok_or_else(|| {
        Error::invalid_input(format!(
            "compaction retired unknown logical fragment {logical_fragment_id}"
        ))
    })?;
    if source_domains
        .insert(logical_fragment_id, domain)
        .is_some_and(|previous| previous != domain)
    {
        return Err(Error::invalid_input(format!(
            "logical fragment {logical_fragment_id} has inconsistent source metadata"
        )));
    }
    Ok(())
}

pub(super) async fn retired_logical_row_ids(
    dataset: &Dataset,
    fragments: &[Fragment],
) -> Result<Option<Vec<u8>>> {
    let mut retired = RoaringTreemap::new();
    let mut physical = Vec::with_capacity(4096);
    for fragment in fragments {
        let Some(deletion_file) = &fragment.deletion_file else {
            continue;
        };
        let fragment_id = u32::try_from(fragment.id).map_err(|_| {
            Error::invalid_input(format!(
                "physical fragment id {} exceeds row-address capacity",
                fragment.id
            ))
        })?;
        let deletions = read_dataset_deletion_file(dataset, fragment.id, deletion_file).await?;
        for offset in deletions.to_sorted_iter() {
            physical.push(RowAddress::new_from_parts(fragment_id, offset));
            if physical.len() == 4096 {
                retired.extend(
                    dataset
                        .resolve_physical_row_ids_async(&physical)
                        .await?
                        .into_iter()
                        .flatten()
                        .map(LogicalRowAddress::raw),
                );
                physical.clear();
            }
        }
    }
    if !physical.is_empty() {
        retired.extend(
            dataset
                .resolve_physical_row_ids_async(&physical)
                .await?
                .into_iter()
                .flatten()
                .map(LogicalRowAddress::raw),
        );
    }
    if retired.is_empty() {
        Ok(None)
    } else {
        let mut sequence = RowIdSequence::new();
        let mut row_ids = Vec::with_capacity(4096);
        for row_id in retired.iter() {
            row_ids.push(row_id);
            if row_ids.len() == 4096 {
                sequence.extend(RowIdSequence::from(row_ids.as_slice()));
                row_ids.clear();
            }
        }
        if !row_ids.is_empty() {
            sequence.extend(RowIdSequence::from(row_ids.as_slice()));
        }
        Ok(Some(write_row_ids(&sequence)))
    }
}

pub(super) fn add_rewrite_provenance(
    dataset: &Dataset,
    new_fragments: &[Fragment],
    logical_row_ids: &[u8],
    retired_row_ids: Option<&[u8]>,
    next_new_fragment_ordinal: &mut u32,
    source_domains: &mut BTreeMap<u32, RowAddressLogicalDomain>,
    delta: &mut RowAddressLayoutDelta,
) -> Result<()> {
    let router = dataset.row_address_router()?;
    let sequence = read_row_ids(logical_row_ids)?;
    validate_default_compaction_logical_order(&sequence)?;
    let mut rows = LogicalRangeCursor::new(&sequence);

    for fragment in new_fragments {
        if fragment.row_id_meta.is_some() || fragment.native_logical_domain.is_some() {
            return Err(Error::invalid_input(
                "storage-version-2.3 speculative compaction output contains row identity metadata",
            ));
        }
        let row_count = u32::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input("compaction output fragment is missing physical_rows")
        })?)
        .map_err(|_| Error::invalid_input("compaction output physical_rows exceeds u32"))?;
        if row_count == 0 {
            return Err(Error::invalid_input(
                "storage-version-2.3 compaction output fragment must not be empty",
            ));
        }
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(*next_new_fragment_ordinal),
            start_offset: 0,
            end_offset: row_count,
        };
        *next_new_fragment_ordinal = next_new_fragment_ordinal
            .checked_add(1)
            .ok_or_else(|| Error::invalid_input("compaction output ordinal overflow"))?;

        let mut fingerprint = RowSequenceFingerprintBuilder::new(target);
        let mut stats = BTreeMap::<u32, (u64, u32, u32)>::new();
        let ranges = rows.take(row_count)?;
        for range in &ranges {
            fingerprint.update_range(*range)?;
            let count = range.len();
            let last = range.end_slot - 1;
            stats
                .entry(range.logical_fragment_id)
                .and_modify(|(total, first, previous_last)| {
                    *total += count;
                    *first = (*first).min(range.start_slot);
                    *previous_last = (*previous_last).max(last);
                })
                .or_insert((count, range.start_slot, last));
        }
        let selection = LogicalRowAddressSelection::from_ranges(ranges)?;
        let mut all_domains_full = true;
        for (logical_fragment_id, (count, first, last)) in &stats {
            let domain = router
                .logical_domain(*logical_fragment_id)?
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "compaction emitted unknown logical fragment {logical_fragment_id}"
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
            output_cardinality: row_count as u64,
            output_row_sequence_fingerprint: fingerprint.finish(row_count as u64)?,
        });
    }
    if !rows.is_empty() {
        return Err(Error::invalid_input(
            "captured logical output sequence is longer than compaction output rows",
        ));
    }

    if let Some(retired_row_ids) = retired_row_ids {
        let retired = read_row_ids(retired_row_ids)?;
        if !retired.is_empty() {
            let selection = if retired_row_ids.len() <= MAX_RETIRED_RANGE_DECODE_BYTES {
                let raw_ranges = validated_logical_ranges(&retired)?;
                let mut ranges = Vec::with_capacity(raw_ranges.len());
                for range in raw_ranges {
                    let first = LogicalRowAddress::try_from(range.start)?;
                    let last = LogicalRowAddress::try_from(range.end - 1)?;
                    register_source_domain(&router, source_domains, first.logical_fragment_id())?;
                    ranges.push(LogicalRowAddressRange::new(
                        first.logical_fragment_id(),
                        first.immutable_slot(),
                        last.immutable_slot().checked_add(1).ok_or_else(|| {
                            Error::format_capacity_exceeded(
                                "retired logical row range end exceeds u32",
                            )
                        })?,
                    ));
                }
                LogicalRowAddressSelection::from_ranges(ranges)?
            } else {
                // High-entropy retirement is information-theoretically large.
                // Stream it into the final compressed bitmap without a second
                // full-size range vector or repeated domain lookups.
                let mut bitmap = RoaringTreemap::new();
                let mut observed_domains = BTreeSet::new();
                for raw in retired.iter() {
                    let address = LogicalRowAddress::try_from(raw)?;
                    if observed_domains.insert(address.logical_fragment_id()) {
                        register_source_domain(
                            &router,
                            source_domains,
                            address.logical_fragment_id(),
                        )?;
                    }
                    bitmap.insert(raw);
                }
                LogicalRowAddressSelection::from_bitmap(bitmap)?
            };
            delta.retired_selections.push(selection);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_table::format::{
        DirectRowAddressPlacement, PackedRunRowAddressPlacement, RowAddressPlacement,
        SelectedRowAddressPlacement,
    };
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn hundred_million_row_native_and_packed_planning_is_range_bounded() {
        const ROWS: u32 = 100_000_000;
        let started = std::time::Instant::now();
        let native = logical_range_sequence(0, 0, ROWS).unwrap();
        assert_eq!(native.len(), ROWS as u64);
        assert!(write_row_ids(&native).len() < 128);

        let sources = (0..100_u32)
            .map(|logical_fragment_id| {
                RowAddressLogicalDomain::new(logical_fragment_id, 1_000_000, 1).unwrap()
            })
            .collect::<Vec<_>>();
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::PackedRun(
            PackedRunRowAddressPlacement::from_sources(sources, 7, 0).unwrap(),
        )];
        let mut sequences = physical_logical_sequences(&layout, &BTreeSet::from([7])).unwrap();
        let packed = sequences.remove(&7).unwrap().pop().unwrap();
        assert_eq!(packed.logical_rows.len(), ROWS as u64);
        assert!(write_row_ids(&packed.logical_rows).len() < 4096);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "100M native/PackedRun metadata planning must not scale with row count"
        );
    }

    #[test]
    fn hundred_million_row_full_domain_retirement_is_range_bounded() {
        const ROWS: u32 = 100_000_000;
        let started = std::time::Instant::now();
        let mut deleted = RoaringBitmap::new();
        deleted.insert_range(0..ROWS);

        let ranges = clustered_deletion_ranges(&deleted).unwrap().unwrap();
        assert_eq!(ranges, vec![0..ROWS]);
        let mut retired = RowIdSequence::new();
        for range in ranges {
            retired.extend(logical_range_sequence(7, range.start, range.end).unwrap());
        }

        assert_eq!(retired.len(), u64::from(ROWS));
        assert!(write_row_ids(&retired).len() < 128);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "100M full-domain retirement must scale with deletion runs"
        );
    }

    #[test]
    fn hundred_million_row_direct_exclusion_retirement_is_range_bounded() {
        const ROWS: u32 = 100_000_000;
        let started = std::time::Instant::now();
        let source = RowAddressLogicalDomain::new(7, ROWS, 1).unwrap();
        let excluded = Arc::new(
            LogicalRowAddressSelection::from_ranges(vec![LogicalRowAddressRange::new(
                7, 25_000_000, 75_000_000,
            )])
            .unwrap(),
        );
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source,
            destination_fragment_id: 9,
            destination_start: 0,
            excluded: Some(excluded),
        })];
        let mut sequences = physical_logical_sequences(&layout, &BTreeSet::from([9])).unwrap();
        let physical = sequences.remove(&9).unwrap().pop().unwrap();
        assert_eq!(physical.excluded_physical_offsets.len(), 50_000_000);
        assert_eq!(
            clustered_deletion_ranges(&physical.excluded_physical_offsets)
                .unwrap()
                .unwrap(),
            vec![25_000_000..75_000_000]
        );

        let mut deleted = RoaringBitmap::new();
        deleted.insert_range(25_000_000..90_000_000);
        let mut owned_deleted = deleted.clone();
        owned_deleted -= &physical.excluded_physical_offsets;
        let retired_ranges = clustered_deletion_ranges(&owned_deleted).unwrap().unwrap();
        assert_eq!(retired_ranges, vec![75_000_000..90_000_000]);
        let mut retired = RowIdSequence::new();
        let mut live = RowIdSequence::new();
        append_intersections(&physical, &retired_ranges, &mut retired).unwrap();
        append_complement(
            &physical,
            &clustered_deletion_ranges(&deleted).unwrap().unwrap(),
            &mut live,
        )
        .unwrap();
        assert_eq!(retired.len(), 15_000_000);
        assert_eq!(live.len(), 35_000_000);
        assert!(write_row_ids(&retired).len() < 128);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "clustered Direct exclusions must map and subtract by range"
        );
    }

    #[test]
    fn hundred_million_row_selected_exclusion_mapping_is_range_bounded() {
        const ROWS: u32 = 100_000_000;
        let started = std::time::Instant::now();
        let source = RowAddressLogicalDomain::new(8, ROWS, 1).unwrap();
        let selection = Arc::new(
            LogicalRowAddressSelection::from_ranges(vec![LogicalRowAddressRange::new(8, 0, ROWS)])
                .unwrap(),
        );
        let excluded = Arc::new(
            LogicalRowAddressSelection::from_ranges(vec![LogicalRowAddressRange::new(
                8, 25_000_000, 75_000_000,
            )])
            .unwrap(),
        );
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::Selected(SelectedRowAddressPlacement {
            source,
            selection,
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: Some(excluded),
        })];
        let mut sequences = physical_logical_sequences(&layout, &BTreeSet::from([10])).unwrap();
        let physical = sequences.remove(&10).unwrap().pop().unwrap();

        assert_eq!(physical.logical_rows.len(), u64::from(ROWS));
        assert_eq!(physical.excluded_physical_offsets.len(), 50_000_000);
        assert_eq!(
            clustered_deletion_ranges(&physical.excluded_physical_offsets)
                .unwrap()
                .unwrap(),
            vec![25_000_000..75_000_000]
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "clustered Selected exclusions must map by rank boundaries"
        );
    }

    #[test]
    fn reused_physical_hole_preserves_the_current_logical_owner() {
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: RowAddressLogicalDomain::new(7, 3, 1).unwrap(),
                destination_fragment_id: 9,
                destination_start: 0,
                excluded: Some(Arc::new(
                    LogicalRowAddressSelection::from_ranges(vec![LogicalRowAddressRange::new(
                        7, 1, 2,
                    )])
                    .unwrap(),
                )),
            }),
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: RowAddressLogicalDomain::new(8, 1, 1).unwrap(),
                destination_fragment_id: 9,
                destination_start: 1,
                excluded: None,
            }),
        ];
        let mut sequences = physical_logical_sequences(&layout, &BTreeSet::from([9])).unwrap();
        let sequences = sequences.remove(&9).unwrap();

        let mut live = RowIdSequence::new();
        let mut retired = RowIdSequence::new();
        let mut entropy = 0;
        append_fragment_provenance(
            &sequences,
            &RoaringBitmap::new(),
            &mut entropy,
            &mut retired,
            &mut live,
        )
        .unwrap();
        assert!(retired.is_empty());
        assert_eq!(
            live.iter().collect::<Vec<_>>(),
            vec![
                LogicalRowAddress::try_new_from_parts(7, 0).unwrap().raw(),
                LogicalRowAddress::try_new_from_parts(8, 0).unwrap().raw(),
                LogicalRowAddress::try_new_from_parts(7, 2).unwrap().raw(),
            ]
        );
        assert!(!live.is_strictly_sorted());

        let mut live = RowIdSequence::new();
        let mut retired = RowIdSequence::new();
        let mut entropy = 0;
        let deleted = RoaringBitmap::from_iter([1]);
        append_fragment_provenance(&sequences, &deleted, &mut entropy, &mut retired, &mut live)
            .unwrap();
        assert_eq!(
            live.iter().collect::<Vec<_>>(),
            vec![
                LogicalRowAddress::try_new_from_parts(7, 0).unwrap().raw(),
                LogicalRowAddress::try_new_from_parts(7, 2).unwrap().raw(),
            ]
        );
        assert_eq!(
            retired.iter().collect::<Vec<_>>(),
            vec![LogicalRowAddress::try_new_from_parts(8, 0).unwrap().raw()]
        );
    }

    #[test]
    #[ignore = "release-scale 100M/1M sustained owner-move compaction benchmark"]
    fn sustained_one_percent_owner_move_compacts_without_entropy_backpressure() {
        const ROWS: u32 = 100_000_000;
        const TOUCHED: u32 = 1_000_000;
        let started = std::time::Instant::now();
        let mut touched = RoaringBitmap::new();
        for index in 0..TOUCHED {
            touched.insert(((index as u64 * 99_991) % ROWS as u64) as u32);
        }
        assert_eq!(touched.len(), u64::from(TOUCHED));
        let selection = Arc::new(
            LogicalRowAddressSelection::from_bitmap(RoaringTreemap::from_bitmaps([(
                7,
                touched.clone(),
            )]))
            .unwrap(),
        );
        let mut selected_exclusions = RoaringBitmap::new();
        append_selected_excluded_physical_offsets(
            selection.as_ref(),
            Some(selection.as_ref()),
            0,
            &mut selected_exclusions,
        )
        .unwrap();
        assert!(selected_exclusions.contains_range(0..TOUCHED));
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: RowAddressLogicalDomain::new(7, ROWS, 1).unwrap(),
                destination_fragment_id: 10,
                destination_start: 0,
                excluded: Some(selection.clone()),
            }),
            RowAddressPlacement::Selected(SelectedRowAddressPlacement {
                source: RowAddressLogicalDomain::new(7, ROWS, 1).unwrap(),
                selection,
                destination_fragment_id: 11,
                destination_start: 0,
                excluded: None,
            }),
        ];
        let mut sequences = physical_logical_sequences(&layout, &BTreeSet::from([10, 11])).unwrap();
        let direct = sequences.remove(&10).unwrap().pop().unwrap();
        let selected = sequences.remove(&11).unwrap().pop().unwrap();
        assert_eq!(direct.excluded_physical_offsets, touched);

        let mut direct_live = RowIdSequence::new();
        let mut selected_live = RowIdSequence::new();
        let mut retired = RowIdSequence::new();
        let mut entropy = 0;
        append_sequence_provenance(
            &direct,
            &touched,
            &mut entropy,
            &mut retired,
            &mut direct_live,
        )
        .unwrap();
        append_sequence_provenance(
            &selected,
            &RoaringBitmap::new(),
            &mut entropy,
            &mut retired,
            &mut selected_live,
        )
        .unwrap();
        assert!(retired.is_empty());
        assert_eq!(entropy, 0, "owner-move holes are not logical deletes");
        assert_eq!(direct_live.len(), u64::from(ROWS - TOUCHED));
        assert_eq!(selected_live.len(), u64::from(TOUCHED));
        assert!(direct_live.is_strictly_sorted());
        assert!(selected_live.is_strictly_sorted());

        let mut canonical_rows = RowAddrTreeMap::from(&direct_live);
        canonical_rows |= RowAddrTreeMap::from(&selected_live);
        assert_eq!(canonical_rows.len(), Some(u64::from(ROWS)));
        let canonical = canonical_logical_sequence(canonical_rows).unwrap();
        assert_eq!(canonical.len(), u64::from(ROWS));
        assert!(canonical.is_strictly_sorted());
        assert!(write_row_ids(&canonical).len() < 128);
        if !cfg!(debug_assertions) {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(30),
                "100M/1M sustained owner-move compaction must scale with compressed membership"
            );
        }
    }

    #[test]
    fn eighty_million_row_packed_retirement_is_range_bounded() {
        const DOMAIN_ROWS: u32 = 1_000_000;
        let started = std::time::Instant::now();
        let mut logical_rows = RowIdSequence::new();
        for logical_fragment_id in 0..100 {
            logical_rows
                .extend(logical_range_sequence(logical_fragment_id, 0, DOMAIN_ROWS).unwrap());
        }
        let physical = PhysicalLogicalSequence {
            destination_start: 0,
            physical_length: 100_000_000,
            logical_rows,
            excluded_physical_offsets: RoaringBitmap::new(),
        };
        let deleted = vec![10_000_000..90_000_000];
        let mut retired = RowIdSequence::new();
        let mut live = RowIdSequence::new();

        append_intersections(&physical, &deleted, &mut retired).unwrap();
        append_complement(&physical, &deleted, &mut live).unwrap();

        assert_eq!(retired.len(), 80_000_000);
        assert_eq!(live.len(), 20_000_000);
        assert!(write_row_ids(&retired).len() < 4_096);
        assert!(write_row_ids(&live).len() < 1_024);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "clustered PackedRun retirement must scale with domains and runs"
        );
    }

    #[test]
    fn high_entropy_deletion_detection_is_bounded() {
        let mut deleted = RoaringBitmap::new();
        for offset in (0..20_000).step_by(2) {
            deleted.insert(offset);
        }
        assert!(clustered_deletion_ranges(&deleted).unwrap().is_none());
    }

    #[test]
    fn full_delete_domains_do_not_consume_entropy_budget() {
        let mut deleted = RoaringBitmap::new();
        deleted.insert_range(0..10_000);
        let mut entropy = 0;
        for logical_fragment_id in 0..10_000 {
            let ranges =
                admitted_deletion_ranges(&deleted, logical_fragment_id, &mut entropy).unwrap();
            assert_eq!(ranges, vec![0..10_000]);
        }
        assert_eq!(entropy, 0);
    }

    #[test]
    fn group_high_entropy_is_rejected_before_range_expansion() {
        let deleted = RoaringBitmap::from_iter((0..2_000).step_by(2));
        let mut entropy = 0;
        for logical_fragment_id in 0..4 {
            admitted_deletion_ranges(&deleted, logical_fragment_id, &mut entropy).unwrap();
        }
        assert_eq!(entropy, 3_996);
        assert!(admitted_deletion_ranges(&deleted, 4, &mut entropy).is_err());
        assert_eq!(entropy, 3_996);
    }
}
