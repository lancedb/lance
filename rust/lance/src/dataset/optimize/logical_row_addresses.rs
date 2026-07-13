// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, BTreeSet};

use futures::{StreamExt, TryStreamExt};
use lance_core::utils::address::{LogicalRowAddress, RowAddress};
use lance_core::{Error, Result};
use lance_table::format::{
    Fragment, LogicalRowAddressRange, LogicalRowAddressSelection, PhysicalToLogicalResolution,
    PlacementMaintenanceRequired, RowAddressLayout, RowAddressLayoutDelta, RowAddressLogicalDomain,
    RowAddressPlacement, RowAddressPlacementDelta, RowAddressPlacementKind, RowAddressRouter,
    RowAddressTargetFragment, RowAddressTargetRange, RowSequenceFingerprintBuilder,
};
use lance_table::rowids::{RowIdSequence, read_row_ids, write_row_ids};
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
    for range in selection.to_ranges()? {
        sequence.extend(logical_range_sequence(
            range.logical_fragment_id,
            range.start_slot,
            range.end_slot,
        )?);
    }
    if sequence.len() != selection.cardinality() {
        return Err(Error::invalid_input(
            "logical selection ranges disagree with selection cardinality",
        ));
    }
    Ok(sequence)
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
                    },
                );
            }
            RowAddressPlacement::Selected(value) => {
                if !physical_fragment_ids.contains(&value.destination_fragment_id) {
                    continue;
                }
                let physical_length = u32::try_from(value.selection.cardinality())
                    .map_err(|_| Error::invalid_input("Selected physical length exceeds u32"))?;
                push(
                    value.destination_fragment_id,
                    PhysicalLogicalSequence {
                        destination_start: value.destination_start,
                        physical_length,
                        logical_rows: selection_sequence(&value.selection)?,
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
                        },
                    );
                }
            }
            RowAddressPlacement::SparseSelection(value) => {
                if !physical_fragment_ids.contains(&value.destination_fragment_id) {
                    continue;
                }
                let mut logical_rows = RowIdSequence::new();
                for source in &value.sources {
                    logical_rows.extend(selection_sequence(&source.selection)?);
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

fn canonical_logical_sequence(
    mut physical_ranges: Vec<std::ops::Range<u64>>,
) -> Result<RowIdSequence> {
    physical_ranges.sort_unstable_by_key(|range| range.start);
    let mut previous_end = None;
    let mut output = RowIdSequence::new();
    for range in physical_ranges {
        if let Some(previous_end) = previous_end
            && previous_end > range.start
        {
            return Err(Error::internal(format!(
                "storage-version-2.3 compaction found duplicate logical ownership around row address {}",
                range.start
            )));
        }
        previous_end = Some(range.end);
        output.extend(RowIdSequence::from(range));
    }
    Ok(output)
}

fn collect_retired_rows(
    router: &RowAddressRouter,
    physical_fragment_id: u32,
    deleted: &RoaringBitmap,
    retired: &mut RoaringTreemap,
) -> Result<()> {
    const BATCH_SIZE: usize = 4096;
    let mut physical = Vec::with_capacity(BATCH_SIZE);
    let mut logical = Vec::with_capacity(BATCH_SIZE);
    for offset in deleted.iter() {
        physical.push(RowAddress::new_from_parts(physical_fragment_id, offset));
        if physical.len() == BATCH_SIZE {
            collect_inline_logical_addresses(router, &physical, &mut logical, true)?;
            physical.clear();
            retired.extend(logical.drain(..));
        }
    }
    if !physical.is_empty() {
        collect_inline_logical_addresses(router, &physical, &mut logical, true)?;
        retired.extend(logical);
    }
    Ok(())
}

/// Build default-compaction identity provenance using only manifest placement
/// metadata and deletion vectors. No data file, footer, or user column is read.
pub(super) async fn plan_default_compaction_rows(
    dataset: &Dataset,
    fragments: &[Fragment],
) -> Result<PlannedDefaultCompactionRows> {
    let router = dataset.row_address_router()?;
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
    let mut placement_sequences = physical_logical_sequences(layout, &physical_fragment_ids)?;
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

    let mut physical_ranges = Vec::new();
    let mut current_run_end = None;
    let mut logical_run_ends = Vec::new();
    let mut requires_full_logical_sort = false;
    let mut retired = RoaringTreemap::new();
    let mut current_deletion_vectors = BTreeMap::new();

    for (fragment_index, (fragment, deleted)) in fragment_states.into_iter().enumerate() {
        let fragment_id = u32::try_from(fragment.id).map_err(|_| {
            Error::invalid_input(format!(
                "physical fragment id {} exceeds row-address capacity",
                fragment.id
            ))
        })?;
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
        collect_retired_rows(&router, fragment_id, &deleted, &mut retired)?;

        let mut fragment_sequence = RowIdSequence::new();
        if let Some(native) = fragment.native_logical_domain {
            let mut sequence =
                logical_range_sequence(native.logical_fragment_id, 0, physical_rows)?;
            sequence.mask(deleted.iter())?;
            fragment_sequence.extend(sequence);
        } else {
            layout.verify_visibility(&fragment, &deleted)?;
            for mut sequence in placement_sequences.remove(&fragment_id).unwrap_or_default() {
                let end = sequence
                    .destination_start
                    .checked_add(sequence.physical_length)
                    .ok_or_else(|| {
                        Error::invalid_input("placement physical range end exceeds u32")
                    })?;
                sequence.logical_rows.mask(
                    deleted
                        .range(sequence.destination_start..end)
                        .map(|offset| offset - sequence.destination_start),
                )?;
                fragment_sequence.extend(sequence.logical_rows);
            }
        }

        let fragment_ranges = validated_logical_ranges(&fragment_sequence)?;
        let is_fragment_monotonic = fragment_ranges
            .windows(2)
            .all(|pair| pair[0].end <= pair[1].start);
        if !is_fragment_monotonic {
            requires_full_logical_sort = true;
        } else if let (Some(run_end), Some(first)) = (
            current_run_end,
            fragment_ranges.first().map(|range| range.start),
        ) && run_end > first
        {
            logical_run_ends.push(u32::try_from(fragment_index).map_err(|_| {
                Error::invalid_input("compaction source fragment count exceeds u32")
            })?);
            current_run_end = fragment_ranges.last().map(|range| range.end);
        } else if let Some(last) = fragment_ranges.last() {
            current_run_end = Some(last.end);
        }
        physical_ranges.extend(fragment_ranges);

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
    let logical_sequence = canonical_logical_sequence(physical_ranges)?;

    let retired_logical_row_ids = if retired.is_empty() {
        None
    } else {
        let mut sequence = RowIdSequence::new();
        let mut chunk = Vec::with_capacity(4096);
        for row_id in retired {
            chunk.push(row_id);
            if chunk.len() == 4096 {
                sequence.extend(RowIdSequence::from(chunk.as_slice()));
                chunk.clear();
            }
        }
        if !chunk.is_empty() {
            sequence.extend(RowIdSequence::from(chunk.as_slice()));
        }
        Some(write_row_ids(&sequence))
    };

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

fn collect_inline_logical_addresses(
    router: &RowAddressRouter,
    physical: &[RowAddress],
    logical: &mut Vec<u64>,
    allow_unmapped: bool,
) -> Result<()> {
    for (physical, resolution) in physical
        .iter()
        .zip(router.physical_to_logical_many(physical)?)
    {
        match resolution {
            PhysicalToLogicalResolution::Logical(address) => logical.push(address.raw()),
            PhysicalToLogicalResolution::Unmapped if allow_unmapped => {}
            PhysicalToLogicalResolution::Unmapped => {
                return Err(Error::internal(format!(
                    "live physical row {} has no logical owner during storage-version-2.3 compaction",
                    u64::from(*physical)
                )));
            }
            PhysicalToLogicalResolution::ExplicitMap { .. } => {
                return Err(Error::not_supported_source(
                    "default compaction cannot resolve ExplicitMap provenance; run explicit placement maintenance first"
                        .into(),
                ));
            }
        }
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
        let mut bitmap = RoaringTreemap::new();
        for raw in retired.iter() {
            let address = LogicalRowAddress::try_from(raw)?;
            let domain = router
                .logical_domain(address.logical_fragment_id())?
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "compaction retired unknown logical fragment {}",
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_table::format::{PackedRunRowAddressPlacement, RowAddressPlacement};
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
}
