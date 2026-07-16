// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Cursor;
use std::sync::{Arc, OnceLock};

use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::{LogicalRowAddress, RowAddress};
use lance_core::{Error, Result};
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Fragment, pb};

pub const ROW_ADDRESS_LAYOUT_ENCODING_VERSION: u32 = 1;
pub const ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE: usize = 16;
pub const ROW_ADDRESS_B_FAST: u64 = 2 * 1024 * 1024;
pub const ROW_ADDRESS_W_FAST: u64 = 64 * 1024 * 1024;
pub const ROW_ADDRESS_EXTENT_HARD_LIMIT: u32 = 32;
const RANK_CHECKPOINT_INTERVAL: u32 = 512;
const SELECT_CHECKPOINT_INTERVAL: u32 = 256;
const MAX_RANGE_ENCODING_RUNS: usize = 4096;
const MAX_SELECTION_DOMAINS: usize = 1_000_000;
const MAX_COMPRESSED_DOMAINS: u32 = 10_000_000;
const INVALID_ID: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, DeepSizeOf)]
pub struct LogicalRowAddressRange {
    pub logical_fragment_id: u32,
    pub start_slot: u32,
    pub end_slot: u32,
}

impl LogicalRowAddressRange {
    pub fn new(logical_fragment_id: u32, start_slot: u32, end_slot: u32) -> Self {
        Self {
            logical_fragment_id,
            start_slot,
            end_slot,
        }
    }

    pub fn len(&self) -> u64 {
        self.end_slot.saturating_sub(self.start_slot) as u64
    }

    pub fn is_empty(&self) -> bool {
        self.start_slot >= self.end_slot
    }

    fn validate(&self) -> Result<()> {
        if self.logical_fragment_id == INVALID_ID {
            return Err(Error::invalid_input(
                "logical row address selection uses the reserved logical fragment id",
            ));
        }
        if self.start_slot >= self.end_slot {
            return Err(Error::invalid_input(format!(
                "logical row address range must be non-empty: logical_fragment_id={}, start_slot={}, end_slot={}",
                self.logical_fragment_id, self.start_slot, self.end_slot
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SelectionBuilderInput {
    Ranges(Vec<LogicalRowAddressRange>),
    Bitmap(RoaringTreemap),
}

impl Default for SelectionBuilderInput {
    fn default() -> Self {
        Self::Ranges(Vec::new())
    }
}

impl DeepSizeOf for SelectionBuilderInput {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::Ranges(ranges) => ranges.deep_size_of_children(context),
            Self::Bitmap(bitmap) => bitmap.serialized_size(),
        }
    }
}

impl SelectionBuilderInput {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Ranges(ranges) => {
                let mut previous: Option<&LogicalRowAddressRange> = None;
                for range in ranges {
                    range.validate()?;
                    if let Some(previous) = previous {
                        if (range.logical_fragment_id, range.start_slot)
                            <= (previous.logical_fragment_id, previous.start_slot)
                        {
                            return Err(Error::invalid_input(
                                "logical row address ranges must be strictly source-sorted",
                            ));
                        }
                        if range.logical_fragment_id == previous.logical_fragment_id
                            && range.start_slot < previous.end_slot
                        {
                            return Err(Error::invalid_input(
                                "logical row address ranges must not overlap",
                            ));
                        }
                    }
                    previous = Some(range);
                }
            }
            Self::Bitmap(bitmap) => {
                if bitmap
                    .max()
                    .is_some_and(|raw| (raw >> 32) as u32 == INVALID_ID)
                {
                    return Err(Error::invalid_input(
                        "logical row address bitmap uses the reserved logical fragment id",
                    ));
                }
                if bitmap.bitmaps().any(|(_, slots)| slots.contains(u32::MAX)) {
                    return Err(Error::invalid_input(
                        "logical row address bitmap contains an unrepresentable terminal slot",
                    ));
                }
            }
        }
        Ok(())
    }

    fn bitmap(&self) -> RoaringTreemap {
        match self {
            Self::Ranges(ranges) => {
                let mut bitmap = RoaringTreemap::new();
                for range in ranges {
                    let start = u64::from(range.logical_fragment_id) << 32;
                    bitmap.insert_range(
                        start + u64::from(range.start_slot)..start + u64::from(range.end_slot),
                    );
                }
                bitmap
            }
            Self::Bitmap(bitmap) => bitmap.clone(),
        }
    }

    fn range_candidate(&self) -> Option<pb::LogicalRowAddressSelection> {
        let ranges = match self {
            Self::Ranges(ranges) if ranges.len() <= MAX_RANGE_ENCODING_RUNS => ranges.clone(),
            Self::Ranges(_) => return None,
            Self::Bitmap(bitmap) => bitmap_to_ranges(bitmap, MAX_RANGE_ENCODING_RUNS)?,
        };
        Some(pb::LogicalRowAddressSelection {
            encoding: Some(pb::logical_row_address_selection::Encoding::Ranges(
                pb::LogicalRowAddressRangeList {
                    ranges: ranges
                        .into_iter()
                        .map(|range| pb::LogicalRowAddressRange {
                            logical_fragment_id: range.logical_fragment_id,
                            start_slot: range.start_slot,
                            end_slot: range.end_slot,
                        })
                        .collect(),
                },
            )),
            canonical_encoding_version: 1,
        })
    }

    fn roaring_candidate(bitmap: &RoaringTreemap) -> Option<pb::LogicalRowAddressSelection> {
        let mut bytes = Vec::with_capacity(bitmap.serialized_size());
        bitmap.serialize_into(&mut bytes).ok()?;
        Some(pb::LogicalRowAddressSelection {
            encoding: Some(pb::logical_row_address_selection::Encoding::RoaringTreemap(
                bytes,
            )),
            canonical_encoding_version: 1,
        })
    }

    fn dense_candidate(bitmap: &RoaringTreemap) -> Option<pb::LogicalRowAddressSelection> {
        let total_universe = bitmap
            .bitmaps()
            .filter_map(|(_, slots)| slots.max())
            .try_fold(0_u64, |total, max_slot| {
                total.checked_add(max_slot as u64 + 1)
            })?;
        if bitmap.len().saturating_mul(16) < total_universe {
            return None;
        }
        let mut encoded_domains = Vec::new();
        for (logical_fragment_id, slots) in bitmap.bitmaps() {
            let max_slot = slots.max()?;
            let universe = max_slot.checked_add(1)?;
            let mut bits = vec![0_u8; (universe as usize).div_ceil(8)];
            let mut runs = slots.iter();
            while let Some(run) = runs.next_range() {
                set_bit_range(
                    &mut bits,
                    u64::from(*run.start()),
                    u64::from(*run.end()) + 1,
                );
            }
            let mut rank_checkpoints = Vec::with_capacity(
                (universe as usize).div_ceil(RANK_CHECKPOINT_INTERVAL as usize) + 1,
            );
            let mut rank = 0_u32;
            for boundary in (0..universe).step_by(RANK_CHECKPOINT_INTERVAL as usize) {
                rank_checkpoints.push(rank);
                let end = universe.min(boundary.saturating_add(RANK_CHECKPOINT_INTERVAL));
                rank = rank.saturating_add(count_bits_fast(&bits, boundary as u64, end as u64));
            }
            rank_checkpoints.push(rank);
            encoded_domains.push(pb::DenseBitsetLogicalDomainSelection {
                logical_fragment_id,
                universe,
                bits,
                rank_checkpoint_interval: RANK_CHECKPOINT_INTERVAL,
                rank_checkpoints,
            });
        }
        Some(pb::LogicalRowAddressSelection {
            encoding: Some(pb::logical_row_address_selection::Encoding::DenseBitset(
                pb::DenseBitsetLogicalRowAddressSelection {
                    domains: encoded_domains,
                },
            )),
            canonical_encoding_version: 1,
        })
    }

    fn elias_fano_candidate(bitmap: &RoaringTreemap) -> Option<pb::LogicalRowAddressSelection> {
        if bitmap.bitmaps().count() > MAX_RANGE_ENCODING_RUNS {
            return None;
        }
        let mut encoded_domains = Vec::new();
        for (logical_fragment_id, slots) in bitmap.bitmaps() {
            let max_slot = slots.max()?;
            let universe = max_slot.checked_add(1)?;
            let cardinality = u32::try_from(slots.len()).ok()?;
            let low_bit_width = elias_fano_low_bit_width(universe, cardinality);
            let mut low_bits =
                vec![0_u8; (cardinality as usize * low_bit_width as usize).div_ceil(8)];
            let high_bit_len = (universe as u64 >> low_bit_width) + cardinality as u64;
            let mut high_bits = vec![0_u8; (high_bit_len as usize).div_ceil(8)];
            let mut select_checkpoints = Vec::with_capacity(
                (cardinality as usize).div_ceil(SELECT_CHECKPOINT_INTERVAL as usize),
            );
            let low_mask = if low_bit_width == 0 {
                0
            } else {
                (1_u32 << low_bit_width) - 1
            };
            for (ordinal, slot) in slots.iter().enumerate() {
                write_packed_u32(&mut low_bits, ordinal, low_bit_width, slot & low_mask);
                let high_position = (slot as u64 >> low_bit_width) + ordinal as u64;
                set_bit(&mut high_bits, high_position);
                if ordinal % SELECT_CHECKPOINT_INTERVAL as usize == 0 {
                    select_checkpoints.push(u32::try_from(high_position).ok()?);
                }
            }
            encoded_domains.push(pb::EliasFanoLogicalDomainSelection {
                logical_fragment_id,
                universe,
                cardinality,
                low_bit_width,
                low_bits,
                high_bits,
                select_checkpoint_interval: SELECT_CHECKPOINT_INTERVAL,
                select_checkpoints,
            });
        }
        Some(pb::LogicalRowAddressSelection {
            encoding: Some(pb::logical_row_address_selection::Encoding::EliasFano(
                pb::EliasFanoLogicalRowAddressSelection {
                    domains: encoded_domains,
                },
            )),
            canonical_encoding_version: 1,
        })
    }

    fn ordinal_elias_fano_candidate(
        bitmap: &RoaringTreemap,
    ) -> Option<pb::LogicalRowAddressSelection> {
        if bitmap.is_empty() {
            return None;
        }
        let mut domain_ids = Vec::new();
        let mut slot_universes = Vec::new();
        let mut universe = 0_u64;
        for (logical_fragment_id, slots) in bitmap.bitmaps() {
            let slot_universe = slots.max()?.checked_add(1)?;
            domain_ids.push(logical_fragment_id);
            slot_universes.push(slot_universe);
            universe = universe.checked_add(slot_universe as u64)?;
        }
        let (low_bit_width, low_bits, high_bits, select_checkpoints) = encode_elias_fano_u64_iter(
            bitmap.bitmaps().flat_map({
                let mut prefix = 0_u64;
                move |(_, slots)| {
                    let domain_prefix = prefix;
                    prefix += u64::from(slots.max().expect("non-empty Roaring domain")) + 1;
                    slots
                        .iter()
                        .map(move |slot| domain_prefix + u64::from(slot))
                }
            }),
            bitmap.len(),
            universe,
        )?;
        let first_logical_fragment_id = *domain_ids.first()?;
        let domain_count = u32::try_from(domain_ids.len()).ok()?;
        let domain_run = pb::LogicalOrdinalDomainRun {
            first_logical_fragment_id,
            domain_count,
            logical_fragment_ids: Some(encode_logical_fragment_ids(&domain_ids)?),
            slot_universes: Some(encode_slot_counts(&slot_universes)?),
        };
        Some(pb::LogicalRowAddressSelection {
            encoding: Some(
                pb::logical_row_address_selection::Encoding::OrdinalEliasFano(
                    pb::EliasFanoOrdinalLogicalRowAddressSelection {
                        domain_runs: vec![domain_run],
                        universe,
                        cardinality: bitmap.len(),
                        low_bit_width,
                        low_bits,
                        high_bits,
                        select_checkpoint_interval: SELECT_CHECKPOINT_INTERVAL,
                        select_checkpoints,
                    },
                ),
            ),
            canonical_encoding_version: 1,
        })
    }

    pub fn canonical_proto(&self) -> pb::LogicalRowAddressSelection {
        fn retain_smaller(
            best: &mut Option<pb::LogicalRowAddressSelection>,
            candidate: Option<pb::LogicalRowAddressSelection>,
        ) {
            if let Some(candidate) = candidate
                && best
                    .as_ref()
                    .is_none_or(|best| candidate.encoded_len() < best.encoded_len())
            {
                *best = Some(candidate);
            }
        }

        fn elias_fano_payload_bytes(universe: u64, cardinality: u64) -> Option<u64> {
            if universe == 0 || cardinality == 0 {
                return None;
            }
            let low_bit_width = if universe <= cardinality {
                0
            } else {
                (universe / cardinality).ilog2()
            };
            let low_bits = cardinality.checked_mul(u64::from(low_bit_width))?;
            let high_bits = (universe >> low_bit_width).checked_add(cardinality)?;
            low_bits.div_ceil(8).checked_add(high_bits.div_ceil(8))
        }

        fn payload_lower_bounds(
            bitmap: &RoaringTreemap,
        ) -> Option<(Option<u64>, Option<u64>, u64)> {
            let mut domain_count = 0_usize;
            let mut total_universe = 0_u64;
            let mut dense_bytes = 0_u64;
            let mut domain_elias_fano_bytes = 0_u64;
            for (_, slots) in bitmap.bitmaps() {
                let universe = u64::from(slots.max()?.checked_add(1)?);
                domain_count = domain_count.checked_add(1)?;
                total_universe = total_universe.checked_add(universe)?;
                dense_bytes = dense_bytes.checked_add(universe.div_ceil(8))?;
                domain_elias_fano_bytes = domain_elias_fano_bytes
                    .checked_add(elias_fano_payload_bytes(universe, slots.len())?)?;
            }
            let dense = (bitmap.len().saturating_mul(16) >= total_universe).then_some(dense_bytes);
            let domain_elias_fano =
                (domain_count <= MAX_RANGE_ENCODING_RUNS).then_some(domain_elias_fano_bytes);
            let ordinal_elias_fano = elias_fano_payload_bytes(total_universe, bitmap.len())?;
            Some((dense, domain_elias_fano, ordinal_elias_fano))
        }

        fn candidate_is_proven_smallest(
            best: &Option<pb::LogicalRowAddressSelection>,
            lower_bounds: impl IntoIterator<Item = Option<u64>>,
        ) -> bool {
            let Some(best) = best else {
                return false;
            };
            let encoded_len = best.encoded_len() as u64;
            lower_bounds
                .into_iter()
                .flatten()
                .all(|lower_bound| encoded_len <= lower_bound)
        }

        // A compaction over many whole logical domains naturally arrives as
        // one dense prefix range per domain.  Once the range count exceeds the
        // inline range limit, Roaring encodes that shape in O(domains) space.
        // Trying the dense and Elias-Fano candidates would materialize every
        // selected row even though they cannot improve on that representation.
        if let Self::Ranges(ranges) = self
            && ranges.len() > MAX_RANGE_ENCODING_RUNS
            && ranges.iter().all(|range| range.start_slot == 0)
            && let Some(selected_rows) = ranges
                .iter()
                .try_fold(0_u64, |total, range| total.checked_add(range.len()))
            && let Some(candidate) = Self::roaring_candidate(&self.bitmap())
            // A dense bitset is the smallest remaining competitor and needs
            // at least one bit per selected row for this full-universe shape.
            && candidate.encoded_len() as u64 <= selected_rows.div_ceil(8)
        {
            return candidate;
        }

        let mut best = self.range_candidate();
        if matches!(
            best.as_ref().and_then(|selection| selection.encoding.as_ref()),
            Some(pb::logical_row_address_selection::Encoding::Ranges(ranges))
                if ranges.ranges.len() <= 1
        ) {
            return best.unwrap_or_default();
        }
        let bitmap = self.bitmap();
        retain_smaller(&mut best, Self::roaring_candidate(&bitmap));
        let lower_bounds = payload_lower_bounds(&bitmap);
        if lower_bounds.is_some_and(|(dense, domain_elias_fano, ordinal_elias_fano)| {
            candidate_is_proven_smallest(
                &best,
                [dense, domain_elias_fano, Some(ordinal_elias_fano)],
            )
        }) {
            return best.unwrap_or_default();
        }
        retain_smaller(&mut best, Self::dense_candidate(&bitmap));
        if lower_bounds.is_some_and(|(_, domain_elias_fano, ordinal_elias_fano)| {
            candidate_is_proven_smallest(&best, [domain_elias_fano, Some(ordinal_elias_fano)])
        }) {
            return best.unwrap_or_default();
        }
        retain_smaller(&mut best, Self::elias_fano_candidate(&bitmap));
        if lower_bounds.is_some_and(|(_, _, ordinal_elias_fano)| {
            candidate_is_proven_smallest(&best, [Some(ordinal_elias_fano)])
        }) {
            return best.unwrap_or_default();
        }
        retain_smaller(&mut best, Self::ordinal_elias_fano_candidate(&bitmap));
        best.unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeEncodedSelection {
    ranges: Arc<[LogicalRowAddressRange]>,
    prefixes: Arc<[u64]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseEncodedSelection {
    encoded: pb::DenseBitsetLogicalRowAddressSelection,
    prefixes: Arc<[u64]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EliasFanoEncodedSelection {
    encoded: pb::EliasFanoLogicalRowAddressSelection,
    prefixes: Arc<[u64]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrdinalEliasFanoEncodedSelection {
    encoded: pb::EliasFanoOrdinalLogicalRowAddressSelection,
    run_universe_prefixes: Arc<[u64]>,
}

#[derive(Debug, Clone)]
pub struct RoaringEncodedSelection {
    bytes: Arc<[u8]>,
    cardinality: u64,
    bitmap: Arc<OnceLock<RoaringTreemap>>,
}

impl PartialEq for RoaringEncodedSelection {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.cardinality == other.cardinality
    }
}

impl Eq for RoaringEncodedSelection {}

impl RoaringEncodedSelection {
    fn bitmap(&self) -> Result<&RoaringTreemap> {
        if let Some(bitmap) = self.bitmap.get() {
            return Ok(bitmap);
        }
        let mut cursor = Cursor::new(self.bytes.as_ref());
        let bitmap = RoaringTreemap::deserialize_from(&mut cursor).map_err(|error| {
            Error::invalid_input(format!(
                "invalid portable Roaring selection payload: {error}"
            ))
        })?;
        if cursor.position() != self.bytes.len() as u64 || bitmap.len() != self.cardinality {
            return Err(Error::invalid_input(
                "portable Roaring selection payload has trailing bytes or changed cardinality",
            ));
        }
        let _ = self.bitmap.set(bitmap);
        Ok(self
            .bitmap
            .get()
            .expect("Roaring bitmap was just initialized"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicalRowAddressSelection {
    Ranges(RangeEncodedSelection),
    Dense(DenseEncodedSelection),
    EliasFano(EliasFanoEncodedSelection),
    OrdinalEliasFano(OrdinalEliasFanoEncodedSelection),
    Roaring(RoaringEncodedSelection),
}

impl Eq for LogicalRowAddressSelection {}

impl Default for LogicalRowAddressSelection {
    fn default() -> Self {
        Self::Ranges(RangeEncodedSelection {
            ranges: Arc::from([]),
            prefixes: Arc::from([0]),
        })
    }
}

impl DeepSizeOf for LogicalRowAddressSelection {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::Ranges(value) => {
                value.ranges.deep_size_of_children(context)
                    + value.prefixes.deep_size_of_children(context)
            }
            Self::Dense(value) => {
                value.encoded.encoded_len() + value.prefixes.deep_size_of_children(context)
            }
            Self::EliasFano(value) => {
                value.encoded.encoded_len() + value.prefixes.deep_size_of_children(context)
            }
            Self::OrdinalEliasFano(value) => {
                value.encoded.encoded_len()
                    + value.run_universe_prefixes.deep_size_of_children(context)
            }
            Self::Roaring(value) => value.bytes.len(),
        }
    }
}

pub struct LogicalRowAddressSelectionBuilder {
    input: SelectionBuilderInput,
}

impl LogicalRowAddressSelectionBuilder {
    pub fn from_ranges(mut ranges: Vec<LogicalRowAddressRange>) -> Result<Self> {
        ranges.sort_unstable();
        let mut canonical = Vec::<LogicalRowAddressRange>::with_capacity(ranges.len());
        for range in ranges {
            range.validate()?;
            if let Some(previous) = canonical.last_mut()
                && previous.logical_fragment_id == range.logical_fragment_id
                && range.start_slot <= previous.end_slot
            {
                previous.end_slot = previous.end_slot.max(range.end_slot);
            } else {
                canonical.push(range);
            }
        }
        Ok(Self {
            input: SelectionBuilderInput::Ranges(canonical),
        })
    }

    pub fn from_bitmap(bitmap: &RoaringTreemap) -> Result<Self> {
        let input = SelectionBuilderInput::Bitmap(bitmap.clone());
        input.validate()?;
        Ok(Self { input })
    }

    pub fn build(self) -> Result<LogicalRowAddressSelection> {
        LogicalRowAddressSelection::try_from(self.input.canonical_proto())
    }
}

impl LogicalRowAddressSelection {
    pub fn from_ranges(ranges: Vec<LogicalRowAddressRange>) -> Result<Self> {
        LogicalRowAddressSelectionBuilder::from_ranges(ranges)?.build()
    }

    pub fn from_bitmap(bitmap: RoaringTreemap) -> Result<Self> {
        LogicalRowAddressSelectionBuilder::from_bitmap(&bitmap)?.build()
    }

    pub fn from_full_domains(domains: &[RowAddressLogicalDomain]) -> Result<Self> {
        Self::from_ranges(
            domains
                .iter()
                .map(|domain| {
                    LogicalRowAddressRange::new(domain.logical_fragment_id, 0, domain.slot_count)
                })
                .collect(),
        )
    }

    pub fn cardinality(&self) -> u64 {
        match self {
            Self::Ranges(value) => *value.prefixes.last().unwrap_or(&0),
            Self::Dense(value) => *value.prefixes.last().unwrap_or(&0),
            Self::EliasFano(value) => *value.prefixes.last().unwrap_or(&0),
            Self::OrdinalEliasFano(value) => value.encoded.cardinality,
            Self::Roaring(value) => value.cardinality,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }

    pub fn contains(&self, address: LogicalRowAddress) -> Result<bool> {
        Ok(self.rank(address)?.is_some())
    }

    pub fn rank(&self, address: LogicalRowAddress) -> Result<Option<u64>> {
        Ok(match self {
            Self::Ranges(value) => range_selection_rank(value, address),
            Self::Dense(value) => dense_selection_rank(value, address),
            Self::EliasFano(value) => elias_fano_selection_rank(value, address),
            Self::OrdinalEliasFano(value) => ordinal_selection_rank(value, address)?,
            Self::Roaring(value) => {
                let bitmap = value.bitmap()?;
                bitmap
                    .contains(address.raw())
                    .then(|| bitmap.rank(address.raw()) - 1)
            }
        })
    }

    pub fn select(&self, ordinal: u64) -> Result<Option<LogicalRowAddress>> {
        if ordinal >= self.cardinality() {
            return Ok(None);
        }
        Ok(match self {
            Self::Ranges(value) => range_selection_select(value, ordinal),
            Self::Dense(value) => dense_selection_select(value, ordinal),
            Self::EliasFano(value) => elias_fano_selection_select(value, ordinal),
            Self::OrdinalEliasFano(value) => ordinal_selection_select(value, ordinal)?,
            Self::Roaring(value) => value
                .bitmap()?
                .select(ordinal)
                .map(LogicalRowAddress::try_from)
                .transpose()?,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = Result<LogicalRowAddress>> + '_ {
        (0..self.cardinality()).map(|ordinal| {
            self.select(ordinal)?.ok_or_else(|| {
                Error::invalid_input("logical selection cardinality exceeds encoded values")
            })
        })
    }

    /// Visit selected addresses in ascending order with a single codec pass.
    ///
    /// The runtime is proportional to the encoded payload plus selection
    /// cardinality. In particular, this does not issue one rank/select lookup
    /// per address. The visitor's first error stops decoding and is returned.
    pub fn try_for_each_address(
        &self,
        mut visit: impl FnMut(LogicalRowAddress) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Ranges(value) => visit_range_selection_addresses(value, &mut visit),
            Self::Dense(value) => visit_dense_selection_addresses(value, &mut visit),
            Self::EliasFano(value) => visit_elias_fano_selection_addresses(value, &mut visit),
            Self::OrdinalEliasFano(value) => visit_ordinal_selection_addresses(value, &mut visit),
            Self::Roaring(value) => {
                for raw in value.bitmap()?.iter() {
                    visit(LogicalRowAddress::try_from(raw)?)?;
                }
                Ok(())
            }
        }
    }

    /// Return the exact set difference `self - removed` in canonical encoding.
    /// Range-backed coverage stays proportional to its range count plus the
    /// removed selection; it never expands a full logical domain row by row.
    pub fn difference(&self, removed: &Self) -> Result<Self> {
        if self.is_empty() || removed.is_empty() {
            return Ok(self.clone());
        }
        if let (Self::Ranges(base), Self::Ranges(removed)) = (self, removed) {
            return Self::from_ranges(subtract_logical_ranges(&base.ranges, &removed.ranges));
        }
        let mut selected = self.to_roaring_treemap()?;
        selected -= removed.to_roaring_treemap()?;
        Self::from_bitmap(selected)
    }

    /// Return the exact set intersection in canonical encoding.
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        if self.is_empty() || other.is_empty() {
            return Self::from_ranges(Vec::new());
        }
        let selected = self.to_roaring_treemap()? & other.to_roaring_treemap()?;
        Self::from_bitmap(selected)
    }

    /// Return the exact set union in canonical encoding.
    pub fn union(&self, other: &Self) -> Result<Self> {
        if self.is_empty() {
            return Ok(other.clone());
        }
        if other.is_empty() {
            return Ok(self.clone());
        }
        let selected = self.to_roaring_treemap()? | other.to_roaring_treemap()?;
        Self::from_bitmap(selected)
    }

    /// Return whether every selected logical row is also selected by `other`.
    pub fn is_subset_of(&self, other: &Self) -> Result<bool> {
        Ok(self
            .to_roaring_treemap()?
            .is_subset(&other.to_roaring_treemap()?))
    }

    /// Return sorted, non-overlapping value ranges. Range-backed selections are
    /// copied without expanding their rows; other codecs are decoded only when
    /// a caller explicitly needs range materialization.
    pub fn to_ranges(&self) -> Result<Vec<LogicalRowAddressRange>> {
        if let Some(ranges) = self.compact_ranges(usize::MAX)? {
            return Ok(ranges);
        }
        selection_value_ranges(self)
    }

    /// Return logical ranges only when their count does not exceed `max_runs`.
    ///
    /// This is the range-safe counterpart to [`Self::to_ranges`]. Callers that
    /// can process a high-entropy selection directly should use this method to
    /// avoid materializing one range per selected row.
    pub fn to_ranges_bounded(
        &self,
        max_runs: usize,
    ) -> Result<Option<Vec<LogicalRowAddressRange>>> {
        self.compact_ranges(max_runs)
    }

    /// Return ranges only when decoding stays bounded by `max_runs`.
    ///
    /// Full-domain codecs and contiguous Roaring containers remain proportional
    /// to their domain count. High-entropy selections return `None` instead of
    /// being expanded into one range per selected row.
    fn compact_ranges(&self, max_runs: usize) -> Result<Option<Vec<LogicalRowAddressRange>>> {
        match self {
            Self::Ranges(ranges) => {
                Ok((ranges.ranges.len() <= max_runs).then(|| ranges.ranges.to_vec()))
            }
            Self::Dense(value)
                if value.prefixes.windows(2).zip(&value.encoded.domains).all(
                    |(prefixes, domain)| {
                        prefixes[1].checked_sub(prefixes[0]) == Some(u64::from(domain.universe))
                    },
                ) =>
            {
                Ok((value.encoded.domains.len() <= max_runs).then(|| {
                    value
                        .encoded
                        .domains
                        .iter()
                        .map(|domain| {
                            LogicalRowAddressRange::new(
                                domain.logical_fragment_id,
                                0,
                                domain.universe,
                            )
                        })
                        .collect()
                }))
            }
            Self::EliasFano(value)
                if value
                    .encoded
                    .domains
                    .iter()
                    .all(|domain| domain.cardinality == domain.universe) =>
            {
                Ok((value.encoded.domains.len() <= max_runs).then(|| {
                    value
                        .encoded
                        .domains
                        .iter()
                        .map(|domain| {
                            LogicalRowAddressRange::new(
                                domain.logical_fragment_id,
                                0,
                                domain.universe,
                            )
                        })
                        .collect()
                }))
            }
            Self::OrdinalEliasFano(value)
                if value.encoded.cardinality == value.encoded.universe =>
            {
                let domain_count =
                    value
                        .encoded
                        .domain_runs
                        .iter()
                        .try_fold(0_usize, |total, run| {
                            total.checked_add(run.domain_count as usize).ok_or_else(|| {
                                Error::invalid_input("ordinal logical selection domain overflow")
                            })
                        })?;
                if domain_count > max_runs {
                    return Ok(None);
                }
                let mut ranges = Vec::new();
                for run in &value.encoded.domain_runs {
                    let logical_ids = run.logical_fragment_ids.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "ordinal logical selection is missing packed logical fragment ids",
                        )
                    })?;
                    let slot_universes = run.slot_universes.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "ordinal logical selection is missing packed slot universes",
                        )
                    })?;
                    ranges.reserve(run.domain_count as usize);
                    for ordinal in 0..run.domain_count {
                        ranges.push(LogicalRowAddressRange::new(
                            logical_fragment_id_at(
                                run.first_logical_fragment_id,
                                run.domain_count,
                                logical_ids,
                                ordinal,
                            )?,
                            0,
                            packed_slot_count_at(slot_universes, ordinal)?,
                        ));
                    }
                }
                Ok(Some(ranges))
            }
            Self::Roaring(value) => Ok(bitmap_to_ranges(value.bitmap()?, max_runs)),
            _ if self.cardinality() <= max_runs as u64 => {
                let ranges = selection_value_ranges(self)?;
                Ok((ranges.len() <= max_runs).then_some(ranges))
            }
            _ => Ok(None),
        }
    }

    /// Visit the selection as canonical contiguous ranges when the codec can
    /// expose those ranges without materializing selected rows.
    ///
    /// Roaring selections are streamed one container run at a time.  This is
    /// important for packed compaction: a full 100M-row selection spanning
    /// many logical domains is represented by a few thousand Roaring runs,
    /// while ordinal `select` would visit every row.
    fn visit_structural_ranges(
        &self,
        mut visit: impl FnMut(LogicalRowAddressRange) -> Result<()>,
    ) -> Result<bool> {
        match self {
            Self::Ranges(value) => {
                for range in value.ranges.iter().copied() {
                    visit(range)?;
                }
            }
            Self::Dense(value)
                if value.prefixes.windows(2).zip(&value.encoded.domains).all(
                    |(prefixes, domain)| {
                        prefixes[1].checked_sub(prefixes[0]) == Some(u64::from(domain.universe))
                    },
                ) =>
            {
                for domain in &value.encoded.domains {
                    visit(LogicalRowAddressRange::new(
                        domain.logical_fragment_id,
                        0,
                        domain.universe,
                    ))?;
                }
            }
            Self::EliasFano(value)
                if value
                    .encoded
                    .domains
                    .iter()
                    .all(|domain| domain.cardinality == domain.universe) =>
            {
                for domain in &value.encoded.domains {
                    visit(LogicalRowAddressRange::new(
                        domain.logical_fragment_id,
                        0,
                        domain.universe,
                    ))?;
                }
            }
            Self::OrdinalEliasFano(value)
                if value.encoded.cardinality == value.encoded.universe =>
            {
                for run in &value.encoded.domain_runs {
                    let logical_ids = run.logical_fragment_ids.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "ordinal logical selection is missing packed logical fragment ids",
                        )
                    })?;
                    let slot_universes = run.slot_universes.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "ordinal logical selection is missing packed slot universes",
                        )
                    })?;
                    for ordinal in 0..run.domain_count {
                        visit(LogicalRowAddressRange::new(
                            logical_fragment_id_at(
                                run.first_logical_fragment_id,
                                run.domain_count,
                                logical_ids,
                                ordinal,
                            )?,
                            0,
                            packed_slot_count_at(slot_universes, ordinal)?,
                        ))?;
                    }
                }
            }
            Self::Roaring(value) => {
                for (logical_fragment_id, slots) in value.bitmap()?.bitmaps() {
                    let mut runs = slots.iter();
                    while let Some(run) = runs.next_range() {
                        let end_slot = run.end().checked_add(1).ok_or_else(|| {
                            Error::invalid_input(
                                "portable Roaring selection contains a terminal logical slot",
                            )
                        })?;
                        visit(LogicalRowAddressRange::new(
                            logical_fragment_id,
                            *run.start(),
                            end_slot,
                        ))?;
                    }
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Decode this selection into a Roaring tree without expanding range or
    /// already-Roaring encodings row by row.
    pub fn to_roaring_treemap(&self) -> Result<RoaringTreemap> {
        let mut rows = RoaringTreemap::new();
        match self {
            Self::Ranges(value) => {
                for range in value.ranges.iter() {
                    let start =
                        (u64::from(range.logical_fragment_id) << 32) | u64::from(range.start_slot);
                    let end =
                        (u64::from(range.logical_fragment_id) << 32) | u64::from(range.end_slot);
                    rows.insert_range(start..end);
                }
            }
            Self::Roaring(value) => return Ok(value.bitmap()?.clone()),
            Self::OrdinalEliasFano(value)
                if value.encoded.cardinality == value.encoded.universe =>
            {
                for run in &value.encoded.domain_runs {
                    let logical_ids = run.logical_fragment_ids.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "ordinal logical selection is missing packed logical fragment ids",
                        )
                    })?;
                    let slot_universes = run.slot_universes.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "ordinal logical selection is missing packed slot universes",
                        )
                    })?;
                    for ordinal in 0..run.domain_count {
                        let logical_fragment_id = logical_fragment_id_at(
                            run.first_logical_fragment_id,
                            run.domain_count,
                            logical_ids,
                            ordinal,
                        )?;
                        let slot_count = packed_slot_count_at(slot_universes, ordinal)?;
                        let start = u64::from(logical_fragment_id) << 32;
                        rows.insert_range(start..start + u64::from(slot_count));
                    }
                }
            }
            _ => {
                self.try_for_each_address(|address| {
                    rows.insert(address.raw());
                    Ok(())
                })?;
            }
        }
        Ok(rows)
    }

    pub fn overlaps(&self, other: &Self) -> Result<bool> {
        if let (Self::Ranges(left), Self::Ranges(right)) = (self, other) {
            let mut left_index = 0;
            let mut right_index = 0;
            while left_index < left.ranges.len() && right_index < right.ranges.len() {
                let left_range = left.ranges[left_index];
                let right_range = right.ranges[right_index];
                if left_range.logical_fragment_id == right_range.logical_fragment_id
                    && left_range.start_slot < right_range.end_slot
                    && right_range.start_slot < left_range.end_slot
                {
                    return Ok(true);
                }
                if (left_range.logical_fragment_id, left_range.end_slot)
                    <= (right_range.logical_fragment_id, right_range.start_slot)
                {
                    left_index += 1;
                } else {
                    right_index += 1;
                }
            }
            return Ok(false);
        }
        Ok(!self
            .to_roaring_treemap()?
            .is_disjoint(&other.to_roaring_treemap()?))
    }

    /// Distinct logical fragment IDs represented by this selection. Compact
    /// codecs expose their domain directory directly, avoiding a per-row walk
    /// on query-planning paths.
    pub fn logical_fragment_bitmap(&self) -> Result<RoaringBitmap> {
        let mut fragments = RoaringBitmap::new();
        match self {
            Self::Ranges(value) => {
                fragments.extend(value.ranges.iter().map(|range| range.logical_fragment_id));
            }
            Self::Dense(value) => {
                fragments.extend(
                    value
                        .encoded
                        .domains
                        .iter()
                        .map(|domain| domain.logical_fragment_id),
                );
            }
            Self::EliasFano(value) => {
                fragments.extend(
                    value
                        .encoded
                        .domains
                        .iter()
                        .map(|domain| domain.logical_fragment_id),
                );
            }
            Self::OrdinalEliasFano(value) => {
                for run in &value.encoded.domain_runs {
                    let packed = run.logical_fragment_ids.as_ref().ok_or_else(|| {
                        Error::invalid_input(
                            "ordinal logical selection is missing packed logical fragment ids",
                        )
                    })?;
                    for ordinal in 0..run.domain_count {
                        fragments.insert(logical_fragment_id_at(
                            run.first_logical_fragment_id,
                            run.domain_count,
                            packed,
                            ordinal,
                        )?);
                    }
                }
            }
            Self::Roaring(value) => {
                fragments.extend(value.bitmap()?.bitmaps().map(|(domain, _)| domain));
            }
        }
        Ok(fragments)
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Ranges(value) => validate_range_selection(value),
            Self::Dense(value) => validate_dense_headers(&value.encoded).map(|_| ()),
            Self::EliasFano(value) => validate_elias_fano_headers(&value.encoded).map(|_| ()),
            Self::OrdinalEliasFano(value) => {
                validate_ordinal_elias_fano_headers(&value.encoded).map(|_| ())
            }
            Self::Roaring(value) => validate_roaring_selection(value),
        }
    }

    pub fn canonical_proto(&self) -> pb::LogicalRowAddressSelection {
        self.into()
    }
}

fn selection_value_ranges(
    selection: &LogicalRowAddressSelection,
) -> Result<Vec<LogicalRowAddressRange>> {
    let mut ranges = Vec::<LogicalRowAddressRange>::new();
    visit_selection_value_ranges(selection, |range| {
        ranges.push(range);
        Ok(())
    })?;
    Ok(ranges)
}

fn visit_selection_value_ranges(
    selection: &LogicalRowAddressSelection,
    mut visit: impl FnMut(LogicalRowAddressRange) -> Result<()>,
) -> Result<()> {
    let mut current = None::<LogicalRowAddressRange>;
    selection.try_for_each_address(|address| {
        let logical_fragment_id = address.logical_fragment_id();
        let slot = address.immutable_slot();
        if let Some(range) = current.as_mut()
            && range.logical_fragment_id == logical_fragment_id
            && range.end_slot == slot
        {
            range.end_slot = slot
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("logical selection range overflow"))?;
        } else {
            if let Some(range) = current.take() {
                visit(range)?;
            }
            current = Some(LogicalRowAddressRange::new(
                logical_fragment_id,
                slot,
                slot.checked_add(1)
                    .ok_or_else(|| Error::invalid_input("logical selection range overflow"))?,
            ));
        }
        Ok(())
    })?;
    if let Some(range) = current {
        visit(range)?;
    }
    Ok(())
}

fn effective_source_selection(
    selection: LogicalRowAddressSelection,
    excluded: Option<&Arc<LogicalRowAddressSelection>>,
) -> Result<LogicalRowAddressSelection> {
    excluded.map_or(Ok(selection.clone()), |excluded| {
        selection.difference(excluded)
    })
}

fn subtract_logical_ranges(
    base: &[LogicalRowAddressRange],
    removed: &[LogicalRowAddressRange],
) -> Vec<LogicalRowAddressRange> {
    let mut output = Vec::new();
    let mut removed_index = 0;
    for range in base {
        while removed_index < removed.len()
            && (removed[removed_index].logical_fragment_id < range.logical_fragment_id
                || removed[removed_index].logical_fragment_id == range.logical_fragment_id
                    && removed[removed_index].end_slot <= range.start_slot)
        {
            removed_index += 1;
        }
        let mut cursor = range.start_slot;
        let mut index = removed_index;
        while index < removed.len()
            && removed[index].logical_fragment_id == range.logical_fragment_id
            && removed[index].start_slot < range.end_slot
        {
            let exclusion = removed[index];
            if cursor < exclusion.start_slot {
                output.push(LogicalRowAddressRange::new(
                    range.logical_fragment_id,
                    cursor,
                    exclusion.start_slot.min(range.end_slot),
                ));
            }
            cursor = cursor.max(exclusion.end_slot);
            if cursor >= range.end_slot {
                break;
            }
            index += 1;
        }
        if cursor < range.end_slot {
            output.push(LogicalRowAddressRange::new(
                range.logical_fragment_id,
                cursor,
                range.end_slot,
            ));
        }
    }
    output
}

fn inspect_roaring_treemap(bytes: &[u8]) -> Result<u64> {
    if bytes.len() < std::mem::size_of::<u64>() {
        return Err(Error::invalid_input(
            "portable Roaring selection payload is shorter than its header",
        ));
    }
    let declared_domains = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    if declared_domains > MAX_SELECTION_DOMAINS as u64
        || declared_domains > (bytes.len().saturating_sub(8) / 4) as u64
    {
        return Err(Error::invalid_input(
            "portable Roaring selection declares too many domains for its payload",
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let bitmap = RoaringTreemap::deserialize_from(&mut cursor).map_err(|error| {
        Error::invalid_input(format!(
            "invalid portable Roaring selection payload: {error}"
        ))
    })?;
    if cursor.position() != bytes.len() as u64 {
        return Err(Error::invalid_input(
            "portable Roaring selection payload has trailing bytes",
        ));
    }
    if bitmap
        .max()
        .is_some_and(|raw| (raw >> 32) as u32 == INVALID_ID)
        || bitmap.bitmaps().any(|(_, slots)| slots.contains(u32::MAX))
    {
        return Err(Error::invalid_input(
            "portable Roaring selection uses a reserved logical address",
        ));
    }
    Ok(bitmap.len())
}

fn validate_roaring_selection(value: &RoaringEncodedSelection) -> Result<()> {
    if inspect_roaring_treemap(&value.bytes)? != value.cardinality {
        return Err(Error::invalid_input(
            "portable Roaring selection cardinality changed",
        ));
    }
    Ok(())
}

impl TryFrom<pb::LogicalRowAddressSelection> for LogicalRowAddressSelection {
    type Error = Error;

    fn try_from(value: pb::LogicalRowAddressSelection) -> Result<Self> {
        use pb::logical_row_address_selection::Encoding;
        if value.canonical_encoding_version != 1 {
            return Err(Error::invalid_input(format!(
                "unsupported or non-canonical logical selection encoding version: {}",
                value.canonical_encoding_version
            )));
        }
        match value
            .encoding
            .ok_or_else(|| Error::invalid_input("logical row address selection has no encoding"))?
        {
            Encoding::Ranges(ranges) => {
                let ranges = ranges
                    .ranges
                    .into_iter()
                    .map(|range| LogicalRowAddressRange {
                        logical_fragment_id: range.logical_fragment_id,
                        start_slot: range.start_slot,
                        end_slot: range.end_slot,
                    })
                    .collect::<Vec<_>>();
                let prefixes = range_prefixes(&ranges)?;
                let selection = Self::Ranges(RangeEncodedSelection {
                    ranges: ranges.into(),
                    prefixes: prefixes.into(),
                });
                selection.validate()?;
                Ok(selection)
            }
            Encoding::DenseBitset(encoded) => {
                let prefixes = validate_dense_headers(&encoded)?;
                Ok(Self::Dense(DenseEncodedSelection {
                    encoded,
                    prefixes: prefixes.into(),
                }))
            }
            Encoding::EliasFano(encoded) => {
                let prefixes = validate_elias_fano_headers(&encoded)?;
                Ok(Self::EliasFano(EliasFanoEncodedSelection {
                    encoded,
                    prefixes: prefixes.into(),
                }))
            }
            Encoding::OrdinalEliasFano(encoded) => {
                let run_universe_prefixes = validate_ordinal_elias_fano_headers(&encoded)?;
                Ok(Self::OrdinalEliasFano(OrdinalEliasFanoEncodedSelection {
                    encoded,
                    run_universe_prefixes: run_universe_prefixes.into(),
                }))
            }
            Encoding::RoaringTreemap(bytes) => {
                let cardinality = inspect_roaring_treemap(&bytes)?;
                Ok(Self::Roaring(RoaringEncodedSelection {
                    bytes: bytes.into(),
                    cardinality,
                    bitmap: Arc::new(OnceLock::new()),
                }))
            }
        }
    }
}

impl From<&LogicalRowAddressSelection> for pb::LogicalRowAddressSelection {
    fn from(value: &LogicalRowAddressSelection) -> Self {
        use pb::logical_row_address_selection::Encoding;
        let encoding = match value {
            LogicalRowAddressSelection::Ranges(value) => {
                Encoding::Ranges(pb::LogicalRowAddressRangeList {
                    ranges: value
                        .ranges
                        .iter()
                        .map(|range| pb::LogicalRowAddressRange {
                            logical_fragment_id: range.logical_fragment_id,
                            start_slot: range.start_slot,
                            end_slot: range.end_slot,
                        })
                        .collect(),
                })
            }
            LogicalRowAddressSelection::Dense(value) => {
                Encoding::DenseBitset(value.encoded.clone())
            }
            LogicalRowAddressSelection::EliasFano(value) => {
                Encoding::EliasFano(value.encoded.clone())
            }
            LogicalRowAddressSelection::OrdinalEliasFano(value) => {
                Encoding::OrdinalEliasFano(value.encoded.clone())
            }
            LogicalRowAddressSelection::Roaring(value) => {
                Encoding::RoaringTreemap(value.bytes.to_vec())
            }
        };
        Self {
            encoding: Some(encoding),
            canonical_encoding_version: 1,
        }
    }
}

fn range_prefixes(ranges: &[LogicalRowAddressRange]) -> Result<Vec<u64>> {
    let mut prefixes = Vec::with_capacity(ranges.len() + 1);
    prefixes.push(0_u64);
    let mut previous: Option<LogicalRowAddressRange> = None;
    for range in ranges {
        range.validate()?;
        if let Some(previous) = previous
            && ((range.logical_fragment_id, range.start_slot)
                <= (previous.logical_fragment_id, previous.start_slot)
                || range.logical_fragment_id == previous.logical_fragment_id
                    && range.start_slot <= previous.end_slot)
        {
            return Err(Error::invalid_input(
                "canonical logical ranges must be sorted, non-overlapping, and non-adjacent",
            ));
        }
        prefixes.push(
            prefixes
                .last()
                .and_then(|prefix| prefix.checked_add(range.len()))
                .ok_or_else(|| Error::invalid_input("logical selection cardinality overflow"))?,
        );
        previous = Some(*range);
    }
    Ok(prefixes)
}

fn validate_range_selection(value: &RangeEncodedSelection) -> Result<()> {
    if range_prefixes(&value.ranges)? != value.prefixes.as_ref() {
        return Err(Error::invalid_input(
            "logical range selection prefixes do not match its ranges",
        ));
    }
    Ok(())
}

fn range_selection_rank(value: &RangeEncodedSelection, address: LogicalRowAddress) -> Option<u64> {
    let key = (address.logical_fragment_id(), address.immutable_slot());
    let index = value
        .ranges
        .partition_point(|range| (range.logical_fragment_id, range.start_slot) <= key);
    if index == 0 {
        return None;
    }
    let range = value.ranges[index - 1];
    (range.logical_fragment_id == key.0 && key.1 < range.end_slot)
        .then(|| value.prefixes[index - 1] + key.1.saturating_sub(range.start_slot) as u64)
}

fn range_selection_select(
    value: &RangeEncodedSelection,
    ordinal: u64,
) -> Option<LogicalRowAddress> {
    let prefix_index = value.prefixes.partition_point(|prefix| *prefix <= ordinal);
    let range_index = prefix_index.checked_sub(1)?;
    let range = *value.ranges.get(range_index)?;
    LogicalRowAddress::try_new_from_parts(
        range.logical_fragment_id,
        range.start_slot + (ordinal - value.prefixes[range_index]) as u32,
    )
    .ok()
}

fn visit_range_selection_addresses(
    value: &RangeEncodedSelection,
    visit: &mut impl FnMut(LogicalRowAddress) -> Result<()>,
) -> Result<()> {
    for range in value.ranges.iter() {
        for slot in range.start_slot..range.end_slot {
            visit(LogicalRowAddress::try_new_from_parts(
                range.logical_fragment_id,
                slot,
            )?)?;
        }
    }
    Ok(())
}

fn validate_dense_headers(encoded: &pb::DenseBitsetLogicalRowAddressSelection) -> Result<Vec<u64>> {
    if encoded.domains.len() > MAX_SELECTION_DOMAINS {
        return Err(Error::invalid_input("dense selection has too many domains"));
    }
    let mut prefixes = Vec::with_capacity(encoded.domains.len() + 1);
    prefixes.push(0_u64);
    let mut previous_domain = None;
    for domain in &encoded.domains {
        if domain.logical_fragment_id == INVALID_ID
            || previous_domain.is_some_and(|previous| previous >= domain.logical_fragment_id)
            || domain.universe == 0
            || domain.bits.len() != (domain.universe as usize).div_ceil(8)
            || domain.rank_checkpoint_interval != RANK_CHECKPOINT_INTERVAL
        {
            return Err(Error::invalid_input(
                "dense logical selection has invalid domain ordering or header",
            ));
        }
        ensure_unused_high_bits_are_zero(&domain.bits, domain.universe as u64)?;
        let interval_count = (domain.universe as usize).div_ceil(RANK_CHECKPOINT_INTERVAL as usize);
        if domain.rank_checkpoints.len() != interval_count + 1
            || domain.rank_checkpoints.first() != Some(&0)
        {
            return Err(Error::invalid_input(
                "dense logical selection has invalid rank checkpoint count",
            ));
        }
        let mut rank = 0_u32;
        for interval in 0..interval_count {
            if domain.rank_checkpoints[interval] != rank {
                return Err(Error::invalid_input(
                    "dense logical selection rank checkpoint mismatch",
                ));
            }
            let start = interval as u64 * RANK_CHECKPOINT_INTERVAL as u64;
            let end = (start + RANK_CHECKPOINT_INTERVAL as u64).min(domain.universe as u64);
            rank = rank
                .checked_add(count_bits_fast(&domain.bits, start, end))
                .ok_or_else(|| Error::invalid_input("dense selection rank overflow"))?;
        }
        if domain.rank_checkpoints.last() != Some(&rank) {
            return Err(Error::invalid_input(
                "dense logical selection final rank checkpoint mismatch",
            ));
        }
        let next = prefixes
            .last()
            .and_then(|prefix| prefix.checked_add(rank as u64))
            .ok_or_else(|| Error::invalid_input("dense selection cardinality overflow"))?;
        prefixes.push(next);
        previous_domain = Some(domain.logical_fragment_id);
    }
    Ok(prefixes)
}

fn dense_selection_rank(value: &DenseEncodedSelection, address: LogicalRowAddress) -> Option<u64> {
    let domain_index = value
        .encoded
        .domains
        .binary_search_by_key(&address.logical_fragment_id(), |domain| {
            domain.logical_fragment_id
        })
        .ok()?;
    let domain = &value.encoded.domains[domain_index];
    let slot = address.immutable_slot();
    if slot >= domain.universe || !get_bit(&domain.bits, slot as u64) {
        return None;
    }
    let checkpoint_index = slot / domain.rank_checkpoint_interval;
    let checkpoint_start = checkpoint_index * domain.rank_checkpoint_interval;
    let local_rank = domain.rank_checkpoints[checkpoint_index as usize] as u64
        + count_bits_fast(&domain.bits, checkpoint_start as u64, slot as u64 + 1) as u64
        - 1;
    Some(value.prefixes[domain_index] + local_rank)
}

fn dense_selection_select(
    value: &DenseEncodedSelection,
    ordinal: u64,
) -> Option<LogicalRowAddress> {
    let prefix_index = value.prefixes.partition_point(|prefix| *prefix <= ordinal);
    let domain_index = prefix_index.checked_sub(1)?;
    let domain = value.encoded.domains.get(domain_index)?;
    let local = ordinal.checked_sub(value.prefixes[domain_index])? as u32;
    let checkpoint_index = domain
        .rank_checkpoints
        .partition_point(|rank| *rank <= local)
        .saturating_sub(1);
    let mut rank = domain.rank_checkpoints[checkpoint_index];
    let start = checkpoint_index as u32 * domain.rank_checkpoint_interval;
    let end = domain
        .universe
        .min(start.saturating_add(domain.rank_checkpoint_interval));
    for slot in start..end {
        if get_bit(&domain.bits, slot as u64) {
            if rank == local {
                return LogicalRowAddress::try_new_from_parts(domain.logical_fragment_id, slot)
                    .ok();
            }
            rank += 1;
        }
    }
    None
}

fn visit_dense_selection_addresses(
    value: &DenseEncodedSelection,
    visit: &mut impl FnMut(LogicalRowAddress) -> Result<()>,
) -> Result<()> {
    for domain in &value.encoded.domains {
        for slot in set_bit_positions(&domain.bits, domain.universe as u64) {
            visit(LogicalRowAddress::try_new_from_parts(
                domain.logical_fragment_id,
                slot as u32,
            )?)?;
        }
    }
    Ok(())
}

fn validate_elias_fano_headers(
    encoded: &pb::EliasFanoLogicalRowAddressSelection,
) -> Result<Vec<u64>> {
    if encoded.domains.len() > MAX_SELECTION_DOMAINS {
        return Err(Error::invalid_input(
            "Elias-Fano selection has too many domains",
        ));
    }
    let mut prefixes = Vec::with_capacity(encoded.domains.len() + 1);
    prefixes.push(0_u64);
    let mut previous_domain = None;
    for domain in &encoded.domains {
        if domain.logical_fragment_id == INVALID_ID
            || previous_domain.is_some_and(|previous| previous >= domain.logical_fragment_id)
            || domain.universe == 0
            || domain.cardinality == 0
            || domain.low_bit_width != elias_fano_low_bit_width(domain.universe, domain.cardinality)
            || domain.select_checkpoint_interval != SELECT_CHECKPOINT_INTERVAL
        {
            return Err(Error::invalid_input(
                "Elias-Fano logical selection has invalid domain ordering or header",
            ));
        }
        validate_elias_fano_payload(
            domain.universe as u64,
            domain.cardinality as u64,
            domain.low_bit_width,
            &domain.low_bits,
            &domain.high_bits,
            &domain
                .select_checkpoints
                .iter()
                .map(|position| *position as u64)
                .collect::<Vec<_>>(),
        )?;
        let next = prefixes
            .last()
            .and_then(|prefix| prefix.checked_add(domain.cardinality as u64))
            .ok_or_else(|| Error::invalid_input("Elias-Fano cardinality overflow"))?;
        prefixes.push(next);
        previous_domain = Some(domain.logical_fragment_id);
    }
    Ok(prefixes)
}

fn validate_elias_fano_payload(
    universe: u64,
    cardinality: u64,
    low_bit_width: u32,
    low_bits: &[u8],
    high_bits: &[u8],
    checkpoints: &[u64],
) -> Result<()> {
    let low_bit_len = cardinality
        .checked_mul(low_bit_width as u64)
        .ok_or_else(|| Error::invalid_input("Elias-Fano low-bit length overflow"))?;
    let high_bit_len = (universe >> low_bit_width)
        .checked_add(cardinality)
        .ok_or_else(|| Error::invalid_input("Elias-Fano high-bit length overflow"))?;
    let expected_low_bytes = usize::try_from(low_bit_len)
        .map_err(|_| Error::invalid_input("Elias-Fano low-bit payload exceeds memory"))?
        .div_ceil(8);
    let expected_high_bytes = usize::try_from(high_bit_len)
        .map_err(|_| Error::invalid_input("Elias-Fano high-bit payload exceeds memory"))?
        .div_ceil(8);
    if low_bits.len() != expected_low_bytes || high_bits.len() != expected_high_bytes {
        return Err(Error::invalid_input(
            "Elias-Fano payload lengths do not match header",
        ));
    }
    ensure_unused_high_bits_are_zero(low_bits, low_bit_len)?;
    ensure_unused_high_bits_are_zero(high_bits, high_bit_len)?;
    if checkpoints.len()
        != usize::try_from(cardinality)
            .unwrap_or(usize::MAX)
            .div_ceil(SELECT_CHECKPOINT_INTERVAL as usize)
    {
        return Err(Error::invalid_input(
            "Elias-Fano checkpoint count does not match cardinality",
        ));
    }
    let mut ordinal = 0_u64;
    for position in set_bit_positions(high_bits, high_bit_len) {
        if ordinal == cardinality {
            return Err(Error::invalid_input(
                "Elias-Fano high bits exceed declared cardinality",
            ));
        }
        if ordinal.is_multiple_of(SELECT_CHECKPOINT_INTERVAL as u64)
            && checkpoints[ordinal as usize / SELECT_CHECKPOINT_INTERVAL as usize] != position
        {
            return Err(Error::invalid_input(
                "Elias-Fano select checkpoint mismatch",
            ));
        }
        ordinal += 1;
    }
    if ordinal != cardinality {
        return Err(Error::invalid_input(
            "Elias-Fano high-bit cardinality mismatch",
        ));
    }
    Ok(())
}

fn elias_fano_domain_select(
    domain: &pb::EliasFanoLogicalDomainSelection,
    ordinal: u64,
) -> Option<u32> {
    if ordinal >= domain.cardinality as u64 {
        return None;
    }
    let checkpoint_index = ordinal as usize / domain.select_checkpoint_interval as usize;
    let checkpoint_ordinal = checkpoint_index as u64 * domain.select_checkpoint_interval as u64;
    let start_position = *domain.select_checkpoints.get(checkpoint_index)? as u64;
    let high_bit_len = (domain.universe as u64 >> domain.low_bit_width) + domain.cardinality as u64;
    let position = select_set_bit_from(
        &domain.high_bits,
        high_bit_len,
        start_position,
        ordinal - checkpoint_ordinal,
    )?;
    let high = position.checked_sub(ordinal)?;
    let low = read_packed_u32(&domain.low_bits, ordinal as usize, domain.low_bit_width);
    u32::try_from((high << domain.low_bit_width) | low as u64)
        .ok()
        .filter(|slot| *slot < domain.universe)
}

fn elias_fano_selection_rank(
    value: &EliasFanoEncodedSelection,
    address: LogicalRowAddress,
) -> Option<u64> {
    let domain_index = value
        .encoded
        .domains
        .binary_search_by_key(&address.logical_fragment_id(), |domain| {
            domain.logical_fragment_id
        })
        .ok()?;
    let domain = &value.encoded.domains[domain_index];
    let target = address.immutable_slot();
    let mut low = 0_u64;
    let mut high = domain.cardinality as u64;
    while low < high {
        let middle = low + (high - low) / 2;
        match elias_fano_domain_select(domain, middle)?.cmp(&target) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Some(value.prefixes[domain_index] + middle),
        }
    }
    None
}

fn elias_fano_selection_select(
    value: &EliasFanoEncodedSelection,
    ordinal: u64,
) -> Option<LogicalRowAddress> {
    let prefix_index = value.prefixes.partition_point(|prefix| *prefix <= ordinal);
    let domain_index = prefix_index.checked_sub(1)?;
    let domain = value.encoded.domains.get(domain_index)?;
    let slot = elias_fano_domain_select(domain, ordinal - value.prefixes[domain_index])?;
    LogicalRowAddress::try_new_from_parts(domain.logical_fragment_id, slot).ok()
}

fn visit_elias_fano_selection_addresses(
    value: &EliasFanoEncodedSelection,
    visit: &mut impl FnMut(LogicalRowAddress) -> Result<()>,
) -> Result<()> {
    for domain in &value.encoded.domains {
        let high_bit_len =
            (u64::from(domain.universe) >> domain.low_bit_width) + u64::from(domain.cardinality);
        let mut ordinal = 0_u64;
        let mut previous_slot = None;
        for position in set_bit_positions(&domain.high_bits, high_bit_len) {
            let high = position.checked_sub(ordinal).ok_or_else(|| {
                Error::invalid_input("Elias-Fano high position precedes its ordinal")
            })?;
            let low = u64::from(read_packed_u32(
                &domain.low_bits,
                ordinal as usize,
                domain.low_bit_width,
            ));
            let slot = (high << domain.low_bit_width) | low;
            let slot = u32::try_from(slot)
                .ok()
                .filter(|slot| *slot < domain.universe)
                .ok_or_else(|| Error::invalid_input("Elias-Fano slot exceeds its universe"))?;
            if previous_slot.is_some_and(|previous| previous >= slot) {
                return Err(Error::invalid_input(
                    "Elias-Fano slots are not strictly increasing",
                ));
            }
            visit(LogicalRowAddress::try_new_from_parts(
                domain.logical_fragment_id,
                slot,
            )?)?;
            previous_slot = Some(slot);
            ordinal += 1;
        }
        if ordinal != u64::from(domain.cardinality) {
            return Err(Error::invalid_input(
                "Elias-Fano high-bit cardinality changed while streaming",
            ));
        }
    }
    Ok(())
}

fn validate_ordinal_elias_fano_headers(
    encoded: &pb::EliasFanoOrdinalLogicalRowAddressSelection,
) -> Result<Vec<u64>> {
    if encoded.domain_runs.is_empty() || encoded.domain_runs.len() > MAX_SELECTION_DOMAINS {
        return Err(Error::invalid_input(
            "ordinal Elias-Fano selection has invalid domain-run count",
        ));
    }
    let mut prefixes = Vec::with_capacity(encoded.domain_runs.len() + 1);
    prefixes.push(0_u64);
    let mut total_domains = 0_u32;
    let mut previous_last_id = None;
    for run in &encoded.domain_runs {
        validate_compressed_domain_run(run)?;
        total_domains = total_domains
            .checked_add(run.domain_count)
            .ok_or_else(|| Error::invalid_input("ordinal Elias-Fano domain count overflow"))?;
        if total_domains > MAX_COMPRESSED_DOMAINS {
            return Err(Error::invalid_input(
                "ordinal Elias-Fano domain count exceeds format limit",
            ));
        }
        let last_id = packed_logical_id_at(run, run.domain_count - 1)?;
        if previous_last_id.is_some_and(|previous| previous >= run.first_logical_fragment_id) {
            return Err(Error::invalid_input(
                "ordinal Elias-Fano domain runs overlap or are unsorted",
            ));
        }
        let run_universe = packed_slot_prefix(
            run.slot_universes.as_ref().ok_or_else(|| {
                Error::invalid_input("ordinal Elias-Fano run is missing slot universes")
            })?,
            run.domain_count,
        )?;
        prefixes.push(
            prefixes
                .last()
                .and_then(|prefix| prefix.checked_add(run_universe))
                .ok_or_else(|| Error::invalid_input("ordinal universe overflow"))?,
        );
        previous_last_id = Some(last_id);
    }
    if prefixes.last() != Some(&encoded.universe) || encoded.cardinality == 0 {
        return Err(Error::invalid_input(
            "ordinal Elias-Fano universe does not match its domain runs",
        ));
    }
    let expected_width = if encoded.universe <= encoded.cardinality {
        0
    } else {
        (encoded.universe / encoded.cardinality).ilog2()
    };
    if encoded.low_bit_width != expected_width
        || encoded.select_checkpoint_interval != SELECT_CHECKPOINT_INTERVAL
    {
        return Err(Error::invalid_input(
            "ordinal Elias-Fano has invalid low-bit width or checkpoint interval",
        ));
    }
    validate_elias_fano_payload(
        encoded.universe,
        encoded.cardinality,
        encoded.low_bit_width,
        &encoded.low_bits,
        &encoded.high_bits,
        &encoded.select_checkpoints,
    )?;
    Ok(prefixes)
}

fn ordinal_ef_select(
    encoded: &pb::EliasFanoOrdinalLogicalRowAddressSelection,
    ordinal: u64,
) -> Option<u64> {
    if ordinal >= encoded.cardinality {
        return None;
    }
    let checkpoint_index = ordinal as usize / encoded.select_checkpoint_interval as usize;
    let checkpoint_ordinal = checkpoint_index as u64 * encoded.select_checkpoint_interval as u64;
    let start_position = *encoded.select_checkpoints.get(checkpoint_index)?;
    let high_bit_len = (encoded.universe >> encoded.low_bit_width) + encoded.cardinality;
    let position = select_set_bit_from(
        &encoded.high_bits,
        high_bit_len,
        start_position,
        ordinal - checkpoint_ordinal,
    )?;
    let high = position.checked_sub(ordinal)?;
    let low = read_packed_u64(&encoded.low_bits, ordinal as usize, encoded.low_bit_width);
    Some((high << encoded.low_bit_width) | low)
}

fn ordinal_selection_rank(
    value: &OrdinalEliasFanoEncodedSelection,
    address: LogicalRowAddress,
) -> Result<Option<u64>> {
    let Some(flattened) = ordinal_domain_slot_to_flattened(value, address)? else {
        return Ok(None);
    };
    let mut low = 0_u64;
    let mut high = value.encoded.cardinality;
    while low < high {
        let middle = low + (high - low) / 2;
        let current = ordinal_ef_select(&value.encoded, middle).ok_or_else(|| {
            Error::invalid_input("ordinal Elias-Fano select failed within cardinality")
        })?;
        match current.cmp(&flattened) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Ok(Some(middle)),
        }
    }
    Ok(None)
}

fn ordinal_selection_select(
    value: &OrdinalEliasFanoEncodedSelection,
    ordinal: u64,
) -> Result<Option<LogicalRowAddress>> {
    let Some(flattened) = ordinal_ef_select(&value.encoded, ordinal) else {
        return Ok(None);
    };
    ordinal_flattened_to_address(value, flattened)
}

struct OrdinalSelectionDomainCursor<'a> {
    runs: &'a [pb::LogicalOrdinalDomainRun],
    run_index: usize,
    domain_ordinal: u32,
    logical_fragment_id: u32,
    domain_start: u64,
    domain_end: u64,
}

impl<'a> OrdinalSelectionDomainCursor<'a> {
    fn try_new(runs: &'a [pb::LogicalOrdinalDomainRun]) -> Result<Self> {
        let run = runs
            .first()
            .ok_or_else(|| Error::invalid_input("ordinal selection has no domain runs"))?;
        let slots = run
            .slot_universes
            .as_ref()
            .ok_or_else(|| Error::invalid_input("ordinal domain run is missing slot universes"))?;
        let domain_end = u64::from(packed_slot_count_at(slots, 0)?);
        Ok(Self {
            runs,
            run_index: 0,
            domain_ordinal: 0,
            logical_fragment_id: run.first_logical_fragment_id,
            domain_start: 0,
            domain_end,
        })
    }

    fn advance_domain(&mut self) -> Result<bool> {
        use pb::packed_logical_fragment_ids::Encoding;

        let run = &self.runs[self.run_index];
        if self.domain_ordinal + 1 < run.domain_count {
            let ids = run
                .logical_fragment_ids
                .as_ref()
                .ok_or_else(|| Error::invalid_input("ordinal domain run is missing logical ids"))?;
            self.logical_fragment_id = match ids.encoding.as_ref().ok_or_else(|| {
                Error::invalid_input("packed logical fragment ids have no encoding")
            })? {
                Encoding::Consecutive(_) => self.logical_fragment_id.checked_add(1),
                Encoding::PositiveDeltas(deltas) => self
                    .logical_fragment_id
                    .checked_add(bitpacked_u32_value_at(deltas, self.domain_ordinal)?),
            }
            .filter(|id| *id != INVALID_ID)
            .ok_or_else(|| Error::invalid_input("packed logical fragment id overflow"))?;
            self.domain_ordinal += 1;
        } else {
            self.run_index += 1;
            let Some(run) = self.runs.get(self.run_index) else {
                return Ok(false);
            };
            self.domain_ordinal = 0;
            self.logical_fragment_id = run.first_logical_fragment_id;
        }

        let run = &self.runs[self.run_index];
        let slots = run
            .slot_universes
            .as_ref()
            .ok_or_else(|| Error::invalid_input("ordinal domain run is missing slot universes"))?;
        let slot_count = packed_slot_count_at(slots, self.domain_ordinal)?;
        self.domain_start = self.domain_end;
        self.domain_end = self
            .domain_end
            .checked_add(u64::from(slot_count))
            .ok_or_else(|| Error::invalid_input("ordinal domain universe overflow"))?;
        Ok(true)
    }

    fn address_for(&mut self, flattened: u64) -> Result<LogicalRowAddress> {
        while flattened >= self.domain_end {
            if !self.advance_domain()? {
                return Err(Error::invalid_input(
                    "ordinal Elias-Fano value exceeds its domain universe",
                ));
            }
        }
        if flattened < self.domain_start {
            return Err(Error::invalid_input(
                "ordinal Elias-Fano values are not strictly increasing",
            ));
        }
        let slot = u32::try_from(flattened - self.domain_start)
            .map_err(|_| Error::invalid_input("ordinal logical slot exceeds u32"))?;
        LogicalRowAddress::try_new_from_parts(self.logical_fragment_id, slot)
    }
}

fn visit_ordinal_selection_addresses(
    value: &OrdinalEliasFanoEncodedSelection,
    visit: &mut impl FnMut(LogicalRowAddress) -> Result<()>,
) -> Result<()> {
    let encoded = &value.encoded;
    let high_bit_len = (encoded.universe >> encoded.low_bit_width)
        .checked_add(encoded.cardinality)
        .ok_or_else(|| Error::invalid_input("ordinal Elias-Fano high-bit length overflow"))?;
    let low_multiplier = 1_u64
        .checked_shl(encoded.low_bit_width)
        .ok_or_else(|| Error::invalid_input("ordinal Elias-Fano low-bit width exceeds u64"))?;
    let mut domains = OrdinalSelectionDomainCursor::try_new(&encoded.domain_runs)?;
    let mut ordinal = 0_u64;
    let mut previous_flattened = None;
    for position in set_bit_positions(&encoded.high_bits, high_bit_len) {
        let high = position.checked_sub(ordinal).ok_or_else(|| {
            Error::invalid_input("ordinal Elias-Fano high position precedes its ordinal")
        })?;
        let low = read_packed_u64(&encoded.low_bits, ordinal as usize, encoded.low_bit_width);
        let flattened = high
            .checked_mul(low_multiplier)
            .and_then(|value| value.checked_add(low))
            .filter(|value| *value < encoded.universe)
            .ok_or_else(|| Error::invalid_input("ordinal Elias-Fano value exceeds its universe"))?;
        if previous_flattened.is_some_and(|previous| previous >= flattened) {
            return Err(Error::invalid_input(
                "ordinal Elias-Fano values are not strictly increasing",
            ));
        }
        visit(domains.address_for(flattened)?)?;
        previous_flattened = Some(flattened);
        ordinal += 1;
    }
    if ordinal != encoded.cardinality {
        return Err(Error::invalid_input(
            "ordinal Elias-Fano high-bit cardinality changed while streaming",
        ));
    }
    Ok(())
}

fn ordinal_domain_slot_to_flattened(
    value: &OrdinalEliasFanoEncodedSelection,
    address: LogicalRowAddress,
) -> Result<Option<u64>> {
    for (run_index, run) in value.encoded.domain_runs.iter().enumerate() {
        let ids = run
            .logical_fragment_ids
            .as_ref()
            .ok_or_else(|| Error::invalid_input("ordinal domain run is missing logical ids"))?;
        let Some(domain_ordinal) = logical_fragment_ordinal(
            run.first_logical_fragment_id,
            run.domain_count,
            ids,
            address.logical_fragment_id(),
        )?
        else {
            continue;
        };
        let slots = run
            .slot_universes
            .as_ref()
            .ok_or_else(|| Error::invalid_input("ordinal domain run is missing slot universes"))?;
        if address.immutable_slot() >= packed_slot_count_at(slots, domain_ordinal)? {
            return Ok(None);
        }
        let flattened = value.run_universe_prefixes[run_index]
            .checked_add(packed_slot_prefix(slots, domain_ordinal)?)
            .and_then(|prefix| prefix.checked_add(address.immutable_slot() as u64))
            .ok_or_else(|| Error::invalid_input("ordinal logical address overflow"))?;
        return Ok(Some(flattened));
    }
    Ok(None)
}

fn ordinal_flattened_to_address(
    value: &OrdinalEliasFanoEncodedSelection,
    flattened: u64,
) -> Result<Option<LogicalRowAddress>> {
    if flattened >= value.encoded.universe {
        return Ok(None);
    }
    let prefix_index = value
        .run_universe_prefixes
        .partition_point(|prefix| *prefix <= flattened);
    let run_index = prefix_index.saturating_sub(1);
    let run = value
        .encoded
        .domain_runs
        .get(run_index)
        .ok_or_else(|| Error::invalid_input("ordinal flattened value has no domain run"))?;
    let slots = run
        .slot_universes
        .as_ref()
        .ok_or_else(|| Error::invalid_input("ordinal domain run is missing slot universes"))?;
    let run_offset = flattened - value.run_universe_prefixes[run_index];
    let mut low = 0_u32;
    let mut high = run.domain_count;
    while low < high {
        let middle = low + (high - low) / 2;
        if packed_slot_prefix(slots, middle + 1)? <= run_offset {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let slot = u32::try_from(run_offset - packed_slot_prefix(slots, low)?)
        .map_err(|_| Error::invalid_input("ordinal logical slot exceeds u32"))?;
    LogicalRowAddress::try_new_from_parts(packed_logical_id_at(run, low)?, slot).map(Some)
}

fn set_bit_positions(bytes: &[u8], bit_len: u64) -> impl Iterator<Item = u64> + '_ {
    (0..bit_len).filter(|position| get_bit(bytes, *position))
}

fn select_set_bit_from(
    bytes: &[u8],
    bit_len: u64,
    start_position: u64,
    relative_ordinal: u64,
) -> Option<u64> {
    let mut remaining = relative_ordinal;
    for position in start_position..bit_len {
        if get_bit(bytes, position) {
            if remaining == 0 {
                return Some(position);
            }
            remaining -= 1;
        }
    }
    None
}

fn bitmap_to_ranges(
    bitmap: &RoaringTreemap,
    max_runs: usize,
) -> Option<Vec<LogicalRowAddressRange>> {
    let mut ranges = Vec::<LogicalRowAddressRange>::new();
    for (logical_fragment_id, slots) in bitmap.bitmaps() {
        let mut values = slots.iter();
        while let Some(range) = values.next_range() {
            if ranges.len() == max_runs {
                return None;
            }
            ranges.push(LogicalRowAddressRange {
                logical_fragment_id,
                start_slot: *range.start(),
                end_slot: range.end().checked_add(1)?,
            });
        }
    }
    Some(ranges)
}

type EliasFanoU64Encoding = (u32, Vec<u8>, Vec<u8>, Vec<u64>);

fn encode_elias_fano_u64_iter(
    values: impl IntoIterator<Item = u64>,
    cardinality: u64,
    universe: u64,
) -> Option<EliasFanoU64Encoding> {
    if cardinality == 0 || universe == 0 {
        return None;
    }
    let low_bit_width = if universe <= cardinality {
        0
    } else {
        (universe / cardinality).ilog2()
    };
    let low_bit_len = cardinality.checked_mul(low_bit_width as u64)?;
    let mut low_bits = vec![0_u8; usize::try_from(low_bit_len).ok()?.div_ceil(8)];
    let high_bit_len = (universe >> low_bit_width).checked_add(cardinality)?;
    let mut high_bits = vec![0_u8; usize::try_from(high_bit_len).ok()?.div_ceil(8)];
    let low_mask = if low_bit_width == 0 {
        0
    } else {
        (1_u64 << low_bit_width) - 1
    };
    let cardinality_usize = usize::try_from(cardinality).ok()?;
    let mut checkpoints =
        Vec::with_capacity(cardinality_usize.div_ceil(SELECT_CHECKPOINT_INTERVAL as usize));
    let mut previous = None;
    let mut encoded = 0_usize;
    for (ordinal, value) in values.into_iter().enumerate() {
        if ordinal >= cardinality_usize
            || value >= universe
            || previous.is_some_and(|previous| previous >= value)
        {
            return None;
        }
        write_packed_u64(&mut low_bits, ordinal, low_bit_width, value & low_mask);
        let high_position = (value >> low_bit_width).checked_add(ordinal as u64)?;
        set_bit(&mut high_bits, high_position);
        if ordinal % SELECT_CHECKPOINT_INTERVAL as usize == 0 {
            checkpoints.push(high_position);
        }
        previous = Some(value);
        encoded += 1;
    }
    if encoded != cardinality_usize {
        return None;
    }
    Some((low_bit_width, low_bits, high_bits, checkpoints))
}

fn elias_fano_low_bit_width(universe: u32, cardinality: u32) -> u32 {
    if cardinality == 0 || universe <= cardinality {
        0
    } else {
        (universe / cardinality).ilog2()
    }
}

fn set_bit(bytes: &mut [u8], bit: u64) {
    bytes[bit as usize / 8] |= 1 << (bit % 8);
}

fn set_bit_range(bytes: &mut [u8], start: u64, end: u64) {
    if start >= end {
        return;
    }
    let first_full_bit = start.div_ceil(8) * 8;
    for bit in start..end.min(first_full_bit) {
        set_bit(bytes, bit);
    }
    if first_full_bit >= end {
        return;
    }
    let last_full_bit = end / 8 * 8;
    bytes[first_full_bit as usize / 8..last_full_bit as usize / 8].fill(u8::MAX);
    for bit in last_full_bit..end {
        set_bit(bytes, bit);
    }
}

fn get_bit(bytes: &[u8], bit: u64) -> bool {
    bytes[bit as usize / 8] & (1 << (bit % 8)) != 0
}

fn count_bits(bytes: &[u8], start: u64, end: u64) -> u32 {
    (start..end).filter(|bit| get_bit(bytes, *bit)).count() as u32
}

fn count_bits_fast(bytes: &[u8], start: u64, end: u64) -> u32 {
    if start >= end {
        return 0;
    }
    let first_byte = (start / 8) as usize;
    let last_byte = ((end - 1) / 8) as usize;
    if first_byte == last_byte {
        return count_bits(bytes, start, end);
    }
    let mut count = count_bits(bytes, start, ((first_byte + 1) * 8) as u64);
    count += bytes[first_byte + 1..last_byte]
        .iter()
        .map(|byte| byte.count_ones())
        .sum::<u32>();
    count + count_bits(bytes, (last_byte * 8) as u64, end)
}

fn ensure_unused_high_bits_are_zero(bytes: &[u8], bit_len: u64) -> Result<()> {
    if !bit_len.is_multiple_of(8) {
        let used_mask = (1_u8 << (bit_len % 8)) - 1;
        if bytes.last().is_some_and(|last| last & !used_mask != 0) {
            return Err(Error::invalid_input(
                "encoded logical selection has non-zero unused high bits",
            ));
        }
    }
    Ok(())
}

fn write_packed_u32(bytes: &mut [u8], index: usize, width: u32, value: u32) {
    for bit in 0..width {
        if value & (1 << bit) != 0 {
            set_bit(bytes, (index as u64 * width as u64) + bit as u64);
        }
    }
}

fn read_packed_u32(bytes: &[u8], index: usize, width: u32) -> u32 {
    let mut value = 0_u32;
    for bit in 0..width {
        if get_bit(bytes, (index as u64 * width as u64) + bit as u64) {
            value |= 1 << bit;
        }
    }
    value
}

fn write_packed_u64(bytes: &mut [u8], index: usize, width: u32, value: u64) {
    for bit in 0..width {
        if value & (1_u64 << bit) != 0 {
            set_bit(bytes, (index as u64 * width as u64) + bit as u64);
        }
    }
}

fn read_packed_u64(bytes: &[u8], index: usize, width: u32) -> u64 {
    let mut value = 0_u64;
    for bit in 0..width {
        if get_bit(bytes, (index as u64 * width as u64) + bit as u64) {
            value |= 1_u64 << bit;
        }
    }
    value
}

const PREFIX_CHECKPOINT_INTERVAL: u32 = 128;

fn encode_bitpacked_u32(values: &[u32]) -> Option<pb::BitPackedU32Sequence> {
    let max_value = values.iter().copied().max().unwrap_or(0);
    let bit_width = if max_value == 0 {
        0
    } else {
        u32::BITS - max_value.leading_zeros()
    };
    let mut packed = vec![0_u8; (values.len() * bit_width as usize).div_ceil(8)];
    let mut prefix_checkpoints =
        Vec::with_capacity(values.len().div_ceil(PREFIX_CHECKPOINT_INTERVAL as usize) + 1);
    let mut prefix = 0_u64;
    for (index, value) in values.iter().copied().enumerate() {
        if index % PREFIX_CHECKPOINT_INTERVAL as usize == 0 {
            prefix_checkpoints.push(prefix);
        }
        write_packed_u32(&mut packed, index, bit_width, value);
        prefix = prefix.checked_add(value as u64)?;
    }
    prefix_checkpoints.push(prefix);
    Some(pb::BitPackedU32Sequence {
        value_count: u32::try_from(values.len()).ok()?,
        bit_width,
        values: packed,
        prefix_checkpoint_interval: PREFIX_CHECKPOINT_INTERVAL,
        prefix_checkpoints,
    })
}

fn validate_bitpacked_u32(encoded: &pb::BitPackedU32Sequence, expected_count: u32) -> Result<()> {
    if encoded.bit_width > u32::BITS
        || encoded.prefix_checkpoint_interval != PREFIX_CHECKPOINT_INTERVAL
        || encoded.value_count != expected_count
    {
        return Err(Error::invalid_input(
            "bit-packed u32 sequence has invalid count, bit width, or checkpoint interval",
        ));
    }
    let value_count = encoded.value_count as usize;
    let bit_len = value_count
        .checked_mul(encoded.bit_width as usize)
        .ok_or_else(|| Error::invalid_input("bit-packed u32 sequence length overflow"))?;
    if encoded.values.len() != bit_len.div_ceil(8) {
        return Err(Error::invalid_input(
            "bit-packed u32 sequence byte length does not match its header",
        ));
    }
    ensure_unused_high_bits_are_zero(&encoded.values, bit_len as u64)?;
    let expected_checkpoint_count = value_count
        .div_ceil(PREFIX_CHECKPOINT_INTERVAL as usize)
        .saturating_add(1);
    if encoded.prefix_checkpoints.len() != expected_checkpoint_count {
        return Err(Error::invalid_input(
            "bit-packed u32 sequence checkpoint count does not match its header",
        ));
    }
    let mut prefix = 0_u64;
    for index in 0..value_count {
        if index % PREFIX_CHECKPOINT_INTERVAL as usize == 0 {
            let checkpoint = index / PREFIX_CHECKPOINT_INTERVAL as usize;
            if encoded.prefix_checkpoints[checkpoint] != prefix {
                return Err(Error::invalid_input(
                    "bit-packed u32 sequence checkpoints do not match its values",
                ));
            }
        }
        let value = read_packed_u32(&encoded.values, index, encoded.bit_width);
        prefix = prefix
            .checked_add(value as u64)
            .ok_or_else(|| Error::invalid_input("bit-packed u32 sequence prefix sum overflow"))?;
    }
    if encoded.prefix_checkpoints.last() != Some(&prefix) {
        return Err(Error::invalid_input(
            "bit-packed u32 sequence checkpoints do not match its values",
        ));
    }
    Ok(())
}

fn bitpacked_u32_value_at(encoded: &pb::BitPackedU32Sequence, index: u32) -> Result<u32> {
    if index >= encoded.value_count {
        return Err(Error::invalid_input(
            "bit-packed u32 lookup exceeds value_count",
        ));
    }
    Ok(read_packed_u32(
        &encoded.values,
        index as usize,
        encoded.bit_width,
    ))
}

fn bitpacked_u32_prefix(encoded: &pb::BitPackedU32Sequence, count: u32) -> Result<u64> {
    if count > encoded.value_count {
        return Err(Error::invalid_input(
            "bit-packed u32 prefix exceeds value_count",
        ));
    }
    let interval = encoded.prefix_checkpoint_interval;
    let checkpoint_index = count / interval;
    let checkpoint_start = checkpoint_index * interval;
    let mut prefix = *encoded
        .prefix_checkpoints
        .get(checkpoint_index as usize)
        .ok_or_else(|| Error::invalid_input("bit-packed u32 checkpoint is missing"))?;
    for index in checkpoint_start..count {
        prefix = prefix
            .checked_add(bitpacked_u32_value_at(encoded, index)? as u64)
            .ok_or_else(|| Error::invalid_input("bit-packed u32 prefix sum overflow"))?;
    }
    Ok(prefix)
}

fn encode_logical_fragment_ids(ids: &[u32]) -> Option<pb::PackedLogicalFragmentIds> {
    if ids.is_empty() || ids.contains(&INVALID_ID) {
        return None;
    }
    let deltas = ids
        .windows(2)
        .map(|pair| pair[1].checked_sub(pair[0]))
        .collect::<Option<Vec<_>>>()?;
    if deltas.iter().all(|delta| *delta == 1) {
        Some(pb::PackedLogicalFragmentIds {
            encoding: Some(pb::packed_logical_fragment_ids::Encoding::Consecutive(
                pb::ConsecutiveLogicalFragmentIds {},
            )),
        })
    } else if deltas.iter().all(|delta| *delta > 0) {
        Some(pb::PackedLogicalFragmentIds {
            encoding: Some(pb::packed_logical_fragment_ids::Encoding::PositiveDeltas(
                encode_bitpacked_u32(&deltas)?,
            )),
        })
    } else {
        None
    }
}

fn validate_logical_fragment_ids(
    first_id: u32,
    domain_count: u32,
    encoded: &pb::PackedLogicalFragmentIds,
) -> Result<()> {
    use pb::packed_logical_fragment_ids::Encoding;
    if first_id == INVALID_ID || domain_count == 0 || domain_count > MAX_COMPRESSED_DOMAINS {
        return Err(Error::invalid_input(
            "packed logical fragment ids require a valid first id and non-zero count",
        ));
    }
    match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed logical fragment ids have no encoding"))?
    {
        Encoding::Consecutive(_) => {
            first_id
                .checked_add(domain_count - 1)
                .filter(|last| *last != INVALID_ID)
                .ok_or_else(|| Error::invalid_input("packed logical fragment id overflow"))?;
        }
        Encoding::PositiveDeltas(encoded) => {
            validate_bitpacked_u32(encoded, domain_count - 1)?;
            let mut id = first_id;
            for index in 0..encoded.value_count {
                let delta = bitpacked_u32_value_at(encoded, index)?;
                id = id
                    .checked_add(delta)
                    .filter(|id| delta != 0 && *id != INVALID_ID)
                    .ok_or_else(|| Error::invalid_input("packed logical fragment id overflow"))?;
            }
        }
    }
    Ok(())
}

fn logical_fragment_id_at(
    first_id: u32,
    domain_count: u32,
    encoded: &pb::PackedLogicalFragmentIds,
    ordinal: u32,
) -> Result<u32> {
    use pb::packed_logical_fragment_ids::Encoding;
    if ordinal >= domain_count {
        return Err(Error::invalid_input(
            "packed logical id ordinal is out of bounds",
        ));
    }
    let offset = match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed logical fragment ids have no encoding"))?
    {
        Encoding::Consecutive(_) => ordinal as u64,
        Encoding::PositiveDeltas(deltas) => bitpacked_u32_prefix(deltas, ordinal)?,
    };
    u32::try_from(first_id as u64 + offset)
        .ok()
        .filter(|id| *id != INVALID_ID)
        .ok_or_else(|| Error::invalid_input("packed logical fragment id overflow"))
}

fn logical_fragment_ordinal(
    first_id: u32,
    domain_count: u32,
    encoded: &pb::PackedLogicalFragmentIds,
    logical_fragment_id: u32,
) -> Result<Option<u32>> {
    use pb::packed_logical_fragment_ids::Encoding;
    match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed logical fragment ids have no encoding"))?
    {
        Encoding::Consecutive(_) => Ok(logical_fragment_id
            .checked_sub(first_id)
            .filter(|ordinal| *ordinal < domain_count)),
        Encoding::PositiveDeltas(_) => {
            let mut low = 0_u32;
            let mut high = domain_count;
            while low < high {
                let middle = low + (high - low) / 2;
                match logical_fragment_id_at(first_id, domain_count, encoded, middle)?
                    .cmp(&logical_fragment_id)
                {
                    std::cmp::Ordering::Less => low = middle + 1,
                    std::cmp::Ordering::Greater => high = middle,
                    std::cmp::Ordering::Equal => return Ok(Some(middle)),
                }
            }
            Ok(None)
        }
    }
}

fn validate_compressed_domain_run(run: &pb::LogicalOrdinalDomainRun) -> Result<()> {
    validate_logical_fragment_ids(
        run.first_logical_fragment_id,
        run.domain_count,
        run.logical_fragment_ids
            .as_ref()
            .ok_or_else(|| Error::invalid_input("ordinal domain run is missing logical ids"))?,
    )?;
    validate_slot_counts(
        run.domain_count,
        run.slot_universes
            .as_ref()
            .ok_or_else(|| Error::invalid_input("ordinal domain run is missing slot universes"))?,
    )
}

fn packed_logical_id_at(run: &pb::LogicalOrdinalDomainRun, ordinal: u32) -> Result<u32> {
    logical_fragment_id_at(
        run.first_logical_fragment_id,
        run.domain_count,
        run.logical_fragment_ids
            .as_ref()
            .ok_or_else(|| Error::invalid_input("ordinal domain run is missing logical ids"))?,
        ordinal,
    )
}

fn encode_slot_counts(counts: &[u32]) -> Option<pb::PackedSlotCounts> {
    let first = *counts.first()?;
    if first == 0 || counts.contains(&0) {
        return None;
    }
    let encoding = if counts.iter().all(|count| *count == first) {
        pb::packed_slot_counts::Encoding::UniformSlotCount(first)
    } else {
        pb::packed_slot_counts::Encoding::BitPackedSlotCounts(encode_bitpacked_u32(counts)?)
    };
    Some(pb::PackedSlotCounts {
        encoding: Some(encoding),
    })
}

fn validate_slot_counts(domain_count: u32, encoded: &pb::PackedSlotCounts) -> Result<()> {
    use pb::packed_slot_counts::Encoding;
    match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed slot counts have no encoding"))?
    {
        Encoding::UniformSlotCount(count) if *count != 0 => {}
        Encoding::BitPackedSlotCounts(values) => {
            validate_bitpacked_u32(values, domain_count)?;
            for index in 0..domain_count {
                if bitpacked_u32_value_at(values, index)? == 0 {
                    return Err(Error::invalid_input("packed slot count contains zero"));
                }
            }
        }
        _ => return Err(Error::invalid_input("packed slot count contains zero")),
    }
    Ok(())
}

fn packed_slot_count_at(encoded: &pb::PackedSlotCounts, ordinal: u32) -> Result<u32> {
    use pb::packed_slot_counts::Encoding;
    match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed slot counts have no encoding"))?
    {
        Encoding::UniformSlotCount(count) => Ok(*count),
        Encoding::BitPackedSlotCounts(values) => bitpacked_u32_value_at(values, ordinal),
    }
}

fn packed_slot_prefix(encoded: &pb::PackedSlotCounts, count: u32) -> Result<u64> {
    use pb::packed_slot_counts::Encoding;
    match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed slot counts have no encoding"))?
    {
        Encoding::UniformSlotCount(slot_count) => (*slot_count as u64)
            .checked_mul(count as u64)
            .ok_or_else(|| Error::invalid_input("packed slot count prefix overflow")),
        Encoding::BitPackedSlotCounts(values) => bitpacked_u32_prefix(values, count),
    }
}

fn encode_creation_versions(versions: &[u64]) -> Option<pb::PackedCreationVersions> {
    let first = *versions.first()?;
    if first == 0 || versions.contains(&0) {
        return None;
    }
    let encoding = if versions.iter().all(|version| *version == first) {
        pb::packed_creation_versions::Encoding::UniformCreationVersion(first)
    } else {
        let mut runs = Vec::<pb::CreationVersionRun>::new();
        for version in versions {
            if let Some(run) = runs.last_mut()
                && run.creation_version == *version
            {
                run.domain_count = run.domain_count.checked_add(1)?;
            } else {
                runs.push(pb::CreationVersionRun {
                    domain_count: 1,
                    creation_version: *version,
                });
            }
        }
        pb::packed_creation_versions::Encoding::Runs(pb::CreationVersionRuns { runs })
    };
    Some(pb::PackedCreationVersions {
        encoding: Some(encoding),
    })
}

fn validate_creation_versions(
    domain_count: u32,
    encoded: &pb::PackedCreationVersions,
) -> Result<Vec<u32>> {
    use pb::packed_creation_versions::Encoding;
    match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed creation versions have no encoding"))?
    {
        Encoding::UniformCreationVersion(version) if *version != 0 => Ok(Vec::new()),
        Encoding::Runs(runs) => {
            let mut ends = Vec::with_capacity(runs.runs.len());
            let mut end = 0_u32;
            for run in &runs.runs {
                if run.domain_count == 0 || run.creation_version == 0 {
                    return Err(Error::invalid_input(
                        "packed creation-version run has a zero count or version",
                    ));
                }
                end = end.checked_add(run.domain_count).ok_or_else(|| {
                    Error::invalid_input("packed creation-version run count overflow")
                })?;
                ends.push(end);
            }
            if end != domain_count {
                return Err(Error::invalid_input(
                    "packed creation versions do not match domain_count",
                ));
            }
            Ok(ends)
        }
        _ => Err(Error::invalid_input(
            "packed creation version must be non-zero",
        )),
    }
}

fn creation_version_at(
    encoded: &pb::PackedCreationVersions,
    run_ends: &[u32],
    ordinal: u32,
) -> Result<u64> {
    use pb::packed_creation_versions::Encoding;
    match encoded
        .encoding
        .as_ref()
        .ok_or_else(|| Error::invalid_input("packed creation versions have no encoding"))?
    {
        Encoding::UniformCreationVersion(version) => Ok(*version),
        Encoding::Runs(runs) => {
            let index = run_ends.partition_point(|end| *end <= ordinal);
            runs.runs
                .get(index)
                .map(|run| run.creation_version)
                .ok_or_else(|| Error::invalid_input("creation-version ordinal is out of bounds"))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct NativeLogicalDomain {
    pub logical_fragment_id: u32,
    pub creation_version: u64,
}

impl NativeLogicalDomain {
    pub fn new(logical_fragment_id: u32, creation_version: u64) -> Result<Self> {
        if logical_fragment_id == INVALID_ID {
            return Err(Error::invalid_input(
                "native logical domain uses the reserved logical fragment id",
            ));
        }
        if creation_version == 0 {
            return Err(Error::invalid_input(
                "native logical domain creation_version must be non-zero",
            ));
        }
        Ok(Self {
            logical_fragment_id,
            creation_version,
        })
    }
}

impl TryFrom<pb::NativeLogicalDomain> for NativeLogicalDomain {
    type Error = Error;

    fn try_from(value: pb::NativeLogicalDomain) -> Result<Self> {
        Self::new(value.logical_fragment_id, value.creation_version)
    }
}

impl From<&NativeLogicalDomain> for pb::NativeLogicalDomain {
    fn from(value: &NativeLogicalDomain) -> Self {
        Self {
            logical_fragment_id: value.logical_fragment_id,
            creation_version: value.creation_version,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, DeepSizeOf)]
pub struct RowAddressLogicalDomain {
    pub logical_fragment_id: u32,
    pub slot_count: u32,
    pub creation_version: u64,
}

impl RowAddressLogicalDomain {
    pub fn new(logical_fragment_id: u32, slot_count: u32, creation_version: u64) -> Result<Self> {
        if logical_fragment_id == INVALID_ID || slot_count == 0 || creation_version == 0 {
            return Err(Error::invalid_input(format!(
                "logical domain must have a valid id, non-zero slot_count, and non-zero creation_version: logical_fragment_id={}, slot_count={}, creation_version={}",
                logical_fragment_id, slot_count, creation_version
            )));
        }
        Ok(Self {
            logical_fragment_id,
            slot_count,
            creation_version,
        })
    }
}

impl TryFrom<pb::RowAddressLogicalDomain> for RowAddressLogicalDomain {
    type Error = Error;

    fn try_from(value: pb::RowAddressLogicalDomain) -> Result<Self> {
        Self::new(
            value.logical_fragment_id,
            value.slot_count,
            value.creation_version,
        )
    }
}

impl From<&RowAddressLogicalDomain> for pb::RowAddressLogicalDomain {
    fn from(value: &RowAddressLogicalDomain) -> Self {
        Self {
            logical_fragment_id: value.logical_fragment_id,
            slot_count: value.slot_count,
            creation_version: value.creation_version,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct DirectRowAddressPlacement {
    pub source: RowAddressLogicalDomain,
    pub destination_fragment_id: u32,
    pub destination_start: u32,
    pub excluded: Option<Arc<LogicalRowAddressSelection>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedLogicalDomainRun {
    first_logical_fragment_id: u32,
    domain_count: u32,
    logical_fragment_ids: pb::PackedLogicalFragmentIds,
    slot_counts: pb::PackedSlotCounts,
    creation_versions: pb::PackedCreationVersions,
    creation_run_ends: Arc<[u32]>,
}

impl Eq for PackedLogicalDomainRun {}

impl DeepSizeOf for PackedLogicalDomainRun {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.logical_fragment_ids.encoded_len()
            + self.slot_counts.encoded_len()
            + self.creation_versions.encoded_len()
            + self.creation_run_ends.deep_size_of_children(context)
    }
}

impl PackedLogicalDomainRun {
    fn try_new(
        first_logical_fragment_id: u32,
        domain_count: u32,
        logical_fragment_ids: pb::PackedLogicalFragmentIds,
        slot_counts: pb::PackedSlotCounts,
        creation_versions: pb::PackedCreationVersions,
    ) -> Result<Self> {
        validate_logical_fragment_ids(
            first_logical_fragment_id,
            domain_count,
            &logical_fragment_ids,
        )?;
        validate_slot_counts(domain_count, &slot_counts)?;
        let creation_run_ends = validate_creation_versions(domain_count, &creation_versions)?;
        Ok(Self {
            first_logical_fragment_id,
            domain_count,
            logical_fragment_ids,
            slot_counts,
            creation_versions,
            creation_run_ends: creation_run_ends.into(),
        })
    }

    pub fn from_sources(sources: &[RowAddressLogicalDomain]) -> Result<Self> {
        validate_strictly_sorted_domains(sources)?;
        let ids = sources
            .iter()
            .map(|source| source.logical_fragment_id)
            .collect::<Vec<_>>();
        let counts = sources
            .iter()
            .map(|source| source.slot_count)
            .collect::<Vec<_>>();
        let versions = sources
            .iter()
            .map(|source| source.creation_version)
            .collect::<Vec<_>>();
        Self::try_new(
            *ids.first()
                .ok_or_else(|| Error::invalid_input("PackedRun requires source domains"))?,
            u32::try_from(ids.len())
                .map_err(|_| Error::invalid_input("PackedRun has too many source domains"))?,
            encode_logical_fragment_ids(&ids)
                .ok_or_else(|| Error::invalid_input("cannot encode PackedRun logical ids"))?,
            encode_slot_counts(&counts)
                .ok_or_else(|| Error::invalid_input("cannot encode PackedRun slot counts"))?,
            encode_creation_versions(&versions)
                .ok_or_else(|| Error::invalid_input("cannot encode PackedRun creation versions"))?,
        )
    }

    pub fn domain_count(&self) -> u32 {
        self.domain_count
    }

    pub fn first_logical_fragment_id(&self) -> u32 {
        self.first_logical_fragment_id
    }

    pub fn last_logical_fragment_id(&self) -> Result<u32> {
        self.logical_fragment_id_at(self.domain_count - 1)
    }

    pub fn logical_fragment_id_at(&self, ordinal: u32) -> Result<u32> {
        logical_fragment_id_at(
            self.first_logical_fragment_id,
            self.domain_count,
            &self.logical_fragment_ids,
            ordinal,
        )
    }

    pub fn domain_ordinal(&self, logical_fragment_id: u32) -> Result<Option<u32>> {
        logical_fragment_ordinal(
            self.first_logical_fragment_id,
            self.domain_count,
            &self.logical_fragment_ids,
            logical_fragment_id,
        )
    }

    pub fn slot_count_at(&self, ordinal: u32) -> Result<u32> {
        if ordinal >= self.domain_count {
            return Err(Error::invalid_input(
                "PackedRun domain ordinal is out of bounds",
            ));
        }
        packed_slot_count_at(&self.slot_counts, ordinal)
    }

    pub fn slot_prefix(&self, count: u32) -> Result<u64> {
        if count > self.domain_count {
            return Err(Error::invalid_input(
                "PackedRun slot prefix is out of bounds",
            ));
        }
        packed_slot_prefix(&self.slot_counts, count)
    }

    pub fn total_slot_count(&self) -> Result<u64> {
        self.slot_prefix(self.domain_count)
    }

    pub fn domain_at(&self, ordinal: u32) -> Result<RowAddressLogicalDomain> {
        RowAddressLogicalDomain::new(
            self.logical_fragment_id_at(ordinal)?,
            self.slot_count_at(ordinal)?,
            creation_version_at(&self.creation_versions, &self.creation_run_ends, ordinal)?,
        )
    }

    fn ordinal_for_slot_offset(&self, offset: u64) -> Result<Option<(u32, u32)>> {
        if offset >= self.total_slot_count()? {
            return Ok(None);
        }
        use pb::packed_slot_counts::Encoding;
        let ordinal = match self
            .slot_counts
            .encoding
            .as_ref()
            .ok_or_else(|| Error::invalid_input("packed slot counts have no encoding"))?
        {
            Encoding::UniformSlotCount(count) => u32::try_from(offset / *count as u64)
                .map_err(|_| Error::invalid_input("PackedRun ordinal overflow"))?,
            Encoding::BitPackedSlotCounts(_) => {
                let mut low = 0_u32;
                let mut high = self.domain_count;
                while low < high {
                    let middle = low + (high - low) / 2;
                    if self.slot_prefix(middle + 1)? <= offset {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                low
            }
        };
        let local = u32::try_from(offset - self.slot_prefix(ordinal)?)
            .map_err(|_| Error::invalid_input("PackedRun local slot overflow"))?;
        Ok(Some((ordinal, local)))
    }

    fn compressed_slice(&self, start: u32, end: u32) -> Result<Option<Self>> {
        if start >= end || end > self.domain_count {
            return Err(Error::invalid_input(
                "PackedRun compressed slice is empty or out of bounds",
            ));
        }
        if start == 0 && end == self.domain_count {
            return Ok(Some(self.clone()));
        }
        if !matches!(
            self.logical_fragment_ids.encoding,
            Some(pb::packed_logical_fragment_ids::Encoding::Consecutive(_))
        ) || !matches!(
            self.slot_counts.encoding,
            Some(pb::packed_slot_counts::Encoding::UniformSlotCount(_))
        ) {
            let domains = (start..end)
                .map(|ordinal| self.domain_at(ordinal))
                .collect::<Result<Vec<_>>>()?;
            return Self::from_sources(&domains).map(Some);
        }
        let creation_versions = match self
            .creation_versions
            .encoding
            .as_ref()
            .ok_or_else(|| Error::invalid_input("PackedRun creation versions have no encoding"))?
        {
            pb::packed_creation_versions::Encoding::UniformCreationVersion(version) => {
                pb::PackedCreationVersions {
                    encoding: Some(
                        pb::packed_creation_versions::Encoding::UniformCreationVersion(*version),
                    ),
                }
            }
            pb::packed_creation_versions::Encoding::Runs(runs) => {
                let mut sliced = Vec::<pb::CreationVersionRun>::new();
                let mut run_start = 0_u32;
                for run in &runs.runs {
                    let run_end = run_start
                        .checked_add(run.domain_count)
                        .ok_or_else(|| Error::invalid_input("creation-version run end overflow"))?;
                    let overlap_start = run_start.max(start);
                    let overlap_end = run_end.min(end);
                    if overlap_start < overlap_end {
                        let count = overlap_end - overlap_start;
                        if let Some(previous) = sliced.last_mut()
                            && previous.creation_version == run.creation_version
                        {
                            previous.domain_count =
                                previous.domain_count.checked_add(count).ok_or_else(|| {
                                    Error::invalid_input("creation-version slice overflow")
                                })?;
                        } else {
                            sliced.push(pb::CreationVersionRun {
                                domain_count: count,
                                creation_version: run.creation_version,
                            });
                        }
                    }
                    run_start = run_end;
                    if run_start >= end {
                        break;
                    }
                }
                pb::PackedCreationVersions {
                    encoding: Some(pb::packed_creation_versions::Encoding::Runs(
                        pb::CreationVersionRuns { runs: sliced },
                    )),
                }
            }
        };
        Self::try_new(
            self.logical_fragment_id_at(start)?,
            end - start,
            pb::PackedLogicalFragmentIds {
                encoding: Some(pb::packed_logical_fragment_ids::Encoding::Consecutive(
                    pb::ConsecutiveLogicalFragmentIds {},
                )),
            },
            self.slot_counts.clone(),
            creation_versions,
        )
        .map(Some)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct PackedRunRowAddressPlacement {
    pub domains: PackedLogicalDomainRun,
    pub destination_fragment_id: u32,
    pub destination_start: u32,
}

impl PackedRunRowAddressPlacement {
    pub fn from_sources(
        sources: Vec<RowAddressLogicalDomain>,
        destination_fragment_id: u32,
        destination_start: u32,
    ) -> Result<Self> {
        Ok(Self {
            domains: PackedLogicalDomainRun::from_sources(&sources)?,
            destination_fragment_id,
            destination_start,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum RetiredLogicalRowMembership {
    AllRows,
    Selection(Arc<LogicalRowAddressSelection>),
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct RetiredLogicalRowSet {
    pub domains: PackedLogicalDomainRun,
    pub membership: RetiredLogicalRowMembership,
}

impl RetiredLogicalRowSet {
    pub fn all_rows(sources: Vec<RowAddressLogicalDomain>) -> Result<Self> {
        Ok(Self {
            domains: PackedLogicalDomainRun::from_sources(&sources)?,
            membership: RetiredLogicalRowMembership::AllRows,
        })
    }

    pub fn selected(
        sources: Vec<RowAddressLogicalDomain>,
        selection: Arc<LogicalRowAddressSelection>,
    ) -> Result<Self> {
        let value = Self {
            domains: PackedLogicalDomainRun::from_sources(&sources)?,
            membership: RetiredLogicalRowMembership::Selection(selection),
        };
        value.validate()?;
        Ok(value)
    }

    fn source_domain(&self, logical_fragment_id: u32) -> Result<Option<RowAddressLogicalDomain>> {
        self.domains
            .domain_ordinal(logical_fragment_id)?
            .map(|ordinal| self.domains.domain_at(ordinal))
            .transpose()
    }

    fn contains(&self, address: LogicalRowAddress) -> Result<bool> {
        let Some(source) = self.source_domain(address.logical_fragment_id())? else {
            return Ok(false);
        };
        if address.immutable_slot() >= source.slot_count {
            return Ok(false);
        }
        match &self.membership {
            RetiredLogicalRowMembership::AllRows => Ok(true),
            RetiredLogicalRowMembership::Selection(selection) => selection.contains(address),
        }
    }

    fn validate(&self) -> Result<()> {
        if let RetiredLogicalRowMembership::Selection(selection) = &self.membership {
            if selection.is_empty() {
                return Err(Error::invalid_input(
                    "selected retired logical rows must not be empty",
                ));
            }
            for (logical_fragment_id, slots) in selection.to_roaring_treemap()?.bitmaps() {
                let Some(source) = self.source_domain(logical_fragment_id)? else {
                    return Err(Error::invalid_input(format!(
                        "retired logical domain {logical_fragment_id} has no domain metadata"
                    )));
                };
                if slots.max().is_some_and(|slot| slot >= source.slot_count) {
                    return Err(Error::invalid_input(format!(
                        "retired logical domain {logical_fragment_id} exceeds slot_count {}",
                        source.slot_count,
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct SelectedRowAddressPlacement {
    pub source: RowAddressLogicalDomain,
    pub selection: Arc<LogicalRowAddressSelection>,
    pub destination_fragment_id: u32,
    pub destination_start: u32,
    pub excluded: Option<Arc<LogicalRowAddressSelection>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub struct RowAddressExtent {
    pub source_start: u32,
    pub length: u32,
    pub destination_fragment_id: u32,
    pub destination_start: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ExtentListRowAddressPlacement {
    pub source: RowAddressLogicalDomain,
    pub extents: Vec<RowAddressExtent>,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct SparseSelectionSource {
    pub source: RowAddressLogicalDomain,
    pub selection: Arc<LogicalRowAddressSelection>,
    pub excluded: Option<Arc<LogicalRowAddressSelection>>,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct SparseSelectionRowAddressPlacement {
    pub sources: Vec<SparseSelectionSource>,
    pub destination_fragment_id: u32,
    pub destination_start: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ExplicitMapPage {
    pub first_logical_address: u64,
    pub last_logical_address: u64,
    pub row_start: u64,
    pub row_count: u64,
    pub content_fingerprint: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ExplicitMapRowIdPage {
    pub row_start: u64,
    pub row_count: u64,
    pub content_fingerprint: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ExplicitMapDestination {
    pub physical_fragment_id: u32,
    pub destination_start: u32,
    pub row_count: u32,
    pub row_id_file_path: String,
    pub row_id_file_size: u64,
    pub row_id_pages: Vec<ExplicitMapRowIdPage>,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ExplicitMapRowAddressPlacement {
    pub sources: Vec<SparseSelectionSource>,
    pub object_path: String,
    pub object_size: u64,
    pub pages: Vec<ExplicitMapPage>,
    pub destinations: Vec<ExplicitMapDestination>,
    /// Dataset-root base that owns the locator and hidden `_rowid` files.
    /// `None` means the current dataset root.
    pub base_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum RowAddressPlacement {
    Direct(DirectRowAddressPlacement),
    PackedRun(PackedRunRowAddressPlacement),
    Selected(SelectedRowAddressPlacement),
    ExtentList(ExtentListRowAddressPlacement),
    SparseSelection(SparseSelectionRowAddressPlacement),
    ExplicitMap(ExplicitMapRowAddressPlacement),
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct RowAddressDestinationIndexEntry {
    pub physical_fragment_id: u32,
    pub placement_indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, DeepSizeOf)]
pub struct FieldGeneration {
    pub field_id: i32,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ContentGenerationRegion {
    pub selection: Arc<LogicalRowAddressSelection>,
    pub field_ids: Vec<i32>,
    pub generation: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, DeepSizeOf)]
pub struct RowAddressPlacementDebtSummary {
    pub canonical_layout_bytes: u64,
    pub metadata_bytes_written_since_maintenance: u64,
    pub max_extents_per_logical_fragment: u32,
    pub live_physical_rows: u64,
    pub total_physical_rows: u64,
    pub explicit_layout_bytes: u64,
    pub fast_delta_bytes: u64,
    pub explicit_delta_bytes: u64,
    pub explicit_metadata_bytes_written_since_maintenance: u64,
    pub generation_delta_bytes: u64,
    pub generation_metadata_bytes_written_since_maintenance: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct PhysicalRowOwnershipSummary {
    pub physical_fragment_id: u32,
    pub mapped_row_count: u64,
    pub mapped_offsets_fingerprint: Vec<u8>,
    pub deletion_offsets_fingerprint: Vec<u8>,
    pub unowned_row_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowAddressLayout {
    pub encoding_version: u32,
    pub namespace_uuid: Uuid,
    pub placements: Vec<RowAddressPlacement>,
    pub destination_index: Vec<RowAddressDestinationIndexEntry>,
    pub field_default_generations: Vec<FieldGeneration>,
    pub generation_regions: Vec<ContentGenerationRegion>,
    pub index_commit_floors: Vec<FieldGeneration>,
    pub debt_summary: RowAddressPlacementDebtSummary,
    pub fingerprint: Vec<u8>,
    pub physical_row_ownership: Vec<PhysicalRowOwnershipSummary>,
    pub selection_pool: Vec<Arc<LogicalRowAddressSelection>>,
    pub retired_rows: Vec<RetiredLogicalRowSet>,
    pub logical_domain_fingerprint: Vec<u8>,
}

/// An immutable view of a row-address layout whose destination index has been
/// checked against every persisted placement.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedRowAddressDestinationIndex<'a> {
    layout: &'a RowAddressLayout,
}

impl<'a> ValidatedRowAddressDestinationIndex<'a> {
    /// Verify one fragment using only the placements referenced by its
    /// destination-index entry.
    pub fn verify_visibility(
        &self,
        fragment: &Fragment,
        deleted_offsets: &RoaringBitmap,
    ) -> Result<()> {
        self.layout
            .verify_visibility_from_index(fragment, deleted_offsets)
    }

    /// Return each placement referenced by the requested physical fragments
    /// exactly once.
    pub fn placements_for_fragments(
        &self,
        fragment_ids: impl IntoIterator<Item = u32>,
    ) -> Vec<&'a RowAddressPlacement> {
        let mut placement_indices = BTreeSet::new();
        for fragment_id in fragment_ids {
            if let Ok(index) = self
                .layout
                .destination_index
                .binary_search_by_key(&fragment_id, |entry| entry.physical_fragment_id)
            {
                placement_indices.extend(
                    self.layout.destination_index[index]
                        .placement_indices
                        .iter()
                        .copied(),
                );
            }
        }
        placement_indices
            .into_iter()
            .map(|index| &self.layout.placements[index as usize])
            .collect()
    }
}

impl DeepSizeOf for RowAddressLayout {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.placements.deep_size_of_children(context)
            + self.destination_index.deep_size_of_children(context)
            + self
                .field_default_generations
                .deep_size_of_children(context)
            + self.generation_regions.deep_size_of_children(context)
            + self.index_commit_floors.deep_size_of_children(context)
            + self.debt_summary.deep_size_of_children(context)
            + self.fingerprint.deep_size_of_children(context)
            + self.physical_row_ownership.deep_size_of_children(context)
            + self.selection_pool.deep_size_of_children(context)
            + self.retired_rows.deep_size_of_children(context)
            + self
                .logical_domain_fingerprint
                .deep_size_of_children(context)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct RowAddressFieldChange {
    pub selection: LogicalRowAddressSelection,
    pub field_ids: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub struct RowAddressSourceFloor {
    pub field_id: i32,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum RowAddressTargetFragment {
    NewFragmentOrdinal(u32),
    ExistingFragmentId(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub struct RowAddressTargetRange {
    pub fragment: RowAddressTargetFragment,
    pub start_offset: u32,
    pub end_offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum RowAddressPlacementKind {
    Direct,
    PackedRun,
    Selected,
    ExtentList,
    SparseSelection,
    ExplicitMap,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct RowAddressPlacementDelta {
    pub source_selections: Vec<LogicalRowAddressSelection>,
    pub target: RowAddressTargetRange,
    pub placement_kind: RowAddressPlacementKind,
    pub output_cardinality: u64,
    pub output_row_sequence_fingerprint: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct ReplacedContentGeneration {
    pub selection: LogicalRowAddressSelection,
    pub field_ids: Vec<i32>,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct RowAlignedRewriteProof {
    pub physical_fragment_id: u32,
    pub physical_rows: u32,
    pub mapped_offsets_fingerprint: Vec<u8>,
    pub deletion_offsets_fingerprint: Option<Vec<u8>>,
    pub field_change_index: usize,
    pub source_floor_indices: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowAddressLayoutDelta {
    pub source_domains: Vec<RowAddressLogicalDomain>,
    pub placements: Vec<RowAddressPlacementDelta>,
    pub retired_selections: Vec<LogicalRowAddressSelection>,
    pub field_changes: Vec<RowAddressFieldChange>,
    pub source_floors: Vec<RowAddressSourceFloor>,
    pub expected_layout_fingerprint: Vec<u8>,
    pub replaced_generations: Vec<ReplacedContentGeneration>,
    pub row_aligned_rewrite_proofs: Vec<RowAlignedRewriteProof>,
    pub create_namespace_uuid: Option<Uuid>,
    /// External locator metadata keyed by the corresponding placement delta.
    /// This is transaction provenance and is copied into the manifest root
    /// only when the placement kind is [`RowAddressPlacementKind::ExplicitMap`].
    pub explicit_map_placements: BTreeMap<usize, ExplicitMapRowAddressPlacement>,
}

pub struct RowAddressDeltaApplyContext<'a> {
    pub current_fragments: &'a [Fragment],
    pub successor_fragments: &'a [Fragment],
    pub resolved_new_fragment_ids: &'a BTreeMap<u32, u32>,
    /// Deletion vectors from the source snapshot. These are the authority for
    /// validating logical identities explicitly retired by maintenance.
    pub current_deletion_vectors: &'a BTreeMap<u32, &'a RoaringBitmap>,
    /// Source fragments fully deleted by this transaction. Retirement may
    /// consume their still-live rows only when the successor drops the entire
    /// fragment.
    pub newly_fully_deleted_source_fragments: &'a BTreeSet<u32>,
    /// Deletion vectors in the successor snapshot, used to validate physical
    /// ownership summaries for target-only fragments.
    pub deletion_vectors: &'a BTreeMap<u32, &'a RoaringBitmap>,
    pub explicit_map_placements: &'a BTreeMap<usize, ExplicitMapRowAddressPlacement>,
    pub commit_version: u64,
    pub current_max_logical_fragment_id: Option<u32>,
    pub max_logical_fragment_id: Option<u32>,
    // Actual row-address metadata bytes written by the transaction across the
    // manifest, native markers, index coverage, and framing.
    pub row_address_metadata_bytes_written: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowAddressAdmissionMetrics {
    // Layout-only diagnostic. This is not the admission Delta, which the
    // transaction computes from the complete canonical successor manifest.
    pub projected_layout_bytes: u64,
    pub max_extent_fanout: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexGenerationBlocker {
    pub index_id: Uuid,
    pub index_name: String,
    pub field_ids: Vec<i32>,
    pub oldest_generation: u64,
    pub region_bytes: u64,
    pub blocked_transaction_start: u64,
    pub blocked_transaction_end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementMaintenanceRequired {
    ProjectedDeltaBytes {
        projected: u64,
        limit: u64,
    },
    ProjectedEpochBytes {
        projected: u64,
        limit: u64,
    },
    ExtentFanout {
        logical_fragment_id: u32,
        projected: u32,
        limit: u32,
    },
    ExistingExplicitMapRequiresRewrite {
        logical_fragment_id: u32,
    },
    ExplicitMapMetadataRequired {
        placement_delta_index: usize,
    },
    SelectionSubtractionRequiresRewrite {
        logical_fragment_id: u32,
    },
    PackedRunSubtractionRequiresRewrite {
        logical_fragment_id: u32,
    },
    LogicalOrderRequiresRewrite {
        previous_address: u64,
        next_address: u64,
    },
    IndexGenerationBlocked {
        projected_delta_bytes: u64,
        delta_limit: u64,
        projected_epoch_bytes: u64,
        epoch_limit: u64,
        generation_delta_bytes: u64,
        generation_epoch_bytes: u64,
        blocking_indices: Vec<IndexGenerationBlocker>,
    },
}

impl std::fmt::Display for PlacementMaintenanceRequired {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LogicalOrderRequiresRewrite {
                previous_address,
                next_address,
            } => write!(
                formatter,
                "default maintenance cannot reorder logical row addresses \
                 ({previous_address} before {next_address}); use explicit Recluster"
            ),
            _ => write!(formatter, "{self:?}"),
        }
    }
}

impl std::error::Error for PlacementMaintenanceRequired {}

pub fn evaluate_projected_row_address_delta(
    projected_delta_bytes: u64,
) -> Option<PlacementMaintenanceRequired> {
    (projected_delta_bytes > ROW_ADDRESS_B_FAST).then_some(
        PlacementMaintenanceRequired::ProjectedDeltaBytes {
            projected: projected_delta_bytes,
            limit: ROW_ADDRESS_B_FAST,
        },
    )
}

pub fn evaluate_projected_row_address_epoch(
    projected_epoch_bytes: u64,
) -> Option<PlacementMaintenanceRequired> {
    (projected_epoch_bytes > ROW_ADDRESS_W_FAST).then_some(
        PlacementMaintenanceRequired::ProjectedEpochBytes {
            projected: projected_epoch_bytes,
            limit: ROW_ADDRESS_W_FAST,
        },
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowAddressLayoutApplyResult {
    Admitted {
        layout: Box<RowAddressLayout>,
        metrics: RowAddressAdmissionMetrics,
    },
    NotAdmitted {
        reason: PlacementMaintenanceRequired,
        metrics: RowAddressAdmissionMetrics,
    },
}

impl DeepSizeOf for RowAddressLayoutDelta {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.source_domains.deep_size_of_children(context)
            + self.placements.deep_size_of_children(context)
            + self.retired_selections.deep_size_of_children(context)
            + self.field_changes.deep_size_of_children(context)
            + self.source_floors.deep_size_of_children(context)
            + self
                .expected_layout_fingerprint
                .deep_size_of_children(context)
            + self.replaced_generations.deep_size_of_children(context)
            + self
                .row_aligned_rewrite_proofs
                .deep_size_of_children(context)
            + self.explicit_map_placements.deep_size_of_children(context)
    }
}

impl RowAddressLayoutDelta {
    pub fn for_create(namespace_uuid: Uuid) -> Self {
        Self {
            create_namespace_uuid: Some(namespace_uuid),
            ..Self::default()
        }
    }

    pub fn has_explicit_placements(&self) -> bool {
        self.placements
            .iter()
            .any(|placement| placement.placement_kind == RowAddressPlacementKind::ExplicitMap)
            || !self.explicit_map_placements.is_empty()
    }

    /// Return whether every row-address change belongs to one explicit rewrite.
    ///
    /// Source domains, the exact replacement retirement mask, and the expected
    /// root fingerprint are provenance for the explicit rewrite itself.
    /// Generation, namespace, or fast placement changes may not be mixed into
    /// this admission-exempt tier.
    pub fn is_pure_explicit_rewrite(&self) -> bool {
        !self.placements.is_empty()
            && self
                .placements
                .iter()
                .all(|placement| placement.placement_kind == RowAddressPlacementKind::ExplicitMap)
            && self.explicit_map_placements.len() == self.placements.len()
            && self
                .explicit_map_placements
                .keys()
                .copied()
                .eq(0..self.placements.len())
            && self.field_changes.is_empty()
            && self.source_floors.is_empty()
            && self.replaced_generations.is_empty()
            && self.row_aligned_rewrite_proofs.is_empty()
            && self.create_namespace_uuid.is_none()
    }

    fn validate_admission_tier(&self) -> Result<()> {
        if self.has_explicit_placements() && !self.is_pure_explicit_rewrite() {
            return Err(Error::invalid_input(
                "ExplicitMap rewrite provenance cannot be mixed with fast placement, generation, or namespace changes",
            ));
        }
        Ok(())
    }

    /// Remove ExplicitMap placement payloads. The exact replacement retirement
    /// mask belongs to the same explicit tier and is removed with a pure
    /// explicit rewrite; shared fast-path retirement metadata is retained.
    pub fn fast_admission_projection(&self) -> Self {
        let mut projected = self.clone();
        let mut retained_placements = Vec::with_capacity(projected.placements.len());
        for placement in projected.placements {
            if placement.placement_kind != RowAddressPlacementKind::ExplicitMap {
                retained_placements.push(placement);
            }
        }
        projected.placements = retained_placements;
        projected.explicit_map_placements.clear();
        if self.is_pure_explicit_rewrite() {
            projected.retired_selections.clear();
        }
        projected
    }

    pub fn validate_row_aligned_rewrite_proofs(&self) -> Result<()> {
        if self.row_aligned_rewrite_proofs.is_empty() {
            return Ok(());
        }
        if !self.source_domains.is_empty()
            || !self.retired_selections.is_empty()
            || !self.replaced_generations.is_empty()
            || self.create_namespace_uuid.is_some()
            || !self.explicit_map_placements.is_empty()
        {
            return Err(Error::invalid_input(
                "row-aligned rewrite proofs cannot be mixed with source domains, retirement, checkpoint, namespace, or ExplicitMap changes",
            ));
        }
        for placement in &self.placements {
            if placement.placement_kind != RowAddressPlacementKind::Direct
                || !placement.source_selections.is_empty()
                || !matches!(
                    placement.target.fragment,
                    RowAddressTargetFragment::NewFragmentOrdinal(_)
                )
                || !placement.output_row_sequence_fingerprint.is_empty()
            {
                return Err(Error::invalid_input(
                    "row-aligned rewrite proofs may only be mixed with source-free Direct new-fragment provenance",
                ));
            }
        }
        if self.field_changes.len() != self.row_aligned_rewrite_proofs.len() {
            return Err(Error::invalid_input(
                "row-aligned rewrite proofs must reference every field change exactly once",
            ));
        }
        if self
            .source_floors
            .windows(2)
            .any(|pair| pair[0].field_id >= pair[1].field_id)
        {
            return Err(Error::invalid_input(
                "row-aligned rewrite source floors must be strictly sorted by field id",
            ));
        }

        let mut physical_fragments = BTreeSet::new();
        let mut field_changes = BTreeSet::new();
        for proof in &self.row_aligned_rewrite_proofs {
            if proof.physical_rows == 0
                || proof.mapped_offsets_fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                || proof
                    .deletion_offsets_fingerprint
                    .as_ref()
                    .is_some_and(|fingerprint| {
                        fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                    })
            {
                return Err(Error::invalid_input(
                    "row-aligned rewrite proof requires non-zero rows and 16-byte ownership/deletion fingerprints",
                ));
            }
            if !physical_fragments.insert(proof.physical_fragment_id)
                || !field_changes.insert(proof.field_change_index)
            {
                return Err(Error::invalid_input(
                    "row-aligned rewrite proofs contain a duplicate fragment or field-change reference",
                ));
            }
            let change = self
                .field_changes
                .get(proof.field_change_index)
                .ok_or_else(|| {
                    Error::invalid_input(
                        "row-aligned rewrite proof references a missing field change",
                    )
                })?;
            if change.selection.is_empty() {
                return Err(Error::invalid_input(
                    "row-aligned rewrite proof references an empty logical selection",
                ));
            }
            if proof
                .source_floor_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(Error::invalid_input(
                    "row-aligned rewrite proof source-floor indices must be strictly sorted",
                ));
            }
            let floor_fields = proof
                .source_floor_indices
                .iter()
                .map(|index| {
                    self.source_floors
                        .get(*index)
                        .map(|floor| floor.field_id)
                        .ok_or_else(|| {
                            Error::invalid_input(
                                "row-aligned rewrite proof references a missing source floor",
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            if floor_fields != change.field_ids {
                return Err(Error::invalid_input(
                    "row-aligned rewrite proof source floors do not exactly cover its changed fields",
                ));
            }
        }
        Ok(())
    }
}

pub struct RowSequenceFingerprintBuilder {
    hash: u128,
    row_count: u64,
    pending_range: Option<LogicalRowAddressRange>,
}

impl RowSequenceFingerprintBuilder {
    pub fn new(target: RowAddressTargetRange) -> Self {
        const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
        let mut builder = Self {
            hash: OFFSET,
            row_count: 0,
            pending_range: None,
        };
        match target.fragment {
            RowAddressTargetFragment::NewFragmentOrdinal(ordinal) => {
                builder.update_bytes(&[0]);
                builder.update_bytes(&ordinal.to_le_bytes());
            }
            RowAddressTargetFragment::ExistingFragmentId(fragment_id) => {
                builder.update_bytes(&[1]);
                builder.update_bytes(&fragment_id.to_le_bytes());
            }
        }
        builder.update_bytes(&target.start_offset.to_le_bytes());
        builder.update_bytes(&target.end_offset.to_le_bytes());
        builder
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        for byte in bytes {
            self.hash ^= *byte as u128;
            self.hash = self.hash.wrapping_mul(PRIME);
        }
    }

    fn flush_pending_range(&mut self) {
        if let Some(range) = self.pending_range.take() {
            self.update_bytes(&range.logical_fragment_id.to_le_bytes());
            self.update_bytes(&range.start_slot.to_le_bytes());
            self.update_bytes(&range.end_slot.to_le_bytes());
        }
    }

    pub fn update_range(&mut self, range: LogicalRowAddressRange) -> Result<()> {
        range.validate()?;
        let length = range.len();
        if let Some(pending) = &mut self.pending_range
            && pending.logical_fragment_id == range.logical_fragment_id
            && pending.end_slot == range.start_slot
        {
            pending.end_slot = range.end_slot;
        } else {
            self.flush_pending_range();
            self.pending_range = Some(range);
        }
        self.row_count = self
            .row_count
            .checked_add(length)
            .ok_or_else(|| Error::invalid_input("row-sequence cardinality overflow"))?;
        Ok(())
    }

    pub fn update(&mut self, address: LogicalRowAddress) -> Result<()> {
        self.update_range(LogicalRowAddressRange::new(
            address.logical_fragment_id(),
            address.immutable_slot(),
            address
                .immutable_slot()
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("logical row-address slot overflow"))?,
        ))
    }

    pub fn finish(mut self, expected_cardinality: u64) -> Result<Vec<u8>> {
        if self.row_count != expected_cardinality {
            return Err(Error::invalid_input(format!(
                "row-sequence fingerprint saw {} rows, expected {}",
                self.row_count, expected_cardinality
            )));
        }
        self.flush_pending_range();
        Ok(self.hash.to_le_bytes().to_vec())
    }
}

pub fn fingerprint_row_sequence(
    target: RowAddressTargetRange,
    addresses: impl IntoIterator<Item = LogicalRowAddress>,
) -> Result<Vec<u8>> {
    let expected_cardinality = (target.end_offset - target.start_offset) as u64;
    let mut builder = RowSequenceFingerprintBuilder::new(target);
    for address in addresses {
        builder.update(address)?;
    }
    builder.finish(expected_cardinality)
}

impl RowAddressPlacementDelta {
    pub fn expected_row_sequence_fingerprint(&self) -> Result<Vec<u8>> {
        if self.placement_kind == RowAddressPlacementKind::Direct {
            return Ok(Vec::new());
        }
        let mut builder = RowSequenceFingerprintBuilder::new(self.target);
        let mut count = 0_u64;
        for selection in &self.source_selections {
            if !selection.visit_structural_ranges(|range| {
                builder.update_range(range)?;
                count = count
                    .checked_add(range.len())
                    .ok_or_else(|| Error::invalid_input("row-sequence cardinality overflow"))?;
                Ok(())
            })? {
                selection.try_for_each_address(|address| {
                    builder.update(address)?;
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| Error::invalid_input("row-sequence cardinality overflow"))?;
                    Ok(())
                })?;
            }
        }
        if count != self.output_cardinality {
            return Err(Error::invalid_input(
                "placement sources do not match output cardinality",
            ));
        }
        builder.finish(self.output_cardinality)
    }

    pub fn verify_output_row_sequence(&self) -> Result<()> {
        let expected = self.expected_row_sequence_fingerprint()?;
        if expected != self.output_row_sequence_fingerprint {
            return Err(Error::invalid_input(
                "emitted output row sequence does not match placement source order",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum RowReferenceDomain {
    PhysicalRowAddress,
    LegacyStableRowId,
    StableLogicalRowAddress,
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct LogicalIndexCoverageShard {
    pub selection: Option<LogicalRowAddressSelection>,
    pub field_ids: Vec<i32>,
    pub validated_through: Vec<FieldGeneration>,
    pub fingerprint: Vec<u8>,
    pub row_count: u64,
    pub logical_fragment_bitmap: Vec<u8>,
    /// Rows physically present in the immutable index artifact but owned by a
    /// newer segment. `None` means the complete raw selection is effective.
    pub excluded_selection: Option<LogicalRowAddressSelection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalIndexCoverageFile {
    pub path: String,
    pub offset: u64,
    pub byte_length: u64,
    pub global_buffer_index: u32,
    pub object_size: u64,
    pub object_id: Uuid,
    pub artifact_namespace_uuid: Uuid,
    pub artifact_layout_fingerprint: Vec<u8>,
}

impl DeepSizeOf for LogicalIndexCoverageFile {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.path.deep_size_of_children(context)
            + self
                .artifact_layout_fingerprint
                .deep_size_of_children(context)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalIndexCoverageCloneProvenance {
    pub source_namespace_uuid: Uuid,
    pub target_namespace_uuid: Uuid,
    pub source_coverage_fingerprint: Vec<u8>,
    pub transaction_uuid: Uuid,
    pub depth: u32,
    pub is_shallow: bool,
    pub source_manifest_version: u64,
}

impl DeepSizeOf for LogicalIndexCoverageCloneProvenance {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.source_coverage_fingerprint
            .deep_size_of_children(context)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct LogicalIndexCoverage {
    pub shards: Vec<LogicalIndexCoverageShard>,
    pub external: Option<LogicalIndexCoverageFile>,
    pub fingerprint: Vec<u8>,
    pub clone_provenance: Option<LogicalIndexCoverageCloneProvenance>,
    /// Placement-independent logical-domain identity captured only by an
    /// exact, full-current-coverage build. `None` means no manifest-only proof.
    pub full_domain_fingerprint: Option<Box<[u8; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE]>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalRowLocator {
    Physical(RowAddress),
    ExplicitMap {
        placement_index: u32,
        page_index: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementResolution {
    Mapped { locator: PhysicalRowLocator },
    NotLive,
    Unmapped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalToLogicalResolution {
    Logical(LogicalRowAddress),
    ExplicitMap {
        placement_index: u32,
        destination_index: u32,
        destination_row_offset: u32,
    },
    Unmapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalRowRange {
    pub physical_fragment_id: u32,
    pub start_offset: u32,
    pub end_offset: u32,
}

#[derive(Debug, Default)]
struct LogicalRangeOwnershipProof {
    mapped_rows: u64,
    physical_ranges: Vec<PhysicalRowRange>,
}

impl LogicalRangeOwnershipProof {
    fn add_mapped_rows(&mut self, rows: u64) -> Result<()> {
        self.mapped_rows = self
            .mapped_rows
            .checked_add(rows)
            .ok_or_else(|| Error::invalid_input("logical range ownership overflow"))?;
        Ok(())
    }

    fn push_physical_range(
        &mut self,
        physical_fragment_id: u32,
        start_offset: u64,
        end_offset: u64,
    ) -> Result<()> {
        if start_offset >= end_offset {
            return Ok(());
        }
        let start_offset = u32::try_from(start_offset)
            .map_err(|_| Error::invalid_input("physical range start exceeds u32"))?;
        let end_offset = u32::try_from(end_offset)
            .map_err(|_| Error::invalid_input("physical range end exceeds u32"))?;
        if let Some(previous) = self.physical_ranges.last_mut()
            && previous.physical_fragment_id == physical_fragment_id
            && previous.end_offset == start_offset
        {
            previous.end_offset = end_offset;
        } else {
            self.physical_ranges.push(PhysicalRowRange {
                physical_fragment_id,
                start_offset,
                end_offset,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalDomainDestination {
    Inline(PhysicalRowRange),
    ExplicitMap { placement_index: u32 },
}

fn add_full_range_ownership(
    proof: &mut LogicalRangeOwnershipProof,
    logical_range: LogicalRowAddressRange,
    excluded: Option<&LogicalRowAddressSelection>,
    physical_fragment_id: u32,
    destination_start: u64,
) -> Result<bool> {
    let Some(excluded_ranges) = excluded
        .map(|selection| selection.compact_ranges(MAX_SELECTION_DOMAINS))
        .transpose()?
        .flatten()
        .or_else(|| excluded.is_none().then(Vec::new))
    else {
        return Ok(false);
    };
    let mut cursor = logical_range.start_slot;
    for excluded in excluded_ranges
        .into_iter()
        .filter(|range| range.logical_fragment_id == logical_range.logical_fragment_id)
    {
        let excluded_start = excluded.start_slot.max(logical_range.start_slot);
        let excluded_end = excluded.end_slot.min(logical_range.end_slot);
        if excluded_start >= excluded_end {
            continue;
        }
        if cursor < excluded_start {
            proof.push_physical_range(
                physical_fragment_id,
                destination_start + u64::from(cursor),
                destination_start + u64::from(excluded_start),
            )?;
            proof.add_mapped_rows(u64::from(excluded_start - cursor))?;
        }
        cursor = cursor.max(excluded_end);
    }
    if cursor < logical_range.end_slot {
        proof.push_physical_range(
            physical_fragment_id,
            destination_start + u64::from(cursor),
            destination_start + u64::from(logical_range.end_slot),
        )?;
        proof.add_mapped_rows(u64::from(logical_range.end_slot - cursor))?;
    }
    Ok(true)
}

fn add_selected_range_ownership(
    proof: &mut LogicalRangeOwnershipProof,
    logical_range: LogicalRowAddressRange,
    selection: &LogicalRowAddressSelection,
    excluded: Option<&LogicalRowAddressSelection>,
    physical_fragment_id: u32,
    destination_start: u64,
) -> Result<bool> {
    let selected_slots = selection_slots_for_domain(selection, logical_range.logical_fragment_id)?;
    let selected_count =
        selected_slots.range_cardinality(logical_range.start_slot..logical_range.end_slot);
    if selected_count == 0 {
        return Ok(true);
    }
    let selected_before = logical_range
        .start_slot
        .checked_sub(1)
        .map_or(0, |previous| selected_slots.rank(previous));
    let Some(excluded_ranges) = excluded
        .map(|selection| selection.compact_ranges(MAX_SELECTION_DOMAINS))
        .transpose()?
        .flatten()
        .or_else(|| excluded.is_none().then(Vec::new))
    else {
        return Ok(false);
    };
    let physical_start = destination_start
        .checked_add(selected_before)
        .ok_or_else(|| Error::invalid_input("selected destination start overflow"))?;
    let physical_end = physical_start
        .checked_add(selected_count)
        .ok_or_else(|| Error::invalid_input("selected destination end overflow"))?;
    let mut cursor = physical_start;
    for excluded in excluded_ranges
        .into_iter()
        .filter(|range| range.logical_fragment_id == logical_range.logical_fragment_id)
    {
        let excluded_start = excluded.start_slot.max(logical_range.start_slot);
        let excluded_end = excluded.end_slot.min(logical_range.end_slot);
        if excluded_start >= excluded_end {
            continue;
        }
        let excluded_before = excluded_start
            .checked_sub(1)
            .map_or(0, |previous| selected_slots.rank(previous));
        let excluded_count = selected_slots.range_cardinality(excluded_start..excluded_end);
        if excluded_count == 0 {
            continue;
        }
        let excluded_physical_start = destination_start
            .checked_add(excluded_before)
            .ok_or_else(|| Error::invalid_input("selected exclusion start overflow"))?;
        let excluded_physical_end = excluded_physical_start
            .checked_add(excluded_count)
            .ok_or_else(|| Error::invalid_input("selected exclusion end overflow"))?;
        if cursor < excluded_physical_start {
            proof.push_physical_range(physical_fragment_id, cursor, excluded_physical_start)?;
            proof.add_mapped_rows(excluded_physical_start - cursor)?;
        }
        cursor = cursor.max(excluded_physical_end);
    }
    if cursor < physical_end {
        proof.push_physical_range(physical_fragment_id, cursor, physical_end)?;
        proof.add_mapped_rows(physical_end - cursor)?;
    }
    Ok(true)
}

fn compact_excluded_rank_ranges(
    selection: &LogicalRowAddressSelection,
    excluded: &LogicalRowAddressSelection,
    logical_fragment_id: u32,
    destination_prefix: u64,
) -> Result<Option<Vec<(u64, u64)>>> {
    let Some(excluded_ranges) = excluded.compact_ranges(MAX_RANGE_ENCODING_RUNS)? else {
        return Ok(None);
    };
    let selected_slots = selection_slots_for_domain(selection, logical_fragment_id)?;
    let mut rank_ranges = Vec::<(u64, u64)>::with_capacity(excluded_ranges.len());
    for excluded in excluded_ranges {
        if excluded.logical_fragment_id != logical_fragment_id {
            return Err(Error::invalid_input(
                "placement exclusion references a different logical domain",
            ));
        }
        let selected_before = excluded
            .start_slot
            .checked_sub(1)
            .map_or(0, |previous| selected_slots.rank(previous));
        let selected_count =
            selected_slots.range_cardinality(excluded.start_slot..excluded.end_slot);
        if selected_count == 0 {
            continue;
        }
        let start = destination_prefix
            .checked_add(selected_before)
            .ok_or_else(|| Error::invalid_input("placement exclusion rank overflow"))?;
        let end = start
            .checked_add(selected_count)
            .ok_or_else(|| Error::invalid_input("placement exclusion end overflow"))?;
        if let Some((_, previous_end)) = rank_ranges.last_mut()
            && *previous_end == start
        {
            *previous_end = end;
        } else {
            rank_ranges.push((start, end));
        }
    }
    Ok(Some(rank_ranges))
}

#[derive(Debug, Clone)]
pub struct RowAddressRouter {
    layout: Arc<RowAddressLayout>,
    source_index: Vec<(u32, u32)>,
    packed_source_runs: Vec<(u32, u32, u32)>,
    native_domains: Vec<(u32, u32, u32, u64)>,
    native_by_physical: Vec<u32>,
}

impl RowAddressRouter {
    pub fn try_new(
        layout: Arc<RowAddressLayout>,
        fragments: &[Fragment],
        max_logical_fragment_id: Option<u32>,
    ) -> Result<Self> {
        layout.validate_with_fragments(fragments, max_logical_fragment_id)?;
        Self::from_validated_layout(layout, fragments)
    }

    fn from_validated_layout(
        layout: Arc<RowAddressLayout>,
        fragments: &[Fragment],
    ) -> Result<Self> {
        let mut source_index = Vec::<(u32, u32)>::new();
        let mut packed_source_runs = Vec::new();
        for (placement_index, placement) in layout.placements.iter().enumerate() {
            if let RowAddressPlacement::PackedRun(value) = placement
                && matches!(
                    value.domains.logical_fragment_ids.encoding.as_ref(),
                    Some(pb::packed_logical_fragment_ids::Encoding::Consecutive(_))
                )
            {
                packed_source_runs.push((
                    value.domains.first_logical_fragment_id(),
                    value.domains.last_logical_fragment_id()?,
                    placement_index as u32,
                ));
            } else {
                placement.for_each_source(|source| {
                    source_index.push((source.logical_fragment_id, placement_index as u32));
                    Ok(())
                })?;
            }
        }
        source_index.sort_unstable();
        packed_source_runs.sort_unstable();
        if packed_source_runs
            .windows(2)
            .any(|pair| pair[0].1 >= pair[1].0)
        {
            return Err(Error::invalid_input(
                "consecutive PackedRun logical-id ranges must not overlap",
            ));
        }
        let native_domains = native_domain_routes(fragments)?;
        let mut native_by_physical = (0..native_domains.len() as u32).collect::<Vec<_>>();
        native_by_physical.sort_unstable_by_key(|index| native_domains[*index as usize].1);
        Ok(Self {
            layout,
            source_index,
            packed_source_runs,
            native_domains,
            native_by_physical,
        })
    }

    pub fn namespace_uuid(&self) -> Uuid {
        self.layout.namespace_uuid
    }

    pub fn fingerprint(&self) -> &[u8] {
        &self.layout.fingerprint
    }

    pub fn logical_domain_destination_ranges(
        &self,
        logical_fragment_id: u32,
    ) -> Result<Vec<LogicalDomainDestination>> {
        let mut placement_indices = Vec::new();
        let source_start = self
            .source_index
            .partition_point(|(logical, _)| *logical < logical_fragment_id);
        let source_end = self
            .source_index
            .partition_point(|(logical, _)| *logical <= logical_fragment_id);
        placement_indices.extend(
            self.source_index[source_start..source_end]
                .iter()
                .map(|(_, placement_index)| *placement_index),
        );
        let packed_index = self
            .packed_source_runs
            .partition_point(|(first, _, _)| *first <= logical_fragment_id);
        if let Some((_, last, placement_index)) = packed_index
            .checked_sub(1)
            .and_then(|index| self.packed_source_runs.get(index))
            && logical_fragment_id <= *last
            && self.layout.placements[*placement_index as usize]
                .source_domain(logical_fragment_id)?
                .is_some()
        {
            placement_indices.push(*placement_index);
        }
        let mut destinations = Vec::new();
        for placement_index in placement_indices.iter().copied() {
            let placement = &self.layout.placements[placement_index as usize];
            match placement {
                RowAddressPlacement::Direct(value) => {
                    destinations.push(LogicalDomainDestination::Inline(PhysicalRowRange {
                        physical_fragment_id: value.destination_fragment_id,
                        start_offset: value.destination_start,
                        end_offset: value.destination_start + value.source.slot_count,
                    }));
                }
                RowAddressPlacement::PackedRun(value) => {
                    let ordinal = value
                        .domains
                        .domain_ordinal(logical_fragment_id)?
                        .ok_or_else(|| Error::invalid_input("PackedRun source index mismatch"))?;
                    let start_offset = value
                        .destination_start
                        .checked_add(
                            u32::try_from(value.domains.slot_prefix(ordinal)?).map_err(|_| {
                                Error::invalid_input("PackedRun prefix exceeds u32")
                            })?,
                        )
                        .ok_or_else(|| Error::invalid_input("PackedRun destination overflow"))?;
                    destinations.push(LogicalDomainDestination::Inline(PhysicalRowRange {
                        physical_fragment_id: value.destination_fragment_id,
                        start_offset,
                        end_offset: start_offset + value.domains.slot_count_at(ordinal)?,
                    }));
                }
                RowAddressPlacement::Selected(value) => {
                    let end_offset = value
                        .destination_start
                        .checked_add(u32::try_from(value.selection.cardinality()).map_err(
                            |_| Error::invalid_input("Selected cardinality exceeds u32"),
                        )?)
                        .ok_or_else(|| Error::invalid_input("Selected destination overflow"))?;
                    destinations.push(LogicalDomainDestination::Inline(PhysicalRowRange {
                        physical_fragment_id: value.destination_fragment_id,
                        start_offset: value.destination_start,
                        end_offset,
                    }));
                }
                RowAddressPlacement::ExtentList(value) => {
                    destinations.extend(value.extents.iter().map(|extent| {
                        LogicalDomainDestination::Inline(PhysicalRowRange {
                            physical_fragment_id: extent.destination_fragment_id,
                            start_offset: extent.destination_start,
                            end_offset: extent.destination_start + extent.length,
                        })
                    }));
                }
                RowAddressPlacement::SparseSelection(value) => {
                    let mut prefix = 0_u64;
                    for source in &value.sources {
                        if source.source.logical_fragment_id == logical_fragment_id {
                            let start_offset = value
                                .destination_start
                                .checked_add(u32::try_from(prefix).map_err(|_| {
                                    Error::invalid_input("SparseSelection prefix exceeds u32")
                                })?)
                                .ok_or_else(|| {
                                    Error::invalid_input("SparseSelection destination overflow")
                                })?;
                            let end_offset = start_offset
                                .checked_add(
                                    u32::try_from(source.selection.cardinality()).map_err(
                                        |_| {
                                            Error::invalid_input(
                                                "SparseSelection cardinality exceeds u32",
                                            )
                                        },
                                    )?,
                                )
                                .ok_or_else(|| {
                                    Error::invalid_input("SparseSelection destination overflow")
                                })?;
                            destinations.push(LogicalDomainDestination::Inline(PhysicalRowRange {
                                physical_fragment_id: value.destination_fragment_id,
                                start_offset,
                                end_offset,
                            }));
                            break;
                        }
                        prefix = prefix
                            .checked_add(source.selection.cardinality())
                            .ok_or_else(|| {
                                Error::invalid_input("SparseSelection prefix overflow")
                            })?;
                    }
                }
                RowAddressPlacement::ExplicitMap(_) => {
                    destinations.push(LogicalDomainDestination::ExplicitMap { placement_index });
                }
            }
        }
        if destinations.is_empty()
            && let Ok(index) = self
                .native_domains
                .binary_search_by_key(&logical_fragment_id, |(logical, _, _, _)| *logical)
        {
            let (_, physical_fragment_id, slot_count, _) = self.native_domains[index];
            destinations.push(LogicalDomainDestination::Inline(PhysicalRowRange {
                physical_fragment_id,
                start_offset: 0,
                end_offset: slot_count,
            }));
        }
        Ok(destinations)
    }

    /// Return the physical fragments that may own the selected slots without
    /// resolving the slots one by one.
    pub fn logical_selection_destination_fragments(
        &self,
        logical_fragment_id: u32,
        slots: &RoaringBitmap,
    ) -> Result<RoaringBitmap> {
        fn domain_slots(
            selection: &LogicalRowAddressSelection,
            logical_fragment_id: u32,
        ) -> Result<RoaringBitmap> {
            Ok(selection
                .to_roaring_treemap()?
                .bitmaps()
                .find_map(|(domain, slots)| (domain == logical_fragment_id).then(|| slots.clone()))
                .unwrap_or_default())
        }

        fn selected_slots(
            selection: &LogicalRowAddressSelection,
            excluded: Option<&LogicalRowAddressSelection>,
            logical_fragment_id: u32,
        ) -> Result<RoaringBitmap> {
            let mut selected = domain_slots(selection, logical_fragment_id)?;
            if let Some(excluded) = excluded {
                selected -= domain_slots(excluded, logical_fragment_id)?;
            }
            Ok(selected)
        }

        if slots.is_empty() {
            return Ok(RoaringBitmap::new());
        }
        let source_start = self
            .source_index
            .partition_point(|(logical, _)| *logical < logical_fragment_id);
        let source_end = self
            .source_index
            .partition_point(|(logical, _)| *logical <= logical_fragment_id);
        let mut placement_indices = self.source_index[source_start..source_end]
            .iter()
            .map(|(_, placement_index)| *placement_index)
            .collect::<Vec<_>>();
        let packed_index = self
            .packed_source_runs
            .partition_point(|(first, _, _)| *first <= logical_fragment_id);
        if let Some((_, last, placement_index)) = packed_index
            .checked_sub(1)
            .and_then(|index| self.packed_source_runs.get(index))
            && logical_fragment_id <= *last
            && self.layout.placements[*placement_index as usize]
                .source_domain(logical_fragment_id)?
                .is_some()
        {
            placement_indices.push(*placement_index);
        }

        let mut fragments = RoaringBitmap::new();
        for placement_index in placement_indices {
            match &self.layout.placements[placement_index as usize] {
                RowAddressPlacement::Direct(value) => {
                    let mut owned = slots.clone();
                    if let Some(excluded) = value.excluded.as_deref() {
                        owned -= domain_slots(excluded, logical_fragment_id)?;
                    }
                    if !owned.is_empty() {
                        fragments.insert(value.destination_fragment_id);
                    }
                }
                RowAddressPlacement::PackedRun(value) => {
                    fragments.insert(value.destination_fragment_id);
                }
                RowAddressPlacement::Selected(value) => {
                    let owned = selected_slots(
                        value.selection.as_ref(),
                        value.excluded.as_deref(),
                        logical_fragment_id,
                    )?;
                    if !owned.is_disjoint(slots) {
                        fragments.insert(value.destination_fragment_id);
                    }
                }
                RowAddressPlacement::ExtentList(value) => {
                    for extent in &value.extents {
                        let end = extent
                            .source_start
                            .checked_add(extent.length)
                            .ok_or_else(|| Error::invalid_input("extent source range overflow"))?;
                        if slots.range_cardinality(extent.source_start..end) != 0 {
                            fragments.insert(extent.destination_fragment_id);
                        }
                    }
                }
                RowAddressPlacement::SparseSelection(value) => {
                    for source in value
                        .sources
                        .iter()
                        .filter(|source| source.source.logical_fragment_id == logical_fragment_id)
                    {
                        let owned = selected_slots(
                            source.selection.as_ref(),
                            source.excluded.as_deref(),
                            logical_fragment_id,
                        )?;
                        if !owned.is_disjoint(slots) {
                            fragments.insert(value.destination_fragment_id);
                        }
                    }
                }
                RowAddressPlacement::ExplicitMap(value) => {
                    let intersects = value
                        .sources
                        .iter()
                        .filter(|source| source.source.logical_fragment_id == logical_fragment_id)
                        .map(|source| {
                            selected_slots(
                                source.selection.as_ref(),
                                source.excluded.as_deref(),
                                logical_fragment_id,
                            )
                            .map(|owned| !owned.is_disjoint(slots))
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .any(|intersects| intersects);
                    if intersects {
                        fragments.extend(
                            value
                                .destinations
                                .iter()
                                .map(|destination| destination.physical_fragment_id),
                        );
                    }
                }
            }
        }
        if let Ok(index) = self
            .native_domains
            .binary_search_by_key(&logical_fragment_id, |(logical, _, _, _)| *logical)
        {
            fragments.insert(self.native_domains[index].1);
        }
        Ok(fragments)
    }

    /// Prove exact inline ownership for an arbitrary logical slot bitmap.
    ///
    /// The proof applies each placement selection once, so high-entropy update
    /// sets do not multiply the placement payload by their number of runs.
    /// `None` means an ExplicitMap locator must perform the bounded row-level
    /// fallback because its membership is not authoritative in the root.
    fn logical_selection_inline_ownership(
        &self,
        logical_fragment_id: u32,
        slots: &RoaringBitmap,
    ) -> Result<Option<RoaringBitmap>> {
        if slots.is_empty() {
            return Ok(Some(RoaringBitmap::new()));
        }

        let source_start = self
            .source_index
            .partition_point(|(logical, _)| *logical < logical_fragment_id);
        let source_end = self
            .source_index
            .partition_point(|(logical, _)| *logical <= logical_fragment_id);
        let mut placement_indices = self.source_index[source_start..source_end]
            .iter()
            .map(|(_, placement_index)| *placement_index)
            .collect::<Vec<_>>();
        let packed_index = self
            .packed_source_runs
            .partition_point(|(first, _, _)| *first <= logical_fragment_id);
        if let Some((_, last, placement_index)) = packed_index
            .checked_sub(1)
            .and_then(|index| self.packed_source_runs.get(index))
            && logical_fragment_id <= *last
            && self.layout.placements[*placement_index as usize]
                .source_domain(logical_fragment_id)?
                .is_some()
        {
            placement_indices.push(*placement_index);
        }

        let mut covered = RoaringBitmap::new();
        let mut fragments = RoaringBitmap::new();
        let mut record_owned = |mut owned: RoaringBitmap,
                                physical_fragment_id: u32|
         -> Result<()> {
            owned &= slots;
            if owned.is_empty() {
                return Ok(());
            }
            if !covered.is_disjoint(&owned) {
                return Err(Error::invalid_input(format!(
                    "logical selection in fragment {logical_fragment_id} has overlapping physical owners"
                )));
            }
            covered |= &owned;
            fragments.insert(physical_fragment_id);
            Ok(())
        };
        let slots_in_range = |start: u32, end: u32| {
            let mut bounded = RoaringBitmap::new();
            bounded.insert_range(start..end);
            bounded &= slots;
            bounded
        };

        for placement_index in placement_indices {
            match &self.layout.placements[placement_index as usize] {
                RowAddressPlacement::Direct(value) => {
                    let mut owned = slots_in_range(0, value.source.slot_count);
                    if let Some(excluded) = value.excluded.as_deref() {
                        owned -= selection_slots_for_domain(excluded, logical_fragment_id)?;
                    }
                    record_owned(owned, value.destination_fragment_id)?;
                }
                RowAddressPlacement::PackedRun(value) => {
                    let Some(ordinal) = value.domains.domain_ordinal(logical_fragment_id)? else {
                        continue;
                    };
                    record_owned(
                        slots_in_range(0, value.domains.slot_count_at(ordinal)?),
                        value.destination_fragment_id,
                    )?;
                }
                RowAddressPlacement::Selected(value) => {
                    let mut owned =
                        selection_slots_for_domain(&value.selection, logical_fragment_id)?;
                    if let Some(excluded) = value.excluded.as_deref() {
                        owned -= selection_slots_for_domain(excluded, logical_fragment_id)?;
                    }
                    record_owned(owned, value.destination_fragment_id)?;
                }
                RowAddressPlacement::ExtentList(value) => {
                    for extent in &value.extents {
                        let end =
                            extent
                                .source_start
                                .checked_add(extent.length)
                                .ok_or_else(|| {
                                    Error::invalid_input("ExtentList source range overflow")
                                })?;
                        record_owned(
                            slots_in_range(extent.source_start, end),
                            extent.destination_fragment_id,
                        )?;
                    }
                }
                RowAddressPlacement::SparseSelection(value) => {
                    for source in value
                        .sources
                        .iter()
                        .filter(|source| source.source.logical_fragment_id == logical_fragment_id)
                    {
                        let mut owned =
                            selection_slots_for_domain(&source.selection, logical_fragment_id)?;
                        if let Some(excluded) = source.excluded.as_deref() {
                            owned -= selection_slots_for_domain(excluded, logical_fragment_id)?;
                        }
                        record_owned(owned, value.destination_fragment_id)?;
                    }
                }
                RowAddressPlacement::ExplicitMap(value) => {
                    if value
                        .sources
                        .iter()
                        .any(|source| source.source.logical_fragment_id == logical_fragment_id)
                    {
                        return Ok(None);
                    }
                }
            }
        }
        if let Ok(index) = self
            .native_domains
            .binary_search_by_key(&logical_fragment_id, |(logical, _, _, _)| *logical)
        {
            let (_, physical_fragment_id, slot_count, _) = self.native_domains[index];
            record_owned(slots_in_range(0, slot_count), physical_fragment_id)?;
        }
        if &covered != slots {
            let mut missing = slots.clone();
            missing -= &covered;
            return Err(Error::invalid_input(format!(
                "touched logical selection contains unmapped address {}:{}",
                logical_fragment_id,
                missing.min().unwrap_or_default(),
            )));
        }
        Ok(Some(fragments))
    }

    /// Prove the exact physical ownership of a contiguous logical range from
    /// root metadata. `None` means an external or high-entropy exclusion needs
    /// the bounded row resolver fallback.
    fn logical_range_ownership(
        &self,
        range: LogicalRowAddressRange,
    ) -> Result<Option<LogicalRangeOwnershipProof>> {
        range.validate()?;
        let logical_fragment_id = range.logical_fragment_id;
        let source_start = self
            .source_index
            .partition_point(|(logical, _)| *logical < logical_fragment_id);
        let source_end = self
            .source_index
            .partition_point(|(logical, _)| *logical <= logical_fragment_id);
        let mut placement_indices = self.source_index[source_start..source_end]
            .iter()
            .map(|(_, placement_index)| *placement_index)
            .collect::<Vec<_>>();
        let packed_index = self
            .packed_source_runs
            .partition_point(|(first, _, _)| *first <= logical_fragment_id);
        if let Some((_, last, placement_index)) = packed_index
            .checked_sub(1)
            .and_then(|index| self.packed_source_runs.get(index))
            && logical_fragment_id <= *last
            && self.layout.placements[*placement_index as usize]
                .source_domain(logical_fragment_id)?
                .is_some()
        {
            placement_indices.push(*placement_index);
        }

        let mut proof = LogicalRangeOwnershipProof::default();
        for placement_index in placement_indices {
            match &self.layout.placements[placement_index as usize] {
                RowAddressPlacement::Direct(value) => {
                    let start = range.start_slot.min(value.source.slot_count);
                    let end = range.end_slot.min(value.source.slot_count);
                    if start < end
                        && !add_full_range_ownership(
                            &mut proof,
                            LogicalRowAddressRange::new(logical_fragment_id, start, end),
                            value.excluded.as_deref(),
                            value.destination_fragment_id,
                            u64::from(value.destination_start),
                        )?
                    {
                        return Ok(None);
                    }
                }
                RowAddressPlacement::PackedRun(value) => {
                    let Some(ordinal) = value.domains.domain_ordinal(logical_fragment_id)? else {
                        continue;
                    };
                    let slot_count = value.domains.slot_count_at(ordinal)?;
                    let start = range.start_slot.min(slot_count);
                    let end = range.end_slot.min(slot_count);
                    if start < end {
                        let destination_start = u64::from(value.destination_start)
                            .checked_add(value.domains.slot_prefix(ordinal)?)
                            .ok_or_else(|| {
                                Error::invalid_input("PackedRun destination prefix overflow")
                            })?;
                        proof.push_physical_range(
                            value.destination_fragment_id,
                            destination_start + u64::from(start),
                            destination_start + u64::from(end),
                        )?;
                        proof.add_mapped_rows(u64::from(end - start))?;
                    }
                }
                RowAddressPlacement::Selected(value) => {
                    let start = range.start_slot.min(value.source.slot_count);
                    let end = range.end_slot.min(value.source.slot_count);
                    if start < end
                        && !add_selected_range_ownership(
                            &mut proof,
                            LogicalRowAddressRange::new(logical_fragment_id, start, end),
                            &value.selection,
                            value.excluded.as_deref(),
                            value.destination_fragment_id,
                            u64::from(value.destination_start),
                        )?
                    {
                        return Ok(None);
                    }
                }
                RowAddressPlacement::ExtentList(value) => {
                    for extent in &value.extents {
                        let source_end = extent
                            .source_start
                            .checked_add(extent.length)
                            .ok_or_else(|| {
                                Error::invalid_input("ExtentList source range overflow")
                            })?;
                        let start = range.start_slot.max(extent.source_start);
                        let end = range.end_slot.min(source_end);
                        if start >= end {
                            continue;
                        }
                        let destination_start = u64::from(extent.destination_start)
                            + u64::from(start - extent.source_start);
                        proof.push_physical_range(
                            extent.destination_fragment_id,
                            destination_start,
                            destination_start + u64::from(end - start),
                        )?;
                        proof.add_mapped_rows(u64::from(end - start))?;
                    }
                }
                RowAddressPlacement::SparseSelection(value) => {
                    let mut prefix = 0_u64;
                    for source in &value.sources {
                        if source.source.logical_fragment_id == logical_fragment_id {
                            let start = range.start_slot.min(source.source.slot_count);
                            let end = range.end_slot.min(source.source.slot_count);
                            if start < end
                                && !add_selected_range_ownership(
                                    &mut proof,
                                    LogicalRowAddressRange::new(logical_fragment_id, start, end),
                                    &source.selection,
                                    source.excluded.as_deref(),
                                    value.destination_fragment_id,
                                    u64::from(value.destination_start) + prefix,
                                )?
                            {
                                return Ok(None);
                            }
                        }
                        prefix = prefix
                            .checked_add(source.selection.cardinality())
                            .ok_or_else(|| {
                                Error::invalid_input("SparseSelection destination prefix overflow")
                            })?;
                    }
                }
                RowAddressPlacement::ExplicitMap(value) => {
                    if value
                        .sources
                        .iter()
                        .any(|source| source.source.logical_fragment_id == logical_fragment_id)
                    {
                        // The locator object is authoritative for membership;
                        // root page bounds cannot prove every row in a range.
                        return Ok(None);
                    }
                }
            }
        }
        if let Ok(index) = self
            .native_domains
            .binary_search_by_key(&logical_fragment_id, |(logical, _, _, _)| *logical)
        {
            let (_, physical_fragment_id, slot_count, _) = self.native_domains[index];
            let start = range.start_slot.min(slot_count);
            let end = range.end_slot.min(slot_count);
            if start < end {
                proof.push_physical_range(
                    physical_fragment_id,
                    u64::from(start),
                    u64::from(end),
                )?;
                proof.add_mapped_rows(u64::from(end - start))?;
            }
        }
        if proof.mapped_rows > range.len() {
            return Err(Error::invalid_input(format!(
                "logical range fragment_id={}, start={}, end={} has overlapping physical owners",
                range.logical_fragment_id, range.start_slot, range.end_slot,
            )));
        }
        Ok(Some(proof))
    }

    pub fn logical_domain(
        &self,
        logical_fragment_id: u32,
    ) -> Result<Option<RowAddressLogicalDomain>> {
        let mut domain = None;
        let source_start = self
            .source_index
            .partition_point(|(logical, _)| *logical < logical_fragment_id);
        let source_end = self
            .source_index
            .partition_point(|(logical, _)| *logical <= logical_fragment_id);
        for (_, placement_index) in &self.source_index[source_start..source_end] {
            if let Some(candidate) = self.layout.placements[*placement_index as usize]
                .source_domain(logical_fragment_id)?
            {
                if domain.is_some_and(|current| current != candidate) {
                    return Err(Error::invalid_input(
                        "logical domain has inconsistent placement metadata",
                    ));
                }
                domain = Some(candidate);
            }
        }
        let packed_index = self
            .packed_source_runs
            .partition_point(|(first, _, _)| *first <= logical_fragment_id);
        if let Some((_, last, placement_index)) = packed_index
            .checked_sub(1)
            .and_then(|index| self.packed_source_runs.get(index))
            && logical_fragment_id <= *last
            && let Some(candidate) = self.layout.placements[*placement_index as usize]
                .source_domain(logical_fragment_id)?
        {
            if domain.is_some_and(|current| current != candidate) {
                return Err(Error::invalid_input(
                    "logical domain has inconsistent PackedRun metadata",
                ));
            }
            domain = Some(candidate);
        }
        for retired in &self.layout.retired_rows {
            if let Some(candidate) = retired.source_domain(logical_fragment_id)? {
                if domain.is_some_and(|current| current != candidate) {
                    return Err(Error::invalid_input(
                        "logical domain has inconsistent retired metadata",
                    ));
                }
                domain = Some(candidate);
            }
        }
        if let Ok(index) = self
            .native_domains
            .binary_search_by_key(&logical_fragment_id, |(logical, _, _, _)| *logical)
        {
            let (logical_fragment_id, _, slot_count, creation_version) = self.native_domains[index];
            let candidate =
                RowAddressLogicalDomain::new(logical_fragment_id, slot_count, creation_version)?;
            if domain.is_some_and(|current| current != candidate) {
                return Err(Error::invalid_input(
                    "logical domain native and placement metadata disagree",
                ));
            }
            domain = Some(candidate);
        }
        Ok(domain)
    }

    pub fn resolve_many(
        &self,
        addresses: &[LogicalRowAddress],
    ) -> Result<Vec<PlacementResolution>> {
        addresses
            .iter()
            .copied()
            .map(|address| {
                // Retirement is a liveness override.  In particular, an
                // ExplicitMap remains an immutable lookup root after deletes;
                // retired identities must not fall through to its locator.
                if self.layout.is_retired(address)? {
                    return Ok(PlacementResolution::NotLive);
                }
                let mut matched = None;
                let source_start = self
                    .source_index
                    .partition_point(|(logical, _)| *logical < address.logical_fragment_id());
                let source_end = self
                    .source_index
                    .partition_point(|(logical, _)| *logical <= address.logical_fragment_id());
                for (_, placement_index) in &self.source_index[source_start..source_end] {
                    if let Some(locator) = self.layout.placements[*placement_index as usize]
                        .resolve(address, *placement_index)?
                        && matched.replace(locator).is_some()
                    {
                        return Err(Error::invalid_input(format!(
                            "logical address {} is owned by multiple placements",
                            address.raw()
                        )));
                    }
                }
                let packed_index = self
                    .packed_source_runs
                    .partition_point(|(first, _, _)| *first <= address.logical_fragment_id());
                if let Some((_, last, placement_index)) = packed_index
                    .checked_sub(1)
                    .and_then(|index| self.packed_source_runs.get(index))
                    && address.logical_fragment_id() <= *last
                    && let Some(locator) = self.layout.placements[*placement_index as usize]
                        .resolve(address, *placement_index)?
                    && matched.replace(locator).is_some()
                {
                    return Err(Error::invalid_input(format!(
                        "logical address {} is owned by multiple placements",
                        address.raw()
                    )));
                }
                if let Some(locator) = matched {
                    return Ok(PlacementResolution::Mapped { locator });
                }
                if let Ok(index) = self
                    .native_domains
                    .binary_search_by_key(&address.logical_fragment_id(), |(logical, _, _, _)| {
                        *logical
                    })
                    && address.immutable_slot() < self.native_domains[index].2
                {
                    let (_, fragment_id, _, _) = self.native_domains[index];
                    return Ok(PlacementResolution::Mapped {
                        locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(
                            fragment_id,
                            address.immutable_slot(),
                        )),
                    });
                }
                Ok(PlacementResolution::Unmapped)
            })
            .collect()
    }

    pub fn physical_to_logical_many(
        &self,
        addresses: &[RowAddress],
    ) -> Result<Vec<PhysicalToLogicalResolution>> {
        addresses
            .iter()
            .map(|address| {
                if let Ok(index) = self
                    .native_by_physical
                    .binary_search_by_key(&address.fragment_id(), |index| {
                        self.native_domains[*index as usize].1
                    })
                {
                    let (logical_fragment_id, _, slot_count, _) =
                        self.native_domains[self.native_by_physical[index] as usize];
                    if address.row_offset() < slot_count {
                        return LogicalRowAddress::try_new_from_parts(
                            logical_fragment_id,
                            address.row_offset(),
                        )
                        .map(PhysicalToLogicalResolution::Logical);
                    }
                }
                let Some(destination_entry) = self
                    .layout
                    .destination_index
                    .binary_search_by_key(&address.fragment_id(), |entry| {
                        entry.physical_fragment_id
                    })
                    .ok()
                    .map(|index| &self.layout.destination_index[index])
                else {
                    return Ok(PhysicalToLogicalResolution::Unmapped);
                };
                let mut matched = None;
                for placement_index in &destination_entry.placement_indices {
                    let placement = &self.layout.placements[*placement_index as usize];
                    if let Some(logical) = placement.inverse(*address)?
                        && matched
                            .replace(PhysicalToLogicalResolution::Logical(logical))
                            .is_some()
                    {
                        return Err(Error::invalid_input(format!(
                            "physical address {} is owned by multiple placements",
                            u64::from(*address)
                        )));
                    }
                    if let RowAddressPlacement::ExplicitMap(explicit) = placement {
                        for (destination_index, destination) in
                            explicit.destinations.iter().enumerate()
                        {
                            let end = destination.destination_start + destination.row_count;
                            if destination.physical_fragment_id == address.fragment_id()
                                && destination.destination_start <= address.row_offset()
                                && address.row_offset() < end
                            {
                                let resolution = PhysicalToLogicalResolution::ExplicitMap {
                                    placement_index: *placement_index,
                                    destination_index: destination_index as u32,
                                    destination_row_offset: address.row_offset()
                                        - destination.destination_start,
                                };
                                if matched.replace(resolution).is_some() {
                                    return Err(Error::invalid_input(format!(
                                        "physical address {} is owned by multiple placements",
                                        u64::from(*address)
                                    )));
                                }
                            }
                        }
                    }
                }
                Ok(matched.unwrap_or(PhysicalToLogicalResolution::Unmapped))
            })
            .collect()
    }
}

fn physical_fragment_id(value: u64, context: &str) -> Result<u32> {
    let id = u32::try_from(value).map_err(|_| {
        Error::invalid_input(format!(
            "{context} physical fragment id does not fit in u32: {value}"
        ))
    })?;
    if id == INVALID_ID {
        return Err(Error::invalid_input(format!(
            "{context} uses the reserved physical fragment id"
        )));
    }
    Ok(id)
}

fn required<T>(value: Option<T>, field: &str) -> Result<T> {
    value.ok_or_else(|| Error::invalid_input(format!("required field is missing: {field}")))
}

fn pooled_selection(
    selection_pool: &[Arc<LogicalRowAddressSelection>],
    index: Option<u32>,
    field: &str,
) -> Result<Arc<LogicalRowAddressSelection>> {
    let index = required(index, field)? as usize;
    selection_pool.get(index).cloned().ok_or_else(|| {
        Error::invalid_input(format!(
            "{field} index {index} is outside selection pool of length {}",
            selection_pool.len()
        ))
    })
}

fn optional_pooled_selection(
    selection_pool: &[Arc<LogicalRowAddressSelection>],
    index: Option<u32>,
    field: &str,
) -> Result<Option<Arc<LogicalRowAddressSelection>>> {
    index
        .map(|index| pooled_selection(selection_pool, Some(index), field))
        .transpose()
}

fn placement_from_proto(
    value: pb::RowAddressPlacement,
    selection_pool: &[Arc<LogicalRowAddressSelection>],
) -> Result<RowAddressPlacement> {
    use pb::row_address_placement::Codec;
    let placement = match required(value.codec, "RowAddressPlacement.codec")? {
        Codec::Direct(value) => RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: required(value.source, "DirectRowAddressPlacement.source")?.try_into()?,
            destination_fragment_id: physical_fragment_id(
                value.destination_fragment_id,
                "DirectRowAddressPlacement",
            )?,
            destination_start: value.destination_start,
            excluded: optional_pooled_selection(
                selection_pool,
                value.excluded_selection_index,
                "DirectRowAddressPlacement.excluded_selection_index",
            )?,
        }),
        Codec::PackedRun(value) => {
            let domains = PackedLogicalDomainRun::try_new(
                value.first_logical_fragment_id,
                value.domain_count,
                required(
                    value.logical_fragment_ids,
                    "PackedRunRowAddressPlacement.logical_fragment_ids",
                )?,
                required(
                    value.slot_counts,
                    "PackedRunRowAddressPlacement.slot_counts",
                )?,
                required(
                    value.creation_versions,
                    "PackedRunRowAddressPlacement.creation_versions",
                )?,
            )?;
            RowAddressPlacement::PackedRun(PackedRunRowAddressPlacement {
                domains,
                destination_fragment_id: physical_fragment_id(
                    value.destination_fragment_id,
                    "PackedRunRowAddressPlacement",
                )?,
                destination_start: value.destination_start,
            })
        }
        Codec::Selected(value) => RowAddressPlacement::Selected(SelectedRowAddressPlacement {
            source: required(value.source, "SelectedRowAddressPlacement.source")?.try_into()?,
            selection: pooled_selection(
                selection_pool,
                value.selection_index,
                "SelectedRowAddressPlacement.selection_index",
            )?,
            destination_fragment_id: physical_fragment_id(
                value.destination_fragment_id,
                "SelectedRowAddressPlacement",
            )?,
            destination_start: value.destination_start,
            excluded: optional_pooled_selection(
                selection_pool,
                value.excluded_selection_index,
                "SelectedRowAddressPlacement.excluded_selection_index",
            )?,
        }),
        Codec::ExtentList(value) => {
            RowAddressPlacement::ExtentList(ExtentListRowAddressPlacement {
                source: required(value.source, "ExtentListRowAddressPlacement.source")?
                    .try_into()?,
                extents: value
                    .extents
                    .into_iter()
                    .map(|extent| {
                        Ok(RowAddressExtent {
                            source_start: extent.source_start,
                            length: extent.length,
                            destination_fragment_id: physical_fragment_id(
                                extent.destination_fragment_id,
                                "RowAddressExtent",
                            )?,
                            destination_start: extent.destination_start,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            })
        }
        Codec::SparseSelection(value) => {
            RowAddressPlacement::SparseSelection(SparseSelectionRowAddressPlacement {
                sources: value
                    .sources
                    .into_iter()
                    .map(|source| {
                        Ok(SparseSelectionSource {
                            source: required(source.source, "SparseSelectionSource.source")?
                                .try_into()?,
                            selection: pooled_selection(
                                selection_pool,
                                source.selection_index,
                                "SparseSelectionSource.selection_index",
                            )?,
                            excluded: optional_pooled_selection(
                                selection_pool,
                                source.excluded_selection_index,
                                "SparseSelectionSource.excluded_selection_index",
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                destination_fragment_id: physical_fragment_id(
                    value.destination_fragment_id,
                    "SparseSelectionRowAddressPlacement",
                )?,
                destination_start: value.destination_start,
            })
        }
        Codec::ExplicitMap(value) => {
            RowAddressPlacement::ExplicitMap(ExplicitMapRowAddressPlacement {
                sources: value
                    .sources
                    .into_iter()
                    .map(|source| {
                        Ok(SparseSelectionSource {
                            source: required(source.source, "ExplicitMap source domain")?
                                .try_into()?,
                            selection: pooled_selection(
                                selection_pool,
                                source.selection_index,
                                "ExplicitMap source selection_index",
                            )?,
                            excluded: optional_pooled_selection(
                                selection_pool,
                                source.excluded_selection_index,
                                "ExplicitMap source excluded_selection_index",
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                object_path: value.object_path,
                object_size: value.object_size,
                pages: value
                    .pages
                    .into_iter()
                    .map(|page| ExplicitMapPage {
                        first_logical_address: page.first_logical_address,
                        last_logical_address: page.last_logical_address,
                        row_start: page.row_start,
                        row_count: page.row_count,
                        content_fingerprint: page.content_fingerprint,
                    })
                    .collect(),
                destinations: value
                    .destinations
                    .into_iter()
                    .map(|destination| {
                        Ok(ExplicitMapDestination {
                            physical_fragment_id: physical_fragment_id(
                                destination.physical_fragment_id,
                                "ExplicitMapDestination",
                            )?,
                            destination_start: destination.destination_start,
                            row_count: destination.row_count,
                            row_id_file_path: destination.row_id_file_path,
                            row_id_file_size: destination.row_id_file_size,
                            row_id_pages: destination
                                .row_id_pages
                                .into_iter()
                                .map(|page| ExplicitMapRowIdPage {
                                    row_start: page.row_start,
                                    row_count: page.row_count,
                                    content_fingerprint: page.content_fingerprint,
                                })
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                base_id: value.base_id,
            })
        }
    };
    placement.validate()?;
    Ok(placement)
}

impl TryFrom<pb::RowAddressPlacement> for RowAddressPlacement {
    type Error = Error;

    fn try_from(value: pb::RowAddressPlacement) -> Result<Self> {
        placement_from_proto(value, &[])
    }
}

fn encoded_selection_bytes(selection: &LogicalRowAddressSelection) -> Vec<u8> {
    let proto: pb::LogicalRowAddressSelection = selection.into();
    proto.encode_to_vec()
}

fn placement_selections(placement: &RowAddressPlacement) -> Vec<&LogicalRowAddressSelection> {
    match placement {
        RowAddressPlacement::Direct(value) => value.excluded.iter().map(AsRef::as_ref).collect(),
        RowAddressPlacement::Selected(value) => std::iter::once(value.selection.as_ref())
            .chain(value.excluded.iter().map(AsRef::as_ref))
            .collect(),
        RowAddressPlacement::SparseSelection(value) => value
            .sources
            .iter()
            .flat_map(|source| {
                std::iter::once(source.selection.as_ref())
                    .chain(source.excluded.iter().map(AsRef::as_ref))
            })
            .collect(),
        RowAddressPlacement::ExplicitMap(value) => value
            .sources
            .iter()
            .flat_map(|source| {
                std::iter::once(source.selection.as_ref())
                    .chain(source.excluded.iter().map(AsRef::as_ref))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn canonical_selection_indices<'a>(
    selections: impl Iterator<Item = &'a LogicalRowAddressSelection>,
) -> BTreeMap<Vec<u8>, u32> {
    let mut indices = selections
        .map(|selection| (encoded_selection_bytes(selection), 0_u32))
        .collect::<BTreeMap<_, _>>();
    for (index, value) in indices.values_mut().enumerate() {
        *value = index as u32;
    }
    indices
}

fn selection_pool_index(
    selection_indices: &BTreeMap<Vec<u8>, u32>,
    selection: &LogicalRowAddressSelection,
) -> u32 {
    selection_indices[&encoded_selection_bytes(selection)]
}

fn placement_to_proto(
    value: &RowAddressPlacement,
    selection_indices: &BTreeMap<Vec<u8>, u32>,
) -> pb::RowAddressPlacement {
    use pb::row_address_placement::Codec;
    let codec = match value {
        RowAddressPlacement::Direct(value) => Codec::Direct(pb::DirectRowAddressPlacement {
            source: Some((&value.source).into()),
            destination_fragment_id: value.destination_fragment_id as u64,
            destination_start: value.destination_start,
            excluded_selection_index: value
                .excluded
                .as_ref()
                .map(|excluded| selection_pool_index(selection_indices, excluded.as_ref())),
        }),
        RowAddressPlacement::PackedRun(value) => {
            Codec::PackedRun(pb::PackedRunRowAddressPlacement {
                first_logical_fragment_id: value.domains.first_logical_fragment_id,
                domain_count: value.domains.domain_count,
                logical_fragment_ids: Some(value.domains.logical_fragment_ids.clone()),
                slot_counts: Some(value.domains.slot_counts.clone()),
                creation_versions: Some(value.domains.creation_versions.clone()),
                destination_fragment_id: value.destination_fragment_id as u64,
                destination_start: value.destination_start,
            })
        }
        RowAddressPlacement::Selected(value) => Codec::Selected(pb::SelectedRowAddressPlacement {
            source: Some((&value.source).into()),
            selection_index: Some(selection_pool_index(
                selection_indices,
                value.selection.as_ref(),
            )),
            destination_fragment_id: value.destination_fragment_id as u64,
            destination_start: value.destination_start,
            excluded_selection_index: value
                .excluded
                .as_ref()
                .map(|excluded| selection_pool_index(selection_indices, excluded.as_ref())),
        }),
        RowAddressPlacement::ExtentList(value) => {
            Codec::ExtentList(pb::ExtentListRowAddressPlacement {
                source: Some((&value.source).into()),
                extents: value
                    .extents
                    .iter()
                    .map(|extent| pb::RowAddressExtent {
                        source_start: extent.source_start,
                        length: extent.length,
                        destination_fragment_id: extent.destination_fragment_id as u64,
                        destination_start: extent.destination_start,
                    })
                    .collect(),
            })
        }
        RowAddressPlacement::SparseSelection(value) => {
            Codec::SparseSelection(pb::SparseSelectionRowAddressPlacement {
                sources: value
                    .sources
                    .iter()
                    .map(|source| pb::SparseSelectionSource {
                        source: Some((&source.source).into()),
                        selection_index: Some(selection_pool_index(
                            selection_indices,
                            source.selection.as_ref(),
                        )),
                        excluded_selection_index: source.excluded.as_ref().map(|excluded| {
                            selection_pool_index(selection_indices, excluded.as_ref())
                        }),
                    })
                    .collect(),
                destination_fragment_id: value.destination_fragment_id as u64,
                destination_start: value.destination_start,
            })
        }
        RowAddressPlacement::ExplicitMap(value) => {
            Codec::ExplicitMap(pb::ExplicitMapRowAddressPlacement {
                sources: value
                    .sources
                    .iter()
                    .map(|source| pb::SparseSelectionSource {
                        source: Some((&source.source).into()),
                        selection_index: Some(selection_pool_index(
                            selection_indices,
                            source.selection.as_ref(),
                        )),
                        excluded_selection_index: source.excluded.as_ref().map(|excluded| {
                            selection_pool_index(selection_indices, excluded.as_ref())
                        }),
                    })
                    .collect(),
                object_path: value.object_path.clone(),
                object_size: value.object_size,
                pages: value
                    .pages
                    .iter()
                    .map(|page| pb::ExplicitMapPage {
                        first_logical_address: page.first_logical_address,
                        last_logical_address: page.last_logical_address,
                        row_start: page.row_start,
                        row_count: page.row_count,
                        content_fingerprint: page.content_fingerprint.clone(),
                    })
                    .collect(),
                destinations: value
                    .destinations
                    .iter()
                    .map(|destination| pb::ExplicitMapDestination {
                        physical_fragment_id: destination.physical_fragment_id as u64,
                        destination_start: destination.destination_start,
                        row_count: destination.row_count,
                        row_id_file_path: destination.row_id_file_path.clone(),
                        row_id_file_size: destination.row_id_file_size,
                        row_id_pages: destination
                            .row_id_pages
                            .iter()
                            .map(|page| pb::ExplicitMapRowIdPage {
                                row_start: page.row_start,
                                row_count: page.row_count,
                                content_fingerprint: page.content_fingerprint.clone(),
                            })
                            .collect(),
                    })
                    .collect(),
                base_id: value.base_id,
            })
        }
    };
    pb::RowAddressPlacement { codec: Some(codec) }
}

impl From<&RowAddressPlacement> for pb::RowAddressPlacement {
    fn from(value: &RowAddressPlacement) -> Self {
        let selections = placement_selections(value);
        let selection_indices = canonical_selection_indices(selections.into_iter());
        placement_to_proto(value, &selection_indices)
    }
}

impl RowAddressPlacement {
    fn for_each_source(
        &self,
        mut visit: impl FnMut(RowAddressLogicalDomain) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Direct(value) => visit(value.source)?,
            Self::PackedRun(value) => {
                for ordinal in 0..value.domains.domain_count() {
                    visit(value.domains.domain_at(ordinal)?)?;
                }
            }
            Self::Selected(value) => visit(value.source)?,
            Self::ExtentList(value) => visit(value.source)?,
            Self::SparseSelection(value) => {
                for source in &value.sources {
                    visit(source.source)?;
                }
            }
            Self::ExplicitMap(value) => {
                for source in &value.sources {
                    visit(source.source)?;
                }
            }
        }
        Ok(())
    }

    fn source_domain(&self, logical_fragment_id: u32) -> Result<Option<RowAddressLogicalDomain>> {
        match self {
            Self::Direct(value) if value.source.logical_fragment_id == logical_fragment_id => {
                Ok(Some(value.source))
            }
            Self::PackedRun(value) => value
                .domains
                .domain_ordinal(logical_fragment_id)?
                .map(|ordinal| value.domains.domain_at(ordinal))
                .transpose(),
            Self::Selected(value) if value.source.logical_fragment_id == logical_fragment_id => {
                Ok(Some(value.source))
            }
            Self::ExtentList(value) if value.source.logical_fragment_id == logical_fragment_id => {
                Ok(Some(value.source))
            }
            Self::SparseSelection(value) => Ok(value
                .sources
                .iter()
                .find(|source| source.source.logical_fragment_id == logical_fragment_id)
                .map(|source| source.source)),
            Self::ExplicitMap(value) => Ok(value
                .sources
                .iter()
                .find(|source| source.source.logical_fragment_id == logical_fragment_id)
                .map(|source| source.source)),
            _ => Ok(None),
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Direct(value) => {
                if let Some(excluded) = &value.excluded {
                    validate_selection_for_domain(excluded, &value.source)?;
                    if excluded.cardinality() >= value.source.slot_count as u64 {
                        return Err(Error::invalid_input(
                            "Direct exclusion must leave at least one mapped row",
                        ));
                    }
                }
                validate_destination_range(
                    value.destination_fragment_id,
                    value.destination_start,
                    value.source.slot_count as u64,
                )?;
            }
            Self::PackedRun(value) => {
                validate_destination_range(
                    value.destination_fragment_id,
                    value.destination_start,
                    value.domains.total_slot_count()?,
                )?;
            }
            Self::Selected(value) => {
                validate_selection_for_domain(&value.selection, &value.source)?;
                validate_excluded_selection(
                    value.excluded.as_deref(),
                    &value.selection,
                    &value.source,
                    "Selected",
                )?;
                validate_destination_range(
                    value.destination_fragment_id,
                    value.destination_start,
                    value.selection.cardinality(),
                )?;
            }
            Self::ExtentList(value) => {
                if value.extents.is_empty() {
                    return Err(Error::invalid_input(
                        "ExtentList placement must contain at least one extent",
                    ));
                }
                let mut previous_end = 0_u32;
                for extent in &value.extents {
                    let source_end = extent
                        .source_start
                        .checked_add(extent.length)
                        .ok_or_else(|| Error::invalid_input("ExtentList source range overflow"))?;
                    if extent.length == 0
                        || extent.source_start < previous_end
                        || source_end > value.source.slot_count
                    {
                        return Err(Error::invalid_input(
                            "ExtentList source extents must be non-empty, sorted, and within the source domain",
                        ));
                    }
                    validate_destination_range(
                        extent.destination_fragment_id,
                        extent.destination_start,
                        extent.length as u64,
                    )?;
                    previous_end = source_end;
                }
            }
            Self::SparseSelection(value) => {
                if value.sources.is_empty() {
                    return Err(Error::invalid_input(
                        "SparseSelection placement must contain source selections",
                    ));
                }
                validate_sparse_sources(&value.sources)?;
                let cardinality = value.sources.iter().try_fold(0_u64, |total, source| {
                    total.checked_add(source.selection.cardinality())
                });
                validate_destination_range(
                    value.destination_fragment_id,
                    value.destination_start,
                    cardinality.ok_or_else(|| {
                        Error::invalid_input("SparseSelection placement cardinality overflow")
                    })?,
                )?;
            }
            Self::ExplicitMap(value) => {
                if value.sources.is_empty()
                    || value.object_path.is_empty()
                    || value.object_size == 0
                    || value.pages.is_empty()
                    || value.destinations.is_empty()
                {
                    return Err(Error::invalid_input(
                        "ExplicitMap placement requires sources, object metadata, and pages",
                    ));
                }
                validate_explicit_sources(&value.sources)?;
                if value.sources.iter().any(|source| {
                    !selection_is_full_domain(&source.selection, source.source).unwrap_or(false)
                }) {
                    return Err(Error::invalid_input(
                        "ExplicitMap roots must describe complete logical domains; live membership is authoritative in the external locator",
                    ));
                }
                let mut previous_last = None;
                let mut previous_row_end = 0_u64;
                for page in &value.pages {
                    LogicalRowAddress::try_from(page.first_logical_address)?;
                    LogicalRowAddress::try_from(page.last_logical_address)?;
                    let row_end = page.row_start.checked_add(page.row_count).ok_or_else(|| {
                        Error::invalid_input("ExplicitMap page row range overflow")
                    })?;
                    if page.first_logical_address > page.last_logical_address
                        || previous_last.is_some_and(|last| last >= page.first_logical_address)
                        || page.row_count == 0
                        || page.row_start != previous_row_end
                        || page.content_fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                    {
                        return Err(Error::invalid_input(
                            "ExplicitMap pages must be non-empty, logical-sorted, non-overlapping, row-contiguous, and carry a 16-byte content fingerprint",
                        ));
                    }
                    previous_last = Some(page.last_logical_address);
                    previous_row_end = row_end;
                }
                for destination in &value.destinations {
                    if destination.row_id_file_path.is_empty() || destination.row_id_file_size == 0
                    {
                        return Err(Error::invalid_input(
                            "ExplicitMap destination requires hidden _rowid file metadata",
                        ));
                    }
                    let mut previous_page_end = 0_u64;
                    if destination.row_id_pages.is_empty() {
                        return Err(Error::invalid_input(
                            "ExplicitMap destination requires hidden _rowid page fingerprints",
                        ));
                    }
                    for page in &destination.row_id_pages {
                        let page_end =
                            page.row_start.checked_add(page.row_count).ok_or_else(|| {
                                Error::invalid_input(
                                    "ExplicitMap hidden _rowid page row range overflow",
                                )
                            })?;
                        if page.row_count == 0
                            || page.row_start != previous_page_end
                            || page.content_fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                        {
                            return Err(Error::invalid_input(
                                "ExplicitMap hidden _rowid pages must be non-empty, row-contiguous, and carry a 16-byte content fingerprint",
                            ));
                        }
                        previous_page_end = page_end;
                    }
                    if previous_page_end != destination.row_count as u64 {
                        return Err(Error::invalid_input(format!(
                            "ExplicitMap hidden _rowid pages cover {} rows but destination declares {}",
                            previous_page_end, destination.row_count
                        )));
                    }
                    validate_destination_range(
                        destination.physical_fragment_id,
                        destination.destination_start,
                        destination.row_count as u64,
                    )?;
                }
                let source_rows = value
                    .sources
                    .iter()
                    .try_fold(0_u64, |total, source| {
                        total.checked_add(source.selection.cardinality())
                    })
                    .ok_or_else(|| {
                        Error::invalid_input("ExplicitMap source cardinality overflow")
                    })?;
                let destination_rows = value
                    .destinations
                    .iter()
                    .try_fold(0_u64, |total, destination| {
                        total.checked_add(destination.row_count as u64)
                    })
                    .ok_or_else(|| {
                        Error::invalid_input("ExplicitMap destination cardinality overflow")
                    })?;
                if destination_rows > previous_row_end || previous_row_end > source_rows {
                    return Err(Error::invalid_input(format!(
                        "ExplicitMap cardinalities must satisfy current destinations ({}) <= immutable locator ({}) <= source capacity ({})",
                        destination_rows, previous_row_end, source_rows
                    )));
                }
                if destination_rows > source_rows {
                    return Err(Error::invalid_input(format!(
                        "ExplicitMap destinations exceed their source-domain capacity: source_rows={}, destination_rows={}",
                        source_rows, destination_rows
                    )));
                }
            }
        }
        self.for_each_source(|source| {
            RowAddressLogicalDomain::new(
                source.logical_fragment_id,
                source.slot_count,
                source.creation_version,
            )?;
            Ok(())
        })?;
        Ok(())
    }
}

fn validate_strictly_sorted_domains(domains: &[RowAddressLogicalDomain]) -> Result<()> {
    for domain in domains {
        RowAddressLogicalDomain::new(
            domain.logical_fragment_id,
            domain.slot_count,
            domain.creation_version,
        )?;
    }
    if domains
        .windows(2)
        .any(|pair| pair[0].logical_fragment_id >= pair[1].logical_fragment_id)
    {
        return Err(Error::invalid_input(
            "logical domains must be strictly sorted by logical fragment id",
        ));
    }
    Ok(())
}

fn validate_sparse_sources(sources: &[SparseSelectionSource]) -> Result<()> {
    let domains = sources
        .iter()
        .map(|source| source.source)
        .collect::<Vec<_>>();
    validate_strictly_sorted_domains(&domains)?;
    for source in sources {
        validate_selection_for_domain(&source.selection, &source.source)?;
        validate_excluded_selection(
            source.excluded.as_deref(),
            &source.selection,
            &source.source,
            "SparseSelection",
        )?;
    }
    Ok(())
}

fn validate_explicit_sources(sources: &[SparseSelectionSource]) -> Result<()> {
    let domains = sources
        .iter()
        .map(|source| source.source)
        .collect::<Vec<_>>();
    validate_strictly_sorted_domains(&domains)?;
    let mut effective_rows = 0_u64;
    for source in sources {
        validate_selection_for_domain(&source.selection, &source.source)?;
        if let Some(excluded) = source.excluded.as_deref() {
            validate_selection_for_domain(excluded, &source.source)?;
            if excluded.cardinality() > source.selection.cardinality() {
                return Err(Error::invalid_input(
                    "ExplicitMap exclusion exceeds its source selection",
                ));
            }
            if !excluded.is_subset_of(&source.selection)? {
                return Err(Error::invalid_input(
                    "ExplicitMap exclusion contains an address outside its source selection",
                ));
            }
        }
        effective_rows = effective_rows
            .checked_add(
                source.selection.cardinality()
                    - source
                        .excluded
                        .as_ref()
                        .map_or(0, |selection| selection.cardinality()),
            )
            .ok_or_else(|| Error::invalid_input("ExplicitMap effective cardinality overflow"))?;
    }
    if effective_rows == 0 {
        return Err(Error::invalid_input(
            "ExplicitMap exclusions must leave at least one mapped row",
        ));
    }
    Ok(())
}

fn validate_excluded_selection(
    excluded: Option<&LogicalRowAddressSelection>,
    selected: &LogicalRowAddressSelection,
    domain: &RowAddressLogicalDomain,
    context: &str,
) -> Result<()> {
    let Some(excluded) = excluded else {
        return Ok(());
    };
    validate_selection_for_domain(excluded, domain)?;
    if excluded.cardinality() >= selected.cardinality() {
        return Err(Error::invalid_input(format!(
            "{context} exclusion must leave at least one mapped row"
        )));
    }
    if !excluded.is_subset_of(selected)? {
        return Err(Error::invalid_input(format!(
            "{context} exclusion contains an address outside its source selection"
        )));
    }
    Ok(())
}

fn is_excluded(
    excluded: Option<&Arc<LogicalRowAddressSelection>>,
    address: LogicalRowAddress,
) -> Result<bool> {
    excluded
        .map(|selection| selection.contains(address))
        .unwrap_or(Ok(false))
}

fn validate_selection_for_domain(
    selection: &LogicalRowAddressSelection,
    domain: &RowAddressLogicalDomain,
) -> Result<()> {
    selection.validate()?;
    if selection.is_empty() {
        return Err(Error::invalid_input(
            "placement source selection must be non-empty",
        ));
    }
    for ordinal in [0, selection.cardinality() - 1] {
        let address = selection.select(ordinal)?.ok_or_else(|| {
            Error::invalid_input("logical row address selection rank/select mismatch")
        })?;
        if address.logical_fragment_id() != domain.logical_fragment_id
            || address.immutable_slot() >= domain.slot_count
        {
            return Err(Error::invalid_input(format!(
                "selection address {} is outside logical domain {} with slot_count {}",
                address, domain.logical_fragment_id, domain.slot_count
            )));
        }
    }
    Ok(())
}

fn validate_destination_range(fragment_id: u32, start: u32, cardinality: u64) -> Result<()> {
    let end = (start as u64)
        .checked_add(cardinality)
        .ok_or_else(|| Error::invalid_input("physical destination range overflow"))?;
    if fragment_id == INVALID_ID || cardinality == 0 || end > INVALID_ID as u64 {
        return Err(Error::invalid_input(format!(
            "invalid physical destination range: fragment_id={}, start={}, cardinality={}",
            fragment_id, start, cardinality
        )));
    }
    Ok(())
}

impl From<&FieldGeneration> for pb::FieldGeneration {
    fn from(value: &FieldGeneration) -> Self {
        Self {
            field_id: value.field_id,
            generation: value.generation,
        }
    }
}

impl TryFrom<pb::FieldGeneration> for FieldGeneration {
    type Error = Error;

    fn try_from(value: pb::FieldGeneration) -> Result<Self> {
        if value.field_id < 0 || value.generation == 0 {
            return Err(Error::invalid_input(format!(
                "field generation must have a non-negative field id and non-zero generation: field_id={}, generation={}",
                value.field_id, value.generation
            )));
        }
        Ok(Self {
            field_id: value.field_id,
            generation: value.generation,
        })
    }
}

fn content_generation_region_to_proto(
    value: &ContentGenerationRegion,
    selection_indices: &BTreeMap<Vec<u8>, u32>,
) -> pb::ContentGenerationRegion {
    pb::ContentGenerationRegion {
        selection_index: Some(selection_pool_index(
            selection_indices,
            value.selection.as_ref(),
        )),
        field_ids: value.field_ids.clone(),
        generation: value.generation,
    }
}

fn content_generation_region_from_proto(
    value: pb::ContentGenerationRegion,
    selection_pool: &[Arc<LogicalRowAddressSelection>],
) -> Result<ContentGenerationRegion> {
    let region = ContentGenerationRegion {
        selection: pooled_selection(
            selection_pool,
            value.selection_index,
            "ContentGenerationRegion.selection_index",
        )?,
        field_ids: value.field_ids,
        generation: value.generation,
    };
    validate_field_ids(&region.field_ids, "ContentGenerationRegion.field_ids")?;
    if region.selection.is_empty() || region.generation == 0 {
        return Err(Error::invalid_input(
            "content generation region must have a non-empty selection and non-zero generation",
        ));
    }
    Ok(region)
}

impl From<&RowAddressPlacementDebtSummary> for pb::RowAddressPlacementDebtSummary {
    fn from(value: &RowAddressPlacementDebtSummary) -> Self {
        Self {
            canonical_layout_bytes: value.canonical_layout_bytes,
            metadata_bytes_written_since_maintenance: value
                .metadata_bytes_written_since_maintenance,
            max_extents_per_logical_fragment: value.max_extents_per_logical_fragment,
            live_physical_rows: value.live_physical_rows,
            total_physical_rows: value.total_physical_rows,
            explicit_layout_bytes: value.explicit_layout_bytes,
            fast_delta_bytes: value.fast_delta_bytes,
            explicit_delta_bytes: value.explicit_delta_bytes,
            explicit_metadata_bytes_written_since_maintenance: value
                .explicit_metadata_bytes_written_since_maintenance,
            generation_delta_bytes: value.generation_delta_bytes,
            generation_metadata_bytes_written_since_maintenance: value
                .generation_metadata_bytes_written_since_maintenance,
        }
    }
}

impl From<pb::RowAddressPlacementDebtSummary> for RowAddressPlacementDebtSummary {
    fn from(value: pb::RowAddressPlacementDebtSummary) -> Self {
        Self {
            canonical_layout_bytes: value.canonical_layout_bytes,
            metadata_bytes_written_since_maintenance: value
                .metadata_bytes_written_since_maintenance,
            max_extents_per_logical_fragment: value.max_extents_per_logical_fragment,
            live_physical_rows: value.live_physical_rows,
            total_physical_rows: value.total_physical_rows,
            explicit_layout_bytes: value.explicit_layout_bytes,
            fast_delta_bytes: value.fast_delta_bytes,
            explicit_delta_bytes: value.explicit_delta_bytes,
            explicit_metadata_bytes_written_since_maintenance: value
                .explicit_metadata_bytes_written_since_maintenance,
            generation_delta_bytes: value.generation_delta_bytes,
            generation_metadata_bytes_written_since_maintenance: value
                .generation_metadata_bytes_written_since_maintenance,
        }
    }
}

impl From<&PhysicalRowOwnershipSummary> for pb::PhysicalRowOwnershipSummary {
    fn from(value: &PhysicalRowOwnershipSummary) -> Self {
        Self {
            physical_fragment_id: value.physical_fragment_id as u64,
            mapped_row_count: value.mapped_row_count,
            mapped_offsets_fingerprint: value.mapped_offsets_fingerprint.clone(),
            deletion_offsets_fingerprint: value.deletion_offsets_fingerprint.clone(),
            unowned_row_count: value.unowned_row_count,
        }
    }
}

impl TryFrom<pb::PhysicalRowOwnershipSummary> for PhysicalRowOwnershipSummary {
    type Error = Error;

    fn try_from(value: pb::PhysicalRowOwnershipSummary) -> Result<Self> {
        let summary = Self {
            physical_fragment_id: physical_fragment_id(
                value.physical_fragment_id,
                "PhysicalRowOwnershipSummary",
            )?,
            mapped_row_count: value.mapped_row_count,
            mapped_offsets_fingerprint: value.mapped_offsets_fingerprint,
            deletion_offsets_fingerprint: value.deletion_offsets_fingerprint,
            unowned_row_count: value.unowned_row_count,
        };
        if summary.mapped_offsets_fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
            || !summary.deletion_offsets_fingerprint.is_empty()
                && summary.deletion_offsets_fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
        {
            return Err(Error::invalid_input(
                "physical ownership fingerprints must be empty or 16 bytes",
            ));
        }
        Ok(summary)
    }
}

impl From<&RowAddressDestinationIndexEntry> for pb::RowAddressDestinationIndexEntry {
    fn from(value: &RowAddressDestinationIndexEntry) -> Self {
        Self {
            physical_fragment_id: value.physical_fragment_id as u64,
            placement_indices: value.placement_indices.clone(),
        }
    }
}

impl TryFrom<pb::RowAddressDestinationIndexEntry> for RowAddressDestinationIndexEntry {
    type Error = Error;

    fn try_from(value: pb::RowAddressDestinationIndexEntry) -> Result<Self> {
        let entry = Self {
            physical_fragment_id: physical_fragment_id(
                value.physical_fragment_id,
                "RowAddressDestinationIndexEntry",
            )?,
            placement_indices: value.placement_indices,
        };
        if entry.placement_indices.is_empty()
            || entry
                .placement_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::invalid_input(
                "destination index placement indices must be non-empty and strictly sorted",
            ));
        }
        Ok(entry)
    }
}

fn retired_logical_row_set_to_proto(
    value: &RetiredLogicalRowSet,
    selection_indices: &BTreeMap<Vec<u8>, u32>,
) -> pb::RetiredLogicalRowSet {
    use pb::retired_logical_row_set::Membership;
    let membership = match &value.membership {
        RetiredLogicalRowMembership::AllRows => Membership::AllRows(pb::AllRetiredLogicalRows {}),
        RetiredLogicalRowMembership::Selection(selection) => {
            Membership::SelectionIndex(selection_pool_index(selection_indices, selection.as_ref()))
        }
    };
    pb::RetiredLogicalRowSet {
        first_logical_fragment_id: value.domains.first_logical_fragment_id,
        domain_count: value.domains.domain_count,
        logical_fragment_ids: Some(value.domains.logical_fragment_ids.clone()),
        slot_counts: Some(value.domains.slot_counts.clone()),
        creation_versions: Some(value.domains.creation_versions.clone()),
        membership: Some(membership),
    }
}

fn retired_logical_row_set_from_proto(
    value: pb::RetiredLogicalRowSet,
    selection_pool: &[Arc<LogicalRowAddressSelection>],
) -> Result<RetiredLogicalRowSet> {
    use pb::retired_logical_row_set::Membership;
    let domains = PackedLogicalDomainRun::try_new(
        value.first_logical_fragment_id,
        value.domain_count,
        required(
            value.logical_fragment_ids,
            "RetiredLogicalRowSet.logical_fragment_ids",
        )?,
        required(value.slot_counts, "RetiredLogicalRowSet.slot_counts")?,
        required(
            value.creation_versions,
            "RetiredLogicalRowSet.creation_versions",
        )?,
    )?;
    let membership = match required(value.membership, "RetiredLogicalRowSet.membership")? {
        Membership::AllRows(_) => RetiredLogicalRowMembership::AllRows,
        Membership::SelectionIndex(index) => {
            RetiredLogicalRowMembership::Selection(pooled_selection(
                selection_pool,
                Some(index),
                "RetiredLogicalRowSet.selection_index",
            )?)
        }
    };
    let value = RetiredLogicalRowSet {
        domains,
        membership,
    };
    value.validate()?;
    Ok(value)
}

impl From<&RowAddressLayout> for pb::RowAddressLayout {
    fn from(value: &RowAddressLayout) -> Self {
        let selection_indices = canonical_selection_indices(
            value
                .selection_pool
                .iter()
                .map(|selection| selection.as_ref()),
        );
        Self {
            encoding_version: value.encoding_version,
            namespace_uuid: Some((&value.namespace_uuid).into()),
            placements: value
                .placements
                .iter()
                .map(|placement| placement_to_proto(placement, &selection_indices))
                .collect(),
            destination_index: value.destination_index.iter().map(Into::into).collect(),
            field_default_generations: value
                .field_default_generations
                .iter()
                .map(Into::into)
                .collect(),
            generation_regions: value
                .generation_regions
                .iter()
                .map(|region| content_generation_region_to_proto(region, &selection_indices))
                .collect(),
            index_commit_floors: value.index_commit_floors.iter().map(Into::into).collect(),
            debt_summary: Some((&value.debt_summary).into()),
            fingerprint: value.fingerprint.clone(),
            physical_row_ownership: value
                .physical_row_ownership
                .iter()
                .map(Into::into)
                .collect(),
            selection_pool: value
                .selection_pool
                .iter()
                .map(|selection| selection.as_ref().into())
                .collect(),
            retired_rows: value
                .retired_rows
                .iter()
                .map(|retired| retired_logical_row_set_to_proto(retired, &selection_indices))
                .collect(),
            logical_domain_fingerprint: value.logical_domain_fingerprint.clone(),
        }
    }
}

impl TryFrom<pb::RowAddressLayout> for RowAddressLayout {
    type Error = Error;

    fn try_from(value: pb::RowAddressLayout) -> Result<Self> {
        let namespace_uuid = value
            .namespace_uuid
            .as_ref()
            .map(Uuid::try_from)
            .ok_or_else(|| Error::invalid_input("RowAddressLayout.namespace_uuid is missing"))??;
        let selection_pool = value
            .selection_pool
            .into_iter()
            .map(LogicalRowAddressSelection::try_from)
            .map(|selection| selection.map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let layout = Self {
            encoding_version: value.encoding_version,
            namespace_uuid,
            placements: value
                .placements
                .into_iter()
                .map(|placement| placement_from_proto(placement, &selection_pool))
                .collect::<Result<Vec<_>>>()?,
            destination_index: value
                .destination_index
                .into_iter()
                .map(RowAddressDestinationIndexEntry::try_from)
                .collect::<Result<Vec<_>>>()?,
            field_default_generations: value
                .field_default_generations
                .into_iter()
                .map(FieldGeneration::try_from)
                .collect::<Result<Vec<_>>>()?,
            generation_regions: value
                .generation_regions
                .into_iter()
                .map(|region| content_generation_region_from_proto(region, &selection_pool))
                .collect::<Result<Vec<_>>>()?,
            index_commit_floors: value
                .index_commit_floors
                .into_iter()
                .map(FieldGeneration::try_from)
                .collect::<Result<Vec<_>>>()?,
            debt_summary: value.debt_summary.unwrap_or_default().into(),
            fingerprint: value.fingerprint,
            physical_row_ownership: value
                .physical_row_ownership
                .into_iter()
                .map(PhysicalRowOwnershipSummary::try_from)
                .collect::<Result<Vec<_>>>()?,
            retired_rows: value
                .retired_rows
                .into_iter()
                .map(|retired| retired_logical_row_set_from_proto(retired, &selection_pool))
                .collect::<Result<Vec<_>>>()?,
            logical_domain_fingerprint: value.logical_domain_fingerprint,
            selection_pool,
        };
        layout.validate()?;
        Ok(layout)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogicalPhysicalExtent {
    source: RowAddressLogicalDomain,
    source_start: u32,
    length: u32,
    destination_fragment_id: u32,
    destination_start: u32,
}

fn bitmap_ranges(bitmap: &RoaringBitmap) -> impl Iterator<Item = (u32, u32)> + '_ {
    let mut values = bitmap.iter();
    std::iter::from_fn(move || {
        values
            .next_range()
            .and_then(|range| Some((*range.start(), range.end().checked_add(1)?)))
    })
}

fn subtract_source_range(
    source: RowAddressLogicalDomain,
    source_start: u32,
    length: u32,
    destination_fragment_id: u32,
    destination_start: u32,
    removed: Option<&RoaringBitmap>,
    output: &mut Vec<LogicalPhysicalExtent>,
) -> Result<()> {
    let source_end = source_start
        .checked_add(length)
        .ok_or_else(|| Error::invalid_input("source extent end overflow"))?;
    let Some(removed) = removed else {
        output.push(LogicalPhysicalExtent {
            source,
            source_start,
            length,
            destination_fragment_id,
            destination_start,
        });
        return Ok(());
    };
    let mut cursor = source_start;
    for removed_slot in removed.range(source_start..source_end) {
        if cursor < removed_slot {
            output.push(LogicalPhysicalExtent {
                source,
                source_start: cursor,
                length: removed_slot - cursor,
                destination_fragment_id,
                destination_start: destination_start + cursor - source_start,
            });
        }
        cursor = removed_slot
            .checked_add(1)
            .ok_or_else(|| Error::invalid_input("removed source slot overflow"))?;
    }
    if cursor < source_end {
        output.push(LogicalPhysicalExtent {
            source,
            source_start: cursor,
            length: source_end - cursor,
            destination_fragment_id,
            destination_start: destination_start + cursor - source_start,
        });
    }
    Ok(())
}

fn selection_intersects_removed(
    selection: &LogicalRowAddressSelection,
    excluded: Option<&Arc<LogicalRowAddressSelection>>,
    source: RowAddressLogicalDomain,
    removed: &RoaringBitmap,
) -> Result<bool> {
    let mut selected_slots = selection_slots_for_domain(selection, source.logical_fragment_id)?;
    if let Some(excluded) = excluded {
        selected_slots -=
            selection_slots_for_domain(excluded.as_ref(), source.logical_fragment_id)?;
    }
    Ok(!selected_slots.is_disjoint(removed))
}

fn slots_within_domain(slots: &RoaringBitmap, source: RowAddressLogicalDomain) -> RoaringBitmap {
    let mut within_domain = RoaringBitmap::new();
    within_domain.insert_range(0..source.slot_count);
    within_domain &= slots;
    within_domain
}

fn direct_intersects_removed(
    source: RowAddressLogicalDomain,
    excluded: Option<&Arc<LogicalRowAddressSelection>>,
    removed: &RoaringBitmap,
) -> Result<bool> {
    let mut touched = slots_within_domain(removed, source);
    if let Some(excluded) = excluded {
        touched -= selection_slots_for_domain(excluded, source.logical_fragment_id)?;
    }
    Ok(!touched.is_empty())
}

fn selection_bitmap_for_domain(logical_fragment_id: u32, slots: RoaringBitmap) -> RoaringTreemap {
    if slots.is_empty() {
        RoaringTreemap::new()
    } else {
        RoaringTreemap::from_bitmaps([(logical_fragment_id, slots)])
    }
}

fn merge_excluded_slots(
    source: RowAddressLogicalDomain,
    selected: Option<&LogicalRowAddressSelection>,
    existing: Option<&Arc<LogicalRowAddressSelection>>,
    removed: &RoaringBitmap,
) -> Result<Arc<LogicalRowAddressSelection>> {
    let mut excluded = existing
        .map(|selection| selection.to_roaring_treemap())
        .transpose()?
        .unwrap_or_default();
    let mut added = slots_within_domain(removed, source);
    if let Some(selected) = selected {
        added &= selection_slots_for_domain(selected, source.logical_fragment_id)?;
    }
    excluded |= selection_bitmap_for_domain(source.logical_fragment_id, added);
    Ok(Arc::new(LogicalRowAddressSelection::from_bitmap(excluded)?))
}

fn selection_slots_for_domain(
    selection: &LogicalRowAddressSelection,
    logical_fragment_id: u32,
) -> Result<RoaringBitmap> {
    match selection {
        LogicalRowAddressSelection::Ranges(value) => {
            let mut slots = RoaringBitmap::new();
            for range in value
                .ranges
                .iter()
                .filter(|range| range.logical_fragment_id == logical_fragment_id)
            {
                slots.insert_range(range.start_slot..range.end_slot);
            }
            Ok(slots)
        }
        LogicalRowAddressSelection::Roaring(value) => Ok(value
            .bitmap()?
            .bitmaps()
            .find_map(|(domain, slots)| (domain == logical_fragment_id).then(|| slots.clone()))
            .unwrap_or_default()),
        _ => Ok(selection
            .to_roaring_treemap()?
            .bitmaps()
            .find_map(|(domain, slots)| (domain == logical_fragment_id).then(|| slots.clone()))
            .unwrap_or_default()),
    }
}

fn residual_placements_from_extents(
    extents: Vec<LogicalPhysicalExtent>,
) -> Result<Vec<RowAddressPlacement>> {
    let mut by_domain = BTreeMap::<u32, (RowAddressLogicalDomain, Vec<RowAddressExtent>)>::new();
    for extent in extents {
        let entry = by_domain
            .entry(extent.source.logical_fragment_id)
            .or_insert_with(|| (extent.source, Vec::new()));
        if entry.0 != extent.source {
            return Err(Error::invalid_input(
                "logical domain metadata changed while rebuilding residual placement",
            ));
        }
        entry.1.push(RowAddressExtent {
            source_start: extent.source_start,
            length: extent.length,
            destination_fragment_id: extent.destination_fragment_id,
            destination_start: extent.destination_start,
        });
    }
    let mut placements = Vec::with_capacity(by_domain.len());
    for (_, (source, mut source_extents)) in by_domain {
        source_extents.sort_by_key(|extent| extent.source_start);
        let mut merged = Vec::<RowAddressExtent>::with_capacity(source_extents.len());
        for extent in source_extents {
            if let Some(previous) = merged.last_mut()
                && previous.source_start + previous.length == extent.source_start
                && previous.destination_fragment_id == extent.destination_fragment_id
                && previous.destination_start + previous.length == extent.destination_start
            {
                previous.length = previous
                    .length
                    .checked_add(extent.length)
                    .ok_or_else(|| Error::invalid_input("merged extent length overflow"))?;
            } else {
                merged.push(extent);
            }
        }
        let affine_destination = merged.first().and_then(|first| {
            first
                .destination_start
                .checked_sub(first.source_start)
                .map(|base| (first.destination_fragment_id, base))
        });
        let is_affine = affine_destination.is_some_and(|(fragment_id, base)| {
            merged.iter().all(|extent| {
                extent.destination_fragment_id == fragment_id
                    && base
                        .checked_add(extent.source_start)
                        .is_some_and(|start| start == extent.destination_start)
            })
        });
        if is_affine {
            let (destination_fragment_id, destination_start) = affine_destination.unwrap();
            let mut excluded_slots = RoaringBitmap::new();
            let mut cursor = 0_u32;
            for extent in &merged {
                if cursor < extent.source_start {
                    excluded_slots.insert_range(cursor..extent.source_start);
                }
                cursor = extent.source_start + extent.length;
            }
            if cursor < source.slot_count {
                excluded_slots.insert_range(cursor..source.slot_count);
            }
            placements.push(RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source,
                destination_fragment_id,
                destination_start,
                excluded: (!excluded_slots.is_empty())
                    .then(|| {
                        selection_from_slot_bitmap(source.logical_fragment_id, &excluded_slots)
                    })
                    .transpose()?,
            }));
        } else if !merged.is_empty() {
            placements.push(RowAddressPlacement::ExtentList(
                ExtentListRowAddressPlacement {
                    source,
                    extents: merged,
                },
            ));
        }
    }
    Ok(placements)
}

fn placement_inline_bytes(placement: &RowAddressPlacement) -> u64 {
    let selections = placement_selections(placement);
    let indices = canonical_selection_indices(selections.iter().copied());
    let placement_bytes = placement_to_proto(placement, &indices).encoded_len() as u64;
    placement_bytes
        + indices
            .keys()
            .map(|selection| selection.len() as u64)
            .sum::<u64>()
}

fn selection_from_slot_bitmap(
    logical_fragment_id: u32,
    slots: &RoaringBitmap,
) -> Result<Arc<LogicalRowAddressSelection>> {
    let bitmap = RoaringTreemap::from_bitmaps([(logical_fragment_id, slots.clone())]);
    Ok(Arc::new(LogicalRowAddressSelection::from_bitmap(bitmap)?))
}

fn selection_is_full_domain(
    selection: &LogicalRowAddressSelection,
    source: RowAddressLogicalDomain,
) -> Result<bool> {
    if selection.cardinality() != source.slot_count as u64 {
        return Ok(false);
    }
    let first = selection.select(0)?.ok_or_else(|| {
        Error::invalid_input("full-domain selection is missing its first address")
    })?;
    let last = selection
        .select(selection.cardinality() - 1)?
        .ok_or_else(|| Error::invalid_input("full-domain selection is missing its last address"))?;
    Ok(first.logical_fragment_id() == source.logical_fragment_id
        && first.immutable_slot() == 0
        && last.logical_fragment_id() == source.logical_fragment_id
        && last.immutable_slot() == source.slot_count - 1)
}

fn slots_cover_domain(slots: &RoaringBitmap, source: RowAddressLogicalDomain) -> bool {
    slots.range_cardinality(0..source.slot_count) == u64::from(source.slot_count)
}

fn subtract_placement(
    placement: &RowAddressPlacement,
    removed: &BTreeMap<u32, RoaringBitmap>,
    preserved: &mut Vec<RowAddressPlacement>,
    residual: &mut Vec<LogicalPhysicalExtent>,
) -> Result<Option<PlacementMaintenanceRequired>> {
    let touches = match placement {
        RowAddressPlacement::Direct(value) => {
            if let Some(slots) = removed.get(&value.source.logical_fragment_id) {
                if slots_cover_domain(slots, value.source) {
                    return Ok(None);
                }
                direct_intersects_removed(value.source, value.excluded.as_ref(), slots)?
            } else {
                false
            }
        }
        RowAddressPlacement::PackedRun(value) => {
            let mut touches = false;
            for ordinal in 0..value.domains.domain_count() {
                let logical_fragment_id = value.domains.logical_fragment_id_at(ordinal)?;
                if let Some(removed) = removed.get(&logical_fragment_id)
                    && removed
                        .range(0..value.domains.slot_count_at(ordinal)?)
                        .next()
                        .is_some()
                {
                    touches = true;
                    break;
                }
            }
            touches
        }
        RowAddressPlacement::Selected(value) => {
            if let Some(slots) = removed.get(&value.source.logical_fragment_id) {
                slots_cover_domain(slots, value.source)
                    || selection_intersects_removed(
                        &value.selection,
                        value.excluded.as_ref(),
                        value.source,
                        slots,
                    )?
            } else {
                false
            }
        }
        RowAddressPlacement::ExtentList(value) => removed
            .get(&value.source.logical_fragment_id)
            .is_some_and(|slots| {
                value.extents.iter().any(|extent| {
                    slots
                        .range(extent.source_start..extent.source_start + extent.length)
                        .next()
                        .is_some()
                })
            }),
        RowAddressPlacement::SparseSelection(value) => {
            let mut touches = false;
            for source in &value.sources {
                if let Some(slots) = removed.get(&source.source.logical_fragment_id)
                    && (slots_cover_domain(slots, source.source)
                        || selection_intersects_removed(
                            &source.selection,
                            source.excluded.as_ref(),
                            source.source,
                            slots,
                        )?)
                {
                    touches = true;
                    break;
                }
            }
            touches
        }
        RowAddressPlacement::ExplicitMap(value) => {
            let mut updated = value.clone();
            let mut touched = false;
            for source in &mut updated.sources {
                if let Some(slots) = removed.get(&source.source.logical_fragment_id) {
                    if slots_cover_domain(slots, source.source) {
                        source.excluded = Some(source.selection.clone());
                        touched = true;
                    } else if selection_intersects_removed(
                        &source.selection,
                        source.excluded.as_ref(),
                        source.source,
                        slots,
                    )? {
                        source.excluded = Some(merge_excluded_slots(
                            source.source,
                            Some(&source.selection),
                            source.excluded.as_ref(),
                            slots,
                        )?);
                        touched = true;
                    }
                }
            }
            if touched {
                // The external locator is immutable.  A root-side exclusion
                // supersedes touched identities while a new placement takes
                // ownership, without rewriting unrelated rows or the locator.
                if updated.sources.iter().any(|source| {
                    source
                        .excluded
                        .as_ref()
                        .map_or(0, |excluded| excluded.cardinality())
                        < source.selection.cardinality()
                }) {
                    preserved.push(RowAddressPlacement::ExplicitMap(updated));
                }
                return Ok(None);
            }
            false
        }
    };
    if !touches {
        preserved.push(placement.clone());
        return Ok(None);
    }
    match placement {
        RowAddressPlacement::Direct(value) => {
            let removed_slots =
                removed
                    .get(&value.source.logical_fragment_id)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "removed slot map lost direct source logical fragment {}",
                            value.source.logical_fragment_id
                        ))
                    })?;
            let excluded =
                merge_excluded_slots(value.source, None, value.excluded.as_ref(), removed_slots)?;
            if excluded.cardinality() < value.source.slot_count as u64 {
                preserved.push(RowAddressPlacement::Direct(DirectRowAddressPlacement {
                    excluded: Some(excluded),
                    ..value.clone()
                }));
            }
        }
        RowAddressPlacement::PackedRun(value) => {
            let mut touched_ordinals = Vec::<(u32, u32)>::new();
            for ordinal in 0..value.domains.domain_count() {
                let logical_fragment_id = value.domains.logical_fragment_id_at(ordinal)?;
                if let Some(removed) = removed.get(&logical_fragment_id)
                    && removed
                        .range(0..value.domains.slot_count_at(ordinal)?)
                        .next()
                        .is_some()
                {
                    touched_ordinals.push((ordinal, logical_fragment_id));
                }
            }
            touched_ordinals.sort_unstable();
            let mut cursor = 0_u32;
            for (ordinal, logical_fragment_id) in touched_ordinals {
                if cursor < ordinal {
                    let Some(domains) = value.domains.compressed_slice(cursor, ordinal)? else {
                        return Ok(Some(
                            PlacementMaintenanceRequired::PackedRunSubtractionRequiresRewrite {
                                logical_fragment_id,
                            },
                        ));
                    };
                    preserved.push(RowAddressPlacement::PackedRun(
                        PackedRunRowAddressPlacement {
                            domains,
                            destination_fragment_id: value.destination_fragment_id,
                            destination_start: value
                                .destination_start
                                .checked_add(
                                    u32::try_from(value.domains.slot_prefix(cursor)?).map_err(
                                        |_| {
                                            Error::invalid_input(
                                                "PackedRun destination prefix exceeds u32",
                                            )
                                        },
                                    )?,
                                )
                                .ok_or_else(|| {
                                    Error::invalid_input("PackedRun destination overflow")
                                })?,
                        },
                    ));
                }
                let source = value.domains.domain_at(ordinal)?;
                let destination_start = value
                    .destination_start
                    .checked_add(u32::try_from(value.domains.slot_prefix(ordinal)?).map_err(
                        |_| Error::invalid_input("PackedRun destination prefix exceeds u32"),
                    )?)
                    .ok_or_else(|| Error::invalid_input("PackedRun destination overflow"))?;
                let removed_slots = removed.get(&source.logical_fragment_id).ok_or_else(|| {
                    Error::internal(format!(
                        "removed slot map lost packed-run source logical fragment {}",
                        source.logical_fragment_id
                    ))
                })?;
                if slots_cover_domain(removed_slots, source) {
                    cursor = ordinal + 1;
                    continue;
                }
                let excluded = merge_excluded_slots(source, None, None, removed_slots)?;
                if excluded.cardinality() < source.slot_count as u64 {
                    preserved.push(RowAddressPlacement::Direct(DirectRowAddressPlacement {
                        source,
                        destination_fragment_id: value.destination_fragment_id,
                        destination_start,
                        excluded: Some(excluded),
                    }));
                }
                cursor = ordinal + 1;
            }
            if cursor < value.domains.domain_count() {
                let Some(domains) = value
                    .domains
                    .compressed_slice(cursor, value.domains.domain_count())?
                else {
                    return Ok(Some(
                        PlacementMaintenanceRequired::PackedRunSubtractionRequiresRewrite {
                            logical_fragment_id: value.domains.logical_fragment_id_at(cursor)?,
                        },
                    ));
                };
                preserved.push(RowAddressPlacement::PackedRun(
                    PackedRunRowAddressPlacement {
                        domains,
                        destination_fragment_id: value.destination_fragment_id,
                        destination_start: value
                            .destination_start
                            .checked_add(
                                u32::try_from(value.domains.slot_prefix(cursor)?).map_err(
                                    |_| {
                                        Error::invalid_input(
                                            "PackedRun destination prefix exceeds u32",
                                        )
                                    },
                                )?,
                            )
                            .ok_or_else(|| {
                                Error::invalid_input("PackedRun destination overflow")
                            })?,
                    },
                ));
            }
        }
        RowAddressPlacement::Selected(value) => {
            let removed_slots =
                removed
                    .get(&value.source.logical_fragment_id)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "removed slot map lost selected source logical fragment {}",
                            value.source.logical_fragment_id
                        ))
                    })?;
            if slots_cover_domain(removed_slots, value.source) {
                return Ok(None);
            }
            let excluded = merge_excluded_slots(
                value.source,
                Some(&value.selection),
                value.excluded.as_ref(),
                removed_slots,
            )?;
            if excluded.cardinality() < value.selection.cardinality() {
                if selection_is_full_domain(&value.selection, value.source)? {
                    preserved.push(RowAddressPlacement::Direct(DirectRowAddressPlacement {
                        source: value.source,
                        destination_fragment_id: value.destination_fragment_id,
                        destination_start: value.destination_start,
                        excluded: Some(excluded),
                    }));
                } else {
                    preserved.push(RowAddressPlacement::Selected(SelectedRowAddressPlacement {
                        excluded: Some(excluded),
                        ..value.clone()
                    }));
                }
            }
        }
        RowAddressPlacement::ExtentList(value) => {
            if removed
                .get(&value.source.logical_fragment_id)
                .is_some_and(|slots| slots_cover_domain(slots, value.source))
            {
                return Ok(None);
            }
            for extent in &value.extents {
                subtract_source_range(
                    value.source,
                    extent.source_start,
                    extent.length,
                    extent.destination_fragment_id,
                    extent.destination_start,
                    removed.get(&value.source.logical_fragment_id),
                    residual,
                )?;
            }
        }
        RowAddressPlacement::SparseSelection(value) => {
            let mut prefix = 0_u64;
            let mut group_start = value.destination_start;
            let mut group = Vec::new();
            for source in &value.sources {
                let mut updated = source.clone();
                let fully_removed = removed
                    .get(&source.source.logical_fragment_id)
                    .is_some_and(|slots| slots_cover_domain(slots, source.source));
                if !fully_removed
                    && let Some(slots) = removed.get(&source.source.logical_fragment_id)
                    && selection_intersects_removed(
                        &source.selection,
                        source.excluded.as_ref(),
                        source.source,
                        slots,
                    )?
                {
                    let excluded = merge_excluded_slots(
                        source.source,
                        Some(&source.selection),
                        source.excluded.as_ref(),
                        slots,
                    )?;
                    updated.excluded = Some(excluded);
                }
                if fully_removed
                    || updated
                        .excluded
                        .as_ref()
                        .map_or(0, |excluded| excluded.cardinality())
                        == updated.selection.cardinality()
                {
                    if !group.is_empty() {
                        preserved.push(RowAddressPlacement::SparseSelection(
                            SparseSelectionRowAddressPlacement {
                                sources: std::mem::take(&mut group),
                                destination_fragment_id: value.destination_fragment_id,
                                destination_start: group_start,
                            },
                        ));
                    }
                    group_start = value
                        .destination_start
                        .checked_add(
                            u32::try_from(prefix + source.selection.cardinality()).map_err(
                                |_| {
                                    Error::invalid_input(
                                        "SparseSelection destination prefix exceeds u32",
                                    )
                                },
                            )?,
                        )
                        .ok_or_else(|| {
                            Error::invalid_input("SparseSelection destination overflow")
                        })?;
                } else {
                    group.push(updated);
                }
                prefix += source.selection.cardinality();
            }
            if !group.is_empty() {
                preserved.push(RowAddressPlacement::SparseSelection(
                    SparseSelectionRowAddressPlacement {
                        sources: group,
                        destination_fragment_id: value.destination_fragment_id,
                        destination_start: group_start,
                    },
                ));
            }
        }
        RowAddressPlacement::ExplicitMap(_) => unreachable!(),
    }
    Ok(None)
}

fn build_fast_output_placement(
    delta: &RowAddressPlacementDelta,
    target_fragment_id: u32,
    domains: &BTreeMap<u32, RowAddressLogicalDomain>,
) -> Result<RowAddressPlacement> {
    let mut previous_address = None::<u64>;
    let mut slots_by_domain = BTreeMap::<u32, RoaringBitmap>::new();
    for selection in &delta.source_selections {
        if selection.is_empty() {
            continue;
        }
        let first_address = selection
            .select(0)?
            .ok_or_else(|| Error::invalid_input("non-empty source selection has no first row"))?
            .raw();
        if previous_address.is_some_and(|previous| previous >= first_address) {
            return Err(Error::invalid_input(
                "fast-path output rows must be in strict logical source order",
            ));
        }
        let last_address = selection
            .select(selection.cardinality() - 1)?
            .ok_or_else(|| Error::invalid_input("non-empty source selection has no last row"))?
            .raw();
        if merge_selection_slots(selection, &mut slots_by_domain)?.is_some() {
            return Err(Error::invalid_input(
                "fast-path output rows must not contain duplicate logical addresses",
            ));
        }
        previous_address = Some(last_address);
    }
    if slots_by_domain.is_empty() {
        return Err(Error::invalid_input(
            "non-Direct placement output must have source selections",
        ));
    }
    let mut domain_sources = Vec::with_capacity(slots_by_domain.len());
    let mut all_domains_full = true;
    for (logical_fragment_id, slots) in &slots_by_domain {
        let source = *domains.get(logical_fragment_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "missing source-domain metadata for logical fragment {logical_fragment_id}"
            ))
        })?;
        all_domains_full &= slots.len() == u64::from(source.slot_count)
            && slots.min() == Some(0)
            && slots.max() == source.slot_count.checked_sub(1);
        domain_sources.push(source);
    }
    if domain_sources.len() == 1 && all_domains_full {
        let direct = RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: domain_sources[0],
            destination_fragment_id: target_fragment_id,
            destination_start: delta.target.start_offset,
            excluded: None,
        });
        direct.validate()?;
        return Ok(direct);
    }
    if domain_sources.len() > 1 && all_domains_full {
        let packed = RowAddressPlacement::PackedRun(PackedRunRowAddressPlacement::from_sources(
            domain_sources,
            target_fragment_id,
            delta.target.start_offset,
        )?);
        packed.validate()?;
        return Ok(packed);
    }

    let mut sources = Vec::<SparseSelectionSource>::with_capacity(slots_by_domain.len());
    for ((logical_fragment_id, slots), source) in slots_by_domain.iter().zip(domain_sources) {
        sources.push(SparseSelectionSource {
            source,
            selection: selection_from_slot_bitmap(*logical_fragment_id, slots)?,
            excluded: None,
        });
    }
    let mut candidates = Vec::<RowAddressPlacement>::new();
    if sources.len() == 1 {
        let source = &sources[0];
        candidates.push(RowAddressPlacement::Selected(SelectedRowAddressPlacement {
            source: source.source,
            selection: source.selection.clone(),
            destination_fragment_id: target_fragment_id,
            destination_start: delta.target.start_offset,
            excluded: None,
        }));
        let ranges = bitmap_ranges(&slots_by_domain[&source.source.logical_fragment_id])
            .take(ROW_ADDRESS_EXTENT_HARD_LIMIT as usize + 1)
            .collect::<Vec<_>>();
        if ranges.len() <= ROW_ADDRESS_EXTENT_HARD_LIMIT as usize {
            let mut destination_start = delta.target.start_offset;
            let mut extents = Vec::with_capacity(ranges.len());
            for (start, end) in ranges {
                let length = end - start;
                extents.push(RowAddressExtent {
                    source_start: start,
                    length,
                    destination_fragment_id: target_fragment_id,
                    destination_start,
                });
                destination_start = destination_start
                    .checked_add(length)
                    .ok_or_else(|| Error::invalid_input("output extent destination overflow"))?;
            }
            candidates.push(RowAddressPlacement::ExtentList(
                ExtentListRowAddressPlacement {
                    source: source.source,
                    extents,
                },
            ));
        }
    } else {
        candidates.push(RowAddressPlacement::SparseSelection(
            SparseSelectionRowAddressPlacement {
                sources,
                destination_fragment_id: target_fragment_id,
                destination_start: delta.target.start_offset,
            },
        ));
    }
    if candidates.len() == 1 {
        let candidate = candidates.pop().unwrap();
        candidate.validate()?;
        return Ok(candidate);
    }
    for candidate in &candidates {
        candidate.validate()?;
    }
    candidates
        .into_iter()
        .min_by_key(|candidate| {
            (
                placement_inline_bytes(candidate),
                candidate.canonical_key().1,
            )
        })
        .ok_or_else(|| Error::invalid_input("no fast-path output placement candidate"))
}

fn merge_selection_slots(
    selection: &LogicalRowAddressSelection,
    slots_by_domain: &mut BTreeMap<u32, RoaringBitmap>,
) -> Result<Option<u32>> {
    let rows = selection.to_roaring_treemap()?;
    for (logical_fragment_id, selected_slots) in rows.bitmaps() {
        let slots = slots_by_domain.entry(logical_fragment_id).or_default();
        let expected_cardinality = slots.len() + selected_slots.len();
        *slots |= selected_slots;
        if slots.len() != expected_cardinality {
            return Ok(Some(logical_fragment_id));
        }
    }
    Ok(None)
}

fn selection_without(
    selection: &LogicalRowAddressSelection,
    removed: &LogicalRowAddressSelection,
) -> Result<Option<Arc<LogicalRowAddressSelection>>> {
    let remaining = selection.difference(removed)?;
    if remaining.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Arc::new(remaining)))
    }
}

fn normalize_generation_regions(
    regions: Vec<ContentGenerationRegion>,
) -> Result<Vec<ContentGenerationRegion>> {
    let mut grouped = BTreeMap::<(u64, Vec<i32>), RoaringTreemap>::new();
    for mut region in regions {
        region.field_ids.sort_unstable();
        let bitmap = grouped
            .entry((region.generation, region.field_ids))
            .or_default();
        *bitmap |= region.selection.to_roaring_treemap()?;
    }
    grouped
        .into_iter()
        .map(|((generation, field_ids), bitmap)| {
            Ok(ContentGenerationRegion {
                selection: Arc::new(LogicalRowAddressSelection::from_bitmap(bitmap)?),
                field_ids,
                generation,
            })
        })
        .collect()
}

fn normalize_retired_rows(
    retired_rows: Vec<RetiredLogicalRowSet>,
) -> Result<Vec<RetiredLogicalRowSet>> {
    let mut domains = BTreeMap::<u32, (RowAddressLogicalDomain, Option<RoaringBitmap>)>::new();
    let mut selected_rows = RoaringTreemap::new();
    for retired in retired_rows {
        let mut set_domains = BTreeSet::new();
        for ordinal in 0..retired.domains.domain_count() {
            let source = retired.domains.domain_at(ordinal)?;
            set_domains.insert(source.logical_fragment_id);
            match domains.entry(source.logical_fragment_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((
                        source,
                        match &retired.membership {
                            RetiredLogicalRowMembership::AllRows => None,
                            RetiredLogicalRowMembership::Selection(_) => Some(RoaringBitmap::new()),
                        },
                    ));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get().0 != source {
                        return Err(Error::invalid_input(format!(
                            "retired logical domain {} has inconsistent metadata",
                            source.logical_fragment_id
                        )));
                    }
                    if matches!(&retired.membership, RetiredLogicalRowMembership::AllRows) {
                        entry.get_mut().1 = None;
                    }
                }
            }
        }
        if let RetiredLogicalRowMembership::Selection(selection) = retired.membership {
            let selection = selection.to_roaring_treemap()?;
            let selected_domains = selection
                .bitmaps()
                .map(|(logical_fragment_id, _)| logical_fragment_id)
                .collect::<BTreeSet<_>>();
            if selected_domains != set_domains {
                return Err(Error::invalid_input(
                    "selected retired set must contain at least one row from every declared domain",
                ));
            }
            for (logical_fragment_id, selected_slots) in selection.bitmaps() {
                if !set_domains.contains(&logical_fragment_id) {
                    return Err(Error::invalid_input(format!(
                        "retired logical domain {logical_fragment_id} is absent from its set's domain metadata"
                    )));
                }
                let (source, slots) = domains
                    .get_mut(&logical_fragment_id)
                    .ok_or_else(|| Error::internal("retired domain normalization lost metadata"))?;
                if selected_slots
                    .max()
                    .is_some_and(|slot| slot >= source.slot_count)
                {
                    return Err(Error::invalid_input(format!(
                        "retired logical domain {logical_fragment_id} exceeds slot_count {}",
                        source.slot_count,
                    )));
                }
                if let Some(slots) = slots {
                    *slots |= selected_slots;
                }
            }
            selected_rows |= selection;
        }
    }

    let mut fully_retired = Vec::new();
    let mut partially_retired = Vec::new();
    let mut partial_bitmap = selected_rows;
    for (_, (source, slots)) in domains {
        match slots {
            None => {
                let start = u64::from(source.logical_fragment_id) << 32;
                partial_bitmap.remove_range(start..start + u64::from(source.slot_count));
                fully_retired.push(source);
            }
            Some(slots) if slots.len() == source.slot_count as u64 => {
                let start = u64::from(source.logical_fragment_id) << 32;
                partial_bitmap.remove_range(start..start + u64::from(source.slot_count));
                fully_retired.push(source);
            }
            Some(slots) => {
                debug_assert!(!slots.is_empty());
                partially_retired.push(source);
            }
        }
    }
    let mut canonical = Vec::with_capacity(2);
    if !fully_retired.is_empty() {
        canonical.push(RetiredLogicalRowSet::all_rows(fully_retired)?);
    }
    if !partially_retired.is_empty() {
        canonical.push(RetiredLogicalRowSet::selected(
            partially_retired,
            Arc::new(LogicalRowAddressSelection::from_bitmap(partial_bitmap)?),
        )?);
    }
    Ok(canonical)
}

fn remove_retired_domains(
    retired_rows: &[RetiredLogicalRowSet],
    removed_domains: &BTreeSet<u32>,
) -> Result<Vec<RetiredLogicalRowSet>> {
    let mut retained = Vec::new();
    for retired in retired_rows {
        let sources = (0..retired.domains.domain_count())
            .map(|ordinal| retired.domains.domain_at(ordinal))
            .collect::<Result<Vec<_>>>()?;
        let retained_sources = sources
            .into_iter()
            .filter(|source| !removed_domains.contains(&source.logical_fragment_id))
            .collect::<Vec<_>>();
        if retained_sources.is_empty() {
            continue;
        }
        match &retired.membership {
            RetiredLogicalRowMembership::AllRows => {
                retained.push(RetiredLogicalRowSet::all_rows(retained_sources)?);
            }
            RetiredLogicalRowMembership::Selection(selection) => {
                let mut bitmap = selection.to_roaring_treemap()?;
                for logical_fragment_id in removed_domains {
                    let start = u64::from(*logical_fragment_id) << 32;
                    bitmap.remove_range(start..start + (1_u64 << 32));
                }
                if !bitmap.is_empty() {
                    retained.push(RetiredLogicalRowSet::selected(
                        retained_sources,
                        Arc::new(LogicalRowAddressSelection::from_bitmap(bitmap)?),
                    )?);
                }
            }
        }
    }
    normalize_retired_rows(retained)
}

fn target_fragment_id(
    target: RowAddressTargetRange,
    resolved_new_fragment_ids: &BTreeMap<u32, u32>,
) -> Result<u32> {
    match target.fragment {
        RowAddressTargetFragment::NewFragmentOrdinal(ordinal) => resolved_new_fragment_ids
            .get(&ordinal)
            .copied()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "new fragment ordinal {ordinal} has no resolved physical id"
                ))
            }),
        RowAddressTargetFragment::ExistingFragmentId(fragment_id) => Ok(fragment_id),
    }
}

fn max_extent_fanout(placements: &[RowAddressPlacement]) -> Result<(u32, Option<u32>)> {
    let mut fanout = BTreeMap::<u32, u32>::new();
    for placement in placements {
        match placement {
            RowAddressPlacement::Direct(value) => {
                *fanout.entry(value.source.logical_fragment_id).or_default() += 1;
            }
            RowAddressPlacement::PackedRun(value) => {
                for ordinal in 0..value.domains.domain_count() {
                    *fanout
                        .entry(value.domains.logical_fragment_id_at(ordinal)?)
                        .or_default() += 1;
                }
            }
            RowAddressPlacement::Selected(value) => {
                *fanout.entry(value.source.logical_fragment_id).or_default() += 1;
            }
            RowAddressPlacement::ExtentList(value) => {
                *fanout.entry(value.source.logical_fragment_id).or_default() +=
                    value.extents.len() as u32;
            }
            RowAddressPlacement::SparseSelection(value) => {
                for source in &value.sources {
                    *fanout.entry(source.source.logical_fragment_id).or_default() += 1;
                }
            }
            RowAddressPlacement::ExplicitMap(_) => {}
        }
    }
    Ok(fanout
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(domain, count)| (count, Some(domain)))
        .unwrap_or((0, None)))
}

fn logical_domain_identity_fingerprint(
    layout: &RowAddressLayout,
    fragments: &[Fragment],
) -> Option<Vec<u8>> {
    fn insert_domain(
        domains: &mut BTreeMap<u32, RowAddressLogicalDomain>,
        domain: RowAddressLogicalDomain,
    ) -> Option<()> {
        if domains
            .insert(domain.logical_fragment_id, domain)
            .is_some_and(|existing| existing != domain)
        {
            return None;
        }
        Some(())
    }

    let mut domains = BTreeMap::new();
    for placement in &layout.placements {
        match placement {
            RowAddressPlacement::Direct(value) => insert_domain(&mut domains, value.source)?,
            RowAddressPlacement::PackedRun(value) => {
                for ordinal in 0..value.domains.domain_count() {
                    insert_domain(&mut domains, value.domains.domain_at(ordinal).ok()?)?;
                }
            }
            RowAddressPlacement::Selected(value) => insert_domain(&mut domains, value.source)?,
            RowAddressPlacement::ExtentList(value) => insert_domain(&mut domains, value.source)?,
            RowAddressPlacement::SparseSelection(value) => {
                for source in &value.sources {
                    insert_domain(&mut domains, source.source)?;
                }
            }
            RowAddressPlacement::ExplicitMap(value) => {
                for source in &value.sources {
                    insert_domain(&mut domains, source.source)?;
                }
            }
        }
    }
    for fragment in fragments {
        let Some(native) = fragment.native_logical_domain else {
            continue;
        };
        let slot_count = u32::try_from(fragment.physical_rows?).ok()?;
        let domain = RowAddressLogicalDomain::new(
            native.logical_fragment_id,
            slot_count,
            native.creation_version,
        )
        .ok()?;
        insert_domain(&mut domains, domain)?;
    }
    for retired in &layout.retired_rows {
        for ordinal in 0..retired.domains.domain_count() {
            insert_domain(&mut domains, retired.domains.domain_at(ordinal).ok()?)?;
        }
    }

    let mut bytes = Vec::with_capacity(48 + domains.len() * 16);
    bytes.extend_from_slice(b"lance.logical-domain-identity.v1\0");
    bytes.extend_from_slice(&(domains.len() as u64).to_le_bytes());
    for domain in domains.into_values() {
        bytes.extend_from_slice(&domain.logical_fragment_id.to_le_bytes());
        bytes.extend_from_slice(&domain.slot_count.to_le_bytes());
        bytes.extend_from_slice(&domain.creation_version.to_le_bytes());
    }
    Some(stable_fingerprint(&bytes).to_vec())
}

fn placement_owns_any_slots(
    placement: &RowAddressPlacement,
    logical_fragment_id: u32,
    slots: &RoaringBitmap,
    source_hint: Option<usize>,
) -> Result<bool> {
    match placement {
        RowAddressPlacement::Direct(value) => {
            if value.source.logical_fragment_id != logical_fragment_id {
                return Ok(false);
            }
            let mut owned = slots.clone();
            if let Some(excluded) = value.excluded.as_deref() {
                owned -= selection_slots_for_domain(excluded, logical_fragment_id)?;
            }
            Ok(!owned.is_empty())
        }
        RowAddressPlacement::PackedRun(value) => {
            let Some(ordinal) = value.domains.domain_ordinal(logical_fragment_id)? else {
                return Ok(false);
            };
            Ok(slots.range_cardinality(0..value.domains.slot_count_at(ordinal)?) != 0)
        }
        RowAddressPlacement::Selected(value) => {
            if value.source.logical_fragment_id != logical_fragment_id {
                return Ok(false);
            }
            let mut owned = selection_slots_for_domain(&value.selection, logical_fragment_id)?;
            if let Some(excluded) = value.excluded.as_deref() {
                owned -= selection_slots_for_domain(excluded, logical_fragment_id)?;
            }
            Ok(!owned.is_disjoint(slots))
        }
        RowAddressPlacement::ExtentList(value) => Ok(value.source.logical_fragment_id
            == logical_fragment_id
            && value.extents.iter().any(|extent| {
                slots.range_cardinality(extent.source_start..extent.source_start + extent.length)
                    != 0
            })),
        RowAddressPlacement::SparseSelection(value) => {
            let sources: Box<dyn Iterator<Item = &_>> = match source_hint {
                Some(source_index) => Box::new(value.sources.get(source_index).into_iter()),
                None => Box::new(value.sources.iter()),
            };
            for source in sources {
                if source.source.logical_fragment_id != logical_fragment_id {
                    continue;
                }
                let mut owned = selection_slots_for_domain(&source.selection, logical_fragment_id)?;
                if let Some(excluded) = source.excluded.as_deref() {
                    owned -= selection_slots_for_domain(excluded, logical_fragment_id)?;
                }
                if !owned.is_disjoint(slots) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        RowAddressPlacement::ExplicitMap(_) => Ok(false),
    }
}

fn physical_fragment_is_fully_dropped(
    physical_fragment_id: u32,
    context: &RowAddressDeltaApplyContext<'_>,
) -> bool {
    context
        .newly_fully_deleted_source_fragments
        .contains(&physical_fragment_id)
        && !context
            .successor_fragments
            .iter()
            .any(|fragment| fragment.id == u64::from(physical_fragment_id))
}

fn validate_retired_physical_range(
    logical_range: LogicalRowAddressRange,
    physical_range: PhysicalRowRange,
    context: &RowAddressDeltaApplyContext<'_>,
    affected_physical_fragments: &mut BTreeSet<u32>,
) -> Result<()> {
    affected_physical_fragments.insert(physical_range.physical_fragment_id);
    if physical_fragment_is_fully_dropped(physical_range.physical_fragment_id, context) {
        return Ok(());
    }
    let mut not_deleted = RoaringBitmap::new();
    not_deleted.insert_range(physical_range.start_offset..physical_range.end_offset);
    if let Some(deleted) = context
        .current_deletion_vectors
        .get(&physical_range.physical_fragment_id)
    {
        not_deleted -= *deleted;
    }
    if let Some(deleted) = context
        .deletion_vectors
        .get(&physical_range.physical_fragment_id)
    {
        not_deleted -= *deleted;
    }
    if let Some(offset) = not_deleted.min() {
        return Err(Error::invalid_input(format!(
            "retired logical range fragment_id={}, start={}, end={} was live in source fragment {} at offset {}",
            logical_range.logical_fragment_id,
            logical_range.start_slot,
            logical_range.end_slot,
            physical_range.physical_fragment_id,
            offset,
        )));
    }
    Ok(())
}

fn validate_retired_address_batch(
    layout: &RowAddressLayout,
    router: &RowAddressRouter,
    addresses: &[LogicalRowAddress],
    explicit_replaced_domains: &BTreeSet<u32>,
    context: &RowAddressDeltaApplyContext<'_>,
    affected_physical_fragments: &mut BTreeSet<u32>,
) -> Result<()> {
    for (address, resolution) in addresses.iter().zip(router.resolve_many(addresses)?) {
        match resolution {
            PlacementResolution::Mapped {
                locator: PhysicalRowLocator::Physical(physical),
            } => {
                let logical_range = LogicalRowAddressRange::new(
                    address.logical_fragment_id(),
                    address.immutable_slot(),
                    address.immutable_slot().checked_add(1).ok_or_else(|| {
                        Error::invalid_input("retired logical address slot overflow")
                    })?,
                );
                validate_retired_physical_range(
                    logical_range,
                    PhysicalRowRange {
                        physical_fragment_id: physical.fragment_id(),
                        start_offset: physical.row_offset(),
                        end_offset: physical.row_offset().checked_add(1).ok_or_else(|| {
                            Error::invalid_input("retired physical address offset overflow")
                        })?,
                    },
                    context,
                    affected_physical_fragments,
                )?;
            }
            PlacementResolution::Mapped {
                locator:
                    PhysicalRowLocator::ExplicitMap {
                        placement_index, ..
                    },
            } => {
                // The format layer does not perform object I/O. The delete
                // writer supplies exact logical identities; bind them to a
                // transaction which removes a destination of the same root.
                let Some(RowAddressPlacement::ExplicitMap(explicit)) =
                    layout.placements.get(placement_index as usize)
                else {
                    return Err(Error::invalid_input(
                        "retired ExplicitMap address references an invalid placement",
                    ));
                };
                let mut removes_destination = false;
                for destination in &explicit.destinations {
                    if physical_fragment_is_fully_dropped(destination.physical_fragment_id, context)
                    {
                        removes_destination = true;
                        affected_physical_fragments.insert(destination.physical_fragment_id);
                    }
                }
                if !removes_destination
                    && !explicit_replaced_domains.contains(&address.logical_fragment_id())
                {
                    return Err(Error::invalid_input(format!(
                        "retired logical address {} belongs to an ExplicitMap with no deleted destination",
                        address.raw()
                    )));
                }
            }
            PlacementResolution::NotLive
                if explicit_replaced_domains.contains(&address.logical_fragment_id()) => {}
            PlacementResolution::NotLive | PlacementResolution::Unmapped => {
                return Err(Error::invalid_input(format!(
                    "retired logical address {} has no live physical source location",
                    address.raw()
                )));
            }
        }
    }
    Ok(())
}

fn validate_retired_addresses(
    visit_addresses: impl FnOnce(&mut dyn FnMut(LogicalRowAddress) -> Result<()>) -> Result<()>,
    layout: &RowAddressLayout,
    router: &RowAddressRouter,
    explicit_replaced_domains: &BTreeSet<u32>,
    context: &RowAddressDeltaApplyContext<'_>,
    affected_physical_fragments: &mut BTreeSet<u32>,
) -> Result<()> {
    let mut batch = Vec::with_capacity(4096);
    {
        let mut push = |address| {
            batch.push(address);
            if batch.len() == 4096 {
                validate_retired_address_batch(
                    layout,
                    router,
                    &batch,
                    explicit_replaced_domains,
                    context,
                    affected_physical_fragments,
                )?;
                batch.clear();
            }
            Ok(())
        };
        visit_addresses(&mut push)?;
    }
    if !batch.is_empty() {
        validate_retired_address_batch(
            layout,
            router,
            &batch,
            explicit_replaced_domains,
            context,
            affected_physical_fragments,
        )?;
    }
    Ok(())
}

fn validate_touched_addresses(
    addresses: impl Iterator<Item = Result<LogicalRowAddress>>,
    router: &RowAddressRouter,
    explicit_replaced_domains: &BTreeSet<u32>,
    affected_physical_fragments: &mut BTreeSet<u32>,
) -> Result<()> {
    let mut batch = Vec::with_capacity(4096);
    let validate_batch =
        |batch: &[LogicalRowAddress], affected: &mut BTreeSet<u32>| -> Result<()> {
            for (address, resolution) in batch.iter().zip(router.resolve_many(batch)?) {
                match resolution {
                    PlacementResolution::Mapped {
                        locator: PhysicalRowLocator::Physical(physical),
                    } => {
                        affected.insert(physical.fragment_id());
                    }
                    PlacementResolution::Mapped { .. } => {}
                    PlacementResolution::NotLive
                        if explicit_replaced_domains.contains(&address.logical_fragment_id()) => {}
                    _ => {
                        return Err(Error::invalid_input(
                            "touched logical selection contains an unmapped address",
                        ));
                    }
                }
            }
            Ok(())
        };
    for address in addresses {
        batch.push(address?);
        if batch.len() == 4096 {
            validate_batch(&batch, affected_physical_fragments)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        validate_batch(&batch, affected_physical_fragments)?;
    }
    Ok(())
}

fn record_retired_range(
    range: LogicalRowAddressRange,
    explicit_replaced_domains: &BTreeSet<u32>,
    declared_domains: &BTreeMap<u32, RowAddressLogicalDomain>,
    removed: &mut BTreeMap<u32, RoaringBitmap>,
    new_retirements: &mut BTreeMap<u32, RoaringBitmap>,
    explicit_retirement_cardinality: &mut u64,
) -> Result<()> {
    range.validate()?;
    let source = declared_domains
        .get(&range.logical_fragment_id)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "retirement references undeclared logical domain {}",
                range.logical_fragment_id
            ))
        })?;
    if range.end_slot > source.slot_count {
        return Err(Error::invalid_input(format!(
            "retired logical range fragment_id={}, start={}, end={} exceeds source slot_count {}",
            range.logical_fragment_id, range.start_slot, range.end_slot, source.slot_count,
        )));
    }
    let inserted = new_retirements
        .entry(range.logical_fragment_id)
        .or_default()
        .insert_range(range.start_slot..range.end_slot);
    if inserted != range.len() {
        return Err(Error::invalid_input(format!(
            "logical range fragment_id={}, start={}, end={} is retired more than once",
            range.logical_fragment_id, range.start_slot, range.end_slot,
        )));
    }
    if explicit_replaced_domains.contains(&range.logical_fragment_id) {
        *explicit_retirement_cardinality = explicit_retirement_cardinality
            .checked_add(range.len())
            .ok_or_else(|| Error::invalid_input("ExplicitMap retirement cardinality overflow"))?;
        return Ok(());
    }
    let inserted = removed
        .entry(range.logical_fragment_id)
        .or_default()
        .insert_range(range.start_slot..range.end_slot);
    if inserted != range.len() {
        return Err(Error::invalid_input(format!(
            "logical range fragment_id={}, start={}, end={} is both emitted and retired",
            range.logical_fragment_id, range.start_slot, range.end_slot,
        )));
    }
    Ok(())
}

impl RowAddressLayout {
    pub fn new(namespace_uuid: Uuid) -> Self {
        let mut layout = Self {
            encoding_version: ROW_ADDRESS_LAYOUT_ENCODING_VERSION,
            namespace_uuid,
            placements: Vec::new(),
            destination_index: Vec::new(),
            field_default_generations: Vec::new(),
            generation_regions: Vec::new(),
            index_commit_floors: Vec::new(),
            debt_summary: RowAddressPlacementDebtSummary::default(),
            fingerprint: Vec::new(),
            physical_row_ownership: Vec::new(),
            selection_pool: Vec::new(),
            retired_rows: Vec::new(),
            logical_domain_fingerprint: Vec::new(),
        };
        layout.refresh_fingerprint();
        layout
    }

    /// Return the canonical root used for default-fast-path admission.
    ///
    /// ExplicitMap locators, destination ownership, and the replacement
    /// retirement masks for their source domains are a disclosed maintenance
    /// tier. Retirement metadata for fast placements remains in the projection.
    pub fn fast_admission_projection(&self) -> Result<Self> {
        let explicit_domains = self
            .placements
            .iter()
            .filter_map(|placement| match placement {
                RowAddressPlacement::ExplicitMap(explicit) => Some(explicit),
                _ => None,
            })
            .flat_map(|explicit| {
                explicit
                    .sources
                    .iter()
                    .map(|source| source.source.logical_fragment_id)
            })
            .collect::<BTreeSet<_>>();
        let fast_destinations = self
            .placements
            .iter()
            .filter_map(|placement| match placement {
                RowAddressPlacement::ExplicitMap(_) => None,
                other => Some(other),
            })
            .flat_map(RowAddressPlacement::destination_ranges)
            .map(|(physical_fragment_id, _, _)| physical_fragment_id)
            .collect::<BTreeSet<_>>();
        let mut projected = self.clone();
        projected
            .placements
            .retain(|placement| !matches!(placement, RowAddressPlacement::ExplicitMap(_)));
        projected
            .physical_row_ownership
            .retain(|ownership| fast_destinations.contains(&ownership.physical_fragment_id));
        projected.retired_rows =
            remove_retired_domains(&projected.retired_rows, &explicit_domains)?;
        projected.debt_summary.explicit_layout_bytes = 0;
        projected.debt_summary.explicit_delta_bytes = 0;
        projected
            .debt_summary
            .explicit_metadata_bytes_written_since_maintenance = 0;
        projected.canonicalize();
        Ok(projected)
    }

    /// Recompute the layout-only fast/explicit byte split from canonical wire
    /// bytes.  Clone can rewrite namespace/base ownership without applying a
    /// placement delta, so these diagnostics cannot be inherited verbatim.
    pub fn refresh_layout_byte_debt(&mut self) -> Result<()> {
        let mut projected = self.clone();
        projected.fingerprint.clear();
        projected.debt_summary.canonical_layout_bytes = 0;
        projected
            .debt_summary
            .metadata_bytes_written_since_maintenance = 0;
        projected.debt_summary.explicit_layout_bytes = 0;
        projected.debt_summary.fast_delta_bytes = 0;
        projected.debt_summary.explicit_delta_bytes = 0;
        projected
            .debt_summary
            .explicit_metadata_bytes_written_since_maintenance = 0;
        projected.debt_summary.generation_delta_bytes = 0;
        projected
            .debt_summary
            .generation_metadata_bytes_written_since_maintenance = 0;
        projected.canonicalize();
        let total_layout_bytes = {
            let proto: pb::RowAddressLayout = (&projected).into();
            proto.encoded_len() as u64
        };
        let fast_layout_bytes = {
            let proto: pb::RowAddressLayout = (&projected.fast_admission_projection()?).into();
            proto.encoded_len() as u64
        };
        self.debt_summary.canonical_layout_bytes = fast_layout_bytes;
        self.debt_summary.explicit_layout_bytes = total_layout_bytes
            .checked_sub(fast_layout_bytes)
            .ok_or_else(|| Error::internal("fast row-address layout exceeds the full layout"))?;
        Ok(())
    }

    pub fn logical_domain(
        &self,
        fragments: &[Fragment],
        logical_fragment_id: u32,
    ) -> Result<Option<RowAddressLogicalDomain>> {
        let mut metadata = None;
        for placement in &self.placements {
            if let Some(source) = placement.source_domain(logical_fragment_id)? {
                if metadata.is_some_and(|current| current != source) {
                    return Err(Error::invalid_input(format!(
                        "logical domain {logical_fragment_id} has inconsistent placement metadata"
                    )));
                }
                metadata = Some(source);
            }
        }
        for retired in &self.retired_rows {
            if let Some(source) = retired.source_domain(logical_fragment_id)? {
                if metadata.is_some_and(|current| current != source) {
                    return Err(Error::invalid_input(format!(
                        "logical domain {logical_fragment_id} has inconsistent retired metadata"
                    )));
                }
                metadata = Some(source);
            }
        }
        for fragment in fragments {
            let Some(native) = fragment.native_logical_domain else {
                continue;
            };
            if native.logical_fragment_id != logical_fragment_id {
                continue;
            }
            let source = RowAddressLogicalDomain::new(
                logical_fragment_id,
                u32::try_from(fragment.physical_rows.ok_or_else(|| {
                    Error::invalid_input("native logical domain is missing physical_rows")
                })?)
                .map_err(|_| Error::invalid_input("native domain rows exceed u32"))?,
                native.creation_version,
            )?;
            if metadata.is_some_and(|current| current != source) {
                return Err(Error::invalid_input(format!(
                    "logical domain {logical_fragment_id} native and placement metadata disagree"
                )));
            }
            metadata = Some(source);
        }
        Ok(metadata)
    }

    pub fn apply_delta(
        &self,
        delta: &RowAddressLayoutDelta,
        context: &RowAddressDeltaApplyContext<'_>,
    ) -> Result<RowAddressLayoutApplyResult> {
        self.apply_delta_inner(delta, context, true)
    }

    /// Apply a delta for a commit that will refresh and validate the complete
    /// successor manifest before it is persisted.
    ///
    /// The returned admitted layout has passed all checked transformations and
    /// admission limits, but its closure fingerprints still describe the
    /// source snapshot. Callers must refresh and validate the final successor
    /// after applying any schema and metadata-debt changes.
    #[doc(hidden)]
    pub fn apply_delta_for_commit(
        &self,
        delta: &RowAddressLayoutDelta,
        context: &RowAddressDeltaApplyContext<'_>,
    ) -> Result<RowAddressLayoutApplyResult> {
        self.apply_delta_inner(delta, context, false)
    }

    fn apply_delta_inner(
        &self,
        delta: &RowAddressLayoutDelta,
        context: &RowAddressDeltaApplyContext<'_>,
        finalize_successor: bool,
    ) -> Result<RowAddressLayoutApplyResult> {
        delta.validate_admission_tier()?;
        delta.validate_row_aligned_rewrite_proofs()?;
        if context.commit_version == 0 {
            return Err(Error::invalid_input(
                "row-address delta commit_version must be non-zero",
            ));
        }
        if delta.create_namespace_uuid.is_some() {
            return Err(Error::invalid_input(
                "successor row-address delta must not carry create_namespace_uuid",
            ));
        }
        if delta.expected_layout_fingerprint != self.fingerprint {
            return Err(Error::invalid_input(
                "row-address delta expected layout fingerprint is stale",
            ));
        }
        self.validate_with_fragments(
            context.current_fragments,
            context.current_max_logical_fragment_id,
        )?;
        let mut current_fragments_by_id =
            HashMap::<u64, &Fragment>::with_capacity(context.current_fragments.len());
        for fragment in context.current_fragments {
            if current_fragments_by_id
                .insert(fragment.id, fragment)
                .is_some()
            {
                return Err(Error::invalid_input(format!(
                    "source snapshot contains duplicate physical fragment id {}",
                    fragment.id
                )));
            }
        }
        let mut successor_fragments_by_id =
            HashMap::<u64, &Fragment>::with_capacity(context.successor_fragments.len());
        for fragment in context.successor_fragments {
            if successor_fragments_by_id
                .insert(fragment.id, fragment)
                .is_some()
            {
                return Err(Error::invalid_input(format!(
                    "successor snapshot contains duplicate physical fragment id {}",
                    fragment.id
                )));
            }
        }
        validate_strictly_sorted_domains(&delta.source_domains)?;
        let declared_domains = delta
            .source_domains
            .iter()
            .map(|source| (source.logical_fragment_id, *source))
            .collect::<BTreeMap<_, _>>();
        let mut removed = BTreeMap::<u32, RoaringBitmap>::new();
        let mut explicit_replaced_domains = BTreeSet::<u32>::new();
        let mut explicit_source_cardinality = 0_u64;
        let mut explicit_output_cardinality = 0_u64;
        let mut retirement_cardinality = 0_u64;
        let mut explicit_retirement_cardinality = 0_u64;
        let mut new_retirements = BTreeMap::<u32, RoaringBitmap>::new();
        for (placement_index, placement) in delta.placements.iter().enumerate() {
            if placement.target.start_offset >= placement.target.end_offset
                || (placement.placement_kind != RowAddressPlacementKind::ExplicitMap
                    && placement.output_cardinality
                        != (placement.target.end_offset - placement.target.start_offset) as u64)
            {
                return Err(Error::invalid_input(
                    "row-address placement target and output cardinality disagree",
                ));
            }
            let source_cardinality = placement
                .source_selections
                .iter()
                .try_fold(0_u64, |total, selection| {
                    total.checked_add(selection.cardinality())
                })
                .ok_or_else(|| Error::invalid_input("placement source cardinality overflow"))?;
            if placement.placement_kind == RowAddressPlacementKind::Direct
                && placement.source_selections.is_empty()
            {
                if !placement.output_row_sequence_fingerprint.is_empty() {
                    return Err(Error::invalid_input(
                        "source-free Direct placement must not carry an output sequence digest",
                    ));
                }
            } else {
                if (placement.placement_kind != RowAddressPlacementKind::ExplicitMap
                    && source_cardinality != placement.output_cardinality)
                    || source_cardinality == 0
                    || (placement.placement_kind == RowAddressPlacementKind::ExplicitMap
                        && placement.output_cardinality > source_cardinality)
                {
                    return Err(Error::invalid_input(
                        "placement source and output cardinalities disagree",
                    ));
                }
                if placement.placement_kind == RowAddressPlacementKind::ExplicitMap {
                    if placement.output_row_sequence_fingerprint.len()
                        != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                    {
                        return Err(Error::invalid_input(
                            "ExplicitMap output sequence digest must be 16 bytes",
                        ));
                    }
                    if !context
                        .explicit_map_placements
                        .contains_key(&placement_index)
                    {
                        return Ok(RowAddressLayoutApplyResult::NotAdmitted {
                            reason: PlacementMaintenanceRequired::ExplicitMapMetadataRequired {
                                placement_delta_index: placement_index,
                            },
                            metrics: RowAddressAdmissionMetrics {
                                projected_layout_bytes: self.debt_summary.canonical_layout_bytes,
                                max_extent_fanout: self
                                    .debt_summary
                                    .max_extents_per_logical_fragment,
                            },
                        });
                    }
                    for selection in &placement.source_selections {
                        let domains = selection.logical_fragment_bitmap()?;
                        explicit_replaced_domains.extend(domains.iter());
                    }
                    explicit_source_cardinality = explicit_source_cardinality
                        .checked_add(source_cardinality)
                        .ok_or_else(|| {
                            Error::invalid_input("ExplicitMap source cardinality overflow")
                        })?;
                    explicit_output_cardinality = explicit_output_cardinality
                        .checked_add(placement.output_cardinality)
                        .ok_or_else(|| {
                            Error::invalid_input("ExplicitMap output cardinality overflow")
                        })?;
                } else {
                    placement.verify_output_row_sequence()?;
                }
                for selection in &placement.source_selections {
                    if let Some(logical_fragment_id) =
                        merge_selection_slots(selection, &mut removed)?
                    {
                        return Err(Error::invalid_input(format!(
                            "logical fragment {logical_fragment_id} overlaps another placement output"
                        )));
                    }
                }
            }
            let target_fragment_id =
                target_fragment_id(placement.target, context.resolved_new_fragment_ids)?;
            let target_fragment = successor_fragments_by_id
                .get(&u64::from(target_fragment_id))
                .copied()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "placement target fragment {target_fragment_id} is absent from successor"
                    ))
                })?;
            let physical_rows = target_fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input("placement target fragment is missing physical_rows")
            })? as u64;
            if placement.target.end_offset as u64 > physical_rows {
                return Err(Error::invalid_input(
                    "placement target range exceeds successor fragment rows",
                ));
            }
        }
        for selection in &delta.retired_selections {
            if selection.is_empty() {
                return Err(Error::invalid_input(
                    "retired logical row-address selection must not be empty",
                ));
            }
            retirement_cardinality = retirement_cardinality
                .checked_add(selection.cardinality())
                .ok_or_else(|| Error::invalid_input("retirement cardinality overflow"))?;
            if let Some(ranges) = selection.compact_ranges(MAX_SELECTION_DOMAINS)? {
                for range in ranges {
                    record_retired_range(
                        range,
                        &explicit_replaced_domains,
                        &declared_domains,
                        &mut removed,
                        &mut new_retirements,
                        &mut explicit_retirement_cardinality,
                    )?;
                }
            } else {
                visit_selection_value_ranges(selection, |range| {
                    record_retired_range(
                        range,
                        &explicit_replaced_domains,
                        &declared_domains,
                        &mut removed,
                        &mut new_retirements,
                        &mut explicit_retirement_cardinality,
                    )
                })?;
            }
        }
        if !explicit_replaced_domains.is_empty() {
            let expected_retirements = explicit_source_cardinality
                .checked_sub(explicit_output_cardinality)
                .ok_or_else(|| {
                    Error::invalid_input("ExplicitMap output exceeds its source cardinality")
                })?;
            if explicit_retirement_cardinality != expected_retirements
                || retirement_cardinality != expected_retirements
            {
                return Err(Error::invalid_input(format!(
                    "ExplicitMap replacement retirements cover {} rows, expected {expected_retirements}",
                    explicit_retirement_cardinality
                )));
            }
        }
        if removed.keys().copied().collect::<BTreeSet<_>>()
            != declared_domains.keys().copied().collect::<BTreeSet<_>>()
        {
            return Err(Error::invalid_input(
                "row-address delta source_domains must exactly cover touched logical domains",
            ));
        }
        let router = RowAddressRouter::from_validated_layout(
            Arc::new(self.clone()),
            context.current_fragments,
        )?;
        for source in delta.source_domains.iter().copied() {
            let current = router
                .logical_domain(source.logical_fragment_id)?
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "touched logical domain {} is absent from the source snapshot",
                        source.logical_fragment_id
                    ))
                })?;
            if current != source
                || removed[&source.logical_fragment_id]
                    .max()
                    .is_some_and(|slot| slot >= source.slot_count)
            {
                return Err(Error::invalid_input(format!(
                    "source-domain metadata or touched slot disagrees for logical fragment {}",
                    source.logical_fragment_id
                )));
            }
        }
        if !removed.is_empty() {
            let overlap_coverage = RoaringTreemap::from_bitmaps(
                removed
                    .iter()
                    .filter(|(logical_fragment_id, _)| {
                        !explicit_replaced_domains.contains(logical_fragment_id)
                    })
                    .map(|(logical_fragment_id, slots)| (*logical_fragment_id, slots.clone())),
            );
            let overlap = self.retired_logical_row_bitmap_for_coverage(&overlap_coverage)?;
            if let Some(raw) = overlap.min() {
                let address = LogicalRowAddress::try_from(raw)?;
                return Err(Error::invalid_input(format!(
                    "touched logical domain {} contains rows which were not live in the source snapshot",
                    address.logical_fragment_id()
                )));
            }
        }
        let mut affected_physical_fragments = BTreeSet::<u32>::new();
        for selection in &delta.retired_selections {
            if let Some(ranges) = selection.compact_ranges(MAX_SELECTION_DOMAINS)? {
                for range in ranges {
                    if let Some(proof) = router.logical_range_ownership(range)?
                        && proof.mapped_rows == range.len()
                    {
                        for physical_range in proof.physical_ranges {
                            validate_retired_physical_range(
                                range,
                                physical_range,
                                context,
                                &mut affected_physical_fragments,
                            )?;
                        }
                    } else {
                        validate_retired_addresses(
                            |visit| {
                                for slot in range.start_slot..range.end_slot {
                                    visit(LogicalRowAddress::try_new_from_parts(
                                        range.logical_fragment_id,
                                        slot,
                                    )?)?;
                                }
                                Ok(())
                            },
                            self,
                            &router,
                            &explicit_replaced_domains,
                            context,
                            &mut affected_physical_fragments,
                        )?;
                    }
                }
            } else {
                validate_retired_addresses(
                    |visit| selection.try_for_each_address(visit),
                    self,
                    &router,
                    &explicit_replaced_domains,
                    context,
                    &mut affected_physical_fragments,
                )?;
            }
        }
        for (logical_fragment_id, slots) in &removed {
            // Retirement ownership was proved above against the source
            // snapshot, including the case where its last physical fragment
            // is being dropped.  Only emitted rows need a current owner for
            // placement subtraction; requiring one for new retirements would
            // reject a valid Repack of an already non-live row.
            let mut emitted_slots = slots.clone();
            if let Some(retired_slots) = new_retirements.get(logical_fragment_id) {
                emitted_slots -= retired_slots;
            }
            if emitted_slots.is_empty() {
                continue;
            }
            if let Some(fragments) =
                router.logical_selection_inline_ownership(*logical_fragment_id, &emitted_slots)?
            {
                affected_physical_fragments.extend(fragments.iter());
            } else {
                validate_touched_addresses(
                    emitted_slots.iter().map(|slot| {
                        LogicalRowAddress::try_new_from_parts(*logical_fragment_id, slot)
                    }),
                    &router,
                    &explicit_replaced_domains,
                    &mut affected_physical_fragments,
                )?;
            }
        }
        for (placement_index, placement) in delta.placements.iter().enumerate() {
            if placement.placement_kind == RowAddressPlacementKind::ExplicitMap {
                affected_physical_fragments.extend(
                    context.explicit_map_placements[&placement_index]
                        .destinations
                        .iter()
                        .map(|destination| destination.physical_fragment_id),
                );
            } else {
                affected_physical_fragments.insert(target_fragment_id(
                    placement.target,
                    context.resolved_new_fragment_ids,
                )?);
            }
        }

        let mut candidate = self.clone();
        let mut preserved = Vec::new();
        let mut residual = Vec::new();
        for placement in &self.placements {
            if let Some(reason) =
                subtract_placement(placement, &removed, &mut preserved, &mut residual)?
            {
                return Ok(RowAddressLayoutApplyResult::NotAdmitted {
                    reason,
                    metrics: RowAddressAdmissionMetrics {
                        projected_layout_bytes: self.debt_summary.canonical_layout_bytes,
                        max_extent_fanout: self.debt_summary.max_extents_per_logical_fragment,
                    },
                });
            }
        }
        for fragment in context.current_fragments {
            let Some(native) = fragment.native_logical_domain else {
                continue;
            };
            let Some(removed_slots) = removed.get(&native.logical_fragment_id) else {
                continue;
            };
            let successor = successor_fragments_by_id.get(&fragment.id).copied();
            if successor.is_some_and(|successor| successor.native_logical_domain.is_some()) {
                return Err(Error::invalid_input(format!(
                    "updated native logical domain {} must clear its native marker in the successor",
                    native.logical_fragment_id
                )));
            }
            let source = declared_domains[&native.logical_fragment_id];
            if removed_slots.len() < source.slot_count as u64 {
                if successor.is_none() {
                    return Err(Error::invalid_input(
                        "successor drops a native source fragment that still owns untouched rows",
                    ));
                }
                preserved.push(RowAddressPlacement::Direct(DirectRowAddressPlacement {
                    source,
                    destination_fragment_id: physical_fragment_id(
                        fragment.id,
                        "native source fragment",
                    )?,
                    destination_start: 0,
                    excluded: Some(selection_from_slot_bitmap(
                        source.logical_fragment_id,
                        removed_slots,
                    )?),
                }));
            }
        }
        preserved.extend(residual_placements_from_extents(residual)?);
        for (placement_index, placement) in delta.placements.iter().enumerate() {
            if placement.placement_kind == RowAddressPlacementKind::Direct
                && placement.source_selections.is_empty()
            {
                let fragment_id =
                    target_fragment_id(placement.target, context.resolved_new_fragment_ids)?;
                let fragment = successor_fragments_by_id
                    .get(&u64::from(fragment_id))
                    .copied()
                    .expect("target fragment was validated above");
                if fragment.native_logical_domain.is_none() {
                    return Err(Error::invalid_input(
                        "source-free Direct output requires a successor native logical domain",
                    ));
                }
                continue;
            }
            let fragment_id =
                target_fragment_id(placement.target, context.resolved_new_fragment_ids)?;
            if placement.placement_kind == RowAddressPlacementKind::ExplicitMap {
                let explicit = context.explicit_map_placements[&placement_index].clone();
                let explicit_placement = RowAddressPlacement::ExplicitMap(explicit);
                explicit_placement.validate()?;
                let ranges = explicit_placement.destination_ranges();
                let mapped_rows = ranges.iter().try_fold(0_u64, |total, (_, _, rows)| {
                    total.checked_add(*rows).ok_or_else(|| {
                        Error::invalid_input("ExplicitMap destination cardinality overflow")
                    })
                })?;
                if mapped_rows != placement.output_cardinality
                    || !ranges.iter().any(|(destination, start, _)| {
                        *destination == fragment_id && *start == placement.target.start_offset
                    })
                    || ranges.iter().any(|(destination, start, rows)| {
                        successor_fragments_by_id
                            .get(&u64::from(*destination))
                            .and_then(|fragment| fragment.physical_rows)
                            .is_none_or(|physical_rows| {
                                *start as u64 + *rows > physical_rows as u64
                            })
                    })
                {
                    return Err(Error::invalid_input(
                        "ExplicitMap metadata does not match its declared output fragments",
                    ));
                }
                preserved.push(explicit_placement);
            } else {
                preserved.push(build_fast_output_placement(
                    placement,
                    fragment_id,
                    &declared_domains,
                )?);
            }
        }
        if !explicit_replaced_domains.is_empty() {
            candidate.retired_rows =
                remove_retired_domains(&candidate.retired_rows, &explicit_replaced_domains)?;
        }
        if !delta.retired_selections.is_empty() {
            for selection in &delta.retired_selections {
                let retired_domains = selection
                    .logical_fragment_bitmap()?
                    .iter()
                    .map(|logical_fragment_id| declared_domains[&logical_fragment_id])
                    .collect::<Vec<_>>();
                candidate.retired_rows.push(RetiredLogicalRowSet::selected(
                    retired_domains,
                    Arc::new(selection.clone()),
                )?);
            }
            candidate.retired_rows = normalize_retired_rows(candidate.retired_rows)?;
        }
        preserved.retain_mut(|placement| {
            let RowAddressPlacement::ExplicitMap(explicit) = placement else {
                return true;
            };
            explicit.destinations.retain(|destination| {
                successor_fragments_by_id.contains_key(&u64::from(destination.physical_fragment_id))
            });
            !explicit.destinations.is_empty()
        });
        candidate.placements = preserved;
        candidate.apply_generation_delta(delta, context.commit_version)?;
        let previous_ownership = self
            .physical_row_ownership
            .iter()
            .map(|summary| (summary.physical_fragment_id, summary))
            .collect::<BTreeMap<_, _>>();
        candidate.physical_row_ownership.clear();
        candidate.canonicalize();
        let mut successor_target_fragments = context
            .successor_fragments
            .iter()
            .filter(|fragment| fragment.native_logical_domain.is_none())
            .collect::<Vec<_>>();
        successor_target_fragments.sort_unstable_by_key(|fragment| fragment.id);
        for fragment in successor_target_fragments {
            let fragment_id = physical_fragment_id(fragment.id, "target-only fragment")?;
            let unchanged_target_only = !affected_physical_fragments.contains(&fragment_id)
                && current_fragments_by_id
                    .get(&fragment.id)
                    .is_some_and(|current| {
                        current.native_logical_domain.is_none()
                            && current.physical_rows == fragment.physical_rows
                            && current.deletion_file == fragment.deletion_file
                    });
            if unchanged_target_only {
                let summary = previous_ownership.get(&fragment_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "unchanged target-only fragment {fragment_id} is missing its ownership summary"
                    ))
                })?;
                candidate.set_physical_ownership_summary((*summary).clone());
                continue;
            }
            if fragment.deletion_file.is_some() {
                let deleted_offsets = context.deletion_vectors.get(&fragment_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "target-only fragment {fragment_id} is missing its deletion vector input"
                    ))
                })?;
                candidate
                    .refresh_physical_ownership_summary_from_index(fragment, deleted_offsets)?;
            } else {
                candidate.refresh_physical_ownership_summary_from_index(
                    fragment,
                    &RoaringBitmap::new(),
                )?;
            }
        }
        let (max_extent_fanout, max_extent_domain) = max_extent_fanout(&candidate.placements)?;
        candidate.refresh_layout_byte_debt()?;
        let fast_layout_bytes = candidate.debt_summary.canonical_layout_bytes;
        let metrics = RowAddressAdmissionMetrics {
            projected_layout_bytes: fast_layout_bytes,
            max_extent_fanout,
        };
        if max_extent_fanout > ROW_ADDRESS_EXTENT_HARD_LIMIT {
            return Ok(RowAddressLayoutApplyResult::NotAdmitted {
                reason: PlacementMaintenanceRequired::ExtentFanout {
                    logical_fragment_id: max_extent_domain.unwrap_or(INVALID_ID),
                    projected: max_extent_fanout,
                    limit: ROW_ADDRESS_EXTENT_HARD_LIMIT,
                },
                metrics,
            });
        }
        let total_physical_rows = context
            .successor_fragments
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap_or_default() as u64)
            .sum::<u64>();
        let deleted_rows = context
            .successor_fragments
            .iter()
            .filter_map(|fragment| fragment.deletion_file.as_ref())
            .map(|deletion_file| deletion_file.num_deleted_rows.unwrap_or_default() as u64)
            .sum::<u64>();
        candidate
            .debt_summary
            .metadata_bytes_written_since_maintenance = candidate
            .debt_summary
            .metadata_bytes_written_since_maintenance
            .checked_add(context.row_address_metadata_bytes_written)
            .ok_or_else(|| Error::invalid_input("placement metadata write debt overflow"))?;
        candidate.debt_summary.max_extents_per_logical_fragment = max_extent_fanout;
        candidate.debt_summary.total_physical_rows = total_physical_rows;
        candidate.debt_summary.live_physical_rows =
            total_physical_rows
                .checked_sub(deleted_rows)
                .ok_or_else(|| Error::invalid_input("deleted rows exceed physical rows"))?;
        if finalize_successor {
            candidate.refresh_fingerprint_with_fragments(
                context.successor_fragments,
                context.max_logical_fragment_id,
            );
            candidate.validate_with_fragments(
                context.successor_fragments,
                context.max_logical_fragment_id,
            )?;
        }
        Ok(RowAddressLayoutApplyResult::Admitted {
            layout: Box::new(candidate),
            metrics,
        })
    }

    fn apply_generation_delta(
        &mut self,
        delta: &RowAddressLayoutDelta,
        commit_version: u64,
    ) -> Result<()> {
        let current_floors = self
            .index_commit_floors
            .iter()
            .map(|floor| (floor.field_id, floor.generation))
            .collect::<BTreeMap<_, _>>();
        for source_floor in &delta.source_floors {
            if current_floors.get(&source_floor.field_id) != Some(&source_floor.generation) {
                return Err(Error::invalid_input(format!(
                    "source floor for field {} is stale",
                    source_floor.field_id
                )));
            }
        }
        let mut floors = current_floors;
        let mut regions = self.generation_regions.clone();
        for replaced in &delta.replaced_generations {
            let encoded = encoded_selection_bytes(&replaced.selection);
            let index = regions
                .iter()
                .position(|region| {
                    region.generation == replaced.generation
                        && region.field_ids == replaced.field_ids
                        && encoded_selection_bytes(&region.selection) == encoded
                })
                .ok_or_else(|| {
                    Error::invalid_input(
                        "replaced content generation is absent from the current frontier",
                    )
                })?;
            regions.remove(index);
            for field_id in &replaced.field_ids {
                floors
                    .entry(*field_id)
                    .and_modify(|floor| *floor = (*floor).max(replaced.generation))
                    .or_insert(replaced.generation);
            }
        }
        for change in &delta.field_changes {
            validate_field_ids(&change.field_ids, "RowAddressFieldChange.field_ids")?;
            let changed_fields = change.field_ids.iter().copied().collect::<BTreeSet<_>>();
            let mut next_regions = Vec::new();
            for region in regions {
                let affected = region
                    .field_ids
                    .iter()
                    .copied()
                    .filter(|field_id| changed_fields.contains(field_id))
                    .collect::<Vec<_>>();
                if affected.is_empty() {
                    next_regions.push(region);
                    continue;
                }
                let unaffected = region
                    .field_ids
                    .iter()
                    .copied()
                    .filter(|field_id| !changed_fields.contains(field_id))
                    .collect::<Vec<_>>();
                if !unaffected.is_empty() {
                    next_regions.push(ContentGenerationRegion {
                        selection: region.selection.clone(),
                        field_ids: unaffected,
                        generation: region.generation,
                    });
                }
                if let Some(remaining) = selection_without(&region.selection, &change.selection)? {
                    next_regions.push(ContentGenerationRegion {
                        selection: remaining,
                        field_ids: affected,
                        generation: region.generation,
                    });
                }
            }
            next_regions.push(ContentGenerationRegion {
                selection: Arc::new(change.selection.clone()),
                field_ids: change.field_ids.clone(),
                generation: commit_version,
            });
            regions = next_regions;
        }
        self.generation_regions = normalize_generation_regions(regions)?;
        self.index_commit_floors = floors
            .into_iter()
            .map(|(field_id, generation)| FieldGeneration {
                field_id,
                generation,
            })
            .collect();
        Ok(())
    }

    pub fn refresh_fingerprint(&mut self) {
        self.canonicalize();
        // A malformed in-memory closure leaves the required proof empty so
        // manifest validation rejects it before commit.
        self.logical_domain_fingerprint =
            logical_domain_identity_fingerprint(self, &[]).unwrap_or_default();
        self.fingerprint = self.calculate_fingerprint_with_fragments(&[], None);
    }

    /// Merge generation regions that have identical generations and field sets.
    pub fn normalize_generation_frontier(&mut self) -> Result<()> {
        self.generation_regions =
            normalize_generation_regions(std::mem::take(&mut self.generation_regions))?;
        Ok(())
    }

    pub fn refresh_fingerprint_with_fragments(
        &mut self,
        fragments: &[Fragment],
        max_logical_fragment_id: Option<u32>,
    ) {
        self.canonicalize();
        // A malformed in-memory closure leaves the required proof empty so
        // manifest validation rejects it before commit.
        self.logical_domain_fingerprint =
            logical_domain_identity_fingerprint(self, fragments).unwrap_or_default();
        self.fingerprint =
            self.calculate_fingerprint_with_fragments(fragments, max_logical_fragment_id);
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_structure()?;
        if self.fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE {
            return Err(Error::invalid_input(format!(
                "RowAddressLayout fingerprint must be {} bytes, got {}",
                ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE,
                self.fingerprint.len()
            )));
        }
        if self.logical_domain_fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE {
            return Err(Error::invalid_input(format!(
                "RowAddressLayout logical-domain fingerprint must be {} bytes, got {}",
                ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE,
                self.logical_domain_fingerprint.len()
            )));
        }
        Ok(())
    }

    fn calculate_fingerprint_with_fragments(
        &self,
        fragments: &[Fragment],
        max_logical_fragment_id: Option<u32>,
    ) -> Vec<u8> {
        let mut canonical = self.clone();
        canonical.fingerprint.clear();
        canonical.canonicalize();
        let proto: pb::RowAddressLayout = (&canonical).into();
        let mut closure = proto.encode_to_vec();
        let mut fragment_tuples = fragments
            .iter()
            .map(|fragment| {
                (
                    fragment.id,
                    fragment.physical_rows.unwrap_or_default() as u64,
                    fragment.native_logical_domain,
                    fragment.deletion_file.clone(),
                )
            })
            .collect::<Vec<_>>();
        fragment_tuples.sort_unstable_by_key(|tuple| tuple.0);
        closure.extend_from_slice(&(fragment_tuples.len() as u64).to_le_bytes());
        for (physical_fragment_id, physical_rows, native, deletion_file) in fragment_tuples {
            closure.extend_from_slice(&physical_fragment_id.to_le_bytes());
            closure.extend_from_slice(&physical_rows.to_le_bytes());
            match native {
                Some(native) => {
                    closure.push(1);
                    closure.extend_from_slice(&native.logical_fragment_id.to_le_bytes());
                    closure.extend_from_slice(&native.creation_version.to_le_bytes());
                }
                None => closure.push(0),
            }
            match deletion_file {
                Some(deletion_file) => {
                    closure.push(1);
                    closure.extend_from_slice(&deletion_file.read_version.to_le_bytes());
                    closure.extend_from_slice(&deletion_file.id.to_le_bytes());
                    closure.push(match deletion_file.file_type {
                        super::DeletionFileType::Array => 0,
                        super::DeletionFileType::Bitmap => 1,
                    });
                    match deletion_file.num_deleted_rows {
                        Some(count) => {
                            closure.push(1);
                            closure.extend_from_slice(&(count as u64).to_le_bytes());
                        }
                        None => closure.push(0),
                    }
                    match deletion_file.base_id {
                        Some(base_id) => {
                            closure.push(1);
                            closure.extend_from_slice(&base_id.to_le_bytes());
                        }
                        None => closure.push(0),
                    }
                }
                None => closure.push(0),
            }
        }
        match max_logical_fragment_id {
            Some(maximum) => {
                closure.push(1);
                closure.extend_from_slice(&maximum.to_le_bytes());
            }
            None => closure.push(0),
        }
        stable_fingerprint(&closure).to_vec()
    }

    fn canonicalize(&mut self) {
        for placement in &mut self.placements {
            placement.canonicalize();
        }
        let mut indexed = self
            .placements
            .drain(..)
            .enumerate()
            .map(|(index, placement)| {
                let key = placement.canonical_key();
                (index, key, placement)
            })
            .collect::<Vec<_>>();
        indexed.sort_by(|left, right| left.1.cmp(&right.1));
        self.placements = indexed
            .into_iter()
            .map(|(_, _, placement)| placement)
            .collect();
        self.destination_index = build_destination_index(&self.placements);
        self.field_default_generations.sort_unstable();
        self.index_commit_floors.sort_unstable();
        self.physical_row_ownership
            .sort_by_key(|summary| summary.physical_fragment_id);
        self.retired_rows.sort_by_key(|retired| {
            (
                match &retired.membership {
                    RetiredLogicalRowMembership::AllRows => 0_u8,
                    RetiredLogicalRowMembership::Selection(_) => 1,
                },
                retired.domains.first_logical_fragment_id(),
            )
        });
        for region in &mut self.generation_regions {
            region.field_ids.sort_unstable();
        }
        self.generation_regions.sort_by(|left, right| {
            selection_first_address(&left.selection)
                .cmp(&selection_first_address(&right.selection))
                .then_with(|| left.field_ids.cmp(&right.field_ids))
                .then_with(|| left.generation.cmp(&right.generation))
        });
        self.intern_selection_pool();
    }

    fn intern_selection_pool(&mut self) {
        fn selection_key(
            selection: &LogicalRowAddressSelection,
            encoded_by_pointer: &mut HashMap<usize, Arc<[u8]>>,
        ) -> Arc<[u8]> {
            let pointer = std::ptr::from_ref(selection) as usize;
            encoded_by_pointer
                .entry(pointer)
                .or_insert_with(|| Arc::from(encoded_selection_bytes(selection)))
                .clone()
        }

        let mut encoded_by_pointer = HashMap::<usize, Arc<[u8]>>::new();
        let mut selections = BTreeMap::<Arc<[u8]>, Arc<LogicalRowAddressSelection>>::new();
        for placement in &self.placements {
            for selection in placement_selections(placement) {
                selections
                    .entry(selection_key(selection, &mut encoded_by_pointer))
                    .or_insert_with(|| Arc::new(selection.clone()));
            }
        }
        for region in &self.generation_regions {
            selections
                .entry(selection_key(&region.selection, &mut encoded_by_pointer))
                .or_insert_with(|| region.selection.clone());
        }
        for retired in &self.retired_rows {
            if let RetiredLogicalRowMembership::Selection(selection) = &retired.membership {
                selections
                    .entry(selection_key(selection, &mut encoded_by_pointer))
                    .or_insert_with(|| selection.clone());
            }
        }
        for placement in &mut self.placements {
            match placement {
                RowAddressPlacement::Direct(value) => {
                    if let Some(excluded) = &mut value.excluded {
                        let key = selection_key(excluded, &mut encoded_by_pointer);
                        *excluded = selections[&key].clone();
                    }
                }
                RowAddressPlacement::Selected(value) => {
                    let key = selection_key(&value.selection, &mut encoded_by_pointer);
                    value.selection = selections[&key].clone();
                    if let Some(excluded) = &mut value.excluded {
                        let key = selection_key(excluded, &mut encoded_by_pointer);
                        *excluded = selections[&key].clone();
                    }
                }
                RowAddressPlacement::SparseSelection(value) => {
                    for source in &mut value.sources {
                        let key = selection_key(&source.selection, &mut encoded_by_pointer);
                        source.selection = selections[&key].clone();
                        if let Some(excluded) = &mut source.excluded {
                            let key = selection_key(excluded, &mut encoded_by_pointer);
                            *excluded = selections[&key].clone();
                        }
                    }
                }
                RowAddressPlacement::ExplicitMap(value) => {
                    for source in &mut value.sources {
                        let key = selection_key(&source.selection, &mut encoded_by_pointer);
                        source.selection = selections[&key].clone();
                        if let Some(excluded) = &mut source.excluded {
                            let key = selection_key(excluded, &mut encoded_by_pointer);
                            *excluded = selections[&key].clone();
                        }
                    }
                }
                _ => {}
            }
        }
        for region in &mut self.generation_regions {
            let key = selection_key(&region.selection, &mut encoded_by_pointer);
            region.selection = selections[&key].clone();
        }
        for retired in &mut self.retired_rows {
            if let RetiredLogicalRowMembership::Selection(selection) = &mut retired.membership {
                let key = selection_key(selection, &mut encoded_by_pointer);
                *selection = selections[&key].clone();
            }
        }
        self.selection_pool = selections.into_values().collect();
    }

    fn validate_structure(&self) -> Result<()> {
        if self.encoding_version != ROW_ADDRESS_LAYOUT_ENCODING_VERSION {
            return Err(Error::invalid_input(format!(
                "unsupported RowAddressLayout encoding_version: {}",
                self.encoding_version
            )));
        }
        if self.namespace_uuid.is_nil() {
            return Err(Error::invalid_input(
                "RowAddressLayout namespace_uuid must not be nil",
            ));
        }
        for placement in &self.placements {
            placement.validate()?;
        }
        let pool_bytes = self
            .selection_pool
            .iter()
            .map(|selection| encoded_selection_bytes(selection))
            .collect::<Vec<_>>();
        if pool_bytes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::invalid_input(
                "RowAddressLayout selection pool must be strictly byte-sorted",
            ));
        }
        let pool_by_pointer = self
            .selection_pool
            .iter()
            .enumerate()
            .map(|(index, selection)| (std::ptr::from_ref(selection.as_ref()) as usize, index))
            .collect::<HashMap<_, _>>();
        let mut referenced = BTreeSet::new();
        for selection in self
            .placements
            .iter()
            .flat_map(|placement| placement_selections(placement).into_iter())
            .chain(
                self.generation_regions
                    .iter()
                    .map(|region| region.selection.as_ref()),
            )
            .chain(self.retired_rows.iter().filter_map(|retired| {
                if let RetiredLogicalRowMembership::Selection(selection) = &retired.membership {
                    Some(selection.as_ref())
                } else {
                    None
                }
            }))
        {
            let index = *pool_by_pointer
                .get(&(std::ptr::from_ref(selection) as usize))
                .ok_or_else(|| {
                    Error::invalid_input(
                        "layout selection references must share an interned pool allocation",
                    )
                })?;
            referenced.insert(index);
        }
        if referenced.len() != self.selection_pool.len() {
            return Err(Error::invalid_input(
                "RowAddressLayout selection pool contains unreferenced entries",
            ));
        }
        validate_field_generations(&self.field_default_generations, "field_default_generations")?;
        validate_field_generations(&self.index_commit_floors, "index_commit_floors")?;
        for region in &self.generation_regions {
            validate_field_ids(&region.field_ids, "ContentGenerationRegion.field_ids")?;
            if region.selection.is_empty() || region.generation == 0 {
                return Err(Error::invalid_input(
                    "generation regions require a non-empty selection and non-zero generation",
                ));
            }
        }
        for retired in &self.retired_rows {
            retired.validate()?;
        }
        if normalize_retired_rows(self.retired_rows.clone())? != self.retired_rows {
            return Err(Error::invalid_input(
                "retired logical rows are not in canonical full/partial form",
            ));
        }
        validate_logical_source_ownership(&self.placements, &[])?;
        validate_destination_ownership(&self.placements, &[])?;
        let expected_destination_index = build_destination_index(&self.placements);
        if self.destination_index != expected_destination_index {
            return Err(Error::invalid_input(
                "RowAddressLayout destination index does not match placement destinations",
            ));
        }
        if self.debt_summary.live_physical_rows > self.debt_summary.total_physical_rows {
            return Err(Error::invalid_input(
                "RowAddressLayout debt summary has more live rows than total rows",
            ));
        }
        Ok(())
    }

    pub fn validate_with_fragments(
        &self,
        fragments: &[Fragment],
        max_logical_fragment_id: Option<u32>,
    ) -> Result<()> {
        self.validate()?;
        self.validate_fragment_closure(fragments)?;
        let expected_domains =
            logical_domain_identity_fingerprint(self, fragments).ok_or_else(|| {
                Error::invalid_input(
                    "RowAddressLayout cannot derive a canonical logical-domain identity",
                )
            })?;
        if self.logical_domain_fingerprint != expected_domains {
            return Err(Error::invalid_input(
                "RowAddressLayout logical-domain fingerprint does not match its domains",
            ));
        }
        let expected =
            self.calculate_fingerprint_with_fragments(fragments, max_logical_fragment_id);
        if self.fingerprint != expected {
            return Err(Error::invalid_input(
                "RowAddressLayout fingerprint does not match its manifest closure",
            ));
        }
        Ok(())
    }

    pub fn validate_schema_fields(&self, valid_field_ids: &[i32]) -> Result<()> {
        let valid = valid_field_ids.iter().copied().collect::<BTreeSet<_>>();
        let defaults = self
            .field_default_generations
            .iter()
            .map(|generation| generation.field_id)
            .collect::<BTreeSet<_>>();
        let floors = self
            .index_commit_floors
            .iter()
            .map(|generation| generation.field_id)
            .collect::<BTreeSet<_>>();
        if defaults != valid || floors != valid {
            return Err(Error::invalid_input(
                "field_default_generations and index_commit_floors must exactly cover current schema field ids",
            ));
        }
        if self
            .generation_regions
            .iter()
            .flat_map(|region| region.field_ids.iter())
            .any(|field_id| !valid.contains(field_id))
        {
            return Err(Error::invalid_input(
                "row-address generation metadata references a dropped schema field id",
            ));
        }
        Ok(())
    }

    fn validate_fragment_closure(&self, fragments: &[Fragment]) -> Result<()> {
        validate_native_domains(fragments, &self.placements)?;
        validate_native_logical_ownership(&self.placements, fragments)?;
        validate_native_destination_ownership(&self.placements, fragments)?;
        let mut placements_by_domain = BTreeMap::<u32, Vec<(usize, Option<usize>)>>::new();
        for (placement_index, placement) in self.placements.iter().enumerate() {
            if let RowAddressPlacement::SparseSelection(value) = placement {
                for (source_index, source) in value.sources.iter().enumerate() {
                    placements_by_domain
                        .entry(source.source.logical_fragment_id)
                        .or_default()
                        .push((placement_index, Some(source_index)));
                }
            } else {
                placement.for_each_source(|source| {
                    placements_by_domain
                        .entry(source.logical_fragment_id)
                        .or_default()
                        .push((placement_index, None));
                    Ok(())
                })?;
            }
        }
        let native_domains = fragments
            .iter()
            .filter_map(|fragment| {
                fragment
                    .native_logical_domain
                    .map(|native| (native.logical_fragment_id, (fragment, native)))
            })
            .collect::<BTreeMap<_, _>>();
        for retired in &self.retired_rows {
            for ordinal in 0..retired.domains.domain_count() {
                let source = retired.domains.domain_at(ordinal)?;
                for (placement_index, _) in placements_by_domain
                    .get(&source.logical_fragment_id)
                    .into_iter()
                    .flatten()
                {
                    let placement = &self.placements[*placement_index];
                    if let Some(live_source) =
                        placement.source_domain(source.logical_fragment_id)?
                    {
                        if live_source != source {
                            return Err(Error::invalid_input(format!(
                                "retired logical domain {} disagrees with live placement metadata",
                                source.logical_fragment_id
                            )));
                        }
                        if matches!(&retired.membership, RetiredLogicalRowMembership::AllRows)
                            && !matches!(placement, RowAddressPlacement::ExplicitMap(_))
                        {
                            return Err(Error::invalid_input(format!(
                                "fully retired logical domain {} still has a live placement",
                                source.logical_fragment_id
                            )));
                        }
                    }
                }
                if let Some((fragment, native)) = native_domains.get(&source.logical_fragment_id) {
                    let native_source = RowAddressLogicalDomain::new(
                        native.logical_fragment_id,
                        u32::try_from(fragment.physical_rows.ok_or_else(|| {
                            Error::invalid_input("native fragment is missing physical_rows")
                        })?)
                        .map_err(|_| Error::invalid_input("native rows exceed u32"))?,
                        native.creation_version,
                    )?;
                    if native_source != source {
                        return Err(Error::invalid_input(format!(
                            "retired logical domain {} disagrees with native metadata",
                            source.logical_fragment_id
                        )));
                    }
                    return Err(Error::invalid_input(format!(
                        "retired logical domain {} still has a native owner",
                        source.logical_fragment_id
                    )));
                }
            }
            if let RetiredLogicalRowMembership::Selection(selection) = &retired.membership {
                for (logical_fragment_id, slots) in selection.to_roaring_treemap()?.bitmaps() {
                    for (placement_index, source_hint) in placements_by_domain
                        .get(&logical_fragment_id)
                        .into_iter()
                        .flatten()
                    {
                        let placement = &self.placements[*placement_index];
                        if !matches!(placement, RowAddressPlacement::ExplicitMap(_))
                            && placement_owns_any_slots(
                                placement,
                                logical_fragment_id,
                                slots,
                                *source_hint,
                            )?
                        {
                            return Err(Error::invalid_input(format!(
                                "retired logical domain {logical_fragment_id} still overlaps live placement {placement_index}"
                            )));
                        }
                    }
                }
            }
        }
        let fragment_rows = fragments
            .iter()
            .map(|fragment| {
                let id = physical_fragment_id(fragment.id, "DataFragment")?;
                let rows = fragment.physical_rows.ok_or_else(|| {
                    Error::invalid_input(format!(
                        "storage-version-2.3 fragment {} is missing physical_rows",
                        fragment.id
                    ))
                })?;
                Ok((id, rows as u64))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        for placement in &self.placements {
            placement.for_each_mapped_destination_range(|fragment_id, start, length| {
                let physical_rows = fragment_rows.get(&fragment_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "placement references missing physical fragment {}",
                        fragment_id
                    ))
                })?;
                if start as u64 + length > *physical_rows {
                    return Err(Error::invalid_input(format!(
                        "placement destination fragment_id={fragment_id}, start={start}, length={length}, end={} exceeds physical_rows={physical_rows}",
                        start as u64 + length,
                    )));
                }
                Ok(())
            })?;
        }
        self.validate_physical_ownership_summaries(fragments)?;
        Ok(())
    }

    fn validate_physical_ownership_summaries(&self, fragments: &[Fragment]) -> Result<()> {
        if self
            .physical_row_ownership
            .windows(2)
            .any(|pair| pair[0].physical_fragment_id >= pair[1].physical_fragment_id)
        {
            return Err(Error::invalid_input(
                "physical ownership summaries must be strictly sorted by fragment id",
            ));
        }
        let targets = fragments
            .iter()
            .filter(|fragment| fragment.native_logical_domain.is_none())
            .collect::<Vec<_>>();
        if targets.len() != self.physical_row_ownership.len() {
            return Err(Error::invalid_input(
                "every target-only physical fragment must have exactly one ownership summary",
            ));
        }
        for fragment in targets {
            let fragment_id = physical_fragment_id(fragment.id, "target-only DataFragment")?;
            let physical_rows = u64::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input("target-only fragment is missing physical_rows")
            })?)
            .map_err(|_| Error::invalid_input("fragment row count exceeds u64"))?;
            let summary = self
                .physical_row_ownership
                .binary_search_by_key(&fragment_id, |summary| summary.physical_fragment_id)
                .ok()
                .map(|index| &self.physical_row_ownership[index])
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "target-only fragment {fragment_id} is missing an ownership summary"
                    ))
                })?;
            let mut mapped_row_count = 0_u64;
            let mut fingerprint = OffsetRangesFingerprintBuilder::new();
            for_each_indexed_canonical_mapped_offset_range(
                &self.placements,
                &self.destination_index,
                fragment_id,
                |start, end| {
                    mapped_row_count = mapped_row_count
                        .checked_add((end - start) as u64)
                        .ok_or_else(|| Error::invalid_input("mapped row count overflow"))?;
                    fingerprint.update(start, end);
                    Ok(())
                },
            )?;
            let unowned_row_count =
                physical_rows.checked_sub(mapped_row_count).ok_or_else(|| {
                    Error::invalid_input("mapped rows exceed target fragment physical rows")
                })?;
            if summary.mapped_row_count != mapped_row_count
                || summary.unowned_row_count != unowned_row_count
                || summary.mapped_offsets_fingerprint != fingerprint.finish()
            {
                return Err(Error::invalid_input(format!(
                    "physical ownership summary for fragment {fragment_id} does not match placement ranges"
                )));
            }
            match &fragment.deletion_file {
                Some(deletion_file) => {
                    let deleted_count = deletion_file.num_deleted_rows.ok_or_else(|| {
                        Error::invalid_input(
                            "storage-version-2.3 deletion files require num_deleted_rows",
                        )
                    })? as u64;
                    if deleted_count < unowned_row_count
                        || summary.deletion_offsets_fingerprint.len()
                            != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                    {
                        return Err(Error::invalid_input(format!(
                            "fragment {fragment_id} deletion metadata cannot cover unowned rows"
                        )));
                    }
                }
                None => {
                    if unowned_row_count != 0 || !summary.deletion_offsets_fingerprint.is_empty() {
                        return Err(Error::invalid_input(format!(
                            "fragment {fragment_id} has unowned rows without a deletion file"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn verify_visibility(
        &self,
        fragment: &Fragment,
        deleted_offsets: &RoaringBitmap,
    ) -> Result<()> {
        self.verify_visibility_with_placements(fragment, deleted_offsets, self.placements.iter())
    }

    /// Check the persisted destination index once and return an immutable view
    /// that can reuse it for bounded fragment-local lookups.
    pub fn validated_destination_index(&self) -> Result<ValidatedRowAddressDestinationIndex<'_>> {
        if self.destination_index != build_destination_index(&self.placements) {
            return Err(Error::invalid_input(
                "RowAddressLayout destination index does not match placement destinations",
            ));
        }
        Ok(ValidatedRowAddressDestinationIndex { layout: self })
    }

    fn verify_visibility_from_index(
        &self,
        fragment: &Fragment,
        deleted_offsets: &RoaringBitmap,
    ) -> Result<()> {
        let fragment_id = physical_fragment_id(fragment.id, "DataFragment")?;
        let placement_indices = self
            .destination_index
            .binary_search_by_key(&fragment_id, |entry| entry.physical_fragment_id)
            .ok()
            .map(|index| &self.destination_index[index].placement_indices);
        let Some(placement_indices) = placement_indices else {
            return self.verify_visibility_with_placements(
                fragment,
                deleted_offsets,
                std::iter::empty(),
            );
        };
        for placement_index in placement_indices {
            self.placements
                .get(*placement_index as usize)
                .ok_or_else(|| {
                    Error::invalid_input(
                        "row-address destination index references a missing placement",
                    )
                })?;
        }
        self.verify_visibility_with_placements(
            fragment,
            deleted_offsets,
            placement_indices
                .iter()
                .map(|index| &self.placements[*index as usize]),
        )
    }

    fn verify_visibility_with_placements<'a>(
        &self,
        fragment: &Fragment,
        deleted_offsets: &RoaringBitmap,
        placements: impl IntoIterator<Item = &'a RowAddressPlacement>,
    ) -> Result<()> {
        let fragment_id = physical_fragment_id(fragment.id, "DataFragment")?;
        if fragment.native_logical_domain.is_some() {
            return Ok(());
        }
        let physical_rows = u32::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input("target-only fragment is missing physical_rows")
        })?)
        .map_err(|_| Error::invalid_input("fragment row count exceeds u32"))?;
        if deleted_offsets
            .max()
            .is_some_and(|offset| offset >= physical_rows)
        {
            return Err(Error::invalid_input(
                "deletion vector contains an offset outside physical_rows",
            ));
        }
        let summary = self
            .physical_row_ownership
            .binary_search_by_key(&fragment_id, |summary| summary.physical_fragment_id)
            .ok()
            .map(|index| &self.physical_row_ownership[index])
            .ok_or_else(|| Error::invalid_input("physical ownership summary is missing"))?;
        let actual_deletion_fingerprint = if fragment.deletion_file.is_some() {
            fingerprint_deleted_offsets(deleted_offsets)
        } else {
            if !deleted_offsets.is_empty() {
                return Err(Error::invalid_input(
                    "fragment without deletion metadata has deleted offsets",
                ));
            }
            Vec::new()
        };
        if actual_deletion_fingerprint != summary.deletion_offsets_fingerprint {
            return Err(Error::invalid_input(
                "deletion vector content does not match the ownership summary",
            ));
        }
        let mut cursor = 0_u32;
        let mut unowned = 0_u64;
        for_each_canonical_mapped_offset_range(placements, fragment_id, |start, end| {
            if cursor < start {
                if !deleted_offsets.contains_range(cursor..start) {
                    return Err(Error::invalid_input(
                        "an unowned physical offset is not present in the deletion vector",
                    ));
                }
                unowned += (start - cursor) as u64;
            }
            cursor = end;
            Ok(())
        })?;
        if cursor < physical_rows {
            if !deleted_offsets.contains_range(cursor..physical_rows) {
                return Err(Error::invalid_input(
                    "an unowned physical offset is not present in the deletion vector",
                ));
            }
            unowned += (physical_rows - cursor) as u64;
        }
        if unowned != summary.unowned_row_count {
            return Err(Error::invalid_input(
                "ownership summary unowned count does not match placement gaps",
            ));
        }
        Ok(())
    }

    /// Return the compact logical ownership and mapped-offset fingerprint for
    /// a row-aligned rewrite target. ExplicitMap destinations return `None`
    /// because their external hidden `_rowid` file is the membership authority.
    pub fn row_aligned_rewrite_source(
        &self,
        fragment: &Fragment,
    ) -> Result<Option<(LogicalRowAddressSelection, Vec<u8>)>> {
        let fragment_id = physical_fragment_id(fragment.id, "DataFragment")?;
        let physical_rows = u32::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input("row-aligned rewrite fragment is missing physical_rows")
        })?)
        .map_err(|_| Error::invalid_input("row-aligned rewrite fragment rows exceed u32"))?;
        if let Some(native) = fragment.native_logical_domain {
            let source = RowAddressLogicalDomain::new(
                native.logical_fragment_id,
                physical_rows,
                native.creation_version,
            )?;
            return Ok(Some((
                LogicalRowAddressSelection::from_full_domains(&[source])?,
                offset_ranges_fingerprint([(0, physical_rows)]),
            )));
        }

        let mut ownership = RoaringTreemap::new();
        for placement in &self.placements {
            match placement {
                RowAddressPlacement::Direct(value)
                    if value.destination_fragment_id == fragment_id =>
                {
                    ownership |= effective_source_selection(
                        LogicalRowAddressSelection::from_full_domains(&[value.source])?,
                        value.excluded.as_ref(),
                    )?
                    .to_roaring_treemap()?;
                }
                RowAddressPlacement::PackedRun(value)
                    if value.destination_fragment_id == fragment_id =>
                {
                    for ordinal in 0..value.domains.domain_count() {
                        let source = value.domains.domain_at(ordinal)?;
                        let start = u64::from(source.logical_fragment_id) << 32;
                        ownership.insert_range(start..start + u64::from(source.slot_count));
                    }
                }
                RowAddressPlacement::Selected(value)
                    if value.destination_fragment_id == fragment_id =>
                {
                    ownership |= effective_source_selection(
                        value.selection.as_ref().clone(),
                        value.excluded.as_ref(),
                    )?
                    .to_roaring_treemap()?;
                }
                RowAddressPlacement::ExtentList(value) => {
                    let domain_start = u64::from(value.source.logical_fragment_id) << 32;
                    for extent in value
                        .extents
                        .iter()
                        .filter(|extent| extent.destination_fragment_id == fragment_id)
                    {
                        ownership.insert_range(
                            domain_start + u64::from(extent.source_start)
                                ..domain_start
                                    + u64::from(extent.source_start)
                                    + u64::from(extent.length),
                        );
                    }
                }
                RowAddressPlacement::SparseSelection(value)
                    if value.destination_fragment_id == fragment_id =>
                {
                    for source in &value.sources {
                        ownership |= effective_source_selection(
                            source.selection.as_ref().clone(),
                            source.excluded.as_ref(),
                        )?
                        .to_roaring_treemap()?;
                    }
                }
                RowAddressPlacement::ExplicitMap(value)
                    if value
                        .destinations
                        .iter()
                        .any(|destination| destination.physical_fragment_id == fragment_id) =>
                {
                    return Ok(None);
                }
                _ => {}
            }
        }
        let summary = self
            .physical_row_ownership
            .binary_search_by_key(&fragment_id, |summary| summary.physical_fragment_id)
            .ok()
            .map(|index| &self.physical_row_ownership[index])
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "row-aligned rewrite fragment {fragment_id} is missing physical ownership metadata"
                ))
        })?;
        Ok(Some((
            LogicalRowAddressSelection::from_bitmap(ownership)?,
            summary.mapped_offsets_fingerprint.clone(),
        )))
    }

    pub fn refresh_physical_ownership_summary(
        &mut self,
        fragment: &Fragment,
        deleted_offsets: &RoaringBitmap,
    ) -> Result<()> {
        self.destination_index = build_destination_index(&self.placements);
        self.refresh_physical_ownership_summary_from_index(fragment, deleted_offsets)
    }

    fn refresh_physical_ownership_summary_from_index(
        &mut self,
        fragment: &Fragment,
        deleted_offsets: &RoaringBitmap,
    ) -> Result<()> {
        let fragment_id = physical_fragment_id(fragment.id, "DataFragment")?;
        if fragment.native_logical_domain.is_some() {
            self.physical_row_ownership
                .retain(|summary| summary.physical_fragment_id != fragment_id);
            return Ok(());
        }
        let physical_rows = u64::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input("target-only fragment is missing physical_rows")
        })?)
        .map_err(|_| Error::invalid_input("fragment row count exceeds u64"))?;
        let mut mapped_row_count = 0_u64;
        let mut fingerprint = OffsetRangesFingerprintBuilder::new();
        for_each_indexed_canonical_mapped_offset_range(
            &self.placements,
            &self.destination_index,
            fragment_id,
            |start, end| {
                mapped_row_count = mapped_row_count
                    .checked_add((end - start) as u64)
                    .ok_or_else(|| Error::invalid_input("mapped row count overflow"))?;
                fingerprint.update(start, end);
                Ok(())
            },
        )?;
        let summary = PhysicalRowOwnershipSummary {
            physical_fragment_id: fragment_id,
            mapped_row_count,
            mapped_offsets_fingerprint: fingerprint.finish(),
            deletion_offsets_fingerprint: if fragment.deletion_file.is_some() {
                fingerprint_deleted_offsets(deleted_offsets)
            } else {
                Vec::new()
            },
            unowned_row_count: physical_rows
                .checked_sub(mapped_row_count)
                .ok_or_else(|| Error::invalid_input("mapped rows exceed fragment physical rows"))?,
        };
        self.set_physical_ownership_summary(summary);
        self.verify_visibility_from_index(fragment, deleted_offsets)
    }

    fn set_physical_ownership_summary(&mut self, summary: PhysicalRowOwnershipSummary) {
        match self
            .physical_row_ownership
            .binary_search_by_key(&summary.physical_fragment_id, |existing| {
                existing.physical_fragment_id
            }) {
            Ok(index) => self.physical_row_ownership[index] = summary,
            Err(index) => self.physical_row_ownership.insert(index, summary),
        }
    }

    pub fn is_retired(&self, address: LogicalRowAddress) -> Result<bool> {
        for retired in &self.retired_rows {
            if retired.contains(address)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn is_retired_many(&self, addresses: &[LogicalRowAddress]) -> Result<Vec<bool>> {
        addresses
            .iter()
            .copied()
            .map(|address| self.is_retired(address))
            .collect()
    }

    pub fn retired_logical_row_selections(
        &self,
    ) -> impl Iterator<Item = &LogicalRowAddressSelection> {
        self.retired_rows.iter().filter_map(|retired| {
            if let RetiredLogicalRowMembership::Selection(selection) = &retired.membership {
                Some(selection.as_ref())
            } else {
                None
            }
        })
    }

    pub fn retired_logical_domains(&self) -> Result<Vec<RowAddressLogicalDomain>> {
        let mut domains = Vec::new();
        for retired in &self.retired_rows {
            for ordinal in 0..retired.domains.domain_count() {
                domains.push(retired.domains.domain_at(ordinal)?);
            }
        }
        domains.sort_unstable();
        Ok(domains)
    }

    /// Return all retired logical rows as Roaring containers without expanding
    /// high-entropy selections row by row.
    pub fn retired_logical_row_bitmap(&self) -> Result<RoaringTreemap> {
        let mut rows = RoaringTreemap::new();
        for retired in &self.retired_rows {
            match &retired.membership {
                RetiredLogicalRowMembership::AllRows => {
                    for ordinal in 0..retired.domains.domain_count() {
                        let domain = retired.domains.domain_at(ordinal)?;
                        let start = u64::from(domain.logical_fragment_id) << 32;
                        rows.insert_range(start..start + u64::from(domain.slot_count));
                    }
                }
                RetiredLogicalRowMembership::Selection(selection) => {
                    rows |= selection.to_roaring_treemap()?;
                }
            }
        }
        Ok(rows)
    }

    /// Return retired logical rows intersected with an index coverage set.
    ///
    /// Full retired domains are projected from the coverage containers instead
    /// of being expanded first. Selected retirement payloads are intersected
    /// one at a time, so peak memory is bounded by one encoded selection plus
    /// the requested result rather than the table-wide retired population.
    pub fn retired_logical_row_bitmap_for_coverage(
        &self,
        coverage: &RoaringTreemap,
    ) -> Result<RoaringTreemap> {
        let coverage_by_domain = coverage.bitmaps().collect::<BTreeMap<_, _>>();
        let mut rows = BTreeMap::<u32, RoaringBitmap>::new();
        for retired in &self.retired_rows {
            match &retired.membership {
                RetiredLogicalRowMembership::AllRows => {
                    for (logical_fragment_id, slots) in &coverage_by_domain {
                        if retired.source_domain(*logical_fragment_id)?.is_some() {
                            *rows.entry(*logical_fragment_id).or_default() |= *slots;
                        }
                    }
                }
                RetiredLogicalRowMembership::Selection(selection) => {
                    let mut has_relevant_domain = false;
                    for logical_fragment_id in coverage_by_domain.keys() {
                        if retired.source_domain(*logical_fragment_id)?.is_some() {
                            has_relevant_domain = true;
                            break;
                        }
                    }
                    if !has_relevant_domain {
                        continue;
                    }
                    let intersection = selection.to_roaring_treemap()? & coverage;
                    for (logical_fragment_id, slots) in intersection.bitmaps() {
                        *rows.entry(logical_fragment_id).or_default() |= slots;
                    }
                }
            }
        }
        Ok(RoaringTreemap::from_bitmaps(rows))
    }

    pub fn visit_retired_ranges(
        &self,
        mut visit: impl FnMut(LogicalRowAddressRange),
    ) -> Result<()> {
        for retired in &self.retired_rows {
            match &retired.membership {
                RetiredLogicalRowMembership::AllRows => {
                    for ordinal in 0..retired.domains.domain_count() {
                        let domain = retired.domains.domain_at(ordinal)?;
                        visit(LogicalRowAddressRange::new(
                            domain.logical_fragment_id,
                            0,
                            domain.slot_count,
                        ));
                    }
                }
                RetiredLogicalRowMembership::Selection(selection) => {
                    visit_selection_value_ranges(selection, |range| {
                        visit(range);
                        Ok(())
                    })?;
                }
            }
        }
        Ok(())
    }

    pub fn max_current_logical_fragment_id(&self, fragments: &[Fragment]) -> Option<u32> {
        self.placements
            .iter()
            .filter_map(|placement| match placement {
                RowAddressPlacement::Direct(value) => Some(value.source.logical_fragment_id),
                RowAddressPlacement::PackedRun(value) => {
                    value.domains.last_logical_fragment_id().ok()
                }
                RowAddressPlacement::Selected(value) => Some(value.source.logical_fragment_id),
                RowAddressPlacement::ExtentList(value) => Some(value.source.logical_fragment_id),
                RowAddressPlacement::SparseSelection(value) => value
                    .sources
                    .last()
                    .map(|source| source.source.logical_fragment_id),
                RowAddressPlacement::ExplicitMap(value) => value
                    .sources
                    .last()
                    .map(|source| source.source.logical_fragment_id),
            })
            .chain(fragments.iter().filter_map(|fragment| {
                fragment
                    .native_logical_domain
                    .as_ref()
                    .map(|domain| domain.logical_fragment_id)
            }))
            .chain(
                self.retired_rows
                    .iter()
                    .filter_map(|retired| retired.domains.last_logical_fragment_id().ok()),
            )
            .max()
    }

    pub fn current_logical_fragment_bitmap(&self, fragments: &[Fragment]) -> Result<RoaringBitmap> {
        let mut logical_fragments = fragments
            .iter()
            .filter_map(|fragment| {
                fragment
                    .native_logical_domain
                    .as_ref()
                    .map(|domain| domain.logical_fragment_id)
            })
            .collect::<RoaringBitmap>();
        for placement in &self.placements {
            match placement {
                RowAddressPlacement::Direct(value) => {
                    logical_fragments.insert(value.source.logical_fragment_id);
                }
                RowAddressPlacement::PackedRun(value) => {
                    for ordinal in 0..value.domains.domain_count() {
                        logical_fragments.insert(value.domains.logical_fragment_id_at(ordinal)?);
                    }
                }
                RowAddressPlacement::Selected(value) => {
                    logical_fragments.insert(value.source.logical_fragment_id);
                }
                RowAddressPlacement::ExtentList(value) => {
                    logical_fragments.insert(value.source.logical_fragment_id);
                }
                RowAddressPlacement::SparseSelection(value) => {
                    logical_fragments.extend(
                        value
                            .sources
                            .iter()
                            .map(|source| source.source.logical_fragment_id),
                    );
                }
                RowAddressPlacement::ExplicitMap(value) => {
                    logical_fragments.extend(
                        value
                            .sources
                            .iter()
                            .map(|source| source.source.logical_fragment_id),
                    );
                }
            }
        }
        for retired in &self.retired_rows {
            for ordinal in 0..retired.domains.domain_count() {
                logical_fragments.insert(retired.domains.logical_fragment_id_at(ordinal)?);
            }
        }
        Ok(logical_fragments)
    }

    pub fn domain_creation_version(
        &self,
        fragments: &[Fragment],
        logical_fragment_id: u32,
    ) -> Result<Option<u64>> {
        let mut version = None;
        for placement in &self.placements {
            if let Some(source) = placement.source_domain(logical_fragment_id)? {
                merge_creation_version(&mut version, source.creation_version, logical_fragment_id)?;
            }
        }
        for domain in fragments
            .iter()
            .filter_map(|fragment| fragment.native_logical_domain.as_ref())
            .filter(|domain| domain.logical_fragment_id == logical_fragment_id)
        {
            merge_creation_version(&mut version, domain.creation_version, logical_fragment_id)?;
        }
        for retired in &self.retired_rows {
            if let Some(source) = retired.source_domain(logical_fragment_id)? {
                merge_creation_version(&mut version, source.creation_version, logical_fragment_id)?;
            }
        }
        Ok(version)
    }

    pub fn resolve_many(
        &self,
        fragments: &[Fragment],
        addresses: &[LogicalRowAddress],
    ) -> Result<Vec<PlacementResolution>> {
        let mut transient = self.clone();
        transient.refresh_fingerprint_with_fragments(
            fragments,
            self.max_current_logical_fragment_id(fragments),
        );
        RowAddressRouter::try_new(
            Arc::new(transient),
            fragments,
            self.max_current_logical_fragment_id(fragments),
        )?
        .resolve_many(addresses)
    }

    pub fn physical_to_logical_many(
        &self,
        fragments: &[Fragment],
        addresses: &[RowAddress],
    ) -> Result<Vec<PhysicalToLogicalResolution>> {
        let mut transient = self.clone();
        transient.refresh_fingerprint_with_fragments(
            fragments,
            self.max_current_logical_fragment_id(fragments),
        );
        RowAddressRouter::try_new(
            Arc::new(transient),
            fragments,
            self.max_current_logical_fragment_id(fragments),
        )?
        .physical_to_logical_many(addresses)
    }
}

/// Fingerprint one independently readable ExplicitMap page.
///
/// The encoding is column-major and includes the column and row counts, so a
/// page cannot be substituted for a differently shaped page with the same raw
/// values. Callers must pass columns in their persisted schema order.
pub fn fingerprint_explicit_map_u64_page(columns: &[&[u64]]) -> Result<Vec<u8>> {
    let Some(first) = columns.first() else {
        return Err(Error::invalid_input(
            "ExplicitMap page fingerprint requires at least one column",
        ));
    };
    if first.is_empty() || columns.iter().any(|column| column.len() != first.len()) {
        return Err(Error::invalid_input(
            "ExplicitMap page fingerprint columns must be non-empty and have equal lengths",
        ));
    }

    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut hash = OFFSET;
    let mut update = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= *byte as u128;
            hash = hash.wrapping_mul(PRIME);
        }
    };
    update(b"LanceExplicitMapPageV1");
    update(&(columns.len() as u64).to_le_bytes());
    update(&(first.len() as u64).to_le_bytes());
    for (column_index, column) in columns.iter().enumerate() {
        update(&(column_index as u64).to_le_bytes());
        for value in *column {
            update(&value.to_le_bytes());
        }
    }
    Ok(hash.to_le_bytes().to_vec())
}

fn stable_fingerprint(bytes: &[u8]) -> [u8; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE] {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= *byte as u128;
        hash = hash.wrapping_mul(PRIME);
    }
    hash.to_le_bytes()
}

struct OffsetRangesFingerprintBuilder {
    hash: u128,
    count: u64,
}

impl OffsetRangesFingerprintBuilder {
    fn new() -> Self {
        Self {
            hash: 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d,
            count: 0,
        }
    }

    fn update(&mut self, start: u32, end: u32) {
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        for byte in start.to_le_bytes().into_iter().chain(end.to_le_bytes()) {
            self.hash ^= byte as u128;
            self.hash = self.hash.wrapping_mul(PRIME);
        }
        self.count += 1;
    }

    fn finish(mut self) -> Vec<u8> {
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        for byte in self.count.to_le_bytes() {
            self.hash ^= byte as u128;
            self.hash = self.hash.wrapping_mul(PRIME);
        }
        self.hash.to_le_bytes().to_vec()
    }
}

fn offset_ranges_fingerprint(ranges: impl IntoIterator<Item = (u32, u32)>) -> Vec<u8> {
    let mut builder = OffsetRangesFingerprintBuilder::new();
    for (start, end) in ranges {
        builder.update(start, end);
    }
    builder.finish()
}

pub fn fingerprint_deleted_offsets(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut builder = OffsetRangesFingerprintBuilder::new();
    let mut values = bitmap.iter();
    while let Some(range) = values.next_range() {
        builder.update(*range.start(), range.end().saturating_add(1));
    }
    builder.finish()
}

fn for_each_indexed_canonical_mapped_offset_range(
    placements: &[RowAddressPlacement],
    destination_index: &[RowAddressDestinationIndexEntry],
    fragment_id: u32,
    visit: impl FnMut(u32, u32) -> Result<()>,
) -> Result<()> {
    let Some(destination) = destination_index
        .binary_search_by_key(&fragment_id, |entry| entry.physical_fragment_id)
        .ok()
        .map(|index| &destination_index[index])
    else {
        return Ok(());
    };
    for placement_index in &destination.placement_indices {
        placements.get(*placement_index as usize).ok_or_else(|| {
            Error::invalid_input("row-address destination index references a missing placement")
        })?;
    }
    for_each_canonical_mapped_offset_range(
        destination
            .placement_indices
            .iter()
            .map(|index| &placements[*index as usize]),
        fragment_id,
        visit,
    )
}

fn for_each_canonical_mapped_offset_range<'a>(
    placements: impl IntoIterator<Item = &'a RowAddressPlacement>,
    fragment_id: u32,
    mut visit: impl FnMut(u32, u32) -> Result<()>,
) -> Result<()> {
    let mut ranges = Vec::<(u32, u32)>::new();
    for placement in placements {
        placement.for_each_mapped_destination_range(|destination, start, length| {
            if destination == fragment_id {
                let end = u32::try_from(start as u64 + length).map_err(|_| {
                    Error::invalid_input("physical destination range end exceeds u32")
                })?;
                ranges.push((start, end));
            }
            Ok(())
        })?;
    }
    // Hole reuse can interleave ranges from different placements. Sorting only
    // by each placement's first range would therefore report a false overlap.
    ranges.sort_unstable();

    let mut pending: Option<(u32, u32)> = None;
    for (start, end) in ranges {
        if let Some((pending_start, pending_end)) = pending.take() {
            if start < pending_end {
                return Err(Error::invalid_input(format!(
                    "overlapping mapped ownership in physical fragment {fragment_id}: {start}..{end} overlaps {pending_start}..{pending_end}"
                )));
            }
            if start == pending_end {
                pending = Some((pending_start, end));
            } else {
                visit(pending_start, pending_end)?;
                pending = Some((start, end));
            }
        } else {
            pending = Some((start, end));
        }
    }
    if let Some((start, end)) = pending {
        visit(start, end)?;
    }
    Ok(())
}

impl RowAddressPlacement {
    fn canonicalize(&mut self) {
        match self {
            Self::Selected(_) => {}
            Self::ExtentList(value) => value.extents.sort_by_key(|extent| extent.source_start),
            Self::SparseSelection(_) => {}
            Self::ExplicitMap(value) => {
                value.pages.sort_by_key(|page| page.first_logical_address);
            }
            Self::Direct(_) | Self::PackedRun(_) => {}
        }
    }

    fn canonical_key(&self) -> (u64, u8, Vec<u8>) {
        let first = match self {
            Self::Direct(value) => (value.source.logical_fragment_id as u64) << 32,
            Self::PackedRun(value) => (value.domains.first_logical_fragment_id() as u64) << 32,
            Self::Selected(value) => selection_first_address(&value.selection),
            Self::ExtentList(value) => value
                .extents
                .first()
                .map(|extent| {
                    ((value.source.logical_fragment_id as u64) << 32) | extent.source_start as u64
                })
                .unwrap_or(u64::MAX),
            Self::SparseSelection(value) => value
                .sources
                .first()
                .map(|source| selection_first_address(&source.selection))
                .unwrap_or(u64::MAX),
            Self::ExplicitMap(value) => value
                .sources
                .first()
                .map(|source| selection_first_address(&source.selection))
                .unwrap_or(u64::MAX),
        };
        let tag = match self {
            Self::Direct(_) => 0,
            Self::PackedRun(_) => 1,
            Self::Selected(_) => 2,
            Self::ExtentList(_) => 3,
            Self::SparseSelection(_) => 4,
            Self::ExplicitMap(_) => 5,
        };
        let proto: pb::RowAddressPlacement = self.into();
        let mut bytes = proto.encode_to_vec();
        for selection in placement_selections(self) {
            let encoded = encoded_selection_bytes(selection);
            bytes.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&encoded);
        }
        (first, tag, bytes)
    }

    pub fn destination_ranges(&self) -> Vec<(u32, u32, u64)> {
        match self {
            Self::Direct(value) => vec![(
                value.destination_fragment_id,
                value.destination_start,
                value.source.slot_count as u64,
            )],
            Self::PackedRun(value) => vec![(
                value.destination_fragment_id,
                value.destination_start,
                value.domains.total_slot_count().unwrap_or(u64::MAX),
            )],
            Self::Selected(value) => vec![(
                value.destination_fragment_id,
                value.destination_start,
                value.selection.cardinality(),
            )],
            Self::ExtentList(value) => value
                .extents
                .iter()
                .map(|extent| {
                    (
                        extent.destination_fragment_id,
                        extent.destination_start,
                        extent.length as u64,
                    )
                })
                .collect(),
            Self::SparseSelection(value) => vec![(
                value.destination_fragment_id,
                value.destination_start,
                value
                    .sources
                    .iter()
                    .map(|source| source.selection.cardinality())
                    .sum(),
            )],
            Self::ExplicitMap(value) => value
                .destinations
                .iter()
                .map(|destination| {
                    (
                        destination.physical_fragment_id,
                        destination.destination_start,
                        destination.row_count as u64,
                    )
                })
                .collect(),
        }
    }

    fn for_each_mapped_destination_range(
        &self,
        mut visit: impl FnMut(u32, u32, u64) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Direct(value) => {
                if let Some(excluded) = value.excluded.as_deref()
                    && let Some(ranges) = excluded.compact_ranges(MAX_RANGE_ENCODING_RUNS)?
                {
                    return for_each_mapped_range_excluding_ordinal_ranges(
                        value.destination_fragment_id,
                        value.destination_start,
                        value.source.slot_count as u64,
                        ranges.into_iter().map(|range| {
                            if range.logical_fragment_id != value.source.logical_fragment_id {
                                return Err(Error::invalid_input(
                                    "Direct exclusion references a different logical domain",
                                ));
                            }
                            Ok((u64::from(range.start_slot), u64::from(range.end_slot)))
                        }),
                        &mut visit,
                    );
                }
                let mut emitter = MappedRangeEmitter::new(
                    value.destination_fragment_id,
                    value.destination_start,
                    value.source.slot_count as u64,
                    &mut visit,
                );
                if let Some(excluded) = value.excluded.as_deref() {
                    excluded.try_for_each_address(|address| {
                        if address.logical_fragment_id() != value.source.logical_fragment_id {
                            return Err(Error::invalid_input(
                                "Direct exclusion references a different logical domain",
                            ));
                        }
                        emitter.exclude(u64::from(address.immutable_slot()))
                    })?;
                }
                emitter.finish()
            }
            Self::Selected(value) => {
                if let Some(excluded) = value.excluded.as_deref()
                    && let Some(ranges) = compact_excluded_rank_ranges(
                        &value.selection,
                        excluded,
                        value.source.logical_fragment_id,
                        0,
                    )?
                {
                    return for_each_mapped_range_excluding_ordinal_ranges(
                        value.destination_fragment_id,
                        value.destination_start,
                        value.selection.cardinality(),
                        ranges.into_iter().map(Ok),
                        &mut visit,
                    );
                }
                let mut emitter = MappedRangeEmitter::new(
                    value.destination_fragment_id,
                    value.destination_start,
                    value.selection.cardinality(),
                    &mut visit,
                );
                if let Some(excluded) = value.excluded.as_deref() {
                    excluded.try_for_each_address(|address| {
                        let rank = value.selection.rank(address)?.ok_or_else(|| {
                            Error::invalid_input(
                                "Selected exclusion is outside its source selection",
                            )
                        })?;
                        emitter.exclude(rank)
                    })?;
                }
                emitter.finish()
            }
            Self::SparseSelection(value) => {
                let mut prefix = 0_u64;
                let mut compact_ranges = Some(Vec::new());
                for source in &value.sources {
                    if let Some(excluded) = source.excluded.as_deref()
                        && compact_ranges.is_some()
                    {
                        match compact_excluded_rank_ranges(
                            &source.selection,
                            excluded,
                            source.source.logical_fragment_id,
                            prefix,
                        )? {
                            Some(ranges) => {
                                if let Some(accumulated) = compact_ranges.as_mut() {
                                    accumulated.extend(ranges);
                                }
                            }
                            None => compact_ranges = None,
                        }
                    }
                    prefix = prefix
                        .checked_add(source.selection.cardinality())
                        .ok_or_else(|| {
                            Error::invalid_input("SparseSelection destination prefix overflow")
                        })?;
                }
                if let Some(ranges) = compact_ranges {
                    return for_each_mapped_range_excluding_ordinal_ranges(
                        value.destination_fragment_id,
                        value.destination_start,
                        prefix,
                        ranges.into_iter().map(Ok),
                        &mut visit,
                    );
                }
                let mut prefix = 0_u64;
                let mut emitter = MappedRangeEmitter::new(
                    value.destination_fragment_id,
                    value.destination_start,
                    value
                        .sources
                        .iter()
                        .map(|source| source.selection.cardinality())
                        .sum(),
                    &mut visit,
                );
                for source in &value.sources {
                    if let Some(excluded) = &source.excluded {
                        excluded.try_for_each_address(|address| {
                            let rank = source.selection.rank(address)?.ok_or_else(|| {
                                Error::invalid_input(
                                    "SparseSelection exclusion is outside its source selection",
                                )
                            })?;
                            emitter.exclude(prefix.checked_add(rank).ok_or_else(|| {
                                Error::invalid_input("SparseSelection exclusion rank overflow")
                            })?)
                        })?;
                    }
                    prefix = prefix
                        .checked_add(source.selection.cardinality())
                        .ok_or_else(|| {
                            Error::invalid_input("SparseSelection destination prefix overflow")
                        })?;
                }
                emitter.finish()
            }
            _ => {
                let mut ranges = self.destination_ranges();
                ranges.sort_unstable_by_key(|(fragment_id, start, _)| (*fragment_id, *start));
                for (fragment_id, start, length) in ranges {
                    visit(fragment_id, start, length)?;
                }
                Ok(())
            }
        }
    }

    fn resolve(
        &self,
        address: LogicalRowAddress,
        placement_index: u32,
    ) -> Result<Option<PhysicalRowLocator>> {
        let physical = match self {
            Self::Direct(value) => {
                if address.logical_fragment_id() != value.source.logical_fragment_id
                    || address.immutable_slot() >= value.source.slot_count
                {
                    return Ok(None);
                }
                if is_excluded(value.excluded.as_ref(), address)? {
                    return Ok(None);
                }
                Some(RowAddress::new_from_parts(
                    value.destination_fragment_id,
                    value.destination_start + address.immutable_slot(),
                ))
            }
            Self::PackedRun(value) => {
                let Some(ordinal) = value
                    .domains
                    .domain_ordinal(address.logical_fragment_id())?
                else {
                    return Ok(None);
                };
                if address.immutable_slot() >= value.domains.slot_count_at(ordinal)? {
                    return Ok(None);
                }
                let destination_offset =
                    value.domains.slot_prefix(ordinal)? + address.immutable_slot() as u64;
                Some(RowAddress::new_from_parts(
                    value.destination_fragment_id,
                    value.destination_start
                        + u32::try_from(destination_offset).map_err(|_| {
                            Error::invalid_input("PackedRun destination offset exceeds u32")
                        })?,
                ))
            }
            Self::Selected(value) => {
                let Some(rank) = value.selection.rank(address)? else {
                    return Ok(None);
                };
                if is_excluded(value.excluded.as_ref(), address)? {
                    return Ok(None);
                }
                Some(RowAddress::new_from_parts(
                    value.destination_fragment_id,
                    value.destination_start + rank as u32,
                ))
            }
            Self::ExtentList(value) => {
                if address.logical_fragment_id() != value.source.logical_fragment_id {
                    return Ok(None);
                }
                value.extents.iter().find_map(|extent| {
                    let source_end = extent.source_start + extent.length;
                    (extent.source_start <= address.immutable_slot()
                        && address.immutable_slot() < source_end)
                        .then(|| {
                            RowAddress::new_from_parts(
                                extent.destination_fragment_id,
                                extent.destination_start + address.immutable_slot()
                                    - extent.source_start,
                            )
                        })
                })
            }
            Self::SparseSelection(value) => {
                let mut prefix = 0_u64;
                let mut resolved = None;
                for source in &value.sources {
                    if let Some(rank) = source.selection.rank(address)? {
                        if !is_excluded(source.excluded.as_ref(), address)? {
                            resolved = Some(RowAddress::new_from_parts(
                                value.destination_fragment_id,
                                value.destination_start + (prefix + rank) as u32,
                            ));
                        }
                        break;
                    }
                    prefix += source.selection.cardinality();
                }
                resolved
            }
            Self::ExplicitMap(value) => {
                let mut selected = false;
                for source in &value.sources {
                    selected |= source.selection.contains(address)?
                        && !is_excluded(source.excluded.as_ref(), address)?;
                }
                if !selected {
                    return Ok(None);
                }
                let page_index = value
                    .pages
                    .partition_point(|page| page.last_logical_address < address.raw());
                if page_index == value.pages.len()
                    || value.pages[page_index].first_logical_address > address.raw()
                {
                    // ExplicitMap sources describe the immutable logical domain,
                    // while the external locator is authoritative for live
                    // membership. A gap between pages is therefore a retired
                    // address, not corrupt routing metadata.
                    return Ok(None);
                }
                return Ok(Some(PhysicalRowLocator::ExplicitMap {
                    placement_index,
                    page_index: page_index as u32,
                }));
            }
        };
        Ok(physical.map(PhysicalRowLocator::Physical))
    }

    fn inverse(&self, address: RowAddress) -> Result<Option<LogicalRowAddress>> {
        match self {
            Self::Direct(value) => {
                if address.fragment_id() != value.destination_fragment_id
                    || address.row_offset() < value.destination_start
                {
                    return Ok(None);
                }
                let ordinal = address.row_offset() - value.destination_start;
                if ordinal >= value.source.slot_count {
                    return Ok(None);
                }
                let logical = LogicalRowAddress::try_new_from_parts(
                    value.source.logical_fragment_id,
                    ordinal,
                )?;
                if is_excluded(value.excluded.as_ref(), logical)? {
                    Ok(None)
                } else {
                    Ok(Some(logical))
                }
            }
            Self::PackedRun(value) => {
                if address.fragment_id() != value.destination_fragment_id
                    || address.row_offset() < value.destination_start
                {
                    return Ok(None);
                }
                let offset = (address.row_offset() - value.destination_start) as u64;
                let Some((ordinal, local_slot)) = value.domains.ordinal_for_slot_offset(offset)?
                else {
                    return Ok(None);
                };
                LogicalRowAddress::try_new_from_parts(
                    value.domains.logical_fragment_id_at(ordinal)?,
                    local_slot,
                )
                .map(Some)
            }
            Self::Selected(value) => {
                if address.fragment_id() != value.destination_fragment_id
                    || address.row_offset() < value.destination_start
                {
                    return Ok(None);
                }
                let ordinal = (address.row_offset() - value.destination_start) as u64;
                let logical = value.selection.select(ordinal)?;
                if let Some(logical) = logical
                    && is_excluded(value.excluded.as_ref(), logical)?
                {
                    Ok(None)
                } else {
                    Ok(logical)
                }
            }
            Self::ExtentList(value) => {
                for extent in &value.extents {
                    if address.fragment_id() == extent.destination_fragment_id
                        && extent.destination_start <= address.row_offset()
                        && address.row_offset() < extent.destination_start + extent.length
                    {
                        return LogicalRowAddress::try_new_from_parts(
                            value.source.logical_fragment_id,
                            extent.source_start + address.row_offset() - extent.destination_start,
                        )
                        .map(Some);
                    }
                }
                Ok(None)
            }
            Self::SparseSelection(value) => {
                if address.fragment_id() != value.destination_fragment_id
                    || address.row_offset() < value.destination_start
                {
                    return Ok(None);
                }
                let mut ordinal = (address.row_offset() - value.destination_start) as u64;
                for source in &value.sources {
                    if ordinal < source.selection.cardinality() {
                        let logical = source.selection.select(ordinal)?;
                        if let Some(logical) = logical
                            && is_excluded(source.excluded.as_ref(), logical)?
                        {
                            return Ok(None);
                        }
                        return Ok(logical);
                    }
                    ordinal -= source.selection.cardinality();
                }
                Ok(None)
            }
            Self::ExplicitMap(_) => Ok(None),
        }
    }
}

struct MappedRangeEmitter<'a, F> {
    fragment_id: u32,
    destination_start: u32,
    cardinality: u64,
    cursor: u64,
    visit: &'a mut F,
}

impl<F> MappedRangeEmitter<'_, F>
where
    F: FnMut(u32, u32, u64) -> Result<()>,
{
    fn new(
        fragment_id: u32,
        destination_start: u32,
        cardinality: u64,
        visit: &mut F,
    ) -> MappedRangeEmitter<'_, F> {
        MappedRangeEmitter {
            fragment_id,
            destination_start,
            cardinality,
            cursor: 0,
            visit,
        }
    }

    fn emit(&mut self, start: u64, end: u64) -> Result<()> {
        if start >= end {
            return Ok(());
        }
        let destination_start = self
            .destination_start
            .checked_add(
                u32::try_from(start)
                    .map_err(|_| Error::invalid_input("mapped destination offset exceeds u32"))?,
            )
            .ok_or_else(|| Error::invalid_input("mapped destination overflow"))?;
        (self.visit)(self.fragment_id, destination_start, end - start)
    }

    fn exclude(&mut self, excluded: u64) -> Result<()> {
        self.exclude_range(
            excluded,
            excluded
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("placement exclusion ordinal overflow"))?,
        )
    }

    fn exclude_range(&mut self, start: u64, end: u64) -> Result<()> {
        if start < self.cursor || start >= end || end > self.cardinality {
            return Err(Error::invalid_input(
                "placement exclusions must have unique increasing destination rank ranges",
            ));
        }
        self.emit(self.cursor, start)?;
        self.cursor = end;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.emit(self.cursor, self.cardinality)
    }
}

fn for_each_mapped_range_excluding_ordinal_ranges<F>(
    fragment_id: u32,
    destination_start: u32,
    cardinality: u64,
    excluded_ranges: impl IntoIterator<Item = Result<(u64, u64)>>,
    visit: &mut F,
) -> Result<()>
where
    F: FnMut(u32, u32, u64) -> Result<()>,
{
    let mut emitter = MappedRangeEmitter::new(fragment_id, destination_start, cardinality, visit);
    for excluded in excluded_ranges {
        let (start, end) = excluded?;
        emitter.exclude_range(start, end)?;
    }
    emitter.finish()
}

struct DomainOccupancy {
    ranges: Vec<(u32, u32)>,
    sorted: bool,
}

impl Default for DomainOccupancy {
    fn default() -> Self {
        Self {
            ranges: Vec::new(),
            sorted: true,
        }
    }
}

impl DomainOccupancy {
    fn add_range(&mut self, start: u32, end: u32, context: &str) -> Result<()> {
        if start >= end {
            return Err(Error::invalid_input(format!(
                "overlapping or empty row-address ownership in {context}: {start}..{end}"
            )));
        }
        if let Some((previous_start, previous_end)) = self.ranges.last().copied() {
            if previous_start <= start {
                if start < previous_end {
                    return Err(Error::invalid_input(format!(
                        "overlapping row-address ownership in {context}: {start}..{end} overlaps {previous_start}..{previous_end}"
                    )));
                }
            } else {
                self.sorted = false;
            }
        }
        self.ranges.push((start, end));
        Ok(())
    }

    fn validate(&mut self) -> Result<()> {
        if !self.sorted {
            self.ranges.sort_unstable();
        }
        if let Some(pair) = self.ranges.windows(2).find(|pair| pair[1].0 < pair[0].1) {
            return Err(Error::invalid_input(format!(
                "overlapping row-address ownership: {}..{} overlaps {}..{}",
                pair[1].0, pair[1].1, pair[0].0, pair[0].1
            )));
        }
        Ok(())
    }
}

fn validate_destination_ownership(
    placements: &[RowAddressPlacement],
    fragments: &[Fragment],
) -> Result<()> {
    let mut destinations = BTreeMap::<u32, DomainOccupancy>::new();
    for fragment in fragments {
        if fragment.native_logical_domain.is_some() {
            let fragment_id = physical_fragment_id(fragment.id, "native DataFragment")?;
            let rows = u32::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input("native DataFragment is missing physical_rows")
            })?)
            .map_err(|_| Error::invalid_input("native DataFragment row count exceeds u32"))?;
            destinations.entry(fragment_id).or_default().add_range(
                0,
                rows,
                "native physical fragment",
            )?;
        }
    }
    for placement in placements {
        placement.for_each_mapped_destination_range(|fragment_id, start, length| {
            let end = u32::try_from(start as u64 + length)
                .map_err(|_| Error::invalid_input("physical destination range end exceeds u32"))?;
            destinations.entry(fragment_id).or_default().add_range(
                start,
                end,
                "placement physical destination",
            )
        })?;
    }
    for occupancy in destinations.values_mut() {
        occupancy.validate()?;
    }
    Ok(())
}

#[derive(Clone)]
enum EffectiveSourceBase {
    Full,
    Selection(LogicalRowAddressSelection),
}

#[derive(Clone)]
struct EffectiveSourceSelection {
    base: EffectiveSourceBase,
    excluded: Option<Arc<LogicalRowAddressSelection>>,
}

impl EffectiveSourceSelection {
    fn full(excluded: Option<&Arc<LogicalRowAddressSelection>>) -> Self {
        Self {
            base: EffectiveSourceBase::Full,
            excluded: excluded.cloned(),
        }
    }

    fn selected(
        selection: LogicalRowAddressSelection,
        excluded: Option<&Arc<LogicalRowAddressSelection>>,
    ) -> Self {
        Self {
            base: EffectiveSourceBase::Selection(selection),
            excluded: excluded.cloned(),
        }
    }

    fn base_slots(&self, source: RowAddressLogicalDomain) -> Result<Option<RoaringBitmap>> {
        match &self.base {
            EffectiveSourceBase::Full => Ok(None),
            EffectiveSourceBase::Selection(selection) => Ok(Some(selection_slots_for_domain(
                selection,
                source.logical_fragment_id,
            )?)),
        }
    }

    fn excluded_slots(&self, source: RowAddressLogicalDomain) -> Result<RoaringBitmap> {
        self.excluded
            .as_deref()
            .map(|selection| selection_slots_for_domain(selection, source.logical_fragment_id))
            .transpose()
            .map(Option::unwrap_or_default)
    }

    fn is_empty(&self, source: RowAddressLogicalDomain) -> Result<bool> {
        let excluded = self.excluded_slots(source)?;
        Ok(match self.base_slots(source)? {
            None => {
                excluded.range_cardinality(0..source.slot_count) == u64::from(source.slot_count)
            }
            Some(base) => base.difference_len(&excluded) == 0,
        })
    }

    fn overlaps(&self, other: &Self, source: RowAddressLogicalDomain) -> Result<bool> {
        let mut excluded = self.excluded_slots(source)?;
        excluded |= other.excluded_slots(source)?;
        Ok(
            match (self.base_slots(source)?, other.base_slots(source)?) {
                (None, None) => {
                    excluded.range_cardinality(0..source.slot_count) < u64::from(source.slot_count)
                }
                (None, Some(base)) | (Some(base), None) => base.difference_len(&excluded) != 0,
                (Some(left), Some(right)) => {
                    let mut overlap = left & right;
                    overlap -= excluded;
                    !overlap.is_empty()
                }
            },
        )
    }
}

fn placement_effective_sources(
    placement: &RowAddressPlacement,
) -> Result<Vec<(RowAddressLogicalDomain, EffectiveSourceSelection)>> {
    match placement {
        RowAddressPlacement::Direct(value) => Ok(vec![(
            value.source,
            EffectiveSourceSelection::full(value.excluded.as_ref()),
        )]),
        RowAddressPlacement::PackedRun(value) => (0..value.domains.domain_count())
            .map(|ordinal| {
                let source = value.domains.domain_at(ordinal)?;
                Ok((source, EffectiveSourceSelection::full(None)))
            })
            .collect(),
        RowAddressPlacement::Selected(value) => Ok(vec![(
            value.source,
            EffectiveSourceSelection::selected(
                value.selection.as_ref().clone(),
                value.excluded.as_ref(),
            ),
        )]),
        RowAddressPlacement::ExtentList(value) => Ok(vec![(
            value.source,
            EffectiveSourceSelection::selected(
                LogicalRowAddressSelection::from_ranges(
                    value
                        .extents
                        .iter()
                        .map(|extent| {
                            LogicalRowAddressRange::new(
                                value.source.logical_fragment_id,
                                extent.source_start,
                                extent.source_start + extent.length,
                            )
                        })
                        .collect(),
                )?,
                None,
            ),
        )]),
        RowAddressPlacement::SparseSelection(value) => value
            .sources
            .iter()
            .map(|source| {
                Ok((
                    source.source,
                    EffectiveSourceSelection::selected(
                        source.selection.as_ref().clone(),
                        source.excluded.as_ref(),
                    ),
                ))
            })
            .collect(),
        RowAddressPlacement::ExplicitMap(value) => value
            .sources
            .iter()
            .map(|source| {
                Ok((
                    source.source,
                    EffectiveSourceSelection::selected(
                        source.selection.as_ref().clone(),
                        source.excluded.as_ref(),
                    ),
                ))
            })
            .collect(),
    }
}

fn validate_logical_source_ownership(
    placements: &[RowAddressPlacement],
    fragments: &[Fragment],
) -> Result<()> {
    let mut by_domain =
        BTreeMap::<u32, (RowAddressLogicalDomain, Vec<EffectiveSourceSelection>)>::new();
    let mut add =
        |source: RowAddressLogicalDomain, selection: EffectiveSourceSelection| -> Result<()> {
            if selection.is_empty(source)? {
                return Ok(());
            }
            let entry = by_domain
                .entry(source.logical_fragment_id)
                .or_insert_with(|| (source, Vec::new()));
            if entry.0 != source {
                return Err(Error::invalid_input(format!(
                    "logical domain {} has inconsistent source metadata",
                    source.logical_fragment_id
                )));
            }
            for existing in &entry.1 {
                if existing.overlaps(&selection, source)? {
                    return Err(Error::invalid_input(format!(
                        "logical domain {} is owned by overlapping placements",
                        source.logical_fragment_id
                    )));
                }
            }
            entry.1.push(selection);
            Ok(())
        };

    for placement in placements {
        for (source, selection) in placement_effective_sources(placement)? {
            add(source, selection)?;
        }
    }
    for fragment in fragments {
        let Some(native) = fragment.native_logical_domain else {
            continue;
        };
        let source = RowAddressLogicalDomain::new(
            native.logical_fragment_id,
            u32::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input("native logical domain is missing physical_rows")
            })?)
            .map_err(|_| Error::invalid_input("native logical domain rows exceed u32"))?,
            native.creation_version,
        )?;
        add(source, EffectiveSourceSelection::full(None))?;
    }
    Ok(())
}

fn build_destination_index(
    placements: &[RowAddressPlacement],
) -> Vec<RowAddressDestinationIndexEntry> {
    let mut destinations = BTreeMap::<u32, BTreeSet<u32>>::new();
    for (placement_index, placement) in placements.iter().enumerate() {
        for (fragment_id, _, _) in placement.destination_ranges() {
            destinations
                .entry(fragment_id)
                .or_default()
                .insert(placement_index as u32);
        }
    }
    destinations
        .into_iter()
        .map(
            |(physical_fragment_id, placement_indices)| RowAddressDestinationIndexEntry {
                physical_fragment_id,
                placement_indices: placement_indices.into_iter().collect(),
            },
        )
        .collect()
}

fn validate_native_domains(
    fragments: &[Fragment],
    placements: &[RowAddressPlacement],
) -> Result<()> {
    let mut native = BTreeMap::<u32, (u32, u64)>::new();
    for fragment in fragments {
        let Some(domain) = fragment.native_logical_domain else {
            continue;
        };
        let slot_count = u32::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input("native logical domain fragment is missing physical_rows")
        })?)
        .map_err(|_| Error::invalid_input("native logical domain row count exceeds u32"))?;
        NativeLogicalDomain::new(domain.logical_fragment_id, domain.creation_version)?;
        if slot_count == 0
            || native
                .insert(
                    domain.logical_fragment_id,
                    (slot_count, domain.creation_version),
                )
                .is_some()
        {
            return Err(Error::invalid_input(
                "native logical domains must be non-empty and uniquely owned",
            ));
        }
    }
    for placement in placements {
        placement.for_each_source(|source| {
            if let Some((slot_count, creation_version)) = native.get(&source.logical_fragment_id)
                && (*slot_count != source.slot_count
                    || *creation_version != source.creation_version)
            {
                return Err(Error::invalid_input(format!(
                    "native and relocated metadata disagree for logical domain {}",
                    source.logical_fragment_id
                )));
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn validate_native_logical_ownership(
    placements: &[RowAddressPlacement],
    fragments: &[Fragment],
) -> Result<()> {
    let native_domains = fragments
        .iter()
        .filter_map(|fragment| {
            fragment.native_logical_domain.map(|native| {
                let slot_count = u32::try_from(fragment.physical_rows.ok_or_else(|| {
                    Error::invalid_input("native logical domain fragment is missing physical_rows")
                })?)
                .map_err(|_| Error::invalid_input("native logical domain row count exceeds u32"))?;
                Ok((
                    native.logical_fragment_id,
                    RowAddressLogicalDomain::new(
                        native.logical_fragment_id,
                        slot_count,
                        native.creation_version,
                    )?,
                ))
            })
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    if native_domains.is_empty() {
        return Ok(());
    }
    for placement in placements {
        for (source, selection) in placement_effective_sources(placement)? {
            if let Some(native) = native_domains.get(&source.logical_fragment_id) {
                if native != &source {
                    return Err(Error::invalid_input(format!(
                        "native and relocated metadata disagree for logical domain {}",
                        source.logical_fragment_id
                    )));
                }
                if !selection.is_empty(source)? {
                    return Err(Error::invalid_input(format!(
                        "logical domain {} has both native and relocated ownership",
                        source.logical_fragment_id
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_native_destination_ownership(
    placements: &[RowAddressPlacement],
    fragments: &[Fragment],
) -> Result<()> {
    let native_fragments = fragments
        .iter()
        .filter(|fragment| fragment.native_logical_domain.is_some())
        .map(|fragment| physical_fragment_id(fragment.id, "native DataFragment"))
        .collect::<Result<BTreeSet<_>>>()?;
    if native_fragments.is_empty() {
        return Ok(());
    }
    for placement in placements {
        if let Some((fragment_id, _, _)) = placement
            .destination_ranges()
            .into_iter()
            .find(|(fragment_id, _, _)| native_fragments.contains(fragment_id))
        {
            return Err(Error::invalid_input(format!(
                "placement destination overlaps native physical fragment {fragment_id}"
            )));
        }
    }
    Ok(())
}

fn native_domain_routes(fragments: &[Fragment]) -> Result<Vec<(u32, u32, u32, u64)>> {
    let mut native = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        let Some(domain) = fragment.native_logical_domain else {
            continue;
        };
        let fragment_id = physical_fragment_id(fragment.id, "native DataFragment")?;
        let slot_count =
            u32::try_from(fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input("native DataFragment is missing physical_rows")
            })?)
            .map_err(|_| Error::invalid_input("native DataFragment row count exceeds u32"))?;
        native.push((
            domain.logical_fragment_id,
            fragment_id,
            slot_count,
            domain.creation_version,
        ));
    }
    native.sort_unstable();
    if native.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(Error::invalid_input(
            "logical domain has multiple native physical owners",
        ));
    }
    Ok(native)
}

fn merge_creation_version(
    current: &mut Option<u64>,
    candidate: u64,
    logical_fragment_id: u32,
) -> Result<()> {
    if current.is_some_and(|version| version != candidate) {
        return Err(Error::invalid_input(format!(
            "logical domain {} has inconsistent creation versions",
            logical_fragment_id
        )));
    }
    *current = Some(candidate);
    Ok(())
}

fn validate_field_ids(field_ids: &[i32], context: &str) -> Result<()> {
    if field_ids.is_empty()
        || field_ids.iter().any(|field_id| *field_id < 0)
        || field_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(Error::invalid_input(format!(
            "{context} must be non-empty, non-negative, and strictly sorted"
        )));
    }
    Ok(())
}

fn validate_field_generations(values: &[FieldGeneration], context: &str) -> Result<()> {
    if values
        .iter()
        .any(|value| value.field_id < 0 || value.generation == 0)
        || values
            .windows(2)
            .any(|pair| pair[0].field_id >= pair[1].field_id)
    {
        return Err(Error::invalid_input(format!(
            "{context} must have non-negative, strictly sorted field ids and non-zero generations"
        )));
    }
    Ok(())
}

fn selection_first_address(selection: &LogicalRowAddressSelection) -> u64 {
    selection
        .select(0)
        .ok()
        .flatten()
        .map(LogicalRowAddress::raw)
        .unwrap_or(u64::MAX)
}

impl TryFrom<i32> for RowReferenceDomain {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        match pb::RowReferenceDomain::try_from(value) {
            Ok(pb::RowReferenceDomain::PhysicalRowAddress) => Ok(Self::PhysicalRowAddress),
            Ok(pb::RowReferenceDomain::LegacyStableRowId) => Ok(Self::LegacyStableRowId),
            Ok(pb::RowReferenceDomain::StableLogicalRowAddress) => {
                Ok(Self::StableLogicalRowAddress)
            }
            Ok(pb::RowReferenceDomain::Unspecified) | Err(_) => Err(Error::invalid_input(format!(
                "invalid or unspecified row reference domain: {value}"
            ))),
        }
    }
}

impl From<RowReferenceDomain> for i32 {
    fn from(value: RowReferenceDomain) -> Self {
        match value {
            RowReferenceDomain::PhysicalRowAddress => {
                pb::RowReferenceDomain::PhysicalRowAddress as Self
            }
            RowReferenceDomain::LegacyStableRowId => {
                pb::RowReferenceDomain::LegacyStableRowId as Self
            }
            RowReferenceDomain::StableLogicalRowAddress => {
                pb::RowReferenceDomain::StableLogicalRowAddress as Self
            }
        }
    }
}

impl TryFrom<pb::LogicalIndexCoverage> for LogicalIndexCoverage {
    type Error = Error;

    fn try_from(value: pb::LogicalIndexCoverage) -> Result<Self> {
        let coverage = Self {
            shards: value
                .shards
                .into_iter()
                .map(|shard| {
                    let selection = shard
                        .selection
                        .map(LogicalRowAddressSelection::try_from)
                        .transpose()?;
                    let excluded_selection = shard
                        .excluded_selection
                        .map(LogicalRowAddressSelection::try_from)
                        .transpose()?;
                    let validated_through = shard
                        .validated_through
                        .into_iter()
                        .map(FieldGeneration::try_from)
                        .collect::<Result<Vec<_>>>()?;
                    validate_field_ids(&shard.field_ids, "LogicalIndexCoverageShard.field_ids")?;
                    validate_field_generations(
                        &validated_through,
                        "LogicalIndexCoverageShard.validated_through",
                    )?;
                    if selection
                        .as_ref()
                        .is_some_and(|selection| shard.row_count != selection.cardinality())
                        || shard.fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                    {
                        return Err(Error::invalid_input(
                            "logical index coverage shard row_count or fingerprint is invalid",
                        ));
                    }
                    Ok(LogicalIndexCoverageShard {
                        selection,
                        field_ids: shard.field_ids,
                        validated_through,
                        fingerprint: shard.fingerprint,
                        row_count: shard.row_count,
                        logical_fragment_bitmap: shard.logical_fragment_bitmap,
                        excluded_selection,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            external: value
                .external
                .map(|external| -> Result<LogicalIndexCoverageFile> {
                    Ok(LogicalIndexCoverageFile {
                        path: external.path,
                        offset: external.offset,
                        byte_length: external.byte_length,
                        global_buffer_index: external.global_buffer_index.ok_or_else(|| {
                            Error::invalid_input(
                                "LogicalIndexCoverageFile.global_buffer_index is missing",
                            )
                        })?,
                        object_size: external.object_size.ok_or_else(|| {
                            Error::invalid_input("LogicalIndexCoverageFile.object_size is missing")
                        })?,
                        object_id: external
                            .object_id
                            .as_ref()
                            .ok_or_else(|| {
                                Error::invalid_input(
                                    "LogicalIndexCoverageFile.object_id is missing",
                                )
                            })
                            .and_then(Uuid::try_from)?,
                        artifact_namespace_uuid: external
                            .artifact_namespace_uuid
                            .as_ref()
                            .ok_or_else(|| {
                                Error::invalid_input(
                                    "LogicalIndexCoverageFile.artifact_namespace_uuid is missing",
                                )
                            })
                            .and_then(Uuid::try_from)?,
                        artifact_layout_fingerprint: external.artifact_layout_fingerprint,
                    })
                })
                .transpose()?,
            fingerprint: value.fingerprint,
            full_domain_fingerprint: if value.full_domain_fingerprint.is_empty() {
                None
            } else {
                Some(Box::new(
                    value.full_domain_fingerprint.try_into().map_err(
                        |fingerprint: Vec<u8>| {
                            Error::invalid_input(format!(
                                "LogicalIndexCoverage.full_domain_fingerprint requires {} bytes, got {}",
                                ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE,
                                fingerprint.len()
                            ))
                        },
                    )?,
                ))
            },
            clone_provenance: value
                .clone_provenance
                .map(|provenance| -> Result<LogicalIndexCoverageCloneProvenance> {
                    Ok(LogicalIndexCoverageCloneProvenance {
                        source_namespace_uuid: provenance
                            .source_namespace_uuid
                            .as_ref()
                            .ok_or_else(|| {
                                Error::invalid_input(
                                    "LogicalIndexCoverageCloneProvenance.source_namespace_uuid is missing",
                                )
                            })
                            .and_then(Uuid::try_from)?,
                        target_namespace_uuid: provenance
                            .target_namespace_uuid
                            .as_ref()
                            .ok_or_else(|| {
                                Error::invalid_input(
                                    "LogicalIndexCoverageCloneProvenance.target_namespace_uuid is missing",
                                )
                            })
                            .and_then(Uuid::try_from)?,
                        source_coverage_fingerprint: provenance.source_coverage_fingerprint,
                        transaction_uuid: provenance
                            .transaction_uuid
                            .as_ref()
                            .ok_or_else(|| {
                                Error::invalid_input(
                                    "LogicalIndexCoverageCloneProvenance.transaction_uuid is missing",
                                )
                            })
                            .and_then(Uuid::try_from)?,
                        depth: provenance.depth,
                        is_shallow: provenance.is_shallow,
                        source_manifest_version: provenance.source_manifest_version,
                    })
                })
                .transpose()?,
        };
        if coverage.fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
            || coverage
                .external
                .as_ref()
                .is_some_and(|external| external.path.is_empty() || external.byte_length == 0)
        {
            return Err(Error::invalid_input(
                "logical index coverage requires a 16-byte fingerprint and valid external metadata",
            ));
        }
        Ok(coverage)
    }
}

impl From<&LogicalIndexCoverage> for pb::LogicalIndexCoverage {
    fn from(value: &LogicalIndexCoverage) -> Self {
        Self {
            shards: value
                .shards
                .iter()
                .map(|shard| pb::LogicalIndexCoverageShard {
                    selection: shard.selection.as_ref().map(Into::into),
                    field_ids: shard.field_ids.clone(),
                    validated_through: shard.validated_through.iter().map(Into::into).collect(),
                    fingerprint: shard.fingerprint.clone(),
                    row_count: shard.row_count,
                    logical_fragment_bitmap: shard.logical_fragment_bitmap.clone(),
                    excluded_selection: shard.excluded_selection.as_ref().map(Into::into),
                })
                .collect(),
            external: value
                .external
                .as_ref()
                .map(|external| pb::LogicalIndexCoverageFile {
                    path: external.path.clone(),
                    offset: external.offset,
                    byte_length: external.byte_length,
                    global_buffer_index: Some(external.global_buffer_index),
                    object_size: Some(external.object_size),
                    object_id: Some((&external.object_id).into()),
                    artifact_namespace_uuid: Some((&external.artifact_namespace_uuid).into()),
                    artifact_layout_fingerprint: external.artifact_layout_fingerprint.clone(),
                }),
            fingerprint: value.fingerprint.clone(),
            full_domain_fingerprint: value
                .full_domain_fingerprint
                .as_deref()
                .map_or_else(Vec::new, |fingerprint| fingerprint.to_vec()),
            clone_provenance: value.clone_provenance.as_ref().map(|provenance| {
                pb::LogicalIndexCoverageCloneProvenance {
                    source_namespace_uuid: Some((&provenance.source_namespace_uuid).into()),
                    target_namespace_uuid: Some((&provenance.target_namespace_uuid).into()),
                    source_coverage_fingerprint: provenance.source_coverage_fingerprint.clone(),
                    transaction_uuid: Some((&provenance.transaction_uuid).into()),
                    depth: provenance.depth,
                    is_shallow: provenance.is_shallow,
                    source_manifest_version: provenance.source_manifest_version,
                }
            }),
        }
    }
}

impl TryFrom<i32> for RowAddressPlacementKind {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        match pb::transaction::RowAddressPlacementKind::try_from(value) {
            Ok(pb::transaction::RowAddressPlacementKind::Direct) => Ok(Self::Direct),
            Ok(pb::transaction::RowAddressPlacementKind::PackedRun) => Ok(Self::PackedRun),
            Ok(pb::transaction::RowAddressPlacementKind::Selected) => Ok(Self::Selected),
            Ok(pb::transaction::RowAddressPlacementKind::ExtentList) => Ok(Self::ExtentList),
            Ok(pb::transaction::RowAddressPlacementKind::SparseSelection) => {
                Ok(Self::SparseSelection)
            }
            Ok(pb::transaction::RowAddressPlacementKind::ExplicitMap) => Ok(Self::ExplicitMap),
            Ok(pb::transaction::RowAddressPlacementKind::Unspecified) | Err(_) => Err(
                Error::invalid_input(format!("invalid row-address placement kind: {value}")),
            ),
        }
    }
}

impl From<RowAddressPlacementKind> for i32 {
    fn from(value: RowAddressPlacementKind) -> Self {
        match value {
            RowAddressPlacementKind::Direct => {
                pb::transaction::RowAddressPlacementKind::Direct as Self
            }
            RowAddressPlacementKind::PackedRun => {
                pb::transaction::RowAddressPlacementKind::PackedRun as Self
            }
            RowAddressPlacementKind::Selected => {
                pb::transaction::RowAddressPlacementKind::Selected as Self
            }
            RowAddressPlacementKind::ExtentList => {
                pb::transaction::RowAddressPlacementKind::ExtentList as Self
            }
            RowAddressPlacementKind::SparseSelection => {
                pb::transaction::RowAddressPlacementKind::SparseSelection as Self
            }
            RowAddressPlacementKind::ExplicitMap => {
                pb::transaction::RowAddressPlacementKind::ExplicitMap as Self
            }
        }
    }
}

impl TryFrom<pb::transaction::RowAddressLayoutDelta> for RowAddressLayoutDelta {
    type Error = Error;

    fn try_from(value: pb::transaction::RowAddressLayoutDelta) -> Result<Self> {
        let delta = Self {
            source_domains: value
                .source_domains
                .into_iter()
                .map(RowAddressLogicalDomain::try_from)
                .collect::<Result<Vec<_>>>()?,
            placements: value
                .placements
                .into_iter()
                .map(|placement| {
                    let target = required(placement.target, "RowAddressPlacementDelta.target")?;
                    let fragment = match required(
                        target.fragment,
                        "RowAddressTargetRange.fragment",
                    )? {
                        pb::transaction::row_address_target_range::Fragment::NewFragmentOrdinal(
                            ordinal,
                        ) => RowAddressTargetFragment::NewFragmentOrdinal(ordinal),
                        pb::transaction::row_address_target_range::Fragment::ExistingFragmentId(
                            fragment_id,
                        ) => RowAddressTargetFragment::ExistingFragmentId(physical_fragment_id(
                            fragment_id,
                            "RowAddressTargetRange",
                        )?),
                    };
                    let placement = RowAddressPlacementDelta {
                        source_selections: placement
                            .source_selections
                            .into_iter()
                            .map(LogicalRowAddressSelection::try_from)
                            .collect::<Result<Vec<_>>>()?,
                        target: RowAddressTargetRange {
                            fragment,
                            start_offset: target.start_offset,
                            end_offset: target.end_offset,
                        },
                        placement_kind: placement.placement_kind.try_into()?,
                        output_cardinality: placement.output_cardinality,
                        output_row_sequence_fingerprint: placement.output_row_sequence_fingerprint,
                    };
                    if placement.target.start_offset >= placement.target.end_offset
                        || (placement.placement_kind != RowAddressPlacementKind::ExplicitMap
                            && placement.output_cardinality
                                != (placement.target.end_offset - placement.target.start_offset)
                                    as u64)
                    {
                        return Err(Error::invalid_input(
                            "row-address placement delta target and output cardinality disagree",
                        ));
                    }
                    let source_cardinality = placement
                        .source_selections
                        .iter()
                        .map(LogicalRowAddressSelection::cardinality)
                        .sum::<u64>();
                    if !placement.source_selections.is_empty()
                        && placement.placement_kind != RowAddressPlacementKind::ExplicitMap
                        && source_cardinality != placement.output_cardinality
                    {
                        return Err(Error::invalid_input(
                            "row-address placement delta source and output cardinalities disagree",
                        ));
                    }
                    match placement.placement_kind {
                        RowAddressPlacementKind::Direct
                            if !placement.output_row_sequence_fingerprint.is_empty() =>
                        {
                            return Err(Error::invalid_input(
                                "Direct append must not carry a row-sequence fingerprint",
                            ));
                        }
                        RowAddressPlacementKind::Direct => {}
                        _ if placement.output_row_sequence_fingerprint.len()
                            != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE =>
                        {
                            return Err(Error::invalid_input(format!(
                                "non-Direct placement row-sequence fingerprint must be {} bytes",
                                ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
                            )));
                        }
                        _ => {}
                    }
                    Ok(placement)
                })
                .collect::<Result<Vec<_>>>()?,
            retired_selections: value
                .retired_selections
                .into_iter()
                .map(LogicalRowAddressSelection::try_from)
                .collect::<Result<Vec<_>>>()?,
            field_changes: value
                .field_changes
                .into_iter()
                .map(|change| {
                    validate_field_ids(&change.field_ids, "RowAddressFieldChange.field_ids")?;
                    Ok(RowAddressFieldChange {
                        selection: required(change.selection, "RowAddressFieldChange.selection")?
                            .try_into()?,
                        field_ids: change.field_ids,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            source_floors: value
                .source_floors
                .into_iter()
                .map(|floor| {
                    if floor.field_id < 0 || floor.generation == 0 {
                        return Err(Error::invalid_input(
                            "row-address source floor has invalid field id or generation",
                        ));
                    }
                    Ok(RowAddressSourceFloor {
                        field_id: floor.field_id,
                        generation: floor.generation,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            expected_layout_fingerprint: value.expected_layout_fingerprint,
            replaced_generations: value
                .replaced_generations
                .into_iter()
                .map(|generation| {
                    validate_field_ids(
                        &generation.field_ids,
                        "ReplacedContentGeneration.field_ids",
                    )?;
                    if generation.generation == 0 {
                        return Err(Error::invalid_input(
                            "replaced content generation must be non-zero",
                        ));
                    }
                    Ok(ReplacedContentGeneration {
                        selection: required(
                            generation.selection,
                            "ReplacedContentGeneration.selection",
                        )?
                        .try_into()?,
                        field_ids: generation.field_ids,
                        generation: generation.generation,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            row_aligned_rewrite_proofs: value
                .row_aligned_rewrite_proofs
                .into_iter()
                .map(|proof| {
                    Ok(RowAlignedRewriteProof {
                        physical_fragment_id: physical_fragment_id(
                            proof.physical_fragment_id,
                            "RowAlignedRewriteProof",
                        )?,
                        physical_rows: u32::try_from(proof.physical_rows).map_err(|_| {
                            Error::format_capacity_exceeded(format!(
                                "row-aligned rewrite proof physical_rows {} exceeds u32",
                                proof.physical_rows
                            ))
                        })?,
                        mapped_offsets_fingerprint: proof.mapped_offsets_fingerprint,
                        deletion_offsets_fingerprint: proof.deletion_offsets_fingerprint,
                        field_change_index: proof.field_change_index as usize,
                        source_floor_indices: proof
                            .source_floor_indices
                            .into_iter()
                            .map(|index| index as usize)
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            create_namespace_uuid: value
                .create_namespace_uuid
                .as_ref()
                .map(Uuid::try_from)
                .transpose()?,
            explicit_map_placements: value
                .explicit_map_placements
                .into_iter()
                .map(|entry| {
                    let placement_index = entry.placement_delta_index as usize;
                    let domains = entry
                        .sources
                        .into_iter()
                        .map(RowAddressLogicalDomain::try_from)
                        .collect::<Result<Vec<_>>>()?;
                    validate_strictly_sorted_domains(&domains)?;
                    let sources = domains
                        .iter()
                        .copied()
                        .map(|source| {
                            Ok(SparseSelectionSource {
                                source,
                                selection: Arc::new(LogicalRowAddressSelection::from_full_domains(
                                    &[source],
                                )?),
                                excluded: None,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let placement = ExplicitMapRowAddressPlacement {
                        sources,
                        object_path: entry.object_path,
                        object_size: entry.object_size,
                        pages: entry
                            .pages
                            .into_iter()
                            .map(|page| ExplicitMapPage {
                                first_logical_address: page.first_logical_address,
                                last_logical_address: page.last_logical_address,
                                row_start: page.row_start,
                                row_count: page.row_count,
                                content_fingerprint: page.content_fingerprint,
                            })
                            .collect(),
                        destinations: entry
                            .destinations
                            .into_iter()
                            .map(|destination| {
                                Ok(ExplicitMapDestination {
                                    physical_fragment_id: physical_fragment_id(
                                        destination.physical_fragment_id,
                                        "ExplicitMapPlacementDelta destination",
                                    )?,
                                    destination_start: destination.destination_start,
                                    row_count: destination.row_count,
                                    row_id_file_path: destination.row_id_file_path,
                                    row_id_file_size: destination.row_id_file_size,
                                    row_id_pages: destination
                                        .row_id_pages
                                        .into_iter()
                                        .map(|page| ExplicitMapRowIdPage {
                                            row_start: page.row_start,
                                            row_count: page.row_count,
                                            content_fingerprint: page.content_fingerprint,
                                        })
                                        .collect(),
                                })
                            })
                            .collect::<Result<Vec<_>>>()?,
                        base_id: entry.base_id,
                    };
                    RowAddressPlacement::ExplicitMap(placement.clone()).validate()?;
                    Ok((placement_index, placement))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        };
        if delta.create_namespace_uuid.is_some() {
            if !delta.expected_layout_fingerprint.is_empty() {
                return Err(Error::invalid_input(
                    "create row-address delta must not have an expected layout fingerprint",
                ));
            }
        } else if delta.expected_layout_fingerprint.len() != ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE {
            return Err(Error::invalid_input(format!(
                "successor row-address delta expected fingerprint must be {} bytes",
                ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE
            )));
        }
        delta.validate_row_aligned_rewrite_proofs()?;
        Ok(delta)
    }
}

fn placement_delta_to_proto(
    placement: &RowAddressPlacementDelta,
) -> pb::transaction::RowAddressPlacementDelta {
    let fragment = match placement.target.fragment {
        RowAddressTargetFragment::NewFragmentOrdinal(ordinal) => {
            pb::transaction::row_address_target_range::Fragment::NewFragmentOrdinal(ordinal)
        }
        RowAddressTargetFragment::ExistingFragmentId(fragment_id) => {
            pb::transaction::row_address_target_range::Fragment::ExistingFragmentId(
                fragment_id as u64,
            )
        }
    };
    pb::transaction::RowAddressPlacementDelta {
        source_selections: placement.source_selections.iter().map(Into::into).collect(),
        target: Some(pb::transaction::RowAddressTargetRange {
            fragment: Some(fragment),
            start_offset: placement.target.start_offset,
            end_offset: placement.target.end_offset,
        }),
        placement_kind: placement.placement_kind.into(),
        output_cardinality: placement.output_cardinality,
        output_row_sequence_fingerprint: placement.output_row_sequence_fingerprint.clone(),
    }
}

impl From<&RowAddressLayoutDelta> for pb::transaction::RowAddressLayoutDelta {
    fn from(value: &RowAddressLayoutDelta) -> Self {
        Self {
            source_domains: value.source_domains.iter().map(Into::into).collect(),
            placements: value
                .placements
                .iter()
                .map(placement_delta_to_proto)
                .collect(),
            retired_selections: value.retired_selections.iter().map(Into::into).collect(),
            field_changes: value
                .field_changes
                .iter()
                .map(|change| pb::transaction::RowAddressFieldChange {
                    selection: Some((&change.selection).into()),
                    field_ids: change.field_ids.clone(),
                })
                .collect(),
            source_floors: value
                .source_floors
                .iter()
                .map(|floor| pb::transaction::RowAddressSourceFloor {
                    field_id: floor.field_id,
                    generation: floor.generation,
                })
                .collect(),
            expected_layout_fingerprint: value.expected_layout_fingerprint.clone(),
            replaced_generations: value
                .replaced_generations
                .iter()
                .map(|generation| pb::transaction::ReplacedContentGeneration {
                    selection: Some((&generation.selection).into()),
                    field_ids: generation.field_ids.clone(),
                    generation: generation.generation,
                })
                .collect(),
            row_aligned_rewrite_proofs: value
                .row_aligned_rewrite_proofs
                .iter()
                .map(|proof| pb::transaction::RowAlignedRewriteProof {
                    physical_fragment_id: proof.physical_fragment_id as u64,
                    physical_rows: proof.physical_rows as u64,
                    mapped_offsets_fingerprint: proof.mapped_offsets_fingerprint.clone(),
                    deletion_offsets_fingerprint: proof.deletion_offsets_fingerprint.clone(),
                    field_change_index: proof.field_change_index as u32,
                    source_floor_indices: proof
                        .source_floor_indices
                        .iter()
                        .map(|index| *index as u32)
                        .collect(),
                })
                .collect(),
            create_namespace_uuid: value.create_namespace_uuid.as_ref().map(Into::into),
            explicit_map_placements: value
                .explicit_map_placements
                .iter()
                .map(|(placement_delta_index, placement)| {
                    pb::transaction::ExplicitMapPlacementDelta {
                        placement_delta_index: *placement_delta_index as u32,
                        sources: placement
                            .sources
                            .iter()
                            .map(|source| (&source.source).into())
                            .collect(),
                        object_path: placement.object_path.clone(),
                        object_size: placement.object_size,
                        pages: placement
                            .pages
                            .iter()
                            .map(|page| pb::ExplicitMapPage {
                                first_logical_address: page.first_logical_address,
                                last_logical_address: page.last_logical_address,
                                row_start: page.row_start,
                                row_count: page.row_count,
                                content_fingerprint: page.content_fingerprint.clone(),
                            })
                            .collect(),
                        destinations: placement
                            .destinations
                            .iter()
                            .map(|destination| pb::ExplicitMapDestination {
                                physical_fragment_id: destination.physical_fragment_id as u64,
                                destination_start: destination.destination_start,
                                row_count: destination.row_count,
                                row_id_file_path: destination.row_id_file_path.clone(),
                                row_id_file_size: destination.row_id_file_size,
                                row_id_pages: destination
                                    .row_id_pages
                                    .iter()
                                    .map(|page| pb::ExplicitMapRowIdPage {
                                        row_start: page.row_start,
                                        row_count: page.row_count,
                                        content_fingerprint: page.content_fingerprint.clone(),
                                    })
                                    .collect(),
                            })
                            .collect(),
                        base_id: placement.base_id,
                    }
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use lance_core::datatypes::Schema;
    use lance_file::version::LanceFileVersion;

    use super::*;
    use crate::format::{DataStorageFormat, Manifest};

    fn domain(id: u32, slot_count: u32) -> RowAddressLogicalDomain {
        RowAddressLogicalDomain::new(id, slot_count, 1).unwrap()
    }

    fn selection(id: u32, slots: impl IntoIterator<Item = u32>) -> LogicalRowAddressSelection {
        let bitmap = slots
            .into_iter()
            .map(|slot| ((id as u64) << 32) | slot as u64)
            .collect::<RoaringTreemap>();
        LogicalRowAddressSelection::from_bitmap(bitmap).unwrap()
    }

    fn streamed_addresses(selection: &LogicalRowAddressSelection) -> Vec<LogicalRowAddress> {
        let mut addresses = Vec::with_capacity(selection.cardinality() as usize);
        selection
            .try_for_each_address(|address| {
                addresses.push(address);
                Ok(())
            })
            .unwrap();
        addresses
    }

    fn fragment(id: u64, rows: usize, native: Option<(u32, u64)>) -> Fragment {
        let mut fragment = Fragment::new(id).with_physical_rows(rows);
        fragment.native_logical_domain = native.map(|(logical_fragment_id, creation_version)| {
            NativeLogicalDomain::new(logical_fragment_id, creation_version).unwrap()
        });
        fragment
    }

    fn with_deletions(
        mut fragment: Fragment,
        id: u64,
        deleted_offsets: &RoaringBitmap,
    ) -> Fragment {
        fragment.deletion_file = Some(crate::format::DeletionFile {
            read_version: 1,
            id,
            file_type: crate::format::DeletionFileType::Bitmap,
            num_deleted_rows: Some(deleted_offsets.len() as usize),
            base_id: None,
        });
        fragment
    }

    fn placement_delta(
        layout: &RowAddressLayout,
        source: RowAddressLogicalDomain,
        slots: impl IntoIterator<Item = u32>,
        target: RowAddressTargetRange,
        placement_kind: RowAddressPlacementKind,
    ) -> RowAddressLayoutDelta {
        let slots = slots.into_iter().collect::<Vec<_>>();
        let source_selection = selection(source.logical_fragment_id, slots.iter().copied());
        let addresses = slots
            .iter()
            .map(|slot| {
                LogicalRowAddress::try_new_from_parts(source.logical_fragment_id, *slot).unwrap()
            })
            .collect::<Vec<_>>();
        RowAddressLayoutDelta {
            source_domains: vec![source],
            placements: vec![RowAddressPlacementDelta {
                source_selections: vec![source_selection],
                target,
                placement_kind,
                output_cardinality: slots.len() as u64,
                output_row_sequence_fingerprint: fingerprint_row_sequence(target, addresses)
                    .unwrap(),
            }],
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        }
    }

    #[test]
    fn adaptive_selection_round_trip_and_sparse_size() {
        let mut bitmap = RoaringTreemap::new();
        for index in 0_u64..1_000_000 {
            let slot = (index * 99_991) % 100_000_000;
            bitmap.insert((7_u64 << 32) | slot);
        }
        let selection = LogicalRowAddressSelection::from_bitmap(bitmap).unwrap();
        let proto = selection.canonical_proto();
        assert!(
            proto.encoded_len() <= 1_310_720,
            "1% sparse selection encoded to {} bytes",
            proto.encoded_len()
        );
        let decoded = LogicalRowAddressSelection::try_from(proto).unwrap();
        assert_eq!(selection, decoded);
    }

    #[test]
    fn selection_streaming_matches_select_for_every_codec() {
        let mut bitmap = RoaringTreemap::new();
        for logical_fragment_id in [3_u32, 9] {
            for slot in (0..1_024_u32).filter(|slot| slot % 3 != 1) {
                bitmap.insert((u64::from(logical_fragment_id) << 32) | u64::from(slot));
            }
        }
        let range_input = SelectionBuilderInput::Ranges(vec![
            LogicalRowAddressRange::new(3, 2, 7),
            LogicalRowAddressRange::new(9, 11, 15),
        ]);
        let codecs = [
            ("ranges", range_input.range_candidate().unwrap()),
            (
                "dense",
                SelectionBuilderInput::dense_candidate(&bitmap).unwrap(),
            ),
            (
                "elias-fano",
                SelectionBuilderInput::elias_fano_candidate(&bitmap).unwrap(),
            ),
            (
                "ordinal-elias-fano",
                SelectionBuilderInput::ordinal_elias_fano_candidate(&bitmap).unwrap(),
            ),
            (
                "roaring",
                SelectionBuilderInput::roaring_candidate(&bitmap).unwrap(),
            ),
        ];

        for (codec, encoded) in codecs {
            let selection = LogicalRowAddressSelection::try_from(encoded).unwrap();
            let selected = selection.iter().collect::<Result<Vec<_>>>().unwrap();
            assert_eq!(streamed_addresses(&selection), selected, "codec={codec}");
        }
    }

    #[test]
    fn high_entropy_selection_streams_in_one_payload_pass() {
        const UNIVERSE: u32 = 1_000_000;
        let bitmap = (0..UNIVERSE)
            .step_by(5)
            .map(|slot| (17_u64 << 32) | u64::from(slot))
            .collect::<RoaringTreemap>();
        let expected_count = bitmap.len();
        let expected_checksum = bitmap
            .iter()
            .fold(0_u64, |checksum, raw| checksum.wrapping_add(raw));
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: expected_count as u32,
        };
        let expected_fingerprint = fingerprint_row_sequence(
            target,
            bitmap
                .iter()
                .map(|raw| LogicalRowAddress::try_from(raw).unwrap()),
        )
        .unwrap();
        let codecs = [
            SelectionBuilderInput::dense_candidate(&bitmap).unwrap(),
            SelectionBuilderInput::elias_fano_candidate(&bitmap).unwrap(),
            SelectionBuilderInput::roaring_candidate(&bitmap).unwrap(),
        ];

        for encoded in codecs {
            let selection = LogicalRowAddressSelection::try_from(encoded).unwrap();
            let mut count = 0_u64;
            let mut checksum = 0_u64;
            let mut previous = None;
            selection
                .try_for_each_address(|address| {
                    assert!(previous.is_none_or(|previous| previous < address));
                    previous = Some(address);
                    count += 1;
                    checksum = checksum.wrapping_add(address.raw());
                    Ok(())
                })
                .unwrap();
            assert_eq!(count, expected_count);
            assert_eq!(checksum, expected_checksum);
            let delta = RowAddressPlacementDelta {
                source_selections: vec![selection],
                target,
                placement_kind: RowAddressPlacementKind::Selected,
                output_cardinality: expected_count,
                output_row_sequence_fingerprint: Vec::new(),
            };
            assert_eq!(
                delta.expected_row_sequence_fingerprint().unwrap(),
                expected_fingerprint
            );
        }
    }

    #[test]
    fn high_entropy_exclusions_stream_mapped_ranges() {
        const ROWS: u32 = 10_000;
        let collect_ranges = |placement: &RowAddressPlacement| {
            let mut ranges = Vec::new();
            placement
                .for_each_mapped_destination_range(|fragment_id, start, length| {
                    ranges.push((fragment_id, start, length));
                    Ok(())
                })
                .unwrap();
            ranges
        };

        let selected = Arc::new(selection(0, 0..ROWS));
        let excluded = Arc::new(selection(0, (0..ROWS).step_by(2)));
        assert!(
            excluded
                .compact_ranges(MAX_RANGE_ENCODING_RUNS)
                .unwrap()
                .is_none()
        );
        let direct = RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: domain(0, ROWS),
            destination_fragment_id: 10,
            destination_start: 100,
            excluded: Some(excluded.clone()),
        });
        let direct_ranges = collect_ranges(&direct);
        assert_eq!(direct_ranges.len(), ROWS as usize / 2);
        assert_eq!(direct_ranges.first(), Some(&(10, 101, 1)));
        assert_eq!(direct_ranges.last(), Some(&(10, 100 + ROWS - 1, 1)));

        let selected_placement = RowAddressPlacement::Selected(SelectedRowAddressPlacement {
            source: domain(0, ROWS),
            selection: selected,
            destination_fragment_id: 10,
            destination_start: 100,
            excluded: Some(excluded),
        });
        assert_eq!(collect_ranges(&selected_placement), direct_ranges);

        let sparse = RowAddressPlacement::SparseSelection(SparseSelectionRowAddressPlacement {
            sources: (0..2)
                .map(|logical_fragment_id| SparseSelectionSource {
                    source: domain(logical_fragment_id, ROWS),
                    selection: Arc::new(selection(logical_fragment_id, 0..ROWS)),
                    excluded: Some(Arc::new(selection(
                        logical_fragment_id,
                        (0..ROWS).step_by(2),
                    ))),
                })
                .collect(),
            destination_fragment_id: 11,
            destination_start: 100,
        });
        let sparse_ranges = collect_ranges(&sparse);
        assert_eq!(sparse_ranges.len(), ROWS as usize);
        assert_eq!(sparse_ranges.first(), Some(&(11, 101, 1)));
        assert_eq!(sparse_ranges.last(), Some(&(11, 100 + ROWS * 2 - 1, 1)));
    }

    #[test]
    fn ordinal_elias_fano_stays_bounded_across_many_domains() {
        for domain_count in [10_000_u32, 100_000] {
            let mut bitmap = RoaringTreemap::new();
            for logical_fragment_id in 0..domain_count {
                for index in 0..10_u32 {
                    let slot =
                        (logical_fragment_id.wrapping_mul(31) + index.wrapping_mul(97)) % 997;
                    bitmap.insert(((logical_fragment_id as u64) << 32) | slot as u64);
                }
            }
            let selection = LogicalRowAddressSelection::from_bitmap(bitmap).unwrap();
            let proto = selection.canonical_proto();
            assert!(
                proto.encoded_len() <= 1_310_720,
                "{} domains encoded to {} bytes",
                domain_count,
                proto.encoded_len()
            );
            assert!(matches!(
                proto.encoding,
                Some(pb::logical_row_address_selection::Encoding::OrdinalEliasFano(_))
            ));
            let decoded = LogicalRowAddressSelection::try_from(proto).unwrap();
            assert_eq!(selection, decoded);
            let mut previous = None;
            let mut streamed = 0_u64;
            decoded
                .try_for_each_address(|address| {
                    assert!(previous.is_none_or(|previous| previous < address));
                    previous = Some(address);
                    streamed += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(streamed, u64::from(domain_count) * 10);
        }
    }

    #[test]
    fn packed_run_uniform_payload_is_constant_size() {
        for domain_count in [10_000_u32, 100_000] {
            let placement = RowAddressPlacement::PackedRun(
                PackedRunRowAddressPlacement::from_sources(
                    (0..domain_count).map(|id| domain(id, 1_000)).collect(),
                    42,
                    0,
                )
                .unwrap(),
            );
            let proto: pb::RowAddressPlacement = (&placement).into();
            assert!(
                proto.encoded_len() <= 64,
                "{} uniform domains encoded to {} bytes",
                domain_count,
                proto.encoded_len()
            );
            assert_eq!(RowAddressPlacement::try_from(proto).unwrap(), placement);
        }
    }

    #[test]
    fn apply_delta_packed_run_100m_uses_domain_algebra() {
        const DOMAIN_COUNT: u32 = 10_000;
        const ROWS_PER_DOMAIN: u32 = 10_000;
        const TOTAL_ROWS: u32 = DOMAIN_COUNT * ROWS_PER_DOMAIN;

        let domains = (0..DOMAIN_COUNT)
            .map(|logical_fragment_id| domain(logical_fragment_id, ROWS_PER_DOMAIN))
            .collect::<Vec<_>>();
        let current_fragments = (0..DOMAIN_COUNT)
            .map(|fragment_id| {
                fragment(
                    u64::from(fragment_id),
                    ROWS_PER_DOMAIN as usize,
                    Some((fragment_id, 1)),
                )
            })
            .collect::<Vec<_>>();
        let successor_fragments =
            vec![fragment(u64::from(DOMAIN_COUNT), TOTAL_ROWS as usize, None)];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(&current_fragments, Some(DOMAIN_COUNT - 1));

        let source_selection = LogicalRowAddressSelection::from_full_domains(&domains).unwrap();
        assert!(matches!(
            &source_selection,
            LogicalRowAddressSelection::Roaring(_)
        ));
        assert_eq!(
            source_selection.to_ranges().unwrap().len(),
            DOMAIN_COUNT as usize
        );
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: TOTAL_ROWS,
        };
        let mut placement = RowAddressPlacementDelta {
            source_selections: vec![source_selection],
            target,
            placement_kind: RowAddressPlacementKind::PackedRun,
            output_cardinality: u64::from(TOTAL_ROWS),
            output_row_sequence_fingerprint: Vec::new(),
        };
        placement.output_row_sequence_fingerprint =
            placement.expected_row_sequence_fingerprint().unwrap();
        let delta = RowAddressLayoutDelta {
            source_domains: domains.clone(),
            placements: vec![placement],
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        };
        let resolved = BTreeMap::from([(0, DOMAIN_COUNT)]);
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &current_fragments,
                    successor_fragments: &successor_fragments,
                    resolved_new_fragment_ids: &resolved,
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(DOMAIN_COUNT - 1),
                    max_logical_fragment_id: Some(DOMAIN_COUNT - 1),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
            panic!("full-domain PackedRun relocation should be admitted");
        };
        assert_eq!(layout.placements.len(), 1);
        let RowAddressPlacement::PackedRun(packed) = &layout.placements[0] else {
            panic!("full-domain relocation should stay a PackedRun");
        };
        assert_eq!(packed.domains.domain_count(), DOMAIN_COUNT);
        assert_eq!(
            packed.domains.total_slot_count().unwrap(),
            u64::from(TOTAL_ROWS)
        );
        let proto: pb::RowAddressPlacement = (&layout.placements[0]).into();
        assert!(proto.encoded_len() <= 64);

        let second_target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: TOTAL_ROWS,
        };
        let mut second_placement = RowAddressPlacementDelta {
            source_selections: vec![
                LogicalRowAddressSelection::from_full_domains(&domains).unwrap(),
            ],
            target: second_target,
            placement_kind: RowAddressPlacementKind::PackedRun,
            output_cardinality: u64::from(TOTAL_ROWS),
            output_row_sequence_fingerprint: Vec::new(),
        };
        second_placement.output_row_sequence_fingerprint = second_placement
            .expected_row_sequence_fingerprint()
            .unwrap();
        let second_delta = RowAddressLayoutDelta {
            source_domains: domains,
            placements: vec![second_placement],
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        };
        let second_successor = vec![fragment(
            u64::from(DOMAIN_COUNT + 1),
            TOTAL_ROWS as usize,
            None,
        )];
        let second_resolved = BTreeMap::from([(0, DOMAIN_COUNT + 1)]);
        let second_result = layout
            .apply_delta(
                &second_delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &successor_fragments,
                    successor_fragments: &second_successor,
                    resolved_new_fragment_ids: &second_resolved,
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 3,
                    current_max_logical_fragment_id: Some(DOMAIN_COUNT - 1),
                    max_logical_fragment_id: Some(DOMAIN_COUNT - 1),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted {
            layout: second_layout,
            ..
        } = second_result
        else {
            panic!("repeated full-domain PackedRun relocation should be admitted");
        };
        assert!(matches!(
            &second_layout.placements[..],
            [RowAddressPlacement::PackedRun(_)]
        ));
    }

    #[test]
    fn apply_delta_retires_100m_rows_with_domain_algebra() {
        const DOMAIN_COUNT: u32 = 10_000;
        const ROWS_PER_DOMAIN: u32 = 10_000;

        let domains = (0..DOMAIN_COUNT)
            .map(|logical_fragment_id| domain(logical_fragment_id, ROWS_PER_DOMAIN))
            .collect::<Vec<_>>();
        let current_fragments = (0..DOMAIN_COUNT)
            .map(|fragment_id| {
                fragment(
                    u64::from(fragment_id),
                    ROWS_PER_DOMAIN as usize,
                    Some((fragment_id, 1)),
                )
            })
            .collect::<Vec<_>>();
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(&current_fragments, Some(DOMAIN_COUNT - 1));
        let retired = LogicalRowAddressSelection::from_full_domains(&domains).unwrap();
        assert_eq!(retired.cardinality(), 100_000_000);
        let delta = RowAddressLayoutDelta {
            source_domains: domains,
            retired_selections: vec![retired],
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        };
        let fully_deleted = (0..DOMAIN_COUNT).collect::<BTreeSet<_>>();
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &current_fragments,
                    successor_fragments: &[],
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &fully_deleted,
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(DOMAIN_COUNT - 1),
                    max_logical_fragment_id: Some(DOMAIN_COUNT - 1),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
            panic!("full-domain retirement should be admitted");
        };
        assert!(layout.placements.is_empty());
        assert_eq!(layout.retired_rows.len(), 1);
        assert!(matches!(
            layout.retired_rows[0].membership,
            RetiredLogicalRowMembership::AllRows
        ));
        let proto = retired_logical_row_set_to_proto(&layout.retired_rows[0], &BTreeMap::new());
        assert!(proto.encoded_len() <= 64);
    }

    #[test]
    fn apply_delta_retires_clustered_50m_range_without_row_expansion() {
        const ROWS: u32 = 100_000_000;
        const RETIRED_START: u32 = 25_000_000;
        const RETIRED_END: u32 = 75_000_000;

        let current_fragment = fragment(0, ROWS as usize, Some((0, 1)));
        let mut deleted = RoaringBitmap::new();
        deleted.insert_range(RETIRED_START..RETIRED_END);
        let successor = with_deletions(fragment(0, ROWS as usize, None), 2, &deleted);
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));
        let retired_range = LogicalRowAddressRange::new(0, RETIRED_START, RETIRED_END);
        let delta = RowAddressLayoutDelta {
            source_domains: vec![domain(0, ROWS)],
            retired_selections: vec![
                LogicalRowAddressSelection::from_ranges(vec![retired_range]).unwrap(),
            ],
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        };
        let deletion_vectors = BTreeMap::from([(0, &deleted)]);
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: std::slice::from_ref(&successor),
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &deletion_vectors,
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
            panic!("clustered retirement should be admitted");
        };
        let [RowAddressPlacement::Direct(direct)] = &layout.placements[..] else {
            panic!("the live native remainder should become Direct");
        };
        assert_eq!(
            direct.excluded.as_ref().unwrap().to_ranges().unwrap(),
            vec![retired_range]
        );
        let mut retired_ranges = Vec::new();
        layout
            .visit_retired_ranges(|range| retired_ranges.push(range))
            .unwrap();
        assert_eq!(retired_ranges, vec![retired_range]);
    }

    #[test]
    fn apply_delta_projects_historical_retirements_to_new_coverage() {
        const HISTORICAL_DOMAIN_COUNT: u32 = 10_000;
        const ROWS_PER_HISTORICAL_DOMAIN: u32 = 10_000;
        const NEW_LOGICAL_FRAGMENT_ID: u32 = 20_000;
        const NEW_ROWS: u32 = 100_000_000;

        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.retired_rows = vec![
            RetiredLogicalRowSet::all_rows(
                (0..HISTORICAL_DOMAIN_COUNT)
                    .map(|logical_fragment_id| {
                        domain(logical_fragment_id, ROWS_PER_HISTORICAL_DOMAIN)
                    })
                    .collect(),
            )
            .unwrap(),
        ];
        let current_fragment = fragment(42, NEW_ROWS as usize, Some((NEW_LOGICAL_FRAGMENT_ID, 1)));
        layout.refresh_fingerprint_with_fragments(
            std::slice::from_ref(&current_fragment),
            Some(NEW_LOGICAL_FRAGMENT_ID),
        );
        let source = domain(NEW_LOGICAL_FRAGMENT_ID, NEW_ROWS);
        let delta = RowAddressLayoutDelta {
            source_domains: vec![source],
            retired_selections: vec![
                LogicalRowAddressSelection::from_full_domains(&[source]).unwrap(),
            ],
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        };
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &[],
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::from([42]),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(NEW_LOGICAL_FRAGMENT_ID),
                    max_logical_fragment_id: Some(NEW_LOGICAL_FRAGMENT_ID),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
            panic!("unrelated historical retirement should not block a new full retirement");
        };
        assert_eq!(layout.retired_rows.len(), 1);
        assert!(matches!(
            layout.retired_rows[0].membership,
            RetiredLogicalRowMembership::AllRows
        ));
        assert_eq!(
            layout.retired_rows[0].domains.domain_count(),
            HISTORICAL_DOMAIN_COUNT + 1
        );
    }

    #[test]
    fn fully_retired_uniform_domains_are_constant_size() {
        for domain_count in [10_000_u32, 100_000] {
            let retired = RetiredLogicalRowSet::all_rows(
                (0..domain_count).map(|id| domain(id, 1_000)).collect(),
            )
            .unwrap();
            let proto = retired_logical_row_set_to_proto(&retired, &BTreeMap::new());
            assert!(
                proto.encoded_len() <= 64,
                "{} fully retired domains encoded to {} bytes",
                domain_count,
                proto.encoded_len()
            );
            assert_eq!(
                retired_logical_row_set_from_proto(proto, &[]).unwrap(),
                retired
            );
        }
    }

    #[test]
    fn retired_rows_normalize_history_and_promote_full_domains() {
        let source = domain(7, 4);
        let partial = normalize_retired_rows(vec![
            RetiredLogicalRowSet::selected(vec![source], Arc::new(selection(7, [1]))).unwrap(),
            RetiredLogicalRowSet::selected(vec![source], Arc::new(selection(7, [3]))).unwrap(),
        ])
        .unwrap();
        assert_eq!(partial.len(), 1);
        assert!(matches!(
            partial[0].membership,
            RetiredLogicalRowMembership::Selection(_)
        ));

        let full = normalize_retired_rows(vec![
            partial[0].clone(),
            RetiredLogicalRowSet::selected(vec![source], Arc::new(selection(7, [0, 2]))).unwrap(),
        ])
        .unwrap();
        assert_eq!(full.len(), 1);
        assert!(matches!(
            full[0].membership,
            RetiredLogicalRowMembership::AllRows
        ));
    }

    #[test]
    fn closure_fingerprint_changes_for_direct_append() {
        let namespace = Uuid::new_v4();
        let mut layout = RowAddressLayout::new(namespace);
        let first = vec![fragment(0, 10, Some((0, 1)))];
        layout.refresh_fingerprint_with_fragments(&first, Some(0));
        layout.validate_with_fragments(&first, Some(0)).unwrap();
        let first_fingerprint = layout.fingerprint.clone();
        let first_domain_fingerprint = layout.logical_domain_fingerprint.clone();

        let mut deleted = layout.clone();
        deleted.retired_rows = vec![RetiredLogicalRowSet::all_rows(vec![domain(0, 10)]).unwrap()];
        deleted.refresh_fingerprint_with_fragments(&[], Some(0));
        deleted.validate_with_fragments(&[], Some(0)).unwrap();
        assert_eq!(
            deleted.logical_domain_fingerprint, first_domain_fingerprint,
            "retirement changes liveness, not logical-domain identity"
        );

        let mut compacted = layout.clone();
        compacted.placements = vec![RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: domain(0, 10),
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: None,
        })];
        compacted.refresh_fingerprint_with_fragments(&[fragment(10, 10, None)], Some(0));
        assert_eq!(
            compacted.logical_domain_fingerprint, first_domain_fingerprint,
            "physical relocation changes placement, not logical-domain identity"
        );

        let second = vec![fragment(0, 10, Some((0, 1))), fragment(1, 10, Some((1, 2)))];
        layout.refresh_fingerprint_with_fragments(&second, Some(1));
        layout.validate_with_fragments(&second, Some(1)).unwrap();
        assert_ne!(layout.fingerprint, first_fingerprint);
        assert_ne!(layout.logical_domain_fingerprint, first_domain_fingerprint);
    }

    #[test]
    fn empty_logical_index_coverage_has_a_canonical_inline_representation() {
        let proto = pb::LogicalIndexCoverage {
            shards: Vec::new(),
            external: None,
            fingerprint: vec![7; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
            clone_provenance: None,
            full_domain_fingerprint: Vec::new(),
        };
        let coverage = LogicalIndexCoverage::try_from(proto.clone()).unwrap();
        assert!(coverage.shards.is_empty());
        assert!(coverage.external.is_none());
        assert_eq!(pb::LogicalIndexCoverage::from(&coverage), proto);

        assert!(
            LogicalIndexCoverage::try_from(pb::LogicalIndexCoverage {
                fingerprint: vec![7; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE - 1],
                ..proto
            })
            .is_err()
        );
    }

    #[test]
    fn router_resolves_fast_path_codecs_and_inverse() {
        let fragments = vec![
            fragment(0, 4, Some((0, 1))),
            fragment(10, 4, None),
            fragment(11, 2, None),
            fragment(12, 4, None),
            fragment(13, 3, None),
        ];
        let placements = vec![
            RowAddressPlacement::PackedRun(
                PackedRunRowAddressPlacement::from_sources(vec![domain(1, 2), domain(2, 2)], 10, 0)
                    .unwrap(),
            ),
            RowAddressPlacement::Selected(SelectedRowAddressPlacement {
                source: domain(3, 4),
                selection: selection(3, [1, 3]).into(),
                destination_fragment_id: 11,
                destination_start: 0,
                excluded: None,
            }),
            RowAddressPlacement::ExtentList(ExtentListRowAddressPlacement {
                source: domain(4, 4),
                extents: vec![
                    RowAddressExtent {
                        source_start: 0,
                        length: 2,
                        destination_fragment_id: 12,
                        destination_start: 0,
                    },
                    RowAddressExtent {
                        source_start: 2,
                        length: 2,
                        destination_fragment_id: 12,
                        destination_start: 2,
                    },
                ],
            }),
            RowAddressPlacement::SparseSelection(SparseSelectionRowAddressPlacement {
                sources: vec![
                    SparseSelectionSource {
                        source: domain(5, 3),
                        selection: selection(5, [0, 2]).into(),
                        excluded: None,
                    },
                    SparseSelectionSource {
                        source: domain(6, 2),
                        selection: selection(6, [1]).into(),
                        excluded: None,
                    },
                ],
                destination_fragment_id: 13,
                destination_start: 0,
            }),
        ];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = placements;
        for fragment in fragments
            .iter()
            .filter(|fragment| fragment.native_logical_domain.is_none())
        {
            layout
                .refresh_physical_ownership_summary(fragment, &RoaringBitmap::new())
                .unwrap();
        }
        layout.refresh_fingerprint_with_fragments(&fragments, Some(6));
        let router = RowAddressRouter::try_new(Arc::new(layout), &fragments, Some(6)).unwrap();

        let addresses = [
            LogicalRowAddress::try_new_from_parts(0, 2).unwrap(),
            LogicalRowAddress::try_new_from_parts(2, 0).unwrap(),
            LogicalRowAddress::try_new_from_parts(3, 3).unwrap(),
            LogicalRowAddress::try_new_from_parts(4, 2).unwrap(),
            LogicalRowAddress::try_new_from_parts(6, 1).unwrap(),
            LogicalRowAddress::try_new_from_parts(3, 0).unwrap(),
        ];
        let resolved = router.resolve_many(&addresses).unwrap();
        let physical = resolved
            .iter()
            .map(|resolution| match resolution {
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(address),
                } => Some(*address),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            physical,
            vec![
                Some(RowAddress::new_from_parts(0, 2)),
                Some(RowAddress::new_from_parts(10, 2)),
                Some(RowAddress::new_from_parts(11, 1)),
                Some(RowAddress::new_from_parts(12, 2)),
                Some(RowAddress::new_from_parts(13, 2)),
                None,
            ]
        );

        let inverse = router
            .physical_to_logical_many(&[
                RowAddress::new_from_parts(10, 2),
                RowAddress::new_from_parts(11, 1),
                RowAddress::new_from_parts(13, 2),
            ])
            .unwrap();
        assert_eq!(
            inverse,
            vec![
                PhysicalToLogicalResolution::Logical(addresses[1]),
                PhysicalToLogicalResolution::Logical(addresses[2]),
                PhysicalToLogicalResolution::Logical(addresses[4]),
            ]
        );
        assert_eq!(
            router.logical_domain_destination_ranges(0).unwrap(),
            vec![LogicalDomainDestination::Inline(PhysicalRowRange {
                physical_fragment_id: 0,
                start_offset: 0,
                end_offset: 4,
            })]
        );
        assert_eq!(
            router.logical_domain_destination_ranges(3).unwrap(),
            vec![LogicalDomainDestination::Inline(PhysicalRowRange {
                physical_fragment_id: 11,
                start_offset: 0,
                end_offset: 2,
            })]
        );
        assert_eq!(
            router.logical_domain_destination_ranges(4).unwrap(),
            vec![
                LogicalDomainDestination::Inline(PhysicalRowRange {
                    physical_fragment_id: 12,
                    start_offset: 0,
                    end_offset: 2,
                }),
                LogicalDomainDestination::Inline(PhysicalRowRange {
                    physical_fragment_id: 12,
                    start_offset: 2,
                    end_offset: 4,
                }),
            ]
        );
        assert_eq!(
            router.logical_domain_destination_ranges(5).unwrap(),
            vec![LogicalDomainDestination::Inline(PhysicalRowRange {
                physical_fragment_id: 13,
                start_offset: 0,
                end_offset: 2,
            })]
        );
        assert_eq!(
            router.logical_domain_destination_ranges(6).unwrap(),
            vec![LogicalDomainDestination::Inline(PhysicalRowRange {
                physical_fragment_id: 13,
                start_offset: 2,
                end_offset: 3,
            })]
        );
    }

    #[test]
    fn router_indexes_interleaved_gapped_packed_runs() {
        let fragments = vec![fragment(10, 2, None), fragment(11, 2, None)];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![
            RowAddressPlacement::PackedRun(
                PackedRunRowAddressPlacement::from_sources(vec![domain(1, 1), domain(3, 1)], 10, 0)
                    .unwrap(),
            ),
            RowAddressPlacement::PackedRun(
                PackedRunRowAddressPlacement::from_sources(vec![domain(2, 1), domain(4, 1)], 11, 0)
                    .unwrap(),
            ),
        ];
        for fragment in &fragments {
            layout
                .refresh_physical_ownership_summary(fragment, &RoaringBitmap::new())
                .unwrap();
        }
        layout.refresh_fingerprint_with_fragments(&fragments, Some(4));
        let router = RowAddressRouter::try_new(Arc::new(layout), &fragments, Some(4)).unwrap();
        let logical = (1..=4)
            .map(|logical_fragment_id| {
                LogicalRowAddress::try_new_from_parts(logical_fragment_id, 0).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            router.resolve_many(&logical).unwrap(),
            vec![
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(10, 0)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(11, 0)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(10, 1)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(11, 1)),
                },
            ]
        );
    }

    #[test]
    fn relocated_domain_retains_creation_version() {
        let fragments = vec![fragment(10, 5, None)];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::PackedRun(
            PackedRunRowAddressPlacement::from_sources(
                vec![RowAddressLogicalDomain::new(7, 5, 3).unwrap()],
                10,
                0,
            )
            .unwrap(),
        )];
        layout
            .refresh_physical_ownership_summary(&fragments[0], &RoaringBitmap::new())
            .unwrap();
        layout.refresh_fingerprint_with_fragments(&fragments, Some(7));
        layout.validate_with_fragments(&fragments, Some(7)).unwrap();
        assert_eq!(
            layout.domain_creation_version(&fragments, 7).unwrap(),
            Some(3)
        );
    }

    #[test]
    fn manifest_v23_requires_clean_layout_contract() {
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );
        assert!(manifest.validate_row_address_contract().is_err());

        let layout = RowAddressLayout::new(Uuid::new_v4());
        manifest.row_address_layout = Some(Arc::new(layout));
        manifest.refresh_row_address_fingerprint().unwrap();
        manifest.validate_row_address_contract().unwrap();
        let proto: pb::Manifest = (&manifest).into();
        let decoded = Manifest::try_from(proto).unwrap();
        assert_eq!(decoded.row_address_layout, manifest.row_address_layout);

        manifest.reader_feature_flags |= crate::feature_flags::FLAG_STABLE_ROW_IDS;
        assert!(manifest.validate_row_address_contract().is_err());
    }

    #[test]
    fn layout_rejects_stale_fingerprint() {
        let fragments = vec![fragment(0, 4, Some((0, 1)))];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(&fragments, Some(0));
        layout.debt_summary.canonical_layout_bytes = 1;
        let error = layout
            .validate_with_fragments(&fragments, Some(0))
            .unwrap_err();
        assert!(error.to_string().contains("fingerprint"));
    }

    #[test]
    fn layout_rejects_physical_ownership_gap_and_overlap() {
        let gap_fragments = vec![fragment(10, 4, None)];
        let mut gap = RowAddressLayout::new(Uuid::new_v4());
        gap.placements = vec![RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: domain(1, 2),
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: None,
        })];
        gap.refresh_fingerprint_with_fragments(&gap_fragments, Some(1));
        let error = gap
            .validate_with_fragments(&gap_fragments, Some(1))
            .unwrap_err();
        assert!(error.to_string().contains("ownership summary"));

        let overlap_fragments = vec![fragment(11, 3, None)];
        let mut overlap = RowAddressLayout::new(Uuid::new_v4());
        overlap.placements = vec![
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(2, 2),
                destination_fragment_id: 11,
                destination_start: 0,
                excluded: None,
            }),
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(3, 2),
                destination_fragment_id: 11,
                destination_start: 1,
                excluded: None,
            }),
        ];
        overlap.refresh_fingerprint_with_fragments(&overlap_fragments, Some(3));
        let error = overlap
            .validate_with_fragments(&overlap_fragments, Some(3))
            .unwrap_err();
        assert!(error.to_string().contains("overlapping"));
    }

    #[test]
    fn fully_retired_domain_can_remain_in_explicit_map_root() {
        let retired_source = domain(0, 2);
        let live_source = domain(1, 2);
        let target = fragment(10, 2, None);
        let first_live = LogicalRowAddress::try_new_from_parts(1, 0).unwrap();
        let last_live = LogicalRowAddress::try_new_from_parts(1, 1).unwrap();
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::ExplicitMap(
            ExplicitMapRowAddressPlacement {
                sources: [retired_source, live_source]
                    .into_iter()
                    .map(|source| {
                        Ok(SparseSelectionSource {
                            source,
                            selection: Arc::new(LogicalRowAddressSelection::from_full_domains(&[
                                source,
                            ])?),
                            excluded: None,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
                    .unwrap(),
                object_path: "data/_row_addresses/locator.lance".to_owned(),
                object_size: 128,
                pages: vec![ExplicitMapPage {
                    first_logical_address: first_live.raw(),
                    last_logical_address: last_live.raw(),
                    row_start: 0,
                    row_count: 2,
                    content_fingerprint: vec![11; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
                }],
                destinations: vec![ExplicitMapDestination {
                    physical_fragment_id: 10,
                    destination_start: 0,
                    row_count: 2,
                    row_id_file_path: "data/_row_addresses/row_ids.lance".to_owned(),
                    row_id_file_size: 64,
                    row_id_pages: vec![ExplicitMapRowIdPage {
                        row_start: 0,
                        row_count: 2,
                        content_fingerprint: vec![12; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
                    }],
                }],
                base_id: None,
            },
        )];
        layout.retired_rows = vec![RetiredLogicalRowSet::all_rows(vec![retired_source]).unwrap()];
        layout
            .refresh_physical_ownership_summary(&target, &RoaringBitmap::new())
            .unwrap();
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&target), Some(1));
        layout
            .validate_with_fragments(std::slice::from_ref(&target), Some(1))
            .unwrap();

        let proto: pb::RowAddressLayout = (&layout).into();
        let decoded = RowAddressLayout::try_from(proto).unwrap();
        decoded
            .validate_with_fragments(std::slice::from_ref(&target), Some(1))
            .unwrap();
        let router =
            RowAddressRouter::try_new(Arc::new(decoded), std::slice::from_ref(&target), Some(1))
                .unwrap();
        assert_eq!(
            router
                .resolve_many(&[LogicalRowAddress::try_new_from_parts(0, 0).unwrap()])
                .unwrap(),
            vec![PlacementResolution::NotLive]
        );
        assert_eq!(
            router.resolve_many(&[first_live]).unwrap(),
            vec![PlacementResolution::Mapped {
                locator: PhysicalRowLocator::ExplicitMap {
                    placement_index: 0,
                    page_index: 0,
                },
            }]
        );
    }

    #[test]
    fn fully_retired_domain_still_rejects_inline_placement() {
        let inline_target = fragment(11, 2, None);
        let inline_source = domain(2, 2);
        let mut inline_layout = RowAddressLayout::new(Uuid::new_v4());
        inline_layout.placements = vec![RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: inline_source,
            destination_fragment_id: 11,
            destination_start: 0,
            excluded: None,
        })];
        inline_layout.retired_rows =
            vec![RetiredLogicalRowSet::all_rows(vec![inline_source]).unwrap()];
        inline_layout
            .refresh_physical_ownership_summary(&inline_target, &RoaringBitmap::new())
            .unwrap();
        inline_layout
            .refresh_fingerprint_with_fragments(std::slice::from_ref(&inline_target), Some(2));
        let error = inline_layout
            .validate_with_fragments(std::slice::from_ref(&inline_target), Some(2))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("fully retired logical domain 2 still has a live placement")
        );
    }

    #[test]
    fn row_address_layout_delta_round_trip() {
        let create = RowAddressLayoutDelta::for_create(Uuid::new_v4());
        let create_proto: pb::transaction::RowAddressLayoutDelta = (&create).into();
        assert_eq!(
            RowAddressLayoutDelta::try_from(create_proto).unwrap(),
            create
        );

        let successor = RowAddressLayoutDelta {
            source_domains: vec![domain(1, 2)],
            retired_selections: vec![selection(1, [1])],
            placements: vec![RowAddressPlacementDelta {
                source_selections: Vec::new(),
                target: RowAddressTargetRange {
                    fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
                    start_offset: 0,
                    end_offset: 2,
                },
                placement_kind: RowAddressPlacementKind::Direct,
                output_cardinality: 2,
                output_row_sequence_fingerprint: Vec::new(),
            }],
            expected_layout_fingerprint: vec![7; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
            ..RowAddressLayoutDelta::default()
        };
        let proto: pb::transaction::RowAddressLayoutDelta = (&successor).into();
        assert_eq!(RowAddressLayoutDelta::try_from(proto).unwrap(), successor);

        let row_aligned = RowAddressLayoutDelta {
            field_changes: vec![RowAddressFieldChange {
                selection: selection(3, [0, 2]),
                field_ids: vec![4, 9],
            }],
            source_floors: vec![
                RowAddressSourceFloor {
                    field_id: 4,
                    generation: 7,
                },
                RowAddressSourceFloor {
                    field_id: 9,
                    generation: 8,
                },
            ],
            expected_layout_fingerprint: vec![7; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
            row_aligned_rewrite_proofs: vec![RowAlignedRewriteProof {
                physical_fragment_id: 11,
                physical_rows: 3,
                mapped_offsets_fingerprint: vec![5; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
                deletion_offsets_fingerprint: Some(vec![6; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE]),
                field_change_index: 0,
                source_floor_indices: vec![0, 1],
            }],
            ..RowAddressLayoutDelta::default()
        };
        let proto: pb::transaction::RowAddressLayoutDelta = (&row_aligned).into();
        assert_eq!(RowAddressLayoutDelta::try_from(proto).unwrap(), row_aligned);

        let source = domain(4, 2);
        let source_selection = selection(4, [0, 1]);
        let explicit = ExplicitMapRowAddressPlacement {
            sources: vec![SparseSelectionSource {
                source,
                selection: Arc::new(source_selection.clone()),
                excluded: None,
            }],
            object_path: "data/_row_addresses/locator.lance".to_owned(),
            object_size: 128,
            pages: vec![ExplicitMapPage {
                first_logical_address: LogicalRowAddress::try_new_from_parts(4, 0).unwrap().raw(),
                last_logical_address: LogicalRowAddress::try_new_from_parts(4, 1).unwrap().raw(),
                row_start: 0,
                row_count: 2,
                content_fingerprint: vec![11; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
            }],
            destinations: vec![ExplicitMapDestination {
                physical_fragment_id: 7,
                destination_start: 0,
                row_count: 2,
                row_id_file_path: "data/_row_addresses/row_ids.lance".to_owned(),
                row_id_file_size: 64,
                row_id_pages: vec![ExplicitMapRowIdPage {
                    row_start: 0,
                    row_count: 2,
                    content_fingerprint: vec![12; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
                }],
            }],
            base_id: Some(9),
        };
        let placement = RowAddressPlacement::ExplicitMap(explicit.clone());
        let selection_pool = vec![explicit.sources[0].selection.clone()];
        let selection_indices =
            canonical_selection_indices(selection_pool.iter().map(|selection| selection.as_ref()));
        let placement_proto = placement_to_proto(&placement, &selection_indices);
        assert_eq!(
            placement_from_proto(placement_proto, &selection_pool).unwrap(),
            placement
        );
        let mut missing_locator_fingerprint = explicit.clone();
        missing_locator_fingerprint.pages[0]
            .content_fingerprint
            .clear();
        let error = RowAddressPlacement::ExplicitMap(missing_locator_fingerprint)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("16-byte content fingerprint"));
        let mut incomplete_destination_pages = explicit.clone();
        incomplete_destination_pages.destinations[0].row_id_pages[0].row_count = 1;
        let error = RowAddressPlacement::ExplicitMap(incomplete_destination_pages)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("pages cover 1 rows"));
        let explicit_delta = RowAddressLayoutDelta {
            source_domains: vec![source],
            placements: vec![RowAddressPlacementDelta {
                source_selections: vec![source_selection],
                target: RowAddressTargetRange {
                    fragment: RowAddressTargetFragment::ExistingFragmentId(7),
                    start_offset: 0,
                    end_offset: 2,
                },
                placement_kind: RowAddressPlacementKind::ExplicitMap,
                output_cardinality: 2,
                output_row_sequence_fingerprint: vec![3; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
            }],
            expected_layout_fingerprint: vec![7; ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE],
            explicit_map_placements: BTreeMap::from([(0, explicit)]),
            ..RowAddressLayoutDelta::default()
        };
        let proto: pb::transaction::RowAddressLayoutDelta = (&explicit_delta).into();
        assert_eq!(
            RowAddressLayoutDelta::try_from(proto).unwrap(),
            explicit_delta
        );
        assert!(explicit_delta.is_pure_explicit_rewrite());
        assert!(
            explicit_delta
                .fast_admission_projection()
                .placements
                .is_empty()
        );

        let mut mixed = explicit_delta;
        mixed.create_namespace_uuid = Some(Uuid::new_v4());
        let error = mixed.validate_admission_tier().unwrap_err();
        assert!(error.to_string().contains("cannot be mixed"));
    }

    #[test]
    fn explicit_map_page_fingerprint_detects_same_size_content_change() {
        let logical = [10, 20, 30, 40];
        let physical = [100, 200, 300, 400];
        let expected = fingerprint_explicit_map_u64_page(&[&logical, &physical]).unwrap();
        let mut corrupted = physical;
        corrupted[2] = 301;
        let actual = fingerprint_explicit_map_u64_page(&[&logical, &corrupted]).unwrap();

        assert_eq!(expected.len(), ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE);
        assert_eq!(actual.len(), ROW_ADDRESS_LAYOUT_FINGERPRINT_SIZE);
        assert_ne!(expected, actual);
    }

    #[test]
    fn canonical_selection_is_origin_independent_and_roaring_stays_lazy() {
        let ranges = vec![
            LogicalRowAddressRange::new(7, 1, 4),
            LogicalRowAddressRange::new(7, 9, 11),
            LogicalRowAddressRange::new(8, 2, 3),
        ];
        let mut bitmap = RoaringTreemap::new();
        for range in &ranges {
            bitmap.insert_range(
                ((range.logical_fragment_id as u64) << 32 | range.start_slot as u64)
                    ..((range.logical_fragment_id as u64) << 32 | range.end_slot as u64),
            );
        }
        let from_ranges = LogicalRowAddressSelection::from_ranges(ranges).unwrap();
        let from_bitmap = LogicalRowAddressSelection::from_bitmap(bitmap.clone()).unwrap();
        assert_eq!(
            from_ranges.canonical_proto().encode_to_vec(),
            from_bitmap.canonical_proto().encode_to_vec()
        );

        let mut bytes = Vec::new();
        bitmap.serialize_into(&mut bytes).unwrap();
        let encoded_len = bytes.len();
        let decoded = LogicalRowAddressSelection::try_from(pb::LogicalRowAddressSelection {
            encoding: Some(pb::logical_row_address_selection::Encoding::RoaringTreemap(
                bytes,
            )),
            canonical_encoding_version: 1,
        })
        .unwrap();
        let LogicalRowAddressSelection::Roaring(roaring) = &decoded else {
            panic!("expected retained Roaring encoding");
        };
        assert!(roaring.bitmap.get().is_none());
        assert!(decoded.deep_size_of() < encoded_len * 2 + 256);
        assert_eq!(decoded.select(0).unwrap().unwrap().raw(), (7_u64 << 32) | 1);
        assert!(roaring.bitmap.get().is_some());
    }

    #[test]
    fn compressed_headers_reject_impossible_domain_counts_without_expansion() {
        let huge_run = pb::LogicalOrdinalDomainRun {
            first_logical_fragment_id: 0,
            domain_count: MAX_COMPRESSED_DOMAINS + 1,
            logical_fragment_ids: Some(pb::PackedLogicalFragmentIds {
                encoding: Some(pb::packed_logical_fragment_ids::Encoding::Consecutive(
                    pb::ConsecutiveLogicalFragmentIds {},
                )),
            }),
            slot_universes: Some(pb::PackedSlotCounts {
                encoding: Some(pb::packed_slot_counts::Encoding::UniformSlotCount(1)),
            }),
        };
        let selection = pb::LogicalRowAddressSelection {
            encoding: Some(
                pb::logical_row_address_selection::Encoding::OrdinalEliasFano(
                    pb::EliasFanoOrdinalLogicalRowAddressSelection {
                        domain_runs: vec![huge_run],
                        universe: 1,
                        cardinality: 1,
                        low_bit_width: 0,
                        low_bits: Vec::new(),
                        high_bits: vec![1],
                        select_checkpoint_interval: SELECT_CHECKPOINT_INTERVAL,
                        select_checkpoints: vec![0],
                    },
                ),
            ),
            canonical_encoding_version: 1,
        };
        assert!(LogicalRowAddressSelection::try_from(selection).is_err());
    }

    #[test]
    fn packed_run_handles_100k_domains_and_router_binary_searches_10k_runs() {
        let large = PackedRunRowAddressPlacement::from_sources(
            (0..100_000).map(|id| domain(id, 1)).collect(),
            42,
            0,
        )
        .unwrap();
        assert_eq!(large.domains.domain_count(), 100_000);
        assert_eq!(large.domains.domain_at(99_999).unwrap(), domain(99_999, 1));
        assert_eq!(
            large.domains.ordinal_for_slot_offset(99_999).unwrap(),
            Some((99_999, 0))
        );

        let target = fragment(50, 10_000, None);
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = (0..10_000_u32)
            .map(|ordinal| {
                RowAddressPlacement::PackedRun(
                    PackedRunRowAddressPlacement::from_sources(
                        vec![domain(ordinal * 2, 1)],
                        50,
                        ordinal,
                    )
                    .unwrap(),
                )
            })
            .collect();
        layout
            .refresh_physical_ownership_summary(&target, &RoaringBitmap::new())
            .unwrap();
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&target), Some(19_998));
        let router = RowAddressRouter::try_new(
            Arc::new(layout),
            std::slice::from_ref(&target),
            Some(19_998),
        )
        .unwrap();
        let logical = LogicalRowAddress::try_new_from_parts(19_998, 0).unwrap();
        assert_eq!(
            router.resolve_many(&[logical]).unwrap(),
            vec![PlacementResolution::Mapped {
                locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(50, 9_999))
            }]
        );
    }

    #[test]
    fn destination_index_bounds_fragment_ownership_lookup() {
        const FRAGMENT_COUNT: u32 = 4096;
        let placements = (0..FRAGMENT_COUNT)
            .map(|fragment_id| {
                RowAddressPlacement::Direct(DirectRowAddressPlacement {
                    source: domain(fragment_id, 1),
                    destination_fragment_id: fragment_id,
                    destination_start: 0,
                    excluded: None,
                })
            })
            .collect::<Vec<_>>();
        let destination_index = build_destination_index(&placements);

        assert_eq!(destination_index.len(), FRAGMENT_COUNT as usize);
        for fragment_id in [0, FRAGMENT_COUNT / 2, FRAGMENT_COUNT - 1] {
            let mut ranges = Vec::new();
            for_each_indexed_canonical_mapped_offset_range(
                &placements,
                &destination_index,
                fragment_id,
                |start, end| {
                    ranges.push((start, end));
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(ranges, vec![(0, 1)]);
            assert_eq!(
                destination_index[fragment_id as usize].placement_indices,
                vec![fragment_id]
            );
        }

        let corrupt_index = vec![RowAddressDestinationIndexEntry {
            physical_fragment_id: 0,
            placement_indices: vec![u32::MAX],
        }];
        let error = for_each_indexed_canonical_mapped_offset_range(
            &placements,
            &corrupt_index,
            0,
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("references a missing placement"));
    }

    #[test]
    fn ownership_allows_deleted_mapped_rows_but_requires_unowned_subset() {
        let mut target = fragment(10, 4, None);
        target.deletion_file = Some(crate::format::DeletionFile {
            read_version: 1,
            id: 7,
            file_type: crate::format::DeletionFileType::Bitmap,
            num_deleted_rows: Some(3),
            base_id: None,
        });
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: domain(1, 2),
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: None,
        })];
        let deleted = RoaringBitmap::from_iter([1, 2, 3]);
        layout
            .refresh_physical_ownership_summary(&target, &deleted)
            .unwrap();
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&target), Some(1));
        layout
            .validate_with_fragments(std::slice::from_ref(&target), Some(1))
            .unwrap();
        layout.verify_visibility(&target, &deleted).unwrap();

        let missing_gap = RoaringBitmap::from_iter([1, 2]);
        assert!(
            layout
                .refresh_physical_ownership_summary(&target, &missing_gap)
                .is_err()
        );
    }

    #[test]
    fn ownership_summary_accepts_interleaved_hole_reuse() {
        let target = fragment(9, 3, None);
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(7, 3),
                destination_fragment_id: 9,
                destination_start: 0,
                excluded: Some(Arc::new(selection(7, [1]))),
            }),
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(8, 1),
                destination_fragment_id: 9,
                destination_start: 1,
                excluded: None,
            }),
        ];

        assert!(layout.destination_index.is_empty());
        layout
            .refresh_physical_ownership_summary(&target, &RoaringBitmap::new())
            .unwrap();
        assert_eq!(
            layout.destination_index,
            vec![RowAddressDestinationIndexEntry {
                physical_fragment_id: 9,
                placement_indices: vec![0, 1],
            }]
        );
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&target), Some(8));
        layout
            .validate_with_fragments(std::slice::from_ref(&target), Some(8))
            .unwrap();
        let destination_index = layout.validated_destination_index().unwrap();
        assert_eq!(
            destination_index
                .placements_for_fragments([target.id as u32, target.id as u32])
                .len(),
            2
        );
        destination_index
            .verify_visibility(&target, &RoaringBitmap::new())
            .unwrap();
        let mut missing_index = layout.clone();
        missing_index.destination_index.clear();
        missing_index
            .verify_visibility(&target, &RoaringBitmap::new())
            .unwrap();
        assert!(
            missing_index
                .validated_destination_index()
                .unwrap_err()
                .to_string()
                .contains("destination index does not match")
        );
        let mut incomplete_index = layout.clone();
        incomplete_index.destination_index[0].placement_indices = vec![0];
        incomplete_index
            .verify_visibility(&target, &RoaringBitmap::new())
            .unwrap();
        assert!(
            incomplete_index
                .validated_destination_index()
                .unwrap_err()
                .to_string()
                .contains("destination index does not match")
        );
        assert!(
            incomplete_index
                .validate_with_fragments(std::slice::from_ref(&target), Some(8))
                .unwrap_err()
                .to_string()
                .contains("destination index does not match")
        );

        let router =
            RowAddressRouter::try_new(Arc::new(layout), std::slice::from_ref(&target), Some(8))
                .unwrap();
        let logical = [
            LogicalRowAddress::try_new_from_parts(7, 0).unwrap(),
            LogicalRowAddress::try_new_from_parts(8, 0).unwrap(),
            LogicalRowAddress::try_new_from_parts(7, 2).unwrap(),
        ];
        assert_eq!(
            router.resolve_many(&logical).unwrap(),
            vec![
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(9, 0)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(9, 1)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(9, 2)),
                },
            ]
        );
    }

    #[test]
    fn selection_pool_serializes_once_and_shares_runtime_payload() {
        let shared = Arc::new(selection(1, [0, 1]));
        let target = fragment(10, 2, None);
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::Selected(SelectedRowAddressPlacement {
            source: domain(1, 2),
            selection: shared.clone(),
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: None,
        })];
        layout.generation_regions = vec![ContentGenerationRegion {
            selection: shared,
            field_ids: vec![0],
            generation: 1,
        }];
        layout
            .refresh_physical_ownership_summary(&target, &RoaringBitmap::new())
            .unwrap();
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&target), Some(1));
        let proto: pb::RowAddressLayout = (&layout).into();
        assert_eq!(proto.selection_pool.len(), 1);
        let pb::row_address_placement::Codec::Selected(selected) =
            proto.placements[0].codec.as_ref().unwrap()
        else {
            panic!("expected Selected placement");
        };
        assert_eq!(selected.selection_index, Some(0));
        assert_eq!(proto.generation_regions[0].selection_index, Some(0));

        let decoded = RowAddressLayout::try_from(proto).unwrap();
        let RowAddressPlacement::Selected(selected) = &decoded.placements[0] else {
            panic!("expected Selected placement");
        };
        assert!(Arc::ptr_eq(&decoded.selection_pool[0], &selected.selection));
        assert!(Arc::ptr_eq(
            &decoded.selection_pool[0],
            &decoded.generation_regions[0].selection
        ));
    }

    #[test]
    fn missing_pool_index_and_duplicate_runtime_routes_are_corruption() {
        let shared = Arc::new(selection(1, [0]));
        let placement = RowAddressPlacement::Selected(SelectedRowAddressPlacement {
            source: domain(1, 1),
            selection: shared,
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: None,
        });
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![placement.clone()];
        layout.refresh_fingerprint();
        let mut proto: pb::RowAddressLayout = (&layout).into();
        let Some(pb::row_address_placement::Codec::Selected(selected)) =
            proto.placements[0].codec.as_mut()
        else {
            panic!("expected Selected placement");
        };
        selected.selection_index = None;
        assert!(RowAddressLayout::try_from(proto).is_err());

        let mut corrupt_layout = RowAddressLayout::new(Uuid::new_v4());
        corrupt_layout.placements = vec![placement.clone(), placement];
        corrupt_layout.destination_index = vec![RowAddressDestinationIndexEntry {
            physical_fragment_id: 10,
            placement_indices: vec![0, 1],
        }];
        let router = RowAddressRouter {
            layout: Arc::new(corrupt_layout),
            source_index: vec![(1, 0), (1, 1)],
            packed_source_runs: Vec::new(),
            native_domains: Vec::new(),
            native_by_physical: Vec::new(),
        };
        let logical = LogicalRowAddress::try_new_from_parts(1, 0).unwrap();
        assert!(router.resolve_many(&[logical]).is_err());
        assert!(
            router
                .physical_to_logical_many(&[RowAddress::new_from_parts(10, 0)])
                .is_err()
        );
    }

    #[test]
    fn emitted_row_sequence_digest_detects_reorder() {
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: 2,
        };
        let first = LogicalRowAddress::try_new_from_parts(1, 0).unwrap();
        let second = LogicalRowAddress::try_new_from_parts(1, 1).unwrap();
        let ordered = fingerprint_row_sequence(target, [first, second]).unwrap();
        let reordered = fingerprint_row_sequence(target, [second, first]).unwrap();
        assert_ne!(ordered, reordered);
        let delta = RowAddressPlacementDelta {
            source_selections: vec![selection(1, [0, 1])],
            target,
            placement_kind: RowAddressPlacementKind::Selected,
            output_cardinality: 2,
            output_row_sequence_fingerprint: reordered,
        };
        assert!(delta.verify_output_row_sequence().is_err());
    }

    #[test]
    fn structural_roaring_fingerprint_matches_row_iteration() {
        let bitmap = [
            (1_u64 << 32),
            (1_u64 << 32) | 1,
            (1_u64 << 32) | 2,
            (1_u64 << 32) | 4,
            (1_u64 << 32) | 7,
            (1_u64 << 32) | 8,
            (3_u64 << 32) | 10,
            (3_u64 << 32) | 11,
            (3_u64 << 32) | 12,
        ]
        .into_iter()
        .collect::<RoaringTreemap>();
        let selection = LogicalRowAddressSelection::try_from(
            SelectionBuilderInput::roaring_candidate(&bitmap).unwrap(),
        )
        .unwrap();
        assert!(matches!(selection, LogicalRowAddressSelection::Roaring(_)));
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(7),
            start_offset: 4,
            end_offset: 4 + bitmap.len() as u32,
        };
        let delta = RowAddressPlacementDelta {
            source_selections: vec![selection],
            target,
            placement_kind: RowAddressPlacementKind::Selected,
            output_cardinality: bitmap.len(),
            output_row_sequence_fingerprint: Vec::new(),
        };
        let addresses = bitmap
            .iter()
            .map(LogicalRowAddress::try_from)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            delta.expected_row_sequence_fingerprint().unwrap(),
            fingerprint_row_sequence(target, addresses).unwrap()
        );
    }

    #[test]
    fn schema_field_contract_rejects_dropped_generation_ids() {
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.field_default_generations = vec![
            FieldGeneration {
                field_id: 0,
                generation: 1,
            },
            FieldGeneration {
                field_id: 1,
                generation: 1,
            },
        ];
        layout.index_commit_floors = vec![FieldGeneration {
            field_id: 2,
            generation: 1,
        }];
        assert!(layout.validate_schema_fields(&[0, 1]).is_err());
    }

    #[test]
    fn apply_delta_native_to_update_and_repeated_update_have_one_owner() {
        let current_fragments = vec![fragment(0, 4, Some((0, 1)))];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.field_default_generations = vec![FieldGeneration {
            field_id: 0,
            generation: 1,
        }];
        layout.index_commit_floors = layout.field_default_generations.clone();
        layout.refresh_fingerprint_with_fragments(&current_fragments, Some(0));

        let old_deleted = RoaringBitmap::from_iter([1, 3]);
        let successor_fragments = vec![
            with_deletions(fragment(0, 4, None), 1, &old_deleted),
            fragment(1, 2, None),
        ];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: 2,
        };
        let mut delta = placement_delta(
            &layout,
            domain(0, 4),
            [1, 3],
            target,
            RowAddressPlacementKind::SparseSelection,
        );
        delta.source_floors = vec![RowAddressSourceFloor {
            field_id: 0,
            generation: 1,
        }];
        delta.field_changes = vec![RowAddressFieldChange {
            selection: selection(0, [1, 3]),
            field_ids: vec![0],
        }];
        let resolved = BTreeMap::from([(0, 1)]);
        let deletion_vectors = BTreeMap::from([(0, &old_deleted)]);
        let explicit = BTreeMap::new();
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &current_fragments,
                    successor_fragments: &successor_fragments,
                    resolved_new_fragment_ids: &resolved,
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &deletion_vectors,
                    explicit_map_placements: &explicit,
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 123,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted {
            layout: updated, ..
        } = result
        else {
            panic!("native update should be admitted");
        };
        assert_eq!(
            updated
                .debt_summary
                .metadata_bytes_written_since_maintenance,
            123
        );
        assert_eq!(updated.generation_regions.len(), 1);
        updated
            .validate_with_fragments(&successor_fragments, Some(0))
            .unwrap();
        let router =
            RowAddressRouter::try_new(Arc::from(updated.clone()), &successor_fragments, Some(0))
                .unwrap();
        let addresses = (0..4)
            .map(|slot| LogicalRowAddress::try_new_from_parts(0, slot).unwrap())
            .collect::<Vec<_>>();
        let resolved_rows = router.resolve_many(&addresses).unwrap();
        for resolution in &resolved_rows {
            assert!(matches!(resolution, PlacementResolution::Mapped { .. }));
        }
        assert_eq!(
            router
                .logical_selection_destination_fragments(0, &RoaringBitmap::from_iter([1]),)
                .unwrap(),
            RoaringBitmap::from_iter([1]),
            "an updated slot must route only to its Selected destination"
        );
        assert_eq!(
            router
                .logical_selection_destination_fragments(0, &RoaringBitmap::from_iter([0, 2]),)
                .unwrap(),
            RoaringBitmap::from_iter([0]),
            "untouched slots must stay on the Direct source fragment"
        );
        assert_eq!(
            router
                .logical_selection_destination_fragments(
                    0,
                    &RoaringBitmap::from_iter([0, 1, 2, 3]),
                )
                .unwrap(),
            RoaringBitmap::from_iter([0, 1])
        );

        let new_old_deleted = RoaringBitmap::from_iter([0]);
        let repeated_fragments = vec![
            successor_fragments[0].clone(),
            with_deletions(fragment(1, 2, None), 2, &new_old_deleted),
            fragment(2, 1, None),
        ];
        let repeated_target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: 1,
        };
        let repeated_delta = placement_delta(
            &updated,
            domain(0, 4),
            [1],
            repeated_target,
            RowAddressPlacementKind::Selected,
        );
        let repeated_resolved = BTreeMap::from([(0, 2)]);
        let repeated_deletions = BTreeMap::from([(0, &old_deleted), (1, &new_old_deleted)]);
        let repeated = updated
            .apply_delta(
                &repeated_delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &successor_fragments,
                    successor_fragments: &repeated_fragments,
                    resolved_new_fragment_ids: &repeated_resolved,
                    current_deletion_vectors: &deletion_vectors,
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &repeated_deletions,
                    explicit_map_placements: &explicit,
                    commit_version: 3,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 456,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted {
            layout: repeated, ..
        } = repeated
        else {
            panic!("repeated update should be admitted");
        };
        assert_eq!(
            repeated
                .debt_summary
                .metadata_bytes_written_since_maintenance,
            579
        );
        let router =
            RowAddressRouter::try_new(Arc::from(repeated), &repeated_fragments, Some(0)).unwrap();
        assert_eq!(
            router.resolve_many(&[addresses[1]]).unwrap(),
            vec![PlacementResolution::Mapped {
                locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(2, 0))
            }]
        );
    }

    #[test]
    fn apply_delta_validates_source_before_explicit_map_admission() {
        let current_fragments = vec![fragment(0, 2, Some((0, 1)))];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(&current_fragments, Some(0));
        layout.debt_summary.live_physical_rows = 1;
        layout.refresh_fingerprint_with_fragments(&current_fragments, Some(0));

        let successor_fragments = vec![fragment(1, 2, None)];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: 2,
        };
        let mut delta = placement_delta(
            &layout,
            domain(0, 2),
            [0, 1],
            target,
            RowAddressPlacementKind::ExplicitMap,
        );
        delta.explicit_map_placements.insert(
            0,
            ExplicitMapRowAddressPlacement {
                sources: Vec::new(),
                object_path: String::new(),
                object_size: 0,
                pages: Vec::new(),
                destinations: Vec::new(),
                base_id: None,
            },
        );
        let resolved = BTreeMap::from([(0, 1)]);

        let error = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &current_fragments,
                    successor_fragments: &successor_fragments,
                    resolved_new_fragment_ids: &resolved,
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("more live rows than total rows"));
    }

    #[test]
    fn apply_delta_retires_only_source_snapshot_deletions() {
        let old_deleted = RoaringBitmap::from_iter([1, 3]);
        let current_fragment = with_deletions(fragment(0, 4, Some((0, 1))), 1, &old_deleted);
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));
        let successor = vec![fragment(1, 2, None)];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: 2,
        };
        let mut delta = placement_delta(
            &layout,
            domain(0, 4),
            [0, 2],
            target,
            RowAddressPlacementKind::Selected,
        );
        delta.retired_selections = vec![selection(0, [1, 3])];
        let current_deletions = BTreeMap::from([(0, &old_deleted)]);
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &successor,
                    resolved_new_fragment_ids: &BTreeMap::from([(0, 1)]),
                    current_deletion_vectors: &current_deletions,
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted {
            layout: compacted_layout,
            ..
        } = result
        else {
            panic!("compaction retirement should be admitted");
        };
        assert_eq!(compacted_layout.retired_rows.len(), 1);
        assert!(matches!(
            compacted_layout.retired_rows[0].membership,
            RetiredLogicalRowMembership::Selection(_)
        ));
        let mut retired_ranges = Vec::new();
        compacted_layout
            .visit_retired_ranges(|range| retired_ranges.push(range))
            .unwrap();
        assert_eq!(
            retired_ranges,
            vec![
                LogicalRowAddressRange::new(0, 1, 2),
                LogicalRowAddressRange::new(0, 3, 4),
            ]
        );
        let router =
            RowAddressRouter::try_new(Arc::from(compacted_layout.clone()), &successor, Some(0))
                .unwrap();
        let addresses = (0..4)
            .map(|slot| LogicalRowAddress::try_new_from_parts(0, slot).unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            router.resolve_many(&[addresses[0]]).unwrap()[0],
            PlacementResolution::Mapped { .. }
        ));
        assert!(matches!(
            router.resolve_many(&[addresses[1]]).unwrap()[0],
            PlacementResolution::NotLive
        ));
        let unknown = LogicalRowAddress::try_new_from_parts(1, 0).unwrap();
        assert_eq!(
            router.resolve_many(&[unknown]).unwrap(),
            vec![PlacementResolution::Unmapped]
        );
        let proto: pb::RowAddressLayout = compacted_layout.as_ref().into();
        let decoded = RowAddressLayout::try_from(proto).unwrap();
        assert_eq!(decoded.retired_rows, compacted_layout.retired_rows);

        let invalid_successor = vec![fragment(1, 1, None)];
        let invalid_target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: 1,
        };
        let mut invalid = placement_delta(
            &layout,
            domain(0, 4),
            [2],
            invalid_target,
            RowAddressPlacementKind::Selected,
        );
        invalid.expected_layout_fingerprint = layout.fingerprint.clone();
        invalid.retired_selections = vec![selection(0, [0, 1, 3])];
        let error = layout
            .apply_delta(
                &invalid,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &invalid_successor,
                    resolved_new_fragment_ids: &BTreeMap::from([(0, 1)]),
                    current_deletion_vectors: &current_deletions,
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("was live in source fragment"));
    }

    #[test]
    fn retired_rows_are_projected_to_index_coverage_before_expanding_full_domains() {
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.retired_rows = vec![
            RetiredLogicalRowSet::all_rows(vec![domain(0, 100_000_000)]).unwrap(),
            RetiredLogicalRowSet::selected(vec![domain(1, 8)], Arc::new(selection(1, [1, 3, 7])))
                .unwrap(),
        ];
        let coverage = RoaringTreemap::from_iter([
            LogicalRowAddress::try_new_from_parts(0, 5).unwrap().raw(),
            LogicalRowAddress::try_new_from_parts(0, 99_999_999)
                .unwrap()
                .raw(),
            LogicalRowAddress::try_new_from_parts(1, 2).unwrap().raw(),
            LogicalRowAddress::try_new_from_parts(1, 3).unwrap().raw(),
            LogicalRowAddress::try_new_from_parts(2, 0).unwrap().raw(),
        ]);

        let retired = layout
            .retired_logical_row_bitmap_for_coverage(&coverage)
            .unwrap();
        assert_eq!(
            retired.iter().collect::<Vec<_>>(),
            vec![
                LogicalRowAddress::try_new_from_parts(0, 5).unwrap().raw(),
                LogicalRowAddress::try_new_from_parts(0, 99_999_999)
                    .unwrap()
                    .raw(),
                LogicalRowAddress::try_new_from_parts(1, 3).unwrap().raw(),
            ]
        );
    }

    #[test]
    fn retired_projection_decodes_multi_domain_selection_once() {
        const DOMAIN_COUNT: u32 = 10_000;
        const SLOT_COUNT: u32 = 997;

        let mut retired_bitmap = RoaringTreemap::new();
        let mut coverage = RoaringTreemap::new();
        for logical_fragment_id in 0..DOMAIN_COUNT {
            for index in 0..10_u32 {
                let slot =
                    (logical_fragment_id.wrapping_mul(31) + index.wrapping_mul(97)) % SLOT_COUNT;
                retired_bitmap.insert((u64::from(logical_fragment_id) << 32) | u64::from(slot));
                if index == 0 {
                    coverage.insert((u64::from(logical_fragment_id) << 32) | u64::from(slot));
                }
            }
        }
        let retired_selection = LogicalRowAddressSelection::from_bitmap(retired_bitmap).unwrap();
        assert!(matches!(
            &retired_selection,
            LogicalRowAddressSelection::OrdinalEliasFano(_)
        ));
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.retired_rows = vec![
            RetiredLogicalRowSet::selected(
                (0..DOMAIN_COUNT)
                    .map(|logical_fragment_id| domain(logical_fragment_id, SLOT_COUNT))
                    .collect(),
                Arc::new(retired_selection),
            )
            .unwrap(),
        ];

        assert_eq!(
            layout
                .retired_logical_row_bitmap_for_coverage(&coverage)
                .unwrap(),
            coverage
        );
    }

    #[test]
    fn apply_delta_whole_fragment_delete_persists_all_rows_frontier() {
        let current_fragment = fragment(0, 4, Some((0, 1)));
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));
        let source = domain(0, 4);
        let delta = RowAddressLayoutDelta {
            source_domains: vec![source],
            retired_selections: vec![
                LogicalRowAddressSelection::from_full_domains(&[source]).unwrap(),
            ],
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..RowAddressLayoutDelta::default()
        };
        let fully_deleted = BTreeSet::from([0]);
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &[],
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &fully_deleted,
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted {
            layout: retired_layout,
            ..
        } = result
        else {
            panic!("whole-fragment retirement should be admitted");
        };
        assert_eq!(retired_layout.retired_rows.len(), 1);
        assert!(matches!(
            retired_layout.retired_rows[0].membership,
            RetiredLogicalRowMembership::AllRows
        ));
        let router = RowAddressRouter::try_new(Arc::from(retired_layout), &[], Some(0)).unwrap();
        let retired = LogicalRowAddress::try_new_from_parts(0, 2).unwrap();
        assert_eq!(
            router.resolve_many(&[retired]).unwrap(),
            vec![PlacementResolution::NotLive]
        );

        let error = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &[],
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("was live in source fragment"));
    }

    #[test]
    fn apply_delta_subtracts_packed_run_without_expanding_domains() {
        let current_fragment = fragment(10, 6, None);
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::PackedRun(
            PackedRunRowAddressPlacement::from_sources(vec![domain(0, 3), domain(1, 3)], 10, 0)
                .unwrap(),
        )];
        layout
            .refresh_physical_ownership_summary(&current_fragment, &RoaringBitmap::new())
            .unwrap();
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(1));

        let deleted = RoaringBitmap::from_iter([1]);
        let successor = vec![
            with_deletions(current_fragment.clone(), 1, &deleted),
            fragment(11, 1, None),
        ];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::ExistingFragmentId(11),
            start_offset: 0,
            end_offset: 1,
        };
        let delta = placement_delta(
            &layout,
            domain(0, 3),
            [1],
            target,
            RowAddressPlacementKind::Selected,
        );
        let deletions = BTreeMap::from([(10, &deleted)]);
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &successor,
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &deletions,
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(1),
                    max_logical_fragment_id: Some(1),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
            panic!("PackedRun subtraction should be admitted");
        };
        let router = RowAddressRouter::try_new(Arc::from(layout), &successor, Some(1)).unwrap();
        assert_eq!(
            router.logical_domain_destination_ranges(1).unwrap(),
            vec![LogicalDomainDestination::Inline(PhysicalRowRange {
                physical_fragment_id: 10,
                start_offset: 3,
                end_offset: 6,
            })]
        );
    }

    #[test]
    fn apply_delta_rejects_duplicate_successor_fragment_ids() {
        let current_fragment = fragment(0, 4, Some((0, 1)));
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));
        let delta = RowAddressLayoutDelta {
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..Default::default()
        };
        let successor = vec![current_fragment.clone(), current_fragment.clone()];

        let error = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &successor,
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("successor snapshot contains duplicate physical fragment id 0")
        );
    }

    #[test]
    fn apply_delta_canonicalizes_shuffled_successor_ownership() {
        let current_fragments = vec![fragment(10, 1, None), fragment(11, 1, None)];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(0, 1),
                destination_fragment_id: 10,
                destination_start: 0,
                excluded: None,
            }),
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(1, 1),
                destination_fragment_id: 11,
                destination_start: 0,
                excluded: None,
            }),
        ];
        for fragment in &current_fragments {
            layout
                .refresh_physical_ownership_summary(fragment, &RoaringBitmap::new())
                .unwrap();
        }
        layout.refresh_fingerprint_with_fragments(&current_fragments, Some(1));
        let delta = RowAddressLayoutDelta {
            expected_layout_fingerprint: layout.fingerprint.clone(),
            ..Default::default()
        };
        let mut successor_fragments = current_fragments.clone();
        successor_fragments.reverse();

        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &current_fragments,
                    successor_fragments: &successor_fragments,
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &BTreeMap::new(),
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(1),
                    max_logical_fragment_id: Some(1),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
            panic!("unchanged shuffled successor should be admitted");
        };

        assert_eq!(
            layout
                .physical_row_ownership
                .iter()
                .map(|summary| summary.physical_fragment_id)
                .collect::<Vec<_>>(),
            vec![10, 11]
        );
        layout
            .validate_with_fragments(&successor_fragments, Some(1))
            .unwrap();
    }

    #[test]
    fn sparse_selection_subtraction_does_not_create_empty_exclusions() {
        let placement = RowAddressPlacement::SparseSelection(SparseSelectionRowAddressPlacement {
            sources: vec![
                SparseSelectionSource {
                    source: domain(0, 2),
                    selection: selection(0, [0]).into(),
                    excluded: None,
                },
                SparseSelectionSource {
                    source: domain(0, 2),
                    selection: selection(0, [1]).into(),
                    excluded: None,
                },
            ],
            destination_fragment_id: 10,
            destination_start: 0,
        });
        let removed = BTreeMap::from([(0, RoaringBitmap::from_iter([0]))]);
        let mut preserved = Vec::new();
        let mut residual = Vec::new();

        assert!(
            subtract_placement(&placement, &removed, &mut preserved, &mut residual)
                .unwrap()
                .is_none()
        );
        assert!(residual.is_empty());
        assert_eq!(preserved.len(), 1);
        preserved[0].validate().unwrap();
        let RowAddressPlacement::SparseSelection(sparse) = &preserved[0] else {
            panic!("unaffected source should remain a sparse selection")
        };
        assert_eq!(sparse.destination_start, 1);
        assert_eq!(sparse.sources.len(), 1);
        assert_eq!(sparse.sources[0].selection.as_ref(), &selection(0, [1]));
        assert!(sparse.sources[0].excluded.is_none());
    }

    #[test]
    fn apply_delta_subtracts_nonuniform_gapped_packed_run() {
        let current_fragment = fragment(10, 9, None);
        let sources = vec![domain(0, 2), domain(2, 3), domain(5, 4)];
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::PackedRun(
            PackedRunRowAddressPlacement::from_sources(sources, 10, 0).unwrap(),
        )];
        layout
            .refresh_physical_ownership_summary(&current_fragment, &RoaringBitmap::new())
            .unwrap();
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(5));

        let deleted = RoaringBitmap::from_iter([3]);
        let successor = vec![
            with_deletions(current_fragment.clone(), 1, &deleted),
            fragment(11, 1, None),
        ];
        let delta = placement_delta(
            &layout,
            domain(2, 3),
            [1],
            RowAddressTargetRange {
                fragment: RowAddressTargetFragment::ExistingFragmentId(11),
                start_offset: 0,
                end_offset: 1,
            },
            RowAddressPlacementKind::Selected,
        );
        let deletions = BTreeMap::from([(10, &deleted)]);

        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &successor,
                    resolved_new_fragment_ids: &BTreeMap::new(),
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &deletions,
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(5),
                    max_logical_fragment_id: Some(5),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
            panic!("nonuniform PackedRun subtraction should be admitted");
        };
        let router = RowAddressRouter::try_new(Arc::from(layout), &successor, Some(5)).unwrap();
        let logical = [(0, 1), (2, 0), (2, 1), (2, 2), (5, 3)].map(|(fragment_id, slot)| {
            LogicalRowAddress::try_new_from_parts(fragment_id, slot).unwrap()
        });
        assert_eq!(
            router.resolve_many(&logical).unwrap(),
            vec![
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(10, 1)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(10, 2)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(11, 0)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(10, 4)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(10, 8)),
                },
            ]
        );
    }

    #[test]
    fn apply_delta_canonical_result_is_history_independent() {
        let current_fragment = fragment(10, 4, None);
        let shared = Arc::new(
            LogicalRowAddressSelection::from_ranges(vec![LogicalRowAddressRange::new(0, 0, 4)])
                .unwrap(),
        );
        let mut selected = RowAddressLayout::new(Uuid::new_v4());
        selected.placements = vec![RowAddressPlacement::Selected(SelectedRowAddressPlacement {
            source: domain(0, 4),
            selection: shared,
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: None,
        })];
        selected
            .refresh_physical_ownership_summary(&current_fragment, &RoaringBitmap::new())
            .unwrap();
        selected
            .refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));
        let mut extent = RowAddressLayout::new(selected.namespace_uuid);
        extent.placements = vec![RowAddressPlacement::ExtentList(
            ExtentListRowAddressPlacement {
                source: domain(0, 4),
                extents: vec![RowAddressExtent {
                    source_start: 0,
                    length: 4,
                    destination_fragment_id: 10,
                    destination_start: 0,
                }],
            },
        )];
        extent
            .refresh_physical_ownership_summary(&current_fragment, &RoaringBitmap::new())
            .unwrap();
        extent.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));
        let deleted = RoaringBitmap::from_iter([1]);
        let successor = vec![
            with_deletions(current_fragment.clone(), 1, &deleted),
            fragment(11, 1, None),
        ];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::ExistingFragmentId(11),
            start_offset: 0,
            end_offset: 1,
        };
        let deletions = BTreeMap::from([(10, &deleted)]);
        let apply = |layout: &RowAddressLayout| {
            let delta = placement_delta(
                layout,
                domain(0, 4),
                [1],
                target,
                RowAddressPlacementKind::Selected,
            );
            let result = layout
                .apply_delta(
                    &delta,
                    &RowAddressDeltaApplyContext {
                        current_fragments: std::slice::from_ref(&current_fragment),
                        successor_fragments: &successor,
                        resolved_new_fragment_ids: &BTreeMap::new(),
                        current_deletion_vectors: &BTreeMap::new(),
                        newly_fully_deleted_source_fragments: &BTreeSet::new(),
                        deletion_vectors: &deletions,
                        explicit_map_placements: &BTreeMap::new(),
                        commit_version: 2,
                        current_max_logical_fragment_id: Some(0),
                        max_logical_fragment_id: Some(0),
                        row_address_metadata_bytes_written: 0,
                    },
                )
                .unwrap();
            let RowAddressLayoutApplyResult::Admitted { layout, .. } = result else {
                panic!("canonical update should be admitted");
            };
            layout
        };
        let selected_result = apply(&selected);
        let extent_result = apply(&extent);
        assert_eq!(selected_result.placements, extent_result.placements);
        assert_eq!(selected_result.fingerprint, extent_result.fingerprint);
    }

    #[test]
    fn sparse_native_update_uses_inline_exclusion_without_mutating_current_layout() {
        let current_fragment = fragment(0, 100, Some((0, 1)));
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));
        let original = layout.clone();
        let touched = (0..66_u32).step_by(2).collect::<Vec<_>>();
        let deleted = RoaringBitmap::from_iter(touched.iter().copied());
        let successor = vec![
            with_deletions(fragment(0, 100, None), 1, &deleted),
            fragment(1, touched.len(), None),
        ];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: touched.len() as u32,
        };
        let delta = placement_delta(
            &layout,
            domain(0, 100),
            touched,
            target,
            RowAddressPlacementKind::Selected,
        );
        let resolved = BTreeMap::from([(0, 1)]);
        let deletions = BTreeMap::from([(0, &deleted)]);
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &successor,
                    resolved_new_fragment_ids: &resolved,
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &deletions,
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout: result, .. } = result else {
            panic!("inline exclusion should avoid extent-fanout backpressure");
        };
        assert!(matches!(
            &result.placements[0],
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                excluded: Some(excluded),
                ..
            }) if excluded.cardinality() == 33
        ));
        let proto: pb::RowAddressLayout = result.as_ref().into();
        let round_trip = RowAddressLayout::try_from(proto).unwrap();
        assert_eq!(&round_trip, result.as_ref());
        let router = RowAddressRouter::try_new(Arc::new(round_trip), &successor, Some(0)).unwrap();
        let logical = [
            LogicalRowAddress::try_new_from_parts(0, 0).unwrap(),
            LogicalRowAddress::try_new_from_parts(0, 1).unwrap(),
        ];
        assert_eq!(
            router.resolve_many(&logical).unwrap(),
            vec![
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(1, 0)),
                },
                PlacementResolution::Mapped {
                    locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(0, 1)),
                },
            ]
        );
        assert_eq!(
            router
                .physical_to_logical_many(&[RowAddress::new_from_parts(0, 0)])
                .unwrap(),
            vec![PhysicalToLogicalResolution::Unmapped]
        );
        assert_eq!(layout, original);
    }

    #[test]
    #[ignore = "release-scale 100M/1M placement admission benchmark"]
    fn one_percent_random_update_on_100m_domain_stays_entropy_bounded() {
        const ROWS: u32 = 100_000_000;
        const TOUCHED: u32 = 1_000_000;
        let started = std::time::Instant::now();

        let current_fragment = fragment(0, ROWS as usize, Some((0, 1)));
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&current_fragment), Some(0));

        let mut deleted = RoaringBitmap::new();
        for index in 0..TOUCHED {
            deleted.insert(((index as u64 * 99_991) % ROWS as u64) as u32);
        }
        assert_eq!(deleted.len(), TOUCHED as u64);
        let successor = vec![
            with_deletions(fragment(0, ROWS as usize, None), 1, &deleted),
            fragment(1, TOUCHED as usize, None),
        ];
        let target = RowAddressTargetRange {
            fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
            start_offset: 0,
            end_offset: TOUCHED,
        };
        let delta = placement_delta(
            &layout,
            domain(0, ROWS),
            deleted.iter(),
            target,
            RowAddressPlacementKind::Selected,
        );
        let resolved = BTreeMap::from([(0, 1)]);
        let deletions = BTreeMap::from([(0, &deleted)]);
        let result = layout
            .apply_delta(
                &delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: std::slice::from_ref(&current_fragment),
                    successor_fragments: &successor,
                    resolved_new_fragment_ids: &resolved,
                    current_deletion_vectors: &BTreeMap::new(),
                    newly_fully_deleted_source_fragments: &BTreeSet::new(),
                    deletion_vectors: &deletions,
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 2,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted { layout, metrics } = result else {
            panic!("isolated 1% random update must fit the inline admission envelope");
        };
        assert!(
            layout
                .placements
                .iter()
                .all(|placement| !matches!(placement, RowAddressPlacement::ExtentList(_)))
        );
        assert_eq!(metrics.max_extent_fanout, 2);
        assert!(
            metrics.projected_layout_bytes <= ROW_ADDRESS_B_FAST,
            "entropy-bounded layout used {} bytes",
            metrics.projected_layout_bytes
        );
        assert!(matches!(
            &layout.placements[0],
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                excluded: Some(excluded),
                ..
            }) if excluded.cardinality() == TOUCHED as u64
        ));

        let repeated_successor = vec![successor[0].clone(), fragment(2, TOUCHED as usize, None)];
        let repeated_delta = placement_delta(
            &layout,
            domain(0, ROWS),
            deleted.iter(),
            RowAddressTargetRange {
                fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
                start_offset: 0,
                end_offset: TOUCHED,
            },
            RowAddressPlacementKind::Selected,
        );
        let repeated_resolved = BTreeMap::from([(0, 2)]);
        let fully_deleted = BTreeSet::from([1]);
        let repeated_result = layout
            .apply_delta(
                &repeated_delta,
                &RowAddressDeltaApplyContext {
                    current_fragments: &successor,
                    successor_fragments: &repeated_successor,
                    resolved_new_fragment_ids: &repeated_resolved,
                    current_deletion_vectors: &deletions,
                    newly_fully_deleted_source_fragments: &fully_deleted,
                    deletion_vectors: &deletions,
                    explicit_map_placements: &BTreeMap::new(),
                    commit_version: 3,
                    current_max_logical_fragment_id: Some(0),
                    max_logical_fragment_id: Some(0),
                    row_address_metadata_bytes_written: 0,
                },
            )
            .unwrap();
        let RowAddressLayoutApplyResult::Admitted {
            layout: repeated_layout,
            metrics: repeated_metrics,
        } = repeated_result
        else {
            panic!("repeated fixed-hot-set update must remain admitted");
        };
        assert!(repeated_metrics.projected_layout_bytes <= ROW_ADDRESS_B_FAST);
        let router =
            RowAddressRouter::try_new(Arc::from(repeated_layout), &repeated_successor, Some(0))
                .unwrap();
        let touched_address =
            LogicalRowAddress::try_new_from_parts(0, deleted.min().unwrap()).unwrap();
        assert_eq!(
            router.resolve_many(&[touched_address]).unwrap(),
            vec![PlacementResolution::Mapped {
                locator: PhysicalRowLocator::Physical(RowAddress::new_from_parts(2, 0)),
            }]
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(60),
            "100M/1M repeated placement admission exceeded the optimized benchmark budget"
        );
    }

    #[test]
    fn sorted_destination_ownership_ranges_are_validated_linearly() {
        let started = std::time::Instant::now();
        let mut occupancy = DomainOccupancy::default();
        for index in 0..100_000_u32 {
            occupancy
                .add_range(index * 2, index * 2 + 1, "linear validation test")
                .unwrap();
        }
        occupancy.validate().unwrap();
        assert_eq!(occupancy.ranges.len(), 100_000);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "sorted destination ownership validation regressed from linear behavior"
        );
    }

    #[test]
    fn fifty_million_clustered_deletions_fingerprint_by_runs() {
        let mut deleted = RoaringBitmap::new();
        deleted.insert_range(25_000_000..75_000_000);
        let started = std::time::Instant::now();
        assert_eq!(
            fingerprint_deleted_offsets(&deleted),
            offset_ranges_fingerprint([(25_000_000, 75_000_000)])
        );
        if !cfg!(debug_assertions) {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(1),
                "clustered deletion fingerprints must scale with Roaring runs"
            );
        }
    }

    #[test]
    fn high_entropy_touched_slots_use_one_bitmap_ownership_proof() {
        const ROWS: u32 = 20_000;
        let touched = RoaringBitmap::from_iter((0..ROWS).step_by(2));
        let touched_selection = Arc::new(selection(0, touched.iter()));
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![
            RowAddressPlacement::Direct(DirectRowAddressPlacement {
                source: domain(0, ROWS),
                destination_fragment_id: 10,
                destination_start: 0,
                excluded: Some(touched_selection.clone()),
            }),
            RowAddressPlacement::Selected(SelectedRowAddressPlacement {
                source: domain(0, ROWS),
                selection: touched_selection,
                destination_fragment_id: 11,
                destination_start: 0,
                excluded: None,
            }),
        ];
        let router = RowAddressRouter {
            layout: Arc::new(layout),
            source_index: vec![(0, 0), (0, 1)],
            packed_source_runs: Vec::new(),
            native_domains: Vec::new(),
            native_by_physical: Vec::new(),
        };

        assert_eq!(
            router
                .logical_selection_inline_ownership(0, &touched)
                .unwrap(),
            Some(RoaringBitmap::from_iter([11]))
        );
        let mut all = RoaringBitmap::new();
        all.insert_range(0..ROWS);
        assert_eq!(
            router.logical_selection_inline_ownership(0, &all).unwrap(),
            Some(RoaringBitmap::from_iter([10, 11]))
        );
    }

    #[test]
    #[ignore = "release-scale 100M/1M row-address validation benchmark"]
    fn streaming_validation_handles_100m_rows_with_1m_exclusions() {
        const ROWS: u32 = 100_000_000;
        const EXCLUDED: u32 = 1_000_000;

        let mut deleted = RoaringBitmap::new();
        for index in 0..EXCLUDED {
            deleted.insert(((index as u64 * 99_991) % ROWS as u64) as u32);
        }
        let excluded = Arc::new(selection(0, deleted.iter()));
        let target = with_deletions(fragment(10, ROWS as usize, None), 1, &deleted);
        let mut layout = RowAddressLayout::new(Uuid::new_v4());
        layout.placements = vec![RowAddressPlacement::Direct(DirectRowAddressPlacement {
            source: domain(0, ROWS),
            destination_fragment_id: 10,
            destination_start: 0,
            excluded: Some(excluded),
        })];
        layout
            .refresh_physical_ownership_summary(&target, &deleted)
            .unwrap();
        layout.refresh_fingerprint_with_fragments(std::slice::from_ref(&target), Some(0));

        let started = std::time::Instant::now();
        layout
            .validate_with_fragments(std::slice::from_ref(&target), Some(0))
            .unwrap();
        layout.verify_visibility(&target, &deleted).unwrap();
        let elapsed = started.elapsed();
        if !cfg!(debug_assertions) {
            let budget = std::time::Duration::from_secs(5);
            assert!(
                elapsed < budget,
                "streaming 100M/1M row-address validation took {elapsed:?}, budget {budget:?}"
            );
        }

        layout.physical_row_ownership[0].mapped_offsets_fingerprint[0] ^= 1;
        assert!(
            layout
                .validate_with_fragments(std::slice::from_ref(&target), Some(0))
                .is_err(),
            "streaming validation must still detect ownership corruption"
        );
    }

    #[test]
    fn full_row_address_delta_admission_is_evaluated_outside_layout_application() {
        assert_eq!(
            evaluate_projected_row_address_delta(ROW_ADDRESS_B_FAST),
            None
        );
        assert_eq!(
            evaluate_projected_row_address_delta(ROW_ADDRESS_B_FAST + 1),
            Some(PlacementMaintenanceRequired::ProjectedDeltaBytes {
                projected: ROW_ADDRESS_B_FAST + 1,
                limit: ROW_ADDRESS_B_FAST,
            })
        );
        assert_eq!(
            evaluate_projected_row_address_epoch(ROW_ADDRESS_W_FAST),
            None
        );
        assert_eq!(
            evaluate_projected_row_address_epoch(ROW_ADDRESS_W_FAST + 1),
            Some(PlacementMaintenanceRequired::ProjectedEpochBytes {
                projected: ROW_ADDRESS_W_FAST + 1,
                limit: ROW_ADDRESS_W_FAST,
            })
        );
    }
}
