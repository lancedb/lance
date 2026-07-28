// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Adopter-private selection for chunk-local variable-width offsets.

use std::sync::Arc;

use prost::Message;

use crate::{
    buffer::LanceBuffer,
    compression::block::{
        MAX_DICTIONARY_ITEMS, fixed_from_u64_values, validate_fixed_payload_len,
        visit_unsigned_values,
    },
    compression::{BlockCompressor, BlockValueType},
    compression_config::CompressionFieldParams,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::{
        delta::encode_deltas,
        dictionary::{self, DictionaryBlockCompressor},
        rle::{self, BlockRleCompressor},
    },
    format::{ProtobufUtils21, pb21::CompressiveEncoding},
};
use lance_core::{Error, Result};

mod selector;
mod statistics;

use selector::{
    Candidate, constant_selection, direct_candidates, estimate_payload, flat_selection,
    range_selection,
};
use statistics::{BoundedDistinctStatsBuilder, SequenceStats, SequenceStatsBuilder};

const GENERAL_SAMPLE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BlockCost {
    buffer_overhead_bytes: u64,
    alignment: u64,
}

impl BlockCost {
    pub(super) fn new(buffer_overhead_bytes: u64, alignment: u64) -> Self {
        Self {
            buffer_overhead_bytes,
            alignment: alignment.max(1),
        }
    }

    fn payload_wire_bytes(self, has_payload: bool, payload_bytes: u64) -> u64 {
        align_wire_bytes(payload_bytes, self.alignment.max(1)).saturating_add(if has_payload {
            self.buffer_overhead_bytes
        } else {
            0
        })
    }
}

fn align_wire_bytes(bytes: u64, alignment: u64) -> u64 {
    bytes
        .checked_add(alignment - 1)
        .map(|padded| padded / alignment * alignment)
        .unwrap_or(u64::MAX)
}

#[derive(Debug)]
struct SelectedBlockCompressor {
    compressor: Box<dyn BlockCompressor>,
    encoding: CompressiveEncoding,
    has_payload: bool,
}

impl SelectedBlockCompressor {
    fn new(
        compressor: Box<dyn BlockCompressor>,
        encoding: CompressiveEncoding,
        has_payload: bool,
    ) -> Self {
        Self {
            compressor,
            encoding,
            has_payload,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum OffsetMetadataPattern {
    Constant(u64),
    Range { start: u64, step: u64 },
}

fn probe_offset_metadata(
    data: &FixedWidthDataBlock,
    ranges: &[std::ops::Range<usize>],
) -> Result<Option<OffsetMetadataPattern>> {
    let value_type = BlockValueType::from_bits(data.bits_per_value)?;
    validate_fixed_payload_len(
        &data.data,
        value_type,
        data.num_values,
        "Block family input",
    )?;
    match value_type {
        BlockValueType::UInt32 => {
            let values = data.data.borrow_to_typed_view::<u32>();
            probe_typed_offset_metadata(values.as_ref(), ranges, u64::from)
        }
        BlockValueType::UInt64 => {
            let values = data.data.borrow_to_typed_view::<u64>();
            probe_typed_offset_metadata(values.as_ref(), ranges, |value| value)
        }
        _ => Err(Error::invalid_input(format!(
            "Generic block family selection only supports u32 or u64, got {} bits",
            value_type.bits_per_value()
        ))),
    }
}

fn probe_typed_offset_metadata<T: Copy>(
    values: &[T],
    ranges: &[std::ops::Range<usize>],
    to_u64: impl Fn(T) -> u64 + Copy,
) -> Result<Option<OffsetMetadataPattern>> {
    let mut common_step = None;
    let mut all_constant = true;
    let mut range_possible = true;

    for (index, range) in ranges.iter().enumerate() {
        let member = values.get(range.clone()).ok_or_else(|| {
            Error::invalid_input(format!(
                "Block family member {index} range {}..{} exceeds {} values",
                range.start,
                range.end,
                values.len()
            ))
        })?;
        let base =
            member.first().copied().map(to_u64).ok_or_else(|| {
                Error::invalid_input("Variable-width offset chunks cannot be empty")
            })?;
        let mut previous = base;
        for value in member.iter().copied().skip(1).map(to_u64) {
            let delta = value.checked_sub(previous).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Block family member {index} must be non-decreasing"
                ))
            })?;
            all_constant &= delta == 0;
            match common_step {
                Some(step) => range_possible &= delta == step,
                None => common_step = Some(delta),
            }
            previous = value;
            if !all_constant && !range_possible {
                return Ok(None);
            }
        }
        range_possible &= member.len() >= 2;
    }

    if all_constant {
        return Ok(Some(OffsetMetadataPattern::Constant(0)));
    }
    Ok(match (range_possible, common_step) {
        (true, Some(step)) if step > 0 => Some(OffsetMetadataPattern::Range { start: 0, step }),
        _ => None,
    })
}

#[derive(Debug, Clone)]
pub(super) struct OffsetSequenceAnalysis {
    pub(super) values: SequenceStats,
    pub(super) deltas: Option<SequenceStats>,
}

#[derive(Debug)]
pub(super) struct OffsetRunAnalysis {
    pub(super) values: SequenceStats,
    pub(super) lengths: SequenceStats,
}

#[derive(Debug)]
pub(super) struct OffsetFamilyAnalysis {
    pub(super) members: Vec<OffsetSequenceAnalysis>,
    pub(super) runs: Option<Vec<OffsetRunAnalysis>>,
    pub(super) dictionary_items: Option<Arc<[u64]>>,
}

pub(super) fn analyze_offset_members(
    data: &FixedWidthDataBlock,
    ranges: &[std::ops::Range<usize>],
    collect_sample: bool,
) -> Result<OffsetFamilyAnalysis> {
    let value_type = BlockValueType::from_bits(data.bits_per_value)?;
    match value_type {
        BlockValueType::UInt32 => {
            let values = data.data.borrow_to_typed_view::<u32>();
            analyze_typed_offsets::<_, false>(
                values.as_ref(),
                ranges,
                value_type,
                collect_sample,
                u64::from,
            )
        }
        BlockValueType::UInt64 => {
            let values = data.data.borrow_to_typed_view::<u64>();
            analyze_typed_offsets::<_, true>(
                values.as_ref(),
                ranges,
                value_type,
                collect_sample,
                |value| value,
            )
        }
        _ => Err(Error::invalid_input(format!(
            "Generic block family selection only supports u32 or u64, got {} bits",
            value_type.bits_per_value()
        ))),
    }
}

fn analyze_typed_offsets<T: Copy, const COLLECT_DICTIONARY: bool>(
    values: &[T],
    ranges: &[std::ops::Range<usize>],
    value_type: BlockValueType,
    collect_sample: bool,
    to_u64: impl Fn(T) -> u64 + Copy,
) -> Result<OffsetFamilyAnalysis> {
    if !collect_sample {
        return analyze_typed_offsets_without_samples::<_, COLLECT_DICTIONARY>(
            values, ranges, value_type, to_u64,
        );
    }
    if ranges.is_empty() {
        return Err(Error::invalid_input(
            "Variable-width offsets require at least one chunk",
        ));
    }
    let sample_limit = if collect_sample {
        GENERAL_SAMPLE_BYTES / ranges.len()
    } else {
        0
    };
    let mut dictionary_items = COLLECT_DICTIONARY.then(BoundedDistinctStatsBuilder::new);
    let members = ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let member = values.get(range.clone()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Block family member {index} range {}..{} exceeds {} values",
                    range.start,
                    range.end,
                    values.len()
                ))
            })?;
            let base = member.first().copied().map(to_u64).ok_or_else(|| {
                Error::invalid_input("Variable-width offset chunks cannot be empty")
            })?;
            let mut value_stats = SequenceStatsBuilder::with_sample_limit(value_type, sample_limit);
            let mut delta_stats = SequenceStatsBuilder::with_sample_limit(value_type, sample_limit);
            let mut previous = None;
            for value in member.iter().copied().map(to_u64) {
                let value = value.checked_sub(base).ok_or_else(|| {
                    Error::invalid_input(
                        "Block family member is smaller than its normalization base",
                    )
                })?;
                value_stats.push(value);
                if COLLECT_DICTIONARY {
                    dictionary_items
                        .as_mut()
                        .expect("dictionary collection is enabled")
                        .push(value);
                }
                if let Some(previous) = previous {
                    delta_stats.push(value.checked_sub(previous).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Block family member {index} must be non-decreasing"
                        ))
                    })?);
                }
                previous = Some(value);
            }
            let value_stats = value_stats.finish();
            if !value_stats.is_non_decreasing {
                return Err(Error::invalid_input(format!(
                    "Block family member {index} must be non-decreasing"
                )));
            }
            let deltas = (value_stats.len >= 2).then(|| delta_stats.finish());
            Ok(OffsetSequenceAnalysis {
                values: value_stats,
                deltas,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let runs = analyze_offset_runs(values, ranges, value_type, collect_sample, to_u64, &members)?;
    Ok(OffsetFamilyAnalysis {
        members,
        runs,
        dictionary_items: dictionary_items.and_then(|items| items.finish()),
    })
}

fn analyze_typed_offsets_without_samples<T: Copy, const COLLECT_DICTIONARY: bool>(
    values: &[T],
    ranges: &[std::ops::Range<usize>],
    value_type: BlockValueType,
    to_u64: impl Fn(T) -> u64 + Copy,
) -> Result<OffsetFamilyAnalysis> {
    if ranges.is_empty() {
        return Err(Error::invalid_input(
            "Variable-width offsets require at least one chunk",
        ));
    }
    let mut dictionary_items = COLLECT_DICTIONARY.then(BoundedDistinctStatsBuilder::new);
    let members = ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let member = values.get(range.clone()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Block family member {index} range {}..{} exceeds {} values",
                    range.start,
                    range.end,
                    values.len()
                ))
            })?;
            let first_raw = member.first().copied().map(to_u64).ok_or_else(|| {
                Error::invalid_input("Variable-width offset chunks cannot be empty")
            })?;
            let base = first_raw;
            let first = 0_u64;
            let mut previous = first;
            let mut max = first;
            let mut run_count = 1_u64;

            let mut delta_len = 0_u64;
            let mut delta_first = None;
            let mut delta_previous = None;
            let mut delta_min = u64::MAX;
            let mut delta_max = 0_u64;
            let mut deltas_non_decreasing = true;
            let mut delta_step = None;
            let mut deltas_are_arithmetic = true;
            let mut delta_run_count = 0_u64;

            if COLLECT_DICTIONARY {
                dictionary_items
                    .as_mut()
                    .expect("dictionary collection is enabled")
                    .push(first);
            }
            for value in member.iter().copied().skip(1).map(to_u64) {
                let value = value.checked_sub(base).ok_or_else(|| {
                    Error::invalid_input(
                        "Block family member is smaller than its normalization base",
                    )
                })?;
                let delta = value.checked_sub(previous).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Block family member {index} must be non-decreasing"
                    ))
                })?;
                if value != previous {
                    run_count += 1;
                }
                previous = value;
                max = value;
                if COLLECT_DICTIONARY {
                    dictionary_items
                        .as_mut()
                        .expect("dictionary collection is enabled")
                        .push(value);
                }

                if delta_first.is_none() {
                    delta_first = Some(delta);
                    delta_run_count = 1;
                }
                if let Some(previous_delta) = delta_previous {
                    deltas_non_decreasing &= delta >= previous_delta;
                    if delta != previous_delta {
                        delta_run_count += 1;
                    }
                    let difference = delta.checked_sub(previous_delta);
                    match (delta_step, difference) {
                        (None, Some(difference)) if delta_len == 1 => {
                            delta_step = Some(difference);
                        }
                        (Some(step), Some(difference)) if step == difference => {}
                        _ => deltas_are_arithmetic = false,
                    }
                }
                delta_previous = Some(delta);
                delta_min = delta_min.min(delta);
                delta_max = delta_max.max(delta);
                delta_len += 1;
            }

            let deltas = (delta_len > 0).then(|| SequenceStats {
                value_type,
                len: delta_len,
                first: delta_first,
                min: delta_min,
                max: delta_max,
                is_non_decreasing: deltas_non_decreasing,
                arithmetic_step: (delta_len >= 2 && deltas_are_arithmetic)
                    .then_some(delta_step)
                    .flatten(),
                run_count: delta_run_count,
                sample: None,
            });
            Ok(OffsetSequenceAnalysis {
                values: SequenceStats {
                    value_type,
                    len: member.len() as u64,
                    first: Some(first),
                    min: first,
                    max,
                    is_non_decreasing: true,
                    arithmetic_step: None,
                    run_count,
                    sample: None,
                },
                deltas,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let runs = analyze_offset_runs(values, ranges, value_type, false, to_u64, &members)?;
    Ok(OffsetFamilyAnalysis {
        members,
        runs,
        dictionary_items: dictionary_items.and_then(BoundedDistinctStatsBuilder::finish),
    })
}

fn analyze_offset_runs<T: Copy>(
    values: &[T],
    ranges: &[std::ops::Range<usize>],
    value_type: BlockValueType,
    collect_sample: bool,
    to_u64: impl Fn(T) -> u64 + Copy,
    members: &[OffsetSequenceAnalysis],
) -> Result<Option<Vec<OffsetRunAnalysis>>> {
    let (total_values, total_runs) =
        members
            .iter()
            .try_fold((0_u64, 0_u64), |(values, runs), member| {
                Ok::<_, Error>((
                    values.checked_add(member.values.len).ok_or_else(|| {
                        Error::invalid_input("Offset family cardinality overflows u64")
                    })?,
                    runs.checked_add(member.values.run_count).ok_or_else(|| {
                        Error::invalid_input("Offset family run count overflows u64")
                    })?,
                ))
            })?;
    if total_runs == 0 || total_runs >= total_values {
        return Ok(None);
    }

    let sample_limit = if collect_sample {
        GENERAL_SAMPLE_BYTES / ranges.len()
    } else {
        0
    };
    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let member = values.get(range.clone()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Block family member {index} range {}..{} exceeds {} values",
                    range.start,
                    range.end,
                    values.len()
                ))
            })?;
            let base = member.first().copied().map(to_u64).ok_or_else(|| {
                Error::invalid_input("Variable-width offset chunks cannot be empty")
            })?;
            let mut run_values = SequenceStatsBuilder::with_sample_limit(value_type, sample_limit);
            let mut run_lengths = SequenceStatsBuilder::without_sample(BlockValueType::UInt64);
            let mut current = None;
            let mut length = 0_u64;
            for value in member.iter().copied().map(to_u64) {
                let value = value.checked_sub(base).ok_or_else(|| {
                    Error::invalid_input(
                        "Block family member is smaller than its normalization base",
                    )
                })?;
                match current {
                    Some(run_value) if run_value == value => {
                        length = length.checked_add(1).ok_or_else(|| {
                            Error::invalid_input("Offset run length overflows u64")
                        })?;
                    }
                    Some(run_value) => {
                        run_values.push(run_value);
                        run_lengths.push(length);
                        current = Some(value);
                        length = 1;
                    }
                    None => {
                        current = Some(value);
                        length = 1;
                    }
                }
            }
            if let Some(run_value) = current {
                run_values.push(run_value);
                run_lengths.push(length);
            }
            Ok(OffsetRunAnalysis {
                values: run_values.finish(),
                lengths: run_lengths.finish(),
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

pub(super) fn offset_member_base(
    block: &FixedWidthDataBlock,
    value_type: BlockValueType,
) -> Result<u64> {
    if block.num_values == 0 {
        return Err(Error::invalid_input(
            "Variable-width offset chunks cannot be empty",
        ));
    }
    Ok(match value_type {
        BlockValueType::UInt8 => u64::from(block.data[0]),
        BlockValueType::UInt16 => u64::from(block.data.borrow_to_typed_view::<u16>().as_ref()[0]),
        BlockValueType::UInt32 => u64::from(block.data.borrow_to_typed_view::<u32>().as_ref()[0]),
        BlockValueType::UInt64 => block.data.borrow_to_typed_view::<u64>().as_ref()[0],
    })
}

pub(super) fn normalize_offset_member(block: FixedWidthDataBlock) -> Result<FixedWidthDataBlock> {
    let value_type = BlockValueType::from_bits(block.bits_per_value)?;
    validate_fixed_payload_len(
        &block.data,
        value_type,
        block.num_values,
        "Block family member",
    )?;
    let base = offset_member_base(&block, value_type)?;
    if base == 0 {
        return Ok(block);
    }
    let capacity = usize::try_from(block.num_values)
        .map_err(|_| Error::invalid_input("Block family cardinality does not fit usize"))?;
    let mut normalized = Vec::with_capacity(capacity);
    visit_unsigned_values(&block, value_type, |value| {
        normalized.push(value.checked_sub(base).ok_or_else(|| {
            Error::invalid_input("Block family member is smaller than its normalization base")
        })?);
        Ok(())
    })?;
    fixed_from_u64_values(&normalized, value_type, "Block family member")
}

#[derive(Debug)]
struct NormalizedOffsetCompressor {
    child: Box<dyn BlockCompressor>,
}

impl NormalizedOffsetCompressor {
    fn new(child: Box<dyn BlockCompressor>) -> Self {
        Self { child }
    }
}

impl BlockCompressor for NormalizedOffsetCompressor {
    fn compress(&self, data: DataBlock) -> Result<Option<LanceBuffer>> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Offset compression requires fixed-width data",
            ));
        };
        self.child
            .compress(DataBlock::FixedWidth(normalize_offset_member(data)?))
    }
}

#[derive(Debug)]
struct DeltaOffsetCompressor {
    child: Box<dyn BlockCompressor>,
}

impl DeltaOffsetCompressor {
    fn new(child: Box<dyn BlockCompressor>) -> Self {
        Self { child }
    }
}

impl BlockCompressor for DeltaOffsetCompressor {
    fn compress(&self, data: DataBlock) -> Result<Option<LanceBuffer>> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Delta offset compression requires fixed-width data",
            ));
        };
        let value_type = BlockValueType::from_bits(data.bits_per_value)?;
        let base = offset_member_base(&data, value_type)?;
        let deltas = encode_deltas(data, base)?;
        self.child.compress(DataBlock::FixedWidth(deltas))
    }
}

#[derive(Debug, Clone)]
pub(super) struct OffsetFamilyMembers {
    data: FixedWidthDataBlock,
    ranges: Arc<[std::ops::Range<usize>]>,
}

impl OffsetFamilyMembers {
    fn into_blocks(self) -> Result<Vec<FixedWidthDataBlock>> {
        let bytes_per_value = usize::try_from(self.data.bits_per_value / 8)
            .map_err(|_| Error::invalid_input("Block family value width does not fit usize"))?;
        self.ranges
            .iter()
            .map(|range| {
                let byte_start = range
                    .start
                    .checked_mul(bytes_per_value)
                    .ok_or_else(|| Error::invalid_input("Block family start overflows usize"))?;
                let byte_len = range
                    .len()
                    .checked_mul(bytes_per_value)
                    .ok_or_else(|| Error::invalid_input("Block family length overflows usize"))?;
                Ok(FixedWidthDataBlock {
                    data: self.data.data.slice_with_length(byte_start, byte_len),
                    bits_per_value: self.data.bits_per_value,
                    num_values: range.len() as u64,
                    block_info: BlockInfo::default(),
                })
            })
            .collect()
    }
}

/// One reusable concrete compressor for all independently framed chunks.
#[derive(Debug)]
pub(super) struct OffsetFamilyCompressor {
    compressor: Box<dyn BlockCompressor>,
    encoding: CompressiveEncoding,
    has_payload: bool,
    estimated_payload_bytes: Arc<[u64]>,
    members: OffsetFamilyMembers,
}

impl OffsetFamilyCompressor {
    pub(super) fn encoding(&self) -> &CompressiveEncoding {
        &self.encoding
    }

    pub(super) fn has_payload(&self) -> bool {
        self.has_payload
    }

    pub(super) fn estimated_payload_bytes(&self) -> &[u64] {
        &self.estimated_payload_bytes
    }

    pub(super) fn compress_members(self) -> Result<Vec<Option<LanceBuffer>>> {
        let has_payload = self.has_payload;
        if !has_payload {
            return Ok(vec![None; self.members.ranges.len()]);
        }
        self.members
            .into_blocks()?
            .into_iter()
            .zip(self.estimated_payload_bytes.iter().copied())
            .enumerate()
            .map(|(index, (data, estimated_payload_bytes))| {
                let payload = self.compressor.compress(DataBlock::FixedWidth(data))?;
                if payload.is_some() != has_payload {
                    return Err(Error::internal(format!(
                        "Block family compressor payload presence changed for member {index}"
                    )));
                }
                let payload_bytes = payload.as_ref().map_or(0, LanceBuffer::len) as u64;
                if payload_bytes != estimated_payload_bytes {
                    return Err(Error::internal(format!(
                        "Block family member {index} estimated {estimated_payload_bytes} bytes but built {payload_bytes} bytes",
                    )));
                }
                Ok(payload)
            })
            .collect()
    }
}

#[derive(Debug)]
pub(super) struct OffsetFamilyCandidate {
    selection: SelectedBlockCompressor,
    estimated_payload_bytes: Vec<u64>,
    estimated_wire_bytes: u64,
    transform_depth: u8,
    decode_cpu_rank: u8,
    stable_rank: u8,
}

impl OffsetFamilyCandidate {
    fn is_better_than(&self, other: &Self) -> bool {
        (
            self.estimated_wire_bytes,
            self.transform_depth,
            self.decode_cpu_rank,
            self.stable_rank,
        ) < (
            other.estimated_wire_bytes,
            other.transform_depth,
            other.decode_cpu_rank,
            other.stable_rank,
        )
    }

    fn finish(self, members: OffsetFamilyMembers) -> OffsetFamilyCompressor {
        OffsetFamilyCompressor {
            compressor: self.selection.compressor,
            encoding: self.selection.encoding,
            has_payload: self.selection.has_payload,
            estimated_payload_bytes: Arc::from(self.estimated_payload_bytes),
            members,
        }
    }

    fn normalize_members(mut self) -> Self {
        self.selection.compressor =
            Box::new(NormalizedOffsetCompressor::new(self.selection.compressor));
        self
    }
}

/// Builds a Delta(Flat) family after a bounded preflight has proven that a
/// complete selector pass cannot change the winner.
pub(super) fn select_delta_flat_offsets(
    data: FixedWidthDataBlock,
    member_ranges: Vec<std::ops::Range<usize>>,
) -> Result<OffsetFamilyCompressor> {
    if member_ranges.is_empty() {
        return Err(Error::invalid_input(
            "Block sequence family requires at least one block",
        ));
    }
    let value_type = BlockValueType::from_bits(data.bits_per_value)?;
    validate_fixed_payload_len(
        &data.data,
        value_type,
        data.num_values,
        "Block family input",
    )?;
    let values_len = usize::try_from(data.num_values)
        .map_err(|_| Error::invalid_input("Block family cardinality does not fit usize"))?;
    let estimated_payload_bytes = member_ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            if range.end > values_len || range.len() < 2 {
                return Err(Error::invalid_input(format!(
                    "Delta block family member {index} range {}..{} is invalid for {values_len} values",
                    range.start, range.end
                )));
            }
            (range.len() as u64 - 1)
                .checked_mul(value_type.bytes_per_value() as u64)
                .ok_or_else(|| Error::invalid_input("Delta family payload size overflows u64"))
        })
        .collect::<Result<Vec<_>>>()?;
    validate_monotonic_offset_members(&data, &member_ranges, value_type)?;

    let child = flat_selection(value_type);
    let child_has_payload = child.has_payload;
    let encoding = ProtobufUtils21::delta(value_type.bits_per_value(), 0, child.encoding);
    let selection = SelectedBlockCompressor::new(
        Box::new(DeltaOffsetCompressor::new(child.compressor)),
        encoding,
        child_has_payload,
    );
    Ok(OffsetFamilyCompressor {
        compressor: selection.compressor,
        encoding: selection.encoding,
        has_payload: selection.has_payload,
        estimated_payload_bytes: Arc::from(estimated_payload_bytes),
        members: OffsetFamilyMembers {
            data,
            ranges: Arc::from(member_ranges),
        },
    })
}

pub(super) fn validate_monotonic_offset_members(
    data: &FixedWidthDataBlock,
    ranges: &[std::ops::Range<usize>],
    value_type: BlockValueType,
) -> Result<()> {
    macro_rules! validate {
        ($ty:ty) => {{
            let values = data.data.borrow_to_typed_view::<$ty>();
            for (index, range) in ranges.iter().enumerate() {
                let member = values.get(range.clone()).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Block family member {index} range {}..{} exceeds {} values",
                        range.start,
                        range.end,
                        values.len()
                    ))
                })?;
                if member.windows(2).any(|pair| pair[1] < pair[0]) {
                    return Err(Error::invalid_input(format!(
                        "Block family member {index} must be non-decreasing"
                    )));
                }
            }
        }};
    }
    match value_type {
        BlockValueType::UInt32 => validate!(u32),
        BlockValueType::UInt64 => validate!(u64),
        _ => {
            return Err(Error::invalid_input(format!(
                "Generic block family selection only supports u32 or u64, got {} bits",
                value_type.bits_per_value()
            )));
        }
    }
    Ok(())
}

/// Selects one reusable concrete compressor for independent offset chunks.
pub(super) fn select_offset_family(
    data: FixedWidthDataBlock,
    member_ranges: Vec<std::ops::Range<usize>>,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<OffsetFamilyCompressor> {
    if member_ranges.is_empty() {
        return Err(Error::invalid_input(
            "Block sequence family requires at least one block",
        ));
    }
    let value_type = BlockValueType::from_bits(data.bits_per_value)?;
    if let Some(pattern) = probe_offset_metadata(&data, &member_ranges)? {
        let candidate = match pattern {
            OffsetMetadataPattern::Constant(value) => offset_metadata_candidate(
                constant_selection(value_type, Some(value)),
                member_ranges.len(),
                cost,
                0,
            ),
            OffsetMetadataPattern::Range { start, step } => offset_metadata_candidate(
                range_selection(value_type, start, step),
                member_ranges.len(),
                cost,
                1,
            ),
        }
        .normalize_members();
        return Ok(candidate.finish(OffsetFamilyMembers {
            data,
            ranges: Arc::from(member_ranges),
        }));
    }
    let collect_sample = !matches!(field_params.compression.as_deref(), Some("none" | "fsst"));
    let analysis = analyze_offset_members(&data, &member_ranges, collect_sample)?;
    let members = OffsetFamilyMembers {
        data,
        ranges: Arc::from(member_ranges),
    };
    let value_stats = analysis
        .members
        .iter()
        .map(|analysis| &analysis.values)
        .collect::<Vec<_>>();

    let mut candidates = offset_direct_candidates(&value_stats, field_params, cost, 0, true)?
        .into_iter()
        .map(OffsetFamilyCandidate::normalize_members)
        .collect::<Vec<_>>();
    if let Some(candidate) = offset_rle_candidate(&analysis, field_params, cost)? {
        candidates.push(candidate.normalize_members());
    }
    if let Some(candidate) = offset_dictionary_candidate(&analysis, field_params, cost)? {
        candidates.push(candidate.normalize_members());
    }
    if value_stats.iter().all(|stats| stats.len >= 2)
        && let Some(base) = common_first(&value_stats)
        && analysis
            .members
            .iter()
            .all(|analysis| analysis.deltas.is_some())
    {
        let delta_stats = analysis
            .members
            .iter()
            .map(|analysis| {
                analysis
                    .deltas
                    .as_ref()
                    .expect("delta presence was checked")
            })
            .collect::<Vec<_>>();
        let child = offset_leaf_candidate(&delta_stats, field_params, cost)?;
        let estimated_payload_bytes = child.estimated_payload_bytes.clone();
        let child_has_payload = child.selection.has_payload;
        let encoding =
            ProtobufUtils21::delta(value_type.bits_per_value(), base, child.selection.encoding);
        let selection = SelectedBlockCompressor::new(
            Box::new(DeltaOffsetCompressor::new(child.selection.compressor)),
            encoding,
            child_has_payload,
        );
        candidates.push(OffsetFamilyCandidate {
            estimated_wire_bytes: offset_wire_bytes(
                &selection.encoding,
                selection.has_payload,
                &estimated_payload_bytes,
                cost,
            ),
            selection,
            estimated_payload_bytes,
            transform_depth: 1,
            decode_cpu_rank: 2,
            stable_rank: 5,
        });
    }

    candidates
        .into_iter()
        .reduce(|best, candidate| {
            if candidate.is_better_than(&best) {
                candidate
            } else {
                best
            }
        })
        .map(|candidate| candidate.finish(members))
        .ok_or_else(|| {
            Error::internal("No applicable block family compression candidate".to_string())
        })
}

fn offset_leaf_candidate(
    stats: &[&SequenceStats],
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<OffsetFamilyCandidate> {
    let value_type = stats[0].value_type;
    if stats.iter().all(|stats| stats.len == 0) {
        return Ok(offset_metadata_candidate(
            constant_selection(value_type, None),
            stats.len(),
            cost,
            0,
        ));
    }
    if let Some(value) = common_constant(stats) {
        return Ok(offset_metadata_candidate(
            constant_selection(value_type, Some(value)),
            stats.len(),
            cost,
            0,
        ));
    }
    if let Some((start, step)) = common_range(stats) {
        return Ok(offset_metadata_candidate(
            range_selection(value_type, start, step),
            stats.len(),
            cost,
            1,
        ));
    }
    offset_direct_candidates(stats, field_params, cost, 0, false)?
        .into_iter()
        .reduce(|best, candidate| {
            if candidate.is_better_than(&best) {
                candidate
            } else {
                best
            }
        })
        .ok_or_else(|| Error::internal("No applicable block family leaf candidate".to_string()))
}

fn offset_rle_candidate(
    analysis: &OffsetFamilyAnalysis,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<Option<OffsetFamilyCandidate>> {
    let Some(runs) = analysis.runs.as_ref() else {
        return Ok(None);
    };

    let run_value_stats = runs.iter().map(|member| &member.values).collect::<Vec<_>>();
    let raw_run_length_stats = runs
        .iter()
        .map(|member| &member.lengths)
        .collect::<Vec<_>>();
    let max_run_length = raw_run_length_stats
        .iter()
        .map(|stats| stats.max)
        .max()
        .unwrap_or(0);
    let run_lengths_are_range = common_range(&raw_run_length_stats).is_some();
    let run_length_type = if run_lengths_are_range && max_run_length <= u32::MAX as u64 {
        BlockValueType::UInt32
    } else if max_run_length <= u8::MAX as u64 {
        BlockValueType::UInt8
    } else if max_run_length <= u16::MAX as u64 {
        BlockValueType::UInt16
    } else if max_run_length <= u32::MAX as u64 {
        BlockValueType::UInt32
    } else {
        return Ok(None);
    };
    let run_length_stats = raw_run_length_stats
        .iter()
        .map(|stats| SequenceStats {
            value_type: run_length_type,
            len: stats.len,
            first: stats.first,
            min: stats.min,
            max: stats.max,
            is_non_decreasing: stats.is_non_decreasing,
            arithmetic_step: stats.arithmetic_step,
            run_count: stats.run_count,
            sample: None,
        })
        .collect::<Vec<_>>();
    let run_length_refs = run_length_stats.iter().collect::<Vec<_>>();

    let values = offset_leaf_candidate(&run_value_stats, field_params, BlockCost::default())?;
    let mut lengths = offset_leaf_candidate(&run_length_refs, field_params, BlockCost::default())?;
    let lengths_are_metadata = matches!(
        lengths.selection.encoding.compression.as_ref(),
        Some(crate::format::pb21::compressive_encoding::Compression::Constant(_))
            | Some(crate::format::pb21::compressive_encoding::Compression::Range(_))
    );
    let values_are_flat = matches!(
        values.selection.encoding.compression.as_ref(),
        Some(crate::format::pb21::compressive_encoding::Compression::Flat(_))
    );
    let lengths_are_flat = matches!(
        lengths.selection.encoding.compression.as_ref(),
        Some(crate::format::pb21::compressive_encoding::Compression::Flat(_))
    );
    if !lengths_are_metadata && !values_are_flat && !lengths_are_flat {
        lengths = offset_flat_candidate(&run_length_refs, BlockCost::default())?;
    }
    let has_payload = values.selection.has_payload || lengths.selection.has_payload;
    let estimated_payload_bytes = values
        .estimated_payload_bytes
        .iter()
        .zip(&lengths.estimated_payload_bytes)
        .map(|(values, lengths)| {
            if has_payload {
                rle::BLOCK_FRAME_BYTES
                    .saturating_add(*values)
                    .saturating_add(*lengths)
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let encoding = ProtobufUtils21::rle(
        values.selection.encoding.clone(),
        lengths.selection.encoding.clone(),
    );
    let selection = SelectedBlockCompressor::new(
        Box::new(BlockRleCompressor::new(
            analysis.members[0].values.value_type,
            run_length_type,
            values.selection.compressor,
            lengths.selection.compressor,
        )),
        encoding,
        has_payload,
    );
    Ok(Some(OffsetFamilyCandidate {
        estimated_wire_bytes: offset_wire_bytes(
            &selection.encoding,
            has_payload,
            &estimated_payload_bytes,
            cost,
        ),
        selection,
        estimated_payload_bytes,
        transform_depth: 1,
        decode_cpu_rank: 3,
        stable_rank: 3,
    }))
}

fn offset_dictionary_candidate(
    analysis: &OffsetFamilyAnalysis,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<Option<OffsetFamilyCandidate>> {
    if analysis.members[0].values.value_type != BlockValueType::UInt64 {
        return Ok(None);
    }
    let Some(dictionary_items) = analysis.dictionary_items.as_ref() else {
        return Ok(None);
    };
    let total_values = analysis.members.iter().try_fold(0_u64, |total, member| {
        total
            .checked_add(member.values.len)
            .ok_or_else(|| Error::invalid_input("Offset family cardinality overflows u64"))
    })?;
    if dictionary_items.len() < 2
        || dictionary_items.len() > MAX_DICTIONARY_ITEMS
        || dictionary_items.len() as u64 >= total_values.div_ceil(2)
    {
        return Ok(None);
    }

    let mut items_builder = SequenceStatsBuilder::without_sample(BlockValueType::UInt64);
    for value in dictionary_items.iter().copied() {
        items_builder.push(value);
    }
    let items_stats = items_builder.finish();
    let item_stats = (0..analysis.members.len())
        .map(|_| &items_stats)
        .collect::<Vec<_>>();
    let index_stats = analysis
        .members
        .iter()
        .map(|member| SequenceStats {
            value_type: BlockValueType::UInt32,
            len: member.values.len,
            first: Some(0),
            min: 0,
            max: dictionary_items.len().saturating_sub(1) as u64,
            is_non_decreasing: false,
            arithmetic_step: None,
            run_count: member.values.len,
            sample: None,
        })
        .collect::<Vec<_>>();
    let index_refs = index_stats.iter().collect::<Vec<_>>();
    let indices = offset_leaf_candidate(&index_refs, field_params, BlockCost::default())?;
    let items = offset_leaf_candidate(&item_stats, field_params, BlockCost::default())?;
    let has_payload = indices.selection.has_payload || items.selection.has_payload;
    let estimated_payload_bytes = indices
        .estimated_payload_bytes
        .iter()
        .zip(&items.estimated_payload_bytes)
        .map(|(indices, items)| {
            if has_payload {
                dictionary::BLOCK_FRAME_BYTES
                    .saturating_add(*indices)
                    .saturating_add(*items)
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let num_dictionary_items = u32::try_from(dictionary_items.len())
        .map_err(|_| Error::invalid_input("Dictionary item count exceeds u32::MAX"))?;
    let encoding = ProtobufUtils21::dictionary(
        indices.selection.encoding.clone(),
        items.selection.encoding.clone(),
        num_dictionary_items,
    );
    let selection = SelectedBlockCompressor::new(
        Box::new(DictionaryBlockCompressor::new(
            BlockValueType::UInt64,
            dictionary_items.clone(),
            indices.selection.compressor,
            items.selection.compressor,
        )),
        encoding,
        has_payload,
    );
    Ok(Some(OffsetFamilyCandidate {
        estimated_wire_bytes: offset_wire_bytes(
            &selection.encoding,
            has_payload,
            &estimated_payload_bytes,
            cost,
        ),
        selection,
        estimated_payload_bytes,
        transform_depth: 1,
        decode_cpu_rank: 3,
        stable_rank: 4,
    }))
}

fn offset_direct_candidates(
    stats: &[&SequenceStats],
    field_params: &CompressionFieldParams,
    cost: BlockCost,
    stable_rank_base: u8,
    allow_general: bool,
) -> Result<Vec<OffsetFamilyCandidate>> {
    let combined = combine_sequence_stats(stats)?;
    direct_candidates(&combined, field_params, stable_rank_base, allow_general)?
        .into_iter()
        .map(|candidate: Candidate| {
            let estimator = candidate.estimator.ok_or_else(|| {
                Error::internal(
                    "Direct block candidate is missing its payload estimator".to_string(),
                )
            })?;
            let estimated_payload_bytes = stats
                .iter()
                .map(|stats| estimate_payload(estimator, stats))
                .collect::<Result<Vec<_>>>()?;
            let estimated_wire_bytes = offset_wire_bytes(
                &candidate.selection.encoding,
                candidate.selection.has_payload,
                &estimated_payload_bytes,
                cost,
            );
            Ok(OffsetFamilyCandidate {
                selection: candidate.selection,
                estimated_payload_bytes,
                estimated_wire_bytes,
                transform_depth: candidate.transform_depth,
                decode_cpu_rank: candidate.decode_cpu_rank,
                stable_rank: candidate.stable_rank,
            })
        })
        .collect()
}

fn offset_flat_candidate(
    stats: &[&SequenceStats],
    cost: BlockCost,
) -> Result<OffsetFamilyCandidate> {
    let value_type = stats[0].value_type;
    if stats.iter().any(|stats| stats.value_type != value_type) {
        return Err(Error::invalid_input(
            "Cannot build a Flat family candidate with different value widths",
        ));
    }
    let estimated_payload_bytes = stats
        .iter()
        .map(|stats| stats.raw_bytes())
        .collect::<Vec<_>>();
    let selection = flat_selection(value_type);
    Ok(OffsetFamilyCandidate {
        estimated_wire_bytes: offset_wire_bytes(
            &selection.encoding,
            true,
            &estimated_payload_bytes,
            cost,
        ),
        selection,
        estimated_payload_bytes,
        transform_depth: 0,
        decode_cpu_rank: 0,
        stable_rank: 0,
    })
}

fn offset_metadata_candidate(
    selection: SelectedBlockCompressor,
    num_blocks: usize,
    cost: BlockCost,
    stable_rank: u8,
) -> OffsetFamilyCandidate {
    let estimated_payload_bytes = vec![0; num_blocks];
    OffsetFamilyCandidate {
        estimated_wire_bytes: offset_wire_bytes(
            &selection.encoding,
            false,
            &estimated_payload_bytes,
            cost,
        ),
        selection,
        estimated_payload_bytes,
        transform_depth: 0,
        decode_cpu_rank: 0,
        stable_rank,
    }
}

fn offset_wire_bytes(
    encoding: &CompressiveEncoding,
    has_payload: bool,
    payload_bytes: &[u64],
    cost: BlockCost,
) -> u64 {
    let payload_cost = payload_bytes.iter().fold(0_u64, |total, payload_bytes| {
        total.saturating_add(cost.payload_wire_bytes(has_payload, *payload_bytes))
    });
    (encoding.encoded_len() as u64).saturating_add(payload_cost)
}

pub(super) fn common_first(stats: &[&SequenceStats]) -> Option<u64> {
    let first = stats.first()?.first?;
    stats
        .iter()
        .all(|stats| stats.first == Some(first))
        .then_some(first)
}

pub(super) fn common_constant(stats: &[&SequenceStats]) -> Option<u64> {
    let value = stats.first()?.first?;
    stats
        .iter()
        .all(|stats| stats.len > 0 && stats.min == value && stats.max == value)
        .then_some(value)
}

pub(super) fn common_range(stats: &[&SequenceStats]) -> Option<(u64, u64)> {
    let first = stats.first()?;
    let start = first.first?;
    let step = first.arithmetic_step?;
    if step == 0 {
        return None;
    }
    stats
        .iter()
        .all(|stats| {
            stats.len >= 2 && stats.first == Some(start) && stats.arithmetic_step == Some(step)
        })
        .then_some((start, step))
}

pub(super) fn combine_sequence_stats(stats: &[&SequenceStats]) -> Result<SequenceStats> {
    let value_type = stats[0].value_type;
    let mut len = 0_u64;
    let mut run_count = 0_u64;
    let mut min = u64::MAX;
    let mut max = 0_u64;
    let mut sample = Vec::new();
    for stats in stats {
        if stats.value_type != value_type {
            return Err(Error::invalid_input(
                "Cannot combine sequence stats with different value widths",
            ));
        }
        len = len
            .checked_add(stats.len)
            .ok_or_else(|| Error::invalid_input("Combined sequence length overflows u64"))?;
        run_count = run_count
            .checked_add(stats.run_count)
            .ok_or_else(|| Error::invalid_input("Combined run count overflows u64"))?;
        if stats.len > 0 {
            min = min.min(stats.min);
            max = max.max(stats.max);
        }
        let remaining = GENERAL_SAMPLE_BYTES.saturating_sub(sample.len());
        let stats_sample = stats.sample_bytes();
        sample.extend_from_slice(&stats_sample[..stats_sample.len().min(remaining)]);
    }
    Ok(SequenceStats {
        value_type,
        len,
        first: stats.first().and_then(|stats| stats.first),
        min: if len == 0 { 0 } else { min },
        max,
        is_non_decreasing: false,
        arithmetic_step: None,
        run_count,
        sample: (!sample.is_empty()).then(|| Arc::from(sample)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_u32(values: &[u32]) -> FixedWidthDataBlock {
        FixedWidthDataBlock {
            data: crate::buffer::LanceBuffer::reinterpret_vec(values.to_vec()),
            bits_per_value: 32,
            num_values: values.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    fn fixed_u64(values: &[u64]) -> FixedWidthDataBlock {
        FixedWidthDataBlock {
            data: crate::buffer::LanceBuffer::reinterpret_vec(values.to_vec()),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    fn root_codec(family: &OffsetFamilyCompressor) -> &'static str {
        use crate::format::pb21::compressive_encoding::Compression;

        match family.encoding().compression.as_ref().unwrap() {
            Compression::Constant(_) => "constant",
            Compression::Range(_) => "range",
            Compression::Delta(_) => "delta",
            Compression::Flat(_) => "flat",
            Compression::Rle(_) => "rle",
            Compression::Dictionary(_) => "dictionary",
            other => panic!("unexpected root codec: {other:?}"),
        }
    }

    #[test]
    fn selects_shared_metadata_patterns_for_independent_members() {
        let cases = [
            (vec![5_u32, 5, 5, 10, 10, 10], vec![0..3, 3..6], "constant"),
            (
                vec![5_u32, 7, 9, 11, 10, 12, 14, 16],
                vec![0..4, 4..8],
                "range",
            ),
        ];
        for (values, ranges, expected) in cases {
            let family = select_offset_family(
                fixed_u32(&values),
                ranges,
                &CompressionFieldParams::default(),
                BlockCost::new(0, 1),
            )
            .unwrap();
            assert_eq!(root_codec(&family), expected);
            assert!(!family.has_payload());
            assert!(
                family
                    .compress_members()
                    .unwrap()
                    .iter()
                    .all(Option::is_none)
            );
        }
    }

    #[test]
    fn compares_delta_against_flat_by_complete_wire_cost() {
        let params = CompressionFieldParams {
            compression: Some("none".to_string()),
            ..Default::default()
        };
        let delta = select_offset_family(
            fixed_u32(&[0, 2, 5, 9, 14]),
            vec![0..5],
            &params,
            BlockCost::new(0, 1),
        )
        .unwrap();
        assert_eq!(root_codec(&delta), "delta");
        assert!(!delta.has_payload());

        let flat = select_offset_family(
            fixed_u32(&[0, 1_u32 << 31, (1_u32 << 31) + 1, u32::MAX]),
            vec![0..4],
            &params,
            BlockCost::new(0, 1),
        )
        .unwrap();
        assert_eq!(root_codec(&flat), "flat");
        assert!(flat.has_payload());
        let payloads = flat.compress_members().unwrap();
        assert_eq!(payloads[0].as_ref().unwrap().len(), 16);
    }

    #[test]
    fn delta_range_handles_a_short_final_chunk() {
        let lengths = (0..4_112_u32)
            .map(|index| 4 + index % 64)
            .collect::<Vec<_>>();
        let mut offsets = Vec::with_capacity(lengths.len() + 1);
        offsets.push(0_u32);
        for length in lengths {
            offsets.push(offsets.last().copied().unwrap() + length);
        }
        let ranges = (0..offsets.len() - 1)
            .step_by(64)
            .map(|start| start..(start + 65).min(offsets.len()))
            .collect::<Vec<_>>();
        let analysis = analyze_offset_members(&fixed_u32(&offsets), &ranges, false).unwrap();
        let delta_stats = analysis
            .members
            .iter()
            .map(|analysis| analysis.deltas.as_ref().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(common_range(&delta_stats), Some((4, 1)));
    }

    #[test]
    fn selects_rle_without_materializing_losing_dictionary() {
        use crate::encodings::physical::dictionary::BLOCK_MATERIALIZATION_COUNT;
        use crate::format::pb21::compressive_encoding::Compression;

        let member = (0..32_u64)
            .flat_map(|value| std::iter::repeat_n(value, 8))
            .collect::<Vec<_>>();
        let mut values = member.clone();
        values.extend_from_slice(&member);
        let ranges = vec![0..member.len(), member.len()..values.len()];

        BLOCK_MATERIALIZATION_COUNT.with(|count| count.set(0));
        let family = select_offset_family(
            fixed_u64(&values),
            ranges,
            &CompressionFieldParams::default(),
            BlockCost::new(0, 1),
        )
        .unwrap();
        assert!(matches!(
            family.encoding().compression.as_ref(),
            Some(Compression::Rle(_))
        ));
        assert!(!family.has_payload());
        assert_eq!(family.estimated_payload_bytes().len(), 2);
        BLOCK_MATERIALIZATION_COUNT.with(|count| assert_eq!(count.get(), 0));

        let payloads = family.compress_members().unwrap();
        assert!(payloads.iter().all(Option::is_none));
        BLOCK_MATERIALIZATION_COUNT.with(|count| assert_eq!(count.get(), 0));
    }

    #[test]
    fn delta_flat_preflight_builds_members_and_rejects_decreasing_offsets() {
        let family =
            select_delta_flat_offsets(fixed_u32(&[0, 2, 5, 0, 1, 4]), vec![0..3, 3..6]).unwrap();
        assert!(family.has_payload());
        assert_eq!(family.estimated_payload_bytes(), &[8, 8]);
        let payloads = family.compress_members().unwrap();
        assert_eq!(payloads.len(), 2);
        assert!(payloads.iter().all(Option::is_some));

        let error = select_delta_flat_offsets(fixed_u32(&[0, 2, 1]), vec![0..3]).unwrap_err();
        assert!(error.to_string().contains("non-decreasing"));
    }
}
