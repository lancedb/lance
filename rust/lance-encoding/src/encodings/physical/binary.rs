// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Basic encodings for variable width data
//!
//! These are not compression but represent the "leaf" encodings for variable length data
//! where we simply match the data with the rules of the structural encoding.
//!
//! These encodings are transparent since we aren't actually doing any compression.  No information
//! is needed in the encoding description.

use arrow_array::OffsetSizeTrait;
use byteorder::{ByteOrder, LittleEndian};
use prost::Message;

use crate::compression::{
    BlockCompressor, BlockDecompressor, BlockValueType, MiniBlockDecompressor,
    VariablePerValueDecompressor, require_block_payload,
};

use crate::buffer::LanceBuffer;
#[cfg(feature = "bitpacking")]
use crate::compression::block::BITPACK_CHUNK_VALUES;
use crate::compression::block::{create_block_decompressor, infer_block_value_type};
use crate::compression_config::CompressionFieldParams;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock};
use crate::encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MAX_MINIBLOCK_VALUES, MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext,
    MiniBlockCompressor,
};
use crate::format::pb21::CompressiveEncoding;
use crate::format::pb21::compressive_encoding::Compression;
use crate::format::{ProtobufUtils21, pb21};

use lance_core::{Error, Result};

mod offsets;

use offsets::{BlockCost, OffsetFamilyCompressor, select_delta_flat_offsets, select_offset_family};

#[derive(Debug)]
pub struct BinaryMiniBlockEncoder {
    minichunk_size: i64,
    generic_offsets: Option<CompressionFieldParams>,
}

impl Default for BinaryMiniBlockEncoder {
    fn default() -> Self {
        Self {
            minichunk_size: *AIM_MINICHUNK_SIZE,
            generic_offsets: None,
        }
    }
}

const DEFAULT_AIM_MINICHUNK_SIZE: i64 = 4 * 1024;

pub static AIM_MINICHUNK_SIZE: std::sync::LazyLock<i64> = std::sync::LazyLock::new(|| {
    std::env::var("LANCE_BINARY_MINIBLOCK_CHUNK_SIZE")
        .unwrap_or_else(|_| DEFAULT_AIM_MINICHUNK_SIZE.to_string())
        .parse::<i64>()
        .unwrap_or(DEFAULT_AIM_MINICHUNK_SIZE)
});

#[derive(Debug, Clone, Copy)]
struct BinaryChunkRange {
    start_offset_index: usize,
    end_offset_index: usize,
}

fn binary_chunk_ranges<N: OffsetSizeTrait>(
    offsets: &[N],
    minichunk_size: i64,
) -> Result<Vec<BinaryChunkRange>> {
    if offsets.is_empty() {
        return Err(Error::invalid_input(
            "Variable-width mini-block offsets cannot be empty",
        ));
    }
    let mut ranges = Vec::new();
    let mut last_offset_in_orig_idx = 0;
    loop {
        let this_last_offset_in_orig_idx =
            search_next_offset_idx(offsets, last_offset_in_orig_idx, minichunk_size)?;
        ranges.push(BinaryChunkRange {
            start_offset_index: last_offset_in_orig_idx,
            end_offset_index: this_last_offset_in_orig_idx,
        });
        if this_last_offset_in_orig_idx == offsets.len() - 1 {
            break;
        }
        last_offset_in_orig_idx = this_last_offset_in_orig_idx;
    }
    Ok(ranges)
}

// Make it to support both i32 and i64 Arrow offsets.
fn chunk_offsets<N: OffsetSizeTrait>(
    offsets: &[N],
    data: &[u8],
    alignment: usize,
    minichunk_size: i64,
) -> Result<(Vec<LanceBuffer>, Vec<MiniBlockChunk>)> {
    let ranges = binary_chunk_ranges(offsets, minichunk_size)?;
    chunk_offsets_with_ranges(offsets, data, alignment, &ranges)
}

fn chunk_offsets_with_ranges<N: OffsetSizeTrait>(
    offsets: &[N],
    data: &[u8],
    alignment: usize,
    ranges: &[BinaryChunkRange],
) -> Result<(Vec<LanceBuffer>, Vec<MiniBlockChunk>)> {
    let byte_width: usize = N::get_byte_width();
    let mut chunk_sizes = Vec::with_capacity(ranges.len());
    let mut chunks = Vec::with_capacity(ranges.len());

    for range in ranges {
        let num_values = range.end_offset_index - range.start_offset_index;
        let chunk_bytes = offsets[range.end_offset_index] - offsets[range.start_offset_index];
        let chunk_bytes = chunk_bytes.to_usize().ok_or_else(|| {
            Error::invalid_input("Variable-width mini-block byte length does not fit usize")
        })?;
        let chunk_size = (num_values + 1)
            .checked_mul(byte_width)
            .and_then(|offset_bytes| offset_bytes.checked_add(chunk_bytes))
            .ok_or_else(|| {
                Error::invalid_input("Variable-width mini-block chunk size overflows usize")
            })?;
        let padded_chunk_size = chunk_size.next_multiple_of(alignment);
        let padded_chunk_size_u32 = u32::try_from(padded_chunk_size).map_err(|_| {
            Error::invalid_input(format!(
                "Variable-width mini-block chunk has {padded_chunk_size} bytes, exceeding u32::MAX"
            ))
        })?;
        chunk_sizes.push(padded_chunk_size);
        chunks.push(MiniBlockChunk {
            log_num_values: if range.end_offset_index == offsets.len() - 1 {
                0
            } else {
                num_values.trailing_zeros() as u8
            },
            buffer_sizes: vec![padded_chunk_size_u32],
        });
    }

    let output_total_bytes = chunk_sizes.iter().copied().sum::<usize>();
    let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

    for (range, padded_chunk_size) in ranges.iter().zip(chunk_sizes) {
        let chunk_output_start = output.len();
        let bytes_start_offset =
            (range.end_offset_index - range.start_offset_index + 1) * byte_width;
        let bytes_start_offset = N::from_usize(bytes_start_offset).ok_or_else(|| {
            Error::invalid_input("Variable-width mini-block offset header does not fit offset type")
        })?;
        let this_chunk_offsets: Vec<N> = offsets[range.start_offset_index..=range.end_offset_index]
            .iter()
            .map(|offset| *offset - offsets[range.start_offset_index] + bytes_start_offset)
            .collect();

        let this_chunk_offsets = LanceBuffer::reinterpret_vec(this_chunk_offsets);
        output.extend_from_slice(&this_chunk_offsets);

        let start_in_orig = offsets[range.start_offset_index]
            .to_usize()
            .ok_or_else(|| {
                Error::invalid_input("Variable-width mini-block start offset does not fit usize")
            })?;
        let end_in_orig = offsets[range.end_offset_index].to_usize().ok_or_else(|| {
            Error::invalid_input("Variable-width mini-block end offset does not fit usize")
        })?;
        if start_in_orig > end_in_orig || end_in_orig > data.len() {
            return Err(Error::invalid_input(format!(
                "Variable-width mini-block byte range {start_in_orig}..{end_in_orig} is invalid for {} bytes",
                data.len()
            )));
        }
        output.extend_from_slice(&data[start_in_orig..end_in_orig]);

        const PAD_BYTE: u8 = 72;
        let encoded_chunk_size = output.len() - chunk_output_start;
        let pad_len = padded_chunk_size - encoded_chunk_size;
        if pad_len > 0 {
            output.extend(std::iter::repeat_n(PAD_BYTE, pad_len));
        }
    }
    Ok((vec![LanceBuffer::reinterpret_vec(output)], chunks))
}

// search for the next offset index to cut the values into a chunk.
// this function incrementally peek the number of values in a chunk,
// each time multiplies the number of values by 2.
// It returns the offset_idx in `offsets` that belongs to this chunk.
fn search_next_offset_idx<N: OffsetSizeTrait>(
    offsets: &[N],
    last_offset_idx: usize,
    minichunk_size: i64,
) -> Result<usize> {
    // MiniBlockChunk uses `log_num_values == 0` as a sentinel for the final chunk. This means we
    // must avoid creating 1-value chunks except for the final chunk, even if the configured
    // `minichunk_size` is too small to fit more than one value.
    let remaining_values = offsets.len().saturating_sub(last_offset_idx + 1);
    if remaining_values <= 1 {
        return Ok(offsets.len() - 1);
    }

    let mut num_values = 2;
    let mut new_num_values = num_values * 2;
    loop {
        if last_offset_idx + new_num_values >= offsets.len() {
            let new_size =
                checked_variable_chunk_size(offsets, last_offset_idx, offsets.len() - 1)?;
            if new_size <= i128::from(minichunk_size) {
                // case 1: can fit the rest of all data into a miniblock
                return Ok(offsets.len() - 1);
            } else {
                // case 2: can only fit the last tried `num_values` into a miniblock
                return Ok(last_offset_idx + num_values);
            }
        }
        let new_size = checked_variable_chunk_size(
            offsets,
            last_offset_idx,
            last_offset_idx + new_num_values,
        )?;
        if new_size <= i128::from(minichunk_size) {
            if new_num_values * 2 > *MAX_MINIBLOCK_VALUES as usize {
                // hit the max number of values limit
                break;
            }
            num_values = new_num_values;
            new_num_values *= 2;
        } else {
            break;
        }
    }
    Ok(last_offset_idx + num_values)
}

fn checked_variable_chunk_size<N: OffsetSizeTrait>(
    offsets: &[N],
    start: usize,
    end: usize,
) -> Result<i128> {
    let start_offset = offsets[start]
        .to_i64()
        .ok_or_else(|| Error::invalid_input("Variable-width chunk start does not fit i64"))?;
    let end_offset = offsets[end]
        .to_i64()
        .ok_or_else(|| Error::invalid_input("Variable-width chunk end does not fit i64"))?;
    let value_bytes = i128::from(end_offset)
        .checked_sub(i128::from(start_offset))
        .filter(|value_bytes| *value_bytes >= 0)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Variable-width offsets decrease between indices {start} and {end}"
            ))
        })?;
    let offset_bytes = (end - start + 1)
        .checked_mul(N::get_byte_width())
        .ok_or_else(|| {
            Error::invalid_input("Variable-width mini-block offset bytes overflow usize")
        })?;
    value_bytes
        .checked_add(offset_bytes as i128)
        .ok_or_else(|| Error::invalid_input("Variable-width mini-block size overflows i128"))
}

fn validate_variable_offsets<N: OffsetSizeTrait>(
    offsets: &[N],
    num_values: u64,
    data_len: usize,
) -> Result<()> {
    validate_variable_offset_endpoints(offsets, num_values, data_len)?;
    let mut previous = None;
    for (index, offset) in offsets.iter().enumerate() {
        let offset = offset.to_usize().ok_or_else(|| {
            Error::invalid_input(format!(
                "Variable-width offset at index {index} is negative or does not fit usize"
            ))
        })?;
        if previous.is_some_and(|previous| offset < previous) {
            return Err(Error::invalid_input(format!(
                "Variable-width offsets decrease at index {index}"
            )));
        }
        previous = Some(offset);
    }
    Ok(())
}

fn validate_variable_offset_endpoints<N: OffsetSizeTrait>(
    offsets: &[N],
    num_values: u64,
    data_len: usize,
) -> Result<()> {
    let expected_offsets = usize::try_from(num_values)
        .ok()
        .and_then(|num_values| num_values.checked_add(1))
        .ok_or_else(|| Error::invalid_input("Variable-width offset count overflows usize"))?;
    if offsets.len() != expected_offsets {
        return Err(Error::invalid_input(format!(
            "Variable-width block has {} offsets, expected {expected_offsets}",
            offsets.len()
        )));
    }
    let first = offsets[0].to_usize().ok_or_else(|| {
        Error::invalid_input("First variable-width offset is negative or does not fit usize")
    })?;
    if first != 0 {
        return Err(Error::invalid_input(format!(
            "Variable-width offsets must start at zero, got {first}"
        )));
    }
    let last = offsets[offsets.len() - 1].to_usize().ok_or_else(|| {
        Error::invalid_input("Final variable-width offset is negative or does not fit usize")
    })?;
    if last != data_len {
        return Err(Error::invalid_input(format!(
            "Final variable-width offset {last} does not equal {data_len} data bytes"
        )));
    }
    Ok(())
}

fn legacy_variable_cost<N: OffsetSizeTrait>(
    encoding: &CompressiveEncoding,
    offsets: &[N],
    ranges: &[BinaryChunkRange],
    bits_per_offset: u64,
    context: MiniBlockCompressionContext,
) -> Result<u64> {
    let offset_bytes = usize::try_from(bits_per_offset / 8)
        .map_err(|_| Error::invalid_input("Offset width does not fit usize"))?;
    ranges
        .iter()
        .try_fold(encoding.encoded_len() as u64, |cost, range| {
            let num_offsets = range.end_offset_index - range.start_offset_index + 1;
            let value_bytes = chunk_value_range(offsets, *range)?.len();
            let raw_bytes = num_offsets
                .checked_mul(offset_bytes)
                .and_then(|bytes| bytes.checked_add(value_bytes))
                .ok_or_else(|| {
                    Error::invalid_input("Legacy variable chunk size overflows usize")
                })?;
            Ok(cost
                .saturating_add(context.chunk_header_bytes(1))
                .saturating_add((raw_bytes as u64).next_multiple_of(8)))
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenericOffsetPreflight {
    Legacy,
    DeltaFlat,
    FullSelection,
}

fn preflight_generic_offsets<N: OffsetSizeTrait>(
    offsets: &[N],
    ranges: &[BinaryChunkRange],
    bits_per_offset: u64,
    legacy_cost: u64,
    context: MiniBlockCompressionContext,
) -> Result<GenericOffsetPreflight> {
    const PROBE_DELTAS_PER_CHUNK: usize = 8;

    let value_bytes = bits_per_offset / 8;
    let mut direct_member_lengths = Vec::with_capacity(ranges.len());
    let mut delta_member_lengths = Vec::with_capacity(ranges.len());
    let mut direct_total_values = 0_u64;
    let mut delta_total_values = 0_u64;
    let mut direct_max = 0_u64;
    let mut observed_delta_max = 0_u64;
    let mut common_delta = None;
    let mut common_delta_range_start = None;
    let mut common_delta_range_step = None;
    let mut constant_delta_possible = true;
    let mut delta_range_possible = true;

    for range in ranges {
        let num_deltas = range.end_offset_index - range.start_offset_index;
        let num_offsets = num_deltas + 1;
        let num_offsets = num_offsets as u64;
        let num_deltas = num_deltas as u64;
        direct_member_lengths.push(num_offsets);
        delta_member_lengths.push(num_deltas);
        direct_total_values = direct_total_values
            .checked_add(num_offsets)
            .ok_or_else(|| Error::invalid_input("Offset family cardinality overflows u64"))?;
        delta_total_values = delta_total_values
            .checked_add(num_deltas)
            .ok_or_else(|| Error::invalid_input("Delta family cardinality overflows u64"))?;

        let start = offsets[range.start_offset_index]
            .to_i64()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| Error::invalid_input("Variable offset does not fit u64"))?;
        let end = offsets[range.end_offset_index]
            .to_i64()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| Error::invalid_input("Variable offset does not fit u64"))?;
        direct_max = direct_max.max(end.checked_sub(start).ok_or_else(|| {
            Error::invalid_input("Variable-width offsets decrease across a chunk")
        })?);

        let probe_end = range
            .start_offset_index
            .saturating_add(PROBE_DELTAS_PER_CHUNK)
            .min(range.end_offset_index);
        if probe_end - range.start_offset_index < 2 {
            delta_range_possible = false;
        }
        let mut previous_delta = None;
        for offset_index in range.start_offset_index..probe_end {
            let start = offsets[offset_index]
                .to_i64()
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| Error::invalid_input("Variable offset does not fit u64"))?;
            let end = offsets[offset_index + 1]
                .to_i64()
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| Error::invalid_input("Variable offset does not fit u64"))?;
            let delta = end.checked_sub(start).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Variable-width offsets decrease at index {}",
                    offset_index + 1
                ))
            })?;
            observed_delta_max = observed_delta_max.max(delta);
            match common_delta {
                Some(common) => constant_delta_possible &= delta == common,
                None => common_delta = Some(delta),
            }
            if previous_delta.is_none() {
                match common_delta_range_start {
                    Some(common) => delta_range_possible &= delta == common,
                    None => common_delta_range_start = Some(delta),
                }
            }
            if let Some(previous) = previous_delta {
                let step = delta.checked_sub(previous);
                match (common_delta_range_step, step) {
                    (Some(common), Some(step)) => delta_range_possible &= step == common,
                    (None, Some(step)) => common_delta_range_step = Some(step),
                    (_, None) => delta_range_possible = false,
                }
            }
            previous_delta = Some(delta);
        }
    }

    if constant_delta_possible || delta_range_possible {
        return Ok(GenericOffsetPreflight::FullSelection);
    }
    let required_bits = |max: u64| {
        if max == 0 {
            1
        } else {
            u64::from(u64::BITS - max.leading_zeros())
        }
    };
    if !family_bitpacking_cannot_reduce(
        direct_total_values,
        &direct_member_lengths,
        required_bits(direct_max),
        bits_per_offset,
    ) || !family_bitpacking_cannot_reduce(
        delta_total_values,
        &delta_member_lengths,
        required_bits(observed_delta_max),
        bits_per_offset,
    ) {
        return Ok(GenericOffsetPreflight::FullSelection);
    }

    let payload_bytes = delta_member_lengths
        .iter()
        .map(|num_values| {
            num_values
                .checked_mul(value_bytes)
                .ok_or_else(|| Error::invalid_input("Delta payload size overflows u64"))
        })
        .collect::<Result<Vec<_>>>()?;
    let encoding = ProtobufUtils21::variable(
        ProtobufUtils21::delta(
            bits_per_offset,
            0,
            ProtobufUtils21::flat(bits_per_offset, None),
        ),
        None,
    );
    let generic_cost =
        generic_variable_cost(&encoding, true, &payload_bytes, offsets, ranges, context)?;
    Ok(if generic_cost < legacy_cost {
        GenericOffsetPreflight::DeltaFlat
    } else {
        GenericOffsetPreflight::Legacy
    })
}

#[cfg(feature = "bitpacking")]
fn family_bitpacking_cannot_reduce(
    total_values: u64,
    member_lengths: &[u64],
    required_bits: u64,
    bits_per_value: u64,
) -> bool {
    if required_bits >= bits_per_value {
        return true;
    }
    if total_values <= BITPACK_CHUNK_VALUES {
        return false;
    }
    member_lengths.iter().all(|num_values| {
        if *num_values >= BITPACK_CHUNK_VALUES {
            return false;
        }
        let padding_cost = required_bits * (BITPACK_CHUNK_VALUES - num_values);
        let tail_savings = (bits_per_value - required_bits) * num_values;
        padding_cost >= tail_savings
    })
}

#[cfg(not(feature = "bitpacking"))]
fn family_bitpacking_cannot_reduce(
    _total_values: u64,
    _member_lengths: &[u64],
    _required_bits: u64,
    _bits_per_value: u64,
) -> bool {
    true
}

fn generic_variable_cost<N: OffsetSizeTrait>(
    encoding: &CompressiveEncoding,
    has_payload: bool,
    payload_bytes: &[u64],
    offsets: &[N],
    ranges: &[BinaryChunkRange],
    context: MiniBlockCompressionContext,
) -> Result<u64> {
    if payload_bytes.len() != ranges.len() {
        return Err(Error::internal(format!(
            "Offset family produced {} estimates for {} chunks",
            payload_bytes.len(),
            ranges.len()
        )));
    }
    ranges.iter().zip(payload_bytes).try_fold(
        encoding.encoded_len() as u64,
        |cost, (range, payload)| {
            let value_bytes = chunk_value_range(offsets, *range)?.len() as u64;
            let payload_cost = if has_payload {
                payload.next_multiple_of(8)
            } else {
                0
            };
            let value_buffers = 1 + u64::from(has_payload);
            Ok(cost
                .saturating_add(context.chunk_header_bytes(value_buffers))
                .saturating_add(payload_cost)
                .saturating_add(value_bytes.next_multiple_of(8)))
        },
    )
}

fn chunk_value_range<N: OffsetSizeTrait>(
    offsets: &[N],
    range: BinaryChunkRange,
) -> Result<std::ops::Range<usize>> {
    let start = offsets[range.start_offset_index]
        .to_usize()
        .ok_or_else(|| Error::invalid_input("Variable chunk start offset does not fit usize"))?;
    let end = offsets[range.end_offset_index]
        .to_usize()
        .ok_or_else(|| Error::invalid_input("Variable chunk end offset does not fit usize"))?;
    if start > end {
        return Err(Error::invalid_input(
            "Variable chunk offsets are decreasing",
        ));
    }
    Ok(start..end)
}

fn build_generic_chunks<N: OffsetSizeTrait>(
    data: VariableWidthBlock,
    offsets: &[N],
    ranges: &[BinaryChunkRange],
    family: OffsetFamilyCompressor,
    encoding: CompressiveEncoding,
) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
    let has_payload = family.has_payload();
    let offset_capacity =
        family
            .estimated_payload_bytes()
            .iter()
            .try_fold(0_usize, |total, payload_bytes| {
                let payload_bytes = usize::try_from(*payload_bytes).map_err(|_| {
                    Error::invalid_input("Generic offset payload size does not fit usize")
                })?;
                total.checked_add(payload_bytes).ok_or_else(|| {
                    Error::invalid_input("Generic offset payload capacity overflows usize")
                })
            })?;
    let value_capacity = ranges.iter().try_fold(0_usize, |total, range| {
        total
            .checked_add(chunk_value_range(offsets, *range)?.len())
            .ok_or_else(|| Error::invalid_input("Variable value capacity overflows usize"))
    })?;
    let payloads = family.compress_members()?;
    let mut offset_data = Vec::with_capacity(offset_capacity);
    let mut value_data = Vec::with_capacity(value_capacity);
    let mut chunks = Vec::with_capacity(ranges.len());

    for ((range, payload), is_last) in ranges
        .iter()
        .zip(payloads)
        .zip((0..ranges.len()).map(|index| index + 1 == ranges.len()))
    {
        let value_range = chunk_value_range(offsets, *range)?;
        if value_range.end > data.data.len() {
            return Err(Error::invalid_input(format!(
                "Variable chunk ends at {}, beyond {} data bytes",
                value_range.end,
                data.data.len()
            )));
        }
        let value_bytes = &data.data[value_range];
        let mut buffer_sizes = Vec::with_capacity(1 + usize::from(has_payload));
        if let Some(payload) = payload {
            buffer_sizes.push(u32::try_from(payload.len()).map_err(|_| {
                Error::invalid_input("Generic offset payload exceeds u32::MAX bytes")
            })?);
            offset_data.extend_from_slice(&payload);
        }
        buffer_sizes.push(u32::try_from(value_bytes.len()).map_err(|_| {
            Error::invalid_input("Variable chunk value payload exceeds u32::MAX bytes")
        })?);
        value_data.extend_from_slice(value_bytes);
        let num_values = range.end_offset_index - range.start_offset_index;
        chunks.push(MiniBlockChunk {
            buffer_sizes,
            log_num_values: if is_last {
                0
            } else {
                num_values.trailing_zeros() as u8
            },
        });
    }

    let mut buffers = Vec::with_capacity(1 + usize::from(has_payload));
    if has_payload {
        buffers.push(LanceBuffer::from(offset_data));
    }
    buffers.push(LanceBuffer::from(value_data));
    Ok((
        MiniBlockCompressed {
            data: buffers,
            chunks,
            num_values: data.num_values,
        },
        encoding,
    ))
}

impl BinaryMiniBlockEncoder {
    pub fn new(minichunk_size: Option<i64>) -> Self {
        Self {
            minichunk_size: minichunk_size.unwrap_or(*AIM_MINICHUNK_SIZE),
            generic_offsets: None,
        }
    }

    pub(crate) fn with_generic_offsets(
        minichunk_size: Option<i64>,
        field_params: CompressionFieldParams,
    ) -> Self {
        Self {
            minichunk_size: minichunk_size.unwrap_or(*AIM_MINICHUNK_SIZE),
            generic_offsets: Some(field_params),
        }
    }

    // put binary data into chunks, every chunk is less than or equal to `minichunk_size`.
    // In each chunk, offsets are put first then followed by binary bytes data, each chunk is padded to 8 bytes.
    // the offsets in the chunk points to the bytes offset in this chunk.
    fn chunk_data(
        &self,
        data: VariableWidthBlock,
        context: MiniBlockCompressionContext,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match data.bits_per_offset {
            32 => {
                let offsets_buffer = data.offsets.clone();
                let offsets = offsets_buffer.borrow_to_typed_slice::<i32>();
                self.chunk_typed_data(offsets.as_ref(), data, 32, 4, context)
            }
            64 => {
                let offsets_buffer = data.offsets.clone();
                let offsets = offsets_buffer.borrow_to_typed_slice::<i64>();
                self.chunk_typed_data(offsets.as_ref(), data, 64, 8, context)
            }
            _ => Err(Error::invalid_input(format!(
                "Unsupported bits_per_offset={}",
                data.bits_per_offset
            ))),
        }
    }

    fn chunk_typed_data<N: OffsetSizeTrait>(
        &self,
        offsets: &[N],
        data: VariableWidthBlock,
        bits_per_offset: u64,
        legacy_alignment: usize,
        context: MiniBlockCompressionContext,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        if context.allows_generic_offsets()
            && let Some(field_params) = self.generic_offsets.as_ref()
        {
            validate_variable_offset_endpoints(offsets, data.num_values, data.data.len())?;
            let ranges = binary_chunk_ranges(offsets, self.minichunk_size)?;
            let legacy_encoding =
                ProtobufUtils21::variable(ProtobufUtils21::flat(bits_per_offset, None), None);
            let legacy_cost =
                legacy_variable_cost(&legacy_encoding, offsets, &ranges, bits_per_offset, context)?;
            let preflight =
                preflight_generic_offsets(offsets, &ranges, bits_per_offset, legacy_cost, context)?;
            let member_ranges = || {
                ranges
                    .iter()
                    .map(|range| {
                        let end = range.end_offset_index.checked_add(1).ok_or_else(|| {
                            Error::invalid_input("Offset block end overflows usize")
                        })?;
                        Ok(range.start_offset_index..end)
                    })
                    .collect::<Result<Vec<_>>>()
            };
            let offset_block = || FixedWidthDataBlock {
                data: data.offsets.clone(),
                bits_per_value: bits_per_offset,
                num_values: offsets.len() as u64,
                block_info: BlockInfo::default(),
            };
            let payload_header_bytes = context
                .chunk_header_bytes(2)
                .saturating_sub(context.chunk_header_bytes(1));
            let block_cost = BlockCost::new(payload_header_bytes, 8);
            let family = match preflight {
                GenericOffsetPreflight::Legacy => {
                    validate_variable_offsets(offsets, data.num_values, data.data.len())?;
                    None
                }
                GenericOffsetPreflight::DeltaFlat => {
                    Some(select_delta_flat_offsets(offset_block(), member_ranges()?)?)
                }
                GenericOffsetPreflight::FullSelection => {
                    let mut offset_params = field_params.clone();
                    // The surrounding mini-block compressor owns general compression.
                    // Offset selection remains structural and compares exact payload sizes.
                    offset_params.compression = Some("none".to_string());
                    Some(select_offset_family(
                        offset_block(),
                        member_ranges()?,
                        &offset_params,
                        block_cost,
                    )?)
                }
            };
            if let Some(family) = family {
                let generic_encoding = ProtobufUtils21::variable(family.encoding().clone(), None);
                let generic_cost = generic_variable_cost(
                    &generic_encoding,
                    family.has_payload(),
                    family.estimated_payload_bytes(),
                    offsets,
                    &ranges,
                    context,
                )?;
                let is_ambiguous_flat = matches!(
                    family.encoding().compression.as_ref(),
                    Some(Compression::Flat(_))
                );
                if !is_ambiguous_flat && generic_cost < legacy_cost {
                    return build_generic_chunks(data, offsets, &ranges, family, generic_encoding);
                }
            }
            let (buffers, chunks) =
                chunk_offsets_with_ranges(offsets, &data.data, legacy_alignment, &ranges)?;
            return Ok((
                MiniBlockCompressed {
                    data: buffers,
                    chunks,
                    num_values: data.num_values,
                },
                legacy_encoding,
            ));
        }

        validate_variable_offsets(offsets, data.num_values, data.data.len())?;
        let (buffers, chunks) =
            chunk_offsets(offsets, &data.data, legacy_alignment, self.minichunk_size)?;
        Ok((
            MiniBlockCompressed {
                data: buffers,
                chunks,
                num_values: data.num_values,
            },
            ProtobufUtils21::variable(ProtobufUtils21::flat(bits_per_offset, None), None),
        ))
    }
}

impl MiniBlockCompressor for BinaryMiniBlockEncoder {
    fn compress(
        &self,
        data: DataBlock,
        context: MiniBlockCompressionContext,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match data {
            DataBlock::VariableWidth(variable_width) => self.chunk_data(variable_width, context),
            _ => Err(Error::invalid_input_source(
                format!(
                    "Cannot compress a data block of type {} with BinaryMiniBlockEncoder",
                    data.name()
                )
                .into(),
            )),
        }
    }
}

#[derive(Debug)]
pub struct BinaryMiniBlockDecompressor {
    layout: BinaryMiniBlockLayout,
}

#[derive(Debug)]
enum BinaryMiniBlockLayout {
    Legacy {
        bits_per_offset: u8,
    },
    Generic {
        value_type: BlockValueType,
        offsets: Box<dyn BlockDecompressor>,
        offsets_have_payload: bool,
        validation: OffsetValidation,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OffsetValidation {
    Endpoints,
    Full,
}

fn offset_validation(compression: &Compression) -> OffsetValidation {
    match compression {
        Compression::Constant(_) | Compression::Range(_) | Compression::Delta(_) => {
            OffsetValidation::Endpoints
        }
        _ => OffsetValidation::Full,
    }
}

impl BinaryMiniBlockDecompressor {
    pub fn new(bits_per_offset: u8) -> Self {
        assert!(matches!(bits_per_offset, 32 | 64));
        Self {
            layout: BinaryMiniBlockLayout::Legacy { bits_per_offset },
        }
    }

    pub fn from_variable(variable: &pb21::Variable) -> Result<Self> {
        if variable.values.is_some() {
            return Err(Error::invalid_input(
                "Binary mini-block Variable values encoding must be absent",
            ));
        }
        let offsets = variable
            .offsets
            .as_ref()
            .ok_or_else(|| Error::invalid_input("Variable encoding is missing offsets"))?;
        let compression = offsets
            .compression
            .as_ref()
            .ok_or_else(|| Error::invalid_input("Variable offsets are missing compression"))?;
        if let Compression::Flat(flat) = compression {
            if flat.data.is_some() || !matches!(flat.bits_per_value, 32 | 64) {
                return Err(Error::invalid_input(format!(
                    "Legacy Variable offsets require uncompressed 32 or 64-bit Flat encoding, got {} bits",
                    flat.bits_per_value
                )));
            }
            Ok(Self {
                layout: BinaryMiniBlockLayout::Legacy {
                    bits_per_offset: flat.bits_per_value as u8,
                },
            })
        } else {
            let value_type = infer_block_value_type(offsets)?;
            if !matches!(value_type, BlockValueType::UInt32 | BlockValueType::UInt64) {
                return Err(Error::invalid_input(format!(
                    "Generic Variable offsets require u32 or u64 output, got {} bits",
                    value_type.bits_per_value()
                )));
            }
            let validation = offset_validation(compression);
            let (offsets, offsets_have_payload) = create_block_decompressor(offsets, value_type)?;
            Ok(Self {
                layout: BinaryMiniBlockLayout::Generic {
                    value_type,
                    offsets,
                    offsets_have_payload,
                    validation,
                },
            })
        }
    }
}

impl MiniBlockDecompressor for BinaryMiniBlockDecompressor {
    // decompress a MiniBlock of binary data, the num_values must be less than or equal
    // to the number of values this MiniBlock has, BinaryMiniBlock doesn't store `the number of values`
    // it has so assertion can not be done here and the caller of `decompress` must ensure
    // `num_values` <= number of values in the chunk.
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        match &self.layout {
            BinaryMiniBlockLayout::Legacy { bits_per_offset } => {
                decode_legacy_binary_miniblock(data, num_values, *bits_per_offset)
            }
            BinaryMiniBlockLayout::Generic {
                value_type,
                offsets,
                offsets_have_payload,
                validation,
            } => decode_generic_binary_miniblock(
                data,
                num_values,
                *value_type,
                offsets.as_ref(),
                *offsets_have_payload,
                *validation,
            ),
        }
    }
}

fn decode_legacy_binary_miniblock(
    mut data: Vec<LanceBuffer>,
    num_values: u64,
    bits_per_offset: u8,
) -> Result<DataBlock> {
    if data.len() != 1 {
        return Err(Error::invalid_input(format!(
            "Legacy Variable mini-block requires 1 buffer, got {}",
            data.len()
        )));
    }
    let data = data.pop().expect("buffer count was checked");
    let num_offsets = num_values
        .checked_add(1)
        .ok_or_else(|| Error::invalid_input("Variable offset count overflows u64"))?;
    let num_offsets = usize::try_from(num_offsets)
        .map_err(|_| Error::invalid_input("Variable offset count does not fit usize"))?;
    match bits_per_offset {
        32 => {
            let offset_bytes = num_offsets.checked_mul(4).ok_or_else(|| {
                Error::invalid_input("Variable offset byte length overflows usize")
            })?;
            if data.len() < offset_bytes {
                return Err(Error::invalid_input(format!(
                    "Legacy Variable mini-block has {} bytes, shorter than {offset_bytes} offset bytes",
                    data.len()
                )));
            }
            let offsets_buffer = data.borrow_to_typed_slice::<u32>();
            let offsets = offsets_buffer
                .get(..num_offsets)
                .ok_or_else(|| Error::invalid_input("Legacy Variable offsets are truncated"))?;
            let start = offsets[0];
            let mut result_offsets = Vec::with_capacity(num_offsets);
            let mut previous = start;
            for (index, offset) in offsets.iter().copied().enumerate() {
                if offset < previous {
                    return Err(Error::invalid_input(format!(
                        "Legacy Variable offsets decrease at index {index}"
                    )));
                }
                let relative = offset - start;
                if relative > i32::MAX as u32 {
                    return Err(Error::invalid_input(format!(
                        "Legacy Variable relative offset {relative} exceeds i32::MAX"
                    )));
                }
                result_offsets.push(relative);
                previous = offset;
            }
            let start = usize::try_from(start)
                .map_err(|_| Error::invalid_input("Variable data start does not fit usize"))?;
            let end = usize::try_from(*offsets.last().expect("offsets are non-empty"))
                .map_err(|_| Error::invalid_input("Variable data end does not fit usize"))?;
            if start < offset_bytes || end < start || end > data.len() {
                return Err(Error::invalid_input(format!(
                    "Legacy Variable data range {start}..{end} is invalid for {} bytes",
                    data.len()
                )));
            }
            Ok(DataBlock::VariableWidth(VariableWidthBlock {
                data: LanceBuffer::from(data[start..end].to_vec()),
                offsets: LanceBuffer::reinterpret_vec(result_offsets),
                bits_per_offset,
                num_values,
                block_info: BlockInfo::new(),
            }))
        }
        64 => {
            let offset_bytes = num_offsets.checked_mul(8).ok_or_else(|| {
                Error::invalid_input("Variable offset byte length overflows usize")
            })?;
            if data.len() < offset_bytes {
                return Err(Error::invalid_input(format!(
                    "Legacy Variable mini-block has {} bytes, shorter than {offset_bytes} offset bytes",
                    data.len()
                )));
            }
            let offsets_buffer = data.borrow_to_typed_slice::<u64>();
            let offsets = offsets_buffer
                .get(..num_offsets)
                .ok_or_else(|| Error::invalid_input("Legacy Variable offsets are truncated"))?;
            let start = offsets[0];
            let mut result_offsets = Vec::with_capacity(num_offsets);
            let mut previous = start;
            for (index, offset) in offsets.iter().copied().enumerate() {
                if offset < previous {
                    return Err(Error::invalid_input(format!(
                        "Legacy Variable offsets decrease at index {index}"
                    )));
                }
                let relative = offset - start;
                if relative > i64::MAX as u64 {
                    return Err(Error::invalid_input(format!(
                        "Legacy Variable relative offset {relative} exceeds i64::MAX"
                    )));
                }
                result_offsets.push(relative);
                previous = offset;
            }
            let start = usize::try_from(start)
                .map_err(|_| Error::invalid_input("Variable data start does not fit usize"))?;
            let end = usize::try_from(*offsets.last().expect("offsets are non-empty"))
                .map_err(|_| Error::invalid_input("Variable data end does not fit usize"))?;
            if start < offset_bytes || end < start || end > data.len() {
                return Err(Error::invalid_input(format!(
                    "Legacy Variable data range {start}..{end} is invalid for {} bytes",
                    data.len()
                )));
            }
            Ok(DataBlock::VariableWidth(VariableWidthBlock {
                data: LanceBuffer::from(data[start..end].to_vec()),
                offsets: LanceBuffer::reinterpret_vec(result_offsets),
                bits_per_offset,
                num_values,
                block_info: BlockInfo::new(),
            }))
        }
        _ => Err(Error::invalid_input(format!(
            "Legacy Variable offsets require 32 or 64 bits, got {bits_per_offset}"
        ))),
    }
}

fn decode_generic_binary_miniblock(
    mut data: Vec<LanceBuffer>,
    num_values: u64,
    value_type: BlockValueType,
    offsets_decoder: &dyn BlockDecompressor,
    offsets_have_payload: bool,
    validation: OffsetValidation,
) -> Result<DataBlock> {
    let num_offsets = num_values
        .checked_add(1)
        .ok_or_else(|| Error::invalid_input("Variable offset count overflows u64"))?;
    let expected_buffers = 1 + usize::from(offsets_have_payload);
    if data.len() != expected_buffers {
        return Err(Error::invalid_input(format!(
            "Generic Variable mini-block requires {expected_buffers} buffers, got {}",
            data.len()
        )));
    }
    let values = data.pop().expect("buffer count was checked");
    let offset_payload = if offsets_have_payload {
        Some(data.pop().expect("buffer count was checked"))
    } else {
        None
    };
    let offsets = offsets_decoder.decompress(offset_payload, num_offsets)?;
    let DataBlock::FixedWidth(offsets) = offsets else {
        return Err(Error::invalid_input(
            "Generic Variable offsets decoded to a non fixed-width block",
        ));
    };
    if offsets.num_values != num_offsets || offsets.bits_per_value != value_type.bits_per_value() {
        return Err(Error::invalid_input(format!(
            "Generic Variable offsets decoded {} {}-bit values, expected {num_offsets} {}-bit values",
            offsets.num_values,
            offsets.bits_per_value,
            value_type.bits_per_value()
        )));
    }
    validate_decoded_offsets(&offsets, value_type, values.len(), validation)?;
    Ok(DataBlock::VariableWidth(VariableWidthBlock {
        data: values,
        offsets: offsets.data,
        bits_per_offset: value_type.bits_per_value() as u8,
        num_values,
        block_info: BlockInfo::new(),
    }))
}

fn validate_decoded_offsets(
    offsets: &FixedWidthDataBlock,
    value_type: BlockValueType,
    data_len: usize,
    validation: OffsetValidation,
) -> Result<()> {
    let signed_max = match value_type {
        BlockValueType::UInt32 => i32::MAX as u64,
        BlockValueType::UInt64 => i64::MAX as u64,
        _ => {
            return Err(Error::invalid_input(
                "Generic Variable offsets require u32 or u64 values",
            ));
        }
    };
    if validation == OffsetValidation::Endpoints {
        let (first, last) = match value_type {
            BlockValueType::UInt32 => {
                let offsets = offsets.data.borrow_to_typed_slice::<u32>();
                let first = offsets.first().copied().ok_or_else(|| {
                    Error::invalid_input("Generic Variable offsets cannot be empty")
                })?;
                let last = offsets.last().copied().expect("first offset was present");
                (u64::from(first), u64::from(last))
            }
            BlockValueType::UInt64 => {
                let offsets = offsets.data.borrow_to_typed_slice::<u64>();
                let first = offsets.first().copied().ok_or_else(|| {
                    Error::invalid_input("Generic Variable offsets cannot be empty")
                })?;
                let last = offsets.last().copied().expect("first offset was present");
                (first, last)
            }
            _ => unreachable!("generic offset type was validated"),
        };
        if first != 0 {
            return Err(Error::invalid_input(format!(
                "Generic Variable offsets must start at zero, got {first}"
            )));
        }
        if last > signed_max {
            return Err(Error::invalid_input(format!(
                "Generic Variable offset {last} exceeds the Arrow signed offset range"
            )));
        }
        if last != data_len as u64 {
            return Err(Error::invalid_input(format!(
                "Generic Variable final offset {last} does not equal {data_len} value bytes"
            )));
        }
        return Ok(());
    }
    let mut previous = None;
    let mut observe = |index: usize, offset: u64| -> Result<()> {
        if index == 0 && offset != 0 {
            return Err(Error::invalid_input(format!(
                "Generic Variable offsets must start at zero, got {offset}"
            )));
        }
        if previous.is_some_and(|previous| offset < previous) {
            return Err(Error::invalid_input(format!(
                "Generic Variable offsets decrease at index {index}"
            )));
        }
        if offset > signed_max {
            return Err(Error::invalid_input(format!(
                "Generic Variable offset {offset} exceeds the Arrow signed offset range"
            )));
        }
        previous = Some(offset);
        Ok(())
    };
    match value_type {
        BlockValueType::UInt32 => {
            for (index, offset) in offsets
                .data
                .borrow_to_typed_slice::<u32>()
                .iter()
                .copied()
                .enumerate()
            {
                observe(index, u64::from(offset))?;
            }
        }
        BlockValueType::UInt64 => {
            for (index, offset) in offsets
                .data
                .borrow_to_typed_slice::<u64>()
                .iter()
                .copied()
                .enumerate()
            {
                observe(index, offset)?;
            }
        }
        _ => unreachable!("generic offset type was validated"),
    }
    let final_offset =
        previous.ok_or_else(|| Error::invalid_input("Generic Variable offsets cannot be empty"))?;
    if final_offset != data_len as u64 {
        return Err(Error::invalid_input(format!(
            "Generic Variable final offset {final_offset} does not equal {data_len} value bytes"
        )));
    }
    Ok(())
}

/// Most basic encoding for variable-width data which does no compression at all
/// The DataBlock memory layout looks like below:
///
/// | bits_per_offset           | bytes_start_offset        | offsets data | bytes data |
/// | ------------------------- | ------------------------- | ------------ | ---------- |
/// | <bits_per_offset>/8 bytes | <bits_per_offset>/8 bytes | offsets_len  | data_len   |
///
/// It's used in VariableEncoder and BinaryBlockDecompressor
///
#[derive(Debug, Default)]
pub struct VariableEncoder {}

impl BlockCompressor for VariableEncoder {
    fn compress(&self, mut data: DataBlock) -> Result<Option<LanceBuffer>> {
        match data {
            DataBlock::VariableWidth(ref mut variable_width_data) => {
                match variable_width_data.bits_per_offset {
                    32 => {
                        let offsets = variable_width_data.offsets.borrow_to_typed_slice::<u32>();
                        let offsets = offsets.as_ref();
                        // The first 4 bytes store the bits per offset, the next 4 bytes store the start
                        // offset of the bytes data, then offsets data, then bytes data.
                        let bytes_start_offset = 4 + 4 + std::mem::size_of_val(offsets) as u32;

                        let output_total_bytes =
                            bytes_start_offset as usize + variable_width_data.data.len();
                        let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

                        // Store bit_per_offset info
                        output.extend_from_slice(&(32_u32).to_le_bytes());

                        // store `bytes_start_offset` in the next 4 bytes of output buffer
                        output.extend_from_slice(&(bytes_start_offset).to_le_bytes());

                        // store offsets
                        output.extend_from_slice(&variable_width_data.offsets);

                        // store bytes
                        output.extend_from_slice(&variable_width_data.data);
                        Ok(LanceBuffer::from(output))
                    }
                    64 => {
                        let offsets = variable_width_data.offsets.borrow_to_typed_slice::<u64>();
                        let offsets = offsets.as_ref();
                        // The first 8 bytes store the bits per offset, the next 8 bytes store the start
                        // offset of the bytes data, then offsets data, then bytes data.
                        let bytes_start_offset = 8 + 8 + std::mem::size_of_val(offsets) as u64;

                        let output_total_bytes =
                            bytes_start_offset as usize + variable_width_data.data.len();
                        let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

                        // Store bit_per_offset info
                        output.extend_from_slice(&(64_u64).to_le_bytes());

                        // store `bytes_start_offset` in the next 8 bytes of output buffer
                        output.extend_from_slice(&(bytes_start_offset).to_le_bytes());

                        // store offsets
                        output.extend_from_slice(&variable_width_data.offsets);

                        // store bytes
                        output.extend_from_slice(&variable_width_data.data);
                        Ok(LanceBuffer::from(output))
                    }
                    _ => Err(Error::invalid_input(format!(
                        "BinaryBlockEncoder does not support {}-bit offsets",
                        variable_width_data.bits_per_offset
                    ))),
                }
            }
            _ => Err(Error::invalid_input(
                "BinaryBlockEncoder requires a variable-width block",
            )),
        }
        .map(Some)
    }
}

impl PerValueCompressor for VariableEncoder {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, CompressiveEncoding)> {
        let DataBlock::VariableWidth(variable) = data else {
            panic!("BinaryPerValueCompressor can only work with Variable Width DataBlock.");
        };

        let encoding = ProtobufUtils21::variable(
            ProtobufUtils21::flat(variable.bits_per_offset as u64, None),
            None,
        );
        Ok((PerValueDataBlock::Variable(variable), encoding))
    }
}

#[derive(Debug, Default)]
pub struct VariableDecoder {}

impl VariablePerValueDecompressor for VariableDecoder {
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock> {
        Ok(DataBlock::VariableWidth(data))
    }
}

#[derive(Debug, Default)]
pub struct BinaryBlockDecompressor {}

impl BlockDecompressor for BinaryBlockDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let data = require_block_payload(data, "Binary block")?;
        // In older (not quite stable) versions we stored the bits per offset as a single byte and then the num_values
        // as four bytes.  However, this led to alignment problems and was wasteful since we already store the num_values
        // in higher layers.
        //
        // In the standard scheme we use 4 bytes for the bits per offset and 4 bytes for the bytes_start_offset and we
        // rely on the passed in num_values to be correct.

        // This isn't perfect but it's probably good enough and the best I think we can do.  The bits per offset will
        // never be more than 255 and it's little endian so the last 3 bytes will always be 0.  These will be the least
        // significant 3 bytes of the number of values in the old scheme.  It's pretty unlikely these are all 0 (that would
        // mean there are at least 16M values in a single page) so we'll use this to determine if the old scheme is used.
        let is_old_scheme = data[1] != 0 || data[2] != 0 || data[3] != 0;

        let (bits_per_offset, bytes_start_offset, offset_start) = if is_old_scheme {
            // Old scheme
            let bits_per_offset = data[0];
            match bits_per_offset {
                32 => {
                    debug_assert_eq!(LittleEndian::read_u32(&data[1..5]), num_values as u32);
                    let bytes_start_offset = LittleEndian::read_u32(&data[5..9]);
                    (bits_per_offset, bytes_start_offset as u64, 9)
                }
                64 => {
                    debug_assert_eq!(LittleEndian::read_u64(&data[1..9]), num_values);
                    let bytes_start_offset = LittleEndian::read_u64(&data[9..17]);
                    (bits_per_offset, bytes_start_offset, 17)
                }
                _ => {
                    return Err(Error::invalid_input_source(
                        format!("Unsupported bits_per_offset={}", bits_per_offset).into(),
                    ));
                }
            }
        } else {
            // Standard scheme
            let bits_per_offset = LittleEndian::read_u32(&data[0..4]) as u8;
            match bits_per_offset {
                32 => {
                    let bytes_start_offset = LittleEndian::read_u32(&data[4..8]);
                    (bits_per_offset, bytes_start_offset as u64, 8)
                }
                64 => {
                    let bytes_start_offset = LittleEndian::read_u64(&data[8..16]);
                    (bits_per_offset, bytes_start_offset, 16)
                }
                _ => {
                    return Err(Error::invalid_input_source(
                        format!("Unsupported bits_per_offset={}", bits_per_offset).into(),
                    ));
                }
            }
        };

        // the next `bytes_start_offset - offset_start` stores the offsets.
        let offsets =
            data.slice_with_length(offset_start, bytes_start_offset as usize - offset_start);

        // the rest are the binary bytes.
        let data = data.slice_with_length(
            bytes_start_offset as usize,
            data.len() - bytes_start_offset as usize,
        );

        Ok(DataBlock::VariableWidth(VariableWidthBlock {
            data,
            offsets,
            bits_per_offset,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        ArrayRef, StringArray,
        builder::{LargeStringBuilder, StringBuilder},
    };
    use arrow_schema::{DataType, Field};

    use crate::{
        constants::{
            COMPRESSION_META_KEY, DICT_DIVISOR_META_KEY, STRUCTURAL_ENCODING_FULLZIP,
            STRUCTURAL_ENCODING_META_KEY, STRUCTURAL_ENCODING_MINIBLOCK,
        },
        testing::check_specific_random,
    };
    use rstest::rstest;
    use std::{collections::HashMap, sync::Arc, vec};

    use crate::{
        compression_config::CompressionFieldParams,
        testing::{
            FnArrayGeneratorProvider, TestCases, check_basic_random,
            check_round_trip_encoding_of_data,
        },
        version::LanceFileVersion,
    };

    fn miniblock_context() -> MiniBlockCompressionContext {
        MiniBlockCompressionContext::new(0, true, true)
    }

    fn decode_binary_miniblocks(
        compressed: MiniBlockCompressed,
        encoding: &CompressiveEncoding,
    ) -> Result<Vec<VariableWidthBlock>> {
        let Compression::Variable(variable) = encoding
            .compression
            .as_ref()
            .ok_or_else(|| Error::invalid_input("missing Variable encoding"))?
        else {
            return Err(Error::invalid_input("expected Variable encoding"));
        };
        let decoder = BinaryMiniBlockDecompressor::from_variable(variable)?;
        let mut buffer_offsets = vec![0_usize; compressed.data.len()];
        let mut values_seen = 0_u64;
        let mut decoded = Vec::with_capacity(compressed.chunks.len());
        for chunk in compressed.chunks {
            let num_values = chunk.num_values(values_seen, compressed.num_values);
            values_seen += num_values;
            let buffers = chunk
                .buffer_sizes
                .iter()
                .zip(compressed.data.iter().zip(&mut buffer_offsets))
                .map(|(size, (buffer, offset))| {
                    let size = *size as usize;
                    let chunk = buffer.slice_with_length(*offset, size);
                    *offset += size;
                    chunk
                })
                .collect();
            let DataBlock::VariableWidth(block) = decoder.decompress(buffers, num_values)? else {
                return Err(Error::internal(
                    "Binary mini-block decoded a non-variable block".to_string(),
                ));
            };
            decoded.push(block);
        }
        Ok(decoded)
    }

    fn variable_block_u32(lengths: &[usize]) -> VariableWidthBlock {
        let mut offsets = Vec::with_capacity(lengths.len() + 1);
        let mut data = Vec::new();
        offsets.push(0_i32);
        for (index, length) in lengths.iter().copied().enumerate() {
            data.extend(std::iter::repeat_n((index % 251) as u8, length));
            offsets.push(i32::try_from(data.len()).unwrap());
        }
        VariableWidthBlock {
            data: LanceBuffer::from(data),
            offsets: LanceBuffer::reinterpret_vec(offsets),
            bits_per_offset: 32,
            num_values: lengths.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    fn assert_decoded_value_lengths(decoded: &[VariableWidthBlock], expected: &[usize]) {
        let actual = decoded
            .iter()
            .flat_map(|block| {
                let offsets = block.offsets.borrow_to_typed_slice::<i32>();
                offsets
                    .windows(2)
                    .map(|pair| (pair[1] - pair[0]) as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn generic_offsets_use_range_for_fixed_width_values() {
        let lengths = vec![3_usize; 2_048];
        let block = variable_block_u32(&lengths);
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(256),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Range(_))
        ));
        assert_eq!(compressed.data.len(), 1);
        assert!(compressed.chunks.len() > 1);
        assert!(
            compressed
                .chunks
                .iter()
                .all(|chunk| chunk.buffer_sizes.len() == 1)
        );

        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert_decoded_value_lengths(&decoded, &lengths);
    }

    #[test]
    fn generic_offsets_use_delta_for_irregular_values() {
        let lengths = (0..4_096)
            .map(|index| [1_usize, 7, 2, 5][index % 4])
            .collect::<Vec<_>>();
        let block = variable_block_u32(&lengths);
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(1_024),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Delta(_))
        ));
        assert_eq!(compressed.data.len(), 2);
        assert!(
            compressed
                .chunks
                .iter()
                .all(|chunk| chunk.buffer_sizes.len() == 2)
        );

        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert_decoded_value_lengths(&decoded, &lengths);
    }

    #[test]
    fn generic_offsets_use_delta_range_for_increasing_lengths() {
        let lengths = (0..4_096)
            .map(|index| 4_usize + index % 64)
            .collect::<Vec<_>>();
        let block = variable_block_u32(&lengths);
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(4_096),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        let Some(Compression::Delta(delta)) = variable
            .offsets
            .as_deref()
            .and_then(|offsets| offsets.compression.as_ref())
        else {
            panic!("expected Delta offsets");
        };
        assert!(matches!(
            delta
                .deltas
                .as_deref()
                .and_then(|deltas| deltas.compression.as_ref()),
            Some(Compression::Range(_))
        ));
        assert_eq!(compressed.data.len(), 1);
        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert_decoded_value_lengths(&decoded, &lengths);
    }

    #[test]
    fn preflight_keeps_legacy_when_delta_flat_does_not_cover_header() {
        let lengths = (0..4_096)
            .map(|index| [16_usize, 22, 17, 20][index % 4])
            .collect::<Vec<_>>();
        let block = variable_block_u32(&lengths);
        let offsets = block.offsets.borrow_to_typed_slice::<i32>();
        let ranges = binary_chunk_ranges(offsets.as_ref(), DEFAULT_AIM_MINICHUNK_SIZE).unwrap();
        let encoding = ProtobufUtils21::variable(ProtobufUtils21::flat(32, None), None);
        let context = MiniBlockCompressionContext::new(0, true, true);
        let legacy_cost =
            legacy_variable_cost(&encoding, offsets.as_ref(), &ranges, 32, context).unwrap();
        assert_eq!(
            preflight_generic_offsets(offsets.as_ref(), &ranges, 32, legacy_cost, context).unwrap(),
            GenericOffsetPreflight::Legacy
        );
    }

    #[test]
    fn generic_offsets_keep_smaller_legacy_container() {
        let block = variable_block_u32(&[1, 7, 2, 5]);
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(4_096),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Flat(_))
        ));
        assert_eq!(compressed.data.len(), 1);
        assert_eq!(compressed.chunks[0].buffer_sizes.len(), 1);
    }

    #[test]
    fn generic_offsets_support_metadata_only_empty_values() {
        let lengths = vec![0_usize; 1_024];
        let block = variable_block_u32(&lengths);
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(256),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Constant(_))
        ));
        assert_eq!(compressed.data.len(), 1);
        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert_decoded_value_lengths(&decoded, &lengths);
    }

    #[test]
    fn generic_offsets_support_u64_range() {
        let num_values = 512_usize;
        let offsets = (0..=num_values)
            .map(|index| (index * 2) as i64)
            .collect::<Vec<_>>();
        let block = VariableWidthBlock {
            data: LanceBuffer::from(vec![1_u8; num_values * 2]),
            offsets: LanceBuffer::reinterpret_vec(offsets),
            bits_per_offset: 64,
            num_values: num_values as u64,
            block_info: BlockInfo::default(),
        };
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(256),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Range(_))
        ));
        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert!(decoded.iter().all(|block| block.bits_per_offset == 64));
    }

    #[test]
    fn legacy_offsets_remain_interleaved_flat() {
        let block = variable_block_u32(&vec![3_usize; 128]);
        let encoder = BinaryMiniBlockEncoder::new(Some(256));
        let (compressed, encoding) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Flat(_))
        ));
        assert_eq!(compressed.data.len(), 1);
        assert!(
            compressed
                .chunks
                .iter()
                .all(|chunk| chunk.buffer_sizes.len() == 1)
        );
    }

    #[test]
    fn legacy_u32_interleaved_bytes_are_stable() {
        let block = variable_block_u32(&[3, 3]);
        let encoder = BinaryMiniBlockEncoder::new(Some(4_096));
        let (compressed, _) = encoder
            .compress(DataBlock::VariableWidth(block), miniblock_context())
            .unwrap();
        assert_eq!(compressed.chunks.len(), 1);
        assert_eq!(compressed.chunks[0].buffer_sizes, [20]);

        let mut expected = Vec::new();
        expected.extend_from_slice(&12_i32.to_le_bytes());
        expected.extend_from_slice(&15_i32.to_le_bytes());
        expected.extend_from_slice(&18_i32.to_le_bytes());
        expected.extend_from_slice(&[0, 0, 0, 1, 1, 1]);
        expected.extend_from_slice(&[72, 72]);
        assert_eq!(compressed.data[0].as_ref(), expected);
    }

    #[test]
    fn generic_offsets_reject_wrong_buffer_count_and_bounds() {
        let variable = pb21::Variable {
            offsets: Some(Box::new(ProtobufUtils21::range(32, 0, 3))),
            values: None,
        };
        let decoder = BinaryMiniBlockDecompressor::from_variable(&variable).unwrap();
        let error = decoder
            .decompress(vec![LanceBuffer::empty(), LanceBuffer::empty()], 2)
            .unwrap_err();
        assert!(error.to_string().contains("requires 1 buffers"));

        let error = decoder
            .decompress(vec![LanceBuffer::from(vec![0_u8; 5])], 2)
            .unwrap_err();
        assert!(error.to_string().contains("final offset 6"));
    }

    #[test]
    fn generic_offsets_only_skip_full_scan_for_monotonic_codecs() {
        let range = ProtobufUtils21::range(32, 0, 3);
        let delta = ProtobufUtils21::delta(32, 0, ProtobufUtils21::flat(32, None));
        let rle = ProtobufUtils21::rle(
            ProtobufUtils21::flat(32, None),
            ProtobufUtils21::constant(None),
        );
        assert_eq!(
            offset_validation(range.compression.as_ref().unwrap()),
            OffsetValidation::Endpoints
        );
        assert_eq!(
            offset_validation(delta.compression.as_ref().unwrap()),
            OffsetValidation::Endpoints
        );
        assert_eq!(
            offset_validation(rle.compression.as_ref().unwrap()),
            OffsetValidation::Full
        );

        let offsets = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(vec![0_u32, 5, 3]),
            bits_per_value: 32,
            num_values: 3,
            block_info: BlockInfo::default(),
        };
        let error =
            validate_decoded_offsets(&offsets, BlockValueType::UInt32, 3, OffsetValidation::Full)
                .unwrap_err();
        assert!(error.to_string().contains("decrease at index 2"));
    }

    #[rstest]
    #[case::range([16_usize; 4], "range")]
    #[case::delta([4_usize, 10, 5, 8], "delta")]
    #[test_log::test(tokio::test)]
    async fn generic_offsets_support_scan_range_take(
        #[case] lengths: [usize; 4],
        #[case] expected_encoding: &str,
    ) {
        let values = StringArray::from_iter_values((0..10_000).map(|index| {
            let len = lengths[index % lengths.len()];
            format!("{index:04x}{}", "x".repeat(len - 4))
        }));
        let metadata = HashMap::from([
            (
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
            ),
            (COMPRESSION_META_KEY.to_string(), "none".to_string()),
            (DICT_DIVISOR_META_KEY.to_string(), "100000".to_string()),
        ]);
        let test_cases = TestCases::basic()
            .with_min_file_version(LanceFileVersion::V2_3)
            .with_expected_encoding(expected_encoding);
        check_round_trip_encoding_of_data(vec![Arc::new(values)], &test_cases, metadata).await;
    }

    fn variable_miniblock_decoder(offsets: CompressiveEncoding) -> BinaryMiniBlockDecompressor {
        let encoding = ProtobufUtils21::variable(offsets, None);
        let Compression::Variable(variable) = encoding.compression.as_ref().unwrap() else {
            unreachable!()
        };
        BinaryMiniBlockDecompressor::from_variable(variable).unwrap()
    }

    #[test]
    fn generic_range_offsets_decode_without_payload() {
        let decoder = variable_miniblock_decoder(ProtobufUtils21::range(32, 0, 3));
        let decoded = decoder
            .decompress(vec![LanceBuffer::from(b"aaabbbccc".to_vec())], 3)
            .unwrap();
        let DataBlock::VariableWidth(decoded) = decoded else {
            panic!("expected variable-width output");
        };
        assert_eq!(
            decoded.offsets.borrow_to_typed_slice::<u32>().as_ref(),
            &[0, 3, 6, 9]
        );
        assert_eq!(decoded.data.as_ref(), b"aaabbbccc");
    }

    #[test]
    fn generic_delta_offsets_decode_with_payload() {
        let decoder = variable_miniblock_decoder(ProtobufUtils21::delta(
            32,
            0,
            ProtobufUtils21::flat(32, None),
        ));
        let decoded = decoder
            .decompress(
                vec![
                    LanceBuffer::reinterpret_vec(vec![2_u32, 0, 3]),
                    LanceBuffer::from(b"abcde".to_vec()),
                ],
                3,
            )
            .unwrap();
        let DataBlock::VariableWidth(decoded) = decoded else {
            panic!("expected variable-width output");
        };
        assert_eq!(
            decoded.offsets.borrow_to_typed_slice::<u32>().as_ref(),
            &[0, 2, 2, 5]
        );
        assert_eq!(decoded.data.as_ref(), b"abcde");
    }

    #[test]
    fn generic_offsets_reject_buffer_count_and_bounds() {
        let decoder = variable_miniblock_decoder(ProtobufUtils21::range(32, 0, 3));
        assert!(
            decoder
                .decompress(vec![LanceBuffer::empty(), LanceBuffer::empty()], 2)
                .unwrap_err()
                .to_string()
                .contains("requires 1 buffer")
        );

        let decoder = variable_miniblock_decoder(ProtobufUtils21::range(32, 0, 4));
        assert!(
            decoder
                .decompress(vec![LanceBuffer::from(vec![0; 7])], 2)
                .unwrap_err()
                .to_string()
                .contains("does not equal 7")
        );
    }

    #[test]
    fn legacy_offsets_reject_malformed_buffers() {
        let decoder = BinaryMiniBlockDecompressor::new(32);
        assert!(
            decoder
                .decompress(Vec::new(), 1)
                .unwrap_err()
                .to_string()
                .contains("requires 1 buffer")
        );
        assert!(
            decoder
                .decompress(vec![LanceBuffer::from(vec![0; 4])], 1)
                .unwrap_err()
                .to_string()
                .contains("shorter than 8")
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_utf8_binary() {
        let field = Field::new("", DataType::Utf8, false);
        check_specific_random(
            field,
            TestCases::basic().with_min_file_version(LanceFileVersion::V2_1),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Utf8, DataType::Binary)] data_type: DataType,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );

        let field = Field::new("", data_type, false).with_metadata(field_metadata);
        check_basic_random(field).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary_fsst(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Binary, DataType::Utf8)] data_type: DataType,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        field_metadata.insert(COMPRESSION_META_KEY.to_string(), "fsst".into());
        let field = Field::new("", data_type, true).with_metadata(field_metadata);
        // TODO (https://github.com/lance-format/lance/issues/4783)
        let test_cases = TestCases::default().with_min_file_version(LanceFileVersion::V2_1);
        check_specific_random(field, test_cases).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_fsst_large_binary(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::LargeBinary, DataType::LargeUtf8)] data_type: DataType,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        field_metadata.insert(COMPRESSION_META_KEY.to_string(), "fsst".into());
        let field = Field::new("", data_type, true).with_metadata(field_metadata);
        check_specific_random(
            field,
            TestCases::basic().with_min_file_version(LanceFileVersion::V2_1),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_binary() {
        let field = Field::new("", DataType::LargeBinary, true);
        check_basic_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_utf8() {
        let field = Field::new("", DataType::LargeUtf8, true);
        check_basic_random(field).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_small_strings(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
    ) {
        use crate::testing::check_basic_generated;

        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        let field = Field::new("", DataType::Utf8, true).with_metadata(field_metadata);
        check_basic_generated(
            field,
            Box::new(FnArrayGeneratorProvider::new(move || {
                lance_datagen::array::utf8_prefix_plus_counter("user_", /*is_large=*/ false)
            })),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_simple_binary(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Utf8, DataType::Binary)] data_type: DataType,
    ) {
        let string_array = StringArray::from(vec![Some("abc"), None, Some("pqr"), None, Some("m")]);
        let string_array = arrow_cast::cast(&string_array, &data_type).unwrap();

        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![0, 1, 3, 4]);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            field_metadata,
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_sliced_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);
        let string_array = string_array.slice(1, 3);

        let test_cases = TestCases::default()
            .with_range(0..1)
            .with_range(0..2)
            .with_range(1..2);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_bigger_than_max_page_size() {
        // Create an array with one single 32MiB string
        let big_string = String::from_iter((0..(32 * 1024 * 1024)).map(|_| '0'));
        let string_array = StringArray::from(vec![
            Some(big_string),
            Some("abc".to_string()),
            None,
            None,
            Some("xyz".to_string()),
        ]);

        // Drop the max page size to 1MiB
        let test_cases = TestCases::default().with_max_page_size(1024 * 1024);

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;

        // This is a regression testing the case where a page with X rows is split into Y parts
        // where the number of parts is not evenly divisible by the number of rows.  In this
        // case we are splitting 90 rows into 4 parts.
        let big_string = String::from_iter((0..(1000 * 1000)).map(|_| '0'));
        let string_array = StringArray::from_iter_values((0..90).map(|_| big_string.clone()));

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &TestCases::default(),
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_strings() {
        // Scenario 1: Some strings are empty

        let values = [Some("abc"), Some(""), None];
        // Test empty list at beginning, middle, and end
        for order in [[0, 1, 2], [1, 0, 2], [2, 0, 1]] {
            let mut string_builder = StringBuilder::new();
            for idx in order {
                string_builder.append_option(values[idx]);
            }
            let string_array = Arc::new(string_builder.finish());
            let test_cases = TestCases::default()
                .with_indices(vec![1])
                .with_indices(vec![0])
                .with_indices(vec![2])
                .with_indices(vec![0, 1]);
            check_round_trip_encoding_of_data(
                vec![string_array.clone()],
                &test_cases,
                HashMap::new(),
            )
            .await;
            let test_cases = test_cases.with_batch_size(1);
            check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new())
                .await;
        }

        // Scenario 2: All strings are empty

        // When encoding an array of empty strings there are no bytes to encode
        // which is strange and we want to ensure we handle it
        let string_array = Arc::new(StringArray::from(vec![Some(""), None, Some("")]));

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases, HashMap::new())
            .await;
        let test_cases = test_cases.with_batch_size(1);
        check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    #[ignore] // This test is quite slow in debug mode
    async fn test_jumbo_string() {
        // This is an overflow test.  We have a list of lists where each list
        // has 1Mi items.  We encode 5000 of these lists and so we have over 4Gi in the
        // offsets range
        let mut string_builder = LargeStringBuilder::new();
        // a 1 MiB string
        let giant_string = String::from_iter((0..(1024 * 1024)).map(|_| '0'));
        for _ in 0..5000 {
            string_builder.append_option(Some(&giant_string));
        }
        let giant_array = Arc::new(string_builder.finish()) as ArrayRef;
        let arrs = vec![giant_array];

        // // We can't validate because our validation relies on concatenating all input arrays
        let test_cases = TestCases::default().without_validation();
        check_round_trip_encoding_of_data(arrs, &test_cases, HashMap::new()).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary_dictionary_encoding(
        #[values(true, false)] with_nulls: bool,
        #[values(100, 500, 35000)] dict_size: u32,
    ) {
        let test_cases = TestCases::default().with_min_file_version(LanceFileVersion::V2_1);
        let strings = (0..dict_size)
            .map(|i| i.to_string())
            .collect::<Vec<String>>();

        let repeated_strings: Vec<_> = strings
            .iter()
            .cycle()
            .take(70000)
            .enumerate()
            .map(|(i, s)| {
                if with_nulls && i % 7 == 0 {
                    None
                } else {
                    Some(s.clone())
                }
            })
            .collect();
        let string_array = Arc::new(StringArray::from(repeated_strings)) as ArrayRef;
        check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_binary_encoding_verification() {
        use lance_datagen::{ByteCount, RowCount};

        let test_cases = TestCases::default()
            .with_expected_encoding("variable")
            .with_min_file_version(LanceFileVersion::V2_1);

        // Test both automatic selection and explicit configuration
        // 1. Test automatic binary encoding selection (small strings that won't trigger FSST)
        let arr_small = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(10), false))
            .into_batch_rows(RowCount::from(1000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr_small], &test_cases, HashMap::new()).await;

        // 2. Test explicit "none" compression to force binary encoding
        let metadata_explicit =
            HashMap::from([("lance-encoding:compression".to_string(), "none".to_string())]);
        let arr_large = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(50), false))
            .into_batch_rows(RowCount::from(2000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr_large], &test_cases, metadata_explicit).await;
    }

    #[test]
    fn test_binary_miniblock_with_misaligned_buffer() {
        use super::BinaryMiniBlockDecompressor;
        use crate::buffer::LanceBuffer;
        use crate::compression::MiniBlockDecompressor;
        use crate::data::DataBlock;

        // Test case 1: u32 offsets
        {
            let decompressor = BinaryMiniBlockDecompressor::new(32);

            // Create test data with u32 offsets
            // BinaryMiniBlock format: all offsets followed by all string data
            // Need to ensure total size is divisible by 4 for u32
            let mut test_data = Vec::new();

            // Offsets section (3 offsets for 2 values + 1 end offset)
            test_data.extend_from_slice(&12u32.to_le_bytes()); // offset to start of strings (after offsets)
            test_data.extend_from_slice(&15u32.to_le_bytes()); // offset to second string
            test_data.extend_from_slice(&20u32.to_le_bytes()); // offset to end

            // String data section
            test_data.extend_from_slice(b"ABCXYZ"); // 6 bytes of string data
            test_data.extend_from_slice(&[0, 0]); // 2 bytes padding to make total 20 bytes (divisible by 4)

            // Create a misaligned buffer by adding padding and slicing
            let mut padded = Vec::with_capacity(test_data.len() + 1);
            padded.push(0xFF); // Padding byte to misalign
            padded.extend_from_slice(&test_data);

            let bytes = bytes::Bytes::from(padded);
            let misaligned = bytes.slice(1..); // Skip first byte to create misalignment

            // Create LanceBuffer with bytes_per_value=1 to bypass alignment check
            let buffer = LanceBuffer::from_bytes(misaligned, 1);

            // Verify the buffer is actually misaligned
            let ptr = buffer.as_ref().as_ptr();
            assert_ne!(
                ptr.align_offset(4),
                0,
                "Test setup: buffer should be misaligned for u32"
            );

            // Decompress with misaligned buffer - should work with borrow_to_typed_slice
            let result = decompressor.decompress(vec![buffer], 2);
            assert!(
                result.is_ok(),
                "Decompression should succeed with misaligned buffer"
            );

            // Verify the data is correct
            if let Ok(DataBlock::VariableWidth(block)) = result {
                assert_eq!(block.num_values, 2);
                // Data should be the strings (including padding from the original buffer)
                assert_eq!(&block.data.as_ref()[..6], b"ABCXYZ");
            } else {
                panic!("Expected VariableWidth block");
            }
        }

        // Test case 2: u64 offsets
        {
            let decompressor = BinaryMiniBlockDecompressor::new(64);

            // Create test data with u64 offsets
            let mut test_data = Vec::new();

            // Offsets section (3 offsets for 2 values + 1 end offset)
            test_data.extend_from_slice(&24u64.to_le_bytes()); // offset to start of strings (after offsets)
            test_data.extend_from_slice(&29u64.to_le_bytes()); // offset to second string
            test_data.extend_from_slice(&40u64.to_le_bytes()); // offset to end (divisible by 8)

            // String data section
            test_data.extend_from_slice(b"HelloWorld"); // 10 bytes of string data
            test_data.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // 6 bytes padding to make total 40 bytes (divisible by 8)

            // Create misaligned buffer
            let mut padded = Vec::with_capacity(test_data.len() + 3);
            padded.extend_from_slice(&[0xFF, 0xFF, 0xFF]); // 3 bytes padding for misalignment
            padded.extend_from_slice(&test_data);

            let bytes = bytes::Bytes::from(padded);
            let misaligned = bytes.slice(3..); // Skip 3 bytes

            let buffer = LanceBuffer::from_bytes(misaligned, 1);

            // Verify misalignment for u64
            let ptr = buffer.as_ref().as_ptr();
            assert_ne!(
                ptr.align_offset(8),
                0,
                "Test setup: buffer should be misaligned for u64"
            );

            // Decompress should succeed
            let result = decompressor.decompress(vec![buffer], 2);
            assert!(
                result.is_ok(),
                "Decompression should succeed with misaligned u64 buffer"
            );

            if let Ok(DataBlock::VariableWidth(block)) = result {
                assert_eq!(block.num_values, 2);
                // Data should be the strings (including padding from the original buffer)
                assert_eq!(&block.data.as_ref()[..10], b"HelloWorld");
            } else {
                panic!("Expected VariableWidth block");
            }
        }
    }
}
