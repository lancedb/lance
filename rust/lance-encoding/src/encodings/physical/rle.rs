// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # RLE (Run-Length Encoding)
//!
//! RLE compression for Lance, optimized for data with repeated values.
//!
//! ## Encoding Format
//!
//! RLE uses a dual-buffer format to store compressed data:
//!
//! - **Values Buffer**: Stores unique values in their original data type
//! - **Lengths Buffer**: Stores the repeat count for each value as u8, u16, or u32
//!
//! ### Example
//!
//! Input data: `[1, 1, 1, 2, 2, 3, 3, 3, 3]`
//!
//! Encoded as:
//! - Values buffer: `[1, 2, 3]` (3 × 4 bytes for i32)
//! - Lengths buffer: `[3, 2, 4]` (3 × 1 byte for u8 in compatibility mode)
//!
//! ### Long Run Handling
//!
//! In compatibility mode, when a run exceeds 255 values, it is split into multiple
//! runs of 255 followed by a final run with the remainder. RLE v2 can use u16 or
//! u32 run lengths to reduce this splitting.
//!
//! ## Supported Types
//!
//! RLE supports all fixed-width primitive types:
//! - 8-bit: u8, i8
//! - 16-bit: u16, i16
//! - 32-bit: u32, i32, f32
//! - 64-bit: u64, i64, f64
//!
//! ## Compression Strategy
//!
//! RLE is automatically selected when:
//! - The run count (number of value transitions) < 50% of total values
//! - This indicates sufficient repetition for RLE to be effective
//!
//! ## MiniBlock Chunk Handling
//!
//! When used in the miniblock path, all chunks share two global buffers (values and lengths).
//! Each chunk's `buffer_sizes` identifies its slice within those global buffers. Non-last chunks
//! contain a power-of-2 number of values.
//!
//! NOTE: The current encoder uses a 2048-value cap per chunk as a workaround for
//! <https://github.com/lancedb/lance/issues/4429>.
//!
//! ## Block Format
//!
//! When used in the block compression path, the encoded output is a single buffer:
//! `[8-byte header: values buffer size][values buffer][run_lengths buffer]`.

use arrow_buffer::ArrowNativeType;
use log::trace;

use crate::buffer::LanceBuffer;
use crate::compression::{BlockCompressor, BlockDecompressor, MiniBlockDecompressor};
use crate::data::DataBlock;
use crate::data::{BlockInfo, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MAX_MINIBLOCK_BYTES, MAX_MINIBLOCK_VALUES, MiniBlockChunk, MiniBlockCompressed,
    MiniBlockCompressor,
};
use crate::format::ProtobufUtils21;
use crate::format::pb21::CompressiveEncoding;

use lance_core::{Error, Result};

/// Width used to encode RLE run lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunLengthWidth {
    /// Compatibility mode. Runs longer than 255 values are split.
    U8,
    /// RLE v2 mode for runs up to 65,535 values per entry.
    U16,
    /// RLE v2 mode for runs up to 4,294,967,295 values per entry.
    U32,
}

impl RunLengthWidth {
    pub(crate) fn from_bits(bits_per_value: u64) -> Option<Self> {
        match bits_per_value {
            8 => Some(Self::U8),
            16 => Some(Self::U16),
            32 => Some(Self::U32),
            _ => None,
        }
    }

    pub(crate) fn bits_per_value(self) -> u64 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::U32 => 32,
        }
    }

    fn bytes_per_value(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }

    fn max_run_length(self) -> u64 {
        match self {
            Self::U8 => u8::MAX as u64,
            Self::U16 => u16::MAX as u64,
            Self::U32 => u32::MAX as u64,
        }
    }

    fn write_length(self, length: u64, dst: &mut Vec<u8>) {
        match self {
            Self::U8 => dst.push(length as u8),
            Self::U16 => dst.extend_from_slice(&(length as u16).to_le_bytes()),
            Self::U32 => dst.extend_from_slice(&(length as u32).to_le_bytes()),
        }
    }

    fn read_length(self, bytes: &[u8]) -> u64 {
        match self {
            Self::U8 => bytes[0] as u64,
            Self::U16 => u16::from_le_bytes([bytes[0], bytes[1]]) as u64,
            Self::U32 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
        }
    }
}

/// Select the lowest-cost run length width for a fixed-width data block.
pub(crate) fn select_run_length_width(
    data: &LanceBuffer,
    num_values: u64,
    bits_per_value: u64,
) -> Result<RunLengthWidth> {
    select_run_length_width_and_size(data, num_values, bits_per_value).map(|(width, _)| width)
}

/// Select the lowest-cost run length width and return the encoded size estimate.
pub(crate) fn select_run_length_width_and_size(
    data: &LanceBuffer,
    num_values: u64,
    bits_per_value: u64,
) -> Result<(RunLengthWidth, u128)> {
    match bits_per_value {
        8 => select_run_length_width_generic::<u8>(data, num_values),
        16 => select_run_length_width_generic::<u16>(data, num_values),
        32 => select_run_length_width_generic::<u32>(data, num_values),
        64 => select_run_length_width_generic::<u64>(data, num_values),
        _ => Err(Error::invalid_input_source(
            format!("RLE encoding bits_per_value must be 8, 16, 32, or 64, got {bits_per_value}")
                .into(),
        )),
    }
}

fn select_run_length_width_generic<T>(
    data: &LanceBuffer,
    num_values: u64,
) -> Result<(RunLengthWidth, u128)>
where
    T: bytemuck::Pod + PartialEq + Copy + ArrowNativeType,
{
    let num_values = usize::try_from(num_values).map_err(|_| {
        Error::invalid_input_source(
            format!("RLE num_values does not fit in usize: {num_values}").into(),
        )
    })?;
    if num_values == 0 {
        return Ok((RunLengthWidth::U8, 0));
    }

    let type_size = std::mem::size_of::<T>();
    let expected_bytes = num_values.checked_mul(type_size).ok_or_else(|| {
        Error::invalid_input_source(
            format!("RLE input byte length overflow: {num_values} values of {type_size} bytes")
                .into(),
        )
    })?;
    if data.len() != expected_bytes {
        return Err(Error::invalid_input_source(
            format!(
                "RLE input data size mismatch: {} bytes for {} values of {} bytes",
                data.len(),
                num_values,
                type_size
            )
            .into(),
        ));
    }

    let values_ref = data.borrow_to_typed_slice::<T>();
    let values: &[T] = values_ref.as_ref();
    let mut costs = [0_u128; 3];

    let mut current_value = values[0];
    let mut current_length = 1_u64;
    for &value in values.iter().skip(1) {
        if value == current_value {
            current_length += 1;
        } else {
            accumulate_width_costs(current_length, type_size, &mut costs);
            current_value = value;
            current_length = 1;
        }
    }
    accumulate_width_costs(current_length, type_size, &mut costs);

    let widths = [RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32];
    let mut best_idx = 0usize;
    let mut best_cost = costs[0];
    for (idx, &cost) in costs.iter().enumerate().skip(1) {
        if cost < best_cost {
            best_idx = idx;
            best_cost = cost;
        }
    }
    Ok((widths[best_idx], best_cost))
}

fn accumulate_width_costs(run_length: u64, type_size: usize, costs: &mut [u128; 3]) {
    let widths = [RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32];
    // The current encoder uses miniblock-sized chunks for both miniblock and block paths.
    let max_segment_values = *MAX_MINIBLOCK_VALUES;
    let mut remaining = run_length;
    while remaining > 0 {
        let segment = remaining.min(max_segment_values);
        for (idx, width) in widths.iter().enumerate() {
            let entries = segment.div_ceil(width.max_run_length());
            costs[idx] += (entries as u128) * ((type_size + width.bytes_per_value()) as u128);
        }
        remaining -= segment;
    }
}

/// RLE encoder for miniblock format
#[derive(Debug)]
pub struct RleEncoder {
    run_length_width: RunLengthWidth,
}

impl Default for RleEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RleEncoder {
    pub fn new() -> Self {
        Self {
            run_length_width: RunLengthWidth::U8,
        }
    }

    pub(crate) fn with_run_length_width(run_length_width: RunLengthWidth) -> Self {
        Self { run_length_width }
    }

    fn encode_data(
        &self,
        data: &LanceBuffer,
        num_values: u64,
        bits_per_value: u64,
    ) -> Result<(Vec<LanceBuffer>, Vec<MiniBlockChunk>)> {
        if num_values == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let num_values = usize::try_from(num_values).map_err(|_| {
            Error::invalid_input_source(
                format!("RLE num_values does not fit in usize: {num_values}").into(),
            )
        })?;
        let bytes_per_value = (bits_per_value / 8) as usize;
        let bytes_per_length = self.run_length_width.bytes_per_value();

        // Pre-allocate global buffers with estimated capacity
        // Assume average compression ratio of ~10:1 (10 values per run)
        let estimated_runs = num_values / 10;
        let mut all_values = Vec::with_capacity(estimated_runs * bytes_per_value);
        let mut all_lengths = Vec::with_capacity(estimated_runs * bytes_per_length);
        let mut chunks = Vec::new();

        let mut offset = 0usize;
        let mut values_remaining = num_values;

        while values_remaining > 0 {
            let values_start = all_values.len();
            let lengths_start = all_lengths.len();

            let (_num_runs, values_processed, is_last_chunk) = match bits_per_value {
                8 => self.encode_chunk_rolling::<u8>(
                    data,
                    offset,
                    values_remaining,
                    &mut all_values,
                    &mut all_lengths,
                ),
                16 => self.encode_chunk_rolling::<u16>(
                    data,
                    offset,
                    values_remaining,
                    &mut all_values,
                    &mut all_lengths,
                ),
                32 => self.encode_chunk_rolling::<u32>(
                    data,
                    offset,
                    values_remaining,
                    &mut all_values,
                    &mut all_lengths,
                ),
                64 => self.encode_chunk_rolling::<u64>(
                    data,
                    offset,
                    values_remaining,
                    &mut all_values,
                    &mut all_lengths,
                ),
                _ => {
                    return Err(Error::invalid_input_source(
                        format!(
                            "RLE encoding bits_per_value must be 8, 16, 32, or 64, got {bits_per_value}"
                        )
                        .into(),
                    ));
                }
            };

            if values_processed == 0 {
                break;
            }

            let log_num_values = if is_last_chunk {
                0
            } else {
                assert!(
                    values_processed.is_power_of_two(),
                    "Non-last chunk must have power-of-2 values"
                );
                values_processed.ilog2() as u8
            };

            let values_size = all_values.len() - values_start;
            let lengths_size = all_lengths.len() - lengths_start;

            let chunk = MiniBlockChunk {
                buffer_sizes: vec![values_size as u32, lengths_size as u32],
                log_num_values,
            };

            chunks.push(chunk);

            offset += values_processed;
            values_remaining -= values_processed;
        }

        // Return exactly two buffers: values and lengths
        Ok((
            vec![
                LanceBuffer::from(all_values),
                LanceBuffer::from(all_lengths),
            ],
            chunks,
        ))
    }

    /// Encodes the largest valid mini-block prefix from `offset`.
    fn encode_chunk_rolling<T>(
        &self,
        data: &LanceBuffer,
        offset: usize,
        values_remaining: usize,
        all_values: &mut Vec<u8>,
        all_lengths: &mut Vec<u8>,
    ) -> (usize, usize, bool)
    where
        T: bytemuck::Pod + PartialEq + Copy + std::fmt::Debug + ArrowNativeType,
    {
        let type_size = std::mem::size_of::<T>();
        let chunk_start = offset * type_size;
        let max_by_count = *MAX_MINIBLOCK_VALUES as usize;
        let max_values = values_remaining.min(max_by_count);
        let chunk_end = chunk_start + max_values * type_size;

        if chunk_start >= data.len() {
            return (0, 0, false);
        }

        let chunk_len = chunk_end.min(data.len()) - chunk_start;
        let chunk_buffer = data.slice_with_length(chunk_start, chunk_len);
        let typed_data_ref = chunk_buffer.borrow_to_typed_slice::<T>();
        let typed_data: &[T] = typed_data_ref.as_ref();
        let max_values = max_values.min(typed_data.len());

        if typed_data.is_empty() {
            return (0, 0, false);
        }

        let values_start = all_values.len();
        let all_remaining_values_fit = values_remaining <= max_by_count;
        let encoded_size = self.encoded_size(&typed_data[..max_values]);
        let (values_to_encode, is_last_chunk) = if all_remaining_values_fit
            && encoded_size <= MAX_MINIBLOCK_BYTES as usize
        {
            (max_values, true)
        } else if let Some(values_to_encode) = self.largest_power_of_two_prefix::<T>(typed_data) {
            (values_to_encode, false)
        } else {
            return (0, 0, false);
        };

        self.encode_values(&typed_data[..values_to_encode], all_values, all_lengths);

        let num_runs = (all_values.len() - values_start) / type_size;
        (num_runs, values_to_encode, is_last_chunk)
    }

    fn largest_power_of_two_prefix<T>(&self, values: &[T]) -> Option<usize>
    where
        T: bytemuck::Pod + PartialEq + Copy,
    {
        let max_prefix = values.len().min(*MAX_MINIBLOCK_VALUES as usize);
        let mut prefix = 1usize << max_prefix.ilog2();
        while prefix > 1 {
            if self.encoded_size(&values[..prefix]) <= MAX_MINIBLOCK_BYTES as usize {
                return Some(prefix);
            }
            prefix >>= 1;
        }
        None
    }

    fn encoded_size<T>(&self, values: &[T]) -> usize
    where
        T: bytemuck::Pod + PartialEq + Copy,
    {
        if values.is_empty() {
            return 0;
        }

        let mut current_value = values[0];
        let mut current_length = 1u64;
        let mut encoded_size = 0usize;

        for &value in values.iter().skip(1) {
            if value == current_value {
                current_length += 1;
            } else {
                encoded_size += self.run_size::<T>(current_length);
                current_value = value;
                current_length = 1;
            }
        }
        encoded_size += self.run_size::<T>(current_length);
        encoded_size
    }

    fn run_size<T>(&self, length: u64) -> usize
    where
        T: bytemuck::Pod,
    {
        let type_size = std::mem::size_of::<T>();
        let run_chunks = length.div_ceil(self.run_length_width.max_run_length()) as usize;
        run_chunks * (type_size + self.run_length_width.bytes_per_value())
    }

    fn encode_values<T>(&self, values: &[T], all_values: &mut Vec<u8>, all_lengths: &mut Vec<u8>)
    where
        T: bytemuck::Pod + PartialEq + Copy,
    {
        if values.is_empty() {
            return;
        }

        let mut current_value = values[0];
        let mut current_length = 1u64;

        for &value in values.iter().skip(1) {
            if value == current_value {
                current_length += 1;
            } else {
                self.add_run(&current_value, current_length, all_values, all_lengths);
                current_value = value;
                current_length = 1;
            }
        }
        self.add_run(&current_value, current_length, all_values, all_lengths);
    }

    fn add_run<T>(
        &self,
        value: &T,
        length: u64,
        all_values: &mut Vec<u8>,
        all_lengths: &mut Vec<u8>,
    ) -> usize
    where
        T: bytemuck::Pod,
    {
        let value_bytes = bytemuck::bytes_of(value);
        let type_size = std::mem::size_of::<T>();
        let max_run_length = self.run_length_width.max_run_length();
        let num_full_chunks = (length / max_run_length) as usize;
        let remainder = length % max_run_length;

        let total_chunks = num_full_chunks + if remainder > 0 { 1 } else { 0 };
        all_values.reserve(total_chunks * type_size);
        all_lengths.reserve(total_chunks * self.run_length_width.bytes_per_value());

        for _ in 0..num_full_chunks {
            all_values.extend_from_slice(value_bytes);
            self.run_length_width
                .write_length(max_run_length, all_lengths);
        }

        if remainder > 0 {
            all_values.extend_from_slice(value_bytes);
            self.run_length_width.write_length(remainder, all_lengths);
        }

        total_chunks * (type_size + self.run_length_width.bytes_per_value())
    }
}

impl MiniBlockCompressor for RleEncoder {
    fn compress(&self, data: DataBlock) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match data {
            DataBlock::FixedWidth(fixed_width) => {
                let num_values = fixed_width.num_values;
                let bits_per_value = fixed_width.bits_per_value;

                let (all_buffers, chunks) =
                    self.encode_data(&fixed_width.data, num_values, bits_per_value)?;

                let compressed = MiniBlockCompressed {
                    data: all_buffers,
                    chunks,
                    num_values,
                };

                let encoding = ProtobufUtils21::rle(
                    ProtobufUtils21::flat(bits_per_value, None),
                    ProtobufUtils21::flat(self.run_length_width.bits_per_value(), None),
                );

                Ok((compressed, encoding))
            }
            _ => Err(Error::invalid_input_source(
                "RLE encoding only supports FixedWidth data blocks".into(),
            )),
        }
    }
}

impl BlockCompressor for RleEncoder {
    // Block format: [8-byte header: values buffer size][values buffer][run_lengths buffer]
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer> {
        match data {
            DataBlock::FixedWidth(fixed_width) => {
                let num_values = fixed_width.num_values;
                let bits_per_value = fixed_width.bits_per_value;

                let (all_buffers, _) =
                    self.encode_data(&fixed_width.data, num_values, bits_per_value)?;

                let values_size = all_buffers[0].len() as u64;

                let mut combined = Vec::new();
                combined.extend_from_slice(&values_size.to_le_bytes());
                combined.extend_from_slice(&all_buffers[0]);
                combined.extend_from_slice(&all_buffers[1]);
                Ok(LanceBuffer::from(combined))
            }
            _ => Err(Error::invalid_input_source(
                "RLE encoding only supports FixedWidth data blocks".into(),
            )),
        }
    }
}

/// RLE decompressor for miniblock format
#[derive(Debug)]
pub struct RleDecompressor {
    bits_per_value: u64,
    run_length_width: RunLengthWidth,
}

impl RleDecompressor {
    pub fn new(bits_per_value: u64) -> Self {
        Self {
            bits_per_value,
            run_length_width: RunLengthWidth::U8,
        }
    }

    pub(crate) fn with_run_length_width(
        bits_per_value: u64,
        run_length_width: RunLengthWidth,
    ) -> Self {
        Self {
            bits_per_value,
            run_length_width,
        }
    }

    fn decode_data(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                bits_per_value: self.bits_per_value,
                data: LanceBuffer::from(vec![]),
                num_values: 0,
                block_info: BlockInfo::default(),
            }));
        }

        if data.len() != 2 {
            return Err(Error::invalid_input_source(
                format!(
                    "RLE decompressor expects exactly 2 buffers, got {}",
                    data.len()
                )
                .into(),
            ));
        }

        let values_buffer = &data[0];
        let lengths_buffer = &data[1];

        let decoded_data = match self.bits_per_value {
            8 => self.decode_generic::<u8>(values_buffer, lengths_buffer, num_values)?,
            16 => self.decode_generic::<u16>(values_buffer, lengths_buffer, num_values)?,
            32 => self.decode_generic::<u32>(values_buffer, lengths_buffer, num_values)?,
            64 => self.decode_generic::<u64>(values_buffer, lengths_buffer, num_values)?,
            _ => {
                return Err(Error::invalid_input_source(
                    format!(
                        "RLE decoding bits_per_value must be 8, 16, 32, or 64, got {}",
                        self.bits_per_value
                    )
                    .into(),
                ));
            }
        };

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_value,
            data: decoded_data,
            num_values,
            block_info: BlockInfo::default(),
        }))
    }

    fn decode_generic<T>(
        &self,
        values_buffer: &LanceBuffer,
        lengths_buffer: &LanceBuffer,
        num_values: u64,
    ) -> Result<LanceBuffer>
    where
        T: bytemuck::Pod + Copy + std::fmt::Debug + ArrowNativeType,
    {
        let type_size = std::mem::size_of::<T>();
        let length_size = self.run_length_width.bytes_per_value();

        if values_buffer.is_empty() || lengths_buffer.is_empty() {
            if num_values == 0 {
                return Ok(LanceBuffer::empty());
            } else {
                return Err(Error::invalid_input_source(
                    format!("Empty buffers but expected {} values", num_values).into(),
                ));
            }
        }

        if !values_buffer.len().is_multiple_of(type_size)
            || !lengths_buffer.len().is_multiple_of(length_size)
        {
            return Err(Error::invalid_input_source(format!(
                "Invalid buffer sizes for RLE {} decoding: values {} bytes (not divisible by {}), lengths {} bytes (not divisible by {})",
                std::any::type_name::<T>(),
                values_buffer.len(),
                type_size,
                lengths_buffer.len(),
                length_size
            )
            .into()));
        }

        let num_runs = values_buffer.len() / type_size;
        let num_length_entries = lengths_buffer.len() / length_size;
        if num_runs != num_length_entries {
            return Err(Error::invalid_input_source(
                format!(
                    "Inconsistent RLE buffers: {} runs but {} length entries",
                    num_runs, num_length_entries
                )
                .into(),
            ));
        }

        let values_ref = values_buffer.borrow_to_typed_slice::<T>();
        let values: &[T] = values_ref.as_ref();
        let lengths = lengths_buffer.as_ref();

        let expected_value_count = usize::try_from(num_values).map_err(|_| {
            Error::invalid_input_source(
                format!("RLE num_values does not fit in usize: {num_values}").into(),
            )
        })?;
        let mut decoded_value_count = 0usize;
        for length_bytes in lengths.chunks_exact(length_size) {
            let length = self.run_length_width.read_length(length_bytes);
            if length == 0 {
                return Err(Error::invalid_input_source(
                    "RLE decoding encountered a zero run length".into(),
                ));
            }
            let length = usize::try_from(length).map_err(|_| {
                Error::invalid_input_source(
                    format!("RLE run length does not fit in usize: {length}").into(),
                )
            })?;
            decoded_value_count = decoded_value_count.checked_add(length).ok_or_else(|| {
                Error::invalid_input_source("RLE run length sum overflowed usize".into())
            })?;
            if decoded_value_count > expected_value_count {
                return Err(Error::invalid_input_source(
                    format!(
                        "RLE decoding overflowed expected value count: produced at least {}, expected {}",
                        decoded_value_count, expected_value_count
                    )
                    .into(),
                ));
            }
        }

        if decoded_value_count != expected_value_count {
            return Err(Error::invalid_input_source(
                format!(
                    "RLE decoding produced {} values, expected {}",
                    decoded_value_count, expected_value_count
                )
                .into(),
            ));
        }

        let mut decoded: Vec<T> = Vec::with_capacity(expected_value_count);
        for (value, length_bytes) in values.iter().zip(lengths.chunks_exact(length_size)) {
            let length = self.run_length_width.read_length(length_bytes) as usize;
            decoded.resize(decoded.len() + length, *value);
        }

        trace!(
            "RLE decoded {} {} values",
            num_values,
            std::any::type_name::<T>()
        );
        Ok(LanceBuffer::reinterpret_vec(decoded))
    }
}

impl MiniBlockDecompressor for RleDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        self.decode_data(data, num_values)
    }
}

impl BlockDecompressor for RleDecompressor {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        // fetch the values_size
        if data.len() < 8 {
            return Err(Error::invalid_input_source(
                format!("Insufficient data size: {}", data.len()).into(),
            ));
        }

        let values_size_bytes: [u8; 8] =
            data[..8].try_into().expect("slice length already checked");
        let values_size: u64 = u64::from_le_bytes(values_size_bytes);

        // parse values
        let values_start: usize = 8;
        let values_size: usize = values_size.try_into().map_err(|_| {
            Error::invalid_input_source(
                format!("Invalid values buffer size: {}", values_size).into(),
            )
        })?;
        let lengths_start = values_start
            .checked_add(values_size)
            .ok_or_else(|| Error::invalid_input_source("Invalid RLE values buffer size".into()))?;

        if data.len() < lengths_start {
            return Err(Error::invalid_input_source(
                format!("Insufficient data size: {}", data.len()).into(),
            ));
        }

        let values_buffer = data.slice_with_length(values_start, values_size);
        let lengths_buffer = data.slice_with_length(lengths_start, data.len() - lengths_start);

        self.decode_data(vec![values_buffer, lengths_buffer], num_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataBlock;
    use crate::encodings::logical::primitive::miniblock::MAX_MINIBLOCK_VALUES;
    use crate::{buffer::LanceBuffer, compression::BlockDecompressor};
    use arrow_array::Int32Array;
    // ========== Core Functionality Tests ==========

    #[test]
    fn test_basic_miniblock_rle_encoding() {
        let encoder = RleEncoder::new();

        // Test basic RLE pattern: [1, 1, 1, 2, 2, 3, 3, 3, 3]
        let array = Int32Array::from(vec![1, 1, 1, 2, 2, 3, 3, 3, 3]);
        let data_block = DataBlock::from_array(array);

        let (compressed, _) = MiniBlockCompressor::compress(&encoder, data_block).unwrap();

        assert_eq!(compressed.num_values, 9);
        assert_eq!(compressed.chunks.len(), 1);

        // Verify compression happened (3 runs instead of 9 values)
        let values_buffer = &compressed.data[0];
        let lengths_buffer = &compressed.data[1];
        assert_eq!(values_buffer.len(), 12); // 3 i32 values
        assert_eq!(lengths_buffer.len(), 3); // 3 u8 lengths
    }

    #[test]
    fn test_long_run_splitting() {
        let encoder = RleEncoder::new();

        // Create a run longer than 255 to test splitting
        let mut data = vec![42i32; 1000]; // Will be split into 255+255+255+235
        data.extend(&[100i32; 300]); // Will be split into 255+45

        let array = Int32Array::from(data);
        let (compressed, _) =
            MiniBlockCompressor::compress(&encoder, DataBlock::from_array(array)).unwrap();

        // Should have 6 runs total (4 for first value, 2 for second)
        let lengths_buffer = &compressed.data[1];
        assert_eq!(lengths_buffer.len(), 6);
    }

    #[test]
    fn test_rle_v2_u16_miniblock_encoding() {
        let encoder = RleEncoder::with_run_length_width(RunLengthWidth::U16);

        let data = vec![42i32; 1000];
        let array = Int32Array::from(data);
        let (compressed, encoding) =
            MiniBlockCompressor::compress(&encoder, DataBlock::from_array(array)).unwrap();

        assert_eq!(compressed.data[0].len(), 4);
        assert_eq!(compressed.data[1].len(), 2);
        assert_eq!(compressed.data[1].as_ref(), &1000u16.to_le_bytes());

        let rle = match encoding.compression.as_ref().unwrap() {
            crate::format::pb21::compressive_encoding::Compression::Rle(rle) => rle,
            other => panic!("expected RLE encoding, got {other:?}"),
        };
        let run_lengths = rle.run_lengths.as_ref().unwrap();
        let flat = match run_lengths.compression.as_ref().unwrap() {
            crate::format::pb21::compressive_encoding::Compression::Flat(flat) => flat,
            other => panic!("expected flat run lengths, got {other:?}"),
        };
        assert_eq!(flat.bits_per_value, 16);

        let decompressor = RleDecompressor::with_run_length_width(32, RunLengthWidth::U16);
        let decompressed = MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        )
        .unwrap();
        match decompressed {
            DataBlock::FixedWidth(block) => {
                let values = block.data.borrow_to_typed_slice::<i32>();
                assert_eq!(values.as_ref(), vec![42i32; 1000]);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    #[test]
    fn test_select_run_length_width_prefers_u16_for_long_runs() {
        let data = vec![7i32; 300];
        let bytes = data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let width =
            select_run_length_width(&LanceBuffer::from(bytes), data.len() as u64, 32).unwrap();
        assert_eq!(width, RunLengthWidth::U16);
    }

    // ========== Round-trip Tests for Different Types ==========

    #[test]
    fn test_round_trip_all_types() {
        // Test u8
        test_round_trip_helper(vec![42u8, 42, 42, 100, 100, 255, 255, 255, 255], 8);

        // Test u16
        test_round_trip_helper(vec![1000u16, 1000, 2000, 2000, 2000, 3000], 16);

        // Test i32
        test_round_trip_helper(vec![100i32, 100, 100, -200, -200, 300, 300, 300, 300], 32);

        // Test u64
        test_round_trip_helper(vec![1_000_000_000u64; 5], 64);
    }

    fn test_round_trip_helper<T>(data: Vec<T>, bits_per_value: u64)
    where
        T: bytemuck::Pod + PartialEq + std::fmt::Debug,
    {
        let encoder = RleEncoder::new();
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| bytemuck::bytes_of(v))
            .copied()
            .collect();

        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value,
            data: LanceBuffer::from(bytes),
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        let (compressed, _) = MiniBlockCompressor::compress(&encoder, block).unwrap();
        let decompressor = RleDecompressor::new(bits_per_value);
        let decompressed = MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        )
        .unwrap();

        match decompressed {
            DataBlock::FixedWidth(ref block) => {
                // Verify the decompressed data length matches expected
                assert_eq!(block.data.len(), data.len() * std::mem::size_of::<T>());
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    // ========== Chunk Boundary Tests ==========

    #[test]
    fn test_power_of_two_chunking() {
        let encoder = RleEncoder::new();

        // Create data that will require multiple chunks
        let test_sizes = vec![1000, 2500, 5000, 10000];

        for size in test_sizes {
            let data: Vec<i32> = (0..size)
                .map(|i| i / 50) // Create runs of 50
                .collect();

            let array = Int32Array::from(data);
            let (compressed, _) =
                MiniBlockCompressor::compress(&encoder, DataBlock::from_array(array)).unwrap();

            // Verify all non-last chunks have power-of-2 values
            for (i, chunk) in compressed.chunks.iter().enumerate() {
                if i < compressed.chunks.len() - 1 {
                    assert!(chunk.log_num_values > 0);
                    let chunk_values = 1u64 << chunk.log_num_values;
                    assert!(chunk_values.is_power_of_two());
                    assert!(chunk_values <= *MAX_MINIBLOCK_VALUES);
                } else {
                    assert_eq!(chunk.log_num_values, 0);
                }
            }
        }
    }

    // ========== Error Handling Tests ==========

    #[test]
    fn test_invalid_buffer_count() {
        let decompressor = RleDecompressor::new(32);
        let result = MiniBlockDecompressor::decompress(
            &decompressor,
            vec![LanceBuffer::from(vec![1, 2, 3, 4])],
            10,
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("expects exactly 2 buffers")
        );
    }

    #[test]
    fn test_buffer_consistency() {
        let decompressor = RleDecompressor::new(32);
        let values = LanceBuffer::from(vec![1, 0, 0, 0]); // 1 i32 value
        let lengths = LanceBuffer::from(vec![5, 10]); // 2 lengths - mismatch!
        let result = MiniBlockDecompressor::decompress(&decompressor, vec![values, lengths], 15);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Inconsistent RLE buffers")
        );
    }

    #[test]
    fn test_u16_length_buffer_must_be_aligned() {
        let decompressor = RleDecompressor::with_run_length_width(32, RunLengthWidth::U16);
        let values = LanceBuffer::from(vec![1, 0, 0, 0]);
        let lengths = LanceBuffer::from(vec![5]);
        let result = MiniBlockDecompressor::decompress(&decompressor, vec![values, lengths], 5);
        assert!(matches!(&result, Err(Error::InvalidInput { .. })));
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not divisible by 2")
        );
    }

    #[test]
    fn test_rle_rejects_underflow_overflow_and_zero_lengths() {
        let decompressor = RleDecompressor::with_run_length_width(32, RunLengthWidth::U16);
        let value = LanceBuffer::from(1i32.to_le_bytes().to_vec());

        let underflow = MiniBlockDecompressor::decompress(
            &decompressor,
            vec![
                value.clone(),
                LanceBuffer::from(4u16.to_le_bytes().to_vec()),
            ],
            5,
        )
        .unwrap_err();
        assert!(underflow.to_string().contains("produced 4 values"));

        let overflow = MiniBlockDecompressor::decompress(
            &decompressor,
            vec![
                value.clone(),
                LanceBuffer::from(6u16.to_le_bytes().to_vec()),
            ],
            5,
        )
        .unwrap_err();
        assert!(
            overflow
                .to_string()
                .contains("overflowed expected value count")
        );

        let zero = MiniBlockDecompressor::decompress(
            &decompressor,
            vec![value, LanceBuffer::from(0u16.to_le_bytes().to_vec())],
            5,
        )
        .unwrap_err();
        assert!(zero.to_string().contains("zero run length"));
    }

    #[test]
    fn test_empty_data_handling() {
        let encoder = RleEncoder::new();

        // Test empty block
        let empty_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::from(vec![]),
            num_values: 0,
            block_info: BlockInfo::default(),
        });

        let (compressed, _) = MiniBlockCompressor::compress(&encoder, empty_block).unwrap();
        assert_eq!(compressed.num_values, 0);
        assert!(compressed.data.is_empty());

        // Test decompression of empty data
        let decompressor = RleDecompressor::new(32);
        let decompressed = MiniBlockDecompressor::decompress(&decompressor, vec![], 0).unwrap();

        match decompressed {
            DataBlock::FixedWidth(ref block) => {
                assert_eq!(block.num_values, 0);
                assert_eq!(block.data.len(), 0);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    // ========== Integration Test ==========

    #[test]
    fn test_multi_chunk_round_trip() {
        let encoder = RleEncoder::new();

        // Create data that spans multiple chunks with mixed patterns
        let mut data = Vec::new();

        // High compression section
        data.extend(vec![999i32; 2000]);
        // Low compression section
        data.extend(0..1000);
        // Another high compression section
        data.extend(vec![777i32; 2000]);

        let array = Int32Array::from(data.clone());
        let (compressed, _) =
            MiniBlockCompressor::compress(&encoder, DataBlock::from_array(array)).unwrap();

        // Manually decompress all chunks
        let mut reconstructed = Vec::new();
        let mut values_offset = 0usize;
        let mut lengths_offset = 0usize;
        let mut values_processed = 0u64;

        // We now have exactly 2 global buffers
        assert_eq!(compressed.data.len(), 2);
        let global_values = &compressed.data[0];
        let global_lengths = &compressed.data[1];

        for chunk in &compressed.chunks {
            let chunk_values = if chunk.log_num_values > 0 {
                1u64 << chunk.log_num_values
            } else {
                compressed.num_values - values_processed
            };

            // Extract chunk buffers from global buffers using buffer_sizes
            let values_size = chunk.buffer_sizes[0] as usize;
            let lengths_size = chunk.buffer_sizes[1] as usize;

            let chunk_values_buffer = global_values.slice_with_length(values_offset, values_size);
            let chunk_lengths_buffer =
                global_lengths.slice_with_length(lengths_offset, lengths_size);

            let decompressor = RleDecompressor::new(32);
            let chunk_data = MiniBlockDecompressor::decompress(
                &decompressor,
                vec![chunk_values_buffer, chunk_lengths_buffer],
                chunk_values,
            )
            .unwrap();

            values_offset += values_size;
            lengths_offset += lengths_size;
            values_processed += chunk_values;

            match chunk_data {
                DataBlock::FixedWidth(ref block) => {
                    let values: &[i32] = bytemuck::cast_slice(block.data.as_ref());
                    reconstructed.extend_from_slice(values);
                }
                _ => panic!("Expected FixedWidth block"),
            }
        }

        assert_eq!(reconstructed, data);
    }

    #[test]
    fn test_1024_boundary_conditions() {
        // Comprehensive test for various boundary conditions at 1024 values
        // This consolidates multiple bug tests that were previously separate
        let encoder = RleEncoder::new();
        let decompressor = RleDecompressor::new(32);

        let test_cases = [
            ("runs_of_2", {
                let mut data = Vec::new();
                for i in 0..512 {
                    data.push(i);
                    data.push(i);
                }
                data
            }),
            ("single_run_1024", vec![42i32; 1024]),
            ("alternating_values", {
                let mut data = Vec::new();
                for i in 0..1024 {
                    data.push(i % 2);
                }
                data
            }),
            ("run_boundary_255s", {
                let mut data = Vec::new();
                data.extend(vec![1i32; 255]);
                data.extend(vec![2i32; 255]);
                data.extend(vec![3i32; 255]);
                data.extend(vec![4i32; 255]);
                data.extend(vec![5i32; 4]);
                data
            }),
            ("unique_values_1024", (0..1024).collect::<Vec<_>>()),
            ("unique_plus_duplicate", {
                // 1023 unique values followed by one duplicate (regression test)
                let mut data = Vec::new();
                for i in 0..1023 {
                    data.push(i);
                }
                data.push(1022i32); // Last value same as second-to-last
                data
            }),
            ("bug_4092_pattern", {
                // Test exact scenario that produces 4092 bytes instead of 4096
                let mut data = Vec::new();
                for i in 0..1022 {
                    data.push(i);
                }
                data.push(999999i32);
                data.push(999999i32);
                data
            }),
        ];

        for (test_name, data) in test_cases.iter() {
            assert_eq!(data.len(), 1024, "Test case {} has wrong length", test_name);

            // Compress the data
            let array = Int32Array::from(data.clone());
            let (compressed, _) =
                MiniBlockCompressor::compress(&encoder, DataBlock::from_array(array)).unwrap();

            // Decompress and verify
            match MiniBlockDecompressor::decompress(
                &decompressor,
                compressed.data,
                compressed.num_values,
            ) {
                Ok(decompressed) => match decompressed {
                    DataBlock::FixedWidth(ref block) => {
                        let values: &[i32] = bytemuck::cast_slice(block.data.as_ref());
                        assert_eq!(
                            values.len(),
                            1024,
                            "Test case {} got {} values, expected 1024",
                            test_name,
                            values.len()
                        );
                        assert_eq!(
                            block.data.len(),
                            4096,
                            "Test case {} got {} bytes, expected 4096",
                            test_name,
                            block.data.len()
                        );
                        assert_eq!(values, &data[..], "Test case {} data mismatch", test_name);
                    }
                    _ => panic!("Test case {} expected FixedWidth block", test_name),
                },
                Err(e) => {
                    if e.to_string().contains("4092") {
                        panic!("Test case {} found bug 4092: {}", test_name, e);
                    }
                    panic!("Test case {} failed with error: {}", test_name, e);
                }
            }
        }
    }

    #[test]
    fn test_low_repetition_50pct_bug() {
        // Test case that reproduces the 4092 bytes bug with low repetition (50%)
        // This simulates the 1M benchmark case
        let encoder = RleEncoder::new();

        // Create 1M values with low repetition (50% chance of change)
        let num_values = 1_048_576; // 1M values
        let mut data = Vec::with_capacity(num_values);
        let mut value = 0i32;
        let mut rng = 12345u64; // Simple deterministic RNG

        for _ in 0..num_values {
            data.push(value);
            // Simple LCG for deterministic randomness
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            // 50% chance to increment value
            if (rng >> 16) & 1 == 1 {
                value += 1;
            }
        }

        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::from(bytes),
            num_values: num_values as u64,
            block_info: BlockInfo::default(),
        });

        let (compressed, _) = MiniBlockCompressor::compress(&encoder, block).unwrap();

        // Debug first few chunks
        for (i, chunk) in compressed.chunks.iter().take(5).enumerate() {
            let _chunk_values = if chunk.log_num_values > 0 {
                1 << chunk.log_num_values
            } else {
                // Last chunk - calculate remaining
                let prev_total: usize = compressed.chunks[..i]
                    .iter()
                    .map(|c| 1usize << c.log_num_values)
                    .sum();
                num_values - prev_total
            };
        }

        // Try to decompress
        let decompressor = RleDecompressor::new(32);
        match MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        ) {
            Ok(decompressed) => match decompressed {
                DataBlock::FixedWidth(ref block) => {
                    assert_eq!(
                        block.data.len(),
                        num_values * 4,
                        "Expected {} bytes but got {}",
                        num_values * 4,
                        block.data.len()
                    );
                }
                _ => panic!("Expected FixedWidth block"),
            },
            Err(e) => {
                if e.to_string().contains("4092") {
                    panic!("Bug reproduced! {}", e);
                } else {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }

    // ========== Encoding Verification Tests ==========

    #[test_log::test(tokio::test)]
    async fn test_rle_encoding_verification() {
        use crate::testing::{TestCases, check_round_trip_encoding_of_data};
        use crate::version::LanceFileVersion;
        use arrow_array::{Array, Int32Array};
        use lance_datagen::{ArrayGenerator, RowCount};
        use std::collections::HashMap;
        use std::sync::Arc;

        let test_cases = TestCases::default()
            .with_expected_encoding("rle")
            .with_min_file_version(LanceFileVersion::V2_1);

        // Test both explicit metadata and automatic selection
        // 1. Test with explicit RLE threshold metadata (also disable BSS)
        let mut metadata_explicit = HashMap::new();
        metadata_explicit.insert(
            "lance-encoding:rle-threshold".to_string(),
            "0.8".to_string(),
        );
        metadata_explicit.insert("lance-encoding:bss".to_string(), "off".to_string());

        let mut generator = RleDataGenerator::new(vec![
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN + 1,
            i32::MIN + 1,
            i32::MIN + 1,
            i32::MIN + 1,
            i32::MIN + 2,
            i32::MIN + 2,
            i32::MIN + 2,
            i32::MIN + 2,
        ]);
        let data_explicit = generator.generate_default(RowCount::from(10000)).unwrap();
        check_round_trip_encoding_of_data(vec![data_explicit], &test_cases, metadata_explicit)
            .await;

        // 2. Test automatic RLE selection based on data characteristics
        // 80% repetition should trigger RLE (> default 50% threshold).
        //
        // Use values with the high bit set so bitpacking can't shrink the values.
        // Explicitly disable BSS to ensure RLE is tested
        let mut metadata = HashMap::new();
        metadata.insert("lance-encoding:bss".to_string(), "off".to_string());

        let mut values = vec![i32::MIN; 8000]; // 80% repetition
        values.extend(
            [
                i32::MIN + 1,
                i32::MIN + 2,
                i32::MIN + 3,
                i32::MIN + 4,
                i32::MIN + 5,
            ]
            .repeat(400),
        ); // 20% variety
        let arr = Arc::new(Int32Array::from(values)) as Arc<dyn Array>;
        check_round_trip_encoding_of_data(vec![arr], &test_cases, metadata).await;
    }

    /// Generator that produces repetitive patterns suitable for RLE
    #[derive(Debug)]
    struct RleDataGenerator {
        pattern: Vec<i32>,
        idx: usize,
    }

    impl RleDataGenerator {
        fn new(pattern: Vec<i32>) -> Self {
            Self { pattern, idx: 0 }
        }
    }

    impl lance_datagen::ArrayGenerator for RleDataGenerator {
        fn generate(
            &mut self,
            _length: lance_datagen::RowCount,
            _rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
        ) -> std::result::Result<std::sync::Arc<dyn arrow_array::Array>, arrow_schema::ArrowError>
        {
            use arrow_array::Int32Array;
            use std::sync::Arc;

            // Generate enough repetitive data to trigger RLE
            let mut values = Vec::new();
            for _ in 0..10000 {
                values.push(self.pattern[self.idx]);
                self.idx = (self.idx + 1) % self.pattern.len();
            }
            Ok(Arc::new(Int32Array::from(values)))
        }

        fn data_type(&self) -> &arrow_schema::DataType {
            &arrow_schema::DataType::Int32
        }

        fn element_size_bytes(&self) -> Option<lance_datagen::ByteCount> {
            Some(lance_datagen::ByteCount::from(4))
        }
    }

    // ========== Block Related tests ==========
    #[test]
    fn test_block_decompressor_rejects_overflowing_values_size() {
        let decompressor = RleDecompressor::new(32);

        let mut data = Vec::new();
        data.extend_from_slice(&u64::MAX.to_le_bytes());
        let result = BlockDecompressor::decompress(&decompressor, LanceBuffer::from(data), 1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid RLE values buffer size")
        );
    }

    #[test]
    fn test_block_decompressor_too_small() {
        let decompressor = RleDecompressor::new(32);
        let result =
            BlockDecompressor::decompress(&decompressor, LanceBuffer::from(vec![1, 2, 3]), 10);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Insufficient data size: 3")
        );
    }

    #[test]
    fn test_block_compressor_header_format() {
        let encoder = RleEncoder::new();

        let data = vec![1i32, 1, 1];
        let array = Int32Array::from(data);
        let compressed = BlockCompressor::compress(&encoder, DataBlock::from_array(array)).unwrap();

        // Verify header format: first 8 bytes should be values_size as u64
        assert!(compressed.len() >= 8);
        let values_size_bytes: [u8; 8] = compressed.as_ref()[..8].try_into().unwrap();
        let values_size = u64::from_le_bytes(values_size_bytes);

        // Values buffer should contain 1 i32 value (4 bytes)
        assert_eq!(values_size, 4);

        // Total size should be: 8 (header) + 4 (values) + 1 (lengths)
        assert_eq!(compressed.len(), 13);
    }

    #[test]
    fn test_block_compressor_round_trip() {
        let encoder = RleEncoder::new();
        let decompressor = RleDecompressor::new(32);

        // Test basic pattern
        let data = vec![1i32, 1, 1, 2, 2, 3, 3, 3, 3];
        let array = Int32Array::from(data.clone());
        let data_block = DataBlock::from_array(array);

        let compressed = BlockCompressor::compress(&encoder, data_block).unwrap();
        let decompressed =
            BlockDecompressor::decompress(&decompressor, compressed, data.len() as u64).unwrap();

        match decompressed {
            DataBlock::FixedWidth(block) => {
                let values: &[i32] = bytemuck::cast_slice(block.data.as_ref());
                assert_eq!(values, &data[..]);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    #[test]
    fn test_block_compressor_large_data() {
        let encoder = RleEncoder::new();
        let decompressor = RleDecompressor::new(32);

        // Create data that will span multiple chunks
        // Each chunks can handle ~2048 values, so use 10K values
        let mut data = Vec::new();
        data.extend(vec![999i32; 3000]); // First ~2 chunks
        data.extend(vec![777i32; 3000]); // Next ~2 chunks
        data.extend(vec![555i32; 4000]); // Final ~2 chunks

        let total_values = data.len();
        assert_eq!(total_values, 10000);

        let array = Int32Array::from(data.clone());
        let compressed = BlockCompressor::compress(&encoder, DataBlock::from_array(array)).unwrap();
        let decompressed =
            BlockDecompressor::decompress(&decompressor, compressed, total_values as u64).unwrap();

        match decompressed {
            DataBlock::FixedWidth(block) => {
                let values: &[i32] = bytemuck::cast_slice(block.data.as_ref());
                assert_eq!(values.len(), total_values);
                assert_eq!(values, &data[..]);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }
}
