// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # RLE (Run-Length Encoding) Miniblock Format
//!
//! RLE compression for Lance miniblock format, optimized for data with repeated values.
//!
//! ## Encoding Format
//!
//! RLE uses a dual-buffer format to store compressed data:
//!
//! - **Values Buffer**: Stores unique values in their original data type
//! - **Lengths Buffer**: Stores the repeat count for each value as u8
//!
//! ### Example
//!
//! Input data: `[1, 1, 1, 2, 2, 3, 3, 3, 3]`
//!
//! Encoded as:
//! - Values buffer: `[1, 2, 3]` (3 × 4 bytes for i32)
//! - Lengths buffer: `[3, 2, 4]` (3 × 1 byte for u8)
//!
//! ### Long Run Handling
//!
//! When a run exceeds 255 values, it is split into multiple runs of 255
//! followed by a final run with the remainder. For example, a run of 1000
//! identical values becomes 4 runs: [255, 255, 255, 235].
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
//! ## Chunk Handling
//!
//! - Maximum chunk size: 4096 values (miniblock constraint)
//! - Each chunk is independently encoded with its own values/lengths buffers
//! - Non-last chunks always contain power-of-2 values
//! - Byte limits are enforced dynamically during encoding

use log::trace;
use snafu::location;

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::data::DataBlock;
use crate::data::{BlockInfo, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor, MAX_MINIBLOCK_BYTES,
    MAX_MINIBLOCK_VALUES,
};
use crate::format::{pb, ProtobufUtils};

use lance_core::{Error, Result};

/// RLE encoder for miniblock format
#[derive(Debug, Default)]
pub struct RleMiniBlockEncoder;

impl RleMiniBlockEncoder {
    pub fn new() -> Self {
        Self
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

        let bytes_per_value = (bits_per_value / 8) as usize;

        let mut all_buffers = Vec::new();
        let mut chunks = Vec::new();

        let mut offset = 0usize;
        let mut values_remaining = num_values as usize;

        while values_remaining > 0 {
            let (values_bytes, lengths_bytes, num_runs, values_processed, is_last_chunk) =
                match bits_per_value {
                    8 => self.encode_chunk_rolling::<u8>(data, offset, values_remaining),
                    16 => self.encode_chunk_rolling::<u16>(data, offset, values_remaining),
                    32 => self.encode_chunk_rolling::<u32>(data, offset, values_remaining),
                    64 => self.encode_chunk_rolling::<u64>(data, offset, values_remaining),
                    _ => unreachable!("RLE encoding bits_per_value must be 8, 16, 32, 64, or 128"),
                };

            if values_processed == 0 {
                break;
            }

            let values_buffer = LanceBuffer::Owned(values_bytes);
            let lengths_buffer = LanceBuffer::Owned(lengths_bytes);

            let log_num_values = if is_last_chunk {
                0
            } else {
                assert!(
                    values_processed.is_power_of_two(),
                    "Non-last chunk must have power-of-2 values"
                );
                values_processed.ilog2() as u8
            };

            let chunk = MiniBlockChunk {
                buffer_sizes: vec![(num_runs * bytes_per_value) as u16, num_runs as u16],
                log_num_values,
            };

            all_buffers.push(values_buffer);
            all_buffers.push(lengths_buffer);
            chunks.push(chunk);

            offset += values_processed;
            values_remaining -= values_processed;
        }

        Ok((all_buffers, chunks))
    }

    /// Encodes a chunk of data using RLE compression with dynamic boundary detection.
    ///
    /// This function processes values sequentially, detecting runs (sequences of identical values)
    /// and encoding them as (value, length) pairs. It dynamically determines whether this chunk
    /// should be the last chunk based on how many values were processed.
    ///
    /// # Key Features:
    /// - Tracks byte usage to ensure we don't exceed MAX_MINIBLOCK_BYTES
    /// - Maintains power-of-2 checkpoints for non-last chunks
    /// - Splits long runs (>255) into multiple entries
    /// - Dynamically determines if this is the last chunk
    ///
    /// # Returns:
    /// - values_bytes: Buffer containing unique run values
    /// - lengths_bytes: Buffer containing run lengths (u8)
    /// - num_runs: Number of runs encoded
    /// - values_processed: Number of input values processed
    /// - is_last_chunk: Whether this chunk processed all remaining values
    fn encode_chunk_rolling<T>(
        &self,
        data: &LanceBuffer,
        offset: usize,
        values_remaining: usize,
    ) -> (Vec<u8>, Vec<u8>, usize, usize, bool)
    where
        T: bytemuck::Pod + PartialEq + Copy + std::fmt::Debug,
    {
        let type_size = std::mem::size_of::<T>();
        let data_slice = data.as_ref();

        let chunk_start = offset * type_size;
        let max_by_count = MAX_MINIBLOCK_VALUES as usize;
        let max_values = values_remaining.min(max_by_count);
        let chunk_end = chunk_start + max_values * type_size;

        if chunk_start >= data_slice.len() {
            return (Vec::new(), Vec::new(), 0, 0, false);
        }

        let chunk_data = &data_slice[chunk_start..chunk_end.min(data_slice.len())];
        let typed_data: &[T] = bytemuck::cast_slice(chunk_data);

        if typed_data.is_empty() {
            return (Vec::new(), Vec::new(), 0, 0, false);
        }

        let estimated_runs = (max_values / 10).max(1).min(max_values);
        let mut values = Vec::with_capacity(estimated_runs * type_size);
        let mut lengths = Vec::with_capacity(estimated_runs);

        let mut current_value = typed_data[0];
        let mut current_length = 1u64;
        let mut values_processed = 1usize;
        let mut bytes_used = 0usize;

        // Power-of-2 checkpoints for ensuring non-last chunks have valid sizes
        // For smaller data types like u8, we can use larger initial checkpoints
        // since they take less space per value
        let checkpoints = match type_size {
            1 => vec![256, 512, 1024, 2048, 4096], // u8 can start from 256
            2 => vec![128, 256, 512, 1024, 2048, 4096], // u16 can start from 128
            _ => vec![64, 128, 256, 512, 1024, 2048, 4096], // u32/u64: no difference
        };
        let valid_checkpoints: Vec<usize> = checkpoints
            .into_iter()
            .filter(|&p| p <= values_remaining)
            .collect();
        let mut checkpoint_idx = 0;

        // Save state at checkpoints so we can roll back if needed
        let mut last_checkpoint_state = None;

        for &value in typed_data[1..].iter() {
            if value == current_value {
                current_length += 1;
                values_processed += 1;
            } else {
                // Calculate space needed (may need multiple u8s if run > 255)
                let run_chunks = current_length.div_ceil(255) as usize;
                let bytes_needed = run_chunks * (type_size + 1);

                // Stop if adding this run would exceed byte limit
                if bytes_used + bytes_needed > MAX_MINIBLOCK_BYTES as usize {
                    if let Some((val_len, len_len, _, checkpoint_values)) = last_checkpoint_state {
                        // Roll back to last power-of-2 checkpoint
                        values.truncate(val_len);
                        lengths.truncate(len_len);
                        return (
                            values,
                            lengths,
                            val_len / type_size,
                            checkpoint_values,
                            false,
                        );
                    }
                    values_processed -= current_length as usize;
                    break;
                }

                bytes_used +=
                    self.add_run(&current_value, current_length, &mut values, &mut lengths);
                current_value = value;
                current_length = 1;
                values_processed += 1;
            }

            if checkpoint_idx < valid_checkpoints.len()
                && values_processed == valid_checkpoints[checkpoint_idx]
            {
                last_checkpoint_state =
                    Some((values.len(), lengths.len(), bytes_used, values_processed));
                checkpoint_idx += 1;
            }
        }

        if values_processed == typed_data.len() {
            let run_chunks = current_length.div_ceil(255) as usize;
            let bytes_needed = run_chunks * (type_size + 1);

            if bytes_used + bytes_needed <= MAX_MINIBLOCK_BYTES as usize {
                let _ = self.add_run(&current_value, current_length, &mut values, &mut lengths);
            } else {
                values_processed -= current_length as usize;
            }
        }

        // Determine if we've processed all remaining values
        let is_last_chunk = values_processed == values_remaining;

        // Non-last chunks must have power-of-2 values for miniblock format
        if !is_last_chunk {
            if values_processed.is_power_of_two() {
                // Already at power-of-2 boundary
            } else if let Some((val_len, len_len, _, checkpoint_values)) = last_checkpoint_state {
                // Roll back to last valid checkpoint
                values.truncate(val_len);
                lengths.truncate(len_len);
                let num_runs = val_len / type_size;
                return (values, lengths, num_runs, checkpoint_values, false);
            } else {
                // No valid checkpoint, can't create a valid chunk
                return (Vec::new(), Vec::new(), 0, 0, false);
            }
        }

        let num_runs = values.len() / type_size;
        (values, lengths, num_runs, values_processed, is_last_chunk)
    }

    fn add_run<T>(
        &self,
        value: &T,
        length: u64,
        values: &mut Vec<u8>,
        lengths: &mut Vec<u8>,
    ) -> usize
    where
        T: bytemuck::Pod,
    {
        let value_bytes = bytemuck::bytes_of(value);
        let type_size = std::mem::size_of::<T>();
        let num_full_chunks = (length / 255) as usize;
        let remainder = (length % 255) as u8;

        let total_chunks = num_full_chunks + if remainder > 0 { 1 } else { 0 };
        values.reserve(total_chunks * type_size);
        lengths.reserve(total_chunks);

        for _ in 0..num_full_chunks {
            values.extend_from_slice(value_bytes);
            lengths.push(255);
        }

        if remainder > 0 {
            values.extend_from_slice(value_bytes);
            lengths.push(remainder);
        }

        total_chunks * (type_size + 1)
    }
}

impl MiniBlockCompressor for RleMiniBlockEncoder {
    fn compress(&self, data: DataBlock) -> Result<(MiniBlockCompressed, pb::ArrayEncoding)> {
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

                let encoding = ProtobufUtils::rle(bits_per_value);

                Ok((compressed, encoding))
            }
            _ => Err(Error::InvalidInput {
                location: location!(),
                source: "RLE encoding only supports FixedWidth data blocks".into(),
            }),
        }
    }
}

/// RLE decompressor for miniblock format
#[derive(Debug)]
pub struct RleMiniBlockDecompressor {
    bits_per_value: u64,
}

impl RleMiniBlockDecompressor {
    pub fn new(bits_per_value: u64) -> Self {
        Self { bits_per_value }
    }

    fn decode_data(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                bits_per_value: self.bits_per_value,
                data: LanceBuffer::Owned(vec![]),
                num_values: 0,
                block_info: BlockInfo::default(),
            }));
        }

        assert_eq!(
            data.len(),
            2,
            "RLE decompressor expects exactly 2 buffers, got {}",
            data.len()
        );

        let values_buffer = &data[0];
        let lengths_buffer = &data[1];

        let decoded_data = match self.bits_per_value {
            8 => self.decode_generic::<u8>(values_buffer, lengths_buffer, num_values)?,
            16 => self.decode_generic::<u16>(values_buffer, lengths_buffer, num_values)?,
            32 => self.decode_generic::<u32>(values_buffer, lengths_buffer, num_values)?,
            64 => self.decode_generic::<u64>(values_buffer, lengths_buffer, num_values)?,
            _ => unreachable!("RLE decoding bits_per_value must be 8, 16, 32, 64, or 128"),
        };

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_value,
            data: LanceBuffer::Owned(decoded_data),
            num_values,
            block_info: BlockInfo::default(),
        }))
    }

    fn decode_generic<T>(
        &self,
        values_buffer: &LanceBuffer,
        lengths_buffer: &LanceBuffer,
        num_values: u64,
    ) -> Result<Vec<u8>>
    where
        T: bytemuck::Pod + Copy + std::fmt::Debug,
    {
        let values_bytes = values_buffer.as_ref();
        let lengths_bytes = lengths_buffer.as_ref();

        let type_size = std::mem::size_of::<T>();

        if values_bytes.is_empty() || lengths_bytes.is_empty() {
            if num_values == 0 {
                return Ok(Vec::new());
            } else {
                return Err(Error::InvalidInput {
                    location: location!(),
                    source: format!("Empty buffers but expected {} values", num_values).into(),
                });
            }
        }

        if values_bytes.len() % type_size != 0 || lengths_bytes.is_empty() {
            return Err(Error::InvalidInput {
                location: location!(),
                source: format!(
                    "Invalid buffer sizes for RLE {} decoding: values {} bytes (not divisible by {}), lengths {} bytes",
                    std::any::type_name::<T>(),
                    values_bytes.len(),
                    type_size,
                    lengths_bytes.len()
                )
                .into(),
            });
        }

        let num_runs = values_bytes.len() / type_size;
        let num_length_entries = lengths_bytes.len();
        assert_eq!(
            num_runs, num_length_entries,
            "Inconsistent RLE buffers: {} runs but {} length entries",
            num_runs, num_length_entries
        );

        let values: &[T] = bytemuck::cast_slice(values_bytes);
        let lengths: &[u8] = lengths_bytes;

        let expected_byte_count = num_values as usize * type_size;
        let mut decoded = Vec::with_capacity(expected_byte_count);

        for (value, &length) in values.iter().zip(lengths.iter()) {
            let run_length = length as usize;
            let bytes_to_write = run_length * type_size;

            if decoded.len() + bytes_to_write > expected_byte_count {
                let remaining_bytes = expected_byte_count - decoded.len();
                let remaining_values = remaining_bytes / type_size;

                for _ in 0..remaining_values {
                    decoded.extend_from_slice(bytemuck::bytes_of(value));
                }
                break;
            }

            for _ in 0..run_length {
                decoded.extend_from_slice(bytemuck::bytes_of(value));
            }
        }

        if decoded.len() != expected_byte_count {
            return Err(Error::InvalidInput {
                location: location!(),
                source: format!(
                    "RLE decoding produced {} bytes, expected {}",
                    decoded.len(),
                    expected_byte_count
                )
                .into(),
            });
        }

        trace!(
            "RLE decoded {} {} values",
            num_values,
            std::any::type_name::<T>()
        );
        Ok(decoded)
    }
}

impl MiniBlockDecompressor for RleMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        self.decode_data(data, num_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::{CompressionStrategy, DefaultCompressionStrategy};
    use crate::data::DataBlock;
    use arrow_array::Int32Array;
    use lance_core::datatypes::Field;

    // ========== Core Functionality Tests ==========

    #[test]
    fn test_basic_rle_encoding() {
        let encoder = RleMiniBlockEncoder::new();

        // Test basic RLE pattern: [1, 1, 1, 2, 2, 3, 3, 3, 3]
        let array = Int32Array::from(vec![1, 1, 1, 2, 2, 3, 3, 3, 3]);
        let data_block = DataBlock::from_array(array);

        let (compressed, _) = encoder.compress(data_block).unwrap();

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
        let encoder = RleMiniBlockEncoder::new();

        // Create a run longer than 255 to test splitting
        let mut data = vec![42i32; 1000]; // Will be split into 255+255+255+235
        data.extend(&[100i32; 300]); // Will be split into 255+45

        let array = Int32Array::from(data);
        let (compressed, _) = encoder.compress(DataBlock::from_array(array)).unwrap();

        // Should have 6 runs total (4 for first value, 2 for second)
        let lengths_buffer = &compressed.data[1];
        assert_eq!(lengths_buffer.len(), 6);
    }

    #[test]
    fn test_compression_strategy_selection() {
        let strategy = DefaultCompressionStrategy;
        let field = Field::new_arrow("test", arrow_schema::DataType::Int32, false).unwrap();

        // High repetition - should select RLE
        let repetitive_array = Int32Array::from(vec![1; 1000]);
        let repetitive_block = DataBlock::from_array(repetitive_array);

        let compressor = strategy
            .create_miniblock_compressor(&field, &repetitive_block)
            .unwrap();
        assert!(format!("{:?}", compressor).contains("RleMiniBlockEncoder"));

        // No repetition - should NOT select RLE
        let unique_array = Int32Array::from((0..1000).collect::<Vec<i32>>());
        let unique_block = DataBlock::from_array(unique_array);

        let compressor = strategy
            .create_miniblock_compressor(&field, &unique_block)
            .unwrap();
        assert!(!format!("{:?}", compressor).contains("RleMiniBlockEncoder"));
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
        let encoder = RleMiniBlockEncoder::new();
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| bytemuck::bytes_of(v))
            .copied()
            .collect();

        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value,
            data: LanceBuffer::Owned(bytes.clone()),
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        let (compressed, _) = encoder.compress(block).unwrap();
        let decompressor = RleMiniBlockDecompressor::new(bits_per_value);
        let decompressed = decompressor
            .decompress(compressed.data, compressed.num_values)
            .unwrap();

        match decompressed {
            DataBlock::FixedWidth(ref block) => {
                assert_eq!(block.data.as_ref(), bytes);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    // ========== Chunk Boundary Tests ==========

    #[test]
    fn test_power_of_two_chunking() {
        let encoder = RleMiniBlockEncoder::new();

        // Create data that will require multiple chunks
        let test_sizes = vec![1000, 2500, 5000, 10000];

        for size in test_sizes {
            let data: Vec<i32> = (0..size)
                .map(|i| i / 50) // Create runs of 50
                .collect();

            let array = Int32Array::from(data);
            let (compressed, _) = encoder.compress(DataBlock::from_array(array)).unwrap();

            // Verify all non-last chunks have power-of-2 values
            for (i, chunk) in compressed.chunks.iter().enumerate() {
                if i < compressed.chunks.len() - 1 {
                    assert!(chunk.log_num_values > 0);
                    let chunk_values = 1u64 << chunk.log_num_values;
                    assert!(chunk_values.is_power_of_two());
                    assert!(chunk_values <= MAX_MINIBLOCK_VALUES);
                } else {
                    assert_eq!(chunk.log_num_values, 0);
                }
            }
        }
    }

    #[test]
    fn test_byte_limit_enforcement() {
        let encoder = RleMiniBlockEncoder::new();

        // Test with 128-bit data which has tighter byte constraints
        let mut data_128 = Vec::new();
        for i in 0..600u128 {
            data_128.extend(&[i, i, i]); // 600 runs * 3 values each
        }

        let bytes_128: Vec<u8> = data_128
            .iter()
            .flat_map(|v: &u128| v.to_le_bytes())
            .collect();

        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 128,
            data: LanceBuffer::Owned(bytes_128),
            num_values: 1800,
            block_info: BlockInfo::default(),
        });

        let (compressed, _) = encoder.compress(block).unwrap();

        // Verify no chunk exceeds byte limit
        for chunk in &compressed.chunks {
            let total_bytes: usize = chunk.buffer_sizes.iter().map(|&s| s as usize).sum();
            assert!(total_bytes <= MAX_MINIBLOCK_BYTES as usize);
        }
    }

    // ========== Error Handling Tests ==========

    #[test]
    #[should_panic(expected = "RLE decompressor expects exactly 2 buffers")]
    fn test_invalid_buffer_count() {
        let decompressor = RleMiniBlockDecompressor::new(32);
        let _ = decompressor.decompress(vec![LanceBuffer::Owned(vec![1, 2, 3, 4])], 10);
    }

    #[test]
    #[should_panic(expected = "Inconsistent RLE buffers")]
    fn test_buffer_consistency() {
        let decompressor = RleMiniBlockDecompressor::new(32);
        let values = LanceBuffer::Owned(vec![1, 0, 0, 0]); // 1 i32 value
        let lengths = LanceBuffer::Owned(vec![5, 10]); // 2 lengths - mismatch!
        let _ = decompressor.decompress(vec![values, lengths], 15);
    }

    #[test]
    fn test_empty_data_handling() {
        let encoder = RleMiniBlockEncoder::new();

        // Test empty block
        let empty_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::Owned(vec![]),
            num_values: 0,
            block_info: BlockInfo::default(),
        });

        let (compressed, _) = encoder.compress(empty_block).unwrap();
        assert_eq!(compressed.num_values, 0);
        assert!(compressed.data.is_empty());

        // Test decompression of empty data
        let decompressor = RleMiniBlockDecompressor::new(32);
        let decompressed = decompressor.decompress(vec![], 0).unwrap();

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
        let encoder = RleMiniBlockEncoder::new();

        // Create data that spans multiple chunks with mixed patterns
        let mut data = Vec::new();

        // High compression section
        data.extend(vec![999i32; 2000]);
        // Low compression section
        data.extend(0..1000);
        // Another high compression section
        data.extend(vec![777i32; 2000]);

        let array = Int32Array::from(data.clone());
        let (compressed, _) = encoder.compress(DataBlock::from_array(array)).unwrap();

        // Manually decompress all chunks
        let mut reconstructed = Vec::new();
        let mut buffer_idx = 0;
        let mut values_processed = 0u64;

        for chunk in &compressed.chunks {
            let chunk_values = if chunk.log_num_values > 0 {
                1u64 << chunk.log_num_values
            } else {
                compressed.num_values - values_processed
            };

            let decompressor = RleMiniBlockDecompressor::new(32);
            let chunk_data = decompressor
                .decompress(
                    vec![
                        compressed.data[buffer_idx].deep_copy(),
                        compressed.data[buffer_idx + 1].deep_copy(),
                    ],
                    chunk_values,
                )
                .unwrap();

            buffer_idx += 2;
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
}
