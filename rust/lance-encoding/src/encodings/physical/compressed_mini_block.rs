// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use log::trace;
use snafu::location;

use crate::{
    buffer::LanceBuffer,
    compression::MiniBlockDecompressor,
    data::DataBlock,
    encodings::{
        logical::primitive::miniblock::{MiniBlockCompressed, MiniBlockCompressor},
        physical::block::{CompressionConfig, GeneralBufferCompressor},
    },
    format::{pb, ProtobufUtils},
    Result,
};

/// A miniblock compressor that wraps another miniblock compressor and applies
/// general-purpose compression (LZ4, Zstd) to the resulting buffers.
#[derive(Debug)]
pub struct CompressedMiniBlockCompressor {
    inner: Box<dyn MiniBlockCompressor>,
    compression: CompressionConfig,
}

impl CompressedMiniBlockCompressor {
    pub fn new(inner: Box<dyn MiniBlockCompressor>, compression: CompressionConfig) -> Self {
        Self { inner, compression }
    }
}

/// Minimum buffer size to consider for compression
const MIN_BUFFER_SIZE_FOR_COMPRESSION: usize = 256;

use super::super::logical::primitive::miniblock::MiniBlockChunk;

/// Encode chunks into a simple binary format
/// Format: [chunk_count(u32)][chunk1][chunk2]...
/// Each chunk: [buffer_count(u16)][buffer_sizes...][log_num_values(u8)]
fn encode_chunks(chunks: &[MiniBlockChunk]) -> Vec<u8> {
    let mut result = Vec::new();

    // Write chunk count
    result.extend_from_slice(&(chunks.len() as u32).to_le_bytes());

    // Write each chunk
    for chunk in chunks {
        // Buffer count
        result.extend_from_slice(&(chunk.buffer_sizes.len() as u16).to_le_bytes());
        // Each buffer size
        for &size in &chunk.buffer_sizes {
            result.extend_from_slice(&size.to_le_bytes());
        }
        // log_num_values
        result.push(chunk.log_num_values);
    }

    result
}

/// Decode chunks from binary format
fn decode_chunks(data: &[u8]) -> Result<Vec<MiniBlockChunk>> {
    use lance_core::Error;

    let mut offset = 0;

    // Read chunk count
    if data.len() < 4 {
        return Err(Error::io(
            "Invalid chunk encoding: insufficient data for chunk count".to_string(),
            location!(),
        ));
    }
    let chunk_count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    offset += 4;

    let mut chunks = Vec::with_capacity(chunk_count);

    // Read each chunk
    for _ in 0..chunk_count {
        // Read buffer count
        if offset + 2 > data.len() {
            return Err(Error::io(
                "Invalid chunk encoding: insufficient data for buffer count".to_string(),
                location!(),
            ));
        }
        let buffer_count = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        // Read buffer sizes
        let mut buffer_sizes = Vec::with_capacity(buffer_count);
        for _ in 0..buffer_count {
            if offset + 2 > data.len() {
                return Err(Error::io(
                    "Invalid chunk encoding: insufficient data for buffer size".to_string(),
                    location!(),
                ));
            }
            let size = u16::from_le_bytes([data[offset], data[offset + 1]]);
            buffer_sizes.push(size);
            offset += 2;
        }

        // Read log_num_values
        if offset >= data.len() {
            return Err(Error::io(
                "Invalid chunk encoding: insufficient data for log_num_values".to_string(),
                location!(),
            ));
        }
        let log_num_values = data[offset];
        offset += 1;

        chunks.push(MiniBlockChunk {
            buffer_sizes,
            log_num_values,
        });
    }

    Ok(chunks)
}

impl MiniBlockCompressor for CompressedMiniBlockCompressor {
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, pb::ArrayEncoding)> {
        // First, compress with the inner compressor
        let (inner_compressed, inner_encoding) = self.inner.compress(page)?;

        // Return the original encoding without compression if there's no data or
        // the first buffer is not large enough
        if inner_compressed.data.is_empty()
            || inner_compressed.data[0].len() < MIN_BUFFER_SIZE_FOR_COMPRESSION
        {
            return Ok((inner_compressed, inner_encoding));
        }

        // Combine all buffers into one for compression
        let mut combined_data = Vec::new();
        for buffer in &inner_compressed.data {
            combined_data.extend_from_slice(buffer.as_ref());
        }

        let original_size = combined_data.len();

        // Compress the combined data
        let compressor = GeneralBufferCompressor::get_compressor(self.compression);
        let mut compressed_data = Vec::new();
        compressor.compress(&combined_data, &mut compressed_data)?;

        let compressed_size = compressed_data.len();

        // If compression doesn't help, return the original
        if original_size <= compressed_size {
            return Ok((inner_compressed, inner_encoding));
        }

        trace!(
            "Combined buffers compressed from {} to {} bytes (ratio: {:.2})",
            original_size,
            compressed_size,
            compressed_size as f32 / original_size as f32
        );

        // Encode the original chunks structure
        let encoded_chunks = encode_chunks(&inner_compressed.chunks);
        let encoded_chunks_size = encoded_chunks.len() as u16;

        // Create the result with two buffers:
        // Buffer 0: Compressed data
        // Buffer 1: Encoded chunks structure
        let compressed_miniblock = MiniBlockCompressed {
            data: vec![
                LanceBuffer::from(compressed_data),
                LanceBuffer::from(encoded_chunks),
            ],
            chunks: vec![MiniBlockChunk {
                buffer_sizes: vec![compressed_size as u16, encoded_chunks_size],
                log_num_values: 0, // 0 indicates this is the final/only chunk
            }],
            num_values: inner_compressed.num_values,
        };

        // Return compressed encoding
        let encoding = ProtobufUtils::compressed_mini_block(inner_encoding, self.compression);
        Ok((compressed_miniblock, encoding))
    }
}

/// A miniblock decompressor that first decompresses buffers using general-purpose
/// compression (LZ4, Zstd) and then delegates to an inner miniblock decompressor.
#[derive(Debug)]
pub struct CompressedMiniBlockDecompressor {
    inner: Box<dyn MiniBlockDecompressor>,
    compression: CompressionConfig,
}

impl CompressedMiniBlockDecompressor {
    pub fn new(inner: Box<dyn MiniBlockDecompressor>, compression: CompressionConfig) -> Self {
        Self { inner, compression }
    }
}

impl MiniBlockDecompressor for CompressedMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        // We expect exactly 2 buffers:
        // Buffer 0: Compressed data
        // Buffer 1: Encoded chunks structure
        if data.len() != 2 {
            return Err(lance_core::Error::io(
                format!(
                    "CompressedMiniBlockDecompressor expects 2 buffers, got {}",
                    data.len()
                ),
                location!(),
            ));
        }

        // Decompress the data
        let decompressor = GeneralBufferCompressor::get_compressor(self.compression);
        let mut decompressed_data = Vec::new();
        decompressor.decompress(&data[0], &mut decompressed_data)?;

        // Decode the chunks structure
        let chunks = decode_chunks(data[1].as_ref())?;

        // Calculate the number of unique buffers
        // Each chunk references global buffers, so we need to determine the unique buffer count
        let mut max_buffer_count = 0;
        for chunk in &chunks {
            max_buffer_count = max_buffer_count.max(chunk.buffer_sizes.len());
        }

        // Calculate total size for each buffer by summing across chunks
        let mut buffer_total_sizes = vec![0usize; max_buffer_count];
        for chunk in &chunks {
            for (i, &size) in chunk.buffer_sizes.iter().enumerate() {
                buffer_total_sizes[i] += size as usize;
            }
        }

        // Reconstruct the original buffers
        let mut reconstructed_buffers = Vec::new();
        let mut offset = 0;

        for buffer_size in buffer_total_sizes {
            if offset + buffer_size > decompressed_data.len() {
                return Err(lance_core::Error::io(
                    format!(
                        "Buffer reconstruction failed: offset {} + size {} exceeds data length {}",
                        offset,
                        buffer_size,
                        decompressed_data.len()
                    ),
                    location!(),
                ));
            }
            let buffer =
                LanceBuffer::from(decompressed_data[offset..offset + buffer_size].to_vec());
            reconstructed_buffers.push(buffer);
            offset += buffer_size;
        }

        // Delegate to the inner decompressor with reconstructed buffers
        self.inner.decompress(reconstructed_buffers, num_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::{DecompressionStrategy, DefaultDecompressionStrategy};
    use crate::data::{BlockInfo, FixedWidthDataBlock};
    use crate::encodings::physical::block::CompressionScheme;
    use crate::encodings::physical::rle::RleMiniBlockEncoder;
    use crate::encodings::physical::value::ValueEncoder;

    #[test]
    fn test_compressed_miniblock_with_rle() {
        // Create a simpler test case first
        let values = vec![1i32, 1, 1, 1, 2, 2, 2, 2];
        let data = LanceBuffer::from_bytes(
            bytemuck::cast_slice(&values).to_vec().into(),
            std::mem::align_of::<i32>() as u64,
        );
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data,
            num_values: 8,
            block_info: BlockInfo::new(),
        });

        // Test RLE compression first without LZ4
        let rle_encoder = RleMiniBlockEncoder;
        let (_rle_compressed, _) = rle_encoder.compress(block).unwrap();

        // Now test with LZ4 wrapper
        // Recreate the block
        let data2 = LanceBuffer::from_bytes(
            bytemuck::cast_slice(&values).to_vec().into(),
            std::mem::align_of::<i32>() as u64,
        );
        let block2 = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data: data2,
            num_values: 8,
            block_info: BlockInfo::new(),
        });

        let inner = Box::new(RleMiniBlockEncoder);
        let compression = CompressionConfig {
            scheme: CompressionScheme::Lz4,
            level: None,
        };
        let compressor = CompressedMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, encoding) = compressor.compress(block2).unwrap();

        // Verify the encoding structure - should be RLE without compression wrapper
        match &encoding.array_encoding {
            Some(pb::array_encoding::ArrayEncoding::Rle(rle)) => {
                assert_eq!(rle.bits_per_value, 32);
            }
            _ => panic!("Expected RLE encoding (not wrapped in CompressedMiniBlock)"),
        }

        // Verify compression was applied
        assert_eq!(compressed.num_values, 8);
        assert!(!compressed.data.is_empty());

        // For this small test, buffers shouldn't be compressed (too small)
        assert_eq!(compressed.data.len(), 2); // RLE produces 2 buffers
    }

    #[test]
    fn test_compressed_miniblock_small_buffers() {
        // Create test data with small amount of values using RLE encoder
        let values = vec![1i32, 1, 2, 2];
        let data = LanceBuffer::from_bytes(
            bytemuck::cast_slice(&values).to_vec().into(),
            std::mem::align_of::<i32>() as u64,
        );
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data,
            num_values: 4,
            block_info: BlockInfo::new(),
        });

        // Create compressor with RLE encoder and LZ4 compression
        let inner = Box::new(RleMiniBlockEncoder);
        let compression = CompressionConfig {
            scheme: CompressionScheme::Lz4,
            level: None,
        };
        let compressor = CompressedMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, _encoding) = compressor.compress(block).unwrap();

        // Verify compression was not applied for small buffers
        assert_eq!(compressed.num_values, 4);
        assert_eq!(compressed.data.len(), 2); // RLE produces 2 buffers
                                              // With only 2 unique values, buffers will be very small
        assert!(compressed.data[0].len() < MIN_BUFFER_SIZE_FOR_COMPRESSION);
        assert!(compressed.data[1].len() < MIN_BUFFER_SIZE_FOR_COMPRESSION);

        // Should return RLE encoding directly
        match &_encoding.array_encoding {
            Some(pb::array_encoding::ArrayEncoding::Rle(rle)) => {
                assert_eq!(rle.bits_per_value, 32);
            }
            _ => panic!("Expected RLE encoding (not wrapped in CompressedMiniBlock)"),
        }
    }

    #[test]
    fn test_compressed_mini_block_with_doubles() {
        // Create test data with doubles - RLE is not efficient for floating point data
        // but LZ4 can find patterns in the byte representation
        let mut values = Vec::with_capacity(1024);
        for i in 0..1024 {
            values.push((i as f64) * 0.1);
        }

        let data = LanceBuffer::from_bytes(
            bytemuck::cast_slice(&values).to_vec().into(),
            std::mem::align_of::<f64>() as u64,
        );
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            data,
            num_values: 1024,
            block_info: BlockInfo::new(),
        });

        // Create compressor with ValueEncoder inner and Zstd compression
        // ValueEncoder is better for floating point data than RLE
        let inner = Box::new(ValueEncoder {});
        let compression = CompressionConfig {
            scheme: CompressionScheme::Zstd,
            level: Some(3),
        };
        let compressor = CompressedMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, encoding) = compressor.compress(block).unwrap();

        // The encoding should be CompressedMiniBlock because doubles compress well with Zstd
        match &encoding.array_encoding {
            Some(pb::array_encoding::ArrayEncoding::CompressedMiniBlock(cm)) => {
                assert!(cm.inner.is_some());
                assert_eq!(cm.compression.as_ref().unwrap().scheme, "zstd");
                assert_eq!(cm.compression.as_ref().unwrap().level, Some(3));
            }
            Some(pb::array_encoding::ArrayEncoding::Flat(_)) => {
                // Also acceptable if compression didn't help
            }
            _ => panic!("Expected CompressedMiniBlock or Flat encoding"),
        }

        // Create decompressor using the encoding
        let decompression_strategy = DefaultDecompressionStrategy::default();
        let decompressor = decompression_strategy
            .create_miniblock_decompressor(&encoding)
            .unwrap();

        // Decompress the data
        let decompressed = decompressor
            .decompress(compressed.data, compressed.num_values)
            .unwrap();

        // Verify the round trip
        match decompressed {
            DataBlock::FixedWidth(decompressed) => {
                assert_eq!(64, decompressed.bits_per_value);
                assert_eq!(1024, decompressed.num_values);
                // Verify the data matches
                let decompressed_values: &[f64] = bytemuck::cast_slice(decompressed.data.as_ref());
                assert_eq!(values, decompressed_values);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    #[test]
    fn test_compressed_mini_block_with_zstd() {
        // Create test data with many unique values to ensure larger buffers
        // RLE is efficient with runs, so we need more unique values
        let mut values = Vec::with_capacity(8192);
        // Create 512 unique values, each repeated 16 times
        for i in 0..512 {
            values.extend(vec![i as u16; 16]);
        }
        let data = LanceBuffer::from_bytes(
            bytemuck::cast_slice(&values).to_vec().into(),
            std::mem::align_of::<u16>() as u64,
        );
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 16,
            data,
            num_values: 8192,
            block_info: BlockInfo::new(),
        });

        // Create compressor with RLE encoder and Zstd compression
        let inner = Box::new(RleMiniBlockEncoder);
        let compression = CompressionConfig {
            scheme: CompressionScheme::Zstd,
            level: Some(3),
        };
        let compressor = CompressedMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, encoding) = compressor.compress(block).unwrap();

        // Verify the encoding structure
        match &encoding.array_encoding {
            Some(pb::array_encoding::ArrayEncoding::CompressedMiniBlock(cm)) => {
                assert!(cm.inner.is_some());
                assert_eq!(cm.compression.as_ref().unwrap().scheme, "zstd");
                assert_eq!(cm.compression.as_ref().unwrap().level, Some(3));
            }
            Some(pb::array_encoding::ArrayEncoding::Rle(_)) => {}
            _ => panic!("Expected CompressedMiniBlock or Rle encoding"),
        }

        // Verify basic properties
        assert_eq!(compressed.num_values, 8192);
        assert_eq!(compressed.data.len(), 2);
    }

    #[test]
    fn test_compressed_mini_block_large_buffers() {
        // Use value encoding which doesn't compress data, ensuring large buffers
        // Create 1024 i32 values (4KB of data)
        let values: Vec<i32> = (0..1024).collect();
        let data = LanceBuffer::from_bytes(
            bytemuck::cast_slice(&values).to_vec().into(),
            std::mem::align_of::<i32>() as u64,
        );
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data,
            num_values: 1024,
            block_info: BlockInfo::new(),
        });

        // Create compressor with ValueEncoder (no compression) and Zstd wrapper
        let inner = Box::new(ValueEncoder {});
        let compression = CompressionConfig {
            scheme: CompressionScheme::Zstd,
            level: Some(3),
        };
        let compressor = CompressedMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, encoding) = compressor.compress(block).unwrap();

        // Should get CompressedMiniBlock encoding since buffer is 4KB
        match &encoding.array_encoding {
            Some(pb::array_encoding::ArrayEncoding::CompressedMiniBlock(cm)) => {
                assert!(cm.inner.is_some());
                assert_eq!(cm.compression.as_ref().unwrap().scheme, "zstd");
                assert_eq!(cm.compression.as_ref().unwrap().level, Some(3));

                // Verify inner encoding is Flat (from ValueEncoder)
                match &cm.inner.as_ref().unwrap().array_encoding {
                    Some(pb::array_encoding::ArrayEncoding::Flat(flat)) => {
                        assert_eq!(flat.bits_per_value, 32);
                    }
                    _ => panic!("Expected Flat inner encoding"),
                }
            }
            _ => panic!("Expected CompressedMiniBlock encoding"),
        }

        assert_eq!(compressed.num_values, 1024);
        // Compressed miniblock produces 2 buffers: compressed data + encoded chunks
        assert_eq!(compressed.data.len(), 2);
    }

    #[test]
    fn test_chunk_encoding_decoding() {
        // Test the chunk encoding and decoding functions
        let chunks = vec![
            MiniBlockChunk {
                buffer_sizes: vec![100, 200],
                log_num_values: 10,
            },
            MiniBlockChunk {
                buffer_sizes: vec![150, 250, 50],
                log_num_values: 8,
            },
            MiniBlockChunk {
                buffer_sizes: vec![300],
                log_num_values: 0,
            },
        ];

        let encoded = encode_chunks(&chunks);
        let decoded = decode_chunks(&encoded).unwrap();

        assert_eq!(chunks.len(), decoded.len());
        for (original, decoded) in chunks.iter().zip(decoded.iter()) {
            assert_eq!(original.buffer_sizes, decoded.buffer_sizes);
            assert_eq!(original.log_num_values, decoded.log_num_values);
        }
    }

    #[test]
    fn test_compressed_mini_block_with_lz4() {
        // Create test data with repeated patterns that LZ4 can compress well
        let mut values = Vec::with_capacity(2048);
        // Create a pattern with some repetition
        for i in 0..256i32 {
            for _ in 0..8 {
                values.push(i);
            }
        }

        let data = LanceBuffer::from_bytes(
            bytemuck::cast_slice(&values).to_vec().into(),
            std::mem::align_of::<i32>() as u64,
        );
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data,
            num_values: 2048,
            block_info: BlockInfo::new(),
        });

        // Create compressor with ValueEncoder and LZ4 compression
        let inner = Box::new(ValueEncoder {});
        let compression = CompressionConfig {
            scheme: CompressionScheme::Lz4,
            level: None,
        };
        let compressor = CompressedMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, encoding) = compressor.compress(block).unwrap();

        // Should get CompressedMiniBlock encoding
        match &encoding.array_encoding {
            Some(pb::array_encoding::ArrayEncoding::CompressedMiniBlock(cm)) => {
                assert!(cm.inner.is_some());
                assert_eq!(cm.compression.as_ref().unwrap().scheme, "lz4");
                assert_eq!(cm.compression.as_ref().unwrap().level, None);

                // Verify inner encoding is Flat (from ValueEncoder)
                match &cm.inner.as_ref().unwrap().array_encoding {
                    Some(pb::array_encoding::ArrayEncoding::Flat(flat)) => {
                        assert_eq!(flat.bits_per_value, 32);
                    }
                    _ => panic!("Expected Flat inner encoding"),
                }
            }
            _ => panic!("Expected CompressedMiniBlock encoding"),
        }

        assert_eq!(compressed.num_values, 2048);
        // Compressed miniblock produces 2 buffers: compressed data + encoded chunks
        assert_eq!(compressed.data.len(), 2);

        // Test decompression
        let decompression_strategy = DefaultDecompressionStrategy::default();
        let decompressor = decompression_strategy
            .create_miniblock_decompressor(&encoding)
            .unwrap();

        // Decompress the data
        let decompressed = decompressor
            .decompress(compressed.data, compressed.num_values)
            .unwrap();

        // Verify the round trip
        match decompressed {
            DataBlock::FixedWidth(decompressed) => {
                assert_eq!(32, decompressed.bits_per_value);
                assert_eq!(2048, decompressed.num_values);
                // Verify the data matches
                let decompressed_values: &[i32] = bytemuck::cast_slice(decompressed.data.as_ref());
                assert_eq!(values, decompressed_values);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }
}
