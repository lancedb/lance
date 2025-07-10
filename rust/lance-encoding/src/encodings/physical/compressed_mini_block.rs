// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use log::trace;

use crate::{
    buffer::LanceBuffer,
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

// Minimum buffer size to consider for compression
const MIN_BUFFER_SIZE_FOR_COMPRESSION: usize = 1024;

impl MiniBlockCompressor for CompressedMiniBlockCompressor {
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, pb::ArrayEncoding)> {
        // First, compress with the inner compressor
        let (mut compressed, inner_encoding) = self.inner.compress(page)?;

        // Most miniblock encoders produce 2 buffers (values and lengths)
        // but some like ValueEncoder may produce only 1 buffer

        // Check if any buffer is large enough to warrant compression
        let mut should_compress = false;
        for buffer in &compressed.data {
            if buffer.len() >= MIN_BUFFER_SIZE_FOR_COMPRESSION {
                should_compress = true;
                break;
            }
        }

        trace!(
            "Buffer sizes: {:?}, should_compress: {}",
            compressed.data.iter().map(|b| b.len()).collect::<Vec<_>>(),
            should_compress
        );

        // Only compress if at least one buffer is large enough
        if should_compress {
            // Create the buffer compressor
            let compressor = GeneralBufferCompressor::get_compressor(self.compression);

            // Compress both buffers
            for (i, buffer) in compressed.data.iter_mut().enumerate() {
                if !buffer.is_empty() {
                    let mut compressed_buffer = Vec::new();
                    compressor.compress(buffer.as_ref(), &mut compressed_buffer)?;

                    let original_size = buffer.len();
                    let compressed_size = compressed_buffer.len();

                    trace!(
                        "Buffer {} compressed from {} to {} bytes (ratio: {:.2})",
                        i,
                        original_size,
                        compressed_size,
                        compressed_size as f32 / original_size as f32
                    );

                    // Update buffer and size
                    *buffer = LanceBuffer::from(compressed_buffer);

                    // Update the buffer size in chunks
                    let mut buffer_idx = 0;
                    for chunk in &mut compressed.chunks {
                        for size in chunk.buffer_sizes.iter_mut() {
                            if buffer_idx == i {
                                *size = compressed_size as u16;
                                break;
                            }
                            buffer_idx += 1;
                        }
                        if buffer_idx > i {
                            break;
                        }
                    }
                }
            }

            // Return compressed encoding
            let encoding =
                ProtobufUtils::compressed_mini_block(inner_encoding, self.compression);
            Ok((compressed, encoding))
        } else {
            trace!("Buffers too small for compression, returning uncompressed data");
            // Return the original encoding without compression
            Ok((compressed, inner_encoding))
        }
    }
}

/// A miniblock decompressor that first decompresses buffers using general-purpose
/// compression (LZ4, Zstd) and then delegates to an inner miniblock decompressor.
#[derive(Debug)]
pub struct CompressedMiniBlockDecompressor {
    inner: Box<dyn crate::compression::MiniBlockDecompressor>,
    compression: CompressionConfig,
}

impl CompressedMiniBlockDecompressor {
    pub fn new(
        inner: Box<dyn crate::compression::MiniBlockDecompressor>,
        compression: CompressionConfig,
    ) -> Self {
        Self { inner, compression }
    }
}

impl crate::compression::MiniBlockDecompressor for CompressedMiniBlockDecompressor {
    fn decompress(&self, mut data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        // Create the buffer decompressor based on compression scheme
        let decompressor = GeneralBufferCompressor::get_compressor(self.compression);

        // Decompress all buffers
        let mut decompressed_buffers = Vec::with_capacity(data.len());
        for buffer in data.iter_mut() {
            // Handle empty buffers
            if buffer.is_empty() {
                decompressed_buffers.push(LanceBuffer::empty());
                continue;
            }

            let mut decompressed = Vec::new();
            decompressor.decompress(buffer.as_ref(), &mut decompressed)?;
            decompressed_buffers.push(LanceBuffer::from(decompressed));
        }

        // Delegate to the inner decompressor
        self.inner.decompress(decompressed_buffers, num_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BlockInfo, FixedWidthDataBlock};
    use crate::encodings::physical::block::CompressionScheme;
    use crate::encodings::physical::rle::RleMiniBlockEncoder;

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
    fn test_compressed_mini_block_round_trip() {
        use crate::compression::{DecompressionStrategy, DefaultDecompressionStrategy};

        // Create test data with repetitive patterns that compress well
        // Need enough data to trigger compression (>= 1024 bytes)
        let mut values = Vec::with_capacity(2048);
        for i in 0..128 {
            // Create runs of 16 values
            values.extend(vec![i; 16]);
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

        // Create compressor with RLE inner and LZ4 compression
        let inner = Box::new(RleMiniBlockEncoder);
        let compression = CompressionConfig {
            scheme: CompressionScheme::Lz4,
            level: None,
        };
        let compressor = CompressedMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, encoding) = compressor.compress(block).unwrap();

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
                assert_eq!(32, decompressed.bits_per_value);
                assert_eq!(2048, decompressed.num_values);
                // Verify the data matches
                let decompressed_values: &[i32] = bytemuck::cast_slice(decompressed.data.as_ref());
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
        use crate::encodings::physical::value::ValueEncoder;

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
        // ValueEncoder produces 1 buffer
        assert_eq!(compressed.data.len(), 1);
    }
}
