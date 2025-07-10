// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use log::trace;

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

impl MiniBlockCompressor for CompressedMiniBlockCompressor {
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, pb::ArrayEncoding)> {
        // First, compress with the inner compressor
        let (mut inner_compressed, inner_encoding) = self.inner.compress(page)?;

        // Return the original encoding without compression if there's no data or
        // the first buffer is not large enough
        if inner_compressed.data.is_empty()
            || inner_compressed.data[0].len() < MIN_BUFFER_SIZE_FOR_COMPRESSION
        {
            return Ok((inner_compressed, inner_encoding));
        }

        let compressor = GeneralBufferCompressor::get_compressor(self.compression);

        let mut compressed_buffer = Vec::new();
        compressor.compress(&inner_compressed.data[0], &mut compressed_buffer)?;

        let original_size = inner_compressed.data[0].len();
        let compressed_size = compressed_buffer.len();

        // If original_size is smaller than compressed_size, we just return the
        // not compressed one.
        if original_size <= compressed_size {
            return Ok((inner_compressed, inner_encoding));
        }

        trace!(
            "Buffer compressed from {} to {} bytes (ratio: {:.2})",
            original_size,
            compressed_size,
            compressed_size as f32 / original_size as f32
        );

        // Update buffer only - chunks don't need updating since they reference global buffers
        inner_compressed.data[0] = LanceBuffer::from(compressed_buffer);

        // Return compressed encoding
        let encoding = ProtobufUtils::compressed_mini_block(inner_encoding, self.compression);
        Ok((inner_compressed, encoding))
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
    fn decompress(&self, mut data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        // Create the buffer decompressor based on compression scheme
        let decompressor = GeneralBufferCompressor::get_compressor(self.compression);

        // Decompress only the first buffer, keep others as-is
        let mut decompressed = Vec::new();
        decompressor.decompress(&data[0], &mut decompressed)?;
        data[0] = LanceBuffer::from(decompressed);

        // Delegate to the inner decompressor
        self.inner.decompress(data, num_values)
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
        // ValueEncoder produces 1 buffer
        assert_eq!(compressed.data.len(), 1);
    }
}
