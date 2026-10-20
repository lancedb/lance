// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::ArrowNativeType;
use log::trace;

use crate::{
    Result,
    buffer::LanceBuffer,
    compression::{
        BlockCompressor, BlockDecompressor, FixedPerValueDecompressor, MiniBlockDecompressor,
        VariablePerValueDecompressor, require_block_payload,
    },
    compression_config::BssMode,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    encodings::{
        logical::primitive::{
            fullzip::{PerValueCompressor, PerValueDataBlock},
            miniblock::{MiniBlockCompressed, MiniBlockCompressionContext, MiniBlockCompressor},
        },
        physical::{
            block::{BufferCompressor, CompressionConfig, GeneralBufferCompressor},
            byte_stream_split::{join_value_bytes, split_value_bytes},
            value::{ValueDecompressor, ValueEncoder},
        },
    },
    format::{
        ProtobufUtils21,
        pb21::{self, CompressiveEncoding, compressive_encoding::Compression},
    },
};
use lance_core::Error;

/// General-purpose compressor that wraps one payload-bearing block codec.
#[derive(Debug)]
pub struct GeneralBlockCompressor {
    child: Box<dyn BlockCompressor>,
    compression: CompressionConfig,
}

impl GeneralBlockCompressor {
    pub fn new(child: Box<dyn BlockCompressor>, compression: CompressionConfig) -> Self {
        Self { child, compression }
    }
}

impl BlockCompressor for GeneralBlockCompressor {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let (payload, child_encoding) = self.child.compress(data)?;
        let payload = payload.ok_or_else(|| {
            Error::invalid_input("General block compression requires a payload-bearing child")
        })?;
        let compressor = GeneralBufferCompressor::get_compressor(self.compression)?;
        let mut compressed = Vec::new();
        compressor.compress(&payload, &mut compressed)?;
        Ok((
            Some(LanceBuffer::from(compressed)),
            ProtobufUtils21::wrapped(self.compression, child_encoding)?,
        ))
    }
}

/// Bounded fixed-width decoder for a general-compressed block payload.
#[derive(Debug)]
pub(crate) struct FixedWidthGeneralBlockDecompressor {
    child: Box<dyn BlockDecompressor>,
    compression: CompressionConfig,
    bits_per_value: u64,
}

impl FixedWidthGeneralBlockDecompressor {
    pub(crate) fn new(
        child: Box<dyn BlockDecompressor>,
        compression: CompressionConfig,
        bits_per_value: u64,
    ) -> Self {
        Self {
            child,
            compression,
            bits_per_value,
        }
    }
}

impl BlockDecompressor for FixedWidthGeneralBlockDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let data = require_block_payload(data, "General fixed-width block")?;
        let expected_bytes = usize::try_from(num_values)
            .ok()
            .and_then(|num_values| {
                num_values.checked_mul(usize::try_from(self.bits_per_value / 8).ok()?)
            })
            .ok_or_else(|| {
                Error::invalid_input("General fixed-width output length overflows usize")
            })?;
        let compressor = GeneralBufferCompressor::get_compressor(self.compression)?;
        let mut decompressed = Vec::new();
        compressor.decompress_exact(&data, &mut decompressed, expected_bytes)?;
        self.child
            .decompress(Some(LanceBuffer::from(decompressed)), num_values)
    }
}

/// A miniblock compressor that wraps another miniblock compressor and applies
/// general-purpose compression (LZ4, Zstd) to the resulting buffers.
#[derive(Debug)]
pub struct GeneralMiniBlockCompressor {
    inner: Box<dyn MiniBlockCompressor>,
    compression: CompressionConfig,
}

impl GeneralMiniBlockCompressor {
    pub fn new(inner: Box<dyn MiniBlockCompressor>, compression: CompressionConfig) -> Self {
        Self { inner, compression }
    }
}

/// Minimum buffer size to consider for compression
const MIN_BUFFER_SIZE_FOR_COMPRESSION: usize = 4 * 1024;

use super::super::logical::primitive::miniblock::MiniBlockChunk;

impl MiniBlockCompressor for GeneralMiniBlockCompressor {
    fn compress(
        &self,
        context: MiniBlockCompressionContext,
        page: DataBlock,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        // First, compress with the inner compressor
        let (inner_compressed, inner_encoding) = self.inner.compress(context, page)?;

        // Return the original encoding without compression if there's no data or
        // the first buffer is not large enough
        if inner_compressed.data.is_empty()
            || inner_compressed.data[0].len() < MIN_BUFFER_SIZE_FOR_COMPRESSION
        {
            return Ok((inner_compressed, inner_encoding));
        }

        // We'll compress each chunk's portion of the first buffer independently
        let first_buffer = &inner_compressed.data[0];
        let mut compressed_first_buffer = Vec::new();
        let mut new_chunks = Vec::with_capacity(inner_compressed.chunks.iter().len());
        let mut offset = 0usize;
        let mut total_original_size = 0usize;

        let compressor = GeneralBufferCompressor::get_compressor(self.compression)?;

        for chunk in &inner_compressed.chunks {
            let chunk_first_buffer_size = chunk.buffer_sizes[0] as usize;

            let chunk_data = &first_buffer.as_ref()[offset..offset + chunk_first_buffer_size];
            total_original_size += chunk_first_buffer_size;

            let compressed_start = compressed_first_buffer.len();
            compressor.compress(chunk_data, &mut compressed_first_buffer)?;
            let compressed_size = compressed_first_buffer.len() - compressed_start;

            // Create new chunk with updated first buffer size
            let mut new_buffer_sizes = chunk.buffer_sizes.clone();
            new_buffer_sizes[0] = compressed_size as u32;

            new_chunks.push(MiniBlockChunk {
                buffer_sizes: new_buffer_sizes,
                log_num_values: chunk.log_num_values,
            });

            offset += chunk_first_buffer_size;
        }

        // Check if compression was effective
        let compressed_total_size = compressed_first_buffer.len();
        if compressed_total_size >= total_original_size {
            // Compression didn't help, return original
            return Ok((inner_compressed, inner_encoding));
        }

        trace!(
            "First buffer compressed from {} to {} bytes (ratio: {:.2})",
            total_original_size,
            compressed_total_size,
            compressed_total_size as f32 / total_original_size as f32
        );

        // Build final buffers: compressed first buffer + remaining original buffers
        let mut final_buffers = vec![LanceBuffer::from(compressed_first_buffer)];
        final_buffers.extend(inner_compressed.data.into_iter().skip(1));

        let compressed_result = MiniBlockCompressed {
            data: final_buffers,
            chunks: new_chunks,
            num_values: inner_compressed.num_values,
        };

        // Return compressed encoding
        let encoding = ProtobufUtils21::wrapped(self.compression, inner_encoding)?;
        Ok((compressed_result, encoding))
    }
}

/// A miniblock decompressor that first decompresses buffers using general-purpose
/// compression (LZ4, Zstd) and then delegates to an inner miniblock decompressor.
#[derive(Debug)]
pub struct GeneralMiniBlockDecompressor {
    inner: Box<dyn MiniBlockDecompressor>,
    compression: CompressionConfig,
}

impl GeneralMiniBlockDecompressor {
    pub fn new(inner: Box<dyn MiniBlockDecompressor>, compression: CompressionConfig) -> Self {
        Self { inner, compression }
    }
}

impl MiniBlockDecompressor for GeneralMiniBlockDecompressor {
    fn num_buffers(&self) -> usize {
        self.inner.num_buffers()
    }

    fn decompress(&self, mut data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if data.len() != self.num_buffers() || data.is_empty() {
            return Err(Error::invalid_input(format!(
                "General mini-block has {} buffers, expected {}",
                data.len(),
                self.num_buffers()
            )));
        }
        let mut decompressed_buffer = Vec::new();

        let decompressor = GeneralBufferCompressor::get_compressor(self.compression)?;
        decompressor.decompress(&data[0], &mut decompressed_buffer)?;
        data[0] = LanceBuffer::from(decompressed_buffer);

        self.inner.decompress(data, num_values)
    }

    fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
        self.inner.decoded_size_bytes(num_values)
    }
}

/// Returns the item width to byte-stream-split a flattened per-value description by.
///
/// Splitting needs every value to be a whole number of multi-byte items, which is not the case
/// when a fixed-size-list layer zips item validity bytes in front of the items.
fn byte_stream_split_item_bytes(values: &CompressiveEncoding) -> Option<usize> {
    let mut current = values;
    loop {
        match current.compression.as_ref()? {
            Compression::FixedSizeList(fsl) if !fsl.has_validity => {
                current = fsl.values.as_deref()?;
            }
            Compression::Flat(flat)
                if flat.bits_per_value >= 16 && flat.bits_per_value.is_multiple_of(8) =>
            {
                return usize::try_from(flat.bits_per_value / 8).ok();
            }
            _ => return None,
        }
    }
}

/// Compresses full-zip fixed-width and fixed-size-list values one value at a time.
///
/// Each value is flattened the way [`ValueEncoder`] stores it (including any zipped item
/// validity), optionally split into byte streams, and compressed on its own so that a single
/// row can be decoded without reading its neighbors. Compressed values differ in size, so the
/// output is variable-width. If compression does not make the page smaller the values are
/// stored uncompressed instead.
#[derive(Debug)]
pub struct GeneralFixedPerValueCompressor {
    compression: CompressionConfig,
    compressor: Box<dyn BufferCompressor>,
    byte_stream_split: BssMode,
}

impl GeneralFixedPerValueCompressor {
    /// `byte_stream_split` controls whether each value is split into byte streams before
    /// compression. [`BssMode::Auto`] and [`BssMode::On`] split whenever a value holds more than
    /// one multi-byte item and no item validity, which matches the Blosc and Zarr default:
    /// splitting within one value rarely hurts general compression and often helps a lot.
    pub fn try_new(compression: CompressionConfig, byte_stream_split: BssMode) -> Result<Self> {
        Ok(Self {
            compression,
            compressor: GeneralBufferCompressor::get_compressor(compression)?,
            byte_stream_split,
        })
    }
}

impl PerValueCompressor for GeneralFixedPerValueCompressor {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, CompressiveEncoding)> {
        if !matches!(data, DataBlock::FixedWidth(_) | DataBlock::FixedSizeList(_)) {
            return Err(Error::invalid_input(format!(
                "General fixed-width per-value compression cannot compress a {} block",
                data.name()
            )));
        }
        let (values, values_encoding) =
            PerValueCompressor::compress(&ValueEncoder::default(), data)?;
        let PerValueDataBlock::Fixed(values) = values else {
            return Err(Error::internal(
                "ValueEncoder produced variable-width per-value data".to_string(),
            ));
        };
        if !values.bits_per_value.is_multiple_of(8) {
            return Err(Error::invalid_input(format!(
                "General fixed-width per-value compression requires byte-aligned values, got {} bits per value",
                values.bits_per_value
            )));
        }
        let bytes_per_value = (values.bits_per_value / 8) as usize;
        if bytes_per_value == 0 {
            return Ok((PerValueDataBlock::Fixed(values), values_encoding));
        }
        let split_item_bytes = match self.byte_stream_split {
            BssMode::Off => None,
            BssMode::On | BssMode::Auto => byte_stream_split_item_bytes(&values_encoding)
                .filter(|item_bytes| bytes_per_value / item_bytes > 1),
        };

        let num_values = values.num_values as usize;
        let mut compressed = Vec::with_capacity(values.data.len() / 2);
        let mut offsets = Vec::with_capacity(num_values + 1);
        offsets.push(0_u64);
        match split_item_bytes {
            Some(item_bytes) => {
                let mut streams = vec![0_u8; bytes_per_value];
                for value in values.data.chunks_exact(bytes_per_value) {
                    split_value_bytes(value, item_bytes, &mut streams);
                    self.compressor.compress(&streams, &mut compressed)?;
                    offsets.push(compressed.len() as u64);
                }
            }
            None => {
                for value in values.data.chunks_exact(bytes_per_value) {
                    self.compressor.compress(value, &mut compressed)?;
                    offsets.push(compressed.len() as u64);
                }
            }
        }

        let bits_per_offset: u8 = if compressed.len() <= u32::MAX as usize {
            32
        } else {
            64
        };
        // A full-zip page stores a length prefix in front of every variable-width value.
        let length_prefix_bytes = num_values * (bits_per_offset as usize / 8);
        if compressed.len() + length_prefix_bytes >= values.data.len() {
            return Ok((PerValueDataBlock::Fixed(values), values_encoding));
        }

        let offsets = if bits_per_offset == 32 {
            LanceBuffer::reinterpret_vec(
                offsets
                    .into_iter()
                    .map(|offset| offset as u32)
                    .collect::<Vec<_>>(),
            )
        } else {
            LanceBuffer::reinterpret_vec(offsets)
        };
        let values_encoding = if split_item_bytes.is_some() {
            ProtobufUtils21::byte_stream_split(values_encoding)
        } else {
            values_encoding
        };
        Ok((
            PerValueDataBlock::Variable(VariableWidthBlock {
                data: LanceBuffer::from(compressed),
                offsets,
                bits_per_offset,
                num_values: values.num_values,
                block_info: BlockInfo::new(),
            }),
            ProtobufUtils21::wrapped(self.compression, values_encoding)?,
        ))
    }
}

/// Decompresses values written by [`GeneralFixedPerValueCompressor`].
#[derive(Debug)]
pub struct GeneralFixedPerValueDecompressor {
    compressor: Box<dyn BufferCompressor>,
    /// The item width each value was byte-stream-split by, or `None` if values were not split.
    split_item_bytes: Option<usize>,
    bytes_per_value: usize,
    values: ValueDecompressor,
}

impl GeneralFixedPerValueDecompressor {
    /// Creates a decompressor for a `General` per-value description whose values are `Flat` or
    /// `FixedSizeList`, optionally wrapped in `ByteStreamSplit`.
    pub fn try_new(description: &pb21::General) -> Result<Self> {
        let compression = description.compression.as_ref().ok_or_else(|| {
            Error::invalid_input("General per-value encoding is missing its compression config")
        })?;
        let compression = CompressionConfig::new(compression.scheme().try_into()?, None);
        let values = description.values.as_deref().ok_or_else(|| {
            Error::invalid_input("General per-value encoding is missing its values encoding")
        })?;
        let (values, is_split) = match values.compression.as_ref() {
            Some(Compression::ByteStreamSplit(split)) => (
                split.values.as_deref().ok_or_else(|| {
                    Error::invalid_input("ByteStreamSplit encoding is missing its values codec")
                })?,
                true,
            ),
            _ => (values, false),
        };
        let decompressor = match values.compression.as_ref() {
            Some(Compression::Flat(flat)) if flat.bits_per_value > 0 && flat.data.is_none() => {
                ValueDecompressor::from_flat(flat)
            }
            Some(Compression::FixedSizeList(fsl)) => ValueDecompressor::from_fsl(fsl)?,
            _ => {
                return Err(Error::invalid_input(
                    "General fixed-width per-value compression requires plain Flat or FixedSizeList values",
                ));
            }
        };
        let bits_per_value = FixedPerValueDecompressor::bits_per_value(&decompressor);
        if bits_per_value == 0 || !bits_per_value.is_multiple_of(8) {
            return Err(Error::invalid_input(format!(
                "General fixed-width per-value compression requires positive byte-aligned values, got {bits_per_value} bits per value"
            )));
        }
        let bytes_per_value = usize::try_from(bits_per_value / 8).map_err(|_| {
            Error::invalid_input(format!(
                "Per-value width of {bits_per_value} bits does not fit usize"
            ))
        })?;
        let split_item_bytes = if is_split {
            let item_bytes = byte_stream_split_item_bytes(values)
                .filter(|item_bytes| bytes_per_value.is_multiple_of(*item_bytes))
                .ok_or_else(|| {
                    Error::invalid_input(
                        "Per-value byte stream split requires multi-byte items without item validity",
                    )
                })?;
            Some(item_bytes)
        } else {
            None
        };
        Ok(Self {
            compressor: GeneralBufferCompressor::get_compressor(compression)?,
            split_item_bytes,
            bytes_per_value,
            values: decompressor,
        })
    }

    /// Returns the compressed bytes of one value, or `None` for a null item.
    ///
    /// Full-zip pages store no payload for null items, and a compressed value is never empty
    /// because every codec prefixes its output with the decompressed length.
    fn value_payload<'a, T: ArrowNativeType>(
        data: &'a [u8],
        window: &[T],
    ) -> Result<Option<&'a [u8]>> {
        let (start, end) = (window[0].as_usize(), window[1].as_usize());
        let payload = data.get(start..end).ok_or_else(|| {
            Error::invalid_input(format!(
                "Per-value offsets {start}..{end} exceed the {} compressed bytes",
                data.len()
            ))
        })?;
        Ok((!payload.is_empty()).then_some(payload))
    }

    fn decompress_values<T: ArrowNativeType>(
        &self,
        data: &[u8],
        offsets: &[T],
        num_values: u64,
        decompressed: &mut Vec<u8>,
    ) -> Result<()> {
        if offsets.len() as u64 != num_values + 1 {
            return Err(Error::invalid_input(format!(
                "Per-value block has {} offsets for {num_values} values",
                offsets.len()
            )));
        }
        let bytes_per_value = self.bytes_per_value;
        match self.split_item_bytes {
            None => {
                for window in offsets.windows(2) {
                    match Self::value_payload(data, window)? {
                        Some(payload) => {
                            self.compressor.decompress_exact(
                                payload,
                                decompressed,
                                bytes_per_value,
                            )?;
                        }
                        None => decompressed.resize(decompressed.len() + bytes_per_value, 0),
                    }
                }
            }
            Some(item_bytes) => {
                let mut streams = Vec::with_capacity(bytes_per_value);
                for window in offsets.windows(2) {
                    let start = decompressed.len();
                    decompressed.resize(start + bytes_per_value, 0);
                    if let Some(payload) = Self::value_payload(data, window)? {
                        streams.clear();
                        self.compressor
                            .decompress_exact(payload, &mut streams, bytes_per_value)?;
                        join_value_bytes(&streams, item_bytes, &mut decompressed[start..]);
                    }
                }
            }
        }
        Ok(())
    }
}

impl VariablePerValueDecompressor for GeneralFixedPerValueDecompressor {
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock> {
        let num_values = data.num_values;
        let output_bytes = usize::try_from(num_values)
            .ok()
            .and_then(|num_values| num_values.checked_mul(self.bytes_per_value))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "{num_values} values of {} bytes overflow usize",
                    self.bytes_per_value
                ))
            })?;
        let mut decompressed = Vec::new();
        decompressed
            .try_reserve_exact(output_bytes)
            .map_err(|error| {
                Error::invalid_input(format!(
                    "Could not reserve {output_bytes} bytes for decompressed values: {error}"
                ))
            })?;
        match data.bits_per_offset {
            32 => self.decompress_values(
                &data.data,
                data.offsets.borrow_to_typed_slice::<u32>().as_ref(),
                num_values,
                &mut decompressed,
            )?,
            64 => self.decompress_values(
                &data.data,
                data.offsets.borrow_to_typed_slice::<u64>().as_ref(),
                num_values,
                &mut decompressed,
            )?,
            bits_per_offset => {
                return Err(Error::invalid_input(format!(
                    "Per-value offsets must be 32 or 64 bits, got {bits_per_offset}"
                )));
            }
        }
        FixedPerValueDecompressor::decompress(
            &self.values,
            FixedWidthDataBlock {
                data: LanceBuffer::from(decompressed),
                bits_per_value: self.bytes_per_value as u64 * 8,
                num_values,
                block_info: BlockInfo::new(),
            },
            num_values,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::compression::{DecompressionStrategy, DefaultDecompressionStrategy};
    use crate::encodings::physical::block::CompressionScheme;
    use crate::encodings::physical::rle::RleEncoder;
    use crate::testing::{
        TestCases, TestEncoding, check_round_trip_encoding_of_data, extract_array_encoding_chain,
    };
    use arrow_array::{
        Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float64Array, Int32Array,
        StringArray, StructArray, UInt16Array, make_array,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Fields};
    use rstest::rstest;

    fn miniblock_context() -> MiniBlockCompressionContext {
        MiniBlockCompressionContext::new(0, true, true)
    }

    #[derive(Debug)]
    struct TestCase {
        name: &'static str,
        inner_encoder: Box<dyn MiniBlockCompressor>,
        compression: CompressionConfig,
        data: DataBlock,
        expected_compressed: bool, // Whether we expect compression to be applied
        min_compression_ratio: f32, // Minimum compression ratio if compressed
    }

    fn create_test_cases() -> Vec<TestCase> {
        vec![
            // Small data with RLE - should not compress due to size threshold
            TestCase {
                name: "small_rle_data",
                inner_encoder: Box::new(RleEncoder::new()),
                compression: CompressionConfig {
                    scheme: CompressionScheme::Lz4,
                    level: None,
                },
                data: create_repeated_i32_block(vec![1, 1, 1, 1, 2, 2, 2, 2]),
                expected_compressed: false,
                min_compression_ratio: 1.0,
            },
            // Large repeated data with RLE + LZ4
            TestCase {
                name: "large_rle_lz4",
                inner_encoder: Box::new(RleEncoder::new()),
                compression: CompressionConfig {
                    scheme: CompressionScheme::Lz4,
                    level: None,
                },
                data: create_pattern_i32_block(2048, |i| (i / 8) as i32),
                expected_compressed: false, // RLE already compresses well, additional LZ4 may not help
                min_compression_ratio: 1.0,
            },
            // Large repeated data with RLE + Zstd
            TestCase {
                name: "large_rle_zstd",
                inner_encoder: Box::new(RleEncoder::new()),
                compression: CompressionConfig {
                    scheme: CompressionScheme::Zstd,
                    level: Some(3),
                },
                data: create_pattern_i32_block(8192, |i| (i / 16) as i32),
                expected_compressed: true, // Zstd might provide additional compression
                min_compression_ratio: 0.9, // But not as much since RLE already compressed
            },
            // Sequential data with ValueEncoder + LZ4
            TestCase {
                name: "sequential_value_lz4",
                inner_encoder: Box::new(ValueEncoder {}),
                compression: CompressionConfig {
                    scheme: CompressionScheme::Lz4,
                    level: None,
                },
                data: create_pattern_i32_block(1024, |i| i as i32),
                expected_compressed: false, // Sequential data doesn't compress well
                min_compression_ratio: 1.0,
            },
            // Float data with ValueEncoder + Zstd
            TestCase {
                name: "float_value_zstd",
                inner_encoder: Box::new(ValueEncoder {}),
                compression: CompressionConfig {
                    scheme: CompressionScheme::Zstd,
                    level: Some(3),
                },
                data: create_pattern_f64_block(1024, |i| i as f64 * 0.1),
                expected_compressed: true,
                min_compression_ratio: 0.9,
            },
        ]
    }

    fn create_repeated_i32_block(values: Vec<i32>) -> DataBlock {
        let array = Int32Array::from(values);
        DataBlock::from_array(array)
    }

    fn create_pattern_i32_block<F>(size: usize, pattern: F) -> DataBlock
    where
        F: Fn(usize) -> i32,
    {
        let values: Vec<i32> = (0..size).map(pattern).collect();
        let array = Int32Array::from(values);
        DataBlock::from_array(array)
    }

    fn create_pattern_f64_block<F>(size: usize, pattern: F) -> DataBlock
    where
        F: Fn(usize) -> f64,
    {
        let values: Vec<f64> = (0..size).map(pattern).collect();
        let array = Float64Array::from(values);
        DataBlock::from_array(array)
    }

    fn run_round_trip_test(test_case: TestCase) {
        let compressor =
            GeneralMiniBlockCompressor::new(test_case.inner_encoder, test_case.compression);

        // Compress the data
        let (compressed, encoding) = compressor
            .compress(miniblock_context(), test_case.data)
            .unwrap();

        // Check if compression was applied as expected
        match &encoding.compression {
            Some(Compression::General(cm)) => {
                assert!(
                    test_case.expected_compressed,
                    "{}: Expected compression to be applied",
                    test_case.name
                );
                assert_eq!(
                    CompressionScheme::try_from(cm.compression.as_ref().unwrap().scheme()).unwrap(),
                    test_case.compression.scheme
                );
            }
            _ => {
                // Could be RLE or other encoding if compression didn't help
                if test_case.expected_compressed {
                    // Check if it's RLE encoding (which means compression didn't help)
                    match &encoding.compression {
                        Some(Compression::Rle(_)) => {
                            // RLE encoding returned - compression didn't help
                        }
                        Some(Compression::Flat(_)) => {
                            // Flat encoding returned - compression didn't help
                        }
                        _ => {
                            panic!(
                                "{}: Expected GeneralMiniBlock but got {:?}",
                                test_case.name, encoding.compression
                            );
                        }
                    }
                }
            }
        }

        // Verify chunks are created correctly
        assert!(
            !compressed.chunks.is_empty(),
            "{}: No chunks created",
            test_case.name
        );

        // Test decompression by simulating the actual miniblock decoding process
        let decompressed_data = decompress_miniblock_chunks(&compressed, &encoding);

        // Verify round trip by checking data size
        // We expect the decompressed data to match the original number of values
        // The bytes per value depends on the test case
        let bytes_per_value = if test_case.name.contains("float") {
            8 // f64
        } else {
            4 // i32
        };
        let expected_bytes = compressed.num_values as usize * bytes_per_value;
        assert_eq!(
            expected_bytes,
            decompressed_data.len(),
            "{}: Data size mismatch",
            test_case.name
        );

        // Check compression ratio if applicable
        if test_case.expected_compressed {
            let compression_ratio = compressed.data[0].len() as f32 / expected_bytes as f32;
            assert!(
                compression_ratio <= test_case.min_compression_ratio,
                "{}: Compression ratio {} > expected {}",
                test_case.name,
                compression_ratio,
                test_case.min_compression_ratio
            );
        }
    }

    fn decompress_miniblock_chunks(
        compressed: &MiniBlockCompressed,
        encoding: &CompressiveEncoding,
    ) -> Vec<u8> {
        let mut decompressed_data = Vec::new();
        let mut offsets = vec![0usize; compressed.data.len()]; // Track offset for each buffer
        let decompression_strategy = DefaultDecompressionStrategy::default();

        for chunk in &compressed.chunks {
            let chunk_values = if chunk.log_num_values > 0 {
                1u64 << chunk.log_num_values
            } else {
                // Last chunk - calculate remaining values
                let decompressed_values =
                    decompressed_data.len() as u64 / get_bytes_per_value(compressed) as u64;
                compressed.num_values.saturating_sub(decompressed_values)
            };

            // Extract buffers for this chunk
            let mut chunk_buffers = Vec::new();
            for (i, &size) in chunk.buffer_sizes.iter().enumerate() {
                if i < compressed.data.len() {
                    let buffer_data =
                        compressed.data[i].slice_with_length(offsets[i], size as usize);
                    chunk_buffers.push(buffer_data);
                    offsets[i] += size as usize;
                }
            }

            // Create a decompressor for this chunk
            let decompressor = decompression_strategy
                .create_miniblock_decompressor(encoding, &decompression_strategy)
                .unwrap();

            // Decompress the chunk
            let chunk_decompressed = decompressor
                .decompress(chunk_buffers, chunk_values)
                .unwrap();

            match chunk_decompressed {
                DataBlock::FixedWidth(ref block) => {
                    decompressed_data.extend_from_slice(block.data.as_ref());
                }
                _ => panic!("Expected FixedWidth block"),
            }
        }

        decompressed_data
    }

    fn get_bytes_per_value(compressed: &MiniBlockCompressed) -> usize {
        // This is a simplification - in reality we'd need to know the data type
        // For our tests, we mostly use i32 (4 bytes) or f64 (8 bytes)
        // We can try to guess based on the data size
        if compressed.num_values == 0 {
            return 4; // Default to i32
        }

        // For float tests, the number is usually 1024 and we use f64
        if compressed.num_values == 1024 {
            return 8; // Likely f64
        }

        4 // Default to i32
    }

    #[test]
    fn test_compressed_mini_block_table_driven() {
        for test_case in create_test_cases() {
            run_round_trip_test(test_case);
        }
    }

    #[test]
    fn test_compressed_mini_block_threshold() {
        // Test that small buffers don't get compressed
        let small_test = TestCase {
            name: "small_buffer_no_compression",
            inner_encoder: Box::new(RleEncoder::new()),
            compression: CompressionConfig {
                scheme: CompressionScheme::Lz4,
                level: None,
            },
            data: create_repeated_i32_block(vec![1, 1, 2, 2]),
            expected_compressed: false,
            min_compression_ratio: 1.0,
        };
        run_round_trip_test(small_test);
    }

    #[test]
    fn test_compressed_mini_block_with_doubles() {
        // Test with large sequential doubles that should compress well with Zstd
        // The test focuses on verifying that GeneralMiniBlock works correctly
        // when wrapping a simple ValueEncoder
        let test_case = TestCase {
            name: "float_values_with_zstd",
            inner_encoder: Box::new(ValueEncoder {}),
            compression: CompressionConfig {
                scheme: CompressionScheme::Zstd,
                level: Some(3),
            },
            // Create enough data to ensure compression is applied
            data: create_pattern_f64_block(1024, |i| (i / 10) as f64),
            expected_compressed: true,
            min_compression_ratio: 0.5, // Zstd should achieve good compression on repetitive data
        };

        run_round_trip_test(test_case);
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
        let compressor = GeneralMiniBlockCompressor::new(inner, compression);

        // Compress the data
        let (compressed, encoding) = compressor.compress(miniblock_context(), block).unwrap();

        // Should get GeneralMiniBlock encoding since buffer is 4KB
        match &encoding.compression {
            Some(Compression::General(cm)) => {
                assert!(cm.values.is_some());
                assert_eq!(
                    cm.compression.as_ref().unwrap().scheme(),
                    pb21::CompressionScheme::CompressionAlgorithmZstd
                );
                assert_eq!(cm.compression.as_ref().unwrap().level, Some(3));

                // Verify inner encoding is Flat (from ValueEncoder)
                match &cm.values.as_ref().unwrap().compression {
                    Some(Compression::Flat(flat)) => {
                        assert_eq!(flat.bits_per_value, 32);
                    }
                    _ => panic!("Expected Flat inner encoding"),
                }
            }
            _ => panic!("Expected GeneralMiniBlock encoding"),
        }

        assert_eq!(compressed.num_values, 1024);
        // ValueEncoder produces 1 buffer, so compressed result also has 1 buffer
        assert_eq!(compressed.data.len(), 1);
    }

    // Special test cases that don't fit the table-driven pattern

    #[test]
    fn test_compressed_mini_block_rle_multiple_buffers() {
        // RLE produces 2 buffers (values and lengths), test that both are handled correctly
        let data = create_repeated_i32_block(vec![1; 100]);
        let compressor = GeneralMiniBlockCompressor::new(
            Box::new(RleEncoder::new()),
            CompressionConfig {
                scheme: CompressionScheme::Lz4,
                level: None,
            },
        );

        let (compressed, _) = compressor.compress(miniblock_context(), data).unwrap();
        // RLE produces 2 buffers, but only the first one is compressed
        assert_eq!(compressed.data.len(), 2);
    }

    #[test]
    fn test_rle_with_general_miniblock_wrapper() {
        // Test that RLE encoding with bits_per_value >= 32 is automatically wrapped
        // in GeneralMiniBlock with LZ4 compression

        // This test directly tests the RLE encoder behavior
        // When bits_per_value >= 32, RLE should be wrapped in GeneralMiniBlock with LZ4

        // Test case 1: 32-bit RLE data
        let test_32 = TestCase {
            name: "rle_32bit_with_general_wrapper",
            inner_encoder: Box::new(RleEncoder::new()),
            compression: CompressionConfig {
                scheme: CompressionScheme::Lz4,
                level: None,
            },
            data: create_repeated_i32_block(vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
            expected_compressed: false, // RLE already compresses well, LZ4 might not help much
            min_compression_ratio: 1.0,
        };

        // For 32-bit RLE, the compression strategy should automatically wrap it
        // Let's directly test the compressor
        let compressor = GeneralMiniBlockCompressor::new(
            Box::new(RleEncoder::new()),
            CompressionConfig {
                scheme: CompressionScheme::Lz4,
                level: None,
            },
        );

        let (_compressed, encoding) = compressor
            .compress(miniblock_context(), test_32.data)
            .unwrap();

        // Verify the encoding structure
        match &encoding.compression {
            Some(Compression::General(cm)) => {
                // Check inner encoding is RLE
                match &cm.values.as_ref().unwrap().compression {
                    Some(Compression::Rle(rle)) => {
                        let Compression::Flat(values) =
                            rle.values.as_ref().unwrap().compression.as_ref().unwrap()
                        else {
                            panic!("Expected flat for RLE values")
                        };
                        let Compression::Flat(run_lengths) = rle
                            .run_lengths
                            .as_ref()
                            .unwrap()
                            .compression
                            .as_ref()
                            .unwrap()
                        else {
                            panic!("Expected flat for RLE run lengths")
                        };
                        assert_eq!(values.bits_per_value, 32);
                        assert_eq!(run_lengths.bits_per_value, 8);
                    }
                    _ => panic!("Expected RLE as inner encoding"),
                }
                // Check compression is LZ4
                assert_eq!(
                    cm.compression.as_ref().unwrap().scheme(),
                    pb21::CompressionScheme::CompressionAlgorithmLz4
                );
            }
            Some(Compression::Rle(_)) => {
                // Also acceptable if compression didn't help
            }
            _ => panic!("Expected GeneralMiniBlock or Rle encoding"),
        }

        // Test case 2: 64-bit RLE data
        let values_64: Vec<i64> = vec![100i64; 50]
            .into_iter()
            .chain(vec![200i64; 50])
            .chain(vec![300i64; 50])
            .collect();
        let array_64 = arrow_array::Int64Array::from(values_64);
        let block_64 = DataBlock::from_array(array_64);

        let compressor_64 = GeneralMiniBlockCompressor::new(
            Box::new(RleEncoder::new()),
            CompressionConfig {
                scheme: CompressionScheme::Lz4,
                level: None,
            },
        );

        let (_compressed_64, encoding_64) = compressor_64
            .compress(miniblock_context(), block_64)
            .unwrap();

        // Verify the encoding structure for 64-bit
        match &encoding_64.compression {
            Some(Compression::General(cm)) => {
                // Check inner encoding is RLE
                match &cm.values.as_ref().unwrap().compression {
                    Some(Compression::Rle(rle)) => {
                        let Compression::Flat(values) =
                            rle.values.as_ref().unwrap().compression.as_ref().unwrap()
                        else {
                            panic!("Expected flat for RLE values")
                        };
                        let Compression::Flat(run_lengths) = rle
                            .run_lengths
                            .as_ref()
                            .unwrap()
                            .compression
                            .as_ref()
                            .unwrap()
                        else {
                            panic!("Expected flat for RLE run lengths")
                        };
                        assert_eq!(values.bits_per_value, 64);
                        assert_eq!(run_lengths.bits_per_value, 8);
                    }
                    _ => panic!("Expected RLE as inner encoding for 64-bit"),
                }
                // Check compression is LZ4
                assert_eq!(
                    cm.compression.as_ref().unwrap().scheme(),
                    pb21::CompressionScheme::CompressionAlgorithmLz4
                );
            }
            Some(Compression::Rle(_)) => {
                // Also acceptable if compression didn't help
            }
            _ => panic!("Expected GeneralMiniBlock or Rle encoding for 64-bit"),
        }
    }

    #[test]
    fn test_compressed_mini_block_empty_data() {
        let empty_array = Int32Array::from(vec![] as Vec<i32>);
        let empty_block = DataBlock::from_array(empty_array);

        let compressor = GeneralMiniBlockCompressor::new(
            Box::new(ValueEncoder {}),
            CompressionConfig {
                scheme: CompressionScheme::Lz4,
                level: None,
            },
        );

        let result = compressor.compress(miniblock_context(), empty_block);
        match result {
            Ok((compressed, _)) => {
                assert_eq!(compressed.num_values, 0);
            }
            Err(_) => {
                // Empty data might not be supported by ValueEncoder
            }
        }
    }

    /// Odd, and wide enough that even `u8` items exceed the 256-byte full-zip cutoff, so values
    /// are never a multiple of a wider power-of-two item width.
    const WIDE_DIMENSION: usize = 257;
    const NUM_ROWS: usize = 100;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Validity {
        AllValid,
        NullRows,
        NullItems,
    }

    fn zstd_metadata() -> HashMap<String, String> {
        HashMap::from([("lance-encoding:compression".to_string(), "zstd".to_string())])
    }

    /// Items repeat in short runs that shift from row to row, so values compress but differ.
    fn compressible_fsl(
        item_type: &DataType,
        dimension: usize,
        num_rows: usize,
        validity: Validity,
    ) -> ArrayRef {
        let items = Int32Array::from_iter_values(
            (0..num_rows * dimension).map(|i| ((i / dimension + (i % dimension) / 8) % 16) as i32),
        );
        let items = arrow_cast::cast(&items, item_type).unwrap();
        // The cast allocates an all-valid null buffer, which the encoder would store as item
        // validity, so only keep one when items are meant to be null.
        let item_validity = (validity == Validity::NullItems)
            .then(|| NullBuffer::from((0..items.len()).map(|i| i % 7 != 0).collect::<Vec<_>>()));
        let items = make_array(
            items
                .to_data()
                .into_builder()
                .nulls(item_validity)
                .build()
                .unwrap(),
        );
        let row_validity = (validity == Validity::NullRows)
            .then(|| NullBuffer::from((0..num_rows).map(|row| row % 5 != 0).collect::<Vec<_>>()));
        // Decoded fixed-size-list items are always nullable.
        let item_field = Arc::new(Field::new("item", item_type.clone(), true));
        Arc::new(FixedSizeListArray::new(
            item_field,
            dimension as i32,
            items,
            row_validity,
        ))
    }

    /// Smooth 16-bit rasters with mild noise, the kind of data Blosc with shuffle compresses well.
    fn smooth_u16_fsl(num_rows: usize, dimension: usize) -> ArrayRef {
        let items = UInt16Array::from_iter_values((0..num_rows * dimension).map(|i| {
            let (row, item) = ((i / dimension) as f64, i % dimension);
            let (y, x) = ((item / 64) as f64, (item % 64) as f64);
            let noise = ((i as u64).wrapping_mul(2_654_435_761) >> 8) % 81;
            (2000.0 + 800.0 * ((y + row) / 9.0).sin() * ((x - row) / 13.0).cos()) as u16
                + noise as u16
        }));
        Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt16, false)),
            dimension as i32,
            Arc::new(items),
            None,
        ))
    }

    fn compress_per_value(
        array: &ArrayRef,
        scheme: CompressionScheme,
        byte_stream_split: BssMode,
    ) -> (PerValueDataBlock, CompressiveEncoding) {
        let compressor = GeneralFixedPerValueCompressor::try_new(
            CompressionConfig::new(scheme, None),
            byte_stream_split,
        )
        .unwrap();
        compressor
            .compress(DataBlock::from_array(array.clone()))
            .unwrap()
    }

    fn decompress_per_value(
        data: VariableWidthBlock,
        encoding: &CompressiveEncoding,
        data_type: &DataType,
    ) -> Result<ArrayRef> {
        let decompressor = DefaultDecompressionStrategy::default()
            .create_variable_per_value_decompressor(encoding)?;
        Ok(make_array(
            decompressor
                .decompress(data)?
                .into_arrow(data_type.clone(), true)?,
        ))
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_general_fixed_per_value_fsl_round_trip(
        #[values(
            DataType::UInt8,
            DataType::Int16,
            DataType::Float16,
            DataType::UInt32,
            DataType::Float32,
            DataType::Int64,
            DataType::Float64,
            DataType::Decimal128(38, 0)
        )]
        item_type: DataType,
        #[values(Validity::AllValid, Validity::NullRows, Validity::NullItems)] validity: Validity,
    ) {
        let array = compressible_fsl(&item_type, WIDE_DIMENSION, NUM_ROWS, validity);
        let mut expected_chain = vec!["general", "zstd", "fixed_size_list", "flat"];
        // Item validity is zipped in front of the items, so only null-free items are split.
        if item_type.primitive_width().is_some_and(|width| width > 1)
            && validity != Validity::NullItems
        {
            expected_chain.push("byte_stream_split");
        }
        let test_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_expected_encoding_chain(expected_chain)
            .with_range(37..61)
            .with_indices(vec![0])
            .with_indices(vec![59])
            .with_indices(vec![60])
            .with_indices(vec![NUM_ROWS as u64 - 1])
            .with_indices(vec![3, 41, 42, 98]);

        // Two arrays so takes cross a page boundary.
        check_round_trip_encoding_of_data(
            vec![array.slice(0, 60), array.slice(60, NUM_ROWS - 60)],
            &test_cases,
            zstd_metadata(),
        )
        .await;
    }

    #[rstest]
    #[case::lz4("lz4", "auto")]
    #[case::zstd_without_split("zstd", "off")]
    #[test_log::test(tokio::test)]
    async fn test_general_fixed_per_value_fsl_options(
        #[case] compression: &str,
        #[case] byte_stream_split: &str,
        #[values(3, WIDE_DIMENSION, 4096)] dimension: usize,
    ) {
        // Narrow values only reach full-zip when it is requested.
        let metadata = HashMap::from([
            (
                "lance-encoding:compression".to_string(),
                compression.to_string(),
            ),
            (
                "lance-encoding:bss".to_string(),
                byte_stream_split.to_string(),
            ),
            (
                "lance-encoding:structural-encoding".to_string(),
                "fullzip".to_string(),
            ),
        ]);
        let array = compressible_fsl(&DataType::Float32, dimension, NUM_ROWS, Validity::NullRows);
        let test_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_range(10..20)
            .with_indices(vec![0])
            .with_indices(vec![NUM_ROWS as u64 - 1])
            .with_indices(vec![5, 6, 50]);

        check_round_trip_encoding_of_data(vec![array], &test_cases, metadata).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_general_fixed_per_value_fixed_size_binary_round_trip() {
        let values = (0..NUM_ROWS)
            .map(|row| {
                (0..1024)
                    .map(|i| ((row + i / 16) % 8) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let array = Arc::new(FixedSizeBinaryArray::try_from_iter(values.into_iter()).unwrap());
        let test_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_expected_encoding_chain(["general", "zstd", "flat"])
            .with_indices(vec![0])
            .with_indices(vec![17, 18, 99]);

        check_round_trip_encoding_of_data(vec![array], &test_cases, zstd_metadata()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_general_fixed_per_value_variable_packed_struct_child() {
        let pixels = compressible_fsl(
            &DataType::UInt16,
            WIDE_DIMENSION,
            NUM_ROWS,
            Validity::AllValid,
        );
        let names = Arc::new(StringArray::from_iter_values(
            (0..NUM_ROWS).map(|row| format!("chip-{row}")),
        )) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("pixels", pixels.data_type().clone(), false).with_metadata(zstd_metadata()),
        ]);
        let array = Arc::new(StructArray::new(fields, vec![names, pixels], None)) as ArrayRef;
        let test_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_expected_encoding_chain([
                "variable_packed_struct",
                "general",
                "byte_stream_split",
            ])
            .with_indices(vec![0])
            .with_indices(vec![50, 99]);

        check_round_trip_encoding_of_data(
            vec![array],
            &test_cases,
            HashMap::from([("lance-encoding:packed".to_string(), "true".to_string())]),
        )
        .await;
    }

    #[test]
    fn test_general_fixed_per_value_compresses_smooth_values() {
        const DIMENSION: usize = 4096;
        let array = smooth_u16_fsl(16, DIMENSION);
        let raw_bytes = (array.len() * DIMENSION * 2) as f64;
        let mut ratios = Vec::with_capacity(2);
        for byte_stream_split in [BssMode::Off, BssMode::Auto] {
            let (data, encoding) =
                compress_per_value(&array, CompressionScheme::Zstd, byte_stream_split);
            let PerValueDataBlock::Variable(data) = data else {
                panic!("smooth values should compress with {byte_stream_split:?}");
            };
            let chain = extract_array_encoding_chain(&encoding);
            assert_eq!(
                chain.iter().any(|tag| tag == "byte_stream_split"),
                byte_stream_split == BssMode::Auto,
                "{chain:?}"
            );
            ratios.push((data.data.len() + 4 * data.num_values as usize) as f64 / raw_bytes);
            let decoded = decompress_per_value(data, &encoding, array.data_type()).unwrap();
            assert_eq!(decoded.as_ref(), array.as_ref());
        }
        let (unsplit, split) = (ratios[0], ratios[1]);
        assert!(split < 0.65, "split ratio {split}");
        assert!(split < unsplit - 0.1, "split {split} vs unsplit {unsplit}");
    }

    #[test]
    fn test_general_fixed_per_value_keeps_incompressible_values_raw() {
        // splitmix64 output has no structure for a general compressor to find.
        let items = UInt16Array::from_iter_values((0..8 * 1024_u64).map(|i| {
            let mut z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) as u16
        }));
        let array = Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt16, false)),
            1024,
            Arc::new(items),
            None,
        )) as ArrayRef;

        let (data, encoding) = compress_per_value(&array, CompressionScheme::Zstd, BssMode::Auto);
        assert!(matches!(data, PerValueDataBlock::Fixed(_)));
        assert_eq!(
            extract_array_encoding_chain(&encoding),
            ["fixed_size_list", "flat"]
        );
    }

    #[test]
    fn test_general_fixed_per_value_decodes_empty_slots_as_null_placeholders() {
        let array = compressible_fsl(&DataType::UInt16, WIDE_DIMENSION, 3, Validity::AllValid);
        let (data, encoding) = compress_per_value(&array, CompressionScheme::Lz4, BssMode::Auto);
        let PerValueDataBlock::Variable(data) = data else {
            panic!("values should compress");
        };
        // A full-zip page stores no payload for a null item, so its slot reaches the decoder empty.
        let offsets = data.offsets.borrow_to_typed_slice::<u32>();
        let (row_1_start, row_1_end) = (offsets[1] as usize, offsets[2] as usize);
        let mut payload = data.data[..row_1_start].to_vec();
        payload.extend_from_slice(&data.data[row_1_end..]);
        let payload_len = payload.len() as u32;
        let block = VariableWidthBlock {
            data: LanceBuffer::from(payload),
            offsets: LanceBuffer::reinterpret_vec(vec![
                0,
                row_1_start as u32,
                row_1_start as u32,
                payload_len,
            ]),
            bits_per_offset: 32,
            num_values: 3,
            block_info: BlockInfo::new(),
        };

        let decoded = decompress_per_value(block, &encoding, array.data_type()).unwrap();
        let expected = array.to_data();
        let decoded = decoded.to_data();
        let items = decoded.child_data()[0].buffer::<u16>(0);
        let expected_items = expected.child_data()[0].buffer::<u16>(0);
        assert_eq!(items[..WIDE_DIMENSION], expected_items[..WIDE_DIMENSION]);
        assert!(
            items[WIDE_DIMENSION..2 * WIDE_DIMENSION]
                .iter()
                .all(|item| *item == 0)
        );
        assert_eq!(
            items[2 * WIDE_DIMENSION..],
            expected_items[2 * WIDE_DIMENSION..]
        );
    }

    #[test]
    fn test_general_fixed_per_value_rejects_offsets_past_payload() {
        let array = compressible_fsl(&DataType::UInt16, WIDE_DIMENSION, 2, Validity::AllValid);
        let (data, encoding) = compress_per_value(&array, CompressionScheme::Zstd, BssMode::Auto);
        let PerValueDataBlock::Variable(data) = data else {
            panic!("values should compress");
        };
        let payload_len = data.data.len() as u32;
        let block = VariableWidthBlock {
            offsets: LanceBuffer::reinterpret_vec(vec![0, payload_len + 1, payload_len + 1]),
            ..data
        };

        let error = decompress_per_value(block, &encoding, array.data_type()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("exceed"), "{error}");
    }

    #[test]
    fn test_general_fixed_per_value_rejects_split_items_with_validity() {
        let description = pb21::General {
            compression: Some(pb21::BufferCompression {
                scheme: pb21::CompressionScheme::CompressionAlgorithmZstd as i32,
                level: None,
            }),
            values: Some(Box::new(ProtobufUtils21::byte_stream_split(
                ProtobufUtils21::fsl(4, true, ProtobufUtils21::flat(16, None)),
            ))),
        };

        let error = GeneralFixedPerValueDecompressor::try_new(&description).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("without item validity"),
            "{error}"
        );
    }
}
