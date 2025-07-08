// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compression traits and definitions for Lance 2.1
//!
//! In 2.1 the first step of encoding is structural encoding, where we shred inputs into
//! leaf arrays and take care of the validity / offsets structure.  Then we pick a structural
//! encoding (mini-block or full-zip) and then we compress the data.
//!
//! This module defines the traits for the compression step.  Each structural encoding has its
//! own compression strategy.
//!
//! Miniblock compression is a block based approach for small data.  Since we introduce some read
//! amplification and decompress entire blocks we are able to use opaque compression.
//!
//! Fullzip compression is a per-value approach where we require that values are transparently
//! compressed so that we can locate them later.

/// Default threshold for RLE compression selection.
/// RLE is chosen when the run count is less than this fraction of total values.
const DEFAULT_RLE_COMPRESSION_THRESHOLD: f64 = 0.5;

use crate::{
    buffer::LanceBuffer,
    data::{DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    encodings::{
        logical::primitive::{fullzip::PerValueCompressor, miniblock::MiniBlockCompressor},
        physical::{
            binary::{
                BinaryBlockDecompressor, BinaryMiniBlockDecompressor, BinaryMiniBlockEncoder,
                VariableDecoder, VariableEncoder,
            },
            bitpack::InlineBitpacking,
            block::CompressedBufferEncoder,
            constant::ConstantDecompressor,
            fsst::{
                FsstMiniBlockDecompressor, FsstMiniBlockEncoder, FsstPerValueDecompressor,
                FsstPerValueEncoder,
            },
            packed::{
                PackedStructFixedWidthMiniBlockDecompressor, PackedStructFixedWidthMiniBlockEncoder,
            },
            rle::{RleMiniBlockDecompressor, RleMiniBlockEncoder},
            value::{ValueDecompressor, ValueEncoder},
        },
    },
    format::{pb, ProtobufUtils},
    statistics::{GetStat, Stat},
};

use arrow::{array::AsArray, datatypes::UInt64Type};
use fsst::fsst::{FSST_LEAST_INPUT_MAX_LENGTH, FSST_LEAST_INPUT_SIZE};
use lance_core::{
    datatypes::{Field, COMPRESSION_META_KEY},
    Error, Result,
};
use snafu::location;

/// Trait for compression algorithms that compress an entire block of data into one opaque
/// and self-described chunk.
///
/// This is actually a _third_ compression strategy used in a few corner cases today (TODO: remove?)
///
/// This is the most general type of compression.  There are no constraints on the method
/// of compression it is assumed that the entire block of data will be present at decompression.
///
/// This is the least appropriate strategy for random access because we must load the entire
/// block to access any single value.  This should only be used for cases where random access is never
/// required (e.g. when encoding metadata buffers like a dictionary or for encoding rep/def
/// mini-block chunks)
pub trait BlockCompressor: std::fmt::Debug + Send + Sync {
    /// Compress the data into a single buffer
    ///
    /// Also returns a description of the compression that can be used to decompress
    /// when reading the data back
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer>;
}

/// A trait to pick which compression to use for given data
///
/// There are several different kinds of compression.
///
/// - Block compression is the most generic, but most difficult to use efficiently
/// - Per-value compression results in either a fixed width data block or a variable
///   width data block.  In other words, there is some number of bits per value.
///   In addition, each value should be independently decompressible.
/// - Mini-block compression results in a small block of opaque data for chunks
///   of rows.  Each block is somewhere between 0 and 16KiB in size.  This is
///   used for narrow data types (both fixed and variable length) where we can
///   fit many values into an 16KiB block.
pub trait CompressionStrategy: Send + Sync + std::fmt::Debug {
    /// Create a block compressor for the given data
    fn create_block_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<(Box<dyn BlockCompressor>, pb::ArrayEncoding)>;

    /// Create a per-value compressor for the given data
    fn create_per_value(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>>;

    /// Create a mini-block compressor for the given data
    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>>;
}

#[derive(Debug, Default)]
pub struct DefaultCompressionStrategy;

impl CompressionStrategy for DefaultCompressionStrategy {
    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        match data {
            DataBlock::FixedWidth(fixed_width_data) => {
                if let Some(compression) = field.metadata.get(COMPRESSION_META_KEY) {
                    if compression == "none" {
                        return Ok(Box::new(ValueEncoder::default()));
                    }
                }

                // Check if RLE would be beneficial
                let run_count = data.expect_single_stat::<UInt64Type>(Stat::RunCount);
                let num_values = fixed_width_data.num_values;

                // Use RLE if the run count is less than the threshold
                if (run_count as f64) < (num_values as f64) * DEFAULT_RLE_COMPRESSION_THRESHOLD
                    && (fixed_width_data.bits_per_value == 8
                        || fixed_width_data.bits_per_value == 16
                        || fixed_width_data.bits_per_value == 32
                        || fixed_width_data.bits_per_value == 64
                        || fixed_width_data.bits_per_value == 128)
                {
                    return Ok(Box::new(RleMiniBlockEncoder::new()));
                }

                let bit_widths = data.expect_stat(Stat::BitWidth);
                let bit_widths = bit_widths.as_primitive::<UInt64Type>();
                // Temporary hack to work around https://github.com/lancedb/lance/issues/3102
                // Ideally we should still be able to bit-pack here (either to 0 or 1 bit per value)
                let has_all_zeros = bit_widths.values().iter().any(|v| *v == 0);
                // The minimum bit packing size is a block of 1024 values.  For very small pages the uncompressed
                // size might be smaller than the compressed size.
                let too_small = bit_widths.len() == 1
                    && InlineBitpacking::min_size_bytes(bit_widths.value(0)) >= data.data_size();
                if !has_all_zeros
                    && !too_small
                    && (fixed_width_data.bits_per_value == 8
                        || fixed_width_data.bits_per_value == 16
                        || fixed_width_data.bits_per_value == 32
                        || fixed_width_data.bits_per_value == 64)
                {
                    Ok(Box::new(InlineBitpacking::new(
                        fixed_width_data.bits_per_value,
                    )))
                } else {
                    Ok(Box::new(ValueEncoder::default()))
                }
            }
            DataBlock::VariableWidth(variable_width_data) => {
                if variable_width_data.bits_per_offset == 32 {
                    let data_size =
                        variable_width_data.expect_single_stat::<UInt64Type>(Stat::DataSize);
                    let max_len =
                        variable_width_data.expect_single_stat::<UInt64Type>(Stat::MaxLength);

                    if max_len >= FSST_LEAST_INPUT_MAX_LENGTH
                        && data_size >= FSST_LEAST_INPUT_SIZE as u64
                    {
                        Ok(Box::new(FsstMiniBlockEncoder::default()))
                    } else {
                        Ok(Box::new(BinaryMiniBlockEncoder::default()))
                    }
                } else if variable_width_data.bits_per_offset == 64 {
                    // TODO: Support FSSTMiniBlockEncoder
                    Ok(Box::new(BinaryMiniBlockEncoder::default()))
                } else {
                    todo!(
                        "Implement MiniBlockCompression for VariableWidth DataBlock with {} bit offsets.",
                        variable_width_data.bits_per_offset
                    )
                }
            }
            DataBlock::Struct(struct_data_block) => {
                // this condition is actually checked at `PrimitiveStructuralEncoder::do_flush`,
                // just being cautious here.
                if struct_data_block
                    .children
                    .iter()
                    .any(|child| !matches!(child, DataBlock::FixedWidth(_)))
                {
                    panic!("packed struct encoding currently only supports fixed-width fields.")
                }
                Ok(Box::new(PackedStructFixedWidthMiniBlockEncoder::default()))
            }
            DataBlock::FixedSizeList(_) => {
                // Ideally we would compress the list items but this creates something of a challenge.
                // We don't want to break lists across chunks and we need to worry about inner validity
                // layers.  If we try and use a compression scheme then it is unlikely to respect these
                // constraints.
                //
                // For now, we just don't compress.  In the future, we might want to consider a more
                // sophisticated approach.
                Ok(Box::new(ValueEncoder::default()))
            }
            _ => Err(Error::NotSupported {
                source: format!(
                    "Mini-block compression not yet supported for block type {}",
                    data.name()
                )
                .into(),
                location: location!(),
            }),
        }
    }

    fn create_per_value(
        &self,
        _field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>> {
        match data {
            DataBlock::FixedWidth(_) => Ok(Box::new(ValueEncoder::default())),
            DataBlock::FixedSizeList(_) => Ok(Box::new(ValueEncoder::default())),
            DataBlock::VariableWidth(variable_width) => {
                let max_len = variable_width.expect_single_stat::<UInt64Type>(Stat::MaxLength);
                let data_size = variable_width.expect_single_stat::<UInt64Type>(Stat::DataSize);

                // If values are very large then use block compression on a per-value basis
                //
                // TODO: Could maybe use median here
                if max_len > 32 * 1024 && data_size >= FSST_LEAST_INPUT_SIZE as u64 {
                    return Ok(Box::new(CompressedBufferEncoder::default()));
                }

                if variable_width.bits_per_offset == 32 {
                    let data_size = variable_width.expect_single_stat::<UInt64Type>(Stat::DataSize);
                    let max_len = variable_width.expect_single_stat::<UInt64Type>(Stat::MaxLength);

                    let variable_compression = Box::new(VariableEncoder::default());

                    if max_len >= FSST_LEAST_INPUT_MAX_LENGTH
                        && data_size >= FSST_LEAST_INPUT_SIZE as u64
                    {
                        Ok(Box::new(FsstPerValueEncoder::new(variable_compression)))
                    } else {
                        Ok(variable_compression)
                    }
                } else {
                    todo!("Implement MiniBlockCompression for VariableWidth DataBlock with 64 bits offsets.")
                }
            }
            _ => unreachable!(
                "Per-value compression not yet supported for block type: {}",
                data.name()
            ),
        }
    }

    fn create_block_compressor(
        &self,
        _field: &Field,
        data: &DataBlock,
    ) -> Result<(Box<dyn BlockCompressor>, pb::ArrayEncoding)> {
        // TODO: We should actually compress here!
        match data {
            // Currently, block compression is used for rep/def (which is fixed width) and for dictionary
            // encoding (which could be fixed width or variable width).
            DataBlock::FixedWidth(fixed_width) => {
                let encoder = Box::new(ValueEncoder::default());
                let encoding = ProtobufUtils::flat_encoding(fixed_width.bits_per_value, 0, None);
                Ok((encoder, encoding))
            }
            DataBlock::VariableWidth(variable_width) => {
                let encoder = Box::new(VariableEncoder::default());
                let encoding = ProtobufUtils::variable(variable_width.bits_per_offset);
                Ok((encoder, encoding))
            }
            _ => unreachable!(),
        }
    }
}

pub trait MiniBlockDecompressor: std::fmt::Debug + Send + Sync {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock>;
}

pub trait FixedPerValueDecompressor: std::fmt::Debug + Send + Sync {
    /// Decompress one or more values
    fn decompress(&self, data: FixedWidthDataBlock, num_values: u64) -> Result<DataBlock>;
    /// The number of bits in each value
    ///
    /// Currently (and probably long term) this must be a multiple of 8
    fn bits_per_value(&self) -> u64;
}

pub trait VariablePerValueDecompressor: std::fmt::Debug + Send + Sync {
    /// Decompress one or more values
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock>;
}

pub trait BlockDecompressor: std::fmt::Debug + Send + Sync {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock>;
}

pub trait DecompressionStrategy: std::fmt::Debug + Send + Sync {
    fn create_miniblock_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn MiniBlockDecompressor>>;

    fn create_fixed_per_value_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn FixedPerValueDecompressor>>;

    fn create_variable_per_value_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn VariablePerValueDecompressor>>;

    fn create_block_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn BlockDecompressor>>;
}

#[derive(Debug, Default)]
pub struct DefaultDecompressionStrategy {}

impl DecompressionStrategy for DefaultDecompressionStrategy {
    fn create_miniblock_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn MiniBlockDecompressor>> {
        match description.array_encoding.as_ref().unwrap() {
            pb::array_encoding::ArrayEncoding::Flat(flat) => {
                Ok(Box::new(ValueDecompressor::from_flat(flat)))
            }
            pb::array_encoding::ArrayEncoding::InlineBitpacking(description) => {
                Ok(Box::new(InlineBitpacking::from_description(description)))
            }
            pb::array_encoding::ArrayEncoding::Variable(variable) => Ok(Box::new(
                BinaryMiniBlockDecompressor::new(variable.bits_per_offset as u8),
            )),
            pb::array_encoding::ArrayEncoding::Fsst(description) => {
                Ok(Box::new(FsstMiniBlockDecompressor::new(description)))
            }
            pb::array_encoding::ArrayEncoding::PackedStructFixedWidthMiniBlock(description) => {
                Ok(Box::new(PackedStructFixedWidthMiniBlockDecompressor::new(
                    description,
                )))
            }
            pb::array_encoding::ArrayEncoding::FixedSizeList(fsl) => {
                // In the future, we might need to do something more complex here if FSL supports
                // compression.
                Ok(Box::new(ValueDecompressor::from_fsl(fsl)))
            }
            pb::array_encoding::ArrayEncoding::Rle(rle) => {
                Ok(Box::new(RleMiniBlockDecompressor::new(rle.bits_per_value)))
            }
            _ => todo!(),
        }
    }

    fn create_fixed_per_value_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn FixedPerValueDecompressor>> {
        match description.array_encoding.as_ref().unwrap() {
            pb::array_encoding::ArrayEncoding::Flat(flat) => {
                Ok(Box::new(ValueDecompressor::from_flat(flat)))
            }
            pb::array_encoding::ArrayEncoding::FixedSizeList(fsl) => {
                Ok(Box::new(ValueDecompressor::from_fsl(fsl)))
            }
            _ => todo!("fixed-per-value decompressor for {:?}", description),
        }
    }

    fn create_variable_per_value_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn VariablePerValueDecompressor>> {
        match *description.array_encoding.as_ref().unwrap() {
            pb::array_encoding::ArrayEncoding::Variable(variable) => {
                assert!(variable.bits_per_offset < u8::MAX as u32);
                Ok(Box::new(VariableDecoder::default()))
            }
            pb::array_encoding::ArrayEncoding::Fsst(ref fsst) => {
                Ok(Box::new(FsstPerValueDecompressor::new(
                    LanceBuffer::from_bytes(fsst.symbol_table.clone(), 1),
                    Box::new(VariableDecoder::default()),
                )))
            }
            pb::array_encoding::ArrayEncoding::Block(ref block) => Ok(Box::new(
                CompressedBufferEncoder::from_scheme(&block.scheme)?,
            )),
            _ => todo!("variable-per-value decompressor for {:?}", description),
        }
    }

    fn create_block_decompressor(
        &self,
        description: &pb::ArrayEncoding,
    ) -> Result<Box<dyn BlockDecompressor>> {
        match description.array_encoding.as_ref().unwrap() {
            pb::array_encoding::ArrayEncoding::Flat(flat) => {
                Ok(Box::new(ValueDecompressor::from_flat(flat)))
            }
            pb::array_encoding::ArrayEncoding::Constant(constant) => {
                let scalar = LanceBuffer::from_bytes(constant.value.clone(), 1);
                Ok(Box::new(ConstantDecompressor::new(scalar)))
            }
            pb::array_encoding::ArrayEncoding::Variable(_) => {
                Ok(Box::new(BinaryBlockDecompressor::default()))
            }
            _ => todo!(),
        }
    }
}
