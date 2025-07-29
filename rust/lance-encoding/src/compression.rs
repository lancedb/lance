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
    compression_config::{CompressionFieldParams, CompressionParams},
    data::{DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    encodings::{
        logical::primitive::{fullzip::PerValueCompressor, miniblock::MiniBlockCompressor},
        physical::{
            binary::{
                BinaryBlockDecompressor, BinaryMiniBlockDecompressor, BinaryMiniBlockEncoder,
                VariableDecoder, VariableEncoder,
            },
            bitpack::InlineBitpacking,
            block::{CompressedBufferEncoder, CompressionConfig, CompressionScheme},
            byte_stream_split::ByteStreamSplitDecompressor,
            constant::ConstantDecompressor,
            fsst::{
                FsstMiniBlockDecompressor, FsstMiniBlockEncoder, FsstPerValueDecompressor,
                FsstPerValueEncoder,
            },
            general::{GeneralMiniBlockCompressor, GeneralMiniBlockDecompressor},
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
    datatypes::{Field, COMPRESSION_META_KEY, RLE_THRESHOLD_META_KEY},
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
pub struct DefaultCompressionStrategy {
    /// Optional user-configured compression parameters
    params: Option<CompressionParams>,
}

impl DefaultCompressionStrategy {
    /// Create a new compression strategy with default behavior
    pub fn new() -> Self {
        Self { params: None }
    }

    /// Create a new compression strategy with user-configured parameters
    pub fn with_params(params: CompressionParams) -> Self {
        Self {
            params: Some(params),
        }
    }

    /// Build compressor based on parameters for fixed-width data
    fn build_fixed_width_compressor(
        &self,
        params: &CompressionFieldParams,
        field: &Field,
        data: &FixedWidthDataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        let bits_per_value = data.bits_per_value;
        let is_byte_aligned = bits_per_value == 8
            || bits_per_value == 16
            || bits_per_value == 32
            || bits_per_value == 64;

        // Get statistics
        let bit_widths = data.expect_stat(Stat::BitWidth);
        let bit_widths = bit_widths.as_primitive::<UInt64Type>();
        let has_all_zeros = bit_widths.values().iter().any(|v| *v == 0);
        let too_small = bit_widths.len() == 1
            && InlineBitpacking::min_size_bytes(bit_widths.value(0)) >= data.data_size();

        // 1. Check for explicit "none" compression
        if params.compression.as_deref() == Some("none") {
            return Ok(Box::new(ValueEncoder::default()));
        }

        // 2. Check metadata override (legacy support)
        if let Some(compression) = field.metadata.get(COMPRESSION_META_KEY) {
            if compression.as_str() == "none" {
                return Ok(Box::new(ValueEncoder::default()));
            }
        }

        // 3. Determine base encoder
        let mut base_encoder: Box<dyn MiniBlockCompressor> = {
            // Check if RLE should be used
            let rle_threshold = params
                .rle_threshold
                .or_else(|| {
                    // Check field metadata for legacy threshold
                    field
                        .metadata
                        .get(RLE_THRESHOLD_META_KEY)
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(DEFAULT_RLE_COMPRESSION_THRESHOLD);

            let run_count = data.expect_single_stat::<UInt64Type>(Stat::RunCount);
            let num_values = data.num_values;

            if (run_count as f64) < (num_values as f64) * rle_threshold && is_byte_aligned {
                Box::new(RleMiniBlockEncoder::new())
            } else if !has_all_zeros && !too_small && is_byte_aligned {
                // Use bitpacking if appropriate
                Box::new(InlineBitpacking::new(bits_per_value))
            } else {
                // Default to no compression for base layer
                Box::new(ValueEncoder::default())
            }
        };

        // 4. Apply general compression if configured
        if let Some(compression_scheme) = &params.compression {
            if compression_scheme != "none" {
                let scheme: CompressionScheme = compression_scheme.parse()?;
                let config = CompressionConfig::new(scheme, params.compression_level);
                base_encoder = Box::new(GeneralMiniBlockCompressor::new(base_encoder, config));
            }
        }

        Ok(base_encoder)
    }

    /// Build compressor based on parameters for variable-width data
    fn build_variable_width_compressor(
        &self,
        params: &CompressionFieldParams,
        data: &VariableWidthBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        if data.bits_per_offset != 32 && data.bits_per_offset != 64 {
            return Err(Error::invalid_input(
                format!(
                    "Variable width compression not supported for {} bit offsets",
                    data.bits_per_offset
                ),
                location!(),
            ));
        }

        // Get statistics
        let data_size = data.expect_single_stat::<UInt64Type>(Stat::DataSize);
        let max_len = data.expect_single_stat::<UInt64Type>(Stat::MaxLength);

        // 1. Check for explicit "none" compression
        if params.compression.as_deref() == Some("none") {
            return Ok(Box::new(BinaryMiniBlockEncoder::default()));
        }

        // 2. Choose base encoder (FSST or Binary)
        let mut base_encoder: Box<dyn MiniBlockCompressor> = if max_len
            >= FSST_LEAST_INPUT_MAX_LENGTH
            && data_size >= FSST_LEAST_INPUT_SIZE as u64
        {
            Box::new(FsstMiniBlockEncoder::default())
        } else {
            Box::new(BinaryMiniBlockEncoder::default())
        };

        // 3. Apply general compression if configured
        if let Some(compression_scheme) = &params.compression {
            if compression_scheme != "none" {
                let scheme: CompressionScheme = compression_scheme.parse()?;
                let config = CompressionConfig::new(scheme, params.compression_level);
                base_encoder = Box::new(GeneralMiniBlockCompressor::new(base_encoder, config));
            }
        }

        Ok(base_encoder)
    }
}

impl CompressionStrategy for DefaultCompressionStrategy {
    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        // If we have user parameters, use them first
        if let Some(params) = &self.params {
            let field_params = params.get_field_params(&field.name, &field.data_type());

            match data {
                DataBlock::FixedWidth(fixed_width_data) => {
                    return self.build_fixed_width_compressor(
                        &field_params,
                        field,
                        fixed_width_data,
                    );
                }
                DataBlock::VariableWidth(variable_width_data) => {
                    return self
                        .build_variable_width_compressor(&field_params, variable_width_data);
                }
                DataBlock::Struct(_) => {
                    // Struct compression doesn't use parameters currently
                    return Ok(Box::new(PackedStructFixedWidthMiniBlockEncoder::default()));
                }
                DataBlock::FixedSizeList(_) => {
                    // FSL doesn't support compression currently
                    return Ok(Box::new(ValueEncoder::default()));
                }
                _ => {
                    // Fall through to default behavior for unsupported types
                }
            }
        }

        // Default behavior (no user parameters)
        match data {
            DataBlock::FixedWidth(fixed_width_data) => {
                let is_byte_width_aligned = fixed_width_data.bits_per_value == 8
                    || fixed_width_data.bits_per_value == 16
                    || fixed_width_data.bits_per_value == 32
                    || fixed_width_data.bits_per_value == 64;
                let bit_widths = data.expect_stat(Stat::BitWidth);
                let bit_widths = bit_widths.as_primitive::<UInt64Type>();
                // Temporary hack to work around https://github.com/lancedb/lance/issues/3102
                // Ideally we should still be able to bit-pack here (either to 0 or 1 bit per value)
                let has_all_zeros = bit_widths.values().iter().any(|v| *v == 0);
                // The minimum bit packing size is a block of 1024 values.  For very small pages the uncompressed
                // size might be smaller than the compressed size.
                let too_small = bit_widths.len() == 1
                    && InlineBitpacking::min_size_bytes(bit_widths.value(0)) >= data.data_size();

                if let Some(compression) = field.metadata.get(COMPRESSION_META_KEY) {
                    if compression.as_str() == "none" {
                        return Ok(Box::new(ValueEncoder::default()));
                    }
                }

                let rle_threshold: f64 = if let Some(value) =
                    field.metadata.get(RLE_THRESHOLD_META_KEY)
                {
                    value.as_str().parse().map_err(|_| {
                        Error::invalid_input("rle threshold is not a valid float64", location!())
                    })?
                } else {
                    DEFAULT_RLE_COMPRESSION_THRESHOLD
                };

                // Check if RLE would be beneficial
                let run_count = data.expect_single_stat::<UInt64Type>(Stat::RunCount);
                let num_values = fixed_width_data.num_values;

                // Use RLE if the run count is less than the threshold
                if (run_count as f64) < (num_values as f64) * rle_threshold && is_byte_width_aligned
                {
                    if fixed_width_data.bits_per_value >= 32 {
                        return Ok(Box::new(GeneralMiniBlockCompressor::new(
                            Box::new(RleMiniBlockEncoder::new()),
                            CompressionConfig::new(CompressionScheme::Lz4, None),
                        )));
                    }
                    return Ok(Box::new(RleMiniBlockEncoder::new()));
                }

                if !has_all_zeros && !too_small && is_byte_width_aligned {
                    Ok(Box::new(InlineBitpacking::new(
                        fixed_width_data.bits_per_value,
                    )))
                } else {
                    Ok(Box::new(ValueEncoder::default()))
                }
            }
            DataBlock::VariableWidth(variable_width_data) => {
                if variable_width_data.bits_per_offset == 32
                    || variable_width_data.bits_per_offset == 64
                {
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

                if variable_width.bits_per_offset == 32 || variable_width.bits_per_offset == 64 {
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
                    panic!("Does not support MiniBlockCompression for VariableWidth DataBlock with {} bits offsets.", variable_width.bits_per_offset);
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
        decompression_strategy: &dyn DecompressionStrategy,
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
        decompression_strategy: &dyn DecompressionStrategy,
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
                let inner_decompressor = decompression_strategy.create_miniblock_decompressor(
                    description.binary.as_ref().unwrap(),
                    decompression_strategy,
                )?;
                Ok(Box::new(FsstMiniBlockDecompressor::new(
                    description,
                    inner_decompressor,
                )))
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
            pb::array_encoding::ArrayEncoding::ByteStreamSplit(bss) => Ok(Box::new(
                ByteStreamSplitDecompressor::new(bss.bits_per_value as usize),
            )),
            pb::array_encoding::ArrayEncoding::GeneralMiniBlock(general) => {
                // Create inner decompressor
                let inner_decompressor = self.create_miniblock_decompressor(
                    general.inner.as_ref().ok_or_else(|| {
                        Error::invalid_input("GeneralMiniBlock missing inner encoding", location!())
                    })?,
                    decompression_strategy,
                )?;

                // Parse compression config
                let compression = general.compression.as_ref().ok_or_else(|| {
                    Error::invalid_input("GeneralMiniBlock missing compression config", location!())
                })?;

                let scheme = compression.scheme.parse()?;

                let compression_config = crate::encodings::physical::block::CompressionConfig::new(
                    scheme,
                    compression.level,
                );

                Ok(Box::new(GeneralMiniBlockDecompressor::new(
                    inner_decompressor,
                    compression_config,
                )))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::LanceBuffer;
    use crate::data::{BlockInfo, DataBlock};
    use arrow::datatypes::{DataType, Field as ArrowField};
    use std::collections::HashMap;

    fn create_test_field(name: &str, data_type: DataType) -> Field {
        let arrow_field = ArrowField::new(name, data_type, true);
        let mut field = Field::try_from(&arrow_field).unwrap();
        field.id = -1;
        field
    }

    fn create_fixed_width_block_with_stats(
        bits_per_value: u64,
        num_values: u64,
        run_count: u64,
    ) -> DataBlock {
        let block = FixedWidthDataBlock {
            bits_per_value,
            data: LanceBuffer::reinterpret_vec(vec![
                0u8;
                (bits_per_value * num_values / 8) as usize
            ]),
            num_values,
            block_info: BlockInfo::default(),
        };

        // Add required statistics
        use crate::statistics::Stat;
        use arrow::array::{ArrayRef, UInt64Array};
        use std::sync::Arc;

        let bit_widths = Arc::new(UInt64Array::from(vec![bits_per_value])) as ArrayRef;
        let run_count_stat = Arc::new(UInt64Array::from(vec![run_count])) as ArrayRef;

        block
            .block_info
            .0
            .write()
            .unwrap()
            .insert(Stat::BitWidth, bit_widths);
        block
            .block_info
            .0
            .write()
            .unwrap()
            .insert(Stat::RunCount, run_count_stat);

        DataBlock::FixedWidth(block)
    }

    fn create_fixed_width_block(bits_per_value: u64, num_values: u64) -> DataBlock {
        // Default run_count is num_values / 4
        create_fixed_width_block_with_stats(bits_per_value, num_values, num_values / 4)
    }

    #[test]
    fn test_parameter_based_compression() {
        let mut params = CompressionParams::new();

        // Configure RLE for ID columns
        params.columns.insert(
            "*_id".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.3),
                compression: Some("lz4".to_string()),
                compression_level: None,
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);
        let field = create_test_field("user_id", DataType::Int32);
        let data = create_fixed_width_block(32, 1000);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should use RLE due to low threshold
        assert!(format!("{:?}", compressor).contains("RleMiniBlockEncoder"));
    }

    #[test]
    fn test_type_level_parameters() {
        let mut params = CompressionParams::new();

        // Configure all Int32 to use specific settings
        params.types.insert(
            "Int32".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.1), // Very low threshold
                compression: Some("zstd".to_string()),
                compression_level: Some(3),
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);
        let field = create_test_field("some_column", DataType::Int32);
        // Create data with very low run count (50 runs for 1000 values = 0.05 ratio)
        let data = create_fixed_width_block_with_stats(32, 1000, 50);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should use RLE due to very low threshold
        assert!(format!("{:?}", compressor).contains("RleMiniBlockEncoder"));
    }

    #[test]
    fn test_none_compression() {
        let mut params = CompressionParams::new();

        // Disable compression for embeddings
        params.columns.insert(
            "embeddings".to_string(),
            CompressionFieldParams {
                compression: Some("none".to_string()),
                ..Default::default()
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);
        let field = create_test_field("embeddings", DataType::Float32);
        let data = create_fixed_width_block(32, 1000);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should use ValueEncoder (no compression)
        assert!(format!("{:?}", compressor).contains("ValueEncoder"));
    }

    #[test]
    fn test_parameter_merge_priority() {
        let mut params = CompressionParams::new();

        // Set type-level
        params.types.insert(
            "Int32".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.5),
                compression: Some("lz4".to_string()),
                ..Default::default()
            },
        );

        // Set column-level (highest priority)
        params.columns.insert(
            "user_id".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.2),
                compression: Some("zstd".to_string()),
                compression_level: Some(6),
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);

        // Get merged params
        let merged = strategy
            .params
            .as_ref()
            .unwrap()
            .get_field_params("user_id", &DataType::Int32);

        // Column params should override type params
        assert_eq!(merged.rle_threshold, Some(0.2));
        assert_eq!(merged.compression, Some("zstd".to_string()));
        assert_eq!(merged.compression_level, Some(6));

        // Test field with only type params
        let merged = strategy
            .params
            .as_ref()
            .unwrap()
            .get_field_params("other_field", &DataType::Int32);
        assert_eq!(merged.rle_threshold, Some(0.5));
        assert_eq!(merged.compression, Some("lz4".to_string()));
        assert_eq!(merged.compression_level, None);
    }

    #[test]
    fn test_pattern_matching() {
        let mut params = CompressionParams::new();

        // Configure pattern for log files
        params.columns.insert(
            "log_*".to_string(),
            CompressionFieldParams {
                compression: Some("zstd".to_string()),
                compression_level: Some(6),
                ..Default::default()
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);

        // Should match pattern
        let merged = strategy
            .params
            .as_ref()
            .unwrap()
            .get_field_params("log_messages", &DataType::Utf8);
        assert_eq!(merged.compression, Some("zstd".to_string()));
        assert_eq!(merged.compression_level, Some(6));

        // Should not match
        let merged = strategy
            .params
            .as_ref()
            .unwrap()
            .get_field_params("messages_log", &DataType::Utf8);
        assert_eq!(merged.compression, None);
    }

    #[test]
    fn test_legacy_metadata_support() {
        let params = CompressionParams::new();
        let strategy = DefaultCompressionStrategy::with_params(params);

        // Test field with "none" compression metadata
        let mut metadata = HashMap::new();
        metadata.insert(COMPRESSION_META_KEY.to_string(), "none".to_string());
        let mut field = create_test_field("some_column", DataType::Int32);
        field.metadata = metadata;

        let data = create_fixed_width_block(32, 1000);
        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();

        // Should respect metadata and use ValueEncoder
        assert!(format!("{:?}", compressor).contains("ValueEncoder"));
    }

    #[test]
    fn test_default_behavior() {
        // Empty params should fall back to default behavior
        let params = CompressionParams::new();
        let strategy = DefaultCompressionStrategy::with_params(params);

        let field = create_test_field("random_column", DataType::Int32);
        // Create data with high run count that won't trigger RLE (600 runs for 1000 values = 0.6 ratio)
        let data = create_fixed_width_block_with_stats(32, 1000, 600);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should use default strategy's decision
        let debug_str = format!("{:?}", compressor);
        assert!(debug_str.contains("ValueEncoder") || debug_str.contains("InlineBitpacking"));
    }
}
