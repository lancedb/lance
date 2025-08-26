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

#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::InlineBitpacking;
use crate::{
    buffer::LanceBuffer,
    compression_config::{BssMode, CompressionFieldParams, CompressionParams},
    constants::{
        BSS_META_KEY, COMPRESSION_LEVEL_META_KEY, COMPRESSION_META_KEY, RLE_THRESHOLD_META_KEY,
    },
    data::{DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    encodings::{
        logical::primitive::{fullzip::PerValueCompressor, miniblock::MiniBlockCompressor},
        physical::{
            binary::{
                BinaryBlockDecompressor, BinaryMiniBlockDecompressor, BinaryMiniBlockEncoder,
                VariableDecoder, VariableEncoder,
            },
            block::{CompressedBufferEncoder, CompressionConfig, CompressionScheme},
            byte_stream_split::{
                should_use_bss, ByteStreamSplitDecompressor, ByteStreamSplitEncoder,
            },
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
    format::{
        pb21::{compressive_encoding::Compression, CompressiveEncoding},
        ProtobufUtils21,
    },
    statistics::{GetStat, Stat},
};

use arrow_array::types::UInt64Type;
use fsst::fsst::{FSST_LEAST_INPUT_MAX_LENGTH, FSST_LEAST_INPUT_SIZE};
use lance_core::{datatypes::Field, error::LanceOptionExt, Error, Result};
use snafu::location;
use std::str::FromStr;

/// Default threshold for RLE compression selection.
/// RLE is chosen when the run count is less than this fraction of total values.
const DEFAULT_RLE_COMPRESSION_THRESHOLD: f64 = 0.5;

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
    ) -> Result<(Box<dyn BlockCompressor>, CompressiveEncoding)>;

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
    /// User-configured compression parameters
    params: CompressionParams,
}

fn try_bss_for_mini_block(
    data: &FixedWidthDataBlock,
    params: &CompressionFieldParams,
) -> Option<Box<dyn MiniBlockCompressor>> {
    // BSS requires general compression to be effective
    // If compression is not set or explicitly disabled, skip BSS
    if params.compression.is_none() || params.compression.as_deref() == Some("none") {
        return None;
    }

    let mode = params.bss.unwrap_or(BssMode::Auto);
    // should_use_bss already checks for supported bit widths (32/64)
    if should_use_bss(data, mode) {
        return Some(Box::new(ByteStreamSplitEncoder::new(
            data.bits_per_value as usize,
        )));
    }
    None
}

fn try_rle_for_mini_block(
    data: &FixedWidthDataBlock,
    params: &CompressionFieldParams,
) -> Option<Box<dyn MiniBlockCompressor>> {
    let bits = data.bits_per_value;
    if !matches!(bits, 8 | 16 | 32 | 64) {
        return None;
    }

    let run_count = data.expect_single_stat::<UInt64Type>(Stat::RunCount);
    let threshold = params
        .rle_threshold
        .unwrap_or(DEFAULT_RLE_COMPRESSION_THRESHOLD);

    if (run_count as f64) < (data.num_values as f64) * threshold {
        return Some(Box::new(RleMiniBlockEncoder::new()));
    }
    None
}

fn try_bitpack_for_mini_block(_data: &FixedWidthDataBlock) -> Option<Box<dyn MiniBlockCompressor>> {
    #[cfg(feature = "bitpacking")]
    {
        use arrow_array::cast::AsArray;

        let bits = _data.bits_per_value;
        if !matches!(bits, 8 | 16 | 32 | 64) {
            return None;
        }

        let bit_widths = _data.expect_stat(Stat::BitWidth);
        let widths = bit_widths.as_primitive::<UInt64Type>();
        let has_all_zeros = widths.values().contains(&0);
        let too_small = widths.len() == 1
            && InlineBitpacking::min_size_bytes(widths.value(0)) >= _data.data_size();

        if !has_all_zeros && !too_small {
            return Some(Box::new(InlineBitpacking::new(bits)));
        }
        None
    }
    #[cfg(not(feature = "bitpacking"))]
    {
        None
    }
}

fn maybe_wrap_general_for_mini_block(
    inner: Box<dyn MiniBlockCompressor>,
    params: &CompressionFieldParams,
) -> Result<Box<dyn MiniBlockCompressor>> {
    match params.compression.as_deref() {
        None | Some("none") | Some("fsst") => Ok(inner),
        Some(raw) => {
            let scheme = CompressionScheme::from_str(raw).map_err(|_| {
                lance_core::Error::invalid_input(
                    format!("Unknown compression scheme: {raw}"),
                    location!(),
                )
            })?;
            let cfg = CompressionConfig::new(scheme, params.compression_level);
            Ok(Box::new(GeneralMiniBlockCompressor::new(inner, cfg)))
        }
    }
}

impl DefaultCompressionStrategy {
    /// Create a new compression strategy with default behavior
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new compression strategy with user-configured parameters
    pub fn with_params(params: CompressionParams) -> Self {
        Self { params }
    }

    /// Parse compression parameters from field metadata
    fn parse_field_metadata(field: &Field) -> CompressionFieldParams {
        let mut params = CompressionFieldParams::default();

        // Parse compression method
        if let Some(compression) = field.metadata.get(COMPRESSION_META_KEY) {
            params.compression = Some(compression.clone());
        }

        // Parse compression level
        if let Some(level) = field.metadata.get(COMPRESSION_LEVEL_META_KEY) {
            params.compression_level = level.parse().ok();
        }

        // Parse RLE threshold
        if let Some(threshold) = field.metadata.get(RLE_THRESHOLD_META_KEY) {
            params.rle_threshold = threshold.parse().ok();
        }

        // Parse BSS mode
        if let Some(bss_str) = field.metadata.get(BSS_META_KEY) {
            match BssMode::parse(bss_str) {
                Some(mode) => params.bss = Some(mode),
                None => {
                    log::warn!("Invalid BSS mode '{}', using default", bss_str);
                }
            }
        }

        params
    }

    fn build_fixed_width_compressor(
        &self,
        params: &CompressionFieldParams,
        data: &FixedWidthDataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        if params.compression.as_deref() == Some("none") {
            return Ok(Box::new(ValueEncoder::default()));
        }

        let base = try_bss_for_mini_block(data, params)
            .or_else(|| try_rle_for_mini_block(data, params))
            .or_else(|| try_bitpack_for_mini_block(data))
            .unwrap_or_else(|| Box::new(ValueEncoder::default()));

        maybe_wrap_general_for_mini_block(base, params)
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

        // 2. Check for explicit "fsst" compression
        if params.compression.as_deref() == Some("fsst") {
            return Ok(Box::new(FsstMiniBlockEncoder::default()));
        }

        // 3. Choose base encoder (FSST or Binary) based on data characteristics
        let mut base_encoder: Box<dyn MiniBlockCompressor> = if max_len
            >= FSST_LEAST_INPUT_MAX_LENGTH
            && data_size >= FSST_LEAST_INPUT_SIZE as u64
        {
            Box::new(FsstMiniBlockEncoder::default())
        } else {
            Box::new(BinaryMiniBlockEncoder::default())
        };

        // 4. Apply general compression if configured
        if let Some(compression_scheme) = &params.compression {
            if compression_scheme != "none" && compression_scheme != "fsst" {
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
        let mut field_params = self
            .params
            .get_field_params(&field.name, &field.data_type());

        // Override with field metadata if present (highest priority)
        let metadata_params = Self::parse_field_metadata(field);
        field_params.merge(&metadata_params);

        match data {
            DataBlock::FixedWidth(fixed_width_data) => {
                self.build_fixed_width_compressor(&field_params, fixed_width_data)
            }
            DataBlock::VariableWidth(variable_width_data) => {
                self.build_variable_width_compressor(&field_params, variable_width_data)
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
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>> {
        let field_params = Self::parse_field_metadata(field);

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

                    // Use FSST if explicitly requested or if data characteristics warrant it
                    if field_params.compression.as_deref() == Some("fsst")
                        || (max_len >= FSST_LEAST_INPUT_MAX_LENGTH
                            && data_size >= FSST_LEAST_INPUT_SIZE as u64)
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
    ) -> Result<(Box<dyn BlockCompressor>, CompressiveEncoding)> {
        // TODO: We should actually compress here!
        match data {
            // Currently, block compression is used for rep/def (which is fixed width) and for dictionary
            // encoding (which could be fixed width or variable width).
            DataBlock::FixedWidth(fixed_width) => {
                let encoder = Box::new(ValueEncoder::default());
                let encoding = ProtobufUtils21::flat(fixed_width.bits_per_value, None);
                Ok((encoder, encoding))
            }
            DataBlock::VariableWidth(variable_width) => {
                let encoder = Box::new(VariableEncoder::default());
                let encoding = ProtobufUtils21::variable(
                    ProtobufUtils21::flat(variable_width.bits_per_offset as u64, None),
                    None,
                );
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
        description: &CompressiveEncoding,
        decompression_strategy: &dyn DecompressionStrategy,
    ) -> Result<Box<dyn MiniBlockDecompressor>>;

    fn create_fixed_per_value_decompressor(
        &self,
        description: &CompressiveEncoding,
    ) -> Result<Box<dyn FixedPerValueDecompressor>>;

    fn create_variable_per_value_decompressor(
        &self,
        description: &CompressiveEncoding,
    ) -> Result<Box<dyn VariablePerValueDecompressor>>;

    fn create_block_decompressor(
        &self,
        description: &CompressiveEncoding,
    ) -> Result<Box<dyn BlockDecompressor>>;
}

#[derive(Debug, Default)]
pub struct DefaultDecompressionStrategy {}

impl DecompressionStrategy for DefaultDecompressionStrategy {
    fn create_miniblock_decompressor(
        &self,
        description: &CompressiveEncoding,
        decompression_strategy: &dyn DecompressionStrategy,
    ) -> Result<Box<dyn MiniBlockDecompressor>> {
        match description.compression.as_ref().unwrap() {
            Compression::Flat(flat) => Ok(Box::new(ValueDecompressor::from_flat(flat))),
            #[cfg(feature = "bitpacking")]
            Compression::InlineBitpacking(description) => {
                Ok(Box::new(InlineBitpacking::from_description(description)))
            }
            #[cfg(not(feature = "bitpacking"))]
            Compression::InlineBitpacking(_) => Err(Error::NotSupported {
                source: "this runtime was not built with bitpacking support".into(),
                location: location!(),
            }),
            Compression::Variable(variable) => {
                let Compression::Flat(offsets) = variable
                    .offsets
                    .as_ref()
                    .unwrap()
                    .compression
                    .as_ref()
                    .unwrap()
                else {
                    panic!("Variable compression only supports flat offsets")
                };
                Ok(Box::new(BinaryMiniBlockDecompressor::new(
                    offsets.bits_per_value as u8,
                )))
            }
            Compression::Fsst(description) => {
                let inner_decompressor = decompression_strategy.create_miniblock_decompressor(
                    description.values.as_ref().unwrap(),
                    decompression_strategy,
                )?;
                Ok(Box::new(FsstMiniBlockDecompressor::new(
                    description,
                    inner_decompressor,
                )))
            }
            Compression::PackedStruct(description) => Ok(Box::new(
                PackedStructFixedWidthMiniBlockDecompressor::new(description),
            )),
            Compression::FixedSizeList(fsl) => {
                // In the future, we might need to do something more complex here if FSL supports
                // compression.
                Ok(Box::new(ValueDecompressor::from_fsl(fsl)))
            }
            Compression::Rle(rle) => {
                let Compression::Flat(values) =
                    rle.values.as_ref().unwrap().compression.as_ref().unwrap()
                else {
                    panic!("RLE compression only supports flat values")
                };
                let Compression::Flat(run_lengths) = rle
                    .run_lengths
                    .as_ref()
                    .unwrap()
                    .compression
                    .as_ref()
                    .unwrap()
                else {
                    panic!("RLE compression only supports flat run lengths")
                };
                assert_eq!(
                    run_lengths.bits_per_value, 8,
                    "RLE compression only supports 8-bit run lengths"
                );
                Ok(Box::new(RleMiniBlockDecompressor::new(
                    values.bits_per_value,
                )))
            }
            Compression::ByteStreamSplit(bss) => {
                let Compression::Flat(values) =
                    bss.values.as_ref().unwrap().compression.as_ref().unwrap()
                else {
                    panic!("ByteStreamSplit compression only supports flat values")
                };
                Ok(Box::new(ByteStreamSplitDecompressor::new(
                    values.bits_per_value as usize,
                )))
            }
            Compression::General(general) => {
                // Create inner decompressor
                let inner_decompressor = self.create_miniblock_decompressor(
                    general.values.as_ref().ok_or_else(|| {
                        Error::invalid_input("GeneralMiniBlock missing inner encoding", location!())
                    })?,
                    decompression_strategy,
                )?;

                // Parse compression config
                let compression = general.compression.as_ref().ok_or_else(|| {
                    Error::invalid_input("GeneralMiniBlock missing compression config", location!())
                })?;

                let scheme = compression.scheme().try_into()?;

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
        description: &CompressiveEncoding,
    ) -> Result<Box<dyn FixedPerValueDecompressor>> {
        match description.compression.as_ref().unwrap() {
            Compression::Flat(flat) => Ok(Box::new(ValueDecompressor::from_flat(flat))),
            Compression::FixedSizeList(fsl) => Ok(Box::new(ValueDecompressor::from_fsl(fsl))),
            _ => todo!("fixed-per-value decompressor for {:?}", description),
        }
    }

    fn create_variable_per_value_decompressor(
        &self,
        description: &CompressiveEncoding,
    ) -> Result<Box<dyn VariablePerValueDecompressor>> {
        match description.compression.as_ref().unwrap() {
            Compression::Variable(variable) => {
                let Compression::Flat(offsets) = variable
                    .offsets
                    .as_ref()
                    .unwrap()
                    .compression
                    .as_ref()
                    .unwrap()
                else {
                    panic!("Variable compression only supports flat offsets")
                };
                assert!(offsets.bits_per_value < u8::MAX as u64);
                Ok(Box::new(VariableDecoder::default()))
            }
            Compression::Fsst(ref fsst) => Ok(Box::new(FsstPerValueDecompressor::new(
                LanceBuffer::from_bytes(fsst.symbol_table.clone(), 1),
                Box::new(VariableDecoder::default()),
            ))),
            Compression::General(ref general) => {
                Ok(Box::new(CompressedBufferEncoder::from_scheme(
                    general.compression.as_ref().expect_ok()?.scheme(),
                )?))
            }
            _ => todo!("variable-per-value decompressor for {:?}", description),
        }
    }

    fn create_block_decompressor(
        &self,
        description: &CompressiveEncoding,
    ) -> Result<Box<dyn BlockDecompressor>> {
        match description.compression.as_ref().unwrap() {
            Compression::Flat(flat) => Ok(Box::new(ValueDecompressor::from_flat(flat))),
            Compression::Constant(constant) => {
                let scalar = LanceBuffer::from_bytes(constant.value.clone(), 1);
                Ok(Box::new(ConstantDecompressor::new(scalar)))
            }
            Compression::Variable(_) => Ok(Box::new(BinaryBlockDecompressor::default())),
            _ => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::LanceBuffer;
    use crate::data::{BlockInfo, DataBlock};
    use arrow_schema::{DataType, Field as ArrowField};
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
        // Create varied data to avoid low entropy
        let bytes_per_value = (bits_per_value / 8) as usize;
        let total_bytes = bytes_per_value * num_values as usize;
        let mut data = vec![0u8; total_bytes];

        // Create data with specified run count
        let values_per_run = (num_values / run_count).max(1);
        let mut run_value = 0u8;

        for i in 0..num_values as usize {
            if i % values_per_run as usize == 0 {
                run_value = run_value.wrapping_add(17); // Use prime to get varied values
            }
            // Fill all bytes of the value to create high entropy
            for j in 0..bytes_per_value {
                let byte_offset = i * bytes_per_value + j;
                if byte_offset < data.len() {
                    data[byte_offset] = run_value.wrapping_add(j as u8);
                }
            }
        }

        let mut block = FixedWidthDataBlock {
            bits_per_value,
            data: LanceBuffer::reinterpret_vec(data),
            num_values,
            block_info: BlockInfo::default(),
        };

        // Compute all statistics including BytePositionEntropy
        use crate::statistics::ComputeStat;
        block.compute_stat();

        DataBlock::FixedWidth(block)
    }

    fn create_fixed_width_block(bits_per_value: u64, num_values: u64) -> DataBlock {
        // Create data with some variety to avoid always triggering BSS
        let bytes_per_value = (bits_per_value / 8) as usize;
        let total_bytes = bytes_per_value * num_values as usize;
        let mut data = vec![0u8; total_bytes];

        // Add some variation to the data to make it more realistic
        for i in 0..num_values as usize {
            let byte_offset = i * bytes_per_value;
            if byte_offset < data.len() {
                data[byte_offset] = (i % 256) as u8;
            }
        }

        let mut block = FixedWidthDataBlock {
            bits_per_value,
            data: LanceBuffer::reinterpret_vec(data),
            num_values,
            block_info: BlockInfo::default(),
        };

        // Compute all statistics including BytePositionEntropy
        use crate::statistics::ComputeStat;
        block.compute_stat();

        DataBlock::FixedWidth(block)
    }

    #[test]
    fn test_parameter_based_compression() {
        let mut params = CompressionParams::new();

        // Configure RLE for ID columns with BSS explicitly disabled
        params.columns.insert(
            "*_id".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.3),
                compression: Some("lz4".to_string()),
                compression_level: None,
                bss: Some(BssMode::Off), // Explicitly disable BSS to test RLE
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);
        let field = create_test_field("user_id", DataType::Int32);

        // Create data with low run count for RLE
        // Use create_fixed_width_block_with_stats which properly sets run count
        let data = create_fixed_width_block_with_stats(32, 1000, 100); // 100 runs out of 1000 values

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should use RLE due to low threshold (0.3) and low run count (100/1000 = 0.1)
        let debug_str = format!("{:?}", compressor);

        // The compressor should be RLE wrapped in general compression
        assert!(debug_str.contains("GeneralMiniBlockCompressor"));
        assert!(debug_str.contains("RleMiniBlockEncoder"));
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
                bss: Some(BssMode::Off), // Disable BSS to test RLE
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
                bss: None,
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);

        // Get merged params
        let merged = strategy
            .params
            .get_field_params("user_id", &DataType::Int32);

        // Column params should override type params
        assert_eq!(merged.rle_threshold, Some(0.2));
        assert_eq!(merged.compression, Some("zstd".to_string()));
        assert_eq!(merged.compression_level, Some(6));

        // Test field with only type params
        let merged = strategy
            .params
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
            .get_field_params("log_messages", &DataType::Utf8);
        assert_eq!(merged.compression, Some("zstd".to_string()));
        assert_eq!(merged.compression_level, Some(6));

        // Should not match
        let merged = strategy
            .params
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

    #[test]
    fn test_field_metadata_compression() {
        let params = CompressionParams::new();
        let strategy = DefaultCompressionStrategy::with_params(params);

        // Test field with compression metadata
        let mut metadata = HashMap::new();
        metadata.insert(COMPRESSION_META_KEY.to_string(), "zstd".to_string());
        metadata.insert(COMPRESSION_LEVEL_META_KEY.to_string(), "6".to_string());
        let mut field = create_test_field("test_column", DataType::Int32);
        field.metadata = metadata;

        let data = create_fixed_width_block(32, 1000);
        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();

        // Should use zstd with level 6
        let debug_str = format!("{:?}", compressor);
        assert!(debug_str.contains("GeneralMiniBlockCompressor"));
    }

    #[test]
    fn test_field_metadata_rle_threshold() {
        let params = CompressionParams::new();
        let strategy = DefaultCompressionStrategy::with_params(params);

        // Test field with RLE threshold metadata
        let mut metadata = HashMap::new();
        metadata.insert(RLE_THRESHOLD_META_KEY.to_string(), "0.8".to_string());
        metadata.insert(BSS_META_KEY.to_string(), "off".to_string()); // Disable BSS to test RLE
        let mut field = create_test_field("test_column", DataType::Int32);
        field.metadata = metadata;

        // Create data with low run count (e.g., 100 runs for 1000 values = 0.1 ratio)
        // This ensures run_count (100) < num_values * threshold (1000 * 0.8 = 800)
        let data = create_fixed_width_block_with_stats(32, 1000, 100);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();

        // Should use RLE because run_count (100) < num_values * threshold (800)
        let debug_str = format!("{:?}", compressor);
        assert!(debug_str.contains("RleMiniBlockEncoder"));
    }

    #[test]
    fn test_field_metadata_override_params() {
        // Set up params with one configuration
        let mut params = CompressionParams::new();
        params.columns.insert(
            "test_column".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.3),
                compression: Some("lz4".to_string()),
                compression_level: None,
                bss: None,
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);

        // Field metadata should override params
        let mut metadata = HashMap::new();
        metadata.insert(COMPRESSION_META_KEY.to_string(), "none".to_string());
        let mut field = create_test_field("test_column", DataType::Int32);
        field.metadata = metadata;

        let data = create_fixed_width_block(32, 1000);
        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();

        // Should use none compression (from metadata) instead of lz4 (from params)
        assert!(format!("{:?}", compressor).contains("ValueEncoder"));
    }

    #[test]
    fn test_field_metadata_mixed_configuration() {
        // Configure type-level params
        let mut params = CompressionParams::new();
        params.types.insert(
            "Int32".to_string(),
            CompressionFieldParams {
                rle_threshold: Some(0.5),
                compression: Some("lz4".to_string()),
                ..Default::default()
            },
        );

        let strategy = DefaultCompressionStrategy::with_params(params);

        // Field metadata provides partial override
        let mut metadata = HashMap::new();
        metadata.insert(COMPRESSION_LEVEL_META_KEY.to_string(), "3".to_string());
        let mut field = create_test_field("test_column", DataType::Int32);
        field.metadata = metadata;

        let data = create_fixed_width_block(32, 1000);
        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();

        // Should use lz4 (from type params) with level 3 (from metadata)
        let debug_str = format!("{:?}", compressor);
        assert!(debug_str.contains("GeneralMiniBlockCompressor"));
    }

    #[test]
    fn test_bss_field_metadata() {
        let params = CompressionParams::new();
        let strategy = DefaultCompressionStrategy::with_params(params);

        // Test BSS "on" mode with compression enabled (BSS requires compression to be effective)
        let mut metadata = HashMap::new();
        metadata.insert(BSS_META_KEY.to_string(), "on".to_string());
        metadata.insert(COMPRESSION_META_KEY.to_string(), "lz4".to_string());
        let arrow_field =
            ArrowField::new("temperature", DataType::Float32, false).with_metadata(metadata);
        let field = Field::try_from(&arrow_field).unwrap();

        // Create float data
        let data = create_fixed_width_block(32, 100);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        let debug_str = format!("{:?}", compressor);
        assert!(debug_str.contains("ByteStreamSplitEncoder"));
    }

    #[test]
    fn test_bss_with_compression() {
        let params = CompressionParams::new();
        let strategy = DefaultCompressionStrategy::with_params(params);

        // Test BSS with LZ4 compression
        let mut metadata = HashMap::new();
        metadata.insert(BSS_META_KEY.to_string(), "on".to_string());
        metadata.insert(COMPRESSION_META_KEY.to_string(), "lz4".to_string());
        let arrow_field =
            ArrowField::new("sensor_data", DataType::Float64, false).with_metadata(metadata);
        let field = Field::try_from(&arrow_field).unwrap();

        // Create double data
        let data = create_fixed_width_block(64, 100);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        let debug_str = format!("{:?}", compressor);
        // Should have BSS wrapped in general compression
        assert!(debug_str.contains("GeneralMiniBlockCompressor"));
        assert!(debug_str.contains("ByteStreamSplitEncoder"));
    }
}
