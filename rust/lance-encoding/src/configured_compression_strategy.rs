// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compression strategy that uses user-configured parameters

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use fsst::fsst::{FSST_LEAST_INPUT_MAX_LENGTH, FSST_LEAST_INPUT_SIZE};
use lance_core::{
    datatypes::{Field, COMPRESSION_META_KEY, RLE_THRESHOLD_META_KEY},
    Error, Result,
};
use snafu::location;

use crate::{
    compression::{BlockCompressor, CompressionStrategy, DefaultCompressionStrategy},
    compression_config::{CompressionFieldParams, CompressionParams},
    data::{DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    encodings::{
        logical::primitive::{fullzip::PerValueCompressor, miniblock::MiniBlockCompressor},
        physical::{
            binary::BinaryMiniBlockEncoder,
            bitpack::InlineBitpacking,
            block::{CompressionConfig, CompressionScheme},
            fsst::FsstMiniBlockEncoder,
            general::GeneralMiniBlockCompressor,
            packed::PackedStructFixedWidthMiniBlockEncoder,
            rle::RleMiniBlockEncoder,
            value::ValueEncoder,
        },
    },
    format::pb,
    statistics::{GetStat, Stat},
};

/// Default RLE threshold if not specified in parameters
const DEFAULT_RLE_THRESHOLD: f64 = 0.5;

/// A compression strategy that applies user-configured parameters
/// before falling back to default strategy
#[derive(Debug)]
pub struct ConfiguredCompressionStrategy {
    /// User-configured compression parameters
    params: Arc<CompressionParams>,
    /// Default strategy as fallback
    default_strategy: DefaultCompressionStrategy,
}

impl ConfiguredCompressionStrategy {
    /// Create a new configured compression strategy
    pub fn new(params: CompressionParams) -> Self {
        Self {
            params: Arc::new(params),
            default_strategy: DefaultCompressionStrategy,
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
                .unwrap_or(DEFAULT_RLE_THRESHOLD);

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

impl CompressionStrategy for ConfiguredCompressionStrategy {
    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        // Get merged parameters for this field
        let params = self
            .params
            .get_field_params(&field.name, &field.data_type());

        // Route to appropriate builder based on data type
        match data {
            DataBlock::FixedWidth(fixed_width) => {
                self.build_fixed_width_compressor(&params, field, fixed_width)
            }
            DataBlock::VariableWidth(variable_width) => {
                self.build_variable_width_compressor(&params, variable_width)
            }
            DataBlock::Struct(_) => {
                // Struct compression doesn't use parameters currently
                Ok(Box::new(PackedStructFixedWidthMiniBlockEncoder::default()))
            }
            DataBlock::FixedSizeList(_) => {
                // FSL doesn't support compression currently
                Ok(Box::new(ValueEncoder::default()))
            }
            _ => {
                // Fall back to default strategy for unsupported types
                self.default_strategy
                    .create_miniblock_compressor(field, data)
            }
        }
    }

    fn create_per_value(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>> {
        // Per-value compression doesn't use parameters currently
        self.default_strategy.create_per_value(field, data)
    }

    fn create_block_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<(Box<dyn BlockCompressor>, pb::ArrayEncoding)> {
        // Block compression doesn't use parameters currently
        self.default_strategy.create_block_compressor(field, data)
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

        let strategy = ConfiguredCompressionStrategy::new(params);
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

        let strategy = ConfiguredCompressionStrategy::new(params);
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

        let strategy = ConfiguredCompressionStrategy::new(params);
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

        let strategy = ConfiguredCompressionStrategy::new(params);

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

        let strategy = ConfiguredCompressionStrategy::new(params);

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
        let strategy = ConfiguredCompressionStrategy::new(params);

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
        let strategy = ConfiguredCompressionStrategy::new(params);

        let field = create_test_field("random_column", DataType::Int32);
        // Create data with high run count that won't trigger RLE (600 runs for 1000 values = 0.6 ratio)
        let data = create_fixed_width_block_with_stats(32, 1000, 600);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should use default strategy's decision
        let debug_str = format!("{:?}", compressor);
        assert!(debug_str.contains("ValueEncoder") || debug_str.contains("InlineBitpacking"));
    }
}
