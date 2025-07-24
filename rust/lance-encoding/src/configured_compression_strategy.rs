// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compression strategy that uses user-configured overrides

use std::sync::Arc;

use lance_core::{
    datatypes::{Field, COMPRESSION_META_KEY},
    Error, Result,
};
use snafu::location;

use crate::{
    compression::{BlockCompressor, CompressionStrategy, DefaultCompressionStrategy},
    compression_config::{CompressionOverrides, CompressionSpec},
    data::DataBlock,
    encodings::{
        logical::primitive::fullzip::PerValueCompressor,
        physical::{
            binary::BinaryMiniBlockEncoder, bitpack::InlineBitpacking, block::CompressionConfig,
            fsst::FsstMiniBlockEncoder, general::GeneralMiniBlockCompressor,
            packed::PackedStructFixedWidthMiniBlockEncoder, rle::RleMiniBlockEncoder,
            value::ValueEncoder,
        },
    },
    format::pb,
};

/// A compression strategy that applies user-configured compression overrides
/// before falling back to default strategy
#[derive(Debug)]
pub struct ConfiguredCompressionStrategy {
    /// User-configured compression overrides
    overrides: Arc<CompressionOverrides>,
    /// Default strategy as fallback
    default_strategy: DefaultCompressionStrategy,
}

impl ConfiguredCompressionStrategy {
    /// Create a new configured compression strategy
    pub fn new(overrides: CompressionOverrides) -> Self {
        Self {
            overrides: Arc::new(overrides),
            default_strategy: DefaultCompressionStrategy,
        }
    }

    /// Get the compression chain for a field, if any
    fn get_compression_chain(&self, field: &Field) -> Option<&Vec<CompressionSpec>> {
        self.overrides
            .get_compression_chain(&field.name, &field.data_type())
    }

    /// Create a miniblock compressor from a compression spec
    fn create_miniblock_from_spec(
        &self,
        spec: &CompressionSpec,
        data: &DataBlock,
    ) -> Result<Box<dyn crate::encodings::logical::primitive::miniblock::MiniBlockCompressor>> {
        match spec {
            CompressionSpec::Rle { rle: _ } => {
                // Validate that data is fixed width and byte-aligned
                if let DataBlock::FixedWidth(fixed_width) = data {
                    let is_byte_aligned = fixed_width.bits_per_value % 8 == 0;
                    if !is_byte_aligned {
                        return Err(Error::invalid_input(
                            format!(
                                "RLE compression requires byte-aligned data, got {} bits per value",
                                fixed_width.bits_per_value
                            ),
                            location!(),
                        ));
                    }
                    Ok(Box::new(RleMiniBlockEncoder::new()))
                } else {
                    Err(Error::invalid_input(
                        "RLE compression requires fixed-width data",
                        location!(),
                    ))
                }
            }
            CompressionSpec::Bitpack { .. } => {
                // Validate that data is fixed width
                if let DataBlock::FixedWidth(fixed_width) = data {
                    Ok(Box::new(InlineBitpacking::new(fixed_width.bits_per_value)))
                } else {
                    Err(Error::invalid_input(
                        "Bitpack compression requires fixed-width data",
                        location!(),
                    ))
                }
            }
            CompressionSpec::Fsst { .. } => {
                // Validate that data is variable width
                if let DataBlock::VariableWidth(_) = data {
                    Ok(Box::new(FsstMiniBlockEncoder::default()))
                } else {
                    Err(Error::invalid_input(
                        "FSST compression requires variable-width data",
                        location!(),
                    ))
                }
            }
            CompressionSpec::General { general } => {
                // General compression wraps another compressor
                // First, we need to determine the inner compressor based on data type
                let inner_compressor: Box<
                    dyn crate::encodings::logical::primitive::miniblock::MiniBlockCompressor,
                > = match data {
                    DataBlock::FixedWidth(_) => Box::new(ValueEncoder::default()),
                    DataBlock::VariableWidth(_) => Box::new(BinaryMiniBlockEncoder::default()),
                    DataBlock::Struct(_) => {
                        Box::new(PackedStructFixedWidthMiniBlockEncoder::default())
                    }
                    _ => {
                        return Err(Error::invalid_input(
                            format!(
                                "General compression not supported for data type: {}",
                                data.name()
                            ),
                            location!(),
                        ))
                    }
                };

                let scheme = general.scheme.parse()?;
                let config = CompressionConfig::new(scheme, general.level);

                Ok(Box::new(GeneralMiniBlockCompressor::new(
                    inner_compressor,
                    config,
                )))
            }
        }
    }

    /// Build a compression chain for miniblock compression
    /// The chain is built from inside out: chain[0] is the innermost compressor
    /// and chain[n-1] is the outermost compressor
    fn build_miniblock_chain(
        &self,
        chain: &[CompressionSpec],
        data: &DataBlock,
    ) -> Result<Box<dyn crate::encodings::logical::primitive::miniblock::MiniBlockCompressor>> {
        if chain.is_empty() {
            return Err(Error::invalid_input(
                "Compression chain cannot be empty",
                location!(),
            ));
        }

        // Start with the first compressor in the chain
        let mut compressor = self.create_miniblock_from_spec(&chain[0], data)?;

        // Wrap each subsequent compressor around the previous one
        for spec in &chain[1..] {
            match spec {
                CompressionSpec::General { general } => {
                    // General compression can wrap any other compressor
                    let scheme = general.scheme.parse()?;
                    let config = CompressionConfig::new(scheme, general.level);
                    compressor = Box::new(GeneralMiniBlockCompressor::new(compressor, config));
                }
                _ => {
                    return Err(Error::invalid_input(
                        format!(
                            "Only General compression can be used as an outer layer in a chain, got {:?}",
                            spec
                        ),
                        location!(),
                    ));
                }
            }
        }

        Ok(compressor)
    }
}

impl CompressionStrategy for ConfiguredCompressionStrategy {
    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn crate::encodings::logical::primitive::miniblock::MiniBlockCompressor>> {
        // Check if field has "none" compression metadata
        if let Some(compression) = field.metadata.get(COMPRESSION_META_KEY) {
            if compression.as_str() == "none" {
                return Ok(Box::new(ValueEncoder::default()));
            }
        }

        // Check for user-configured compression
        if let Some(chain) = self.get_compression_chain(field) {
            return self.build_miniblock_chain(chain, data);
        }

        // Fall back to default strategy
        self.default_strategy
            .create_miniblock_compressor(field, data)
    }

    fn create_per_value(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>> {
        // Per-value compression doesn't use user overrides currently
        // as it's primarily used for large values that are compressed individually
        self.default_strategy.create_per_value(field, data)
    }

    fn create_block_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<(Box<dyn BlockCompressor>, pb::ArrayEncoding)> {
        // Block compression doesn't use user overrides currently
        self.default_strategy.create_block_compressor(field, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::LanceBuffer;
    use crate::compression_config::{BitpackConfig, GeneralConfig, RleConfig};
    use crate::data::{BlockInfo, FixedWidthDataBlock};
    use arrow::datatypes::{DataType, Field as ArrowField};
    use std::collections::HashMap;

    fn create_test_field(name: &str, data_type: DataType) -> Field {
        let arrow_field = ArrowField::new(name, data_type, true);
        let mut field = Field::try_from(&arrow_field).unwrap();
        field.id = -1; // Set a default id
        field
    }

    fn create_fixed_width_block(bits_per_value: u64, num_values: u64) -> DataBlock {
        let block = FixedWidthDataBlock {
            bits_per_value,
            data: LanceBuffer::reinterpret_vec(vec![
                0u8;
                (bits_per_value * num_values / 8) as usize
            ]),
            num_values,
            block_info: BlockInfo::default(),
        };

        // Add required statistics to avoid panic
        use crate::statistics::Stat;
        use arrow::array::{ArrayRef, UInt64Array};
        use std::sync::Arc;

        let bit_widths = Arc::new(UInt64Array::from(vec![bits_per_value])) as ArrayRef;
        let run_count = Arc::new(UInt64Array::from(vec![num_values / 2])) as ArrayRef;

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
            .insert(Stat::RunCount, run_count);

        DataBlock::FixedWidth(block)
    }

    #[test]
    fn test_column_override() {
        let mut overrides = CompressionOverrides::new();
        overrides.columns.insert(
            "user_id".to_string(),
            vec![CompressionSpec::Rle {
                rle: RleConfig { threshold: 0.3 },
            }],
        );

        let strategy = ConfiguredCompressionStrategy::new(overrides);
        let field = create_test_field("user_id", DataType::Int32);
        let data = create_fixed_width_block(32, 1000);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should create RLE compressor
        assert!(format!("{:?}", compressor).contains("RleMiniBlockEncoder"));
    }

    #[test]
    fn test_type_override() {
        let mut overrides = CompressionOverrides::new();
        overrides.types.insert(
            "Int32".to_string(),
            vec![CompressionSpec::Bitpack {
                bitpack: BitpackConfig {},
            }],
        );

        let strategy = ConfiguredCompressionStrategy::new(overrides);
        let field = create_test_field("some_column", DataType::Int32);
        let data = create_fixed_width_block(32, 1000);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should create Bitpack compressor
        assert!(format!("{:?}", compressor).contains("InlineBitpacking"));
    }

    #[test]
    fn test_compression_chain() {
        let mut overrides = CompressionOverrides::new();
        overrides.columns.insert(
            "data".to_string(),
            vec![
                CompressionSpec::Rle {
                    rle: RleConfig { threshold: 0.5 },
                },
                CompressionSpec::General {
                    general: GeneralConfig {
                        scheme: "lz4".to_string(),
                        level: None,
                    },
                },
            ],
        );

        let strategy = ConfiguredCompressionStrategy::new(overrides);
        let field = create_test_field("data", DataType::Int32);
        let data = create_fixed_width_block(32, 1000);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should create GeneralMiniBlockCompressor(LZ4) wrapping RleMiniBlockEncoder
        // The data flow is: data -> RLE -> LZ4
        assert!(format!("{:?}", compressor).contains("GeneralMiniBlockCompressor"));
    }

    #[test]
    fn test_fallback_to_default() {
        let overrides = CompressionOverrides::new();
        let strategy = ConfiguredCompressionStrategy::new(overrides);
        let field = create_test_field("some_column", DataType::Int32);
        let data = create_fixed_width_block(32, 1000);

        // Should fall back to default strategy
        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Default strategy behavior
        assert!(format!("{:?}", compressor).contains("ValueEncoder"));
    }

    #[test]
    fn test_none_compression_metadata() {
        let overrides = CompressionOverrides::new();
        let strategy = ConfiguredCompressionStrategy::new(overrides);

        let mut metadata = HashMap::new();
        metadata.insert(COMPRESSION_META_KEY.to_string(), "none".to_string());
        let mut field = create_test_field("some_column", DataType::Int32);
        field.metadata = metadata;

        let data = create_fixed_width_block(32, 1000);

        let compressor = strategy.create_miniblock_compressor(&field, &data).unwrap();
        // Should create ValueEncoder for "none" compression
        assert!(format!("{:?}", compressor).contains("ValueEncoder"));
    }

    #[test]
    fn test_invalid_chain_configuration() {
        let mut overrides = CompressionOverrides::new();
        // Invalid chain: RLE cannot wrap Bitpack
        overrides.columns.insert(
            "data".to_string(),
            vec![
                CompressionSpec::Bitpack {
                    bitpack: BitpackConfig {},
                },
                CompressionSpec::Rle {
                    rle: RleConfig { threshold: 0.5 },
                },
            ],
        );

        let strategy = ConfiguredCompressionStrategy::new(overrides);
        let field = create_test_field("data", DataType::Int32);
        let data = create_fixed_width_block(32, 1000);

        let result = strategy.create_miniblock_compressor(&field, &data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Only General compression can be used as an outer layer"));
    }

    #[test]
    fn test_end_to_end_override_default_behavior() {
        // Test that ConfiguredCompressionStrategy can override DefaultCompressionStrategy behavior

        // First, let's see what the default strategy produces
        let default_strategy = DefaultCompressionStrategy;
        let field = create_test_field("user_id", DataType::Int32);
        let data = create_fixed_width_block(32, 1000);

        let default_compressor = default_strategy
            .create_miniblock_compressor(&field, &data)
            .unwrap();
        let default_debug = format!("{:?}", default_compressor);

        // Default should produce ValueEncoder for this case
        assert!(default_debug.contains("ValueEncoder"));

        // Now configure overrides to use RLE for user_id columns
        let mut overrides = CompressionOverrides::new();
        overrides.columns.insert(
            "*_id".to_string(),
            vec![CompressionSpec::Rle {
                rle: RleConfig { threshold: 0.5 },
            }],
        );

        let configured_strategy = ConfiguredCompressionStrategy::new(overrides.clone());
        let configured_compressor = configured_strategy
            .create_miniblock_compressor(&field, &data)
            .unwrap();
        let configured_debug = format!("{:?}", configured_compressor);

        // Should now use RLE instead of ValueEncoder
        assert!(configured_debug.contains("RleMiniBlockEncoder"));
        assert!(!configured_debug.contains("ValueEncoder"));

        // Test type-based override
        let mut type_overrides = CompressionOverrides::new();
        type_overrides.types.insert(
            "Int32".to_string(),
            vec![
                CompressionSpec::Bitpack {
                    bitpack: BitpackConfig {},
                },
                CompressionSpec::General {
                    general: GeneralConfig {
                        scheme: "zstd".to_string(),
                        level: Some(3),
                    },
                },
            ],
        );

        let type_strategy = ConfiguredCompressionStrategy::new(type_overrides);
        let field2 = create_test_field("some_other_column", DataType::Int32);
        let type_compressor = type_strategy
            .create_miniblock_compressor(&field2, &data)
            .unwrap();
        let type_debug = format!("{:?}", type_compressor);

        // Should use GeneralMiniBlockCompressor wrapping InlineBitpacking
        assert!(type_debug.contains("GeneralMiniBlockCompressor"));

        // Test that column override takes precedence over type override
        let mut both_overrides = CompressionOverrides::new();
        both_overrides.columns.insert(
            "special_column".to_string(),
            vec![CompressionSpec::Rle {
                rle: RleConfig { threshold: 0.3 },
            }],
        );
        both_overrides.types.insert(
            "Int32".to_string(),
            vec![CompressionSpec::Bitpack {
                bitpack: BitpackConfig {},
            }],
        );

        let both_strategy = ConfiguredCompressionStrategy::new(both_overrides);
        let field3 = create_test_field("special_column", DataType::Int32);
        let both_compressor = both_strategy
            .create_miniblock_compressor(&field3, &data)
            .unwrap();
        let both_debug = format!("{:?}", both_compressor);

        // Column override should win - should use RLE, not Bitpack
        assert!(both_debug.contains("RleMiniBlockEncoder"));
        assert!(!both_debug.contains("InlineBitpacking"));
    }

    #[test]
    fn test_complex_compression_chain() {
        // Test a more complex scenario with real-world patterns
        let mut overrides = CompressionOverrides::new();

        // Configure high-cardinality ID columns to use bitpack + zstd
        overrides.columns.insert(
            "*_id".to_string(),
            vec![
                CompressionSpec::Bitpack {
                    bitpack: BitpackConfig {},
                },
                CompressionSpec::General {
                    general: GeneralConfig {
                        scheme: "zstd".to_string(),
                        level: Some(3),
                    },
                },
            ],
        );

        // Configure low-cardinality status columns to use RLE + lz4
        overrides.columns.insert(
            "*_status".to_string(),
            vec![
                CompressionSpec::Rle {
                    rle: RleConfig { threshold: 0.2 },
                },
                CompressionSpec::General {
                    general: GeneralConfig {
                        scheme: "lz4".to_string(),
                        level: None,
                    },
                },
            ],
        );

        let strategy = ConfiguredCompressionStrategy::new(overrides);

        // Test high-cardinality ID column
        let id_field = create_test_field("user_id", DataType::Int64);
        let id_data = create_fixed_width_block(64, 10000);
        let id_compressor = strategy
            .create_miniblock_compressor(&id_field, &id_data)
            .unwrap();
        assert!(format!("{:?}", id_compressor).contains("GeneralMiniBlockCompressor"));

        // Test low-cardinality status column
        let status_field = create_test_field("order_status", DataType::Int32);
        let status_data = create_fixed_width_block(32, 1000);
        let status_compressor = strategy
            .create_miniblock_compressor(&status_field, &status_data)
            .unwrap();
        assert!(format!("{:?}", status_compressor).contains("GeneralMiniBlockCompressor"));

        // Test unmatched column falls back to default
        let other_field = create_test_field("price", DataType::Float64);
        let other_data = create_fixed_width_block(64, 1000);
        let other_compressor = strategy
            .create_miniblock_compressor(&other_field, &other_data)
            .unwrap();
        assert!(format!("{:?}", other_compressor).contains("ValueEncoder"));
    }
}
