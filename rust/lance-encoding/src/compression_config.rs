// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Configuration for compression overrides
//!
//! This module provides types for configuring compression strategies
//! on a per-column or per-type basis.

use std::collections::HashMap;

use arrow::datatypes::DataType;

/// Compression override configuration
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionOverrides {
    /// Column-level compression configuration
    /// Key: column name pattern (supports wildcards)
    /// Value: compression specification chain (applied from first to last, with each wrapping the previous)
    pub columns: HashMap<String, Vec<CompressionSpec>>,

    /// Type-level compression configuration
    /// Key: data type name (e.g., "Int32", "Utf8", etc.)
    /// Value: compression specification chain (applied from first to last, with each wrapping the previous)
    pub types: HashMap<String, Vec<CompressionSpec>>,
}

impl CompressionOverrides {
    /// Create empty compression overrides
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            types: HashMap::new(),
        }
    }

    /// Get compression chain for a specific field
    pub fn get_compression_chain(
        &self,
        field_name: &str,
        data_type: &DataType,
    ) -> Option<&Vec<CompressionSpec>> {
        // 1. Check exact column match
        if let Some(chain) = self.columns.get(field_name) {
            return Some(chain);
        }

        // 2. Check wildcard patterns
        for (pattern, chain) in &self.columns {
            if matches_pattern(field_name, pattern) {
                return Some(chain);
            }
        }

        // 3. Check type match
        //
        // TOOD: maybe we can allow simple types like `string` and `int`.
        let type_name = data_type.to_string();
        self.types.get(&type_name)
    }
}

impl Default for CompressionOverrides {
    fn default() -> Self {
        Self::new()
    }
}

/// Specification for a single compression method
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionSpec {
    /// RLE compression
    Rle { rle: RleConfig },

    /// Bitpack compression
    Bitpack { bitpack: BitpackConfig },

    /// FSST string compression
    Fsst { fsst: FsstConfig },

    /// General compression (LZ4, Zstd, etc.)
    General { general: GeneralConfig },
}

/// Configuration for RLE compression
#[derive(Debug, Clone, PartialEq)]
pub struct RleConfig {
    /// Threshold for RLE compression (0.0 to 1.0)
    /// RLE is used when run_count < num_values * threshold
    pub threshold: f64,
}

/// Configuration for Bitpack compression
#[derive(Debug, Clone, PartialEq)]
pub struct BitpackConfig {
    // Currently no configuration options
}

/// Configuration for FSST compression
#[derive(Debug, Clone, PartialEq)]
pub struct FsstConfig {
    // Currently no configuration options
}

/// Configuration for general compression
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralConfig {
    /// Compression scheme: "zstd", "lz4", "none"
    pub scheme: String,

    /// Compression level (only for schemes that support it)
    pub level: Option<i32>,
}

impl Default for RleConfig {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

/// Check if a name matches a pattern (supports wildcards)
fn matches_pattern(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }

    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }

    if pattern.contains('*') {
        // Simple glob pattern matching (only supports single * in middle)
        if let Some(pos) = pattern.find('*') {
            let prefix = &pattern[..pos];
            let suffix = &pattern[pos + 1..];
            return name.starts_with(prefix)
                && name.ends_with(suffix)
                && name.len() >= pattern.len() - 1;
        }
    }

    name == pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        assert!(matches_pattern("user_id", "*_id"));
        assert!(matches_pattern("product_id", "*_id"));
        assert!(!matches_pattern("identity", "*_id"));

        assert!(matches_pattern("log_message", "log_*"));
        assert!(matches_pattern("log_level", "log_*"));
        assert!(!matches_pattern("message_log", "log_*"));

        assert!(matches_pattern("test_field_name", "test_*_name"));
        assert!(matches_pattern("test_column_name", "test_*_name"));
        assert!(!matches_pattern("test_name", "test_*_name"));

        assert!(matches_pattern("anything", "*"));
        assert!(matches_pattern("exact_match", "exact_match"));
    }

    #[test]
    fn test_type_name_mapping() {
        // Test that DataType.to_string() returns expected values
        assert_eq!(DataType::Int32.to_string(), "Int32");
        assert_eq!(DataType::UInt64.to_string(), "UInt64");
        assert_eq!(DataType::Float32.to_string(), "Float32");
        assert_eq!(DataType::Utf8.to_string(), "Utf8");
        assert_eq!(DataType::Binary.to_string(), "Binary");
        assert_eq!(DataType::Boolean.to_string(), "Boolean");
        assert_eq!(DataType::Date32.to_string(), "Date32");
        assert_eq!(
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None).to_string(),
            "Timestamp(Millisecond, None)"
        );
    }

    #[test]
    fn test_compression_spec_creation() {
        // Test RLE spec
        let spec = CompressionSpec::Rle {
            rle: RleConfig { threshold: 0.3 },
        };
        match spec {
            CompressionSpec::Rle { rle } => assert_eq!(rle.threshold, 0.3),
            _ => panic!("Expected RLE spec"),
        }

        // Test Bitpack spec
        let spec = CompressionSpec::Bitpack {
            bitpack: BitpackConfig {},
        };
        assert!(matches!(spec, CompressionSpec::Bitpack { .. }));

        // Test General spec
        let spec = CompressionSpec::General {
            general: GeneralConfig {
                scheme: "zstd".to_string(),
                level: Some(3),
            },
        };
        match spec {
            CompressionSpec::General { general } => {
                assert_eq!(general.scheme, "zstd");
                assert_eq!(general.level, Some(3));
            }
            _ => panic!("Expected General spec"),
        }
    }

    #[test]
    fn test_compression_overrides() {
        let mut overrides = CompressionOverrides::new();

        // Add column override
        overrides.columns.insert("*_id".to_string(), vec![]);

        // Add type override
        overrides.types.insert(
            "Utf8".to_string(),
            vec![CompressionSpec::Fsst {
                fsst: FsstConfig {},
            }],
        );

        // Test column match
        let chain = overrides.get_compression_chain("user_id", &DataType::Int64);
        assert!(chain.is_some());
        assert!(chain.unwrap().is_empty());

        // Test type match
        let chain = overrides.get_compression_chain("description", &DataType::Utf8);
        assert!(chain.is_some());
        assert_eq!(chain.unwrap().len(), 1);
    }
}
