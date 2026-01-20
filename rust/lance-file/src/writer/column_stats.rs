// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column statistics collection for Lance data files.
//!
//! This module provides per-zone column statistics (min, max, null_count, nan_count)
//! that are collected during file writing and stored in the file metadata.

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_common::ScalarValue;
use datafusion_expr::Accumulator;
use lance_core::utils::zone::{ZoneBound, ZoneProcessor};
use lance_core::{Error, Result};
use snafu::location;

/// Column statistics for a single zone
#[derive(Debug, Clone)]
pub(super) struct ColumnZoneStatistics {
    pub min: ScalarValue,
    pub max: ScalarValue,
    pub null_count: u32,
    pub nan_count: u32,
    pub bound: ZoneBound,
}

/// Statistics processor for a single column that implements ZoneProcessor trait
pub(super) struct ColumnStatisticsProcessor {
    data_type: DataType,
    min: MinAccumulator,
    max: MaxAccumulator,
    null_count: u32,
    nan_count: u32,
}

impl ColumnStatisticsProcessor {
    pub(super) fn new(data_type: DataType) -> Result<Self> {
        // TODO: Upstream DataFusion accumulators does not handle many nested types
        let min = MinAccumulator::try_new(&data_type)
            .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        let max = MaxAccumulator::try_new(&data_type)
            .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        Ok(Self {
            data_type,
            min,
            max,
            null_count: 0,
            nan_count: 0,
        })
    }

    fn count_nans(array: &ArrayRef) -> u32 {
        match array.data_type() {
            DataType::Float16 => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float16Array>()
                    .unwrap();
                array.values().iter().filter(|&&x| x.is_nan()).count() as u32
            }
            DataType::Float32 => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .unwrap();
                array.values().iter().filter(|&&x| x.is_nan()).count() as u32
            }
            DataType::Float64 => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .unwrap();
                array.values().iter().filter(|&&x| x.is_nan()).count() as u32
            }
            _ => 0,
        }
    }
}

impl ZoneProcessor for ColumnStatisticsProcessor {
    type ZoneStatistics = ColumnZoneStatistics;

    fn process_chunk(&mut self, array: &ArrayRef) -> Result<()> {
        self.null_count += array.null_count() as u32;
        self.nan_count += Self::count_nans(array);
        self.min
            .update_batch(std::slice::from_ref(array))
            .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        self.max
            .update_batch(std::slice::from_ref(array))
            .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        Ok(())
    }

    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
        let stats = ColumnZoneStatistics {
            min: self
                .min
                .evaluate()
                .map_err(|e| Error::invalid_input(e.to_string(), location!()))?,
            max: self
                .max
                .evaluate()
                .map_err(|e| Error::invalid_input(e.to_string(), location!()))?,
            null_count: self.null_count,
            nan_count: self.nan_count,
            bound,
        };

        // Auto-reset for next zone
        self.min = MinAccumulator::try_new(&self.data_type)
            .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        self.max = MaxAccumulator::try_new(&self.data_type)
            .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        self.null_count = 0;
        self.nan_count = 0;

        Ok(stats)
    }
}

/// Convert ScalarValue to string, extracting only the value without type prefix
/// E.g., Int32(42) -> "42", Float64(3.14) -> "3.14", Utf8("hello") -> "hello"
pub(super) fn scalar_value_to_string(value: &ScalarValue) -> String {
    let debug_str = format!("{:?}", value);

    // For string types, extract the quoted value
    if debug_str.starts_with("Utf8(") || debug_str.starts_with("LargeUtf8(") {
        // Extract content between quotes: Utf8("hello") -> "hello"
        if let Some(start) = debug_str.find('"') {
            if let Some(end) = debug_str.rfind('"') {
                if end > start {
                    return debug_str[start + 1..end].to_string();
                }
            }
        }
    }

    // For numeric types, extract content between parentheses
    // Int32(42) -> "42", Float64(3.14) -> "3.14"
    if let Some(start) = debug_str.find('(') {
        if let Some(end) = debug_str.rfind(')') {
            return debug_str[start + 1..end].to_string();
        }
    }

    // Fallback: return the whole debug string (shouldn't happen for supported types)
    debug_str
}

/// Zone size for column statistics (1 million rows per zone)
pub(super) const COLUMN_STATS_ZONE_SIZE: u64 = 1_000_000;
