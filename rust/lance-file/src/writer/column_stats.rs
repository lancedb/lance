// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column statistics collection for Lance data files.
//!
//! This module provides per-zone column statistics
//! that are collected during file writing and stored in the file metadata
//! as a global buffer

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field as ArrowField, Fields};
use lance_arrow_scalar::ArrowScalar;
use lance_arrow_stats::StatisticsAccumulator;
use lance_core::utils::zone::{ZoneBound, ZoneProcessor};
use lance_core::{Error, Result};

/// Zone size for column statistics (1 million rows per zone)
pub(super) const COLUMN_STATS_ZONE_SIZE: u64 = 1_000_000;

/// Column statistics for a single zone
#[derive(Debug, Clone)]
pub(super) struct ColumnZoneStatistics {
    pub min: Option<ArrowScalar>,
    pub max: Option<ArrowScalar>,
    pub null_count: u32,
    pub nan_count: u32,
    pub bound: ZoneBound,
}

/// Statistics processor for a single column that implements ZoneProcessor trait
pub(super) struct ColumnStatisticsProcessor {
    accumulator: StatisticsAccumulator,
}

/// Returns true for types that support min/max aggregation.
fn supports_min_max(data_type: &DataType) -> bool {
    // Skip binary types until column-zone stats can store bounded prefixes instead of full values.
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}

fn count_to_u32(value: u64, stat_name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::invalid_input(format!(
            "Column statistics {} exceeds UInt32: {}",
            stat_name, value
        ))
    })
}

impl ColumnStatisticsProcessor {
    pub(super) fn new(data_type: DataType) -> Result<Self> {
        if !supports_min_max(&data_type) {
            return Err(Error::invalid_input(format!(
                "Column statistics (min/max) not supported for type {:?}",
                data_type
            )));
        }
        Ok(Self {
            accumulator: StatisticsAccumulator::new(&data_type),
        })
    }
}

impl ZoneProcessor for ColumnStatisticsProcessor {
    type ZoneStatistics = ColumnZoneStatistics;

    fn process_chunk(&mut self, array: &ArrayRef) -> Result<()> {
        self.accumulator
            .update(array)
            .map_err(|e| Error::invalid_input(e.to_string()))?;
        Ok(())
    }

    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
        let snapshot = self.accumulator.statistics();
        let stats = ColumnZoneStatistics {
            min: snapshot.min,
            max: snapshot.max,
            null_count: count_to_u32(snapshot.null_count, "null_count")?,
            nan_count: count_to_u32(snapshot.nan_count.unwrap_or(0), "nan_count")?,
            bound,
        };

        self.accumulator.reset();

        Ok(stats)
    }
}

/// Create Arrow struct type for file level ColumnZoneStatistics for a given column type.
pub(super) fn create_column_zone_statistics_struct_type(column_type: &DataType) -> DataType {
    let zone_bound_fields = Fields::from(vec![
        ArrowField::new("fragment_id", DataType::UInt64, false),
        ArrowField::new("start", DataType::UInt64, false),
        ArrowField::new("length", DataType::UInt64, false),
    ]);

    DataType::Struct(Fields::from(vec![
        // min and max are nullable because they can be null for empty zones
        ArrowField::new("min", column_type.clone(), true),
        ArrowField::new("max", column_type.clone(), true),
        ArrowField::new("null_count", DataType::UInt32, false),
        ArrowField::new("nan_count", DataType::UInt32, false),
        ArrowField::new("bound", DataType::Struct(zone_bound_fields), false),
    ]))
}

/// Create Arrow struct type for consolidated zone statistics for a given column type.
pub fn create_consolidated_zone_struct_type(column_type: &DataType) -> DataType {
    DataType::Struct(Fields::from(vec![
        ArrowField::new("fragment_id", DataType::UInt64, false),
        ArrowField::new("zone_start", DataType::UInt64, false),
        ArrowField::new("zone_length", DataType::UInt64, false),
        ArrowField::new("null_count", DataType::UInt32, false),
        ArrowField::new("nan_count", DataType::UInt32, false),
        ArrowField::new("min_value", column_type.clone(), true),
        ArrowField::new("max_value", column_type.clone(), true),
    ]))
}
