// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column statistics collection for Lance data files.
//!
//! This module provides per-zone column statistics
//! that are collected during file writing and stored in the file metadata
//! as a global buffer

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field as ArrowField, Fields};
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_common::ScalarValue;
use datafusion_expr::Accumulator;
use lance_core::utils::zone::{ZoneBound, ZoneProcessor};
use lance_core::{Error, Result};

/// Zone size for column statistics (1 million rows per zone)
pub(super) const COLUMN_STATS_ZONE_SIZE: u64 = 1_000_000;

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

/// Returns true for types that support min/max aggregation.
/// We exclude nested types (Struct, List, etc.) because DataFusion's try_new can succeed
/// for them but comparison fails at runtime. For other types we delegate to try_new.
fn supports_min_max(data_type: &DataType) -> bool {
    // Exclude types that try_new accepts but fail at runtime when comparing.
    // FixedSizeList is excluded because extension types (e.g. bfloat16) use it as storage;
    // min/max arrays then lack extension metadata and cause schema mismatch.
    if matches!(
        data_type,
        DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Struct(_)
            | DataType::Map(_, _)
            | DataType::RunEndEncoded(_, _)
            | DataType::Dictionary(_, _)
    ) {
        return false;
    }
    MinAccumulator::try_new(data_type).is_ok() && MaxAccumulator::try_new(data_type).is_ok()
}

impl ColumnStatisticsProcessor {
    pub(super) fn new(data_type: DataType) -> Result<Self> {
        if !supports_min_max(&data_type) {
            return Err(Error::invalid_input(format!(
                "Column statistics (min/max) not supported for type {:?}",
                data_type
            )));
        }
        let min =
            MinAccumulator::try_new(&data_type).map_err(|e| Error::invalid_input(e.to_string()))?;
        let max =
            MaxAccumulator::try_new(&data_type).map_err(|e| Error::invalid_input(e.to_string()))?;
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
            .map_err(|e| Error::invalid_input(e.to_string()))?;
        self.max
            .update_batch(std::slice::from_ref(array))
            .map_err(|e| Error::invalid_input(e.to_string()))?;
        Ok(())
    }

    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
        let stats = ColumnZoneStatistics {
            min: self
                .min
                .evaluate()
                .map_err(|e| Error::invalid_input(e.to_string()))?,
            max: self
                .max
                .evaluate()
                .map_err(|e| Error::invalid_input(e.to_string()))?,
            null_count: self.null_count,
            nan_count: self.nan_count,
            bound,
        };

        // Auto-reset for next zone
        self.min = MinAccumulator::try_new(&self.data_type)
            .map_err(|e| Error::invalid_input(e.to_string()))?;
        self.max = MaxAccumulator::try_new(&self.data_type)
            .map_err(|e| Error::invalid_input(e.to_string()))?;
        self.null_count = 0;
        self.nan_count = 0;

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
