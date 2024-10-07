// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Data layouts to represent encoded data in a sub-Arrow format
//!
//! These [`DataBlock`] structures represent physical layouts.  They fill a gap somewhere
//! between [`arrow_data::data::ArrayData`] (which, as a collection of buffers, is too
//! generic because it doesn't give us enough information about what those buffers represent)
//! and [`arrow_array::array::Array`] (which is too specific, because it cares about the
//! logical data type).
//!
//! In addition, the layouts represented here are slightly stricter than Arrow's layout rules.
//! For example, offset buffers MUST start with 0.  These additional restrictions impose a
//! slight penalty on encode (to normalize arrow data) but make the development of encoders
//! and decoders easier (since they can rely on a normalized representation)

use std::{ops::Range, sync::Arc};

use arrow::array::{ArrayData, ArrayDataBuilder, AsArray};
use arrow_array::{new_null_array, Array, ArrayRef, Int32Array, Int64Array, UInt64Array};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use arrow_schema::DataType;
use lance_arrow::DataTypeExt;
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::{buffer::LanceBuffer, data::{DataBlock, DictionaryDataBlock, FixedWidthDataBlock, NullableDataBlock, OpaqueBlock, StructDataBlock, VariableWidthBlock}};

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stat {
    Min,
    Max,
    BitWidth,
    DataSize,
    Cardinality,
    FixedSize,
    NullCount,
}
pub trait GetStat {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>>;
}

impl GetStat for DataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        match self {
            DataBlock::AllNull(_) => {
                // the null count is not calculated here as this enum is going to deprecated anyway
                Some(Arc::new(Int64Array::from(vec![0i64])))
            }
            DataBlock::Nullable(data_block) => data_block.get_stat(stat),
            DataBlock::FixedWidth(data_block) => data_block.get_stat(stat),
            DataBlock::FixedSizeList(data_block) => data_block.child.get_stat(stat),
            DataBlock::VariableWidth(data_block) => data_block.get_stat(stat),
            DataBlock::Opaque(data_block) => data_block.get_stat(stat),
            DataBlock::Struct(data_block) => data_block.get_stat(stat),
            DataBlock::Dictionary(data_block) => data_block.get_stat(stat),
        }
    }
}

impl GetStat for NullableDataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        // initialize statistics
        if self.info.is_empty() {
            self.compute_statistics();    
        }
        match stat {
            Stat::Min | Stat::Max | Stat::BitWidth | Stat::Cardinality | Stat::FixedSize => {
                self.data.get_stat(stat)
            }
            Stat::DataSize => {
                Some(self.info.get(&stat).unwrap().clone())
            }
            Stat::NullCount => {
                Some(self.info.get(&stat).unwrap().clone())
            }
        }
    }
}

impl GetStat for FixedWidthDataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        // initialize statistics
        if self.info.is_empty() {
            self.compute_statistics();    
        }
        Some(self.info.get(&stat).unwrap().clone())
    }
}

impl GetStat for VariableWidthBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        if self.info.is_empty() {
            self.compute_statistics();
        }
        match stat {
            Stat::BitWidth => {
                None
            },
            _ => {
                Some(self.info.get(&stat).unwrap().clone())
            },
        }
    }
}

impl VariableWidthBlock {
    fn compute_statistics(&mut self) {
        let min_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::NullCount, min_array);

        let max_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::Max, max_array);

        let cardinality_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::Cardinality, cardinality_array);

        let data_size_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::DataSize, data_size_array);

        let null_count_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::NullCount, null_count_array);

        let fixed_size_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::FixedSize, fixed_size_array);

    }
}

impl FixedWidthDataBlock {
    fn compute_statistics(&mut self) {
        let min_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::NullCount, min_array);

        let max_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::Max, max_array);

        let bit_width_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::BitWidth, bit_width_array);

        let cardinality_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::Cardinality, cardinality_array);

        let data_size_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::DataSize, data_size_array);

        let null_count_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::NullCount, null_count_array);

        let fixed_size_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::FixedSize, fixed_size_array);
    }
}

impl NullableDataBlock {
    fn compute_statistics(&mut self) {
        // How to compute null_count from NullableDataBlock?
        let null_count_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::NullCount, null_count_array);

        let data_size_array = Arc::new(Int64Array::from(vec![0]));
        self.info.insert(Stat::NullCount, data_size_array);
    }
}

impl GetStat for OpaqueBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        None
    }
}

impl GetStat for DictionaryDataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        None
    }
}

impl GetStat for StructDataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        None
    }
}