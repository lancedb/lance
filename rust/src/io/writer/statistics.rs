// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Statistics collection utilities

use std::collections::HashMap;

use arrow_array::{
    builder::{make_builder, ArrayBuilder, PrimitiveBuilder},
    types::UInt32Type,
    Array, ArrayRef, RecordBatch,
};
use arrow_schema::DataType;

use crate::datatypes::Field;

/// Statistics for a single column chunk.
#[derive(Debug, PartialEq)]
pub struct StatisticsRow {
    /// Number of nulls in this column chunk.
    pub(crate) null_count: u32,
    /// Minimum value in this column chunk, if any
    pub(crate) min_value: Option<Box<dyn Array>>,
    /// Maximum value in this column chunk, if any
    pub(crate) max_value: Option<Box<dyn Array>>,
}

pub fn collect_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    todo!();
}

pub struct StatisticsCollector {
    builders: HashMap<i32, (Field, StatisticsBuilder)>,
}

impl StatisticsCollector {
    pub fn new(fields: &[Field]) -> Self {
        let builders = fields
            .iter()
            .map(|f| (f.id, (f.clone(), StatisticsBuilder::new(&f.data_type()))))
            .collect();
        Self { builders }
    }

    pub fn get_builder(&mut self, field_id: i32) -> Option<&mut StatisticsBuilder> {
        self.builders.get_mut(&field_id).map(|(_, b)| b)
    }

    pub fn finish(&mut self) -> RecordBatch {
        todo!()
    }
}

pub struct StatisticsBuilder {
    null_count: PrimitiveBuilder<UInt32Type>,
    min_value: Box<dyn ArrayBuilder>,
    max_value: Box<dyn ArrayBuilder>,
}

impl StatisticsBuilder {
    fn new(data_type: &DataType) -> Self {
        let null_count = PrimitiveBuilder::<UInt32Type>::new();
        let min_value = make_builder(data_type, 1);
        let max_value = make_builder(data_type, 1);
        Self {
            null_count,
            min_value,
            max_value,
        }
    }

    pub fn append(&mut self, row: StatisticsRow) {
        todo!()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_collect_primitive_stats() {
        // Test ints, dates, etc.
    }

    #[test]
    fn test_collect_float_stats() {
        // Test floats with all edge cases like Inf, -Inf, -0, NaN.
    }

    #[test]
    fn test_collect_binary_stats() {
        // Test string, binary with truncation and null values.
    }

    #[test]
    fn test_collect_dictionary_stats() {
        // Test dictionary with null values.
    }
}
