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

use std::{fmt, sync::Arc};

use arrow_array::{Array, UInt64Array};

use crate::data::{
    DataBlock, DictionaryDataBlock, FixedWidthDataBlock, OpaqueBlock, StructDataBlock,
    VariableWidthBlock,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stat {
    BitWidth,
    DataSize,
    Cardinality,
    FixedSize,
    NullCount,
}

impl fmt::Debug for Stat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitWidth => write!(f, "BitWidth"),
            Self::DataSize => write!(f, "DataSize"),
            Self::Cardinality => write!(f, "Cardinality"),
            Self::FixedSize => write!(f, "FixedSize"),
            Self::NullCount => write!(f, "NullCount"),
        }
    }
}

impl fmt::Display for Stat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
pub trait GetStat {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>>;
}

impl GetStat for DataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        match self {
            Self::AllNull(_) => {
                //  the statistics is not calculated here as this enum is going to deprecated soon anyway
                None
            }
            Self::Nullable(_) => {
                //  the statistics is not calculated here as this enum is going to deprecated soon anyway
                None
            }
            Self::FixedWidth(data_block) => data_block.get_stat(stat),
            Self::FixedSizeList(data_block) => data_block.child.get_stat(stat),
            Self::VariableWidth(data_block) => data_block.get_stat(stat),
            Self::Opaque(data_block) => data_block.get_stat(stat),
            Self::Struct(data_block) => data_block.get_stat(stat),
            Self::Dictionary(data_block) => data_block.get_stat(stat),
        }
    }
}

impl GetStat for FixedWidthDataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        match stat {
            Stat::NullCount => None,
            _ => {
                // initialize statistics
                if self.block_info.info.read().unwrap().is_empty() {
                    self.compute_statistics();
                }
                self.block_info.info.read().unwrap().get(&stat).cloned()
            }
        }
    }
}

impl GetStat for VariableWidthBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        match stat {
            Stat::BitWidth => None,
            Stat::NullCount => None,
            _ => {
                if self.block_info.info.read().unwrap().is_empty() {
                    self.compute_statistics();
                }
                self.block_info.info.read().unwrap().get(&stat).cloned()
            }
        }
    }
}

impl VariableWidthBlock {
    fn compute_statistics(&mut self) {
        let data_size = self.data_size();
        let data_size_array = Arc::new(UInt64Array::from(vec![data_size]));
        let mut info = self.block_info.info.write().unwrap();
        info.insert(Stat::DataSize, data_size_array);
    }
}

impl FixedWidthDataBlock {
    fn compute_statistics(&mut self) {
        let data_size = self.data_size();
        let data_size_array = Arc::new(UInt64Array::from(vec![data_size]));
        let mut info = self.block_info.info.write().unwrap();
        info.insert(Stat::DataSize, data_size_array);
    }
}

impl GetStat for OpaqueBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        match stat {
            Stat::DataSize => self.block_info.info.read().unwrap().get(&stat).cloned(),
            _ => None,
        }
    }
}

impl GetStat for DictionaryDataBlock {
    fn get_stat(&mut self, _stat: Stat) -> Option<Arc<dyn Array>> {
        None
    }
}

impl GetStat for StructDataBlock {
    fn get_stat(&mut self, _stat: Stat) -> Option<Arc<dyn Array>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::Int32Type;
    use arrow_array::UInt64Array;
    use arrow_schema::{DataType, Field};
    use lance_datagen::{array, ArrayGeneratorExt, RowCount, DEFAULT_SEED};
    use rand::SeedableRng;

    use crate::statistics::{GetStat, Stat};

    use super::DataBlock;

    use arrow::compute::concat;
    use arrow_array::Array;
    #[test]
    fn test_data_size_stat() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, false, false]);
        let arr1 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let arr2 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let arr3 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_arrays(&[arr1.clone(), arr2.clone(), arr3.clone()], 9);

        let concatenated_array = concat(&[
            &*Arc::new(arr1.clone()) as &dyn Array,
            &*Arc::new(arr2.clone()) as &dyn Array,
            &*Arc::new(arr3.clone()) as &dyn Array,
        ])
        .unwrap();

        let data_size_array = block.get_stat(Stat::DataSize).unwrap_or_else(|| {
            panic!(
                "A data block of type: {} should have valid {} statistics",
                block.name(),
                Stat::DataSize
            )
        });
        let data_size = data_size_array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);

        let total_buffer_size: usize = concatenated_array
            .to_data()
            .buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum();
        assert!(data_size == total_buffer_size as u64);

        // test FixedSizeList
        let mut gen = lance_datagen::array::rand_type(&DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int32, false)),
            1024,
        ));
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        let data_size_array = block.get_stat(Stat::DataSize).unwrap_or_else(|| {
            panic!(
                "A data block of type: {} should have valid {} statistics",
                block.name(),
                Stat::DataSize
            )
        });
        let data_size = data_size_array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        // the data buffer of `FixedSizeList` resides in its `child_data`
        let total_buffer_size: usize = arr
            .to_data()
            .child_data()
            .iter()
            .flat_map(|child| child.buffers().iter().map(|buffer| buffer.len()))
            .sum();
        assert!(data_size == total_buffer_size as u64);

        // test DataType::Binary
        let mut gen = lance_datagen::array::rand_type(&DataType::Binary);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        let data_size_array = block.get_stat(Stat::DataSize).unwrap_or_else(|| {
            panic!(
                "A data block of type: {} should have valid {} statistics",
                block.name(),
                Stat::DataSize
            )
        });

        let data_size = data_size_array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        let total_buffer_size: usize = arr
            .to_data()
            .buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum();
        assert!(data_size == total_buffer_size as u64);

        // test DataType::Struct
        let fields = vec![
            Arc::new(Field::new("int_field", DataType::Int32, false)),
            Arc::new(Field::new("float_field", DataType::Float32, false)),
            Arc::new(Field::new(
                "fsl_field",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 5),
                false,
            )),
        ]
        .into();

        let mut gen = lance_datagen::array::rand_type(&DataType::Struct(fields));
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::DataSize).is_none(),
            "Expected Stat::DataSize to be None for data block of type: {}",
            block.name()
        );

        // test DataType::Dictionary
        let mut gen = array::rand_type(&DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        ));
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::DataSize).is_none(),
            "Expected Stat::DataSize to be None for data block of type: {}",
            block.name()
        );

        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, false]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::DataSize).is_none(),
            "Expected Stat::DataSize to be None for data block of type: {}",
            block.name()
        );
    }
    #[test]
    fn test_null_count_stat() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);

        // test DataType::Int32Type
        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, false, false]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::NullCount).is_none(),
            "Expected Stat::NullCount to be None for data block of type: {}",
            block.name()
        );

        // test FixedSizeList
        let mut gen = lance_datagen::array::rand_type(&DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int32, false)),
            1024,
        ));
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::NullCount).is_none(),
            "Expected Stat::NullCount to be None for data block of type: {}",
            block.name()
        );

        // test DataType::Binary
        let mut gen = lance_datagen::array::rand_type(&DataType::Binary);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::NullCount).is_none(),
            "Expected Stat::NullCount to be None for data block of type: {}",
            block.name()
        );

        // test DataType::Struct
        let fields = vec![
            Arc::new(Field::new("int_field", DataType::Int32, false)),
            Arc::new(Field::new("float_field", DataType::Float32, false)),
            Arc::new(Field::new(
                "fsl_field",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 5),
                false,
            )),
        ]
        .into();

        let mut gen = lance_datagen::array::rand_type(&DataType::Struct(fields));
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::NullCount).is_none(),
            "Expected Stat::NullCount to be None for data block of type: {}",
            block.name()
        );

        // test DataType::Dictionary
        let mut gen = array::rand_type(&DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        ));
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::NullCount).is_none(),
            "Expected Stat::NullCount to be None for data block of type: {}",
            block.name()
        );
    }
}
