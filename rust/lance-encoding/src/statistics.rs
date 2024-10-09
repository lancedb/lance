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

use arrow_array::{Array, Int64Array, UInt64Array};

use crate::data::{
    DataBlock, DictionaryDataBlock, FixedWidthDataBlock, NullableDataBlock, OpaqueBlock,
    StructDataBlock, VariableWidthBlock,
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
                // the null count is not calculated here as this enum is going to deprecated anyway
                Some(Arc::new(Int64Array::from(vec![0i64])))
            }
            Self::Nullable(data_block) => data_block.get_stat(stat),
            Self::FixedWidth(data_block) => data_block.get_stat(stat),
            Self::FixedSizeList(data_block) => data_block.child.get_stat(stat),
            Self::VariableWidth(data_block) => data_block.get_stat(stat),
            Self::Opaque(data_block) => data_block.get_stat(stat),
            Self::Struct(data_block) => data_block.get_stat(stat),
            Self::Dictionary(data_block) => data_block.get_stat(stat),
        }
    }
}

impl GetStat for NullableDataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        // Initialize statistics if not already done
        if self.block_info.info.read().unwrap().is_empty() {
            self.compute_statistics();
        }

        match stat {
            Stat::BitWidth | Stat::Cardinality | Stat::FixedSize => self.data.get_stat(stat),
            Stat::DataSize | Stat::NullCount => {
                self.block_info.info.read().unwrap().get(&stat).cloned()
            }
        }
    }
}

impl GetStat for FixedWidthDataBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        // initialize statistics
        if self.block_info.info.read().unwrap().is_empty() {
            self.compute_statistics();
        }
        match stat {
            Stat::NullCount => None,
            _ => self.block_info.info.read().unwrap().get(&stat).cloned(),
        }
    }
}

impl GetStat for VariableWidthBlock {
    fn get_stat(&mut self, stat: Stat) -> Option<Arc<dyn Array>> {
        if self.block_info.info.read().unwrap().is_empty() {
            self.compute_statistics();
        }
        match stat {
            Stat::BitWidth => None,
            Stat::NullCount => None,
            _ => self.block_info.info.read().unwrap().get(&stat).cloned(),
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

impl NullableDataBlock {
    fn compute_statistics(&mut self) {
        let null_count = self.count_nulls();
        let null_count_array = Arc::new(UInt64Array::from(vec![null_count]));

        let data_size = self.data_size();
        let data_size_array = Arc::new(UInt64Array::from(vec![data_size]));

        let mut info = self.block_info.info.write().unwrap();
        info.insert(Stat::NullCount, null_count_array);
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
    use arrow_array::{ArrayRef, Int32Array, StringArray, UInt64Array};
    use lance_datagen::{array, ArrayGeneratorExt, RowCount, DEFAULT_SEED};
    use rand::SeedableRng;

    use crate::statistics::{GetStat, Stat};

    use super::DataBlock;

    use arrow::compute::concat;
    use arrow_array::Array;
    #[test]
    fn test_count_nulls() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);

        // test NullableDatablock with FixedWidthDataBlock inside
        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, false, true]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 1);

        let arr = gen.generate(RowCount::from(6), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 2);

        let arr = gen.generate(RowCount::from(9), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 3);

        let arr = gen.generate(RowCount::from(90), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 30);

        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, true]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 2);

        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, true]);
        let arr = gen.generate(RowCount::from(6), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 4);

        let arr = gen.generate(RowCount::from(9), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 6);

        let arr = gen.generate(RowCount::from(90), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 60);

        // test NullableDatablock with VariableWidthDataBlock inside
        let mut gen = array::rand_varbin(8.into(), false).with_nulls(&[false, false, true]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 1);

        let arr = gen.generate(RowCount::from(6), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 2);

        let arr = gen.generate(RowCount::from(9), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 3);

        let arr = gen.generate(RowCount::from(90), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 30);

        let mut gen = array::rand_varbin(8.into(), false).with_nulls(&[false, true, true]);

        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 2);

        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, true]);
        let arr = gen.generate(RowCount::from(6), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 4);

        let arr = gen.generate(RowCount::from(9), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 6);

        let arr = gen.generate(RowCount::from(90), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 60);

        let mut gen = array::rand_varbin(800.into(), false).with_nulls(&[false, true, true]);

        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 2);

        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, true]);
        let arr = gen.generate(RowCount::from(6), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 4);

        let arr = gen.generate(RowCount::from(9), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 6);

        let arr = gen.generate(RowCount::from(90), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 60);

        // test NullableDatablock with VariableWidthDataBlock inside, with datatype LargeString
        let mut gen = array::rand_varbin(800.into(), true).with_nulls(&[false, true, true]);

        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr);
        assert!(block.as_nullable().unwrap().count_nulls() == 2);
    }
    #[test]
    fn test_null_count_stat() {
        // Converting string arrays that contain nulls to DataBlock
        let strings1 = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let strings2 = StringArray::from(vec![Some("a"), Some("b")]);
        let strings3 = StringArray::from(vec![Option::<&'static str>::None, None]);

        let arrays = &[strings1, strings2, strings3]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let mut block = DataBlock::from_arrays(arrays, 7);

        let null_count_array = block.get_stat(Stat::NullCount).unwrap();
        let null_count = null_count_array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        assert!(null_count == 3);

        let ints1 = Int32Array::from(vec![Some(1), None, Some(2)]);
        let ints2 = Int32Array::from(vec![Some(3), Some(4)]);
        let ints3 = Int32Array::from(vec![None, None]);

        let arrays = &[ints1, ints2, ints3]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let mut block = DataBlock::from_arrays(arrays, 7);

        let null_count_array = block.get_stat(Stat::NullCount).unwrap();
        let null_count = null_count_array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        assert!(null_count == 3);

        let ints1 = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
        let ints2 = Int32Array::from(vec![Some(3), Some(4)]);
        let ints3 = Int32Array::from(vec![Some(5), Some(6)]);

        let arrays = &[ints1, ints2, ints3]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let mut block = DataBlock::from_arrays(arrays, 7);

        assert!(block.get_stat(Stat::NullCount).is_none());
    }

    #[test]
    fn test_data_size_stat() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, false]);
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

        if let Some(data_size_array) = block.get_stat(Stat::DataSize) {
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
            let total_nulls_size_in_bytes = (concatenated_array.nulls().unwrap().len() + 7) / 8;
            assert!(data_size == (total_buffer_size + total_nulls_size_in_bytes) as u64);
        } else {
            panic!(
                "A data block of type: {} should have valid {:?} statistics",
                block.name(),
                Stat::DataSize
            );
        }
    }
}
