// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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
            Self::FixedSizeList(_) => todo!("get_stat for FixedSizeList is not implemented yet."),
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
        // compute this datablock's data_size
        let data_size = self.data_size();
        let data_size_array = Arc::new(UInt64Array::from(vec![data_size]));

        // compute this datablock's max_bit_width
        let max_bit_width = self.max_bit_width();

        let mut info = self.block_info.info.write().unwrap();
        info.insert(Stat::DataSize, data_size_array);
        info.insert(Stat::BitWidth, max_bit_width);
    }

    fn max_bit_width(&mut self) -> Arc<dyn Array> {
        match self.bits_per_value {
            8 => {
                let u8_slice_ref = self.data.borrow_to_typed_slice::<u8>();
                let u8_slice = u8_slice_ref.as_ref();
                let max_value = u8_slice.iter().fold(0, |acc, &x| acc | x);
                let max_bit_width = self.bits_per_value - max_value.leading_zeros() as u64;
                Arc::new(UInt64Array::from(vec![max_bit_width]))
            }
            16 => {
                let u16_slice_ref = self.data.borrow_to_typed_slice::<u16>();
                let u16_slice = u16_slice_ref.as_ref();
                let max_value = u16_slice.iter().fold(0, |acc, &x| acc | x);
                let max_bit_width = self.bits_per_value - max_value.leading_zeros() as u64;
                Arc::new(UInt64Array::from(vec![max_bit_width]))
            }
            32 => {
                let u32_slice_ref = self.data.borrow_to_typed_slice::<u32>();
                let u32_slice = u32_slice_ref.as_ref();
                let max_value = u32_slice.iter().fold(0, |acc, &x| acc | x);
                let max_bit_width = self.bits_per_value - max_value.leading_zeros() as u64;
                Arc::new(UInt64Array::from(vec![max_bit_width]))
            }
            64 => {
                let u64_slice_ref = self.data.borrow_to_typed_slice::<u64>();
                let u64_slice = u64_slice_ref.as_ref();
                let max_value = u64_slice.iter().fold(0, |acc, &x| acc | x);
                let max_bit_width = self.bits_per_value - max_value.leading_zeros() as u64;
                Arc::new(UInt64Array::from(vec![max_bit_width]))
            }
            // when self.bits_per_value is not (8, 16, 32, 64), it is already bit-packed or we don't
            // bit-pack them(Decimal128, Decimal256), so we return `self.bit_per_value` as their `max_bit_width`
            _ => Arc::new(UInt64Array::from(vec![self.bits_per_value])),
        }
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
    use arrow_array::{
        ArrayRef, Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };
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

    #[test]
    fn test_bit_width_stat_for_integers() {
        let int8_array = Int8Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int8_array
        );

        let int8_array = Int8Array::from(vec![0x1, 0x2, 0x3, 0x7F]);
        let array_ref: ArrayRef = Arc::new(int8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![7])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int8_array
        );

        let int8_array = Int8Array::from(vec![0x1, 0x2, 0x3, 0xF, 0x1F]);
        let array_ref: ArrayRef = Arc::new(int8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![5])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int8_array
        );

        let int8_array = Int8Array::from(vec![-1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![8])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int8_array
        );

        let int16_array = Int16Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int16_array
        );

        let int16_array = Int16Array::from(vec![0x1, 0x2, 0x3, 0x7F]);
        let array_ref: ArrayRef = Arc::new(int16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![7])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int16_array
        );

        let int16_array = Int16Array::from(vec![0x1, 0x2, 0x3, 0xFF]);
        let array_ref: ArrayRef = Arc::new(int16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![8])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int16_array
        );

        let int16_array = Int16Array::from(vec![0x1, 0x2, 0x3, 0x1FF]);
        let array_ref: ArrayRef = Arc::new(int16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![9])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int16_array
        );
        let int16_array = Int16Array::from(vec![0x1, 0x2, 0x3, 0xF, 0x1F]);
        let array_ref: ArrayRef = Arc::new(int16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![5])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int16_array
        );

        let int16_array = Int16Array::from(vec![-1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![16])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int16_array
        );

        let int32_array = Int32Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int32_array
        );

        let int32_array = Int32Array::from(vec![0x1, 0x2, 0x3, 0xFF]);
        let array_ref: ArrayRef = Arc::new(int32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![8])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int32_array
        );

        let int32_array = Int32Array::from(vec![0x1, 0x2, 0x3, 0xFF, 0x1FF]);
        let array_ref: ArrayRef = Arc::new(int32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![9])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int32_array
        );

        let int32_array = Int32Array::from(vec![-1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![32])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int32_array
        );

        let int32_array = Int32Array::from(vec![-1, 2, 3, -88]);
        let array_ref: ArrayRef = Arc::new(int32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![32])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int32_array
        );

        let int64_array = Int64Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int64_array
        );

        let int64_array = Int64Array::from(vec![0x1, 0x2, 0x3, 0xFF]);
        let array_ref: ArrayRef = Arc::new(int64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![8])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int64_array
        );

        let int64_array = Int64Array::from(vec![0x1, 0x2, 0x3, 0xFF, 0x1FF]);
        let array_ref: ArrayRef = Arc::new(int64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![9])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int64_array
        );

        let int64_array = Int64Array::from(vec![-1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![64])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int64_array
        );

        let int64_array = Int64Array::from(vec![-1, 2, 3, -88]);
        let array_ref: ArrayRef = Arc::new(int64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![64])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            int64_array
        );

        let uint8_array = UInt8Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(uint8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint8_array
        );

        let uint8_array = UInt8Array::from(vec![0x1, 0x2, 0x3, 0x7F]);
        let array_ref: ArrayRef = Arc::new(uint8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![7])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint8_array
        );

        let uint8_array = UInt8Array::from(vec![0x1, 0x2, 0x3, 0xF, 0x1F]);
        let array_ref: ArrayRef = Arc::new(uint8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![5])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint8_array
        );

        let uint8_array = UInt8Array::from(vec![1, 2, 3, 0xF]);
        let array_ref: ArrayRef = Arc::new(uint8_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![4])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint8_array
        );

        let uint16_array = UInt16Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(uint16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint16_array
        );

        let uint16_array = UInt16Array::from(vec![0x1, 0x2, 0x3, 0x7F]);
        let array_ref: ArrayRef = Arc::new(uint16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![7])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint16_array
        );

        let uint16_array = UInt16Array::from(vec![0x1, 0x2, 0x3, 0xFF]);
        let array_ref: ArrayRef = Arc::new(uint16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![8])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint16_array
        );

        let uint16_array = UInt16Array::from(vec![0x1, 0x2, 0x3, 0x1FF]);
        let array_ref: ArrayRef = Arc::new(uint16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![9])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint16_array
        );
        let uint16_array = UInt16Array::from(vec![0x1, 0x2, 0x3, 0xF, 0x1F]);
        let array_ref: ArrayRef = Arc::new(uint16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![5])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint16_array
        );

        let uint16_array = UInt16Array::from(vec![1, 2, 3, 0xFFFF]);
        let array_ref: ArrayRef = Arc::new(uint16_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![16])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint16_array
        );

        let uint32_array = UInt32Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(uint32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint32_array
        );

        let uint32_array = UInt32Array::from(vec![0x1, 0x2, 0x3, 0xFF]);
        let array_ref: ArrayRef = Arc::new(uint32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![8])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint32_array
        );

        let uint32_array = UInt32Array::from(vec![0x1, 0x2, 0x3, 0xFF, 0x1FF]);
        let array_ref: ArrayRef = Arc::new(uint32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![9])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint32_array
        );

        let uint32_array = UInt32Array::from(vec![1, 2, 3, 0xF]);
        let array_ref: ArrayRef = Arc::new(uint32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![4])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint32_array
        );

        let uint32_array = UInt32Array::from(vec![1, 2, 3, 0x77]);
        let array_ref: ArrayRef = Arc::new(uint32_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![7])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint32_array
        );

        let uint64_array = UInt64Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(uint64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint64_array
        );

        let uint64_array = UInt64Array::from(vec![0x1, 0x2, 0x3, 0xFF]);
        let array_ref: ArrayRef = Arc::new(uint64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![8])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint64_array
        );

        let uint64_array = UInt64Array::from(vec![0x1, 0x2, 0x3, 0xFF, 0x1FF]);
        let array_ref: ArrayRef = Arc::new(uint64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![9])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint64_array
        );

        let uint64_array = UInt64Array::from(vec![0, 2, 3, 0xFFFF]);
        let array_ref: ArrayRef = Arc::new(uint64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![16])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint64_array
        );

        let uint64_array = UInt64Array::from(vec![1, 2, 3, 0xFFFF_FFFF_FFFF_FFFF]);
        let array_ref: ArrayRef = Arc::new(uint64_array.clone());
        let mut block = DataBlock::from_array(array_ref);

        let expected_bit_width = Arc::new(UInt64Array::from(vec![64])) as ArrayRef;
        let actual_bit_width = block.get_stat(Stat::BitWidth);

        assert_eq!(
            actual_bit_width,
            Some(expected_bit_width.clone()),
            "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
            expected_bit_width,
            uint64_array
        );
    }

    #[test]
    fn test_bit_width_when_none() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = lance_datagen::array::rand_type(&DataType::Binary);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let mut block = DataBlock::from_array(arr.clone());
        assert_eq!(
            block.get_stat(Stat::BitWidth),
            None,
            "Expected Stat::BitWidth to be None for data block: {:?}",
            block.name()
        );
    }
}
