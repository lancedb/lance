// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    fmt,
    hash::{Hash, RandomState},
    sync::Arc,
};

use arrow_array::{Array, UInt64Array};
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use num_traits::PrimInt;

use crate::data::{
    AllNullDataBlock, DataBlock, DictionaryDataBlock, FixedWidthDataBlock, OpaqueBlock,
    StructDataBlock, VariableWidthBlock,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stat {
    BitWidth,
    DataSize,
    Cardinality,
    FixedSize,
    NullCount,
    MaxLength,
}

impl fmt::Debug for Stat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitWidth => write!(f, "BitWidth"),
            Self::DataSize => write!(f, "DataSize"),
            Self::Cardinality => write!(f, "Cardinality"),
            Self::FixedSize => write!(f, "FixedSize"),
            Self::NullCount => write!(f, "NullCount"),
            Self::MaxLength => write!(f, "MaxLength"),
        }
    }
}

impl fmt::Display for Stat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait ComputeStat {
    fn compute_stat(&mut self);
}

impl ComputeStat for DataBlock {
    fn compute_stat(&mut self) {
        match self {
            Self::Empty() => {}
            Self::Constant(_) => {}
            Self::AllNull(_) => {}
            Self::Nullable(data_block) => data_block.data.compute_stat(),
            Self::FixedWidth(data_block) => data_block.compute_stat(),
            Self::FixedSizeList(_) => {}
            Self::VariableWidth(data_block) => data_block.compute_stat(),
            Self::Opaque(data_block) => data_block.compute_stat(),
            Self::Struct(_) => {}
            Self::Dictionary(_) => {}
        }
    }
}

impl ComputeStat for VariableWidthBlock {
    fn compute_stat(&mut self) {
        if !self.block_info.0.read().unwrap().is_empty() {
            panic!("compute_stat should only be called once during DataBlock construction");
        }
        let data_size = self.data_size();
        let data_size_array = Arc::new(UInt64Array::from(vec![data_size]));

        let cardinality_array = self.cardinality();

        let max_length_array = self.max_length();

        let mut info = self.block_info.0.write().unwrap();
        info.insert(Stat::DataSize, data_size_array);
        info.insert(Stat::Cardinality, cardinality_array);
        info.insert(Stat::MaxLength, max_length_array);
    }
}

impl ComputeStat for FixedWidthDataBlock {
    fn compute_stat(&mut self) {
        // compute this datablock's data_size
        let data_size = self.data_size();
        let data_size_array = Arc::new(UInt64Array::from(vec![data_size]));

        // compute this datablock's max_bit_width
        let max_bit_widths = self.max_bit_widths();

        // the MaxLength of FixedWidthDataBlock is it's self.bits_per_value / 8
        let max_len = self.bits_per_value / 8;
        let max_len_array = Arc::new(UInt64Array::from(vec![max_len]));

        let mut info = self.block_info.0.write().unwrap();
        info.insert(Stat::DataSize, data_size_array);
        info.insert(Stat::BitWidth, max_bit_widths);
        info.insert(Stat::MaxLength, max_len_array);
    }
}

impl ComputeStat for OpaqueBlock {
    fn compute_stat(&mut self) {
        // compute this datablock's data_size
        let data_size = self.data_size();
        let data_size_array = Arc::new(UInt64Array::from(vec![data_size]));
        let mut info = self.block_info.0.write().unwrap();
        info.insert(Stat::DataSize, data_size_array);
    }
}

pub trait GetStat {
    fn get_stat(&self, stat: Stat) -> Option<Arc<dyn Array>>;
}

impl GetStat for DataBlock {
    fn get_stat(&self, stat: Stat) -> Option<Arc<dyn Array>> {
        match self {
            Self::Empty() => None,
            Self::Constant(_) => None,
            Self::AllNull(data_block) => data_block.get_stat(stat),
            Self::Nullable(data_block) => data_block.data.get_stat(stat),
            Self::FixedWidth(data_block) => data_block.get_stat(stat),
            Self::FixedSizeList(_) => None,
            Self::VariableWidth(data_block) => data_block.get_stat(stat),
            Self::Opaque(data_block) => data_block.get_stat(stat),
            Self::Struct(data_block) => data_block.get_stat(stat),
            Self::Dictionary(data_block) => data_block.get_stat(stat),
        }
    }
}

impl GetStat for VariableWidthBlock {
    fn get_stat(&self, stat: Stat) -> Option<Arc<dyn Array>> {
        match stat {
            Stat::BitWidth => None,
            Stat::NullCount => None,
            _ => {
                if self.block_info.0.read().unwrap().is_empty() {
                    panic!("get_stat should be called after statistics are computed");
                }
                self.block_info.0.read().unwrap().get(&stat).cloned()
            }
        }
    }
}

impl VariableWidthBlock {
    // Caveat: the computation here assumes VariableWidthBlock.offsets maps directly to VariableWidthBlock.data
    // without any adjustment(for example, no null_adjustment for offsets)
    fn cardinality(&mut self) -> Arc<dyn Array> {
        match self.bits_per_offset {
            32 => {
                let offsets_ref = self.offsets.borrow_to_typed_slice::<u32>();
                let offsets: &[u32] = offsets_ref.as_ref();
                const PRECISION: u8 = 12;
                let mut hll: HyperLogLogPlus<&[u8], RandomState> =
                    HyperLogLogPlus::new(PRECISION, RandomState::new()).unwrap();

                for i in 0..self.num_values as usize {
                    let start = offsets[i] as usize;
                    let end = offsets[i + 1] as usize;
                    let binary_slice = &self.data[start..end];
                    hll.insert(binary_slice);
                }
                let cardinality = hll.count() as u64;
                Arc::new(UInt64Array::from(vec![cardinality]))
            }
            64 => {
                let offsets_ref = self.offsets.borrow_to_typed_slice::<u64>();
                let offsets: &[u64] = offsets_ref.as_ref();
                const PRECISION: u8 = 12;
                let mut hll: HyperLogLogPlus<&[u8], RandomState> =
                    HyperLogLogPlus::new(PRECISION, RandomState::new()).unwrap();

                for i in 0..self.num_values as usize {
                    let start = offsets[i] as usize;
                    let end = offsets[i + 1] as usize;
                    let binary_slice = &self.data[start..end];
                    hll.insert(binary_slice);
                }
                let cardinality = hll.count() as u64;
                Arc::new(UInt64Array::from(vec![cardinality]))
            }
            _ => {
                unreachable!("the bits_per_offset of VariableWidthBlock can only be 32 or 64")
            }
        }
    }

    fn max_length(&mut self) -> Arc<dyn Array> {
        match self.bits_per_offset {
            32 => {
                let offsets = self.offsets.borrow_to_typed_slice::<u32>();
                let offsets = offsets.as_ref();
                let max_len = offsets
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .max()
                    .unwrap_or(0);
                Arc::new(UInt64Array::from(vec![max_len as u64]))
            }
            64 => {
                let offsets = self.offsets.borrow_to_typed_slice::<u64>();
                let offsets = offsets.as_ref();
                let max_len = offsets
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .max()
                    .unwrap_or(0);
                Arc::new(UInt64Array::from(vec![max_len]))
            }
            _ => {
                unreachable!("the type of offsets in VariableWidth can only be u32 or u64");
            }
        }
    }
}

impl GetStat for AllNullDataBlock {
    fn get_stat(&self, stat: Stat) -> Option<Arc<dyn Array>> {
        match stat {
            Stat::NullCount => {
                let null_count = self.num_values;
                Some(Arc::new(UInt64Array::from(vec![null_count])))
            }
            Stat::DataSize => Some(Arc::new(UInt64Array::from(vec![0]))),
            _ => None,
        }
    }
}

impl GetStat for FixedWidthDataBlock {
    fn get_stat(&self, stat: Stat) -> Option<Arc<dyn Array>> {
        match stat {
            Stat::NullCount => None,
            _ => {
                if self.block_info.0.read().unwrap().is_empty() {
                    panic!("get_stat should be called after statistics are computed");
                }
                self.block_info.0.read().unwrap().get(&stat).cloned()
            }
        }
    }
}

impl FixedWidthDataBlock {
    fn max_bit_widths(&mut self) -> Arc<dyn Array> {
        assert!(self.num_values > 0);

        const CHUNK_SIZE: usize = 1024;

        fn calculate_max_bit_width<T: PrimInt>(slice: &[T], bits_per_value: u64) -> Vec<u64> {
            slice
                .chunks(CHUNK_SIZE)
                .map(|chunk| {
                    let max_value = chunk.iter().fold(T::zero(), |acc, &x| acc | x);
                    bits_per_value - max_value.leading_zeros() as u64
                })
                .collect()
        }

        match self.bits_per_value {
            8 => {
                let u8_slice = self.data.borrow_to_typed_slice::<u8>();
                let u8_slice = u8_slice.as_ref();
                Arc::new(UInt64Array::from(calculate_max_bit_width(
                    u8_slice,
                    self.bits_per_value,
                )))
            }
            16 => {
                let u16_slice = self.data.borrow_to_typed_slice::<u16>();
                let u16_slice = u16_slice.as_ref();
                Arc::new(UInt64Array::from(calculate_max_bit_width(
                    u16_slice,
                    self.bits_per_value,
                )))
            }
            32 => {
                let u32_slice = self.data.borrow_to_typed_slice::<u32>();
                let u32_slice = u32_slice.as_ref();
                Arc::new(UInt64Array::from(calculate_max_bit_width(
                    u32_slice,
                    self.bits_per_value,
                )))
            }
            64 => {
                let u64_slice = self.data.borrow_to_typed_slice::<u64>();
                let u64_slice = u64_slice.as_ref();
                Arc::new(UInt64Array::from(calculate_max_bit_width(
                    u64_slice,
                    self.bits_per_value,
                )))
            }
            _ => Arc::new(UInt64Array::from(vec![self.bits_per_value])),
        }
    }
}

impl GetStat for OpaqueBlock {
    fn get_stat(&self, stat: Stat) -> Option<Arc<dyn Array>> {
        match stat {
            Stat::DataSize => self.block_info.0.read().unwrap().get(&stat).cloned(),
            _ => None,
        }
    }
}

impl GetStat for DictionaryDataBlock {
    fn get_stat(&self, _stat: Stat) -> Option<Arc<dyn Array>> {
        None
    }
}

impl GetStat for StructDataBlock {
    fn get_stat(&self, _stat: Stat) -> Option<Arc<dyn Array>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray,
        UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };
    use arrow_schema::{DataType, Field};
    use lance_arrow::DataTypeExt;
    use lance_datagen::{array, ArrayGeneratorExt, RowCount, DEFAULT_SEED};
    use rand::SeedableRng;

    use crate::statistics::{GetStat, Stat};

    use super::DataBlock;

    use arrow::{compute::concat, datatypes::Int32Type};
    use arrow_array::Array;
    #[test]
    fn test_data_size_stat() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, false, false]);
        let arr1 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let arr2 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let arr3 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_arrays(&[arr1.clone(), arr2.clone(), arr3.clone()], 9);

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

        // test DataType::Binary
        let mut gen = lance_datagen::array::rand_type(&DataType::Binary);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());
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
        let block = DataBlock::from_array(arr.clone());
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
        let block = DataBlock::from_array(arr.clone());
        assert!(
            block.get_stat(Stat::DataSize).is_none(),
            "Expected Stat::DataSize to be None for data block of type: {}",
            block.name()
        );

        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, false]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());
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
    }

    #[test]
    fn test_bit_width_stat_for_integers() {
        let int8_array = Int8Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int8_array.clone());
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
        let block = DataBlock::from_array(array_ref);

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
    fn test_bit_width_stat_more_than_1024() {
        for data_type in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
        ] {
            let array1 = Int64Array::from(vec![3; 1024]);
            let array2 = Int64Array::from(vec![8; 1024]);
            let array3 = Int64Array::from(vec![-1; 10]);
            let array1 = arrow_cast::cast(&array1, &data_type).unwrap();
            let array2 = arrow_cast::cast(&array2, &data_type).unwrap();
            let array3 = arrow_cast::cast(&array3, &data_type).unwrap();

            let arrays: Vec<&dyn arrow::array::Array> =
                vec![array1.as_ref(), array2.as_ref(), array3.as_ref()];
            let concatenated = concat(&arrays).unwrap();
            let block = DataBlock::from_array(concatenated.clone());

            let expected_bit_width = Arc::new(UInt64Array::from(vec![
                2,
                4,
                (data_type.byte_width() * 8) as u64,
            ])) as ArrayRef;
            let actual_bit_widths = block.get_stat(Stat::BitWidth);
            assert_eq!(
                actual_bit_widths,
                Some(expected_bit_width.clone()),
                "Expected Stat::BitWidth to be {:?} for data block generated from array: {:?}",
                expected_bit_width,
                concatenated
            );
        }
    }

    #[test]
    fn test_bit_width_when_none() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = lance_datagen::array::rand_type(&DataType::Binary);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());
        assert_eq!(
            block.get_stat(Stat::BitWidth),
            None,
            "Expected Stat::BitWidth to be None for data block: {:?}",
            block.name()
        );
    }

    #[test]
    fn test_cardinality_variable_width_datablock() {
        let string_array = StringArray::from(vec![Some("hello"), Some("world")]);
        let block = DataBlock::from_array(string_array.clone());
        let expected_cardinality = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_cardinality = block.get_stat(Stat::Cardinality);

        assert_eq!(
            actual_cardinality,
            Some(expected_cardinality.clone()),
            "Expected Stat::Cardinality to be {:?} for data block generated from array: {:?}",
            expected_cardinality,
            string_array,
        );

        let string_array = StringArray::from(vec![
            Some("to be named by variables"),
            Some("to be passed as arguments to procedures"),
            Some("to be returned as values of procedures"),
        ]);
        let block = DataBlock::from_array(string_array.clone());
        let expected_cardinality = Arc::new(UInt64Array::from(vec![3])) as ArrayRef;
        let actual_cardinality = block.get_stat(Stat::Cardinality);

        assert_eq!(
            actual_cardinality,
            Some(expected_cardinality.clone()),
            "Expected Stat::Cardinality to be {:?} for data block generated from array: {:?}",
            expected_cardinality,
            string_array,
        );

        let string_array = StringArray::from(vec![
            Some("Samuel Eilenberg"),
            Some("Saunders Mac Lane"),
            Some("Samuel Eilenberg"),
        ]);
        let block = DataBlock::from_array(string_array.clone());
        let expected_cardinality = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_cardinality = block.get_stat(Stat::Cardinality);

        assert_eq!(
            actual_cardinality,
            Some(expected_cardinality.clone()),
            "Expected Stat::Cardinality to be {:?} for data block generated from array: {:?}",
            expected_cardinality,
            string_array,
        );

        let string_array = LargeStringArray::from(vec![Some("hello"), Some("world")]);
        let block = DataBlock::from_array(string_array.clone());
        let expected_cardinality = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_cardinality = block.get_stat(Stat::Cardinality);

        assert_eq!(
            actual_cardinality,
            Some(expected_cardinality.clone()),
            "Expected Stat::Cardinality to be {:?} for data block generated from array: {:?}",
            expected_cardinality,
            string_array,
        );

        let string_array = LargeStringArray::from(vec![
            Some("to be named by variables"),
            Some("to be passed as arguments to procedures"),
            Some("to be returned as values of procedures"),
        ]);
        let block = DataBlock::from_array(string_array.clone());
        let expected_cardinality = Arc::new(UInt64Array::from(vec![3])) as ArrayRef;
        let actual_cardinality = block.get_stat(Stat::Cardinality);

        assert_eq!(
            actual_cardinality,
            Some(expected_cardinality.clone()),
            "Expected Stat::Cardinality to be {:?} for data block generated from array: {:?}",
            expected_cardinality,
            string_array,
        );

        let string_array = LargeStringArray::from(vec![
            Some("Samuel Eilenberg"),
            Some("Saunders Mac Lane"),
            Some("Samuel Eilenberg"),
        ]);
        let block = DataBlock::from_array(string_array.clone());
        let expected_cardinality = Arc::new(UInt64Array::from(vec![2])) as ArrayRef;
        let actual_cardinality = block.get_stat(Stat::Cardinality);

        assert_eq!(
            actual_cardinality,
            Some(expected_cardinality.clone()),
            "Expected Stat::Cardinality to be {:?} for data block generated from array: {:?}",
            expected_cardinality,
            string_array,
        );
    }

    #[test]
    fn test_max_length_variable_width_datablock() {
        let string_array = StringArray::from(vec![Some("hello"), Some("world")]);
        let block = DataBlock::from_array(string_array.clone());

        let expected_max_length =
            Arc::new(UInt64Array::from(vec![string_array.value_length(0) as u64])) as ArrayRef;
        let actual_max_length = block.get_stat(Stat::MaxLength);

        assert_eq!(actual_max_length, Some(expected_max_length.clone()),);

        let string_array = StringArray::from(vec![
            Some("to be named by variables"),
            Some("to be passed as arguments to procedures"), // string that has max length
            Some("to be returned as values of procedures"),
        ]);
        let block = DataBlock::from_array(string_array.clone());

        let expected_max_length =
            Arc::new(UInt64Array::from(vec![string_array.value_length(1) as u64])) as ArrayRef;
        let actual_max_length = block.get_stat(Stat::MaxLength);

        assert_eq!(actual_max_length, Some(expected_max_length));

        let string_array = StringArray::from(vec![
            Some("Samuel Eilenberg"),
            Some("Saunders Mac Lane"), // string that has max length
            Some("Samuel Eilenberg"),
        ]);
        let block = DataBlock::from_array(string_array.clone());

        let expected_max_length =
            Arc::new(UInt64Array::from(vec![string_array.value_length(1) as u64])) as ArrayRef;
        let actual_max_length = block.get_stat(Stat::MaxLength);

        assert_eq!(actual_max_length, Some(expected_max_length),);

        let string_array = LargeStringArray::from(vec![Some("hello"), Some("world")]);
        let block = DataBlock::from_array(string_array.clone());

        let expected_max_length =
            Arc::new(UInt64Array::from(vec![string_array.value(0).len() as u64])) as ArrayRef;
        let actual_max_length = block.get_stat(Stat::MaxLength);

        assert_eq!(actual_max_length, Some(expected_max_length),);

        let string_array = LargeStringArray::from(vec![
            Some("to be named by variables"),
            Some("to be passed as arguments to procedures"), // string that has max length
            Some("to be returned as values of procedures"),
        ]);
        let block = DataBlock::from_array(string_array.clone());

        let expected_max_length =
            Arc::new(UInt64Array::from(vec![string_array.value_length(1) as u64])) as ArrayRef;
        let actual_max_length = block.get_stat(Stat::MaxLength);

        assert_eq!(actual_max_length, Some(expected_max_length));
    }
}
