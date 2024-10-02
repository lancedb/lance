use arrow::datatypes::{
    Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int8Array, PrimitiveArray, StringArray, UInt8Array};
use arrow_schema::DataType;
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use lance_arrow::DataTypeExt;
use std::{cell::RefCell, collections::hash_map::RandomState, rc::Rc};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stat {
    Min,
    Max,
    Bitwidth,
    DataSize,
    Cardinality,
    FixedSize,
}

pub struct StatsSet {
    pub data_type: DataType,
    pub values: HashMap<Stat, Arc<dyn Array>>,
    pub children: Option<Vec<Rc<RefCell<StatsSet>>>>,
    // granuality may be added here
}

impl StatsSet {
    pub fn new_from_arrays(arrays: &[ArrayRef]) -> Self {
        let mut stats_set = StatsSet {
            data_type: arrays[0].data_type().clone(),
            values: HashMap::new(),
            children: None,
        };

        stats_set.compute(arrays);
        stats_set
    }

    pub fn get(&self, stat: Stat) -> Option<ArrayRef> {
        self.values.get(&stat).map(|array| Arc::clone(array))
    }

    pub fn compute(&mut self, arrays: &[ArrayRef]) {
        if arrays.is_empty() {
            return;
        }

        let data_size = arrays
            .iter()
            .map(|arr| arr.get_buffer_memory_size() as u64)
            .sum::<u64>();
        let data_size_array = Arc::new(PrimitiveArray::from(vec![data_size]));
        self.values.insert(Stat::DataSize, data_size_array);

        self.compute_min(arrays);

        self.compute_cadinality(arrays);

        self.compute_compressed_bit_width_for_non_neg(arrays);

        self.compute_fixed_size(arrays);
    }

    fn compute_min(&mut self, data: &[ArrayRef]) {
        match data[0].data_type() {
            DataType::Int8 => {
                let min_value = data
                    .iter()
                    .flat_map(|array| {
                        let primitive_array = array.as_any().downcast_ref::<Int8Array>().unwrap();
                        primitive_array.values().iter()
                    })
                    .min()
                    .unwrap();

                let min_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![*min_value]));
                self.values.insert(Stat::Min, min_array);
            }
            DataType::UInt8 => {
                let min_value = data
                    .iter()
                    .flat_map(|array| {
                        let primitive_array = array.as_any().downcast_ref::<UInt8Array>().unwrap();
                        primitive_array.values().iter()
                    })
                    .min()
                    .unwrap();

                let min_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![*min_value]));
                self.values.insert(Stat::Min, min_array);
            }
            _ => {}
        };
    }

    fn compute_cadinality(&mut self, arrays: &[ArrayRef]) {
        const PRECISION: u8 = 12;
        let mut res = 0f64;
        for arr in arrays {
            match arr.data_type() {
                DataType::Utf8 => {
                    let mut hll: HyperLogLogPlus<String, RandomState> =
                        HyperLogLogPlus::new(PRECISION, RandomState::new()).unwrap();

                    let string_array = arr.as_any().downcast_ref::<StringArray>().unwrap();
                    for value in string_array.iter().flatten() {
                        hll.insert(value);
                    }
                    res = hll.count();
                    let cadinality_array = Arc::new(PrimitiveArray::from(vec![res as u64]));
                    self.values.insert(Stat::Cardinality, cadinality_array);

                    // do we want to do recursive call and compute statistics for offsets and bytes here?

                }
                DataType::Int8 => {
                    let mut hll: HyperLogLogPlus<i8, RandomState> =
                        HyperLogLogPlus::new(PRECISION, RandomState::new()).unwrap();
                    let int8_array = arr.as_any().downcast_ref::<Int8Array>().unwrap();
                    for value in int8_array.values().iter() {
                        hll.insert(value);
                    }
                    res = hll.count();
                }
                DataType::UInt8 => {
                    let mut hll: HyperLogLogPlus<u8, RandomState> =
                        HyperLogLogPlus::new(PRECISION, RandomState::new()).unwrap();
                    let int8_array = arr.as_any().downcast_ref::<UInt8Array>().unwrap();
                    for value in int8_array.values().iter() {
                        hll.insert(value);
                    }
                    res = hll.count();
                }
                _ => {}
            }
        }
    }
}

impl StatsSet {
    fn compute_fixed_size(&mut self, arrays: &[ArrayRef]) {
        match arrays[0].data_type() {
            DataType::Binary | DataType::LargeBinary | DataType::Utf8 | DataType::LargeUtf8 => {
                // make sure no array has an empty string
                if !arrays.iter().all(|arr| {
                    if let Some(arr) = arr.as_string_opt::<i32>() {
                        arr.iter().flatten().all(|s| !s.is_empty())
                    } else if let Some(arr) = arr.as_binary_opt::<i32>() {
                        arr.iter().flatten().all(|s| !s.is_empty())
                    } else if let Some(arr) = arr.as_string_opt::<i64>() {
                        arr.iter().flatten().all(|s| !s.is_empty())
                    } else if let Some(arr) = arr.as_binary_opt::<i64>() {
                        arr.iter().flatten().all(|s| !s.is_empty())
                    } else {
                        panic!("wrong dtype");
                    }
                }) {
                    return;
                }

                let lengths = arrays
                    .iter()
                    .flat_map(|arr| {
                        if let Some(arr) = arr.as_string_opt::<i32>() {
                            let offsets = arr.offsets().inner();
                            offsets
                                .windows(2)
                                .map(|w| (w[1] - w[0]) as u64)
                                .collect::<Vec<_>>()
                        } else if let Some(arr) = arr.as_binary_opt::<i32>() {
                            let offsets = arr.offsets().inner();
                            offsets
                                .windows(2)
                                .map(|w| (w[1] - w[0]) as u64)
                                .collect::<Vec<_>>()
                        } else if let Some(arr) = arr.as_string_opt::<i64>() {
                            let offsets = arr.offsets().inner();
                            offsets
                                .windows(2)
                                .map(|w| (w[1] - w[0]) as u64)
                                .collect::<Vec<_>>()
                        } else if let Some(arr) = arr.as_binary_opt::<i64>() {
                            let offsets = arr.offsets().inner();
                            offsets
                                .windows(2)
                                .map(|w| (w[1] - w[0]) as u64)
                                .collect::<Vec<_>>()
                        } else {
                            panic!("wrong dtype");
                        }
                    })
                    .collect::<Vec<_>>();

                // find first non-zero value in lengths
                let first_non_zero = lengths.iter().position(|&x| x != 0);
                if let Some(first_non_zero) = first_non_zero {
                    // make sure all lengths are equal to first_non_zero length or zero
                    if !lengths
                        .iter()
                        .all(|&x| x == 0 || x == lengths[first_non_zero])
                    {
                        return;
                    }
                    let lengths_array =
                        Arc::new(PrimitiveArray::from(vec![lengths[first_non_zero]]));
                    self.values.insert(Stat::FixedSize, lengths_array);
                }
            }
            _ => {
            }
        }
    }
}

impl StatsSet {
    fn compute_compressed_bit_width_for_non_neg(&mut self, arrays: &[ArrayRef]) {
        debug_assert!(!arrays.is_empty());

        let res;
        match arrays[0].data_type() {
            DataType::UInt8 => {
                let mut global_max: u8 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<UInt8Type>>()
                        .unwrap();
                    let array_max = arrow::compute::bit_or(primitive_array);
                    global_max = global_max.max(array_max.unwrap_or(0));
                }
                let num_bits = arrays[0].data_type().byte_width() as u64 * 8
                    - global_max.leading_zeros() as u64;
                // we will have constant encoding later
                if num_bits == 0 {
                    res = 1;
                } else {
                    res = num_bits;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }

            DataType::Int8 => {
                let mut global_max_width: u64 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Int8Type>>()
                        .unwrap();
                    let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                    global_max_width =
                        global_max_width.max(8 - array_max_width.leading_zeros() as u64);
                }
                if global_max_width == 0 {
                    res = 1;
                } else {
                    res = global_max_width;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }

            DataType::UInt16 => {
                let mut global_max: u16 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<UInt16Type>>()
                        .unwrap();
                    let array_max = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                    global_max = global_max.max(array_max);
                }
                let num_bits = arrays[0].data_type().byte_width() as u64 * 8
                    - global_max.leading_zeros() as u64;
                if num_bits == 0 {
                    res = 1;
                } else {
                    res = num_bits;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }

            DataType::Int16 => {
                let mut global_max_width: u64 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Int16Type>>()
                        .unwrap();
                    let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                    global_max_width =
                        global_max_width.max(16 - array_max_width.leading_zeros() as u64);
                }
                if global_max_width == 0 {
                    res = 1;
                } else {
                    res = global_max_width;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }

            DataType::UInt32 => {
                let mut global_max: u32 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<UInt32Type>>()
                        .unwrap();
                    let array_max = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                    global_max = global_max.max(array_max);
                }
                let num_bits = arrays[0].data_type().byte_width() as u64 * 8
                    - global_max.leading_zeros() as u64;
                if num_bits == 0 {
                    res = 1;
                } else {
                    res = num_bits;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }

            DataType::Int32 => {
                let mut global_max_width: u64 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Int32Type>>()
                        .unwrap();
                    let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                    global_max_width =
                        global_max_width.max(32 - array_max_width.leading_zeros() as u64);
                }
                if global_max_width == 0 {
                    res = 1;
                } else {
                    res = global_max_width;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }

            DataType::UInt64 => {
                let mut global_max: u64 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<UInt64Type>>()
                        .unwrap();
                    let array_max = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                    global_max = global_max.max(array_max);
                }
                let num_bits = arrays[0].data_type().byte_width() as u64 * 8
                    - global_max.leading_zeros() as u64;
                if num_bits == 0 {
                    res = 1;
                } else {
                    res = num_bits;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }

            DataType::Int64 => {
                let mut global_max_width: u64 = 0;
                for array in arrays {
                    let primitive_array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Int64Type>>()
                        .unwrap();
                    let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                    global_max_width =
                        global_max_width.max(64 - array_max_width.leading_zeros() as u64);
                }
                if global_max_width == 0 {
                    res = 1;
                } else {
                    res = global_max_width;
                }
                let bitwidth_array: Arc<dyn Array> = Arc::new(PrimitiveArray::from(vec![res]));
                self.values.insert(Stat::Bitwidth, bitwidth_array);
            }
            _ => {}
        };
    }
}
