// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

/// Bits per value for FixedWidth dictionary values (currently only 128-bit is supported)
pub const DICT_FIXED_WIDTH_BITS_PER_VALUE: u64 = 128;
/// Bits per index for dictionary indices (always i32)
pub const DICT_INDICES_BITS_PER_VALUE: u64 = 32;

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
        UInt64Type, UInt8Type,
    },
    Array, DictionaryArray, PrimitiveArray, UInt64Array,
};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use arrow_select::take::TakeOptions;
use lance_core::{error::LanceOptionExt, utils::hash::U8SliceKey, Error, Result};
use snafu::location;

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    statistics::{ComputeStat, GetStat, Stat},
};

// Helper function for normalize_dict_nulls
fn normalize_dict_nulls_impl<K: ArrowDictionaryKeyType>(
    array: Arc<dyn Array>,
) -> Result<Arc<dyn Array>> {
    // TODO: Fast path when there is only one null index? (common case)

    let dict_array = array.as_dictionary_opt::<K>().expect_ok()?;

    if dict_array.values().null_count() == 0 {
        return Ok(array);
    }

    let mut mapping = vec![None; dict_array.values().len()];
    let mut skipped = 0;
    let mut valid_indices = Vec::with_capacity(dict_array.values().len());
    for (old_idx, is_valid) in dict_array.values().nulls().expect_ok()?.iter().enumerate() {
        if is_valid {
            // Should be safe since we are only decreasing K values (e.g. won't overflow u8 keys into u16)
            mapping[old_idx] = Some(K::Native::from_usize(old_idx - skipped).expect_ok()?);
            valid_indices.push(old_idx as u64);
        } else {
            skipped += 1;
            mapping[old_idx] = None;
        }
    }

    let mut keys_builder = PrimitiveArray::<K>::builder(dict_array.keys().len());
    for key in dict_array.keys().iter() {
        if let Some(key) = key {
            if let Some(mapped) = mapping[key.to_usize().expect_ok()?] {
                // Valid item
                keys_builder.append_value(mapped);
            } else {
                // Null via values
                keys_builder.append_null();
            }
        } else {
            // Null via keys
            keys_builder.append_null();
        }
    }
    let keys = keys_builder.finish();

    let valid_indices = UInt64Array::from(valid_indices);
    let values = arrow_select::take::take(
        dict_array.values(),
        &valid_indices,
        Some(TakeOptions {
            check_bounds: false,
        }),
    )?;

    Ok(Arc::new(DictionaryArray::new(keys, values)) as Arc<dyn Array>)
}

/// In Arrow a dictionary array can have nulls in two different places:
/// 1. The keys can be null
/// 2. The values can be null
///
/// We want to normalize this so that all nulls are in the keys.  This way we can store
/// the nulls with the keys as rep-def values the same as any other array.
pub fn normalize_dict_nulls(array: Arc<dyn Array>) -> Result<Arc<dyn Array>> {
    match array.data_type() {
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::UInt8 => normalize_dict_nulls_impl::<UInt8Type>(array),
            DataType::UInt16 => normalize_dict_nulls_impl::<UInt16Type>(array),
            DataType::UInt32 => normalize_dict_nulls_impl::<UInt32Type>(array),
            DataType::UInt64 => normalize_dict_nulls_impl::<UInt64Type>(array),
            DataType::Int8 => normalize_dict_nulls_impl::<Int8Type>(array),
            DataType::Int16 => normalize_dict_nulls_impl::<Int16Type>(array),
            DataType::Int32 => normalize_dict_nulls_impl::<Int32Type>(array),
            DataType::Int64 => normalize_dict_nulls_impl::<Int64Type>(array),
            _ => Err(Error::NotSupported {
                source: format!("Unsupported dictionary key type: {}", key_type).into(),
                location: location!(),
            }),
        },
        _ => Err(Error::Internal {
            message: format!("Data type is not a dictionary: {}", array.data_type()),
            location: location!(),
        }),
    }
}

fn dict_encode_variable_width<T>(
    variable_width_data_block: &VariableWidthBlock,
    bits_per_offset: u8,
    cardinality: u64,
) -> Option<(DataBlock, DataBlock)>
where
    T: ArrowNativeType,
    usize: TryFrom<T>,
{
    let mut map = HashMap::new();
    let offsets = variable_width_data_block
        .offsets
        .borrow_to_typed_slice::<T>();
    let offsets = offsets.as_ref();

    let max_len = variable_width_data_block
        .get_stat(Stat::MaxLength)
        .expect("VariableWidth DataBlock should have valid `Stat::MaxLength` statistics");
    let max_len = max_len.as_primitive::<UInt64Type>().value(0);

    let mut dictionary_buffer: Vec<u8> = Vec::with_capacity((max_len * cardinality) as usize);
    let mut dictionary_offsets_buffer = vec![T::default()];
    let mut curr_idx = 0;
    let mut indices_buffer = Vec::with_capacity(variable_width_data_block.num_values as usize);

    for window in offsets.windows(2) {
        let start = usize::try_from(window[0]).ok().unwrap();
        let end = usize::try_from(window[1]).ok().unwrap();

        let key = &variable_width_data_block.data[start..end];

        let idx: i32 = *map.entry(U8SliceKey(key)).or_insert_with(|| {
            dictionary_buffer.extend_from_slice(key);
            dictionary_offsets_buffer.push(T::from_usize(dictionary_buffer.len()).unwrap());
            curr_idx += 1;
            curr_idx - 1
        });

        indices_buffer.push(idx);

        if dictionary_buffer.len()
            + indices_buffer.len() * 32 / 8
            + dictionary_offsets_buffer.len() * (bits_per_offset / 8) as usize
            > variable_width_data_block.data_size() as usize
        {
            return None;
        }
    }

    let dictionary_data_block = DataBlock::VariableWidth(VariableWidthBlock {
        data: LanceBuffer::reinterpret_vec(dictionary_buffer),
        offsets: LanceBuffer::reinterpret_vec(dictionary_offsets_buffer),
        bits_per_offset,
        num_values: curr_idx as u64,
        block_info: BlockInfo::default(),
    });

    let mut indices_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(indices_buffer),
        bits_per_value: DICT_INDICES_BITS_PER_VALUE,
        num_values: variable_width_data_block.num_values,
        block_info: BlockInfo::default(),
    });
    indices_data_block.compute_stat();

    Some((indices_data_block, dictionary_data_block))
}

/// Dictionary encodes a data block
///
/// Currently only supported for some common cases (string / binary / u128)
///
/// Returns a block of indices (will always be a fixed width data block) and a block of dictionary
pub fn dictionary_encode(mut data_block: DataBlock) -> Option<(DataBlock, DataBlock)> {
    let cardinality = data_block
        .get_stat(Stat::Cardinality)
        .unwrap()
        .as_primitive::<UInt64Type>()
        .value(0);
    let data_block_size = data_block.data_size() as usize;
    match data_block {
        DataBlock::FixedWidth(ref mut fixed_width_data_block) => {
            // Currently FixedWidth DataBlock with only bits_per_value 128 has cardinality
            // TODO: a follow up PR to support `FixedWidth DataBlock with bits_per_value == 256`.
            let mut map = HashMap::new();
            let u128_slice = fixed_width_data_block.data.borrow_to_typed_slice::<u128>();
            let u128_slice = u128_slice.as_ref();
            let mut dictionary_buffer = Vec::with_capacity(cardinality as usize);
            let mut indices_buffer = Vec::with_capacity(fixed_width_data_block.num_values as usize);
            let mut curr_idx: i32 = 0;

            for &value in u128_slice.iter() {
                let idx = *map.entry(value).or_insert_with(|| {
                    dictionary_buffer.push(value);
                    curr_idx += 1;
                    curr_idx - 1
                });
                indices_buffer.push(idx);
                if (dictionary_buffer.len() * 128 / 8
                    + indices_buffer.len() * DICT_INDICES_BITS_PER_VALUE as usize / 8)
                    > data_block_size
                {
                    return None;
                }
            }
            let dictionary_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(dictionary_buffer),
                bits_per_value: DICT_FIXED_WIDTH_BITS_PER_VALUE,
                num_values: curr_idx as u64,
                block_info: BlockInfo::default(),
            });
            let mut indices_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(indices_buffer),
                bits_per_value: DICT_INDICES_BITS_PER_VALUE,
                num_values: fixed_width_data_block.num_values,
                block_info: BlockInfo::default(),
            });
            // Todo: if we decide to do eager statistics computing, wrap statistics computing
            // in DataBlock constructor.
            indices_data_block.compute_stat();

            Some((indices_data_block, dictionary_data_block))
        }
        DataBlock::VariableWidth(ref variable_width_data_block) => {
            match variable_width_data_block.bits_per_offset {
                32 => dict_encode_variable_width::<u32>(variable_width_data_block, 32, cardinality),
                64 => dict_encode_variable_width::<u64>(variable_width_data_block, 64, cardinality),
                _ => unreachable!("Variable width offsets can only be 32 or 64 bits"),
            }
        }
        _ => {
            unreachable!("dictionary encode called with data block {:?}", data_block)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::LanceBuffer,
        data::{BlockInfo, FixedWidthDataBlock},
    };
    use arrow_array::{Array, StringArray};
    use std::sync::Arc;

    #[test]
    fn test_dictionary_encode_abort_fixed_width() {
        // Create a u128 block with very high cardinality where dict encoding
        // would result in larger data (dictionary overhead + indices > original)
        let num_values = 120u64;

        // Create actual data: each value is unique u128 so dictionary encode will not be helpful
        let mut data = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            data.push(i as u128);
        }

        let mut data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: DICT_FIXED_WIDTH_BITS_PER_VALUE,
            data: LanceBuffer::reinterpret_vec(data),
            num_values,
            block_info: BlockInfo::default(),
        });

        // Compute stats naturally
        data_block.compute_stat();

        // Dictionary encoding should abort and return None
        let result = dictionary_encode(data_block);
        assert!(
            result.is_none(),
            "Dictionary encoding should abort for high cardinality u128 data"
        );
    }

    #[test]
    fn test_dictionary_encode_success_fixed_width() {
        // Create a u128 block with low cardinality where dict encoding helps
        let num_values = 120u64;
        let cardinality = 3u64;

        // Create data with few unique u128 values
        let mut data = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            data.push((i % cardinality) as u128);
        }

        let mut data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: DICT_FIXED_WIDTH_BITS_PER_VALUE,
            data: LanceBuffer::reinterpret_vec(data),
            num_values,
            block_info: BlockInfo::default(),
        });

        // Compute stats naturally
        data_block.compute_stat();

        // Dictionary encoding should succeed and return Some
        let result = dictionary_encode(data_block);
        assert!(
            result.is_some(),
            "Dictionary encoding should succeed for low cardinality u128 data"
        );

        if let Some((indices, dictionary)) = result {
            // Verify indices block
            if let DataBlock::FixedWidth(indices_block) = indices {
                assert_eq!(indices_block.num_values, num_values);
                assert_eq!(indices_block.bits_per_value, DICT_INDICES_BITS_PER_VALUE);
            } else {
                panic!("Expected FixedWidth indices block");
            }

            // Verify dictionary block
            if let DataBlock::FixedWidth(dict_block) = dictionary {
                assert_eq!(dict_block.num_values, cardinality);
                assert_eq!(dict_block.bits_per_value, DICT_FIXED_WIDTH_BITS_PER_VALUE);
            } else {
                panic!("Expected FixedWidth dictionary block");
            }
        }
    }

    #[test]
    fn test_dictionary_encode_abort_variable_width() {
        // Create a variable-width block with high cardinality where dict encoding
        // won't provide sufficient benefit
        let num_values = 120u64;
        let mut values = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            values.push(format!("unique_value_{:04}", i));
        }
        let array = StringArray::from(values);
        // from_array already computes stats
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);

        // Dictionary encoding should abort and return None
        let result = dictionary_encode(data_block);
        assert!(
            result.is_none(),
            "Dictionary encoding should abort for high cardinality string data"
        );
    }

    #[test]
    fn test_dictionary_encode_success_low_cardinality() {
        // Create a variable-width block with low cardinality where dict encoding helps
        let num_values = 120u64;
        let cardinality = 3u64;

        let mut values = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            values.push(format!("value_{}", i % cardinality));
        }

        let array = StringArray::from(values);
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);

        // Dictionary encoding should succeed and return Some
        let result = dictionary_encode(data_block);
        assert!(
            result.is_some(),
            "Dictionary encoding should succeed for low cardinality data"
        );

        if let Some((indices, dictionary)) = result {
            // Verify indices block
            if let DataBlock::FixedWidth(indices_block) = indices {
                assert_eq!(indices_block.num_values, num_values);
                assert_eq!(indices_block.bits_per_value, DICT_INDICES_BITS_PER_VALUE);
            } else {
                panic!("Expected FixedWidth indices block");
            }

            // Verify dictionary block
            if let DataBlock::VariableWidth(dict_block) = dictionary {
                assert_eq!(dict_block.num_values, cardinality);
            } else {
                panic!("Expected VariableWidth dictionary block");
            }
        }
    }
}
