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

use std::cmp::Ordering;
use std::{collections::hash_map::DefaultHasher, hash::Hash, hash::Hasher};

use arrow::array::{as_largestring_array, as_string_array};
use arrow_array::{
    cast::as_primitive_array,
    types::{
        Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrowNumericType, GenericStringArray, OffsetSizeTrait, PrimitiveArray, UInt64Array,
};
use arrow_schema::DataType;
use num_traits::bounds::Bounded;

use crate::{Result};

/// Argmax on a [PrimitiveArray].
///
/// Returns the index of the max value in the array.
pub fn argmax<T: ArrowNumericType>(array: &PrimitiveArray<T>) -> Option<u32>
where
    T::Native: PartialOrd + Bounded,
{
    let mut max_idx: Option<u32> = None;
    let mut max_value = T::Native::min_value();
    for (idx, value) in array.iter().enumerate() {
        if let Some(value) = value {
            if let Some(Ordering::Greater) = value.partial_cmp(&max_value) {
                max_value = value;
                max_idx = Some(idx as u32);
            }
        }
    }
    max_idx
}

/// Argmin on a [PrimitiveArray].
///
/// Returns the index of the min value in the array.
pub fn argmin<T: ArrowNumericType>(array: &PrimitiveArray<T>) -> Option<u32>
where
    T::Native: PartialOrd + Bounded,
{
    let mut min_idx: Option<u32> = None;
    let mut min_value = T::Native::max_value();
    for (idx, value) in array.iter().enumerate() {
        if let Some(value) = value {
            if let Some(Ordering::Less) = value.partial_cmp(&min_value) {
                min_value = value;
                min_idx = Some(idx as u32);
            }
        }
    }
    min_idx
}

fn hash_numeric_type<T: ArrowNumericType>(array: &PrimitiveArray<T>) -> Result<UInt64Array>
where
    T::Native: Hash,
{
    let mut builder = UInt64Array::builder(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
        } else {
            let mut s = DefaultHasher::new();
            array.value(i).hash(&mut s);
            builder.append_value(s.finish());
        }
    }
    Ok(builder.finish())
}

fn hash_string_type<O: OffsetSizeTrait>(array: &GenericStringArray<O>) -> Result<UInt64Array> {
    let mut builder = UInt64Array::builder(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
        } else {
            let mut s = DefaultHasher::new();
            array.value(i).hash(&mut s);
            builder.append_value(s.finish());
        }
    }
    Ok(builder.finish())
}

/// Calculate hash values for an Arrow Array, using `std::hash::Hash` in rust.
pub fn hash(array: &dyn Array) -> Result<UInt64Array> {
    match array.data_type() {
        DataType::UInt8 => hash_numeric_type(as_primitive_array::<UInt8Type>(array)),
        DataType::UInt16 => hash_numeric_type(as_primitive_array::<UInt16Type>(array)),
        DataType::UInt32 => hash_numeric_type(as_primitive_array::<UInt32Type>(array)),
        DataType::UInt64 => hash_numeric_type(as_primitive_array::<UInt64Type>(array)),
        DataType::Int8 => hash_numeric_type(as_primitive_array::<Int8Type>(array)),
        DataType::Int16 => hash_numeric_type(as_primitive_array::<Int16Type>(array)),
        DataType::Int32 => hash_numeric_type(as_primitive_array::<Int32Type>(array)),
        DataType::Int64 => hash_numeric_type(as_primitive_array::<Int64Type>(array)),
        DataType::Utf8 => hash_string_type(as_string_array(array)),
        DataType::LargeUtf8 => hash_string_type(as_largestring_array(array)),
        _ => Err(Error::Arrow {
            message: format!(
                "Hash only supports integer or string array, got: {}",
                array.data_type()
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        collections::HashSet,
        f32::{INFINITY, NAN, NEG_INFINITY},
    };

    use arrow_array::{
        Float32Array, Int16Array, Int8Array, LargeStringArray, StringArray, UInt32Array, UInt8Array,
    };

    #[test]
    fn test_argmax() {
        let f = Float32Array::from(vec![1.0, 5.0, 3.0, 2.0, 20.0, 8.2, 3.5]);
        assert_eq!(argmax(&f), Some(4));

        let f = Float32Array::from(vec![1.0, 5.0, NAN, 3.0, 2.0, 20.0, INFINITY, 3.5]);
        assert_eq!(argmax(&f), Some(6));

        let f = Float32Array::from_iter(vec![Some(2.0), None, Some(20.0), Some(NAN)]);
        assert_eq!(argmax(&f), Some(2));

        let f = Float32Array::from(vec![NAN, NAN, NAN]);
        assert_eq!(argmax(&f), None);

        let i = Int16Array::from(vec![1, 5, 3, 2, 20, 8, 16]);
        assert_eq!(argmax(&i), Some(4));

        let u = UInt32Array::from(vec![1, 5, 3, 2, 20, 8, 16]);
        assert_eq!(argmax(&u), Some(4));

        let empty_vec: Vec<i16> = vec![];
        let emtpy = Int16Array::from(empty_vec);
        assert_eq!(argmax(&emtpy), None)
    }

    #[test]
    fn test_argmin() {
        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, 20.0, 8.2, 3.5]);
        assert_eq!(argmin(&f), Some(2));

        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, 20.0, NAN]);
        assert_eq!(argmin(&f), Some(2));

        let f = Float32Array::from_iter(vec![Some(2.0), None, Some(NAN)]);
        assert_eq!(argmin(&f), Some(0));

        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, NEG_INFINITY, NAN]);
        assert_eq!(argmin(&f), Some(3));

        let f = Float32Array::from_iter(vec![NAN, NAN, NAN, NAN]);
        assert_eq!(argmin(&f), None);

        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, 20.0, 8.2, 3.5]);
        assert_eq!(argmin(&f), Some(2));

        let i = Int16Array::from_iter(vec![5, 3, 2, 20, 8, 16]);
        assert_eq!(argmin(&i), Some(2));

        let u = UInt32Array::from_iter(vec![5, 3, 2, 20, 8, 16]);
        assert_eq!(argmin(&u), Some(2));

        let empty_vec: Vec<i16> = vec![];
        let emtpy = Int16Array::from(empty_vec);
        assert_eq!(argmin(&emtpy), None)
    }

    #[test]
    fn test_numeric_hashes() {
        let a: UInt8Array = [1_u8, 2, 3, 4, 5].iter().copied().collect();
        let ha = hash(&a).unwrap();
        let distinct_values: HashSet<u64> = ha.values().iter().copied().collect();
        assert_eq!(distinct_values.len(), 5, "hash should be distinct");

        let b: Int8Array = [1_i8, 2, 3, 4, 5].iter().copied().collect();
        let hb = hash(&b).unwrap();

        assert_eq!(ha, hb, "hash of the same numeric value should be the same");
    }

    #[test]
    fn test_string_hashes() {
        let a = StringArray::from(vec!["a", "b", "ccc", "dec", "e", "a"]);
        let h = hash(&a).unwrap();
        // first and last value are the same.
        assert_eq!(h.value(0), h.value(5));

        // Other than that, all values should be distinct
        let distinct_values: HashSet<u64> = h.values().iter().copied().collect();
        assert_eq!(distinct_values.len(), 5);

        let a = LargeStringArray::from(vec!["a", "b", "ccc", "dec", "e", "a"]);
        let h = hash(&a).unwrap();
        // first and last value are the same.
        assert_eq!(h.value(0), h.value(5));
    }

    #[test]
    fn test_hash_unsupported_type() {
        let a = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(hash(&a).is_err());
    }
}
