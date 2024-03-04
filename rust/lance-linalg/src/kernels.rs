// Copyright 2024 Lance Developers.
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
use std::iter::Sum;
use std::sync::Arc;
use std::{collections::hash_map::DefaultHasher, hash::Hash, hash::Hasher};

use arrow_array::{
    cast::{as_largestring_array, as_primitive_array, as_string_array, AsArray},
    types::{
        Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, ArrowNumericType, ArrowPrimitiveType, FixedSizeListArray, GenericStringArray,
    OffsetSizeTrait, PrimitiveArray, UInt64Array,
};
use arrow_schema::{ArrowError, DataType};
use lance_arrow::ArrowFloatType;
use num_traits::{bounds::Bounded, Float, Num};

use crate::{Error, MatrixView, Result};

/// Argmax on a [PrimitiveArray].
///
/// Returns the index of the max value in the array.
pub fn argmax<T: Num + Bounded + PartialOrd>(iter: impl Iterator<Item = T>) -> Option<u32> {
    let mut max_idx: Option<u32> = None;
    let mut max_value = T::min_value();
    for (idx, value) in iter.enumerate() {
        if let Some(Ordering::Greater) = value.partial_cmp(&max_value) {
            max_value = value;
            max_idx = Some(idx as u32);
        }
    }
    max_idx
}

pub fn argmax_opt<T: Num + Bounded + PartialOrd>(
    iter: impl Iterator<Item = Option<T>>,
) -> Option<u32> {
    let mut max_idx: Option<u32> = None;
    let mut max_value = T::min_value();
    for (idx, value) in iter.enumerate() {
        if let Some(value) = value {
            if let Some(Ordering::Greater) = value.partial_cmp(&max_value) {
                max_value = value;
                max_idx = Some(idx as u32);
            }
        }
    }
    max_idx
}

/// Argmin over an iterator. Fused the operation in iterator to avoid memory allocation.
///
/// Returns the index of the min value in the array.
///
pub fn argmin<T: Num + PartialOrd + Copy>(iter: impl Iterator<Item = T>) -> Option<u32>
where
    T: Bounded,
{
    argmin_value(iter).map(|(idx, _)| idx)
}

/// Return both argmin and minimal value over an iterator.
///
/// Return
/// ------
/// - `Some(idx, min_value)` or
/// - `None` if iterator is empty or all are `Nan/Inf`.
pub fn argmin_value<T: Num + Bounded + PartialOrd + Copy>(
    iter: impl Iterator<Item = T>,
) -> Option<(u32, T)> {
    argmin_value_opt(iter.map(Some))
}

/// Returns the minimal value (float) and the index (argmin) from an Iterator.
///
/// Return `None` if the iterator is empty or all are `Nan/Inf`.
pub fn argmin_value_float<T: Float>(iter: impl Iterator<Item = T>) -> Option<(u32, T)> {
    let mut min_idx = None;
    let mut min_value = T::infinity();
    for (idx, value) in iter.enumerate() {
        if value < min_value {
            min_value = value;
            min_idx = Some(idx as u32);
        }
    }
    min_idx.map(|idx| (idx, min_value))
}

pub fn argmin_value_opt<T: Num + Bounded + PartialOrd>(
    iter: impl Iterator<Item = Option<T>>,
) -> Option<(u32, T)> {
    let mut min_idx: Option<u32> = None;
    let mut min_value = T::max_value();
    for (idx, value) in iter.enumerate() {
        if let Some(value) = value {
            if let Some(Ordering::Less) = value.partial_cmp(&min_value) {
                min_value = value;
                min_idx = Some(idx as u32);
            }
        }
    }
    min_idx.map(|idx| (idx, min_value))
}

/// Argmin over an `Option<Float>` iterator.
///
#[inline]
pub fn argmin_opt<T: Num + Bounded + PartialOrd>(
    iter: impl Iterator<Item = Option<T>>,
) -> Option<u32> {
    argmin_value_opt(iter).map(|(idx, _)| idx)
}

/// L2 normalize a vector.
///
/// Returns an iterator of normalized values.
pub fn normalize<T: Float + Sum>(v: &[T]) -> impl Iterator<Item = T> + '_ {
    let l2_norm = v.iter().map(|x| x.powi(2)).sum::<T>().sqrt();
    v.iter().map(move |&x| x / l2_norm)
}

fn do_normalize_arrow<T: ArrowPrimitiveType>(arr: &dyn Array) -> Result<ArrayRef>
where
    <T as ArrowPrimitiveType>::Native: Float + Sum,
{
    let v = arr.as_primitive::<T>();
    Ok(Arc::new(PrimitiveArray::<T>::from_iter_values(normalize(v.values()))) as ArrayRef)
}

pub fn normalize_arrow(v: &dyn Array) -> Result<ArrayRef> {
    match v.data_type() {
        DataType::Float16 => do_normalize_arrow::<Float16Type>(v),
        DataType::Float32 => do_normalize_arrow::<Float32Type>(v),
        DataType::Float64 => do_normalize_arrow::<Float64Type>(v),
        _ => Err(Error::SchemaError(format!(
            "Normalize only supports float array, got: {}",
            v.data_type()
        ))),
    }
}

fn do_normalize_fsl<T: ArrowPrimitiveType + ArrowFloatType>(
    fsl: &FixedSizeListArray,
) -> Result<FixedSizeListArray>
where
    <T as ArrowPrimitiveType>::Native: Float + Sum,
{
    let mat = MatrixView::<T>::try_from(fsl).map_err(|e| {
        Error::SchemaError(format!("Convert FixedSizeList to MatrixView failed: {}", e))
    })?;
    let normalized = mat.normalize();
    normalized.try_into().map_err(|e| {
        Error::SchemaError(format!("Convert MatrixView to FixedSizeList failed: {}", e))
    })
}

/// L2 normalize a [FixedSizeListArray] (of vectors).
pub fn normalize_fsl(fsl: &FixedSizeListArray) -> Result<FixedSizeListArray> {
    match fsl.value_type() {
        DataType::Float16 => do_normalize_fsl::<Float16Type>(fsl),
        DataType::Float32 => do_normalize_fsl::<Float32Type>(fsl),
        DataType::Float64 => do_normalize_fsl::<Float64Type>(fsl),
        _ => Err(ArrowError::SchemaError(format!(
            "Normalize only supports float array, got: {}",
            fsl.value_type()
        ))),
    }
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
        _ => Err(ArrowError::SchemaError(format!(
            "Hash only supports integer or string array, got: {}",
            array.data_type()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        collections::HashSet,
        f32::{INFINITY, NAN, NEG_INFINITY},
    };

    use approx::assert_relative_eq;
    use arrow_array::{
        Float32Array, Int16Array, Int8Array, LargeStringArray, StringArray, UInt32Array, UInt8Array,
    };

    #[test]
    fn test_argmax() {
        let f = Float32Array::from(vec![1.0, 5.0, 3.0, 2.0, 20.0, 8.2, 3.5]);
        assert_eq!(argmax(f.values().iter().copied()), Some(4));

        let f = Float32Array::from(vec![1.0, 5.0, NAN, 3.0, 2.0, 20.0, INFINITY, 3.5]);
        assert_eq!(argmax_opt(f.iter()), Some(6));

        let f = Float32Array::from_iter(vec![Some(2.0), None, Some(20.0), Some(NAN)]);
        assert_eq!(argmax_opt(f.iter()), Some(2));

        let f = Float32Array::from(vec![NAN, NAN, NAN]);
        assert_eq!(argmax(f.values().iter().copied()), None);

        let i = Int16Array::from(vec![1, 5, 3, 2, 20, 8, 16]);
        assert_eq!(argmax(i.values().iter().copied()), Some(4));

        let u = UInt32Array::from(vec![1, 5, 3, 2, 20, 8, 16]);
        assert_eq!(argmax(u.values().iter().copied()), Some(4));

        let empty_vec: Vec<i16> = vec![];
        let empty = Int16Array::from(empty_vec);
        assert_eq!(argmax_opt(empty.iter()), None)
    }

    #[test]
    fn test_argmin() {
        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, 20.0, 8.2, 3.5]);
        assert_eq!(argmin(f.values().iter().copied()), Some(2));

        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, 20.0, NAN]);
        assert_eq!(argmin_opt(f.iter()), Some(2));

        let f = Float32Array::from_iter(vec![Some(2.0), None, Some(NAN)]);
        assert_eq!(argmin_opt(f.iter()), Some(0));

        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, NEG_INFINITY, NAN]);
        assert_eq!(argmin(f.values().iter().copied()), Some(3));

        let f = Float32Array::from_iter(vec![NAN, NAN, NAN, NAN]);
        assert_eq!(argmin(f.values().iter().copied()), None);

        let f = Float32Array::from_iter(vec![5.0, 3.0, 2.0, 20.0, 8.2, 3.5]);
        assert_eq!(argmin(f.values().iter().copied()), Some(2));

        let i = Int16Array::from_iter(vec![5, 3, 2, 20, 8, 16]);
        assert_eq!(argmin(i.values().iter().copied()), Some(2));

        let u = UInt32Array::from_iter(vec![5, 3, 2, 20, 8, 16]);
        assert_eq!(argmin(u.values().iter().copied()), Some(2));

        let empty_vec: Vec<i16> = vec![];
        let emtpy = Int16Array::from(empty_vec);
        assert_eq!(argmin_opt(emtpy.iter()), None)
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

    #[test]
    fn test_normalize_vector() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let l2_norm = v.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
        assert_relative_eq!(l2_norm, 55_f32.sqrt());
        let normalized = normalize(&v).collect::<Vec<f32>>();
        normalized
            .iter()
            .enumerate()
            .for_each(|(idx, &x)| assert_relative_eq!(x, (idx + 1) as f32 / 55.0_f32.sqrt()));
        assert_relative_eq!(1.0, normalized.iter().map(|&x| x.powi(2)).sum::<f32>());
    }
}
