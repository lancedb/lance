// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::cmp::Ordering;

use arrow_array::{ArrowNumericType, PrimitiveArray};

/// Argmax on a [PrimitiveArray].
///
/// Returns the index of the max value in the array.
pub fn argmax<T: ArrowNumericType>(array: &PrimitiveArray<T>) -> Option<u32>
where
    T::Native: PartialOrd,
{
    array
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| match (x, y) {
            (None, _) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(vx), Some(vy)) => vx.partial_cmp(vy).unwrap(),
        })
        .map(|(idx, _)| idx as u32)
}

/// Argmin on a [PrimitiveArray].
///
/// Returns the index of the min value in the array.
pub fn argmin<T: ArrowNumericType>(array: &PrimitiveArray<T>) -> Option<u32>
where
    T::Native: PartialOrd,
{
    array
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| match (x, y) {
            (None, _) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(vx), Some(vy)) => vy.partial_cmp(vx).unwrap(),
        })
        .map(|(idx, _)| idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, Int16Array, UInt32Array};

    #[test]
    fn test_argmax() {
        let f = Float32Array::from_iter(vec![1.0, 5.0, 3.0, 2.0, 20.0, 8.2, 3.5]);
        assert_eq!(argmax(&f), Some(4));

        let i = Int16Array::from_iter(vec![1, 5, 3, 2, 20, 8, 16]);
        assert_eq!(argmax(&i), Some(4));

        let u = UInt32Array::from_iter(vec![1, 5, 3, 2, 20, 8, 16]);
        assert_eq!(argmax(&u), Some(4));

        let empty_vec: Vec<i16> = vec![];
        let emtpy = Int16Array::from(empty_vec);
        assert_eq!(argmax(&emtpy), None)
    }

    #[test]
    fn test_argmin() {
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
}
