// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// TODO: probably move this to lance-encoding

use std::ops::Sub;

/// Delta encoding.
/// The first element is subtracted from all elements in the slice.
/// Returns the original first element and a new vector with the encoded values.
/// Panic if the slice is empty.
pub fn delta_encode<T: Sub<Output = T> + Copy>(data: &[T]) -> (T, Vec<T>) {
    let mut encoded = Vec::from_iter(data.iter().copied());
    let first = delta_encode_inplace(&mut encoded);
    return (first, encoded);
}

/// In-place delta encoding.
/// The first element is subtracted from all elements in the slice.
/// Returns the original first element.
/// Panic if the slice is empty.
pub fn delta_encode_inplace<T: Sub<Output = T> + Copy>(data: &mut [T]) -> T {
    let first = data[0];
    data.iter_mut().for_each(|v| {
        *v = *v - first;
    });
    return first;
}
