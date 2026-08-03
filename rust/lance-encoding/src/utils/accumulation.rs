// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An accumulation queue accumulates arrays until we have enough data to flush.

use std::sync::{Arc, Weak};

use arrow_array::{Array, ArrayRef, cast::AsArray, make_array};
use lance_arrow::deepcopy::deep_copy_array_sliced;
use lance_arrow::memory::array_slice_memory_size;
use log::{debug, trace};

#[derive(Debug)]
pub struct AccumulationQueue {
    cache_bytes: u64,
    keep_original_array: bool,
    buffered_arrays: Vec<ArrayRef>,
    detached_dictionary_values: Vec<(Weak<dyn Array>, ArrayRef)>,
    current_bytes: u64,
    // Row number of the first item in buffered_arrays, reset on flush
    row_number: u64,
    // Number of top level rows represented in buffered_arrays, reset on flush
    num_rows: u64,
    // This is only for logging / debugging purposes
    column_index: u32,
}

impl AccumulationQueue {
    pub fn new(cache_bytes: u64, column_index: u32, keep_original_array: bool) -> Self {
        Self {
            cache_bytes,
            buffered_arrays: Vec::new(),
            detached_dictionary_values: Vec::new(),
            current_bytes: 0,
            column_index,
            keep_original_array,
            row_number: u64::MAX,
            num_rows: 0,
        }
    }

    /// Adds an array to the queue, if there is enough data then the queue is flushed
    /// and returned
    pub fn insert(
        &mut self,
        array: ArrayRef,
        row_number: u64,
        num_rows: u64,
    ) -> Option<(Vec<ArrayRef>, u64, u64)> {
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.num_rows += num_rows;
        self.current_bytes += array_slice_memory_size(array.as_ref()) as u64;
        if self.current_bytes > self.cache_bytes {
            debug!(
                "Flushing column {} page of size {} bytes (unencoded)",
                self.column_index, self.current_bytes
            );
            // Dictionary slices already in the queue use detached values. Reuse
            // those values for the final slice so concatenation still recognizes
            // one shared dictionary. Other arrays need no copy before a flush.
            let array = if !self.keep_original_array
                && !self.buffered_arrays.is_empty()
                && array.as_any_dictionary_opt().is_some()
            {
                self.deep_copy_array_sliced(array)
            } else {
                array
            };
            self.buffered_arrays.push(array);
            self.current_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            let num_rows = self.num_rows;
            self.num_rows = 0;
            Some((self.take_buffered_arrays(), row_number, num_rows))
        } else {
            trace!(
                "Accumulating data for column {}.  Now at {} bytes",
                self.column_index, self.current_bytes
            );
            if self.keep_original_array {
                self.buffered_arrays.push(array);
            } else {
                let array = self.deep_copy_array_sliced(array);
                self.buffered_arrays.push(array)
            }
            None
        }
    }

    pub fn flush(&mut self) -> Option<(Vec<ArrayRef>, u64, u64)> {
        if self.buffered_arrays.is_empty() {
            trace!(
                "No final flush since no data at column {}",
                self.column_index
            );
            None
        } else {
            trace!(
                "Final flush of column {} which has {} bytes",
                self.column_index, self.current_bytes
            );
            self.current_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            let num_rows = self.num_rows;
            self.num_rows = 0;
            Some((self.take_buffered_arrays(), row_number, num_rows))
        }
    }

    /// Estimated bytes retained by arrays that have not been emitted yet.
    pub fn pending_bytes(&self) -> u64 {
        self.current_bytes
    }

    fn deep_copy_array_sliced(&mut self, array: ArrayRef) -> ArrayRef {
        let Some(dictionary) = array.as_any_dictionary_opt() else {
            return deep_copy_array_sliced(array.as_ref());
        };

        let source_values = dictionary.values();
        let detached_values = self
            .detached_dictionary_values
            .iter()
            .find_map(|(source, detached)| {
                source
                    .upgrade()
                    .filter(|source| Arc::ptr_eq(source, source_values))
                    .map(|_| detached.clone())
            })
            .unwrap_or_else(|| {
                let detached = deep_copy_array_sliced(source_values.as_ref());
                self.detached_dictionary_values
                    .push((Arc::downgrade(source_values), detached.clone()));
                detached
            });

        let keys = deep_copy_array_sliced(dictionary.keys()).to_data();
        // SAFETY: `keys` is a value-identical logical copy of this dictionary's
        // primitive keys. `detached_values` is a value-identical copy of the same
        // source values array, so the original key bounds and dictionary types
        // remain valid when the two are combined.
        let data = unsafe {
            keys.into_builder()
                .data_type(array.data_type().clone())
                .child_data(vec![detached_values.to_data()])
                .build_unchecked()
        };
        make_array(data)
    }

    fn take_buffered_arrays(&mut self) -> Vec<ArrayRef> {
        self.detached_dictionary_values.clear();
        std::mem::take(&mut self.buffered_arrays)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, DictionaryArray, Int8Array, StringArray, types::Int8Type};
    use rstest::rstest;

    use super::*;

    #[test]
    fn sliced_utf8_cache_does_not_copy_the_full_parent() {
        let value = "x".repeat(1024 * 1024);
        let parent = Arc::new(StringArray::from(vec![value.as_str(); 16])) as ArrayRef;
        let slice = parent.slice(0, 2);
        assert!(array_slice_memory_size(slice.as_ref()) < 3 * 1024 * 1024);

        let mut queue = AccumulationQueue::new(8 * 1024 * 1024, 0, false);
        assert!(queue.insert(slice, 0, 2).is_none());
        let retained_bytes = queue.buffered_arrays[0].get_buffer_memory_size();

        assert!(
            retained_bytes < 3 * 1024 * 1024,
            "a 2 MiB logical slice retained {retained_bytes} bytes"
        );
    }

    #[test]
    fn sliced_dictionary_values_do_not_retain_the_full_parent() {
        let value = "x".repeat(1024 * 1024);
        let parent_values = Arc::new(StringArray::from(vec![value.as_str(); 16])) as ArrayRef;
        let values = parent_values.slice(0, 2);
        let keys = Int8Array::from(vec![0_i8, 1_i8]);
        let dictionary =
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap()) as ArrayRef;
        assert!(array_slice_memory_size(dictionary.as_ref()) < 3 * 1024 * 1024);
        let expected = dictionary.to_data();

        let mut queue = AccumulationQueue::new(8 * 1024 * 1024, 0, false);
        assert!(queue.insert(dictionary, 0, 2).is_none());
        assert_eq!(queue.buffered_arrays[0].to_data(), expected);
        let retained_bytes = queue.buffered_arrays[0].get_buffer_memory_size();

        assert!(
            retained_bytes < 3 * 1024 * 1024,
            "a 2 MiB dictionary retained {retained_bytes} bytes"
        );
    }

    #[rstest]
    #[case::final_flush(false)]
    #[case::threshold_flush(true)]
    fn copied_dictionary_slices_reuse_shared_values(#[case] is_threshold_flush: bool) {
        let values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let keys = Int8Array::from((0..100).map(|index| (index % 2) as i8).collect::<Vec<_>>());
        let dictionary =
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap()) as ArrayRef;

        let slice_bytes = array_slice_memory_size(dictionary.slice(0, 10).as_ref()) as u64;
        let cache_bytes = if is_threshold_flush {
            slice_bytes * 4
        } else {
            8 * 1024 * 1024
        };
        let mut queue = AccumulationQueue::new(cache_bytes, 0, false);
        let mut flushed = None;
        for offset in (0..50).step_by(10) {
            flushed = queue.insert(dictionary.slice(offset, 10), offset as u64, 10);
        }
        let (arrays, _, num_rows) = if is_threshold_flush {
            flushed.unwrap()
        } else {
            assert!(flushed.is_none());
            queue.flush().unwrap()
        };
        let data = crate::data::DataBlock::from_arrays(&arrays, num_rows);
        let dictionary = data.as_dictionary().unwrap();

        assert_eq!(dictionary.dictionary.num_values(), 2);
    }
}
