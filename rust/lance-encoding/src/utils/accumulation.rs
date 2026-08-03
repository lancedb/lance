// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An accumulation queue accumulates arrays until we have enough data to flush.

use arrow_array::ArrayRef;
use lance_arrow::deepcopy::deep_copy_array_sliced;
use lance_arrow::memory::array_slice_memory_size;
use log::{debug, trace};

#[derive(Debug)]
pub struct AccumulationQueue {
    cache_bytes: u64,
    keep_original_array: bool,
    buffered_arrays: Vec<ArrayRef>,
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
            // Push into buffered_arrays without copy since we are about to flush anyways
            self.buffered_arrays.push(array);
            self.current_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            let num_rows = self.num_rows;
            self.num_rows = 0;
            Some((
                std::mem::take(&mut self.buffered_arrays),
                row_number,
                num_rows,
            ))
        } else {
            trace!(
                "Accumulating data for column {}.  Now at {} bytes",
                self.column_index, self.current_bytes
            );
            if self.keep_original_array {
                self.buffered_arrays.push(array);
            } else {
                self.buffered_arrays
                    .push(deep_copy_array_sliced(array.as_ref()))
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
            Some((
                std::mem::take(&mut self.buffered_arrays),
                row_number,
                num_rows,
            ))
        }
    }

    /// Estimated bytes retained by arrays that have not been emitted yet.
    pub fn pending_bytes(&self) -> u64 {
        self.current_bytes
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, StringArray};

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
}
