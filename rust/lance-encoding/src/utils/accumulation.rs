// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An accumulation queue accumulates arrays until we have enough data to flush.

use arrow_array::{Array, ArrayRef, cast::AsArray};
use arrow_schema::DataType;
use lance_arrow::deepcopy::{deep_copy_array, deep_copy_array_sliced};
use log::{debug, trace};

fn is_view(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8View | DataType::BinaryView)
}

fn is_dictionary_or_view(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Dictionary(_, _)) || is_view(data_type)
}

/// Whether `pred` holds for the type or one nested in it.  Fixed size lists and packed
/// structs reach the queue whole; list items are handed to their own encoders.
fn contains(data_type: &DataType, pred: fn(&DataType) -> bool) -> bool {
    pred(data_type)
        || match data_type {
            DataType::FixedSizeList(item, _) => contains(item.data_type(), pred),
            DataType::Struct(fields) => fields.iter().any(|f| contains(f.data_type(), pred)),
            DataType::Dictionary(_, values) => contains(values, pred),
            _ => false,
        }
}

/// Bytes the array refers to.  A zero-copy slice shares its parent's buffers, so
/// `get_array_memory_size` would report the whole parent allocation for every slice.
fn referenced_bytes(array: &dyn Array) -> u64 {
    let whole = || array.get_array_memory_size() as u64;
    // get_slice_memory_size counts the views of a view array but not the data they point to
    let view_data = match array.data_type() {
        DataType::Utf8View => array.as_string_view().total_buffer_bytes_used(),
        DataType::BinaryView => array.as_binary_view().total_buffer_bytes_used(),
        data_type if contains(data_type, is_view) => return whole(),
        _ => 0,
    };
    array
        .to_data()
        .get_slice_memory_size()
        .map_or_else(|_| whole(), |bytes| (bytes + view_data) as u64)
}

/// Copies only the bytes the array refers to, into buffers it does not share with the
/// array: an FFI-imported array references the memory of its entire batch.
fn copy_referenced(array: &dyn Array) -> ArrayRef {
    match array.data_type() {
        // gc copies just the referenced data but may reuse the views and validity, which
        // the sliced copy then copies; MutableArrayData alone would share the data
        DataType::Utf8View => deep_copy_array_sliced(&array.as_string_view().gc()),
        DataType::BinaryView => deep_copy_array_sliced(&array.as_binary_view().gc()),
        // MutableArrayData panics on a dictionary whose key range is full and shares,
        // rather than copies, the data of view arrays
        data_type if contains(data_type, is_dictionary_or_view) => deep_copy_array(array),
        _ => deep_copy_array_sliced(array),
    }
}

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
        let array_bytes = referenced_bytes(array.as_ref());
        if self.current_bytes + array_bytes > self.cache_bytes {
            debug!(
                "Flushing column {} page of size {} bytes (unencoded)",
                self.column_index,
                self.current_bytes + array_bytes
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
            let (array, array_bytes) = if self.keep_original_array {
                (array, array_bytes)
            } else {
                // The copy holds only what it refers to, so its memory size is what buffering
                // costs, including the per-array overhead that dominates tiny batches
                let copy = copy_referenced(array.as_ref());
                let copy_bytes = copy.get_array_memory_size() as u64;
                (copy, copy_bytes)
            };
            self.current_bytes += array_bytes;
            trace!(
                "Accumulating data for column {}.  Now at {} bytes",
                self.column_index, self.current_bytes
            );
            self.buffered_arrays.push(array);
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{types::Int8Type, *};
    use arrow_data::ArrayData;

    use super::{AccumulationQueue, copy_referenced};

    #[test]
    fn test_sliced_input_accounting() {
        // counting or copying the whole parent would flush on every one of the 16 slices
        let strings: Vec<_> = (0..1024).map(|i| format!("{i:032}")).collect();
        let string_array: ArrayRef = Arc::new(StringArray::from_iter_values(&strings));
        let view_array: ArrayRef = Arc::new(StringViewArray::from_iter_values(&strings));
        for parent in [string_array, view_array] {
            let mut queue = AccumulationQueue::new(10_000, 0, false);
            let flushes = (0..16).filter_map(|i| queue.insert(parent.slice(i * 64, 64), 0, 64));
            assert!(flushes.count() <= 4, "{}", parent.data_type());
        }
    }

    #[test]
    fn test_per_array_overhead_accounting() {
        // a 1-row copy refers to 4 B but owns at least a 64 B allocation
        let mut queue = AccumulationQueue::new(64 * 1024, 0, false);
        let tiny = |i| Arc::new(Int32Array::from(vec![i])) as ArrayRef;
        let mut flushes = (0..).filter_map(|i| queue.insert(tiny(i), 0, 1));
        assert!(flushes.next().unwrap().2 <= 1024);
    }

    #[test]
    fn test_buffered_copy_isolation() {
        // An FFI-imported input keeps its whole batch alive.  Views inlined (8 B) and out of
        // line (32 B), and an Int8 dictionary using all 128 keys, which MutableArrayData rejects
        let strings = |len| (0..1024).map(move |i| (i % 7 != 0).then(|| format!("{i:0len$}")));
        let views = |len| Arc::new(StringViewArray::from_iter(strings(len))) as ArrayRef;
        let keys: Vec<_> = (0..128).map(|i| i.to_string()).collect();
        let dict: DictionaryArray<Int8Type> = keys.iter().map(String::as_str).collect();
        let ptrs = |d: ArrayData| {
            let nulls = d.nulls().map(|n| n.buffer());
            Vec::from_iter(d.buffers().iter().chain(nulls).map(|b| b.as_ptr()))
        };
        for input in [views(8), views(32), Arc::new(dict)] {
            let slice = input.slice(64, 64);
            let copy = copy_referenced(slice.as_ref());
            assert_eq!(copy.as_ref(), slice.as_ref());
            let original = ptrs(slice.to_data());
            assert!(ptrs(copy.to_data()).iter().all(|p| !original.contains(p)));
        }
    }
}
