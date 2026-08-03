// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;

use arrow_array::{Array, RecordBatch};
use arrow_data::ArrayData;
use arrow_schema::DataType;

/// Slice memory split by whether slicing rows can reduce it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SliceMemorySize {
    /// Bytes referenced independently of the selected top-level rows.
    pub shared: usize,
    /// Bytes that scale with the selected top-level rows.
    pub incremental: usize,
}

impl SliceMemorySize {
    /// Creates an estimate whose entire size scales with the row count.
    pub fn incremental(bytes: usize) -> Self {
        Self {
            shared: 0,
            incremental: bytes,
        }
    }

    /// Returns the complete slice size.
    pub fn total(self) -> usize {
        self.shared.saturating_add(self.incremental)
    }
}

/// Estimates the bytes referenced by one array slice.
///
/// Unlike [`Array::get_array_memory_size`], this counts only the current
/// slice's buffer windows. View arrays are the exception: Arrow omits their
/// variadic data buffers from its slice-aware result, so those shared buffers
/// are counted at full capacity in the safe direction.
pub fn array_slice_memory_size(array: &dyn Array) -> usize {
    array_slice_memory_size_parts(array).total()
}

/// Estimates the bytes referenced by all array slices in a record batch.
pub fn batch_slice_memory_size(batch: &RecordBatch) -> usize {
    batch_slice_memory_size_parts(batch).total()
}

/// Estimates slice memory while separating shared dictionary values.
///
/// Slicing a dictionary changes its keys but retains its complete values
/// child. Keeping these components separate lets callers avoid treating the
/// shared values as though they shrink proportionally with the row count.
pub fn array_slice_memory_size_parts(array: &dyn Array) -> SliceMemorySize {
    array_data_slice_memory_size_parts(&array.to_data())
}

/// Estimates all batch columns while separating shared dictionary values.
pub fn batch_slice_memory_size_parts(batch: &RecordBatch) -> SliceMemorySize {
    batch
        .columns()
        .iter()
        .map(|array| array_slice_memory_size_parts(array.as_ref()))
        .fold(SliceMemorySize::default(), |size, column| SliceMemorySize {
            shared: size.shared.saturating_add(column.shared),
            incremental: size.incremental.saturating_add(column.incremental),
        })
}

fn array_data_slice_memory_size_parts(data: &ArrayData) -> SliceMemorySize {
    let children = data
        .child_data()
        .iter()
        .map(array_data_slice_memory_size_parts)
        .collect::<Vec<_>>();
    let child_total = children
        .iter()
        .fold(0_usize, |total, child| total.saturating_add(child.total()));
    let total = match data.get_slice_memory_size() {
        Ok(size) => size.saturating_add(view_data_buffers_size(data)),
        Err(_) => data.get_array_memory_size(),
    };
    let own = total.saturating_sub(child_total);

    if matches!(data.data_type(), DataType::Dictionary(_, _)) {
        SliceMemorySize {
            shared: child_total,
            incremental: own,
        }
    } else {
        children
            .into_iter()
            .fold(SliceMemorySize::incremental(own), |size, child| {
                SliceMemorySize {
                    shared: size.shared.saturating_add(child.shared),
                    incremental: size.incremental.saturating_add(child.incremental),
                }
            })
    }
}

fn view_own_data_buffers_size(data: &ArrayData) -> usize {
    if matches!(data.data_type(), DataType::Utf8View | DataType::BinaryView) {
        // buffers()[0] contains the fixed-size views and is already included by
        // get_slice_memory_size. The remaining buffers contain variadic data.
        data.buffers()
            .iter()
            .skip(1)
            .map(|buffer| buffer.capacity())
            .sum()
    } else {
        0
    }
}

fn view_data_buffers_size(data: &ArrayData) -> usize {
    data.child_data()
        .iter()
        .fold(view_own_data_buffers_size(data), |size, child| {
            size.saturating_add(view_data_buffers_size(child))
        })
}

/// Counts memory used by buffers of Arrow arrays and RecordBatches.
///
/// This is meant to capture how much memory is being used by the Arrow data
/// structures as they are. It does not represent the memory used if the data
/// were to be serialized and then deserialized. In particular:
///
/// * This does not double count memory used by buffers shared by multiple
///   arrays or batches. Round-tripped data may use more memory because of this.
/// * This counts the **total** size of the buffers, even if the array is a slice.
///   Round-tripped data may use less memory because of this.
#[derive(Default)]
pub struct MemoryAccumulator {
    seen: HashSet<usize>,
    total: usize,
}

impl MemoryAccumulator {
    pub fn record_array(&mut self, array: &dyn Array) {
        let data = array.to_data();
        self.record_array_data(&data);
    }

    fn record_array_data(&mut self, data: &ArrayData) {
        for buffer in data.buffers() {
            let ptr = buffer.as_ptr();
            if self.seen.insert(ptr as usize) {
                self.total += buffer.capacity();
            }
        }

        if let Some(nulls) = data.nulls() {
            let null_buf = nulls.inner().inner();
            let ptr = null_buf.as_ptr();
            if self.seen.insert(ptr as usize) {
                self.total += null_buf.capacity();
            }
        }

        for child in data.child_data() {
            self.record_array_data(child);
        }
    }

    pub fn record_batch(&mut self, batch: &RecordBatch) {
        for array in batch.columns() {
            self.record_array(array);
        }
    }

    pub fn total(&self) -> usize {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, DictionaryArray, Int32Array, StringArray, types::Int32Type};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    #[test]
    fn test_memory_accumulator() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let slice = batch.slice(1, 2);

        let mut acc = MemoryAccumulator::default();

        // Should record whole buffer, not just slice
        acc.record_batch(&slice);
        assert_eq!(acc.total(), 3 * std::mem::size_of::<i32>());

        // Should not double count
        acc.record_batch(&slice);
        assert_eq!(acc.total(), 3 * std::mem::size_of::<i32>());
    }

    #[test]
    fn dictionary_size_separates_shared_values_from_keys() {
        let values = Arc::new(StringArray::from(vec!["alpha", "beta"])) as ArrayRef;
        let keys = Int32Array::from(vec![0, 1, 0, 1]);
        let dictionary = DictionaryArray::<Int32Type>::try_new(keys, values.clone()).unwrap();

        let size = array_slice_memory_size_parts(&dictionary);

        assert_eq!(size.shared, array_slice_memory_size(values.as_ref()));
        assert_eq!(size.total(), array_slice_memory_size(&dictionary));
        assert!(size.incremental >= 4 * std::mem::size_of::<i32>());
    }
}
