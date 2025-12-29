// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, BooleanArray, MapArray};
use arrow_buffer::{BooleanBufferBuilder, OffsetBuffer, ScalarBuffer};

pub trait MapArrayExt {
    /// Filters out masked null items from the map array
    ///
    /// Similar to ListArrayExt::filter_garbage_nulls, but for Map arrays.
    /// Null map entries may have non-zero length with garbage values that should be ignored.
    /// This function filters the entries array to remove the garbage values.
    ///
    /// The output map will always have zero-length nulls.
    fn filter_garbage_nulls(&self) -> Self;

    /// Returns a copy of the map's entries array that has been sliced to size
    ///
    /// Similar to ListArrayExt::trimmed_values, but for Map arrays.
    fn trimmed_entries(&self) -> Arc<dyn Array>;
}

impl MapArrayExt for MapArray {
    fn filter_garbage_nulls(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let Some(validity) = self.nulls().cloned() else {
            return self.clone();
        };

        let entries = self.entries();
        let mut should_keep = BooleanBufferBuilder::new(entries.len());

        // Handle preamble (entries before first offset)
        should_keep.append_n(self.offsets().first().copied().unwrap_or(0) as usize, false);

        let mut new_offsets: Vec<i32> = Vec::with_capacity(self.len() + 1);
        new_offsets.push(0);
        let mut cur_len: i32 = 0;
        for (offset, is_valid) in self.offsets().windows(2).zip(validity.iter()) {
            let len = offset[1] - offset[0];
            should_keep.append_n(len as usize, is_valid);
            if is_valid {
                cur_len += len;
            }
            new_offsets.push(cur_len);
        }

        // Handle trailer (entries after last offset)
        should_keep.append_n(entries.len() - should_keep.len(), false);

        let should_keep = BooleanArray::new(should_keep.finish(), None);
        let new_entries = arrow_select::filter::filter(entries, &should_keep)
            .expect("filter should succeed")
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .expect("map entries should be struct")
            .clone();

        let (entries_field, keys_sorted) = match self.data_type() {
            arrow_schema::DataType::Map(field, sorted) => (field.clone(), *sorted),
            _ => unreachable!(),
        };

        Self::new(
            entries_field,
            OffsetBuffer::new(ScalarBuffer::from(new_offsets)),
            new_entries,
            Some(validity),
            keys_sorted,
        )
    }

    fn trimmed_entries(&self) -> Arc<dyn Array> {
        let first_value = self.offsets().first().copied().unwrap_or(0) as usize;
        let last_value = self.offsets().last().copied().unwrap_or(0) as usize;
        Arc::new(self.entries().slice(first_value, last_value - first_value))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, MapArray, StringArray, StructArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields};

    use super::*;

    fn make_test_map() -> MapArray {
        // Create a map array with 3 maps:
        // - map 0: {"a": 1, "b": 2} (valid, 2 entries)
        // - map 1: {"c": 3} (null but has 1 garbage entry)
        // - map 2: {"d": 4, "e": 5} (valid, 2 entries)
        let keys = StringArray::from(vec!["a", "b", "c", "d", "e"]);
        let values = Int32Array::from(vec![1, 2, 3, 4, 5]);

        let entries_fields = Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            Field::new("values", DataType::Int32, true),
        ]);
        let entries = StructArray::new(
            entries_fields.clone(),
            vec![Arc::new(keys), Arc::new(values)],
            None,
        );

        let entries_field = Arc::new(Field::new("entries", DataType::Struct(entries_fields), false));
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 3, 5]));
        let validity = NullBuffer::from(vec![true, false, true]);

        MapArray::new(entries_field, offsets, entries, Some(validity), false)
    }

    #[test]
    fn test_filter_garbage_nulls() {
        let map_arr = make_test_map();
        assert_eq!(map_arr.len(), 3);
        assert_eq!(map_arr.entries().len(), 5);

        let filtered = map_arr.filter_garbage_nulls();
        assert_eq!(filtered.len(), 3);
        // Garbage entry from null map should be removed
        assert_eq!(filtered.entries().len(), 4);

        // Check offsets: [0, 2, 2, 4] (null map now has 0 length)
        assert_eq!(filtered.value_offsets(), &[0, 2, 2, 4]);
    }

    #[test]
    fn test_trimmed_entries() {
        let map_arr = make_test_map();
        let trimmed = map_arr.trimmed_entries();
        assert_eq!(trimmed.len(), 5);
    }
}
