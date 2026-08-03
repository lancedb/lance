// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, MapArray, RecordBatch, cast::AsArray};
use arrow_schema::DataType;
use lance_arrow::{
    list::ListArrayExt,
    memory::{SliceMemorySize, batch_slice_memory_size_parts},
};

/// Estimates the uncompressed values and structural levels produced by a
/// current-format writer for one logical batch slice.
pub(super) fn estimated_write_batch_size(batch: &RecordBatch) -> SliceMemorySize {
    let structural_bytes = batch.columns().iter().fold(0_u64, |bytes, array| {
        bytes.saturating_add(estimated_structural_bytes(array, 0, false, false))
    });
    let memory = batch_slice_memory_size_parts(batch);
    SliceMemorySize {
        shared: memory.shared,
        incremental: memory
            .incremental
            .saturating_add(structural_bytes.min(usize::MAX as u64) as usize),
    }
}

fn estimated_structural_bytes(
    array: &ArrayRef,
    ancestor_slots: u64,
    has_repetition: bool,
    has_definition: bool,
) -> u64 {
    let slots = ancestor_slots.saturating_add(array.len() as u64);
    let has_definition = has_definition || array.null_count() > 0;

    match array.data_type() {
        DataType::Struct(_) => {
            let struct_array = array.as_struct();
            struct_array.columns().iter().fold(0, |bytes, child| {
                let child = if child.len() == struct_array.len() {
                    child.clone()
                } else {
                    child.slice(struct_array.offset(), struct_array.len())
                };
                bytes.saturating_add(estimated_structural_bytes(
                    &child,
                    slots,
                    has_repetition,
                    has_definition,
                ))
            })
        }
        DataType::List(_) => {
            estimated_structural_bytes(&array.as_list::<i32>().trimmed_values(), slots, true, true)
        }
        DataType::LargeList(_) => {
            estimated_structural_bytes(&array.as_list::<i64>().trimmed_values(), slots, true, true)
        }
        DataType::Map(_, _) => {
            let Some(map_array) = array.as_any().downcast_ref::<MapArray>() else {
                return u64::MAX;
            };
            let Some(first) = map_array
                .offsets()
                .first()
                .and_then(|value| usize::try_from(*value).ok())
            else {
                return u64::MAX;
            };
            let Some(last) = map_array
                .offsets()
                .last()
                .and_then(|value| usize::try_from(*value).ok())
            else {
                return u64::MAX;
            };
            let Some(entries_len) = last.checked_sub(first) else {
                return u64::MAX;
            };
            let entries = Arc::new(map_array.entries().slice(first, entries_len)) as ArrayRef;
            estimated_structural_bytes(&entries, slots, true, true)
        }
        DataType::FixedSizeList(field, dimension) if field.data_type().is_nested() => {
            let list_array = array.as_fixed_size_list();
            let Ok(dimension) = usize::try_from(*dimension) else {
                return u64::MAX;
            };
            let values_len = array.len().saturating_mul(dimension);
            let values = if list_array.values().len() == values_len {
                list_array.values().clone()
            } else {
                let values_offset = array.offset().saturating_mul(dimension);
                if values_offset.saturating_add(values_len) > list_array.values().len() {
                    return u64::MAX;
                }
                list_array.values().slice(values_offset, values_len)
            };
            estimated_structural_bytes(&values, slots, has_repetition, has_definition)
        }
        _ => {
            let has_definition = has_definition || matches!(array.data_type(), DataType::Null);
            let bytes_per_level = (u64::from(has_repetition) + u64::from(has_definition))
                * std::mem::size_of::<u16>() as u64;
            slots.saturating_mul(bytes_per_level)
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int32Array, NullArray, StructArray};
    use arrow_schema::{Field, Fields, Schema};

    use super::*;

    #[test]
    fn estimates_structural_null_levels() {
        let num_rows = 1000;
        let struct_fields = Fields::from(vec![Arc::new(Field::new("value", DataType::Null, true))]);
        let struct_array = Arc::new(StructArray::new(
            struct_fields.clone(),
            vec![Arc::new(NullArray::new(num_rows))],
            None,
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "item",
            DataType::Struct(struct_fields),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![struct_array]).unwrap();

        assert!(estimated_write_batch_size(&batch).total() >= num_rows * 2);
    }

    #[test]
    fn estimates_only_the_current_slice() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter(0..1000))]).unwrap();
        let slice = batch.slice(100, 10);

        assert!(
            estimated_write_batch_size(&slice).total() < estimated_write_batch_size(&batch).total()
        );
    }
}
