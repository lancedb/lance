// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Write-time nullability validation that counts only nulls a reader can
//! observe. A null beneath a null ancestor is not a value of the field: the
//! ancestor masks the slot, its contents are unconstrained, and the repdef
//! encoding records it as the ancestor's null. Rejecting it would turn away
//! valid Arrow -- including batches produced by scanning this same dataset.

use arrow_array::{Array, ArrayRef, OffsetSizeTrait, cast::AsArray};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, NullBuffer, OffsetBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_core::datatypes::Field;
use lance_core::{Error, Result};

/// Verify that every visible null in `array` sits under a nullable field.
pub fn verify_visible_nullability(array: &dyn Array, field: &Field) -> Result<()> {
    verify(array, field, None)
}

fn nullability_err(field: &Field) -> Error {
    Error::invalid_input(format!(
        "The field `{}` contained null values even though the field is marked non-null in the schema",
        field.name
    ))
}

/// `hidden` marks slots masked by a null ancestor; `None` means all visible.
fn verify(array: &dyn Array, field: &Field, hidden: Option<&BooleanBuffer>) -> Result<()> {
    if !field.nullable && has_visible_null(array, hidden) {
        return Err(nullability_err(field));
    }
    match array.data_type() {
        DataType::Struct(_) => {
            let array = array.as_struct();
            let child_hidden = merge_hidden(hidden, array.nulls());
            for (child_field, child) in field.children.iter().zip(array.columns()) {
                verify(child.as_ref(), child_field, child_hidden.as_ref())?;
            }
            Ok(())
        }
        // No child `Field` means no declared nullability below this level:
        // the item type lives in the logical type string instead.
        DataType::List(_) => {
            let Some(child_field) = field.children.first() else {
                return Ok(());
            };
            let array = array.as_list::<i32>();
            let (values, child_hidden) = offsets_visible_window(
                hidden,
                array.nulls(),
                array.offsets(),
                array.values().as_ref(),
            );
            verify(values.as_ref(), child_field, child_hidden.as_ref())
        }
        DataType::LargeList(_) => {
            let Some(child_field) = field.children.first() else {
                return Ok(());
            };
            let array = array.as_list::<i64>();
            let (values, child_hidden) = offsets_visible_window(
                hidden,
                array.nulls(),
                array.offsets(),
                array.values().as_ref(),
            );
            verify(values.as_ref(), child_field, child_hidden.as_ref())
        }
        DataType::FixedSizeList(_, _) => {
            let Some(child_field) = field.children.first() else {
                return Ok(());
            };
            let array = array.as_fixed_size_list();
            let child_hidden = fixed_size_hidden(hidden, array);
            verify(array.values().as_ref(), child_field, child_hidden.as_ref())
        }
        DataType::Map(_, _) => {
            let Some(child_field) = field.children.first() else {
                return Ok(());
            };
            let array = array.as_map();
            let (entries, child_hidden) =
                offsets_visible_window(hidden, array.nulls(), array.offsets(), array.entries());
            verify(entries.as_ref(), child_field, child_hidden.as_ref())
        }
        _ => {
            // Types the visibility walk does not model keep the strict
            // raw-null check. The guard is what keeps `to_data()` off the
            // primitive fast path, where Lance declares no children.
            if field.children.is_empty() {
                return Ok(());
            }
            for (child_field, child) in field.children.iter().zip(array.to_data().child_data()) {
                verify_strict(child, child_field)?;
            }
            Ok(())
        }
    }
}

fn verify_strict(array: &ArrayData, field: &Field) -> Result<()> {
    if !field.nullable && array.null_count() > 0 {
        return Err(nullability_err(field));
    }
    for (child_field, child) in field.children.iter().zip(array.child_data()) {
        verify_strict(child, child_field)?;
    }
    Ok(())
}

fn has_visible_null(array: &dyn Array, hidden: Option<&BooleanBuffer>) -> bool {
    if array.null_count() == 0 {
        return false;
    }
    // Arrow derives the null count from the null buffer, so a non-zero count
    // here means there is one to walk.
    let (Some(nulls), Some(hidden)) = (array.nulls(), hidden) else {
        return true;
    };
    nulls
        .iter()
        .zip(hidden.iter())
        .any(|(valid, hidden)| !valid && !hidden)
}

/// Slot-aligned children (struct): a slot is hidden for the child if it was
/// already hidden or this level is null there.
fn merge_hidden(
    hidden: Option<&BooleanBuffer>,
    nulls: Option<&NullBuffer>,
) -> Option<BooleanBuffer> {
    match (hidden, nulls) {
        (None, None) => None,
        (Some(hidden), None) => Some(hidden.clone()),
        (None, Some(nulls)) => Some(!nulls.inner()),
        (Some(hidden), Some(nulls)) => Some(hidden | &!nulls.inner()),
    }
}

/// Offset-mapped children (list, map): slice the child to the offset domain
/// `[first, last)` -- a value outside it is unreachable through any slot, as
/// with the retained buffer behind a zero-copy slice -- and mark the values
/// inside it that only hidden or null slots reach. Offsets are
/// non-decreasing, so consecutive windows tile the domain exactly: with no
/// hidden and no null slots every value in it is visible, and no bitmap is
/// needed. Cost is proportional to the referenced window, never the full
/// retained child.
fn offsets_visible_window<O: OffsetSizeTrait>(
    hidden: Option<&BooleanBuffer>,
    nulls: Option<&NullBuffer>,
    offsets: &OffsetBuffer<O>,
    values: &dyn Array,
) -> (ArrayRef, Option<BooleanBuffer>) {
    let first = offsets.first().map_or(0, |o| o.as_usize());
    let last = offsets.last().map_or(0, |o| o.as_usize());
    let values = values.slice(first, last - first);
    if hidden.is_none() && nulls.is_none_or(|n| n.null_count() == 0) {
        return (values, None);
    }
    let mut builder = BooleanBufferBuilder::new(last - first);
    builder.append_n(last - first, true);
    for (slot, window) in offsets.windows(2).enumerate() {
        if slot_hidden(hidden, nulls, slot) {
            continue;
        }
        for value in window[0].as_usize()..window[1].as_usize() {
            builder.set_bit(value - first, false);
        }
    }
    (values, Some(builder.finish()))
}

fn fixed_size_hidden(
    hidden: Option<&BooleanBuffer>,
    array: &arrow_array::FixedSizeListArray,
) -> Option<BooleanBuffer> {
    if hidden.is_none() && array.null_count() == 0 {
        return None;
    }
    let width = array.value_length() as usize;
    let values_len = array.values().len();
    let mut builder = BooleanBufferBuilder::new(values_len);
    builder.append_n(values_len, true);
    for slot in 0..array.len() {
        if slot_hidden(hidden, array.nulls(), slot) {
            continue;
        }
        // Not `value_offset`: that computes in i32 and can wrap on a batch
        // with more than 2^31 child slots.
        let start = slot * width;
        for value in start..start + width {
            builder.set_bit(value, false);
        }
    }
    Some(builder.finish())
}

fn slot_hidden(hidden: Option<&BooleanBuffer>, nulls: Option<&NullBuffer>, slot: usize) -> bool {
    hidden.is_some_and(|h| h.value(slot)) || nulls.is_some_and(|n| n.is_null(slot))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, ListArray, StructArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field as ArrowField, Fields};

    use super::*;

    /// The declared field is stricter than the batch's own arrow schema, as
    /// with a writer schema taken from a manifest: arrow would reject the
    /// visible-null constructions below at build time under a non-null field.
    fn strict_struct_field() -> Field {
        let child = ArrowField::new("x", DataType::Int32, false);
        Field::try_from(&ArrowField::new(
            "s",
            DataType::Struct(vec![child].into()),
            true,
        ))
        .unwrap()
    }

    /// Slicing moves the null buffer's bit offset; the walk must stay
    /// aligned with it at every level.
    #[test]
    fn test_sliced_struct_alignment() {
        let relaxed = Fields::from(vec![ArrowField::new("x", DataType::Int32, true)]);
        // rows: [{x: 1}, null (masked x null), {x: null} (visible)]
        let array = StructArray::new(
            relaxed,
            vec![Arc::new(Int32Array::from(vec![Some(1), None, None])) as ArrayRef],
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let field = strict_struct_field();

        assert!(verify_visible_nullability(&array, &field).is_err());
        assert!(verify_visible_nullability(&array.slice(0, 2), &field).is_ok());
        assert!(verify_visible_nullability(&array.slice(2, 1), &field).is_err());
    }

    /// Every slot masked: no null below is a value of the child, so a strict
    /// child field is satisfied. `hidden` is `Some` on any path that masks,
    /// so the `None` arm of `has_visible_null` never sees a hidden slot.
    #[test]
    fn test_fully_masked_parent_has_no_visible_null() {
        let relaxed = Fields::from(vec![ArrowField::new("x", DataType::Int32, true)]);
        let array = StructArray::new(
            relaxed,
            vec![Arc::new(Int32Array::from(vec![None, None, None])) as ArrayRef],
            Some(NullBuffer::from(vec![false, false, false])),
        );

        assert!(verify_visible_nullability(&array, &strict_struct_field()).is_ok());
    }

    /// A sliced list keeps its full values array; only nulls inside a
    /// referenced, visible window may reject.
    #[test]
    fn test_sliced_list_window_visibility() {
        let relaxed = Arc::new(ArrowField::new("item", DataType::Int32, true));
        // windows: [0, 2) valid, [2, 4) null (masked), [4, 6) valid+null (visible)
        let array = ListArray::new(
            relaxed,
            OffsetBuffer::new(vec![0, 2, 4, 6].into()),
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                None,
                None,
                Some(5),
                None,
            ])) as ArrayRef,
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let strict_item = ArrowField::new("item", DataType::Int32, false);
        let field = Field::try_from(&ArrowField::new(
            "v",
            DataType::List(Arc::new(strict_item)),
            true,
        ))
        .unwrap();

        assert!(verify_visible_nullability(&array, &field).is_err());
        // Dropping the last slot leaves only masked nulls in referenced windows.
        assert!(verify_visible_nullability(&array.slice(0, 2), &field).is_ok());
        assert!(verify_visible_nullability(&array.slice(2, 1), &field).is_err());
        // A non-zero-based slice mixing masked and visible windows: the
        // visibility bitmap covers the offset domain, so every window index
        // is relative to the slice's first offset.
        assert!(verify_visible_nullability(&array.slice(1, 2), &field).is_err());
        assert!(verify_visible_nullability(&array.slice(1, 1), &field).is_ok());
    }

    /// A slice with no validity buffer still hides the child values its
    /// offset window no longer reaches.
    #[test]
    fn test_sliced_list_unreachable_values_hidden() {
        let relaxed = Arc::new(ArrowField::new("item", DataType::Int32, true));
        let array = ListArray::new(
            relaxed,
            OffsetBuffer::new(vec![0, 2, 4].into()),
            Arc::new(Int32Array::from(vec![None, Some(2), Some(3), Some(4)])) as ArrayRef,
            None,
        );
        let strict_item = ArrowField::new("item", DataType::Int32, false);
        let field = Field::try_from(&ArrowField::new(
            "v",
            DataType::List(Arc::new(strict_item)),
            true,
        ))
        .unwrap();

        // The null at value 0 is visible in the full array...
        assert!(verify_visible_nullability(&array, &field).is_err());
        // ...and unreachable once the first slot is sliced away.
        assert!(verify_visible_nullability(&array.slice(1, 1), &field).is_ok());
    }

    /// The offset domain must span the whole child for the fast path, so a
    /// never-sliced list built over a longer values buffer hides its
    /// unreferenced tail the same way -- as does a zero-length list.
    #[test]
    fn test_short_cover_list_tail_hidden() {
        let relaxed = Arc::new(ArrowField::new("item", DataType::Int32, true));
        let strict_item = ArrowField::new("item", DataType::Int32, false);
        let field = Field::try_from(&ArrowField::new(
            "v",
            DataType::List(Arc::new(strict_item)),
            true,
        ))
        .unwrap();

        let short_cover = ListArray::new(
            relaxed.clone(),
            OffsetBuffer::new(vec![0, 2].into()),
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None])) as ArrayRef,
            None,
        );
        assert!(verify_visible_nullability(&short_cover, &field).is_ok());

        let zero_slots = ListArray::new(
            relaxed,
            OffsetBuffer::new(vec![0].into()),
            Arc::new(Int32Array::from(vec![None])) as ArrayRef,
            None,
        );
        assert!(verify_visible_nullability(&zero_slots, &field).is_ok());
    }
}
