// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extension to arrow struct arrays

use arrow_array::{Array, StructArray, cast::AsArray, make_array};
use arrow_buffer::NullBuffer;
use arrow_schema::ArrowError;

pub trait StructArrayExt {
    /// Ensures the struct array's slicing is normalized: any offset is pushed into
    /// the children rather than left on the struct itself.
    ///
    /// In arrow-rs a `StructArray` cannot carry a slice offset. Both
    /// [`arrow_array::Array::slice`] and `StructArray::from(ArrayData)` push the
    /// offset into the children (slicing them) at construction time, so any
    /// `StructArray` reaching this method is already normalized and this is a
    /// no-op. The arrow-cpp shape (offset on the parent, children left as-is)
    /// cannot be represented by an arrow-rs `StructArray`.
    ///
    /// The method is kept for API stability and to assert the invariant.
    fn normalize_slicing(&self) -> Result<Self, ArrowError>
    where
        Self: Sized;

    /// Structs are allowed to mask valid items.  For example, a struct array might be:
    ///
    /// [ {"items": [1, 2, 3]}, NULL, {"items": NULL}]
    ///
    /// However, the underlying items array might be: [[1, 2, 3], [4, 5], NULL]
    ///
    /// The [4, 5] list is masked out because the struct array is null.
    ///
    /// The struct validity would be [true, false, true] and the list validity would be [true, true, false]
    ///
    /// This method pushes nulls down into all children.  In the above example the list validity would become
    /// [true, false, false].
    ///
    /// This method is not recursive.  If a child is a struct array it will not push that child's nulls down.
    ///
    /// This method does not remove garbage lists.  It only updates the validity so a future call to
    /// [crate::list::ListArrayExt::filter_garbage_nulls] will remove the garbage lists (without
    /// this pushdown it would not)
    fn pushdown_nulls(&self) -> Result<Self, ArrowError>
    where
        Self: Sized;
}

impl StructArrayExt for StructArray {
    fn normalize_slicing(&self) -> Result<Self, ArrowError>
    where
        Self: Sized,
    {
        // An arrow-rs `StructArray` is always already normalized: the offset is
        // pushed into the children at construction (see the trait docs), so there
        // is nothing to do. The assert documents that invariant and trips in tests
        // if a future arrow-rs change ever violates it.
        debug_assert!(
            self.offset() == 0 && self.columns().iter().all(|c| c.len() == self.len()),
            "StructArray reached normalize_slicing without being normalized"
        );
        Ok(self.clone())
    }

    fn pushdown_nulls(&self) -> Result<Self, ArrowError>
    where
        Self: Sized,
    {
        let Some(validity) = self.nulls() else {
            return Ok(self.clone());
        };
        let data = self.to_data();
        let children = data
            .child_data()
            .iter()
            .map(|c| {
                if let Some(child_validity) = c.nulls() {
                    let new_validity = child_validity.inner() & validity.inner();
                    c.clone()
                        .into_builder()
                        .nulls(Some(NullBuffer::from(new_validity)))
                        .build()
                } else {
                    Ok(c.clone()
                        .into_builder()
                        .nulls(Some(validity.clone()))
                        .build()?)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let arr = make_array(data.into_builder().child_data(children).build()?);
        Ok(arr.as_struct().clone())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Int32Array, StructArray, cast::AsArray, make_array};
    use arrow_schema::{DataType, Field, Fields};
    use std::sync::Arc;

    use crate::r#struct::StructArrayExt;

    #[test]
    fn test_normalize_slicing_no_offset() {
        let x = Int32Array::from(vec![1, 2, 3]);
        let y = Int32Array::from(vec![4, 5, 6]);
        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ]),
            vec![Arc::new(x), Arc::new(y)],
            None,
        );

        let normalized = struct_array.normalize_slicing().unwrap();
        assert_eq!(normalized, struct_array);
    }

    #[test]
    fn test_arrow_rs_slicing() {
        let x = Int32Array::from(vec![1, 2, 3, 4]);
        let y = Int32Array::from(vec![5, 6, 7, 8]);
        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ]),
            vec![Arc::new(x), Arc::new(y)],
            None,
        );

        // Slicing with arrow-rs propagates the slicing to the children so there should
        // be no change needed to the struct array
        let sliced = struct_array.slice(1, 2);
        let normalized = sliced.normalize_slicing().unwrap();

        assert_eq!(normalized, sliced);
    }

    #[test]
    fn test_arrow_cpp_slicing() {
        let x = Int32Array::from(vec![1, 2, 3, 4]);
        let y = Int32Array::from(vec![5, 6, 7, 8]);
        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ]),
            vec![Arc::new(x), Arc::new(y)],
            None,
        );

        let data = struct_array.to_data();
        let sliced = data.into_builder().offset(1).len(2).build().unwrap();
        let sliced = make_array(sliced);
        let normalized = sliced.as_struct().clone().normalize_slicing().unwrap();

        assert_eq!(normalized, struct_array.slice(1, 2));
    }
}
