// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{future::Future, sync::Arc};

use arrow_array::{cast::AsArray, Array, ArrayRef, ListArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};

use lance_core::Result;

/// A transform function to be used as part of
/// [`transform_array_post`] or [`transform_batch_post`].
pub trait ArrayTransformer {
    /// Potentially transform an array
    ///
    /// Return None to indicate the array was not transformed or return
    /// a new field and array to indicate the array was transformed
    fn transform(
        array: &dyn Array,
        field: &Field,
    ) -> impl Future<Output = Result<Option<(FieldRef, ArrayRef)>>>;
}

/// Applies a transformation to an array.  If the array is a nested array
/// (struct or list) then it will recursively apply the transformation to
/// the children
///
/// The transformation is not applied to transformed arrays, even if they
/// are nested.
///
/// First we attempt to apply the transformation to the array and then, if
/// that fails, we try and apply the transformation to the array's chilren
/// and then, if no child was modified, we return None
#[async_recursion::async_recursion(?Send)]
pub async fn transform_array_post<Transformer: ArrayTransformer>(
    arr: &dyn Array,
    field: &Field,
) -> Result<Option<(FieldRef, ArrayRef)>> {
    match arr.data_type() {
        DataType::Struct(fields) => {
            let struct_arr = arr.as_struct();
            let mut new_columns = Vec::new();
            let mut new_fields = Vec::new();
            let mut has_new = false;
            for (col, child_field) in struct_arr.columns().iter().zip(fields) {
                if let Some((transformed_field, transformed_arr)) =
                    Transformer::transform(col, child_field.as_ref()).await?
                {
                    has_new = true;
                    new_fields.push(transformed_field);
                    new_columns.push(transformed_arr);
                } else if let Some((transformed_field, transitive_arr)) =
                    transform_array_post::<Transformer>(&col, child_field).await?
                {
                    has_new = true;
                    new_fields.push(transformed_field);
                    new_columns.push(transitive_arr);
                } else {
                    new_fields.push(child_field.clone());
                    new_columns.push(col.clone());
                }
            }
            if has_new {
                println!("Fields={:?} Columns={:?}", new_fields, new_columns);
                for arr in &new_columns {
                    println!("Nulls: {:?}", arr.nulls());
                }
                let fields = Fields::from(new_fields);
                let new_arr = Arc::new(StructArray::new(
                    fields.clone(),
                    new_columns,
                    arr.nulls().cloned(),
                ));
                let new_field =
                    Field::new(field.name(), DataType::Struct(fields), field.is_nullable());
                Ok(Some((Arc::new(new_field), new_arr as ArrayRef)))
            } else {
                Ok(None)
            }
        }
        DataType::List(items_field) => {
            let list_arr = arr.as_list();
            let rewrap = |transformed_items_field: FieldRef, transformed_items_arr: ArrayRef| {
                let new_arr = ListArray::new(
                    transformed_items_field.clone(),
                    list_arr.offsets().clone(),
                    transformed_items_arr,
                    list_arr.nulls().cloned(),
                );
                let new_field = Field::new(
                    field.name(),
                    DataType::List(transformed_items_field),
                    field.is_nullable(),
                );
                Ok(Some((Arc::new(new_field), Arc::new(new_arr) as ArrayRef)))
            };
            if let Some((transformed_items_field, transformed_items_arr)) =
                Transformer::transform(list_arr.values().as_ref(), items_field.as_ref()).await?
            {
                rewrap(transformed_items_field, transformed_items_arr)
            } else if let Some((transformed_items_field, transformed_items_arr)) =
                transform_array_post::<Transformer>(
                    list_arr.values().as_ref(),
                    items_field.as_ref(),
                )
                .await?
            {
                rewrap(transformed_items_field, transformed_items_arr)
            } else {
                Ok(None)
            }
        }
        _ => Transformer::transform(arr, field).await,
    }
}

/// Applies a transformation to a batch of data
///
/// This is similar to [`transform_array_post`]
pub async fn transform_batch_post<Transformer: ArrayTransformer>(
    batch: &RecordBatch,
) -> Result<Option<RecordBatch>> {
    let mut new_columns = Vec::new();
    let mut new_fields = Vec::new();
    let mut has_new = false;
    for (col, field) in batch.columns().iter().zip(batch.schema().fields()) {
        if let Some((transformed_field, transformed_arr)) =
            Transformer::transform(col, field).await?
        {
            has_new = true;
            new_fields.push(transformed_field);
            new_columns.push(transformed_arr);
        } else if let Some((transformed_field, transitive_arr)) =
            transform_array_post::<Transformer>(&col, field).await?
        {
            has_new = true;
            new_fields.push(transformed_field);
            new_columns.push(transitive_arr);
        } else {
            new_fields.push(field.clone());
            new_columns.push(col.clone());
        }
    }
    if has_new {
        let new_schema = Schema::new(new_fields);
        Ok(Some(RecordBatch::try_new(
            Arc::new(new_schema),
            new_columns,
        )?))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {

    use std::{future::Future, sync::Arc};

    use arrow_array::{
        Array, ArrayRef, Float32Array, Float64Array, Int32Array, ListArray, RecordBatch,
        StructArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_cast::cast;
    use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};

    use crate::util::transform_batch_post;

    use super::{ArrayTransformer, Result};

    #[tokio::test]
    async fn test_transform() {
        struct Fp32ToFp64;

        impl ArrayTransformer for Fp32ToFp64 {
            fn transform(
                array: &dyn Array,
                field: &Field,
            ) -> impl Future<Output = Result<Option<(FieldRef, ArrayRef)>>> {
                async move {
                    match field.data_type() {
                        DataType::Float32 => {
                            let new_field =
                                Field::new(field.name(), DataType::Float64, field.is_nullable());
                            println!("Nulls: {:?}", array.nulls());
                            println!("input: {:?}", array);
                            let fp64_arr = cast(array, &DataType::Float64)?;
                            println!("FP64 Nulls: {:?}", fp64_arr.nulls());
                            println!("output: {:?}", fp64_arr);
                            Ok(Some((Arc::new(new_field), fp64_arr)))
                        }
                        _ => Ok(None),
                    }
                }
            }
        }

        // Creating:
        //
        // {
        //   "top_level_float": [1.0],
        //   "list": [
        //      [{"nested": [2.0], "unmodified": [3]}]
        //    ]
        // }
        //

        let tlf = Float32Array::from(vec![1.0]);
        let nested = Float32Array::from(vec![2.0]);
        let unmodified = Int32Array::from(vec![3]);
        let struc = StructArray::new(
            Fields::from(vec![
                Field::new("nested", DataType::Float32, true),
                Field::new("unmodified", DataType::Int32, true),
            ]),
            vec![Arc::new(nested), Arc::new(unmodified.clone())],
            None,
        );
        let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![0, 1]));
        let list = ListArray::new(
            Arc::new(Field::new("item", struc.data_type().clone(), true)),
            offsets,
            Arc::new(struc) as ArrayRef,
            None,
        );
        let schema = Schema::new(vec![
            Field::new("top_level_float", DataType::Float32, true),
            Field::new("list", list.data_type().clone(), true),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(tlf), Arc::new(list)]).unwrap();

        // Same thing but with f64 instead of f32
        let expected_tlf = Float64Array::from(vec![1.0]);
        let expected_nested = Float64Array::from(vec![2.0]);
        let expected_struc = StructArray::new(
            Fields::from(vec![
                Field::new("nested", DataType::Float64, true),
                Field::new("unmodified", DataType::Int32, true),
            ]),
            vec![Arc::new(expected_nested), Arc::new(unmodified)],
            None,
        );
        let expected_offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![0, 1]));
        let expected_list = ListArray::new(
            Arc::new(Field::new("item", expected_struc.data_type().clone(), true)),
            expected_offsets,
            Arc::new(expected_struc) as ArrayRef,
            None,
        );
        let expected_schema = Schema::new(vec![
            Field::new("top_level_float", DataType::Float64, true),
            Field::new("list", expected_list.data_type().clone(), true),
        ]);
        let expected = RecordBatch::try_new(
            Arc::new(expected_schema),
            vec![Arc::new(expected_tlf), Arc::new(expected_list)],
        )
        .unwrap();

        let actual = transform_batch_post::<Fp32ToFp64>(&batch)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(expected, actual);
    }
}
