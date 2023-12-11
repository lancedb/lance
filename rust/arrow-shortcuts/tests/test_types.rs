use std::sync::Arc;

use arrow_array::FixedSizeBinaryArray;
use arrow_schema::Schema;
use arrow_shortcuts::macros::{arr_field, arr_schema, arr_type};
use arrow_shortcuts::{
    arrow_schema::{DataType, Field},
    util::ArrowTypeInfer,
};

#[test]
pub fn test_types() {
    // ----- Basic primitives
    assert_eq!(arr_type!(!), DataType::Null);
    assert_eq!(arr_type!(bool), DataType::Boolean);
    assert_eq!(arr_type!(i8), DataType::Int8);
    assert_eq!(arr_type!(i16), DataType::Int16);
    assert_eq!(arr_type!(i32), DataType::Int32);
    assert_eq!(arr_type!(i64), DataType::Int64);
    assert_eq!(arr_type!(u8), DataType::UInt8);
    assert_eq!(arr_type!(u16), DataType::UInt16);
    assert_eq!(arr_type!(u32), DataType::UInt32);
    assert_eq!(arr_type!(u64), DataType::UInt64);
    assert_eq!(arr_type!(f32), DataType::Float32);
    assert_eq!(arr_type!(f64), DataType::Float64);
    assert_eq!(arr_type!(&str), DataType::Utf8);
    assert_eq!(arr_type!(String), DataType::Utf8);

    // ----- Fixed size lists
    assert_eq!(
        arr_type!([i32; 5]),
        DataType::FixedSizeList(Arc::new(Field::new("item", arr_type!(i32), true)), 5)
    );
    // Can surround any type with parens to make implicit fields non-nullable
    assert_eq!(
        arr_type!(([i32; 5])),
        DataType::FixedSizeList(Arc::new(Field::new("item", arr_type!(i32), false)), 5)
    );
    const DIM: i32 = 128;
    // Make sure we can use a non-literal for fixed-size-list size
    assert_eq!(
        arr_type!([i32; DIM]),
        DataType::FixedSizeList(Arc::new(Field::new("item", arr_type!(i32), true)), DIM)
    );
    // Nested list
    assert_eq!(
        arr_type!([[i32; 2]; 4]),
        DataType::FixedSizeList(Arc::new(Field::new("item", arr_type!([i32; 2]), true)), 4)
    );

    // ----- Variable size lists

    assert_eq!(
        arr_type!([i32]),
        DataType::List(Arc::new(Field::new("item", arr_type!(i32), true)))
    );
    assert_eq!(
        arr_type!([[i32; 4]]),
        DataType::List(Arc::new(Field::new("item", arr_type!([i32; 4]), true)))
    );

    // ----- Structs

    assert_eq!(
        arr_type!({
            foo: i32,
            bar: f32,
        }),
        DataType::Struct([arr_field!(foo: i32), arr_field!(bar: f32),].into())
    );
    // Nested is ok
    assert_eq!(
        arr_type!({
            score: f32,
            location: {
                x: f32,
                y: f64
            }
        }),
        DataType::Struct(
            [
                arr_field!(score: f32),
                Arc::new(Field::new(
                    "location",
                    DataType::Struct([arr_field!(x: f32), arr_field!(y: f64)].into()),
                    true
                ))
            ]
            .into()
        )
    );

    // ----- Fields

    assert_eq!(
        arr_field!(x: i32),
        Arc::new(Field::new("x", arr_type!(i32), true))
    );

    // ----- Schemas

    assert_eq!(
        arr_schema!({
            x: i32,
            y: f32
        }),
        Schema::new([arr_field!(x: i32), arr_field!(y: f32),])
    );

    // Even nested schemas
    assert_eq!(
        arr_schema!({
            vector: [f32; 128],
            metadata: {
                caption: &str,
                user_score: f64
            }
        }),
        Schema::new([
            arr_field!(vector: [f32; 128]),
            arr_field!(metadata: {
                caption: &str,
                user_score: f64
            })
        ])
    );

    // ----- Limitations

    // Can't do lists of nested types
    // arr_type!([{x: i32}]); - ERR
}

struct Uuid;

impl ArrowTypeInfer for Uuid {
    type ArrayType = FixedSizeBinaryArray;

    fn arrow_type() -> DataType {
        DataType::FixedSizeBinary(16)
    }
}

#[test]
pub fn custom_types() {
    assert_eq!(arr_type!(Uuid), DataType::FixedSizeBinary(16));
}
