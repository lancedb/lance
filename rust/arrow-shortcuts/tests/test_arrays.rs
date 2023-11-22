use arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    NullArray, StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_shortcuts::macros::arr_array;

#[test]
pub fn test_arrays() {
    // ----- Basic primitives
    assert_eq!(
        arr_array!([1, (), 5] as u8),
        UInt8Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1, (), 5] as u16),
        UInt16Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1, (), 5] as u32),
        UInt32Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1, (), 5] as u64),
        UInt64Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1, (), 5] as i8),
        Int8Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1, (), 5] as i16),
        Int16Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1, (), 5] as i32),
        Int32Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1, (), 5] as i64),
        Int64Array::from(vec![Some(1), None, Some(5)])
    );
    assert_eq!(
        arr_array!([1.0, (), 5.0] as f32),
        Float32Array::from(vec![Some(1.0), None, Some(5.0)])
    );
    assert_eq!(
        arr_array!([1.0, (), 5.0] as f64),
        Float64Array::from(vec![Some(1.0), None, Some(5.0)])
    );
    assert_eq!(
        arr_array!([false, false, true] as bool),
        BooleanArray::from(vec![Some(false), Some(false), Some(true)])
    );
    assert_eq!(
        arr_array!(["x", "y", "z"] as &str),
        StringArray::from(vec![Some("x"), Some("y"), Some("z")])
    );
    // All null
    assert_eq!(
        arr_array!([(), (), ()] as f64),
        Float64Array::from(vec![None, None, None])
    );
    // Null typed array
    assert_eq!(arr_array!([(), ()] as !), NullArray::new(2),)

    // TODO
    // nested arrays
    // list arrays
}
