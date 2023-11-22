use std::sync::Arc;

use arrow_schema::DataType;
use arrow_shortcuts::arrow_array::{RecordBatch, StringArray, UInt16Array, UInt8Array};
use arrow_shortcuts::arrow_schema::{Field, Schema};
use arrow_shortcuts::macros::arr_batch;

#[test]
pub fn test_arrays() {
    let batch = arr_batch!({
        x: [1, 2, ()] as u8,
        y: [4, (), 5] as u16,
        strings: [(), "x", "y"] as &str,
        // Not yet supported
        // vecs: [[1, 2, 3], [4, (), 6], ()] as &[u16],
    });
    let expected_schema = Schema::new(vec![
        Field::new("x", DataType::UInt8, true),
        Field::new("y", DataType::UInt16, true),
        Field::new("strings", DataType::Utf8, true),
    ]);
    let expected_batch = RecordBatch::try_new(
        Arc::new(expected_schema),
        vec![
            Arc::new(UInt8Array::from_iter(&[Some(1), Some(2), None])),
            Arc::new(UInt16Array::from_iter(&[Some(4), None, Some(5)])),
            Arc::new(StringArray::from_iter(&[None, Some("x"), Some("y")])),
        ],
    )
    .unwrap();
    assert_eq!(batch, expected_batch);
}
