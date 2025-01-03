// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{make_array, Array, RecordBatch};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::{ArrayData, ArrayDataBuilder};

pub fn deep_copy_buffer(buffer: &Buffer) -> Buffer {
    Buffer::from(buffer.as_slice())
}

fn deep_copy_nulls(nulls: Option<&NullBuffer>) -> Option<NullBuffer> {
    let nulls = nulls?;
    let bit_buffer = deep_copy_buffer(nulls.inner().inner());
    Some(unsafe {
        NullBuffer::new_unchecked(
            BooleanBuffer::new(bit_buffer, nulls.offset(), nulls.len()),
            nulls.null_count(),
        )
    })
}

pub fn deep_copy_array_data(data: &ArrayData) -> ArrayData {
    let data_type = data.data_type().clone();
    let len = data.len();
    let nulls = deep_copy_nulls(data.nulls());
    let offset = data.offset();
    let buffers = data
        .buffers()
        .iter()
        .map(deep_copy_buffer)
        .collect::<Vec<_>>();
    let child_data = data
        .child_data()
        .iter()
        .map(deep_copy_array_data)
        .collect::<Vec<_>>();
    unsafe {
        ArrayDataBuilder::new(data_type)
            .len(len)
            .nulls(nulls)
            .offset(offset)
            .buffers(buffers)
            .child_data(child_data)
            .build_unchecked()
    }
}

pub fn deep_copy_array(array: &dyn Array) -> Arc<dyn Array> {
    let data = array.to_data();
    let data = deep_copy_array_data(&data);
    make_array(data)
}

pub fn deep_copy_batch(batch: &RecordBatch) -> crate::Result<RecordBatch> {
    let arrays = batch
        .columns()
        .iter()
        .map(|array| deep_copy_array(array))
        .collect::<Vec<_>>();
    RecordBatch::try_new(batch.schema(), arrays)
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int32Array};

    #[test]
    fn test_deep_copy_sliced_array_with_nulls() {
        let array = Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
        ]));
        let sliced_array = array.slice(1, 3);
        let copied_array = super::deep_copy_array(&sliced_array);
        assert_eq!(sliced_array.len(), copied_array.len());
        assert_eq!(sliced_array.nulls(), copied_array.nulls());
    }
}
