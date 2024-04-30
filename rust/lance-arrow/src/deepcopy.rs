// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{make_array, Array, RecordBatch};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::ArrayData;

pub fn deep_copy_buffer(buffer: &Buffer) -> Buffer {
    Buffer::from(Vec::from(buffer.as_slice()))
}

fn deep_copy_nulls(nulls: &NullBuffer) -> Buffer {
    deep_copy_buffer(nulls.inner().inner())
}

pub fn deep_copy_array_data(data: &ArrayData) -> ArrayData {
    let data_type = data.data_type().clone();
    let len = data.len();
    let null_count = data.null_count();
    let null_bit_buffer = data.nulls().map(deep_copy_nulls);
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
        ArrayData::new_unchecked(
            data_type,
            len,
            Some(null_count),
            null_bit_buffer,
            offset,
            buffers,
            child_data,
        )
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
    RecordBatch::try_new(batch.schema().clone(), arrays)
}
