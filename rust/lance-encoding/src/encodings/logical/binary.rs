// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{ByteArrayType, UInt8Type},
    Array, ArrayRef, BinaryArray, GenericByteArray, GenericListArray, ListArray, StringArray,
    UInt8Array,
};

use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use log::trace;

use crate::{
    decoder::{DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask},
    encoder::{EncodedPage, FieldEncoder},
};

use super::{list::ListFieldEncoder, primitive::PrimitiveFieldEncoder};

// TODO: Support large string, binary, large binary

/// A logical scheduler for utf8/binary pages which assumes the data are encoded as List<u8>
#[derive(Debug)]
pub struct BinaryPageScheduler {
    varbin_scheduler: Box<dyn LogicalPageScheduler>,
    data_type: DataType,
}

impl BinaryPageScheduler {
    // Create a new ListPageScheduler
    pub fn new(varbin_scheduler: Box<dyn LogicalPageScheduler>, data_type: DataType) -> Self {
        Self {
            varbin_scheduler,
            data_type,
        }
    }
}

impl LogicalPageScheduler for BinaryPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &Arc<dyn crate::EncodingsIo>,
        sink: &tokio::sync::mpsc::UnboundedSender<Box<dyn crate::decoder::LogicalPageDecoder>>,
    ) -> Result<()> {
        trace!("Scheduling binary for {} ranges", ranges.len());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        self.varbin_scheduler
            .schedule_ranges(ranges, scheduler, &tx)?;

        while let Some(decoder) = rx.recv().now_or_never() {
            let wrapped = BinaryPageDecoder {
                inner: decoder.unwrap(),
                data_type: self.data_type.clone(),
            };
            sink.send(Box::new(wrapped)).unwrap();
        }

        Ok(())
    }

    fn schedule_take(
        &self,
        indices: &[u32],
        scheduler: &Arc<dyn crate::EncodingsIo>,
        sink: &tokio::sync::mpsc::UnboundedSender<Box<dyn crate::decoder::LogicalPageDecoder>>,
    ) -> Result<()> {
        trace!("Scheduling binary for {} indices", indices.len());
        self.schedule_ranges(
            &indices
                .iter()
                .map(|&idx| idx..(idx + 1))
                .collect::<Vec<_>>(),
            scheduler,
            sink,
        )
    }

    fn num_rows(&self) -> u32 {
        self.varbin_scheduler.num_rows()
    }
}

pub struct BinaryPageDecoder {
    inner: Box<dyn LogicalPageDecoder>,
    data_type: DataType,
}

impl LogicalPageDecoder for BinaryPageDecoder {
    fn wait<'a>(
        &'a mut self,
        num_rows: u32,
        source: &'a mut tokio::sync::mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'a, Result<()>> {
        self.inner.wait(num_rows, source)
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        let inner_task = self.inner.drain(num_rows)?;
        Ok(NextDecodeTask {
            has_more: inner_task.has_more,
            num_rows: inner_task.num_rows,
            task: Box::new(BinaryArrayDecoder {
                inner: inner_task.task,
                data_type: self.data_type.clone(),
            }),
        })
    }

    fn unawaited(&self) -> u32 {
        self.inner.unawaited()
    }

    fn avail(&self) -> u32 {
        self.inner.avail()
    }
}

pub struct BinaryArrayDecoder {
    inner: Box<dyn DecodeArrayTask>,
    data_type: DataType,
}

impl BinaryArrayDecoder {
    fn from_list_array(data_type: &DataType, array: &GenericListArray<i32>) -> ArrayRef {
        let values = array
            .values()
            .as_primitive::<UInt8Type>()
            .values()
            .inner()
            .clone();
        match data_type {
            DataType::Utf8 => Arc::new(StringArray::new(
                array.offsets().clone(),
                values,
                array.nulls().cloned(),
            )),
            DataType::Binary => Arc::new(BinaryArray::new(
                array.offsets().clone(),
                values,
                array.nulls().cloned(),
            )),
            _ => panic!("Binary decoder does not support data type {}", data_type),
        }
    }
}

impl DecodeArrayTask for BinaryArrayDecoder {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let data_type = self.data_type;
        let arr = self.inner.decode()?;
        let list_arr = arr.as_list::<i32>();
        Ok(Self::from_list_array(&data_type, list_arr))
    }
}

/// An encoder which encodes string arrays as List<u8>
pub struct BinaryFieldEncoder {
    varbin_encoder: Box<dyn FieldEncoder>,
}

impl BinaryFieldEncoder {
    pub fn new(cache_bytes_per_column: u64, column_index: u32) -> Self {
        let items_encoder = Box::new(
            PrimitiveFieldEncoder::try_new(
                cache_bytes_per_column,
                &DataType::UInt8,
                column_index + 1,
            )
            .unwrap(),
        );
        Self {
            varbin_encoder: Box::new(ListFieldEncoder::new(
                items_encoder,
                cache_bytes_per_column,
                column_index,
            )),
        }
    }

    fn byte_to_list_array<T: ByteArrayType<Offset = i32>>(
        array: &GenericByteArray<T>,
    ) -> ListArray {
        let values = UInt8Array::new(
            ScalarBuffer::<u8>::new(array.values().clone(), 0, array.values().len()),
            None,
        );
        let list_field = Field::new("item", DataType::UInt8, true);
        ListArray::new(
            Arc::new(list_field),
            array.offsets().clone(),
            Arc::new(values),
            array.nulls().cloned(),
        )
    }

    fn to_list_array(array: ArrayRef) -> ListArray {
        match array.data_type() {
            DataType::Utf8 => Self::byte_to_list_array(array.as_string::<i32>()),
            DataType::Binary => Self::byte_to_list_array(array.as_binary::<i32>()),
            _ => panic!("Binary encoder does not support {}", array.data_type()),
        }
    }
}

impl FieldEncoder for BinaryFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        let list_array = Self::to_list_array(array);
        self.varbin_encoder.maybe_encode(Arc::new(list_array))
    }

    fn flush(&mut self) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        self.varbin_encoder.flush()
    }

    fn num_columns(&self) -> u32 {
        2
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{DataType, Field};

    use crate::testing::check_round_trip_encoding;

    #[test_log::test(tokio::test)]
    async fn test_utf8() {
        let field = Field::new("", DataType::Utf8, false);
        check_round_trip_encoding(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_binary() {
        let field = Field::new("", DataType::Binary, false);
        check_round_trip_encoding(field).await;
    }
}
