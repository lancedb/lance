// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow_array::{
    cast::AsArray, types::UInt8Type, Array, ArrayRef, ListArray, StringArray, UInt8Array,
};

use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use log::trace;

use crate::{
    decoder::{DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask},
    encoder::{EncodedPage, FieldEncoder},
    encodings::physical::basic::BasicEncoder,
};

use super::{list::ListFieldEncoder, primitive::PrimitiveFieldEncoder};

// TODO: Support large string, binary, large binary

/// A logical scheduler for utf8 pages which assumes the data are encoded as List<u8>
#[derive(Debug)]
pub struct Utf8PageScheduler {
    varbin_scheduler: Box<dyn LogicalPageScheduler>,
}

impl Utf8PageScheduler {
    // Create a new ListPageScheduler
    pub fn new(varbin_scheduler: Box<dyn LogicalPageScheduler>) -> Self {
        Self { varbin_scheduler }
    }
}

impl LogicalPageScheduler for Utf8PageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &Arc<dyn crate::EncodingsIo>,
        sink: &tokio::sync::mpsc::UnboundedSender<Box<dyn crate::decoder::LogicalPageDecoder>>,
    ) -> Result<()> {
        trace!("Scheduling utf8 for {} ranges", ranges.len());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        self.varbin_scheduler
            .schedule_ranges(ranges, scheduler, &tx)?;

        while let Some(decoder) = rx.recv().now_or_never() {
            let wrapped = Utf8PageDecoder {
                inner: decoder.unwrap(),
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
        trace!("Scheduling utf8 for {} indices", indices.len());
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

pub struct Utf8PageDecoder {
    inner: Box<dyn LogicalPageDecoder>,
}

impl LogicalPageDecoder for Utf8PageDecoder {
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
            task: Box::new(Utf8ArrayDecoder {
                inner: inner_task.task,
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

pub struct Utf8ArrayDecoder {
    inner: Box<dyn DecodeArrayTask>,
}

impl DecodeArrayTask for Utf8ArrayDecoder {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let arr = self.inner.decode()?;
        let list_arr = arr.as_list::<i32>();
        let values = list_arr
            .values()
            .as_primitive::<UInt8Type>()
            .values()
            .inner()
            .clone();
        Ok(Arc::new(StringArray::new(
            list_arr.offsets().clone(),
            values,
            list_arr.nulls().cloned(),
        )))
    }
}

/// An encoder which encodes string arrays as List<u8>
pub struct Utf8FieldEncoder {
    varbin_encoder: Box<dyn FieldEncoder>,
}

impl Utf8FieldEncoder {
    pub fn new(cache_bytes_per_column: u64, column_index: u32) -> Self {
        let bytes_encoder = Arc::new(BasicEncoder::new(column_index + 1));
        let items_encoder = Box::new(PrimitiveFieldEncoder::new(
            cache_bytes_per_column,
            bytes_encoder,
        ));
        Self {
            varbin_encoder: Box::new(ListFieldEncoder::new(
                items_encoder,
                cache_bytes_per_column,
                column_index,
            )),
        }
    }
}

impl FieldEncoder for Utf8FieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        let utf8_array = array.as_string::<i32>();
        let values = UInt8Array::new(
            ScalarBuffer::<u8>::new(utf8_array.values().clone(), 0, utf8_array.values().len()),
            None,
        );
        let list_field = Field::new("item", DataType::UInt8, true);
        let list_array = ListArray::new(
            Arc::new(list_field),
            utf8_array.offsets().clone(),
            Arc::new(values),
            utf8_array.nulls().cloned(),
        );
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

    use crate::{
        encodings::logical::utf8::Utf8FieldEncoder, testing::check_round_trip_field_encoding,
    };

    #[test_log::test(tokio::test)]
    async fn test_utf8() {
        let encoder = Utf8FieldEncoder::new(4096, 0);
        let field = Field::new("", DataType::Utf8, false);

        check_round_trip_field_encoding(encoder, field).await;
    }
}
