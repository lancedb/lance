// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, GenericByteArray, GenericListArray, OffsetSizeTrait,
    cast::AsArray,
    types::{BinaryType, ByteArrayType, LargeBinaryType, LargeUtf8Type, UInt8Type, Utf8Type},
};

use arrow_schema::DataType;
use futures::{FutureExt, future::BoxFuture};
use lance_core::Result;
use log::trace;

use crate::{
    decoder::{
        DecodeArrayTask, FilterExpression, MessageType, NextDecodeTask, PriorityRange,
        ScheduledScanLine, SchedulerContext,
    },
    previous::decoder::{DecoderReady, FieldScheduler, LogicalPageDecoder, SchedulingJob},
};

/// Wraps a varbin scheduler and uses a BinaryPageDecoder to cast
/// the result to the appropriate type
#[derive(Debug)]
pub struct BinarySchedulingJob<'a> {
    scheduler: &'a BinaryFieldScheduler,
    inner: Box<dyn SchedulingJob + 'a>,
}

impl SchedulingJob for BinarySchedulingJob<'_> {
    fn schedule_next(
        &mut self,
        context: &mut SchedulerContext,
        priority: &dyn PriorityRange,
    ) -> Result<ScheduledScanLine> {
        let inner_scan = self.inner.schedule_next(context, priority)?;
        let wrapped_decoders = inner_scan
            .decoders
            .into_iter()
            .map(|message| {
                let decoder = message.into_legacy();
                MessageType::DecoderReady(DecoderReady {
                    path: decoder.path,
                    decoder: Box::new(BinaryPageDecoder {
                        inner: decoder.decoder,
                        data_type: self.scheduler.data_type.clone(),
                    }),
                })
            })
            .collect::<Vec<_>>();
        Ok(ScheduledScanLine {
            decoders: wrapped_decoders,
            rows_scheduled: inner_scan.rows_scheduled,
        })
    }

    fn num_rows(&self) -> u64 {
        self.inner.num_rows()
    }
}

/// A logical scheduler for utf8/binary pages which assumes the data are encoded as `List<u8>`
#[derive(Debug)]
pub struct BinaryFieldScheduler {
    varbin_scheduler: Arc<dyn FieldScheduler>,
    data_type: DataType,
}

impl BinaryFieldScheduler {
    // Create a new ListPageScheduler
    pub fn new(varbin_scheduler: Arc<dyn FieldScheduler>, data_type: DataType) -> Self {
        Self {
            varbin_scheduler,
            data_type,
        }
    }
}

impl FieldScheduler for BinaryFieldScheduler {
    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[std::ops::Range<u64>],
        filter: &FilterExpression,
    ) -> Result<Box<dyn SchedulingJob + 'a>> {
        trace!("Scheduling binary for {} ranges", ranges.len());
        let varbin_job = self.varbin_scheduler.schedule_ranges(ranges, filter)?;
        Ok(Box::new(BinarySchedulingJob {
            scheduler: self,
            inner: varbin_job,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.varbin_scheduler.num_rows()
    }

    fn initialize<'a>(
        &'a self,
        _filter: &'a FilterExpression,
        _context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        // 2.0 schedulers do not need to initialize
        std::future::ready(Ok(())).boxed()
    }
}

#[derive(Debug)]
pub struct BinaryPageDecoder {
    inner: Box<dyn LogicalPageDecoder>,
    data_type: DataType,
}

impl LogicalPageDecoder for BinaryPageDecoder {
    fn wait_for_loaded(&mut self, num_rows: u64) -> BoxFuture<'_, Result<()>> {
        self.inner.wait_for_loaded(num_rows)
    }

    fn drain(&mut self, num_rows: u64) -> Result<NextDecodeTask> {
        let inner_task = self.inner.drain(num_rows)?;
        Ok(NextDecodeTask {
            num_rows: inner_task.num_rows,
            task: Box::new(BinaryArrayDecoder {
                inner: inner_task.task,
                data_type: self.data_type.clone(),
            }),
        })
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn rows_loaded(&self) -> u64 {
        self.inner.rows_loaded()
    }

    fn num_rows(&self) -> u64 {
        self.inner.num_rows()
    }

    fn rows_drained(&self) -> u64 {
        self.inner.rows_drained()
    }
}

pub struct BinaryArrayDecoder {
    inner: Box<dyn DecodeArrayTask>,
    data_type: DataType,
}

impl BinaryArrayDecoder {
    fn from_list_array<T: ByteArrayType>(array: &GenericListArray<T::Offset>) -> ArrayRef
    where
        T::Offset: OffsetSizeTrait,
    {
        let values = array
            .values()
            .as_primitive::<UInt8Type>()
            .values()
            .inner()
            .clone();
        let offsets = array.offsets().clone();
        Arc::new(GenericByteArray::<T>::new(
            offsets,
            values,
            array.nulls().cloned(),
        ))
    }

    /// Convert a LargeList (i64 offsets) array to a small-offset byte array (Utf8 / Binary).
    /// Returns an error if the data exceeds the i32 offset limit.
    fn from_large_list_to_small<T: ByteArrayType<Offset = i32>>(
        array: &GenericListArray<i64>,
    ) -> Result<ArrayRef> {
        let values = array
            .values()
            .as_primitive::<UInt8Type>()
            .values()
            .inner()
            .clone();
        let large_offsets = array.offsets();

        // Check if the offsets fit in i32
        let last_offset = large_offsets[large_offsets.len() - 1];
        if last_offset <= i32::MAX as i64 {
            // Safe to downcast to i32 offsets
            let small: Vec<i32> = large_offsets.iter().map(|&o| o as i32).collect();
            let offsets = arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(small));
            Ok(Arc::new(GenericByteArray::<T>::new(
                offsets,
                values,
                array.nulls().cloned(),
            )))
        } else {
            // Data exceeds 2GB -- cannot fit in i32 offsets.
            // This should not happen in practice because the batch-size feedback
            // loop limits batch sizes, but if it does we return a clear error.
            Err(lance_core::Error::invalid_input(format!(
                "A single batch of variable-length data exceeded 2 GiB ({} bytes). \
                 Use LargeUtf8 / LargeBinary in your schema, or reduce the batch size.",
                last_offset
            )))
        }
    }
}

impl DecodeArrayTask for BinaryArrayDecoder {
    fn decode(self: Box<Self>) -> Result<(ArrayRef, u64)> {
        let data_type = self.data_type;
        let (arr, _) = self.inner.decode()?;
        let result = match data_type {
            DataType::Binary => {
                // Internal representation is always LargeList (i64 offsets) now
                if arr.data_type()
                    == &DataType::LargeList(Arc::new(arrow_schema::Field::new(
                        "item",
                        DataType::UInt8,
                        false,
                    )))
                {
                    Self::from_large_list_to_small::<BinaryType>(arr.as_list::<i64>())?
                } else {
                    Self::from_list_array::<BinaryType>(arr.as_list::<i32>())
                }
            }
            DataType::LargeBinary => Self::from_list_array::<LargeBinaryType>(arr.as_list::<i64>()),
            DataType::Utf8 => {
                if arr.data_type()
                    == &DataType::LargeList(Arc::new(arrow_schema::Field::new(
                        "item",
                        DataType::UInt8,
                        false,
                    )))
                {
                    Self::from_large_list_to_small::<Utf8Type>(arr.as_list::<i64>())?
                } else {
                    Self::from_list_array::<Utf8Type>(arr.as_list::<i32>())
                }
            }
            DataType::LargeUtf8 => Self::from_list_array::<LargeUtf8Type>(arr.as_list::<i64>()),
            _ => {
                return Err(lance_core::Error::internal(
                    "Binary decoder does not support this data type",
                ));
            }
        };
        // data_size is only tracked in the v2.1 structural decode path; the legacy
        // v2.0 path does not need it so we return 0.
        Ok((result, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::LargeListBuilder;
    use arrow_array::builder::PrimitiveBuilder;
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt8Type;
    use arrow_buffer::Buffer;
    use arrow_data::ArrayData;

    #[test]
    fn test_from_large_list_to_small() {
        // Create a small LargeList array (i64 offsets) to test downcasting
        let values_builder = PrimitiveBuilder::<UInt8Type>::new();
        let mut builder = LargeListBuilder::new(values_builder);

        // Add "hello"
        builder.values().append_slice(b"hello");
        builder.append(true);

        // Add "world"
        builder.values().append_slice(b"world");
        builder.append(true);

        let large_list_array = builder.finish();

        // Convert to small StringArray
        let result =
            BinaryArrayDecoder::from_large_list_to_small::<Utf8Type>(&large_list_array).unwrap();

        assert_eq!(result.data_type(), &DataType::Utf8);
        let string_array = result.as_string::<i32>();
        assert_eq!(string_array.len(), 2);
        assert_eq!(string_array.value(0), "hello");
        assert_eq!(string_array.value(1), "world");
    }

    #[test]
    fn test_from_large_list_to_small_overflow() {
        // We can manually craft an array with a fake offset that exceeds i32::MAX
        // to test the 2GB limit check without allocating 2GB of memory.
        // Create offsets that exceed i32::MAX
        let offsets = Buffer::from_slice_ref([0_i64, (i32::MAX as i64) + 10]);

        // Field for LargeList
        let field = Arc::new(arrow_schema::Field::new("item", DataType::UInt8, true));

        // SAFETY: The deliberately oversized offset is invalid for the one-byte child array.
        // This lets the test exercise BinaryArrayDecoder's overflow guard without allocating
        // more than 2 GiB of child values.
        let values_data = unsafe {
            ArrayData::builder(DataType::UInt8)
                .len(1)
                .add_buffer(Buffer::from_slice_ref([0_u8]))
                .build_unchecked()
        };

        // SAFETY: This intentionally constructs an invalid LargeList array with offsets beyond
        // the child value length so the helper can reject the i64-to-i32 downcast before building
        // a Binary / Utf8 array.
        let list_data = unsafe {
            ArrayData::builder(DataType::LargeList(field))
                .len(1)
                .add_buffer(offsets)
                .add_child_data(values_data)
                .build_unchecked()
        };
        let large_list_array = GenericListArray::<i64>::from(list_data);

        let result = BinaryArrayDecoder::from_large_list_to_small::<Utf8Type>(&large_list_array);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, lance_core::Error::InvalidInput { .. }));
        assert!(err.to_string().contains("exceeded 2 GiB"));
    }
}
