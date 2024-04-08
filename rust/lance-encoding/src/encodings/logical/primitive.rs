use std::sync::Arc;

use arrow_array::{
    new_null_array,
    types::{
        ArrowPrimitiveType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, IntervalDayTimeType, IntervalMonthDayNanoType, IntervalYearMonthType,
        Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    ArrayRef, BooleanArray, FixedSizeListArray, PrimitiveArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use bytes::BytesMut;
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};

use lance_core::{Error, Result};
use tokio::sync::mpsc;

use crate::{
    decoder::{
        DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask, PageInfo,
        PhysicalPageDecoder, PhysicalPageScheduler,
    },
    encoder::{ArrayEncoder, EncodedPage, FieldEncoder},
    encodings::physical::{decoder_from_array_encoding, ColumnBuffers, PageBuffers},
    EncodingsIo,
};

/// A page scheduler for primitive fields
///
/// This maps to exactly one physical page and it assumes that the top-level
/// encoding of the page is "basic".  The basic encoding decodes into an
/// optional buffer of validity and a fixed-width buffer of values
/// which is exactly what we need to create a primitive array.
///
/// Note: we consider booleans and fixed-size-lists of primitive types to be
/// primitive types.  This is slightly different than arrow-rs's definition
#[derive(Debug)]
pub struct PrimitivePageScheduler {
    data_type: DataType,
    physical_decoder: Box<dyn PhysicalPageScheduler>,
    num_rows: u32,
}

impl PrimitivePageScheduler {
    pub fn new(data_type: DataType, page: Arc<PageInfo>, buffers: ColumnBuffers) -> Self {
        let page_buffers = PageBuffers {
            column_buffers: buffers,
            positions: &page.buffer_offsets,
        };
        Self {
            data_type,
            physical_decoder: decoder_from_array_encoding(&page.encoding, &page_buffers),
            num_rows: page.num_rows,
        }
    }
}

impl LogicalPageScheduler for PrimitivePageScheduler {
    fn num_rows(&self) -> u32 {
        self.num_rows
    }

    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();
        trace!("Scheduling ranges {:?} from physical page", ranges);
        let physical_decoder = self
            .physical_decoder
            .schedule_ranges(ranges, scheduler.as_ref());

        let logical_decoder = PrimitiveFieldDecoder {
            data_type: self.data_type.clone(),
            unloaded_physical_decoder: Some(physical_decoder),
            physical_decoder: None,
            rows_drained: 0,
            num_rows,
        };

        sink.send(Box::new(logical_decoder)).unwrap();
        Ok(())
    }

    fn schedule_take(
        &self,
        indices: &[u32],
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        trace!(
            "Scheduling take of {} indices from physical page",
            indices.len()
        );
        self.schedule_ranges(
            &indices
                .iter()
                .map(|&idx| idx..(idx + 1))
                .collect::<Vec<_>>(),
            scheduler,
            sink,
        )
    }
}

struct PrimitiveFieldDecoder {
    data_type: DataType,
    unloaded_physical_decoder: Option<BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>>,
    physical_decoder: Option<Arc<dyn PhysicalPageDecoder>>,
    num_rows: u32,
    rows_drained: u32,
}

struct PrimitiveFieldDecodeTask {
    rows_to_skip: u32,
    rows_to_take: u32,
    physical_decoder: Arc<dyn PhysicalPageDecoder>,
    data_type: DataType,
}

impl DecodeArrayTask for PrimitiveFieldDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        // There are two buffers, the validity buffer and the values buffer
        // We start by assuming the validity buffer will not be required
        let mut capacities = [(0, false), (0, true)];
        self.physical_decoder.update_capacity(
            self.rows_to_skip,
            self.rows_to_take,
            &mut capacities,
        );
        // At this point we know the size needed for each buffer
        let mut bufs = capacities
            .into_iter()
            .map(|(num_bytes, is_needed)| {
                // Only allocate the validity buffer if it is needed, otherwise we
                // create an empty BytesMut (does not require allocation)
                if is_needed {
                    BytesMut::with_capacity(num_bytes as usize)
                } else {
                    BytesMut::default()
                }
            })
            .collect::<Vec<_>>();

        // Go ahead and fill the validity / values buffers
        self.physical_decoder
            .decode_into(self.rows_to_skip, self.rows_to_take, &mut bufs);

        // Convert the two buffers into an Arrow array
        Self::primitive_array_from_buffers(&self.data_type, bufs, self.rows_to_take)
    }
}

impl PrimitiveFieldDecodeTask {
    // TODO: Does this capability exist upstream somewhere?  I couldn't find
    // it from a simple scan but it seems the ability to convert two buffers
    // into a primitive array is pretty fundamental.
    fn new_primitive_array<T: ArrowPrimitiveType>(
        buffers: Vec<BytesMut>,
        num_rows: u32,
        data_type: &DataType,
    ) -> ArrayRef {
        let mut buffer_iter = buffers.into_iter();
        let null_buffer = buffer_iter.next().unwrap();
        let null_buffer = if null_buffer.is_empty() {
            None
        } else {
            let null_buffer = null_buffer.freeze().into();
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from_bytes(null_buffer),
                0,
                num_rows as usize,
            )))
        };

        let data_buffer = buffer_iter.next().unwrap().freeze();
        let data_buffer = Buffer::from(data_buffer);
        let data_buffer = ScalarBuffer::<T::Native>::new(data_buffer, 0, num_rows as usize);

        // The with_data_type is needed here to recover the parameters for types like Decimal/Timestamp
        Arc::new(
            PrimitiveArray::<T>::new(data_buffer, null_buffer).with_data_type(data_type.clone()),
        )
    }

    fn primitive_array_from_buffers(
        data_type: &DataType,
        buffers: Vec<BytesMut>,
        num_rows: u32,
    ) -> Result<ArrayRef> {
        match data_type {
            DataType::Boolean => {
                let mut buffer_iter = buffers.into_iter();
                let null_buffer = buffer_iter.next().unwrap();
                let null_buffer = if null_buffer.is_empty() {
                    None
                } else {
                    let null_buffer = null_buffer.freeze().into();
                    Some(NullBuffer::new(BooleanBuffer::new(
                        Buffer::from_bytes(null_buffer),
                        0,
                        num_rows as usize,
                    )))
                };

                let data_buffer = buffer_iter.next().unwrap().freeze();
                let data_buffer = Buffer::from(data_buffer);
                let data_buffer = BooleanBuffer::new(data_buffer, 0, num_rows as usize);

                Ok(Arc::new(BooleanArray::new(data_buffer, null_buffer)))
            }
            DataType::Date32 => Ok(Self::new_primitive_array::<Date32Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Date64 => Ok(Self::new_primitive_array::<Date64Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Decimal128(_, _) => Ok(Self::new_primitive_array::<Decimal128Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Decimal256(_, _) => Ok(Self::new_primitive_array::<Decimal256Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Duration(units) => Ok(match units {
                TimeUnit::Second => {
                    Self::new_primitive_array::<DurationSecondType>(buffers, num_rows, data_type)
                }
                TimeUnit::Microsecond => Self::new_primitive_array::<DurationMicrosecondType>(
                    buffers, num_rows, data_type,
                ),
                TimeUnit::Millisecond => Self::new_primitive_array::<DurationMillisecondType>(
                    buffers, num_rows, data_type,
                ),
                TimeUnit::Nanosecond => Self::new_primitive_array::<DurationNanosecondType>(
                    buffers, num_rows, data_type,
                ),
            }),
            DataType::Float16 => Ok(Self::new_primitive_array::<Float16Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Float32 => Ok(Self::new_primitive_array::<Float32Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Float64 => Ok(Self::new_primitive_array::<Float64Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Int16 => Ok(Self::new_primitive_array::<Int16Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Int32 => Ok(Self::new_primitive_array::<Int32Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Int64 => Ok(Self::new_primitive_array::<Int64Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Int8 => Ok(Self::new_primitive_array::<Int8Type>(
                buffers, num_rows, data_type,
            )),
            DataType::Interval(unit) => Ok(match unit {
                IntervalUnit::DayTime => {
                    Self::new_primitive_array::<IntervalDayTimeType>(buffers, num_rows, data_type)
                }
                IntervalUnit::MonthDayNano => {
                    Self::new_primitive_array::<IntervalMonthDayNanoType>(
                        buffers, num_rows, data_type,
                    )
                }
                IntervalUnit::YearMonth => {
                    Self::new_primitive_array::<IntervalYearMonthType>(buffers, num_rows, data_type)
                }
            }),
            DataType::Null => Ok(new_null_array(data_type, num_rows as usize)),
            DataType::Time32(unit) => match unit {
                TimeUnit::Millisecond => Ok(Self::new_primitive_array::<Time32MillisecondType>(
                    buffers, num_rows, data_type,
                )),
                TimeUnit::Second => Ok(Self::new_primitive_array::<Time32SecondType>(
                    buffers, num_rows, data_type,
                )),
                _ => Err(Error::IO {
                    message: format!("invalid time unit {:?} for 32-bit time type", unit),
                    location: location!(),
                }),
            },
            DataType::Time64(unit) => match unit {
                TimeUnit::Microsecond => Ok(Self::new_primitive_array::<Time64MicrosecondType>(
                    buffers, num_rows, data_type,
                )),
                TimeUnit::Nanosecond => Ok(Self::new_primitive_array::<Time64NanosecondType>(
                    buffers, num_rows, data_type,
                )),
                _ => Err(Error::IO {
                    message: format!("invalid time unit {:?} for 64-bit time type", unit),
                    location: location!(),
                }),
            },
            DataType::Timestamp(unit, _) => Ok(match unit {
                TimeUnit::Microsecond => Self::new_primitive_array::<TimestampMicrosecondType>(
                    buffers, num_rows, data_type,
                ),
                TimeUnit::Millisecond => Self::new_primitive_array::<TimestampMillisecondType>(
                    buffers, num_rows, data_type,
                ),
                TimeUnit::Nanosecond => Self::new_primitive_array::<TimestampNanosecondType>(
                    buffers, num_rows, data_type,
                ),
                TimeUnit::Second => {
                    Self::new_primitive_array::<TimestampSecondType>(buffers, num_rows, data_type)
                }
            }),
            DataType::UInt16 => Ok(Self::new_primitive_array::<UInt16Type>(
                buffers, num_rows, data_type,
            )),
            DataType::UInt32 => Ok(Self::new_primitive_array::<UInt32Type>(
                buffers, num_rows, data_type,
            )),
            DataType::UInt64 => Ok(Self::new_primitive_array::<UInt64Type>(
                buffers, num_rows, data_type,
            )),
            DataType::UInt8 => Ok(Self::new_primitive_array::<UInt8Type>(
                buffers, num_rows, data_type,
            )),
            DataType::FixedSizeList(items, dimension) => {
                let items_array = Self::primitive_array_from_buffers(
                    items.data_type(),
                    buffers,
                    num_rows * (*dimension as u32),
                )?;
                Ok(Arc::new(FixedSizeListArray::new(
                    items.clone(),
                    *dimension,
                    items_array,
                    None,
                )))
            }
            _ => Err(Error::IO {
                message: format!(
                    "The data type {} cannot be decoded from a primitive encoding",
                    data_type
                ),
                location: location!(),
            }),
        }
    }
}

impl LogicalPageDecoder for PrimitiveFieldDecoder {
    fn wait<'a>(
        &'a mut self,
        _: u32,
        _: &'a mut mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'a, Result<()>> {
        async move {
            let physical_decoder = self.unloaded_physical_decoder.take().unwrap().await?;
            self.physical_decoder = Some(Arc::from(physical_decoder));
            Ok(())
        }
        .boxed()
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        let rows_to_skip = self.rows_drained;
        let rows_to_take = num_rows;

        self.rows_drained += rows_to_take;

        let task = Box::new(PrimitiveFieldDecodeTask {
            rows_to_skip,
            rows_to_take,
            physical_decoder: self.physical_decoder.as_ref().unwrap().clone(),
            data_type: self.data_type.clone(),
        });

        Ok(NextDecodeTask {
            task,
            num_rows: rows_to_take,
            has_more: self.rows_drained != self.num_rows,
        })
    }

    fn unawaited(&self) -> u32 {
        if self.unloaded_physical_decoder.is_some() {
            self.num_rows
        } else {
            0
        }
    }

    fn avail(&self) -> u32 {
        if self.unloaded_physical_decoder.is_some() {
            0
        } else {
            self.num_rows - self.rows_drained
        }
    }
}

pub struct PrimitiveFieldEncoder {
    cache_bytes: u64,
    buffered_arrays: Vec<ArrayRef>,
    current_bytes: u64,
    encoder: Arc<dyn ArrayEncoder>,
}

impl PrimitiveFieldEncoder {
    pub fn new(cache_bytes: u64, encoder: Arc<dyn ArrayEncoder>) -> Self {
        Self {
            cache_bytes,
            buffered_arrays: Vec::with_capacity(8),
            current_bytes: 0,
            encoder,
        }
    }

    // Creates an encode task, consuming all buffered data
    fn do_flush(&mut self) -> BoxFuture<'static, Result<EncodedPage>> {
        let mut arrays = Vec::new();
        std::mem::swap(&mut arrays, &mut self.buffered_arrays);
        self.current_bytes = 0;
        let encoder = self.encoder.clone();

        tokio::task::spawn(async move { encoder.encode(&arrays) })
            .map(|res_res| res_res.unwrap())
            .boxed()
    }
}

impl FieldEncoder for PrimitiveFieldEncoder {
    // Buffers data, if there is enough to write a page then we create an encode task
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        self.current_bytes += array.get_array_memory_size() as u64;
        self.buffered_arrays.push(array);
        if self.current_bytes > self.cache_bytes {
            Ok(vec![self.do_flush()])
        } else {
            Ok(vec![])
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        if self.current_bytes > 0 {
            Ok(vec![self.do_flush()])
        } else {
            Ok(vec![])
        }
    }

    fn num_columns(&self) -> u32 {
        1
    }
}
