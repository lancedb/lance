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
    ArrayRef, BooleanArray, PrimitiveArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use bytes::BytesMut;
use futures::{future::BoxFuture, FutureExt};
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::{
    decoder::{
        DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask, PageInfo,
        PhysicalPageDecoder,
    },
    encoder::{ArrayEncoder, EncodedPage, FieldEncoder},
    EncodingsIo,
};

/// A page schedule for primitive fields
///
/// This maps to exactly one physical page and it assumes that the top-level
/// encoding of the page is "basic".  The basic encoding decodes into an
/// optional buffer of validity and a fixed-width buffer of values
/// which is exactly what we need to create a primitive array.
pub struct PrimitivePageScheduler {
    data_type: DataType,
    page: Arc<PageInfo>,
}

impl PrimitivePageScheduler {
    /// Create a new instance
    ///
    /// # Arguments
    ///
    /// * `data_type` - The Arrow type of the field.  This must be a primitive data type.
    ///   (although our definition of primitive here includes boolean and fixed size list
    ///   which is slightly different than arrow-rs' definition)
    /// * `page` - The physical page info
    pub fn new(data_type: DataType, page: Arc<PageInfo>) -> Self {
        Self { data_type, page }
    }
}

impl LogicalPageScheduler for PrimitivePageScheduler {
    fn num_rows(&self) -> u32 {
        self.page.num_rows
    }

    fn schedule_range(
        &self,
        range: std::ops::Range<u32>,
        scheduler: &Arc<dyn EncodingsIo>,
    ) -> Result<Box<dyn LogicalPageDecoder>> {
        let num_rows = range.end - range.start;
        let physical_decoder = self.page.decoder.schedule_range(range, scheduler.as_ref());

        let logical_decoder = PrimitiveFieldDecoder {
            data_type: self.data_type.clone(),
            unloaded_physical_decoder: Some(physical_decoder),
            physical_decoder: None,
            rows_drained: 0,
            num_rows,
        };

        Ok(Box::new(logical_decoder))
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

        Arc::new(PrimitiveArray::<T>::new(data_buffer, null_buffer))
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
            DataType::Date32 => Ok(Self::new_primitive_array::<Date32Type>(buffers, num_rows)),
            DataType::Date64 => Ok(Self::new_primitive_array::<Date64Type>(buffers, num_rows)),
            DataType::Decimal128(_, _) => Ok(Self::new_primitive_array::<Decimal128Type>(
                buffers, num_rows,
            )),
            DataType::Decimal256(_, _) => Ok(Self::new_primitive_array::<Decimal256Type>(
                buffers, num_rows,
            )),
            DataType::Duration(units) => Ok(match units {
                TimeUnit::Second => {
                    Self::new_primitive_array::<DurationSecondType>(buffers, num_rows)
                }
                TimeUnit::Microsecond => {
                    Self::new_primitive_array::<DurationMicrosecondType>(buffers, num_rows)
                }
                TimeUnit::Millisecond => {
                    Self::new_primitive_array::<DurationMillisecondType>(buffers, num_rows)
                }
                TimeUnit::Nanosecond => {
                    Self::new_primitive_array::<DurationNanosecondType>(buffers, num_rows)
                }
            }),
            DataType::Float16 => Ok(Self::new_primitive_array::<Float16Type>(buffers, num_rows)),
            DataType::Float32 => Ok(Self::new_primitive_array::<Float32Type>(buffers, num_rows)),
            DataType::Float64 => Ok(Self::new_primitive_array::<Float64Type>(buffers, num_rows)),
            DataType::Int16 => Ok(Self::new_primitive_array::<Int16Type>(buffers, num_rows)),
            DataType::Int32 => Ok(Self::new_primitive_array::<Int32Type>(buffers, num_rows)),
            DataType::Int64 => Ok(Self::new_primitive_array::<Int64Type>(buffers, num_rows)),
            DataType::Int8 => Ok(Self::new_primitive_array::<Int8Type>(buffers, num_rows)),
            DataType::Interval(unit) => Ok(match unit {
                IntervalUnit::DayTime => {
                    Self::new_primitive_array::<IntervalDayTimeType>(buffers, num_rows)
                }
                IntervalUnit::MonthDayNano => {
                    Self::new_primitive_array::<IntervalMonthDayNanoType>(buffers, num_rows)
                }
                IntervalUnit::YearMonth => {
                    Self::new_primitive_array::<IntervalYearMonthType>(buffers, num_rows)
                }
            }),
            DataType::Null => Ok(new_null_array(data_type, num_rows as usize)),
            DataType::Time32(unit) => match unit {
                TimeUnit::Millisecond => Ok(Self::new_primitive_array::<Time32MillisecondType>(
                    buffers, num_rows,
                )),
                TimeUnit::Second => Ok(Self::new_primitive_array::<Time32SecondType>(
                    buffers, num_rows,
                )),
                _ => Err(Error::IO {
                    message: format!("invalid time unit {:?} for 32-bit time type", unit),
                    location: location!(),
                }),
            },
            DataType::Time64(unit) => match unit {
                TimeUnit::Microsecond => Ok(Self::new_primitive_array::<Time64MicrosecondType>(
                    buffers, num_rows,
                )),
                TimeUnit::Nanosecond => Ok(Self::new_primitive_array::<Time64NanosecondType>(
                    buffers, num_rows,
                )),
                _ => Err(Error::IO {
                    message: format!("invalid time unit {:?} for 64-bit time type", unit),
                    location: location!(),
                }),
            },
            DataType::Timestamp(unit, _) => Ok(match unit {
                TimeUnit::Microsecond => {
                    Self::new_primitive_array::<TimestampMicrosecondType>(buffers, num_rows)
                }
                TimeUnit::Millisecond => {
                    Self::new_primitive_array::<TimestampMillisecondType>(buffers, num_rows)
                }
                TimeUnit::Nanosecond => {
                    Self::new_primitive_array::<TimestampNanosecondType>(buffers, num_rows)
                }
                TimeUnit::Second => {
                    Self::new_primitive_array::<TimestampSecondType>(buffers, num_rows)
                }
            }),
            DataType::UInt16 => Ok(Self::new_primitive_array::<UInt16Type>(buffers, num_rows)),
            DataType::UInt32 => Ok(Self::new_primitive_array::<UInt32Type>(buffers, num_rows)),
            DataType::UInt64 => Ok(Self::new_primitive_array::<UInt64Type>(buffers, num_rows)),
            DataType::UInt8 => Ok(Self::new_primitive_array::<UInt8Type>(buffers, num_rows)),
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
    fn wait(&mut self) -> BoxFuture<Result<()>> {
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

    fn avail(&self) -> u32 {
        self.num_rows - self.rows_drained
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
    fn do_flush(&mut self) -> BoxFuture<'static, Result<Vec<EncodedPage>>> {
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
    ) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>> {
        self.current_bytes += array.get_array_memory_size() as u64;
        self.buffered_arrays.push(array);
        if self.current_bytes > self.cache_bytes {
            Some(self.do_flush())
        } else {
            None
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>> {
        if self.current_bytes > 0 {
            Some(self.do_flush())
        } else {
            None
        }
    }

    fn num_columns(&self) -> u32 {
        1
    }
}
