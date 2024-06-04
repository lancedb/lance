// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{fmt::Debug, ops::Range, sync::Arc};

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
    ArrayRef, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray, PrimitiveArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use bytes::BytesMut;
use futures::{future::BoxFuture, FutureExt};
use lance_arrow::deepcopy::deep_copy_array;
use log::{debug, trace};
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::{
    decoder::{
        DecodeArrayTask, FieldScheduler, LogicalPageDecoder, NextDecodeTask, PageInfo,
        PageScheduler, PhysicalPageDecoder, ScheduledScanLine, SchedulerContext, SchedulingJob,
    },
    encoder::{ArrayEncodingStrategy, EncodeTask, EncodedPage, FieldEncoder},
    encodings::physical::{decoder_from_array_encoding, ColumnBuffers, PageBuffers},
};

#[derive(Debug)]
struct PrimitivePage {
    scheduler: Box<dyn PageScheduler>,
    num_rows: u32,
}

/// A field scheduler for primitive fields
///
/// This maps to exactly one column and it assumes that the top-level
/// encoding of each page is "basic".  The basic encoding decodes into an
/// optional buffer of validity and a fixed-width buffer of values
/// which is exactly what we need to create a primitive array.
///
/// Note: we consider booleans and fixed-size-lists of primitive types to be
/// primitive types.  This is slightly different than arrow-rs's definition
#[derive(Debug)]
pub struct PrimitiveFieldScheduler {
    data_type: DataType,
    page_schedulers: Vec<PrimitivePage>,
    num_rows: u64,
}

impl PrimitiveFieldScheduler {
    pub fn new(data_type: DataType, pages: Arc<[PageInfo]>, buffers: ColumnBuffers) -> Self {
        let page_schedulers = pages
            .iter()
            .map(|page| {
                let page_buffers = PageBuffers {
                    column_buffers: buffers,
                    positions_and_sizes: &page.buffer_offsets_and_sizes,
                };
                let scheduler = decoder_from_array_encoding(&page.encoding, &page_buffers);
                PrimitivePage {
                    scheduler,
                    num_rows: page.num_rows,
                }
            })
            .collect::<Vec<_>>();
        let num_rows = page_schedulers.iter().map(|p| p.num_rows as u64).sum();
        Self {
            data_type,
            page_schedulers,
            num_rows,
        }
    }
}

#[derive(Debug)]
struct PrimitiveFieldSchedulingJob<'a> {
    scheduler: &'a PrimitiveFieldScheduler,
    ranges: Vec<Range<u64>>,
    page_idx: usize,
    range_idx: usize,
    range_offset: u64,
    global_row_offset: u64,
}

impl<'a> PrimitiveFieldSchedulingJob<'a> {
    pub fn new(scheduler: &'a PrimitiveFieldScheduler, ranges: Vec<Range<u64>>) -> Self {
        Self {
            scheduler,
            ranges,
            page_idx: 0,
            range_idx: 0,
            range_offset: 0,
            global_row_offset: 0,
        }
    }
}

impl<'a> SchedulingJob for PrimitiveFieldSchedulingJob<'a> {
    fn schedule_next(
        &mut self,
        context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<ScheduledScanLine> {
        debug_assert!(self.range_idx < self.ranges.len());
        // Get our current range
        let mut range = self.ranges[self.range_idx].clone();
        range.start += self.range_offset;

        let mut cur_page = &self.scheduler.page_schedulers[self.page_idx];
        trace!(
            "Current range is {:?} and current page has {} rows",
            range,
            cur_page.num_rows
        );
        // Skip entire pages until we have some overlap with our next range
        while cur_page.num_rows as u64 + self.global_row_offset <= range.start {
            self.global_row_offset += cur_page.num_rows as u64;
            self.page_idx += 1;
            trace!("Skipping entire page of {} rows", cur_page.num_rows);
            cur_page = &self.scheduler.page_schedulers[self.page_idx];
        }

        // Now the cur_page has overlap with range.  Continue looping through ranges
        // until we find a range that exceeds the current page

        let mut ranges_in_page = Vec::new();
        while cur_page.num_rows as u64 + self.global_row_offset > range.start {
            range.start = range.start.max(self.global_row_offset);
            let start_in_page = range.start - self.global_row_offset;
            let end_in_page = start_in_page + (range.end - range.start);
            let end_in_page = end_in_page.min(cur_page.num_rows as u64) as u32;
            let last_in_range = (end_in_page as u64 + self.global_row_offset) >= range.end;

            ranges_in_page.push(start_in_page as u32..end_in_page);
            if last_in_range {
                self.range_idx += 1;
                if self.range_idx == self.ranges.len() {
                    break;
                }
                range = self.ranges[self.range_idx].clone();
            } else {
                break;
            }
        }

        let num_rows_in_next = ranges_in_page.iter().map(|r| r.end - r.start).sum();
        trace!(
            "Scheduling {} rows across {} ranges from page with {} rows",
            num_rows_in_next,
            ranges_in_page.len(),
            cur_page.num_rows
        );

        self.global_row_offset += cur_page.num_rows as u64;
        self.page_idx += 1;

        let physical_decoder = cur_page.scheduler.schedule_ranges(
            &ranges_in_page,
            context.io().as_ref(),
            top_level_row,
        );

        let logical_decoder = PrimitiveFieldDecoder {
            data_type: self.scheduler.data_type.clone(),
            unloaded_physical_decoder: Some(physical_decoder),
            physical_decoder: None,
            rows_drained: 0,
            num_rows: num_rows_in_next,
        };

        let decoder = Box::new(logical_decoder);
        let decoder_ready = context.locate_decoder(decoder);
        Ok(ScheduledScanLine {
            decoders: vec![decoder_ready],
            rows_scheduled: num_rows_in_next,
        })
    }
}

impl FieldScheduler for PrimitiveFieldScheduler {
    fn num_rows(&self) -> u64 {
        self.num_rows
    }

    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[std::ops::Range<u64>],
    ) -> Result<Box<dyn SchedulingJob + 'a>> {
        Ok(Box::new(PrimitiveFieldSchedulingJob::new(
            self,
            ranges.to_vec(),
        )))
    }
}

struct PrimitiveFieldDecoder {
    data_type: DataType,
    unloaded_physical_decoder: Option<BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>>>,
    physical_decoder: Option<Arc<dyn PhysicalPageDecoder>>,
    num_rows: u32,
    rows_drained: u32,
}

impl Debug for PrimitiveFieldDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimitiveFieldDecoder")
            .field("data_type", &self.data_type)
            .field("num_rows", &self.num_rows)
            .field("rows_drained", &self.rows_drained)
            .finish()
    }
}

struct PrimitiveFieldDecodeTask {
    rows_to_skip: u32,
    rows_to_take: u32,
    physical_decoder: Arc<dyn PhysicalPageDecoder>,
    data_type: DataType,
}

impl DecodeArrayTask for PrimitiveFieldDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        // We start by assuming that no buffers are required.  The number of buffers needed is based
        // on the data type.  Most data types need two buffers but each layer of fixed-size-list, for
        // example, adds another validity buffer
        let mut capacities = vec![(0, false); self.physical_decoder.num_buffers() as usize];
        let mut all_null = false;
        self.physical_decoder.update_capacity(
            self.rows_to_skip,
            self.rows_to_take,
            &mut capacities,
            &mut all_null,
        );

        if all_null {
            return Ok(new_null_array(&self.data_type, self.rows_to_take as usize));
        }

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
            .decode_into(self.rows_to_skip, self.rows_to_take, &mut bufs)?;

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
        let data_buffer = Buffer::from_bytes(data_buffer.into());
        let data_buffer = ScalarBuffer::<T::Native>::new(data_buffer, 0, num_rows as usize);

        // The with_data_type is needed here to recover the parameters for types like Decimal/Timestamp
        Arc::new(
            PrimitiveArray::<T>::new(data_buffer, null_buffer).with_data_type(data_type.clone()),
        )
    }

    fn bytes_to_validity(bytes: BytesMut, num_rows: u32) -> Option<NullBuffer> {
        if bytes.is_empty() {
            None
        } else {
            let null_buffer = bytes.freeze().into();
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from_bytes(null_buffer),
                0,
                num_rows as usize,
            )))
        }
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
                let null_buffer = Self::bytes_to_validity(null_buffer, num_rows);

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
                _ => Err(Error::io(
                    format!("invalid time unit {:?} for 32-bit time type", unit),
                    location!(),
                )),
            },
            DataType::Time64(unit) => match unit {
                TimeUnit::Microsecond => Ok(Self::new_primitive_array::<Time64MicrosecondType>(
                    buffers, num_rows, data_type,
                )),
                TimeUnit::Nanosecond => Ok(Self::new_primitive_array::<Time64NanosecondType>(
                    buffers, num_rows, data_type,
                )),
                _ => Err(Error::io(
                    format!("invalid time unit {:?} for 64-bit time type", unit),
                    location!(),
                )),
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
            DataType::FixedSizeBinary(dimension) => {
                let mut buffers_iter = buffers.into_iter();
                let fsb_validity = buffers_iter.next().unwrap();
                let fsb_nulls = Self::bytes_to_validity(fsb_validity, num_rows);

                let fsb_values = buffers_iter.next().unwrap();
                let fsb_values = Buffer::from_bytes(fsb_values.freeze().into());
                Ok(Arc::new(FixedSizeBinaryArray::new(
                    *dimension, fsb_values, fsb_nulls,
                )))
            }
            DataType::FixedSizeList(items, dimension) => {
                let mut buffers_iter = buffers.into_iter();
                let fsl_validity = buffers_iter.next().unwrap();
                let fsl_nulls = Self::bytes_to_validity(fsl_validity, num_rows);

                let remaining_buffers = buffers_iter.collect::<Vec<_>>();
                let items_array = Self::primitive_array_from_buffers(
                    items.data_type(),
                    remaining_buffers,
                    num_rows * (*dimension as u32),
                )?;
                Ok(Arc::new(FixedSizeListArray::new(
                    items.clone(),
                    *dimension,
                    items_array,
                    fsl_nulls,
                )))
            }
            _ => Err(Error::io(
                format!(
                    "The data type {} cannot be decoded from a primitive encoding",
                    data_type
                ),
                location!(),
            )),
        }
    }
}

impl LogicalPageDecoder for PrimitiveFieldDecoder {
    // TODO: In the future, at some point, we may consider partially waiting for primitive pages by
    // breaking up large I/O into smaller I/O as a way to accelerate the "time-to-first-decode"
    fn wait(&mut self, _: u32) -> BoxFuture<Result<()>> {
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

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

#[derive(Debug)]
pub struct AccumulationQueue {
    cache_bytes: u64,
    keep_original_array: bool,
    buffered_arrays: Vec<ArrayRef>,
    current_bytes: u64,
    // This is only for logging / debugging purposes
    column_index: u32,
}

impl AccumulationQueue {
    pub fn new(cache_bytes: u64, column_index: u32, keep_original_array: bool) -> Self {
        Self {
            cache_bytes,
            buffered_arrays: Vec::new(),
            current_bytes: 0,
            column_index,
            keep_original_array,
        }
    }

    /// Adds an array to the queue, if there is enough data then the queue is flushed
    /// and returned
    pub fn insert(&mut self, array: ArrayRef) -> Option<Vec<ArrayRef>> {
        self.current_bytes += array.get_array_memory_size() as u64;
        if self.current_bytes > self.cache_bytes {
            debug!(
                "Flushing column {} page of size {} bytes (unencoded)",
                self.column_index, self.current_bytes
            );
            // Push into buffered_arrays without copy since we are about to flush anyways
            self.buffered_arrays.push(array);
            self.current_bytes = 0;
            Some(std::mem::take(&mut self.buffered_arrays))
        } else {
            trace!(
                "Accumulating data for column {}.  Now at {} bytes",
                self.column_index,
                self.current_bytes
            );
            if self.keep_original_array {
                self.buffered_arrays.push(array);
            } else {
                self.buffered_arrays.push(deep_copy_array(array.as_ref()))
            }
            None
        }
    }

    pub fn flush(&mut self) -> Option<Vec<ArrayRef>> {
        if self.buffered_arrays.is_empty() {
            trace!(
                "No final flush since no data at column {}",
                self.column_index
            );
            None
        } else {
            trace!(
                "Final flush of column {} which has {} bytes",
                self.column_index,
                self.current_bytes
            );
            self.current_bytes = 0;
            Some(std::mem::take(&mut self.buffered_arrays))
        }
    }
}

pub struct PrimitiveFieldEncoder {
    accumulation_queue: AccumulationQueue,
    array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
    column_index: u32,
}

impl PrimitiveFieldEncoder {
    pub fn try_new(
        cache_bytes: u64,
        keep_original_array: bool,
        array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
        column_index: u32,
    ) -> Result<Self> {
        Ok(Self {
            accumulation_queue: AccumulationQueue::new(
                cache_bytes,
                column_index,
                keep_original_array,
            ),
            column_index,
            array_encoding_strategy,
        })
    }

    // Creates an encode task, consuming all buffered data
    fn do_flush(&mut self, arrays: Vec<ArrayRef>) -> Result<EncodeTask> {
        let encoder = self.array_encoding_strategy.create_array_encoder(&arrays)?;
        let column_idx = self.column_index;

        Ok(tokio::task::spawn(async move {
            let num_rows = arrays.iter().map(|arr| arr.len() as u32).sum();
            let mut buffer_index = 0;
            let array = encoder.encode(&arrays, &mut buffer_index)?;
            Ok(EncodedPage {
                array,
                num_rows,
                column_idx,
            })
        })
        .map(|res_res| res_res.unwrap())
        .boxed())
    }
}

impl FieldEncoder for PrimitiveFieldEncoder {
    // Buffers data, if there is enough to write a page then we create an encode task
    fn maybe_encode(&mut self, array: ArrayRef) -> Result<Vec<EncodeTask>> {
        if let Some(arrays) = self.accumulation_queue.insert(array) {
            Ok(vec![self.do_flush(arrays)?])
        } else {
            Ok(vec![])
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self) -> Result<Vec<EncodeTask>> {
        if let Some(arrays) = self.accumulation_queue.flush() {
            Ok(vec![self.do_flush(arrays)?])
        } else {
            Ok(vec![])
        }
    }

    fn num_columns(&self) -> u32 {
        1
    }
}
