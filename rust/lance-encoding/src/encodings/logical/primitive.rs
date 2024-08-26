// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{fmt::Debug, ops::Range, sync::Arc, vec};

use arrow_array::{make_array, ArrayRef};
use arrow_buffer::bit_util;
use arrow_schema::DataType;
use futures::{future::BoxFuture, FutureExt};
use lance_arrow::deepcopy::deep_copy_array;
use log::{debug, trace};
use snafu::{location, Location};

use lance_core::{datatypes::Field, Result};

use crate::{
    decoder::{
        DecodeArrayTask, FieldScheduler, FilterExpression, LogicalPageDecoder, NextDecodeTask,
        PageInfo, PageScheduler, PrimitivePageDecoder, PriorityRange, ScheduledScanLine,
        SchedulerContext, SchedulingJob,
    },
    encoder::{
        ArrayEncodingStrategy, EncodeTask, EncodedColumn, EncodedPage, EncodingOptions,
        FieldEncoder,
    },
    encodings::physical::{decoder_from_array_encoding, ColumnBuffers, PageBuffers},
};

#[derive(Debug)]
struct PrimitivePage {
    scheduler: Box<dyn PageScheduler>,
    num_rows: u64,
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
    should_validate: bool,
    column_index: u32,
}

impl PrimitiveFieldScheduler {
    pub fn new(
        column_index: u32,
        data_type: DataType,
        pages: Arc<[PageInfo]>,
        buffers: ColumnBuffers,
        should_validate: bool,
    ) -> Self {
        let page_schedulers = pages
            .iter()
            // Buggy versions of Lance could sometimes create empty pages
            .filter(|page| page.num_rows > 0)
            .map(|page| {
                let page_buffers = PageBuffers {
                    column_buffers: buffers,
                    positions_and_sizes: &page.buffer_offsets_and_sizes,
                };
                let scheduler =
                    decoder_from_array_encoding(&page.encoding, &page_buffers, &data_type);
                PrimitivePage {
                    scheduler,
                    num_rows: page.num_rows,
                }
            })
            .collect::<Vec<_>>();
        let num_rows = page_schedulers.iter().map(|p| p.num_rows).sum();
        Self {
            data_type,
            page_schedulers,
            num_rows,
            should_validate,
            column_index,
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
        priority: &dyn PriorityRange,
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
        while cur_page.num_rows + self.global_row_offset <= range.start {
            self.global_row_offset += cur_page.num_rows;
            self.page_idx += 1;
            trace!("Skipping entire page of {} rows", cur_page.num_rows);
            cur_page = &self.scheduler.page_schedulers[self.page_idx];
        }

        // Now the cur_page has overlap with range.  Continue looping through ranges
        // until we find a range that exceeds the current page

        let mut ranges_in_page = Vec::new();
        while cur_page.num_rows + self.global_row_offset > range.start {
            range.start = range.start.max(self.global_row_offset);
            let start_in_page = range.start - self.global_row_offset;
            let end_in_page = start_in_page + (range.end - range.start);
            let end_in_page = end_in_page.min(cur_page.num_rows);
            let last_in_range = (end_in_page + self.global_row_offset) >= range.end;

            ranges_in_page.push(start_in_page..end_in_page);
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

        self.global_row_offset += cur_page.num_rows;
        self.page_idx += 1;

        let physical_decoder = cur_page.scheduler.schedule_ranges(
            &ranges_in_page,
            context.io(),
            priority.current_priority(),
        );

        let logical_decoder = PrimitiveFieldDecoder {
            data_type: self.scheduler.data_type.clone(),
            column_index: self.scheduler.column_index,
            unloaded_physical_decoder: Some(physical_decoder),
            physical_decoder: None,
            rows_drained: 0,
            num_rows: num_rows_in_next,
            should_validate: self.scheduler.should_validate,
        };

        let decoder = Box::new(logical_decoder);
        let decoder_ready = context.locate_decoder(decoder);
        Ok(ScheduledScanLine {
            decoders: vec![decoder_ready],
            rows_scheduled: num_rows_in_next,
        })
    }

    fn num_rows(&self) -> u64 {
        self.ranges.iter().map(|r| r.end - r.start).sum()
    }
}

impl FieldScheduler for PrimitiveFieldScheduler {
    fn num_rows(&self) -> u64 {
        self.num_rows
    }

    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[std::ops::Range<u64>],
        // TODO: Could potentially use filter to simplify decode, something of a micro-optimization probably
        _filter: &FilterExpression,
    ) -> Result<Box<dyn SchedulingJob + 'a>> {
        Ok(Box::new(PrimitiveFieldSchedulingJob::new(
            self,
            ranges.to_vec(),
        )))
    }
}

pub struct PrimitiveFieldDecoder {
    data_type: DataType,
    unloaded_physical_decoder: Option<BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>>>,
    physical_decoder: Option<Arc<dyn PrimitivePageDecoder>>,
    should_validate: bool,
    num_rows: u64,
    rows_drained: u64,
    column_index: u32,
}

impl PrimitiveFieldDecoder {
    pub fn new_from_data(
        physical_decoder: Arc<dyn PrimitivePageDecoder>,
        data_type: DataType,
        num_rows: u64,
        should_validate: bool,
    ) -> Self {
        Self {
            data_type,
            unloaded_physical_decoder: None,
            physical_decoder: Some(physical_decoder),
            should_validate,
            num_rows,
            rows_drained: 0,
            column_index: u32::MAX,
        }
    }
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
    rows_to_skip: u64,
    rows_to_take: u64,
    should_validate: bool,
    physical_decoder: Arc<dyn PrimitivePageDecoder>,
    data_type: DataType,
}

impl DecodeArrayTask for PrimitiveFieldDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let block = self
            .physical_decoder
            .decode(self.rows_to_skip, self.rows_to_take)?;

        Ok(make_array(
            block.into_arrow(self.data_type, self.should_validate)?,
        ))
    }
}

impl LogicalPageDecoder for PrimitiveFieldDecoder {
    // TODO: In the future, at some point, we may consider partially waiting for primitive pages by
    // breaking up large I/O into smaller I/O as a way to accelerate the "time-to-first-decode"
    fn wait_for_loaded(&mut self, loaded_need: u64) -> BoxFuture<Result<()>> {
        log::trace!(
            "primitive wait for more than {} rows on column {} (page has {} rows)",
            loaded_need,
            self.column_index,
            self.num_rows
        );
        async move {
            let physical_decoder = self.unloaded_physical_decoder.take().unwrap().await?;
            self.physical_decoder = Some(Arc::from(physical_decoder));
            Ok(())
        }
        .boxed()
    }

    fn drain(&mut self, num_rows: u64) -> Result<NextDecodeTask> {
        if self.physical_decoder.as_ref().is_none() {
            return Err(lance_core::Error::Internal {
                message: format!("drain was called on primitive field decoder for data type {} on column {} but the decoder was never awaited", self.data_type, self.column_index),
                location: location!(),
            });
        }

        let rows_to_skip = self.rows_drained;
        let rows_to_take = num_rows;

        self.rows_drained += rows_to_take;

        let task = Box::new(PrimitiveFieldDecodeTask {
            rows_to_skip,
            rows_to_take,
            should_validate: self.should_validate,
            physical_decoder: self.physical_decoder.as_ref().unwrap().clone(),
            data_type: self.data_type.clone(),
        });

        Ok(NextDecodeTask {
            task,
            num_rows: rows_to_take,
            has_more: self.rows_drained != self.num_rows,
        })
    }

    fn rows_loaded(&self) -> u64 {
        if self.unloaded_physical_decoder.is_some() {
            0
        } else {
            self.num_rows
        }
    }

    fn rows_drained(&self) -> u64 {
        if self.unloaded_physical_decoder.is_some() {
            0
        } else {
            self.rows_drained
        }
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
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
    field: Field,
    max_page_bytes: u64,
}

impl PrimitiveFieldEncoder {
    pub fn try_new(
        options: &EncodingOptions,
        array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
        column_index: u32,
        field: Field,
    ) -> Result<Self> {
        Ok(Self {
            accumulation_queue: AccumulationQueue::new(
                options.cache_bytes_per_column,
                column_index,
                options.keep_original_array,
            ),
            column_index,
            max_page_bytes: options.max_page_bytes,
            array_encoding_strategy,
            field,
        })
    }

    fn create_encode_task(&mut self, arrays: Vec<ArrayRef>) -> Result<EncodeTask> {
        let encoder = self
            .array_encoding_strategy
            .create_array_encoder(&arrays, &self.field)?;
        let column_idx = self.column_index;

        Ok(tokio::task::spawn(async move {
            let num_rows = arrays.iter().map(|arr| arr.len() as u64).sum();
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

    // Creates an encode task, consuming all buffered data
    fn do_flush(&mut self, arrays: Vec<ArrayRef>) -> Result<Vec<EncodeTask>> {
        if arrays.len() == 1 {
            let array = arrays.into_iter().next().unwrap();
            let size_bytes = array.get_buffer_memory_size();
            let num_parts = bit_util::ceil(size_bytes, self.max_page_bytes as usize);
            // Can't slice it finer than 1 page per row
            let num_parts = num_parts.min(array.len());
            if num_parts <= 1 {
                // One part and it fits in a page
                Ok(vec![self.create_encode_task(vec![array])?])
            } else {
                // One part and it needs to be sliced into multiple pages

                // This isn't perfect (items in the array might not all have the same size)
                // but it's a reasonable stab for now)
                let mut tasks = Vec::with_capacity(num_parts);
                let mut offset = 0;
                let part_size = bit_util::ceil(array.len(), num_parts);
                for _ in 0..num_parts {
                    let avail = array.len() - offset;
                    let chunk_size = avail.min(part_size);
                    let part = array.slice(offset, chunk_size);
                    let task = self.create_encode_task(vec![part])?;
                    tasks.push(task);
                    offset += chunk_size;
                }
                Ok(tasks)
            }
        } else {
            // Multiple parts that (presumably) all fit in a page
            //
            // TODO: Could check here if there are any jumbo parts in the mix that need splitting
            Ok(vec![self.create_encode_task(arrays)?])
        }
    }
}

impl FieldEncoder for PrimitiveFieldEncoder {
    // Buffers data, if there is enough to write a page then we create an encode task
    fn maybe_encode(&mut self, array: ArrayRef) -> Result<Vec<EncodeTask>> {
        if let Some(arrays) = self.accumulation_queue.insert(array) {
            Ok(self.do_flush(arrays)?)
        } else {
            Ok(vec![])
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self) -> Result<Vec<EncodeTask>> {
        if let Some(arrays) = self.accumulation_queue.flush() {
            Ok(self.do_flush(arrays)?)
        } else {
            Ok(vec![])
        }
    }

    fn num_columns(&self) -> u32 {
        1
    }

    fn finish(&mut self) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        std::future::ready(Ok(vec![EncodedColumn::default()])).boxed()
    }
}
