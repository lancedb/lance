// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::VecDeque, ops::Range, sync::Arc};

use arrow_array::{cast::AsArray, ArrayRef, StructArray};
use arrow_schema::{DataType, Fields};
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};

use crate::{
    decoder::{
        DecodeArrayTask, DecoderReady, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask,
        SchedulerContext,
    },
    encoder::{EncodeTask, EncodedArray, EncodedPage, FieldEncoder},
    format::pb,
};
use lance_core::{Error, Result};

/// A scheduler for structs
///
/// The implementation is actually a bit more tricky than one might initially think.  We can't just
/// go through and schedule each column one after the other.  This would mean our decode can't start
/// until nearly all the data has arrived (since we need data from each column)
///
/// Instead, we schedule in row-major fashion, described in detail below.
///
/// Note: this scheduler is the starting point for all decoding.  This is because we treat the top-level
/// record batch as a non-nullable struct.
///
/// This means this scheduler has to deal with u64 indices / ranges because the top-level
/// scheduler spans up to u64 rows.
#[derive(Debug)]
pub struct SimpleStructScheduler {
    children: Vec<Vec<Arc<dyn LogicalPageScheduler>>>,
    child_fields: Fields,
    // A single page cannot contain more than u32 rows.  However, we also use SimpleStructScheduler
    // at the top level and a single file *can* contain more than u32 rows.
    num_rows: u64,
    // True if this is the top-level decoder (and we should send scan line messages)
    is_root: bool,
}

impl SimpleStructScheduler {
    fn new_with_params(
        children: Vec<Vec<Arc<dyn LogicalPageScheduler>>>,
        child_fields: Fields,
        is_root: bool,
    ) -> Self {
        debug_assert!(!children.is_empty());
        let num_rows = children[0].iter().map(|page| page.num_rows() as u64).sum();
        // Ensure that all the children have the same number of rows
        Self {
            children,
            child_fields,
            num_rows,
            is_root,
        }
    }

    pub fn new(children: Vec<Vec<Arc<dyn LogicalPageScheduler>>>, child_fields: Fields) -> Self {
        Self::new_with_params(children, child_fields, false)
    }

    pub fn new_root(
        children: Vec<Vec<Arc<dyn LogicalPageScheduler>>>,
        child_fields: Fields,
    ) -> Self {
        Self::new_with_params(children, child_fields, true)
    }

    pub fn new_root_decoder_ranges(&self, ranges: &[Range<u64>]) -> SimpleStructDecoder {
        let rows_to_read = ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u64>();
        SimpleStructDecoder::new(self.child_fields.clone(), rows_to_read)
    }

    pub fn new_root_decoder_indices(&self, indices: &[u64]) -> SimpleStructDecoder {
        SimpleStructDecoder::new(self.child_fields.clone(), indices.len() as u64)
    }

    pub fn schedule_ranges_u64(
        &self,
        ranges: &[Range<u64>],
        mut context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<()> {
        for range in ranges.iter().cloned() {
            let mut rows_to_read = range.end - range.start;
            trace!(
                "Scheduling struct decode of range {:?} ({} rows)",
                range,
                rows_to_read
            );

            let mut field_status =
                vec![RangeFieldWalkStatus::new_from_range(range); self.children.len()];

            // NOTE: The order in which we are scheduling tasks here is very important.  We want to schedule the I/O so that
            // we can deliver completed rows as quickly as possible to the decoder.  This means we want to schedule in row-major
            // order from start to the end.  E.g. if we schedule one column at a time then the decoder is going to have to wait
            // until almost all the I/O is finished before it can return a single batch.
            //
            // Luckily, we can do this using a simple greedy algorithm.  We iterate through each column independently.  For each
            // pass through the metadata we look for any column that doesn't have any "queued rows".  Once we find it we schedule
            // the next page for that column and increase its queued rows.  After each pass we should have some data queued for
            // each column.  We take the column with the least amount of queued data and decrement that amount from the queued
            // rows total of all columns.

            // As we schedule, we create decoders.  These decoders are immediately sent to the decode thread
            // to allow decoding to start.

            // TODO: Instead of advancing one page at a time on each column we could make this algorithm aware of the
            // batch size.  Then we would advance a column until it has enough rows to fill the next batch.  This would
            // mainly be useful in cases like "take from fixed-size-list<struct<...>>" since the take from fsl becomes a
            // schedule_ranges against the struct with many tiny ranges and then we end up converting each range into a single
            // batch of I/O with the current algorithm.
            //
            // The downside of the current algorithm is that many tiny I/O batches means less opportunity for in-batch coalescing.
            // Then again, if our outer batch coalescing is super good then maybe we don't bother

            let mut current_top_level_row = top_level_row;

            while rows_to_read > 0 {
                let mut min_rows_added = u32::MAX;
                for (col_idx, field_scheduler) in self.children.iter().enumerate() {
                    let status = &mut field_status[col_idx];
                    if status.rows_queued == 0 {
                        trace!("Need additional rows for column {}", col_idx);
                        let mut next_page = &field_scheduler[status.page_offset as usize];

                        while status.rows_to_skip >= next_page.num_rows() as u64 {
                            status.rows_to_skip -= next_page.num_rows() as u64;
                            status.page_offset += 1;
                            trace!("Skipping entire page of {} rows", next_page.num_rows());
                            next_page = &field_scheduler[status.page_offset as usize];
                        }

                        debug_assert!(status.rows_to_skip < u32::MAX as u64);

                        let page_range_start = status.rows_to_skip as u32;
                        let page_rows_remaining = next_page.num_rows() - page_range_start;
                        let rows_to_take =
                            status.rows_to_take.min(page_rows_remaining as u64) as u32;
                        let page_range = page_range_start..(page_range_start + rows_to_take);

                        trace!(
                            "Taking {} rows from column {} starting at page offset {}",
                            rows_to_take,
                            col_idx,
                            page_range_start,
                        );
                        let scope = context.push(self.child_fields[col_idx].name(), col_idx as u32);
                        next_page.schedule_ranges(
                            &[page_range],
                            scope.context,
                            current_top_level_row,
                        )?;
                        context = scope.pop();

                        status.rows_queued += rows_to_take;
                        status.rows_to_take -= rows_to_take as u64;
                        status.page_offset += 1;
                        status.rows_to_skip = 0;

                        min_rows_added = min_rows_added.min(rows_to_take);
                    } else {
                        trace!(
                            "Using {} queued rows for column {}",
                            col_idx,
                            status.rows_queued
                        );
                        min_rows_added = min_rows_added.min(status.rows_queued);
                    }
                }
                if min_rows_added == 0 {
                    panic!("Error in scheduling logic, panic to avoid infinite loop");
                }
                rows_to_read -= min_rows_added as u64;
                current_top_level_row += min_rows_added as u64;
                if self.is_root {
                    trace!(
                        "Scheduler scan complete ({} rows now scheduled)",
                        current_top_level_row - top_level_row
                    );
                    context
                        .sink
                        .send(crate::decoder::DecoderMessage::ScanLine(
                            current_top_level_row - top_level_row,
                        ))
                        .unwrap();
                }
                for field_status in &mut field_status {
                    field_status.rows_queued -= min_rows_added;
                }
            }
        }
        Ok(())
    }

    pub fn schedule_take_u64(
        &self,
        indices: &[u64],
        mut context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<()> {
        trace!(
            "Scheduling struct decode of {} indices with top_level_row={}",
            indices.len(),
            top_level_row
        );

        // Create a cursor into indices for each column
        let mut field_status =
            vec![TakeFieldWalkStatus::new_from_indices(indices); self.children.len()];
        let mut rows_to_read = indices.len() as u32;

        // NOTE: See schedule_range for a description of the scheduling algorithm
        let mut current_top_level_row = top_level_row;
        while rows_to_read > 0 {
            trace!("Beginning scheduler scan of columns");
            let mut min_rows_added = u32::MAX;
            for (col_idx, field_scheduler) in self.children.iter().enumerate() {
                let status = &mut field_status[col_idx];
                if status.rows_queued == 0 {
                    trace!("Need additional rows for column {}", col_idx);
                    let mut indices_in_page = Vec::new();
                    let mut next_page = None;
                    // Loop through the pages in this column until we find one with overlapping indices
                    while indices_in_page.is_empty() {
                        let next_candidate_page = &field_scheduler[status.page_offset as usize];
                        indices_in_page = status.advance_page(next_candidate_page.num_rows());
                        trace!(
                            "{}",
                            if indices_in_page.is_empty() {
                                format!(
                                    "Skipping entire page of {} rows for column {}",
                                    next_candidate_page.num_rows(),
                                    col_idx,
                                )
                            } else {
                                format!(
                                    "Found page for column {} with {} rows that had {} overlapping indices",
                                    col_idx,
                                    next_candidate_page.num_rows(),
                                    indices_in_page.len()
                                )
                            }
                        );
                        next_page = Some(next_candidate_page);
                    }

                    // We should be guaranteed to get at least one page
                    let next_page = next_page.unwrap();

                    let scope = context.push(self.child_fields[col_idx].name(), col_idx as u32);
                    next_page.schedule_take(
                        &indices_in_page,
                        scope.context,
                        current_top_level_row,
                    )?;
                    context = scope.pop();

                    let rows_scheduled = indices_in_page.len() as u32;
                    status.rows_queued += rows_scheduled;

                    min_rows_added = min_rows_added.min(rows_scheduled);
                } else {
                    // TODO: Unit tests are not covering this path right now
                    trace!(
                        "Using {} already queued rows for column {}",
                        status.rows_queued,
                        col_idx
                    );
                    min_rows_added = min_rows_added.min(status.rows_queued);
                }
            }
            if min_rows_added == 0 {
                panic!("Error in scheduling logic, panic to avoid infinite loop");
            }
            trace!(
                "One scheduling pass complete, {} rows added",
                min_rows_added
            );
            rows_to_read -= min_rows_added;
            current_top_level_row += min_rows_added as u64;
            if self.is_root {
                trace!(
                    "Scheduler scan complete ({} rows now scheduled)",
                    current_top_level_row - top_level_row
                );
                context
                    .sink
                    .send(crate::decoder::DecoderMessage::ScanLine(
                        current_top_level_row - top_level_row,
                    ))
                    .unwrap();
            }
            for field_status in &mut field_status {
                field_status.rows_queued -= min_rows_added;
            }
        }
        Ok(())
    }
}

// As we schedule a range we keep one of these per column so that we know
// how far into the column we have already scheduled.
#[derive(Debug, Clone, Copy)]
struct RangeFieldWalkStatus {
    rows_to_skip: u64,
    rows_to_take: u64,
    page_offset: u32,
    rows_queued: u32,
}

impl RangeFieldWalkStatus {
    fn new_from_range(range: Range<u64>) -> Self {
        Self {
            rows_to_skip: range.start,
            rows_to_take: range.end - range.start,
            page_offset: 0,
            rows_queued: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct TakeFieldWalkStatus<'a> {
    indices: &'a [u64],
    indices_index: usize,
    page_offset: u32,
    rows_queued: u32,
    rows_passed: u64,
}

impl<'a> TakeFieldWalkStatus<'a> {
    fn new_from_indices(indices: &'a [u64]) -> Self {
        Self {
            indices,
            indices_index: 0,
            page_offset: 0,
            rows_queued: 0,
            rows_passed: 0,
        }
    }

    // If the next page has `rows_in_page` rows then return the indices that would be included
    // in that page (the returned indices are relative to the start of the page)
    fn advance_page(&mut self, rows_in_page: u32) -> Vec<u32> {
        let mut indices = Vec::new();
        while self.indices_index < self.indices.len()
            && (self.indices[self.indices_index] - self.rows_passed) < rows_in_page as u64
        {
            indices.push((self.indices[self.indices_index] - self.rows_passed) as u32);
            self.indices_index += 1;
        }
        self.rows_passed += rows_in_page as u64;
        self.page_offset += 1;
        indices
    }
}

impl LogicalPageScheduler for SimpleStructScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[Range<u32>],
        context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<()> {
        // Send info to the decoder thread so it knows a struct is here.  In the future we will also
        // send validity info here.
        //
        // Note: the root decoder is treated differently which is why this is here and not in schedule_ranges_u64
        context.emit(Box::new(SimpleStructDecoder::new(
            self.child_fields.clone(),
            ranges
                .iter()
                .map(|range| range.end as u64 - range.start as u64)
                .sum(),
        )));

        let ranges = ranges
            .iter()
            .map(|range| range.start as u64..range.end as u64)
            .collect::<Vec<_>>();
        self.schedule_ranges_u64(&ranges, context, top_level_row)
    }

    fn num_rows(&self) -> u32 {
        if self.num_rows > u32::MAX as u64 {
            // Not great, but works since we never call num_rows on the top-level scheduler
            unreachable!("Call to num_rows on a top-level SimpleStructScheduler with more than u32::MAX rows")
        }
        self.num_rows as u32
    }

    fn schedule_take(
        &self,
        indices: &[u32],
        context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<()> {
        // Send info to the decoder thread so it knows a struct is here.  In the future we will also
        // send validity info here.
        //
        // Note: the root decoder is treated differently which is why this is here and not in schedule_ranges_u64
        context.emit(Box::new(SimpleStructDecoder::new(
            self.child_fields.clone(),
            indices.len() as u64,
        )));

        let indices = indices.iter().map(|idx| *idx as u64).collect::<Vec<_>>();
        self.schedule_take_u64(&indices, context, top_level_row)
    }
}

#[derive(Debug)]
struct ChildState {
    // As child decoders are scheduled they are added to this queue
    // Once the decoder is fully drained it is popped from this queue
    //
    // TODO: It may be a minor perf optimization, in some rare cases, if we have a separate
    // "fully awaited but not yet drained" queue so we don't loop through fully awaited pages
    // during each call to wait.
    //
    // Note: This queue may have more than one page in it if the batch size is very large
    // or pages are very small
    // TODO: Test this case
    scheduled: VecDeque<Box<dyn LogicalPageDecoder>>,
    // Rows that should still be coming over the channel source but haven't yet been
    // put into the awaited queue
    rows_unawaited: u64,
    // Rows that have been pulled out of the channel source, awaited, and are ready to
    // be drained
    rows_available: u64,
    // The field index in the struct (used for debugging / logging)
    field_index: u32,
}

struct CompositeDecodeTask {
    // One per child
    tasks: Vec<Box<dyn DecodeArrayTask>>,
    num_rows: u32,
    has_more: bool,
}

impl CompositeDecodeTask {
    fn decode(self) -> Result<ArrayRef> {
        let arrays = self
            .tasks
            .into_iter()
            .map(|task| task.decode())
            .collect::<Result<Vec<_>>>()?;
        let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
        // TODO: If this is a primitive column we should be able to avoid this
        // allocation + copy with "page bridging" which could save us a few CPU
        // cycles.
        //
        // This optimization is probably most important for super fast storage like NVME
        // where the page size can be smaller.
        Ok(arrow_select::concat::concat(&array_refs)?)
    }
}

impl ChildState {
    fn new(num_rows: u64, field_index: u32) -> Self {
        Self {
            scheduled: VecDeque::new(),
            rows_unawaited: num_rows,
            rows_available: 0,
            field_index,
        }
    }

    // Wait for the next set of rows to arrive.
    async fn wait(&mut self, num_rows: u64) -> Result<()> {
        trace!(
            "Struct child {} waiting for {} rows and {} are available already",
            self.field_index,
            num_rows,
            self.rows_available
        );
        let mut remaining = num_rows.saturating_sub(self.rows_available);
        for next_decoder in &mut self.scheduled {
            if next_decoder.unawaited() > 0 {
                let rows_to_wait = remaining.min(next_decoder.unawaited() as u64) as u32;
                trace!(
                    "Struct await an additional {} rows from the current page",
                    rows_to_wait
                );
                // Even though we wait for X rows we might actually end up
                // loading more than that
                let previously_avail = next_decoder.avail();
                // We might only await part of a page.  This is important for things
                // like the struct<struct<...>> case where we have one outer page, one
                // middle page, and then a bunch of inner pages.  If we await the entire
                // middle page then we will have to wait for all the inner pages to arrive
                // before we can start decoding.
                next_decoder.wait(rows_to_wait).await?;
                let newly_avail = next_decoder.avail() - previously_avail;
                trace!("The await loaded {} rows", newly_avail);
                self.rows_available += newly_avail as u64;
                // Need to use saturating_sub here because we might have asked for range
                // 0-1000 and this page we just loaded might cover 900-1100 and so newly_avail
                // is 200 but rows_unawaited is only 100
                //
                // TODO: Unit tests may not be covering this branch right now
                self.rows_unawaited = self.rows_unawaited.saturating_sub(newly_avail as u64);
                remaining -= rows_to_wait as u64;
                if remaining == 0 {
                    break;
                }
            }
        }
        if remaining > 0 {
            Err(Error::Internal { message: format!("The struct field at index {} is still waiting for {} rows but ran out of scheduled pages", self.field_index, remaining), location: location!() })
        } else {
            Ok(())
        }
    }

    fn drain(&mut self, num_rows: u64) -> Result<CompositeDecodeTask> {
        trace!("Struct draining {} rows", num_rows);
        debug_assert!(self.rows_available >= num_rows);

        self.rows_available -= num_rows;
        let mut remaining = num_rows;
        let mut composite = CompositeDecodeTask {
            tasks: Vec::new(),
            num_rows: 0,
            has_more: true,
        };
        while remaining > 0 {
            let next = self.scheduled.front_mut().unwrap();
            let rows_to_take = remaining.min(next.avail() as u64) as u32;
            let next_task = next.drain(rows_to_take)?;
            if next.avail() == 0 && next.unawaited() == 0 {
                trace!("Completely drained page");
                self.scheduled.pop_front();
            }
            remaining -= rows_to_take as u64;
            composite.tasks.push(next_task.task);
            composite.num_rows += next_task.num_rows;
        }
        composite.has_more = self.rows_available != 0 || self.rows_unawaited != 0;
        Ok(composite)
    }
}

#[derive(Debug)]
pub struct SimpleStructDecoder {
    children: Vec<ChildState>,
    child_fields: Fields,
    data_type: DataType,
}

impl SimpleStructDecoder {
    fn new(child_fields: Fields, num_rows: u64) -> Self {
        let data_type = DataType::Struct(child_fields.clone());
        Self {
            children: child_fields
                .iter()
                .enumerate()
                .map(|(idx, _)| ChildState::new(num_rows, idx as u32))
                .collect(),
            child_fields,
            data_type,
        }
    }

    pub fn avail_u64(&self) -> u64 {
        self.children
            .iter()
            .map(|c| c.rows_available)
            .min()
            .unwrap()
    }

    // Rows are unawaited if they are unawaited in any child column
    pub fn unawaited_u64(&self) -> u64 {
        self.children
            .iter()
            .map(|c| c.rows_unawaited)
            .max()
            .unwrap()
    }

    pub fn wait_u64(&mut self, num_rows: u64) -> BoxFuture<Result<()>> {
        async move {
            for child in self.children.iter_mut() {
                child.wait(num_rows).await?;
            }
            Ok(())
        }
        .boxed()
    }

    pub fn drain_u64(&mut self, num_rows: u64) -> Result<NextDecodeTask> {
        let child_tasks = self
            .children
            .iter_mut()
            .map(|child| child.drain(num_rows))
            .collect::<Result<Vec<_>>>()?;
        let num_rows = child_tasks[0].num_rows;
        let has_more = child_tasks[0].has_more;
        debug_assert!(child_tasks.iter().all(|task| task.num_rows == num_rows));
        debug_assert!(child_tasks.iter().all(|task| task.has_more == has_more));
        Ok(NextDecodeTask {
            task: Box::new(SimpleStructDecodeTask {
                children: child_tasks,
                child_fields: self.child_fields.clone(),
            }),
            num_rows,
            has_more,
        })
    }
}

impl LogicalPageDecoder for SimpleStructDecoder {
    fn accept_child(&mut self, mut child: DecoderReady) -> Result<()> {
        // children with empty path should not be delivered to this method
        let child_idx = child.path.pop_front().unwrap();
        if child.path.is_empty() {
            // This decoder is intended for us
            self.children[child_idx as usize]
                .scheduled
                .push_back(child.decoder);
        } else {
            // This decoder is intended for one of our children
            let intended = self.children[child_idx as usize].scheduled.back_mut().ok_or_else(|| Error::Internal { message: format!("Decoder scheduled for child at index {} but we don't have any child at that index yet", child_idx), location: location!() })?;
            intended.accept_child(child)?;
        }
        Ok(())
    }

    fn wait(&mut self, num_rows: u32) -> BoxFuture<Result<()>> {
        async move {
            for child in self.children.iter_mut() {
                child.wait(num_rows as u64).await?;
            }
            Ok(())
        }
        .boxed()
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        let child_tasks = self
            .children
            .iter_mut()
            .map(|child| child.drain(num_rows as u64))
            .collect::<Result<Vec<_>>>()?;
        let num_rows = child_tasks[0].num_rows;
        let has_more = child_tasks[0].has_more;
        debug_assert!(child_tasks.iter().all(|task| task.num_rows == num_rows));
        debug_assert!(child_tasks.iter().all(|task| task.has_more == has_more));
        Ok(NextDecodeTask {
            task: Box::new(SimpleStructDecodeTask {
                children: child_tasks,
                child_fields: self.child_fields.clone(),
            }),
            num_rows,
            has_more,
        })
    }

    // Rows are available only if they are available in every child column
    fn avail(&self) -> u32 {
        let avail = self.avail_u64();
        debug_assert!(avail <= u32::MAX as u64);
        avail as u32
    }

    // Rows are unawaited if they are unawaited in any child column
    fn unawaited(&self) -> u32 {
        let unawaited = self.unawaited_u64();
        debug_assert!(unawaited <= u32::MAX as u64);
        unawaited as u32
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

struct SimpleStructDecodeTask {
    children: Vec<CompositeDecodeTask>,
    child_fields: Fields,
}

impl DecodeArrayTask for SimpleStructDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let child_arrays = self
            .children
            .into_iter()
            .map(|child| child.decode())
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(StructArray::try_new(
            self.child_fields,
            child_arrays,
            None,
        )?))
    }
}

pub struct StructFieldEncoder {
    children: Vec<Box<dyn FieldEncoder>>,
    column_index: u32,
    num_rows_seen: u32,
}

impl StructFieldEncoder {
    #[allow(dead_code)]
    pub fn new(children: Vec<Box<dyn FieldEncoder>>, column_index: u32) -> Self {
        Self {
            children,
            column_index,
            num_rows_seen: 0,
        }
    }
}

impl FieldEncoder for StructFieldEncoder {
    fn maybe_encode(&mut self, array: ArrayRef) -> Result<Vec<EncodeTask>> {
        self.num_rows_seen += array.len() as u32;
        let struct_array = array.as_struct();
        let child_tasks = self
            .children
            .iter_mut()
            .zip(struct_array.columns().iter())
            .map(|(encoder, arr)| encoder.maybe_encode(arr.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(child_tasks.into_iter().flatten().collect::<Vec<_>>())
    }

    fn flush(&mut self) -> Result<Vec<EncodeTask>> {
        let child_tasks = self
            .children
            .iter_mut()
            .map(|encoder| encoder.flush())
            .collect::<Result<Vec<_>>>()?;
        let mut child_tasks = child_tasks.into_iter().flatten().collect::<Vec<_>>();
        let num_rows_seen = self.num_rows_seen;
        let column_index = self.column_index;
        // In this "simple struct / no nulls" case we emit a single header page at
        // the very end which covers the entire struct.
        child_tasks.push(
            std::future::ready(Ok(EncodedPage {
                array: EncodedArray {
                    buffers: vec![],
                    encoding: pb::ArrayEncoding {
                        array_encoding: Some(pb::array_encoding::ArrayEncoding::Struct(
                            pb::SimpleStruct {},
                        )),
                    },
                },
                num_rows: num_rows_seen,
                column_idx: column_index,
            }))
            .boxed(),
        );
        Ok(child_tasks)
    }

    fn num_columns(&self) -> u32 {
        self.children
            .iter()
            .map(|child| child.num_columns())
            .sum::<u32>()
            + 1
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_array::{
        builder::{Int32Builder, ListBuilder},
        Array, ArrayRef, Int32Array, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases,
    };

    #[test_log::test(tokio::test)]
    async fn test_simple_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let field = Field::new("", data_type, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_struct_list() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new(
                "inner_list",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
            Field::new("outer_int", DataType::Int32, true),
        ]));
        let field = Field::new("row", data_type, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_complicated_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("int", DataType::Int32, true),
            Field::new(
                "inner",
                DataType::Struct(Fields::from(vec![
                    Field::new("inner_int", DataType::Int32, true),
                    Field::new(
                        "inner_list",
                        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                        true,
                    ),
                ])),
                true,
            ),
            Field::new("outer_binary", DataType::Binary, true),
        ]));
        let field = Field::new("row", data_type, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_ragged_scheduling() {
        // This test covers scheduling when batches straddle page boundaries

        // Create a list with 10k nulls
        let items_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(items_builder);
        for _ in 0..10000 {
            list_builder.append_null();
        }
        let list_array = Arc::new(list_builder.finish());
        let int_array = Arc::new(Int32Array::from_iter_values(0..10000));
        let fields = vec![
            Field::new("", list_array.data_type().clone(), true),
            Field::new("", int_array.data_type().clone(), true),
        ];
        let struct_array = Arc::new(StructArray::new(
            Fields::from(fields),
            vec![list_array, int_array],
            None,
        )) as ArrayRef;
        let struct_arrays = (0..10000)
            // Intentionally skip in some randomish amount to create more ragged scheduling
            .step_by(437)
            .map(|offset| struct_array.slice(offset, 437.min(10000 - offset)))
            .collect::<Vec<_>>();
        check_round_trip_encoding_of_data(struct_arrays, &TestCases::default()).await;
    }
}
