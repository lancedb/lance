// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BinaryHeap, VecDeque},
    ops::Range,
    sync::Arc,
};

use arrow_array::{cast::AsArray, Array, ArrayRef, StructArray};
use arrow_schema::{DataType, Fields};
use futures::{
    future::BoxFuture,
    stream::{FuturesOrdered, FuturesUnordered},
    FutureExt, StreamExt, TryStreamExt,
};
use itertools::Itertools;
use log::trace;
use snafu::{location, Location};

use crate::{
    decoder::{
        DecodeArrayTask, DecodedArray, DecoderReady, FieldScheduler, FilterExpression, LoadedPage,
        LogicalPageDecoder, MessageType, NextDecodeTask, PageEncoding, PriorityRange,
        ScheduledScanLine, SchedulerContext, SchedulingJob, StructuralDecodeArrayTask,
        StructuralFieldDecoder, StructuralFieldScheduler, StructuralSchedulingJob,
    },
    encoder::{EncodeTask, EncodedColumn, EncodedPage, FieldEncoder, OutOfLineBuffers},
    format::pb,
    repdef::RepDefBuilder,
};
use lance_core::{Error, Result};

use super::primitive::StructuralPrimitiveFieldDecoder;

#[derive(Debug)]
struct SchedulingJobWithStatus<'a> {
    col_idx: u32,
    col_name: &'a str,
    job: Box<dyn SchedulingJob + 'a>,
    rows_scheduled: u64,
    rows_remaining: u64,
}

impl<'a> PartialEq for SchedulingJobWithStatus<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.col_idx == other.col_idx
    }
}

impl<'a> Eq for SchedulingJobWithStatus<'a> {}

impl<'a> PartialOrd for SchedulingJobWithStatus<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for SchedulingJobWithStatus<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Note this is reversed to make it min-heap
        other.rows_scheduled.cmp(&self.rows_scheduled)
    }
}

/// Scheduling job for struct data
///
/// The order in which we schedule the children is important.  We want to schedule the child
/// with the least amount of data first.
///
/// This allows us to decode entire rows as quickly as possible
#[derive(Debug)]
struct SimpleStructSchedulerJob<'a> {
    scheduler: &'a SimpleStructScheduler,
    /// A min-heap whose key is the # of rows currently scheduled
    children: BinaryHeap<SchedulingJobWithStatus<'a>>,
    rows_scheduled: u64,
    num_rows: u64,
    initialized: bool,
}

impl<'a> SimpleStructSchedulerJob<'a> {
    fn new(
        scheduler: &'a SimpleStructScheduler,
        children: Vec<Box<dyn SchedulingJob + 'a>>,
        num_rows: u64,
    ) -> Self {
        let children = children
            .into_iter()
            .enumerate()
            .map(|(idx, job)| SchedulingJobWithStatus {
                col_idx: idx as u32,
                col_name: scheduler.child_fields[idx].name(),
                job,
                rows_scheduled: 0,
                rows_remaining: num_rows,
            })
            .collect::<BinaryHeap<_>>();
        Self {
            scheduler,
            children,
            rows_scheduled: 0,
            num_rows,
            initialized: false,
        }
    }
}

impl<'a> SchedulingJob for SimpleStructSchedulerJob<'a> {
    fn schedule_next(
        &mut self,
        mut context: &mut SchedulerContext,
        priority: &dyn PriorityRange,
    ) -> Result<ScheduledScanLine> {
        let mut decoders = Vec::new();
        if !self.initialized {
            // Send info to the decoder thread so it knows a struct is here.  In the future we will also
            // send validity info here.
            let struct_decoder = Box::new(SimpleStructDecoder::new(
                self.scheduler.child_fields.clone(),
                self.num_rows,
            ));
            let struct_decoder = context.locate_decoder(struct_decoder);
            decoders.push(MessageType::DecoderReady(struct_decoder));
            self.initialized = true;
        }
        let old_rows_scheduled = self.rows_scheduled;
        // Schedule as many children as we need to until we have scheduled at least one
        // complete row
        while old_rows_scheduled == self.rows_scheduled {
            let mut next_child = self.children.pop().unwrap();
            trace!("Scheduling more rows for child {}", next_child.col_idx);
            let scoped = context.push(next_child.col_name, next_child.col_idx);
            let child_scan = next_child.job.schedule_next(scoped.context, priority)?;
            trace!(
                "Scheduled {} rows for child {}",
                child_scan.rows_scheduled,
                next_child.col_idx
            );
            next_child.rows_scheduled += child_scan.rows_scheduled;
            next_child.rows_remaining -= child_scan.rows_scheduled;
            decoders.extend(child_scan.decoders);
            self.children.push(next_child);
            self.rows_scheduled = self.children.peek().unwrap().rows_scheduled;
            context = scoped.pop();
        }
        let struct_rows_scheduled = self.rows_scheduled - old_rows_scheduled;
        Ok(ScheduledScanLine {
            decoders,
            rows_scheduled: struct_rows_scheduled,
        })
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// A scheduler for structs
///
/// The implementation is actually a bit more tricky than one might initially think.  We can't just
/// go through and schedule each column one after the other.  This would mean our decode can't start
/// until nearly all the data has arrived (since we need data from each column)
///
/// Instead, we schedule in row-major fashion
///
/// Note: this scheduler is the starting point for all decoding.  This is because we treat the top-level
/// record batch as a non-nullable struct.
#[derive(Debug)]
pub struct SimpleStructScheduler {
    children: Vec<Arc<dyn FieldScheduler>>,
    child_fields: Fields,
    num_rows: u64,
}

impl SimpleStructScheduler {
    pub fn new(children: Vec<Arc<dyn FieldScheduler>>, child_fields: Fields) -> Self {
        debug_assert!(!children.is_empty());
        let num_rows = children[0].num_rows();
        debug_assert!(children.iter().all(|child| child.num_rows() == num_rows));
        Self {
            children,
            child_fields,
            num_rows,
        }
    }
}

impl FieldScheduler for SimpleStructScheduler {
    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[Range<u64>],
        filter: &FilterExpression,
    ) -> Result<Box<dyn SchedulingJob + 'a>> {
        let child_schedulers = self
            .children
            .iter()
            .map(|child| child.schedule_ranges(ranges, filter))
            .collect::<Result<Vec<_>>>()?;
        let num_rows = child_schedulers[0].num_rows();
        Ok(Box::new(SimpleStructSchedulerJob::new(
            self,
            child_schedulers,
            num_rows,
        )))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }

    fn initialize<'a>(
        &'a self,
        _filter: &'a FilterExpression,
        _context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        let futures = self
            .children
            .iter()
            .map(|child| child.initialize(_filter, _context))
            .collect::<FuturesUnordered<_>>();
        async move {
            futures
                .map(|res| res.map(|_| ()))
                .try_collect::<Vec<_>>()
                .await?;
            Ok(())
        }
        .boxed()
    }
}

#[derive(Debug)]
struct StructuralSchedulingJobWithStatus<'a> {
    col_idx: u32,
    col_name: &'a str,
    job: Box<dyn StructuralSchedulingJob + 'a>,
    rows_scheduled: u64,
    rows_remaining: u64,
}

impl<'a> PartialEq for StructuralSchedulingJobWithStatus<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.col_idx == other.col_idx
    }
}

impl<'a> Eq for StructuralSchedulingJobWithStatus<'a> {}

impl<'a> PartialOrd for StructuralSchedulingJobWithStatus<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for StructuralSchedulingJobWithStatus<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Note this is reversed to make it min-heap
        other.rows_scheduled.cmp(&self.rows_scheduled)
    }
}

/// Scheduling job for struct data
///
/// The order in which we schedule the children is important.  We want to schedule the child
/// with the least amount of data first.
///
/// This allows us to decode entire rows as quickly as possible
#[derive(Debug)]
struct RepDefStructSchedulingJob<'a> {
    /// A min-heap whose key is the # of rows currently scheduled
    children: BinaryHeap<StructuralSchedulingJobWithStatus<'a>>,
    rows_scheduled: u64,
}

impl<'a> RepDefStructSchedulingJob<'a> {
    fn new(
        scheduler: &'a StructuralStructScheduler,
        children: Vec<Box<dyn StructuralSchedulingJob + 'a>>,
        num_rows: u64,
    ) -> Self {
        let children = children
            .into_iter()
            .enumerate()
            .map(|(idx, job)| StructuralSchedulingJobWithStatus {
                col_idx: idx as u32,
                col_name: scheduler.child_fields[idx].name(),
                job,
                rows_scheduled: 0,
                rows_remaining: num_rows,
            })
            .collect::<BinaryHeap<_>>();
        Self {
            children,
            rows_scheduled: 0,
        }
    }
}

impl<'a> StructuralSchedulingJob for RepDefStructSchedulingJob<'a> {
    fn schedule_next(
        &mut self,
        mut context: &mut SchedulerContext,
    ) -> Result<Option<ScheduledScanLine>> {
        let mut decoders = Vec::new();
        let old_rows_scheduled = self.rows_scheduled;
        // Schedule as many children as we need to until we have scheduled at least one
        // complete row
        while old_rows_scheduled == self.rows_scheduled {
            let mut next_child = self.children.pop().unwrap();
            let scoped = context.push(next_child.col_name, next_child.col_idx);
            let child_scan = next_child.job.schedule_next(scoped.context)?;
            // next_child is the least-scheduled child and, if it's done, that
            // means we are completely done.
            if child_scan.is_none() {
                return Ok(None);
            }
            let child_scan = child_scan.unwrap();

            trace!(
                "Scheduled {} rows for child {}",
                child_scan.rows_scheduled,
                next_child.col_idx
            );
            next_child.rows_scheduled += child_scan.rows_scheduled;
            next_child.rows_remaining -= child_scan.rows_scheduled;
            decoders.extend(child_scan.decoders);
            self.children.push(next_child);
            self.rows_scheduled = self.children.peek().unwrap().rows_scheduled;
            context = scoped.pop();
        }
        let struct_rows_scheduled = self.rows_scheduled - old_rows_scheduled;
        Ok(Some(ScheduledScanLine {
            decoders,
            rows_scheduled: struct_rows_scheduled,
        }))
    }
}

/// A scheduler for structs
///
/// The implementation is actually a bit more tricky than one might initially think.  We can't just
/// go through and schedule each column one after the other.  This would mean our decode can't start
/// until nearly all the data has arrived (since we need data from each column to yield a batch)
///
/// Instead, we schedule in row-major fashion
///
/// Note: this scheduler is the starting point for all decoding.  This is because we treat the top-level
/// record batch as a non-nullable struct.
#[derive(Debug)]
pub struct StructuralStructScheduler {
    children: Vec<Box<dyn StructuralFieldScheduler>>,
    child_fields: Fields,
}

impl StructuralStructScheduler {
    pub fn new(children: Vec<Box<dyn StructuralFieldScheduler>>, child_fields: Fields) -> Self {
        debug_assert!(!children.is_empty());
        Self {
            children,
            child_fields,
        }
    }
}

impl StructuralFieldScheduler for StructuralStructScheduler {
    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[Range<u64>],
        filter: &FilterExpression,
    ) -> Result<Box<dyn StructuralSchedulingJob + 'a>> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();

        let child_schedulers = self
            .children
            .iter()
            .map(|child| child.schedule_ranges(ranges, filter))
            .collect::<Result<Vec<_>>>()?;

        Ok(Box::new(RepDefStructSchedulingJob::new(
            self,
            child_schedulers,
            num_rows,
        )))
    }

    fn initialize<'a>(
        &'a mut self,
        filter: &'a FilterExpression,
        context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        let children_initialization = self
            .children
            .iter_mut()
            .map(|child| child.initialize(filter, context))
            .collect::<FuturesUnordered<_>>();
        async move {
            children_initialization
                .map(|res| res.map(|_| ()))
                .try_collect::<Vec<_>>()
                .await?;
            Ok(())
        }
        .boxed()
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
    // Rows that have been awaited
    rows_loaded: u64,
    // Rows that have drained
    rows_drained: u64,
    // Rows that have been popped (the decoder has been completely drained and removed from `scheduled`)
    rows_popped: u64,
    // Total number of rows in the struct
    num_rows: u64,
    // The field index in the struct (used for debugging / logging)
    field_index: u32,
}

struct CompositeDecodeTask {
    // One per child
    tasks: Vec<Box<dyn DecodeArrayTask>>,
    num_rows: u64,
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
            rows_loaded: 0,
            rows_drained: 0,
            rows_popped: 0,
            num_rows,
            field_index,
        }
    }

    // Wait for the next set of rows to arrive
    //
    // Wait until we have at least `loaded_need` loaded and stop as soon as we
    // go above that limit.
    async fn wait_for_loaded(&mut self, loaded_need: u64) -> Result<()> {
        trace!(
            "Struct child {} waiting for more than {} rows to be loaded and {} are fully loaded already",
            self.field_index,
            loaded_need,
            self.rows_loaded,
        );
        let mut fully_loaded = self.rows_popped;
        for (page_idx, next_decoder) in self.scheduled.iter_mut().enumerate() {
            if next_decoder.rows_unloaded() > 0 {
                let mut current_need = loaded_need;
                current_need -= fully_loaded;
                let rows_in_page = next_decoder.num_rows();
                let need_for_page = (rows_in_page - 1).min(current_need);
                trace!(
                    "Struct child {} page {} will wait until more than {} rows loaded from page with {} rows",
                    self.field_index,
                    page_idx,
                    need_for_page,
                    rows_in_page,
                );
                // We might only await part of a page.  This is important for things
                // like the struct<struct<...>> case where we have one outer page, one
                // middle page, and then a bunch of inner pages.  If we await the entire
                // middle page then we will have to wait for all the inner pages to arrive
                // before we can start decoding.
                next_decoder.wait_for_loaded(need_for_page).await?;
                let now_loaded = next_decoder.rows_loaded();
                fully_loaded += now_loaded;
                trace!(
                    "Struct child {} page {} await and now has {} loaded rows and we have {} fully loaded",
                    self.field_index,
                    page_idx,
                    now_loaded,
                    fully_loaded
                );
            } else {
                fully_loaded += next_decoder.num_rows();
            }
            if fully_loaded > loaded_need {
                break;
            }
        }
        self.rows_loaded = fully_loaded;
        trace!(
            "Struct child {} loaded {} new rows and now {} are loaded",
            self.field_index,
            fully_loaded,
            self.rows_loaded
        );
        Ok(())
    }

    fn drain(&mut self, num_rows: u64) -> Result<CompositeDecodeTask> {
        trace!("Struct draining {} rows", num_rows);

        trace!(
            "Draining {} rows from struct page with {} rows already drained",
            num_rows,
            self.rows_drained
        );
        let mut remaining = num_rows;
        let mut composite = CompositeDecodeTask {
            tasks: Vec::new(),
            num_rows: 0,
            has_more: true,
        };
        while remaining > 0 {
            let next = self.scheduled.front_mut().unwrap();
            let rows_to_take = remaining.min(next.rows_left());
            let next_task = next.drain(rows_to_take)?;
            if next.rows_left() == 0 {
                trace!("Completely drained page");
                self.rows_popped += next.num_rows();
                self.scheduled.pop_front();
            }
            remaining -= rows_to_take;
            composite.tasks.push(next_task.task);
            composite.num_rows += next_task.num_rows;
        }
        self.rows_drained += num_rows;
        composite.has_more = self.rows_drained != self.num_rows;
        Ok(composite)
    }
}

// Wrapper around ChildState that orders using rows_unawaited
struct WaitOrder<'a>(&'a mut ChildState);

impl Eq for WaitOrder<'_> {}
impl PartialEq for WaitOrder<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.rows_loaded == other.0.rows_loaded
    }
}
impl Ord for WaitOrder<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Note: this is inverted so we have a min-heap
        other.0.rows_loaded.cmp(&self.0.rows_loaded)
    }
}
impl PartialOrd for WaitOrder<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct StructuralStructDecoder {
    children: Vec<Box<dyn StructuralFieldDecoder>>,
    data_type: DataType,
    child_fields: Fields,
}

impl StructuralStructDecoder {
    pub fn new(fields: Fields, should_validate: bool) -> Self {
        let children = fields
            .iter()
            .map(|field| Self::field_to_decoder(field, should_validate))
            .collect();
        let data_type = DataType::Struct(fields.clone());
        Self {
            data_type,
            children,
            child_fields: fields,
        }
    }

    fn field_to_decoder(
        field: &Arc<arrow_schema::Field>,
        should_validate: bool,
    ) -> Box<dyn StructuralFieldDecoder> {
        match field.data_type() {
            DataType::Struct(fields) => Box::new(Self::new(fields.clone(), should_validate)),
            DataType::List(_) | DataType::LargeList(_) => todo!(),
            DataType::RunEndEncoded(_, _) => todo!(),
            DataType::ListView(_) | DataType::LargeListView(_) => todo!(),
            DataType::Map(_, _) => todo!(),
            DataType::Union(_, _) => todo!(),
            _ => Box::new(StructuralPrimitiveFieldDecoder::new(field, should_validate)),
        }
    }
}

impl StructuralFieldDecoder for StructuralStructDecoder {
    fn accept_page(&mut self, mut child: LoadedPage) -> Result<()> {
        // children with empty path should not be delivered to this method
        let child_idx = child.path.pop_front().unwrap();
        // This decoder is intended for one of our children
        self.children[child_idx as usize].accept_page(child)?;
        Ok(())
    }

    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn StructuralDecodeArrayTask>> {
        let child_tasks = self
            .children
            .iter_mut()
            .map(|child| child.drain(num_rows))
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(RepDefStructDecodeTask {
            children: child_tasks,
            child_fields: self.child_fields.clone(),
        }))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

#[derive(Debug)]
struct RepDefStructDecodeTask {
    children: Vec<Box<dyn StructuralDecodeArrayTask>>,
    child_fields: Fields,
}

impl StructuralDecodeArrayTask for RepDefStructDecodeTask {
    fn decode(self: Box<Self>) -> Result<DecodedArray> {
        let arrays = self
            .children
            .into_iter()
            .map(|task| task.decode())
            .collect::<Result<Vec<_>>>()?;
        let mut children = Vec::with_capacity(arrays.len());
        let mut arrays_iter = arrays.into_iter();
        let first_array = arrays_iter.next().unwrap();

        // The repdef should be identical across all children at this point
        let mut repdef = first_array.repdef;
        children.push(first_array.array);
        for array in arrays_iter {
            children.push(array.array);
        }

        let validity = repdef.unravel_validity();
        let array = StructArray::new(self.child_fields, children, validity);
        Ok(DecodedArray {
            array: Arc::new(array),
            repdef,
        })
    }
}

#[derive(Debug)]
pub struct SimpleStructDecoder {
    children: Vec<ChildState>,
    child_fields: Fields,
    data_type: DataType,
    num_rows: u64,
}

impl SimpleStructDecoder {
    pub fn new(child_fields: Fields, num_rows: u64) -> Self {
        let data_type = DataType::Struct(child_fields.clone());
        Self {
            children: child_fields
                .iter()
                .enumerate()
                .map(|(idx, _)| ChildState::new(num_rows, idx as u32))
                .collect(),
            child_fields,
            data_type,
            num_rows,
        }
    }

    async fn do_wait_for_loaded(&mut self, loaded_need: u64) -> Result<()> {
        let mut wait_orders = self
            .children
            .iter_mut()
            .filter_map(|child| {
                if child.rows_loaded <= loaded_need {
                    Some(WaitOrder(child))
                } else {
                    None
                }
            })
            .collect::<BinaryHeap<_>>();
        while !wait_orders.is_empty() {
            let next_waiter = wait_orders.pop().unwrap();
            let next_highest = wait_orders
                .peek()
                .map(|w| w.0.rows_loaded)
                .unwrap_or(u64::MAX);
            // Wait until you have the number of rows needed, or at least more than the
            // next highest waiter
            let limit = loaded_need.min(next_highest);
            next_waiter.0.wait_for_loaded(limit).await?;
            log::trace!(
                "Struct child {} finished await pass and now {} are loaded",
                next_waiter.0.field_index,
                next_waiter.0.rows_loaded
            );
            if next_waiter.0.rows_loaded <= loaded_need {
                wait_orders.push(next_waiter);
            }
        }
        Ok(())
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

    fn wait_for_loaded(&mut self, loaded_need: u64) -> BoxFuture<Result<()>> {
        self.do_wait_for_loaded(loaded_need).boxed()
    }

    fn drain(&mut self, num_rows: u64) -> Result<NextDecodeTask> {
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

    fn rows_loaded(&self) -> u64 {
        self.children.iter().map(|c| c.rows_loaded).min().unwrap()
    }

    fn rows_drained(&self) -> u64 {
        // All children should have the same number of rows drained
        debug_assert!(self
            .children
            .iter()
            .all(|c| c.rows_drained == self.children[0].rows_drained));
        self.children[0].rows_drained
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
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

/// A structural encoder for struct fields
///
/// The struct's validity is added to the rep/def builder
/// and the builder is cloned to all children.
pub struct StructStructuralEncoder {
    children: Vec<Box<dyn FieldEncoder>>,
}

impl StructStructuralEncoder {
    pub fn new(children: Vec<Box<dyn FieldEncoder>>) -> Self {
        Self { children }
    }
}

impl FieldEncoder for StructStructuralEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        mut repdef: RepDefBuilder,
        row_number: u64,
    ) -> Result<Vec<EncodeTask>> {
        let struct_array = array.as_struct();
        if let Some(validity) = struct_array.nulls() {
            repdef.add_validity_bitmap(validity.clone());
        } else {
            repdef.add_no_null(struct_array.len());
        }
        let child_tasks = self
            .children
            .iter_mut()
            .zip(struct_array.columns().iter())
            .map(|(encoder, arr)| {
                encoder.maybe_encode(arr.clone(), external_buffers, repdef.clone(), row_number)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(child_tasks.into_iter().flatten().collect::<Vec<_>>())
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        self.children
            .iter_mut()
            .map(|encoder| encoder.flush(external_buffers))
            .flatten_ok()
            .collect::<Result<Vec<_>>>()
    }

    fn num_columns(&self) -> u32 {
        self.children
            .iter()
            .map(|child| child.num_columns())
            .sum::<u32>()
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        let mut child_columns = self
            .children
            .iter_mut()
            .map(|child| child.finish(external_buffers))
            .collect::<FuturesOrdered<_>>();
        async move {
            let mut encoded_columns = Vec::with_capacity(child_columns.len());
            while let Some(child_cols) = child_columns.next().await {
                encoded_columns.extend(child_cols?);
            }
            Ok(encoded_columns)
        }
        .boxed()
    }
}

pub struct StructFieldEncoder {
    children: Vec<Box<dyn FieldEncoder>>,
    column_index: u32,
    num_rows_seen: u64,
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
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
    ) -> Result<Vec<EncodeTask>> {
        self.num_rows_seen += array.len() as u64;
        let struct_array = array.as_struct();
        let child_tasks = self
            .children
            .iter_mut()
            .zip(struct_array.columns().iter())
            .map(|(encoder, arr)| {
                encoder.maybe_encode(arr.clone(), external_buffers, repdef.clone(), row_number)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(child_tasks.into_iter().flatten().collect::<Vec<_>>())
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        let child_tasks = self
            .children
            .iter_mut()
            .map(|encoder| encoder.flush(external_buffers))
            .collect::<Result<Vec<_>>>()?;
        Ok(child_tasks.into_iter().flatten().collect::<Vec<_>>())
    }

    fn num_columns(&self) -> u32 {
        self.children
            .iter()
            .map(|child| child.num_columns())
            .sum::<u32>()
            + 1
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        let mut child_columns = self
            .children
            .iter_mut()
            .map(|child| child.finish(external_buffers))
            .collect::<FuturesOrdered<_>>();
        let num_rows_seen = self.num_rows_seen;
        let column_index = self.column_index;
        async move {
            let mut columns = Vec::new();
            // Add a column for the struct header
            let mut header = EncodedColumn::default();
            header.final_pages.push(EncodedPage {
                data: Vec::new(),
                description: PageEncoding::Legacy(pb::ArrayEncoding {
                    array_encoding: Some(pb::array_encoding::ArrayEncoding::Struct(
                        pb::SimpleStruct {},
                    )),
                }),
                num_rows: num_rows_seen,
                column_idx: column_index,
                row_number: 0, // Not used by legacy encoding
            });
            columns.push(header);
            // Now run finish on the children
            while let Some(child_cols) = child_columns.next().await {
                columns.extend(child_cols?);
            }
            Ok(columns)
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {

    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{
        builder::{Int32Builder, ListBuilder},
        Array, ArrayRef, Int32Array, StructArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Fields};

    use crate::{
        testing::{check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases},
        version::LanceFileVersion,
    };

    #[test_log::test(tokio::test)]
    async fn test_simple_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let field = Field::new("", data_type, false);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_nullable_struct() {
        // Test data struct<score: int32, location: struct<x: int32, y: int32>>
        // - score: null
        //   location:
        //     x: 1
        //     y: 6
        // - score: 12
        //   location:
        //     x: 2
        //     y: null
        // - score: 13
        //   location:
        //     x: 3
        //     y: 8
        // - score: 14
        //   location: null
        // - null
        //
        let inner_fields = Fields::from(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, true),
        ]);
        let inner_struct = DataType::Struct(inner_fields.clone());
        let outer_fields = Fields::from(vec![
            Field::new("score", DataType::Int32, true),
            Field::new("location", inner_struct, true),
        ]);

        let x_vals = Int32Array::from(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let y_vals = Int32Array::from(vec![Some(6), None, Some(8), Some(9), Some(10)]);
        let scores = Int32Array::from(vec![None, Some(12), Some(13), Some(14), Some(15)]);

        let location_validity = NullBuffer::from(vec![true, true, true, false, true]);
        let locations = StructArray::new(
            inner_fields,
            vec![Arc::new(x_vals), Arc::new(y_vals)],
            Some(location_validity),
        );

        let rows_validity = NullBuffer::from(vec![true, true, true, true, false]);
        let rows = StructArray::new(
            outer_fields,
            vec![Arc::new(scores), Arc::new(locations)],
            Some(rows_validity),
        );

        let test_cases = TestCases::default().with_file_version(LanceFileVersion::V2_1);

        check_round_trip_encoding_of_data(vec![Arc::new(rows)], &test_cases, HashMap::new()).await;
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
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
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
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
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
        check_round_trip_encoding_of_data(struct_arrays, &TestCases::default(), HashMap::new())
            .await;
    }
}
