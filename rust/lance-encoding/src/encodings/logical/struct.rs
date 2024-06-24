// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BinaryHeap, VecDeque},
    ops::Range,
    sync::Arc,
};

use arrow_array::{cast::AsArray, ArrayRef, StructArray};
use arrow_schema::{DataType, Fields};
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};

use crate::{
    decoder::{
        DecodeArrayTask, DecoderReady, FieldScheduler, FilterExpression, LogicalPageDecoder,
        NextDecodeTask, ScheduledScanLine, SchedulerContext, SchedulingJob,
    },
    encoder::{EncodeTask, EncodedArray, EncodedColumn, EncodedPage, FieldEncoder},
    format::pb,
};
use lance_core::{Error, Result};

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
        top_level_row: u64,
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
            decoders.push(struct_decoder);
            self.initialized = true;
        }
        let old_rows_scheduled = self.rows_scheduled;
        // Schedule as many children as we need to until we have scheduled at least one
        // complete row
        while old_rows_scheduled == self.rows_scheduled {
            let mut next_child = self.children.pop().unwrap();
            trace!("Scheduling more rows for child {}", next_child.col_idx);
            let scoped = context.push(next_child.col_name, next_child.col_idx);
            let child_scan = next_child
                .job
                .schedule_next(scoped.context, top_level_row)?;
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
    // A single page cannot contain more than u32 rows.  However, we also use SimpleStructScheduler
    // at the top level and a single file *can* contain more than u32 rows.
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
                let rows_to_wait = remaining.min(next_decoder.unawaited());
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
                self.rows_available += newly_avail;
                // Need to use saturating_sub here because we might have asked for range
                // 0-1000 and this page we just loaded might cover 900-1100 and so newly_avail
                // is 200 but rows_unawaited is only 100
                //
                // TODO: Unit tests may not be covering this branch right now
                self.rows_unawaited = self.rows_unawaited.saturating_sub(newly_avail);
                remaining -= rows_to_wait;
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
            let rows_to_take = remaining.min(next.avail());
            let next_task = next.drain(rows_to_take)?;
            if next.avail() == 0 && next.unawaited() == 0 {
                trace!("Completely drained page");
                self.scheduled.pop_front();
            }
            remaining -= rows_to_take;
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
        }
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

    fn wait(&mut self, num_rows: u64) -> BoxFuture<Result<()>> {
        async move {
            for child in self.children.iter_mut() {
                child.wait(num_rows).await?;
            }
            Ok(())
        }
        .boxed()
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

    // Rows are available only if they are available in every child column
    fn avail(&self) -> u64 {
        self.children
            .iter()
            .map(|c| c.rows_available)
            .min()
            .unwrap()
    }

    // Rows are unawaited if they are unawaited in any child column
    fn unawaited(&self) -> u64 {
        self.children
            .iter()
            .map(|c| c.rows_unawaited)
            .max()
            .unwrap()
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
    fn maybe_encode(&mut self, array: ArrayRef) -> Result<Vec<EncodeTask>> {
        self.num_rows_seen += array.len() as u64;
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

    fn finish(&mut self) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        async move {
            let mut columns = Vec::new();
            // Add a column for the struct header
            columns.push(EncodedColumn::default());
            for child in self.children.iter_mut() {
                columns.extend(child.finish().await?);
            }
            Ok(columns)
        }
        .boxed()
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
