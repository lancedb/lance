use std::{collections::VecDeque, ops::Range, sync::Arc};

use arrow_array::{cast::AsArray, ArrayRef, StructArray};
use arrow_schema::Fields;
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use tokio::sync::mpsc;

use crate::{
    decoder::{DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask},
    encoder::{EncodedPage, FieldEncoder},
    format::pb,
    EncodingsIo,
};
use lance_core::Result;

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
#[derive(Debug)]
pub struct SimpleStructScheduler {
    children: Vec<Vec<Box<dyn LogicalPageScheduler>>>,
    child_fields: Fields,
    num_rows: u32,
}

impl SimpleStructScheduler {
    pub fn new(children: Vec<Vec<Box<dyn LogicalPageScheduler>>>, child_fields: Fields) -> Self {
        debug_assert!(!children.is_empty());
        let num_rows = children[0].iter().map(|page| page.num_rows()).sum();
        // Ensure that all the children have the same number of rows
        debug_assert!(children.iter().all(|col| col
            .iter()
            .map(|page| page.num_rows())
            .sum::<u32>()
            == num_rows));
        Self {
            children,
            child_fields,
            num_rows,
        }
    }
}

// As we schedule we keep one of these per column so that we know
// how far into the column we have already scheduled.
#[derive(Debug, Clone, Copy)]
struct FieldWalkStatus {
    rows_to_skip: u32,
    rows_to_take: u32,
    page_offset: u32,
    rows_queued: u32,
}

impl FieldWalkStatus {
    fn new_from_range(range: Range<u32>) -> Self {
        Self {
            rows_to_skip: range.start,
            rows_to_take: range.end - range.start,
            page_offset: 0,
            rows_queued: 0,
        }
    }
}

impl LogicalPageScheduler for SimpleStructScheduler {
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        let mut rows_to_read = range.end - range.start;
        trace!(
            "Scheduling struct decode of range {:?} ({} rows)",
            range,
            rows_to_read
        );

        // Before we do anything, send a struct decoder to the decode thread so it can start decoding the pages
        // we are about to send.
        //
        // This will need to get a tiny bit more complicated once structs have their own nullability and that nullability
        // information starts to span multiple pages.
        sink.send(Box::new(SimpleStructDecoder::new(
            self.child_fields.clone(),
            rows_to_read,
        )))
        .unwrap();

        let mut field_status = vec![FieldWalkStatus::new_from_range(range); self.children.len()];

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

        while rows_to_read > 0 {
            let mut min_rows_added = u32::MAX;
            for (col_idx, field_scheduler) in self.children.iter().enumerate() {
                let status = &mut field_status[col_idx];
                if status.rows_queued == 0 {
                    trace!("Need additional rows for column {}", col_idx);
                    let mut next_page = &field_scheduler[status.page_offset as usize];

                    while status.rows_to_skip >= next_page.num_rows() {
                        status.rows_to_skip -= next_page.num_rows();
                        status.page_offset += 1;
                        trace!("Skipping entire page of {} rows", next_page.num_rows());
                        next_page = &field_scheduler[status.page_offset as usize];
                    }

                    let page_range_start = status.rows_to_skip;
                    let page_rows_remaining = next_page.num_rows() - page_range_start;
                    let rows_to_take = status.rows_to_take.min(page_rows_remaining);
                    let page_range = page_range_start..(page_range_start + rows_to_take);

                    trace!(
                        "Taking {} rows from column {} starting at page offset {} from page {:?}",
                        rows_to_take,
                        col_idx,
                        page_range_start,
                        next_page
                    );
                    next_page.schedule_range(page_range, scheduler, sink)?;

                    status.rows_queued += rows_to_take;
                    status.rows_to_take -= rows_to_take;
                    status.page_offset += 1;
                    status.rows_to_skip = 0;

                    min_rows_added = min_rows_added.min(rows_to_take);
                }
            }
            if min_rows_added == 0 {
                panic!("Error in scheduling logic, panic to avoid infinite loop");
            }
            rows_to_read -= min_rows_added;
            for field_status in &mut field_status {
                field_status.rows_queued -= min_rows_added;
            }
        }
        Ok(())
    }

    fn num_rows(&self) -> u32 {
        self.num_rows
    }
}

struct ChildState {
    // As we decode a column we pull pages out of the channel source and into
    // a queue for that column.  Since we await as soon as we pull the page from
    // the source there is no need for a separate unawaited queue.
    //
    // Technically though, these pages are only "partially awaited"
    //
    // Note: This queue may have more than one page in it if the batch size is very large
    // or pages are very small
    // TODO: Test this case
    //
    // Then we drain this queue pages as we decode.
    awaited: VecDeque<Box<dyn LogicalPageDecoder>>,
    // Rows that should still be coming over the channel source but haven't yet been
    // put into the awaited queue
    rows_unawaited: u32,
    // Rows that have been pulled out of the channel source, awaited, and are ready to
    // be drained
    rows_available: u32,
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
    fn new(num_rows: u32) -> Self {
        Self {
            awaited: VecDeque::new(),
            rows_unawaited: num_rows,
            rows_available: 0,
        }
    }

    async fn wait(
        &mut self,
        num_rows: u32,
        source: &mut mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        trace!(
            "Struct waiting for {} rows and {} are available already",
            num_rows,
            self.rows_available
        );
        // Note: this code is the inverse of the row-major scheduling algorithm
        let mut remaining = num_rows.saturating_sub(self.rows_available);
        if remaining > 0 {
            if let Some(back) = self.awaited.back_mut() {
                if back.unawaited() > 0 {
                    let rows_to_wait = remaining.min(back.unawaited());
                    trace!(
                        "Struct await an additional {} rows from the current page",
                        rows_to_wait
                    );
                    // Even though we wait for X rows we might actually end up
                    // loading more than that
                    let previously_avail = back.avail();
                    back.wait(rows_to_wait, source).await?;
                    let newly_avail = back.avail() - previously_avail;
                    trace!("The await loaded {} rows", newly_avail);
                    self.rows_available += newly_avail;
                    self.rows_unawaited -= newly_avail;
                    remaining -= rows_to_wait;
                }
            }

            while remaining > 0 {
                // Because we schedule in row-major fashion we know the next page
                // will belong to this column.
                let mut decoder = source.recv().await.unwrap();
                let rows_to_wait = remaining.min(decoder.unawaited());
                trace!(
                    "Struct received new page and awaiting {} rows",
                    rows_to_wait
                );
                // We might only await part of a page.  This is important for things
                // like the struct<struct<...>> case where we have one outer page, one
                // middle page, and then a bunch of inner pages.  If we await the entire
                // middle page then we will have to wait for all the inner pages to arrive
                // before we can start decoding.
                //
                // TODO: test this case
                decoder.wait(rows_to_wait, source).await?;
                let newly_avail = decoder.avail();
                self.awaited.push_back(decoder);
                self.rows_available += newly_avail;
                self.rows_unawaited -= newly_avail;
                remaining -= rows_to_wait;
                trace!(
                    "The new await loaded {} rows and {} are still needed",
                    newly_avail,
                    remaining
                );
            }
        }
        Ok(())
    }

    fn drain(&mut self, num_rows: u32) -> Result<CompositeDecodeTask> {
        trace!("Struct draining {} rows", num_rows);
        debug_assert!(self.rows_available >= num_rows);
        debug_assert!(num_rows > 0);
        self.rows_available -= num_rows;
        let mut remaining = num_rows;
        let mut composite = CompositeDecodeTask {
            tasks: Vec::new(),
            num_rows: 0,
            has_more: true,
        };
        while remaining > 0 {
            let next = self.awaited.front_mut().unwrap();
            let rows_to_take = remaining.min(next.avail());
            let next_task = next.drain(rows_to_take)?;
            if next.avail() == 0 && next.unawaited() == 0 {
                trace!("Completely drained page");
                self.awaited.pop_front();
            }
            remaining -= rows_to_take;
            composite.tasks.push(next_task.task);
            composite.num_rows += next_task.num_rows;
        }
        composite.has_more = self.rows_available != 0 || self.rows_unawaited != 0;
        Ok(composite)
    }
}

struct SimpleStructDecoder {
    children: Vec<ChildState>,
    child_fields: Fields,
}

impl SimpleStructDecoder {
    fn new(child_fields: Fields, num_rows: u32) -> Self {
        Self {
            children: child_fields
                .iter()
                .map(|_| ChildState::new(num_rows))
                .collect(),
            child_fields,
        }
    }
}

impl LogicalPageDecoder for SimpleStructDecoder {
    fn wait<'a>(
        &'a mut self,
        num_rows: u32,
        source: &'a mut mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'a, Result<()>> {
        async move {
            for child in &mut self.children {
                child.wait(num_rows, source).await?;
            }
            Ok(())
        }
        .boxed()
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
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
    fn avail(&self) -> u32 {
        self.children
            .iter()
            .map(|c| c.rows_available)
            .min()
            .unwrap()
    }

    // Rows are unawaited if they are unawaited in any child column
    fn unawaited(&self) -> u32 {
        self.children
            .iter()
            .map(|c| c.rows_unawaited)
            .max()
            .unwrap()
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

struct StructFieldEncoder {
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
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
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

    fn flush(&mut self) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
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
                buffers: vec![],
                encoding: pb::ArrayEncoding {
                    array_encoding: Some(pb::array_encoding::ArrayEncoding::Struct(
                        pb::SimpleStruct {},
                    )),
                },
                num_rows: num_rows_seen,
                column_idx: column_index,
            }))
            .boxed(),
        );
        Ok(child_tasks)
    }

    fn num_columns(&self) -> u32 {
        self.children.len() as u32 + 1
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Fields};

    use crate::{
        encoder::FieldEncoder,
        encodings::{
            logical::{primitive::PrimitiveFieldEncoder, r#struct::StructFieldEncoder},
            physical::basic::BasicEncoder,
        },
        testing::check_round_trip_field_encoding,
    };

    #[test_log::test(tokio::test)]
    async fn test_simple_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let child_encoders = vec![
            Box::new(PrimitiveFieldEncoder::new(
                4096,
                Arc::new(BasicEncoder::new(1)),
            )) as Box<dyn FieldEncoder>,
            Box::new(PrimitiveFieldEncoder::new(
                4096,
                Arc::new(BasicEncoder::new(2)),
            )) as Box<dyn FieldEncoder>,
        ];
        let encoder = StructFieldEncoder::new(child_encoders, 0);
        let field = Field::new("", data_type, false);
        check_round_trip_field_encoding(encoder, field).await;
    }
}
