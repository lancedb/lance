// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::VecDeque, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    ArrayRef, Int32Array, Int64Array, LargeListArray, ListArray, UInt32Array,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field};
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use tokio::{sync::mpsc, task::JoinHandle};

use lance_core::Result;

use crate::{
    decoder::{DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask},
    encoder::{EncodedPage, FieldEncoder},
    encodings::physical::basic::BasicEncoder,
    format::pb,
    EncodingsIo,
};

use super::primitive::PrimitiveFieldEncoder;

/// A page scheduler for list fields that encodes offsets in one field and items in another
///
/// TODO: Implement list nullability
///
/// The list scheduler is somewhat unique because it requires indirect I/O.  We cannot know the
/// ranges we need simply by looking at the metadata.  This means that list scheduling doesn't
/// fit neatly into the two-thread schedule-loop / decode-loop model.  To handle this, when a
/// list page is scheduled, we only schedule the I/O for the offsets and then we immediately
/// launch a new tokio task.  This new task waits for the offsets, decodes them, and then
/// schedules the I/O for the items.  Keep in mind that list items can be lists themselves.  If
/// that is the case then this indirection will continue.  The decode task that is returned will
/// only finish `wait`ing when all of the I/O has completed.
///
/// Whenever we schedule follow-up I/O like this the priority is based on the top-level row
/// index.  This helps ensure that earlier rows get finished completely (including follow up
/// tasks) before we perform I/O for later rows.
///
/// TODO: Actually implement the priority system described above
///
/// Note: The length of the list page is 1 less than the length of the offsets page
// TODO: Right now we are assuming that list offsets and list items are written at the same time.
// As a result, we know which item pages correspond to which index page and there are no item
// pages which overlap two different index pages.
//
// In the metadata for the page we store only the u64 num_items referenced by the page.
//
// We could relax this constraint.  Either each index page could store the u64 offset into
// the total range of item pages or each index page could store a u64 num_items and a u32
// first_item_page_offset.
#[derive(Debug)]
pub struct ListPageScheduler {
    offsets_scheduler: Box<dyn LogicalPageScheduler>,
    items_schedulers: Arc<Vec<Box<dyn LogicalPageScheduler>>>,
    offset_type: DataType,
}

impl ListPageScheduler {
    // Create a new ListPageScheduler
    pub fn new(
        offsets_scheduler: Box<dyn LogicalPageScheduler>,
        items_schedulers: Vec<Box<dyn LogicalPageScheduler>>,
        // Should be int32 or int64
        offset_type: DataType,
    ) -> Self {
        match &offset_type {
            DataType::Int32 | DataType::Int64 => {}
            _ => panic!(),
        }
        Self {
            offsets_scheduler,
            items_schedulers: Arc::new(items_schedulers),
            offset_type,
        }
    }
}

impl LogicalPageScheduler for ListPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        let num_rows = ranges.iter().map(|range| range.end - range.start).sum();
        // TODO: Should coalesce here (e.g. if receiving take(&[0, 1, 2]))
        // otherwise we are double-dipping on the offsets scheduling
        let offsets_ranges = ranges
            .iter()
            .map(|range| range.start..(range.end + 1))
            .collect::<Vec<_>>();
        let num_offsets = offsets_ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum();
        trace!("Scheduling list offsets ranges: {:?}", offsets_ranges);
        // Create a channel for the internal schedule / decode loop that is unique
        // to this page.
        let (tx, mut rx) = mpsc::unbounded_channel();
        self.offsets_scheduler
            .schedule_ranges(&offsets_ranges, scheduler, &tx)?;
        let mut scheduled_offsets = rx.recv().now_or_never().unwrap().unwrap();
        let items_schedulers = self.items_schedulers.clone();
        let ranges = ranges.to_vec();
        let scheduler = scheduler.clone();

        // First we schedule, as normal, the I/O for the offsets.  Then we immediately spawn
        // a task to decode those offsets and schedule the I/O for the items AND wait for
        // the items.  If we wait until the decode task has launched then we will be delaying
        // the I/O for the items until we need them which is not good.  Better to spend some
        // eager CPU and start loading the items immediately.
        let indirect_fut = tokio::task::spawn(async move {
            // We know the offsets are a primitive array and thus will not need additional
            // pages.  We can use a dummy receiver to match the decoder API
            let (_, mut dummy_rx) = mpsc::unbounded_channel();
            scheduled_offsets.wait(num_rows, &mut dummy_rx).await?;
            let decode_task = scheduled_offsets.drain(num_offsets)?;
            let offsets = decode_task.task.decode()?;
            let numeric_offsets = offsets.as_primitive::<Int32Type>();
            // Given ranges [1..3, 5..6] where each list has 10 items we get offsets [[10, 20, 30], [50, 60]]
            // and we need to normalize to [0, 10, 20, 30]
            let mut normalized_offsets =
                Vec::with_capacity(numeric_offsets.len() - ranges.len() + 1);
            normalized_offsets.push(0);
            let mut last_normalized_offset = 0;
            let offsets_values = numeric_offsets.values();

            let mut item_ranges = VecDeque::new();
            let mut offsets_offset: u32 = 0;
            for range in ranges {
                let num_lists = range.end - range.start;
                let items_start = offsets_values[offsets_offset as usize] as u32;
                let items_end = offsets_values[(offsets_offset + num_lists) as usize] as u32;
                normalized_offsets.extend(
                    offsets_values
                        .slice(offsets_offset as usize, (num_lists + 1) as usize)
                        .windows(2)
                        .map(|w| {
                            let length = w[1] - w[0];
                            last_normalized_offset += length as u32;
                            last_normalized_offset
                        }),
                );
                trace!(
                    "List offsets range of {:?} maps to item range {:?}..{:?}",
                    range,
                    items_start,
                    items_end
                );
                offsets_offset += num_lists + 1;
                item_ranges.push_back(items_start..items_end);
            }

            let (tx, mut rx) = mpsc::unbounded_channel();

            trace!(
                "Indirectly scheduling items ranges {:?} from {} list items pages",
                item_ranges,
                items_schedulers.len()
            );
            let mut item_schedulers = VecDeque::from_iter(items_schedulers.iter());
            let mut row_offset = 0;
            let mut next_scheduler = item_schedulers.pop_front().unwrap();
            let mut next_range = item_ranges.pop_front().unwrap();
            let mut next_item_ranges = Vec::new();

            // TODO: Test List<List<...>>
            // This is a bit complicated.  We have a list of ranges and we have a list of
            // item schedulers.  We walk through both lists, scheduling the overlap.  For
            // example, if we need items [500...1000], [2200..2300] [2500...4000] and we have 5 item
            // pages with 1000 rows each then we need to schedule:
            //
            // page 0: 500..1000
            // page 1: nothing
            // page 2: 200..300, 500..1000
            // page 3: 0..1000
            // page 4: nothing
            loop {
                let current_scheduler_end = row_offset + next_scheduler.num_rows();
                if next_range.start > current_scheduler_end {
                    // All requested items are past this page, continue
                    row_offset += next_scheduler.num_rows();
                    if !next_item_ranges.is_empty() {
                        next_scheduler.schedule_ranges(&next_item_ranges, &scheduler, &tx)?;
                    }
                    next_scheduler = item_schedulers.pop_front().unwrap();
                } else if next_range.end <= current_scheduler_end {
                    // Range entirely contained in current scheduler
                    let page_range = (next_range.start - row_offset)..(next_range.end - row_offset);
                    next_item_ranges.push(page_range);
                    if let Some(item_range) = item_ranges.pop_front() {
                        next_range = item_range;
                    } else {
                        // We have processed all pages
                        break;
                    }
                } else {
                    // Range partially contained in current scheduler
                    let page_range = (next_range.start - row_offset)..next_scheduler.num_rows();
                    next_range = current_scheduler_end..next_range.end;
                    next_item_ranges.push(page_range);
                    row_offset += next_scheduler.num_rows();
                    if !next_item_ranges.is_empty() {
                        next_scheduler.schedule_ranges(&next_item_ranges, &scheduler, &tx)?;
                    }
                    next_scheduler = item_schedulers.pop_front().unwrap();
                }
            }
            if !next_item_ranges.is_empty() {
                next_scheduler.schedule_ranges(&next_item_ranges, &scheduler, &tx)?;
            }
            let mut item_decoders = Vec::new();
            drop(tx);
            while let Some(mut item_decoder) = rx.recv().await {
                item_decoder.wait(item_decoder.unawaited(), &mut rx).await?;
                item_decoders.push(item_decoder);
            }

            Ok(IndirectlyLoaded {
                offsets: normalized_offsets,
                item_decoders,
            })
        });
        sink.send(Box::new(ListPageDecoder {
            offsets: Vec::new(),
            item_decoders: VecDeque::new(),
            num_rows,
            rows_drained: 0,
            unloaded: Some(indirect_fut),
            offset_type: self.offset_type.clone(),
        }))
        .unwrap();
        Ok(())
    }

    // A list page's length is one less than the length of the offsets
    fn num_rows(&self) -> u32 {
        self.offsets_scheduler.num_rows() - 1
    }

    fn schedule_take(
        &self,
        indices: &[u32],
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        trace!("Scheduling list offsets for {} indices", indices.len());
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

/// As soon as the first call to decode comes in we wait for all indirect I/O to
/// complete.
///
/// Once the indirect I/O is finished we pull items out of `unawaited`, wait them
/// (this wait should return immedately) and then push them into `item_decoders`.
///
/// We then drain from `item_decoders`, popping item pages off as we finish with
/// them.
///
/// TODO: Test the case where a single list page has multiple items pages
struct ListPageDecoder {
    unloaded: Option<JoinHandle<Result<IndirectlyLoaded>>>,
    // offsets will have already been decoded as part of the indirect I/O
    // and so we store ArrayRef and not Box<dyn LogicalPageDecoder>
    offsets: Vec<u32>,
    // Items will not yet be decoded, we at least try and do that part
    // on the decode thread
    item_decoders: VecDeque<Box<dyn LogicalPageDecoder>>,
    num_rows: u32,
    rows_drained: u32,
    offset_type: DataType,
}

struct ListDecodeTask {
    offsets: Vec<u32>,
    items: Vec<Box<dyn DecodeArrayTask>>,
    offset_type: DataType,
}

impl DecodeArrayTask for ListDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let items = self
            .items
            .into_iter()
            .map(|task| task.decode())
            .collect::<Result<Vec<_>>>()?;
        let item_refs = items.iter().map(|item| item.as_ref()).collect::<Vec<_>>();
        // TODO: could maybe try and "page bridge" these at some point
        // (assuming item type is primitive) to avoid the concat
        let items = arrow_select::concat::concat(&item_refs)?;
        // TODO: we default to nullable true here, should probably use the nullability given to
        // us from the input schema
        let item_field = Arc::new(Field::new("item", items.data_type().clone(), true));

        // The offsets are already decoded but they need to be shifted back to 0
        // TODO: Can these branches be simplified?
        let offsets = UInt32Array::from(self.offsets);
        match &self.offset_type {
            DataType::Int32 => {
                let offsets = arrow_cast::cast(&offsets, &DataType::Int32)?;
                let offsets_i32 = offsets.as_primitive::<Int32Type>();
                let min_offset = Int32Array::new_scalar(offsets_i32.value(0));
                let offsets = arrow_arith::numeric::sub(&offsets_i32, &min_offset)?;
                let offsets_i32 = offsets.as_primitive::<Int32Type>();
                let offsets = OffsetBuffer::new(offsets_i32.values().clone());

                Ok(Arc::new(ListArray::try_new(
                    item_field, offsets, items, None,
                )?))
            }
            DataType::Int64 => {
                let offsets = arrow_cast::cast(&offsets, &DataType::Int64)?;
                let offsets_i64 = offsets.as_primitive::<Int64Type>();
                let min_offset = Int64Array::new_scalar(offsets_i64.value(0));
                let offsets = arrow_arith::numeric::sub(&offsets_i64, &min_offset)?;
                let offsets_i64 = offsets.as_primitive::<Int64Type>();
                let offsets = OffsetBuffer::new(offsets_i64.values().clone());

                Ok(Arc::new(LargeListArray::try_new(
                    item_field, offsets, items, None,
                )?))
            }
            _ => panic!("ListDecodeTask with data type that is not i32 or i64"),
        }
    }
}

impl LogicalPageDecoder for ListPageDecoder {
    fn wait<'a>(
        &'a mut self,
        // No support for partial wait
        _num_rows: u32,
        // We will never pull from source because of indirect I/O
        _source: &'a mut mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'a, Result<()>> {
        async move {
            // wait for the indirect I/O to finish, if we haven't already.  We don't need to
            // wait for anything after that because we eagerly load item pages as part of the
            // indirect thread
            if self.unloaded.is_some() {
                let indirectly_loaded = self.unloaded.take().unwrap().await.unwrap()?;
                self.offsets = indirectly_loaded.offsets;
                self.item_decoders.extend(indirectly_loaded.item_decoders);
            }
            Ok(())
        }
        .boxed()
    }

    fn unawaited(&self) -> u32 {
        match self.unloaded {
            None => 0,
            Some(_) => self.num_rows,
        }
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        // We already have the offsets but need to drain the item pages
        let offsets = self.offsets
            [self.rows_drained as usize..(self.rows_drained + num_rows + 1) as usize]
            .to_vec();
        let start = offsets[0];
        let end = offsets[offsets.len() - 1];
        let mut num_items_to_drain = end - start;

        let mut item_decodes = Vec::new();
        while num_items_to_drain > 0 {
            let next_item_page = self.item_decoders.front_mut().unwrap();
            let avail = next_item_page.avail();
            let to_take = num_items_to_drain.min(avail);
            num_items_to_drain -= to_take;
            let next_task = next_item_page.drain(to_take)?;

            if !next_task.has_more {
                self.item_decoders.pop_front();
            }
            item_decodes.push(next_task.task);
        }

        self.rows_drained += num_rows;
        Ok(NextDecodeTask {
            has_more: self.avail() > 0,
            num_rows,
            task: Box::new(ListDecodeTask {
                offsets,
                items: item_decodes,
                offset_type: self.offset_type.clone(),
            }) as Box<dyn DecodeArrayTask>,
        })
    }

    fn avail(&self) -> u32 {
        match self.unloaded {
            Some(_) => 0,
            None => self.num_rows - self.rows_drained,
        }
    }
}

struct IndirectlyLoaded {
    offsets: Vec<u32>,
    item_decoders: Vec<Box<dyn LogicalPageDecoder>>,
}

pub struct ListFieldEncoder {
    indices_encoder: PrimitiveFieldEncoder,
    items_encoder: Box<dyn FieldEncoder>,
}

impl ListFieldEncoder {
    pub fn new(
        items_encoder: Box<dyn FieldEncoder>,
        cache_bytes_per_columns: u64,
        column_index: u32,
    ) -> Self {
        Self {
            indices_encoder: PrimitiveFieldEncoder::new(
                cache_bytes_per_columns,
                Arc::new(BasicEncoder::new(column_index)),
            ),
            items_encoder,
        }
    }

    fn combine_index_tasks(
        index_tasks: Result<Vec<BoxFuture<'static, Result<EncodedPage>>>>,
        item_tasks: Result<Vec<BoxFuture<'static, Result<EncodedPage>>>>,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        let mut index_tasks = index_tasks?;
        let item_tasks = item_tasks?;
        index_tasks.extend(item_tasks);
        Ok(index_tasks)
    }

    fn wrap_index_encode_tasks(
        tasks: Result<Vec<BoxFuture<'static, Result<EncodedPage>>>>,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        tasks.map(|tasks| {
            tasks
                .into_iter()
                .map(|page_task| {
                    async move {
                        let page = page_task.await?;
                        Ok(EncodedPage {
                            buffers: page.buffers,
                            column_idx: page.column_idx,
                            num_rows: page.num_rows,
                            encoding: pb::ArrayEncoding {
                                array_encoding: Some(pb::array_encoding::ArrayEncoding::List(
                                    Box::new(pb::List {
                                        offsets: Some(Box::new(page.encoding)),
                                    }),
                                )),
                            },
                        })
                    }
                    .boxed()
                })
                .collect::<Vec<_>>()
        })
    }
}

impl FieldEncoder for ListFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        let items = match array.data_type() {
            DataType::List(_) => array.as_list::<i32>().values().clone(),
            DataType::LargeList(_) => array.as_list::<i64>().values().clone(),
            _ => panic!(),
        };
        let offsets = match array.data_type() {
            DataType::List(_) => {
                let offsets = array.as_list::<i32>().offsets().clone();
                Arc::new(Int32Array::new(offsets.into_inner(), None)) as ArrayRef
            }
            DataType::LargeList(_) => {
                let offsets = array.as_list::<i64>().offsets().clone();
                Arc::new(Int64Array::new(offsets.into_inner(), None)) as ArrayRef
            }
            _ => panic!(),
        };
        let index_tasks = self.indices_encoder.maybe_encode(offsets);
        let index_tasks = Self::wrap_index_encode_tasks(index_tasks);
        let item_tasks = self.items_encoder.maybe_encode(items);
        Self::combine_index_tasks(index_tasks, item_tasks)
    }

    fn flush(&mut self) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>> {
        let index_tasks = self.indices_encoder.flush();
        let index_tasks = Self::wrap_index_encode_tasks(index_tasks);
        let item_tasks = self.items_encoder.flush();
        Self::combine_index_tasks(index_tasks, item_tasks)
    }

    fn num_columns(&self) -> u32 {
        2
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use crate::{
        encodings::{
            logical::{list::ListFieldEncoder, primitive::PrimitiveFieldEncoder},
            physical::basic::BasicEncoder,
        },
        testing::check_round_trip_field_encoding,
    };

    #[test_log::test(tokio::test)]
    async fn test_simple_list() {
        let data_type = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let items_encoder = Box::new(PrimitiveFieldEncoder::new(
            4096,
            Arc::new(BasicEncoder::new(1)),
        ));
        let encoder = ListFieldEncoder::new(items_encoder, 4096, 0);
        let field = Field::new("", data_type, false);
        check_round_trip_field_encoding(encoder, field).await;
    }
}
