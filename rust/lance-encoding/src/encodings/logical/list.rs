use std::{collections::VecDeque, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    ArrayRef, Int32Array, Int64Array, LargeListArray, ListArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field};
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};
use tokio::{sync::mpsc, task::JoinHandle};

use lance_core::{Error, Result};

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
    fn schedule_range(
        &self,
        range: std::ops::Range<u32>,
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        trace!("Scheduling list offsets range: {:?}", range);
        let num_rows = range.end - range.start;
        let num_offsets = num_rows + 1;
        let offsets_range = range.start..(range.end + 1);
        // Create a channel for the internal schedule / decode loop that is unique
        // to this page.
        let (tx, mut rx) = mpsc::unbounded_channel();
        self.offsets_scheduler
            .schedule_range(offsets_range, scheduler, &tx)?;
        let mut scheduled_offsets = rx.recv().now_or_never().unwrap().unwrap();
        let items_schedulers = self.items_schedulers.clone();
        let scheduler = scheduler.clone();

        // First we schedule, as normal, the I/O for the offsets.  Then we immediately spawn
        // a task to decode those offsets and schedule the I/O for the items.  If we wait until
        // the decode task has launched then we will be delaying the I/O for the items until we
        // need them which is not good.  Better to spend some eager CPU and start loading the
        // items immediately.
        let indirect_fut = tokio::task::spawn(async move {
            // We know the offsets are a primitive array and thus will not need additional
            // pages.  We can use a dummy receiver to match the decoder API
            let (_, mut dummy_rx) = mpsc::unbounded_channel();
            scheduled_offsets.wait(num_rows, &mut dummy_rx).await?;
            let decode_task = scheduled_offsets.drain(num_offsets)?;
            let offsets = decode_task.task.decode()?;
            let numeric_offsets = offsets.as_primitive::<Int32Type>();
            let start = numeric_offsets.values()[0] as u32;
            let end = numeric_offsets.values()[numeric_offsets.len() - 1] as u32;
            trace!(
                "List offsets range of {:?} maps to item range {:?}..{:?}",
                range,
                start,
                end
            );

            let mut rows_to_take = end - start;
            let mut rows_to_skip = start;
            let (tx, mut rx) = mpsc::unbounded_channel();

            trace!(
                "Indirectly scheduling items from {} list items pages",
                items_schedulers.len()
            );
            for item_scheduler in items_schedulers.as_ref() {
                if item_scheduler.num_rows() < rows_to_skip {
                    rows_to_skip -= item_scheduler.num_rows()
                } else {
                    let rows_avail = item_scheduler.num_rows() - rows_to_skip;
                    let to_take = rows_to_take.min(rows_avail);
                    let page_range = rows_to_skip..(rows_to_skip + to_take);
                    // Note that, if we have List<List<...>> then this call will schedule yet another round
                    // of I/O :)
                    item_scheduler.schedule_range(page_range, &scheduler, &tx)?;
                    rows_to_skip = 0;
                    rows_to_take -= to_take;
                }
            }
            let mut item_decoders = Vec::new();
            drop(tx);
            while let Some(item_decoder) = rx.recv().now_or_never().unwrap() {
                item_decoders.push(item_decoder);
            }

            Ok(IndirectlyLoaded {
                offsets,
                item_decoders,
            })
        });
        sink.send(Box::new(ListPageDecoder {
            offsets: None,
            unawaited: VecDeque::new(),
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
}

/// As soon as the first call to decode comes in we wait for all indirect I/O to
/// complete.  TODO: We could potentially be lazier here, to investigate.
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
    unawaited: VecDeque<Box<dyn LogicalPageDecoder>>,
    // offsets will have already been decoded as part of the indirect I/O
    // and so we store ArrayRef and not Box<dyn LogicalPageDecoder>
    offsets: Option<ArrayRef>,
    // Items will not yet be decoded, we at least try and do that part
    // on the decode thread
    item_decoders: VecDeque<Box<dyn LogicalPageDecoder>>,
    num_rows: u32,
    rows_drained: u32,
    offset_type: DataType,
}

impl ListPageDecoder {
    fn rows_immediately_available(&self) -> u32 {
        self.item_decoders.iter().map(|d| d.avail()).sum()
    }
}

struct ListDecodeTask {
    offsets: ArrayRef,
    items: Vec<Box<dyn DecodeArrayTask>>,
    offset_type: DataType,
}

impl DecodeArrayTask for ListDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let offsets = self.offsets;
        let items = self
            .items
            .into_iter()
            .map(|task| task.decode())
            .collect::<Result<Vec<_>>>()?;
        let item_refs = items.iter().map(|item| item.as_ref()).collect::<Vec<_>>();
        // TODO: could maybe try and "page bridge" these at some point
        // (assuming item type is primitive) to avoid the concat
        let items = arrow_select::concat::concat(&item_refs)?;
        let nulls = offsets.nulls().cloned();
        // TODO: we default to nullable true here, should probably use the nullability given to
        // us from the input schema
        let item_field = Arc::new(Field::new("item", items.data_type().clone(), true));

        // The offsets are already decoded but they need to be shifted back to 0
        // TODO: Can these branches be simplified?
        match &self.offset_type {
            DataType::Int32 => {
                let offsets = arrow_cast::cast(&offsets, &DataType::Int32)?;
                let offsets_i32 = offsets.as_primitive::<Int32Type>();
                let min_offset = Int32Array::new_scalar(offsets_i32.value(0));
                let offsets = arrow_arith::numeric::sub(&offsets_i32, &min_offset)?;
                let offsets_i32 = offsets.as_primitive::<Int32Type>();
                let offsets = OffsetBuffer::new(offsets_i32.values().clone());

                Ok(Arc::new(ListArray::try_new(
                    item_field, offsets, items, nulls,
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
                    item_field, offsets, items, nulls,
                )?))
            }
            _ => panic!("ListDecodeTask with data type that is not i32 or i64"),
        }
    }
}

impl LogicalPageDecoder for ListPageDecoder {
    fn wait<'a>(
        &'a mut self,
        num_rows: u32,
        source: &'a mut mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'a, Result<()>> {
        async move {
            // First, wait for the indirect I/O to finish, if we haven't already
            if self.unloaded.is_some() {
                let indirectly_loaded = self.unloaded.take().unwrap().await.unwrap()?;
                self.offsets = Some(indirectly_loaded.offsets);
                self.unawaited.extend(indirectly_loaded.item_decoders);
            }
            // Next, pull as many items as we can from decoders that have already
            // been "awaited".
            let avail = self.rows_immediately_available();
            if avail >= num_rows {
                return Ok(());
            }
            let mut remaining = num_rows - avail;
            if let Some(partial) = self.item_decoders.back_mut() {
                let rows_left_unawaited = partial.unawaited();
                if rows_left_unawaited > 0 {
                    let rows_to_take = rows_left_unawaited.min(remaining);
                    partial.wait(rows_to_take, source).await?;
                    remaining -= rows_to_take;
                }
            }
            // Finally, pull items from the unawaited queue
            while remaining > 0 {
                let mut next_to_await = self.unawaited.pop_front().ok_or_else(|| Error::Internal { message: format!("list page was asked for {} rows but ran out of item pages before that happened", remaining), location: location!() })?;
                let rows_to_take = next_to_await.unawaited().min(remaining);
                next_to_await.wait(rows_to_take, source).await?;
                remaining -= rows_to_take;
                self.item_decoders.push_back(next_to_await);
            }
            Ok(())
        }
        .boxed()
    }

    fn unawaited(&self) -> u32 {
        match self.unloaded {
            None => {
                let partial = self
                    .item_decoders
                    .back()
                    .map(|d| d.unawaited())
                    .unwrap_or(0);
                partial + self.unawaited.iter().map(|d| d.unawaited()).sum::<u32>()
            }
            Some(_) => self.num_rows,
        }
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        // We already have the offsets but need to drain the item pages
        let offsets = self
            .offsets
            .as_ref()
            .unwrap()
            .slice(self.rows_drained as usize, num_rows as usize + 1);

        // TODO: Support for large list, nullability, will probably change this away from Int32Type
        let offsets_i32 = offsets.as_primitive::<Int32Type>();
        let start = offsets_i32.values()[0];
        let end = offsets_i32.values()[offsets_i32.len() - 1];
        let mut num_items_to_drain = end - start;

        let mut item_decodes = Vec::new();
        while num_items_to_drain > 0 {
            let next_item_page = self.item_decoders.front_mut().unwrap();
            let avail = next_item_page.avail();
            let to_take = num_items_to_drain.min(avail as i32) as u32;
            num_items_to_drain -= to_take as i32;
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
        self.num_rows - self.rows_drained
    }
}

struct IndirectlyLoaded {
    offsets: ArrayRef,
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
