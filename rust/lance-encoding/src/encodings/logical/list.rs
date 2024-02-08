use std::{collections::VecDeque, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Int32Type, Int64Type},
    ArrayRef, Int32Array, Int64Array, LargeListArray, ListArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field};
use futures::{future::BoxFuture, FutureExt, TryFutureExt};
use tokio::task::JoinHandle;

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
/// Offsets are encoded as u64.  This is neccesary beacuse a single page of list offsets could
/// reference multiple pages of item data and thus these offsets can exceed the u32 range.  If
/// this does happen then the list data type should be large list.
///
/// Encoding offsets as u64 does not impose much penalty for smaller lists.  The encoding used
/// to store the u64 offsets should be doing some kind of bit-packing to ensure that the excess
/// range does not incur an I/O penalty.  So for small list pages we are likely using considerably
/// less than 32 bits per offset.
///
/// The list scheduler is somewhat unique because it requires indirect I/O.  We cannot know the
/// ranges we need simply by looking at the metadata.  This means that list scheduling doesn't
/// fit neatly into the two-thread schedule-loop / decode-loop model.  To handle this, when a
/// list page is scheduled, we only schedule the I/O for the offsets and then we immediately
/// launch a new thread task.  This new thread task waits for the offsets, decodes them, and then
/// schedules the I/O for the items.  Keep in mind that list items can be lists themselves.  If
/// that is the case then this indirection will continue.  The decode task that is returned will
/// only finish `wait`ing when all of the I/O has completed.
///
/// Whenever we schedule follow-up I/O like this the priority is based on the top-level row
/// index.  This helps ensure that earlier rows get finished completely (including follow up
/// tasks) before we perform I/O for later rows.
///
/// Note: The length of the list page is 1 less than the legnth of the offsets page
// TODO: Right now we are assuming that list offsets and list items are written at the same time.
// As a result, we know which item pages correspond to which index page and there are no item
// pages which overlap two different index pages.
//
// In the metadata for the page we store only the u64 num_items referenced by the page.
//
// We could relax this constraint.  Either each index page could store the u64 offset into
// the total range of item pages or each index page could store a u64 num_items and a u32
// first_item_page_offset.
pub struct ListPageScheduler {
    offsets_scheduler: Box<dyn LogicalPageScheduler>,
    items_schedulers: Arc<Vec<Box<dyn LogicalPageScheduler>>>,
    offset_type: DataType,
}

impl ListPageScheduler {
    /// Create a new ListPageScheduler
    ///
    /// # Arguments
    ///
    /// * `offsets_scheduler` The scheduler to load the offsets, arrays should be u64
    /// * `items_schedulers` The schedulers to load the items.  A list page will map
    ///   to exactly one page of offsets but may contain more than one page of items
    /// * `offset_type` The data type of the index.  This must either be Int32Type (in
    ///   which case this will decode to ListType) or Int64Type (in which case this will
    ///   decode to LargeListType).  Other types will result in an error.
    pub fn new(
        offsets_scheduler: Box<dyn LogicalPageScheduler>,
        items_schedulers: Vec<Box<dyn LogicalPageScheduler>>,
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
    ) -> Result<Box<dyn LogicalPageDecoder>> {
        let num_rows = range.end - range.start;
        let num_offsets = num_rows + 1;
        let offsets_range = range.start..(range.end + 1);
        let mut scheduled_offsets = self
            .offsets_scheduler
            .schedule_range(offsets_range, scheduler)?;
        let items_schedulers = self.items_schedulers.clone();
        let scheduler = scheduler.clone();

        // First we schedule, as normal, the I/O for the offsets.  Then we immediately spawn
        // a task to decode those offsets and schedule the I/O for the items.  If we wait until
        // the decode task has launched then we will be delaying the I/O for the items until we
        // need them which is not good.  Better to spend some eager CPU and start loading the
        // items immediately.
        let indirect_fut = tokio::task::spawn(async move {
            scheduled_offsets.wait().await?;
            let decode_task = scheduled_offsets.drain(num_offsets)?;
            let offsets = decode_task.task.decode()?;
            let numeric_offsets = offsets.as_primitive::<Int32Type>();
            let start = numeric_offsets.values()[0] as u32;
            let end = numeric_offsets.values()[numeric_offsets.len() - 1] as u32;

            let mut rows_to_take = end - start;
            let mut rows_to_skip = start;
            let mut item_decoders = Vec::new();

            for item_scheduler in items_schedulers.as_ref() {
                if item_scheduler.num_rows() < rows_to_skip {
                    rows_to_skip -= item_scheduler.num_rows()
                } else {
                    let rows_avail = item_scheduler.num_rows() - rows_to_skip;
                    let to_take = rows_to_take.min(rows_avail);
                    let page_range = rows_to_skip..(rows_to_skip + to_take);
                    // Note that, if we have List<List<...>> then this call will schedule yet another round
                    // of I/O :)
                    item_decoders.push(item_scheduler.schedule_range(page_range, &scheduler)?);
                    rows_to_skip = 0;
                    rows_to_take -= to_take;
                }
            }

            for item in &mut item_decoders {
                // These waits could happen here or as part of ListPageDecoder::wait, it doesn't
                // really matter.
                item.wait().await?;
            }
            Ok(IndirectlyLoaded {
                offsets,
                item_decoders,
            })
        });
        Ok(Box::new(ListPageDecoder {
            offsets: None,
            item_decoders: None,
            num_rows,
            rows_drained: 0,
            unloaded: Some(indirect_fut),
            offset_type: self.offset_type.clone(),
        }))
    }

    // A list page's length is tied to the length of the offsets
    fn num_rows(&self) -> u32 {
        self.offsets_scheduler.num_rows() - 1
    }
}

struct ListPageDecoder {
    unloaded: Option<JoinHandle<Result<IndirectlyLoaded>>>,
    // offsets will have already been decoded as part of the indirect I/O
    offsets: Option<ArrayRef>,
    // Items will not yet be decoded, we at least try and do that part
    // on the decode thread
    item_decoders: Option<VecDeque<Box<dyn LogicalPageDecoder>>>,
    num_rows: u32,
    rows_drained: u32,
    offset_type: DataType,
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
        let item_field = Arc::new(Field::new(
            "item",
            items.data_type().clone(),
            items.is_nullable(),
        ));

        // The offsets are already decoded but they need to be shifted back to 0
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
    fn wait(&mut self) -> BoxFuture<Result<()>> {
        async move {
            let indirectly_loaded = self.unloaded.take().unwrap().await.unwrap()?;
            self.offsets = Some(indirectly_loaded.offsets);
            self.item_decoders = Some(
                indirectly_loaded
                    .item_decoders
                    .into_iter()
                    .collect::<VecDeque<_>>(),
            );
            Ok(())
        }
        .boxed()
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        // We already have the offsets but need to drain the item pages
        let offsets = self
            .offsets
            .as_ref()
            .unwrap()
            .slice(self.rows_drained as usize, num_rows as usize + 1);

        // TODO: This should be u64, not i32
        let offsets_i32 = offsets.as_primitive::<Int32Type>();
        let start = offsets_i32.values()[0];
        let end = offsets_i32.values()[offsets_i32.len() - 1];
        let mut num_items_to_drain = end - start;

        let mut item_decodes = Vec::new();
        let item_decoders = self.item_decoders.as_mut().unwrap();
        while num_items_to_drain > 0 {
            let next_item_page = item_decoders.front_mut().unwrap();
            let avail = next_item_page.avail();
            let to_take = num_items_to_drain.min(avail as i32) as u32;
            num_items_to_drain -= to_take as i32;
            let next_task = next_item_page.drain(to_take)?;

            if !next_task.has_more {
                item_decoders.pop_front();
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
        index_tasks: Option<BoxFuture<'static, Result<Vec<EncodedPage>>>>,
        item_tasks: Option<BoxFuture<'static, Result<Vec<EncodedPage>>>>,
    ) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>> {
        match (index_tasks, item_tasks) {
            (None, None) => None,
            (Some(tasks_fut), None) => Some(tasks_fut),
            (None, Some(tasks_fut)) => Some(tasks_fut),
            (Some(index_tasks), Some(item_tasks)) => Some(
                async move {
                    let mut idx_pages = index_tasks.await?;
                    let item_pages = item_tasks.await?;
                    idx_pages.extend(item_pages);
                    Ok(idx_pages)
                }
                .boxed(),
            ),
        }
    }

    fn wrap_index_encode_tasks(
        tasks: Option<BoxFuture<'static, Result<Vec<EncodedPage>>>>,
    ) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>> {
        tasks.map(|tasks| {
            tasks
                .map_ok(|pages| {
                    pages
                        .into_iter()
                        .map(|page| EncodedPage {
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
                        .collect::<Vec<_>>()
                })
                .boxed()
        })
    }
}

impl FieldEncoder for ListFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>> {
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

    fn flush(&mut self) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>> {
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
mod tests {}
