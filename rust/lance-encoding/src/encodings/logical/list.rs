// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::VecDeque, ops::Range, sync::Arc};

use arrow_array::{
    cast::AsArray,
    new_empty_array,
    types::{Int32Type, Int64Type, UInt64Type},
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeListArray, ListArray, UInt64Array,
};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, Buffer, NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field};
use futures::{future::BoxFuture, FutureExt};
use log::{info, trace};
use tokio::{sync::mpsc, task::JoinHandle};

use lance_core::Result;

use crate::{
    decoder::{DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask},
    encoder::{ArrayEncoder, EncodeTask, EncodedArray, EncodedPage, FieldEncoder},
    format::pb,
    EncodingsIo,
};

use super::primitive::{AccumulationQueue, PrimitiveFieldEncoder};

/// A page scheduler for list fields that encodes offsets in one field and items in another
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
// TODO: Right now we are assuming that list offsets and list items are written at the same time.
// As a result, we know which item pages correspond to which offsets page and there are no item
// pages which overlap two different offsets pages.
//
// We could relax this constraint.  Either each offsets page could store the u64 offset into
// the total range of item pages or each offsets page could store a u64 num_items and a u32
// first_item_page_offset.
#[derive(Debug)]
pub struct ListPageScheduler {
    offsets_scheduler: Box<dyn LogicalPageScheduler>,
    items_schedulers: Arc<Vec<Box<dyn LogicalPageScheduler>>>,
    items_type: DataType,
    offset_type: DataType,
    last_valid_offset: u64,
}

impl ListPageScheduler {
    // Create a new ListPageScheduler
    pub fn new(
        offsets_scheduler: Box<dyn LogicalPageScheduler>,
        items_schedulers: Vec<Box<dyn LogicalPageScheduler>>,
        items_type: DataType,
        // Should be int32 or int64
        offset_type: DataType,
        last_valid_offset: u64,
    ) -> Self {
        match &offset_type {
            DataType::Int32 | DataType::Int64 => {}
            _ => panic!(),
        }
        Self {
            offsets_scheduler,
            items_schedulers: Arc::new(items_schedulers),
            items_type,
            offset_type,
            last_valid_offset,
        }
    }

    /// Given a list of offsets and a list of requested item ranges we need to rewrite the offsets so that
    /// they appear as expected for a list array.  This involves a number of tasks:
    ///
    ///  * Nulls in the offsets are represented by oversize values and these need to be converted to
    ///    the appropriate length
    ///  * For each range we (usually) load N + 1 offsets, so if we have 5 ranges we have 5 extra values
    ///    and we need to drop 4 of those.
    ///  * Ranges may not start at 0 and, while we don't strictly need to, we want to go ahead and normalize
    ///    the offsets so that the first offset is 0.
    ///
    /// Throughout the comments we will consider the following example case:
    ///
    /// The user requests the following ranges of lists: [0..3, 5..6]
    ///
    /// This is a total of 4 lists.  The loaded offsets are [10, 20, 120, 150, 60].  The last valid offset is 99.
    ///
    /// Our desired output offsets are going to be [0, 10, 20, 20, 30] and the item ranges are [0..20] and [50..60]
    /// The validity array is [true, true, false, true]
    fn decode_offsets(
        offsets: &dyn Array,
        offset_ranges: &[Range<u32>],
        last_valid_offset: u64,
    ) -> (VecDeque<Range<u64>>, Vec<u64>, BooleanBuffer) {
        // In our example this is [10, 20, 120, 50, 60]
        let numeric_offsets = offsets.as_primitive::<UInt64Type>();
        // In our example there are 4 total lists
        let total_num_lists = offset_ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u32>();
        let mut normalized_offsets = Vec::with_capacity(total_num_lists as usize);
        let mut validity_buffer = BooleanBufferBuilder::new(total_num_lists as usize);
        // The first output offset is always 0 no matter what
        normalized_offsets.push(0);
        let mut last_normalized_offset = 0;
        let offsets_values = numeric_offsets.values();

        let mut item_ranges = VecDeque::new();
        let mut offsets_offset: u32 = 0;
        // Only the first range is allowed to start with 0
        debug_assert!(offset_ranges.iter().skip(1).all(|r| r.start > 0));
        for range in offset_ranges {
            // The # of lists in this particular range
            let num_lists = range.end - range.start;

            // Because we know the first offset is always 0 we don't store that.  This means we have special
            // logic if a range starts at 0 (we didn't need to read an extra offset value in that case)
            // In our example we enter this special case on the first range (0..3) but not the second (5..6)
            // This means the first range, which has 3 lists, maps to 3 values in our offsets array [10, 20, 120]
            // However, the second range, which has 1 list, maps to 2 values in our offsets array [150, 60]
            let (items_range, offsets_to_norm_start, num_offsets_to_norm) = if range.start == 0 {
                // In our example items start is 0 and items_end is 20
                let first_offset_idx = 0_usize;
                let num_offsets = num_lists as usize;
                let items_start = 0;
                let mut items_end = offsets_values[num_offsets as usize - 1];
                // Repair any null value
                if items_end > last_valid_offset {
                    items_end = items_end - last_valid_offset - 1;
                }
                let items_range = items_start..items_end;

                (items_range, first_offset_idx, num_offsets)
            } else {
                // In our example, offsets_offset will be 3, items_start will be 50, and items_end will
                // be 60
                let first_offset_idx = offsets_offset as usize;
                let num_offsets = num_lists as usize + 1;
                let mut items_start = offsets_values[first_offset_idx];
                if items_start > last_valid_offset {
                    items_start = items_start - last_valid_offset - 1;
                }
                let mut items_end = offsets_values[first_offset_idx + num_offsets - 1];
                if items_end > last_valid_offset {
                    items_end = items_end - last_valid_offset - 1;
                }
                let items_range = items_start..items_end;
                (items_range, first_offset_idx, num_offsets)
            };

            // TODO: Maybe consider writing whether there are nulls or not as part of the
            // page description.  Then we can skip all validity work (and all these if branches
            // comparing with last_valid_offset) when there are no nulls.

            // We calculate validity from all elements but the first (or all elements
            // if this is the special zero-start case)
            //
            // So, in our first pass through, we consider [10, 20, 120] (1 null)
            // In our second pass through we only consider [60] (0 nulls)
            // Note that the 150 is null but we only loaded it to know where the 50-60 list started
            // and it doesn't actually correspond to a list (e.g. list 4 is null but we aren't loading it
            // here)
            let validity_start = if range.start == 0 {
                0
            } else {
                offsets_to_norm_start + 1
            };
            for off in offsets_values
                .slice(validity_start, num_lists as usize)
                .iter()
            {
                validity_buffer.append(*off <= last_valid_offset);
            }

            // In our special case we need to account for the offset 0-first_item
            if range.start == 0 {
                let first_item = offsets_values[0];
                if first_item > last_valid_offset {
                    let normalized = first_item - last_valid_offset - 1;
                    normalized_offsets.push(normalized);
                    last_normalized_offset = normalized;
                } else {
                    normalized_offsets.push(first_item);
                    last_normalized_offset = first_item;
                }
            }

            // Finally, we go through and shift the offsets.  If we just returned them as is (taking care of
            // nulls) we would get [0, 10, 20, 20, 60] but our last list only has 10 items, not 40.
            normalized_offsets.extend(
                offsets_values
                    .slice(offsets_to_norm_start, num_offsets_to_norm)
                    .windows(2)
                    .map(|w| {
                        let start = if w[0] > last_valid_offset {
                            w[0] - last_valid_offset - 1
                        } else {
                            w[0]
                        };
                        let end = if w[1] > last_valid_offset {
                            w[1] - last_valid_offset - 1
                        } else {
                            w[1]
                        };
                        if end < start {
                            panic!("End is less than start in window {:?} with last_valid_offset={} we get start={} and end={}", w, last_valid_offset, start, end);
                        }
                        debug_assert!(end >= start);
                        let length = end - start;
                        last_normalized_offset += length;
                        last_normalized_offset
                    }),
            );
            trace!(
                "List offsets range of {:?} maps to item range {:?}",
                range,
                items_range
            );
            offsets_offset += num_offsets_to_norm as u32;
            item_ranges.push_back(items_range);
        }

        let validity = validity_buffer.finish();
        (item_ranges, normalized_offsets, validity)
    }
}

impl LogicalPageScheduler for ListPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        // TODO: Shortcut here if the request covers the entire range (can be determined by
        // the first_invalid_offset).  If this is the case we don't need any indirect I/O.  We
        // know we need the entirety of the list items.
        let num_rows = ranges.iter().map(|range| range.end - range.start).sum();
        // TODO: Should coalesce here (e.g. if receiving take(&[0, 1, 2]))
        // otherwise we are double-dipping on the offsets scheduling
        let offsets_ranges = ranges
            .iter()
            .map(|range| {
                if range.start == 0 {
                    // If the start is 0 then we don't need to read an extra value because we know
                    // we are starting from 0
                    0..range.end
                } else {
                    // If the start is not 0 we need to read one more offset so we know the length
                    // of the first item
                    (range.start - 1)..range.end
                }
            })
            .collect::<Vec<_>>();
        let num_offsets = offsets_ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum();
        let last_valid_offset = self.last_valid_offset;
        trace!("Scheduling list offsets ranges: {:?}", offsets_ranges);
        // Create a channel for the internal schedule / decode loop that is unique
        // to this page.
        let (tx, mut rx) = mpsc::unbounded_channel();
        self.offsets_scheduler
            .schedule_ranges(&offsets_ranges, scheduler, &tx)?;
        let mut scheduled_offsets = rx.try_recv().unwrap();
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

            let (mut item_ranges, offsets, validity) =
                Self::decode_offsets(offsets.as_ref(), &ranges, last_valid_offset);
            let (tx, mut rx) = mpsc::unbounded_channel();

            trace!(
                "Indirectly scheduling items ranges {:?} from {} list items pages",
                item_ranges,
                items_schedulers.len()
            );

            // This can happen, for example, when there are only empty lists
            if items_schedulers.is_empty() {
                debug_assert!(item_ranges.iter().all(|r| r.start == r.end));
                return Ok(IndirectlyLoaded {
                    item_decoders: Vec::new(),
                    offsets,
                    validity,
                });
            }

            let mut item_schedulers = VecDeque::from_iter(items_schedulers.iter());
            let mut row_offset = 0_u64;
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
                let current_scheduler_end = row_offset + next_scheduler.num_rows() as u64;
                if next_range.start > current_scheduler_end {
                    // All requested items are past this page, continue
                    row_offset += next_scheduler.num_rows() as u64;
                    if !next_item_ranges.is_empty() {
                        next_scheduler.schedule_ranges(&next_item_ranges, &scheduler, &tx)?;
                        next_item_ranges.clear();
                    }
                    next_scheduler = item_schedulers.pop_front().unwrap();
                } else if next_range.end <= current_scheduler_end {
                    // Range entirely contained in current scheduler
                    let page_range = (next_range.start - row_offset) as u32
                        ..(next_range.end - row_offset) as u32;
                    next_item_ranges.push(page_range);
                    if let Some(item_range) = item_ranges.pop_front() {
                        next_range = item_range;
                    } else {
                        // We have processed all pages
                        break;
                    }
                } else {
                    // Range partially contained in current scheduler
                    let page_range =
                        (next_range.start - row_offset) as u32..next_scheduler.num_rows();
                    next_range = current_scheduler_end..next_range.end;
                    next_item_ranges.push(page_range);
                    row_offset += next_scheduler.num_rows() as u64;
                    if !next_item_ranges.is_empty() {
                        next_scheduler.schedule_ranges(&next_item_ranges, &scheduler, &tx)?;
                        next_item_ranges.clear();
                    }
                    next_scheduler = item_schedulers.pop_front().unwrap();
                }
            }
            if !next_item_ranges.is_empty() {
                next_scheduler.schedule_ranges(&next_item_ranges, &scheduler, &tx)?;
            }
            let mut item_decoders = Vec::new();
            drop(tx);
            // TODO(urgent): A single list page can have multiple item pages and they could be huge.
            // For example, 1Mi lists with 100MiB of data in each list.
            //
            // The 1Mi offsets would fit neatly into a single page
            //
            // However, the 100GiB of data would be spread across many pages.
            //
            // This is all ok.  THhe problem is that we are currently waiting for all pages
            // to finish loading.  If the read batch size is small (e.g. 10) then we should
            // only wait for a few pages before returning a batch.
            //
            // TODO(not-so-urgent): Even further in the future we should be able to automatically
            // shrink the read batch size if we detect the batches are going to be huge (maybe
            // even achieve this with a read_batch_bytes parameter, though some estimation may
            // still be required)
            while let Some(mut item_decoder) = rx.recv().await {
                item_decoder.wait(item_decoder.unawaited(), &mut rx).await?;
                item_decoders.push(item_decoder);
            }

            Ok(IndirectlyLoaded {
                offsets,
                validity,
                item_decoders,
            })
        });
        sink.send(Box::new(ListPageDecoder {
            offsets: Vec::new(),
            validity: BooleanBuffer::new(Buffer::from_vec(Vec::<u8>::default()), 0, 0),
            item_decoders: VecDeque::new(),
            num_rows,
            rows_drained: 0,
            unloaded: Some(indirect_fut),
            items_type: self.items_type.clone(),
            offset_type: self.offset_type.clone(),
        }))
        .unwrap();
        Ok(())
    }

    fn num_rows(&self) -> u32 {
        self.offsets_scheduler.num_rows()
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
    // offsets and validity will have already been decoded as part of the indirect I/O
    offsets: Vec<u64>,
    validity: BooleanBuffer,
    // Items will not yet be decoded, we at least try and do that part
    // on the decode thread
    item_decoders: VecDeque<Box<dyn LogicalPageDecoder>>,
    num_rows: u32,
    rows_drained: u32,
    items_type: DataType,
    offset_type: DataType,
}

struct ListDecodeTask {
    offsets: Vec<u64>,
    validity: BooleanBuffer,
    items: Vec<Box<dyn DecodeArrayTask>>,
    items_type: DataType,
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
        let items = if item_refs.is_empty() {
            // This can happen if we have are building an array made only of empty lists
            new_empty_array(&self.items_type)
        } else {
            arrow_select::concat::concat(&item_refs)?
        };
        // TODO: we default to nullable true here, should probably use the nullability given to
        // us from the input schema
        let item_field = Arc::new(Field::new("item", self.items_type.clone(), true));

        // The offsets are already decoded but they need to be shifted back to 0 and cast
        // to the appropriate type
        // TODO: This shift is not strictly required since a list array's offsets don't have
        // to start at zero but doing the shift makes testing easier.
        //
        // Although, in some cases, the shift IS strictly required since the unshifted offsets
        // may cross i32::MAX even though the shifted offsets do not
        let offsets = UInt64Array::from(self.offsets);
        let validity = NullBuffer::new(self.validity);
        let validity = if validity.null_count() == 0 {
            None
        } else {
            Some(validity)
        };
        let min_offset = UInt64Array::new_scalar(offsets.value(0));
        let offsets = arrow_arith::numeric::sub(&offsets, &min_offset)?;
        match &self.offset_type {
            DataType::Int32 => {
                let offsets = arrow_cast::cast(&offsets, &DataType::Int32)?;
                let offsets_i32 = offsets.as_primitive::<Int32Type>();
                let offsets = OffsetBuffer::new(offsets_i32.values().clone());

                Ok(Arc::new(ListArray::try_new(
                    item_field, offsets, items, validity,
                )?))
            }
            DataType::Int64 => {
                let offsets = arrow_cast::cast(&offsets, &DataType::Int64)?;
                let offsets_i64 = offsets.as_primitive::<Int64Type>();
                let offsets = OffsetBuffer::new(offsets_i64.values().clone());

                Ok(Arc::new(LargeListArray::try_new(
                    item_field, offsets, items, validity,
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
                let indirectly_loaded = self.unloaded.take().unwrap().await;
                if indirectly_loaded.is_err() {
                    match indirectly_loaded.unwrap_err().try_into_panic() {
                        Ok(err) => std::panic::resume_unwind(err),
                        Err(err) => panic!("{:?}", err),
                    };
                }
                let indirectly_loaded = indirectly_loaded.unwrap()?;
                self.offsets = indirectly_loaded.offsets;
                self.validity = indirectly_loaded.validity;
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
        let mut actual_num_rows = num_rows;
        let item_start = self.offsets[self.rows_drained as usize];
        if self.offset_type != DataType::Int64 {
            // We might not be able to drain `num_rows` because that request might contain more than 2^31 items
            // so we need to figure out how many rows we can actually drain.
            while actual_num_rows > 0 {
                let num_items =
                    self.offsets[(self.rows_drained + actual_num_rows) as usize] - item_start;
                if num_items <= i32::MAX as u64 {
                    break;
                }
                // TODO: This could be slow.  Maybe faster to start from zero?
                actual_num_rows -= 1;
            }
        }
        if actual_num_rows < num_rows {
            info!("Only decoding {} rows instead of {} because total data size would exceed 2^31 items", actual_num_rows, num_rows);
        }
        let offsets = self.offsets
            [self.rows_drained as usize..(self.rows_drained + actual_num_rows + 1) as usize]
            .to_vec();
        let validity = self
            .validity
            .slice(self.rows_drained as usize, actual_num_rows as usize);
        let start = offsets[0];
        let end = offsets[offsets.len() - 1];
        let mut num_items_to_drain = end - start;

        let mut item_decodes = Vec::new();
        while num_items_to_drain > 0 {
            let next_item_page = self.item_decoders.front_mut().unwrap();
            let avail = next_item_page.avail();
            let to_take = num_items_to_drain.min(avail as u64) as u32;
            num_items_to_drain -= to_take as u64;
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
                validity,
                items: item_decodes,
                items_type: self.items_type.clone(),
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
    offsets: Vec<u64>,
    validity: BooleanBuffer,
    item_decoders: Vec<Box<dyn LogicalPageDecoder>>,
}

impl std::fmt::Debug for IndirectlyLoaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndirectlyLoaded")
            .field("offsets", &self.offsets)
            .field("validity", &self.validity)
            .finish()
    }
}

/// An encoder for list offsets that "stitches" offsets and encodes nulls into the offsets
///
/// If we need to encode several list arrays into a single page then we need to "stitch" the offsets
/// For example, imagine we have list arrays [[0, 1], [2]] and [[3, 4, 5]].
///
/// We will have offset arrays [0, 2, 3] and [0, 3].  We don't want to encode [0, 2, 3, 0, 3].  What
/// we want is [0, 2, 3, 6]
///
/// This encoder also handles validity by converting a null value into an oversized offset.  For example,
/// if we have four lists with offsets [0, 20, 20, 20, 30] and the list at index 2 is null (note that
/// the list at index 1 is empty) then we turn this into offsets [0, 20, 20, 51, 30].  We replace a null
/// offset with previous_offset + max_offset + 1.  This makes it possible to load a single item from the
/// list array.
///
/// These offsets are always stored on disk as a u64 array.  First, this is because its simply much more
/// likely than one expects that this is needed, even if our lists are not massive.  This is because we
/// only write an offsets page when we have enough data.  This means we will probably accumulate a million
/// offsets or more before we bother to write a page. If our lists have a few thousand items a piece then
/// we end up passing the u32::MAX boundary.
///
/// The second reason is that list offsets are very easily compacted with delta + bit packing and so those
/// u64 offsets should easily be shrunk down before being put on disk.
///
/// This encoder can encode both lists and large lists.  It can decode the resulting column into either type
/// as well. (TODO: Test and enable large lists)
///
/// You can even write as a large list and decode as a regular list (as long as no single list has more than
/// 2^31 items) or vice versa.  You could even encode a mixed stream of list and large list (but unclear that
/// would ever be useful)
#[derive(Debug)]
struct ListOffsetsEncoder {
    // An accumulation queue, we insert both offset arrays and validity arrays into this queue
    accumulation_queue: AccumulationQueue,
    // The inner encoder of offset values
    inner_encoder: Arc<dyn ArrayEncoder>,
    column_index: u32,
}

impl ListOffsetsEncoder {
    fn new(cache_bytes: u64, keep_original_array: bool, column_index: u32) -> Self {
        Self {
            accumulation_queue: AccumulationQueue::new(
                cache_bytes,
                column_index,
                keep_original_array,
            ),
            inner_encoder: PrimitiveFieldEncoder::array_encoder_from_data_type(&DataType::UInt64)
                .unwrap()
                .into(),
            column_index,
        }
    }

    /// Given a list array, return the offsets as a standalone ArrayRef (either an Int32Array or Int64Array)
    fn extract_offsets(list_arr: &dyn Array) -> ArrayRef {
        match list_arr.data_type() {
            DataType::List(_) => {
                let offsets = list_arr.as_list::<i32>().offsets().clone();
                Arc::new(Int32Array::new(offsets.into_inner(), None))
            }
            DataType::LargeList(_) => {
                let offsets = list_arr.as_list::<i64>().offsets().clone();
                Arc::new(Int64Array::new(offsets.into_inner(), None))
            }
            _ => panic!(),
        }
    }

    /// Converts the validity of a list array into a boolean array.  If there is no validity information
    /// then this is an empty boolean array.
    fn extract_validity(list_arr: &dyn Array) -> ArrayRef {
        if let Some(validity) = list_arr.nulls() {
            Arc::new(BooleanArray::new(validity.inner().clone(), None))
        } else {
            // We convert None validity into an empty array because the accumulation queue can't
            // handle Option<ArrayRef>
            new_empty_array(&DataType::Boolean)
        }
    }

    fn make_encode_task(&self, arrays: Vec<ArrayRef>) -> EncodeTask {
        let inner_encoder = self.inner_encoder.clone();
        let column_idx = self.column_index;
        // At this point we should have 2*N arrays where the 0, 2, ... arrays are integer offsets
        // and the 1, 3, ... arrays are boolean
        let offset_arrays = arrays.iter().step_by(2).cloned().collect::<Vec<_>>();
        let validity_arrays = arrays.into_iter().skip(1).step_by(2).collect::<Vec<_>>();

        tokio::task::spawn(async move {
            let num_rows =
                offset_arrays.iter().map(|arr| arr.len()).sum::<usize>() - offset_arrays.len();
            let num_rows = num_rows as u32;
            let mut buffer_index = 0;
            let array = Self::do_encode(
                offset_arrays,
                validity_arrays,
                &mut buffer_index,
                num_rows,
                inner_encoder,
            )?;
            Ok(EncodedPage {
                array,
                num_rows,
                column_idx,
            })
        })
        .map(|res_res| res_res.unwrap())
        .boxed()
    }

    fn maybe_encode_offsets_and_validity(&mut self, list_arr: &dyn Array) -> Option<EncodeTask> {
        let offsets = Self::extract_offsets(list_arr);
        let validity = Self::extract_validity(list_arr);
        // Either inserting the offsets OR inserting the validity could cause the
        // accumulation queue to fill up
        if let Some(mut arrays) = self.accumulation_queue.insert(offsets) {
            arrays.push(validity);
            Some(self.make_encode_task(arrays))
        } else if let Some(arrays) = self.accumulation_queue.insert(validity) {
            Some(self.make_encode_task(arrays))
        } else {
            None
        }
    }

    fn flush(&mut self) -> Option<EncodeTask> {
        if let Some(arrays) = self.accumulation_queue.flush() {
            Some(self.make_encode_task(arrays))
        } else {
            None
        }
    }

    // Get's the total number of items covered by an array of offsets (keeping in
    // mind that the first offset may not be zero)
    fn get_offset_span(array: &dyn Array) -> u64 {
        match array.data_type() {
            DataType::Int32 => {
                let arr_i32 = array.as_primitive::<Int32Type>();
                (arr_i32.value(arr_i32.len() - 1) - arr_i32.value(0)) as u64
            }
            DataType::Int64 => {
                let arr_i64 = array.as_primitive::<Int64Type>();
                (arr_i64.value(arr_i64.len() - 1) - arr_i64.value(0)) as u64
            }
            _ => panic!(),
        }
    }

    // This is where we do the work to actually shift the offsets and encode nulls
    // Note that the output is u64 and the input could be i32 OR i64.
    fn extend_offsets_vec_u64(
        dest: &mut Vec<u64>,
        offsets: &dyn Array,
        validity: Option<&BooleanArray>,
        base: u64,
        end: u64,
    ) {
        match offsets.data_type() {
            DataType::Int32 => {
                let offsets_i32 = offsets.as_primitive::<Int32Type>();
                let start = offsets_i32.value(0) as u64;
                if let Some(validity) = validity {
                    dest.extend(
                        offsets_i32
                            .values()
                            .iter()
                            .skip(1)
                            .zip(validity.values().iter())
                            .map(|(&off, valid)| {
                                let end = if valid { 0 } else { end + 1 };
                                off as u64 - start + base + end
                            }),
                    );
                } else {
                    dest.extend(
                        offsets_i32
                            .values()
                            .iter()
                            .skip(1)
                            .map(|&v| v as u64 - start + base),
                    );
                }
            }
            DataType::Int64 => {
                let offsets_i64 = offsets.as_primitive::<Int32Type>();
                let start = offsets_i64.value(0) as u64;
                if let Some(validity) = validity {
                    dest.extend(
                        offsets_i64
                            .values()
                            .iter()
                            .skip(1)
                            .zip(validity.values().iter())
                            .map(|(&off, valid)| {
                                let end = if valid { 0 } else { end + 1 };
                                off as u64 - start + base + end
                            }),
                    )
                } else {
                    dest.extend(
                        offsets_i64
                            .values()
                            .iter()
                            .skip(1)
                            .map(|&v| v as u64 - start + base),
                    );
                }
            }
            _ => panic!("Invalid list offsets data type {:?}", offsets.data_type()),
        }
    }

    fn do_encode_u64(
        offset_arrays: Vec<ArrayRef>,
        validity: Vec<Option<&BooleanArray>>,
        num_offsets: u32,
        total_span: u64,
        buffer_index: &mut u32,
        inner_encoder: Arc<dyn ArrayEncoder>,
    ) -> Result<EncodedArray> {
        let mut offsets = Vec::with_capacity(num_offsets as usize);
        for (offsets_arr, validity_arr) in offset_arrays.iter().zip(validity) {
            let mut last_prev_offset = offsets.last().copied().unwrap_or(0);
            if last_prev_offset > total_span {
                last_prev_offset = last_prev_offset - total_span - 1;
            }
            Self::extend_offsets_vec_u64(
                &mut offsets,
                &offsets_arr,
                validity_arr,
                last_prev_offset,
                total_span,
            );
        }
        inner_encoder.encode(&[Arc::new(UInt64Array::from(offsets))], buffer_index)
    }

    fn do_encode(
        offset_arrays: Vec<ArrayRef>,
        validity_arrays: Vec<ArrayRef>,
        buffer_index: &mut u32,
        num_offsets: u32,
        inner_encoder: Arc<dyn ArrayEncoder>,
    ) -> Result<EncodedArray> {
        let validity_arrays = validity_arrays
            .iter()
            .map(|v| {
                if v.is_empty() {
                    None
                } else {
                    Some(v.as_boolean())
                }
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(offset_arrays.len(), validity_arrays.len());
        let total_span = offset_arrays
            .iter()
            .map(|arr| Self::get_offset_span(arr.as_ref()))
            .sum::<u64>();
        let encoded_offsets = Self::do_encode_u64(
            offset_arrays,
            validity_arrays,
            num_offsets,
            total_span,
            buffer_index,
            inner_encoder,
        )?;
        Ok(EncodedArray {
            buffers: encoded_offsets.buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::List(Box::new(
                    pb::List {
                        offsets: Some(Box::new(encoded_offsets.encoding)),
                        first_invalid_offset: total_span,
                    },
                ))),
            },
        })
    }
}

pub struct ListFieldEncoder {
    offsets_encoder: ListOffsetsEncoder,
    items_encoder: Box<dyn FieldEncoder>,
}

impl ListFieldEncoder {
    pub fn new(
        items_encoder: Box<dyn FieldEncoder>,
        cache_bytes_per_columns: u64,
        keep_original_array: bool,
        column_index: u32,
    ) -> Self {
        Self {
            offsets_encoder: ListOffsetsEncoder::new(
                cache_bytes_per_columns,
                keep_original_array,
                column_index,
            ),
            items_encoder,
        }
    }

    fn combine_tasks(
        offsets_tasks: Result<Vec<EncodeTask>>,
        item_tasks: Result<Vec<EncodeTask>>,
    ) -> Result<Vec<EncodeTask>> {
        let mut all_tasks = offsets_tasks?;
        let item_tasks = item_tasks?;
        all_tasks.extend(item_tasks);
        Ok(all_tasks)
    }
}

impl FieldEncoder for ListFieldEncoder {
    fn maybe_encode(&mut self, array: ArrayRef) -> Result<Vec<EncodeTask>> {
        // The list may have an offset / shorter length which means the underlying
        // values array could be longer than what we need to encode and so we need
        // to slice down to the region of interest.
        let items = match array.data_type() {
            DataType::List(_) => {
                let list_arr = array.as_list::<i32>();
                let items_start = list_arr.value_offsets()[list_arr.offset()] as usize;
                let items_end =
                    list_arr.value_offsets()[list_arr.offset() + list_arr.len()] as usize;
                list_arr
                    .values()
                    .slice(items_start, items_end - items_start)
            }
            DataType::LargeList(_) => {
                let list_arr = array.as_list::<i64>();
                let items_start = list_arr.value_offsets()[list_arr.offset()] as usize;
                let items_end =
                    list_arr.value_offsets()[list_arr.offset() + list_arr.len()] as usize;
                list_arr
                    .values()
                    .slice(items_start, items_end - items_start)
            }
            _ => panic!(),
        };
        let offsets_tasks = self
            .offsets_encoder
            .maybe_encode_offsets_and_validity(array.as_ref())
            .map(|task| Ok(vec![task]))
            .unwrap_or_else(|| Ok(Vec::default()));
        let item_tasks = self.items_encoder.maybe_encode(items);
        Self::combine_tasks(offsets_tasks, item_tasks)
    }

    fn flush(&mut self) -> Result<Vec<EncodeTask>> {
        let offsets_tasks = self
            .offsets_encoder
            .flush()
            .map(|task| Ok(vec![task]))
            .unwrap_or_else(|| Ok(Vec::default()));
        let item_tasks = self.items_encoder.flush();
        Self::combine_tasks(offsets_tasks, item_tasks)
    }

    fn num_columns(&self) -> u32 {
        2
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_array::{
        builder::{Int32Builder, ListBuilder},
        ArrayRef, BooleanArray, ListArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases,
    };

    fn make_list_type(inner_type: DataType) -> DataType {
        DataType::List(Arc::new(Field::new("item", inner_type, true)))
    }

    #[test_log::test(tokio::test)]
    async fn test_list() {
        let field = Field::new("", make_list_type(DataType::Int32), true);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_simple_list() {
        let items_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(items_builder);
        list_builder.append_value([Some(1), Some(2), Some(3)]);
        list_builder.append_value([Some(4), Some(5)]);
        list_builder.append_null();
        list_builder.append_value([Some(6), Some(7), Some(8)]);
        let list_array = list_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![1, 3]);
        check_round_trip_encoding_of_data(vec![Arc::new(list_array)], &test_cases).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_lists() {
        // When encoding a list of empty lists there are no items to encode
        // which is strange and we want to ensure we handle it
        let items_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(items_builder);
        list_builder.append(true);
        list_builder.append_null();
        list_builder.append(true);
        let list_array = list_builder.finish();

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![Arc::new(list_array)], &test_cases).await;
    }

    #[test_log::test(tokio::test)]
    #[ignore] // This test is quite slow in debug mode
    async fn test_jumbo_list() {
        // This is an overflow test.  We have a list of lists where each list
        // has 1Mi items.  We encode 5000 of these lists and so we have over 4Gi in the
        // offsets range
        let items = BooleanArray::new_null(1024 * 1024);
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 1024 * 1024]));
        let list_arr = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Boolean, true)),
            offsets,
            Arc::new(items),
            None,
        )) as ArrayRef;
        let arrs = vec![list_arr; 5000];

        // We can't validate because our validation relies on concatenating all input arrays
        let test_cases = TestCases::default().without_validation();
        check_round_trip_encoding_of_data(arrs, &test_cases).await;
    }
}
