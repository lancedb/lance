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
use arrow_schema::{DataType, Field, Fields};
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};
use tokio::task::JoinHandle;

use lance_core::{Error, Result};

use crate::{
    decoder::{
        DecodeArrayTask, DecoderMessage, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask,
        SchedulerContext,
    },
    encoder::{
        ArrayEncoder, CoreBufferEncodingStrategy, EncodeTask, EncodedArray, EncodedPage,
        FieldEncoder,
    },
    encodings::{
        logical::r#struct::SimpleStructScheduler,
        physical::{
            basic::BasicEncoder,
            value::{CompressionScheme, ValueEncoder},
        },
    },
    format::pb,
};

use super::{primitive::AccumulationQueue, r#struct::SimpleStructDecoder};

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
#[derive(Debug)]
pub struct ListPageScheduler {
    offsets_scheduler: Arc<dyn LogicalPageScheduler>,
    items_schedulers: Arc<Vec<Arc<dyn LogicalPageScheduler>>>,
    items_type: DataType,
    offset_type: DataType,
    null_offset_adjustment: u64,
    // Two list pages might share an items page.  For example, when given a List<Struct<...>>
    // the struct items page is often very large (if there are no nulls in the struct it will
    // be 1 giant page) since it is just a header page.  This means the second list page starts
    // at some offset into the items page which we record here.
    first_items_page_offset: u32,
}

impl ListPageScheduler {
    // Create a new ListPageScheduler
    pub fn new(
        offsets_scheduler: Arc<dyn LogicalPageScheduler>,
        items_schedulers: Vec<Arc<dyn LogicalPageScheduler>>,
        items_type: DataType,
        // Should be int32 or int64
        offset_type: DataType,
        null_offset_adjustment: u64,
        first_items_page_offset: u32,
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
            null_offset_adjustment,
            first_items_page_offset,
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
    /// The null_offset_adjustment will be 100.
    ///
    /// Our desired output offsets are going to be [0, 10, 20, 20, 30] and the item ranges are [0..20] and [50..60]
    /// The validity array is [true, true, false, true]
    fn decode_offsets(
        offsets: &dyn Array,
        offset_ranges: &[Range<u32>],
        null_offset_adjustment: u64,
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
        // All ranges should be non-empty
        debug_assert!(offset_ranges.iter().all(|r| r.end > r.start));
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
                let items_end = offsets_values[num_offsets - 1] % null_offset_adjustment;
                let items_range = items_start..items_end;
                (items_range, first_offset_idx, num_offsets)
            } else {
                // In our example, offsets_offset will be 3, items_start will be 50, and items_end will
                // be 60
                let first_offset_idx = offsets_offset as usize;
                let num_offsets = num_lists as usize + 1;
                let items_start = offsets_values[first_offset_idx] % null_offset_adjustment;
                let items_end =
                    offsets_values[first_offset_idx + num_offsets - 1] % null_offset_adjustment;
                let items_range = items_start..items_end;
                (items_range, first_offset_idx, num_offsets)
            };

            // TODO: Maybe consider writing whether there are nulls or not as part of the
            // page description.  Then we can skip all validity work.  Not clear if that will
            // be any benefit though.

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
                validity_buffer.append(*off < null_offset_adjustment);
            }

            // In our special case we need to account for the offset 0-first_item
            if range.start == 0 {
                let first_item = offsets_values[0] % null_offset_adjustment;
                normalized_offsets.push(first_item);
                last_normalized_offset = first_item;
            }

            // Finally, we go through and shift the offsets.  If we just returned them as is (taking care of
            // nulls) we would get [0, 10, 20, 20, 60] but our last list only has 10 items, not 40 and so we
            // need to shift that 60 to a 40.
            normalized_offsets.extend(
                offsets_values
                    .slice(offsets_to_norm_start, num_offsets_to_norm)
                    .windows(2)
                    .map(|w| {
                        let start = w[0] % null_offset_adjustment;
                        let end = w[1] % null_offset_adjustment;
                        if end < start {
                            panic!("End is less than start in window {:?} with null_offset_adjustment={} we get start={} and end={}", w, null_offset_adjustment, start, end);
                        }
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
            if !items_range.is_empty() {
                item_ranges.push_back(items_range);
            }
        }

        let validity = validity_buffer.finish();
        (item_ranges, normalized_offsets, validity)
    }
}

impl LogicalPageScheduler for ListPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        context: &mut SchedulerContext,
        top_level_row: u64,
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
        let null_offset_adjustment = self.null_offset_adjustment;
        trace!("Scheduling list offsets ranges: {:?}", offsets_ranges);
        // Create a channel for the offsets
        let mut temporary = context.temporary(None);
        self.offsets_scheduler
            .schedule_ranges(&offsets_ranges, &mut temporary, top_level_row)?;
        let offset_decoders = temporary.into_decoders();
        let num_offset_decoders = offset_decoders.len();
        let mut scheduled_offsets =
            offset_decoders
                .into_iter()
                .next()
                .ok_or_else(|| Error::Internal {
                    message: format!("scheduling offsets yielded {} pages", num_offset_decoders),
                    location: location!(),
                })?;

        let items_schedulers = self.items_schedulers.clone();
        let ranges = ranges.to_vec();
        let items_type = self.items_type.clone();
        let first_items_page_offset = self.first_items_page_offset;
        let mut indirect_context = context.temporary(Some(top_level_row));

        // First we schedule, as normal, the I/O for the offsets.  Then we immediately spawn
        // a task to decode those offsets and schedule the I/O for the items AND wait for
        // the items.  If we wait until the decode task has launched then we will be delaying
        // the I/O for the items until we need them which is not good.  Better to spend some
        // eager CPU and start loading the items immediately.
        let indirect_fut = tokio::task::spawn(async move {
            // We know the offsets are a primitive array and thus will not need additional
            // pages.  We can use a dummy receiver to match the decoder API
            scheduled_offsets.wait(num_rows).await?;
            let decode_task = scheduled_offsets.drain(num_offsets)?;
            let offsets = decode_task.task.decode()?;

            let (item_ranges, offsets, validity) =
                Self::decode_offsets(offsets.as_ref(), &ranges, null_offset_adjustment);

            trace!(
                "Indirectly scheduling items ranges {:?} from {} list items pages",
                item_ranges,
                items_schedulers.len()
            );

            // All requested lists are empty
            if items_schedulers.is_empty() || item_ranges.is_empty() {
                debug_assert!(item_ranges.iter().all(|r| r.start == r.end));
                return Ok(IndirectlyLoaded {
                    root_decoder: None,
                    offsets,
                    validity,
                });
            }

            // Create a new root scheduler, which has one column, which is our items data
            let indirect_root_scheduler = SimpleStructScheduler::new_root(
                vec![items_schedulers.as_ref().clone()],
                Fields::from(vec![Field::new("item", items_type, true)]),
            );

            let patched_item_ranges = item_ranges
                .clone()
                .into_iter()
                .map(|range| {
                    (range.start + first_items_page_offset as u64)
                        ..(range.end + first_items_page_offset as u64)
                })
                .collect::<Vec<_>>();

            // Immediately run the scheduling and process the decode messages (we could start
            // a new thread here for decode to run in parallel but, at the moment, that seems
            // like overkill)
            indirect_root_scheduler.schedule_ranges_u64(
                &patched_item_ranges,
                &mut indirect_context,
                top_level_row,
            )?;
            let mut root_decoder =
                indirect_root_scheduler.new_root_decoder_ranges(&patched_item_ranges);

            for message in indirect_context.into_messages() {
                if let DecoderMessage::Decoder(decoder) = message {
                    debug_assert!(!decoder.path.is_empty());
                    root_decoder.accept_child(decoder)?;
                }
            }

            Ok(IndirectlyLoaded {
                offsets,
                validity,
                root_decoder: Some(root_decoder),
            })
        });
        let data_type = match &self.offset_type {
            DataType::Int32 => {
                DataType::List(Arc::new(Field::new("item", self.items_type.clone(), true)))
            }
            DataType::Int64 => {
                DataType::LargeList(Arc::new(Field::new("item", self.items_type.clone(), true)))
            }
            _ => panic!("Unexpected offset type {}", self.offset_type),
        };
        context.emit(Box::new(ListPageDecoder {
            offsets: Vec::new(),
            validity: BooleanBuffer::new(Buffer::from_vec(Vec::<u8>::default()), 0, 0),
            item_decoder: None,
            rows_drained: 0,
            lists_available: 0,
            num_rows,
            unloaded: Some(indirect_fut),
            items_type: self.items_type.clone(),
            offset_type: self.offset_type.clone(),
            data_type,
        }));
        Ok(())
    }

    fn num_rows(&self) -> u32 {
        self.offsets_scheduler.num_rows()
    }

    fn schedule_take(
        &self,
        indices: &[u32],
        context: &mut SchedulerContext,
        top_level_row: u64,
    ) -> Result<()> {
        trace!("Scheduling list offsets for {} indices", indices.len());
        self.schedule_ranges(
            &indices
                .iter()
                .map(|&idx| idx..(idx + 1))
                .collect::<Vec<_>>(),
            context,
            top_level_row,
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
#[derive(Debug)]
struct ListPageDecoder {
    unloaded: Option<JoinHandle<Result<IndirectlyLoaded>>>,
    // offsets and validity will have already been decoded as part of the indirect I/O
    offsets: Vec<u64>,
    validity: BooleanBuffer,
    item_decoder: Option<SimpleStructDecoder>,
    lists_available: u32,
    num_rows: u32,
    rows_drained: u32,
    items_type: DataType,
    offset_type: DataType,
    data_type: DataType,
}

struct ListDecodeTask {
    offsets: Vec<u64>,
    validity: BooleanBuffer,
    // Will be None if there are no items (all empty / null lists)
    items: Option<Box<dyn DecodeArrayTask>>,
    items_type: DataType,
    offset_type: DataType,
}

impl DecodeArrayTask for ListDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let items = self
            .items
            .map(|items| {
                // When we run the indirect I/O we wrap things in a struct array with a single field
                // named "item".  We can unwrap that now.
                let wrapped_items = items.decode()?;
                Result::Ok(wrapped_items.as_struct().column(0).clone())
            })
            .unwrap_or_else(|| Ok(new_empty_array(&self.items_type)))?;

        // TODO: we default to nullable true here, should probably use the nullability given to
        // us from the input schema
        let item_field = Arc::new(Field::new("item", self.items_type.clone(), true));

        // The offsets are already decoded but they need to be shifted back to 0 and cast
        // to the appropriate type
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
    fn wait(&mut self, num_rows: u32) -> BoxFuture<Result<()>> {
        async move {
            // wait for the indirect I/O to finish, run the scheduler for the indirect
            // I/O and then wait for enough items to arrive
            if self.unloaded.is_some() {
                trace!("List scheduler needs to wait for indirect I/O to complete");
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
                self.item_decoder = indirectly_loaded.root_decoder;
            }
            trace!(
                "List decoder is waiting for {} rows and {} are already available and {} are unawaited",
                num_rows,
                self.lists_available,
                self.num_rows - self.rows_drained
            );
            if self.lists_available >= num_rows {
                self.lists_available -= num_rows;
                return Ok(());
            }
            let num_rows = num_rows - self.lists_available;
            self.lists_available = 0;
            let offset_wait_start = self.rows_drained + self.lists_available;
            let item_start = self.offsets[offset_wait_start as usize];
            let mut items_needed =
                self.offsets[offset_wait_start as usize + num_rows as usize] - item_start;
            if items_needed > 0 {
                // First discount any already available items
                let items_already_available = self.item_decoder.as_mut().unwrap().avail_u64();
                trace!(
                    "List's items decoder needs {} items and already has {} items available",
                    items_needed,
                    items_already_available,
                );
                items_needed = items_needed.saturating_sub(items_already_available);
                if items_needed > 0 {
                    self.item_decoder.as_mut().unwrap().wait_u64(items_needed).await?;
                }
            }
            // This is technically undercounting a little.  It's possible that we loaded a big items
            // page with many items and then only needed a few of them for the requested lists.  However,
            // to find the exact number of lists that are available we would need to walk through the item
            // lengths and it's faster to just undercount here.
            self.lists_available += num_rows;
            Ok(())
        }
        .boxed()
    }

    fn unawaited(&self) -> u32 {
        self.num_rows - self.lists_available - self.rows_drained
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        self.lists_available -= num_rows;
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
                // TODO: This could be slow.  Maybe faster to start from zero or do binary search.  Investigate when
                // actually adding support for smaller than requested batches
                actual_num_rows -= 1;
            }
        }
        if actual_num_rows < num_rows {
            // TODO: We should be able to automatically
            // shrink the read batch size if we detect the batches are going to be huge (maybe
            // even achieve this with a read_batch_bytes parameter, though some estimation may
            // still be required)
            return Err(Error::NotSupported { source: format!("loading a batch of {} lists would require creating an array with over i32::MAX items and we don't yet support returning smaller than requested batches", num_rows).into(), location: location!() });
        }
        let offsets = self.offsets
            [self.rows_drained as usize..(self.rows_drained + actual_num_rows + 1) as usize]
            .to_vec();
        let validity = self
            .validity
            .slice(self.rows_drained as usize, actual_num_rows as usize);
        let start = offsets[0];
        let end = offsets[offsets.len() - 1];
        let num_items_to_drain = end - start;

        let item_decode = if num_items_to_drain == 0 {
            None
        } else {
            self.item_decoder
                .as_mut()
                .map(|item_decoder| Result::Ok(item_decoder.drain_u64(num_items_to_drain)?.task))
                .transpose()?
        };

        self.rows_drained += num_rows;
        Ok(NextDecodeTask {
            has_more: self.avail() > 0 || self.unawaited() > 0,
            num_rows,
            task: Box::new(ListDecodeTask {
                offsets,
                validity,
                items: item_decode,
                items_type: self.items_type.clone(),
                offset_type: self.offset_type.clone(),
            }) as Box<dyn DecodeArrayTask>,
        })
    }

    fn avail(&self) -> u32 {
        self.lists_available
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

struct IndirectlyLoaded {
    offsets: Vec<u64>,
    validity: BooleanBuffer,
    root_decoder: Option<SimpleStructDecoder>,
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
            inner_encoder: Arc::new(BasicEncoder::new(Box::new(
                ValueEncoder::try_new(Arc::new(CoreBufferEncodingStrategy {
                    compression_scheme: CompressionScheme::None,
                }))
                .unwrap(),
            ))),
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
        // At this point we should have 2*N arrays where the even-indexed arrays are integer offsets
        // and the odd-indexed arrays are boolean validity bitmaps
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
        // The offset of this list into the destination
        base: u64,
        null_offset_adjustment: u64,
    ) {
        match offsets.data_type() {
            DataType::Int32 => {
                let offsets_i32 = offsets.as_primitive::<Int32Type>();
                let start = offsets_i32.value(0) as u64;
                // If we want to take a list from start..X and change it into
                // a list from end..X then we need to add (base - start) to all elements
                // Note that `modifier` may be negative but (item + modifier) will always be >= 0
                let modifier = base as i64 - start as i64;
                if let Some(validity) = validity {
                    dest.extend(
                        offsets_i32
                            .values()
                            .iter()
                            .skip(1)
                            .zip(validity.values().iter())
                            .map(|(&off, valid)| {
                                (off as i64 + modifier) as u64
                                    + (!valid as u64 * null_offset_adjustment)
                            }),
                    );
                } else {
                    dest.extend(
                        offsets_i32
                            .values()
                            .iter()
                            .skip(1)
                            // Subtract by `start` so offsets start at 0
                            .map(|&v| (v as i64 + modifier) as u64),
                    );
                }
            }
            DataType::Int64 => {
                let offsets_i64 = offsets.as_primitive::<Int64Type>();
                let start = offsets_i64.value(0) as u64;
                // If we want to take a list from start..X and change it into
                // a list from end..X then we need to add (base - start) to all elements
                // Note that `modifier` may be negative but (item + modifier) will always be >= 0
                let modifier = base as i64 - start as i64;
                if let Some(validity) = validity {
                    dest.extend(
                        offsets_i64
                            .values()
                            .iter()
                            .skip(1)
                            .zip(validity.values().iter())
                            .map(|(&off, valid)| {
                                (off + modifier) as u64 + (!valid as u64 * null_offset_adjustment)
                            }),
                    )
                } else {
                    dest.extend(
                        offsets_i64
                            .values()
                            .iter()
                            .skip(1)
                            .map(|&v| (v + modifier) as u64),
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
        null_offset_adjustment: u64,
        buffer_index: &mut u32,
        inner_encoder: Arc<dyn ArrayEncoder>,
    ) -> Result<EncodedArray> {
        let mut offsets = Vec::with_capacity(num_offsets as usize);
        for (offsets_arr, validity_arr) in offset_arrays.iter().zip(validity) {
            let last_prev_offset = offsets.last().copied().unwrap_or(0) % null_offset_adjustment;
            Self::extend_offsets_vec_u64(
                &mut offsets,
                &offsets_arr,
                validity_arr,
                last_prev_offset,
                null_offset_adjustment,
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
        // See encodings.proto for reasoning behind this value
        let null_offset_adjustment = total_span + 1;
        let encoded_offsets = Self::do_encode_u64(
            offset_arrays,
            validity_arrays,
            num_offsets,
            null_offset_adjustment,
            buffer_index,
            inner_encoder,
        )?;
        Ok(EncodedArray {
            buffers: encoded_offsets.buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::List(Box::new(
                    pb::List {
                        offsets: Some(Box::new(encoded_offsets.encoding)),
                        null_offset_adjustment,
                        num_items: total_span,
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
        offsets_tasks: Vec<EncodeTask>,
        item_tasks: Vec<EncodeTask>,
    ) -> Result<Vec<EncodeTask>> {
        let mut all_tasks = offsets_tasks;
        let item_tasks = item_tasks;
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
            .map(|task| vec![task])
            .unwrap_or_default();
        let mut item_tasks = self.items_encoder.maybe_encode(items)?;
        if !offsets_tasks.is_empty() && item_tasks.is_empty() {
            // An items page cannot currently be shared by two different offsets pages.  This is
            // a limitation in the current scheduler and could be addressed in the future.  As a result
            // we always need to encode the items page if we encode the offsets page.
            //
            // In practice this isn't usually too bad unless we are targetting very small pages.
            item_tasks = self.items_encoder.flush()?;
        }
        Self::combine_tasks(offsets_tasks, item_tasks)
    }

    fn flush(&mut self) -> Result<Vec<EncodeTask>> {
        let offsets_tasks = self
            .offsets_encoder
            .flush()
            .map(|task| vec![task])
            .unwrap_or_default();
        let item_tasks = self.items_encoder.flush()?;
        Self::combine_tasks(offsets_tasks, item_tasks)
    }

    fn num_columns(&self) -> u32 {
        self.items_encoder.num_columns() + 1
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
    use arrow_schema::{DataType, Field, Fields};

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
    async fn test_nested_list() {
        let field = Field::new("", make_list_type(DataType::Utf8), true);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_list_struct_list() {
        let struct_type = DataType::Struct(Fields::from(vec![Field::new(
            "inner_str",
            DataType::Utf8,
            false,
        )]));

        let field = Field::new("", make_list_type(struct_type), true);
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
        // Scenario 1: Some lists are empty

        let values = [vec![Some(1), Some(2), Some(3)], vec![], vec![None]];
        // Test empty list at beginning, middle, and end
        for order in [[0, 1, 2], [1, 0, 2], [2, 0, 1]] {
            let items_builder = Int32Builder::new();
            let mut list_builder = ListBuilder::new(items_builder);
            for idx in order {
                list_builder.append_value(values[idx].clone());
            }
            let list_array = Arc::new(list_builder.finish());
            let test_cases = TestCases::default()
                .with_indices(vec![1])
                .with_indices(vec![0])
                .with_indices(vec![2]);
            check_round_trip_encoding_of_data(vec![list_array.clone()], &test_cases).await;
            let test_cases = test_cases.with_batch_size(1);
            check_round_trip_encoding_of_data(vec![list_array], &test_cases).await;
        }

        // Scenario 2: All lists are empty

        // When encoding a list of empty lists there are no items to encode
        // which is strange and we want to ensure we handle it
        let items_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(items_builder);
        list_builder.append(true);
        list_builder.append_null();
        list_builder.append(true);
        let list_array = Arc::new(list_builder.finish());

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![list_array.clone()], &test_cases).await;
        let test_cases = test_cases.with_batch_size(1);
        check_round_trip_encoding_of_data(vec![list_array], &test_cases).await;
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
