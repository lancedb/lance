// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::VecDeque, fmt::Debug, iter, ops::Range, sync::Arc, vec};

use arrow::array::AsArray;
use arrow_array::{make_array, types::UInt64Type, Array, ArrayRef, PrimitiveArray};
use arrow_buffer::{bit_util, BooleanBuffer, NullBuffer};
use arrow_schema::{DataType, Field as ArrowField};
use futures::{future::BoxFuture, stream::FuturesUnordered, FutureExt, TryStreamExt};
use lance_arrow::deepcopy::deep_copy_array;
use lance_core::utils::bit::pad_bytes;
use log::{debug, trace};
use snafu::{location, Location};

use crate::data::{AllNullDataBlock, DataBlock};
use crate::statistics::{GetStat, Stat};
use lance_core::{datatypes::Field, utils::tokio::spawn_cpu, Result};

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlockBuilder, FixedWidthDataBlock, UsedEncoding},
    decoder::{
        BlockDecompressor, ColumnInfo, DecodeArrayTask, DecodePageTask, DecodedArray, DecodedPage,
        DecompressorStrategy, FieldScheduler, FilterExpression, LoadedPage, LogicalPageDecoder,
        MessageType, MiniBlockDecompressor, NextDecodeTask, PageEncoding, PageInfo, PageScheduler,
        PrimitivePageDecoder, PriorityRange, ScheduledScanLine, SchedulerContext, SchedulingJob,
        StructuralDecodeArrayTask, StructuralFieldDecoder, StructuralFieldScheduler,
        StructuralPageDecoder, StructuralSchedulingJob, UnloadedPage,
    },
    encoder::{
        ArrayEncodingStrategy, CompressionStrategy, EncodeTask, EncodedColumn, EncodedPage,
        EncodingOptions, FieldEncoder, MiniBlockChunk, MiniBlockCompressed, OutOfLineBuffers,
    },
    encodings::physical::{decoder_from_array_encoding, ColumnBuffers, PageBuffers},
    format::{pb, ProtobufUtils},
    repdef::{LevelBuffer, RepDefBuilder, RepDefUnraveler},
    EncodingsIo,
};

#[derive(Debug)]
struct PrimitivePage {
    scheduler: Box<dyn PageScheduler>,
    num_rows: u64,
    page_index: u32,
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
            .enumerate()
            // Buggy versions of Lance could sometimes create empty pages
            .filter(|(page_index, page)| {
                log::trace!("Skipping empty page with index {}", page_index);
                page.num_rows > 0
            })
            .map(|(page_index, page)| {
                let page_buffers = PageBuffers {
                    column_buffers: buffers,
                    positions_and_sizes: &page.buffer_offsets_and_sizes,
                };
                let scheduler = decoder_from_array_encoding(
                    page.encoding.as_legacy(),
                    &page_buffers,
                    &data_type,
                );
                PrimitivePage {
                    scheduler,
                    num_rows: page.num_rows,
                    page_index: page_index as u32,
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
            "Scheduling {} rows across {} ranges from page with {} rows (priority={}, column_index={}, page_index={})",
            num_rows_in_next,
            ranges_in_page.len(),
            cur_page.num_rows,
            priority.current_priority(),
            self.scheduler.column_index,
            cur_page.page_index,
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
            page_index: cur_page.page_index,
        };

        let decoder = Box::new(logical_decoder);
        let decoder_ready = context.locate_decoder(decoder);
        Ok(ScheduledScanLine {
            decoders: vec![MessageType::DecoderReady(decoder_ready)],
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

    fn initialize<'a>(
        &'a self,
        _filter: &'a FilterExpression,
        _context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        // 2.0 schedulers do not need to initialize
        std::future::ready(Ok(())).boxed()
    }
}

/// A trait for figuring out how to schedule the data within
/// a single page.
trait StructuralPageScheduler: std::fmt::Debug + Send {
    /// Fetches any metadata required for the page
    fn initialize<'a>(&'a mut self, io: &Arc<dyn EncodingsIo>) -> BoxFuture<'a, Result<()>>;
    /// Schedules the read of the given ranges in the page
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &dyn EncodingsIo,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>>;
}

/// Metadata describing the decoded size of a mini-block
#[derive(Debug)]
struct ChunkMeta {
    num_values: u64,
    chunk_size_bytes: u64,
}

/// A task to decode a one or more mini-blocks of data into an output batch
///
/// Note: Two batches might share the same mini-block of data.  When this happens
/// then each batch gets a copy of the block and each batch decodes the block independently.
///
/// This means we have duplicated work but it is necessary to avoid having to synchronize
/// the decoding of the block. (TODO: test this theory)
#[derive(Debug)]
struct DecodeMiniBlockTask {
    // The decompressors for the rep, def, and value buffers
    rep_decompressor: Arc<dyn BlockDecompressor>,
    def_decompressor: Arc<dyn BlockDecompressor>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    // The mini-blocks to decode
    //
    // For each mini-block we also have the ranges of rows that we want to decode
    // from that mini-block.  For example, if the user asks for rows 10, 10000, and 20000
    // then we will have three chunks here and each chunk will have a small range of 1 row.
    chunks: Vec<ScheduledChunk>,
    // The offset into the first chunk that we want to start decoding from
    offset_into_first_chunk: u64,
    // The total number of rows that we are decoding
    num_rows: u64,
}

impl DecodeMiniBlockTask {
    fn decode_levels(
        rep_decompressor: &dyn BlockDecompressor,
        levels: LanceBuffer,
    ) -> Result<Option<impl AsRef<[u16]>>> {
        let rep = rep_decompressor.decompress(levels)?;
        match rep {
            DataBlock::FixedWidth(mut rep) => Ok(Some(rep.data.borrow_to_typed_slice::<u16>())),
            DataBlock::Constant(constant) => {
                assert_eq!(constant.data.len(), 2);
                if constant.data[0] == 0 && constant.data[1] == 0 {
                    Ok(None)
                } else {
                    // Maybe in the future we will encode all-null def or
                    // constant rep (all 1-item lists?) in a constant encoding
                    // but that doesn't happen today so we don't need to worry.
                    todo!()
                }
            }
            _ => unreachable!(),
        }
    }

    // We are building a LevelBuffer (levels) and want to copy into it `total_len`
    // values from `level_buf` starting at `offset`.
    //
    // We need to handle both the case where `levels` is None (no nulls encountered
    // yet) and the case where `level_buf` is None (the input we are copying from has
    // no nulls)
    fn extend_levels(
        offset: usize,
        range: Range<u64>,
        levels: &mut Option<LevelBuffer>,
        level_buf: &Option<impl AsRef<[u16]>>,
        dest_offset: usize,
    ) {
        if let Some(level_buf) = level_buf {
            if levels.is_none() {
                // This is the first non-empty def buf we've hit, fill in the past
                // with 0 (valid)
                let mut new_levels_vec =
                    LevelBuffer::with_capacity(offset + (range.end - range.start) as usize);
                new_levels_vec.extend(iter::repeat(0).take(dest_offset));
                *levels = Some(new_levels_vec);
            }
            levels.as_mut().unwrap().extend(
                level_buf.as_ref()[range.start as usize..range.end as usize]
                    .iter()
                    .copied(),
            );
        } else if let Some(levels) = levels {
            let num_values = (range.end - range.start) as usize;
            // This is an all-valid level_buf but we had nulls earlier and so we
            // need to materialize it
            levels.extend(iter::repeat(0).take(num_values));
        }
    }
}

impl DecodePageTask for DecodeMiniBlockTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        // First, we create output buffers for the rep and def and data
        let mut repbuf: Option<LevelBuffer> = None;
        let mut defbuf: Option<LevelBuffer> = None;
        let rep_decompressor = self.rep_decompressor;
        let def_decompressor = self.def_decompressor;

        let mut remaining = self.num_rows;
        let estimated_size_bytes = self
            .chunks
            .iter()
            .map(|chunk| chunk.data.len())
            .sum::<usize>()
            * 2;
        let mut data_builder =
            DataBlockBuilder::with_capacity_estimate(estimated_size_bytes as u64);
        let mut to_skip = self.offset_into_first_chunk;
        // We need to keep track of the offset into repbuf/defbuf that we are building up
        let mut level_offset = 0;
        // Now we iterate through each chunk and decode the data into the output buffers
        for chunk in self.chunks.into_iter() {
            // We always decode the entire chunk
            let buf = chunk.data.into_buffer();
            // The first 6 bytes describe the size of the remaining buffers
            let bytes_rep = u16::from_le_bytes([buf[0], buf[1]]) as usize;
            let bytes_def = u16::from_le_bytes([buf[2], buf[3]]) as usize;
            let bytes_val = u16::from_le_bytes([buf[4], buf[5]]) as usize;

            debug_assert!(buf.len() >= bytes_rep + bytes_def + bytes_val + 6);
            debug_assert!(
                buf.len()
                    <= bytes_rep
                        + bytes_def
                        + bytes_val
                        + 6
                        + 1 // P1
                        + (2 * MINIBLOCK_MAX_PADDING) // P2/P3
            );
            let p1 = bytes_rep % 2;
            let rep = buf.slice_with_length(6, bytes_rep);
            let def = buf.slice_with_length(6 + bytes_rep + p1, bytes_def);
            let p2 = pad_bytes::<MINIBLOCK_ALIGNMENT>(6 + bytes_rep + p1 + bytes_def);
            let values = buf.slice_with_length(6 + bytes_rep + bytes_def + p2, bytes_val);

            let mut values = self
                .value_decompressor
                .decompress(LanceBuffer::Borrowed(values), chunk.vals_in_chunk)?;

            let rep = Self::decode_levels(rep_decompressor.as_ref(), LanceBuffer::Borrowed(rep))?;
            let def = Self::decode_levels(def_decompressor.as_ref(), LanceBuffer::Borrowed(def))?;

            // We've decoded the entire block.  Now we need to factor in:
            // - The offset into the first chunk
            // - The ranges the user asked for
            // - The total # of rows in this task
            //
            // From these we can figure out which values to keep.
            //
            // For example, maybe we've are asked to decode 100 rows, with an offset of 50, from
            // a block with 1024 values, and the user asked for the ranges 400..500 and 600..700
            //
            // In this case we want to take the values 450..500 and 600..650 from the block.
            let mut offset = to_skip;
            for range in chunk.ranges {
                if to_skip > range.end - range.start {
                    to_skip -= range.end - range.start;
                    continue;
                }
                // Subtract skip from start of range
                let range = range.start + to_skip..range.end;
                to_skip = 0;

                // Truncate range to fit remaining
                let range_len = range.end - range.start;
                let to_take = range_len.min(remaining);
                let range = range.start..range.start + to_take;

                // Grab values and add to what we are building
                Self::extend_levels(
                    offset as usize,
                    range.clone(),
                    &mut repbuf,
                    &rep,
                    level_offset,
                );
                Self::extend_levels(
                    offset as usize,
                    range.clone(),
                    &mut defbuf,
                    &def,
                    level_offset,
                );
                data_builder.append(&mut values, range);
                remaining -= to_take;
                offset += to_take;
                level_offset += to_take as usize;
            }
        }
        debug_assert_eq!(remaining, 0);

        let data = data_builder.finish();

        Ok(DecodedPage {
            data,
            repetition: repbuf,
            definition: defbuf,
        })
    }
}

/// Decodes mini-block formatted data.  See [`PrimitiveStructuralEncoder`] for more
/// details on the different layouts.
#[derive(Debug)]
struct MiniBlockDecoder {
    rep_decompressor: Arc<dyn BlockDecompressor>,
    def_decompressor: Arc<dyn BlockDecompressor>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    data: VecDeque<ScheduledChunk>,
    offset_in_current_chunk: u64,
    num_rows: u64,
}

impl StructuralPageDecoder for MiniBlockDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let mut remaining = num_rows;
        let mut chunks = Vec::new();
        let offset_into_first_chunk = self.offset_in_current_chunk;
        while remaining > 0 {
            if remaining >= self.data.front().unwrap().vals_targeted - self.offset_in_current_chunk
            {
                // We are fully consuming the next chunk
                let chunk = self.data.pop_front().unwrap();
                remaining -= chunk.vals_targeted - self.offset_in_current_chunk;
                chunks.push(chunk);
                self.offset_in_current_chunk = 0;
            } else {
                // We are partially consuming the next chunk
                let chunk = self.data.front().unwrap().clone();
                self.offset_in_current_chunk += remaining;
                debug_assert!(self.offset_in_current_chunk > 0);
                remaining = 0;
                chunks.push(chunk);
            }
        }
        Ok(Box::new(DecodeMiniBlockTask {
            chunks,
            rep_decompressor: self.rep_decompressor.clone(),
            def_decompressor: self.def_decompressor.clone(),
            value_decompressor: self.value_decompressor.clone(),
            num_rows,
            offset_into_first_chunk,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// A scheduler for simple all-null data
///
/// "simple" all-null data is data that is all null and only has a single level of definition and
/// no repetition.  We don't need to read any data at all in this case.
#[derive(Debug, Default)]
pub struct SimpleAllNullScheduler {}

impl StructuralPageScheduler for SimpleAllNullScheduler {
    fn initialize<'a>(&'a mut self, _io: &Arc<dyn EncodingsIo>) -> BoxFuture<'a, Result<()>> {
        std::future::ready(Ok(())).boxed()
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        _io: &dyn EncodingsIo,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        Ok(std::future::ready(Ok(
            Box::new(SimpleAllNullPageDecoder { num_rows }) as Box<dyn StructuralPageDecoder>
        ))
        .boxed())
    }
}

/// A page decode task for all-null data without any
/// repetition and only a single level of definition
#[derive(Debug)]
struct SimpleAllNullDecodePageTask {
    num_values: u64,
}
impl DecodePageTask for SimpleAllNullDecodePageTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        Ok(DecodedPage {
            data: DataBlock::AllNull(AllNullDataBlock {
                num_values: self.num_values,
            }),
            repetition: None,
            definition: Some(vec![1; self.num_values as usize]),
        })
    }
}

#[derive(Debug)]
pub struct SimpleAllNullPageDecoder {
    num_rows: u64,
}

impl StructuralPageDecoder for SimpleAllNullPageDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        Ok(Box::new(SimpleAllNullDecodePageTask {
            num_values: num_rows,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// A scheduler for a page that has been encoded with the mini-block layout
#[derive(Debug)]
pub struct MiniBlockScheduler {
    // These come from the protobuf
    meta_buf_position: u64,
    meta_buf_size: u64,
    data_buf_position: u64,
    priority: u64,
    rows_in_page: u64,
    rep_decompressor: Arc<dyn BlockDecompressor>,
    def_decompressor: Arc<dyn BlockDecompressor>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    // This is set after initialization
    chunk_meta: Vec<ChunkMeta>,
}

impl MiniBlockScheduler {
    fn try_new(
        buffer_offsets_and_sizes: &[(u64, u64)],
        priority: u64,
        rows_in_page: u64,
        layout: &pb::MiniBlockLayout,
        decompressors: &dyn DecompressorStrategy,
    ) -> Result<Self> {
        let (meta_buf_position, meta_buf_size) = buffer_offsets_and_sizes[0];
        // We don't use the data buf size since we can get it from the metadata
        let (data_buf_position, _) = buffer_offsets_and_sizes[1];
        let rep_decompressor =
            decompressors.create_block_decompressor(layout.rep_compression.as_ref().unwrap())?;
        let def_decompressor =
            decompressors.create_block_decompressor(layout.def_compression.as_ref().unwrap())?;
        let value_decompressor = decompressors
            .create_miniblock_decompressor(layout.value_compression.as_ref().unwrap())?;
        Ok(Self {
            meta_buf_position,
            meta_buf_size,
            data_buf_position,
            rep_decompressor: rep_decompressor.into(),
            def_decompressor: def_decompressor.into(),
            value_decompressor: value_decompressor.into(),
            priority,
            rows_in_page,
            chunk_meta: Vec::new(),
        })
    }

    /// Calculates the overlap between a user-supplied range and a chunk of mini-block data
    fn calc_overlap(
        range: &mut Range<u64>,
        chunk: &ChunkMeta,
        rows_offset: u64,
        dst: &mut ScheduledChunk,
    ) -> ChunkOverlap {
        if range.start > chunk.num_values + rows_offset {
            ChunkOverlap::RangeAfterChunk
        } else {
            let start_in_chunk = range.start - rows_offset;
            let end_in_chunk = (start_in_chunk + range.end - range.start).min(chunk.num_values);
            let rows_in_chunk = end_in_chunk - start_in_chunk;
            range.start += rows_in_chunk;
            dst.ranges.push(start_in_chunk..end_in_chunk);
            ChunkOverlap::Overlap
        }
    }
}

#[derive(Debug)]
struct ScheduledChunk {
    data: LanceBuffer,
    // The total number of values in the chunk, not all values may be targeted
    vals_in_chunk: u64,
    // The number of values that are targeted by the ranges.  This should be the
    // same as the sum of `Self::ranges`
    vals_targeted: u64,
    ranges: Vec<Range<u64>>,
}

impl Clone for ScheduledChunk {
    fn clone(&self) -> Self {
        Self {
            data: self.data.try_clone().unwrap(),
            vals_in_chunk: self.vals_in_chunk,
            ranges: self.ranges.clone(),
            vals_targeted: self.vals_targeted,
        }
    }
}

pub enum ChunkOverlap {
    RangeAfterChunk,
    Overlap,
}

impl StructuralPageScheduler for MiniBlockScheduler {
    fn initialize<'a>(&'a mut self, io: &Arc<dyn EncodingsIo>) -> BoxFuture<'a, Result<()>> {
        let metadata = io.submit_single(
            self.meta_buf_position..self.meta_buf_position + self.meta_buf_size,
            0,
        );
        async move {
            let bytes = metadata.await?;
            assert!(bytes.len() % 2 == 0);
            let mut bytes = LanceBuffer::from_bytes(bytes, 2);
            let words = bytes.borrow_to_typed_slice::<u16>();
            let words = words.as_ref();
            self.chunk_meta.reserve(words.len());
            let mut rows_counter = 0;
            for (word_idx, word) in words.iter().enumerate() {
                let log_num_values = word & 0x0F;
                let divided_bytes = word >> 4;
                let num_bytes = (divided_bytes as usize + 1) * MINIBLOCK_ALIGNMENT;
                debug_assert!(num_bytes > 0);
                let num_values = if word_idx < words.len() - 1 {
                    debug_assert!(log_num_values > 0);
                    1 << log_num_values
                } else {
                    debug_assert_eq!(log_num_values, 0);
                    self.rows_in_page - rows_counter
                };
                rows_counter += num_values;

                self.chunk_meta.push(ChunkMeta {
                    num_values,
                    chunk_size_bytes: num_bytes as u64,
                });
            }
            Ok(())
        }
        .boxed()
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &dyn EncodingsIo,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>> {
        let mut chunk_meta_iter = self.chunk_meta.iter();
        let mut current_chunk = chunk_meta_iter.next().unwrap();
        let mut row_offset = 0;
        let mut bytes_offset = 0;

        let mut scheduled_chunks = VecDeque::with_capacity(self.chunk_meta.len());
        let mut ranges_to_req = Vec::with_capacity(self.chunk_meta.len());
        let mut num_rows = 0;

        let mut current_scheduled_chunk = ScheduledChunk {
            data: LanceBuffer::empty(),
            ranges: Vec::new(),
            vals_in_chunk: current_chunk.num_values,
            vals_targeted: 0,
        };

        // There can be both multiple ranges per chunk and multiple chunks per range
        for range in ranges {
            num_rows += range.end - range.start;
            let mut range = range.clone();
            while !range.is_empty() {
                Self::calc_overlap(
                    &mut range,
                    current_chunk,
                    row_offset,
                    &mut current_scheduled_chunk,
                );
                // Might be empty if entire chunk is skipped
                if !range.is_empty() {
                    if !current_scheduled_chunk.ranges.is_empty() {
                        scheduled_chunks.push_back(current_scheduled_chunk);
                        ranges_to_req.push(
                            (self.data_buf_position + bytes_offset)
                                ..(self.data_buf_position
                                    + bytes_offset
                                    + current_chunk.chunk_size_bytes),
                        );
                    }
                    row_offset += current_chunk.num_values;
                    bytes_offset += current_chunk.chunk_size_bytes;
                    if let Some(next_chunk) = chunk_meta_iter.next() {
                        current_chunk = next_chunk;
                    }
                    current_scheduled_chunk = ScheduledChunk {
                        data: LanceBuffer::empty(),
                        ranges: Vec::new(),
                        vals_in_chunk: current_chunk.num_values,
                        vals_targeted: 0,
                    };
                }
            }
        }
        if !current_scheduled_chunk.ranges.is_empty() {
            scheduled_chunks.push_back(current_scheduled_chunk);
            ranges_to_req.push(
                (self.data_buf_position + bytes_offset)
                    ..(self.data_buf_position + bytes_offset + current_chunk.chunk_size_bytes),
            );
        }

        let data = io.submit_request(ranges_to_req, self.priority);

        let rep_decompressor = self.rep_decompressor.clone();
        let def_decompressor = self.def_decompressor.clone();
        let value_decompressor = self.value_decompressor.clone();

        for scheduled_chunk in scheduled_chunks.iter_mut() {
            scheduled_chunk.vals_targeted =
                scheduled_chunk.ranges.iter().map(|r| r.end - r.start).sum();
        }

        Ok(async move {
            let data = data.await?;
            for (chunk, data) in scheduled_chunks.iter_mut().zip(data) {
                chunk.data = LanceBuffer::from_bytes(data, 1);
            }
            Ok(Box::new(MiniBlockDecoder {
                rep_decompressor,
                def_decompressor,
                value_decompressor,
                data: scheduled_chunks,
                offset_in_current_chunk: 0,
                num_rows,
            }) as Box<dyn StructuralPageDecoder>)
        }
        .boxed())
    }
}

#[derive(Debug)]
struct StructuralPrimitiveFieldSchedulingJob<'a> {
    scheduler: &'a StructuralPrimitiveFieldScheduler,
    ranges: Vec<Range<u64>>,
    page_idx: usize,
    range_idx: usize,
    range_offset: u64,
    global_row_offset: u64,
}

impl<'a> StructuralPrimitiveFieldSchedulingJob<'a> {
    pub fn new(scheduler: &'a StructuralPrimitiveFieldScheduler, ranges: Vec<Range<u64>>) -> Self {
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

impl<'a> StructuralSchedulingJob for StructuralPrimitiveFieldSchedulingJob<'a> {
    fn schedule_next(
        &mut self,
        context: &mut SchedulerContext,
    ) -> Result<Option<ScheduledScanLine>> {
        if self.range_idx >= self.ranges.len() {
            return Ok(None);
        }
        // Get our current range
        let mut range = self.ranges[self.range_idx].clone();
        range.start += self.range_offset;
        let priority = range.start;

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
            "Scheduling {} rows across {} ranges from page with {} rows (priority={}, column_index={}, page_index={})",
            num_rows_in_next,
            ranges_in_page.len(),
            cur_page.num_rows,
            priority,
            self.scheduler.column_index,
            cur_page.page_index,
        );

        self.global_row_offset += cur_page.num_rows;
        self.page_idx += 1;

        let page_decoder = cur_page
            .scheduler
            .schedule_ranges(&ranges_in_page, context.io().as_ref())?;

        let cur_path = context.current_path();
        let page_index = cur_page.page_index;
        let unloaded_page = async move {
            let page_decoder = page_decoder.await?;
            Ok(LoadedPage {
                decoder: page_decoder,
                path: cur_path,
                page_index,
            })
        }
        .boxed();

        Ok(Some(ScheduledScanLine {
            decoders: vec![MessageType::UnloadedPage(UnloadedPage(unloaded_page))],
            rows_scheduled: num_rows_in_next,
        }))
    }
}

#[derive(Debug)]
struct PageInfoAndScheduler {
    page_index: usize,
    num_rows: u64,
    scheduler: Box<dyn StructuralPageScheduler>,
}

/// A scheduler for a leaf node
///
/// Here we look at the layout of the various pages and delegate scheduling to a scheduler
/// appropriate for the layout of the page.
#[derive(Debug)]
pub struct StructuralPrimitiveFieldScheduler {
    page_schedulers: Vec<PageInfoAndScheduler>,
    column_index: u32,
}

impl StructuralPrimitiveFieldScheduler {
    pub fn try_new(
        column_info: &ColumnInfo,
        decompressors: &dyn DecompressorStrategy,
    ) -> Result<Self> {
        let page_schedulers = column_info
            .page_infos
            .iter()
            .enumerate()
            .map(|(page_index, page_info)| {
                Self::page_info_to_scheduler(page_info, page_index, decompressors)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            page_schedulers,
            column_index: column_info.index,
        })
    }

    fn page_info_to_scheduler(
        page_info: &PageInfo,
        page_index: usize,
        decompressors: &dyn DecompressorStrategy,
    ) -> Result<PageInfoAndScheduler> {
        let scheduler: Box<dyn StructuralPageScheduler> =
            match page_info.encoding.as_structural().layout.as_ref() {
                Some(pb::page_layout::Layout::MiniBlockLayout(mini_block)) => {
                    Box::new(MiniBlockScheduler::try_new(
                        &page_info.buffer_offsets_and_sizes,
                        page_info.priority,
                        page_info.num_rows,
                        mini_block,
                        decompressors,
                    )?)
                }
                Some(pb::page_layout::Layout::AllNullLayout(_)) => {
                    Box::new(SimpleAllNullScheduler::default()) as Box<dyn StructuralPageScheduler>
                }
                _ => todo!(),
            };
        Ok(PageInfoAndScheduler {
            page_index,
            num_rows: page_info.num_rows,
            scheduler,
        })
    }
}

impl StructuralFieldScheduler for StructuralPrimitiveFieldScheduler {
    fn initialize<'a>(
        &'a mut self,
        _filter: &'a FilterExpression,
        context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        let page_init = self
            .page_schedulers
            .iter_mut()
            .map(|s| s.scheduler.initialize(context.io()))
            .collect::<FuturesUnordered<_>>();
        async move {
            page_init.try_collect::<Vec<_>>().await?;
            Ok(())
        }
        .boxed()
    }

    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[Range<u64>],
        _filter: &FilterExpression,
    ) -> Result<Box<dyn StructuralSchedulingJob + 'a>> {
        let ranges = ranges.to_vec();
        Ok(Box::new(StructuralPrimitiveFieldSchedulingJob::new(
            self, ranges,
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
    page_index: u32,
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
            page_index: u32::MAX,
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

        let array = make_array(block.into_arrow(self.data_type.clone(), self.should_validate)?);

        // This is a bit of a hack to work around https://github.com/apache/arrow-rs/issues/6302
        //
        // We change from nulls-in-dictionary (storage format) to nulls-in-indices (arrow-rs preferred
        // format)
        //
        // The calculation of logical_nulls is not free and would be good to avoid in the future
        if let DataType::Dictionary(_, _) = self.data_type {
            let dict = array.as_any_dictionary();
            if let Some(nulls) = array.logical_nulls() {
                let new_indices = dict.keys().to_data();
                let new_array = make_array(
                    new_indices
                        .into_builder()
                        .nulls(Some(nulls))
                        .add_child_data(dict.values().to_data())
                        .data_type(dict.data_type().clone())
                        .build()?,
                );
                return Ok(new_array);
            }
        }
        Ok(array)
    }
}

impl LogicalPageDecoder for PrimitiveFieldDecoder {
    // TODO: In the future, at some point, we may consider partially waiting for primitive pages by
    // breaking up large I/O into smaller I/O as a way to accelerate the "time-to-first-decode"
    fn wait_for_loaded(&mut self, loaded_need: u64) -> BoxFuture<Result<()>> {
        log::trace!(
            "primitive wait for more than {} rows on column {} and page {} (page has {} rows)",
            loaded_need,
            self.column_index,
            self.page_index,
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

/// Takes the output from several pages decoders and
/// concatenates them.
#[derive(Debug)]
pub struct StructuralCompositeDecodeArrayTask {
    tasks: Vec<Box<dyn DecodePageTask>>,
    num_values: u64,
    data_type: DataType,
    should_validate: bool,
}

impl StructuralDecodeArrayTask for StructuralCompositeDecodeArrayTask {
    fn decode(self: Box<Self>) -> Result<DecodedArray> {
        let mut arrays = Vec::with_capacity(self.tasks.len());
        let mut all_rep = LevelBuffer::with_capacity(self.num_values as usize);
        let mut all_def = LevelBuffer::with_capacity(self.num_values as usize);
        let mut offset = 0;
        let mut has_def = false;
        for task in self.tasks {
            let decoded = task.decode()?;

            if let Some(rep) = &decoded.repetition {
                // Note: if one chunk has repetition, all chunks will have repetition
                // and so all_rep will either end up with len=num_values or len=0
                all_rep.extend(rep);
            }
            if let Some(def) = &decoded.definition {
                if !has_def {
                    // This is the first validity we have seen, need to backfill with all-valid
                    // if we've processed any all-valid pages
                    has_def = true;
                    all_def.extend(iter::repeat(0).take(offset));
                }
                all_def.extend(def);
            }

            let array = make_array(
                decoded
                    .data
                    .into_arrow(self.data_type.clone(), self.should_validate)?,
            );

            offset += array.len();
            arrays.push(array);
        }
        let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
        let array = arrow_select::concat::concat(&array_refs)?;
        let all_rep = if all_rep.is_empty() {
            None
        } else {
            Some(all_rep)
        };
        let all_def = if all_def.is_empty() {
            None
        } else {
            Some(all_def)
        };
        let mut repdef = RepDefUnraveler::new(all_rep, all_def);

        // The primitive array itself has a validity
        let mut validity = repdef.unravel_validity();
        if matches!(self.data_type, DataType::Null) {
            // Null arrays don't have a validity but we still pretend they do for consistency's sake
            // up until this point.  We need to remove it here.
            validity = None;
        }
        if let Some(validity) = validity.as_ref() {
            assert!(validity.len() == array.len());
        }
        // SAFETY: We are just replacing the validity and asserted it is the correct size
        let array = make_array(unsafe {
            array
                .to_data()
                .into_builder()
                .nulls(validity)
                .build_unchecked()
        });
        Ok(DecodedArray { array, repdef })
    }
}

#[derive(Debug)]
pub struct StructuralPrimitiveFieldDecoder {
    field: Arc<ArrowField>,
    page_decoders: VecDeque<Box<dyn StructuralPageDecoder>>,
    should_validate: bool,
    rows_drained_in_current: u64,
}

impl StructuralPrimitiveFieldDecoder {
    pub fn new(field: &Arc<ArrowField>, should_validate: bool) -> Self {
        Self {
            field: field.clone(),
            page_decoders: VecDeque::new(),
            should_validate,
            rows_drained_in_current: 0,
        }
    }
}

impl StructuralFieldDecoder for StructuralPrimitiveFieldDecoder {
    fn accept_page(&mut self, child: LoadedPage) -> Result<()> {
        assert!(child.path.is_empty());
        self.page_decoders.push_back(child.decoder);
        Ok(())
    }

    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn StructuralDecodeArrayTask>> {
        let mut remaining = num_rows;
        let mut tasks = Vec::new();
        while remaining > 0 {
            let cur_page = self.page_decoders.front_mut().unwrap();
            let num_in_page = cur_page.num_rows() - self.rows_drained_in_current;
            let to_take = num_in_page.min(remaining);

            let task = cur_page.drain(to_take)?;
            tasks.push(task);

            if to_take == num_in_page {
                self.page_decoders.pop_front();
                self.rows_drained_in_current = 0;
            } else {
                self.rows_drained_in_current += to_take;
            }

            remaining -= to_take;
        }
        Ok(Box::new(StructuralCompositeDecodeArrayTask {
            tasks,
            data_type: self.field.data_type().clone(),
            should_validate: self.should_validate,
            num_values: num_rows,
        }))
    }

    fn data_type(&self) -> &DataType {
        self.field.data_type()
    }
}

#[derive(Debug)]
pub struct AccumulationQueue {
    cache_bytes: u64,
    keep_original_array: bool,
    buffered_arrays: Vec<ArrayRef>,
    current_bytes: u64,
    // Row number of the first item in buffered_arrays, reset on flush
    row_number: u64,
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
            row_number: u64::MAX,
        }
    }

    /// Adds an array to the queue, if there is enough data then the queue is flushed
    /// and returned
    pub fn insert(&mut self, array: ArrayRef, row_number: u64) -> Option<(Vec<ArrayRef>, u64)> {
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.current_bytes += array.get_array_memory_size() as u64;
        if self.current_bytes > self.cache_bytes {
            debug!(
                "Flushing column {} page of size {} bytes (unencoded)",
                self.column_index, self.current_bytes
            );
            // Push into buffered_arrays without copy since we are about to flush anyways
            self.buffered_arrays.push(array);
            self.current_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            Some((std::mem::take(&mut self.buffered_arrays), row_number))
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

    pub fn flush(&mut self) -> Option<(Vec<ArrayRef>, u64)> {
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
            let row_number = self.row_number;
            self.row_number = 0;
            Some((std::mem::take(&mut self.buffered_arrays), row_number))
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
        let data_type = self.field.data_type();

        Ok(tokio::task::spawn(async move {
            let num_values = arrays.iter().map(|arr| arr.len() as u64).sum();
            let data = DataBlock::from_arrays(&arrays, num_values);
            let mut buffer_index = 0;
            let array = encoder.encode(data, &data_type, &mut buffer_index)?;
            let (data, description) = array.into_buffers();
            Ok(EncodedPage {
                data,
                description: PageEncoding::Legacy(description),
                num_rows: num_values,
                column_idx,
                row_number: 0, // legacy encoders do not use
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
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        _external_buffers: &mut OutOfLineBuffers,
        _repdef: RepDefBuilder,
        _row_number: u64,
    ) -> Result<Vec<EncodeTask>> {
        if let Some(arrays) = self.accumulation_queue.insert(array, /*row_number=*/ 0) {
            Ok(self.do_flush(arrays.0)?)
        } else {
            Ok(vec![])
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self, _external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        if let Some(arrays) = self.accumulation_queue.flush() {
            Ok(self.do_flush(arrays.0)?)
        } else {
            Ok(vec![])
        }
    }

    fn num_columns(&self) -> u32 {
        1
    }

    fn finish(
        &mut self,
        _external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        std::future::ready(Ok(vec![EncodedColumn::default()])).boxed()
    }
}

// We align and pad mini-blocks to 8 byte boundaries for two reasons.  First,
// to allow us to store a chunk size in 12 bits.
//
// If we directly record the size in bytes with 12 bits we would be limited to
// 4KiB which is too small.  Since we know each mini-block consists of 8 byte
// words we can store the # of words instead which gives us 32KiB.  We want
// at least 24KiB so we can handle even the worst case of
// - 4Ki values compressed into an 8186 byte buffer
// - 4 bytes to describe rep & def lengths
// - 16KiB of rep & def buffer (this will almost never happen but life is easier if we
//   plan for it)
//
// Second, each chunk in a mini-block is aligned to 8 bytes.  This allows multi-byte
// values like offsets to be stored in a mini-block and safely read back out.  It also
// helps ensure zero-copy reads in cases where zero-copy is possible (e.g. no decoding
// needed).
//
// Note: by "aligned to 8 bytes" we mean BOTH "aligned to 8 bytes from the start of
// the page" and "aligned to 8 bytes from the start of the file."
const MINIBLOCK_ALIGNMENT: usize = 8;
const MINIBLOCK_MAX_PADDING: usize = MINIBLOCK_ALIGNMENT - 1;

/// An encoder for primitive (leaf) arrays
///
/// This encoder is fairly complicated and follows a number of paths depending
/// on the data.
///
/// First, we convert the validity & offsets information into repetition and
/// definition levels.  Then we compress the data itself into a single buffer.
///
/// If the data is narrow then we encode the data in small chunks (each chunk
/// should be a few disk sectors and contains a buffer of repetition, a buffer
/// of definition, and a buffer of value data).  This approach is called
/// "mini-block".  These mini-blocks are stored into a single data buffer.
///
/// If the data is wide then we zip together the repetition and definition value
/// with the value data into a single buffer.  This approach is called "zipped".
///
/// If there is any repetition information then we create a repetition index (TODO)
///
/// In addition, the compression process may create zero or more metadata buffers.
/// For example, a dictionary compression will create dictionary metadata.  Any
/// mini-block approach has a metadata buffer of block sizes.  This metadata is
/// stored in a separate buffer on disk and read at initialization time.
///
/// TODO: We should concatenate metadata buffers from all pages into a single buffer
/// at (roughly) the end of the file so there is, at most, one read per column of
/// metadata per file.
pub struct PrimitiveStructuralEncoder {
    // Accumulates arrays until we have enough data to justify a disk page
    accumulation_queue: AccumulationQueue,
    accumulated_repdefs: Vec<RepDefBuilder>,
    // The compression strategy we will use to compress the data
    compression_strategy: Arc<dyn CompressionStrategy>,
    column_index: u32,
    field: Field,
}

impl PrimitiveStructuralEncoder {
    pub fn try_new(
        options: &EncodingOptions,
        compression_strategy: Arc<dyn CompressionStrategy>,
        column_index: u32,
        field: Field,
    ) -> Result<Self> {
        Ok(Self {
            accumulation_queue: AccumulationQueue::new(
                options.cache_bytes_per_column,
                column_index,
                options.keep_original_array,
            ),
            accumulated_repdefs: Vec::new(),
            column_index,
            compression_strategy,
            field,
        })
    }

    // TODO: This is a heuristic we may need to tune at some point
    //
    // As data gets narrow then the "zipping" process gets too expensive
    //   and we prefer mini-block
    // As data gets wide then the # of values per block shrinks (very wide)
    //   data doesn't even fit in a mini-block and the block overhead gets
    //   too large and we prefer zipped.
    fn is_narrow(data_block: &DataBlock) -> bool {

        const MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE: u64 = 256;

        if let Some(max_len_array) = data_block.get_stat(Stat::MaxLength) {
            let max_len_array = max_len_array
                .as_any()
                .downcast_ref::<PrimitiveArray<UInt64Type>>()
                .unwrap();
            if max_len_array.value(0) < MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE {
                return true;
            }
        }
        false
    }

    // Converts value data, repetition levels, and definition levels into a single
    // buffer of mini-blocks.  In addition, creates a buffer of mini-block metadata
    // which tells us the size of each block.
    //
    // Each chunk is serialized as:
    // | rep_len (2 bytes) | def_len (2 bytes) | values_len (2 bytes) | rep | P1 | def | P2 | values | P3 |
    //
    // P1 - Up to 1 padding byte to ensure `def` is 2-byte aligned
    // P2 - Up to 7 padding bytes to ensure `values` is 8-byte aligned
    // P3 - Up to 7 padding bytes to ensure the chunk is a multiple of 8 bytes (this also ensures
    //      that the next `chunk` is 8-byte aligned)
    //
    // rep is guaranteed to be 2-byte aligned
    // def is guaranteed to be 2-byte aligned
    // values is guaranteed to be 8-byte aligned
    // rep_len, def_len, and values_len are guaranteed to be 2-byte aligned but this shouldn't matter.
    //
    // Each block has a u16 word of metadata.  The upper 12 bits contain 1/6 the
    // # of bytes in the block (if the block does not have an even number of bytes
    // then up to 7 bytes of padding are added).  The lower 4 bits describe the log_2
    // number of values (e.g. if there are 1024 then the lower 4 bits will be
    // 0xA)  All blocks except the last must have power-of-two number of values.
    // This not only makes metadata smaller but it makes decoding easier since
    // batch sizes are typically a power of 2.  4 bits would allow us to express
    // up to 16Ki values but we restrict this further to 4Ki values.
    //
    // This means blocks can have 1 to 4Ki values and 8 - 32Ki bytes.
    //
    // All metadata words are serialized (as little endian) into a single buffer
    // of metadata values.
    fn serialize_miniblocks(
        miniblocks: MiniBlockCompressed,
        rep: Vec<LanceBuffer>,
        def: Vec<LanceBuffer>,
    ) -> (LanceBuffer, LanceBuffer) {
        let bytes_rep = rep.iter().map(|r| r.len()).sum::<usize>();
        let bytes_def = def.iter().map(|d| d.len()).sum::<usize>();
        let max_bytes_repdef_len = rep.len() * 4;
        let max_padding = miniblocks.chunks.len() * (1 + (2 * MINIBLOCK_MAX_PADDING));
        let mut data_buffer = Vec::with_capacity(
            miniblocks.data.len()      // `values`
                + bytes_rep            // `rep_len * num_blocks`
                + bytes_def            // `def_len * num_blocks`
                + max_bytes_repdef_len // `rep` and `def`
                + max_padding, // `P1`, `P2`, and `P3` for each block
        );
        let mut meta_buffer = Vec::with_capacity(miniblocks.data.len() * 2);

        let mut value_offset = 0;
        for ((chunk, rep), def) in miniblocks.chunks.into_iter().zip(rep).zip(def) {
            let start_len = data_buffer.len();
            // Start of chunk should be aligned
            debug_assert_eq!(start_len % MINIBLOCK_ALIGNMENT, 0);

            assert!(rep.len() < u16::MAX as usize);
            assert!(def.len() < u16::MAX as usize);
            let bytes_rep = rep.len() as u16;
            let bytes_def = def.len() as u16;
            let bytes_val = chunk.num_bytes;

            // Each chunk starts with the size of the rep buffer (2 bytes) the size of
            // the def buffer (2 bytes) and the size of the values buffer (2 bytes)
            data_buffer.extend_from_slice(&bytes_rep.to_le_bytes());
            data_buffer.extend_from_slice(&bytes_def.to_le_bytes());
            data_buffer.extend_from_slice(&bytes_val.to_le_bytes());

            data_buffer.extend_from_slice(&rep);
            // In theory we should insert P1 here.  However, since we do not have bit-packing of rep
            // def levels yet we can skip this step.
            debug_assert_eq!(data_buffer.len() % 2, 0);
            data_buffer.extend_from_slice(&def);

            let p2 = pad_bytes::<MINIBLOCK_ALIGNMENT>(data_buffer.len());
            // SAFETY: We ensured the data buffer would be large enough when we allocated
            data_buffer.extend(iter::repeat(0).take(p2));

            let num_value_bytes = chunk.num_bytes as usize;
            let values =
                &miniblocks.data[value_offset as usize..value_offset as usize + num_value_bytes];
            debug_assert_eq!(data_buffer.len() % MINIBLOCK_ALIGNMENT, 0);
            data_buffer.extend_from_slice(values);

            let p3 = pad_bytes::<MINIBLOCK_ALIGNMENT>(data_buffer.len());
            data_buffer.extend(iter::repeat(0).take(p3));
            value_offset += num_value_bytes as u64;

            let chunk_bytes = data_buffer.len() - start_len;
            assert!(chunk_bytes <= 16 * 1024);
            assert!(chunk_bytes > 0);
            assert_eq!(chunk_bytes % 8, 0);
            // We subtract 1 here from chunk_bytes because we want to be able to express
            // a size of 32KiB and not (32Ki - 8)B which is what we'd get otherwise with
            // 0xFFF
            let divided_bytes = chunk_bytes / MINIBLOCK_ALIGNMENT;
            let divided_bytes_minus_one = (divided_bytes - 1) as u64;

            let metadata = ((divided_bytes_minus_one << 4) | chunk.log_num_values as u64) as u16;
            meta_buffer.extend_from_slice(&metadata.to_le_bytes());
        }

        (
            LanceBuffer::Owned(data_buffer),
            LanceBuffer::Owned(meta_buffer),
        )
    }

    /// Compresses a buffer of levels
    ///
    /// TODO: Use bit-packing here
    fn compress_levels(
        levels: Option<LevelBuffer>,
        num_values: u64,
        compression_strategy: &dyn CompressionStrategy,
        chunks: &[MiniBlockChunk],
    ) -> Result<(Vec<LanceBuffer>, pb::ArrayEncoding)> {
        if let Some(levels) = levels {
            debug_assert_eq!(num_values as usize, levels.len());
            // Make the levels into a FixedWidth data block
            let mut levels_buf = LanceBuffer::reinterpret_vec(levels);
            let levels_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: levels_buf.borrow_and_clone(),
                bits_per_value: 16,
                num_values,
                block_info: BlockInfo::new(),
                used_encoding: UsedEncoding::new(),
            });
            let levels_field = Field::new_arrow("", DataType::UInt16, false)?;
            // Pick a block compressor
            let (compressor, compressor_desc) =
                compression_strategy.create_block_compressor(&levels_field, &levels_block)?;
            // Compress blocks of levels (sized according to the chunks)
            let mut buffers = Vec::with_capacity(chunks.len());
            let mut off = 0;
            let mut values_counter = 0;
            for chunk in chunks {
                let chunk_num_values = chunk.num_values(values_counter, num_values);
                values_counter += chunk_num_values;
                let level_bytes = chunk_num_values as usize * 2;
                let chunk_levels = levels_buf.slice_with_length(off, level_bytes);
                let chunk_levels_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: chunk_levels,
                    bits_per_value: 16,
                    num_values: chunk_num_values,
                    block_info: BlockInfo::new(),
                    used_encoding: UsedEncoding::new(),
                });
                let compressed_levels = compressor.compress(chunk_levels_block)?;
                off += level_bytes;
                buffers.push(compressed_levels);
            }
            Ok((buffers, compressor_desc))
        } else {
            // Everything is valid or we have no repetition so we encode as a constant
            // array of 0
            let data = chunks.iter().map(|_| LanceBuffer::empty()).collect();
            let scalar = 0_u16.to_le_bytes().to_vec();
            let encoding = ProtobufUtils::constant(scalar, num_values);
            Ok((data, encoding))
        }
    }

    fn encode_simple_all_null(
        column_idx: u32,
        num_rows: u64,
        row_number: u64,
    ) -> Result<EncodedPage> {
        let description = ProtobufUtils::simple_all_null_layout();
        Ok(EncodedPage {
            column_idx,
            data: vec![],
            description: PageEncoding::Structural(description),
            num_rows,
            row_number,
        })
    }

    fn encode_miniblock(
        column_idx: u32,
        field: &Field,
        compression_strategy: &dyn CompressionStrategy,
        data: DataBlock,
        repdefs: Vec<RepDefBuilder>,
        row_number: u64,
    ) -> Result<EncodedPage> {
        let repdef = RepDefBuilder::serialize(repdefs);

        if let DataBlock::AllNull(_null_block) = data {
            // If we got here then all the data is null but we have rep/def information that
            // we need to store.
            todo!()
        }

        let num_values = data.num_values();
        // The validity is encoded in repdef so we can remove it
        let data = data.remove_validity();

        let compressor = compression_strategy.create_miniblock_compressor(field, &data)?;
        let (compressed_data, value_encoding) = compressor.compress(data)?;

        let (compressed_rep, rep_encoding) = Self::compress_levels(
            repdef.repetition_levels,
            num_values,
            compression_strategy,
            &compressed_data.chunks,
        )?;

        let (compressed_def, def_encoding) = Self::compress_levels(
            repdef.definition_levels,
            num_values,
            compression_strategy,
            &compressed_data.chunks,
        )?;

        // TODO: Parquet sparsely encodes values here.  We could do the same but
        // then we won't have log2 values per chunk.  This means more metadata
        // and potentially more decoder asymmetry.  However, it may be worth
        // investigating at some point

        let (block_value_buffer, block_meta_buffer) =
            Self::serialize_miniblocks(compressed_data, compressed_rep, compressed_def);

        let description =
            ProtobufUtils::miniblock_layout(rep_encoding, def_encoding, value_encoding);
        Ok(EncodedPage {
            num_rows: num_values,
            column_idx,
            data: vec![block_meta_buffer, block_value_buffer],
            description: PageEncoding::Structural(description),
            row_number,
        })
    }

    // Creates an encode task, consuming all buffered data
    fn do_flush(
        &mut self,
        arrays: Vec<ArrayRef>,
        repdefs: Vec<RepDefBuilder>,
        row_number: u64,
    ) -> Result<Vec<EncodeTask>> {
        let column_idx = self.column_index;
        let compression_strategy = self.compression_strategy.clone();
        let field = self.field.clone();
        let task = spawn_cpu(move || {
            let num_values = arrays.iter().map(|arr| arr.len() as u64).sum();
            let num_nulls = arrays
                .iter()
                .map(|arr| arr.logical_nulls().map(|n| n.null_count()).unwrap_or(0) as u64)
                .sum::<u64>();

            if num_values == num_nulls && repdefs.iter().all(|rd| rd.is_simple_validity()) {
                log::debug!(
                    "Encoding column {} with {} rows using simple-null layout",
                    column_idx,
                    num_values
                );
                Self::encode_simple_all_null(column_idx, num_values, row_number)
            } else {
                let data_block = DataBlock::from_arrays(&arrays, num_values);
                log::debug!(
                    "Encoding column {} with {} rows using mini-block layout",
                    column_idx,
                    num_values
                );
                if Self::is_narrow(&data_block) {
                    Self::encode_miniblock(
                        column_idx,
                        &field,
                        compression_strategy.as_ref(),
                        data_block,
                        repdefs,
                        row_number,
                    )
                } else {
                    todo!("Full zipped encoding")
                }
            }
        })
        .boxed();
        Ok(vec![task])
    }

    fn extract_validity_buf(array: &dyn Array, repdef: &mut RepDefBuilder) {
        if let Some(validity) = array.nulls() {
            repdef.add_validity_bitmap(validity.clone());
        } else {
            repdef.add_no_null(array.len());
        }
    }

    fn extract_validity(array: &dyn Array, repdef: &mut RepDefBuilder) {
        match array.data_type() {
            DataType::Null => {
                repdef.add_validity_bitmap(NullBuffer::new(BooleanBuffer::new_unset(array.len())));
            }
            DataType::FixedSizeList(_, _) => {
                Self::extract_validity_buf(array, repdef);
                Self::extract_validity(array.as_fixed_size_list().values(), repdef);
            }
            DataType::Dictionary(_, _) => {
                unreachable!()
            }
            _ => Self::extract_validity_buf(array, repdef),
        }
    }
}

impl FieldEncoder for PrimitiveStructuralEncoder {
    // Buffers data, if there is enough to write a page then we create an encode task
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        _external_buffers: &mut OutOfLineBuffers,
        mut repdef: RepDefBuilder,
        row_number: u64,
    ) -> Result<Vec<EncodeTask>> {
        Self::extract_validity(array.as_ref(), &mut repdef);
        self.accumulated_repdefs.push(repdef);

        if let Some((arrays, row_number)) = self.accumulation_queue.insert(array, row_number) {
            let accumulated_repdefs = std::mem::take(&mut self.accumulated_repdefs);
            Ok(self.do_flush(arrays, accumulated_repdefs, row_number)?)
        } else {
            Ok(vec![])
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self, _external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        if let Some((arrays, row_number)) = self.accumulation_queue.flush() {
            let accumulated_repdefs = std::mem::take(&mut self.accumulated_repdefs);
            Ok(self.do_flush(arrays, accumulated_repdefs, row_number)?)
        } else {
            Ok(vec![])
        }
    }

    fn num_columns(&self) -> u32 {
        1
    }

    fn finish(
        &mut self,
        _external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        std::future::ready(Ok(vec![EncodedColumn::default()])).boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int8Array, StringArray};

    use crate::encodings::logical::primitive::PrimitiveStructuralEncoder;

    use super::DataBlock;

    #[test]
    fn test_is_narrow() {
        let int8_array = Int8Array::from(vec![1, 2, 3]);
        let array_ref: ArrayRef = Arc::new(int8_array);
        let block = DataBlock::from_array(array_ref);

        assert!(PrimitiveStructuralEncoder::is_narrow(&block));

        let string_array = StringArray::from(vec![Some("hello"), Some("world")]);
        let block = DataBlock::from_array(string_array);
        assert!(PrimitiveStructuralEncoder::is_narrow(&block));

        let string_array = StringArray::from(vec![
            Some("hello world".repeat(100)),
            Some("world".to_string()),
        ]);
        let block = DataBlock::from_array(string_array);
        assert!((!PrimitiveStructuralEncoder::is_narrow(&block)));
    }
}
