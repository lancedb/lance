// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::VecDeque, fmt::Debug, iter, ops::Range, sync::Arc, u64, vec};

use arrow::array::AsArray;
use arrow_array::{make_array, Array, ArrayRef, UInt64Array};
use arrow_buffer::bit_util;
use arrow_schema::{DataType, Field as ArrowField};
use futures::{future::BoxFuture, stream::FuturesUnordered, FutureExt, TryStreamExt};
use lance_arrow::deepcopy::deep_copy_array;
use log::{debug, trace};
use snafu::{location, Location};

use lance_core::{datatypes::Field, utils::tokio::spawn_cpu, Result};

use crate::{
    buffer::LanceBuffer,
    data::{DataBlock, DataBlockBuilder, FixedWidthDataBlock},
    decoder::{
        ColumnInfo, DecodeArrayTask, DecodePageTask, DecodedArray, DecodedPage,
        DecompressorStrategy, FieldScheduler, FilterExpression, FixedPerValueDecompressor,
        LoadedPage, LogicalPageDecoder, MessageType, MiniBlockDecompressor, NextDecodeTask,
        PageEncoding, PageInfo, PageScheduler, PrimitivePageDecoder, PriorityRange,
        ScheduledScanLine, SchedulerContext, SchedulingJob, StructuralDecodeArrayTask,
        StructuralFieldDecoder, StructuralFieldScheduler, StructuralPageDecoder,
        StructuralSchedulingJob, UnloadedPage,
    },
    encoder::{
        ArrayEncodingStrategy, EncodeTask, EncodedColumn, EncodedPage, EncodingOptions,
        FieldEncoder, FixedPerValueCompressor, MiniBlockCompressed, MiniBlockCompressor,
        OutOfLineBuffers, MINIBLOCK_SECTOR_SIZE,
    },
    encodings::physical::{decoder_from_array_encoding, ColumnBuffers, PageBuffers},
    format::{
        pb::{self, ArrayEncoding},
        ProtobufUtils,
    },
    repdef::{LevelBuffer, RepDefBuilder, RepDefUnraveler, SerializedRepDefs},
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

pub trait StructuralPageScheduler: std::fmt::Debug + Send {
    fn initialize<'a>(&'a mut self, io: &Arc<dyn EncodingsIo>) -> BoxFuture<'a, Result<()>>;
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &dyn EncodingsIo,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>>;
}

#[derive(Debug)]
struct ChunkMeta {
    num_values: u64,
    value_size_bytes: u64,
    total_size_bytes: u64,
}

#[derive(Debug)]
struct DecodeMiniBlockTask {
    rep_decompressor: Arc<dyn FixedPerValueDecompressor>,
    def_decompressor: Arc<dyn FixedPerValueDecompressor>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    chunks: Vec<ScheduledChunk>,
    num_rows: u64,
}

impl DecodePageTask for DecodeMiniBlockTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let mut repbuf = Vec::with_capacity(self.num_rows as usize);
        let mut defbuf = Vec::with_capacity(self.num_rows as usize);

        let bits_per_rep = self.rep_decompressor.bits_per_value();
        let bits_per_def = self.def_decompressor.bits_per_value();
        assert!(bits_per_rep % 8 == 0);
        assert!(bits_per_def % 8 == 0);

        let bytes_per_rep = bits_per_rep / 8;
        let bytes_per_def = bits_per_def / 8;

        let mut remaining = self.num_rows;
        let estimated_size_bytes = self
            .chunks
            .iter()
            .map(|chunk| chunk.data.len())
            .sum::<usize>()
            * 2;
        let mut data_builder =
            DataBlockBuilder::with_capacity_estimate(estimated_size_bytes as u64);
        for chunk in self.chunks.into_iter() {
            debug_assert!(remaining >= chunk.ranges.iter().map(|r| r.end - r.start).sum());
            let buf = chunk.data.into_buffer();
            let bytes_rep = (bytes_per_rep * chunk.vals_in_chunk) as usize;
            let bytes_def = (bytes_per_def * chunk.vals_in_chunk) as usize;
            let rep = buf.slice_with_length(0, bytes_rep);
            let def = buf.slice_with_length(bytes_rep, bytes_def);
            let values = buf.slice(bytes_rep + bytes_def);
            println!(
                "Rep   : {}",
                LanceBuffer::Borrowed(rep.clone()).as_spaced_hex(2)
            );
            println!(
                "Def   : {}",
                LanceBuffer::Borrowed(def.clone()).as_spaced_hex(2)
            );
            println!(
                "Values: {}",
                LanceBuffer::Borrowed(values.clone()).as_spaced_hex(1)
            );

            let values = self
                .value_decompressor
                .decompress(LanceBuffer::Borrowed(values), chunk.vals_in_chunk)?;
            data_builder.append(values, &chunk.ranges);

            let rep = self
                .rep_decompressor
                .decompress(LanceBuffer::Borrowed(rep), chunk.vals_in_chunk)?;
            let mut rep = rep.as_fixed_width().unwrap();
            let rep = rep.data.borrow_to_typed_slice::<u16>();
            let def = self
                .def_decompressor
                .decompress(LanceBuffer::Borrowed(def), chunk.vals_in_chunk)?;
            let mut def = def.as_fixed_width().unwrap();
            let def = def.data.borrow_to_typed_slice::<u16>();
            for range in chunk.ranges {
                repbuf.extend_from_slice(&rep.as_ref()[range.start as usize..range.end as usize]);
                dbg!(&repbuf);
                defbuf.extend_from_slice(&def.as_ref()[range.start as usize..range.end as usize]);
                dbg!(&defbuf);
                remaining -= range.end - range.start;
            }
        }
        debug_assert_eq!(remaining, 0);

        let data = data_builder.finish();
        Ok(DecodedPage {
            data,
            repetition: Some(repbuf),
            definition: Some(defbuf),
        })
    }
}

#[derive(Debug)]
struct MiniBlockDecoder {
    rep_decompressor: Arc<dyn FixedPerValueDecompressor>,
    def_decompressor: Arc<dyn FixedPerValueDecompressor>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    data: VecDeque<ScheduledChunk>,
    offset_in_current_chunk: u64,
    num_rows: u64,
}

impl StructuralPageDecoder for MiniBlockDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let mut remaining = num_rows;
        let mut chunks = Vec::new();
        while remaining > 0 {
            if remaining >= self.data.front().unwrap().vals_in_chunk - self.offset_in_current_chunk
            {
                let chunk = self.data.pop_front().unwrap();
                remaining -= chunk.vals_in_chunk;
                chunks.push(chunk);
                self.offset_in_current_chunk = 0;
            } else {
                let mut chunk = self.data.front().unwrap().clone();
                chunk.vals_in_chunk = remaining;
                self.offset_in_current_chunk = remaining;
                remaining = 0;
                chunks.push(chunk);
            }
        }
        Ok(Box::new(DecodeMiniBlockTask {
            chunks,
            rep_decompressor: self.rep_decompressor.clone(),
            def_decompressor: self.def_decompressor.clone(),
            value_decompressor: self.value_decompressor.clone(),
            num_rows: num_rows,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

#[derive(Debug)]
pub struct MiniBlockScheduler {
    // These come from the protobuf
    meta_buf_position: u64,
    meta_buf_size: u64,
    data_buf_position: u64,
    data_buf_size: u64,
    priority: u64,
    rows_in_page: u64,
    rep_decompressor: Arc<dyn FixedPerValueDecompressor>,
    def_decompressor: Arc<dyn FixedPerValueDecompressor>,
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
        let (data_buf_position, data_buf_size) = buffer_offsets_and_sizes[1];
        println!("Created miniblock scheduler with meta buffer at {} of size {} and data buffer at {} of size {}", meta_buf_position, meta_buf_size, data_buf_position, data_buf_size);
        let rep_decompressor = decompressors
            .create_fixed_per_value_decompressor(layout.rep_compression.as_ref().unwrap())?;
        let def_decompressor = decompressors
            .create_fixed_per_value_decompressor(layout.def_compression.as_ref().unwrap())?;
        let value_decompressor = decompressors
            .create_miniblock_decompressor(layout.value_compression.as_ref().unwrap())?;
        Ok(Self {
            meta_buf_position,
            meta_buf_size,
            data_buf_position,
            data_buf_size,
            rep_decompressor: rep_decompressor.into(),
            def_decompressor: def_decompressor.into(),
            value_decompressor: value_decompressor.into(),
            priority,
            rows_in_page,
            chunk_meta: Vec::new(),
        })
    }

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
    vals_in_chunk: u64,
    ranges: Vec<Range<u64>>,
}

impl Clone for ScheduledChunk {
    fn clone(&self) -> Self {
        Self {
            data: self.data.try_clone().unwrap(),
            vals_in_chunk: self.vals_in_chunk.clone(),
            ranges: self.ranges.clone(),
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
            println!(
                "Processing mini block metadata: {}",
                LanceBuffer::from_bytes(bytes.clone(), 1).as_hex()
            );
            self.chunk_meta.reserve(bytes.len());
            let mut rows_counter = 0;
            let mut bytes_counter = 0;
            for (byte_idx, byte) in bytes.iter().enumerate() {
                let log_num_values = byte & 0x0F;
                let num_values = if log_num_values > 0 {
                    1 << log_num_values
                } else {
                    0
                };
                let num_bytes = (byte >> 4) as u64 * MINIBLOCK_SECTOR_SIZE;
                rows_counter += num_values;
                if byte_idx < bytes.len() - 1 {
                    debug_assert!(num_values > 0);
                    debug_assert!(num_bytes > 0);
                    let bytes_rep =
                        (self.rep_decompressor.bits_per_value() + num_values).div_ceil(8);
                    let bytes_def =
                        (self.def_decompressor.bits_per_value() + num_values).div_ceil(8);
                    let total_size = bytes_rep + bytes_def + num_bytes;
                    self.chunk_meta.push(ChunkMeta {
                        num_values: num_values as u64,
                        value_size_bytes: num_bytes,
                        total_size_bytes: total_size,
                    });
                    bytes_counter += total_size;
                } else {
                    debug_assert_eq!(num_values, 0);
                    debug_assert_eq!(num_bytes, 0);
                    let bytes_rep =
                        (self.rep_decompressor.bits_per_value() + self.rows_in_page).div_ceil(8);
                    let bytes_def =
                        (self.def_decompressor.bits_per_value() + self.rows_in_page).div_ceil(8);
                    let total_size = self.data_buf_size - bytes_counter;
                    let value_size = total_size - bytes_rep - bytes_def;
                    self.chunk_meta.push(ChunkMeta {
                        num_values: self.rows_in_page - rows_counter,
                        value_size_bytes: value_size,
                        total_size_bytes: total_size,
                    })
                }
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
        };

        // There can be both multiple ranges per chunk and multiple chunks per range
        for range in ranges {
            num_rows += range.end - range.start;
            println!("Processing range {:?}", range);
            let mut range = range.clone();
            while !range.is_empty() {
                println!(
                    "Comparing with chunk starting at {} with {} values and {} bytes of values and {} total bytes",
                    row_offset, current_chunk.num_values, current_chunk.value_size_bytes, current_chunk.total_size_bytes
                );
                Self::calc_overlap(
                    &mut range,
                    current_chunk,
                    row_offset,
                    &mut current_scheduled_chunk,
                );
                println!("Range after overlap: {:?}", range);
                // Might be empty if entire chunk is skipped
                if !range.is_empty() {
                    if !current_scheduled_chunk.ranges.is_empty() {
                        scheduled_chunks.push_back(current_scheduled_chunk);
                        ranges_to_req.push(
                            (self.data_buf_position + bytes_offset)
                                ..(self.data_buf_position
                                    + bytes_offset
                                    + current_chunk.total_size_bytes),
                        );
                    }
                    row_offset += current_chunk.num_values;
                    bytes_offset += current_chunk.total_size_bytes;
                    if let Some(next_chunk) = chunk_meta_iter.next() {
                        current_chunk = next_chunk;
                    }
                    current_scheduled_chunk = ScheduledChunk {
                        data: LanceBuffer::empty(),
                        ranges: Vec::new(),
                        vals_in_chunk: current_chunk.num_values,
                    };
                }
            }
        }
        if !current_scheduled_chunk.ranges.is_empty() {
            scheduled_chunks.push_back(current_scheduled_chunk);
            ranges_to_req.push(
                (self.data_buf_position + bytes_offset)
                    ..(self.data_buf_position + bytes_offset + current_chunk.total_size_bytes),
            );
        }

        println!(
            "scheduled_chunks={:?} ranges_to_req={:?}",
            scheduled_chunks, ranges_to_req
        );

        let data = io.submit_request(ranges_to_req, self.priority);

        let rep_decompressor = self.rep_decompressor.clone();
        let def_decompressor = self.def_decompressor.clone();
        let value_decompressor = self.value_decompressor.clone();
        Ok(async move {
            let data = data.await?;
            for (chunk, data) in scheduled_chunks.iter_mut().zip(data) {
                chunk.data = LanceBuffer::from_bytes(data, 1);
                println!("Chunk data: {}", chunk.data.as_spaced_hex(1));
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
            page_schedulers: page_schedulers,
            column_index: column_info.index,
        })
    }

    fn page_info_to_scheduler(
        page_info: &PageInfo,
        page_index: usize,
        decompressors: &dyn DecompressorStrategy,
    ) -> Result<PageInfoAndScheduler> {
        let scheduler = match page_info.encoding.as_structural().layout.as_ref() {
            Some(pb::page_layout::Layout::MiniBlockLayout(mini_block)) => {
                Box::new(MiniBlockScheduler::try_new(
                    &page_info.buffer_offsets_and_sizes,
                    page_info.priority,
                    page_info.num_rows,
                    mini_block,
                    decompressors,
                )?)
            }
            _ => todo!(),
        };
        Ok(PageInfoAndScheduler {
            page_index,
            num_rows: page_info.num_rows,
            scheduler: scheduler,
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
        let mut repdef = if has_def {
            RepDefUnraveler::new_with_def(all_rep, all_def)
        } else {
            RepDefUnraveler::new_all_valid(all_rep)
        };
        println!("Repdef Unraveler: {:?}", repdef);

        // The primitive array itself has a validity
        let validity = repdef.unravel_validity();
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

pub struct RepDefPrimitiveFieldEncoder {
    accumulation_queue: AccumulationQueue,
    accumulated_repdefs: Vec<RepDefBuilder>,
    array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
    repdef_compressor: Arc<dyn FixedPerValueCompressor>,
    column_index: u32,
    field: Field,
}

impl RepDefPrimitiveFieldEncoder {
    pub fn try_new(
        options: &EncodingOptions,
        array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
        column_index: u32,
        field: Field,
    ) -> Result<Self> {
        // Dummy field to pick the compression to use for the repdef data
        // TODO: We definitely want bit packing here.  Parquet also uses RLE which seems like
        // a good idea
        let repdef_field = Field::new_arrow("", DataType::UInt16, false)?;
        let repdef_compressor = array_encoding_strategy.create_fixed_per_value(&repdef_field)?;
        Ok(Self {
            accumulation_queue: AccumulationQueue::new(
                options.cache_bytes_per_column,
                column_index,
                options.keep_original_array,
            ),
            accumulated_repdefs: Vec::new(),
            column_index,
            array_encoding_strategy,
            field,
            repdef_compressor: repdef_compressor.into(),
        })
    }

    fn is_narrow(num_rows: u64, num_bytes: u64) -> bool {
        let avg_bytes_per_row = num_bytes as f64 / num_rows as f64;
        avg_bytes_per_row < 128.0
    }

    fn compress_repdefs(
        serialized: Vec<SerializedRepDefs>,
        repdef_compressor: &dyn FixedPerValueCompressor,
    ) -> Result<(
        FixedWidthDataBlock,
        ArrayEncoding,
        FixedWidthDataBlock,
        ArrayEncoding,
    )> {
        let num_values = serialized
            .iter()
            .map(|s| s.definition_levels.len())
            .sum::<usize>();
        // FIXME: Handle all-null / all-valid cases
        let mut def_levels = Vec::with_capacity(num_values);
        let mut rep_levels = Vec::with_capacity(num_values);
        for s in serialized {
            def_levels.extend(s.definition_levels);
            rep_levels.extend(s.repetition_levels);
        }
        let def_levels_buf = LanceBuffer::reinterpret_vec(def_levels);
        let def_levels_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 16,
            data: def_levels_buf,
            num_values: num_values as u64,
        });
        let (compressed_def, def_encoding) = repdef_compressor.compress(def_levels_block)?;

        let rep_levels_buf = LanceBuffer::reinterpret_vec(rep_levels);
        let rep_levels_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 16,
            data: rep_levels_buf,
            num_values: num_values as u64,
        });
        let (compressed_rep, rep_encoding) = repdef_compressor.compress(rep_levels_block)?;

        Ok((compressed_rep, rep_encoding, compressed_def, def_encoding))
    }

    fn serialize_miniblocks(
        miniblocks: MiniBlockCompressed,
        rep: FixedWidthDataBlock,
        def: FixedWidthDataBlock,
    ) -> LanceBuffer {
        // TODO: The logic here will get trickier when rep/def are not a multiple of 8
        // bits per value.  We'll need to do some bit shifting as we store them.
        //
        // Although maybe it would be simpler to compress them in chunks that match the
        // miniblock chunk lengths.
        assert!(rep.bits_per_value % 8 == 0);
        assert!(def.bits_per_value % 8 == 0);

        // We may have up to 1 byte per chunk of overhead due to bits per rep / def not being a multiple of 8
        // TODO: This above comment is talking about a future where we actually compress the rep/def
        let mut buffer = Vec::with_capacity(
            miniblocks.data.len() + rep.data.len() + def.data.len() + (miniblocks.chunks.len() * 2),
        );
        // These better be 2 right now
        let bytes_per_rep = rep.bits_per_value / 8;
        let bytes_per_def = def.bits_per_value / 8;

        let mut rep_offset = 0;
        let mut def_offset = 0;
        let mut value_offset = 0;
        for chunk in miniblocks.chunks {
            let num_vals = if chunk.log_num_values == 0 {
                miniblocks.num_values - value_offset
            } else {
                chunk.num_values()
            };

            let num_rep_bytes = num_vals * bytes_per_rep;
            let num_def_bytes = num_vals * bytes_per_def;
            let rep = &rep.data[rep_offset as usize..rep_offset as usize + num_rep_bytes as usize];
            buffer.extend_from_slice(rep);
            let def = &def.data[def_offset as usize..def_offset as usize + num_def_bytes as usize];
            buffer.extend_from_slice(def);

            let num_value_bytes = if chunk.log_num_values == 0 {
                miniblocks.data.len() - value_offset as usize
            } else {
                chunk.num_bytes() as usize
            };
            let values =
                &miniblocks.data[value_offset as usize..value_offset as usize + num_value_bytes];
            buffer.extend_from_slice(values);

            rep_offset += num_rep_bytes;
            def_offset += num_def_bytes;
            value_offset += num_value_bytes as u64;
        }

        LanceBuffer::Owned(buffer)
    }

    // TODO: We could potentially filter nulls but then how do we make sure we get
    // 2^N rows in each block?  Maybe pass the null indices to the compressor and let
    // it make sure to count them?  Though that might end up giving us tiny blocks.  Something
    // to investigate.
    #[allow(dead_code)]
    fn filter_nulls(arrays: &[ArrayRef], repdefs: &[SerializedRepDefs]) -> Result<Vec<ArrayRef>> {
        // FIXME: I think I've made this harder than it needs to be because I was thinking the # of
        // arrays and the # of repdefs might not be the same.
        let mut offset = 0;
        let valid_indices = repdefs.iter().flat_map(|repdef| {
            let offset_copy = offset;
            let valid = repdef
                .definition_levels
                .iter()
                .enumerate()
                .filter_map(move |(i, d)| if *d == 0 { Some(i + offset_copy) } else { None });
            offset += repdef.len();
            valid
        });
        let mut valid_indices_per_arr = Vec::with_capacity(arrays.len());
        let mut arrays_iter = arrays.iter();
        let mut next_len = arrays_iter.next().unwrap().len();
        let mut curr_valid_indices = Vec::with_capacity(next_len);
        let mut offset = 0;
        for idx in valid_indices {
            if idx < next_len {
                curr_valid_indices.push(idx as u64 - offset);
            } else {
                offset = next_len as u64;
                valid_indices_per_arr.push(curr_valid_indices);
                let arr_len = arrays_iter.next().unwrap().len();
                curr_valid_indices = Vec::with_capacity(arr_len);
                next_len += arr_len;
            }
        }
        valid_indices_per_arr.push(curr_valid_indices);

        let mut filtered_arrays = Vec::with_capacity(arrays.len());
        for (arr, indices) in arrays.iter().zip(valid_indices_per_arr) {
            if !indices.is_empty() {
                let indices = UInt64Array::from(indices);
                let filtered = arrow::compute::take(arr, &indices, None)?;
                filtered_arrays.push(filtered);
            }
        }
        Ok(filtered_arrays)
    }

    fn encode_miniblock(
        column_idx: u32,
        compressor: Box<dyn MiniBlockCompressor>,
        repdef_compressor: Arc<dyn FixedPerValueCompressor>,
        arrays: Vec<ArrayRef>,
        repdefs: Vec<RepDefBuilder>,
        num_values: u64,
        row_number: u64,
    ) -> Result<EncodedPage> {
        let serialized_repdefs = repdefs
            .into_iter()
            .map(|r| r.serialize())
            .collect::<Vec<_>>();

        let data = DataBlock::from_arrays(&arrays, num_values).remove_validity();

        let repdefs = Self::compress_repdefs(serialized_repdefs, repdef_compressor.as_ref())?;

        let (compressed_data, value_encoding) = compressor.compress(data)?;

        let block_meta_buffer = compressed_data.serialize_meta();
        let block_value_buffer = Self::serialize_miniblocks(compressed_data, repdefs.0, repdefs.2);

        let description = ProtobufUtils::miniblock(repdefs.1, repdefs.3, value_encoding);
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
        let array_encoding_strategy = self.array_encoding_strategy.clone();
        let field = self.field.clone();
        let repdef_compressor = self.repdef_compressor.clone();
        let task = spawn_cpu(move || {
            let num_values = arrays.iter().map(|arr| arr.len() as u64).sum();
            let num_bytes = arrays
                .iter()
                .map(|arr| arr.get_buffer_memory_size() as u64)
                .sum();

            // TODO: Calculation of statistics that can be used to choose compression algorithm

            if Self::is_narrow(num_values, num_bytes) {
                let compressor = array_encoding_strategy.create_miniblock_compressor(&field)?;
                Self::encode_miniblock(
                    column_idx,
                    compressor,
                    repdef_compressor,
                    arrays,
                    repdefs,
                    num_values,
                    row_number,
                )
            } else {
                todo!("Full zipped encoding")
            }
        })
        .boxed();
        Ok(vec![task])
    }

    fn extract_validity_buf(array: &dyn Array, repdef: &mut RepDefBuilder) {
        if array.null_count() == 0 {
            repdef.add_no_null(array.len());
        } else if array.null_count() == array.len() {
            repdef.add_all_null(array.len());
        } else {
            repdef.add_validity_bitmap(array.nulls().cloned().unwrap());
        }
    }

    fn extract_validity(array: &dyn Array, repdef: &mut RepDefBuilder) {
        match array.data_type() {
            DataType::Null => {
                repdef.add_all_null(array.len());
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

impl FieldEncoder for RepDefPrimitiveFieldEncoder {
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
