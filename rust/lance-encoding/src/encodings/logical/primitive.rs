// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::{HashMap, VecDeque},
    env,
    fmt::Debug,
    iter,
    ops::Range,
    sync::Arc,
    vec,
};

use crate::constants::{
    DICT_DIVISOR_META_KEY, STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY,
    STRUCTURAL_ENCODING_MINIBLOCK,
};
use arrow::array::AsArray;
use arrow_array::{make_array, types::UInt64Type, Array, ArrayRef, PrimitiveArray};
use arrow_buffer::{BooleanBuffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field as ArrowField};
use futures::{future::BoxFuture, stream::FuturesOrdered, FutureExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::deepcopy::deep_copy_nulls;
use lance_core::{
    cache::{CacheKey, Context, DeepSizeOf},
    error::Error,
    utils::{bit::pad_bytes, hash::U8SliceKey},
};
use log::trace;
use snafu::location;

use crate::{
    compression::{
        BlockDecompressor, CompressionStrategy, DecompressionStrategy, MiniBlockDecompressor,
    },
    data::{AllNullDataBlock, DataBlock, VariableWidthBlock},
    utils::bytepack::BytepackedIntegerEncoder,
};
use crate::{
    compression::{FixedPerValueDecompressor, VariablePerValueDecompressor},
    encodings::logical::primitive::fullzip::PerValueDataBlock,
};
use crate::{
    encodings::logical::primitive::miniblock::MiniBlockChunk, utils::bytepack::ByteUnpacker,
};
use crate::{
    encodings::logical::primitive::miniblock::MiniBlockCompressed,
    statistics::{ComputeStat, GetStat, Stat},
};
use crate::{
    repdef::{
        build_control_word_iterator, CompositeRepDefUnraveler, ControlWordIterator,
        ControlWordParser, DefinitionInterpretation, RepDefSlicer,
    },
    utils::accumulation::AccumulationQueue,
};
use lance_core::{datatypes::Field, utils::tokio::spawn_cpu, Result};

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlockBuilder, FixedWidthDataBlock},
    decoder::{
        ColumnInfo, DecodePageTask, DecodedArray, DecodedPage, FilterExpression, LoadedPage,
        MessageType, PageEncoding, PageInfo, ScheduledScanLine, SchedulerContext,
        StructuralDecodeArrayTask, StructuralFieldDecoder, StructuralFieldScheduler,
        StructuralPageDecoder, StructuralSchedulingJob, UnloadedPage,
    },
    encoder::{
        EncodeTask, EncodedColumn, EncodedPage, EncodingOptions, FieldEncoder, OutOfLineBuffers,
    },
    format::{pb, ProtobufUtils},
    repdef::{LevelBuffer, RepDefBuilder, RepDefUnraveler},
    EncodingsIo,
};

pub mod fullzip;
pub mod miniblock;

const FILL_BYTE: u8 = 0xFE;

/// A trait for figuring out how to schedule the data within
/// a single page.
trait StructuralPageScheduler: std::fmt::Debug + Send {
    /// Fetches any metadata required for the page
    fn initialize<'a>(
        &'a mut self,
        io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>>;
    /// Loads metadata from a previous initialize call
    fn load(&mut self, data: &Arc<dyn CachedPageData>);
    /// Schedules the read of the given ranges in the page
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>>;
}

/// Metadata describing the decoded size of a mini-block
#[derive(Debug)]
struct ChunkMeta {
    num_values: u64,
    chunk_size_bytes: u64,
    offset_bytes: u64,
}

/// A mini-block chunk that has been decoded and decompressed
#[derive(Debug)]
struct DecodedMiniBlockChunk {
    rep: Option<ScalarBuffer<u16>>,
    def: Option<ScalarBuffer<u16>>,
    values: DataBlock,
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
    rep_decompressor: Option<Arc<dyn BlockDecompressor>>,
    def_decompressor: Option<Arc<dyn BlockDecompressor>>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    dictionary_data: Option<Arc<DataBlock>>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    num_buffers: u64,
    max_visible_level: u16,
    instructions: Vec<(ChunkDrainInstructions, LoadedChunk)>,
}

impl DecodeMiniBlockTask {
    fn decode_levels(
        rep_decompressor: &dyn BlockDecompressor,
        levels: LanceBuffer,
        num_levels: u16,
    ) -> Result<ScalarBuffer<u16>> {
        let rep = rep_decompressor.decompress(levels, num_levels as u64)?;
        let mut rep = rep.as_fixed_width().unwrap();
        debug_assert_eq!(rep.num_values, num_levels as u64);
        debug_assert_eq!(rep.bits_per_value, 16);
        Ok(rep.data.borrow_to_typed_slice::<u16>())
    }

    // We are building a LevelBuffer (levels) and want to copy into it `total_len`
    // values from `level_buf` starting at `offset`.
    //
    // We need to handle both the case where `levels` is None (no nulls encountered
    // yet) and the case where `level_buf` is None (the input we are copying from has
    // no nulls)
    fn extend_levels(
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
                    LevelBuffer::with_capacity(dest_offset + (range.end - range.start) as usize);
                new_levels_vec.extend(iter::repeat_n(0, dest_offset));
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
            levels.extend(iter::repeat_n(0, num_values));
        }
    }

    /// Maps a range of rows to a range of items and a range of levels
    ///
    /// If there is no repetition information this just returns the range as-is.
    ///
    /// If there is repetition information then we need to do some work to figure out what
    /// range of items corresponds to the requested range of rows.
    ///
    /// For example, if the data is [[1, 2, 3], [4, 5], [6, 7]] and the range is 1..2 (i.e. just row
    /// 1) then the user actually wants items 3..5.  In the above case the rep levels would be:
    ///
    /// Idx: 0 1 2 3 4 5 6
    /// Rep: 1 0 0 1 0 1 0
    ///
    /// So the start (1) maps to the second 1 (idx=3) and the end (2) maps to the third 1 (idx=5)
    ///
    /// If there are invisible items then we don't count them when calculating the range of items we
    /// are interested in but we do count them when calculating the range of levels we are interested
    /// in.  As a result we have to return both the item range (first return value) and the level range
    /// (second return value).
    ///
    /// For example, if the data is [[1, 2, 3], [4, 5], NULL, [6, 7, 8]] and the range is 2..4 then the
    /// user wants items 5..8 but they want levels 5..9.  In the above case the rep/def levels would be:
    ///
    /// Idx: 0 1 2 3 4 5 6 7 8
    /// Rep: 1 0 0 1 0 1 1 0 0
    /// Def: 0 0 0 0 0 1 0 0 0
    /// Itm: 1 2 3 4 5 6 7 8
    ///
    /// Finally, we have to contend with the fact that chunks may or may not start with a "preamble" of
    /// trailing values that finish up a list from the previous chunk.  In this case the first item does
    /// not start at max_rep because it is a continuation of the previous chunk.  For our purposes we do
    /// not consider this a "row" and so the range 0..1 will refer to the first row AFTER the preamble.
    ///
    /// We have a separate parameter (`preamble_action`) to control whether we want the preamble or not.
    ///
    /// Note that the "trailer" is considered a "row" and if we want it we should include it in the range.
    fn map_range(
        range: Range<u64>,
        rep: Option<&impl AsRef<[u16]>>,
        def: Option<&impl AsRef<[u16]>>,
        max_rep: u16,
        max_visible_def: u16,
        // The total number of items (not rows) in the chunk.  This is not quite the same as
        // rep.len() / def.len() because it doesn't count invisible items
        total_items: u64,
        preamble_action: PreambleAction,
    ) -> (Range<u64>, Range<u64>) {
        if let Some(rep) = rep {
            let mut rep = rep.as_ref();
            // If there is a preamble and we need to skip it then do that first.  The work is the same
            // whether there is def information or not
            let mut items_in_preamble = 0;
            let first_row_start = match preamble_action {
                PreambleAction::Skip | PreambleAction::Take => {
                    let first_row_start = if let Some(def) = def.as_ref() {
                        let mut first_row_start = None;
                        for (idx, (rep, def)) in rep.iter().zip(def.as_ref()).enumerate() {
                            if *rep == max_rep {
                                first_row_start = Some(idx);
                                break;
                            }
                            if *def <= max_visible_def {
                                items_in_preamble += 1;
                            }
                        }
                        first_row_start
                    } else {
                        let first_row_start = rep.iter().position(|&r| r == max_rep);
                        items_in_preamble = first_row_start.unwrap_or(rep.len());
                        first_row_start
                    };
                    // It is possible for a chunk to be entirely partial values but if it is then it
                    // should never show up as a preamble to skip
                    if first_row_start.is_none() {
                        assert!(preamble_action == PreambleAction::Take);
                        return (0..total_items, 0..rep.len() as u64);
                    }
                    let first_row_start = first_row_start.unwrap() as u64;
                    rep = &rep[first_row_start as usize..];
                    first_row_start
                }
                PreambleAction::Absent => {
                    debug_assert!(rep[0] == max_rep);
                    0
                }
            };

            // We hit this case when all we needed was the preamble
            if range.start == range.end {
                debug_assert!(preamble_action == PreambleAction::Take);
                return (0..items_in_preamble as u64, 0..first_row_start);
            }
            assert!(range.start < range.end);

            let mut rows_seen = 0;
            let mut new_start = 0;
            let mut new_levels_start = 0;

            if let Some(def) = def {
                let def = &def.as_ref()[first_row_start as usize..];

                // range.start == 0 always maps to 0 (even with invis items), otherwise we need to walk
                let mut lead_invis_seen = 0;

                if range.start > 0 {
                    if def[0] > max_visible_def {
                        lead_invis_seen += 1;
                    }
                    for (idx, (rep, def)) in rep.iter().zip(def).skip(1).enumerate() {
                        if *rep == max_rep {
                            rows_seen += 1;
                            if rows_seen == range.start {
                                new_start = idx as u64 + 1 - lead_invis_seen;
                                new_levels_start = idx as u64 + 1;
                                break;
                            }
                            if *def > max_visible_def {
                                lead_invis_seen += 1;
                            }
                        }
                    }
                }

                rows_seen += 1;

                let mut new_end = u64::MAX;
                let mut new_levels_end = rep.len() as u64;
                let new_start_is_visible = def[new_levels_start as usize] <= max_visible_def;
                let mut tail_invis_seen = if new_start_is_visible { 0 } else { 1 };
                for (idx, (rep, def)) in rep[(new_levels_start + 1) as usize..]
                    .iter()
                    .zip(&def[(new_levels_start + 1) as usize..])
                    .enumerate()
                {
                    if *rep == max_rep {
                        rows_seen += 1;
                        if rows_seen == range.end + 1 {
                            new_end = idx as u64 + new_start + 1 - tail_invis_seen;
                            new_levels_end = idx as u64 + new_levels_start + 1;
                            break;
                        }
                        if *def > max_visible_def {
                            tail_invis_seen += 1;
                        }
                    }
                }

                if new_end == u64::MAX {
                    new_levels_end = rep.len() as u64;
                    // This is the total number of visible items (minus any items in the preamble)
                    let total_invis_seen = lead_invis_seen + tail_invis_seen;
                    new_end = rep.len() as u64 - total_invis_seen;
                }

                assert_ne!(new_end, u64::MAX);

                // Adjust for any skipped preamble
                if preamble_action == PreambleAction::Skip {
                    // TODO: Should this be items_in_preamble?  If so, add a
                    // unit test for this case
                    new_start += first_row_start;
                    new_end += first_row_start;
                    new_levels_start += first_row_start;
                    new_levels_end += first_row_start;
                } else if preamble_action == PreambleAction::Take {
                    debug_assert_eq!(new_start, 0);
                    debug_assert_eq!(new_levels_start, 0);
                    new_end += first_row_start;
                    new_levels_end += first_row_start;
                }

                (new_start..new_end, new_levels_start..new_levels_end)
            } else {
                // Easy case, there are no invisible items, so we don't need to check for them
                // The items range and levels range will be the same.  We do still need to walk
                // the rep levels to find the row boundaries

                // range.start == 0 always maps to 0, otherwise we need to walk
                if range.start > 0 {
                    for (idx, rep) in rep.iter().skip(1).enumerate() {
                        if *rep == max_rep {
                            rows_seen += 1;
                            if rows_seen == range.start {
                                new_start = idx as u64 + 1;
                                break;
                            }
                        }
                    }
                }
                let mut new_end = rep.len() as u64;
                // range.end == max_items always maps to rep.len(), otherwise we need to walk
                if range.end < total_items {
                    for (idx, rep) in rep[(new_start + 1) as usize..].iter().enumerate() {
                        if *rep == max_rep {
                            rows_seen += 1;
                            if rows_seen == range.end {
                                new_end = idx as u64 + new_start + 1;
                                break;
                            }
                        }
                    }
                }

                // Adjust for any skipped preamble
                if preamble_action == PreambleAction::Skip {
                    new_start += first_row_start;
                    new_end += first_row_start;
                } else if preamble_action == PreambleAction::Take {
                    debug_assert_eq!(new_start, 0);
                    new_end += first_row_start;
                }

                (new_start..new_end, new_start..new_end)
            }
        } else {
            // No repetition info, easy case, just use the range as-is and the item
            // and level ranges are the same
            (range.clone(), range)
        }
    }

    // Unserialize a miniblock into a collection of vectors
    fn decode_miniblock_chunk(
        &self,
        buf: &LanceBuffer,
        items_in_chunk: u64,
    ) -> Result<DecodedMiniBlockChunk> {
        let mut offset = 0;
        let num_levels = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
        offset += 2;

        let rep_size = if self.rep_decompressor.is_some() {
            let rep_size = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
            offset += 2;
            Some(rep_size)
        } else {
            None
        };
        let def_size = if self.def_decompressor.is_some() {
            let def_size = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
            offset += 2;
            Some(def_size)
        } else {
            None
        };
        let buffer_sizes = (0..self.num_buffers)
            .map(|_| {
                let size = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
                offset += 2;
                size
            })
            .collect::<Vec<_>>();

        offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);

        let rep = rep_size.map(|rep_size| {
            let rep = buf.slice_with_length(offset, rep_size as usize);
            offset += rep_size as usize;
            offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);
            rep
        });

        let def = def_size.map(|def_size| {
            let def = buf.slice_with_length(offset, def_size as usize);
            offset += def_size as usize;
            offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);
            def
        });

        let buffers = buffer_sizes
            .into_iter()
            .map(|buf_size| {
                let buf = buf.slice_with_length(offset, buf_size as usize);
                offset += buf_size as usize;
                offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);
                buf
            })
            .collect::<Vec<_>>();

        let values = self
            .value_decompressor
            .decompress(buffers, items_in_chunk)?;

        let rep = rep
            .map(|rep| {
                Self::decode_levels(
                    self.rep_decompressor.as_ref().unwrap().as_ref(),
                    rep,
                    num_levels,
                )
            })
            .transpose()?;
        let def = def
            .map(|def| {
                Self::decode_levels(
                    self.def_decompressor.as_ref().unwrap().as_ref(),
                    def,
                    num_levels,
                )
            })
            .transpose()?;

        Ok(DecodedMiniBlockChunk { rep, def, values })
    }
}

impl DecodePageTask for DecodeMiniBlockTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        // First, we create output buffers for the rep and def and data
        let mut repbuf: Option<LevelBuffer> = None;
        let mut defbuf: Option<LevelBuffer> = None;

        let max_rep = self.def_meaning.iter().filter(|l| l.is_list()).count() as u16;

        // This is probably an over-estimate but it's quick and easy to calculate
        let estimated_size_bytes = self
            .instructions
            .iter()
            .map(|(_, chunk)| chunk.data.len())
            .sum::<usize>()
            * 2;
        let mut data_builder =
            DataBlockBuilder::with_capacity_estimate(estimated_size_bytes as u64);

        // We need to keep track of the offset into repbuf/defbuf that we are building up
        let mut level_offset = 0;
        // Now we iterate through each instruction and process it
        for (instructions, chunk) in self.instructions.iter() {
            // TODO: It's very possible that we have duplicate `buf` in self.instructions and we
            // don't want to decode the buf again and again on the same thread.

            let DecodedMiniBlockChunk { rep, def, values } =
                self.decode_miniblock_chunk(&chunk.data, chunk.items_in_chunk)?;

            // Our instructions tell us which rows we want to take from this chunk
            let row_range_start =
                instructions.rows_to_skip + instructions.chunk_instructions.rows_to_skip;
            let row_range_end = row_range_start + instructions.rows_to_take;

            // We use the rep info to map the row range to an item range / levels range
            let (item_range, level_range) = Self::map_range(
                row_range_start..row_range_end,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                self.max_visible_level,
                chunk.items_in_chunk,
                instructions.preamble_action,
            );

            // Now we append the data to the output buffers
            Self::extend_levels(level_range.clone(), &mut repbuf, &rep, level_offset);
            Self::extend_levels(level_range.clone(), &mut defbuf, &def, level_offset);
            level_offset += (level_range.end - level_range.start) as usize;
            data_builder.append(&values, item_range);
        }

        let data = data_builder.finish();

        let unraveler = RepDefUnraveler::new(repbuf, defbuf, self.def_meaning.clone());

        // if dictionary encoding is applied, do dictionary decode here.
        if let Some(dictionary) = &self.dictionary_data {
            // assume the indices are uniformly distributed.
            let estimated_size_bytes = dictionary.data_size()
                * (data.num_values() + dictionary.num_values() - 1)
                / dictionary.num_values();
            let mut data_builder = DataBlockBuilder::with_capacity_estimate(estimated_size_bytes);

            // if dictionary encoding is applied, decode indices based on their actual bit width
            if let DataBlock::FixedWidth(mut fixed_width_data_block) = data {
                match fixed_width_data_block.bits_per_value {
                    32 => {
                        let indices = fixed_width_data_block.data.borrow_to_typed_slice::<i32>();
                        let indices = indices.as_ref();

                        indices.iter().for_each(|&idx| {
                            data_builder.append(dictionary, idx as u64..idx as u64 + 1);
                        });
                    }
                    64 => {
                        let indices = fixed_width_data_block.data.borrow_to_typed_slice::<i64>();
                        let indices = indices.as_ref();

                        indices.iter().for_each(|&idx| {
                            data_builder.append(dictionary, idx as u64..idx as u64 + 1);
                        });
                    }
                    _ => {
                        return Err(lance_core::Error::Internal {
                            message: format!(
                                "Unsupported dictionary index bit width: {} bits",
                                fixed_width_data_block.bits_per_value
                            ),
                            location: location!(),
                        });
                    }
                }

                let data = data_builder.finish();
                return Ok(DecodedPage {
                    data,
                    repdef: unraveler,
                });
            }
        }

        Ok(DecodedPage {
            data,
            repdef: unraveler,
        })
    }
}

/// A chunk that has been loaded by the miniblock scheduler (but not
/// yet decoded)
#[derive(Debug)]
struct LoadedChunk {
    data: LanceBuffer,
    items_in_chunk: u64,
    byte_range: Range<u64>,
    chunk_idx: usize,
}

impl Clone for LoadedChunk {
    fn clone(&self) -> Self {
        Self {
            // Safe as we always create borrowed buffers here
            data: self.data.try_clone().unwrap(),
            items_in_chunk: self.items_in_chunk,
            byte_range: self.byte_range.clone(),
            chunk_idx: self.chunk_idx,
        }
    }
}

/// Decodes mini-block formatted data.  See [`PrimitiveStructuralEncoder`] for more
/// details on the different layouts.
#[derive(Debug)]
struct MiniBlockDecoder {
    rep_decompressor: Option<Arc<dyn BlockDecompressor>>,
    def_decompressor: Option<Arc<dyn BlockDecompressor>>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    loaded_chunks: VecDeque<LoadedChunk>,
    instructions: VecDeque<ChunkInstructions>,
    offset_in_current_chunk: u64,
    num_rows: u64,
    num_buffers: u64,
    dictionary: Option<Arc<DataBlock>>,
}

/// See [`MiniBlockScheduler`] for more details on the scheduling and decoding
/// process for miniblock encoded data.
impl StructuralPageDecoder for MiniBlockDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let mut items_desired = num_rows;
        let mut need_preamble = false;
        let mut skip_in_chunk = self.offset_in_current_chunk;
        let mut drain_instructions = Vec::new();
        while items_desired > 0 || need_preamble {
            let (instructions, consumed) = self
                .instructions
                .front()
                .unwrap()
                .drain_from_instruction(&mut items_desired, &mut need_preamble, &mut skip_in_chunk);

            while self.loaded_chunks.front().unwrap().chunk_idx
                != instructions.chunk_instructions.chunk_idx
            {
                self.loaded_chunks.pop_front();
            }
            drain_instructions.push((instructions, self.loaded_chunks.front().unwrap().clone()));
            if consumed {
                self.instructions.pop_front();
            }
        }
        // We can throw away need_preamble here because it must be false.  If it were true it would mean
        // we were still in the middle of loading rows.  We do need to latch skip_in_chunk though.
        self.offset_in_current_chunk = skip_in_chunk;

        let max_visible_level = self
            .def_meaning
            .iter()
            .take_while(|l| !l.is_list())
            .map(|l| l.num_def_levels())
            .sum::<u16>();

        Ok(Box::new(DecodeMiniBlockTask {
            instructions: drain_instructions,
            def_decompressor: self.def_decompressor.clone(),
            rep_decompressor: self.rep_decompressor.clone(),
            value_decompressor: self.value_decompressor.clone(),
            dictionary_data: self.dictionary.clone(),
            def_meaning: self.def_meaning.clone(),
            num_buffers: self.num_buffers,
            max_visible_level,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

#[derive(Debug)]
struct CachedComplexAllNullState {
    rep: Option<ScalarBuffer<u16>>,
    def: Option<ScalarBuffer<u16>>,
}

impl DeepSizeOf for CachedComplexAllNullState {
    fn deep_size_of_children(&self, _ctx: &mut Context) -> usize {
        self.rep.as_ref().map(|buf| buf.len() * 2).unwrap_or(0)
            + self.def.as_ref().map(|buf| buf.len() * 2).unwrap_or(0)
    }
}

impl CachedPageData for CachedComplexAllNullState {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

/// A scheduler for all-null data that has repetition and definition levels
///
/// We still need to do some I/O in this case because we need to figure out what kind of null we
/// are dealing with (null list, null struct, what level null struct, etc.)
///
/// TODO: Right now we just load the entire rep/def at initialization time and cache it.  This is a touch
/// RAM aggressive and maybe we want something more lazy in the future.  On the other hand, it's simple
/// and fast so...maybe not :)
#[derive(Debug)]
pub struct ComplexAllNullScheduler {
    // Set from protobuf
    buffer_offsets_and_sizes: Arc<[(u64, u64)]>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    repdef: Option<Arc<CachedComplexAllNullState>>,
}

impl ComplexAllNullScheduler {
    pub fn new(
        buffer_offsets_and_sizes: Arc<[(u64, u64)]>,
        def_meaning: Arc<[DefinitionInterpretation]>,
    ) -> Self {
        Self {
            buffer_offsets_and_sizes,
            def_meaning,
            repdef: None,
        }
    }
}

impl StructuralPageScheduler for ComplexAllNullScheduler {
    fn initialize<'a>(
        &'a mut self,
        io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        // Fully load the rep & def buffers, as needed
        let (rep_pos, rep_size) = self.buffer_offsets_and_sizes[0];
        let (def_pos, def_size) = self.buffer_offsets_and_sizes[1];
        let has_rep = rep_size > 0;
        let has_def = def_size > 0;

        let mut reads = Vec::with_capacity(2);
        if has_rep {
            reads.push(rep_pos..rep_pos + rep_size);
        }
        if has_def {
            reads.push(def_pos..def_pos + def_size);
        }

        let data = io.submit_request(reads, 0);

        async move {
            let data = data.await?;
            let mut data_iter = data.into_iter();

            let rep = if has_rep {
                let rep = data_iter.next().unwrap();
                let mut rep = LanceBuffer::from_bytes(rep, 2);
                let rep = rep.borrow_to_typed_slice::<u16>();
                Some(rep)
            } else {
                None
            };

            let def = if has_def {
                let def = data_iter.next().unwrap();
                let mut def = LanceBuffer::from_bytes(def, 2);
                let def = def.borrow_to_typed_slice::<u16>();
                Some(def)
            } else {
                None
            };

            let repdef = Arc::new(CachedComplexAllNullState { rep, def });

            self.repdef = Some(repdef.clone());

            Ok(repdef as Arc<dyn CachedPageData>)
        }
        .boxed()
    }

    fn load(&mut self, data: &Arc<dyn CachedPageData>) {
        self.repdef = Some(
            data.clone()
                .as_arc_any()
                .downcast::<CachedComplexAllNullState>()
                .unwrap(),
        );
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        _io: &Arc<dyn EncodingsIo>,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>> {
        let ranges = VecDeque::from_iter(ranges.iter().cloned());
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        Ok(std::future::ready(Ok(Box::new(ComplexAllNullPageDecoder {
            ranges,
            rep: self.repdef.as_ref().unwrap().rep.clone(),
            def: self.repdef.as_ref().unwrap().def.clone(),
            num_rows,
            def_meaning: self.def_meaning.clone(),
        }) as Box<dyn StructuralPageDecoder>))
        .boxed())
    }
}

#[derive(Debug)]
pub struct ComplexAllNullPageDecoder {
    ranges: VecDeque<Range<u64>>,
    rep: Option<ScalarBuffer<u16>>,
    def: Option<ScalarBuffer<u16>>,
    num_rows: u64,
    def_meaning: Arc<[DefinitionInterpretation]>,
}

impl ComplexAllNullPageDecoder {
    fn drain_ranges(&mut self, num_rows: u64) -> Vec<Range<u64>> {
        let mut rows_desired = num_rows;
        let mut ranges = Vec::with_capacity(self.ranges.len());
        while rows_desired > 0 {
            let front = self.ranges.front_mut().unwrap();
            let avail = front.end - front.start;
            if avail > rows_desired {
                ranges.push(front.start..front.start + rows_desired);
                front.start += rows_desired;
                rows_desired = 0;
            } else {
                ranges.push(self.ranges.pop_front().unwrap());
                rows_desired -= avail;
            }
        }
        ranges
    }
}

impl StructuralPageDecoder for ComplexAllNullPageDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let drained_ranges = self.drain_ranges(num_rows);
        Ok(Box::new(DecodeComplexAllNullTask {
            ranges: drained_ranges,
            rep: self.rep.clone(),
            def: self.def.clone(),
            def_meaning: self.def_meaning.clone(),
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// We use `ranges` to slice into `rep` and `def` and create rep/def buffers
/// for the null data.
#[derive(Debug)]
pub struct DecodeComplexAllNullTask {
    ranges: Vec<Range<u64>>,
    rep: Option<ScalarBuffer<u16>>,
    def: Option<ScalarBuffer<u16>>,
    def_meaning: Arc<[DefinitionInterpretation]>,
}

impl DecodeComplexAllNullTask {
    fn decode_level(
        &self,
        levels: &Option<ScalarBuffer<u16>>,
        num_values: u64,
    ) -> Option<Vec<u16>> {
        levels.as_ref().map(|levels| {
            let mut referenced_levels = Vec::with_capacity(num_values as usize);
            for range in &self.ranges {
                referenced_levels.extend(
                    levels[range.start as usize..range.end as usize]
                        .iter()
                        .copied(),
                );
            }
            referenced_levels
        })
    }
}

impl DecodePageTask for DecodeComplexAllNullTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let num_values = self.ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        let data = DataBlock::AllNull(AllNullDataBlock { num_values });
        let rep = self.decode_level(&self.rep, num_values);
        let def = self.decode_level(&self.def, num_values);
        let unraveler = RepDefUnraveler::new(rep, def, self.def_meaning);
        Ok(DecodedPage {
            data,
            repdef: unraveler,
        })
    }
}

/// A scheduler for simple all-null data
///
/// "simple" all-null data is data that is all null and only has a single level of definition and
/// no repetition.  We don't need to read any data at all in this case.
#[derive(Debug, Default)]
pub struct SimpleAllNullScheduler {}

impl StructuralPageScheduler for SimpleAllNullScheduler {
    fn initialize<'a>(
        &'a mut self,
        _io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        std::future::ready(Ok(Arc::new(NoCachedPageData) as Arc<dyn CachedPageData>)).boxed()
    }

    fn load(&mut self, _cache: &Arc<dyn CachedPageData>) {}

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        _io: &Arc<dyn EncodingsIo>,
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
        let unraveler = RepDefUnraveler::new(
            None,
            Some(vec![1; self.num_values as usize]),
            Arc::new([DefinitionInterpretation::NullableItem]),
        );
        Ok(DecodedPage {
            data: DataBlock::AllNull(AllNullDataBlock {
                num_values: self.num_values,
            }),
            repdef: unraveler,
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

#[derive(Debug, Clone)]
struct MiniBlockSchedulerDictionary {
    // These come from the protobuf
    dictionary_decompressor: Arc<dyn BlockDecompressor>,
    dictionary_buf_position_and_size: (u64, u64),
    dictionary_data_alignment: u64,
    num_dictionary_items: u64,
}

/// Individual block metadata within a MiniBlock repetition index.
#[derive(Debug)]
struct MiniBlockRepIndexBlock {
    // The index of the first row that starts after the beginning of this block.  If the block
    // has a preamble this will be the row after the preamble.  If the block is entirely preamble
    // then this will be a row that starts in some future block.
    first_row: u64,
    // The number of rows in the block, including the trailer but not the preamble.
    // Can be 0 if the block is entirely preamble
    starts_including_trailer: u64,
    // Whether the block has a preamble
    has_preamble: bool,
    // Whether the block has a trailer
    has_trailer: bool,
}

impl DeepSizeOf for MiniBlockRepIndexBlock {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

/// Repetition index for MiniBlock encoding.
///
/// Stores block-level offset information to enable efficient random
/// access to nested data structures within mini-blocks.
#[derive(Debug)]
struct MiniBlockRepIndex {
    blocks: Vec<MiniBlockRepIndexBlock>,
}

impl DeepSizeOf for MiniBlockRepIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.blocks.deep_size_of_children(context)
    }
}

impl MiniBlockRepIndex {
    fn decode(rep_index: &[Vec<u64>]) -> Self {
        let mut chunk_has_preamble = false;
        let mut offset = 0;
        let mut blocks = Vec::with_capacity(rep_index.len());
        for chunk_rep in rep_index {
            let ends_count = chunk_rep[0];
            let partial_count = chunk_rep[1];

            let chunk_has_trailer = partial_count > 0;
            let mut starts_including_trailer = ends_count;
            if chunk_has_trailer {
                starts_including_trailer += 1;
            }
            if chunk_has_preamble {
                starts_including_trailer -= 1;
            }

            blocks.push(MiniBlockRepIndexBlock {
                first_row: offset,
                starts_including_trailer,
                has_preamble: chunk_has_preamble,
                has_trailer: chunk_has_trailer,
            });

            chunk_has_preamble = chunk_has_trailer;
            offset += starts_including_trailer;
        }

        Self { blocks }
    }
}

/// State that is loaded once and cached for future lookups
#[derive(Debug)]
struct MiniBlockCacheableState {
    /// Metadata that describes each chunk in the page
    chunk_meta: Vec<ChunkMeta>,
    /// The decoded repetition index
    rep_index: MiniBlockRepIndex,
    /// The dictionary for the page, if any
    dictionary: Option<Arc<DataBlock>>,
}

impl DeepSizeOf for MiniBlockCacheableState {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.rep_index.deep_size_of_children(context)
            + self
                .dictionary
                .as_ref()
                .map(|dict| dict.data_size() as usize)
                .unwrap_or(0)
    }
}

impl CachedPageData for MiniBlockCacheableState {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

/// A scheduler for a page that has been encoded with the mini-block layout
///
/// Scheduling mini-block encoded data is simple in concept and somewhat complex
/// in practice.
///
/// First, during initialization, we load the chunk metadata, the repetition index,
/// and the dictionary (these last two may not be present)
///
/// Then, during scheduling, we use the user's requested row ranges and the repetition
/// index to determine which chunks we need and which rows we need from those chunks.
///
/// For example, if the repetition index is: [50, 3], [50, 0], [10, 0] and the range
/// from the user is 40..60 then we need to:
///
///  - Read the first chunk and skip the first 40 rows, then read 10 full rows, and
///    then read 3 items for the 11th row of our range.
///  - Read the second chunk and read the remaining items in our 11th row and then read
///    the remaining 9 full rows.
///
/// Then, if we are going to decode that in batches of 5, we need to make decode tasks.
/// The first two decode tasks will just need the first chunk.  The third decode task will
/// need the first chunk (for the trailer which has the 11th row in our range) and the second
/// chunk.  The final decode task will just need the second chunk.
///
/// The above prose descriptions are what are represented by [`ChunkInstructions`] and
/// [`ChunkDrainInstructions`].
#[derive(Debug)]
pub struct MiniBlockScheduler {
    // These come from the protobuf
    buffer_offsets_and_sizes: Vec<(u64, u64)>,
    priority: u64,
    items_in_page: u64,
    repetition_index_depth: u16,
    num_buffers: u64,
    rep_decompressor: Option<Arc<dyn BlockDecompressor>>,
    def_decompressor: Option<Arc<dyn BlockDecompressor>>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    dictionary: Option<MiniBlockSchedulerDictionary>,
    // This is set after initialization
    page_meta: Option<Arc<MiniBlockCacheableState>>,
}

impl MiniBlockScheduler {
    fn try_new(
        buffer_offsets_and_sizes: &[(u64, u64)],
        priority: u64,
        items_in_page: u64,
        layout: &pb::MiniBlockLayout,
        decompressors: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        let rep_decompressor = layout
            .rep_compression
            .as_ref()
            .map(|rep_compression| {
                decompressors
                    .create_block_decompressor(rep_compression)
                    .map(Arc::from)
            })
            .transpose()?;
        let def_decompressor = layout
            .def_compression
            .as_ref()
            .map(|def_compression| {
                decompressors
                    .create_block_decompressor(def_compression)
                    .map(Arc::from)
            })
            .transpose()?;
        let def_meaning = layout
            .layers
            .iter()
            .map(|l| ProtobufUtils::repdef_layer_to_def_interp(*l))
            .collect::<Vec<_>>();
        let value_decompressor = decompressors.create_miniblock_decompressor(
            layout.value_compression.as_ref().unwrap(),
            decompressors,
        )?;

        let dictionary = if let Some(dictionary_encoding) = layout.dictionary.as_ref() {
            let num_dictionary_items = layout.num_dictionary_items;
            match dictionary_encoding.array_encoding.as_ref().unwrap() {
                pb::array_encoding::ArrayEncoding::Variable(_) => {
                    Some(MiniBlockSchedulerDictionary {
                        dictionary_decompressor: decompressors
                            .create_block_decompressor(dictionary_encoding)?
                            .into(),
                        dictionary_buf_position_and_size: buffer_offsets_and_sizes[2],
                        dictionary_data_alignment: 4,
                        num_dictionary_items,
                    })
                }
                pb::array_encoding::ArrayEncoding::Flat(_) => Some(MiniBlockSchedulerDictionary {
                    dictionary_decompressor: decompressors
                        .create_block_decompressor(dictionary_encoding)?
                        .into(),
                    dictionary_buf_position_and_size: buffer_offsets_and_sizes[2],
                    dictionary_data_alignment: 16,
                    num_dictionary_items,
                }),
                _ => {
                    unreachable!("Currently only encodings `BinaryBlock` and `Flat` used for encoding MiniBlock dictionary.")
                }
            }
        } else {
            None
        };

        Ok(Self {
            buffer_offsets_and_sizes: buffer_offsets_and_sizes.to_vec(),
            rep_decompressor,
            def_decompressor,
            value_decompressor: value_decompressor.into(),
            repetition_index_depth: layout.repetition_index_depth as u16,
            num_buffers: layout.num_buffers,
            priority,
            items_in_page,
            dictionary,
            def_meaning: def_meaning.into(),
            page_meta: None,
        })
    }

    fn lookup_chunks(&self, chunk_indices: &[usize]) -> Vec<LoadedChunk> {
        let page_meta = self.page_meta.as_ref().unwrap();
        chunk_indices
            .iter()
            .map(|&chunk_idx| {
                let chunk_meta = &page_meta.chunk_meta[chunk_idx];
                let bytes_start = chunk_meta.offset_bytes;
                let bytes_end = bytes_start + chunk_meta.chunk_size_bytes;
                LoadedChunk {
                    byte_range: bytes_start..bytes_end,
                    items_in_chunk: chunk_meta.num_values,
                    chunk_idx,
                    data: LanceBuffer::empty(),
                }
            })
            .collect()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum PreambleAction {
    Take,
    Skip,
    Absent,
}

// TODO: Add test cases for the all-preamble and all-trailer cases

// When we schedule a chunk we use the repetition index (or, if none exists, just the # of items
// in each chunk) to map a user requested range into a set of ChunkInstruction objects which tell
// us how exactly to read from the chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ChunkInstructions {
    // The index of the chunk to read
    chunk_idx: usize,
    // A "preamble" is when a chunk begins with a continuation of the previous chunk's list.  If there
    // is no repetition index there is never a preamble.
    //
    // It's possible for a chunk to be entirely premable.  For example, if there is a really large list
    // that spans several chunks.
    preamble: PreambleAction,
    // How many complete rows (not including the preamble or trailer) to skip
    //
    // If this is non-zero then premable must not be Take
    rows_to_skip: u64,
    // How many complete (non-preamble / non-trailer) rows to take
    rows_to_take: u64,
    // A "trailer" is when a chunk ends with a partial list.  If there is no repetition index there is
    // never a trailer.
    //
    // It's possible for a chunk to be entirely trailer.  This would mean the chunk starts with the beginning
    // of a list and that list is continued in the next chunk.
    //
    // If this is true then we want to include the trailer
    take_trailer: bool,
}

// First, we schedule a bunch of [`ChunkInstructions`] based on the users ranges.  Then we
// start decoding them, based on a batch size, which might not align with what we scheduled.
//
// This results in `ChunkDrainInstructions` which targets a contiguous slice of a `ChunkInstructions`
//
// So if `ChunkInstructions` is "skip preamble, skip 10, take 50, take trailer" and we are decoding in
// batches of size 10 we might have a `ChunkDrainInstructions` that targets that chunk and has its own
// skip of 17 and take of 10.  This would mean we decode the chunk, skip the preamble and 27 rows, and
// then take 10 rows.
//
// One very confusing bit is that `rows_to_take` includes the trailer.  So if we have two chunks:
//  -no preamble, skip 5, take 10, take trailer
//  -take preamble, skip 0, take 50, no trailer
//
// and we are draining 20 rows then the drain instructions for the first batch will be:
//  - no preamble, skip 0 (from chunk 0), take 11 (from chunk 0)
//  - take preamble (from chunk 1), skip 0 (from chunk 1), take 9 (from chunk 1)
#[derive(Debug, PartialEq, Eq)]
struct ChunkDrainInstructions {
    chunk_instructions: ChunkInstructions,
    rows_to_skip: u64,
    rows_to_take: u64,
    preamble_action: PreambleAction,
}

impl ChunkInstructions {
    // Given a repetition index and a set of user ranges we need to figure out how to read from the chunks
    //
    // We assume that `user_ranges` are in sorted order and non-overlapping
    //
    // The output will be a set of `ChunkInstructions` which tell us how to read from the chunks
    fn schedule_instructions(
        rep_index: &MiniBlockRepIndex,
        user_ranges: &[Range<u64>],
    ) -> Vec<Self> {
        // This is an in-exact capacity guess but pretty good.  The actual capacity can be
        // smaller if instructions are merged.  It can be larger if there are multiple instructions
        // per row which can happen with lists.
        let mut chunk_instructions = Vec::with_capacity(user_ranges.len());

        for user_range in user_ranges {
            let mut rows_needed = user_range.end - user_range.start;
            let mut need_preamble = false;

            // Need to find the first chunk with a first row >= user_range.start.  If there are
            // multiple chunks with the same first row we need to take the first one.
            let mut block_index = match rep_index
                .blocks
                .binary_search_by_key(&user_range.start, |block| block.first_row)
            {
                Ok(idx) => {
                    // Slightly tricky case, we may need to walk backwards a bit to make sure we
                    // are grabbing first eligible chunk
                    let mut idx = idx;
                    while idx > 0 && rep_index.blocks[idx - 1].first_row == user_range.start {
                        idx -= 1;
                    }
                    idx
                }
                // Easy case.  idx is greater, and idx - 1 is smaller, so idx - 1 contains the start
                Err(idx) => idx - 1,
            };

            let mut to_skip = user_range.start - rep_index.blocks[block_index].first_row;

            while rows_needed > 0 || need_preamble {
                let chunk = &rep_index.blocks[block_index];
                let rows_avail = chunk.starts_including_trailer - to_skip;
                debug_assert!(rows_avail > 0);

                let rows_to_take = rows_avail.min(rows_needed);
                rows_needed -= rows_to_take;

                let mut take_trailer = false;
                let preamble = if chunk.has_preamble {
                    if need_preamble {
                        PreambleAction::Take
                    } else {
                        PreambleAction::Skip
                    }
                } else {
                    PreambleAction::Absent
                };
                let mut rows_to_take_no_trailer = rows_to_take;

                // Are we taking the trailer?  If so, make sure we mark that we need the preamble
                if rows_to_take == rows_avail && chunk.has_trailer {
                    take_trailer = true;
                    need_preamble = true;
                    rows_to_take_no_trailer -= 1;
                } else {
                    need_preamble = false;
                };

                chunk_instructions.push(Self {
                    preamble,
                    chunk_idx: block_index,
                    rows_to_skip: to_skip,
                    rows_to_take: rows_to_take_no_trailer,
                    take_trailer,
                });

                to_skip = 0;
                block_index += 1;
            }
        }

        // If there were multiple ranges we may have multiple instructions for a single chunk.  Merge them now if they
        // are _adjacent_ (i.e. don't merge "take first row of chunk 0" and "take third row of chunk 0" into "take 2
        // rows of chunk 0 starting at 0")
        if user_ranges.len() > 1 {
            // TODO: Could probably optimize this allocation away
            let mut merged_instructions = Vec::with_capacity(chunk_instructions.len());
            let mut instructions_iter = chunk_instructions.into_iter();
            merged_instructions.push(instructions_iter.next().unwrap());
            for instruction in instructions_iter {
                let last = merged_instructions.last_mut().unwrap();
                if last.chunk_idx == instruction.chunk_idx
                    && last.rows_to_take + last.rows_to_skip == instruction.rows_to_skip
                {
                    last.rows_to_take += instruction.rows_to_take;
                    last.take_trailer |= instruction.take_trailer;
                } else {
                    merged_instructions.push(instruction);
                }
            }
            merged_instructions
        } else {
            chunk_instructions
        }
    }

    fn drain_from_instruction(
        &self,
        rows_desired: &mut u64,
        need_preamble: &mut bool,
        skip_in_chunk: &mut u64,
    ) -> (ChunkDrainInstructions, bool) {
        // If we need the premable then we shouldn't be skipping anything
        debug_assert!(!*need_preamble || *skip_in_chunk == 0);
        let mut rows_avail = self.rows_to_take - *skip_in_chunk;
        let has_preamble = self.preamble != PreambleAction::Absent;
        let preamble_action = match (*need_preamble, has_preamble) {
            (true, true) => PreambleAction::Take,
            (true, false) => panic!("Need preamble but there isn't one"),
            (false, true) => PreambleAction::Skip,
            (false, false) => PreambleAction::Absent,
        };

        // Did the scheduled chunk have a trailer?  If so, we have one extra row available
        if self.take_trailer {
            rows_avail += 1;
        }

        // How many rows are we actually taking in this take step (including the preamble
        // and trailer both as individual rows)
        let rows_taking = if *rows_desired >= rows_avail {
            // We want all the rows.  If there is a trailer we are grabbing it and will need
            // the preamble of the next chunk
            *need_preamble = self.take_trailer;
            rows_avail
        } else {
            // We aren't taking all the rows.  Even if there is a trailer we aren't taking
            // it so we will not need the preamble
            *need_preamble = false;
            *rows_desired
        };
        let rows_skipped = *skip_in_chunk;

        // Update the state for the next iteration
        let consumed_chunk = if *rows_desired >= rows_avail {
            *rows_desired -= rows_avail;
            *skip_in_chunk = 0;
            true
        } else {
            *skip_in_chunk += *rows_desired;
            *rows_desired = 0;
            false
        };

        (
            ChunkDrainInstructions {
                chunk_instructions: self.clone(),
                rows_to_skip: rows_skipped,
                rows_to_take: rows_taking,
                preamble_action,
            },
            consumed_chunk,
        )
    }
}

impl StructuralPageScheduler for MiniBlockScheduler {
    fn initialize<'a>(
        &'a mut self,
        io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        // We always need to fetch chunk metadata.  We may also need to fetch a dictionary and
        // we may also need to fetch the repetition index.  Here, we gather what buffers we
        // need.
        let (meta_buf_position, meta_buf_size) = self.buffer_offsets_and_sizes[0];
        let value_buf_position = self.buffer_offsets_and_sizes[1].0;
        let mut bufs_needed = 1;
        if self.dictionary.is_some() {
            bufs_needed += 1;
        }
        if self.repetition_index_depth > 0 {
            bufs_needed += 1;
        }
        let mut required_ranges = Vec::with_capacity(bufs_needed);
        required_ranges.push(meta_buf_position..meta_buf_position + meta_buf_size);
        if let Some(ref dictionary) = self.dictionary {
            required_ranges.push(
                dictionary.dictionary_buf_position_and_size.0
                    ..dictionary.dictionary_buf_position_and_size.0
                        + dictionary.dictionary_buf_position_and_size.1,
            );
        }
        if self.repetition_index_depth > 0 {
            let (rep_index_pos, rep_index_size) = self.buffer_offsets_and_sizes.last().unwrap();
            required_ranges.push(*rep_index_pos..*rep_index_pos + *rep_index_size);
        }
        let io_req = io.submit_request(required_ranges, 0);

        async move {
            let mut buffers = io_req.await?.into_iter().fuse();
            let meta_bytes = buffers.next().unwrap();
            let dictionary_bytes = self.dictionary.as_ref().and_then(|_| buffers.next());
            let rep_index_bytes = buffers.next();

            // Parse the metadata and build the chunk meta
            assert!(meta_bytes.len() % 2 == 0);
            let mut bytes = LanceBuffer::from_bytes(meta_bytes, 2);
            let words = bytes.borrow_to_typed_slice::<u16>();
            let words = words.as_ref();

            let mut chunk_meta = Vec::with_capacity(words.len());

            let mut rows_counter = 0;
            let mut offset_bytes = value_buf_position;
            for (word_idx, word) in words.iter().enumerate() {
                let log_num_values = word & 0x0F;
                let divided_bytes = word >> 4;
                let num_bytes = (divided_bytes as usize + 1) * MINIBLOCK_ALIGNMENT;
                debug_assert!(num_bytes > 0);
                let num_values = if word_idx < words.len() - 1 {
                    debug_assert!(log_num_values > 0);
                    1 << log_num_values
                } else {
                    debug_assert!(
                        log_num_values == 0
                            || (1 << log_num_values) == (self.items_in_page - rows_counter)
                    );
                    self.items_in_page - rows_counter
                };
                rows_counter += num_values;

                chunk_meta.push(ChunkMeta {
                    num_values,
                    chunk_size_bytes: num_bytes as u64,
                    offset_bytes,
                });
                offset_bytes += num_bytes as u64;
            }

            // Build the repetition index
            let rep_index = if let Some(rep_index_data) = rep_index_bytes {
                // If we have a repetition index then we use that
                // TODO: Compress the repetition index :)
                assert!(rep_index_data.len() % 8 == 0);
                let mut repetition_index_vals = LanceBuffer::from_bytes(rep_index_data, 8);
                let repetition_index_vals = repetition_index_vals.borrow_to_typed_slice::<u64>();
                // Unflatten
                repetition_index_vals
                    .as_ref()
                    .chunks_exact(self.repetition_index_depth as usize + 1)
                    .map(|c| c.to_vec())
                    .collect::<Vec<_>>()
            } else {
                // Default rep index is just the number of items in each chunk
                // with 0 partials/leftovers
                chunk_meta
                    .iter()
                    .map(|c| vec![c.num_values, 0])
                    .collect::<Vec<_>>()
            };

            let mut page_meta = MiniBlockCacheableState {
                chunk_meta,
                rep_index: MiniBlockRepIndex::decode(&rep_index),
                dictionary: None,
            };

            // decode dictionary
            if let Some(ref mut dictionary) = self.dictionary {
                let dictionary_data = dictionary_bytes.unwrap();
                page_meta.dictionary =
                    Some(Arc::new(dictionary.dictionary_decompressor.decompress(
                        LanceBuffer::from_bytes(
                            dictionary_data,
                            dictionary.dictionary_data_alignment,
                        ),
                        dictionary.num_dictionary_items,
                    )?));
            };
            let page_meta = Arc::new(page_meta);
            self.page_meta = Some(page_meta.clone());
            Ok(page_meta as Arc<dyn CachedPageData>)
        }
        .boxed()
    }

    fn load(&mut self, data: &Arc<dyn CachedPageData>) {
        self.page_meta = Some(
            data.clone()
                .as_arc_any()
                .downcast::<MiniBlockCacheableState>()
                .unwrap(),
        );
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();

        let page_meta = self.page_meta.as_ref().unwrap();

        let chunk_instructions =
            ChunkInstructions::schedule_instructions(&page_meta.rep_index, ranges);

        debug_assert_eq!(
            num_rows,
            chunk_instructions
                .iter()
                .map(|ci| {
                    let taken = ci.rows_to_take;
                    if ci.take_trailer {
                        taken + 1
                    } else {
                        taken
                    }
                })
                .sum::<u64>()
        );

        let chunks_needed = chunk_instructions
            .iter()
            .map(|ci| ci.chunk_idx)
            .unique()
            .collect::<Vec<_>>();
        let mut loaded_chunks = self.lookup_chunks(&chunks_needed);
        let chunk_ranges = loaded_chunks
            .iter()
            .map(|c| c.byte_range.clone())
            .collect::<Vec<_>>();
        let loaded_chunk_data = io.submit_request(chunk_ranges, self.priority);

        let rep_decompressor = self.rep_decompressor.clone();
        let def_decompressor = self.def_decompressor.clone();
        let value_decompressor = self.value_decompressor.clone();
        let num_buffers = self.num_buffers;
        let dictionary = page_meta
            .dictionary
            .as_ref()
            .map(|dictionary| dictionary.clone());
        let def_meaning = self.def_meaning.clone();

        let res = async move {
            let loaded_chunk_data = loaded_chunk_data.await?;
            for (loaded_chunk, chunk_data) in loaded_chunks.iter_mut().zip(loaded_chunk_data) {
                loaded_chunk.data = LanceBuffer::from_bytes(chunk_data, 1);
            }

            Ok(Box::new(MiniBlockDecoder {
                rep_decompressor,
                def_decompressor,
                value_decompressor,
                def_meaning,
                loaded_chunks: VecDeque::from_iter(loaded_chunks),
                instructions: VecDeque::from(chunk_instructions),
                offset_in_current_chunk: 0,
                dictionary,
                num_rows,
                num_buffers,
            }) as Box<dyn StructuralPageDecoder>)
        }
        .boxed();
        Ok(res)
    }
}

#[derive(Debug, Clone, Copy)]
struct FullZipRepIndexDetails {
    buf_position: u64,
    bytes_per_value: u64, // Will be 1, 2, 4, or 8
}

#[derive(Debug)]
enum PerValueDecompressor {
    Fixed(Arc<dyn FixedPerValueDecompressor>),
    Variable(Arc<dyn VariablePerValueDecompressor>),
}

#[derive(Debug)]
struct FullZipDecodeDetails {
    value_decompressor: PerValueDecompressor,
    def_meaning: Arc<[DefinitionInterpretation]>,
    ctrl_word_parser: ControlWordParser,
    max_rep: u16,
    max_visible_def: u16,
}

/// A scheduler for full-zip encoded data
///
/// When the data type has a fixed-width then we simply need to map from
/// row ranges to byte ranges using the fixed-width of the data type.
///
/// When the data type is variable-width or has any repetition then a
/// repetition index is required.
#[derive(Debug)]
pub struct FullZipScheduler {
    data_buf_position: u64,
    rep_index: Option<FullZipRepIndexDetails>,
    priority: u64,
    rows_in_page: u64,
    bits_per_offset: u8,
    details: Arc<FullZipDecodeDetails>,
    /// Cached state containing the decoded repetition index
    cached_state: Option<Arc<FullZipCacheableState>>,
    /// Whether to enable caching of repetition indices
    enable_cache: bool,
}

impl FullZipScheduler {
    fn try_new(
        buffer_offsets_and_sizes: &[(u64, u64)],
        priority: u64,
        rows_in_page: u64,
        layout: &pb::FullZipLayout,
        decompressors: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        // We don't need the data_buf_size because either the data type is
        // fixed-width (and we can tell size from rows_in_page) or it is not
        // and we have a repetition index.
        let (data_buf_position, _) = buffer_offsets_and_sizes[0];
        let rep_index = buffer_offsets_and_sizes.get(1).map(|(pos, len)| {
            let num_reps = rows_in_page + 1;
            let bytes_per_rep = len / num_reps;
            debug_assert_eq!(len % num_reps, 0);
            debug_assert!(
                bytes_per_rep == 1
                    || bytes_per_rep == 2
                    || bytes_per_rep == 4
                    || bytes_per_rep == 8
            );
            FullZipRepIndexDetails {
                buf_position: *pos,
                bytes_per_value: bytes_per_rep,
            }
        });

        let value_decompressor = match layout.details {
            Some(pb::full_zip_layout::Details::BitsPerValue(_)) => {
                let decompressor = decompressors.create_fixed_per_value_decompressor(
                    layout.value_compression.as_ref().unwrap(),
                )?;
                PerValueDecompressor::Fixed(decompressor.into())
            }
            Some(pb::full_zip_layout::Details::BitsPerOffset(_)) => {
                let decompressor = decompressors.create_variable_per_value_decompressor(
                    layout.value_compression.as_ref().unwrap(),
                )?;
                PerValueDecompressor::Variable(decompressor.into())
            }
            None => {
                panic!("Full-zip layout must have a `details` field");
            }
        };
        let ctrl_word_parser = ControlWordParser::new(
            layout.bits_rep.try_into().unwrap(),
            layout.bits_def.try_into().unwrap(),
        );
        let def_meaning = layout
            .layers
            .iter()
            .map(|l| ProtobufUtils::repdef_layer_to_def_interp(*l))
            .collect::<Vec<_>>();

        let max_rep = def_meaning.iter().filter(|d| d.is_list()).count() as u16;
        let max_visible_def = def_meaning
            .iter()
            .filter(|d| !d.is_list())
            .map(|d| d.num_def_levels())
            .sum();

        let bits_per_offset = match layout.details {
            Some(pb::full_zip_layout::Details::BitsPerValue(_)) => 32,
            Some(pb::full_zip_layout::Details::BitsPerOffset(bits_per_offset)) => {
                bits_per_offset as u8
            }
            None => panic!("Full-zip layout must have a `details` field"),
        };

        let details = Arc::new(FullZipDecodeDetails {
            value_decompressor,
            def_meaning: def_meaning.into(),
            ctrl_word_parser,
            max_rep,
            max_visible_def,
        });
        Ok(Self {
            data_buf_position,
            rep_index,
            details,
            priority,
            rows_in_page,
            bits_per_offset,
            cached_state: None,
            enable_cache: false, // Default to false, will be set later
        })
    }

    /// Creates a decoder from the loaded data
    fn create_decoder(
        details: Arc<FullZipDecodeDetails>,
        data: VecDeque<LanceBuffer>,
        num_rows: u64,
        bits_per_offset: u8,
    ) -> Result<Box<dyn StructuralPageDecoder>> {
        match &details.value_decompressor {
            PerValueDecompressor::Fixed(decompressor) => {
                let bits_per_value = decompressor.bits_per_value();
                if bits_per_value == 0 {
                    return Err(lance_core::Error::Internal {
                        message: "Invalid encoding: bits_per_value must be greater than 0".into(),
                        location: location!(),
                    });
                }
                if bits_per_value % 8 != 0 {
                    return Err(lance_core::Error::NotSupported {
                        source: "Bit-packed full-zip encoding (non-byte-aligned values) is not yet implemented".into(),
                        location: location!(),
                    });
                }
                let bytes_per_value = bits_per_value / 8;
                let total_bytes_per_value =
                    bytes_per_value as usize + details.ctrl_word_parser.bytes_per_word();
                Ok(Box::new(FixedFullZipDecoder {
                    details,
                    data,
                    num_rows,
                    offset_in_current: 0,
                    bytes_per_value: bytes_per_value as usize,
                    total_bytes_per_value,
                }) as Box<dyn StructuralPageDecoder>)
            }
            PerValueDecompressor::Variable(_decompressor) => {
                Ok(Box::new(VariableFullZipDecoder::new(
                    details,
                    data,
                    num_rows,
                    bits_per_offset,
                    bits_per_offset,
                )))
            }
        }
    }

    /// Extracts byte ranges from a repetition index buffer
    /// The buffer contains pairs of (start, end) values for each range
    fn extract_byte_ranges_from_pairs(
        buffer: LanceBuffer,
        bytes_per_value: u64,
        data_buf_position: u64,
    ) -> Vec<Range<u64>> {
        ByteUnpacker::new(buffer, bytes_per_value as usize)
            .chunks(2)
            .into_iter()
            .map(|mut c| {
                let start = c.next().unwrap() + data_buf_position;
                let end = c.next().unwrap() + data_buf_position;
                start..end
            })
            .collect::<Vec<_>>()
    }

    /// Extracts byte ranges from a cached repetition index buffer
    /// The buffer contains all values and we need to extract specific ranges
    fn extract_byte_ranges_from_cached(
        buffer: &LanceBuffer,
        ranges: &[Range<u64>],
        bytes_per_value: u64,
        data_buf_position: u64,
    ) -> Vec<Range<u64>> {
        ranges
            .iter()
            .map(|r| {
                let start_offset = (r.start * bytes_per_value) as usize;
                let end_offset = (r.end * bytes_per_value) as usize;

                let start_slice = &buffer[start_offset..start_offset + bytes_per_value as usize];
                let start_val =
                    ByteUnpacker::new(start_slice.iter().copied(), bytes_per_value as usize)
                        .next()
                        .unwrap();

                let end_slice = &buffer[end_offset..end_offset + bytes_per_value as usize];
                let end_val =
                    ByteUnpacker::new(end_slice.iter().copied(), bytes_per_value as usize)
                        .next()
                        .unwrap();

                (data_buf_position + start_val)..(data_buf_position + end_val)
            })
            .collect()
    }

    /// Computes the ranges in the repetition index that need to be loaded
    fn compute_rep_index_ranges(
        ranges: &[Range<u64>],
        rep_index: &FullZipRepIndexDetails,
    ) -> Vec<Range<u64>> {
        ranges
            .iter()
            .flat_map(|r| {
                let first_val_start =
                    rep_index.buf_position + (r.start * rep_index.bytes_per_value);
                let first_val_end = first_val_start + rep_index.bytes_per_value;
                let last_val_start = rep_index.buf_position + (r.end * rep_index.bytes_per_value);
                let last_val_end = last_val_start + rep_index.bytes_per_value;
                [first_val_start..first_val_end, last_val_start..last_val_end]
            })
            .collect()
    }

    /// Resolves byte ranges from repetition index (either from cache or disk)
    async fn resolve_byte_ranges(
        data_buf_position: u64,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
        rep_index: &FullZipRepIndexDetails,
        cached_state: Option<&Arc<FullZipCacheableState>>,
        priority: u64,
    ) -> Result<Vec<Range<u64>>> {
        if let Some(cached_state) = cached_state {
            // Use cached repetition index
            Ok(Self::extract_byte_ranges_from_cached(
                &cached_state.rep_index_buffer,
                ranges,
                rep_index.bytes_per_value,
                data_buf_position,
            ))
        } else {
            // Load from disk
            let rep_ranges = Self::compute_rep_index_ranges(ranges, rep_index);
            let rep_data = io.submit_request(rep_ranges, priority).await?;
            let rep_buffer = LanceBuffer::concat(
                &rep_data
                    .into_iter()
                    .map(|d| LanceBuffer::from_bytes(d, 1))
                    .collect::<Vec<_>>(),
            );
            Ok(Self::extract_byte_ranges_from_pairs(
                rep_buffer,
                rep_index.bytes_per_value,
                data_buf_position,
            ))
        }
    }

    /// Schedules ranges in the presence of a repetition index
    fn schedule_ranges_rep(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
        rep_index: FullZipRepIndexDetails,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>> {
        // Copy necessary fields to avoid lifetime issues
        let data_buf_position = self.data_buf_position;
        let cached_state = self.cached_state.clone();
        let priority = self.priority;
        let details = self.details.clone();
        let bits_per_offset = self.bits_per_offset;
        let ranges = ranges.to_vec();
        let io_clone = io.clone();

        Ok(async move {
            // Step 1: Resolve byte ranges from repetition index
            let byte_ranges = Self::resolve_byte_ranges(
                data_buf_position,
                &ranges,
                &io_clone,
                &rep_index,
                cached_state.as_ref(),
                priority,
            )
            .await?;

            // Step 2: Load data
            let data = io_clone.submit_request(byte_ranges, priority).await?;
            let data = data
                .into_iter()
                .map(|d| LanceBuffer::from_bytes(d, 1))
                .collect::<VecDeque<_>>();

            // Step 3: Calculate total rows
            let num_rows: u64 = ranges.iter().map(|r| r.end - r.start).sum();

            // Step 4: Create decoder
            Self::create_decoder(details, data, num_rows, bits_per_offset)
        }
        .boxed())
    }

    // In the simple case there is no repetition and we just have large fixed-width
    // rows of data.  We can just map row ranges to byte ranges directly using the
    // fixed-width of the data type.
    fn schedule_ranges_simple(
        &self,
        ranges: &[Range<u64>],
        io: &dyn EncodingsIo,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>> {
        // Convert row ranges to item ranges (i.e. multiply by items per row)
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();

        let PerValueDecompressor::Fixed(decompressor) = &self.details.value_decompressor else {
            unreachable!()
        };

        // Convert item ranges to byte ranges (i.e. multiply by bytes per item)
        let bits_per_value = decompressor.bits_per_value();
        assert_eq!(bits_per_value % 8, 0);
        let bytes_per_value = bits_per_value / 8;
        let bytes_per_cw = self.details.ctrl_word_parser.bytes_per_word();
        let total_bytes_per_value = bytes_per_value + bytes_per_cw as u64;
        let byte_ranges = ranges.iter().map(|r| {
            debug_assert!(r.end <= self.rows_in_page);
            let start = self.data_buf_position + r.start * total_bytes_per_value;
            let end = self.data_buf_position + r.end * total_bytes_per_value;
            start..end
        });

        // Request byte ranges
        let data = io.submit_request(byte_ranges.collect(), self.priority);

        let details = self.details.clone();

        Ok(async move {
            let data = data.await?;
            let data = data
                .into_iter()
                .map(|d| LanceBuffer::from_bytes(d, 1))
                .collect();
            Ok(Box::new(FixedFullZipDecoder {
                details,
                data,
                num_rows,
                offset_in_current: 0,
                bytes_per_value: bytes_per_value as usize,
                total_bytes_per_value: total_bytes_per_value as usize,
            }) as Box<dyn StructuralPageDecoder>)
        }
        .boxed())
    }
}

/// Cacheable state for FullZip encoding, storing the decoded repetition index
#[derive(Debug)]
struct FullZipCacheableState {
    /// The raw repetition index buffer for future decoding
    rep_index_buffer: LanceBuffer,
}

impl DeepSizeOf for FullZipCacheableState {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.rep_index_buffer.len()
    }
}

impl CachedPageData for FullZipCacheableState {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

impl StructuralPageScheduler for FullZipScheduler {
    /// Initializes the scheduler. If there's a repetition index, loads and caches it.
    /// Otherwise returns NoCachedPageData.
    fn initialize<'a>(
        &'a mut self,
        io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        // Check if caching is enabled and we have a repetition index
        if self.enable_cache && self.rep_index.is_some() {
            let rep_index = self.rep_index.as_ref().unwrap();
            // Calculate the total size of the repetition index
            let total_size = (self.rows_in_page + 1) * rep_index.bytes_per_value;
            let rep_index_range = rep_index.buf_position..(rep_index.buf_position + total_size);

            // Load the repetition index buffer
            let io_clone = io.clone();
            let future = async move {
                let rep_index_data = io_clone.submit_request(vec![rep_index_range], 0).await?;
                let rep_index_buffer = LanceBuffer::from_bytes(rep_index_data[0].clone(), 1);

                // Create and return the cacheable state
                Ok(Arc::new(FullZipCacheableState { rep_index_buffer }) as Arc<dyn CachedPageData>)
            };

            future.boxed()
        } else {
            // Caching disabled or no repetition index, skip caching
            std::future::ready(Ok(Arc::new(NoCachedPageData) as Arc<dyn CachedPageData>)).boxed()
        }
    }

    /// Loads previously cached repetition index data from the cache system.
    /// This method is called when a scheduler instance needs to use cached data
    /// that was initialized by another instance or in a previous operation.
    fn load(&mut self, cache: &Arc<dyn CachedPageData>) {
        // Try to downcast to our specific cache type
        if let Ok(cached_state) = cache
            .clone()
            .as_arc_any()
            .downcast::<FullZipCacheableState>()
        {
            // Store the cached state for use in schedule_ranges
            self.cached_state = Some(cached_state);
        }
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>> {
        if let Some(rep_index) = self.rep_index {
            self.schedule_ranges_rep(ranges, io, rep_index)
        } else {
            self.schedule_ranges_simple(ranges, io.as_ref())
        }
    }
}

/// A decoder for full-zip encoded data when the data has a fixed-width
///
/// Here we need to unzip the control words from the values themselves and
/// then decompress the requested values.
///
/// We use a PerValueDecompressor because we will only be decompressing the
/// requested data.  This decoder / scheduler does not do any read amplification.
#[derive(Debug)]
struct FixedFullZipDecoder {
    details: Arc<FullZipDecodeDetails>,
    data: VecDeque<LanceBuffer>,
    offset_in_current: usize,
    bytes_per_value: usize,
    total_bytes_per_value: usize,
    num_rows: u64,
}

impl FixedFullZipDecoder {
    fn slice_next_task(&mut self, num_rows: u64) -> FullZipDecodeTaskItem {
        debug_assert!(num_rows > 0);
        let cur_buf = self.data.front_mut().unwrap();
        let start = self.offset_in_current;
        if self.details.ctrl_word_parser.has_rep() {
            // This is a slightly slower path.  In order to figure out where to split we need to
            // examine the rep index so we can convert num_lists to num_rows
            let mut rows_started = 0;
            // We always need at least one value.  Now loop through until we have passed num_rows
            // values
            let mut num_items = 0;
            while self.offset_in_current < cur_buf.len() {
                let control = self.details.ctrl_word_parser.parse_desc(
                    &cur_buf[self.offset_in_current..],
                    self.details.max_rep,
                    self.details.max_visible_def,
                );
                if control.is_new_row {
                    if rows_started == num_rows {
                        break;
                    }
                    rows_started += 1;
                }
                num_items += 1;
                if control.is_visible {
                    self.offset_in_current += self.total_bytes_per_value;
                } else {
                    self.offset_in_current += self.details.ctrl_word_parser.bytes_per_word();
                }
            }

            let task_slice = cur_buf.slice_with_length(start, self.offset_in_current - start);
            if self.offset_in_current == cur_buf.len() {
                self.data.pop_front();
                self.offset_in_current = 0;
            }

            FullZipDecodeTaskItem {
                data: PerValueDataBlock::Fixed(FixedWidthDataBlock {
                    data: task_slice,
                    bits_per_value: self.bytes_per_value as u64 * 8,
                    num_values: num_items,
                    block_info: BlockInfo::new(),
                }),
                rows_in_buf: rows_started,
            }
        } else {
            // If there's no repetition we can calculate the slicing point by just multiplying
            // the number of rows by the total bytes per value
            let cur_buf = self.data.front_mut().unwrap();
            let bytes_avail = cur_buf.len() - self.offset_in_current;
            let offset_in_cur = self.offset_in_current;

            let bytes_needed = num_rows as usize * self.total_bytes_per_value;
            let mut rows_taken = num_rows;
            let task_slice = if bytes_needed >= bytes_avail {
                self.offset_in_current = 0;
                rows_taken = bytes_avail as u64 / self.total_bytes_per_value as u64;
                self.data
                    .pop_front()
                    .unwrap()
                    .slice_with_length(offset_in_cur, bytes_avail)
            } else {
                self.offset_in_current += bytes_needed;
                cur_buf.slice_with_length(offset_in_cur, bytes_needed)
            };
            FullZipDecodeTaskItem {
                data: PerValueDataBlock::Fixed(FixedWidthDataBlock {
                    data: task_slice,
                    bits_per_value: self.bytes_per_value as u64 * 8,
                    num_values: rows_taken,
                    block_info: BlockInfo::new(),
                }),
                rows_in_buf: rows_taken,
            }
        }
    }
}

impl StructuralPageDecoder for FixedFullZipDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let mut task_data = Vec::with_capacity(self.data.len());
        let mut remaining = num_rows;
        while remaining > 0 {
            let task_item = self.slice_next_task(remaining);
            remaining -= task_item.rows_in_buf;
            task_data.push(task_item);
        }
        Ok(Box::new(FixedFullZipDecodeTask {
            details: self.details.clone(),
            data: task_data,
            bytes_per_value: self.bytes_per_value,
            num_rows: num_rows as usize,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// A decoder for full-zip encoded data when the data has a variable-width
///
/// Here we need to unzip the control words AND lengths from the values and
/// then decompress the requested values.
#[derive(Debug)]
struct VariableFullZipDecoder {
    details: Arc<FullZipDecodeDetails>,
    decompressor: Arc<dyn VariablePerValueDecompressor>,
    data: LanceBuffer,
    offsets: LanceBuffer,
    rep: ScalarBuffer<u16>,
    def: ScalarBuffer<u16>,
    repdef_starts: Vec<usize>,
    data_starts: Vec<usize>,
    offset_starts: Vec<usize>,
    visible_item_counts: Vec<u64>,
    bits_per_offset: u8,
    current_idx: usize,
    num_rows: u64,
}

impl VariableFullZipDecoder {
    fn new(
        details: Arc<FullZipDecodeDetails>,
        data: VecDeque<LanceBuffer>,
        num_rows: u64,
        in_bits_per_length: u8,
        out_bits_per_offset: u8,
    ) -> Self {
        let decompressor = match details.value_decompressor {
            PerValueDecompressor::Variable(ref d) => d.clone(),
            _ => unreachable!(),
        };

        assert_eq!(in_bits_per_length % 8, 0);
        assert!(out_bits_per_offset == 32 || out_bits_per_offset == 64);

        let mut decoder = Self {
            details,
            decompressor,
            data: LanceBuffer::empty(),
            offsets: LanceBuffer::empty(),
            rep: LanceBuffer::empty().borrow_to_typed_slice(),
            def: LanceBuffer::empty().borrow_to_typed_slice(),
            bits_per_offset: out_bits_per_offset,
            repdef_starts: Vec::with_capacity(num_rows as usize + 1),
            data_starts: Vec::with_capacity(num_rows as usize + 1),
            offset_starts: Vec::with_capacity(num_rows as usize + 1),
            visible_item_counts: Vec::with_capacity(num_rows as usize + 1),
            current_idx: 0,
            num_rows,
        };

        // There's no great time to do this and this is the least worst time.  If we don't unzip then
        // we can't slice the data during the decode phase.  This is because we need the offsets to be
        // unpacked to know where the values start and end.
        //
        // We don't want to unzip on the decode thread because that is a single-threaded path
        // We don't want to unzip on the scheduling thread because that is a single-threaded path
        //
        // Fortunately, we know variable length data will always be read indirectly and so we can do it
        // here, which should be on the indirect thread.  The primary disadvantage to doing it here is that
        // we load all the data into memory and then throw it away only to load it all into memory again during
        // the decode.
        //
        // There are some alternatives to investigate:
        //   - Instead of just reading the beginning and end of the rep index we could read the entire
        //     range in between.  This will give us the break points that we need for slicing and won't increase
        //     the number of IOPs but it will mean we are doing more total I/O and we need to load the rep index
        //     even when doing a full scan.
        //   - We could force each decode task to do a full unzip of all the data.  Each decode task now
        //     has to do more work but the work is all fused.
        //   - We could just try doing this work on the decode thread and see if it is a problem.
        decoder.unzip(data, in_bits_per_length, out_bits_per_offset, num_rows);

        decoder
    }

    unsafe fn parse_length(data: &[u8], bits_per_offset: u8) -> u64 {
        match bits_per_offset {
            8 => *data.get_unchecked(0) as u64,
            16 => u16::from_le_bytes([*data.get_unchecked(0), *data.get_unchecked(1)]) as u64,
            32 => u32::from_le_bytes([
                *data.get_unchecked(0),
                *data.get_unchecked(1),
                *data.get_unchecked(2),
                *data.get_unchecked(3),
            ]) as u64,
            64 => u64::from_le_bytes([
                *data.get_unchecked(0),
                *data.get_unchecked(1),
                *data.get_unchecked(2),
                *data.get_unchecked(3),
                *data.get_unchecked(4),
                *data.get_unchecked(5),
                *data.get_unchecked(6),
                *data.get_unchecked(7),
            ]),
            _ => unreachable!(),
        }
    }

    fn unzip(
        &mut self,
        data: VecDeque<LanceBuffer>,
        in_bits_per_length: u8,
        out_bits_per_offset: u8,
        num_rows: u64,
    ) {
        // This undercounts if there are lists but, at this point, we don't really know how many items we have
        let mut rep = Vec::with_capacity(num_rows as usize);
        let mut def = Vec::with_capacity(num_rows as usize);
        let bytes_cw = self.details.ctrl_word_parser.bytes_per_word() * num_rows as usize;

        // This undercounts if there are lists
        // It can also overcount if there are invisible items
        let bytes_per_offset = out_bits_per_offset as usize / 8;
        let bytes_offsets = bytes_per_offset * (num_rows as usize + 1);
        let mut offsets_data = Vec::with_capacity(bytes_offsets);

        let bytes_per_length = in_bits_per_length as usize / 8;
        let bytes_lengths = bytes_per_length * num_rows as usize;

        let bytes_data = data.iter().map(|d| d.len()).sum::<usize>();
        // This overcounts since bytes_lengths and bytes_cw are undercounts
        // It can also undercount if there are invisible items (hence the saturating_sub)
        let mut unzipped_data =
            Vec::with_capacity((bytes_data - bytes_cw).saturating_sub(bytes_lengths));

        let mut current_offset = 0_u64;
        let mut visible_item_count = 0_u64;
        for databuf in data.into_iter() {
            let mut databuf = databuf.as_ref();
            while !databuf.is_empty() {
                let data_start = unzipped_data.len();
                let offset_start = offsets_data.len();
                // We might have only-rep or only-def, neither, or both.  They move at the same
                // speed though so we only need one index into it
                let repdef_start = rep.len().max(def.len());
                // TODO: Kind of inefficient we parse the control word twice here
                let ctrl_desc = self.details.ctrl_word_parser.parse_desc(
                    databuf,
                    self.details.max_rep,
                    self.details.max_visible_def,
                );
                self.details
                    .ctrl_word_parser
                    .parse(databuf, &mut rep, &mut def);
                databuf = &databuf[self.details.ctrl_word_parser.bytes_per_word()..];

                if ctrl_desc.is_new_row {
                    self.repdef_starts.push(repdef_start);
                    self.data_starts.push(data_start);
                    self.offset_starts.push(offset_start);
                    self.visible_item_counts.push(visible_item_count);
                }
                if ctrl_desc.is_visible {
                    visible_item_count += 1;
                    if ctrl_desc.is_valid_item {
                        // Safety: Data should have at least bytes_per_length bytes remaining
                        debug_assert!(databuf.len() >= bytes_per_length);
                        let length = unsafe { Self::parse_length(databuf, in_bits_per_length) };
                        match out_bits_per_offset {
                            32 => offsets_data
                                .extend_from_slice(&(current_offset as u32).to_le_bytes()),
                            64 => offsets_data.extend_from_slice(&current_offset.to_le_bytes()),
                            _ => unreachable!(),
                        };
                        databuf = &databuf[bytes_per_offset..];
                        unzipped_data.extend_from_slice(&databuf[..length as usize]);
                        databuf = &databuf[length as usize..];
                        current_offset += length;
                    } else {
                        // Null items still get an offset
                        match out_bits_per_offset {
                            32 => offsets_data
                                .extend_from_slice(&(current_offset as u32).to_le_bytes()),
                            64 => offsets_data.extend_from_slice(&current_offset.to_le_bytes()),
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
        self.repdef_starts.push(rep.len().max(def.len()));
        self.data_starts.push(unzipped_data.len());
        self.offset_starts.push(offsets_data.len());
        self.visible_item_counts.push(visible_item_count);
        match out_bits_per_offset {
            32 => offsets_data.extend_from_slice(&(current_offset as u32).to_le_bytes()),
            64 => offsets_data.extend_from_slice(&current_offset.to_le_bytes()),
            _ => unreachable!(),
        };
        self.rep = ScalarBuffer::from(rep);
        self.def = ScalarBuffer::from(def);
        self.data = LanceBuffer::Owned(unzipped_data);
        self.offsets = LanceBuffer::Owned(offsets_data);
    }
}

impl StructuralPageDecoder for VariableFullZipDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let start = self.current_idx;
        let end = start + num_rows as usize;

        // This might seem a little peculiar.  We are returning the entire data for every single
        // batch.  This is because the offsets are relative to the start of the data.  In other words
        // imagine we have a data buffer that is 100 bytes long and the offsets are [0, 10, 20, 30, 40]
        // and we return in batches of two.  The second set of offsets will be [20, 30, 40].
        //
        // So either we pay for a copy to normalize the offsets or we just return the entire data buffer
        // which is slightly cheaper.
        let data = self.data.borrow_and_clone();

        let offset_start = self.offset_starts[start];
        let offset_end = self.offset_starts[end] + (self.bits_per_offset as usize / 8);
        let offsets = self
            .offsets
            .slice_with_length(offset_start, offset_end - offset_start);

        let repdef_start = self.repdef_starts[start];
        let repdef_end = self.repdef_starts[end];
        let rep = if self.rep.is_empty() {
            self.rep.clone()
        } else {
            self.rep.slice(repdef_start, repdef_end - repdef_start)
        };
        let def = if self.def.is_empty() {
            self.def.clone()
        } else {
            self.def.slice(repdef_start, repdef_end - repdef_start)
        };

        let visible_item_counts_start = self.visible_item_counts[start];
        let visible_item_counts_end = self.visible_item_counts[end];
        let num_visible_items = visible_item_counts_end - visible_item_counts_start;

        self.current_idx += num_rows as usize;

        Ok(Box::new(VariableFullZipDecodeTask {
            details: self.details.clone(),
            decompressor: self.decompressor.clone(),
            data,
            offsets,
            bits_per_offset: self.bits_per_offset,
            num_visible_items,
            rep,
            def,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

#[derive(Debug)]
struct VariableFullZipDecodeTask {
    details: Arc<FullZipDecodeDetails>,
    decompressor: Arc<dyn VariablePerValueDecompressor>,
    data: LanceBuffer,
    offsets: LanceBuffer,
    bits_per_offset: u8,
    num_visible_items: u64,
    rep: ScalarBuffer<u16>,
    def: ScalarBuffer<u16>,
}

impl DecodePageTask for VariableFullZipDecodeTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let block = VariableWidthBlock {
            data: self.data,
            offsets: self.offsets,
            bits_per_offset: self.bits_per_offset,
            num_values: self.num_visible_items,
            block_info: BlockInfo::new(),
        };
        let decomopressed = self.decompressor.decompress(block)?;
        let rep = self.rep.to_vec();
        let def = self.def.to_vec();
        let unraveler =
            RepDefUnraveler::new(Some(rep), Some(def), self.details.def_meaning.clone());
        Ok(DecodedPage {
            data: decomopressed,
            repdef: unraveler,
        })
    }
}

#[derive(Debug)]
struct FullZipDecodeTaskItem {
    data: PerValueDataBlock,
    rows_in_buf: u64,
}

/// A task to unzip and decompress full-zip encoded data when that data
/// has a fixed-width.
#[derive(Debug)]
struct FixedFullZipDecodeTask {
    details: Arc<FullZipDecodeDetails>,
    data: Vec<FullZipDecodeTaskItem>,
    num_rows: usize,
    bytes_per_value: usize,
}

impl DecodePageTask for FixedFullZipDecodeTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        // Multiply by 2 to make a stab at the size of the output buffer (which will be decompressed and thus bigger)
        let estimated_size_bytes = self
            .data
            .iter()
            .map(|task_item| task_item.data.data_size() as usize)
            .sum::<usize>()
            * 2;
        let mut data_builder =
            DataBlockBuilder::with_capacity_estimate(estimated_size_bytes as u64);

        if self.details.ctrl_word_parser.bytes_per_word() == 0 {
            // Fast path, no need to unzip because there is no rep/def
            //
            // We decompress each buffer and add it to our output buffer
            for task_item in self.data.into_iter() {
                let PerValueDataBlock::Fixed(fixed_data) = task_item.data else {
                    unreachable!()
                };
                let PerValueDecompressor::Fixed(decompressor) = &self.details.value_decompressor
                else {
                    unreachable!()
                };
                debug_assert_eq!(fixed_data.num_values, task_item.rows_in_buf);
                let decompressed = decompressor.decompress(fixed_data, task_item.rows_in_buf)?;
                data_builder.append(&decompressed, 0..task_item.rows_in_buf);
            }

            let unraveler = RepDefUnraveler::new(None, None, self.details.def_meaning.clone());

            Ok(DecodedPage {
                data: data_builder.finish(),
                repdef: unraveler,
            })
        } else {
            // Slow path, unzipping needed
            let mut rep = Vec::with_capacity(self.num_rows);
            let mut def = Vec::with_capacity(self.num_rows);

            for task_item in self.data.into_iter() {
                let PerValueDataBlock::Fixed(fixed_data) = task_item.data else {
                    unreachable!()
                };
                let mut buf_slice = fixed_data.data.as_ref();
                let num_values = fixed_data.num_values as usize;
                // We will be unzipping repdef in to `rep` and `def` and the
                // values into `values` (which contains the compressed values)
                let mut values = Vec::with_capacity(
                    fixed_data.data.len()
                        - (self.details.ctrl_word_parser.bytes_per_word() * num_values),
                );
                let mut visible_items = 0;
                for _ in 0..num_values {
                    // Extract rep/def
                    self.details
                        .ctrl_word_parser
                        .parse(buf_slice, &mut rep, &mut def);
                    buf_slice = &buf_slice[self.details.ctrl_word_parser.bytes_per_word()..];

                    let is_visible = def
                        .last()
                        .map(|d| *d <= self.details.max_visible_def)
                        .unwrap_or(true);
                    if is_visible {
                        // Extract value
                        values.extend_from_slice(buf_slice[..self.bytes_per_value].as_ref());
                        buf_slice = &buf_slice[self.bytes_per_value..];
                        visible_items += 1;
                    }
                }

                // Finally, we decompress the values and add them to our output buffer
                let values_buf = LanceBuffer::Owned(values);
                let fixed_data = FixedWidthDataBlock {
                    bits_per_value: self.bytes_per_value as u64 * 8,
                    block_info: BlockInfo::new(),
                    data: values_buf,
                    num_values: visible_items,
                };
                let PerValueDecompressor::Fixed(decompressor) = &self.details.value_decompressor
                else {
                    unreachable!()
                };
                let decompressed = decompressor.decompress(fixed_data, visible_items)?;
                data_builder.append(&decompressed, 0..visible_items);
            }

            let repetition = if rep.is_empty() { None } else { Some(rep) };
            let definition = if def.is_empty() { None } else { Some(def) };

            let unraveler =
                RepDefUnraveler::new(repetition, definition, self.details.def_meaning.clone());
            let data = data_builder.finish();

            Ok(DecodedPage {
                data,
                repdef: unraveler,
            })
        }
    }
}

#[derive(Debug)]
struct StructuralPrimitiveFieldSchedulingJob<'a> {
    scheduler: &'a StructuralPrimitiveFieldScheduler,
    ranges: Vec<Range<u64>>,
    page_idx: usize,
    range_idx: usize,
    global_row_offset: u64,
}

impl<'a> StructuralPrimitiveFieldSchedulingJob<'a> {
    pub fn new(scheduler: &'a StructuralPrimitiveFieldScheduler, ranges: Vec<Range<u64>>) -> Self {
        Self {
            scheduler,
            ranges,
            page_idx: 0,
            range_idx: 0,
            global_row_offset: 0,
        }
    }
}

impl StructuralSchedulingJob for StructuralPrimitiveFieldSchedulingJob<'_> {
    fn schedule_next(
        &mut self,
        context: &mut SchedulerContext,
    ) -> Result<Option<ScheduledScanLine>> {
        if self.range_idx >= self.ranges.len() {
            return Ok(None);
        }
        // Get our current range
        let mut range = self.ranges[self.range_idx].clone();
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
            .schedule_ranges(&ranges_in_page, context.io())?;

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
        decompressors: &dyn DecompressionStrategy,
        cache_repetition_index: bool,
    ) -> Result<Self> {
        let page_schedulers = column_info
            .page_infos
            .iter()
            .enumerate()
            .map(|(page_index, page_info)| {
                Self::page_info_to_scheduler(
                    page_info,
                    page_index,
                    column_info.index as usize,
                    decompressors,
                    cache_repetition_index,
                )
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
        _column_index: usize,
        decompressors: &dyn DecompressionStrategy,
        cache_repetition_index: bool,
    ) -> Result<PageInfoAndScheduler> {
        let scheduler: Box<dyn StructuralPageScheduler> =
            match page_info.encoding.as_structural().layout.as_ref() {
                Some(pb::page_layout::Layout::MiniBlockLayout(mini_block)) => {
                    Box::new(MiniBlockScheduler::try_new(
                        &page_info.buffer_offsets_and_sizes,
                        page_info.priority,
                        mini_block.num_items,
                        mini_block,
                        decompressors,
                    )?)
                }
                Some(pb::page_layout::Layout::FullZipLayout(full_zip)) => {
                    let mut scheduler = FullZipScheduler::try_new(
                        &page_info.buffer_offsets_and_sizes,
                        page_info.priority,
                        page_info.num_rows,
                        full_zip,
                        decompressors,
                    )?;
                    scheduler.enable_cache = cache_repetition_index;
                    Box::new(scheduler)
                }
                Some(pb::page_layout::Layout::AllNullLayout(all_null)) => {
                    let def_meaning = all_null
                        .layers
                        .iter()
                        .map(|l| ProtobufUtils::repdef_layer_to_def_interp(*l))
                        .collect::<Vec<_>>();
                    if def_meaning.len() == 1
                        && def_meaning[0] == DefinitionInterpretation::NullableItem
                    {
                        Box::new(SimpleAllNullScheduler::default())
                            as Box<dyn StructuralPageScheduler>
                    } else {
                        Box::new(ComplexAllNullScheduler::new(
                            page_info.buffer_offsets_and_sizes.clone(),
                            def_meaning.into(),
                        )) as Box<dyn StructuralPageScheduler>
                    }
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

pub trait CachedPageData: Any + Send + Sync + DeepSizeOf + 'static {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static>;
}

pub struct NoCachedPageData;

impl DeepSizeOf for NoCachedPageData {
    fn deep_size_of_children(&self, _ctx: &mut Context) -> usize {
        0
    }
}
impl CachedPageData for NoCachedPageData {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

pub struct CachedFieldData {
    pages: Vec<Arc<dyn CachedPageData>>,
}

impl DeepSizeOf for CachedFieldData {
    fn deep_size_of_children(&self, ctx: &mut Context) -> usize {
        self.pages.deep_size_of_children(ctx)
    }
}

// Cache key for field data
#[derive(Debug, Clone)]
pub struct FieldDataCacheKey {
    pub column_index: u32,
}

impl CacheKey for FieldDataCacheKey {
    type ValueType = CachedFieldData;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.column_index.to_string().into()
    }
}

impl StructuralFieldScheduler for StructuralPrimitiveFieldScheduler {
    fn initialize<'a>(
        &'a mut self,
        _filter: &'a FilterExpression,
        context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        let cache_key = FieldDataCacheKey {
            column_index: self.column_index,
        };
        let cache = context.cache().clone();

        async move {
            if let Some(cached_data) = cache.get_with_key(&cache_key).await {
                self.page_schedulers
                    .iter_mut()
                    .zip(cached_data.pages.iter())
                    .for_each(|(page_scheduler, cached_data)| {
                        page_scheduler.scheduler.load(cached_data);
                    });
                return Ok(());
            }

            let page_data = self
                .page_schedulers
                .iter_mut()
                .map(|s| s.scheduler.initialize(context.io()))
                .collect::<FuturesOrdered<_>>();

            let page_data = page_data.try_collect::<Vec<_>>().await?;
            let cached_data = Arc::new(CachedFieldData { pages: page_data });
            cache.insert_with_key(&cache_key, cached_data).await;
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

/// Takes the output from several pages decoders and
/// concatenates them.
#[derive(Debug)]
pub struct StructuralCompositeDecodeArrayTask {
    tasks: Vec<Box<dyn DecodePageTask>>,
    should_validate: bool,
    data_type: DataType,
}

impl StructuralCompositeDecodeArrayTask {
    fn restore_validity(
        array: Arc<dyn Array>,
        unraveler: &mut CompositeRepDefUnraveler,
    ) -> Arc<dyn Array> {
        let validity = unraveler.unravel_validity(array.len());
        let Some(validity) = validity else {
            return array;
        };
        if array.data_type() == &DataType::Null {
            // We unravel from a null array but we don't add the null buffer because arrow-rs doesn't like it
            return array;
        }
        assert_eq!(validity.len(), array.len());
        // SAFETY: We've should have already asserted the buffers are all valid, we are just
        // adding null buffers to the array here
        make_array(unsafe {
            array
                .to_data()
                .into_builder()
                .nulls(Some(validity))
                .build_unchecked()
        })
    }
}

impl StructuralDecodeArrayTask for StructuralCompositeDecodeArrayTask {
    fn decode(self: Box<Self>) -> Result<DecodedArray> {
        let mut arrays = Vec::with_capacity(self.tasks.len());
        let mut unravelers = Vec::with_capacity(self.tasks.len());
        for task in self.tasks {
            let decoded = task.decode()?;
            unravelers.push(decoded.repdef);

            let array = make_array(
                decoded
                    .data
                    .into_arrow(self.data_type.clone(), self.should_validate)?,
            );

            arrays.push(array);
        }
        let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
        let array = arrow_select::concat::concat(&array_refs)?;
        let mut repdef = CompositeRepDefUnraveler::new(unravelers);

        let array = Self::restore_validity(array, &mut repdef);

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
            should_validate: self.should_validate,
            data_type: self.field.data_type().clone(),
        }))
    }

    fn data_type(&self) -> &DataType {
        self.field.data_type()
    }
}

/// The serialized representation of full-zip data
struct SerializedFullZip {
    /// The zipped values buffer
    values: LanceBuffer,
    /// The repetition index (only present if there is repetition)
    repetition_index: Option<LanceBuffer>,
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

    keep_original_array: bool,
    accumulated_repdefs: Vec<RepDefBuilder>,
    // The compression strategy we will use to compress the data
    compression_strategy: Arc<dyn CompressionStrategy>,
    column_index: u32,
    field: Field,
    encoding_metadata: Arc<HashMap<String, String>>,
}

struct CompressedLevelsChunk {
    data: LanceBuffer,
    num_levels: u16,
}

struct CompressedLevels {
    data: Vec<CompressedLevelsChunk>,
    compression: pb::ArrayEncoding,
    rep_index: Option<LanceBuffer>,
}

struct SerializedMiniBlockPage {
    num_buffers: u64,
    data: LanceBuffer,
    metadata: LanceBuffer,
}

impl PrimitiveStructuralEncoder {
    pub fn try_new(
        options: &EncodingOptions,
        compression_strategy: Arc<dyn CompressionStrategy>,
        column_index: u32,
        field: Field,
        encoding_metadata: Arc<HashMap<String, String>>,
    ) -> Result<Self> {
        Ok(Self {
            accumulation_queue: AccumulationQueue::new(
                options.cache_bytes_per_column,
                column_index,
                options.keep_original_array,
            ),
            keep_original_array: options.keep_original_array,
            accumulated_repdefs: Vec::new(),
            column_index,
            compression_strategy,
            field,
            encoding_metadata,
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

    fn prefers_miniblock(
        data_block: &DataBlock,
        encoding_metadata: &HashMap<String, String>,
    ) -> bool {
        // If the user specifically requested miniblock then use it
        if let Some(user_requested) = encoding_metadata.get(STRUCTURAL_ENCODING_META_KEY) {
            return user_requested.to_lowercase() == STRUCTURAL_ENCODING_MINIBLOCK;
        }
        // Otherwise only use miniblock if it is narrow
        Self::is_narrow(data_block)
    }

    fn prefers_fullzip(encoding_metadata: &HashMap<String, String>) -> bool {
        // Fullzip is the backup option so the only reason we wouldn't use it is if the
        // user specifically requested not to use it (in which case we're probably going
        // to emit an error)
        if let Some(user_requested) = encoding_metadata.get(STRUCTURAL_ENCODING_META_KEY) {
            return user_requested.to_lowercase() == STRUCTURAL_ENCODING_FULLZIP;
        }
        true
    }

    // Converts value data, repetition levels, and definition levels into a single
    // buffer of mini-blocks.  In addition, creates a buffer of mini-block metadata
    // which tells us the size of each block.  Finally, if repetition is present then
    // we also create a buffer for the repetition index.
    //
    // Each chunk is serialized as:
    // | num_bufs (1 byte) | buf_lens (2 bytes per buffer) | P | buf0 | P | buf1 | ... | bufN | P |
    //
    // P - Padding inserted to ensure each buffer is 8-byte aligned and the buffer size is a multiple
    //     of 8 bytes (so that the next chunk is 8-byte aligned).
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
    //
    // If there is repetition then we also create a repetition index.  This is a
    // single buffer of integer vectors (stored in row major order).  There is one
    // entry for each chunk.  The size of the vector is based on the depth of random
    // access we want to support.
    //
    // A vector of size 2 is the minimum and will support row-based random access (e.g.
    // "take the 57th row").  A vector of size 3 will support 1 level of nested access
    // (e.g. "take the 3rd item in the 57th row").  A vector of size 4 will support 2
    // levels of nested access and so on.
    //
    // The first number in the vector is the number of top-level rows that complete in
    // the chunk.  The second number is the number of second-level rows that complete
    // after the final top-level row completed (or beginning of the chunk if no top-level
    // row completes in the chunk).  And so on.  The final number in the vector is always
    // the number of leftover items not covered by earlier entries in the vector.
    //
    // Currently we are limited to 0 levels of nested access but that will change in the
    // future.
    //
    // The repetition index and the chunk metadata are read at initialization time and
    // cached in memory.
    fn serialize_miniblocks(
        miniblocks: MiniBlockCompressed,
        rep: Option<Vec<CompressedLevelsChunk>>,
        def: Option<Vec<CompressedLevelsChunk>>,
    ) -> SerializedMiniBlockPage {
        let bytes_rep = rep
            .as_ref()
            .map(|rep| rep.iter().map(|r| r.data.len()).sum::<usize>())
            .unwrap_or(0);
        let bytes_def = def
            .as_ref()
            .map(|def| def.iter().map(|d| d.data.len()).sum::<usize>())
            .unwrap_or(0);
        let bytes_data = miniblocks.data.iter().map(|d| d.len()).sum::<usize>();
        let mut num_buffers = miniblocks.data.len();
        if rep.is_some() {
            num_buffers += 1;
        }
        if def.is_some() {
            num_buffers += 1;
        }
        // 2 bytes for the length of each buffer and up to 7 bytes of padding per buffer
        let max_extra = 9 * num_buffers;
        let mut data_buffer = Vec::with_capacity(bytes_rep + bytes_def + bytes_data + max_extra);
        let mut meta_buffer = Vec::with_capacity(miniblocks.chunks.len() * 2);

        let mut rep_iter = rep.map(|r| r.into_iter());
        let mut def_iter = def.map(|d| d.into_iter());

        let mut buffer_offsets = vec![0; miniblocks.data.len()];
        for chunk in miniblocks.chunks {
            let start_pos = data_buffer.len();
            // Start of chunk should be aligned
            debug_assert_eq!(start_pos % MINIBLOCK_ALIGNMENT, 0);

            let rep = rep_iter.as_mut().map(|r| r.next().unwrap());
            let def = def_iter.as_mut().map(|d| d.next().unwrap());

            // Write the number of levels, or 0 if there is no rep/def
            let num_levels = rep
                .as_ref()
                .map(|r| r.num_levels)
                .unwrap_or(def.as_ref().map(|d| d.num_levels).unwrap_or(0));
            data_buffer.extend_from_slice(&num_levels.to_le_bytes());

            // Write the buffer lengths
            if let Some(rep) = rep.as_ref() {
                let bytes_rep = u16::try_from(rep.data.len()).unwrap();
                data_buffer.extend_from_slice(&bytes_rep.to_le_bytes());
            }
            if let Some(def) = def.as_ref() {
                let bytes_def = u16::try_from(def.data.len()).unwrap();
                data_buffer.extend_from_slice(&bytes_def.to_le_bytes());
            }

            for buffer_size in &chunk.buffer_sizes {
                let bytes = *buffer_size;
                data_buffer.extend_from_slice(&bytes.to_le_bytes());
            }

            // Pad
            let add_padding = |data_buffer: &mut Vec<u8>| {
                let pad = pad_bytes::<MINIBLOCK_ALIGNMENT>(data_buffer.len());
                data_buffer.extend(iter::repeat_n(FILL_BYTE, pad));
            };
            add_padding(&mut data_buffer);

            // Write the buffers themselves
            if let Some(rep) = rep.as_ref() {
                data_buffer.extend_from_slice(&rep.data);
                add_padding(&mut data_buffer);
            }
            if let Some(def) = def.as_ref() {
                data_buffer.extend_from_slice(&def.data);
                add_padding(&mut data_buffer);
            }
            for (buffer_size, (buffer, buffer_offset)) in chunk
                .buffer_sizes
                .iter()
                .zip(miniblocks.data.iter().zip(buffer_offsets.iter_mut()))
            {
                let start = *buffer_offset;
                let end = start + *buffer_size as usize;
                *buffer_offset += *buffer_size as usize;
                data_buffer.extend_from_slice(&buffer[start..end]);
                add_padding(&mut data_buffer);
            }

            let chunk_bytes = data_buffer.len() - start_pos;
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

        let data_buffer = LanceBuffer::Owned(data_buffer);
        let metadata_buffer = LanceBuffer::Owned(meta_buffer);

        SerializedMiniBlockPage {
            num_buffers: miniblocks.data.len() as u64,
            data: data_buffer,
            metadata: metadata_buffer,
        }
    }

    /// Compresses a buffer of levels into chunks
    ///
    /// If these are repetition levels then we also calculate the repetition index here (that
    /// is the third return value)
    fn compress_levels(
        mut levels: RepDefSlicer<'_>,
        num_elements: u64,
        compression_strategy: &dyn CompressionStrategy,
        chunks: &[MiniBlockChunk],
        // This will be 0 if we are compressing def levels
        max_rep: u16,
    ) -> Result<CompressedLevels> {
        let mut rep_index = if max_rep > 0 {
            Vec::with_capacity(chunks.len())
        } else {
            vec![]
        };
        // Make the levels into a FixedWidth data block
        let num_levels = levels.num_levels() as u64;
        let mut levels_buf = levels.all_levels().try_clone().unwrap();
        let levels_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: levels_buf.borrow_and_clone(),
            bits_per_value: 16,
            num_values: num_levels,
            block_info: BlockInfo::new(),
        });
        let levels_field = Field::new_arrow("", DataType::UInt16, false)?;
        // Pick a block compressor
        let (compressor, compressor_desc) =
            compression_strategy.create_block_compressor(&levels_field, &levels_block)?;
        // Compress blocks of levels (sized according to the chunks)
        let mut level_chunks = Vec::with_capacity(chunks.len());
        let mut values_counter = 0;
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_num_values = chunk.num_values(values_counter, num_elements);
            values_counter += chunk_num_values;
            let mut chunk_levels = if chunk_idx < chunks.len() - 1 {
                levels.slice_next(chunk_num_values as usize)
            } else {
                levels.slice_rest()
            };
            let num_chunk_levels = (chunk_levels.len() / 2) as u64;
            if max_rep > 0 {
                // If max_rep > 0 then we are working with rep levels and we need
                // to calculate the repetition index.  The repetition index for a
                // chunk is currently 2 values (in the future it may be more).
                //
                // The first value is the number of rows that _finish_ in the
                // chunk.
                //
                // The second value is the number of "leftovers" after the last
                // finished row in the chunk.
                let rep_values = chunk_levels.borrow_to_typed_slice::<u16>();
                let rep_values = rep_values.as_ref();

                // We skip 1 here because a max_rep at spot 0 doesn't count as a finished list (we
                // will count it in the previous chunk)
                let mut num_rows = rep_values.iter().skip(1).filter(|v| **v == max_rep).count();
                let num_leftovers = if chunk_idx < chunks.len() - 1 {
                    rep_values
                        .iter()
                        .rev()
                        .position(|v| *v == max_rep)
                        // # of leftovers includes the max_rep spot
                        .map(|pos| pos + 1)
                        .unwrap_or(rep_values.len())
                } else {
                    // Last chunk can't have leftovers
                    0
                };

                if chunk_idx != 0 && rep_values[0] == max_rep {
                    // This chunk starts with a new row and so, if we thought we had leftovers
                    // in the previous chunk, we were mistaken
                    // TODO: Can use unchecked here
                    let rep_len = rep_index.len();
                    if rep_index[rep_len - 1] != 0 {
                        // We thought we had leftovers but that was actually a full row
                        rep_index[rep_len - 2] += 1;
                        rep_index[rep_len - 1] = 0;
                    }
                }

                if chunk_idx == chunks.len() - 1 {
                    // The final list
                    num_rows += 1;
                }
                rep_index.push(num_rows as u64);
                rep_index.push(num_leftovers as u64);
            }
            let chunk_levels_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: chunk_levels,
                bits_per_value: 16,
                num_values: num_chunk_levels,
                block_info: BlockInfo::new(),
            });
            let compressed_levels = compressor.compress(chunk_levels_block)?;
            level_chunks.push(CompressedLevelsChunk {
                data: compressed_levels,
                num_levels: num_chunk_levels as u16,
            });
        }
        debug_assert_eq!(levels.num_levels_remaining(), 0);
        let rep_index = if rep_index.is_empty() {
            None
        } else {
            Some(LanceBuffer::reinterpret_vec(rep_index))
        };
        Ok(CompressedLevels {
            data: level_chunks,
            compression: compressor_desc,
            rep_index,
        })
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

    // Encodes a page where all values are null but we have rep/def
    // information that we need to store (e.g. to distinguish between
    // different kinds of null)
    fn encode_complex_all_null(
        column_idx: u32,
        repdefs: Vec<RepDefBuilder>,
        row_number: u64,
        num_rows: u64,
    ) -> Result<EncodedPage> {
        let repdef = RepDefBuilder::serialize(repdefs);

        // TODO: Actually compress repdef
        let rep_bytes = if let Some(rep) = repdef.repetition_levels.as_ref() {
            LanceBuffer::reinterpret_slice(rep.clone())
        } else {
            LanceBuffer::empty()
        };

        let def_bytes = if let Some(def) = repdef.definition_levels.as_ref() {
            LanceBuffer::reinterpret_slice(def.clone())
        } else {
            LanceBuffer::empty()
        };

        let description = ProtobufUtils::all_null_layout(&repdef.def_meaning);
        Ok(EncodedPage {
            column_idx,
            data: vec![rep_bytes, def_bytes],
            description: PageEncoding::Structural(description),
            num_rows,
            row_number,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_miniblock(
        column_idx: u32,
        field: &Field,
        compression_strategy: &dyn CompressionStrategy,
        data: DataBlock,
        repdefs: Vec<RepDefBuilder>,
        row_number: u64,
        dictionary_data: Option<DataBlock>,
        num_rows: u64,
    ) -> Result<EncodedPage> {
        let repdef = RepDefBuilder::serialize(repdefs);

        if let DataBlock::AllNull(_null_block) = data {
            // If we got here then all the data is null but we have rep/def information that
            // we need to store.
            todo!()
        }

        let num_items = data.num_values();

        let compressor = compression_strategy.create_miniblock_compressor(field, &data)?;
        let (compressed_data, value_encoding) = compressor.compress(data)?;

        let max_rep = repdef.def_meaning.iter().filter(|l| l.is_list()).count() as u16;

        let mut compressed_rep = repdef
            .rep_slicer()
            .map(|rep_slicer| {
                Self::compress_levels(
                    rep_slicer,
                    num_items,
                    compression_strategy,
                    &compressed_data.chunks,
                    max_rep,
                )
            })
            .transpose()?;

        let (rep_index, rep_index_depth) =
            match compressed_rep.as_mut().and_then(|cr| cr.rep_index.as_mut()) {
                Some(rep_index) => (Some(rep_index.borrow_and_clone()), 1),
                None => (None, 0),
            };

        let mut compressed_def = repdef
            .def_slicer()
            .map(|def_slicer| {
                Self::compress_levels(
                    def_slicer,
                    num_items,
                    compression_strategy,
                    &compressed_data.chunks,
                    /*max_rep=*/ 0,
                )
            })
            .transpose()?;

        // TODO: Parquet sparsely encodes values here.  We could do the same but
        // then we won't have log2 values per chunk.  This means more metadata
        // and potentially more decoder asymmetry.  However, it may be worth
        // investigating at some point

        let rep_data = compressed_rep
            .as_mut()
            .map(|cr| std::mem::take(&mut cr.data));
        let def_data = compressed_def
            .as_mut()
            .map(|cd| std::mem::take(&mut cd.data));

        let serialized = Self::serialize_miniblocks(compressed_data, rep_data, def_data);

        // Metadata, Data, Dictionary, (maybe) Repetition Index
        let mut data = Vec::with_capacity(4);
        data.push(serialized.metadata);
        data.push(serialized.data);

        if let Some(dictionary_data) = dictionary_data {
            let num_dictionary_items = dictionary_data.num_values();
            // field in `create_block_compressor` is not used currently.
            let dummy_dictionary_field = Field::new_arrow("", DataType::UInt16, false)?;

            let (compressor, dictionary_encoding) = compression_strategy
                .create_block_compressor(&dummy_dictionary_field, &dictionary_data)?;
            let dictionary_buffer = compressor.compress(dictionary_data)?;

            data.push(dictionary_buffer);
            if let Some(rep_index) = rep_index {
                data.push(rep_index);
            }

            let description = ProtobufUtils::miniblock_layout(
                compressed_rep.map(|cr| cr.compression),
                compressed_def.map(|cd| cd.compression),
                value_encoding,
                rep_index_depth,
                serialized.num_buffers,
                Some((dictionary_encoding, num_dictionary_items)),
                &repdef.def_meaning,
                num_items,
            );
            Ok(EncodedPage {
                num_rows,
                column_idx,
                data,
                description: PageEncoding::Structural(description),
                row_number,
            })
        } else {
            let description = ProtobufUtils::miniblock_layout(
                compressed_rep.map(|cr| cr.compression),
                compressed_def.map(|cd| cd.compression),
                value_encoding,
                rep_index_depth,
                serialized.num_buffers,
                None,
                &repdef.def_meaning,
                num_items,
            );

            if let Some(mut rep_index) = rep_index {
                let view = rep_index.borrow_to_typed_slice::<u64>();
                let total = view.chunks_exact(2).map(|c| c[0]).sum::<u64>();
                debug_assert_eq!(total, num_rows);

                data.push(rep_index);
            }

            Ok(EncodedPage {
                num_rows,
                column_idx,
                data,
                description: PageEncoding::Structural(description),
                row_number,
            })
        }
    }

    // For fixed-size data we encode < control word | data > for each value
    fn serialize_full_zip_fixed(
        fixed: FixedWidthDataBlock,
        mut repdef: ControlWordIterator,
        num_values: u64,
    ) -> SerializedFullZip {
        let len = fixed.data.len() + repdef.bytes_per_word() * num_values as usize;
        let mut zipped_data = Vec::with_capacity(len);

        let max_rep_index_val = if repdef.has_repetition() {
            len as u64
        } else {
            // Setting this to 0 means we won't write a repetition index
            0
        };
        let mut rep_index_builder =
            BytepackedIntegerEncoder::with_capacity(num_values as usize + 1, max_rep_index_val);

        // I suppose we can just pad to the nearest byte but I'm not sure we need to worry about this anytime soon
        // because it is unlikely compression of large values is going to yield a result that is not byte aligned
        assert_eq!(
            fixed.bits_per_value % 8,
            0,
            "Non-byte aligned full-zip compression not yet supported"
        );

        let bytes_per_value = fixed.bits_per_value as usize / 8;

        let mut data_iter = fixed.data.chunks_exact(bytes_per_value);
        let mut offset = 0;
        while let Some(control) = repdef.append_next(&mut zipped_data) {
            if control.is_new_row {
                // We have finished a row
                debug_assert!(offset <= len);
                // SAFETY: We know that `start <= len`
                unsafe { rep_index_builder.append(offset as u64) };
            }
            if control.is_visible {
                let value = data_iter.next().unwrap();
                zipped_data.extend_from_slice(value);
            }
            offset = zipped_data.len();
        }

        debug_assert_eq!(zipped_data.len(), len);
        // Put the final value in the rep index
        // SAFETY: `zipped_data.len() == len`
        unsafe {
            rep_index_builder.append(zipped_data.len() as u64);
        }

        let zipped_data = LanceBuffer::Owned(zipped_data);
        let rep_index = rep_index_builder.into_data();
        let rep_index = if rep_index.is_empty() {
            None
        } else {
            Some(LanceBuffer::Owned(rep_index))
        };
        SerializedFullZip {
            values: zipped_data,
            repetition_index: rep_index,
        }
    }

    // For variable-size data we encode < control word | length | data > for each value
    //
    // In addition, we create a second buffer, the repetition index
    fn serialize_full_zip_variable(
        mut variable: VariableWidthBlock,
        mut repdef: ControlWordIterator,
        num_items: u64,
    ) -> SerializedFullZip {
        let bytes_per_offset = variable.bits_per_offset as usize / 8;
        assert_eq!(
            variable.bits_per_offset % 8,
            0,
            "Only byte-aligned offsets supported"
        );
        let len = variable.data.len()
            + repdef.bytes_per_word() * num_items as usize
            + bytes_per_offset * variable.num_values as usize;
        let mut buf = Vec::with_capacity(len);

        let max_rep_index_val = len as u64;
        let mut rep_index_builder =
            BytepackedIntegerEncoder::with_capacity(num_items as usize + 1, max_rep_index_val);

        // TODO: byte pack the item lengths with varint encoding
        match bytes_per_offset {
            4 => {
                let offs = variable.offsets.borrow_to_typed_slice::<u32>();
                let mut rep_offset = 0;
                let mut windows_iter = offs.as_ref().windows(2);
                while let Some(control) = repdef.append_next(&mut buf) {
                    if control.is_new_row {
                        // We have finished a row
                        debug_assert!(rep_offset <= len);
                        // SAFETY: We know that `buf.len() <= len`
                        unsafe { rep_index_builder.append(rep_offset as u64) };
                    }
                    if control.is_visible {
                        let window = windows_iter.next().unwrap();
                        if control.is_valid_item {
                            buf.extend_from_slice(&(window[1] - window[0]).to_le_bytes());
                            buf.extend_from_slice(
                                &variable.data[window[0] as usize..window[1] as usize],
                            );
                        }
                    }
                    rep_offset = buf.len();
                }
            }
            8 => {
                let offs = variable.offsets.borrow_to_typed_slice::<u64>();
                let mut rep_offset = 0;
                let mut windows_iter = offs.as_ref().windows(2);
                while let Some(control) = repdef.append_next(&mut buf) {
                    if control.is_new_row {
                        // We have finished a row
                        debug_assert!(rep_offset <= len);
                        // SAFETY: We know that `buf.len() <= len`
                        unsafe { rep_index_builder.append(rep_offset as u64) };
                    }
                    if control.is_visible {
                        let window = windows_iter.next().unwrap();
                        if control.is_valid_item {
                            buf.extend_from_slice(&(window[1] - window[0]).to_le_bytes());
                            buf.extend_from_slice(
                                &variable.data[window[0] as usize..window[1] as usize],
                            );
                        }
                    }
                    rep_offset = buf.len();
                }
            }
            _ => panic!("Unsupported offset size"),
        }

        // We might have saved a few bytes by not copying lengths when the length was zero.  However,
        // if we are over `len` then we have a bug.
        debug_assert!(buf.len() <= len);
        // Put the final value in the rep index
        // SAFETY: `zipped_data.len() == len`
        unsafe {
            rep_index_builder.append(buf.len() as u64);
        }

        let zipped_data = LanceBuffer::Owned(buf);
        let rep_index = rep_index_builder.into_data();
        debug_assert!(!rep_index.is_empty());
        let rep_index = Some(LanceBuffer::Owned(rep_index));
        SerializedFullZip {
            values: zipped_data,
            repetition_index: rep_index,
        }
    }

    /// Serializes data into a single buffer according to the full-zip format which zips
    /// together the repetition, definition, and value data into a single buffer.
    fn serialize_full_zip(
        compressed_data: PerValueDataBlock,
        repdef: ControlWordIterator,
        num_items: u64,
    ) -> SerializedFullZip {
        match compressed_data {
            PerValueDataBlock::Fixed(fixed) => {
                Self::serialize_full_zip_fixed(fixed, repdef, num_items)
            }
            PerValueDataBlock::Variable(var) => {
                Self::serialize_full_zip_variable(var, repdef, num_items)
            }
        }
    }

    fn encode_full_zip(
        column_idx: u32,
        field: &Field,
        compression_strategy: &dyn CompressionStrategy,
        data: DataBlock,
        repdefs: Vec<RepDefBuilder>,
        row_number: u64,
        num_lists: u64,
    ) -> Result<EncodedPage> {
        let repdef = RepDefBuilder::serialize(repdefs);
        let max_rep = repdef
            .repetition_levels
            .as_ref()
            .map_or(0, |r| r.iter().max().copied().unwrap_or(0));
        let max_def = repdef
            .definition_levels
            .as_ref()
            .map_or(0, |d| d.iter().max().copied().unwrap_or(0));

        // To handle FSL we just flatten
        // let data = data.flatten();

        let (num_items, num_visible_items) =
            if let Some(rep_levels) = repdef.repetition_levels.as_ref() {
                // If there are rep levels there may be "invisible" items and we need to encode
                // rep_levels.len() things which might be larger than data.num_values()
                (rep_levels.len() as u64, data.num_values())
            } else {
                // If there are no rep levels then we encode data.num_values() things
                (data.num_values(), data.num_values())
            };

        let max_visible_def = repdef.max_visible_level.unwrap_or(u16::MAX);

        let repdef_iter = build_control_word_iterator(
            repdef.repetition_levels.as_deref(),
            max_rep,
            repdef.definition_levels.as_deref(),
            max_def,
            max_visible_def,
            num_items as usize,
        );
        let bits_rep = repdef_iter.bits_rep();
        let bits_def = repdef_iter.bits_def();

        let compressor = compression_strategy.create_per_value(field, &data)?;
        let (compressed_data, value_encoding) = compressor.compress(data)?;

        let description = match &compressed_data {
            PerValueDataBlock::Fixed(fixed) => ProtobufUtils::fixed_full_zip_layout(
                bits_rep,
                bits_def,
                fixed.bits_per_value as u32,
                value_encoding,
                &repdef.def_meaning,
                num_items as u32,
                num_visible_items as u32,
            ),
            PerValueDataBlock::Variable(variable) => ProtobufUtils::variable_full_zip_layout(
                bits_rep,
                bits_def,
                variable.bits_per_offset as u32,
                value_encoding,
                &repdef.def_meaning,
                num_items as u32,
                num_visible_items as u32,
            ),
        };

        let zipped = Self::serialize_full_zip(compressed_data, repdef_iter, num_items);

        let data = if let Some(repindex) = zipped.repetition_index {
            vec![zipped.values, repindex]
        } else {
            vec![zipped.values]
        };

        Ok(EncodedPage {
            num_rows: num_lists,
            column_idx,
            data,
            description: PageEncoding::Structural(description),
            row_number,
        })
    }

    fn dictionary_encode(mut data_block: DataBlock) -> (DataBlock, DataBlock) {
        let cardinality = data_block
            .get_stat(Stat::Cardinality)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .value(0);
        match data_block {
            DataBlock::FixedWidth(ref mut fixed_width_data_block) => {
                // Currently FixedWidth DataBlock with only bits_per_value 128 has cardinality
                // TODO: a follow up PR to support `FixedWidth DataBlock with bits_per_value == 256`.
                let mut map = HashMap::new();
                let u128_slice = fixed_width_data_block.data.borrow_to_typed_slice::<u128>();
                let u128_slice = u128_slice.as_ref();
                let mut dictionary_buffer = Vec::with_capacity(cardinality as usize);
                let mut indices_buffer =
                    Vec::with_capacity(fixed_width_data_block.num_values as usize);
                let mut curr_idx: i32 = 0;
                u128_slice.iter().for_each(|&value| {
                    let idx = *map.entry(value).or_insert_with(|| {
                        dictionary_buffer.push(value);
                        curr_idx += 1;
                        curr_idx - 1
                    });
                    indices_buffer.push(idx);
                });
                let dictionary_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::reinterpret_vec(dictionary_buffer),
                    bits_per_value: 128,
                    num_values: curr_idx as u64,
                    block_info: BlockInfo::default(),
                });
                let mut indices_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::reinterpret_vec(indices_buffer),
                    bits_per_value: 32,
                    num_values: fixed_width_data_block.num_values,
                    block_info: BlockInfo::default(),
                });
                // Todo: if we decide to do eager statistics computing, wrap statistics computing
                // in DataBlock constructor.
                indices_data_block.compute_stat();

                (indices_data_block, dictionary_data_block)
            }
            DataBlock::VariableWidth(ref mut variable_width_data_block) => {
                match variable_width_data_block.bits_per_offset {
                    32 => {
                        let mut map = HashMap::new();
                        let offsets = variable_width_data_block
                            .offsets
                            .borrow_to_typed_slice::<u32>();
                        let offsets = offsets.as_ref();

                        let max_len = variable_width_data_block.get_stat(Stat::MaxLength).expect(
                            "VariableWidth DataBlock should have valid `Stat::DataSize` statistics",
                        );
                        let max_len = max_len.as_primitive::<UInt64Type>().value(0);

                        let mut dictionary_buffer: Vec<u8> =
                            Vec::with_capacity((max_len * cardinality) as usize);
                        let mut dictionary_offsets_buffer = vec![0];
                        let mut curr_idx = 0;
                        let mut indices_buffer =
                            Vec::with_capacity(variable_width_data_block.num_values as usize);

                        offsets
                            .iter()
                            .zip(offsets.iter().skip(1))
                            .for_each(|(&start, &end)| {
                                let key =
                                    &variable_width_data_block.data[start as usize..end as usize];
                                let idx: i32 = *map.entry(U8SliceKey(key)).or_insert_with(|| {
                                    dictionary_buffer.extend_from_slice(key);
                                    dictionary_offsets_buffer.push(dictionary_buffer.len() as u32);
                                    curr_idx += 1;
                                    curr_idx - 1
                                });
                                indices_buffer.push(idx);
                            });

                        let dictionary_data_block = DataBlock::VariableWidth(VariableWidthBlock {
                            data: LanceBuffer::reinterpret_vec(dictionary_buffer),
                            offsets: LanceBuffer::reinterpret_vec(dictionary_offsets_buffer),
                            bits_per_offset: 32,
                            num_values: curr_idx as u64,
                            block_info: BlockInfo::default(),
                        });

                        let mut indices_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                            data: LanceBuffer::reinterpret_vec(indices_buffer),
                            bits_per_value: 32,
                            num_values: variable_width_data_block.num_values,
                            block_info: BlockInfo::default(),
                        });
                        // Todo: if we decide to do eager statistics computing, wrap statistics computing
                        // in DataBlock constructor.
                        indices_data_block.compute_stat();

                        (indices_data_block, dictionary_data_block)
                    }
                    64 => {
                        let mut map = HashMap::new();
                        let offsets = variable_width_data_block
                            .offsets
                            .borrow_to_typed_slice::<u64>();
                        let offsets = offsets.as_ref();

                        let max_len = variable_width_data_block.get_stat(Stat::MaxLength).expect(
                            "VariableWidth DataBlock should have valid `Stat::DataSize` statistics",
                        );
                        let max_len = max_len.as_primitive::<UInt64Type>().value(0);

                        let mut dictionary_buffer: Vec<u8> =
                            Vec::with_capacity((max_len * cardinality) as usize);
                        let mut dictionary_offsets_buffer = vec![0];
                        let mut curr_idx = 0;
                        let mut indices_buffer =
                            Vec::with_capacity(variable_width_data_block.num_values as usize);

                        offsets
                            .iter()
                            .zip(offsets.iter().skip(1))
                            .for_each(|(&start, &end)| {
                                let key =
                                    &variable_width_data_block.data[start as usize..end as usize];
                                let idx: i64 = *map.entry(U8SliceKey(key)).or_insert_with(|| {
                                    dictionary_buffer.extend_from_slice(key);
                                    dictionary_offsets_buffer.push(dictionary_buffer.len() as u64);
                                    curr_idx += 1;
                                    curr_idx - 1
                                });
                                indices_buffer.push(idx);
                            });

                        let dictionary_data_block = DataBlock::VariableWidth(VariableWidthBlock {
                            data: LanceBuffer::reinterpret_vec(dictionary_buffer),
                            offsets: LanceBuffer::reinterpret_vec(dictionary_offsets_buffer),
                            bits_per_offset: 64,
                            num_values: curr_idx as u64,
                            block_info: BlockInfo::default(),
                        });

                        let mut indices_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                            data: LanceBuffer::reinterpret_vec(indices_buffer),
                            bits_per_value: 64,
                            num_values: variable_width_data_block.num_values,
                            block_info: BlockInfo::default(),
                        });
                        // Todo: if we decide to do eager statistics computing, wrap statistics computing
                        // in DataBlock constructor.
                        indices_data_block.compute_stat();

                        (indices_data_block, dictionary_data_block)
                    }
                    _ => {
                        unreachable!()
                    }
                }
            }
            _ => {
                unreachable!("dictionary encode called with data block {:?}", data_block)
            }
        }
    }

    fn should_dictionary_encode(data_block: &DataBlock, field: &Field) -> bool {
        // Don't dictionary encode tiny arrays
        let too_small = env::var("LANCE_ENCODING_DICT_TOO_SMALL")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(100);
        if data_block.num_values() < too_small {
            return false;
        }

        // Somewhat arbitrary threshold rule.  Apply dictionary encoding if the number of unique
        // values is less than 1/2 the total number of values.
        let divisor: u64 = field
            .metadata
            .get(DICT_DIVISOR_META_KEY)
            .map(|val| val.parse().ok())
            .unwrap_or_else(|| {
                env::var("LANCE_ENCODING_DICT_DIVISOR")
                    .ok()
                    .and_then(|val| val.parse().ok())
            })
            .unwrap_or(2);

        // Cap on cardinality.  This should be pushed into the cardinality estimation to avoid
        // spending too much time estimating cardinality.
        let max_cardinality = env::var("LANCE_ENCODING_DICT_MAX_CARDINALITY")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(100000);

        let threshold = (data_block.num_values() / divisor).min(max_cardinality);

        let cardinality = if let Some(cardinality_array) = data_block.get_stat(Stat::Cardinality) {
            cardinality_array.as_primitive::<UInt64Type>().value(0)
        } else {
            u64::MAX
        };

        cardinality < threshold
    }

    // Creates an encode task, consuming all buffered data
    fn do_flush(
        &mut self,
        arrays: Vec<ArrayRef>,
        repdefs: Vec<RepDefBuilder>,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        let column_idx = self.column_index;
        let compression_strategy = self.compression_strategy.clone();
        let field = self.field.clone();
        let encoding_metadata = self.encoding_metadata.clone();
        let task = spawn_cpu(move || {
            let num_values = arrays.iter().map(|arr| arr.len() as u64).sum();

            if num_values == 0 {
                // We should not encode empty arrays.  So if we get here that should mean that we
                // either have all empty lists or all null lists (or a mix).  We still need to encode
                // the rep/def information but we can skip the data encoding.
                return Self::encode_complex_all_null(column_idx, repdefs, row_number, num_rows);
            }
            let num_nulls = arrays
                .iter()
                .map(|arr| arr.logical_nulls().map(|n| n.null_count()).unwrap_or(0) as u64)
                .sum::<u64>();

            if num_values == num_nulls {
                return if repdefs.iter().all(|rd| rd.is_simple_validity()) {
                    log::debug!(
                        "Encoding column {} with {} items using simple-null layout",
                        column_idx,
                        num_values
                    );
                    // Simple case, no rep/def and all nulls, we don't need to encode any data
                    Self::encode_simple_all_null(column_idx, num_values, row_number)
                } else {
                    // If we get here then we have definition levels and we need to store those
                    Self::encode_complex_all_null(column_idx, repdefs, row_number, num_rows)
                };
            }

            let data_block = DataBlock::from_arrays(&arrays, num_values);

            // if the `data_block` is a `StructDataBlock`, then this is a struct with packed struct encoding.
            if let DataBlock::Struct(ref struct_data_block) = data_block {
                if struct_data_block
                    .children
                    .iter()
                    .any(|child| !matches!(child, DataBlock::FixedWidth(_)))
                {
                    panic!("packed struct encoding currently only supports fixed-width fields.")
                }
            }

            // The top-level validity is encoded in repdef so we can remove it.
            let data_block = data_block.remove_outer_validity();


            if Self::should_dictionary_encode(&data_block, &field) {
                log::debug!(
                    "Encoding column {} with {} items using dictionary encoding (mini-block layout)",
                    column_idx,
                    num_values
                );
                let (indices_data_block, dictionary_data_block) =
                    Self::dictionary_encode(data_block);
                Self::encode_miniblock(
                    column_idx,
                    &field,
                    compression_strategy.as_ref(),
                    indices_data_block,
                    repdefs,
                    row_number,
                    Some(dictionary_data_block),
                    num_rows,
                )
            } else if Self::prefers_miniblock(&data_block, encoding_metadata.as_ref()) {
                log::debug!(
                    "Encoding column {} with {} items using mini-block layout",
                    column_idx,
                    num_values
                );
                Self::encode_miniblock(
                    column_idx,
                    &field,
                    compression_strategy.as_ref(),
                    data_block,
                    repdefs,
                    row_number,
                    None,
                    num_rows,
                )
            } else if Self::prefers_fullzip(encoding_metadata.as_ref()) {
                log::debug!(
                    "Encoding column {} with {} items using full-zip layout",
                    column_idx,
                    num_values
                );
                Self::encode_full_zip(
                    column_idx,
                    &field,
                    compression_strategy.as_ref(),
                    data_block,
                    repdefs,
                    row_number,
                    num_rows,
                )
            } else {
                Err(Error::InvalidInput { source: format!("Cannot determine structural encoding for field {}.  This typically indicates an invalid value of the field metadata key {}", field.name, STRUCTURAL_ENCODING_META_KEY).into(), location: location!() })
            }
        })
        .boxed();
        Ok(vec![task])
    }

    fn extract_validity_buf(
        array: &dyn Array,
        repdef: &mut RepDefBuilder,
        keep_original_array: bool,
    ) {
        if let Some(validity) = array.nulls() {
            if keep_original_array {
                repdef.add_validity_bitmap(validity.clone());
            } else {
                repdef.add_validity_bitmap(deep_copy_nulls(Some(validity)).unwrap());
            }
        } else {
            repdef.add_no_null(array.len());
        }
    }

    fn extract_validity(array: &dyn Array, repdef: &mut RepDefBuilder, keep_original_array: bool) {
        match array.data_type() {
            DataType::Null => {
                repdef.add_validity_bitmap(NullBuffer::new(BooleanBuffer::new_unset(array.len())));
            }
            DataType::Dictionary(_, _) => {
                unreachable!()
            }
            // Extract our validity buf but NOT any child validity bufs. (they will be encoded in
            // as part of the values).  Note: for FSL we do not use repdef.add_fsl because we do
            // NOT want to increase the repdef depth.
            //
            // This would be quite catasrophic for something like vector embeddings.  Imagine we
            // had thousands of vectors and some were null but no vector contained null items.  If
            // we treated the vectors (primitive FSL) like we treat structural FSL we would end up
            // with a rep/def value for every single item in the vector.
            _ => Self::extract_validity_buf(array, repdef, keep_original_array),
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
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        Self::extract_validity(array.as_ref(), &mut repdef, self.keep_original_array);
        self.accumulated_repdefs.push(repdef);

        if let Some((arrays, row_number, num_rows)) =
            self.accumulation_queue.insert(array, row_number, num_rows)
        {
            let accumulated_repdefs = std::mem::take(&mut self.accumulated_repdefs);
            Ok(self.do_flush(arrays, accumulated_repdefs, row_number, num_rows)?)
        } else {
            Ok(vec![])
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self, _external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        if let Some((arrays, row_number, num_rows)) = self.accumulation_queue.flush() {
            let accumulated_repdefs = std::mem::take(&mut self.accumulated_repdefs);
            Ok(self.do_flush(arrays, accumulated_repdefs, row_number, num_rows)?)
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
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::{collections::VecDeque, sync::Arc};

    use crate::encodings::logical::primitive::{
        ChunkDrainInstructions, PrimitiveStructuralEncoder,
    };
    use arrow_array::{ArrayRef, Int8Array, StringArray};

    use super::{
        ChunkInstructions, DataBlock, DecodeMiniBlockTask, FixedPerValueDecompressor,
        FixedWidthDataBlock, FullZipCacheableState, FullZipDecodeDetails, FullZipRepIndexDetails,
        FullZipScheduler, MiniBlockRepIndex, PerValueDecompressor, PreambleAction,
        StructuralPageScheduler,
    };

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

    #[test]
    fn test_map_range() {
        // Null in the middle
        // [[A, B, C], [D, E], NULL, [F, G, H]]
        let rep = Some(vec![1, 0, 0, 1, 0, 1, 1, 0, 0]);
        let def = Some(vec![0, 0, 0, 0, 0, 1, 0, 0, 0]);
        let max_visible_def = 0;
        let total_items = 8;
        let max_rep = 1;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..3, 0..3);
        check(1..2, 3..5, 3..5);
        check(2..3, 5..5, 5..6);
        check(3..4, 5..8, 6..9);
        check(0..2, 0..5, 0..5);
        check(1..3, 3..5, 3..6);
        check(2..4, 5..8, 5..9);
        check(0..3, 0..5, 0..6);
        check(1..4, 3..8, 3..9);
        check(0..4, 0..8, 0..9);

        // Null at start
        // [NULL, [A, B], [C]]
        let rep = Some(vec![1, 1, 0, 1]);
        let def = Some(vec![1, 0, 0, 0]);
        let max_visible_def = 0;
        let total_items = 3;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..0, 0..1);
        check(1..2, 0..2, 1..3);
        check(2..3, 2..3, 3..4);
        check(0..2, 0..2, 0..3);
        check(1..3, 0..3, 1..4);
        check(0..3, 0..3, 0..4);

        // Null at end
        // [[A], [B, C], NULL]
        let rep = Some(vec![1, 1, 0, 1]);
        let def = Some(vec![0, 0, 0, 1]);
        let max_visible_def = 0;
        let total_items = 3;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..1, 0..1);
        check(1..2, 1..3, 1..3);
        check(2..3, 3..3, 3..4);
        check(0..2, 0..3, 0..3);
        check(1..3, 1..3, 1..4);
        check(0..3, 0..3, 0..4);

        // No nulls, with repetition
        // [[A, B], [C, D], [E, F]]
        let rep = Some(vec![1, 0, 1, 0, 1, 0]);
        let def: Option<&[u16]> = None;
        let max_visible_def = 0;
        let total_items = 6;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..2, 0..2);
        check(1..2, 2..4, 2..4);
        check(2..3, 4..6, 4..6);
        check(0..2, 0..4, 0..4);
        check(1..3, 2..6, 2..6);
        check(0..3, 0..6, 0..6);

        // No repetition, with nulls (this case is trivial)
        // [A, B, NULL, C]
        let rep: Option<&[u16]> = None;
        let def = Some(vec![0, 0, 1, 0]);
        let max_visible_def = 1;
        let total_items = 4;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..1, 0..1);
        check(1..2, 1..2, 1..2);
        check(2..3, 2..3, 2..3);
        check(0..2, 0..2, 0..2);
        check(1..3, 1..3, 1..3);
        check(0..3, 0..3, 0..3);

        // Tricky case, this chunk is a continuation and starts with a rep-index = 0
        // [[..., A] [B, C], NULL]
        //
        // What we do will depend on the preamble action
        let rep = Some(vec![0, 1, 0, 1]);
        let def = Some(vec![0, 0, 0, 1]);
        let max_visible_def = 0;
        let total_items = 3;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Take,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 0..3, 0..3);
        check(0..2, 0..3, 0..4);

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Skip,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 1..3, 1..3);
        check(1..2, 3..3, 3..4);
        check(0..2, 1..3, 1..4);

        // Another preamble case but now it doesn't end with a new list
        // [[..., A], NULL, [D, E]]
        //
        // What we do will depend on the preamble action
        let rep = Some(vec![0, 1, 1, 0]);
        let def = Some(vec![0, 1, 0, 0]);
        let max_visible_def = 0;
        let total_items = 4;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Take,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 0..1, 0..2);
        check(0..2, 0..3, 0..4);

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Skip,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 1..1, 1..2);
        check(1..2, 1..3, 2..4);
        check(0..2, 1..3, 1..4);

        // Now a preamble case without any definition levels
        // [[..., A] [B, C], [D]]
        let rep = Some(vec![0, 1, 0, 1]);
        let def: Option<Vec<u16>> = None;
        let max_visible_def = 0;
        let total_items = 4;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Take,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 0..3, 0..3);
        check(0..2, 0..4, 0..4);

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Skip,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 1..3, 1..3);
        check(1..2, 3..4, 3..4);
        check(0..2, 1..4, 1..4);
    }

    #[test]
    fn test_schedule_instructions() {
        let repetition_index = vec![vec![5, 2], vec![3, 0], vec![4, 7], vec![2, 0]];
        let repetition_index = MiniBlockRepIndex::decode(&repetition_index);

        let check = |user_ranges, expected_instructions| {
            let instructions =
                ChunkInstructions::schedule_instructions(&repetition_index, user_ranges);
            assert_eq!(instructions, expected_instructions);
        };

        // The instructions we expect if we're grabbing the whole range
        let expected_take_all = vec![
            ChunkInstructions {
                chunk_idx: 0,
                preamble: PreambleAction::Absent,
                rows_to_skip: 0,
                rows_to_take: 5,
                take_trailer: true,
            },
            ChunkInstructions {
                chunk_idx: 1,
                preamble: PreambleAction::Take,
                rows_to_skip: 0,
                rows_to_take: 2,
                take_trailer: false,
            },
            ChunkInstructions {
                chunk_idx: 2,
                preamble: PreambleAction::Absent,
                rows_to_skip: 0,
                rows_to_take: 4,
                take_trailer: true,
            },
            ChunkInstructions {
                chunk_idx: 3,
                preamble: PreambleAction::Take,
                rows_to_skip: 0,
                rows_to_take: 1,
                take_trailer: false,
            },
        ];

        // Take all as 1 range
        check(&[0..14], expected_take_all.clone());

        // Take all a individual rows
        check(
            &[
                0..1,
                1..2,
                2..3,
                3..4,
                4..5,
                5..6,
                6..7,
                7..8,
                8..9,
                9..10,
                10..11,
                11..12,
                12..13,
                13..14,
            ],
            expected_take_all,
        );

        // Test some partial takes

        // 2 rows in the same chunk but not contiguous
        check(
            &[0..1, 3..4],
            vec![
                ChunkInstructions {
                    chunk_idx: 0,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 0,
                    rows_to_take: 1,
                    take_trailer: false,
                },
                ChunkInstructions {
                    chunk_idx: 0,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 3,
                    rows_to_take: 1,
                    take_trailer: false,
                },
            ],
        );

        // Taking just a trailer/preamble
        check(
            &[5..6],
            vec![
                ChunkInstructions {
                    chunk_idx: 0,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 5,
                    rows_to_take: 0,
                    take_trailer: true,
                },
                ChunkInstructions {
                    chunk_idx: 1,
                    preamble: PreambleAction::Take,
                    rows_to_skip: 0,
                    rows_to_take: 0,
                    take_trailer: false,
                },
            ],
        );

        // Skipping an entire chunk
        check(
            &[7..10],
            vec![
                ChunkInstructions {
                    chunk_idx: 1,
                    preamble: PreambleAction::Skip,
                    rows_to_skip: 1,
                    rows_to_take: 1,
                    take_trailer: false,
                },
                ChunkInstructions {
                    chunk_idx: 2,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 0,
                    rows_to_take: 2,
                    take_trailer: false,
                },
            ],
        );
    }

    #[test]
    fn test_drain_instructions() {
        fn drain_from_instructions(
            instructions: &mut VecDeque<ChunkInstructions>,
            mut rows_desired: u64,
            need_preamble: &mut bool,
            skip_in_chunk: &mut u64,
        ) -> Vec<ChunkDrainInstructions> {
            // Note: instructions.len() is an upper bound, we typically take much fewer
            let mut drain_instructions = Vec::with_capacity(instructions.len());
            while rows_desired > 0 || *need_preamble {
                let (next_instructions, consumed_chunk) = instructions
                    .front()
                    .unwrap()
                    .drain_from_instruction(&mut rows_desired, need_preamble, skip_in_chunk);
                if consumed_chunk {
                    instructions.pop_front();
                }
                drain_instructions.push(next_instructions);
            }
            drain_instructions
        }

        let repetition_index = vec![vec![5, 2], vec![3, 0], vec![4, 7], vec![2, 0]];
        let repetition_index = MiniBlockRepIndex::decode(&repetition_index);
        let user_ranges = vec![1..7, 10..14];

        // First, schedule the ranges
        let scheduled = ChunkInstructions::schedule_instructions(&repetition_index, &user_ranges);

        let mut to_drain = VecDeque::from(scheduled.clone());

        // Now we drain in batches of 4

        let mut need_preamble = false;
        let mut skip_in_chunk = 0;

        let next_batch =
            drain_from_instructions(&mut to_drain, 4, &mut need_preamble, &mut skip_in_chunk);

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 4);
        assert_eq!(
            next_batch,
            vec![ChunkDrainInstructions {
                chunk_instructions: scheduled[0].clone(),
                rows_to_take: 4,
                rows_to_skip: 0,
                preamble_action: PreambleAction::Absent,
            }]
        );

        let next_batch =
            drain_from_instructions(&mut to_drain, 4, &mut need_preamble, &mut skip_in_chunk);

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 2);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[0].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 4,
                    preamble_action: PreambleAction::Absent,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[1].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[2].clone(),
                    rows_to_take: 2,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Absent,
                }
            ]
        );

        let next_batch =
            drain_from_instructions(&mut to_drain, 2, &mut need_preamble, &mut skip_in_chunk);

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 0);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[2].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 2,
                    preamble_action: PreambleAction::Absent,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[3].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
            ]
        );

        // Regression case.  Need a chunk with preamble, rows, and trailer (the middle chunk here)
        let repetition_index = vec![vec![5, 2], vec![3, 3], vec![20, 0]];
        let repetition_index = MiniBlockRepIndex::decode(&repetition_index);
        let user_ranges = vec![0..28];

        // First, schedule the ranges
        let scheduled = ChunkInstructions::schedule_instructions(&repetition_index, &user_ranges);

        let mut to_drain = VecDeque::from(scheduled.clone());

        // Drain first chunk and some of second chunk

        let mut need_preamble = false;
        let mut skip_in_chunk = 0;

        let next_batch =
            drain_from_instructions(&mut to_drain, 7, &mut need_preamble, &mut skip_in_chunk);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[0].clone(),
                    rows_to_take: 6,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Absent,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[1].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
            ]
        );

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 1);

        // Now, the tricky part.  We drain the second chunk, including the trailer, and need to make sure
        // we get a drain task to take the preamble of the third chunk (and nothing else)
        let next_batch =
            drain_from_instructions(&mut to_drain, 2, &mut need_preamble, &mut skip_in_chunk);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[1].clone(),
                    rows_to_take: 2,
                    rows_to_skip: 1,
                    preamble_action: PreambleAction::Skip,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[2].clone(),
                    rows_to_take: 0,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
            ]
        );

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 0);
    }

    #[tokio::test]
    async fn test_fullzip_repetition_index_caching() {
        use crate::testing::SimulatedScheduler;
        use lance_core::cache::LanceCache;

        // Simplified FixedPerValueDecompressor for testing
        #[derive(Debug)]
        struct TestFixedDecompressor;

        impl FixedPerValueDecompressor for TestFixedDecompressor {
            fn decompress(
                &self,
                _data: FixedWidthDataBlock,
                _num_rows: u64,
            ) -> crate::Result<DataBlock> {
                unimplemented!("Test decompressor")
            }

            fn bits_per_value(&self) -> u64 {
                32
            }
        }

        // Create test repetition index data
        let rows_in_page = 100u64;
        let bytes_per_value = 4u64;
        let _rep_index_size = (rows_in_page + 1) * bytes_per_value;

        // Create mock repetition index data
        let mut rep_index_data = Vec::new();
        for i in 0..=rows_in_page {
            let offset = (i * 100) as u32; // Each row starts at i * 100 bytes
            rep_index_data.extend_from_slice(&offset.to_le_bytes());
        }

        // Simulate storage with the repetition index at position 1000
        let mut full_data = vec![0u8; 1000];
        full_data.extend_from_slice(&rep_index_data);
        full_data.extend_from_slice(&vec![0u8; 10000]); // Add some data after

        let data = bytes::Bytes::from(full_data);
        let io = Arc::new(SimulatedScheduler::new(data));
        let _cache = Arc::new(LanceCache::with_capacity(1024 * 1024));

        // Create FullZipScheduler with repetition index
        let mut scheduler = FullZipScheduler {
            data_buf_position: 0,
            rep_index: Some(FullZipRepIndexDetails {
                buf_position: 1000,
                bytes_per_value,
            }),
            priority: 0,
            rows_in_page,
            bits_per_offset: 32,
            details: Arc::new(FullZipDecodeDetails {
                value_decompressor: PerValueDecompressor::Fixed(Arc::new(TestFixedDecompressor)),
                def_meaning: Arc::new([crate::repdef::DefinitionInterpretation::NullableItem]),
                ctrl_word_parser: crate::repdef::ControlWordParser::new(0, 1),
                max_rep: 0,
                max_visible_def: 0,
            }),
            cached_state: None,
            enable_cache: true, // Enable caching for test
        };

        // First initialization should load and cache the repetition index
        let io_dyn: Arc<dyn crate::EncodingsIo> = io.clone();
        let cached_data1 = scheduler.initialize(&io_dyn).await.unwrap();

        // Verify that we got a FullZipCacheableState (not NoCachedPageData)
        let is_cached = cached_data1
            .clone()
            .as_arc_any()
            .downcast::<FullZipCacheableState>()
            .is_ok();
        assert!(
            is_cached,
            "Expected FullZipCacheableState, got NoCachedPageData"
        );

        // Load the cached data into the scheduler
        scheduler.load(&cached_data1);

        // Verify that cached_state is now populated
        assert!(
            scheduler.cached_state.is_some(),
            "cached_state should be populated after load"
        );

        // Verify the cached data contains the repetition index
        let cached_state = scheduler.cached_state.as_ref().unwrap();

        // Test that schedule_ranges_rep uses the cached data
        let ranges = vec![0..10, 20..30];
        let result = scheduler.schedule_ranges_rep(
            &ranges,
            &io_dyn,
            FullZipRepIndexDetails {
                buf_position: 1000,
                bytes_per_value,
            },
        );

        // The result should be OK (not an error)
        assert!(
            result.is_ok(),
            "schedule_ranges_rep should succeed with cached data"
        );

        // Second scheduler instance should be able to use the cached data
        let mut scheduler2 = FullZipScheduler {
            data_buf_position: 0,
            rep_index: Some(FullZipRepIndexDetails {
                buf_position: 1000,
                bytes_per_value,
            }),
            priority: 0,
            rows_in_page,
            bits_per_offset: 32,
            details: scheduler.details.clone(),
            cached_state: None,
            enable_cache: true, // Enable caching for test
        };

        // Load cached data from the first scheduler
        scheduler2.load(&cached_data1);
        assert!(
            scheduler2.cached_state.is_some(),
            "Second scheduler should have cached_state after load"
        );

        // Verify that both schedulers have the same cached data
        let cached_state2 = scheduler2.cached_state.as_ref().unwrap();
        assert!(
            Arc::ptr_eq(cached_state, cached_state2),
            "Both schedulers should share the same cached data"
        );
    }

    #[tokio::test]
    async fn test_fullzip_cache_config_controls_caching() {
        use crate::testing::SimulatedScheduler;

        // Simplified FixedPerValueDecompressor for testing
        #[derive(Debug)]
        struct TestFixedDecompressor;

        impl FixedPerValueDecompressor for TestFixedDecompressor {
            fn decompress(
                &self,
                _data: FixedWidthDataBlock,
                _num_rows: u64,
            ) -> crate::Result<DataBlock> {
                unimplemented!("Test decompressor")
            }

            fn bits_per_value(&self) -> u64 {
                32
            }
        }

        // Test that enable_cache flag actually controls caching behavior
        let rows_in_page = 1000_u64;
        let bytes_per_value = 4_u64;

        // Create simulated data
        let rep_index_data = vec![0u8; ((rows_in_page + 1) * bytes_per_value) as usize];
        let value_data = vec![0u8; 4000]; // Dummy value data
        let mut full_data = vec![0u8; 1000]; // Padding before rep index
        full_data.extend_from_slice(&rep_index_data);
        full_data.extend_from_slice(&value_data);

        let data = bytes::Bytes::from(full_data);
        let io = Arc::new(SimulatedScheduler::new(data));

        // Test 1: With caching disabled
        let mut scheduler_no_cache = FullZipScheduler {
            data_buf_position: 0,
            rep_index: Some(FullZipRepIndexDetails {
                buf_position: 1000,
                bytes_per_value,
            }),
            priority: 0,
            rows_in_page,
            bits_per_offset: 32,
            details: Arc::new(FullZipDecodeDetails {
                value_decompressor: PerValueDecompressor::Fixed(Arc::new(TestFixedDecompressor)),
                def_meaning: Arc::new([crate::repdef::DefinitionInterpretation::NullableItem]),
                ctrl_word_parser: crate::repdef::ControlWordParser::new(0, 1),
                max_rep: 0,
                max_visible_def: 0,
            }),
            cached_state: None,
            enable_cache: false, // Caching disabled
        };

        let io_dyn: Arc<dyn crate::EncodingsIo> = io.clone();
        let cached_data = scheduler_no_cache.initialize(&io_dyn).await.unwrap();

        // Should return NoCachedPageData when caching is disabled
        assert!(
            cached_data
                .as_arc_any()
                .downcast_ref::<super::NoCachedPageData>()
                .is_some(),
            "With enable_cache=false, should return NoCachedPageData"
        );

        // Test 2: With caching enabled
        let mut scheduler_with_cache = FullZipScheduler {
            data_buf_position: 0,
            rep_index: Some(FullZipRepIndexDetails {
                buf_position: 1000,
                bytes_per_value,
            }),
            priority: 0,
            rows_in_page,
            bits_per_offset: 32,
            details: Arc::new(FullZipDecodeDetails {
                value_decompressor: PerValueDecompressor::Fixed(Arc::new(TestFixedDecompressor)),
                def_meaning: Arc::new([crate::repdef::DefinitionInterpretation::NullableItem]),
                ctrl_word_parser: crate::repdef::ControlWordParser::new(0, 1),
                max_rep: 0,
                max_visible_def: 0,
            }),
            cached_state: None,
            enable_cache: true, // Caching enabled
        };

        let cached_data2 = scheduler_with_cache.initialize(&io_dyn).await.unwrap();

        // Should return FullZipCacheableState when caching is enabled
        assert!(
            cached_data2
                .as_arc_any()
                .downcast_ref::<super::FullZipCacheableState>()
                .is_some(),
            "With enable_cache=true, should return FullZipCacheableState"
        );
    }
}
