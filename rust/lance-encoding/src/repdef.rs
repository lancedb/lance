// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for rep-def levels
//!
//! Repetition and definition levels are a way to encode multipile validity / offsets arrays
//! into a single buffer.  They are a form of "zipping" buffers together that takes advantage
//! of the fact that, if the outermost array is invalid, then the validity of the inner items
//! is irrelevant.
//!
//! Note: the concept of repetition & definition levels comes from the Dremel paper and has
//! been implemented in Apache Parquet.  However, the implementation here is not neccesarily
//! compatible with Parquet.  For example, we use 0 to represent the "inner-most" item and
//! Parquet uses 0 to represent the "outer-most" item.
//!
//! # Repetition Levels
//!
//! With repetition levels we convert a sparse array of offsets into a dense array of levels.
//! These levels are marked non-zero whenever a new list begins.  In other words, given the
//! list array with 3 rows [{<0,1>, <>, <2>}, {<3>}, {}], [], [{<4>}] we would have three
//! offsets arrays:
//!
//! Outer-most ([]): [0, 3, 3, 4]
//! Middle     ({}): [0, 3, 4, 4, 5]
//! Inner      (<>): [0, 2, 2, 3, 4, 5]
//! Values         : [0, 1, 2, 3, 4]
//!
//! We can convert these into repetition levels as follows:
//!
//! | Values | Repetition |
//! | ------ | ---------- |
//! |      0 |          3 | // Start of outer-most list
//! |      1 |          0 | // Continues inner-most list (no new lists)
//! |      - |          1 | // Start of new inner-most list (empty list)
//! |      2 |          1 | // Start of new inner-most list
//! |      3 |          2 | // Start of new middle list
//! |      - |          2 | // Start of new middle list (empty list)
//! |      - |          3 | // Start of new outer-most list (empty list)
//! |      4 |          0 | // Start of new outer-most list
//!
//! Note: We actually have MORE repetition levels than values.  This is because the repetition
//! levels need to be able to represent empty lists.
//!
//! # Definition Levels
//!
//! Definition levels are simpler.  We can think of them as zipping together various validity (from
//! different levels of nesting) into a single buffer.  For example, we could zip the arrays
//! [1, 1, 0, 0] and [1, 0, 1, 0] into [11, 10, 01, 00].  However, 00 and 01 have redundancy in them.
//! If the outer level is null then the validity of the inner levels is irrelevant.  To save space
//! we instead encode a "level" which is the "depth" of the null.  Let's look at a more complete example:
//!
//! Array: [{"middle": {"inner": 1]}}, NULL, {"middle": NULL}, {"middle": {"inner": NULL}}]
//!
//! In Arrow we would have the following validity arrays:
//! Outer validity : 1, 0, 1, 1
//! Middle validity: 1, ?, 0, 1
//! Inner validity : 1, ?, ?, 0
//! Values         : 1, ?, ?, ?
//!
//! The ? values are undefined in the Arrow format.  We can convert these into definition levels as follows:
//!
//! | Values | Definition |
//! | ------ | ---------- |
//! |      1 |          0 | // Valid at all levels
//! |      - |          3 | // Null at outer level
//! |      - |          2 | // Null at middle level
//! |      - |          1 | // Null at inner level
//!
//! # Compression
//!
//! Note that we only need 2 bits of definition levels to represent 3 levels of nesting.  Definition
//! levels are always more compact than the input validity arrays (if there is only 1 level of nesting
//! then they are the same size).
//!
//! Repetition levels are more complex.  If there are very large lists then a sparse array of offsets
//! (which has one element per list) might be more compact than a dense array of repetition levels
//! (which has one element per list value, possibly even more if there are empty lists).
//!
//! However, both repetition levels and definition levels are typically very compressible with RLE.
//!
//! However, in Lance we don't always take advantage of that compression because we want to be able
//! to zip rep-def levels together with our values.  This gives us fewer IOPS when accessing row values.
use std::sync::Arc;

use arrow_array::OffsetSizeTrait;
use arrow_buffer::{bit_util, BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};

use crate::{buffer::LanceBuffer, data::FixedWidthDataBlock};

// We assume 16 bits is good enough for rep-def levels.  This gives us
// 65536 levels of struct nested and 16 levels of list nesting.
pub type LevelBuffer = Vec<u16>;

#[derive(Clone)]
pub enum RawRepDef {
    Offsets(Arc<[i64]>),
    Validity(BooleanBuffer),
    AllNull(),
    NoNull(),
}

#[derive(Debug)]
pub struct SerializedRepDefs {
    pub repetition_levels: LevelBuffer,
    pub definition_levels: LevelBuffer,
    // Repetition levels which are all null
    pub all_null: Vec<u32>,
    // Repetition levels which are all valid
    pub all_valid: Vec<u32>,
}

impl SerializedRepDefs {
    fn empty() -> Self {
        SerializedRepDefs {
            repetition_levels: Vec::new(),
            definition_levels: Vec::new(),
            all_null: Vec::new(),
            all_valid: Vec::new(),
        }
    }
}

pub struct SerializerContext {
    last_offsets: Option<Arc<[i64]>>,
    rep_levels: LevelBuffer,
    def_levels: LevelBuffer,
    current_rep: u16,
    current_def: u16,
}

impl SerializerContext {
    fn new_from_validity(validity: &BooleanBuffer) -> Self {
        let len = validity.len();
        let mut ctx = Self {
            last_offsets: None,
            rep_levels: vec![0; len],
            def_levels: vec![0; len],
            current_rep: 1,
            current_def: 2,
        };
        ctx.def_levels
            .iter_mut()
            .zip(validity.iter())
            .for_each(|(def, valid)| {
                // 0 = inner-most is valid
                // 1 = inner-most is null
                if !valid {
                    *def = 1;
                }
            });
        ctx
    }

    fn new_from_off(last_offsets: Arc<[i64]>) -> Self {
        let len = last_offsets[last_offsets.len() - 1] as usize;
        Self {
            last_offsets: Some(last_offsets),
            rep_levels: vec![0; len],
            def_levels: vec![0; len],
            current_rep: 1,
            current_def: 1,
        }
    }

    fn concat_def(&mut self, validity: &BooleanBuffer) {
        if let Some(last_offsets) = &self.last_offsets {
            last_offsets
                .windows(2)
                .zip(validity.iter())
                .for_each(|(w, valid)| {
                    if !valid {
                        self.def_levels[w[0] as usize..w[1] as usize].fill(self.current_def);
                    }
                });
        } else {
            self.def_levels
                .iter_mut()
                .zip(validity.iter())
                .for_each(|(def, valid)| {
                    if !valid {
                        *def = self.current_def;
                    }
                });
        }
    }

    fn build(self, all_null: Vec<u32>, all_valid: Vec<u32>) -> SerializedRepDefs {
        SerializedRepDefs {
            definition_levels: self.def_levels,
            repetition_levels: self.rep_levels,
            all_null,
            all_valid,
        }
    }
}

#[derive(Clone, Default)]
pub struct RepDefBuilder {
    repdefs: Vec<RawRepDef>,
    len: Option<usize>,
}

impl RepDefBuilder {
    fn check_validity_len(&mut self, validity: &NullBuffer) {
        if let Some(len) = self.len {
            assert!(validity.len() == len);
        }
        self.len = Some(validity.len());
    }

    pub fn add_validity_bitmap(&mut self, validity: NullBuffer) {
        self.check_validity_len(&validity);
        self.repdefs
            .push(RawRepDef::Validity(validity.into_inner()));
    }

    pub fn add_all_null(&mut self) {
        self.repdefs.push(RawRepDef::AllNull());
    }

    pub fn add_no_null(&mut self) {
        self.repdefs.push(RawRepDef::NoNull());
    }

    fn check_offset_len(&mut self, offsets: &Vec<i64>) {
        if let Some(len) = self.len {
            assert!(offsets.len() == len + 1);
        }
        self.len = Some(offsets[offsets.len() - 1] as usize);
    }

    pub fn add_offsets<O: OffsetSizeTrait>(&mut self, repetition: OffsetBuffer<O>) {
        // We should be able to zero-copy
        if O::IS_LARGE {
            let inner = repetition.into_inner();
            let len = inner.len();
            let i64_buff = ScalarBuffer::new(inner.into_inner(), 0, len);
            let offsets = Vec::from(i64_buff);
            self.check_offset_len(&offsets);
            self.repdefs.push(RawRepDef::Offsets(offsets.into()));
        } else {
            let inner = repetition.into_inner();
            let len = inner.len();
            let casted = ScalarBuffer::<i32>::new(inner.into_inner(), 0, len)
                .iter()
                .copied()
                .map(|o| o as i64)
                .collect();
            self.check_offset_len(&casted);
            self.repdefs.push(RawRepDef::Offsets(casted.into()));
        }
    }

    fn offsets_to_rep_no_ctx(offsets: &Arc<[i64]>) -> SerializerContext {
        let mut ctx = SerializerContext::new_from_off(offsets.clone());
        // All offsets except last get put into levels buffer (last is implicitly encoded as buffer len)
        for off in offsets[..offsets.len() - 1].iter() {
            ctx.rep_levels[*off as usize] = 1;
        }
        ctx.current_rep += 1;
        ctx
    }

    fn offsets_to_rep_with_ctx(
        offsets: &Arc<[i64]>,
        mut ctx: SerializerContext,
    ) -> SerializerContext {
        if let Some(last_offsets) = &ctx.last_offsets {
            let mut new_last_off = Vec::with_capacity(offsets.len());
            for off in offsets[..offsets.len() - 1].iter() {
                let offset_ctx = last_offsets[*off as usize];
                new_last_off.push(offset_ctx);
                ctx.rep_levels[offset_ctx as usize] = ctx.current_rep;
            }
            ctx.last_offsets = Some(new_last_off.into());
        } else {
            for off in offsets[..offsets.len() - 1].iter() {
                ctx.rep_levels[*off as usize] = ctx.current_rep;
            }
            ctx.last_offsets = Some(offsets.clone());
        }
        ctx.current_rep += 1;
        ctx
    }

    fn offsets_to_rep(offsets: &Arc<[i64]>, ctx: Option<SerializerContext>) -> SerializerContext {
        match ctx {
            None => Self::offsets_to_rep_no_ctx(offsets),
            Some(ctx) => Self::offsets_to_rep_with_ctx(offsets, ctx),
        }
    }

    fn validity_to_def(
        validity: &BooleanBuffer,
        ctx: Option<SerializerContext>,
    ) -> SerializerContext {
        match ctx {
            None => SerializerContext::new_from_validity(validity),
            Some(mut ctx) => {
                ctx.concat_def(validity);
                ctx.current_def += 1;
                ctx
            }
        }
    }

    pub fn serialize(&self) -> SerializedRepDefs {
        // Add the deepest as-is
        let mut ctx = None;

        let mut all_null = Vec::new();
        let mut all_valid = Vec::new();

        for (idx, rep_def) in self.repdefs.iter().enumerate().rev() {
            match rep_def {
                RawRepDef::Validity(def) => {
                    ctx = Some(Self::validity_to_def(def, ctx));
                }
                RawRepDef::Offsets(rep) => {
                    ctx = Some(Self::offsets_to_rep(rep, ctx));
                }
                RawRepDef::AllNull() => {
                    all_null.push(idx as u32);
                }
                RawRepDef::NoNull() => {
                    all_valid.push(idx as u32);
                }
            }
        }

        ctx.map(|ctx| ctx.build(all_null, all_valid))
            .unwrap_or_else(|| SerializedRepDefs::empty())
    }
}

pub struct MiniBlock {
    data: LanceBuffer,
    bits_per_val: Vec<usize>,
    rows_per_block: usize,
    bytes_per_block: usize,
    num_values: usize,
}

impl MiniBlock {
    pub fn borrow_and_clone(&mut self) -> Self {
        Self {
            data: self.data.borrow_and_clone(),
            bits_per_val: self.bits_per_val.clone(),
            rows_per_block: self.rows_per_block,
            bytes_per_block: self.bytes_per_block,
            num_values: self.num_values,
        }
    }

    pub fn maybe_serialize(parts: Vec<FixedWidthDataBlock>) -> Option<Self> {
        assert!(!parts.is_empty());
        let num_values = parts[0].num_values as usize;
        let bits_per_val = parts
            .iter()
            .map(|p| p.bits_per_value as usize)
            .collect::<Vec<_>>();
        let bits_per_row = bits_per_val.iter().sum::<usize>();
        // There is a -1 here because we always have one extra row of padding to avoid any single
        // value being split across two mini blocks and so the last partial value in each buffer is
        // duplicated in the next buffer.  E.g. if we have 4 bits per value and a buffer ends with
        // a byte AABBBBCC then the next buffer will start with a byte AABBBBCC and an (implicit)
        // bit offset of 6.
        //
        // TODO: We might be able to be more aggressive here.  It seems we shouldn't ever need more
        // than 1 byte (per part) of padding per mini-block instead of 1 row of padding per mini-block.
        // In practice, the difference is probably 1-2 bytes per mini-block of wasted padding.
        let rows_per_block = ((MINI_BLOCK_SIZE - parts.len()) * 8 / bits_per_row) - 1;
        if rows_per_block < MIN_ROWS_PER_MINI_BLOCK {
            return None;
        }

        let num_blocks = bit_util::ceil(num_values, rows_per_block);
        let mut buf = vec![0; num_blocks * MINI_BLOCK_SIZE];

        // This will be updated to represent the number of bytes (per block) occupied
        // by all previous parts
        let mut part_offset = 0;
        for part in parts.into_iter() {
            let part_bit_stride = part.bits_per_value as usize * rows_per_block;
            let part_byte_stride = bit_util::ceil(part_bit_stride, 8);
            let part_extra_bytes = bit_util::ceil(part.bits_per_value as usize, 8);

            // Pointer into where we are writing to
            let mut buf_pointer = part_offset;
            // How many values we've processed so far
            let mut values_read = 0;
            // Pointer into the part buffer we are reading from
            let mut part_buf_pointer = 0;
            while values_read < num_values {
                let num_parts_in_block = rows_per_block.min(num_values - values_read);
                let (num_bytes_to_copy, start_offset) = if values_read == 0 {
                    (part_byte_stride, part_extra_bytes)
                } else {
                    (part_byte_stride + part_extra_bytes, 0)
                };
                let num_bytes_to_copy = num_bytes_to_copy.min(part.data.len() - part_buf_pointer);
                buf[(buf_pointer + start_offset)..(buf_pointer + start_offset + num_bytes_to_copy)]
                    .copy_from_slice(
                        &part.data[part_buf_pointer..part_buf_pointer + num_bytes_to_copy],
                    );
                values_read += num_parts_in_block;
                buf_pointer += MINI_BLOCK_SIZE;
                part_buf_pointer += num_bytes_to_copy - part_extra_bytes;
            }

            part_offset += part_byte_stride + part_extra_bytes;
        }

        let data = LanceBuffer::Owned(buf);

        Some(Self {
            data,
            bits_per_val,
            rows_per_block: rows_per_block,
            bytes_per_block: MINI_BLOCK_SIZE,
            num_values,
        })
    }

    pub fn deserialize(self) -> Vec<FixedWidthDataBlock> {
        let num_blocks = self.data.len() / self.bytes_per_block;
        let mut parts = Vec::with_capacity(self.bits_per_val.len());

        let mut part_offset = 0;
        for bits_per_val in self.bits_per_val.iter() {
            let num_bytes = bit_util::ceil(self.num_values as usize * bits_per_val, 8) as usize;
            let mut part_data = Vec::with_capacity(num_bytes);

            let part_bit_stride = bits_per_val * self.rows_per_block;
            let part_byte_stride = bit_util::ceil(part_bit_stride, 8);
            let part_extra_bytes = bit_util::ceil(*bits_per_val, 8);

            let mut bytes_remaining = num_bytes;
            for block_index in 0..num_blocks {
                let block_start = block_index * self.bytes_per_block;
                let block_end = block_start + self.bytes_per_block;
                let block = &self.data[block_start..block_end];

                let part_start = part_offset + part_extra_bytes;
                let bytes_in_part = part_byte_stride.min(bytes_remaining);
                let part_end = part_start + bytes_in_part;

                bytes_remaining -= bytes_in_part;

                part_data.extend_from_slice(&block[part_start..part_end]);
            }

            let part_data = LanceBuffer::Owned(part_data);
            parts.push(FixedWidthDataBlock {
                bits_per_value: *bits_per_val as u64,
                data: part_data,
                num_values: self.num_values as u64,
            });

            part_offset += part_byte_stride + part_extra_bytes;
        }

        parts
    }
}

// These settings give mini blocks that are disk-sector size and have at least 256 rows
//
// If the data values are too wide then we fall back to full-zip
// TODO: Experiment with these values
const MINI_BLOCK_SIZE: usize = 4096;
const MIN_ROWS_PER_MINI_BLOCK: usize = 64;

#[cfg(test)]
mod tests {
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};

    use crate::{
        buffer::LanceBuffer,
        data::FixedWidthDataBlock,
        repdef::{MiniBlock, MINI_BLOCK_SIZE},
    };

    use super::RepDefBuilder;

    fn validity(values: &[bool]) -> NullBuffer {
        NullBuffer::from_iter(values.iter().copied())
    }

    fn offsets(values: &[i64]) -> OffsetBuffer<i64> {
        OffsetBuffer::<i64>::new(ScalarBuffer::from_iter(values.iter().copied()))
    }

    #[test]
    fn test_repdef() {
        let mut builder = RepDefBuilder::default();
        builder.add_validity_bitmap(validity(&[true, false, true]));
        builder.add_offsets(offsets(&[0, 2, 3, 5]));
        builder.add_validity_bitmap(validity(&[true, true, true, false, true]));
        builder.add_offsets(offsets(&[0, 1, 3, 5, 7, 9]));
        builder.add_validity_bitmap(validity(&[
            true, true, true, false, false, false, true, true, false,
        ]));

        let repdefs = builder.serialize();

        assert_eq!(vec![0, 0, 0, 3, 3, 2, 2, 0, 1], repdefs.definition_levels);
        assert_eq!(vec![2, 1, 0, 2, 0, 2, 0, 1, 0], repdefs.repetition_levels);
    }

    #[test]
    fn test_special_validity() {
        let mut builder = RepDefBuilder::default();
        builder.add_validity_bitmap(validity(&[true, false, true]));
        builder.add_all_null();
        builder.add_validity_bitmap(validity(&[true, true, false]));
        builder.add_no_null();

        let repdefs = builder.serialize();

        assert_eq!(vec![0, 2, 1], repdefs.definition_levels);
        assert_eq!(vec![1], repdefs.all_null);
        assert_eq!(vec![3], repdefs.all_valid);
    }

    #[test]
    fn test_zip_mini_block() {
        // 2 bits per value of rep, 5 values = 10 bits = 2 bytes
        // 1001 0011 10XX
        let rep_levels = LanceBuffer::copy_array([0x93, 0x80]);
        let mut rep_levels = FixedWidthDataBlock {
            bits_per_value: 2,
            data: rep_levels,
            num_values: 5,
        };
        // 4 bits per value of def, 5 values = 20 bits = 3 bytes
        let def_levels = LanceBuffer::copy_array([0xFC, 0xA3, 0x10]);
        let mut def_levels = FixedWidthDataBlock {
            bits_per_value: 4,
            data: def_levels,
            num_values: 5,
        };
        // 12 bits of value, 5 values, 60 bites, 8 bytes
        let values = LanceBuffer::copy_array([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]);
        let mut values = FixedWidthDataBlock {
            bits_per_value: 12,
            data: values,
            num_values: 5,
        };

        let parts = vec![
            rep_levels.borrow_and_clone(),
            def_levels.borrow_and_clone(),
            values.borrow_and_clone(),
        ];

        let miniblock = MiniBlock::maybe_serialize(parts).unwrap();

        assert_eq!(miniblock.data.len(), 4096);
        assert_eq!(miniblock.rows_per_block, 1818);
        assert_eq!(miniblock.bytes_per_block, MINI_BLOCK_SIZE);

        let unzipped = miniblock.deserialize();

        assert_eq!(unzipped.len(), 3);
        assert_eq!(unzipped[0], rep_levels);
        assert_eq!(unzipped[1], def_levels);
        assert_eq!(unzipped[2], values);

        // More complicated example where we have 3 mini blocks worth of data
        // Same rules, 2 bits of rep, 4 bits of def, 12 bits of value so 18 bits per
        // row or 1819 values per block.  With 5000 values that gives us:
        //
        // Block 1 - 1819 values (114 bytes rep, )
        // Block 2 - 1819 values
        // Block 3 - 1362 values

        // 5000 values @ 2 bits per value = 10000 bits = 1250 bytes
        let rep_levels = LanceBuffer::Owned(vec![0x93; 5000 / 4]);
        // 5000 values @ 4 bits per value = 20000 bits = 2500 bytes
        let def_levels = LanceBuffer::Owned(vec![0xFC; 5000 / 2]);
        // 5000 values @ 12 bits per value = 60000 bits = 7500 bytes
        let values = LanceBuffer::Owned(vec![0x12; 5000 * 3 / 2]);

        let mut rep_levels = FixedWidthDataBlock {
            bits_per_value: 2,
            data: rep_levels,
            num_values: 5000,
        };
        let mut def_levels = FixedWidthDataBlock {
            bits_per_value: 4,
            data: def_levels,
            num_values: 5000,
        };
        let mut values = FixedWidthDataBlock {
            bits_per_value: 12,
            data: values,
            num_values: 5000,
        };

        let parts = vec![
            rep_levels.borrow_and_clone(),
            def_levels.borrow_and_clone(),
            values.borrow_and_clone(),
        ];
        let miniblock = MiniBlock::maybe_serialize(parts).unwrap();

        assert_eq!(miniblock.data.len(), 4096 * 3);
        assert_eq!(miniblock.rows_per_block, 1818);
        assert_eq!(miniblock.bytes_per_block, MINI_BLOCK_SIZE);

        let unzipped = miniblock.deserialize();

        assert_eq!(unzipped.len(), 3);
        assert_eq!(unzipped[0], rep_levels);
        assert_eq!(unzipped[1], def_levels);
        assert_eq!(unzipped[2], values);
    }
}
