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
//! |      - |          2 | // Start of new inner-most list (empty list)
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
//! [1, 1, 0, 0] and [1, 0, 1, 0] into [11, 10, 01, 00].  However, 00 and 01 are redundant.  If the
//! outer level is null then the validity of the inner levels is irrelevant.  To save space we instead
//! encode a "level" which is the "depth" of the null.  Let's look at a more complete example:
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
//! levels are always more compact than the input validity arrays.
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
use arrow_buffer::{ArrowNativeType, BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use lance_core::{Error, Result};
use snafu::{location, Location};

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

    pub fn len(&self) -> usize {
        self.definition_levels.len()
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

    pub fn has_nulls(&self) -> bool {
        self.repdefs
            .iter()
            .any(|rd| matches!(rd, RawRepDef::AllNull() | RawRepDef::Validity(_)))
    }

    pub fn add_validity_bitmap(&mut self, validity: NullBuffer) {
        self.check_validity_len(&validity);
        self.repdefs
            .push(RawRepDef::Validity(validity.into_inner()));
    }

    // TODO: We should be using RawRepDef::AllNull / RawRepDef::NoNull but downstream
    // can't handle it yet
    pub fn add_all_null(&mut self, len: usize) {
        self.repdefs
            .push(RawRepDef::Validity(BooleanBuffer::new_unset(len)))
        // self.repdefs.push(RawRepDef::AllNull());
    }

    pub fn add_no_null(&mut self, len: usize) {
        self.repdefs
            .push(RawRepDef::Validity(BooleanBuffer::new_set(len)))
        // self.repdefs.push(RawRepDef::NoNull());
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

#[derive(Debug)]
enum UnravelerValidity {
    AllValid,
    DefLevels(LevelBuffer),
}

#[derive(Debug)]
pub struct RepDefUnraveler {
    rep_levels: LevelBuffer,
    def_levels: UnravelerValidity,
    // Current definition level to compare to.
    current_def_cmp: u16,
}

impl RepDefUnraveler {
    pub fn new_with_def(rep_levels: LevelBuffer, def_levels: LevelBuffer) -> Self {
        Self {
            rep_levels,
            def_levels: UnravelerValidity::DefLevels(def_levels),
            current_def_cmp: 0,
        }
    }

    pub fn new_all_valid(rep_levels: LevelBuffer) -> Self {
        Self {
            rep_levels,
            def_levels: UnravelerValidity::AllValid,
            current_def_cmp: 0,
        }
    }

    pub fn unravel_offsets<T: ArrowNativeType>(&mut self) -> Result<OffsetBuffer<T>> {
        let mut offsets: Vec<T> = Vec::with_capacity(self.rep_levels.len() + 1);
        let mut curlen: usize = 0;
        let to_offset = |val: usize| {
            T::from_usize(val)
            .ok_or_else(|| Error::invalid_input("A single batch had more than i32::MAX values and so a large container type is required", location!()))
        };
        if let UnravelerValidity::DefLevels(def_levels) = &mut self.def_levels {
            assert!(self.rep_levels.len() == def_levels.len());
            // This is a strange access pattern.  We are iterating over the rep/def levels and
            // at the same time writing the rep/def levels.  This means we need both a mutable
            // and immutable reference to the rep/def levels.
            //
            // SAFETY: We can cheat mutability here because the write will always be to values
            //         we have already read in the iteration
            // SAFETY: We are doing our own bounds checking by asserting the lens are the same
            unsafe {
                let mut rep_read_iter = self.rep_levels.as_mut_ptr();
                let mut def_read_iter = def_levels.as_mut_ptr();

                let mut rep_write_iter = self.rep_levels.as_mut_ptr();
                let mut def_write_iter = def_levels.as_mut_ptr();

                let rep_end = rep_read_iter.add(self.rep_levels.len());
                while rep_read_iter != rep_end {
                    if *rep_read_iter != 0 {
                        // Finish the current list
                        offsets.push(to_offset(curlen)?);
                        *rep_write_iter = *rep_read_iter - 1;
                        *def_write_iter = *def_read_iter;
                        rep_write_iter = rep_write_iter.add(1);
                        def_write_iter = def_write_iter.add(1);
                    }
                    curlen += 1;
                    rep_read_iter = rep_read_iter.add(1);
                    def_read_iter = def_read_iter.add(1);
                }
            }
            offsets.push(to_offset(curlen)?);
            self.rep_levels.truncate(offsets.len() - 1);
            def_levels.truncate(offsets.len() - 1);
            Ok(OffsetBuffer::new(ScalarBuffer::from(offsets)))
        } else {
            // SAFETY: See above loop
            unsafe {
                let mut rep_read_iter = self.rep_levels.as_mut_ptr();
                let mut rep_write_iter = self.rep_levels.as_mut_ptr();

                let rep_end = rep_read_iter.add(self.rep_levels.len());
                while rep_read_iter != rep_end {
                    if *rep_read_iter != 0 {
                        // Finish the current list
                        offsets.push(to_offset(curlen)?);
                        *rep_write_iter = *rep_read_iter - 1;
                        rep_write_iter = rep_write_iter.add(1);
                    }
                    curlen += 1;
                    rep_read_iter = rep_read_iter.add(1);
                }
            }
            offsets.push(to_offset(curlen)?);
            self.rep_levels.truncate(offsets.len() - 1);
            Ok(OffsetBuffer::new(ScalarBuffer::from(offsets)))
        }
    }

    pub fn unravel_validity(&mut self) -> Option<NullBuffer> {
        let def_levels = match &self.def_levels {
            UnravelerValidity::DefLevels(def_levels) => def_levels,
            UnravelerValidity::AllValid => return None,
        };
        let current_def_cmp = self.current_def_cmp;
        self.current_def_cmp += 1;
        let validity = BooleanBuffer::from_iter(def_levels.iter().map(|&r| r <= current_def_cmp));
        if validity.count_set_bits() == validity.len() {
            None
        } else {
            Some(NullBuffer::new(validity))
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};

    use crate::repdef::RepDefUnraveler;

    use super::RepDefBuilder;

    fn validity(values: &[bool]) -> NullBuffer {
        NullBuffer::from_iter(values.iter().copied())
    }

    fn offsets_32(values: &[i32]) -> OffsetBuffer<i32> {
        OffsetBuffer::<i32>::new(ScalarBuffer::from_iter(values.iter().copied()))
    }

    fn offsets_64(values: &[i64]) -> OffsetBuffer<i64> {
        OffsetBuffer::<i64>::new(ScalarBuffer::from_iter(values.iter().copied()))
    }

    #[test]
    fn test_repdef() {
        let mut builder = RepDefBuilder::default();
        builder.add_validity_bitmap(validity(&[true, false, true]));
        builder.add_offsets(offsets_64(&[0, 2, 3, 5]));
        builder.add_validity_bitmap(validity(&[true, true, true, false, true]));
        builder.add_offsets(offsets_64(&[0, 1, 3, 5, 7, 9]));
        builder.add_validity_bitmap(validity(&[
            true, true, true, false, false, false, true, true, false,
        ]));

        let repdefs = builder.serialize();

        assert_eq!(vec![0, 0, 0, 3, 3, 2, 2, 0, 1], repdefs.definition_levels);
        assert_eq!(vec![2, 1, 0, 2, 0, 2, 0, 1, 0], repdefs.repetition_levels);

        let mut unraveler =
            RepDefUnraveler::new_with_def(repdefs.repetition_levels, repdefs.definition_levels);

        // Note: validity doesn't exactly round-trip because repdef normalizes some of the
        // redundant validity values
        assert_eq!(
            unraveler.unravel_validity(),
            Some(validity(&[
                true, true, true, false, false, false, false, true, false
            ]))
        );
        assert_eq!(
            unraveler.unravel_offsets::<i32>().unwrap().inner(),
            offsets_32(&[0, 1, 3, 5, 7, 9]).inner()
        );
        assert_eq!(
            unraveler.unravel_validity(),
            Some(validity(&[true, true, false, false, true]))
        );
        assert_eq!(
            unraveler.unravel_offsets::<i32>().unwrap().inner(),
            offsets_32(&[0, 2, 3, 5]).inner()
        );
        assert_eq!(
            unraveler.unravel_validity(),
            Some(validity(&[true, false, true]))
        );
    }

    // Disabled until we can handle special validity downstream
    // #[test]
    // fn test_special_validity() {
    //     let mut builder = RepDefBuilder::default();
    //     builder.add_validity_bitmap(validity(&[true, false, true]));
    //     builder.add_all_null();
    //     builder.add_validity_bitmap(validity(&[true, true, false]));
    //     builder.add_no_null();

    //     let repdefs = builder.serialize();

    //     assert_eq!(vec![0, 2, 1], repdefs.definition_levels);
    //     assert_eq!(vec![1], repdefs.all_null);
    //     assert_eq!(vec![3], repdefs.all_valid);
    // }
}
