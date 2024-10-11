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

// TODO: Right now, if a layer has no nulls, but other layers do, then we still use
//       up a repetition layer for the no-null spot.  For example, if we have four
//       levels of rep: [has nulls, has nulls, no nulls, has nulls] then we will say:
//       0 = valid
//       1 = layer 4 null
//       2 = layer 3 null
//       3 = layer 2 null (useless)
//       4 = layer 1 null
//
// This means we end up with 3 bits per level instead of 2.  We could instead record
// the layers that are all null somewhere else and not require wider rep levels.

use std::sync::Arc;

use arrow_array::OffsetSizeTrait;
use arrow_buffer::{
    ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, NullBuffer, OffsetBuffer, ScalarBuffer,
};
use lance_core::{Error, Result};
use snafu::{location, Location};

// We assume 16 bits is good enough for rep-def levels.  This gives us
// 65536 levels of struct nested and 16 levels of list nesting.
pub type LevelBuffer = Vec<u16>;

#[derive(Clone, Debug)]
enum RawRepDef {
    Offsets(Arc<[i64]>),
    Validity(BooleanBuffer),
    NoNull(usize),
}

#[derive(Debug)]
pub struct SerializedRepDefs {
    // If None, there are no lists
    pub repetition_levels: Option<LevelBuffer>,
    // If None, there are no nulls
    pub definition_levels: Option<LevelBuffer>,
}

impl SerializedRepDefs {
    fn empty() -> Self {
        Self {
            repetition_levels: None,
            definition_levels: None,
        }
    }
}

struct SerializerContext {
    last_offsets: Option<Arc<[i64]>>,
    rep_levels: LevelBuffer,
    def_levels: LevelBuffer,
    current_rep: u16,
    current_def: u16,
    has_nulls: bool,
}

impl SerializerContext {
    fn new(len: usize, has_nulls: bool) -> Self {
        Self {
            last_offsets: None,
            rep_levels: LevelBuffer::with_capacity(len),
            def_levels: if has_nulls {
                LevelBuffer::with_capacity(len)
            } else {
                LevelBuffer::default()
            },
            current_rep: 1,
            current_def: 1,
            has_nulls: false,
        }
    }

    fn record_all_valid(&mut self, len: usize) {
        self.current_def += 1;
        if self.def_levels.is_empty() {
            self.def_levels.resize(len, 0);
        }
    }

    fn record_offsets(&mut self, offsets: &Arc<[i64]>) {
        let rep_level = self.current_rep;
        self.current_rep += 1;
        if let Some(last_offsets) = &self.last_offsets {
            let mut new_last_off = Vec::with_capacity(offsets.len());
            for off in offsets[..offsets.len() - 1].iter() {
                let offset_ctx = last_offsets[*off as usize];
                new_last_off.push(offset_ctx);
                self.rep_levels[offset_ctx as usize] = rep_level;
            }
            self.last_offsets = Some(new_last_off.into());
        } else {
            self.rep_levels.resize(*offsets.last().unwrap() as usize, 0);
            for off in offsets[..offsets.len() - 1].iter() {
                self.rep_levels[*off as usize] = rep_level;
            }
            self.last_offsets = Some(offsets.clone());
        }
    }

    fn record_validity(&mut self, validity: &BooleanBuffer) {
        self.has_nulls = true;
        let def_level = self.current_def;
        self.current_def += 1;
        if self.def_levels.is_empty() {
            self.def_levels.resize(validity.len(), 0);
        }
        if let Some(last_offsets) = &self.last_offsets {
            last_offsets
                .windows(2)
                .zip(validity.iter())
                .for_each(|(w, valid)| {
                    if !valid {
                        self.def_levels[w[0] as usize..w[1] as usize].fill(def_level);
                    }
                });
        } else {
            self.def_levels
                .iter_mut()
                .zip(validity.iter())
                .for_each(|(def, valid)| {
                    if !valid {
                        *def = def_level;
                    }
                });
        }
    }

    fn build(self) -> SerializedRepDefs {
        SerializedRepDefs {
            definition_levels: if self.has_nulls {
                Some(self.def_levels)
            } else {
                None
            },
            repetition_levels: if self.current_rep > 1 {
                Some(self.rep_levels)
            } else {
                None
            },
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

    fn num_layers(&self) -> usize {
        self.repdefs.len()
    }

    fn is_empty(&self) -> bool {
        self.repdefs
            .iter()
            .all(|r| matches!(r, RawRepDef::NoNull(_)))
    }

    pub fn has_nulls(&self) -> bool {
        self.repdefs
            .iter()
            .any(|rd| matches!(rd, RawRepDef::Validity(_)))
    }

    pub fn add_validity_bitmap(&mut self, validity: NullBuffer) {
        self.check_validity_len(&validity);
        self.repdefs
            .push(RawRepDef::Validity(validity.into_inner()));
    }

    pub fn add_all_null(&mut self, len: usize) {
        self.repdefs
            .push(RawRepDef::Validity(BooleanBuffer::new_unset(len)))
    }

    pub fn add_no_null(&mut self, len: usize) {
        self.repdefs.push(RawRepDef::NoNull(len));
    }

    fn check_offset_len(&mut self, offsets: &[i64]) {
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
                .collect::<Vec<_>>();
            self.check_offset_len(&casted);
            self.repdefs.push(RawRepDef::Offsets(casted.into()));
        }
    }

    // TODO: This is lazy.  We shouldn't need this concatenation pass.  We should be able
    // to concatenate as we build up the rep/def levels but I'm saving that for a
    // future optimization.
    fn concat_layers<'a>(mut layers: impl Iterator<Item = &'a RawRepDef>, len: usize) -> RawRepDef {
        let first = layers.next().unwrap();
        match &first {
            RawRepDef::NoNull(_) | RawRepDef::Validity(_) => {
                // Also lazy, building up a validity buffer just to throw it away
                // if there are no nulls
                let mut has_nulls = false;
                let mut builder = BooleanBufferBuilder::new(len);
                for layer in std::iter::once(first).chain(layers) {
                    match layer {
                        RawRepDef::NoNull(num_valid) => {
                            builder.append_n(*num_valid, true);
                        }
                        RawRepDef::Validity(validity) => {
                            has_nulls = true;
                            builder.append_buffer(validity);
                        }
                        _ => unreachable!(),
                    }
                }
                if has_nulls {
                    RawRepDef::Validity(builder.finish())
                } else {
                    RawRepDef::NoNull(builder.len())
                }
            }
            RawRepDef::Offsets(offsets) => {
                let mut all_offsets = Vec::with_capacity(len);
                all_offsets.extend(offsets.iter().copied());
                for layer in layers {
                    let last = *all_offsets.last().unwrap();
                    let RawRepDef::Offsets(offsets) = layer else {
                        unreachable!()
                    };
                    all_offsets.extend(offsets.iter().skip(1).map(|off| *off + last));
                }
                RawRepDef::Offsets(all_offsets.into())
            }
        }
    }

    pub fn serialize(builders: Vec<Self>) -> SerializedRepDefs {
        if builders.is_empty() {
            return SerializedRepDefs::empty();
        }
        if builders.iter().all(|b| b.is_empty()) {
            // No repetition, all-valid
            return SerializedRepDefs::empty();
        }
        let has_nulls = builders.iter().any(|b| b.has_nulls());
        let total_len = builders.iter().map(|b| b.len.unwrap()).sum();
        let mut context = SerializerContext::new(total_len, has_nulls);
        debug_assert!(builders
            .iter()
            .all(|b| b.num_layers() == builders[0].num_layers()));
        for layer_index in (0..builders[0].num_layers()).rev() {
            let layer =
                Self::concat_layers(builders.iter().map(|b| &b.repdefs[layer_index]), total_len);
            match layer {
                RawRepDef::Validity(def) => {
                    context.record_validity(&def);
                }
                RawRepDef::Offsets(rep) => {
                    context.record_offsets(&rep);
                }
                RawRepDef::NoNull(len) => {
                    context.record_all_valid(len);
                }
            }
        }
        context.build()
    }
}

#[derive(Debug)]
pub struct RepDefUnraveler {
    rep_levels: Option<LevelBuffer>,
    def_levels: Option<LevelBuffer>,
    // Current definition level to compare to.
    current_def_cmp: u16,
}

impl RepDefUnraveler {
    pub fn new(rep_levels: Option<LevelBuffer>, def_levels: Option<LevelBuffer>) -> Self {
        Self {
            rep_levels,
            def_levels,
            current_def_cmp: 0,
        }
    }

    pub fn unravel_offsets<T: ArrowNativeType>(&mut self) -> Result<OffsetBuffer<T>> {
        let rep_levels = self
            .rep_levels
            .as_mut()
            .expect("Expected repetition level but data didn't contain repetition");
        let mut offsets: Vec<T> = Vec::with_capacity(rep_levels.len() + 1);
        let mut curlen: usize = 0;
        let to_offset = |val: usize| {
            T::from_usize(val)
            .ok_or_else(|| Error::invalid_input("A single batch had more than i32::MAX values and so a large container type is required", location!()))
        };
        if let Some(def_levels) = &mut self.def_levels {
            assert!(rep_levels.len() == def_levels.len());
            // This is a strange access pattern.  We are iterating over the rep/def levels and
            // at the same time writing the rep/def levels.  This means we need both a mutable
            // and immutable reference to the rep/def levels.
            //
            // SAFETY: We can cheat mutability here because the write will always be to values
            //         we have already read in the iteration
            // SAFETY: We are doing our own bounds checking by asserting the lens are the same
            unsafe {
                let mut rep_read_iter = rep_levels.as_mut_ptr();
                let mut def_read_iter = def_levels.as_mut_ptr();

                let mut rep_write_iter = rep_levels.as_mut_ptr();
                let mut def_write_iter = def_levels.as_mut_ptr();

                let rep_end = rep_read_iter.add(rep_levels.len());
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
            rep_levels.truncate(offsets.len() - 1);
            def_levels.truncate(offsets.len() - 1);
            Ok(OffsetBuffer::new(ScalarBuffer::from(offsets)))
        } else {
            // SAFETY: See above loop
            unsafe {
                let mut rep_read_iter = rep_levels.as_mut_ptr();
                let mut rep_write_iter = rep_levels.as_mut_ptr();

                let rep_end = rep_read_iter.add(rep_levels.len());
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
            rep_levels.truncate(offsets.len() - 1);
            Ok(OffsetBuffer::new(ScalarBuffer::from(offsets)))
        }
    }

    pub fn unravel_validity(&mut self) -> Option<NullBuffer> {
        let Some(def_levels) = &self.def_levels else {
            return None;
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
        // Basic case, rep & def
        let mut builder = RepDefBuilder::default();
        builder.add_validity_bitmap(validity(&[true, false, true]));
        builder.add_offsets(offsets_64(&[0, 2, 3, 5]));
        builder.add_validity_bitmap(validity(&[true, true, true, false, true]));
        builder.add_offsets(offsets_64(&[0, 1, 3, 5, 7, 9]));
        builder.add_validity_bitmap(validity(&[
            true, true, true, false, false, false, true, true, false,
        ]));

        let repdefs = RepDefBuilder::serialize(vec![builder]);
        let rep = repdefs.repetition_levels.unwrap();
        let def = repdefs.definition_levels.unwrap();

        assert_eq!(vec![0, 0, 0, 3, 3, 2, 2, 0, 1], def);
        assert_eq!(vec![2, 1, 0, 2, 0, 2, 0, 1, 0], rep);

        let mut unraveler = RepDefUnraveler::new(Some(rep), Some(def));

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

    #[test]
    fn test_repdef_all_valid() {
        let mut builder = RepDefBuilder::default();
        builder.add_no_null(3);
        builder.add_offsets(offsets_64(&[0, 2, 3, 5]));
        builder.add_no_null(5);
        builder.add_offsets(offsets_64(&[0, 1, 3, 5, 7, 9]));
        builder.add_no_null(9);

        let repdefs = RepDefBuilder::serialize(vec![builder]);
        let rep = repdefs.repetition_levels.unwrap();
        assert!(repdefs.definition_levels.is_none());

        assert_eq!(vec![2, 1, 0, 2, 0, 2, 0, 1, 0], rep);

        let mut unraveler = RepDefUnraveler::new(Some(rep), None);

        assert_eq!(unraveler.unravel_validity(), None);
        assert_eq!(
            unraveler.unravel_offsets::<i32>().unwrap().inner(),
            offsets_32(&[0, 1, 3, 5, 7, 9]).inner()
        );
        assert_eq!(unraveler.unravel_validity(), None);
        assert_eq!(
            unraveler.unravel_offsets::<i32>().unwrap().inner(),
            offsets_32(&[0, 2, 3, 5]).inner()
        );
        assert_eq!(unraveler.unravel_validity(), None);
    }

    #[test]
    fn test_repdef_no_rep() {
        let mut builder = RepDefBuilder::default();
        builder.add_no_null(3);
        builder.add_validity_bitmap(validity(&[false, false, true, true, true]));
        builder.add_validity_bitmap(validity(&[false, true, true, true, false]));

        let repdefs = RepDefBuilder::serialize(vec![builder]);
        assert!(repdefs.repetition_levels.is_none());
        let def = repdefs.definition_levels.unwrap();

        assert_eq!(vec![2, 2, 0, 0, 1], def);

        let mut unraveler = RepDefUnraveler::new(None, Some(def));

        assert_eq!(
            unraveler.unravel_validity(),
            Some(validity(&[false, false, true, true, false]))
        );
        assert_eq!(
            unraveler.unravel_validity(),
            Some(validity(&[false, false, true, true, true]))
        );
        assert_eq!(unraveler.unravel_validity(), None);
    }

    #[test]
    fn test_repdef_multiple_builders() {
        // Basic case, rep & def
        let mut builder1 = RepDefBuilder::default();
        builder1.add_validity_bitmap(validity(&[true]));
        builder1.add_offsets(offsets_64(&[0, 2]));
        builder1.add_validity_bitmap(validity(&[true, true]));
        builder1.add_offsets(offsets_64(&[0, 1, 3]));
        builder1.add_validity_bitmap(validity(&[true, true, true]));

        let mut builder2 = RepDefBuilder::default();
        builder2.add_validity_bitmap(validity(&[false, true]));
        builder2.add_offsets(offsets_64(&[0, 1, 3]));
        builder2.add_validity_bitmap(validity(&[true, false, true]));
        builder2.add_offsets(offsets_64(&[0, 2, 4, 6]));
        builder2.add_validity_bitmap(validity(&[false, false, false, true, true, false]));

        let repdefs = RepDefBuilder::serialize(vec![builder1, builder2]);
        let rep = repdefs.repetition_levels.unwrap();
        let def = repdefs.definition_levels.unwrap();

        assert_eq!(vec![2, 1, 0, 2, 0, 2, 0, 1, 0], rep);
        assert_eq!(vec![0, 0, 0, 3, 3, 2, 2, 0, 1], def);
    }
}
