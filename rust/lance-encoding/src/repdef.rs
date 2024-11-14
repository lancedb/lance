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
//! been implemented in Apache Parquet.  However, the implementation here is not necessarily
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

use std::{iter::Zip, sync::Arc};

use arrow_array::OffsetSizeTrait;
use arrow_buffer::{
    ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, NullBuffer, OffsetBuffer, ScalarBuffer,
};
use lance_core::{utils::bit::log_2_ceil, Error, Result};
use snafu::{location, Location};

// We assume 16 bits is good enough for rep-def levels.  This gives us
// 65536 levels of struct nesting and list nesting.
pub type LevelBuffer = Vec<u16>;

// As we build up rep/def from arrow arrays we record a
// series of RawRepDef objects
#[derive(Clone, Debug)]
enum RawRepDef {
    Offsets(Arc<[i64]>),
    Validity(BooleanBuffer),
    NoNull(usize),
}

/// Represents repetition and definition levels that have been
/// serialized into a pair of (optional) level buffers
#[derive(Debug)]
pub struct SerializedRepDefs {
    // If None, there are no lists
    pub repetition_levels: Option<LevelBuffer>,
    // If None, there are no nulls
    pub definition_levels: Option<LevelBuffer>,
}

impl SerializedRepDefs {
    /// Creates an empty SerializedRepDefs (no repetition, all valid)
    pub fn empty() -> Self {
        Self {
            repetition_levels: None,
            definition_levels: None,
        }
    }
}

/// The RepDefBuilder is used to collect offsets & validity buffers
/// from arrow structures.  Once we have those we use the SerializerContext
/// to build the actual repetition and definition levels by walking through
/// the arrow constructs in reverse order.
///
/// The algorithm for definition levels is pretty simple
///
/// Given:
///  - a validity buffer of [T, F, F, T, T]
///  - a current def level of 5
///  - a current definitions of [0, 1, 3, 3, 0]
///
/// We walk through the definitions and replace them with
///   the current level whenever a value is invalid.  Thus
///   our output is: [0, 5, 5, 3, 0]
///
/// The algorithm for repetition levels is more complex.
///
/// The first time we see an offsets buffer we initialize the
/// rep levels to have a value of 1 whenever a list starts and 0
/// otherwise.
///
/// So, given offsets of [0, 3, 5] and no repetition we create
/// rep levels [1 0 0 1 0]
///
/// However, we also record the offsets into our current rep and
/// def levels and all operations happen in context of those offsets.
///
/// For example, continuing the above scenario we might then see validity
/// of [T, F].  This is strange since our validity bitmap has 2 items but
/// we would have 5 definition levels.  We can use our current offsets
/// ([0, 3, 5]) to expand [T, F] into [T, T, T, F, F].
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

/// A structure used to collect validity buffers and offsets from arrow
/// arrays and eventually create repetition and definition levels
///
/// As we are encoding the structural encoders are given this struct and
/// will record the arrow information into it.  Once we hit a leaf node we
/// serialize the data into rep/def levels and write these into the page.
#[derive(Clone, Default)]
pub struct RepDefBuilder {
    // The rep/def info we have collected so far
    repdefs: Vec<RawRepDef>,
    // The current length, can get larger as we traverse lists (e.g. an
    // array might have 5 lists which results in 50 items)
    //
    // Starts uninitialized until we see the first rep/def item
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

    /// Returns true if there is only a single layer of definition
    pub fn is_simple_validity(&self) -> bool {
        self.repdefs.len() == 1 && matches!(self.repdefs[0], RawRepDef::Validity(_))
    }

    /// Return True if any layer has a validity bitmap
    ///
    /// Return False if all layers are non-null (the def levels can
    /// be skipped in this case)
    pub fn has_nulls(&self) -> bool {
        self.repdefs
            .iter()
            .any(|rd| matches!(rd, RawRepDef::Validity(_)))
    }

    /// Registers a nullable validity bitmap
    pub fn add_validity_bitmap(&mut self, validity: NullBuffer) {
        self.check_validity_len(&validity);
        self.repdefs
            .push(RawRepDef::Validity(validity.into_inner()));
    }

    /// Registers an all-valid validity layer
    pub fn add_no_null(&mut self, len: usize) {
        self.repdefs.push(RawRepDef::NoNull(len));
    }

    fn check_offset_len(&mut self, offsets: &[i64]) {
        if let Some(len) = self.len {
            assert!(offsets.len() == len + 1);
        }
        self.len = Some(offsets[offsets.len() - 1] as usize);
    }

    /// Adds a layer of offsets
    ///
    /// Note: a List/LargeList/etc. array has both offsets and validity.  The
    /// caller should register the validity before registering the offsets
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

    /// Converts the validity / offsets buffers that have been gathered so far
    /// into repetition and definition levels
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

/// Starts with serialized repetition and definition levels and unravels
/// them into validity buffers and offsets buffers
///
/// This is used during decoding to create the necessary arrow structures
#[derive(Debug)]
pub struct RepDefUnraveler {
    rep_levels: Option<LevelBuffer>,
    def_levels: Option<LevelBuffer>,
    // Current definition level to compare to.
    current_def_cmp: u16,
}

impl RepDefUnraveler {
    /// Creates a new unraveler from serialized repetition and definition information
    pub fn new(rep_levels: Option<LevelBuffer>, def_levels: Option<LevelBuffer>) -> Self {
        Self {
            rep_levels,
            def_levels,
            current_def_cmp: 0,
        }
    }

    /// Unravels a layer of offsets from the unraveler into the given offset width
    ///
    /// When decoding a list the caller should first unravel the offsets and then
    /// unravel the validity (this is the opposite order used during encoding)
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
            let mut read_idx = 0;
            let mut write_idx = 0;
            while read_idx < rep_levels.len() {
                // SAFETY: We assert that rep_levels and def_levels have the same
                // len and read_idx and write_idx can never go past the end.
                unsafe {
                    let rep_val = *rep_levels.get_unchecked(read_idx);
                    if rep_val != 0 {
                        // Finish the current list
                        offsets.push(to_offset(curlen)?);
                        *rep_levels.get_unchecked_mut(write_idx) = rep_val - 1;
                        *def_levels.get_unchecked_mut(write_idx) =
                            *def_levels.get_unchecked(read_idx);
                        write_idx += 1;
                    }
                    curlen += 1;
                    read_idx += 1;
                }
            }
            offsets.push(to_offset(curlen)?);
            rep_levels.truncate(offsets.len() - 1);
            def_levels.truncate(offsets.len() - 1);
            Ok(OffsetBuffer::new(ScalarBuffer::from(offsets)))
        } else {
            // SAFETY: See above loop
            let mut read_idx = 0;
            let mut write_idx = 0;
            while read_idx < rep_levels.len() {
                // SAFETY: read_idx / write_idx cannot go past rep_levels.len()
                unsafe {
                    let rep_val = *rep_levels.get_unchecked(read_idx);
                    if rep_val != 0 {
                        // Finish the current list
                        offsets.push(to_offset(curlen)?);
                        *rep_levels.get_unchecked_mut(write_idx) = rep_val - 1;
                        write_idx += 1;
                    }
                    curlen += 1;
                    read_idx += 1;
                }
            }
            offsets.push(to_offset(curlen)?);
            rep_levels.truncate(offsets.len() - 1);
            Ok(OffsetBuffer::new(ScalarBuffer::from(offsets)))
        }
    }

    /// Unravels a layer of validity from the definition levels
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

/// A [`ControlWordIterator`] when there are both repetition and definition levels
///
/// The iterator will put the repetition level in the upper bits and the definition
/// level in the lower bits.  The number of bits used for each level is determined
/// by the width of the repetition and definition levels.
#[derive(Debug)]
pub struct BinaryControlWordIterator<I: Iterator<Item = (u16, u16)>, W> {
    repdef: I,
    def_width: usize,
    rep_mask: u16,
    def_mask: u16,
    bits_rep: u8,
    bits_def: u8,
    phantom: std::marker::PhantomData<W>,
}

impl<I: Iterator<Item = (u16, u16)>> BinaryControlWordIterator<I, u8> {
    fn append_next(&mut self, buf: &mut Vec<u8>) {
        let next = self.repdef.next().unwrap();
        let control_word: u8 =
            (((next.0 & self.rep_mask) as u8) << self.def_width) + ((next.1 & self.def_mask) as u8);
        buf.push(control_word);
    }
}

impl<I: Iterator<Item = (u16, u16)>> BinaryControlWordIterator<I, u16> {
    fn append_next(&mut self, buf: &mut Vec<u8>) {
        let next = self.repdef.next().unwrap();
        let control_word: u16 =
            ((next.0 & self.rep_mask) << self.def_width) + (next.1 & self.def_mask);
        let control_word = control_word.to_le_bytes();
        buf.push(control_word[0]);
        buf.push(control_word[1]);
    }
}

impl<I: Iterator<Item = (u16, u16)>> BinaryControlWordIterator<I, u32> {
    fn append_next(&mut self, buf: &mut Vec<u8>) {
        let next = self.repdef.next().unwrap();
        let control_word: u32 = (((next.0 & self.rep_mask) as u32) << self.def_width)
            + ((next.1 & self.def_mask) as u32);
        let control_word = control_word.to_le_bytes();
        buf.push(control_word[0]);
        buf.push(control_word[1]);
        buf.push(control_word[2]);
        buf.push(control_word[3]);
    }
}

/// A [`ControlWordIterator`] when there are only definition levels or only repetition levels
#[derive(Debug)]
pub struct UnaryControlWordIterator<I: Iterator<Item = u16>, W> {
    repdef: I,
    level_mask: u16,
    bits_rep: u8,
    bits_def: u8,
    phantom: std::marker::PhantomData<W>,
}

impl<I: Iterator<Item = u16>> UnaryControlWordIterator<I, u8> {
    fn append_next(&mut self, buf: &mut Vec<u8>) {
        let next = self.repdef.next().unwrap();
        buf.push((next & self.level_mask) as u8);
    }
}

impl<I: Iterator<Item = u16>> UnaryControlWordIterator<I, u16> {
    fn append_next(&mut self, buf: &mut Vec<u8>) {
        let next = self.repdef.next().unwrap() & self.level_mask;
        let control_word = next.to_le_bytes();
        buf.push(control_word[0]);
        buf.push(control_word[1]);
    }
}

impl<I: Iterator<Item = u16>> UnaryControlWordIterator<I, u32> {
    fn append_next(&mut self, buf: &mut Vec<u8>) {
        let next = (self.repdef.next().unwrap() & self.level_mask) as u32;
        let control_word = next.to_le_bytes();
        buf.push(control_word[0]);
        buf.push(control_word[1]);
        buf.push(control_word[2]);
        buf.push(control_word[3]);
    }
}

/// A [`ControlWordIterator`] when there are no repetition or definition levels
#[derive(Debug)]
pub struct NilaryControlWordIterator;

/// Helper function to get a bit mask of the given width
fn get_mask(width: u16) -> u16 {
    (1 << width) - 1
}

/// An iterator that generates control words from repetition and definition levels
///
/// "Control word" is just a fancy term for a single u8/u16/u32 that contains both
/// the repetition and definition in it.
///
/// In the large majority of case we only need a single byte to represent both the
/// repetition and definition levels.  However, if there is deep nesting then we may
/// need two bytes.  In the worst case we need 4 bytes though this suggests hundreds of
/// levels of nesting which seems unlikely to encounter in practice.
#[derive(Debug)]
pub enum ControlWordIterator {
    Binary8(BinaryControlWordIterator<Zip<std::vec::IntoIter<u16>, std::vec::IntoIter<u16>>, u8>),
    Binary16(BinaryControlWordIterator<Zip<std::vec::IntoIter<u16>, std::vec::IntoIter<u16>>, u16>),
    Binary32(BinaryControlWordIterator<Zip<std::vec::IntoIter<u16>, std::vec::IntoIter<u16>>, u32>),
    Unary8(UnaryControlWordIterator<std::vec::IntoIter<u16>, u8>),
    Unary16(UnaryControlWordIterator<std::vec::IntoIter<u16>, u16>),
    Unary32(UnaryControlWordIterator<std::vec::IntoIter<u16>, u32>),
    Nilary(NilaryControlWordIterator),
}

impl ControlWordIterator {
    /// Appends the next control word to the buffer
    pub fn append_next(&mut self, buf: &mut Vec<u8>) {
        match self {
            Self::Binary8(iter) => iter.append_next(buf),
            Self::Binary16(iter) => iter.append_next(buf),
            Self::Binary32(iter) => iter.append_next(buf),
            Self::Unary8(iter) => iter.append_next(buf),
            Self::Unary16(iter) => iter.append_next(buf),
            Self::Unary32(iter) => iter.append_next(buf),
            Self::Nilary(_) => {}
        }
    }

    /// Returns the number of bytes per control word
    pub fn bytes_per_word(&self) -> usize {
        match self {
            Self::Binary8(_) => 1,
            Self::Binary16(_) => 2,
            Self::Binary32(_) => 4,
            Self::Unary8(_) => 1,
            Self::Unary16(_) => 2,
            Self::Unary32(_) => 4,
            Self::Nilary(_) => 0,
        }
    }

    /// Returns the number of bits used for the repetition level
    pub fn bits_rep(&self) -> u8 {
        match self {
            Self::Binary8(iter) => iter.bits_rep,
            Self::Binary16(iter) => iter.bits_rep,
            Self::Binary32(iter) => iter.bits_rep,
            Self::Unary8(iter) => iter.bits_rep,
            Self::Unary16(iter) => iter.bits_rep,
            Self::Unary32(iter) => iter.bits_rep,
            Self::Nilary(_) => 0,
        }
    }

    /// Returns the number of bits used for the definition level
    pub fn bits_def(&self) -> u8 {
        match self {
            Self::Binary8(iter) => iter.bits_def,
            Self::Binary16(iter) => iter.bits_def,
            Self::Binary32(iter) => iter.bits_def,
            Self::Unary8(iter) => iter.bits_def,
            Self::Unary16(iter) => iter.bits_def,
            Self::Unary32(iter) => iter.bits_def,
            Self::Nilary(_) => 0,
        }
    }
}

/// Builds a [`ControlWordIterator`] from repetition and definition levels
/// by first calculating the width needed and then creating the iterator
/// with the appropriate width
pub fn build_control_word_iterator(
    rep: Option<Vec<u16>>,
    max_rep: u16,
    def: Option<Vec<u16>>,
    max_def: u16,
) -> ControlWordIterator {
    let rep_width = if max_rep == 0 {
        0
    } else {
        log_2_ceil(max_rep as u32) as u16
    };
    let rep_mask = if max_rep == 0 { 0 } else { get_mask(rep_width) };
    let def_width = if max_def == 0 {
        0
    } else {
        log_2_ceil(max_def as u32) as u16
    };
    let def_mask = if max_def == 0 { 0 } else { get_mask(def_width) };
    let total_width = rep_width + def_width;
    match (rep, def) {
        (Some(rep), Some(def)) => {
            let iter = rep.into_iter().zip(def);
            let def_width = def_width as usize;
            if total_width <= 8 {
                ControlWordIterator::Binary8(BinaryControlWordIterator {
                    repdef: iter,
                    rep_mask,
                    def_mask,
                    def_width,
                    bits_rep: rep_width as u8,
                    bits_def: def_width as u8,
                    phantom: std::marker::PhantomData,
                })
            } else if total_width <= 16 {
                ControlWordIterator::Binary16(BinaryControlWordIterator {
                    repdef: iter,
                    rep_mask,
                    def_mask,
                    def_width,
                    bits_rep: rep_width as u8,
                    bits_def: def_width as u8,
                    phantom: std::marker::PhantomData,
                })
            } else {
                ControlWordIterator::Binary32(BinaryControlWordIterator {
                    repdef: iter,
                    rep_mask,
                    def_mask,
                    def_width,
                    bits_rep: rep_width as u8,
                    bits_def: def_width as u8,
                    phantom: std::marker::PhantomData,
                })
            }
        }
        (Some(lev), None) => {
            let iter = lev.into_iter();
            if total_width <= 8 {
                ControlWordIterator::Unary8(UnaryControlWordIterator {
                    repdef: iter,
                    level_mask: rep_mask,
                    bits_rep: total_width as u8,
                    bits_def: 0,
                    phantom: std::marker::PhantomData,
                })
            } else if total_width <= 16 {
                ControlWordIterator::Unary16(UnaryControlWordIterator {
                    repdef: iter,
                    level_mask: rep_mask,
                    bits_rep: total_width as u8,
                    bits_def: 0,
                    phantom: std::marker::PhantomData,
                })
            } else {
                ControlWordIterator::Unary32(UnaryControlWordIterator {
                    repdef: iter,
                    level_mask: rep_mask,
                    bits_rep: total_width as u8,
                    bits_def: 0,
                    phantom: std::marker::PhantomData,
                })
            }
        }
        (None, Some(lev)) => {
            let iter = lev.into_iter();
            if total_width <= 8 {
                ControlWordIterator::Unary8(UnaryControlWordIterator {
                    repdef: iter,
                    level_mask: def_mask,
                    bits_rep: 0,
                    bits_def: total_width as u8,
                    phantom: std::marker::PhantomData,
                })
            } else if total_width <= 16 {
                ControlWordIterator::Unary16(UnaryControlWordIterator {
                    repdef: iter,
                    level_mask: def_mask,
                    bits_rep: 0,
                    bits_def: total_width as u8,
                    phantom: std::marker::PhantomData,
                })
            } else {
                ControlWordIterator::Unary32(UnaryControlWordIterator {
                    repdef: iter,
                    level_mask: def_mask,
                    bits_rep: 0,
                    bits_def: total_width as u8,
                    phantom: std::marker::PhantomData,
                })
            }
        }
        (None, None) => ControlWordIterator::Nilary(NilaryControlWordIterator {}),
    }
}

/// A parser to unwrap control words into repetition and definition levels
///
/// This is the inverse of the [`ControlWordIterator`].
#[derive(Copy, Clone, Debug)]
pub enum ControlWordParser {
    // First item is the bits to shift, second is the mask to apply (the mask can be
    // calculated from the bits to shift but we don't want to calculate it each time)
    BOTH8(u8, u32),
    BOTH16(u8, u32),
    BOTH32(u8, u32),
    REP8,
    REP16,
    REP32,
    DEF8,
    DEF16,
    DEF32,
    NIL,
}

impl ControlWordParser {
    fn parse_both<const WORD_SIZE: u8>(
        src: &[u8],
        dst_rep: &mut Vec<u16>,
        dst_def: &mut Vec<u16>,
        bits_to_shift: u8,
        mask_to_apply: u32,
    ) {
        match WORD_SIZE {
            1 => {
                let word = src[0];
                let rep = word >> bits_to_shift;
                let def = word & (mask_to_apply as u8);
                dst_rep.push(rep as u16);
                dst_def.push(def as u16);
            }
            2 => {
                let word = u16::from_le_bytes([src[0], src[1]]);
                let rep = word >> bits_to_shift;
                let def = word & mask_to_apply as u16;
                dst_rep.push(rep);
                dst_def.push(def);
            }
            4 => {
                let word = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                let rep = word >> bits_to_shift;
                let def = word & mask_to_apply;
                dst_rep.push(rep as u16);
                dst_def.push(def as u16);
            }
            _ => unreachable!(),
        }
    }

    fn parse_one<const WORD_SIZE: u8>(src: &[u8], dst: &mut Vec<u16>) {
        match WORD_SIZE {
            1 => {
                let word = src[0];
                dst.push(word as u16);
            }
            2 => {
                let word = u16::from_le_bytes([src[0], src[1]]);
                dst.push(word);
            }
            4 => {
                let word = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                dst.push(word as u16);
            }
            _ => unreachable!(),
        }
    }

    /// Returns the number of bytes per control word
    pub fn bytes_per_word(&self) -> usize {
        match self {
            Self::BOTH8(..) => 1,
            Self::BOTH16(..) => 2,
            Self::BOTH32(..) => 4,
            Self::REP8 => 1,
            Self::REP16 => 2,
            Self::REP32 => 4,
            Self::DEF8 => 1,
            Self::DEF16 => 2,
            Self::DEF32 => 4,
            Self::NIL => 0,
        }
    }

    /// Appends the next control word to the rep & def buffers
    ///
    /// `src` should be pointing at the first byte (little endian) of the control word
    ///
    /// `dst_rep` and `dst_def` are the buffers to append the rep and def levels to.
    /// They will not be appended to if not needed.
    pub fn parse(&self, src: &[u8], dst_rep: &mut Vec<u16>, dst_def: &mut Vec<u16>) {
        match self {
            Self::BOTH8(bits_to_shift, mask_to_apply) => {
                Self::parse_both::<1>(src, dst_rep, dst_def, *bits_to_shift, *mask_to_apply)
            }
            Self::BOTH16(bits_to_shift, mask_to_apply) => {
                Self::parse_both::<2>(src, dst_rep, dst_def, *bits_to_shift, *mask_to_apply)
            }
            Self::BOTH32(bits_to_shift, mask_to_apply) => {
                Self::parse_both::<4>(src, dst_rep, dst_def, *bits_to_shift, *mask_to_apply)
            }
            Self::REP8 => Self::parse_one::<1>(src, dst_rep),
            Self::REP16 => Self::parse_one::<2>(src, dst_rep),
            Self::REP32 => Self::parse_one::<4>(src, dst_rep),
            Self::DEF8 => Self::parse_one::<1>(src, dst_def),
            Self::DEF16 => Self::parse_one::<2>(src, dst_def),
            Self::DEF32 => Self::parse_one::<4>(src, dst_def),
            Self::NIL => {}
        }
    }

    /// Creates a new parser from the number of bits used for the repetition and definition levels
    pub fn new(bits_rep: u8, bits_def: u8) -> Self {
        let total_bits = bits_rep + bits_def;

        enum WordSize {
            One,
            Two,
            Four,
        }

        let word_size = if total_bits <= 8 {
            WordSize::One
        } else if total_bits <= 16 {
            WordSize::Two
        } else {
            WordSize::Four
        };

        match (bits_rep > 0, bits_def > 0, word_size) {
            (false, false, _) => Self::NIL,
            (false, true, WordSize::One) => Self::DEF8,
            (false, true, WordSize::Two) => Self::DEF16,
            (false, true, WordSize::Four) => Self::DEF32,
            (true, false, WordSize::One) => Self::REP8,
            (true, false, WordSize::Two) => Self::REP16,
            (true, false, WordSize::Four) => Self::REP32,
            (true, true, WordSize::One) => Self::BOTH8(bits_def, get_mask(bits_def as u16) as u32),
            (true, true, WordSize::Two) => Self::BOTH16(bits_def, get_mask(bits_def as u16) as u32),
            (true, true, WordSize::Four) => {
                Self::BOTH32(bits_def, get_mask(bits_def as u16) as u32)
            }
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

    #[test]
    fn test_control_words() {
        // Convert to control words, verify expected, convert back, verify same as original
        fn check(
            rep: Vec<u16>,
            def: Vec<u16>,
            expected_values: Vec<u8>,
            expected_bytes_per_word: usize,
            expected_bits_rep: u8,
            expected_bits_def: u8,
        ) {
            let num_vals = rep.len().max(def.len());
            let max_rep = rep.iter().max().copied().unwrap_or(0);
            let max_def = def.iter().max().copied().unwrap_or(0);

            let in_rep = if rep.is_empty() {
                None
            } else {
                Some(rep.clone())
            };
            let in_def = if def.is_empty() {
                None
            } else {
                Some(def.clone())
            };

            let mut iter = super::build_control_word_iterator(in_rep, max_rep, in_def, max_def);
            assert_eq!(iter.bytes_per_word(), expected_bytes_per_word);
            assert_eq!(iter.bits_rep(), expected_bits_rep);
            assert_eq!(iter.bits_def(), expected_bits_def);
            let mut cw_vec = Vec::with_capacity(num_vals * iter.bytes_per_word());

            for _ in 0..num_vals {
                iter.append_next(&mut cw_vec);
            }

            assert_eq!(expected_values, cw_vec);

            let parser = super::ControlWordParser::new(expected_bits_rep, expected_bits_def);

            let mut rep_out = Vec::with_capacity(num_vals);
            let mut def_out = Vec::with_capacity(num_vals);

            if expected_bytes_per_word > 0 {
                for slice in cw_vec.chunks_exact(expected_bytes_per_word) {
                    parser.parse(slice, &mut rep_out, &mut def_out);
                }
            }

            assert_eq!(rep, rep_out);
            assert_eq!(def, def_out);
        }

        // Each will need 4 bits and so we should get 1-byte control words
        let rep = vec![0_u16, 7, 3, 2, 9, 8, 12, 5];
        let def = vec![5_u16, 3, 1, 2, 12, 15, 0, 2];
        let expected = vec![
            0b00000101, // 0, 5
            0b01110011, // 7, 3
            0b00110001, // 3, 1
            0b00100010, // 2, 2
            0b10011100, // 9, 12
            0b10001111, // 8, 15
            0b11000000, // 12, 0
            0b01010010, // 5, 2
        ];
        check(rep, def, expected, 1, 4, 4);

        // Now we need 5 bits for def so we get 2-byte control words
        let rep = vec![0_u16, 7, 3, 2, 9, 8, 12, 5];
        let def = vec![5_u16, 3, 1, 2, 12, 22, 0, 2];
        let expected = vec![
            0b00000101, 0b00000000, // 0, 5
            0b11100011, 0b00000000, // 7, 3
            0b01100001, 0b00000000, // 3, 1
            0b01000010, 0b00000000, // 2, 2
            0b00101100, 0b00000001, // 9, 12
            0b00010110, 0b00000001, // 8, 22
            0b10000000, 0b00000001, // 12, 0
            0b10100010, 0b00000000, // 5, 2
        ];
        check(rep, def, expected, 2, 4, 5);

        // Just rep, 4 bits so 1 byte each
        let levels = vec![0_u16, 7, 3, 2, 9, 8, 12, 5];
        let expected = vec![
            0b00000000, // 0
            0b00000111, // 7
            0b00000011, // 3
            0b00000010, // 2
            0b00001001, // 9
            0b00001000, // 8
            0b00001100, // 12
            0b00000101, // 5
        ];
        check(levels.clone(), Vec::default(), expected.clone(), 1, 4, 0);

        // Just def
        check(Vec::default(), levels, expected, 1, 0, 4);

        // No rep, no def, no bytes
        check(Vec::default(), Vec::default(), Vec::default(), 0, 0, 0);
    }
}
