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

use std::{
    iter::{Copied, Zip},
    sync::Arc,
};

use arrow_array::OffsetSizeTrait;
use arrow_buffer::{
    ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, NullBuffer, OffsetBuffer, ScalarBuffer,
};
use lance_core::{utils::bit::log_2_ceil, Error, Result};
use snafu::{location, Location};

use crate::buffer::LanceBuffer;

// We assume 16 bits is good enough for rep-def levels.  This gives us
// 65536 levels of struct nesting and list nesting.
pub type LevelBuffer = Vec<u16>;

/// Represents information that we extract from a list array as we are
/// encoding
#[derive(Clone, Debug)]
struct OffsetDesc {
    offsets: Arc<[i64]>,
    specials: Arc<[SpecialOffset]>,
    validity: Option<BooleanBuffer>,
    has_empty_lists: bool,
    num_values: usize,
}

/// Represents validity information that we extract from non-list arrays (that
/// have nulls) as we are encoding
#[derive(Clone, Debug)]
struct ValidityDesc {
    validity: Option<BooleanBuffer>,
    num_values: usize,
}

// As we build up rep/def from arrow arrays we record a
// series of RawRepDef objects.  Each one corresponds to layer
// in the array structure
#[derive(Clone, Debug)]
enum RawRepDef {
    Offsets(OffsetDesc),
    Validity(ValidityDesc),
}

impl RawRepDef {
    // Are there any nulls in this layer
    fn has_nulls(&self) -> bool {
        match self {
            Self::Offsets(OffsetDesc { validity, .. }) => validity.is_some(),
            Self::Validity(ValidityDesc { validity, .. }) => validity.is_some(),
        }
    }

    // How many values are in this layer
    fn num_values(&self) -> usize {
        match self {
            Self::Offsets(OffsetDesc { num_values, .. }) => *num_values,
            Self::Validity(ValidityDesc { num_values, .. }) => *num_values,
        }
    }
}

/// Represents repetition and definition levels that have been
/// serialized into a pair of (optional) level buffers
#[derive(Debug)]
pub struct SerializedRepDefs {
    /// The repetition levels, one per item
    ///
    /// If None, there are no lists
    pub repetition_levels: Option<Arc<[u16]>>,
    /// The definition levels, one per item
    ///
    /// If None, there are no nulls
    pub definition_levels: Option<Arc<[u16]>>,
    /// Special records indicate empty / null lists
    ///
    /// These do not have any mapping to items.  There may be empty or there may
    /// be more special records than items or anywhere in between.
    pub special_records: Vec<SpecialRecord>,
    /// The meaning of each definition level
    pub def_meaning: Vec<DefinitionInterpretation>,
    /// The maximum level that is "visible" from the lowest level
    ///
    /// This is the last level before we encounter a list level of some kind.  Once we've
    /// hit a list level then nulls in any level beyond do not map to actual items.
    ///
    /// This is None if there are no lists
    pub max_visible_level: Option<u16>,
}

impl SerializedRepDefs {
    pub fn new(
        repetition_levels: Option<LevelBuffer>,
        definition_levels: Option<LevelBuffer>,
        special_records: Vec<SpecialRecord>,
        def_meaning: Vec<DefinitionInterpretation>,
    ) -> Self {
        let first_list = def_meaning.iter().position(|level| level.is_list());
        let max_visible_level = first_list.map(|first_list| {
            def_meaning
                .iter()
                .map(|level| level.num_def_levels())
                .take(first_list)
                .sum::<u16>()
        });
        Self {
            repetition_levels: repetition_levels.map(Arc::from),
            definition_levels: definition_levels.map(Arc::from),
            special_records,
            def_meaning,
            max_visible_level,
        }
    }

    /// Creates an empty SerializedRepDefs (no repetition, all valid)
    pub fn empty(def_meaning: Vec<DefinitionInterpretation>) -> Self {
        Self {
            repetition_levels: None,
            definition_levels: None,
            special_records: Vec::new(),
            def_meaning,
            max_visible_level: None,
        }
    }

    pub fn rep_slicer(&self) -> Option<RepDefSlicer> {
        self.repetition_levels
            .as_ref()
            .map(|rep| RepDefSlicer::new(self, rep.clone()))
    }

    pub fn def_slicer(&self) -> Option<RepDefSlicer> {
        self.definition_levels
            .as_ref()
            .map(|def| RepDefSlicer::new(self, def.clone()))
    }

    /// Creates a version of the SerializedRepDefs with the specials collapsed into
    /// the repetition and definition levels
    pub fn collapse_specials(self) -> Self {
        if self.special_records.is_empty() {
            return self;
        }

        // If we have specials then we must have repetition
        let rep = self.repetition_levels.unwrap();

        let new_len = rep.len() + self.special_records.len();

        let mut new_rep = Vec::with_capacity(new_len);
        let mut new_def = Vec::with_capacity(new_len);

        // Now we just merge the rep/def levels and the specials into one list.  There is just
        // one tricky part.  If a non-special is added after a special item then it swaps its
        // repetition level with the special item.
        if let Some(def) = self.definition_levels {
            let mut def_itr = def.iter();
            let mut rep_itr = rep.iter();
            let mut special_itr = self.special_records.into_iter().peekable();
            let mut last_special = None;

            for idx in 0..new_len {
                if let Some(special) = special_itr.peek() {
                    if special.pos == idx {
                        new_rep.push(special.rep_level);
                        new_def.push(special.def_level);
                        special_itr.next();
                        last_special = Some(new_rep.last_mut().unwrap());
                    } else {
                        let rep = if let Some(last_special) = last_special {
                            let rep = *last_special;
                            *last_special = *rep_itr.next().unwrap();
                            rep
                        } else {
                            *rep_itr.next().unwrap()
                        };
                        new_rep.push(rep);
                        new_def.push(*def_itr.next().unwrap());
                        last_special = None;
                    }
                } else {
                    let rep = if let Some(last_special) = last_special {
                        let rep = *last_special;
                        *last_special = *rep_itr.next().unwrap();
                        rep
                    } else {
                        *rep_itr.next().unwrap()
                    };
                    new_rep.push(rep);
                    new_def.push(*def_itr.next().unwrap());
                    last_special = None;
                }
            }
        } else {
            let mut rep_itr = rep.iter();
            let mut special_itr = self.special_records.into_iter().peekable();
            let mut last_special = None;

            for idx in 0..new_len {
                if let Some(special) = special_itr.peek() {
                    if special.pos == idx {
                        new_rep.push(special.rep_level);
                        new_def.push(special.def_level);
                        special_itr.next();
                        last_special = Some(new_rep.last_mut().unwrap());
                    } else {
                        let rep = if let Some(last_special) = last_special {
                            let rep = *last_special;
                            *last_special = *rep_itr.next().unwrap();
                            rep
                        } else {
                            *rep_itr.next().unwrap()
                        };
                        new_rep.push(rep);
                        new_def.push(0);
                        last_special = None;
                    }
                } else {
                    let rep = if let Some(last_special) = last_special {
                        let rep = *last_special;
                        *last_special = *rep_itr.next().unwrap();
                        rep
                    } else {
                        *rep_itr.next().unwrap()
                    };
                    new_rep.push(rep);
                    new_def.push(0);
                    last_special = None;
                }
            }
        }

        Self {
            repetition_levels: Some(new_rep.into()),
            definition_levels: Some(new_def.into()),
            special_records: Vec::new(),
            def_meaning: self.def_meaning,
            max_visible_level: self.max_visible_level,
        }
    }
}

pub struct RepDefSlicer<'a> {
    repdef: &'a SerializedRepDefs,
    to_slice: LanceBuffer,
    current: usize,
}

impl<'a> RepDefSlicer<'a> {
    fn new(repdef: &'a SerializedRepDefs, levels: Arc<[u16]>) -> Self {
        Self {
            repdef,
            to_slice: LanceBuffer::reinterpret_slice(levels),
            current: 0,
        }
    }

    pub fn num_levels(&self) -> usize {
        self.to_slice.len()
    }

    pub fn all_levels(&self) -> &LanceBuffer {
        &self.to_slice
    }

    pub fn slice_next(&mut self, num_values: usize) -> LanceBuffer {
        let start = self.current;
        let Some(max_visible_level) = self.repdef.max_visible_level else {
            // No lists, should be 1:1 mapping from levels to values
            self.current = start + num_values;
            return self.to_slice.slice_with_length(start * 2, num_values * 2);
        };
        if let Some(def) = self.repdef.definition_levels.as_ref() {
            // There are lists and there are def levels.  That means there may be
            // more rep/def levels than values.  We need to scan the def levels to figure
            // out which items are "invisible" and skip over them
            let mut def_itr = def[start..].iter();
            let mut num_taken = 0;
            let mut num_passed = 0;
            while num_taken < num_values {
                let def_level = *def_itr.next().unwrap();
                if def_level <= max_visible_level {
                    num_taken += 1;
                }
                num_passed += 1;
            }
            self.current = start + num_passed;
            self.to_slice.slice_with_length(start * 2, num_passed * 2)
        } else {
            // No def levels, should be 1:1 mapping from levels to values
            self.current = start + num_values;
            self.to_slice.slice_with_length(start * 2, num_values * 2)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SpecialRecord {
    /// The position of the special record in the items array
    ///
    /// Note that this is the position in the "expanded" items array (including the specials)
    ///
    /// For example, if we have five items [I0, I1, ..., I4] and two specials [S0(pos=3), S1(pos=6)] then
    /// the combined array is [I0, I1, I2, S0, I3, I4, S1].
    ///
    /// Another tricky fact is that a special "swaps" the repetition level of the matching item when it is
    /// being inserted into the combined list.  So, if items are [I0(rep=2), I1(rep=1), I2(rep=2), I3(rep=0)]
    /// and a special is S0(pos=2, rep=1) then the combined list is
    /// [I0(rep=2), I1(rep=1), S0(rep=2), I2(rep=1), I3(rep=0)].
    ///
    /// Or, to put it in practice we start with [[I0], [I1]], [[I2, I3]] and after inserting our special
    /// we have [[I0], [I1]], [S0, [I2, I3]]
    pos: usize,
    /// The definition level of the special record.  This is never 0 and is used to distinguish between an
    /// empty list and a null list.
    def_level: u16,
    /// The repetition level of the special record.  This is never 0 and is used to indicate which level of
    /// nesting the special record is at.
    rep_level: u16,
}

/// This tells us how an array handles definition.  Given a stack of
/// these and a nested array and a set of definition levels we can calculate
/// how we should interpret the definition levels.
///
/// For example, if the interpretation is [AllValidItem, NullableItem] then
/// a 0 means "valid item" and a 1 means "null struct".  If the interpretation
/// is [NullableItem, NullableItem] then a 0 means "valid item" and a 1 means
/// "null item" and a 2 means "null struct".
///
/// Lists are tricky because we might use up to two definition levels for a
/// single layer of list nesting because we need one value to indicate "empty list"
/// and another value to indicate "null list".
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DefinitionInterpretation {
    AllValidItem,
    AllValidList,
    NullableItem,
    NullableList,
    EmptyableList,
    NullableAndEmptyableList,
}

impl DefinitionInterpretation {
    /// How many definition levels do we need for this layer
    pub fn num_def_levels(&self) -> u16 {
        match self {
            Self::AllValidItem => 0,
            Self::AllValidList => 0,
            Self::NullableItem => 1,
            Self::NullableList => 1,
            Self::EmptyableList => 1,
            Self::NullableAndEmptyableList => 2,
        }
    }

    /// Does this layer have nulls?
    pub fn is_all_valid(&self) -> bool {
        matches!(
            self,
            Self::AllValidItem | Self::AllValidList | Self::EmptyableList
        )
    }

    /// Does this layer represent a list?
    pub fn is_list(&self) -> bool {
        matches!(
            self,
            Self::AllValidList
                | Self::NullableList
                | Self::EmptyableList
                | Self::NullableAndEmptyableList
        )
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
    last_offsets: Option<Vec<usize>>,
    last_offsets_full: Option<Vec<usize>>,
    specials: Vec<SpecialRecord>,
    def_meaning: Vec<DefinitionInterpretation>,
    rep_levels: LevelBuffer,
    def_levels: LevelBuffer,
    current_rep: u16,
    current_def: u16,
    has_nulls: bool,
}

impl SerializerContext {
    fn new(len: usize, has_nulls: bool, has_offsets: bool, num_layers: usize) -> Self {
        let def_meaning = Vec::with_capacity(num_layers);
        Self {
            last_offsets: None,
            last_offsets_full: None,
            rep_levels: if has_offsets {
                vec![0; len]
            } else {
                LevelBuffer::default()
            },
            def_levels: if has_nulls {
                vec![0; len]
            } else {
                LevelBuffer::default()
            },
            def_meaning,
            current_rep: 1,
            current_def: 1,
            has_nulls: false,
            specials: Vec::default(),
        }
    }

    fn checkout_def(&mut self, meaning: DefinitionInterpretation) -> u16 {
        let def = self.current_def;
        self.current_def += meaning.num_def_levels();
        self.def_meaning.push(meaning);
        def
    }

    fn record_offsets(&mut self, offset_desc: &OffsetDesc) {
        let rep_level = self.current_rep;
        let (null_list_level, empty_list_level) =
            match (offset_desc.validity.is_some(), offset_desc.has_empty_lists) {
                (true, true) => {
                    let level =
                        self.checkout_def(DefinitionInterpretation::NullableAndEmptyableList);
                    (level, level + 1)
                }
                (true, false) => (self.checkout_def(DefinitionInterpretation::NullableList), 0),
                (false, true) => (
                    0,
                    self.checkout_def(DefinitionInterpretation::EmptyableList),
                ),
                (false, false) => {
                    self.checkout_def(DefinitionInterpretation::AllValidList);
                    (0, 0)
                }
            };
        self.current_rep += 1;
        if let Some(last_offsets) = &self.last_offsets {
            let last_offsets_full = self.last_offsets_full.as_ref().unwrap();
            let mut new_last_off = Vec::with_capacity(offset_desc.offsets.len());
            let mut new_last_off_full = Vec::with_capacity(offset_desc.offsets.len());
            let mut empties_seen = 0;
            for off in offset_desc.offsets.windows(2) {
                let offset_ctx = last_offsets[off[0] as usize];
                new_last_off.push(offset_ctx);
                new_last_off_full.push(last_offsets_full[off[0] as usize] + empties_seen);
                self.rep_levels[offset_ctx] = rep_level;
                if off[0] == off[1] {
                    empties_seen += 1;
                }
            }
            self.last_offsets = Some(new_last_off);
            self.last_offsets_full = Some(new_last_off_full);
        } else {
            let mut new_last_off = Vec::with_capacity(offset_desc.offsets.len());
            let mut new_last_off_full = Vec::with_capacity(offset_desc.offsets.len());
            let mut empties_seen = 0;
            for off in offset_desc.offsets.windows(2) {
                self.rep_levels[off[0] as usize] = rep_level;
                new_last_off.push(off[0] as usize);
                new_last_off_full.push(off[0] as usize + empties_seen);
                if off[0] == off[1] {
                    empties_seen += 1;
                }
            }
            self.last_offsets = Some(new_last_off);
            self.last_offsets_full = Some(new_last_off_full);
        }

        // Must update specials _after_ setting last_offsets_full
        let last_offsets_full = self.last_offsets_full.as_ref().unwrap();
        let num_combined_specials = self.specials.len() + offset_desc.specials.len();
        let mut new_specials = Vec::with_capacity(num_combined_specials);
        let mut new_inserted = 0;
        let mut old_specials_itr = self.specials.iter().peekable();
        let mut specials_itr = offset_desc.specials.iter().peekable();
        for _ in 0..num_combined_specials {
            if let Some(old_special) = old_specials_itr.peek() {
                let old_special_pos = old_special.pos + new_inserted;
                if let Some(new_special) = specials_itr.peek() {
                    let new_special_pos = last_offsets_full[new_special.pos()];
                    if old_special_pos < new_special_pos {
                        let mut old_special = *old_specials_itr.next().unwrap();
                        old_special.pos = old_special_pos;
                        new_specials.push(old_special);
                    } else {
                        let new_special = specials_itr.next().unwrap();
                        new_specials.push(SpecialRecord {
                            pos: new_special_pos,
                            def_level: if matches!(new_special, SpecialOffset::EmptyList(_)) {
                                empty_list_level
                            } else {
                                null_list_level
                            },
                            rep_level,
                        });
                        new_inserted += 1;
                    }
                } else {
                    let mut old_special = *old_specials_itr.next().unwrap();
                    old_special.pos = old_special_pos;
                    new_specials.push(old_special);
                }
            } else {
                let new_special = specials_itr.next().unwrap();
                new_specials.push(SpecialRecord {
                    pos: last_offsets_full[new_special.pos()],
                    def_level: if matches!(new_special, SpecialOffset::EmptyList(_)) {
                        empty_list_level
                    } else {
                        null_list_level
                    },
                    rep_level,
                });
                new_inserted += 1;
            }
        }
        self.specials = new_specials;
    }

    fn do_record_validity(&mut self, validity: &BooleanBuffer, null_level: u16) {
        self.has_nulls = true;
        assert!(!self.def_levels.is_empty());
        if let Some(last_offsets) = &self.last_offsets {
            last_offsets
                .windows(2)
                .zip(validity.iter())
                .for_each(|(w, valid)| {
                    if !valid {
                        self.def_levels[w[0]..w[1]].fill(null_level);
                    }
                });
        } else {
            self.def_levels
                .iter_mut()
                .zip(validity.iter())
                .for_each(|(def, valid)| {
                    if !valid {
                        *def = null_level;
                    }
                });
        }
    }

    fn record_validity(&mut self, validity_desc: &ValidityDesc) {
        if let Some(validity) = validity_desc.validity.as_ref() {
            let def_level = self.checkout_def(DefinitionInterpretation::NullableItem);
            self.do_record_validity(validity, def_level);
        } else {
            self.checkout_def(DefinitionInterpretation::AllValidItem);
        }
    }

    fn build(self) -> SerializedRepDefs {
        let definition_levels = if self.has_nulls {
            Some(self.def_levels)
        } else {
            None
        };
        let repetition_levels = if self.current_rep > 1 {
            Some(self.rep_levels)
        } else {
            None
        };
        SerializedRepDefs::new(
            repetition_levels,
            definition_levels,
            self.specials,
            self.def_meaning,
        )
    }
}

/// As we are encoding we record information about "specials" which are
/// empty lists or null lists.
#[derive(Debug, Copy, Clone)]
enum SpecialOffset {
    NullList(usize),
    EmptyList(usize),
}

impl SpecialOffset {
    fn pos(&self) -> usize {
        match self {
            Self::NullList(pos) => *pos,
            Self::EmptyList(pos) => *pos,
        }
    }
}

/// A structure used to collect validity buffers and offsets from arrow
/// arrays and eventually create repetition and definition levels
///
/// As we are encoding the structural encoders are given this struct and
/// will record the arrow information into it.  Once we hit a leaf node we
/// serialize the data into rep/def levels and write these into the page.
#[derive(Clone, Default, Debug)]
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

    /// The builder is "empty" if there is no repetition and no nulls.  In this case we don't need
    /// to store anything to disk (except the description)
    fn is_empty(&self) -> bool {
        self.repdefs
            .iter()
            .all(|r| matches!(r, RawRepDef::Validity(ValidityDesc { validity: None, .. })))
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
        self.repdefs.iter().any(|rd| {
            matches!(
                rd,
                RawRepDef::Validity(ValidityDesc {
                    validity: Some(_),
                    ..
                })
            )
        })
    }

    pub fn has_offsets(&self) -> bool {
        self.repdefs
            .iter()
            .any(|rd| matches!(rd, RawRepDef::Offsets(OffsetDesc { .. })))
    }

    /// Registers a nullable validity bitmap
    pub fn add_validity_bitmap(&mut self, validity: NullBuffer) {
        self.check_validity_len(&validity);
        self.repdefs.push(RawRepDef::Validity(ValidityDesc {
            num_values: validity.len(),
            validity: Some(validity.into_inner()),
        }));
    }

    /// Registers an all-valid validity layer
    pub fn add_no_null(&mut self, len: usize) {
        self.repdefs.push(RawRepDef::Validity(ValidityDesc {
            validity: None,
            num_values: len,
        }));
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
    pub fn add_offsets<O: OffsetSizeTrait>(
        &mut self,
        repetition: OffsetBuffer<O>,
        validity: Option<NullBuffer>,
    ) {
        // We should be able to zero-copy
        if O::IS_LARGE {
            let inner = repetition.into_inner();
            let len = inner.len();
            let i64_buff = ScalarBuffer::new(inner.into_inner(), 0, len);
            let offsets = Vec::from(i64_buff);
            let mut specials = Vec::new();
            let mut has_empty_lists = false;
            if let Some(validity) = validity.as_ref() {
                for (idx, (_, valid)) in offsets
                    .windows(2)
                    .zip(validity.iter())
                    .enumerate()
                    .filter(|(_, (off, _))| off[0] == off[1])
                {
                    if valid {
                        has_empty_lists = true;
                        specials.push(SpecialOffset::EmptyList(idx));
                    } else {
                        specials.push(SpecialOffset::NullList(idx));
                    }
                }
            } else {
                for (idx, _) in offsets
                    .windows(2)
                    .enumerate()
                    .filter(|(_, off)| off[0] == off[1])
                {
                    has_empty_lists = true;
                    specials.push(SpecialOffset::EmptyList(idx));
                }
            };
            self.check_offset_len(&offsets);
            self.repdefs.push(RawRepDef::Offsets(OffsetDesc {
                num_values: offsets.len() - 1,
                offsets: offsets.into(),
                validity: validity.map(|v| v.into_inner()),
                has_empty_lists,
                specials: specials.into(),
            }));
        } else {
            let inner = repetition.into_inner();
            let len = inner.len();
            let mut casted = Vec::with_capacity(len);
            let mut has_empty_lists = false;
            let mut specials = Vec::new();
            if let Some(validity) = validity.as_ref() {
                let scalar_off = ScalarBuffer::<i32>::new(inner.into_inner(), 0, len);
                for (idx, (off, valid)) in scalar_off.windows(2).zip(validity.iter()).enumerate() {
                    if off[0] == off[1] {
                        if valid {
                            has_empty_lists = true;
                            specials.push(SpecialOffset::EmptyList(idx));
                        } else {
                            specials.push(SpecialOffset::NullList(idx));
                        }
                    }
                    casted.push(off[0] as i64);
                }
                casted.push(*scalar_off.last().unwrap() as i64);
            } else {
                let scalar_off = ScalarBuffer::<i32>::new(inner.into_inner(), 0, len);
                for (idx, off) in scalar_off.windows(2).enumerate() {
                    if off[0] == off[1] {
                        has_empty_lists = true;
                        specials.push(SpecialOffset::EmptyList(idx));
                    }
                    casted.push(off[0] as i64);
                }
                casted.push(*scalar_off.last().unwrap() as i64);
            };
            self.check_offset_len(&casted);
            self.repdefs.push(RawRepDef::Offsets(OffsetDesc {
                num_values: casted.len() - 1,
                offsets: casted.into(),
                validity: validity.map(|v| v.into_inner()),
                has_empty_lists,
                specials: specials.into(),
            }));
        }
    }

    // When we are encoding data it arrives in batches.  For each batch we create a RepDefBuilder and collect the
    // various validity buffers and offset buffers from that batch.  Once we have enough batches to write a page we
    // need to take this collection of RepDefBuilders and concatenate them and then serialize them into rep/def levels.
    //
    // TODO: In the future, we may concatenate and serialize at the same time?
    //
    // This method takes care of the concatenation part.  First we collect all of layer 0 from each builder, then we
    // call this method.  Then we collect all of layer 1 from each builder and call this method.  And so on.
    //
    // That means this method should get a collection of `RawRepDef` where each item is the same kind (all validity or
    // all offsets) though the nullability / lengths may be different in each layer.
    fn concat_layers<'a>(
        layers: impl Iterator<Item = &'a RawRepDef>,
        num_layers: usize,
    ) -> RawRepDef {
        // We make two passes through the layers.  The first determines if we need to pay the cost of allocating
        // buffers.  The second pass actually adds the values.
        let mut collected = Vec::with_capacity(num_layers);
        let mut has_nulls = false;
        let mut is_offsets = false;
        let mut num_specials = 0;
        let mut all_has_empty_lists = false;
        let mut all_num_values = 0;
        for layer in layers {
            has_nulls |= layer.has_nulls();
            if let RawRepDef::Offsets(OffsetDesc {
                specials,
                has_empty_lists,
                ..
            }) = layer
            {
                all_has_empty_lists |= *has_empty_lists;
                is_offsets = true;
                num_specials += specials.len();
            }
            collected.push(layer);
            all_num_values += layer.num_values();
        }

        // Shortcut if there are no nulls
        if !has_nulls && !is_offsets {
            return RawRepDef::Validity(ValidityDesc {
                validity: None,
                num_values: all_num_values,
            });
        }

        // Only allocate if needed
        let mut validity_builder = if has_nulls {
            BooleanBufferBuilder::new(all_num_values)
        } else {
            BooleanBufferBuilder::new(0)
        };
        let mut all_offsets = if is_offsets {
            let mut all_offsets = Vec::with_capacity(all_num_values);
            all_offsets.push(0);
            all_offsets
        } else {
            Vec::new()
        };
        let mut all_specials = Vec::with_capacity(num_specials);

        for layer in collected {
            match layer {
                RawRepDef::Validity(ValidityDesc {
                    validity: Some(validity),
                    ..
                }) => {
                    validity_builder.append_buffer(validity);
                }
                RawRepDef::Validity(ValidityDesc {
                    validity: None,
                    num_values,
                }) => {
                    validity_builder.append_n(*num_values, true);
                }
                RawRepDef::Offsets(OffsetDesc {
                    offsets,
                    validity: Some(validity),
                    has_empty_lists,
                    specials,
                    ..
                }) => {
                    all_has_empty_lists |= has_empty_lists;
                    validity_builder.append_buffer(validity);
                    let existing_lists = all_offsets.len() - 1;
                    let last = *all_offsets.last().unwrap();
                    all_offsets.extend(offsets.iter().skip(1).map(|off| *off + last));
                    all_specials.extend(specials.iter().map(|s| match s {
                        SpecialOffset::NullList(pos) => {
                            SpecialOffset::NullList(*pos + existing_lists)
                        }
                        SpecialOffset::EmptyList(pos) => {
                            SpecialOffset::EmptyList(*pos + existing_lists)
                        }
                    }));
                }
                RawRepDef::Offsets(OffsetDesc {
                    offsets,
                    validity: None,
                    has_empty_lists,
                    num_values,
                    specials,
                }) => {
                    all_has_empty_lists |= has_empty_lists;
                    if has_nulls {
                        validity_builder.append_n(*num_values, true);
                    }
                    let last = *all_offsets.last().unwrap();
                    let existing_lists = all_offsets.len() - 1;
                    all_offsets.extend(offsets.iter().skip(1).map(|off| *off + last));
                    all_specials.extend(specials.iter().map(|s| match s {
                        SpecialOffset::NullList(pos) => {
                            SpecialOffset::NullList(*pos + existing_lists)
                        }
                        SpecialOffset::EmptyList(pos) => {
                            SpecialOffset::EmptyList(*pos + existing_lists)
                        }
                    }));
                }
            }
        }
        let validity = if has_nulls {
            Some(validity_builder.finish())
        } else {
            None
        };
        if all_offsets.is_empty() {
            RawRepDef::Validity(ValidityDesc {
                validity,
                num_values: all_num_values,
            })
        } else {
            RawRepDef::Offsets(OffsetDesc {
                offsets: all_offsets.into(),
                validity,
                has_empty_lists: all_has_empty_lists,
                num_values: all_num_values,
                specials: all_specials.into(),
            })
        }
    }

    /// Converts the validity / offsets buffers that have been gathered so far
    /// into repetition and definition levels
    pub fn serialize(builders: Vec<Self>) -> SerializedRepDefs {
        assert!(!builders.is_empty());
        if builders.iter().all(|b| b.is_empty()) {
            // No repetition, all-valid
            return SerializedRepDefs::empty(
                builders
                    .first()
                    .unwrap()
                    .repdefs
                    .iter()
                    .map(|_| DefinitionInterpretation::AllValidItem)
                    .collect::<Vec<_>>(),
            );
        }
        let has_nulls = builders.iter().any(|b| b.has_nulls());
        let has_offsets = builders.iter().any(|b| b.has_offsets());
        let total_len = builders.iter().map(|b| b.len.unwrap()).sum();
        let num_layers = builders[0].num_layers();
        let mut context = SerializerContext::new(total_len, has_nulls, has_offsets, num_layers);
        let combined_layers = (0..num_layers)
            .map(|layer_index| {
                Self::concat_layers(
                    builders.iter().map(|b| &b.repdefs[layer_index]),
                    builders.len(),
                )
            })
            .collect::<Vec<_>>();
        debug_assert!(builders
            .iter()
            .all(|b| b.num_layers() == builders[0].num_layers()));
        for layer in combined_layers.into_iter().rev() {
            match layer {
                RawRepDef::Validity(def) => {
                    context.record_validity(&def);
                }
                RawRepDef::Offsets(rep) => {
                    context.record_offsets(&rep);
                }
            }
        }
        context.build().collapse_specials()
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
    // Maps from definition level to the rep level at which that definition level is visible
    levels_to_rep: Vec<u16>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    // Current definition level to compare to.
    current_def_cmp: u16,
    // Current rep level, determines which specials we can see
    current_rep_cmp: u16,
    // Current layer index, 0 means inner-most layer and it counts up from there.  Used to index
    // into special_defs
    current_layer: usize,
}

impl RepDefUnraveler {
    /// Creates a new unraveler from serialized repetition and definition information
    pub fn new(
        rep_levels: Option<LevelBuffer>,
        def_levels: Option<LevelBuffer>,
        def_meaning: Arc<[DefinitionInterpretation]>,
    ) -> Self {
        let mut levels_to_rep = Vec::with_capacity(def_meaning.len());
        let mut rep_counter = 0;
        // Level=0 is always visible and means valid item
        levels_to_rep.push(0);
        for meaning in def_meaning.as_ref() {
            match meaning {
                DefinitionInterpretation::AllValidItem | DefinitionInterpretation::AllValidList => {
                    // There is no corresponding level, so nothing to put in levels_to_rep
                }
                DefinitionInterpretation::NullableItem => {
                    // Some null structs are not visible at inner rep levels in cases like LIST<STRUCT<LIST<...>>>
                    levels_to_rep.push(rep_counter);
                }
                DefinitionInterpretation::NullableList => {
                    rep_counter += 1;
                    levels_to_rep.push(rep_counter);
                }
                DefinitionInterpretation::EmptyableList => {
                    rep_counter += 1;
                    levels_to_rep.push(rep_counter);
                }
                DefinitionInterpretation::NullableAndEmptyableList => {
                    rep_counter += 1;
                    levels_to_rep.push(rep_counter);
                    levels_to_rep.push(rep_counter);
                }
            }
        }
        Self {
            rep_levels,
            def_levels,
            current_def_cmp: 0,
            current_rep_cmp: 0,
            levels_to_rep,
            current_layer: 0,
            def_meaning,
        }
    }

    pub fn is_all_valid(&self) -> bool {
        self.def_meaning[self.current_layer].is_all_valid()
    }

    /// If the current level is a repetition layer then this returns the number of lists
    /// at this level.
    ///
    /// This is not valid to call when the current level is a struct/primitive layer because
    /// in some cases there may be no rep or def information to know this.
    pub fn max_lists(&self) -> usize {
        debug_assert!(
            self.def_meaning[self.current_layer] != DefinitionInterpretation::NullableItem
        );
        self.rep_levels
            .as_ref()
            // Worst case every rep item is max_rep and a new list
            .map(|levels| levels.len())
            .unwrap_or(0)
    }

    /// Unravels a layer of offsets from the unraveler into the given offset width
    ///
    /// When decoding a list the caller should first unravel the offsets and then
    /// unravel the validity (this is the opposite order used during encoding)
    pub fn unravel_offsets<T: ArrowNativeType>(
        &mut self,
        offsets: &mut Vec<T>,
        validity: Option<&mut BooleanBufferBuilder>,
    ) -> Result<()> {
        let rep_levels = self
            .rep_levels
            .as_mut()
            .expect("Expected repetition level but data didn't contain repetition");
        let valid_level = self.current_def_cmp;
        let (null_level, empty_level) = match self.def_meaning[self.current_layer] {
            DefinitionInterpretation::NullableList => {
                self.current_def_cmp += 1;
                (valid_level + 1, 0)
            }
            DefinitionInterpretation::EmptyableList => {
                self.current_def_cmp += 1;
                (0, valid_level + 1)
            }
            DefinitionInterpretation::NullableAndEmptyableList => {
                self.current_def_cmp += 2;
                (valid_level + 1, valid_level + 2)
            }
            DefinitionInterpretation::AllValidList => (0, 0),
            _ => unreachable!(),
        };
        let max_level = null_level.max(empty_level);
        self.current_layer += 1;

        let mut curlen: usize = offsets.last().map(|o| o.as_usize()).unwrap_or(0);

        // If offsets is empty this is a no-op.  If offsets is not empty that means we already
        // added a set of offsets.  For example, we might have added [0, 3, 5] (2 lists).  Now
        // say we want to add [0, 1, 4] (2 lists).  We should get [0, 3, 5, 6, 9] (4 lists).  If
        // we don't pop here we get [0, 3, 5, 5, 6, 9] which is wrong.
        //
        // Or, to think about it another way, if every unraveler adds the starting 0 and the trailing
        // length then we have N + unravelers.len() values instead of N + 1.
        offsets.pop();

        let to_offset = |val: usize| {
            T::from_usize(val)
            .ok_or_else(|| Error::invalid_input("A single batch had more than i32::MAX values and so a large container type is required", location!()))
        };
        self.current_rep_cmp += 1;
        if let Some(def_levels) = &mut self.def_levels {
            assert!(rep_levels.len() == def_levels.len());
            // It's possible validity is None even if we have def levels.  For example, we might have
            // empty lists (which require def levels) but no nulls.
            let mut push_validity: Box<dyn FnMut(bool)> = if let Some(validity) = validity {
                Box::new(|is_valid| validity.append(is_valid))
            } else {
                Box::new(|_| {})
            };
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
                        let def_val = *def_levels.get_unchecked(read_idx);
                        // Copy over
                        *rep_levels.get_unchecked_mut(write_idx) = rep_val - 1;
                        *def_levels.get_unchecked_mut(write_idx) = def_val;
                        write_idx += 1;

                        if def_val == 0 {
                            // This is a valid list
                            offsets.push(to_offset(curlen)?);
                            curlen += 1;
                            push_validity(true);
                        } else if def_val > max_level {
                            // This is not visible at this rep level, do not add to offsets, but keep in repdef
                        } else if def_val == null_level {
                            // This is a null list
                            offsets.push(to_offset(curlen)?);
                            push_validity(false);
                        } else if def_val == empty_level {
                            // This is an empty list
                            offsets.push(to_offset(curlen)?);
                            push_validity(true);
                        } else {
                            // New valid list starting with null item
                            offsets.push(to_offset(curlen)?);
                            curlen += 1;
                            push_validity(true);
                        }
                    } else {
                        curlen += 1;
                    }
                    read_idx += 1;
                }
            }
            offsets.push(to_offset(curlen)?);
            rep_levels.truncate(write_idx);
            def_levels.truncate(write_idx);
            Ok(())
        } else {
            // SAFETY: See above loop
            let mut read_idx = 0;
            let mut write_idx = 0;
            let old_offsets_len = offsets.len();
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
            let num_new_lists = offsets.len() - old_offsets_len;
            offsets.push(to_offset(curlen)?);
            rep_levels.truncate(offsets.len() - 1);
            if let Some(validity) = validity {
                // Even though we don't have validity it is possible another unraveler did and so we need
                // to push all valids
                validity.append_n(num_new_lists, true);
            }
            Ok(())
        }
    }

    pub fn skip_validity(&mut self) {
        debug_assert!(
            self.def_meaning[self.current_layer] == DefinitionInterpretation::AllValidItem
        );
        self.current_layer += 1;
    }

    /// Unravels a layer of validity from the definition levels
    pub fn unravel_validity(&mut self, validity: &mut BooleanBufferBuilder) {
        debug_assert!(
            self.def_meaning[self.current_layer] != DefinitionInterpretation::AllValidItem
        );
        self.current_layer += 1;

        let def_levels = &self.def_levels.as_ref().unwrap();

        let current_def_cmp = self.current_def_cmp;
        self.current_def_cmp += 1;

        for is_valid in def_levels.iter().filter_map(|&level| {
            if self.levels_to_rep[level as usize] <= self.current_rep_cmp {
                Some(level <= current_def_cmp)
            } else {
                None
            }
        }) {
            validity.append(is_valid);
        }
    }
}

/// As we decode we may extract rep/def information from multiple pages (or multiple
/// chunks within a page).
///
/// For each chunk we create an unraveler.  Each unraveler can have a completely different
/// interpretation (e.g. one page might contain null items but no null structs and the next
/// page might have null structs but no null items).
///
/// Concatenating these unravelers would be tricky and expensive so instead we have a
/// composite unraveler which unravels across multiple unravelers.
///
/// Note: this class should be used even if there is only one page / unraveler.  This is
/// because the `RepDefUnraveler`'s API is more complex (it's meant to be called by this
/// class)
#[derive(Debug)]
pub struct CompositeRepDefUnraveler {
    unravelers: Vec<RepDefUnraveler>,
}

impl CompositeRepDefUnraveler {
    pub fn new(unravelers: Vec<RepDefUnraveler>) -> Self {
        Self { unravelers }
    }

    /// Unravels a layer of validity
    ///
    /// Returns None if there are no null items in this layer
    pub fn unravel_validity(&mut self, num_values: usize) -> Option<NullBuffer> {
        let is_all_valid = self
            .unravelers
            .iter()
            .all(|unraveler| unraveler.is_all_valid());

        if is_all_valid {
            for unraveler in self.unravelers.iter_mut() {
                unraveler.skip_validity();
            }
            None
        } else {
            let mut validity = BooleanBufferBuilder::new(num_values);
            for unraveler in self.unravelers.iter_mut() {
                unraveler.unravel_validity(&mut validity);
            }
            Some(NullBuffer::new(validity.finish()))
        }
    }

    /// Unravels a layer of offsets (and the validity for that layer)
    pub fn unravel_offsets<T: ArrowNativeType>(
        &mut self,
    ) -> Result<(OffsetBuffer<T>, Option<NullBuffer>)> {
        let mut is_all_valid = true;
        let mut max_num_lists = 0;
        for unraveler in self.unravelers.iter() {
            is_all_valid &= unraveler.is_all_valid();
            max_num_lists += unraveler.max_lists();
        }

        let mut validity = if is_all_valid {
            None
        } else {
            // Note: This is probably an over-estimate and potentially even an under-estimate.  We only know
            // right now how many items we have and not how many rows.  (TODO: Shouldn't we know the # of rows?)
            Some(BooleanBufferBuilder::new(max_num_lists))
        };

        let mut offsets = Vec::with_capacity(max_num_lists + 1);

        for unraveler in self.unravelers.iter_mut() {
            unraveler.unravel_offsets(&mut offsets, validity.as_mut())?;
        }

        Ok((
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            validity.map(|mut v| NullBuffer::new(v.finish())),
        ))
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

// We're really going out of our way to avoid boxing here but this will be called on a per-value basis
// so it is in the critical path.
type SpecificBinaryControlWordIterator<'a, T> = BinaryControlWordIterator<
    Zip<Copied<std::slice::Iter<'a, u16>>, Copied<std::slice::Iter<'a, u16>>>,
    T,
>;

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
pub enum ControlWordIterator<'a> {
    Binary8(SpecificBinaryControlWordIterator<'a, u8>),
    Binary16(SpecificBinaryControlWordIterator<'a, u16>),
    Binary32(SpecificBinaryControlWordIterator<'a, u32>),
    Unary8(UnaryControlWordIterator<Copied<std::slice::Iter<'a, u16>>, u8>),
    Unary16(UnaryControlWordIterator<Copied<std::slice::Iter<'a, u16>>, u16>),
    Unary32(UnaryControlWordIterator<Copied<std::slice::Iter<'a, u16>>, u32>),
    Nilary(NilaryControlWordIterator),
}

impl ControlWordIterator<'_> {
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
pub fn build_control_word_iterator<'a>(
    rep: Option<&'a [u16]>,
    max_rep: u16,
    def: Option<&'a [u16]>,
    max_def: u16,
) -> ControlWordIterator<'a> {
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
            let iter = rep.iter().copied().zip(def.iter().copied());
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
            let iter = lev.iter().copied();
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
            let iter = lev.iter().copied();
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

    use crate::repdef::{
        CompositeRepDefUnraveler, DefinitionInterpretation, RepDefUnraveler, SerializedRepDefs,
    };

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
    fn test_repdef_basic() {
        // Basic case, rep & def
        let mut builder = RepDefBuilder::default();
        builder.add_offsets(
            offsets_64(&[0, 2, 2, 5]),
            Some(validity(&[true, false, true])),
        );
        builder.add_offsets(
            offsets_64(&[0, 1, 3, 5, 5, 9]),
            Some(validity(&[true, true, true, false, true])),
        );
        builder.add_validity_bitmap(validity(&[
            true, true, true, false, false, false, true, true, false,
        ]));

        let repdefs = RepDefBuilder::serialize(vec![builder]);
        let rep = repdefs.repetition_levels.unwrap();
        let def = repdefs.definition_levels.unwrap();

        assert_eq!(vec![0, 0, 0, 3, 1, 1, 2, 1, 0, 0, 1], *def);
        assert_eq!(vec![2, 1, 0, 2, 2, 0, 1, 1, 0, 0, 0], *rep);

        let mut unraveler = CompositeRepDefUnraveler::new(vec![RepDefUnraveler::new(
            Some(rep.as_ref().to_vec()),
            Some(def.as_ref().to_vec()),
            repdefs.def_meaning.into(),
        )]);

        // Note: validity doesn't exactly round-trip because repdef normalizes some of the
        // redundant validity values
        assert_eq!(
            unraveler.unravel_validity(9),
            Some(validity(&[
                true, true, true, false, false, false, true, true, false
            ]))
        );
        let (off, val) = unraveler.unravel_offsets::<i32>().unwrap();
        assert_eq!(off.inner(), offsets_32(&[0, 1, 3, 5, 5, 9]).inner());
        assert_eq!(val, Some(validity(&[true, true, true, false, true])));
        let (off, val) = unraveler.unravel_offsets::<i32>().unwrap();
        assert_eq!(off.inner(), offsets_32(&[0, 2, 2, 5]).inner());
        assert_eq!(val, Some(validity(&[true, false, true])));
    }

    #[test]
    fn test_repdef_simple_null_empty_list() {
        let check = |repdefs: SerializedRepDefs, last_def: DefinitionInterpretation| {
            let rep = repdefs.repetition_levels.unwrap();
            let def = repdefs.definition_levels.unwrap();

            assert_eq!([1, 0, 1, 1, 0, 0], *rep);
            assert_eq!([0, 0, 2, 0, 1, 0], *def);
            assert!(repdefs.special_records.is_empty());
            assert_eq!(
                vec![DefinitionInterpretation::NullableItem, last_def,],
                repdefs.def_meaning
            );
        };

        // Null list and empty list should be serialized mostly the same

        // Null case
        let mut builder = RepDefBuilder::default();
        builder.add_offsets(
            offsets_32(&[0, 2, 2, 5]),
            Some(validity(&[true, false, true])),
        );
        builder.add_validity_bitmap(validity(&[true, true, true, false, true]));

        let repdefs = RepDefBuilder::serialize(vec![builder]);

        check(repdefs, DefinitionInterpretation::NullableList);

        // Empty case
        let mut builder = RepDefBuilder::default();
        builder.add_offsets(offsets_32(&[0, 2, 2, 5]), None);
        builder.add_validity_bitmap(validity(&[true, true, true, false, true]));

        let repdefs = RepDefBuilder::serialize(vec![builder]);

        check(repdefs, DefinitionInterpretation::EmptyableList);
    }

    #[test]
    fn test_repdef_complex_null_empty() {
        let mut builder = RepDefBuilder::default();
        builder.add_offsets(
            offsets_32(&[0, 4, 4, 4, 6]),
            Some(validity(&[true, false, true, true])),
        );
        builder.add_offsets(
            offsets_32(&[0, 1, 1, 2, 2, 2, 3]),
            Some(validity(&[true, false, true, false, true, true])),
        );
        builder.add_no_null(3);

        let repdefs = RepDefBuilder::serialize(vec![builder]);

        let rep = repdefs.repetition_levels.unwrap();
        let def = repdefs.definition_levels.unwrap();

        assert_eq!([2, 1, 1, 1, 2, 2, 2, 1], *rep);
        assert_eq!([0, 1, 0, 1, 3, 4, 2, 0], *def);
    }

    #[test]
    fn test_repdef_empty_list_no_null() {
        // Tests when we have some empty lists but no null lists.  This case
        // caused some bugs because we have definition but no nulls
        let mut builder = RepDefBuilder::default();
        builder.add_offsets(offsets_32(&[0, 4, 4, 4, 6]), None);
        builder.add_no_null(6);

        let repdefs = RepDefBuilder::serialize(vec![builder]);

        let rep = repdefs.repetition_levels.unwrap();
        let def = repdefs.definition_levels.unwrap();

        assert_eq!([1, 0, 0, 0, 1, 1, 1, 0], *rep);
        assert_eq!([0, 0, 0, 0, 1, 1, 0, 0], *def);

        let mut unraveler = CompositeRepDefUnraveler::new(vec![RepDefUnraveler::new(
            Some(rep.as_ref().to_vec()),
            Some(def.as_ref().to_vec()),
            repdefs.def_meaning.into(),
        )]);

        assert_eq!(unraveler.unravel_validity(6), None);
        let (off, val) = unraveler.unravel_offsets::<i32>().unwrap();
        assert_eq!(off.inner(), offsets_32(&[0, 4, 4, 4, 6]).inner());
        assert_eq!(val, None);
    }

    #[test]
    fn test_repdef_all_valid() {
        let mut builder = RepDefBuilder::default();
        builder.add_offsets(offsets_64(&[0, 2, 3, 5]), None);
        builder.add_offsets(offsets_64(&[0, 1, 3, 5, 7, 9]), None);
        builder.add_no_null(9);

        let repdefs = RepDefBuilder::serialize(vec![builder]);
        let rep = repdefs.repetition_levels.unwrap();
        assert!(repdefs.definition_levels.is_none());

        assert_eq!([2, 1, 0, 2, 0, 2, 0, 1, 0], *rep);

        let mut unraveler = CompositeRepDefUnraveler::new(vec![RepDefUnraveler::new(
            Some(rep.as_ref().to_vec()),
            None,
            repdefs.def_meaning.into(),
        )]);

        assert_eq!(unraveler.unravel_validity(9), None);
        let (off, val) = unraveler.unravel_offsets::<i32>().unwrap();
        assert_eq!(off.inner(), offsets_32(&[0, 1, 3, 5, 7, 9]).inner());
        assert_eq!(val, None);
        let (off, val) = unraveler.unravel_offsets::<i32>().unwrap();
        assert_eq!(off.inner(), offsets_32(&[0, 2, 3, 5]).inner());
        assert_eq!(val, None);
    }

    #[test]
    fn test_repdef_no_rep() {
        let mut builder = RepDefBuilder::default();
        builder.add_no_null(5);
        builder.add_validity_bitmap(validity(&[false, false, true, true, true]));
        builder.add_validity_bitmap(validity(&[false, true, true, true, false]));

        let repdefs = RepDefBuilder::serialize(vec![builder]);
        assert!(repdefs.repetition_levels.is_none());
        let def = repdefs.definition_levels.unwrap();

        assert_eq!([2, 2, 0, 0, 1], *def);

        let mut unraveler = CompositeRepDefUnraveler::new(vec![RepDefUnraveler::new(
            None,
            Some(def.as_ref().to_vec()),
            repdefs.def_meaning.into(),
        )]);

        assert_eq!(
            unraveler.unravel_validity(5),
            Some(validity(&[false, false, true, true, false]))
        );
        assert_eq!(
            unraveler.unravel_validity(5),
            Some(validity(&[false, false, true, true, true]))
        );
        assert_eq!(unraveler.unravel_validity(5), None);
    }

    #[test]
    fn test_composite_unravel() {
        let mut builder = RepDefBuilder::default();
        builder.add_offsets(
            offsets_64(&[0, 2, 2, 5]),
            Some(validity(&[true, false, true])),
        );
        let repdef1 = RepDefBuilder::serialize(vec![builder]);

        let mut builder = RepDefBuilder::default();
        builder.add_offsets(offsets_64(&[0, 1, 3, 5, 7, 9]), None);
        let repdef2 = RepDefBuilder::serialize(vec![builder]);

        let unravel1 = RepDefUnraveler::new(
            repdef1.repetition_levels.map(|l| l.to_vec()),
            repdef1.definition_levels.map(|l| l.to_vec()),
            repdef1.def_meaning.into(),
        );
        let unravel2 = RepDefUnraveler::new(
            repdef2.repetition_levels.map(|l| l.to_vec()),
            repdef2.definition_levels.map(|l| l.to_vec()),
            repdef2.def_meaning.into(),
        );

        let mut unraveler = CompositeRepDefUnraveler::new(vec![unravel1, unravel2]);

        let (off, val) = unraveler.unravel_offsets::<i32>().unwrap();
        assert_eq!(
            off.inner(),
            offsets_32(&[0, 2, 2, 5, 6, 8, 10, 12, 14]).inner()
        );
        assert_eq!(
            val,
            Some(validity(&[true, false, true, true, true, true, true, true]))
        );
    }

    #[test]
    fn test_repdef_multiple_builders() {
        // Basic case, rep & def
        let mut builder1 = RepDefBuilder::default();
        builder1.add_offsets(offsets_64(&[0, 2]), None);
        builder1.add_offsets(offsets_64(&[0, 1, 3]), None);
        builder1.add_validity_bitmap(validity(&[true, true, true]));

        let mut builder2 = RepDefBuilder::default();
        builder2.add_offsets(offsets_64(&[0, 0, 3]), Some(validity(&[false, true])));
        builder2.add_offsets(
            offsets_64(&[0, 2, 2, 6]),
            Some(validity(&[true, false, true])),
        );
        builder2.add_validity_bitmap(validity(&[false, false, false, true, true, false]));

        let repdefs = RepDefBuilder::serialize(vec![builder1, builder2]);

        let rep = repdefs.repetition_levels.unwrap();
        let def = repdefs.definition_levels.unwrap();

        assert_eq!([2, 1, 0, 2, 2, 0, 1, 1, 0, 0, 0], *rep);
        assert_eq!([0, 0, 0, 3, 1, 1, 2, 1, 0, 0, 1], *def);
    }

    #[test]
    fn test_control_words() {
        // Convert to control words, verify expected, convert back, verify same as original
        fn check(
            rep: &[u16],
            def: &[u16],
            expected_values: Vec<u8>,
            expected_bytes_per_word: usize,
            expected_bits_rep: u8,
            expected_bits_def: u8,
        ) {
            let num_vals = rep.len().max(def.len());
            let max_rep = rep.iter().max().copied().unwrap_or(0);
            let max_def = def.iter().max().copied().unwrap_or(0);

            let in_rep = if rep.is_empty() { None } else { Some(rep) };
            let in_def = if def.is_empty() { None } else { Some(def) };

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

            assert_eq!(rep, rep_out.as_slice());
            assert_eq!(def, def_out.as_slice());
        }

        // Each will need 4 bits and so we should get 1-byte control words
        let rep = &[0_u16, 7, 3, 2, 9, 8, 12, 5];
        let def = &[5_u16, 3, 1, 2, 12, 15, 0, 2];
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
        let rep = &[0_u16, 7, 3, 2, 9, 8, 12, 5];
        let def = &[5_u16, 3, 1, 2, 12, 22, 0, 2];
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
        let levels = &[0_u16, 7, 3, 2, 9, 8, 12, 5];
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
        check(levels, &[], expected.clone(), 1, 4, 0);

        // Just def
        check(&[], levels, expected, 1, 0, 4);

        // No rep, no def, no bytes
        check(&[], &[], Vec::default(), 0, 0, 0);
    }
}
