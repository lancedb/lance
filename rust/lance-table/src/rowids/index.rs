// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::BTreeMap, ops::{Range, RangeInclusive}};

use lance_core::utils::address::RowAddress;
use rangemap::{RangeInclusiveMap, RangeMap};

use super::{RowIdSequence, U64Segment};

/// An index of row ids
/// 
/// This index is used to map row ids to their corresponding addresses.
/// 
/// This is structured as a B tree, where the keys and values can either
/// be a single value or a range.
pub struct RowIdIndex(RangeInclusiveMap<u64, (U64Segment, U64Segment)>);

impl RowIdIndex {
    /// Create a new index from a list of fragment ids and their corresponding row id sequences.
    pub fn new(fragment_indices: &[(u32, RowIdSequence)]) -> Self {
        let mut pieces = fragment_indices
            .iter()
            .flat_map(|(fragment_id, sequence)| decompose_sequence(*fragment_id, sequence))
            .collect::<Vec<_>>();
        pieces.sort_by_key(|(range, _)| *range.start());
        todo!("Handle overlapping ranges")
        
    }
}

fn decompose_sequence(
    fragment_id: u32,
    sequence: &RowIdSequence,
) -> Vec<(RangeInclusive<u64>, (U64Segment, U64Segment))> {
    let mut start_address: u64 = RowAddress::first_row(fragment_id).into();
    sequence.0
        .iter()
        .filter_map(|segment| {
            let segment_len = segment.len() as u64;
            let address_segment = U64Segment::Range(start_address..(start_address as u64 + segment_len));
            start_address += segment_len as u64;

            let coverage = segment.range()?;
            
            Some((coverage, (segment.clone(), address_segment)))
        })
        .collect()
}