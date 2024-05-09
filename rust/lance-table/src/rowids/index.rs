// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::RangeInclusive;

use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use rangemap::RangeInclusiveMap;
use snafu::{location, Location};

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
    pub fn new(fragment_indices: &[(u32, RowIdSequence)]) -> Result<Self> {
        let mut pieces = fragment_indices
            .iter()
            .flat_map(|(fragment_id, sequence)| decompose_sequence(*fragment_id, sequence))
            .collect::<Vec<_>>();
        pieces.sort_by_key(|(range, _)| *range.start());

        // Check for overlapping ranges and if found, return a NotImplementedError.
        // We don't expect this pattern yet.
        if pieces.windows(2).any(|w| w[0].0.end() >= w[1].0.start()) {
            return Err(Error::NotSupported {
                source: "Overlapping ranges are not yet supported".into(),
                location: location!(),
            });
        }

        Ok(Self(RangeInclusiveMap::from_iter(pieces)))
    }

    pub fn get(&self, row_id: u64) -> Option<RowAddress> {
        let (row_id_segment, address_segment) = self.0.get(&row_id)?;
        let pos = row_id_segment.position(row_id)?;
        let address = address_segment.get(pos)?;
        Some(RowAddress::new_from_id(address))
    }
}

fn decompose_sequence(
    fragment_id: u32,
    sequence: &RowIdSequence,
) -> Vec<(RangeInclusive<u64>, (U64Segment, U64Segment))> {
    let mut start_address: u64 = RowAddress::first_row(fragment_id).into();
    sequence
        .0
        .iter()
        .filter_map(|segment| {
            let segment_len = segment.len() as u64;
            let address_segment = U64Segment::Range(start_address..(start_address + segment_len));
            start_address += segment_len;

            let coverage = segment.range()?;

            Some((coverage, (segment.clone(), address_segment)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address_range(frag_id: u32, offset_start: u64, offset_end: u64) -> U64Segment {
        let start = u64::from(RowAddress::first_row(frag_id)) + offset_start;
        let end = start + (offset_end - offset_start) + 1;
        U64Segment::Range(start..end)
    }

    #[test]
    fn test_decompose_sequence() {
        let fragment_id = 10;
        let sequence = RowIdSequence(vec![
            U64Segment::Range(0..10),
            U64Segment::Tombstones(10),
            U64Segment::SortedArray(vec![20, 25, 30].into()),
            U64Segment::Array(vec![40, 50, 60].into()),
        ]);

        let pieces = decompose_sequence(fragment_id, &sequence);

        // Row id ranges
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].0, 0..=9);
        // Tombstones are not included in the address range
        assert_eq!(pieces[1].0, 20..=30);
        assert_eq!(pieces[2].0, 40..=60);

        // Sequences
        assert_eq!(&pieces[0].1 .0, &sequence.0[0]);
        assert_eq!(&pieces[1].1 .0, &sequence.0[2]);
        assert_eq!(&pieces[2].1 .0, &sequence.0[3]);

        // Row address ranges
        assert_eq!(pieces[0].1 .1, address_range(fragment_id, 0, 9));
        // Tombstones are not included in the index, but their addresses should
        // be skipped.
        assert_eq!(pieces[1].1 .1, address_range(fragment_id, 20, 22));
        assert_eq!(pieces[2].1 .1, address_range(fragment_id, 23, 25));
    }

    #[test]
    fn test_new_index() {
        let fragment_indices = vec![
            (
                10,
                RowIdSequence(vec![
                    U64Segment::Range(0..10),
                    U64Segment::Tombstones(10),
                    U64Segment::SortedArray(vec![20, 25, 30].into()),
                ]),
            ),
            (
                20,
                RowIdSequence(vec![U64Segment::Array(vec![40, 50, 60].into())]),
            ),
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check various queries.
        assert_eq!(index.get(0), Some(RowAddress::new_from_parts(10, 0)));
        assert_eq!(index.get(15), None);
        assert_eq!(index.get(25), Some(RowAddress::new_from_parts(10, 21)));
        assert_eq!(index.get(40), Some(RowAddress::new_from_parts(20, 0)));
        assert_eq!(index.get(60), Some(RowAddress::new_from_parts(20, 2)));
        assert_eq!(index.get(61), None);
    }
}
