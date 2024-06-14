// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::RangeInclusive;
use std::sync::Arc;

use deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use rangemap::RangeInclusiveMap;
use snafu::{location, Location};

use super::{RowIdSequence, U64Segment};

/// An index of row ids
///
/// This index is used to map row ids to their corresponding addresses. These
/// addresses correspond to physical positions in the dataset. See [RowAddress].
///
/// This structure only contains rows that physically exist. However, it may
/// map to addresses that have been tombstoned. A separate tombstone index is
/// used to track tombstoned rows.
// (Implementation)
// Disjoint ranges of row ids are stored as the keys of the map. The values are
// a pair of segments. The first segment is the row ids, and the second segment
// is the addresses.
pub struct RowIdIndex(RangeInclusiveMap<u64, (U64Segment, U64Segment)>);

impl RowIdIndex {
    /// Create a new index from a list of fragment ids and their corresponding row id sequences.
    pub fn new(fragment_indices: &[(u32, Arc<RowIdSequence>)]) -> Result<Self> {
        let mut pieces = fragment_indices
            .iter()
            .flat_map(|(fragment_id, sequence)| decompose_sequence(*fragment_id, sequence))
            .collect::<Vec<_>>();
        pieces.sort_by_key(|(range, _)| *range.start());

        // Check for overlapping ranges and if found, return a NotImplementedError.
        // We don't expect this pattern yet until we make row ids stable after
        // updates.
        if pieces.windows(2).any(|w| w[0].0.end() >= w[1].0.start()) {
            return Err(Error::NotSupported {
                source: "Overlapping ranges are not yet supported".into(),
                location: location!(),
            });
        }

        Ok(Self(RangeInclusiveMap::from_iter(pieces)))
    }

    /// Get the address for a given row id.
    ///
    /// Will return None if the row id does not exist in the index.
    pub fn get(&self, row_id: u64) -> Option<RowAddress> {
        let (row_id_segment, address_segment) = self.0.get(&row_id)?;
        let pos = row_id_segment.position(row_id)?;
        let address = address_segment.get(pos)?;
        Some(RowAddress::new_from_id(address))
    }
}

impl DeepSizeOf for RowIdIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.0
            .iter()
            .map(|(_, (row_id_segment, address_segment))| {
                (2 * std::mem::size_of::<u64>())
                    + std::mem::size_of::<(U64Segment, U64Segment)>()
                    + row_id_segment.deep_size_of_children(context)
                    + address_segment.deep_size_of_children(context)
            })
            .sum()
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

    #[test]
    fn test_new_index() {
        let fragment_indices = vec![
            (
                10,
                Arc::new(RowIdSequence(vec![
                    U64Segment::Range(0..10),
                    U64Segment::RangeWithHoles {
                        range: 10..17,
                        holes: vec![12, 15].into(),
                    },
                    U64Segment::SortedArray(vec![20, 25, 30].into()),
                ])),
            ),
            (
                20,
                Arc::new(RowIdSequence(vec![
                    U64Segment::RangeWithBitmap {
                        range: 17..20,
                        bitmap: [true, false, true].as_slice().into(),
                    },
                    U64Segment::Array(vec![40, 50, 60].into()),
                ])),
            ),
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check various queries.
        assert_eq!(index.get(0), Some(RowAddress::new_from_parts(10, 0)));
        assert_eq!(index.get(15), None);
        assert_eq!(index.get(16), Some(RowAddress::new_from_parts(10, 14)));
        assert_eq!(index.get(17), Some(RowAddress::new_from_parts(20, 0)));
        assert_eq!(index.get(25), Some(RowAddress::new_from_parts(10, 16)));
        assert_eq!(index.get(40), Some(RowAddress::new_from_parts(20, 2)));
        assert_eq!(index.get(60), Some(RowAddress::new_from_parts(20, 4)));
        assert_eq!(index.get(61), None);
    }
}
