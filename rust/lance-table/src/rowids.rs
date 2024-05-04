// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::{Range, RangeInclusive};

use snafu::{location, Location};

use lance_core::{Error, Result};

use self::encoded_array::EncodedU64Array;

mod encoded_array;
pub mod serde;
mod index;

pub use index::RowIdIndex;

/// A row ID is a unique identifier for a row in a table.
///
/// The will be initially assigned as an incrementing id.
///
/// When a row is deleted, the row id will be marked as a tombstone.
pub struct RowId;

impl RowId {
    pub fn new_range(max_row_id: Option<u64>, nrows: u64) -> Result<Range<u64>> {
        if let Some(max_row_id) = max_row_id {
            let start = max_row_id + 1;
            let end = start
                .checked_add(nrows)
                .ok_or_else(|| Error::invalid_input("Ran out of row IDs", location!()))?;
            Ok(start..end)
        } else {
            Ok(0..nrows)
        }
    }

    /// Returns the tombstone row id. This is u64::MAX.
    pub fn tombstone() -> u64 {
        u64::MAX
    }
}

/// A sequence of row ids.
///
/// Row ids are u64s that:
///
/// 1. Are **unique** within a table (except for tombstones)
/// 2. Are *often* but not always sorted and/or contiguous.
///
/// This sequence of row ids is optimized to be compact when the row ids are
/// contiguous and sorted. However, it does not require that the row ids are
/// contiguous or sorted.
///
/// We can make optimizations that assume uniqueness.
#[derive(Debug)]
pub struct RowIdSequence(Vec<U64Segment>);

impl std::fmt::Display for RowIdSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.iter();
        let mut first_10 = Vec::new();
        let mut last_10 = Vec::new();
        while let Some(row_id) = iter.next() {
            first_10.push(row_id);
            if first_10.len() > 10 {
                break;
            }
        }

        while let Some(row_id) = iter.next_back() {
            last_10.push(row_id);
            if last_10.len() > 10 {
                break;
            }
        }
        last_10.reverse();

        let theres_more = iter.next().is_some();

        write!(f, "[")?;
        for row_id in first_10 {
            write!(f, "{}", row_id)?;
        }
        if theres_more {
            write!(f, ", ...")?;
        }
        for row_id in last_10 {
            write!(f, ", {}", row_id)?;
        }
        write!(f, "]")
    }
}

impl From<Range<u64>> for RowIdSequence {
    fn from(range: Range<u64>) -> Self {
        RowIdSequence(vec![U64Segment::Range(range)])
    }
}

impl RowIdSequence {
    fn iter(&self) -> impl DoubleEndedIterator<Item = u64> + '_ {
        self.0.iter().flat_map(|segment| segment.iter())
    }

    pub fn len(&self) -> u64 {
        self.iter().count() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Combines this row id sequence with another row id sequence.
    pub fn extend(&mut self, other: RowIdSequence) {
        // If the last element of this sequence and the first element of next
        // sequence are ranges, we might be able to combine them into a single
        // range.
        if let (Some(U64Segment::Range(range1)), Some(U64Segment::Range(range2))) =
            (self.0.last(), other.0.first())
        {
            if range1.end == range2.start {
                let new_range = U64Segment::Range(range1.start..range2.end);
                self.0.pop();
                self.0.push(new_range);
                self.0.extend(other.0.into_iter().skip(1));
                return;
            }
        }
        self.0.extend(other.0);
    }

    /// Mark a set of row ids as deleted. Their value will be replaced with tombstones.
    pub fn delete(&mut self, row_ids: impl IntoIterator<Item = u64>) {
        // Order the row ids by position in which they appear in the sequence.
        let mut positions = self.find_ids(row_ids);
        positions.sort_by_key(|(_, pos)| pos.clone());

        let capacity = self.0.capacity();
        let old_segments = std::mem::replace(&mut self.0, Vec::with_capacity(capacity));
        let mut remaining_segments = old_segments.as_slice();

        let mut positions_iter = positions.into_iter().peekable();
        let mut position = if let Some(pos) = positions_iter.next() {
            pos
        } else {
            // No row ids to delete.
            self.0 = old_segments;
            return;
        };
        let mut position_batch = Vec::new();
        loop {
            // Add all segments up to the segment containing the row id.
            let segments_handled = old_segments.len() - remaining_segments.len();
            let segments_to_add = position.1 .0 - segments_handled;
            self.0
                .extend_from_slice(&remaining_segments[..segments_to_add]);
            remaining_segments = &remaining_segments[segments_to_add..];

            let segment;
            (segment, remaining_segments) = remaining_segments.split_first().unwrap();

            // Handle all positions for this segment now.
            position_batch.push(position.1 .1);
            while let Some(next_position) = positions_iter.peek() {
                if next_position.1 .0 != position.1 .0 {
                    position = positions_iter.next().unwrap();
                    break;
                }
                position_batch.push(next_position.1 .1);
                position = positions_iter.next().unwrap();
            }

            Self::delete_from_segment(&mut self.0, segment, &position_batch);
            position_batch.clear();

            if positions_iter.peek().is_none() {
                break;
            }
        }

        // Add the remaining segments.
        self.0.extend_from_slice(&remaining_segments);
    }

    /// Find the row ids in the sequence.
    ///
    /// Returns the row ids and, if found, the position of the segment containing
    /// it and the offset within the segment.
    fn find_ids(&self, row_ids: impl IntoIterator<Item = u64>) -> Vec<(u64, (usize, usize))> {
        // Often, the row ids will already be provided in the order they appear.
        // So the optimal way to search will be to cycle through rather than
        // restarting the search from the beginning each time.
        let mut segment_iter = self.0.iter().enumerate().cycle();

        row_ids
            .into_iter()
            .filter_map(|row_id| {
                let mut i = 0;
                // If we've cycled through all segments, we know the row id is not in the sequence.
                while i < self.0.len() {
                    let (segment_idx, segment) = segment_iter.next().unwrap();
                    if segment
                        .range()
                        .map_or(false, |range| range.contains(&row_id))
                    {
                        let offset = match segment {
                            U64Segment::Tombstones(_) => {
                                unreachable!("Tombstones should not be in the sequence")
                            }
                            U64Segment::Range(range) => Some((row_id - range.start) as usize),
                            U64Segment::SortedArray(array) => array.binary_search(row_id).ok(),
                            U64Segment::Array(array) => array.iter().position(|v| v == row_id),
                        };
                        if let Some(offset) = offset {
                            return Some((row_id, (segment_idx, offset)));
                        }
                        // The row id was not found it the segment. It might be in a later segment.
                    }
                    i += 1;
                }
                None
            })
            .collect()
    }

    /// Replace the positions in the segment with tombstones, pushing the new
    /// segments onto the destination vector.
    ///
    /// This might involve splitting the segment into multiple segments.
    /// It also might increment the tombstone count of the previous segment.
    fn delete_from_segment(dest: &mut Vec<U64Segment>, segment: &U64Segment, positions: &[usize]) {
        // Offset to the first position in segment that we haven't added to dest.
        let mut offset = 0;
        for &position in positions {
            // Add portio of segment up to the position.
            if position > offset {
                dest.push(segment.slice(offset, position - offset));
            }

            // Add the tombstone. If the last segment is a tombstone, increment the count
            // instead of appending a new tombstone segment.
            match dest.last_mut() {
                Some(U64Segment::Tombstones(count)) => *count += 1,
                _ => dest.push(U64Segment::Tombstones(1)),
            }

            offset = position + 1;
        }

        // Add the remaining slice of the segment.
        if offset < segment.len() {
            dest.push(segment.slice(offset, segment.len() - offset));
        }
    }
}

/// Different ways to represent a sequence of u64s.
///
/// This is designed to be especially efficient for sequences that are sorted,
/// but not meaningfully larger than a Vec<u64> in the worst case.
#[derive(Debug, PartialEq, Eq, Clone)]
enum U64Segment {
    /// A contiguous sequence of tombstones. This is the only way to represent
    /// tombstones. All other segments are assumed to not contain tombstones.
    Tombstones(u64),
    Range(Range<u64>),
    SortedArray(EncodedU64Array),
    Array(EncodedU64Array),
}

impl U64Segment {
    fn iter(&self) -> Box<dyn DoubleEndedIterator<Item = u64> + '_> {
        match self {
            U64Segment::Tombstones(count) => Box::new((0..*count).map(|_| RowId::tombstone())),
            U64Segment::Range(range) => Box::new(range.clone()),
            U64Segment::SortedArray(array) => Box::new(array.iter()),
            U64Segment::Array(array) => Box::new(array.iter()),
        }
    }

    fn len(&self) -> usize {
        match self {
            U64Segment::Tombstones(count) => *count as usize,
            U64Segment::Range(range) => (range.end - range.start) as usize,
            U64Segment::SortedArray(array) => array.len(),
            U64Segment::Array(array) => array.len(),
        }
    }

    /// Get the min and max value of the segment, excluding tombstones.
    fn range(&self) -> Option<RangeInclusive<u64>> {
        match self {
            U64Segment::Tombstones(_) => None,
            U64Segment::Range(range) => Some(range.start..=(range.end - 1)),
            U64Segment::SortedArray(array) => {
                // We can assume that the array is sorted.
                let min_value = array.first().unwrap();
                let max_value = array.last().unwrap();
                Some(min_value..=max_value)
            }
            U64Segment::Array(array) => {
                let min_value = array.min().unwrap();
                let max_value = array.max().unwrap();
                Some(min_value..=max_value)
            }
        }
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        match self {
            U64Segment::Tombstones(_) => U64Segment::Tombstones(len as u64),
            U64Segment::Range(range) => {
                let start = range.start + offset as u64;
                U64Segment::Range(start..(start + len as u64))
            }
            U64Segment::SortedArray(array) => {
                // TODO: this could be optimized.
                U64Segment::SortedArray(array.slice(offset, len))
            }
            U64Segment::Array(array) => U64Segment::Array(array.slice(offset, len)),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use pretty_assertions::assert_eq;

    #[test]
    fn test_row_id_new_range() {
        let range = RowId::new_range(None, 10).unwrap();
        assert_eq!(range, 0..10);

        let range = RowId::new_range(Some(10), 10).unwrap();
        assert_eq!(range, 11..21);

        let range = RowId::new_range(Some(u64::MAX - 10), 8).unwrap();
        assert_eq!(range, (u64::MAX - 9)..(u64::MAX - 1));

        let range = RowId::new_range(Some(u64::MAX - 10), 11);
        assert!(range.is_err());
        assert!(
            matches!(range.unwrap_err(), Error::InvalidInput { source, .. } if source.to_string().contains("Ran out of row IDs"))
        );
    }

    #[test]
    fn test_row_id_sequence_from_range() {
        let sequence = RowIdSequence::from(0..10);
        assert_eq!(sequence.len(), 10);
        assert_eq!(sequence.is_empty(), false);

        let iter = sequence.iter();
        assert_eq!(iter.collect::<Vec<_>>(), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_row_id_sequence_extend() {
        let mut sequence = RowIdSequence::from(0..10);
        sequence.extend(RowIdSequence::from(10..20));
        assert_eq!(sequence.0, vec![U64Segment::Range(0..20)]);

        let mut sequence = RowIdSequence::from(0..10);
        sequence.extend(RowIdSequence::from(20..30));
        assert_eq!(
            sequence.0,
            vec![U64Segment::Range(0..10), U64Segment::Range(20..30)]
        );
    }

    #[test]
    fn test_row_id_sequence_delete() {
        let mut sequence = RowIdSequence::from(0..10);
        sequence.delete(vec![1, 3, 5, 7, 9]);
        assert_eq!(
            sequence.0,
            vec![
                U64Segment::Range(0..1),
                U64Segment::Tombstones(1),
                U64Segment::Range(2..3),
                U64Segment::Tombstones(1),
                U64Segment::Range(4..5),
                U64Segment::Tombstones(1),
                U64Segment::Range(6..7),
                U64Segment::Tombstones(1),
                U64Segment::Range(8..9),
                U64Segment::Tombstones(1),
            ]
        );

        let mut sequence = RowIdSequence::from(0..10);
        sequence.extend(RowIdSequence::from(12..20));
        sequence.delete(vec![0, 9, 10, 11, 12, 13]);
        assert_eq!(
            sequence.0,
            vec![
                U64Segment::Tombstones(1),
                U64Segment::Range(1..9),
                U64Segment::Tombstones(3),
                U64Segment::Range(14..20),
            ]
        );

        let mut sequence = RowIdSequence::from(0..10);
        sequence.delete(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(sequence.0, vec![U64Segment::Tombstones(10)]);

        let mut sequence = RowIdSequence::from(0..10);
        sequence.delete(vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(sequence.0, vec![U64Segment::Tombstones(10)]);
    }
}
