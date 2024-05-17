// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

// These are all internal data structures, and are private.
mod bitmap;
mod encoded_array;
mod index;
mod segment;
mod serde;

// These are the public API.
pub use index::RowIdIndex;
pub use serde::{read_row_ids, write_row_ids};

use segment::U64Segment;

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
#[derive(Debug, Clone)]
pub struct RowIdSequence(Vec<U64Segment>);

impl std::fmt::Display for RowIdSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.iter();
        let mut first_10 = Vec::new();
        let mut last_10 = Vec::new();
        for row_id in iter.by_ref() {
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
        Self(vec![U64Segment::Range(range)])
    }
}

impl RowIdSequence {
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = u64> + '_ {
        self.0.iter().flat_map(|segment| segment.iter())
    }

    pub fn len(&self) -> u64 {
        self.iter().count() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Combines this row id sequence with another row id sequence.
    pub fn extend(&mut self, other: Self) {
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
        let (row_ids, offsets) = self.find_ids(row_ids);

        let capacity = self.0.capacity();
        let old_segments = std::mem::replace(&mut self.0, Vec::with_capacity(capacity));
        let mut remaining_segments = old_segments.as_slice();

        for (segment_idx, range) in offsets {
            let segments_handled = old_segments.len() - remaining_segments.len();
            let segments_to_add = segment_idx - segments_handled;
            self.0
                .extend_from_slice(&remaining_segments[..segments_to_add]);
            remaining_segments = &remaining_segments[segments_to_add..];

            let segment;
            (segment, remaining_segments) = remaining_segments.split_first().unwrap();

            let segment_ids = &row_ids[range];
            self.0.push(segment.delete(segment_ids));
        }

        // Add the remaining segments.
        self.0.extend_from_slice(remaining_segments);
    }

    /// Find the row ids in the sequence.
    ///
    /// Returns the row ids sorted by their appearane in the sequence.
    /// Also returns the segment index and the range where that segment's
    /// row id matches are found in the returned row id vector.
    fn find_ids(
        &self,
        row_ids: impl IntoIterator<Item = u64>,
    ) -> (Vec<u64>, Vec<(usize, Range<usize>)>) {
        // Often, the row ids will already be provided in the order they appear.
        // So the optimal way to search will be to cycle through rather than
        // restarting the search from the beginning each time.
        let mut segment_iter = self.0.iter().enumerate().cycle();

        let mut segment_matches = vec![Vec::new(); self.0.len()];

        row_ids.into_iter().for_each(|row_id| {
            let mut i = 0;
            // If we've cycled through all segments, we know the row id is not in the sequence.
            while i < self.0.len() {
                let (segment_idx, segment) = segment_iter.next().unwrap();
                if segment
                    .range()
                    .map_or(false, |range| range.contains(&row_id))
                {
                    if let Some(offset) = segment.position(row_id) {
                        segment_matches.get_mut(segment_idx).unwrap().push(offset);
                    }
                    // The row id was not found it the segment. It might be in a later segment.
                }
                i += 1;
            }
        });
        for matches in &mut segment_matches {
            matches.sort_unstable();
        }

        let mut offset = 0;
        let segment_ranges = segment_matches
            .iter()
            .enumerate()
            .filter(|(_, matches)| !matches.is_empty())
            .map(|(segment_idx, matches)| {
                let range = offset..offset + matches.len();
                offset += matches.len();
                (segment_idx, range)
            })
            .collect();
        let row_ids = segment_matches
            .into_iter()
            .enumerate()
            .flat_map(|(segment_idx, offset)| {
                offset
                    .into_iter()
                    .map(move |offset| self.0[segment_idx].get(offset).unwrap())
            })
            .collect();

        (row_ids, segment_ranges)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use pretty_assertions::assert_eq;
    use test::bitmap::Bitmap;

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
        let mut expected_bitmap = Bitmap::new_empty(9);
        for i in vec![0, 2, 4, 8] {
            expected_bitmap.set(i as usize);
        }
        assert_eq!(
            sequence.0,
            vec![U64Segment::RangeWithBitmap {
                range: 0..9,
                bitmap: expected_bitmap
            },]
        );

        let mut sequence = RowIdSequence::from(0..10);
        sequence.extend(RowIdSequence::from(12..20));
        sequence.delete(vec![0, 9, 10, 11, 12, 13]);
        assert_eq!(
            sequence.0,
            vec![U64Segment::Range(1..9), U64Segment::Range(14..20),]
        );

        let mut sequence = RowIdSequence::from(0..10);
        sequence.delete(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(sequence.0, vec![]);
    }
}
