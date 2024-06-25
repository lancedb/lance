// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//! Indices for mapping row ids to their corresponding addresses.
//!
//! Each fragment in a table has a [RowIdSequence] that contains the row ids
//! in the order they appear in the fragment. The [RowIdIndex] aggregates these
//! sequences and maps row ids to their corresponding addresses across the
//! whole dataset.
//!
//! [RowIdSequence]s are serialized individually and stored in the fragment
//! metadata. Use [read_row_ids] and [write_row_ids] to read and write these
//! sequences. The on-disk format is designed to align well with the in-memory
//! representation, to avoid unnecessary deserialization.
use std::ops::Range;

// TODO: replace this with Arrow BooleanBuffer.

// These are all internal data structures, and are private.
mod bitmap;
mod encoded_array;
mod index;
mod segment;
mod serde;

use deepsize::DeepSizeOf;
// These are the public API.
pub use index::RowIdIndex;
use lance_core::{utils::mask::RowIdTreeMap, Error, Result};
use lance_io::ReadBatchParams;
pub use serde::{read_row_ids, write_row_ids};

use snafu::{location, Location};

use segment::U64Segment;

use crate::utils::LanceIteratorExtension;

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
#[derive(Debug, Clone, DeepSizeOf, PartialEq, Eq)]
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
        self.0.iter().map(|segment| segment.len() as u64).sum()
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
        // TODO: add other optimizations, such as combining two RangeWithHoles.
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
    /// Returns the row ids sorted by their appearance in the sequence.
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

    pub fn slice(&self, offset: usize, len: usize) -> RowIdSeqSlice<'_> {
        if len == 0 {
            return RowIdSeqSlice {
                segments: &[],
                offset_start: 0,
                offset_last: 0,
            };
        }

        // Find the starting position
        let mut offset_start = offset;
        let mut segment_offset = 0;
        for segment in &self.0 {
            let segment_len = segment.len();
            if offset_start < segment_len {
                break;
            }
            offset_start -= segment_len;
            segment_offset += 1;
        }

        // Find the ending position
        let mut offset_last = offset_start + len;
        let mut segment_offset_last = segment_offset;
        for segment in &self.0[segment_offset..] {
            let segment_len = segment.len();
            if offset_last <= segment_len {
                break;
            }
            offset_last -= segment_len;
            segment_offset_last += 1;
        }

        RowIdSeqSlice {
            segments: &self.0[segment_offset..=segment_offset_last],
            offset_start,
            offset_last,
        }
    }

    /// Get the row id at the given index.
    ///
    /// If the index is out of bounds, this will return None.
    pub fn get(&self, index: usize) -> Option<u64> {
        let mut offset = 0;
        for segment in &self.0 {
            let segment_len = segment.len();
            if index < offset + segment_len {
                return segment.get(index - offset);
            }
            offset += segment_len;
        }
        None
    }
}

impl From<&RowIdSequence> for RowIdTreeMap {
    fn from(row_ids: &RowIdSequence) -> Self {
        let mut tree_map = Self::new();
        for segment in &row_ids.0 {
            match segment {
                U64Segment::Range(range) => {
                    tree_map.insert_range(range.clone());
                }
                U64Segment::RangeWithBitmap { range, bitmap } => {
                    tree_map.insert_range(range.clone());
                    for (i, val) in range.clone().enumerate() {
                        if !bitmap.get(i) {
                            tree_map.remove(val);
                        }
                    }
                }
                U64Segment::RangeWithHoles { range, holes } => {
                    tree_map.insert_range(range.clone());
                    for hole in holes.iter() {
                        tree_map.remove(hole);
                    }
                }
                U64Segment::SortedArray(array) | U64Segment::Array(array) => {
                    for val in array.iter() {
                        tree_map.insert(val);
                    }
                }
            }
        }
        tree_map
    }
}

#[derive(Debug)]
pub struct RowIdSeqSlice<'a> {
    /// Current slice of the segments we cover
    segments: &'a [U64Segment],
    /// Offset into the first segment to start iterating from
    offset_start: usize,
    /// Offset into the last segment to stop iterating at
    offset_last: usize,
}

impl<'a> RowIdSeqSlice<'a> {
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        let mut known_size = self.segments.iter().map(|segment| segment.len()).sum();
        known_size -= self.offset_start;
        known_size -= self.segments.last().map(|s| s.len()).unwrap_or_default() - self.offset_last;

        let end = self.segments.len().saturating_sub(1);
        self.segments
            .iter()
            .enumerate()
            .flat_map(move |(i, segment)| {
                match i {
                    0 if self.segments.len() == 1 => {
                        let len = self.offset_last - self.offset_start;
                        // TODO: Optimize this so we don't have to use skip
                        // (take is probably fine though.)
                        Box::new(segment.iter().skip(self.offset_start).take(len))
                            as Box<dyn Iterator<Item = u64>>
                    }
                    0 => Box::new(segment.iter().skip(self.offset_start)),
                    i if i == end => Box::new(segment.iter().take(self.offset_last)),
                    _ => Box::new(segment.iter()),
                }
            })
            .exact_size(known_size)
    }
}

/// Re-chunk a sequences of row ids into chunks of a given size.
///
/// # Errors
///
/// Will return an error if the sum of the chunk sizes is not equal to the total
/// number of row ids in the sequences.
pub fn rechunk_sequences(
    sequences: impl IntoIterator<Item = RowIdSequence>,
    chunk_sizes: impl IntoIterator<Item = u64>,
) -> Result<Vec<RowIdSequence>> {
    // TODO: return an iterator. (with a good size hint?)
    let chunk_size_iter = chunk_sizes.into_iter();
    let mut chunked_sequences = Vec::with_capacity(chunk_size_iter.size_hint().0);
    let mut segment_iter = sequences
        .into_iter()
        .flat_map(|sequence| sequence.0.into_iter())
        .peekable();

    let mut segment_offset = 0_u64;
    for chunk_size in chunk_size_iter {
        let mut sequence = RowIdSequence(Vec::new());
        let mut remaining = chunk_size;

        let too_many_segments_error = || {
            Error::invalid_input(
                "Got too many segments for the provided chunk lengths",
                location!(),
            )
        };

        while remaining > 0 {
            let remaining_in_segment = segment_iter
                .peek()
                .map_or(0, |segment| segment.len() as u64 - segment_offset);
            match (remaining_in_segment.cmp(&remaining), remaining_in_segment) {
                (std::cmp::Ordering::Greater, _) => {
                    // Can only push part of the segment, we are done with this chunk.
                    let segment = segment_iter
                        .peek()
                        .ok_or_else(too_many_segments_error)?
                        .slice(segment_offset as usize, remaining as usize);
                    sequence.extend(RowIdSequence(vec![segment]));
                    segment_offset += remaining;
                    remaining = 0;
                }
                (_, 0) => {
                    // Can push the entire segment.
                    let segment = segment_iter.next().ok_or_else(too_many_segments_error)?;
                    sequence.extend(RowIdSequence(vec![segment]));
                    remaining = 0;
                }
                (_, _) => {
                    // Push remaining segment
                    let segment = segment_iter
                        .next()
                        .ok_or_else(too_many_segments_error)?
                        .slice(segment_offset as usize, remaining_in_segment as usize);
                    sequence.extend(RowIdSequence(vec![segment]));
                    segment_offset = 0;
                    remaining -= remaining_in_segment;
                }
            }
        }

        chunked_sequences.push(sequence);
    }

    if segment_iter.peek().is_some() {
        return Err(Error::invalid_input(
            "Got too few segments for the provided chunk lengths",
            location!(),
        ));
    }

    Ok(chunked_sequences)
}

/// Selects the row ids from a sequence based on the provided offsets.
pub fn select_row_ids<'a>(
    sequence: &'a RowIdSequence,
    offsets: &'a ReadBatchParams,
) -> Result<Vec<u64>> {
    let out_of_bounds_err = |offset: u32| {
        Error::invalid_input(
            format!(
                "Index out of bounds: {} for sequence of length {}",
                offset,
                sequence.len()
            ),
            location!(),
        )
    };

    match offsets {
        // TODO: Optimize this if indices are sorted, which is a common case.
        ReadBatchParams::Indices(indices) => indices
            .values()
            .iter()
            .map(|index| {
                sequence
                    .get(*index as usize)
                    .ok_or_else(|| out_of_bounds_err(*index))
            })
            .collect(),
        ReadBatchParams::Range(range) => {
            if range.end > sequence.len() as usize {
                return Err(out_of_bounds_err(range.end as u32));
            }
            let sequence = sequence.slice(range.start, range.end - range.start);
            Ok(sequence.iter().collect())
        }
        ReadBatchParams::RangeFull => Ok(sequence.iter().collect()),
        ReadBatchParams::RangeTo(to) => {
            if to.end > sequence.len() as usize {
                return Err(out_of_bounds_err(to.end as u32));
            }
            let len = to.end;
            let sequence = sequence.slice(0, len);
            Ok(sequence.iter().collect())
        }
        ReadBatchParams::RangeFrom(from) => {
            let sequence = sequence.slice(from.start, sequence.len() as usize - from.start);
            Ok(sequence.iter().collect())
        }
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
        for i in [0, 2, 4, 6, 8] {
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
        assert_eq!(sequence.0, vec![U64Segment::Range(0..0)]);
    }

    #[test]
    fn test_row_id_slice() {
        // The type of sequence isn't that relevant to the implementation, so
        // we can just have a single one with all the segment types.
        let sequence = RowIdSequence(vec![
            U64Segment::Range(30..35), // 5
            U64Segment::RangeWithHoles {
                // 8
                range: 50..60,
                holes: vec![53, 54].into(),
            },
            U64Segment::SortedArray(vec![7, 9].into()), // 2
            U64Segment::RangeWithBitmap {
                range: 0..5,
                bitmap: [true, false, true, false, true].as_slice().into(),
            },
            U64Segment::Array(vec![35, 39].into()),
            U64Segment::Range(40..50),
        ]);

        // All possible offsets and lengths
        for offset in 0..sequence.len() as usize {
            for len in 0..sequence.len() as usize {
                if offset + len > sequence.len() as usize {
                    continue;
                }
                let slice = sequence.slice(offset, len);

                let actual = slice.iter().collect::<Vec<_>>();
                let expected = sequence.iter().skip(offset).take(len).collect::<Vec<_>>();
                assert_eq!(
                    actual, expected,
                    "Failed for offset {} and len {}",
                    offset, len
                );

                let (claimed_size, claimed_max) = slice.iter().size_hint();
                assert_eq!(claimed_max, Some(claimed_size)); // Exact size hint
                assert_eq!(claimed_size, actual.len()); // Correct size hint
            }
        }
    }

    #[test]
    fn test_row_id_slice_empty() {
        let sequence = RowIdSequence::from(0..10);
        let slice = sequence.slice(10, 0);
        assert_eq!(slice.iter().collect::<Vec<_>>(), Vec::<u64>::new());
    }

    #[test]
    fn test_row_id_sequence_rechunk() {
        fn assert_rechunked(
            input: Vec<RowIdSequence>,
            chunk_sizes: Vec<u64>,
            expected: Vec<RowIdSequence>,
        ) {
            let chunked = rechunk_sequences(input, chunk_sizes).unwrap();
            assert_eq!(chunked, expected);
        }

        // Small pieces to larger ones
        let many_segments = vec![
            RowIdSequence(vec![U64Segment::Range(0..5), U64Segment::Range(35..40)]),
            RowIdSequence::from(10..18),
            RowIdSequence::from(18..28),
            RowIdSequence::from(28..30),
        ];
        let fewer_segments = vec![
            RowIdSequence(vec![U64Segment::Range(0..5), U64Segment::Range(35..40)]),
            RowIdSequence::from(10..30),
        ];
        assert_rechunked(
            many_segments.clone(),
            fewer_segments.iter().map(|seq| seq.len()).collect(),
            fewer_segments.clone(),
        );

        // Large pieces to smaller ones
        assert_rechunked(
            fewer_segments.clone(),
            many_segments.iter().map(|seq| seq.len()).collect(),
            many_segments.clone(),
        );

        // Equal pieces
        assert_rechunked(
            many_segments.clone(),
            many_segments.iter().map(|seq| seq.len()).collect(),
            many_segments.clone(),
        );

        // Too few segments -> error
        let result = rechunk_sequences(many_segments.clone(), vec![100]);
        assert!(result.is_err());

        // Too many segments -> error
        let result = rechunk_sequences(many_segments.clone(), vec![5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_row_ids() {
        // All forms of offsets
        let offsets = [
            ReadBatchParams::Indices(vec![1, 3, 9, 5, 7, 6].into()),
            ReadBatchParams::Range(2..8),
            ReadBatchParams::RangeFull,
            ReadBatchParams::RangeTo(..5),
            ReadBatchParams::RangeFrom(5..),
        ];

        // Sequences with all segment types. These have at least 10 elements,
        // so they are valid for all the above offsets.
        let sequences = [
            RowIdSequence(vec![
                U64Segment::Range(0..5),
                U64Segment::RangeWithHoles {
                    range: 50..60,
                    holes: vec![53, 54].into(),
                },
                U64Segment::SortedArray(vec![7, 9].into()),
            ]),
            RowIdSequence(vec![
                U64Segment::RangeWithBitmap {
                    range: 0..5,
                    bitmap: [true, false, true, false, true].as_slice().into(),
                },
                U64Segment::Array(vec![30, 20, 10].into()),
                U64Segment::Range(40..50),
            ]),
        ];

        for params in offsets {
            for sequence in &sequences {
                let row_ids = select_row_ids(sequence, &params).unwrap();
                let flat_sequence = sequence.iter().collect::<Vec<_>>();

                // Transform params into bounded ones
                let selection: Vec<usize> = match &params {
                    ReadBatchParams::RangeFull => (0..flat_sequence.len()).collect(),
                    ReadBatchParams::RangeTo(to) => (0..to.end).collect(),
                    ReadBatchParams::RangeFrom(from) => (from.start..flat_sequence.len()).collect(),
                    ReadBatchParams::Range(range) => range.clone().collect(),
                    ReadBatchParams::Indices(indices) => {
                        indices.values().iter().map(|i| *i as usize).collect()
                    }
                };

                let expected = selection
                    .into_iter()
                    .map(|i| flat_sequence[i])
                    .collect::<Vec<_>>();
                assert_eq!(
                    row_ids, expected,
                    "Failed for params {:?} on the sequence {:?}",
                    &params, sequence
                );
            }
        }
    }

    #[test]
    fn test_select_row_ids_out_of_bounds() {
        let offsets = [
            ReadBatchParams::Indices(vec![1, 1000, 4].into()),
            ReadBatchParams::Range(2..1000),
            ReadBatchParams::RangeTo(..1000),
        ];

        let sequence = RowIdSequence::from(0..10);

        for params in offsets {
            let result = select_row_ids(&sequence, &params);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), Error::InvalidInput { .. }));
        }
    }

    #[test]
    fn test_row_id_sequence_to_treemap() {
        let sequence = RowIdSequence(vec![
            U64Segment::Range(0..5),
            U64Segment::RangeWithHoles {
                range: 50..60,
                holes: vec![53, 54].into(),
            },
            U64Segment::SortedArray(vec![7, 9].into()),
            U64Segment::RangeWithBitmap {
                range: 10..15,
                bitmap: [true, false, true, false, true].as_slice().into(),
            },
            U64Segment::Array(vec![35, 39].into()),
            U64Segment::Range(40..50),
        ]);

        let tree_map = RowIdTreeMap::from(&sequence);
        let expected = vec![
            0, 1, 2, 3, 4, 7, 9, 10, 12, 14, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            51, 52, 55, 56, 57, 58, 59,
        ]
        .into_iter()
        .collect::<RowIdTreeMap>();
        assert_eq!(tree_map, expected);
    }
}
