// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::{Range, RangeInclusive};

use super::{bitmap::Bitmap, encoded_array::EncodedU64Array};

/// Different ways to represent a sequence of distinct u64s.
///
/// This is designed to be especially efficient for sequences that are sorted,
/// but not meaningfully larger than a Vec<u64> in the worst case.
///
/// The representation is chosen based on the properties of the sequence:
///                                                           
///  Sorted?───►Yes ───►Contiguous?─► Yes─► Range            
///    │                ▼                                 
///    │                No                                
///    │                ▼                                 
///    │              Dense?─────► Yes─► RangeWithBitmap/RangeWithHoles
///    │                ▼                                 
///    │                No─────────────► SortedArray      
///    ▼                                                    
///    No──────────────────────────────► Array            
///
/// "Dense" is decided based on ____.
///
/// Size of RangeWithBitMap for N values:
///     8 bytes + 8 bytes + ceil((max - min) / 8) bytes
/// Size of SortedArray for N values (assuming u16 packed):
///     8 bytes + 8 bytes + 8 bytes + 2 bytes * N
///
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum U64Segment {
    /// A contiguous sorted range of row ids.
    ///
    /// Total size: 16 bytes
    Range(Range<u64>),
    /// A sorted range of row ids, that is mostly contiguous.
    ///
    /// Total size: 24 bytes + n_holes * 4 bytes
    /// Use when: 32 * n_holes < max - min
    RangeWithHoles {
        range: Range<u64>,
        /// Bitmap of offsets from the start of the range that are holes.
        /// This is sorted, so binary search can be used. It's typically
        /// relatively small.
        holes: EncodedU64Array,
    },
    /// A sorted range of row ids, that is mostly contiguous.
    ///
    /// Bitmap is 1 when the value is present, 0 when it's missing.
    ///
    /// Total size: 24 bytes + ceil((max - min) / 8) bytes
    /// Use when: max - min > 16 * len
    RangeWithBitmap { range: Range<u64>, bitmap: Bitmap },
    /// A sorted array of row ids, that is sparse.
    ///
    /// Total size: 24 bytes + 2 * n_values bytes
    SortedArray(EncodedU64Array),
    /// An array of row ids, that is not sorted.
    Array(EncodedU64Array),
}

/// Statistics about a segment of u64s.
#[derive(Debug)]
struct SegmentStats {
    /// Min value in the segment.
    min: u64,
    /// Max value in the segment
    max: u64,
    /// Total number of values in the segment
    count: u64,
    /// Whether the segment is sorted
    sorted: bool,
}

impl SegmentStats {
    fn n_holes(&self) -> u64 {
        debug_assert!(self.sorted);
        if self.count == 0 {
            0
        } else {
            let total_slots = self.max - self.min + 1;
            total_slots - self.count
        }
    }
}

impl U64Segment {
    /// Return the values that are missing from the slice.
    fn holes_in_slice<'a>(
        range: RangeInclusive<u64>,
        existing: impl IntoIterator<Item = u64> + 'a,
    ) -> impl Iterator<Item = u64> + 'a {
        let mut existing = existing.into_iter().peekable();
        range.clone().filter(move |val| {
            if let Some(&existing_val) = existing.peek() {
                if existing_val == *val {
                    existing.next();
                    return false;
                }
            }
            true
        })
    }

    fn compute_stats(values: impl IntoIterator<Item = u64>) -> SegmentStats {
        let mut sorted = true;
        let mut min = u64::MAX;
        let mut max = 0;
        let mut count = 0;

        for val in values {
            count += 1;
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
            if sorted && count > 1 && val < max {
                sorted = false;
            }
        }

        if count == 0 {
            min = 0;
            max = 0;
        }

        SegmentStats {
            min,
            max,
            count,
            sorted,
        }
    }

    fn sorted_sequence_sizes(stats: &SegmentStats) -> [usize; 3] {
        let n_holes = stats.n_holes();
        let total_slots = stats.max - stats.min + 1;

        let range_with_holes = 24 + 4 * n_holes as usize;
        let range_with_bitmap = 24 + (total_slots as f64 / 8.0).ceil() as usize;
        let sorted_array = 24 + 2 * stats.count as usize;

        [range_with_holes, range_with_bitmap, sorted_array]
    }

    fn from_stats_and_sequence(
        stats: SegmentStats,
        sequence: impl IntoIterator<Item = u64>,
    ) -> Self {
        if stats.sorted {
            let n_holes = stats.n_holes();
            if stats.count == 0 {
                Self::Range(0..0)
            } else if n_holes == 0 {
                Self::Range(stats.min..(stats.max + 1))
            } else {
                let sizes = Self::sorted_sequence_sizes(&stats);
                let min_size = sizes.iter().min().unwrap();
                if min_size == &sizes[0] {
                    let range = stats.min..(stats.max + 1);
                    let mut holes =
                        Self::holes_in_slice(stats.min..=stats.max, sequence).collect::<Vec<_>>();
                    holes.sort_unstable();
                    let holes = EncodedU64Array::from(holes);

                    Self::RangeWithHoles { range, holes }
                } else if min_size == &sizes[1] {
                    let range = stats.min..(stats.max + 1);
                    let mut bitmap = Bitmap::new_full((stats.max - stats.min) as usize + 1);

                    for hole in Self::holes_in_slice(stats.min..=stats.max, sequence) {
                        let offset = (hole - stats.min) as usize;
                        bitmap.clear(offset);
                    }

                    Self::RangeWithBitmap { range, bitmap }
                } else {
                    // Must use array, but at least it's sorted
                    Self::SortedArray(EncodedU64Array::from_iter(sequence))
                }
            }
        } else {
            // Must use array
            Self::Array(EncodedU64Array::from_iter(sequence))
        }
    }

    pub fn from_slice(slice: &[u64]) -> Self {
        let stats = Self::compute_stats(slice.iter().copied());
        Self::from_stats_and_sequence(stats, slice.iter().copied())
    }
}

impl U64Segment {
    pub fn iter(&self) -> Box<dyn DoubleEndedIterator<Item = u64> + '_> {
        match self {
            Self::Range(range) => Box::new(range.clone()),
            Self::RangeWithHoles { range, holes } => {
                Box::new((range.start..range.end).filter(move |&val| {
                    // TODO: we could write a more optimal version of this
                    // iterator, but would need special handling to make it
                    // double ended.
                    holes.binary_search(val).is_err()
                }))
            }
            Self::RangeWithBitmap { range, bitmap } => {
                Box::new((range.start..range.end).filter(|val| {
                    let offset = (val - range.start) as usize;
                    bitmap.get(offset)
                }))
            }
            Self::SortedArray(array) => Box::new(array.iter()),
            Self::Array(array) => Box::new(array.iter()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Range(range) => (range.end - range.start) as usize,
            Self::RangeWithHoles { range, holes } => {
                let holes = holes.iter().count();
                (range.end - range.start) as usize - holes
            }
            Self::RangeWithBitmap { range, bitmap } => {
                let holes = bitmap.count_zeros();
                (range.end - range.start) as usize - holes
            }
            Self::SortedArray(array) => array.len(),
            Self::Array(array) => array.len(),
        }
    }

    /// Get the min and max value of the segment, excluding tombstones.
    pub fn range(&self) -> Option<RangeInclusive<u64>> {
        match self {
            Self::Range(range) if range.is_empty() => None,
            Self::Range(range)
            | Self::RangeWithBitmap { range, .. }
            | Self::RangeWithHoles { range, .. } => Some(range.start..=(range.end - 1)),
            Self::SortedArray(array) => {
                // We can assume that the array is sorted.
                let min_value = array.first().unwrap();
                let max_value = array.last().unwrap();
                Some(min_value..=max_value)
            }
            Self::Array(array) => {
                let min_value = array.min().unwrap();
                let max_value = array.max().unwrap();
                Some(min_value..=max_value)
            }
        }
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        match self {
            Self::Range(range) => {
                let start = range.start + offset as u64;
                Self::Range(start..(start + len as u64))
            }
            Self::RangeWithHoles { range, holes } => {
                let start = range.start + offset as u64;
                let end = start + len as u64;

                let start = holes.binary_search(start).unwrap_or_else(|x| x) as u64;
                let end = holes.binary_search(end).unwrap_or_else(|x| x) as u64;
                let holes_len = end - start;

                if holes_len == 0 {
                    Self::Range(start..end)
                } else {
                    let holes = holes.slice(start as usize, holes_len as usize);
                    Self::RangeWithHoles {
                        range: start..end,
                        holes,
                    }
                }
            }
            Self::RangeWithBitmap { range, bitmap } => {
                let start = range.start + offset as u64;
                let end = start + len as u64;

                let bitmap = bitmap.slice(offset, len);
                if bitmap.count_ones() == len {
                    // Bitmap no longer serves a purpose
                    Self::Range(start..end)
                    // TODO: could also have a case where we switch back to RangeWithHoles
                } else {
                    Self::RangeWithBitmap {
                        range: start..end,
                        bitmap: bitmap.into(),
                    }
                }
            }
            Self::SortedArray(array) => Self::SortedArray(array.slice(offset, len)),
            Self::Array(array) => Self::Array(array.slice(offset, len)),
        }
    }

    pub fn position(&self, val: u64) -> Option<usize> {
        match self {
            Self::Range(range) => {
                if range.contains(&val) {
                    Some((val - range.start) as usize)
                } else {
                    None
                }
            }
            Self::RangeWithHoles { range, holes } => {
                if range.contains(&val) && holes.binary_search(val).is_err() {
                    let offset = (val - range.start) as usize;
                    let holes = holes.iter().take_while(|&hole| hole < val).count();
                    Some(offset - holes)
                } else {
                    None
                }
            }
            Self::RangeWithBitmap { range, bitmap } => {
                if range.contains(&val) && bitmap.get((val - range.start) as usize) {
                    let offset = (val - range.start) as usize;
                    let num_zeros = bitmap.slice(0, offset).count_zeros();
                    Some(offset - num_zeros)
                } else {
                    None
                }
            }
            Self::SortedArray(array) => array.binary_search(val).ok(),
            Self::Array(array) => array.iter().position(|v| v == val),
        }
    }

    pub fn get(&self, i: usize) -> Option<u64> {
        match self {
            Self::Range(range) => match range.start.checked_add(i as u64) {
                Some(val) if val < range.end => Some(val),
                _ => None,
            },
            Self::RangeWithHoles { range, .. } => {
                if i >= (range.end - range.start) as usize {
                    return None;
                }
                self.iter().nth(i)
            }
            Self::RangeWithBitmap { range, .. } => {
                if i >= (range.end - range.start) as usize {
                    return None;
                }
                self.iter().nth(i)
            }
            Self::SortedArray(array) => array.get(i),
            Self::Array(array) => array.get(i),
        }
    }

    /// Delete a set of row ids from the segment.
    /// The row ids are assumed to be in the segment. (within the range, not
    /// already deleted.)
    /// They are also assumed to be ordered by appearance in the segment.
    pub fn delete(&self, vals: &[u64]) -> Self {
        // TODO: can we enforce these assumptions? or make them safer?
        debug_assert!(vals.iter().all(|&val| self.range().unwrap().contains(&val)));

        let make_new_iter = || {
            let mut vals_iter = vals.iter().copied().peekable();
            self.iter().filter(move |val| {
                if let Some(&next_val) = vals_iter.peek() {
                    if next_val == *val {
                        vals_iter.next();
                        return false;
                    }
                }
                true
            })
        };
        let stats = Self::compute_stats(make_new_iter());

        // Then just use Self::From_stats_and_sequence
        Self::from_stats_and_sequence(stats, make_new_iter())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_segments() {
        fn check_segment(values: &[u64], expected: &U64Segment) {
            let segment = U64Segment::from_slice(values);
            assert_eq!(segment, *expected);
            assert_eq!(values.len(), segment.len());

            let roundtripped = segment.iter().collect::<Vec<_>>();
            assert_eq!(roundtripped, values);

            let expected_min = values.iter().copied().min();
            let expected_max = values.iter().copied().max();
            match segment.range() {
                Some(range) => {
                    assert_eq!(range.start(), &expected_min.unwrap());
                    assert_eq!(range.end(), &expected_max.unwrap());
                }
                None => {
                    assert_eq!(expected_min, None);
                    assert_eq!(expected_max, None);
                }
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(segment.get(i), Some(*value), "i = {}", i);
                assert_eq!(segment.position(*value), Some(i), "i = {}", i);
            }

            check_segment_iter(&segment);
        }

        fn check_segment_iter(segment: &U64Segment) {
            // Should be able to iterate forwards and backwards, and get the same thing.
            let forwards = segment.iter().collect::<Vec<_>>();
            let mut backwards = segment.iter().rev().collect::<Vec<_>>();
            backwards.reverse();
            assert_eq!(forwards, backwards);

            // Should be able to pull from both sides in lockstep.
            let mut expected = Vec::with_capacity(segment.len());
            let mut actual = Vec::with_capacity(segment.len());
            let mut iter = segment.iter();
            // Alternating forwards and backwards
            for i in 0..segment.len() {
                if i % 2 == 0 {
                    actual.push(iter.next().unwrap());
                    expected.push(segment.get(i / 2).unwrap());
                } else {
                    let i = segment.len() - 1 - i / 2;
                    actual.push(iter.next_back().unwrap());
                    expected.push(segment.get(i).unwrap());
                };
            }
            assert_eq!(expected, actual);
        }

        // Empty
        check_segment(&[], &U64Segment::Range(0..0));

        // Single value
        check_segment(&[42], &U64Segment::Range(42..43));

        // Contiguous range
        check_segment(
            &(100..200).collect::<Vec<_>>(),
            &U64Segment::Range(100..200),
        );

        // Range with a hole
        let values = (0..1000).filter(|&x| x != 100).collect::<Vec<_>>();
        check_segment(
            &values,
            &U64Segment::RangeWithHoles {
                range: 0..1000,
                holes: vec![100].into(),
            },
        );

        // Range with every other value missing
        let values = (0..1000).filter(|&x| x % 2 == 0).collect::<Vec<_>>();
        check_segment(
            &values,
            &U64Segment::RangeWithBitmap {
                range: 0..999,
                bitmap: Bitmap::from((0..999).map(|x| x % 2 == 0).collect::<Vec<_>>().as_slice()),
            },
        );

        // Sparse but sorted sequence
        check_segment(
            &[1, 7000, 24000],
            &U64Segment::SortedArray(vec![1, 7000, 24000].into()),
        );

        // Sparse unsorted sequence
        check_segment(
            &[7000, 1, 24000],
            &U64Segment::Array(vec![7000, 1, 24000].into()),
        );
    }
}
