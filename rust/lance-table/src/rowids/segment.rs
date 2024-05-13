// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::{Range, RangeInclusive};

use super::{bitmap::Bitmap, encoded_array::EncodedU64Array};

/// Different ways to represent a sequence of u64s.
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
        self.max - self.min - self.count + 1
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

        SegmentStats {
            min,
            max,
            count,
            sorted,
        }
    }

    fn from_stats_and_sequence(
        stats: SegmentStats,
        sequence: impl IntoIterator<Item = u64>,
    ) -> Self {
        let n_holes = stats.n_holes();

        if stats.sorted {
            if n_holes == 0 {
                Self::Range(stats.min..stats.max)
            } else if 32 * n_holes < stats.max - stats.min {
                let range = stats.min..(stats.max + 1);

                let mut holes =
                    Self::holes_in_slice(stats.min..=stats.max, sequence).collect::<Vec<_>>();
                holes.sort_unstable();
                let holes = EncodedU64Array::from(holes);

                Self::RangeWithHoles { range, holes }
            } else if stats.max - stats.min > 16 * stats.count {
                let range = stats.min..(stats.max + 1);
                let mut bitmap = Bitmap::new((stats.max - stats.min) as usize);

                for hole in Self::holes_in_slice(stats.min..=stats.max, sequence.into_iter()) {
                    let offset = (hole - stats.min) as usize;
                    bitmap.set(offset);
                }

                Self::RangeWithBitmap { range, bitmap }
            } else {
                // Must use array, but at least it's sorted
                Self::SortedArray(EncodedU64Array::from_iter(sequence))
            }
        } else {
            // Must use array
            Self::Array(EncodedU64Array::from_iter(sequence))
        }
    }

    pub fn from_slice(slice: &[u64]) -> Self {
        let stats = Self::compute_stats(slice.iter().copied());
        let n_holes = stats.n_holes();

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
                Box::new((range.start..range.end).filter_map(|val| {
                    let offset = (val - range.start) as usize;
                    if bitmap.get(offset - 1) {
                        Some(val)
                    } else {
                        None
                    }
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
                let holes = bitmap.count_ones();
                (range.end - range.start) as usize - holes as usize
            }
            Self::SortedArray(array) => array.len(),
            Self::Array(array) => array.len(),
        }
    }

    /// Get the min and max value of the segment, excluding tombstones.
    pub fn range(&self) -> Option<RangeInclusive<u64>> {
        match self {
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
                        bitmap,
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
                if range.contains(&val) && !bitmap.get((val - range.start) as usize) {
                    let offset = (val - range.start) as usize;
                    let num_zeros = bitmap.len() - bitmap.slice(0, offset).count_ones();
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
            Self::RangeWithHoles { range, holes } => {
                let val = range.start + i as u64;
                if holes.binary_search(val).is_err() {
                    Some(val)
                } else {
                    None
                }
            }
            Self::RangeWithBitmap { range, bitmap } => {
                let val = range.start + i as u64;
                if !bitmap.get(i) {
                    Some(val)
                } else {
                    None
                }
            }
            Self::SortedArray(array) => array.get(i),
            Self::Array(array) => array.get(i),
        }
    }

    /// Delete a set of row ids from the segment.
    /// The row ids are assumed to be in the segment. (within the range, not
    /// already deleted.)
    /// They are also assumed to be sorted.
    pub fn delete(&self, vals: &[u64]) -> Self {
        // Collect some stats

        // Then just use Self::From_stats_and_sequence
        todo!()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_segment() {
        todo!()
    }

    #[test]
    fn test_segment_iterator() {
        todo!("validate double ended iterator");
    }
}
