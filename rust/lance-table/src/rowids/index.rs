// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::Arc;

use super::{RowIdSequence, U64Segment};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};

/// An index of row ids
///
/// This index maps row ids to their corresponding addresses. These addresses
/// correspond to physical positions in the dataset. See [RowAddress].
///
/// This structure only reports rows that physically exist and are live. It may
/// map to addresses that have been tombstoned. A separate tombstone index is
/// used to track tombstoned rows.
// (Implementation)
// One entry per fragment holds the fragment row id sequence, its deletion
// vector, and the row id range each segment of that sequence covers. A lookup
// picks the fragments whose range can hold the id, then asks the covering
// segment for the position of the id. Building the index reads segment bounds
// only, so the build cost follows the segment count rather than the row count.
#[derive(Debug)]
pub struct RowIdIndex {
    /// Fragments that hold at least one row id, sorted by their lowest row id.
    fragments: Vec<FragmentEntry>,
    /// `max_end[i]` is the highest row id that `fragments[..=i]` cover. A lookup
    /// walks back from the last fragment starting at or below the wanted id and
    /// stops once this bound falls below it.
    max_end: Vec<u64>,
}

pub struct FragmentRowIdIndex {
    pub fragment_id: u32,
    pub row_id_sequence: Arc<RowIdSequence>,
    pub deletion_vector: Arc<DeletionVector>,
}

/// One segment of a fragment row id sequence, plus the physical offset that the
/// first row of the segment sits at inside the fragment.
#[derive(Debug)]
struct SegmentEntry {
    seq_idx: usize,
    range: RangeInclusive<u64>,
    start_offset: u32,
    /// Position of each row id in an unsorted [`U64Segment::Array`]. That
    /// encoding answers `position` by walking the segment, so a lookup without
    /// this map costs the segment length. The searchable encodings leave it
    /// `None` and use `position` directly.
    positions: Option<HashMap<u64, u32>>,
}

#[derive(Debug)]
struct FragmentEntry {
    fragment_id: u32,
    sequence: Arc<RowIdSequence>,
    deletion_vector: Arc<DeletionVector>,
    segments: Vec<SegmentEntry>,
    start: u64,
    end: u64,
}

/// Row id bounds of a segment. `None` for an empty segment.
fn segment_bounds(segment: &U64Segment) -> Option<RangeInclusive<u64>> {
    segment.range()
}

/// Row id to position map for an unsorted [`U64Segment::Array`], `None` for the
/// encodings that `position` searches. The first position of a repeated id wins,
/// which is what `position` returns.
fn array_positions(segment: &U64Segment) -> Option<HashMap<u64, u32>> {
    let U64Segment::Array(_) = segment else {
        return None;
    };
    let mut positions = HashMap::with_capacity(segment.len());
    for (position, row_id) in segment.iter().enumerate() {
        positions.entry(row_id).or_insert(position as u32);
    }
    Some(positions)
}

impl FragmentEntry {
    fn new(source: &FragmentRowIdIndex) -> Option<Self> {
        let mut segments: Vec<SegmentEntry> = Vec::new();
        let mut start_offset: u32 = 0;
        for (seq_idx, segment) in source.row_id_sequence.0.iter().enumerate() {
            let len = segment.len() as u32;
            if let Some(range) = segment_bounds(segment) {
                segments.push(SegmentEntry {
                    seq_idx,
                    range,
                    start_offset,
                    positions: array_positions(segment),
                });
            }
            start_offset += len;
        }
        let start = segments.iter().map(|entry| *entry.range.start()).min()?;
        let end = segments.iter().map(|entry| *entry.range.end()).max()?;
        Some(Self {
            fragment_id: source.fragment_id,
            sequence: source.row_id_sequence.clone(),
            deletion_vector: source.deletion_vector.clone(),
            segments,
            start,
            end,
        })
    }

    /// Row ids this fragment holds inside `lo..=hi`, in physical order.
    fn ids_in_window(&self, lo: u64, hi: u64) -> impl Iterator<Item = u64> + '_ {
        self.segments
            .iter()
            .filter(move |entry| *entry.range.start() <= hi && lo <= *entry.range.end())
            .flat_map(|entry| self.sequence.0[entry.seq_idx].iter())
            .filter(move |row_id| (lo..=hi).contains(row_id))
    }

    fn rows(&self) -> usize {
        self.segments
            .iter()
            .map(|entry| self.sequence.0[entry.seq_idx].len())
            .sum()
    }

    /// Address of `row_id` inside this fragment. `None` when the fragment does
    /// not hold the id, or holds it as a deleted row. A deleted id resolves the
    /// same way an absent one does, so the caller keeps looking in the remaining
    /// candidates.
    fn resolve(&self, row_id: u64) -> Option<RowAddress> {
        for entry in &self.segments {
            if !entry.range.contains(&row_id) {
                continue;
            }
            let position = match &entry.positions {
                Some(positions) => positions.get(&row_id).map(|position| *position as usize),
                None => self.sequence.0[entry.seq_idx].position(row_id),
            };
            let Some(position) = position else {
                continue;
            };
            let row_offset = entry.start_offset + position as u32;
            if self.deletion_vector.contains(row_offset) {
                return None;
            }
            return Some(RowAddress::new_from_parts(self.fragment_id, row_offset));
        }
        None
    }
}

impl RowIdIndex {
    /// Create a new index from a list of fragment ids and their corresponding row id sequences.
    pub fn new(fragment_indices: &[FragmentRowIdIndex]) -> Result<Self> {
        let mut fragments: Vec<FragmentEntry> = fragment_indices
            .iter()
            .filter_map(FragmentEntry::new)
            .collect();
        fragments.sort_unstable_by_key(|entry| entry.start);

        let mut max_end = Vec::with_capacity(fragments.len());
        let mut running = 0_u64;
        for entry in &fragments {
            running = running.max(entry.end);
            max_end.push(running);
        }

        reject_duplicate_row_ids(&fragments)?;

        Ok(Self { fragments, max_end })
    }

    /// Get the address for a given row id.
    ///
    /// Will return None if the row id does not exist in the index.
    pub fn get(&self, row_id: u64) -> Option<RowAddress> {
        self.candidates(row_id)
            .find_map(|(_, entry)| entry.resolve(row_id))
    }

    /// Get addresses for many row ids in one pass over the index.
    ///
    /// Returns one entry per input id, in input order (`None` for missing).
    /// Resolves each id through [`Self::get`], so both APIs return the same
    /// address for the same id. Sorts a working copy of the input first, which
    /// keeps neighbouring ids on the same fragment and its segments warm.
    pub fn get_many(&self, row_ids: &[u64]) -> Vec<Option<RowAddress>> {
        let mut out = vec![None; row_ids.len()];
        if row_ids.is_empty() {
            return out;
        }

        let mut sorted: Vec<(u64, usize)> = row_ids.iter().copied().zip(0..row_ids.len()).collect();
        sorted.sort_unstable_by_key(|&(row_id, _)| row_id);

        for (row_id, orig_idx) in sorted {
            out[orig_idx] = self.get(row_id);
        }
        out
    }

    /// Fragments that can hold `row_id`, highest starting id first.
    fn candidates(&self, row_id: u64) -> impl Iterator<Item = (usize, &FragmentEntry)> {
        let upper = self
            .fragments
            .partition_point(|entry| entry.start <= row_id);
        self.fragments[..upper]
            .iter()
            .enumerate()
            .rev()
            .take_while(move |(slot, _)| self.max_end[*slot] >= row_id)
            .filter(move |(_, entry)| entry.end >= row_id)
    }
}

/// Reject a row id that is live in two fragments.
///
/// Two fragments can only claim the same id where their row id ranges intersect,
/// so this walks the intersecting pairs and only the ids inside each intersection.
/// A dataset whose fragments hold disjoint ranges pays a range comparison per
/// neighbour; one whose ranges all intersect pays a pass over the ids in them.
fn reject_duplicate_row_ids(fragments: &[FragmentEntry]) -> Result<()> {
    for (slot, left) in fragments.iter().enumerate() {
        for right in fragments[slot + 1..]
            .iter()
            .take_while(|right| right.start <= left.end)
        {
            let lo = left.start.max(right.start);
            let hi = left.end.min(right.end);
            let (scanned, probed) = if left.rows() <= right.rows() {
                (left, right)
            } else {
                (right, left)
            };
            for row_id in scanned.ids_in_window(lo, hi) {
                if scanned.resolve(row_id).is_some() && probed.resolve(row_id).is_some() {
                    return Err(Error::internal(format!(
                        "row id index corrupt: stable row id {row_id} is live in multiple fragments"
                    )));
                }
            }
        }
    }
    Ok(())
}

impl DeepSizeOf for RowIdIndex {
    /// Charges the row id sequences and deletion vectors the index keeps alive
    /// through its `Arc`s, so a cache that weighs this entry bounds every
    /// allocation the entry retains.
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        let fragment_bytes: usize = self
            .fragments
            .iter()
            .map(|entry| {
                std::mem::size_of::<FragmentEntry>()
                    + entry.sequence.deep_size_of_children(context)
                    + entry.deletion_vector.deep_size_of_children(context)
                    + entry
                        .segments
                        .iter()
                        .map(|segment| {
                            std::mem::size_of::<SegmentEntry>()
                                + segment.positions.as_ref().map_or(0, |positions| {
                                    positions.capacity() * std::mem::size_of::<(u64, u32)>()
                                })
                        })
                        .sum::<usize>()
            })
            .sum();
        fragment_bytes + self.max_end.len() * std::mem::size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{
        prelude::{Just, Strategy, any},
        prop_assert, prop_assert_eq,
    };

    #[test]
    fn test_get_and_get_many_agree_on_a_duplicated_row_id() {
        let make = |fragment_id, row_ids: &[u64]| FragmentRowIdIndex {
            fragment_id,
            row_id_sequence: Arc::new(RowIdSequence::from(row_ids)),
            deletion_vector: Arc::new(DeletionVector::default()),
        };
        assert!(RowIdIndex::new(&[make(1, &[0, 2]), make(2, &[1, 2])]).is_err());

        let index = RowIdIndex::new(&[make(1, &[0, 2]), make(2, &[1, 3])]).unwrap();
        assert_eq!(index.get(2), index.get_many(&[0, 2])[1]);
        assert_eq!(index.get(0), index.get_many(&[0, 2])[0]);
    }

    #[test]
    fn test_unsorted_array_segment_resolves_every_position() {
        let row_ids: Vec<u64> = (0..64_u64).rev().collect();
        let index = RowIdIndex::new(&[FragmentRowIdIndex {
            fragment_id: 7,
            row_id_sequence: Arc::new(RowIdSequence::from(row_ids.as_slice())),
            deletion_vector: Arc::new(DeletionVector::default()),
        }])
        .unwrap();

        for (position, row_id) in row_ids.iter().enumerate() {
            assert_eq!(
                index.get(*row_id),
                Some(RowAddress::new_from_parts(7, position as u32)),
                "row id {row_id}"
            );
        }
        assert_eq!(index.get(64), None);
    }

    #[test]
    fn test_new_index() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![
                    U64Segment::Range(0..10),
                    U64Segment::RangeWithHoles {
                        range: 10..17,
                        holes: vec![12, 15].into(),
                    },
                    U64Segment::SortedArray(vec![20, 25, 30].into()),
                ])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![
                    U64Segment::RangeWithBitmap {
                        range: 17..20,
                        bitmap: [true, false, true].as_slice().into(),
                    },
                    U64Segment::Array(vec![40, 50, 60].into()),
                ])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
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

    #[test]
    fn test_new_index_overlap() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 23,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![3, 6, 9].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 42,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![2, 5, 8].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![1, 4, 7].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check various queries.
        assert_eq!(index.get(1), Some(RowAddress::new_from_parts(10, 0)));
        assert_eq!(index.get(2), Some(RowAddress::new_from_parts(42, 0)));
        assert_eq!(index.get(3), Some(RowAddress::new_from_parts(23, 0)));
        assert_eq!(index.get(4), Some(RowAddress::new_from_parts(10, 1)));
        assert_eq!(index.get(5), Some(RowAddress::new_from_parts(42, 1)));
        assert_eq!(index.get(6), Some(RowAddress::new_from_parts(23, 1)));
        assert_eq!(index.get(7), Some(RowAddress::new_from_parts(10, 2)));
        assert_eq!(index.get(8), Some(RowAddress::new_from_parts(42, 2)));
        assert_eq!(index.get(9), Some(RowAddress::new_from_parts(23, 2)));
    }

    #[test]
    fn test_new_index_unsorted_row_ids() {
        // Test case with unsorted row ids within fragments
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Array(
                    vec![9, 3, 6].into(), // Unsorted array
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Array(
                    vec![8, 2, 5].into(), // Unsorted array
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 30,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Array(
                    vec![7, 1, 4].into(), // Unsorted array
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check that all row ids can be found regardless of their order in the segments
        assert_eq!(index.get(1), Some(RowAddress::new_from_parts(30, 1)));
        assert_eq!(index.get(2), Some(RowAddress::new_from_parts(20, 1)));
        assert_eq!(index.get(3), Some(RowAddress::new_from_parts(10, 1)));
        assert_eq!(index.get(4), Some(RowAddress::new_from_parts(30, 2)));
        assert_eq!(index.get(5), Some(RowAddress::new_from_parts(20, 2)));
        assert_eq!(index.get(6), Some(RowAddress::new_from_parts(10, 2)));
        assert_eq!(index.get(7), Some(RowAddress::new_from_parts(30, 0)));
        assert_eq!(index.get(8), Some(RowAddress::new_from_parts(20, 0)));
        assert_eq!(index.get(9), Some(RowAddress::new_from_parts(10, 0)));

        // Check that non-existent row ids return None
        assert_eq!(index.get(0), None);
        assert_eq!(index.get(10), None);
    }

    #[test]
    fn test_new_index_partial_overlap() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 0,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::RangeWithHoles {
                    range: 0..100,
                    holes: vec![50].into(),
                }])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 1,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(50..51)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Check various queries.
        assert_eq!(index.get(0), Some(RowAddress::new_from_parts(0, 0)));
        assert_eq!(index.get(49), Some(RowAddress::new_from_parts(0, 49)));
        assert_eq!(index.get(50), Some(RowAddress::new_from_parts(1, 0)));
        assert_eq!(index.get(51), Some(RowAddress::new_from_parts(0, 50)));
        assert_eq!(index.get(99), Some(RowAddress::new_from_parts(0, 98)));
    }

    #[test]
    fn test_overlapping_chunks_sparse_with_deletions() {
        // Interleaved (overlapping) id ranges plus a deletion that leaves a hole,
        // so the union doesn't tile the span. Every live id must still resolve.
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![1, 3, 5, 7, 9].into(),
                )])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::SortedArray(
                    vec![0, 2, 4, 6, 8].into(),
                )])),
                // Delete offset 2 (id 4) -> a hole in the span.
                deletion_vector: Arc::new(DeletionVector::from_iter(vec![2])),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(index.get(0), Some(RowAddress::new_from_parts(20, 0)));
        assert_eq!(index.get(1), Some(RowAddress::new_from_parts(10, 0)));
        assert_eq!(index.get(2), Some(RowAddress::new_from_parts(20, 1)));
        assert_eq!(index.get(3), Some(RowAddress::new_from_parts(10, 1)));
        assert_eq!(index.get(4), None);
        // Surviving ids keep their original offsets (the hole is not compacted).
        assert_eq!(index.get(6), Some(RowAddress::new_from_parts(20, 3)));
        assert_eq!(index.get(8), Some(RowAddress::new_from_parts(20, 4)));
        assert_eq!(index.get(9), Some(RowAddress::new_from_parts(10, 4)));
    }

    #[test]
    fn test_index_with_deletion_vector() {
        let deletion_vector = DeletionVector::from_iter(vec![2, 3]);

        let fragment_indices = vec![FragmentRowIdIndex {
            fragment_id: 10,
            row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(0..6)])),
            deletion_vector: Arc::new(deletion_vector),
        }];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(index.get(0), Some(RowAddress::new_from_parts(10, 0)));
        assert_eq!(index.get(1), Some(RowAddress::new_from_parts(10, 1)));
        assert_eq!(index.get(4), Some(RowAddress::new_from_parts(10, 4)));
        assert_eq!(index.get(5), Some(RowAddress::new_from_parts(10, 5)));

        assert_eq!(index.get(2), None);
        assert_eq!(index.get(3), None);
    }

    #[test]
    fn test_empty_fragment_sequences() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(5..8)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(index.get(5), Some(RowAddress::new_from_parts(20, 0)));
        assert_eq!(index.get(7), Some(RowAddress::new_from_parts(20, 2)));
        assert_eq!(index.get(4), None);
    }

    #[test]
    fn test_completely_empty_index() {
        let fragment_indices = vec![];
        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(index.get(0), None);
        assert_eq!(index.get(100), None);
    }

    #[test]
    fn test_non_overlapping_ranges() {
        let fragment_indices = vec![
            FragmentRowIdIndex {
                fragment_id: 10,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(0..5)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 20,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(5..10)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
            FragmentRowIdIndex {
                fragment_id: 30,
                row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(10..15)])),
                deletion_vector: Arc::new(DeletionVector::default()),
            },
        ];

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        assert_eq!(index.get(0), Some(RowAddress::new_from_parts(10, 0)));
        assert_eq!(index.get(4), Some(RowAddress::new_from_parts(10, 4)));
        assert_eq!(index.get(5), Some(RowAddress::new_from_parts(20, 0)));
        assert_eq!(index.get(9), Some(RowAddress::new_from_parts(20, 4)));
        assert_eq!(index.get(10), Some(RowAddress::new_from_parts(30, 0)));
        assert_eq!(index.get(14), Some(RowAddress::new_from_parts(30, 4)));
    }

    fn arbitrary_row_ids(
        num_fragments_range: std::ops::Range<usize>,
        frag_size_range: std::ops::Range<usize>,
    ) -> impl Strategy<Value = Vec<(u32, Arc<RowIdSequence>)>> {
        let fragment_sizes = proptest::collection::vec(frag_size_range, num_fragments_range);
        fragment_sizes.prop_flat_map(|fragment_sizes| {
            let num_rows = fragment_sizes.iter().sum::<usize>() as u64;
            let row_ids = 0..num_rows;
            let row_ids = row_ids.collect::<Vec<_>>();
            let row_ids_shuffled = proptest::strategy::Just(row_ids).prop_shuffle();
            row_ids_shuffled.prop_map(move |row_ids| {
                let mut sequences = Vec::with_capacity(fragment_sizes.len());
                let mut i = 0;
                for size in &fragment_sizes {
                    let end = i + size;
                    let sequence =
                        RowIdSequence(vec![U64Segment::from_slice(row_ids[i..end].into())]);
                    sequences.push((i as u32, Arc::new(sequence)));
                    i = end;
                }
                sequences
            })
        })
    }

    fn arbitrary_row_ids_with_deletions(
        num_fragments_range: std::ops::Range<usize>,
        frag_size_range: std::ops::Range<usize>,
    ) -> impl Strategy<Value = Vec<(u32, Arc<RowIdSequence>, Arc<DeletionVector>)>> {
        arbitrary_row_ids(num_fragments_range, frag_size_range)
            .prop_flat_map(|row_ids| {
                let num_rows = row_ids
                    .iter()
                    .map(|(_, sequence)| sequence.len() as usize)
                    .sum::<usize>();
                (
                    Just(row_ids),
                    proptest::collection::vec(any::<bool>(), num_rows),
                )
            })
            .prop_map(|(row_ids, deleted_rows)| {
                let mut deleted_rows = deleted_rows.into_iter();
                row_ids
                    .into_iter()
                    .map(|(fragment_id, sequence)| {
                        let mut deletion_bitmap = roaring::RoaringBitmap::new();
                        for offset in 0..sequence.len() as u32 {
                            if deleted_rows.next().unwrap() {
                                deletion_bitmap.insert(offset);
                            }
                        }
                        (
                            fragment_id,
                            sequence,
                            Arc::new(DeletionVector::Bitmap(deletion_bitmap)),
                        )
                    })
                    .collect()
            })
    }

    #[test]
    fn test_large_range_segments_no_deletions() {
        // Simulates a real-world scenario: many fragments with large Range segments
        // and no deletions. Before optimization, this would iterate over all rows
        // (O(total_rows)). After optimization, it's O(num_fragments).
        let rows_per_fragment = 250_000u64;
        let num_fragments = 100u32;
        let mut offset = 0u64;

        let fragment_indices: Vec<FragmentRowIdIndex> = (0..num_fragments)
            .map(|frag_id| {
                let start = offset;
                offset += rows_per_fragment;
                FragmentRowIdIndex {
                    fragment_id: frag_id,
                    row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(
                        start..start + rows_per_fragment,
                    )])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                }
            })
            .collect();

        let start = std::time::Instant::now();
        let index = RowIdIndex::new(&fragment_indices).unwrap();
        let elapsed = start.elapsed();

        // Verify correctness at boundaries
        assert_eq!(index.get(0), Some(RowAddress::new_from_parts(0, 0)));
        assert_eq!(
            index.get(rows_per_fragment - 1),
            Some(RowAddress::new_from_parts(0, rows_per_fragment as u32 - 1))
        );
        assert_eq!(
            index.get(rows_per_fragment),
            Some(RowAddress::new_from_parts(1, 0))
        );
        let last_row = num_fragments as u64 * rows_per_fragment - 1;
        assert_eq!(
            index.get(last_row),
            Some(RowAddress::new_from_parts(
                num_fragments - 1,
                rows_per_fragment as u32 - 1
            ))
        );
        assert_eq!(index.get(last_row + 1), None);

        // With the optimization, building an index for 25M rows across 100 fragments
        // should complete in well under 1 second (typically < 1ms).
        assert!(
            elapsed.as_secs() < 1,
            "Index build took {:?} for {} fragments x {} rows = {} total rows. \
             This suggests the O(rows) -> O(fragments) optimization is not working.",
            elapsed,
            num_fragments,
            rows_per_fragment,
            num_fragments as u64 * rows_per_fragment,
        );
    }

    #[test]
    fn test_large_range_segments_with_deletions() {
        let rows_per_fragment = 1_000u64;
        let num_fragments = 10u32;
        let mut offset = 0u64;

        let fragment_indices: Vec<FragmentRowIdIndex> = (0..num_fragments)
            .map(|frag_id| {
                let start = offset;
                offset += rows_per_fragment;

                // Delete every 3rd row (offsets 0, 3, 6, ...) within each fragment.
                let mut deleted = roaring::RoaringBitmap::new();
                for i in (0..rows_per_fragment as u32).step_by(3) {
                    deleted.insert(i);
                }

                FragmentRowIdIndex {
                    fragment_id: frag_id,
                    row_id_sequence: Arc::new(RowIdSequence(vec![U64Segment::Range(
                        start..start + rows_per_fragment,
                    )])),
                    deletion_vector: Arc::new(DeletionVector::Bitmap(deleted)),
                }
            })
            .collect();

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        // Deleted rows (offset 0, 3, 6, ...) should not be found.
        // Row ID 0 has offset 0 in fragment 0 -> deleted.
        assert_eq!(index.get(0), None);
        // Row ID 3 has offset 3 in fragment 0 -> deleted.
        assert_eq!(index.get(3), None);

        // Non-deleted rows should resolve correctly.
        // Row ID 1 has offset 1 in fragment 0 -> address (frag=0, row=1).
        assert_eq!(index.get(1), Some(RowAddress::new_from_parts(0, 1)));
        // Row ID 2 has offset 2 in fragment 0 -> address (frag=0, row=2).
        assert_eq!(index.get(2), Some(RowAddress::new_from_parts(0, 2)));
        // Row ID 4 has offset 4 in fragment 0 -> address (frag=0, row=4).
        assert_eq!(index.get(4), Some(RowAddress::new_from_parts(0, 4)));

        // Check second fragment: row IDs start at 1000.
        // Row ID 1000 has offset 0 in fragment 1 -> deleted.
        assert_eq!(index.get(rows_per_fragment), None);
        // Row ID 1001 has offset 1 in fragment 1 -> address (frag=1, row=1).
        assert_eq!(
            index.get(rows_per_fragment + 1),
            Some(RowAddress::new_from_parts(1, 1))
        );

        // Last fragment, last non-deleted row.
        // Row ID 9999 has offset 999 in fragment 9 -> 999 % 3 == 0 -> deleted.
        let last_row = num_fragments as u64 * rows_per_fragment - 1;
        assert_eq!(index.get(last_row), None);
        // Row ID 9998 has offset 998 -> 998 % 3 == 2 -> not deleted.
        assert_eq!(
            index.get(last_row - 1),
            Some(RowAddress::new_from_parts(num_fragments - 1, 998))
        );

        // Out of range.
        assert_eq!(index.get(last_row + 1), None);
    }

    proptest::proptest! {
        #[test]
        fn test_new_index_robustness(
            row_ids in arbitrary_row_ids_with_deletions(0..5, 0..32)
        ) {
            let fragment_indices: Vec<FragmentRowIdIndex> = row_ids
                .iter()
                .map(|(frag_id, sequence, deletion_vector)| FragmentRowIdIndex {
                    fragment_id: *frag_id,
                    row_id_sequence: sequence.clone(),
                    deletion_vector: deletion_vector.clone(),
                })
                .collect();

            let index = RowIdIndex::new(&fragment_indices).unwrap();
            for (frag_id, sequence, deletion_vector) in row_ids.iter() {
                for (local_offset, row_id) in sequence.iter().enumerate() {
                    let expected = if deletion_vector.contains(local_offset as u32) {
                        None
                    } else {
                        Some(RowAddress::new_from_parts(*frag_id, local_offset as u32))
                    };
                    prop_assert_eq!(
                        index.get(row_id),
                        expected,
                        "Row id {} in sequence {:?} not found in index {:?}",
                        row_id,
                        sequence,
                        index
                    );
                }
            }
        }

        #[test]
        fn test_new_index_moved_row_id(
            row_id in any::<u64>(),
            source_fragment in 0u32..1024,
            fragment_delta in 1u32..1024,
        ) {
            let target_fragment = source_fragment + fragment_delta;
            let fragment_indices = [
                FragmentRowIdIndex {
                    fragment_id: source_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::Bitmap(
                        roaring::RoaringBitmap::from_iter([0]),
                    )),
                },
                FragmentRowIdIndex {
                    fragment_id: target_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                },
            ];

            let index = RowIdIndex::new(&fragment_indices).unwrap();
            prop_assert_eq!(
                index.get(row_id),
                Some(RowAddress::new_from_parts(target_fragment, 0))
            );
        }

        #[test]
        fn test_new_index_rejects_duplicate_live_row_id(
            row_id in any::<u64>(),
            first_fragment in 0u32..1024,
            fragment_delta in 1u32..1024,
        ) {
            let second_fragment = first_fragment + fragment_delta;
            let fragment_indices = [
                FragmentRowIdIndex {
                    fragment_id: first_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                },
                FragmentRowIdIndex {
                    fragment_id: second_fragment,
                    row_id_sequence: Arc::new(RowIdSequence::from(&[row_id][..])),
                    deletion_vector: Arc::new(DeletionVector::default()),
                },
            ];

            let error = RowIdIndex::new(&fragment_indices).unwrap_err();
            let is_internal = matches!(&error, Error::Internal { .. });
            let expected_message =
                format!("stable row id {row_id} is live in multiple fragments");
            let error_message = error.to_string();
            prop_assert!(is_internal);
            prop_assert!(error_message.contains(&expected_message));
        }
    }
}
