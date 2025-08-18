// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::RangeInclusive;
use std::sync::Arc;

use super::{RowIdSequence, U64Segment};
use deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::Result;
use rangemap::RangeInclusiveMap;

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
#[derive(Debug)]
pub struct RowIdIndex(RangeInclusiveMap<u64, (U64Segment, U64Segment)>);

pub struct FragmentRowIdIndex {
    pub fragment_id: u32,
    pub row_id_sequence: Arc<RowIdSequence>,
    pub deletion_vector: Arc<DeletionVector>,
}

impl RowIdIndex {
    /// Create a new index from a list of fragment ids and their corresponding row id sequences.
    pub fn new(fragment_indices: &[FragmentRowIdIndex]) -> Result<Self> {
        let chunks = fragment_indices
            .iter()
            .flat_map(decompose_sequence)
            .collect::<Vec<_>>();

        let mut final_chunks = Vec::new();
        for processed_chunk in prep_index_chunks(chunks) {
            match processed_chunk {
                SomethingIndexChunk::NonOverlapping(chunk) => {
                    final_chunks.push(chunk);
                }
                SomethingIndexChunk::Overlapping(range, overlapping_chunks) => {
                    debug_assert_eq!(
                        range.end() - range.start() + 1,
                        overlapping_chunks
                            .iter()
                            .map(|(_, (seq, _))| seq.len() as u64)
                            .sum::<u64>(),
                        "Wrong range for {:?}, chunks: {:?}",
                        range,
                        overlapping_chunks,
                    );
                    // Merge overlapping chunks.
                    let merged_chunk = merge_overlapping_chunks(overlapping_chunks)?;
                    final_chunks.push(merged_chunk);
                }
            }
        }

        Ok(Self(RangeInclusiveMap::from_iter(final_chunks)))
    }

    /// Get the address for a given row id.
    ///
    /// Will return None if the row id does not exist in the index.
    pub fn get(&self, row_id: u64) -> Option<RowAddress> {
        let (row_id_segment, address_segment) = self.0.get(&row_id)?;
        let pos = row_id_segment.position(row_id)?;
        let address = address_segment.get(pos)?;
        Some(RowAddress::from(address))
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
    frag_index: &FragmentRowIdIndex,
) -> Vec<(RangeInclusive<u64>, (U64Segment, U64Segment))> {
    let mut start_address: u64 = RowAddress::first_row(frag_index.fragment_id).into();
    let mut current_offset = 0u32;

    frag_index
        .row_id_sequence
        .0
        .iter()
        .filter_map(|segment| {
            let segment_len = segment.len();

            let active_pairs: Vec<(u64, u64)> = segment
                .iter()
                .enumerate()
                .filter_map(|(i, row_id)| {
                    let row_offset = current_offset + i as u32;
                    if !frag_index.deletion_vector.contains(row_offset) {
                        let address = start_address + i as u64;
                        Some((row_id, address))
                    } else {
                        None
                    }
                })
                .collect();

            current_offset += segment_len as u32;
            start_address += segment_len as u64;

            if active_pairs.is_empty() {
                return None;
            }

            let row_ids: Vec<u64> = active_pairs.iter().map(|(rid, _)| *rid).collect();
            let addresses: Vec<u64> = active_pairs.iter().map(|(_, addr)| *addr).collect();

            let row_id_segment = U64Segment::from_iter(row_ids.iter().copied());
            let address_segment = U64Segment::from_iter(addresses.iter().copied());

            let coverage = row_id_segment.range()?;

            Some((coverage, (row_id_segment, address_segment)))
        })
        .collect()
}

type IndexChunk = (RangeInclusive<u64>, (U64Segment, U64Segment));

#[derive(Debug)]
enum SomethingIndexChunk {
    NonOverlapping(IndexChunk),
    Overlapping(RangeInclusive<u64>, Vec<IndexChunk>),
}

impl SomethingIndexChunk {
    fn range_end(&self) -> u64 {
        match self {
            Self::NonOverlapping((range, _)) => *range.end(),
            Self::Overlapping(range, _) => *range.end(),
        }
    }
}

/// Given a vector of index chunks, sort them and return an iterator of index chunks.
///
/// The iterator will yield chunks that are non-overlapping or a set of chunks
/// that are overlapping.
fn prep_index_chunks(mut chunks: Vec<IndexChunk>) -> impl Iterator<Item = SomethingIndexChunk> {
    chunks.sort_by_key(|(range, _)| u64::MAX - *range.start());

    let mut output = Vec::new();

    // Start assuming non-overlapping in first chunk.
    if let Some(first_chunk) = chunks.pop() {
        output.push(SomethingIndexChunk::NonOverlapping(first_chunk));
    } else {
        // Early return for empty.
        return output.into_iter();
    }

    let mut current_range = 0..=0;
    let mut current_overlap = Vec::new();
    while let Some(chunk) = chunks.pop() {
        debug_assert_eq!(
            current_overlap
                .iter()
                .map(|(range, _): &IndexChunk| *range.start())
                .min()
                .unwrap_or_default(),
            *current_range.start(),
        );
        debug_assert_eq!(
            current_overlap
                .iter()
                .map(|(range, _): &IndexChunk| *range.end())
                .max()
                .unwrap_or_default(),
            *current_range.end(),
        );

        if current_overlap.is_empty() {
            // We haven't found overlap yet.
            let last_chunk_end = output.last().unwrap().range_end();
            if *chunk.0.start() <= last_chunk_end {
                // We have found overlap.
                match output.pop().unwrap() {
                    SomethingIndexChunk::NonOverlapping(chunk) => {
                        current_overlap.push(chunk);
                    }
                    _ => unreachable!(),
                }
                current_overlap.push(chunk);

                let range_start = *current_overlap.first().unwrap().0.start();
                let range_end = *current_overlap
                    .last()
                    .unwrap()
                    .0
                    .end()
                    .max(current_overlap.first().unwrap().0.end());
                current_range = range_start..=range_end;
            } else {
                // We are still in non-overlapping space.
                output.push(SomethingIndexChunk::NonOverlapping(chunk));
            }
        } else {
            // We are making an overlap chunk
            if chunk.0.start() <= current_range.end() {
                // We are still in overlap.
                let range_end = *chunk.0.end().max(current_range.end());
                current_range = *current_range.start()..=range_end;

                current_overlap.push(chunk);
            } else {
                // We have exited overlap.
                output.push(SomethingIndexChunk::Overlapping(
                    std::mem::replace(&mut current_range, 0..=0),
                    std::mem::take(&mut current_overlap),
                ));
                output.push(SomethingIndexChunk::NonOverlapping(chunk));
            }
        }
    }
    debug_assert_eq!(
        current_overlap
            .iter()
            .map(|(range, _): &IndexChunk| *range.start())
            .min()
            .unwrap_or_default(),
        *current_range.start(),
    );
    debug_assert_eq!(
        current_overlap
            .iter()
            .map(|(range, _): &IndexChunk| *range.end())
            .max()
            .unwrap_or_default(),
        *current_range.end(),
    );

    if !current_overlap.is_empty() {
        output.push(SomethingIndexChunk::Overlapping(
            current_range.clone(),
            current_overlap,
        ));
    }

    output.into_iter()
}

fn merge_overlapping_chunks(overlapping_chunks: Vec<IndexChunk>) -> Result<IndexChunk> {
    let total_capacity = overlapping_chunks
        .iter()
        .map(|(_, (row_ids, _))| row_ids.len())
        .sum();
    let mut values = Vec::with_capacity(total_capacity);
    for (_, (row_ids, row_addrs)) in overlapping_chunks.iter() {
        values.extend(row_ids.iter().zip(row_addrs.iter()));
    }
    values.sort_by_key(|(row_id, _)| *row_id);
    let row_id_segment = U64Segment::from_iter(values.iter().map(|(row_id, _)| *row_id));
    let address_segment = U64Segment::from_iter(values.iter().map(|(_, row_addr)| *row_addr));

    let range = row_id_segment.range().unwrap();

    Ok((range, (row_id_segment, address_segment)))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // TODO: what if the row ids are not internally sorted?
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
}
