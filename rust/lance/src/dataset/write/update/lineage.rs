// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::rowids::RowLineageSpill;
use lance_table::format::RowDatasetVersionSequence;
use lance_table::rowids::segment::U64Segment;
use lance_table::rowids::{RowIdSequence, write_row_ids};

/// Bound every contiguous output slice, including a byte-driven file split.
/// The two scans capture ids in fixed-size metadata chunks, so their segment
/// boundaries do not depend on the projected user columns or scan batches.
/// Only the final output's exact encoding decides whether its metadata spills.
pub(super) fn plan_carried_lineage_spill(
    limit: usize,
    row_ids: &RowIdSequence,
    created_at: &RowDatasetVersionSequence,
) -> RowLineageSpill {
    let row_id_bytes = row_ids
        .segments()
        .iter()
        .map(segment_slice_bound)
        .sum::<u128>();
    let version_bytes = created_at
        .runs
        .iter()
        .map(|run| {
            // The span uses the same tag/length wrapper as a row-id segment. Add
            // one version varint and the enclosing run's tag/length wrapper.
            segment_slice_bound(&run.span) + 2 * (1 + 10)
        })
        .sum::<u128>();
    RowLineageSpill {
        row_ids: row_id_bytes > limit as u128,
        created_at: version_bytes > limit as u128,
        last_updated_at: false,
    }
}

fn segment_slice_bound(segment: &U64Segment) -> u128 {
    if segment.is_empty() {
        return 0;
    }
    if let U64Segment::Range(range) = segment {
        // A nonzero slice start can need more protobuf bytes than the whole
        // range's zero start. The final one-row slice maximizes both varints.
        return write_row_ids(&RowIdSequence::from(range.end - 1..range.end)).len() as u128;
    }
    let Some(range) = segment.range() else {
        return 0;
    };
    let width = if range.end() - range.start() <= u16::MAX as u64 {
        2
    } else if range.end() - range.start() <= u32::MAX as u64 {
        4
    } else {
        8
    };
    let mut payload = width * segment.len() as u128;
    if !matches!(segment, U64Segment::Array(_)) && *range.end() != u64::MAX {
        // For a sorted slice, from_slice chooses the minimum of 2*rows,
        // 4*holes and ceil(span/8), plus their common 24-byte estimate.
        // Its wire payload is at most width/2 times that estimate: arrays
        // use at most `width` bytes/value, holes at most `width` bytes/hole,
        // and a bitmap one byte/eight slots. Every slice's span only shrinks.
        let span = *range.end() as u128 - *range.start() as u128 + 1;
        payload = payload.min((width / 2) * span.div_ceil(8));
    }
    // The largest envelope is RangeWithHoles: sequence segment, segment
    // oneof, start, end, holes message, array oneof, base, and byte payload.
    // Each has a one-byte tag and at most a ten-byte varint/length prefix.
    payload + 8 * (1 + 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_table::format::RowDatasetVersionRun;
    use lance_table::rowids::version::{rechunk_version_sequences, write_dataset_versions};
    use lance_table::rowids::{read_row_ids, rechunk_sequences};
    use prost::Message;
    use rstest::rstest;

    fn sequence_from_segments(segments: Vec<U64Segment>) -> RowIdSequence {
        let proto = lance_table::format::pb::RowIdSequence {
            segments: segments.into_iter().map(Into::into).collect(),
        };
        read_row_ids(&proto.encode_to_vec()).unwrap()
    }

    #[test]
    fn slicing_a_range_can_increase_its_encoded_size() {
        let whole = RowIdSequence::from(0..128);
        let tail = RowIdSequence::from(127..128);
        let budget = write_row_ids(&whole).len();
        assert!(write_row_ids(&tail).len() > budget);
        let plan = plan_carried_lineage_spill(budget, &whole, &RowDatasetVersionSequence::new());
        assert!(plan.row_ids);
        assert!(!plan.created_at);
        assert!(!plan.last_updated_at);
    }

    #[rstest]
    #[case::range(RowIdSequence::from(0..128))]
    #[case::high_range(RowIdSequence::from((1 << 49)..(1 << 49) + 128))]
    #[case::bitmap(RowIdSequence::from((0..128).map(|i| i * 2).collect::<Vec<_>>().as_slice()))]
    #[case::holes(RowIdSequence::from((0..128).filter(|i| *i != 63).collect::<Vec<_>>().as_slice()))]
    #[case::sorted_array(RowIdSequence::from((0..32).map(|i| i * (1_u64 << 40)).collect::<Vec<_>>().as_slice()))]
    #[case::maximum_value(RowIdSequence::from([u64::MAX - 3, u64::MAX - 1, u64::MAX].as_slice()))]
    #[case::u16_boundary(RowIdSequence::from([0, 1, 65535, 65536, 65537].as_slice()))]
    #[case::u32_boundary(RowIdSequence::from([0, 1, u32::MAX as u64, u32::MAX as u64 + 1, u32::MAX as u64 + 2].as_slice()))]
    #[case::array(RowIdSequence::from((0..128).map(|i| (i * 97) % 128).collect::<Vec<_>>().as_slice()))]
    #[case::explicit_array(sequence_from_segments(vec![U64Segment::Array((0..128).collect::<Vec<u64>>().into())]))]
    #[case::segments(sequence_from_segments(vec![U64Segment::Range(0..64), U64Segment::Range(128..192)]))]
    #[test]
    fn planned_bound_covers_reencoded_slices(#[case] sequence: RowIdSequence) {
        let bound = sequence
            .segments()
            .iter()
            .map(segment_slice_bound)
            .sum::<u128>();
        let rows = sequence.len();
        for start in 0..rows {
            for end in start + 1..=rows {
                let sizes = [start, end - start, rows - end];
                let chunks = rechunk_sequences([sequence.clone()], sizes, false).unwrap();
                assert!(
                    write_row_ids(&chunks[1]).len() as u128 <= bound,
                    "slice {start}..{end}, {sequence:?}"
                );
            }
        }
        let exact = bound as usize;
        let empty = RowDatasetVersionSequence::new();
        assert!(!plan_carried_lineage_spill(exact, &sequence, &empty).row_ids);
        assert!(plan_carried_lineage_spill(exact - 1, &sequence, &empty).row_ids);
    }

    #[test]
    fn version_bound_covers_run_boundaries() {
        let versions = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..127),
                    version: 127,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(127..130),
                    version: 128,
                },
            ],
        };
        let bound = versions
            .runs
            .iter()
            .map(|run| segment_slice_bound(&run.span) + 22)
            .sum::<u128>();
        for cut in 1..130 {
            let chunks =
                rechunk_version_sequences([versions.clone()], [cut, 130 - cut], false).unwrap();
            for chunk in chunks {
                assert!(write_dataset_versions(&chunk).len() as u128 <= bound);
            }
        }
        assert!(
            !plan_carried_lineage_spill(200 * 1024, &RowIdSequence::from(0..10_000_000), &versions)
                .any()
        );
    }
}
