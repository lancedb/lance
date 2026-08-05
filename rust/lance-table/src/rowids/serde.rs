// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{format::pb, rowids::bitmap::Bitmap};
use lance_core::{Error, Result};

use super::{RowIdSequence, U64Segment, encoded_array::EncodedU64Array};
use prost::Message;

const ROW_ID_METADATA: &str = "row ID metadata";

fn corrupt_row_id_metadata(message: impl Into<String>) -> Error {
    Error::corrupt_file_named(ROW_ID_METADATA, message)
}

fn validate_range(segment_type: &str, start: u64, end: u64) -> Result<usize> {
    let len = end.checked_sub(start).ok_or_else(|| {
        corrupt_row_id_metadata(format!(
            "{segment_type} range start {start} exceeds end {end}"
        ))
    })?;
    usize::try_from(len).map_err(|_| {
        corrupt_row_id_metadata(format!(
            "{segment_type} range length {len} for start {start} and end {end} exceeds usize::MAX"
        ))
    })
}

fn validate_packed_array_length(array_type: &str, byte_len: usize, width: usize) -> Result<()> {
    if !byte_len.is_multiple_of(width) {
        return Err(corrupt_row_id_metadata(format!(
            "encoded {array_type} array byte length {byte_len} is not a multiple of element width {width}"
        )));
    }
    Ok(())
}

fn first_descending_pair(array: &EncodedU64Array) -> Option<(usize, u64, u64)> {
    match array {
        EncodedU64Array::U16 { offsets, .. } => offsets
            .windows(2)
            .position(|pair| pair[0] > pair[1])
            .map(|index| (index, offsets[index] as u64, offsets[index + 1] as u64)),
        EncodedU64Array::U32 { offsets, .. } => offsets
            .windows(2)
            .position(|pair| pair[0] > pair[1])
            .map(|index| (index, offsets[index] as u64, offsets[index + 1] as u64)),
        EncodedU64Array::U64(values) => values
            .windows(2)
            .position(|pair| pair[0] > pair[1])
            .map(|index| (index, values[index], values[index + 1])),
    }
}

impl TryFrom<pb::RowIdSequence> for RowIdSequence {
    type Error = Error;

    fn try_from(pb: pb::RowIdSequence) -> Result<Self> {
        Ok(Self(
            pb.segments
                .into_iter()
                .map(U64Segment::try_from)
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

impl TryFrom<pb::U64Segment> for U64Segment {
    type Error = Error;

    fn try_from(pb: pb::U64Segment) -> Result<Self> {
        use pb::u64_segment as pb_seg;
        use pb::u64_segment::Segment::*;
        match pb.segment {
            Some(Range(pb_seg::Range { start, end })) => {
                validate_range("Range", start, end)?;
                Ok(Self::Range(start..end))
            }
            Some(RangeWithHoles(pb_seg::RangeWithHoles { start, end, holes })) => {
                validate_range("RangeWithHoles", start, end)?;
                let holes = holes
                    .ok_or_else(|| Error::invalid_input("missing hole"))?
                    .try_into()?;
                Ok(Self::RangeWithHoles {
                    range: start..end,
                    holes,
                })
            }
            Some(RangeWithBitmap(pb_seg::RangeWithBitmap { start, end, bitmap })) => {
                let range_len = validate_range("RangeWithBitmap", start, end)?;
                let expected_bitmap_len = range_len.div_ceil(8);
                if bitmap.len() != expected_bitmap_len {
                    return Err(corrupt_row_id_metadata(format!(
                        "RangeWithBitmap byte length {} does not match expected {expected_bitmap_len} for range start {start}, end {end}, and length {range_len}",
                        bitmap.len()
                    )));
                }
                let remainder = range_len % 8;
                if remainder != 0 {
                    let padding_mask = !((1_u8 << remainder) - 1);
                    let last_byte = bitmap[expected_bitmap_len - 1];
                    if last_byte & padding_mask != 0 {
                        return Err(corrupt_row_id_metadata(format!(
                            "RangeWithBitmap padding bits must be zero for range start {start}, end {end}, and length {range_len}: last byte {last_byte:#04x} has padding mask {padding_mask:#04x} set"
                        )));
                    }
                }
                Ok(Self::RangeWithBitmap {
                    range: start..end,
                    bitmap: Bitmap {
                        data: bitmap,
                        len: range_len,
                    },
                })
            }
            Some(SortedArray(array)) => {
                let array = EncodedU64Array::try_from(array)?;
                if let Some((index, previous, next)) = first_descending_pair(&array) {
                    return Err(corrupt_row_id_metadata(format!(
                        "SortedArray values are not sorted at indices {index} and {}: {previous} exceeds {next}",
                        index + 1
                    )));
                }
                Ok(Self::SortedArray(array))
            }
            Some(Array(array)) => Ok(Self::Array(EncodedU64Array::try_from(array)?)),
            // TODO: why non-exhaustive?
            // Some(_) => Err(Error::invalid_input("unknown segment type")),
            None => Err(Error::invalid_input("missing segment type")),
        }
    }
}

impl TryFrom<pb::EncodedU64Array> for EncodedU64Array {
    type Error = Error;

    fn try_from(pb: pb::EncodedU64Array) -> Result<Self> {
        use pb::encoded_u64_array as pb_arr;
        use pb::encoded_u64_array::Array::*;
        match pb.array {
            Some(U16Array(pb_arr::U16Array { base, offsets })) => {
                validate_packed_array_length("u16", offsets.len(), 2)?;
                let offsets = offsets
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Ok(Self::U16 { base, offsets })
            }
            Some(U32Array(pb_arr::U32Array { base, offsets })) => {
                validate_packed_array_length("u32", offsets.len(), 4)?;
                let offsets = offsets
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(Self::U32 { base, offsets })
            }
            Some(U64Array(pb_arr::U64Array { values })) => {
                validate_packed_array_length("u64", values.len(), 8)?;
                let values = values
                    .chunks_exact(8)
                    .map(|chunk| {
                        u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();
                Ok(Self::U64(values))
            }
            // TODO: shouldn't this enum be non-exhaustive?
            // Some(_) => Err(Error::invalid_input("unknown array type")),
            None => Err(Error::invalid_input("missing array type")),
        }
    }
}

impl From<RowIdSequence> for pb::RowIdSequence {
    fn from(sequence: RowIdSequence) -> Self {
        Self {
            segments: sequence.0.into_iter().map(pb::U64Segment::from).collect(),
        }
    }
}

impl From<U64Segment> for pb::U64Segment {
    fn from(segment: U64Segment) -> Self {
        match segment {
            U64Segment::Range(range) => Self {
                segment: Some(pb::u64_segment::Segment::Range(pb::u64_segment::Range {
                    start: range.start,
                    end: range.end,
                })),
            },
            U64Segment::RangeWithHoles { range, holes } => Self {
                segment: Some(pb::u64_segment::Segment::RangeWithHoles(
                    pb::u64_segment::RangeWithHoles {
                        start: range.start,
                        end: range.end,
                        holes: Some(holes.into()),
                    },
                )),
            },
            U64Segment::RangeWithBitmap { range, bitmap } => Self {
                segment: Some(pb::u64_segment::Segment::RangeWithBitmap(
                    pb::u64_segment::RangeWithBitmap {
                        start: range.start,
                        end: range.end,
                        bitmap: bitmap.data,
                    },
                )),
            },
            U64Segment::SortedArray(array) => Self {
                segment: Some(pb::u64_segment::Segment::SortedArray(array.into())),
            },
            U64Segment::Array(array) => Self {
                segment: Some(pb::u64_segment::Segment::Array(array.into())),
            },
        }
    }
}

impl From<EncodedU64Array> for pb::EncodedU64Array {
    fn from(array: EncodedU64Array) -> Self {
        match array {
            EncodedU64Array::U16 { base, offsets } => Self {
                array: Some(pb::encoded_u64_array::Array::U16Array(
                    pb::encoded_u64_array::U16Array {
                        base,
                        offsets: offsets
                            .iter()
                            .flat_map(|&offset| offset.to_le_bytes().to_vec())
                            .collect(),
                    },
                )),
            },
            EncodedU64Array::U32 { base, offsets } => Self {
                array: Some(pb::encoded_u64_array::Array::U32Array(
                    pb::encoded_u64_array::U32Array {
                        base,
                        offsets: offsets
                            .iter()
                            .flat_map(|&offset| offset.to_le_bytes().to_vec())
                            .collect(),
                    },
                )),
            },
            EncodedU64Array::U64(values) => Self {
                array: Some(pb::encoded_u64_array::Array::U64Array(
                    pb::encoded_u64_array::U64Array {
                        values: values
                            .iter()
                            .flat_map(|&value| value.to_le_bytes().to_vec())
                            .collect(),
                    },
                )),
            },
        }
    }
}

/// Serialize a rowid sequence to a buffer.
pub fn write_row_ids(sequence: &RowIdSequence) -> Vec<u8> {
    let pb_sequence = pb::RowIdSequence::from(sequence.clone());
    pb_sequence.encode_to_vec()
}

/// Deserialize a rowid sequence from some bytes.
pub fn read_row_ids(reader: &[u8]) -> Result<RowIdSequence> {
    let pb_sequence = pb::RowIdSequence::decode(reader)?;
    RowIdSequence::try_from(pb_sequence)
}

#[cfg(test)]
mod test {
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    fn assert_corrupt_segment(segment: pb::u64_segment::Segment, message: &str) {
        let serialized = pb::RowIdSequence {
            segments: vec![pb::U64Segment {
                segment: Some(segment),
            }],
        }
        .encode_to_vec();

        let error = read_row_ids(&serialized).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error.to_string().contains(message),
            "expected error to contain {message:?}, got {error}"
        );
    }

    #[test]
    fn test_write_read_row_ids() {
        let mut sequence = RowIdSequence::from(0..20);
        sequence.0.push(U64Segment::Range(30..100));
        sequence.0.push(U64Segment::RangeWithHoles {
            range: 100..200,
            holes: EncodedU64Array::U64(vec![104, 108, 150]),
        });
        let mut bitmap = Bitmap::new_empty(100);
        bitmap.set(99);
        sequence.0.push(U64Segment::RangeWithBitmap {
            range: 200..300,
            bitmap,
        });
        sequence
            .0
            .push(U64Segment::SortedArray(EncodedU64Array::U16 {
                base: 200,
                offsets: vec![1, 2, 3],
            }));
        sequence
            .0
            .push(U64Segment::Array(EncodedU64Array::U64(vec![3, 1, 2])));

        let serialized = write_row_ids(&sequence);

        let sequence2 = read_row_ids(&serialized).unwrap();

        assert_eq!(sequence.0, sequence2.0);
    }

    #[rstest]
    #[case::u16(
        pb::encoded_u64_array::Array::U16Array(pb::encoded_u64_array::U16Array {
            base: 0,
            offsets: vec![1],
        }),
        "encoded u16 array byte length 1 is not a multiple of element width 2"
    )]
    #[case::u32(
        pb::encoded_u64_array::Array::U32Array(pb::encoded_u64_array::U32Array {
            base: 0,
            offsets: vec![1, 2, 3],
        }),
        "encoded u32 array byte length 3 is not a multiple of element width 4"
    )]
    #[case::u64(
        pb::encoded_u64_array::Array::U64Array(pb::encoded_u64_array::U64Array {
            values: vec![1, 2, 3, 4, 5, 6, 7],
        }),
        "encoded u64 array byte length 7 is not a multiple of element width 8"
    )]
    fn test_rejects_misaligned_encoded_array(
        #[case] array: pb::encoded_u64_array::Array,
        #[case] message: &str,
    ) {
        assert_corrupt_segment(
            pb::u64_segment::Segment::Array(pb::EncodedU64Array { array: Some(array) }),
            message,
        );
    }

    #[rstest]
    #[case::range(pb::u64_segment::Segment::Range(pb::u64_segment::Range {
        start: 10,
        end: 9,
    }))]
    #[case::range_with_holes(pb::u64_segment::Segment::RangeWithHoles(
        pb::u64_segment::RangeWithHoles {
            start: 10,
            end: 9,
            holes: Some(pb::EncodedU64Array {
                array: Some(pb::encoded_u64_array::Array::U64Array(
                    pb::encoded_u64_array::U64Array { values: Vec::new() },
                )),
            }),
        }
    ))]
    #[case::range_with_bitmap(pb::u64_segment::Segment::RangeWithBitmap(
        pb::u64_segment::RangeWithBitmap {
            start: 10,
            end: 9,
            bitmap: Vec::new(),
        }
    ))]
    fn test_rejects_reversed_range(#[case] segment: pb::u64_segment::Segment) {
        assert_corrupt_segment(segment, "range start 10 exceeds end 9");
    }

    #[rstest]
    #[case::short(vec![0])]
    #[case::long(vec![0, 0, 0])]
    fn test_rejects_incorrect_bitmap_length(#[case] bitmap: Vec<u8>) {
        assert_corrupt_segment(
            pb::u64_segment::Segment::RangeWithBitmap(pb::u64_segment::RangeWithBitmap {
                start: 5,
                end: 14,
                bitmap,
            }),
            "does not match expected 2 for range start 5, end 14, and length 9",
        );
    }

    #[test]
    fn test_rejects_set_bitmap_padding_bits() {
        assert_corrupt_segment(
            pb::u64_segment::Segment::RangeWithBitmap(pb::u64_segment::RangeWithBitmap {
                start: 5,
                end: 14,
                bitmap: vec![0xff, 0x03],
            }),
            "padding bits must be zero",
        );
    }

    #[rstest]
    #[case::u16(pb::encoded_u64_array::Array::U16Array(
        pb::encoded_u64_array::U16Array {
            base: 100,
            offsets: vec![2, 0, 1, 0],
        }
    ))]
    #[case::u32(pb::encoded_u64_array::Array::U32Array(
        pb::encoded_u64_array::U32Array {
            base: 100,
            offsets: vec![2, 0, 0, 0, 1, 0, 0, 0],
        }
    ))]
    #[case::u64(pb::encoded_u64_array::Array::U64Array(
        pb::encoded_u64_array::U64Array {
            values: vec![2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    ))]
    fn test_rejects_unsorted_sorted_array(#[case] array: pb::encoded_u64_array::Array) {
        assert_corrupt_segment(
            pb::u64_segment::Segment::SortedArray(pb::EncodedU64Array { array: Some(array) }),
            "SortedArray values are not sorted at indices 0 and 1: 2 exceeds 1",
        );
    }
}
