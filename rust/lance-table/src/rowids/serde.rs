// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::{Read, Write};

use super::{encoded_array::EncodedU64Array, RowIdSequence, U64Segment};

/// Serialize a rowid sequence to a writer.
pub fn write_row_ids<W: Write>(sequence: &RowIdSequence, writer: &mut W) -> std::io::Result<()> {
    // First, write number of segmens
    writer.write(&(sequence.0.len() as u64).to_le_bytes())?;

    // Then write each segment
    for segment in &sequence.0 {
        match segment {
            U64Segment::Tombstones(n) => {
                writer.write(&0u8.to_le_bytes())?;
                writer.write(&n.to_le_bytes())?;
            }
            U64Segment::Range(range) => {
                writer.write(&1u8.to_le_bytes())?;
                writer.write(&range.start.to_le_bytes())?;
                writer.write(&range.end.to_le_bytes())?;
            }
            U64Segment::SortedArray(array) => {
                writer.write(&2u8.to_le_bytes())?;
                write_array(&array, writer)?;
            }
            U64Segment::Array(array) => {
                writer.write(&3u8.to_le_bytes())?;
                write_array(&array, writer)?;
            }
        }
    }

    Ok(())
}

fn write_array<W: Write>(array: &EncodedU64Array, writer: &mut W) -> std::io::Result<()> {
    // length
    writer.write(&array.len().to_le_bytes())?;
    match array {
        EncodedU64Array::U16 { offsets, base } => {
            writer.write(&16u8.to_le_bytes())?;
            writer.write(&base.to_le_bytes())?;
            for &value in offsets {
                writer.write(&value.to_le_bytes())?;
            }
        }
        EncodedU64Array::U32 { offsets, base } => {
            writer.write(&32u8.to_le_bytes())?;
            writer.write(&base.to_le_bytes())?;
            for &value in offsets {
                writer.write(&value.to_le_bytes())?;
            }
        }
        EncodedU64Array::U64(offsets) => {
            writer.write(&64u8.to_le_bytes())?;
            for &value in offsets {
                writer.write(&value.to_le_bytes())?;
            }
        }
    }

    Ok(())
}

/// Deserialize a rowid sequence from a reader.
pub fn read_row_ids<R: Read>(reader: &mut R) -> std::io::Result<RowIdSequence> {
    let mut size_buf = [0u8; 8];
    reader.read_exact(&mut size_buf)?;
    let size = u64::from_le_bytes(size_buf);

    let mut segments = Vec::with_capacity(size as usize);

    for _ in 0..size {
        let mut segment_type_buf = [0u8; 1];
        reader.read_exact(&mut segment_type_buf)?;
        let segment_type = segment_type_buf[0];

        match segment_type {
            0 => {
                let mut n_buf = [0u8; 8];
                reader.read_exact(&mut n_buf)?;
                let n = u64::from_le_bytes(n_buf);
                segments.push(U64Segment::Tombstones(n));
            }
            1 => {
                let mut start_buf = [0u8; 8];
                reader.read_exact(&mut start_buf)?;
                let start = u64::from_le_bytes(start_buf);

                let mut end_buf = [0u8; 8];
                reader.read_exact(&mut end_buf)?;
                let end = u64::from_le_bytes(end_buf);

                segments.push(U64Segment::Range(start..end));
            }
            2 => {
                let array = read_array(reader)?;
                segments.push(U64Segment::SortedArray(array));
            }
            3 => {
                let array = read_array(reader)?;
                segments.push(U64Segment::Array(array));
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid segment type",
                ));
            }
        }
    }

    Ok(RowIdSequence(segments))
}

fn read_array<R: Read>(reader: &mut R) -> std::io::Result<EncodedU64Array> {
    let mut size_buf = [0u8; 8];
    reader.read_exact(&mut size_buf)?;
    let size = u64::from_le_bytes(size_buf);

    let mut array_buf = [0u8; 1];
    reader.read_exact(&mut array_buf)?;
    let array_type = array_buf[0];

    match array_type {
        16 => {
            let mut base_buf = [0u8; 8];
            reader.read_exact(&mut base_buf)?;
            let base = u64::from_le_bytes(base_buf);

            let mut offsets = Vec::with_capacity(size as usize);
            for _ in 0..size {
                let mut offset_buf = [0u8; 2];
                reader.read_exact(&mut offset_buf)?;
                let offset = u16::from_le_bytes(offset_buf);
                offsets.push(offset);
            }

            Ok(EncodedU64Array::U16 { offsets, base })
        }
        32 => {
            let mut base_buf = [0u8; 8];
            reader.read_exact(&mut base_buf)?;
            let base = u64::from_le_bytes(base_buf);

            let mut offsets = Vec::with_capacity(size as usize);
            for _ in 0..size {
                let mut offset_buf = [0u8; 4];
                reader.read_exact(&mut offset_buf)?;
                let offset = u32::from_le_bytes(offset_buf);
                offsets.push(offset);
            }

            Ok(EncodedU64Array::U32 { offsets, base })
        }
        64 => {
            let mut offsets = Vec::with_capacity(size as usize);
            for _ in 0..size {
                let mut offset_buf = [0u8; 8];
                reader.read_exact(&mut offset_buf)?;
                let offset = u64::from_le_bytes(offset_buf);
                offsets.push(offset);
            }

            Ok(EncodedU64Array::U64(offsets))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid array type",
        )),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::io::Cursor;

    #[test]
    fn test_write_read_row_ids() {
        let mut sequence = RowIdSequence::from(0..20);
        sequence.0.push(U64Segment::Tombstones(10));
        sequence.0.push(U64Segment::Range(30..100));
        sequence
            .0
            .push(U64Segment::SortedArray(EncodedU64Array::U16 {
                base: 200,
                offsets: vec![1, 2, 3],
            }));
        sequence
            .0
            .push(U64Segment::Array(EncodedU64Array::U64(vec![1, 2, 3])));

        let mut writer = Cursor::new(Vec::new());
        write_row_ids(&sequence, &mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let sequence2 = read_row_ids(&mut reader).unwrap();

        assert_eq!(sequence.0, sequence2.0);
    }
}
