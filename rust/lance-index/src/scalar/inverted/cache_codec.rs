// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache codec impls for FTS index entries.
//!
//! Serializes [`PostingList`] and [`Positions`] cache values for persistent
//! cache backends. The format is a small variant tag plus a JSON header for
//! scalar metadata, with Arrow-backed payload sections written as zero-copy
//! Arrow IPC streams via [`lance_arrow::ipc`]. The raw byte buffer inside
//! [`SharedPositionStream`] is written via [`write_len_prefixed_bytes`] and
//! read back via [`read_len_prefixed_bytes_at`] -- both zero-copy slices into
//! the input `Bytes` allocation.
//!
//! This is the FTS counterpart of `partition_serde.rs` for vector indices.

use std::io::Write;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, Float32Array, LargeBinaryArray, ListArray, RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use lance_arrow::ipc::{
    read_ipc_stream_single_at, read_len_prefixed_bytes_at, write_ipc_stream,
    write_len_prefixed_bytes,
};
use lance_core::cache::CacheCodecImpl;
use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};

use super::index::{
    CompressedPositionStorage, CompressedPostingList, PlainPostingList, PositionStreamCodec,
    Positions, PostingList, PostingTailCodec, SharedPositionStream,
};

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

const POSTING_VARIANT_PLAIN: u8 = 0;
const POSTING_VARIANT_COMPRESSED: u8 = 1;

const POSITIONS_TAG_NONE: u8 = 0;
const POSITIONS_TAG_LEGACY: u8 = 1;
const POSITIONS_TAG_SHARED: u8 = 2;

const POSTING_TAIL_CODEC_FIXED32: u8 = 0;
const POSTING_TAIL_CODEC_VARINT_DELTA: u8 = 1;

const POSITION_STREAM_CODEC_VARINT_DOC_DELTA: u8 = 0;
const POSITION_STREAM_CODEC_PACKED_DELTA: u8 = 1;

// ---------------------------------------------------------------------------
// Codec enum byte mappings
// ---------------------------------------------------------------------------

fn posting_tail_codec_to_u8(c: PostingTailCodec) -> u8 {
    match c {
        PostingTailCodec::Fixed32 => POSTING_TAIL_CODEC_FIXED32,
        PostingTailCodec::VarintDelta => POSTING_TAIL_CODEC_VARINT_DELTA,
    }
}

fn u8_to_posting_tail_codec(v: u8) -> Result<PostingTailCodec> {
    match v {
        POSTING_TAIL_CODEC_FIXED32 => Ok(PostingTailCodec::Fixed32),
        POSTING_TAIL_CODEC_VARINT_DELTA => Ok(PostingTailCodec::VarintDelta),
        _ => Err(Error::io(format!("unknown posting tail codec: {v}"))),
    }
}

fn position_stream_codec_to_u8(c: PositionStreamCodec) -> u8 {
    match c {
        PositionStreamCodec::VarintDocDelta => POSITION_STREAM_CODEC_VARINT_DOC_DELTA,
        PositionStreamCodec::PackedDelta => POSITION_STREAM_CODEC_PACKED_DELTA,
    }
}

fn u8_to_position_stream_codec(v: u8) -> Result<PositionStreamCodec> {
    match v {
        POSITION_STREAM_CODEC_VARINT_DOC_DELTA => Ok(PositionStreamCodec::VarintDocDelta),
        POSITION_STREAM_CODEC_PACKED_DELTA => Ok(PositionStreamCodec::PackedDelta),
        _ => Err(Error::io(format!("unknown position stream codec: {v}"))),
    }
}

// ---------------------------------------------------------------------------
// Header / tag I/O helpers (mirrors partition_serde.rs)
// ---------------------------------------------------------------------------

fn write_json_header(writer: &mut dyn Write, header: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(header)?;
    write_len_prefixed_bytes(writer, &bytes)?;
    Ok(())
}

fn read_json_header<T: serde::de::DeserializeOwned>(data: &Bytes, offset: &mut usize) -> Result<T> {
    let bytes = read_len_prefixed_bytes_at(data, offset).map_err(|e| Error::io(e.to_string()))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::io(format!("failed to deserialize cache header: {e}")))
}

fn write_u8(writer: &mut dyn Write, value: u8) -> Result<()> {
    writer
        .write_all(&[value])
        .map_err(|e| Error::io(format!("failed to write tag byte: {e}")))
}

fn read_u8(data: &Bytes, offset: &mut usize) -> Result<u8> {
    let bytes = data.as_ref();
    if *offset >= bytes.len() {
        return Err(Error::io(
            "truncated cache entry: missing tag byte".to_string(),
        ));
    }
    let v = bytes[*offset];
    *offset += 1;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Position storage serde (shared by PostingList variants and Positions)
// ---------------------------------------------------------------------------

const POSITION_LIST_COLUMN: &str = "position_list";
const BLOCK_OFFSETS_COLUMN: &str = "block_offsets";
const ROW_IDS_COLUMN: &str = "row_ids";
const FREQUENCIES_COLUMN: &str = "frequencies";
const BLOCKS_COLUMN: &str = "blocks";

#[derive(Serialize, Deserialize)]
struct SharedPositionsHeader {
    codec: u8,
}

fn write_position_storage(
    writer: &mut dyn Write,
    storage: &CompressedPositionStorage,
) -> Result<()> {
    match storage {
        CompressedPositionStorage::LegacyPerDoc(list) => {
            write_u8(writer, POSITIONS_TAG_LEGACY)?;
            let schema = Arc::new(Schema::new(vec![Field::new(
                POSITION_LIST_COLUMN,
                list.data_type().clone(),
                list.is_nullable(),
            )]));
            let batch = RecordBatch::try_new(schema, vec![Arc::new(list.clone())])?;
            write_ipc_stream(&batch, writer)?;
        }
        CompressedPositionStorage::SharedStream(stream) => {
            write_u8(writer, POSITIONS_TAG_SHARED)?;
            let header = SharedPositionsHeader {
                codec: position_stream_codec_to_u8(stream.codec()),
            };
            write_json_header(writer, &header)?;

            let offsets = UInt32Array::from(stream.block_offsets().to_vec());
            let schema = Arc::new(Schema::new(vec![Field::new(
                BLOCK_OFFSETS_COLUMN,
                DataType::UInt32,
                false,
            )]));
            let batch = RecordBatch::try_new(schema, vec![Arc::new(offsets)])?;
            write_ipc_stream(&batch, writer)?;

            write_len_prefixed_bytes(writer, stream.bytes())?;
        }
    }
    Ok(())
}

fn read_position_storage(
    data: &Bytes,
    offset: &mut usize,
    tag: u8,
) -> Result<CompressedPositionStorage> {
    match tag {
        POSITIONS_TAG_LEGACY => {
            let batch =
                read_ipc_stream_single_at(data, offset).map_err(|e| Error::io(e.to_string()))?;
            let list = batch
                .column(0)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| Error::io("legacy position column is not a ListArray".to_string()))?
                .clone();
            Ok(CompressedPositionStorage::LegacyPerDoc(list))
        }
        POSITIONS_TAG_SHARED => {
            let header: SharedPositionsHeader = read_json_header(data, offset)?;
            let codec = u8_to_position_stream_codec(header.codec)?;

            let batch =
                read_ipc_stream_single_at(data, offset).map_err(|e| Error::io(e.to_string()))?;
            let block_offsets = batch
                .column(0)
                .as_primitive_opt::<UInt32Type>()
                .ok_or_else(|| Error::io("block_offsets column is not UInt32".to_string()))?
                .values()
                .to_vec();

            // Zero copy: read_len_prefixed_bytes_at returns a Bytes slice
            // backed by the same allocation as `data`, and SharedPositionStream
            // now stores its byte buffer as Bytes -- no copy on read.
            let bytes =
                read_len_prefixed_bytes_at(data, offset).map_err(|e| Error::io(e.to_string()))?;

            Ok(CompressedPositionStorage::SharedStream(
                SharedPositionStream::new(codec, block_offsets, bytes),
            ))
        }
        other => Err(Error::io(format!("unknown positions tag: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// PostingList codec
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct PlainPostingHeader {
    max_score: Option<f32>,
}

#[derive(Serialize, Deserialize)]
struct CompressedPostingHeader {
    max_score: f32,
    length: u32,
    posting_tail_codec: u8,
}

impl CacheCodecImpl for PostingList {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()> {
        match self {
            Self::Plain(plain) => {
                write_u8(writer, POSTING_VARIANT_PLAIN)?;
                serialize_plain(writer, plain)
            }
            Self::Compressed(compressed) => {
                write_u8(writer, POSTING_VARIANT_COMPRESSED)?;
                serialize_compressed(writer, compressed)
            }
        }
    }

    fn deserialize(data: &Bytes) -> Result<Self> {
        let mut offset = 0;
        let variant = read_u8(data, &mut offset)?;
        match variant {
            POSTING_VARIANT_PLAIN => Ok(Self::Plain(deserialize_plain(data, &mut offset)?)),
            POSTING_VARIANT_COMPRESSED => {
                Ok(Self::Compressed(deserialize_compressed(data, &mut offset)?))
            }
            other => Err(Error::io(format!("unknown PostingList variant: {other}"))),
        }
    }
}

fn serialize_plain(writer: &mut dyn Write, plain: &PlainPostingList) -> Result<()> {
    let header = PlainPostingHeader {
        max_score: plain.max_score,
    };
    write_json_header(writer, &header)?;

    let row_ids = UInt64Array::new(plain.row_ids.clone(), None);
    let frequencies = Float32Array::new(plain.frequencies.clone(), None);
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_IDS_COLUMN, DataType::UInt64, false),
        Field::new(FREQUENCIES_COLUMN, DataType::Float32, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(frequencies)])?;
    write_ipc_stream(&batch, writer)?;

    match &plain.positions {
        Some(list) => {
            // Plain postings can only carry per-doc legacy positions; reuse
            // the shared encoder.
            write_position_storage(
                writer,
                &CompressedPositionStorage::LegacyPerDoc(list.clone()),
            )?;
        }
        None => write_u8(writer, POSITIONS_TAG_NONE)?,
    }
    Ok(())
}

fn deserialize_plain(data: &Bytes, offset: &mut usize) -> Result<PlainPostingList> {
    let header: PlainPostingHeader = read_json_header(data, offset)?;

    let batch = read_ipc_stream_single_at(data, offset).map_err(|e| Error::io(e.to_string()))?;
    let row_ids = batch
        .column(0)
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| Error::io("row_ids column is not UInt64".to_string()))?
        .values()
        .clone();
    let frequencies = batch
        .column(1)
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| Error::io("frequencies column is not Float32".to_string()))?
        .values()
        .clone();

    let positions_tag = read_u8(data, offset)?;
    let positions = match positions_tag {
        POSITIONS_TAG_NONE => None,
        POSITIONS_TAG_LEGACY => match read_position_storage(data, offset, positions_tag)? {
            CompressedPositionStorage::LegacyPerDoc(list) => Some(list),
            CompressedPositionStorage::SharedStream(_) => {
                unreachable!("shared stream tag was read as legacy variant (this is a bug)")
            }
        },
        other => {
            return Err(Error::io(format!(
                "Plain posting list cannot have positions tag {other}"
            )));
        }
    };

    Ok(PlainPostingList::new(
        row_ids,
        frequencies,
        header.max_score,
        positions,
    ))
}

fn serialize_compressed(writer: &mut dyn Write, posting: &CompressedPostingList) -> Result<()> {
    let header = CompressedPostingHeader {
        max_score: posting.max_score,
        length: posting.length,
        posting_tail_codec: posting_tail_codec_to_u8(posting.posting_tail_codec),
    };
    write_json_header(writer, &header)?;

    let schema = Arc::new(Schema::new(vec![Field::new(
        BLOCKS_COLUMN,
        DataType::LargeBinary,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(posting.blocks.clone())])?;
    write_ipc_stream(&batch, writer)?;

    match &posting.positions {
        Some(storage) => write_position_storage(writer, storage)?,
        None => write_u8(writer, POSITIONS_TAG_NONE)?,
    }
    Ok(())
}

fn deserialize_compressed(data: &Bytes, offset: &mut usize) -> Result<CompressedPostingList> {
    let header: CompressedPostingHeader = read_json_header(data, offset)?;
    let posting_tail_codec = u8_to_posting_tail_codec(header.posting_tail_codec)?;

    let batch = read_ipc_stream_single_at(data, offset).map_err(|e| Error::io(e.to_string()))?;
    let blocks = batch
        .column(0)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| Error::io("blocks column is not a LargeBinaryArray".to_string()))?
        .clone();

    let positions_tag = read_u8(data, offset)?;
    let positions = if positions_tag == POSITIONS_TAG_NONE {
        None
    } else {
        Some(read_position_storage(data, offset, positions_tag)?)
    };

    Ok(CompressedPostingList::new(
        blocks,
        header.max_score,
        header.length,
        posting_tail_codec,
        positions,
    ))
}

// ---------------------------------------------------------------------------
// Positions codec
// ---------------------------------------------------------------------------

impl CacheCodecImpl for Positions {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()> {
        write_position_storage(writer, &self.0)
    }

    fn deserialize(data: &Bytes) -> Result<Self> {
        let mut offset = 0;
        let tag = read_u8(data, &mut offset)?;
        if tag == POSITIONS_TAG_NONE {
            return Err(Error::io(
                "Positions cache entry cannot encode the None variant".to_string(),
            ));
        }
        let storage = read_position_storage(data, &mut offset, tag)?;
        Ok(Self(storage))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use arrow::buffer::ScalarBuffer;
    use arrow_array::LargeBinaryArray;
    use arrow_array::builder::{Int32Builder, ListBuilder};
    use bytes::Bytes;
    use lance_core::cache::CacheCodecImpl;

    use super::super::index::{
        CompressedPositionStorage, CompressedPostingList, PlainPostingList, PositionStreamCodec,
        Positions, PostingList, PostingTailCodec, SharedPositionStream,
    };

    fn legacy_positions(rows: &[&[i32]]) -> arrow_array::ListArray {
        let mut builder = ListBuilder::new(Int32Builder::new());
        for row in rows {
            for v in *row {
                builder.values().append_value(*v);
            }
            builder.append(true);
        }
        builder.finish()
    }

    fn assert_plain_eq(a: &PlainPostingList, b: &PlainPostingList) {
        assert_eq!(a.row_ids.as_ref(), b.row_ids.as_ref());
        assert_eq!(a.frequencies.as_ref(), b.frequencies.as_ref());
        assert_eq!(a.max_score, b.max_score);
        match (&a.positions, &b.positions) {
            (None, None) => {}
            (Some(x), Some(y)) => assert_eq!(x, y),
            _ => panic!("positions mismatch"),
        }
    }

    fn assert_position_storage_eq(a: &CompressedPositionStorage, b: &CompressedPositionStorage) {
        match (a, b) {
            (
                CompressedPositionStorage::LegacyPerDoc(x),
                CompressedPositionStorage::LegacyPerDoc(y),
            ) => assert_eq!(x, y),
            (
                CompressedPositionStorage::SharedStream(x),
                CompressedPositionStorage::SharedStream(y),
            ) => {
                assert_eq!(x.codec(), y.codec());
                assert_eq!(x.block_offsets(), y.block_offsets());
                assert_eq!(x.bytes(), y.bytes());
            }
            _ => panic!("position storage variant mismatch"),
        }
    }

    fn roundtrip_posting_list(entry: &PostingList) -> PostingList {
        let mut buf = Vec::new();
        entry.serialize(&mut buf).unwrap();
        PostingList::deserialize(&Bytes::from(buf)).unwrap()
    }

    fn roundtrip_positions(entry: &Positions) -> Positions {
        let mut buf = Vec::new();
        entry.serialize(&mut buf).unwrap();
        Positions::deserialize(&Bytes::from(buf)).unwrap()
    }

    fn assert_slice_points_into_bytes(slice: &[u8], bytes: &Bytes) {
        let slice_start = slice.as_ptr() as usize;
        let slice_end = slice_start + slice.len();
        let bytes_start = bytes.as_ptr() as usize;
        let bytes_end = bytes_start + bytes.len();
        assert!(
            slice_start >= bytes_start && slice_end <= bytes_end,
            "slice [{slice_start:#x}, {slice_end:#x}) should point into bytes \
             [{bytes_start:#x}, {bytes_end:#x})",
        );
    }

    #[test]
    fn plain_posting_list_no_positions_roundtrip() {
        let plain = PlainPostingList::new(
            ScalarBuffer::from(vec![10u64, 20, 30]),
            ScalarBuffer::from(vec![0.5f32, 1.0, 1.5]),
            Some(2.0),
            None,
        );
        let entry = PostingList::Plain(plain.clone());
        match roundtrip_posting_list(&entry) {
            PostingList::Plain(restored) => assert_plain_eq(&plain, &restored),
            PostingList::Compressed(_) => panic!("expected Plain variant"),
        }
    }

    #[test]
    fn plain_posting_list_with_positions_roundtrip() {
        let plain = PlainPostingList::new(
            ScalarBuffer::from(vec![1u64, 2]),
            ScalarBuffer::from(vec![1.0f32, 1.0]),
            None,
            Some(legacy_positions(&[&[3, 7], &[1, 4, 9]])),
        );
        let entry = PostingList::Plain(plain.clone());
        match roundtrip_posting_list(&entry) {
            PostingList::Plain(restored) => assert_plain_eq(&plain, &restored),
            PostingList::Compressed(_) => panic!("expected Plain variant"),
        }
    }

    #[test]
    fn compressed_posting_list_no_positions_roundtrip() {
        // Two synthetic block payloads -- content is opaque to the codec.
        let blocks = LargeBinaryArray::from_opt_vec(vec![
            Some(&[1u8, 2, 3, 4, 5][..]),
            Some(&[6, 7, 8, 9, 10][..]),
        ]);
        let posting =
            CompressedPostingList::new(blocks, 3.5, 42, PostingTailCodec::VarintDelta, None);
        let entry = PostingList::Compressed(posting.clone());
        match roundtrip_posting_list(&entry) {
            PostingList::Compressed(restored) => {
                assert_eq!(restored.max_score, posting.max_score);
                assert_eq!(restored.length, posting.length);
                assert_eq!(restored.posting_tail_codec, posting.posting_tail_codec);
                assert_eq!(restored.blocks, posting.blocks);
                assert!(restored.positions.is_none());
            }
            PostingList::Plain(_) => panic!("expected Compressed variant"),
        }
    }

    #[test]
    fn compressed_posting_list_legacy_positions_roundtrip() {
        let blocks = LargeBinaryArray::from_opt_vec(vec![Some(&[1u8, 2, 3][..])]);
        let posting = CompressedPostingList::new(
            blocks,
            1.25,
            5,
            PostingTailCodec::Fixed32,
            Some(CompressedPositionStorage::LegacyPerDoc(legacy_positions(
                &[&[0, 4, 8]],
            ))),
        );
        let entry = PostingList::Compressed(posting.clone());
        match roundtrip_posting_list(&entry) {
            PostingList::Compressed(restored) => {
                assert_eq!(restored.posting_tail_codec, posting.posting_tail_codec);
                assert_position_storage_eq(
                    restored.positions.as_ref().unwrap(),
                    posting.positions.as_ref().unwrap(),
                );
            }
            PostingList::Plain(_) => panic!("expected Compressed variant"),
        }
    }

    #[test]
    fn compressed_posting_list_shared_stream_roundtrip() {
        for codec in [
            PositionStreamCodec::VarintDocDelta,
            PositionStreamCodec::PackedDelta,
        ] {
            let blocks = LargeBinaryArray::from_opt_vec(vec![Some(&[9u8; 16][..])]);
            let stream = SharedPositionStream::new(
                codec,
                vec![0u32, 4, 11],
                Bytes::from((0u8..32).collect::<Vec<_>>()),
            );
            let posting = CompressedPostingList::new(
                blocks,
                7.0,
                3,
                PostingTailCodec::VarintDelta,
                Some(CompressedPositionStorage::SharedStream(stream)),
            );
            let entry = PostingList::Compressed(posting.clone());
            match roundtrip_posting_list(&entry) {
                PostingList::Compressed(restored) => {
                    assert_position_storage_eq(
                        restored.positions.as_ref().unwrap(),
                        posting.positions.as_ref().unwrap(),
                    );
                }
                PostingList::Plain(_) => panic!("expected Compressed variant"),
            }
        }
    }

    #[test]
    fn shared_stream_deserialize_borrows_from_input_bytes() {
        let blocks = LargeBinaryArray::from_opt_vec(vec![Some(&[9u8; 16][..])]);
        let expected_stream = SharedPositionStream::new(
            PositionStreamCodec::PackedDelta,
            vec![0u32, 4, 11],
            Bytes::from((0u8..32).collect::<Vec<_>>()),
        );
        let posting = CompressedPostingList::new(
            blocks,
            7.0,
            3,
            PostingTailCodec::VarintDelta,
            Some(CompressedPositionStorage::SharedStream(
                expected_stream.clone(),
            )),
        );
        let mut buf = Vec::new();
        PostingList::Compressed(posting)
            .serialize(&mut buf)
            .unwrap();
        let serialized = Bytes::from(buf);

        let restored = PostingList::deserialize(&serialized).unwrap();
        let PostingList::Compressed(restored) = restored else {
            panic!("expected Compressed variant");
        };
        let Some(CompressedPositionStorage::SharedStream(stream)) = restored.positions else {
            panic!("expected shared-stream positions");
        };

        assert_eq!(stream.codec(), expected_stream.codec());
        assert_eq!(stream.block_offsets(), expected_stream.block_offsets());
        assert_eq!(stream.bytes(), expected_stream.bytes());
        assert_slice_points_into_bytes(stream.bytes(), &serialized);
    }

    #[test]
    fn positions_legacy_roundtrip() {
        let positions = Positions(CompressedPositionStorage::LegacyPerDoc(legacy_positions(
            &[&[1, 2, 3], &[], &[10]],
        )));
        let restored = roundtrip_positions(&positions);
        assert_position_storage_eq(&positions.0, &restored.0);
    }

    #[test]
    fn positions_shared_stream_roundtrip() {
        let stream = SharedPositionStream::new(
            PositionStreamCodec::PackedDelta,
            vec![0u32, 8],
            Bytes::from(vec![0xAAu8; 24]),
        );
        let positions = Positions(CompressedPositionStorage::SharedStream(stream));
        let restored = roundtrip_positions(&positions);
        assert_position_storage_eq(&positions.0, &restored.0);
    }

    #[test]
    fn truncated_data_errors() {
        let plain = PlainPostingList::new(
            ScalarBuffer::from(vec![1u64]),
            ScalarBuffer::from(vec![1.0f32]),
            None,
            None,
        );
        let entry = PostingList::Plain(plain);
        let mut buf = Vec::new();
        entry.serialize(&mut buf).unwrap();
        buf.truncate(buf.len() / 2);
        assert!(PostingList::deserialize(&Bytes::from(buf)).is_err());
    }
}
