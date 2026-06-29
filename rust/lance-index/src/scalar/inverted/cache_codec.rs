// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache codec impls for FTS index entries.
//!
//! Serializes [`PostingList`] and [`Positions`] cache values for persistent
//! cache backends, behind the stabilized envelope written by
//! [`CacheCodec`](lance_core::cache::CacheCodec).
//!
//! Every variant uses a protobuf header (see `protos-cache/cache.proto`, with the
//! tail/position codecs and position-storage kind as proto enums) followed by
//! 64-byte-aligned Arrow IPC sections and, where applicable, raw blobs:
//!
//! - the compressed posting list: an IPC section for `blocks`, then the
//!   position sections (legacy IPC, or shared block-offsets IPC + a raw blob of
//!   the [`SharedPositionStream`] byte buffer, which has its own portable
//!   encoding);
//! - the plain posting list: an IPC section of `(row_ids, frequencies)`, then
//!   an optional legacy position IPC section;
//! - the standalone [`Positions`] codec: the position sections alone.
//!
//! All sections read back zero-copy via [`lance_arrow::ipc`]. This is the FTS
//! counterpart of `partition_serde.rs` for vector indices.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, Float32Array, LargeBinaryArray, ListArray, RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};
use lance_core::{Error, Result};

use crate::cache_pb::{
    CompressedPostingHeader, PlainPostingHeader, PositionStorage as PbPositionStorage,
    PositionStreamCodec as PbPositionStreamCodec, PositionsHeader, PostingListGroupHeader,
    PostingTailCodec as PbPostingTailCodec,
};

use super::index::{
    CompressedPositionStorage, CompressedPostingList, PlainPostingList, PositionStreamCodec,
    Positions, PostingList, PostingListGroup, PostingTailCodec, SharedPositionStream,
};

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

const POSTING_VARIANT_PLAIN: u8 = 0;
const POSTING_VARIANT_COMPRESSED: u8 = 1;

// ---------------------------------------------------------------------------
// Codec enum mappings
// ---------------------------------------------------------------------------

// Posting lists carry their discriminants as protobuf enums in the header;
// these map to/from the in-memory Rust enums.

fn posting_tail_codec_to_proto(c: PostingTailCodec) -> PbPostingTailCodec {
    match c {
        PostingTailCodec::Fixed32 => PbPostingTailCodec::Fixed32,
        PostingTailCodec::VarintDelta => PbPostingTailCodec::VarintDelta,
    }
}

fn proto_to_posting_tail_codec(c: PbPostingTailCodec) -> PostingTailCodec {
    match c {
        PbPostingTailCodec::Fixed32 => PostingTailCodec::Fixed32,
        PbPostingTailCodec::VarintDelta => PostingTailCodec::VarintDelta,
    }
}

fn position_stream_codec_to_proto(c: PositionStreamCodec) -> PbPositionStreamCodec {
    match c {
        PositionStreamCodec::VarintDocDelta => PbPositionStreamCodec::VarintDocDelta,
        PositionStreamCodec::PackedDelta => PbPositionStreamCodec::PackedDelta,
    }
}

fn proto_to_position_stream_codec(c: PbPositionStreamCodec) -> PositionStreamCodec {
    match c {
        PbPositionStreamCodec::VarintDocDelta => PositionStreamCodec::VarintDocDelta,
        PbPositionStreamCodec::PackedDelta => PositionStreamCodec::PackedDelta,
    }
}

// ---------------------------------------------------------------------------
// Position storage sections (shared by PostingList variants and Positions)
// ---------------------------------------------------------------------------

const POSITION_LIST_COLUMN: &str = "position_list";
const BLOCK_OFFSETS_COLUMN: &str = "block_offsets";
const ROW_IDS_COLUMN: &str = "row_ids";
const FREQUENCIES_COLUMN: &str = "frequencies";
const BLOCKS_COLUMN: &str = "blocks";

fn legacy_positions_batch(list: &ListArray) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        POSITION_LIST_COLUMN,
        list.data_type().clone(),
        list.is_nullable(),
    )]));
    Ok(RecordBatch::try_new(schema, vec![Arc::new(list.clone())])?)
}

fn read_legacy_positions(r: &mut CacheEntryReader<'_>) -> Result<ListArray> {
    let batch = r.read_ipc()?;
    Ok(batch
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| Error::io("legacy position column is not a ListArray".to_string()))?
        .clone())
}

/// Write the position sections (the bytes after the header) for `storage`. The
/// caller's header proto carries the storage kind and shared-stream codec.
fn write_position_sections(
    w: &mut CacheEntryWriter<'_>,
    storage: &CompressedPositionStorage,
) -> Result<()> {
    match storage {
        CompressedPositionStorage::LegacyPerDoc(list) => {
            w.write_ipc(&legacy_positions_batch(list)?)?;
        }
        CompressedPositionStorage::SharedStream(stream) => {
            let offsets = UInt32Array::from(stream.block_offsets().to_vec());
            let schema = Arc::new(Schema::new(vec![Field::new(
                BLOCK_OFFSETS_COLUMN,
                DataType::UInt32,
                false,
            )]));
            let batch = RecordBatch::try_new(schema, vec![Arc::new(offsets)])?;
            w.write_ipc(&batch)?;
            w.write_raw(stream.bytes())?;
        }
    }
    Ok(())
}

/// Read the position sections for the given `storage` kind and (for shared
/// streams) `stream_codec`. Returns `None` only when `storage` is
/// [`PbPositionStorage::None`].
fn read_position_sections(
    r: &mut CacheEntryReader<'_>,
    storage: PbPositionStorage,
    stream_codec: PositionStreamCodec,
) -> Result<Option<CompressedPositionStorage>> {
    match storage {
        PbPositionStorage::None => Ok(None),
        PbPositionStorage::Legacy => Ok(Some(CompressedPositionStorage::LegacyPerDoc(
            read_legacy_positions(r)?,
        ))),
        PbPositionStorage::Shared => {
            let batch = r.read_ipc()?;
            let block_offsets = batch
                .column(0)
                .as_primitive_opt::<UInt32Type>()
                .ok_or_else(|| Error::io("block_offsets column is not UInt32".to_string()))?
                .values()
                .to_vec();
            // Zero copy: read_raw returns a Bytes slice backed by the same
            // allocation as the input, and SharedPositionStream stores its byte
            // buffer as Bytes -- no copy on read.
            let bytes = r.read_raw()?;
            Ok(Some(CompressedPositionStorage::SharedStream(
                SharedPositionStream::new(stream_codec, block_offsets, bytes),
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// PostingList codec
// ---------------------------------------------------------------------------

impl CacheCodecImpl for PostingList {
    const TYPE_ID: &'static str = "lance.fts.PostingList";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        match self {
            Self::Plain(plain) => {
                w.write_u8(POSTING_VARIANT_PLAIN)?;
                serialize_plain(w, plain)
            }
            Self::Compressed(compressed) => {
                w.write_u8(POSTING_VARIANT_COMPRESSED)?;
                serialize_compressed(w, compressed)
            }
        }
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        let variant = r.read_u8()?;
        match variant {
            POSTING_VARIANT_PLAIN => Ok(Self::Plain(deserialize_plain(r)?)),
            POSTING_VARIANT_COMPRESSED => Ok(Self::Compressed(deserialize_compressed(r)?)),
            other => Err(Error::io(format!("unknown PostingList variant: {other}"))),
        }
    }
}

fn serialize_plain(w: &mut CacheEntryWriter<'_>, plain: &PlainPostingList) -> Result<()> {
    // Plain postings carry only per-doc legacy positions (or none).
    let position_storage = if plain.positions.is_some() {
        PbPositionStorage::Legacy
    } else {
        PbPositionStorage::None
    };
    let header = PlainPostingHeader {
        max_score: plain.max_score,
        position_storage: position_storage as i32,
    };
    w.write_header(&header)?;

    let row_ids = UInt64Array::new(plain.row_ids.clone(), None);
    let frequencies = Float32Array::new(plain.frequencies.clone(), None);
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_IDS_COLUMN, DataType::UInt64, false),
        Field::new(FREQUENCIES_COLUMN, DataType::Float32, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(frequencies)])?;
    w.write_ipc(&batch)?;

    if let Some(list) = &plain.positions {
        w.write_ipc(&legacy_positions_batch(list)?)?;
    }
    Ok(())
}

fn deserialize_plain(r: &mut CacheEntryReader<'_>) -> Result<PlainPostingList> {
    let header: PlainPostingHeader = r.read_header()?;

    let batch = r.read_ipc()?;
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

    let positions = match header.position_storage() {
        PbPositionStorage::None => None,
        PbPositionStorage::Legacy => Some(read_legacy_positions(r)?),
        PbPositionStorage::Shared => {
            return Err(Error::io(
                "Plain posting list cannot have a shared position stream".to_string(),
            ));
        }
    };

    Ok(PlainPostingList::new(
        row_ids,
        frequencies,
        header.max_score,
        positions,
    ))
}

/// The compressed posting list is serialized with a protobuf header followed
/// by 64-byte-aligned Arrow IPC sections (for the `blocks`, and for shared
/// position block-offsets) and a raw blob (for the shared position byte
/// stream, which already has its own portable encoding).
fn serialize_compressed(
    w: &mut CacheEntryWriter<'_>,
    posting: &CompressedPostingList,
) -> Result<()> {
    let (position_storage, position_stream_codec) = match &posting.positions {
        None => (PbPositionStorage::None, PbPositionStreamCodec::default()),
        Some(CompressedPositionStorage::LegacyPerDoc(_)) => {
            (PbPositionStorage::Legacy, PbPositionStreamCodec::default())
        }
        Some(CompressedPositionStorage::SharedStream(stream)) => (
            PbPositionStorage::Shared,
            position_stream_codec_to_proto(stream.codec()),
        ),
    };

    let header = CompressedPostingHeader {
        max_score: posting.max_score,
        length: posting.length,
        posting_tail_codec: posting_tail_codec_to_proto(posting.posting_tail_codec) as i32,
        position_storage: position_storage as i32,
        position_stream_codec: position_stream_codec as i32,
    };
    w.write_header(&header)?;

    let schema = Arc::new(Schema::new(vec![Field::new(
        BLOCKS_COLUMN,
        DataType::LargeBinary,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(posting.blocks.clone())])?;
    w.write_ipc(&batch)?;

    if let Some(storage) = &posting.positions {
        write_position_sections(w, storage)?;
    }
    Ok(())
}

fn deserialize_compressed(r: &mut CacheEntryReader<'_>) -> Result<CompressedPostingList> {
    let header: CompressedPostingHeader = r.read_header()?;
    let posting_tail_codec = proto_to_posting_tail_codec(header.posting_tail_codec());

    let batch = r.read_ipc()?;
    let blocks = batch
        .column(0)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| Error::io("blocks column is not a LargeBinaryArray".to_string()))?
        .clone();

    let stream_codec = proto_to_position_stream_codec(header.position_stream_codec());
    let positions = read_position_sections(r, header.position_storage(), stream_codec)?;

    Ok(CompressedPostingList::new(
        blocks,
        header.max_score,
        header.length,
        posting_tail_codec,
        positions,
    ))
}

// ---------------------------------------------------------------------------
// PostingListGroup codec
// ---------------------------------------------------------------------------

/// Serializes a [`PostingListGroup`] as a member-count header followed by each
/// member posting list written **inline** through the same writer. Reusing the
/// [`PostingList`] codec inline (rather than into per-member sub-buffers) keeps
/// each member's Arrow IPC sections 64-byte aligned within the group entry, so
/// they read back zero-copy. Member bodies are self-delimiting, so they need no
/// length prefixes to separate them. See issue #7040.
impl CacheCodecImpl for PostingListGroup {
    const TYPE_ID: &'static str = "lance.fts.PostingListGroup";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        let count = u32::try_from(self.posting_lists.len())
            .map_err(|_| Error::io("posting list group too large to serialize".to_string()))?;
        w.write_header(&PostingListGroupHeader { count })?;
        for posting in &self.posting_lists {
            posting.serialize(w)?;
        }
        Ok(())
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        let header: PostingListGroupHeader = r.read_header()?;
        let mut posting_lists = Vec::with_capacity(header.count as usize);
        for _ in 0..header.count {
            posting_lists.push(PostingList::deserialize(r)?);
        }
        Ok(Self::new(posting_lists))
    }
}

// ---------------------------------------------------------------------------
// Positions codec
// ---------------------------------------------------------------------------

impl CacheCodecImpl for Positions {
    const TYPE_ID: &'static str = "lance.fts.Positions";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        let (position_storage, position_stream_codec) = match &self.0 {
            CompressedPositionStorage::LegacyPerDoc(_) => {
                (PbPositionStorage::Legacy, PbPositionStreamCodec::default())
            }
            CompressedPositionStorage::SharedStream(stream) => (
                PbPositionStorage::Shared,
                position_stream_codec_to_proto(stream.codec()),
            ),
        };
        let header = PositionsHeader {
            position_storage: position_storage as i32,
            position_stream_codec: position_stream_codec as i32,
        };
        w.write_header(&header)?;
        write_position_sections(w, &self.0)
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        let header: PositionsHeader = r.read_header()?;
        let stream_codec = proto_to_position_stream_codec(header.position_stream_codec());
        read_position_sections(r, header.position_storage(), stream_codec)?
            .map(Self)
            .ok_or_else(|| {
                Error::io("Positions cache entry cannot encode the None variant".to_string())
            })
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
    use lance_core::Result;
    use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};

    use super::super::index::{
        CompressedPositionStorage, CompressedPostingList, PlainPostingList, PositionStreamCodec,
        Positions, PostingList, PostingListGroup, PostingTailCodec, SharedPositionStream,
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

    /// Serialize a codec body (no envelope) into a standalone buffer.
    fn body_bytes<T: CacheCodecImpl>(entry: &T) -> Bytes {
        let mut buf = Vec::new();
        let mut w = CacheEntryWriter::new(&mut buf);
        entry.serialize(&mut w).unwrap();
        Bytes::from(buf)
    }

    /// Deserialize a codec body (no envelope) at the current build's version.
    fn from_body<T: CacheCodecImpl>(data: &Bytes) -> Result<T> {
        let mut r = CacheEntryReader::new(data, 0, T::CURRENT_VERSION);
        T::deserialize(&mut r)
    }

    fn roundtrip_posting_list(entry: &PostingList) -> PostingList {
        from_body::<PostingList>(&body_bytes(entry)).unwrap()
    }

    fn roundtrip_positions(entry: &Positions) -> Positions {
        from_body::<Positions>(&body_bytes(entry)).unwrap()
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
        let serialized = body_bytes(&PostingList::Compressed(posting));

        let restored = from_body::<PostingList>(&serialized).unwrap();
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
    fn posting_list_group_roundtrip() {
        // Mix of plain and compressed members, including an empty group.
        let plain = PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(vec![1u64, 2, 3]),
            ScalarBuffer::from(vec![1.0f32, 2.0, 3.0]),
            Some(4.0),
            None,
        ));
        let compressed = PostingList::Compressed(CompressedPostingList::new(
            LargeBinaryArray::from_opt_vec(vec![Some(&[1u8, 2, 3][..])]),
            2.5,
            7,
            PostingTailCodec::VarintDelta,
            None,
        ));

        for members in [
            Vec::new(),
            vec![plain.clone()],
            vec![plain.clone(), compressed, plain],
        ] {
            let group = PostingListGroup::new(members.clone());
            let restored = from_body::<PostingListGroup>(&body_bytes(&group)).unwrap();
            assert_eq!(restored.posting_lists.len(), members.len());
            for (a, b) in members.iter().zip(restored.posting_lists.iter()) {
                match (a, b) {
                    (PostingList::Plain(x), PostingList::Plain(y)) => assert_plain_eq(x, y),
                    (PostingList::Compressed(x), PostingList::Compressed(y)) => {
                        assert_eq!(x.blocks, y.blocks);
                        assert_eq!(x.length, y.length);
                        assert_eq!(x.max_score, y.max_score);
                    }
                    _ => panic!("variant mismatch in group roundtrip"),
                }
            }
        }
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
        let mut buf = body_bytes(&entry).to_vec();
        buf.truncate(buf.len() / 2);
        assert!(from_body::<PostingList>(&Bytes::from(buf)).is_err());
    }

    /// Tests covering the stabilized envelope + compressed proto format,
    /// exercised through the full type-erased [`CacheCodec`] (envelope + body).
    mod stable_format {
        use std::sync::Arc;

        use arrow_array::Array;
        use lance_core::cache::CacheCodec;
        use prost::Message;

        use super::*;
        use crate::cache_pb::{CompressedPostingHeader, PostingTailCodec as PbPostingTailCodec};

        type ArcAny = Arc<dyn std::any::Any + Send + Sync>;

        fn codec() -> CacheCodec {
            CacheCodec::from_impl::<PostingList>()
        }

        /// Serialize an entry through the full codec (envelope + body).
        fn serialize_entry(entry: PostingList) -> Vec<u8> {
            let any: ArcAny = Arc::new(entry);
            let mut buf = Vec::new();
            codec().serialize(&any, &mut buf).unwrap();
            buf
        }

        /// A `Bytes` whose base address is 64-byte aligned, modelling a backend
        /// that reads cache entries into an aligned buffer.
        fn aligned_bytes(payload: &[u8]) -> Bytes {
            const ALIGN: usize = 64;
            let mut v = vec![0u8; payload.len() + ALIGN];
            let pad = (ALIGN - (v.as_ptr() as usize % ALIGN)) % ALIGN;
            v[pad..pad + payload.len()].copy_from_slice(payload);
            Bytes::from(v).slice(pad..pad + payload.len())
        }

        fn compressed_with_shared_positions() -> PostingList {
            let blocks =
                LargeBinaryArray::from_opt_vec(vec![Some(&[9u8; 48][..]), Some(&[1u8; 48])]);
            let stream = SharedPositionStream::new(
                PositionStreamCodec::PackedDelta,
                vec![0u32, 4, 11],
                Bytes::from((0u8..64).collect::<Vec<_>>()),
            );
            PostingList::Compressed(CompressedPostingList::new(
                blocks,
                7.0,
                3,
                PostingTailCodec::VarintDelta,
                Some(CompressedPositionStorage::SharedStream(stream)),
            ))
        }

        /// The compressed `blocks` (an aligned IPC section) and the shared
        /// position blob (a raw section) must both be borrowed zero-copy from
        /// the input even though the envelope pushes them to a non-zero,
        /// non-aligned starting offset.
        #[test]
        fn compressed_sections_are_zero_copy_through_envelope() {
            let serialized = aligned_bytes(&serialize_entry(compressed_with_shared_positions()));
            let restored = codec().deserialize(&serialized).hit().unwrap();
            let restored = restored.downcast::<PostingList>().unwrap();
            let PostingList::Compressed(restored) = restored.as_ref() else {
                panic!("expected Compressed");
            };

            let base = serialized.as_ptr() as usize;
            let end = base + serialized.len();
            let points_in = |ptr: usize| ptr >= base && ptr < end;

            // blocks IPC section decoded in place (no realigning memcpy).
            for buf in restored.blocks.to_data().buffers() {
                assert!(
                    points_in(buf.as_ptr() as usize),
                    "blocks buffer was realigned out of the input — misaligned IPC section",
                );
            }
            // shared position raw blob borrowed in place.
            let Some(CompressedPositionStorage::SharedStream(stream)) = &restored.positions else {
                panic!("expected shared stream");
            };
            assert!(points_in(stream.bytes().as_ptr() as usize));
        }

        /// Every member of a `PostingListGroup` must also decode zero-copy. The
        /// group writes its members inline so each member's IPC sections stay
        /// 64-byte aligned within the entry; embedding members in per-member
        /// sub-buffers would land them at arbitrary offsets and force a
        /// realigning memcpy on load.
        #[test]
        fn group_member_sections_are_zero_copy_through_envelope() {
            let make_member = |fill: u8| {
                let blocks =
                    LargeBinaryArray::from_opt_vec(vec![Some(&[fill; 48][..]), Some(&[fill; 48])]);
                PostingList::Compressed(CompressedPostingList::new(
                    blocks,
                    7.0,
                    3,
                    PostingTailCodec::VarintDelta,
                    None,
                ))
            };
            let group = PostingListGroup::new(vec![make_member(9), make_member(1)]);

            let group_codec = CacheCodec::from_impl::<PostingListGroup>();
            let any: ArcAny = Arc::new(group);
            let mut buf = Vec::new();
            group_codec.serialize(&any, &mut buf).unwrap();
            let serialized = aligned_bytes(&buf);

            let restored = group_codec.deserialize(&serialized).hit().unwrap();
            let restored = restored.downcast::<PostingListGroup>().unwrap();

            let base = serialized.as_ptr() as usize;
            let end = base + serialized.len();
            let points_in = |ptr: usize| ptr >= base && ptr < end;

            assert_eq!(restored.posting_lists.len(), 2);
            for member in &restored.posting_lists {
                let PostingList::Compressed(member) = member else {
                    panic!("expected Compressed member");
                };
                for buf in member.blocks.to_data().buffers() {
                    assert!(
                        points_in(buf.as_ptr() as usize),
                        "group member blocks buffer was realigned out of the input — \
                         misaligned IPC section",
                    );
                }
            }
        }

        /// The plain posting's row-id/frequency IPC section must also decode
        /// zero-copy through the envelope + proto header.
        #[test]
        fn plain_sections_are_zero_copy_through_envelope() {
            let plain = PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from((0u64..64).collect::<Vec<_>>()),
                ScalarBuffer::from(vec![1.0f32; 64]),
                Some(2.0),
                None,
            ));
            let serialized = aligned_bytes(&serialize_entry(plain));
            let restored = codec().deserialize(&serialized).hit().unwrap();
            let restored = restored.downcast::<PostingList>().unwrap();
            let PostingList::Plain(restored) = restored.as_ref() else {
                panic!("expected Plain");
            };

            let base = serialized.as_ptr() as usize;
            let end = base + serialized.len();
            // The row_ids ScalarBuffer must borrow from the input allocation.
            let ptr = restored.row_ids.as_ptr() as usize;
            assert!(
                ptr >= base && ptr < end,
                "row_ids buffer was realigned out of the input — misaligned IPC section",
            );
        }

        /// Additive proto fields (lever #1) must not break decoding: an unknown
        /// field number appended to the header is ignored.
        #[test]
        fn header_proto_ignores_unknown_fields() {
            let header = CompressedPostingHeader {
                max_score: 1.5,
                length: 9,
                posting_tail_codec: PbPostingTailCodec::VarintDelta as i32,
                ..Default::default()
            };
            let mut bytes = header.encode_to_vec();
            // Append an unknown field #15, varint wire type (0), value 7.
            bytes.push(15 << 3);
            bytes.push(7);
            let decoded = CompressedPostingHeader::decode(bytes.as_slice()).unwrap();
            assert_eq!(decoded.length, 9);
            assert_eq!(decoded.max_score, 1.5);
        }

        /// An entry written by a different codec (foreign TYPE_ID) misses.
        #[test]
        fn foreign_type_id_is_miss() {
            // A PostingListGroup entry carries a different TYPE_ID in its
            // envelope; reading it as a PostingList must miss, not misread it.
            let group = PostingListGroup::new(vec![]);
            let any: ArcAny = Arc::new(group);
            let mut buf = Vec::new();
            CacheCodec::from_impl::<PostingListGroup>()
                .serialize(&any, &mut buf)
                .unwrap();
            assert!(codec().deserialize(&Bytes::from(buf)).hit().is_none());
        }

        /// An entry written by a newer build (higher type_version) misses.
        #[test]
        fn future_type_version_is_miss() {
            let mut buf = serialize_entry(compressed_with_shared_positions());
            // Patch the envelope's type_version (magic[4] + ver[1] + len[2] +
            // type_id[N]) to a value beyond what this build understands.
            let type_id_len = u16::from_le_bytes([buf[5], buf[6]]) as usize;
            let version_off = 4 + 1 + 2 + type_id_len;
            buf[version_off..version_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(codec().deserialize(&Bytes::from(buf)).hit().is_none());
        }

        /// A pre-stabilization blob (no magic) self-heals to a miss.
        #[test]
        fn pre_stabilization_blob_is_miss() {
            // Old format led with a u64 LE length prefix, never our magic.
            let mut blob = (30u64).to_le_bytes().to_vec();
            blob.extend_from_slice(&[0u8; 30]);
            assert!(codec().deserialize(&Bytes::from(blob)).hit().is_none());
        }

        /// A structurally-valid envelope whose body leads with an out-of-range
        /// variant tag self-heals to a `BodyError` miss rather than panicking or
        /// misreading the remaining bytes.
        #[test]
        fn unknown_posting_variant_is_miss() {
            use lance_core::cache::{CacheDecode, CacheMissReason};

            let mut buf = serialize_entry(compressed_with_shared_positions());
            // The variant tag is the first body byte, right after the envelope
            // (magic[4] + ver[1] + type_id_len[2] + type_id[N] + type_version[4]).
            let type_id_len = u16::from_le_bytes([buf[5], buf[6]]) as usize;
            let variant_off = 4 + 1 + 2 + type_id_len + 4;
            buf[variant_off] = 2; // neither PLAIN (0) nor COMPRESSED (1)
            match codec().deserialize(&Bytes::from(buf)) {
                CacheDecode::Miss(reason) => assert_eq!(reason, CacheMissReason::BodyError),
                CacheDecode::Hit(_) => panic!("expected a BodyError miss, got a hit"),
            }
        }
    }
}
