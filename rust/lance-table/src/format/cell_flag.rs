// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Snapshot-level field-scoped Boolean flag metadata.

use std::io::Cursor;

use bytes::Buf;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use object_store::path::Path;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::pb;

const MAX_INLINE_CELL_FLAG_ROOT_BYTES: usize = 4 * 1024 * 1024;
const CELL_FLAG_BITMAP_MAGIC: &[u8; 4] = b"LCF1";
const CELL_FLAG_BITMAP_HEADER_BYTES: usize = 13;
const CELL_FLAG_BITMAP_ROARING: u8 = 0;
const CELL_FLAG_BITMAP_BITSET: u8 = 1;
const CELL_FLAG_BITMAP_ZSTD_ROARING: u8 = 2;
const CELL_FLAG_BITMAP_ZSTD_BITSET: u8 = 3;
const CELL_FLAG_BITMAP_STRIDE: u8 = 4;
// Mutation materialization has a 32 MiB input limit and reserves 1 MiB for
// Roaring container overhead. Keep persisted states below that boundary so a
// one-row change can always be represented instead of creating immutable state.
const MAX_CELL_FLAG_BITMAP_MEMORY_BYTES: usize = 30 * 1024 * 1024;
const CELL_FLAG_ROOT_MAGIC: &[u8; 4] = b"LCG1";
const CELL_FLAG_ROOT_HEADER_BYTES: usize = 13;
const CELL_FLAG_ROOT_RAW: u8 = 0;
const CELL_FLAG_ROOT_ZSTD: u8 = 1;
const MAX_CELL_FLAG_ROOT_PROTO_BYTES: usize = 32 * 1024 * 1024;
const MAX_CELL_FLAG_ROOT_MEMORY_BYTES: usize = 64 * 1024 * 1024;
// This format-level estimate is deliberately platform-independent. It covers
// the protobuf fragment vector at up to twice its final length while decoding,
// the typed fragment vector allocated during conversion, and fixed oneof/file
// storage on 64-bit targets with headroom for allocator bookkeeping.
const CELL_FLAG_ROOT_FRAGMENT_MEMORY_BYTES: usize = 512;
const CELL_FLAG_ROOT_FIXED_MEMORY_BYTES: usize = 256;

fn bitset_memory_size(bytes: usize) -> usize {
    // An array-backed Roaring container can use two bytes per set bit. A
    // bitset byte can represent eight set bits, so 16x plus container overhead
    // is a conservative retained-memory bound before decoding.
    bytes
        .saturating_mul(16)
        .saturating_add(bytes.div_ceil(8192).saturating_mul(32))
}

fn roaring_memory_size(bytes: usize) -> usize {
    // Portable Roaring is compact enough that its byte length is not an upper
    // bound on decoded Vec capacities and container allocations. Use the same
    // conservative expansion as bitset reconstruction so every LCF1 variant
    // declares one comparable retained-memory budget.
    bitset_memory_size(bytes)
}

/// Encode a non-empty bitmap using the smaller of portable Roaring and a dense bitset.
pub fn encode_cell_flag_bitmap(bitmap: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut roaring = Vec::with_capacity(bitmap.serialized_size());
    bitmap
        .serialize_into(&mut roaring)
        .map_err(|error| Error::internal(format!("Failed to encode Cell flag bitmap: {error}")))?;
    let roaring_decoded_size = roaring.len();
    let roaring_memory_size = roaring_memory_size(roaring_decoded_size);
    let compressed_roaring = zstd::bulk::compress(&roaring, 1).map_err(|error| {
        Error::internal(format!("Failed to compress Cell flag bitmap: {error}"))
    })?;
    let mut candidates = vec![(CELL_FLAG_BITMAP_ROARING, roaring_memory_size, roaring)];
    if compressed_roaring.len() + 8 < candidates[0].2.len() {
        let mut payload = Vec::with_capacity(compressed_roaring.len() + 8);
        payload.extend_from_slice(&(roaring_decoded_size as u64).to_le_bytes());
        payload.extend_from_slice(&compressed_roaring);
        candidates.push((CELL_FLAG_BITMAP_ZSTD_ROARING, roaring_memory_size, payload));
    }

    let bitset_len = bitmap.max().map_or(0, |value| value as usize / 8 + 1);
    if bitset_len < candidates[0].2.len() {
        let mut bitset = vec![0_u8; bitset_len];
        for value in bitmap.iter() {
            bitset[value as usize / 8] |= 1 << (value % 8);
        }
        let bitset_memory_size = bitset_memory_size(bitset.len());
        let compressed_bitset = zstd::bulk::compress(&bitset, 1).map_err(|error| {
            Error::internal(format!("Failed to compress Cell flag bitset: {error}"))
        })?;
        candidates.push((CELL_FLAG_BITMAP_BITSET, bitset_memory_size, bitset));
        if compressed_bitset.len() + 8 < candidates.last().expect("bitset candidate").2.len() {
            let mut payload = Vec::with_capacity(compressed_bitset.len() + 8);
            payload.extend_from_slice(&(bitset_len as u64).to_le_bytes());
            payload.extend_from_slice(&compressed_bitset);
            candidates.push((CELL_FLAG_BITMAP_ZSTD_BITSET, bitset_memory_size, payload));
        }
    }
    if let Some(stride) = encode_stride_bitmap(bitmap)
        && stride.len() < candidates[0].2.len()
    {
        candidates.push((CELL_FLAG_BITMAP_STRIDE, roaring_memory_size, stride));
    }
    let (encoding, memory_size, payload) = candidates
        .into_iter()
        .min_by_key(|(_, _, payload)| payload.len())
        .expect("Roaring candidate is always present");
    if memory_size > MAX_CELL_FLAG_BITMAP_MEMORY_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag bitmap requires {} bytes, maximum is {}",
            memory_size, MAX_CELL_FLAG_BITMAP_MEMORY_BYTES
        )));
    }
    let mut encoded = Vec::with_capacity(CELL_FLAG_BITMAP_HEADER_BYTES + payload.len());
    encoded.extend_from_slice(CELL_FLAG_BITMAP_MAGIC);
    encoded.push(encoding);
    encoded.extend_from_slice(&(memory_size as u64).to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

/// Encode a bitmap for latency-sensitive query binding.
///
/// This retains portable Roaring's fast decode path and only uses Zstd when it
/// reduces that representation directly. Denser encodings remain available to
/// transaction sidecars and external bitmap objects through
/// [`encode_cell_flag_bitmap`].
pub fn encode_cell_flag_query_bitmap(bitmap: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut roaring = Vec::with_capacity(bitmap.serialized_size());
    bitmap.serialize_into(&mut roaring).map_err(|error| {
        Error::internal(format!("Failed to encode Cell flag query bitmap: {error}"))
    })?;
    let decoded_size = roaring.len();
    let memory_size = roaring_memory_size(decoded_size);
    let compressed = zstd::bulk::compress(&roaring, 1).map_err(|error| {
        Error::internal(format!(
            "Failed to compress Cell flag query bitmap: {error}"
        ))
    })?;
    let (encoding, payload) = if compressed.len() + 8 < roaring.len() {
        let mut payload = Vec::with_capacity(compressed.len() + 8);
        payload.extend_from_slice(&(decoded_size as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);
        (CELL_FLAG_BITMAP_ZSTD_ROARING, payload)
    } else {
        (CELL_FLAG_BITMAP_ROARING, roaring)
    };
    if memory_size > MAX_CELL_FLAG_BITMAP_MEMORY_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag query bitmap requires {} bytes, maximum is {}",
            memory_size, MAX_CELL_FLAG_BITMAP_MEMORY_BYTES
        )));
    }
    let mut encoded = Vec::with_capacity(CELL_FLAG_BITMAP_HEADER_BYTES + payload.len());
    encoded.extend_from_slice(CELL_FLAG_BITMAP_MAGIC);
    encoded.push(encoding);
    encoded.extend_from_slice(&(memory_size as u64).to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

#[derive(Debug, Default)]
struct CellFlagRootWireSize {
    fragment_count: usize,
    dynamic_bytes: usize,
}

fn take_protobuf_delimited<'a>(bytes: &mut &'a [u8], context: &str) -> Result<&'a [u8]> {
    let length = prost::encoding::decode_varint(bytes)
        .map_err(|error| Error::invalid_input(format!("Invalid {context} length: {error}")))?;
    let length = usize::try_from(length)
        .map_err(|_| Error::invalid_input(format!("{context} length exceeds this platform")))?;
    if length > bytes.remaining() {
        return Err(Error::invalid_input(format!(
            "{context} length {} exceeds the {} remaining bytes",
            length,
            bytes.remaining()
        )));
    }
    let (value, remaining) = bytes.split_at(length);
    *bytes = remaining;
    Ok(value)
}

fn decode_protobuf_varint(bytes: &mut &[u8], context: &str) -> Result<()> {
    prost::encoding::decode_varint(bytes)
        .map(|_| ())
        .map_err(|error| Error::invalid_input(format!("Invalid {context}: {error}")))
}

fn skip_protobuf_field(
    bytes: &mut &[u8],
    tag: u32,
    wire_type: prost::encoding::WireType,
    context: &str,
) -> Result<()> {
    prost::encoding::skip_field(wire_type, tag, bytes, Default::default())
        .map_err(|error| Error::invalid_input(format!("Invalid {context}: {error}")))
}

fn scan_cell_flag_file_wire(mut bytes: &[u8]) -> Result<usize> {
    let mut dynamic_bytes = 0usize;
    while bytes.has_remaining() {
        let (tag, wire_type) = prost::encoding::decode_key(&mut bytes)
            .map_err(|error| Error::invalid_input(format!("Invalid Cell flag file: {error}")))?;
        match tag {
            1 | 4 => {
                prost::encoding::check_wire_type(
                    prost::encoding::WireType::LengthDelimited,
                    wire_type,
                )
                .map_err(|error| {
                    Error::invalid_input(format!("Invalid Cell flag file field {tag}: {error}"))
                })?;
                let value = take_protobuf_delimited(&mut bytes, "Cell flag file field")?;
                dynamic_bytes = dynamic_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| Error::invalid_input("Cell flag root dynamic size overflow"))?;
            }
            2 | 3 | 5 => {
                prost::encoding::check_wire_type(prost::encoding::WireType::Varint, wire_type)
                    .map_err(|error| {
                        Error::invalid_input(format!("Invalid Cell flag file field {tag}: {error}"))
                    })?;
                decode_protobuf_varint(&mut bytes, "Cell flag file field")?;
            }
            _ => skip_protobuf_field(&mut bytes, tag, wire_type, "Cell flag file field")?,
        }
    }
    Ok(dynamic_bytes)
}

fn scan_cell_flag_fragment_wire(mut bytes: &[u8]) -> Result<usize> {
    let mut dynamic_bytes = 0usize;
    while bytes.has_remaining() {
        let (tag, wire_type) = prost::encoding::decode_key(&mut bytes).map_err(|error| {
            Error::invalid_input(format!("Invalid Cell flag fragment: {error}"))
        })?;
        match tag {
            1..=3 => {
                prost::encoding::check_wire_type(prost::encoding::WireType::Varint, wire_type)
                    .map_err(|error| {
                        Error::invalid_input(format!(
                            "Invalid Cell flag fragment field {tag}: {error}"
                        ))
                    })?;
                decode_protobuf_varint(&mut bytes, "Cell flag fragment field")?;
            }
            4 => {
                prost::encoding::check_wire_type(
                    prost::encoding::WireType::LengthDelimited,
                    wire_type,
                )
                .map_err(|error| {
                    Error::invalid_input(format!(
                        "Invalid Cell flag fragment partial field: {error}"
                    ))
                })?;
                let file = take_protobuf_delimited(&mut bytes, "Cell flag fragment partial field")?;
                dynamic_bytes = dynamic_bytes
                    .checked_add(scan_cell_flag_file_wire(file)?)
                    .ok_or_else(|| Error::invalid_input("Cell flag root dynamic size overflow"))?;
            }
            5 => {
                prost::encoding::check_wire_type(
                    prost::encoding::WireType::LengthDelimited,
                    wire_type,
                )
                .map_err(|error| {
                    Error::invalid_input(format!(
                        "Invalid Cell flag fragment inline field: {error}"
                    ))
                })?;
                let inline =
                    take_protobuf_delimited(&mut bytes, "Cell flag fragment inline field")?;
                dynamic_bytes = dynamic_bytes
                    .checked_add(inline.len())
                    .ok_or_else(|| Error::invalid_input("Cell flag root dynamic size overflow"))?;
            }
            _ => skip_protobuf_field(&mut bytes, tag, wire_type, "Cell flag fragment field")?,
        }
    }
    Ok(dynamic_bytes)
}

fn scan_cell_flag_root_wire(mut bytes: &[u8]) -> Result<CellFlagRootWireSize> {
    let mut size = CellFlagRootWireSize::default();
    while bytes.has_remaining() {
        let (tag, wire_type) = prost::encoding::decode_key(&mut bytes)
            .map_err(|error| Error::invalid_input(format!("Invalid Cell flag root: {error}")))?;
        if tag == 1 {
            prost::encoding::check_wire_type(prost::encoding::WireType::LengthDelimited, wire_type)
                .map_err(|error| {
                    Error::invalid_input(format!("Invalid Cell flag root fragment field: {error}"))
                })?;
            let fragment = take_protobuf_delimited(&mut bytes, "Cell flag root fragment field")?;
            size.fragment_count = size
                .fragment_count
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("Cell flag root fragment count overflow"))?;
            size.dynamic_bytes = size
                .dynamic_bytes
                .checked_add(scan_cell_flag_fragment_wire(fragment)?)
                .ok_or_else(|| Error::invalid_input("Cell flag root dynamic size overflow"))?;
        } else {
            skip_protobuf_field(&mut bytes, tag, wire_type, "Cell flag root field")?;
        }
    }
    Ok(size)
}

fn cell_flag_root_memory_size_from_parts(
    fragment_count: usize,
    dynamic_bytes: usize,
    encoded_size: usize,
    decoded_size: usize,
) -> Result<usize> {
    let fragment_bytes = fragment_count
        .checked_mul(CELL_FLAG_ROOT_FRAGMENT_MEMORY_BYTES)
        .ok_or_else(|| Error::invalid_input("Cell flag root fragment memory size overflow"))?;
    let dynamic_bytes = dynamic_bytes
        .checked_mul(2)
        .ok_or_else(|| Error::invalid_input("Cell flag root dynamic memory size overflow"))?;
    let memory_size = encoded_size
        .checked_add(decoded_size)
        .and_then(|bytes| bytes.checked_add(fragment_bytes))
        .and_then(|bytes| bytes.checked_add(dynamic_bytes))
        .and_then(|bytes| bytes.checked_add(CELL_FLAG_ROOT_FIXED_MEMORY_BYTES))
        .ok_or_else(|| Error::invalid_input("Cell flag root memory size overflow"))?;
    if memory_size > MAX_CELL_FLAG_ROOT_MEMORY_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag root memory size {} exceeds the {} byte limit",
            memory_size, MAX_CELL_FLAG_ROOT_MEMORY_BYTES
        )));
    }
    Ok(memory_size)
}

fn cell_flag_root_memory_size(
    root: &pb::CellFlagRoot,
    encoded_size: usize,
    decoded_size: usize,
) -> Result<usize> {
    let dynamic_bytes = root.fragments.iter().try_fold(0usize, |total, fragment| {
        let bytes = match fragment.state.as_ref() {
            Some(pb::cell_flag_fragment::State::Partial(file)) => file
                .path
                .len()
                .checked_add(file.inline_bytes.as_ref().map_or(0, Vec::len)),
            Some(pb::cell_flag_fragment::State::InlinePartial(bytes)) => Some(bytes.len()),
            Some(pb::cell_flag_fragment::State::AllSet(_)) | None => Some(0),
        }
        .ok_or_else(|| Error::invalid_input("Cell flag root dynamic size overflow"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid_input("Cell flag root dynamic size overflow"))
    })?;
    cell_flag_root_memory_size_from_parts(
        root.fragments.len(),
        dynamic_bytes,
        encoded_size,
        decoded_size,
    )
}

/// Encode a complete root, compressing repeated fragment metadata as one unit.
pub fn encode_cell_flag_root(root: &pb::CellFlagRoot) -> Result<(Vec<u8>, usize)> {
    use prost::Message;

    let decoded_size = root.encoded_len();
    if decoded_size > MAX_CELL_FLAG_ROOT_PROTO_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag root protobuf size {} exceeds the {} byte limit",
            decoded_size, MAX_CELL_FLAG_ROOT_PROTO_BYTES
        )));
    }
    let raw = root.encode_to_vec();
    let compressed = zstd::bulk::compress(&raw, 1)
        .map_err(|error| Error::internal(format!("Failed to compress Cell flag root: {error}")))?;
    let (encoding, payload) = if compressed.len() < raw.len() {
        (CELL_FLAG_ROOT_ZSTD, compressed)
    } else {
        (CELL_FLAG_ROOT_RAW, raw)
    };
    let mut encoded = Vec::with_capacity(CELL_FLAG_ROOT_HEADER_BYTES + payload.len());
    encoded.extend_from_slice(CELL_FLAG_ROOT_MAGIC);
    encoded.push(encoding);
    encoded.extend_from_slice(&(decoded_size as u64).to_le_bytes());
    encoded.extend_from_slice(&payload);
    let memory_size = cell_flag_root_memory_size(root, encoded.len(), decoded_size)?;
    Ok((encoded, memory_size))
}

/// Decode a complete root and return its retained-memory declaration.
pub fn decode_cell_flag_root(bytes: &[u8]) -> Result<(pb::CellFlagRoot, usize)> {
    use prost::Message;

    if bytes.len() < CELL_FLAG_ROOT_HEADER_BYTES
        || &bytes[..CELL_FLAG_ROOT_MAGIC.len()] != CELL_FLAG_ROOT_MAGIC
    {
        return Err(Error::invalid_input(
            "Cell flag root has an invalid encoding header",
        ));
    }
    let decoded_size = u64::from_le_bytes(
        bytes[5..CELL_FLAG_ROOT_HEADER_BYTES]
            .try_into()
            .expect("root header length checked"),
    );
    let decoded_size = usize::try_from(decoded_size)
        .map_err(|_| Error::invalid_input("Cell flag root protobuf size exceeds this platform"))?;
    if decoded_size > MAX_CELL_FLAG_ROOT_PROTO_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag root protobuf size {} exceeds the {} byte limit",
            decoded_size, MAX_CELL_FLAG_ROOT_PROTO_BYTES
        )));
    }
    let payload = &bytes[CELL_FLAG_ROOT_HEADER_BYTES..];
    let decoded = match bytes[4] {
        CELL_FLAG_ROOT_RAW => {
            if payload.len() != decoded_size {
                return Err(Error::invalid_input(format!(
                    "Cell flag root has size {}, expected {}",
                    payload.len(),
                    decoded_size
                )));
            }
            payload.to_vec()
        }
        CELL_FLAG_ROOT_ZSTD => zstd::bulk::decompress(payload, decoded_size).map_err(|error| {
            Error::invalid_input(format!("Invalid compressed Cell flag root: {error}"))
        })?,
        encoding => {
            return Err(Error::invalid_input(format!(
                "Cell flag root has unknown encoding {}",
                encoding
            )));
        }
    };
    if decoded.len() != decoded_size {
        return Err(Error::invalid_input(format!(
            "Cell flag root decoded to {} bytes, expected {}",
            decoded.len(),
            decoded_size
        )));
    }
    let wire_size = scan_cell_flag_root_wire(decoded.as_slice())?;
    cell_flag_root_memory_size_from_parts(
        wire_size.fragment_count,
        wire_size.dynamic_bytes,
        bytes.len(),
        decoded_size,
    )?;
    let root = pb::CellFlagRoot::decode(decoded.as_slice())
        .map_err(|error| Error::invalid_input(format!("Invalid Cell flag root: {error}")))?;
    let memory_size = cell_flag_root_memory_size(&root, bytes.len(), decoded_size)?;
    Ok((root, memory_size))
}

/// Return the retained-memory budget declared by an adaptive bitmap.
pub fn cell_flag_bitmap_memory_size(bytes: &[u8]) -> Result<usize> {
    if bytes.len() < CELL_FLAG_BITMAP_HEADER_BYTES
        || &bytes[..CELL_FLAG_BITMAP_MAGIC.len()] != CELL_FLAG_BITMAP_MAGIC
    {
        return Err(Error::invalid_input(
            "Cell flag bitmap has an invalid encoding header",
        ));
    }
    let memory_size = u64::from_le_bytes(
        bytes[5..CELL_FLAG_BITMAP_HEADER_BYTES]
            .try_into()
            .expect("bitmap header length checked"),
    );
    let memory_size = usize::try_from(memory_size)
        .map_err(|_| Error::invalid_input("Cell flag bitmap memory size exceeds this platform"))?;
    if memory_size > MAX_CELL_FLAG_BITMAP_MEMORY_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag bitmap memory size {} exceeds the {} byte limit",
            memory_size, MAX_CELL_FLAG_BITMAP_MEMORY_BYTES
        )));
    }
    Ok(memory_size)
}

/// Decode and validate Lance's adaptive bitmap representation.
pub fn decode_cell_flag_bitmap(bytes: &[u8]) -> Result<RoaringBitmap> {
    let memory_size = cell_flag_bitmap_memory_size(bytes)?;
    let payload = &bytes[CELL_FLAG_BITMAP_HEADER_BYTES..];
    match bytes[4] {
        CELL_FLAG_BITMAP_ROARING => {
            let required_memory = roaring_memory_size(payload.len());
            if memory_size != required_memory {
                return Err(Error::invalid_input(format!(
                    "Cell flag Roaring bitmap declares memory size {}, expected {}",
                    memory_size, required_memory
                )));
            }
            RoaringBitmap::deserialize_from(&mut Cursor::new(payload)).map_err(|error| {
                Error::invalid_input(format!("Invalid Cell flag Roaring bitmap: {error}"))
            })
        }
        CELL_FLAG_BITMAP_BITSET => {
            let required_memory = bitset_memory_size(payload.len());
            if memory_size != required_memory {
                return Err(Error::invalid_input(format!(
                    "Cell flag bitset declares memory size {}, expected {}",
                    memory_size, required_memory
                )));
            }
            let values = payload.iter().enumerate().flat_map(|(byte_index, byte)| {
                (0..8).filter_map(move |bit| {
                    (byte & (1 << bit) != 0).then_some((byte_index * 8 + bit) as u32)
                })
            });
            RoaringBitmap::from_sorted_iter(values)
                .map_err(|error| Error::invalid_input(format!("Invalid Cell flag bitset: {error}")))
        }
        CELL_FLAG_BITMAP_ZSTD_ROARING => {
            let decoded = decode_zstd_cell_flag_bitmap_payload(payload, memory_size)?;
            RoaringBitmap::deserialize_from(&mut Cursor::new(decoded)).map_err(|error| {
                Error::invalid_input(format!(
                    "Invalid compressed Cell flag Roaring bitmap: {error}"
                ))
            })
        }
        CELL_FLAG_BITMAP_ZSTD_BITSET => {
            if payload.len() < 8 {
                return Err(Error::invalid_input(
                    "Compressed Cell flag bitset is missing its decoded length",
                ));
            }
            let decoded_size = u64::from_le_bytes(
                payload[..8]
                    .try_into()
                    .expect("compressed header length checked"),
            );
            let decoded_size = usize::try_from(decoded_size).map_err(|_| {
                Error::invalid_input("Compressed Cell flag bitset size exceeds this platform")
            })?;
            if bitset_memory_size(decoded_size) != memory_size {
                return Err(Error::invalid_input(
                    "Compressed Cell flag bitset has an invalid memory size",
                ));
            }
            let decoded = zstd::bulk::decompress(&payload[8..], decoded_size).map_err(|error| {
                Error::invalid_input(format!("Invalid compressed Cell flag bitset: {error}"))
            })?;
            if decoded.len() != decoded_size {
                return Err(Error::invalid_input(format!(
                    "Compressed Cell flag bitset decoded to {} bytes, expected {}",
                    decoded.len(),
                    decoded_size
                )));
            }
            let values = decoded.iter().enumerate().flat_map(|(byte_index, byte)| {
                (0..8).filter_map(move |bit| {
                    (byte & (1 << bit) != 0).then_some((byte_index * 8 + bit) as u32)
                })
            });
            RoaringBitmap::from_sorted_iter(values).map_err(|error| {
                Error::invalid_input(format!("Invalid compressed Cell flag bitset: {error}"))
            })
        }
        CELL_FLAG_BITMAP_STRIDE => decode_stride_bitmap(payload, memory_size),
        encoding => Err(Error::invalid_input(format!(
            "Cell flag bitmap has unknown encoding {}",
            encoding
        ))),
    }
}

fn encode_stride_bitmap(bitmap: &RoaringBitmap) -> Option<Vec<u8>> {
    let mut values = bitmap.iter();
    let start = values.next()?;
    let second = values.next();
    let step = second.map_or(1, |second| second - start);
    let mut previous = second.unwrap_or(start);
    for value in values {
        if value - previous != step {
            return None;
        }
        previous = value;
    }
    let count = u32::try_from(bitmap.len()).ok()?;
    let mut encoded = Vec::with_capacity(12);
    encoded.extend_from_slice(&start.to_le_bytes());
    encoded.extend_from_slice(&step.to_le_bytes());
    encoded.extend_from_slice(&count.to_le_bytes());
    Some(encoded)
}

fn decode_stride_bitmap(payload: &[u8], memory_size: usize) -> Result<RoaringBitmap> {
    if payload.len() != 12 {
        return Err(Error::invalid_input(format!(
            "Cell flag stride bitmap has size {}, expected 12",
            payload.len()
        )));
    }
    let start = u32::from_le_bytes(payload[0..4].try_into().expect("stride length checked"));
    let step = u32::from_le_bytes(payload[4..8].try_into().expect("stride length checked"));
    let count = u32::from_le_bytes(payload[8..12].try_into().expect("stride length checked"));
    if step == 0 || count == 0 {
        return Err(Error::invalid_input(
            "Cell flag stride bitmap requires a non-zero step and count",
        ));
    }
    let last = u64::from(start)
        .checked_add(u64::from(step).saturating_mul(u64::from(count - 1)))
        .filter(|last| *last <= u64::from(u32::MAX))
        .ok_or_else(|| Error::invalid_input("Cell flag stride bitmap exceeds u32 offsets"))?;
    let required_memory = roaring_memory_size(stride_bitmap_serialized_size(
        start,
        step,
        count,
        last as u32,
    )?);
    if required_memory > memory_size {
        return Err(Error::invalid_input(format!(
            "Cell flag stride bitmap declares memory size {}, decoded representation requires {}",
            memory_size, required_memory
        )));
    }
    let bitmap = if step == 1 {
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert_range(start..=last as u32);
        bitmap
    } else {
        RoaringBitmap::from_sorted_iter((0..count).map(|index| start + step.saturating_mul(index)))
            .map_err(|error| Error::invalid_input(format!("Invalid Cell flag stride: {error}")))?
    };
    let decoded_memory = roaring_memory_size(bitmap.serialized_size());
    if decoded_memory > memory_size {
        return Err(Error::invalid_input(format!(
            "Cell flag stride bitmap declares memory size {}, decoded representation requires {}",
            memory_size, decoded_memory
        )));
    }
    Ok(bitmap)
}

fn stride_bitmap_serialized_size(start: u32, step: u32, count: u32, last: u32) -> Result<usize> {
    // These are the fixed sizes in the portable Roaring format. Computing the
    // reconstructed representation before allocation prevents a small STRIDE
    // declaration from materializing a much larger bitmap.
    const ARRAY_LIMIT: u64 = 4096;
    const BITMAP_CONTAINER_BYTES: usize = 8192;
    const RUN_CONTAINER_BYTES: usize = 6;
    const NO_OFFSET_THRESHOLD: usize = 4;

    let start = u64::from(start);
    let step = u64::from(step);
    let last = u64::from(last);
    let mut container_count = 0usize;
    let mut container_bytes = 0usize;
    let mut has_run_container = false;

    for key in (start >> 16)..=(last >> 16) {
        let container_start = key << 16;
        let container_end = container_start | u64::from(u16::MAX);
        let lower = start.max(container_start);
        let upper = last.min(container_end);
        let first_index = (lower - start).div_ceil(step);
        let last_index = (upper - start) / step;
        if first_index > last_index || first_index >= u64::from(count) {
            continue;
        }
        let last_index = last_index.min(u64::from(count - 1));
        let cardinality = last_index - first_index + 1;
        let bytes = if step == 1 && cardinality > 2 {
            has_run_container = true;
            RUN_CONTAINER_BYTES
        } else if cardinality <= ARRAY_LIMIT {
            usize::try_from(cardinality)
                .ok()
                .and_then(|cardinality| cardinality.checked_mul(2))
                .ok_or_else(|| {
                    Error::invalid_input("Cell flag stride bitmap memory size overflow")
                })?
        } else {
            BITMAP_CONTAINER_BYTES
        };
        container_count = container_count.checked_add(1).ok_or_else(|| {
            Error::invalid_input("Cell flag stride bitmap container count overflow")
        })?;
        container_bytes = container_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid_input("Cell flag stride bitmap memory size overflow"))?;
    }

    let header_bytes = if has_run_container {
        let offsets = if container_count >= NO_OFFSET_THRESHOLD {
            container_count.checked_mul(4)
        } else {
            Some(0)
        };
        container_count
            .checked_mul(4)
            .and_then(|descriptions| descriptions.checked_add(container_count.div_ceil(8)))
            .and_then(|bytes| bytes.checked_add(4))
            .and_then(|bytes| bytes.checked_add(offsets?))
    } else {
        container_count
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_add(8))
    }
    .ok_or_else(|| Error::invalid_input("Cell flag stride bitmap header size overflow"))?;
    header_bytes
        .checked_add(container_bytes)
        .ok_or_else(|| Error::invalid_input("Cell flag stride bitmap memory size overflow"))
}

fn decode_zstd_cell_flag_bitmap_payload(payload: &[u8], memory_size: usize) -> Result<Vec<u8>> {
    if payload.len() < 8 {
        return Err(Error::invalid_input(
            "Compressed Cell flag bitmap is missing its decoded length",
        ));
    }
    let decoded_size = u64::from_le_bytes(
        payload[..8]
            .try_into()
            .expect("compressed header length checked"),
    );
    let decoded_size = usize::try_from(decoded_size).map_err(|_| {
        Error::invalid_input("Compressed Cell flag bitmap size exceeds this platform")
    })?;
    if roaring_memory_size(decoded_size) != memory_size {
        return Err(Error::invalid_input(format!(
            "Compressed Cell flag bitmap decoded size {} has an invalid memory declaration {}",
            decoded_size, memory_size
        )));
    }
    let decoded = zstd::bulk::decompress(&payload[8..], decoded_size).map_err(|error| {
        Error::invalid_input(format!("Invalid compressed Cell flag bitmap: {error}"))
    })?;
    if decoded.len() != decoded_size {
        return Err(Error::invalid_input(format!(
            "Compressed Cell flag bitmap decoded to {} bytes, expected {}",
            decoded.len(),
            decoded_size
        )));
    }
    Ok(decoded)
}

/// Stable schema-level identity for a field-scoped Boolean flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct CellFlagDefinition {
    /// Dataset-unique ID that is never reused.
    pub flag_id: u32,
    /// Stable Lance schema field ID this flag is scoped to.
    pub field_id: i32,
    /// User-visible name, unique among flags registered for this field.
    pub name: String,
}

impl From<&CellFlagDefinition> for pb::CellFlagDefinition {
    fn from(value: &CellFlagDefinition) -> Self {
        Self {
            flag_id: value.flag_id,
            field_id: value.field_id,
            name: value.name.clone(),
        }
    }
}

impl TryFrom<pb::CellFlagDefinition> for CellFlagDefinition {
    type Error = Error;

    fn try_from(value: pb::CellFlagDefinition) -> Result<Self> {
        if value.name.is_empty() {
            return Err(Error::invalid_input(format!(
                "Cell flag {} for field ID {} has an empty name",
                value.flag_id, value.field_id
            )));
        }
        Ok(Self {
            flag_id: value.flag_id,
            field_id: value.field_id,
            name: value.name,
        })
    }
}

/// Reference to an immutable cell-flag object under a dataset root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct CellFlagFile {
    /// Path relative to the referenced dataset root.
    pub path: String,
    /// Exact encoded size in bytes.
    pub size_bytes: u64,
    /// Upper bound on bytes retained while the object is decoded.
    pub memory_size_bytes: u64,
    /// Optional external dataset base ID.
    pub base_id: Option<u32>,
    /// Optional exact inline copy of a small immutable root object.
    #[serde(default)]
    pub inline_bytes: Option<Vec<u8>>,
}

impl From<&CellFlagFile> for pb::CellFlagFile {
    fn from(value: &CellFlagFile) -> Self {
        Self {
            path: value.path.clone(),
            size_bytes: value.size_bytes,
            base_id: value.base_id,
            inline_bytes: value.inline_bytes.clone(),
            memory_size_bytes: value.memory_size_bytes,
        }
    }
}

impl TryFrom<pb::CellFlagFile> for CellFlagFile {
    type Error = Error;

    fn try_from(value: pb::CellFlagFile) -> Result<Self> {
        let file = Self {
            path: value.path,
            size_bytes: value.size_bytes,
            memory_size_bytes: value.memory_size_bytes,
            base_id: value.base_id,
            inline_bytes: value.inline_bytes,
        };
        file.validate_inline_copy()?;
        if file.memory_size_bytes == 0 {
            return Err(Error::invalid_input(format!(
                "Cell flag file '{}' must declare a non-zero memory size",
                file.path
            )));
        }
        if !file.path.is_empty() {
            file.validate_namespace()?;
        }
        Ok(file)
    }
}

impl CellFlagFile {
    fn validate_inline_copy(&self) -> Result<()> {
        if self.path.is_empty() {
            if self.size_bytes != 0 {
                return Err(Error::invalid_input(
                    "Inline-only cell flag root cannot declare an object size",
                ));
            }
            if self.inline_bytes.as_ref().is_none_or(Vec::is_empty) {
                return Err(Error::invalid_input(
                    "Inline-only cell flag root must contain encoded bytes",
                ));
            }
        } else if let Some(bytes) = self.inline_bytes.as_ref()
            && bytes.len() as u64 != self.size_bytes
        {
            return Err(Error::invalid_input(format!(
                "Inline cell flag file '{}' has size {}, expected {}",
                self.path,
                bytes.len(),
                self.size_bytes
            )));
        }
        if self
            .inline_bytes
            .as_ref()
            .is_some_and(|bytes| bytes.len() > MAX_INLINE_CELL_FLAG_ROOT_BYTES)
        {
            return Err(Error::invalid_input(format!(
                "Inline cell flag file '{}' has size {}, maximum is {}",
                self.path, self.size_bytes, MAX_INLINE_CELL_FLAG_ROOT_BYTES
            )));
        }
        Ok(())
    }

    fn validate_namespace(&self) -> Result<()> {
        let path = Path::parse(&self.path).map_err(|error| {
            Error::invalid_input(format!(
                "Invalid cell flag file path '{}': {}",
                self.path, error
            ))
        })?;
        let mut parts = path.parts();
        if !parts
            .next()
            .is_some_and(|part| part.as_ref() == "_cell_flags")
        {
            return Err(Error::invalid_input(format!(
                "Cell flag file '{}' must be under '_cell_flags'",
                self.path
            )));
        }
        Ok(())
    }

    fn validate_kind(&self, kind: &str, expected_parts: usize, suffix: &str) -> Result<()> {
        self.validate_namespace()?;
        let path = Path::parse(&self.path)?;
        let parts = path
            .parts()
            .map(|part| part.as_ref().to_string())
            .collect::<Vec<_>>();
        if parts.len() != expected_parts || parts.get(1).map(String::as_str) != Some(kind) {
            return Err(Error::invalid_input(format!(
                "Cell flag {} file '{}' has an invalid path layout",
                kind, self.path
            )));
        }
        parts[2].parse::<u32>().map_err(|_| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' has an invalid flag ID segment",
                kind, self.path
            ))
        })?;
        if kind == "bitmaps" {
            parts[3].parse::<u64>().map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag bitmap file '{}' has an invalid fragment ID segment",
                    self.path
                ))
            })?;
        }
        let file_name = parts.last().ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' has an empty path",
                kind, self.path
            ))
        })?;
        let uuid = file_name.strip_suffix(suffix).ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' must end in '{}'",
                kind, self.path, suffix
            ))
        })?;
        Uuid::parse_str(uuid).map_err(|_| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' has an invalid immutable object ID",
                kind, self.path
            ))
        })?;
        Ok(())
    }

    /// Validate that this file is an immutable flag root.
    pub fn validate_root_path(&self) -> Result<()> {
        self.validate_inline_copy()?;
        if self.path.is_empty() {
            return Ok(());
        }
        self.validate_kind("roots", 4, ".root")
    }

    /// Validate this root's namespace against its manifest flag ID.
    pub fn validate_root_path_for_flag(&self, flag_id: u32) -> Result<()> {
        self.validate_root_path()?;
        if self.path.is_empty() {
            return Ok(());
        }
        let path_flag_id = Path::parse(&self.path)?
            .parts()
            .nth(2)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag root '{}' is missing its flag ID segment",
                    self.path
                ))
            })?
            .as_ref()
            .parse::<u32>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag root '{}' has an invalid flag ID segment",
                    self.path
                ))
            })?;
        if path_flag_id != flag_id {
            return Err(Error::invalid_input(format!(
                "Cell flag root '{}' is under flag ID {}, expected {}",
                self.path, path_flag_id, flag_id
            )));
        }
        Ok(())
    }

    /// Validate that this file is an immutable partial flag bitmap.
    pub fn validate_bitmap_path(&self) -> Result<()> {
        self.validate_inline_copy()?;
        self.validate_kind("bitmaps", 5, ".rbm")?;
        if self.inline_bytes.is_some() {
            return Err(Error::invalid_input(format!(
                "Cell flag bitmap file '{}' cannot contain inline root bytes",
                self.path
            )));
        }
        Ok(())
    }

    /// Validate this bitmap's namespace against its root entry.
    pub fn validate_bitmap_path_for_fragment(&self, flag_id: u32, fragment_id: u64) -> Result<()> {
        self.validate_bitmap_path()?;
        let path = Path::parse(&self.path)?;
        let parts = path.parts().collect::<Vec<_>>();
        let path_flag_id = parts
            .get(2)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' is missing its flag ID segment",
                    self.path
                ))
            })?
            .as_ref()
            .parse::<u32>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' has an invalid flag ID segment",
                    self.path
                ))
            })?;
        let path_fragment_id = parts
            .get(3)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' is missing its fragment ID segment",
                    self.path
                ))
            })?
            .as_ref()
            .parse::<u64>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' has an invalid fragment ID segment",
                    self.path
                ))
            })?;
        if path_flag_id != flag_id || path_fragment_id != fragment_id {
            return Err(Error::invalid_input(format!(
                "Cell flag bitmap '{}' is under flag ID {} and fragment {}, expected flag ID {} and fragment {}",
                self.path, path_flag_id, path_fragment_id, flag_id, fragment_id
            )));
        }
        Ok(())
    }
}

/// Manifest descriptor for one registered flag with at least one true row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct CellFlagState {
    /// Stable dataset flag ID.
    pub flag_id: u32,
    /// Immutable root for this snapshot.
    pub root: CellFlagFile,
}

impl From<&CellFlagState> for pb::CellFlagState {
    fn from(value: &CellFlagState) -> Self {
        Self {
            flag_id: value.flag_id,
            root: Some((&value.root).into()),
        }
    }
}

impl TryFrom<pb::CellFlagState> for CellFlagState {
    type Error = Error;

    fn try_from(value: pb::CellFlagState) -> Result<Self> {
        let root = value.root.ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag state for flag ID {} is missing its root",
                value.flag_id
            ))
        })?;
        let root: CellFlagFile = root.try_into()?;
        root.validate_root_path_for_flag(value.flag_id)?;
        Ok(Self {
            flag_id: value.flag_id,
            root,
        })
    }
}

/// Materialized immutable root for one registered flag.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct CellFlagRoot {
    /// Non-empty fragment states, sorted by fragment ID.
    pub fragments: Vec<CellFlagFragment>,
}

/// Materialized flag state for one physical fragment.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct CellFlagFragment {
    /// Fragment ID in the snapshot.
    pub fragment_id: u64,
    /// Number of physical rows when this state was written.
    pub physical_rows: u64,
    /// Compact flag state.
    pub state: CellFlagFragmentState,
}

/// Compact flag representation for a fragment.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum CellFlagFragmentState {
    /// Every physical row is true.
    All,
    /// A non-empty, non-full Roaring bitmap of physical row offsets.
    Partial(CellFlagFile),
    /// A small non-empty, non-full portable Roaring bitmap embedded in the root.
    InlinePartial(Vec<u8>),
}

impl From<&CellFlagRoot> for pb::CellFlagRoot {
    fn from(value: &CellFlagRoot) -> Self {
        Self {
            fragments: value.fragments.iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<pb::CellFlagRoot> for CellFlagRoot {
    type Error = Error;

    fn try_from(value: pb::CellFlagRoot) -> Result<Self> {
        if value.fragments.is_empty() {
            return Err(Error::invalid_input(
                "Cell flag root must contain at least one non-empty fragment state",
            ));
        }
        let mut fragments = Vec::with_capacity(value.fragments.len());
        let mut previous = None;
        for fragment in value.fragments {
            let fragment: CellFlagFragment = fragment.try_into()?;
            if previous.is_some_and(|id| id >= fragment.fragment_id) {
                return Err(Error::invalid_input(
                    "Cell flag root fragment IDs must be strictly increasing",
                ));
            }
            previous = Some(fragment.fragment_id);
            fragments.push(fragment);
        }
        Ok(Self { fragments })
    }
}

impl From<&CellFlagFragment> for pb::CellFlagFragment {
    fn from(value: &CellFlagFragment) -> Self {
        let state = match &value.state {
            CellFlagFragmentState::All => pb::cell_flag_fragment::State::AllSet(true),
            CellFlagFragmentState::Partial(file) => {
                pb::cell_flag_fragment::State::Partial(file.into())
            }
            CellFlagFragmentState::InlinePartial(bytes) => {
                pb::cell_flag_fragment::State::InlinePartial(bytes.clone())
            }
        };
        Self {
            fragment_id: value.fragment_id,
            physical_rows: value.physical_rows,
            state: Some(state),
        }
    }
}

impl TryFrom<pb::CellFlagFragment> for CellFlagFragment {
    type Error = Error;

    fn try_from(value: pb::CellFlagFragment) -> Result<Self> {
        if value.physical_rows == 0 {
            return Err(Error::invalid_input(format!(
                "Cell flag fragment {} must have at least one physical row",
                value.fragment_id
            )));
        }
        let state = match value.state.ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag fragment {} is missing its state",
                value.fragment_id
            ))
        })? {
            pb::cell_flag_fragment::State::AllSet(true) => CellFlagFragmentState::All,
            pb::cell_flag_fragment::State::AllSet(false) => {
                return Err(Error::invalid_input(format!(
                    "Cell flag fragment {} encodes all_set=false",
                    value.fragment_id
                )));
            }
            pb::cell_flag_fragment::State::Partial(file) => {
                let file: CellFlagFile = file.try_into()?;
                file.validate_bitmap_path()?;
                if file.size_bytes == 0 {
                    return Err(Error::invalid_input(format!(
                        "Partial cell flag file '{}' must have a non-zero size",
                        file.path
                    )));
                }
                CellFlagFragmentState::Partial(file)
            }
            pb::cell_flag_fragment::State::InlinePartial(bytes) => {
                if bytes.is_empty() {
                    return Err(Error::invalid_input(format!(
                        "Inline partial cell flag for fragment {} must be non-empty",
                        value.fragment_id
                    )));
                }
                CellFlagFragmentState::InlinePartial(bytes)
            }
        };
        Ok(Self {
            fragment_id: value.fragment_id,
            physical_rows: value.physical_rows,
            state,
        })
    }
}

#[cfg(test)]
mod tests {
    use roaring::RoaringBitmap;

    use super::*;

    #[test]
    fn adaptive_bitmap_chooses_the_smaller_encoding_and_round_trips() {
        let sparse = RoaringBitmap::from_iter([1, 1_000_000]);
        let sparse_bytes = encode_cell_flag_bitmap(&sparse).unwrap();
        assert_eq!(decode_cell_flag_bitmap(&sparse_bytes).unwrap(), sparse);

        let periodic = RoaringBitmap::from_iter((0..10_000).step_by(2));
        let periodic_bytes = encode_cell_flag_bitmap(&periodic).unwrap();
        assert_eq!(periodic_bytes[4], CELL_FLAG_BITMAP_STRIDE);
        assert!(periodic_bytes.len() < periodic.serialized_size());
        assert_eq!(decode_cell_flag_bitmap(&periodic_bytes).unwrap(), periodic);
        assert!(cell_flag_bitmap_memory_size(&periodic_bytes).unwrap() >= periodic_bytes.len());
    }

    #[test]
    fn adaptive_bitmap_rejects_understated_memory() {
        let bitmap = RoaringBitmap::from_iter((0..10_000).step_by(2));
        let mut bytes = encode_cell_flag_bitmap(&bitmap).unwrap();
        bytes[5..CELL_FLAG_BITMAP_HEADER_BYTES].copy_from_slice(&1_u64.to_le_bytes());
        assert!(decode_cell_flag_bitmap(&bytes).is_err());
    }

    #[test]
    fn bitmap_memory_limit_preserves_row_change_headroom() {
        let mut bytes = Vec::with_capacity(CELL_FLAG_BITMAP_HEADER_BYTES);
        bytes.extend_from_slice(CELL_FLAG_BITMAP_MAGIC);
        bytes.push(CELL_FLAG_BITMAP_ROARING);
        bytes.extend_from_slice(&(MAX_CELL_FLAG_BITMAP_MEMORY_BYTES as u64).to_le_bytes());
        assert_eq!(
            cell_flag_bitmap_memory_size(&bytes).unwrap(),
            MAX_CELL_FLAG_BITMAP_MEMORY_BYTES
        );

        bytes[5..CELL_FLAG_BITMAP_HEADER_BYTES]
            .copy_from_slice(&((MAX_CELL_FLAG_BITMAP_MEMORY_BYTES + 1) as u64).to_le_bytes());
        assert!(cell_flag_bitmap_memory_size(&bytes).is_err());

        let one_row = RoaringBitmap::from_iter([0]);
        let empty = RoaringBitmap::new();
        assert!(
            MAX_CELL_FLAG_BITMAP_MEMORY_BYTES
                + roaring_memory_size(one_row.serialized_size())
                + roaring_memory_size(empty.serialized_size())
                + 1024 * 1024
                <= 32 * 1024 * 1024
        );
    }

    #[test]
    fn bitmap_writers_reject_undecodable_memory_declarations() {
        let bitmap = RoaringBitmap::from_sorted_iter(
            (0_u32..16_000_000).map(|index| index.saturating_mul(17)),
        )
        .unwrap();
        assert!(bitmap.serialized_size() > MAX_CELL_FLAG_BITMAP_MEMORY_BYTES);
        assert!(encode_cell_flag_bitmap(&bitmap).is_err());
        assert!(encode_cell_flag_query_bitmap(&bitmap).is_err());
    }

    #[test]
    fn stride_bitmap_accepts_a_noncanonical_retained_memory_upper_bound() {
        let bitmap = RoaringBitmap::from_iter(0..100);
        let bytes = encode_cell_flag_bitmap(&bitmap).unwrap();
        assert_eq!(bytes[4], CELL_FLAG_BITMAP_STRIDE);

        let declared_memory = cell_flag_bitmap_memory_size(&bytes).unwrap();
        let decoded = decode_cell_flag_bitmap(&bytes).unwrap();
        assert_eq!(decoded, bitmap);
        assert!(decoded.serialized_size() < declared_memory);
    }

    #[test]
    fn stride_bitmap_size_is_validated_before_materialization() {
        for (start, step, count) in [
            (0, 1, 1),
            (0, 1, 100),
            (65_535, 1, 3),
            (65_534, 1, 5),
            (0, 2, 100_000),
            (1, 17, 100_000),
            (0, 65_537, 1_000),
        ] {
            let last = start + step * (count - 1);
            let bitmap = if step == 1 {
                let mut bitmap = RoaringBitmap::new();
                bitmap.insert_range(start..=last);
                bitmap
            } else {
                RoaringBitmap::from_sorted_iter((0..count).map(|index| start + step * index))
                    .unwrap()
            };
            assert_eq!(
                stride_bitmap_serialized_size(start, step, count, last).unwrap(),
                bitmap.serialized_size(),
                "start={start}, step={step}, count={count}"
            );
        }

        let declared_memory = 1024 * 1024;
        let mut encoded = Vec::with_capacity(CELL_FLAG_BITMAP_HEADER_BYTES + 12);
        encoded.extend_from_slice(CELL_FLAG_BITMAP_MAGIC);
        encoded.push(CELL_FLAG_BITMAP_STRIDE);
        encoded.extend_from_slice(&(declared_memory as u64).to_le_bytes());
        encoded.extend_from_slice(&0_u32.to_le_bytes());
        encoded.extend_from_slice(&17_u32.to_le_bytes());
        encoded.extend_from_slice(&((declared_memory * 8) as u32).to_le_bytes());

        let error = decode_cell_flag_bitmap(&encoded).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("decoded representation requires")
        );

        let mut encoded = Vec::with_capacity(CELL_FLAG_BITMAP_HEADER_BYTES + 12);
        encoded.extend_from_slice(CELL_FLAG_BITMAP_MAGIC);
        encoded.push(CELL_FLAG_BITMAP_STRIDE);
        encoded.extend_from_slice(&(16_793_618_u64).to_le_bytes());
        encoded.extend_from_slice(&0_u32.to_le_bytes());
        encoded.extend_from_slice(&2_u32.to_le_bytes());
        encoded.extend_from_slice(&67_108_865_u32.to_le_bytes());
        let error = decode_cell_flag_bitmap(&encoded).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("decoded representation requires")
        );
    }

    #[test]
    fn query_bitmap_keeps_fast_roaring_decode_and_root_compresses_repetition() {
        let periodic = RoaringBitmap::from_iter((0..100_000).step_by(10));
        let query_bytes = encode_cell_flag_query_bitmap(&periodic).unwrap();
        assert!(matches!(
            query_bytes[4],
            CELL_FLAG_BITMAP_ROARING | CELL_FLAG_BITMAP_ZSTD_ROARING
        ));
        assert_eq!(decode_cell_flag_bitmap(&query_bytes).unwrap(), periodic);

        let fragment = pb::CellFlagFragment {
            fragment_id: 0,
            physical_rows: 100_000,
            state: Some(pb::cell_flag_fragment::State::InlinePartial(query_bytes)),
        };
        let root = pb::CellFlagRoot {
            fragments: (0..100)
                .map(|fragment_id| pb::CellFlagFragment {
                    fragment_id,
                    ..fragment.clone()
                })
                .collect(),
        };
        let raw_size = prost::Message::encoded_len(&root);
        let (encoded, memory_size) = encode_cell_flag_root(&root).unwrap();
        assert!(memory_size >= raw_size);
        assert!(encoded.len() * 4 < raw_size);
        assert_eq!(
            decode_cell_flag_root(&encoded).unwrap(),
            (root, memory_size)
        );
    }

    #[test]
    fn root_memory_size_bounds_large_fragment_conversion() {
        let root = pb::CellFlagRoot {
            fragments: (0..100_000)
                .map(|fragment_id| pb::CellFlagFragment {
                    fragment_id,
                    physical_rows: 1,
                    state: Some(pb::cell_flag_fragment::State::AllSet(true)),
                })
                .collect(),
        };

        let (encoded, memory_size) = encode_cell_flag_root(&root).unwrap();
        let (decoded, decoded_memory_size) = decode_cell_flag_root(&encoded).unwrap();
        let retained_size = CellFlagRoot::try_from(decoded).unwrap().deep_size_of();

        assert_eq!(decoded_memory_size, memory_size);
        assert!(retained_size <= memory_size);
    }

    #[test]
    fn root_wire_scan_rejects_fragment_count_before_prost_materialization() {
        let fragment_count =
            MAX_CELL_FLAG_ROOT_MEMORY_BYTES.div_ceil(CELL_FLAG_ROOT_FRAGMENT_MEMORY_BYTES) + 1;
        let mut decoded = Vec::with_capacity(fragment_count * 2);
        for _ in 0..fragment_count {
            // CellFlagRoot.fragments, containing an empty CellFlagFragment.
            decoded.extend_from_slice(&[0x0a, 0x00]);
        }
        let compressed = zstd::bulk::compress(&decoded, 1).unwrap();
        let mut encoded = Vec::with_capacity(CELL_FLAG_ROOT_HEADER_BYTES + compressed.len());
        encoded.extend_from_slice(CELL_FLAG_ROOT_MAGIC);
        encoded.push(CELL_FLAG_ROOT_ZSTD);
        encoded.extend_from_slice(&(decoded.len() as u64).to_le_bytes());
        encoded.extend_from_slice(&compressed);

        let error = decode_cell_flag_root(&encoded).unwrap_err();
        assert!(error.to_string().contains("memory size"));
    }

    #[test]
    fn root_wire_scan_rejects_highly_compressible_inline_memory() {
        use prost::Message;

        let inline_bytes = vec![0_u8; MAX_CELL_FLAG_ROOT_MEMORY_BYTES / 3 + 1];
        let root = pb::CellFlagRoot {
            fragments: vec![pb::CellFlagFragment {
                fragment_id: 1,
                physical_rows: inline_bytes.len() as u64 * 8,
                state: Some(pb::cell_flag_fragment::State::InlinePartial(inline_bytes)),
            }],
        };
        let decoded = root.encode_to_vec();
        let compressed = zstd::bulk::compress(&decoded, 1).unwrap();
        assert!(compressed.len() * 100 < decoded.len());
        let mut encoded = Vec::with_capacity(CELL_FLAG_ROOT_HEADER_BYTES + compressed.len());
        encoded.extend_from_slice(CELL_FLAG_ROOT_MAGIC);
        encoded.push(CELL_FLAG_ROOT_ZSTD);
        encoded.extend_from_slice(&(decoded.len() as u64).to_le_bytes());
        encoded.extend_from_slice(&compressed);

        let error = decode_cell_flag_root(&encoded).unwrap_err();
        assert!(error.to_string().contains("memory size"));
    }

    #[test]
    fn root_wire_scan_rejects_malformed_fragment_length() {
        let decoded = [0x0a, 0x80];
        let mut encoded = Vec::with_capacity(CELL_FLAG_ROOT_HEADER_BYTES + decoded.len());
        encoded.extend_from_slice(CELL_FLAG_ROOT_MAGIC);
        encoded.push(CELL_FLAG_ROOT_RAW);
        encoded.extend_from_slice(&(decoded.len() as u64).to_le_bytes());
        encoded.extend_from_slice(&decoded);

        assert!(decode_cell_flag_root(&encoded).is_err());
    }

    #[test]
    fn root_round_trip_and_validation() {
        assert!(
            CellFlagRoot::try_from(pb::CellFlagRoot {
                fragments: Vec::new()
            })
            .is_err()
        );

        let root = CellFlagRoot {
            fragments: vec![
                CellFlagFragment {
                    fragment_id: 1,
                    physical_rows: 5,
                    state: CellFlagFragmentState::All,
                },
                CellFlagFragment {
                    fragment_id: 3,
                    physical_rows: 8,
                    state: CellFlagFragmentState::Partial(CellFlagFile {
                        path: "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm"
                            .to_string(),
                        size_bytes: 12,
                        memory_size_bytes: 12,
                        base_id: Some(4),
                        inline_bytes: None,
                    }),
                },
                CellFlagFragment {
                    fragment_id: 5,
                    physical_rows: 10,
                    state: CellFlagFragmentState::InlinePartial(vec![1, 2, 3]),
                },
            ],
        };
        let proto = pb::CellFlagRoot::from(&root);
        assert_eq!(CellFlagRoot::try_from(proto).unwrap(), root);

        let duplicate = pb::CellFlagRoot {
            fragments: vec![
                pb::CellFlagFragment::from(&root.fragments[0]),
                pb::CellFlagFragment::from(&root.fragments[0]),
            ],
        };
        assert!(CellFlagRoot::try_from(duplicate).is_err());

        let empty_inline = pb::CellFlagRoot {
            fragments: vec![pb::CellFlagFragment {
                fragment_id: 1,
                physical_rows: 5,
                state: Some(pb::cell_flag_fragment::State::InlinePartial(Vec::new())),
            }],
        };
        assert!(CellFlagRoot::try_from(empty_inline).is_err());
    }

    #[test]
    fn cell_flag_files_reject_wrong_roles_and_layouts() {
        let root = CellFlagFile {
            path: "_cell_flags/roots/7/00000000-0000-0000-0000-000000000001.root".to_string(),
            size_bytes: 12,
            memory_size_bytes: 12,
            base_id: None,
            inline_bytes: Some(vec![0; 12]),
        };
        assert!(root.validate_root_path().is_ok());
        assert!(root.validate_bitmap_path().is_err());

        let inline_root = CellFlagFile {
            path: String::new(),
            size_bytes: 0,
            memory_size_bytes: 12,
            base_id: Some(3),
            inline_bytes: Some(vec![0; 12]),
        };
        assert!(inline_root.validate_root_path_for_flag(7).is_ok());
        assert!(inline_root.validate_bitmap_path().is_err());
        assert!(
            CellFlagFile {
                size_bytes: 12,
                ..inline_root.clone()
            }
            .validate_root_path()
            .is_err()
        );
        assert!(
            CellFlagFile {
                inline_bytes: None,
                ..inline_root
            }
            .validate_root_path()
            .is_err()
        );

        let bitmap = CellFlagFile {
            path: "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm".to_string(),
            size_bytes: 12,
            memory_size_bytes: 12,
            base_id: None,
            inline_bytes: None,
        };
        assert!(bitmap.validate_bitmap_path().is_ok());
        assert!(bitmap.validate_root_path().is_err());

        for invalid_path in [
            "outside/roots/7/00000000-0000-0000-0000-000000000001.root",
            "_cell_flags/roots/not-a-field/00000000-0000-0000-0000-000000000001.root",
            "_cell_flags/roots/7/not-a-uuid.root",
            "_cell_flags/bitmaps/7/not-a-fragment/00000000-0000-0000-0000-000000000001.rbm",
            "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.root",
            "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm/extra",
        ] {
            let file = CellFlagFile {
                path: invalid_path.to_string(),
                size_bytes: 12,
                memory_size_bytes: 12,
                base_id: None,
                inline_bytes: None,
            };
            assert!(
                file.validate_root_path().is_err() && file.validate_bitmap_path().is_err(),
                "invalid path was accepted: {invalid_path}"
            );
        }
    }
}
