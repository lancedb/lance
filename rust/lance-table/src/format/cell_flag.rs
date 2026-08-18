// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Snapshot-level field-scoped Boolean flag metadata.

use std::io::Cursor;

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
const MAX_CELL_FLAG_BITMAP_MEMORY_BYTES: usize = 64 * 1024 * 1024;
const CELL_FLAG_ROOT_MAGIC: &[u8; 4] = b"LCG1";
const CELL_FLAG_ROOT_HEADER_BYTES: usize = 13;
const CELL_FLAG_ROOT_RAW: u8 = 0;
const CELL_FLAG_ROOT_ZSTD: u8 = 1;
const MAX_CELL_FLAG_ROOT_MEMORY_BYTES: usize = 32 * 1024 * 1024;

fn bitset_memory_size(bytes: usize) -> usize {
    // An array-backed Roaring container can use two bytes per set bit. A
    // bitset byte can represent eight set bits, so 16x plus container overhead
    // is a conservative retained-memory bound before decoding.
    bytes
        .saturating_mul(16)
        .saturating_add(bytes.div_ceil(8192).saturating_mul(32))
}

/// Encode a non-empty bitmap using the smaller of portable Roaring and a dense bitset.
pub fn encode_cell_flag_bitmap(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut roaring = Vec::with_capacity(bitmap.serialized_size());
    bitmap
        .serialize_into(&mut roaring)
        .expect("RoaringBitmap serialization to Vec cannot fail");
    let roaring_memory_size = roaring.len();
    let compressed_roaring = zstd::bulk::compress(&roaring, 1)
        .expect("Zstd compression of an in-memory Cell Flag bitmap cannot fail");
    let mut candidates = vec![(CELL_FLAG_BITMAP_ROARING, roaring_memory_size, roaring)];
    if compressed_roaring.len() + 8 < candidates[0].2.len() {
        let mut payload = Vec::with_capacity(compressed_roaring.len() + 8);
        payload.extend_from_slice(&(roaring_memory_size as u64).to_le_bytes());
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
        let compressed_bitset = zstd::bulk::compress(&bitset, 1)
            .expect("Zstd compression of an in-memory Cell Flag bitset cannot fail");
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
    let mut encoded = Vec::with_capacity(CELL_FLAG_BITMAP_HEADER_BYTES + payload.len());
    encoded.extend_from_slice(CELL_FLAG_BITMAP_MAGIC);
    encoded.push(encoding);
    encoded.extend_from_slice(&(memory_size as u64).to_le_bytes());
    encoded.extend_from_slice(&payload);
    encoded
}

/// Encode a bitmap for latency-sensitive query binding.
///
/// This retains portable Roaring's fast decode path and only uses Zstd when it
/// reduces that representation directly. Denser encodings remain available to
/// transaction sidecars and external bitmap objects through
/// [`encode_cell_flag_bitmap`].
pub fn encode_cell_flag_query_bitmap(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut roaring = Vec::with_capacity(bitmap.serialized_size());
    bitmap
        .serialize_into(&mut roaring)
        .expect("RoaringBitmap serialization to Vec cannot fail");
    let memory_size = roaring.len();
    let compressed = zstd::bulk::compress(&roaring, 1)
        .expect("Zstd compression of an in-memory Cell Flag bitmap cannot fail");
    let (encoding, payload) = if compressed.len() + 8 < roaring.len() {
        let mut payload = Vec::with_capacity(compressed.len() + 8);
        payload.extend_from_slice(&(memory_size as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);
        (CELL_FLAG_BITMAP_ZSTD_ROARING, payload)
    } else {
        (CELL_FLAG_BITMAP_ROARING, roaring)
    };
    let mut encoded = Vec::with_capacity(CELL_FLAG_BITMAP_HEADER_BYTES + payload.len());
    encoded.extend_from_slice(CELL_FLAG_BITMAP_MAGIC);
    encoded.push(encoding);
    encoded.extend_from_slice(&(memory_size as u64).to_le_bytes());
    encoded.extend_from_slice(&payload);
    encoded
}

/// Encode a complete root, compressing repeated fragment metadata as one unit.
pub fn encode_cell_flag_root(root: &pb::CellFlagRoot) -> Result<(Vec<u8>, usize)> {
    use prost::Message;

    let memory_size = root.encoded_len();
    if memory_size > MAX_CELL_FLAG_ROOT_MEMORY_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag root memory size {} exceeds the {} byte limit",
            memory_size, MAX_CELL_FLAG_ROOT_MEMORY_BYTES
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
    encoded.extend_from_slice(&(memory_size as u64).to_le_bytes());
    encoded.extend_from_slice(&payload);
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
    let memory_size = u64::from_le_bytes(
        bytes[5..CELL_FLAG_ROOT_HEADER_BYTES]
            .try_into()
            .expect("root header length checked"),
    );
    let memory_size = usize::try_from(memory_size)
        .map_err(|_| Error::invalid_input("Cell flag root memory size exceeds this platform"))?;
    if memory_size > MAX_CELL_FLAG_ROOT_MEMORY_BYTES {
        return Err(Error::invalid_input(format!(
            "Cell flag root memory size {} exceeds the {} byte limit",
            memory_size, MAX_CELL_FLAG_ROOT_MEMORY_BYTES
        )));
    }
    let payload = &bytes[CELL_FLAG_ROOT_HEADER_BYTES..];
    let decoded = match bytes[4] {
        CELL_FLAG_ROOT_RAW => {
            if payload.len() != memory_size {
                return Err(Error::invalid_input(format!(
                    "Cell flag root has size {}, expected {}",
                    payload.len(),
                    memory_size
                )));
            }
            payload.to_vec()
        }
        CELL_FLAG_ROOT_ZSTD => zstd::bulk::decompress(payload, memory_size).map_err(|error| {
            Error::invalid_input(format!("Invalid compressed Cell flag root: {error}"))
        })?,
        encoding => {
            return Err(Error::invalid_input(format!(
                "Cell flag root has unknown encoding {}",
                encoding
            )));
        }
    };
    let root = pb::CellFlagRoot::decode(decoded.as_slice())
        .map_err(|error| Error::invalid_input(format!("Invalid Cell flag root: {error}")))?;
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
            if memory_size != payload.len() {
                return Err(Error::invalid_input(format!(
                    "Cell flag Roaring bitmap declares memory size {}, expected {}",
                    memory_size,
                    payload.len()
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
    if step > 1 && count as usize > memory_size.saturating_mul(8) {
        return Err(Error::invalid_input(
            "Cell flag stride bitmap cardinality exceeds its memory bound",
        ));
    }
    let bitmap = if step == 1 {
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert_range(start..=last as u32);
        bitmap
    } else {
        RoaringBitmap::from_sorted_iter((0..count).map(|index| start + step.saturating_mul(index)))
            .map_err(|error| Error::invalid_input(format!("Invalid Cell flag stride: {error}")))?
    };
    if bitmap.serialized_size() != memory_size {
        return Err(Error::invalid_input(format!(
            "Cell flag stride bitmap declares memory size {}, expected {}",
            memory_size,
            bitmap.serialized_size()
        )));
    }
    Ok(bitmap)
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
    if decoded_size != memory_size {
        return Err(Error::invalid_input(format!(
            "Compressed Cell flag bitmap declares decoded size {}, expected {}",
            decoded_size, memory_size
        )));
    }
    zstd::bulk::decompress(&payload[8..], decoded_size).map_err(|error| {
        Error::invalid_input(format!("Invalid compressed Cell flag bitmap: {error}"))
    })
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
        file.validate_namespace()?;
        Ok(file)
    }
}

impl CellFlagFile {
    fn validate_inline_copy(&self) -> Result<()> {
        if let Some(bytes) = self.inline_bytes.as_ref()
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
        self.validate_kind("roots", 4, ".root")
    }

    /// Validate this root's namespace against its manifest flag ID.
    pub fn validate_root_path_for_flag(&self, flag_id: u32) -> Result<()> {
        self.validate_root_path()?;
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
        let sparse_bytes = encode_cell_flag_bitmap(&sparse);
        assert_eq!(decode_cell_flag_bitmap(&sparse_bytes).unwrap(), sparse);

        let periodic = RoaringBitmap::from_iter((0..10_000).step_by(2));
        let periodic_bytes = encode_cell_flag_bitmap(&periodic);
        assert_eq!(periodic_bytes[4], CELL_FLAG_BITMAP_STRIDE);
        assert!(periodic_bytes.len() < periodic.serialized_size());
        assert_eq!(decode_cell_flag_bitmap(&periodic_bytes).unwrap(), periodic);
        assert!(cell_flag_bitmap_memory_size(&periodic_bytes).unwrap() >= periodic_bytes.len());
    }

    #[test]
    fn adaptive_bitmap_rejects_understated_memory() {
        let bitmap = RoaringBitmap::from_iter((0..10_000).step_by(2));
        let mut bytes = encode_cell_flag_bitmap(&bitmap);
        bytes[5..CELL_FLAG_BITMAP_HEADER_BYTES].copy_from_slice(&1_u64.to_le_bytes());
        assert!(decode_cell_flag_bitmap(&bytes).is_err());
    }

    #[test]
    fn query_bitmap_keeps_fast_roaring_decode_and_root_compresses_repetition() {
        let periodic = RoaringBitmap::from_iter((0..100_000).step_by(10));
        let query_bytes = encode_cell_flag_query_bitmap(&periodic);
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
        assert_eq!(memory_size, raw_size);
        assert!(encoded.len() * 4 < raw_size);
        assert_eq!(decode_cell_flag_root(&encoded).unwrap(), (root, raw_size));
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
