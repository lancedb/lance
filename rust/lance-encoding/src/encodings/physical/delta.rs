// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Binary delta encoding inspired by Git's packfile delta format.
//!
//! Encodes a target buffer as a series of copy/insert instructions relative
//! to a source (base) buffer. Copy instructions reference byte ranges in the
//! source; insert instructions carry literal bytes from the target.
//!
//! For best compression, data should be sorted so that similar values
//! (e.g., successive versions of the same file) are adjacent.

/// Rabin rolling hash window size (bytes).
const RABIN_WINDOW: usize = 16;

/// Minimum match length to emit a copy instead of an insert.
const MIN_COPY_LEN: usize = 4;

/// Maximum insert instruction payload (opcode encodes length in 7 bits).
const MAX_INSERT_LEN: usize = 127;

/// Maximum copy size per instruction (3 size bytes, but 0 means 0x10000).
const MAX_COPY_SIZE: usize = 0xFFFFFF;

/// Block size for indexing the source buffer (non-overlapping windows).
const INDEX_BLOCK_SIZE: usize = 16;

/// Delta must be at most this fraction of target size to be worth keeping.
const DELTA_RATIO_THRESHOLD: f64 = 0.8;

/// Maximum varint bytes for a u64 (ceil(64/7) = 10).
const MAX_VARINT_BYTES: usize = 10;

// ---------------------------------------------------------------------------
// Rabin rolling hash
// ---------------------------------------------------------------------------

const RABIN_MULTIPLIER: u32 = 257;

/// Precomputed RABIN_MULTIPLIER^RABIN_WINDOW mod 2^32, used to remove the
/// oldest byte's contribution from the rolling hash in O(1).
#[cfg(test)]
const RABIN_POW_WINDOW: u32 = {
    let mut p: u32 = 1;
    let mut i = 0;
    while i < RABIN_WINDOW {
        p = p.wrapping_mul(RABIN_MULTIPLIER);
        i += 1;
    }
    p
};

/// Compute the initial hash over exactly `RABIN_WINDOW` bytes.
fn rabin_hash_init(data: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in &data[..RABIN_WINDOW] {
        h = h.wrapping_mul(RABIN_MULTIPLIER).wrapping_add(b as u32);
    }
    h
}

/// Roll the hash forward: remove `old_byte`, add `new_byte`.
#[cfg(test)]
#[inline(always)]
fn rabin_hash_roll(h: u32, old_byte: u8, new_byte: u8) -> u32 {
    h.wrapping_mul(RABIN_MULTIPLIER)
        .wrapping_sub(RABIN_POW_WINDOW.wrapping_mul(old_byte as u32))
        .wrapping_add(new_byte as u32)
}

// ---------------------------------------------------------------------------
// Delta index
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// Index built over the source buffer for fast match lookup.
struct DeltaIndex<'a> {
    /// Map from hash → list of offsets in the source where that hash occurs.
    table: HashMap<u32, Vec<usize>>,
    source: &'a [u8],
}

impl<'a> DeltaIndex<'a> {
    fn new(source: &'a [u8]) -> Self {
        let mut table: HashMap<u32, Vec<usize>> = HashMap::new();

        if source.len() >= RABIN_WINDOW {
            let mut offset = 0;
            while offset + RABIN_WINDOW <= source.len() {
                let h = rabin_hash_init(&source[offset..offset + RABIN_WINDOW]);
                let entries = table.entry(h).or_default();
                if entries.len() < 64 {
                    entries.push(offset);
                }
                offset += INDEX_BLOCK_SIZE;
            }
        }

        Self { table, source }
    }

    /// Find the longest match in the source for target data starting at `target_pos`.
    fn find_match(&self, target: &[u8], target_pos: usize) -> Option<(usize, usize)> {
        if target_pos + RABIN_WINDOW > target.len() {
            return None;
        }

        let h = rabin_hash_init(&target[target_pos..target_pos + RABIN_WINDOW]);
        let entries = self.table.get(&h)?;

        let mut best_offset = 0;
        let mut best_len = 0;

        for &src_offset in entries {
            let max_len = std::cmp::min(self.source.len() - src_offset, target.len() - target_pos);

            let mut len = 0;
            while len < max_len && self.source[src_offset + len] == target[target_pos + len] {
                len += 1;
            }

            if len >= MIN_COPY_LEN && len > best_len {
                best_len = len;
                best_offset = src_offset;
            }
        }

        if best_len >= MIN_COPY_LEN {
            Some((best_offset, best_len))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction encoding (Git-compatible format)
// ---------------------------------------------------------------------------

/// Encode a variable-length integer (LEB128 unsigned).
fn encode_varint(mut val: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val > 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if val == 0 {
            break;
        }
    }
}

/// Decode a variable-length integer, returning (value, bytes_consumed).
fn decode_varint(data: &[u8]) -> Result<(u64, usize), DeltaError> {
    let mut val: u64 = 0;
    let mut shift: u32 = 0;
    let mut i = 0;
    loop {
        if i >= data.len() {
            return Err(DeltaError::Truncated);
        }
        if i >= MAX_VARINT_BYTES {
            return Err(DeltaError::VarIntOverflow);
        }
        let byte = data[i];
        val |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok((val, i))
}

/// Encode a copy instruction using a stack buffer (no heap allocation).
fn encode_copy(offset: usize, size: usize, out: &mut Vec<u8>) {
    let offset = offset as u32;
    let size = size as u32;
    let mut opcode: u8 = 0x80;
    let mut extra = [0u8; 7];
    let mut extra_len = 0;

    if offset & 0xFF != 0 {
        opcode |= 0x01;
        extra[extra_len] = (offset & 0xFF) as u8;
        extra_len += 1;
    }
    if offset & 0xFF00 != 0 {
        opcode |= 0x02;
        extra[extra_len] = ((offset >> 8) & 0xFF) as u8;
        extra_len += 1;
    }
    if offset & 0xFF_0000 != 0 {
        opcode |= 0x04;
        extra[extra_len] = ((offset >> 16) & 0xFF) as u8;
        extra_len += 1;
    }
    if offset & 0xFF00_0000 != 0 {
        opcode |= 0x08;
        extra[extra_len] = ((offset >> 24) & 0xFF) as u8;
        extra_len += 1;
    }

    let encoded_size = if size == 0x10000 { 0u32 } else { size };
    if encoded_size & 0xFF != 0 {
        opcode |= 0x10;
        extra[extra_len] = (encoded_size & 0xFF) as u8;
        extra_len += 1;
    }
    if encoded_size & 0xFF00 != 0 {
        opcode |= 0x20;
        extra[extra_len] = ((encoded_size >> 8) & 0xFF) as u8;
        extra_len += 1;
    }
    if encoded_size & 0xFF_0000 != 0 {
        opcode |= 0x40;
        extra[extra_len] = ((encoded_size >> 16) & 0xFF) as u8;
        extra_len += 1;
    }

    out.push(opcode);
    out.extend_from_slice(&extra[..extra_len]);
}

fn encode_insert(data: &[u8], out: &mut Vec<u8>) {
    debug_assert!(!data.is_empty() && data.len() <= MAX_INSERT_LEN);
    out.push(data.len() as u8);
    out.extend_from_slice(data);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Create a binary delta that transforms `source` into `target`.
///
/// Returns the encoded delta bytes, or `None` if the delta would not
/// achieve at least a 20% size reduction over storing `target` as-is.
pub fn create_delta(source: &[u8], target: &[u8]) -> Option<Vec<u8>> {
    let index = DeltaIndex::new(source);
    let mut delta = Vec::with_capacity(target.len() / 2);

    encode_varint(source.len() as u64, &mut delta);
    encode_varint(target.len() as u64, &mut delta);

    let mut target_pos = 0;
    let mut insert_start = 0;

    while target_pos < target.len() {
        if let Some((src_offset, match_len)) = index.find_match(target, target_pos) {
            if target_pos > insert_start {
                flush_insert(target, insert_start, target_pos, &mut delta);
            }

            let mut remaining = match_len;
            let mut off = src_offset;
            while remaining > 0 {
                let chunk = std::cmp::min(remaining, MAX_COPY_SIZE);
                encode_copy(off, chunk, &mut delta);
                off += chunk;
                remaining -= chunk;
            }

            target_pos += match_len;
            insert_start = target_pos;
        } else {
            target_pos += 1;
        }
    }

    if target_pos > insert_start {
        flush_insert(target, insert_start, target_pos, &mut delta);
    }

    let threshold = (target.len() as f64 * DELTA_RATIO_THRESHOLD) as usize;
    if delta.len() < threshold {
        Some(delta)
    } else {
        None
    }
}

fn flush_insert(target: &[u8], start: usize, end: usize, out: &mut Vec<u8>) {
    let data = &target[start..end];
    for chunk in data.chunks(MAX_INSERT_LEN) {
        encode_insert(chunk, out);
    }
}

/// Apply a delta to a source buffer, producing the original target.
pub fn apply_delta(source: &[u8], delta: &[u8]) -> Result<Vec<u8>, DeltaError> {
    if delta.is_empty() {
        return Err(DeltaError::Truncated);
    }

    let mut pos = 0;

    let (src_size, n) = decode_varint(&delta[pos..])?;
    pos += n;
    let (tgt_size, n) = decode_varint(&delta[pos..])?;
    pos += n;

    if src_size as usize != source.len() {
        return Err(DeltaError::SourceSizeMismatch {
            expected: src_size as usize,
            actual: source.len(),
        });
    }

    let mut output = Vec::with_capacity(tgt_size as usize);

    while pos < delta.len() {
        let cmd = delta[pos];
        pos += 1;

        if cmd & 0x80 != 0 {
            let mut offset: u32 = 0;
            let mut size: u32 = 0;

            if cmd & 0x01 != 0 {
                if pos >= delta.len() {
                    return Err(DeltaError::Truncated);
                }
                offset |= delta[pos] as u32;
                pos += 1;
            }
            if cmd & 0x02 != 0 {
                if pos >= delta.len() {
                    return Err(DeltaError::Truncated);
                }
                offset |= (delta[pos] as u32) << 8;
                pos += 1;
            }
            if cmd & 0x04 != 0 {
                if pos >= delta.len() {
                    return Err(DeltaError::Truncated);
                }
                offset |= (delta[pos] as u32) << 16;
                pos += 1;
            }
            if cmd & 0x08 != 0 {
                if pos >= delta.len() {
                    return Err(DeltaError::Truncated);
                }
                offset |= (delta[pos] as u32) << 24;
                pos += 1;
            }

            if cmd & 0x10 != 0 {
                if pos >= delta.len() {
                    return Err(DeltaError::Truncated);
                }
                size |= delta[pos] as u32;
                pos += 1;
            }
            if cmd & 0x20 != 0 {
                if pos >= delta.len() {
                    return Err(DeltaError::Truncated);
                }
                size |= (delta[pos] as u32) << 8;
                pos += 1;
            }
            if cmd & 0x40 != 0 {
                if pos >= delta.len() {
                    return Err(DeltaError::Truncated);
                }
                size |= (delta[pos] as u32) << 16;
                pos += 1;
            }

            if size == 0 {
                size = 0x10000;
            }

            let offset = offset as usize;
            let size = size as usize;

            if offset + size > source.len() {
                return Err(DeltaError::CopyOutOfBounds {
                    offset,
                    size,
                    source_len: source.len(),
                });
            }

            output.extend_from_slice(&source[offset..offset + size]);
        } else if cmd > 0 {
            let len = cmd as usize;
            if pos + len > delta.len() {
                return Err(DeltaError::Truncated);
            }
            output.extend_from_slice(&delta[pos..pos + len]);
            pos += len;
        } else {
            return Err(DeltaError::InvalidOpcode);
        }
    }

    if output.len() != tgt_size as usize {
        return Err(DeltaError::TargetSizeMismatch {
            expected: tgt_size as usize,
            actual: output.len(),
        });
    }

    Ok(output)
}

/// Errors that can occur during delta application.
#[derive(Debug)]
pub enum DeltaError {
    Truncated,
    VarIntOverflow,
    SourceSizeMismatch { expected: usize, actual: usize },
    TargetSizeMismatch { expected: usize, actual: usize },
    CopyOutOfBounds {
        offset: usize,
        size: usize,
        source_len: usize,
    },
    InvalidOpcode,
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "delta data truncated"),
            Self::VarIntOverflow => write!(f, "varint exceeds u64 capacity"),
            Self::SourceSizeMismatch { expected, actual } => {
                write!(f, "source size mismatch: expected {expected}, got {actual}")
            }
            Self::TargetSizeMismatch { expected, actual } => {
                write!(f, "target size mismatch: expected {expected}, got {actual}")
            }
            Self::CopyOutOfBounds {
                offset,
                size,
                source_len,
            } => write!(
                f,
                "copy out of bounds: offset {offset} + size {size} > source len {source_len}"
            ),
            Self::InvalidOpcode => write!(f, "invalid opcode 0 in delta"),
        }
    }
}

impl std::error::Error for DeltaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        for val in [0u64, 1, 127, 128, 16383, 16384, u32::MAX as u64, u64::MAX] {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf);
            let (decoded, len) = decode_varint(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, buf.len());
        }
    }

    #[test]
    fn test_varint_truncated() {
        assert!(matches!(decode_varint(&[]), Err(DeltaError::Truncated)));
        assert!(matches!(
            decode_varint(&[0x80]),
            Err(DeltaError::Truncated)
        ));
    }

    #[test]
    fn test_varint_overflow() {
        // 11 continuation bytes — exceeds MAX_VARINT_BYTES
        let bad = [0x80u8; 11];
        assert!(matches!(
            decode_varint(&bad),
            Err(DeltaError::VarIntOverflow)
        ));
    }

    #[test]
    fn test_identical_buffers() {
        let data = b"Hello, world! This is a test of delta encoding.";
        let delta = create_delta(data, data);
        if let Some(delta) = &delta {
            let restored = apply_delta(data, delta).unwrap();
            assert_eq!(restored, data);
        }
    }

    #[test]
    fn test_small_edit() {
        let source = b"fn main() {\n    println!(\"hello world\");\n}\n";
        let target = b"fn main() {\n    println!(\"hello lance\");\n}\n";
        if let Some(delta) = create_delta(source, target) {
            let restored = apply_delta(source, &delta).unwrap();
            assert_eq!(restored, target.as_slice());
        }
    }

    #[test]
    fn test_larger_similar_files() {
        let mut source = String::new();
        for i in 0..100 {
            source.push_str(&format!(
                "line {}: the quick brown fox jumps over the lazy dog\n",
                i
            ));
        }
        let mut target = source.clone();
        target = target.replace("line 10:", "line 10 MODIFIED:");
        target = target.replace("line 50:", "line 50 MODIFIED:");
        target.push_str("// new line at the end\n");

        let delta =
            create_delta(source.as_bytes(), target.as_bytes()).expect("should compress well");
        assert!(
            delta.len() < target.len(),
            "delta {} should be smaller than target {}",
            delta.len(),
            target.len()
        );

        let restored = apply_delta(source.as_bytes(), &delta).unwrap();
        assert_eq!(restored, target.as_bytes());
    }

    #[test]
    fn test_completely_different() {
        let source = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let target = b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        if let Some(delta) = create_delta(source, target) {
            let restored = apply_delta(source, &delta).unwrap();
            assert_eq!(restored, target.as_slice());
        }
    }

    #[test]
    fn test_empty_target() {
        let source = b"some source data here";
        let target = b"";
        if let Some(delta) = create_delta(source, target) {
            let restored = apply_delta(source, &delta).unwrap();
            assert!(restored.is_empty());
        }
    }

    #[test]
    fn test_empty_source() {
        let source = b"";
        let target = b"new content";
        if let Some(delta) = create_delta(source, target) {
            let restored = apply_delta(source, &delta).unwrap();
            assert_eq!(restored, target.as_slice());
        }
    }

    #[test]
    fn test_chained_deltas() {
        let v1 = b"version 1 of the file with some content that stays the same\n\
                    and more lines here\nand more lines here\n";
        let v2 = b"version 2 of the file with some content that stays the same\n\
                    and more lines here\nand a new line\n";
        let v3 = b"version 3 of the file with some content that stays the same\n\
                    and more lines here\nand a new line\nand another\n";

        if let Some(d1) = create_delta(v1, v2) {
            let restored_v2 = apply_delta(v1, &d1).unwrap();
            assert_eq!(restored_v2, v2.as_slice());

            if let Some(d2) = create_delta(&restored_v2, v3) {
                let restored_v3 = apply_delta(&restored_v2, &d2).unwrap();
                assert_eq!(restored_v3, v3.as_slice());
            }
        }
    }

    #[test]
    fn test_source_code_diff() {
        let source = r#"
use std::collections::HashMap;

fn process_data(input: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    for &byte in input {
        if byte > 0 {
            result.push(byte);
        }
    }
    result
}

fn main() {
    let data = vec![1, 2, 0, 3, 0, 4];
    let processed = process_data(&data);
    println!("Result: {:?}", processed);
}
"#;
        let target = r#"
use std::collections::HashMap;

fn process_data(input: &[u8], threshold: u8) -> Vec<u8> {
    let mut result = Vec::new();
    for &byte in input {
        if byte > threshold {
            result.push(byte);
        }
    }
    result
}

fn main() {
    let data = vec![1, 2, 0, 3, 0, 4];
    let processed = process_data(&data, 1);
    println!("Result: {:?}", processed);
    println!("Done!");
}
"#;
        let delta = create_delta(source.as_bytes(), target.as_bytes())
            .expect("similar source code should delta well");
        assert!(delta.len() < target.len());

        let restored = apply_delta(source.as_bytes(), &delta).unwrap();
        assert_eq!(String::from_utf8(restored).unwrap(), target,);
    }

    #[test]
    fn test_apply_delta_truncated_copy() {
        // Craft a delta with a copy instruction that's truncated
        let source = b"hello world";
        let mut delta = Vec::new();
        encode_varint(source.len() as u64, &mut delta);
        encode_varint(5, &mut delta); // target size 5
        delta.push(0x81); // copy with offset byte 0 present, but no byte follows
        assert!(matches!(
            apply_delta(source, &delta),
            Err(DeltaError::Truncated)
        ));
    }

    #[test]
    fn test_rolling_hash_consistency() {
        let data = b"abcdefghijklmnopqrstuvwxyz012345678";
        let h1 = rabin_hash_init(&data[0..RABIN_WINDOW]);
        // Roll from position 0 to position 1
        let h2_rolled = rabin_hash_roll(h1, data[0], data[RABIN_WINDOW]);
        let h2_direct = rabin_hash_init(&data[1..1 + RABIN_WINDOW]);
        assert_eq!(h2_rolled, h2_direct);
    }
}
