// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Binary delta encoding inspired by Git's packfile delta format.
//!
//! Encodes a target buffer as a series of copy/insert instructions relative
//! to a source (base) buffer. Copy instructions reference byte ranges in the
//! source; insert instructions carry literal bytes from the target.

use std::collections::HashMap;

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

// ---------------------------------------------------------------------------
// Rabin hash helpers
// ---------------------------------------------------------------------------

/// Simple polynomial rolling hash over a window of bytes.
/// Not the full Rabin with lookup tables — a simpler variant that's
/// sufficient for finding matching blocks.
fn rabin_hash(data: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in data {
        h = h.wrapping_mul(257).wrapping_add(b as u32);
    }
    h
}

/// Index built over the source buffer for fast match lookup.
struct DeltaIndex {
    /// Map from hash → list of offsets in the source where that hash occurs.
    table: HashMap<u32, Vec<usize>>,
    source: Vec<u8>,
}

impl DeltaIndex {
    fn new(source: &[u8]) -> Self {
        let mut table: HashMap<u32, Vec<usize>> = HashMap::new();

        if source.len() >= RABIN_WINDOW {
            let mut offset = 0;
            while offset + RABIN_WINDOW <= source.len() {
                let h = rabin_hash(&source[offset..offset + RABIN_WINDOW]);
                let entries = table.entry(h).or_default();
                // Limit bucket size to avoid quadratic behavior on repetitive data
                if entries.len() < 64 {
                    entries.push(offset);
                }
                offset += INDEX_BLOCK_SIZE;
            }
        }

        Self {
            table,
            source: source.to_vec(),
        }
    }

    /// Find the longest match in the source for target data starting at `target_pos`.
    /// Returns (source_offset, match_length) or None.
    fn find_match(&self, target: &[u8], target_pos: usize) -> Option<(usize, usize)> {
        if target_pos + RABIN_WINDOW > target.len() {
            return None;
        }

        let h = rabin_hash(&target[target_pos..target_pos + RABIN_WINDOW]);
        let entries = self.table.get(&h)?;

        let mut best_offset = 0;
        let mut best_len = 0;

        for &src_offset in entries {
            // Verify the hash match with actual byte comparison and extend
            let max_len = std::cmp::min(
                self.source.len() - src_offset,
                target.len() - target_pos,
            );

            let mut len = 0;
            while len < max_len
                && self.source[src_offset + len] == target[target_pos + len]
            {
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
fn decode_varint(data: &[u8]) -> (u64, usize) {
    let mut val: u64 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        let byte = data[i];
        val |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (val, i)
}

/// Encode a copy instruction.
/// Format: opcode byte with bit 7 set, followed by non-zero offset/size bytes.
fn encode_copy(offset: usize, size: usize, out: &mut Vec<u8>) {
    let offset = offset as u32;
    let size = size as u32;
    let mut opcode: u8 = 0x80;
    let mut extra = Vec::with_capacity(7);

    if offset & 0xFF != 0 {
        opcode |= 0x01;
        extra.push((offset & 0xFF) as u8);
    }
    if offset & 0xFF00 != 0 {
        opcode |= 0x02;
        extra.push(((offset >> 8) & 0xFF) as u8);
    }
    if offset & 0xFF_0000 != 0 {
        opcode |= 0x04;
        extra.push(((offset >> 16) & 0xFF) as u8);
    }
    if offset & 0xFF00_0000 != 0 {
        opcode |= 0x08;
        extra.push(((offset >> 24) & 0xFF) as u8);
    }

    // Size: if size == 0x10000, encode as 0 (special case in Git format)
    let encoded_size = if size == 0x10000 { 0u32 } else { size };
    if encoded_size & 0xFF != 0 {
        opcode |= 0x10;
        extra.push((encoded_size & 0xFF) as u8);
    }
    if encoded_size & 0xFF00 != 0 {
        opcode |= 0x20;
        extra.push(((encoded_size >> 8) & 0xFF) as u8);
    }
    if encoded_size & 0xFF_0000 != 0 {
        opcode |= 0x40;
        extra.push(((encoded_size >> 16) & 0xFF) as u8);
    }

    out.push(opcode);
    out.extend_from_slice(&extra);
}

/// Encode an insert instruction.
/// Format: opcode byte (1-127) = literal count, followed by that many bytes.
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
/// Returns the encoded delta bytes, or `None` if the delta would be
/// larger than the target (caller should store the target as-is).
pub fn create_delta(source: &[u8], target: &[u8]) -> Option<Vec<u8>> {
    let index = DeltaIndex::new(source);
    let mut delta = Vec::with_capacity(target.len() / 2);

    // Header: source size, target size
    encode_varint(source.len() as u64, &mut delta);
    encode_varint(target.len() as u64, &mut delta);

    let mut target_pos = 0;
    let mut insert_start = 0;

    while target_pos < target.len() {
        if let Some((src_offset, match_len)) = index.find_match(target, target_pos) {
            // Flush pending insert
            if target_pos > insert_start {
                flush_insert(target, insert_start, target_pos, &mut delta);
            }

            // Emit copy instruction(s), splitting if > MAX_COPY_SIZE
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

    // Flush trailing insert
    if target_pos > insert_start {
        flush_insert(target, insert_start, target_pos, &mut delta);
    }

    // Only use delta if it's actually smaller
    if delta.len() < target.len() {
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
///
/// Returns an error if the delta is malformed or sizes don't match.
pub fn apply_delta(source: &[u8], delta: &[u8]) -> Result<Vec<u8>, DeltaError> {
    if delta.is_empty() {
        return Err(DeltaError::Truncated);
    }

    let mut pos = 0;

    // Read header
    let (src_size, n) = decode_varint(&delta[pos..]);
    pos += n;
    let (tgt_size, n) = decode_varint(&delta[pos..]);
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
            // Copy instruction
            let mut offset: u32 = 0;
            let mut size: u32 = 0;

            if cmd & 0x01 != 0 {
                offset |= delta[pos] as u32;
                pos += 1;
            }
            if cmd & 0x02 != 0 {
                offset |= (delta[pos] as u32) << 8;
                pos += 1;
            }
            if cmd & 0x04 != 0 {
                offset |= (delta[pos] as u32) << 16;
                pos += 1;
            }
            if cmd & 0x08 != 0 {
                offset |= (delta[pos] as u32) << 24;
                pos += 1;
            }

            if cmd & 0x10 != 0 {
                size |= delta[pos] as u32;
                pos += 1;
            }
            if cmd & 0x20 != 0 {
                size |= (delta[pos] as u32) << 8;
                pos += 1;
            }
            if cmd & 0x40 != 0 {
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
            // Insert instruction
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
            let (decoded, len) = decode_varint(&buf);
            assert_eq!(decoded, val);
            assert_eq!(len, buf.len());
        }
    }

    #[test]
    fn test_identical_buffers() {
        let data = b"Hello, world! This is a test of delta encoding.";
        let delta = create_delta(data, data).expect("identical data should produce small delta");
        let restored = apply_delta(data, &delta).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn test_small_edit() {
        let source = b"fn main() {\n    println!(\"hello world\");\n}\n";
        let target = b"fn main() {\n    println!(\"hello lance\");\n}\n";
        let delta = create_delta(source, target);
        // Whether or not delta is smaller, apply_delta should roundtrip if we have one
        if let Some(delta) = &delta {
            let restored = apply_delta(source, delta).unwrap();
            assert_eq!(restored, target.as_slice());
        }
    }

    #[test]
    fn test_larger_similar_files() {
        // Simulate two versions of a source file
        let mut source = String::new();
        for i in 0..100 {
            source.push_str(&format!("line {}: the quick brown fox jumps over the lazy dog\n", i));
        }
        let mut target = source.clone();
        // Change a few lines
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
        // Delta might be None (larger than target) or Some
        if let Some(delta) = create_delta(source, target) {
            let restored = apply_delta(source, &delta).unwrap();
            assert_eq!(restored, target.as_slice());
        }
        // Either way, this shouldn't panic
    }

    #[test]
    fn test_empty_target() {
        let source = b"some source data here";
        let target = b"";
        let delta = create_delta(source, target);
        if let Some(delta) = &delta {
            let restored = apply_delta(source, delta).unwrap();
            assert!(restored.is_empty());
        }
    }

    #[test]
    fn test_empty_source() {
        let source = b"";
        let target = b"new content";
        // Delta from empty source = all inserts, likely larger than target
        let delta = create_delta(source, target);
        if let Some(delta) = &delta {
            let restored = apply_delta(source, delta).unwrap();
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

        // Chain: v1 -> v2 -> v3
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
        assert_eq!(
            String::from_utf8(restored).unwrap(),
            target,
        );
    }
}
