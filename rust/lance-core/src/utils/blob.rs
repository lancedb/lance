// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use object_store::path::Path;
use rand::RngCore;

/// Directory name for blob sidecar files.
pub const BLOB_SIDECAR_DIR: &str = "_blob";

/// Format a dedicated blob sidecar path.
///
/// Layout: `_blob/<fragment_id>/<field_id>/<blob_id>.raw`
pub fn blob_path(base: &Path, fragment_id: u32, field_id: u32, blob_id: u32) -> Path {
    let file_name = format!("{:08x}.raw", blob_id);
    base.child(BLOB_SIDECAR_DIR)
        .child(format!("{:08x}", fragment_id))
        .child(format!("{:08x}", field_id))
        .child(file_name.as_str())
}

/// Generate a high-entropy prefix using the same pattern as data file names.
///
/// Pattern: first 24 bits as binary, remaining 13 bytes as hex (26 chars).
pub fn generate_random_prefix() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);

    let mut out = String::with_capacity(50);

    for &b in &bytes[..3] {
        for i in (0..8).rev() {
            out.push(if (b >> i) & 1 == 1 { '1' } else { '0' });
        }
    }

    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &b in &bytes[3..] {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_path_formatting() {
        let base = Path::from("base");
        let path = blob_path(&base, 0x10, 1, 2);
        assert_eq!(
            path.to_string(),
            "base/_blob/00000010/00000001/00000002.raw"
        );
    }
}
