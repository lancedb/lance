// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use object_store::path::Path;

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
