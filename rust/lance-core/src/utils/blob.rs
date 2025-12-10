// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use object_store::path::Path;

/// Format a dedicated blob sidecar path for a data file.
///
/// Layout: `<base>/<data_file_key>/<blob_id>.raw`
/// - `base` is typically the dataset's data directory.
/// - `data_file_key` is the stem of the data file (without extension).
pub fn dedicated_blob_path(base: &Path, data_file_key: &str, blob_id: u32) -> Path {
    let file_name = format!("{:08x}.raw", blob_id);
    base.child(data_file_key).child(file_name.as_str())
}

/// Format a packed blob sidecar path for a data file.
///
/// Layout: `<base>/<data_file_key>/<blob_id>.pack`
pub fn pack_blob_path(base: &Path, data_file_key: &str, blob_id: u32) -> Path {
    let file_name = format!("{:08x}.pack", blob_id);
    base.child(data_file_key).child(file_name.as_str())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedicated_blob_path_formatting() {
        let base = Path::from("base");
        let path = dedicated_blob_path(&base, "deadbeef", 2);
        assert_eq!(path.to_string(), "base/deadbeef/00000002.raw");
    }

    #[test]
    fn test_pack_blob_path_formatting() {
        let base = Path::from("base");
        let path = pack_blob_path(&base, "cafebabe", 3);
        assert_eq!(path.to_string(), "base/cafebabe/00000003.pack");
    }
}
