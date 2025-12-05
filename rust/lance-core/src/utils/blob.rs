// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use object_store::path::Path;

/// Format a dedicated blob sidecar path for a data file.
///
/// Layout: `<base>/<data_file_key>/<blob_id>.raw`
/// - `base` is typically the dataset's data directory.
/// - `data_file_key` is the stem of the data file (without extension).
pub fn blob_path(base: &Path, data_file_key: &str, blob_id: u32) -> Path {
    let file_name = format!("{:08x}.raw", blob_id);
    base.child(data_file_key).child(file_name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_path_formatting() {
        let base = Path::from("base");
        let path = blob_path(&base, "deadbeef", 2);
        assert_eq!(path.to_string(), "base/deadbeef/00000002.raw");
    }
}
