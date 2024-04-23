// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Feature flags

use lance_table::format::Manifest;

pub const FLAG_DELETION_FILES: u64 = 1;

/// Set the reader and writer feature flags in the manifest based on the contents of the manifest.
pub fn apply_feature_flags(manifest: &mut Manifest) {
    // Reset flags
    manifest.reader_feature_flags = 0;
    manifest.writer_feature_flags = 0;

    let has_deletion_files = manifest
        .fragments
        .iter()
        .any(|frag| frag.deletion_file.is_some());
    if has_deletion_files {
        // Both readers and writers need to be able to read deletion files
        manifest.reader_feature_flags |= FLAG_DELETION_FILES;
        manifest.writer_feature_flags |= FLAG_DELETION_FILES;
    }
}

pub fn can_read_dataset(reader_flags: u64) -> bool {
    reader_flags <= 1
}

pub fn can_write_dataset(writer_flags: u64) -> bool {
    writer_flags <= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_check() {
        assert!(can_read_dataset(0));
        assert!(can_read_dataset(super::FLAG_DELETION_FILES));
        assert!(!can_read_dataset(super::FLAG_DELETION_FILES + 1));
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES));
        assert!(!can_write_dataset(super::FLAG_DELETION_FILES + 1));
    }
}
