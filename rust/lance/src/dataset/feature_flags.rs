// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Feature flags

use snafu::{location, Location};

use crate::{Error, Result};
use lance_table::format::Manifest;

pub const FLAG_DELETION_FILES: u64 = 1;
pub const FLAG_ROW_IDS: u64 = 2;

/// Set the reader and writer feature flags in the manifest based on the contents of the manifest.
pub fn apply_feature_flags(manifest: &mut Manifest) -> Result<()> {
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

    // If any fragment has row ids, they must all have row ids.
    let has_row_ids = manifest
        .fragments
        .iter()
        .any(|frag| frag.row_id_meta.is_some());
    if has_row_ids {
        if !manifest
            .fragments
            .iter()
            .all(|frag| frag.row_id_meta.is_some())
        {
            return Err(Error::invalid_input(
                "All fragments must have row ids",
                location!(),
            ));
        }
        manifest.reader_feature_flags |= FLAG_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_ROW_IDS;
    }

    Ok(())
}

pub fn can_read_dataset(reader_flags: u64) -> bool {
    reader_flags <= 2
}

pub fn can_write_dataset(writer_flags: u64) -> bool {
    writer_flags <= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_check() {
        assert!(can_read_dataset(0));
        assert!(can_read_dataset(super::FLAG_DELETION_FILES));
        assert!(can_read_dataset(super::FLAG_ROW_IDS));
        assert!(can_read_dataset(
            super::FLAG_DELETION_FILES | super::FLAG_ROW_IDS
        ));
        assert!(!can_read_dataset(super::FLAG_ROW_IDS + 1));
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES));
        assert!(can_write_dataset(super::FLAG_ROW_IDS));
        assert!(can_write_dataset(
            super::FLAG_DELETION_FILES | super::FLAG_ROW_IDS
        ));
        assert!(!can_write_dataset(super::FLAG_ROW_IDS + 1));
    }
}
