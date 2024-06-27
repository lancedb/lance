// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Feature flags

use snafu::{location, Location};

use crate::format::Manifest;
use lance_core::{Error, Result};

/// Fragments may contain deletion files, which record the tombstones of
/// soft-deleted rows.
pub const FLAG_DELETION_FILES: u64 = 1;
/// Row ids are table after moves, but not updates. Fragments contain an index
/// mapping row ids to row addresses.
pub const FLAG_MOVE_STABLE_ROW_IDS: u64 = 2;
/// Files are written with the new v2 format (temporary flag, will be removed
/// once v2 is the default format)
pub const FLAG_USE_V2_FORMAT: u64 = 4;
/// The first bit that is unknown as a feature flag
pub const FLAG_UNKNOWN: u64 = 8;

/// Set the reader and writer feature flags in the manifest based on the contents of the manifest.
pub fn apply_feature_flags(manifest: &mut Manifest, enable_stable_row_id: bool) -> Result<()> {
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
    if has_row_ids || enable_stable_row_id {
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
        manifest.reader_feature_flags |= FLAG_MOVE_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_MOVE_STABLE_ROW_IDS;
    }

    Ok(())
}

pub fn can_read_dataset(reader_flags: u64) -> bool {
    reader_flags < FLAG_UNKNOWN
}

pub fn can_write_dataset(writer_flags: u64) -> bool {
    writer_flags < FLAG_UNKNOWN
}

pub fn should_use_legacy_format(writer_flags: u64) -> bool {
    writer_flags & FLAG_USE_V2_FORMAT == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_check() {
        assert!(can_read_dataset(0));
        assert!(can_read_dataset(super::FLAG_DELETION_FILES));
        assert!(can_read_dataset(super::FLAG_MOVE_STABLE_ROW_IDS));
        assert!(can_read_dataset(super::FLAG_USE_V2_FORMAT));
        assert!(can_read_dataset(
            super::FLAG_DELETION_FILES | super::FLAG_MOVE_STABLE_ROW_IDS
        ));
        assert!(!can_read_dataset(super::FLAG_UNKNOWN));
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES));
        assert!(can_write_dataset(super::FLAG_MOVE_STABLE_ROW_IDS));
        assert!(can_read_dataset(super::FLAG_USE_V2_FORMAT));
        assert!(can_write_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_MOVE_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT
        ));
        assert!(!can_write_dataset(super::FLAG_UNKNOWN));
    }
}
