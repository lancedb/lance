// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Feature flags

use snafu::location;

use crate::format::Manifest;
use lance_core::{Error, Result};

/// Fragments may contain deletion files, which record the tombstones of
/// soft-deleted rows.
pub const FLAG_DELETION_FILES: u64 = 1;
/// Row ids are stable for both moves and updates. Fragments contain an index
/// mapping row ids to row addresses.
pub const FLAG_STABLE_ROW_IDS: u64 = 2;
/// Files are written with the new v2 format (this flag is no longer used)
pub const FLAG_USE_V2_FORMAT_DEPRECATED: u64 = 4;
/// Table config is present
pub const FLAG_TABLE_CONFIG: u64 = 8;
/// Dataset is a shallow clone with external base paths
pub const FLAG_SHALLOW_CLONE: u64 = 16;
/// The first bit that is unknown as a feature flag
pub const FLAG_UNKNOWN: u64 = 32;

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
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_STABLE_ROW_IDS;
    }

    // Test whether any table metadata has been set
    if !manifest.config.is_empty() {
        manifest.writer_feature_flags |= FLAG_TABLE_CONFIG;
    }

    // Check if this is a shallow clone dataset by examining base_paths
    if !manifest.base_paths.is_empty() {
        manifest.reader_feature_flags |= FLAG_SHALLOW_CLONE;
        manifest.writer_feature_flags |= FLAG_SHALLOW_CLONE;
    }

    Ok(())
}

pub fn can_read_dataset(reader_flags: u64) -> bool {
    reader_flags < FLAG_UNKNOWN
}

pub fn can_write_dataset(writer_flags: u64) -> bool {
    writer_flags < FLAG_UNKNOWN
}

pub fn has_deprecated_v2_feature_flag(writer_flags: u64) -> bool {
    writer_flags & FLAG_USE_V2_FORMAT_DEPRECATED != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::BasePath;

    #[test]
    fn test_read_check() {
        assert!(can_read_dataset(0));
        assert!(can_read_dataset(super::FLAG_DELETION_FILES));
        assert!(can_read_dataset(super::FLAG_STABLE_ROW_IDS));
        assert!(can_read_dataset(super::FLAG_USE_V2_FORMAT_DEPRECATED));
        assert!(can_read_dataset(super::FLAG_TABLE_CONFIG));
        assert!(can_read_dataset(super::FLAG_SHALLOW_CLONE));
        assert!(can_read_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED
        ));
        assert!(!can_read_dataset(super::FLAG_UNKNOWN));
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES));
        assert!(can_write_dataset(super::FLAG_STABLE_ROW_IDS));
        assert!(can_write_dataset(super::FLAG_USE_V2_FORMAT_DEPRECATED));
        assert!(can_write_dataset(super::FLAG_TABLE_CONFIG));
        assert!(can_write_dataset(super::FLAG_SHALLOW_CLONE));
        assert!(can_write_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED
                | super::FLAG_TABLE_CONFIG
                | super::FLAG_SHALLOW_CLONE
        ));
        assert!(!can_write_dataset(super::FLAG_UNKNOWN));
    }

    #[test]
    fn test_shallow_clone_feature_flags() {
        use crate::format::{DataStorageFormat, Manifest};
        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema;
        use std::collections::HashMap;
        use std::sync::Arc;
        // Create a basic schema for testing
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "test_field",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        // Test 1: Normal dataset (no base_paths) should not have FLAG_SHALLOW_CLONE
        let mut normal_manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            None,
            HashMap::new(), // Empty base_paths
        );
        apply_feature_flags(&mut normal_manifest, false).unwrap();
        assert_eq!(normal_manifest.reader_feature_flags & FLAG_SHALLOW_CLONE, 0);
        assert_eq!(normal_manifest.writer_feature_flags & FLAG_SHALLOW_CLONE, 0);
        // Test 2: Cloned dataset (with base_paths) should have FLAG_SHALLOW_CLONE
        let mut base_paths: HashMap<u32, BasePath> = HashMap::new();
        base_paths.insert(
            1,
            BasePath {
                id: 1,
                name: Some("test_ref".to_string()),
                is_dataset_root: true,
                path: "/path/to/original".to_string(),
            },
        );
        let mut cloned_manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            None,
            base_paths,
        );
        apply_feature_flags(&mut cloned_manifest, false).unwrap();
        assert_ne!(cloned_manifest.reader_feature_flags & FLAG_SHALLOW_CLONE, 0);
        assert_ne!(cloned_manifest.writer_feature_flags & FLAG_SHALLOW_CLONE, 0);
    }
}
