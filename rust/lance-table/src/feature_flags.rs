// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Feature flags

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
/// Dataset uses multiple base paths (for shallow clones or multi-base datasets)
pub const FLAG_BASE_PATHS: u64 = 16;
/// Disable writing transaction file under _transaction/, this flag is set when we only want to write inline transaction in manifest
pub const FLAG_DISABLE_TRANSACTION_FILE: u64 = 32;
/// The dataset relies on one or more experimental features, named in the
/// manifest's `experimental_reader_features` / `experimental_writer_features`.
///
/// This bit is the fail-closed anchor for the experimental-feature mechanism.
/// It is set whenever the corresponding experimental feature list is non-empty.
/// Libraries built before a given experiment existed treat bit 64 as unknown
/// (it was previously `FLAG_UNKNOWN`) and reject the dataset without needing to
/// parse the feature names. Libraries that understand the mechanism defer to the
/// name lists via [`can_read_dataset`] / [`can_write_dataset`]. See
/// `rust/lance-table/design/experimental_feature_flags.md`.
pub const FLAG_EXPERIMENTAL: u64 = 64;
/// The first bit that is unknown as a feature flag.
pub const FLAG_UNKNOWN: u64 = 128;

// The experimental bit must be a *known* bit so admission is gated by the name
// lists, not the bitmap boundary.
const _: () = assert!(FLAG_EXPERIMENTAL < FLAG_UNKNOWN);

/// The experimental feature names this build understands.
///
/// Each name is registered here only when the Cargo feature that implements it
/// is enabled, so a default build understands none and therefore rejects any
/// dataset that declares an experimental feature. When an experiment graduates,
/// its name moves out of this list and onto a dedicated feature-flag bit.
pub fn known_experimental_features() -> &'static [&'static str] {
    &[
        // Register experimental feature names here, gated by their Cargo feature:
        //   #[cfg(feature = "unstable-action-transactions")] "action-transactions",
    ]
}

/// Whether every name in `experimental_features` is understood by this build.
fn understands_experimental_features(experimental_features: &[String]) -> bool {
    let known = known_experimental_features();
    experimental_features
        .iter()
        .all(|feature| known.contains(&feature.as_str()))
}

/// Set the reader and writer feature flags in the manifest based on the contents of the manifest.
pub fn apply_feature_flags(
    manifest: &mut Manifest,
    enable_stable_row_id: bool,
    disable_transaction_file: bool,
) -> Result<()> {
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
            return Err(Error::invalid_input("All fragments must have row ids"));
        }
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_STABLE_ROW_IDS;
    }

    // Test whether any table metadata has been set
    if !manifest.config.is_empty() {
        manifest.writer_feature_flags |= FLAG_TABLE_CONFIG;
    }

    // Check if this dataset uses multiple base paths (for shallow clones or multi-base datasets)
    if !manifest.base_paths.is_empty() {
        manifest.reader_feature_flags |= FLAG_BASE_PATHS;
        manifest.writer_feature_flags |= FLAG_BASE_PATHS;
    }

    if disable_transaction_file {
        manifest.writer_feature_flags |= FLAG_DISABLE_TRANSACTION_FILE;
    }

    // The experimental bit is the fail-closed anchor for the named experimental
    // feature lists; keep it in sync with them.
    if !manifest.experimental_reader_features.is_empty() {
        manifest.reader_feature_flags |= FLAG_EXPERIMENTAL;
    }
    if !manifest.experimental_writer_features.is_empty() {
        manifest.writer_feature_flags |= FLAG_EXPERIMENTAL;
    }
    Ok(())
}

/// Whether this build can read a dataset with the given reader feature flags and
/// declared experimental reader features.
///
/// Rejects if any non-experimental unknown bit is set, or if any declared
/// experimental feature is not understood by this build.
pub fn can_read_dataset(reader_flags: u64, experimental_reader_features: &[String]) -> bool {
    if reader_flags & !(FLAG_UNKNOWN - 1) != 0 {
        // A bit at or above FLAG_UNKNOWN is set — a feature we don't know about.
        return false;
    }
    understands_experimental_features(experimental_reader_features)
}

/// Whether this build can write to a dataset with the given writer feature flags
/// and declared experimental writer features. See [`can_read_dataset`].
pub fn can_write_dataset(writer_flags: u64, experimental_writer_features: &[String]) -> bool {
    if writer_flags & !(FLAG_UNKNOWN - 1) != 0 {
        return false;
    }
    understands_experimental_features(experimental_writer_features)
}

pub fn has_deprecated_v2_feature_flag(writer_flags: u64) -> bool {
    writer_flags & FLAG_USE_V2_FORMAT_DEPRECATED != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::BasePath;

    const NO_FEATURES: &[String] = &[];

    #[test]
    fn test_read_check() {
        assert!(can_read_dataset(0, NO_FEATURES));
        assert!(can_read_dataset(super::FLAG_DELETION_FILES, NO_FEATURES));
        assert!(can_read_dataset(super::FLAG_STABLE_ROW_IDS, NO_FEATURES));
        assert!(can_read_dataset(
            super::FLAG_USE_V2_FORMAT_DEPRECATED,
            NO_FEATURES
        ));
        assert!(can_read_dataset(super::FLAG_TABLE_CONFIG, NO_FEATURES));
        assert!(can_read_dataset(super::FLAG_BASE_PATHS, NO_FEATURES));
        assert!(can_read_dataset(
            super::FLAG_DISABLE_TRANSACTION_FILE,
            NO_FEATURES
        ));
        assert!(can_read_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED,
            NO_FEATURES
        ));
        assert!(!can_read_dataset(super::FLAG_UNKNOWN, NO_FEATURES));
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0, NO_FEATURES));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES, NO_FEATURES));
        assert!(can_write_dataset(super::FLAG_STABLE_ROW_IDS, NO_FEATURES));
        assert!(can_write_dataset(
            super::FLAG_USE_V2_FORMAT_DEPRECATED,
            NO_FEATURES
        ));
        assert!(can_write_dataset(super::FLAG_TABLE_CONFIG, NO_FEATURES));
        assert!(can_write_dataset(super::FLAG_BASE_PATHS, NO_FEATURES));
        assert!(can_write_dataset(
            super::FLAG_DISABLE_TRANSACTION_FILE,
            NO_FEATURES
        ));
        assert!(can_write_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED
                | super::FLAG_TABLE_CONFIG
                | super::FLAG_BASE_PATHS,
            NO_FEATURES
        ));
        assert!(!can_write_dataset(super::FLAG_UNKNOWN, NO_FEATURES));
    }

    #[test]
    fn test_experimental_feature_admission() {
        // A default build understands no experimental features, so any declared
        // experimental feature is rejected on both read and write paths.
        let declared = vec!["some-experiment".to_string()];
        assert!(!can_read_dataset(FLAG_EXPERIMENTAL, &declared));
        assert!(!can_write_dataset(FLAG_EXPERIMENTAL, &declared));

        // The experimental bit with no declared features is harmless — there is
        // nothing the reader could fail to understand. (Pre-mechanism libraries
        // still reject bit 64, since their FLAG_UNKNOWN was 64.)
        assert!(can_read_dataset(FLAG_EXPERIMENTAL, NO_FEATURES));
        assert!(can_write_dataset(FLAG_EXPERIMENTAL, NO_FEATURES));
    }

    #[test]
    fn test_apply_sets_experimental_flag() {
        use crate::format::{DataStorageFormat, Manifest};
        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema;
        use std::collections::HashMap;
        use std::sync::Arc;

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "x",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.experimental_writer_features = vec!["some-experiment".to_string()];

        apply_feature_flags(&mut manifest, false, false).unwrap();

        assert_ne!(manifest.writer_feature_flags & FLAG_EXPERIMENTAL, 0);
        assert_eq!(manifest.reader_feature_flags & FLAG_EXPERIMENTAL, 0);
    }

    #[test]
    fn test_base_paths_feature_flags() {
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
        // Test 1: Normal dataset (no base_paths) should not have FLAG_BASE_PATHS
        let mut normal_manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(), // Empty base_paths
        );
        apply_feature_flags(&mut normal_manifest, false, false).unwrap();
        assert_eq!(normal_manifest.reader_feature_flags & FLAG_BASE_PATHS, 0);
        assert_eq!(normal_manifest.writer_feature_flags & FLAG_BASE_PATHS, 0);
        // Test 2: Dataset with base_paths (shallow clone or multi-base) should have FLAG_BASE_PATHS
        let mut base_paths: HashMap<u32, BasePath> = HashMap::new();
        base_paths.insert(
            1,
            BasePath::new(
                1,
                "file:///path/to/original".to_string(),
                Some("test_ref".to_string()),
                true,
            ),
        );
        let mut multi_base_manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            base_paths,
        );
        apply_feature_flags(&mut multi_base_manifest, false, false).unwrap();
        assert_ne!(
            multi_base_manifest.reader_feature_flags & FLAG_BASE_PATHS,
            0
        );
        assert_ne!(
            multi_base_manifest.writer_feature_flags & FLAG_BASE_PATHS,
            0
        );
    }
}
