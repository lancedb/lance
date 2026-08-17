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
/// Fragments contain data overlay files, which supply new values for a subset of
/// cells without rewriting base data files. A reader that does not understand
/// overlays must refuse the dataset, since ignoring an overlay would silently
/// return stale base values.
///
/// Data overlay files are not yet a released feature: in release builds this flag
/// is treated as unknown (so a release reader/writer refuses an overlay dataset)
/// unless [`ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV`] is set, which lets benchmarks opt in.
/// Debug builds always understand it so tests exercise the path.
pub const FLAG_UNSTABLE_DATA_OVERLAY_FILES: u64 = 64;
/// `index_catchup` is maintained on this table, so a missing entry means the
/// index is *not* caught up rather than fully caught up.
///
/// A reader without this bit would read a missing `index_catchup` entry as
/// "fully caught up" and could answer an index-only query without the SSTables
/// holding the newest rows. A writer without it would change an index without
/// invalidating the catch-up position recorded for that index, leaving a stale
/// position behind. Both must refuse the table.
pub const FLAG_MEM_WAL_INDEX_CATCHUP: u64 = 128;
/// The first bit that is unknown as a feature flag
pub const FLAG_UNKNOWN: u64 = 256;

// This build only understands flags below the unknown boundary, so a bit
// allocated at or above it would be refused by the very readers meant to use it.
const _: () = assert!(FLAG_MEM_WAL_INDEX_CATCHUP < FLAG_UNKNOWN);

/// Environment variable that opts a release build into reading and writing data
/// overlay files before the feature is generally released.
pub const ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV: &str = "LANCE_ENABLE_UNSTABLE_DATA_OVERLAY_FILES";

/// Set the reader and writer feature flags in the manifest based on the contents of the manifest.
pub fn apply_feature_flags(
    manifest: &mut Manifest,
    enable_stable_row_id: bool,
    disable_transaction_file: bool,
) -> Result<()> {
    // Carried across the reset. This bit is not derivable from the manifest --
    // it depends on the `__lance_mem_wal` index details, which a manifest only
    // points at -- and this function runs twice per commit: once in
    // `build_manifest` and again in `write_manifest_file`. Dropping it here
    // would clear it immediately before the write, so an activated table would
    // report success and stay legacy.
    //
    // Only a consistent state carries: one bit set is neither mode, and
    // `inherit_mem_wal_index_catchup` refuses it at the boundary where a
    // manifest is derived from another. Reaching here half-set means the
    // manifest was already written that way, so leave it for the reader check
    // rather than silently completing it.
    let mem_wal_index_catchup = if manifest.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0
        && manifest.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0
    {
        FLAG_MEM_WAL_INDEX_CATCHUP
    } else {
        0
    };

    // Capture before the reset below. The flag is not derivable from fragment
    // content alone (see the migration path in `migrate_to_stable_row_ids`),
    // so it must be carried across the reset explicitly, just like the MemWAL
    // index-catchup bit above.
    let had_stable_row_ids = manifest.reader_feature_flags & FLAG_STABLE_ROW_IDS != 0
        && manifest.writer_feature_flags & FLAG_STABLE_ROW_IDS != 0;

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

    // Set stable row IDs flag only when explicitly requested or when it was
    // already set on the manifest before the reset above. The carry-through
    // (`had_stable_row_ids`) is a safety net: subsequent commits on an already-
    // migrated table inherit `use_stable_row_ids = true` from the dataset, which
    // sets `enable_stable_row_id`; the carry catches any path that forgets to
    // propagate the flag and prevents accidentally downgrading the table.
    if enable_stable_row_id || had_stable_row_ids {
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

    // Overlay files change cell values on read, so a reader that ignores them
    // would return stale base values. Both readers and writers must understand
    // them.
    let has_overlays = manifest
        .fragments
        .iter()
        .any(|frag| !frag.overlays.is_empty());
    if has_overlays {
        manifest.reader_feature_flags |= FLAG_UNSTABLE_DATA_OVERLAY_FILES;
        manifest.writer_feature_flags |= FLAG_UNSTABLE_DATA_OVERLAY_FILES;
    }

    if disable_transaction_file {
        manifest.writer_feature_flags |= FLAG_DISABLE_TRANSACTION_FILE;
    }

    manifest.reader_feature_flags |= mem_wal_index_catchup;
    manifest.writer_feature_flags |= mem_wal_index_catchup;

    Ok(())
}

/// Carry [`FLAG_MEM_WAL_INDEX_CATCHUP`] from the manifest a new one is derived
/// from.
///
/// [`apply_feature_flags`] carries this bit across its own reset, but it only
/// ever sees one manifest. It cannot help where a *new* manifest is derived from
/// an existing one -- `Manifest::new_from_previous` and `shallow_clone` both
/// zero the feature words -- because the destination starts with nothing to
/// carry. That transition is this function's job.
///
/// A half-set state is refused rather than normalized: one bit set means a
/// legacy reader or a legacy writer is still permitted, which is neither mode.
pub fn inherit_mem_wal_index_catchup(destination: &mut Manifest, source: &Manifest) -> Result<()> {
    let reader = source.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0;
    let writer = source.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0;
    match (reader, writer) {
        (false, false) => Ok(()),
        (true, true) => {
            destination.reader_feature_flags |= FLAG_MEM_WAL_INDEX_CATCHUP;
            destination.writer_feature_flags |= FLAG_MEM_WAL_INDEX_CATCHUP;
            Ok(())
        }
        _ => Err(Error::invalid_input(
            "Manifest has only one of the MemWAL index-catchup reader and writer \
             feature bits set, so its catch-up semantics are undefined",
        )),
    }
}

/// Whether this build understands data overlay files: always in debug builds,
/// and in release builds only when [`ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV`] is set.
fn data_overlay_files_enabled() -> bool {
    cfg!(debug_assertions) || std::env::var_os(ENABLE_UNSTABLE_DATA_OVERLAY_FILES_ENV).is_some()
}

/// Clear `flag` from `flags` when its gating feature is not enabled in this
/// build; leave it set otherwise. One call per unstable flag, so support for
/// several unstable features chains cleanly.
fn mark_supported(flags: &mut u64, flag: u64, feature_enabled: bool) {
    if !feature_enabled {
        *flags &= !flag;
    }
}

/// The feature-flag bits this build understands, given whether overlay support
/// is enabled. Split out from [`supported_flags`] so the policy is testable
/// without toggling the build profile or environment.
fn supported_flags_when(overlay_enabled: bool) -> u64 {
    let mut supported = FLAG_UNKNOWN - 1;
    mark_supported(
        &mut supported,
        FLAG_UNSTABLE_DATA_OVERLAY_FILES,
        overlay_enabled,
    );
    supported
}

fn supported_flags() -> u64 {
    supported_flags_when(data_overlay_files_enabled())
}

pub fn can_read_dataset(reader_flags: u64) -> bool {
    reader_flags & !supported_flags() == 0
}

pub fn can_write_dataset(writer_flags: u64) -> bool {
    writer_flags & !supported_flags() == 0
}

pub fn has_deprecated_v2_feature_flag(writer_flags: u64) -> bool {
    writer_flags & FLAG_USE_V2_FORMAT_DEPRECATED != 0
}

/// Refuse a manifest whose MemWAL index-catchup bits disagree.
///
/// One word set and the other not is neither mode: it would let a legacy reader
/// or a legacy writer through on a table where the other half is enforcing. The
/// commit path refuses to *produce* this, so seeing it on read means the
/// manifest was written by something that did not.
pub fn validate_mem_wal_index_catchup_flags(manifest: &Manifest) -> Result<()> {
    let reader = manifest.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0;
    let writer = manifest.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP != 0;
    if reader != writer {
        return Err(Error::invalid_input(
            "Manifest has only one of the MemWAL index-catchup reader and writer \
             feature bits set, so its catch-up semantics are undefined",
        ));
    }
    Ok(())
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
        assert!(can_read_dataset(super::FLAG_BASE_PATHS));
        assert!(can_read_dataset(super::FLAG_DISABLE_TRANSACTION_FILE));
        // Overlay support is gated on the build profile / env opt-in, so the
        // flag is readable exactly when overlays are enabled (see
        // test_data_overlay_flag_release_gating for the full policy).
        assert_eq!(
            can_read_dataset(super::FLAG_UNSTABLE_DATA_OVERLAY_FILES),
            data_overlay_files_enabled()
        );
        assert!(can_read_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED
        ));
        assert!(!can_read_dataset(super::FLAG_UNKNOWN));
    }

    #[test]
    fn test_data_overlay_flag_release_gating() {
        // Release default (overlays disabled): the overlay flag is treated as
        // unknown so the dataset is refused, while other known flags still pass.
        let supported = supported_flags_when(false);
        assert_eq!(supported & FLAG_UNSTABLE_DATA_OVERLAY_FILES, 0);
        assert_eq!(FLAG_DELETION_FILES & !supported, 0);
        assert_ne!(FLAG_UNSTABLE_DATA_OVERLAY_FILES & !supported, 0);
        // Enabled (debug or env opt-in): the overlay flag is understood.
        let supported = supported_flags_when(true);
        assert_eq!(FLAG_UNSTABLE_DATA_OVERLAY_FILES & !supported, 0);
    }

    #[test]
    fn test_apply_feature_flags_sets_overlay_flag() {
        use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
        use crate::format::{DataFile, DataStorageFormat, Fragment};
        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema;
        use roaring::RoaringBitmap;
        use std::collections::HashMap;
        use std::sync::Arc;

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "id",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let mut fragment = Fragment::new(0);
        fragment.overlays = vec![DataOverlayFile {
            data_file: DataFile::new_legacy_from_fields("o.lance", vec![0], None),
            coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
            committed_version: 1,
        }];
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![fragment]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        apply_feature_flags(&mut manifest, false, false).unwrap();
        assert_ne!(
            manifest.reader_feature_flags & FLAG_UNSTABLE_DATA_OVERLAY_FILES,
            0
        );
        assert_ne!(
            manifest.writer_feature_flags & FLAG_UNSTABLE_DATA_OVERLAY_FILES,
            0
        );
    }

    #[test]
    fn test_write_check() {
        assert!(can_write_dataset(0));
        assert!(can_write_dataset(super::FLAG_DELETION_FILES));
        assert!(can_write_dataset(super::FLAG_STABLE_ROW_IDS));
        assert!(can_write_dataset(super::FLAG_USE_V2_FORMAT_DEPRECATED));
        assert!(can_write_dataset(super::FLAG_TABLE_CONFIG));
        assert!(can_write_dataset(super::FLAG_BASE_PATHS));
        assert!(can_write_dataset(super::FLAG_DISABLE_TRANSACTION_FILE));
        // Overlay support is gated on the build profile / env opt-in, so the
        // flag is writable exactly when overlays are enabled (see
        // test_data_overlay_flag_release_gating for the full policy).
        assert_eq!(
            can_write_dataset(super::FLAG_UNSTABLE_DATA_OVERLAY_FILES),
            data_overlay_files_enabled()
        );
        assert!(can_write_dataset(
            super::FLAG_DELETION_FILES
                | super::FLAG_STABLE_ROW_IDS
                | super::FLAG_USE_V2_FORMAT_DEPRECATED
                | super::FLAG_TABLE_CONFIG
                | super::FLAG_BASE_PATHS
        ));
        assert!(!can_write_dataset(super::FLAG_UNKNOWN));
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

    /// The MemWAL bit depends on the `__lance_mem_wal` index details, which a
    /// manifest cannot see — it holds only a byte offset to its index section.
    /// So an unrelated recomputation must preserve the bit rather than derive
    /// it, or a later transaction would silently downgrade the table to legacy
    /// semantics and a reader would treat missing coverage as complete.
    #[test]
    fn inheriting_carries_the_mem_wal_bit_from_the_source() {
        let mut source = empty_manifest();
        source.reader_feature_flags = FLAG_MEM_WAL_INDEX_CATCHUP;
        source.writer_feature_flags = FLAG_MEM_WAL_INDEX_CATCHUP;
        // What `Manifest::new_from_previous` hands us: both words zeroed.
        let mut destination = empty_manifest();

        inherit_mem_wal_index_catchup(&mut destination, &source).unwrap();

        assert_ne!(
            destination.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP,
            0
        );
        assert_ne!(
            destination.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP,
            0
        );
    }

    #[test]
    fn inheriting_refuses_a_half_set_source() {
        for (reader, writer) in [
            (FLAG_MEM_WAL_INDEX_CATCHUP, 0),
            (0, FLAG_MEM_WAL_INDEX_CATCHUP),
        ] {
            let mut source = empty_manifest();
            source.reader_feature_flags = reader;
            source.writer_feature_flags = writer;
            let mut destination = empty_manifest();

            let err = inherit_mem_wal_index_catchup(&mut destination, &source).unwrap_err();

            assert!(err.to_string().contains("only one of"), "{err}");
        }
    }

    #[test]
    fn apply_feature_flags_carries_the_mem_wal_bit_across_its_reset() {
        // It runs twice per commit -- `build_manifest` and `write_manifest_file`
        // -- so dropping the bit here would clear it immediately before the
        // write, and an activated table would report success and stay legacy.
        let mut manifest = empty_manifest();
        manifest.reader_feature_flags = FLAG_MEM_WAL_INDEX_CATCHUP;
        manifest.writer_feature_flags = FLAG_MEM_WAL_INDEX_CATCHUP;

        apply_feature_flags(&mut manifest, false, false).unwrap();

        assert_ne!(
            manifest.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP,
            0
        );
        assert_ne!(
            manifest.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP,
            0
        );
    }

    #[test]
    fn apply_feature_flags_drops_a_half_set_mem_wal_bit() {
        // Neither mode, so leave it for the reader check rather than completing it.
        let mut manifest = empty_manifest();
        manifest.reader_feature_flags = FLAG_MEM_WAL_INDEX_CATCHUP;

        apply_feature_flags(&mut manifest, false, false).unwrap();

        assert_eq!(
            manifest.reader_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP,
            0
        );
        assert_eq!(
            manifest.writer_feature_flags & FLAG_MEM_WAL_INDEX_CATCHUP,
            0
        );
    }

    fn empty_manifest() -> Manifest {
        use crate::format::DataStorageFormat;
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema;
        use std::collections::HashMap;
        use std::sync::Arc;

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, false)]);
        Manifest::new(
            Schema::try_from(&arrow_schema).unwrap(),
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        )
    }

    /// A build that does not know the bit must refuse the table rather than
    /// continue with legacy semantics.
    #[test]
    fn the_mem_wal_bit_is_below_the_unknown_boundary() {
        assert!(can_read_dataset(FLAG_MEM_WAL_INDEX_CATCHUP));
        assert!(can_write_dataset(FLAG_MEM_WAL_INDEX_CATCHUP));
        // The next bit up is still unknown, so allocating this one did not
        // silently widen what this build claims to understand.
        assert!(!can_read_dataset(FLAG_UNKNOWN));
    }
}
