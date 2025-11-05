// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transaction definitions for updating datasets
//!
//! Prior to creating a new manifest, a transaction must be created representing
//! the changes being made to the dataset. By representing them as incremental
//! changes, we can detect whether concurrent operations are compatible with
//! one another. We can also rebuild manifests when retrying committing a
//! manifest.
//!
//! ## Conflict Resolution
//!
//! Transactions are compatible with one another if they don't conflict.
//! Currently, conflict resolution always assumes a Serializable isolation
//! level.
//!
//! Below are the compatibilities between conflicting transactions. The columns
//! represent the operation that has been applied, while the rows represent the
//! operation that is being checked for compatibility to see if it can retry.
//! ✅ indicates that the operation is compatible, while ❌ indicates that it is
//! a conflict. Some operations have additional conditions that must be met for
//! them to be compatible.
//!
//! NOTE/TODO(rmeng): DataReplacement conflict resolution is not fully implemented
//!
//! |                  | Append | Delete / Update | Overwrite/Create | Create Index | Rewrite | Merge | Project | UpdateConfig | DataReplacement |
//! |------------------|--------|-----------------|------------------|--------------|---------|-------|---------|--------------|-----------------|
//! | Append           | ✅     | ✅              | ❌                | ✅           | ✅      | ❌     | ❌      | ✅           | ✅
//! | Delete / Update  | ✅     | 1️⃣              | ❌                | ✅           | 1️⃣      | ❌     | ❌      | ✅           | ✅
//! | Overwrite/Create | ✅     | ✅              | ✅                | ✅           | ✅      | ✅     | ✅      | 2️⃣           | ✅
//! | Create index     | ✅     | ✅              | ❌                | ✅           | ✅      | ✅     | ✅      | ✅           | 3️⃣
//! | Rewrite          | ✅     | 1️⃣              | ❌                | ❌           | 1️⃣      | ❌     | ❌      | ✅           | 3️⃣
//! | Merge            | ❌     | ❌              | ❌                | ❌           | ✅      | ❌     | ❌      | ✅           | ✅
//! | Project          | ✅     | ✅              | ❌                | ❌           | ✅      | ❌     | ✅      | ✅           | ✅
//! | UpdateConfig     | ✅     | ✅              | 2️⃣                | ✅           | ✅      | ✅     | ✅      | 2️⃣           | ✅
//! | DataReplacement  | ✅     | ✅              | ❌                | 3️⃣           | 1️⃣      | ✅     | 3️⃣      | ✅           | 3️⃣
//!
//! 1️⃣ Delete, update, and rewrite are compatible with each other and themselves only if
//! they affect distinct fragments. Otherwise, they conflict.
//! 2️⃣ Operations that mutate the config conflict if one of the operations upserts a key
//! that if referenced by another concurrent operation or if both operations modify the schema
//! metadata or the same field metadata.
//! 3️⃣ DataReplacement on a column without index is compatible with any operation AS LONG AS
//! the operation does not modify the region of the column being replaced.
//!

use super::ManifestWriteConfig;
use crate::dataset::transaction::UpdateMode::RewriteRows;
use crate::index::mem_wal::update_mem_wal_index_in_indices_list;
use crate::utils::temporal::timestamp_to_nanos;
use deepsize::DeepSizeOf;
use lance_core::{datatypes::Schema, Error, Result};
use lance_file::{datatypes::Fields, version::LanceFileVersion};
use lance_index::mem_wal::MemWal;
use lance_index::{frag_reuse::FRAG_REUSE_INDEX_NAME, is_system_index};
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::{apply_feature_flags, FLAG_STABLE_ROW_IDS};
use lance_table::rowids::read_row_ids;
use lance_table::{
    format::{
        pb, BasePath, DataFile, DataStorageFormat, Fragment, IndexMetadata, Manifest, RowIdMeta,
    },
    io::{
        commit::CommitHandler,
        manifest::{read_manifest, read_manifest_indexes},
    },
    rowids::{write_row_ids, RowIdSequence},
};
use object_store::path::Path;
use roaring::RoaringBitmap;
use snafu::location;
use std::cmp::Ordering;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use uuid::Uuid;

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct Transaction {
    /// The version of the table this transaction is based off of. If this is
    /// the first transaction, this should be 0.
    pub read_version: u64,
    pub uuid: String,
    pub operation: Operation,
    pub tag: Option<String>,
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
}

#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct DataReplacementGroup(pub u64, pub DataFile);

/// An entry for a map update. If value is None, the key will be removed from the map.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct UpdateMapEntry {
    /// The key of the map entry to update.
    pub key: String,
    /// The value to set for the key.
    pub value: Option<String>,
}

impl From<(String, Option<String>)> for UpdateMapEntry {
    fn from((key, value): (String, Option<String>)) -> Self {
        Self { key, value }
    }
}

impl From<(String, String)> for UpdateMapEntry {
    fn from((key, value): (String, String)) -> Self {
        Self::from((key, Some(value)))
    }
}

impl From<(&str, Option<&str>)> for UpdateMapEntry {
    fn from((key, value): (&str, Option<&str>)) -> Self {
        Self {
            key: key.to_string(),
            value: value.map(str::to_owned),
        }
    }
}

impl From<(&str, &str)> for UpdateMapEntry {
    fn from((key, value): (&str, &str)) -> Self {
        Self::from((key, Some(value)))
    }
}

/// Represents updates to a map (either incremental or replacement)
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct UpdateMap {
    pub update_entries: Vec<UpdateMapEntry>,
    /// If true, the map will be replaced entirely with the new entries.
    /// If false, the new entries will be merged with the existing map.
    pub replace: bool,
}

/// An operation on a dataset.
#[derive(Debug, Clone, DeepSizeOf)]
pub enum Operation {
    /// Adding new fragments to the dataset. The fragments contained within
    /// haven't yet been assigned a final ID.
    Append { fragments: Vec<Fragment> },
    /// Updated fragments contain those that have been modified with new deletion
    /// files. The deleted fragment IDs are those that should be removed from
    /// the manifest.
    Delete {
        updated_fragments: Vec<Fragment>,
        deleted_fragment_ids: Vec<u64>,
        predicate: String,
    },
    /// Overwrite the entire dataset with the given fragments. This is also
    /// used when initially creating a table.
    Overwrite {
        fragments: Vec<Fragment>,
        schema: Schema,
        config_upsert_values: Option<HashMap<String, String>>,
        initial_bases: Option<Vec<BasePath>>,
    },
    /// A new index has been created.
    CreateIndex {
        /// The new secondary indices,
        /// any existing indices with the same name will be replaced.
        new_indices: Vec<IndexMetadata>,
        /// The indices that have been modified.
        removed_indices: Vec<IndexMetadata>,
    },
    /// Data is rewritten but *not* modified. This is used for things like
    /// compaction or re-ordering. Contains the old fragments and the new
    /// ones that have been replaced.
    ///
    /// This operation will modify the row addresses of existing rows and
    /// so any existing index covering a rewritten fragment will need to be
    /// remapped.
    Rewrite {
        /// Groups of fragments that have been modified
        groups: Vec<RewriteGroup>,
        /// Indices that have been updated with the new row addresses
        rewritten_indices: Vec<RewrittenIndex>,
        /// The fragment reuse index to be created or updated to
        frag_reuse_index: Option<IndexMetadata>,
    },
    /// Replace data in a column in the dataset with new data. This is used for
    /// null column population where we replace an entirely null column with a
    /// new column that has data.
    ///
    /// This operation will only allow replacing files that contain the same schema
    /// e.g. if the original files contain columns A, B, C and the new files contain
    /// only columns A, B then the operation is not allowed. As we would need to split
    /// the original files into two files, one with column A, B and the other with column C.
    ///
    /// Corollary to the above: the operation will also not allow replacing files unless the
    /// affected columns all have the same datafile layout across the fragments being replaced.
    ///
    /// e.g. if fragments being replaced contain files with different schema layouts on
    /// the column being replaced, the operation is not allowed.
    /// say frag_1: [A] [B, C] and frag_2: [A, B] [C] and we are trying to replace column A
    /// with a new column A, the operation is not allowed.
    DataReplacement {
        replacements: Vec<DataReplacementGroup>,
    },
    /// Merge a new column in
    /// 'fragments' is the final fragments include all data files, the new fragments must align with old ones at rows.
    /// 'schema' is not forced to include existed columns, which means we could use Merge to drop column data
    Merge {
        fragments: Vec<Fragment>,
        schema: Schema,
    },
    /// Restore an old version of the database
    Restore { version: u64 },
    /// Reserves fragment ids for future use
    /// This can be used when row ids need to be known before a transaction
    /// has been committed.  It is used during a rewrite operation to allow
    /// indices to be remapped to the new row ids as part of the operation.
    ReserveFragments { num_fragments: u32 },

    /// Update values in the dataset.
    ///
    /// Updates are generally vertical or horizontal.
    ///
    /// A vertical update adds new rows.  In this case, the updated_fragments
    /// will only have existing rows deleted and will not have any new fields added.
    /// All new data will be contained in new_fragments.
    /// This is what is used by a merge_insert that matches the whole schema and what
    /// is used by the dataset updater.
    ///
    /// A horizontal update adds new columns.  In this case, the updated fragments
    /// may have fields removed or added.  It is even possible for a field to be tombstoned
    /// and then added back in the same update. (which is a field modification).  If any
    /// fields are modified in this way then they need to be added to the fields_modified list.
    /// This way we can correctly update the indices.
    /// This is what is used by a merge insert that does not match the whole schema.
    Update {
        /// Ids of fragments that have been moved
        removed_fragment_ids: Vec<u64>,
        /// Fragments that have been updated
        updated_fragments: Vec<Fragment>,
        /// Fragments that have been added
        new_fragments: Vec<Fragment>,
        /// The fields that have been modified
        fields_modified: Vec<u32>,
        /// The MemWAL (pre-image) that should be marked as merged after this transaction
        mem_wal_to_merge: Option<MemWal>,
        /// The fields that used to judge whether to preserve the new frag's id into
        /// the frag bitmap of the specified indices.
        fields_for_preserving_frag_bitmap: Vec<u32>,
        /// The mode of update
        update_mode: Option<UpdateMode>,
    },

    /// Project to a new schema. This only changes the schema, not the data.
    Project { schema: Schema },

    /// Update the dataset configuration.
    UpdateConfig {
        config_updates: Option<UpdateMap>,
        table_metadata_updates: Option<UpdateMap>,
        schema_metadata_updates: Option<UpdateMap>,
        field_metadata_updates: HashMap<i32, UpdateMap>,
    },
    /// Update the state of MemWALs.
    UpdateMemWalState {
        added: Vec<MemWal>,
        updated: Vec<MemWal>,
        removed: Vec<MemWal>,
    },

    /// Clone a dataset.
    Clone {
        is_shallow: bool,
        ref_name: Option<String>,
        ref_version: u64,
        ref_path: String,
        branch_name: Option<String>,
    },

    // Update base paths in the dataset (currently only supports adding new bases).
    UpdateBases {
        /// The new base paths to add to the manifest.
        new_bases: Vec<BasePath>,
    },
}

#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum UpdateMode {
    /// rows are deleted in current fragments and rewritten in new fragments.
    /// This is most optimal when the majority of columns are being rewritten
    /// or only a few rows are being updated.
    RewriteRows,

    /// within each fragment, columns are fully rewritten and inserted as new data files.
    /// Old versions of columns are tombstoned. This is most optimal when most rows are affected
    /// but a small subset of columns are affected.
    RewriteColumns,
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Append { .. } => write!(f, "Append"),
            Self::Delete { .. } => write!(f, "Delete"),
            Self::Overwrite { .. } => write!(f, "Overwrite"),
            Self::CreateIndex { .. } => write!(f, "CreateIndex"),
            Self::Rewrite { .. } => write!(f, "Rewrite"),
            Self::Merge { .. } => write!(f, "Merge"),
            Self::Restore { .. } => write!(f, "Restore"),
            Self::ReserveFragments { .. } => write!(f, "ReserveFragments"),
            Self::Update { .. } => write!(f, "Update"),
            Self::Project { .. } => write!(f, "Project"),
            Self::UpdateConfig { .. } => write!(f, "UpdateConfig"),
            Self::DataReplacement { .. } => write!(f, "DataReplacement"),
            Self::Clone { .. } => write!(f, "Clone"),
            Self::UpdateMemWalState { .. } => write!(f, "UpdateMemWalState"),
            Self::UpdateBases { .. } => write!(f, "UpdateBases"),
        }
    }
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        // Many of the operations contain `Vec<T>` where the order of the
        // elements don't matter. So we need to compare them in a way that
        // ignores the order of the elements.
        // TODO: we can make it so the vecs are always constructed in order.
        // Then we can use `==` instead of `compare_vec`.
        fn compare_vec<T: PartialEq>(a: &[T], b: &[T]) -> bool {
            a.len() == b.len() && a.iter().all(|f| b.contains(f))
        }
        match (self, other) {
            (Self::Append { fragments: a }, Self::Append { fragments: b }) => compare_vec(a, b),
            (
                Self::Clone {
                    is_shallow: a_is_shallow,
                    ref_name: a_ref_name,
                    ref_version: a_ref_version,
                    ref_path: a_source_path,
                    branch_name: a_branch_name,
                },
                Self::Clone {
                    is_shallow: b_is_shallow,
                    ref_name: b_ref_name,
                    ref_version: b_ref_version,
                    ref_path: b_source_path,
                    branch_name: b_branch_name,
                },
            ) => {
                a_is_shallow == b_is_shallow
                    && a_ref_name == b_ref_name
                    && a_ref_version == b_ref_version
                    && a_source_path == b_source_path
                    && a_branch_name == b_branch_name
            }
            (
                Self::Delete {
                    updated_fragments: a_updated,
                    deleted_fragment_ids: a_deleted,
                    predicate: a_predicate,
                },
                Self::Delete {
                    updated_fragments: b_updated,
                    deleted_fragment_ids: b_deleted,
                    predicate: b_predicate,
                },
            ) => {
                compare_vec(a_updated, b_updated)
                    && compare_vec(a_deleted, b_deleted)
                    && a_predicate == b_predicate
            }
            (
                Self::Overwrite {
                    fragments: a_fragments,
                    schema: a_schema,
                    config_upsert_values: a_config,
                    initial_bases: a_initial,
                },
                Self::Overwrite {
                    fragments: b_fragments,
                    schema: b_schema,
                    config_upsert_values: b_config,
                    initial_bases: b_initial,
                },
            ) => {
                compare_vec(a_fragments, b_fragments)
                    && a_schema == b_schema
                    && a_config == b_config
                    && a_initial == b_initial
            }
            (
                Self::CreateIndex {
                    new_indices: a_new,
                    removed_indices: a_removed,
                },
                Self::CreateIndex {
                    new_indices: b_new,
                    removed_indices: b_removed,
                },
            ) => compare_vec(a_new, b_new) && compare_vec(a_removed, b_removed),
            (
                Self::Rewrite {
                    groups: a_groups,
                    rewritten_indices: a_indices,
                    frag_reuse_index: a_frag_reuse_index,
                },
                Self::Rewrite {
                    groups: b_groups,
                    rewritten_indices: b_indices,
                    frag_reuse_index: b_frag_reuse_index,
                },
            ) => {
                compare_vec(a_groups, b_groups)
                    && compare_vec(a_indices, b_indices)
                    && a_frag_reuse_index == b_frag_reuse_index
            }
            (
                Self::Merge {
                    fragments: a_fragments,
                    schema: a_schema,
                },
                Self::Merge {
                    fragments: b_fragments,
                    schema: b_schema,
                },
            ) => compare_vec(a_fragments, b_fragments) && a_schema == b_schema,
            (Self::Restore { version: a }, Self::Restore { version: b }) => a == b,
            (
                Self::ReserveFragments { num_fragments: a },
                Self::ReserveFragments { num_fragments: b },
            ) => a == b,
            (
                Self::Update {
                    removed_fragment_ids: a_removed,
                    updated_fragments: a_updated,
                    new_fragments: a_new,
                    fields_modified: a_fields,
                    mem_wal_to_merge: a_mem_wal_to_merge,
                    fields_for_preserving_frag_bitmap: a_fields_for_preserving_frag_bitmap,
                    update_mode: a_update_mode,
                },
                Self::Update {
                    removed_fragment_ids: b_removed,
                    updated_fragments: b_updated,
                    new_fragments: b_new,
                    fields_modified: b_fields,
                    mem_wal_to_merge: b_mem_wal_to_merge,
                    fields_for_preserving_frag_bitmap: b_fields_for_preserving_frag_bitmap,
                    update_mode: b_update_mode,
                },
            ) => {
                compare_vec(a_removed, b_removed)
                    && compare_vec(a_updated, b_updated)
                    && compare_vec(a_new, b_new)
                    && compare_vec(a_fields, b_fields)
                    && a_mem_wal_to_merge == b_mem_wal_to_merge
                    && compare_vec(
                        a_fields_for_preserving_frag_bitmap,
                        b_fields_for_preserving_frag_bitmap,
                    )
                    && a_update_mode == b_update_mode
            }
            (Self::Project { schema: a }, Self::Project { schema: b }) => a == b,
            (
                Self::UpdateConfig {
                    config_updates: a_config,
                    table_metadata_updates: a_table_metadata,
                    schema_metadata_updates: a_schema,
                    field_metadata_updates: a_field,
                },
                Self::UpdateConfig {
                    config_updates: b_config,
                    table_metadata_updates: b_table_metadata,
                    schema_metadata_updates: b_schema,
                    field_metadata_updates: b_field,
                },
            ) => {
                a_config == b_config
                    && a_table_metadata == b_table_metadata
                    && a_schema == b_schema
                    && a_field == b_field
            }
            (
                Self::DataReplacement { replacements: a },
                Self::DataReplacement { replacements: b },
            ) => a.len() == b.len() && a.iter().all(|r| b.contains(r)),
            // Handle all remaining combinations.
            // We spell out all combinations explicitly to prevent
            // us accidentally handling a new case in the wrong way.
            (Self::Append { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Append { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Delete { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Overwrite { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::CreateIndex { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Rewrite { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Merge { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Restore { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::ReserveFragments { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Update { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Project { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::UpdateConfig { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::DataReplacement { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::UpdateMemWalState { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (
                Self::UpdateMemWalState {
                    added: a_added,
                    updated: a_updated,
                    removed: a_removed,
                },
                Self::UpdateMemWalState {
                    added: b_added,
                    updated: b_updated,
                    removed: b_removed,
                },
            ) => {
                compare_vec(a_added, b_added)
                    && compare_vec(a_updated, b_updated)
                    && compare_vec(a_removed, b_removed)
            }
            (Self::Clone { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::UpdateBases { new_bases: a }, Self::UpdateBases { new_bases: b }) => {
                compare_vec(a, b)
            }

            (Self::UpdateBases { .. }, Self::Append { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Delete { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Overwrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::CreateIndex { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Rewrite { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Merge { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Restore { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::ReserveFragments { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Update { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Project { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::UpdateConfig { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::DataReplacement { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::UpdateMemWalState { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateBases { .. }, Self::Clone { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }

            (Self::Append { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Delete { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Overwrite { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::CreateIndex { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Rewrite { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Merge { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Restore { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::ReserveFragments { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Update { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Project { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateConfig { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::DataReplacement { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::UpdateMemWalState { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
            (Self::Clone { .. }, Self::UpdateBases { .. }) => {
                std::mem::discriminant(self) == std::mem::discriminant(other)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RewrittenIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub new_index_details: prost_types::Any,
    pub new_index_version: u32,
}

impl DeepSizeOf for RewrittenIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.new_index_details
            .type_url
            .deep_size_of_children(context)
            + self.new_index_details.value.deep_size_of_children(context)
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct RewriteGroup {
    pub old_fragments: Vec<Fragment>,
    pub new_fragments: Vec<Fragment>,
}

impl PartialEq for RewriteGroup {
    fn eq(&self, other: &Self) -> bool {
        fn compare_vec<T: PartialEq>(a: &[T], b: &[T]) -> bool {
            a.len() == b.len() && a.iter().all(|f| b.contains(f))
        }
        compare_vec(&self.old_fragments, &other.old_fragments)
            && compare_vec(&self.new_fragments, &other.new_fragments)
    }
}

impl Operation {
    /// Returns the config keys that have been upserted by this operation.
    fn get_upsert_config_keys(&self) -> Vec<String> {
        match self {
            Self::Overwrite {
                config_upsert_values: Some(upsert_values),
                ..
            } => {
                let vec: Vec<String> = upsert_values.keys().cloned().collect();
                vec
            }
            Self::UpdateConfig {
                config_updates: Some(config_updates),
                ..
            } => config_updates
                .update_entries
                .iter()
                .filter_map(|entry| {
                    if entry.value.is_some() {
                        Some(entry.key.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::<String>::new(),
        }
    }

    /// Returns the config keys that have been deleted by this operation.
    fn get_delete_config_keys(&self) -> Vec<String> {
        match self {
            Self::UpdateConfig {
                config_updates: Some(config_updates),
                ..
            } => config_updates
                .update_entries
                .iter()
                .filter_map(|entry| {
                    if entry.value.is_none() {
                        Some(entry.key.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::<String>::new(),
        }
    }

    pub(crate) fn modifies_same_metadata(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::UpdateConfig {
                    schema_metadata_updates,
                    field_metadata_updates,
                    ..
                },
                Self::UpdateConfig {
                    schema_metadata_updates: other_schema_metadata,
                    field_metadata_updates: other_field_metadata,
                    ..
                },
            ) => {
                if schema_metadata_updates.is_some() && other_schema_metadata.is_some() {
                    return true;
                }
                if !field_metadata_updates.is_empty() && !other_field_metadata.is_empty() {
                    for field in field_metadata_updates.keys() {
                        if other_field_metadata.contains_key(field) {
                            return true;
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Check whether another operation upserts a key that is referenced by another operation
    pub(crate) fn upsert_key_conflict(&self, other: &Self) -> bool {
        let self_upsert_keys = self.get_upsert_config_keys();
        let other_upsert_keys = other.get_upsert_config_keys();

        let self_delete_keys = self.get_delete_config_keys();
        let other_delete_keys = other.get_delete_config_keys();

        self_upsert_keys
            .iter()
            .any(|x| other_upsert_keys.contains(x) || other_delete_keys.contains(x))
            || other_upsert_keys
                .iter()
                .any(|x| self_upsert_keys.contains(x) || self_delete_keys.contains(x))
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Append { .. } => "Append",
            Self::Delete { .. } => "Delete",
            Self::Overwrite { .. } => "Overwrite",
            Self::CreateIndex { .. } => "CreateIndex",
            Self::Rewrite { .. } => "Rewrite",
            Self::Merge { .. } => "Merge",
            Self::ReserveFragments { .. } => "ReserveFragments",
            Self::Restore { .. } => "Restore",
            Self::Update { .. } => "Update",
            Self::Project { .. } => "Project",
            Self::UpdateConfig { .. } => "UpdateConfig",
            Self::DataReplacement { .. } => "DataReplacement",
            Self::UpdateMemWalState { .. } => "UpdateMemWalState",
            Self::Clone { .. } => "Clone",
            Self::UpdateBases { .. } => "UpdateBases",
        }
    }
}

/// Helper function to apply UpdateMap changes to a HashMap<String, String>
fn apply_update_map(
    target: &mut std::collections::HashMap<String, String>,
    update_map: &UpdateMap,
) {
    if update_map.replace {
        // Full replacement - clear existing and replace with new entries that have values
        target.clear();
        for entry in &update_map.update_entries {
            if let Some(value) = &entry.value {
                target.insert(entry.key.clone(), value.clone());
            }
        }
    } else {
        // Incremental update - merge entries
        for entry in &update_map.update_entries {
            if let Some(value) = &entry.value {
                target.insert(entry.key.clone(), value.clone());
            } else {
                target.remove(&entry.key);
            }
        }
    }
}

/// Helper function to translate old-style config updates to new UpdateMap format
pub fn translate_config_updates(
    upsert_values: &std::collections::HashMap<String, String>,
    delete_keys: &[String],
) -> UpdateMap {
    let mut update_entries = Vec::new();

    // Add upsert entries (with values)
    for (key, value) in upsert_values {
        update_entries.push(UpdateMapEntry {
            key: key.clone(),
            value: Some(value.clone()),
        });
    }

    // Add delete entries (without values)
    for key in delete_keys {
        update_entries.push(UpdateMapEntry {
            key: key.clone(),
            value: None,
        });
    }

    UpdateMap {
        update_entries,
        replace: false, // Old style was always incremental
    }
}

/// Helper function to translate old-style schema metadata to new UpdateMap format
pub fn translate_schema_metadata_updates(
    schema_metadata: &std::collections::HashMap<String, String>,
) -> UpdateMap {
    let update_entries = schema_metadata
        .iter()
        .map(|(key, value)| UpdateMapEntry {
            key: key.clone(),
            value: Some(value.clone()),
        })
        .collect();

    UpdateMap {
        update_entries,
        replace: true, // Old style schema metadata was full replacement
    }
}

impl From<&UpdateMap> for pb::transaction::UpdateMap {
    fn from(update_map: &UpdateMap) -> Self {
        Self {
            update_entries: update_map
                .update_entries
                .iter()
                .map(|entry| pb::transaction::UpdateMapEntry {
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                })
                .collect(),
            replace: update_map.replace,
        }
    }
}

impl From<&pb::transaction::UpdateMap> for UpdateMap {
    fn from(pb_update_map: &pb::transaction::UpdateMap) -> Self {
        Self {
            update_entries: pb_update_map
                .update_entries
                .iter()
                .map(|entry| UpdateMapEntry {
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                })
                .collect(),
            replace: pb_update_map.replace,
        }
    }
}

/// Add TransactionBuilder for flexibly setting option without using `mut`
pub struct TransactionBuilder {
    read_version: u64,
    // uuid is optional for builder since it can autogenerate
    uuid: Option<String>,
    operation: Operation,
    tag: Option<String>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
}

impl TransactionBuilder {
    pub fn new(read_version: u64, operation: Operation) -> Self {
        Self {
            read_version,
            uuid: None,
            operation,
            tag: None,
            transaction_properties: None,
        }
    }

    pub fn uuid(mut self, uuid: String) -> Self {
        self.uuid = Some(uuid);
        self
    }

    pub fn tag(mut self, tag: Option<String>) -> Self {
        self.tag = tag;
        self
    }

    pub fn transaction_properties(
        mut self,
        transaction_properties: Option<Arc<HashMap<String, String>>>,
    ) -> Self {
        self.transaction_properties = transaction_properties;
        self
    }

    pub fn build(self) -> Transaction {
        let uuid = self
            .uuid
            .unwrap_or_else(|| Uuid::new_v4().hyphenated().to_string());
        Transaction {
            read_version: self.read_version,
            uuid,
            operation: self.operation,
            tag: self.tag,
            transaction_properties: self.transaction_properties,
        }
    }
}

impl Transaction {
    pub fn new_from_version(read_version: u64, operation: Operation) -> Self {
        TransactionBuilder::new(read_version, operation).build()
    }

    pub fn new(read_version: u64, operation: Operation, tag: Option<String>) -> Self {
        TransactionBuilder::new(read_version, operation)
            .tag(tag)
            .build()
    }

    fn fragments_with_ids<'a, T>(
        new_fragments: T,
        fragment_id: &'a mut u64,
    ) -> impl Iterator<Item = Fragment> + 'a
    where
        T: IntoIterator<Item = Fragment> + 'a,
    {
        new_fragments.into_iter().map(move |mut f| {
            if f.id == 0 {
                f.id = *fragment_id;
                *fragment_id += 1;
            }
            f
        })
    }

    fn data_storage_format_from_files(
        fragments: &[Fragment],
        user_requested: Option<LanceFileVersion>,
    ) -> Result<DataStorageFormat> {
        if let Some(file_version) = Fragment::try_infer_version(fragments)? {
            // Ensure user-requested matches data files
            if let Some(user_requested) = user_requested {
                if user_requested != file_version {
                    return Err(Error::invalid_input(
                        format!("User requested data storage version ({}) does not match version in data files ({})", user_requested, file_version),
                        location!(),
                    ));
                }
            }
            Ok(DataStorageFormat::new(file_version))
        } else {
            // If no files use user-requested or default
            Ok(user_requested
                .map(DataStorageFormat::new)
                .unwrap_or_default())
        }
    }

    pub(crate) async fn restore_old_manifest(
        object_store: &ObjectStore,
        commit_handler: &dyn CommitHandler,
        base_path: &Path,
        version: u64,
        config: &ManifestWriteConfig,
        tx_path: &str,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let location = commit_handler
            .resolve_version_location(base_path, version, &object_store.inner)
            .await?;
        let mut manifest = read_manifest(object_store, &location.path, location.size).await?;
        manifest.set_timestamp(timestamp_to_nanos(config.timestamp));
        manifest.transaction_file = Some(tx_path.to_string());
        let indices = read_manifest_indexes(object_store, &location, &manifest).await?;
        Ok((manifest, indices))
    }

    /// Create a new manifest from the current manifest and the transaction.
    ///
    /// `current_manifest` should only be None if the dataset does not yet exist.
    pub(crate) fn build_manifest(
        &self,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestWriteConfig,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        if config.use_stable_row_ids
            && current_manifest
                .map(|m| !m.uses_stable_row_ids())
                .unwrap_or_default()
        {
            return Err(Error::NotSupported {
                source: "Cannot enable stable row ids on existing dataset".into(),
                location: location!(),
            });
        }
        let mut reference_paths = match current_manifest {
            Some(m) => m.base_paths.clone(),
            None => HashMap::new(),
        };

        if let Operation::Overwrite {
            initial_bases: Some(initial_bases),
            ..
        } = &self.operation
        {
            if current_manifest.is_none() {
                // CREATE mode: registering base paths
                // Base IDs should have been assigned during write operation
                // Validate uniqueness and insert them into the manifest
                for base_path in initial_bases.iter() {
                    if reference_paths.contains_key(&base_path.id) {
                        return Err(Error::invalid_input(
                            format!(
                                "Duplicate base path ID {} detected. Base path IDs must be unique.",
                                base_path.id
                            ),
                            location!(),
                        ));
                    }
                    reference_paths.insert(base_path.id, base_path.clone());
                }
            } else {
                // OVERWRITE mode with initial_bases should have been rejected by validation
                // This branch should never be reached
                return Err(Error::invalid_input(
                    "OVERWRITE mode cannot register new bases. This should have been caught by validation.",
                    location!(),
                ));
            }
        }

        // Get the schema and the final fragment list
        let schema = match self.operation {
            Operation::Overwrite { ref schema, .. } => schema.clone(),
            Operation::Merge { ref schema, .. } => schema.clone(),
            Operation::Project { ref schema, .. } => schema.clone(),
            _ => {
                if let Some(current_manifest) = current_manifest {
                    current_manifest.schema.clone()
                } else {
                    return Err(Error::Internal {
                        message: "Cannot create a new dataset without a schema".to_string(),
                        location: location!(),
                    });
                }
            }
        };

        let mut fragment_id = if matches!(self.operation, Operation::Overwrite { .. }) {
            0
        } else {
            current_manifest
                .and_then(|m| m.max_fragment_id())
                .map(|id| id + 1)
                .unwrap_or(0)
        };
        let mut final_fragments = Vec::new();
        let mut final_indices = current_indices;

        let mut next_row_id = {
            // Only use row ids if the feature flag is set already or
            match (current_manifest, config.use_stable_row_ids) {
                (Some(manifest), _) if manifest.reader_feature_flags & FLAG_STABLE_ROW_IDS != 0 => {
                    Some(manifest.next_row_id)
                }
                (None, true) => Some(0),
                (_, false) => None,
                (Some(_), true) => {
                    return Err(Error::NotSupported {
                        source: "Cannot enable stable row ids on existing dataset".into(),
                        location: location!(),
                    });
                }
            }
        };

        let maybe_existing_fragments =
            current_manifest
                .map(|m| m.fragments.as_ref())
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "No current manifest was provided while building manifest for operation {}",
                        self.operation.name()
                    ),
                    location: location!(),
                });

        match &self.operation {
            Operation::Clone { .. } => {
                return Err(Error::Internal {
                    message: "Clone operation should not enter build_manifest.".to_string(),
                    location: location!(),
                })
            }
            Operation::Append { ref fragments } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let mut new_fragments =
                    Self::fragments_with_ids(fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Add version metadata for all new fragments
                    let new_version = current_manifest.map(|m| m.version + 1).unwrap_or(1);
                    for fragment in new_fragments.iter_mut() {
                        let version_meta =
                            lance_table::rowids::version::build_version_meta(fragment, new_version);
                        fragment.last_updated_at_version_meta = version_meta.clone();
                        fragment.created_at_version_meta = version_meta;
                    }
                }
                final_fragments.extend(new_fragments);
            }
            Operation::Delete {
                ref updated_fragments,
                ref deleted_fragment_ids,
                ..
            } => {
                // Remove the deleted fragments
                final_fragments.extend(maybe_existing_fragments?.clone());
                final_fragments.retain(|f| !deleted_fragment_ids.contains(&f.id));
                final_fragments.iter_mut().for_each(|f| {
                    for updated in updated_fragments {
                        if updated.id == f.id {
                            *f = updated.clone();
                        }
                    }
                });
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                mem_wal_to_merge,
                fields_for_preserving_frag_bitmap,
                update_mode,
            } => {
                // Extract existing fragments once for reuse
                let existing_fragments = maybe_existing_fragments?;

                // Apply updates to existing fragments
                let updated_frags: Vec<Fragment> = existing_fragments
                    .iter()
                    .filter_map(|f| {
                        if removed_fragment_ids.contains(&f.id) {
                            return None;
                        }
                        if let Some(updated) = updated_fragments.iter().find(|uf| uf.id == f.id) {
                            Some(updated.clone())
                        } else {
                            Some(f.clone())
                        }
                    })
                    .collect();

                // Update version metadata for updated fragments if stable row IDs are enabled
                // Note: We don't update version metadata for fragments with deletion vectors
                // because the version sequences are indexed by physical row position, not logical position.
                // Version metadata for deleted rows will be filtered out during scan using the deletion vector.
                if next_row_id.is_some() {
                    // Version metadata will be properly set during compaction when deletions are materialized
                }

                final_fragments.extend(updated_frags);

                // If we updated any fields, remove those fragments from indices covering those fields
                Self::prune_updated_fields_from_indices(
                    &mut final_indices,
                    updated_fragments,
                    fields_modified,
                );

                let mut new_fragments =
                    Self::fragments_with_ids(new_fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();

                // Assign row IDs to any fragments that don't have them yet
                // (e.g., inserted rows from merge_insert operations)
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                }

                // Set version metadata for newly created fragments (updated rows)
                // Preserve created_at from original fragments, set last_updated to new version
                if next_row_id.is_some() {
                    let new_version = current_manifest.map(|m| m.version + 1).unwrap_or(1);

                    // Build a map of original fragment ID -> original fragment for lookup
                    let original_frags_map: std::collections::HashMap<u64, &Fragment> =
                        existing_fragments.iter().map(|f| (f.id, f)).collect();

                    for fragment in new_fragments.iter_mut() {
                        // For update operations with RewriteRows mode:
                        // - Rows are deleted from old fragments and rewritten to new fragments
                        // - last_updated_at should be the current version (when update happened)
                        // - created_at should be preserved from the original fragment

                        // Read row IDs from this fragment to find original fragments
                        let row_ids = if let Some(row_id_meta) = &fragment.row_id_meta {
                            match row_id_meta {
                                lance_table::format::RowIdMeta::Inline(data) => {
                                    lance_table::rowids::read_row_ids(data).ok()
                                }
                                lance_table::format::RowIdMeta::External(_) => None,
                            }
                        } else {
                            None
                        };

                        if let Some(row_ids) = row_ids {
                            // Extract created_at version for each row from original fragments
                            let physical_rows = fragment.physical_rows.unwrap_or(0);
                            let mut created_at_versions = Vec::with_capacity(physical_rows);

                            for row_id in row_ids.iter() {
                                // Row ID format: upper 32 bits = fragment ID, lower 32 bits = row offset
                                let orig_frag_id = row_id >> 32;
                                let row_offset = (row_id & 0xFFFFFFFF) as usize;

                                // Look up the original fragment
                                if let Some(orig_frag) = original_frags_map.get(&orig_frag_id) {
                                    // Get created_at version from original fragment's metadata
                                    let created_version = if let Some(created_meta) =
                                        &orig_frag.created_at_version_meta
                                    {
                                        // Load and index into the version sequence
                                        match created_meta.load_sequence() {
                                            Ok(seq) => {
                                                let versions: Vec<u64> = seq.versions().collect();
                                                versions.get(row_offset).copied().unwrap_or(1)
                                            }
                                            Err(_e) => {
                                                1 // Default to version 1 on error
                                            }
                                        }
                                    } else {
                                        // No metadata on original fragment, default to version 1
                                        1
                                    };
                                    created_at_versions.push(created_version);
                                } else {
                                    // Original fragment not found, default to version 1
                                    created_at_versions.push(1);
                                }
                            }

                            // Build version metadata from the collected versions
                            // Compress into runs: consecutive identical versions become one run
                            let mut runs = Vec::new();
                            if !created_at_versions.is_empty() {
                                let mut current_version = created_at_versions[0];
                                let mut run_start = 0u64;

                                for (i, &version) in created_at_versions.iter().enumerate().skip(1)
                                {
                                    if version != current_version {
                                        // End current run, start new one
                                        runs.push(lance_table::format::RowDatasetVersionRun {
                                            span: lance_table::rowids::segment::U64Segment::Range(
                                                run_start..i as u64,
                                            ),
                                            version: current_version,
                                        });
                                        current_version = version;
                                        run_start = i as u64;
                                    }
                                }
                                // Add final run
                                runs.push(lance_table::format::RowDatasetVersionRun {
                                    span: lance_table::rowids::segment::U64Segment::Range(
                                        run_start..created_at_versions.len() as u64,
                                    ),
                                    version: current_version,
                                });
                            }

                            let created_at_seq =
                                lance_table::format::RowDatasetVersionSequence { runs };
                            fragment.created_at_version_meta = Some(
                                lance_table::format::RowDatasetVersionMeta::from_sequence(
                                    &created_at_seq,
                                )
                                .map_err(|e| Error::Internal {
                                    message: format!(
                                        "Failed to create created_at version metadata: {}",
                                        e
                                    ),
                                    location: location!(),
                                })?,
                            );

                            // Set last_updated_at to the new version for all rows
                            let last_updated_meta =
                                lance_table::rowids::version::build_version_meta(
                                    fragment,
                                    new_version,
                                );
                            fragment.last_updated_at_version_meta = last_updated_meta;
                        } else {
                            // Fallback: can't read row IDs, set both to new version
                            let version_meta = lance_table::rowids::version::build_version_meta(
                                fragment,
                                new_version,
                            );
                            fragment.last_updated_at_version_meta = version_meta.clone();
                            fragment.created_at_version_meta = version_meta;
                        }
                    }
                }

                if config.use_stable_row_ids
                    && update_mode.is_some()
                    && *update_mode == Some(RewriteRows)
                {
                    let pure_updated_frag_ids =
                        Self::collect_pure_rewrite_row_update_frags_ids(&new_fragments)?;

                    // collect all the original frag ids that contains the updated rows
                    let original_fragment_ids: Vec<u64> = removed_fragment_ids
                        .iter()
                        .chain(updated_fragments.iter().map(|f| &f.id))
                        .copied()
                        .collect();

                    Self::register_pure_rewrite_rows_update_frags_in_indices(
                        &mut final_indices,
                        &pure_updated_frag_ids,
                        &original_fragment_ids,
                        fields_for_preserving_frag_bitmap,
                    );
                }

                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Note: Version metadata is already set above (lines 1627-1755)
                    // for Update operations, preserving created_at from original fragments.
                    // Don't overwrite it here.
                }
                // Identify fragments that were updated or newly created in this update
                let mut target_ids: HashSet<u64> = HashSet::new();
                target_ids.extend(new_fragments.iter().map(|f| f.id));
                final_fragments.extend(new_fragments);
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments);

                if let Some(mem_wal_to_merge) = mem_wal_to_merge {
                    update_mem_wal_index_in_indices_list(
                        self.read_version,
                        current_manifest.map_or(1, |m| m.version + 1),
                        &mut final_indices,
                        vec![],
                        vec![MemWal {
                            state: lance_index::mem_wal::State::Merged,
                            ..mem_wal_to_merge.clone()
                        }],
                        vec![mem_wal_to_merge.clone()],
                    )?;
                }
            }
            Operation::Overwrite { ref fragments, .. } => {
                let mut new_fragments =
                    Self::fragments_with_ids(fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Add version metadata for all new fragments
                    let new_version = current_manifest.map(|m| m.version + 1).unwrap_or(1);
                    for fragment in new_fragments.iter_mut() {
                        let version_meta =
                            lance_table::rowids::version::build_version_meta(fragment, new_version);
                        fragment.last_updated_at_version_meta = version_meta.clone();
                        fragment.created_at_version_meta = version_meta;
                    }
                }
                final_fragments.extend(new_fragments);
                final_indices = Vec::new();
            }
            Operation::Rewrite {
                ref groups,
                ref rewritten_indices,
                ref frag_reuse_index,
            } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let current_version = current_manifest.map(|m| m.version).unwrap_or_default();
                Self::handle_rewrite_fragments(
                    &mut final_fragments,
                    groups,
                    &mut fragment_id,
                    current_version,
                    next_row_id.as_ref(),
                )?;

                if next_row_id.is_some() {
                    // We can re-use indices, but need to rewrite the fragment bitmaps
                    debug_assert!(rewritten_indices.is_empty());
                    for index in final_indices.iter_mut() {
                        if let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                            *fragment_bitmap =
                                Self::recalculate_fragment_bitmap(fragment_bitmap, groups)?;
                        }
                    }
                } else {
                    Self::handle_rewrite_indices(&mut final_indices, rewritten_indices, groups)?;
                }

                if let Some(frag_reuse_index) = frag_reuse_index {
                    final_indices.retain(|idx| idx.name != frag_reuse_index.name);
                    final_indices.push(frag_reuse_index.clone());
                }
            }
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                final_indices.retain(|existing_index| {
                    !new_indices
                        .iter()
                        .any(|new_index| new_index.name == existing_index.name)
                        && !removed_indices
                            .iter()
                            .any(|old_index| old_index.uuid == existing_index.uuid)
                });
                final_indices.extend(new_indices.clone());
            }
            Operation::ReserveFragments { .. } | Operation::UpdateConfig { .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
            }
            Operation::Merge { ref fragments, .. } => {
                final_fragments.extend(fragments.clone());

                // Some fields that have indices may have been removed, so we should
                // remove those indices as well.
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
            }
            Operation::Project { .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());

                // We might have removed all fields for certain data files, so
                // we should remove the data files that are no longer relevant.
                let remaining_field_ids = schema
                    .fields_pre_order()
                    .map(|f| f.id)
                    .collect::<HashSet<_>>();
                for fragment in final_fragments.iter_mut() {
                    fragment.files.retain(|file| {
                        file.fields
                            .iter()
                            .any(|field_id| remaining_field_ids.contains(field_id))
                    });
                }

                // Some fields that have indices may have been removed, so we should
                // remove those indices as well.
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
            }
            Operation::Restore { .. } => {
                unreachable!()
            }
            Operation::DataReplacement { replacements } => {
                log::warn!("Building manifest with DataReplacement operation. This operation is not stable yet, please use with caution.");

                let (old_fragment_ids, new_datafiles): (Vec<&u64>, Vec<&DataFile>) = replacements
                    .iter()
                    .map(|DataReplacementGroup(fragment_id, new_file)| (fragment_id, new_file))
                    .unzip();

                // 1. make sure the new files all have the same fields / or empty
                // NOTE: arguably this requirement could be relaxed in the future
                // for the sake of simplicity, we require the new files to have the same fields
                if new_datafiles
                    .iter()
                    .map(|f| f.fields.clone())
                    .collect::<HashSet<_>>()
                    .len()
                    > 1
                {
                    let field_info = new_datafiles
                        .iter()
                        .enumerate()
                        .map(|(id, f)| (id, f.fields.clone()))
                        .fold("".to_string(), |acc, (id, fields)| {
                            format!("{}File {}: {:?}\n", acc, id, fields)
                        });

                    return Err(Error::invalid_input(
                        format!(
                            "All new data files must have the same fields, but found different fields:\n{field_info}"
                        ),
                        location!(),
                    ));
                }

                let existing_fragments = maybe_existing_fragments?;

                // 2. check that the fragments being modified have isomorphic layouts along the columns being replaced
                // 3. add modified fragments to final_fragments
                for (frag_id, new_file) in old_fragment_ids.iter().zip(new_datafiles) {
                    let frag = existing_fragments
                        .iter()
                        .find(|f| f.id == **frag_id)
                        .ok_or_else(|| {
                            Error::invalid_input(
                                "Fragment being replaced not found in existing fragments",
                                location!(),
                            )
                        })?;
                    let mut new_frag = frag.clone();

                    // TODO(rmeng): check new file and fragment are the same length

                    let mut columns_covered = HashSet::new();
                    for file in &mut new_frag.files {
                        if file.fields == new_file.fields
                            && file.file_major_version == new_file.file_major_version
                            && file.file_minor_version == new_file.file_minor_version
                        {
                            // assign the new file path / size to the fragment
                            file.path = new_file.path.clone();
                            file.file_size_bytes = new_file.file_size_bytes.clone();
                        }
                        columns_covered.extend(file.fields.iter());
                    }
                    // SPECIAL CASE: if the column(s) being replaced are not covered by the fragment
                    // Then it means it's a all-NULL column that is being replaced with real data
                    // just add it to the final fragments
                    if columns_covered.is_disjoint(&new_file.fields.iter().collect()) {
                        new_frag.add_file(
                            new_file.path.clone(),
                            new_file.fields.clone(),
                            new_file.column_indices.clone(),
                            &LanceFileVersion::try_from_major_minor(
                                new_file.file_major_version,
                                new_file.file_minor_version,
                            )
                            .expect("Expected valid file version"),
                            new_file.file_size_bytes.get(),
                        );
                    }

                    // Nothing changed in the current fragment, which is not expected -- error out
                    if &new_frag == frag {
                        return Err(Error::invalid_input(
                            "Expected to modify the fragment but no changes were made. This means the new data files does not align with any exiting datafiles. Please check if the schema of the new data files matches the schema of the old data files including the file major and minor versions",
                            location!(),
                        ));
                    }
                    final_fragments.push(new_frag);
                }

                let fragments_changed = old_fragment_ids
                    .iter()
                    .cloned()
                    .cloned()
                    .collect::<HashSet<_>>();

                // 4. push fragments that didn't change back to final_fragments
                let unmodified_fragments = existing_fragments
                    .iter()
                    .filter(|f| !fragments_changed.contains(&f.id))
                    .cloned()
                    .collect::<Vec<_>>();

                final_fragments.extend(unmodified_fragments);
            }
            Operation::UpdateMemWalState {
                added,
                updated,
                removed,
            } => {
                update_mem_wal_index_in_indices_list(
                    self.read_version,
                    current_manifest.map_or(1, |m| m.version + 1),
                    &mut final_indices,
                    added.clone(),
                    updated.clone(),
                    removed.clone(),
                )?;
            }
            Operation::UpdateBases { .. } => {
                // UpdateBases operation doesn't modify fragments or indices
                // Base paths are handled in the manifest creation section below
                final_fragments.extend(maybe_existing_fragments?.clone());
            }
        };

        // If a fragment was reserved then it may not belong at the end of the fragments list.
        final_fragments.sort_by_key(|frag| frag.id);

        // Clean up data files that only contain tombstoned fields
        Self::remove_tombstoned_data_files(&mut final_fragments);

        let user_requested_version = match (&config.storage_format, config.use_legacy_format) {
            (Some(storage_format), _) => Some(storage_format.lance_file_version()?),
            (None, Some(true)) => Some(LanceFileVersion::Legacy),
            (None, Some(false)) => Some(LanceFileVersion::V2_0),
            (None, None) => None,
        };

        let mut manifest = if let Some(current_manifest) = current_manifest {
            // OVERWRITE with initial_bases on existing dataset is not allowed (caught by validation)
            // So we always use new_from_previous which preserves base_paths
            let mut prev_manifest =
                Manifest::new_from_previous(current_manifest, schema, Arc::new(final_fragments));

            if let (Some(user_requested_version), Operation::Overwrite { .. }) =
                (user_requested_version, &self.operation)
            {
                // If this is an overwrite operation and the user has requested a specific version
                // then overwrite with that version.  Otherwise, if the user didn't request a specific
                // version, then overwrite with whatever version we had before.
                prev_manifest.data_storage_format = DataStorageFormat::new(user_requested_version);
            }

            prev_manifest
        } else {
            let data_storage_format =
                Self::data_storage_format_from_files(&final_fragments, user_requested_version)?;
            Manifest::new(
                schema,
                Arc::new(final_fragments),
                data_storage_format,
                reference_paths,
            )
        };

        manifest.tag.clone_from(&self.tag);

        if config.auto_set_feature_flags {
            apply_feature_flags(&mut manifest, config.use_stable_row_ids)?;
        }
        manifest.set_timestamp(timestamp_to_nanos(config.timestamp));

        manifest.update_max_fragment_id();

        match &self.operation {
            Operation::Overwrite {
                config_upsert_values: Some(tm),
                ..
            } => {
                manifest.config_mut().extend(tm.clone());
            }
            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            } => {
                if let Some(config_updates) = config_updates {
                    let mut config = manifest.config.clone();
                    apply_update_map(&mut config, config_updates);
                    manifest.config = config;
                }
                if let Some(table_metadata_updates) = table_metadata_updates {
                    let mut table_metadata = manifest.table_metadata.clone();
                    apply_update_map(&mut table_metadata, table_metadata_updates);
                    manifest.table_metadata = table_metadata;
                }
                if let Some(schema_metadata_updates) = schema_metadata_updates {
                    let mut schema_metadata = manifest.schema.metadata.clone();
                    apply_update_map(&mut schema_metadata, schema_metadata_updates);
                    manifest.schema.metadata = schema_metadata;
                }
                for (field_id, field_metadata_update) in field_metadata_updates {
                    if let Some(field) = manifest.schema.field_by_id_mut(*field_id) {
                        apply_update_map(&mut field.metadata, field_metadata_update);
                    } else {
                        return Err(Error::InvalidInput {
                            source: format!("Field with id {} does not exist", field_id).into(),
                            location: location!(),
                        });
                    }
                }
            }
            _ => {}
        }

        // Handle UpdateBases operation to update manifest base_paths
        if let Operation::UpdateBases { new_bases } = &self.operation {
            // Validate and add new base paths to the manifest
            for new_base in new_bases {
                // Check for conflicts with existing base paths
                if let Some(existing_base) = manifest
                    .base_paths
                    .values()
                    .find(|bp| bp.name == new_base.name || bp.path == new_base.path)
                {
                    return Err(Error::invalid_input(
                        format!(
                            "Conflict detected: Base path with name '{:?}' or path '{}' already exists. Existing: name='{:?}', path='{}'",
                            new_base.name, new_base.path, existing_base.name, existing_base.path
                        ),
                        location!(),
                    ));
                }

                // Assign a new ID if not already assigned
                let mut base_to_add = new_base.clone();
                if base_to_add.id == 0 {
                    let next_id = manifest
                        .base_paths
                        .keys()
                        .max()
                        .map(|&id| id + 1)
                        .unwrap_or(1);
                    base_to_add.id = next_id;
                }

                manifest.base_paths.insert(base_to_add.id, base_to_add);
            }
        }

        if let Operation::ReserveFragments { num_fragments } = self.operation {
            manifest.max_fragment_id = Some(manifest.max_fragment_id.unwrap_or(0) + num_fragments);
        }

        manifest.transaction_file = Some(transaction_file_path.to_string());

        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, final_indices))
    }

    fn register_pure_rewrite_rows_update_frags_in_indices(
        indices: &mut [IndexMetadata],
        pure_update_frag_ids: &[u64],
        original_fragment_ids: &[u64],
        fields_for_preserving_frag_bitmap: &[u32],
    ) {
        if pure_update_frag_ids.is_empty() {
            return;
        }

        let value_updated_field_set = fields_for_preserving_frag_bitmap
            .iter()
            .collect::<HashSet<_>>();

        for index in indices.iter_mut() {
            let index_covers_modified_field = index.fields.iter().any(|field_id| {
                value_updated_field_set.contains(&u32::try_from(*field_id).unwrap())
            });

            if !index_covers_modified_field {
                if let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                    // check if all the original fragments contains the updating rows are covered
                    // by the index(index fragment bitmap contains these frag ids).
                    // if not, that means not all the updating rows are indexed, so we could not
                    // index them.
                    let index_covers_all_original_fragments = original_fragment_ids
                        .iter()
                        .all(|&fragment_id| fragment_bitmap.contains(fragment_id as u32));

                    if index_covers_all_original_fragments {
                        for fragment_id in pure_update_frag_ids.iter().map(|f| *f as u32) {
                            fragment_bitmap.insert(fragment_id);
                        }
                    }
                }
            }
        }
    }

    /// If an operation modifies one or more fields in a fragment then we need to remove
    /// that fragment from any indices that cover one of the modified fields.
    fn prune_updated_fields_from_indices(
        indices: &mut [IndexMetadata],
        updated_fragments: &[Fragment],
        fields_modified: &[u32],
    ) {
        if fields_modified.is_empty() {
            return;
        }

        // If we modified any fields in the fragments then we need to remove those fragments
        // from the index if the index covers one of those modified fields.
        let fields_modified_set = fields_modified.iter().collect::<HashSet<_>>();
        for index in indices.iter_mut() {
            if index
                .fields
                .iter()
                .any(|field_id| fields_modified_set.contains(&u32::try_from(*field_id).unwrap()))
            {
                if let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                    for fragment_id in updated_fragments.iter().map(|f| f.id as u32) {
                        fragment_bitmap.remove(fragment_id);
                    }
                }
            }
        }
    }

    fn is_vector_index(index: &IndexMetadata) -> bool {
        if let Some(details) = &index.index_details {
            details.type_url.ends_with("VectorIndexDetails")
        } else {
            false
        }
    }

    /// Remove data files that only contain tombstoned fields (-2)
    /// These files no longer contain any live data and can be safely dropped
    fn remove_tombstoned_data_files(fragments: &mut [Fragment]) {
        for fragment in fragments {
            fragment.files.retain(|file| {
                // Keep file if it has at least one non-tombstoned field
                file.fields.iter().any(|&field_id| field_id != -2)
            });
        }
    }

    fn retain_relevant_indices(
        indices: &mut Vec<IndexMetadata>,
        schema: &Schema,
        fragments: &[Fragment],
    ) {
        let field_ids = schema
            .fields_pre_order()
            .map(|f| f.id)
            .collect::<HashSet<_>>();

        // Remove indices for fields no longer in schema
        indices.retain(|existing_index| {
            existing_index
                .fields
                .iter()
                .all(|field_id| field_ids.contains(field_id))
                || is_system_index(existing_index)
        });

        // Fragment bitmaps are now immutable and always represent the fragments that
        // the index contains row IDs for, regardless of whether those fragments still exist.
        // This ensures consistent prefiltering behavior and clear semantics.

        // Apply retention logic for indices with empty bitmaps per index name
        // (except for fragment reuse indices which are always kept)
        let mut indices_by_name: std::collections::HashMap<String, Vec<&IndexMetadata>> =
            std::collections::HashMap::new();

        // Group indices by name
        for index in indices.iter() {
            if index.name != FRAG_REUSE_INDEX_NAME {
                indices_by_name
                    .entry(index.name.clone())
                    .or_default()
                    .push(index);
            }
        }

        // Build a set of UUIDs to keep based on retention rules
        let mut uuids_to_keep = std::collections::HashSet::new();

        let existing_fragments = fragments
            .iter()
            .map(|f| f.id as u32)
            .collect::<RoaringBitmap>();

        // For each group of indices with the same name
        for (_, same_name_indices) in indices_by_name {
            if same_name_indices.len() > 1 {
                // Separate empty and non-empty indices
                let (empty_indices, non_empty_indices): (Vec<_>, Vec<_>) =
                    same_name_indices.iter().partition(|index| {
                        index
                            .effective_fragment_bitmap(&existing_fragments)
                            .as_ref()
                            .is_none_or(|bitmap| bitmap.is_empty())
                    });

                if non_empty_indices.is_empty() {
                    // All indices are empty - for scalar indices, keep only the first (oldest) one
                    // For vector indices, remove all of them
                    let mut sorted_indices = empty_indices;
                    sorted_indices.sort_by_key(|index: &&IndexMetadata| index.dataset_version); // Sort by ascending dataset_version

                    // Keep only the first (oldest) if it's not a vector index
                    if let Some(oldest) = sorted_indices.first() {
                        if !Self::is_vector_index(oldest) {
                            uuids_to_keep.insert(oldest.uuid);
                        }
                    }
                } else {
                    // At least one index has non-empty bitmap - keep all non-empty indices
                    for index in non_empty_indices {
                        uuids_to_keep.insert(index.uuid);
                    }
                }
            } else {
                // Single index - keep it unless it's an empty vector index
                if let Some(index) = same_name_indices.first() {
                    let is_empty = index
                        .effective_fragment_bitmap(&existing_fragments)
                        .as_ref()
                        .is_none_or(|bitmap| bitmap.is_empty());
                    let is_vector = Self::is_vector_index(index);

                    // Keep the index unless it's an empty vector index
                    if !is_empty || !is_vector {
                        uuids_to_keep.insert(index.uuid);
                    }
                }
            }
        }

        // Use Vec::retain to safely remove indices
        indices.retain(|index| {
            index.name == FRAG_REUSE_INDEX_NAME || uuids_to_keep.contains(&index.uuid)
        });
    }

    fn recalculate_fragment_bitmap(
        old: &RoaringBitmap,
        groups: &[RewriteGroup],
    ) -> Result<RoaringBitmap> {
        let mut new_bitmap = old.clone();
        for group in groups {
            let any_in_index = group
                .old_fragments
                .iter()
                .any(|frag| old.contains(frag.id as u32));
            let all_in_index = group
                .old_fragments
                .iter()
                .all(|frag| old.contains(frag.id as u32));
            // Any rewrite group may or may not be covered by the index.  However, if any fragment
            // in a rewrite group was previously covered by the index then all fragments in the rewrite
            // group must have been previously covered by the index.  plan_compaction takes care of
            // this for us so this should be safe to assume.
            if any_in_index {
                if all_in_index {
                    for frag_id in group.old_fragments.iter().map(|frag| frag.id as u32) {
                        new_bitmap.remove(frag_id);
                    }
                    new_bitmap.extend(group.new_fragments.iter().map(|frag| frag.id as u32));
                } else {
                    return Err(Error::invalid_input("The compaction plan included a rewrite group that was a split of indexed and non-indexed data", location!()));
                }
            }
        }
        Ok(new_bitmap)
    }

    fn handle_rewrite_indices(
        indices: &mut [IndexMetadata],
        rewritten_indices: &[RewrittenIndex],
        groups: &[RewriteGroup],
    ) -> Result<()> {
        let mut modified_indices = HashSet::new();

        for rewritten_index in rewritten_indices {
            if !modified_indices.insert(rewritten_index.old_id) {
                return Err(Error::invalid_input(format!("An invalid compaction plan must have been generated because multiple tasks modified the same index: {}", rewritten_index.old_id), location!()));
            }

            let index = indices
                .iter_mut()
                .find(|idx| idx.uuid == rewritten_index.old_id)
                .ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "Invalid compaction plan refers to index {} which does not exist",
                            rewritten_index.old_id
                        ),
                        location!(),
                    )
                })?;

            index.fragment_bitmap = Some(Self::recalculate_fragment_bitmap(
                index.fragment_bitmap.as_ref().ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "Cannot rewrite index {} which did not store fragment bitmap",
                            index.uuid
                        ),
                        location!(),
                    )
                })?,
                groups,
            )?);
            index.uuid = rewritten_index.new_id;
        }
        Ok(())
    }

    fn handle_rewrite_fragments(
        final_fragments: &mut Vec<Fragment>,
        groups: &[RewriteGroup],
        fragment_id: &mut u64,
        version: u64,
        _next_row_id: Option<&u64>,
    ) -> Result<()> {
        for group in groups {
            // If the old fragments are contiguous, find the range
            let replace_range = {
                let start = final_fragments.iter().enumerate().find(|(_, f)| f.id == group.old_fragments[0].id)
                    .ok_or_else(|| Error::CommitConflict { version, source:
                    format!("dataset does not contain a fragment a rewrite operation wants to replace: id={}", group.old_fragments[0].id).into() , location:location!()})?.0;

                // Verify old_fragments matches contiguous range
                let mut i = 1;
                loop {
                    if i == group.old_fragments.len() {
                        break Some(start..start + i);
                    }
                    if final_fragments[start + i].id != group.old_fragments[i].id {
                        break None;
                    }
                    i += 1;
                }
            };

            let new_fragments = Self::fragments_with_ids(group.new_fragments.clone(), fragment_id)
                .collect::<Vec<_>>();

            // Version metadata for rewritten fragments is handled by the compaction code
            // (recalc_versions_for_rewritten_fragments) which preserves version information
            // from the original fragments. We don't modify it here.

            if let Some(replace_range) = replace_range {
                // Efficiently path using slice
                final_fragments.splice(replace_range, new_fragments);
            } else {
                // Slower path for non-contiguous ranges
                for fragment in group.old_fragments.iter() {
                    final_fragments.retain(|f| f.id != fragment.id);
                }
                final_fragments.extend(new_fragments);
            }
        }
        Ok(())
    }

    /// collect the pure(the num of row IDs are equal to the physical rows) "rewrite rows" updated fragment ids
    fn collect_pure_rewrite_row_update_frags_ids(fragments: &[Fragment]) -> Result<Vec<u64>> {
        let mut pure_update_frag_ids = Vec::new();

        for fragment in fragments {
            let physical_rows = fragment.physical_rows.ok_or_else(|| Error::Internal {
                message: "Fragment does not have physical rows".into(),
                location: location!(),
            })? as u64;

            if let Some(row_id_meta) = &fragment.row_id_meta {
                let existing_row_count = match row_id_meta {
                    RowIdMeta::Inline(data) => {
                        let sequence = read_row_ids(data)?;
                        sequence.len() as u64
                    }
                    _ => 0,
                };

                // only filter the fragments that match: all the rows have row id,
                // which means it does not contain inserted rows in this fragment
                if existing_row_count == physical_rows {
                    pure_update_frag_ids.push(fragment.id);
                }
            }
        }

        Ok(pure_update_frag_ids)
    }

    fn assign_row_ids(next_row_id: &mut u64, fragments: &mut [Fragment]) -> Result<()> {
        for fragment in fragments {
            let physical_rows = fragment.physical_rows.ok_or_else(|| Error::Internal {
                message: "Fragment does not have physical rows".into(),
                location: location!(),
            })? as u64;

            if fragment.row_id_meta.is_some() {
                // we may meet merge insert case, it only has partial row ids.
                // so here, we need to check if the row ids match the physical rows
                // if yes, continue
                // if not, fill the remaining row ids to the physical rows, then update row_id_meta

                // Check if existing row IDs match the physical rows count
                let existing_row_count = match &fragment.row_id_meta {
                    Some(RowIdMeta::Inline(data)) => {
                        // Parse the serialized row ID sequence to get the count
                        let sequence = read_row_ids(data)?;
                        sequence.len() as u64
                    }
                    _ => 0,
                };

                match existing_row_count.cmp(&physical_rows) {
                    Ordering::Equal => {
                        // Row IDs already match physical rows, continue to next fragment
                        continue;
                    }
                    Ordering::Less => {
                        // Partial row IDs - need to fill the remaining ones
                        let remaining_rows = physical_rows - existing_row_count;
                        let new_row_ids = *next_row_id..(*next_row_id + remaining_rows);

                        // Merge existing and new row IDs
                        let combined_sequence = match &fragment.row_id_meta {
                            Some(RowIdMeta::Inline(data)) => read_row_ids(data)?,
                            _ => {
                                return Err(Error::Internal {
                                    message: "Failed to deserialize existing row ID sequence"
                                        .into(),
                                    location: location!(),
                                })
                            }
                        };

                        let mut row_ids: Vec<u64> = combined_sequence.iter().collect();
                        for row_id in new_row_ids {
                            row_ids.push(row_id);
                        }
                        let combined_sequence = RowIdSequence::from(row_ids.as_slice());

                        let serialized = write_row_ids(&combined_sequence);
                        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
                        *next_row_id += remaining_rows;
                    }
                    Ordering::Greater => {
                        // More row IDs than physical rows - this shouldn't happen
                        return Err(Error::Internal {
                            message: format!(
                                "Fragment has more row IDs ({}) than physical rows ({})",
                                existing_row_count, physical_rows
                            ),
                            location: location!(),
                        });
                    }
                }
            } else {
                let row_ids = *next_row_id..(*next_row_id + physical_rows);
                let sequence = RowIdSequence::from(row_ids);
                // TODO: write to a separate file if large. Possibly share a file with other fragments.
                let serialized = write_row_ids(&sequence);
                fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
                *next_row_id += physical_rows;
            }
        }
        Ok(())
    }
}

impl From<&DataReplacementGroup> for pb::transaction::DataReplacementGroup {
    fn from(DataReplacementGroup(fragment_id, new_file): &DataReplacementGroup) -> Self {
        Self {
            fragment_id: *fragment_id,
            new_file: Some(new_file.into()),
        }
    }
}

/// Convert a protobug DataReplacementGroup to a rust native DataReplacementGroup
/// this is unfortunately TryFrom instead of From because of the Option in the pb::DataReplacementGroup
impl TryFrom<pb::transaction::DataReplacementGroup> for DataReplacementGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::DataReplacementGroup) -> Result<Self> {
        Ok(Self(
            message.fragment_id,
            message
                .new_file
                .ok_or(Error::invalid_input(
                    "DataReplacementGroup must have a new_file",
                    location!(),
                ))?
                .try_into()?,
        ))
    }
}

impl TryFrom<pb::Transaction> for Transaction {
    type Error = Error;

    fn try_from(message: pb::Transaction) -> Result<Self> {
        let operation = match message.operation {
            Some(pb::transaction::Operation::Append(pb::transaction::Append { fragments })) => {
                Operation::Append {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                }
            }
            Some(pb::transaction::Operation::Clone(pb::transaction::Clone {
                is_shallow,
                ref_name,
                ref_version,
                ref_path,
                branch_name,
            })) => Operation::Clone {
                is_shallow,
                ref_name,
                ref_version,
                ref_path,
                branch_name,
            },
            Some(pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            })) => Operation::Delete {
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                deleted_fragment_ids,
                predicate,
            },
            Some(pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
                config_upsert_values,
                initial_bases,
            })) => {
                let config_upsert_option = if config_upsert_values.is_empty() {
                    Some(config_upsert_values)
                } else {
                    None
                };

                Operation::Overwrite {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                    schema: Schema::from(&Fields(schema)),
                    config_upsert_values: config_upsert_option,
                    initial_bases: if initial_bases.is_empty() {
                        None
                    } else {
                        Some(initial_bases.into_iter().map(BasePath::from).collect())
                    },
                }
            }
            Some(pb::transaction::Operation::ReserveFragments(
                pb::transaction::ReserveFragments { num_fragments },
            )) => Operation::ReserveFragments { num_fragments },
            Some(pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                old_fragments,
                new_fragments,
                groups,
                rewritten_indices,
            })) => {
                let groups = if !groups.is_empty() {
                    groups
                        .into_iter()
                        .map(RewriteGroup::try_from)
                        .collect::<Result<_>>()?
                } else {
                    vec![RewriteGroup {
                        old_fragments: old_fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                        new_fragments: new_fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                    }]
                };
                let rewritten_indices = rewritten_indices
                    .iter()
                    .map(RewrittenIndex::try_from)
                    .collect::<Result<_>>()?;

                Operation::Rewrite {
                    groups,
                    rewritten_indices,
                    frag_reuse_index: None,
                }
            }
            Some(pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices,
                removed_indices,
            })) => Operation::CreateIndex {
                new_indices: new_indices
                    .into_iter()
                    .map(IndexMetadata::try_from)
                    .collect::<Result<_>>()?,
                removed_indices: removed_indices
                    .into_iter()
                    .map(IndexMetadata::try_from)
                    .collect::<Result<_>>()?,
            },
            Some(pb::transaction::Operation::Merge(pb::transaction::Merge {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
            })) => Operation::Merge {
                fragments: fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                schema: Schema::from(&Fields(schema)),
            },
            Some(pb::transaction::Operation::Restore(pb::transaction::Restore { version })) => {
                Operation::Restore { version }
            }
            Some(pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                mem_wal_to_merge,
                fields_for_preserving_frag_bitmap,
                update_mode,
            })) => Operation::Update {
                removed_fragment_ids,
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                new_fragments: new_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                fields_modified,
                mem_wal_to_merge: mem_wal_to_merge.map(|m| MemWal::try_from(m).unwrap()),
                fields_for_preserving_frag_bitmap,
                update_mode: match update_mode {
                    0 => Some(UpdateMode::RewriteRows),
                    1 => Some(UpdateMode::RewriteColumns),
                    _ => Some(UpdateMode::RewriteRows),
                },
            },
            Some(pb::transaction::Operation::Project(pb::transaction::Project { schema })) => {
                Operation::Project {
                    schema: Schema::from(&Fields(schema)),
                }
            }
            Some(pb::transaction::Operation::UpdateConfig(update_config)) => {
                // Check if new-style fields are present
                let has_new_fields = update_config.config_updates.is_some()
                    || update_config.table_metadata_updates.is_some()
                    || update_config.schema_metadata_updates.is_some()
                    || !update_config.field_metadata_updates.is_empty();

                // Check if old-style fields are present
                let has_old_fields = !update_config.upsert_values.is_empty()
                    || !update_config.delete_keys.is_empty()
                    || !update_config.schema_metadata.is_empty()
                    || !update_config.field_metadata.is_empty();

                // Error if both are present
                if has_new_fields && has_old_fields {
                    return Err(Error::InvalidInput {
                        source: "Cannot mix old and new style UpdateConfig fields".into(),
                        location: location!(),
                    });
                }

                if has_old_fields {
                    // Translate old-style to new-style
                    let config_updates = if !update_config.upsert_values.is_empty()
                        || !update_config.delete_keys.is_empty()
                    {
                        Some(translate_config_updates(
                            &update_config.upsert_values,
                            &update_config.delete_keys,
                        ))
                    } else {
                        None
                    };

                    let schema_metadata_updates = if !update_config.schema_metadata.is_empty() {
                        Some(translate_schema_metadata_updates(
                            &update_config.schema_metadata,
                        ))
                    } else {
                        None
                    };

                    let field_metadata_updates = update_config
                        .field_metadata
                        .into_iter()
                        .map(|(field_id, field_meta_update)| {
                            (
                                field_id as i32,
                                translate_schema_metadata_updates(&field_meta_update.metadata),
                            )
                        })
                        .collect();

                    Operation::UpdateConfig {
                        config_updates,
                        table_metadata_updates: None,
                        schema_metadata_updates,
                        field_metadata_updates,
                    }
                } else {
                    // Use new-style fields directly (convert from protobuf)
                    Operation::UpdateConfig {
                        config_updates: update_config.config_updates.as_ref().map(UpdateMap::from),
                        table_metadata_updates: update_config
                            .table_metadata_updates
                            .as_ref()
                            .map(UpdateMap::from),
                        schema_metadata_updates: update_config
                            .schema_metadata_updates
                            .as_ref()
                            .map(UpdateMap::from),
                        field_metadata_updates: update_config
                            .field_metadata_updates
                            .iter()
                            .map(|(field_id, pb_update_map)| {
                                (*field_id, UpdateMap::from(pb_update_map))
                            })
                            .collect(),
                    }
                }
            }
            Some(pb::transaction::Operation::DataReplacement(
                pb::transaction::DataReplacement { replacements },
            )) => Operation::DataReplacement {
                replacements: replacements
                    .into_iter()
                    .map(DataReplacementGroup::try_from)
                    .collect::<Result<Vec<_>>>()?,
            },
            Some(pb::transaction::Operation::UpdateMemWalState(
                pb::transaction::UpdateMemWalState {
                    added,
                    updated,
                    removed,
                },
            )) => Operation::UpdateMemWalState {
                added: added
                    .into_iter()
                    .map(|m| MemWal::try_from(m).unwrap())
                    .collect(),
                updated: updated
                    .into_iter()
                    .map(|m| MemWal::try_from(m).unwrap())
                    .collect(),
                removed: removed
                    .into_iter()
                    .map(|m| MemWal::try_from(m).unwrap())
                    .collect(),
            },
            Some(pb::transaction::Operation::UpdateBases(pb::transaction::UpdateBases {
                new_bases,
            })) => Operation::UpdateBases {
                new_bases: new_bases.into_iter().map(BasePath::from).collect(),
            },
            None => {
                return Err(Error::Internal {
                    message: "Transaction message did not contain an operation".to_string(),
                    location: location!(),
                });
            }
        };
        Ok(Self {
            read_version: message.read_version,
            uuid: message.uuid.clone(),
            operation,
            tag: if message.tag.is_empty() {
                None
            } else {
                Some(message.tag.clone())
            },
            transaction_properties: if message.transaction_properties.is_empty() {
                None
            } else {
                Some(Arc::new(message.transaction_properties))
            },
        })
    }
}

impl TryFrom<&pb::transaction::rewrite::RewrittenIndex> for RewrittenIndex {
    type Error = Error;

    fn try_from(message: &pb::transaction::rewrite::RewrittenIndex) -> Result<Self> {
        Ok(Self {
            old_id: message
                .old_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::io(
                        "required field (old_id) missing from message".to_string(),
                        location!(),
                    )
                })??,
            new_id: message
                .new_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::io(
                        "required field (new_id) missing from message".to_string(),
                        location!(),
                    )
                })??,
            new_index_details: message
                .new_index_details
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_input(
                        "new_index_details is a required field".to_string(),
                        location!(),
                    )
                })?
                .clone(),
            new_index_version: message.new_index_version,
        })
    }
}

impl TryFrom<pb::transaction::rewrite::RewriteGroup> for RewriteGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::rewrite::RewriteGroup) -> Result<Self> {
        Ok(Self {
            old_fragments: message
                .old_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
            new_fragments: message
                .new_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl From<&Transaction> for pb::Transaction {
    fn from(value: &Transaction) -> Self {
        let operation = match &value.operation {
            Operation::Append { fragments } => {
                pb::transaction::Operation::Append(pb::transaction::Append {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                })
            }
            Operation::Clone {
                is_shallow,
                ref_name,
                ref_version,
                ref_path,
                branch_name,
            } => pb::transaction::Operation::Clone(pb::transaction::Clone {
                is_shallow: *is_shallow,
                ref_name: ref_name.clone(),
                ref_version: *ref_version,
                ref_path: ref_path.clone(),
                branch_name: branch_name.clone(),
            }),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                deleted_fragment_ids: deleted_fragment_ids.clone(),
                predicate: predicate.clone(),
            }),
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
                initial_bases,
            } => {
                pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: Fields::from(schema).0,
                    schema_metadata: Default::default(), // TODO: handle metadata
                    config_upsert_values: config_upsert_values
                        .clone()
                        .unwrap_or(Default::default()),
                    initial_bases: initial_bases
                        .as_ref()
                        .map(|paths| {
                            paths
                                .iter()
                                .cloned()
                                .map(|bp: BasePath| -> pb::BasePath { bp.into() })
                                .collect::<Vec<pb::BasePath>>()
                        })
                        .unwrap_or_default(),
                })
            }
            Operation::ReserveFragments { num_fragments } => {
                pb::transaction::Operation::ReserveFragments(pb::transaction::ReserveFragments {
                    num_fragments: *num_fragments,
                })
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index: _,
            } => pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                groups: groups
                    .iter()
                    .map(pb::transaction::rewrite::RewriteGroup::from)
                    .collect(),
                rewritten_indices: rewritten_indices
                    .iter()
                    .map(|rewritten| rewritten.into())
                    .collect(),
                ..Default::default()
            }),
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices: new_indices.iter().map(pb::IndexMetadata::from).collect(),
                removed_indices: removed_indices
                    .iter()
                    .map(pb::IndexMetadata::from)
                    .collect(),
            }),
            Operation::Merge { fragments, schema } => {
                pb::transaction::Operation::Merge(pb::transaction::Merge {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: Fields::from(schema).0,
                    schema_metadata: Default::default(), // TODO: handle metadata
                })
            }
            Operation::Restore { version } => {
                pb::transaction::Operation::Restore(pb::transaction::Restore { version: *version })
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                mem_wal_to_merge,
                fields_for_preserving_frag_bitmap,
                update_mode,
            } => pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids: removed_fragment_ids.clone(),
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                new_fragments: new_fragments.iter().map(pb::DataFragment::from).collect(),
                fields_modified: fields_modified.clone(),
                mem_wal_to_merge: mem_wal_to_merge.as_ref().map(|m| m.into()),
                fields_for_preserving_frag_bitmap: fields_for_preserving_frag_bitmap.clone(),
                update_mode: update_mode
                    .as_ref()
                    .map(|mode| match mode {
                        UpdateMode::RewriteRows => 0,
                        UpdateMode::RewriteColumns => 1,
                    })
                    .unwrap_or(0),
            }),
            Operation::Project { schema } => {
                pb::transaction::Operation::Project(pb::transaction::Project {
                    schema: Fields::from(schema).0,
                })
            }
            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            } => pb::transaction::Operation::UpdateConfig(pb::transaction::UpdateConfig {
                config_updates: config_updates
                    .as_ref()
                    .map(pb::transaction::UpdateMap::from),
                table_metadata_updates: table_metadata_updates
                    .as_ref()
                    .map(pb::transaction::UpdateMap::from),
                schema_metadata_updates: schema_metadata_updates
                    .as_ref()
                    .map(pb::transaction::UpdateMap::from),
                field_metadata_updates: field_metadata_updates
                    .iter()
                    .map(|(field_id, update_map)| {
                        (*field_id, pb::transaction::UpdateMap::from(update_map))
                    })
                    .collect(),
                // Leave old fields empty - we only write new-style fields
                upsert_values: Default::default(),
                delete_keys: Default::default(),
                schema_metadata: Default::default(),
                field_metadata: Default::default(),
            }),
            Operation::DataReplacement { replacements } => {
                pb::transaction::Operation::DataReplacement(pb::transaction::DataReplacement {
                    replacements: replacements
                        .iter()
                        .map(pb::transaction::DataReplacementGroup::from)
                        .collect(),
                })
            }
            Operation::UpdateMemWalState {
                added,
                updated,
                removed,
            } => {
                pb::transaction::Operation::UpdateMemWalState(pb::transaction::UpdateMemWalState {
                    added: added
                        .iter()
                        .map(pb::mem_wal_index_details::MemWal::from)
                        .collect::<Vec<_>>(),
                    updated: updated
                        .iter()
                        .map(pb::mem_wal_index_details::MemWal::from)
                        .collect::<Vec<_>>(),
                    removed: removed
                        .iter()
                        .map(pb::mem_wal_index_details::MemWal::from)
                        .collect::<Vec<_>>(),
                })
            }
            Operation::UpdateBases { new_bases } => {
                pb::transaction::Operation::UpdateBases(pb::transaction::UpdateBases {
                    new_bases: new_bases
                        .iter()
                        .cloned()
                        .map(|bp: BasePath| -> pb::BasePath { bp.into() })
                        .collect::<Vec<pb::BasePath>>(),
                })
            }
        };

        let transaction_properties = value
            .transaction_properties
            .as_ref()
            .map(|arc| arc.as_ref().clone())
            .unwrap_or_default();
        Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            operation: Some(operation),
            tag: value.tag.clone().unwrap_or("".to_string()),
            transaction_properties,
        }
    }
}

impl From<&RewrittenIndex> for pb::transaction::rewrite::RewrittenIndex {
    fn from(value: &RewrittenIndex) -> Self {
        Self {
            old_id: Some((&value.old_id).into()),
            new_id: Some((&value.new_id).into()),
            new_index_details: Some(value.new_index_details.clone()),
            new_index_version: value.new_index_version,
        }
    }
}

impl From<&RewriteGroup> for pb::transaction::rewrite::RewriteGroup {
    fn from(value: &RewriteGroup) -> Self {
        Self {
            old_fragments: value
                .old_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
            new_fragments: value
                .new_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
        }
    }
}

/// Validate the operation is valid for the given manifest.
pub fn validate_operation(manifest: Option<&Manifest>, operation: &Operation) -> Result<()> {
    let manifest = match (manifest, operation) {
        (
            None,
            Operation::Overwrite {
                fragments, schema, ..
            },
        ) => {
            // Validate here because we are going to return early.
            schema_fragments_valid(None, schema, fragments)?;

            return Ok(());
        }
        (None, Operation::Clone { .. }) => return Ok(()),
        (Some(manifest), _) => manifest,
        (None, _) => {
            return Err(Error::invalid_input(
                format!(
                    "Cannot apply operation {} to non-existent dataset",
                    operation.name()
                ),
                location!(),
            ));
        }
    };

    match operation {
        Operation::Append { fragments } => {
            // Fragments must contain all fields in the schema
            schema_fragments_valid(Some(manifest), &manifest.schema, fragments)
        }
        Operation::Project { schema } => {
            schema_fragments_valid(Some(manifest), schema, manifest.fragments.as_ref())
        }
        Operation::Merge { fragments, schema } => {
            merge_fragments_valid(manifest, fragments)?;
            schema_fragments_valid(Some(manifest), schema, fragments)
        }
        Operation::Overwrite {
            fragments,
            schema,
            config_upsert_values: None,
            initial_bases: _,
        } => schema_fragments_valid(Some(manifest), schema, fragments),
        Operation::Update {
            updated_fragments,
            new_fragments,
            ..
        } => {
            schema_fragments_valid(Some(manifest), &manifest.schema, updated_fragments)?;
            schema_fragments_valid(Some(manifest), &manifest.schema, new_fragments)
        }
        _ => Ok(()),
    }
}

fn schema_fragments_valid(
    manifest: Option<&Manifest>,
    schema: &Schema,
    fragments: &[Fragment],
) -> Result<()> {
    if let Some(manifest) = manifest {
        if manifest.data_storage_format.lance_file_version()? == LanceFileVersion::Legacy {
            return schema_fragments_legacy_valid(schema, fragments);
        }
    }
    // validate that each data file at least contains one field.
    for fragment in fragments {
        for data_file in &fragment.files {
            if data_file.fields.iter().len() == 0 {
                return Err(Error::invalid_input(
                    format!("Datafile {} does not contain any fields", data_file.path),
                    location!(),
                ));
            }
        }
    }
    Ok(())
}

/// Check that each fragment contains all fields in the schema.
/// It is not required that the schema contains all fields in the fragment.
/// There may be masked fields.
fn schema_fragments_legacy_valid(schema: &Schema, fragments: &[Fragment]) -> Result<()> {
    // TODO: add additional validation. Consider consolidating with various
    // validate() methods in the codebase.
    for fragment in fragments {
        for field in schema.fields_pre_order() {
            if !fragment
                .files
                .iter()
                .flat_map(|f| f.fields.iter())
                .any(|f_id| f_id == &field.id)
            {
                return Err(Error::invalid_input(
                    format!(
                        "Fragment {} does not contain field {:?}",
                        fragment.id, field
                    ),
                    location!(),
                ));
            }
        }
    }
    Ok(())
}

/// Validate that Merge operations preserve all original fragments.
/// Merge operations should only add columns or rows, not reduce fragments.
/// This ensures fragments correspond at one-to-one with the original fragment list.
fn merge_fragments_valid(manifest: &Manifest, new_fragments: &[Fragment]) -> Result<()> {
    let original_fragments = manifest.fragments.as_ref();

    // Additional validation: ensure we're not accidentally reducing the fragment count
    if new_fragments.len() < original_fragments.len() {
        return Err(Error::invalid_input(
            format!(
                "Merge operation reduced fragment count from {} to {}. \
                 Merge operations should only add columns, not reduce fragments.",
                original_fragments.len(),
                new_fragments.len()
            ),
            location!(),
        ));
    }

    // Collect new fragment IDs
    let new_fragment_map: HashMap<u64, &Fragment> =
        new_fragments.iter().map(|f| (f.id, f)).collect();

    // Check that all original fragments are preserved in the new fragments list
    // Validate that each original fragment's metadata is preserved
    let mut missing_fragments: Vec<u64> = Vec::new();
    for original_fragment in original_fragments {
        if let Some(new_fragment) = new_fragment_map.get(&original_fragment.id) {
            // Validate physical_rows (row count) hasn't changed
            if original_fragment.physical_rows != new_fragment.physical_rows {
                return Err(Error::invalid_input(
                    format!(
                        "Merge operation changed row count for fragment {}. \
                         Original: {:?}, New: {:?}. \
                         Merge operations should preserve fragment row counts and only add new columns.",
                        original_fragment.id,
                        original_fragment.physical_rows,
                        new_fragment.physical_rows
                    ),
                    location!(),
                ));
            }
        } else {
            missing_fragments.push(original_fragment.id);
        }
    }

    if !missing_fragments.is_empty() {
        return Err(Error::invalid_input(
            format!(
                "Merge operation is missing original fragments: {:?}. \
                 Merge operations should preserve all original fragments and only add new columns. \
                 Expected fragments: {:?}, but got: {:?}",
                missing_fragments,
                original_fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
                new_fragment_map.keys().copied().collect::<Vec<_>>()
            ),
            location!(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_io::utils::CachedFileSize;

    #[test]
    fn test_rewrite_fragments() {
        let existing_fragments: Vec<Fragment> = (0..10).map(Fragment::new).collect();

        let mut final_fragments = existing_fragments;
        let rewrite_groups = vec![
            // Since these are contiguous, they will be put in the same location
            // as 1 and 2.
            RewriteGroup {
                old_fragments: vec![Fragment::new(1), Fragment::new(2)],
                // These two fragments were previously reserved
                new_fragments: vec![Fragment::new(15), Fragment::new(16)],
            },
            // These are not contiguous, so they will be inserted at the end.
            RewriteGroup {
                old_fragments: vec![Fragment::new(5), Fragment::new(8)],
                // We pretend this id was not reserved.  Does not happen in practice today
                // but we want to leave the door open.
                new_fragments: vec![Fragment::new(0)],
            },
        ];

        let mut fragment_id = 20;
        let version = 0;

        Transaction::handle_rewrite_fragments(
            &mut final_fragments,
            &rewrite_groups,
            &mut fragment_id,
            version,
            None,
        )
        .unwrap();

        assert_eq!(fragment_id, 21);

        let expected_fragments: Vec<Fragment> = vec![
            Fragment::new(0),
            Fragment::new(15),
            Fragment::new(16),
            Fragment::new(3),
            Fragment::new(4),
            Fragment::new(6),
            Fragment::new(7),
            Fragment::new(9),
            Fragment::new(20),
        ];

        assert_eq!(final_fragments, expected_fragments);
    }

    #[test]
    fn test_merge_fragments_valid() {
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema as LanceSchema;
        use lance_table::format::Manifest;
        use std::sync::Arc;

        // Create a simple schema for testing
        let schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]);

        // Create original fragments
        let original_fragments = vec![Fragment::new(1), Fragment::new(2), Fragment::new(3)];

        // Create a manifest with original fragments
        let manifest = Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(original_fragments),
            DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );

        // Test 1: Empty fragments should fail
        let empty_fragments = vec![];
        let result = merge_fragments_valid(&manifest, &empty_fragments);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("reduced fragment count"));

        // Test 2: Missing original fragments should fail
        let missing_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            // Fragment 3 is missing
            Fragment::new(4), // New fragment
        ];
        let result = merge_fragments_valid(&manifest, &missing_fragments);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing original fragments"));

        // Test 3: Reduced fragment count should fail
        let reduced_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            // Fragment 3 is missing, no new fragments added
        ];
        let result = merge_fragments_valid(&manifest, &reduced_fragments);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("reduced fragment count"));

        // Test 4: Valid merge with all original fragments plus new ones should succeed
        let valid_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            Fragment::new(3),
            Fragment::new(4), // New fragment
            Fragment::new(5), // Another new fragment
        ];
        let result = merge_fragments_valid(&manifest, &valid_fragments);
        assert!(result.is_ok());

        // Test 5: Same fragments (no new ones) should succeed
        let same_fragments = vec![Fragment::new(1), Fragment::new(2), Fragment::new(3)];
        let result = merge_fragments_valid(&manifest, &same_fragments);
        assert!(result.is_ok());
    }

    #[test]
    fn test_remove_tombstoned_data_files() {
        // Create a fragment with mixed data files: some normal, some fully tombstoned
        let mut fragment = Fragment::new(1);

        // Add a normal data file with valid field IDs
        fragment.files.push(DataFile {
            path: "normal.lance".to_string(),
            fields: vec![1, 2, 3],
            column_indices: vec![],
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(1000),
            base_id: None,
        });

        // Add a data file with all fields tombstoned
        fragment.files.push(DataFile {
            path: "all_tombstoned.lance".to_string(),
            fields: vec![-2, -2, -2],
            column_indices: vec![],
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(500),
            base_id: None,
        });

        // Add a data file with mixed tombstoned and valid fields
        fragment.files.push(DataFile {
            path: "mixed.lance".to_string(),
            fields: vec![4, -2, 5],
            column_indices: vec![],
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(750),
            base_id: None,
        });

        // Add another fully tombstoned file
        fragment.files.push(DataFile {
            path: "another_tombstoned.lance".to_string(),
            fields: vec![-2],
            column_indices: vec![],
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(250),
            base_id: None,
        });

        let mut fragments = vec![fragment];

        // Apply the cleanup
        Transaction::remove_tombstoned_data_files(&mut fragments);

        // Should have removed the two fully tombstoned files
        assert_eq!(fragments[0].files.len(), 2);
        assert_eq!(fragments[0].files[0].path, "normal.lance");
        assert_eq!(fragments[0].files[1].path, "mixed.lance");
    }

    #[test]
    fn test_assign_row_ids_new_fragment() {
        // Test assigning row IDs to a fragment without existing row IDs
        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(100),
            row_id_meta: None,
            files: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 0;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        assert_eq!(next_row_id, 100);
        assert!(fragments[0].row_id_meta.is_some());

        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 100);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, (0..100).collect::<Vec<u64>>());
        } else {
            panic!("Expected inline row ID metadata");
        }
    }

    #[test]
    fn test_assign_row_ids_existing_complete() {
        // Test with fragment that already has complete row IDs
        let existing_sequence = RowIdSequence::from(0..50);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(50),
            row_id_meta: Some(RowIdMeta::Inline(serialized)),
            files: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 100;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // next_row_id should not change
        assert_eq!(next_row_id, 100);

        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 50);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, (0..50).collect::<Vec<u64>>());
        } else {
            panic!("Expected inline row ID metadata");
        }
    }

    #[test]
    fn test_assign_row_ids_partial_existing() {
        // Test with fragment that has partial row IDs (merge insert case)
        let existing_sequence = RowIdSequence::from(0..30);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(50), // More physical rows than existing row IDs
            row_id_meta: Some(RowIdMeta::Inline(serialized)),
            files: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 100;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // next_row_id should advance by 20 (50 - 30)
        assert_eq!(next_row_id, 120);

        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 50);
            let row_ids: Vec<u64> = sequence.iter().collect();
            // Should contain original 0-29 plus new 100-119
            let mut expected = (0..30).collect::<Vec<u64>>();
            expected.extend(100..120);
            assert_eq!(row_ids, expected);
        } else {
            panic!("Expected inline row ID metadata");
        }
    }

    #[test]
    fn test_assign_row_ids_excess_row_ids() {
        // Test error case where fragment has more row IDs than physical rows
        let existing_sequence = RowIdSequence::from(0..60);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: Some(50), // Less physical rows than existing row IDs
            row_id_meta: Some(RowIdMeta::Inline(serialized)),
            files: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 100;

        let result = Transaction::assign_row_ids(&mut next_row_id, &mut fragments);

        assert!(result.is_err());
        if let Err(Error::Internal { message, .. }) = result {
            assert!(message.contains("more row IDs (60) than physical rows (50)"));
        } else {
            panic!("Expected Internal error about excess row IDs");
        }
    }

    #[test]
    fn test_assign_row_ids_multiple_fragments() {
        // Test with multiple fragments, some with existing row IDs, some without
        let existing_sequence = RowIdSequence::from(500..520);
        let serialized = write_row_ids(&existing_sequence);

        let mut fragments = vec![
            Fragment {
                id: 1,
                physical_rows: Some(30), // No existing row IDs
                row_id_meta: None,
                files: vec![],
                deletion_file: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            },
            Fragment {
                id: 2,
                physical_rows: Some(25), // Partial existing row IDs
                row_id_meta: Some(RowIdMeta::Inline(serialized)),
                files: vec![],
                deletion_file: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
            },
        ];
        let mut next_row_id = 1000;

        Transaction::assign_row_ids(&mut next_row_id, &mut fragments).unwrap();

        // Should advance by 30 (first fragment) + 5 (second fragment partial)
        assert_eq!(next_row_id, 1035);

        // Check first fragment
        if let Some(RowIdMeta::Inline(data)) = &fragments[0].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 30);
            let row_ids: Vec<u64> = sequence.iter().collect();
            assert_eq!(row_ids, (1000..1030).collect::<Vec<u64>>());
        } else {
            panic!("Expected inline row ID metadata for first fragment");
        }

        // Check second fragment
        if let Some(RowIdMeta::Inline(data)) = &fragments[1].row_id_meta {
            let sequence = read_row_ids(data).unwrap();
            assert_eq!(sequence.len(), 25);
            let row_ids: Vec<u64> = sequence.iter().collect();
            // Should contain original 500-519 plus new 1030-1034
            let mut expected = (500..520).collect::<Vec<u64>>();
            expected.extend(1030..1035);
            assert_eq!(row_ids, expected);
        } else {
            panic!("Expected inline row ID metadata for second fragment");
        }
    }

    #[test]
    fn test_assign_row_ids_missing_physical_rows() {
        // Test error case where fragment doesn't have physical_rows set
        let mut fragments = vec![Fragment {
            id: 1,
            physical_rows: None,
            row_id_meta: None,
            files: vec![],
            deletion_file: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        }];
        let mut next_row_id = 0;

        let result = Transaction::assign_row_ids(&mut next_row_id, &mut fragments);

        assert!(result.is_err());
        if let Err(Error::Internal { message, .. }) = result {
            assert!(message.contains("Fragment does not have physical rows"));
        } else {
            panic!("Expected Internal error about missing physical rows");
        }
    }

    // Helper functions for retain_relevant_indices tests
    fn create_test_index(
        name: &str,
        field_id: i32,
        dataset_version: u64,
        fragment_bitmap: Option<RoaringBitmap>,
        is_vector: bool,
    ) -> IndexMetadata {
        use prost_types::Any;
        use std::sync::Arc;
        use uuid::Uuid;

        let index_details = if is_vector {
            Some(Arc::new(Any {
                type_url: "type.googleapis.com/lance.index.VectorIndexDetails".to_string(),
                value: vec![],
            }))
        } else {
            Some(Arc::new(Any {
                type_url: "type.googleapis.com/lance.index.ScalarIndexDetails".to_string(),
                value: vec![],
            }))
        };

        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![field_id],
            name: name.to_string(),
            dataset_version,
            fragment_bitmap,
            index_details,
            index_version: 1,
            created_at: None,
            base_id: None,
        }
    }

    fn create_system_index(name: &str, field_id: i32) -> IndexMetadata {
        use prost_types::Any;
        use std::sync::Arc;
        use uuid::Uuid;

        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![field_id],
            name: name.to_string(),
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::from_iter([1, 2])),
            index_details: Some(Arc::new(Any {
                type_url: "type.googleapis.com/lance.index.SystemIndexDetails".to_string(),
                value: vec![],
            })),
            index_version: 1,
            created_at: None,
            base_id: None,
        }
    }

    fn create_test_schema(field_ids: &[i32]) -> Schema {
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_core::datatypes::Schema as LanceSchema;

        let fields: Vec<ArrowField> = field_ids
            .iter()
            .map(|id| ArrowField::new(format!("field_{}", id), DataType::Int32, false))
            .collect();

        let arrow_schema = ArrowSchema::new(fields);
        let mut lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // Assign field IDs
        for (i, field_id) in field_ids.iter().enumerate() {
            lance_schema.mut_field_by_id(i as i32).unwrap().id = *field_id;
        }

        lance_schema
    }

    #[test]
    fn test_retain_indices_removes_missing_fields() {
        let schema = create_test_schema(&[1, 2]);
        let fragments = vec![Fragment::new(1), Fragment::new(2)];

        let mut indices = vec![
            create_test_index("idx1", 1, 1, Some(RoaringBitmap::from_iter([1])), false),
            create_test_index("idx2", 2, 1, Some(RoaringBitmap::from_iter([1])), false),
            create_test_index("idx3", 99, 1, Some(RoaringBitmap::from_iter([1])), false), // Field doesn't exist
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 2);
        assert!(indices.iter().all(|idx| idx.fields[0] != 99));
    }

    #[test]
    fn test_retain_indices_keeps_system_indices() {
        use lance_index::mem_wal::MEM_WAL_INDEX_NAME;

        let schema = create_test_schema(&[1, 2]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_system_index(FRAG_REUSE_INDEX_NAME, 99), // Field doesn't exist but should be kept
            create_system_index(MEM_WAL_INDEX_NAME, 99), // Field doesn't exist but should be kept
            create_test_index("regular_idx", 99, 1, Some(RoaringBitmap::new()), false), // Should be removed
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 2);
        assert!(indices.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));
        assert!(indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME));
    }

    #[test]
    fn test_retain_indices_keeps_fragment_reuse_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_system_index(FRAG_REUSE_INDEX_NAME, 1),
            create_test_index("other_idx", 1, 1, Some(RoaringBitmap::new()), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Fragment reuse index should always be kept
        assert!(indices.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));
    }

    #[test]
    fn test_retain_single_empty_scalar_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![create_test_index(
            "scalar_idx",
            1,
            1,
            Some(RoaringBitmap::new()), // Empty bitmap
            false,
        )];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Single empty scalar index should be kept
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn test_retain_single_empty_vector_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![create_test_index(
            "vector_idx",
            1,
            1,
            Some(RoaringBitmap::new()), // Empty bitmap
            true,
        )];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Single empty vector index should be removed
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_retain_single_nonempty_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut scalar_indices = vec![create_test_index(
            "scalar_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1])),
            false,
        )];

        let mut vector_indices = vec![create_test_index(
            "vector_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1])),
            true,
        )];

        Transaction::retain_relevant_indices(&mut scalar_indices, &schema, &fragments);
        Transaction::retain_relevant_indices(&mut vector_indices, &schema, &fragments);

        // Both should be kept
        assert_eq!(scalar_indices.len(), 1);
        assert_eq!(vector_indices.len(), 1);
    }

    #[test]
    fn test_retain_single_index_with_none_bitmap() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut scalar_indices = vec![create_test_index("scalar_idx", 1, 1, None, false)];
        let mut vector_indices = vec![create_test_index("vector_idx", 1, 1, None, true)];

        Transaction::retain_relevant_indices(&mut scalar_indices, &schema, &fragments);
        Transaction::retain_relevant_indices(&mut vector_indices, &schema, &fragments);

        // Scalar should be kept, vector should be removed
        assert_eq!(scalar_indices.len(), 1);
        assert_eq!(vector_indices.len(), 0);
    }

    #[test]
    fn test_retain_multiple_empty_scalar_indices_keeps_oldest() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("idx", 1, 3, Some(RoaringBitmap::new()), false),
            create_test_index("idx", 1, 1, Some(RoaringBitmap::new()), false), // Oldest
            create_test_index("idx", 1, 2, Some(RoaringBitmap::new()), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should keep only the oldest (dataset_version = 1)
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].dataset_version, 1);
    }

    #[test]
    fn test_retain_multiple_empty_vector_indices_removes_all() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("vec_idx", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("vec_idx", 1, 2, Some(RoaringBitmap::new()), true),
            create_test_index("vec_idx", 1, 3, Some(RoaringBitmap::new()), true),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // All empty vector indices should be removed
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_retain_mixed_empty_nonempty_keeps_nonempty() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("idx", 1, 1, Some(RoaringBitmap::new()), false), // Empty
            create_test_index("idx", 1, 2, Some(RoaringBitmap::from_iter([1])), false), // Non-empty
            create_test_index("idx", 1, 3, Some(RoaringBitmap::new()), false), // Empty
            create_test_index("idx", 1, 4, Some(RoaringBitmap::from_iter([1])), false), // Non-empty
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should keep only non-empty indices
        assert_eq!(indices.len(), 2);
        assert!(indices
            .iter()
            .all(|idx| idx.dataset_version == 2 || idx.dataset_version == 4));
    }

    #[test]
    fn test_retain_mixed_empty_nonempty_vector_keeps_nonempty() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("vec_idx", 1, 1, Some(RoaringBitmap::new()), true), // Empty
            create_test_index("vec_idx", 1, 2, Some(RoaringBitmap::from_iter([1])), true), // Non-empty
            create_test_index("vec_idx", 1, 3, Some(RoaringBitmap::new()), true),          // Empty
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should keep only non-empty index
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].dataset_version, 2);
    }

    #[test]
    fn test_retain_fragment_bitmap_with_nonexistent_fragments() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1), Fragment::new(2)]; // Only fragments 1 and 2 exist

        let mut indices = vec![create_test_index(
            "idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1, 2, 3, 4])), // References non-existent fragments 3, 4
            false,
        )];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Should still keep the index (effective bitmap will be intersection with existing)
        assert_eq!(indices.len(), 1);
        // Original bitmap should be unchanged
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_retain_effective_empty_bitmap_single_index() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(5), Fragment::new(6)];

        // Bitmap references fragments that don't exist, so effective bitmap is empty
        let mut scalar_indices = vec![create_test_index(
            "scalar_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1, 2, 3])),
            false,
        )];

        let mut vector_indices = vec![create_test_index(
            "vector_idx",
            1,
            1,
            Some(RoaringBitmap::from_iter([1, 2, 3])),
            true,
        )];

        Transaction::retain_relevant_indices(&mut scalar_indices, &schema, &fragments);
        Transaction::retain_relevant_indices(&mut vector_indices, &schema, &fragments);

        // Scalar should be kept (single index, even if effective bitmap is empty)
        // Vector should be removed (empty effective bitmap)
        assert_eq!(scalar_indices.len(), 1);
        assert_eq!(vector_indices.len(), 0);
    }

    #[test]
    fn test_retain_different_index_names() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("idx_a", 1, 1, Some(RoaringBitmap::new()), false),
            create_test_index("idx_b", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("idx_c", 1, 1, Some(RoaringBitmap::from_iter([1])), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // idx_a (empty scalar) should be kept, idx_b (empty vector) removed, idx_c (non-empty) kept
        assert_eq!(indices.len(), 2);
        assert!(indices.iter().any(|idx| idx.name == "idx_a"));
        assert!(indices.iter().any(|idx| idx.name == "idx_c"));
        assert!(!indices.iter().any(|idx| idx.name == "idx_b"));
    }

    #[test]
    fn test_retain_empty_indices_vec() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices: Vec<IndexMetadata> = vec![];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_retain_all_indices_removed() {
        let schema = create_test_schema(&[1]);
        let fragments = vec![Fragment::new(1)];

        let mut indices = vec![
            create_test_index("vec1", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("vec2", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("idx3", 99, 1, Some(RoaringBitmap::from_iter([1])), false), // Bad field
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_retain_complex_scenario() {
        let schema = create_test_schema(&[1, 2]);
        let fragments = vec![Fragment::new(1), Fragment::new(2)];

        let mut indices = vec![
            // System index - should always be kept
            create_system_index(FRAG_REUSE_INDEX_NAME, 1),
            // Group "idx_a" - all empty scalars, keep oldest
            create_test_index("idx_a", 1, 3, Some(RoaringBitmap::new()), false),
            create_test_index("idx_a", 1, 1, Some(RoaringBitmap::new()), false), // Oldest
            create_test_index("idx_a", 1, 2, Some(RoaringBitmap::new()), false),
            // Group "vec_b" - all empty vectors, remove all
            create_test_index("vec_b", 1, 1, Some(RoaringBitmap::new()), true),
            create_test_index("vec_b", 1, 2, Some(RoaringBitmap::new()), true),
            // Group "idx_c" - mixed empty/non-empty, keep non-empty
            create_test_index("idx_c", 2, 1, Some(RoaringBitmap::new()), false),
            create_test_index("idx_c", 2, 2, Some(RoaringBitmap::from_iter([1])), false), // Keep
            create_test_index("idx_c", 2, 3, Some(RoaringBitmap::from_iter([2])), false), // Keep
            // Single non-empty - keep
            create_test_index("idx_d", 1, 1, Some(RoaringBitmap::from_iter([1, 2])), false),
            // Index with bad field - remove
            create_test_index("idx_e", 99, 1, Some(RoaringBitmap::from_iter([1])), false),
        ];

        Transaction::retain_relevant_indices(&mut indices, &schema, &fragments);

        // Expected: frag_reuse, idx_a (oldest), idx_c (2 non-empty), idx_d = 5 total
        assert_eq!(indices.len(), 5);

        // Verify system index kept
        assert!(indices.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));

        // Verify idx_a kept oldest only
        let idx_a_indices: Vec<_> = indices.iter().filter(|idx| idx.name == "idx_a").collect();
        assert_eq!(idx_a_indices.len(), 1);
        assert_eq!(idx_a_indices[0].dataset_version, 1);

        // Verify vec_b all removed
        assert!(!indices.iter().any(|idx| idx.name == "vec_b"));

        // Verify idx_c kept non-empty only
        let idx_c_indices: Vec<_> = indices.iter().filter(|idx| idx.name == "idx_c").collect();
        assert_eq!(idx_c_indices.len(), 2);
        assert!(idx_c_indices
            .iter()
            .all(|idx| idx.dataset_version == 2 || idx.dataset_version == 3));

        // Verify idx_d kept
        assert!(indices.iter().any(|idx| idx.name == "idx_d"));

        // Verify idx_e removed (bad field)
        assert!(!indices.iter().any(|idx| idx.name == "idx_e"));
    }
}
