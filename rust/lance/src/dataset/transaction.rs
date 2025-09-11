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
        pb::{self, IndexMetadata},
        DataFile, DataStorageFormat, Fragment, Index, Manifest, RowIdMeta,
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
    /// If the transaction modified the blobs dataset, this is the operation
    /// to apply to the blobs dataset.
    ///
    /// If this is `None`, then the blobs dataset was not modified
    pub blobs_op: Option<Operation>,
    pub tag: Option<String>,
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BlobsOperation {
    /// The operation did not modify the blobs dataset
    Unchanged,
    /// The operation modified the blobs dataset, contains the new version of the blobs dataset
    Updated(u64),
}

#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct DataReplacementGroup(pub u64, pub DataFile);

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
    },
    /// A new index has been created.
    CreateIndex {
        /// The new secondary indices that are being added
        new_indices: Vec<Index>,
        /// The indices that have been modified.
        removed_indices: Vec<Index>,
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
        frag_reuse_index: Option<Index>,
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
    },

    /// Project to a new schema. This only changes the schema, not the data.
    Project { schema: Schema },

    /// Update the dataset configuration.
    UpdateConfig {
        upsert_values: Option<HashMap<String, String>>,
        delete_keys: Option<Vec<String>>,
        schema_metadata: Option<HashMap<String, String>>,
        field_metadata: Option<HashMap<u32, HashMap<String, String>>>,
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
    },
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
                },
                Self::Clone {
                    is_shallow: b_is_shallow,
                    ref_name: b_ref_name,
                    ref_version: b_ref_version,
                    ref_path: b_source_path,
                },
            ) => {
                a_is_shallow == b_is_shallow
                    && a_ref_name == b_ref_name
                    && a_ref_version == b_ref_version
                    && a_source_path == b_source_path
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
                },
                Self::Overwrite {
                    fragments: b_fragments,
                    schema: b_schema,
                    config_upsert_values: b_config,
                },
            ) => {
                compare_vec(a_fragments, b_fragments)
                    && a_schema == b_schema
                    && a_config == b_config
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
                },
                Self::Update {
                    removed_fragment_ids: b_removed,
                    updated_fragments: b_updated,
                    new_fragments: b_new,
                    fields_modified: b_fields,
                    mem_wal_to_merge: b_mem_wal_to_merge,
                },
            ) => {
                compare_vec(a_removed, b_removed)
                    && compare_vec(a_updated, b_updated)
                    && compare_vec(a_new, b_new)
                    && compare_vec(a_fields, b_fields)
                    && a_mem_wal_to_merge == b_mem_wal_to_merge
            }
            (Self::Project { schema: a }, Self::Project { schema: b }) => a == b,
            (
                Self::UpdateConfig {
                    upsert_values: a_upsert,
                    delete_keys: a_delete,
                    schema_metadata: a_schema,
                    field_metadata: a_field,
                },
                Self::UpdateConfig {
                    upsert_values: b_upsert,
                    delete_keys: b_delete,
                    schema_metadata: b_schema,
                    field_metadata: b_field,
                },
            ) => {
                a_upsert == b_upsert
                    && a_delete.as_ref().map(|v| {
                        let mut v = v.clone();
                        v.sort();
                        v
                    }) == b_delete.as_ref().map(|v| {
                        let mut v = v.clone();
                        v.sort();
                        v
                    })
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
                upsert_values: Some(uv),
                ..
            } => {
                let vec: Vec<String> = uv.keys().cloned().collect();
                vec
            }
            _ => Vec::<String>::new(),
        }
    }

    /// Returns the config keys that have been deleted by this operation.
    fn get_delete_config_keys(&self) -> Vec<String> {
        match self {
            Self::UpdateConfig {
                delete_keys: Some(dk),
                ..
            } => dk.clone(),
            _ => Vec::<String>::new(),
        }
    }

    pub(crate) fn modifies_same_metadata(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::UpdateConfig {
                    schema_metadata,
                    field_metadata,
                    ..
                },
                Self::UpdateConfig {
                    schema_metadata: other_schema_metadata,
                    field_metadata: other_field_metadata,
                    ..
                },
            ) => {
                if schema_metadata.is_some() && other_schema_metadata.is_some() {
                    return true;
                }
                if let Some(field_metadata) = field_metadata {
                    if let Some(other_field_metadata) = other_field_metadata {
                        for field in field_metadata.keys() {
                            if other_field_metadata.contains_key(field) {
                                return true;
                            }
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
        }
    }
}

/// Add TransactionBuilder for flexibly setting option without using `mut`
pub struct TransactionBuilder {
    read_version: u64,
    // uuid is optional for builder since it can autogenerate
    uuid: Option<String>,
    operation: Operation,
    blobs_op: Option<Operation>,
    tag: Option<String>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
}

impl TransactionBuilder {
    pub fn new(read_version: u64, operation: Operation) -> Self {
        Self {
            read_version,
            uuid: None,
            operation,
            blobs_op: None,
            tag: None,
            transaction_properties: None,
        }
    }

    pub fn uuid(mut self, uuid: String) -> Self {
        self.uuid = Some(uuid);
        self
    }

    pub fn blobs_op(mut self, blobs_op: Option<Operation>) -> Self {
        self.blobs_op = blobs_op;
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
            blobs_op: self.blobs_op,
            tag: self.tag,
            transaction_properties: self.transaction_properties,
        }
    }
}

impl Transaction {
    pub fn new_from_version(read_version: u64, operation: Operation) -> Self {
        TransactionBuilder::new(read_version, operation).build()
    }

    pub fn with_blobs_op(self, blobs_op: Option<Operation>) -> Self {
        Self { blobs_op, ..self }
    }

    pub fn new(
        read_version: u64,
        operation: Operation,
        blobs_op: Option<Operation>,
        tag: Option<String>,
    ) -> Self {
        TransactionBuilder::new(read_version, operation)
            .blobs_op(blobs_op)
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
    ) -> Result<(Manifest, Vec<Index>)> {
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
        current_indices: Vec<Index>,
        transaction_file_path: &str,
        config: &ManifestWriteConfig,
        new_blob_version: Option<u64>,
    ) -> Result<(Manifest, Vec<Index>)> {
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
        let reference_paths = match current_manifest {
            Some(m) => m.base_paths.clone(),
            None => HashMap::new(),
        };

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
            } => {
                final_fragments.extend(maybe_existing_fragments?.iter().filter_map(|f| {
                    if removed_fragment_ids.contains(&f.id) {
                        return None;
                    }
                    if let Some(updated) = updated_fragments.iter().find(|uf| uf.id == f.id) {
                        Some(updated.clone())
                    } else {
                        Some(f.clone())
                    }
                }));

                // If we updated any fields, remove those fragments from indices covering those fields
                Self::prune_updated_fields_from_indices(
                    &mut final_indices,
                    updated_fragments,
                    fields_modified,
                );

                let mut new_fragments =
                    Self::fragments_with_ids(new_fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                }
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
            let mut prev_manifest = Manifest::new_from_previous(
                current_manifest,
                schema,
                Arc::new(final_fragments),
                new_blob_version,
            );

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
                new_blob_version,
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
            } => manifest.update_config(tm.clone()),
            Operation::UpdateConfig {
                upsert_values,
                delete_keys,
                schema_metadata,
                field_metadata,
            } => {
                // Delete is handled first. If the same key is referenced by upsert and
                // delete, then upserted key-value pair will remain.
                if let Some(delete_keys) = delete_keys {
                    manifest.delete_config_keys(
                        delete_keys
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                }
                if let Some(upsert_values) = upsert_values {
                    manifest.update_config(upsert_values.clone());
                }
                if let Some(schema_metadata) = schema_metadata {
                    manifest.replace_schema_metadata(schema_metadata.clone());
                }
                if let Some(field_metadata) = field_metadata {
                    for (field_id, metadata) in field_metadata {
                        manifest.replace_field_metadata(*field_id as i32, metadata.clone())?;
                    }
                }
            }
            _ => {}
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

    /// If an operation modifies one or more fields in a fragment then we need to remove
    /// that fragment from any indices that cover one of the modified fields.
    fn prune_updated_fields_from_indices(
        indices: &mut [Index],
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

    fn is_vector_index(index: &Index) -> bool {
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

    fn retain_relevant_indices(indices: &mut Vec<Index>, schema: &Schema, _fragments: &[Fragment]) {
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
        let mut indices_by_name: std::collections::HashMap<String, Vec<&Index>> =
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

        // For each group of indices with the same name
        for (_, same_name_indices) in indices_by_name {
            if same_name_indices.len() > 1 {
                // Separate empty and non-empty indices
                let (empty_indices, non_empty_indices): (Vec<_>, Vec<_>) =
                    same_name_indices.iter().partition(|index| {
                        index
                            .fragment_bitmap
                            .as_ref()
                            .is_none_or(|bitmap| bitmap.is_empty())
                    });

                if non_empty_indices.is_empty() {
                    // All indices are empty - for scalar indices, keep only the first (oldest) one
                    // For vector indices, remove all of them
                    let mut sorted_indices = empty_indices;
                    sorted_indices.sort_by_key(|index: &&Index| index.dataset_version); // Sort by ascending dataset_version

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
                        .fragment_bitmap
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
        indices: &mut [Index],
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

            let new_fragments = Self::fragments_with_ids(group.new_fragments.clone(), fragment_id);
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
            })) => Operation::Clone {
                is_shallow,
                ref_name,
                ref_version,
                ref_path,
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
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
                removed_indices: removed_indices
                    .into_iter()
                    .map(Index::try_from)
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
            },
            Some(pb::transaction::Operation::Project(pb::transaction::Project { schema })) => {
                Operation::Project {
                    schema: Schema::from(&Fields(schema)),
                }
            }
            Some(pb::transaction::Operation::UpdateConfig(pb::transaction::UpdateConfig {
                upsert_values,
                delete_keys,
                schema_metadata,
                field_metadata,
            })) => {
                let upsert_values = match upsert_values.len() {
                    0 => None,
                    _ => Some(upsert_values),
                };
                let delete_keys = match delete_keys.len() {
                    0 => None,
                    _ => Some(delete_keys),
                };
                let schema_metadata = match schema_metadata.len() {
                    0 => None,
                    _ => Some(schema_metadata),
                };
                let field_metadata = match field_metadata.len() {
                    0 => None,
                    _ => Some(
                        field_metadata
                            .into_iter()
                            .map(|(field_id, field_meta_update)| {
                                (field_id, field_meta_update.metadata)
                            })
                            .collect(),
                    ),
                };
                Operation::UpdateConfig {
                    upsert_values,
                    delete_keys,
                    schema_metadata,
                    field_metadata,
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
            None => {
                return Err(Error::Internal {
                    message: "Transaction message did not contain an operation".to_string(),
                    location: location!(),
                });
            }
        };
        let blobs_op = message
            .blob_operation
            .map(|blob_op| match blob_op {
                pb::transaction::BlobOperation::BlobAppend(pb::transaction::Append {
                    fragments,
                }) => Result::Ok(Operation::Append {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                }),
                pb::transaction::BlobOperation::BlobOverwrite(pb::transaction::Overwrite {
                    fragments,
                    schema,
                    schema_metadata: _schema_metadata, // TODO: handle metadata
                    config_upsert_values,
                }) => {
                    let config_upsert_option = if config_upsert_values.is_empty() {
                        Some(config_upsert_values)
                    } else {
                        None
                    };

                    Ok(Operation::Overwrite {
                        fragments: fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                        schema: Schema::from(&Fields(schema)),
                        config_upsert_values: config_upsert_option,
                    })
                }
            })
            .transpose()?;
        Ok(Self {
            read_version: message.read_version,
            uuid: message.uuid.clone(),
            operation,
            blobs_op,
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
            } => pb::transaction::Operation::Clone(pb::transaction::Clone {
                is_shallow: *is_shallow,
                ref_name: ref_name.clone(),
                ref_version: *ref_version,
                ref_path: ref_path.clone(),
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
            } => {
                pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: Fields::from(schema).0,
                    schema_metadata: Default::default(), // TODO: handle metadata
                    config_upsert_values: config_upsert_values
                        .clone()
                        .unwrap_or(Default::default()),
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
                new_indices: new_indices.iter().map(IndexMetadata::from).collect(),
                removed_indices: removed_indices.iter().map(IndexMetadata::from).collect(),
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
            } => pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids: removed_fragment_ids.clone(),
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                new_fragments: new_fragments.iter().map(pb::DataFragment::from).collect(),
                fields_modified: fields_modified.clone(),
                mem_wal_to_merge: mem_wal_to_merge
                    .as_ref()
                    .map(pb::mem_wal_index_details::MemWal::from),
            }),
            Operation::Project { schema } => {
                pb::transaction::Operation::Project(pb::transaction::Project {
                    schema: Fields::from(schema).0,
                })
            }
            Operation::UpdateConfig {
                upsert_values,
                delete_keys,
                schema_metadata,
                field_metadata,
            } => pb::transaction::Operation::UpdateConfig(pb::transaction::UpdateConfig {
                upsert_values: upsert_values.clone().unwrap_or(Default::default()),
                delete_keys: delete_keys.clone().unwrap_or(Default::default()),
                schema_metadata: schema_metadata.clone().unwrap_or(Default::default()),
                field_metadata: field_metadata
                    .as_ref()
                    .map(|field_metadata| {
                        field_metadata
                            .iter()
                            .map(|(field_id, metadata)| {
                                (
                                    *field_id,
                                    pb::transaction::update_config::FieldMetadataUpdate {
                                        metadata: metadata.clone(),
                                    },
                                )
                            })
                            .collect()
                    })
                    .unwrap_or(Default::default()),
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
        };

        let blob_operation = value.blobs_op.as_ref().map(|op| match op {
            Operation::Append { fragments } => {
                pb::transaction::BlobOperation::BlobAppend(pb::transaction::Append {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                })
            }
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
            } => {
                pb::transaction::BlobOperation::BlobOverwrite(pb::transaction::Overwrite {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: Fields::from(schema).0,
                    schema_metadata: Default::default(), // TODO: handle metadata
                    config_upsert_values: config_upsert_values
                        .clone()
                        .unwrap_or(Default::default()),
                })
            }
            _ => panic!("Invalid blob operation: {:?}", value),
        });

        let transaction_properties = value
            .transaction_properties
            .as_ref()
            .map(|arc| arc.as_ref().clone())
            .unwrap_or_default();
        Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            operation: Some(operation),
            blob_operation,
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
            None,
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
            },
            Fragment {
                id: 2,
                physical_rows: Some(25), // Partial existing row IDs
                row_id_meta: Some(RowIdMeta::Inline(serialized)),
                files: vec![],
                deletion_file: None,
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
}
