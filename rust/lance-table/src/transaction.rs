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
//! For more details please refer to the
//! [Transaction Specification](https://lance.org/format/table/transaction/#transaction-types).

mod conflicts;
mod row_version;

use row_version::resolve_update_version_metadata;
mod proto;
mod validate;

#[cfg(test)]
pub(crate) mod test_support;

use validate::merge_fragment_physically_rewritten;
pub use validate::validate_operation;

use crate::feature_flags::{FLAG_STABLE_ROW_IDS, apply_feature_flags};
use crate::format::key_existence::KeyExistenceFilter;
use crate::format::overlay::staleness::collect_overlay_stale_frags;
use crate::format::{
    BasePath, DataFile, DataStorageFormat, Fragment, IndexFile, IndexMetadata, Manifest,
    ManifestBuildConfig, overlay::DataOverlayFile,
};
use crate::io::{
    commit::CommitHandler,
    manifest::{read_manifest, read_manifest_indexes},
};
use crate::rowids::{RowIdSequence, version::build_version_meta};
use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::system_index::is_system_index;
use crate::system_index::mem_wal::{CompactedSsTable, update_mem_wal_index_compacted_sstables};
use crate::transaction::UpdateMode::{RewriteColumns, RewriteRows};
use lance_core::datatypes::{
    LANCE_UNENFORCED_CLUSTERING_KEY_POSITION, LANCE_UNENFORCED_PRIMARY_KEY,
    LANCE_UNENFORCED_PRIMARY_KEY_POSITION,
};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result, datatypes::Schema};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use roaring::RoaringBitmap;
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

/// Overlay files to append to a single fragment, in order (the last entry is
/// newest). The overlays are appended to the fragment's existing `overlays`
/// list rather than replacing it, so overlays written by concurrent commits are
/// preserved. Each overlay's `committed_version` is stamped to the new dataset
/// version at commit time (re-stamped on retry).
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct DataOverlayGroup {
    pub fragment_id: u64,
    pub overlays: Vec<DataOverlayFile>,
}

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
    /// say `frag_1: [A] [B, C]` and `frag_2: [A, B] [C]` and we are trying to replace column A
    /// with a new column A, the operation is not allowed.
    DataReplacement {
        replacements: Vec<DataReplacementGroup>,
    },
    /// Attach overlay files to fragments, supplying new values for a subset of
    /// `(physical offset, field)` cells without rewriting the fragments' base
    /// data files. See [`DataOverlayFile`] and the Data Overlay Files
    /// specification for resolution, coverage, and versioning rules.
    DataOverlay { groups: Vec<DataOverlayGroup> },
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
        /// MemWAL SSTables to mark as compacted after this transaction.
        compacted_sstables: Vec<CompactedSsTable>,
        /// The fields that used to judge whether to preserve the new frag's id into
        /// the frag bitmap of the specified indices.
        fields_for_preserving_frag_bitmap: Vec<u32>,
        /// The mode of update
        update_mode: Option<UpdateMode>,
        /// Optional filter for detecting conflicts on inserted row keys.
        /// Only tracks keys from INSERT operations during merge insert, not updates.
        inserted_rows_filter: Option<KeyExistenceFilter>,
        /// Physical row offsets (per fragment) that matched `update_columns` for RewriteColumns.
        /// `None` means callers did not supply offsets; `build_manifest` skips partial refresh then.
        updated_fragment_offsets: Option<UpdatedFragmentOffsets>,
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
    /// Update SSTable compaction progress in the MemWAL index.
    ///
    /// This is used during merge-insert to atomically record which
    /// SSTables have been compacted into the base table.
    UpdateMemWalState {
        compacted_sstables: Vec<CompactedSsTable>,
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

/// Matched physical row offsets per fragment for a partial [`UpdateMode::RewriteColumns`] update.
///
/// Used with stable row IDs so `build_manifest` can refresh row-level version
/// metadata only for rows that were rewritten.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UpdatedFragmentOffsets(pub HashMap<u64, RoaringBitmap>);

impl DeepSizeOf for UpdatedFragmentOffsets {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.0.iter().fold(0_usize, |acc, (frag_id, bitmap)| {
            acc + frag_id.deep_size_of_children(context)
                + (bitmap.len() as usize).saturating_mul(std::mem::size_of::<u32>())
        })
    }
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
            Self::DataOverlay { .. } => write!(f, "DataOverlay"),
            Self::Clone { .. } => write!(f, "Clone"),
            Self::UpdateMemWalState { .. } => write!(f, "UpdateMemWalState"),
            Self::UpdateBases { .. } => write!(f, "UpdateBases"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RewrittenIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub new_index_details: prost_types::Any,
    pub new_index_version: u32,
    /// Files in the new index with their sizes.
    /// Empty list from older writers that didn't persist this field.
    pub new_index_files: Option<Vec<IndexFile>>,
}

impl DeepSizeOf for RewrittenIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
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
            Self::DataOverlay { .. } => "DataOverlay",
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
            if let Some(user_requested) = user_requested
                && user_requested != file_version
            {
                return Err(Error::invalid_input(format!(
                    "User requested data storage version ({}) does not match version in data files ({})",
                    user_requested, file_version
                )));
            }
            Ok(DataStorageFormat::new(file_version))
        } else {
            // If no files use user-requested or default
            Ok(user_requested
                .map(DataStorageFormat::new)
                .unwrap_or_default())
        }
    }

    pub async fn restore_old_manifest(
        object_store: &ObjectStore,
        commit_handler: &dyn CommitHandler,
        base_path: &Path,
        version: u64,
        config: &ManifestBuildConfig,
        tx_path: &str,
        current_manifest: &Manifest,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let location = commit_handler
            .resolve_version_location(base_path, version, &object_store.inner)
            .await?;
        let mut manifest = read_manifest(object_store, &location.path, location.size).await?;
        manifest.set_timestamp(config.timestamp_nanos);
        manifest.transaction_file = Some(tx_path.to_string());
        let indices = read_manifest_indexes(object_store, &location, &manifest).await?;
        manifest.max_fragment_id = manifest
            .max_fragment_id
            .max(current_manifest.max_fragment_id);
        Ok((manifest, indices))
    }

    /// Create a new manifest from the current manifest and the transaction.
    ///
    /// `current_manifest` should only be None if the dataset does not yet exist.
    pub fn build_manifest(
        &self,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<IndexMetadata>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        if config.use_stable_row_ids
            && current_manifest
                .map(|m| !m.uses_stable_row_ids())
                .unwrap_or_default()
        {
            return Err(Error::not_supported_source(
                "Cannot enable stable row ids on existing dataset".into(),
            ));
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
                        return Err(Error::invalid_input(format!(
                            "Duplicate base path ID {} detected. Base path IDs must be unique.",
                            base_path.id
                        )));
                    }
                    reference_paths.insert(base_path.id, base_path.clone());
                }
            } else {
                // OVERWRITE mode with initial_bases should have been rejected by validation
                // This branch should never be reached
                return Err(Error::invalid_input(
                    "OVERWRITE mode cannot register new bases. This should have been caught by validation.",
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
                    return Err(Error::internal(
                        "Cannot create a new dataset without a schema".to_string(),
                    ));
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
                    return Err(Error::not_supported_source(
                        "Cannot enable stable row ids on existing dataset".into(),
                    ));
                }
            }
        };

        let maybe_existing_fragments =
            current_manifest
                .map(|m| m.fragments.as_ref())
                .ok_or_else(|| {
                    Error::internal(format!(
                        "No current manifest was provided while building manifest for operation {}",
                        self.operation.name()
                    ))
                });

        let new_version = current_manifest.map_or(1, |m| m.version + 1);

        match &self.operation {
            Operation::Clone { .. } => {
                return Err(Error::internal(
                    "Clone operation should not enter build_manifest.".to_string(),
                ));
            }
            Operation::Append { fragments } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let mut new_fragments =
                    Self::fragments_with_ids(fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Add version metadata for all new fragments
                    for fragment in new_fragments.iter_mut() {
                        let version_meta = build_version_meta(fragment, new_version);
                        fragment.last_updated_at_version_meta = version_meta.clone();
                        fragment.created_at_version_meta = version_meta;
                    }
                }
                final_fragments.extend(new_fragments);
            }
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
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
                compacted_sstables,
                fields_for_preserving_frag_bitmap,
                update_mode,
                updated_fragment_offsets,
                ..
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
                            let mut updated = updated.clone();
                            // Carry forward the fragment's current overlays (which
                            // may include ones added by a concurrent commit). An
                            // in-place column rewrite then tombstones the overlaid
                            // fields it rewrote, since the fresh base values
                            // supersede them.
                            updated.overlays = f.overlays.clone();
                            if matches!(update_mode, Some(RewriteColumns)) {
                                crate::format::overlay::tombstone_overlay_fields(
                                    &mut updated.overlays,
                                    fields_modified,
                                );
                            }
                            Some(updated)
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

                if next_row_id.is_some()
                    && matches!(update_mode, Some(RewriteColumns))
                    && let Some(UpdatedFragmentOffsets(off_map)) = updated_fragment_offsets
                    && !off_map.is_empty()
                {
                    let prev_version = current_manifest.map(|m| m.version).unwrap_or(0);
                    for fragment in final_fragments.iter_mut() {
                        let Some(bitmap) = off_map.get(&fragment.id) else {
                            continue;
                        };
                        if bitmap.is_empty() {
                            continue;
                        }
                        // Skip fragments with no existing version metadata: the helper
                        // would fill unmatched rows with prev_version, fabricating a
                        // last_updated stamp for rows that never had one.
                        if fragment.last_updated_at_version_meta.is_none() {
                            continue;
                        }
                        let offsets: Vec<usize> = bitmap.iter().map(|o| o as usize).collect();
                        crate::rowids::version::refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
                            fragment,
                            &offsets,
                            new_version,
                            prev_version,
                        )?;
                    }
                }

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

                if next_row_id.is_some() {
                    resolve_update_version_metadata(
                        existing_fragments,
                        new_fragments.as_mut_slice(),
                        new_version,
                    )?;
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

                    // The original fragments that carried an overlay: their moved rows may have a
                    // stale index entry (see `register_pure_rewrite_rows_update_frags_in_indices`).
                    let original_overlaid_frags: HashMap<u32, &Fragment> = existing_fragments
                        .iter()
                        .filter(|f| original_fragment_ids.contains(&f.id) && !f.overlays.is_empty())
                        .map(|f| (f.id as u32, f))
                        .collect();

                    Self::register_pure_rewrite_rows_update_frags_in_indices(
                        &mut final_indices,
                        &pure_updated_frag_ids,
                        &original_fragment_ids,
                        fields_for_preserving_frag_bitmap,
                        &original_overlaid_frags,
                        &schema,
                    )?;
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

                if !compacted_sstables.is_empty() {
                    update_mem_wal_index_compacted_sstables(
                        &mut final_indices,
                        new_version,
                        compacted_sstables.clone(),
                    )?;
                }
            }
            Operation::Overwrite { fragments, .. } => {
                let mut new_fragments =
                    Self::fragments_with_ids(fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                    // Add version metadata for all new fragments
                    for fragment in new_fragments.iter_mut() {
                        let version_meta = build_version_meta(fragment, new_version);
                        fragment.last_updated_at_version_meta = version_meta.clone();
                        fragment.created_at_version_meta = version_meta;
                    }
                }
                final_fragments.extend(new_fragments);
                final_indices = Vec::new();
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index,
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

                // A full compaction materializes a fragment's overlays into fresh
                // base data. Any index older than one of those overlays was built on
                // the pre-overlay values, so drop the rewritten fragment from its
                // coverage to keep it from serving stale values.
                Self::prune_overlay_stale_fields_from_indices(&mut final_indices, groups);

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
                let removed_uuids = removed_indices
                    .iter()
                    .map(|old_index| old_index.uuid)
                    .collect::<HashSet<_>>();
                let new_uuids = new_indices
                    .iter()
                    .map(|new_index| new_index.uuid)
                    .collect::<HashSet<_>>();
                final_indices.retain(|existing_index| {
                    !removed_uuids.contains(&existing_index.uuid)
                        && !new_uuids.contains(&existing_index.uuid)
                });
                final_indices.extend(new_indices.clone());
            }
            Operation::ReserveFragments { .. } | Operation::UpdateConfig { .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
            }
            Operation::Merge { fragments, .. } => {
                let existing_fragments = maybe_existing_fragments?;
                let mut merged_fragments = fragments.clone();
                if next_row_id.is_some() {
                    let prev_by_id: HashMap<u64, &Fragment> =
                        existing_fragments.iter().map(|f| (f.id, f)).collect();
                    for fragment in merged_fragments.iter_mut() {
                        match prev_by_id.get(&fragment.id) {
                            Some(prev) => {
                                if merge_fragment_physically_rewritten(prev, fragment) {
                                    crate::rowids::version::refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
                                        fragment,
                                        new_version,
                                    )?;
                                }
                            }
                            None => {
                                // Brand-new fragment ID not present in the previous manifest.
                                // Set both last_updated and created version meta, consistent
                                // with Append/Overwrite for genuinely new fragments.
                                crate::rowids::version::refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
                                    fragment,
                                    new_version,
                                )?;
                                fragment.created_at_version_meta =
                                    fragment.last_updated_at_version_meta.clone();
                            }
                        }
                    }
                }
                final_fragments.extend(merged_fragments);

                // A Merge can rewrite a column's data file in place; the field stays
                // in the schema, so the index is retained -- prune its now-stale
                // entries for the rewritten fragments.
                Self::prune_merge_rewritten_fields_from_indices(
                    &mut final_indices,
                    existing_fragments,
                    fragments,
                );

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
                log::warn!(
                    "Building manifest with DataReplacement operation. This operation is not stable yet, please use with caution."
                );

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

                    return Err(Error::invalid_input(format!(
                        "All new data files must have the same fields, but found different fields:\n{field_info}"
                    )));
                }

                let existing_fragments = maybe_existing_fragments?;

                // Collect replaced field IDs before consuming new_datafiles
                let replaced_fields: Vec<u32> = new_datafiles
                    .first()
                    .map(|f| {
                        f.fields
                            .iter()
                            .filter(|&&id| id >= 0)
                            .map(|&id| id as u32)
                            .collect()
                    })
                    .unwrap_or_default();

                // 2. check that the fragments being modified have isomorphic layouts along the columns being replaced
                // 3. add modified fragments to final_fragments
                for (frag_id, new_file) in old_fragment_ids.iter().zip(new_datafiles) {
                    let frag = existing_fragments
                        .iter()
                        .find(|f| f.id == **frag_id)
                        .ok_or_else(|| {
                            Error::invalid_input(
                                "Fragment being replaced not found in existing fragments",
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
                            // assign the new file path / size / base to the fragment
                            file.path = new_file.path.clone();
                            file.file_size_bytes = new_file.file_size_bytes.clone();
                            file.base_id = new_file.base_id;
                        }
                        columns_covered.extend(file.fields.iter());
                    }
                    // SPECIAL CASE: if the column(s) being replaced are not covered by the fragment
                    // Then it means it's a all-NULL column that is being replaced with real data
                    // just add it to the final fragments. Push the DataFile as
                    // given so every field (including base_id) is preserved.
                    if columns_covered.is_disjoint(&new_file.fields.iter().collect()) {
                        LanceFileVersion::try_from_major_minor(
                            new_file.file_major_version,
                            new_file.file_minor_version,
                        )
                        .expect("Expected valid file version");
                        new_frag.files.push(new_file.clone());
                    }

                    // Nothing changed in the current fragment, which is not expected -- error out
                    if &new_frag == frag {
                        return Err(Error::invalid_input(
                            "Expected to modify the fragment but no changes were made. This means the new data files does not align with any exiting datafiles. Please check if the schema of the new data files matches the schema of the old data files including the file major and minor versions",
                        ));
                    }

                    // New base values for these fields supersede any overlay
                    // still shadowing them; tombstone the overlaid fields so the
                    // replacement is not silently masked.
                    crate::format::overlay::tombstone_overlay_fields(
                        &mut new_frag.overlays,
                        &replaced_fields,
                    );

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

                // 5. Invalidate index bitmaps for replaced fields
                let modified_fragments: Vec<Fragment> = final_fragments
                    .iter()
                    .filter(|f| fragments_changed.contains(&f.id))
                    .cloned()
                    .collect();

                Self::prune_updated_fields_from_indices(
                    &mut final_indices,
                    &modified_fragments,
                    &replaced_fields,
                );
            }
            Operation::DataOverlay { groups } => {
                // Stamp each overlay with the version this commit is producing.
                // build_manifest re-runs on every retry with an updated
                // current_manifest, so this is naturally re-stamped on retry.
                let new_version = current_manifest.map_or(1, |m| m.version + 1);

                let existing_fragments = maybe_existing_fragments?;
                // Multiple groups may target the same fragment; merge them in
                // order rather than letting a HashMap collapse drop all but the
                // last group's overlays.
                let mut overlays_by_fragment: HashMap<u64, Vec<&DataOverlayFile>> = HashMap::new();
                for group in groups {
                    overlays_by_fragment
                        .entry(group.fragment_id)
                        .or_default()
                        .extend(group.overlays.iter());
                }

                // Every group must target an existing fragment. Build a set of
                // existing ids once so this is O(groups + fragments) rather than
                // O(groups * fragments).
                let existing_fragment_ids: HashSet<u64> =
                    existing_fragments.iter().map(|f| f.id).collect();
                for fragment_id in overlays_by_fragment.keys() {
                    if !existing_fragment_ids.contains(fragment_id) {
                        return Err(Error::invalid_input(format!(
                            "DataOverlay targets fragment {fragment_id}, which does not exist"
                        )));
                    }
                }

                for fragment in existing_fragments {
                    let mut fragment = fragment.clone();
                    if let Some(new_overlays) = overlays_by_fragment.get(&fragment.id) {
                        // Appended (not replaced) so concurrently-written overlays
                        // survive; later entries are newer.
                        fragment
                            .overlays
                            .extend(new_overlays.iter().map(|&overlay| {
                                let mut overlay = overlay.clone();
                                overlay.committed_version = new_version;
                                overlay
                            }));
                    }
                    final_fragments.push(fragment);
                }
            }
            Operation::UpdateMemWalState { compacted_sstables } => {
                update_mem_wal_index_compacted_sstables(
                    &mut final_indices,
                    new_version,
                    compacted_sstables.clone(),
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

        // Enforce the newest-last overlay ordering invariant at the write
        // boundary. Load normalizes with a sort; this rejects any commit path
        // that assembled a fragment's overlays out of order.
        for fragment in &final_fragments {
            if !fragment.overlays.is_empty() {
                crate::format::overlay::verify_overlays_newest_last(&fragment.overlays)?;
            }
        }

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
            // Internal operations (e.g. CreateIndex) build with the default config,
            // which has use_stable_row_ids = false. Without inheriting from the previous
            // manifest, apply_feature_flags would clear FLAG_STABLE_ROW_IDS.
            let inherited = current_manifest
                .map(|m| m.uses_stable_row_ids())
                .unwrap_or(false);
            let use_stable_row_ids = config.use_stable_row_ids || inherited;
            apply_feature_flags(
                &mut manifest,
                use_stable_row_ids,
                config.disable_transaction_file,
            )?;
        }
        manifest.set_timestamp(config.timestamp_nanos);

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
                // The unenforced primary and clustering keys are reserved
                // schema properties: each is immutable once set, and its
                // reserved metadata keys cannot be written with an invalid
                // value. Capture the prior keys, and whether this transaction
                // writes a reserved key, before applying the updates so
                // violations can be rejected below. This runs on every apply,
                // including conflict-rebase, so it also rejects the
                // concurrent-writer race.
                let primary_key_before: Vec<i32> = manifest
                    .schema
                    .unenforced_primary_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                let writes_primary_key = field_metadata_updates.values().any(|update| {
                    update.update_entries.iter().any(|entry| {
                        entry.key == LANCE_UNENFORCED_PRIMARY_KEY
                            || entry.key == LANCE_UNENFORCED_PRIMARY_KEY_POSITION
                    })
                });
                let clustering_key_before: Vec<i32> = manifest
                    .schema
                    .unenforced_clustering_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                let writes_clustering_key = field_metadata_updates.values().any(|update| {
                    update
                        .update_entries
                        .iter()
                        .any(|entry| entry.key == LANCE_UNENFORCED_CLUSTERING_KEY_POSITION)
                });
                for (field_id, field_metadata_update) in field_metadata_updates {
                    if let Some(field) = manifest.schema.field_by_id_mut(*field_id) {
                        apply_update_map(&mut field.metadata, field_metadata_update);
                        // Also set unenforced primary key based on updated field metadata.
                        field.unenforced_primary_key_position = field
                            .metadata
                            .get(LANCE_UNENFORCED_PRIMARY_KEY_POSITION)
                            .and_then(|s| s.parse::<u32>().ok())
                            .or_else(|| {
                                field
                                    .metadata
                                    .get(LANCE_UNENFORCED_PRIMARY_KEY)
                                    .filter(|s| {
                                        matches!(s.to_lowercase().as_str(), "true" | "1" | "yes")
                                    })
                                    .map(|_| 0)
                            });
                        // Also set unenforced clustering key based on updated
                        // field metadata.
                        field.unenforced_clustering_key_position = field
                            .metadata
                            .get(LANCE_UNENFORCED_CLUSTERING_KEY_POSITION)
                            .and_then(|s| s.parse::<u32>().ok());
                    } else {
                        return Err(Error::invalid_input_source(
                            format!("Field with id {} does not exist", field_id).into(),
                        ));
                    }
                }
                let primary_key_after: Vec<i32> = manifest
                    .schema
                    .unenforced_primary_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                if !primary_key_before.is_empty() {
                    // The primary key is already set: reject any change to it,
                    // and any write that touches a reserved primary key.
                    if writes_primary_key || primary_key_after != primary_key_before {
                        return Err(Error::invalid_input(
                            "the unenforced primary key is a reserved key and cannot be changed once set",
                        ));
                    }
                } else if writes_primary_key && primary_key_after.is_empty() {
                    // A reserved primary key was written but did not install a
                    // valid primary key (e.g. a non-marker flag value or a
                    // non-numeric position).
                    return Err(Error::invalid_input(
                        "the unenforced primary key is a reserved key and cannot be set to an invalid value",
                    ));
                }
                let clustering_key_after: Vec<i32> = manifest
                    .schema
                    .unenforced_clustering_key()
                    .iter()
                    .map(|field| field.id)
                    .collect();
                if !clustering_key_before.is_empty() {
                    // The clustering key is already set: reject any change to
                    // it, and any write that touches the reserved key.
                    if writes_clustering_key || clustering_key_after != clustering_key_before {
                        return Err(Error::invalid_input(
                            "the unenforced clustering key is a reserved key and cannot be changed once set",
                        ));
                    }
                } else if writes_clustering_key && clustering_key_after.is_empty() {
                    // The reserved clustering key was written but did not
                    // install a valid clustering key (e.g. a non-numeric
                    // position value).
                    return Err(Error::invalid_input(
                        "the unenforced clustering key is a reserved key and cannot be set to an invalid value",
                    ));
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
                    return Err(Error::invalid_input(format!(
                        "Conflict detected: Base path with name '{:?}' or path '{}' already exists. Existing: name='{:?}', path='{}'",
                        new_base.name, new_base.path, existing_base.name, existing_base.path
                    )));
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
        original_overlaid_frags: &HashMap<u32, &Fragment>,
        schema: &Schema,
    ) -> Result<()> {
        if pure_update_frag_ids.is_empty() {
            return Ok(());
        }

        let value_updated_field_set = fields_for_preserving_frag_bitmap
            .iter()
            .collect::<HashSet<_>>();

        for index in indices.iter_mut() {
            let index_covers_modified_field = index.fields.iter().any(|field_id| {
                value_updated_field_set.contains(&u32::try_from(*field_id).unwrap())
            });
            if index_covers_modified_field {
                continue;
            }
            let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() else {
                continue;
            };

            // Check that all the original fragments containing the updated rows are covered by
            // the index. If not, some updated rows were not indexed, so we cannot index them.
            let index_covers_all_original_fragments = original_fragment_ids
                .iter()
                .all(|&fragment_id| fragment_bitmap.contains(fragment_id as u32));
            if !index_covers_all_original_fragments {
                continue;
            }

            // A rewrite materializes overlays.  If any of those overlays touched the
            // column being indexed then the rewrite will modify that column.  As a
            // result, that index will no longer cover the fragment and it does not
            // count as a pure rewrite and we must exclude it from the index's fragment
            // bitmap.
            let mut overlay_stale = RoaringBitmap::new();
            collect_overlay_stale_frags(
                index,
                original_overlaid_frags,
                &mut overlay_stale,
                schema,
            )?;
            if !overlay_stale.is_empty() {
                continue;
            }

            if let Some(fragment_bitmap) = index.fragment_bitmap.as_mut() {
                for fragment_id in pure_update_frag_ids.iter().map(|f| *f as u32) {
                    fragment_bitmap.insert(fragment_id);
                }
            }
        }
        Ok(())
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
                && let Some(fragment_bitmap) = &mut index.fragment_bitmap
            {
                for fragment_id in updated_fragments.iter().map(|f| f.id as u32) {
                    fragment_bitmap.remove(fragment_id);
                }
            }
        }
    }

    /// Map each (non-tombstoned) field id in a fragment to the path of the data
    /// file that backs it.
    fn fragment_field_paths(frag: &Fragment) -> HashMap<i32, &str> {
        let mut map = HashMap::new();
        for file in &frag.files {
            for &field_id in file.fields.iter() {
                if field_id >= 0 {
                    map.insert(field_id, file.path.as_str());
                }
            }
        }
        map
    }

    /// A `Merge` can rewrite a column's data *in place* -- the field stays in the
    /// schema but its backing data file changes (the overlay fragment carries a new
    /// file for the field and tombstones its old field id). `retain_relevant_indices`
    /// only drops indices for *removed* fields, so without this the index keeps
    /// covering the rewritten fragments with stale entries. Remove each such fragment
    /// from any index covering a field whose backing data file changed.
    fn prune_merge_rewritten_fields_from_indices(
        indices: &mut [IndexMetadata],
        prev_fragments: &[Fragment],
        new_fragments: &[Fragment],
    ) {
        let prev_by_id: HashMap<u64, &Fragment> =
            prev_fragments.iter().map(|f| (f.id, f)).collect();
        for new_frag in new_fragments {
            let Some(prev) = prev_by_id.get(&new_frag.id) else {
                continue; // brand-new fragment: nothing stale to prune
            };
            let prev_paths = Self::fragment_field_paths(prev);
            let new_paths = Self::fragment_field_paths(new_frag);
            // Fields still present whose backing file path changed == rewritten data.
            let changed: Vec<u32> = prev_paths
                .iter()
                .filter(|(field_id, prev_path)| {
                    new_paths
                        .get(*field_id)
                        .is_some_and(|new_path| new_path != *prev_path)
                })
                .map(|(field_id, _)| *field_id as u32)
                .collect();
            if changed.is_empty() {
                continue;
            }
            Self::prune_updated_fields_from_indices(
                indices,
                std::slice::from_ref(new_frag),
                &changed,
            );
        }
    }

    /// After a `Rewrite` fully compacts a fragment, its data overlays are baked
    /// into the new fragment's base data. An index built *before* one of those
    /// overlays (`overlay.committed_version > index.dataset_version`) indexed the
    /// stale pre-overlay values -- and unlike a live overlay, the compacted
    /// fragment no longer signals that staleness to the query path. Drop each
    /// rewritten (new) fragment from the coverage of any index covering a field
    /// such an overlay supplied, so those rows fall back to a flat scan.
    fn prune_overlay_stale_fields_from_indices(
        indices: &mut [IndexMetadata],
        groups: &[RewriteGroup],
    ) {
        for group in groups {
            // field id -> newest overlay committed_version supplying that field
            let mut overlaid_field_versions: HashMap<i32, u64> = HashMap::new();
            for old_frag in &group.old_fragments {
                for overlay in &old_frag.overlays {
                    for &field_id in overlay.data_file.fields.iter() {
                        if field_id < 0 {
                            // Tombstoned (obsolete) overlay field: supplies nothing.
                            continue;
                        }
                        let entry = overlaid_field_versions.entry(field_id).or_insert(0);
                        *entry = (*entry).max(overlay.committed_version);
                    }
                }
            }
            if overlaid_field_versions.is_empty() {
                continue;
            }

            let new_fragment_ids = group
                .new_fragments
                .iter()
                .map(|f| f.id as u32)
                .collect::<Vec<_>>();
            for index in indices.iter_mut() {
                let is_stale = index.fields.iter().any(|field_id| {
                    overlaid_field_versions
                        .get(field_id)
                        .is_some_and(|&overlay_version| overlay_version > index.dataset_version)
                });
                if is_stale && let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                    for new_id in &new_fragment_ids {
                        fragment_bitmap.remove(*new_id);
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

        // Fragment bitmaps record which fragments the index was originally built for.
        // Operations like updates and data replacement prune these bitmaps, and
        // effective_fragment_bitmap intersects with existing fragments at query time.

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
                    if let Some(oldest) = sorted_indices.first()
                        && !Self::is_vector_index(oldest)
                    {
                        uuids_to_keep.insert(oldest.uuid);
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
                    return Err(Error::invalid_input(
                        "The compaction plan included a rewrite group that was a split of indexed and non-indexed data",
                    ));
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
                return Err(Error::invalid_input(format!(
                    "An invalid compaction plan must have been generated because multiple tasks modified the same index: {}",
                    rewritten_index.old_id
                )));
            }

            // Skip indices that no longer exist (may have been removed by concurrent operation)
            let Some(index) = indices
                .iter_mut()
                .find(|idx| idx.uuid == rewritten_index.old_id)
            else {
                continue;
            };

            index.fragment_bitmap = Some(Self::recalculate_fragment_bitmap(
                index.fragment_bitmap.as_ref().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Cannot rewrite index {} which did not store fragment bitmap",
                        index.uuid
                    ))
                })?,
                groups,
            )?);
            index.uuid = rewritten_index.new_id;
            // Update file sizes to match the new index files. When not available
            // (e.g., from older writers), clear the old file sizes to avoid
            // using stale sizes from the pre-remap index.
            index.files = rewritten_index.new_index_files.clone();
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
                let start = final_fragments
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.id == group.old_fragments[0].id)
                    .ok_or_else(|| {
                        Error::commit_conflict_source(
                            version,
                            format!(
                                "dataset does not contain a fragment a rewrite operation wants to replace: id={}",
                                group.old_fragments[0].id
                            )
                            .into(),
                        )
                    })?
                    .0;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::RowIdMeta;
    use crate::format::overlay::OverlayCoverage;
    use crate::rowids::write_row_ids;
    use crate::transaction::test_support::{
        default_build_config, make_stable_row_id_manifest, overlay_with_field,
        sample_index_metadata, sample_manifest,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema as LanceSchema;
    use lance_file::version::LanceFileVersion;
    use lance_io::utils::CachedFileSize;
    use std::collections::HashMap;
    use std::sync::Arc;

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
    fn test_create_index_build_manifest_keeps_unremoved_same_name_indices() {
        let manifest = sample_manifest();
        let first_index = sample_index_metadata("vector_idx");
        let second_index = sample_index_metadata("vector_idx");
        let third_index = sample_index_metadata("vector_idx");

        let transaction = Transaction::new(
            manifest.version,
            Operation::CreateIndex {
                new_indices: vec![third_index.clone()],
                removed_indices: vec![second_index.clone()],
            },
            None,
        );

        let (_, final_indices) = transaction
            .build_manifest(
                Some(&manifest),
                vec![first_index.clone(), second_index.clone()],
                "txn",
                &default_build_config(),
            )
            .unwrap();

        assert_eq!(final_indices.len(), 2);
        assert!(final_indices.iter().any(|idx| idx.uuid == first_index.uuid));
        assert!(final_indices.iter().any(|idx| idx.uuid == third_index.uuid));
        assert!(
            !final_indices
                .iter()
                .any(|idx| idx.uuid == second_index.uuid)
        );
    }

    #[test]
    fn test_create_index_build_manifest_deduplicates_relisted_indices_by_uuid() {
        let manifest = sample_manifest();
        let first_index = sample_index_metadata("vector_idx");
        let second_index = sample_index_metadata("vector_idx");
        let third_index = sample_index_metadata("vector_idx");

        let transaction = Transaction::new(
            manifest.version,
            Operation::CreateIndex {
                new_indices: vec![first_index.clone(), third_index.clone()],
                removed_indices: vec![second_index.clone()],
            },
            None,
        );

        let (_, final_indices) = transaction
            .build_manifest(
                Some(&manifest),
                vec![first_index.clone(), second_index.clone()],
                "txn",
                &default_build_config(),
            )
            .unwrap();

        assert_eq!(final_indices.len(), 2);
        assert_eq!(
            final_indices
                .iter()
                .filter(|idx| idx.uuid == first_index.uuid)
                .count(),
            1
        );
        assert!(final_indices.iter().any(|idx| idx.uuid == third_index.uuid));
        assert!(
            !final_indices
                .iter()
                .any(|idx| idx.uuid == second_index.uuid)
        );
    }

    #[test]
    fn test_remove_tombstoned_data_files() {
        // Create a fragment with mixed data files: some normal, some fully tombstoned
        let mut fragment = Fragment::new(1);

        // Add a normal data file with valid field IDs
        fragment.files.push(DataFile {
            path: "normal.lance".to_string(),
            fields: Arc::from([1, 2, 3]),
            column_indices: Arc::from([]),
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(1000),
            base_id: None,
        });

        // Add a data file with all fields tombstoned
        fragment.files.push(DataFile {
            path: "all_tombstoned.lance".to_string(),
            fields: Arc::from([-2, -2, -2]),
            column_indices: Arc::from([]),
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(500),
            base_id: None,
        });

        // Add a data file with mixed tombstoned and valid fields
        fragment.files.push(DataFile {
            path: "mixed.lance".to_string(),
            fields: Arc::from([4, -2, 5]),
            column_indices: Arc::from([]),
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::new(750),
            base_id: None,
        });

        // Add another fully tombstoned file
        fragment.files.push(DataFile {
            path: "another_tombstoned.lance".to_string(),
            fields: Arc::from([-2_i32]),
            column_indices: Arc::from([]),
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
            files: None,
        }
    }

    fn create_system_index(name: &str, field_id: i32) -> IndexMetadata {
        use prost_types::Any;
        use std::sync::Arc;

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
            files: None,
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
        use crate::system_index::mem_wal::MEM_WAL_INDEX_NAME;

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
        assert!(
            indices
                .iter()
                .all(|idx| idx.dataset_version == 2 || idx.dataset_version == 4)
        );
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
        assert!(
            idx_c_indices
                .iter()
                .all(|idx| idx.dataset_version == 2 || idx.dataset_version == 3)
        );

        // Verify idx_d kept
        assert!(indices.iter().any(|idx| idx.name == "idx_d"));

        // Verify idx_e removed (bad field)
        assert!(!indices.iter().any(|idx| idx.name == "idx_e"));
    }

    #[test]
    fn test_handle_rewrite_indices_skips_missing_index() {
        // Create an empty indices list
        let mut indices = vec![];

        // Create rewritten_indices referring to a non-existent index
        let rewritten_indices = vec![RewrittenIndex {
            old_id: Uuid::new_v4(),
            new_id: Uuid::new_v4(),
            new_index_details: prost_types::Any {
                type_url: String::new(),
                value: vec![],
            },
            new_index_version: 1,
            new_index_files: None,
        }];

        // Should succeed (skip missing index) instead of error
        let result = Transaction::handle_rewrite_indices(&mut indices, &rewritten_indices, &[]);
        assert!(result.is_ok());
        assert!(indices.is_empty());
    }

    /// When a fragment has no existing last_updated_at_version_meta (None), a
    /// partial RewriteColumns refresh must leave it as None rather than fabricating
    /// prev_version for unmatched rows.
    #[test]
    fn test_partial_rewrite_skips_fragment_with_no_version_meta() {
        let row_ids = RowIdSequence::from([10u64, 11, 12, 13, 14].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids)));

        let (major, minor) = lance_file::version::LanceFileVersion::Stable.to_numbers();
        let data_file = DataFile::new("data.lance", vec![0], vec![0], major, minor, None, None);

        let fragment = Fragment {
            id: 1,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let manifest = make_stable_row_id_manifest(vec![fragment.clone()]);

        // Simulate a RewriteColumns update that matched offsets 1 and 3
        let off_map = HashMap::from([(1u64, RoaringBitmap::from_iter([1u32, 3]))]);
        let tx = Transaction::new(
            manifest.version,
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![fragment],
                new_fragments: vec![],
                fields_modified: vec![],
                compacted_sstables: vec![],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: Some(UpdatedFragmentOffsets(off_map)),
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert!(
            out.fragments[0].last_updated_at_version_meta.is_none(),
            "fragment with no prior version metadata must not have fabricated prev_version stamped on unmatched rows"
        );
    }

    #[test]
    fn merge_build_manifest_refreshes_last_updated_when_data_files_change_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use lance_file::version::LanceFileVersion;

        let (major, minor) = LanceFileVersion::Stable.to_numbers();
        let mk_file = |path: &str| DataFile::new(path, vec![0], vec![0], major, minor, None, None);

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        let row_ids = RowIdSequence::from([100u64, 101, 102, 103, 104].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids)));

        let prev_fragment = Fragment {
            id: 0,
            files: vec![mk_file("before.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let mut manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![prev_fragment.clone()]),
            DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.next_row_id = 100;

        let merged_fragment = Fragment {
            files: vec![mk_file("after.lance")],
            ..prev_fragment
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![merged_fragment],
                schema: lance_schema,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert_eq!(out.version, 2);
        let frag = &out.fragments[0];
        let seq = frag
            .last_updated_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        assert_eq!(seq.version_at(0).unwrap(), 2);
        assert_eq!(seq.version_at(4).unwrap(), 2);
    }

    #[test]
    fn merge_build_manifest_skips_refresh_when_carry_forward_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use crate::rowids::version::{RowDatasetVersionMeta, RowDatasetVersionSequence};
        use lance_file::version::LanceFileVersion;

        let (major, minor) = LanceFileVersion::Stable.to_numbers();
        let data_file = DataFile::new("same.lance", vec![0], vec![0], major, minor, None, None);

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        let row_ids = RowIdSequence::from([200u64, 201, 202, 203, 204].as_slice());
        let row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&row_ids)));

        let uniform_v1 = RowDatasetVersionSequence::from_uniform_row_count(5, 1);
        let meta_v1 = RowDatasetVersionMeta::from_sequence(&uniform_v1).unwrap();

        let prev_fragment = Fragment {
            id: 0,
            files: vec![data_file.clone()],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: row_id_meta.clone(),
            physical_rows: Some(5),
            last_updated_at_version_meta: Some(meta_v1.clone()),
            created_at_version_meta: None,
        };

        let mut manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![prev_fragment]),
            DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.next_row_id = 100;

        let merged_fragment = Fragment {
            id: 0,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta,
            physical_rows: Some(5),
            last_updated_at_version_meta: Some(meta_v1),
            created_at_version_meta: None,
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![merged_fragment],
                schema: lance_schema,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let seq = out.fragments[0]
            .last_updated_at_version_meta
            .as_ref()
            .unwrap()
            .load_sequence()
            .unwrap();
        assert_eq!(seq.version_at(0).unwrap(), 1);
        assert_eq!(seq.version_at(4).unwrap(), 1);
    }

    #[test]
    fn merge_build_manifest_no_last_updated_refresh_without_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use lance_file::version::LanceFileVersion;

        let (major, minor) = LanceFileVersion::Stable.to_numbers();
        let mk_file = |path: &str| DataFile::new(path, vec![0], vec![0], major, minor, None, None);

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        let prev_fragment = Fragment {
            id: 0,
            files: vec![mk_file("before.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(5),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![prev_fragment.clone()]),
            DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );
        assert_eq!(
            manifest.reader_feature_flags & FLAG_STABLE_ROW_IDS,
            0,
            "manifest must not use stable row IDs for this guard test"
        );

        let merged_fragment = Fragment {
            files: vec![mk_file("after.lance")],
            ..prev_fragment
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![merged_fragment],
                schema: lance_schema,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert!(
            out.fragments[0].last_updated_at_version_meta.is_none(),
            "without stable row IDs, Merge must not populate per-row last_updated metadata"
        );
    }

    #[test]
    fn merge_build_manifest_sets_both_version_meta_for_new_fragment_id_stable_row_ids() {
        use crate::feature_flags::FLAG_STABLE_ROW_IDS;
        use lance_file::version::LanceFileVersion;

        let (major, minor) = LanceFileVersion::Stable.to_numbers();
        let mk_file = |path: &str| DataFile::new(path, vec![0], vec![0], major, minor, None, None);

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let lance_schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // Existing fragment (id=0) with stable row IDs
        let row_ids_0 = RowIdSequence::from([10u64, 11, 12].as_slice());
        let existing_fragment = Fragment {
            id: 0,
            files: vec![mk_file("existing.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&row_ids_0))),
            physical_rows: Some(3),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let mut manifest = Manifest::new(
            lance_schema.clone(),
            Arc::new(vec![existing_fragment.clone()]),
            DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );
        manifest.reader_feature_flags |= FLAG_STABLE_ROW_IDS;
        manifest.next_row_id = 100;
        manifest.version = 1;

        // New fragment (id=1) not present in prev manifest — exercises the None branch
        let row_ids_1 = RowIdSequence::from([20u64, 21, 22, 23].as_slice());
        let new_fragment = Fragment {
            id: 1,
            files: vec![mk_file("new.lance")],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&row_ids_1))),
            physical_rows: Some(4),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let tx = Transaction::new(
            manifest.version,
            Operation::Merge {
                fragments: vec![existing_fragment, new_fragment],
                schema: lance_schema,
            },
            None,
        );

        let (out, _) = tx
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        assert_eq!(out.version, 2);

        let new_frag = out.fragments.iter().find(|f| f.id == 1).unwrap();

        // last_updated_at_version must be set to the commit version
        let last_updated_seq = new_frag
            .last_updated_at_version_meta
            .as_ref()
            .expect("new fragment must have last_updated_at_version_meta")
            .load_sequence()
            .unwrap();
        assert_eq!(last_updated_seq.version_at(0).unwrap(), 2);
        assert_eq!(last_updated_seq.version_at(3).unwrap(), 2);

        // created_at_version must also be set — must not be None
        let created_seq = new_frag
            .created_at_version_meta
            .as_ref()
            .expect("new fragment must have created_at_version_meta")
            .load_sequence()
            .unwrap();
        assert_eq!(created_seq.version_at(0).unwrap(), 2);
        assert_eq!(created_seq.version_at(3).unwrap(), 2);
    }

    // --- Proposal 1: range pre-filter ---

    // --- Proposal 2: version sequence cache ---

    #[test]
    fn test_prune_overlay_stale_fields_from_indices() {
        // Fragment 0 carried an overlay on field 1 committed at v5, and was
        // fully compacted into new fragment 7.
        let mut old_frag = Fragment::new(0);
        old_frag.overlays = vec![overlay_with_field(1, 5)];
        let groups = vec![RewriteGroup {
            old_fragments: vec![old_frag],
            new_fragments: vec![Fragment::new(7)],
        }];

        // Post-remap state: every index already covers the new fragment (7).
        let covering = || Some(RoaringBitmap::from_iter([7u32]));
        let mut indices = vec![
            // Stale: covers the overlaid field 1, built (v2) before the overlay.
            create_test_index("stale", 1, 2, covering(), false),
            // Not stale: covers field 1 but built at the overlay's version (v5);
            // `committed_version > dataset_version` is false at equality.
            create_test_index("fresh", 1, 5, covering(), false),
            // Unrelated: covers field 2, which the overlay never touched.
            create_test_index("unrelated", 2, 2, covering(), false),
        ];

        Transaction::prune_overlay_stale_fields_from_indices(&mut indices, &groups);

        assert!(
            !indices[0].fragment_bitmap.as_ref().unwrap().contains(7),
            "stale index must drop the rewritten fragment from its coverage"
        );
        assert!(
            indices[1].fragment_bitmap.as_ref().unwrap().contains(7),
            "an index built at/after the overlay is not stale"
        );
        assert!(
            indices[2].fragment_bitmap.as_ref().unwrap().contains(7),
            "an index on an un-overlaid field is unaffected"
        );
    }

    #[test]
    fn test_data_overlay_build_manifest_multi_fragment() {
        // Overlays targeting two distinct fragments are each applied and stamped.
        // A targeted fragment already carrying an overlay (committed at v3) gets
        // the new overlay appended and stamped while its existing overlay is
        // preserved, and a fragment the operation does not target is passed
        // through with its existing overlays untouched.
        let mut frag0 = Fragment::new(0);
        frag0.overlays = vec![overlay_with_field(5, 3)]; // targeted, pre-existing at v3
        let frag1 = Fragment::new(1);
        let mut frag2 = Fragment::new(2);
        frag2.overlays = vec![overlay_with_field(9, 3)]; // untargeted, committed at v3
        let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let mut manifest = Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(vec![frag0, frag1, frag2]),
            crate::format::DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );
        // The pre-existing overlays were committed at v3, so the current
        // manifest must be at least that version; the new commit then stamps
        // its overlay at v4, keeping the fragment's overlays newest-last.
        manifest.version = 3;

        let txn = Transaction::new(
            manifest.version,
            Operation::DataOverlay {
                groups: vec![
                    DataOverlayGroup {
                        fragment_id: 0,
                        overlays: vec![overlay_with_field(1, 0)],
                    },
                    DataOverlayGroup {
                        fragment_id: 1,
                        overlays: vec![overlay_with_field(2, 0)],
                    },
                ],
            },
            None,
        );

        let (result, _) = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let frag = |id: u64| {
            result
                .fragments
                .iter()
                .find(|f| f.id == id)
                .unwrap_or_else(|| panic!("fragment {id} missing from result"))
        };
        // The already-overlaid target keeps its v3 overlay and appends the new
        // one, stamped to the new version.
        assert_eq!(frag(0).overlays.len(), 2);
        assert_eq!(frag(0).overlays[0].committed_version, 3);
        assert_eq!(frag(0).overlays[1].committed_version, result.version);
        // The fresh target gets its overlay, stamped to the new version.
        assert_eq!(frag(1).overlays.len(), 1);
        assert_eq!(frag(1).overlays[0].committed_version, result.version);
        // The untargeted fragment is unchanged: same overlay, original version.
        assert_eq!(frag(2).overlays.len(), 1);
        assert_eq!(frag(2).overlays[0].committed_version, 3);
        assert!(result.version > manifest.version);
    }

    #[test]
    fn test_data_replacement_tombstones_overlaid_fields() {
        // A DataReplacement writing new base values for field 5 must stop any
        // overlay from shadowing those cells: field 5 is tombstoned in place
        // (preserving the overlay's field 3), and an overlay covering only field
        // 5 is dropped entirely.
        let mut fragment = Fragment::new(0);
        fragment.files = vec![
            DataFile::new_legacy_from_fields("f3.lance", vec![3], None),
            DataFile::new_legacy_from_fields("f5.lance", vec![5], None),
        ];
        fragment.overlays = vec![
            DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("o35.lance", vec![3, 5], None),
                coverage: OverlayCoverage::sparse(vec![
                    roaring::RoaringBitmap::from_iter([0u32]),
                    roaring::RoaringBitmap::from_iter([0u32]),
                ]),
                committed_version: 3,
            },
            DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields("o5.lance", vec![5], None),
                coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0u32])),
                committed_version: 3,
            },
        ];

        let schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        let manifest = Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(vec![fragment]),
            crate::format::DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );

        let txn = Transaction::new(
            manifest.version,
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(
                    0,
                    DataFile::new_legacy_from_fields("f5-new.lance", vec![5], None),
                )],
            },
            None,
        );

        let (result, _) = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let frag = &result.fragments[0];
        // The base data file for field 5 was swapped in.
        assert!(frag.files.iter().any(|f| f.path == "f5-new.lance"));
        // The [3, 5] overlay keeps field 3 and tombstones field 5; the [5]-only
        // overlay is dropped.
        assert_eq!(frag.overlays.len(), 1);
        assert_eq!(frag.overlays[0].data_file.fields.as_ref(), &[3, -2]);
    }

    #[test]
    fn test_data_overlay_build_manifest_merges_duplicate_groups() {
        // Two groups targeting the same fragment must both survive (a HashMap
        // collapse would have dropped the first).
        let manifest = sample_manifest();
        let txn = Transaction::new(
            manifest.version,
            Operation::DataOverlay {
                groups: vec![
                    DataOverlayGroup {
                        fragment_id: 0,
                        overlays: vec![overlay_with_field(1, 0)],
                    },
                    DataOverlayGroup {
                        fragment_id: 0,
                        overlays: vec![overlay_with_field(2, 0)],
                    },
                ],
            },
            None,
        );

        let (result, _) = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap();

        let overlays = &result.fragments[0].overlays;
        assert_eq!(overlays.len(), 2);
        assert_eq!(overlays[0].data_file.fields.as_ref(), [1i32].as_slice());
        assert_eq!(overlays[1].data_file.fields.as_ref(), [2i32].as_slice());
    }

    #[test]
    fn test_data_overlay_build_manifest_rejects_unknown_fragment() {
        let manifest = sample_manifest();
        let txn = Transaction::new(
            manifest.version,
            Operation::DataOverlay {
                groups: vec![DataOverlayGroup {
                    fragment_id: 99,
                    overlays: vec![overlay_with_field(1, 0)],
                }],
            },
            None,
        );
        let err = txn
            .build_manifest(Some(&manifest), vec![], "txn", &default_build_config())
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"), "{err}");
    }
}
