// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Caches for Lance datasets. They are organized in a hierarchical manner to
//! avoid collisions.
//!
//!  GlobalMetadataCache
//!     │
//!     ├─► DSMetadataCache (prefixed by dataset URI)
//!     │    │
//!     └────┴──► FileMetadataCache (prefixed by file path)

use std::{borrow::Cow, ops::Deref};

use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{
    cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache},
    utils::deletion::DeletionVector,
};
use lance_select::RowAddrMask;
use lance_table::{
    format::{CellFlagRoot, DeletionFile, DeletionFileType, Manifest, RowIdMeta},
    rowids::{RowIdIndex, RowIdSequence},
};
use object_store::path::Path;
use roaring::RoaringBitmap;

use crate::dataset::transaction::Transaction;

/// A type-safe wrapper around a LanceCache that enforces namespaces for dataset metadata.
pub struct GlobalMetadataCache(pub(super) LanceCache);

impl GlobalMetadataCache {
    pub fn for_dataset(&self, uri: &str) -> DSMetadataCache {
        // Create a sub-cache for the dataset by adding the URI as a key prefix.
        // This prevents collisions between different datasets.
        DSMetadataCache(self.0.with_key_prefix(uri))
    }
}

impl Clone for GlobalMetadataCache {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl DeepSizeOf for GlobalMetadataCache {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// A type-safe wrapper around a LanceCache that enforces namespaces and keys
/// for dataset metadata.
pub struct DSMetadataCache(pub(crate) LanceCache);

impl Deref for DSMetadataCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Cache key types for type-safe cache access
#[derive(Debug)]
pub struct ManifestKey<'a> {
    pub version: u64,
    pub e_tag: Option<&'a str>,
}

impl CacheKey for ManifestKey<'_> {
    type ValueType = Manifest;
    fn key(&self) -> Cow<'_, str> {
        if let Some(e_tag) = self.e_tag {
            Cow::Owned(format!("manifest/{}/{}", self.version, e_tag))
        } else {
            Cow::Owned(format!("manifest/{}", self.version))
        }
    }
    fn type_name() -> &'static str {
        "Manifest"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.manifest-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
        if let Some(e_tag) = self.e_tag {
            builder.write_some();
            builder.write_str(e_tag);
        } else {
            builder.write_none();
        }
    }
}

#[derive(Debug)]
pub struct TransactionKey {
    pub version: u64,
}

/// Cache key for a validated immutable Cell Flag root in one snapshot.
///
/// The cache is already namespaced by the destination dataset URI. The source
/// URI is still part of the key because shallow clones may read immutable
/// cell-flag objects from more than one external dataset base.
#[derive(Debug)]
pub struct CellFlagRootKey {
    pub source_uri: String,
    pub path: String,
    pub size_bytes: u64,
    pub memory_size_bytes: u64,
    pub inline_hash: Option<[u8; 32]>,
}

impl CacheKey for CellFlagRootKey {
    type ValueType = CellFlagRoot;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "cell_flag/root/{}/{}/{}/{}",
            self.source_uri, self.path, self.size_bytes, self.memory_size_bytes
        ))
    }

    fn type_name() -> &'static str {
        "CellFlagRoot"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.cell-flag-root-key", 3)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(&self.source_uri);
        builder.write_str(&self.path);
        builder.write_u64(self.size_bytes);
        builder.write_u64(self.memory_size_bytes);
        if let Some(inline_hash) = self.inline_hash {
            builder.write_some();
            builder.write_fixed_bytes(&inline_hash);
        } else {
            builder.write_none();
        }
    }
}

/// Cache key for a validated immutable partial Cell Flag bitmap.
#[derive(Debug)]
pub struct CellFlagBitmapKey {
    pub source_uri: String,
    pub path: String,
    pub size_bytes: u64,
    pub memory_size_bytes: u64,
    pub flag_id: u32,
    pub fragment_id: u64,
    pub physical_rows: u64,
    pub inline_hash: Option<[u8; 32]>,
}

impl CacheKey for CellFlagBitmapKey {
    type ValueType = RoaringBitmap;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "cell_flag/bitmap/{}/{}/{}/{}/{}/{}",
            self.source_uri,
            self.path,
            self.flag_id,
            self.fragment_id,
            self.size_bytes,
            self.memory_size_bytes
        ))
    }

    fn type_name() -> &'static str {
        "CellFlagBitmap"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.cell-flag-bitmap-key", 3)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(&self.source_uri);
        builder.write_str(&self.path);
        builder.write_u64(self.size_bytes);
        builder.write_u64(self.memory_size_bytes);
        builder.write_u32(self.flag_id);
        builder.write_u64(self.fragment_id);
        builder.write_u64(self.physical_rows);
        if let Some(inline_hash) = self.inline_hash {
            builder.write_some();
            builder.write_fixed_bytes(&inline_hash);
        } else {
            builder.write_none();
        }
    }
}

impl CacheKey for TransactionKey {
    type ValueType = Transaction;
    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("txn/{}", self.version))
    }
    fn type_name() -> &'static str {
        "Transaction"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.transaction-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
    }
}

#[derive(Debug)]
pub struct DeletionFileKey<'a> {
    pub fragment_id: u64,
    pub deletion_file: &'a DeletionFile,
}

impl CacheKey for DeletionFileKey<'_> {
    type ValueType = DeletionVector;
    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "deletion/{}/{}/{}/{}",
            self.fragment_id,
            self.deletion_file.read_version,
            self.deletion_file.id,
            self.deletion_file.file_type.suffix()
        ))
    }
    fn type_name() -> &'static str {
        "DeletionVector"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.deletion-file-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.fragment_id);
        builder.write_u64(self.deletion_file.read_version);
        builder.write_u64(self.deletion_file.id);
        builder.write_variant(match &self.deletion_file.file_type {
            DeletionFileType::Array => 0,
            DeletionFileType::Bitmap => 1,
        });
        if let Some(base_id) = self.deletion_file.base_id {
            builder.write_some();
            builder.write_u32(base_id);
        } else {
            builder.write_none();
        }
    }
}

#[derive(Debug)]
pub struct RowAddrMaskKey {
    pub version: u64,
    /// `Some(hash)` when the mask is restricted to a fragment subset; `None`
    /// when it covers all fragments in the dataset. Two consumers that ask
    /// for different subsets must not poison each other's cache entry.
    pub restrict_hash: Option<u64>,
}

impl CacheKey for RowAddrMaskKey {
    type ValueType = RowAddrMask;
    fn key(&self) -> Cow<'_, str> {
        match self.restrict_hash {
            None => Cow::Owned(format!("row_addr_mask/{}", self.version)),
            Some(h) => Cow::Owned(format!("row_addr_mask/{}/{:x}", self.version, h)),
        }
    }
    fn type_name() -> &'static str {
        "RowAddrMask"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.row-address-mask-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
        if let Some(restrict_hash) = self.restrict_hash {
            builder.write_some();
            builder.write_u64(restrict_hash);
        } else {
            builder.write_none();
        }
    }
}

#[derive(Debug)]
pub struct RowIdIndexKey {
    pub version: u64,
}

impl CacheKey for RowIdIndexKey {
    type ValueType = RowIdIndex;
    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("row_id_index/{}", self.version))
    }
    fn type_name() -> &'static str {
        "RowIdIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.row-id-index-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
    }
}

#[derive(Debug)]
pub struct RowIdSequenceKey<'a> {
    pub fragment_id: u64,
    /// Where the sequence is stored. A fragment id alone is not enough: this
    /// cache is namespaced by dataset URI only (see
    /// [`GlobalMetadataCache::for_dataset`]), and a dataset dropped and
    /// recreated at the same URI restarts fragment ids at 0, so a reused id
    /// would otherwise be served the earlier generation's sequence (#7645).
    /// The `row_id_meta` differentiates generations of dataset fragments.
    ///
    /// Any operation that changes which row ids a fragment holds also writes it
    /// new `row_id_meta`, so generations stay distinct; operations that leave
    /// row ids alone (deletes, added columns) leave it untouched and keep
    /// hitting the cache.
    ///
    /// For inline metadata the identity of the sequence *is* its encoded bytes,
    /// so the key uses
    /// [`InlineRowIds::digest`](lance_table::format::InlineRowIds::digest),
    /// which those bytes memoize on first use — an array-encoded sequence is
    /// 8 bytes per row, too much to rehash on every lookup.
    pub row_id_meta: &'a RowIdMeta,
}

impl CacheKey for RowIdSequenceKey<'_> {
    type ValueType = RowIdSequence;
    // Only the legacy display form. Identity comes from `write_key` below.
    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("row_id_sequence/{}", self.fragment_id))
    }
    fn type_name() -> &'static str {
        "RowIdSequence"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.row-id-sequence-key", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.fragment_id);
        match self.row_id_meta {
            RowIdMeta::Inline(data) => {
                builder.write_variant(0);
                builder.write_fixed_bytes(data.digest());
            }
            RowIdMeta::External(file) => {
                builder.write_variant(1);
                builder.write_str(&file.path);
                builder.write_u64(file.offset);
                builder.write_u64(file.size);
            }
        }
    }
}

impl DSMetadataCache {
    /// Create a file-specific metadata cache with the given prefix.
    /// This is used by file readers and other components that need file-level caching.
    pub(crate) fn file_metadata_cache(&self, prefix: &Path) -> LanceCache {
        self.0.with_key_prefix(prefix.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_table::format::ExternalFile;
    use lance_table::rowids::write_row_ids;

    use super::*;

    #[tokio::test]
    async fn cell_flag_cache_keys_separate_declared_memory_sizes() {
        let cache = LanceCache::with_capacity(4096);
        let root_key = |memory_size_bytes| CellFlagRootKey {
            source_uri: "memory://dataset".to_string(),
            path: "_cell_flags/roots/0/root.root".to_string(),
            size_bytes: 128,
            memory_size_bytes,
            inline_hash: None,
        };
        cache
            .insert_with_key(
                &root_key(256),
                Arc::new(CellFlagRoot {
                    fragments: Vec::new(),
                }),
            )
            .await;
        assert_ne!(root_key(256).key(), root_key(512).key());
        assert!(cache.get_with_key(&root_key(512)).await.is_none());

        let bitmap_key = |memory_size_bytes| CellFlagBitmapKey {
            source_uri: "memory://dataset".to_string(),
            path: "_cell_flags/bitmaps/0/0/bitmap.rbm".to_string(),
            size_bytes: 64,
            memory_size_bytes,
            flag_id: 0,
            fragment_id: 0,
            physical_rows: 10,
            inline_hash: None,
        };
        cache
            .insert_with_key(&bitmap_key(128), Arc::new(RoaringBitmap::from_iter([1, 3])))
            .await;
        assert_ne!(bitmap_key(128).key(), bitmap_key(256).key());
        assert!(cache.get_with_key(&bitmap_key(256)).await.is_none());
    }

    #[tokio::test]
    async fn deletion_file_key_separates_storage_bases() {
        let cache = LanceCache::with_capacity(4096);
        let deletion_file = DeletionFile {
            read_version: 3,
            id: 4,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: Some(1),
            base_id: None,
        };
        cache
            .insert_with_key(
                &DeletionFileKey {
                    fragment_id: 2,
                    deletion_file: &deletion_file,
                },
                Arc::new(DeletionVector::NoDeletions),
            )
            .await;

        let deletion_file_on_other_base = DeletionFile {
            base_id: Some(7),
            ..deletion_file
        };
        assert!(
            cache
                .get_with_key(&DeletionFileKey {
                    fragment_id: 2,
                    deletion_file: &deletion_file_on_other_base,
                })
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn row_id_sequence_key_separates_fragment_generations() {
        // A dataset dropped and recreated at the same URI restarts fragment ids,
        // so the same id must not resolve to the earlier generation's sequence.
        let cache = LanceCache::with_capacity(4096);
        let first_generation = RowIdMeta::Inline(write_row_ids(&(0..100).into()).into());
        let key = RowIdSequenceKey {
            fragment_id: 0,
            row_id_meta: &first_generation,
        };
        cache
            .insert_with_key(&key, Arc::new(RowIdSequence::from(0..100)))
            .await;
        assert!(cache.get_with_key(&key).await.is_some());

        let second_generation = RowIdMeta::Inline(write_row_ids(&(100..160).into()).into());
        assert!(
            cache
                .get_with_key(&RowIdSequenceKey {
                    fragment_id: 0,
                    row_id_meta: &second_generation,
                })
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn row_id_sequence_key_separates_external_slices() {
        // External metadata is a read-only legacy shape, but the same slice of
        // the same file is the only thing that may share a cache entry.
        let cache = LanceCache::with_capacity(4096);
        let external = |offset| {
            RowIdMeta::External(ExternalFile {
                path: "_row_ids/1.rowids".into(),
                offset,
                size: 16,
            })
        };
        let first_slice = external(0);
        cache
            .insert_with_key(
                &RowIdSequenceKey {
                    fragment_id: 0,
                    row_id_meta: &first_slice,
                },
                Arc::new(RowIdSequence::from(0..100)),
            )
            .await;

        let second_slice = external(16);
        assert!(
            cache
                .get_with_key(&RowIdSequenceKey {
                    fragment_id: 0,
                    row_id_meta: &second_slice,
                })
                .await
                .is_none()
        );
        // An inline sequence never aliases an external one.
        let inline = RowIdMeta::Inline(write_row_ids(&(0..100).into()).into());
        assert!(
            cache
                .get_with_key(&RowIdSequenceKey {
                    fragment_id: 0,
                    row_id_meta: &inline,
                })
                .await
                .is_none()
        );
    }
}
