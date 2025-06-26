// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Caches for Lance datasets. They are organized in a hierarchical manner to
//! avoid collisions.d
//!
//!  GlobalMetadataCache
//!     │
//!     ├─► DSMetadataCache (prefixed by dataset URI)
//!     │    │
//!     └────┴──► FileMetadataCache (prefixed by file path)

use std::{borrow::Cow, ops::Deref};

use lance_core::{
    cache::{CacheKey, LanceCache},
    utils::{deletion::DeletionVector, mask::RowIdMask},
};
use lance_table::{
    format::{DeletionFile, Manifest},
    rowids::{RowIdIndex, RowIdSequence},
};
use object_store::path::Path;

use crate::dataset::transaction::Transaction;

/// A type-safe wrapper around a LanceCache that enforces namespaces for dataset metadata.
pub struct GlobalMetadataCache(pub(super) LanceCache);

impl GlobalMetadataCache {
    pub fn for_dataset(&self, uri: &str) -> DSMetadataCache {
        // Create a sub-cache for the dataset by adding the URI as a key prefix.
        // This prevents collisions between different datasets.
        DSMetadataCache(self.0.with_key_prefix(uri))
    }

    /// Create a file-specific metadata cache with the given prefix.
    /// This is used by file readers and other components that need file-level caching.
    pub(crate) fn file_metadata_cache(&self, path: &Path) -> LanceCache {
        self.0.with_key_prefix(path.as_ref())
    }
}

impl Clone for GlobalMetadataCache {
    fn clone(&self) -> Self {
        Self(self.0.clone())
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
}

#[derive(Debug)]
pub struct TransactionKey {
    pub version: u64,
}

impl CacheKey for TransactionKey {
    type ValueType = Transaction;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("txn/{}", self.version))
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
}

#[derive(Debug)]
pub struct RowIdMaskKey {
    pub version: u64,
}

impl CacheKey for RowIdMaskKey {
    type ValueType = RowIdMask;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("row_id_mask/{}", self.version))
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
}

#[derive(Debug)]
pub struct RowIdSequenceKey {
    pub fragment_id: u64,
}

impl CacheKey for RowIdSequenceKey {
    type ValueType = RowIdSequence;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("row_id_sequence/{}", self.fragment_id))
    }
}

impl DSMetadataCache {
    /// Create a file-specific metadata cache with the given prefix.
    /// This is used by file readers and other components that need file-level caching.
    pub(crate) fn file_metadata_cache(&self, prefix: &Path) -> LanceCache {
        self.0.with_key_prefix(prefix.as_ref())
    }
}
