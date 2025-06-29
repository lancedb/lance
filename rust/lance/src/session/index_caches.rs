// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Caches for Lance indices. They are organized in a hierarchical manner to
//! avoid collisions.
//!
//!  GlobalIndexCache
//!     │
//!     ├─► DSIndexCache (prefixed by dataset URI)
//!     │    │
//!     └────┴──► Index-specific cache (prefixed by index UUID)

use std::{borrow::Cow, ops::Deref, sync::Arc};

use lance_core::cache::{CacheKey, LanceCache};
use lance_index::{
    frag_reuse::FragReuseIndex,
    scalar::{ScalarIndex, ScalarIndexType},
    vector::{VectorIndex, VectorIndexCacheEntry},
};
use lance_table::format::Index;

/// A type-safe wrapper around a LanceCache that enforces namespaces for index data.
pub struct GlobalIndexCache(pub(super) LanceCache);

impl GlobalIndexCache {
    pub fn for_dataset(&self, uri: &str) -> DSIndexCache {
        // Create a sub-cache for the dataset by adding the URI as a key prefix.
        // This prevents collisions between different datasets.
        DSIndexCache(self.0.with_key_prefix(uri))
    }
}

impl Clone for GlobalIndexCache {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Deref for GlobalIndexCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A type-safe wrapper around a LanceCache that enforces namespaces and keys
/// for dataset-specific index data.
pub struct DSIndexCache(pub(crate) LanceCache);

impl Deref for DSIndexCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DSIndexCache {
    /// Create an index-specific cache with the given UUID prefix.
    pub fn for_index(&self, uuid: &str) -> LanceCache {
        self.0.with_key_prefix(uuid)
    }
}

// Cache key types for type-safe cache access

#[derive(Debug)]
pub struct ScalarIndexKey<'a> {
    pub uuid: &'a str,
}

impl CacheKey for ScalarIndexKey<'_> {
    type ValueType = Arc<dyn ScalarIndex>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("scalar/{}", self.uuid))
    }
}

#[derive(Debug)]
pub struct VectorIndexKey<'a> {
    pub uuid: &'a str,
}

impl CacheKey for VectorIndexKey<'_> {
    type ValueType = Arc<dyn VectorIndex>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("vector/{}", self.uuid))
    }
}

#[derive(Debug)]
pub struct VectorPartitionKey<'a> {
    pub uuid: &'a str,
    pub partition: u32,
}

impl CacheKey for VectorPartitionKey<'_> {
    type ValueType = Arc<dyn VectorIndexCacheEntry>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("vector_partition/{}/{}", self.uuid, self.partition))
    }
}

#[derive(Debug)]
pub struct FragReuseIndexKey<'a> {
    pub uuid: &'a str,
}

impl CacheKey for FragReuseIndexKey<'_> {
    type ValueType = FragReuseIndex;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("frag_reuse/{}", self.uuid))
    }
}

#[derive(Debug)]
pub struct IndexMetadataKey<'a> {
    pub dataset_base_path: &'a str,
    pub version: u64,
}

impl CacheKey for IndexMetadataKey<'_> {
    type ValueType = Vec<Index>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "metadata/{}:{}",
            self.dataset_base_path, self.version
        ))
    }
}

#[derive(Debug)]
pub struct ScalarIndexTypeKey<'a> {
    pub uuid: &'a str,
}

impl CacheKey for ScalarIndexTypeKey<'_> {
    type ValueType = ScalarIndexType;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("type/{}", self.uuid))
    }
}
