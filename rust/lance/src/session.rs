// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use deepsize::DeepSizeOf;
use lance_core::cache::LanceCache;
use lance_core::{Error, Result};
use lance_index::IndexType;
use lance_io::object_store::ObjectStoreRegistry;
use snafu::location;

use crate::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use crate::session::caches::GlobalMetadataCache;
use crate::session::index_caches::GlobalIndexCache;

use self::index_extension::IndexExtension;

pub(crate) mod caches;
pub(crate) mod index_caches;
pub mod index_extension;

/// A user session tracks the runtime state.
#[derive(Clone)]
pub struct Session {
    /// Global cache for opened indices.
    ///
    /// Sub-caches are created from this cache for each dataset by adding the
    /// URI and index UUID as a key prefix. If there is a fragment re-use index,
    /// that is also in the key prefix. This prevents collisions between different
    /// datasets and indices.
    pub(crate) index_cache: GlobalIndexCache,

    /// Global cache for file metadata.
    ///
    /// Sub-caches are created from this cache for each dataset by adding the
    /// URI as a key prefix. See the [`LanceDataset::metadata_cache`] field.
    /// This prevents collisions between different datasets.
    pub(crate) metadata_cache: caches::GlobalMetadataCache,

    pub(crate) index_extensions: HashMap<(IndexType, String), Arc<dyn IndexExtension>>,

    store_registry: Arc<ObjectStoreRegistry>,
}

impl DeepSizeOf for Session {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        let mut size = 0;
        // Measure the actual cache contents through the wrapper types
        size += self.index_cache.deep_size_of_children(context);
        size += self.metadata_cache.deep_size_of_children(context);
        for ext in self.index_extensions.values() {
            size += ext.deep_size_of_children(context);
        }
        size
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field(
                "index_cache",
                &format!("IndexCache(items={})", self.index_cache.0.approx_size(),),
            )
            .field(
                "file_metadata_cache",
                &format!(
                    "LanceCache(items={}, size_bytes={})",
                    self.metadata_cache.0.approx_size(),
                    self.metadata_cache.0.approx_size_bytes(),
                ),
            )
            .field(
                "index_extensions",
                &self.index_extensions.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Session {
    /// Create a new session.
    ///
    /// Parameters:
    ///
    /// - ***index_cache_size***: the size of the index cache.
    /// - ***metadata_cache_size***: the size of the metadata cache.
    /// - ***store_registry***: the object store registry to use when opening
    ///   datasets. This determines which schemes are available, and also allows
    ///   re-using object stores.
    pub fn new(
        index_cache_size: usize,
        metadata_cache_size: usize,
        store_registry: Arc<ObjectStoreRegistry>,
    ) -> Self {
        Self {
            index_cache: GlobalIndexCache(LanceCache::with_capacity(index_cache_size)),
            metadata_cache: GlobalMetadataCache(LanceCache::with_capacity(metadata_cache_size)),
            index_extensions: HashMap::new(),
            store_registry,
        }
    }

    /// Register a new index extension.
    ///
    /// A name can only be registered once per type of index extension.
    ///
    /// Parameters:
    ///
    /// - ***name***: the name of the extension.
    /// - ***extension***: the extension to register.
    pub fn register_index_extension(
        &mut self,
        name: String,
        extension: Arc<dyn IndexExtension>,
    ) -> Result<()> {
        match extension.index_type() {
            IndexType::Vector => {
                if self
                    .index_extensions
                    .contains_key(&(IndexType::Vector, name.clone()))
                {
                    return Err(Error::invalid_input(
                        format!("{name} is already registered"),
                        location!(),
                    ));
                }

                if let Some(ext) = extension.to_vector() {
                    self.index_extensions
                        .insert((IndexType::Vector, name), ext.to_generic());
                } else {
                    return Err(Error::invalid_input(
                        format!("{name} is not a vector index extension"),
                        location!(),
                    ));
                }
            }
            _ => {
                return Err(Error::invalid_input(
                    format!(
                        "scalar index extension is not support yet: {}",
                        extension.index_type()
                    ),
                    location!(),
                ));
            }
        }

        Ok(())
    }

    /// Return the current size of the session in bytes
    pub fn size_bytes(&self) -> u64 {
        // We re-expose deep_size_of here so that users don't
        // need the deepsize crate themselves (e.g. to use deep_size_of)
        self.deep_size_of() as u64
    }

    pub fn approx_num_items(&self) -> usize {
        self.index_cache.0.approx_size()
            + self.metadata_cache.0.approx_size()
            + self.index_extensions.len()
    }

    /// Get the object store registry.
    pub fn store_registry(&self) -> Arc<ObjectStoreRegistry> {
        self.store_registry.clone()
    }

    pub async fn metadata_cache_stats(&self) -> lance_core::cache::CacheStats {
        self.metadata_cache.0.stats().await
    }

    pub async fn index_cache_stats(&self) -> lance_core::cache::CacheStats {
        self.index_cache.0.stats().await
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new(
            DEFAULT_INDEX_CACHE_SIZE,
            DEFAULT_METADATA_CACHE_SIZE,
            Arc::new(ObjectStoreRegistry::default()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_index::vector::VectorIndex;

    #[tokio::test]
    async fn test_disable_index_cache() {
        let no_cache = Session::new(0, 0, Default::default());
        assert!(no_cache
            .index_cache
            .get_unsized::<dyn VectorIndex>("abc")
            .await
            .is_none());
    }
}
