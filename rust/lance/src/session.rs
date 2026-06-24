// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::cache::{CacheBackend, CacheKeyIterator, LanceCache};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_index::IndexType;
use lance_io::object_store::ObjectStoreRegistry;
use lance_io::spill::{LocalSpillStore, SpillStore};

use crate::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use crate::session::caches::GlobalMetadataCache;
use crate::session::index_caches::GlobalIndexCache;

use self::index_extension::IndexExtension;

pub(crate) mod caches;
pub mod index_caches;
pub(crate) mod index_extension;

/// A user session holds the runtime state for a [`crate::Dataset`]
///
/// A session will be created automatically when a Dataset is opened.  However, you
/// can manually create the session and provide it to the Dataset builder in order
/// to share runtime state between multiple datasets.
///
/// This can be used to share caches between multiple datasets, increasing the hit
/// rate and reducing the amount of memory used.
///
/// A session contains two different caches:
///  - The index cache is used to cache opened indices and will cache index data
///  - The metadata cache is used to cache a variety of dataset metadata (more
///    details can be found in the [performance guide](https://lance.org/guide/performance/)
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

    spill_store: Arc<dyn SpillStore>,
}

impl DeepSizeOf for Session {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
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
                &format!("LanceCache(items={})", self.metadata_cache.0.approx_size(),),
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
            spill_store: Arc::new(LocalSpillStore::default()),
        }
    }

    /// Create a session with a custom index cache backend.
    ///
    /// The provided backend will be used for caching index data. The metadata
    /// cache will use the default Moka-based backend with the given capacity.
    pub fn with_index_cache_backend(
        index_cache_backend: Arc<dyn CacheBackend>,
        metadata_cache_size: usize,
        store_registry: Arc<ObjectStoreRegistry>,
    ) -> Self {
        Self {
            index_cache: GlobalIndexCache(LanceCache::with_backend(index_cache_backend)),
            metadata_cache: GlobalMetadataCache(LanceCache::with_capacity(metadata_cache_size)),
            index_extensions: HashMap::new(),
            store_registry,
            spill_store: Arc::new(LocalSpillStore::default()),
        }
    }

    /// Replace the spill store used by this session.
    ///
    /// This is a builder-style method that consumes and returns `self`, making
    /// it easy to chain during session construction:
    ///
    /// ```rust,no_run
    /// # use lance::session::Session;
    /// # use lance_io::spill::LocalSpillStore;
    /// # use std::sync::Arc;
    /// let session = Session::default()
    ///     .with_spill_store(Arc::new(LocalSpillStore::with_cap(1 << 30).unwrap()));
    /// ```
    pub fn with_spill_store(mut self, store: Arc<dyn SpillStore>) -> Self {
        self.spill_store = store;
        self
    }

    /// Return a reference to the session's spill store.
    ///
    /// Callers use this to obtain reclaimable scratch space for intermediate
    /// state that overflows memory (e.g. index builders).
    pub fn spill_store(&self) -> &dyn SpillStore {
        &*self.spill_store
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
                    return Err(Error::invalid_input(format!(
                        "{name} is already registered"
                    )));
                }

                if let Some(ext) = extension.to_vector() {
                    self.index_extensions
                        .insert((IndexType::Vector, name), ext.to_generic());
                } else {
                    return Err(Error::invalid_input(format!(
                        "{name} is not a vector index extension"
                    )));
                }
            }
            _ => {
                return Err(Error::invalid_input(format!(
                    "scalar index extension is not support yet: {}",
                    extension.index_type()
                )));
            }
        }

        Ok(())
    }

    /// Return the current size of the session in bytes
    ///
    /// Keep in mind that this is not trivial to compute, as we will need to walk the caches
    pub fn size_bytes(&self) -> u64 {
        // We re-expose deep_size_of here so that users don't
        // need the deepsize crate themselves (e.g. to use deep_size_of)
        self.deep_size_of() as u64
    }

    /// Get the approximate number of items in the session.
    ///
    /// This is a rough estimate of the number of items in the session.  It is not
    /// exact and is not guaranteed to be accurate.
    pub fn approx_num_items(&self) -> usize {
        self.index_cache.0.approx_size()
            + self.metadata_cache.0.approx_size()
            + self.index_extensions.len()
    }

    /// Get the object store registry.
    pub fn store_registry(&self) -> Arc<ObjectStoreRegistry> {
        self.store_registry.clone()
    }

    /// Get a reference to the raw metadata cache (for use in index reconstruction).
    pub fn file_metadata_cache(&self) -> &LanceCache {
        &self.metadata_cache.0
    }

    /// Fetch statistics for the metadata cache
    pub async fn metadata_cache_stats(&self) -> lance_core::cache::CacheStats {
        self.metadata_cache.0.stats().await
    }

    /// Fetch statistics for the index cache
    pub async fn index_cache_stats(&self) -> lance_core::cache::CacheStats {
        self.index_cache.0.stats().await
    }

    /// Return an iterator over keys currently held by the index cache.
    ///
    /// Returns `None` when the index cache backend does not support key
    /// inventory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lance::session::Session;
    /// # async fn example() {
    /// let session = Session::default();
    /// let keys = session.index_cache_keys().await;
    /// assert!(keys.is_some());
    /// # }
    /// ```
    pub async fn index_cache_keys(&self) -> Option<CacheKeyIterator<'_>> {
        self.index_cache.0.keys().await
    }

    /// Return an iterator over keys currently held by the metadata cache.
    ///
    /// Returns `None` when the metadata cache backend does not support key
    /// inventory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lance::session::Session;
    /// # async fn example() {
    /// let session = Session::default();
    /// let keys = session.metadata_cache_keys().await;
    /// assert!(keys.is_some());
    /// # }
    /// ```
    pub async fn metadata_cache_keys(&self) -> Option<CacheKeyIterator<'_>> {
        self.metadata_cache.0.keys().await
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
    use lance_core::cache::{CacheKey, UnsizedCacheKey};
    use lance_index::vector::VectorIndex;
    use std::borrow::Cow;
    use tokio::io::AsyncWriteExt;

    struct TestKey(&'static str);
    impl CacheKey for TestKey {
        type ValueType = Vec<i32>;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }

        fn type_name() -> &'static str {
            "TestVec"
        }
    }

    struct TestUnsizedKey(&'static str);
    impl UnsizedCacheKey for TestUnsizedKey {
        type ValueType = dyn VectorIndex;
        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }

        fn type_name() -> &'static str {
            "TestUnsized"
        }
    }

    #[tokio::test]
    async fn test_disable_index_cache() {
        let no_cache = Session::new(0, 0, Default::default());
        assert!(
            no_cache
                .index_cache
                .get_unsized_with_key(&TestUnsizedKey("abc"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_session_cache_keys() {
        let session = Session::new(10_000, 10_000, Default::default());

        session
            .index_cache
            .insert_with_key(&TestKey("index-key"), Arc::new(vec![1]))
            .await;
        session
            .metadata_cache
            .0
            .insert_with_key(&TestKey("metadata-key"), Arc::new(vec![2]))
            .await;

        let index_keys = session
            .index_cache_keys()
            .await
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(index_keys.len(), 1);
        assert_eq!(index_keys[0].prefix(), "");
        assert_eq!(index_keys[0].key(), "index-key");
        assert_eq!(index_keys[0].type_name(), "TestVec");

        let metadata_keys = session
            .metadata_cache_keys()
            .await
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(metadata_keys.len(), 1);
        assert_eq!(metadata_keys[0].prefix(), "");
        assert_eq!(metadata_keys[0].key(), "metadata-key");
        assert_eq!(metadata_keys[0].type_name(), "TestVec");

        assert_ne!(index_keys, metadata_keys);
    }

    #[tokio::test]
    async fn test_default_session_has_spill_store() {
        let session = Session::default();
        // Should be able to allocate a spill and write to it without error.
        let (mut writer, _spill) = session.spill_store().new_spill().await.unwrap();
        writer.write_all(b"scratch").await.unwrap();
        lance_io::traits::Writer::shutdown(writer.as_mut())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_custom_spill_store_injected() {
        let capped = Arc::new(LocalSpillStore::with_cap(50).unwrap());
        let session = Session::default().with_spill_store(capped);

        let (mut writer, _spill) = session.spill_store().new_spill().await.unwrap();
        // Writing 51 bytes exceeds the 50-byte cap; the typed error is wrapped
        // in an io::Error by the writer and recovered on conversion.
        let io_err = writer.write_all(&[0u8; 51]).await.unwrap_err();
        let err: lance_core::Error = io_err.into();
        assert!(
            matches!(
                err,
                lance_core::Error::DiskCapExceeded { cap_bytes: 50, .. }
            ),
            "expected DiskCapExceeded, got {err}"
        );
    }
}
