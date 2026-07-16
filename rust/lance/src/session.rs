// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use lance_core::cache::{
    CacheBackend, CacheEntryRecord, CacheKeyIterator, InternalCacheKey, LanceCache,
};
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

/// Summary of entries currently held by one cache.
///
/// This is intended for diagnostics. It is computed from a weakly consistent
/// snapshot of backend entry inventory and may race with concurrent cache
/// insertions or evictions. A summary has total counts and sizes, grouped into
/// [`CacheGroupSummary`] values, and each group contains
/// [`CacheComponentSummary`] values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheSummary {
    /// Number of entries included in this summary.
    pub total_entries: usize,
    /// Total backend-accounted size in bytes for entries included here.
    pub total_size_bytes: usize,
    /// Entries grouped by summary label.
    pub groups: Vec<CacheGroupSummary>,
}

/// Summary of cache entries under one summary group.
///
/// In [`CacheSummary::groups`], index summaries use one group per Lance cache
/// prefix while metadata summaries may use a synthetic group label to keep
/// diagnostics compact.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheGroupSummary {
    /// Group label.
    ///
    /// Index summaries use the Lance cache prefix. Metadata summaries may use
    /// a synthetic label to keep diagnostics compact.
    pub cache_prefix: String,
    /// Number of entries under this group.
    pub entry_count: usize,
    /// Total backend-accounted size in bytes under this group.
    pub size_bytes: usize,
    /// Entries further grouped by component and value type.
    pub components: Vec<CacheComponentSummary>,
}

/// Summary of one cache component within a cache summary group.
///
/// Components are compact key-derived labels paired with a value type. This
/// keeps common key families, such as IVF partitions or manifest entries,
/// visible without returning every cache key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheComponentSummary {
    /// Compact component name derived from the cache key.
    pub component: String,
    /// Lance cache value type name.
    pub type_name: String,
    /// Number of entries for this component and value type.
    pub entry_count: usize,
    /// Total backend-accounted size in bytes for this component and value type.
    pub size_bytes: usize,
}

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

    /// Return a compact read-time summary of entries held by the index cache.
    ///
    /// Returns `None` when the configured index cache backend does not support
    /// entry inventory.
    ///
    /// ```
    /// # use lance::session::Session;
    /// # async fn example() -> lance::Result<()> {
    /// let session = Session::default();
    /// if let Some(summary) = session.index_cache_summary().await? {
    ///     for group in &summary.groups {
    ///         for component in &group.components {
    ///             let _ = (group.cache_prefix.as_str(), component.component.as_str());
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn index_cache_summary(&self) -> Result<Option<CacheSummary>> {
        let Some(records) = self.index_cache.0.entry_records().await else {
            return Ok(None);
        };

        summarize_cache_records(records, cache_prefix_group, index_cache_component).map(Some)
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

    /// Return a compact read-time summary of entries held by the metadata cache.
    ///
    /// Returns `None` when the configured metadata cache backend does not
    /// support entry inventory.
    ///
    /// ```
    /// # use lance::session::Session;
    /// # async fn example() -> lance::Result<()> {
    /// let session = Session::default();
    /// if let Some(summary) = session.metadata_cache_summary().await? {
    ///     let total_bytes = summary.total_size_bytes;
    ///     let total_entries = summary.total_entries;
    ///     let _ = (total_bytes, total_entries);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn metadata_cache_summary(&self) -> Result<Option<CacheSummary>> {
        let Some(records) = self.metadata_cache.0.entry_records().await else {
            return Ok(None);
        };

        summarize_cache_records(records, metadata_cache_group, metadata_cache_component).map(Some)
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

#[derive(Default)]
struct CacheGroupAccumulator {
    entry_count: usize,
    size_bytes: usize,
    components: BTreeMap<(String, String), CacheComponentSummary>,
}

fn summarize_cache_records(
    records: impl Iterator<Item = CacheEntryRecord>,
    group_name: fn(&InternalCacheKey) -> String,
    component_name: fn(&InternalCacheKey) -> String,
) -> Result<CacheSummary> {
    let mut groups = BTreeMap::<String, CacheGroupAccumulator>::new();
    let mut total_entries = 0usize;
    let mut total_size_bytes = 0usize;

    for record in records {
        total_entries = checked_cache_summary_add(total_entries, 1, "total entry count")?;
        total_size_bytes =
            checked_cache_summary_add(total_size_bytes, record.size_bytes, "total size bytes")?;

        let group_label = group_name(&record.key);
        let group = groups.entry(group_label.clone()).or_default();
        group.entry_count = checked_cache_summary_add(
            group.entry_count,
            1,
            format!("entry count for cache group '{group_label}'"),
        )?;
        group.size_bytes = checked_cache_summary_add(
            group.size_bytes,
            record.size_bytes,
            format!("size bytes for cache group '{group_label}'"),
        )?;

        let component = component_name(&record.key);
        let type_name = record.key.type_name().to_string();
        let component_label = format!("{component}/{type_name}");
        let component_summary = group
            .components
            .entry((component.clone(), type_name.clone()))
            .or_insert_with(|| CacheComponentSummary {
                component,
                type_name,
                entry_count: 0,
                size_bytes: 0,
            });
        component_summary.entry_count = checked_cache_summary_add(
            component_summary.entry_count,
            1,
            format!("entry count for cache component '{component_label}'"),
        )?;
        component_summary.size_bytes = checked_cache_summary_add(
            component_summary.size_bytes,
            record.size_bytes,
            format!("size bytes for cache component '{component_label}'"),
        )?;
    }

    Ok(CacheSummary {
        total_entries,
        total_size_bytes,
        groups: groups
            .into_iter()
            .map(|(cache_prefix, group)| CacheGroupSummary {
                cache_prefix,
                entry_count: group.entry_count,
                size_bytes: group.size_bytes,
                components: group.components.into_values().collect(),
            })
            .collect(),
    })
}

fn checked_cache_summary_add(
    current: usize,
    increment: usize,
    context: impl AsRef<str>,
) -> Result<usize> {
    current.checked_add(increment).ok_or_else(|| {
        Error::invalid_input(format!(
            "Cache summary overflow while adding {increment} to {} (current value: {current})",
            context.as_ref()
        ))
    })
}

fn cache_prefix_group(key: &InternalCacheKey) -> String {
    key.prefix().trim_end_matches('/').to_string()
}

fn metadata_cache_group(_key: &InternalCacheKey) -> String {
    "<metadata>".to_string()
}

fn index_cache_component(key: &InternalCacheKey) -> String {
    cache_key_component(key.key(), key.type_name())
}

fn metadata_cache_component(key: &InternalCacheKey) -> String {
    cache_key_component(key.key(), key.type_name())
}

fn cache_key_component(key: &str, type_name: &str) -> String {
    let has_path_separator = key.contains('/');
    let key = key.split('/').next().unwrap_or_default();
    if key.is_empty() {
        return "<empty>".to_string();
    }
    if type_name == "Vec<IndexMetadata>" && key.chars().all(|ch| ch.is_ascii_digit()) {
        return "version".to_string();
    }
    if let Some((column_index, view_tag)) = key.split_once(':')
        && !column_index.is_empty()
        && !view_tag.is_empty()
        && column_index.chars().all(|ch| ch.is_ascii_digit())
    {
        return "field_data".to_string();
    }

    for prefix in [
        "ivf-",
        "page-",
        "postings-",
        "posting-list-",
        "posting-metadata-",
        "positions-",
    ] {
        if key.starts_with(prefix) {
            return prefix.trim_end_matches('-').to_string();
        }
    }

    if has_path_separator {
        return key.to_string();
    }

    type_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::Future;
    use lance_core::cache::{
        CacheBackend, CacheCodec, CacheEntry, CacheKey, InternalCacheKey, UnsizedCacheKey,
    };
    use lance_index::vector::VectorIndex;
    use object_store::path::Path;
    use rstest::rstest;
    use std::borrow::Cow;
    use std::pin::Pin;
    use tokio::io::AsyncWriteExt;
    use uuid::Uuid;

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

    #[derive(Debug)]
    struct NoInventoryBackend;

    #[async_trait]
    impl CacheBackend for NoInventoryBackend {
        async fn get(
            &self,
            _key: &InternalCacheKey,
            _codec: Option<CacheCodec>,
        ) -> Option<CacheEntry> {
            None
        }

        async fn insert(
            &self,
            _key: &InternalCacheKey,
            _entry: CacheEntry,
            _size_bytes: usize,
            _codec: Option<CacheCodec>,
        ) {
        }

        async fn get_or_insert<'a>(
            &self,
            _key: &InternalCacheKey,
            loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
            _codec: Option<CacheCodec>,
        ) -> Result<(CacheEntry, bool)> {
            let (entry, _) = loader.await?;
            Ok((entry, false))
        }

        async fn invalidate_prefix(&self, _prefix: &str) {}

        async fn clear(&self) {}

        async fn num_entries(&self) -> usize {
            0
        }

        async fn size_bytes(&self) -> usize {
            0
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
    async fn test_session_cache_summaries() {
        let session = Session::new(10_000, 10_000, Default::default());
        let index_uuid = Uuid::nil();

        let ds_index_cache = session.index_cache.for_dataset("memory://");
        let index_cache = ds_index_cache.for_index(&index_uuid, None);
        index_cache
            .insert_with_key(&TestKey("ivf-0"), Arc::new(vec![1]))
            .await;
        index_cache
            .insert_with_key(&TestKey("ivf-1"), Arc::new(vec![2]))
            .await;
        index_cache
            .insert_with_key(&TestKey("page-0"), Arc::new(vec![3]))
            .await;

        let index_summary = session.index_cache_summary().await.unwrap().unwrap();
        assert_eq!(index_summary.total_entries, 3);
        assert_eq!(index_summary.groups.len(), 1);
        let index_group = &index_summary.groups[0];
        assert_eq!(index_group.cache_prefix, format!("memory:///{index_uuid}"));
        assert_eq!(index_group.entry_count, 3);
        assert_eq!(index_group.components.len(), 2);
        assert_eq!(index_group.components[0].component, "ivf");
        assert_eq!(index_group.components[0].type_name, "TestVec");
        assert_eq!(index_group.components[0].entry_count, 2);
        assert!(index_group.components[0].size_bytes > 0);
        assert_eq!(index_group.components[1].component, "page");
        assert_eq!(index_group.components[1].type_name, "TestVec");
        assert_eq!(index_group.components[1].entry_count, 1);
        assert!(index_group.components[1].size_bytes > 0);
        assert_eq!(
            index_summary.total_size_bytes,
            index_group
                .components
                .iter()
                .map(|component| component.size_bytes)
                .sum::<usize>()
        );

        let ds_metadata_cache = session.metadata_cache.for_dataset("memory://");
        ds_metadata_cache
            .insert_with_key(&TestKey("manifest/1"), Arc::new(vec![4]))
            .await;
        ds_metadata_cache
            .insert_with_key(&TestKey("txn/1"), Arc::new(vec![5]))
            .await;
        let file_metadata_cache =
            ds_metadata_cache.file_metadata_cache(&Path::from("data/0.lance"));
        file_metadata_cache
            .insert_with_key(&TestKey(""), Arc::new(vec![6]))
            .await;

        let metadata_summary = session.metadata_cache_summary().await.unwrap().unwrap();
        assert_eq!(metadata_summary.total_entries, 3);
        assert_eq!(metadata_summary.groups.len(), 1);
        let metadata_group = &metadata_summary.groups[0];
        assert_eq!(metadata_group.cache_prefix, "<metadata>");
        assert_eq!(
            metadata_group
                .components
                .iter()
                .map(|component| component.entry_count)
                .sum::<usize>(),
            3
        );
        assert!(
            metadata_group
                .components
                .iter()
                .any(|component| component.component == "manifest")
        );
        assert!(
            metadata_group
                .components
                .iter()
                .any(|component| component.component == "txn")
        );
        assert!(
            metadata_group
                .components
                .iter()
                .any(|component| component.component == "<empty>")
        );
    }

    #[rstest]
    #[case::frag_reuse(
        "frag_reuse/00000000-0000-0000-0000-000000000000",
        "FragReuseIndex",
        "frag_reuse"
    )]
    #[case::type_details(
        "type/00000000-0000-0000-0000-000000000000",
        "ScalarIndexDetails",
        "type"
    )]
    #[case::index_metadata_version("12", "Vec<IndexMetadata>", "version")]
    #[case::numeric_bitmap_value("12", "Bitmap", "Bitmap")]
    #[case::field_data("0:default", "FieldData", "field_data")]
    #[case::ivf_partition("ivf-12", "LegacyIVFPartition", "ivf")]
    #[case::scalar_value("some-value", "Bitmap", "Bitmap")]
    fn test_cache_key_component_names_are_compact(
        #[case] key: &str,
        #[case] type_name: &str,
        #[case] expected_component: &str,
    ) {
        assert_eq!(cache_key_component(key, type_name), expected_component);
    }

    #[tokio::test]
    async fn test_session_cache_summary_unsupported_backend() {
        let session = Session::with_index_cache_backend(
            Arc::new(NoInventoryBackend),
            10_000,
            Default::default(),
        );

        assert!(session.index_cache_summary().await.unwrap().is_none());
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
