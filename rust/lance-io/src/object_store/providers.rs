// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{Arc, RwLock, Weak},
};

use object_store::path::Path;
use snafu::location;
use url::Url;

use super::{tracing::ObjectStoreTracingExt, ObjectStore, ObjectStoreParams};
use lance_core::error::{Error, LanceOptionExt, Result};

#[cfg(feature = "aws")]
pub mod aws;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "gcp")]
pub mod gcp;
pub mod local;
pub mod memory;
#[cfg(feature = "oss")]
pub mod oss;

#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore>;

    /// Extract the path relative to the base of the store.
    ///
    /// For example, in S3 the path is relative to the bucket. So a URL of
    /// `s3://bucket/path/to/file` would return `path/to/file`.
    ///
    /// Meanwhile, for a file store, the path is relative to the filesystem root.
    /// So a URL of `file:///path/to/file` would return `/path/to/file`.
    fn extract_path(&self, url: &Url) -> Path {
        Path::from(url.path())
    }
}

/// A registry of object store providers.
///
/// Use [`Self::default()`] to create one with the available default providers.
/// This includes (depending on features enabled):
/// - `memory`: An in-memory object store.
/// - `file`: A local file object store, with optimized code paths.
/// - `file-object-store`: A local file object store that uses the ObjectStore API,
///   for all operations. Used for testing with ObjectStore wrappers.
/// - `s3`: An S3 object store.
/// - `s3+ddb`: An S3 object store with DynamoDB for metadata.
/// - `az`: An Azure Blob Storage object store.
/// - `gs`: A Google Cloud Storage object store.
///
/// Use [`Self::empty()`] to create an empty registry, with no providers registered.
///
/// The registry also caches object stores that are currently in use. It holds
/// weak references to the object stores, so they are not held onto. If an object
/// store is no longer in use, it will be removed from the cache on the next
/// call to either [`Self::active_stores()`] or [`Self::get_store()`].
#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: RwLock<HashMap<String, Arc<dyn ObjectStoreProvider>>>,
    // Cache of object stores currently in use. We use a weak reference so the
    // cache itself doesn't keep them alive if no object store is actually using
    // it.
    active_stores: RwLock<HashMap<(String, ObjectStoreParams), Weak<ObjectStore>>>,
}

/// Convert a URL to a cache key.
///
/// We truncate to the first path segment. This should capture
/// buckets and prefixes. We keep URL params since those might be
/// important.
///
/// * s3://bucket/path?param=value -> s3://bucket/path?param=value
/// * file:///path/to/file -> file:///
fn cache_url(url: &Url) -> String {
    if ["file", "file-object-store", "memory"].contains(&url.scheme()) {
        // For file URLs, we want to cache the URL without the path.
        // This is because the path can be different for different
        // object stores, but we want to cache the object store itself.
        format!("{}://", url.scheme())
    } else {
        // Bucket is parsed as domain, so we just drop the path.
        let mut url = url.clone();
        url.set_path("");
        url.to_string()
    }
}

impl ObjectStoreRegistry {
    /// Create a new registry with no providers registered.
    ///
    /// Typically, you want to use [`Self::default()`] instead, so you get the
    /// default providers.
    pub fn empty() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            active_stores: RwLock::new(HashMap::new()),
        }
    }

    /// Get the object store provider for a given scheme.
    pub fn get_provider(&self, scheme: &str) -> Option<Arc<dyn ObjectStoreProvider>> {
        self.providers
            .read()
            .expect("ObjectStoreRegistry lock poisoned")
            .get(scheme)
            .cloned()
    }

    /// Get a list of all active object stores.
    ///
    /// Calling this will also clean up any weak references to object stores that
    /// are no longer valid.
    pub fn active_stores(&self) -> Vec<Arc<ObjectStore>> {
        let mut found_inactive = false;
        let output = self
            .active_stores
            .read()
            .expect("ObjectStoreRegistry lock poisoned")
            .values()
            .filter_map(|weak| match weak.upgrade() {
                Some(store) => Some(store),
                None => {
                    found_inactive = true;
                    None
                }
            })
            .collect();

        if found_inactive {
            // Clean up the cache by removing any weak references that are no longer valid
            let mut cache_lock = self
                .active_stores
                .write()
                .expect("ObjectStoreRegistry lock poisoned");
            cache_lock.retain(|_, weak| weak.upgrade().is_some());
        }
        output
    }

    /// Get an object store for a given base path and parameters.
    ///
    /// If the object store is already in use, it will return a strong reference
    /// to the object store. If the object store is not in use, it will create a
    /// new object store and return a strong reference to it.
    pub async fn get_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<Arc<ObjectStore>> {
        let cache_path = cache_url(&base_path);
        let cache_key = (cache_path, params.clone());

        // Check if we have a cached store for this base path and params
        {
            let maybe_store = self
                .active_stores
                .read()
                .ok()
                .expect_ok()?
                .get(&cache_key)
                .cloned();
            if let Some(store) = maybe_store {
                if let Some(store) = store.upgrade() {
                    return Ok(store);
                } else {
                    // Remove the weak reference if it is no longer valid
                    let mut cache_lock = self
                        .active_stores
                        .write()
                        .expect("ObjectStoreRegistry lock poisoned");
                    if let Some(store) = cache_lock.get(&cache_key) {
                        if store.upgrade().is_none() {
                            // Remove the weak reference if it is no longer valid
                            cache_lock.remove(&cache_key);
                        }
                    }
                }
            }
        }

        let scheme = base_path.scheme();
        let Some(provider) = self.get_provider(scheme) else {
            let mut message = format!("No object store provider found for scheme: '{}'", scheme);
            if let Ok(providers) = self.providers.read() {
                let valid_schemes = providers.keys().cloned().collect::<Vec<_>>().join(", ");
                message.push_str(&format!("\nValid schemes: {}", valid_schemes));
            }

            return Err(Error::invalid_input(message, location!()));
        };
        let mut store = provider.new_store(base_path, params).await?;

        store.inner = store.inner.traced();

        if let Some(wrapper) = &params.object_store_wrapper {
            store.inner = wrapper.wrap(store.inner);
        }

        let store = Arc::new(store);

        {
            // Insert the store into the cache
            let mut cache_lock = self.active_stores.write().ok().expect_ok()?;
            cache_lock.insert(cache_key, Arc::downgrade(&store));
        }

        Ok(store)
    }
}

impl Default for ObjectStoreRegistry {
    fn default() -> Self {
        let mut providers: HashMap<String, Arc<dyn ObjectStoreProvider>> = HashMap::new();

        providers.insert("memory".into(), Arc::new(memory::MemoryStoreProvider));
        providers.insert("file".into(), Arc::new(local::FileStoreProvider));
        // The "file" scheme has special optimized code paths that bypass
        // the ObjectStore API for better performance. However, this can make it
        // hard to test when using ObjectStore wrappers, such as IOTrackingStore.
        // So we provide a "file-object-store" scheme that uses the ObjectStore API.
        // The specialized code paths are differentiated by the scheme name.
        providers.insert(
            "file-object-store".into(),
            Arc::new(local::FileStoreProvider),
        );

        #[cfg(feature = "aws")]
        {
            let aws = Arc::new(aws::AwsStoreProvider);
            providers.insert("s3".into(), aws.clone());
            providers.insert("s3+ddb".into(), aws);
        }
        #[cfg(feature = "azure")]
        providers.insert("az".into(), Arc::new(azure::AzureBlobStoreProvider));
        #[cfg(feature = "gcp")]
        providers.insert("gs".into(), Arc::new(gcp::GcsStoreProvider));
        #[cfg(feature = "oss")]
        providers.insert("oss".into(), Arc::new(oss::OssStoreProvider));
        Self {
            providers: RwLock::new(providers),
            active_stores: RwLock::new(HashMap::new()),
        }
    }
}

impl ObjectStoreRegistry {
    /// Add a new object store provider to the registry. The provider will be used
    /// in [`Self::get_store()`] when a URL is passed with a matching scheme.
    pub fn insert(&self, scheme: &str, provider: Arc<dyn ObjectStoreProvider>) {
        self.providers
            .write()
            .expect("ObjectStoreRegistry lock poisoned")
            .insert(scheme.into(), provider);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_url() {
        let cases = [
            ("s3://bucket/path?param=value", "s3://bucket?param=value"),
            ("file:///path/to/file", "file://"),
            ("file-object-store:///path/to/file", "file-object-store://"),
            ("memory:///", "memory://"),
            (
                "http://example.com/path?param=value",
                "http://example.com/?param=value",
            ),
        ];

        for (url, expected_cache_url) in cases {
            let url = Url::parse(url).unwrap();
            let cache_url = cache_url(&url);
            assert_eq!(cache_url, expected_cache_url);
        }
    }
}
