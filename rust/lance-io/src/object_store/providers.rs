// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

use snafu::location;
use tokio::sync::RwLock;
use url::Url;

use super::{ObjectStore, ObjectStoreParams};
use lance_core::error::{Error, Result};

#[cfg(feature = "aws")]
pub mod aws;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "gcp")]
pub mod gcp;
pub mod local;
pub mod memory;

#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore>;
}

#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: Mutex<HashMap<String, Arc<dyn ObjectStoreProvider>>>,
    // Cache of object stores currently in use. We use a weak reference so the
    // cache itself doesn't keep them alive if no object store is actually using
    // it.
    cache: RwLock<HashMap<(String, ObjectStoreParams), Weak<ObjectStore>>>,
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
    if url.scheme() == "file" || url.scheme() == "file-object-store" {
        // For file URLs, we want to cache the URL without the path.
        // This is because the path can be different for different
        // object stores, but we want to cache the object store itself.
        format!("{}://", url.scheme())
    } else {
        // drop path after bucket, but keep query params
        // e.g. s3://bucket/path?param=value -> s3://bucket?param=value
        let mut url = url.clone();
        let first_segment = url
            .path_segments()
            .and_then(|mut iter| iter.next().map(|s| s.to_string()));
        if let Some(first_segment) = first_segment {
            url.set_path(&first_segment);
        }
        url.to_string()
    }
}

impl ObjectStoreRegistry {
    pub fn empty() -> Self {
        Self {
            providers: Mutex::new(HashMap::new()),
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_provider(&self, scheme: &str) -> Option<Arc<dyn ObjectStoreProvider>> {
        self.providers
            .lock()
            .expect("ObjectStoreRegistry lock poisoned")
            .get(scheme)
            .cloned()
    }

    pub async fn get_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<Arc<ObjectStore>> {
        let cache_path = cache_url(&base_path);
        let cache_key = (cache_path, params.clone());

        // Check if we have a cached store for this base path and params
        {
            if let Some(store) = self.cache.read().await.get(&cache_key) {
                if let Some(store) = store.upgrade() {
                    return Ok(store);
                } else {
                    // Remove the weak reference if it is no longer valid
                    let mut cache_lock = self.cache.write().await;
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
        let provider = self.get_provider(scheme).ok_or_else(|| {
            Error::invalid_input(
                format!("No object store provider found for scheme: {}", scheme),
                location!(),
            )
        })?;
        let store = provider.new_store(base_path, params).await?;
        let store = Arc::new(store);

        {
            // Insert the store into the cache
            let mut cache_lock = self.cache.write().await;
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
        Self {
            providers: Mutex::new(providers),
            cache: RwLock::new(HashMap::new()),
        }
    }
}

impl ObjectStoreRegistry {
    pub fn insert(&self, scheme: &str, provider: Arc<dyn ObjectStoreProvider>) {
        self.providers
            .lock()
            .expect("ObjectStoreRegistry lock poisoned")
            .insert(scheme.into(), provider);
    }
}
