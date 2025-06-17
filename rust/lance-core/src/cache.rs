// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache implementation

use std::any::{Any, TypeId};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use futures::Future;
use moka::sync::Cache;

use crate::Result;

pub use deepsize::{Context, DeepSizeOf};

type ArcAny = Arc<dyn Any + Send + Sync>;

#[derive(Clone)]
struct SizedRecord {
    record: ArcAny,
    size_accessor: Arc<dyn Fn(&ArcAny) -> usize + Send + Sync>,
}

impl std::fmt::Debug for SizedRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SizedRecord")
            .field("record", &self.record)
            .finish()
    }
}

impl SizedRecord {
    fn new<T: DeepSizeOf + Send + Sync + 'static>(record: Arc<T>) -> Self {
        // +8 for the size of the Arc pointer itself
        let size_accessor =
            |record: &ArcAny| -> usize { record.downcast_ref::<T>().unwrap().deep_size_of() + 8 };
        Self {
            record,
            size_accessor: Arc::new(size_accessor),
        }
    }
}

#[derive(Clone)]
pub struct LanceCache {
    cache: Arc<Cache<(String, TypeId), SizedRecord>>,
    prefix: String,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl std::fmt::Debug for LanceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceCache")
            .field("cache", &self.cache)
            .finish()
    }
}

impl DeepSizeOf for LanceCache {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.cache
            .iter()
            .map(|(_, v)| (v.size_accessor)(&v.record))
            .sum()
    }
}

impl LanceCache {
    pub fn with_capacity(capacity: usize) -> Self {
        let cache = Cache::builder()
            .max_capacity(capacity as u64)
            .weigher(|_, v: &SizedRecord| {
                (v.size_accessor)(&v.record).try_into().unwrap_or(u32::MAX)
            })
            .support_invalidation_closures()
            .build();
        Self {
            cache: Arc::new(cache),
            prefix: String::new(),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn no_cache() -> Self {
        Self {
            cache: Arc::new(Cache::new(0)),
            prefix: String::new(),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Appends a prefix to the cache key
    ///
    /// If this cache already has a prefix, the new prefix will be appended to
    /// the existing one.
    ///
    /// Prefixes are used to create a namespace for the cache keys to avoid
    /// collisions between different caches.
    pub fn with_key_prefix(&self, prefix: &str) -> Self {
        Self {
            cache: self.cache.clone(),
            prefix: format!("{}{}/", self.prefix, prefix),
            hits: self.hits.clone(),
            misses: self.misses.clone(),
        }
    }

    fn get_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }

    /// Invalidate all entries in the cache that start with the given prefix
    ///
    /// The given prefix is appended to the existing prefix of the cache. If you
    /// want to invalidate all at the current prefix, pass an empty string.
    pub fn invalidate_prefix(&self, prefix: &str) {
        let full_prefix = format!("{}{}", self.prefix, prefix);
        self.cache
            .invalidate_entries_if(move |(key, _typeid), _value| key.starts_with(&full_prefix))
            .expect("Cache configured correctly");
    }

    pub fn size(&self) -> usize {
        self.cache.run_pending_tasks();
        self.cache.entry_count() as usize
    }

    pub fn approx_size(&self) -> usize {
        self.cache.entry_count() as usize
    }

    pub fn size_bytes(&self) -> usize {
        self.cache.run_pending_tasks();
        self.approx_size_bytes()
    }

    pub fn approx_size_bytes(&self) -> usize {
        self.cache.weighted_size() as usize
    }

    pub fn insert<T: DeepSizeOf + Send + Sync + 'static>(&self, key: &str, metadata: Arc<T>) {
        let key = self.get_key(key);
        self.cache
            .insert((key, TypeId::of::<T>()), SizedRecord::new(metadata));
    }

    pub fn insert_unsized<T: DeepSizeOf + Send + Sync + 'static + ?Sized>(
        &self,
        key: &str,
        metadata: Arc<T>,
    ) {
        // In order to make the data Sized, we wrap in another pointer.
        self.insert(key, Arc::new(metadata))
    }

    pub fn get<T: DeepSizeOf + Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        let key = self.get_key(key);
        if let Some(metadata) = self.cache.get(&(key, TypeId::of::<T>())) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(metadata.record.clone().downcast::<T>().unwrap())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn get_unsized<T: DeepSizeOf + Send + Sync + 'static + ?Sized>(
        &self,
        key: &str,
    ) -> Option<Arc<T>> {
        let outer = self.get::<Arc<T>>(key)?;
        Some(outer.as_ref().clone())
    }

    /// Get an item
    ///
    /// If it exists in the cache return that
    ///
    /// If it doesn't then run `loader` to load the item, insert into cache, and return
    pub async fn get_or_insert<T: DeepSizeOf + Send + Sync + 'static, F, Fut>(
        &self,
        key: String,
        loader: F,
    ) -> Result<Arc<T>>
    where
        F: FnOnce(&str) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let full_key = self.get_key(&key);
        if let Some(metadata) = self.cache.get(&(full_key, TypeId::of::<T>())) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(metadata.record.clone().downcast::<T>().unwrap());
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let metadata = Arc::new(loader(&key).await?);
        self.insert(&key, metadata.clone());
        Ok(metadata)
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_bytes() {
        let item = Arc::new(vec![1, 2, 3]);
        let item_size = item.deep_size_of(); // Size of Arc<Vec<i32>>
        let capacity = 10 * item_size;

        let cache = LanceCache::with_capacity(capacity);
        assert_eq!(cache.size_bytes(), 0);
        assert_eq!(cache.approx_size_bytes(), 0);

        let item = Arc::new(vec![1, 2, 3]);
        cache.insert("key", item.clone());
        assert_eq!(cache.size(), 1);
        assert_eq!(cache.size_bytes(), item_size);
        assert_eq!(cache.approx_size_bytes(), item_size);

        let retrieved = cache.get::<Vec<i32>>("key").unwrap();
        assert_eq!(*retrieved, *item);

        // Test eviction based on size
        for i in 0..20 {
            cache.insert(&format!("key_{}", i), Arc::new(vec![i, i, i]));
        }
        assert_eq!(cache.size_bytes(), capacity);
        assert_eq!(cache.size(), 10);
    }

    #[test]
    fn test_cache_trait_objects() {
        #[derive(Debug, DeepSizeOf)]
        struct MyType(i32);

        trait MyTrait: DeepSizeOf + Send + Sync + Any {
            fn as_any(&self) -> &dyn Any;
        }

        impl MyTrait for MyType {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        let item = Arc::new(MyType(42));
        let item_dyn: Arc<dyn MyTrait> = item;

        let cache = LanceCache::with_capacity(1000);
        cache.insert_unsized("test", item_dyn);

        let retrieved = cache.get_unsized::<dyn MyTrait>("test").unwrap();
        let retrieved = retrieved.as_any().downcast_ref::<MyType>().unwrap();
        assert_eq!(retrieved.0, 42);
    }

    #[test]
    fn test_cache_stats_basic() {
        let cache = LanceCache::with_capacity(1000);

        // Initially no hits or misses
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        // Miss on first get
        let result = cache.get::<Vec<i32>>("nonexistent");
        assert!(result.is_none());
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);

        // Insert and then hit
        cache.insert("key1", Arc::new(vec![1, 2, 3]));
        let result = cache.get::<Vec<i32>>("key1");
        assert!(result.is_some());
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        // Another hit
        let result = cache.get::<Vec<i32>>("key1");
        assert!(result.is_some());
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);

        // Another miss
        let result = cache.get::<Vec<i32>>("nonexistent2");
        assert!(result.is_none());
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_cache_stats_with_prefixes() {
        let base_cache = LanceCache::with_capacity(1000);
        let prefixed_cache = base_cache.with_key_prefix("test");

        // Stats should be shared between base and prefixed cache
        let stats = base_cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        let stats = prefixed_cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        // Miss on prefixed cache
        let result = prefixed_cache.get::<Vec<i32>>("key1");
        assert!(result.is_none());

        // Both should show the miss
        let stats = base_cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);

        let stats = prefixed_cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);

        // Insert through prefixed cache and hit
        prefixed_cache.insert("key1", Arc::new(vec![1, 2, 3]));
        let result = prefixed_cache.get::<Vec<i32>>("key1");
        assert!(result.is_some());

        // Both should show the hit
        let stats = base_cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        let stats = prefixed_cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_stats_unsized() {
        #[derive(Debug, DeepSizeOf)]
        struct MyType(i32);

        trait MyTrait: DeepSizeOf + Send + Sync + Any {
            fn as_any(&self) -> &dyn Any;
        }

        impl MyTrait for MyType {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        let cache = LanceCache::with_capacity(1000);

        // Miss on unsized get
        let result = cache.get_unsized::<dyn MyTrait>("test");
        assert!(result.is_none());
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);

        // Insert and hit on unsized
        let item = Arc::new(MyType(42));
        let item_dyn: Arc<dyn MyTrait> = item;
        cache.insert_unsized("test", item_dyn);

        let result = cache.get_unsized::<dyn MyTrait>("test");
        assert!(result.is_some());
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_cache_stats_get_or_insert() {
        let cache = LanceCache::with_capacity(1000);

        // First call should be a miss and load the value
        let result: Arc<Vec<i32>> = cache
            .get_or_insert("key1".to_string(), |_key| async { Ok(vec![1, 2, 3]) })
            .await
            .unwrap();
        assert_eq!(*result, vec![1, 2, 3]);

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);

        // Second call should be a hit
        let result: Arc<Vec<i32>> = cache
            .get_or_insert("key1".to_string(), |_key| async {
                panic!("Should not be called")
            })
            .await
            .unwrap();
        assert_eq!(*result, vec![1, 2, 3]);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        // Different key should be another miss
        let result: Arc<Vec<i32>> = cache
            .get_or_insert("key2".to_string(), |_key| async { Ok(vec![4, 5, 6]) })
            .await
            .unwrap();
        assert_eq!(*result, vec![4, 5, 6]);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }
}
