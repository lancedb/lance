// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache implementation

use std::any::{Any, TypeId};
use std::sync::Arc;

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
        let size_accessor =
            |record: &ArcAny| -> usize { record.clone().downcast::<T>().unwrap().deep_size_of() };
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
        }
    }

    pub fn no_cache() -> Self {
        Self {
            cache: Arc::new(Cache::new(0)),
            prefix: String::new(),
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
        self.cache
            .get(&(key, TypeId::of::<T>()))
            .map(|metadata| metadata.record.clone().downcast::<T>().unwrap())
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
        if let Some(metadata) = self.get::<T>(&key) {
            return Ok(metadata);
        }

        let metadata = Arc::new(loader(&key).await?);
        self.insert(&key, metadata.clone());
        Ok(metadata)
    }
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
}
