// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache implementation

use std::any::{Any, TypeId};
use std::sync::Arc;

use futures::Future;
use moka::sync::Cache;
use object_store::path::Path;

use crate::Result;

pub const DEFAULT_INDEX_CACHE_SIZE: usize = 128;
pub const DEFAULT_METADATA_CACHE_SIZE: usize = 128;

type ArcAny = Arc<dyn Any + Send + Sync>;

/// Cache for various metadata about files.
///
/// The cache is keyed by the file path and the type of metadata.
#[derive(Clone, Debug)]
pub struct FileMetadataCache {
    cache: Arc<Cache<(Path, TypeId), ArcAny>>,
}

impl FileMetadataCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Cache::new(capacity as u64)),
        }
    }

    pub fn get<T: Send + Sync + 'static>(&self, path: &Path) -> Option<Arc<T>> {
        self.cache
            .get(&(path.to_owned(), TypeId::of::<T>()))
            .map(|metadata| metadata.clone().downcast::<T>().unwrap())
    }

    pub fn insert<T: Send + Sync + 'static>(&self, path: Path, metadata: Arc<T>) {
        self.cache.insert((path, TypeId::of::<T>()), metadata);
    }

    /// Get an item
    ///
    /// If it exists in the cache return that
    ///
    /// If it doesn't then run `loader` to load the item, insert into cache, and return
    pub async fn get_or_insert<T: Send + Sync + 'static, F, Fut>(
        &self,
        path: &Path,
        loader: F,
    ) -> Result<Arc<T>>
    where
        F: Fn(&Path) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        if let Some(metadata) = self.get::<T>(path) {
            return Ok(metadata);
        }

        let metadata = Arc::new(loader(path).await?);
        self.insert(path.to_owned(), metadata.clone());
        Ok(metadata)
    }
}
