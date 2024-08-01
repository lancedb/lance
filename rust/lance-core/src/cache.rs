// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache implementation

use std::any::{Any, TypeId};
use std::sync::Arc;

use deepsize::{Context, DeepSizeOf};
use futures::Future;
use moka::sync::Cache;
use object_store::path::Path;

use crate::Result;

pub const DEFAULT_INDEX_CACHE_SIZE: usize = 128;
pub const DEFAULT_METADATA_CACHE_SIZE: usize = 128;

type ArcAny = Arc<dyn Any + Send + Sync>;

#[derive(Clone)]
struct SizedRecord {
    record: ArcAny,
    size_accessor: Arc<dyn Fn(ArcAny) -> usize + Send + Sync>,
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
            |record: ArcAny| -> usize { record.downcast_ref::<T>().unwrap().deep_size_of() };
        Self {
            record,
            size_accessor: Arc::new(size_accessor),
        }
    }
}

/// Cache for various metadata about files.
///
/// The cache is keyed by the file path and the type of metadata.
#[derive(Clone, Debug)]
pub struct FileMetadataCache {
    cache: Arc<Cache<(Path, TypeId), SizedRecord>>,
}

impl DeepSizeOf for FileMetadataCache {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.cache
            .iter()
            .map(|(_, v)| (v.size_accessor)(v.record))
            .sum()
    }
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
            .map(|metadata| metadata.record.clone().downcast::<T>().unwrap())
    }

    pub fn insert<T: DeepSizeOf + Send + Sync + 'static>(&self, path: Path, metadata: Arc<T>) {
        self.cache
            .insert((path, TypeId::of::<T>()), SizedRecord::new(metadata));
    }

    /// Get an item
    ///
    /// If it exists in the cache return that
    ///
    /// If it doesn't then run `loader` to load the item, insert into cache, and return
    pub async fn get_or_insert<T: DeepSizeOf + Send + Sync + 'static, F, Fut>(
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
