// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache implementation

use std::any::{Any, TypeId};
use std::sync::Arc;

use futures::Future;
use moka::sync::Cache;
use object_store::path::Path;

use crate::utils::path::LancePathExt;
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
            |record: &ArcAny| -> usize { record.downcast_ref::<T>().unwrap().deep_size_of() };
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
    cache: Option<Arc<Cache<(Path, TypeId), SizedRecord>>>,
    base_path: Option<Path>,
}

impl DeepSizeOf for FileMetadataCache {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.cache
            .as_ref()
            .map(|cache| {
                cache
                    .iter()
                    .map(|(_, v)| (v.size_accessor)(&v.record))
                    .sum()
            })
            .unwrap_or(0)
    }
}

pub enum CapacityMode {
    Items,
    Bytes,
}

impl FileMetadataCache {
    /// Instantiates a new cache which, for legacy reasons, uses Items capacity mode.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Some(Arc::new(Cache::new(capacity as u64))),
            base_path: None,
        }
    }

    /// Instantiates a dummy cache that will never cache anything.
    pub fn no_cache() -> Self {
        Self {
            cache: None,
            base_path: None,
        }
    }

    /// Instantiates a new cache with a given capacity and capacity mode.
    pub fn with_capacity(capacity: usize, mode: CapacityMode) -> Self {
        match mode {
            CapacityMode::Items => Self::new(capacity),
            CapacityMode::Bytes => Self {
                cache: Some(Arc::new(
                    Cache::builder()
                        .weigher(|_, v: &SizedRecord| {
                            (v.size_accessor)(&v.record).try_into().unwrap_or(u32::MAX)
                        })
                        .build(),
                )),
                base_path: None,
            },
        }
    }

    /// Creates a new cache which shares the same underlying cache but prepends `base_path` to all
    /// keys.
    pub fn with_base_path(&self, base_path: Path) -> Self {
        Self {
            cache: self.cache.clone(),
            base_path: Some(base_path),
        }
    }

    pub fn size(&self) -> usize {
        if let Some(cache) = self.cache.as_ref() {
            cache.run_pending_tasks();
            cache.entry_count() as usize
        } else {
            0
        }
    }

    /// Fetch an item from the cache, using a str as the key
    pub fn get_by_str<T: Send + Sync + 'static>(&self, path: &str) -> Option<Arc<T>> {
        self.get(&Path::parse(path).unwrap())
    }

    /// Fetch an item from the cache
    pub fn get<T: Send + Sync + 'static>(&self, path: &Path) -> Option<Arc<T>> {
        let cache = self.cache.as_ref()?;
        let temp: Path;
        let path = if let Some(base_path) = &self.base_path {
            temp = base_path.child_path(path);
            &temp
        } else {
            path
        };
        cache
            .get(&(path.to_owned(), TypeId::of::<T>()))
            .map(|metadata| metadata.record.clone().downcast::<T>().unwrap())
    }

    /// Insert an item into the cache
    pub fn insert<T: DeepSizeOf + Send + Sync + 'static>(&self, path: Path, metadata: Arc<T>) {
        let Some(cache) = self.cache.as_ref() else {
            return;
        };
        let path = if let Some(base_path) = &self.base_path {
            base_path.child_path(&path)
        } else {
            path
        };
        cache.insert((path, TypeId::of::<T>()), SizedRecord::new(metadata));
    }

    /// Insert an item into the cache, using a str as the key
    pub fn insert_by_str<T: DeepSizeOf + Send + Sync + 'static>(
        &self,
        key: &str,
        metadata: Arc<T>,
    ) {
        self.insert(Path::parse(key).unwrap(), metadata);
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
