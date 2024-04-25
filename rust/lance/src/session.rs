// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::cache::FileMetadataCache;
use lance_core::{Error, Result};
use lance_index::IndexType;
use snafu::{location, Location};

use crate::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use crate::index::cache::IndexCache;

use self::index_extension::IndexExtension;

pub mod index_extension;

/// A user session tracks the runtime state.
#[derive(Clone)]
pub struct Session {
    /// Cache for opened indices.
    pub(crate) index_cache: IndexCache,

    /// Cache for file metadata
    pub(crate) file_metadata_cache: FileMetadataCache,

    pub(crate) index_extensions: HashMap<(IndexType, String), Arc<dyn IndexExtension>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Session()")
    }
}

impl Session {
    /// Create a new session.
    ///
    /// Parameters:
    ///
    /// - ***index_cache_size***: the size of the index cache.
    pub fn new(index_cache_size: usize, metadata_cache_size: usize) -> Self {
        Self {
            index_cache: IndexCache::new(index_cache_size),
            file_metadata_cache: FileMetadataCache::new(metadata_cache_size),
            index_extensions: HashMap::new(),
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
            IndexType::Scalar => {
                return Err(Error::invalid_input(
                    "scalar index extension is not support yet".to_string(),
                    location!(),
                ));
            }
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
        }

        Ok(())
    }
}

impl Default for Session {
    fn default() -> Self {
        Self {
            index_cache: IndexCache::new(DEFAULT_INDEX_CACHE_SIZE),
            file_metadata_cache: FileMetadataCache::new(DEFAULT_METADATA_CACHE_SIZE),
            index_extensions: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::Float32Type;
    use std::sync::Arc;

    use crate::index::vector::pq::PQIndex;
    use lance_index::vector::pq::ProductQuantizerImpl;
    use lance_linalg::distance::MetricType;

    #[test]
    fn test_disable_index_cache() {
        let no_cache = Session::new(0, 0);
        assert!(no_cache.index_cache.get_vector("abc").is_none());
        let no_cache = Arc::new(no_cache);

        let pq = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
            1,
            8,
            1,
            Arc::new(vec![0.0f32; 8].into()),
            MetricType::L2,
        ));
        let idx = Arc::new(PQIndex::new(pq, MetricType::L2));
        no_cache.index_cache.insert_vector("abc", idx);

        assert!(no_cache.index_cache.get_vector("abc").is_none());
        assert_eq!(no_cache.index_cache.len_vector(), 0);
    }

    #[test]
    fn test_basic() {
        let session = Session::new(10, 1);
        let session = Arc::new(session);

        let pq = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
            1,
            8,
            1,
            Arc::new(vec![0.0f32; 8].into()),
            MetricType::L2,
        ));
        let idx = Arc::new(PQIndex::new(pq, MetricType::L2));
        assert_eq!(session.index_cache.get_size(), 0);

        assert_eq!(session.index_cache.hit_rate(), 1.0);
        session.index_cache.insert_vector("abc", idx.clone());

        let found = session.index_cache.get_vector("abc");
        assert!(found.is_some());
        assert_eq!(format!("{:?}", found.unwrap()), format!("{:?}", idx));
        assert_eq!(session.index_cache.hit_rate(), 1.0);
        assert!(session.index_cache.get_vector("def").is_none());
        assert_eq!(session.index_cache.hit_rate(), 0.5);
        assert!(session.index_cache.get_vector("abc").is_some());
        assert_eq!(session.index_cache.len_vector(), 1);
        assert_eq!(session.index_cache.get_size(), 1);

        for iter_idx in 0..100 {
            let pq_other = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
                1,
                8,
                1,
                Arc::new(vec![0.0f32; 8].into()),
                MetricType::L2,
            ));
            let idx_other = Arc::new(PQIndex::new(pq_other, MetricType::L2));
            session
                .index_cache
                .insert_vector(format!("{iter_idx}").as_str(), idx_other.clone());
        }

        // Capacity is 10 so there should be at most 10 items
        assert_eq!(session.index_cache.len_vector(), 10);
    }
}
