// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use deepsize::DeepSizeOf;
use lance_core::cache::{CapacityMode, LanceCache};
use lance_core::{Error, Result};
use lance_index::IndexType;
use snafu::location;

use crate::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};

use self::index_extension::IndexExtension;

pub mod index_extension;

/// A user session tracks the runtime state.
#[derive(Clone, DeepSizeOf)]
pub struct Session {
    /// Cache for opened indices.
    pub(crate) index_cache: LanceCache,

    /// Cache for file metadata
    pub(crate) file_metadata_cache: LanceCache,

    pub(crate) index_extensions: HashMap<(IndexType, String), Arc<dyn IndexExtension>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field(
                "index_cache",
                &format!("IndexCache(items={})", self.index_cache.approx_size(),),
            )
            .field(
                "file_metadata_cache",
                &format!(
                    "FileMetadataCache(items={})",
                    self.file_metadata_cache.approx_size(),
                ),
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
    pub fn new(index_cache_size: usize, metadata_cache_size: usize) -> Self {
        Self {
            index_cache: LanceCache::with_capacity(index_cache_size, CapacityMode::Bytes),
            file_metadata_cache: LanceCache::with_capacity(
                metadata_cache_size,
                CapacityMode::Bytes,
            ),
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
            _ => {
                return Err(Error::invalid_input(
                    format!(
                        "scalar index extension is not support yet: {}",
                        extension.index_type()
                    ),
                    location!(),
                ));
            }
        }

        Ok(())
    }

    /// Return the current size of the session in bytes
    pub fn size_bytes(&self) -> u64 {
        // We re-expose deep_size_of here so that users don't
        // need the deepsize crate themselves (e.g. to use deep_size_of)
        self.deep_size_of() as u64
    }

    pub fn cache_size_bytes(&self) -> usize {
        self.index_cache.size_bytes() + self.file_metadata_cache.size_bytes()
    }

    pub fn approx_num_items(&self) -> usize {
        self.index_cache.approx_size()
            + self.file_metadata_cache.approx_size()
            + self.index_extensions.len()
    }
}

impl Default for Session {
    fn default() -> Self {
        Self {
            // TODO: should we just drop item capacity mode?
            index_cache: LanceCache::with_capacity(DEFAULT_INDEX_CACHE_SIZE, CapacityMode::Bytes),
            file_metadata_cache: LanceCache::with_capacity(
                DEFAULT_METADATA_CACHE_SIZE,
                CapacityMode::Bytes,
            ),
            index_extensions: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use std::sync::Arc;

    use crate::index::vector::pq::PQIndex;
    use lance_index::vector::{pq::ProductQuantizer, VectorIndex};
    use lance_linalg::distance::DistanceType;

    #[test]
    fn test_disable_index_cache() {
        let no_cache = Session::new(0, 0);
        assert!(no_cache
            .index_cache
            .get_unsized::<dyn VectorIndex>("abc")
            .is_none());
        let no_cache = Arc::new(no_cache);

        let pq = ProductQuantizer::new(
            1,
            8,
            1,
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 8]), 1)
                .unwrap(),
            DistanceType::L2,
        );
        let idx = Arc::new(PQIndex::new(pq, DistanceType::L2));
        no_cache
            .index_cache
            .insert_unsized::<dyn VectorIndex>("abc".to_string(), idx);

        assert!(no_cache
            .index_cache
            .get_unsized::<dyn VectorIndex>("abc")
            .is_none());
        assert_eq!(no_cache.index_cache.size(), 0);
    }
}
