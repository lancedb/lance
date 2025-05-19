// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::scalar::IndexStore;
use crate::{Index, IndexType};
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use futures;
use futures::future::join_all;
use lance_core::{Error, Result};
use lance_table::format::pb;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;
use std::{any::Any, collections::HashMap, sync::Arc};
use uuid::Uuid;

const BATCH_SIZE: usize = i32::MAX as usize - 1024 * 1024;

const MAX_ROWS_PER_CHUNK: usize = 2 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseVersion {
    pub dataset_version: u64,
    pub row_addr_map_path: String,
}

impl From<&FragReuseVersion> for pb::fragment_reuse_index_details::Version {
    fn from(version: &FragReuseVersion) -> Self {
        Self {
            dataset_version: version.dataset_version,
            row_addr_map_path: version.row_addr_map_path.clone(),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::Version> for FragReuseVersion {
    type Error = Error;

    fn try_from(version: pb::fragment_reuse_index_details::Version) -> Result<Self> {
        Ok(Self {
            dataset_version: version.dataset_version,
            row_addr_map_path: version.row_addr_map_path,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseIndexDetails {
    pub versions: Vec<FragReuseVersion>,
}

impl From<&FragReuseIndexDetails> for pb::FragmentReuseIndexDetails {
    fn from(details: &FragReuseIndexDetails) -> Self {
        Self {
            versions: details.versions.iter().map(|m| m.into()).collect(),
        }
    }
}

impl TryFrom<pb::FragmentReuseIndexDetails> for FragReuseIndexDetails {
    type Error = Error;

    fn try_from(details: pb::FragmentReuseIndexDetails) -> Result<Self> {
        Ok(Self {
            versions: details
                .versions
                .into_iter()
                .map(|m| m.try_into())
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

/// An index that stores row ID maps.
/// A row ID map describes the mapping from old row address to new address after compactions.
/// Each version contains the mapping for one round of compaction.
#[derive(Debug, Clone)]
pub struct FragReuseIndex {
    pub details: FragReuseIndexDetails,

    // row ID map of all compaction rounds
    // ordered from oldest to newest
    pub row_addr_maps: Vec<HashMap<u64, Option<u64>>>,

    store: Arc<dyn IndexStore>,
}

impl DeepSizeOf for FragReuseIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.details.deep_size_of_children(context) + self.store.deep_size_of_children(context)
    }
}

impl FragReuseIndex {
    pub fn new(
        row_addr_maps: Vec<HashMap<u64, Option<u64>>>,
        details: FragReuseIndexDetails,
        store: Arc<dyn IndexStore>,
    ) -> Self {
        Self {
            row_addr_maps,
            details,
            store,
        }
    }

    pub async fn load(
        details: FragReuseIndexDetails,
        store: Arc<dyn IndexStore>,
    ) -> Result<Arc<Self>> {
        let row_addr_maps = join_all(
            details
                .versions
                .iter()
                .map(|version| load_row_addr_map(&version.row_addr_map_path, store.clone())),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

        Ok(Arc::new(Self::new(row_addr_maps, details, store)))
    }
}

#[derive(Serialize)]
struct FragReuseStatistics {
    num_versions: usize,
    num_rows: usize,
}

#[async_trait]
impl Index for FragReuseIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "FragReuseIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = FragReuseStatistics {
            num_versions: self.details.versions.len(),
            num_rows: self.details.versions.len(),
        };
        serde_json::to_value(stats).map_err(|e| Error::Internal {
            message: format!("failed to serialize fragment reuse index statistics: {}", e),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        // TODO: can preload the row ID map
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::FragmentReuse
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

/// Save a row ID map to a Lance index file.
pub async fn save_row_addr_map(
    index_store: Arc<dyn IndexStore>,
    dataset_version: &u64,
    row_addr_map: &HashMap<u64, Option<u64>>,
) -> Result<String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("old_row_addr", DataType::UInt64, false),
        Field::new("new_row_addr", DataType::UInt64, true),
    ]));

    // need a UUID in case of concurrent compactions writing
    let file_path = format!("{}_row_addr_map_{}.lance", dataset_version, Uuid::new_v4());
    let mut index_file = index_store
        .new_index_file(&file_path, schema.clone())
        .await?;

    let mut old_row_addrs = Vec::new();
    let mut new_row_addrs = Vec::new();
    let mut rows = 0;

    for (old_row_addr, new_row_addr) in row_addr_map.iter() {
        // flush current batch
        if rows > BATCH_SIZE {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(UInt64Array::from(old_row_addrs.clone())),
                    Arc::new(UInt64Array::from(new_row_addrs.clone())),
                ],
            )?;

            index_file.write_record_batch(batch).await?;

            old_row_addrs.clear();
            new_row_addrs.clear();
            rows = 0;
        }

        old_row_addrs.push(*old_row_addr);
        new_row_addrs.push(*new_row_addr);
        rows += 1;
    }

    if !old_row_addrs.is_empty() {
        // write remaining
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(old_row_addrs)),
                Arc::new(UInt64Array::from(new_row_addrs)),
            ],
        )?;

        index_file.write_record_batch(batch).await?;
    }

    index_file.finish().await?;
    Ok(file_path)
}

pub async fn load_row_addr_map(
    file_path: &str,
    store: Arc<dyn IndexStore>,
) -> Result<HashMap<u64, Option<u64>>> {
    let index_file = store.open_index_file(file_path).await?;
    let total_rows = index_file.num_rows();

    let mut row_addr_map = HashMap::new();
    if total_rows == 0 {
        return Ok(row_addr_map);
    }

    for start_row in (0..total_rows).step_by(MAX_ROWS_PER_CHUNK) {
        let end_row = (start_row + MAX_ROWS_PER_CHUNK).min(total_rows);
        let chunk = index_file.read_range(start_row..end_row, None).await?;

        if chunk.num_rows() == 0 {
            continue;
        }

        let old_row_addrs = chunk
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let new_row_addrs = chunk
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        for idx in 0..chunk.num_rows() {
            let new_row_addr = if new_row_addrs.is_null(idx) {
                None
            } else {
                Some(new_row_addrs.value(idx))
            };
            row_addr_map.insert(old_row_addrs.value(idx), new_row_addr);
        }
    }

    Ok(row_addr_map)
}

#[cfg(test)]
pub mod tests {
    use crate::reuse::*;
    use crate::scalar::lance_format::LanceIndexStore;
    use lance_core::cache::FileMetadataCache;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_save_and_load_row_addr_map() {
        // Create a test row ID map, make it not aligned with chunk size
        let mut row_addr_map = HashMap::new();
        for i in 0..(1024 * 1024 + 100) {
            // Some rows map to new IDs, some are deleted (None)
            if i % 3 == 0 {
                row_addr_map.insert(i, None);
            } else {
                row_addr_map.insert(i, Some(i * 2));
            }
        }

        // Create a temporary store
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            FileMetadataCache::no_cache(),
        ));

        // Save the row ID map
        let dataset_version = 1;
        let file_path = save_row_addr_map(test_store.clone(), &dataset_version, &row_addr_map)
            .await
            .expect("Failed to save row ID map");

        // Load the row ID map back
        let loaded_map = load_row_addr_map(&file_path, test_store)
            .await
            .expect("Failed to load row ID map");

        // Verify the loaded map matches the original
        assert_eq!(loaded_map.len(), row_addr_map.len(), "Map size mismatch");
        for (old_addr, new_addr) in &row_addr_map {
            assert_eq!(
                loaded_map.get(old_addr),
                Some(new_addr),
                "Mismatch for old_addr {}",
                old_addr
            );
        }
    }

    #[tokio::test]
    async fn test_load_frag_reuse_index() {
        // Create a temporary store
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            FileMetadataCache::no_cache(),
        ));

        // Create multiple versions of row ID maps
        let mut versions = Vec::new();
        let mut row_addr_maps = Vec::new();

        // Version 1: Simple mapping
        let mut map1 = HashMap::new();
        for i in 0..100 {
            map1.insert(i, Some(i * 2));
        }
        let file_path1 = save_row_addr_map(test_store.clone(), &1, &map1)
            .await
            .expect("Failed to save row ID map");
        versions.push(FragReuseVersion {
            dataset_version: 1,
            row_addr_map_path: file_path1,
        });
        row_addr_maps.push(map1);

        // Version 2: Some deletions
        let mut map2 = HashMap::new();
        for i in 0..200 {
            if i % 3 == 0 {
                map2.insert(i, None);
            } else {
                map2.insert(i, Some(i * 3));
            }
        }
        let file_path2 = save_row_addr_map(test_store.clone(), &2, &map2)
            .await
            .expect("Failed to save row ID map");
        versions.push(FragReuseVersion {
            dataset_version: 2,
            row_addr_map_path: file_path2,
        });
        row_addr_maps.push(map2);

        // Create and save the FragReuseIndex
        let details = FragReuseIndexDetails { versions };
        let index = FragReuseIndex::new(row_addr_maps.clone(), details.clone(), test_store.clone());

        // Load the index back
        let loaded_index = FragReuseIndex::load(details, test_store)
            .await
            .expect("Failed to load FragReuseIndex");

        // Verify the loaded index matches the original
        assert_eq!(
            loaded_index.row_addr_maps.len(),
            row_addr_maps.len(),
            "Number of row ID maps mismatch"
        );

        // Verify each version's row ID map
        for (i, (original_map, loaded_map)) in row_addr_maps
            .iter()
            .zip(loaded_index.row_addr_maps.iter())
            .enumerate()
        {
            assert_eq!(
                loaded_map.len(),
                original_map.len(),
                "Map size mismatch for version {}",
                i + 1
            );
            for (old_addr, new_addr) in original_map {
                assert_eq!(
                    loaded_map.get(old_addr),
                    Some(new_addr),
                    "Mismatch for old_addr {} in version {}",
                    old_addr,
                    i + 1
                );
            }
        }
    }
}
