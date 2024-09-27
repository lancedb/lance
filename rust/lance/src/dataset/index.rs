// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::DatasetIndexExt;
use lance_table::format::Index;
use serde::{Deserialize, Serialize};

use crate::index::remap_index;
use crate::Dataset;

use super::optimize::{IndexRemapper, IndexRemapperOptions, RemappedIndex};

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetIndexRemapperOptions {}

impl IndexRemapperOptions for DatasetIndexRemapperOptions {
    fn create_remapper(
        &self,
        dataset: &Dataset,
    ) -> crate::Result<Box<dyn super::optimize::IndexRemapper>> {
        Ok(Box::new(DatasetIndexRemapper {
            dataset: Arc::new(dataset.clone()),
        }))
    }
}

struct DatasetIndexRemapper {
    dataset: Arc<Dataset>,
}

impl DatasetIndexRemapper {
    async fn remap_index(
        &self,
        index: &Index,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RemappedIndex> {
        let new_uuid = remap_index(&self.dataset, &index.uuid, mapping).await?;
        Ok(RemappedIndex::new(index.uuid, new_uuid))
    }
}

#[async_trait]
impl IndexRemapper for DatasetIndexRemapper {
    async fn remap_indices(
        &self,
        mapping: HashMap<u64, Option<u64>>,
        affected_fragment_ids: &[u64],
    ) -> Result<Vec<RemappedIndex>> {
        let affected_frag_ids = HashSet::<u64>::from_iter(affected_fragment_ids.iter().copied());
        let indices = self.dataset.load_indices().await?;
        let mut remapped = Vec::with_capacity(indices.len());
        for index in indices.iter() {
            let needs_remapped = match &index.fragment_bitmap {
                None => true,
                Some(fragment_bitmap) => fragment_bitmap
                    .iter()
                    .any(|frag_idx| affected_frag_ids.contains(&(frag_idx as u64))),
            };
            if needs_remapped {
                remapped.push(self.remap_index(index, &mapping).await?);
            }
        }
        Ok(remapped)
    }
}

pub trait LanceIndexStoreExt {
    fn from_dataset(dataset: &Dataset, uuid: &str) -> Self;
}

impl LanceIndexStoreExt for LanceIndexStore {
    fn from_dataset(dataset: &Dataset, uuid: &str) -> Self {
        let index_dir = dataset.indices_dir().child(uuid);
        Self::new(
            dataset.object_store.as_ref().clone(),
            index_dir,
            dataset.session.file_metadata_cache.clone(),
        )
    }
}
