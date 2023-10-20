// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashSet;
use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use lance_core::{
    format::{Fragment, Index},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

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
        for index in indices {
            let needs_remapped = match &index.fragment_bitmap {
                None => true,
                Some(fragment_bitmap) => fragment_bitmap
                    .iter()
                    .any(|frag_idx| affected_frag_ids.contains(&(frag_idx as u64))),
            };
            if needs_remapped {
                remapped.push(self.remap_index(&index, &mapping).await?);
            }
        }
        Ok(remapped)
    }
}

/// Returns the fragment ids that are not indexed by this index.
pub async fn unindexed_fragments(index: &Index, dataset: &Dataset) -> Result<Vec<Fragment>> {
    if index.dataset_version == dataset.version().version {
        return Ok(vec![]);
    }
    if let Some(bitmap) = index.fragment_bitmap.as_ref() {
        Ok(dataset
            .fragments()
            .iter()
            .filter(|f| !bitmap.contains(f.id as u32))
            .cloned()
            .collect::<Vec<_>>())
    } else {
        let ds = dataset.checkout_version(index.dataset_version).await?;
        let max_fragment_id_idx = ds.manifest.max_fragment_id().ok_or_else(|| Error::IO {
            message: "No fragments in index version".to_string(),
            location: location!(),
        })?;
        let max_fragment_id_ds = dataset
            .manifest
            .max_fragment_id()
            .ok_or_else(|| Error::IO {
                message: "No fragments in dataset version".to_string(),
                location: location!(),
            })?;
        if max_fragment_id_idx < max_fragment_id_ds {
            dataset.manifest.fragments_since(&ds.manifest)
        } else {
            Ok(vec![])
        }
    }
}
