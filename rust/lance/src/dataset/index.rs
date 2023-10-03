use std::collections::HashSet;
use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::format::{Fragment, Index};
use crate::index::DatasetIndexExt;
use crate::Dataset;
use crate::Result;

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

pub struct DatasetIndexRemapper {
    dataset: Arc<Dataset>,
}

impl DatasetIndexRemapper {
    async fn remap_index(
        &self,
        index: &Index,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RemappedIndex> {
        let new_uuid = self.dataset.remap_index(&index.uuid, mapping).await?;
        Ok(RemappedIndex::new(index.uuid, new_uuid))
    }
}

#[async_trait]
impl IndexRemapper for DatasetIndexRemapper {
    async fn remap_indices(
        &self,
        mapping: HashMap<u64, Option<u64>>,
        fragments: &[Fragment],
    ) -> Result<Vec<RemappedIndex>> {
        let affected_frag_ids: HashSet<u64> = fragments.iter().map(|f| f.id).collect();
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
