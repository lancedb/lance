// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod frag_reuse;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::dataset::optimize::remapping::RemapResult;
use crate::dataset::optimize::RemappedIndex;
use crate::index::remap_index;
use crate::index::scalar::infer_scalar_index_details;
use crate::Dataset;
use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::DatasetIndexExt;
use lance_table::format::pb::VectorIndexDetails;
use lance_table::format::IndexMetadata;
use serde::{Deserialize, Serialize};
use snafu::location;

use super::optimize::{IndexRemapper, IndexRemapperOptions};

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
        index: &IndexMetadata,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RemapResult> {
        remap_index(&self.dataset, &index.uuid, mapping).await
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
            let needs_remapped = index.name != FRAG_REUSE_INDEX_NAME
                && match &index.fragment_bitmap {
                    None => true,
                    Some(fragment_bitmap) => fragment_bitmap
                        .iter()
                        .any(|frag_idx| affected_frag_ids.contains(&(frag_idx as u64))),
                };
            if needs_remapped {
                let remap_result = self.remap_index(index, &mapping).await?;
                match remap_result {
                    RemapResult::Drop => continue,
                    RemapResult::Keep(id) => {
                        let index_details = match &index.index_details {
                            Some(index_details) => index_details.as_ref().clone(),
                            None => {
                                // Migration path, if we didn't store details before then use the default
                                // details.
                                assert!(index.fields.len() == 1);
                                let field = index.fields.first().unwrap();
                                let field =
                                    self.dataset.schema().field_by_id(*field).ok_or_else(|| {
                                        Error::Internal {
                                            message: format!(
                                                "Index {} references field {} which does not exist",
                                                index.uuid, field
                                            ),
                                            location: location!(),
                                        }
                                    })?;

                                if matches!(field.data_type(), DataType::FixedSizeList(..)) {
                                    prost_types::Any::from_msg(&VectorIndexDetails::default())?
                                } else {
                                    infer_scalar_index_details(&self.dataset, &field.name, index)
                                        .await?
                                        .as_ref()
                                        .clone()
                                }
                            }
                        };
                        remapped.push(RemappedIndex {
                            old_id: id,
                            new_id: id,
                            index_details,
                            index_version: index.index_version as u32,
                        });
                    }
                    RemapResult::Remapped(remapped_index) => {
                        remapped.push(remapped_index);
                    }
                }
            }
        }
        Ok(remapped)
    }
}

pub trait LanceIndexStoreExt {
    /// Create an index store for a new index (will always be absolute with no base id)
    fn from_dataset_for_new(dataset: &Dataset, uuid: &str) -> Result<Self>
    where
        Self: Sized;

    /// Open an index store for an existing index (might be relative or absolute)
    fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self>
    where
        Self: Sized;
}

impl LanceIndexStoreExt for LanceIndexStore {
    fn from_dataset_for_new(dataset: &Dataset, uuid: &str) -> Result<Self> {
        let index_dir = dataset.indices_dir().child(uuid);
        let cache = dataset.metadata_cache.file_metadata_cache(&index_dir);
        Ok(Self::new(
            dataset.object_store.clone(),
            index_dir,
            Arc::new(cache),
        ))
    }

    fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self> {
        let index_dir = dataset
            .indice_files_dir(index)?
            .child(index.uuid.to_string());
        let cache = dataset.metadata_cache.file_metadata_cache(&index_dir);
        Ok(Self::new(
            dataset.object_store.clone(),
            index_dir,
            Arc::new(cache),
        ))
    }
}
