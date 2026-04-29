// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod frag_reuse;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Dataset;
use crate::dataset::optimize::RemappedIndex;
use crate::dataset::optimize::remapping::RemapResult;
use crate::index::DatasetIndexExt;
use crate::index::remap_index;
use crate::index::scalar::infer_scalar_index_details;
use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_encoding::version::LanceFileVersion;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use lance_table::format::pb::VectorIndexDetails;
use serde::{Deserialize, Serialize};

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
                                        Error::internal(format!(
                                            "Index {} references field {} which does not exist",
                                            index.uuid, field
                                        ))
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
                            files: index.files.clone(),
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

#[async_trait]
pub trait LanceIndexStoreExt {
    /// Create an index store for a new index (will always be absolute with no base id)
    fn from_dataset_for_new(dataset: &Dataset, uuid: &str) -> Result<Self>
    where
        Self: Sized;

    /// Open an index store for an existing index (might be relative or absolute)
    async fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self>
    where
        Self: Sized;
}

/// Extract the lance file version from a dataset, floored at V2_0.
///
/// Index files should never use the legacy format. If the dataset uses legacy
/// format or doesn't have a version set, V2_0 is used as the minimum.
pub(crate) fn dataset_format_version(dataset: &Dataset) -> LanceFileVersion {
    dataset
        .manifest
        .data_storage_format
        .lance_file_version()
        .ok()
        .map(|v| v.resolve().max(LanceFileVersion::V2_0))
        .unwrap_or(LanceFileVersion::V2_0)
}

#[async_trait]
impl LanceIndexStoreExt for LanceIndexStore {
    fn from_dataset_for_new(dataset: &Dataset, uuid: &str) -> Result<Self> {
        let index_dir = dataset.indices_dir().child(uuid);
        let cache = dataset.metadata_cache.file_metadata_cache(&index_dir);
        let format_version = dataset_format_version(dataset);
        Ok(Self::with_format_version(
            dataset.object_store.clone(),
            index_dir,
            Arc::new(cache),
            format_version,
        ))
    }

    async fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self> {
        let index_dir = dataset
            .indice_files_dir(index)?
            .child(index.uuid.to_string());
        let cache = dataset.metadata_cache.file_metadata_cache(&index_dir);
        let format_version = dataset_format_version(dataset);
        let object_store = dataset.object_store_for_index(index).await?;
        let store =
            Self::with_format_version(object_store, index_dir, Arc::new(cache), format_version);
        Ok(store.with_file_sizes(index.file_size_map()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_index::IndexType;
    use lance_linalg::distance::MetricType;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_remapper_only_touches_segments_with_affected_fragments() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(16.into()),
            )
            .into_reader_rows(RowCount::from(40), BatchCount::from(2));

        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 20,
                max_rows_per_group: 20,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(
            fragments.len() >= 2,
            "expected at least two fragments for this test"
        );
        let target_fragments = fragments.iter().take(2).collect::<Vec<_>>();

        let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        let first_segment_uuid = Uuid::new_v4();
        let second_segment_uuid = Uuid::new_v4();
        let built_index = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .index_uuid(first_segment_uuid.to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        let first_segment_dir = dataset.indices_dir().child(first_segment_uuid.to_string());
        let second_segment_dir = dataset.indices_dir().child(second_segment_uuid.to_string());
        for file_name in ["index.idx", "auxiliary.idx"] {
            dataset
                .object_store
                .as_ref()
                .copy(
                    &first_segment_dir.child(file_name),
                    &second_segment_dir.child(file_name),
                )
                .await
                .unwrap();
        }

        let segments = [
            IndexMetadata {
                uuid: first_segment_uuid,
                fragment_bitmap: Some(std::iter::once(target_fragments[0].id() as u32).collect()),
                ..built_index.clone()
            },
            IndexMetadata {
                uuid: second_segment_uuid,
                fragment_bitmap: Some(std::iter::once(target_fragments[1].id() as u32).collect()),
                ..built_index
            },
        ];

        let segments = segments
            .iter()
            .map(|segment| {
                crate::index::IndexSegment::new(
                    segment.uuid,
                    segment
                        .fragment_bitmap
                        .as_ref()
                        .expect("test segment metadata should have fragment coverage")
                        .iter(),
                    segment
                        .index_details
                        .as_ref()
                        .expect("test segment metadata should have index details")
                        .clone(),
                    segment.index_version,
                )
            })
            .collect::<Vec<_>>();

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();
        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        let committed_ids = committed
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();
        let unaffected_segment_id = committed
            .iter()
            .find(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|bitmap| bitmap.contains(target_fragments[1].id() as u32))
            })
            .map(|segment| segment.uuid)
            .expect("expected one committed segment to cover the unaffected fragment");

        let remapper = DatasetIndexRemapperOptions::default()
            .create_remapper(&dataset)
            .unwrap();
        let remapped = remapper
            .remap_indices(HashMap::new(), &[target_fragments[0].id() as u64])
            .await
            .unwrap();

        assert_eq!(remapped.len(), 1);
        assert!(committed_ids.contains(&remapped[0].old_id));
        assert_ne!(remapped[0].old_id, unaffected_segment_id);
        assert_ne!(remapped[0].new_id, unaffected_segment_id);
    }
}
