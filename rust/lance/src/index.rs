// Copyright 2024 Lance Developers.
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

//! Secondary Index
//!

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use lance_index::pb::index::Implementation;
use lance_index::scalar::expression::IndexInformationProvider;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::ScalarIndex;
pub use lance_index::IndexParams;
use lance_index::{pb, DatasetIndexExt, Index, IndexType, INDEX_FILE_NAME};
use lance_io::traits::Reader;
use lance_io::utils::{read_message, read_message_from_buf, read_metadata_offset};
use lance_table::format::Fragment;
use lance_table::format::Index as IndexMetadata;
use lance_table::io::manifest::read_manifest_indexes;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

pub(crate) mod append;
pub(crate) mod cache;
pub(crate) mod prefilter;
pub mod scalar;
pub mod vector;

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::append::append_index;
use crate::index::vector::remap_vector_index;
use crate::io::commit::commit_transaction;
use crate::{dataset::Dataset, Error, Result};

use self::scalar::build_scalar_index;
use self::vector::{build_vector_index, VectorIndex, VectorIndexParams};

/// Builds index.
#[async_trait]
pub trait IndexBuilder {
    fn index_type() -> IndexType;

    async fn build(&self) -> Result<()>;
}

pub(crate) async fn remap_index(
    dataset: &Dataset,
    index_id: &Uuid,
    row_id_map: &HashMap<u64, Option<u64>>,
) -> Result<Uuid> {
    // Load indices from the disk.
    let indices = dataset.load_indices().await?;
    let matched = indices
        .iter()
        .find(|i| i.uuid == *index_id)
        .ok_or_else(|| Error::Index {
            message: format!("Index with id {} does not exist", index_id),
            location: location!(),
        })?;

    if matched.fields.len() > 1 {
        return Err(Error::Index {
            message: "Remapping indices with multiple fields is not supported".to_string(),
            location: location!(),
        });
    }
    let field = matched
        .fields
        .first()
        .expect("An index existed with no fields");

    let field = dataset.schema().field_by_id(*field).unwrap();

    let new_id = Uuid::new_v4();

    let generic = dataset
        .open_generic_index(&field.name, &index_id.to_string())
        .await?;

    match generic.index_type() {
        IndexType::Scalar => {
            let index_dir = dataset.indices_dir().child(new_id.to_string());
            let new_store = LanceIndexStore::new((*dataset.object_store).clone(), index_dir);

            let scalar_index = dataset
                .open_scalar_index(&field.name, &index_id.to_string())
                .await?;
            scalar_index.remap(row_id_map, &new_store).await?;
        }
        IndexType::Vector => {
            remap_vector_index(
                Arc::new(dataset.clone()),
                &field.name,
                index_id,
                &new_id,
                matched,
                row_id_map,
            )
            .await?;
        }
    }

    Ok(new_id)
}

#[derive(Debug)]
pub struct ScalarIndexInfo {
    indexed_columns: HashMap<String, DataType>,
}

impl IndexInformationProvider for ScalarIndexInfo {
    fn get_index(&self, col: &str) -> Option<&DataType> {
        self.indexed_columns.get(col)
    }
}

async fn open_index_proto(dataset: &Dataset, reader: &dyn Reader) -> Result<pb::Index> {
    let object_store = dataset.object_store();

    let file_size = reader.size().await?;
    let block_size = object_store.block_size();
    let begin = if file_size < block_size {
        0
    } else {
        file_size - block_size
    };
    let tail_bytes = reader.get_range(begin..file_size).await?;
    let metadata_pos = read_metadata_offset(&tail_bytes)?;
    let proto: pb::Index = if metadata_pos < file_size - tail_bytes.len() {
        // We have not read the metadata bytes yet.
        read_message(reader, metadata_pos).await?
    } else {
        let offset = tail_bytes.len() - (file_size - metadata_pos);
        read_message_from_buf(&tail_bytes.slice(offset..))?
    };
    Ok(proto)
}

#[async_trait]
impl DatasetIndexExt for Dataset {
    #[instrument(skip_all)]
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<()> {
        if columns.len() != 1 {
            return Err(Error::Index {
                message: "Only support building index on 1 column at the moment".to_string(),
                location: location!(),
            });
        }
        let column = columns[0];
        let Some(field) = self.schema().field(column) else {
            return Err(Error::Index {
                message: format!("CreateIndex: column '{column}' does not exist"),
                location: location!(),
            });
        };

        // Load indices from the disk.
        let indices = self.load_indices().await?;
        let index_name = name.unwrap_or(format!("{column}_idx"));
        if let Some(idx) = indices.iter().find(|i| i.name == index_name) {
            if idx.fields == [field.id] && !replace {
                return Err(Error::Index {
                    message: format!(
                        "Index name '{index_name} already exists, \
                        please specify a different name or use replace=True"
                    ),
                    location: location!(),
                });
            };
            if idx.fields != [field.id] {
                return Err(Error::Index {
                    message: format!(
                        "Index name '{index_name} already exists with different fields, \
                        please specify a different name"
                    ),
                    location: location!(),
                });
            }
        }

        let index_id = Uuid::new_v4();
        match index_type {
            IndexType::Scalar => {
                build_scalar_index(self, column, &index_id.to_string()).await?;
            }
            IndexType::Vector => {
                // Vector index params.
                let vec_params = params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| Error::Index {
                        message: "Vector index type must take a VectorIndexParams".to_string(),
                        location: location!(),
                    })?;

                build_vector_index(self, column, &index_name, &index_id.to_string(), vec_params)
                    .await?;
            }
        }

        let new_idx = IndexMetadata {
            uuid: index_id,
            name: index_name,
            fields: vec![field.id],
            dataset_version: self.manifest.version,
            fragment_bitmap: Some(self.get_fragments().iter().map(|f| f.id() as u32).collect()),
        };
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            None,
        );

        let new_manifest = commit_transaction(
            self,
            self.object_store(),
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(new_manifest);

        Ok(())
    }

    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let dataset_dir = self.base.to_string();
        if let Some(indices) = self
            .session
            .index_cache
            .get_metadata(&dataset_dir, self.version().version)
        {
            return Ok(indices);
        }

        let manifest_file = self.manifest_file(self.version().version).await?;
        let loaded_indices: Arc<Vec<IndexMetadata>> =
            read_manifest_indexes(&self.object_store, &manifest_file, &self.manifest)
                .await?
                .into();

        self.session.index_cache.insert_metadata(
            &dataset_dir,
            self.version().version,
            loaded_indices.clone(),
        );
        Ok(loaded_indices)
    }

    async fn load_scalar_index_for_column(&self, col: &str) -> Result<Option<IndexMetadata>> {
        Ok(self
            .load_indices()
            .await?
            .iter()
            .filter(|idx| idx.fields.len() == 1)
            .find(|idx| {
                let field = self.schema().field_by_id(idx.fields[0]);
                if let Some(field) = field {
                    field.name == col
                } else {
                    false
                }
            })
            .cloned())
    }

    #[instrument(skip_all)]
    async fn optimize_indices(&mut self, options: &OptimizeOptions) -> Result<()> {
        let dataset = Arc::new(self.clone());
        let indices = self.load_indices().await?;

        let name_to_indices = indices
            .iter()
            .map(|idx| (idx.name.clone(), idx))
            .into_group_map();

        let mut new_indices = vec![];
        let mut removed_indices = vec![];
        for deltas in name_to_indices.values() {
            let Some((new_id, removed, mut new_frag_ids)) =
                merge_indices(dataset.clone(), deltas.as_slice(), options).await?
            else {
                continue;
            };
            for removed_idx in removed.iter() {
                new_frag_ids |= removed_idx.fragment_bitmap.as_ref().unwrap();
            }

            let last_idx = deltas.last().expect("Delte indices should not be empty");
            let new_idx = IndexMetadata {
                uuid: new_id,
                name: last_idx.name.clone(), // Keep the same name
                fields: last_idx.fields.clone(),
                dataset_version: self.manifest.version,
                fragment_bitmap: Some(new_frag_ids),
            };
            removed_indices.extend(removed.iter().map(|&idx| idx.clone()));
            if deltas.len() > removed.len() {
                new_indices.extend(
                    deltas[0..(deltas.len() - removed.len())]
                        .iter()
                        .map(|&idx| idx.clone()),
                );
            }
            new_indices.push(new_idx);
        }

        if new_indices.is_empty() {
            return Ok(());
        }

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            },
            None,
        );

        let new_manifest = commit_transaction(
            self,
            self.object_store(),
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(new_manifest);
        Ok(())
    }

    async fn index_statistics(&self, index_name: &str) -> Result<String> {
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.is_empty() {
            return Err(Error::IndexNotFound {
                identity: format!("name={}", index_name),
                location: location!(),
            });
        }

        let column = self
            .schema()
            .field_by_id(metadatas[0].fields[0])
            .map(|f| f.name.as_str())
            .ok_or(Error::IndexNotFound {
                identity: index_name.to_string(),
                location: location!(),
            })?;

        // Open all delta indices
        let indices = stream::iter(metadatas.iter())
            .then(|m| async move { self.open_generic_index(column, &m.uuid.to_string()).await })
            .try_collect::<Vec<_>>()
            .await?;

        // Stastistics for each delta index.
        let indices_stats = indices
            .iter()
            .map(|idx| idx.statistics())
            .collect::<Result<Vec<_>>>()?;

        let unindexed_fragments = self.unindexed_fragments(index_name).await?;
        let mut num_unindexed_rows = 0;
        for f in unindexed_fragments.iter() {
            num_unindexed_rows += f.num_rows().ok_or(Error::Index {
                message: format!("fragment {} has no rows", f.id),
                location: location!(),
            })?;
        }
        let num_unindexed_fragments = unindexed_fragments.len();
        let num_indexed_fragments = self.fragments().len() - num_unindexed_fragments;
        let num_indexed_rows = self.count_rows().await? - num_unindexed_rows;

        let stats = json!({
            "index_type": indices_stats[0]["index_type"],
            "name": index_name,
            "num_indices": metadatas.len(),
            "indices": indices_stats,
            "num_indexed_fragments": num_indexed_fragments,
            "num_indexed_rows": num_indexed_rows,
            "num_unindexed_fragments": num_unindexed_fragments,
            "num_unindexed_rows": num_unindexed_rows,
        });

        serde_json::to_string(&stats).map_err(|e| Error::Index {
            message: format!("Failed to serialize index statistics: {}", e),
            location: location!(),
        })
    }
}

/// A trait for internal dataset utilities
///
/// Internal use only. No API stability guarantees.
#[async_trait]
pub(crate) trait DatasetIndexInternalExt: DatasetIndexExt {
    /// Opens an index (scalar or vector) as a generic index
    async fn open_generic_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn Index>>;
    /// Opens the requested scalar index
    async fn open_scalar_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn ScalarIndex>>;
    /// Opens the requested vector index
    async fn open_vector_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn VectorIndex>>;
    /// Loads information about all the available scalar indices on the dataset
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo>;

    /// Return the fragments that are not covered by any of the deltas of the index.
    async fn unindexed_fragments(&self, idx_name: &str) -> Result<Vec<Fragment>>;
}

#[async_trait]
impl DatasetIndexInternalExt for Dataset {
    async fn open_generic_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn Index>> {
        // Checking for cache existence is cheap so we just check both scalar and vector caches
        if let Some(index) = self.session.index_cache.get_scalar(uuid) {
            return Ok(index.as_index());
        }
        if let Some(index) = self.session.index_cache.get_vector(uuid) {
            return Ok(index.as_index());
        }

        // Sometimes we want to open an index and we don't care if it is a scalar or vector index.
        // For example, we might want to get statistics for an index, regardless of type.
        //
        // Currently, we solve this problem by checking for the existence of INDEX_FILE_NAME since
        // only vector indices have this file.  In the future, once we support multiple kinds of
        // scalar indices, we may start having this file with scalar indices too.  Once that happens
        // we can just read this file and look at the `implementation` or `index_type` fields to
        // determine what kind of index it is.
        let index_dir = self.indices_dir().child(uuid);
        let index_file = index_dir.child(INDEX_FILE_NAME);
        if self.object_store.exists(&index_file).await? {
            let index = self.open_vector_index(column, uuid).await?;
            Ok(index.as_index())
        } else {
            let index = self.open_scalar_index(column, uuid).await?;
            Ok(index.as_index())
        }
    }

    async fn open_scalar_index(&self, _column: &str, uuid: &str) -> Result<Arc<dyn ScalarIndex>> {
        if let Some(index) = self.session.index_cache.get_scalar(uuid) {
            return Ok(index);
        }

        let index = crate::index::scalar::open_scalar_index(self, uuid).await?;
        self.session.index_cache.insert_scalar(uuid, index.clone());
        Ok(index)
    }

    async fn open_vector_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn VectorIndex>> {
        if let Some(index) = self.session.index_cache.get_vector(uuid) {
            return Ok(index);
        }

        let index_dir = self.indices_dir().child(uuid);
        let index_file = index_dir.child(INDEX_FILE_NAME);
        let reader: Arc<dyn Reader> = self.object_store.open(&index_file).await?.into();

        let proto = open_index_proto(self, reader.as_ref()).await?;
        match &proto.implementation {
            Some(Implementation::VectorIndex(vector_index)) => {
                let dataset = Arc::new(self.clone());
                crate::index::vector::open_vector_index(
                    dataset,
                    column,
                    uuid,
                    vector_index,
                    index_dir,
                    reader,
                )
                .await
            }
            None => Err(Error::Internal {
                message: "Index proto was missing implementation field".into(),
                location: location!(),
            }),
        }
    }

    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo> {
        let indices = self.load_indices().await?;
        let schema = self.schema();
        let indexed_fields = indices
        .iter()
        .filter(|idx| idx.fields.len() == 1)
        .map(|idx| {
            let field = idx.fields[0];
            let field = schema.field_by_id(field).ok_or_else(|| Error::Internal { message: format!("Index referenced a field with id {field} which did not exist in the schema"), location: location!() });
            field.map(|field| (field.name.clone(), field.data_type()))
        }).collect::<Result<Vec<_>>>()?;
        let index_info_map = HashMap::from_iter(indexed_fields);
        Ok(ScalarIndexInfo {
            indexed_columns: index_info_map,
        })
    }

    async fn unindexed_fragments(&self, name: &str) -> Result<Vec<Fragment>> {
        let indices = self.load_indices_by_name(name).await?;
        let mut total_fragment_bitmap = RoaringBitmap::new();
        for idx in indices.iter() {
            total_fragment_bitmap |= idx.fragment_bitmap.as_ref().ok_or(Error::Index {
                message: "Please upgrade lance to 0.8+ to use this function".to_string(),
                location: location!(),
            })?;
        }
        Ok(self
            .fragments()
            .iter()
            .filter(|f| !total_fragment_bitmap.contains(f.id as u32))
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::builder::DatasetBuilder;

    use super::*;

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::*;
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_recreate_index() {
        const DIM: i32 = 8;
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "v",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
            Field::new(
                "o",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
        ]));
        let data = generate_random_array(2048 * DIM as usize);
        let batches: Vec<RecordBatch> = vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(FixedSizeListArray::try_new_from_values(data.clone(), DIM).unwrap()),
                Arc::new(FixedSizeListArray::try_new_from_values(data, DIM).unwrap()),
            ],
        )
        .unwrap()];

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(2, 8, 2, false, MetricType::L2, 2);
        dataset
            .create_index(&["v"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        dataset
            .create_index(&["o"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Create index again
        dataset
            .create_index(&["v"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Can not overwrite an index on different columns.
        assert!(dataset
            .create_index(
                &["v"],
                IndexType::Vector,
                Some("o_idx".to_string()),
                &params,
                true,
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_count_index_rows() {
        let test_dir = tempdir().unwrap();
        let dimensions = 16;
        let column_name = "vec";
        let field = Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let schema = Arc::new(Schema::new(vec![field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset.validate().await.unwrap();

        // Make sure it returns None if there's no index with the passed identifier
        assert!(dataset.index_statistics("bad_id").await.is_err());
        // Create an index
        let params = VectorIndexParams::ivf_pq(10, 8, 2, false, MetricType::L2, 10);
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);

        // Now we'll append some rows which shouldn't be indexed and see the
        // count change
        let float_arr = generate_random_array(512 * dimensions as usize);
        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema);
        dataset.append(reader, None).await.unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
    }

    #[tokio::test]
    async fn test_optimize_delta_indices() {
        let test_dir = tempdir().unwrap();
        let dimensions = 16;
        let column_name = "vec";
        let field = Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let schema = Arc::new(Schema::new(vec![field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        let params = VectorIndexParams::ivf_pq(10, 8, 2, false, MetricType::L2, 10);
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
        dataset.append(reader, None).await.unwrap();
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        dataset
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 0, // Just create index for delta
            })
            .await
            .unwrap();
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 2);

        dataset
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 2,
            })
            .await
            .unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 1);
    }
}
