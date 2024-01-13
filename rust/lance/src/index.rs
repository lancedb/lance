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

//! Secondary Index
//!

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::format::Fragment;
use lance_core::io::{
    read_message, read_message_from_buf, read_metadata_offset, reader::read_manifest_indexes,
    Reader,
};
use lance_index::pb::index::Implementation;
use lance_index::scalar::expression::IndexInformationProvider;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::ScalarIndex;
use lance_index::{pb, Index, IndexType, INDEX_FILE_NAME};
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

pub(crate) mod append;
pub(crate) mod cache;
pub(crate) mod prefilter;
pub mod scalar;
pub mod vector;

use crate::dataset::index::unindexed_fragments;
use crate::dataset::transaction::{Operation, Transaction};
use crate::format::Index as IndexMetadata;
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

pub trait IndexParams: Send + Sync {
    fn as_any(&self) -> &dyn Any;
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

/// Extends Dataset with secondary index.
#[async_trait]
pub trait DatasetIndexExt {
    /// Create indices on columns.
    ///
    /// Upon finish, a new dataset version is generated.
    ///
    /// Parameters:
    ///
    ///  - `columns`: the columns to build the indices on.
    ///  - `index_type`: specify [`IndexType`].
    ///  - `name`: optional index name. Must be unique in the dataset.
    ///            if not provided, it will auto-generate one.
    ///  - `params`: index parameters.
    ///  - `replace`: replace the existing index if it exists.
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<()>;

    /// Read all indices of this Dataset version.
    async fn load_indices(&self) -> Result<Vec<IndexMetadata>>;

    /// Loads a specific index with the given id
    async fn load_index(&self, uuid: &str) -> Result<Option<IndexMetadata>> {
        self.load_indices()
            .await
            .map(|indices| indices.into_iter().find(|idx| idx.uuid.to_string() == uuid))
    }

    /// Loads a specific index with the given index name
    async fn load_index_by_name(&self, name: &str) -> Result<Option<IndexMetadata>> {
        self.load_indices()
            .await
            .map(|indices| indices.into_iter().find(|idx| idx.name == name))
    }

    /// Loads a specific index with the given index name.
    async fn load_scalar_index_for_column(&self, col: &str) -> Result<Option<IndexMetadata>>;

    /// Optimize indices.
    async fn optimize_indices(&mut self) -> Result<()>;

    /// Find index with a given index_name and return its serialized statistics.
    async fn index_statistics(&self, index_name: &str) -> Result<Option<String>>;
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

async fn count_unindexed_rows(ds: &Dataset, index_uuid: &str) -> Result<Option<usize>> {
    let index = ds.load_index(index_uuid).await?;

    if let Some(index) = index {
        let unindexed_frags = unindexed_fragments(&index, ds).await?;
        let unindexed_rows = unindexed_frags
            .iter()
            .map(Fragment::num_rows)
            // sum the number of rows in each fragment if no fragment returned None from row_count
            .try_fold(0, |a, b| b.map(|b| a + b).ok_or(()));

        Ok(unindexed_rows.ok())
    } else {
        Ok(None)
    }
}

async fn count_indexed_rows(ds: &Dataset, index_uuid: &str) -> Result<Option<usize>> {
    let count_rows = ds.count_rows();
    let count_unindexed_rows = count_unindexed_rows(ds, index_uuid);

    let (count_rows, count_unindexed_rows) = futures::try_join!(count_rows, count_unindexed_rows)?;

    match count_unindexed_rows {
        Some(count_unindexed_rows) => Ok(Some(count_rows - count_unindexed_rows)),
        None => Ok(None),
    }
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
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(new_manifest);

        Ok(())
    }

    async fn load_indices(&self) -> Result<Vec<IndexMetadata>> {
        let manifest_file = self.manifest_file(self.version().version).await?;
        read_manifest_indexes(&self.object_store, &manifest_file, &self.manifest).await
    }

    async fn load_scalar_index_for_column(&self, col: &str) -> Result<Option<IndexMetadata>> {
        Ok(self
            .load_indices()
            .await?
            .into_iter()
            .filter(|idx| idx.fields.len() == 1)
            .find(|idx| {
                let field = self.schema().field_by_id(idx.fields[0]);
                if let Some(field) = field {
                    field.name == col
                } else {
                    false
                }
            }))
    }

    #[instrument(skip_all)]
    async fn optimize_indices(&mut self) -> Result<()> {
        let dataset = Arc::new(self.clone());
        // Append index
        let indices = self.load_indices().await?;

        let mut new_indices = vec![];
        let mut removed_indices = vec![];
        for idx in indices.as_slice() {
            if idx.dataset_version == self.manifest.version {
                continue;
            }
            let Some((new_id, new_frag_ids)) = append_index(dataset.clone(), idx).await? else {
                continue;
            };

            let new_idx = IndexMetadata {
                uuid: new_id,
                name: idx.name.clone(),
                fields: idx.fields.clone(),
                dataset_version: self.manifest.version,
                fragment_bitmap: new_frag_ids,
            };
            removed_indices.push(idx.clone());
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
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(new_manifest);
        Ok(())
    }

    async fn index_statistics(&self, index_name: &str) -> Result<Option<String>> {
        let index_uuid = self
            .load_index_by_name(index_name)
            .await?
            .map(|idx| idx.uuid.to_string());

        if let Some(index_uuid) = index_uuid {
            let index_statistics = self
                .open_generic_index("vector", &index_uuid)
                .await?
                .statistics()
                .unwrap();
            Ok(Some(index_statistics))
        } else {
            Ok(None)
        }
    }
}

/// A trait for internal dataset utilities
#[async_trait]
pub(crate) trait DatasetIndexInternalExt {
    /// Opens an index (scalar or vector) as a generic index
    async fn open_generic_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn Index>>;
    /// Opens the requested scalar index
    async fn open_scalar_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn ScalarIndex>>;
    /// Opens the requested vector index
    async fn open_vector_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn VectorIndex>>;
    /// Loads information about all the available scalar indices on the dataset
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo>;
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
}

#[cfg(test)]
mod tests {
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
}
