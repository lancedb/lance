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

use async_trait::async_trait;
use snafu::{location, Location};
use uuid::Uuid;

pub(crate) mod append;
pub(crate) mod cache;
pub(crate) mod prefilter;
pub mod vector;

use crate::dataset::transaction::{Operation, Transaction};
use crate::format::Index as IndexMetadata;
use crate::index::append::append_index;
use crate::index::vector::remap_vector_index;
use crate::io::commit::commit_transaction;
use crate::{dataset::Dataset, Error, Result};
pub use lance_index::{pb, Index};
use lance_index::IndexType;

use self::vector::{build_vector_index, VectorIndexParams};


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

    remap_vector_index(
        Arc::new(dataset.clone()),
        &field.name,
        index_id,
        &new_id,
        matched,
        row_id_map,
    )
    .await?;

    Ok(new_id)
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

    /// Optimize indices.
    async fn optimize_indices(&mut self) -> Result<()>;
}

#[async_trait]
impl DatasetIndexExt for Dataset {
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
            let Some(new_id) = append_index(dataset.clone(), idx).await? else {
                continue;
            };

            let new_idx = IndexMetadata {
                uuid: new_id,
                name: idx.name.clone(),
                fields: idx.fields.clone(),
                dataset_version: self.manifest.version,
                fragment_bitmap: Some(self.get_fragments().iter().map(|f| f.id() as u32).collect()),
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
