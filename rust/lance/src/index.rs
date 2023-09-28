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
use std::fmt;
use std::sync::Arc;

use arrow_array::UInt64Array;
use async_trait::async_trait;
use uuid::Uuid;

/// Protobuf definitions for the index on-disk format.
pub mod pb {
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.index.pb.rs"));
}

pub(crate) mod cache;
pub(crate) mod prefilter;
pub mod vector;

use crate::dataset::transaction::{Operation, Transaction};
use crate::format::Index as IndexMetadata;
use crate::io::commit::commit_transaction;
use crate::session::Session;
use crate::{dataset::Dataset, Error, Result};

use self::vector::{build_vector_index, VectorIndexParams};

/// Trait of a secondary index.
#[async_trait]
pub(crate) trait Index: Send + Sync {
    /// Cast to [Any].
    fn as_any(&self) -> &dyn Any;

    // TODO: if we ever make this public, do so in such a way that `serde_json`
    // isn't exposed at the interface. That way mismatched versions isn't an issue.
    fn statistics(&self) -> Result<serde_json::Value>;

    /// Remap row id. This is used when the dataset goes thru compaction.
    /// return an **new** index id with remapped row id.
    async fn remap_row_id(&self, _mapper: fn(UInt64Array) -> UInt64Array) -> Result<String> {
        Err(Error::Index {
            message: "Index does not support remapping".to_string(),
        })
    }
}

/// Index Type
pub enum IndexType {
    // Preserve 0-100 for simple indices.

    // 100+ and up for vector index.
    /// Flat vector index.
    Vector = 100,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Vector => write!(f, "Vector"),
        }
    }
}

/// Builds index.
#[async_trait]
pub trait IndexBuilder {
    fn index_type() -> IndexType;

    async fn build(&self) -> Result<()>;
}

pub trait IndexParams: Send + Sync {
    fn as_any(&self) -> &dyn Any;
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
        &self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<Dataset>;
}

#[async_trait]
impl DatasetIndexExt for Dataset {
    async fn create_index(
        &self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<Self> {
        if columns.len() != 1 {
            return Err(Error::Index {
                message: "Only support building index on 1 column at the moment".to_string(),
            });
        }
        let column = columns[0];
        let Some(field) = self.schema().field(column) else {
            return Err(Error::Index {
                message: format!("CreateIndex: column '{column}' does not exist"),
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
                });
            };
            if idx.fields != [field.id] {
                return Err(Error::Index {
                    message: format!(
                        "Index name '{index_name} already exists with different fields, \
                        please specify a different name"
                    ),
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
                    })?;

                build_vector_index(self, column, &index_name, &index_id.to_string(), vec_params)
                    .await?;
            }
        }

        let new_idx = IndexMetadata::new(index_id, &index_name, &[field.id], self.manifest.version);
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
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

        Ok(Self {
            object_store: self.object_store.clone(),
            base: self.base.clone(),
            manifest: Arc::new(new_manifest),
            session: Arc::new(Session::default()),
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
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(2, 8, 2, false, MetricType::L2, 2);
        let dataset = dataset
            .create_index(&["v"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        let dataset = dataset
            .create_index(&["o"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Create index again
        let dataset = dataset
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
