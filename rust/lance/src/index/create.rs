// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::future::BoxFuture;
use lance_core::datatypes::Field;
use lance_index::{scalar::CreatedIndex, IndexParams, IndexType, VECTOR_INDEX_VERSION};
use lance_table::format::Index as IndexMetadata;
use snafu::location;
use std::{future::IntoFuture, sync::Arc};
use tracing::instrument;
use uuid::Uuid;

use crate::{
    dataset::{
        transaction::{Operation, Transaction},
        Dataset,
    },
    index::{
        scalar::{build_inverted_index, build_scalar_index},
        vector::{
            build_empty_vector_index, build_vector_index, VectorIndexParams, LANCE_VECTOR_INDEX,
        },
        vector_index_details, DatasetIndexExt, DatasetIndexInternalExt,
    },
    Error, Result,
};
use lance_index::{
    metrics::NoOpMetricsCollector,
    scalar::{inverted::tokenizer::InvertedIndexParams, ScalarIndexParams, LANCE_SCALAR_INDEX},
};

pub struct CreateIndexBuilder<'a> {
    dataset: &'a mut Dataset,
    columns: Vec<String>,
    index_type: IndexType,
    params: &'a dyn IndexParams,
    name: Option<String>,
    replace: bool,
    train: bool,
}

impl<'a> CreateIndexBuilder<'a> {
    pub fn new(
        dataset: &'a mut Dataset,
        columns: &[&str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> Self {
        Self {
            dataset,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            index_type,
            params,
            name: None,
            replace: false,
            train: true,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn replace(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }

    pub fn train(mut self, train: bool) -> Self {
        self.train = train;
        self
    }

    async fn execute(self) -> Result<()> {
        if self.columns.len() != 1 {
            if self.columns.is_empty() || self.index_type != IndexType::Inverted {
                return Err(Error::Index {
                    message: "Only support building index on 1 column at the moment".to_string(),
                    location: location!(),
                });
            }
            if self.name.is_some() {
                return Err(Error::Index {
                    message: "Only allow to specify name when building index on single column"
                        .to_string(),
                    location: location!(),
                });
            }
        }

        // If train is true but dataset is empty, automatically set train to false
        let train = if self.train {
            self.dataset.count_rows(None).await? > 0
        } else {
            false
        };

        let mut created_indices = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let field = self.dataset.schema().field(column).ok_or(Error::Index {
                message: format!("CreateIndex: column '{column}' does not exist"),
                location: location!(),
            })?;
            let index_name = self.name.clone().unwrap_or(format!("{column}_idx"));
            let index_id = Uuid::new_v4();
            let created_index = self
                .execute_impl(column, field, index_name.clone(), index_id, train)
                .await?;

            let new_idx = IndexMetadata {
                uuid: index_id,
                name: index_name,
                fields: vec![field.id],
                dataset_version: self.dataset.manifest.version,
                fragment_bitmap: if train {
                    // Include all fragments if training occurred
                    Some(
                        self.dataset
                            .get_fragments()
                            .iter()
                            .map(|f| f.id() as u32)
                            .collect(),
                    )
                } else {
                    // Empty bitmap for untrained indices
                    Some(roaring::RoaringBitmap::new())
                },
                index_details: Some(Arc::new(created_index.index_details)),
                index_version: created_index.index_version as i32,
                created_at: Some(chrono::Utc::now()),
                base_id: None,
            };
            created_indices.push(new_idx);
        }

        let transaction = Transaction::new(
            self.dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: created_indices,
                removed_indices: vec![],
            },
            /*blobs_op= */ None,
            None,
        );

        self.dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await?;
        Ok(())
    }

    #[instrument(skip_all)]
    async fn execute_impl(
        &self,
        column: &str,
        field: &Field,
        index_name: String,
        index_id: Uuid,
        train: bool,
    ) -> Result<CreatedIndex> {
        // Load indices from the disk.
        let indices = self.dataset.load_indices().await?;
        let fri = self
            .dataset
            .open_frag_reuse_index(&NoOpMetricsCollector)
            .await?;
        if let Some(idx) = indices.iter().find(|i| i.name == index_name) {
            if idx.fields == [field.id] && !self.replace {
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

        let created_index = match (self.index_type, self.params.index_name()) {
            (
                IndexType::Bitmap
                | IndexType::BTree
                | IndexType::Inverted
                | IndexType::NGram
                | IndexType::ZoneMap
                | IndexType::LabelList,
                LANCE_SCALAR_INDEX,
            ) => {
                let params = ScalarIndexParams::for_builtin(self.index_type.try_into()?);
                build_scalar_index(self.dataset, column, &index_id.to_string(), &params, train)
                    .await?
            }
            (IndexType::Scalar, LANCE_SCALAR_INDEX) => {
                // Guess the index type
                let params = self
                    .params
                    .as_any()
                    .downcast_ref::<ScalarIndexParams>()
                    .ok_or_else(|| Error::Index {
                        message: "Scalar index type must take a ScalarIndexParams".to_string(),
                        location: location!(),
                    })?;
                build_scalar_index(self.dataset, column, &index_id.to_string(), params, train)
                    .await?
            }
            (IndexType::Inverted, _) => {
                // Inverted index params.
                let inverted_params = self
                    .params
                    .as_any()
                    .downcast_ref::<InvertedIndexParams>()
                    .ok_or_else(|| Error::Index {
                        message: "Inverted index type must take a InvertedIndexParams".to_string(),
                        location: location!(),
                    })?;

                build_inverted_index(
                    self.dataset,
                    column,
                    &index_id.to_string(),
                    inverted_params,
                    train,
                )
                .await?
            }
            (IndexType::Vector, LANCE_VECTOR_INDEX) => {
                // Vector index params.
                let vec_params = self
                    .params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| Error::Index {
                        message: "Vector index type must take a VectorIndexParams".to_string(),
                        location: location!(),
                    })?;

                if train {
                    // this is a large future so move it to heap
                    Box::pin(build_vector_index(
                        self.dataset,
                        column,
                        &index_name,
                        &index_id.to_string(),
                        vec_params,
                        fri,
                    ))
                    .await?;
                } else {
                    // Create empty vector index
                    build_empty_vector_index(
                        self.dataset,
                        column,
                        &index_name,
                        &index_id.to_string(),
                        vec_params,
                    )
                    .await?;
                }
                CreatedIndex {
                    index_details: vector_index_details(),
                    index_version: VECTOR_INDEX_VERSION,
                }
            }
            // Can't use if let Some(...) here because it's not stable yet.
            // TODO: fix after https://github.com/rust-lang/rust/issues/51114
            (IndexType::Vector, name)
                if self
                    .dataset
                    .session
                    .index_extensions
                    .contains_key(&(IndexType::Vector, name.to_string())) =>
            {
                let ext = self
                    .dataset
                    .session
                    .index_extensions
                    .get(&(IndexType::Vector, name.to_string()))
                    .expect("already checked")
                    .clone()
                    .to_vector()
                    // this should never happen because we control the registration
                    // if this fails, the registration logic has a bug
                    .ok_or(Error::Internal {
                        message: "unable to cast index extension to vector".to_string(),
                        location: location!(),
                    })?;

                if train {
                    ext.create_index(self.dataset, column, &index_id.to_string(), self.params)
                        .await?;
                } else {
                    todo!("create empty vector index when train=false");
                }
                CreatedIndex {
                    index_details: vector_index_details(),
                    index_version: VECTOR_INDEX_VERSION,
                }
            }
            (IndexType::FragmentReuse, _) => {
                return Err(Error::Index {
                    message: "Fragment reuse index can only be created through compaction"
                        .to_string(),
                    location: location!(),
                })
            }
            (index_type, index_name) => {
                return Err(Error::Index {
                    message: format!(
                        "Index type {index_type} with name {index_name} is not supported"
                    ),
                    location: location!(),
                });
            }
        };

        Ok(created_index)
    }
}

impl<'a> IntoFuture for CreateIndexBuilder<'a> {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'a, Result<()>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}
