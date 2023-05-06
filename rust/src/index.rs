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

use async_trait::async_trait;
use uuid::Uuid;

/// Protobuf definitions for the index on-disk format.
pub mod pb {
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.index.pb.rs"));
}

mod cache;
pub mod vector;

use crate::dataset::write_manifest_file;
use crate::format::Index as IndexMetadata;
use crate::{dataset::Dataset, Error, Result};

use self::vector::{build_vector_index, VectorIndexParams};

pub trait Index {
    fn uuid(&self) -> &str;
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
            IndexType::Vector => write!(f, "Vector"),
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
    async fn create_index(
        &self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
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
    ) -> Result<Self> {
        if columns.len() != 1 {
            return Err(Error::Index(
                "Only support building index on 1 column at the moment".to_string(),
            ));
        }
        let column = columns[0];
        let Some(field) = self.schema().field(column) else {
            return Err(Error::Index(format!(
                "CreateIndex: column '{column}' does not exist"
            )));
        };

        // Load indices from the disk.
        let mut indices = self.load_indices().await?;

        let index_name = name.unwrap_or(format!("{column}_idx"));
        if indices.iter().any(|i| i.name == index_name) {
            return Err(Error::Index(format!(
                "Index name '{index_name} already exists'"
            )));
        }

        let index_id = Uuid::new_v4();
        match index_type {
            IndexType::Vector => {
                // Vector index params.
                let vec_params = params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| {
                        Error::Index("Vector index type must take a VectorIndexParams".to_string())
                    })?;

                build_vector_index(
                    self,
                    column,
                    &index_name,
                    &index_id.to_string(),
                    &vec_params,
                )
                .await?;
            }
        }

        let latest_manifest = self.latest_manifest().await?;
        let mut new_manifest = self.manifest.as_ref().clone();
        new_manifest.version = latest_manifest.version + 1;

        // Write index metadata down
        let new_idx = IndexMetadata::new(index_id, &index_name, &[field.id], new_manifest.version);
        indices.push(new_idx);

        write_manifest_file(&self.object_store, &mut new_manifest, Some(indices)).await?;

        Ok(Self {
            object_store: self.object_store.clone(),
            base: self.base.clone(),
            manifest: Arc::new(new_manifest),
        })
    }
}
