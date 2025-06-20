// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance_core::{Error, Result};
use snafu::location;

use crate::{optimize::OptimizeOptions, scalar::ScalarIndexType, IndexParams, IndexType};
use lance_table::format::Index;
use uuid::Uuid;

/// A set of criteria used to filter potential indices to use for a query
#[derive(Debug, Default)]
pub struct ScalarIndexCriteria<'a> {
    /// Only consider indices for this column (this also means the index
    /// maps to a single column)
    pub for_column: Option<&'a str>,
    /// Only consider indices with this name
    pub has_name: Option<&'a str>,
    /// Only consider indices with this type
    pub has_type: Option<ScalarIndexType>,
    /// Only consider indices that support exact equality
    pub supports_exact_equality: bool,
}

impl<'a> ScalarIndexCriteria<'a> {
    /// Only consider indices for this column (this also means the index
    /// maps to a single column)
    pub fn for_column(mut self, column: &'a str) -> Self {
        self.for_column = Some(column);
        self
    }

    /// Only consider indices with this name
    pub fn with_name(mut self, name: &'a str) -> Self {
        self.has_name = Some(name);
        self
    }

    /// Only consider indices with this type
    pub fn with_type(mut self, ty: ScalarIndexType) -> Self {
        self.has_type = Some(ty);
        self
    }

    /// Only consider indices that support exact equality
    ///
    /// This will disqualify, for example, the ngram and inverted indices
    /// or an index like a bloom filter
    pub fn supports_exact_equality(mut self) -> Self {
        self.supports_exact_equality = true;
        self
    }
}

// Extends Lance Dataset with secondary index.
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

    /// Drop indices by name.
    ///
    /// Upon finish, a new dataset version is generated.
    ///
    /// Parameters:
    ///
    /// - `name`: the name of the index to drop.
    async fn drop_index(&mut self, name: &str) -> Result<()>;

    /// Prewarm an index by name.
    ///
    /// This will load the index into memory and cache it.
    ///
    /// Generally, this should only be called when it is known the entire index will
    /// fit into the index cache.
    ///
    /// This is a hint that is not enforced by all indices today.  Some indices may choose
    /// to ignore this hint.
    async fn prewarm_index(&self, name: &str) -> Result<()>;

    /// Read all indices of this Dataset version.
    ///
    /// The indices are lazy loaded and cached in memory within the [`Dataset`] instance.
    /// The cache is invalidated when the dataset version (Manifest) is changed.
    async fn load_indices(&self) -> Result<Arc<Vec<Index>>>;

    /// Loads all the indies of a given UUID.
    ///
    /// Note that it is possible to have multiple indices with the same UUID,
    /// as they are the deltas of the same index.
    async fn load_index(&self, uuid: &str) -> Result<Option<Index>> {
        self.load_indices().await.map(|indices| {
            indices
                .iter()
                .find(|idx| idx.uuid.to_string() == uuid)
                .cloned()
        })
    }

    /// Loads a specific index with the given index name
    ///
    /// Returns
    /// -------
    /// - `Ok(indices)`: if the index exists, returns the index.
    /// - `Ok(vec![])`: if the index does not exist.
    /// - `Err(e)`: if there is an error loading indices.
    ///
    async fn load_indices_by_name(&self, name: &str) -> Result<Vec<Index>> {
        self.load_indices().await.map(|indices| {
            indices
                .iter()
                .filter(|idx| idx.name == name)
                .cloned()
                .collect()
        })
    }

    /// Loads a specific index with the given index name.
    /// This function only works for indices that are unique.
    /// If there are multiple indices sharing the same name, please use [load_indices_by_name]
    ///
    /// Returns
    /// -------
    /// - `Ok(Some(index))`: if the index exists, returns the index.
    /// - `Ok(None)`: if the index does not exist.
    /// - `Err(e)`: Index error if there are multiple indexes sharing the same name.
    ///
    async fn load_index_by_name(&self, name: &str) -> Result<Option<Index>> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            Ok(None)
        } else if indices.len() == 1 {
            Ok(Some(indices[0].clone()))
        } else {
            Err(Error::Index {
                message: format!("Found multiple indices of the same name: {:?}, please use load_indices_by_name", 
                    indices.iter().map(|idx| &idx.name).collect::<Vec<_>>()),
                location: location!(),
            })
        }
    }

    /// Loads a specific index with the given index name.
    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: ScalarIndexCriteria<'b>,
    ) -> Result<Option<Index>>;

    /// Optimize indices.
    async fn optimize_indices(&mut self, options: &OptimizeOptions) -> Result<()>;

    /// Find index with a given index_name and return its serialized statistics.
    ///
    /// If the index does not exist, return Error.
    async fn index_statistics(&self, index_name: &str) -> Result<String>;

    async fn commit_existing_index(
        &mut self,
        index_name: &str,
        column: &str,
        index_id: Uuid,
    ) -> Result<()>;

    async fn read_index_partition(
        &self,
        index_name: &str,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream>;
}
