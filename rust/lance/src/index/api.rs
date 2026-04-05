// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance_index::{IndexParams, IndexType, optimize::OptimizeOptions};
use lance_table::format::IndexMetadata;

use crate::{Error, Result};

/// Extends [`crate::Dataset`] with secondary index APIs.
#[async_trait]
pub trait DatasetIndexExt {
    type IndexBuilder<'a>
    where
        Self: 'a;

    /// Create a builder for creating an index on columns.
    ///
    /// This returns a builder that can be configured with additional options
    /// like `name()`, `replace()`, and `train()` before awaiting to execute.
    fn create_index_builder<'a>(
        &'a mut self,
        columns: &'a [&'a str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> Self::IndexBuilder<'a>;

    /// Create indices on columns.
    ///
    /// Upon finish, a new dataset version is generated.
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<IndexMetadata>;

    /// Drop indices by name.
    ///
    /// Upon finish, a new dataset version is generated.
    async fn drop_index(&mut self, name: &str) -> Result<()>;

    /// Prewarm an index by name.
    ///
    /// This will load the index into memory and cache it.
    async fn prewarm_index(&self, name: &str) -> Result<()>;

    /// Read all indices of this Dataset version.
    ///
    /// The indices are lazy loaded and cached in memory within the `Dataset` instance.
    /// The cache is invalidated when the dataset version (Manifest) is changed.
    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>>;

    /// Loads all the indices of a given UUID.
    ///
    /// Note that it is possible to have multiple indices with the same UUID,
    /// as they are the deltas of the same index.
    async fn load_index(&self, uuid: &str) -> Result<Option<IndexMetadata>> {
        self.load_indices().await.map(|indices| {
            indices
                .iter()
                .find(|idx| idx.uuid.to_string() == uuid)
                .cloned()
        })
    }

    /// Loads a specific index with the given index name.
    ///
    /// Returns `Ok(vec![])` if the index does not exist.
    async fn load_indices_by_name(&self, name: &str) -> Result<Vec<IndexMetadata>> {
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
    /// If there are multiple indices sharing the same name, please use [`Self::load_indices_by_name`].
    async fn load_index_by_name(&self, name: &str) -> Result<Option<IndexMetadata>> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            Ok(None)
        } else if indices.len() == 1 {
            Ok(Some(indices[0].clone()))
        } else {
            Err(Error::index(format!(
                "Found multiple indices of the same name: {:?}, please use load_indices_by_name",
                indices.iter().map(|idx| &idx.name).collect::<Vec<_>>()
            )))
        }
    }

    /// Describes indexes in a dataset.
    ///
    /// This method should only access the index metadata and should not load the index into memory.
    async fn describe_indices<'a, 'b>(
        &'a self,
        criteria: Option<lance_index::IndexCriteria<'b>>,
    ) -> Result<Vec<Arc<dyn lance_index::IndexDescription>>>;

    /// Loads a specific scalar index using the provided criteria.
    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: lance_index::IndexCriteria<'b>,
    ) -> Result<Option<IndexMetadata>>;

    /// Optimize indices.
    async fn optimize_indices(&mut self, options: &OptimizeOptions) -> Result<()>;

    /// Find an index with the given name and return its serialized statistics.
    async fn index_statistics(&self, index_name: &str) -> Result<String>;

    /// Merge one caller-defined group of existing uncommitted index segments into a
    /// single segment.
    async fn merge_existing_index_segments(
        &self,
        segments: Vec<IndexMetadata>,
    ) -> Result<IndexMetadata>;

    /// Commit one or more existing physical index segments as a logical index.
    async fn commit_existing_index_segments(
        &mut self,
        index_name: &str,
        column: &str,
        segments: Vec<IndexMetadata>,
    ) -> Result<()>;

    async fn read_index_partition(
        &self,
        index_name: &str,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream>;
}
