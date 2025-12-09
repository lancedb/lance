// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance_core::{Error, Result};
use snafu::location;

use crate::{optimize::OptimizeOptions, IndexParams, IndexType};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

/// A set of criteria used to filter potential indices to use for a query
#[derive(Debug, Default)]
pub struct IndexCriteria<'a> {
    /// Only consider indices for this column (this also means the index
    /// maps to a single column)
    pub for_column: Option<&'a str>,
    /// Only consider indices with this name
    pub has_name: Option<&'a str>,
    /// If true, only consider indices that support FTS
    pub must_support_fts: bool,
    /// If true, only consider indices that support exact equality
    pub must_support_exact_equality: bool,
}

impl<'a> IndexCriteria<'a> {
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

    /// Only consider indices that support FTS
    pub fn supports_fts(mut self) -> Self {
        self.must_support_fts = true;
        self
    }

    /// Only consider indices that support exact equality
    ///
    /// This will disqualify, for example, the ngram and inverted indices
    /// or an index like a bloom filter
    pub fn supports_exact_equality(mut self) -> Self {
        self.must_support_exact_equality = true;
        self
    }
}

#[deprecated(since = "0.39.0", note = "Use IndexCriteria instead")]
pub type ScalarIndexCriteria<'a> = IndexCriteria<'a>;

/// Additional information about an index
///
/// Note that a single index might consist of multiple segments.  Each segment has its own
/// UUID and collection of files and covers some subset of the data fragments.
///
/// All segments in an index should have the same index type and index details.
pub trait IndexDescription: Send + Sync {
    /// Returns the index name
    ///
    /// This is the user-defined name of the index.  It is shared by all segments of the index
    /// and is what is used to refer to the index in the API.  It is guaranteed to be unique
    /// within the dataset.
    fn name(&self) -> &str;

    /// Returns the index metadata
    ///
    /// This is the raw metadata information stored in the manifest.  There is one
    /// IndexMetadata for each segment of the index.
    fn metadata(&self) -> &[IndexMetadata];

    /// Returns the index type URL
    ///
    /// This is extracted from the type url of the index details
    fn type_url(&self) -> &str;

    /// Returns the index type
    ///
    /// This is a short string identifier that is friendlier than the type URL but not
    /// guaranteed to be unique.
    ///
    /// This is calculated by the plugin and will be "Unknown" if no plugin could be found
    /// for the type URL.
    fn index_type(&self) -> &str;

    /// Returns the number of rows indexed by the index, across all segments.
    ///
    /// This is an approximate count and may include rows that have been
    /// deleted.
    fn rows_indexed(&self) -> u64;

    /// Returns the ids of the fields that the index is built on.
    fn field_ids(&self) -> &[u32];

    /// Returns a JSON string representation of the index details
    ///
    /// The format of these details will vary depending on the index type and
    /// since indexes can be provided by plugins we cannot fully define it here.
    ///
    /// However, plugins should do their best to maintain backwards compatibility
    /// and consider this method part of the public API.
    ///
    /// See individual index plugins for more description of the expected format.
    ///
    /// The conversion from Any to JSON is controlled by the index
    /// plugin.  As a result, this method may fail if there is no plugin
    /// available for the index.
    fn details(&self) -> Result<String>;
}

// Extends Lance Dataset with secondary index.
#[async_trait]
pub trait DatasetIndexExt {
    type IndexBuilder<'a>
    where
        Self: 'a;

    /// Create a builder for creating an index on columns.
    ///
    /// This returns a builder that can be configured with additional options
    /// like `name()`, `replace()`, and `train()` before awaiting to execute.
    ///
    /// # Parameters
    /// - `columns`: the columns to build the indices on.
    /// - `index_type`: specify [`IndexType`].
    /// - `params`: index parameters.
    fn create_index_builder<'a>(
        &'a mut self,
        columns: &'a [&'a str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> Self::IndexBuilder<'a>;

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
    /// The indices are lazy loaded and cached in memory within the `Dataset` instance.
    /// The cache is invalidated when the dataset version (Manifest) is changed.
    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>>;

    /// Loads all the indies of a given UUID.
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

    /// Loads a specific index with the given index name
    ///
    /// Returns
    /// -------
    /// - `Ok(indices)`: if the index exists, returns the index.
    /// - `Ok(vec![])`: if the index does not exist.
    /// - `Err(e)`: if there is an error loading indices.
    ///
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
    /// If there are multiple indices sharing the same name, please use [`Self::load_indices_by_name`]
    ///
    /// Returns
    /// -------
    /// - `Ok(Some(index))`: if the index exists, returns the index.
    /// - `Ok(None)`: if the index does not exist.
    /// - `Err(e)`: Index error if there are multiple indexes sharing the same name.
    ///
    async fn load_index_by_name(&self, name: &str) -> Result<Option<IndexMetadata>> {
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

    /// Describes indexes in a dataset
    ///
    /// This method should only access the index metadata and should not load the index into memory.
    ///
    /// More detailed information may be available from `index_statistics` but that will require
    /// loading the index into memory.
    async fn describe_indices<'a, 'b>(
        &'a self,
        criteria: Option<IndexCriteria<'b>>,
    ) -> Result<Vec<Arc<dyn IndexDescription>>>;

    /// Loads a specific index with the given index name.
    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: IndexCriteria<'b>,
    ) -> Result<Option<IndexMetadata>>;

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
