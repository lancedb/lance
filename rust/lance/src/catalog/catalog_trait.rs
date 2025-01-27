// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatchIterator;
use arrow_schema::Schema;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::path::LancePathExt;
use lance_io::object_store::{ObjectStore, ObjectStoreExt, ObjectStoreParams, WrappingObjectStore};
use lance_table::format::{Index, Manifest};
use lance_table::io::commit::{CommitError, CommitHandler, ManifestNamingScheme, ManifestWriter};
use object_store::{path::Path, ObjectStore as OSObjectStore};
use std::fmt::Debug;
use std::sync::Arc;
use url::Url;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::WriteParams;
use crate::{Dataset, Result};

/// Contains all the information that Lance needs to access a table
pub struct TableReference {
    /// Base URI of the table
    pub uri: String,
    /// Object store wrapper used to access the table
    pub object_store: ObjectStore,
    /// Commit handler used to handle new commits to the table
    pub commit_handler: Arc<dyn CommitHandler>,
    /// Parameters used to read / write to the object store
    pub store_params: Option<ObjectStoreParams>,
}

/// Trait to be implemented by any catalog that Lance can use
#[async_trait::async_trait]
pub trait Catalog: std::fmt::Debug + Send + Sync {
    /// Create a new table in the catalog
    ///
    /// Returns a table reference that can be used to read/write to the table
    async fn create_table(&self, name: &str, schema: Arc<Schema>) -> Result<TableReference>;
    /// Drop a table from the catalog
    async fn drop_table(&self, name: &str) -> Result<()>;
    /// Get a reference to a table in the catalog
    ///
    /// Returns a table reference that can be used to read/write to the table
    async fn get_table(&self, name: &str) -> Result<TableReference>;
    /// List all tables in the catalog
    ///
    /// The `start_after` parameter is an optional table name that, if provided, will
    /// start the listing after the named table. If the named table is not found, the
    /// listing will start after the table that would be named if it existed.
    ///
    /// The `limit` parameter is an optional limit on the number of tables returned.
    async fn list_tables(
        &self,
        start_after: Option<String>,
        limit: Option<u32>,
    ) -> Result<Vec<String>>;
}

/// A simplified catalog that puts all tables in a single base directory
///
/// The object store's CAS primitives are used for commit handling.
///
/// This object store is simplistic but has zero dependencies (beyond an object store
/// of some kind)
#[derive(Debug)]
pub struct ListingCatalog {
    base_path: Path,
    object_store: ObjectStore,
    commit_handler: Arc<dyn CommitHandler>,
}

const LANCE_EXTENSION: &str = "lance";

fn format_table_url_or_path(is_url: bool, scheme: &str, base_path: &str, name: &str) -> String {
    if is_url {
        format!("{}:///{}/{}.{}", scheme, base_path, name, LANCE_EXTENSION)
    } else {
        format!("{}/{}.{}", base_path, name, LANCE_EXTENSION)
    }
}

impl ListingCatalog {
    pub fn new(base_path: Path, object_store: ObjectStore) -> Self {
        Self {
            base_path,
            object_store: object_store.clone(),
            commit_handler: Arc::new(ListingCommitHandler {
                object_store: object_store.into(),
            }),
        }
    }
}

#[async_trait::async_trait]
impl Catalog for ListingCatalog {
    async fn create_table(&self, name: &str, schema: Arc<Schema>) -> Result<TableReference> {
        let table_url = format_table_url_or_path(
            true,
            &*self.object_store.scheme,
            self.base_path.as_ref(),
            name,
        );

        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone()),
            &table_url,
            Some(WriteParams {
                commit_handler: Option::from(self.commit_handler.clone()),
                store_params: Some(ObjectStoreParams {
                    object_store: Some((
                        self.object_store.clone().inner,
                        Url::parse(&*table_url).unwrap(),
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .await?;

        Ok(TableReference {
            uri: ds.uri().to_string(),
            object_store: self.object_store.clone(),
            commit_handler: self.commit_handler.clone(),
            store_params: Some(ObjectStoreParams {
                object_store: Some((
                    self.object_store.clone().inner,
                    Url::parse(&*table_url).unwrap(),
                )),
                ..Default::default()
            }),
        })
    }

    async fn drop_table(&self, name: &str) -> Result<()> {
        let table_path = format_table_url_or_path(
            false,
            &*self.object_store.scheme,
            self.base_path.as_ref(),
            name,
        );

        self.object_store.remove_dir_all(table_path).await?;

        Ok(())
    }

    async fn get_table(&self, name: &str) -> Result<TableReference> {
        let table_url = format_table_url_or_path(
            true,
            &*self.object_store.scheme,
            self.base_path.as_ref(),
            name,
        );

        let ds = DatasetBuilder::from_uri(&table_url)
            .with_commit_handler(self.commit_handler.clone())
            .with_object_store(
                self.object_store.clone().inner,
                Url::parse(&*table_url).unwrap(),
                self.commit_handler.clone(),
            )
            .load()
            .await?;

        Ok(TableReference {
            uri: ds.uri().to_string(),
            object_store: self.object_store.clone(),
            commit_handler: self.commit_handler.clone(),
            store_params: Some(ObjectStoreParams {
                object_store: Some((
                    self.object_store.clone().inner,
                    Url::parse(&*table_url).unwrap(),
                )),
                ..Default::default()
            }),
        })
    }

    async fn list_tables(
        &self,
        start_after: Option<String>,
        limit: Option<u32>,
    ) -> Result<Vec<String>> {
        let mut f = self
            .object_store
            .read_dir(self.base_path.clone())
            .await?
            .iter()
            .map(std::path::Path::new)
            .filter(|path| {
                let is_lance = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e == LANCE_EXTENSION);
                is_lance.unwrap_or(false)
            })
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str().map(String::from)))
            .collect::<Vec<String>>();
        f.sort();
        if let Some(start_after) = start_after {
            let index = f
                .iter()
                .position(|name| name.as_str() > start_after.as_str())
                .unwrap_or(f.len());
            f.drain(0..index);
        }
        if let Some(limit) = limit {
            f.truncate(limit as usize);
        }
        Ok(f)
    }
}

pub struct ListingCommitHandler {
    object_store: ObjectStore,
}

impl Debug for ListingCommitHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ListingCommitHandler").finish()
    }
}

#[async_trait]
impl CommitHandler for ListingCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
    ) -> std::result::Result<Path, CommitError> {
        let version_path = naming_scheme.manifest_path(base_path, manifest.version);
        manifest_writer(object_store, manifest, indices, &version_path).await?;

        Ok(version_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

    #[tokio::test]
    async fn test_listing_catalog() {
        let object_store = ObjectStore::memory();
        let base_path = Path::parse("/catalog").unwrap();

        // Init a listing catalog
        let catalog = ListingCatalog::new(base_path.clone(), object_store.clone());

        // Create a table
        let field_a = ArrowField::new("a", DataType::Int32, true);
        let schema = Arc::new(ArrowSchema::new(vec![field_a.clone()]));
        let table_ref = catalog
            .create_table("test_table", schema.clone())
            .await
            .unwrap();

        // Verify the table was created
        let table_ref_fetched = catalog.get_table("test_table").await.unwrap();
        assert_eq!(table_ref.uri, table_ref_fetched.uri);

        // List tables
        let tables = catalog.list_tables(None, None).await.unwrap();
        assert_eq!(tables, vec!["test_table"]);

        // Drop the table
        catalog.drop_table("test_table").await.unwrap();

        // Verify the table was dropped
        let tables = catalog.list_tables(None, None).await.unwrap();
        assert!(tables.is_empty());
    }
}
