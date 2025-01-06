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

use std::sync::Arc;

use lance_io::object_store::{ObjectStore, ObjectStoreParams, WrappingObjectStore};
use lance_table::io::commit::CommitHandler;

use crate::{Result};

/// Contains all the information that Lance needs to access a table
pub struct TableReference {
    /// Base URI of the table
    pub uri: String,
    /// Object store wrapper used to access the table
    pub store_wrapper: Option<Arc<dyn WrappingObjectStore>>,
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
    async fn create_table(&self, name: &str) -> Result<TableReference>;
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
    base_path: object_store::path::Path,
    object_store: ObjectStore,
}

const LANCE_EXTENSION: &str = "lance";

impl ListingCatalog {
    pub fn new(base_path: object_store::path::Path, object_store: ObjectStore) -> Self {
        Self {
            base_path,
            object_store,
        }
    }
}

#[async_trait::async_trait]
impl Catalog for ListingCatalog {
    async fn create_table(&self, name: &str) -> Result<TableReference> {
        todo!()
    }

    async fn drop_table(&self, name: &str) -> Result<()> {
        todo!()
    }

    async fn get_table(&self, name: &str) -> Result<TableReference> {
        todo!()
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