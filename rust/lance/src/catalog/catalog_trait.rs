// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::catalog::database::Database;
use crate::catalog::dataset_identifier::DatasetIdentifier;
use crate::dataset::Dataset;
use std::collections::{HashMap, HashSet};

pub trait Catalog {
    /// Initialize the catalog.
    fn initialize(&self, name: &str, properties: &HashMap<&str, &str>) -> Result<(), String>;

    /// List all datasets under a specified database.
    fn list_datasets(&self, database: &Database) -> Vec<DatasetIdentifier>;

    /// Create a new dataset in the catalog.
    fn create_dataset(
        &self,
        identifier: &DatasetIdentifier,
        location: &str,
    ) -> Result<Dataset, String>;

    /// Check if a dataset exists in the catalog.
    fn dataset_exists(&self, identifier: &DatasetIdentifier) -> bool;

    /// Drop a dataset from the catalog.
    fn drop_dataset(&self, identifier: &DatasetIdentifier) -> Result<(), String>;

    /// Drop a dataset from the catalog and purge the metadata.
    fn drop_dataset_with_purge(
        &self,
        identifier: &DatasetIdentifier,
        purge: &bool,
    ) -> Result<(), String>;

    /// Rename a dataset in the catalog.
    fn rename_dataset(
        &self,
        from: &DatasetIdentifier,
        to: &DatasetIdentifier,
    ) -> Result<(), String>;

    /// Load a dataset from the catalog.
    fn load_dataset(&self, name: &DatasetIdentifier) -> Result<Dataset, String>;

    /// Invalidate cached table metadata from current catalog.
    fn invalidate_dataset(&self, identifier: &DatasetIdentifier) -> Result<(), String>;

    /// Register a dataset in the catalog.
    fn register_dataset(&self, identifier: &DatasetIdentifier) -> Result<Dataset, String>;

    /// Create a database in the catalog.
    fn create_database(
        &self,
        database: &Database,
        metadata: HashMap<String, String>,
    ) -> Result<(), String>;

    /// List top-level databases from the catalog.
    fn list_databases(&self) -> Vec<Database> {
        self.list_child_databases(&Database::empty())
            .unwrap_or_default()
    }

    /// List child databases from the database.
    fn list_child_databases(&self, database: &Database) -> Result<Vec<Database>, String>;

    /// Load metadata properties for a database.
    fn load_database_metadata(
        &self,
        database: &Database,
    ) -> Result<HashMap<String, String>, String>;

    /// Drop a database.
    fn drop_database(&self, database: &Database) -> Result<bool, String>;

    /// Set a collection of properties on a database in the catalog.
    fn set_properties(
        &self,
        database: &Database,
        properties: HashMap<String, String>,
    ) -> Result<bool, String>;

    /// Remove a set of property keys from a database in the catalog.
    fn remove_properties(
        &self,
        database: &Database,
        properties: HashSet<String>,
    ) -> Result<bool, String>;

    /// Checks whether the database exists.
    fn database_exists(&self, database: &Database) -> bool {
        self.load_database_metadata(database).is_ok()
    }
}
