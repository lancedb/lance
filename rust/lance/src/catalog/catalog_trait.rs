// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::catalog::dataset_identifier::DatasetIdentifier;
use crate::catalog::namespace::Namespace;
use crate::dataset::Dataset;
use std::collections::HashMap;

pub trait Catalog {
    /// List all datasets under a specified namespace.
    fn list_datasets(&self, namespace: &Namespace) -> Vec<DatasetIdentifier>;

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

    /// Initialize the catalog.
    fn initialize(&self, name: &str, properties: &HashMap<&str, &str>) -> Result<(), String>;
}
