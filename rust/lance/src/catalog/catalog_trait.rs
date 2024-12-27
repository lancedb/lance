// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::catalog::dataset_identifier::DatasetIdentifier;
use crate::catalog::namespace::Namespace;
use crate::dataset::Dataset;
use std::collections::{HashMap, HashSet};

pub trait Catalog {
    /// Initialize the catalog.
    fn initialize(&self, name: &str, properties: &HashMap<&str, &str>) -> Result<(), String>;

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

    /// Create a namespace in the catalog.
    fn create_namespace(
        &self,
        namespace: &Namespace,
        metadata: HashMap<String, String>,
    ) -> Result<(), String>;

    /// List top-level namespaces from the catalog.
    fn list_namespaces(&self) -> Vec<Namespace> {
        self.list_child_namespaces(&Namespace::empty())
            .unwrap_or_default()
    }

    /// List child namespaces from the namespace.
    fn list_child_namespaces(&self, namespace: &Namespace) -> Result<Vec<Namespace>, String>;

    /// Load metadata properties for a namespace.
    fn load_namespace_metadata(
        &self,
        namespace: &Namespace,
    ) -> Result<HashMap<String, String>, String>;

    /// Drop a namespace.
    fn drop_namespace(&self, namespace: &Namespace) -> Result<bool, String>;

    /// Set a collection of properties on a namespace in the catalog.
    fn set_properties(
        &self,
        namespace: &Namespace,
        properties: HashMap<String, String>,
    ) -> Result<bool, String>;

    /// Remove a set of property keys from a namespace in the catalog.
    fn remove_properties(
        &self,
        namespace: &Namespace,
        properties: HashSet<String>,
    ) -> Result<bool, String>;

    /// Checks whether the Namespace exists.
    fn namespace_exists(&self, namespace: &Namespace) -> bool {
        self.load_namespace_metadata(namespace).is_ok()
    }
}
