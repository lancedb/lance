// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::catalog::namespace::Namespace;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct DatasetIdentifier {
    namespace: Namespace,
    name: String,
}

impl DatasetIdentifier {
    pub fn of(names: &[&str]) -> Self {
        assert!(
            !names.is_empty(),
            "Cannot create dataset identifier without a dataset name"
        );
        let namespace = Namespace::of(&names[..names.len() - 1]);
        let name = names[names.len() - 1].to_string();
        Self { namespace, name }
    }

    pub fn of_namespace(namespace: Namespace, name: &str) -> Self {
        assert!(!name.is_empty(), "Invalid dataset name: null or empty");
        Self {
            namespace,
            name: name.to_string(),
        }
    }

    pub fn parse(identifier: &str) -> Self {
        let parts: Vec<&str> = identifier.split('.').collect();
        Self::of(&parts)
    }

    pub fn has_namespace(&self) -> bool {
        !self.namespace.is_empty()
    }

    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn to_lowercase(&self) -> Self {
        let new_levels: Vec<String> = self
            .namespace
            .levels()
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let new_name = self.name.to_lowercase();
        Self::of_namespace(
            Namespace::of(&new_levels.iter().map(String::as_str).collect::<Vec<&str>>()),
            &new_name,
        )
    }
}

impl PartialEq for DatasetIdentifier {
    fn eq(&self, other: &Self) -> bool {
        self.namespace == other.namespace && self.name == other.name
    }
}

impl Eq for DatasetIdentifier {}

impl Hash for DatasetIdentifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.name.hash(state);
    }
}

impl fmt::Display for DatasetIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.has_namespace() {
            write!(f, "{}.{}", self.namespace, self.name)
        } else {
            write!(f, "{}", self.name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::DefaultHasher;

    #[test]
    fn test_dataset_identifier_of() {
        let ds_id = DatasetIdentifier::of(&["namespace1", "namespace2", "dataset"]);
        assert_eq!(
            ds_id.namespace().levels(),
            &vec!["namespace1".to_string(), "namespace2".to_string()]
        );
        assert_eq!(ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_of_namespace() {
        let namespace = Namespace::of(&["namespace1", "namespace2"]);
        let ds_id = DatasetIdentifier::of_namespace(namespace.clone(), "dataset");
        assert_eq!(ds_id.namespace(), &namespace);
        assert_eq!(ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_parse() {
        let ds_id = DatasetIdentifier::parse("namespace1.namespace2.dataset");
        assert_eq!(
            ds_id.namespace().levels(),
            &vec!["namespace1".to_string(), "namespace2".to_string()]
        );
        assert_eq!(ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_has_namespace() {
        let ds_id = DatasetIdentifier::parse("namespace1.namespace2.dataset");
        assert!(ds_id.has_namespace());

        let ds_id_no_ns = DatasetIdentifier::of(&["dataset"]);
        assert!(!ds_id_no_ns.has_namespace());
    }

    #[test]
    fn test_dataset_identifier_to_lowercase() {
        let ds_id = DatasetIdentifier::parse("Namespace1.Namespace2.Dataset");
        let lower_ds_id = ds_id.to_lowercase();
        assert_eq!(
            lower_ds_id.namespace().levels(),
            &vec!["namespace1".to_string(), "namespace2".to_string()]
        );
        assert_eq!(lower_ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_equality() {
        let ds_id1 = DatasetIdentifier::parse("namespace1.namespace2.dataset");
        let ds_id2 = DatasetIdentifier::parse("namespace1.namespace2.dataset");
        let ds_id3 = DatasetIdentifier::parse("namespace1.namespace2.other_dataset");
        assert_eq!(ds_id1, ds_id2);
        assert_ne!(ds_id1, ds_id3);
    }

    #[test]
    fn test_dataset_identifier_hash() {
        let ds_id1 = DatasetIdentifier::parse("namespace1.namespace2.dataset");
        let ds_id2 = DatasetIdentifier::parse("namespace1.namespace2.dataset");
        let mut hasher1 = DefaultHasher::new();
        ds_id1.hash(&mut hasher1);
        let mut hasher2 = DefaultHasher::new();
        ds_id2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_dataset_identifier_display() {
        let ds_id = DatasetIdentifier::parse("namespace1.namespace2.dataset");
        assert_eq!(format!("{}", ds_id), "namespace1.namespace2.dataset");

        let ds_id_no_ns = DatasetIdentifier::of(&["dataset"]);
        assert_eq!(format!("{}", ds_id_no_ns), "dataset");
    }
}
