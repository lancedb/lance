// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::catalog::database::Database;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct DatasetIdentifier {
    database: Database,
    name: String,
}

impl DatasetIdentifier {
    pub fn of(names: &[&str]) -> Self {
        assert!(
            !names.is_empty(),
            "Cannot create dataset identifier without a dataset name"
        );
        let database = Database::of(&names[..names.len() - 1]);
        let name = names[names.len() - 1].to_string();
        Self {
            database: database,
            name,
        }
    }

    pub fn of_database(database: Database, name: &str) -> Self {
        assert!(!name.is_empty(), "Invalid dataset name: null or empty");
        Self {
            database: database,
            name: name.to_string(),
        }
    }

    pub fn parse(identifier: &str) -> Self {
        let parts: Vec<&str> = identifier.split('.').collect();
        Self::of(&parts)
    }

    pub fn has_database(&self) -> bool {
        !self.database.is_empty()
    }

    pub fn database(&self) -> &Database {
        &self.database
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn to_lowercase(&self) -> Self {
        let new_levels: Vec<String> = self
            .database
            .levels()
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let new_name = self.name.to_lowercase();
        Self::of_database(
            Database::of(&new_levels.iter().map(String::as_str).collect::<Vec<&str>>()),
            &new_name,
        )
    }
}

impl PartialEq for DatasetIdentifier {
    fn eq(&self, other: &Self) -> bool {
        self.database == other.database && self.name == other.name
    }
}

impl Eq for DatasetIdentifier {}

impl Hash for DatasetIdentifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.database.hash(state);
        self.name.hash(state);
    }
}

impl fmt::Display for DatasetIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.has_database() {
            write!(f, "{}.{}", self.database, self.name)
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
        let ds_id = DatasetIdentifier::of(&["database1", "database2", "dataset"]);
        assert_eq!(
            ds_id.database().levels(),
            &vec!["database1".to_string(), "database2".to_string()]
        );
        assert_eq!(ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_of_database() {
        let database = Database::of(&["database1", "database2"]);
        let ds_id = DatasetIdentifier::of_database(database.clone(), "dataset");
        assert_eq!(ds_id.database(), &database);
        assert_eq!(ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_parse() {
        let ds_id = DatasetIdentifier::parse("database1.database2.dataset");
        assert_eq!(
            ds_id.database().levels(),
            &vec!["database1".to_string(), "database2".to_string()]
        );
        assert_eq!(ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_has_database() {
        let ds_id = DatasetIdentifier::parse("database1.database2.dataset");
        assert!(ds_id.has_database());

        let ds_id_no_ns = DatasetIdentifier::of(&["dataset"]);
        assert!(!ds_id_no_ns.has_database());
    }

    #[test]
    fn test_dataset_identifier_to_lowercase() {
        let ds_id = DatasetIdentifier::parse("Database1.Database2.Dataset");
        let lower_ds_id = ds_id.to_lowercase();
        assert_eq!(
            lower_ds_id.database().levels(),
            &vec!["database1".to_string(), "database2".to_string()]
        );
        assert_eq!(lower_ds_id.name(), "dataset");
    }

    #[test]
    fn test_dataset_identifier_equality() {
        let ds_id1 = DatasetIdentifier::parse("database1.database2.dataset");
        let ds_id2 = DatasetIdentifier::parse("database1.database2.dataset");
        let ds_id3 = DatasetIdentifier::parse("database1.database2.other_dataset");
        assert_eq!(ds_id1, ds_id2);
        assert_ne!(ds_id1, ds_id3);
    }

    #[test]
    fn test_dataset_identifier_hash() {
        let ds_id1 = DatasetIdentifier::parse("database1.database2.dataset");
        let ds_id2 = DatasetIdentifier::parse("database1.database2.dataset");
        let mut hasher1 = DefaultHasher::new();
        ds_id1.hash(&mut hasher1);
        let mut hasher2 = DefaultHasher::new();
        ds_id2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_dataset_identifier_display() {
        let ds_id = DatasetIdentifier::parse("database1.database2.dataset");
        assert_eq!(format!("{}", ds_id), "database1.database2.dataset");

        let ds_id_no_ns = DatasetIdentifier::of(&["dataset"]);
        assert_eq!(format!("{}", ds_id_no_ns), "dataset");
    }
}
