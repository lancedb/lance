// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct Database {
    levels: Vec<String>,
}

impl Database {
    pub fn empty() -> Self {
        Self { levels: Vec::new() }
    }

    pub fn of(levels: &[&str]) -> Self {
        assert!(
            levels.iter().all(|&level| level != "\0"),
            "Cannot create a database with the null-byte character"
        );
        Self {
            levels: levels.iter().map(|&s| s.to_string()).collect(),
        }
    }

    pub fn levels(&self) -> &[String] {
        &self.levels
    }

    pub fn level(&self, pos: usize) -> &str {
        &self.levels[pos]
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn length(&self) -> usize {
        self.levels.len()
    }
}

impl PartialEq for Database {
    fn eq(&self, other: &Self) -> bool {
        self.levels == other.levels
    }
}

impl Eq for Database {}

impl Hash for Database {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.levels.hash(state);
    }
}

impl fmt::Display for Database {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.levels.join("."))
    }
}

impl fmt::Debug for Database {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Database")
            .field("levels", &self.levels)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::DefaultHasher;

    #[test]
    fn test_empty_database() {
        let ns = Database::empty();
        assert!(ns.is_empty());
        assert_eq!(ns.length(), 0);
        assert_eq!(ns.levels().len(), 0);
    }

    #[test]
    fn test_database_of() {
        let ns = Database::of(&["level1", "level2"]);
        assert!(!ns.is_empty());
        assert_eq!(ns.length(), 2);
        assert_eq!(ns.level(0), "level1");
        assert_eq!(ns.level(1), "level2");
    }

    #[test]
    #[should_panic(expected = "Cannot create a database with the null-byte character")]
    fn test_database_of_with_null_byte() {
        Database::of(&["level1", "\0"]);
    }

    #[test]
    fn test_database_levels() {
        let ns = Database::of(&["level1", "level2"]);
        let levels = ns.levels();
        assert_eq!(levels, &vec!["level1".to_string(), "level2".to_string()]);
    }

    #[test]
    fn test_database_equality() {
        let ns1 = Database::of(&["level1", "level2"]);
        let ns2 = Database::of(&["level1", "level2"]);
        let ns3 = Database::of(&["level1", "level3"]);
        assert_eq!(ns1, ns2);
        assert_ne!(ns1, ns3);
    }

    #[test]
    fn test_database_hash() {
        let ns1 = Database::of(&["level1", "level2"]);
        let ns2 = Database::of(&["level1", "level2"]);
        let mut hasher1 = DefaultHasher::new();
        ns1.hash(&mut hasher1);
        let mut hasher2 = DefaultHasher::new();
        ns2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_database_display() {
        let ns = Database::of(&["level1", "level2"]);
        assert_eq!(format!("{}", ns), "level1.level2");
    }

    #[test]
    fn test_database_debug() {
        let ns = Database::of(&["level1", "level2"]);
        assert_eq!(
            format!("{:?}", ns),
            "Database { levels: [\"level1\", \"level2\"] }"
        );
    }
}
