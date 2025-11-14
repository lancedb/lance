// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conflict detection interface and implementation
//!
//! This module provides the core conflict detection logic for merge insert operations.
//! It handles the intersection-based conflict detection algorithm using Bloom Filters.

use std::sync::Arc;

use lance_core::Result;

use super::primary_key_filter::PrimaryKeyBloomFilter;

/// Result of conflict detection
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictDetectionResult {
    /// No conflict detected - operations can proceed
    NoConflict,
    /// Potential conflict detected - operation should be retried
    /// Contains information about the conflicting transaction
    Conflict {
        /// UUID of the conflicting transaction
        conflicting_transaction_uuid: String,
        /// Version of the conflicting transaction
        conflicting_version: u64,
        /// Whether this might be a false positive (only relevant for Bloom Filters)
        might_be_false_positive: bool,
    },
}

impl ConflictDetectionResult {
    /// Check if there is a conflict
    pub fn has_conflict(&self) -> bool {
        matches!(self, Self::Conflict { .. })
    }

    /// Get the conflicting transaction UUID if there is a conflict
    pub fn conflicting_uuid(&self) -> Option<&str> {
        match self {
            Self::Conflict {
                conflicting_transaction_uuid,
                ..
            } => Some(conflicting_transaction_uuid),
            Self::NoConflict => None,
        }
    }

    /// Check if the conflict might be a false positive
    pub fn might_be_false_positive(&self) -> bool {
        match self {
            Self::Conflict {
                might_be_false_positive,
                ..
            } => *might_be_false_positive,
            Self::NoConflict => false,
        }
    }
}

/// Transaction information for conflict detection
#[derive(Debug, Clone)]
pub struct TransactionInfo {
    /// Transaction UUID
    pub uuid: String,
    /// Transaction version
    pub version: u64,
    /// Primary key Bloom Filter for this transaction
    pub primary_key_filter: Option<Arc<PrimaryKeyBloomFilter>>,
}

/// Conflict detector interface
pub trait ConflictDetector {
    /// Check for conflicts between the current transaction and other concurrent transactions
    fn detect_conflicts(
        &self,
        current_filter: &PrimaryKeyBloomFilter,
        concurrent_transactions: &[TransactionInfo],
    ) -> Result<Vec<ConflictDetectionResult>>;

    /// Check for conflict between two specific filters
    fn check_filter_conflict(
        &self,
        filter1: &PrimaryKeyBloomFilter,
        filter2: &PrimaryKeyBloomFilter,
        transaction_uuid: &str,
        transaction_version: u64,
    ) -> Result<ConflictDetectionResult>;
}

/// Default implementation of conflict detector
#[derive(Debug, Default)]
pub struct DefaultConflictDetector {
    /// Whether to be conservative about Bloom Filter conflicts
    /// If true, any non-empty Bloom Filter intersection is considered a conflict
    conservative_mode: bool,
}

impl DefaultConflictDetector {
    /// Create a new default conflict detector
    pub fn new() -> Self {
        Self {
            conservative_mode: true,
        }
    }

    /// Create a conflict detector with specified conservative mode
    pub fn with_conservative_mode(conservative_mode: bool) -> Self {
        Self { conservative_mode }
    }

    /// Set conservative mode
    pub fn set_conservative_mode(&mut self, conservative: bool) {
        self.conservative_mode = conservative;
    }
}

impl ConflictDetector for DefaultConflictDetector {
    fn detect_conflicts(
        &self,
        current_filter: &PrimaryKeyBloomFilter,
        concurrent_transactions: &[TransactionInfo],
    ) -> Result<Vec<ConflictDetectionResult>> {
        let mut conflicts = Vec::new();

        for transaction in concurrent_transactions {
            if let Some(ref other_filter) = transaction.primary_key_filter {
                let result = self.check_filter_conflict(
                    current_filter,
                    other_filter,
                    &transaction.uuid,
                    transaction.version,
                )?;

                if result.has_conflict() {
                    conflicts.push(result);
                }
            }
        }

        Ok(conflicts)
    }

    fn check_filter_conflict(
        &self,
        filter1: &PrimaryKeyBloomFilter,
        filter2: &PrimaryKeyBloomFilter,
        transaction_uuid: &str,
        transaction_version: u64,
    ) -> Result<ConflictDetectionResult> {
        // Skip conflict detection if either filter is empty
        if filter1.is_empty() || filter2.is_empty() {
            return Ok(ConflictDetectionResult::NoConflict);
        }

        // Check for intersection
        let has_intersection = filter1.has_intersection(filter2);

        if has_intersection {
            // Determine if this might be a false positive
            let might_be_false_positive = self.is_potentially_false_positive(filter1, filter2);

            Ok(ConflictDetectionResult::Conflict {
                conflicting_transaction_uuid: transaction_uuid.to_string(),
                conflicting_version: transaction_version,
                might_be_false_positive,
            })
        } else {
            Ok(ConflictDetectionResult::NoConflict)
        }
    }
}

impl DefaultConflictDetector {
    /// Determine if a conflict might be a false positive
    /// This is relevant when Bloom Filters are involved
    fn is_potentially_false_positive(
        &self,
        filter1: &PrimaryKeyBloomFilter,
        filter2: &PrimaryKeyBloomFilter,
    ) -> bool {
        // For the simplified implementation, check if either filter might have false positives
        filter1.might_have_false_positives() || filter2.might_have_false_positives()
    }
}

/// Utility functions for conflict detection
pub mod utils {
    use super::*;

    /// Create a conflict detection result for a specific transaction
    pub fn create_conflict_result(
        transaction_uuid: String,
        transaction_version: u64,
        might_be_false_positive: bool,
    ) -> ConflictDetectionResult {
        ConflictDetectionResult::Conflict {
            conflicting_transaction_uuid: transaction_uuid,
            conflicting_version: transaction_version,
            might_be_false_positive,
        }
    }

    /// Check if any of the conflict results indicate a definite conflict
    /// (not a potential false positive)
    pub fn has_definite_conflict(results: &[ConflictDetectionResult]) -> bool {
        results
            .iter()
            .any(|result| result.has_conflict() && !result.might_be_false_positive())
    }

    /// Filter out potential false positives from conflict results
    pub fn filter_false_positives(
        results: Vec<ConflictDetectionResult>,
    ) -> Vec<ConflictDetectionResult> {
        // Only retain definite conflicts (exclude NoConflict and false positives)
        results
            .into_iter()
            .filter(|result| result.has_conflict() && !result.might_be_false_positive())
            .collect()
    }

    /// Get the first definite conflict (non-false-positive)
    pub fn first_definite_conflict(
        results: &[ConflictDetectionResult],
    ) -> Option<&ConflictDetectionResult> {
        results
            .iter()
            .find(|result| result.has_conflict() && !result.might_be_false_positive())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::conflict_detection::primary_key_filter::{
        PrimaryKeyBloomFilter, PrimaryKeyValue,
    };

    #[test]
    fn test_no_conflict_empty_filters() {
        let detector = DefaultConflictDetector::new();
        let filter1 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let filter2 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);

        let result = detector
            .check_filter_conflict(&filter1, &filter2, "uuid1", 1)
            .unwrap();

        assert_eq!(result, ConflictDetectionResult::NoConflict);
    }

    #[test]
    fn test_conflict_detection_with_intersection() {
        let detector = DefaultConflictDetector::new();
        let mut filter1 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let mut filter2 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);

        let shared_key = PrimaryKeyValue::String("shared_key".to_string());
        filter1.insert(shared_key.clone()).unwrap();
        filter2.insert(shared_key).unwrap();

        let result = detector
            .check_filter_conflict(&filter1, &filter2, "uuid2", 2)
            .unwrap();

        assert!(result.has_conflict());
        assert_eq!(result.conflicting_uuid(), Some("uuid2"));
    }

    #[test]
    fn test_multiple_transaction_conflict_detection() {
        let detector = DefaultConflictDetector::new();
        let mut current_filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let shared_key = PrimaryKeyValue::String("shared_key".to_string());
        current_filter.insert(shared_key.clone()).unwrap();

        // Create concurrent transactions
        let mut filter1 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        filter1.insert(shared_key).unwrap();

        let mut filter2 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        filter2
            .insert(PrimaryKeyValue::String("other_key".to_string()))
            .unwrap();

        let transactions = vec![
            TransactionInfo {
                uuid: "tx1".to_string(),
                version: 1,
                primary_key_filter: Some(Arc::new(filter1)),
            },
            TransactionInfo {
                uuid: "tx2".to_string(),
                version: 2,
                primary_key_filter: Some(Arc::new(filter2)),
            },
        ];

        let results = detector
            .detect_conflicts(&current_filter, &transactions)
            .unwrap();

        // Should detect conflict with tx1 but not tx2
        assert_eq!(results.len(), 1);
        assert!(results[0].has_conflict());
        assert_eq!(results[0].conflicting_uuid(), Some("tx1"));
    }

    #[test]
    fn test_conflict_detection_result_methods() {
        let conflict = ConflictDetectionResult::Conflict {
            conflicting_transaction_uuid: "test_uuid".to_string(),
            conflicting_version: 42,
            might_be_false_positive: true,
        };

        assert!(conflict.has_conflict());
        assert_eq!(conflict.conflicting_uuid(), Some("test_uuid"));
        assert!(conflict.might_be_false_positive());

        let no_conflict = ConflictDetectionResult::NoConflict;
        assert!(!no_conflict.has_conflict());
        assert_eq!(no_conflict.conflicting_uuid(), None);
        assert!(!no_conflict.might_be_false_positive());
    }

    #[test]
    fn test_utils_functions() {
        let results = vec![
            ConflictDetectionResult::Conflict {
                conflicting_transaction_uuid: "tx1".to_string(),
                conflicting_version: 1,
                might_be_false_positive: true,
            },
            ConflictDetectionResult::Conflict {
                conflicting_transaction_uuid: "tx2".to_string(),
                conflicting_version: 2,
                might_be_false_positive: false,
            },
            ConflictDetectionResult::NoConflict,
        ];

        assert!(utils::has_definite_conflict(&results));

        let definite_conflicts = utils::filter_false_positives(results.clone());
        assert_eq!(definite_conflicts.len(), 1);
        assert_eq!(definite_conflicts[0].conflicting_uuid(), Some("tx2"));

        let first_definite = utils::first_definite_conflict(&results);
        assert!(first_definite.is_some());
        assert_eq!(first_definite.unwrap().conflicting_uuid(), Some("tx2"));
    }
}
