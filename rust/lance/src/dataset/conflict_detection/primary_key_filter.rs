// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Simplified Primary Key Bloom Filter implementation for conflict detection
//!
//! This is a simplified version that focuses on the core functionality needed
//! for the merge insert operation type correction.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};
use snafu::location;

/// Primary key value that can be used in conflict detection
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimaryKeyValue {
    String(String),
    Int64(i64),
    UInt64(u64),
    Binary(Vec<u8>),
    Composite(Vec<PrimaryKeyValue>),
}

impl PrimaryKeyValue {
    /// Convert the primary key value to bytes for hashing
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::String(s) => s.as_bytes().to_vec(),
            Self::Int64(i) => i.to_le_bytes().to_vec(),
            Self::UInt64(u) => u.to_le_bytes().to_vec(),
            Self::Binary(b) => b.clone(),
            Self::Composite(values) => {
                let mut result = Vec::new();
                for value in values {
                    result.extend_from_slice(&value.to_bytes());
                    result.push(0); // separator
                }
                result
            }
        }
    }

    /// Get a hash of the primary key value
    pub fn hash_value(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.to_bytes().hash(&mut hasher);
        hasher.finish()
    }
}

/// Simplified Primary Key Bloom Filter that uses exact storage for now
/// This is a temporary implementation to fix compilation issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryKeyBloomFilter {
    /// Exact mapping of primary key hashes for conflict detection
    key_hashes: HashMap<u64, bool>,
    /// Column names that form the primary key
    primary_key_columns: Vec<String>,
}

impl PrimaryKeyBloomFilter {
    /// Create a new Primary Key Bloom Filter
    pub fn new(primary_key_columns: Vec<String>) -> Self {
        Self {
            key_hashes: HashMap::new(),
            primary_key_columns,
        }
    }

    /// Add a primary key to the filter
    pub fn insert(&mut self, key: PrimaryKeyValue) -> Result<()> {
        let hash = key.hash_value();
        self.key_hashes.insert(hash, true);
        Ok(())
    }

    /// Check if a primary key might be present
    pub fn contains(&self, key: &PrimaryKeyValue) -> bool {
        let hash = key.hash_value();
        self.key_hashes.contains_key(&hash)
    }

    /// Check for intersection with another filter
    pub fn has_intersection(&self, other: Self) -> bool {
        // Check if any keys overlap
        for key in self.key_hashes.keys() {
            if other.key_hashes.contains_key(key) {
                return true;
            }
        }
        false
    }

    /// Get the primary key columns
    pub fn primary_key_columns(&self) -> &[String] {
        &self.primary_key_columns
    }

    /// Get the estimated size in bytes
    pub fn estimated_size_bytes(&self) -> usize {
        // Rough estimate: 8 bytes per key + 1 byte per value + HashMap overhead
        self.key_hashes.len() * 9 + 64
    }

    /// Get the number of items
    pub fn len(&self) -> usize {
        self.key_hashes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.key_hashes.is_empty()
    }

    /// Serialize to bytes for storage in transaction files
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::InvalidInput {
            source: format!("Failed to serialize PrimaryKeyBloomFilter: {}", e).into(),
            location: location!(),
        })
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| Error::InvalidInput {
            source: format!("Failed to deserialize PrimaryKeyBloomFilter: {}", e).into(),
            location: location!(),
        })
    }

    /// Check if this filter might produce false positives
    /// For the simplified version, this is always false since we use exact storage
    pub fn might_have_false_positives(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::conflict_detection::{
        conflict_detector::{ConflictDetector, DefaultConflictDetector},
        primary_key_filter::{PrimaryKeyBloomFilter, PrimaryKeyValue},
    };
    use crate::dataset::transaction::{Operation, Transaction, TransactionBuilder};

    #[test]
    fn test_primary_key_value_hash() {
        let key1 = PrimaryKeyValue::String("test".to_string());
        let key2 = PrimaryKeyValue::String("test".to_string());
        let key3 = PrimaryKeyValue::String("different".to_string());

        assert_eq!(key1.hash_value(), key2.hash_value());
        assert_ne!(key1.hash_value(), key3.hash_value());
    }

    #[test]
    fn test_filter_operations() {
        let mut filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let key = PrimaryKeyValue::String("test_key".to_string());

        // Insert and check
        filter.insert(key.clone()).unwrap();
        assert!(filter.contains(&key));

        // Check non-existent key
        let other_key = PrimaryKeyValue::String("other_key".to_string());
        assert!(!filter.contains(&other_key));
    }

    #[test]
    fn test_intersection_detection() {
        let mut filter1 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let mut filter2 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);

        let key1 = PrimaryKeyValue::String("shared_key".to_string());
        let key2 = PrimaryKeyValue::String("unique_key1".to_string());
        let key3 = PrimaryKeyValue::String("unique_key2".to_string());

        // Add shared key to both filters
        filter1.insert(key1.clone()).unwrap();
        filter1.insert(key2).unwrap();

        filter2.insert(key1).unwrap();
        filter2.insert(key3).unwrap();

        // Should detect intersection
        assert!(filter1.has_intersection(filter2));
    }

    #[test]
    fn test_serialization() {
        let mut filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let key = PrimaryKeyValue::String("test_key".to_string());
        filter.insert(key.clone()).unwrap();

        // Serialize and deserialize
        let bytes = filter.to_bytes().unwrap();
        let deserialized = PrimaryKeyBloomFilter::from_bytes(&bytes).unwrap();

        assert!(deserialized.contains(&key));
        assert_eq!(
            deserialized.primary_key_columns(),
            filter.primary_key_columns()
        );
    }

    #[test]
    fn test_bloom_filter_creation_and_basic_operations() {
        let mut bloom_filter = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);

        let key1 = PrimaryKeyValue::String("alice".to_string());
        let key2 = PrimaryKeyValue::String("bob".to_string());
        let key3 = PrimaryKeyValue::String("charlie".to_string());

        bloom_filter.insert(key1.clone()).unwrap();
        bloom_filter.insert(key2.clone()).unwrap();

        assert!(bloom_filter.contains(&key1));
        assert!(bloom_filter.contains(&key2));
        assert!(!bloom_filter.contains(&key3));

        assert_eq!(bloom_filter.len(), 2);
        assert!(!bloom_filter.is_empty());
    }

    #[test]
    fn test_composite_primary_key_handling() {
        let mut bloom_filter =
            PrimaryKeyBloomFilter::new(vec!["tenant_id".to_string(), "user_id".to_string()]);

        let composite_key1 = PrimaryKeyValue::Composite(vec![
            PrimaryKeyValue::String("tenant_a".to_string()),
            PrimaryKeyValue::String("user_001".to_string()),
        ]);

        let composite_key2 = PrimaryKeyValue::Composite(vec![
            PrimaryKeyValue::String("tenant_a".to_string()),
            PrimaryKeyValue::String("user_002".to_string()),
        ]);

        let composite_key3 = PrimaryKeyValue::Composite(vec![
            PrimaryKeyValue::String("tenant_b".to_string()),
            PrimaryKeyValue::String("user_001".to_string()),
        ]);

        bloom_filter.insert(composite_key1.clone()).unwrap();
        bloom_filter.insert(composite_key2.clone()).unwrap();

        assert!(bloom_filter.contains(&composite_key1));
        assert!(bloom_filter.contains(&composite_key2));
        assert!(!bloom_filter.contains(&composite_key3));
    }

    #[test]
    fn test_bloom_filter_serialization_deserialization() {
        let mut original_filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);

        let test_keys = vec![
            PrimaryKeyValue::String("test_1".to_string()),
            PrimaryKeyValue::String("test_2".to_string()),
            PrimaryKeyValue::Int64(12345),
            PrimaryKeyValue::Int64(67890),
        ];

        for key in test_keys.iter() {
            original_filter.insert(key.clone()).unwrap();
        }

        let serialized_data = original_filter.to_bytes().unwrap();
        assert!(!serialized_data.is_empty());

        let deserialized_filter = PrimaryKeyBloomFilter::from_bytes(&serialized_data).unwrap();

        assert_eq!(original_filter.len(), deserialized_filter.len());

        for key in &test_keys {
            assert_eq!(
                original_filter.contains(key),
                deserialized_filter.contains(key),
                "ser/deser not equal: {:?}",
                key
            );
        }
    }

    #[test]
    fn test_conflict_detection_with_overlapping_keys() {
        let detector = DefaultConflictDetector::new();

        let mut filter1 = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);
        let mut filter2 = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);

        let keys1 = [
            PrimaryKeyValue::String("alice".to_string()),
            PrimaryKeyValue::String("bob".to_string()),
            PrimaryKeyValue::String("charlie".to_string()),
        ];

        for key in keys1.iter() {
            filter1.insert(key.clone()).unwrap();
        }

        let keys2 = [
            PrimaryKeyValue::String("charlie".to_string()), // duplicated!
            PrimaryKeyValue::String("david".to_string()),
            PrimaryKeyValue::String("eve".to_string()),
        ];

        for key in keys2.iter() {
            filter2.insert(key.clone()).unwrap();
        }

        let conflict_result = detector
            .check_filter_conflict(&filter1, &filter2, "test_transaction_uuid", 2)
            .unwrap();

        assert!(conflict_result.has_conflict(), "should detect conflict");
        assert_eq!(
            conflict_result.conflicting_uuid(),
            Some("test_transaction_uuid")
        );
    }

    #[test]
    fn test_conflict_detection_with_no_overlap() {
        let detector = DefaultConflictDetector::new();

        let mut filter1 = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);
        let mut filter2 = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);

        let keys1 = [
            PrimaryKeyValue::String("alice".to_string()),
            PrimaryKeyValue::String("bob".to_string()),
            PrimaryKeyValue::String("charlie".to_string()),
        ];

        for key in keys1.iter() {
            filter1.insert(key.clone()).unwrap();
        }

        let keys2 = [
            PrimaryKeyValue::String("david".to_string()),
            PrimaryKeyValue::String("eve".to_string()),
            PrimaryKeyValue::String("frank".to_string()),
        ];

        for key in keys2.iter() {
            filter2.insert(key.clone()).unwrap();
        }

        let conflict_result = detector
            .check_filter_conflict(&filter1, &filter2, "test_transaction_uuid", 2)
            .unwrap();

        assert!(
            !conflict_result.has_conflict(),
            "should not detect conflict"
        );
    }

    #[test]
    fn test_transaction_with_bloom_filter() {
        let mut bloom_filter = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);

        let keys = [
            PrimaryKeyValue::String("transaction_user_1".to_string()),
            PrimaryKeyValue::String("transaction_user_2".to_string()),
        ];

        for key in keys.iter() {
            bloom_filter.insert(key.clone()).unwrap();
        }

        let filter_data = bloom_filter.to_bytes().unwrap();

        let transaction = TransactionBuilder::new(
            1, // read_version
            Operation::Merge {
                fragments: Vec::new(),
                schema: lance_core::datatypes::Schema::default(),
            },
        )
        .primary_key_bloom_filter(Some(filter_data.clone()))
        .build();

        assert!(transaction.primary_key_bloom_filter.is_some());
        assert_eq!(transaction.primary_key_bloom_filter.unwrap(), filter_data);
    }

    #[test]
    fn test_transaction_protobuf_serialization_with_bloom_filter() {
        use lance_table::format::pb;

        let mut bloom_filter = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);
        bloom_filter
            .insert(PrimaryKeyValue::String("test_user".to_string()))
            .unwrap();
        let filter_data = bloom_filter.to_bytes().unwrap();

        let original_transaction = TransactionBuilder::new(
            1,
            Operation::Merge {
                fragments: Vec::new(),
                schema: lance_core::datatypes::Schema::default(),
            },
        )
        .primary_key_bloom_filter(Some(filter_data.clone()))
        .build();

        let pb_transaction: pb::Transaction = (&original_transaction).into();
        assert!(pb_transaction.primary_key_bloom_filter.is_some());
        assert_eq!(
            pb_transaction.primary_key_bloom_filter.as_ref().unwrap(),
            &filter_data
        );

        let deserialized_transaction: Transaction = pb_transaction.try_into().unwrap();
        assert!(deserialized_transaction.primary_key_bloom_filter.is_some());
        assert_eq!(
            deserialized_transaction.primary_key_bloom_filter.unwrap(),
            filter_data
        );
    }

    #[test]
    fn test_threshold_based_storage_strategy() {
        // Test that a small dataset uses the exact map.
        let mut small_filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);

        // Insert a small amount of data.
        for i in 0..5 {
            let key = PrimaryKeyValue::String(format!("small_{}", i));
            small_filter.insert(key).unwrap();
        }

        // A small dataset should use a small amount of memory.
        let small_size = small_filter.estimated_size_bytes();
        assert!(small_size < 1024, "Small dataset should use less memory");

        // Test a large dataset.
        let mut large_filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);

        // Insert a large amount of data.
        for i in 0..1000 {
            let key = PrimaryKeyValue::String(format!("large_{:04}", i));
            large_filter.insert(key).unwrap();
        }

        let large_size = large_filter.estimated_size_bytes();

        // Validate storage efficiency for a large dataset.
        // The current implementation uses a HashMap, so its size is predictable.
        let estimated_exact_size = 1000 * 9 + 64; // Based on current implementation
        assert_eq!(
            large_size, estimated_exact_size,
            "Large filter size should match the exact map estimation"
        );

        // This check is for a future state where a real bloom filter is used.
        // For now, it will use the exact map strategy.
        assert!(
            large_size < 200 * 1024,
            "The filter should not exceed the threshold for switching to a bloom filter yet"
        );
    }

    #[test]
    fn test_end_to_end_conflict_detection_workflow() {
        // Simulate the end-to-end conflict detection workflow.

        // Step 1: Create the Bloom Filter for the first transaction.
        let mut transaction1_filter = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);
        let transaction1_keys = [
            PrimaryKeyValue::String("workflow_user_1".to_string()),
            PrimaryKeyValue::String("workflow_user_2".to_string()),
            PrimaryKeyValue::String("workflow_user_3".to_string()),
        ];

        for key in transaction1_keys.iter() {
            transaction1_filter.insert(key.clone()).unwrap();
        }

        let transaction1_data = transaction1_filter.to_bytes().unwrap();

        // Step 2: Create the first transaction.
        let transaction1 = TransactionBuilder::new(
            1,
            Operation::Merge {
                fragments: Vec::new(),
                schema: lance_core::datatypes::Schema::default(),
            },
        )
        .primary_key_bloom_filter(Some(transaction1_data))
        .build();

        // Step 3: Create the Bloom Filter for the second transaction (with a conflict).
        let mut transaction2_filter = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);
        let transaction2_keys = [
            PrimaryKeyValue::String("workflow_user_3".to_string()), // Conflict!
            PrimaryKeyValue::String("workflow_user_4".to_string()),
            PrimaryKeyValue::String("workflow_user_5".to_string()),
        ];

        for key in transaction2_keys.iter() {
            transaction2_filter.insert(key.clone()).unwrap();
        }

        let transaction2_data = transaction2_filter.to_bytes().unwrap();

        // Step 4: Create the second transaction.
        let transaction2 = TransactionBuilder::new(
            1,
            Operation::Merge {
                fragments: Vec::new(),
                schema: lance_core::datatypes::Schema::default(),
            },
        )
        .primary_key_bloom_filter(Some(transaction2_data))
        .build();

        // Step 5: Perform conflict detection.
        if let (Some(filter1_data), Some(filter2_data)) = (
            &transaction1.primary_key_bloom_filter,
            &transaction2.primary_key_bloom_filter,
        ) {
            let filter1 = PrimaryKeyBloomFilter::from_bytes(filter1_data).unwrap();
            let filter2 = PrimaryKeyBloomFilter::from_bytes(filter2_data).unwrap();

            let detector = DefaultConflictDetector::new();
            let conflict_result = detector
                .check_filter_conflict(&filter1, &filter2, &transaction2.uuid, 2)
                .unwrap();

            // Step 6: Validate the conflict detection result.
            assert!(
                conflict_result.has_conflict(),
                "A conflict should be detected"
            );
            assert_eq!(
                conflict_result.conflicting_uuid(),
                Some(&transaction2.uuid).map(|x| x.as_str())
            );
            // Since the current implementation uses an exact map, there are no false positives.
            assert!(!conflict_result.might_be_false_positive());
        } else {
            panic!("Transactions should contain Bloom Filter data");
        }

        // Step 7: Test the no-conflict case.
        let mut transaction3_filter = PrimaryKeyBloomFilter::new(vec!["user_id".to_string()]);
        let transaction3_keys = [
            PrimaryKeyValue::String("workflow_user_6".to_string()),
            PrimaryKeyValue::String("workflow_user_7".to_string()),
            PrimaryKeyValue::String("workflow_user_8".to_string()),
        ];

        for key in transaction3_keys.iter() {
            transaction3_filter.insert(key.clone()).unwrap();
        }

        let transaction3_data = transaction3_filter.to_bytes().unwrap();
        let transaction3 = TransactionBuilder::new(
            1,
            Operation::Merge {
                fragments: Vec::new(),
                schema: lance_core::datatypes::Schema::default(),
            },
        )
        .primary_key_bloom_filter(Some(transaction3_data))
        .build();

        // Check for conflicts between transaction 1 and 3.
        if let (Some(filter1_data), Some(filter3_data)) = (
            &transaction1.primary_key_bloom_filter,
            &transaction3.primary_key_bloom_filter,
        ) {
            let filter1 = PrimaryKeyBloomFilter::from_bytes(filter1_data).unwrap();
            let filter3 = PrimaryKeyBloomFilter::from_bytes(filter3_data).unwrap();

            let detector = DefaultConflictDetector::new();
            let no_conflict_result = detector
                .check_filter_conflict(&filter1, &filter3, &transaction3.uuid, 3)
                .unwrap();

            assert!(
                !no_conflict_result.has_conflict(),
                "No conflict should be detected"
            );
        }
    }

    #[test]
    fn test_false_positive_handling() {
        // Test handling of Bloom Filter false positives.
        let detector = DefaultConflictDetector::new();

        // Create two filters. The current implementation uses an exact map,
        // so real false positives are not possible. This test structure is for
        // a future state with a probabilistic Bloom Filter.
        let mut filter1 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let mut filter2 = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);

        // Insert one set of keys into filter1.
        for i in 0..1000 {
            let key = PrimaryKeyValue::String(format!("set1_{:04}", i));
            filter1.insert(key).unwrap();
        }

        // Insert a completely different set of keys into filter2.
        for i in 0..1000 {
            let key = PrimaryKeyValue::String(format!("set2_{:04}", i));
            filter2.insert(key).unwrap();
        }

        // Check for conflicts.
        let conflict_result = detector
            .check_filter_conflict(&filter1, &filter2, "false_positive_test", 2)
            .unwrap();

        // With the current exact map implementation, no conflict should be found.
        assert!(
            !conflict_result.has_conflict(),
            "No conflict should be detected with disjoint key sets"
        );

        // This part of the test validates behavior for a future bloom filter.
        // If a conflict were detected, it must be flagged as a potential false positive.
        if conflict_result.has_conflict() {
            assert!(conflict_result.might_be_false_positive());
        }
    }
}
