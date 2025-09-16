// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Simplified Primary Key Filter (ExactSet for now) used for conflict detection
//!
//! This is a simplified version that focuses on the core functionality needed
//! for the merge insert operation type correction.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use lance_core::Result;
use lance_table::format::pb;

/// Primary key value that can be used in conflict detection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone)]
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
    pub fn has_intersection(&self, other: &Self) -> bool {
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

    /// Return an iterator over key hashes
    pub fn hashes(&self) -> impl Iterator<Item = &u64> {
        self.key_hashes.keys()
    }

    /// Convert to typed protobuf PrimaryKeyFilter (ExactSet variant)
    pub fn to_pb_filter(&self) -> pb::PrimaryKeyFilter {
        let exact = pb::ExactSet {
            key_hashes: self.key_hashes.keys().copied().collect(),
        };
        pb::PrimaryKeyFilter {
            columns: self.primary_key_columns.clone(),
            filter: Some(pb::primary_key_filter::Filter::ExactSet(exact)),
        }
    }

    /// Get the number of items
    pub fn len(&self) -> usize {
        self.key_hashes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.key_hashes.is_empty()
    }
    /// Check if this filter might produce false positives
    /// For the simplified version, this is always false since we use exact storage
    pub fn might_have_false_positives(&self) -> bool {
        false
    }
}

/// Typed PrimaryKeyFilter model used to bridge protobuf field and in-memory logic.
#[derive(Debug, Clone)]
pub enum FilterType {
    ExactSet(HashSet<u64>),
    Bloom {
        bitmap: Vec<u8>,
        num_hashes: u32,
        bitmap_bits: u32,
    },
}

#[derive(Debug, Clone)]
pub struct PrimaryKeyFilterModel {
    pub columns: Vec<String>,
    pub filter: FilterType,
}

impl PrimaryKeyFilterModel {
    pub fn from_exact_bloom(bloom: &PrimaryKeyBloomFilter) -> Self {
        let hashes: HashSet<u64> = bloom.hashes().copied().collect();
        Self {
            columns: bloom.primary_key_columns.clone(),
            filter: FilterType::ExactSet(hashes),
        }
    }

    pub fn to_pb(&self) -> pb::PrimaryKeyFilter {
        match &self.filter {
            FilterType::ExactSet(hashes) => pb::PrimaryKeyFilter {
                columns: self.columns.clone(),
                filter: Some(pb::primary_key_filter::Filter::ExactSet(pb::ExactSet {
                    key_hashes: hashes.iter().copied().collect(),
                })),
            },
            FilterType::Bloom {
                bitmap,
                num_hashes,
                bitmap_bits,
            } => pb::PrimaryKeyFilter {
                columns: self.columns.clone(),
                filter: Some(pb::primary_key_filter::Filter::Bloom(pb::BloomFilterData {
                    bitmap: bitmap.clone(),
                    num_hashes: *num_hashes,
                    bitmap_bits: *bitmap_bits,
                })),
            },
        }
    }

    pub fn from_pb(message: &pb::PrimaryKeyFilter) -> Result<Self> {
        let columns = message.columns.clone();
        let filter = match message.filter.as_ref() {
            Some(pb::primary_key_filter::Filter::ExactSet(exact)) => {
                FilterType::ExactSet(exact.key_hashes.iter().copied().collect())
            }
            Some(pb::primary_key_filter::Filter::Bloom(b)) => FilterType::Bloom {
                bitmap: b.bitmap.clone(),
                num_hashes: b.num_hashes,
                bitmap_bits: b.bitmap_bits,
            },
            None => {
                // Treat missing filter as empty exact set
                FilterType::ExactSet(HashSet::new())
            }
        };
        Ok(Self { columns, filter })
    }

    /// Determine intersection and whether it might be a false positive
    pub fn intersects(&self, other: &Self) -> (bool, bool) {
        match (&self.filter, &other.filter) {
            (FilterType::ExactSet(a), FilterType::ExactSet(b)) => {
                let has = a.iter().any(|h| b.contains(h));
                (has, false)
            }
            (
                FilterType::ExactSet(a),
                FilterType::Bloom {
                    bitmap,
                    num_hashes,
                    bitmap_bits,
                },
            ) => {
                let has = a
                    .iter()
                    .any(|h| bloom_contains_hash(*h, bitmap, *num_hashes, *bitmap_bits));
                (has, has) // potential false positives when bloom says contains
            }
            (
                FilterType::Bloom {
                    bitmap,
                    num_hashes,
                    bitmap_bits,
                },
                FilterType::ExactSet(b),
            ) => {
                let has = b
                    .iter()
                    .any(|h| bloom_contains_hash(*h, bitmap, *num_hashes, *bitmap_bits));
                (has, has)
            }
            (
                FilterType::Bloom { bitmap: a_bits, .. },
                FilterType::Bloom { bitmap: b_bits, .. },
            ) => {
                let has = bloom_bitwise_and_nonzero(a_bits, b_bits);
                (has, has)
            }
        }
    }
}

fn bloom_contains_hash(hash: u64, bitmap: &[u8], num_hashes: u32, bitmap_bits: u32) -> bool {
    if bitmap_bits == 0 || bitmap.is_empty() || num_hashes == 0 {
        return false;
    }
    let m = bitmap_bits as u64;
    let mut seed = 0x9e3779b97f4a7c15u64; // golden ratio constant
    for _i in 0..num_hashes {
        let pos = ((hash.wrapping_add(seed)) % m) as usize;
        if !bit_test(bitmap, pos) {
            return false;
        }
        seed = seed.rotate_left(13) ^ 0x517cc1b727220a95u64;
    }
    true
}

fn bit_test(bitmap: &[u8], bit_index: usize) -> bool {
    let byte_index = bit_index / 8;
    if byte_index >= bitmap.len() {
        return false;
    }
    let mask = 1u8 << (bit_index % 8);
    (bitmap[byte_index] & mask) != 0
}

fn bloom_bitwise_and_nonzero(a: &[u8], b: &[u8]) -> bool {
    let len = std::cmp::min(a.len(), b.len());
    for i in 0..len {
        if (a[i] & b[i]) != 0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use crate::dataset::conflict_detection::{
        conflict_detector::{ConflictDetector, DefaultConflictDetector},
        primary_key_filter::{PrimaryKeyBloomFilter, PrimaryKeyValue},
    };
    use crate::dataset::conflict_detection::{FilterType, PrimaryKeyFilterModel};
    use lance_table::format::pb;

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
        assert!(filter1.has_intersection(&filter2));
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
    fn test_pb_exact_set_encode_decode_and_intersection() {
        let mut filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        let k1 = PrimaryKeyValue::String("a".to_string());
        let k2 = PrimaryKeyValue::String("b".to_string());
        filter.insert(k1.clone()).unwrap();
        filter.insert(k2).unwrap();

        // to pb
        let pb_filter = filter.to_pb_filter();
        // from pb
        let model = PrimaryKeyFilterModel::from_pb(&pb_filter).unwrap();
        assert_eq!(model.columns, vec!["id".to_string()]);
        match model.filter {
            FilterType::ExactSet(ref hashes) => {
                assert_eq!(hashes.len(), 2);
            }
            _ => panic!("expected exact set"),
        }

        // intersection
        let mut other = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
        other.insert(k1).unwrap();
        let other_model = PrimaryKeyFilterModel::from_exact_bloom(&other);
        let (has, fp) = model.intersects(&other_model);
        assert!(has);
        assert!(!fp);
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
    fn test_pb_performance_baseline_sizes() {
        fn make_keys(n: usize) -> Vec<PrimaryKeyValue> {
            (0..n)
                .map(|i| PrimaryKeyValue::String(format!("k{:06}", i)))
                .collect()
        }
        for &n in &[1000usize, 10_000usize] {
            let mut filter = PrimaryKeyBloomFilter::new(vec!["id".to_string()]);
            for k in make_keys(n) {
                filter.insert(k).unwrap();
            }
            let pb = filter.to_pb_filter();
            // validate content scales
            match pb.filter {
                Some(pb::primary_key_filter::Filter::ExactSet(ex)) => {
                    assert_eq!(ex.key_hashes.len(), n);
                }
                Some(pb::primary_key_filter::Filter::Bloom(b)) => {
                    assert!(!b.bitmap.is_empty());
                }
                None => panic!("missing filter"),
            }
        }
    }
}
