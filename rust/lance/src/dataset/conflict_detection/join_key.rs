// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use deepsize::DeepSizeOf;
use lance_core::Result;
use lance_index::scalar::bloomfilter::sbbf::{Sbbf, SbbfBuilder};
use lance_table::format::pb;

const DEFAULT_NUMBER_OF_ITEMS: u64 = 8192;
const DEFAULT_PROBABILITY: f64 = 0.00057;

/// Join key value that can be used in conflict detection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JoinKeyValue {
    String(String),
    Int64(i64),
    UInt64(u64),
    Binary(Vec<u8>),
    Composite(Vec<JoinKeyValue>),
}

impl JoinKeyValue {
    /// Convert the join key value to bytes for hashing
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

    /// Get a hash of the join key value
    pub fn hash_value(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.to_bytes().hash(&mut hasher);
        hasher.finish()
    }
}

/// Simplified Join Key Bloom Filter backed by SBBF
/// Now uses a probabilistic Split Block Bloom Filter for membership tests.
#[derive(Debug, Clone)]
pub struct JoinKeyBloomFilter {
    sbbf: Sbbf,
    /// Column names that form the join key
    join_key_columns: Vec<String>,
    /// Number of items inserted (for len())
    item_count: usize,
}

impl JoinKeyBloomFilter {
    /// Create a new Join Key Bloom Filter using SBBF with default parameters.
    pub fn new(join_key_columns: Vec<String>) -> Self {
        let sbbf = SbbfBuilder::new()
            .expected_items(DEFAULT_NUMBER_OF_ITEMS)
            .false_positive_probability(DEFAULT_PROBABILITY)
            .build()
            .expect("Failed to build SBBF for JoinKeyBloomFilter");
        Self {
            sbbf,
            join_key_columns,
            item_count: 0,
        }
    }

    /// Add a join key to the filter
    pub fn insert(&mut self, key: JoinKeyValue) -> Result<()> {
        let bytes = key.to_bytes();
        self.sbbf.insert(&bytes[..]);
        self.item_count += 1;
        Ok(())
    }

    /// Check if a join key might be present
    pub fn contains(&self, key: &JoinKeyValue) -> bool {
        let bytes = key.to_bytes();
        self.sbbf.check(&bytes[..])
    }

    /// Check for intersection with another filter
    pub fn has_intersection(&self, other: &Self) -> bool {
        let a = self.sbbf.to_bytes();
        let b = other.sbbf.to_bytes();
        bloom_bitwise_and_nonzero(&a, &b)
    }

    /// Get the join key columns
    pub fn join_key_columns(&self) -> &[String] {
        &self.join_key_columns
    }

    /// Get the estimated size in bytes
    pub fn estimated_size_bytes(&self) -> usize {
        self.sbbf.size_bytes()
    }

    /// Convert to typed protobuf JoinKeyMetadata (Bloom variant)
    pub fn to_pb_filter(&self) -> pb::JoinKeyMetadata {
        let bitmap = self.sbbf.to_bytes();
        pb::JoinKeyMetadata {
            columns: self.join_key_columns.clone(),
            filter: Some(pb::join_key_metadata::Filter::Bloom(pb::BloomFilterData {
                bitmap,
                num_hashes: 8,
                bitmap_bits: (self.sbbf.size_bytes() as u32) * 8,
            })),
        }
    }

    /// Get the number of items
    pub fn len(&self) -> usize {
        self.item_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.item_count == 0
    }

    /// Check if this filter might produce false positives (Bloom filters are probabilistic)
    pub fn might_have_false_positives(&self) -> bool {
        true
    }
}

/// Typed JoinKeyMetadata model used to bridge protobuf field and in-memory logic.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub enum FilterType {
    ExactSet(HashSet<u64>),
    Bloom {
        bitmap: Vec<u8>,
        num_hashes: u32,
        bitmap_bits: u32,
    },
}

#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct JoinKeyMetadata {
    pub columns: Vec<String>,
    pub filter: FilterType,
}

impl JoinKeyMetadata {
    pub fn from_exact_bloom(bloom: &JoinKeyBloomFilter) -> Self {
        // Legacy function name: now produces Bloom filter from SBBF
        let bitmap = bloom.sbbf.to_bytes();
        let bitmap_bits = (bloom.sbbf.size_bytes() as u32) * 8;
        Self {
            columns: bloom.join_key_columns.clone(),
            filter: FilterType::Bloom {
                bitmap,
                num_hashes: 8,
                bitmap_bits,
            },
        }
    }

    pub fn to_pb(&self) -> pb::JoinKeyMetadata {
        match &self.filter {
            FilterType::ExactSet(hashes) => pb::JoinKeyMetadata {
                columns: self.columns.clone(),
                filter: Some(pb::join_key_metadata::Filter::ExactSet(pb::ExactSet {
                    key_hashes: hashes.iter().copied().collect(),
                })),
            },
            FilterType::Bloom {
                bitmap,
                num_hashes,
                bitmap_bits,
            } => pb::JoinKeyMetadata {
                columns: self.columns.clone(),
                filter: Some(pb::join_key_metadata::Filter::Bloom(pb::BloomFilterData {
                    bitmap: bitmap.clone(),
                    num_hashes: *num_hashes,
                    bitmap_bits: *bitmap_bits,
                })),
            },
        }
    }

    pub fn from_pb(message: &pb::JoinKeyMetadata) -> Result<Self> {
        let columns = message.columns.clone();
        let filter = match message.filter.as_ref() {
            Some(pb::join_key_metadata::Filter::ExactSet(exact)) => {
                FilterType::ExactSet(exact.key_hashes.iter().copied().collect())
            }
            Some(pb::join_key_metadata::Filter::Bloom(b)) => FilterType::Bloom {
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
        join_key::{JoinKeyBloomFilter, JoinKeyValue},
    };
    use crate::dataset::conflict_detection::{FilterType, JoinKeyMetadata};
    use lance_table::format::pb;

    #[test]
    fn test_join_key_value_hash() {
        let key1 = JoinKeyValue::String("test".to_string());
        let key2 = JoinKeyValue::String("test".to_string());
        let key3 = JoinKeyValue::String("different".to_string());

        assert_eq!(key1.hash_value(), key2.hash_value());
        assert_ne!(key1.hash_value(), key3.hash_value());
    }

    #[test]
    fn test_filter_operations() {
        let mut filter = JoinKeyBloomFilter::new(vec!["id".to_string()]);
        let key = JoinKeyValue::String("test_key".to_string());

        // Insert and check
        filter.insert(key.clone()).unwrap();
        assert!(filter.contains(&key));

        // Check non-existent key
        let other_key = JoinKeyValue::String("other_key".to_string());
        assert!(!filter.contains(&other_key));
    }

    #[test]
    fn test_intersection_detection() {
        let mut filter1 = JoinKeyBloomFilter::new(vec!["id".to_string()]);
        let mut filter2 = JoinKeyBloomFilter::new(vec!["id".to_string()]);

        let key1 = JoinKeyValue::String("shared_key".to_string());
        let key2 = JoinKeyValue::String("unique_key1".to_string());
        let key3 = JoinKeyValue::String("unique_key2".to_string());

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
        let mut bloom_filter = JoinKeyBloomFilter::new(vec!["user_id".to_string()]);

        let key1 = JoinKeyValue::String("alice".to_string());
        let key2 = JoinKeyValue::String("bob".to_string());
        let key3 = JoinKeyValue::String("charlie".to_string());

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
            JoinKeyBloomFilter::new(vec!["tenant_id".to_string(), "user_id".to_string()]);

        let composite_key1 = JoinKeyValue::Composite(vec![
            JoinKeyValue::String("tenant_a".to_string()),
            JoinKeyValue::String("user_001".to_string()),
        ]);

        let composite_key2 = JoinKeyValue::Composite(vec![
            JoinKeyValue::String("tenant_a".to_string()),
            JoinKeyValue::String("user_002".to_string()),
        ]);

        let composite_key3 = JoinKeyValue::Composite(vec![
            JoinKeyValue::String("tenant_b".to_string()),
            JoinKeyValue::String("user_001".to_string()),
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

        let mut filter1 = JoinKeyBloomFilter::new(vec!["user_id".to_string()]);
        let mut filter2 = JoinKeyBloomFilter::new(vec!["user_id".to_string()]);

        let keys1 = [
            JoinKeyValue::String("alice".to_string()),
            JoinKeyValue::String("bob".to_string()),
            JoinKeyValue::String("charlie".to_string()),
        ];

        for key in keys1.iter() {
            filter1.insert(key.clone()).unwrap();
        }

        let keys2 = [
            JoinKeyValue::String("charlie".to_string()), // duplicated!
            JoinKeyValue::String("david".to_string()),
            JoinKeyValue::String("eve".to_string()),
        ];

        for key in keys2.iter() {
            filter2.insert(key.clone()).unwrap();
        }

        let conflict_result = detector
            .check_filter_conflict(
                &JoinKeyMetadata::from_exact_bloom(&filter1),
                &JoinKeyMetadata::from_exact_bloom(&filter2),
                "test_transaction_uuid",
                2,
            )
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

        let mut filter1 = JoinKeyBloomFilter::new(vec!["user_id".to_string()]);
        let mut filter2 = JoinKeyBloomFilter::new(vec!["user_id".to_string()]);

        let keys1 = [
            JoinKeyValue::String("alice".to_string()),
            JoinKeyValue::String("bob".to_string()),
            JoinKeyValue::String("charlie".to_string()),
        ];

        for key in keys1.iter() {
            filter1.insert(key.clone()).unwrap();
        }

        let keys2 = [
            JoinKeyValue::String("david".to_string()),
            JoinKeyValue::String("eve".to_string()),
            JoinKeyValue::String("frank".to_string()),
        ];

        for key in keys2.iter() {
            filter2.insert(key.clone()).unwrap();
        }

        let conflict_result = detector
            .check_filter_conflict(
                &JoinKeyMetadata::from_exact_bloom(&filter1),
                &JoinKeyMetadata::from_exact_bloom(&filter2),
                "test_transaction_uuid",
                2,
            )
            .unwrap();

        assert!(
            !conflict_result.has_conflict(),
            "should not detect conflict"
        );
    }

    #[test]
    fn test_pb_exact_set_encode_decode_and_intersection() {
        let mut filter = JoinKeyBloomFilter::new(vec!["id".to_string()]);
        let k1 = JoinKeyValue::String("a".to_string());
        let k2 = JoinKeyValue::String("b".to_string());
        filter.insert(k1.clone()).unwrap();
        filter.insert(k2).unwrap();

        let pb_filter = filter.to_pb_filter();
        let model = JoinKeyMetadata::from_pb(&pb_filter).unwrap();
        assert_eq!(model.columns, vec!["id".to_string()]);
        match model.filter {
            FilterType::Bloom {
                ref bitmap,
                num_hashes,
                bitmap_bits,
            } => {
                assert!(!bitmap.is_empty());
                assert_eq!(num_hashes, 8);
                assert_eq!(bitmap_bits as usize, bitmap.len() * 8);
            }
            _ => panic!("expected bloom"),
        }

        let mut other = JoinKeyBloomFilter::new(vec!["id".to_string()]);
        other.insert(k1).unwrap();
        let other_model = JoinKeyMetadata::from_exact_bloom(&other);
        let (has, fp) = model.intersects(&other_model);
        assert!(has);
        assert!(fp);
    }

    #[test]
    fn test_threshold_based_storage_strategy() {
        let mut small_filter = JoinKeyBloomFilter::new(vec!["id".to_string()]);
        for i in 0..5 {
            let key = JoinKeyValue::String(format!("small_{}", i));
            small_filter.insert(key).unwrap();
        }
        let small_size = small_filter.estimated_size_bytes();
        assert!(small_size == 16 * 1024 || small_size == 32 * 1024);

        let mut large_filter = JoinKeyBloomFilter::new(vec!["id".to_string()]);
        for i in 0..1000 {
            let key = JoinKeyValue::String(format!("large_{:04}", i));
            large_filter.insert(key).unwrap();
        }
        let large_size = large_filter.estimated_size_bytes();
        assert_eq!(small_size, large_size);
        assert!(large_size < 200 * 1024);
    }

    #[test]
    fn test_pb_performance_baseline_sizes() {
        fn make_keys(n: usize) -> Vec<JoinKeyValue> {
            (0..n)
                .map(|i| JoinKeyValue::String(format!("k{:06}", i)))
                .collect()
        }
        for &n in &[1000usize, 10_000usize] {
            let mut filter = JoinKeyBloomFilter::new(vec!["id".to_string()]);
            for k in make_keys(n) {
                filter.insert(k).unwrap();
            }
            let pb = filter.to_pb_filter();
            match pb.filter {
                Some(pb::join_key_metadata::Filter::Bloom(b)) => {
                    assert!(!b.bitmap.is_empty());
                    assert_eq!(b.bitmap_bits as usize, b.bitmap.len() * 8);
                }
                _ => panic!("expected bloom"),
            }
        }
    }
}
