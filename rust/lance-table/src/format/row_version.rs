// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row version tracking for cross-version diff functionality
//!
//! This module provides data structures and functionality to track the latest
//! update version for each row in a Lance dataset, enabling efficient
//! cross-version diff operations.

// mod storage;
//
// pub use storage::{};

use std::ops::Range;

use deepsize::DeepSizeOf;
use lance_core::Error;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::format::pb;

/// A segment of row version information, optimized for different storage patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum RowVersionSegment {
    /// Uniform range: consecutive rows with the same version number
    /// This is the most common case when rows are inserted in batches
    UniformRange {
        start_row: u64,
        end_row: u64,
        version: u64,
    },
    /// Incremental range: consecutive rows with incrementally increasing versions
    /// This occurs when rows are updated in sequence
    IncrementalRange {
        start_row: u64,
        end_row: u64,
        start_version: u64,
    },
    /// Sparse storage: discrete (row_id, version) pairs
    /// Used when version patterns are irregular
    Sparse(Vec<(u64, u64)>),
}

impl RowVersionSegment {
    /// Get the number of rows covered by this segment
    pub fn len(&self) -> u64 {
        match self {
            Self::UniformRange {
                start_row, end_row, ..
            } => end_row - start_row,
            Self::IncrementalRange {
                start_row, end_row, ..
            } => end_row - start_row,
            Self::Sparse(pairs) => pairs.len() as u64,
        }
    }

    /// Check if this segment is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the version for a specific row ID within this segment
    pub fn get_version(&self, row_id: u64) -> Option<u64> {
        match self {
            Self::UniformRange {
                start_row,
                end_row,
                version,
            } => {
                if row_id >= *start_row && row_id < *end_row {
                    Some(*version)
                } else {
                    None
                }
            }
            Self::IncrementalRange {
                start_row,
                end_row,
                start_version,
            } => {
                if row_id >= *start_row && row_id < *end_row {
                    Some(start_version + (row_id - start_row))
                } else {
                    None
                }
            }
            Self::Sparse(pairs) => pairs
                .iter()
                .find(|(id, _)| *id == row_id)
                .map(|(_, version)| *version),
        }
    }

    /// Get the row ID range covered by this segment
    pub fn row_range(&self) -> Option<Range<u64>> {
        match self {
            Self::UniformRange {
                start_row, end_row, ..
            } => Some(*start_row..*end_row),
            Self::IncrementalRange {
                start_row, end_row, ..
            } => Some(*start_row..*end_row),
            Self::Sparse(pairs) => {
                if pairs.is_empty() {
                    None
                } else {
                    let min_row = pairs.iter().map(|(id, _)| *id).min().unwrap();
                    let max_row = pairs.iter().map(|(id, _)| *id).max().unwrap();
                    Some(min_row..max_row + 1)
                }
            }
        }
    }

    /// Iterate over all (row_id, version) pairs in this segment
    pub fn iter_with_versions(&self) -> Box<dyn Iterator<Item = (u64, u64)> + '_> {
        match self {
            Self::UniformRange {
                start_row,
                end_row,
                version,
            } => Box::new((*start_row..*end_row).map(move |row_id| (row_id, *version))),
            Self::IncrementalRange {
                start_row,
                end_row,
                start_version,
            } => Box::new(
                (*start_row..*end_row)
                    .map(move |row_id| (row_id, start_version + (row_id - start_row))),
            ),
            Self::Sparse(pairs) => Box::new(pairs.iter().copied()),
        }
    }

    /// Get the minimum version in this segment
    pub fn min_version(&self) -> Option<u64> {
        match self {
            Self::UniformRange { version, .. } => Some(*version),
            Self::IncrementalRange { start_version, .. } => Some(*start_version),
            Self::Sparse(pairs) => pairs.iter().map(|(_, v)| *v).min(),
        }
    }

    /// Get the maximum version in this segment
    pub fn max_version(&self) -> Option<u64> {
        match self {
            Self::UniformRange { version, .. } => Some(*version),
            Self::IncrementalRange {
                start_row,
                end_row,
                start_version,
            } => Some(start_version + (end_row - start_row - 1)),
            Self::Sparse(pairs) => pairs.iter().map(|(_, v)| *v).max(),
        }
    }

    /// Create an optimal segment from a list of (row_id, version) pairs
    pub fn from_pairs(mut pairs: Vec<(u64, u64)>) -> Self {
        if pairs.is_empty() {
            return Self::Sparse(pairs);
        }

        // Sort by row_id
        pairs.sort_by_key(|(row_id, _)| *row_id);

        // Check if this can be a uniform range
        if pairs.len() > 1 {
            let first_version = pairs[0].1;
            let is_uniform = pairs.iter().all(|(_, v)| *v == first_version);
            let is_contiguous = pairs.windows(2).all(|w| w[1].0 == w[0].0 + 1);

            if is_uniform && is_contiguous {
                return Self::UniformRange {
                    start_row: pairs[0].0,
                    end_row: pairs.last().unwrap().0 + 1,
                    version: first_version,
                };
            }

            // Check if this can be an incremental range
            if is_contiguous {
                let is_incremental = pairs.windows(2).all(|w| w[1].1 == w[0].1 + 1);
                if is_incremental {
                    return Self::IncrementalRange {
                        start_row: pairs[0].0,
                        end_row: pairs.last().unwrap().0 + 1,
                        start_version: pairs[0].1,
                    };
                }
            }
        }

        // Fall back to sparse storage
        Self::Sparse(pairs)
    }
}

/// Sequence of row latest update versions
///
/// This structure tracks the latest update version for each row in a fragment,
/// optimized for different patterns of version assignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf, Default)]
pub struct RowLatestUpdateVersionSequence(Vec<RowVersionSegment>);

impl RowLatestUpdateVersionSequence {
    /// Create a new empty version sequence
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a version sequence from a list of (row_id, version) pairs
    pub fn from_pairs(pairs: Vec<(u64, u64)>) -> Self {
        if pairs.is_empty() {
            return Self::new();
        }

        // Group pairs by contiguous patterns to create optimal segments
        let mut segments = Vec::new();
        let mut current_pairs = Vec::new();

        let mut sorted_pairs = pairs;
        sorted_pairs.sort_by_key(|(row_id, _)| *row_id);

        for (row_id, version) in sorted_pairs {
            current_pairs.push((row_id, version));

            // Create a segment when we have enough pairs or detect a pattern break
            if current_pairs.len() >= 100 {
                // Arbitrary threshold for segment size
                segments.push(RowVersionSegment::from_pairs(current_pairs.clone()));
                current_pairs.clear();
            }
        }

        // Add remaining pairs as a segment
        if !current_pairs.is_empty() {
            segments.push(RowVersionSegment::from_pairs(current_pairs));
        }

        Self(segments)
    }

    /// Get the version for a specific row ID
    pub fn get_version(&self, row_id: u64) -> Option<u64> {
        for segment in &self.0 {
            if let Some(version) = segment.get_version(row_id) {
                return Some(version);
            }
        }
        None
    }

    /// Get the total number of rows tracked
    pub fn len(&self) -> u64 {
        self.0.iter().map(|s| s.len()).sum()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty() || self.0.iter().all(|s| s.is_empty())
    }

    /// Iterate over all (row_id, version) pairs
    pub fn iter_with_versions(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.0
            .iter()
            .flat_map(|segment| segment.iter_with_versions())
    }

    /// Get the minimum version in the sequence
    pub fn min_version(&self) -> Option<u64> {
        self.0.iter().filter_map(|s| s.min_version()).min()
    }

    /// Get the maximum version in the sequence
    pub fn max_version(&self) -> Option<u64> {
        self.0.iter().filter_map(|s| s.max_version()).max()
    }

    /// Add a new (row_id, version) pair to the sequence
    pub fn insert(&mut self, row_id: u64, version: u64) {
        // For simplicity, add as a sparse segment
        // In a production implementation, we might want to merge with existing segments
        self.0
            .push(RowVersionSegment::Sparse(vec![(row_id, version)]));
    }

    /// Extend this sequence with another sequence
    pub fn extend(&mut self, other: Self) {
        self.0.extend(other.0);
    }

    /// Get all row IDs that have versions greater than the specified threshold
    pub fn rows_with_version_greater_than(&self, threshold: u64) -> Vec<u64> {
        self.iter_with_versions()
            .filter(|(_, version)| *version > threshold)
            .map(|(row_id, _)| row_id)
            .collect()
    }

    /// Serialize to bytes for storage
    pub fn to_bytes(&self) -> lance_core::Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::Internal {
            message: format!("Failed to serialize RowLatestUpdateVersionSequence: {}", e),
            location: location!(),
        })
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> lance_core::Result<Self> {
        bincode::deserialize(data).map_err(|e| Error::Internal {
            message: format!(
                "Failed to deserialize RowLatestUpdateVersionSequence: {}",
                e
            ),
            location: location!(),
        })
    }
}

// impl Default for RowLatestUpdateVersionSequence {
//     fn default() -> Self {
//         Self::new()
//     }
// }
//
// impl From<Vec<(u64, u64)>> for RowLatestUpdateVersionSequence {
//     fn from(pairs: Vec<(u64, u64)>) -> Self {
//         Self::from_pairs(&pairs)
//     }
// }

/// Metadata about the location of row version sequence data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum RowLatestUpdateVersionMeta {
    /// Small sequences stored inline in the fragment metadata
    Inline(Vec<u8>),
    /// Large sequences stored in external files
    External(ExternalFile),
}

/// Reference to an external file containing row version data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct ExternalFile {
    pub path: String,
    pub offset: u64,
    pub size: u64,
}

impl RowLatestUpdateVersionMeta {
    /// Create inline metadata from a version sequence
    pub fn from_sequence_inline(
        sequence: &RowLatestUpdateVersionSequence,
    ) -> lance_core::Result<Self> {
        let bytes = sequence.to_bytes()?;
        Ok(Self::Inline(bytes))
    }

    /// Create external metadata reference
    pub fn from_external_file(path: String, offset: u64, size: u64) -> Self {
        Self::External(ExternalFile { path, offset, size })
    }

    /// Load the version sequence from this metadata
    pub async fn load_sequence(
        &self,
        _object_store: &dyn object_store::ObjectStore,
    ) -> lance_core::Result<RowLatestUpdateVersionSequence> {
        match self {
            Self::Inline(data) => RowLatestUpdateVersionSequence::from_bytes(data),
            Self::External(_file) => {
                // TODO: Implement external file loading
                // This would involve reading from the object store at the specified path/offset/size
                todo!("External file loading not yet implemented")
            }
        }
    }
}

// Protobuf conversion implementations
impl TryFrom<pb::data_fragment::RowLatestUpdatedVersionSequence> for RowLatestUpdateVersionMeta {
    type Error = Error;

    fn try_from(
        value: pb::data_fragment::RowLatestUpdatedVersionSequence,
    ) -> Result<Self, Self::Error> {
        match value {
            pb::data_fragment::RowLatestUpdatedVersionSequence::InlineRowLatestUpdatedVersions(data) => {
                Ok(Self::Inline(data))
            }
            pb::data_fragment::RowLatestUpdatedVersionSequence::ExternalRowLatestUpdatedVersions(file) => {
                Ok(Self::External(ExternalFile {
                    path: file.path,
                    offset: file.offset,
                    size: file.size,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_range_segment() {
        let segment = RowVersionSegment::UniformRange {
            start_row: 10,
            end_row: 20,
            version: 5,
        };

        assert_eq!(segment.len(), 10);
        assert_eq!(segment.get_version(15), Some(5));
        assert_eq!(segment.get_version(5), None);
        assert_eq!(segment.get_version(25), None);
        assert_eq!(segment.min_version(), Some(5));
        assert_eq!(segment.max_version(), Some(5));
    }

    #[test]
    fn test_incremental_range_segment() {
        let segment = RowVersionSegment::IncrementalRange {
            start_row: 10,
            end_row: 15,
            start_version: 100,
        };

        assert_eq!(segment.len(), 5);
        assert_eq!(segment.get_version(10), Some(100));
        assert_eq!(segment.get_version(12), Some(102));
        assert_eq!(segment.get_version(14), Some(104));
        assert_eq!(segment.get_version(15), None);
        assert_eq!(segment.min_version(), Some(100));
        assert_eq!(segment.max_version(), Some(104));
    }

    #[test]
    fn test_sparse_segment() {
        let pairs = vec![(1, 10), (5, 20), (10, 30)];
        let segment = RowVersionSegment::Sparse(pairs.clone());

        assert_eq!(segment.len(), 3);
        assert_eq!(segment.get_version(1), Some(10));
        assert_eq!(segment.get_version(5), Some(20));
        assert_eq!(segment.get_version(10), Some(30));
        assert_eq!(segment.get_version(3), None);
        assert_eq!(segment.min_version(), Some(10));
        assert_eq!(segment.max_version(), Some(30));
    }

    #[test]
    fn test_segment_from_pairs_uniform() {
        let pairs = vec![(10, 5), (11, 5), (12, 5), (13, 5)];
        let segment = RowVersionSegment::from_pairs(pairs);

        match segment {
            RowVersionSegment::UniformRange {
                start_row,
                end_row,
                version,
            } => {
                assert_eq!(start_row, 10);
                assert_eq!(end_row, 14);
                assert_eq!(version, 5);
            }
            _ => panic!("Expected UniformRange"),
        }
    }

    #[test]
    fn test_segment_from_pairs_incremental() {
        let pairs = vec![(10, 100), (11, 101), (12, 102), (13, 103)];
        let segment = RowVersionSegment::from_pairs(pairs);

        match segment {
            RowVersionSegment::IncrementalRange {
                start_row,
                end_row,
                start_version,
            } => {
                assert_eq!(start_row, 10);
                assert_eq!(end_row, 14);
                assert_eq!(start_version, 100);
            }
            _ => panic!("Expected IncrementalRange"),
        }
    }

    #[test]
    fn test_segment_from_pairs_sparse() {
        let pairs = vec![(1, 10), (5, 20), (10, 15)];
        let segment = RowVersionSegment::from_pairs(pairs.clone());

        match segment {
            RowVersionSegment::Sparse(stored_pairs) => {
                assert_eq!(stored_pairs, pairs);
            }
            _ => panic!("Expected Sparse"),
        }
    }

    #[test]
    fn test_version_sequence() {
        let pairs = vec![(1, 10), (2, 10), (3, 10), (10, 20), (15, 25)];
        let sequence = RowLatestUpdateVersionSequence::from_pairs(pairs);

        assert_eq!(sequence.len(), 5);
        assert_eq!(sequence.get_version(2), Some(10));
        assert_eq!(sequence.get_version(10), Some(20));
        assert_eq!(sequence.get_version(15), Some(25));
        assert_eq!(sequence.get_version(5), None);
        assert_eq!(sequence.min_version(), Some(10));
        assert_eq!(sequence.max_version(), Some(25));

        let rows_gt_15 = sequence.rows_with_version_greater_than(15);
        assert_eq!(rows_gt_15, vec![10, 15]);
    }

    #[test]
    fn test_serialization() {
        let pairs = vec![(1, 10), (2, 11), (3, 12)];
        let sequence = RowLatestUpdateVersionSequence::from_pairs(pairs);

        let bytes = sequence.to_bytes().unwrap();
        let deserialized = RowLatestUpdateVersionSequence::from_bytes(&bytes).unwrap();

        assert_eq!(sequence, deserialized);
    }
}
