// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row version tracking for cross-version diff functionality
//!
//! This module provides data structures and functionality to track the latest
//! update version for each row in a Lance dataset, enabling efficient
//! cross-version diff operations.

use std::ops::Range;

use deepsize::DeepSizeOf;
use lance_core::Error;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::format::{pb, ExternalFile, Fragment};
use crate::rowids::read_row_ids;

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
}

/// Metadata about the location of row version sequence data
/// Following the same pattern as RowIdMeta
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum RowLatestUpdateVersionMeta {
    /// Small sequences stored inline in the fragment metadata
    Inline(Vec<u8>),
    /// Large sequences stored in external files
    External(ExternalFile),
}

impl RowLatestUpdateVersionMeta {
    /// Create inline metadata from a version sequence
    pub fn from_sequence(sequence: &RowLatestUpdateVersionSequence) -> lance_core::Result<Self> {
        let bytes = write_row_latest_update_versions(sequence);
        Ok(Self::Inline(bytes))
    }

    /// Create external metadata reference
    pub fn from_external_file(path: String, offset: u64, size: u64) -> Self {
        Self::External(ExternalFile { path, offset, size })
    }

    /// Load the version sequence from this metadata
    pub fn load_sequence(&self) -> lance_core::Result<RowLatestUpdateVersionSequence> {
        match self {
            Self::Inline(data) => read_row_latest_update_versions(data),
            Self::External(_file) => {
                todo!("External file loading not yet implemented")
            }
        }
    }
}

/// Helper function to convert RowLatestUpdateVersionMeta to protobuf format
pub fn row_latest_update_version_meta_to_pb(
    meta: &Option<RowLatestUpdateVersionMeta>,
) -> Option<pb::data_fragment::RowLatestUpdatedVersionSequence> {
    meta.as_ref().map(|m| match m {
        RowLatestUpdateVersionMeta::Inline(data) => {
            pb::data_fragment::RowLatestUpdatedVersionSequence::InlineRowLatestUpdatedVersions(
                data.clone(),
            )
        }
        RowLatestUpdateVersionMeta::External(file) => {
            pb::data_fragment::RowLatestUpdatedVersionSequence::ExternalRowLatestUpdatedVersions(
                pb::ExternalFile {
                    path: file.path.clone(),
                    offset: file.offset,
                    size: file.size,
                },
            )
        }
    })
}

/// Serialize a row latest update version sequence to a buffer (following RowIdSequence pattern)
pub fn write_row_latest_update_versions(sequence: &RowLatestUpdateVersionSequence) -> Vec<u8> {
    use prost::Message;

    // Convert to protobuf sequence
    let pb_sequence = pb::RowLatestUpdateVersionSequence {
        segments: sequence
            .0
            .iter()
            .map(|segment| {
                let segment_data = match segment {
                    RowVersionSegment::UniformRange {
                        start_row,
                        end_row,
                        version,
                    } => pb::row_version_segment::Segment::UniformRange(
                        pb::row_version_segment::UniformRange {
                            start_row: *start_row,
                            end_row: *end_row,
                            version: *version,
                        },
                    ),
                    RowVersionSegment::IncrementalRange {
                        start_row,
                        end_row,
                        start_version,
                    } => pb::row_version_segment::Segment::IncrementalRange(
                        pb::row_version_segment::IncrementalRange {
                            start_row: *start_row,
                            end_row: *end_row,
                            start_version: *start_version,
                        },
                    ),
                    RowVersionSegment::Sparse(pairs) => pb::row_version_segment::Segment::Sparse(
                        pb::row_version_segment::SparseVersions {
                            pairs: pairs
                                .iter()
                                .map(
                                    |(row_id, version)| pb::row_version_segment::RowVersionPair {
                                        row_id: *row_id,
                                        version: *version,
                                    },
                                )
                                .collect(),
                        },
                    ),
                };
                pb::RowVersionSegment {
                    segment: Some(segment_data),
                }
            })
            .collect(),
    };

    pb_sequence.encode_to_vec()
}

/// Deserialize a row latest update version sequence from bytes (following RowIdSequence pattern)
pub fn read_row_latest_update_versions(
    data: &[u8],
) -> lance_core::Result<RowLatestUpdateVersionSequence> {
    let pb_sequence =
        pb::RowLatestUpdateVersionSequence::decode(data).map_err(|e| Error::Internal {
            message: format!("Failed to decode RowLatestUpdateVersionSequence: {}", e),
            location: location!(),
        })?;

    let segments = pb_sequence
        .segments
        .into_iter()
        .map(|pb_segment| match pb_segment.segment {
            Some(pb::row_version_segment::Segment::UniformRange(uniform)) => {
                Ok(RowVersionSegment::UniformRange {
                    start_row: uniform.start_row,
                    end_row: uniform.end_row,
                    version: uniform.version,
                })
            }
            Some(pb::row_version_segment::Segment::IncrementalRange(incremental)) => {
                Ok(RowVersionSegment::IncrementalRange {
                    start_row: incremental.start_row,
                    end_row: incremental.end_row,
                    start_version: incremental.start_version,
                })
            }
            Some(pb::row_version_segment::Segment::Sparse(sparse)) => {
                let pairs = sparse
                    .pairs
                    .into_iter()
                    .map(|pair| (pair.row_id, pair.version))
                    .collect();
                Ok(RowVersionSegment::Sparse(pairs))
            }
            None => Err(Error::Internal {
                message: "Missing segment data".to_string(),
                location: location!(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(RowLatestUpdateVersionSequence(segments))
}

/// Set version metadata for a list of fragments
pub fn set_version_metadata_for_fragments(fragments: &mut [Fragment], current_version: u64) {
    for fragment in fragments.iter_mut() {
        // Only set version metadata if the fragment has rows
        if let Some(physical_rows) = fragment.physical_rows {
            if physical_rows > 0 {
                let row_ids = if let Some(row_id_meta) = &fragment.row_id_meta {
                    match row_id_meta {
                        crate::format::RowIdMeta::Inline(data) => {
                            // Deserialize row IDs from the inline data
                            let sequence = read_row_ids(data).unwrap();
                            let row_ids: Vec<u64> = sequence.iter().collect();
                            row_ids
                        }
                        crate::format::RowIdMeta::External(_file) => {
                            todo!("Currently, not supported!")
                        }
                    }
                } else {
                    panic!("Can not find row id meta, please make sure you have enabled stable row id.")
                };

                // Check if fragment already has version metadata
                let version_sequence =
                    if let Some(existing_meta) = &fragment.row_latest_update_version_meta {
                        let existing_sequence = existing_meta.load_sequence().unwrap();

                        let mut all_pairs = Vec::new();

                        // First, add all existing pairs that are not being updated
                        for (row_id, version) in existing_sequence.iter_with_versions() {
                            if !row_ids.contains(&row_id) {
                                all_pairs.push((row_id, version));
                            }
                        }

                        // Then, add/update pairs for all row_ids with current_version
                        for row_id in row_ids {
                            all_pairs.push((row_id, current_version));
                            all_pairs.push((row_id, current_version));
                        }

                        RowLatestUpdateVersionSequence::from_pairs(all_pairs)
                    } else {
                        // Create new version sequence with all rows set to current version
                        let pairs: Vec<(u64, u64)> = row_ids
                            .into_iter()
                            .map(|row_id| (row_id, current_version))
                            .collect();

                        RowLatestUpdateVersionSequence::from_pairs(pairs)
                    };

                // Create inline metadata from the sequence
                fragment.row_latest_update_version_meta =
                    Some(RowLatestUpdateVersionMeta::from_sequence(&version_sequence).unwrap());
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
    }

    #[test]
    fn test_sparse_segment() {
        let pairs = vec![(1, 10), (5, 20), (10, 30)];
        let segment = RowVersionSegment::Sparse(pairs);

        assert_eq!(segment.len(), 3);
        assert_eq!(segment.get_version(1), Some(10));
        assert_eq!(segment.get_version(5), Some(20));
        assert_eq!(segment.get_version(10), Some(30));
        assert_eq!(segment.get_version(3), None);
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

        let rows_gt_15 = sequence.rows_with_version_greater_than(15);
        assert_eq!(rows_gt_15, vec![10, 15]);
    }

    #[test]
    fn test_serialization() {
        let pairs = vec![(1, 10), (2, 11), (3, 12)];
        let sequence = RowLatestUpdateVersionSequence::from_pairs(pairs);

        let meta = RowLatestUpdateVersionMeta::from_sequence(&sequence).unwrap();
        let deserialized = meta.load_sequence().unwrap();

        assert_eq!(sequence, deserialized);
    }
}
