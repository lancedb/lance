// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row version tracking for cross-version diff functionality
//!
//! This module provides data structures and functionality to track the latest
//! update version for each row in a Lance dataset, enabling efficient
//! cross-version diff operations.

use deepsize::DeepSizeOf;
use lance_core::Error;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::format::{pb, ExternalFile, Fragment};
use crate::rowids::segment::U64Segment;
use crate::rowids::{read_row_ids, RowIdSequence};

/// A run of identical versions over a contiguous span of row positions.
///
/// Span is expressed as a U64Segment over row offsets (0..N within a fragment),
/// not over row IDs. This keeps the encoding aligned with RowIdSequence order
/// and enables zipped iteration without building a map.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct RowVersionRun {
    span: U64Segment,
    version: u64,
}

impl RowVersionRun {
    /// Number of rows covered by this run.
    pub fn len(&self) -> usize {
        self.span.len()
    }

    /// Whether this run covers no rows.
    pub fn is_empty(&self) -> bool {
        self.span.is_empty()
    }

    /// The version value of this run.
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// Sequence of row latest update versions
///
/// Stores version runs aligned to the positional order of RowIdSequence.
/// Provides sequential iterators and optional lightweight indexing for
/// efficient random access.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf, Default)]
pub struct RowLatestUpdateVersionSequence {
    runs: Vec<RowVersionRun>,
    /// Optional cumulative run lengths index (prefix sums of run lengths).
    /// Not serialized; built on demand via `build_index`.
    index: Option<Vec<usize>>, // prefix_end[i] = total length up to and including runs[i]
}

impl RowLatestUpdateVersionSequence {
    /// Create a new empty version sequence
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            index: None,
        }
    }

    /// Create a version sequence with a single uniform run of `row_count` rows.
    pub fn from_uniform_row_count(row_count: u64, version: u64) -> Self {
        if row_count == 0 {
            return Self::new();
        }
        let run = RowVersionRun {
            span: U64Segment::Range(0..row_count),
            version,
        };
        Self {
            runs: vec![run],
            index: None,
        }
    }

    /// Number of rows tracked by this sequence (sum of run lengths).
    pub fn len(&self) -> u64 {
        self.runs.iter().map(|s| s.len() as u64).sum()
    }

    /// Empty if there are no runs or all runs are empty.
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty() || self.runs.iter().all(|s| s.is_empty())
    }

    /// Returns a forward iterator over versions, expanding runs lazily.
    pub fn versions(&self) -> VersionsIter<'_> {
        VersionsIter::new(&self.runs)
    }

    /// Zipped iteration of row ids and versions without materializing a map.
    /// The positional order must be the same between `row_ids` and this sequence.
    pub fn zip_rows_with_versions<'a>(
        &'a self,
        row_ids: &'a RowIdSequence,
    ) -> impl Iterator<Item = (u64, u64)> + 'a {
        row_ids.iter().zip(self.versions())
    }

    /// Build (or rebuild) a lightweight prefix-sum index for random access.
    /// The index stores cumulative run lengths: prefix_end[i] = sum(runs[0..=i].len).
    pub fn build_index(&mut self) {
        let mut prefix = Vec::with_capacity(self.runs.len());
        let mut acc = 0usize;
        for r in &self.runs {
            acc += r.len();
            prefix.push(acc);
        }
        self.index = Some(prefix);
    }

    /// Random access: get the version at global row position `index`.
    /// If an index has been built (via `build_index`), performs a binary search (O(log R)).
    /// Otherwise, falls back to a linear scan (O(R)). Returns None if out of bounds.
    pub fn version_at(&self, index: usize) -> Option<u64> {
        if let Some(prefix) = &self.index {
            if prefix.is_empty() {
                return None;
            }
            // Bounds check
            if index >= *prefix.last().unwrap() {
                return None;
            }
            // Binary search for the first prefix_end > index
            match prefix.binary_search(&(index + 1)) {
                Ok(pos) => Some(self.runs[pos].version()),
                Err(pos) => Some(self.runs[pos].version()),
            }
        } else {
            // Linear scan across runs
            let mut offset = 0usize;
            for run in &self.runs {
                let len = run.len();
                if index < offset + len {
                    return Some(run.version());
                }
                offset += len;
            }
            None
        }
    }

    /// Get the version associated with a specific row id.
    /// This reconstructs the positional offset from RowIdSequence and then
    /// performs `version_at` lookup.
    pub fn get_version_for_row_id(&self, row_ids: &RowIdSequence, row_id: u64) -> Option<u64> {
        let mut offset = 0usize;
        for seg in &row_ids.0 {
            if seg.range().is_some_and(|r| r.contains(&row_id)) {
                if let Some(local) = seg.position(row_id) {
                    return self.version_at(offset + local);
                }
            }
            offset += seg.len();
        }
        None
    }

    /// Convenience: collect row IDs with version strictly greater than `threshold`.
    pub fn rows_with_version_greater_than(
        &self,
        row_ids: &RowIdSequence,
        threshold: u64,
    ) -> Vec<u64> {
        row_ids
            .iter()
            .zip(self.versions())
            .filter_map(|(rid, v)| if v > threshold { Some(rid) } else { None })
            .collect()
    }
}

/// Iterator over versions expanding runs lazily.
pub struct VersionsIter<'a> {
    runs: &'a [RowVersionRun],
    run_idx: usize,
    remaining_in_run: usize,
    current_version: u64,
}

impl<'a> VersionsIter<'a> {
    fn new(runs: &'a [RowVersionRun]) -> Self {
        let mut it = Self {
            runs,
            run_idx: 0,
            remaining_in_run: 0,
            current_version: 0,
        };
        it.advance_run();
        it
    }

    fn advance_run(&mut self) {
        if self.run_idx < self.runs.len() {
            let run = &self.runs[self.run_idx];
            self.remaining_in_run = run.len();
            self.current_version = run.version();
        } else {
            self.remaining_in_run = 0;
        }
    }
}

impl<'a> Iterator for VersionsIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_in_run == 0 {
            // Move to next run
            self.run_idx += 1;
            if self.run_idx >= self.runs.len() {
                return None;
            }
            self.advance_run();
        }
        self.remaining_in_run = self.remaining_in_run.saturating_sub(1);
        Some(self.current_version)
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
    // Convert to protobuf sequence
    let pb_sequence = pb::RowLatestUpdateVersionSequence {
        runs: sequence
            .runs
            .iter()
            .map(|run| pb::RowVersionRun {
                span: Some(pb::U64Segment::from(run.span.clone())),
                version: run.version,
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
        .runs
        .into_iter()
        .map(|pb_run| {
            let positions_pb = pb_run.span.ok_or_else(|| Error::Internal {
                message: "Missing positions in RowVersionRun".to_string(),
                location: location!(),
            })?;
            let segment = U64Segment::try_from(positions_pb)?;
            Ok(RowVersionRun {
                span: segment,
                version: pb_run.version,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(RowLatestUpdateVersionSequence {
        runs: segments,
        index: None,
    })
}

/// Set version metadata for a list of fragments
pub fn set_version_metadata_for_fragments(fragments: &mut [Fragment], current_version: u64) {
    for fragment in fragments.iter_mut() {
        // Only set version metadata if the fragment has rows
        if let Some(physical_rows) = fragment.physical_rows {
            if physical_rows > 0 {
                let row_count = if let Some(row_id_meta) = &fragment.row_id_meta {
                    match row_id_meta {
                        crate::format::RowIdMeta::Inline(data) => {
                            let sequence = read_row_ids(data).unwrap();
                            sequence.len()
                        }
                        crate::format::RowIdMeta::External(_file) => {
                            todo!("Currently, not supported!")
                        }
                    }
                } else {
                    panic!("Can not find row id meta, please make sure you have enabled stable row id.")
                };

                let version_sequence = RowLatestUpdateVersionSequence::from_uniform_row_count(
                    row_count,
                    current_version,
                );

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
    fn test_versions_iter_and_zip() {
        // Build runs: [3 * v1] [2 * v2] [1 * v3]
        let seq = RowLatestUpdateVersionSequence {
            runs: vec![
                RowVersionRun {
                    span: U64Segment::Range(0..3),
                    version: 1,
                },
                RowVersionRun {
                    span: U64Segment::Range(0..2),
                    version: 2,
                },
                RowVersionRun {
                    span: U64Segment::Range(0..1),
                    version: 3,
                },
            ],
            index: None,
        };
        let row_ids = RowIdSequence::from(100..106);

        let versions: Vec<u64> = seq.versions().collect();
        assert_eq!(versions, vec![1, 1, 1, 2, 2, 3]);

        let pairs: Vec<(u64, u64)> = seq.zip_rows_with_versions(&row_ids).collect();
        assert_eq!(
            pairs,
            vec![(100, 1), (101, 1), (102, 1), (103, 2), (104, 2), (105, 3)]
        );
    }

    #[test]
    fn test_version_random_access_indexing() {
        let mut seq = RowLatestUpdateVersionSequence {
            runs: vec![
                RowVersionRun {
                    span: U64Segment::Range(0..3),
                    version: 1,
                },
                RowVersionRun {
                    span: U64Segment::Range(0..2),
                    version: 2,
                },
                RowVersionRun {
                    span: U64Segment::Range(0..1),
                    version: 3,
                },
            ],
            index: None,
        };
        assert_eq!(seq.version_at(0), Some(1));
        assert_eq!(seq.version_at(2), Some(1));
        assert_eq!(seq.version_at(3), Some(2));
        assert_eq!(seq.version_at(4), Some(2));
        assert_eq!(seq.version_at(5), Some(3));
        assert_eq!(seq.version_at(6), None);

        // Build index and verify binary search path
        seq.build_index();
        assert_eq!(seq.version_at(0), Some(1));
        assert_eq!(seq.version_at(2), Some(1));
        assert_eq!(seq.version_at(3), Some(2));
        assert_eq!(seq.version_at(4), Some(2));
        assert_eq!(seq.version_at(5), Some(3));
        assert_eq!(seq.version_at(6), None);
    }

    #[test]
    fn test_serialization_round_trip() {
        let seq = RowLatestUpdateVersionSequence {
            runs: vec![
                RowVersionRun {
                    span: U64Segment::Range(0..4),
                    version: 42,
                },
                RowVersionRun {
                    span: U64Segment::Range(0..3),
                    version: 99,
                },
            ],
            index: None,
        };
        let bytes = write_row_latest_update_versions(&seq);
        let seq2 = read_row_latest_update_versions(&bytes).unwrap();
        assert_eq!(seq2.runs.len(), 2);
        assert_eq!(seq2.len(), 7);
        assert_eq!(seq2.version_at(0), Some(42));
        assert_eq!(seq2.version_at(5), Some(99));
    }

    #[test]
    fn test_get_version_for_row_id() {
        let seq = RowLatestUpdateVersionSequence {
            runs: vec![
                RowVersionRun {
                    span: U64Segment::Range(0..2),
                    version: 8,
                },
                RowVersionRun {
                    span: U64Segment::Range(0..2),
                    version: 9,
                },
            ],
            index: None,
        };
        let rows = RowIdSequence::from(10..14); // row ids: 10,11,12,13
        assert_eq!(seq.get_version_for_row_id(&rows, 10), Some(8));
        assert_eq!(seq.get_version_for_row_id(&rows, 11), Some(8));
        assert_eq!(seq.get_version_for_row_id(&rows, 12), Some(9));
        assert_eq!(seq.get_version_for_row_id(&rows, 13), Some(9));
        assert_eq!(seq.get_version_for_row_id(&rows, 99), None);
    }
}
