// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Error;
use snafu::Snafu;

/// Why a column write could not be staged or committed.
///
/// Returned as the `source` of an [`Error::InvalidInput`]. Recover it with
/// [`ColumnWriteError::of`] to branch on the failure programmatically; the
/// message text is for humans and is not a stable interface.
///
/// The variants split into three groups. [`Self::FragmentNotFound`] and
/// [`Self::RowCountMismatch`] come from staging. The staged-file and
/// prepare-time variants report a commit the caller assembled wrongly. The
/// freshness variants ([`Self::StaleSchema`], [`Self::StaleFragmentData`],
/// [`Self::OrphanedReplacements`], [`Self::UncoveredNonNullableColumn`],
/// [`Self::ReadVersionUnavailable`]) report that the dataset moved under a
/// prepared write; the column must be recomputed against the current dataset
/// and staged again.
#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum ColumnWriteError {
    #[snafu(display("fragment {fragment_id} not found in dataset"))]
    FragmentNotFound { fragment_id: u64 },

    #[snafu(display(
        "column '{name}' already exists; project it out of the dataset schema to rewrite it \
         rather than numbering it as a new column"
    ))]
    ColumnAlreadyExists { name: String },

    #[snafu(display("column '{name}' is a reserved name and cannot be used in a Lance dataset"))]
    ReservedColumnName { name: String },

    #[snafu(display(
        "column data for fragment {fragment_id} has {staged_rows} rows but the fragment has \
         {physical_rows} physical rows"
    ))]
    RowCountMismatch {
        fragment_id: u64,
        staged_rows: u64,
        physical_rows: u64,
    },

    #[snafu(display("no column replacements to commit"))]
    NoReplacements,

    #[snafu(display(
        "file '{path}' was not staged by write_fragment_column (missing or unparsable \
         read-version metadata)"
    ))]
    NotStaged { path: String },

    #[snafu(display(
        "staged files span dataset versions {first} and {second}; stage all fragments against \
         one version"
    ))]
    MixedReadVersions { first: u64, second: u64 },

    #[snafu(display(
        "staged files disagree on field id {field_id}: '{name}' ({data_type}) vs '{other_name}' \
         ({other_data_type})"
    ))]
    StagedSchemasDisagree {
        field_id: i32,
        name: String,
        data_type: String,
        other_name: String,
        other_data_type: String,
    },

    #[snafu(display("staged file '{path}' records no field ids"))]
    NoStagedFieldIds { path: String },

    #[snafu(display(
        "staged file '{path}' records field id {field_id}, which is not in its staged schema"
    ))]
    StagedFieldNotInSchema { path: String, field_id: i32 },

    #[snafu(display(
        "fragment {fragment_id} has multiple staged files covering field id {field_id}"
    ))]
    DuplicateFieldCoverage { fragment_id: u64, field_id: i32 },

    #[snafu(display(
        "dataset version {read_version} the columns were staged against is no longer available; \
         recompute against the current dataset"
    ))]
    ReadVersionUnavailable { read_version: u64 },

    #[snafu(display(
        "fragment {fragment_id} does not exist at dataset version {read_version} the write was \
         prepared against"
    ))]
    FragmentNotInReadVersion { fragment_id: u64, read_version: u64 },

    #[snafu(display(
        "staged column '{name}' ({data_type}) conflicts with field id {field_id} ('{other_name}', \
         {other_data_type}) in dataset version {read_version} it was prepared against"
    ))]
    FieldContractConflict {
        field_id: i32,
        name: String,
        data_type: String,
        other_name: String,
        other_data_type: String,
        read_version: u64,
    },

    #[snafu(display(
        "field id {field_id} ('{name}', {data_type}) was computed against a stale schema; \
         recompute the column against the current dataset"
    ))]
    StaleSchema {
        field_id: i32,
        name: String,
        data_type: String,
    },

    #[snafu(display(
        "fragment {fragment_id}'s data for field id {field_id} changed since the column was \
         prepared; recompute against the current dataset"
    ))]
    StaleFragmentData { fragment_id: u64, field_id: i32 },

    #[snafu(display(
        "column replacements reference fragments {fragment_ids:?}, which are no longer in the \
         dataset; re-run the column write against the current fragments"
    ))]
    OrphanedReplacements { fragment_ids: Vec<u64> },

    #[snafu(display(
        "non-nullable column '{name}' has no staged data for fragment {fragment_id} (possibly \
         appended concurrently); stage every live fragment and recommit"
    ))]
    UncoveredNonNullableColumn { name: String, fragment_id: u64 },
}

impl ColumnWriteError {
    /// Recover the column-write cause behind a [`Dataset::write_fragment_column`]
    /// or [`Dataset::commit_column_writes`] failure, if `error` came from one.
    ///
    /// [`Dataset::write_fragment_column`]: crate::Dataset::write_fragment_column
    /// [`Dataset::commit_column_writes`]: crate::Dataset::commit_column_writes
    pub fn of(error: &Error) -> Option<&Self> {
        match error {
            Error::InvalidInput { source, .. } => source.downcast_ref(),
            _ => None,
        }
    }

    /// True when the dataset moved under a prepared write. The staged files
    /// cannot be replayed; recompute the column against the current dataset.
    pub fn needs_recompute(&self) -> bool {
        matches!(
            self,
            Self::StaleSchema { .. }
                | Self::StaleFragmentData { .. }
                | Self::OrphanedReplacements { .. }
                | Self::UncoveredNonNullableColumn { .. }
                | Self::ReadVersionUnavailable { .. }
        )
    }
}

impl From<ColumnWriteError> for Error {
    fn from(error: ColumnWriteError) -> Self {
        Self::invalid_input_source(Box::new(error))
    }
}
