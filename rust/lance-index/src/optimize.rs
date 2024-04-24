// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// How to handle new unindexed data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NewDataHandling {
    /// Do not index new data
    Ignore,
    /// Index all unindexed data
    IndexAll,
    /// Index only new data in specified fragments. The fragments are
    /// specified by their ids.
    Fragments(Vec<u32>),
}

/// How to merge indices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexHandling {
    /// Put all new data into it's own delta index.
    ///
    /// If NewDataHandling::Ignore is used, this is a no-op.
    NewDelta,
    /// Merge new data and the latest N indices into a single index.
    ///
    /// If NewDataHandling::Ignore is used, this just merges the latest N indices.
    /// Unless N=1, then this is a no-op.
    MergeLatestN(usize),
    /// Merge all indices into a single index.
    MergeAll,
    /// Merge new data and the indices with the specified UUIDs. Only indices with
    /// the same name will be merged together. You can pass the UUIDs of the
    /// deltas of multiple indices, and they will be merged together into one
    /// index per name.
    ///
    /// If NewDataHandling::Ignore is used, this just merges the specified indices.
    MergeIndices(Vec<Uuid>),
}

/// Options for optimizing all indices.
///
/// To create a delta index with new data, write:
///
/// ```rust
/// # use lance_index::optimize::{OptimizeOptions, NewDataHandling, IndexHandling};
/// OptimizeOptions {
///     new_data_handling: NewDataHandling::IndexAll,
///     index_handling: IndexHandling::NewDelta,
/// };
/// ```
///
/// To merge all existing indices without adding new data, write:
///
/// ```rust
/// # use lance_index::optimize::{OptimizeOptions, NewDataHandling, IndexHandling};
/// OptimizeOptions {
///    new_data_handling: NewDataHandling::Ignore,
///    index_handling: IndexHandling::MergeAll,
/// }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeOptions {
    /// How to handle new unindexed data.
    pub new_data_handling: NewDataHandling,

    /// How to merge indices.
    pub index_handling: IndexHandling,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            new_data_handling: NewDataHandling::IndexAll,
            index_handling: IndexHandling::MergeLatestN(1),
        }
    }
}
