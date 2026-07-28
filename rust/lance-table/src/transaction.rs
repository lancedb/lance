// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transaction definitions for updating datasets
//!
//! Prior to creating a new manifest, a transaction must be created representing
//! the changes being made to the dataset. By representing them as incremental
//! changes, we can detect whether concurrent operations are compatible with
//! one another. We can also rebuild manifests when retrying committing a
//! manifest.
//!
//! For more details please refer to the
//! [Transaction Specification](https://lance.org/format/table/transaction/#transaction-types).
//!
//! The work splits along these lines:
//!
//! ```text
//! builder            Transaction: an operation plus the version it was based on
//! operation          the vocabulary of changes an operation can describe
//! update_map         incremental edits to the manifest's string maps
//! validate           pre-commit checks against the manifest being replaced
//! manifest_build     applying an operation to produce the next manifest
//! index_maintenance    how that narrows or drops index metadata
//! row_version          how it assigns row ids and per-row version metadata
//! conflicts          whether two operations collide, for the commit retry path
//! proto              the persisted protobuf encoding of all of the above
//! ```

mod builder;
mod conflicts;
mod index_maintenance;
mod manifest_build;
mod operation;
mod proto;
mod row_version;
mod update_map;
mod validate;

#[cfg(test)]
pub(crate) mod test_support;

pub use builder::{Transaction, TransactionBuilder};
pub use operation::{
    DataOverlayGroup, DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, UpdateMode,
    UpdatedFragmentOffsets,
};
pub use update_map::{
    UpdateMap, UpdateMapEntry, translate_config_updates, translate_schema_metadata_updates,
};
pub use validate::validate_operation;
