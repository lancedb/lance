// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pre-commit validation of an operation against the manifest it applies to.
//!
//! These checks reject transactions that could not produce a coherent manifest —
//! a fragment list that disagrees with the schema, a merge that silently dropped
//! or rewrote data files — before any manifest is written.

use crate::format::{Fragment, Manifest};
use crate::transaction::Operation;
use crate::transaction::action::{Action, UserOperation};
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_file::version::LanceFileVersion;
use std::collections::{HashMap, HashSet};

/// Validate the operation is valid for the given manifest.
pub fn validate_operation(manifest: Option<&Manifest>, operation: &Operation) -> Result<()> {
    let manifest = match (manifest, operation) {
        (
            None,
            Operation::Overwrite {
                fragments, schema, ..
            },
        ) => {
            // Validate here because we are going to return early.
            schema_fragments_valid(None, schema, fragments)?;

            return Ok(());
        }
        (None, Operation::Clone { .. }) => return Ok(()),
        (Some(manifest), _) => manifest,
        (None, _) => {
            return Err(Error::invalid_input(format!(
                "Cannot apply operation {} to non-existent dataset",
                operation.name()
            )));
        }
    };

    match operation {
        Operation::Append { fragments } => {
            // Fragments must contain all fields in the schema
            schema_fragments_valid(Some(manifest), &manifest.schema, fragments)
        }
        Operation::Project { schema } => {
            schema_fragments_valid(Some(manifest), schema, manifest.fragments.as_ref())
        }
        Operation::Merge { fragments, schema } => {
            merge_fragments_valid(manifest, fragments)?;
            schema_fragments_valid(Some(manifest), schema, fragments)
        }
        Operation::Overwrite {
            fragments,
            schema,
            config_upsert_values: None,
            initial_bases: _,
        } => {
            // Pass None for manifest because Overwrite replaces all fragments.
            // The old manifest's storage format is irrelevant for validating
            // the new fragments (e.g., LEGACY→STABLE transitions).
            schema_fragments_valid(None, schema, fragments)
        }
        Operation::Update {
            updated_fragments,
            new_fragments,
            ..
        } => {
            schema_fragments_valid(Some(manifest), &manifest.schema, updated_fragments)?;
            schema_fragments_valid(Some(manifest), &manifest.schema, new_fragments)
        }
        Operation::UserOperation(operation) => validate_user_operation(operation),
        _ => Ok(()),
    }
}

/// Check an action-based operation's own structure, independent of the manifest.
///
/// Local tokens are scoped to the whole operation, not to a single `UserAction`,
/// so distinctness is checked across the flattened action list. Apply would also
/// catch a duplicate, but rejecting here keeps the diagnostic at the API boundary
/// where the caller built the operation.
fn validate_user_operation(operation: &UserOperation) -> Result<()> {
    if operation.actions.iter().all(|step| step.actions.is_empty()) {
        return Err(Error::invalid_input(
            "a UserOperation must contain at least one action",
        ));
    }

    let mut seen = HashSet::new();
    for action in operation.actions() {
        let Action::AddBase(add_base) = action;
        if !seen.insert(add_base.local) {
            return Err(Error::invalid_input(format!(
                "local token {} appears more than once in this UserOperation; \
                 local tokens must be distinct within an operation",
                add_base.local
            )));
        }
    }
    Ok(())
}

fn schema_fragments_valid(
    manifest: Option<&Manifest>,
    schema: &Schema,
    fragments: &[Fragment],
) -> Result<()> {
    if let Some(manifest) = manifest
        && manifest.data_storage_format.lance_file_version()? == LanceFileVersion::Legacy
    {
        return schema_fragments_legacy_valid(schema, fragments);
    }
    // validate that each data file at least contains one field.
    for fragment in fragments {
        for data_file in &fragment.files {
            if data_file.fields.iter().len() == 0 {
                return Err(Error::invalid_input(format!(
                    "Datafile {} does not contain any fields",
                    data_file.path
                )));
            }
        }
    }
    Ok(())
}

/// Check that each fragment contains all fields in the schema.
/// It is not required that the schema contains all fields in the fragment.
/// There may be masked fields.
fn schema_fragments_legacy_valid(schema: &Schema, fragments: &[Fragment]) -> Result<()> {
    // TODO: add additional validation. Consider consolidating with various
    // validate() methods in the codebase.
    for fragment in fragments {
        for field in schema.fields_pre_order() {
            if !fragment
                .files
                .iter()
                .flat_map(|f| f.fields.iter())
                .any(|f_id| f_id == &field.id)
            {
                return Err(Error::invalid_input(format!(
                    "Fragment {} does not contain field {:?}",
                    fragment.id, field
                )));
            }
        }
    }
    Ok(())
}

/// Returns true if Operation::Merge rewrote this fragment's column data files (Fragment::files
/// changed versus the previous manifest). Used to bump last_updated_at_version_meta only when
/// new column values were materialized to disk.
///
/// Deletion file changes alone are not treated as rewrites: tombstones remove rows but
/// survivors did not receive new column bytes; stamping last_updated for those rows would be
/// incorrect for CDF.
#[inline]
pub(super) fn merge_fragment_physically_rewritten(prev: &Fragment, merged: &Fragment) -> bool {
    debug_assert_eq!(prev.id, merged.id);
    if prev.files.len() != merged.files.len() {
        return true;
    }
    // Compare identity fields only. file_size_bytes is an AtomicU64 cache that
    // concurrent scans can populate in place on the manifest's DataFile, so it
    // must not be part of the rewrite check.
    prev.files.iter().zip(merged.files.iter()).any(|(p, m)| {
        p.path != m.path
            || p.fields != m.fields
            || p.column_indices != m.column_indices
            || p.file_major_version != m.file_major_version
            || p.file_minor_version != m.file_minor_version
            || p.base_id != m.base_id
    })
}

/// Validate that Merge operations preserve all original fragments.
/// Merge operations should only add columns or rows, not reduce fragments.
/// This ensures fragments correspond at one-to-one with the original fragment list.
fn merge_fragments_valid(manifest: &Manifest, new_fragments: &[Fragment]) -> Result<()> {
    let original_fragments = manifest.fragments.as_ref();

    // Additional validation: ensure we're not accidentally reducing the fragment count
    if new_fragments.len() < original_fragments.len() {
        return Err(Error::invalid_input(format!(
            "Merge operation reduced fragment count from {} to {}. \
             Merge operations should only add columns, not reduce fragments.",
            original_fragments.len(),
            new_fragments.len()
        )));
    }

    // Collect new fragment IDs
    let new_fragment_map: HashMap<u64, &Fragment> =
        new_fragments.iter().map(|f| (f.id, f)).collect();

    // Check that all original fragments are preserved in the new fragments list
    // Validate that each original fragment's metadata is preserved
    let mut missing_fragments: Vec<u64> = Vec::new();
    for original_fragment in original_fragments {
        if let Some(new_fragment) = new_fragment_map.get(&original_fragment.id) {
            // Validate physical_rows (row count) hasn't changed
            if original_fragment.physical_rows != new_fragment.physical_rows {
                return Err(Error::invalid_input(format!(
                    "Merge operation changed row count for fragment {}. \
                     Original: {:?}, New: {:?}. \
                     Merge operations should preserve fragment row counts and only add new columns.",
                    original_fragment.id,
                    original_fragment.physical_rows,
                    new_fragment.physical_rows
                )));
            }
        } else {
            missing_fragments.push(original_fragment.id);
        }
    }

    if !missing_fragments.is_empty() {
        return Err(Error::invalid_input(format!(
            "Merge operation is missing original fragments: {:?}. \
             Merge operations should preserve all original fragments and only add new columns. \
             Expected fragments: {:?}, but got: {:?}",
            missing_fragments,
            original_fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            new_fragment_map.keys().copied().collect::<Vec<_>>()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{DataFile, DataStorageFormat};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema as LanceSchema;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_merge_fragments_valid() {
        // Create a simple schema for testing
        let schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]);

        // Create original fragments
        let original_fragments = vec![Fragment::new(1), Fragment::new(2), Fragment::new(3)];

        // Create a manifest with original fragments
        let manifest = Manifest::new(
            LanceSchema::try_from(&schema).unwrap(),
            Arc::new(original_fragments),
            DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );

        // Test 1: Empty fragments should fail
        let empty_fragments = vec![];
        let result = merge_fragments_valid(&manifest, &empty_fragments);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reduced fragment count")
        );

        // Test 2: Missing original fragments should fail
        let missing_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            // Fragment 3 is missing
            Fragment::new(4), // New fragment
        ];
        let result = merge_fragments_valid(&manifest, &missing_fragments);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("missing original fragments")
        );

        // Test 3: Reduced fragment count should fail
        let reduced_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            // Fragment 3 is missing, no new fragments added
        ];
        let result = merge_fragments_valid(&manifest, &reduced_fragments);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reduced fragment count")
        );

        // Test 4: Valid merge with all original fragments plus new ones should succeed
        let valid_fragments = vec![
            Fragment::new(1),
            Fragment::new(2),
            Fragment::new(3),
            Fragment::new(4), // New fragment
            Fragment::new(5), // Another new fragment
        ];
        let result = merge_fragments_valid(&manifest, &valid_fragments);
        assert!(result.is_ok());

        // Test 5: Same fragments (no new ones) should succeed
        let same_fragments = vec![Fragment::new(1), Fragment::new(2), Fragment::new(3)];
        let result = merge_fragments_valid(&manifest, &same_fragments);
        assert!(result.is_ok());
    }

    /// Regression test for https://github.com/lance-format/lance/issues/6417
    ///
    /// When overwriting a LEGACY dataset with STABLE-format fragments, the
    /// validation should not use the old manifest's format. STABLE fragments
    /// omit struct parent fields, which the strict legacy check rejects.
    #[test]
    fn test_overwrite_legacy_to_stable_with_struct_fields() {
        use arrow_schema::Fields;

        // Schema: id (field 0), name (field 1), address (field 2, struct parent),
        //   city (field 3), country (field 4)
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
            ArrowField::new(
                "address",
                DataType::Struct(Fields::from(vec![
                    ArrowField::new("city", DataType::Utf8, false),
                    ArrowField::new("country", DataType::Utf8, false),
                ])),
                false,
            ),
        ]);
        let schema = LanceSchema::try_from(&arrow_schema).unwrap();

        // Old manifest is LEGACY format
        let legacy_manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![Fragment::new(0)]),
            DataStorageFormat::new(LanceFileVersion::Legacy),
            HashMap::new(),
        );

        // New fragments in STABLE format omit struct parent field (id=2),
        // only including leaf fields: id=0, name=1, city=3, country=4
        let stable_fragment = Fragment {
            id: 0,
            files: vec![DataFile::new(
                "data.lance",
                vec![0, 1, 3, 4], // no field 2 (struct parent)
                vec![0, 1, 2, 3],
                lance_file::format::MAJOR_VERSION as u32,
                lance_file::format::MINOR_VERSION as u32,
                None,
                None,
            )],
            physical_rows: Some(10),
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };

        let operation = Operation::Overwrite {
            fragments: vec![stable_fragment],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };

        // This should succeed — the old manifest's LEGACY format should not
        // cause strict validation of the new STABLE fragments.
        validate_operation(Some(&legacy_manifest), &operation).unwrap();
    }

    fn user_operation(steps: Vec<Vec<u32>>) -> Operation {
        Operation::UserOperation(UserOperation {
            description: "test".to_string(),
            uuid: "u".to_string(),
            read_version: 1,
            actions: steps
                .into_iter()
                .map(|locals| crate::transaction::UserAction {
                    description: "step".to_string(),
                    actions: locals
                        .into_iter()
                        .map(|local| {
                            Action::AddBase(crate::transaction::AddBase {
                                local,
                                name: Some(format!("base-{local}")),
                                is_dataset_root: false,
                                path: format!("s3://bucket/{local}"),
                            })
                        })
                        .collect(),
                })
                .collect(),
        })
    }

    fn manifest() -> Manifest {
        Manifest::new(
            LanceSchema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::default(),
            HashMap::new(),
        )
    }

    #[test]
    fn test_user_operation_with_distinct_local_tokens_is_valid() {
        validate_operation(
            Some(&manifest()),
            &user_operation(vec![vec![0, 1], vec![2]]),
        )
        .unwrap();
    }

    #[test]
    fn test_user_operation_with_duplicate_local_token_is_rejected() {
        // Tokens are scoped to the whole operation, not to a single UserAction, so a
        // token reused across two steps is still a duplicate.
        let err = validate_operation(Some(&manifest()), &user_operation(vec![vec![0], vec![0]]))
            .unwrap_err();

        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        assert!(
            err.to_string()
                .contains("local token 0 appears more than once")
        );
    }

    #[test]
    fn test_user_operation_with_no_actions_is_rejected() {
        for steps in [vec![], vec![vec![]]] {
            let err = validate_operation(Some(&manifest()), &user_operation(steps)).unwrap_err();
            assert!(
                err.to_string().contains("at least one action"),
                "got: {err}"
            );
        }
    }
}
