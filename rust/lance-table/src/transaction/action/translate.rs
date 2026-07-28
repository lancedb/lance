// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Expressing a legacy [`Operation`] as actions.
//!
//! This is a parity-testing device and the eventual write-path bridge: it lets a
//! legacy operation and its action decomposition be applied to the same manifest
//! and compared, which is what proves the action path is not quietly diverging.
//!
//! It is deliberately *not* used on the conflict path. Reducing everything to
//! action-versus-action there would need a translation for every operation (this
//! translates one), some operations cannot translate without the prior manifest at
//! their own read version, and it would mean allocating an action list for every
//! concurrent transaction just to learn which regions it touched. Footprints
//! ([`ManifestMask`]) answer that question directly.
//!
//! [`ManifestMask`]: super::mask::ManifestMask

use super::{Action, AddBase};
use crate::transaction::Operation;
use lance_core::{Error, Result};

impl TryFrom<&Operation> for Vec<Action> {
    type Error = Error;

    fn try_from(operation: &Operation) -> Result<Self> {
        match operation {
            Operation::UpdateBases { new_bases } => new_bases
                .iter()
                .enumerate()
                .map(|(index, base)| {
                    if base.id != 0 {
                        // The caller pre-assigned an id, which the action model does
                        // not express: ids are minted at apply. Rejecting is the only
                        // faithful answer — translating would silently drop the id.
                        return Err(Error::invalid_input(format!(
                            "cannot express UpdateBases entry with pre-assigned base id {} \
                             as an AddBase action; base ids are minted when the action is applied",
                            base.id
                        )));
                    }
                    let local = u32::try_from(index).map_err(|_| {
                        Error::invalid_input(format!(
                            "UpdateBases carries {} bases, more than a local token can address",
                            new_bases.len()
                        ))
                    })?;
                    Ok(Action::AddBase(AddBase {
                        local,
                        name: base.name.clone(),
                        is_dataset_root: base.is_dataset_root,
                        path: base.path.clone(),
                    }))
                })
                .collect(),
            other => Err(Error::not_supported(format!(
                "expressing the {} operation as actions is not supported yet",
                other.name()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BasePath, DataStorageFormat, Manifest};
    use crate::transaction::action::{UserAction, UserOperation};
    use crate::transaction::test_support::default_build_config;
    use crate::transaction::{Transaction, TransactionBuilder};
    use lance_core::datatypes::Schema;
    use std::collections::HashMap;
    use std::sync::Arc;

    const TXN_PATH: &str = "_transactions/parity.txn";

    fn current() -> Manifest {
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.base_paths = HashMap::from_iter([
            (
                1,
                BasePath::new(1, "s3://bucket/a".into(), Some("a".into()), false),
            ),
            (
                4,
                BasePath::new(4, "s3://bucket/b".into(), Some("b".into()), false),
            ),
        ]);
        manifest
    }

    fn unassigned(name: Option<&str>, path: &str) -> BasePath {
        BasePath::new(0, path.to_string(), name.map(str::to_string), false)
    }

    #[test]
    fn test_update_bases_translates_to_one_add_base_per_entry() {
        let operation = Operation::UpdateBases {
            new_bases: vec![
                unassigned(Some("warm"), "s3://bucket/warm"),
                unassigned(None, "/local/cold"),
            ],
        };
        let actions = Vec::<Action>::try_from(&operation).unwrap();

        assert_eq!(
            actions,
            vec![
                Action::AddBase(AddBase {
                    local: 0,
                    name: Some("warm".to_string()),
                    is_dataset_root: false,
                    path: "s3://bucket/warm".to_string(),
                }),
                Action::AddBase(AddBase {
                    local: 1,
                    name: None,
                    is_dataset_root: false,
                    path: "/local/cold".to_string(),
                }),
            ]
        );
    }

    #[test]
    fn test_preassigned_base_id_is_rejected() {
        let operation = Operation::UpdateBases {
            new_bases: vec![BasePath::new(9, "s3://bucket/x".into(), None, false)],
        };
        let err = Vec::<Action>::try_from(&operation).unwrap_err();

        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        assert!(err.to_string().contains("pre-assigned base id 9"));
    }

    #[test]
    fn test_other_operations_are_not_supported_yet() {
        let operation = Operation::ReserveFragments { num_fragments: 1 };
        let err = Vec::<Action>::try_from(&operation).unwrap_err();

        assert!(
            matches!(err, Error::NotSupported { .. }),
            "expected NotSupported, got: {err:?}"
        );
        assert!(err.to_string().contains("ReserveFragments"));
    }

    /// The most valuable test in the slice: a legacy `UpdateBases` and its action
    /// decomposition must produce the same manifest. `transaction_file` and
    /// `timestamp_nanos` are excluded because they differ by construction.
    #[test]
    fn test_round_trip_parity_with_legacy_update_bases() {
        let current = current();
        let config = default_build_config();

        let legacy_operation = Operation::UpdateBases {
            new_bases: vec![
                unassigned(Some("warm"), "s3://bucket/warm"),
                unassigned(None, "/local/cold"),
            ],
        };
        let actions = Vec::<Action>::try_from(&legacy_operation).unwrap();

        let legacy_transaction: Transaction =
            TransactionBuilder::new(current.version, legacy_operation).build();
        let (legacy, legacy_indices) = legacy_transaction
            .build_manifest(Some(&current), vec![], TXN_PATH, &config)
            .unwrap();

        let user_operation = UserOperation {
            description: "ALTER TABLE t ADD BASE".to_string(),
            uuid: "0197f6bd-0000-4000-8000-000000000000".to_string(),
            read_version: current.version,
            actions: vec![UserAction {
                description: "register bases".to_string(),
                actions,
            }],
        };
        let (v2, v2_indices) = user_operation
            .build_manifest(Some(&current), vec![], None, TXN_PATH, &config)
            .unwrap();

        assert_eq!(v2.base_paths, legacy.base_paths);
        assert_eq!(v2.schema, legacy.schema);
        assert_eq!(v2.fragments, legacy.fragments);
        assert_eq!(v2.version, legacy.version);
        assert_eq!(v2.next_row_id, legacy.next_row_id);
        assert_eq!(v2.max_fragment_id, legacy.max_fragment_id);
        assert_eq!(v2.reader_feature_flags, legacy.reader_feature_flags);
        assert_eq!(v2.writer_feature_flags, legacy.writer_feature_flags);
        assert_eq!(v2.config, legacy.config);
        assert_eq!(v2.table_metadata, legacy.table_metadata);
        assert_eq!(v2.data_storage_format, legacy.data_storage_format);
        assert_eq!(v2_indices.len(), legacy_indices.len());

        // The ids the two paths mint must agree, not merely the count.
        assert_eq!(v2.base_paths[&5].name.as_deref(), Some("warm"));
        assert_eq!(v2.base_paths[&6].path, "/local/cold");
    }
}
