// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Assembling the next manifest from an operation's actions.
//!
//! This is the action vocabulary's counterpart to [`manifest_build`]. It stays
//! separate rather than threading actions through that function's two match
//! phases: the legacy assembly is a frozen compatibility surface, and an action
//! applies to the manifest directly instead of being folded into a post-image.
//!
//! [`manifest_build`]: crate::transaction::manifest_build

use super::UserOperation;
use super::refs::TxnContext;
use crate::feature_flags::apply_feature_flags;
use crate::format::{IndexMetadata, Manifest, ManifestBuildConfig};
use lance_core::{Error, Result};

impl UserOperation {
    /// Build the manifest this operation produces from the one it applies to.
    ///
    /// `tag` is the enclosing transaction's tag. Actions are applied in order, so
    /// a later action can resolve a `Local` token minted by an earlier one.
    pub(crate) fn build_manifest(
        &self,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<IndexMetadata>,
        tag: Option<&str>,
        transaction_file_path: &str,
        config: &ManifestBuildConfig,
    ) -> Result<(Manifest, Vec<IndexMetadata>)> {
        let Some(current_manifest) = current_manifest else {
            return Err(Error::not_supported(
                "an action-based operation cannot create a dataset; \
                 it can only be applied to an existing one",
            ));
        };

        // An identity baseline: every change to schema, fragments and bases comes
        // from an action, so nothing here pre-empts one.
        let mut manifest = Manifest::new_from_previous(
            current_manifest,
            current_manifest.schema.clone(),
            current_manifest.fragments.clone(),
        );

        manifest.tag = tag.map(str::to_string);

        if config.auto_set_feature_flags {
            // Mirrors the legacy path: the flag is inherited, so a config built
            // with the default `use_stable_row_ids = false` does not clear it.
            let use_stable_row_ids =
                config.use_stable_row_ids || current_manifest.uses_stable_row_ids();
            apply_feature_flags(
                &mut manifest,
                use_stable_row_ids,
                config.disable_transaction_file,
            )?;
        }
        manifest.set_timestamp(config.timestamp_nanos);
        manifest.update_max_fragment_id();

        let mut ctx = TxnContext::default();
        for action in self.actions() {
            action.apply(&mut manifest, &mut ctx)?;
        }

        manifest.transaction_file = Some(transaction_file_path.to_string());

        Ok((manifest, current_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BasePath, DataStorageFormat};
    use crate::transaction::action::{Action, AddBase, UserAction};
    use crate::transaction::test_support::default_build_config;
    use lance_core::datatypes::Schema;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn operation(actions: Vec<Action>) -> UserOperation {
        UserOperation {
            description: "test".to_string(),
            uuid: "u".to_string(),
            read_version: 1,
            actions: vec![UserAction {
                description: "step".to_string(),
                actions,
            }],
        }
    }

    fn add_base(local: u32, name: Option<&str>, path: &str) -> Action {
        Action::AddBase(AddBase {
            local,
            name: name.map(str::to_string),
            is_dataset_root: false,
            path: path.to_string(),
        })
    }

    fn current() -> Manifest {
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.base_paths = HashMap::from_iter([(
            1,
            BasePath::new(1, "s3://bucket/a".into(), Some("a".into()), false),
        )]);
        manifest
    }

    #[test]
    fn test_build_manifest_applies_actions_and_stamps_scaffolding() {
        let config = ManifestBuildConfig {
            timestamp_nanos: 42,
            ..default_build_config()
        };
        let (manifest, indices) = operation(vec![add_base(0, Some("warm"), "s3://bucket/warm")])
            .build_manifest(
                Some(&current()),
                vec![],
                Some("v7"),
                "_txn/abc.txn",
                &config,
            )
            .unwrap();

        assert_eq!(manifest.base_paths.len(), 2);
        assert_eq!(manifest.base_paths[&2].name.as_deref(), Some("warm"));
        // The pre-existing base is carried forward untouched.
        assert_eq!(manifest.base_paths[&1].path, "s3://bucket/a");
        assert_eq!(manifest.version, current().version + 1);
        assert_eq!(manifest.tag.as_deref(), Some("v7"));
        assert_eq!(manifest.timestamp_nanos, 42);
        assert_eq!(manifest.transaction_file.as_deref(), Some("_txn/abc.txn"));
        assert!(indices.is_empty());
    }

    #[test]
    fn test_build_manifest_applies_actions_in_order_across_steps() {
        let operation = UserOperation {
            description: "test".to_string(),
            uuid: "u".to_string(),
            read_version: 1,
            actions: vec![
                UserAction {
                    description: "first".to_string(),
                    actions: vec![add_base(0, Some("warm"), "s3://bucket/warm")],
                },
                UserAction {
                    description: "second".to_string(),
                    actions: vec![add_base(1, Some("cold"), "s3://bucket/cold")],
                },
            ],
        };
        let (manifest, _) = operation
            .build_manifest(
                Some(&current()),
                vec![],
                None,
                "_txn/abc.txn",
                &default_build_config(),
            )
            .unwrap();

        // Ids are minted in application order from the existing maximum.
        assert_eq!(manifest.base_paths[&2].name.as_deref(), Some("warm"));
        assert_eq!(manifest.base_paths[&3].name.as_deref(), Some("cold"));
    }

    #[test]
    fn test_build_manifest_preserves_indices() {
        let indices = vec![IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: "idx".to_string(),
            fields: vec![0],
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        }];
        let (_, built) = operation(vec![add_base(0, Some("warm"), "s3://bucket/warm")])
            .build_manifest(
                Some(&current()),
                indices.clone(),
                None,
                "_txn/abc.txn",
                &default_build_config(),
            )
            .unwrap();

        assert_eq!(built.len(), 1);
        assert_eq!(built[0].name, indices[0].name);
    }

    #[test]
    fn test_build_manifest_rejects_dataset_creation() {
        let err = operation(vec![add_base(0, Some("warm"), "s3://bucket/warm")])
            .build_manifest(None, vec![], None, "_txn/abc.txn", &default_build_config())
            .unwrap_err();

        assert!(
            matches!(err, Error::NotSupported { .. }),
            "expected NotSupported, got: {err:?}"
        );
    }

    #[test]
    fn test_build_manifest_propagates_action_failure() {
        // A colliding name aborts the whole assembly; no partial manifest escapes.
        let err = operation(vec![add_base(0, Some("a"), "s3://bucket/other")])
            .build_manifest(
                Some(&current()),
                vec![],
                None,
                "_txn/abc.txn",
                &default_build_config(),
            )
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }));
    }

    #[test]
    fn test_build_manifest_rejects_duplicate_local_tokens_at_apply() {
        let err = operation(vec![
            add_base(0, Some("warm"), "s3://bucket/warm"),
            add_base(0, Some("cold"), "s3://bucket/cold"),
        ])
        .build_manifest(
            Some(&current()),
            vec![],
            None,
            "_txn/abc.txn",
            &default_build_config(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("local base token 0"));
    }
}
