// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Minting a new base path.

use super::mask::{Key, ManifestMask, Region};
use super::refs::TxnContext;
use crate::format::pb;
use crate::format::{BasePath, Manifest};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Mint a new base path.
///
/// The `BasePath` fields are flattened rather than embedded so the meaningless
/// `id` — which `actions.proto` documents as ignored — has no in-memory
/// representation; [`AddBase::apply`] constructs the `BasePath` with the id it
/// mints.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddBase {
    /// Token for the base id this action mints, resolved at apply. Must be
    /// distinct from every other local token in the same operation.
    pub local: u32,
    /// Optional human-readable alias, matching `BasePath::name`. `None` means the
    /// base is addressed only by path.
    pub name: Option<String>,
    /// Whether this base is a dataset root rather than a plain file directory.
    pub is_dataset_root: bool,
    /// The full URI, e.g. `"s3://bucket/path"`, matching `BasePath::path`. Not an
    /// `object_store::Path`, which is store-relative and percent-normalized and
    /// so would drop the scheme and bucket.
    pub path: String,
}

impl AddBase {
    pub(super) fn apply(&self, manifest: &mut Manifest, ctx: &mut TxnContext) -> Result<()> {
        // A name or path already in use cannot be freed by retrying, so the
        // collision check runs before anything is minted.
        if let Some(existing) = manifest
            .base_paths
            .values()
            .find(|base| (self.name.is_some() && base.name == self.name) || base.path == self.path)
        {
            return Err(Error::invalid_input(format!(
                "Conflict detected: Base path with name '{:?}' or path '{}' already exists. Existing: name='{:?}', path='{}'",
                self.name, self.path, existing.name, existing.path
            )));
        }

        // Base ids are *not* a manifest counter: there is no `max_base_id` field in
        // `Manifest` or `table.proto`, so the next id is derived from the current
        // maximum key. Nothing in the tree removes a base, so ids happen to be
        // monotone — but only by that accident. Relocation-by-watermark
        // (actions.proto principle 4) assumes a never-reused counter, so OSS-1529
        // needs either a real counter field or a written-down never-remove
        // invariant. Within one lineage, which is all this does, deriving from the
        // max is equivalent.
        let minted = match manifest.base_paths.keys().max() {
            Some(&max) => max.checked_add(1).ok_or_else(|| {
                Error::invalid_input(format!(
                    "cannot mint a base id: the maximum base id {max} is already u32::MAX"
                ))
            })?,
            None => 1,
        };

        manifest.base_paths.insert(
            minted,
            BasePath::new(
                minted,
                self.path.clone(),
                self.name.clone(),
                self.is_dataset_root,
            ),
        );
        ctx.bind_base(self.local, minted)
    }

    pub(super) fn writes(&self) -> ManifestMask {
        let mut keys = vec![Key::Path(self.path.clone())];
        keys.extend(self.name.clone().map(Key::Name));
        ManifestMask::default().with_keys(Region::Bases, keys)
    }
}

impl TryFrom<pb::AddBase> for AddBase {
    type Error = Error;

    fn try_from(message: pb::AddBase) -> Result<Self> {
        let base = message
            .base
            .ok_or_else(|| Error::invalid_input("AddBase message did not contain a base path"))?;
        Ok(Self {
            local: message.local,
            name: base.name,
            is_dataset_root: base.is_dataset_root,
            path: base.path,
        })
    }
}

impl From<&AddBase> for pb::AddBase {
    fn from(action: &AddBase) -> Self {
        Self {
            local: action.local,
            base: Some(pb::BasePath {
                // Ignored on read; the id is stamped in at apply.
                id: 0,
                name: action.name.clone(),
                is_dataset_root: action.is_dataset_root,
                path: action.path.clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataStorageFormat;
    use lance_core::datatypes::Schema;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn manifest_with_bases(bases: Vec<BasePath>) -> Manifest {
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.base_paths = bases.into_iter().map(|base| (base.id, base)).collect();
        manifest
    }

    fn add_base(local: u32, name: Option<&str>, path: &str) -> AddBase {
        AddBase {
            local,
            name: name.map(str::to_string),
            is_dataset_root: false,
            path: path.to_string(),
        }
    }

    #[test]
    fn test_apply_mints_first_id_on_empty_base_map() {
        let mut manifest = manifest_with_bases(vec![]);
        let mut ctx = TxnContext::default();

        add_base(0, Some("warm"), "s3://bucket/warm")
            .apply(&mut manifest, &mut ctx)
            .unwrap();

        assert_eq!(manifest.base_paths.len(), 1);
        let base = &manifest.base_paths[&1];
        assert_eq!(base.id, 1);
        assert_eq!(base.name.as_deref(), Some("warm"));
        assert_eq!(base.path, "s3://bucket/warm");
        assert!(!base.is_dataset_root);
    }

    #[test]
    fn test_apply_mints_above_existing_max() {
        // Not `len() + 1`: the id comes from the maximum existing key, so a gap in
        // the map does not cause a collision.
        let mut manifest = manifest_with_bases(vec![
            BasePath::new(1, "s3://bucket/a".into(), Some("a".into()), false),
            BasePath::new(7, "s3://bucket/b".into(), Some("b".into()), false),
        ]);
        let mut ctx = TxnContext::default();

        add_base(0, Some("c"), "s3://bucket/c")
            .apply(&mut manifest, &mut ctx)
            .unwrap();

        assert_eq!(manifest.base_paths[&8].name.as_deref(), Some("c"));
    }

    #[test]
    fn test_two_actions_mint_distinct_ids_and_bind_both() {
        let mut manifest = manifest_with_bases(vec![]);
        let mut ctx = TxnContext::default();

        add_base(0, Some("warm"), "s3://bucket/warm")
            .apply(&mut manifest, &mut ctx)
            .unwrap();
        add_base(1, Some("cold"), "s3://bucket/cold")
            .apply(&mut manifest, &mut ctx)
            .unwrap();

        assert_eq!(ctx.bound_base(0), Some(1));
        assert_eq!(ctx.bound_base(1), Some(2));
        assert_eq!(manifest.base_paths.len(), 2);
    }

    #[test]
    fn test_apply_rejects_duplicate_name() {
        let mut manifest = manifest_with_bases(vec![BasePath::new(
            1,
            "s3://bucket/a".into(),
            Some("warm".into()),
            false,
        )]);
        let err = add_base(0, Some("warm"), "s3://bucket/other")
            .apply(&mut manifest, &mut TxnContext::default())
            .unwrap_err();

        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        assert!(err.to_string().contains("already exists"));
        assert_eq!(manifest.base_paths.len(), 1);
    }

    #[test]
    fn test_apply_rejects_duplicate_path() {
        let mut manifest = manifest_with_bases(vec![BasePath::new(
            1,
            "s3://bucket/a".into(),
            Some("warm".into()),
            false,
        )]);
        let err = add_base(0, Some("cold"), "s3://bucket/a")
            .apply(&mut manifest, &mut TxnContext::default())
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn test_unnamed_bases_do_not_collide_on_name() {
        // Two bases may both be unnamed: `None` is the absence of a claim, not a
        // name that can be taken. (The legacy apply path compares `name == name`
        // unconditionally and so wrongly rejects this; see the PR description.)
        let mut manifest =
            manifest_with_bases(vec![BasePath::new(1, "s3://bucket/a".into(), None, false)]);
        add_base(0, None, "s3://bucket/b")
            .apply(&mut manifest, &mut TxnContext::default())
            .unwrap();

        assert_eq!(manifest.base_paths.len(), 2);
        assert_eq!(manifest.base_paths[&2].path, "s3://bucket/b");
    }

    #[test]
    fn test_writes_claims_name_and_path() {
        let named = add_base(0, Some("warm"), "s3://bucket/warm").writes();
        assert_eq!(
            named.conflicts_with(&add_base(1, Some("warm"), "s3://bucket/other").writes()),
            Some(super::super::mask::Conflict::Incompatible)
        );
        assert_eq!(
            named.conflicts_with(&add_base(1, Some("cold"), "s3://bucket/warm").writes()),
            Some(super::super::mask::Conflict::Incompatible)
        );
        assert_eq!(
            named.conflicts_with(&add_base(1, Some("cold"), "s3://bucket/cold").writes()),
            None
        );
    }

    #[test]
    fn test_writes_of_unnamed_base_claims_only_path() {
        let unnamed = add_base(0, None, "s3://bucket/a").writes();
        assert_eq!(
            unnamed.conflicts_with(&add_base(1, None, "s3://bucket/b").writes()),
            None
        );
        assert_eq!(
            unnamed.conflicts_with(&add_base(1, None, "s3://bucket/a").writes()),
            Some(super::super::mask::Conflict::Incompatible)
        );
    }

    #[test]
    fn test_proto_roundtrips() {
        for action in [
            add_base(0, Some("warm"), "s3://bucket/warm"),
            add_base(3, None, "/local/path"),
            AddBase {
                local: 1,
                name: Some("root".into()),
                is_dataset_root: true,
                path: "s3://bucket/root".into(),
            },
        ] {
            let message = pb::AddBase::from(&action);
            assert_eq!(AddBase::try_from(message).unwrap(), action);
        }
    }

    #[test]
    fn test_proto_without_base_is_rejected() {
        let err = AddBase::try_from(pb::AddBase {
            local: 0,
            base: None,
        })
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("did not contain a base path"));
    }
}
