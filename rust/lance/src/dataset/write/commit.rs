// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_table::{
    format::DataStorageFormat,
    io::commit::{CommitConfig, CommitHandler, ManifestNamingScheme},
};
use snafu::{location, Location};

use crate::{
    dataset::{
        builder::DatasetBuilder,
        commit_detached_transaction, commit_new_dataset, commit_transaction,
        refs::Tags,
        transaction::{Operation, Transaction},
        ManifestWriteConfig, ReadParams,
    },
    session::Session,
    Dataset, Error, Result,
};

use super::{resolve_commit_handler, WriteDestination};

/// Create a new commit from a [`Transaction`].
///
/// Transactions can be created using a write method like [`super::InsertBuilder::execute_uncommitted`].
#[derive(Debug, Clone)]
pub struct CommitBuilder<'a> {
    dest: WriteDestination<'a>,
    use_move_stable_row_ids: Option<bool>,
    enable_v2_manifest_paths: bool,
    storage_format: Option<LanceFileVersion>,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    store_params: Option<ObjectStoreParams>,
    object_store_registry: Arc<ObjectStoreRegistry>,
    object_store: Option<Arc<ObjectStore>>,
    session: Option<Arc<Session>>,
    detached: bool,
    commit_config: CommitConfig,
}

impl<'a> CommitBuilder<'a> {
    pub fn new(dest: impl Into<WriteDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            use_move_stable_row_ids: None,
            enable_v2_manifest_paths: false,
            storage_format: None,
            commit_handler: None,
            store_params: None,
            object_store_registry: Default::default(),
            object_store: None,
            session: None,
            detached: false,
            commit_config: Default::default(),
        }
    }

    /// Whether to use move-stable row ids. This makes the `_rowid` column stable
    /// after compaction, but not updates.
    ///
    /// This is only used for new datasets. Existing datasets will use their
    /// existing setting.
    ///
    /// **Default is false.**
    pub fn use_move_stable_row_ids(mut self, use_move_stable_row_ids: bool) -> Self {
        self.use_move_stable_row_ids = Some(use_move_stable_row_ids);
        self
    }

    /// Pass the storage format to use for the dataset.
    ///
    /// This is only needed when creating a new empty table. If any data files are
    /// passed, the storage format will be inferred from the data files.
    ///
    /// All data files must use the same storage format as the existing dataset.
    /// If a different format is passed, an error will be returned.
    pub fn with_storage_format(mut self, storage_format: LanceFileVersion) -> Self {
        self.storage_format = Some(storage_format);
        self
    }

    /// Pass an object store to use.
    pub fn with_object_store(mut self, object_store: Arc<ObjectStore>) -> Self {
        self.object_store = Some(object_store);
        self
    }

    /// Pass a commit handler to use for the dataset.
    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.commit_handler = Some(commit_handler);
        self
    }

    /// Pass store parameters to use for the dataset.
    ///
    /// If an object store is passed, these parameters will be ignored.
    pub fn with_store_params(mut self, store_params: ObjectStoreParams) -> Self {
        self.store_params = Some(store_params);
        self
    }

    /// Pass an object store registry to use.
    ///
    /// If an object store is passed, this registry will be ignored.
    pub fn with_object_store_registry(
        mut self,
        object_store_registry: Arc<ObjectStoreRegistry>,
    ) -> Self {
        self.object_store_registry = object_store_registry;
        self
    }

    /// Pass a session to use for the dataset.
    ///
    /// If a session is not passed, but a dataset is used as the destination,
    /// then the dataset's session will be used.
    ///
    /// By passing a session or re-using a dataset, you can re-use the
    /// file metadata and index caches, which can significantly improve
    /// performance.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    ///  If set to true, and this is a new dataset, uses the new v2 manifest
    ///  paths. These allow constant-time lookups for the latest manifest on object storage.
    ///  This parameter has no effect on existing datasets. To migrate an existing
    ///  dataset, use the [`Dataset::migrate_manifest_paths_v2`] method. **Default is False.**
    ///
    /// <div class="warning">
    ///  WARNING: turning this on will make the dataset unreadable for older
    ///  versions of Lance (prior to 0.17.0).
    /// </div>
    pub fn enable_v2_manifest_paths(mut self, enable: bool) -> Self {
        self.enable_v2_manifest_paths = enable;
        self
    }

    /// Commit a version that is not part of the mainline history.
    ///
    /// This commit will never show up in the dataset's history.
    ///
    /// This can be used to stage changes or to handle "secondary" datasets
    /// whose lineage is tracked elsewhere.
    pub fn with_detached(mut self, detached: bool) -> Self {
        self.detached = detached;
        self
    }

    /// Set the maximum number of retries for commit operations.
    ///
    /// If a commit operation fails, it will be retried up to `max_retries` times.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.commit_config.num_retries = max_retries;
        self
    }

    pub async fn execute(self, transaction: Transaction) -> Result<Dataset> {
        let (object_store, base_path, commit_handler) = match &self.dest {
            WriteDestination::Dataset(dataset) => (
                dataset.object_store.clone(),
                dataset.base.clone(),
                dataset.commit_handler.clone(),
            ),
            WriteDestination::Uri(uri) => {
                let (object_store, base_path) = ObjectStore::from_uri_and_params(
                    self.object_store_registry.clone(),
                    uri,
                    &self.store_params.clone().unwrap_or_default(),
                )
                .await?;
                let mut object_store = Arc::new(object_store);
                let commit_handler = if self.commit_handler.is_some() && self.object_store.is_some()
                {
                    self.commit_handler.as_ref().unwrap().clone()
                } else {
                    resolve_commit_handler(uri, self.commit_handler.clone(), &self.store_params)
                        .await?
                };
                if let Some(passed_store) = self.object_store {
                    object_store = passed_store;
                }
                (object_store, base_path, commit_handler)
            }
        };

        let dest = match &self.dest {
            WriteDestination::Dataset(dataset) => WriteDestination::Dataset(dataset.clone()),
            WriteDestination::Uri(uri) => {
                // Check if it already exists.
                let mut builder = DatasetBuilder::from_uri(uri).with_read_params(ReadParams {
                    store_options: self.store_params.clone(),
                    commit_handler: self.commit_handler.clone(),
                    object_store_registry: self.object_store_registry.clone(),
                    ..Default::default()
                });

                // If read_version is zero, then it might not have originally been
                // passed. We can assume the latest version.
                if transaction.read_version > 0 {
                    builder = builder.with_version(transaction.read_version)
                }

                match builder.load().await {
                    Ok(dataset) => WriteDestination::Dataset(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        WriteDestination::Uri(uri)
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        if dest.dataset().is_none() && !matches!(transaction.operation, Operation::Overwrite { .. })
        {
            return Err(Error::DatasetNotFound {
                path: base_path.to_string(),
                source: "The dataset must already exist unless the operation is Overwrite".into(),
                location: location!(),
            });
        }

        let manifest_naming_scheme = if let Some(ds) = dest.dataset() {
            ds.manifest_naming_scheme
        } else if self.enable_v2_manifest_paths {
            ManifestNamingScheme::V2
        } else {
            ManifestNamingScheme::V1
        };

        let use_move_stable_row_ids = if let Some(ds) = dest.dataset() {
            ds.manifest.uses_move_stable_row_ids()
        } else {
            self.use_move_stable_row_ids.unwrap_or(false)
        };

        // Validate storage format matches existing dataset
        if let Some(ds) = dest.dataset() {
            if let Some(storage_format) = self.storage_format {
                let passed_storage_format = DataStorageFormat::new(storage_format);
                if ds.manifest.data_storage_format != passed_storage_format
                    && !matches!(transaction.operation, Operation::Overwrite { .. })
                {
                    return Err(Error::InvalidInput {
                        source: format!(
                            "Storage format mismatch. Existing dataset uses {:?}, but new data uses {:?}",
                            ds.manifest.data_storage_format,
                            passed_storage_format
                        ).into(),
                        location: location!(),
                    });
                }
            }
        }

        let manifest_config = ManifestWriteConfig {
            use_move_stable_row_ids,
            storage_format: self.storage_format.map(DataStorageFormat::new),
            ..Default::default()
        };

        let manifest = if let Some(dataset) = dest.dataset() {
            if self.detached {
                if matches!(manifest_naming_scheme, ManifestNamingScheme::V1) {
                    return Err(Error::NotSupported {
                        source: "detached commits cannot be used with v1 manifest paths".into(),
                        location: location!(),
                    });
                }
                commit_detached_transaction(
                    dataset,
                    object_store.as_ref(),
                    commit_handler.as_ref(),
                    &transaction,
                    &manifest_config,
                    &self.commit_config,
                )
                .await?
            } else {
                commit_transaction(
                    dataset,
                    object_store.as_ref(),
                    commit_handler.as_ref(),
                    &transaction,
                    &manifest_config,
                    &self.commit_config,
                    manifest_naming_scheme,
                )
                .await?
            }
        } else if self.detached {
            // I think we may eventually want this, and we can probably handle it, but leaving a TODO for now
            return Err(Error::NotSupported {
                source: "detached commits cannot currently be used to create new datasets".into(),
                location: location!(),
            });
        } else {
            commit_new_dataset(
                object_store.as_ref(),
                commit_handler.as_ref(),
                &base_path,
                &transaction,
                &manifest_config,
                manifest_naming_scheme,
            )
            .await?
        };

        let tags = Tags::new(
            object_store.clone(),
            commit_handler.clone(),
            base_path.clone(),
        );

        match &self.dest {
            WriteDestination::Dataset(dataset) => Ok(Dataset {
                manifest: Arc::new(manifest),
                session: self.session.unwrap_or(dataset.session.clone()),
                ..dataset.as_ref().clone()
            }),
            WriteDestination::Uri(uri) => Ok(Dataset {
                object_store,
                base: base_path,
                uri: uri.to_string(),
                manifest: Arc::new(manifest),
                session: self.session.unwrap_or_default(),
                commit_handler,
                tags,
                manifest_naming_scheme,
            }),
        }
    }
}
