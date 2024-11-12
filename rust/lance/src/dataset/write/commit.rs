// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_table::{
    format::DataStorageFormat,
    io::commit::{commit_handler_from_url, CommitConfig, CommitHandler, ManifestNamingScheme},
};
use object_store::path::Path;
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

use super::InsertDestination;

#[derive(Debug, Clone)]
pub struct CommitBuilder<'a> {
    dest: InsertDestination<'a>,
    use_move_stable_row_ids: bool,
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
    pub fn new(dest: impl Into<InsertDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            use_move_stable_row_ids: false,
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

    pub fn use_move_stable_row_ids(mut self, use_move_stable_row_ids: bool) -> Self {
        self.use_move_stable_row_ids = use_move_stable_row_ids;
        self
    }

    pub fn with_storage_format(mut self, storage_format: LanceFileVersion) -> Self {
        self.storage_format = Some(storage_format);
        self
    }

    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.commit_handler = Some(commit_handler);
        self
    }

    pub fn with_store_params(mut self, store_params: ObjectStoreParams) -> Self {
        self.store_params = Some(store_params);
        self
    }

    pub fn with_object_store_registry(
        mut self,
        object_store_registry: Arc<ObjectStoreRegistry>,
    ) -> Self {
        self.object_store_registry = object_store_registry;
        self
    }

    pub fn with_object_store(mut self, object_store: Arc<ObjectStore>) -> Self {
        self.object_store = Some(object_store);
        self
    }

    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    ///  If set to true, and this is a new dataset, uses the new v2 manifest
    ///  paths. These allow constant-time lookups for the latest manifest on object storage.
    ///  This parameter has no effect on existing datasets. To migrate an existing
    ///  dataset, use the [`Self::migrate_manifest_paths_v2`] method. WARNING: turning
    ///  this on will make the dataset unreadable for older versions of Lance
    ///  (prior to 0.17.0). Default is False.
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
            InsertDestination::Dataset(dataset) => (
                dataset.object_store.clone(),
                dataset.base.clone(),
                dataset.commit_handler.clone(),
            ),
            InsertDestination::Uri(uri) => {
                let (mut object_store, base_path, commit_handler) = Self::params_from_uri(
                    uri,
                    &self.commit_handler,
                    &self.store_params,
                    self.object_store_registry.clone(),
                )
                .await?;
                if let Some(passed_store) = self.object_store {
                    object_store = passed_store
                }
                (object_store, base_path, commit_handler)
            }
        };

        let dest = match &self.dest {
            InsertDestination::Dataset(dataset) => InsertDestination::Dataset(dataset.clone()),
            InsertDestination::Uri(uri) => {
                // Check if it already exists.
                let builder = DatasetBuilder::from_uri(uri).with_read_params(ReadParams {
                    store_options: self.store_params.clone(),
                    commit_handler: self.commit_handler.clone(),
                    object_store_registry: self.object_store_registry.clone(),
                    ..Default::default()
                });

                match builder.load().await {
                    Ok(dataset) => InsertDestination::Dataset(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        InsertDestination::Uri(uri)
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

        let manifest_config = ManifestWriteConfig {
            use_move_stable_row_ids: self.use_move_stable_row_ids,
            storage_format: self.storage_format.map(DataStorageFormat::new),
            ..Default::default()
        };

        let manifest = if let Some(dataset) = dest.dataset() {
            if self.detached {
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
        } else {
            if self.detached {
                // I think we may eventually want this, and we can probably handle it, but leaving a TODO for now
                return Err(Error::NotSupported {
                    source: "detached commits cannot currently be used to create new datasets"
                        .into(),
                    location: location!(),
                });
            } else {
                commit_new_dataset(
                    &object_store.as_ref(),
                    commit_handler.as_ref(),
                    &base_path,
                    &transaction,
                    &manifest_config,
                    manifest_naming_scheme,
                )
                .await?
            }
        };

        let tags = Tags::new(
            object_store.clone(),
            commit_handler.clone(),
            base_path.clone(),
        );

        match &self.dest {
            InsertDestination::Dataset(dataset) => Ok(Dataset {
                manifest: Arc::new(manifest),
                session: self.session.unwrap_or(dataset.session.clone()),
                ..dataset.as_ref().clone()
            }),
            InsertDestination::Uri(uri) => Ok(Dataset {
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

    async fn params_from_uri(
        uri: &str,
        commit_handler: &Option<Arc<dyn CommitHandler>>,
        store_options: &Option<ObjectStoreParams>,
        object_store_registry: Arc<ObjectStoreRegistry>,
    ) -> Result<(Arc<ObjectStore>, Path, Arc<dyn CommitHandler>)> {
        let (mut object_store, base_path) = match store_options.as_ref() {
            Some(store_options) => {
                ObjectStore::from_uri_and_params(object_store_registry, uri, store_options).await?
            }
            None => ObjectStore::from_uri(uri).await?,
        };

        if let Some(block_size) = store_options.as_ref().and_then(|opts| opts.block_size) {
            object_store.set_block_size(block_size);
        }

        let commit_handler = match &commit_handler {
            None => {
                if store_options.is_some() && store_options.as_ref().unwrap().object_store.is_some()
                {
                    return Err(Error::InvalidInput { source: "when creating a dataset with a custom object store the commit_handler must also be specified".into(), location: location!() });
                }
                commit_handler_from_url(uri, store_options).await?
            }
            Some(commit_handler) => {
                if uri.starts_with("s3+ddb") {
                    return Err(Error::InvalidInput {
                        source:
                            "`s3+ddb://` scheme and custom commit handler are mutually exclusive"
                                .into(),
                        location: location!(),
                    });
                } else {
                    commit_handler.clone()
                }
            }
        };

        Ok((Arc::new(object_store), base_path, commit_handler))
    }
}
