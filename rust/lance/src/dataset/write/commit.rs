// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_table::{
    format::{is_detached_version, DataStorageFormat},
    io::commit::{CommitConfig, CommitHandler, ManifestNamingScheme},
};
use snafu::location;

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

        let session = self
            .session
            .or_else(|| self.dest.dataset().map(|ds| ds.session.clone()))
            .unwrap_or_default();

        let dest = match &self.dest {
            WriteDestination::Dataset(dataset) => WriteDestination::Dataset(dataset.clone()),
            WriteDestination::Uri(uri) => {
                // Check if it already exists.
                let mut builder = DatasetBuilder::from_uri(uri)
                    .with_read_params(ReadParams {
                        store_options: self.store_params.clone(),
                        commit_handler: self.commit_handler.clone(),
                        object_store_registry: self.object_store_registry.clone(),
                        ..Default::default()
                    })
                    .with_session(session.clone());

                // If we are using a detached version, we need to load the dataset.
                // Otherwise, we are writing to the main history, and need to check
                // out the latest version.
                if is_detached_version(transaction.read_version) {
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

        let (manifest, manifest_file) = if let Some(dataset) = dest.dataset() {
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
                &session,
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
                manifest_file,
                session,
                ..dataset.as_ref().clone()
            }),
            WriteDestination::Uri(uri) => Ok(Dataset {
                object_store,
                base: base_path,
                uri: uri.to_string(),
                manifest: Arc::new(manifest),
                manifest_file,
                session,
                commit_handler,
                tags,
                manifest_naming_scheme,
            }),
        }
    }

    /// Commit a set of transactions as a single new version.
    ///
    /// <div class="warning">
    ///   Only works for append transactions right now. Other kinds of transactions
    ///   will be supported in the future.
    /// </div>
    pub async fn execute_batch(self, transactions: Vec<Transaction>) -> Result<BatchCommitResult> {
        if transactions.is_empty() {
            return Err(Error::InvalidInput {
                source: "No transactions to commit".into(),
                location: location!(),
            });
        }
        if transactions
            .iter()
            .any(|t| !matches!(t.operation, Operation::Append { .. }))
        {
            return Err(Error::NotSupported {
                source: "Only append transactions are supported in batch commits".into(),
                location: location!(),
            });
        }

        let read_version = transactions.iter().map(|t| t.read_version).min().unwrap();
        let blob_new_frags = transactions
            .iter()
            .flat_map(|t| &t.blobs_op)
            .flat_map(|b| match b {
                Operation::Append { fragments } => fragments.clone(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let blobs_op = if blob_new_frags.is_empty() {
            None
        } else {
            Some(Operation::Append {
                fragments: blob_new_frags,
            })
        };

        let merged = Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Append {
                fragments: transactions
                    .iter()
                    .flat_map(|t| match &t.operation {
                        Operation::Append { fragments } => fragments.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
            },
            read_version,
            blobs_op,
            tag: None,
        };
        let dataset = self.execute(merged.clone()).await?;
        Ok(BatchCommitResult { dataset, merged })
    }
}

pub struct BatchCommitResult {
    pub dataset: Dataset,
    /// The final transaction that was committed.
    pub merged: Transaction,
    // TODO: Reject conflicts that need to be retried.
    // /// Transactions that were rejected due to conflicts.
    // pub rejected: Vec<Transaction>,
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_table::{
        format::{DataFile, Fragment},
        io::commit::RenameCommitHandler,
    };
    use url::Url;

    use crate::dataset::{InsertBuilder, WriteParams};

    use super::*;

    fn sample_fragment() -> Fragment {
        Fragment {
            id: 0,
            files: vec![DataFile {
                path: "file.lance".to_string(),
                fields: vec![0],
                column_indices: vec![0],
                file_major_version: 2,
                file_minor_version: 0,
            }],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(10),
        }
    }

    fn sample_transaction(read_version: u64) -> Transaction {
        Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Append {
                fragments: vec![sample_fragment()],
            },
            read_version,
            blobs_op: None,
            tag: None,
        }
    }

    #[tokio::test]
    async fn test_reuse_session() {
        // Need to use in-memory for accurate IOPS tracking.
        use crate::utils::test::IoTrackingStore;

        // Create new dataset
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let memory_store = Arc::new(object_store::memory::InMemory::new());
        let (io_stats_wrapper, io_stats) = IoTrackingStore::new_wrapper();
        let store_params = ObjectStoreParams {
            object_store_wrapper: Some(io_stats_wrapper),
            object_store: Some((memory_store.clone(), Url::parse("memory://test").unwrap())),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://test")
            .with_params(&WriteParams {
                store_params: Some(store_params.clone()),
                commit_handler: Some(Arc::new(RenameCommitHandler)),
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        let mut dataset = Arc::new(dataset);

        let reset_iops = || {
            io_stats.lock().unwrap().read_iops = 0;
            io_stats.lock().unwrap().write_iops = 0;
        };
        let get_new_iops = || {
            let read_iops = io_stats.lock().unwrap().read_iops;
            let write_iops = io_stats.lock().unwrap().write_iops;
            reset_iops();
            (read_iops, write_iops)
        };

        let (initial_reads, initial_writes) = get_new_iops();
        assert!(initial_reads > 0);
        assert!(initial_writes > 0);

        // Commit transaction 5 times
        for i in 0..5 {
            let new_ds = CommitBuilder::new(dataset.clone())
                .execute(sample_transaction(1))
                .await
                .unwrap();
            dataset = Arc::new(new_ds);
            assert_eq!(dataset.manifest().version, i + 2);

            // Because we are writing transactions sequentially, and caching them,
            // we shouldn't need to read anything from disk. Except we do need
            // to check for the latest version to see if we need to do conflict
            // resolution.
            let (reads, writes) = get_new_iops();
            assert_eq!(reads, 1, "i = {}", i);
            // Should see 3 IOPs:
            // 1. Write the transaction files
            // 2. Write the manifest
            // 3. Atomically rename the manifest
            assert_eq!(writes, 3, "i = {}", i);
        }

        // Commit transaction with URI and session
        let new_ds = CommitBuilder::new("memory://test")
            .with_store_params(store_params.clone())
            .with_commit_handler(Arc::new(RenameCommitHandler))
            .with_session(dataset.session.clone())
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest().version, 7);
        // Session should still be re-used
        // However, the dataset needs to be loaded, so an additional two IOPs
        // are needed.
        let (reads, writes) = get_new_iops();
        assert_eq!(reads, 3);
        assert_eq!(writes, 3);

        // Commit transaction with URI and no session
        let new_ds = CommitBuilder::new("memory://test")
            .with_store_params(store_params)
            .with_commit_handler(Arc::new(RenameCommitHandler))
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest().version, 8);
        // Now we have to load all previous transactions.
        let (reads, writes) = get_new_iops();
        assert!(reads > 20);
        assert_eq!(writes, 3);
    }

    #[tokio::test]
    async fn test_commit_batch() {
        // Create a dataset
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://test")
            .execute(vec![batch])
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        // Attempting to commit empty gives error
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![])
            .await;
        assert!(matches!(res, Err(Error::InvalidInput { .. })));

        // Attempting to commit update gives error
        let update_transaction = Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Update {
                updated_fragments: vec![],
                new_fragments: vec![],
                removed_fragment_ids: vec![],
            },
            read_version: 1,
            blobs_op: None,
            tag: None,
        };
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![update_transaction])
            .await;
        assert!(matches!(res, Err(Error::NotSupported { .. })));

        // Doing multiple appends includes all.
        let append1 = sample_transaction(1);
        let append2 = sample_transaction(2);
        let mut expected_fragments = vec![];
        if let Operation::Append { fragments } = &append1.operation {
            expected_fragments.extend(fragments.clone());
        }
        if let Operation::Append { fragments } = &append2.operation {
            expected_fragments.extend(fragments.clone());
        }
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![append1.clone(), append2.clone()])
            .await
            .unwrap();
        let transaction = res.merged;
        assert!(
            matches!(transaction.operation, Operation::Append { fragments } if fragments == expected_fragments)
        );
        assert_eq!(transaction.read_version, 1);
        assert!(transaction.blobs_op.is_none());
    }
}
